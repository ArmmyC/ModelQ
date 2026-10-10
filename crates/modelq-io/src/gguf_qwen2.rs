//! Exports a Qwen2 checkpoint as a runtime-compatible GGUF file with Q8_0 or
//! Q4_0 weights (ADR 0032 and ADR 0033, milestone M5).
//!
//! The mapping follows llama.cpp's `qwen2` architecture, as its loader reads it
//! (tensor names, hyperparameter keys, and the `gpt2` byte-level BPE tokenizer
//! with the `qwen2` pre-tokenizer). Every source tensor is either mapped or the
//! export fails: nothing is dropped silently, and the architecture is refused
//! unless `config.json` says `qwen2`.
//!
//! Two-dimensional weights become the chosen block format (ADR 0008's Q8_0 or
//! ADR 0033's Q4_0), and one-dimensional norms and biases stay F32, as
//! llama.cpp's own quantized conversions do. Those are the only two
//! representations written.

use std::{
    fmt,
    path::{Path, PathBuf},
};

use modelq_quant::{gguf_q4_0, gguf_q8_0};
use serde_json::Value;

use crate::{
    gguf_writer::{
        GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GgufFile, GgufWriteError, MetadataValue,
        TensorRecord,
    },
    safetensors::{SafetensorsError, TensorSource},
    sharded::{SafetensorsInput, ShardedError},
};

/// The architecture name llama.cpp uses for Qwen2 models.
pub const ARCHITECTURE: &str = "qwen2";
/// `general.file_type` for an all-Q8_0 file with F32 norms (llama.cpp's `MOSTLY_Q8_0`).
pub const FILE_TYPE_MOSTLY_Q8_0: u32 = 7;
/// `general.file_type` for an all-Q4_0 file with F32 norms (llama.cpp's `MOSTLY_Q4_0`).
pub const FILE_TYPE_MOSTLY_Q4_0: u32 = 2;
/// `general.quantization_version` written by the GGML library at the pinned release.
pub const QUANTIZATION_VERSION: u32 = 2;

/// The block format of the two-dimensional weights. One-dimensional tensors are always F32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuantization {
    /// 8-bit blocks with one binary16 scale per 32 values (ADR 0008, ADR 0032).
    Q8_0,
    /// 4-bit blocks with one binary16 scale per 32 values (ADR 0033).
    Q4_0,
}

impl WeightQuantization {
    /// `general.file_type` for a file of this format with F32 norms.
    pub const fn file_type(self) -> u32 {
        match self {
            Self::Q8_0 => FILE_TYPE_MOSTLY_Q8_0,
            Self::Q4_0 => FILE_TYPE_MOSTLY_Q4_0,
        }
    }

    const fn ggml_type(self) -> u32 {
        match self {
            Self::Q8_0 => GGML_TYPE_Q8_0,
            Self::Q4_0 => GGML_TYPE_Q4_0,
        }
    }

    /// Quantizes one tensor's values into the serialized block bytes.
    fn quantize(self, values: &[f32], shape: &[usize]) -> Result<Vec<u8>, String> {
        match self {
            Self::Q8_0 => gguf_q8_0::quantize_shaped(values, shape)
                .map(gguf_q8_0::QuantizedQ8_0::into_bytes)
                .map_err(|error| error.to_string()),
            Self::Q4_0 => gguf_q4_0::quantize_shaped(values, shape)
                .map(gguf_q4_0::QuantizedQ4_0::into_bytes)
                .map_err(|error| error.to_string()),
        }
    }
}

const TOKEN_NORMAL: i32 = 1;
const TOKEN_CONTROL: i32 = 3;
const TOKEN_USER_DEFINED: i32 = 4;
const TOKEN_UNUSED: i32 = 5;

/// Errors from exporting a Qwen2 checkpoint.
#[derive(Debug)]
pub enum Qwen2ExportError {
    /// A file could not be read.
    Io { path: PathBuf, detail: String },
    /// A file is not valid JSON.
    Json { file: &'static str, detail: String },
    /// `config.json` names an architecture other than `qwen2`.
    UnsupportedArchitecture { model_type: String },
    /// A required field is missing or has the wrong type.
    MissingField {
        file: &'static str,
        field: &'static str,
    },
    /// A checkpoint tensor has no GGUF name in this mapping.
    UnmappedTensor { name: String },
    /// A tensor the architecture needs is absent from the checkpoint.
    MissingTensor { name: String },
    /// A tensor's shape does not match the configuration.
    ShapeMismatch { name: String, detail: String },
    /// The tokenizer cannot be converted.
    Tokenizer { detail: String },
    /// A tensor could not be quantized.
    Quantize { name: String, detail: String },
    /// The checkpoint could not be opened or read.
    Source { detail: String },
    /// The GGUF file could not be written.
    Write(GgufWriteError),
}

impl fmt::Display for Qwen2ExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, detail } => {
                write!(formatter, "could not read {}: {detail}", path.display())
            }
            Self::Json { file, detail } => write!(formatter, "{file} is not valid JSON: {detail}"),
            Self::UnsupportedArchitecture { model_type } => write!(
                formatter,
                "model_type {model_type:?} is not supported; this exporter writes qwen2 only"
            ),
            Self::MissingField { file, field } => {
                write!(formatter, "{file} has no usable {field:?}")
            }
            Self::UnmappedTensor { name } => {
                write!(
                    formatter,
                    "checkpoint tensor {name:?} has no qwen2 GGUF name; refusing to drop it"
                )
            }
            Self::MissingTensor { name } => {
                write!(formatter, "the checkpoint has no tensor {name:?}")
            }
            Self::ShapeMismatch { name, detail } => write!(formatter, "{name:?}: {detail}"),
            Self::Tokenizer { detail } => write!(formatter, "tokenizer: {detail}"),
            Self::Quantize { name, detail } => {
                write!(formatter, "{name:?} could not be quantized: {detail}")
            }
            Self::Source { detail } => write!(formatter, "checkpoint: {detail}"),
            Self::Write(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for Qwen2ExportError {}

impl From<GgufWriteError> for Qwen2ExportError {
    fn from(error: GgufWriteError) -> Self {
        Self::Write(error)
    }
}

impl From<ShardedError> for Qwen2ExportError {
    fn from(error: ShardedError) -> Self {
        Self::Source {
            detail: error.to_string(),
        }
    }
}

impl From<SafetensorsError> for Qwen2ExportError {
    fn from(error: SafetensorsError) -> Self {
        Self::Source {
            detail: error.to_string(),
        }
    }
}

/// Summary of one export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportReport {
    /// Two-dimensional tensors written in the chosen block format.
    pub quantized_tensors: usize,
    /// Tensors written as F32.
    pub f32_tensors: usize,
    /// Metadata entries written.
    pub metadata_entries: usize,
    /// Vocabulary size taken from `config.json`.
    pub vocab_size: usize,
    /// Bytes in the output file.
    pub output_bytes: u64,
}

/// The hyperparameters llama.cpp's `qwen2` loader reads, from `config.json`.
#[derive(Debug, Clone, PartialEq)]
struct Hyperparameters {
    context_length: u32,
    embedding_length: u32,
    feed_forward_length: u32,
    block_count: u32,
    head_count: u32,
    head_count_kv: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    vocab_size: usize,
    tie_word_embeddings: bool,
}

fn read_json(path: &Path, file: &'static str) -> Result<Value, Qwen2ExportError> {
    let text = std::fs::read_to_string(path).map_err(|error| Qwen2ExportError::Io {
        path: path.to_owned(),
        detail: error.to_string(),
    })?;
    serde_json::from_str(&text).map_err(|error| Qwen2ExportError::Json {
        file,
        detail: error.to_string(),
    })
}

fn required_u32(config: &Value, field: &'static str) -> Result<u32, Qwen2ExportError> {
    config
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|&value| value > 0)
        .ok_or(Qwen2ExportError::MissingField {
            file: "config.json",
            field,
        })
}

fn required_f32(config: &Value, field: &'static str) -> Result<f32, Qwen2ExportError> {
    config
        .get(field)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or(Qwen2ExportError::MissingField {
            file: "config.json",
            field,
        })
}

fn hyperparameters(config: &Value) -> Result<Hyperparameters, Qwen2ExportError> {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if model_type != ARCHITECTURE {
        return Err(Qwen2ExportError::UnsupportedArchitecture {
            model_type: model_type.to_owned(),
        });
    }
    let embedding_length = required_u32(config, "hidden_size")?;
    let head_count = required_u32(config, "num_attention_heads")?;
    if embedding_length % head_count != 0 {
        return Err(Qwen2ExportError::ShapeMismatch {
            name: "config.json".to_owned(),
            detail: format!(
                "hidden_size {embedding_length} is not divisible by {head_count} heads"
            ),
        });
    }
    Ok(Hyperparameters {
        context_length: required_u32(config, "max_position_embeddings")?,
        embedding_length,
        feed_forward_length: required_u32(config, "intermediate_size")?,
        block_count: required_u32(config, "num_hidden_layers")?,
        head_count,
        head_count_kv: required_u32(config, "num_key_value_heads")?,
        rms_norm_eps: required_f32(config, "rms_norm_eps")?,
        rope_theta: required_f32(config, "rope_theta")?,
        vocab_size: usize::try_from(required_u32(config, "vocab_size")?).map_err(|_| {
            Qwen2ExportError::MissingField {
                file: "config.json",
                field: "vocab_size",
            }
        })?,
        tie_word_embeddings: config
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

/// The GGUF tokenizer metadata: the token list, token types, merges and the end token.
#[derive(Debug, Clone, PartialEq)]
struct Tokenizer {
    tokens: Vec<String>,
    token_types: Vec<i32>,
    merges: Vec<String>,
    eos_token_id: u32,
}

fn tokenizer(directory: &Path, vocab_size: usize) -> Result<Tokenizer, Qwen2ExportError> {
    let tokenizer_json = read_json(&directory.join("tokenizer.json"), "tokenizer.json")?;
    let model = tokenizer_json
        .get("model")
        .ok_or_else(|| Qwen2ExportError::Tokenizer {
            detail: "tokenizer.json has no model".to_owned(),
        })?;
    if model.get("type").and_then(Value::as_str) != Some("BPE") {
        return Err(Qwen2ExportError::Tokenizer {
            detail: "only byte-level BPE tokenizers are supported".to_owned(),
        });
    }
    let vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| Qwen2ExportError::Tokenizer {
            detail: "tokenizer.json has no vocab".to_owned(),
        })?;

    let mut tokens: Vec<Option<String>> = vec![None; vocab_size];
    let mut types = vec![TOKEN_UNUSED; vocab_size];
    for (token, id) in vocab {
        let index = id
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&index| index < vocab_size)
            .ok_or_else(|| Qwen2ExportError::Tokenizer {
                detail: format!(
                    "token {token:?} has an id outside the {vocab_size}-entry vocabulary"
                ),
            })?;
        if tokens[index].is_some() {
            return Err(Qwen2ExportError::Tokenizer {
                detail: format!("id {index} is used twice"),
            });
        }
        tokens[index] = Some(token.clone());
        types[index] = TOKEN_NORMAL;
    }

    for added in tokenizer_json
        .get("added_tokens")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let index = added
            .get("id")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&index| index < vocab_size)
            .ok_or_else(|| Qwen2ExportError::Tokenizer {
                detail: "an added token has an id outside the vocabulary".to_owned(),
            })?;
        let content = added
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| Qwen2ExportError::Tokenizer {
                detail: "an added token has no content".to_owned(),
            })?;
        if let Some(existing) = &tokens[index] {
            if existing != content {
                return Err(Qwen2ExportError::Tokenizer {
                    detail: format!("id {index} is both {existing:?} and {content:?}"),
                });
            }
        }
        tokens[index] = Some(content.to_owned());
        types[index] = if added
            .get("special")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            TOKEN_CONTROL
        } else {
            TOKEN_USER_DEFINED
        };
    }

    // llama.cpp needs one entry per embedding row. Ids the tokenizer does not
    // define are placeholders marked unused, as llama.cpp's own converter does.
    let tokens: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(index, token)| token.unwrap_or_else(|| format!("[PAD{index}]")))
        .collect();

    let merges = model
        .get("merges")
        .and_then(Value::as_array)
        .ok_or_else(|| Qwen2ExportError::Tokenizer {
            detail: "tokenizer.json has no merges".to_owned(),
        })?
        .iter()
        .map(|merge| match merge {
            Value::String(text) => Ok(text.clone()),
            Value::Array(pair) => match pair.as_slice() {
                [Value::String(left), Value::String(right)] => Ok(format!("{left} {right}")),
                _ => Err(Qwen2ExportError::Tokenizer {
                    detail: "a merge is not a pair of strings".to_owned(),
                }),
            },
            _ => Err(Qwen2ExportError::Tokenizer {
                detail: "a merge is neither a string nor a pair".to_owned(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let config = read_json(
        &directory.join("tokenizer_config.json"),
        "tokenizer_config.json",
    )?;
    let eos = match config.get("eos_token") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Object(object)) => object
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| Qwen2ExportError::Tokenizer {
                detail: "eos_token has no content".to_owned(),
            })?,
        _ => {
            return Err(Qwen2ExportError::Tokenizer {
                detail: "tokenizer_config.json has no eos_token".to_owned(),
            });
        }
    };
    let eos_token_id = tokens
        .iter()
        .position(|token| *token == eos)
        .and_then(|index| u32::try_from(index).ok())
        .ok_or_else(|| Qwen2ExportError::Tokenizer {
            detail: format!("the end token {eos:?} is not in the vocabulary"),
        })?;

    Ok(Tokenizer {
        tokens,
        token_types: types,
        merges,
        eos_token_id,
    })
}

/// The GGUF name of a checkpoint tensor, or `None` if the qwen2 mapping has no name for it.
fn gguf_name(name: &str, layers: u32) -> Option<String> {
    match name {
        "model.embed_tokens.weight" => return Some("token_embd.weight".to_owned()),
        "model.norm.weight" => return Some("output_norm.weight".to_owned()),
        "lm_head.weight" => return Some("output.weight".to_owned()),
        _ => {}
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (index, suffix) = rest.split_once('.')?;
    if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let index: u32 = index.parse().ok()?;
    if index >= layers {
        return None;
    }
    let mapped = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.v_proj.bias" => "attn_v.bias",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{index}.{mapped}"))
}

/// The GGUF tensor names llama.cpp's qwen2 loader requires for this configuration.
fn required_names(hyper: &Hyperparameters) -> Vec<String> {
    let mut names = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
    ];
    if !hyper.tie_word_embeddings {
        names.push("output.weight".to_owned());
    }
    for layer in 0..hyper.block_count {
        for suffix in [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_q.bias",
            "attn_k.weight",
            "attn_k.bias",
            "attn_v.weight",
            "attn_v.bias",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            names.push(format!("blk.{layer}.{suffix}"));
        }
    }
    names
}

/// Exports the checkpoint in `model_dir` as a GGUF file at `output`, with its
/// two-dimensional weights in `quantization`.
///
/// `model_dir` holds `config.json`, `tokenizer.json`, `tokenizer_config.json`,
/// and the SafeTensors weights (one file, or an index with its shards). The
/// output is created new; an existing file is refused.
pub fn export_qwen2(
    model_dir: &Path,
    output: &Path,
    quantization: WeightQuantization,
) -> Result<ExportReport, Qwen2ExportError> {
    let config = read_json(&model_dir.join("config.json"), "config.json")?;
    let hyper = hyperparameters(&config)?;
    let vocabulary = tokenizer(model_dir, hyper.vocab_size)?;
    let source = SafetensorsInput::open(model_dir)?;

    let mut mapped: Vec<(String, String, Vec<usize>)> = Vec::new();
    for summary in source.tensor_summaries() {
        let gguf = gguf_name(&summary.name, hyper.block_count).ok_or_else(|| {
            Qwen2ExportError::UnmappedTensor {
                name: summary.name.clone(),
            }
        })?;
        mapped.push((gguf, summary.name, summary.shape));
    }
    mapped.sort();

    for required in required_names(&hyper) {
        if !mapped.iter().any(|(gguf, _, _)| *gguf == required) {
            return Err(Qwen2ExportError::MissingTensor { name: required });
        }
    }

    let mut file = GgufFile::new();
    let mut metadata = |key: &str, value: MetadataValue| file.add_metadata(key, value);
    metadata(
        "general.architecture",
        MetadataValue::String(ARCHITECTURE.to_owned()),
    )?;
    metadata(
        "general.name",
        MetadataValue::String(model_dir.file_name().map_or_else(
            || "qwen2".to_owned(),
            |name| name.to_string_lossy().into_owned(),
        )),
    )?;
    metadata(
        "general.file_type",
        MetadataValue::U32(quantization.file_type()),
    )?;
    metadata(
        "general.quantization_version",
        MetadataValue::U32(QUANTIZATION_VERSION),
    )?;
    metadata("general.alignment", MetadataValue::U32(32))?;
    metadata(
        "qwen2.context_length",
        MetadataValue::U32(hyper.context_length),
    )?;
    metadata(
        "qwen2.embedding_length",
        MetadataValue::U32(hyper.embedding_length),
    )?;
    metadata(
        "qwen2.feed_forward_length",
        MetadataValue::U32(hyper.feed_forward_length),
    )?;
    metadata("qwen2.block_count", MetadataValue::U32(hyper.block_count))?;
    metadata(
        "qwen2.attention.head_count",
        MetadataValue::U32(hyper.head_count),
    )?;
    metadata(
        "qwen2.attention.head_count_kv",
        MetadataValue::U32(hyper.head_count_kv),
    )?;
    metadata(
        "qwen2.attention.layer_norm_rms_epsilon",
        MetadataValue::F32(hyper.rms_norm_eps),
    )?;
    metadata("qwen2.rope.freq_base", MetadataValue::F32(hyper.rope_theta))?;
    metadata(
        "tokenizer.ggml.model",
        MetadataValue::String("gpt2".to_owned()),
    )?;
    metadata(
        "tokenizer.ggml.pre",
        MetadataValue::String("qwen2".to_owned()),
    )?;
    metadata(
        "tokenizer.ggml.tokens",
        MetadataValue::StringArray(vocabulary.tokens.clone()),
    )?;
    metadata(
        "tokenizer.ggml.token_type",
        MetadataValue::I32Array(vocabulary.token_types.clone()),
    )?;
    metadata(
        "tokenizer.ggml.merges",
        MetadataValue::StringArray(vocabulary.merges.clone()),
    )?;
    metadata(
        "tokenizer.ggml.eos_token_id",
        MetadataValue::U32(vocabulary.eos_token_id),
    )?;
    let metadata_entries = file.metadata().len();

    let mut quantized_tensors = 0;
    let mut f32_tensors = 0;
    for (gguf, source_name, shape) in &mapped {
        let values: Vec<f32> = source.with_tensor(source_name, |view| view.values().collect())?;
        let dimensions: Vec<u64> = shape
            .iter()
            .rev()
            .map(|&dimension| u64::try_from(dimension).unwrap_or(u64::MAX))
            .collect();
        let (ggml_type, data) = if shape.len() == 2 {
            let quantized = quantization.quantize(&values, shape).map_err(|detail| {
                Qwen2ExportError::Quantize {
                    name: source_name.clone(),
                    detail,
                }
            })?;
            quantized_tensors += 1;
            (quantization.ggml_type(), quantized)
        } else if shape.len() == 1 {
            f32_tensors += 1;
            (
                GGML_TYPE_F32,
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            )
        } else {
            return Err(Qwen2ExportError::ShapeMismatch {
                name: source_name.clone(),
                detail: format!("rank {} is not a weight or a vector", shape.len()),
            });
        };
        file.add_tensor(TensorRecord {
            name: gguf.clone(),
            dimensions,
            ggml_type,
            data,
        })?;
    }

    file.write_new(output)?;
    let output_bytes = std::fs::metadata(output)
        .map(|metadata| metadata.len())
        .map_err(|error| Qwen2ExportError::Io {
            path: output.to_owned(),
            detail: error.to_string(),
        })?;
    Ok(ExportReport {
        quantized_tensors,
        f32_tensors,
        metadata_entries,
        vocab_size: hyper.vocab_size,
        output_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen2_names_map_to_llama_cpp_names() {
        assert_eq!(
            gguf_name("model.layers.3.self_attn.q_proj.bias", 24).as_deref(),
            Some("blk.3.attn_q.bias")
        );
        assert_eq!(
            gguf_name("model.layers.23.mlp.down_proj.weight", 24).as_deref(),
            Some("blk.23.ffn_down.weight")
        );
        assert_eq!(
            gguf_name("model.embed_tokens.weight", 24).as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            gguf_name("model.norm.weight", 24).as_deref(),
            Some("output_norm.weight")
        );
    }

    #[test]
    fn unknown_or_out_of_range_names_are_refused() {
        assert!(gguf_name("model.layers.24.mlp.up_proj.weight", 24).is_none());
        assert!(gguf_name("model.layers.0.self_attn.rotary_emb.inv_freq", 24).is_none());
        assert!(gguf_name("model.layers.+1.mlp.up_proj.weight", 24).is_none());
        assert!(gguf_name("visual.blocks.0.attn.qkv.weight", 24).is_none());
    }

    #[test]
    fn required_names_cover_every_layer_once() {
        let hyper = Hyperparameters {
            context_length: 32,
            embedding_length: 64,
            feed_forward_length: 128,
            block_count: 2,
            head_count: 4,
            head_count_kv: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            vocab_size: 8,
            tie_word_embeddings: true,
        };
        let names = required_names(&hyper);
        assert_eq!(names.len(), 2 + 2 * 12);
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), names.len());
        assert!(
            !names.iter().any(|name| name == "output.weight"),
            "tied models have no output tensor"
        );
    }

    #[test]
    fn only_qwen2_is_accepted() {
        let config: Value = serde_json::json!({"model_type": "llama"});
        assert!(matches!(
            hyperparameters(&config),
            Err(Qwen2ExportError::UnsupportedArchitecture { .. })
        ));
    }
}
