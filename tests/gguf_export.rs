//! Exporting a tiny Qwen2 checkpoint as a Q8_0 GGUF file (ADR 0032).
//!
//! The checkpoint is random, so the model is meaningless; the tests check the
//! export's structure, its counts, and its refusals. Runtime compatibility is
//! checked separately against llama.cpp on Linux (see ADR 0032).

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::id,
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::gguf_qwen2::{Qwen2ExportError, export_qwen2_q8_0};
use serde_json::{Map, Value, json};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

const VOCAB: usize = 32;
const HIDDEN: usize = 64;
const KV_DIM: usize = 32;
const INTERMEDIATE: usize = 128;
const LAYERS: usize = 2;

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let serial = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("modelq-gguf-export-{label}-{}-{serial}", id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn values(count: usize, seed: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as f32) * 0.31 + seed).sin() * 0.4)
        .collect()
}

fn safetensors(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut header = Map::new();
    let mut data = Vec::new();
    for (name, shape, tensor_values) in tensors {
        let begin = data.len();
        for value in tensor_values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}),
        );
    }
    let mut json = serde_json::to_vec(&Value::Object(header)).unwrap();
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&json);
    bytes.extend_from_slice(&data);
    bytes
}

/// The tensors a tiny Qwen2 model with tied embeddings has, plus any extra names.
fn tiny_tensors(extra: &[&str]) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut tensors = vec![
        (
            "model.embed_tokens.weight".to_owned(),
            vec![VOCAB, HIDDEN],
            values(VOCAB * HIDDEN, 1.0),
        ),
        (
            "model.norm.weight".to_owned(),
            vec![HIDDEN],
            values(HIDDEN, 2.0),
        ),
    ];
    for layer in 0..LAYERS {
        let prefix = format!("model.layers.{layer}.");
        let seed = 3.0 + layer as f32;
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![HIDDEN]),
            ("self_attn.q_proj.weight", vec![HIDDEN, HIDDEN]),
            ("self_attn.q_proj.bias", vec![HIDDEN]),
            ("self_attn.k_proj.weight", vec![KV_DIM, HIDDEN]),
            ("self_attn.k_proj.bias", vec![KV_DIM]),
            ("self_attn.v_proj.weight", vec![KV_DIM, HIDDEN]),
            ("self_attn.v_proj.bias", vec![KV_DIM]),
            ("self_attn.o_proj.weight", vec![HIDDEN, HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
            ("mlp.gate_proj.weight", vec![INTERMEDIATE, HIDDEN]),
            ("mlp.up_proj.weight", vec![INTERMEDIATE, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, INTERMEDIATE]),
        ] {
            let count: usize = shape.iter().product();
            tensors.push((format!("{prefix}{suffix}"), shape, values(count, seed)));
        }
    }
    for name in extra {
        tensors.push(((*name).to_owned(), vec![HIDDEN], values(HIDDEN, 9.0)));
    }
    tensors
}

fn write_model(directory: &Path, model_type: &str, extra: &[&str]) {
    let config = json!({
        "model_type": model_type,
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
    });
    fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut vocab = Map::new();
    for id in 0..VOCAB - 1 {
        vocab.insert(format!("t{id}"), json!(id));
    }
    vocab.insert("<|endoftext|>".to_owned(), json!(VOCAB - 1));
    let tokenizer = json!({
        "model": {"type": "BPE", "vocab": vocab, "merges": ["t1 t2"]},
        "added_tokens": [{"id": VOCAB - 1, "content": "<|endoftext|>", "special": true}],
    });
    fs::write(
        directory.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    fs::write(
        directory.join("tokenizer_config.json"),
        serde_json::to_vec(&json!({"eos_token": "<|endoftext|>"})).unwrap(),
    )
    .unwrap();
    fs::write(
        directory.join("model.safetensors"),
        safetensors(&tiny_tensors(extra)),
    )
    .unwrap();
}

#[test]
fn a_tiny_qwen2_checkpoint_exports_to_a_q8_0_gguf_file() {
    let dir = TestDir::new("export");
    let model = dir.join("model");
    fs::create_dir_all(&model).unwrap();
    write_model(&model, "qwen2", &[]);
    let output = dir.join("tiny.gguf");

    let report = export_qwen2_q8_0(&model, &output).expect("the tiny checkpoint exports");
    // Q8_0: the embedding and seven matrices per layer. F32: the final norm, and
    // two norms and three biases per layer.
    assert_eq!(report.q8_0_tensors, 1 + 7 * LAYERS);
    assert_eq!(report.f32_tensors, 1 + 5 * LAYERS);
    assert_eq!(report.vocab_size, VOCAB);
    let bytes = fs::read(&output).unwrap();
    assert_eq!(&bytes[..4], b"GGUF");
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        3,
        "GGUF version 3"
    );
    assert_eq!(report.output_bytes, bytes.len() as u64);
    assert_eq!(bytes.len() % 32, 0, "the file ends on the alignment");
    assert!(
        bytes
            .windows(b"qwen2".len())
            .any(|window| window == b"qwen2"),
        "the architecture name is written"
    );
}

#[test]
fn a_tensor_without_a_qwen2_name_is_refused_not_dropped() {
    let dir = TestDir::new("unmapped");
    let model = dir.join("model");
    fs::create_dir_all(&model).unwrap();
    write_model(
        &model,
        "qwen2",
        &["model.layers.0.self_attn.rotary_emb.inv_freq"],
    );
    let error = export_qwen2_q8_0(&model, &dir.join("out.gguf")).unwrap_err();
    assert!(
        matches!(error, Qwen2ExportError::UnmappedTensor { ref name } if name.contains("rotary")),
        "{error}"
    );
    assert!(
        !dir.join("out.gguf").exists(),
        "nothing is written when the export fails"
    );
}

#[test]
fn other_architectures_are_refused() {
    let dir = TestDir::new("architecture");
    let model = dir.join("model");
    fs::create_dir_all(&model).unwrap();
    write_model(&model, "llama", &[]);
    let error = export_qwen2_q8_0(&model, &dir.join("out.gguf")).unwrap_err();
    assert!(
        matches!(error, Qwen2ExportError::UnsupportedArchitecture { .. }),
        "{error}"
    );
}

#[test]
fn an_existing_output_is_never_replaced() {
    let dir = TestDir::new("exists");
    let model = dir.join("model");
    fs::create_dir_all(&model).unwrap();
    write_model(&model, "qwen2", &[]);
    let output = dir.join("out.gguf");
    fs::write(&output, b"previous artifact").unwrap();
    let error = export_qwen2_q8_0(&model, &output).unwrap_err();
    assert!(matches!(error, Qwen2ExportError::Write(_)), "{error}");
    assert_eq!(fs::read(&output).unwrap(), b"previous artifact");
}

#[test]
fn the_export_is_deterministic() {
    let dir = TestDir::new("determinism");
    let model = dir.join("model");
    fs::create_dir_all(&model).unwrap();
    write_model(&model, "qwen2", &[]);
    let first = dir.join("first.gguf");
    let second = dir.join("second.gguf");
    export_qwen2_q8_0(&model, &first).unwrap();
    export_qwen2_q8_0(&model, &second).unwrap();
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
}
