//! Transformer Engine multi-matrix NVFP4 container (schema version 2).
//!
//! One SafeTensors file (or shard) holds many rank-two matrices in the
//! rowwise 1x16 profile of [`crate::transformer_engine`] plus any preserved
//! tensors, with a manifest describing exactly the tensors in that file.  The
//! writer streams: packed E2M1 bytes come straight from the bounded quantizer
//! (the same bytes the native NVFP4 container stores), then the zero-padded
//! scale matrix is written row by row from the held block scales, then the
//! amax.  This module makes no hardware or runtime compatibility claim; see
//! the design and ADR 0021.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File},
    io::Write,
    ops::Range,
    path::Path,
};

use modelq_quant::{
    float::{fp4_e2m1, fp8_e4m3},
    nvfp4,
};
use serde_json::Value;

use crate::{
    nvfp4::{
        Nvfp4Execution, Nvfp4WriterError, TemporaryOutput, create_temporary_file, io_error,
        json_object, json_shape, json_string_map, paths_refer_to_same_file,
        stream_quantized_payload,
    },
    safetensors::{MappedSafetensors, SafetensorsError, TensorSource, TensorSummary},
    transformer_engine::{
        TRANSFORMER_ENGINE_NVFP4_BLOCK_SIZE as BLOCK_SIZE,
        TRANSFORMER_ENGINE_NVFP4_GLOBAL_SCALE_DENOMINATOR as GLOBAL_SCALE_DENOMINATOR,
        TRANSFORMER_ENGINE_NVFP4_PROFILE as PROFILE,
        TRANSFORMER_ENGINE_NVFP4_SCALE_COLUMN_ALIGNMENT as COLUMN_ALIGNMENT,
        TRANSFORMER_ENGINE_NVFP4_SCALE_ROW_ALIGNMENT as ROW_ALIGNMENT,
    },
};

/// `modelq.format` value of a version-2 container.
pub const TE_CONTAINER_FORMAT: &str = "transformer-engine-nvfp4-safetensors-v2";
/// Runtime name recorded in the manifest.
pub const TE_RUNTIME_NAME: &str = "transformer_engine";
/// The only Transformer Engine release this container targets.
pub const TE_RUNTIME_VERSION: &str = "2.19.0";

const RESERVED_METADATA_NAME: &str = "__metadata__";
const U8_DTYPE: &str = "U8";
const F32_DTYPE: &str = "F32";
const HEADER_LENGTH_BYTES: usize = 8;
const HEADER_ALIGNMENT: usize = 8;
const MAX_HEADER_SIZE: usize = 100_000_000;
const SCHEMA_VERSION: u64 = 2;

/// The role of one tensor in a Transformer Engine container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeOutputRole {
    /// A source tensor copied unchanged.
    Preserved,
    /// Packed rowwise E2M1 values, `[M, K/2]`.
    RowwiseData,
    /// Zero-padded E4M3 block scales, `[round_up(M,128), round_up(K/16,4)]`.
    RowwiseScaleInv,
    /// One F32 tensor amax, `[1]`.
    AmaxRowwise,
}

/// One output tensor's metadata and contiguous data-region range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeOutputTensorPlan {
    /// Output tensor name.
    pub name: String,
    /// Source tensor this output represents.
    pub source_name: String,
    /// Output SafeTensors dtype name.
    pub dtype: String,
    /// Output tensor shape.
    pub shape: Vec<usize>,
    /// Payload bytes reserved.
    pub byte_len: u64,
    /// Half-open range relative to the SafeTensors data section.
    pub data_offsets: Range<u64>,
    /// Why this tensor exists.
    pub role: TeOutputRole,
}

/// A complete data-region plan for one container file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeOutputPlan {
    /// Output tensors: each source in ascending name order, a matrix's three
    /// fields in `rowwise_data`, `rowwise_scale_inv`, `amax_rowwise` order.
    pub tensors: Vec<TeOutputTensorPlan>,
    /// Total bytes of the data section.
    pub total_data_bytes: u64,
    quantized_names: Vec<String>,
}

impl TeOutputPlan {
    /// Finds an output tensor by exact name.
    pub fn tensor(&self, name: &str) -> Option<&TeOutputTensorPlan> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Source names exported as Transformer Engine matrices, ascending.
    pub fn quantized_source_names(&self) -> &[String] {
        &self.quantized_names
    }
}

/// Reasons a source set and selection cannot form a container layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TeLayoutError {
    /// A source name appears more than once.
    DuplicateSourceName { name: String },
    /// A source uses the SafeTensors metadata key as its name.
    ReservedSourceName { name: String },
    /// A selected name appears more than once.
    DuplicateSelectedName { name: String },
    /// A selected name is not a source tensor.
    SelectedNameNotFound { name: String },
    /// A selected source has a dtype the quantizer cannot read.
    UnsupportedQuantizedDtype { name: String, dtype: String },
    /// A selected source is not an eligible matrix.
    InvalidQuantizedShape {
        name: String,
        shape: Vec<usize>,
        reason: &'static str,
    },
    /// A generated field name collides with another tensor.
    OutputNameCollision { name: String },
    /// Offsets or sizes overflowed.
    SizeOverflow { name: String },
}

impl fmt::Display for TeLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSourceName { name } => {
                write!(
                    formatter,
                    "source tensor name {name:?} appears more than once"
                )
            }
            Self::ReservedSourceName { name } => {
                write!(
                    formatter,
                    "source tensor name {name:?} is reserved by SafeTensors"
                )
            }
            Self::DuplicateSelectedName { name } => {
                write!(formatter, "selected tensor {name:?} appears more than once")
            }
            Self::SelectedNameNotFound { name } => {
                write!(formatter, "selected tensor {name:?} is not a source tensor")
            }
            Self::UnsupportedQuantizedDtype { name, dtype } => write!(
                formatter,
                "tensor {name:?} uses unsupported dtype {dtype:?} for NVFP4 export"
            ),
            Self::InvalidQuantizedShape {
                name,
                shape,
                reason,
            } => write!(
                formatter,
                "tensor {name:?} with shape {shape:?} cannot be a Transformer Engine matrix: {reason}"
            ),
            Self::OutputNameCollision { name } => {
                write!(
                    formatter,
                    "generated tensor name {name:?} collides with another tensor"
                )
            }
            Self::SizeOverflow { name } => {
                write!(formatter, "size or offset of {name:?} overflows")
            }
        }
    }
}

impl std::error::Error for TeLayoutError {}

fn round_up(value: usize, alignment: usize) -> Option<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

/// Returns the physical `rowwise_scale_inv` shape for a `[rows, columns]`
/// matrix: `[round_up(rows, 128), round_up(columns / 16, 4)]`.
pub fn te_scale_shape(rows: usize, columns: usize) -> Option<[usize; 2]> {
    Some([
        round_up(rows, ROW_ALIGNMENT)?,
        round_up(columns / BLOCK_SIZE, COLUMN_ALIGNMENT)?,
    ])
}

/// Recovers the decode scale from a stored amax.  An all-zero tensor stores
/// an amax of zero and decodes through a unit scale.
pub fn te_global_scale(amax: f32) -> f32 {
    if amax == 0.0 {
        1.0
    } else {
        amax / GLOBAL_SCALE_DENOMINATOR
    }
}

fn validate_matrix(source: &TensorSummary) -> Result<[usize; 2], TeLayoutError> {
    let invalid = |reason: &'static str| TeLayoutError::InvalidQuantizedShape {
        name: source.name.clone(),
        shape: source.shape.clone(),
        reason,
    };
    if !matches!(source.dtype.as_str(), "F32" | "F16" | "BF16") {
        return Err(TeLayoutError::UnsupportedQuantizedDtype {
            name: source.name.clone(),
            dtype: source.dtype.clone(),
        });
    }
    let [rows, columns] = source.shape[..] else {
        return Err(invalid("only rank-two matrices are exported"));
    };
    if rows == 0 || columns == 0 {
        return Err(invalid("dimensions must be positive"));
    }
    if columns % BLOCK_SIZE != 0 {
        return Err(invalid("the final dimension must be divisible by 16"));
    }
    if rows % BLOCK_SIZE != 0 {
        return Err(invalid("the leading dimension must be divisible by 16"));
    }
    Ok([rows, columns])
}

/// Plans a container file for `sources`, exporting `quantized_names` as
/// Transformer Engine matrices and preserving every other source unchanged.
pub fn plan_te_output(
    sources: &[TensorSummary],
    quantized_names: &[String],
) -> Result<TeOutputPlan, TeLayoutError> {
    let mut sorted: Vec<&TensorSummary> = sources.iter().collect();
    sorted.sort_by(|left, right| left.name.cmp(&right.name));
    for pair in sorted.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(TeLayoutError::DuplicateSourceName {
                name: pair[0].name.clone(),
            });
        }
    }
    if let Some(source) = sorted
        .iter()
        .find(|source| source.name == RESERVED_METADATA_NAME)
    {
        return Err(TeLayoutError::ReservedSourceName {
            name: source.name.clone(),
        });
    }
    let source_names: BTreeSet<&str> = sorted.iter().map(|source| source.name.as_str()).collect();
    let mut selected = BTreeSet::new();
    for name in quantized_names {
        if !selected.insert(name.as_str()) {
            return Err(TeLayoutError::DuplicateSelectedName { name: name.clone() });
        }
        if !source_names.contains(name.as_str()) {
            return Err(TeLayoutError::SelectedNameNotFound { name: name.clone() });
        }
    }

    let mut tensors = Vec::new();
    let mut output_names = BTreeSet::new();
    let mut cursor = 0_u64;
    let push = |tensors: &mut Vec<TeOutputTensorPlan>,
                output_names: &mut BTreeSet<String>,
                cursor: &mut u64,
                source: &str,
                name: String,
                dtype: &str,
                shape: Vec<usize>,
                byte_len: u64,
                role: TeOutputRole|
     -> Result<(), TeLayoutError> {
        if name == RESERVED_METADATA_NAME
            || (role != TeOutputRole::Preserved && source_names.contains(name.as_str()))
            || !output_names.insert(name.clone())
        {
            return Err(TeLayoutError::OutputNameCollision { name });
        }
        let end = cursor
            .checked_add(byte_len)
            .ok_or_else(|| TeLayoutError::SizeOverflow { name: name.clone() })?;
        tensors.push(TeOutputTensorPlan {
            name,
            source_name: source.to_owned(),
            dtype: dtype.to_owned(),
            shape,
            byte_len,
            data_offsets: *cursor..end,
            role,
        });
        *cursor = end;
        Ok(())
    };

    for source in sorted {
        if !selected.contains(source.name.as_str()) {
            push(
                &mut tensors,
                &mut output_names,
                &mut cursor,
                &source.name,
                source.name.clone(),
                &source.dtype,
                source.shape.clone(),
                source.byte_len,
                TeOutputRole::Preserved,
            )?;
            continue;
        }
        let [rows, columns] = validate_matrix(source)?;
        let overflow = || TeLayoutError::SizeOverflow {
            name: source.name.clone(),
        };
        let [padded_rows, padded_blocks] = te_scale_shape(rows, columns).ok_or_else(overflow)?;
        let data_bytes = rows.checked_mul(columns / 2).ok_or_else(overflow)?;
        let scale_bytes = padded_rows
            .checked_mul(padded_blocks)
            .ok_or_else(overflow)?;
        let as_u64 = |value: usize| u64::try_from(value).map_err(|_| overflow());
        push(
            &mut tensors,
            &mut output_names,
            &mut cursor,
            &source.name,
            format!("{}.rowwise_data", source.name),
            U8_DTYPE,
            vec![rows, columns / 2],
            as_u64(data_bytes)?,
            TeOutputRole::RowwiseData,
        )?;
        push(
            &mut tensors,
            &mut output_names,
            &mut cursor,
            &source.name,
            format!("{}.rowwise_scale_inv", source.name),
            U8_DTYPE,
            vec![padded_rows, padded_blocks],
            as_u64(scale_bytes)?,
            TeOutputRole::RowwiseScaleInv,
        )?;
        push(
            &mut tensors,
            &mut output_names,
            &mut cursor,
            &source.name,
            format!("{}.amax_rowwise", source.name),
            F32_DTYPE,
            vec![1],
            4,
            TeOutputRole::AmaxRowwise,
        )?;
    }

    let mut quantized_names: Vec<String> = quantized_names.to_vec();
    quantized_names.sort();
    Ok(TeOutputPlan {
        tensors,
        total_data_bytes: cursor,
        quantized_names,
    })
}

/// Errors returned while reading or validating a container.
#[derive(Debug)]
pub enum TeContainerError {
    /// A file could not be opened or parsed as SafeTensors.
    Safetensors(SafetensorsError),
    /// The container violates the schema; the message says how.
    Invalid { message: String },
}

impl fmt::Display for TeContainerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Safetensors(error) => error.fmt(formatter),
            Self::Invalid { message } => {
                write!(formatter, "invalid Transformer Engine container: {message}")
            }
        }
    }
}

impl std::error::Error for TeContainerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Safetensors(error) => Some(error),
            Self::Invalid { .. } => None,
        }
    }
}

impl From<SafetensorsError> for TeContainerError {
    fn from(error: SafetensorsError) -> Self {
        Self::Safetensors(error)
    }
}

fn invalid(message: impl Into<String>) -> TeContainerError {
    TeContainerError::Invalid {
        message: message.into(),
    }
}

/// One tensor described by a container manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TeManifestTensor {
    /// A Transformer Engine matrix and its three field names.
    Quantized {
        logical_shape: [usize; 2],
        original_dtype: String,
        rowwise_data: String,
        rowwise_scale_inv: String,
        amax_rowwise: String,
    },
    /// A tensor copied unchanged.
    Preserved { dtype: String, shape: Vec<usize> },
}

/// The validated manifest of one container file, by source tensor name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeContainerManifest {
    /// Tensors of this file, keyed by source name.
    pub tensors: BTreeMap<String, TeManifestTensor>,
}

fn build_manifest(
    summaries: &[TensorSummary],
    plan: &TeOutputPlan,
) -> Result<String, Nvfp4WriterError> {
    let quantized: BTreeSet<&str> = plan
        .quantized_source_names()
        .iter()
        .map(String::as_str)
        .collect();
    let mut tensors = BTreeMap::new();
    for summary in summaries {
        let mut record = BTreeMap::new();
        if quantized.contains(summary.name.as_str()) {
            record.insert("action".to_owned(), Value::String("quantized".to_owned()));
            record.insert("logical_shape".to_owned(), json_shape(&summary.shape)?);
            record.insert(
                "original_dtype".to_owned(),
                Value::String(summary.dtype.clone()),
            );
            let mut fields = BTreeMap::new();
            for (key, suffix) in [
                ("rowwise_data", "rowwise_data"),
                ("rowwise_scale_inv", "rowwise_scale_inv"),
                ("amax_rowwise", "amax_rowwise"),
            ] {
                fields.insert(
                    key.to_owned(),
                    Value::String(format!("{}.{suffix}", summary.name)),
                );
            }
            record.insert("fields".to_owned(), json_object(fields));
        } else {
            record.insert("action".to_owned(), Value::String("preserved".to_owned()));
            record.insert("dtype".to_owned(), Value::String(summary.dtype.clone()));
            record.insert("shape".to_owned(), json_shape(&summary.shape)?);
        }
        tensors.insert(summary.name.clone(), json_object(record));
    }

    let mut quantization = BTreeMap::new();
    quantization.insert(
        "block_scale_format".to_owned(),
        Value::String("E4M3".to_owned()),
    );
    quantization.insert("block_size".to_owned(), Value::from(BLOCK_SIZE as u64));
    quantization.insert("data_format".to_owned(), Value::String("E2M1".to_owned()));
    quantization.insert(
        "scaling".to_owned(),
        Value::String("rowwise_1x16_tensor_global".to_owned()),
    );
    let mut runtime = BTreeMap::new();
    runtime.insert("name".to_owned(), Value::String(TE_RUNTIME_NAME.to_owned()));
    runtime.insert(
        "version".to_owned(),
        Value::String(TE_RUNTIME_VERSION.to_owned()),
    );
    let mut scale_storage = BTreeMap::new();
    scale_storage.insert("gemm_swizzled".to_owned(), Value::Bool(false));
    scale_storage.insert(
        "padding".to_owned(),
        Value::Array(vec![
            Value::from(ROW_ALIGNMENT as u64),
            Value::from(COLUMN_ALIGNMENT as u64),
        ]),
    );

    let mut manifest = BTreeMap::new();
    manifest.insert(
        "global_scale_denominator".to_owned(),
        serde_json::Number::from_f64(f64::from(GLOBAL_SCALE_DENOMINATOR))
            .map(Value::Number)
            .expect("the denominator is finite"),
    );
    manifest.insert("profile_id".to_owned(), Value::String(PROFILE.to_owned()));
    manifest.insert("quantization".to_owned(), json_object(quantization));
    manifest.insert("runtime".to_owned(), json_object(runtime));
    manifest.insert("scale_storage".to_owned(), json_object(scale_storage));
    manifest.insert("schema_version".to_owned(), Value::from(SCHEMA_VERSION));
    manifest.insert("tensors".to_owned(), json_object(tensors));
    serde_json::to_string(&json_object(manifest))
        .map_err(|source| Nvfp4WriterError::Serialization { source })
}

fn build_header(
    summaries: &[TensorSummary],
    plan: &TeOutputPlan,
    destination: &Path,
) -> Result<Vec<u8>, Nvfp4WriterError> {
    let mut sorted: Vec<&TensorSummary> = summaries.iter().collect();
    sorted.sort_by(|left, right| left.name.cmp(&right.name));
    let sorted: Vec<TensorSummary> = sorted.into_iter().cloned().collect();

    let mut metadata = BTreeMap::new();
    metadata.insert("modelq.format".to_owned(), TE_CONTAINER_FORMAT.to_owned());
    metadata.insert("modelq.manifest".to_owned(), build_manifest(&sorted, plan)?);

    let mut root = BTreeMap::new();
    root.insert(RESERVED_METADATA_NAME.to_owned(), json_string_map(metadata));
    for tensor in &plan.tensors {
        let mut descriptor = BTreeMap::new();
        descriptor.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::from(tensor.data_offsets.start),
                Value::from(tensor.data_offsets.end),
            ]),
        );
        descriptor.insert("dtype".to_owned(), Value::String(tensor.dtype.clone()));
        descriptor.insert("shape".to_owned(), json_shape(&tensor.shape)?);
        root.insert(tensor.name.clone(), json_object(descriptor));
    }

    let raw =
        serde_json::to_vec(&root).map_err(|source| Nvfp4WriterError::Serialization { source })?;
    let padded_len = raw
        .len()
        .checked_add(HEADER_ALIGNMENT - 1)
        .ok_or(Nvfp4WriterError::HeaderLengthOverflow)?
        / HEADER_ALIGNMENT
        * HEADER_ALIGNMENT;
    if padded_len > MAX_HEADER_SIZE {
        return Err(Nvfp4WriterError::HeaderTooLarge {
            path: destination.to_owned(),
            size: padded_len,
        });
    }
    let total = HEADER_LENGTH_BYTES
        .checked_add(padded_len)
        .ok_or(Nvfp4WriterError::HeaderLengthOverflow)?;
    let mut header = Vec::with_capacity(total);
    header.extend_from_slice(
        &u64::try_from(padded_len)
            .map_err(|_| Nvfp4WriterError::HeaderLengthOverflow)?
            .to_le_bytes(),
    );
    header.extend_from_slice(&raw);
    header.resize(total, b' ');
    Ok(header)
}

/// Writes a Transformer Engine container with the sequential quantizer.
///
/// See [`write_te_nvfp4_safetensors_with`].
pub fn write_te_nvfp4_safetensors(
    source: &impl TensorSource,
    plan: &TeOutputPlan,
    destination: impl AsRef<Path>,
) -> Result<(), Nvfp4WriterError> {
    write_te_nvfp4_safetensors_with(source, plan, destination, Nvfp4Execution::Sequential)
}

/// Writes a Transformer Engine container from a checked plan.
///
/// The source stays read-only.  The destination must not exist and must not
/// be one of the source files; data goes to a temporary file that is renamed
/// into place only after a complete, synchronized write, so any failure
/// leaves nothing behind.  `execution` selects the sequential or parallel
/// quantizer; both write identical bytes.
pub fn write_te_nvfp4_safetensors_with(
    source: &impl TensorSource,
    plan: &TeOutputPlan,
    destination: impl AsRef<Path>,
    execution: Nvfp4Execution,
) -> Result<(), Nvfp4WriterError> {
    let destination = destination.as_ref().to_owned();
    if destination.file_name().is_none() {
        return Err(Nvfp4WriterError::InvalidDestination { path: destination });
    }
    if let Some(conflict) = source
        .source_paths()
        .into_iter()
        .find(|path| paths_refer_to_same_file(path, &destination))
    {
        return Err(Nvfp4WriterError::SourceDestinationConflict {
            source: conflict,
            destination,
        });
    }
    if destination.exists() {
        return Err(Nvfp4WriterError::DestinationExists { path: destination });
    }

    let summaries = source.tensor_summaries();
    let expected = plan_te_output(&summaries, plan.quantized_source_names()).map_err(|error| {
        Nvfp4WriterError::TransformerEngineLayout {
            message: error.to_string(),
        }
    })?;
    if expected != *plan {
        return Err(Nvfp4WriterError::PlanMismatch);
    }

    let header = build_header(&summaries, plan, &destination)?;
    let shapes: BTreeMap<&str, &[usize]> = summaries
        .iter()
        .map(|summary| (summary.name.as_str(), summary.shape.as_slice()))
        .collect();
    let (temporary_path, mut file) = create_temporary_file(&destination)?;
    let mut temporary = TemporaryOutput::new(temporary_path.clone());

    let write_result = (|| {
        file.write_all(&header)
            .map_err(|source| io_error(&destination, source))?;
        write_data(&mut file, &destination, source, plan, &shapes, execution)?;
        file.sync_all()
            .map_err(|source| io_error(&destination, source))
    })();
    drop(file);
    write_result?;

    // Never replace a destination that appeared while writing.
    if destination.exists() {
        return Err(Nvfp4WriterError::DestinationExists { path: destination });
    }
    fs::rename(&temporary_path, &destination).map_err(|source| io_error(&destination, source))?;
    temporary.committed = true;
    Ok(())
}

fn write_data(
    file: &mut File,
    output_path: &Path,
    source: &impl TensorSource,
    plan: &TeOutputPlan,
    shapes: &BTreeMap<&str, &[usize]>,
    execution: Nvfp4Execution,
) -> Result<(), Nvfp4WriterError> {
    let mut cursor = 0_u64;
    let mut quantized: BTreeMap<String, nvfp4::StreamedQuantization> = BTreeMap::new();
    let layout_error = |message: String| Nvfp4WriterError::TransformerEngineLayout { message };
    let write = |file: &mut File, bytes: &[u8]| {
        file.write_all(bytes)
            .map_err(|source| io_error(output_path, source))
    };

    for tensor in &plan.tensors {
        if tensor.data_offsets.start != cursor {
            return Err(Nvfp4WriterError::PlanOffsetMismatch {
                name: tensor.name.clone(),
                expected_start: cursor,
                actual_start: tensor.data_offsets.start,
            });
        }
        match tensor.role {
            TeOutputRole::Preserved => {
                source
                    .with_tensor_bytes(&tensor.source_name, |bytes| {
                        let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
                        if actual != tensor.byte_len {
                            return Err(Nvfp4WriterError::DataLengthMismatch {
                                name: tensor.name.clone(),
                                expected: tensor.byte_len,
                                actual,
                            });
                        }
                        write(file, bytes)
                    })
                    .map_err(|source| Nvfp4WriterError::Source { source })??;
            }
            TeOutputRole::RowwiseData => {
                let streamed = stream_quantized_payload(
                    source,
                    &tensor.source_name,
                    &tensor.name,
                    tensor.byte_len,
                    file,
                    output_path,
                    execution,
                )?;
                quantized.insert(tensor.source_name.clone(), streamed);
            }
            TeOutputRole::RowwiseScaleInv => {
                let streamed = quantized.get(&tensor.source_name).ok_or_else(|| {
                    Nvfp4WriterError::MissingQuantizedTensor {
                        tensor_name: tensor.source_name.clone(),
                    }
                })?;
                let shape = shapes.get(tensor.source_name.as_str()).ok_or_else(|| {
                    layout_error(format!("missing shape of {:?}", tensor.source_name))
                })?;
                let [rows, columns] = shape[..] else {
                    return Err(layout_error(format!(
                        "{:?} is not rank two",
                        tensor.source_name
                    )));
                };
                let [padded_rows, padded_blocks] = te_scale_shape(rows, columns)
                    .ok_or_else(|| layout_error("scale shape overflows".to_owned()))?;
                let blocks_per_row = columns / BLOCK_SIZE;
                let scales = streamed.block_scales();
                if scales.len() != rows * blocks_per_row {
                    return Err(layout_error(format!(
                        "{:?} produced {} block scales instead of {}",
                        tensor.source_name,
                        scales.len(),
                        rows * blocks_per_row
                    )));
                }
                let zero_row = vec![0_u8; padded_blocks];
                let mut written = 0_u64;
                for row in scales.chunks(blocks_per_row) {
                    write(file, row)?;
                    write(file, &zero_row[..padded_blocks - blocks_per_row])?;
                    written += padded_blocks as u64;
                }
                for _ in rows..padded_rows {
                    write(file, &zero_row)?;
                    written += padded_blocks as u64;
                }
                if written != tensor.byte_len {
                    return Err(Nvfp4WriterError::DataLengthMismatch {
                        name: tensor.name.clone(),
                        expected: tensor.byte_len,
                        actual: written,
                    });
                }
            }
            TeOutputRole::AmaxRowwise => {
                let streamed = quantized.get(&tensor.source_name).ok_or_else(|| {
                    Nvfp4WriterError::MissingQuantizedTensor {
                        tensor_name: tensor.source_name.clone(),
                    }
                })?;
                let amax = if streamed.block_scales().iter().all(|&scale| scale == 0) {
                    0.0
                } else {
                    streamed.global_scale() * GLOBAL_SCALE_DENOMINATOR
                };
                if !amax.is_finite() {
                    return Err(layout_error(format!(
                        "the derived amax of {:?} is not finite",
                        tensor.source_name
                    )));
                }
                write(file, &amax.to_le_bytes())?;
            }
        }
        cursor = tensor.data_offsets.end;
    }
    if cursor != plan.total_data_bytes {
        return Err(Nvfp4WriterError::DataLengthMismatch {
            name: "<data section>".to_owned(),
            expected: plan.total_data_bytes,
            actual: cursor,
        });
    }
    Ok(())
}

// ----------------------------------------------------------------- reading

fn manifest_object<'a>(
    value: &'a Value,
    what: &str,
) -> Result<&'a serde_json::Map<String, Value>, TeContainerError> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{what} must be a JSON object")))
}

fn manifest_str<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
    what: &str,
) -> Result<&'a str, TeContainerError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("{what}: {key} must be a string")))
}

fn manifest_shape(
    object: &serde_json::Map<String, Value>,
    key: &str,
    what: &str,
) -> Result<Vec<usize>, TeContainerError> {
    object
        .get(key)
        .and_then(Value::as_array)
        .and_then(|values| {
            values
                .iter()
                .map(|value| value.as_u64().and_then(|value| usize::try_from(value).ok()))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| {
            invalid(format!(
                "{what}: {key} must be a list of non-negative integers"
            ))
        })
}

/// Reads and validates one container file's manifest against its tensor
/// header: the fixed schema values, every listed tensor and field present with
/// the exact dtype and shape, and no tensor the manifest does not account for.
pub fn read_te_container_manifest(
    file: &MappedSafetensors,
) -> Result<TeContainerManifest, TeContainerError> {
    let metadata = file.metadata();
    if metadata.get("modelq.format").map(String::as_str) != Some(TE_CONTAINER_FORMAT) {
        return Err(invalid(format!(
            "modelq.format must be {TE_CONTAINER_FORMAT:?}"
        )));
    }
    let text = metadata
        .get("modelq.manifest")
        .ok_or_else(|| invalid("modelq.manifest is missing"))?;
    let manifest: Value = serde_json::from_str(text).map_err(|error| invalid(error.to_string()))?;
    let root = manifest_object(&manifest, "the manifest")?;

    if root.get("schema_version").and_then(Value::as_u64) != Some(SCHEMA_VERSION) {
        return Err(invalid("schema_version must be 2"));
    }
    if manifest_str(root, "profile_id", "manifest")? != PROFILE {
        return Err(invalid(format!("profile_id must be {PROFILE:?}")));
    }
    let runtime = manifest_object(
        root.get("runtime")
            .ok_or_else(|| invalid("runtime is missing"))?,
        "runtime",
    )?;
    if manifest_str(runtime, "name", "runtime")? != TE_RUNTIME_NAME
        || manifest_str(runtime, "version", "runtime")? != TE_RUNTIME_VERSION
    {
        return Err(invalid(format!(
            "runtime must be {TE_RUNTIME_NAME} {TE_RUNTIME_VERSION}"
        )));
    }
    let quantization = manifest_object(
        root.get("quantization")
            .ok_or_else(|| invalid("quantization is missing"))?,
        "quantization",
    )?;
    if manifest_str(quantization, "data_format", "quantization")? != "E2M1"
        || manifest_str(quantization, "block_scale_format", "quantization")? != "E4M3"
        || manifest_str(quantization, "scaling", "quantization")? != "rowwise_1x16_tensor_global"
        || quantization.get("block_size").and_then(Value::as_u64) != Some(BLOCK_SIZE as u64)
    {
        return Err(invalid("quantization block does not match the profile"));
    }
    let storage = manifest_object(
        root.get("scale_storage")
            .ok_or_else(|| invalid("scale_storage is missing"))?,
        "scale_storage",
    )?;
    if storage.get("gemm_swizzled") != Some(&Value::Bool(false))
        || manifest_shape(storage, "padding", "scale_storage")? != [ROW_ALIGNMENT, COLUMN_ALIGNMENT]
    {
        return Err(invalid("scale_storage does not match the profile"));
    }
    if root.get("global_scale_denominator").and_then(Value::as_f64)
        != Some(f64::from(GLOBAL_SCALE_DENOMINATOR))
    {
        return Err(invalid("global_scale_denominator must be 2688.0"));
    }

    let listed = manifest_object(
        root.get("tensors")
            .ok_or_else(|| invalid("tensors is missing"))?,
        "tensors",
    )?;
    let actual: BTreeMap<String, TensorSummary> = file
        .tensors()
        .map(|summary| (summary.name.clone(), summary.clone()))
        .collect();
    let mut accounted: BTreeSet<String> = BTreeSet::new();
    let mut tensors = BTreeMap::new();
    for (name, entry) in listed {
        let what = format!("tensor {name:?}");
        let entry = manifest_object(entry, &what)?;
        match manifest_str(entry, "action", &what)? {
            "preserved" => {
                let dtype = manifest_str(entry, "dtype", &what)?.to_owned();
                let shape = manifest_shape(entry, "shape", &what)?;
                let summary = actual.get(name).ok_or_else(|| {
                    invalid(format!("{what} is listed but missing from the file"))
                })?;
                if summary.dtype != dtype || summary.shape != shape {
                    return Err(invalid(format!("{what} does not match its manifest entry")));
                }
                accounted.insert(name.clone());
                tensors.insert(name.clone(), TeManifestTensor::Preserved { dtype, shape });
            }
            "quantized" => {
                let [rows, columns] = manifest_shape(entry, "logical_shape", &what)?[..] else {
                    return Err(invalid(format!("{what}: logical_shape must be rank two")));
                };
                if rows == 0 || columns == 0 || rows % BLOCK_SIZE != 0 || columns % BLOCK_SIZE != 0
                {
                    return Err(invalid(format!(
                        "{what}: dimensions must be positive multiples of 16"
                    )));
                }
                let fields = manifest_object(
                    entry
                        .get("fields")
                        .ok_or_else(|| invalid(format!("{what}: fields is missing")))?,
                    "fields",
                )?;
                let [padded_rows, padded_blocks] = te_scale_shape(rows, columns)
                    .ok_or_else(|| invalid(format!("{what}: scale shape overflows")))?;
                let mut names = Vec::new();
                for (key, dtype, shape) in [
                    ("rowwise_data", U8_DTYPE, vec![rows, columns / 2]),
                    (
                        "rowwise_scale_inv",
                        U8_DTYPE,
                        vec![padded_rows, padded_blocks],
                    ),
                    ("amax_rowwise", F32_DTYPE, vec![1]),
                ] {
                    let field = manifest_str(fields, key, &what)?.to_owned();
                    let summary = actual.get(&field).ok_or_else(|| {
                        invalid(format!("{what}: field {field:?} is missing from the file"))
                    })?;
                    if summary.dtype != dtype || summary.shape != shape {
                        return Err(invalid(format!(
                            "{what}: field {field:?} must be {dtype} {shape:?}"
                        )));
                    }
                    accounted.insert(field.clone());
                    names.push(field);
                }
                tensors.insert(
                    name.clone(),
                    TeManifestTensor::Quantized {
                        logical_shape: [rows, columns],
                        original_dtype: manifest_str(entry, "original_dtype", &what)?.to_owned(),
                        rowwise_data: names[0].clone(),
                        rowwise_scale_inv: names[1].clone(),
                        amax_rowwise: names[2].clone(),
                    },
                );
            }
            other => return Err(invalid(format!("{what}: unknown action {other:?}"))),
        }
    }
    if let Some(extra) = actual.keys().find(|name| !accounted.contains(*name)) {
        return Err(invalid(format!(
            "tensor {extra:?} is in the file but not in the manifest"
        )));
    }
    Ok(TeContainerManifest { tensors })
}

/// Validates one matrix's raw fields and returns a lazy decoder.
///
/// Checks the byte lengths, that all scale padding is zero, that every used
/// scale is a valid non-negative E4M3 value, that the amax is finite and
/// non-negative, and that a zero scale block holds only zero values.  Values
/// are reconstructed as `e2m1 * e4m3_scale * (amax / 2688)`, exactly as the
/// runtime recovers them; an all-zero tensor stores `amax = 0`.
pub fn te_matrix_values<'a>(
    rows: usize,
    columns: usize,
    rowwise_data: &'a [u8],
    rowwise_scale_inv: &'a [u8],
    amax: f32,
) -> Result<impl ExactSizeIterator<Item = f32> + 'a, TeContainerError> {
    if rows == 0 || columns == 0 || columns % BLOCK_SIZE != 0 {
        return Err(invalid(format!(
            "shape [{rows}, {columns}] is not a valid matrix"
        )));
    }
    let [padded_rows, padded_blocks] =
        te_scale_shape(rows, columns).ok_or_else(|| invalid("scale shape overflows"))?;
    let blocks_per_row = columns / BLOCK_SIZE;
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid("element count overflows"))?;
    if rowwise_data.len() != elements / 2 {
        return Err(invalid(format!(
            "rowwise_data has {} bytes instead of {}",
            rowwise_data.len(),
            elements / 2
        )));
    }
    if rowwise_scale_inv.len() != padded_rows * padded_blocks {
        return Err(invalid(format!(
            "rowwise_scale_inv has {} bytes instead of {}",
            rowwise_scale_inv.len(),
            padded_rows * padded_blocks
        )));
    }
    if !amax.is_finite() || amax < 0.0 {
        return Err(invalid(format!(
            "amax {amax:?} must be finite and non-negative"
        )));
    }

    let code = |index: usize| -> u8 {
        let byte = rowwise_data[index / 2];
        if index % 2 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        }
    };
    let mut all_scales_zero = true;
    for (row, scales) in rowwise_scale_inv.chunks(padded_blocks).enumerate() {
        let (used, padding) = scales.split_at(blocks_per_row);
        if row >= rows {
            if scales.iter().any(|&byte| byte != 0) {
                return Err(invalid(format!("scale padding row {row} is not zero")));
            }
            continue;
        }
        if padding.iter().any(|&byte| byte != 0) {
            return Err(invalid(format!("scale padding in row {row} is not zero")));
        }
        for (block, &bits) in used.iter().enumerate() {
            let decoded = fp8_e4m3::decode(bits);
            if bits != 0 && (bits & 0x80 != 0 || !decoded.is_finite() || decoded <= 0.0) {
                return Err(invalid(format!(
                    "scale of row {row} block {block} is not a positive E4M3 value: {bits:#04x}"
                )));
            }
            if bits == 0 {
                let first = row * columns + block * BLOCK_SIZE;
                if (first..first + BLOCK_SIZE).any(|index| code(index) & 0x07 != 0) {
                    return Err(invalid(format!(
                        "row {row} block {block} has a zero scale but nonzero values"
                    )));
                }
            } else {
                all_scales_zero = false;
            }
        }
    }
    if amax == 0.0 && !all_scales_zero {
        return Err(invalid("amax is zero but the tensor has nonzero scales"));
    }
    let global_scale = te_global_scale(amax);
    if !global_scale.is_finite() || global_scale <= 0.0 {
        return Err(invalid(format!(
            "amax {amax:?} gives an invalid decode scale"
        )));
    }

    Ok((0..elements).map(move |index| {
        let row = index / columns;
        let block = (index % columns) / BLOCK_SIZE;
        let scale = fp8_e4m3::decode(rowwise_scale_inv[row * padded_blocks + block]);
        fp4_e2m1::decode(code(index)) * scale * global_scale
    }))
}
