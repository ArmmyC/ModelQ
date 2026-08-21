//! Streaming ModelQ-native SafeTensors output.
//!
//! The writer consumes a checked [`OutputLayoutPlan`]
//! and a read-only mapped source. It writes the header once, then processes
//! one planned tensor at a time. Quantized tensors use the reference scalar
//! INT8 algorithm; preserved tensors are copied byte-for-byte from the source
//! mapping.

use std::{
    collections::BTreeMap,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use serde_json::Value;

use crate::{
    io::{
        layout::{LayoutError, OutputLayoutPlan, OutputTensorRole, plan_output_layout},
        safetensors::{MappedSafetensors, SafetensorsError, TensorSummary},
    },
    quant::{
        int8::{
            DEFAULT_CHUNK_ELEMENTS, Int8Error, QuantizationStreamError, quantize_replay_chunks,
        },
        policy::{PolicyAction, TensorDecision},
    },
};

const HEADER_LENGTH_BYTES: usize = 8;
const HEADER_ALIGNMENT: usize = 8;
const MAX_HEADER_SIZE: usize = 100_000_000;
const RESERVED_METADATA_NAME: &str = "__metadata__";
const MODELQ_FORMAT: &str = "modelq-native";
const MODELQ_FORMAT_VERSION: &str = "1";
const MODELQ_QUANTIZATION: &str = "int8";
const MODELQ_SCHEME: &str = "symmetric-per-tensor";
const MODELQ_ALGORITHM: &str = "max-abs-scale-v0";
const MODELQ_ROUNDING: &str = "ties-away-from-zero";
const MODELQ_QMIN: &str = "-127";
const MODELQ_QMAX: &str = "127";
const MANIFEST_SCHEMA: &str = "modelq.int8.manifest.v1";
const SCALE_BYTE_LEN: u64 = 4;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Errors returned while creating a ModelQ-native SafeTensors output.
#[derive(Debug)]
pub enum WriterError {
    /// The destination already exists. The writer never replaces an existing
    /// file, which keeps failed conversions from destroying prior artifacts.
    DestinationExists { path: PathBuf },
    /// The source and destination resolve to the same existing path.
    SourceDestinationConflict {
        source: PathBuf,
        destination: PathBuf,
    },
    /// The destination does not name a file path.
    InvalidDestination { path: PathBuf },
    /// Source metadata and the supplied plan could not be reconciled.
    Layout { source: LayoutError },
    /// The caller supplied a plan different from the plan derived from the
    /// current source metadata and decisions.
    PlanMismatch,
    /// A mapped source read failed.
    Source { source: SafetensorsError },
    /// The scalar quantizer rejected one source tensor.
    Quantization {
        tensor_name: String,
        source: Int8Error,
    },
    /// The deterministic header could not be serialized.
    Serialization { source: serde_json::Error },
    /// An output file operation failed.
    Io { path: PathBuf, source: io::Error },
    /// The serialized header exceeded the SafeTensors header limit.
    HeaderTooLarge { path: PathBuf, size: usize },
    /// Header length arithmetic overflowed.
    HeaderLengthOverflow,
    /// A payload length did not match the plan.
    DataLengthMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },
    /// A supplied plan did not have contiguous data offsets.
    PlanOffsetMismatch {
        name: String,
        expected_start: u64,
        actual_start: u64,
    },
    /// A scale tensor appeared without its preceding quantized payload.
    MissingScale { tensor_name: String },
}

impl fmt::Display for WriterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DestinationExists { path } => {
                write!(
                    formatter,
                    "output destination {} already exists",
                    path.display()
                )
            }
            Self::SourceDestinationConflict {
                source,
                destination,
            } => write!(
                formatter,
                "output destination {} is the source file {}; refusing in-place writing",
                destination.display(),
                source.display()
            ),
            Self::InvalidDestination { path } => {
                write!(
                    formatter,
                    "output path {} is not a file path",
                    path.display()
                )
            }
            Self::Layout { source } => {
                write!(formatter, "cannot write invalid output layout: {source}")
            }
            Self::PlanMismatch => formatter.write_str(
                "the supplied output layout does not match the current source metadata and policy",
            ),
            Self::Source { source } => write!(formatter, "could not read source tensor: {source}"),
            Self::Quantization {
                tensor_name,
                source,
            } => write!(
                formatter,
                "could not quantize tensor {tensor_name:?}: {source}"
            ),
            Self::Serialization { source } => {
                write!(
                    formatter,
                    "could not serialize SafeTensors header: {source}"
                )
            }
            Self::Io { path, source } => {
                write!(formatter, "could not write {}: {source}", path.display())
            }
            Self::HeaderTooLarge { path, size } => write!(
                formatter,
                "SafeTensors header for {} is too large ({size} bytes)",
                path.display()
            ),
            Self::HeaderLengthOverflow => {
                formatter.write_str("SafeTensors header length arithmetic overflowed")
            }
            Self::DataLengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "output tensor {name:?} requires {expected} bytes but the payload has {actual}"
            ),
            Self::PlanOffsetMismatch {
                name,
                expected_start,
                actual_start,
            } => write!(
                formatter,
                "output tensor {name:?} starts at {actual_start}, expected {expected_start}"
            ),
            Self::MissingScale { tensor_name } => write!(
                formatter,
                "quantization scale for source tensor {tensor_name:?} was not produced"
            ),
        }
    }
}

impl std::error::Error for WriterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Layout { source } => Some(source),
            Self::Source { source } => Some(source),
            Self::Quantization { source, .. } => Some(source),
            Self::Serialization { source } => Some(source),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Writes a planned ModelQ-native INT8 SafeTensors file.
///
/// The source mapping remains read-only. The destination must not already
/// exist; data is first written to a unique temporary file in the destination
/// directory and is renamed into place only after the complete output has
/// been flushed and synchronized. Any failure therefore leaves both the
/// source and the requested destination unchanged.
pub fn write_safetensors(
    source: &MappedSafetensors,
    plan: &OutputLayoutPlan,
    decisions: &[TensorDecision],
    destination: impl AsRef<Path>,
) -> Result<(), WriterError> {
    let destination = destination.as_ref().to_owned();
    if destination.file_name().is_none() {
        return Err(WriterError::InvalidDestination { path: destination });
    }
    if paths_refer_to_same_file(source.path(), &destination) {
        return Err(WriterError::SourceDestinationConflict {
            source: source.path().to_owned(),
            destination,
        });
    }
    if destination.exists() {
        return Err(WriterError::DestinationExists { path: destination });
    }

    let inspection = source.inspection();
    let expected_plan = plan_output_layout(&inspection.tensors, decisions)
        .map_err(|source| WriterError::Layout { source })?;
    if expected_plan != *plan {
        return Err(WriterError::PlanMismatch);
    }

    let header = build_header(&inspection.tensors, decisions, plan)?;
    let (temporary_path, mut file) = create_temporary_file(&destination)?;
    let mut temporary = TemporaryOutput::new(temporary_path.clone());

    let write_result = (|| {
        file.write_all(&header)
            .map_err(|source| io_error(&destination, source))?;
        write_data(&mut file, &destination, source, plan)?;
        file.sync_all()
            .map_err(|source| io_error(&destination, source))
    })();
    drop(file);
    write_result?;

    // Refuse a destination that appeared while the temporary output was being
    // produced. On Unix rename would otherwise replace it; preserving the
    // existing artifact is the safer cross-platform behavior.
    if destination.exists() {
        return Err(WriterError::DestinationExists { path: destination });
    }
    fs::rename(&temporary_path, &destination).map_err(|source| io_error(&destination, source))?;
    temporary.committed = true;
    Ok(())
}

fn build_header(
    summaries: &[TensorSummary],
    decisions: &[TensorDecision],
    plan: &OutputLayoutPlan,
) -> Result<Vec<u8>, WriterError> {
    let mut sorted_summaries = summaries.iter().collect::<Vec<_>>();
    sorted_summaries.sort_by(|left, right| left.name.cmp(&right.name));
    let decisions_by_name = decisions
        .iter()
        .map(|decision| (decision.name.clone(), decision))
        .collect::<BTreeMap<_, _>>();

    let mut manifest_tensors = BTreeMap::new();
    for summary in sorted_summaries {
        let decision = decisions_by_name
            .get(&summary.name)
            .expect("plan validation guarantees a decision for every source");
        let mut record = BTreeMap::new();
        match decision.action {
            PolicyAction::Preserve => {
                record.insert("action".to_owned(), Value::String("preserved".to_owned()));
                record.insert(
                    "original_dtype".to_owned(),
                    Value::String(summary.dtype.clone()),
                );
                record.insert("original_shape".to_owned(), json_shape(&summary.shape)?);
                record.insert(
                    "tensor_name".to_owned(),
                    Value::String(summary.name.clone()),
                );
            }
            PolicyAction::Quantize => {
                record.insert("action".to_owned(), Value::String("quantized".to_owned()));
                record.insert(
                    "original_dtype".to_owned(),
                    Value::String(summary.dtype.clone()),
                );
                record.insert("original_shape".to_owned(), json_shape(&summary.shape)?);
                record.insert("qdata_dtype".to_owned(), Value::String("I8".to_owned()));
                record.insert(
                    "qdata_name".to_owned(),
                    Value::String(format!("{}.qdata", summary.name)),
                );
                record.insert("scale_dtype".to_owned(), Value::String("F32".to_owned()));
                record.insert(
                    "scale_name".to_owned(),
                    Value::String(format!("{}.scale", summary.name)),
                );
                record.insert("scale_shape".to_owned(), Value::Array(Vec::new()));
            }
        }
        manifest_tensors.insert(summary.name.clone(), json_object(record));
    }

    let mut manifest = BTreeMap::new();
    manifest.insert(
        "schema".to_owned(),
        Value::String(MANIFEST_SCHEMA.to_owned()),
    );
    manifest.insert("tensors".to_owned(), json_object(manifest_tensors));
    let manifest_json = serde_json::to_string(&json_object(manifest))
        .map_err(|source| WriterError::Serialization { source })?;

    let mut metadata = BTreeMap::new();
    metadata.insert("modelq.algorithm".to_owned(), MODELQ_ALGORITHM.to_owned());
    metadata.insert("modelq.format".to_owned(), MODELQ_FORMAT.to_owned());
    metadata.insert(
        "modelq.format_version".to_owned(),
        MODELQ_FORMAT_VERSION.to_owned(),
    );
    metadata.insert("modelq.manifest".to_owned(), manifest_json);
    metadata.insert(
        "modelq.quantization".to_owned(),
        MODELQ_QUANTIZATION.to_owned(),
    );
    metadata.insert("modelq.qmax".to_owned(), MODELQ_QMAX.to_owned());
    metadata.insert("modelq.qmin".to_owned(), MODELQ_QMIN.to_owned());
    metadata.insert("modelq.rounding".to_owned(), MODELQ_ROUNDING.to_owned());
    metadata.insert("modelq.scheme".to_owned(), MODELQ_SCHEME.to_owned());

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

    let raw_header =
        serde_json::to_vec(&root).map_err(|source| WriterError::Serialization { source })?;
    let padded_len = raw_header
        .len()
        .checked_add(HEADER_ALIGNMENT - 1)
        .ok_or(WriterError::HeaderLengthOverflow)?
        / HEADER_ALIGNMENT
        * HEADER_ALIGNMENT;
    if padded_len > MAX_HEADER_SIZE {
        return Err(WriterError::HeaderTooLarge {
            path: PathBuf::from("<planned output>"),
            size: padded_len,
        });
    }
    let total_header_len = HEADER_LENGTH_BYTES
        .checked_add(padded_len)
        .ok_or(WriterError::HeaderLengthOverflow)?;
    let padded_len_u64 =
        u64::try_from(padded_len).map_err(|_| WriterError::HeaderLengthOverflow)?;
    let mut header = Vec::with_capacity(total_header_len);
    header.extend_from_slice(&padded_len_u64.to_le_bytes());
    header.extend_from_slice(&raw_header);
    header.resize(total_header_len, b' ');
    Ok(header)
}

fn write_data(
    file: &mut File,
    output_path: &Path,
    source: &MappedSafetensors,
    plan: &OutputLayoutPlan,
) -> Result<(), WriterError> {
    let mut cursor = 0_u64;
    let mut scales = BTreeMap::new();

    for tensor in &plan.tensors {
        if tensor.data_offsets.start != cursor {
            return Err(WriterError::PlanOffsetMismatch {
                name: tensor.name.clone(),
                expected_start: cursor,
                actual_start: tensor.data_offsets.start,
            });
        }
        let planned_len = tensor
            .data_offsets
            .end
            .checked_sub(tensor.data_offsets.start)
            .ok_or_else(|| WriterError::DataLengthMismatch {
                name: tensor.name.clone(),
                expected: tensor.byte_len,
                actual: 0,
            })?;
        if planned_len != tensor.byte_len {
            return Err(WriterError::DataLengthMismatch {
                name: tensor.name.clone(),
                expected: tensor.byte_len,
                actual: planned_len,
            });
        }

        match tensor.role {
            OutputTensorRole::Preserved => {
                let bytes = source
                    .tensor_bytes(&tensor.source_name)
                    .map_err(|source| WriterError::Source { source })?;
                write_payload(file, output_path, tensor, bytes)?;
            }
            OutputTensorRole::QuantizedData => {
                let view = source
                    .tensor(&tensor.source_name)
                    .map_err(|source| WriterError::Source { source })?;
                let mut actual = 0_u64;
                let stream_result = quantize_replay_chunks(
                    || view.values(),
                    DEFAULT_CHUNK_ELEMENTS,
                    |chunk| {
                        let chunk_bytes = u64::try_from(chunk.len()).map_err(|_| {
                            WriterError::DataLengthMismatch {
                                name: tensor.name.clone(),
                                expected: tensor.byte_len,
                                actual: u64::MAX,
                            }
                        })?;
                        actual = actual.checked_add(chunk_bytes).ok_or_else(|| {
                            WriterError::DataLengthMismatch {
                                name: tensor.name.clone(),
                                expected: tensor.byte_len,
                                actual: u64::MAX,
                            }
                        })?;
                        write_i8_values(file, output_path, chunk)
                    },
                );
                let scale = match stream_result {
                    Ok(scale) => scale,
                    Err(QuantizationStreamError::Quantization(source)) => {
                        return Err(WriterError::Quantization {
                            tensor_name: tensor.source_name.clone(),
                            source,
                        });
                    }
                    Err(QuantizationStreamError::Callback(error)) => return Err(error),
                };
                if actual != tensor.byte_len {
                    return Err(WriterError::DataLengthMismatch {
                        name: tensor.name.clone(),
                        expected: tensor.byte_len,
                        actual,
                    });
                }
                scales.insert(tensor.source_name.clone(), scale);
            }
            OutputTensorRole::QuantizationScale => {
                let scale = scales.remove(&tensor.source_name).ok_or_else(|| {
                    WriterError::MissingScale {
                        tensor_name: tensor.source_name.clone(),
                    }
                })?;
                if tensor.byte_len != SCALE_BYTE_LEN {
                    return Err(WriterError::DataLengthMismatch {
                        name: tensor.name.clone(),
                        expected: tensor.byte_len,
                        actual: SCALE_BYTE_LEN,
                    });
                }
                file.write_all(&scale.to_le_bytes())
                    .map_err(|source| io_error(output_path, source))?;
            }
        }
        cursor = tensor.data_offsets.end;
    }

    if cursor != plan.total_data_bytes {
        return Err(WriterError::DataLengthMismatch {
            name: "<data section>".to_owned(),
            expected: plan.total_data_bytes,
            actual: cursor,
        });
    }
    Ok(())
}

fn write_payload(
    file: &mut File,
    output_path: &Path,
    tensor: &crate::io::layout::OutputTensorPlan,
    bytes: &[u8],
) -> Result<(), WriterError> {
    let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if actual != tensor.byte_len {
        return Err(WriterError::DataLengthMismatch {
            name: tensor.name.clone(),
            expected: tensor.byte_len,
            actual,
        });
    }
    file.write_all(bytes)
        .map_err(|source| io_error(output_path, source))
}

fn write_i8_values(file: &mut File, output_path: &Path, values: &[i8]) -> Result<(), WriterError> {
    const CHUNK_SIZE: usize = 8192;
    let mut bytes = [0_u8; CHUNK_SIZE];
    for chunk in values.chunks(CHUNK_SIZE) {
        for (byte, &value) in bytes.iter_mut().zip(chunk) {
            *byte = value as u8;
        }
        file.write_all(&bytes[..chunk.len()])
            .map_err(|source| io_error(output_path, source))?;
    }
    Ok(())
}

fn json_object(entries: BTreeMap<String, Value>) -> Value {
    Value::Object(entries.into_iter().collect())
}

fn json_string_map(entries: BTreeMap<String, String>) -> Value {
    let values = entries
        .into_iter()
        .map(|(key, value)| (key, Value::String(value)))
        .collect();
    Value::Object(values)
}

fn json_shape(shape: &[usize]) -> Result<Value, WriterError> {
    serde_json::to_value(shape).map_err(|source| WriterError::Serialization { source })
}

fn create_temporary_file(destination: &Path) -> Result<(PathBuf, File), WriterError> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let file_name = destination
        .file_name()
        .ok_or_else(|| WriterError::InvalidDestination {
            path: destination.to_owned(),
        })?
        .to_string_lossy();

    for _ in 0..32 {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let temporary_path = parent.join(format!(".{file_name}.modelq-{}-{id}.tmp", process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(io_error(destination, source)),
        }
    }

    Err(io_error(
        destination,
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a unique temporary output path",
        ),
    ))
}

fn io_error(path: &Path, source: io::Error) -> WriterError {
    WriterError::Io {
        path: path.to_owned(),
        source,
    }
}

fn paths_refer_to_same_file(source: &Path, destination: &Path) -> bool {
    if source == destination {
        return true;
    }
    match (fs::canonicalize(source), fs::canonicalize(destination)) {
        (Ok(source), Ok(destination)) => source == destination,
        _ => false,
    }
}

struct TemporaryOutput {
    path: PathBuf,
    committed: bool,
}

impl TemporaryOutput {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            committed: false,
        }
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.path);
        }
    }
}
