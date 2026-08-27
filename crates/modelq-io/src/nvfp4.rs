//! ModelQ-native NVFP4 SafeTensors planning and writing.
//!
//! The planner and writer follow ADR 0011.  They deliberately remain a
//! library-only path: the existing CLI still writes INT8, and no runtime
//! compatibility is implied by this native convention.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    ops::Range,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use serde_json::{Map, Value};

use crate::safetensors::{MappedSafetensors, SafetensorsError, TensorSummary};
use modelq_quant::nvfp4::{self, Nvfp4Error};

const U8_DTYPE: &str = "U8";
const F32_DTYPE: &str = "F32";
const RESERVED_METADATA_NAME: &str = "__metadata__";
const GLOBAL_SCALE_BYTE_LEN: u64 = 4;
const HEADER_LENGTH_BYTES: usize = 8;
const HEADER_ALIGNMENT: usize = 8;
const MAX_HEADER_SIZE: usize = 100_000_000;
const MODELQ_FORMAT: &str = "modelq-native";
const MODELQ_FORMAT_VERSION: &str = "1";
const MODELQ_COMPATIBILITY_LEVEL: &str = "container-valid";
const MODELQ_QUANTIZATION: &str = "nvfp4";
const MODELQ_SCHEME: &str = "weight-only-blockwise";
const MODELQ_ALGORITHM: &str = "e2m1-e4m3-global-v0";
const MODELQ_ELEMENT_FORMAT: &str = "fp4-e2m1";
const MODELQ_BLOCK_SCALE_FORMAT: &str = "fp8-e4m3";
const MODELQ_GLOBAL_SCALE_DTYPE: &str = "F32";
const MODELQ_GLOBAL_SCALE_SEMANTICS: &str = "decode";
const MODELQ_BLOCK_SIZE: &str = "16";
const MODELQ_PACKING: &str = "e2m1-low-nibble-first";
const MODELQ_ROUNDING: &str = "nearest-even";
const MANIFEST_SCHEMA: &str = "modelq.nvfp4.manifest.v1";

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// The role of one tensor in a native NVFP4 output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nvfp4OutputRole {
    /// An input tensor copied to the output without quantization.
    Preserved,
    /// Packed E2M1 values, two values per U8 byte.
    QuantizedData,
    /// Raw FP8 E4M3 bit patterns, one byte per 16-value block.
    BlockScales,
    /// One F32 decode scale for the source tensor.
    GlobalScale,
}

/// One output tensor's metadata and contiguous data-region range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nvfp4OutputTensorPlan {
    /// Output tensor name.
    pub name: String,
    /// Original source tensor represented by this output tensor.
    pub source_name: String,
    /// Output SafeTensors dtype name.
    pub dtype: String,
    /// Output tensor shape.
    pub shape: Vec<usize>,
    /// Number of payload bytes reserved for this tensor.
    pub byte_len: u64,
    /// Half-open range relative to the SafeTensors data section.
    pub data_offsets: Range<u64>,
    /// Why this output tensor exists.
    pub role: Nvfp4OutputRole,
}

/// A complete native NVFP4 output data-region plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nvfp4OutputPlan {
    /// Output tensors in deterministic source/name order.
    pub tensors: Vec<Nvfp4OutputTensorPlan>,
    /// Total bytes required by the output data section.
    pub total_data_bytes: u64,
    quantized_names: Vec<String>,
}

impl Nvfp4OutputPlan {
    /// Finds an output tensor by its exact output name.
    pub fn tensor(&self, name: &str) -> Option<&Nvfp4OutputTensorPlan> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Returns the source names selected for NVFP4 quantization.
    pub fn quantized_source_names(&self) -> &[String] {
        &self.quantized_names
    }
}

/// One original tensor reconstructed from a native NVFP4 companion trio.
#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4DecodedTensor {
    /// Original source tensor name from the manifest.
    pub name: String,
    /// Original source SafeTensors dtype recorded by the writer.
    pub original_dtype: String,
    /// Original source tensor shape recorded by the writer.
    pub original_shape: Vec<usize>,
    /// Reconstructed values in the scalar F32 reference domain.
    pub values: Vec<f32>,
}

/// Errors returned when source metadata and an NVFP4 selection cannot form a
/// safe output layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Nvfp4LayoutError {
    /// A source tensor name appears more than once.
    DuplicateSourceName { name: String },
    /// A selected source name appears more than once.
    DuplicateSelectedName { name: String },
    /// A selected source name is not present in the source set.
    SelectedNameNotFound { name: String },
    /// The source tensor name is reserved by SafeTensors.
    ReservedSourceName { name: String },
    /// A generated or preserved output name would be used more than once.
    OutputNameCollision { name: String },
    /// The selected source dtype is outside the scalar NVFP4 input path.
    UnsupportedQuantizedDtype { name: String, dtype: String },
    /// The selected source shape violates the 16-value final-dimension rule.
    InvalidShape { name: String, shape: Vec<usize> },
    /// Multiplying shape dimensions overflowed `u64`.
    ShapeElementCountOverflow { name: String, shape: Vec<usize> },
    /// Advancing the output data cursor overflowed `u64`.
    OutputByteLengthOverflow {
        name: String,
        offset: u64,
        byte_len: u64,
    },
}

impl fmt::Display for Nvfp4LayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSourceName { name } => {
                write!(
                    formatter,
                    "source tensor name {name:?} appears more than once"
                )
            }
            Self::DuplicateSelectedName { name } => write!(
                formatter,
                "NVFP4 selection contains source tensor {name:?} more than once"
            ),
            Self::SelectedNameNotFound { name } => write!(
                formatter,
                "NVFP4 selection names source tensor {name:?}, which is not present"
            ),
            Self::ReservedSourceName { name } => write!(
                formatter,
                "source tensor name {name:?} is reserved by SafeTensors"
            ),
            Self::OutputNameCollision { name } => {
                write!(
                    formatter,
                    "output tensor name {name:?} would be used more than once"
                )
            }
            Self::UnsupportedQuantizedDtype { name, dtype } => write!(
                formatter,
                "tensor {name:?} uses unsupported NVFP4 source dtype {dtype:?}"
            ),
            Self::InvalidShape { name, shape } => write!(
                formatter,
                "NVFP4 tensor {name:?} shape {shape:?} must have positive dimensions and a final dimension divisible by {}",
                nvfp4::BLOCK_SIZE
            ),
            Self::ShapeElementCountOverflow { name, shape } => {
                write!(
                    formatter,
                    "shape {shape:?} for tensor {name:?} overflows its element count"
                )
            }
            Self::OutputByteLengthOverflow {
                name,
                offset,
                byte_len,
            } => write!(
                formatter,
                "placing output tensor {name:?} at {offset} with {byte_len} bytes overflows the data region"
            ),
        }
    }
}

impl std::error::Error for Nvfp4LayoutError {}

/// Errors returned while validating and decoding a native NVFP4 SafeTensors
/// file.
#[derive(Debug)]
pub enum Nvfp4ReaderError {
    /// A mapped source read failed.
    Source { source: SafetensorsError },
    /// A required file-level metadata entry is absent.
    MissingMetadata { key: String },
    /// A file-level metadata entry has an unexpected value.
    MetadataMismatch {
        key: String,
        expected: String,
        actual: String,
    },
    /// The manifest string was not valid JSON.
    ManifestJson { source: serde_json::Error },
    /// The manifest did not follow the native NVFP4 schema.
    InvalidManifest { message: String },
    /// A manifest-referenced tensor is absent from the SafeTensors file.
    MissingTensor { name: String },
    /// The SafeTensors file contains a tensor not described by the manifest.
    UnexpectedTensor { name: String },
    /// A companion descriptor disagrees with the manifest-derived layout.
    TensorMismatch {
        name: String,
        field: String,
        expected: String,
        actual: String,
    },
    /// The manifest claims an unsupported original source dtype.
    UnsupportedOriginalDtype { name: String, dtype: String },
    /// The scalar NVFP4 representation failed validation or reconstruction.
    Quantization {
        tensor_name: String,
        source: Nvfp4Error,
    },
}

impl fmt::Display for Nvfp4ReaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source { source } => write!(formatter, "could not read NVFP4 source: {source}"),
            Self::MissingMetadata { key } => {
                write!(formatter, "NVFP4 file metadata is missing required key {key:?}")
            }
            Self::MetadataMismatch {
                key,
                expected,
                actual,
            } => write!(
                formatter,
                "NVFP4 file metadata {key:?} is {actual:?}, expected {expected:?}"
            ),
            Self::ManifestJson { source } => {
                write!(formatter, "NVFP4 manifest is not valid JSON: {source}")
            }
            Self::InvalidManifest { message } => {
                write!(formatter, "invalid NVFP4 manifest: {message}")
            }
            Self::MissingTensor { name } => {
                write!(formatter, "NVFP4 manifest references missing tensor {name:?}")
            }
            Self::UnexpectedTensor { name } => {
                write!(formatter, "SafeTensors tensor {name:?} is not described by NVFP4 manifest")
            }
            Self::TensorMismatch {
                name,
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "NVFP4 tensor {name:?} field {field:?} is {actual:?}, expected {expected:?}"
            ),
            Self::UnsupportedOriginalDtype { name, dtype } => write!(
                formatter,
                "NVFP4 tensor {name:?} has unsupported original dtype {dtype:?}"
            ),
            Self::Quantization {
                tensor_name,
                source,
            } => write!(formatter, "could not decode NVFP4 tensor {tensor_name:?}: {source}"),
        }
    }
}

impl std::error::Error for Nvfp4ReaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source { source } => Some(source),
            Self::ManifestJson { source } => Some(source),
            Self::Quantization { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Plans all native NVFP4 output tensors and contiguous data offsets.
///
/// The caller explicitly selects source names for NVFP4.  Every other source
/// is preserved under its original name.  Sources and selections may arrive
/// in any order; the returned plan is deterministic by source name.
pub fn plan_nvfp4_output(
    sources: &[TensorSummary],
    quantized_names: &[String],
) -> Result<Nvfp4OutputPlan, Nvfp4LayoutError> {
    let mut sorted_sources = sources.iter().collect::<Vec<_>>();
    sorted_sources.sort_by(|left, right| left.name.cmp(&right.name));
    for pair in sorted_sources.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(Nvfp4LayoutError::DuplicateSourceName {
                name: pair[0].name.clone(),
            });
        }
    }

    let source_names = sorted_sources
        .iter()
        .map(|source| source.name.clone())
        .collect::<BTreeSet<_>>();
    if let Some(name) = source_names
        .iter()
        .find(|name| name.as_str() == RESERVED_METADATA_NAME)
    {
        return Err(Nvfp4LayoutError::ReservedSourceName { name: name.clone() });
    }

    let mut selected_names = BTreeSet::new();
    for name in quantized_names {
        if !selected_names.insert(name.clone()) {
            return Err(Nvfp4LayoutError::DuplicateSelectedName { name: name.clone() });
        }
        if !source_names.contains(name) {
            return Err(Nvfp4LayoutError::SelectedNameNotFound { name: name.clone() });
        }
    }

    let mut output_names = BTreeSet::new();
    for source in &sorted_sources {
        if !selected_names.contains(&source.name) {
            continue;
        }
        for suffix in [".qdata", ".block_scale", ".global_scale"] {
            let generated = format!("{}{}", source.name, suffix);
            if source_names.contains(&generated) || generated == RESERVED_METADATA_NAME {
                return Err(Nvfp4LayoutError::OutputNameCollision { name: generated });
            }
        }
    }
    let mut output_tensors = Vec::new();
    let mut cursor = 0_u64;

    for source in sorted_sources {
        if !selected_names.contains(&source.name) {
            append_tensor(
                &mut output_tensors,
                &mut output_names,
                &mut cursor,
                PendingTensor {
                    source_name: source.name.clone(),
                    name: source.name.clone(),
                    dtype: source.dtype.clone(),
                    shape: source.shape.clone(),
                    byte_len: source.byte_len,
                    role: Nvfp4OutputRole::Preserved,
                },
            )?;
            continue;
        }

        if !is_supported_source_dtype(&source.dtype) {
            return Err(Nvfp4LayoutError::UnsupportedQuantizedDtype {
                name: source.name.clone(),
                dtype: source.dtype.clone(),
            });
        }
        validate_quantized_shape(source)?;
        let element_count = checked_element_count(source)?;
        let qdata_shape = packed_shape(&source.shape);
        let block_scale_shape = block_scale_shape(&source.shape);

        append_tensor(
            &mut output_tensors,
            &mut output_names,
            &mut cursor,
            PendingTensor {
                source_name: source.name.clone(),
                name: format!("{}.qdata", source.name),
                dtype: U8_DTYPE.to_owned(),
                shape: qdata_shape,
                byte_len: element_count / 2,
                role: Nvfp4OutputRole::QuantizedData,
            },
        )?;
        append_tensor(
            &mut output_tensors,
            &mut output_names,
            &mut cursor,
            PendingTensor {
                source_name: source.name.clone(),
                name: format!("{}.block_scale", source.name),
                dtype: U8_DTYPE.to_owned(),
                shape: block_scale_shape,
                byte_len: element_count / u64::try_from(nvfp4::BLOCK_SIZE).unwrap_or(u64::MAX),
                role: Nvfp4OutputRole::BlockScales,
            },
        )?;
        append_tensor(
            &mut output_tensors,
            &mut output_names,
            &mut cursor,
            PendingTensor {
                source_name: source.name.clone(),
                name: format!("{}.global_scale", source.name),
                dtype: F32_DTYPE.to_owned(),
                shape: Vec::new(),
                byte_len: GLOBAL_SCALE_BYTE_LEN,
                role: Nvfp4OutputRole::GlobalScale,
            },
        )?;
    }

    Ok(Nvfp4OutputPlan {
        tensors: output_tensors,
        total_data_bytes: cursor,
        quantized_names: selected_names.into_iter().collect(),
    })
}

fn is_supported_source_dtype(dtype: &str) -> bool {
    matches!(dtype, "F32" | "F16" | "BF16")
}

fn validate_quantized_shape(source: &TensorSummary) -> Result<(), Nvfp4LayoutError> {
    if source.shape.is_empty()
        || source.shape.contains(&0)
        || !source
            .shape
            .last()
            .is_some_and(|&dimension| dimension % nvfp4::BLOCK_SIZE == 0)
    {
        return Err(Nvfp4LayoutError::InvalidShape {
            name: source.name.clone(),
            shape: source.shape.clone(),
        });
    }
    Ok(())
}

fn checked_element_count(source: &TensorSummary) -> Result<u64, Nvfp4LayoutError> {
    source
        .shape
        .iter()
        .try_fold(1_u64, |count, &dimension| {
            count.checked_mul(u64::try_from(dimension).ok()?)
        })
        .ok_or_else(|| Nvfp4LayoutError::ShapeElementCountOverflow {
            name: source.name.clone(),
            shape: source.shape.clone(),
        })
}

fn packed_shape(shape: &[usize]) -> Vec<usize> {
    let mut packed = shape.to_vec();
    let last = packed
        .last_mut()
        .expect("quantized shape validation guarantees a final dimension");
    *last /= 2;
    packed
}

fn block_scale_shape(shape: &[usize]) -> Vec<usize> {
    let mut block = shape.to_vec();
    let last = block
        .last_mut()
        .expect("quantized shape validation guarantees a final dimension");
    *last /= nvfp4::BLOCK_SIZE;
    block
}

struct PendingTensor {
    source_name: String,
    name: String,
    dtype: String,
    shape: Vec<usize>,
    byte_len: u64,
    role: Nvfp4OutputRole,
}

fn append_tensor(
    output_tensors: &mut Vec<Nvfp4OutputTensorPlan>,
    output_names: &mut BTreeSet<String>,
    cursor: &mut u64,
    pending: PendingTensor,
) -> Result<(), Nvfp4LayoutError> {
    if !output_names.insert(pending.name.clone()) {
        return Err(Nvfp4LayoutError::OutputNameCollision { name: pending.name });
    }
    let end = cursor.checked_add(pending.byte_len).ok_or_else(|| {
        Nvfp4LayoutError::OutputByteLengthOverflow {
            name: pending.name.clone(),
            offset: *cursor,
            byte_len: pending.byte_len,
        }
    })?;
    output_tensors.push(Nvfp4OutputTensorPlan {
        name: pending.name,
        source_name: pending.source_name,
        dtype: pending.dtype,
        shape: pending.shape,
        byte_len: pending.byte_len,
        data_offsets: *cursor..end,
        role: pending.role,
    });
    *cursor = end;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Nvfp4LayoutError, Nvfp4OutputRole, TensorSummary, plan_nvfp4_output};

    fn summary(name: &str, dtype: &str, shape: Vec<usize>, byte_len: u64) -> TensorSummary {
        TensorSummary {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape,
            byte_len,
        }
    }

    #[test]
    fn plans_deterministic_companions_and_preserved_sources() {
        let sources = [
            summary("weight", "F32", vec![2, 16], 128),
            summary("norm", "F32", vec![16], 64),
        ];
        let selected = ["weight".to_owned()];

        let plan = plan_nvfp4_output(&sources, &selected).expect("the source shape is valid");

        assert_eq!(
            plan.tensors
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            [
                "norm",
                "weight.qdata",
                "weight.block_scale",
                "weight.global_scale"
            ]
        );
        assert_eq!(plan.tensors[0].role, Nvfp4OutputRole::Preserved);
        assert_eq!(plan.tensors[0].data_offsets, 0..64);
        assert_eq!(plan.tensors[1].data_offsets, 64..80);
        assert_eq!(plan.tensors[2].data_offsets, 80..82);
        assert_eq!(plan.tensors[3].data_offsets, 82..86);
        assert_eq!(plan.total_data_bytes, 86);
    }

    #[test]
    fn rejects_invalid_final_dimension() {
        let error = plan_nvfp4_output(
            &[summary("weight", "F32", vec![2, 8], 64)],
            &["weight".to_owned()],
        )
        .expect_err("the final dimension must be block-aligned");

        assert!(matches!(error, Nvfp4LayoutError::InvalidShape { .. }));
    }

    #[test]
    fn rejects_empty_or_zero_shape() {
        for shape in [Vec::new(), vec![0, 16]] {
            let error = plan_nvfp4_output(
                &[summary("weight", "F32", shape, 0)],
                &["weight".to_owned()],
            )
            .expect_err("NVFP4 cannot represent an empty or zero-sized source");

            assert!(matches!(error, Nvfp4LayoutError::InvalidShape { .. }));
        }
    }

    #[test]
    fn rejects_unsupported_selected_dtype() {
        for dtype in ["F64", "U8"] {
            let error = plan_nvfp4_output(
                &[summary("weight", dtype, vec![1, 16], 64)],
                &["weight".to_owned()],
            )
            .expect_err("only F32, F16, and BF16 are supported source views");

            assert!(matches!(
                error,
                Nvfp4LayoutError::UnsupportedQuantizedDtype { .. }
            ));
        }
    }

    #[test]
    fn rejects_missing_and_duplicate_selection_names() {
        let source = [summary("weight", "F32", vec![1, 16], 64)];
        let missing = plan_nvfp4_output(&source, &["missing".to_owned()])
            .expect_err("a selection must refer to a source tensor");
        assert!(matches!(
            missing,
            Nvfp4LayoutError::SelectedNameNotFound { .. }
        ));

        let duplicate = plan_nvfp4_output(&source, &["weight".to_owned(), "weight".to_owned()])
            .expect_err("a source can be selected only once");
        assert!(matches!(
            duplicate,
            Nvfp4LayoutError::DuplicateSelectedName { .. }
        ));
    }

    #[test]
    fn rejects_reserved_and_duplicate_source_names() {
        let reserved = plan_nvfp4_output(&[summary("__metadata__", "U8", vec![1], 1)], &[])
            .expect_err("the SafeTensors metadata key is reserved");
        assert!(matches!(
            reserved,
            Nvfp4LayoutError::ReservedSourceName { .. }
        ));

        let duplicate = plan_nvfp4_output(
            &[
                summary("weight", "F32", vec![1, 16], 64),
                summary("weight", "F32", vec![1, 16], 64),
            ],
            &[],
        )
        .expect_err("source names must be unique");
        assert!(matches!(
            duplicate,
            Nvfp4LayoutError::DuplicateSourceName { .. }
        ));
    }

    #[test]
    fn rejects_generated_name_collision_with_source() {
        let error = plan_nvfp4_output(
            &[
                summary("weight", "F32", vec![1, 16], 64),
                summary("weight.qdata", "U8", vec![8], 8),
            ],
            &["weight".to_owned()],
        )
        .expect_err("generated companion names cannot shadow source tensors");

        assert!(matches!(
            error,
            Nvfp4LayoutError::OutputNameCollision { .. }
        ));
    }

    #[test]
    fn rejects_shape_product_overflow() {
        let error = plan_nvfp4_output(
            &[summary("weight", "F32", vec![usize::MAX, 2, 16], 0)],
            &["weight".to_owned()],
        )
        .expect_err("the shape product must be checked before byte planning");

        assert!(matches!(
            error,
            Nvfp4LayoutError::ShapeElementCountOverflow { .. }
        ));
    }

    #[test]
    fn rejects_output_byte_length_overflow() {
        let error = plan_nvfp4_output(
            &[
                summary("a", "U8", vec![1], u64::MAX),
                summary("weight", "F32", vec![1, 16], 64),
            ],
            &["weight".to_owned()],
        )
        .expect_err("the output cursor must be checked for overflow");

        assert!(matches!(
            error,
            Nvfp4LayoutError::OutputByteLengthOverflow { .. }
        ));
    }
}

/// Reads and decodes a ModelQ-native NVFP4 SafeTensors file.
///
/// Every tensor described by the manifest is validated.  Quantized records
/// are returned as reconstructed F32 values; preserved records are validated
/// and remain available byte-for-byte through [`MappedSafetensors`].
pub fn read_nvfp4_safetensors(
    source: &MappedSafetensors,
) -> Result<Vec<Nvfp4DecodedTensor>, Nvfp4ReaderError> {
    let metadata = source.metadata();
    for &(key, expected) in &[
        ("modelq.format", MODELQ_FORMAT),
        ("modelq.format_version", MODELQ_FORMAT_VERSION),
        ("modelq.compatibility_level", MODELQ_COMPATIBILITY_LEVEL),
        ("modelq.quantization", MODELQ_QUANTIZATION),
        ("modelq.scheme", MODELQ_SCHEME),
        ("modelq.algorithm", MODELQ_ALGORITHM),
        ("modelq.element_format", MODELQ_ELEMENT_FORMAT),
        ("modelq.block_scale_format", MODELQ_BLOCK_SCALE_FORMAT),
        ("modelq.global_scale_dtype", MODELQ_GLOBAL_SCALE_DTYPE),
        ("modelq.global_scale_semantics", MODELQ_GLOBAL_SCALE_SEMANTICS),
        ("modelq.block_size", MODELQ_BLOCK_SIZE),
        ("modelq.packing", MODELQ_PACKING),
        ("modelq.rounding", MODELQ_ROUNDING),
    ] {
        require_file_metadata(metadata, key, expected)?;
    }

    let manifest_text = metadata
        .get("modelq.manifest")
        .ok_or_else(|| Nvfp4ReaderError::MissingMetadata {
            key: "modelq.manifest".to_owned(),
        })?;
    let manifest: Value =
        serde_json::from_str(manifest_text).map_err(|source| Nvfp4ReaderError::ManifestJson {
            source,
        })?;
    let manifest_object = manifest
        .as_object()
        .ok_or_else(|| invalid_manifest("the manifest root must be a JSON object"))?;
    let schema = required_manifest_string(manifest_object, "schema", "manifest")?;
    if schema != MANIFEST_SCHEMA {
        return Err(invalid_manifest(format!(
            "schema {schema:?} is not {MANIFEST_SCHEMA:?}"
        )));
    }
    let records = manifest_object
        .get("tensors")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_manifest("manifest field \"tensors\" must be an object"))?;

    let summaries = source.inspection().tensors;
    let summaries_by_name = summaries
        .iter()
        .map(|summary| (summary.name.clone(), summary))
        .collect::<BTreeMap<_, _>>();
    let mut expected_output_names = BTreeSet::new();
    let mut decoded = Vec::new();

    for (source_name, value) in records {
        let record = value.as_object().ok_or_else(|| {
            invalid_manifest(format!("tensor {source_name:?} record must be an object"))
        })?;
        let action = required_manifest_string(record, "action", source_name)?;
        match action.as_str() {
            "preserved" => {
                read_preserved_record(
                    source,
                    &summaries_by_name,
                    &mut expected_output_names,
                    source_name,
                    record,
                )?;
            }
            "quantized" => {
                let tensor = read_quantized_record(
                    source,
                    &summaries_by_name,
                    &mut expected_output_names,
                    source_name,
                    record,
                )?;
                decoded.push(tensor);
            }
            other => {
                return Err(invalid_manifest(format!(
                    "tensor {source_name:?} has unknown action {other:?}"
                )));
            }
        }
    }

    for summary in &summaries {
        if !expected_output_names.contains(&summary.name) {
            return Err(Nvfp4ReaderError::UnexpectedTensor {
                name: summary.name.clone(),
            });
        }
    }

    decoded.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(decoded)
}

fn require_file_metadata(
    metadata: &BTreeMap<String, String>,
    key: &str,
    expected: &str,
) -> Result<(), Nvfp4ReaderError> {
    let actual = metadata
        .get(key)
        .ok_or_else(|| Nvfp4ReaderError::MissingMetadata {
            key: key.to_owned(),
        })?;
    if actual != expected {
        return Err(Nvfp4ReaderError::MetadataMismatch {
            key: key.to_owned(),
            expected: expected.to_owned(),
            actual: actual.clone(),
        });
    }
    Ok(())
}

fn read_preserved_record(
    source: &MappedSafetensors,
    summaries_by_name: &BTreeMap<String, &TensorSummary>,
    expected_output_names: &mut BTreeSet<String>,
    source_name: &str,
    record: &Map<String, Value>,
) -> Result<(), Nvfp4ReaderError> {
    let tensor_name = required_manifest_string(record, "tensor_name", source_name)?;
    if tensor_name != source_name {
        return Err(invalid_manifest(format!(
            "preserved record {source_name:?} names tensor {tensor_name:?}"
        )));
    }
    let dtype = required_manifest_string(record, "original_dtype", source_name)?;
    let shape = required_manifest_shape(record, "original_shape", source_name)?;
    let summary = summaries_by_name
        .get(tensor_name.as_str())
        .copied()
        .ok_or_else(|| Nvfp4ReaderError::MissingTensor {
            name: tensor_name.clone(),
        })?;
    validate_summary_descriptor(summary, &tensor_name, &dtype, &shape)?;
    source
        .tensor_bytes(&tensor_name)
        .map_err(|source| Nvfp4ReaderError::Source { source })?;
    insert_expected_output_name(expected_output_names, tensor_name)
}

fn read_quantized_record(
    source: &MappedSafetensors,
    summaries_by_name: &BTreeMap<String, &TensorSummary>,
    expected_output_names: &mut BTreeSet<String>,
    source_name: &str,
    record: &Map<String, Value>,
) -> Result<Nvfp4DecodedTensor, Nvfp4ReaderError> {
    let original_dtype = required_manifest_string(record, "original_dtype", source_name)?;
    if !is_supported_source_dtype(&original_dtype) {
        return Err(Nvfp4ReaderError::UnsupportedOriginalDtype {
            name: source_name.to_owned(),
            dtype: original_dtype,
        });
    }
    let original_shape = required_manifest_shape(record, "original_shape", source_name)?;
    let element_count = manifest_element_count(source_name, &original_shape)?;

    let axis = required_manifest_i64(record, "axis", source_name)?;
    if axis != -1 {
        return Err(invalid_manifest(format!(
            "quantized record {source_name:?} axis is {axis}, expected -1"
        )));
    }
    let block_size = required_manifest_u64(record, "block_size", source_name)?;
    if block_size != u64::try_from(nvfp4::BLOCK_SIZE).unwrap_or(u64::MAX) {
        return Err(invalid_manifest(format!(
            "quantized record {source_name:?} block_size is {block_size}, expected {}",
            nvfp4::BLOCK_SIZE
        )));
    }

    let qdata_name = required_manifest_string(record, "qdata_name", source_name)?;
    let expected_qdata_name = format!("{source_name}.qdata");
    if qdata_name != expected_qdata_name {
        return Err(invalid_manifest(format!(
            "quantized record {source_name:?} names qdata {qdata_name:?}, expected {expected_qdata_name:?}"
        )));
    }
    let qdata_dtype = required_manifest_string(record, "qdata_dtype", source_name)?;
    require_manifest_string_value(&qdata_dtype, U8_DTYPE, "qdata_dtype", source_name)?;
    let qdata_shape = required_manifest_shape(record, "qdata_shape", source_name)?;
    let expected_qdata_shape = packed_shape(&original_shape);
    require_manifest_shape_value(
        &qdata_shape,
        &expected_qdata_shape,
        "qdata_shape",
        source_name,
    )?;
    let qdata_byte_len = u64::try_from(element_count / nvfp4::VALUES_PER_BYTE)
        .map_err(|_| invalid_manifest(format!("qdata byte length overflows for {source_name:?}")))?;
    let qdata_summary = require_companion_summary(
        summaries_by_name,
        &qdata_name,
        U8_DTYPE,
        &expected_qdata_shape,
        qdata_byte_len,
    )?;

    let block_scale_name =
        required_manifest_string(record, "block_scale_name", source_name)?;
    let expected_block_scale_name = format!("{source_name}.block_scale");
    if block_scale_name != expected_block_scale_name {
        return Err(invalid_manifest(format!(
            "quantized record {source_name:?} names block scales {block_scale_name:?}, expected {expected_block_scale_name:?}"
        )));
    }
    let block_scale_dtype =
        required_manifest_string(record, "block_scale_dtype", source_name)?;
    require_manifest_string_value(
        &block_scale_dtype,
        U8_DTYPE,
        "block_scale_dtype",
        source_name,
    )?;
    let manifest_block_scale_shape =
        required_manifest_shape(record, "block_scale_shape", source_name)?;
    let expected_block_scale_shape = block_scale_shape(&original_shape);
    require_manifest_shape_value(
        &manifest_block_scale_shape,
        &expected_block_scale_shape,
        "block_scale_shape",
        source_name,
    )?;
    let block_scale_byte_len = u64::try_from(element_count / nvfp4::BLOCK_SIZE)
        .map_err(|_| invalid_manifest(format!("block-scale byte length overflows for {source_name:?}")))?;
    let block_scale_summary = require_companion_summary(
        summaries_by_name,
        &block_scale_name,
        U8_DTYPE,
        &expected_block_scale_shape,
        block_scale_byte_len,
    )?;

    let global_scale_name =
        required_manifest_string(record, "global_scale_name", source_name)?;
    let expected_global_scale_name = format!("{source_name}.global_scale");
    if global_scale_name != expected_global_scale_name {
        return Err(invalid_manifest(format!(
            "quantized record {source_name:?} names global scale {global_scale_name:?}, expected {expected_global_scale_name:?}"
        )));
    }
    let global_scale_dtype =
        required_manifest_string(record, "global_scale_dtype", source_name)?;
    require_manifest_string_value(
        &global_scale_dtype,
        F32_DTYPE,
        "global_scale_dtype",
        source_name,
    )?;
    let global_scale_shape =
        required_manifest_shape(record, "global_scale_shape", source_name)?;
    let expected_global_scale_shape: &[usize] = &[];
    require_manifest_shape_value(
        &global_scale_shape,
        expected_global_scale_shape,
        "global_scale_shape",
        source_name,
    )?;
    let global_scale_summary = require_companion_summary(
        summaries_by_name,
        &global_scale_name,
        F32_DTYPE,
        expected_global_scale_shape,
        GLOBAL_SCALE_BYTE_LEN,
    )?;

    require_manifest_string_value(
        &required_manifest_string(record, "packing", source_name)?,
        "low-nibble-first",
        "packing",
        source_name,
    )?;
    require_manifest_string_value(
        &required_manifest_string(record, "block_scale_encoding", source_name)?,
        "e4m3-bit-pattern",
        "block_scale_encoding",
        source_name,
    )?;
    require_manifest_string_value(
        &required_manifest_string(record, "global_scale_semantics", source_name)?,
        MODELQ_GLOBAL_SCALE_SEMANTICS,
        "global_scale_semantics",
        source_name,
    )?;

    insert_expected_output_name(expected_output_names, qdata_name.clone())?;
    insert_expected_output_name(expected_output_names, block_scale_name.clone())?;
    insert_expected_output_name(expected_output_names, global_scale_name.clone())?;

    let qdata = source
        .tensor_bytes(&qdata_summary.name)
        .map_err(|source| Nvfp4ReaderError::Source { source })?;
    let block_scales = source
        .tensor_bytes(&block_scale_summary.name)
        .map_err(|source| Nvfp4ReaderError::Source { source })?;
    let global_scale_bytes = source
        .tensor_bytes(&global_scale_summary.name)
        .map_err(|source| Nvfp4ReaderError::Source { source })?;
    let global_scale = <[u8; 4]>::try_from(global_scale_bytes)
        .map(f32::from_le_bytes)
        .map_err(|_| invalid_manifest("global scale payload is not four bytes"))?;
    let quantized = nvfp4::QuantizedTensor::from_parts(
        qdata.to_vec(),
        block_scales.to_vec(),
        global_scale,
        element_count,
    )
    .map_err(|source| Nvfp4ReaderError::Quantization {
        tensor_name: source_name.to_owned(),
        source,
    })?;
    let values = quantized
        .dequantize()
        .map_err(|source| Nvfp4ReaderError::Quantization {
            tensor_name: source_name.to_owned(),
            source,
        })?;

    Ok(Nvfp4DecodedTensor {
        name: source_name.to_owned(),
        original_dtype,
        original_shape,
        values,
    })
}

fn require_companion_summary<'a>(
    summaries_by_name: &'a BTreeMap<String, &TensorSummary>,
    name: &str,
    dtype: &str,
    shape: &[usize],
    byte_len: u64,
) -> Result<&'a TensorSummary, Nvfp4ReaderError> {
    let summary = summaries_by_name
        .get(name)
        .copied()
        .ok_or_else(|| Nvfp4ReaderError::MissingTensor {
            name: name.to_owned(),
        })?;
    validate_summary_descriptor(summary, name, dtype, shape)?;
    if summary.byte_len != byte_len {
        return Err(Nvfp4ReaderError::TensorMismatch {
            name: name.to_owned(),
            field: "byte_len".to_owned(),
            expected: byte_len.to_string(),
            actual: summary.byte_len.to_string(),
        });
    }
    Ok(summary)
}

fn validate_summary_descriptor(
    summary: &TensorSummary,
    name: &str,
    dtype: &str,
    shape: &[usize],
) -> Result<(), Nvfp4ReaderError> {
    if summary.dtype != dtype {
        return Err(Nvfp4ReaderError::TensorMismatch {
            name: name.to_owned(),
            field: "dtype".to_owned(),
            expected: dtype.to_owned(),
            actual: summary.dtype.clone(),
        });
    }
    if summary.shape != shape {
        return Err(Nvfp4ReaderError::TensorMismatch {
            name: name.to_owned(),
            field: "shape".to_owned(),
            expected: format!("{shape:?}"),
            actual: format!("{:?}", summary.shape),
        });
    }
    Ok(())
}

fn insert_expected_output_name(
    expected_output_names: &mut BTreeSet<String>,
    name: String,
) -> Result<(), Nvfp4ReaderError> {
    if !expected_output_names.insert(name.clone()) {
        return Err(invalid_manifest(format!(
            "manifest maps more than one record to output tensor {name:?}"
        )));
    }
    Ok(())
}

fn required_manifest_string(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<String, Nvfp4ReaderError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            invalid_manifest(format!(
                "{context:?} field {key:?} must be a string"
            ))
        })
}

fn required_manifest_shape(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<Vec<usize>, Nvfp4ReaderError> {
    let values = object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| {
            invalid_manifest(format!("{context:?} field {key:?} must be an array"))
        })?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let dimension = value.as_u64().ok_or_else(|| {
                invalid_manifest(format!(
                    "{context:?} field {key:?} dimension {index} must be a non-negative integer"
                ))
            })?;
            usize::try_from(dimension).map_err(|_| {
                invalid_manifest(format!(
                    "{context:?} field {key:?} dimension {index} does not fit usize"
                ))
            })
        })
        .collect()
}

fn required_manifest_u64(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<u64, Nvfp4ReaderError> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid_manifest(format!("{context:?} field {key:?} must be an integer")))
}

fn required_manifest_i64(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<i64, Nvfp4ReaderError> {
    object
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid_manifest(format!("{context:?} field {key:?} must be an integer")))
}

fn require_manifest_string_value(
    actual: &str,
    expected: &str,
    field: &str,
    context: &str,
) -> Result<(), Nvfp4ReaderError> {
    if actual != expected {
        return Err(Nvfp4ReaderError::TensorMismatch {
            name: context.to_owned(),
            field: field.to_owned(),
            expected: expected.to_owned(),
            actual: actual.to_owned(),
        });
    }
    Ok(())
}

fn require_manifest_shape_value(
    actual: &[usize],
    expected: &[usize],
    field: &str,
    context: &str,
) -> Result<(), Nvfp4ReaderError> {
    if actual != expected {
        return Err(Nvfp4ReaderError::TensorMismatch {
            name: context.to_owned(),
            field: field.to_owned(),
            expected: format!("{expected:?}"),
            actual: format!("{actual:?}"),
        });
    }
    Ok(())
}

fn manifest_element_count(
    name: &str,
    shape: &[usize],
) -> Result<usize, Nvfp4ReaderError> {
    if shape.is_empty()
        || shape.contains(&0)
        || !shape
            .last()
            .is_some_and(|&dimension| dimension % nvfp4::BLOCK_SIZE == 0)
    {
        return Err(invalid_manifest(format!(
            "quantized tensor {name:?} shape {shape:?} must have positive dimensions and a final dimension divisible by {}",
            nvfp4::BLOCK_SIZE
        )));
    }
    shape
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| {
            invalid_manifest(format!(
                "quantized tensor {name:?} shape {shape:?} overflows its element count"
            ))
        })
}

fn invalid_manifest(message: impl Into<String>) -> Nvfp4ReaderError {
    Nvfp4ReaderError::InvalidManifest {
        message: message.into(),
    }
}

/// Errors returned while creating a native NVFP4 SafeTensors output.
#[derive(Debug)]
pub enum Nvfp4WriterError {
    /// The destination already exists.
    DestinationExists { path: PathBuf },
    /// The source and destination resolve to the same existing path.
    SourceDestinationConflict {
        source: PathBuf,
        destination: PathBuf,
    },
    /// The destination does not name a file path.
    InvalidDestination { path: PathBuf },
    /// Source metadata and the supplied plan could not be reconciled.
    Layout { source: Nvfp4LayoutError },
    /// The supplied plan differs from the current source and selection.
    PlanMismatch,
    /// A mapped source read failed.
    Source { source: SafetensorsError },
    /// The scalar NVFP4 quantizer rejected one source tensor.
    Quantization {
        tensor_name: String,
        source: Nvfp4Error,
    },
    /// The deterministic header could not be serialized.
    Serialization { source: serde_json::Error },
    /// An output file operation failed.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
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
    /// A plan did not have contiguous data offsets.
    PlanOffsetMismatch {
        name: String,
        expected_start: u64,
        actual_start: u64,
    },
    /// A companion appeared without the selected source's packed payload.
    MissingQuantizedTensor { tensor_name: String },
}

impl fmt::Display for Nvfp4WriterError {
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
            Self::Layout { source } => write!(formatter, "invalid NVFP4 output layout: {source}"),
            Self::PlanMismatch => formatter.write_str(
                "the supplied NVFP4 output plan does not match the current source metadata",
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
            Self::MissingQuantizedTensor { tensor_name } => write!(
                formatter,
                "NVFP4 companion for source tensor {tensor_name:?} has no quantized payload"
            ),
        }
    }
}

impl std::error::Error for Nvfp4WriterError {
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

/// Writes a ModelQ-native NVFP4 SafeTensors file from a checked output plan.
///
/// The source mapping remains read-only.  The destination must not already
/// exist; data is written to a unique temporary file beside it and renamed
/// into place only after the complete header and data section have been
/// synchronized.  This is a library-only container writer: it makes no
/// runtime or hardware compatibility claim.
pub fn write_nvfp4_safetensors(
    source: &MappedSafetensors,
    plan: &Nvfp4OutputPlan,
    destination: impl AsRef<Path>,
) -> Result<(), Nvfp4WriterError> {
    let destination = destination.as_ref().to_owned();
    if destination.file_name().is_none() {
        return Err(Nvfp4WriterError::InvalidDestination { path: destination });
    }
    if paths_refer_to_same_file(source.path(), &destination) {
        return Err(Nvfp4WriterError::SourceDestinationConflict {
            source: source.path().to_owned(),
            destination,
        });
    }
    if destination.exists() {
        return Err(Nvfp4WriterError::DestinationExists { path: destination });
    }

    let inspection = source.inspection();
    let expected_plan = plan_nvfp4_output(&inspection.tensors, plan.quantized_source_names())
        .map_err(|source| Nvfp4WriterError::Layout { source })?;
    if expected_plan != *plan {
        return Err(Nvfp4WriterError::PlanMismatch);
    }

    let header = build_header(&inspection.tensors, plan, &destination)?;
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

    // On Unix, rename would replace a destination that appeared while the
    // temporary file was being produced.  Preserve that artifact instead.
    if destination.exists() {
        return Err(Nvfp4WriterError::DestinationExists { path: destination });
    }
    fs::rename(&temporary_path, &destination).map_err(|source| io_error(&destination, source))?;
    temporary.committed = true;
    Ok(())
}

fn build_header(
    summaries: &[TensorSummary],
    plan: &Nvfp4OutputPlan,
    destination: &Path,
) -> Result<Vec<u8>, Nvfp4WriterError> {
    let mut sorted_summaries = summaries.iter().collect::<Vec<_>>();
    sorted_summaries.sort_by(|left, right| left.name.cmp(&right.name));
    let selected_names = plan
        .quantized_source_names()
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();

    let mut manifest_tensors = BTreeMap::new();
    for summary in sorted_summaries {
        let mut record = BTreeMap::new();
        if selected_names.contains(&summary.name) {
            let qdata = plan
                .tensor(&format!("{}.qdata", summary.name))
                .ok_or(Nvfp4WriterError::PlanMismatch)?;
            let block_scale = plan
                .tensor(&format!("{}.block_scale", summary.name))
                .ok_or(Nvfp4WriterError::PlanMismatch)?;
            let global_scale = plan
                .tensor(&format!("{}.global_scale", summary.name))
                .ok_or(Nvfp4WriterError::PlanMismatch)?;

            record.insert("action".to_owned(), Value::String("quantized".to_owned()));
            record.insert(
                "original_dtype".to_owned(),
                Value::String(summary.dtype.clone()),
            );
            record.insert("original_shape".to_owned(), json_shape(&summary.shape)?);
            record.insert("axis".to_owned(), Value::from(-1_i64));
            record.insert(
                "block_size".to_owned(),
                Value::from(u64::try_from(nvfp4::BLOCK_SIZE).unwrap_or(u64::MAX)),
            );
            record.insert("qdata_name".to_owned(), Value::String(qdata.name.clone()));
            record.insert("qdata_dtype".to_owned(), Value::String(qdata.dtype.clone()));
            record.insert("qdata_shape".to_owned(), json_shape(&qdata.shape)?);
            record.insert(
                "block_scale_name".to_owned(),
                Value::String(block_scale.name.clone()),
            );
            record.insert(
                "block_scale_dtype".to_owned(),
                Value::String(block_scale.dtype.clone()),
            );
            record.insert(
                "block_scale_shape".to_owned(),
                json_shape(&block_scale.shape)?,
            );
            record.insert(
                "global_scale_name".to_owned(),
                Value::String(global_scale.name.clone()),
            );
            record.insert(
                "global_scale_dtype".to_owned(),
                Value::String(global_scale.dtype.clone()),
            );
            record.insert(
                "global_scale_shape".to_owned(),
                json_shape(&global_scale.shape)?,
            );
            record.insert(
                "packing".to_owned(),
                Value::String("low-nibble-first".to_owned()),
            );
            record.insert(
                "block_scale_encoding".to_owned(),
                Value::String("e4m3-bit-pattern".to_owned()),
            );
            record.insert(
                "global_scale_semantics".to_owned(),
                Value::String(MODELQ_GLOBAL_SCALE_SEMANTICS.to_owned()),
            );
        } else {
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
        manifest_tensors.insert(summary.name.clone(), json_object(record));
    }

    let mut manifest = BTreeMap::new();
    manifest.insert(
        "schema".to_owned(),
        Value::String(MANIFEST_SCHEMA.to_owned()),
    );
    manifest.insert("tensors".to_owned(), json_object(manifest_tensors));
    let manifest_json = serde_json::to_string(&json_object(manifest))
        .map_err(|source| Nvfp4WriterError::Serialization { source })?;

    let mut metadata = BTreeMap::new();
    metadata.insert("modelq.algorithm".to_owned(), MODELQ_ALGORITHM.to_owned());
    metadata.insert(
        "modelq.block_scale_format".to_owned(),
        MODELQ_BLOCK_SCALE_FORMAT.to_owned(),
    );
    metadata.insert("modelq.block_size".to_owned(), MODELQ_BLOCK_SIZE.to_owned());
    metadata.insert(
        "modelq.compatibility_level".to_owned(),
        MODELQ_COMPATIBILITY_LEVEL.to_owned(),
    );
    metadata.insert(
        "modelq.element_format".to_owned(),
        MODELQ_ELEMENT_FORMAT.to_owned(),
    );
    metadata.insert("modelq.format".to_owned(), MODELQ_FORMAT.to_owned());
    metadata.insert(
        "modelq.format_version".to_owned(),
        MODELQ_FORMAT_VERSION.to_owned(),
    );
    metadata.insert(
        "modelq.global_scale_dtype".to_owned(),
        MODELQ_GLOBAL_SCALE_DTYPE.to_owned(),
    );
    metadata.insert(
        "modelq.global_scale_semantics".to_owned(),
        MODELQ_GLOBAL_SCALE_SEMANTICS.to_owned(),
    );
    metadata.insert("modelq.manifest".to_owned(), manifest_json);
    metadata.insert("modelq.packing".to_owned(), MODELQ_PACKING.to_owned());
    metadata.insert(
        "modelq.quantization".to_owned(),
        MODELQ_QUANTIZATION.to_owned(),
    );
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
        serde_json::to_vec(&root).map_err(|source| Nvfp4WriterError::Serialization { source })?;
    let padded_len = raw_header
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
    let total_header_len = HEADER_LENGTH_BYTES
        .checked_add(padded_len)
        .ok_or(Nvfp4WriterError::HeaderLengthOverflow)?;
    let padded_len_u64 =
        u64::try_from(padded_len).map_err(|_| Nvfp4WriterError::HeaderLengthOverflow)?;
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
    plan: &Nvfp4OutputPlan,
) -> Result<(), Nvfp4WriterError> {
    let mut cursor = 0_u64;
    let mut quantized_tensors = BTreeMap::new();

    for tensor in &plan.tensors {
        if tensor.data_offsets.start != cursor {
            return Err(Nvfp4WriterError::PlanOffsetMismatch {
                name: tensor.name.clone(),
                expected_start: cursor,
                actual_start: tensor.data_offsets.start,
            });
        }
        let planned_len = tensor
            .data_offsets
            .end
            .checked_sub(tensor.data_offsets.start)
            .ok_or_else(|| Nvfp4WriterError::DataLengthMismatch {
                name: tensor.name.clone(),
                expected: tensor.byte_len,
                actual: 0,
            })?;
        if planned_len != tensor.byte_len {
            return Err(Nvfp4WriterError::DataLengthMismatch {
                name: tensor.name.clone(),
                expected: tensor.byte_len,
                actual: planned_len,
            });
        }

        match tensor.role {
            Nvfp4OutputRole::Preserved => {
                let bytes = source
                    .tensor_bytes(&tensor.source_name)
                    .map_err(|source| Nvfp4WriterError::Source { source })?;
                write_payload(file, output_path, tensor, bytes)?;
            }
            Nvfp4OutputRole::QuantizedData => {
                let view = source
                    .tensor(&tensor.source_name)
                    .map_err(|source| Nvfp4WriterError::Source { source })?;
                let values = view.values().collect::<Vec<_>>();
                let quantized =
                    nvfp4::quantize_shaped(&values, view.shape()).map_err(|source| {
                        Nvfp4WriterError::Quantization {
                            tensor_name: tensor.source_name.clone(),
                            source,
                        }
                    })?;
                write_payload(file, output_path, tensor, quantized.packed_values())?;
                quantized_tensors.insert(tensor.source_name.clone(), quantized);
            }
            Nvfp4OutputRole::BlockScales => {
                let quantized = quantized_tensors.get(&tensor.source_name).ok_or_else(|| {
                    Nvfp4WriterError::MissingQuantizedTensor {
                        tensor_name: tensor.source_name.clone(),
                    }
                })?;
                write_payload(file, output_path, tensor, quantized.block_scales())?;
            }
            Nvfp4OutputRole::GlobalScale => {
                let quantized = quantized_tensors.get(&tensor.source_name).ok_or_else(|| {
                    Nvfp4WriterError::MissingQuantizedTensor {
                        tensor_name: tensor.source_name.clone(),
                    }
                })?;
                let bytes = quantized.global_scale().to_le_bytes();
                write_payload(file, output_path, tensor, &bytes)?;
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

fn write_payload(
    file: &mut File,
    output_path: &Path,
    tensor: &Nvfp4OutputTensorPlan,
    bytes: &[u8],
) -> Result<(), Nvfp4WriterError> {
    let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if actual != tensor.byte_len {
        return Err(Nvfp4WriterError::DataLengthMismatch {
            name: tensor.name.clone(),
            expected: tensor.byte_len,
            actual,
        });
    }
    file.write_all(bytes)
        .map_err(|source| io_error(output_path, source))
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

fn json_shape(shape: &[usize]) -> Result<Value, Nvfp4WriterError> {
    serde_json::to_value(shape).map_err(|source| Nvfp4WriterError::Serialization { source })
}

fn create_temporary_file(destination: &Path) -> Result<(PathBuf, File), Nvfp4WriterError> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let file_name = destination
        .file_name()
        .ok_or_else(|| Nvfp4WriterError::InvalidDestination {
            path: destination.to_owned(),
        })?
        .to_string_lossy();

    for _ in 0..32 {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let temporary_path = parent.join(format!(
            ".{file_name}.modelq-nvfp4-{}-{id}.tmp",
            process::id()
        ));
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

fn io_error(path: &Path, source: io::Error) -> Nvfp4WriterError {
    Nvfp4WriterError::Io {
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
