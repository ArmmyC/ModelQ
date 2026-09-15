//! CPU-only Transformer Engine NVFP4 rowwise export profile.
//!
//! This module adapts the validated ModelQ-native NVFP4 representation into
//! the three logical rowwise fields used by Transformer Engine's 1x16
//! quantization contract.  It intentionally performs no CUDA allocation,
//! scale swizzle, columnwise transpose, or container serialization.  A
//! returned value is profile-valid CPU data, not a runtime- or
//! hardware-validated checkpoint.

use std::fmt;

use modelq_quant::nvfp4::{self, QuantizedTensor};

/// Name of the profile produced by this module.
pub const TRANSFORMER_ENGINE_NVFP4_PROFILE: &str = "transformer-engine.nvfp4.rowwise.1x16.v1";
/// Number of E2M1 values in one rowwise block.
pub const TRANSFORMER_ENGINE_NVFP4_BLOCK_SIZE: usize = 16;
/// Row alignment required by Transformer Engine's rowwise scale buffer.
pub const TRANSFORMER_ENGINE_NVFP4_SCALE_ROW_ALIGNMENT: usize = 128;
/// Column alignment required by Transformer Engine's rowwise scale buffer.
pub const TRANSFORMER_ENGINE_NVFP4_SCALE_COLUMN_ALIGNMENT: usize = 4;

const FP4_MAX: f32 = 6.0;
const FP8_E4M3_MAX: f32 = 448.0;
const GLOBAL_SCALE_DENOMINATOR: f32 = FP4_MAX * FP8_E4M3_MAX;
const RESERVED_METADATA_NAME: &str = "__metadata__";

/// Errors returned while adapting a native NVFP4 tensor to the profile.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerEngineNvfp4Error {
    /// The source name is empty, whitespace-only, or reserved by SafeTensors.
    InvalidName { name: String },
    /// The source shape is not a supported rowwise matrix shape.
    InvalidShape { shape: Vec<usize> },
    /// Shape multiplication overflowed `usize`.
    ShapeElementCountOverflow { shape: Vec<usize> },
    /// The native element count differs from the requested shape.
    PayloadLengthMismatch { expected: usize, actual: usize },
    /// The native packed byte count differs from the requested shape.
    PackedLengthMismatch { expected: usize, actual: usize },
    /// The native block-scale count differs from the requested shape.
    BlockScaleCountMismatch { expected: usize, actual: usize },
    /// Rounding a logical dimension up to an alignment overflowed `usize`.
    AlignmentOverflow { dimension: usize, alignment: usize },
    /// The padded scale matrix size overflowed `usize`.
    ScaleBufferSizeOverflow,
    /// The native global decode scale is not positive and finite.
    InvalidGlobalScale { scale: f32 },
    /// The derived tensor amax is not finite.
    DerivedAmaxNonFinite { scale: f32 },
    /// The native payload failed ModelQ's validation rules.
    InvalidNativePayload { source: nvfp4::Nvfp4Error },
}

impl fmt::Display for TransformerEngineNvfp4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName { name } => write!(
                formatter,
                "Transformer Engine NVFP4 profile name must be non-empty and not reserved: {name:?}"
            ),
            Self::InvalidShape { shape } => write!(
                formatter,
                "Transformer Engine NVFP4 rowwise shape {shape:?} must have at least two positive dimensions with a final dimension divisible by {TRANSFORMER_ENGINE_NVFP4_BLOCK_SIZE}"
            ),
            Self::ShapeElementCountOverflow { shape } => write!(
                formatter,
                "Transformer Engine NVFP4 shape {shape:?} overflows its element count"
            ),
            Self::PayloadLengthMismatch { expected, actual } => write!(
                formatter,
                "Transformer Engine NVFP4 shape requires {expected} values but the native payload describes {actual}"
            ),
            Self::PackedLengthMismatch { expected, actual } => write!(
                formatter,
                "Transformer Engine NVFP4 shape requires {expected} packed bytes but the native payload has {actual}"
            ),
            Self::BlockScaleCountMismatch { expected, actual } => write!(
                formatter,
                "Transformer Engine NVFP4 shape requires {expected} block scales but the native payload has {actual}"
            ),
            Self::AlignmentOverflow {
                dimension,
                alignment,
            } => write!(
                formatter,
                "rounding Transformer Engine NVFP4 dimension {dimension} to alignment {alignment} overflowed"
            ),
            Self::ScaleBufferSizeOverflow => {
                formatter.write_str("Transformer Engine NVFP4 padded scale matrix size overflowed")
            }
            Self::InvalidGlobalScale { scale } => write!(
                formatter,
                "Transformer Engine NVFP4 native global decode scale must be finite and positive: {scale:?}"
            ),
            Self::DerivedAmaxNonFinite { scale } => write!(
                formatter,
                "Transformer Engine NVFP4 global amax derived from native scale {scale:?} is not finite"
            ),
            Self::InvalidNativePayload { source } => {
                write!(formatter, "native NVFP4 payload is invalid: {source}")
            }
        }
    }
}

impl std::error::Error for TransformerEngineNvfp4Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidNativePayload { source } => Some(source),
            _ => None,
        }
    }
}

/// Owned CPU buffers for the Transformer Engine rowwise 1x16 profile.
///
/// `rowwise_data` has the logical source shape with its final dimension
/// halved because two E2M1 values share one byte.  `rowwise_scale_inv` is a
/// row-major U8 matrix whose logical leading region contains the native E4M3
/// block scales and whose aligned tail is zero padding.  `amax_rowwise` is the
/// one-element F32 tensor amax used by Transformer Engine's tensor-scaling
/// contract.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerEngineNvfp4Tensor {
    /// Exact original source tensor name.
    pub name: String,
    /// Logical, unpadded source shape.
    pub source_shape: Vec<usize>,
    /// Packed rowwise E2M1 bytes.
    pub rowwise_data: Vec<u8>,
    /// Logical shape of `rowwise_data`.
    pub rowwise_data_shape: Vec<usize>,
    /// Row-major padded E4M3 block-scale bytes.
    pub rowwise_scale_inv: Vec<u8>,
    /// Physical shape of `rowwise_scale_inv`, including alignment padding.
    pub rowwise_scale_inv_shape: Vec<usize>,
    /// One tensor-wide F32 amax value; its serialized shape is `[1]`.
    pub amax_rowwise: f32,
}

impl TransformerEngineNvfp4Tensor {
    /// Returns the deterministic output name for the packed rowwise data.
    pub fn rowwise_data_name(&self) -> String {
        format!("{}.rowwise_data", self.name)
    }

    /// Returns the deterministic output name for the rowwise E4M3 scales.
    pub fn rowwise_scale_inv_name(&self) -> String {
        format!("{}.rowwise_scale_inv", self.name)
    }

    /// Returns the deterministic output name for the tensor amax.
    pub fn amax_rowwise_name(&self) -> String {
        format!("{}.amax_rowwise", self.name)
    }
}

/// Exports a validated ModelQ-native NVFP4 tensor to the Transformer Engine
/// rowwise 1x16 profile.
///
/// The source tensor is flattened into `M` rows of `K` values, where `K` is
/// the final logical dimension.  Packed values are copied unchanged.  Native
/// E4M3 block scales are copied into the leading `[M, K / 16]` region of a
/// zero-filled `[round_up(M, 128), round_up(K / 16, 4)]` scale matrix.  For a
/// nonzero tensor, the exported amax is `global_scale * (448 * 6)` in F32
/// arithmetic, so a consumer recovers the global decode factor by dividing by
/// the same denominator.  An all-zero tensor exports an amax of `0.0`.
pub fn export_transformer_engine_nvfp4(
    name: &str,
    shape: &[usize],
    quantized: &QuantizedTensor,
) -> Result<TransformerEngineNvfp4Tensor, TransformerEngineNvfp4Error> {
    validate_name(name)?;
    validate_shape(shape)?;

    let elements = checked_product(shape)?;
    let last_dimension = *shape
        .last()
        .expect("shape validation guarantees a final dimension");
    let rows = checked_product(&shape[..shape.len() - 1])?;

    if quantized.len() != elements {
        return Err(TransformerEngineNvfp4Error::PayloadLengthMismatch {
            expected: elements,
            actual: quantized.len(),
        });
    }

    let expected_packed = nvfp4::packed_len(elements);
    if quantized.packed_values().len() != expected_packed {
        return Err(TransformerEngineNvfp4Error::PackedLengthMismatch {
            expected: expected_packed,
            actual: quantized.packed_values().len(),
        });
    }

    let expected_blocks = nvfp4::block_count(elements);
    if quantized.block_scales().len() != expected_blocks {
        return Err(TransformerEngineNvfp4Error::BlockScaleCountMismatch {
            expected: expected_blocks,
            actual: quantized.block_scales().len(),
        });
    }

    nvfp4::validate_parts(
        quantized.packed_values(),
        quantized.block_scales(),
        quantized.global_scale(),
        elements,
    )
    .map_err(|source| TransformerEngineNvfp4Error::InvalidNativePayload { source })?;

    let global_scale = quantized.global_scale();
    if !global_scale.is_finite() || global_scale <= 0.0 {
        return Err(TransformerEngineNvfp4Error::InvalidGlobalScale {
            scale: global_scale,
        });
    }

    let blocks_per_row = last_dimension / TRANSFORMER_ENGINE_NVFP4_BLOCK_SIZE;
    let padded_rows = round_up(rows, TRANSFORMER_ENGINE_NVFP4_SCALE_ROW_ALIGNMENT)?;
    let padded_blocks = round_up(
        blocks_per_row,
        TRANSFORMER_ENGINE_NVFP4_SCALE_COLUMN_ALIGNMENT,
    )?;
    let scale_length = padded_rows
        .checked_mul(padded_blocks)
        .ok_or(TransformerEngineNvfp4Error::ScaleBufferSizeOverflow)?;
    let mut rowwise_scale_inv = vec![0_u8; scale_length];

    for row in 0..rows {
        let source_start = row * blocks_per_row;
        let source_end = source_start + blocks_per_row;
        let target_start = row * padded_blocks;
        let target_end = target_start + blocks_per_row;
        rowwise_scale_inv[target_start..target_end]
            .copy_from_slice(&quantized.block_scales()[source_start..source_end]);
    }

    let mut rowwise_data_shape = shape.to_vec();
    *rowwise_data_shape
        .last_mut()
        .expect("shape validation guarantees a final dimension") /= 2;

    let amax_rowwise = if quantized.block_scales().iter().all(|&scale| scale == 0) {
        0.0
    } else {
        let amax = global_scale * GLOBAL_SCALE_DENOMINATOR;
        if !amax.is_finite() {
            return Err(TransformerEngineNvfp4Error::DerivedAmaxNonFinite {
                scale: global_scale,
            });
        }
        amax
    };

    Ok(TransformerEngineNvfp4Tensor {
        name: name.to_owned(),
        source_shape: shape.to_vec(),
        rowwise_data: quantized.packed_values().to_vec(),
        rowwise_data_shape,
        rowwise_scale_inv,
        rowwise_scale_inv_shape: vec![padded_rows, padded_blocks],
        amax_rowwise,
    })
}

fn validate_name(name: &str) -> Result<(), TransformerEngineNvfp4Error> {
    if name.trim().is_empty() || name == RESERVED_METADATA_NAME {
        return Err(TransformerEngineNvfp4Error::InvalidName {
            name: name.to_owned(),
        });
    }
    Ok(())
}

fn validate_shape(shape: &[usize]) -> Result<(), TransformerEngineNvfp4Error> {
    let Some(&last_dimension) = shape.last() else {
        return Err(TransformerEngineNvfp4Error::InvalidShape {
            shape: shape.to_vec(),
        });
    };
    if shape.len() < 2
        || shape.contains(&0)
        || last_dimension % TRANSFORMER_ENGINE_NVFP4_BLOCK_SIZE != 0
    {
        return Err(TransformerEngineNvfp4Error::InvalidShape {
            shape: shape.to_vec(),
        });
    }
    Ok(())
}

fn checked_product(values: &[usize]) -> Result<usize, TransformerEngineNvfp4Error> {
    values
        .iter()
        .try_fold(1_usize, |product, &value| product.checked_mul(value))
        .ok_or_else(|| TransformerEngineNvfp4Error::ShapeElementCountOverflow {
            shape: values.to_vec(),
        })
}

fn round_up(dimension: usize, alignment: usize) -> Result<usize, TransformerEngineNvfp4Error> {
    let remainder = dimension % alignment;
    if remainder == 0 {
        return Ok(dimension);
    }
    dimension.checked_add(alignment - remainder).ok_or(
        TransformerEngineNvfp4Error::AlignmentOverflow {
            dimension,
            alignment,
        },
    )
}
