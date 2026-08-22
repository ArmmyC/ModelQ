//! ModelQ-native scalar NVFP4 reference quantization.
//!
//! NVFP4 combines signed FP4 E2M1 values with one positive FP8 E4M3 scale for
//! each [`BLOCK_SIZE`] values and one F32 decode scale for the tensor.  This
//! module implements the deterministic, data-free, weight-only baseline from
//! ADR 0010.  It deliberately does not implement Transformer Engine swizzles,
//! transposed runtime buffers, activation quantization, or a container format.

use std::fmt;

use crate::float::{fp4_e2m1, fp8_e4m3};

/// Number of values sharing one FP8 E4M3 block scale.
pub const BLOCK_SIZE: usize = 16;
/// Number of FP4 values stored in one byte.
pub const VALUES_PER_BYTE: usize = 2;
/// Maximum finite magnitude of the E2M1 element format.
pub const FP4_MAX: f32 = fp4_e2m1::MAX_FINITE;
/// Maximum finite magnitude of the E4M3 block-scale format.
pub const FP8_MAX: f32 = fp8_e4m3::MAX_FINITE;
const SCALE_PRODUCT: f32 = FP4_MAX * FP8_MAX;
const MIN_POSITIVE_E4M3_BITS: u8 = 0x01;
const MIN_POSITIVE_F32: f32 = f32::from_bits(1);

/// Errors returned by the ModelQ-native NVFP4 reference implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum Nvfp4Error {
    /// A source value is not finite.
    NonFiniteInput { index: usize, value: f32 },
    /// A shaped tensor is empty or its final dimension is not block-aligned.
    InvalidShape { shape: Vec<usize> },
    /// The product of shaped tensor dimensions overflowed `usize`.
    ShapeElementCountOverflow { shape: Vec<usize> },
    /// The source length does not match the checked shape product.
    ShapeLengthMismatch { expected: usize, actual: usize },
    /// A packed E2M1 nibble is outside the four-bit range.
    PackedValueOutOfRange { index: usize, value: u8 },
    /// The packed payload length does not match the element count.
    PackedLengthMismatch {
        elements: usize,
        expected: usize,
        actual: usize,
    },
    /// The block-scale count does not match the element count.
    BlockScaleCountMismatch { expected: usize, actual: usize },
    /// A block scale is not a positive finite E4M3 value or zero.
    InvalidBlockScale { block: usize, bits: u8 },
    /// A zero block scale was paired with a nonzero E2M1 value.
    ZeroScaleWithNonzeroValue { block: usize, index: usize },
    /// The tensor-wide decode scale is not finite and positive.
    InvalidGlobalScale { scale: f32 },
    /// A reconstructed value is not finite.
    DequantizedValueOverflow { index: usize },
}

impl fmt::Display for Nvfp4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput { index, value } => {
                write!(
                    formatter,
                    "NVFP4 input at index {index} is not finite: {value:?}"
                )
            }
            Self::InvalidShape { shape } => write!(
                formatter,
                "NVFP4 shape {shape:?} must have positive dimensions and a final dimension divisible by {BLOCK_SIZE}"
            ),
            Self::ShapeElementCountOverflow { shape } => {
                write!(
                    formatter,
                    "NVFP4 shape {shape:?} overflows its element count"
                )
            }
            Self::ShapeLengthMismatch { expected, actual } => write!(
                formatter,
                "NVFP4 shape describes {expected} values but source contains {actual}"
            ),
            Self::PackedValueOutOfRange { index, value } => write!(
                formatter,
                "NVFP4 packed value at index {index} is outside [0, 15]: {value}"
            ),
            Self::PackedLengthMismatch {
                elements,
                expected,
                actual,
            } => write!(
                formatter,
                "NVFP4 data for {elements} elements requires {expected} packed bytes but has {actual}"
            ),
            Self::BlockScaleCountMismatch { expected, actual } => write!(
                formatter,
                "NVFP4 data requires {expected} block scales but has {actual}"
            ),
            Self::InvalidBlockScale { block, bits } => write!(
                formatter,
                "NVFP4 block {block} has invalid E4M3 scale bits: {bits:#04x}"
            ),
            Self::ZeroScaleWithNonzeroValue { block, index } => write!(
                formatter,
                "NVFP4 block {block} has a zero scale but a nonzero value at index {index}"
            ),
            Self::InvalidGlobalScale { scale } => write!(
                formatter,
                "NVFP4 global decode scale must be finite and positive: {scale:?}"
            ),
            Self::DequantizedValueOverflow { index } => write!(
                formatter,
                "NVFP4 dequantized value at index {index} is not finite"
            ),
        }
    }
}

impl std::error::Error for Nvfp4Error {}

/// A packed ModelQ-native NVFP4 tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    packed: Vec<u8>,
    block_scales: Vec<u8>,
    global_scale: f32,
    elements: usize,
}

impl QuantizedTensor {
    /// Builds an NVFP4 tensor from its native fields after validation.
    pub fn from_parts(
        packed: Vec<u8>,
        block_scales: Vec<u8>,
        global_scale: f32,
        elements: usize,
    ) -> Result<Self, Nvfp4Error> {
        validate_parts(&packed, &block_scales, global_scale, elements)?;
        Ok(Self {
            packed,
            block_scales,
            global_scale,
            elements,
        })
    }

    /// Returns packed E2M1 bytes, with element zero in each low nibble.
    pub fn packed_values(&self) -> &[u8] {
        &self.packed
    }

    /// Alias for [`Self::packed_values`].
    pub fn packed(&self) -> &[u8] {
        self.packed_values()
    }

    /// Returns one E4M3 bit pattern for each 16-value block.
    pub fn block_scales(&self) -> &[u8] {
        &self.block_scales
    }

    /// Alias for [`Self::block_scales`].
    pub fn scales(&self) -> &[u8] {
        self.block_scales()
    }

    /// Returns the F32 decode scale applied to the whole tensor.
    pub const fn global_scale(&self) -> f32 {
        self.global_scale
    }

    /// Returns the number of source values represented by this tensor.
    pub const fn len(&self) -> usize {
        self.elements
    }

    /// Returns whether this tensor represents no values.
    pub const fn is_empty(&self) -> bool {
        self.elements == 0
    }

    /// Returns the number of packed payload bytes.
    pub fn packed_len(&self) -> usize {
        self.packed.len()
    }

    /// Unpacks the E2M1 nibbles after validating the payload length.
    pub fn unpacked_values(&self) -> Result<Vec<u8>, Nvfp4Error> {
        unpack(&self.packed, self.elements)
    }

    /// Reconstructs F32 values using the stored scales.
    pub fn dequantize(&self) -> Result<Vec<f32>, Nvfp4Error> {
        dequantize(
            &self.packed,
            &self.block_scales,
            self.global_scale,
            self.elements,
        )
    }

    /// Consumes the tensor and returns packed values, block scales, global
    /// scale, and element count in native representation order.
    pub fn into_parts(self) -> (Vec<u8>, Vec<u8>, f32, usize) {
        (
            self.packed,
            self.block_scales,
            self.global_scale,
            self.elements,
        )
    }
}

/// Quantizes finite F32 values with the native 16-value NVFP4 hierarchy.
///
/// The returned global scale is a decode scale.  Each source value is
/// reconstructed as `e2m1 * e4m3_block_scale * global_scale`.  The input is
/// treated as a flattened row-major stream; shape metadata belongs to a
/// caller or a future container layer.  Use [`quantize_shaped`] when the
/// source tensor's final dimension must be checked against the native block
/// layout.
pub fn quantize(values: &[f32]) -> Result<QuantizedTensor, Nvfp4Error> {
    let global_amax = max_abs(values)?;
    let global_scale = if global_amax == 0.0 {
        1.0
    } else {
        (global_amax / SCALE_PRODUCT).max(MIN_POSITIVE_F32)
    };

    let mut unpacked = Vec::with_capacity(values.len());
    let mut block_scales = Vec::with_capacity(block_count(values.len()));
    for (block, chunk) in values.chunks(BLOCK_SIZE).enumerate() {
        let block_amax = chunk
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        if block_amax == 0.0 {
            block_scales.push(0);
            unpacked.extend(std::iter::repeat_n(0, chunk.len()));
            continue;
        }

        let scale_input = (block_amax / global_amax) * FP8_MAX;
        let mut scale_bits = fp8_e4m3::encode(scale_input);
        if scale_bits == 0 {
            scale_bits = MIN_POSITIVE_E4M3_BITS;
        }
        let decoded_block_scale = fp8_e4m3::decode(scale_bits);
        block_scales.push(scale_bits);

        for (offset, &value) in chunk.iter().enumerate() {
            let index = block * BLOCK_SIZE + offset;
            let scaled = ((value / global_amax) * SCALE_PRODUCT) / decoded_block_scale;
            let code = fp4_e2m1::encode(scaled)
                .map_err(|_| Nvfp4Error::NonFiniteInput { index, value })?;
            unpacked.push(code);
        }
    }

    let packed = pack(&unpacked)?;
    Ok(QuantizedTensor {
        packed,
        block_scales,
        global_scale,
        elements: values.len(),
    })
}

/// Quantizes a shaped row-major tensor after checking the native block rule.
///
/// NVFP4 groups consecutive values along the final dimension, so this entry
/// point requires non-zero dimensions and a final dimension divisible by
/// [`BLOCK_SIZE`].  The returned representation remains flat; callers retain
/// the original `shape` for their container or tensor metadata.
pub fn quantize_shaped(values: &[f32], shape: &[usize]) -> Result<QuantizedTensor, Nvfp4Error> {
    if shape.is_empty()
        || shape.contains(&0)
        || !shape
            .last()
            .is_some_and(|&dimension| dimension % BLOCK_SIZE == 0)
    {
        return Err(Nvfp4Error::InvalidShape {
            shape: shape.to_vec(),
        });
    }

    let expected = shape
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| Nvfp4Error::ShapeElementCountOverflow {
            shape: shape.to_vec(),
        })?;
    if expected != values.len() {
        return Err(Nvfp4Error::ShapeLengthMismatch {
            expected,
            actual: values.len(),
        });
    }

    quantize(values)
}

/// Packs E2M1 bit patterns two per byte, with the first value in the low
/// nibble.  An odd final value leaves the high nibble zero.
pub fn pack(values: &[u8]) -> Result<Vec<u8>, Nvfp4Error> {
    let mut packed = Vec::with_capacity(packed_len(values.len()));
    for (pair_index, pair) in values.chunks(VALUES_PER_BYTE).enumerate() {
        let first = pair[0];
        if first > 0x0f {
            return Err(Nvfp4Error::PackedValueOutOfRange {
                index: pair_index * VALUES_PER_BYTE,
                value: first,
            });
        }
        let second = pair.get(1).copied().unwrap_or(0);
        if second > 0x0f {
            return Err(Nvfp4Error::PackedValueOutOfRange {
                index: pair_index * VALUES_PER_BYTE + 1,
                value: second,
            });
        }
        packed.push(first | (second << 4));
    }
    Ok(packed)
}

/// Unpacks E2M1 bit patterns from low-nibble-first bytes.
pub fn unpack(packed: &[u8], elements: usize) -> Result<Vec<u8>, Nvfp4Error> {
    let expected = packed_len(elements);
    if packed.len() != expected {
        return Err(Nvfp4Error::PackedLengthMismatch {
            elements,
            expected,
            actual: packed.len(),
        });
    }

    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        let byte = packed[index / VALUES_PER_BYTE];
        values.push(if index % VALUES_PER_BYTE == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        });
    }
    Ok(values)
}

/// Reconstructs F32 values from ModelQ-native NVFP4 fields.
pub fn dequantize(
    packed: &[u8],
    block_scales: &[u8],
    global_scale: f32,
    elements: usize,
) -> Result<Vec<f32>, Nvfp4Error> {
    validate_parts(packed, block_scales, global_scale, elements)?;
    let values = unpack(packed, elements)?;
    let mut output = Vec::with_capacity(elements);
    for (index, code) in values.into_iter().enumerate() {
        let block = index / BLOCK_SIZE;
        let block_scale_bits = block_scales[block];
        let block_scale = fp8_e4m3::decode(block_scale_bits);
        let reconstructed = fp4_e2m1::decode(code) * block_scale * global_scale;
        if !reconstructed.is_finite() {
            return Err(Nvfp4Error::DequantizedValueOverflow { index });
        }
        output.push(reconstructed);
    }
    Ok(output)
}

/// Validates packed values, block scales, and the tensor-wide decode scale.
pub fn validate_parts(
    packed: &[u8],
    block_scales: &[u8],
    global_scale: f32,
    elements: usize,
) -> Result<(), Nvfp4Error> {
    if !global_scale.is_finite() || global_scale <= 0.0 {
        return Err(Nvfp4Error::InvalidGlobalScale {
            scale: global_scale,
        });
    }

    let expected_packed = packed_len(elements);
    if packed.len() != expected_packed {
        return Err(Nvfp4Error::PackedLengthMismatch {
            elements,
            expected: expected_packed,
            actual: packed.len(),
        });
    }

    let expected_blocks = block_count(elements);
    if block_scales.len() != expected_blocks {
        return Err(Nvfp4Error::BlockScaleCountMismatch {
            expected: expected_blocks,
            actual: block_scales.len(),
        });
    }

    let values = unpack(packed, elements)?;
    for (block, &scale_bits) in block_scales.iter().enumerate() {
        validate_block_scale(scale_bits, block)?;
        if scale_bits == 0 {
            let start = block.saturating_mul(BLOCK_SIZE);
            let end = elements.min(start.saturating_add(BLOCK_SIZE));
            if let Some(offset) = values[start..end].iter().position(|&code| code & 0x07 != 0) {
                return Err(Nvfp4Error::ZeroScaleWithNonzeroValue {
                    block,
                    index: start + offset,
                });
            }
        }
    }
    Ok(())
}

/// Returns the packed byte count for an element count.
pub const fn packed_len(elements: usize) -> usize {
    if elements % VALUES_PER_BYTE == 0 {
        elements / VALUES_PER_BYTE
    } else {
        elements / VALUES_PER_BYTE + 1
    }
}

/// Returns the number of fixed-size NVFP4 blocks for an element count.
pub const fn block_count(elements: usize) -> usize {
    if elements % BLOCK_SIZE == 0 {
        elements / BLOCK_SIZE
    } else {
        elements / BLOCK_SIZE + 1
    }
}

fn max_abs(values: &[f32]) -> Result<f32, Nvfp4Error> {
    let mut maximum = 0.0_f32;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(Nvfp4Error::NonFiniteInput { index, value });
        }
        maximum = maximum.max(value.abs());
    }
    Ok(maximum)
}

fn validate_block_scale(bits: u8, block: usize) -> Result<(), Nvfp4Error> {
    if bits == 0 {
        return Ok(());
    }
    let decoded = fp8_e4m3::decode(bits);
    if bits & 0x80 != 0 || !decoded.is_finite() || decoded <= 0.0 {
        return Err(Nvfp4Error::InvalidBlockScale { block, bits });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK_SIZE, FP4_MAX, FP8_MAX, Nvfp4Error, block_count, dequantize, pack, packed_len,
        quantize, quantize_shaped, unpack, validate_parts,
    };

    #[test]
    fn packs_and_unpacks_low_nibble_first() {
        let values = [0x0, 0x1, 0x7, 0xf, 0x8];
        let packed = pack(&values).expect("all E2M1 codes fit in a nibble");
        assert_eq!(packed, [0x10, 0xf7, 0x08]);
        assert_eq!(
            unpack(&packed, values.len()).expect("length matches"),
            values
        );
    }

    #[test]
    fn quantizes_and_reconstructs_one_exact_e2m1_block() {
        let source = [
            -FP4_MAX, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
            FP4_MAX,
        ];
        let quantized = quantize(&source).expect("finite values are valid");

        assert_eq!(quantized.len(), BLOCK_SIZE);
        assert_eq!(quantized.block_scales(), [0x7e]);
        assert_eq!(
            quantized.unpacked_values().expect("packed length matches"),
            [
                0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                0x06, 0x07,
            ]
        );
        let reconstructed = quantized.dequantize().expect("parts validate");
        for (actual, expected) in reconstructed.iter().zip(source) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn validates_shaped_final_dimension_and_preserves_flat_encoding() {
        let source = [0.0_f32; BLOCK_SIZE * 2];
        let quantized = quantize_shaped(&source, &[2, BLOCK_SIZE])
            .expect("two complete final-dimension blocks are valid");
        assert_eq!(quantized.len(), source.len());
        assert_eq!(quantized.packed_values(), [0; BLOCK_SIZE]);

        assert!(matches!(
            quantize_shaped(&source, &[4, BLOCK_SIZE / 2]),
            Err(Nvfp4Error::InvalidShape { .. })
        ));
        assert!(matches!(
            quantize_shaped(&source, &[]),
            Err(Nvfp4Error::InvalidShape { .. })
        ));
        assert!(matches!(
            quantize_shaped(&source, &[0, BLOCK_SIZE]),
            Err(Nvfp4Error::InvalidShape { .. })
        ));
    }

    #[test]
    fn checks_shaped_element_count_without_overflow() {
        let source = [0.0_f32; BLOCK_SIZE];
        assert_eq!(
            quantize_shaped(&source, &[2, BLOCK_SIZE]),
            Err(Nvfp4Error::ShapeLengthMismatch {
                expected: BLOCK_SIZE * 2,
                actual: BLOCK_SIZE,
            })
        );
        assert!(matches!(
            quantize_shaped(&source, &[usize::MAX, BLOCK_SIZE]),
            Err(Nvfp4Error::ShapeElementCountOverflow { .. })
        ));
    }

    #[test]
    fn uses_a_global_scale_and_independent_block_scales() {
        let mut source = vec![0.0; BLOCK_SIZE * 2];
        source[0] = 1.0;
        source[BLOCK_SIZE] = 6.0;
        let quantized = quantize(&source).expect("finite values are valid");

        assert_eq!(quantized.block_scales().len(), 2);
        assert!(quantized.block_scales()[0] < quantized.block_scales()[1]);
        assert!((quantized.global_scale() - 6.0 / (FP4_MAX * FP8_MAX)).abs() < 1e-9);
        let reconstructed = quantized.dequantize().expect("parts validate");
        assert!((reconstructed[0] - 1.0).abs() < 0.05);
        assert!((reconstructed[BLOCK_SIZE] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn zero_and_empty_tensors_have_explicit_safe_scales() {
        let zero = quantize(&[0.0; BLOCK_SIZE + 1]).expect("zero values are valid");
        assert_eq!(zero.global_scale(), 1.0);
        assert_eq!(zero.block_scales(), [0, 0]);
        assert_eq!(zero.packed_values(), [0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            zero.dequantize().expect("zero parts validate"),
            [0.0; BLOCK_SIZE + 1]
        );

        let empty = quantize(&[]).expect("empty tensors are valid");
        assert!(empty.is_empty());
        assert_eq!(empty.packed_values(), []);
        assert_eq!(empty.block_scales(), []);
        assert_eq!(empty.dequantize().expect("empty parts validate"), []);
    }

    #[test]
    fn nonfinite_inputs_are_rejected_before_writing_parts() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = quantize(&[1.0, value]).expect_err("nonfinite input is invalid");
            assert!(matches!(error, Nvfp4Error::NonFiniteInput { index: 1, .. }));
        }
    }

    #[test]
    fn validates_lengths_scales_and_zero_blocks() {
        assert_eq!(
            validate_parts(&[0], &[], 1.0, 1).expect_err("one value needs one block scale"),
            Nvfp4Error::BlockScaleCountMismatch {
                expected: 1,
                actual: 0,
            }
        );
        assert_eq!(
            validate_parts(&[0], &[0], 1.0, 3).expect_err("three values need two bytes"),
            Nvfp4Error::PackedLengthMismatch {
                elements: 3,
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            validate_parts(&[0], &[0x7f], 1.0, 1).expect_err("NaN scale is invalid"),
            Nvfp4Error::InvalidBlockScale {
                block: 0,
                bits: 0x7f,
            }
        );
        assert_eq!(
            validate_parts(&[0x01], &[0], 1.0, 1)
                .expect_err("nonzero value cannot use a zero scale"),
            Nvfp4Error::ZeroScaleWithNonzeroValue { block: 0, index: 0 }
        );
        assert_eq!(
            validate_parts(&[0], &[0], 0.0, 1).expect_err("global scale must be positive"),
            Nvfp4Error::InvalidGlobalScale { scale: 0.0 }
        );
    }

    #[test]
    fn supports_parts_round_trip_and_reports_lengths() {
        let source = [0.25; BLOCK_SIZE];
        let quantized = quantize(&source).expect("finite values are valid");
        let (packed, scales, global_scale, elements) = quantized.clone().into_parts();
        assert_eq!(packed.len(), packed_len(elements));
        assert_eq!(scales.len(), block_count(elements));
        let restored = super::QuantizedTensor::from_parts(packed, scales, global_scale, elements)
            .expect("quantized parts remain valid");
        assert_eq!(restored, quantized);
        assert_eq!(
            dequantize(
                restored.packed(),
                restored.scales(),
                restored.global_scale(),
                restored.len()
            )
            .expect("parts validate"),
            restored.dequantize().expect("parts validate")
        );
    }

    #[test]
    fn clamps_e4m3_scale_underflow_without_dividing_by_zero() {
        let mut source = vec![0.0; BLOCK_SIZE * 2];
        source[0] = f32::from_bits(1);
        source[BLOCK_SIZE] = 1.0;
        let quantized = quantize(&source).expect("finite subnormal values are valid");
        assert_eq!(quantized.block_scales()[0], 0x01);
        assert!(quantized.dequantize().is_ok());
    }
}
