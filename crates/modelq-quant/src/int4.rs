//! Scalar, symmetric group-wise INT4 reference quantization.
//!
//! This module deliberately favors a small, inspectable implementation over
//! throughput. Values are quantized in contiguous groups, each group has its
//! own positive F32 scale, and two signed four-bit values are stored per byte.
//! The low nibble stores the first value and the high nibble stores the second
//! value. The symmetric quantizer emits `[-7, 7]`; the representable signed
//! nibble value `-8` is reserved and is rejected during dequantization.

use std::fmt;

/// The smallest value representable by a signed four-bit two's-complement
/// nibble.
pub const SIGNED_MIN: i8 = -8;
/// The largest value representable by a signed four-bit two's-complement
/// nibble.
pub const SIGNED_MAX: i8 = 7;
/// The lowest value emitted by the symmetric INT4 quantizer.
///
/// `-8` is intentionally unused so that the negative and positive ranges have
/// the same magnitude.
pub const SYMMETRIC_MIN: i8 = -7;
/// The highest value emitted by the symmetric INT4 quantizer.
pub const SYMMETRIC_MAX: i8 = 7;
/// Number of four-bit values stored in one byte.
pub const VALUES_PER_BYTE: usize = 2;

const DEFAULT_ZERO_SCALE: f32 = 1.0;

/// Errors returned by the scalar INT4 reference implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum Int4Error {
    /// A group size of zero cannot define a group.
    InvalidGroupSize { group_size: usize },
    /// The source contains a NaN or infinity at the reported index.
    NonFiniteInput { index: usize, value: f32 },
    /// A group scale cannot represent a symmetric mapping.
    InvalidScale { group: usize, scale: f32 },
    /// A scaled source value could not be represented as a finite rounding
    /// input.
    QuantizedValueOverflow { index: usize },
    /// A value passed to the generic nibble packer is outside signed INT4.
    PackedValueOutOfRange { index: usize, value: i8 },
    /// A dequantization input is outside the symmetric INT4 range.
    QuantizedValueOutOfRange { index: usize, value: i8 },
    /// The packed byte count does not match the requested element count.
    PackedLengthMismatch {
        elements: usize,
        expected: usize,
        actual: usize,
    },
    /// The number of scales does not match the number of groups.
    ScaleCountMismatch { expected: usize, actual: usize },
    /// A finite scale and quantized value would produce a non-finite result.
    DequantizedValueOverflow { index: usize },
}

impl fmt::Display for Int4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGroupSize { group_size } => {
                write!(formatter, "INT4 group size must be positive: {group_size}")
            }
            Self::NonFiniteInput { index, value } => {
                write!(formatter, "input at index {index} is not finite: {value:?}")
            }
            Self::InvalidScale { group, scale } => write!(
                formatter,
                "INT4 scale for group {group} must be finite and positive: {scale:?}"
            ),
            Self::QuantizedValueOverflow { index } => {
                write!(formatter, "scaled value at index {index} is not finite")
            }
            Self::PackedValueOutOfRange { index, value } => write!(
                formatter,
                "packed INT4 value at index {index} is outside [{SIGNED_MIN}, {SIGNED_MAX}]: {value}"
            ),
            Self::QuantizedValueOutOfRange { index, value } => write!(
                formatter,
                "quantized INT4 value at index {index} is outside [{SYMMETRIC_MIN}, {SYMMETRIC_MAX}]: {value}"
            ),
            Self::PackedLengthMismatch {
                elements,
                expected,
                actual,
            } => write!(
                formatter,
                "packed INT4 data for {elements} elements requires {expected} bytes but has {actual}"
            ),
            Self::ScaleCountMismatch { expected, actual } => write!(
                formatter,
                "INT4 data requires {expected} group scales but has {actual}"
            ),
            Self::DequantizedValueOverflow { index } => {
                write!(
                    formatter,
                    "dequantized INT4 value at index {index} is not finite"
                )
            }
        }
    }
}

impl std::error::Error for Int4Error {}

/// A packed group-wise INT4 tensor and its per-group scales.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    packed: Vec<u8>,
    scales: Vec<f32>,
    elements: usize,
    group_size: usize,
}

impl QuantizedTensor {
    /// Returns the packed bytes in storage order.
    pub fn packed_values(&self) -> &[u8] {
        &self.packed
    }

    /// Alias for [`Self::packed_values`].
    pub fn packed(&self) -> &[u8] {
        self.packed_values()
    }

    /// Returns one positive scale for each contiguous group.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Returns the number of source elements represented by this tensor.
    pub const fn len(&self) -> usize {
        self.elements
    }

    /// Returns whether this tensor represents no elements.
    pub const fn is_empty(&self) -> bool {
        self.elements == 0
    }

    /// Returns the contiguous group size used during quantization.
    pub const fn group_size(&self) -> usize {
        self.group_size
    }

    /// Returns the number of packed payload bytes.
    pub fn packed_len(&self) -> usize {
        self.packed.len()
    }

    /// Unpacks the signed INT4 values after validating the packed length.
    pub fn unpacked_values(&self) -> Result<Vec<i8>, Int4Error> {
        unpack(&self.packed, self.elements)
    }

    /// Reconstructs reference F32 values using the per-group scales.
    pub fn dequantize(&self) -> Result<Vec<f32>, Int4Error> {
        dequantize(&self.packed, &self.scales, self.elements, self.group_size)
    }

    /// Consumes the tensor and returns its packed bytes, scales, element count,
    /// and group size.
    pub fn into_parts(self) -> (Vec<u8>, Vec<f32>, usize, usize) {
        (self.packed, self.scales, self.elements, self.group_size)
    }
}

/// Validates a group size before processing any values.
pub fn validate_group_size(group_size: usize) -> Result<(), Int4Error> {
    if group_size == 0 {
        return Err(Int4Error::InvalidGroupSize { group_size });
    }
    Ok(())
}

/// Quantizes contiguous values with one symmetric scale per group.
///
/// Groups are formed in linear storage order. The final group may contain
/// fewer than `group_size` elements. Each scale is the group's maximum
/// absolute value divided by `7`; empty and all-zero groups use a scale of
/// `1.0`. Quantized values are rounded with [`f32::round`] and clamped to
/// `[-7, 7]` before being packed two per byte.
pub fn quantize(values: &[f32], group_size: usize) -> Result<QuantizedTensor, Int4Error> {
    validate_group_size(group_size)?;
    let scales = scales_for(values, group_size)?;
    let mut unpacked = Vec::with_capacity(values.len());

    for (group, &scale) in scales.iter().enumerate() {
        let start = group * group_size;
        let end = values.len().min(start.saturating_add(group_size));
        for (offset, &value) in values[start..end].iter().enumerate() {
            unpacked.push(quantize_value_for_group(
                value,
                scale,
                start + offset,
                group,
            )?);
        }
    }

    let packed = pack(&unpacked)?;
    Ok(QuantizedTensor {
        packed,
        scales,
        elements: values.len(),
        group_size,
    })
}

/// Computes one scale for each contiguous group.
pub fn scales_for(values: &[f32], group_size: usize) -> Result<Vec<f32>, Int4Error> {
    validate_group_size(group_size)?;
    let group_count = group_count(values.len(), group_size);
    let mut scales = Vec::with_capacity(group_count);

    for group in 0..group_count {
        let start = group * group_size;
        let end = values.len().min(start.saturating_add(group_size));
        scales.push(scale_for_group(&values[start..end], group, start)?);
    }

    Ok(scales)
}

/// Alias for [`scales_for`] using the terminology from the representation.
pub fn group_scales(values: &[f32], group_size: usize) -> Result<Vec<f32>, Int4Error> {
    scales_for(values, group_size)
}

/// Quantizes one value with a previously computed group scale.
pub fn quantize_value(value: f32, scale: f32, index: usize) -> Result<i8, Int4Error> {
    quantize_value_for_group(value, scale, index, 0)
}

/// Packs signed INT4 values two per byte, with the first value in the low
/// nibble. An odd final value leaves the high nibble zero.
pub fn pack(values: &[i8]) -> Result<Vec<u8>, Int4Error> {
    let byte_count = values.len() / VALUES_PER_BYTE + values.len() % VALUES_PER_BYTE;
    let mut packed = Vec::with_capacity(byte_count);

    for (index, pair) in values.chunks(VALUES_PER_BYTE).enumerate() {
        let first = pair[0];
        if !(SIGNED_MIN..=SIGNED_MAX).contains(&first) {
            return Err(Int4Error::PackedValueOutOfRange {
                index: index * VALUES_PER_BYTE,
                value: first,
            });
        }
        let low = (first as u8) & 0x0f;
        let high = if let Some(&second) = pair.get(1) {
            if !(SIGNED_MIN..=SIGNED_MAX).contains(&second) {
                return Err(Int4Error::PackedValueOutOfRange {
                    index: index * VALUES_PER_BYTE + 1,
                    value: second,
                });
            }
            ((second as u8) & 0x0f) << 4
        } else {
            0
        };
        packed.push(low | high);
    }

    Ok(packed)
}

/// Unpacks signed INT4 values from two's-complement nibbles.
pub fn unpack(packed: &[u8], elements: usize) -> Result<Vec<i8>, Int4Error> {
    let expected = packed_len(elements);
    if packed.len() != expected {
        return Err(Int4Error::PackedLengthMismatch {
            elements,
            expected,
            actual: packed.len(),
        });
    }

    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        values.push(unpack_value(packed, index));
    }
    Ok(values)
}

/// Dequantizes packed symmetric INT4 values using one scale per group.
pub fn dequantize(
    packed: &[u8],
    scales: &[f32],
    elements: usize,
    group_size: usize,
) -> Result<Vec<f32>, Int4Error> {
    validate_dequantization(packed, scales, elements, group_size)?;

    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        let quantized = unpack_value(packed, index);
        let scale = scales[index / group_size];
        let reconstructed = f32::from(quantized) * scale;
        if !reconstructed.is_finite() {
            return Err(Int4Error::DequantizedValueOverflow { index });
        }
        values.push(reconstructed);
    }
    Ok(values)
}

/// Validates packed values, group scales, and dequantized finiteness without
/// allocating reconstructed values.
pub fn validate_dequantization(
    packed: &[u8],
    scales: &[f32],
    elements: usize,
    group_size: usize,
) -> Result<(), Int4Error> {
    validate_group_size(group_size)?;
    let expected_packed = packed_len(elements);
    if packed.len() != expected_packed {
        return Err(Int4Error::PackedLengthMismatch {
            elements,
            expected: expected_packed,
            actual: packed.len(),
        });
    }

    let expected_groups = group_count(elements, group_size);
    if scales.len() != expected_groups {
        return Err(Int4Error::ScaleCountMismatch {
            expected: expected_groups,
            actual: scales.len(),
        });
    }
    for (group, &scale) in scales.iter().enumerate() {
        validate_scale(scale, group)?;
    }

    for index in 0..elements {
        let quantized = unpack_value(packed, index);
        if !(SYMMETRIC_MIN..=SYMMETRIC_MAX).contains(&quantized) {
            return Err(Int4Error::QuantizedValueOutOfRange {
                index,
                value: quantized,
            });
        }
        let reconstructed = f32::from(quantized) * scales[index / group_size];
        if !reconstructed.is_finite() {
            return Err(Int4Error::DequantizedValueOverflow { index });
        }
    }
    Ok(())
}

fn scale_for_group(values: &[f32], group: usize, start: usize) -> Result<f32, Int4Error> {
    let mut max_abs = 0.0_f32;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(Int4Error::NonFiniteInput {
                index: start.saturating_add(index),
                value,
            });
        }
        max_abs = max_abs.max(value.abs());
    }

    let mut scale = if max_abs == 0.0 {
        DEFAULT_ZERO_SCALE
    } else {
        max_abs / f32::from(SYMMETRIC_MAX)
    };
    if scale == 0.0 {
        scale = f32::from_bits(1);
    }
    validate_scale(scale, group)?;
    Ok(scale)
}

fn quantize_value_for_group(
    value: f32,
    scale: f32,
    index: usize,
    group: usize,
) -> Result<i8, Int4Error> {
    validate_scale(scale, group)?;
    if !value.is_finite() {
        return Err(Int4Error::NonFiniteInput { index, value });
    }
    let scaled = value / scale;
    if !scaled.is_finite() {
        return Err(Int4Error::QuantizedValueOverflow { index });
    }
    Ok(scaled
        .round()
        .clamp(f32::from(SYMMETRIC_MIN), f32::from(SYMMETRIC_MAX)) as i8)
}

fn validate_scale(scale: f32, group: usize) -> Result<(), Int4Error> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Int4Error::InvalidScale { group, scale });
    }
    Ok(())
}

fn group_count(elements: usize, group_size: usize) -> usize {
    elements / group_size + usize::from(elements % group_size != 0)
}

fn packed_len(elements: usize) -> usize {
    elements / VALUES_PER_BYTE + elements % VALUES_PER_BYTE
}

fn unpack_value(packed: &[u8], index: usize) -> i8 {
    let byte = packed[index / VALUES_PER_BYTE];
    let nibble = if index % VALUES_PER_BYTE == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    };
    if nibble & 0x08 == 0 {
        nibble as i8
    } else {
        nibble as i8 - 16
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Int4Error, SIGNED_MAX, SIGNED_MIN, SYMMETRIC_MIN, dequantize, group_scales, pack, quantize,
        scales_for, unpack, validate_dequantization,
    };

    #[test]
    fn computes_independent_scales_for_each_group() {
        let values = [-1.0, 3.0, -2.0, 6.0, 0.0];
        let scales = scales_for(&values, 2).expect("the groups are valid");

        assert_eq!(scales, [3.0 / 7.0, 6.0 / 7.0, 1.0]);
        assert_eq!(
            group_scales(&values, 2).expect("the alias is equivalent"),
            scales
        );
    }

    #[test]
    fn quantizes_groups_and_handles_an_odd_tensor_length() {
        let quantized = quantize(&[-1.0, 3.0, -2.0, 6.0, 0.0], 2)
            .expect("finite values and a positive group size are valid");

        assert_eq!(quantized.len(), 5);
        assert_eq!(quantized.group_size(), 2);
        assert_eq!(quantized.scales(), [3.0 / 7.0, 6.0 / 7.0, 1.0]);
        assert_eq!(quantized.packed_values(), [0x7e, 0x7e, 0x00]);
        assert_eq!(
            quantized
                .unpacked_values()
                .expect("packed values are valid"),
            [-2, 7, -2, 7, 0]
        );
        assert_eq!(
            quantized.dequantize().expect("scales and values are valid"),
            [-2.0 * (3.0 / 7.0), 3.0, -2.0 * (6.0 / 7.0), 6.0, 0.0]
        );
    }

    #[test]
    fn packs_and_unpacks_signed_nibble_golden_values() {
        let values = [SIGNED_MIN, -7, -1, 0, 1, SIGNED_MAX];
        let packed = pack(&values).expect("all values fit in signed INT4");

        assert_eq!(packed, [0x98, 0x0f, 0x71]);
        assert_eq!(
            unpack(&packed, values.len()).expect("the byte count matches"),
            values
        );
    }

    #[test]
    fn packs_and_unpacks_an_odd_number_of_values() {
        let values = [SIGNED_MIN, SIGNED_MAX, -1, 1, 7];
        let packed = pack(&values).expect("all values fit in signed INT4");

        assert_eq!(packed, [0x78, 0x1f, 0x07]);
        assert_eq!(
            unpack(&packed, values.len()).expect("the padding nibble is ignored"),
            values
        );
    }

    #[test]
    fn rejects_invalid_group_sizes_and_packed_values() {
        assert_eq!(
            quantize(&[1.0], 0).expect_err("zero group sizes are invalid"),
            Int4Error::InvalidGroupSize { group_size: 0 }
        );
        assert_eq!(
            pack(&[-9]).expect_err("values below signed INT4 are invalid"),
            Int4Error::PackedValueOutOfRange {
                index: 0,
                value: -9
            }
        );
        assert_eq!(
            unpack(&[0], 3).expect_err("one byte cannot hold three values"),
            Int4Error::PackedLengthMismatch {
                elements: 3,
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn rejects_non_finite_values_and_reserved_negative_endpoint() {
        let error = quantize(&[1.0, f32::NAN], 2).expect_err("NaN input is invalid");
        assert!(matches!(
            error,
            Int4Error::NonFiniteInput { index: 1, value } if value.is_nan()
        ));

        let packed = pack(&[SYMMETRIC_MIN - 1]).expect("-8 is packable signed INT4");
        assert_eq!(
            dequantize(&packed, &[1.0], 1, 1)
                .expect_err("the symmetric quantizer does not emit -8"),
            Int4Error::QuantizedValueOutOfRange {
                index: 0,
                value: SYMMETRIC_MIN - 1
            }
        );
    }

    #[test]
    fn validates_scales_and_group_counts_without_allocating() {
        assert_eq!(
            validate_dequantization(&[0], &[], 1, 1)
                .expect_err("one element needs one group scale"),
            Int4Error::ScaleCountMismatch {
                expected: 1,
                actual: 0
            }
        );
        assert_eq!(
            validate_dequantization(&[0], &[0.0], 1, 1).expect_err("zero scales are invalid"),
            Int4Error::InvalidScale {
                group: 0,
                scale: 0.0
            }
        );
        validate_dequantization(&[], &[], 0, 8).expect("empty tensors have no groups");
    }
}
