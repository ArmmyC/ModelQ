//! Reference GGUF Q4_0 block quantization (ADR 0033).
//!
//! This module intentionally implements one format only. It mirrors llama.cpp's
//! reference representation at the pinned release: each block contains one
//! little-endian binary16 scale followed by 32 four-bit codes, packed two per
//! byte. A code `q` stands for the value `(q - 8) * scale`, so the codes cover
//! -8 to +7 steps, and the scale carries the sign of the largest-magnitude value.

use std::fmt;

use half::f16;

/// Number of source values represented by one Q4_0 block.
pub const BLOCK_ELEMENTS: usize = 32;
/// Number of bytes occupied by one Q4_0 block: the scale and 16 packed code pairs.
pub const BLOCK_BYTES: usize = 2 + BLOCK_ELEMENTS / 2;
/// llama.cpp's stable numeric type identifier for Q4_0 tensors.
pub const GGML_TYPE_Q4_0: u32 = 2;
const HALF_BLOCK: usize = BLOCK_ELEMENTS / 2;

/// Errors returned by the Q4_0 reference quantizer.
#[derive(Debug, Clone, PartialEq)]
pub enum Q4_0Error {
    /// A source value is NaN or infinite.
    NonFiniteInput { index: usize, value: f32 },
    /// Q4_0 requires at least one complete block.
    InvalidLength { elements: usize },
    /// A shaped tensor's dimensions are not valid for this block format.
    InvalidShape { shape: Vec<usize> },
    /// The shape product overflowed `usize`.
    ElementCountOverflow { shape: Vec<usize> },
    /// The source length does not equal the checked shape product.
    LengthMismatch { expected: usize, actual: usize },
    /// Serialized bytes do not contain the expected number of complete blocks.
    InvalidDataLength { expected: usize, actual: usize },
}

impl fmt::Display for Q4_0Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput { index, value } => {
                write!(
                    formatter,
                    "Q4_0 input at index {index} is not finite: {value:?}"
                )
            }
            Self::InvalidLength { elements } => write!(
                formatter,
                "Q4_0 requires a non-empty length divisible by {BLOCK_ELEMENTS}, got {elements}"
            ),
            Self::InvalidShape { shape } => write!(
                formatter,
                "Q4_0 shape {shape:?} must have positive dimensions and a final dimension divisible by {BLOCK_ELEMENTS}"
            ),
            Self::ElementCountOverflow { shape } => {
                write!(
                    formatter,
                    "Q4_0 shape {shape:?} overflows its element count"
                )
            }
            Self::LengthMismatch { expected, actual } => write!(
                formatter,
                "Q4_0 shape describes {expected} values but source contains {actual}"
            ),
            Self::InvalidDataLength { expected, actual } => write!(
                formatter,
                "Q4_0 data should contain {expected} bytes but contains {actual}"
            ),
        }
    }
}

impl std::error::Error for Q4_0Error {}

/// A serialized Q4_0 tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedQ4_0 {
    data: Vec<u8>,
    elements: usize,
}

impl QuantizedQ4_0 {
    /// Builds a Q4_0 tensor from its serialized block bytes.
    pub fn from_bytes(data: Vec<u8>, elements: usize) -> Result<Self, Q4_0Error> {
        validate_element_length(elements)?;
        let expected = block_count(elements) * BLOCK_BYTES;
        if data.len() != expected {
            return Err(Q4_0Error::InvalidDataLength {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, elements })
    }

    /// Returns the serialized Q4_0 bytes in block order.
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consumes the tensor and returns its serialized bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Returns the number of represented source values.
    pub const fn len(&self) -> usize {
        self.elements
    }

    /// Returns whether this tensor contains no values.
    pub const fn is_empty(&self) -> bool {
        self.elements == 0
    }

    /// Returns the number of Q4_0 blocks.
    pub const fn block_count(&self) -> usize {
        self.elements / BLOCK_ELEMENTS
    }

    /// Dequantizes using the binary16 scale stored in each block.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut values = vec![0.0_f32; self.elements];
        for (block, output) in self
            .data
            .chunks_exact(BLOCK_BYTES)
            .zip(values.chunks_exact_mut(BLOCK_ELEMENTS))
        {
            let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            for (index, &byte) in block[2..].iter().enumerate() {
                output[index] = (f32::from(byte & 0x0F) - 8.0) * scale;
                output[index + HALF_BLOCK] = (f32::from(byte >> 4) - 8.0) * scale;
            }
        }
        values
    }
}

/// Quantizes a flat F32 slice into llama.cpp-compatible Q4_0 blocks.
///
/// Each block takes its value of largest magnitude (the first one, on a tie)
/// and sets `d = max / -8`, stored as binary16. Each value becomes
/// `min(15, trunc(value / d + 8.5))`, computed in single precision in the same
/// order as llama.cpp's reference. Element `j` of a block goes in the low nibble
/// of byte `j`, and element `j + 16` in the high nibble. A zero block stores a
/// negative-zero scale and codes of 8, which decode to zero.
pub fn quantize(values: &[f32]) -> Result<QuantizedQ4_0, Q4_0Error> {
    validate_element_length(values.len())?;
    let mut data = Vec::with_capacity(block_count(values.len()) * BLOCK_BYTES);

    for (block_index, block_values) in values.chunks_exact(BLOCK_ELEMENTS).enumerate() {
        let mut max_abs = 0.0_f32;
        let mut largest = 0.0_f32;
        for (within_block, &value) in block_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(Q4_0Error::NonFiniteInput {
                    index: block_index * BLOCK_ELEMENTS + within_block,
                    value,
                });
            }
            if max_abs < value.abs() {
                max_abs = value.abs();
                largest = value;
            }
        }

        let scale = largest / -8.0;
        data.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        let inverse_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let mut codes = [0_u8; BLOCK_ELEMENTS];
        for (code, &value) in codes.iter_mut().zip(block_values) {
            // `as u8` truncates toward zero, as the C cast does; the minimum keeps the code in a nibble.
            *code = ((value * inverse_scale + 8.5_f32) as u8).min(15);
        }
        for (&low, &high) in codes[..HALF_BLOCK].iter().zip(&codes[HALF_BLOCK..]) {
            data.push(low | (high << 4));
        }
    }

    Ok(QuantizedQ4_0 {
        data,
        elements: values.len(),
    })
}

/// Quantizes a shaped tensor after checking the Q4_0 row/block constraint.
///
/// GGUF quantized tensors require the final (fastest-changing) dimension to be
/// divisible by 32. The serialized dimensions are handled by the GGUF module;
/// this function validates the source shape before flattening it.
pub fn quantize_shaped(values: &[f32], shape: &[usize]) -> Result<QuantizedQ4_0, Q4_0Error> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(Q4_0Error::InvalidShape {
            shape: shape.to_vec(),
        });
    }
    if !shape
        .last()
        .is_some_and(|&dimension| dimension % BLOCK_ELEMENTS == 0)
    {
        return Err(Q4_0Error::InvalidShape {
            shape: shape.to_vec(),
        });
    }
    let expected = shape
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension));
    let expected = expected.ok_or_else(|| Q4_0Error::ElementCountOverflow {
        shape: shape.to_vec(),
    })?;
    if expected != values.len() {
        return Err(Q4_0Error::LengthMismatch {
            expected,
            actual: values.len(),
        });
    }
    quantize(values)
}

fn validate_element_length(elements: usize) -> Result<(), Q4_0Error> {
    if elements == 0 || elements % BLOCK_ELEMENTS != 0 {
        return Err(Q4_0Error::InvalidLength { elements });
    }
    Ok(())
}

const fn block_count(elements: usize) -> usize {
    elements / BLOCK_ELEMENTS
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::{
        BLOCK_BYTES, BLOCK_ELEMENTS, GGML_TYPE_Q4_0, Q4_0Error, quantize, quantize_shaped,
    };

    #[test]
    fn emits_the_reference_block_layout() {
        // x[i] = i - 16: the largest magnitude is -16 (index 0), so d = -16 / -8 = 2.
        // Index j in 0..16 goes to the low nibble of byte j and index j + 16 to the high
        // nibble. The tie at -7.5 rounds toward +infinity, as llama.cpp's reference does.
        let values: Vec<f32> = (0..BLOCK_ELEMENTS)
            .map(|index| index as f32 - 16.0)
            .collect();

        let quantized = quantize(&values).expect("one complete block is valid");

        assert_eq!(GGML_TYPE_Q4_0, 2);
        assert_eq!(quantized.bytes().len(), BLOCK_BYTES);
        assert_eq!(
            quantized.bytes(),
            &[
                0x00, 0x40, // scale 2.0, binary16 little-endian
                0x80, 0x91, 0x91, 0xA2, 0xA2, 0xB3, 0xB3, 0xC4, 0xC4, 0xD5, 0xD5, 0xE6, 0xE6, 0xF7,
                0xF7, 0xF8,
            ]
        );
    }

    #[test]
    fn zero_block_has_a_negative_zero_scale_and_codes_of_eight() {
        let quantized = quantize(&[0.0; BLOCK_ELEMENTS]).expect("zero block is valid");

        let mut expected = vec![0x00, 0x80];
        expected.extend([0x88; 16]);
        assert_eq!(quantized.bytes(), expected.as_slice());
        assert_eq!(quantized.dequantize(), vec![0.0; BLOCK_ELEMENTS]);
    }

    #[test]
    fn the_largest_value_sets_the_sign_and_the_negative_extreme_clamps() {
        let mut values = [0.0_f32; BLOCK_ELEMENTS];
        values[0] = 8.0;
        values[1] = -8.0;

        let decoded = quantize(&values)
            .expect("one complete block is valid")
            .dequantize();

        // The first largest magnitude is +8, so d = -1. +8 is code 0 and decodes exactly.
        // -8 would need code 16, which clamps to 15 and decodes to -7.
        assert_eq!(decoded[0], 8.0);
        assert_eq!(decoded[1], -7.0);
    }

    #[test]
    fn each_value_is_within_one_scale_step_of_its_source() {
        let values: Vec<f32> = (0..4 * BLOCK_ELEMENTS)
            .map(|index| ((index as f32) * 0.37).sin() * 3.0)
            .collect();

        let decoded = quantize(&values)
            .expect("four complete blocks are valid")
            .dequantize();

        for (block, (source, output)) in values
            .chunks_exact(BLOCK_ELEMENTS)
            .zip(decoded.chunks_exact(BLOCK_ELEMENTS))
            .enumerate()
        {
            let largest = source
                .iter()
                .fold(0.0_f32, |acc, value| acc.max(value.abs()));
            // One step is largest / 8. The stored binary16 scale adds at most 8 * 2^-11 of a
            // step, so the bound is (1 + 2^-8) * largest / 8.
            let bound = largest / 8.0 * (1.0 + 1.0 / 256.0);
            for (index, (&source_value, &decoded_value)) in source.iter().zip(output).enumerate() {
                assert!(
                    (source_value - decoded_value).abs() <= bound,
                    "block {block}, value {index}: {source_value} decodes to {decoded_value}"
                );
            }
        }
    }

    #[test]
    fn the_stored_scale_is_the_binary16_of_max_over_minus_eight() {
        let values: Vec<f32> = (0..BLOCK_ELEMENTS)
            .map(|index| if index == 5 { -3.3 } else { 0.1 })
            .collect();

        let quantized = quantize(&values).expect("one complete block is valid");
        let scale = f16::from_bits(u16::from_le_bytes([
            quantized.bytes()[0],
            quantized.bytes()[1],
        ]))
        .to_f32();

        assert_eq!(scale, f16::from_f32(-3.3 / -8.0).to_f32());
    }

    #[test]
    fn enforces_the_last_dimension_constraint() {
        assert_eq!(
            quantize_shaped(&[0.0; 64], &[2, 32])
                .expect("the final dimension is one complete block")
                .len(),
            64
        );
        assert!(matches!(
            quantize_shaped(&[0.0; 64], &[4, 16]),
            Err(Q4_0Error::InvalidShape { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_values_and_incomplete_blocks() {
        assert!(matches!(
            quantize(&[0.0; BLOCK_ELEMENTS - 1]),
            Err(Q4_0Error::InvalidLength { .. })
        ));
        let mut values = [0.0_f32; BLOCK_ELEMENTS];
        values[7] = f32::NAN;
        assert!(matches!(
            quantize(&values),
            Err(Q4_0Error::NonFiniteInput { index: 7, .. })
        ));
    }

    #[test]
    fn validates_serialized_length() {
        assert!(matches!(
            super::QuantizedQ4_0::from_bytes(vec![0; BLOCK_BYTES - 1], BLOCK_ELEMENTS),
            Err(Q4_0Error::InvalidDataLength { .. })
        ));
    }
}
