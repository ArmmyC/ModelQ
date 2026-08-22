//! Reference GGUF Q8_0 block quantization.
//!
//! This module intentionally implements one format only.  It mirrors the
//! current llama.cpp reference representation: each block contains one
//! little-endian binary16 scale followed by 32 signed eight-bit values.

use std::fmt;

use half::f16;

/// Number of source values represented by one Q8_0 block.
pub const BLOCK_ELEMENTS: usize = 32;
/// Number of bytes occupied by one Q8_0 block.
pub const BLOCK_BYTES: usize = 2 + BLOCK_ELEMENTS;
/// llama.cpp's stable numeric type identifier for Q8_0 tensors.
pub const GGML_TYPE_Q8_0: u32 = 8;
const QMAX: f32 = 127.0;

/// Errors returned by the Q8_0 reference quantizer.
#[derive(Debug, Clone, PartialEq)]
pub enum Q8_0Error {
    /// A source value is NaN or infinite.
    NonFiniteInput { index: usize, value: f32 },
    /// Q8_0 requires at least one complete block.
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

impl fmt::Display for Q8_0Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput { index, value } => {
                write!(
                    formatter,
                    "Q8_0 input at index {index} is not finite: {value:?}"
                )
            }
            Self::InvalidLength { elements } => write!(
                formatter,
                "Q8_0 requires a non-empty length divisible by {BLOCK_ELEMENTS}, got {elements}"
            ),
            Self::InvalidShape { shape } => write!(
                formatter,
                "Q8_0 shape {shape:?} must have positive dimensions and a final dimension divisible by {BLOCK_ELEMENTS}"
            ),
            Self::ElementCountOverflow { shape } => {
                write!(
                    formatter,
                    "Q8_0 shape {shape:?} overflows its element count"
                )
            }
            Self::LengthMismatch { expected, actual } => write!(
                formatter,
                "Q8_0 shape describes {expected} values but source contains {actual}"
            ),
            Self::InvalidDataLength { expected, actual } => write!(
                formatter,
                "Q8_0 data should contain {expected} bytes but contains {actual}"
            ),
        }
    }
}

impl std::error::Error for Q8_0Error {}

/// A serialized Q8_0 tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedQ8_0 {
    data: Vec<u8>,
    elements: usize,
}

impl QuantizedQ8_0 {
    /// Builds a Q8_0 tensor from its serialized block bytes.
    pub fn from_bytes(data: Vec<u8>, elements: usize) -> Result<Self, Q8_0Error> {
        validate_element_length(elements)?;
        let expected = block_count(elements) * BLOCK_BYTES;
        if data.len() != expected {
            return Err(Q8_0Error::InvalidDataLength {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, elements })
    }

    /// Returns the serialized Q8_0 bytes in block order.
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

    /// Returns the number of Q8_0 blocks.
    pub const fn block_count(&self) -> usize {
        self.elements / BLOCK_ELEMENTS
    }

    /// Dequantizes using the binary16 scale stored in each block.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.elements);
        for block in self.data.chunks_exact(BLOCK_BYTES) {
            let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            values.extend(
                block[2..]
                    .iter()
                    .map(|&value| f32::from(i8::from_le_bytes([value])) * scale),
            );
        }
        values
    }
}

/// Quantizes a flat F32 slice into llama.cpp-compatible Q8_0 blocks.
///
/// Each block computes `d = max(abs(values)) / 127`, stores `d` as a
/// little-endian binary16 value, and rounds `value / d` to a signed byte.  A
/// zero block stores a zero scale and zero quantized values.
pub fn quantize(values: &[f32]) -> Result<QuantizedQ8_0, Q8_0Error> {
    validate_element_length(values.len())?;
    let mut data = Vec::with_capacity(block_count(values.len()) * BLOCK_BYTES);

    for (block_index, block_values) in values.chunks_exact(BLOCK_ELEMENTS).enumerate() {
        let mut max_abs = 0.0_f32;
        for (within_block, &value) in block_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(Q8_0Error::NonFiniteInput {
                    index: block_index * BLOCK_ELEMENTS + within_block,
                    value,
                });
            }
            max_abs = max_abs.max(value.abs());
        }

        let scale = max_abs / QMAX;
        data.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        let inverse_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        for &value in block_values {
            let quantized = (value * inverse_scale).round().clamp(-QMAX, QMAX) as i8;
            data.push(quantized as u8);
        }
    }

    Ok(QuantizedQ8_0 {
        data,
        elements: values.len(),
    })
}

/// Quantizes a shaped tensor after checking the Q8_0 row/block constraint.
///
/// GGUF quantized tensors require the final (fastest-changing) dimension to
/// be divisible by 32.  The serialized dimensions are handled by the GGUF
/// module; this function validates the source shape before flattening it.
pub fn quantize_shaped(values: &[f32], shape: &[usize]) -> Result<QuantizedQ8_0, Q8_0Error> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(Q8_0Error::InvalidShape {
            shape: shape.to_vec(),
        });
    }
    if !shape
        .last()
        .is_some_and(|&dimension| dimension % BLOCK_ELEMENTS == 0)
    {
        return Err(Q8_0Error::InvalidShape {
            shape: shape.to_vec(),
        });
    }
    let expected = shape
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension));
    let expected = expected.ok_or_else(|| Q8_0Error::ElementCountOverflow {
        shape: shape.to_vec(),
    })?;
    if expected != values.len() {
        return Err(Q8_0Error::LengthMismatch {
            expected,
            actual: values.len(),
        });
    }
    quantize(values)
}

fn validate_element_length(elements: usize) -> Result<(), Q8_0Error> {
    if elements == 0 || elements % BLOCK_ELEMENTS != 0 {
        return Err(Q8_0Error::InvalidLength { elements });
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
        BLOCK_BYTES, BLOCK_ELEMENTS, GGML_TYPE_Q8_0, Q8_0Error, quantize, quantize_shaped,
    };

    #[test]
    fn emits_the_reference_block_layout() {
        let mut values = [0.0_f32; BLOCK_ELEMENTS];
        values[0] = -1.0;
        values[1] = -0.5;
        values[2] = 0.5;
        values[3] = 1.0;

        let quantized = quantize(&values).expect("one complete block is valid");

        assert_eq!(GGML_TYPE_Q8_0, 8);
        assert_eq!(quantized.bytes().len(), BLOCK_BYTES);
        assert_eq!(
            quantized.bytes()[..2],
            f16::from_f32(1.0 / 127.0).to_bits().to_le_bytes()
        );
        assert_eq!(
            &quantized.bytes()[2..6],
            &[-127_i8, -64, 64, 127]
                .into_iter()
                .map(|value| value as u8)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn zero_block_has_zero_scale_and_values() {
        let quantized = quantize(&[0.0; BLOCK_ELEMENTS]).expect("zero block is valid");

        assert_eq!(quantized.bytes(), vec![0; BLOCK_BYTES]);
        assert_eq!(quantized.dequantize(), vec![0.0; BLOCK_ELEMENTS]);
    }

    #[test]
    fn round_trip_uses_the_stored_binary16_scale() {
        let values = (0..BLOCK_ELEMENTS)
            .map(|index| index as f32 / 10.0)
            .collect::<Vec<_>>();
        let quantized = quantize(&values).expect("one complete block is valid");
        let scale = f16::from_bits(u16::from_le_bytes([
            quantized.bytes()[0],
            quantized.bytes()[1],
        ]))
        .to_f32();

        assert_eq!(quantized.dequantize()[BLOCK_ELEMENTS - 1], 127.0 * scale);
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
            Err(Q8_0Error::InvalidShape { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_values_and_incomplete_blocks() {
        assert!(matches!(
            quantize(&[0.0; BLOCK_ELEMENTS - 1]),
            Err(Q8_0Error::InvalidLength { .. })
        ));
        let mut values = [0.0_f32; BLOCK_ELEMENTS];
        values[7] = f32::NAN;
        assert!(matches!(
            quantize(&values),
            Err(Q8_0Error::NonFiniteInput { index: 7, .. })
        ));
    }

    #[test]
    fn validates_serialized_length() {
        assert!(matches!(
            super::QuantizedQ8_0::from_bytes(vec![0; BLOCK_BYTES - 1], BLOCK_ELEMENTS),
            Err(Q8_0Error::InvalidDataLength { .. })
        ));
    }
}
