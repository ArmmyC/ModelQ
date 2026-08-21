//! Scalar, symmetric per-tensor INT8 reference quantization.

use std::fmt;

/// The lowest value emitted by the symmetric INT8 representation.
///
/// `-128` is intentionally unused so that the positive and negative ranges
/// have the same magnitude.
pub const SYMMETRIC_MIN: i8 = -127;
/// The highest value emitted by the symmetric INT8 representation.
pub const SYMMETRIC_MAX: i8 = 127;

/// Default number of elements held by the bounded scalar processing path.
pub const DEFAULT_CHUNK_ELEMENTS: usize = 16 * 1024;

const DEFAULT_ZERO_SCALE: f32 = 1.0;

/// Errors returned by the scalar INT8 reference implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum Int8Error {
    /// The source contains a NaN or infinity at the reported index.
    NonFiniteInput { index: usize, value: f32 },
    /// A caller supplied a scale that cannot represent a symmetric mapping.
    InvalidScale { scale: f32 },
    /// A dequantization input is outside the representation's defined range.
    QuantizedValueOutOfRange { index: usize, value: i8 },
    /// A finite scale and quantized value would produce a non-finite result.
    DequantizedValueOverflow { index: usize },
    /// A scaled source value could not be represented as a finite rounding
    /// input. This should only be reachable for an invalid floating-point
    /// implementation or an unexpectedly extreme input.
    QuantizedValueOverflow { index: usize },
    /// A bounded quantization call was given no room for an output chunk.
    InvalidChunkSize { chunk_size: usize },
}

impl fmt::Display for Int8Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput { index, value } => {
                write!(formatter, "input at index {index} is not finite: {value:?}")
            }
            Self::InvalidScale { scale } => {
                write!(
                    formatter,
                    "INT8 scale must be finite and positive: {scale:?}"
                )
            }
            Self::QuantizedValueOutOfRange { index, value } => write!(
                formatter,
                "quantized value at index {index} is outside [{SYMMETRIC_MIN}, {SYMMETRIC_MAX}]: {value}"
            ),
            Self::DequantizedValueOverflow { index } => {
                write!(
                    formatter,
                    "dequantized value at index {index} is not finite"
                )
            }
            Self::QuantizedValueOverflow { index } => {
                write!(formatter, "scaled value at index {index} is not finite")
            }
            Self::InvalidChunkSize { chunk_size } => {
                write!(
                    formatter,
                    "quantization chunk size must be positive: {chunk_size}"
                )
            }
        }
    }
}

impl std::error::Error for Int8Error {}

/// An INT8 tensor and the per-tensor scale needed to reconstruct it.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    values: Vec<i8>,
    scale: f32,
}

/// Error returned by the bounded replay path when either quantization or its
/// output callback fails.
#[derive(Debug)]
pub enum QuantizationStreamError<E> {
    /// The source value or scale could not be quantized.
    Quantization(Int8Error),
    /// The caller's chunk callback failed.
    Callback(E),
}

impl QuantizedTensor {
    /// Builds an INT8 tensor from already quantized values and a scale.
    ///
    /// The values and scale are validated with the same rules used by
    /// [`dequantize`], making this constructor suitable for alternate
    /// execution backends that produce the representation directly.
    pub fn from_parts(values: Vec<i8>, scale: f32) -> Result<Self, Int8Error> {
        validate_dequantization(values.iter().copied(), scale)?;
        Ok(Self { values, scale })
    }

    /// Returns the quantized values in storage order.
    pub fn values(&self) -> &[i8] {
        &self.values
    }

    /// Returns the positive scale shared by every value in this tensor.
    pub const fn scale(&self) -> f32 {
        self.scale
    }

    /// Returns the number of quantized elements.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether this tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Consumes the tensor and returns its INT8 storage values.
    pub fn into_values(self) -> Vec<i8> {
        self.values
    }

    /// Reconstructs reference [`f32`] values using this tensor's scale.
    pub fn dequantize(&self) -> Vec<f32> {
        self.values
            .iter()
            .map(|&value| f32::from(value) * self.scale)
            .collect()
    }
}

/// Quantizes a tensor with one symmetric scale shared by all elements.
///
/// The scale is `max(abs(values)) / 127`. The output range is deliberately
/// `[-127, 127]`, leaving the asymmetric `-128` INT8 value unused. Values are
/// rounded with [`f32::round`], whose documented policy is ties away from
/// zero, and then clamped to the defined symmetric range. Empty and all-zero
/// inputs use a scale of `1.0`, so their dequantized values remain zero.
pub fn quantize(values: &[f32]) -> Result<QuantizedTensor, Int8Error> {
    let scale = scale_for(values.iter().copied())?;

    let mut quantized = Vec::with_capacity(values.len());
    for (index, &value) in values.iter().enumerate() {
        quantized.push(quantize_value(value, scale, index)?);
    }

    Ok(QuantizedTensor {
        values: quantized,
        scale,
    })
}

/// Computes the symmetric per-tensor scale from a one-pass value iterator.
pub fn scale_for<I>(values: I) -> Result<f32, Int8Error>
where
    I: IntoIterator<Item = f32>,
{
    let mut max_abs = 0.0_f32;
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(Int8Error::NonFiniteInput { index, value });
        }
        max_abs = max_abs.max(value.abs());
    }

    let mut scale = if max_abs == 0.0 {
        DEFAULT_ZERO_SCALE
    } else {
        max_abs / f32::from(SYMMETRIC_MAX)
    };
    // A subnormal input can underflow during the division above. Keeping the
    // scale positive makes the conversion well-defined; such tiny values use
    // only a small portion of the available INT8 range.
    if scale == 0.0 {
        scale = f32::from_bits(1);
    }
    Ok(scale)
}

/// Quantizes one value with a previously computed symmetric scale.
pub fn quantize_value(value: f32, scale: f32, index: usize) -> Result<i8, Int8Error> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Int8Error::InvalidScale { scale });
    }
    if !value.is_finite() {
        return Err(Int8Error::NonFiniteInput { index, value });
    }
    let scaled = value / scale;
    if !scaled.is_finite() {
        return Err(Int8Error::QuantizedValueOverflow { index });
    }
    let rounded = scaled.round();
    let clamped = rounded.clamp(f32::from(SYMMETRIC_MIN), f32::from(SYMMETRIC_MAX));
    Ok(clamped as i8)
}

/// Computes one tensor scale and emits its quantized values in bounded chunks.
///
/// `values` is a replayable factory: it is called once to compute the global
/// scale and once to produce quantized chunks. The callback must consume each
/// borrowed chunk before returning; the next chunk reuses the same bounded
/// allocation. No source or reconstructed `Vec<f32>` is created.
pub fn quantize_replay_chunks<F, I, C, E>(
    mut values: F,
    chunk_size: usize,
    emit: C,
) -> Result<f32, QuantizationStreamError<E>>
where
    F: FnMut() -> I,
    I: IntoIterator<Item = f32>,
    C: FnMut(&[i8]) -> Result<(), E>,
{
    if chunk_size == 0 {
        return Err(QuantizationStreamError::Quantization(
            Int8Error::InvalidChunkSize { chunk_size },
        ));
    }
    let scale = scale_for(values()).map_err(QuantizationStreamError::Quantization)?;
    quantize_replay_chunks_with_scale(&mut values, scale, chunk_size, emit)?;
    Ok(scale)
}

/// Emits bounded quantized chunks using a caller-provided scale.
pub fn quantize_replay_chunks_with_scale<F, I, C, E>(
    values: &mut F,
    scale: f32,
    chunk_size: usize,
    mut emit: C,
) -> Result<(), QuantizationStreamError<E>>
where
    F: FnMut() -> I,
    I: IntoIterator<Item = f32>,
    C: FnMut(&[i8]) -> Result<(), E>,
{
    if chunk_size == 0 {
        return Err(QuantizationStreamError::Quantization(
            Int8Error::InvalidChunkSize { chunk_size },
        ));
    }
    let mut chunk = Vec::with_capacity(chunk_size);
    for (index, value) in values().into_iter().enumerate() {
        chunk.push(
            quantize_value(value, scale, index).map_err(QuantizationStreamError::Quantization)?,
        );
        if chunk.len() == chunk_size {
            emit(&chunk).map_err(QuantizationStreamError::Callback)?;
            chunk.clear();
        }
    }
    if !chunk.is_empty() {
        emit(&chunk).map_err(QuantizationStreamError::Callback)?;
    }
    Ok(())
}

/// Dequantizes symmetric INT8 values with a caller-provided scale.
pub fn dequantize(values: &[i8], scale: f32) -> Result<Vec<f32>, Int8Error> {
    validate_dequantization(values.iter().copied(), scale)?;
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let reconstructed = f32::from(value) * scale;
            if !reconstructed.is_finite() {
                return Err(Int8Error::DequantizedValueOverflow { index });
            }
            Ok(reconstructed)
        })
        .collect()
}

/// Validates quantized values and their scale without allocating a
/// reconstructed value buffer.
pub fn validate_dequantization<I>(values: I, scale: f32) -> Result<(), Int8Error>
where
    I: IntoIterator<Item = i8>,
{
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Int8Error::InvalidScale { scale });
    }
    for (index, value) in values.into_iter().enumerate() {
        if !(SYMMETRIC_MIN..=SYMMETRIC_MAX).contains(&value) {
            return Err(Int8Error::QuantizedValueOutOfRange { index, value });
        }
        let reconstructed = f32::from(value) * scale;
        if !reconstructed.is_finite() {
            return Err(Int8Error::DequantizedValueOverflow { index });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_CHUNK_ELEMENTS, Int8Error, QuantizationStreamError, SYMMETRIC_MAX, SYMMETRIC_MIN,
        dequantize, quantize, quantize_replay_chunks,
    };

    #[test]
    fn quantizes_zero_tensor_without_a_zero_scale() {
        let quantized = quantize(&[0.0, -0.0, 0.0]).expect("zero values are valid");

        assert_eq!(quantized.scale(), 1.0);
        assert_eq!(quantized.values(), [0, 0, 0]);
        assert_eq!(quantized.dequantize(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn golden_rounding_is_symmetric_and_deterministic() {
        let quantized = quantize(&[-1.0, -0.5, 0.0, 0.5, 1.0])
            .expect("finite positive and negative values are valid");

        assert_eq!(quantized.scale(), 1.0 / 127.0);
        // 0.5 / scale is exactly 63.5, so ties-away-from-zero produces ±64.
        assert_eq!(quantized.values(), [-127, -64, 0, 64, 127]);
        assert_eq!(
            quantized.dequantize(),
            [-1.0, -64.0 / 127.0, 0.0, 64.0 / 127.0, 1.0]
        );
    }

    #[test]
    fn scales_by_the_largest_absolute_value_and_stays_in_range() {
        let quantized = quantize(&[-2.0, 0.5, 4.0]).expect("finite values are valid");

        assert_eq!(quantized.scale(), 4.0 / 127.0);
        assert_eq!(quantized.values(), [-64, 16, 127]);
        assert!(
            quantized
                .values()
                .iter()
                .all(|&value| (SYMMETRIC_MIN..=SYMMETRIC_MAX).contains(&value))
        );
    }

    #[test]
    fn dequantizes_values_with_the_supplied_scale() {
        let values = [-127, -1, 0, 1, 127];

        let reconstructed = dequantize(&values, 0.25).expect("the scale is valid");

        assert_eq!(reconstructed, [-31.75, -0.25, 0.0, 0.25, 31.75]);
    }

    #[test]
    fn rejects_non_finite_source_values() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = quantize(&[1.0, value]).expect_err("non-finite values are rejected");
            match error {
                Int8Error::NonFiniteInput {
                    index: 1,
                    value: actual,
                } => assert_eq!(actual.to_bits(), value.to_bits()),
                other => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_invalid_dequantization_inputs() {
        assert_eq!(
            dequantize(&[0], 0.0).expect_err("zero scale is invalid"),
            Int8Error::InvalidScale { scale: 0.0 }
        );
        assert_eq!(
            dequantize(&[-128], 1.0).expect_err("-128 is outside the symmetric range"),
            Int8Error::QuantizedValueOutOfRange {
                index: 0,
                value: -128
            }
        );
    }

    #[test]
    fn empty_tensor_is_valid() {
        let quantized = quantize(&[]).expect("empty tensors are valid");

        assert!(quantized.is_empty());
        assert_eq!(quantized.len(), 0);
        assert_eq!(quantized.scale(), 1.0);
        assert!(quantized.dequantize().is_empty());
    }

    #[test]
    fn replay_chunks_match_reference_for_a_large_tensor() {
        let values = (0..DEFAULT_CHUNK_ELEMENTS + 37)
            .map(|index| (index as f32 - 512.0) / 31.0)
            .collect::<Vec<_>>();
        let reference = quantize(&values).expect("the generated values are finite");
        let mut chunks = Vec::new();
        let scale = quantize_replay_chunks(
            || values.iter().copied(),
            127,
            |chunk| {
                chunks.extend_from_slice(chunk);
                Ok::<(), ()>(())
            },
        )
        .expect("bounded replay quantization succeeds");

        assert_eq!(scale, reference.scale());
        assert_eq!(chunks, reference.values());
    }

    #[test]
    fn rejects_zero_replay_chunk_size() {
        let error = quantize_replay_chunks(|| [1.0_f32].into_iter(), 0, |_| Ok::<(), ()>(()))
            .expect_err("zero-sized chunks are invalid");
        assert!(matches!(
            error,
            QuantizationStreamError::Quantization(Int8Error::InvalidChunkSize { chunk_size: 0 })
        ));
    }
}
