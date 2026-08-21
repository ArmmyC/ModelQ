//! Numerical reconstruction and compression diagnostics.
//!
//! Reconstruction metrics accept an iterator for the reconstructed values, so
//! callers can measure a mapped or freshly dequantized tensor without first
//! allocating a duplicate `Vec<f32>`.

use std::fmt;

use crate::quant::int8::{QuantizedTensor, SYMMETRIC_MAX, SYMMETRIC_MIN};

/// Errors returned while calculating diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticsError {
    /// The source and reconstructed iterators did not contain the same number
    /// of elements.
    LengthMismatch {
        /// Number of source values supplied.
        source_len: usize,
        /// Number of reconstructed values observed.
        reconstructed_len: usize,
    },
    /// The source contains a NaN or infinity at the reported index.
    NonFiniteSource { index: usize, value: f32 },
    /// The reconstructed values contain a NaN or infinity at the reported
    /// index.
    NonFiniteReconstructed { index: usize, value: f32 },
    /// A platform-sized element count could not be represented as `u64`.
    ElementCountOverflow,
    /// Adding quantized payload and overhead byte counts overflowed `u64`.
    ByteCountOverflow,
}

impl fmt::Display for DiagnosticsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                source_len,
                reconstructed_len,
            } => write!(
                formatter,
                "source has {source_len} values but reconstruction has {reconstructed_len}"
            ),
            Self::NonFiniteSource { index, value } => {
                write!(
                    formatter,
                    "source value at index {index} is not finite: {value:?}"
                )
            }
            Self::NonFiniteReconstructed { index, value } => write!(
                formatter,
                "reconstructed value at index {index} is not finite: {value:?}"
            ),
            Self::ElementCountOverflow => {
                write!(formatter, "element count cannot be represented as u64")
            }
            Self::ByteCountOverflow => write!(formatter, "diagnostic byte count overflowed u64"),
        }
    }
}

impl std::error::Error for DiagnosticsError {}

/// Reconstruction-quality metrics calculated in one streaming pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructionMetrics {
    /// Number of paired source and reconstructed values.
    pub elements: u64,
    /// Mean squared error.
    pub mse: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Largest absolute reconstruction error.
    pub max_abs_error: f64,
    /// Signal-to-quantization-noise ratio in decibels when defined.
    pub sqnr_db: Option<f64>,
}

/// Compression byte accounting for one quantized tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionAccounting {
    /// Bytes occupied by the original tensor representation.
    pub source_bytes: u64,
    /// Bytes occupied by the quantized payload, excluding metadata overhead.
    pub quantized_payload_bytes: u64,
    /// Scale and other representation metadata bytes.
    pub overhead_bytes: u64,
    /// Payload plus overhead bytes required by the quantized representation.
    pub total_quantized_bytes: u64,
}

impl CompressionAccounting {
    /// Returns the source-to-quantized size ratio, or `None` for an empty
    /// quantized representation.
    pub fn compression_ratio(&self) -> Option<f64> {
        (self.total_quantized_bytes != 0)
            .then(|| self.source_bytes as f64 / self.total_quantized_bytes as f64)
    }
}

/// Tensor-level diagnostics combining reconstruction metrics and byte counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorDiagnostics {
    /// Number of source elements.
    pub elements: u64,
    /// Original tensor size in bytes.
    pub source_bytes: u64,
    /// Quantized payload plus representation overhead in bytes.
    pub quantized_bytes: u64,
    /// Mean squared reconstruction error.
    pub mse: f64,
    /// Mean absolute reconstruction error.
    pub mae: f64,
    /// Largest absolute reconstruction error.
    pub max_abs_error: f64,
    /// Signal-to-quantization-noise ratio in decibels when defined.
    pub sqnr_db: Option<f64>,
    /// Number of quantized values at either symmetric endpoint.
    pub saturated_values: u64,
}

impl TensorDiagnostics {
    /// Returns the source-to-quantized size ratio, or `None` for zero output
    /// bytes.
    pub fn compression_ratio(&self) -> Option<f64> {
        (self.quantized_bytes != 0).then(|| self.source_bytes as f64 / self.quantized_bytes as f64)
    }
}

/// Calculates reconstruction metrics without collecting reconstructed values.
pub fn reconstruction_metrics<I>(
    source: &[f32],
    reconstructed: I,
) -> Result<ReconstructionMetrics, DiagnosticsError>
where
    I: IntoIterator<Item = f32>,
{
    let mut reconstructed = reconstructed.into_iter();
    let mut squared_error_sum = 0.0_f64;
    let mut absolute_error_sum = 0.0_f64;
    let mut source_energy = 0.0_f64;
    let mut max_abs_error = 0.0_f64;

    for (index, &source_value) in source.iter().enumerate() {
        if !source_value.is_finite() {
            return Err(DiagnosticsError::NonFiniteSource {
                index,
                value: source_value,
            });
        }
        let Some(reconstructed_value) = reconstructed.next() else {
            return Err(DiagnosticsError::LengthMismatch {
                source_len: source.len(),
                reconstructed_len: index,
            });
        };
        if !reconstructed_value.is_finite() {
            return Err(DiagnosticsError::NonFiniteReconstructed {
                index,
                value: reconstructed_value,
            });
        }

        let source_value = f64::from(source_value);
        let reconstructed_value = f64::from(reconstructed_value);
        let absolute_error = (source_value - reconstructed_value).abs();
        squared_error_sum += absolute_error * absolute_error;
        absolute_error_sum += absolute_error;
        source_energy += source_value * source_value;
        max_abs_error = max_abs_error.max(absolute_error);
    }

    if reconstructed.next().is_some() {
        let mut reconstructed_len = source.len().saturating_add(1);
        for _ in reconstructed {
            reconstructed_len = reconstructed_len.saturating_add(1);
        }
        return Err(DiagnosticsError::LengthMismatch {
            source_len: source.len(),
            reconstructed_len,
        });
    }

    let elements =
        u64::try_from(source.len()).map_err(|_| DiagnosticsError::ElementCountOverflow)?;
    let elements_f64 = elements as f64;
    let mse = if elements == 0 {
        0.0
    } else {
        squared_error_sum / elements_f64
    };
    let mae = if elements == 0 {
        0.0
    } else {
        absolute_error_sum / elements_f64
    };
    let sqnr_db = (source_energy > 0.0 && squared_error_sum > 0.0)
        .then(|| 10.0 * (source_energy / squared_error_sum).log10());

    Ok(ReconstructionMetrics {
        elements,
        mse,
        mae,
        max_abs_error,
        sqnr_db,
    })
}

/// Counts values that reached either endpoint of the symmetric INT8 range.
pub fn saturation_count(values: &[i8]) -> u64 {
    values
        .iter()
        .filter(|&&value| value == SYMMETRIC_MIN || value == SYMMETRIC_MAX)
        .count() as u64
}

/// Accounts for payload and representation overhead bytes with overflow
/// checking.
pub fn compression_accounting(
    source_bytes: u64,
    quantized_payload_bytes: u64,
    overhead_bytes: u64,
) -> Result<CompressionAccounting, DiagnosticsError> {
    let total_quantized_bytes = quantized_payload_bytes
        .checked_add(overhead_bytes)
        .ok_or(DiagnosticsError::ByteCountOverflow)?;
    Ok(CompressionAccounting {
        source_bytes,
        quantized_payload_bytes,
        overhead_bytes,
        total_quantized_bytes,
    })
}

/// Calculates all Task 7 diagnostics for a scalar INT8 tensor.
///
/// Dequantized values are generated directly from the quantized slice and
/// consumed by [`reconstruction_metrics`]; no reconstructed `Vec<f32>` is
/// allocated.
pub fn int8_tensor_diagnostics(
    source: &[f32],
    quantized: &QuantizedTensor,
    source_bytes: u64,
    scale_bytes: u64,
) -> Result<TensorDiagnostics, DiagnosticsError> {
    let scale = quantized.scale();
    let metrics = reconstruction_metrics(
        source,
        quantized
            .values()
            .iter()
            .map(|&value| f32::from(value) * scale),
    )?;
    let quantized_payload_bytes =
        u64::try_from(quantized.values().len()).map_err(|_| DiagnosticsError::ByteCountOverflow)?;
    let accounting = compression_accounting(source_bytes, quantized_payload_bytes, scale_bytes)?;

    Ok(TensorDiagnostics {
        elements: metrics.elements,
        source_bytes,
        quantized_bytes: accounting.total_quantized_bytes,
        mse: metrics.mse,
        mae: metrics.mae,
        max_abs_error: metrics.max_abs_error,
        sqnr_db: metrics.sqnr_db,
        saturated_values: saturation_count(quantized.values()),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        DiagnosticsError, compression_accounting, int8_tensor_diagnostics, reconstruction_metrics,
        saturation_count,
    };
    use crate::quant::int8::quantize;

    #[test]
    fn calculates_hand_checked_reconstruction_metrics() {
        let metrics = reconstruction_metrics(&[1.0, 2.0, 3.0], [0.0, 3.0, 5.0])
            .expect("the arrays have matching finite values");

        assert_eq!(metrics.elements, 3);
        assert_eq!(metrics.mse, 2.0);
        assert_eq!(metrics.mae, 4.0 / 3.0);
        assert_eq!(metrics.max_abs_error, 2.0);
        assert_eq!(metrics.sqnr_db, Some(10.0 * (14.0_f64 / 6.0).log10()));
    }

    #[test]
    fn streams_empty_and_perfect_reconstruction() {
        let metrics =
            reconstruction_metrics(&[], std::iter::empty()).expect("empty reconstruction is valid");
        assert_eq!(metrics.elements, 0);
        assert_eq!(metrics.mse, 0.0);
        assert_eq!(metrics.mae, 0.0);
        assert_eq!(metrics.max_abs_error, 0.0);
        assert_eq!(metrics.sqnr_db, None);

        let metrics = reconstruction_metrics(&[1.0, -2.0], [1.0, -2.0])
            .expect("perfect reconstruction is valid");
        assert_eq!(metrics.mse, 0.0);
        assert_eq!(metrics.mae, 0.0);
        assert_eq!(metrics.max_abs_error, 0.0);
        assert_eq!(metrics.sqnr_db, None);
    }

    #[test]
    fn rejects_length_and_non_finite_mismatches() {
        assert_eq!(
            reconstruction_metrics(&[1.0, 2.0], [1.0]).expect_err("short reconstruction fails"),
            DiagnosticsError::LengthMismatch {
                source_len: 2,
                reconstructed_len: 1
            }
        );
        assert_eq!(
            reconstruction_metrics(&[1.0], [1.0, 2.0]).expect_err("long reconstruction fails"),
            DiagnosticsError::LengthMismatch {
                source_len: 1,
                reconstructed_len: 2
            }
        );
        assert!(matches!(
            reconstruction_metrics(&[f32::NAN], [0.0]),
            Err(DiagnosticsError::NonFiniteSource { index: 0, .. })
        ));
        assert!(matches!(
            reconstruction_metrics(&[0.0], [f32::INFINITY]),
            Err(DiagnosticsError::NonFiniteReconstructed { index: 0, .. })
        ));
    }

    #[test]
    fn counts_saturated_int8_values() {
        assert_eq!(saturation_count(&[-127, -1, 0, 1, 127, 127]), 3);
    }

    #[test]
    fn accounts_for_payload_overhead_and_ratio() {
        let accounting = compression_accounting(1024, 256, 4).expect("the byte counts fit in u64");

        assert_eq!(accounting.total_quantized_bytes, 260);
        assert_eq!(accounting.compression_ratio(), Some(1024.0 / 260.0));
    }

    #[test]
    fn combines_int8_metrics_without_a_reconstructed_buffer() {
        let source = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let quantized = quantize(&source).expect("the fixture is finite");

        let diagnostics = int8_tensor_diagnostics(&source, &quantized, 20, 4)
            .expect("the quantized tensor has matching values");

        assert_eq!(diagnostics.elements, 5);
        assert_eq!(diagnostics.source_bytes, 20);
        assert_eq!(diagnostics.quantized_bytes, 9);
        assert_eq!(diagnostics.saturated_values, 2);
        assert!(diagnostics.mse > 0.0);
        assert!(diagnostics.mae > 0.0);
        assert!(diagnostics.max_abs_error > 0.0);
        assert_eq!(diagnostics.compression_ratio(), Some(20.0 / 9.0));
    }
}
