//! Numerical reconstruction and compression diagnostics.
//!
//! Reconstruction metrics accept an iterator for the reconstructed values, so
//! callers can measure a mapped or freshly dequantized tensor without first
//! allocating a duplicate `Vec<f32>`.

use std::fmt;

use crate::quant::int8::{
    Int8Error, QuantizationStreamError, QuantizedTensor, SYMMETRIC_MAX, SYMMETRIC_MIN,
    quantize_replay_chunks_with_scale, scale_for,
};

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
    /// The bounded INT8 path rejected a source value or scale.
    Quantization { source: Int8Error },
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
            Self::Quantization { source } => {
                write!(
                    formatter,
                    "could not quantize values for diagnostics: {source}"
                )
            }
        }
    }
}

impl std::error::Error for DiagnosticsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Quantization { source } => Some(source),
            _ => None,
        }
    }
}

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
    reconstruction_metrics_streaming(source.iter().copied(), reconstructed)
}

/// Calculates reconstruction metrics from one-pass source and reconstructed
/// iterators without collecting either sequence into a `Vec<f32>`.
pub fn reconstruction_metrics_streaming<S, R>(
    source: S,
    reconstructed: R,
) -> Result<ReconstructionMetrics, DiagnosticsError>
where
    S: IntoIterator<Item = f32>,
    R: IntoIterator<Item = f32>,
{
    let mut source = source.into_iter();
    let mut reconstructed = reconstructed.into_iter();
    let mut accumulator = MetricsAccumulator::default();

    loop {
        match (source.next(), reconstructed.next()) {
            (Some(source_value), Some(reconstructed_value)) => {
                accumulator.push(source_value, reconstructed_value)?;
            }
            (None, None) => return accumulator.finish(),
            (Some(_), None) => {
                let source_len = accumulator
                    .elements
                    .saturating_add(1)
                    .saturating_add(source.count());
                return Err(DiagnosticsError::LengthMismatch {
                    source_len,
                    reconstructed_len: accumulator.elements,
                });
            }
            (None, Some(_)) => {
                let reconstructed_len = accumulator
                    .elements
                    .saturating_add(1)
                    .saturating_add(reconstructed.count());
                return Err(DiagnosticsError::LengthMismatch {
                    source_len: accumulator.elements,
                    reconstructed_len,
                });
            }
        }
    }
}

/// Counts values that reached either endpoint of the symmetric INT8 range.
pub fn saturation_count(values: &[i8]) -> u64 {
    saturation_count_iter(values.iter().copied())
}

/// Counts saturated values from a one-pass quantized iterator.
pub fn saturation_count_iter<I>(values: I) -> u64
where
    I: IntoIterator<Item = i8>,
{
    values
        .into_iter()
        .filter(|&value| value == SYMMETRIC_MIN || value == SYMMETRIC_MAX)
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

/// Calculates INT8 diagnostics with bounded replay processing.
///
/// The value factory is called once to compute the per-tensor scale and once
/// to process bounded quantized chunks. Source and reconstructed values never
/// need a whole-tensor `Vec<f32>` allocation.
pub fn int8_tensor_diagnostics_replay<F, I>(
    mut values: F,
    chunk_size: usize,
    source_bytes: u64,
    scale_bytes: u64,
) -> Result<(f32, TensorDiagnostics), DiagnosticsError>
where
    F: FnMut() -> I,
    I: IntoIterator<Item = f32>,
{
    let scale = scale_for(values()).map_err(|source| DiagnosticsError::Quantization { source })?;
    let mut source_values = values().into_iter();
    let mut accumulator = MetricsAccumulator::default();
    let mut quantized_payload_bytes = 0_u64;
    let mut saturated_values = 0_u64;

    let stream_result = quantize_replay_chunks_with_scale(
        &mut values,
        scale,
        chunk_size,
        |chunk| -> Result<(), DiagnosticsError> {
            for &quantized in chunk {
                let Some(source_value) = source_values.next() else {
                    return Err(DiagnosticsError::LengthMismatch {
                        source_len: accumulator.elements,
                        reconstructed_len: accumulator.elements.saturating_add(1),
                    });
                };
                accumulator.push(source_value, f32::from(quantized) * scale)?;
                quantized_payload_bytes = quantized_payload_bytes
                    .checked_add(1)
                    .ok_or(DiagnosticsError::ByteCountOverflow)?;
                if quantized == SYMMETRIC_MIN || quantized == SYMMETRIC_MAX {
                    saturated_values = saturated_values.saturating_add(1);
                }
            }
            Ok(())
        },
    );
    match stream_result {
        Ok(()) => {}
        Err(QuantizationStreamError::Quantization(source)) => {
            return Err(DiagnosticsError::Quantization { source });
        }
        Err(QuantizationStreamError::Callback(source)) => return Err(source),
    }

    if source_values.next().is_some() {
        let source_len = accumulator
            .elements
            .saturating_add(1)
            .saturating_add(source_values.count());
        return Err(DiagnosticsError::LengthMismatch {
            source_len,
            reconstructed_len: accumulator.elements,
        });
    }

    let metrics = accumulator.finish()?;
    let accounting = compression_accounting(source_bytes, quantized_payload_bytes, scale_bytes)?;
    Ok((
        scale,
        TensorDiagnostics {
            elements: metrics.elements,
            source_bytes,
            quantized_bytes: accounting.total_quantized_bytes,
            mse: metrics.mse,
            mae: metrics.mae,
            max_abs_error: metrics.max_abs_error,
            sqnr_db: metrics.sqnr_db,
            saturated_values,
        },
    ))
}

#[derive(Default)]
struct MetricsAccumulator {
    elements: usize,
    squared_error_sum: f64,
    absolute_error_sum: f64,
    source_energy: f64,
    max_abs_error: f64,
}

impl MetricsAccumulator {
    fn push(
        &mut self,
        source_value: f32,
        reconstructed_value: f32,
    ) -> Result<(), DiagnosticsError> {
        let index = self.elements;
        if !source_value.is_finite() {
            return Err(DiagnosticsError::NonFiniteSource {
                index,
                value: source_value,
            });
        }
        if !reconstructed_value.is_finite() {
            return Err(DiagnosticsError::NonFiniteReconstructed {
                index,
                value: reconstructed_value,
            });
        }

        let source_value = f64::from(source_value);
        let reconstructed_value = f64::from(reconstructed_value);
        let absolute_error = (source_value - reconstructed_value).abs();
        self.squared_error_sum += absolute_error * absolute_error;
        self.absolute_error_sum += absolute_error;
        self.source_energy += source_value * source_value;
        self.max_abs_error = self.max_abs_error.max(absolute_error);
        self.elements = self.elements.saturating_add(1);
        Ok(())
    }

    fn finish(self) -> Result<ReconstructionMetrics, DiagnosticsError> {
        let elements =
            u64::try_from(self.elements).map_err(|_| DiagnosticsError::ElementCountOverflow)?;
        let elements_f64 = elements as f64;
        let mse = if elements == 0 {
            0.0
        } else {
            self.squared_error_sum / elements_f64
        };
        let mae = if elements == 0 {
            0.0
        } else {
            self.absolute_error_sum / elements_f64
        };
        let sqnr_db = (self.source_energy > 0.0 && self.squared_error_sum > 0.0)
            .then(|| 10.0 * (self.source_energy / self.squared_error_sum).log10());

        Ok(ReconstructionMetrics {
            elements,
            mse,
            mae,
            max_abs_error: self.max_abs_error,
            sqnr_db,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DiagnosticsError, compression_accounting, int8_tensor_diagnostics,
        int8_tensor_diagnostics_replay, reconstruction_metrics, reconstruction_metrics_streaming,
        saturation_count,
    };
    use crate::quant::int8::{DEFAULT_CHUNK_ELEMENTS, quantize};

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

    #[test]
    fn replay_diagnostics_match_reference_for_a_large_tensor() {
        let source = (0..DEFAULT_CHUNK_ELEMENTS + 19)
            .map(|index| (index as f32 - 300.0) / 17.0)
            .collect::<Vec<_>>();
        let quantized = quantize(&source).expect("the generated values are finite");
        let reference = int8_tensor_diagnostics(&source, &quantized, (source.len() * 4) as u64, 4)
            .expect("the reference diagnostics are valid");
        let (scale, replayed) = int8_tensor_diagnostics_replay(
            || source.iter().copied(),
            257,
            (source.len() * 4) as u64,
            4,
        )
        .expect("bounded diagnostics are valid");

        assert_eq!(scale, quantized.scale());
        assert_eq!(replayed, reference);
        assert_eq!(
            reconstruction_metrics_streaming(
                source.iter().copied(),
                quantized
                    .values()
                    .iter()
                    .map(|&value| f32::from(value) * quantized.scale()),
            )
            .expect("streaming metrics are valid"),
            reconstruction_metrics(
                &source,
                quantized
                    .values()
                    .iter()
                    .map(|&value| f32::from(value) * quantized.scale()),
            )
            .expect("reference metrics are valid")
        );
    }
}
