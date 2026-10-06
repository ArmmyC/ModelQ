//! Bounded parallel CPU execution for the streaming INT8 CLI path.
//!
//! Three entry points mirror the scalar streaming functions:
//!
//! - [`quantize_replay_chunks`] (writer path);
//! - [`tensor_diagnostics_replay`] (progress diagnostics); and
//! - [`validate_and_measure`] (post-write validation and metrics).
//!
//! Each reads a chunk of the replayable source on the calling thread, then
//! divides the chunk among a bounded number of scoped workers that write
//! disjoint slices.  Quantization is elementwise through the scalar
//! [`modelq_quant::int8`] functions and the tensor maximum is order
//! independent, so quantized bytes and scales are bit-identical to the scalar
//! path for any worker count.
//!
//! Error sums are the one order-sensitive quantity.  They are accumulated over
//! fixed blocks of [`METRICS_BLOCK_ELEMENTS`] values and the block totals are
//! merged in index order, so reported metrics depend only on that constant,
//! never on the worker count or chunk size.  For tensors of at most one block
//! they equal the scalar left-to-right sums exactly; for larger tensors they
//! can differ from the scalar sums in the last few bits of the `f64` result.

use std::fmt;

use modelq_quant::{
    diagnostics::{
        DiagnosticsError, MetricsAccumulator, ReconstructionMetrics, TensorDiagnostics,
        compression_accounting,
    },
    int8::{
        Int8Error, SYMMETRIC_MAX, SYMMETRIC_MIN, quantize_chunk_into, scale_from_max_abs,
        scan_chunk_max_abs,
    },
};

use crate::{
    cpu::{MAX_WORKERS, ParallelConfig},
    schedule::{self, RANGES_PER_WORKER, WorkerPanicked},
};

/// Default values per chunk (4,194,304 values: 16 MiB of `f32`).
pub const DEFAULT_CHUNK_ELEMENTS: usize = 1 << 22;

/// Values per error-sum block; the unit of deterministic summation.
pub const METRICS_BLOCK_ELEMENTS: usize = 4096;

/// Blocks below which an extra worker is not worth its start-up cost.
const MIN_BLOCKS_PER_WORKER: usize = 4;

/// Errors returned by the parallel INT8 paths.  `E` is the caller's callback
/// error (use [`std::convert::Infallible`] where there is none).
#[derive(Debug)]
pub enum Int8ParallelError<E> {
    /// Zero workers or a chunk smaller than one metrics block.
    InvalidConfig {
        workers: usize,
        chunk_elements: usize,
    },
    /// The scalar INT8 rules rejected a value, scale, or quantized byte.
    Quantization(Int8Error),
    /// Metrics could not be computed or the inputs differ in length.
    Diagnostics(DiagnosticsError),
    /// The caller's chunk callback failed.
    Callback(E),
    /// A worker thread panicked.
    WorkerPanicked,
}

impl<E: fmt::Display> fmt::Display for Int8ParallelError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig {
                workers,
                chunk_elements,
            } => write!(
                formatter,
                "parallel INT8 requires at least one worker and a chunk of at least {METRICS_BLOCK_ELEMENTS} values, got workers={workers}, chunk_elements={chunk_elements}"
            ),
            Self::Quantization(error) => error.fmt(formatter),
            Self::Diagnostics(error) => error.fmt(formatter),
            Self::Callback(error) => error.fmt(formatter),
            Self::WorkerPanicked => formatter.write_str("parallel INT8 worker panicked"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for Int8ParallelError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Quantization(error) => Some(error),
            Self::Diagnostics(error) => Some(error),
            Self::Callback(error) => Some(error),
            Self::InvalidConfig { .. } | Self::WorkerPanicked => None,
        }
    }
}

struct Settings {
    workers: usize,
    chunk_values: usize,
}

fn settings<E>(config: ParallelConfig) -> Result<Settings, Int8ParallelError<E>> {
    if config.workers == 0 || config.chunk_elements < METRICS_BLOCK_ELEMENTS {
        return Err(Int8ParallelError::InvalidConfig {
            workers: config.workers,
            chunk_elements: config.chunk_elements,
        });
    }
    Ok(Settings {
        workers: config.workers.min(MAX_WORKERS),
        // Whole blocks per chunk, so block boundaries never depend on it.
        chunk_values: config.chunk_elements / METRICS_BLOCK_ELEMENTS * METRICS_BLOCK_ELEMENTS,
    })
}

/// Values per range: a whole number of metrics blocks.
fn range_len(len: usize, workers: usize) -> usize {
    let blocks = len.div_ceil(METRICS_BLOCK_ELEMENTS);
    let used = (workers * RANGES_PER_WORKER)
        .min(blocks.div_ceil(MIN_BLOCKS_PER_WORKER))
        .max(1);
    blocks.div_ceil(used) * METRICS_BLOCK_ELEMENTS
}

/// Runs `work` on every item through the shared dynamic scheduler.
fn run_all<T, R, E>(
    items: Vec<T>,
    threads: usize,
    work: impl Fn(usize, T) -> R + Sync,
) -> Result<Vec<R>, Int8ParallelError<E>>
where
    T: Send,
    R: Send,
{
    schedule::run_all(items, threads, work)
        .map_err(|WorkerPanicked| Int8ParallelError::WorkerPanicked)
}

/// Computes the tensor scale from the largest absolute value over all
/// chunks, reporting the first non-finite value by tensor-wide index.
fn scale_pass<F, I, E>(values: &mut F, settings: &Settings) -> Result<f32, Int8ParallelError<E>>
where
    F: FnMut() -> I,
    I: IntoIterator<Item = f32>,
{
    let mut buffer: Vec<f32> = Vec::with_capacity(settings.chunk_values);
    let mut source = values().into_iter();
    let mut max_abs = 0.0_f32;
    let mut seen = 0_usize;
    loop {
        buffer.clear();
        buffer.extend(source.by_ref().take(settings.chunk_values));
        if buffer.is_empty() {
            break;
        }
        let per_range = range_len(buffer.len(), settings.workers);
        let first_index = seen;
        let results = run_all(
            buffer.chunks(per_range).collect(),
            settings.workers,
            |index, range| scan_chunk_max_abs(range, first_index + index * per_range),
        )?;
        // Ranges are in index order, so the first error is the lowest index.
        for result in results {
            max_abs = max_abs.max(result.map_err(Int8ParallelError::Quantization)?);
        }
        seen += buffer.len();
    }
    Ok(scale_from_max_abs(max_abs))
}

/// Quantizes a replayable source in bounded memory and emits the `i8` values
/// in order, returning the tensor scale.
///
/// Same contract and identical output as
/// [`modelq_quant::int8::quantize_replay_chunks`]; `config.chunk_elements` is
/// rounded down to whole 4096-value blocks.
pub fn quantize_replay_chunks<F, I, C, E>(
    mut values: F,
    config: ParallelConfig,
    mut emit: C,
) -> Result<f32, Int8ParallelError<E>>
where
    F: FnMut() -> I,
    I: IntoIterator<Item = f32>,
    C: FnMut(&[i8]) -> Result<(), E>,
{
    let settings = settings(config)?;
    let scale = scale_pass(&mut values, &settings)?;

    let mut buffer: Vec<f32> = Vec::with_capacity(settings.chunk_values);
    let mut quantized: Vec<i8> = Vec::with_capacity(settings.chunk_values);
    let mut source = values().into_iter();
    let mut start = 0_usize;
    loop {
        buffer.clear();
        buffer.extend(source.by_ref().take(settings.chunk_values));
        if buffer.is_empty() {
            break;
        }
        quantized.resize(buffer.len(), 0);
        let per_range = range_len(buffer.len(), settings.workers);
        let first_index = start;
        let items: Vec<_> = buffer
            .chunks(per_range)
            .zip(quantized.chunks_mut(per_range))
            .collect();
        let results = run_all(items, settings.workers, |index, (range, output)| {
            quantize_chunk_into(range, scale, first_index + index * per_range, output)
        })?;
        for result in results {
            result.map_err(Int8ParallelError::Quantization)?;
        }
        start += buffer.len();
        emit(&quantized).map_err(Int8ParallelError::Callback)?;
    }
    Ok(scale)
}

/// Per-block partial results of the diagnostics pass.
#[derive(Default)]
struct BlockPartial {
    metrics: MetricsAccumulator,
    saturated: u64,
}

/// Computes the INT8 scale and tensor diagnostics with bounded memory.
///
/// Same contract as [`modelq_quant::diagnostics::int8_tensor_diagnostics_replay`]
/// except for the summation order described in the module documentation.
pub fn tensor_diagnostics_replay<F, I>(
    mut values: F,
    config: ParallelConfig,
    source_bytes: u64,
    scale_bytes: u64,
) -> Result<(f32, TensorDiagnostics), Int8ParallelError<std::convert::Infallible>>
where
    F: FnMut() -> I,
    I: IntoIterator<Item = f32>,
{
    let settings = settings(config)?;
    let scale = scale_pass(&mut values, &settings)?;

    let mut buffer: Vec<f32> = Vec::with_capacity(settings.chunk_values);
    let mut quantized: Vec<i8> = Vec::with_capacity(settings.chunk_values);
    let mut total = MetricsAccumulator::default();
    let mut saturated_values = 0_u64;
    let mut source = values().into_iter();
    let mut start = 0_usize;
    loop {
        buffer.clear();
        buffer.extend(source.by_ref().take(settings.chunk_values));
        if buffer.is_empty() {
            break;
        }
        quantized.resize(buffer.len(), 0);
        let per_range = range_len(buffer.len(), settings.workers);
        let first_index = start;
        let items: Vec<_> = buffer
            .chunks(per_range)
            .zip(quantized.chunks_mut(per_range))
            .collect();
        let results = run_all(items, settings.workers, |index, (range, output)| {
            let range_start = first_index + index * per_range;
            quantize_chunk_into(range, scale, range_start, output)
                .map_err(Int8ParallelError::Quantization)?;
            let mut partials = Vec::new();
            for (block, (values, quantized)) in range
                .chunks(METRICS_BLOCK_ELEMENTS)
                .zip(output.chunks(METRICS_BLOCK_ELEMENTS))
                .enumerate()
            {
                let block_start = range_start + block * METRICS_BLOCK_ELEMENTS;
                let mut partial = BlockPartial::default();
                for (offset, (&source_value, &quantized_value)) in
                    values.iter().zip(quantized).enumerate()
                {
                    partial
                        .metrics
                        .push_at(
                            block_start + offset,
                            source_value,
                            f32::from(quantized_value) * scale,
                        )
                        .map_err(Int8ParallelError::Diagnostics)?;
                    if quantized_value == SYMMETRIC_MIN || quantized_value == SYMMETRIC_MAX {
                        partial.saturated += 1;
                    }
                }
                partials.push(partial);
            }
            Ok::<_, Int8ParallelError<std::convert::Infallible>>(partials)
        })?;
        for result in results {
            for partial in result? {
                total.merge(&partial.metrics);
                saturated_values += partial.saturated;
            }
        }
        start += buffer.len();
    }

    let quantized_payload_bytes = u64::try_from(start)
        .map_err(|_| Int8ParallelError::Diagnostics(DiagnosticsError::ElementCountOverflow))?;
    let metrics = total.finish().map_err(Int8ParallelError::Diagnostics)?;
    let accounting = compression_accounting(source_bytes, quantized_payload_bytes, scale_bytes)
        .map_err(Int8ParallelError::Diagnostics)?;
    Ok((
        scale,
        TensorDiagnostics {
            elements: metrics.elements,
            source_bytes: accounting.source_bytes,
            quantized_bytes: accounting.total_quantized_bytes,
            mse: metrics.mse,
            mae: metrics.mae,
            max_abs_error: metrics.max_abs_error,
            sqnr_db: metrics.sqnr_db,
            saturated_values,
        },
    ))
}

/// Validates written INT8 bytes and measures their reconstruction error.
///
/// First checks every byte is within the symmetric range and decodes to a
/// finite value (as [`modelq_quant::int8::validate_dequantization`] does, with
/// the lowest offending index reported), then compares the replayed source
/// with the reconstruction.  Returns the metrics and the number of values at
/// either symmetric endpoint.
pub fn validate_and_measure<F, I>(
    mut values: F,
    quantized: &[u8],
    scale: f32,
    config: ParallelConfig,
) -> Result<(ReconstructionMetrics, u64), Int8ParallelError<std::convert::Infallible>>
where
    F: FnMut() -> I,
    I: IntoIterator<Item = f32>,
{
    let settings = settings(config)?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Int8ParallelError::Quantization(Int8Error::InvalidScale {
            scale,
        }));
    }

    // Range and finiteness check of the whole payload, in parallel.
    let per_range = range_len(quantized.len(), settings.workers);
    let checks = run_all(
        quantized.chunks(per_range).collect(),
        settings.workers,
        |index, range| {
            for (offset, &byte) in range.iter().enumerate() {
                let value = byte as i8;
                let index = index * per_range + offset;
                if !(SYMMETRIC_MIN..=SYMMETRIC_MAX).contains(&value) {
                    return Err(Int8Error::QuantizedValueOutOfRange { index, value });
                }
                if !(f32::from(value) * scale).is_finite() {
                    return Err(Int8Error::DequantizedValueOverflow { index });
                }
            }
            Ok(())
        },
    )?;
    for check in checks {
        check.map_err(Int8ParallelError::Quantization)?;
    }

    let mut buffer: Vec<f32> = Vec::with_capacity(settings.chunk_values.min(quantized.len()));
    let mut total = MetricsAccumulator::default();
    let mut saturated_values = 0_u64;
    let mut source = values().into_iter();
    let mut start = 0_usize;
    loop {
        buffer.clear();
        buffer.extend(source.by_ref().take(settings.chunk_values));
        if buffer.is_empty() {
            break;
        }
        let end = start + buffer.len();
        let Some(chunk_quantized) = quantized.get(start..end) else {
            let source_len = start + buffer.len() + source.by_ref().count();
            return Err(Int8ParallelError::Diagnostics(
                DiagnosticsError::LengthMismatch {
                    source_len,
                    reconstructed_len: quantized.len(),
                },
            ));
        };
        let per_range = range_len(buffer.len(), settings.workers);
        let first_index = start;
        let items: Vec<_> = buffer
            .chunks(per_range)
            .zip(chunk_quantized.chunks(per_range))
            .collect();
        let results = run_all(items, settings.workers, |index, (range, bytes)| {
            let range_start = first_index + index * per_range;
            let mut partials = Vec::new();
            for (block, (values, bytes)) in range
                .chunks(METRICS_BLOCK_ELEMENTS)
                .zip(bytes.chunks(METRICS_BLOCK_ELEMENTS))
                .enumerate()
            {
                let block_start = range_start + block * METRICS_BLOCK_ELEMENTS;
                let mut partial = BlockPartial::default();
                for (offset, (&source_value, &byte)) in values.iter().zip(bytes).enumerate() {
                    let value = byte as i8;
                    partial
                        .metrics
                        .push_at(block_start + offset, source_value, f32::from(value) * scale)
                        .map_err(Int8ParallelError::Diagnostics)?;
                    if value == SYMMETRIC_MIN || value == SYMMETRIC_MAX {
                        partial.saturated += 1;
                    }
                }
                partials.push(partial);
            }
            Ok::<_, Int8ParallelError<std::convert::Infallible>>(partials)
        })?;
        for result in results {
            for partial in result? {
                total.merge(&partial.metrics);
                saturated_values += partial.saturated;
            }
        }
        start = end;
    }
    if start != quantized.len() {
        return Err(Int8ParallelError::Diagnostics(
            DiagnosticsError::LengthMismatch {
                source_len: start,
                reconstructed_len: quantized.len(),
            },
        ));
    }

    let metrics = total.finish().map_err(Int8ParallelError::Diagnostics)?;
    Ok((metrics, saturated_values))
}

#[cfg(test)]
mod tests {
    use modelq_quant::{
        diagnostics::{
            DiagnosticsError, int8_tensor_diagnostics_replay, reconstruction_metrics_streaming,
        },
        int8::{self, Int8Error},
    };

    use super::{
        Int8ParallelError, METRICS_BLOCK_ELEMENTS, quantize_replay_chunks,
        tensor_diagnostics_replay, validate_and_measure,
    };
    use crate::cpu::ParallelConfig;

    fn spread(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0;
                unit * 2.0_f32.powi(((state >> 8) % 12) as i32 - 6)
            })
            .collect()
    }

    fn scalar_quantize(values: &[f32]) -> (Vec<i8>, f32) {
        let mut bytes = Vec::new();
        let scale = int8::quantize_replay_chunks(
            || values.iter().copied(),
            1024,
            |chunk| {
                bytes.extend_from_slice(chunk);
                Ok::<(), ()>(())
            },
        )
        .expect("scalar succeeds");
        (bytes, scale)
    }

    const CONFIGS: [(usize, usize); 6] = [
        (1, METRICS_BLOCK_ELEMENTS),
        (2, METRICS_BLOCK_ELEMENTS),
        (3, 3 * METRICS_BLOCK_ELEMENTS),
        (7, 70_000),
        (18, 1 << 22),
        (1000, METRICS_BLOCK_ELEMENTS),
    ];

    #[test]
    fn quantization_matches_the_scalar_path_byte_for_byte() {
        for count in [1, 4095, 4096, 4097, 70_001, 400_000] {
            let values = spread(count, count as u64);
            let (expected, expected_scale) = scalar_quantize(&values);
            for (workers, chunk_elements) in CONFIGS {
                let mut bytes = Vec::new();
                let scale = quantize_replay_chunks(
                    || values.iter().copied(),
                    ParallelConfig::new(workers, chunk_elements),
                    |chunk| -> Result<(), ()> {
                        bytes.extend_from_slice(chunk);
                        Ok(())
                    },
                )
                .expect("parallel succeeds");
                let label = format!("count={count} workers={workers} chunk={chunk_elements}");
                assert_eq!(scale.to_bits(), expected_scale.to_bits(), "{label}");
                assert_eq!(bytes, expected, "{label}");
            }
        }
    }

    #[test]
    fn zero_and_extreme_tensors_match_the_scalar_path() {
        for values in [vec![0.0_f32; 9000], {
            let mut v = vec![1e-30_f32; 9000];
            v[8999] = 3.0e38;
            v
        }] {
            let (expected, expected_scale) = scalar_quantize(&values);
            let mut bytes = Vec::new();
            let scale = quantize_replay_chunks(
                || values.iter().copied(),
                ParallelConfig::new(4, METRICS_BLOCK_ELEMENTS),
                |chunk| -> Result<(), ()> {
                    bytes.extend_from_slice(chunk);
                    Ok(())
                },
            )
            .expect("parallel succeeds");
            assert_eq!(scale.to_bits(), expected_scale.to_bits());
            assert_eq!(bytes, expected);
        }
    }

    #[test]
    fn reports_the_first_non_finite_index_regardless_of_workers() {
        let mut values = spread(200_000, 5);
        values[123_456] = f32::NAN;
        values[150_000] = f32::INFINITY;
        for workers in [1, 4, 18] {
            let error = quantize_replay_chunks(
                || values.iter().copied(),
                ParallelConfig::new(workers, METRICS_BLOCK_ELEMENTS),
                |_| Ok::<(), ()>(()),
            )
            .expect_err("non-finite input is rejected");
            assert!(
                matches!(
                    error,
                    Int8ParallelError::Quantization(Int8Error::NonFiniteInput {
                        index: 123_456,
                        ..
                    })
                ),
                "workers={workers}: {error:?}"
            );
        }
    }

    #[test]
    fn diagnostics_are_identical_for_every_worker_count_and_close_to_scalar() {
        let values = spread(300_000, 11);
        let source_bytes = (values.len() * 4) as u64;
        let (scalar_scale, scalar) =
            int8_tensor_diagnostics_replay(|| values.iter().copied(), 1024, source_bytes, 4)
                .expect("scalar succeeds");

        let mut results = Vec::new();
        for (workers, chunk_elements) in CONFIGS {
            let (scale, diagnostics) = tensor_diagnostics_replay(
                || values.iter().copied(),
                ParallelConfig::new(workers, chunk_elements),
                source_bytes,
                4,
            )
            .expect("parallel succeeds");
            assert_eq!(scale.to_bits(), scalar_scale.to_bits());
            assert_eq!(diagnostics.saturated_values, scalar.saturated_values);
            assert_eq!(
                diagnostics.max_abs_error.to_bits(),
                scalar.max_abs_error.to_bits()
            );
            assert_eq!(diagnostics.quantized_bytes, scalar.quantized_bytes);
            for (parallel, reference) in
                [(diagnostics.mse, scalar.mse), (diagnostics.mae, scalar.mae)]
            {
                assert!(
                    (parallel - reference).abs() <= reference.abs() * 1e-12,
                    "{parallel} vs {reference}"
                );
            }
            results.push(diagnostics);
        }
        // Block summation: bitwise identical regardless of workers and chunks.
        for diagnostics in &results {
            assert_eq!(diagnostics.mse.to_bits(), results[0].mse.to_bits());
            assert_eq!(diagnostics.mae.to_bits(), results[0].mae.to_bits());
            assert_eq!(
                diagnostics.sqnr_db.map(f64::to_bits),
                results[0].sqnr_db.map(f64::to_bits)
            );
        }
    }

    #[test]
    fn single_block_tensors_equal_the_scalar_sums_exactly() {
        let values = spread(METRICS_BLOCK_ELEMENTS, 3);
        let (_, scalar) = int8_tensor_diagnostics_replay(|| values.iter().copied(), 1024, 1, 4)
            .expect("scalar succeeds");
        let (_, parallel) = tensor_diagnostics_replay(
            || values.iter().copied(),
            ParallelConfig::new(8, METRICS_BLOCK_ELEMENTS),
            1,
            4,
        )
        .expect("parallel succeeds");
        assert_eq!(parallel.mse.to_bits(), scalar.mse.to_bits());
        assert_eq!(parallel.mae.to_bits(), scalar.mae.to_bits());
        assert_eq!(
            parallel.sqnr_db.map(f64::to_bits),
            scalar.sqnr_db.map(f64::to_bits)
        );
    }

    #[test]
    fn validation_matches_scalar_metrics_and_flags_bad_bytes() {
        let values = spread(150_000, 21);
        let (quantized, scale) = scalar_quantize(&values);
        let bytes: Vec<u8> = quantized.iter().map(|&value| value as u8).collect();
        let scalar = reconstruction_metrics_streaming(
            values.iter().copied(),
            quantized.iter().map(|&value| f32::from(value) * scale),
        )
        .expect("scalar succeeds");

        let mut reference = None;
        for (workers, chunk_elements) in CONFIGS {
            let (metrics, saturated) = validate_and_measure(
                || values.iter().copied(),
                &bytes,
                scale,
                ParallelConfig::new(workers, chunk_elements),
            )
            .expect("valid payload");
            assert_eq!(metrics.elements, scalar.elements);
            assert_eq!(
                metrics.max_abs_error.to_bits(),
                scalar.max_abs_error.to_bits()
            );
            assert!((metrics.mse - scalar.mse).abs() <= scalar.mse * 1e-12);
            assert_eq!(
                saturated,
                quantized.iter().filter(|&&v| v == -127 || v == 127).count() as u64
            );
            let key = (metrics.mse.to_bits(), metrics.mae.to_bits());
            assert_eq!(*reference.get_or_insert(key), key);
        }

        // -128 is outside the symmetric range; the lowest index is reported.
        let mut corrupt = bytes.clone();
        corrupt[90_000] = 0x80;
        corrupt[120_000] = 0x80;
        for workers in [1, 8] {
            let error = validate_and_measure(
                || values.iter().copied(),
                &corrupt,
                scale,
                ParallelConfig::new(workers, METRICS_BLOCK_ELEMENTS),
            )
            .expect_err("-128 is invalid");
            assert!(
                matches!(
                    error,
                    Int8ParallelError::Quantization(Int8Error::QuantizedValueOutOfRange {
                        index: 90_000,
                        value: -128
                    })
                ),
                "{error:?}"
            );
        }

        let short = validate_and_measure(
            || values.iter().copied(),
            &bytes[..1000],
            scale,
            ParallelConfig::new(2, METRICS_BLOCK_ELEMENTS),
        )
        .expect_err("length mismatch");
        assert!(matches!(
            short,
            Int8ParallelError::Diagnostics(DiagnosticsError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn rejects_invalid_configuration_and_propagates_callback_errors() {
        let values = spread(10_000, 1);
        for config in [ParallelConfig::new(0, 8192), ParallelConfig::new(2, 100)] {
            assert!(matches!(
                quantize_replay_chunks(|| values.iter().copied(), config, |_| Ok::<(), ()>(())),
                Err(Int8ParallelError::InvalidConfig { .. })
            ));
        }
        let error = quantize_replay_chunks(
            || values.iter().copied(),
            ParallelConfig::new(2, METRICS_BLOCK_ELEMENTS),
            |_| Err("sink failed"),
        )
        .expect_err("callback failure is surfaced");
        assert!(matches!(error, Int8ParallelError::Callback("sink failed")));
    }
}
