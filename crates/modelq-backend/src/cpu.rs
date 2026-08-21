//! Portable CPU execution for the scalar quantization representations.
//!
//! The parallel paths compute scales with the scalar reference algorithms,
//! then divide the value payload into a bounded number of disjoint slices.
//! Workers write directly into the final output buffers, so they do not build
//! one temporary allocation per chunk. The scalar implementations remain the
//! correctness oracle and are intentionally not replaced.

use std::{fmt, thread};

use modelq_quant::{int4, int8};

/// Default number of source elements processed by one worker iteration.
pub const DEFAULT_CHUNK_ELEMENTS: usize = int8::DEFAULT_CHUNK_ELEMENTS;

/// Hard upper bound on the number of worker threads created by this backend.
///
/// Callers can request fewer workers through [`ParallelConfig`]. The bound
/// prevents an accidentally huge worker count from turning a small chunk size
/// into an unbounded number of operating-system threads.
pub const MAX_WORKERS: usize = 64;

/// Configuration shared by the parallel INT8 and INT4 CPU paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelConfig {
    /// Maximum number of worker threads to create.
    pub workers: usize,
    /// Number of source elements assigned to one worker iteration.
    pub chunk_elements: usize,
}

impl ParallelConfig {
    /// Creates a parallel configuration.
    pub const fn new(workers: usize, chunk_elements: usize) -> Self {
        Self {
            workers,
            chunk_elements,
        }
    }

    /// Creates a configuration using the host's available parallelism.
    pub fn automatic(chunk_elements: usize) -> Self {
        let workers = thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(MAX_WORKERS);
        Self::new(workers, chunk_elements)
    }

    fn validate(self) -> Result<(), ParallelError> {
        if self.workers == 0 || self.chunk_elements == 0 {
            return Err(ParallelError::InvalidConfig {
                workers: self.workers,
                chunk_elements: self.chunk_elements,
            });
        }
        Ok(())
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::automatic(DEFAULT_CHUNK_ELEMENTS)
    }
}

/// Errors returned by the portable parallel CPU paths.
#[derive(Debug)]
pub enum ParallelError {
    /// A caller supplied a zero-sized worker pool or chunk.
    InvalidConfig {
        /// Requested worker count.
        workers: usize,
        /// Requested source chunk size.
        chunk_elements: usize,
    },
    /// The scalar INT8 reference path rejected the source or result.
    Int8(int8::Int8Error),
    /// The scalar INT4 reference path rejected the source or result.
    Int4(int4::Int4Error),
    /// A worker thread panicked while processing its assigned slice.
    WorkerPanicked,
}

impl fmt::Display for ParallelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig {
                workers,
                chunk_elements,
            } => write!(
                formatter,
                "parallel CPU configuration requires positive workers and chunk size, got workers={workers}, chunk_elements={chunk_elements}"
            ),
            Self::Int8(source) => write!(formatter, "parallel INT8 quantization failed: {source}"),
            Self::Int4(source) => write!(formatter, "parallel INT4 quantization failed: {source}"),
            Self::WorkerPanicked => formatter.write_str("parallel CPU worker panicked"),
        }
    }
}

impl std::error::Error for ParallelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Int8(source) => Some(source),
            Self::Int4(source) => Some(source),
            Self::InvalidConfig { .. } | Self::WorkerPanicked => None,
        }
    }
}

impl From<int8::Int8Error> for ParallelError {
    fn from(source: int8::Int8Error) -> Self {
        Self::Int8(source)
    }
}

impl From<int4::Int4Error> for ParallelError {
    fn from(source: int4::Int4Error) -> Self {
        Self::Int4(source)
    }
}

/// Quantizes F32 values with the symmetric per-tensor INT8 representation.
///
/// Scale calculation is exactly the scalar reference operation. Quantized
/// values are written into one preallocated output buffer by at most
/// `min(config.workers, 64, source_chunks)` workers. Each worker owns a
/// disjoint slice and reuses the configured chunk size while iterating, so the
/// backend adds no per-worker payload allocation and preserves source order.
pub fn quantize_int8(
    values: &[f32],
    config: ParallelConfig,
) -> Result<int8::QuantizedTensor, ParallelError> {
    config.validate()?;
    let scale = int8::scale_for(values.iter().copied())?;
    let mut output = vec![0_i8; values.len()];

    parallel_int8_values(values, &mut output, scale, config)?;

    int8::QuantizedTensor::from_parts(output, scale).map_err(ParallelError::from)
}

/// Quantizes F32 values with symmetric group-wise INT4.
///
/// Group scales are computed by the scalar reference implementation. Workers
/// then pack directly into the final byte buffer, with each byte owning two
/// source values. This keeps odd-length tensors deterministic and avoids an
/// intermediate unpacked output allocation.
pub fn quantize_int4(
    values: &[f32],
    group_size: usize,
    config: ParallelConfig,
) -> Result<int4::QuantizedTensor, ParallelError> {
    config.validate()?;
    int4::validate_group_size(group_size)?;
    let scales = int4::scales_for(values, group_size)?;
    let mut packed = vec![0_u8; values.len().div_ceil(int4::VALUES_PER_BYTE)];

    parallel_int4_bytes(values, &scales, group_size, &mut packed, config)?;

    int4::QuantizedTensor::from_parts(packed, scales, values.len(), group_size)
        .map_err(ParallelError::from)
}

fn parallel_int8_values(
    values: &[f32],
    output: &mut [i8],
    scale: f32,
    config: ParallelConfig,
) -> Result<(), ParallelError> {
    let worker_count = effective_worker_count(values.len(), config);
    if worker_count == 0 {
        return Ok(());
    }

    let partition_size = values.len().div_ceil(worker_count);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (partition, (input, output)) in values
            .chunks(partition_size)
            .zip(output.chunks_mut(partition_size))
            .enumerate()
        {
            let base_index = partition * partition_size;
            handles.push(scope.spawn(move || -> Result<(), ParallelError> {
                for (chunk_offset, (input_chunk, output_chunk)) in input
                    .chunks(config.chunk_elements)
                    .zip(output.chunks_mut(config.chunk_elements))
                    .enumerate()
                {
                    let chunk_base = base_index + chunk_offset * config.chunk_elements;
                    for (offset, (&value, destination)) in
                        input_chunk.iter().zip(output_chunk.iter_mut()).enumerate()
                    {
                        *destination = int8::quantize_value(value, scale, chunk_base + offset)
                            .map_err(ParallelError::from)?;
                    }
                }
                Ok(())
            }));
        }
        join_workers(handles)
    })
}

fn parallel_int4_bytes(
    values: &[f32],
    scales: &[f32],
    group_size: usize,
    packed: &mut [u8],
    config: ParallelConfig,
) -> Result<(), ParallelError> {
    let worker_count = effective_worker_count(values.len(), config);
    if worker_count == 0 {
        return Ok(());
    }

    let partition_size = packed.len().div_ceil(worker_count);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (partition, output) in packed.chunks_mut(partition_size).enumerate() {
            let first_byte = partition * partition_size;
            handles.push(scope.spawn(move || -> Result<(), ParallelError> {
                for (local_byte, destination) in output.iter_mut().enumerate() {
                    let first_index = (first_byte + local_byte) * int4::VALUES_PER_BYTE;
                    let first_scale = scales[first_index / group_size];
                    let first = int4::quantize_value(values[first_index], first_scale, first_index)
                        .map_err(ParallelError::from)?;
                    let second = if let Some(&value) = values.get(first_index + 1) {
                        let index = first_index + 1;
                        int4::quantize_value(value, scales[index / group_size], index)
                            .map_err(ParallelError::from)?
                    } else {
                        0
                    };
                    *destination = (first as u8 & 0x0f) | ((second as u8 & 0x0f) << 4);
                }
                Ok(())
            }));
        }
        join_workers(handles)
    })
}

fn effective_worker_count(elements: usize, config: ParallelConfig) -> usize {
    if elements == 0 {
        return 0;
    }
    let chunks = elements.div_ceil(config.chunk_elements);
    config.workers.min(MAX_WORKERS).min(chunks).max(1)
}

fn join_workers<'scope>(
    handles: Vec<thread::ScopedJoinHandle<'scope, Result<(), ParallelError>>>,
) -> Result<(), ParallelError> {
    for handle in handles {
        match handle.join() {
            Ok(result) => result?,
            Err(_) => return Err(ParallelError::WorkerPanicked),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MAX_WORKERS, ParallelConfig, ParallelError, quantize_int4, quantize_int8};
    use modelq_quant::{int4, int8};

    fn config() -> ParallelConfig {
        ParallelConfig::new(3, 17)
    }

    #[test]
    fn int8_matches_scalar_reference_exactly() {
        let values = (0..10_003)
            .map(|index| (index as f32 - 4_000.0) / 37.0)
            .collect::<Vec<_>>();
        let scalar = int8::quantize(&values).expect("the fixture is finite");
        let parallel = quantize_int8(&values, config()).expect("parallel quantization succeeds");

        assert_eq!(parallel.scale(), scalar.scale());
        assert_eq!(parallel.values(), scalar.values());
    }

    #[test]
    fn int4_matches_scalar_reference_for_odd_lengths_and_crossed_groups() {
        let values = (0..10_007)
            .map(|index| ((index as f32 % 97.0) - 48.0) / 11.0)
            .collect::<Vec<_>>();
        let scalar = int4::quantize(&values, 31).expect("the fixture is finite");
        let parallel =
            quantize_int4(&values, 31, config()).expect("parallel quantization succeeds");

        assert_eq!(parallel.packed_values(), scalar.packed_values());
        assert_eq!(parallel.scales(), scalar.scales());
        assert_eq!(parallel.len(), scalar.len());
        assert_eq!(parallel.group_size(), scalar.group_size());
    }

    #[test]
    fn empty_inputs_use_the_scalar_zero_conventions_without_threads() {
        let int8 = quantize_int8(&[], config()).expect("empty INT8 input is valid");
        assert_eq!(int8.scale(), 1.0);
        assert!(int8.is_empty());

        let int4 = quantize_int4(&[], 8, config()).expect("empty INT4 input is valid");
        assert!(int4.is_empty());
        assert!(int4.scales().is_empty());
    }

    #[test]
    fn invalid_configuration_is_rejected_before_processing() {
        let error =
            quantize_int8(&[1.0], ParallelConfig::new(0, 4)).expect_err("zero workers are invalid");
        assert!(matches!(
            error,
            ParallelError::InvalidConfig {
                workers: 0,
                chunk_elements: 4
            }
        ));

        let error = quantize_int4(&[1.0], 2, ParallelConfig::new(2, 0))
            .expect_err("zero chunks are invalid");
        assert!(matches!(
            error,
            ParallelError::InvalidConfig {
                workers: 2,
                chunk_elements: 0
            }
        ));
    }

    #[test]
    fn requested_worker_count_is_hard_bounded() {
        let values = vec![1.0_f32; MAX_WORKERS * 2 + 1];
        let result = quantize_int8(&values, ParallelConfig::new(usize::MAX, 1))
            .expect("bounded worker count should still quantize");
        assert_eq!(result.len(), values.len());
    }
}
