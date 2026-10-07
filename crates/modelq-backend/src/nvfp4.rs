//! Bounded parallel CPU NVFP4 quantization.
//!
//! [`quantize_replay_chunks`] has the contract of
//! [`modelq_quant::nvfp4::quantize_replay_chunks`]: the value source is read
//! twice (tensor-wide amax, then data) and packed bytes are emitted in order.
//! Within each chunk the whole blocks are divided among a bounded number of
//! scoped worker threads, each writing a disjoint slice of the chunk's packed
//! bytes and block scales through the shared
//! [`modelq_quant::nvfp4::encode_chunk`] kernel.  Because every worker runs the
//! reference kernel on block-aligned ranges and the global amax is exact
//! (`max` is order-independent), the output is bit-identical to the scalar
//! path for any worker count and chunk size.
//!
//! The source iterator is consumed on the calling thread, so it does not need
//! to be `Send`.  Memory is one chunk of `f32` values plus its packed bytes,
//! and the block scales (one byte per 16 values).

use std::fmt;

use modelq_quant::nvfp4::{
    self, BLOCK_SIZE, Nvfp4Error, ScaleSelection, StreamedQuantization, block_count,
    encode_chunk_with, packed_len, scan_chunk_amax,
};

use crate::{
    cpu::{MAX_WORKERS, ParallelConfig},
    prefetch::with_feed,
    schedule::{self, RANGES_PER_WORKER, WorkerPanicked},
};

/// Default values per chunk (2,097,152 values: 8 MiB of `f32`).
///
/// With the reader thread of [`crate::prefetch`] a smaller chunk gives a finer
/// pipeline and smaller buffers; measured best between 1M and 4M values, while
/// still large enough that per-chunk thread start-up stays small.
pub const DEFAULT_CHUNK_ELEMENTS: usize = 1 << 21;

/// Fewest blocks in a range, so a range is always worth scheduling.
const MIN_BLOCKS_PER_RANGE: usize = 64;

/// Errors returned by the parallel NVFP4 path.  `E` is the caller's callback
/// error.
#[derive(Debug)]
pub enum Nvfp4ParallelError<E> {
    /// Zero workers or a chunk smaller than one block.
    InvalidConfig {
        workers: usize,
        chunk_elements: usize,
    },
    /// The source values or shape were rejected by the scalar reference rules.
    Quantization(Nvfp4Error),
    /// The caller's chunk callback failed.
    Callback(E),
    /// A worker thread panicked.
    WorkerPanicked,
}

impl<E: fmt::Display> fmt::Display for Nvfp4ParallelError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig {
                workers,
                chunk_elements,
            } => write!(
                formatter,
                "parallel NVFP4 requires at least one worker and a chunk of at least {BLOCK_SIZE} values, got workers={workers}, chunk_elements={chunk_elements}"
            ),
            Self::Quantization(error) => error.fmt(formatter),
            Self::Callback(error) => error.fmt(formatter),
            Self::WorkerPanicked => formatter.write_str("parallel NVFP4 worker panicked"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for Nvfp4ParallelError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Quantization(error) => Some(error),
            Self::Callback(error) => Some(error),
            Self::InvalidConfig { .. } | Self::WorkerPanicked => None,
        }
    }
}

/// Quantizes a shaped row-major tensor from a replayable source using up to
/// `config.workers` threads (at most [`MAX_WORKERS`]).
///
/// `config.chunk_elements` is the number of values read per chunk, rounded
/// down to whole 16-value blocks.  The result and the emitted bytes are
/// identical to [`modelq_quant::nvfp4::quantize_shaped`].
pub fn quantize_replay_chunks<F, I, C, E>(
    shape: &[usize],
    values: F,
    config: ParallelConfig,
    emit: C,
) -> Result<StreamedQuantization, Nvfp4ParallelError<E>>
where
    F: FnMut() -> I + Send,
    I: IntoIterator<Item = f32>,
    C: FnMut(&[u8]) -> Result<(), E>,
{
    quantize_replay_chunks_with(shape, values, config, ScaleSelection::default(), emit)
}

/// [`quantize_replay_chunks`] with an explicit block-scale selection; the
/// output is bit-identical to [`modelq_quant::nvfp4::quantize_replay_chunks_with`]
/// for any worker count and chunk size.
pub fn quantize_replay_chunks_with<F, I, C, E>(
    shape: &[usize],
    values: F,
    config: ParallelConfig,
    selection: ScaleSelection,
    mut emit: C,
) -> Result<StreamedQuantization, Nvfp4ParallelError<E>>
where
    F: FnMut() -> I + Send,
    I: IntoIterator<Item = f32>,
    C: FnMut(&[u8]) -> Result<(), E>,
{
    let fail = Nvfp4ParallelError::Quantization;
    if config.workers == 0 || config.chunk_elements < BLOCK_SIZE {
        return Err(Nvfp4ParallelError::InvalidConfig {
            workers: config.workers,
            chunk_elements: config.chunk_elements,
        });
    }
    let expected = nvfp4::checked_shape_elements(shape).map_err(fail)?;
    let workers = config.workers.min(MAX_WORKERS);
    let chunk_values = config.chunk_elements / BLOCK_SIZE * BLOCK_SIZE;
    let buffer_capacity = chunk_values.min(expected);
    // Reading the source is serial.  With more than one worker and at least
    // two chunks, a reader thread fills the next chunk while the workers
    // process the current one; smaller jobs read inline and start no thread.
    let overlap = workers > 1 && expected > chunk_values;

    let (global_amax, block_scales) = with_feed(
        values,
        chunk_values,
        buffer_capacity,
        2,
        overlap,
        |feed| -> Result<(f32, Vec<u8>), Nvfp4ParallelError<E>> {
            let mut buffer: Vec<f32> = Vec::with_capacity(buffer_capacity);
            let panicked = |WorkerPanicked| Nvfp4ParallelError::WorkerPanicked;

            // Pass 1: tensor-wide amax, with the first non-finite value
            // reported by its tensor-wide index.
            let mut global_amax = 0.0_f32;
            let mut seen = 0_usize;
            feed.start_pass();
            loop {
                feed.fill(&mut buffer).map_err(panicked)?;
                if buffer.is_empty() {
                    break;
                }
                global_amax = global_amax.max(parallel_amax(&buffer, seen, workers)?);
                seen += buffer.len();
            }
            if seen != expected {
                return Err(fail(Nvfp4Error::ShapeLengthMismatch {
                    expected,
                    actual: seen,
                }));
            }

            // Pass 2: encode each chunk across the workers and emit in order.
            let mut packed: Vec<u8> = Vec::with_capacity(packed_len(buffer_capacity));
            let mut block_scales = vec![0_u8; block_count(expected)];
            let mut start = 0_usize;
            feed.start_pass();
            loop {
                feed.fill(&mut buffer).map_err(panicked)?;
                if buffer.is_empty() {
                    break;
                }
                let first_block = start / BLOCK_SIZE;
                let scales =
                    &mut block_scales[first_block..first_block + block_count(buffer.len())];
                packed.resize(packed_len(buffer.len()), 0);
                parallel_encode(
                    &buffer,
                    start,
                    global_amax,
                    selection,
                    &mut packed,
                    scales,
                    workers,
                )?;
                start += buffer.len();
                emit(&packed).map_err(Nvfp4ParallelError::Callback)?;
            }
            if start != expected {
                return Err(fail(Nvfp4Error::ShapeLengthMismatch {
                    expected,
                    actual: start,
                }));
            }
            Ok((global_amax, block_scales))
        },
    )
    .map_err(|WorkerPanicked| Nvfp4ParallelError::WorkerPanicked)??;

    Ok(StreamedQuantization::new(
        block_scales,
        nvfp4::global_scale_for_amax(global_amax),
        expected,
    ))
}

/// Values per range: a whole number of blocks, so every range after the
/// first starts on a block boundary.  A chunk is cut into several ranges per
/// worker (see [`crate::schedule`]) so faster cores can take more of them; tiny
/// chunks get fewer ranges so no worker is started for almost no work.
fn range_values(blocks: usize, workers: usize) -> usize {
    let ranges = (workers * RANGES_PER_WORKER)
        .min(blocks.div_ceil(MIN_BLOCKS_PER_RANGE))
        .max(1);
    blocks.div_ceil(ranges) * BLOCK_SIZE
}

fn parallel_amax<E>(
    values: &[f32],
    first_index: usize,
    workers: usize,
) -> Result<f32, Nvfp4ParallelError<E>> {
    let per_range = range_values(block_count(values.len()), workers);
    let results = schedule::run_all(
        values.chunks(per_range).collect(),
        workers,
        |index, range| scan_chunk_amax(range, first_index + index * per_range),
    )
    .map_err(|WorkerPanicked| Nvfp4ParallelError::WorkerPanicked)?;

    // Ranges are in index order, so the first error is the lowest index.
    let mut amax = 0.0_f32;
    for result in results {
        amax = amax.max(result.map_err(Nvfp4ParallelError::Quantization)?);
    }
    Ok(amax)
}

fn parallel_encode<E>(
    values: &[f32],
    first_index: usize,
    global_amax: f32,
    selection: ScaleSelection,
    packed: &mut [u8],
    scales: &mut [u8],
    workers: usize,
) -> Result<(), Nvfp4ParallelError<E>> {
    let per_range = range_values(block_count(values.len()), workers);
    let items: Vec<_> = values
        .chunks(per_range)
        .zip(packed.chunks_mut(per_range / 2))
        .zip(scales.chunks_mut(per_range / BLOCK_SIZE))
        .collect();
    let results = schedule::run_all(items, workers, |index, ((range, packed), scales)| {
        encode_chunk_with(
            range,
            first_index + index * per_range,
            global_amax,
            selection,
            packed,
            scales,
        )
    })
    .map_err(|WorkerPanicked| Nvfp4ParallelError::WorkerPanicked)?;

    for result in results {
        result.map_err(Nvfp4ParallelError::Quantization)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use modelq_quant::nvfp4::{self, Nvfp4Error};

    use super::{Nvfp4ParallelError, quantize_replay_chunks};
    use crate::cpu::ParallelConfig;

    fn spread_values(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|index| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0;
                let exponent = ((state >> 8) % 24) as i32 - 12;
                if index % 17 == 0 {
                    0.0
                } else {
                    unit * 2.0_f32.powi(exponent)
                }
            })
            .collect()
    }

    fn run(
        values: &[f32],
        shape: &[usize],
        workers: usize,
        chunk_elements: usize,
    ) -> (Vec<u8>, nvfp4::StreamedQuantization) {
        let mut packed = Vec::new();
        let result = quantize_replay_chunks(
            shape,
            || values.iter().copied(),
            ParallelConfig::new(workers, chunk_elements),
            |chunk| -> Result<(), ()> {
                packed.extend_from_slice(chunk);
                Ok(())
            },
        )
        .expect("parallel quantization succeeds");
        (packed, result)
    }

    fn assert_matches_reference(values: &[f32], shape: &[usize]) {
        let reference = nvfp4::quantize_shaped(values, shape).expect("reference succeeds");
        // Tiny chunks force many chunk boundaries; large worker counts exceed
        // the work available and the hard bound.
        for workers in [1, 2, 3, 7, 18, 1000] {
            for chunk_elements in [16, 48, 4096, 70_000, 1 << 22] {
                let (packed, streamed) = run(values, shape, workers, chunk_elements);
                let label = format!("workers={workers} chunk_elements={chunk_elements}");
                assert_eq!(packed, reference.packed(), "{label}");
                assert_eq!(streamed.block_scales(), reference.block_scales(), "{label}");
                assert_eq!(
                    streamed.global_scale().to_bits(),
                    reference.global_scale().to_bits(),
                    "{label}"
                );
                assert_eq!(streamed.elements(), values.len(), "{label}");
            }
        }
    }

    #[test]
    fn matches_the_scalar_reference_bit_for_bit() {
        // 600 blocks exceeds MIN_BLOCKS_PER_WORKER, so multi-threaded ranges
        // really run; the odd row counts split ranges mid-row.
        for (rows, columns, seed) in [(1, 16, 1), (5, 48, 2), (600, 16, 3), (37, 4096, 4)] {
            let values = spread_values(rows * columns, seed);
            assert_matches_reference(&values, &[rows, columns]);
        }
    }

    #[test]
    fn matches_the_reference_for_edge_inputs() {
        assert_matches_reference(&vec![0.0; 16 * 600], &[600, 16]);

        let mut outlier = vec![0.001_f32; 16 * 600];
        outlier[16 * 599 + 3] = 6.0e6;
        assert_matches_reference(&outlier, &[600, 16]);

        let mut tiny = vec![0.0_f32; 16 * 600];
        tiny[0] = f32::from_bits(1);
        tiny[16 * 300] = 1.0;
        assert_matches_reference(&tiny, &[600, 16]);
    }

    #[test]
    fn reports_the_first_non_finite_index_regardless_of_workers() {
        let mut values = spread_values(16 * 4000, 7);
        values[40_000] = f32::NAN;
        values[50_000] = f32::INFINITY;
        for workers in [1, 4, 18] {
            for chunk_elements in [4096, 1 << 22] {
                let error = quantize_replay_chunks(
                    &[4000, 16],
                    || values.iter().copied(),
                    ParallelConfig::new(workers, chunk_elements),
                    |_| Ok::<(), ()>(()),
                )
                .expect_err("non-finite input is rejected");
                assert!(
                    matches!(
                        error,
                        Nvfp4ParallelError::Quantization(Nvfp4Error::NonFiniteInput {
                            index: 40_000,
                            ..
                        })
                    ),
                    "workers={workers} chunk_elements={chunk_elements}: {error:?}"
                );
            }
        }
    }

    #[test]
    fn validates_configuration_shape_and_length() {
        let values = vec![1.0_f32; 32];
        let call = |shape: &[usize], config: ParallelConfig, source: &[f32]| {
            quantize_replay_chunks(
                shape,
                || source.iter().copied(),
                config,
                |_| Ok::<(), ()>(()),
            )
        };
        assert!(matches!(
            call(&[2, 16], ParallelConfig::new(0, 64), &values),
            Err(Nvfp4ParallelError::InvalidConfig { workers: 0, .. })
        ));
        assert!(matches!(
            call(&[2, 16], ParallelConfig::new(2, 15), &values),
            Err(Nvfp4ParallelError::InvalidConfig {
                chunk_elements: 15,
                ..
            })
        ));
        assert!(matches!(
            call(&[2, 15], ParallelConfig::new(2, 64), &values),
            Err(Nvfp4ParallelError::Quantization(
                Nvfp4Error::InvalidShape { .. }
            ))
        ));
        assert!(matches!(
            call(&[3, 16], ParallelConfig::new(2, 64), &values),
            Err(Nvfp4ParallelError::Quantization(
                Nvfp4Error::ShapeLengthMismatch {
                    expected: 48,
                    actual: 32
                }
            ))
        ));
    }

    #[test]
    fn emits_ordered_bounded_chunks_and_propagates_callback_errors() {
        let values = spread_values(16 * 10, 9);
        let mut sizes = Vec::new();
        quantize_replay_chunks(
            &[10, 16],
            || values.iter().copied(),
            ParallelConfig::new(4, 64),
            |chunk| {
                sizes.push(chunk.len());
                Ok::<(), ()>(())
            },
        )
        .expect("succeeds");
        // 64 values = 4 blocks = 32 packed bytes per chunk: 4, 4, 2 blocks.
        assert_eq!(sizes, [32, 32, 16]);

        let error = quantize_replay_chunks(
            &[10, 16],
            || values.iter().copied(),
            ParallelConfig::new(4, 64),
            |_| Err("sink failed"),
        )
        .expect_err("callback failure is surfaced");
        assert!(matches!(error, Nvfp4ParallelError::Callback("sink failed")));
    }
}
