//! Comparative benchmark for the sequential and parallel NVFP4 streaming paths.
//!
//! Run with `cargo bench --bench nvfp4_parallel`. Like `cpu_parallel`, it
//! reports timings and speedups without asserting that parallel execution wins
//! on every machine. The parallel output is checked against the sequential
//! output before timing so a fast wrong answer cannot be reported.

use std::hint::black_box;
use std::time::{Duration, Instant};

use modelq::backend::{
    cpu::ParallelConfig,
    nvfp4::{DEFAULT_CHUNK_ELEMENTS, quantize_replay_chunks as parallel},
};
use modelq::quant::nvfp4::{DEFAULT_CHUNK_BLOCKS, quantize_replay_chunks as sequential};

const ELEMENTS: usize = 1 << 24;
const COLUMNS: usize = 4096;
const ITERATIONS: usize = 3;

fn main() {
    let values: Vec<f32> = (0..ELEMENTS)
        .map(|index| (((index * 2_654_435_761) % 100_003) as f32 - 50_000.0) / 977.0)
        .collect();
    let shape = [ELEMENTS / COLUMNS, COLUMNS];
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);

    let run_sequential = || {
        let mut bytes = Vec::with_capacity(ELEMENTS / 2);
        let result = sequential(
            &shape,
            || values.iter().copied(),
            DEFAULT_CHUNK_BLOCKS,
            |chunk| -> Result<(), ()> {
                bytes.extend_from_slice(chunk);
                Ok(())
            },
        )
        .expect("sequential quantization succeeds");
        (bytes, result)
    };
    let run_parallel = |workers: usize| {
        let mut bytes = Vec::with_capacity(ELEMENTS / 2);
        let result = parallel(
            &shape,
            || values.iter().copied(),
            ParallelConfig::new(workers, DEFAULT_CHUNK_ELEMENTS),
            |chunk| -> Result<(), ()> {
                bytes.extend_from_slice(chunk);
                Ok(())
            },
        )
        .expect("parallel quantization succeeds");
        (bytes, result)
    };

    let reference = run_sequential();
    let scalar = measure(run_sequential);

    println!("NVFP4 streaming benchmark ({ELEMENTS} F32 elements, {available} logical CPUs)");
    println!(
        "sequential: {:.1} ms ({:.1} Mval/s)",
        millis(scalar),
        rate(scalar)
    );
    let mut counts = vec![2, 4, 8, available];
    counts.retain(|&workers| workers >= 2 && workers <= available);
    counts.sort_unstable();
    counts.dedup();
    for workers in counts {
        assert!(
            run_parallel(workers) == reference,
            "parallel output with {workers} workers differs from sequential"
        );
        let time = measure(|| run_parallel(workers));
        println!(
            "parallel x{workers:<2}: {:.1} ms ({:.1} Mval/s) speedup={:.2}x",
            millis(time),
            rate(time),
            scalar.as_secs_f64() / time.as_secs_f64()
        );
    }
}

fn measure<R>(mut operation: impl FnMut() -> R) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        black_box(operation());
        best = best.min(start.elapsed());
    }
    best
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn rate(duration: Duration) -> f64 {
    ELEMENTS as f64 / duration.as_secs_f64() / 1e6
}
