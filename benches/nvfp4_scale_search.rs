//! Cost of the minimum-error block-scale search against the default rule.
//!
//! Run with `cargo bench --bench nvfp4_scale_search`. For each search radius it
//! times the sequential and the parallel streaming quantizer over the same
//! data as the default rule, in one process so the ratios are comparable, and
//! first checks that parallel output equals sequential output for that radius.
//! The numbers are observational and vary with the machine.

use std::hint::black_box;
use std::time::{Duration, Instant};

use modelq::backend::{
    cpu::ParallelConfig,
    nvfp4::{DEFAULT_CHUNK_ELEMENTS, quantize_replay_chunks_with as parallel},
};
use modelq::quant::nvfp4::{
    DEFAULT_CHUNK_BLOCKS, ScaleSelection, quantize_replay_chunks_with as sequential,
};

const ELEMENTS: usize = 1 << 24;
const COLUMNS: usize = 4096;
const ITERATIONS: usize = 5;

fn main() {
    // Weight-like data: small values with an occasional large one.
    let values: Vec<f32> = (0..ELEMENTS)
        .map(|index| {
            let base = (((index * 2_654_435_761) % 100_003) as f32 - 50_000.0) / 977_000.0;
            if index % 331 == 0 { base * 8.0 } else { base }
        })
        .collect();
    let shape = [ELEMENTS / COLUMNS, COLUMNS];
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);

    let run_sequential = |selection: ScaleSelection| {
        let mut bytes = Vec::with_capacity(ELEMENTS / 2);
        let result = sequential(
            &shape,
            || values.iter().copied(),
            DEFAULT_CHUNK_BLOCKS,
            selection,
            |chunk| -> Result<(), ()> {
                bytes.extend_from_slice(chunk);
                Ok(())
            },
        )
        .expect("sequential quantization succeeds");
        (bytes, result)
    };
    let run_parallel = |selection: ScaleSelection, workers: usize| {
        let mut bytes = Vec::with_capacity(ELEMENTS / 2);
        let result = parallel(
            &shape,
            || values.iter().copied(),
            ParallelConfig::new(workers, DEFAULT_CHUNK_ELEMENTS),
            selection,
            |chunk| -> Result<(), ()> {
                bytes.extend_from_slice(chunk);
                Ok(())
            },
        )
        .expect("parallel quantization succeeds");
        (bytes, result)
    };

    println!("NVFP4 block-scale search cost ({ELEMENTS} F32 elements, {available} logical CPUs)");
    println!(
        "{:<14}{:>14}{:>10}{:>14}{:>10}",
        "selection", "sequential", "vs default", "parallel", "vs default"
    );
    let mut default_times = None;
    for selection in [
        ScaleSelection::Amax,
        ScaleSelection::MinMse { radius: 1 },
        ScaleSelection::MinMse { radius: 2 },
        ScaleSelection::MinMse { radius: 4 },
        ScaleSelection::MinMse { radius: 6 },
        ScaleSelection::MinMse { radius: 8 },
    ] {
        let reference = run_sequential(selection);
        assert!(
            run_parallel(selection, available) == reference,
            "parallel output differs from sequential for {}",
            selection.label()
        );
        let sequential_time = measure(|| run_sequential(selection));
        let parallel_time = measure(|| run_parallel(selection, available));
        let (default_sequential, default_parallel) =
            *default_times.get_or_insert((sequential_time, parallel_time));
        println!(
            "{:<14}{:>11.1} ms{:>9.2}x{:>11.1} ms{:>9.2}x",
            selection.label(),
            millis(sequential_time),
            sequential_time.as_secs_f64() / default_sequential.as_secs_f64(),
            millis(parallel_time),
            parallel_time.as_secs_f64() / default_parallel.as_secs_f64(),
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
