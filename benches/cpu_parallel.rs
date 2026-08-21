//! Small comparative benchmark for the scalar and parallel CPU paths.
//!
//! Run with `cargo bench --bench cpu_parallel`. The output reports a speedup
//! ratio instead of asserting that parallel execution wins on every machine.

use std::hint::black_box;
use std::time::{Duration, Instant};

use modelq::backend::cpu::{ParallelConfig, quantize_int4, quantize_int8};
use modelq::quant::{int4, int8};

const ELEMENTS: usize = 1 << 20;
const ITERATIONS: usize = 5;

fn main() {
    let values = (0..ELEMENTS)
        .map(|index| ((index % 1021) as f32 - 510.0) / 37.0)
        .collect::<Vec<_>>();
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(8);
    let config = ParallelConfig::new(workers, 16 * 1024);

    let scalar_int8 = measure(|| int8::quantize(black_box(&values)));
    let parallel_int8 = measure(|| quantize_int8(black_box(&values), config));
    let scalar_int4 = measure(|| int4::quantize(black_box(&values), 128));
    let parallel_int4 = measure(|| quantize_int4(black_box(&values), 128, config));

    println!("CPU parallel benchmark ({ELEMENTS} F32 elements, {workers} workers)");
    report("INT8", scalar_int8, parallel_int8);
    report("INT4", scalar_int4, parallel_int4);
}

fn measure<F, R>(mut operation: F) -> Duration
where
    F: FnMut() -> R,
{
    let mut best = Duration::MAX;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        black_box(operation());
        best = best.min(start.elapsed());
    }
    best
}

fn report(name: &str, scalar: Duration, parallel: Duration) {
    let speedup = scalar.as_secs_f64() / parallel.as_secs_f64();
    println!(
        "{name}: scalar={:.3} ms parallel={:.3} ms speedup={speedup:.2}x",
        scalar.as_secs_f64() * 1_000.0,
        parallel.as_secs_f64() * 1_000.0,
    );
}
