//! Comparative benchmark for the scalar and parallel INT8 streaming phases
//! used by `modelq quantize --format int8`.
//!
//! Run with `cargo bench --bench int8_streaming`. The CLI makes three passes
//! per quantized tensor: progress diagnostics, the writer, and post-write
//! validation. Timings are observational; the parallel quantized bytes are
//! checked against the scalar bytes before timing.

use std::hint::black_box;
use std::time::{Duration, Instant};

use modelq::backend::{cpu::ParallelConfig, int8 as parallel};
use modelq::diagnostics::{
    int8_tensor_diagnostics_replay, reconstruction_metrics_streaming, saturation_count_iter,
};
use modelq::quant::int8::{self, DEFAULT_CHUNK_ELEMENTS};

const ELEMENTS: usize = 1 << 24;
const ITERATIONS: usize = 9;

fn main() {
    let values: Vec<f32> = (0..ELEMENTS)
        .map(|index| (((index * 2_654_435_761) % 100_003) as f32 - 50_000.0) / 977.0)
        .collect();
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);

    let scalar_write = || {
        let mut bytes: Vec<u8> = Vec::with_capacity(ELEMENTS);
        let scale = int8::quantize_replay_chunks(
            || values.iter().copied(),
            DEFAULT_CHUNK_ELEMENTS,
            |chunk| -> Result<(), ()> {
                bytes.extend(chunk.iter().map(|&value| value as u8));
                Ok(())
            },
        )
        .expect("scalar quantization succeeds");
        (bytes, scale)
    };
    let parallel_write = |workers: usize| {
        let mut bytes: Vec<u8> = Vec::with_capacity(ELEMENTS);
        let scale = parallel::quantize_replay_chunks(
            || values.iter().copied(),
            ParallelConfig::new(workers, parallel::DEFAULT_CHUNK_ELEMENTS),
            |chunk| -> Result<(), ()> {
                bytes.extend(chunk.iter().map(|&value| value as u8));
                Ok(())
            },
        )
        .expect("parallel quantization succeeds");
        (bytes, scale)
    };
    let (qdata, scale) = scalar_write();
    let source_bytes = (ELEMENTS * 4) as u64;

    let scalar_diagnostics = measure(|| {
        int8_tensor_diagnostics_replay(
            || values.iter().copied(),
            DEFAULT_CHUNK_ELEMENTS,
            source_bytes,
            4,
        )
    });
    let scalar_validate = measure(|| {
        int8::validate_dequantization(qdata.iter().map(|&value| value as i8), scale)
            .expect("valid");
        let metrics = reconstruction_metrics_streaming(
            values.iter().copied(),
            qdata.iter().map(|&value| f32::from(value as i8) * scale),
        );
        (
            metrics,
            saturation_count_iter(qdata.iter().map(|&value| value as i8)),
        )
    });
    let scalar_writer = measure(scalar_write);

    println!("INT8 streaming benchmark ({ELEMENTS} F32 elements, {available} logical CPUs)");
    println!(
        "scalar: diagnostics={:.1} ms writer={:.1} ms validate={:.1} ms total={:.1} ms",
        millis(scalar_diagnostics),
        millis(scalar_writer),
        millis(scalar_validate),
        millis(scalar_diagnostics + scalar_writer + scalar_validate)
    );
    let scalar_total = scalar_diagnostics + scalar_writer + scalar_validate;

    let mut counts = vec![1, 2, 4, 8, available];
    counts.retain(|&workers| workers <= available);
    counts.sort_unstable();
    counts.dedup();
    for workers in counts {
        let config = ParallelConfig::new(workers, parallel::DEFAULT_CHUNK_ELEMENTS);
        assert!(
            parallel_write(workers) == (qdata.clone(), scale),
            "parallel bytes with {workers} workers differ from scalar"
        );
        let diagnostics = measure(|| {
            parallel::tensor_diagnostics_replay(|| values.iter().copied(), config, source_bytes, 4)
        });
        let validate = measure(|| {
            parallel::validate_and_measure(|| values.iter().copied(), &qdata, scale, config)
        });
        let writer = measure(|| parallel_write(workers));
        let total = diagnostics + writer + validate;
        println!(
            "parallel x{workers:<2}: diagnostics={:.1} ms writer={:.1} ms validate={:.1} ms total={:.1} ms speedup={:.2}x",
            millis(diagnostics),
            millis(writer),
            millis(validate),
            millis(total),
            scalar_total.as_secs_f64() / total.as_secs_f64()
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
