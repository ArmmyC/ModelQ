# ADR 0018: Parallel INT8 in the CLI

- Status: Accepted and implemented (Task 36)
- Date: 2026-10-06
- Scope: `modelq quantize --format int8` uses bounded parallel CPU execution for its writer, progress diagnostics, and post-write validation; `--threads N` now applies to both formats
- Builds on: [ADR 0007](0007-cpu-parallel-dispatch.md), [ADR 0016](0016-parallel-nvfp4.md)

## Context

[ADR 0007](0007-cpu-parallel-dispatch.md) added parallel INT8 as a library path over a whole in-memory slice. The CLI never used it: it streams each tensor from a memory-mapped checkpoint in bounded memory, so it ran the scalar replay functions. Profiling 16M values showed the INT8 CLI makes three passes per quantized tensor:

| Phase | Scalar time |
| --- | --- |
| progress diagnostics (scale pass, quantize, error metrics) | about 250 ms |
| writer (scale pass, quantize) | about 170 ms |
| post-write validation (range check, error metrics, saturation count) | about 115 ms |

## Decision

Add `modelq_backend::int8` with three functions that mirror the scalar streaming functions and share their bounded-memory structure from [ADR 0016](0016-parallel-nvfp4.md): read a chunk of the replayable source on the calling thread, then split it among scoped workers writing disjoint slices.

- `quantize_replay_chunks` (writer path), `tensor_diagnostics_replay` (progress), and `validate_and_measure` (validation).
- `modelq-quant` exposes the shared pieces: `scale_from_max_abs`, `scan_chunk_max_abs`, `quantize_chunk_into`, and a public `MetricsAccumulator` with `push_at` and `merge`. Quantization still goes through the scalar `quantize_value` per element, and the tensor maximum is order-independent, so quantized bytes and scales are **bit-identical** to the scalar path for any worker count and chunk size.
- **Dynamic scheduling.** Each chunk is cut into about four ranges per worker, handed out from a shared queue, and results are re-sorted by range index. A slow core takes fewer ranges instead of stalling the chunk, which mattered on a hybrid performance/efficiency CPU, and the output never depends on which thread ran which range.
- **Deterministic metrics.** Error sums are the only order-sensitive quantity in floating point. They are accumulated over fixed blocks of 4096 values and the block totals are merged in index order, so reported MSE, MAE, and SQNR depend only on that constant, never on worker count or chunk size. For tensors of at most 4096 values they equal the scalar left-to-right sums exactly; for larger tensors they can differ from the previous scalar CLI output in the last few bits of the `f64`, which the three-significant-digit report does not show. Maximum error and saturation counts are exact.
- **CLI.** The default is every CPU. `--threads N` sets the worker count and `--threads 1` runs the scalar writer. Progress diagnostics and validation always use the blocked metrics (with one worker when `--threads 1`) so that reported numbers are the same for every `--threads` value. The command prints which mode ran. `--threads 0` is rejected before anything is written. `--exclude` and `--no-default-excludes` remain NVFP4-only.
- **Writer API.** `write_safetensors_with(.., Int8Execution)` selects `Sequential` or `Parallel(ParallelConfig)`; `write_safetensors` stays sequential.
- Non-finite inputs report the lowest index regardless of worker count, and validation reports the lowest out-of-range or overflowing quantized byte.

## Measured effect

`cargo bench --bench int8_streaming` (16,777,216 F32 values; Docker on a 14-core, 18-thread hybrid CPU; best of nine; the parallel quantized bytes are compared with the scalar bytes before timing; two runs agreed to about 5%):

| Mode | Diagnostics | Writer | Validation | Total | Speedup |
| --- | --- | --- | --- | --- | --- |
| scalar | 252 ms | 172 ms | 120 ms | 543 ms | 1.00x |
| 2 workers | 133 ms | 89 ms | 63 ms | 285 ms | 1.91x |
| 4 workers | 90 ms | 72 ms | 49 ms | 210 ms | 2.58x |
| 8 workers | 77 ms | 55 ms | 33 ms | 165 ms | 3.30x |
| 18 workers | 76 ms | 61 ms | 34 ms | 171 ms | 3.18x |

Scaling stops around 8 workers: the per-element work is cheap, so the serial chunk fill on the calling thread (about 14 ms per pass over 16M values, five passes in total) and memory bandwidth dominate. An earlier version with equal static ranges reached about 2.9x at 18 workers on the same machine, before dynamic scheduling. Numbers are observational and vary with CPU topology and load.

## Consequences

- Large INT8 conversions run about 3x faster end to end with identical files.
- Reported metrics are identical for every thread count, tested at the CLI level.
- Overlapping the serial fill with compute (a reader thread) would raise the ceiling but needs `Send` bounds on the value source; it is left for later. The NVFP4 backend still uses static ranges and could adopt the same dynamic queue.
- Peak memory stays bounded: one chunk of `f32` (16 MiB by default), its `i8` output, and per-block partial sums.
