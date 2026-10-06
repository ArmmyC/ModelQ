# ADR 0020: Overlapping the Chunk Fill with Parallel Work

- Status: Accepted and implemented (Task 38)
- Date: 2026-10-07
- Scope: `modelq-backend` internals for the parallel INT8 and NVFP4 paths; no change to any produced byte or CLI option
- Builds on: [ADR 0016](0016-parallel-nvfp4.md), [ADR 0018](0018-parallel-int8-cli.md), [ADR 0019](0019-shared-parallel-scheduler.md)

## Context

[ADR 0019](0019-shared-parallel-scheduler.md) measured that at 8 or more workers about half of the parallel run time is serial: the calling thread reads each chunk from the source iterator (a memory-mapped tensor, converting `F16`/`BF16` to `f32`) before the workers can process it. Load balancing cannot reduce that; hiding it behind the parallel work can.

## Decision

Add a crate-private `prefetch` module. A `Feed` delivers a replayable source as passes of chunks:

- an **inline** feed fills each chunk on the calling thread, exactly as before; and
- a **prefetching** feed runs the source on one reader thread that fills the next chunk while the workers process the current one. A fixed pool of two recycled chunk buffers circulates between the reader and the consumer (so three chunk buffers exist in total, including the one being processed), keeping memory bounded. The reader runs straight through every pass, so the second pass starts filling while the first finishes.

Both feeds deliver the same chunks in the same order, so results are independent of which one ran. A unit test compares them for many lengths and chunk sizes, an early-stopping consumer is shown not to hang, and a panicking source is reported as an error instead of propagating.

**When it is used.** With more than one worker and a source known to span at least two chunks. NVFP4 knows the length from the shape; INT8 uses the iterator's `size_hint`, and an unknown length (such as a filtered iterator) reads inline. Small tensors therefore start no extra thread.

**API.** The value factory `F` is now `FnMut() -> I + Send` in `modelq_backend::{int8, nvfp4}`, because it runs on the reader thread. The iterator it returns is created and consumed there and need not be `Send`. Every caller in the workspace (the writers, CLI, tests, benchmarks) already passes closures over shared references, which are `Send`.

**Chunk size.** The reader changes the best chunk size, so the default drops from 4,194,304 to 2,097,152 values (8 MiB) for both backends: a smaller chunk gives a finer pipeline and smaller buffers. Sweeping 0.25M to 4M values showed the old inline path best at 4M and the prefetching path best at 2M; tiny chunks lose to per-chunk thread start-up in both.

## Measured effect

Old (`main`, inline fill, 4M chunks) and new builds alternated three times, best of nine per measurement, 16,777,216 F32 values, Docker on a 14-core/18-thread hybrid CPU, at the shipped defaults:

| Path | Workers | Old | New | Change |
| --- | --- | --- | --- | --- |
| NVFP4 | 8 | 48.4 / 48.9 / 47.6 ms | 38.3 / 36.7 / 36.4 ms | about -23% |
| NVFP4 | 18 | 43.2 / 45.6 / 43.9 ms | 40.8 / 40.3 / 41.4 ms | about -7% |
| INT8 (diagnostics + writer + validation) | 8 | 134 / 142 / 139 ms | 110 / 127 / 109 ms | about -19% |
| INT8 (same) | 18 | 127 / 131 / 132 ms | 117 / 140 / 119 ms | about -5%, noisy |

At the same chunk size of 2M values, the new build was about 25% faster than the old one for NVFP4 (40 against 52 ms at 8 workers) and about 30% faster for INT8 (105 against 149 ms), and the old build was best at 4M, so the table above is the honest end-to-end comparison.

Gains are modest and shrink at high worker counts: with 18 workers plus the reader the CPU is oversubscribed and the serial remainder (amax reduction, output copy, per-chunk thread start-up) dominates. A sweep of 4 to 18 workers showed no clear benefit from leaving a CPU free for the reader (NVFP4 about 36 to 42 ms and INT8 about 108 to 126 ms for 8 to 18 workers), so the default worker count is unchanged. Numbers are observational and vary with CPU topology and load, and the machine drifted by about 10% over time, which is why old and new were always compared back to back.

## Consequences

- Large tensors convert about 20% faster at moderate worker counts, byte for byte the same files.
- The next serial costs are the per-chunk thread start-up (a persistent worker pool would remove it) and the output copy.
- Not done: spawning no thread for the reader when the source is already a plain slice (a zero-copy path would skip the fill entirely).
