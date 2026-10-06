# ADR 0019: Shared Dynamic Scheduler for the Parallel Backends

- Status: Accepted and implemented (Task 37)
- Date: 2026-10-06
- Scope: `modelq-backend` internals; no change to any produced byte, CLI option, or public API
- Builds on: [ADR 0016](0016-parallel-nvfp4.md), [ADR 0018](0018-parallel-int8-cli.md)

## Context

[ADR 0018](0018-parallel-int8-cli.md) introduced dynamic scheduling for the INT8 CLI path: each chunk is cut into several small ranges per worker, workers take the next range from a shared queue, and results are returned in range order. The NVFP4 backend still split each chunk into one equal range per worker and carried its own copy of the thread-spawning code. The question for this task was whether porting the dynamic queue to NVFP4 would help it on a hybrid performance/efficiency CPU, as it did for INT8.

## Decision

Move the scheduler into a crate-private `schedule` module (`run_all`, `RANGES_PER_WORKER`) used by both `modelq_backend::int8` and `modelq_backend::nvfp4`, and delete the duplicated spawn logic. NVFP4 now cuts each chunk into up to four ranges per worker, with at least 64 blocks per range so tiny chunks do not spawn work for nothing. Results are returned in range order, so output and the lowest-index error reporting are unchanged, and the existing byte-identity tests across worker counts and chunk sizes pass unmodified. The scheduler has its own unit tests: result order for any thread count (including zero), each item runs exactly once, no thread is spawned for zero or one item, and a panicking worker is reported.

## Measured effect: neutral

- **End to end** (`cargo bench --bench nvfp4_parallel`, 16,777,216 values, Docker on a 14-core/18-thread hybrid CPU, best of nine, old and new builds alternated three times to cancel the machine's roughly 10% drift): at 8 workers the old and new builds took 60.5/62.2/72.3 ms and 54.8/64.1/58.4 ms; at 18 workers 59.6/62.2/60.7 ms and 61.6/59.2/58.2 ms; at 2 and 4 workers they were indistinguishable. This is within noise.
- **Encode step alone** (16M values, static equal ranges versus a four-ranges-per-worker queue, two runs): at 18 workers about 17.4 and 17.3 ms static against 15.3 and 16.4 ms dynamic; at 8 workers 20.2 and 18.8 ms against 17.5 and 19.0 ms. Dynamic is equal to slightly better, up to about 10%, which is the expected direction but small.

The first comparison I made, between measurements taken minutes apart, appeared to show a regression at 2 to 4 workers. It was machine drift, not the scheduler: even a one-range-per-worker queue, equivalent to the old static split, measured slower than the earlier baseline. Only back-to-back alternation of the two builds is trustworthy on this machine.

## Why it is neutral, and what that implies

At 18 workers the NVFP4 encode of 16M values takes about 16 ms of a roughly 58 ms end-to-end run. Most of the rest is serial work on the calling thread: the chunk fill from the source iterator (about 14 ms per pass over 16M values, two passes) plus the amax reduction and output copy. Load balancing the parallel part cannot move that. INT8 benefited more in [ADR 0018](0018-parallel-int8-cli.md) because its diagnostics and validation phases do heavier per-element work in the parallel section.

The next lever for both formats is therefore overlapping the serial chunk fill with the parallel work, for example with a reader thread that fills the next chunk while workers process the current one. That needs the value source to be `Send` and is not done here.

## Consequences

- One scheduler to maintain and test instead of two, with unchanged output and no measurable end-to-end change.
- NVFP4 gains the same robustness to uneven core speeds as INT8, which shows up only when the parallel section dominates.
- The conclusion that fill overlap is the larger remaining gain is recorded with measurements, so the next task starts from data.
