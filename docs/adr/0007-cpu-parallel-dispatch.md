# ADR 0007: Bounded Parallel CPU Quantization

- Status: Accepted
- Date: 2026-08-22
- Scope: portable parallel execution for the reference INT8 and INT4 paths

## Context

ModelQ's scalar INT8 and group-wise INT4 implementations are the correctness
reference, but a full tensor can contain millions of values. Task 17 needs a
faster CPU path without making a GPU or native SIMD library a requirement, and
without turning a small chunk size into an unbounded thread pool.

The workspace extraction in ADR 0004 deliberately left the backend boundary
empty. A second execution implementation now makes that boundary concrete.

## Decision

Add a `modelq-backend` workspace crate with a portable `cpu` module. It uses
only `std::thread::scope` and the standard library; no rayon, OS-specific
thread API, or native dependency is required.

The public entry points are:

- `cpu::quantize_int8(values, config)`;
- `cpu::quantize_int4(values, group_size, config)`; and
- `cpu::ParallelConfig { workers, chunk_elements }`.

Both paths follow the same sequence:

1. Compute scales with the existing scalar reference functions.
2. Allocate one final output buffer.
3. Divide that buffer into disjoint slices assigned to at most the requested
   worker count, capped at 64 workers and the number of available source
   chunks.
4. Have each worker write directly into its slice while reusing the configured
   chunk size.
5. Construct and validate the public scalar quantized representation.

INT8 workers write quantized bytes directly. INT4 workers pack two values per
byte directly, including the zero high nibble for an odd final element.
Because the scale pass and per-value rounding call the scalar functions, the
parallel results preserve scalar storage order and exact F32 results.

## Bounded-memory rule

The backend does not create one temporary payload allocation per task or one
thread per chunk. Worker count is explicitly bounded, and workers borrow the
source while writing disjoint output slices. The INT8 path has one output
buffer; the INT4 path has one packed output buffer plus its group-scale vector.
The existing replay-chunk API remains available for streaming I/O; Task 17 does
not silently change the writer to require a whole source tensor in memory.

## Verification and benchmark

Unit tests compare parallel INT8 and INT4 output byte-for-byte with the scalar
reference, including odd lengths, group boundaries, empty tensors, invalid
configuration, and a worker-count hard bound. The comparative benchmark is
available with:

```bash
cargo bench --bench cpu_parallel
```

It reports scalar and parallel timings plus a speedup ratio for both formats.
The benchmark is intentionally observational: CPU topology, scheduler, and
input size determine whether parallel execution helps, so a fixed speedup is
not claimed as a correctness condition.

## Consequences

### Benefits

- CPU remains the universal fallback on Linux, Windows, and macOS.
- Scalar reference code remains available as an oracle for future SIMD/GPU
  backends.
- The execution boundary is now a real workspace crate rather than a stub.
- No third-party dependency or platform-specific code is introduced.

### Costs and limitations

- The current API accepts an in-memory F32 slice; it is not yet a parallel
  streaming reader. The writer continues to use bounded scalar replay chunks.
- Parallel work adds thread scheduling overhead and may be slower for small
  tensors or machines with one available core.
- Scale calculation is intentionally serial so the reference result and
  floating-point rounding remain deterministic.

## Alternatives considered

### Add rayon immediately

Rejected for this step. Standard scoped threads provide the small bounded
worker primitive needed here without adding a dependency to every platform.
Rayon can be reconsidered if later scheduling needs justify it.

### Replace the scalar implementation

Rejected. Keeping a scalar oracle is required for differential tests and future
optimized backends.

### Parallelize the writer first

Rejected. The writer's replay API is intentionally bounded and sequential;
parallel quantization should first prove representation equivalence and a
useful speedup before it changes file-I/O orchestration.
