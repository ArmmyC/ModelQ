# ADR 0016: Bounded Parallel NVFP4 Quantization

- Status: Accepted and implemented (Task 34)
- Date: 2026-10-06
- Scope: CPU-parallel execution of the streaming NVFP4 quantizer, selected by `modelq quantize --format nvfp4 --threads N`
- Builds on: [ADR 0007](0007-cpu-parallel-dispatch.md), [ADR 0014](0014-nvfp4-cli-export.md)

## Context

The streaming NVFP4 quantizer from Task 31 is bounded in memory but single-threaded. Profiling 16M values on the development machine (Linux container, 18 logical CPUs) showed the encode pass costs about 1000 ms while the amax pass costs about 44 ms, so encoding dominates and its blocks are independent once the tensor-wide amax is known.

## Decision

- **One shared kernel.** `modelq_quant::nvfp4::encode_chunk` encodes a block-aligned run into caller-provided packed and block-scale slices; `scan_chunk_amax` finds a chunk's amax and the first non-finite index. The reference `quantize`, the sequential streaming path, and the parallel path all use the same block encoder, so they cannot diverge. `global_scale_for_amax` and `checked_shape_elements` are public for the same reason.
- **Parallel streaming.** `modelq_backend::nvfp4::quantize_replay_chunks` has the same two-pass contract as the sequential function. Each chunk is read on the calling thread (so the source iterator need not be `Send`), then its whole blocks are split into contiguous ranges that scoped threads encode into disjoint slices of the chunk's packed bytes and block scales. Pass 1 reduces per-range amaxes the same way. Workers are capped by `ParallelConfig::workers`, the 64-worker hard bound, and one worker per 256 blocks, so small tensors do not spawn threads.
- **Bit-identical by construction.** `max` is order-independent and every range runs the reference kernel on a block boundary, so output is identical for any worker count and chunk size. When several non-finite values exist, ranges are scanned in index order, so the lowest index is reported as in the sequential path.
- **Bounded memory.** One chunk of `f32` values (default 4,194,304 values, 16 MiB) plus its packed bytes, and the block-scale vector (1/16 of the element count). The larger default chunk keeps per-chunk thread start-up small next to the encode work.
- **Writer and CLI.** `write_nvfp4_safetensors_with(source, plan, destination, Nvfp4Execution)` selects `Sequential` or `Parallel(ParallelConfig)`; the existing `write_nvfp4_safetensors` stays sequential. `modelq quantize --format nvfp4` defaults to parallel on all CPUs; `--threads N` sets the worker count and `--threads 1` selects the sequential reference path. `--threads` is rejected with `--format int8`, which is unchanged, and `--threads 0` is rejected before anything is written. The command prints which mode ran.

## Measured effect

`cargo bench --bench nvfp4_parallel` (16,777,216 F32 values; Docker on a 14-core, 18-thread hybrid CPU; best of three; parallel output compared with sequential before timing):

| Workers | Time | Speedup |
| --- | --- | --- |
| sequential | 1117 ms (15 Mval/s) | 1.00x |
| 2 | 608 ms | 1.84x |
| 4 | 432 ms | 2.58x |
| 8 | 321 ms | 3.48x |
| 18 | 245 ms (68 Mval/s) | 4.56x |

The encode step alone scales 6.5x on 18 threads for a 4M-value chunk (243 ms to 37 ms), so most of the remaining wall time is the encode kernel's own scaling on a hybrid core mix, plus the serial chunk fill (about 14 ms per pass over 16M values). Numbers are observational and vary with CPU topology, scheduler, and input size; no speedup is a correctness condition.

## Consequences

- Real checkpoints convert several times faster with unchanged bytes; tests assert byte-identical files for many worker counts and chunk sizes, at the library and CLI levels.
- The single-thread encode kernel is slow (about 58 ns per value). Making it faster (bit-identically) is the larger remaining lever and is separate work, as is SIMD.
- Overlapping the serial chunk fill with encoding (a reader thread) was not added; it would need `Send` bounds on the value source and was not justified by the measured fill cost.
- INT8 CLI behavior is unchanged.
