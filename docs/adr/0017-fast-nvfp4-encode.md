# ADR 0017: Faster NVFP4 Encode Kernel

- Status: Accepted and implemented (Task 35)
- Date: 2026-10-06
- Scope: the per-value and per-block encode path of the ModelQ-native NVFP4 quantizer; no change to the format, the CLI, or any produced byte
- Builds on: [ADR 0009](0009-fp4-fp8-codecs.md), [ADR 0010](0010-nvfp4-research-spike.md), [ADR 0016](0016-parallel-nvfp4.md)

## Context

After Task 34 the NVFP4 encode step dominated and cost about 58 ns per value on one thread. Profiling showed why:

- `fp8_e4m3::encode` ran once per block and searched all 127 finite codes, calling `powi` to decode each candidate.
- `fp4_e2m1::encode` ran once per value and searched 8 candidates, decoding each.
- `fp8_e4m3::decode` ran per block (and per value when validating output) and used `powi`.
- Each block allocated scratch vectors that were then copied into the output slices.

## Decision

Make the path faster without changing a single output bit.

- **FP4 E2M1 encode** is seven comparisons against the midpoints between representable magnitudes (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), summed into the code. Strict or non-strict comparison at each midpoint implements round-to-nearest-even (the tie goes to the neighbor with even bits). The loop has no branches or table lookups, so the compiler can vectorize it.
- **FP8 E4M3 encode** saturates at 448 and otherwise scales the magnitude by the exact power-of-two grid spacing of its binade, rounds with `round_ties_even`, and computes the code as `(exponent + 6) * 8 + steps`. The parity of the integer equals the code's low bit, so ties break correctly across binade boundaries, and a carry lands on the next binade's first code.
- **Decoders** are `const` tables built from exact F32 bit patterns (no `powi`).
- **One block kernel.** `encode_block_into` writes codes to a stack array and `encode_chunk` packs them straight into the caller's slices. The reference `quantize`, the sequential streaming path, and the parallel path all call `encode_chunk`, so they still cannot diverge. The two per-value divisions stay divisions, since a reciprocal multiply would change rounding.
- **Error behavior is preserved.** A NaN produced while scaling is detected once per block, and the first offending value is reported with its tensor-wide index, as before.

## Verification

The original search-based codecs and the original whole-tensor algorithm are kept as test-only oracles (`float::reference`, `reference_quantize`).

- Decode tables equal the original decoders bit for bit for all 16 and 256 patterns.
- The encoders equal the originals at every grid point, every midpoint, and four ulps either side of each, for both signs, plus special values; on 150,000 pseudorandom values spanning 2^-16 to 2^10; and, in an ignored test, on all 2^32 F32 bit patterns (run on Linux in release mode; see the result below).
- The production kernel equals the original algorithm on many random tensors, partial final blocks, odd lengths, zero blocks, outliers, subnormal scales, negative zero, and exact ties. The streaming, parallel, writer, and CLI byte-identity tests from Tasks 31 to 34 are unchanged and pass.

### A defect in the original codecs

The exhaustive comparison found that the original search is wrong for huge magnitudes: from about 2^26, `magnitude - candidate` rounds to the same F32 for every candidate, every candidate ties, and the first candidate (zero) wins, so `encode(3.4e38)` returned 0 instead of saturating as ADR 0009 documents. The new encoders saturate, and E5M2, which had the same defect, was fixed identically. The two implementations agree exactly for magnitudes below 2^20, which contains every value NVFP4 can produce (scaled elements within about 6, block scales within 448), so no existing output changes; above 2^20 the tests require documented saturation instead.

## Measured effect

`cargo bench --bench nvfp4_parallel` (16,777,216 F32 values; Docker on a 14-core, 18-thread hybrid CPU; best of three; parallel output compared with sequential before timing). "Before" is the Task 34 result in [ADR 0016](0016-parallel-nvfp4.md).

| Mode | Before | After | Speedup vs. before sequential |
| --- | --- | --- | --- |
| sequential | 1117 ms (15 Mval/s) | 158 ms (106 Mval/s) | 7.1x |
| 2 workers | 608 ms | 101 ms | 11.0x |
| 4 workers | 432 ms | 64 ms | 17.5x |
| 8 workers | 321 ms | 47 ms (358 Mval/s) | 23.8x |
| 18 workers | 245 ms | 51 ms | 21.8x |

The single-thread kernel is about 7x faster, so a 7B-parameter model that took roughly 7.7 minutes of NVFP4 encode on one thread now takes about 1.1 minutes, or well under a minute with parallel workers. Parallel scaling now flattens past 8 workers: with the encode this cheap, the serial chunk fill and amax pass are a larger share, and the 18 logical CPUs include efficiency cores. These numbers are observational and vary with CPU topology and input; no speedup is a correctness condition.

The exhaustive check of all 2^32 F32 bit patterns against the original encoders passed on Linux in release mode (about 79 minutes).

## Consequences

- Faster conversion for every NVFP4 caller, sequential or parallel, and faster output validation, with identical files.
- The oracle codecs stay in the test build, so a future SIMD or hardware-specific kernel has an independent reference to compare against.
- No SIMD intrinsics or platform-specific code were added. The element encoders are branch-free so a future explicit SIMD kernel can be checked against the same oracles; whether the compiler vectorizes the current loop was not inspected.
