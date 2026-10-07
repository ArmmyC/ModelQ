# ADR 0025: Minimum-Error Block-Scale Search for NVFP4

- Status: Accepted (Task 44)
- Date: 2026-10-07
- Scope: an opt-in encoder mode, `--scale-search <RADIUS>`, that chooses each block's E4M3 scale by minimum reconstruction error; the default output is unchanged
- Result: **on Qwen2.5-0.5B, radius 6 cuts the WikiText-2 perplexity increase from +9.93% to +8.07% (about a fifth of the loss), at roughly 1.2x the wall-clock time of the default when run in parallel; the container passes the B200 proof 168 of 168.**
- Builds on: [ADR 0010](0010-nvfp4-research-spike.md), [ADR 0021](0021-transformer-engine-multi-matrix-container.md), [ADR 0024](0024-nvfp4-output-quality-qwen2-5-0-5b.md)

## Context

[ADR 0024](0024-nvfp4-output-quality-qwen2-5-0-5b.md) measured what the reference quantizer costs a real model: +9.9% perplexity on Qwen2.5-0.5B. The reference rule picks each 16-value block's scale from the block's largest value alone, rounded to the nearest E4M3 code, so that the largest value lands near the top of the E2M1 range. Rounding that scale to three mantissa bits can clip the block's largest values (when it rounds down) or leave range unused (when it rounds up), and minimizing the *largest* value's error is not the same as minimizing the block's error.

## Decision

Add `ScaleSelection` to the quantizer, with the default `Amax` (the reference rule, unchanged) and an opt-in `MinMse { radius }`:

- Start from the reference scale code. Try every nonzero E4M3 code within `radius` of it. For each candidate, re-encode the block's values with that scale and sum the squared difference to the decoded value (`e2m1 * scale * global_scale`, the decoder's own arithmetic). Keep the candidate with the smallest error.
- Ties are deterministic: the reference scale wins, then the nearer code, then the lower code. `radius = 0` is bit-identical to the default.
- The per-tensor global scale is untouched, and the output is the same representation (E2M1 values, E4M3 block scales, one decode scale), so **no decoder, container schema, or runtime changes**. A runtime sees different scale bytes, nothing else.
- It is threaded through the whole-slice, sequential streaming and parallel streaming quantizers and both writers (`Nvfp4Settings`), and exposed as `modelq quantize --format nvfp4|nvfp4-te --scale-search <RADIUS>` (0 to 32; rejected for INT8). The choice is recorded for provenance only: `modelq.scale_selection` in native metadata, and `encoder.scale_selection` in the Transformer Engine manifest, and only when it is not the default. The default output carries no new key and is byte-identical to before, so the earlier hardware evidence for default output stays valid.

## Verification

- Radius 0 equals the reference rule bit for bit; the search never has a higher error than the reference on any tested tensor and usually has a lower one.
- An independent brute-force oracle in the tests (written with the public checked codecs, outside the production loop, enumerating candidates and applying the tie rule) chooses the same scale as the production search for 160 tensor-and-radius combinations.
- Streaming equals whole-slice, and parallel equals sequential, for the search across chunk sizes and worker counts, at the quantizer, writer and CLI levels. Non-finite input is reported with the same index as before. Searched containers pass both the Rust and the independent Python validators.

## Quality and cost on Qwen2.5-0.5B

Measured exactly as in ADR 0024 (float32 simulated weights, WikiText-2 test, 146 windows of 2048 tokens, one run, L4), with every container compared against the same original-model baseline (perplexity 13.0699). Raw result: [`docs/validation/nvfp4-scale-search-qwen2.5-0.5b-wikitext2.json`](../validation/nvfp4-scale-search-qwen2.5-0.5b-wikitext2.json).

| Block scales | Perplexity | Increase | KL (nats/token) | Top-1 agreement | Mean weight error |
| --- | --- | --- | --- | --- | --- |
| default rule | 14.3683 | +9.93% | 0.1009 | 83.49% | 9.44% |
| search, radius 1 | 14.1836 | +8.52% | 0.0883 | 84.64% | 8.62% |
| search, radius 2 | 14.1920 | +8.59% | 0.0887 | 84.65% | 8.58% |
| search, radius 4 | 14.1642 | +8.37% | 0.0837 | 84.97% | 8.34% |
| search, radius 6 | 14.1251 | +8.07% | 0.0803 | 85.34% | 8.15% |
| search, radius 8 | 14.1254 | +8.08% | 0.0803 | 85.34% | 8.15% |

The gain saturates at radius 6: radius 8 is no better. That matches the arithmetic: six E4M3 codes span roughly a factor of 0.6 to 1.7 around the reference scale, which already includes scaling the block's maximum to 4 instead of 6. Radius 2 is marginally *worse* than radius 1 in perplexity although its weight error is lower, so minimum weight error and minimum perplexity are related but not identical. Radius 12 was not evaluated; the saturation makes it unlikely to matter.

**Cost** (`cargo bench --bench nvfp4_scale_search`, 16.8M values, Modal CPU with 24 logical CPUs; ratios are what matter, the absolute times depend on the machine): one thread takes 1141 ms at radius 6 against 141 ms for the default rule, 8.1x; the parallel path takes 148 ms against 122 ms, 1.22x, because the serial chunk fill still dominates there (radius 8: 10.2x and 1.29x; radius 1: 2.8x and 1.00x). Conversion of the 0.5B model is therefore a matter of seconds in parallel; with `--threads 1` it is noticeably slower.

**Hardware.** The radius-6 container of Qwen2.5-0.5B passed the same B200 proof as ADR 0023 (single file): 168 of 168 matrices loaded into Transformer Engine 2.19.0, dequantized to a reference re-derived with the same search from the source weights (maximum difference 1.2e-7), and passed one TN GEMM each (median error 3.8e-6, maximum 1.6e-5). Raw result: [`docs/validation/te-qwen2.5-0.5b-scale-search-r6-b200.json`](../validation/te-qwen2.5-0.5b-scale-search-r6-b200.json). The runtime does not care how scales were chosen, as expected, but this was run and not assumed.

## What this does not show

- One model, one text, one run each; the improvement could differ on other models, sizes, or tasks, and no downstream task accuracy was measured. The residual loss (+8.1%) is still visible.
- The search minimizes per-block weight error, not output error, so it is a proxy; the non-monotonic radius 1 and 2 results illustrate that.
- It does not cover choosing the global scale (clipping), 2D 16x16 blocks, or calibration with activations, any of which could do better or worse.
- Only the single-file container was proven on the B200 for radius 6 (the sharded layout is independent of scale selection and was validated for the default in ADR 0023).

## Consequences

- Users who accept slower single-threaded conversion can recover about a fifth of the NVFP4 quality loss on this model with `--scale-search 6`, with no change to the runtime path.
- The default stays the reference rule, so existing artifacts and their validation are unaffected. Whether to make radius 6 the default is a separate decision that would change default output bytes and should be made after more models are measured.
- The same harness (`modal_runtime_proof.py::convert_scale_search`, `modal_quality_eval.py --names ...`) can score further encoder ideas against these numbers.
