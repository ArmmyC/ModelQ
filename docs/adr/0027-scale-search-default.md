# ADR 0027: Radius 6 Is the CLI Default for NVFP4 Block Scales

- Status: Accepted (Task 46)
- Date: 2026-10-08
- Scope: the default of `modelq quantize --format nvfp4|nvfp4-te`; the library API and the INT8 path are unchanged
- Result: **without a flag, the CLI now picks each block's scale by the minimum-error search with radius 6. `--scale-search 0` selects the reference rule and reproduces the previous default output byte for byte. The new default passed the B200 proof both as a single file and as a sharded directory (168 of 168 matrices each).**
- Builds on: [ADR 0025](0025-nvfp4-block-scale-search.md), [ADR 0026](0026-block-scale-search-across-models.md)

## Context

ADR 0025 added the search as an opt-in and kept the reference rule as the default until more models were measured. ADR 0026 measured three Qwen2.5 models (0.5B, 0.5B-Instruct, 1.5B): radius 6 lowered the perplexity increase by 19% to 24% and the KL divergence by 20% to 21% on every one, radius 8 and 12 added nothing, and the radius-6 containers passed the B200 proof. It recommended 6 as the radius to use if the default changed.

## Decision

- The CLI default for `--scale-search` is **6** (`DEFAULT_SCALE_SEARCH_RADIUS` in `src/nvfp4_command.rs`). Omitting the flag is exactly `--scale-search 6`; a test asserts the two outputs are byte-identical for both formats.
- `--scale-search 0` is the reference rule and gives the same bytes as every earlier version of the tool. It is the way to reproduce older artifacts.
- The library keeps the reference rule as its default (`quantize`, `Nvfp4Settings::reference()`, `write_nvfp4_safetensors`, `write_te_nvfp4_safetensors`). Those functions are the reference implementation that the independent oracles and the validators compare against, and library callers see no change. Only the command-line default moved.
- The report always states the rule: `Block scales: min-mse:r6 (minimum-error search)` or `Block scales: amax (reference rule)`. As before, a non-default rule is recorded in the output metadata (`modelq.scale_selection` natively, `encoder.scale_selection` in the Transformer Engine manifest), so every output made with the new default says which rule produced it, and an output made with `--scale-search 0` carries no such key and is byte-identical to the old default.
- INT8 is unaffected; `--scale-search` is still rejected for `--format int8`.

## What changes for users

- **Bytes.** Default conversions of the same checkpoint produce different scale bytes than before. Values, layout, container schema and decoders are unchanged, so existing readers and the Transformer Engine runtime accept the new files; the earlier hardware evidence for default-rule output stays valid for output made with `--scale-search 0`.
- **Quality.** About a fifth less quality loss on the three models measured (ADR 0026).
- **Time.** About 1.2x the wall time of the reference rule with all CPUs and about 8x with `--threads 1` at radius 6 (ADR 0025). A pipeline that converts single-threaded and cares about time should pass `--scale-search 0`.

## Verification

- Linux test suite (Rust 1.85, Modal): 246 passed, 0 failed, 2 ignored. The tests that compare CLI output with the library's reference output now pass `--scale-search 0` explicitly, and the CLI test asserts that the implicit default equals radius 6, that radius 0 reports the reference rule, that the searched output has a lower error than the reference, and that the output does not depend on the thread count.
- End-to-end hardware proof of the default path: Qwen2.5-0.5B (revision `060db6499f`) was converted inside Modal with plain `modelq quantize --format nvfp4-te` (no flag) both as a single file and as a sharded directory, and each passed the same B200 proof as before with Transformer Engine 2.19.0: 168 of 168 matrices loaded, dequantized to a reference re-derived from the source weights with radius 6 (maximum difference 1.2e-7), and passed a TN GEMM (median error 3.8e-6, maximum 1.6e-5). The single-file output is 473,840,528 bytes, the same as the radius-6 container of ADR 0025. Raw result: [`te-qwen2.5-0.5b-default-scale-search-b200.json`](../validation/te-qwen2.5-0.5b-default-scale-search-b200.json). This also covers the sharded layout for searched scales, which ADR 0025 listed as not proven.
- The Modal tools were updated to match: fixture builds pin `--scale-search 0` (their reference uses the reference rule), and model conversions use the CLI default with a radius-6 reference.

## What this does not show

The evidence for the choice is the same as in ADR 0026: three models of one family, one text, weight-only simulated quantization, no downstream task accuracy, no other architecture. A model for which the search is worse than the reference rule has not been seen, but it has not been ruled out; `--scale-search 0` is the escape hatch. The proof of the default path covers one model.

## Consequences

- New conversions are better by default, at a time cost the user can opt out of with one flag.
- Output from before and after this change differs in the scale bytes. The manifest and metadata state the rule for new default output, so the two are distinguishable.
- Any later change to the encoder (clipping, 2D blocks, calibration) should be measured against this default, and added the same way: opt-in first, default after multi-model evidence.
