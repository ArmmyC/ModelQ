# ADR 0026: Block-Scale Search Across Three Qwen2.5 Models

- Status: Accepted (Task 45)
- Date: 2026-10-08
- Scope: measurement only; no code path changed. The default output and the `--scale-search` behavior are as in ADR 0025
- Result: **radius 6 lowers the WikiText-2 perplexity increase on all three models tested, by about a fifth (19% to 24% of the loss), and radii 8 and 12 add nothing. Radius-6 containers of the two new models pass the B200 proof (168 of 168 and 196 of 196).**
- Builds on: [ADR 0024](0024-nvfp4-output-quality-qwen2-5-0-5b.md), [ADR 0025](0025-nvfp4-block-scale-search.md)

## Context

[ADR 0025](0025-nvfp4-block-scale-search.md) showed a gain from the block-scale search on one model and said that whether radius 6 should become the default needs more models. It also left radius 12 unevaluated and proved only one model's container on the hardware. This task repeats the measurement on two more models.

## Method

The same pipeline as ADR 0024 and 0025, unchanged apart from the model being a parameter: each model was downloaded inside Modal at its pinned revision (ungated, Apache-2.0, SHA-256 verified), converted with `modelq quantize --format nvfp4-te` at radii 0, 1, 4, 6, 8 and 12, and each container was substituted into a float32 copy of the model and scored against the original on WikiText-2 test (146 windows of 2048 tokens) on an L4. Radius 0 is the default rule, bit for bit (ADR 0025). Models:

- Qwen2.5-0.5B (from ADR 0025; only radius 12 is new).
- Qwen2.5-0.5B-Instruct (revision `7ae557604a`), the same architecture after instruction tuning.
- Qwen2.5-1.5B (revision `8faed761d4`), the same family at three times the parameters.

Raw results: [`nvfp4-scale-search-qwen2.5-0.5b-instruct-wikitext2.json`](../validation/nvfp4-scale-search-qwen2.5-0.5b-instruct-wikitext2.json), [`nvfp4-scale-search-qwen2.5-1.5b-wikitext2.json`](../validation/nvfp4-scale-search-qwen2.5-1.5b-wikitext2.json), [`nvfp4-scale-search-qwen2.5-0.5b-r12-wikitext2.json`](../validation/nvfp4-scale-search-qwen2.5-0.5b-r12-wikitext2.json).

## Results

Perplexity increase over the original model (lower is better); in brackets, mean KL in nats per token.

| Block scales | Qwen2.5-0.5B | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B |
| --- | --- | --- | --- |
| original perplexity | 13.0699 | 14.2425 | 9.2648 |
| default rule | +9.93% (0.1009) | +12.66% (0.1181) | +6.71% (0.0758) |
| radius 1 | +8.52% (0.0883) | +11.63% (0.1044) | +5.53% (0.0650) |
| radius 4 | +8.37% (0.0837) | +11.52% (0.1012) | +5.37% (0.0613) |
| radius 6 | +8.07% (0.0803) | +9.62% (0.0932) | +5.31% (0.0603) |
| radius 8 | +8.08% (0.0803) | +9.63% (0.0932) | +5.31% (0.0603) |
| radius 12 | +8.08% (0.0803) | +9.63% (0.0932) | +5.31% (0.0603) |

At radius 6, against the default rule:

| | Perplexity increase | Mean KL | Top-1 agreement | Mean weight error |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 9.93% to 8.07% (19% less) | 0.1009 to 0.0803 (20% less) | 83.49% to 85.34% | 9.44% to 8.15% |
| Qwen2.5-0.5B-Instruct | 12.66% to 9.62% (24% less) | 0.1181 to 0.0932 (21% less) | 82.36% to 84.09% | 9.44% to 8.15% |
| Qwen2.5-1.5B | 6.71% to 5.31% (21% less) | 0.0758 to 0.0603 (20% less) | 86.51% to 87.91% | 9.45% to 8.14% |

What this supports:

- **The gain is consistent.** The KL divergence, which is a smoother measure than perplexity, falls by 20% to 21% on all three models, and top-1 agreement rises by 1.4 to 1.9 points. Mean weight error falls from about 9.4% to about 8.1% regardless of the model, so the weight-level effect is a property of the encoder, not of one checkpoint.
- **Radius 6 is enough.** Radii 8 and 12 give the same results to within noise on every model (the 0.5B-Instruct and 1.5B values at 8 and 12 agree to the printed digits, and radius 12 on 0.5B equals radius 8). The radius-12 question left open in ADR 0025 is closed.
- **A larger model loses less, with or without the search** (+6.7% against +9.9% default, for 1.5B against 0.5B), and the search still removes about a fifth of what is lost.
- **Perplexity is noisier than KL.** On the Instruct model perplexity moves little from radius 1 to 4 (11.63% to 11.52%) and then drops at radius 6 (9.62%), while KL falls steadily (0.1044, 0.1012, 0.0932). The single-run, single-text perplexity should not be read to two digits.

## Hardware

The radius-6 containers of both new models passed the same B200 proof as ADR 0023 and 0025 (Transformer Engine 2.19.0, one TN GEMM per matrix, dequantized to a reference re-derived from the source weights with the same search):

| Model | Matrices passed | Max dequantization difference | GEMM max abs error (median / max) |
| --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 168 of 168 | 1.2e-7 | 3.8e-6 / 2.4e-5 |
| Qwen2.5-1.5B | 196 of 196 | 1.2e-7 | 2.0e-5 / 5.7e-5 |

Raw results: [`te-qwen2.5-0.5b-instruct-scale-search-r6-b200.json`](../validation/te-qwen2.5-0.5b-instruct-scale-search-r6-b200.json), [`te-qwen2.5-1.5b-scale-search-r6-b200.json`](../validation/te-qwen2.5-1.5b-scale-search-r6-b200.json). Together with ADR 0025 that is three models proven on hardware with searched scales. Only the default rule and radius 6 were put through the hardware proof for the new models; the other radii were scored for quality only.

## What this does not show

- Three models from one family (Qwen2.5, 0.5B to 1.5B, one of them a fine-tune of another). No other architecture, no model above 1.5B, and no mixture-of-experts model was measured; the improvement could differ there.
- One text (WikiText-2) and one run per cell. No downstream task accuracy was measured, and WikiText-2 perplexity is a proxy for the quality of generated text.
- Weight-only simulated quantization in float32, as in ADR 0024; activations and the real FP4 GEMM path are not part of the quality numbers (the hardware proof checks GEMM correctness against the dequantized reference, not task quality).
- No comparison with other quantization methods or formats.

## Decision

The default stays the reference rule for now. The evidence is now consistent across three models, and a reviewer could reasonably want radius 6 to be the default, but that changes the bytes of every default conversion and the cost trade (about 1.2x wall time in parallel and about 8x on one thread, ADR 0025) falls on everyone, so it is recorded as a recommendation rather than done silently: **if the default is changed, 6 is the radius to use**, and the change should be its own task that updates the earlier hardware evidence for default output.

## Consequences

- Users who convert Qwen2.5-class models can use `--scale-search 6` with the expectation, from three models, of roughly a fifth less quality loss, and have no reason to use a larger radius.
- The harness takes the model as a parameter (`modal_quality_eval.py --model ... --names ...`; `modal_runtime_proof.py::search_experiment --model ... --base ... --radii ...`), so a further model costs one command per stage.
