# ADR 0024: Output Quality of the NVFP4 Export on Qwen2.5-0.5B

- Status: Accepted (Task 43)
- Date: 2026-10-07
- Scope: how far ModelQ's NVFP4 weights move a real language model's outputs, measured on WikiText-2
- Result: **perplexity rose from 13.07 to 14.37 (+9.9%); the mean KL divergence from the original is 0.101 nats per token; the two models agree on the top-1 token at 83.5% of positions.**
- Builds on: [ADR 0023](0023-real-model-validation-qwen2-5-0-5b.md), which established that the export loads and multiplies correctly in Transformer Engine but did not measure model quality

## Question

ADR 0023 left one thing open: the hardware proof compares Transformer Engine with the container's own interpretation of the quantized weights, so a large quantization error would not have failed it. This task measures the error that matters to a user: how different the model's outputs are after the weights are replaced by their NVFP4 values.

## Method

`tools/quality_eval/` (`quality_eval.py`, run on Modal by `modal_quality_eval.py`):

- **Models.** The original `Qwen/Qwen2.5-0.5B` (revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, weights SHA-256 verified) loaded in float32, and a second copy in which the 168 matrices exported in Task 42 are overwritten with the values decoded from that task's actual `te.safetensors` container (rowwise E2M1 values, E4M3 block scales, amax, decoded exactly as the runtime recovers them). Every container matrix must match a model parameter by name and shape, or the run fails, so a mismatch cannot silently leave a weight unquantized. The tied embedding and all biases and norms stay at original precision, as the exporter preserves them.
- **Simulated quantization.** Both models run in float32 on a GPU through Hugging Face `transformers`. No Transformer Engine, activation quantization, or low-precision kernel is involved, so the result isolates the effect of the quantized *weights*. A real Transformer Engine forward pass also quantizes activations and would differ.
- **Text.** The WikiText-2 (`wikitext-2-raw-v1`) test split from `Salesforce/wikitext` at revision `b08601e04326c79dfdd32d625aee71d232d685c3` (SHA-256 verified; licensed CC BY-SA 3.0 and GFDL), joined with blank lines and tokenized with the model's tokenizer into 146 non-overlapping windows of 2048 tokens (299,078 tokens, of which 298,862 are scored).
- **Metrics**, over the same windows: perplexity of each model; the mean KL divergence KL(original || quantized) of the next-token distributions over all scored positions; and the fraction of positions where both models predict the same top-1 token. The metric code is unit-tested against direct computations on a toy model (including the KL definition and perplexity), and the baseline must land in a plausible range or the run aborts.

Environment: PyTorch 2.9.0+cu128, `transformers` 4.57.6, an NVIDIA L4, one run (209 s in total including the downloads, on an L4). Raw result: [`docs/validation/nvfp4-quality-qwen2.5-0.5b-wikitext2.json`](../validation/nvfp4-quality-qwen2.5-0.5b-wikitext2.json).

## Results

| | Original (fp32) | NVFP4 weights |
| --- | --- | --- |
| Perplexity on WikiText-2 | 13.07 | 14.37 (+9.93%) |
| Mean KL(original \|\| NVFP4) | | 0.1009 nats/token |
| Top-1 agreement with the original | | 83.49% |

The original's perplexity of 13.07 is plausible for a 0.5B model on WikiText-2 and passed the run's sanity bound (between 3 and 60); it was not checked against a published figure for this exact setup. The per-matrix relative Frobenius error between the original and decoded weights is uniform: mean 9.44%, median 9.45%, maximum 9.56% across all 168 matrices. That uniformity is expected for round-to-nearest quantization on a fixed block structure, and it matches the weight-reconstruction figures of the conversion report (lowest SQNR about 20.4 dB).

## Interpretation

Replacing the weights with their NVFP4 values costs this model about a tenth of its perplexity on this text, and the two models disagree on the most likely next token at roughly one position in six. That is a visible loss, not a negligible one, and it is the cost of this particular scheme: round-to-nearest, one E4M3 scale per 16 values, a per-tensor float scale, no calibration, no clipping search, weight-only.

What this does **not** establish:

- **One model, one text, one run.** The figures describe Qwen2.5-0.5B on WikiText-2 with 2048-token windows. Other models, sizes, tasks, and context lengths can behave differently; no downstream task accuracy (question answering, code, instruction following) was measured.
- **No comparison point.** No other quantization method or precision was evaluated alongside (an INT8 reference row was offered and declined), so the size of this loss relative to INT8, INT4, or other 4-bit schemes is unknown here.
- **Not the runtime's numerics.** Activation quantization, Transformer Engine's kernels, and accumulation order were not part of the measurement.
- **No threshold.** The run is a measurement, not a gate, and nothing here says whether +9.9% is acceptable for any purpose.

## Consequences

- The question left open by ADR 0023 has a first, bounded answer, and the claim about the export is now stated more completely: loads and multiplies correctly on a B200 (ADR 0023), at a measured cost of about +10% perplexity on this model and text (this ADR).
- If a lower loss is wanted, the measurement gives a baseline for trying improvements such as better scale selection, clipping, or the 2D 16x16 blocks Transformer Engine uses for weights; each would be judged against these numbers.
- `modal_quality_eval.py --name <volume-directory>` reruns the evaluation for a container produced by `validate_model`; it downloads the model and dataset inside Modal and spends a few minutes of L4 time.
