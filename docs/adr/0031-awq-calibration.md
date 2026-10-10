# ADR 0031: AWQ Calibration for Group-Wise INT4

- Status: Accepted for measurement (Task 50, milestone M4 of ADR 0028). It is not the default: data-free quantization stays the default, as ADR 0028 section 4 requires.
- Date: 2026-10-10
- Scope: activation-aware weight scaling (AWQ) as the first calibration method. It produces a rescaled checkpoint, which the ordinary data-free quantizer then turns into a ModelQ group-wise INT4 file (ADR 0030). The Rust quantizer, the file format and the default policy are unchanged.
- Builds on: [ADR 0028](0028-open-source-quantizer-scope.md) (sections 4 and M4), [ADR 0030](0030-group-wise-low-bit-formats.md) (the INT4 format), [ADR 0024](0024-nvfp4-output-quality-qwen2-5-0-5b.md) and [ADR 0026](0026-block-scale-search-across-models.md) (the evaluation protocol)

## Context

ADR 0028 added calibration to the roadmap, but only once a data-free baseline was measured. M3 gave us group-wise INT4 and a measured baseline is needed before calibration can be judged: the acceptance rule is that a calibration method must beat the data-free method on the same evaluation, or this ADR records that it does not.

Data-free quantization rounds each weight group to its own largest value. It knows nothing about which input channels carry large activations, and an error in a salient channel costs more than the same error elsewhere. AWQ (Lin et al., 2023) addresses this by scaling the salient channels up before quantization, so that their weights keep more precision, and folding the scale back out of the computation.

## Decisions

### 1. AWQ first; GPTQ later behind the same interface

AWQ is the first method for two reasons. It needs one forward pass over calibration text and a small search per block, so it is simple to verify exactly. The interface is a function from a model and calibration windows to per-block scales; GPTQ (second-order rounding within each layer) can implement the same interface later and be measured against this result.

### 2. The scaling is an exact change of variables

For a block of linear layers that read the same input, the scale `s` multiplies the weight columns and divides the input, so `W x = (W diag(s)) (diag(1/s) x)`. The division is folded into whatever produces the input:

- **A:** `input_layernorm` divides, and `q_proj`, `k_proj`, `v_proj` columns multiply.
- **C:** `post_attention_layernorm` divides, and `gate_proj`, `up_proj` columns multiply.
- **D:** `up_proj` rows divide, and `down_proj` columns multiply.

The attention value/output pair (`v_proj` to `o_proj`) is not scaled. With grouped-query attention, one value channel feeds several query heads, so the change of variables cannot be applied per output channel there. Qwen2.5-0.5B uses 14 query heads and 2 key/value heads.

The rescaled checkpoint computes the same function as the original, and this is tested rather than assumed: a tiny model's logits must match within 1e-4 (`tools/calibration/test_awq.py`), and the full model's rescaled, unquantized checkpoint gives the original perplexity exactly (below).

### 3. The search

Each block's scale is `s = mean|x|^alpha`, normalized to unit geometric spread. Twenty values of `alpha` in [0, 1) are tried, and each is scored by the squared output error of the block's linear layers on calibration activations, where each candidate's weights are quantized with ModelQ's INT4 rules (symmetric, group 128, ties away from zero). The score is `||X W^T - X (Q(W diag(s)) diag(1/s))^T||^2` summed over the block. The search uses only the float model's activations, which are unchanged by the folding.

### 4. Calibration data are named and recorded

- **Calibration:** the WikiText-2 raw train split, `Salesforce/wikitext` at revision `b08601e0`, file `train-00000-of-00001.parquet`, SHA-256 verified. Thirty-two windows of 512 tokens (16,384 tokens) are taken at evenly spaced offsets (0, 81223, ... 2517911 out of 2,518,423 train tokens); the offsets are recorded.
- **Evaluation:** the WikiText-2 raw test split of the same revision, 146 windows of 2048 tokens, as in ADR 0024 and ADR 0026.

The train and test splits are disjoint documents of the same corpus. That is a weaker separation than a different corpus would give, and it favors calibration: the activations on test text resemble those on train text. Section "What this does not show" returns to this.

### 5. Policy for the embedding and output head

The data-free INT4 path currently quantizes every floating tensor above the minimum size, including the tied embedding and output head (169 quantized tensors). NVFP4's default preserves them (`embed_tokens`, `lm_head`, `embeddings`; the decision recorded in the NVFP4 work). The measurement below reports both policies, evaluated from the same files: the embedding kept at the model's values is the NVFP4-style default.

## Results

Qwen2.5-0.5B, float32 simulated INT4 (weight-only), group 128, 146 windows of 2048 tokens of WikiText-2 test (298,862 scored positions), L4 GPU. The original model's perplexity is 13.0699.

| Variant | Perplexity | Increase | KL (nats/token) | Top-1 agreement |
| --- | --- | --- | --- | --- |
| Rescaled checkpoint, unquantized (equivalence) | 13.0699 | +0.000% | 0.000000 | 100.00% |
| Data-free INT4, embedding quantized (current default) | 19.8858 | +52.15% | 0.4319 | 67.15% |
| Data-free INT4, embedding kept | 18.1918 | +39.19% | 0.3456 | 70.76% |
| **AWQ INT4, embedding quantized** | **17.4853** | **+33.78%** | **0.2962** | **72.33%** |
| **AWQ INT4, embedding kept** | **16.0571** | **+22.86%** | **0.2132** | **77.17%** |

The evaluation files are the ones `modelq quantize` wrote: `--format int4 --group-size 128` on the original checkpoint (data-free) and on the rescaled checkpoint (AWQ). The decoded weights match the Python quantization of their source exactly: the cross-check compares all 168 projection matrices and finds a maximum difference of 0.

AWQ reduces the perplexity increase by 35% (embedding quantized) and by 42% (embedding kept), relative to the matching data-free variant. Both reductions are measured on the same windows.

The calibration search reduced the summed block output error by 27% (`error_unscaled` 43.2 million, `error_scaled` 31.4 million). The chosen alphas were mostly 0.2 to 0.3 (58 of the 72 blocks), with a few larger values.

Raw result, with the calibration record (offsets, search summary per block, the Rust quantize reports and the cross-check): [`m4-awq-qwen2.5-0.5b-int4-wikitext2.json`](../validation/m4-awq-qwen2.5-0.5b-int4-wikitext2.json).

## Verification

- Python unit tests (33, run locally): round-half-away rounding, including the value just below one half that a `floor(|x| + 0.5)` rule gets wrong in float32; the 2- and 3-bit golden vectors shared with the Rust tests; the rescaled checkpoint's logits equal the original's; untransformed tensors are byte-identical; the error summary is consistent; the low-bit reader decodes a written file to the quantizer's values; the golden decodes match Rust.
- Cross-language: the Rust INT4 output's codes and scales equal the Python quantizer's on all 169 quantized tensors (zero mismatches). The evaluation's decoded weights equal the Python quantization of their source (maximum difference 0 on 168 matrices).
- Equivalence on the real model (see Results).

### Two defects found and fixed during measurement

These matter for anyone reproducing the numbers:

1. **Floating-point rounding.** `round(x)` was written as `sign(x) * floor(|x| + 0.5)`. In float32 this rounds 0.49999997 up to 1, and Rust's `f32::round` does not. The fraction is now taken as `x - trunc(x)`, which is exact.
2. **Division on the GPU.** `max_abs / 7` with a Python scalar divisor is computed on the GPU as a multiply by the reciprocal, which differs from the CPU and from Rust in the last bit for about half the groups. A division by a tensor is exact on both devices, and the scale now divides by a tensor. Before the fix, 111 million of 279 million test values differed from the CPU in their last bit, and at least one of those differences flipped a rounding tie and changed a code (this is the cross-check's 0.145 difference). After the fix, the GPU and the CPU agree on all 279 million values.

The first measurement had both defects. The final run (reported here) has neither. The two runs produced the same alpha histogram and the same perplexities, so in this case the defects did not change the chosen scales; they did break the cross-check.

## What this does not show

- **One model, one calibration sample, one run.** The result is for Qwen2.5-0.5B only. Calibration used 32 windows of 512 tokens, fewer than the 128 windows commonly used for AWQ, and one set of evenly spaced windows. No variance across window choices or seeds was measured.
- **In-distribution calibration.** Calibration and evaluation both come from WikiText-2, so the gain may be larger than on unrelated text. A held-out corpus is the next test before any broader claim.
- **Simulated quantization.** Weights are decoded to float32; no kernel, no activation quantization. The output format is a ModelQ-native file that no runtime reads (ADR 0030), so no runtime compatibility is claimed.
- **Not all blocks are scaled.** The v-to-o pair is skipped because of grouped-query attention. Including it could add gain.
- **No downstream evaluation.** Perplexity, KL and top-1 agreement on one text are proxies, not task accuracy.

## Decision

- AWQ is accepted as the first calibration method: on the same evaluation it beats the data-free method under both policies (ADR 0028 M4 acceptance).
- It is not the default. The data-free path remains the default, and calibration is opt-in.
- The embedding policy is the open decision. The embedding-kept variant is better for both methods, and it matches the NVFP4 default. Making it the INT4 default would change the bytes of every default INT4 output, so it should be a separate decision, as the scale-search default was (ADR 0027). Decided in [ADR 0036](0036-int4-default.md): the INT4 default keeps the embedding at its source precision.

## Consequences

- The calibration tool is `tools/calibration/awq.py` with the Modal experiment `tools/calibration/modal_awq.py`. It writes a rescaled SafeTensors checkpoint and a record; `modelq quantize` then produces the INT4 file. The Python evaluator reads both low-bit files and plain checkpoints (`tools/quality_eval/quality_eval.py`).
- A user runs AWQ as two commands today: the calibration script, then `modelq quantize --format int4`. Wiring it into `modelq quantize` behind a `--calibration awq` flag is a follow-up; it needs the Python sidecar to be packaged with the binary (M6), as the evaluation already does.
- The rescaled checkpoint stores the transformed tensors as F32 and the rest unchanged; the output stores the same bytes as the data-free path for the untransformed tensors.
- Any later calibration method (GPTQ) is measured against this ADR's numbers, on the same windows and the same policy variants.
