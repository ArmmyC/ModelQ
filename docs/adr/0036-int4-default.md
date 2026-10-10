# ADR 0036: The INT4 Default Keeps the Vocabulary Matrices at Source Precision

- Status: Accepted (Task 57). It settles the open embedding-policy decision of ADR 0028 and ADR 0031. It is accepted on the end-to-end run below.
- Date: 2026-10-10
- Scope: the default of `modelq quantize --format int4`. The token embedding and the output head, when a checkpoint has its own, are preserved at their source precision. The names are the ones the NVFP4 default preserves: `embed_tokens`, `lm_head` and `embeddings`. The other two-dimensional weights stay INT4, and one-dimensional tensors stay as they were. The file layout and the INT4 codec are unchanged.
- Builds on: [ADR 0030](0030-group-wise-low-bit-formats.md) (the group-wise format), [ADR 0031](0031-awq-calibration.md) (the measurement, and the open decision it left), [ADR 0027](0027-scale-search-default.md) (the precedent for a default that changes output bytes and is decided on its own).

## Context

ADR 0031 measured the data-free INT4 default. It quantizes every floating tensor above the minimum size, including the tied embedding and output head. With the embedding kept at the model's values, the full-split perplexity cost falls from +52.15% to +39.19%.

The NVFP4 default already keeps those matrices, so INT4 was the outlier. Changing the INT4 default changes the bytes of every default INT4 output, so it was decided on its own, as ADR 0027 was.

The owner chose to keep the embedding at full precision by default.

## Decisions

1. The INT4 default preserves every tensor whose name contains `embed_tokens`, `lm_head` or `embeddings`. These are the NVFP4 default's names (`DEFAULT_EXCLUDED_NAME_PARTS`), so both formats agree on what a vocabulary matrix is.
2. A preserved tensor is written in its source dtype. For the pinned model that is BF16, so the copy is exact, and its size is the BF16 size, not the INT4 size.
3. The reason is recorded as `ExcludedByName` in the policy. The report counts the tensor as preserved.
4. The rule applies to INT4 only. int3, int2 and int1 are experimental, and they keep their default. Extending the rule to them is a one-line change, if it is wanted.
5. There is no flag for the old output. ADR 0031 holds its numbers.

## Verification

1. **Tests.** The Linux suite (Rust 1.85) passes with rustfmt clean. Clippy on Rust 1.99 exits 0. The new tests are a unit test in `modelq-quant` for `preserve_named`, and an integration test. The integration test runs `modelq quantize --format int4` on a source with `model.embed_tokens.weight` and a layer, and checks that the manifest records the embedding as preserved and the layer as quantized.

2. **End to end (Modal, L4).** The pinned Qwen2.5-0.5B revision was quantized with the default `modelq quantize --format int4` and evaluated with `modelq eval --model hf:Qwen/Qwen2.5-0.5B --download` on the full split. The plan was 290 source tensors: 168 quantized, and 122 preserved, which are the embedding and the 121 norms and biases. The reopened output validated, and the 122 preserved tensors are identical to the source.

3. **Anchor.** ADR 0031 measured this configuration, with the embedding kept at the model's values, at 18.191818421875144 on the full split. The eval reproduces it exactly.

## Results

| Default | Output size | Perplexity, full split | Increase |
| --- | --- | --- | --- |
| Before: embedding quantized | 262,701,736 bytes (263 MB) | 19.885816 | +52.15% |
| After: embedding preserved | 462,649,320 bytes (463 MB) | 18.191818 | +39.19% |

The original model is 13.069886 on the same split.

## What this does not show

- The untied output head follows the same rule, but no untied model was measured.
- One model, and the data-free quantizer only. AWQ with the embedding kept reached +22.86% in ADR 0031, but AWQ is not yet wired into `modelq quantize`.
- The experimental formats keep their previous default. Their quality with the embedding kept was not measured.

## Consequences

- A default INT4 file for this model is about 200 MB larger (199,947,584 bytes), and its perplexity cost is about 13 points lower.
- The smaller file is no longer available from the CLI. ADR 0031's record holds its numbers. A flag can be added if it is needed.
