# ADR 0034: GGUF Q4_0 with an 8-bit Vocabulary Matrix

- Status: Accepted (Task 54). The claim is runtime-compatible for Qwen2 models in this format on CPU, with llama.cpp v0.6.0. It supersedes decision 2 of ADR 0033 (the tensor policy for Q4_0).
- Date: 2026-10-10
- Scope: `modelq quantize <model-dir | hf:...> --format gguf-q4_0 --output <file>.gguf`, for the `qwen2` architecture. Only the tensor policy of the Q4_0 export changes. The block rules, metadata, tokenizer and tensor names are unchanged.
- Builds on: [ADR 0033](0033-gguf-q4-0-export.md) (the Q4_0 rule, and the follow-up measurement of where its cost comes from), [ADR 0032](0032-gguf-q8-0-export.md) (the file layout and the runtime check), [ADR 0028](0028-open-source-quantizer-scope.md) (the open embedding-policy question).

## Context

ADR 0033 wrote the Q4_0 file with the token embedding at 4 bits, as llama.cpp's reference does. Its follow-up measured the cost on Qwen2.5-0.5B under llama.cpp's scoring rule. The all-Q4_0 file is 15.9% above the original model's perplexity. The 4-bit embedding accounts for about 4.5 of those points. A Q8_0 embedding recovers almost all of them for about 68 MB more than the Q4_0 file. Full precision adds only 0.07 points more over Q8_0, for about 400 MB more.

The owner chose the Q8_0 embedding as the policy for the Q4_0 export.

## Decisions

### 1. The vocabulary matrices are Q8_0

Under the Q4_0 format, the two vocabulary-sized matrices are written as Q8_0, with ADR 0008's block rule:

- `token_embd.weight`, the token embedding. For a tied checkpoint it is also the output head, and it is the only vocabulary matrix.
- `output.weight`, the output head, when a checkpoint has its own (an untied checkpoint).

Every other two-dimensional weight is Q4_0, with ADR 0033's rule. One-dimensional norms and biases stay F32.

Both matrices have one row per vocabulary entry. The embedding is the measured case; the untied output head follows the same rule but is not measured (see "What this does not show").

### 2. The file type stays MOSTLY_Q4_0

`general.file_type` stays `MOSTLY_Q4_0` (2). It names the dominant block type. llama.cpp has no file type for this mixture. The verification shows that llama.cpp loads the file with this value. It does not show anything else about how llama.cpp uses the value.

### 3. The layout is no longer llama.cpp's Q4_0 layout

llama.cpp's own Q4_0 conversion quantizes the embedding at 4 bits, so the two files differ in that one tensor. Each tensor still follows its own reference rule: the Q4_0 tensors follow ADR 0033, and the Q8_0 tensor follows ADR 0008.

### 4. Status

The status is `runtime-compatible: llama.cpp v0.6.0 (Qwen2, CPU)`, because the four acceptance criteria below pass.

## Acceptance criteria

Fixed before the run:

1. Every Q4_0 block decodes within `(1 + 2^-8) * max|block| / 8` of its source. Every Q8_0 tensor decodes within its per-tensor bound (ADR 0032).
2. The stored bytes of every Q4_0 block equal gguf-py's Q4_0 quantizer output.
3. llama.cpp loads the file, and `llama-completion` generates from the prompt.
4. llama.cpp's perplexity on the same 20 windows is within 1% of the Python evaluation of the decoded file, under llama.cpp's scoring rule.

## Verification

Everything ran on Linux in Modal, against the pinned release and the pinned Qwen2.5-0.5B revision `060db6499f32`. Raw evidence: [`m5-gguf-q4_0-q8-vocabulary-qwen2.5-0.5b-llama-cpp.json`](../validation/m5-gguf-q4_0-q8-vocabulary-qwen2.5-0.5b-llama-cpp.json). The script is `tools/gguf/modal_llama_verify.py`, run with `--format-id gguf-q4_0`.

1. **Tests.** The workspace suite on Linux (Rust 1.85): 351 passed, 0 failed, 2 ignored, with rustfmt clean. That is one more test than ADR 0033, the untied-checkpoint test, which checks that the output head is Q8_0 and the linear layers are Q4_0. CI-equivalent clippy on Rust 1.99 exits 0 with no warnings.

2. **Export.** `modelq quantize hf:Qwen/Qwen2.5-0.5B --format gguf-q4_0` exits 0 and writes 352,151,616 bytes, as predicted by the follow-up measurement. It reports 168 Q4_0 tensors, 1 Q8_0 tensor (the tied embedding) and 121 F32 tensors.

3. **Rule and reader.** llama.cpp's gguf-py reads the file: architecture `qwen2`, 290 tensors. Every Q4_0 block is within its bound (criterion 1). All 11,182,080 Q4_0 blocks have the same bytes as gguf-py's Q4_0 quantizer (criterion 2). The Q8_0 embedding decodes with a worst absolute error of 0.00079, within its bound of 0.00091.

4. **Loads and runs.** `llama-completion` loads the file and generates 24 greedy tokens from "The capital of France is":

   ` Paris. It is the capital of France. It is the capital of France. It is the capital of France. It\n\n`

   The continuation repeats one sentence. That is a quality observation about greedy decoding on a small model at 4 bits, not a runtime failure. It is one prompt, not a quality measure. Criterion 3 requires only that the file loads and generates.

5. **Quality, on the same tokens.** On 20 windows of 2048 tokens (20,460 scored tokens), under llama.cpp's scoring rule:

   | Measurement | Perplexity | vs original |
   | --- | --- | --- |
   | Python, original model | 11.304 | — |
   | Python, decoded file | 12.605 | +11.5% |
   | llama.cpp, file | 12.628 | +11.7% |

   llama.cpp is 0.18% above the decoded Python value (criterion 4, within 1%). Scoring every position gives 12.701 for the original and 14.217 for the decoded file (+11.9%). The Python value agrees with the follow-up measurement's Q8_0-embedding variant, to four decimals.

   For comparison, ADR 0033's all-Q4_0 file is +15.9% and the Q8_0 file is +0.05%.

## Status

The status is **runtime-compatible: llama.cpp v0.6.0 (Qwen2, CPU)**. All four acceptance criteria pass. The status says that the file loads and runs in that runtime. It does not say that the file is good: item 5 gives the quality cost.

## What this does not show

- The perplexity is measured on Qwen2.5-0.5B, whose output head is tied to the embedding. An untied checkpoint follows the same rule, but its output head is not measured.
- One architecture, one model size, one prompt, 20 windows, CPU only, single-file output only.
- The remaining cost, about 11.5 points, comes from the 4-bit linear layers. This decision does not change it. A more accurate 4-bit type (Q4_K_M, a separate ADR) would have to improve that part.

## Consequences

- The Q4_0 file is 352 MB, against 284 MB with a 4-bit embedding and 531 MB for Q8_0. Its perplexity cost on this model drops from about 16% to about 11.5%.
- The file no longer matches llama.cpp's own Q4_0 file in the embedding tensor. Anyone comparing the two should expect that difference.
- ADR 0033's decision 2 is superseded by this ADR.
