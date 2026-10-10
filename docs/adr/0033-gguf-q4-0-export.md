# ADR 0033: GGUF Q4_0 Export for Qwen2, Verified with llama.cpp

- Status: Accepted (Task 52, a follow-up to milestone M5 of ADR 0028). The claim is runtime-compatible for Qwen2 models in Q4_0 on CPU, with llama.cpp v0.6.0. No other architecture, quantization type, or backend is claimed. The file loads and runs, but it costs a lot of quality on the measured model (see Verification, item 5).
- Date: 2026-10-10
- Scope: `modelq quantize <model-dir | hf:...> --format gguf-q4_0 --output <file>.gguf`, for the `qwen2` architecture. It adds one block type to ADR 0032's writer and exporter. Metadata, tokenizer and tensor names are unchanged.
- Builds on: [ADR 0032](0032-gguf-q8-0-export.md) (the file layout, the tensor policy, and the runtime check), [ADR 0028](0028-open-source-quantizer-scope.md) (section 5 and open question 3, which name a 4-bit type for normal users).

## Context

ADR 0032 exports Q8_0, which is close to the original but still large for a laptop. Most people who run a local model download a 4-bit GGUF file, and ADR 0032 named a 4-bit type as the next GGUF type.

Q4_0 is the simplest 4-bit GGUF block format: one binary16 scale per 32 values and one 4-bit code per value, with no super-blocks, sub-scales or per-tensor type choices. The type most people download, Q4_K_M, has a different and much more complex rule (256-value super-blocks with six-bit sub-scales, and some tensors in another type). It is a separate decision with its own ADR.

## Decisions

### 1. The rule is llama.cpp's reference

The rule is the one in `quantize_row_q4_0_ref` and `dequantize_row_q4_0` in `ggml/src/ggml-quants.c`, with `block_q4_0` in `ggml/src/ggml-common.h`, at the pinned commit `d81235049384534c167caea52b85a694f6103d14`. A 4-bit file with another rule would not be Q4_0. Taking the rule from the reference also keeps the output comparable with llama.cpp's own conversions, as ADR 0032 did for Q8_0.

For each block of 32 values:

- `max` is the value with the largest magnitude, and the first one on a tie. The scale is `d = max / -8`, stored as binary16 with round to nearest even. So `d` has the opposite sign to `max`.
- Each value `x` gets the code `min(15, trunc(x * (1 / d) + 8.5))`. The product and the sum are each rounded in single precision, with no fused multiply-add.
- The code `q` decodes to `(q - 8) * d`.
- Element `j` for `j` in 0 to 15 is the low nibble of byte `j`. Element `j + 16` is the high nibble of byte `j`. A block is 18 bytes.
- A zero block stores the negative-zero scale (`0x8000`) and codes of 8, which decode to zero.

Two consequences differ from ADR 0008's Q8_0 rule:

- Ties round toward +infinity, so -7.5 steps becomes -7. Q8_0 rounds ties away from zero.
- The codes span -8 to +7 steps. The extreme on the side opposite to `max` clamps to 7 steps, so one value per block can be off by a full step. The error bound is `(1 + 2^-8) * max|block| / 8`, not half a step.

llama.cpp's own build may fuse the multiply and add. The gguf-py quantizer notes that its reference "depends on FMA". A fused build can change individual codes on exact ties, but it does not change the format.

### 2. The tensor policy is ADR 0032's

Two-dimensional weights become Q4_0, including the token embedding and the output tensor when a checkpoint has one. One-dimensional norms and biases stay F32.

llama.cpp's quantizer follows the same policy. In `src/llama-quant.cpp` at the pinned commit, the `*_norm.weight` tensors are excluded (line 300), and `output.weight` is quantized unless `quantize_output_tensor` is off (line 302). No condition excludes the embedding. The one Q4_0-specific rule there moves some `ffn_down` layers to Q4_1 (lines 661 to 666). It applies only when an importance matrix is given, and ModelQ does not use one.

Quantizing the embedding costs more quality at 4 bits than at 8. The verification measures the total cost, and the follow-up in Verification item 6 splits it. Keeping the embedding at a higher precision is still an open decision (ADR 0028), and it would change this format's output bytes.

### 3. Status

The status starts as `representation-valid`: the rule has a written specification (this ADR), golden vectors and property tests in `crates/modelq-quant/src/gguf_q4_0.rs`, and a byte comparison with the reference quantizer. It moves to `runtime-compatible: llama.cpp v0.6.0 (Qwen2, CPU)` once the four acceptance criteria below pass.

## Acceptance criteria

Fixed before the run:

1. Every Q4_0 block decodes within `(1 + 2^-8) * max|block| / 8` of its source values.
2. The stored bytes equal gguf-py's Q4_0 quantizer output for every block. gguf-py mirrors llama.cpp's reference, so this checks that the rule is implemented as written.
3. llama.cpp loads the file, and `llama-completion` generates from the prompt.
4. llama.cpp's perplexity on the same 20 windows is within 1% of the Python evaluation of the decoded file under llama.cpp's scoring rule. The 1% tolerance is the same order as the Q8_0 gap (0.2%, ADR 0032) with room for activation-quantization differences.

## Verification

Everything ran on Linux in Modal, against the pinned release and the pinned Qwen2.5-0.5B revision `060db6499f32`. Raw evidence: [`m5-gguf-q4_0-qwen2.5-0.5b-llama-cpp.json`](../validation/m5-gguf-q4_0-qwen2.5-0.5b-llama-cpp.json). The verification script is `tools/gguf/modal_llama_verify.py`, run with `--format-id gguf-q4_0`.

1. **Tests.** The workspace suite on Linux (Rust 1.85): 350 passed, 0 failed, 2 ignored, with `cargo fmt --check` clean. The new tests cover the golden block for `x_i = i - 16` (bytes `0x00 0x40`, then `0x80 0x91 0x91 0xA2 ...`), the zero block, the sign and clamp case, the per-block bound, the stored scale, and the export's tensor types and file type. The CI-equivalent clippy on Rust 1.99 exits 0 with no warnings.

2. **Export.** `modelq quantize hf:Qwen/Qwen2.5-0.5B --format gguf-q4_0` exits 0 and writes 284,084,288 bytes, 53% of the Q8_0 file. It reports 169 Q4_0 tensors and 121 F32 tensors, with 19 metadata entries and a vocabulary of 151,936.

3. **Rule and reader.** llama.cpp's gguf-py reads the file: architecture `qwen2`, 290 tensors, vocabulary 151,936. Every block is within its bound (criterion 1); the worst absolute error is 0.1035. All 15,436,288 blocks have the same bytes as gguf-py's Q4_0 quantizer (criterion 2).

4. **Loads and runs.** `llama-completion` loads the file and generates 24 greedy tokens from "The capital of France is": ` ______.\nA. Paris\nB. London\nC. Tokyo\nD. Shanghai\nAnswer: A\n\nThe most\n\n`. The text is fluent, but this is one prompt. It is not a quality measure. The Q8_0 file continued the same prompt differently.

5. **Quality, on the same tokens.** llama.cpp's perplexity scores only the second half of each window (see ADR 0032). On 20 windows of 2048 tokens (20,460 scored tokens):

   | Measurement | Perplexity |
   | --- | --- |
   | llama.cpp, Q4_0 file | 13.124 |
   | Python, decoded Q4_0 weights | 13.106 |
   | Python, original model | 11.304 |
   | Python, decoded Q8_0 weights (ADR 0032) | 11.310 |

   llama.cpp is 0.14% above the decoded Python value (criterion 4, within 1%). Under this rule the Q4_0 file is 15.9% above the original; the Q8_0 file was 0.05% above. Scoring every position of each window gives 12.701 for the original and 14.780 for the decoded Q4_0 file (+16.4%), so the two rules agree on the size of the cost.

   This is a large cost. The file runs and matches the runtime, but on this model it loses a lot of quality. The split between the embedding and the linear layers is measured in the follow-up below.

6. **Follow-up: where the cost comes from.** The same Python evaluation was repeated with each part of the model at its original, Q4_0 or Q8_0 value (`tools/gguf/modal_embedding_split.py`; evidence: [`m5-gguf-q4_0-embedding-split-qwen2.5-0.5b.json`](../validation/m5-gguf-q4_0-embedding-split-qwen2.5-0.5b.json)). Under llama.cpp's rule:

   | Embedding (`model.embed_tokens`) | Linear layers | Perplexity | Increase | File (estimate) |
   | --- | --- | --- | --- | --- |
   | original | original | 11.304 | 0 | 752 MB |
   | Q4_0 | Q4_0 (the file above) | 13.106 | +15.9% | 284 MB |
   | Q4_0 | original | 11.714 | +3.6% | 284 MB |
   | original | Q4_0 | 12.597 | +11.4% | 752 MB |
   | Q8_0 | Q4_0 | 12.605 | +11.5% | 352 MB |

   The embedding accounts for about 4.5 of the 15.9 points: the 4-bit embedding costs 3.6 points alone, and 4.5 points on top of the 4-bit linear layers. Keeping it at Q8_0 recovers almost all of that, for about 68 MB more than the Q4_0 file. Full precision improves on Q8_0 by only 0.07 points, and costs about 468 MB more. The 4-bit linear layers cost 11.4 points alone; the rest of the 15.9 is an interaction of about 0.9 points. The Q4_0 round trip reproduces ADR 0033's numbers exactly, so the method is the same one.

   The Q8_0 embedding here comes from gguf-py's quantizer, which can differ from ModelQ's Q8_0 on exact rounding ties. The effect of that difference is far smaller than the differences measured here.

## Status

The status is **runtime-compatible: llama.cpp v0.6.0 (Qwen2, CPU)**. All four acceptance criteria pass. The status says that the file loads and runs in that runtime. It does not say that the file is good: item 5 gives the quality cost.

## What this does not show

- One architecture, one model size, one quantization type. Other `qwen2` models, and other architectures, are not measured.
- CPU only. The GPU backends of llama.cpp are not verified.
- The quality cost is measured on one model, with 20 windows and one prompt. The 16% increase is what this file costs on Qwen2.5-0.5B. Larger models usually lose less. The follow-up measures the split between the embedding and the linear layers on this model only.
- Q4_0 is the simplest 4-bit type. Q4_K_M, which most people download, is generally more accurate at a similar size. ModelQ does not write or verify it; that is the next decision.
- The file is single-file only, with no chat template.

## Consequences

- A user can produce a 4-bit Qwen2 file in one command that llama.cpp loads and runs, at about half the size of the Q8_0 file.
- On this model the file is much worse than the Q8_0 file. A user who needs quality should use Q8_0 until a more accurate 4-bit type exists.
- Whether the embedding stays at a higher precision is the open decision (ADR 0028). The follow-up shows that a Q8_0 embedding recovers almost all of the embedding's share of the cost, for about 68 MB. The export does not change: the policy is still llama.cpp's, so the Q4_0 file keeps the 4-bit embedding. Changing that default changes the output bytes, so it needs a separate decision.
- The linear layers, at 4 bits, account for about 11 of the 16 points. That part is not addressed by the embedding choice; it is what a more accurate 4-bit type such as Q4_K_M would have to improve.
