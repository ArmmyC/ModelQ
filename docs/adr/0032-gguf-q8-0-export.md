# ADR 0032: GGUF Q8_0 Export for Qwen2, Verified with llama.cpp

- Status: Accepted (Task 51, milestone M5 of ADR 0028). The claim is runtime-compatible for Qwen2 models in Q8_0 on CPU, with llama.cpp v0.6.0. No other architecture, quantization type, or backend is claimed.
- Date: 2026-10-10
- Scope: `modelq quantize <model-dir> --format gguf-q8_0 --output <file>.gguf`, for the `qwen2` architecture. The Q8_0 block layout is ADR 0008's; this ADR adds the model-level layout, the tokenizer, and the runtime verification that ADR 0008 deliberately did not do.
- Builds on: [ADR 0008](0008-gguf-q8-0-compatibility-spike.md) (the Q8_0 representation and the one-tensor spike), [ADR 0028](0028-open-source-quantizer-scope.md) (section 5 and M5), [ADR 0030](0030-group-wise-low-bit-formats.md) (status levels)

## Context

ADR 0008 established the Q8_0 block layout and a container-valid one-tensor fixture (Level 2). It stated that a runnable model needs a valid architecture and a runtime load path (Level 3). ADR 0028 asks for one GGUF quantization type that a runtime loads for a real model, since local users mostly run GGUF files.

## Decisions

### 1. One type, one architecture, one runtime

Q8_0 is the first type because it is the simplest exact block layout (32 values, one F16 scale, 32 signed bytes) and its quantization rule can be checked byte-for-byte against the reference. The architecture is `qwen2`, because it is the architecture ModelQ's measurements already cover (ADR 0024 to 0031). The runtime is llama.cpp, pinned to its release tag **`v0.6.0`** (commit `d81235049384534c167caea52b85a694f6103d14`). ADR 0008's commit `d775b89` is still a valid reference for the one-tensor fixture; the model check uses the release tag.

### 2. The file layout

The writer (`crates/modelq-io/src/gguf_writer.rs`) emits GGUF version 3 with the standard header, the key-value records, tensor-info records, a 32-byte alignment for every tensor offset, and dimensions in GGML order (the reverse of the row-major shape). Metadata written:

- `general.architecture = "qwen2"`, `general.name` (the model directory's name), `general.file_type = 7` (`MOSTLY_Q8_0`), `general.quantization_version = 2`, `general.alignment = 32`;
- `qwen2.context_length`, `qwen2.embedding_length`, `qwen2.feed_forward_length`, `qwen2.block_count`, `qwen2.attention.head_count`, `qwen2.attention.head_count_kv`, `qwen2.attention.layer_norm_rms_epsilon`, `qwen2.rope.freq_base`, from `config.json`;
- the tokenizer: `tokenizer.ggml.model = "gpt2"` (byte-level BPE), `tokenizer.ggml.pre = "qwen2"`, `tokenizer.ggml.tokens`, `tokenizer.ggml.token_type`, `tokenizer.ggml.merges`, and `tokenizer.ggml.eos_token_id` (from `tokenizer.json` and `tokenizer_config.json`).

Ids the tokenizer does not define are written as `[PAD<id>]` with type *unused*, because llama.cpp needs one token per embedding row (151,936 for Qwen2.5-0.5B). Added tokens are *control* if special and *user-defined* otherwise.

### 3. Names, types, and refusals

Checkpoint names map to llama.cpp's names (`model.layers.N.self_attn.q_proj.weight` → `blk.N.attn_q.weight`, and so on). Two-dimensional weights, including the embedding, become Q8_0; one-dimensional norms and biases stay F32, as llama.cpp's own Q8_0 conversion does. Every tensor is mapped or the export fails: an unknown tensor, a missing tensor, an unsupported architecture (`model_type` other than `qwen2`), an unsupported tokenizer, or an existing output file all stop the export before anything is written.

### 4. The quantization rule

Each block of 32 values has scale `d = max|x| / 127` computed in F32. Values are multiplied by `1/d`, rounded half away from zero, and clamped to ±127. The scale is stored as binary16 (`half` crate, round to nearest even). This is ADR 0008's rule, and it is the rule llama.cpp uses.

### 5. Hub input

`modelq quantize hf:<owner>/<name> --format gguf-q8_0` also fetches `config.json`, `tokenizer.json` and `tokenizer_config.json` (each verified against the Hub's checksums, as the weights are; ADR 0029). The fetch fails if a file is missing.

## Verification

Everything below was run on Linux in Modal, against the pinned release and the pinned Qwen2.5-0.5B revision `060db6499f32`. Raw evidence: [`m5-gguf-q8_0-qwen2.5-0.5b-llama-cpp.json`](../validation/m5-gguf-q8_0-qwen2.5-0.5b-llama-cpp.json).

1. **Export.** `modelq quantize hf:Qwen/Qwen2.5-0.5B --format gguf-q8_0` exits 0, writes 531,064,896 bytes (the BF16 source is 988,097,824), and reports 169 Q8_0 tensors and 121 F32 tensors.

2. **Independent reader.** llama.cpp's own GGUF library (the `gguf` package) reads the file: architecture `qwen2`, vocabulary 151,936, 290 tensors. Each tensor, decoded, matches the source weight: F32 tensors exactly, and Q8_0 tensors within `(0.5 + 127 * 2^-11) * max|w| / 127`. That bound allows for the binary16 scale. The worst absolute error is 0.0063. A stricter bound (half a step) was tried first and was too strict; the arithmetic is in the verification script.

3. **Tokenizer.** llama.cpp's tokenizer produces exactly the Hugging Face token ids for the whole WikiText-2 test text: 299,078 tokens, no mismatches.

4. **Loads and runs.** `llama-completion` loads the file and generates from the prompt "The capital of France is" with greedy decoding: " Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of France".

5. **Quality, same tokens.** llama.cpp's perplexity scores only the second half of each window (`first = n_ctx / 2`; see `tools/perplexity` in the pinned release). With that rule, on the first 20 windows of 2048 tokens (20,460 scored tokens):

   | Measurement | Perplexity |
   | --- | --- |
   | llama.cpp, Q8_0 file | 11.332 |
   | Python, decoded Q8_0 weights | 11.310 |
   | Python, original model | 11.304 |

   The Q8_0 file adds 0.05% over the original. The runtime is 0.2% above the decoded weights. The likely cause is that llama.cpp's Q8_0 matrix kernels also quantize the activations, but that is not measured here.

   The scoring rule matters for comparisons. Scoring every position of each window, as ADR 0024 to 0031 do, gives 12.70 for the same windows. The two numbers measure different things and must not be compared.

## Status

The status is **runtime-compatible: llama.cpp v0.6.0 (Qwen2, CPU)**. The fixture-level claim of ADR 0008 (container-valid) is superseded for files that hold a whole model. The Rust integration tests (`tests/gguf_export.rs`) cover the export's structure, its counts, and its refusals, and they run on Linux in Modal.

## What this does not show

- One architecture, one model size, one quantization type. Other `qwen2` models with a different number of layers, heads, or vocabulary are not measured, and no other architecture is supported.
- CPU only. The GPU backends of llama.cpp are not verified.
- One prompt for generation and 20 windows for perplexity. Output quality beyond these checks is not evaluated.
- The file is single-file only. Sharded GGUF output is not written.
- No chat template and no `general.description`. The file is for completion, not for chat.

## Consequences

- Users with a Qwen2 checkpoint can produce a file that llama.cpp loads, with one command, on their own machine.
- The ModelQ quantization rules (ADR 0008) and the llama.cpp reference agree at the block level, so Q8_0 output is comparable with llama.cpp's own conversion.
- Other GGUF types (Q4_K_M and the others) are separate decisions with their own references; the next one is a separate ADR.
