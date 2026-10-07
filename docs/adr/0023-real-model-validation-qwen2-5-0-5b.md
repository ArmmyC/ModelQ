# ADR 0023: Real-Model Validation on Qwen2.5-0.5B

- Status: Accepted (Task 42)
- Date: 2026-10-07
- Scope: converting a real open checkpoint with `modelq quantize --format nvfp4-te` and proving every exported matrix on an NVIDIA B200 with Transformer Engine 2.19.0
- Result: **all 168 exported matrices loaded, dequantized to the reference, and passed their TN GEMM, in both the single-file and the sharded container.**
- Builds on: [ADR 0021](0021-transformer-engine-multi-matrix-container.md), [ADR 0022](0022-transformer-engine-multi-matrix-hardware-validation.md)

## Why a real model

ADR 0022 validated the container on synthetic matrices chosen to exercise padding and eligibility. A real checkpoint tests what a synthetic one cannot: real shapes (including non-square projections), real weight distributions, BF16 input, a tied embedding that must be preserved, and the checkpoint's own mix of weights and biases.

## The model

| | |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B` |
| Revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| License | Apache-2.0 (not gated) |
| File | `model.safetensors`, 988,097,824 bytes, BF16 |
| SHA-256 | `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` (matched the value Hugging Face publishes for the file) |

The model was chosen from three Apache-2.0 candidates (SmolLM2-135M, Qwen2.5-0.5B, TinyLlama-1.1B), all of whose linear weights satisfy the exporter's shape rules. The download happened inside Modal, pinned to that commit and checked against the published hash; nothing was downloaded to the development machine, and the weights are not stored in the repository.

## What was run

`tools/transformer_engine_nvfp4/modal_runtime_proof.py::validate_model` on Modal:

1. **CPU stage (Linux, Rust 1.85, built from this repository).** Download and verify the model; run the new `te_reference_from_source` example, which re-quantizes every eligible matrix from the *source* weights with the native NVFP4 quantizer and writes its dequantization as an F32 reference (an oracle independent of the container writer and decoders); convert the model with the real CLI to a single-file container and to a sharded container (`--max-shard-size 128MB`).
2. **GPU stage (NVIDIA B200).** `validate_multi.py runtime` on both containers: validate the container and reference on the CPU, then for every matrix build the pinned TE `NVFP4Tensor`, compare TE's dequantization with the reference (`rtol=1e-5`, `atol=1e-5`), and run one TN GEMM against the oracle (`rtol=0.125`, `atol=0.0675`).

Environment: the image and versions of ADR 0022 (Python 3.12.1, PyTorch 2.9.0+cu128, CUDA 12.8, cuDNN 9.10.2, driver 580.95.05, TE 2.19.0, NVIDIA B200, compute capability 10.0). Raw results: [`docs/validation/te-qwen2.5-0.5b-b200.json`](../validation/te-qwen2.5-0.5b-b200.json).

## Results

**Conversion** (CLI report): 168 matrices exported and 122 tensors preserved. The preserved tensors are the tied token embedding `model.embed_tokens.weight` `[151936, 896]` (excluded by the default name rule), the 72 attention biases, and the layer and final norm vectors. The output is 473,840,480 bytes against 988,097,824 for the source (smaller by 514,257,344 bytes); the preserved BF16 embedding is more than half of the output. Weight reconstruction over the exported matrices: maximum MSE 5.95e-5, maximum MAE 4.54e-3, maximum absolute error 0.176, lowest SQNR 20.39 dB. The whole fetch-and-convert stage took 61 s.

**Hardware proof**, identical for both containers:

| Matrix shape | Count | Worst TN GEMM max abs error |
| --- | --- | --- |
| `[128, 896]` (key and value projections) | 48 | 1.3e-5 |
| `[896, 896]` (query and output projections) | 48 | 1.5e-5 |
| `[896, 4864]` (MLP down projection) | 24 | 1.2e-5 |
| `[4864, 896]` (MLP gate and up projections) | 48 | 1.9e-5 |

All 168 passed. TE's dequantization differed from the independent reference by at most 1.2e-7 for every matrix, and the median GEMM error was 4.0e-6. No matrix needed an exclusion beyond those the exporter already applies, and the final-dimension-divisible-by-32 rule from ADR 0022 held on real shapes (the final dimensions are 896 = 28 x 32 and 4864 = 152 x 32).

## What this shows, and what it does not

It shows that the exporter, container, and pinned runtime agree on a real checkpoint end to end: every exported weight is accepted by Transformer Engine on a Blackwell GPU, is decoded to the value the reference predicts, and multiplies correctly in one forward TN GEMM, regardless of whether the container is one file or several shards.

It does **not** show that the quantized model is good. The oracle compares TE with the container's own interpretation of the quantized weights, so a large quantization error would not fail it. The only quality figures are weight-reconstruction errors against the original weights (lowest SQNR about 20 dB). Model-output quality such as perplexity or task accuracy was not measured, and neither was a forward pass of the model. Round-to-nearest weight-only NVFP4 with this block structure may or may not preserve accuracy for this model; that needs an inference stack, which is outside this project's scope as a quantization compiler. Also not established: other models and architectures, other TE versions, GPUs, or CUDA stacks, `te.Linear` integration, and anything beyond the loading-plus-one-GEMM scope of ADR 0022.

## Consequences

- The claim in ADR 0022 now rests on a real checkpoint with 168 matrices of four distinct shapes, not only on synthetic fixtures.
- `validate_model --model <id> --name <dir>` repeats the run for any ungated Apache-2.0 or MIT model with a single `model.safetensors`; multi-file checkpoints are not yet supported by that stage.
- The next informative step is output-quality evidence for the quantized weights, for which the exported container is an input but the measurement belongs to an inference stack.
