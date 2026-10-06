# ADR 0014: NVFP4 CLI Export over Sharded Input

- Status: Accepted and implemented (Tasks 31 and 32)
- Date: 2026-10-06
- Scope: `modelq quantize --format nvfp4` writing the ModelQ-native NVFP4 SafeTensors convention
- Design: [NVFP4 CLI export design](../superpowers/specs/2026-10-06-nvfp4-cli-export-design.md)

## Context

[ADR 0011](0011-nvfp4-native-safetensors-convention.md) defines the ModelQ-native NVFP4 container and the library had a planner, writer, and reader for it, but the CLI exposed only INT8 and the writer required one mapped file and a whole-tensor `Vec<f32>`. [ADR 0003](0003-sharded-safetensors-input-design.md) input discovery was wired into the CLI for INT8 in Task 30.

## Decision

`modelq quantize <MODEL> --format nvfp4 --output <PATH>` reads a single file, checkpoint directory, or index path and writes one ModelQ-native NVFP4 file. It is `cpu` only. The output makes no Transformer Engine, TensorRT, or other runtime-compatibility claim, and the command says so in its report.

- **Streaming quantizer.** `modelq_quant::nvfp4::quantize_replay_chunks` reads a replayable value source twice (tensor-wide amax, then data) and emits packed bytes in chunks of whole blocks. It shares one block encoder with `quantize`, and tests assert bit-identical packed bytes, block scales, and global scale against `quantize_shaped` for several chunk sizes and edge inputs. Memory is one chunk plus the block scales (1/16 of the element count).
- **Selection policy.** `Nvfp4Policy` quantizes a tensor only if it is floating (F32/F16/BF16), has rank two or more, a non-zero final dimension divisible by 16, at least 1024 elements, and no excluded name part. Everything else is preserved byte-for-byte with a printed reason. Structural reasons are evaluated before name exclusions.
- **Default exclusions.** Names containing `embed_tokens`, `lm_head`, or `embeddings` are preserved. `--exclude <SUBSTRING>` (repeatable) adds more and `--no-default-excludes` removes the built-in list. Both flags are rejected with `--format int8`. This is a heuristic chosen because silently quantizing the embedding or output head is the more harmful mistake; it is visible in the printed policy line and every per-tensor reason.
- **Writer.** `write_nvfp4_safetensors` now takes any `TensorSource` and streams `qdata` directly to the temporary file, then writes the held block scales and global scale. It refuses a destination that matches any source shard, and commits by renaming a completed temporary file, so a quantization error leaves no output.
- **Validation.** After writing, the command reopens the output and, per tensor and in bounded memory, compares preserved bytes with the source and reconstructs each quantized tensor with `dequantize_iter` (which validates lengths, scales, and zero-scale blocks without allocating) to report max MSE, MAE, absolute error, and lowest SQNR. It does not call `read_nvfp4_safetensors` because that returns every decoded tensor as `Vec<f32>`; manifest correctness is covered by the writer and reader tests.

Sharded and single-file inputs holding the same tensors produce byte-identical output (tested).

## Not included

Output sharding, the Transformer Engine rowwise container for many matrices, columnwise data or scale swizzling, GPU quantization, SIMD or parallel NVFP4, 2D weight scaling, random Hadamard transforms, stochastic rounding, calibration, and activation quantization. INT8 behavior is unchanged.

## Consequences

- Real multi-file checkpoints can be converted to the native NVFP4 container without holding a whole tensor in `f32`.
- A failed validation exits non-zero and names the output, but, as with INT8, does not delete the committed file.
- The policy's name exclusions are a convention, not a model-architecture analysis; models with other embedding or head names need `--exclude`.
- Runtime use still requires a separate, hardware-validated export step.
