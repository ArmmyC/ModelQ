# ADR 0021: Transformer Engine Multi-Matrix Container (Schema v2)

- Status: Accepted and implemented (Task 40)
- Date: 2026-10-07
- Scope: `modelq quantize --format nvfp4-te`, a container of many rank-two rowwise-NVFP4 matrices for Transformer Engine 2.19.0
- Hardware compatibility: **validated in Task 41 for the tested shapes; see [ADR 0022](0022-transformer-engine-multi-matrix-hardware-validation.md).** This ADR describes the CPU-side implementation, which was CPU-validated only when written; the hardware run then added one eligibility rule (final dimension divisible by 32).
- Design: [multi-matrix export design](../superpowers/specs/2026-10-07-transformer-engine-multi-matrix-export-design.md); builds on [ADR 0012](0012-transformer-engine-nvfp4-export-profile.md), [ADR 0013](0013-transformer-engine-nvfp4-runtime-container.md), [ADR 0014](0014-nvfp4-cli-export.md), [ADR 0015](0015-sharded-output.md)

## Context

[ADR 0013](0013-transformer-engine-nvfp4-runtime-container.md) established a single 64x64 matrix artifact whose rowwise 1x16 fields passed one TN GEMM on an NVIDIA B200. That is a proof of one matrix, not something usable on a checkpoint. This ADR records the CPU side of generalizing it, following the design chosen on 2026-10-07: keep the validated 1x16 rowwise convention, and prove it later per matrix on hardware.

## Decision

**Container (schema v2).** A standard SafeTensors file or shard with `modelq.format = transformer-engine-nvfp4-safetensors-v2`. For each exported source tensor `W` of logical shape `[M, K]` it holds `W.rowwise_data` (`U8`, `[M, K/2]`), `W.rowwise_scale_inv` (`U8`, `[round_up(M,128), round_up(K/16,4)]`, zero padded, not swizzled) and `W.amax_rowwise` (`F32`, `[1]`), exactly as in ADR 0013. All other source tensors are preserved byte for byte. Within a file, sources appear in ascending name order and a matrix's fields in the order data, scales, amax, so the writer can stream. `modelq.manifest` is deterministic JSON covering exactly that file's tensors, with the fixed profile id, the TE 2.19.0 pin, the quantization block, scale storage (padding `[128, 4]`, not swizzled), the global-scale denominator 2688, and per-tensor entries (`quantized` with logical shape, original dtype and field names, or `preserved` with dtype and shape). Single-matrix v1 files stay valid; v2 is a separate schema.

**Eligibility.** The NVFP4 policy plus Transformer Engine rules, enabled by `Nvfp4Policy::transformer_engine()`: exactly rank two, a leading dimension divisible by 16, and (added after the Task 41 hardware run, ADR 0022) a final dimension divisible by 32. The second matches `NVFP4Quantizer.is_quantizable` in the pinned v2.19 source, which requires both the last dimension and the product of the leading dimensions to be divisible by 16. Rank three and above (for example stacked expert weights) are preserved because flattening them would share one amax across slices. Every decision prints its reason. Default name exclusions (`embed_tokens`, `lm_head`, `embeddings`) are unchanged.

**Writer.** `plan_te_output` lays out every field's shape and offset from source shapes; `write_te_nvfp4_safetensors_with` streams the packed bytes through the same bounded sequential or parallel quantizer as the native writer (the streaming step is now one shared function), writes the padded scale matrix row by row from the held block scales without materializing it, then the amax (`global_scale * 2688`, zero for an all-zero tensor, as in ADR 0013). It refuses to overwrite a source or an existing destination and commits by renaming a completed temporary file, so a failure leaves nothing behind.

**Validation.** `read_te_container_manifest` checks the manifest against the file's header (every listed tensor and field present with the exact dtype and shape, no unlisted tensor, fixed schema values). `te_matrix_values` validates a matrix's three fields (lengths, zero padding in rows and columns, positive finite E4M3 scales, finite non-negative amax, zero-scale blocks holding only zero values) and decodes it lazily as `e2m1 * e4m3_scale * (amax / 2688)`, which is how the runtime recovers the scale. The CLI reopens the output and, per matrix and in bounded memory, reports the same error metrics as the native path.

**CLI.** `--format nvfp4-te` with the existing `--exclude`, `--no-default-excludes`, `--threads` and `--max-shard-size`. A sharded output keeps a matrix's three fields in one shard and gives every shard its own manifest. When written, the final report said "CPU-validated only: multi-matrix Transformer Engine compatibility is not yet hardware-validated"; after Task 41 it states the validated scope instead (ADR 0022).

**Fixture and Python check.** `cargo run -p modelq-io --example te_multi_matrix_fixture -- <dir>` writes a synthetic checkpoint (matrices from `[32,32]` to `[1024,4096]` including `[48,96]` and `[144,80]` whose scale padding is not trivial, plus preserved tensors for every ineligible class) and an F32 reference of every exported matrix. `tools/transformer_engine_nvfp4/validate_multi.py cpu` is an independent numpy implementation that validates a file or sharded directory and decodes each matrix against that reference, importing neither Transformer Engine nor PyTorch. Its `runtime` mode was a placeholder that exited non-zero until Task 41 implemented it.

## Evidence

- The container's three fields for a `[64, 64]` matrix equal, byte for byte, the output of the single-matrix profile function that produced the hardware-validated ADR 0013 artifact (test `fields_equal_the_hardware_validated_single_matrix_profile`).
- Packed bytes and logical scales equal the native NVFP4 quantizer's for every exported matrix, including partially padded shapes; the amax equals `global_scale * 2688` bit for bit.
- Output is byte-identical for sequential and parallel execution across worker counts and chunk sizes, for single-file and sharded inputs, and a sharded output holds the same tensors as the single-file output.
- The independent Python validator accepted the Rust-produced single-file and sharded containers of the synthetic fixture and decoded every matrix with a maximum absolute difference of 0.0 against the reference. The Python tests cover the corruption classes (padding, scales, amax, shapes, manifest values, shard indexes) and found and fixed one bug in the validator itself (a matrix repeated across shards was overwritten silently).

## What is not claimed

*(As written for the CPU-only increment; superseded in scope by ADR 0022, which validated it on a B200 for the tested shapes.)* No Transformer Engine, CUDA, or hardware compatibility of this schema. Transformer Engine may impose requirements this design does not know (for example further alignment of operands or outputs) and shapes beyond those tested stay unverified; the run in Task 41 exists to find that. Also out of scope: columnwise data, TE's 2D 16x16 weight convention (this container uses 1x16, so its numerics are not what TE would produce for weights itself), grouped or stacked weights, writing swizzled scales, `te.Linear` integration, training, TE versions other than 2.19.0, and any inference capability.

## Consequences

- A real checkpoint can be converted to a container the existing bridge's constructor can load per matrix, with bounded memory and sharding.
- A matrix with a small leading dimension still stores 128 scale rows; this is the TE layout and is accepted.
- Task 41 runs the pinned bridge on a Blackwell host per matrix; only a passing run changes the compatibility statement, and only for what it tested.
