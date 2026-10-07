# Transformer Engine Multi-Matrix Export

- Status: approved (decisions 1 and 2 chosen 2026-10-07); CPU side implemented in Task 40 (ADR 0021); hardware proof run in Task 41 (ADR 0022), which added the final-dimension-divisible-by-32 rule
- Date: 2026-10-07
- Scope: `modelq quantize --format nvfp4-te`, a SafeTensors container holding many rowwise-NVFP4 matrices for Transformer Engine 2.19.0, plus the validation plan for a later hardware run
- Related decisions: [ADR 0011](../../adr/0011-nvfp4-native-safetensors-convention.md), [ADR 0012](../../adr/0012-transformer-engine-nvfp4-export-profile.md), [ADR 0013](../../adr/0013-transformer-engine-nvfp4-runtime-container.md), [ADR 0014](../../adr/0014-nvfp4-cli-export.md), [ADR 0015](../../adr/0015-sharded-output.md), [Task 28 design](2026-10-03-transformer-engine-nvfp4-runtime-container-design.md)

## Goal

Turn the single-matrix, hardware-validated artifact of Task 28 into something usable on a real checkpoint:

```text
single | sharded SafeTensors checkpoint
              |
              v
  TE eligibility policy (rank-2, both dims %16, plus the NVFP4 policy)
              |
              v
  streaming rowwise NVFP4 quantizer (bounded memory, parallel)
              |
              v
  TE container v2: per matrix  rowwise_data / rowwise_scale_inv / amax_rowwise
                   preserved tensors verbatim; manifest per file; optional shards + index
              |
              v
  Python bridge (pinned TE 2.19.0) builds one NVFP4Tensor per matrix
```

It is still a quantization compiler's output, not an inference engine: no model code, no activation quantization, no whole-model forward pass.

## Context and findings

- The Task 28 artifact is one matrix per file, 64x64, with hardware validation on one B200 for one TN GEMM only. Everything beyond that remains unverified.
- The Rust side already has what a multi-matrix writer needs: the profile mapper (`export_transformer_engine_nvfp4`), the bounded parallel quantizer, the NVFP4 policy, sharded input and output, and per-shard manifests. What it lacks is a streaming writer for the TE field layout and a multi-tensor manifest. The existing profile function takes a whole `QuantizedTensor` in memory and is not suitable for large checkpoints.
- **TE's shape requirement is stricter than ours.** In the pinned v2.19 `NVFP4Quantizer.is_quantizable`, a tensor qualifies only if its last dimension *and* the product of all leading dimensions are divisible by 16 ([source](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/pytorch/tensor/nvfp4_tensor.py)). The native NVFP4 policy only checks the last dimension, so a TE export needs an extra rule.
- **TE's own weight convention differs from ours.** TE defaults to 2D 16x16 block scaling for weights so rowwise and columnwise copies are numerically equal ([NVFP4 guide](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)). Our profile uses 1x16 rowwise scales, which is what Task 28 validated for a forward TN GEMM. NVFP4 GEMM supports only the TN layout.
- The scale tensor is `(round_up(M, 128), round_up(ceil(K/16), 4))`, where `M` is the product of the leading dimensions; the container already stores it zero-padded and un-swizzled, and the bridge tells TE so.

## Decisions

### 1. Scaling convention: keep 1x16 rowwise (chosen)

The container carries only the rowwise representation validated in Task 28, for forward-pass weights. It stores no columnwise data and makes no claim about training or backward passes. TE's 2D 16x16 weight convention would need a new quantizer, columnwise payloads, and a fresh hardware validation of different numerics; it is a possible later profile (`...rowwise.2x16x16...`), not part of this work. The practical trade-off is documented: 1x16 uses more scale bytes but finer scaling than TE's own 2D weights.

### 2. Hardware proof scope: per-matrix load and GEMM (chosen)

Task 41 validates, on a Blackwell host, every matrix of a synthetic multi-matrix checkpoint with varied shapes. For each: load into TE, compare TE dequantization with the CPU reference, and run one TN GEMM against the oracle. It does not wire weights into `te.Linear` or run a model forward pass.

### 3. Container: v2, one file per shard, many matrices per file

Standard SafeTensors, `modelq.format = transformer-engine-nvfp4-safetensors-v2`. Task 28's v1 single-matrix files stay valid and readable; v2 is the multi-matrix generalization. For every quantized source tensor `W` (logical `[M, K]`):

| Tensor | Dtype | Shape |
| --- | --- | --- |
| `W.rowwise_data` | `U8` | `[M, K/2]` |
| `W.rowwise_scale_inv` | `U8` | `[round_up(M,128), round_up(K/16,4)]`, zero padded, not swizzled |
| `W.amax_rowwise` | `F32` | `[1]` |

Packed E2M1 bytes are exactly the native NVFP4 bytes; scales and amax follow ADR 0013 (`amax = global_decode_scale * 2688`, zero for all-zero tensors, and the `rowwise_scale_inv` bytes are E4M3 local decode scales despite the TE name). Preserved tensors keep their names and bytes. The per-matrix field order inside the file is `rowwise_data`, `rowwise_scale_inv`, `amax_rowwise`, with matrices and preserved tensors interleaved in ascending source-name order, so the writer can stream data first and place scales and amax after (the single-file v1 writer sorts all tensor names, which would put `amax_rowwise` first).

`modelq.manifest` is deterministic JSON, one per file, covering exactly that file's tensors, so a shard is self-contained:

```json
{
  "schema_version": 2,
  "profile_id": "transformer-engine.nvfp4.rowwise.1x16.v1",
  "runtime": { "name": "transformer_engine", "version": "2.19.0" },
  "quantization": { "data_format": "E2M1", "block_scale_format": "E4M3", "block_size": 16,
                    "scaling": "rowwise_1x16_tensor_global" },
  "scale_storage": { "padding": [128, 4], "gemm_swizzled": false },
  "global_scale_denominator": 2688.0,
  "tensors": {
    "layers.0.weight": { "action": "quantized", "logical_shape": [144, 80],
      "original_dtype": "BF16",
      "fields": { "rowwise_data": "...", "rowwise_scale_inv": "...", "amax_rowwise": "..." } },
    "norm.weight": { "action": "preserved", "dtype": "F32", "shape": [4096] }
  }
}
```

The profile id, runtime pin, quantization block, and scale storage are fixed by schema version 2. A sharded output reuses ADR 0015: `model-NNNNN-of-MMMMM.safetensors` plus `model.safetensors.index.json` over the output field names, with a quantized matrix's three fields always in one shard.

### 4. Eligibility: the NVFP4 policy plus two TE rules

A tensor is exported as a TE matrix only if it passes the existing NVFP4 policy (floating dtype, minimum size, name exclusions, last dimension divisible by 16) **and**:

1. **rank exactly 2.** A rank-3 or higher tensor (for example a stacked mixture-of-experts weight) would flatten its leading dimensions into `M` and share one amax across experts, which differs from per-expert quantization; such tensors are preserved with a printed reason. Supporting grouped weights is a separate design.
2. **`M` divisible by 16**, matching TE's `is_quantizable`.

`Nvfp4Policy` gains a Transformer Engine mode adding these rules and two new decision reasons; INT8 and the native NVFP4 path keep their behavior. Every decision is printed with its reason, as today. Default name exclusions (`embed_tokens`, `lm_head`, `embeddings`) apply unchanged.

### 5. Writer and validation (Rust)

- A planner (`plan_te_nvfp4_output`) computes every field's shape and offset from source shapes, like the native planner, so the data section is laid out before any quantization.
- A streaming writer takes any `TensorSource` (single or sharded), streams `rowwise_data` from the bounded parallel quantizer (same quantizer, same bytes as native), then writes `rowwise_scale_inv` row by row from the held block scales with zero padding (never materializing the padded matrix) and `amax_rowwise`. It supports `Nvfp4Execution::{Sequential, Parallel}`, atomic temporary-file commit, and refuses to overwrite a source shard, like the native writer.
- A CPU validator reopens the output and, per matrix and in bounded memory, un-pads the scales, reconstructs values with the shared decode (`dequantize_iter`), and reports the same error metrics as the native path. It also checks that padding is zero and that the manifest matches the fields. Test-only checks assert that the data bytes and logical scales equal the native NVFP4 result for the same source, and that outputs are byte-identical across thread counts and between single-file and sharded inputs.

### 6. CLI

`modelq quantize <MODEL> --format nvfp4-te --output <PATH>` with the existing `--exclude`, `--no-default-excludes`, `--threads`, and `--max-shard-size`. The final report states the profile, the TE pin, and the validation status: **"CPU-validated container; multi-matrix Transformer Engine compatibility is not yet hardware-validated"** until Task 41 passes, and the ADR and README repeat it. No Level 3 or Level 4 claim for v2 is made before then.

### 7. Python bridge and hardware proof (Task 41)

The existing `tools/transformer_engine_nvfp4/` tool gains a v2 mode:

- **CPU container check** (no TE, no CUDA): manifest and field shapes, dtypes, zero padding, amax finiteness, and the reference file, for a single file or a directory with an index.
- **Runtime mode** (explicit, Linux, pinned environment): for each quantized matrix, build the TE 2.19.0 `NVFP4Tensor` exactly as in Task 28 (rowwise buffers only, columnwise `None`, `with_gemm_swizzled_scales=False`), compare TE dequantization with the CPU reference (`rtol=1e-5`, `atol=1e-5`), then run `general_gemm(weight, second_operand, layout="TN")` with a deterministic `[64, K]` second operand and compare with `TE-dequantized second operand @ ModelQ-dequantized weight.T` (`rtol=0.125`, `atol=0.0675`), requiring finite output of shape `[64, M]`. Any failure exits non-zero; an unavailable GPU is not a pass.
- **Synthetic fixture** (Rust example, never checked in as a binary): matrices chosen to exercise padding and eligibility, for example `[16,16]` (smallest), `[64,64]` (the Task 28 case), `[48,96]` and `[144,80]` (`M` not a multiple of 128, `K/16` not a multiple of 4), `[256,512]`, `[1024,4096]` (several chunks); plus preserved tensors (a norm vector, an integer tensor, an `lm_head` excluded by name, a rank-3 tensor, and an `[70,64]` matrix with `M` not divisible by 16), in single-file and sharded forms. A separate F32 reference file holds the CPU dequantization of every matrix; it is test-only and never part of the runtime artifact.
- The run records the exact environment (image, Python, PyTorch, CUDA, cuDNN, driver, TE, GPU) as in ADR 0013, and the cuDNN library-path caveat found then applies.

## Out of scope

- Columnwise data, 2D 16x16 weight scaling, and training or backward use.
- Grouped or stacked weights (rank 3 and above), tied weights, and fused QKV handling.
- Writing GEMM-swizzled scales; TE swizzles at runtime as validated in Task 28.
- Model code, `te.Linear` integration, activation quantization, and any inference claim.
- TE versions other than 2.19.0, non-Blackwell hardware, and non-TE runtimes.
- Changing the native NVFP4 or INT8 outputs or CLI behavior.

## Task breakdown

- **Task 40 (CPU only):** TE policy mode and reasons; field planner; streaming v2 writer with sharded output; CPU reader and validator; `--format nvfp4-te`; the fixture generator; Python v2 CPU container check; ADR 0021. Verified on Linux CI; makes no hardware claim.
- **Task 41 (needs a Blackwell host):** run the Python runtime mode on the synthetic fixture, fix any incompatibility it exposes, and record the environment and results. Only a passing run changes the compatibility statement, and only for the schema, TE version, shapes, and operation actually tested.

## Risks and open items

- The "`M` divisible by 16" rule is taken from the pinned v2.19 source as read for this design; the implementation must cite the exact lines and re-check them, and the hardware run is the real test.
- TE GEMM may impose further alignment (for example on the second operand or output) that this design does not know; Task 41 exists to find that, and shapes beyond those tested stay unclaimed.
- Scale padding adds bytes for small `M` (for example `M = 16` stores 128 scale rows). This is the TE layout and is accepted.
- 1x16 scaling is not what TE would produce for weights itself. Numerical equivalence with TE-quantized weights is not claimed.

## Alternatives considered

- **One file per matrix (repeat v1):** rejected; a real model has hundreds of matrices and the loader would need an external index.
- **Convert the native NVFP4 file to TE afterwards:** workable, but doubles the write and read, and the field layout differs enough that a direct streaming writer is clearer.
- **Reuse the in-memory profile function:** rejected for real checkpoints, since it needs the whole quantized tensor in memory.
- **2D 16x16 weights now:** declined in decision 1.
- **Quantizing rank-3 tensors with per-slice amax:** needs a different field layout and TE grouped-GEMM semantics; deferred.
