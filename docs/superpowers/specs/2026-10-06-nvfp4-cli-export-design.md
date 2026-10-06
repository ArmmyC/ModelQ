# NVFP4 CLI Export over Sharded Input

- Status: approved (Decision 2 option A chosen 2026-10-06); implemented in Tasks 31 and 32 (see ADR 0014)
- Date: 2026-10-06
- Scope: a `modelq quantize --format nvfp4` command that reads a single or sharded SafeTensors checkpoint and writes one ModelQ-native NVFP4 SafeTensors file
- Related decisions: [ADR 0003](../../adr/0003-sharded-safetensors-input-design.md), [ADR 0010](../../adr/0010-nvfp4-research-spike.md), [ADR 0011](../../adr/0011-nvfp4-native-safetensors-convention.md), [ADR 0012](../../adr/0012-transformer-engine-nvfp4-export-profile.md), [ADR 0013](../../adr/0013-transformer-engine-nvfp4-runtime-container.md)

## Goal

Make the existing, tested NVFP4 library path usable on real checkpoints:

```text
single file | checkpoint directory | *.safetensors.index.json
                     |
                     v
        SafetensorsInput (Task 29/30)
                     |
                     v
   NVFP4 selection policy  -->  per-tensor decision + reason
                     |
                     v
   streaming NVFP4 quantizer (bounded memory)
                     |
                     v
   ModelQ-native NVFP4 SafeTensors (ADR 0011), reopened and validated
```

The output is the ModelQ-native container. It is not a Transformer Engine, TensorRT, or llama.cpp checkpoint, and the command makes no runtime-compatibility claim.

## Context

What exists today:

- `modelq_quant::nvfp4::quantize` / `quantize_shaped` take a whole `&[f32]`, so a caller must hold the full tensor as `f32` plus an unpacked intermediate. This conflicts with the bounded-memory rule that INT8 already follows.
- `modelq_io::nvfp4` has a planner, writer, and reader for ADR 0011, but the writer takes `&MappedSafetensors` (single file only) and the CLI does not call it.
- The CLI's INT8 path now runs over `TensorSource` (single file or sharded) with a name-ordered catalog.
- The only NVFP4 runtime proof is one 64x64 matrix on TE 2.19.0 / B200 (ADR 0013). That is a separate artifact schema and is not extended here.

NVFP4 needs a tensor-wide amax before any block can be encoded, because every block scale is expressed relative to it. A one-pass design would require holding the whole tensor. Two passes over the mapped source avoid that.

## Decisions

### 1. Target container: ModelQ-native (ADR 0011), not the TE profile

`--format nvfp4` writes ADR 0011. The Transformer Engine rowwise container stays a single-matrix proof until a multi-matrix TE schema is designed on its own (it needs a manifest for many tensors, a decision on columnwise data, and a hardware re-validation). Producing it from the native file later is a pure mapping step (ADR 0012), so nothing is lost by deferring.

### 2. Selection policy

A tensor is quantized only if all of these hold; otherwise it is preserved byte-for-byte with an explicit reason:

1. floating dtype (F32, F16, BF16);
2. rank >= 2 (vectors such as norms and biases are preserved);
3. final dimension is non-zero and divisible by 16 (the native block rule), and no dimension is zero;
4. element count >= the existing default minimum (`QuantizationPolicy`);
5. the name does not match an exclusion.

Exclusions: by default, names containing `embed_tokens`, `lm_head`, or `embeddings` are preserved, since weight-only low-bit schemes conventionally keep the vocabulary projection and embedding at higher precision. `--exclude <substring>` (repeatable) adds more; `--no-default-excludes` removes the built-in list. The built-in list and every decision reason are printed, so the choice is auditable and never silent.

The policy lives beside the INT8 policy in `modelq-quant` as a separate type (`Nvfp4Policy`) with its own `DecisionReason`-style enum. INT8 behavior does not change.

*Reviewed and accepted (option A):* name-substring exclusion is a heuristic. The alternative is shape/dtype rules only, with no default name exclusions. I recommend the defaults above because quantizing `lm_head`/embeddings silently is the more harmful failure, and the flag removes them.

### 3. Streaming quantizer

Add `modelq_quant::nvfp4::quantize_streaming` that takes a replayable iterator source (as the INT8 replay path does) and a shape, performs pass 1 for the global amax, then pass 2 to emit packed bytes and block scales in bounded chunks aligned to whole 16-value blocks. Memory is `O(chunk)` plus the block-scale vector (one byte per 16 values, 1/16 of the element count, which is small enough to hold and is needed before the writer places it).

Requirement: for any input, the streaming result is bit-identical to `quantize_shaped` (packed bytes, block scales, global scale). Tests assert this across random and edge inputs (all zeros, single huge outlier, subnormal-range values, chunk boundaries that split a block boundary, lengths that are exact multiples and not).

The existing whole-slice function stays as the reference; the streaming one is validated against it, as INT8 does.

### 4. Writer over `TensorSource`

Generalize `write_nvfp4_safetensors` and `plan_nvfp4_output`'s caller path to `&impl TensorSource`, the same refactor already applied to the INT8 writer. The writer refuses any destination that matches any source shard, writes to a temporary file, and renames after sync. Default Cargo tests remain CPU-only.

Data section order follows the ADR 0011 plan (deterministic by source name). Because block scales and the global scale are produced while the quantized data is streamed, the writer streams `qdata` in plan order and holds each tensor's block scales until their planned position. This is the only non-constant buffer and is 1/16 of that tensor's element count.

### 5. CLI behavior

```text
modelq quantize <MODEL> --format nvfp4 --output <PATH>
                [--device cpu] [--exclude <SUBSTRING>]... [--no-default-excludes]
```

- `<MODEL>`: single file, checkpoint directory, or index path (same discovery as INT8).
- Output: one SafeTensors file; the destination must not exist and must not be one of the source shards.
- `--device` accepts only `cpu`.
- `--exclude` and `--no-default-excludes` apply only to `nvfp4`; passing them with `int8` is an error.
- Progress lines list every tensor with quantize/preserve and the reason, as the INT8 command does.
- After writing, the command reopens the output with `read_nvfp4_safetensors`, dequantizes every quantized tensor, and reports max MSE, max MAE, max absolute error, lowest SQNR, and the size change. A failed reopen or any non-finite result exits non-zero and names the output path; as with INT8, the already-committed file is not deleted.
- The final report states: "ModelQ-native NVFP4 output; no runtime compatibility is implied."

### 6. Errors and failure safety

All source validation (index, shard headers, name checks) happens in `SafetensorsInput::open` before any output is created. A quantization error (non-finite input) names the tensor and element index and leaves no output, because the writer commits only by renaming a completed temporary file.

## Out of scope

- Output sharding or an output index.
- Multi-matrix Transformer Engine export, columnwise data, scale swizzle, or any new hardware claim.
- GPU/CUDA quantization, SIMD, parallel dispatch for NVFP4 (the INT8 parallel path remains separate).
- 2D (16x16) weight scaling, random Hadamard transforms, stochastic rounding, calibration, activation quantization.
- Changing the INT8 command's behavior or output.

## Testing

1. Quantizer: streaming equals `quantize_shaped` bit-for-bit (property-style cases above), plus error cases (NaN/inf at a known index; shape not divisible by 16).
2. Policy: table-driven tests for each rule and reason, including default and custom exclusions.
3. Writer: sharded source and single-file source produce identical output bytes for the same tensors; destination-conflict and existing-destination refusal; determinism across repeated runs.
4. CLI (single and two-shard fixtures): output reopens through the NVFP4 reader, names/shapes match the plan, preserved tensors are byte-identical, excluded and ineligible tensors are preserved with the expected reason, `--exclude` with `int8` is rejected.
5. A memory-shape check: a test with a tensor larger than the streaming chunk proves quantization completes without constructing a whole-tensor `Vec<f32>` (asserted by construction through the replay API, not by measuring RSS).

Linux (Rust 1.85) runs are the authoritative check; native Windows test execution can be blocked by Smart App Control and is reported separately.

## Task breakdown

- **Task 31:** streaming NVFP4 quantizer in `modelq-quant`, with equivalence tests. No I/O or CLI changes.
- **Task 32:** `Nvfp4Policy`, generalize the NVFP4 writer to `TensorSource`, and add the CLI path, report, and documentation (ADR 0014).

Each task is independently reviewable and leaves the CLI working.

## Alternatives considered

- **Emit the TE container directly from the CLI:** rejected; its schema is single-matrix and hardware-validated only for that case.
- **Quantize with `quantize_shaped` and accept whole-tensor memory:** rejected; it makes the first real-checkpoint run fail on large embeddings or MLP weights and contradicts the bounded-memory goal.
- **Shape-only policy with no name exclusions:** viable and simpler; see the open question in Decision 2.
- **A per-tensor global scale computed from a sample:** rejected; it would change the numeric result versus the reference quantizer and break bit-identity.
