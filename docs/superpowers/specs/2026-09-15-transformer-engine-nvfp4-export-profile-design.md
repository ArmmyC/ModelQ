# Transformer Engine NVFP4 Rowwise Export Profile

- Status: Proposed
- Date: 2026-09-15
- Scope: the first CPU-only runtime-profile export increment for NVFP4

## Goal

Define one explicit Transformer Engine NVFP4 profile that can be produced from
ModelQ's existing native scalar representation without adding CUDA, PyTorch, or
Transformer Engine as Rust dependencies.  The implementation will make the
runtime-facing shapes, names, padding, and scale direction inspectable and
testable.  It will not claim that a generated artifact can already be loaded by
Transformer Engine on Blackwell hardware.

## Context

ADR 0010 defines the ModelQ-native numerical reference: packed E2M1 values,
one E4M3 block scale for each 16 values, and one F32 decode scale.  ADR 0011
defines the corresponding ModelQ-native SafeTensors convention and explicitly
keeps runtime layout separate.  Task 26 added an optional Transformer Engine
reference-fixture harness with a deterministic rowwise 1x16 recipe, but it does
not produce a runtime export.

The current Transformer Engine NVFP4 documentation describes a rowwise 1D
layout with packed data shaped `[M, K / 2]`, E4M3 scale bytes shaped
`[round_up(M, 128), round_up(K / 16, 4)]`, and tensor-scaling amax metadata.
The public PyTorch API calls these buffers `rowwise_data`,
`rowwise_scale_inv`, and `amax_rowwise`.  Columnwise data, GEMM-swizzled scale
buffers, 2D weight scaling, RHT, stochastic rounding, and 4over6 are separate
contracts and are outside this profile.

## Options considered

### A. CPU-only Transformer Engine rowwise profile planner (selected)

Adapt a validated `modelq_quant::nvfp4::QuantizedTensor` plus its source shape
into an in-memory profile object.  The planner copies the packed rowwise data,
expands the native block-scale stream into Transformer Engine's padded scale
matrix, derives the tensor amax from ModelQ's decode scale, and exposes the
runtime field names and shapes.  This is small enough to test on every CI
platform and leaves the future file writer or GPU bridge independent.

### B. Write a Transformer Engine-specific SafeTensors container now

This would add a complete container writer and a naming/metadata convention in
the same task.  Transformer Engine's documented tensor object is an in-memory
CUDA object rather than a portable checkpoint container, so choosing a file
envelope before a real load test would risk creating a plausible but unusable
artifact.  The file writer remains a follow-up after the profile is validated.

### C. Target TensorRT instead

TensorRT's explicit NVFP4 interface uses a different per-block scale contract
and a Q/DQ graph/container boundary.  It is a valid future exporter, but it
would not reuse the Transformer Engine fixture work and would mix two runtime
contracts into this increment.

## Decision

Implement option A as profile
`transformer-engine.nvfp4.rowwise.1x16.v1` in
`crates/modelq-io/src/transformer_engine.rs`.

The public entry point will be:

```rust
pub fn export_transformer_engine_nvfp4(
    name: &str,
    shape: &[usize],
    quantized: &modelq_quant::nvfp4::QuantizedTensor,
) -> Result<TransformerEngineNvfp4Tensor, TransformerEngineNvfp4Error>
```

The returned value is a validated, owned CPU representation of the three
runtime-facing rowwise fields:

```text
<name>.rowwise_data       : U8, shape [d0, ..., d_last / 2]
<name>.rowwise_scale_inv  : U8, shape [round_up(M, 128), round_up(K / 16, 4)]
<name>.amax_rowwise       : F32, shape [1]
```

where `K = shape.last()` and `M = product(shape[..shape.len()-1])`.  The
logical element count is `M * K`.

### Data mapping

- `rowwise_data` is copied byte-for-byte from ModelQ's packed payload.  Each
  byte keeps element zero in the low nibble and element one in the high nibble.
- The first `M` rows and first `K / 16` columns of `rowwise_scale_inv` are the
  native E4M3 block-scale bytes in row-major order.  Additional aligned rows
  and columns are zero-filled padding.  No scale swizzle is applied.
- `amax_rowwise` is one F32 tensor-wide amax.  For a nonzero tensor it is
  computed as `global_decode_scale * (448.0 * 6.0)` in F32 arithmetic.  An
  all-zero tensor exports `0.0` amax while retaining ModelQ's safe native
  all-zero payload and scale convention.
- The E4M3 bytes remain the local decode scales.  A consumer reconstructs the
  global factor as `amax_rowwise / (448.0 * 6.0)` for a nonzero tensor; the
  profile does not silently rename a decode scale as an encode multiplier.

### Validation and errors

The exporter rejects:

- an empty name or the reserved SafeTensors metadata name;
- a rank-one-or-lower shape, an empty shape, a zero dimension, or a final
  dimension not divisible by 16;
- shape products that overflow `usize`;
- a quantized payload length or block-scale count that disagrees with the
  shape;
- an invalid native global scale or a non-finite derived amax; and
- any native payload that fails the existing NVFP4 validator.

The profile accepts arbitrary positive `M` and pads the scale matrix to the
documented Transformer Engine alignment.  It does not silently add data rows
or change the logical shape.  A later hardware compatibility test may impose
additional model-shape restrictions if a pinned runtime requires them.

### Compatibility boundary

The result is **profile-valid CPU data**, not a Level 3 runtime-compatible
checkpoint and not a Level 4 hardware-validated artifact.  This increment does
not include a SafeTensors writer, CUDA allocation, Transformer Engine object
construction, GEMM scale swizzling, columnwise transpose, or a Blackwell test.
Those steps require a pinned runtime/container decision and an actual fixture
load path.

## Module and API boundaries

- `modelq-quant::nvfp4` remains the numerical oracle and is not changed to
  know about Transformer Engine names or padding.
- `modelq-io::transformer_engine` owns the runtime-specific shape mapping,
  padded scale allocation, field naming, and validation errors.
- `modelq-io/src/lib.rs` exposes the module; no CLI flag or default command is
  added.
- No new external dependency is needed.

## Testing

Unit and integration coverage will verify:

1. a `[2, 32]` synthetic tensor maps to data shape `[2, 16]` and padded scale
   shape `[128, 4]`;
2. unpadded data and scale bytes are preserved exactly, while every padding
   byte is zero;
3. the derived amax reproduces the ModelQ global-scale relationship and the
   all-zero case exports zero amax;
4. generated names and profile metadata are deterministic;
5. malformed names, shapes, lengths, scales, and overflow cases are rejected;
6. repeated exports are byte-for-byte/equality deterministic; and
7. the existing workspace tests, formatting, and clippy checks remain green.

The optional Transformer Engine fixture test remains the external numerical
oracle.  No fixture is added or marked as runtime-compatible by this task.

## Follow-up boundary

After this profile is reviewed and implemented, the next separate design must
choose the serialized container and demonstrate construction or loading by a
pinned Transformer Engine release on Blackwell before the compatibility level
can be raised.
