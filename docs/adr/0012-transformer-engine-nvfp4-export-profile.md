# ADR 0012: Transformer Engine NVFP4 Rowwise Export Profile

- Status: Accepted for the profile-planner increment
- Date: 2026-09-15
- Scope: CPU-side mapping of ModelQ-native NVFP4 data to Transformer Engine's rowwise 1x16 field contract

## Context

ADR 0010 defines ModelQ's scalar NVFP4 reference representation: packed E2M1
values, one positive E4M3 scale for every 16 values, and one F32 tensor-wide
decode scale.  ADR 0011 maps that representation to a ModelQ-native
SafeTensors convention and explicitly makes no runtime-compatibility claim.
Task 26 adds an optional numerical fixture harness for a deterministic
Transformer Engine reference quantizer, but it does not produce a runtime
buffer or checkpoint.

Transformer Engine's current NVFP4 documentation describes a rowwise 1D
layout with packed U8 data, E4M3 scale bytes, and one tensor amax for the
standard tensor-scaling mode.  Its rowwise scale matrix is padded to 128 rows
and four scale columns.  Columnwise data, GEMM-swizzled scales, 2D weight
scaling, random Hadamard transforms, stochastic rounding, and 4over6 are
additional contracts.  The primary references consulted for this decision
are the [Transformer Engine NVFP4 guide](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html),
the [Transformer Engine PyTorch API](https://nvidia.github.io/TransformerEngine/api/pytorch.html),
and the [Transformer Engine common C API](https://nvidia.github.io/TransformerEngine/api/common.html).

## Decision

ModelQ adds the profile identifier
`transformer-engine.nvfp4.rowwise.1x16.v1` in
`modelq_io::transformer_engine`.  The profile is a CPU-only planner and
validator.  It adapts an already validated
`modelq_quant::nvfp4::QuantizedTensor` and its logical source shape; it does
not re-quantize floating-point values.

### Public entry point

```rust
pub fn export_transformer_engine_nvfp4(
    name: &str,
    shape: &[usize],
    quantized: &modelq_quant::nvfp4::QuantizedTensor,
) -> Result<TransformerEngineNvfp4Tensor, TransformerEngineNvfp4Error>
```

The result owns three logical Transformer Engine fields and deterministic names:

```text
<name>.rowwise_data       : U8, shape [d0, ..., d_last / 2]
<name>.rowwise_scale_inv  : U8, shape [round_up(M, 128), round_up(K / 16, 4)]
<name>.amax_rowwise       : F32, shape [1]
```

Here `K` is the final source dimension and `M` is the product of all preceding
dimensions.  The logical source shape is retained separately from the padded
scale shape.

### Data and scale mapping

- `rowwise_data` is copied byte-for-byte from ModelQ's packed payload.  The
  low-nibble-first order is preserved; no transpose or nibble repacking occurs.
- The first `M` rows and first `K / 16` columns of `rowwise_scale_inv` contain
  the native E4M3 block-scale bytes in row-major order.  Rows and columns added
  solely for the 128-by-4 alignment are zero-filled.
- For a nonzero tensor, `amax_rowwise` is the F32 product
  `global_decode_scale * (448.0 * 6.0)`.  A consumer recovers the global
  decode factor by dividing by the same denominator.  For an all-zero tensor,
  the profile emits a zero amax and keeps the native zero payload convention.
- The profile calls the scale field `rowwise_scale_inv` to match Transformer
  Engine's field name, while documenting that the copied E4M3 bytes are local
  block decode scales and the tensor-wide factor is represented by amax.

### Validation

The exporter rejects empty or reserved names, rank-one shapes, zero or
non-positive dimensions, final dimensions not divisible by 16, shape-product
overflow, native length/count mismatches, invalid native scales, non-finite
derived amax values, and any payload rejected by the existing NVFP4 validator.
All alignment and output-size arithmetic is checked before allocation.

## Compatibility boundary

The result is **profile-valid CPU data**.  It is not a Level 3
runtime-compatible checkpoint and not a Level 4 hardware-validated artifact.
This increment intentionally does not include:

- a SafeTensors or other serialized container;
- construction of a Transformer Engine CUDA tensor object;
- columnwise data or transpose generation;
- GEMM scale swizzling or pre-swizzled scale metadata;
- activation quantization, RHT, stochastic rounding, or 4over6; or
- a Blackwell runtime/GEMM load test.

The profile version is a ModelQ mapping version, not a claim that every
Transformer Engine release accepts the same serialized artifact.  A future
container exporter must pin a Transformer Engine release, name its container
and tensor metadata contract, and validate a generated fixture on the target
hardware before raising the compatibility level.

## Alternatives considered

### Write a Transformer Engine-specific checkpoint immediately

Rejected for this increment.  Transformer Engine documents an in-memory tensor
object and hardware-facing layouts, but not one portable checkpoint envelope
that ModelQ can safely infer without a real load test.  A planner makes the
field and padding contract testable before choosing a container.

### Treat ModelQ-native SafeTensors as a Transformer Engine checkpoint

Rejected.  ADR 0011 uses native companion names and an unpadded block-scale
stream.  Runtime field names and alignment padding must be explicit rather
than inferred from a mathematically equivalent representation.

### Target TensorRT in the same module

Rejected.  TensorRT's explicit NVFP4 path has a different scale and Q/DQ
container contract.  It deserves a separate named exporter and compatibility
test rather than a mixed-runtime abstraction.

## Consequences

The profile is small, deterministic, and testable on Windows, Linux, and macOS
without GPU libraries.  Existing scalar quantization and native SafeTensors
behavior remain unchanged.  The owned result can be used by a future writer or
CUDA bridge without making those dependencies part of the base build.

The result is intentionally not sufficient to run a Transformer Engine GEMM:
the serialized envelope, optional swizzles, and hardware validation remain
open work with their own design and acceptance criteria.

## Compatibility impact

This ADR adds one public `modelq-io` module and does not change existing file
formats, CLI commands, or dependency versions.  Existing ModelQ-native INT8
and NVFP4 files remain unchanged.  No runtime-compatible fixture is added by
this decision.
