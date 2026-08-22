# ADR 0010: NVFP4 Research Boundary and Reference Plan

- Status: Accepted
- Date: 2026-08-22
- Scope: a data-free, weight-only NVFP4 reference representation and its future validation path

## Context

Task 19 added the scalar FP4 E2M1 and FP8 E4M3 element codecs.  NVFP4 is
larger than either codec alone: it combines the 4-bit values with a local
scale and a tensor-wide scale, and runtimes add packing, transposes, padding,
scale swizzles, and metadata contracts.  A mathematically correct E2M1
payload must not be described as a TensorRT or Transformer Engine checkpoint
without validating those additional details.

The current primary references are NVIDIA's [Transformer Engine 2.18 NVFP4
guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html),
[Transformer Engine common API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html),
[NVFP4 transpose API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/c/transpose.html),
[CUDA FP4 E2M1 documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-math-api/cuda_math_api/struct____nv__fp4__e2m1.html),
[TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html),
the [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html),
the [NVIDIA Model Optimizer recipe guide](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/ptq.md),
and the [CUTLASS Blackwell narrow-precision guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md).

## Decision

Record a small, deterministic ModelQ-native baseline first.  It is a
reference for a future CPU implementation, not a runtime exporter.  The
baseline is deliberately narrower than the complete Transformer Engine
training recipe: it quantizes weights only, uses one-dimensional groups of 16,
uses deterministic nearest-even rounding, and does not apply stochastic
rounding, random Hadamard transforms, or four-over-six scale selection.

### Exact element encoding

One NVFP4 element is a signed E2M1 value: bit 3 is the sign, bits 2..1 are the
two exponent bits, and bit 0 is the explicit mantissa bit.  The format has no
Inf or NaN encoding.  The positive code table is:

| Bits | Value |
| --- | ---: |
| `0x0` | `+0.0` |
| `0x1` | `+0.5` |
| `0x2` | `+1.0` |
| `0x3` | `+1.5` |
| `0x4` | `+2.0` |
| `0x5` | `+3.0` |
| `0x6` | `+4.0` |
| `0x7` | `+6.0` |

Setting bit 3 applies the sign, so `0x8` is negative zero and `0xf` is
`-6.0`.  This is the same E2M1 element boundary already exercised by
`modelq_quant::float::fp4_e2m1`; CUDA documents the no-Inf/NaN encoding and
satfinite nearest conversion semantics.

For the deterministic offline path, scaled values are rounded to the nearest
E2M1 value, with exact halfway cases choosing the even significand.  Values
outside `[-6, 6]` are clipped before the cast.  Input NaN and infinity are
rejected by the future weight-only API rather than silently becoming a model
weight.  Transformer Engine's training recipe may use stochastic rounding for
gradients; that is a separate algorithm and is not part of this data-free
baseline.

### Hierarchical scale semantics

For a source value `x` in a block, the logical NVFP4 reconstruction is:

```text
x_hat = q_e2m1 * s_block_e4m3 * s_global_f32
```

The constants are `fp4_max = 6.0` and `fp8_e4m3_max = 448.0`.  The decode
global scale is:

```text
s_global_f32 = global_amax / (448.0 * 6.0)
```

where `global_amax` is the maximum absolute source value in the tensor.  For
an all-zero tensor, the encode scale is defined as `1.0` and all codes and
block scales are zero.  To make the direction explicit, the corresponding
encode multiplier is `g_encode = 1.0 / s_global_f32` (or `1.0` for the zero
case).

For each block, compute:

```text
block_amax = max(abs(x_i))
s_block_unrounded = (block_amax / 6.0) * g_encode
s_block_e4m3 = round_to_fp8_e4m3(s_block_unrounded)
q_i = round_to_fp4_e2m1((x_i * g_encode) / s_block_e4m3)
```

The stored block scale is the positive finite FP8 E4M3 bit pattern.  A zero
block stores a zero block scale and zero FP4 codes.  The decoder multiplies
the decoded E2M1 value by the decoded E4M3 block scale and the F32 global
decode scale.  All amax and intermediate scale calculations are F32 in the
reference algorithm; the quantized artifacts retain the exact FP8 byte and
F32 scalar.

If a nonzero block's rounded E4M3 scale would underflow to zero, the native
reference clamps it to the smallest positive E4M3 code (`0x01`) before
encoding its values.  This avoids division by zero and makes the underflow
policy explicit; a runtime exporter must verify whether its target uses the
same fallback.

This is the two-level scale hierarchy described by NVIDIA.  The public
Transformer Engine API also exposes inverse/encode scale buffers, so a future
runtime adapter must name whether a field stores a decode scale or its
reciprocal rather than relying on a generic `scale` label.

### Reference block and packing layout

The ModelQ-native baseline treats a weight tensor as a row-major 2D matrix
and groups 16 consecutive values along its last dimension.  The first
implementation will require that dimension to be divisible by 16; padding is
not implicit.  Its logical fields are:

```text
qdata        : two E2M1 values per byte, element 0 in the low nibble
block_scale  : one positive E4M3 byte per 16-value block
global_scale : one F32 decode scale per tensor
shape        : original unpadded tensor shape
```

The field names and container metadata are a future output-convention
decision; this ADR only fixes the reference meaning.  The low-nibble-first
choice makes the native byte stream deterministic and testable, but it is not
a claim about any runtime's tensor-memory swizzle.

NVIDIA's runtime documentation describes additional layouts.  Transformer
Engine uses 16 consecutive values for one-dimensional scaling and can use a
single scale for a 16x16 weight tile.  Its rowwise data is kept separately from
columnwise data; columnwise data is transposed, scale tensors are padded for
hardware alignment, and scale values are swizzled before GEMM.  NVIDIA's
transpose API also explicitly requires nibble repacking for packed NVFP4 data.
Those details belong to a named runtime profile, not to the ModelQ-native
reference stream.

### Data-free weight-only conversion algorithm

The future scalar implementation will process one floating-point weight
tensor at a time without a calibration forward pass:

1. Convert the source F32/F16/BF16 values to the F32 reference domain and
   reject non-finite values.
2. Validate the matrix shape and the last-dimension multiple of 16.
3. Scan once for `global_amax` and each 16-value `block_amax`.
4. Compute the global decode/encode scale pair above.
5. For each nonzero block, encode its scale to FP8 E4M3 (using the explicit
   `0x01` underflow fallback above), then encode each scaled value to E2M1
   with nearest-even rounding and finite clipping.  Emit zero scale/codes for
   an all-zero block.
6. Pack the E2M1 codes, retain the FP8 block scales and F32 global scale, and
   stream the result into a ModelQ-native representation.
7. Decode the result immediately with the scalar reference path and report
   reconstruction diagnostics before an output is accepted.

The algorithm is deterministic for identical source bytes and metadata.  It
does not quantize activations, require a dataset, use RHT, use stochastic
rounding, or perform MSE/GPTQ/learned scale searches.  Those are future
recipes, not hidden behavior of the baseline.

### ModelQ-native versus runtime-compatible output

The baseline is **Level 1 — representation-valid** only.  A future native
container may describe `qdata`, `block_scale`, `global_scale`, group size,
axis, packing order, and an explicit convention version.  It must also retain
the source shape and say that its row-major 1D grouping is ModelQ-native.

It is not automatically usable by Transformer Engine or TensorRT:

- Transformer Engine's weight recipe can use 16x16 scaling and requires
  rowwise/columnwise copies, transposed data, padded scale tensors, and
  swizzled scales for GEMM.
- TensorRT's current explicit NVFP4 description uses a per-block scale and a
  cast of `clip(x / s, -6, 6)`; its documented scale storage is FP16/FP32 and
  does not define ModelQ's two-field E4M3-plus-global container.
- Model Optimizer's weight-only recipes and export metadata are another
  producer contract; their version, tensor names, packed buffers, and scale
  direction must be captured when a specific exporter is selected.

Therefore a runtime exporter must name the runtime and version, map the native
fields to its exact scale representation, produce its required transposes and
swizzles, and validate the resulting artifact.  No runtime compatibility is
claimed by this ADR.

### Test and reference strategy

The implementation work following this spike will retain the scalar FP4/FP8
codecs as the oracle and add:

- exhaustive checks of all 16 E2M1 codes and both nibble orders;
- golden zero, single-block, all-equal, maximum-range, subnormal-scale, and
  non-multiple shape cases;
- checked arithmetic for block counts, packed lengths, and scale lengths;
- deterministic repeated conversions and encode/decode reconstruction checks;
- property tests that every finite source value is decoded with its selected
  block/global scale and that no finite block scale encodes as E4M3 NaN or
  infinity; and
- differential fixtures captured from NVIDIA Model Optimizer's W4A16 NVFP4
  weight-only recipe and Transformer Engine's NVFP4 quantizer on a Blackwell
  machine.  The fixtures will record source bytes, packed codes, block-scale
  bytes, global scale, shape, and software versions.

Runtime validation is a separate Level 3/4 gate.  The first path is a tiny
deterministic matrix/GEMM on an NVIDIA Blackwell GPU (SM100 or later) using a
pinned Transformer Engine release and the current supported TensorRT 11.x
toolchain.  It will:

1. compare ModelQ's scalar dequantization to the NVIDIA reference values;
2. compare the runtime-specific exported bytes and metadata;
3. build an engine or Transformer Engine GEMM with the exact named layout;
4. compare outputs against an F32/BF16 reference within a documented error
   tolerance; and
5. record the GPU compute capability, driver, CUDA, Transformer Engine,
   TensorRT, and exporter versions.

Hardware emulation or a successful CPU decode is not Level 4 hardware
validation.  TensorRT's support matrix distinguishes emulated FP4 from true
hardware acceleration, so the final gate must run on Blackwell silicon.

## Alternatives considered

### Implement NVFP4 immediately

Rejected.  The element codec is ready, but selecting a block layout and
runtime metadata without a reference fixture would risk producing plausible
bytes that no named runtime can consume.

### Treat the Transformer Engine layout as the universal format

Rejected.  Transformer Engine's rowwise/columnwise, transpose, padding, and
swizzle requirements are execution contracts.  They should not constrain the
portable ModelQ-native representation before an exporter target is chosen.

### Use TensorRT's single per-block scale as the NVFP4 definition

Rejected.  TensorRT documents a useful explicit-quantization interface, while
Transformer Engine documents a two-level E4M3-plus-F32 hierarchy.  The
relationship between them belongs in a tested exporter mapping, not an
unqualified format alias.

### Enable all training enhancements in the baseline

Rejected.  Stochastic rounding, RHT, four-over-six candidate selection, and
MSE/GPTQ scale searches serve specific training or calibration workflows.
Keeping them out of the data-free reference makes the first implementation
reproducible and keeps later recipes measurable.

## Consequences

This ADR gives future implementation tasks a concrete numerical oracle,
grouping rule, and conversion sequence while preserving the project's
compatibility-level honesty.  It adds no Rust code, Cargo dependency, GPU
requirement, container convention, or runtime claim.  The first NVFP4 coding
tasks should implement the native scalar representation and tests, followed
by a separately reviewed exporter for one pinned NVIDIA runtime.

## Non-goals

This ADR does not implement:

- NVFP4 quantization, packing, or dequantization code;
- a SafeTensors/GGUF/native-container metadata convention;
- Transformer Engine scale swizzles or transposed runtime buffers;
- TensorRT/TensorRT-LLM/Model Optimizer export;
- activation quantization, calibration, RHT, stochastic rounding, 4over6,
  MSE, GPTQ, or learned scales; or
- CUDA, GPU detection, or Blackwell hardware support in the base build.
