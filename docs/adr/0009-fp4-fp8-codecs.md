# ADR 0009: Reference FP4 and FP8 Element Codecs

- Status: Accepted
- Date: 2026-08-22
- Scope: data-free element encode/decode for FP4 E2M1, FP8 E4M3, and FP8 E5M2

## Context

Task 19 needs auditable bit codecs before ModelQ adds block scaling, tensor
packing, or a runtime-specific FP4/FP8 exporter.  The codecs must preserve the
representation boundary: they describe one element, not a quantization recipe
or a container format.

## Decision

Add `modelq_quant::float` with three small modules:

- `fp4_e2m1`: four bits, one sign bit, two exponent bits, one fraction bit,
  exponent bias 1, and maximum finite magnitude 6;
- `fp8_e4m3`: eight bits, one sign bit, four exponent bits, three fraction
  bits, exponent bias 7, and maximum finite magnitude 448; and
- `fp8_e5m2`: eight bits, one sign bit, five exponent bits, two fraction bits,
  exponent bias 15, and maximum finite magnitude 57,344.

The E2M1 finite magnitudes are `0`, `0.5`, `1`, `1.5`, `2`, `3`, `4`, and `6`,
with a sign bit applied to each magnitude.  It has no infinity or NaN
encoding.  E4M3 reserves `0x7f` and `0xff` as NaN and treats the other
exponent-all-ones values as finite; it has no infinity encoding.  E5M2 uses
exponent `0x1f` with zero fraction for signed infinity and a non-zero fraction
for NaN.

## Conversion semantics

All encoders use round-to-nearest-even.  At an exact halfway point, the
candidate whose stored significand least-significant bit is zero wins.  Signed
zero is preserved.

The encoder follows an explicit `satfinite` policy for values outside the
finite range: finite overflow and infinity become the signed maximum finite
code.  FP8 NaNs become the canonical positive `0x7f` NaN.  FP4 cannot encode a
NaN, so `fp4_e2m1::encode` returns `CodecError::NaNNotRepresentable` instead
of silently changing it.  Decoding is total over every bit pattern; all
special values are returned as the corresponding F32 zero, finite, infinity,
or NaN value.

These choices follow the current [NVIDIA CUDA E2M1
documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-math-api/cuda_math_api/struct____nv_fp4_e2m1.html)
for satfinite, nearest-rounding conversion and no-Inf/NaN representation, and
the [CUDA FP8 conversion
documentation](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)
and [NVIDIA CUTLASS FP8 software
reference](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/float8.h)
for FP8 layout, nearest-even conversion, canonical NaN, and satfinite
conversion.  They are documented here so a future runtime-specific path can
make any intentional semantic difference visible.

## Verification

The module exhaustively decodes and re-encodes all 16 FP4 patterns and all 256
patterns for each FP8 format.  The tests account for canonical NaN and
satfinite infinity mappings, then separately check known decode vectors,
rounding ties, signed zero, NaN handling, and saturation boundaries.

## Non-goals

This ADR does not define:

- per-tensor, per-channel, or block scaling;
- nibble packing for FP4 arrays;
- NVFP4's block-of-16 E4M3 scales and global F32 scale;
- GGUF/SafeTensors metadata; or
- CUDA, Transformer Engine, TensorRT, or hardware compatibility.

Those are separate representation, algorithm, container, and runtime decisions.
