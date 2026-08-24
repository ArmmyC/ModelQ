# ADR 0011: ModelQ-Native NVFP4 SafeTensors Convention

- Status: Accepted for the future native writer
- Date: 2026-08-25
- Scope: a self-describing SafeTensors convention for ModelQ-native NVFP4

## Context

ADR 0010 defines the ModelQ-native NVFP4 reference representation: signed
E2M1 values, one positive E4M3 block scale for every 16 values, and one F32
decode scale for the tensor.  The scalar implementation and its shape-aware
entry point now validate and round-trip those fields, but the in-memory result
is still a flat representation.  A future checkpoint writer needs an explicit
mapping from those fields to SafeTensors tensors and metadata.

ADR 0002 established the ModelQ-native INT8 convention.  It uses standard
SafeTensors tensors plus one required `__metadata__` manifest so a decoder can
discover original names, shapes, dtypes, preservation decisions, and the
quantized companions without model-specific knowledge.  NVFP4 needs the same
self-description, with two packed byte streams and one scalar rather than one
INT8 payload and one scale.

This convention is for ModelQ's own tools.  It does not define Transformer
Engine swizzles, TensorRT explicit-quantization metadata, a CUDA buffer layout,
or any other runtime contract.

## Decision

ModelQ will represent native NVFP4 output as a standard SafeTensors file with
a required string-valued `__metadata__` manifest.  A successfully written file
is **Level 2 — container-valid** and remains explicitly ModelQ-native; it is
not automatically runtime-compatible.

### File-level metadata

SafeTensors metadata values are strings, so numeric values and the manifest are
encoded as strings.  These keys are mandatory for an NVFP4 file:

| Key | Required value | Purpose |
| --- | --- | --- |
| `modelq.format` | `modelq-native` | Distinguishes the file from a runtime export. |
| `modelq.format_version` | `1` | Version of the common ModelQ-native envelope. |
| `modelq.compatibility_level` | `container-valid` | States the highest claim made by this file. |
| `modelq.quantization` | `nvfp4` | Identifies the representation. |
| `modelq.scheme` | `weight-only-blockwise` | Identifies the data-free weight recipe. |
| `modelq.algorithm` | `e2m1-e4m3-global-v0` | Identifies the ADR 0010 scale algorithm. |
| `modelq.element_format` | `fp4-e2m1` | Identifies the four-bit element codec. |
| `modelq.block_scale_format` | `fp8-e4m3` | Identifies the one-byte block-scale codec. |
| `modelq.global_scale_dtype` | `F32` | Identifies the tensor-wide scale storage. |
| `modelq.global_scale_semantics` | `decode` | Prevents confusing the stored scale with its reciprocal. |
| `modelq.block_size` | `16` | Number of values sharing one block scale. |
| `modelq.packing` | `e2m1-low-nibble-first` | Element zero is in each byte's low nibble. |
| `modelq.rounding` | `nearest-even` | Rounding policy used by the reference encoder. |
| `modelq.manifest` | JSON string | Self-describing tensor map defined below. |

The manifest is UTF-8 JSON with no insignificant whitespace.  Its tensor keys
are sorted lexicographically by the writer.  A decoder must reject an unknown
manifest schema or missing required fields and may ignore unknown fields for
forward-compatible extensions.

### Manifest schema

The first schema is `modelq.nvfp4.manifest.v1`:

```json
{
  "schema": "modelq.nvfp4.manifest.v1",
  "tensors": {
    "layer.weight": {
      "action": "quantized",
      "original_dtype": "F16",
      "original_shape": [4096, 4096],
      "axis": -1,
      "block_size": 16,
      "qdata_name": "layer.weight.qdata",
      "qdata_dtype": "U8",
      "qdata_shape": [4096, 2048],
      "block_scale_name": "layer.weight.block_scale",
      "block_scale_dtype": "U8",
      "block_scale_shape": [4096, 256],
      "global_scale_name": "layer.weight.global_scale",
      "global_scale_dtype": "F32",
      "global_scale_shape": [],
      "packing": "low-nibble-first",
      "block_scale_encoding": "e4m3-bit-pattern",
      "global_scale_semantics": "decode"
    },
    "layer.norm": {
      "action": "preserved",
      "original_dtype": "F32",
      "original_shape": [4096],
      "tensor_name": "layer.norm"
    }
  }
}
```

Manifest tensor keys are the exact original SafeTensors names.  Every source
tensor has exactly one record.  A quantized record has the fields shown above;
a preserved record has `action`, `original_dtype`, `original_shape`, and
`tensor_name`.  Unsupported or policy-preserved tensors are never silently
dropped.

### Quantized tensor payloads and shapes

The source shape must have at least one dimension, all dimensions must be
positive, and the final dimension must be divisible by 16.  Let the original
shape be `[d0, ..., d_last]`.  The writer emits exactly three companion
tensors:

1. `<source_name>.qdata` with SafeTensors dtype `U8` and shape
   `[d0, ..., d_last / 2]`.
2. `<source_name>.block_scale` with SafeTensors dtype `U8` and shape
   `[d0, ..., d_last / 16]`.
3. `<source_name>.global_scale` with SafeTensors dtype `F32` and scalar shape
   `[]`.

The physical shapes are intentional.  SafeTensors has no standard four-bit
NVFP4 dtype, so `qdata` stores two E2M1 elements per byte.  For byte index `j`,
the element at flattened index `2*j` is the low nibble and the element at
`2*j + 1` is the high nibble.  The final dimension is always even under this
convention, so there is no partial byte for a quantized tensor.

`block_scale` stores one raw E4M3 byte for every 16 consecutive values along
the final dimension.  Its flattened order follows the row-major source order.
Zero blocks store a zero scale and zero E2M1 magnitudes.  Non-zero blocks store
a positive finite E4M3 bit pattern.  `global_scale` stores the positive F32
decode scale from ADR 0010, not its reciprocal.  A decoder reconstructs:

```text
x_hat = decode_e2m1(qdata_nibble)
      * decode_e4m3(block_scale_byte)
      * global_scale_f32
```

The manifest's `qdata_shape`, `block_scale_shape`, dtype strings, axis, block
size, packing, and scale semantics are authoritative and must agree with the
file-level metadata.  A decoder must also derive the expected physical shapes
from `original_shape` and reject mismatches rather than guessing.

### Names, preservation, and collisions

Generated companion names use deterministic suffixes exactly as shown.  Before
opening the destination, a writer must reject any collision between:

- a generated `.qdata`, `.block_scale`, or `.global_scale` name;
- an original source tensor name;
- another generated name; or
- the reserved `__metadata__` key.

The writer must fail rather than rename, overwrite, or drop a tensor.  A
preserved tensor is copied under its original name with its original dtype,
shape, and bytes; it receives no companion tensors.

### Decoding procedure

A native decoder performs these checks and steps:

1. Require `modelq.format=modelq-native`, `modelq.format_version=1`,
   `modelq.quantization=nvfp4`, `modelq.compatibility_level=container-valid`,
   and the `modelq.nvfp4.manifest.v1` schema.
2. For each preserved record, read `tensor_name` and copy that SafeTensors
   tensor as-is.
3. For each quantized record, validate the original shape, derive the two
   physical shapes, and require the three named tensors to have the declared
   dtypes, shapes, and byte lengths.
4. Read the U8 qdata and block-scale bytes, unpack low nibble first, decode the
   E2M1 and E4M3 bit patterns, and multiply by the F32 global decode scale.
5. Return the reconstructed values in the F32 reference domain.  The
   `original_dtype` and `original_shape` fields remain available to callers
   that need to present or convert the result back to source metadata.

The decoder must validate finite scales, zero-block rules, block counts,
checked shape products, and all byte ranges before reconstructing values.

### Determinism and compatibility boundary

The writer must produce deterministic metadata ordering, manifest ordering,
generated names, physical shapes, and payload bytes for identical source bytes
and metadata.  It must use the existing safe output behavior: do not write
in-place over the source, plan offsets before conversion, and leave an
existing destination untouched when validation fails.

This convention provides a Level 1 representation definition and, once a
writer emits a structurally valid SafeTensors file, a Level 2 container claim.
It makes no Level 3 runtime-compatible or Level 4 hardware-validated claim.
Transformer Engine rowwise/columnwise buffers, transposes, padding, swizzled
scales, TensorRT scale contracts, and Blackwell execution belong to a future
named exporter and compatibility test.

## Alternatives considered

### Store packed qdata under the original shape

Rejected.  The packed payload has half as many bytes as E2M1 elements, so the
SafeTensors shape would not describe its physical byte count.  The explicit
packed qdata shape makes byte accounting and memory mapping unambiguous.

### Use a non-standard four-bit SafeTensors dtype

Rejected.  The native convention must be readable with standard SafeTensors
`U8` handling and must not pretend that a runtime-specific low-bit dtype exists
in the container.

### Store block scales as F32

Rejected.  The native representation's block scale is an exact E4M3 bit
pattern.  Widening it to F32 would lose the distinction between the encoded
scale bytes and their decoded numeric values and would change the format's
scale overhead.

### Combine block and global scales into one tensor

Rejected.  The separate tensors preserve the two-level hierarchy, make scale
direction explicit, and allow a decoder to validate each field independently.

### Reuse the INT8 manifest schema

Rejected.  NVFP4 has different payload counts, physical shapes, scale formats,
and reconstruction rules.  A distinct schema prevents a decoder from
silently interpreting NVFP4 bytes as INT8 data.

## Consequences

### Benefits

- A future writer can plan all three quantized payloads before writing data.
- A decoder can reconstruct NVFP4 without model-specific names or hidden shape
  assumptions.
- U8 payloads remain inspectable and portable across standard SafeTensors
  implementations.
- The explicit compatibility level prevents a native file from being mistaken
  for a Transformer Engine or TensorRT artifact.

### Costs and limitations

- Two U8 companion tensors add header and scale overhead for each quantized
  tensor, in addition to the F32 global scalar.
- The convention requires a positive, block-aligned final dimension and does
  not implicitly pad weights.
- A native reader/writer still needs to be implemented and tested; this ADR
  alone does not add a CLI format or checkpoint conversion path.
- Changing names, metadata, shapes, or reconstruction semantics requires a new
  manifest schema or format version.

## Follow-up

The next implementation increment may add a native NVFP4 SafeTensors layout
planner and writer using this schema, with synthetic fixtures that reopen the
file and compare decoded values to `modelq_quant::nvfp4`.  Runtime-specific
export must remain a separate ADR and must identify its runtime, version,
container, exact layout, and hardware validation path before code is added.

## Compatibility impact

This ADR adds a documented future convention and changes no current reader,
writer, CLI, or public quantizer behavior.  Existing ModelQ-native INT8 files
continue to use `modelq.int8.manifest.v1`.  An NVFP4 writer must reject
malformed shapes, names, scales, and plans before creating its destination.
