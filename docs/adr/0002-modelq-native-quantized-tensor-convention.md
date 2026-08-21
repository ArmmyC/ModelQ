# ADR 0002: ModelQ-Native INT8 SafeTensors Convention

- Status: Accepted for the v0 reference writer
- Date: 2026-08-21
- Scope: symmetric per-tensor INT8 output

## Context

ModelQ needs an output representation for the scalar INT8 path before a
SafeTensors writer is implemented. SafeTensors can store the raw INT8 payload
and scale tensors, but it does not define how those tensors are related or how
the original source tensor should be reconstructed. A decoder must therefore
be able to discover the relationship from the file itself, without knowing a
particular model's tensor names or relying on undocumented conventions.

The representation is an interchange format for ModelQ's own tools. It is not
a claim of compatibility with an inference runtime that happens to support
SafeTensors or INT8.

## Decision

ModelQ will write a standard SafeTensors file with a required, string-valued
`__metadata__` manifest. The manifest identifies the file as ModelQ-native,
describes the INT8 algorithm, and maps every source tensor to either a
quantized pair or a preserved output tensor.

### File-level metadata

The following `__metadata__` keys are mandatory. SafeTensors metadata values
are strings, so numeric values and the manifest are encoded as strings.

| Key | Required value | Purpose |
| --- | --- | --- |
| `modelq.format` | `modelq-native` | Explicitly distinguishes this format from runtime-compatible SafeTensors. |
| `modelq.format_version` | `1` | Version marker for the output convention. |
| `modelq.quantization` | `int8` | Quantized representation selected by the writer. |
| `modelq.scheme` | `symmetric-per-tensor` | One scale is shared by one source tensor. |
| `modelq.algorithm` | `max-abs-scale-v0` | Scale and quantization algorithm identifier. |
| `modelq.rounding` | `ties-away-from-zero` | Rounding policy used before clamping. |
| `modelq.qmin` | `-127` | Lowest emitted INT8 value. |
| `modelq.qmax` | `127` | Highest emitted INT8 value; `-128` is unused. |
| `modelq.manifest` | JSON string | Self-describing tensor map defined below. |

The manifest value is UTF-8 JSON encoded as one metadata string. Its top-level
shape is:

```json
{
  "schema": "modelq.int8.manifest.v1",
  "tensors": {
    "layer.weight": {
      "action": "quantized",
      "original_dtype": "F16",
      "original_shape": [4096, 4096],
      "qdata_name": "layer.weight.qdata",
      "qdata_dtype": "I8",
      "scale_name": "layer.weight.scale",
      "scale_dtype": "F32",
      "scale_shape": []
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

Manifest tensor keys are the exact original SafeTensors tensor names. The
writer must emit deterministic JSON with UTF-8 text, lexicographically ordered
tensor keys, and no insignificant whitespace. A decoder must treat unknown
manifest fields as forward-compatible extensions and must reject an unknown
`schema` value or missing required fields.

### Quantized tensor names and payloads

For an input tensor named `<source_name>` selected for quantization, the output
contains exactly two tensors:

1. `<source_name>.qdata` with SafeTensors dtype `I8` and the original shape.
   Its payload contains one signed INT8 value per source element, in the
   symmetric range `[-127, 127]`.
2. `<source_name>.scale` with SafeTensors dtype `F32` and scalar shape `[]`.
   It contains the one positive per-tensor scale used for all qdata values.

The original tensor name is not also emitted as a third payload tensor. The
manifest records both generated names, the original dtype, and the original
shape. The qdata shape must equal `original_shape`; the scale must be a finite,
positive F32 scalar.

Generated names are deterministic suffixes, not hashes, so a decoder can
reconstruct the relationship even without the manifest. The manifest remains
authoritative and is required for original dtype and preservation information.

Before writing, a writer must reject a source set if any generated `.qdata` or
`.scale` name collides with an existing source name, the reserved
`__metadata__` key, or another output name. It must fail rather than silently
rename or drop a tensor.

### Preserved tensors

Every source tensor is represented by exactly one manifest record. A tensor not
selected for INT8, including every non-floating or small tensor, is preserved
under its original name with its original dtype, shape, and payload bytes.
Its manifest record has `"action": "preserved"` and a `tensor_name` equal to
the original name. No `.qdata` or `.scale` companion is emitted for it.

No source tensor may be silently dropped. An unsupported dtype or an unsafe
name collision is an explicit writer error, not an implicit preservation or
skip.

### Decoding procedure

A generic decoder can reconstruct the output as follows:

1. Read `__metadata__` and require `modelq.format=modelq-native`,
   `modelq.format_version=1`, and `modelq.manifest` with schema
   `modelq.int8.manifest.v1`.
2. For each manifest record with `action=preserved`, read `tensor_name` and
   copy that tensor as-is.
3. For each record with `action=quantized`, read the `I8` qdata tensor and
   scalar `F32` scale named by the record. Reconstruct each reference value as
   `f32(q) * scale`.
4. Use `original_dtype` and `original_shape` from the record when presenting or
   converting the reconstructed source tensor. Validate that qdata shape and
   element count match `original_shape`.

The reconstruction algorithm is intentionally independent of any model
architecture or runtime-specific tensor naming rules.

## Alternatives considered

### Reuse the original tensor name for qdata

Rejected because the original dtype and payload would be replaced without a
mechanically obvious indication that a scale tensor is required. Separate
`.qdata` and `.scale` names make the representation inspectable.

### Store scale as a one-element `[1]` tensor

Rejected for the v0 convention. A scalar shape `[]` states that the scale is a
single value, avoids an artificial dimension, and is already supported by the
ModelQ tensor metadata checks.

### Put one metadata key beside every tensor

Rejected in favor of one manifest. A JSON manifest keeps arbitrary source names
and all per-tensor fields together, avoids metadata-key escaping rules, and
gives decoders one schema/version boundary to validate.

### Emit `U8` instead of `I8` qdata

Rejected because the reference algorithm's defined range is signed and uses
the standard SafeTensors `I8` dtype directly. Future packed formats may choose
other dtypes under a new convention version.

## Consequences

### Benefits

- A decoder can discover every output tensor, action, dtype, shape, qdata name,
  and scale name from standard SafeTensors plus one JSON manifest.
- Preserved tensors remain byte-for-byte ordinary SafeTensors tensors.
- The representation is deterministic and can be planned before writing.
- The explicit `modelq-native` marker prevents accidental claims of runtime
  interoperability.

### Costs and limitations

- Quantized tensors use two output tensors, increasing header and scale
  overhead for very small tensors. The Task 8 policy avoids those tensors by
  preserving small inputs.
- The convention is specific to ModelQ and is not automatically consumable by
  existing inference runtimes.
- Name collision checks are required before output creation.
- Changing names, metadata, or reconstruction semantics requires a new format
  version or manifest schema.

## Compatibility impact

This ADR defines a new ModelQ-native format and does not change the existing
SafeTensors inspection or reader APIs. The future writer must reject malformed
or conflicting plans before opening the final output path. A later runtime
exporter may translate this representation into a runtime-specific format,
but that translation is outside this convention.
