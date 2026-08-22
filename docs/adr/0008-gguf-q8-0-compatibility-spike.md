# ADR 0008: Focused GGUF Q8_0 Compatibility Spike

- Status: Accepted
- Date: 2026-08-22
- Scope: one ModelQ F32 tensor encoded as one GGUF v3 Q8_0 tensor

## Context

ModelQ needs a concrete container-format checkpoint before attempting a broad
GGUF reader or exporter. “GGUF support” is too large to establish safely in a
single step: the file format contains many metadata types, tensor types, model
architectures, and runtime assumptions. Task 18 therefore requires one exact
quantization path, a pinned runtime reference, and an independently inspectable
fixture.

## Decision

Implement the current llama.cpp Q8_0 representation and only the minimal GGUF
v3 records needed for a deterministic one-tensor fixture.

The pinned reference is llama.cpp commit
[`d775b8967a46d8beb110d444aa3b8938179e0dd8`](https://github.com/ggml-org/llama.cpp/commit/d775b8967a46d8beb110d444aa3b8938179e0dd8),
which is the repository’s current `master` tip for this spike. The local CMake
configuration at that commit reports version `0.2.0-dev`. The exact
representation is:

- `GGML_TYPE_Q8_0 = 8`;
- 32 source values per block;
- one little-endian binary16 scale (`d`) followed by 32 signed int8 values;
- `d = max(abs(block)) / 127`; and
- each quantized value is `round(value / d)`, with a zero block represented by
  a zero scale and zero bytes.

The quantizer computes `d` in F32 for the reciprocal and rounding pass, then
stores that same scale converted to binary16; dequantization uses the stored
binary16 value.

ModelQ uses the already-pinned workspace `half = 2.4.1` crate directly in the
quantization crate so the stored scale is the exact binary16 value. This is a
representation dependency, not a llama.cpp build or runtime dependency.

The GGUF writer emits version 3, little-endian fields, 32-byte tensor-data
alignment, three UINT32 metadata values (`general.alignment`,
`general.quantization_version`, and `general.file_type`), one tensor-info
record, and the padded Q8_0 payload. Tensor dimensions are written in the
order used by the reference GGUF writer and are restored to ModelQ order by
the focused inspector. It intentionally omits `general.architecture`: this
fixture is a container-format probe, not a runnable model.

## Compatibility level

The fixture is **Level 2 — container-valid**. It can be parsed by the pinned
llama.cpp GGUF reader, but it is not claimed to be a complete model and is not
used to claim `llama-cli` inference compatibility. Level 3 requires a valid
model architecture and runtime load path; Level 4 additionally requires
hardware validation. Neither is in scope for this spike.

## Validation procedure

The repository contains a deterministic fixture generator:

```text
cargo run -p modelq-io --example gguf_q8_0_fixture -- C:\path\fixture.gguf
```

To validate it with the pinned llama.cpp reader, build the `llama-gguf` target
from a checkout at the exact commit above, then run:

```text
llama-gguf C:\path\fixture.gguf r n
```

The final `n` asks the inspection tool not to require model-specific tensor
data checks. The command verifies that llama.cpp accepts the GGUF header,
metadata, tensor info, alignment, and Q8_0 payload as a container. The Rust
tests independently inspect the same structure, verify deterministic bytes,
round-trip the stored binary16 scales, and refuse incomplete blocks.

## Alternatives considered

### Implement every GGUF quantization type

Rejected. It would create a broad API and make the compatibility claim hard to
audit. Future types should each arrive with their own reference mapping and
runtime fixture.

### Choose Q4_0 for the first spike

Rejected for now. ModelQ’s existing INT4 path is group-wise and does not map
directly to Q4_0’s packed block layout. Q8_0 has a small exact block format
that can be compared byte-for-byte with the existing F32 reference data.

### Add a llama.cpp build dependency to Cargo

Rejected. The Rust crate remains portable and dependency-light; external
runtime validation is a documented, pinned verification step rather than a
link-time requirement.
