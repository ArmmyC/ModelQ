# ADR 0030: Group-Wise Low-Bit Formats and the Format Registry

- Status: Accepted for the codec, the file format and the CLI (Task 49, milestone M3 of ADR 0028); no quality claim is made for these formats
- Date: 2026-10-10
- Scope: the symmetric group-wise INT4 format, the experimental INT3, INT2 and INT1 formats, the ModelQ-native file layout for group-wise outputs, and the registry that states the status of every format the CLI writes
- Builds on: [ADR 0002](0002-modelq-native-quantized-tensor-convention.md) (INT8 convention), [ADR 0011](0011-nvfp4-native-safetensors-convention.md) (NVFP4 native convention), [ADR 0028](0028-open-source-quantizer-scope.md) (section 2, format statuses)

## Context

ADR 0028 section 2 requires that every output carry an honest status, and milestone M3 adds INT4 with group-wise scales, then experimental INT3, INT2 and INT1. An INT4 reference codec already existed in `modelq-quant` (PROJECT.md Stage B), but nothing reached the CLI or a file. Before this ADR the CLI could write INT8 (per-tensor) and the two NVFP4 formats only.

## Decisions

### 1. The codec: symmetric group-wise, 2 to 8 bits, and sign at 1 bit

- Values are grouped in storage order. Groups have `group_size` values; the last group may be shorter. The default group size is 128.
- Symmetric scheme (2 to 8 bits): one F32 scale per group, `max|w| / qmax` with `qmax = 2^(bits-1) - 1`. Values are rounded with ties away from zero and clamped to `[-qmax, qmax]`. The two's-complement code `-2^(bits-1)` is reserved and rejected when decoding. A zero group uses scale `1.0`; a scale that underflows to zero becomes the smallest positive subnormal. These rules are the INT4 reference's, so 4-bit output is bit-identical to it.
- Sign scheme (1 bit): one bit per value says whether it is non-negative; the scale is the group's mean absolute value, the least-squares choice for one sign per value. A zero group has scale `0.0` and decodes to exact zeros. INT1 is this scheme, because a symmetric one-bit code has no non-zero magnitude.

Symmetric rounding guarantees that every reconstructed value is within half a group scale of its source. The validator checks this bound for every value.

### 2. Packing: an LSB-first bitstream

Value `i` occupies stream bits `[i * bits, (i + 1) * bits)`, and byte `k` holds bits `[8k, 8k + 8)`. The last byte is padded with zero bits. At 4 bits this is the low-nibble-first layout of INT4. Because 8 values always fill a whole number of bytes, a stream can be encoded in pieces and each piece is byte-aligned.

### 3. The file: ModelQ-native SafeTensors, convention version 2

Each quantized source tensor `<name>` becomes two output tensors:

- `<name>.qdata`: dtype `U8`, shape `[ceil(elements * bits / 8)]`, the packed codes;
- `<name>.scale`: dtype `F32`, shape `[ceil(elements / group_size)]`, one scale per group.

Preserved tensors are copied unchanged. The `__metadata__` entries are `modelq.format = modelq-native`, `modelq.format_version = 2`, `modelq.quantization` (`int4`, `int3`, `int2` or `int1`), `modelq.scheme` (`symmetric-group-wise` or `sign-group-wise`), `modelq.algorithm`, `modelq.packing = lsb-first-bitstream`, `modelq.rounding = ties-away-from-zero`, `modelq.bits`, `modelq.group_size`, and either `modelq.qmin`/`modelq.qmax` (symmetric) or `modelq.code_map` (sign). `modelq.manifest` is a canonical JSON string with schema `modelq.lowbit.manifest.v1`, which maps every source tensor name to its outputs and records its original dtype, shape and element count.

Format version 2 is separate from the INT8 convention, which stays at version 1 (ADR 0002). A reader selects the decoder from `modelq.quantization`, and an unknown version or schema is an error rather than a guess.

### 4. Writing and sharding

The writer plans the layout with the same planner as INT8, generalized by a `QuantizedEncoding` parameter so that INT8 is unchanged. It streams one source tensor at a time: groups are buffered, codes go to the output as bytes are complete, and each tensor's group scales are held in memory (4 bytes per group) until they are written. `--max-shard-size` works as it does for INT8, with each shard carrying its own manifest.

Writes go to a temporary file that is renamed on success, and an existing destination is never replaced.

### 5. Validation

After writing, every output file is reopened, its manifest is read, every quantized tensor is decoded from its `qdata` and `scale` tensors, and the result is compared with the source. The report gives the maximum MSE, MAE and absolute error, the lowest SQNR, and the number of values outside the half-scale bound (which must be zero for symmetric formats). Preserved tensors must match the source byte for byte. Each source tensor must appear exactly once across the files.

### 6. The format registry and the experimental gate

`modelq-quant::formats` lists every format the CLI writes, with its bit width, scheme, default group size, status, container and specification. `modelq formats` prints it. The statuses are the ones ADR 0028 defines, and a status is never higher than its evidence:

| Format | Status | Needs `--experimental` |
| --- | --- | --- |
| int8 | representation-valid | no |
| int4 | representation-valid | no |
| int3, int2, int1 | experimental | yes |
| nvfp4 | representation-valid | no |
| nvfp4-te | hardware-validated: Transformer Engine 2.19.0 on NVIDIA B200 | no |

Writing an experimental format without `--experimental` is refused before any file is created.

## Consequences

- Users can write 4-bit group-wise files with one command, and can try 3, 2 and 1 bits deliberately.
- The 4-bit output is bit-identical to the INT4 reference, so the earlier reference tests now protect the CLI path as well.
- `modelq eval` reads only Transformer Engine containers, so **no quality is measured for these formats yet**. A low-bit output must not be relied on for quality until evaluation reads this format (a follow-up that ADR 0028 M1 will need). Resolved by ADR 0035.
- Encoding is sequential. Parallel encoding is a later optimization, measured separately (PROJECT.md section 25).
- Group-wise scales are not calibrated: each group uses its own maximum (or mean absolute value), with no search. The scale search of ADR 0025 could be applied to these formats later, and would need its own evidence.

## Verification

- Codec (`modelq-quant::lowbit`, 19 unit tests): every single code at every position for every width from 1 to 8 bits; every eight-value pattern at 1 and 2 bits; random round trips at every width and length; a streaming writer that matches whole-slice packing; reserved and out-of-range codes rejected; ties away from zero and clamping; hand-computed golden vectors for 2-bit, 3-bit and sign groups (packed bytes `0x74`, `[0xAB, 0x0C, 0x00]` and `0x05`); zero-group scales; non-finite inputs reported with their index; partial last groups; the reconstruction error bounded by half a scale for every width and group size; and **4-bit output bit-identical to the INT4 reference** (packed bytes and scales) across six group sizes and seven lengths.
- Registry (`modelq-quant::formats`, 5 tests): unique identifiers, the experimental flag matching the experimental status, status labels, bit widths matching the codecs, and unknown names rejected.
- Layout and file (`modelq-io`, 7 library tests in `tests/lowbit_io.rs`): four configurations (4-bit with group 128, 3-bit with group 32, 2-bit with group 7, sign with group 64) each written, reopened, decoded and validated, with zero values outside the half-scale bound for the symmetric formats; the metadata and manifest keys; byte-identical reruns; refusal to replace an existing file; refusal of a plan made for another encoding; a non-low-bit file is not read as one; and a source whose preserved tensor differs is caught.
- CLI (`tests/lowbit_cli.rs`, 12 tests, run on Linux in Modal): `modelq formats` lists every status; the experimental formats are refused without `--experimental` and write nothing; the 4-bit default group is 128; `--group-size` is recorded and validated; `--group-size` is refused for INT8; zero is a usage error; the sign format writes its code map; sharded output is written and validated shard by shard; repeated runs give the same bytes; and an existing output is not replaced. The first Linux run exposed one bug, which is fixed: a sharded write also lists its JSON index, and validation tried to open it as a shard.
- Whole suite on Linux in Modal: `cargo fmt --check` passes, `cargo test --workspace` gives 327 passed, 0 failed, 2 ignored; clippy with `-D warnings` passes on all targets.

Not verified: the quality of any group-wise output (no evaluation reads these files yet), and the CLI on Windows (the test executable is blocked by Application Control on the development PC, as before).
