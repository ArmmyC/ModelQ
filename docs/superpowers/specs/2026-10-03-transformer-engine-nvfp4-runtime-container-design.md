# Transformer Engine NVFP4 Runtime Container and GEMM Proof

- Status: design approved in chat; awaiting written-spec review
- Date: 2026-10-03
- Scope: one NVFP4 weight matrix in a SafeTensors artifact, loaded by a pinned Transformer Engine bridge and used in one Blackwell GEMM
- Related decisions: [ADR 0012](../../adr/0012-transformer-engine-nvfp4-export-profile.md), [Transformer Engine NVFP4 export profile design](2026-09-15-transformer-engine-nvfp4-export-profile-design.md)

## Goal

Take the CPU-side rowwise profile from ADR 0012 across one serialization and runtime boundary:

```text
ModelQ Rust NVFP4 profile
        |
        v
ModelQ-described SafeTensors artifact
        |
        v
Python bridge pinned to Transformer Engine 2.19.0
        |
        v
one NVFP4 GEMM on Blackwell
```

This proves a narrow compatibility path for one matrix and one operation. It does not load or execute a complete model. The project continues to be a quantization compiler/toolkit, not an inference engine, as required by `PROJECT.md`.

## Context

ADR 0012 defines `transformer-engine.nvfp4.rowwise.1x16.v1` as a CPU-only mapping with three logical fields: packed rowwise E2M1 data, a 128-by-4-aligned rowwise E4M3 scale matrix, and a one-element F32 rowwise amax. It explicitly does not provide a container, a Transformer Engine tensor object, columnwise data, GEMM scale swizzling, or a hardware compatibility claim.

Transformer Engine v2.19.0 is the selected runtime version. Its `NVFP4Tensor` source accepts optional columnwise fields, and its `general_gemm` implementation selects rowwise or columnwise representations according to the TN layout and operand role. In TN layout, the first operand uses its rowwise representation and the second operand uses its rowwise representation. The proposed test therefore places the exported weight in the first, transposed operand position and quantizes a synthetic second operand with Transformer Engine. This is one specific tested usage, not a claim that a rowwise-only tensor supports every Transformer Engine operation. See the [v2.19 NVFP4 tensor source](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/pytorch/tensor/nvfp4_tensor.py) and [v2.19 GEMM source](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/pytorch/cpp_extensions/gemm.py).

The v2.19 installation guide requires Linux x86_64, CUDA 12.1 or newer (12.8 or newer for Blackwell), a compatible NVIDIA driver, and cuDNN 9.3 or newer. The NVFP4 guide requires SM100/Blackwell or later. These requirements make the runtime proof an optional Linux hardware test; they do not affect the cross-platform Rust library. See the [v2.19 installation guide](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/docs/installation.rst) and [NVFP4 guide](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html).

## Decisions

### 1. Scope: one matrix and one GEMM

The test artifact contains one logical rank-two weight matrix of shape `[M, K]`, with `K` divisible by 16. A deterministic synthetic second operand has shape `[M, N]`. The test computes one TN GEMM, `weight.T @ second_operand`, and expects an output of shape `[K, N]`.

The exported matrix is consumed unchanged as the first GEMM operand. Transformer Engine quantizes only the synthetic second operand for the test. No ModelQ activation-quantization feature is added.

### 2. Container: one SafeTensors file per matrix

The Rust writer emits a standard SafeTensors file containing exactly these runtime tensors for the selected matrix:

| Tensor name | SafeTensors dtype | Shape |
| --- | --- | --- |
| `<name>.rowwise_data` | `U8` | `[M, K/2]` |
| `<name>.rowwise_scale_inv` | `U8` | `[round_up(M,128), round_up(K/16,4)]` |
| `<name>.amax_rowwise` | `F32` | `[1]` |

Packed values are copied byte-for-byte from the validated ModelQ NVFP4 payload. Logical scales are copied into the leading region of a zero-filled padded matrix. Padding is zero. Scales are not GEMM-swizzled in the file. The amax/global-scale relationship is exactly the one defined in ADR 0012.

The SafeTensors `__metadata__` map contains string values under `modelq.format` and `modelq.manifest`. `modelq.format` is `transformer-engine-nvfp4-safetensors-v1`. `modelq.manifest` is deterministic JSON with these required fields:

```json
{
  "schema_version": 1,
  "profile_id": "transformer-engine.nvfp4.rowwise.1x16.v1",
  "runtime": { "name": "transformer_engine", "version": "2.19.0" },
  "tensor_name": "weight",
  "logical_shape": [64, 64],
  "fields": {
    "rowwise_data": "weight.rowwise_data",
    "rowwise_scale_inv": "weight.rowwise_scale_inv",
    "amax_rowwise": "weight.amax_rowwise"
  },
  "quantization": {
    "data_format": "E2M1",
    "block_scale_format": "E4M3",
    "block_size": 16,
    "scaling": "rowwise_1x16_tensor_global"
  },
  "scale_storage": { "padding": [128, 4], "gemm_swizzled": false },
  "global_scale_denominator": 2688.0
}
```

The names and logical shape vary with the exported matrix. The other values above are fixed by schema version 1. The manifest records that stored scales are padded but not GEMM-swizzled.

The artifact does not contain a Python pickle, a Torch checkpoint, or test-only reference weights. It is a ModelQ-described SafeTensors artifact that requires the accompanying bridge; it is not advertised as a general Transformer Engine checkpoint.

### 3. Runtime bridge: isolated, pinned, and explicit

A separate Python tool reads SafeTensors and validates its metadata and field shapes before attempting CUDA work. It constructs a v2.19.0 `NVFP4Tensor` from the rowwise tensors with columnwise data set to `None`, `fp4_dtype` set to E2M1, and GEMM-swizzled-scale state set to false. Its `NVFP4Quantizer` is configured for rowwise 1x16 scaling, with 2D scaling, RHT, stochastic rounding, and 4over6 disabled. The bridge does not requantize or transpose the exported weight.

For the GEMM, the bridge uses the v2.19.0 `general_gemm` TN path with the exported weight as the first operand. Transformer Engine owns any runtime-only layout preparation required by that path. Runtime-prepared buffers are temporary and are not written back to the artifact.

The optional Python environment pins `transformer-engine==2.19.0` and its Python-side dependencies separately from Cargo. The validator reports the exact Python, PyTorch, CUDA runtime, driver, Transformer Engine, and GPU versions used. Cargo.toml/Cargo.lock gain no Transformer Engine, CUDA, or Python dependencies.

### 4. Numerical oracle and scope of the proof

A small Rust example generates a deterministic 64-by-64 test matrix, applies the existing ModelQ NVFP4 quantizer/profile/writer, and also writes a separate reference SafeTensors file containing the F32 dequantized matrix produced by ModelQ. The reference is test-only and is never included in the runtime artifact.

The Python validator creates a deterministic `[64,64]` second operand with Transformer Engine's deterministic 1x16 NVFP4 quantizer, dequantizes that operand for the reference, and compares the TE GEMM result against:

```text
ModelQ-dequantized exported weight.T @ TE-dequantized second operand
```

The GEMM result must have shape `[64,64]`, contain only finite values, and satisfy `torch.testing.assert_close` with the fixed tolerance `rtol=0.125` and `atol=0.0675`. Before the GEMM, the bridge also checks that TE's loaded/dequantized weight agrees with the Rust-generated ModelQ reference at `rtol=1e-5` and `atol=1e-5`. The validator reports maximum absolute error and the environment versions on success.

This proof establishes compatibility only for the named artifact schema, Transformer Engine 2.19.0, the tested TN GEMM usage, and a Blackwell-or-newer GPU. It does not establish whole-model compatibility or compatibility with other TE versions.

## Data flow and validation boundaries

1. Rust validates the native NVFP4 value and shape, maps it using ADR 0012, and writes the three tensor fields plus versioned manifest.
2. Rust tests reopen the generated artifact through ModelQ's SafeTensors reader and verify field names, dtype/shape metadata, payload bytes, padding, and manifest values.
3. A Python CPU-only mode opens both fixture files through the SafeTensors Python reader and validates the manifest and tensor metadata without importing Transformer Engine or creating a CUDA context.
4. The explicit runtime mode checks the exact TE version and Blackwell capability, constructs the runtime tensor, creates the deterministic second operand, runs the single TN GEMM, compares it with the reference, and prints the tested environment.

Each boundary has its own failure. Invalid or unsupported schema, missing tensor fields, wrong dtype/shape, nonzero alignment padding, mismatched TE version, absent CUDA, an unsupported GPU, tensor construction failure, GEMM failure, non-finite output, or numerical mismatch returns a failing status with an actionable diagnostic. Hardware unavailability is never reported as a passing GEMM test.

## Planned files and dependencies

- `crates/modelq-io/src/writer.rs`: add the narrow TE-profile SafeTensors writer alongside existing output behavior; do not redesign the general writer in this task.
- `crates/modelq-io/src/transformer_engine.rs`: expose the writer through the existing profile boundary if that keeps the API cohesive.
- `tests/transformer_engine_nvfp4_safetensors.rs`: CPU contract tests for serialization and metadata.
- `crates/modelq-io/examples/transformer_engine_nvfp4_fixture.rs`: generate the disposable runtime artifact and separate F32 reference file.
- `tools/transformer_engine_nvfp4/validate.py`: Python CPU-container and Blackwell runtime modes.
- `tools/transformer_engine_nvfp4/requirements.txt`: optional Python dependencies with TE pinned to 2.19.0; no Python dependency is added to the Rust application.
- `tools/transformer_engine_nvfp4/README.md`: environment prerequisites and exact commands for generating and validating the fixture.
- `docs/adr/0013-transformer-engine-nvfp4-runtime-container.md`: record the finalized serialization/runtime boundary and compatibility claim.
- `README.md`: summarize the capability and clearly label the hardware requirement and scope.

The existing `serde_json` dependency is sufficient for the manifest. No new Rust dependency, CUDA feature, CLI command, GitHub Actions workflow, full-model loader, benchmark, or checked-in binary fixture is planned.

## Verification and compatibility language

The implementation task will run:

1. `cargo test --workspace --all-targets`;
2. `cargo fmt --all -- --check`;
3. `cargo clippy --workspace --all-targets -- -D warnings`;
4. the fixture generator and Rust SafeTensors round-trip tests;
5. the Python CPU container check; and
6. the explicit runtime validator on a compatible Linux Blackwell host.

The first five checks can establish representation/container correctness. Only a successful sixth check establishes the Task 28 hardware/runtime proof. If no compatible Blackwell host is available, implementation may still be checked for CPU/container correctness, but the task must remain explicitly unverified at the hardware level and must not claim Level 3 or Level 4 compatibility. No GPU test is added to the default cross-platform Cargo test suite.

## Alternatives considered

- **Serialize a pickled `NVFP4Tensor`:** rejected because the artifact would depend on Python object serialization and a specific library object version rather than a safe, explicit tensor container.
- **Call the ModelQ-native NVFP4 SafeTensors convention a TE checkpoint:** rejected because ADR 0011 and ADR 0012 define different field names, padding, and scale metadata. The new artifact needs its own explicit profile manifest and runtime check.
- **Store or manufacture columnwise weight data:** rejected for this single TN proof. It is absent from the current profile, and generating a new direction could change quantization semantics. The test chooses the documented TN operand slot that consumes the available rowwise representation; a compatibility failure is surfaced rather than hidden.
- **Add TE/CUDA to the Rust build or run the GPU test in ordinary CI:** rejected to keep Rust CPU-only and the cross-platform fallback intact. Runtime validation remains an explicit external hardware test.
- **Load a whole transformer model:** rejected because it would add architecture, checkpoint, activation, and inference concerns beyond the approved single-matrix proof.

## Upstream references

- [Transformer Engine v2.19.0 package](https://pypi.org/project/transformer-engine/2.19.0/)
- [Transformer Engine v2.19 NVFP4 tensor source](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/pytorch/tensor/nvfp4_tensor.py)
- [Transformer Engine v2.19 GEMM source](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/pytorch/cpp_extensions/gemm.py)
- [Transformer Engine v2.19 installation guide](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/docs/installation.rst)
- [Transformer Engine NVFP4 guide](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)
