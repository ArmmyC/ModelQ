# ADR 0013: Transformer Engine NVFP4 Runtime Container

- Status: Accepted
- Date: 2026-10-04
- Scope: one rowwise NVFP4 matrix, an explicit SafeTensors container, and a pinned single-GEMM validator
- Hardware compatibility: validated for the declared schema and single TN GEMM on NVIDIA B200 (Levels 3 and 4)

## Goal and context

Task 28 extends the CPU profile in [ADR 0012](0012-transformer-engine-nvfp4-export-profile.md) with a deterministic serialized artifact and an optional Transformer Engine (TE) 2.19.0 bridge. The goal is to test one exported matrix in one Blackwell GEMM. This does not add whole-model loading or inference. The approved [design](../superpowers/specs/2026-10-03-transformer-engine-nvfp4-runtime-container-design.md) defines this boundary.

The CPU representation and container checks are implemented. The explicit hardware path passed on an NVIDIA B200 on 2026-10-06. This establishes runtime and hardware compatibility only for the exact schema, TE version, and TN GEMM described here; it does not establish whole-model loading or inference.

## Accepted container contract

`modelq_io::writer::write_transformer_engine_nvfp4_safetensors` writes one rank-two matrix per standard SafeTensors file. The logical shape is `[M, K]`, both dimensions are positive, and `K` is divisible by 16. The writer validates the supplied profile before writing. The file contains exactly three tensors:

| Tensor name | Dtype | Physical shape | Fixture shape |
| --- | --- | --- | --- |
| `<name>.rowwise_data` | `U8` | `[M, K/2]` | `[64, 32]` |
| `<name>.rowwise_scale_inv` | `U8` | `[round_up(M,128), round_up(K/16,4)]` | `[128, 4]` |
| `<name>.amax_rowwise` | `F32` | `[1]` | `[1]` |

Packed E2M1 data is copied byte-for-byte with ModelQ's low-nibble-first order. The logical E4M3 block-scale bytes occupy the leading `[M, K/16]` region in row-major order. All alignment padding is zero. Stored scales are not GEMM-swizzled. The artifact has no columnwise representation.

For a nonzero tensor, amax is `global_decode_scale * (448.0 * 6.0)`; division by `2688.0` recovers the global decode factor. All-zero tensors retain the native zero payload convention and use zero amax. The scale field's TE name is `rowwise_scale_inv`, but its E4M3 bytes are local block decode scales.

The SafeTensors `__metadata__` map has string values: `modelq.format` is `transformer-engine-nvfp4-safetensors-v1`, and `modelq.manifest` is deterministic JSON. For the fixture it decodes to:

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

Only the tensor name, derived field names, and logical shape vary with the matrix. Other values are fixed by schema version 1. This is a ModelQ-described artifact requiring the bridge, not a general TE checkpoint or Python object serialization.

## Constructor and GEMM boundary

The explicit Python runtime command first validates the artifact and separate reference on CPU, then requires logical shape `[64,64]` and the pinned hardware environment. It constructs a TE 2.19.0 `NVFP4Tensor` with the exported rowwise buffers, `dtype=torch.float32`, `fp4_dtype=DType.kFloat4E2M1`, all columnwise buffers and amax set to `None`, `with_gemm_swizzled_scales=False`, `row_scaled_nvfp4=False`, and `nvfp4_use_4over6=False`.

Its `NVFP4Quantizer` uses rowwise 1x16 scaling with columnwise output, 2D quantization, RHT, stochastic rounding, random sign masks, and 4over6 disabled. Only the deterministic synthetic second operand is TE-quantized. The exported weight is neither requantized nor transposed by the bridge.

The call is `general_gemm(weight, second_operand, out_dtype=torch.float32, layout="TN")`. Under the pinned TE 2.19 row-major convention this computes `second_operand @ weight.T`, producing `[64,64]`. Both operand slots use rowwise representations in this TN path. TE owns runtime-only layout preparation; those temporary buffers are not written back to the artifact. This operation does not establish support for other dimensions, layouts, or TE versions. The convention follows the [v2.19 Python GEMM wrapper](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/pytorch/cpp_extensions/gemm.py) and [C++ shape contract](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/pytorch/csrc/extensions/gemm.cpp).

## Numerical oracle and validation

The Rust fixture example quantizes a deterministic `[64,64]` matrix and writes two separate files. The runtime artifact contains only the three fields above. The test-only reference file contains one F32 `[64,64]` tensor, `weight.dequantized_reference`, with metadata `modelq.reference_schema=transformer-engine-nvfp4-reference-v1`. Its values come from ModelQ's native dequantization.

The CPU command checks schema values, exact field identities, dtypes, physical shapes, zero scale padding, finite nonnegative amax, and the separate reference's schema, shape, dtype, and finite values. It imports neither TE nor PyTorch and creates no CUDA context. It does not numerically establish TE compatibility.

The runtime command first compares TE dequantization of the loaded weight with the Rust reference using `rtol=1e-5`, `atol=1e-5`. Its GEMM oracle is:

```text
TE-dequantized second operand @ ModelQ-dequantized exported weight.T
```

The output must have shape `[64,64]`, be finite, and satisfy `torch.testing.assert_close` with `rtol=0.125`, `atol=0.0675`. Only after success does the command report maximum absolute error and Python, PyTorch, CUDA runtime, cuDNN, driver, TE, GPU, and compute-capability versions. Invalid artifacts, unsupported environments, construction failures, GEMM failures, and numerical mismatches return nonzero. An unavailable GPU is not a pass.

## Prerequisites and compatibility status

CPU checks require Rust for fixture generation and Python with NumPy and SafeTensors. Runtime validation additionally requires Linux x86_64, CUDA 12.8+, a compatible NVIDIA driver, cuDNN 9.3+, a Blackwell-or-newer GPU (compute capability 10.0+), a compatible CUDA-enabled PyTorch installation, and `transformer-engine[pytorch]==2.19.0`. See the [v2.19 installation requirements](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/docs/installation.rst) and [v2.19 tensor implementation](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/pytorch/tensor/nvfp4_tensor.py).

The [tool README](../../tools/transformer_engine_nvfp4/README.md) gives separate CPU and hardware commands. The recorded B200 run establishes Level 3 runtime compatibility and Level 4 hardware validation for this single operation only. It does not cover other shapes, layouts, TE versions, hardware, or complete models.

### Recorded Blackwell proof

The explicit runtime command passed in a one-shot Modal container on 2026-10-06. The tested environment was:

| Component | Tested version |
| --- | --- |
| Container image | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` |
| Python | `3.12.1` |
| PyTorch | `2.9.0+cu128` |
| CUDA runtime | `12.8` |
| cuDNN | `91002` (`9.10.2`) |
| NVIDIA driver | `580.95.05` |
| Transformer Engine | `2.19.0` |
| GPU | `NVIDIA B200`, compute capability `10.0` |

The validator reported `max_abs_error=0.000152587890625`; the loaded weight also passed the separate TE-dequantization comparison and the GEMM output shape, finiteness, and tolerance checks. In this container, the PyTorch wheel supplied cuDNN alongside a system cuDNN installation, so the validator subprocess was directed to the PyTorch-wheel libraries through `CUDNN_HOME`, `CUDNN_PATH`, and `LD_LIBRARY_PATH`. See the tool README for the tested paths. The first attempt exposed a mixed-library symbol error; the successful run used the consistent PyTorch-wheel cuDNN path.

## Alternatives and consequences

- Pickling `NVFP4Tensor` was rejected because it couples serialization to Python object internals.
- Calling the ADR 0011 native format a TE checkpoint was rejected because runtime field names, padding, and metadata differ.
- Manufacturing columnwise weight data was rejected for this single rowwise TN operation because it could change quantization semantics.
- Adding TE/CUDA dependencies to Cargo or ordinary CPU CI was rejected; hardware validation is an explicit optional Python operation.
- A whole-model loader was excluded because architecture, activations, and inference exceed this single-matrix boundary.

The artifact is explicit, deterministic, and CPU-testable independently of GPU libraries. Existing native formats and the INT8 CLI remain unchanged. Optional Python dependencies stay outside Cargo; no new Rust dependency or general NVFP4 CLI command is introduced. Whole-model loading, inference, other GEMM shapes/layouts, and other TE releases remain outside this validated boundary.
