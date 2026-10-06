# Transformer Engine NVFP4 container validator

Task 28 provides a SafeTensors artifact for one rowwise NVFP4 matrix and a bridge pinned to Transformer Engine 2.19.0 for one TN GEMM. The single-matrix runtime check passed on an NVIDIA B200 on 2026-10-06, establishing Level 3 runtime compatibility and Level 4 hardware validation only for this schema, TE release, and operation. The reported GEMM maximum absolute error was `0.000152587890625`. This does not establish whole-model compatibility or inference support. See the [recorded environment and proof details in ADR 0013](../../docs/adr/0013-transformer-engine-nvfp4-runtime-container.md#recorded-blackwell-proof).

## CPU container check

Run from the repository root in a Python environment with NumPy and SafeTensors. Fixture generation requires the repository's supported Rust toolchain. The following paths are Linux/macOS examples; on Windows substitute existing temporary-directory paths. Use two distinct output paths that do not already exist. The fixture generator refuses existing destinations.

```bash
cargo run -p modelq-io --example transformer_engine_nvfp4_fixture -- /tmp/te-runtime.safetensors /tmp/te-reference.safetensors
python -m pip install 'numpy>=1.24,<3' 'safetensors>=0.4,<1'
python tools/transformer_engine_nvfp4/validate.py cpu /tmp/te-runtime.safetensors /tmp/te-reference.safetensors
```

The runtime file contains `weight.rowwise_data` (U8 `[64,32]`), `weight.rowwise_scale_inv` (U8 `[128,4]`), and `weight.amax_rowwise` (F32 `[1]`), plus the versioned manifest. The second file holds only the separate F32 `[64,64]` ModelQ-dequantized reference and its schema metadata. Reference values are not part of the runtime artifact.

The CPU command checks metadata, exact field names, dtypes, shapes, zero padding, amax, and the separate reference. On success it prints the profile, logical shape, and `CPU/container check only; Transformer Engine and CUDA were not run.` It does not import PyTorch or TE. A passing CPU command is not a runtime-compatibility claim.

## Explicit Blackwell runtime check

Prepare a separate Linux x86_64 Python environment with CUDA 12.8 or newer, a compatible NVIDIA driver, cuDNN 9.3 or newer, and a Blackwell-or-newer NVIDIA GPU (compute capability 10.0 or newer). Install a compatible CUDA-enabled PyTorch build for that environment before installing the optional requirements. The [pinned TE installation guide](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/docs/installation.rst) describes the upstream prerequisites. `nvidia-smi` must be available for driver reporting.

The recorded Modal run used `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`, Python 3.12.1, and PyTorch 2.9.0+cu128. That image had both system cuDNN and the cuDNN installed with the PyTorch wheel. To keep TE and its dependent cuDNN libraries on the wheel version, the validator subprocess was launched with `CUDNN_HOME` and `CUDNN_PATH` set to `/usr/local/lib/python3.12/site-packages/nvidia/cudnn`, and that directory's `lib` path prefixed to `LD_LIBRARY_PATH`. If reproducing this specific container setup, use equivalent paths for the installed Python environment; do not mix system and wheel cuDNN libraries.

From the repository root in that environment, install the requirements file, which selects the TE PyTorch extra pinned to 2.19.0, then use the generated fixture pair:

```bash
python -m pip install -r tools/transformer_engine_nvfp4/requirements.txt
python tools/transformer_engine_nvfp4/validate.py runtime /tmp/te-runtime.safetensors /tmp/te-reference.safetensors
```

The runtime command validates the files before CUDA work and rejects unsupported platforms, TE versions, CUDA/cuDNN versions, or GPUs. It loads the exported rowwise weight unchanged, checks TE dequantization against the Rust reference (`rtol=1e-5`, `atol=1e-5`), and TE-quantizes only a deterministic synthetic `[64,64]` second operand.

`general_gemm(weight, second_operand, layout="TN")` follows TE 2.19's convention: `second_operand @ weight.T`. The F32 GEMM result is compared with `TE-dequantized second operand @ ModelQ-dequantized exported weight.T` using `rtol=0.125`, `atol=0.0675`; output must be finite and `[64,64]`. TE handles runtime layout preparation without rewriting the artifact.

The successful runtime command prints maximum absolute error and Python, PyTorch, CUDA runtime, cuDNN, driver, TE, GPU, and compute-capability versions. The recorded NVIDIA B200 run reported `max_abs_error=0.000152587890625` and passed the artifact dequantization comparison and single TN GEMM checks. Failures return nonzero; an unavailable GPU is not a pass. This validation does not cover other shapes, layouts, TE versions, hardware, or complete models.
