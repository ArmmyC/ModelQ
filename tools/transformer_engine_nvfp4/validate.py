"""Validate the ModelQ Transformer Engine NVFP4 container and optional GEMM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open


FORMAT = "transformer-engine-nvfp4-safetensors-v1"
PROFILE = "transformer-engine.nvfp4.rowwise.1x16.v1"
REFERENCE_SCHEMA = "transformer-engine-nvfp4-reference-v1"
REFERENCE_TENSOR = "weight.dequantized_reference"


class ValidationError(ValueError):
    """An artifact does not satisfy the pinned NVFP4 container contract."""


@dataclass(frozen=True)
class ValidatedArtifact:
    manifest: dict[str, Any]
    tensors: dict[str, np.ndarray]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _check_tensor(
    tensors: dict[str, np.ndarray], name: str, dtype: np.dtype, shape: tuple[int, ...]
) -> np.ndarray:
    _require(name in tensors, f"missing tensor {name}")
    array = tensors[name]
    _require(array.dtype == dtype, f"{name}: expected dtype {dtype}, got {array.dtype}")
    _require(array.shape == shape, f"{name}: expected shape {shape}, got {array.shape}")
    return array


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def validate_manifest(
    metadata: dict[str, str], manifest: Any, tensors: dict[str, np.ndarray]
) -> None:
    """Check schema fields, tensor identities, physical shapes, and padding."""
    _require(metadata.get("modelq.format") == FORMAT, f"modelq.format must be {FORMAT}")
    _require(isinstance(manifest, dict), "modelq.manifest must be a JSON object")
    _require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
             "schema_version must be 1")
    _require(manifest.get("profile_id") == PROFILE, f"profile_id must be {PROFILE}")
    _require(manifest.get("runtime") == {"name": "transformer_engine", "version": "2.19.0"},
             "runtime must be transformer_engine 2.19.0")

    name = manifest.get("tensor_name")
    _require(isinstance(name, str) and bool(name.strip()) and name != "__metadata__",
             "tensor_name must be nonempty and not reserved")
    shape = manifest.get("logical_shape")
    _require(isinstance(shape, list) and len(shape) == 2 and
             all(type(size) is int and size > 0 for size in shape),
             "logical_shape must have two positive integer dimensions")
    rows, columns = shape
    _require(columns % 16 == 0, "logical_shape K must be divisible by 16")

    fields = {
        "rowwise_data": f"{name}.rowwise_data",
        "rowwise_scale_inv": f"{name}.rowwise_scale_inv",
        "amax_rowwise": f"{name}.amax_rowwise",
    }
    _require(manifest.get("fields") == fields, "fields must match tensor_name and rowwise schema")
    _require(manifest.get("quantization") == {
        "data_format": "E2M1",
        "block_scale_format": "E4M3",
        "block_size": 16,
        "scaling": "rowwise_1x16_tensor_global",
    }, "quantization must describe E2M1 rowwise 1x16 scaling")
    _require(manifest.get("scale_storage") == {
        "padding": [128, 4], "gemm_swizzled": False,
    }, "scale_storage must specify unswizzled [128, 4] padding")
    denominator = manifest.get("global_scale_denominator")
    _require(type(denominator) in (int, float) and denominator == 2688.0,
             "global_scale_denominator must be 2688.0")

    expected_names = set(fields.values())
    actual_names = set(tensors)
    missing = expected_names - actual_names
    unexpected = actual_names - expected_names
    _require(not missing, f"missing tensor(s): {', '.join(sorted(missing))}")
    _require(not unexpected, f"unexpected tensor(s): {', '.join(sorted(unexpected))}")
    _require(len(tensors) == 3, "expected exactly three rowwise tensors")

    _check_tensor(tensors, fields["rowwise_data"], np.dtype("uint8"), (rows, columns // 2))
    block_columns = columns // 16
    scales = _check_tensor(
        tensors, fields["rowwise_scale_inv"], np.dtype("uint8"),
        (_round_up(rows, 128), _round_up(block_columns, 4)),
    )
    _require(not np.any(scales[rows:, :]), "rowwise_scale_inv row padding must be zero")
    _require(not np.any(scales[:rows, block_columns:]),
             "rowwise_scale_inv column padding must be zero")
    amax = _check_tensor(tensors, fields["amax_rowwise"], np.dtype("float32"), (1,))
    _require(bool(np.isfinite(amax).all()) and bool((amax >= 0).all()),
             "amax_rowwise must be finite and nonnegative")


def validate_container(path: pathlib.Path) -> ValidatedArtifact:
    """Read and validate one rowwise NVFP4 runtime artifact on CPU."""
    try:
        with safe_open(str(path), framework="np") as reader:
            metadata = reader.metadata() or {}
            tensors = {name: reader.get_tensor(name) for name in reader.keys()}
    except (OSError, SafetensorError, TypeError) as error:
        raise ValidationError(f"{path}: cannot read SafeTensors artifact: {error}") from error
    _require("modelq.manifest" in metadata, "missing modelq.manifest metadata")
    try:
        manifest = json.loads(metadata["modelq.manifest"])
    except (TypeError, ValueError) as error:
        raise ValidationError(f"modelq.manifest is invalid JSON: {error}") from error
    validate_manifest(metadata, manifest, tensors)
    return ValidatedArtifact(manifest=manifest, tensors=tensors)


def validate_reference(path: pathlib.Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Read the separate finite F32 reference tensor on CPU."""
    try:
        with safe_open(str(path), framework="np") as reader:
            metadata = reader.metadata() or {}
            tensors = {name: reader.get_tensor(name) for name in reader.keys()}
    except (OSError, SafetensorError, TypeError) as error:
        raise ValidationError(f"{path}: cannot read reference SafeTensors: {error}") from error
    _require(metadata.get("modelq.reference_schema") == REFERENCE_SCHEMA,
             f"modelq.reference_schema must be {REFERENCE_SCHEMA}")
    _require(set(tensors) == {REFERENCE_TENSOR},
             f"reference must contain exactly one tensor named {REFERENCE_TENSOR}")
    reference = _check_tensor(tensors, REFERENCE_TENSOR, np.dtype("float32"), expected_shape)
    _require(bool(np.isfinite(reference).all()), "reference values must be finite")
    return reference


def validate_cpu_fixture(
    artifact_path: pathlib.Path, reference_path: pathlib.Path
) -> tuple[ValidatedArtifact, np.ndarray]:
    """Validate the runtime artifact and its separately stored reference."""
    artifact = validate_container(artifact_path)
    shape = tuple(artifact.manifest["logical_shape"])
    reference = validate_reference(reference_path, shape)
    return artifact, reference


def validate_runtime_preflight(
    system: str,
    machine: str,
    te_version: str,
    cuda_available: bool,
    cuda_version: str | None,
    cudnn_version: int | None,
    capability: tuple[int, int] | None,
) -> None:
    """Reject any environment outside the pinned Blackwell proof boundary."""
    _validate_runtime_platform(system, machine)
    _require(te_version == "2.19.0", "runtime proof requires Transformer Engine 2.19.0")
    _require(cuda_available, "runtime proof requires an available CUDA GPU")
    try:
        cuda_numbers = tuple(int(part) for part in cuda_version.split(".")[:2]) if cuda_version else ()
    except ValueError:
        cuda_numbers = ()
    _require(len(cuda_numbers) == 2 and cuda_numbers >= (12, 8),
             "runtime proof requires CUDA 12.8 or newer")
    _require(cudnn_version is not None and cudnn_version >= 90300,
             "runtime proof requires cuDNN 9.3 or newer (90300)")
    _require(capability is not None and capability[0] >= 10,
             "runtime proof requires compute capability 10.0 or newer")


def _validate_runtime_platform(system: str, machine: str) -> None:
    _require(system == "Linux", "runtime proof requires Linux")
    _require(machine.lower() in ("x86_64", "amd64"), "runtime proof requires x86_64")


def run_blackwell_gemm(
    artifact_path: pathlib.Path, reference_path: pathlib.Path
) -> dict[str, Any]:
    """Run the explicit TE 2.19.0 TN GEMM against the Rust F32 oracle."""
    artifact, reference_values = validate_cpu_fixture(artifact_path, reference_path)
    manifest = artifact.manifest
    _require(manifest["logical_shape"] == [64, 64],
             "runtime proof requires logical_shape [64, 64]")

    system, machine = platform.system(), platform.machine()
    _validate_runtime_platform(system, machine)
    try:
        te_version = importlib.metadata.version("transformer-engine")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValidationError("runtime proof requires Transformer Engine 2.19.0 package") from error
    _require(te_version == "2.19.0", "runtime proof requires Transformer Engine 2.19.0")

    try:
        import torch
    except ImportError as error:
        raise ValidationError(f"runtime proof requires CUDA-enabled PyTorch: {error}") from error
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    capability = torch.cuda.get_device_capability() if cuda_available else None
    validate_runtime_preflight(
        system, machine, te_version, cuda_available, cuda_version, cudnn_version, capability
    )
    device = torch.device("cuda", torch.cuda.current_device())
    gpu_name = torch.cuda.get_device_name()
    try:
        driver_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValidationError(f"cannot query NVIDIA driver with nvidia-smi: {error}") from error
    driver = driver_result.stdout.strip().splitlines()
    _require(bool(driver) and bool(driver[0].strip()),
             "nvidia-smi returned no NVIDIA driver version")

    try:
        from transformer_engine.pytorch.constants import DType
        from transformer_engine.pytorch import NVFP4Quantizer, NVFP4Tensor
        from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

        fields = manifest["fields"]
        rowwise_data = torch.as_tensor(artifact.tensors[fields["rowwise_data"]], device=device)
        rowwise_scale_inv = torch.as_tensor(
            artifact.tensors[fields["rowwise_scale_inv"]], device=device
        )
        amax_rowwise = torch.as_tensor(artifact.tensors[fields["amax_rowwise"]], device=device)
        quantizer = NVFP4Quantizer(
            fp4_dtype=DType.kFloat4E2M1,
            rowwise=True,
            columnwise=False,
            with_2d_quantization=False,
            with_rht=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
            nvfp4_use_4over6=False,
        )
        weight = NVFP4Tensor(
            shape=(64, 64),
            dtype=torch.float32,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=None,
            columnwise_scale_inv=None,
            amax_rowwise=amax_rowwise,
            amax_columnwise=None,
            fp4_dtype=DType.kFloat4E2M1,
            quantizer=quantizer,
            with_gemm_swizzled_scales=False,
            row_scaled_nvfp4=False,
            nvfp4_use_4over6=False,
            device=device,
        )
        with torch.no_grad():
            reference_weight = torch.as_tensor(reference_values, dtype=torch.float32, device=device)
            indices = torch.arange(4096, dtype=torch.int32, device=device)
            rhs_values = (((indices * 37) % 251) - 125).to(torch.float32).div_(32.0).reshape(64, 64)
            torch.testing.assert_close(
                weight.dequantize(dtype=torch.float32), reference_weight, rtol=1e-5, atol=1e-5
            )
            rhs_quantized = quantizer.quantize(rhs_values)
            rhs_dequantized = rhs_quantized.dequantize(dtype=torch.float32)
            out, _, _, _ = general_gemm(
                weight, rhs_quantized, out_dtype=torch.float32, layout="TN"
            )
            expected = reference_weight.T @ rhs_dequantized
            _require(tuple(out.shape) == (64, 64),
                     f"TN GEMM output shape must be (64, 64), got {tuple(out.shape)}")
            _require(bool(torch.isfinite(out).all().item()), "TN GEMM output must be finite")
            torch.testing.assert_close(out, expected, rtol=0.125, atol=0.0675)
            max_abs_error = (out - expected).abs().max().item()
    except Exception as error:
        if isinstance(error, ValidationError):
            raise
        raise ValidationError(f"Transformer Engine NVFP4 TN GEMM failed: {error}") from error

    return {
        "max_abs_error": max_abs_error,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": cuda_version,
        "cudnn": cudnn_version,
        "driver": driver[0].strip(),
        "transformer_engine": te_version,
        "gpu": gpu_name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("cpu", help="validate SafeTensors fields without CUDA")
    cpu.add_argument("artifact", type=pathlib.Path)
    cpu.add_argument("reference", type=pathlib.Path)
    runtime = commands.add_parser("runtime", help="run one TE 2.19.0 Blackwell TN GEMM")
    runtime.add_argument("artifact", type=pathlib.Path)
    runtime.add_argument("reference", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "runtime":
            report = run_blackwell_gemm(args.artifact, args.reference)
            for key, value in report.items():
                print(f"{key}={value}")
            return 0
        artifact, _ = validate_cpu_fixture(args.artifact, args.reference)
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    manifest = artifact.manifest
    print(f"schema_version={manifest['schema_version']} profile_id={manifest['profile_id']}")
    print(f"logical_shape={manifest['logical_shape']}")
    print("CPU/container check only; Transformer Engine and CUDA were not run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
