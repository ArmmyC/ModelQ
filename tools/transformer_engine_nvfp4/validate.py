"""CPU validation for the ModelQ Transformer Engine NVFP4 SafeTensors profile."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pathlib
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
    except (OSError, SafetensorError) as error:
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
    except (OSError, SafetensorError) as error:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("cpu", help="validate SafeTensors fields without CUDA")
    cpu.add_argument("artifact", type=pathlib.Path)
    cpu.add_argument("reference", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
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
