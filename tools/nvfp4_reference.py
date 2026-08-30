#!/usr/bin/env python3
"""Capture and validate an opt-in Transformer Engine NVFP4 reference fixture.

The repository deliberately does not depend on PyTorch or Transformer Engine.
Those imports therefore live inside :func:`capture_fixture`, which is only
used on a CUDA/Blackwell machine prepared for the optional capture step.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.metadata
import json
import math
import os
import pathlib
import struct
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "modelq.nvfp4.reference.v1"
BLOCK_SIZE = 16
FP4_MAX = 6.0
FP8_MAX = 448.0
SOURCE_SHAPE = [2, 32]
SOURCE_VALUES = [
    -6.0,
    -5.0,
    -4.0,
    -3.5,
    -3.0,
    -2.5,
    -2.0,
    -1.75,
    -1.5,
    -1.25,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    -0.125,
    0.0,
    0.0,
    0.125,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.5,
    3.5,
    4.0,
    5.0,
    6.0,
    2.25,
    -2.25,
    0.0,
    0.1,
    -0.2,
    0.3,
    -0.4,
    0.6,
    -0.8,
    1.1,
    -1.4,
    1.8,
    -2.2,
    2.7,
    -3.2,
    3.8,
    -4.4,
    5.2,
    -5.8,
    0.0,
    -0.125,
    0.25,
    -0.5,
    0.75,
    -1.0,
    1.25,
    -1.5,
    1.75,
    -2.5,
    2.75,
    -3.5,
    3.75,
    -4.5,
    4.75,
]

EXPECTED_RECIPE = {
    "quantization_dim": "1x16",
    "deterministic": True,
    "disable_2d_quantization": True,
    "disable_rht": True,
    "disable_stochastic_rounding": True,
    "nvfp4_4over6": "none",
}
EXPECTED_REPRESENTATION = {
    "fp4": "E2M1",
    "block_scale": "E4M3",
    "block_size": BLOCK_SIZE,
    "global_scale": "F32_decode",
    "packing": "low_nibble_first",
}
FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


class FixtureError(ValueError):
    """Raised when a fixture is malformed or capture prerequisites are absent."""


def _f32(value: float) -> float:
    """Round a Python number to the IEEE-754 binary32 value used by ModelQ."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", _f32(value)))[0]


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _f32_mul(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _decode_e4m3(bits: int) -> float:
    """Decode ModelQ's finite E4M3 scale format for fixture sanity checks."""

    negative = bits & 0x80 != 0
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    if exponent == 0:
        magnitude = _f32(_f32(mantissa) * _f32(2.0**-9))
    else:
        significand = _f32(_f32(1.0) + _f32(_f32(mantissa) / _f32(8.0)))
        magnitude = _f32(significand * _f32(2.0 ** (exponent - 7)))
    return -magnitude if negative else magnitude


def _dequantized_bits(
    packed: Sequence[int], block_scales: Sequence[int], global_scale: float, elements: int
) -> list[int]:
    """Reconstruct ModelQ-native values with explicit binary32 roundings."""

    output: list[int] = []
    for index in range(elements):
        packed_byte = packed[index // 2]
        code = packed_byte & 0x0F if index % 2 == 0 else packed_byte >> 4
        element = FP4_VALUES[code]
        block_scale = _decode_e4m3(block_scales[index // BLOCK_SIZE])
        reconstructed = _f32_mul(_f32_mul(element, block_scale), global_scale)
        output.append(_f32_bits(reconstructed))
    return output


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureError(f"{name} must be an object")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FixtureError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        suffix = f" in [0, {maximum}]" if maximum is not None else ""
        raise FixtureError(f"{name} must be an integer{suffix}")
    return value


def _require_bits_list(
    tensor: Mapping[str, Any], field: str, expected_length: int, *, finite: bool = True
) -> list[int]:
    value = tensor.get(field)
    if not isinstance(value, list) or len(value) != expected_length:
        actual = len(value) if isinstance(value, list) else "not a list"
        raise FixtureError(f"tensor.{field} must contain {expected_length} integers (got {actual})")
    bits = [_require_int(item, f"tensor.{field}[{index}]", maximum=0xFFFFFFFF) for index, item in enumerate(value)]
    if finite:
        for index, item in enumerate(bits):
            if not math.isfinite(_f32_from_bits(item)):
                raise FixtureError(f"tensor.{field}[{index}] must encode a finite F32 value")
    return bits


def _require_byte_list(
    tensor: Mapping[str, Any], field: str, expected_length: int
) -> list[int]:
    value = tensor.get(field)
    if not isinstance(value, list) or len(value) != expected_length:
        actual = len(value) if isinstance(value, list) else "not a list"
        raise FixtureError(f"tensor.{field} must contain {expected_length} bytes (got {actual})")
    return [
        _require_int(item, f"tensor.{field}[{index}]", maximum=0xFF)
        for index, item in enumerate(value)
    ]


def validate_fixture_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a copy of a v1 fixture document.

    Validation is intentionally independent of PyTorch so it can run on every
    developer machine and in ordinary CI without CUDA or Transformer Engine.
    """

    root = _require_mapping(document, "fixture")
    if root.get("schema") != SCHEMA:
        raise FixtureError(f"schema must be {SCHEMA!r}")

    producer = _require_mapping(root.get("producer"), "producer")
    if producer.get("name") != "transformer-engine":
        raise FixtureError("producer.name must be 'transformer-engine'")
    for field in ("version", "commit", "pytorch", "cuda", "device"):
        _require_string(producer.get(field), f"producer.{field}")
    capability = _require_string(producer.get("compute_capability"), "producer.compute_capability")
    try:
        major_text, minor_text = capability.split(".", maxsplit=1)
        major = int(major_text)
        minor = int(minor_text)
    except (ValueError, TypeError) as error:
        raise FixtureError("producer.compute_capability must be formatted as MAJOR.MINOR") from error
    if major < 10 or minor < 0:
        raise FixtureError("producer.compute_capability must identify a Blackwell-class GPU (>= 10.0)")

    recipe = _require_mapping(root.get("recipe"), "recipe")
    for field, expected in EXPECTED_RECIPE.items():
        if recipe.get(field) != expected:
            raise FixtureError(f"recipe.{field} must be {expected!r}")

    representation = _require_mapping(root.get("representation"), "representation")
    for field, expected in EXPECTED_REPRESENTATION.items():
        if representation.get(field) != expected:
            raise FixtureError(f"representation.{field} must be {expected!r}")

    tensor = _require_mapping(root.get("tensor"), "tensor")
    _require_string(tensor.get("name"), "tensor.name")
    if tensor.get("source_dtype") != "F32":
        raise FixtureError("tensor.source_dtype must be F32 for reference schema v1")

    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        or shape[-1] % BLOCK_SIZE != 0
    ):
        raise FixtureError(
            f"tensor.shape must contain positive dimensions with a final dimension divisible by {BLOCK_SIZE}"
        )
    elements = math.prod(shape)
    source_bits = _require_bits_list(tensor, "source_values_f32_bits", elements)
    packed = _require_byte_list(tensor, "packed_u8", (elements + 1) // 2)
    block_scales = _require_byte_list(tensor, "block_scale_u8", (elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    global_bits = _require_int(
        tensor.get("global_scale_f32_bits"), "tensor.global_scale_f32_bits", maximum=0xFFFFFFFF
    )
    global_scale = _f32_from_bits(global_bits)
    if not math.isfinite(global_scale) or global_scale <= 0.0:
        raise FixtureError("tensor.global_scale_f32_bits must encode a positive finite F32")
    expected_bits = _require_bits_list(tensor, "expected_dequant_f32_bits", elements)

    if elements % 2 and packed[-1] & 0xF0:
        raise FixtureError("tensor.packed_u8 has non-zero padding in its final high nibble")
    for block, bits in enumerate(block_scales):
        if bits == 0:
            continue
        decoded = _decode_e4m3(bits)
        if bits & 0x80 or not math.isfinite(decoded) or decoded <= 0.0:
            raise FixtureError(f"tensor.block_scale_u8[{block}] is not a positive E4M3 value")
    for index in range(elements):
        code = packed[index // 2] & 0x0F if index % 2 == 0 else packed[index // 2] >> 4
        if block_scales[index // BLOCK_SIZE] == 0 and code & 0x07:
            raise FixtureError(
                f"tensor.block_scale_u8[{index // BLOCK_SIZE}] is zero for non-zero FP4 value {index}"
            )

    reconstructed_bits = _dequantized_bits(packed, block_scales, global_scale, elements)
    if reconstructed_bits != expected_bits:
        raise FixtureError("tensor.expected_dequant_f32_bits does not match the packed payload")
    if len(source_bits) != elements:
        raise FixtureError("tensor.source_values_f32_bits length does not match tensor.shape")

    return copy.deepcopy(dict(root))


def validate_fixture(path: pathlib.Path) -> dict[str, Any]:
    """Load and validate a fixture JSON file."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureError(f"could not read fixture {path}: {error}") from error
    return validate_fixture_document(document)


def _tensor_bytes(tensor: Any, torch: Any) -> list[int]:
    """Return a contiguous tensor's raw bytes without requiring NumPy."""

    raw = tensor.detach().to(device="cpu").contiguous().view(dtype=torch.uint8)
    return [int(value) for value in raw.flatten().tolist()]


def _discover_te_version() -> str:
    for distribution in (
        "transformer-engine",
        "transformer_engine",
        "transformer-engine-cu12",
        "transformer-engine-cu13",
    ):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        module = importlib.import_module("transformer_engine")
    except ImportError as error:
        raise FixtureError("Transformer Engine is not installed") from error
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version.strip():
        raise FixtureError("could not determine the Transformer Engine version")
    return version


def _discover_te_commit(module: Any, explicit_commit: str | None) -> str:
    commit = explicit_commit or os.environ.get("MODELQ_NVFP4_TE_COMMIT")
    if not commit:
        for attribute in ("__git_version__", "git_version", "__commit__"):
            candidate = getattr(module, attribute, None)
            if isinstance(candidate, str) and candidate.strip():
                commit = candidate
                break
    if not commit or not commit.strip():
        raise FixtureError(
            "a Transformer Engine source commit is required; pass --te-commit or set MODELQ_NVFP4_TE_COMMIT"
        )
    return commit


def capture_fixture(output_path: pathlib.Path, *, te_commit: str | None = None) -> dict[str, Any]:
    """Capture the deterministic 1x16 Transformer Engine reference tensor.

    Capture is intentionally guarded to CUDA devices with a Blackwell-class
    compute capability.  The resulting JSON is portable and needs no runtime
    Transformer Engine dependency to validate or consume in the Rust test.
    """

    try:
        torch = importlib.import_module("torch")
        te_module = importlib.import_module("transformer_engine")
        reference_utils = importlib.import_module(
            "transformer_engine.pytorch.custom_recipes.reference_utils"
        )
        reference_module = importlib.import_module(
            "transformer_engine.pytorch.custom_recipes.reference_nvfp4"
        )
        quantizer_type = reference_module.NVFP4QuantizerRef
    except (ImportError, AttributeError) as error:
        raise FixtureError(
            "capture requires PyTorch and Transformer Engine's NVFP4 reference implementation"
        ) from error

    if not torch.cuda.is_available():
        raise FixtureError("capture requires a CUDA-enabled PyTorch installation")
    capability = tuple(torch.cuda.get_device_capability())
    if capability < (10, 0):
        raise FixtureError(
            f"capture requires a Blackwell-class GPU (compute capability >= 10.0), found {capability[0]}.{capability[1]}"
        )

    device = torch.device("cuda")
    source = torch.tensor(SOURCE_VALUES, dtype=torch.float32, device=device).reshape(SOURCE_SHAPE)
    quantizer = quantizer_type(
        dtype=reference_utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=False,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, BLOCK_SIZE),
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=False,
        with_rht=False,
        with_random_sign_mask=False,
    )
    with torch.no_grad():
        result = quantizer.quantize(source)

    if result.data is None or result.scale is None or result.global_amax_row is None:
        raise FixtureError("Transformer Engine did not return rowwise NVFP4 data and scales")
    packed = _tensor_bytes(result.data, torch)
    block_scales = _tensor_bytes(result.scale, torch)
    global_amax = _f32(float(result.global_amax_row.reshape(-1)[0].item()))
    global_scale = _f32(global_amax / _f32(FP4_MAX * FP8_MAX)) if global_amax else 1.0
    expected = _dequantized_bits(packed, block_scales, global_scale, len(SOURCE_VALUES))

    transformer_engine_version = _discover_te_version()
    commit = _discover_te_commit(te_module, te_commit)
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "producer": {
            "name": "transformer-engine",
            "version": transformer_engine_version,
            "commit": commit,
            "pytorch": str(torch.__version__),
            "cuda": str(torch.version.cuda or "unknown"),
            "device": str(torch.cuda.get_device_name()),
            "compute_capability": f"{capability[0]}.{capability[1]}",
        },
        "recipe": copy.deepcopy(EXPECTED_RECIPE),
        "representation": copy.deepcopy(EXPECTED_REPRESENTATION),
        "tensor": {
            "name": "modelq.nvfp4.reference",
            "source_dtype": "F32",
            "shape": SOURCE_SHAPE,
            "source_values_f32_bits": [_f32_bits(value) for value in SOURCE_VALUES],
            "packed_u8": packed,
            "block_scale_u8": block_scales,
            "global_scale_f32_bits": _f32_bits(global_scale),
            "expected_dequant_f32_bits": expected,
        },
    }
    validated = validate_fixture_document(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser("validate", help="validate an existing fixture")
    validate_parser.add_argument("path", type=pathlib.Path)

    capture_parser = commands.add_parser(
        "capture", help="capture a fixture on a supported CUDA/Blackwell host"
    )
    capture_parser.add_argument("path", type=pathlib.Path)
    capture_parser.add_argument(
        "--te-commit",
        help="Transformer Engine source commit (or use MODELQ_NVFP4_TE_COMMIT)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate":
            document = validate_fixture(arguments.path)
            tensor = document["tensor"]
            print(
                f"validated {arguments.path}: producer={document['producer']['version']} "
                f"tensor={tensor['name']} shape={tensor['shape']}"
            )
        else:
            document = capture_fixture(arguments.path, te_commit=arguments.te_commit)
            tensor = document["tensor"]
            print(
                f"captured {arguments.path}: producer={document['producer']['version']} "
                f"tensor={tensor['name']} shape={tensor['shape']}"
            )
    except FixtureError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
