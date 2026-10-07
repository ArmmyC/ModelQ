"""CPU container check for the multi-matrix Transformer Engine NVFP4 container.

Validates schema version 2 files (or a directory of shards with an index)
written by `modelq quantize --format nvfp4-te`, entirely on the CPU: it imports
neither Transformer Engine nor PyTorch and creates no CUDA context.  It checks
the manifest, every field's dtype and shape, zero scale padding, scale and amax
validity, and (optionally) that decoding every matrix matches the separate F32
reference from `te_multi_matrix_fixture`.

Passing this check shows the container is well formed and decodes to the
reference.  It does not show Transformer Engine accepts it: that needs the
explicit Blackwell runtime run (Task 41), which this tool does not implement.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from safetensors import safe_open

import validate  # reuses ValidationError and helpers from the single-matrix tool

FORMAT = "transformer-engine-nvfp4-safetensors-v2"
PROFILE = validate.PROFILE
RUNTIME = {"name": "transformer_engine", "version": "2.19.0"}
REFERENCE_SCHEMA = "transformer-engine-nvfp4-reference-v2"
BLOCK_SIZE = 16
GLOBAL_SCALE_DENOMINATOR = 2688.0
ROW_ALIGNMENT = 128
COLUMN_ALIGNMENT = 4
REFERENCE_SUFFIX = ".dequantized_reference"
INDEX_NAME = "model.safetensors.index.json"

ValidationError = validate.ValidationError
_require = validate._require
_round_up = validate._round_up

# E2M1 magnitudes by 3-bit code; bit 3 is the sign.
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _build_e4m3_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.float32)
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
        exponent = (bits >> 3) & 0x0F
        mantissa = bits & 0x07
        if exponent == 0x0F and mantissa == 0x07:
            table[bits] = np.nan
        elif exponent == 0:
            table[bits] = sign * mantissa * 2.0**-9
        else:
            table[bits] = sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return table


E4M3 = _build_e4m3_table()


@dataclass
class MatrixEntry:
    """One validated Transformer Engine matrix."""

    name: str
    logical_shape: tuple[int, int]
    original_dtype: str
    file: pathlib.Path
    rowwise_data: str
    rowwise_scale_inv: str
    amax_rowwise: str


@dataclass
class ValidatedContainer:
    """Everything the CPU check established about a container."""

    files: list[pathlib.Path] = field(default_factory=list)
    matrices: dict[str, MatrixEntry] = field(default_factory=dict)
    preserved: dict[str, pathlib.Path] = field(default_factory=dict)


def container_files(path: pathlib.Path) -> list[pathlib.Path]:
    """Returns the SafeTensors files of a single-file or sharded container."""
    if path.is_file():
        return [path]
    _require(path.is_dir(), f"{path} is neither a file nor a directory")
    index_path = path / INDEX_NAME
    _require(index_path.is_file(), f"{path} has no {INDEX_NAME}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as error:
        raise ValidationError(f"{index_path} is not a valid index: {error}") from error
    _require(bool(shards), f"{index_path} lists no shards")
    files = []
    for shard in shards:
        _require(
            isinstance(shard, str) and shard == pathlib.Path(shard).name,
            f"shard reference {shard!r} is not a plain file name",
        )
        shard_path = path / shard
        _require(shard_path.is_file(), f"shard {shard_path} is missing")
        files.append(shard_path)
    return files


def _tensor_metadata(handle, name: str) -> tuple[str, tuple[int, ...]]:
    tensor_slice = handle.get_slice(name)
    return tensor_slice.get_dtype(), tuple(tensor_slice.get_shape())


def _check_manifest_header(manifest: dict, context: str) -> None:
    _require(isinstance(manifest, dict), f"{context}: manifest must be an object")
    _require(manifest.get("schema_version") == 2, f"{context}: schema_version must be 2")
    _require(manifest.get("profile_id") == PROFILE, f"{context}: profile_id mismatch")
    _require(manifest.get("runtime") == RUNTIME, f"{context}: runtime must be {RUNTIME}")
    _require(
        manifest.get("quantization")
        == {
            "data_format": "E2M1",
            "block_scale_format": "E4M3",
            "block_size": BLOCK_SIZE,
            "scaling": "rowwise_1x16_tensor_global",
        },
        f"{context}: quantization block does not match the profile",
    )
    _require(
        manifest.get("scale_storage")
        == {"padding": [ROW_ALIGNMENT, COLUMN_ALIGNMENT], "gemm_swizzled": False},
        f"{context}: scale_storage does not match the profile",
    )
    _require(
        manifest.get("global_scale_denominator") == GLOBAL_SCALE_DENOMINATOR,
        f"{context}: global_scale_denominator must be {GLOBAL_SCALE_DENOMINATOR}",
    )
    _require(isinstance(manifest.get("tensors"), dict), f"{context}: tensors must be an object")


def _validate_scale_payload(
    context: str, rows: int, columns: int, scale_inv: np.ndarray, data: np.ndarray, amax: float
) -> None:
    """Checks zero padding, valid scales, zero-scale blocks, and the amax."""
    blocks = columns // BLOCK_SIZE
    padded_rows, padded_blocks = scale_inv.shape
    _require(np.isfinite(amax) and amax >= 0.0, f"{context}: amax {amax} must be finite and non-negative")
    _require(
        not scale_inv[rows:].any(), f"{context}: scale padding rows are not zero"
    )
    _require(
        not scale_inv[:rows, blocks:].any(), f"{context}: scale padding columns are not zero"
    )
    used = scale_inv[:rows, :blocks]
    decoded = E4M3[used]
    nonzero = used != 0
    _require(
        bool(np.all(np.isfinite(decoded[nonzero]) & (decoded[nonzero] > 0.0))),
        f"{context}: a scale is not a positive finite E4M3 value",
    )
    codes = _unpack_codes(data, rows, columns)
    zero_blocks = (~nonzero)[:, :, None] & (
        (codes.reshape(rows, blocks, BLOCK_SIZE) & 0x07) != 0
    )
    _require(not zero_blocks.any(), f"{context}: a zero-scale block holds nonzero values")
    _require(
        amax != 0.0 or not nonzero.any(),
        f"{context}: amax is zero but the tensor has nonzero scales",
    )


def _unpack_codes(data: np.ndarray, rows: int, columns: int) -> np.ndarray:
    """Returns the 4-bit codes of a `[rows, columns/2]` U8 matrix, low nibble first."""
    flat = data.reshape(-1)
    codes = np.empty(flat.size * 2, dtype=np.uint8)
    codes[0::2] = flat & 0x0F
    codes[1::2] = flat >> 4
    return codes.reshape(rows, columns)


def decode_matrix(rows: int, columns: int, data: np.ndarray, scale_inv: np.ndarray, amax: float) -> np.ndarray:
    """Decodes one matrix as the runtime does: `e2m1 * e4m3_scale * (amax / 2688)`."""
    codes = _unpack_codes(data, rows, columns)
    magnitude = _E2M1[codes & 0x07]
    values = np.where(codes & 0x08, -magnitude, magnitude).astype(np.float32)
    blocks = columns // BLOCK_SIZE
    scales = E4M3[scale_inv[:rows, :blocks]]
    global_scale = np.float32(1.0) if amax == 0.0 else np.float32(amax) / np.float32(GLOBAL_SCALE_DENOMINATOR)
    return (
        values.reshape(rows, blocks, BLOCK_SIZE) * scales[:, :, None] * global_scale
    ).reshape(rows, columns).astype(np.float32)


def validate_file(path: pathlib.Path, result: ValidatedContainer) -> None:
    """Validates one container file and adds its tensors to `result`."""
    context = str(path)
    with safe_open(path, framework="numpy") as handle:
        metadata = handle.metadata() or {}
        _require(metadata.get("modelq.format") == FORMAT, f"{context}: modelq.format must be {FORMAT!r}")
        try:
            manifest = json.loads(metadata.get("modelq.manifest", ""))
        except ValueError as error:
            raise ValidationError(f"{context}: modelq.manifest is not valid JSON: {error}") from error
        _check_manifest_header(manifest, context)

        actual = {name: _tensor_metadata(handle, name) for name in handle.keys()}
        accounted: set[str] = set()
        for name, entry in manifest["tensors"].items():
            what = f"{context}: tensor {name!r}"
            _require(isinstance(entry, dict), f"{what} must be an object")
            if entry.get("action") == "preserved":
                _require(name in actual, f"{what} is listed but missing from the file")
                dtype, shape = actual[name]
                _require(
                    dtype == entry.get("dtype") and list(shape) == entry.get("shape"),
                    f"{what} does not match its manifest entry",
                )
                accounted.add(name)
                _require(
                    name not in result.matrices and name not in result.preserved,
                    f"{what} appears in more than one shard",
                )
                result.preserved[name] = path
                continue
            _require(entry.get("action") == "quantized", f"{what}: unknown action {entry.get('action')!r}")
            logical = entry.get("logical_shape")
            _require(
                isinstance(logical, list) and len(logical) == 2 and all(isinstance(v, int) for v in logical),
                f"{what}: logical_shape must be two integers",
            )
            rows, columns = logical
            _require(
                rows > 0 and columns > 0 and rows % BLOCK_SIZE == 0 and columns % BLOCK_SIZE == 0,
                f"{what}: dimensions must be positive multiples of 16",
            )
            padded = (_round_up(rows, ROW_ALIGNMENT), _round_up(columns // BLOCK_SIZE, COLUMN_ALIGNMENT))
            fields = entry.get("fields")
            _require(isinstance(fields, dict), f"{what}: fields must be an object")
            expected = {
                "rowwise_data": ("U8", (rows, columns // 2)),
                "rowwise_scale_inv": ("U8", padded),
                "amax_rowwise": ("F32", (1,)),
            }
            for key, (dtype, shape) in expected.items():
                field_name = fields.get(key)
                _require(field_name in actual, f"{what}: field {key} ({field_name!r}) is missing")
                _require(
                    actual[field_name] == (dtype, shape),
                    f"{what}: field {field_name!r} must be {dtype} {shape}, found {actual[field_name]}",
                )
                accounted.add(field_name)
            data = handle.get_tensor(fields["rowwise_data"])
            scale_inv = handle.get_tensor(fields["rowwise_scale_inv"])
            amax = float(handle.get_tensor(fields["amax_rowwise"])[0])
            _validate_scale_payload(what, rows, columns, scale_inv, data, amax)
            _require(
                name not in result.matrices and name not in result.preserved,
                f"{what} appears in more than one shard",
            )
            result.matrices[name] = MatrixEntry(
                name=name,
                logical_shape=(rows, columns),
                original_dtype=str(entry.get("original_dtype")),
                file=path,
                rowwise_data=fields["rowwise_data"],
                rowwise_scale_inv=fields["rowwise_scale_inv"],
                amax_rowwise=fields["amax_rowwise"],
            )
        extra = sorted(set(actual) - accounted)
        _require(not extra, f"{context}: tensors not in the manifest: {extra}")
    result.files.append(path)


def validate_multi_container(path: pathlib.Path) -> ValidatedContainer:
    """Validates a single-file or sharded container and returns its contents."""
    result = ValidatedContainer()
    for file in container_files(path):
        validate_file(file, result)
    _require(bool(result.matrices), "the container holds no Transformer Engine matrices")
    return result


def load_matrix(entry: MatrixEntry) -> tuple[np.ndarray, np.ndarray, float]:
    """Reads a validated matrix's three fields."""
    with safe_open(entry.file, framework="numpy") as handle:
        return (
            handle.get_tensor(entry.rowwise_data),
            handle.get_tensor(entry.rowwise_scale_inv),
            float(handle.get_tensor(entry.amax_rowwise)[0]),
        )


def validate_reference_multi(
    container: ValidatedContainer, reference_path: pathlib.Path, *, rtol: float = 1e-5, atol: float = 1e-5
) -> float:
    """Decodes every matrix and compares it with the F32 reference.

    Returns the largest absolute difference.
    """
    worst = 0.0
    with safe_open(reference_path, framework="numpy") as handle:
        _require(
            (handle.metadata() or {}).get("modelq.reference_schema") == REFERENCE_SCHEMA,
            f"{reference_path}: modelq.reference_schema must be {REFERENCE_SCHEMA!r}",
        )
        expected_names = {name + REFERENCE_SUFFIX for name in container.matrices}
        found = set(handle.keys())
        _require(found == expected_names, f"reference tensors {sorted(found)} do not match matrices {sorted(expected_names)}")
        for name, entry in container.matrices.items():
            reference = handle.get_tensor(name + REFERENCE_SUFFIX)
            rows, columns = entry.logical_shape
            _require(
                reference.dtype == np.float32 and reference.shape == (rows, columns),
                f"reference for {name!r} must be F32 {(rows, columns)}",
            )
            _require(bool(np.isfinite(reference).all()), f"reference for {name!r} is not finite")
            data, scale_inv, amax = load_matrix(entry)
            decoded = decode_matrix(rows, columns, data, scale_inv, amax)
            _require(
                bool(np.allclose(decoded, reference, rtol=rtol, atol=atol)),
                f"{name!r} does not decode to its reference within rtol={rtol}, atol={atol}",
            )
            worst = max(worst, float(np.max(np.abs(decoded - reference))))
    return worst


# ----------------------------------------------------------------- runtime proof

SECOND_OPERAND_ROWS = 64
DEQUANTIZE_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}
GEMM_TOLERANCE = {"rtol": 0.125, "atol": 0.0675}
RESULT_MARKER = "RESULT_JSON: "


class Backend(Protocol):
    """What the proof needs from a runtime; the real one wraps Transformer Engine."""

    def environment(self) -> dict[str, Any]:
        """Preflights the environment and returns the versions to record."""

    def load_weight(self, entry: MatrixEntry, data: np.ndarray, scale_inv: np.ndarray, amax: float) -> Any:
        """Builds the runtime tensor from the exported rowwise fields."""

    def dequantize(self, weight: Any) -> np.ndarray:
        """Returns the runtime's own dequantization as an F32 array."""

    def gemm_error(self, weight: Any, reference: np.ndarray) -> float:
        """Runs one TN GEMM and returns its maximum absolute error.

        Raises `ValidationError` if the output has the wrong shape, is not
        finite, or is outside the GEMM tolerance.
        """


def second_operand_values(columns: int) -> np.ndarray:
    """The deterministic `[64, columns]` second operand, identical to Task 28's recipe."""
    indices = np.arange(SECOND_OPERAND_ROWS * columns, dtype=np.int64)
    values = (((indices * 37) % 251) - 125).astype(np.float32) / np.float32(32.0)
    return values.reshape(SECOND_OPERAND_ROWS, columns)


def evaluate_matrices(
    container: ValidatedContainer, reference_path: pathlib.Path, backend: Backend
) -> list[dict[str, Any]]:
    """Checks every matrix and keeps going after a failure.

    A paid hardware run should reveal every incompatibility at once, so a
    failing matrix is recorded and the loop continues.
    """
    results = []
    with safe_open(reference_path, framework="numpy") as reference_file:
        for name, entry in sorted(container.matrices.items()):
            rows, columns = entry.logical_shape
            record: dict[str, Any] = {"name": name, "shape": [rows, columns], "status": "pass"}
            try:
                reference = reference_file.get_tensor(name + REFERENCE_SUFFIX)
                data, scale_inv, amax = load_matrix(entry)
                weight = backend.load_weight(entry, data, scale_inv, amax)
                runtime_values = np.asarray(backend.dequantize(weight), dtype=np.float32)
                _require(
                    runtime_values.shape == reference.shape
                    and bool(np.allclose(runtime_values, reference, **DEQUANTIZE_TOLERANCE)),
                    "the runtime's dequantization does not match the CPU reference "
                    f"(max difference {float(np.max(np.abs(runtime_values - reference))) if runtime_values.shape == reference.shape else 'shape mismatch'})",
                )
                record["dequantize_max_abs_error"] = float(np.max(np.abs(runtime_values - reference)))
                record["gemm_max_abs_error"] = backend.gemm_error(weight, reference)
            except Exception as error:  # noqa: BLE001 - every failure must be recorded
                record["status"] = "fail"
                record["detail"] = f"{type(error).__name__}: {error}"
            results.append(record)
    return results


class TransformerEngineBackend:
    """Transformer Engine 2.19.0 on a Blackwell GPU, following Task 28's recipe."""

    def __init__(self) -> None:
        self._torch = None
        self._te: dict[str, Any] = {}
        self._device = None
        self._quantizer = None

    def environment(self) -> dict[str, Any]:
        system, machine = platform.system(), platform.machine()
        validate._validate_runtime_platform(system, machine)
        try:
            te_version = importlib.metadata.version("transformer-engine")
        except importlib.metadata.PackageNotFoundError as error:
            raise ValidationError("runtime proof requires the Transformer Engine 2.19.0 package") from error
        try:
            import torch
        except ImportError as error:
            raise ValidationError(f"runtime proof requires CUDA-enabled PyTorch: {error}") from error
        cuda_available = torch.cuda.is_available()
        capability = torch.cuda.get_device_capability() if cuda_available else None
        validate.validate_runtime_preflight(
            system, machine, te_version, cuda_available, torch.version.cuda,
            torch.backends.cudnn.version(), capability,
        )
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ValidationError(f"cannot query NVIDIA driver with nvidia-smi: {error}") from error
        _require(bool(driver) and bool(driver[0].strip()), "nvidia-smi returned no NVIDIA driver version")

        from transformer_engine.pytorch import NVFP4Quantizer, NVFP4Tensor
        from transformer_engine.pytorch.constants import DType
        from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

        self._torch = torch
        self._te = {"NVFP4Tensor": NVFP4Tensor, "DType": DType, "general_gemm": general_gemm}
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._quantizer = NVFP4Quantizer(
            fp4_dtype=DType.kFloat4E2M1,
            rowwise=True,
            columnwise=False,
            with_2d_quantization=False,
            with_rht=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
            nvfp4_use_4over6=False,
        )
        return {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "driver": driver[0].strip(),
            "transformer_engine": te_version,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": f"{capability[0]}.{capability[1]}",
        }

    def load_weight(self, entry: MatrixEntry, data: np.ndarray, scale_inv: np.ndarray, amax: float) -> Any:
        torch, device, te = self._torch, self._device, self._te
        return te["NVFP4Tensor"](
            shape=tuple(entry.logical_shape),
            dtype=torch.float32,
            rowwise_data=torch.as_tensor(data, device=device),
            rowwise_scale_inv=torch.as_tensor(scale_inv, device=device),
            columnwise_data=None,
            columnwise_scale_inv=None,
            amax_rowwise=torch.as_tensor(np.array([amax], dtype=np.float32), device=device),
            amax_columnwise=None,
            fp4_dtype=te["DType"].kFloat4E2M1,
            quantizer=self._quantizer,
            with_gemm_swizzled_scales=False,
            row_scaled_nvfp4=False,
            nvfp4_use_4over6=False,
            device=device,
        )

    def dequantize(self, weight: Any) -> np.ndarray:
        with self._torch.no_grad():
            return weight.dequantize(dtype=self._torch.float32).cpu().numpy()

    def gemm_error(self, weight: Any, reference: np.ndarray) -> float:
        torch, device = self._torch, self._device
        rows, columns = reference.shape
        with torch.no_grad():
            reference_weight = torch.as_tensor(reference, dtype=torch.float32, device=device)
            rhs = torch.as_tensor(second_operand_values(columns), device=device)
            rhs_quantized = self._quantizer.quantize(rhs)
            rhs_dequantized = rhs_quantized.dequantize(dtype=torch.float32)
            out, _, _, _ = self._te["general_gemm"](
                weight, rhs_quantized, out_dtype=torch.float32, layout="TN"
            )
            expected = rhs_dequantized @ reference_weight.T
            _require(
                tuple(out.shape) == (SECOND_OPERAND_ROWS, rows),
                f"TN GEMM output shape must be {(SECOND_OPERAND_ROWS, rows)}, got {tuple(out.shape)}",
            )
            _require(bool(torch.isfinite(out).all().item()), "TN GEMM output must be finite")
            torch.testing.assert_close(out, expected, **GEMM_TOLERANCE)
            return float((out - expected).abs().max().item())


def run_runtime_proof(
    container_path: pathlib.Path, reference_path: pathlib.Path, backend: Backend | None = None
) -> dict[str, Any]:
    """Validates on the CPU, then runs every matrix through the runtime."""
    container = validate_multi_container(container_path)
    validate_reference_multi(container, reference_path)
    backend = backend if backend is not None else TransformerEngineBackend()
    environment = backend.environment()
    results = evaluate_matrices(container, reference_path, backend)
    return {
        "container": str(container_path),
        "environment": environment,
        "matrices": results,
        "passed": all(record["status"] == "pass" for record in results),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    cpu = sub.add_parser("cpu", help="CPU container check (no TE, no CUDA)")
    cpu.add_argument("--container", type=pathlib.Path, required=True, help="file, or directory with an index")
    cpu.add_argument("--reference", type=pathlib.Path, help="F32 reference file to compare against")
    runtime = sub.add_parser("runtime", help="explicit Blackwell runtime proof (Linux, pinned TE 2.19.0)")
    runtime.add_argument("--container", type=pathlib.Path, required=True, help="file, or directory with an index")
    runtime.add_argument("--reference", type=pathlib.Path, required=True, help="F32 reference file")
    args = parser.parse_args(argv)

    if args.mode == "runtime":
        try:
            outcome = run_runtime_proof(args.container, args.reference)
        except ValidationError as error:
            print(f"validation failed: {error}", file=sys.stderr)
            return 1
        for record in outcome["matrices"]:
            detail = record.get("detail") or (
                f"dequantize_max_abs_error={record['dequantize_max_abs_error']} "
                f"gemm_max_abs_error={record['gemm_max_abs_error']}"
            )
            print(f"  {record['status']:4} {record['name']} {record['shape']}: {detail}")
        print(RESULT_MARKER + json.dumps(outcome))
        if not outcome["passed"]:
            print("validation failed: at least one matrix did not pass", file=sys.stderr)
            return 1
        print("runtime proof passed for every matrix")
        return 0
    try:
        container = validate_multi_container(args.container)
        print(
            f"container ok: {len(container.matrices)} matrices, "
            f"{len(container.preserved)} preserved tensors, {len(container.files)} file(s)"
        )
        for name, entry in sorted(container.matrices.items()):
            print(f"  {name}: shape={list(entry.logical_shape)} dtype={entry.original_dtype}")
        if args.reference is not None:
            worst = validate_reference_multi(container, args.reference)
            print(f"reference ok: max_abs_difference={worst}")
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    print("CPU check passed; Transformer Engine compatibility is not established by this check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
