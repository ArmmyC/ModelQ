"""Readers for every container `modelq quantize` writes, so `modelq eval` can score any of them (ADR 0035).

`detect_container` names the format of a file from its header. `substitute_container` decodes the
file and writes the decoded matrices into a model's parameters, with names and shapes checked, as
`quality_eval` does for the Transformer Engine and group-wise low-bit formats. The decoders follow
the rules of the writers:

- INT8 (ADR 0002): each quantized tensor is `qdata * scale`, with an I8 payload and one F32 scale.
- NVFP4, native (ADR 0011): each value is `e2m1 * e4m3_block_scale * global_scale`, with the low
  nibble of each byte first and one E4M3 scale per 16 values.
- GGUF (ADR 0032, ADR 0033): Q8_0 and Q4_0 blocks decode as llama.cpp's reference does; F32 is raw.

Low-bit (ADR 0030) and Transformer Engine NVFP4 (ADR 0012) containers are read by `quality_eval`.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

import quality_eval as qe
import validate_multi as vm

NATIVE_FORMAT = "modelq-native"
INT8_SCHEMA = "modelq.int8.manifest.v1"
NVFP4_SCHEMA = "modelq.nvfp4.manifest.v1"
NVFP4_BLOCK = 16
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_F32 = 0
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGUF_BLOCK = 32
GGUF_BLOCK_BYTES = {GGML_Q4_0: 18, GGML_Q8_0: 34}


def detect_container(path: pathlib.Path) -> str:
    """Names the format of a ModelQ output: gguf, nvfp4-te, lowbit, int8 or nvfp4.

    Raises ValueError for any other file, so that an unsupported file is never read as a model.
    """
    with path.open("rb") as handle:
        if handle.read(4) == GGUF_MAGIC:
            return "gguf"
    from safetensors import safe_open

    try:
        with safe_open(path, framework="np") as reader:
            metadata = reader.metadata() or {}
    except Exception as error:  # noqa: BLE001 - safetensors raises its own error types
        raise ValueError(f"{path} is neither a GGUF file nor a SafeTensors container: {error}") from error
    if metadata.get("modelq.format") == vm.FORMAT:
        return "nvfp4-te"
    if metadata.get("modelq.format") == NATIVE_FORMAT:
        quantization = metadata.get("modelq.quantization")
        if quantization in ("int1", "int2", "int3", "int4"):
            return "lowbit"
        if quantization == "int8":
            return "int8"
        if quantization == "nvfp4":
            return "nvfp4"
    raise ValueError(f"{path} is not a ModelQ container that the evaluation can read")


def _manifest(metadata: dict[str, str], schema: str) -> dict[str, Any]:
    text = metadata.get("modelq.manifest")
    if text is None:
        raise ValueError("the container has no manifest")
    manifest = json.loads(text)
    if manifest.get("schema") != schema:
        raise ValueError(f"unknown manifest schema {manifest.get('schema')!r}; expected {schema!r}")
    return manifest


def _record_error(report: qe.SubstitutionReport, name: str, decoded: np.ndarray, original: np.ndarray) -> None:
    denominator = float(np.linalg.norm(original))
    report.relative_errors[name] = (
        float(np.linalg.norm(decoded - original) / denominator) if denominator > 0.0 else 0.0
    )
    report.replaced.append(name)


def _install(parameter: Any, decoded: np.ndarray, torch: Any) -> None:
    parameter.copy_(torch.from_numpy(np.ascontiguousarray(decoded)).to(device=parameter.device, dtype=parameter.dtype))


def _check_parameters(parameters: dict[str, Any], covered: set[str], path: pathlib.Path) -> None:
    missing = set(parameters) - covered
    if missing:
        raise ValueError(f"{path} lacks parameters: {sorted(missing)[:5]}")


# --- INT8 (ADR 0002, ADR 0011's manifest conventions) --------------------------------------


def substitute_int8_weights(model: Any, path: pathlib.Path, torch: Any) -> qe.SubstitutionReport:
    """Overwrites the model's parameters from an INT8 file: `qdata * scale` per quantized tensor."""
    from safetensors import safe_open

    parameters = dict(model.named_parameters())
    report = qe.SubstitutionReport()
    with safe_open(path, framework="pt") as reader:
        tensors = _manifest(reader.metadata() or {}, INT8_SCHEMA)["tensors"]
        with torch.no_grad():
            for name, entry in sorted(tensors.items()):
                parameter = parameters.get(name)
                if parameter is None:
                    raise ValueError(f"file tensor {name!r} has no model parameter")
                shape = tuple(entry["original_shape"])
                if shape != tuple(parameter.shape):
                    raise ValueError(f"{name!r}: file shape {shape} != parameter shape {tuple(parameter.shape)}")
                original = parameter.detach().to("cpu", dtype=torch.float32).numpy()
                if entry["action"] == "preserved":
                    decoded = reader.get_tensor(entry["tensor_name"]).to(torch.float32).numpy()
                    _install(parameter, decoded, torch)
                    continue
                if entry["action"] != "quantized":
                    raise ValueError(f"{name!r}: unknown action {entry['action']!r}")
                qdata = reader.get_tensor(entry["qdata_name"]).numpy().astype(np.float32)
                scale = np.float32(reader.get_tensor(entry["scale_name"]).numpy())
                decoded = (qdata * scale).reshape(shape).astype(np.float32)
                _record_error(report, name, decoded, original)
                _install(parameter, decoded, torch)
        _check_parameters(parameters, set(tensors), path)
    return report


# --- NVFP4, native (ADR 0011) ----------------------------------------------------------------


def decode_nvfp4(packed: np.ndarray, block_scales: np.ndarray, global_scale: float, elements: int) -> np.ndarray:
    """Decodes ModelQ's native NVFP4 fields to float32: `e2m1 * e4m3_block_scale * global_scale`."""
    if elements % NVFP4_BLOCK:
        raise ValueError(f"{elements} values is not a whole number of {NVFP4_BLOCK}-value blocks")
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    block_scales = np.asarray(block_scales, dtype=np.uint8).reshape(-1)
    if packed.size != elements // 2:
        raise ValueError(f"{elements} values need {elements // 2} packed bytes, got {packed.size}")
    if block_scales.size != elements // NVFP4_BLOCK:
        raise ValueError(f"{elements} values need {elements // NVFP4_BLOCK} block scales, got {block_scales.size}")
    # The low nibble of each byte is the earlier value (the Rust unpacker's rule).
    codes = np.stack([packed & 0x0F, packed >> 4], axis=1).reshape(-1)[:elements]
    magnitude = vm._E2M1[codes & 0x07]
    signed = np.where(codes & 0x08, -magnitude, magnitude).astype(np.float32)
    scales = vm.E4M3[block_scales].astype(np.float32)
    if not np.all(np.isfinite(scales)):
        raise ValueError("a block scale is not a finite E4M3 value")
    per_value = np.repeat(scales, NVFP4_BLOCK)
    values = (signed * per_value) * np.float32(global_scale)
    if not np.all(np.isfinite(values)):
        raise ValueError("a decoded value overflows float32")
    return values.astype(np.float32)


def substitute_nvfp4_native_weights(model: Any, path: pathlib.Path, torch: Any) -> qe.SubstitutionReport:
    """Overwrites the model's parameters from a native NVFP4 file."""
    from safetensors import safe_open

    parameters = dict(model.named_parameters())
    report = qe.SubstitutionReport()
    with safe_open(path, framework="pt") as reader:
        tensors = _manifest(reader.metadata() or {}, NVFP4_SCHEMA)["tensors"]
        with torch.no_grad():
            for name, entry in sorted(tensors.items()):
                parameter = parameters.get(name)
                if parameter is None:
                    raise ValueError(f"file tensor {name!r} has no model parameter")
                shape = tuple(entry["original_shape"])
                if shape != tuple(parameter.shape):
                    raise ValueError(f"{name!r}: file shape {shape} != parameter shape {tuple(parameter.shape)}")
                if entry["action"] == "preserved":
                    decoded = reader.get_tensor(entry["tensor_name"]).to(torch.float32).numpy()
                    _install(parameter, decoded, torch)
                    continue
                if entry["action"] != "quantized":
                    raise ValueError(f"{name!r}: unknown action {entry['action']!r}")
                original = parameter.detach().to("cpu", dtype=torch.float32).numpy()
                elements = math.prod(shape)
                packed = reader.get_tensor(entry["qdata_name"]).numpy()
                block_scales = reader.get_tensor(entry["block_scale_name"]).numpy()
                global_scale = float(reader.get_tensor(entry["global_scale_name"]).numpy())
                decoded = decode_nvfp4(packed, block_scales, global_scale, elements).reshape(shape)
                _record_error(report, name, decoded, original)
                _install(parameter, decoded, torch)
        _check_parameters(parameters, set(tensors), path)
    return report


# --- GGUF (ADR 0032, ADR 0033) ----------------------------------------------------------------

# GGUF value types for scalars, as struct formats (little-endian).
_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


@dataclass(frozen=True)
class GgufTensor:
    """One tensor's record. `dims` are in GGML order: the first dimension varies fastest."""

    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int


@dataclass
class GgufFile:
    """The parsed header of a GGUF file, with the whole file memory-mapped for its tensor data."""

    metadata: dict[str, Any]
    tensors: dict[str, GgufTensor]
    data_start: int
    buffer: np.ndarray


class _Cursor:
    def __init__(self, buffer: np.ndarray) -> None:
        self.buffer = buffer
        self.position = 0

    def take(self, count: int) -> bytes:
        end = self.position + count
        if end > len(self.buffer):
            raise ValueError("the GGUF header is truncated")
        chunk = self.buffer[self.position : end].tobytes()
        self.position = end
        return chunk

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.unpack("<Q")).decode("utf-8")

    def value(self, kind: int) -> Any:
        if kind in _GGUF_SCALARS:
            return self.unpack(_GGUF_SCALARS[kind])
        if kind == _GGUF_STRING:
            return self.string()
        if kind == _GGUF_ARRAY:
            element_kind = self.unpack("<I")
            count = self.unpack("<Q")
            return [self.value(element_kind) for _ in range(count)]
        raise ValueError(f"unsupported GGUF value type {kind}")


def read_gguf(path: pathlib.Path) -> GgufFile:
    """Parses a GGUF version 3 file's header; tensor data is read on demand."""
    buffer = np.memmap(path, dtype=np.uint8, mode="r")
    cursor = _Cursor(buffer)
    if cursor.take(4) != GGUF_MAGIC:
        raise ValueError(f"{path} is not a GGUF file")
    version = cursor.unpack("<I")
    if version != GGUF_VERSION:
        raise ValueError(f"GGUF version {version} is not supported; the evaluation reads version {GGUF_VERSION}")
    tensor_count = cursor.unpack("<Q")
    kv_count = cursor.unpack("<Q")
    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = cursor.string()
        metadata[key] = cursor.value(cursor.unpack("<I"))
    tensors: dict[str, GgufTensor] = {}
    for _ in range(tensor_count):
        name = cursor.string()
        dims = tuple(cursor.unpack("<Q") for _ in range(cursor.unpack("<I")))
        ggml_type = cursor.unpack("<I")
        offset = cursor.unpack("<Q")
        if name in tensors:
            raise ValueError(f"duplicate GGUF tensor {name!r}")
        tensors[name] = GgufTensor(name, dims, ggml_type, offset)
    alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
    data_start = -(-cursor.position // alignment) * alignment
    return GgufFile(metadata, tensors, data_start, buffer)


def gguf_values(gguf: GgufFile, tensor: GgufTensor) -> np.ndarray:
    """Decodes one tensor to a flat float32 array, in row-major order."""
    count = math.prod(tensor.dims)
    start = gguf.data_start + tensor.offset
    if tensor.ggml_type == GGML_F32:
        raw = np.asarray(gguf.buffer[start : start + 4 * count]).copy()
        if raw.size != 4 * count:
            raise ValueError(f"GGUF tensor {tensor.name!r} is truncated")
        return raw.view("<f4").astype(np.float32)
    if tensor.ggml_type not in GGUF_BLOCK_BYTES:
        raise ValueError(f"GGUF tensor {tensor.name!r} has type {tensor.ggml_type}; the evaluation reads F32, Q8_0 and Q4_0")
    if count % GGUF_BLOCK:
        raise ValueError(f"GGUF tensor {tensor.name!r} has {count} values, not a whole number of 32-value blocks")
    block_bytes = GGUF_BLOCK_BYTES[tensor.ggml_type]
    blocks = count // GGUF_BLOCK
    raw = np.asarray(gguf.buffer[start : start + blocks * block_bytes]).copy()
    if raw.size != blocks * block_bytes:
        raise ValueError(f"GGUF tensor {tensor.name!r} is truncated")
    raw = raw.reshape(blocks, block_bytes)
    # Each block starts with a binary16 scale; the same product gguf-py and llama.cpp compute.
    scales = raw[:, :2].copy().view("<f2").reshape(blocks).astype(np.float32)
    if tensor.ggml_type == GGML_Q8_0:
        quants = raw[:, 2:].copy().view(np.int8).astype(np.float32)
    else:
        packed = raw[:, 2:]
        low = (packed & 0x0F).astype(np.float32) - 8.0
        high = (packed >> 4).astype(np.float32) - 8.0
        quants = np.concatenate([low, high], axis=1)
    return (quants * scales[:, None]).reshape(-1).astype(np.float32)


_GGUF_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_GGUF_LAYER_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def hf_name_for_gguf(name: str) -> str | None:
    """The Hugging Face parameter name of a llama.cpp qwen2 tensor name (the inverse of the exporter's map)."""
    if name in _GGUF_NAMES:
        return _GGUF_NAMES[name]
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if match is None:
        return None
    mapped = _GGUF_LAYER_NAMES.get(match.group(2))
    return None if mapped is None else f"model.layers.{int(match.group(1))}.{mapped}"


def substitute_gguf_weights(model: Any, path: pathlib.Path, torch: Any) -> qe.SubstitutionReport:
    """Overwrites the model's parameters from a qwen2 GGUF file; F32 tensors are copied, Q8_0 and Q4_0 decoded."""
    gguf = read_gguf(path)
    architecture = gguf.metadata.get("general.architecture")
    if architecture != "qwen2":
        raise ValueError(f"{path} is a {architecture!r} GGUF file; the evaluation reads qwen2 files")
    parameters = dict(model.named_parameters())
    report = qe.SubstitutionReport()
    covered: set[str] = set()
    with torch.no_grad():
        for name, tensor in sorted(gguf.tensors.items()):
            hf_name = hf_name_for_gguf(name)
            if hf_name is None:
                raise ValueError(f"GGUF tensor {name!r} has no Hugging Face name")
            parameter = parameters.get(hf_name)
            if parameter is None:
                raise ValueError(f"GGUF tensor {name!r} maps to {hf_name!r}, which the model does not have")
            shape = tuple(reversed(tensor.dims))
            if shape != tuple(parameter.shape):
                raise ValueError(f"{name!r}: file shape {shape} != parameter shape {tuple(parameter.shape)}")
            decoded = gguf_values(gguf, tensor).reshape(shape)
            if tensor.ggml_type == GGML_F32:
                _install(parameter, decoded, torch)
            else:
                original = parameter.detach().to("cpu", dtype=torch.float32).numpy()
                _record_error(report, hf_name, decoded, original)
                _install(parameter, decoded, torch)
            covered.add(hf_name)
    _check_parameters(parameters, covered, path)
    return report


# --- dispatch --------------------------------------------------------------------------------


def substitute_container(model: Any, path: pathlib.Path, torch: Any) -> tuple[str, qe.SubstitutionReport]:
    """Detects the container's format and overwrites the model's parameters from it."""
    kind = detect_container(path)
    if kind == "gguf":
        return kind, substitute_gguf_weights(model, path, torch)
    if kind == "nvfp4-te":
        return kind, qe.substitute_nvfp4_weights(model, path, torch)
    if kind == "lowbit":
        return kind, qe.substitute_lowbit_weights(model, path, torch)
    if kind == "int8":
        return kind, substitute_int8_weights(model, path, torch)
    return kind, substitute_nvfp4_native_weights(model, path, torch)
