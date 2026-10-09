"""Output-quality evaluation of ModelQ's NVFP4 export.

Compares an original causal language model with a copy whose weights were
replaced by the values decoded from a ModelQ Transformer Engine NVFP4
container, on the same text, and reports how far the quantized model's
outputs move.  The decoded values are used in float32 ("simulated"
quantization), so the comparison isolates the effect of the quantized weights:
no runtime, activation quantization, or kernel numerics are involved.

Metrics, all computed over the same non-overlapping windows:

- perplexity of the original and of the quantized model;
- the mean KL divergence KL(original || quantized) of the next-token
  distributions, averaged over all predicted positions; and
- the fraction of positions where both models predict the same top-1 token.

This is a measurement, not a pass/fail gate.  It says how much quality the
weight quantization costs on one model and one text; it does not certify the
model for any use.
"""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

# The container decoder lives with the Transformer Engine tools.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "transformer_engine_nvfp4"))
import validate_multi  # noqa: E402  (path set above)

KL_CHUNK_POSITIONS = 256


@dataclass
class SubstitutionReport:
    """What `substitute_nvfp4_weights` replaced and how far each matrix moved."""

    replaced: list[str] = field(default_factory=list)
    relative_errors: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        errors = sorted(self.relative_errors.values())
        return {
            "matrices_replaced": len(self.replaced),
            "relative_frobenius_error_mean": float(np.mean(errors)) if errors else None,
            "relative_frobenius_error_median": errors[len(errors) // 2] if errors else None,
            "relative_frobenius_error_max": errors[-1] if errors else None,
        }


def make_windows(token_ids: Iterable[int], length: int) -> list[list[int]]:
    """Non-overlapping windows of exactly `length` tokens; a short tail is dropped."""
    if length < 2:
        raise ValueError("a window needs at least two tokens to score a prediction")
    tokens = list(token_ids)
    return [tokens[start : start + length] for start in range(0, len(tokens) - length + 1, length)]


def substitute_nvfp4_weights(model: Any, container_path: pathlib.Path, torch: Any) -> SubstitutionReport:
    """Overwrites the model's matching parameters with the container's decoded matrices.

    Every matrix in the container must correspond to a parameter of the same
    name and shape; anything else is an error, so a naming mismatch cannot
    silently leave a weight unquantized.  Tensors the container preserved are
    left alone (they are byte-identical to the source).  Returns the per-matrix
    relative Frobenius error between the original and the decoded weights.
    """
    container = validate_multi.validate_multi_container(container_path)
    parameters = dict(model.named_parameters())
    report = SubstitutionReport()
    with torch.no_grad():
        for name, entry in sorted(container.matrices.items()):
            if name not in parameters:
                raise ValueError(f"container matrix {name!r} has no matching model parameter")
            parameter = parameters[name]
            rows, columns = entry.logical_shape
            if tuple(parameter.shape) != (rows, columns):
                raise ValueError(
                    f"{name!r}: container shape {(rows, columns)} != parameter shape {tuple(parameter.shape)}"
                )
            data, scale_inv, amax = validate_multi.load_matrix(entry)
            decoded = validate_multi.decode_matrix(rows, columns, data, scale_inv, amax)
            original = parameter.detach().to("cpu", dtype=torch.float32).numpy()
            denominator = float(np.linalg.norm(original))
            report.relative_errors[name] = (
                float(np.linalg.norm(decoded - original) / denominator) if denominator > 0.0 else 0.0
            )
            parameter.copy_(torch.from_numpy(decoded).to(device=parameter.device, dtype=parameter.dtype))
            report.replaced.append(name)
    return report


def _window_metrics(base_logits: Any, quant_logits: Any, labels: Any, torch: Any) -> dict[str, float]:
    """Metrics for one window, from logits at positions 0..L-2 and labels 1..L-1."""
    positions = base_logits.shape[0]
    base_nll = quant_nll = kl = 0.0
    agree = 0
    for start in range(0, positions, KL_CHUNK_POSITIONS):
        end = min(start + KL_CHUNK_POSITIONS, positions)
        base_log = torch.log_softmax(base_logits[start:end].float(), dim=-1)
        quant_log = torch.log_softmax(quant_logits[start:end].float(), dim=-1)
        chunk_labels = labels[start:end]
        rows = torch.arange(end - start, device=base_log.device)
        base_nll += float(-base_log[rows, chunk_labels].sum())
        quant_nll += float(-quant_log[rows, chunk_labels].sum())
        kl += float((base_log.exp() * (base_log - quant_log)).sum())
        agree += int((base_log.argmax(dim=-1) == quant_log.argmax(dim=-1)).sum())
    return {
        "positions": positions,
        "base_nll": base_nll,
        "quant_nll": quant_nll,
        "kl": kl,
        "agree": agree,
    }


def evaluate_pair(base_model: Any, quant_model: Any, windows: list[list[int]], torch: Any, device: Any) -> dict[str, Any]:
    """Runs both models over every window and aggregates the three metrics."""
    totals = {"positions": 0, "base_nll": 0.0, "quant_nll": 0.0, "kl": 0.0, "agree": 0}
    base_model.eval()
    quant_model.eval()
    with torch.no_grad():
        for window in windows:
            ids = torch.tensor([window], device=device)
            base_logits = base_model(ids).logits[0, :-1]
            quant_logits = quant_model(ids).logits[0, :-1]
            metrics = _window_metrics(base_logits, quant_logits, ids[0, 1:], torch)
            for key in totals:
                totals[key] += metrics[key]
    positions = totals["positions"]
    if positions == 0:
        raise ValueError("no positions were evaluated; the text is shorter than one window")
    return {
        "windows": len(windows),
        "positions": positions,
        "perplexity_original": math.exp(totals["base_nll"] / positions),
        "perplexity_quantized": math.exp(totals["quant_nll"] / positions),
        "mean_kl_original_to_quantized": totals["kl"] / positions,
        "top1_agreement": totals["agree"] / positions,
    }


def relative_perplexity_increase(result: dict[str, Any]) -> float:
    """`(ppl_quantized - ppl_original) / ppl_original`."""
    original = result["perplexity_original"]
    return (result["perplexity_quantized"] - original) / original


# --- group-wise low-bit files (ADR 0030) and plain checkpoints ---------------------------

LOWBIT_SCHEMA = "modelq.lowbit.manifest.v1"
_LOWBIT_SCHEMES = {"symmetric-group-wise": "symmetric", "sign-group-wise": "sign"}


@dataclass
class LowBitManifest:
    """The codec parameters and tensor entries of a ModelQ group-wise low-bit file."""

    bits: int
    group_size: int
    scheme: str
    tensors: dict[str, dict[str, Any]]


def read_lowbit_manifest(metadata: dict[str, str]) -> LowBitManifest | None:
    """Parses the manifest of a file written by `modelq quantize --format int4|int3|int2|int1`.

    Returns None for files that are not group-wise low-bit output, such as INT8
    files, and raises for a low-bit file that is malformed.
    """
    import json

    if metadata.get("modelq.format") != "modelq-native":
        return None
    if metadata.get("modelq.quantization") not in ("int1", "int2", "int3", "int4"):
        return None
    if metadata.get("modelq.format_version") != "2":
        raise ValueError(f"unsupported low-bit format version {metadata.get('modelq.format_version')!r}")
    scheme = _LOWBIT_SCHEMES.get(metadata.get("modelq.scheme", ""))
    if scheme is None:
        raise ValueError(f"unsupported low-bit scheme {metadata.get('modelq.scheme')!r}")
    manifest = json.loads(metadata["modelq.manifest"])
    if manifest.get("schema") != LOWBIT_SCHEMA:
        raise ValueError(f"unknown manifest schema {manifest.get('schema')!r}")
    return LowBitManifest(
        bits=int(metadata["modelq.bits"]),
        group_size=int(metadata["modelq.group_size"]),
        scheme=scheme,
        tensors=manifest["tensors"],
    )


def unpack_codes(packed: Any, elements: int, bits: int) -> Any:
    """Reads `elements` unsigned codes from an LSB-first bitstream (ADR 0030)."""
    if len(packed) != -(-elements * bits // 8):
        raise ValueError(f"{elements} values at {bits} bits need {-(-elements * bits // 8)} bytes, got {len(packed)}")
    stream = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")[: elements * bits]
    place_values = (1 << np.arange(bits, dtype=np.int64))
    return (stream.reshape(elements, bits).astype(np.int64) * place_values).sum(axis=1)


def decode_lowbit(packed: Any, scales: Any, elements: int, bits: int, group_size: int, scheme: str) -> Any:
    """Decodes codes and group scales to float32 values, with the Rust rules for each scheme."""
    codes = unpack_codes(packed, elements, bits)
    scales = np.asarray(scales, dtype=np.float32)
    if len(scales) != -(-elements // group_size):
        raise ValueError(f"expected {-(-elements // group_size)} group scales, got {len(scales)}")
    per_value_scale = np.repeat(scales, group_size)[:elements]
    if scheme == "sign":
        values = np.where(codes == 1, 1.0, -1.0).astype(np.float32)
        return values * per_value_scale
    sign_bit = 1 << (bits - 1)
    signed = np.where(codes >= sign_bit, codes - (1 << bits), codes)
    if np.any(signed == -sign_bit):
        raise ValueError("the reserved minimum code appears in the file")
    return (signed.astype(np.float32) * per_value_scale).astype(np.float32)


def substitute_lowbit_weights(
    model: Any,
    container_path: pathlib.Path,
    torch: Any,
    keep_original: Iterable[str] = (),
) -> SubstitutionReport:
    """Overwrites the model's parameters from a group-wise low-bit file.

    Quantized tensors are decoded from their packed codes and scales. Preserved
    tensors are copied from the file too, because a rescaled checkpoint changes
    some of them (AWQ's RMSNorm weights, ADR 0031).

    Quantized tensors named in `keep_original` are left at the model's values.
    That evaluates a policy that preserves them (the NVFP4 default preserves the
    embedding and output head) from a file that quantized them.
    """
    keep = set(keep_original)
    from safetensors import safe_open

    parameters = dict(model.named_parameters())
    report = SubstitutionReport()
    with safe_open(container_path, framework="pt") as reader:
        manifest = read_lowbit_manifest(reader.metadata() or {})
        if manifest is None:
            raise ValueError(f"{container_path} is not a ModelQ group-wise low-bit file")
        with torch.no_grad():
            for name, entry in sorted(manifest.tensors.items()):
                parameter = parameters.get(name)
                if parameter is None:
                    raise ValueError(f"file tensor {name!r} has no model parameter")
                if entry["action"] == "preserved":
                    tensor = reader.get_tensor(entry["tensor_name"]).to(torch.float32)
                    if tuple(tensor.shape) != tuple(parameter.shape):
                        raise ValueError(f"{name!r}: file shape {tuple(tensor.shape)} != {tuple(parameter.shape)}")
                    parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
                    continue
                if name in keep:
                    continue
                shape = tuple(entry["original_shape"])
                if shape != tuple(parameter.shape):
                    raise ValueError(f"{name!r}: container shape {shape} != parameter shape {tuple(parameter.shape)}")
                packed = reader.get_tensor(entry["qdata_name"]).numpy()
                scales = reader.get_tensor(entry["scale_name"]).numpy()
                decoded = decode_lowbit(
                    packed, scales, entry["elements"], manifest.bits, manifest.group_size, manifest.scheme
                ).reshape(shape)
                original = parameter.detach().to("cpu", dtype=torch.float32).numpy()
                denominator = float(np.linalg.norm(original))
                report.relative_errors[name] = (
                    float(np.linalg.norm(decoded - original) / denominator) if denominator > 0.0 else 0.0
                )
                parameter.copy_(torch.from_numpy(decoded).to(device=parameter.device, dtype=parameter.dtype))
                report.replaced.append(name)
    return report


def substitute_checkpoint_weights(model: Any, checkpoint_path: pathlib.Path, torch: Any) -> SubstitutionReport:
    """Overwrites every parameter from a plain SafeTensors checkpoint, with names and shapes checked.

    Used for a rescaled, unquantized checkpoint. The tied output head is the
    only parameter a checkpoint may omit.
    """
    from safetensors import safe_open

    parameters = dict(model.named_parameters())
    report = SubstitutionReport()
    with safe_open(checkpoint_path, framework="pt") as reader:
        names = set(reader.keys())
        missing = set(parameters) - names - {"lm_head.weight"}
        if missing:
            raise ValueError(f"checkpoint lacks parameters: {sorted(missing)[:5]}")
        with torch.no_grad():
            for name in sorted(names):
                parameter = parameters.get(name)
                if parameter is None:
                    raise ValueError(f"checkpoint tensor {name!r} has no model parameter")
                tensor = reader.get_tensor(name).to(torch.float32)
                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise ValueError(f"{name!r}: checkpoint shape {tuple(tensor.shape)} != {tuple(parameter.shape)}")
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
                report.replaced.append(name)
    return report
