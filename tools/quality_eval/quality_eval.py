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
