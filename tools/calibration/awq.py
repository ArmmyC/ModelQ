"""Activation-aware weight scaling (AWQ) for ModelQ's group-wise quantizer (ADR 0031).

AWQ chooses, for each block of linear layers that reads the same input, a
per-channel scale s. The channels that carry large activations get a larger
scale, so after quantization their weights keep more precision. The scale is an
exact change of variables, so the float model computes the same function:

    W x = (W diag(s)) (diag(1/s) x)

The 1/s is folded into whatever produces the block's input (an RMSNorm weight,
or the rows of the up projection), and s multiplies the block's weight columns.
The rescaled checkpoint is then quantized by the ordinary data-free quantizer,
which is what makes the result a plain ModelQ low-bit file.

Blocks rescaled in a Qwen2 decoder layer:

- A: input_layernorm -> q_proj, k_proj, v_proj (one input, three outputs)
- C: post_attention_layernorm -> gate_proj, up_proj
- D: up_proj -> down_proj (the scale acts on the MLP's intermediate channels)

The attention value/output pair (v_proj -> o_proj) is not rescaled. With
grouped-query attention, one value channel feeds several query heads, so the
change of variables cannot be applied per output channel there.

The search picks each block's exponent alpha from 20 values in [0, 1), scoring
each candidate by the squared output error of the block's linear layers on
calibration activations, with the candidate's weights quantized by the same
rules as ModelQ's INT4 (ADR 0030).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

ALPHA_GRID = tuple(index / 20 for index in range(20))
MINIMUM_ACTIVATION = 1e-4
INDEX_FILE = "model.safetensors.index.json"


def round_half_away(values: Any, torch_module: Any = torch) -> Any:
    """Rounds to the nearest integer, ties away from zero, as Rust's f32::round.

    The fraction is taken as `values - trunc(values)`, which is exact in float32.
    The simpler `floor(|v| + 0.5)` is not: for 0.49999997, adding 0.5 rounds up
    to 1.0 and gives the wrong code.
    """
    truncated = torch_module.trunc(values)
    fraction = values - truncated
    return truncated + torch_module.sign(fraction) * (torch_module.abs(fraction) >= 0.5).to(values.dtype)


def pseudo_quantize(weight: Any, group_size: int, bits: int = 4) -> Any:
    """Quantizes and dequantizes a matrix with ModelQ's symmetric group-wise rules.

    Groups are runs of `group_size` consecutive values along the input
    dimension. Each group's scale is max|w| / qmax (1.0 for an all-zero group),
    values are rounded ties-away-from-zero and clamped to [-qmax, qmax], and the
    result is the code times the scale (ADR 0030).
    """
    rows, columns = weight.shape
    if columns % group_size != 0:
        raise ValueError(f"{columns} columns are not a multiple of the group size {group_size}")
    qmax = 2 ** (bits - 1) - 1
    grouped = weight.reshape(rows, columns // group_size, group_size)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    # Dividing by a tensor keeps the division exact on every device. Dividing by a
    # Python scalar may be computed as a multiply by the reciprocal on a GPU, which
    # differs from the CPU and from Rust by one unit in the last place.
    scale = torch.where(max_abs > 0, max_abs / torch.full_like(max_abs, float(qmax)), torch.ones_like(max_abs))
    codes = round_half_away(grouped / scale).clamp(-qmax, qmax)
    return (codes * scale).reshape(rows, columns)


def channel_scale(activation_mean: Any, alpha: float) -> Any:
    """AWQ's per-channel scale: mean|x|^alpha, normalized to unit geometric spread."""
    scale = activation_mean.clamp(min=MINIMUM_ACTIVATION) ** alpha
    return scale / torch.sqrt(scale.max() * scale.min())


def block_error(inputs: Any, weights: Iterable[Any], scale: Any, group_size: int) -> float:
    """Squared output error of a block's linear layers with a candidate scale.

    The candidate quantizes W diag(s); the effective weight is that divided back
    by s, so the error is measured in the original input space.
    """
    total = 0.0
    for weight in weights:
        effective = pseudo_quantize(weight * scale[None, :], group_size) / scale[None, :]
        reference = inputs @ weight.T
        approximation = inputs @ effective.T
        total += float(((reference - approximation) ** 2).sum())
    return total


@dataclass
class BlockScale:
    """The scale chosen for one block of one decoder layer."""

    layer: int
    block: str
    alpha: float
    scale: Any
    error_unscaled: float
    error_scaled: float
    linears: tuple[str, ...] = field(default_factory=tuple)


def search_block(
    layer: int,
    block: str,
    inputs: Any,
    weights: dict[str, Any],
    group_size: int,
) -> BlockScale:
    """Finds the alpha whose scale minimizes the block's output error."""
    activation_mean = inputs.abs().mean(dim=0)
    ones = torch.ones_like(activation_mean)
    tensors = list(weights.values())
    unscaled = block_error(inputs, tensors, ones, group_size)
    best: tuple[float, float, Any] | None = None
    for alpha in ALPHA_GRID:
        scale = channel_scale(activation_mean, alpha)
        error = block_error(inputs, tensors, scale, group_size)
        if best is None or error < best[0]:
            best = (error, alpha, scale)
    assert best is not None
    error, alpha, scale = best
    return BlockScale(
        layer=layer,
        block=block,
        alpha=alpha,
        scale=scale,
        error_unscaled=unscaled,
        error_scaled=error,
        linears=tuple(weights),
    )


def layer_prefix(layer: int) -> str:
    return f"model.layers.{layer}."


def rescalings(blocks: list[BlockScale]) -> dict[str, list[Callable[[Any], Any]]]:
    """The rescalings of each tensor the blocks change, keyed by tensor name and in block order.

    A tensor can belong to two blocks: up_proj is a column-scaled linear of block C
    and a row-scaled input of block D. Its rescalings apply in block order, as the
    sequential fold always did. Each rescaling takes one checkpoint tensor and returns
    it as float32. It depends only on that tensor and on its block's scale, so tensors
    can be rescaled wherever they are stored, in any order (ADR 0038).

    For A and C the RMSNorm weight is divided by s and the block's weight columns are
    multiplied by s. For D the up projection's rows are divided by s and the down
    projection's columns are multiplied by s.
    """
    rescale: dict[str, list[Callable[[Any], Any]]] = {}

    def add(name: str, step: Callable[[Any], Any]) -> None:
        rescale.setdefault(name, []).append(step)

    for block in blocks:
        prefix = layer_prefix(block.layer)
        # The checkpoint tensors live on the CPU; the scale may have been computed on
        # the GPU during the search, so it is moved to match them.
        scale = block.scale.detach().to(device="cpu", dtype=torch.float32)
        if block.block in ("A", "C"):
            norm = "input_layernorm.weight" if block.block == "A" else "post_attention_layernorm.weight"
            add(prefix + norm, lambda tensor, scale=scale: tensor.to(torch.float32) / scale)
            for name in block.linears:
                add(prefix + name + ".weight", lambda tensor, scale=scale: tensor.to(torch.float32) * scale[None, :])
        elif block.block == "D":
            add(prefix + "mlp.up_proj.weight", lambda tensor, scale=scale: tensor.to(torch.float32) / scale[:, None])
            add(prefix + "mlp.down_proj.weight", lambda tensor, scale=scale: tensor.to(torch.float32) * scale[None, :])
        else:
            raise ValueError(f"unknown block {block.block!r}")
    return rescale


def rescale_tensor(steps: list[Callable[[Any], Any]], tensor: Any) -> Any:
    """Applies one tensor's rescalings, in block order."""
    for step in steps:
        tensor = step(tensor)
    return tensor


def _capture_block_inputs(
    model: Any,
    layer_index: int,
    windows: list[list[int]],
    device: Any,
    batch_size: int,
) -> dict[str, Any]:
    """The inputs of one layer's three blocks over every window, in window order (ADR 0038).

    The hooks sit on this layer only, so memory holds this layer's inputs and no others.
    The hooks do not change the model's computation, so the inputs are the same tensors
    that a single run over all layers would give.
    """
    layer = model.model.layers[layer_index]
    watched = {
        "attn": layer.self_attn.q_proj,
        "mlp": layer.mlp.gate_proj,
        "down": layer.mlp.down_proj,
    }
    captured: dict[str, list[Any]] = {name: [] for name in watched}

    def capture(name: str):
        def hook(_module, inputs):
            tensor = inputs[0].detach()
            captured[name].append(tensor.reshape(-1, tensor.shape[-1]).float())

        return hook

    handles = [module.register_forward_pre_hook(capture(name)) for name, module in watched.items()]
    try:
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                batch = torch.tensor(windows[start : start + batch_size], device=device)
                # The base model suffices: the hooks sit inside the decoder layers, and
                # skipping the output head avoids a vocabulary-sized logits tensor. The
                # cache setting is left at its default: turning it off changes the attention
                # kernels on a GPU, and with them the scales (ADR 0038).
                model.model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(parts) for name, parts in captured.items()}


def _search_layer(index: int, layer: Any, captured: dict[str, Any], group_size: int) -> list[BlockScale]:
    """The three blocks of one decoder layer, in the order A, C, D."""
    return [
        search_block(
            index,
            "A",
            captured["attn"],
            {
                "self_attn.q_proj": layer.self_attn.q_proj.weight.detach().float(),
                "self_attn.k_proj": layer.self_attn.k_proj.weight.detach().float(),
                "self_attn.v_proj": layer.self_attn.v_proj.weight.detach().float(),
            },
            group_size,
        ),
        search_block(
            index,
            "C",
            captured["mlp"],
            {
                "mlp.gate_proj": layer.mlp.gate_proj.weight.detach().float(),
                "mlp.up_proj": layer.mlp.up_proj.weight.detach().float(),
            },
            group_size,
        ),
        search_block(
            index,
            "D",
            captured["down"],
            {"mlp.down_proj": layer.mlp.down_proj.weight.detach().float()},
            group_size,
        ),
    ]


def calibrate(
    model: Any,
    windows: list[list[int]],
    group_size: int,
    device: Any,
    batch_size: int = 4,
) -> list[BlockScale]:
    """Searches the scales of every layer on the calibration windows, one layer at a time (ADR 0038).

    `windows` are token id lists of equal length. The activations come from the
    unmodified float model; folding a scale leaves those activations unchanged, so
    the search does not need to rerun the model after each block. Each layer's inputs
    come from a run of the model of their own, so memory holds one layer's inputs at a
    time, not every layer's. The blocks come out in the order A, C, D for each layer,
    as before.
    """
    model.eval()
    blocks: list[BlockScale] = []
    for index, layer in enumerate(model.model.layers):
        captured = _capture_block_inputs(model, index, windows, device, batch_size)
        blocks.extend(_search_layer(index, layer, captured, group_size))
        del captured
    return blocks


def summarize(blocks: list[BlockScale]) -> dict[str, Any]:
    """A JSON-ready summary of the search, for the calibration record."""
    total_unscaled = sum(block.error_unscaled for block in blocks)
    total_scaled = sum(block.error_scaled for block in blocks)
    return {
        "alpha_grid": list(ALPHA_GRID),
        "blocks": [
            {
                "layer": block.layer,
                "block": block.block,
                "linears": list(block.linears),
                "alpha": block.alpha,
                "error_unscaled": block.error_unscaled,
                "error_scaled": block.error_scaled,
            }
            for block in blocks
        ],
        "total_error_unscaled": total_unscaled,
        "total_error_scaled": total_scaled,
        "error_reduction": (
            1.0 - total_scaled / total_unscaled if total_unscaled > 0 else math.nan
        ),
    }


def write_scaled_checkpoint(
    source: Any,
    blocks: list[BlockScale],
    destination: Any,
    metadata: dict[str, str],
) -> None:
    """Writes the rescaled checkpoint as one file: transformed tensors as F32, all others unchanged.

    `source` is an open `safetensors.safe_open` handle. Keeping the other
    tensors' dtypes means the quantized output stores the same bytes for them
    as the data-free path does.
    """
    from safetensors.torch import save_file

    rescale = rescalings(blocks)
    names = set(source.keys())
    missing = sorted(set(rescale) - names)
    if missing:
        raise ValueError(f"the checkpoint has no tensor named {missing[0]!r}")
    tensors = {}
    for name in sorted(names):
        tensor = source.get_tensor(name)
        if name in rescale:
            tensor = rescale_tensor(rescale[name], tensor)
        tensors[name] = tensor.contiguous()
    save_file(tensors, destination, metadata=metadata)


def write_scaled_shards(
    model_dir: Any,
    blocks: list[BlockScale],
    destination: Any,
    metadata: dict[str, str],
) -> list[str]:
    """Writes the rescaled checkpoint in the layout of a sharded checkpoint (ADR 0038).

    `model_dir` holds `model.safetensors.index.json` and the shards it names.
    `destination` becomes a directory with the same shard names and the same weight
    map, so the INT4 writer reads it as it reads any sharded checkpoint. The index's
    total size is recomputed, because the rescaled tensors are F32. Shards are
    rescaled one at a time, so only one source shard is held at once. Returns the
    shard names, in order.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    model_dir = Path(model_dir)
    destination = Path(destination)
    index = json.loads((model_dir / INDEX_FILE).read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    rescale = rescalings(blocks)
    missing = sorted(set(rescale) - set(weight_map))
    if missing:
        raise ValueError(f"the index has no tensor named {missing[0]!r}")
    shards = sorted(set(weight_map.values()))
    destination.mkdir(parents=True, exist_ok=False)
    total_size = 0
    for shard in shards:
        assigned = {name for name, owner in weight_map.items() if owner == shard}
        tensors: dict[str, Any] = {}
        with safe_open(model_dir / shard, framework="pt") as source:
            if set(source.keys()) != assigned:
                raise ValueError(f"{shard} does not hold the tensors that the index assigns to it")
            for name in sorted(assigned):
                tensor = source.get_tensor(name)
                if name in rescale:
                    tensor = rescale_tensor(rescale[name], tensor)
                tensors[name] = tensor.contiguous()
        total_size += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        save_file(tensors, destination / shard, metadata=metadata)
    index["metadata"] = {**index.get("metadata", {}), "total_size": total_size}
    (destination / INDEX_FILE).write_text(json.dumps(index, indent=2), encoding="utf-8")
    return shards
