"""AWQ calibration for `modelq quantize --format int4 --calibration awq` (ADR 0037).

Runs the activation-aware search of `awq.py` on WikiText-2 calibration windows and writes a rescaled,
unquantized SafeTensors checkpoint. `modelq quantize` then quantizes that checkpoint with its ordinary
INT4 writer, so the result is an ordinary ModelQ INT4 file. The calibration settings go to a JSON
report, which `modelq quantize` records in the output file.

Usage (from the repository root)::

    py -3 tools/calibration/modelq_awq.py --model <model dir> \\
        --calibration-data <train.parquet> --output <scaled.safetensors> --report <report.json>

With ``--download`` and no ``--calibration-data``, the pinned WikiText-2 train split is fetched and
its SHA-256 is checked. Without either, nothing is downloaded and the script refuses to run.

The windows are evenly spaced across the train split and recorded by offset, as in ADR 0031.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "quality_eval"))
import awq  # noqa: E402  (path set above)

DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TRAIN_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
TRAIN_SHA256 = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
TRANSFORM = "awq-v1"
CALIBRATION_WINDOWS = 32
CALIBRATION_LENGTH = 512
BATCH_SIZE = 4
DEFAULT_GROUP_SIZE = 128


def calibration_offsets(token_count: int, windows: int, length: int) -> list[int]:
    """Evenly spaced window starts across the token stream, first at 0 and last at the final full window."""
    if windows < 2:
        raise ValueError("calibration needs at least two windows")
    if token_count < length:
        raise ValueError(f"the calibration text has {token_count} tokens, fewer than one {length}-token window")
    last_start = token_count - length
    return [round(index * last_start / (windows - 1)) for index in range(windows)]


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_train_split() -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    path = pathlib.Path(
        hf_hub_download(DATASET, TRAIN_FILE, repo_type="dataset", revision=DATASET_REVISION)
    )
    if _sha256(path) != TRAIN_SHA256:
        raise SystemExit("sha256 mismatch for the WikiText-2 train split")
    return path


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calibrate(
    model_dir: pathlib.Path,
    dataset: pathlib.Path,
    output: pathlib.Path,
    group_size: int,
    device_name: str,
) -> dict[str, Any]:
    import pyarrow.parquet as pq
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    device = _resolve_device(torch, device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    text = "\n\n".join(pq.read_table(dataset).column("text").to_pylist())
    token_ids = tokenizer(text, return_tensors=None)["input_ids"]
    offsets = calibration_offsets(len(token_ids), CALIBRATION_WINDOWS, CALIBRATION_LENGTH)
    windows = [token_ids[start : start + CALIBRATION_LENGTH] for start in offsets]

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
    blocks = awq.calibrate(model, windows, group_size=group_size, device=device, batch_size=BATCH_SIZE)
    summary = awq.summarize(blocks)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with safe_open(model_dir / "model.safetensors", framework="pt") as source:
        awq.write_scaled_checkpoint(source, blocks, output, {"modelq.transform": TRANSFORM})

    return {
        "transform": TRANSFORM,
        "dataset": {"id": DATASET, "revision": DATASET_REVISION, "file": TRAIN_FILE, "sha256": TRAIN_SHA256},
        "calibration": {
            "windows": CALIBRATION_WINDOWS,
            "window_length": CALIBRATION_LENGTH,
            "offsets": offsets,
            "tokens": CALIBRATION_WINDOWS * CALIBRATION_LENGTH,
            "group_size": group_size,
            "batch_size": BATCH_SIZE,
            "alpha_grid_values": len(awq.ALPHA_GRID),
        },
        "summary": summary,
        "device": str(device),
        "torch": torch.__version__,
        "seconds": round(time.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AWQ calibration for modelq quantize (ADR 0037).")
    parser.add_argument("--model", type=pathlib.Path, required=True, help="a local model directory")
    parser.add_argument("--calibration-data", type=pathlib.Path, help="local WikiText-2 train Parquet file")
    parser.add_argument("--download", action="store_true", help="fetch the pinned WikiText-2 train split if no file is given")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="where to write the rescaled checkpoint")
    parser.add_argument("--report", type=pathlib.Path, required=True, help="where to write the JSON report")
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)

    if not (args.model / "model.safetensors").is_file():
        raise SystemExit(f"{args.model} is not a local model directory with a single model.safetensors")
    if args.calibration_data is not None:
        dataset = args.calibration_data
    elif args.download:
        dataset = _fetch_train_split()
    else:
        raise SystemExit("without --download, pass --calibration-data with a local WikiText-2 train Parquet file")
    if not dataset.is_file():
        raise SystemExit(f"calibration data not found: {dataset}")

    report = calibrate(args.model, dataset, args.output, args.group_size, args.device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"AWQ calibration: {report['calibration']['windows']} windows of {report['calibration']['window_length']} tokens "
        f"on {report['device']}, {report['seconds']} s; report written to {args.report}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
