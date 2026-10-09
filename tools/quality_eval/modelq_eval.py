"""Local quality evaluation for ModelQ outputs (ADR 0028, milestone M1).

Runs on the user's own machine, with no cloud service.  It compares an original
causal language model with copies whose matrices were replaced by the values
decoded from one or more ModelQ Transformer Engine NVFP4 containers, on
WikiText-2, and writes a JSON report.  The metrics and the windowing are the
same as the Modal evaluation that produced ADR 0024 and ADR 0026, so the
numbers can be compared directly.

Usage (from the repository root)::

    py -3 tools/quality_eval/modelq_eval.py \\
        --model Qwen/Qwen2.5-0.5B --download \\
        --container out/te.safetensors --report out/quality.json

Downloads happen only when ``--download`` is given, and then only for the
pinned revisions below; each file's SHA-256 is checked against the Hub's
metadata before it is used.  Without ``--download``, ``--model`` must be a
local directory and ``--dataset`` a local Parquet file.

The output is a measurement, not a certification (see ``quality_eval.py``).
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
import quality_eval as qe  # noqa: E402  (path set above)

HF = "https://huggingface.co"
WINDOW = 2048
ALLOWED_LICENSES = {"apache-2.0", "mit"}

# Pinned to the revisions measured in ADR 0024 and ADR 0026.
PINNED_MODEL = {
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
}
DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
DATASET_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_model(model_id: str, revision: str) -> pathlib.Path:
    """Downloads the pinned model files into the Hugging Face cache and verifies the weights."""
    from huggingface_hub import hf_hub_download, model_info

    info = model_info(model_id, revision=revision, files_metadata=True)
    license_id = (info.card_data.license if info.card_data else None) or None
    if license_id not in ALLOWED_LICENSES or getattr(info, "gated", False):
        raise SystemExit(f"{model_id} must be an ungated Apache-2.0 or MIT model (license: {license_id})")
    weights_entry = next(s for s in info.siblings if s.rfilename == "model.safetensors")
    local = None
    for filename in ("model.safetensors", *TOKENIZER_FILES):
        local = pathlib.Path(hf_hub_download(model_id, filename, revision=revision))
    directory = local.parent
    if weights_entry.lfs is not None and _sha256(directory / "model.safetensors") != weights_entry.lfs.sha256:
        raise SystemExit(f"sha256 mismatch for {model_id}@{revision} model.safetensors")
    return directory


def _fetch_dataset() -> pathlib.Path:
    """Downloads the pinned WikiText-2 test split and verifies it."""
    from huggingface_hub import hf_hub_download

    path = pathlib.Path(
        hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION)
    )
    if _sha256(path) != DATASET_SHA256:
        raise SystemExit("sha256 mismatch for the WikiText-2 test split")
    return path


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model_dir: pathlib.Path,
    containers: dict[str, pathlib.Path],
    dataset: pathlib.Path,
    device_name: str,
    max_windows: int | None,
) -> dict[str, Any]:
    """Scores each container against the original model on the same windows."""
    import pyarrow.parquet as pq
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    device = _resolve_device(torch, device_name)
    text = "\n\n".join(pq.read_table(dataset).column("text").to_pylist())
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    token_ids = tokenizer(text, return_tensors=None)["input_ids"]
    windows = qe.make_windows(token_ids, WINDOW)
    if max_windows is not None:
        windows = windows[:max_windows]

    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
    variants: dict[str, Any] = {}
    for name, container in containers.items():
        quantized = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
        substitution = qe.substitute_nvfp4_weights(quantized, container, torch)
        result = qe.evaluate_pair(base, quantized, windows, torch, device)
        result["relative_perplexity_increase"] = qe.relative_perplexity_increase(result)
        variants[name] = {
            "container": str(container),
            "substitution": substitution.summary(),
            "result": result,
        }
        del quantized
    baseline = next(iter(variants.values()))["result"]["perplexity_original"]
    if not 3.0 < baseline < 60.0:
        raise SystemExit(f"implausible baseline perplexity {baseline:.3f}; check the model and dataset")
    return {
        "text": {"characters": len(text), "tokens": len(token_ids), "window": WINDOW, "windows": len(windows)},
        "variants": variants,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "dtype": "float32",
        },
        "seconds": round(time.time() - started, 1),
    }


def _print_table(report: dict[str, Any]) -> None:
    print(
        f"{'container':<28}{'ppl orig':>10}{'ppl quant':>11}{'increase':>10}"
        f"{'KL nats/tok':>13}{'top-1 agree':>13}{'rel.err':>9}"
    )
    for name, variant in report["variants"].items():
        metrics = variant["result"]
        print(
            f"{name:<28}{metrics['perplexity_original']:>10.4f}{metrics['perplexity_quantized']:>11.4f}"
            f"{metrics['relative_perplexity_increase'] * 100:>9.2f}%"
            f"{metrics['mean_kl_original_to_quantized']:>13.6f}"
            f"{metrics['top1_agreement'] * 100:>12.2f}%"
            f"{variant['substitution']['relative_frobenius_error_mean'] * 100:>8.2f}%"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ModelQ NVFP4 containers on WikiText-2 locally.")
    parser.add_argument("--model", required=True, help="pinned model id (with --download) or a local directory")
    parser.add_argument("--download", action="store_true", help="allow downloading the pinned model and dataset")
    parser.add_argument("--dataset", type=pathlib.Path, help="local WikiText-2 test Parquet file")
    parser.add_argument(
        "--container",
        action="append",
        type=pathlib.Path,
        required=True,
        metavar="PATH",
        help="a ModelQ Transformer Engine NVFP4 container; repeat to compare several",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-windows", type=int, default=None, help="score only the first N windows")
    parser.add_argument("--report", type=pathlib.Path, required=True, help="where to write the JSON report")
    args = parser.parse_args(argv)

    if args.download:
        if args.model not in PINNED_MODEL:
            raise SystemExit(f"--download supports only the pinned models: {', '.join(PINNED_MODEL)}")
        model_dir = _fetch_model(args.model, PINNED_MODEL[args.model])
        dataset = args.dataset or _fetch_dataset()
    else:
        model_dir = pathlib.Path(args.model)
        if not (model_dir / "model.safetensors").is_file():
            raise SystemExit(f"{model_dir} is not a local model directory (no model.safetensors)")
        if args.dataset is None:
            raise SystemExit("without --download, pass --dataset with a local WikiText-2 test Parquet file")
        dataset = args.dataset
    for container in args.container:
        if not container.is_file():
            raise SystemExit(f"container not found: {container}")

    containers = {container.stem: container for container in args.container}
    report = evaluate(model_dir, containers, dataset, args.device, args.max_windows)
    report["model"] = {"id": args.model, "revision": PINNED_MODEL.get(args.model)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_table(report)
    print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
