"""Modal runner for the NVFP4 output-quality evaluation.

Evaluates the container written by `modal_runtime_proof.py::validate_model`
(stored in the `modelq-te-fixtures` volume) against the original model on
WikiText-2, on a small GPU:

    python -m modal run tools/quality_eval/modal_quality_eval.py --name qwen2.5-0.5b

Everything is downloaded inside Modal, never to the local machine: the model
files at the pinned commit (the weights' SHA-256 is checked against Hugging
Face's published value) and the WikiText-2 test split (SHA-256 checked the
same way).  The result is written to `modal_results/<name>-quality.json`.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
import urllib.request

import modal

TOOLS = pathlib.Path(__file__).resolve().parent
TE_TOOLS = TOOLS.parent / "transformer_engine_nvfp4"
REPO = TOOLS.parents[1] if len(TOOLS.parents) > 1 else TOOLS
HF = "https://huggingface.co"
BUILT_ROOT = "/built"
MODEL = "Qwen/Qwen2.5-0.5B"
DATASET = "Salesforce/wikitext"
DATASET_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
WINDOW = 2048
ALLOWED_LICENSES = {"apache-2.0", "mit"}

app = modal.App("modelq-nvfp4-quality-eval")
volume = modal.Volume.from_name("modelq-te-fixtures", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.0", "transformers>=4.50,<5", "numpy>=1.24,<3", "safetensors>=0.4,<1",
                 "pyarrow", "huggingface_hub")
    .add_local_file(str(TOOLS / "quality_eval.py"), "/root/quality_eval.py")
    .add_local_file(str(TE_TOOLS / "validate_multi.py"), "/root/transformer_engine_nvfp4/validate_multi.py")
    .add_local_file(str(TE_TOOLS / "validate.py"), "/root/transformer_engine_nvfp4/validate.py")
)


def _api(path: str) -> dict:
    with urllib.request.urlopen(f"{HF}/api/{path}") as response:
        return json.load(response)


def _download_verified(url: str, destination: pathlib.Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            handle.write(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"sha256 mismatch for {url}: {digest.hexdigest()} != {expected_sha256}")


@app.function(image=image, gpu="L4", timeout=3600, volumes={BUILT_ROOT: volume})
def evaluate(name: str = "qwen2.5-0.5b") -> dict:
    """Downloads the pinned model and dataset, then compares original and NVFP4 on WikiText-2."""
    import sys

    import pyarrow.parquet as pq
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/transformer_engine_nvfp4")
    import quality_eval as qe

    started = time.time()
    volume.reload()
    container = pathlib.Path(BUILT_ROOT) / name / "te.safetensors"
    if not container.is_file():
        raise RuntimeError(f"{container} not found; run modal_runtime_proof.py::validate_model first")

    # --- model files, pinned and verified -------------------------------------------------
    model_api = _api(f"models/{MODEL}?blobs=true")
    revision = model_api["sha"]
    license_id = (model_api.get("cardData") or {}).get("license")
    if license_id not in ALLOWED_LICENSES or model_api.get("gated"):
        raise RuntimeError(f"{MODEL} must be an ungated Apache-2.0 or MIT model")
    weights = next(f for f in model_api["siblings"] if f["rfilename"] == "model.safetensors")
    local = pathlib.Path("/tmp/model")
    local.mkdir()
    for filename in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                     "vocab.json", "merges.txt"):
        hf_hub_download(MODEL, filename, revision=revision, local_dir=local)
    _download_verified(f"{HF}/{MODEL}/resolve/{revision}/model.safetensors",
                       local / "model.safetensors", weights["lfs"]["sha256"])

    # --- dataset, pinned and verified -----------------------------------------------------
    data_api = _api(f"datasets/{DATASET}?blobs=true")
    data_revision = data_api["sha"]
    data_license = (data_api.get("cardData") or {}).get("license")
    entry = next(f for f in data_api["siblings"] if f["rfilename"] == DATASET_FILE)
    parquet = pathlib.Path("/tmp/wikitext2-test.parquet")
    _download_verified(f"{HF}/datasets/{DATASET}/resolve/{data_revision}/{DATASET_FILE}",
                       parquet, entry["lfs"]["sha256"])
    text = "\n\n".join(pq.read_table(parquet).column("text").to_pylist())

    tokenizer = AutoTokenizer.from_pretrained(local)
    token_ids = tokenizer(text, return_tensors=None)["input_ids"]
    windows = qe.make_windows(token_ids, WINDOW)

    # --- the two models, both float32 -----------------------------------------------------
    device = torch.device("cuda")
    base = AutoModelForCausalLM.from_pretrained(local, torch_dtype=torch.float32).to(device)
    quantized = AutoModelForCausalLM.from_pretrained(local, torch_dtype=torch.float32).to(device)
    substitution = qe.substitute_nvfp4_weights(quantized, container, torch)
    result = qe.evaluate_pair(base, quantized, windows, torch, device)
    result["relative_perplexity_increase"] = qe.relative_perplexity_increase(result)

    # A pipeline that is broken (wrong text, wrong weights) would give an absurd
    # baseline; fail loudly instead of reporting a meaningless comparison.
    if not 3.0 < result["perplexity_original"] < 60.0:
        raise RuntimeError(f"implausible baseline perplexity {result['perplexity_original']}")

    return {
        "model": {"id": MODEL, "revision": revision, "license": license_id,
                  "weights_sha256": weights["lfs"]["sha256"]},
        "dataset": {"id": DATASET, "file": DATASET_FILE, "revision": data_revision,
                    "license": data_license, "sha256": entry["lfs"]["sha256"]},
        "text": {"characters": len(text), "tokens": len(token_ids), "window": WINDOW,
                 "windows": len(windows), "tokens_scored": result["positions"]},
        "container": str(container),
        "substitution": substitution.summary(),
        "result": result,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "gpu": torch.cuda.get_device_name(), "dtype": "float32"},
        "seconds": round(time.time() - started, 1),
    }


@app.local_entrypoint()
def main(name: str = "qwen2.5-0.5b") -> None:
    result = evaluate.remote(name)
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    path = results / f"{name}-quality.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics = result["result"]
    print(json.dumps({k: result[k] for k in ("model", "dataset", "text", "substitution", "environment")}, indent=2))
    print(f"perplexity original  : {metrics['perplexity_original']:.4f}")
    print(f"perplexity NVFP4     : {metrics['perplexity_quantized']:.4f}"
          f"  ({metrics['relative_perplexity_increase'] * 100:+.2f}%)")
    print(f"mean KL(orig||NVFP4) : {metrics['mean_kl_original_to_quantized']:.6f} nats/token")
    print(f"top-1 agreement      : {metrics['top1_agreement'] * 100:.2f}%")
    print(f"result written to {path}")
