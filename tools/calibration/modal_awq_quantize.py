"""End-to-end check of `modelq quantize --format int4 --calibration awq` (ADR 0037).

Everything runs in one Modal container with a GPU, and nothing is downloaded to the local machine.

1. builds the release `modelq` CLI from this repository;
2. quantizes the pinned Qwen2.5-0.5B revision with AWQ calibration on the pinned WikiText-2 train split,
   using `modelq quantize --format int4 --calibration awq --download`;
3. checks the output's provenance metadata and that no temporary files were left behind;
4. evaluates the output with `modelq eval` on the full WikiText-2 test split.

ADR 0031 recorded 16.05708635670797 for AWQ INT4 with the embedding kept, on the same split. The
pipeline must reproduce it: the calibration windows, the search and the INT4 writer are the same.

Run with::

    py -3 -m modal run tools/calibration/modal_awq_quantize.py::main
"""

import json
import os
import pathlib
import subprocess
import sys
import time

import modal

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent

MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
ADR_0031_AWQ_EMBEDDING_KEPT = 16.05708635670797

app = modal.App("modelq-awq-quantize")

image = (
    modal.Image.from_registry("rust:1.85", add_python="3.12")
    .pip_install(
        "torch==2.9.0", "transformers>=4.50,<5", "numpy>=1.24,<3", "safetensors>=0.4,<1",
        "pyarrow", "huggingface_hub",
    )
    .add_local_dir(
        str(REPO),
        "/repo",
        ignore=modal.FilePatternMatcher("target", ".git", "**/__pycache__", "**/*.pyc", "modal_results"),
    )
)


def _run(command, cwd=None, env=None, timeout=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, check=False, timeout=timeout
    )


@app.function(image=image, gpu="L4", cpu=8, memory=49152, timeout=10800)
def verify_awq_quantize() -> dict:
    from safetensors import safe_open

    started = time.time()
    scratch = pathlib.Path("/tmp/awq-quantize")
    scratch.mkdir(parents=True, exist_ok=True)
    cache = scratch / "cache"
    modelq = "/repo/target/release/modelq"
    result: dict = {"model": {"id": MODEL, "revision": MODEL_REVISION}}

    build = _run(["cargo", "build", "--release", "--bin", "modelq"], cwd="/repo")
    if build.returncode != 0:
        raise RuntimeError("cargo build failed:\n" + build.stderr[-4000:])

    output = scratch / "int4-awq.safetensors"
    env = dict(os.environ, MODELQ_PYTHON=sys.executable)
    quantize = _run([
        modelq, "quantize", f"hf:{MODEL}", "--format", "int4", "--calibration", "awq", "--download",
        "--device", "cuda", "--revision", MODEL_REVISION, "--cache-dir", str(cache),
        "--output", str(output),
    ], cwd="/repo", env=env, timeout=7200)
    result["quantize"] = {
        "exit_code": quantize.returncode,
        "stdout": quantize.stdout[-3000:],
        "stderr": quantize.stderr[-1500:],
        "bytes": output.stat().st_size if output.exists() else None,
    }
    if quantize.returncode != 0:
        raise RuntimeError("quantize failed:\n" + quantize.stdout[-2000:] + quantize.stderr[-2000:])

    with safe_open(output, framework="np") as reader:
        metadata = reader.metadata() or {}
    result["metadata"] = {key: metadata[key] for key in metadata if key.startswith("modelq.") and key != "modelq.manifest"}
    result["leftover_temporary_files"] = sorted(path.name for path in output.parent.glob(".modelq-awq-*"))

    report_path = scratch / "eval.json"
    evaluate = _run([
        modelq, "eval", "--model", f"hf:{MODEL}", "--revision", MODEL_REVISION, "--download",
        "--device", "cuda", "--container", str(output), "--report", str(report_path),
    ], cwd="/repo", env=env, timeout=5400)
    result["eval"] = {"exit_code": evaluate.returncode, "stdout": evaluate.stdout[-1500:], "stderr": evaluate.stderr[-1000:]}
    if evaluate.returncode != 0:
        raise RuntimeError("eval failed:\n" + evaluate.stderr[-3000:])
    result["eval"]["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    result["seconds"] = round(time.time() - started, 1)
    return result


@app.local_entrypoint()
def main() -> None:
    result = verify_awq_quantize.remote()
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    (results / "awq-quantize.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["quantize"]["stdout"])
    print("bytes:", result["quantize"]["bytes"])
    print("metadata:", json.dumps(result["metadata"], indent=2))
    print("leftover temporary files:", result["leftover_temporary_files"])
    for name, variant in result["eval"]["report"]["variants"].items():
        metrics = variant["result"]
        measured = metrics["perplexity_quantized"]
        print(
            f"{name}: format {variant['format']}, orig {metrics['perplexity_original']!r}, "
            f"quant {measured!r}, increase {metrics['relative_perplexity_increase'] * 100:+.4f}%, "
            f"difference from ADR 0031 {measured - ADR_0031_AWQ_EMBEDDING_KEPT!r}"
        )
    print(f"result written to {results / 'awq-quantize.json'}")
