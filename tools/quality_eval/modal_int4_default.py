"""End-to-end check of the INT4 default that keeps the vocabulary matrices at source precision (ADR 0036).

Everything runs in one Modal container with a GPU, and nothing is downloaded to the local machine.

1. builds the release `modelq` CLI from this repository;
2. quantizes the pinned Qwen2.5-0.5B revision with `modelq quantize --format int4`, using the default;
3. evaluates the result with `modelq eval --model hf:...` on the full WikiText-2 split.

ADR 0031 recorded 18.191818421875144 for INT4 with the embedding kept at its source values on the
same split. The default must reproduce that number, and the file must grow by the embedding's
BF16 size rather than by more.

Run with::

    py -3 -m modal run tools/quality_eval/modal_int4_default.py::main
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

app = modal.App("modelq-int4-default")

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


@app.function(image=image, gpu="L4", cpu=8, memory=49152, timeout=7200)
def verify_int4_default() -> dict:
    started = time.time()
    scratch = pathlib.Path("/tmp/int4-default")
    scratch.mkdir(parents=True, exist_ok=True)
    cache = scratch / "cache"
    modelq = "/repo/target/release/modelq"
    result: dict = {"model": {"id": MODEL, "revision": MODEL_REVISION}}

    build = _run(["cargo", "build", "--release", "--bin", "modelq"], cwd="/repo")
    if build.returncode != 0:
        raise RuntimeError("cargo build failed:\n" + build.stderr[-4000:])

    output = scratch / "int4.safetensors"
    quantize = _run([
        modelq, "quantize", f"hf:{MODEL}", "--format", "int4", "--revision", MODEL_REVISION,
        "--cache-dir", str(cache), "--output", str(output),
    ], timeout=3600)
    result["quantize"] = {
        "exit_code": quantize.returncode,
        "bytes": output.stat().st_size if output.exists() else None,
        "stdout": quantize.stdout[-1500:],
        "stderr": quantize.stderr[-800:],
    }
    if quantize.returncode != 0:
        raise RuntimeError("quantize failed:\n" + quantize.stderr[-2000:])

    report_path = scratch / "report.json"
    env = dict(os.environ, MODELQ_PYTHON=sys.executable)
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
    result = verify_int4_default.remote()
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    (results / "int4-default.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["quantize"]["stdout"])
    print("bytes:", result["quantize"]["bytes"])
    for name, variant in result["eval"]["report"]["variants"].items():
        metrics = variant["result"]
        print(
            f"{name}: format {variant['format']}, orig {metrics['perplexity_original']!r}, "
            f"quant {metrics['perplexity_quantized']!r}, increase {metrics['relative_perplexity_increase'] * 100:+.4f}%"
        )
    print(f"result written to {results / 'int4-default.json'}")
