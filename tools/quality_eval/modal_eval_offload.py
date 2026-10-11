"""Verification of the offloaded evaluation (ADR 0040), in Modal.

Everything runs in one Modal container with a GPU, and nothing is downloaded to the local machine.

1. builds the release `modelq` CLI from this repository;
2. Qwen2.5-0.5B (pinned): AWQ INT4 with `modelq quantize --calibration awq --download`, then
   `modelq eval` on the full WikiText-2 test split. The quantized perplexity must be ADR 0031's
   anchor 16.05708635670797 exactly: moving a model between devices must not change the numbers;
3. Qwen2.5-3B (pinned, two shards): the same quantization, then `modelq eval` on 20 windows and on
   the full split, which needs both models on the device at once without offloading.

A stage that fails does not discard the stages before it: the result is written with the error.

Run with::

    py -3 -m modal run tools/quality_eval/modal_eval_offload.py::main
"""

import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

import modal

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent

SMALL_MODEL = "Qwen/Qwen2.5-0.5B"
SMALL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
ADR_0031_AWQ_EMBEDDING_KEPT = 16.05708635670797
LARGE_MODEL = "Qwen/Qwen2.5-3B"
LARGE_REVISION = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"

app = modal.App("modelq-eval-offload")

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


def _run(command, env=None, timeout=None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd="/repo", env=env, capture_output=True, text=True, check=False, timeout=timeout)


def _tail(completed: subprocess.CompletedProcess) -> dict:
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-3000:],
        "stderr": completed.stderr[-2000:],
    }


@app.function(image=image, gpu="L4", cpu=8, memory=49152, timeout=14400)
def verify_eval_offload() -> dict:
    started = time.time()
    scratch = pathlib.Path("/tmp/eval-offload")
    scratch.mkdir(parents=True, exist_ok=True)
    cache = scratch / "cache"
    modelq = "/repo/target/release/modelq"
    env = dict(os.environ, MODELQ_PYTHON=sys.executable)
    result: dict = {
        "small_model": {"id": SMALL_MODEL, "revision": SMALL_REVISION},
        "large_model": {"id": LARGE_MODEL, "revision": LARGE_REVISION},
        "anchor_adr_0031": ADR_0031_AWQ_EMBEDDING_KEPT,
    }

    try:
        build = _run(["cargo", "build", "--release", "--bin", "modelq"])
        if build.returncode != 0:
            raise RuntimeError("cargo build failed:\n" + build.stderr[-4000:])

        def quantize(model: str, revision: str, output: pathlib.Path) -> dict:
            completed = _run(
                [
                    modelq, "quantize", f"hf:{model}", "--revision", revision, "--cache-dir", str(cache),
                    "--format", "int4", "--calibration", "awq", "--download", "--device", "cuda",
                    "--output", str(output),
                ],
                env=env, timeout=7200,
            )
            entry = _tail(completed)
            entry["bytes"] = output.stat().st_size if output.exists() else None
            if completed.returncode != 0:
                raise RuntimeError(f"quantize {model} failed:\n" + completed.stdout[-2000:] + completed.stderr[-2000:])
            return entry

        def evaluate(model: str, revision: str, container: pathlib.Path, report: pathlib.Path, windows=None) -> dict:
            arguments = [
                modelq, "eval", "--model", f"hf:{model}", "--revision", revision, "--download",
                "--device", "cuda", "--container", str(container), "--report", str(report),
            ]
            if windows is not None:
                arguments += ["--max-windows", str(windows)]
            completed = _run(arguments, env=env, timeout=7200)
            entry = _tail(completed)
            if completed.returncode != 0:
                raise RuntimeError(f"eval {model} failed:\n" + completed.stderr[-3000:])
            entry["report"] = json.loads(report.read_text(encoding="utf-8"))
            return entry

        small_container = scratch / "int4-0.5b.safetensors"
        result["small_quantize"] = quantize(SMALL_MODEL, SMALL_REVISION, small_container)
        result["small_eval"] = evaluate(SMALL_MODEL, SMALL_REVISION, small_container, scratch / "eval-0.5b.json")

        large_container = scratch / "int4-3b.safetensors"
        result["large_quantize"] = quantize(LARGE_MODEL, LARGE_REVISION, large_container)
        result["large_eval_20_windows"] = evaluate(
            LARGE_MODEL, LARGE_REVISION, large_container, scratch / "eval-3b-20.json", windows=20
        )
        result["large_eval_full"] = evaluate(LARGE_MODEL, LARGE_REVISION, large_container, scratch / "eval-3b-full.json")
    except Exception:
        result["error"] = traceback.format_exc()[-6000:]
    result["seconds"] = round(time.time() - started, 1)
    return result


@app.local_entrypoint()
def main() -> None:
    result = verify_eval_offload.remote()
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    path = results / "eval-offload.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key in ("small_eval", "large_eval_20_windows", "large_eval_full"):
        if key in result and "report" in result[key]:
            report = result[key]["report"]
            for name, variant in report["variants"].items():
                metrics = variant["result"]
                print(
                    f"{key} {name}: windows {metrics['windows']}, orig {metrics['perplexity_original']!r}, "
                    f"quant {metrics['perplexity_quantized']!r}, peak device memory "
                    f"{report['environment'].get('peak_device_memory_bytes')}"
                )
    if "small_eval" in result and "report" in result["small_eval"]:
        variant = next(iter(result["small_eval"]["report"]["variants"].values()))
        difference = variant["result"]["perplexity_quantized"] - ADR_0031_AWQ_EMBEDDING_KEPT
        print(f"0.5B difference from ADR 0031 anchor: {difference!r}")
    print(f"result written to {path}")
    if "error" in result:
        print(result["error"])
        raise SystemExit(1)
