"""End-to-end check of `modelq eval` on every container `modelq quantize` writes (ADR 0035).

Everything runs in one Modal container with a GPU, and nothing is downloaded to the local machine.

1. builds the release `modelq` CLI from this repository;
2. quantizes the pinned Qwen2.5-0.5B revision with `modelq quantize hf:...` to every format that
   can be evaluated: int8, int4, nvfp4, nvfp4-te, gguf-q8_0 and gguf-q4_0;
3. runs `modelq eval --model hf:... --download` once with all six containers, on the first 20
   windows (comparable with ADR 0032 and ADR 0033) and on the full WikiText-2 test split.

The report of each run is returned and saved under modal_results/.

Run with::

    py -3 -m modal run tools/quality_eval/modal_eval_containers.py::main
"""

import json
import pathlib
import subprocess
import time

import modal

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent

MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
FORMATS = [
    ("int8", "safetensors"),
    ("int4", "safetensors"),
    ("nvfp4", "safetensors"),
    ("nvfp4-te", "safetensors"),
    ("gguf-q8_0", "gguf"),
    ("gguf-q4_0", "gguf"),
]

app = modal.App("modelq-eval-containers")

image = (
    modal.Image.from_registry("rust:1.85", add_python="3.12")
    .pip_install(
        "torch==2.9.0", "transformers>=4.50,<5", "numpy>=1.24,<3", "safetensors>=0.4,<1",
        "pyarrow", "huggingface_hub", "gguf",
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


@app.function(image=image, gpu="L4", cpu=8, memory=49152, timeout=14400)
def verify_eval() -> dict:
    import os
    import sys

    started = time.time()
    scratch = pathlib.Path("/tmp/eval-containers")
    containers_dir = scratch / "containers"
    containers_dir.mkdir(parents=True, exist_ok=True)
    cache = scratch / "cache"
    modelq = "/repo/target/release/modelq"
    result: dict = {"model": {"id": MODEL, "revision": MODEL_REVISION}, "quantize": {}, "eval": {}}

    build = _run(["cargo", "build", "--release", "--bin", "modelq"], cwd="/repo")
    if build.returncode != 0:
        raise RuntimeError("cargo build failed:\n" + build.stderr[-4000:])

    paths = []
    for fmt, extension in FORMATS:
        output = containers_dir / f"{fmt}.{extension}"
        step = _run([
            modelq, "quantize", f"hf:{MODEL}", "--format", fmt, "--revision", MODEL_REVISION,
            "--cache-dir", str(cache), "--output", str(output),
        ], timeout=3600)
        result["quantize"][fmt] = {
            "exit_code": step.returncode,
            "bytes": output.stat().st_size if output.exists() else None,
            "stdout": step.stdout[-600:],
            "stderr": step.stderr[-600:],
        }
        if step.returncode != 0:
            raise RuntimeError(f"quantize {fmt} failed:\n" + step.stderr[-2000:])
        paths.append(output)

    env = dict(os.environ, MODELQ_PYTHON=sys.executable)
    for label, extra in (("first-20-windows", ["--max-windows", "20"]), ("full-split", [])):
        report_path = scratch / f"report-{label}.json"
        command = [
            modelq, "eval", "--model", f"hf:{MODEL}", "--revision", MODEL_REVISION, "--download",
            "--device", "cuda", "--report", str(report_path), *extra,
        ]
        for path in paths:
            command += ["--container", str(path)]
        step = _run(command, cwd="/repo", env=env, timeout=7200)
        result["eval"][label] = {
            "exit_code": step.returncode,
            "stdout": step.stdout[-3000:],
            "stderr": step.stderr[-2000:],
        }
        if step.returncode != 0:
            raise RuntimeError(f"eval {label} failed:\n" + step.stderr[-3000:])
        result["eval"][label]["report"] = json.loads(report_path.read_text(encoding="utf-8"))

    result["seconds"] = round(time.time() - started, 1)
    return result


@app.local_entrypoint()
def main() -> None:
    result = verify_eval.remote()
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    (results / "eval-containers.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for label, run in result["eval"].items():
        report = run["report"]
        print(f"== {label}  (exit {run['exit_code']})")
        for name, variant in report["variants"].items():
            metrics = variant["result"]
            print(
                f"  {name:<14} {variant['format']:<10} "
                f"orig {metrics['perplexity_original']:.6f}  quant {metrics['perplexity_quantized']:.6f}  "
                f"increase {metrics['relative_perplexity_increase'] * 100:+.3f}%  "
                f"top1 {metrics['top1_agreement'] * 100:.2f}%  matrices {variant['substitution']['matrices_replaced']}"
            )
    print(f"result written to {results / 'eval-containers.json'}")
