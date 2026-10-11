"""End-to-end check of AWQ calibration on sharded checkpoints (ADR 0038), in Modal.

Everything runs in one Modal container with a GPU, and nothing is downloaded to the local machine.

1. builds the release `modelq` CLI from this repository;
2. re-shards the pinned Qwen2.5-0.5B checkpoint, which is one file, into several shards with the
   same tensors and a Hub-style index;
3. quantizes both layouts with `modelq quantize --format int4 --calibration awq --download`;
4. checks that every tensor of the two INT4 outputs is identical, and that no temporary files are left;
5. evaluates the sharded output on the full WikiText-2 test split: ADR 0031's anchor must be reproduced;
6. quantizes the real two-shard Hub checkpoint Qwen/Qwen2.5-3B with the same command, and runs a short
   evaluation on it.

A stage that fails does not discard the stages before it: the result is written with the error.

Run with::

    py -3 -m modal run tools/calibration/modal_awq_sharded.py::main
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback

import modal

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent

MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
ADR_0031_AWQ_EMBEDDING_KEPT = 16.05708635670797
LARGE_MODEL = "Qwen/Qwen2.5-3B"
LARGE_REVISION = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
SHARD_BUDGET_BYTES = 300 * 1024 * 1024
SINGLE_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"

app = modal.App("modelq-awq-sharded")

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


def _tensors(path) -> dict:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as reader:
        return {name: reader.get_tensor(name) for name in reader.keys()}


def _metadata(path) -> dict:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as reader:
        metadata = reader.metadata() or {}
    return {key: metadata[key] for key in metadata if key.startswith("modelq.") and key != "modelq.manifest"}


def _reshard(source_file: pathlib.Path, destination: pathlib.Path, budget: int) -> dict:
    """Splits a one-file checkpoint into shards of about `budget` bytes, with a Hub-style index.

    The tensors are copied unchanged. A shard may exceed the budget only when one tensor does.
    """
    from safetensors.torch import save_file

    tensors = _tensors(source_file)
    groups: list[list[str]] = []
    current: list[str] = []
    used = 0
    for name in sorted(tensors):
        size = tensors[name].numel() * tensors[name].element_size()
        if current and used + size > budget:
            groups.append(current)
            current, used = [], 0
        current.append(name)
        used += size
    if current:
        groups.append(current)

    destination.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    for number, group in enumerate(groups, start=1):
        shard = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        save_file({name: tensors[name].contiguous() for name in group}, destination / shard, metadata={"format": "pt"})
        for name in group:
            weight_map[name] = shard
            total_size += tensors[name].numel() * tensors[name].element_size()
    (destination / INDEX_FILE).write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
    return {"shards": len(groups), "tensors": len(weight_map), "total_size": total_size}


def _compare(first: dict, second: dict) -> dict:
    import torch

    mismatches = []
    for name in sorted(set(first) | set(second)):
        if name not in first or name not in second:
            mismatches.append(name)
            continue
        a, b = first[name], second[name]
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            mismatches.append(name)
    return {
        "tensors": len(first),
        "same_names": set(first) == set(second),
        "identical": not mismatches,
        "mismatches": mismatches[:10],
    }


def _quantize_result(completed: subprocess.CompletedProcess, output: pathlib.Path, scratch: pathlib.Path) -> dict:
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-1500:],
        "bytes": output.stat().st_size if output.exists() else None,
        "leftover_temporary_files": sorted(path.name for path in scratch.glob(".modelq-awq-*")),
    }


@app.function(image=image, gpu="L4", cpu=8, memory=49152, timeout=14400)
def verify_sharded_awq() -> dict:
    from huggingface_hub import snapshot_download

    started = time.time()
    scratch = pathlib.Path("/tmp/awq-sharded")
    scratch.mkdir(parents=True, exist_ok=True)
    cache = scratch / "cache"
    modelq = "/repo/target/release/modelq"
    result: dict = {
        "model": {"id": MODEL, "revision": MODEL_REVISION},
        "large_model": {"id": LARGE_MODEL, "revision": LARGE_REVISION},
    }

    try:
        build = _run(["cargo", "build", "--release", "--bin", "modelq"], cwd="/repo")
        if build.returncode != 0:
            raise RuntimeError("cargo build failed:\n" + build.stderr[-4000:])

        env = dict(os.environ, MODELQ_PYTHON=sys.executable)
        single_dir = pathlib.Path(snapshot_download(MODEL, revision=MODEL_REVISION, cache_dir=str(cache)))
        sharded_dir = scratch / "sharded"
        if sharded_dir.exists():
            shutil.rmtree(sharded_dir)
        result["resharded"] = _reshard(single_dir / SINGLE_FILE, sharded_dir, SHARD_BUDGET_BYTES)
        for path in single_dir.iterdir():
            if path.is_file() and path.name != SINGLE_FILE:
                shutil.copy(path, sharded_dir / path.name)

        outputs = {}
        for label, model_dir in (("single-file", single_dir), ("sharded", sharded_dir)):
            output = scratch / f"int4-{label}.safetensors"
            completed = _run(
                [
                    modelq, "quantize", str(model_dir), "--format", "int4", "--calibration", "awq",
                    "--download", "--device", "cuda", "--output", str(output),
                ],
                cwd="/repo", env=env, timeout=7200,
            )
            result[label] = _quantize_result(completed, output, scratch)
            if completed.returncode != 0:
                raise RuntimeError(f"quantize ({label}) failed:\n" + completed.stdout[-2000:] + completed.stderr[-2000:])
            outputs[label] = output

        single_tensors = _tensors(outputs["single-file"])
        sharded_tensors = _tensors(outputs["sharded"])
        result["comparison"] = _compare(single_tensors, sharded_tensors)
        del single_tensors, sharded_tensors
        result["metadata"] = {
            "single-file": _metadata(outputs["single-file"]),
            "sharded": _metadata(outputs["sharded"]),
        }
        calibration = json.loads(result["metadata"]["sharded"]["modelq.calibration"])
        result["checkpoint_record"] = calibration.get("checkpoint")

        report_path = scratch / "eval-sharded.json"
        evaluate = _run(
            [
                modelq, "eval", "--model", f"hf:{MODEL}", "--revision", MODEL_REVISION, "--download",
                "--device", "cuda", "--container", str(outputs["sharded"]), "--report", str(report_path),
            ],
            cwd="/repo", env=env, timeout=5400,
        )
        result["eval"] = {"exit_code": evaluate.returncode, "stdout": evaluate.stdout[-1500:], "stderr": evaluate.stderr[-1000:]}
        if evaluate.returncode != 0:
            raise RuntimeError("eval failed:\n" + evaluate.stderr[-3000:])
        result["eval"]["report"] = json.loads(report_path.read_text(encoding="utf-8"))

        large_output = scratch / "int4-qwen2.5-3b.safetensors"
        large = _run(
            [
                modelq, "quantize", f"hf:{LARGE_MODEL}", "--revision", LARGE_REVISION, "--cache-dir", str(cache),
                "--format", "int4", "--calibration", "awq", "--download", "--device", "cuda",
                "--output", str(large_output),
            ],
            cwd="/repo", env=env, timeout=10800,
        )
        result["large"] = _quantize_result(large, large_output, scratch)
        if large.returncode != 0:
            raise RuntimeError("quantize (Qwen2.5-3B) failed:\n" + large.stdout[-2000:] + large.stderr[-2000:])

        large_eval_report = scratch / "eval-3b.json"
        large_eval = _run(
            [
                modelq, "eval", "--model", f"hf:{LARGE_MODEL}", "--revision", LARGE_REVISION, "--download",
                "--device", "cuda", "--container", str(large_output), "--max-windows", "20",
                "--report", str(large_eval_report),
            ],
            cwd="/repo", env=env, timeout=5400,
        )
        result["large_eval"] = {"exit_code": large_eval.returncode, "stderr": large_eval.stderr[-1000:]}
        if large_eval.returncode == 0:
            result["large_eval"]["report"] = json.loads(large_eval_report.read_text(encoding="utf-8"))
    except Exception:
        result["error"] = traceback.format_exc()[-6000:]
    result["seconds"] = round(time.time() - started, 1)
    return result


@app.local_entrypoint()
def main() -> None:
    result = verify_sharded_awq.remote()
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    path = results / "awq-sharded.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("re-sharded:", result.get("resharded"))
    for label in ("single-file", "sharded"):
        if label in result:
            print(f"{label} quantize exit", result[label]["exit_code"], "bytes", result[label]["bytes"])
    if "comparison" in result:
        print("comparison:", result["comparison"])
    print("checkpoint record (sharded):", result.get("checkpoint_record"))
    if "eval" in result and "report" in result["eval"]:
        for name, variant in result["eval"]["report"]["variants"].items():
            metrics = variant["result"]
            measured = metrics["perplexity_quantized"]
            print(
                f"{name}: orig {metrics['perplexity_original']!r}, quant {measured!r}, "
                f"increase {metrics['relative_perplexity_increase'] * 100:+.4f}%, "
                f"difference from ADR 0031 {measured - ADR_0031_AWQ_EMBEDDING_KEPT!r}"
            )
    if "large" in result:
        print("Qwen2.5-3B quantize exit", result["large"]["exit_code"], "bytes", result["large"]["bytes"])
        print(result["large"]["stdout"])
    if "large_eval" in result and "report" in result["large_eval"]:
        for name, variant in result["large_eval"]["report"]["variants"].items():
            print(f"Qwen2.5-3B {name} (20 windows): {variant['result']}")
    print(f"result written to {path}")
    if "error" in result:
        print(result["error"])
        raise SystemExit(1)
