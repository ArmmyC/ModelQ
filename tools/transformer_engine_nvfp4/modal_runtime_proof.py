"""One-shot Modal runner for the multi-matrix Transformer Engine runtime proof.

Builds (and caches) an image with Transformer Engine 2.19.0 compiled for
Blackwell, starts a single-use NVIDIA B200 container, and runs
`validate_multi.py runtime` on every container in a fixtures directory.  The
image recipe is the one recorded in ADR 0013 for the single-matrix proof.

    MODELQ_FIXTURES=<dir> python -m modal run tools/transformer_engine_nvfp4/modal_runtime_proof.py

`<dir>` must hold `reference.safetensors` and one or more containers: files
named `*.safetensors` other than the reference, and subdirectories that contain
`model.safetensors.index.json`.  Generate them with `te_multi_matrix_fixture`
and `modelq quantize --format nvfp4-te` (see the tool README).  The result is
written to `<dir>/modal_result.json`.

To build the fixtures on Linux as well (the Rust example and CLI run on a CPU
container from this repository's source, and write the containers plus the
Rust-generated reference to a Modal volume), then prove them on the GPU:

    python -m modal run tools/transformer_engine_nvfp4/modal_runtime_proof.py::build_and_prove

Pass `--name NAME` to choose the volume directory and `--sweep` for the
shape-sweep fixture used to find which shapes the runtime accepts.  Results go
to `modal_results/<name>.json`.

Running this spends Modal GPU time; the TE build itself runs on CPU during
image build and is cached by Modal for later runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
import urllib.request

import modal

TOOLS = pathlib.Path(__file__).resolve().parent
# Inside the container this file lives at /root, which has no repository above it;
# the repository path is only needed on the machine that starts the run.
REPO = TOOLS.parents[1] if len(TOOLS.parents) > 1 else TOOLS
FIXTURES = pathlib.Path(os.environ.get("MODELQ_FIXTURES", "")).resolve() if os.environ.get("MODELQ_FIXTURES") else None
REMOTE_FIXTURES = "/root/fixtures"
BUILT_ROOT = "/built"
CUDNN_ROOT = "/usr/local/lib/python3.12/site-packages/nvidia/cudnn"
RESULT_MARKER = "RESULT_JSON: "

app = modal.App("modelq-te-multi-matrix-proof")
volume = modal.Volume.from_name("modelq-te-fixtures", create_if_missing=True)

# A CPU image that compiles this repository on Linux.  The source is mounted at
# container start, so the cached toolchain layer survives code changes.
rust_image = modal.Image.from_registry("rust:1.85", add_python="3.12").add_local_dir(
    str(REPO),
    "/repo",
    ignore=modal.FilePatternMatcher("target", ".git", "**/__pycache__", "**/*.pyc", "modal_results"),
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "cmake", "ninja-build", "git")
    .env({"MAX_JOBS": "8", "NVTE_FRAMEWORK": "pytorch", "PYTHONUNBUFFERED": "1"})
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install pybind11 packaging ninja cmake",
        "python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 'torch==2.9.0'",
        "CXX=g++ NVTE_WITH_NCCL_EP=0 python -m pip install --no-build-isolation "
        "'numpy>=1.24,<3' 'safetensors>=0.4,<1' 'transformer-engine[pytorch]==2.19.0'",
    )
    # Mounted at container start rather than baked in, so editing the
    # validator never invalidates the cached Transformer Engine build.
    .add_local_file(str(TOOLS / "validate.py"), "/root/validate.py")
    .add_local_file(str(TOOLS / "validate_multi.py"), "/root/validate_multi.py")
)
if FIXTURES is not None:
    image = image.add_local_dir(str(FIXTURES), REMOTE_FIXTURES)


def _containers(root: pathlib.Path) -> list[pathlib.Path]:
    found = []
    for entry in sorted(root.iterdir()):
        if entry.name == "reference.safetensors" or entry.name == "source.safetensors":
            continue
        if entry.is_file() and entry.suffix == ".safetensors":
            found.append(entry)
        elif entry.is_dir() and (entry / "model.safetensors.index.json").is_file():
            found.append(entry)
    return found


@app.function(image=rust_image, cpu=8, memory=16384, timeout=3600, volumes={BUILT_ROOT: volume})
def build_fixtures(name: str = "final", sweep: bool = False) -> dict:
    """Builds the fixture and containers on Linux into the volume directory `name`."""

    def run(*command: str) -> str:
        completed = subprocess.run(
            command, cwd="/repo", capture_output=True, text=True, check=False,
            env={**os.environ, "CARGO_TERM_COLOR": "never"},
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{' '.join(command)} failed:\n{completed.stdout}\n{completed.stderr[-4000:]}")
        return completed.stdout

    out = pathlib.Path(BUILT_ROOT) / name
    if out.exists():
        raise RuntimeError(f"{out} already exists; choose another --name")
    scratch = pathlib.Path("/tmp/fixture")
    generated = run(
        "cargo", "run", "-q", "--release", "-p", "modelq-io", "--example",
        "te_multi_matrix_fixture", "--", str(scratch), *(["--sweep"] if sweep else []),
    )
    source = scratch / "source.safetensors"
    convert = ["cargo", "run", "-q", "--release", "--bin", "modelq", "--", "quantize", str(source),
               "--format", "nvfp4-te"]
    report = run(*convert, "--output", str(scratch / "te.safetensors"))
    if not sweep:
        run(*convert, "--max-shard-size", "2MB", "--output", str(scratch / "te-sharded"))
    # The source stays out of the proof directory so only containers are mounted.
    (scratch / "source.safetensors").rename(pathlib.Path("/tmp/source.safetensors"))
    shutil.copytree(scratch, out)
    volume.commit()
    return {"directory": str(out), "listing": sorted(p.name for p in out.iterdir()),
            "fixture_log": generated, "convert_report": report}


ALLOWED_LICENSES = {"apache-2.0", "mit"}
HF = "https://huggingface.co"


@app.function(image=rust_image, cpu=8, memory=32768, timeout=7200, volumes={BUILT_ROOT: volume})
def fetch_and_convert(model: str, name: str) -> dict:
    """Downloads a model's `model.safetensors` and converts it, on Linux in Modal.

    The download is pinned to the repository's current commit and its SHA-256
    is checked against the value Hugging Face publishes, and only permissively
    licensed models are accepted.  The converted containers and the reference
    (re-quantized from the source weights by the native quantizer) are written
    to the volume directory `name`; the source weights are not kept.
    """
    started = time.time()
    with urllib.request.urlopen(f"{HF}/api/models/{model}?blobs=true") as response:
        api = json.load(response)
    revision = api["sha"]
    license_id = (api.get("cardData") or {}).get("license")
    if license_id not in ALLOWED_LICENSES:
        raise RuntimeError(f"{model} has license {license_id!r}; only {sorted(ALLOWED_LICENSES)} are accepted")
    if api.get("gated"):
        raise RuntimeError(f"{model} is gated")
    entry = next((f for f in api["siblings"] if f["rfilename"] == "model.safetensors"), None)
    if entry is None:
        raise RuntimeError(f"{model} has no single model.safetensors file")
    expected = entry["lfs"]["sha256"]

    source = pathlib.Path("/tmp/model.safetensors")
    digest = hashlib.sha256()
    with urllib.request.urlopen(f"{HF}/{model}/resolve/{revision}/model.safetensors") as response, source.open("wb") as handle:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            handle.write(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"sha256 mismatch for {model}@{revision}: {digest.hexdigest()} != {expected}")
    size = source.stat().st_size

    def run(*command: str) -> str:
        completed = subprocess.run(
            command, cwd="/repo", capture_output=True, text=True, check=False,
            env={**os.environ, "CARGO_TERM_COLOR": "never"},
        )
        if completed.returncode != 0:
            raise RuntimeError(" ".join(command) + " failed:" + chr(10) + completed.stdout[-2000:] + chr(10) + completed.stderr[-4000:])
        return completed.stdout

    out = pathlib.Path(BUILT_ROOT) / name
    if out.exists():
        raise RuntimeError(f"{out} already exists; choose another --name")
    scratch = pathlib.Path("/tmp/model-out")
    scratch.mkdir()
    reference_log = run(
        "cargo", "run", "-q", "--release", "-p", "modelq-io", "--example",
        "te_reference_from_source", "--", str(source), str(scratch / "reference.safetensors"),
    )
    convert = ["cargo", "run", "-q", "--release", "--bin", "modelq", "--", "quantize", str(source),
               "--format", "nvfp4-te"]
    single_report = run(*convert, "--output", str(scratch / "te.safetensors"))
    run(*convert, "--max-shard-size", "128MB", "--output", str(scratch / "te-sharded"))
    shutil.copytree(scratch, out)
    volume.commit()
    return {
        "model": model, "revision": revision, "license": license_id, "source_bytes": size,
        "source_sha256": expected, "directory": str(out), "listing": sorted(p.name for p in out.iterdir()),
        "reference_log": reference_log.strip(), "convert_report": single_report,
        "seconds": round(time.time() - started, 1),
    }


@app.function(image=rust_image, cpu=4, memory=16384, timeout=7200, volumes={BUILT_ROOT: volume})
def convert_scale_search(model: str, base: str, radii: str) -> dict:
    """Converts one model once per block-scale search radius, on Linux in Modal.

    The model is downloaded at its pinned commit and hash-checked exactly as in
    `fetch_and_convert`.  For each radius in the comma-separated `radii` it
    writes the volume directory `<base>-r<radius>` holding `te.safetensors`
    and the matching `reference.safetensors` (derived from the source weights
    with the same scale rule), and returns the CLI's conversion report.
    """
    started = time.time()
    with urllib.request.urlopen(f"{HF}/api/models/{model}?blobs=true") as response:
        api = json.load(response)
    revision = api["sha"]
    license_id = (api.get("cardData") or {}).get("license")
    if license_id not in ALLOWED_LICENSES or api.get("gated"):
        raise RuntimeError(f"{model} must be an ungated Apache-2.0 or MIT model")
    entry = next(f for f in api["siblings"] if f["rfilename"] == "model.safetensors")
    source = pathlib.Path("/tmp/model.safetensors")
    digest = hashlib.sha256()
    with urllib.request.urlopen(f"{HF}/{model}/resolve/{revision}/model.safetensors") as response, source.open("wb") as handle:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            handle.write(chunk)
    if digest.hexdigest() != entry["lfs"]["sha256"]:
        raise RuntimeError(f"sha256 mismatch for {model}@{revision}")

    def run(*command: str) -> str:
        completed = subprocess.run(
            command, cwd="/repo", capture_output=True, text=True, check=False,
            env={**os.environ, "CARGO_TERM_COLOR": "never"},
        )
        if completed.returncode != 0:
            raise RuntimeError(" ".join(command) + " failed:" + chr(10) + completed.stdout[-2000:] + chr(10) + completed.stderr[-4000:])
        return completed.stdout

    reports = {}
    for radius in [int(part) for part in radii.split(",")]:
        out = pathlib.Path(BUILT_ROOT) / f"{base}-r{radius}"
        if out.exists():
            raise RuntimeError(f"{out} already exists")
        scratch = pathlib.Path(f"/tmp/search-{radius}")
        scratch.mkdir()
        reference_log = run(
            "cargo", "run", "-q", "--release", "-p", "modelq-io", "--example", "te_reference_from_source",
            "--", str(source), str(scratch / "reference.safetensors"), "--scale-search", str(radius),
        )
        report = run(
            "cargo", "run", "-q", "--release", "--bin", "modelq", "--", "quantize", str(source),
            "--format", "nvfp4-te", "--scale-search", str(radius), "--output", str(scratch / "te.safetensors"),
        )
        shutil.copytree(scratch, out)
        volume.commit()
        reports[str(radius)] = {"directory": str(out), "reference_log": reference_log.strip(), "convert_report": report}
    return {"model": model, "revision": revision, "radii": reports, "seconds": round(time.time() - started, 1)}


@app.function(image=rust_image, cpu=8, memory=16384, timeout=3600)
def verify_repo() -> dict:
    """Runs the formatting check and the whole Rust test suite on Linux (Rust 1.85, CPU only)."""
    env = {**os.environ, "CARGO_TERM_COLOR": "never"}
    steps = {}
    for label, command in {
        "components": ["rustup", "component", "add", "rustfmt"],
        "fmt": ["cargo", "fmt", "--check"],
        "test": ["cargo", "test", "--workspace", "--no-fail-fast"],
    }.items():
        completed = subprocess.run(command, cwd="/repo", capture_output=True, text=True, env=env, check=False)
        steps[label] = {"exit_code": completed.returncode, "tail": (completed.stdout + completed.stderr)[-6000:]}
        passed = failed = ignored = 0
        for line in completed.stdout.splitlines():
            if line.startswith("test result:"):
                fields = line.replace(";", " ").split()
                passed += int(fields[3])
                failed += int(fields[5])
                ignored += int(fields[7])
        if label == "test":
            steps[label].update({"passed": passed, "failed": failed, "ignored": ignored})
    return steps


@app.function(image=rust_image, cpu=8, memory=16384, timeout=3600)
def bench_scale_search() -> str:
    """Runs the block-scale search cost benchmark on Linux in Modal (CPU only)."""
    completed = subprocess.run(
        ["cargo", "bench", "--bench", "nvfp4_scale_search"], cwd="/repo",
        capture_output=True, text=True, check=False, env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout[-2000:] + chr(10) + completed.stderr[-3000:])
    return completed.stdout


@app.function(
    image=image,
    gpu="B200",
    timeout=3600,
    startup_timeout=300,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    volumes={BUILT_ROOT: volume},
)
def prove(root: str = REMOTE_FIXTURES) -> dict:
    """Runs the runtime proof for every container in `root` on one B200."""
    if root.startswith(BUILT_ROOT):
        volume.reload()
    cudnn_lib = pathlib.Path(CUDNN_ROOT) / "lib"
    if not cudnn_lib.is_dir():
        raise RuntimeError(f"PyTorch wheel cuDNN library directory not found: {cudnn_lib}")
    # The base image also ships a system cuDNN; mixing the two caused a symbol
    # error in the Task 28 run, so point everything at the PyTorch wheel's copy.
    env = dict(os.environ)
    env["CUDNN_HOME"] = CUDNN_ROOT
    env["CUDNN_PATH"] = CUDNN_ROOT
    env["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    root = pathlib.Path(root)
    reference = root / "reference.safetensors"
    runs = []
    for container in _containers(root):
        completed = subprocess.run(
            [
                "python", "/root/validate_multi.py", "runtime",
                "--container", str(container),
                "--reference", str(reference),
            ],
            env=env, capture_output=True, text=True, check=False,
        )
        outcome = None
        for line in completed.stdout.splitlines():
            if line.startswith(RESULT_MARKER):
                outcome = json.loads(line[len(RESULT_MARKER):])
        runs.append(
            {
                "container": container.name,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr[-4000:],
                "outcome": outcome,
            }
        )
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    return {"runs": runs, "nvidia_smi": smi.stdout}


@app.local_entrypoint()
def main() -> None:
    if FIXTURES is None or not (FIXTURES / "reference.safetensors").is_file():
        raise SystemExit("set MODELQ_FIXTURES to a directory containing reference.safetensors")
    if not _containers(FIXTURES):
        raise SystemExit(f"no containers found in {FIXTURES}")
    result = prove.remote()
    (FIXTURES / "modal_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    failed = False
    for run in result["runs"]:
        print(f"== {run['container']}: exit code {run['exit_code']}")
        print(run["stdout"])
        if run["stderr"].strip():
            print("-- stderr --")
            print(run["stderr"])
        failed = failed or run["exit_code"] != 0
    print(f"result written to {FIXTURES / 'modal_result.json'}")
    if failed:
        raise SystemExit(1)


@app.local_entrypoint()
def build_and_prove(name: str = "final", sweep: bool = False) -> None:
    """Builds the fixtures on Linux, then runs the runtime proof on a B200."""
    built = build_fixtures.remote(name, sweep)
    print(f"fixtures built in volume directory {built['directory']}: {built['listing']}")
    print(built["fixture_log"])
    result = prove.remote(built["directory"])
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    (results / f"{name}.json").write_text(
        json.dumps({"fixtures": built, **result}, indent=2), encoding="utf-8"
    )
    failed = False
    for run in result["runs"]:
        print(f"== {run['container']}: exit code {run['exit_code']}")
        print(run["stdout"])
        if run["stderr"].strip():
            print("-- stderr --")
            print(run["stderr"])
        failed = failed or run["exit_code"] != 0
    print(f"result written to {results / (name + '.json')}")
    if failed:
        raise SystemExit(1)


@app.local_entrypoint()
def verify() -> None:
    """Runs `cargo fmt --check` and `cargo test --workspace` on Linux in Modal (CPU only)."""
    steps = verify_repo.remote()
    for label, step in steps.items():
        print(f"== {label}: exit code {step['exit_code']}", {k: v for k, v in step.items() if k in ("passed", "failed", "ignored")})
        if step["exit_code"] != 0:
            print(step["tail"])
    if any(step["exit_code"] != 0 for step in steps.values()):
        raise SystemExit(1)


@app.local_entrypoint()
def validate_model(model: str = "Qwen/Qwen2.5-0.5B", name: str = "qwen2.5-0.5b") -> None:
    """Downloads `model` inside Modal, converts it, and proves every exported matrix on a B200."""
    built = fetch_and_convert.remote(model, name)
    print(f"{built['model']}@{built['revision']} ({built['license']}, {built['source_bytes']} bytes) "
          f"converted in {built['seconds']} s: {built['listing']}")
    print(built["reference_log"])
    print(built["convert_report"][-1800:])
    result = prove.remote(built["directory"])
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    (results / f"{name}.json").write_text(
        json.dumps({"model": built, **result}, indent=2), encoding="utf-8"
    )
    failed = False
    for run in result["runs"]:
        outcome = run["outcome"]
        print(f"== {run['container']}: exit code {run['exit_code']}")
        if outcome:
            matrices = outcome["matrices"]
            passed = sum(1 for m in matrices if m["status"] == "pass")
            print(f"   {passed}/{len(matrices)} matrices passed; environment {outcome['environment']}")
            for record in matrices:
                if record["status"] != "pass":
                    print(f"   FAIL {record['name']} {record['shape']}: {record.get('detail')}")
        if run["stderr"].strip():
            print("-- stderr --")
            print(run["stderr"])
        failed = failed or run["exit_code"] != 0
    print(f"result written to {results / (name + '.json')}")
    if failed:
        raise SystemExit(1)


@app.local_entrypoint()
def search_experiment(model: str = "Qwen/Qwen2.5-0.5B", base: str = "qwen2.5-0.5b", radii: str = "1,2,4,6,8,12") -> None:
    """Converts `model` once per block-scale search radius into volume directories `<base>-r<radius>`."""
    result = convert_scale_search.remote(model, base, radii)
    print(f"{result['model']}@{result['revision']} converted in {result['seconds']} s")
    for radius, report in result["radii"].items():
        lines = [l for l in report["convert_report"].splitlines()
                 if l.strip().startswith(("Block scales", "Output bytes", "Max MSE", "Max MAE", "Lowest SQNR", "Tensors:"))]
        print(f"-- radius {radius}: {report['directory']}")
        print(chr(10).join(lines))


@app.local_entrypoint()
def bench_search() -> None:
    """Prints the cost of the block-scale search relative to the default rule."""
    print(bench_scale_search.remote())


@app.local_entrypoint()
def prove_existing(name: str) -> None:
    """Runs the B200 runtime proof on containers already in the volume directory `name`."""
    result = prove.remote(f"{BUILT_ROOT}/{name}")
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    (results / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    failed = False
    for run in result["runs"]:
        outcome = run["outcome"]
        print(f"== {run['container']}: exit code {run['exit_code']}")
        if outcome:
            matrices = outcome["matrices"]
            print(f"   {sum(1 for m in matrices if m['status'] == 'pass')}/{len(matrices)} matrices passed; {outcome['environment']}")
            for record in matrices:
                if record["status"] != "pass":
                    print(f"   FAIL {record['name']} {record['shape']}: {record.get('detail')}")
        failed = failed or run["exit_code"] != 0
    print(f"result written to {results / (name + '.json')}")
    if failed:
        raise SystemExit(1)

