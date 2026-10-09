"""Real-Hub check for `modelq quantize hf:<model>` (ADR 0029, milestone M2), in Modal.

Everything runs inside Modal, never on the local machine:

1. builds the release CLI on Linux;
2. runs `modelq quantize hf:<model> --revision <commit>` against the real
   Hugging Face Hub, which downloads the weights into a cache inside Modal and
   verifies them;
3. compares the output with the container the same CLI produced earlier from a
   local copy of the same model (the `qwen2.5-0.5b-default-r6` directory in the
   `modelq-te-fixtures` volume): the two must be byte-identical;
4. runs the same command again to check that the verified cache is reused.

Run with::

    python -m modal run tools/hub_input/modal_hub_check.py
"""

import hashlib
import json
import pathlib
import subprocess
import time

import modal

_HERE = pathlib.Path(__file__).resolve()
# Inside the Modal container the script sits at /root, not in the repository.
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
BUILT_ROOT = "/built"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
REFERENCE = "qwen2.5-0.5b-default-r6/te.safetensors"

app = modal.App("modelq-hub-input-check")
volume = modal.Volume.from_name("modelq-te-fixtures", create_if_missing=True)

rust_image = modal.Image.from_registry("rust:1.85", add_python="3.12").add_local_dir(
    str(REPO),
    "/repo",
    ignore=modal.FilePatternMatcher("target", ".git", "**/__pycache__", "**/*.pyc", "modal_results"),
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], cwd: str = "/repo") -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


@app.function(image=rust_image, cpu=8, memory=32768, timeout=7200, volumes={BUILT_ROOT: volume})
def hub_quantize(model: str, revision: str) -> dict:
    started = time.time()
    build = _run(["cargo", "build", "--release", "--bin", "modelq"])
    if build.returncode != 0:
        raise RuntimeError(f"cargo build failed:\n{build.stderr[-4000:]}")
    binary = "/repo/target/release/modelq"

    scratch = pathlib.Path("/tmp/hub-check")
    cache = scratch / "cache"
    output = scratch / "te.safetensors"
    scratch.mkdir(parents=True, exist_ok=True)
    command = [
        binary, "quantize", f"hf:{model}", "--format", "nvfp4-te", "--revision", revision,
        "--cache-dir", str(cache), "--output", str(output),
    ]

    first_started = time.time()
    first = _run(command)
    first_seconds = round(time.time() - first_started, 1)
    if first.returncode != 0:
        raise RuntimeError(f"first run failed:\n{first.stdout[-3000:]}\n{first.stderr[-3000:]}")

    reference = pathlib.Path(BUILT_ROOT) / REFERENCE
    result = {
        "model": model,
        "revision": revision,
        "first_run": {"seconds": first_seconds, "stdout": first.stdout, "stderr_tail": first.stderr[-2000:]},
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "reference": str(reference),
        "reference_bytes": reference.stat().st_size if reference.is_file() else None,
        "reference_sha256": _sha256(reference) if reference.is_file() else None,
    }

    if output.exists():
        output.unlink()
    second_started = time.time()
    second = _run(command)
    result["second_run"] = {
        "seconds": round(time.time() - second_started, 1),
        "exit_code": second.returncode,
        "stdout": second.stdout,
        "stderr_tail": second.stderr[-2000:],
    }
    result["seconds"] = round(time.time() - started, 1)
    result["byte_identical_to_local_conversion"] = (
        result["output_sha256"] == result["reference_sha256"]
    )
    return result


@app.local_entrypoint()
def main(model: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION) -> None:
    result = hub_quantize.remote(model, revision)
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    path = results / "hub-input-check.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"first run: {result['first_run']['seconds']} s")
    print(result["first_run"]["stdout"])
    print(f"second run: exit {result['second_run']['exit_code']} in {result['second_run']['seconds']} s")
    print(result["second_run"]["stdout"])
    print(f"output {result['output_bytes']} bytes, sha256 {result['output_sha256']}")
    print(f"reference {result['reference_bytes']} bytes, sha256 {result['reference_sha256']}")
    print(f"byte-identical: {result['byte_identical_to_local_conversion']}")
    print(f"result written to {path}")
    if not result["byte_identical_to_local_conversion"] or result["second_run"]["exit_code"] != 0:
        raise SystemExit(1)


@app.function(image=rust_image, cpu=8, memory=32768, timeout=3600)
def cargo_test(arguments: list[str]) -> dict:
    """Runs one `cargo test` invocation and returns its full output, for diagnosis."""
    completed = _run(["cargo", "test", *arguments, "--", "--test-threads=1"])
    output = completed.stdout + completed.stderr
    return {"exit_code": completed.returncode, "output": output[-120000:]}


@app.local_entrypoint()
def debug_tests(arguments: str = "-p modelq --test hub_input") -> None:
    result = cargo_test.remote(arguments.split())
    print(f"exit code {result['exit_code']}")
    print(result["output"])


# The same toolchain as CI's stable Rust (1.99), so that lints match CI.
rust_199_image = modal.Image.from_registry("rust:1.99", add_python="3.12").add_local_dir(
    str(REPO),
    "/repo",
    ignore=modal.FilePatternMatcher("target", ".git", "**/__pycache__", "**/*.pyc", "modal_results"),
)


@app.function(image=rust_199_image, cpu=8, memory=32768, timeout=3600)
def clippy_199() -> dict:
    """Runs the CI lint command on Rust 1.99 and returns the full output."""
    version = _run(["rustc", "--version"])
    installed = _run(["rustup", "component", "add", "clippy"])
    if installed.returncode != 0:
        raise RuntimeError("could not add clippy: " + installed.stderr[-2000:])
    completed = _run(
        ["cargo", "clippy", "--workspace", "--all-targets", "--all-features", "--", "-D", "warnings"]
    )
    output = completed.stdout + completed.stderr
    return {"rustc": version.stdout.strip(), "exit_code": completed.returncode, "output": output[-120000:]}


@app.local_entrypoint()
def lint_199() -> None:
    result = clippy_199.remote()
    print(result["rustc"])
    print(f"exit code {result['exit_code']}")
    print(result["output"])
