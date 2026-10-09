"""AWQ versus data-free INT4 on Qwen2.5-0.5B, run in Modal (ADR 0031, milestone M4).

Everything runs inside Modal; no model or dataset file is downloaded to the
local machine. In one GPU container it:

1. downloads the pinned model and the pinned WikiText-2 train and test splits,
   verifying every file's SHA-256 against the Hub metadata;
2. calibrates AWQ scales on evenly spaced windows of the *train* split and
   writes the rescaled checkpoint;
3. builds the Rust CLI and quantizes two checkpoints to INT4 with group size 128:
   the original (data-free) and the rescaled one (AWQ);
4. evaluates on the same 146 windows of the *test* split: the original float
   model, the rescaled but unquantized checkpoint (an equivalence check), the
   data-free INT4 file, and the AWQ INT4 file;
5. cross-checks the Rust output against the Python quantizer: the decoded
   weights of each file must equal the Python quantization of its source.

Run with::

    python -m modal run tools/calibration/modal_awq.py
"""

import hashlib
import json
import pathlib
import subprocess
import time

import modal

TOOLS = pathlib.Path(__file__).resolve().parent
QUALITY = TOOLS.parent / "quality_eval"
TE_TOOLS = TOOLS.parent / "transformer_engine_nvfp4"
_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
HF = "https://huggingface.co"
MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TRAIN_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
TEST_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
ALLOWED_LICENSES = {"apache-2.0", "mit"}
GROUP_SIZE = 128
CALIBRATION_WINDOWS = 32
CALIBRATION_LENGTH = 512
EVALUATION_LENGTH = 2048

app = modal.App("modelq-awq-calibration")

image = (
    modal.Image.from_registry("rust:1.85", add_python="3.12")
    .pip_install(
        "torch==2.9.0", "transformers>=4.50,<5", "numpy>=1.24,<3", "safetensors>=0.4,<1",
        "pyarrow", "huggingface_hub", "pytest",
    )
    .add_local_file(str(QUALITY / "quality_eval.py"), "/root/quality_eval.py")
    .add_local_file(str(TE_TOOLS / "validate_multi.py"), "/root/transformer_engine_nvfp4/validate_multi.py")
    .add_local_file(str(TE_TOOLS / "validate.py"), "/root/transformer_engine_nvfp4/validate.py")
    .add_local_file(str(TOOLS / "awq.py"), "/root/awq.py")
    .add_local_dir(
        str(REPO),
        "/repo",
        ignore=modal.FilePatternMatcher("target", ".git", "**/__pycache__", "**/*.pyc", "modal_results"),
    )
)


def _api(path: str) -> dict:
    import urllib.request

    with urllib.request.urlopen(f"{HF}/api/{path}") as response:
        return json.load(response)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(url: str, destination: pathlib.Path, expected: str) -> None:
    import urllib.request

    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            handle.write(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"sha256 mismatch for {url}")


def _run(command: list[str], cwd: str = "/repo") -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


@app.function(image=image, gpu="L4", cpu=8, memory=49152, timeout=10800)
def run_experiment() -> dict:
    import sys

    import pyarrow.parquet as pq
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/transformer_engine_nvfp4")
    import awq
    import quality_eval as qe

    started = time.time()
    device = torch.device("cuda")
    scratch = pathlib.Path("/tmp/awq")
    scratch.mkdir(parents=True, exist_ok=True)

    # --- model and dataset: pinned, verified --------------------------------------------
    model_api = _api(f"models/{MODEL}/revision/{MODEL_REVISION}?blobs=true")
    license_id = (model_api.get("cardData") or {}).get("license")
    if license_id not in ALLOWED_LICENSES or model_api.get("gated"):
        raise RuntimeError(f"{MODEL} must be an ungated Apache-2.0 or MIT model")
    weights_entry = next(f for f in model_api["siblings"] if f["rfilename"] == "model.safetensors")
    model_dir = scratch / "model"
    model_dir.mkdir()
    for filename in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                     "vocab.json", "merges.txt"):
        hf_hub_download(MODEL, filename, revision=MODEL_REVISION, local_dir=model_dir)
    weights = model_dir / "model.safetensors"
    _download_verified(f"{HF}/{MODEL}/resolve/{MODEL_REVISION}/model.safetensors", weights,
                       weights_entry["lfs"]["sha256"])

    data_api = _api(f"datasets/{DATASET}/revision/{DATASET_REVISION}?blobs=true")
    entries = {f["rfilename"]: f for f in data_api["siblings"]}
    train_path = scratch / "train.parquet"
    test_path = scratch / "test.parquet"
    _download_verified(f"{HF}/datasets/{DATASET}/resolve/{DATASET_REVISION}/{TRAIN_FILE}", train_path,
                       entries[TRAIN_FILE]["lfs"]["sha256"])
    _download_verified(f"{HF}/datasets/{DATASET}/resolve/{DATASET_REVISION}/{TEST_FILE}", test_path,
                       entries[TEST_FILE]["lfs"]["sha256"])

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    train_text = "\n\n".join(pq.read_table(train_path).column("text").to_pylist())
    test_text = "\n\n".join(pq.read_table(test_path).column("text").to_pylist())
    train_ids = tokenizer(train_text, return_tensors=None)["input_ids"]
    test_ids = tokenizer(test_text, return_tensors=None)["input_ids"]
    evaluation_windows = qe.make_windows(test_ids, EVALUATION_LENGTH)

    # Calibration windows: evenly spaced across the train split, recorded by offset.
    last_start = len(train_ids) - CALIBRATION_LENGTH
    starts = [round(index * last_start / (CALIBRATION_WINDOWS - 1)) for index in range(CALIBRATION_WINDOWS)]
    calibration_windows = [train_ids[start : start + CALIBRATION_LENGTH] for start in starts]

    # --- AWQ calibration -----------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
    blocks = awq.calibrate(model, calibration_windows, group_size=GROUP_SIZE, device=device, batch_size=4)
    summary = awq.summarize(blocks)
    scaled_path = scratch / "awq-scaled.safetensors"
    with safe_open(weights, framework="pt") as source:
        awq.write_scaled_checkpoint(source, blocks, scaled_path, {"modelq.transform": "awq-v1"})
    del model
    torch.cuda.empty_cache()

    # --- Rust CLI: data-free and AWQ INT4 -----------------------------------------------
    build = _run(["cargo", "build", "--release", "--bin", "modelq"])
    if build.returncode != 0:
        raise RuntimeError(f"cargo build failed:\n{build.stderr[-4000:]}")
    binary = "/repo/target/release/modelq"
    data_free_path = scratch / "data-free-int4.safetensors"
    awq_path = scratch / "awq-int4.safetensors"
    quantize_logs = {}
    for label, source_path, output_path in (
        ("data-free", weights, data_free_path),
        ("awq", scaled_path, awq_path),
    ):
        result = _run([binary, "quantize", str(source_path), "--format", "int4",
                       "--group-size", str(GROUP_SIZE), "--output", str(output_path)])
        if result.returncode != 0:
            raise RuntimeError(f"{label} quantize failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        quantize_logs[label] = result.stdout[-4000:]

    # --- evaluation on the same windows ---------------------------------------------------
    def fresh_model():
        return AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)

    base = fresh_model()
    variants = {}
    checks = {}
    # The NVFP4 default preserves the embedding and the output head; the same
    # policy is evaluated here from the same files, keeping those at the model's values.
    keep = ("model.embed_tokens.weight", "lm_head.weight")
    for label, loader in (
        ("scaled-unquantized", lambda m: qe.substitute_checkpoint_weights(m, scaled_path, torch)),
        ("data-free-int4", lambda m: qe.substitute_lowbit_weights(m, data_free_path, torch)),
        ("data-free-int4-embedding-kept",
         lambda m: qe.substitute_lowbit_weights(m, data_free_path, torch, keep_original=keep)),
        ("awq-int4", lambda m: qe.substitute_lowbit_weights(m, awq_path, torch)),
        ("awq-int4-embedding-kept",
         lambda m: qe.substitute_lowbit_weights(m, awq_path, torch, keep_original=keep)),
    ):
        variant = fresh_model()
        report = loader(variant)
        result = qe.evaluate_pair(base, variant, evaluation_windows, torch, device)
        result["relative_perplexity_increase"] = qe.relative_perplexity_increase(result)
        variants[label] = {"substitution": report.summary(), "result": result}
        del variant
        torch.cuda.empty_cache()

    # Cross-check: the Rust output must decode to the Python quantization of its source.
    with safe_open(weights, framework="pt") as source_reader, safe_open(scaled_path, framework="pt") as scaled_reader:
        for label, source_reader_handle, container in (
            ("data-free-int4", source_reader, data_free_path),
            ("awq-int4", scaled_reader, awq_path),
        ):
            model_copy = fresh_model()
            qe.substitute_lowbit_weights(model_copy, container, torch)
            parameters = dict(model_copy.named_parameters())
            worst = 0.0
            compared = 0
            for name in sorted(parameters):
                if not name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                                      "gate_proj.weight", "up_proj.weight", "down_proj.weight")):
                    continue
                source_tensor = source_reader_handle.get_tensor(name).to(torch.float32)
                expected = awq.pseudo_quantize(source_tensor.to(device), GROUP_SIZE, bits=4)
                actual = parameters[name].detach()
                worst = max(worst, float((actual - expected).abs().max()))
                compared += 1
            checks[label] = {"matrices_compared": compared, "max_abs_difference": worst}
            del model_copy
            torch.cuda.empty_cache()

    return {
        "model": {"id": MODEL, "revision": MODEL_REVISION, "license": license_id,
                  "weights_sha256": weights_entry["lfs"]["sha256"]},
        "dataset": {"id": DATASET, "revision": DATASET_REVISION,
                    "train_file": TRAIN_FILE, "train_sha256": entries[TRAIN_FILE]["lfs"]["sha256"],
                    "test_file": TEST_FILE, "test_sha256": entries[TEST_FILE]["lfs"]["sha256"],
                    "license": (data_api.get("cardData") or {}).get("license")},
        "calibration": {
            "source": f"{DATASET}@{DATASET_REVISION} {TRAIN_FILE}",
            "windows": CALIBRATION_WINDOWS,
            "window_length": CALIBRATION_LENGTH,
            "window_starts": starts,
            "train_tokens": len(train_ids),
            "group_size": GROUP_SIZE,
            "search": summary,
        },
        "evaluation": {
            "source": f"{DATASET}@{DATASET_REVISION} {TEST_FILE}",
            "windows": len(evaluation_windows),
            "window_length": EVALUATION_LENGTH,
            "test_tokens": len(test_ids),
        },
        "variants": variants,
        "cross_check": checks,
        "quantize_logs": quantize_logs,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "gpu": torch.cuda.get_device_name(), "dtype": "float32"},
        "seconds": round(time.time() - started, 1),
    }


@app.local_entrypoint()
def main() -> None:
    result = run_experiment.remote()
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    path = results / "awq-qwen2.5-0.5b.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{'variant':<22}{'ppl orig':>10}{'ppl quant':>11}{'increase':>10}{'KL nats/tok':>13}{'top-1 agree':>13}")
    for name, variant in result["variants"].items():
        metrics = variant["result"]
        print(
            f"{name:<22}{metrics['perplexity_original']:>10.4f}{metrics['perplexity_quantized']:>11.4f}"
            f"{metrics['relative_perplexity_increase'] * 100:>9.3f}%"
            f"{metrics['mean_kl_original_to_quantized']:>13.6f}{metrics['top1_agreement'] * 100:>12.2f}%"
        )
    for name, check in result["cross_check"].items():
        print(f"cross-check {name}: {check['matrices_compared']} matrices, max |difference| {check['max_abs_difference']:.3e}")
    print(f"result written to {path}")


@app.function(image=image, cpu=8, memory=32768, timeout=3600)
def diagnose_int4() -> dict:
    """Compares the Rust INT4 codes and scales with the Python quantizer, tensor by tensor."""
    import sys

    import numpy as np
    import torch
    from safetensors import safe_open

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/transformer_engine_nvfp4")
    import awq
    import quality_eval as qe

    scratch = pathlib.Path("/tmp/diag")
    scratch.mkdir(parents=True, exist_ok=True)
    model_entry = _api(f"models/{MODEL}/revision/{MODEL_REVISION}?blobs=true")
    weights_entry = next(f for f in model_entry["siblings"] if f["rfilename"] == "model.safetensors")
    weights = scratch / "model.safetensors"
    _download_verified(f"{HF}/{MODEL}/resolve/{MODEL_REVISION}/model.safetensors", weights,
                       weights_entry["lfs"]["sha256"])

    build = _run(["cargo", "build", "--release", "--bin", "modelq"])
    if build.returncode != 0:
        raise RuntimeError("cargo build failed: " + build.stderr[-4000:])
    output = scratch / "dfree.safetensors"
    result = _run(["/repo/target/release/modelq", "quantize", str(weights), "--format", "int4",
                   "--group-size", str(GROUP_SIZE), "--output", str(output)])
    if result.returncode != 0:
        raise RuntimeError("quantize failed: " + result.stdout[-2000:] + " " + result.stderr[-2000:])

    report = {"tensors": {}}
    with safe_open(weights, framework="pt") as source, safe_open(output, framework="pt") as file:
        manifest = qe.read_lowbit_manifest(file.metadata())
        for name, entry in sorted(manifest.tensors.items()):
            if entry["action"] != "quantized":
                continue
            matrix = source.get_tensor(name).to(torch.float32)
            rows, columns = matrix.shape
            elements = entry["elements"]
            packed = file.get_tensor(entry["qdata_name"]).numpy()
            file_scales = file.get_tensor(entry["scale_name"]).numpy()
            file_codes = qe.unpack_codes(packed, elements, 4)
            file_signed = np.where(file_codes >= 8, file_codes - 16, file_codes)

            grouped = matrix.reshape(rows, columns // GROUP_SIZE, GROUP_SIZE)
            max_abs = grouped.abs().amax(dim=-1, keepdim=True)
            python_scales = torch.where(max_abs > 0, max_abs / 7, torch.ones_like(max_abs)).reshape(-1).numpy()
            python_codes = awq.round_half_away(grouped / torch.from_numpy(python_scales).reshape(
                rows, columns // GROUP_SIZE, 1)).clamp(-7, 7).reshape(-1).to(torch.int64).numpy()

            scale_mismatch = int(np.sum(file_scales != python_scales))
            code_mismatch_positions = np.nonzero(file_signed != python_codes)[0]
            first = []
            for index in code_mismatch_positions[:3]:
                group = int(index) // GROUP_SIZE
                first.append({
                    "index": int(index),
                    "value": float(matrix.reshape(-1)[index]),
                    "group_scale_file": float(file_scales[group]),
                    "group_scale_python": float(python_scales[group]),
                    "code_file": int(file_signed[index]),
                    "code_python": int(python_codes[index]),
                    "value_over_scale": float(matrix.reshape(-1)[index]) / float(python_scales[group]),
                })
            report["tensors"][name] = {
                "scale_mismatches": scale_mismatch,
                "code_mismatches": int(len(code_mismatch_positions)),
                "first": first,
            }
    mismatched = {k: v for k, v in report["tensors"].items() if v["code_mismatches"] or v["scale_mismatches"]}
    report["summary"] = {
        "tensors_checked": len(report["tensors"]),
        "tensors_with_mismatch": len(mismatched),
        "total_code_mismatches": sum(v["code_mismatches"] for v in report["tensors"].values()),
        "total_scale_mismatches": sum(v["scale_mismatches"] for v in report["tensors"].values()),
    }
    report["mismatched_examples"] = dict(list(mismatched.items())[:3])
    return report


@app.local_entrypoint()
def diagnose() -> None:
    report = diagnose_int4.remote()
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["mismatched_examples"], indent=2)[:6000])


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1800)
def gpu_versus_cpu_quantization() -> dict:
    """Counts elements where the GPU and CPU quantize-dequantize disagree on random matrices."""
    import sys

    import torch

    sys.path.insert(0, "/root")
    import awq

    generator = torch.Generator().manual_seed(0)
    total = 0
    differing_elements = 0
    worst = 0.0
    for _ in range(64):
        matrix = torch.randn(4864, 896, generator=generator) * 0.05
        on_cpu = awq.pseudo_quantize(matrix, GROUP_SIZE, bits=4)
        on_gpu = awq.pseudo_quantize(matrix.cuda(), GROUP_SIZE, bits=4).cpu()
        differences = (on_cpu - on_gpu).abs()
        differing_elements += int((differences > 0).sum())
        worst = max(worst, float(differences.max()))
        total += matrix.numel()
    return {"elements": total, "differing_elements": differing_elements, "max_difference": worst,
            "gpu": torch.cuda.get_device_name()}


@app.local_entrypoint()
def gpu_check() -> None:
    print(gpu_versus_cpu_quantization.remote())


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1800)
def gpu_operation_probe() -> dict:
    """Finds which elementwise operation of the quantizer differs between GPU and CPU."""
    import torch

    generator = torch.Generator().manual_seed(0)
    values = torch.randn(1_000_000, generator=generator) * 3.0
    scale = torch.full_like(values, 0.4375)
    cpu = {}
    gpu = {}
    for label, device in (("cpu", "cpu"), ("gpu", "cuda")):
        v = values.to(device)
        s = scale.to(device)
        divided = v / s
        truncated = torch.trunc(divided)
        fraction = divided - truncated
        rounded = truncated + torch.sign(fraction) * (torch.abs(fraction) >= 0.5).to(divided.dtype)
        target = cpu if label == "cpu" else gpu
        target.update({"divided": divided.cpu(), "truncated": truncated.cpu(),
                       "fraction": fraction.cpu(), "rounded": rounded.cpu()})
    report = {}
    for key in ("divided", "truncated", "fraction", "rounded"):
        differing = (cpu[key] != gpu[key]).sum().item()
        report[key] = {"differing": int(differing),
                       "max_abs_difference": float((cpu[key] - gpu[key]).abs().max())}
    return report


@app.local_entrypoint()
def probe() -> None:
    print(gpu_operation_probe.remote())


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1800)
def gpu_pseudo_probe() -> dict:
    """Compares each step of the group quantizer on GPU and CPU for one random matrix."""
    import sys

    import torch

    sys.path.insert(0, "/root")
    import awq

    generator = torch.Generator().manual_seed(0)
    matrix = torch.randn(4864, 896, generator=generator) * 0.05
    steps = {}
    for label, device in (("cpu", "cpu"), ("gpu", "cuda")):
        w = matrix.to(device)
        grouped = w.reshape(w.shape[0], w.shape[1] // GROUP_SIZE, GROUP_SIZE)
        max_abs = grouped.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(max_abs > 0, max_abs / 7, torch.ones_like(max_abs))
        divided = grouped / scale
        codes = awq.round_half_away(divided).clamp(-7, 7)
        steps[label] = {"max_abs": max_abs.cpu(), "scale": scale.cpu(), "divided": divided.cpu(), "codes": codes.cpu()}
    report = {}
    for key in ("max_abs", "scale", "divided", "codes"):
        cpu_value, gpu_value = steps["cpu"][key], steps["gpu"][key]
        report[key] = {"differing": int((cpu_value != gpu_value).sum()),
                       "max_abs_difference": float((cpu_value - gpu_value).abs().max())}
    return report


@app.local_entrypoint()
def pseudo_probe() -> None:
    print(gpu_pseudo_probe.remote())
