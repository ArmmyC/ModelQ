"""Runtime verification of ModelQ's GGUF Q8_0 export with llama.cpp (ADR 0032, M5).

Everything runs in one Modal container with a GPU (for the reference model)
and the CPU (for llama.cpp). Nothing is downloaded to the local machine.

1. builds the release CLI and exports the pinned Qwen2.5-0.5B revision with
   `modelq quantize hf:... --format gguf-q8_0`;
2. checks the file with llama.cpp's own GGUF reader library (the `gguf` package)
   and compares each Q8_0 tensor, decoded, with the original weights;
3. builds llama.cpp at the pinned release tag and loads the file with `llama-completion`
   to generate text;
4. measures perplexity with `llama-perplexity` on the first WikiText-2 test
   windows, and on the same windows evaluates the original model and the model
   with the decoded Q8_0 weights, so the runtime's number can be checked against
   a Python evaluation of the same file.

Run with::

    python -m modal run tools/gguf/modal_llama_verify.py::main
"""

import hashlib
import json
import pathlib
import re
import subprocess
import time

import modal

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
QUALITY = _HERE.parents[1] / "quality_eval" if len(_HERE.parents) > 1 else _HERE.parent
TE_TOOLS = _HERE.parents[1] / "transformer_engine_nvfp4" if len(_HERE.parents) > 1 else _HERE.parent

HF = "https://huggingface.co"
MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TEST_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
# The llama.cpp release tag this file is verified against, and its commit.
LLAMA_TAG = "v0.6.0"
LLAMA_COMMIT = "d81235049384534c167caea52b85a694f6103d14"
WINDOW = 2048
CHUNKS = 20
PROMPT = "The capital of France is"

app = modal.App("modelq-gguf-llama-verify")

image = (
    modal.Image.from_registry("rust:1.85", add_python="3.12")
    .pip_install(
        "torch==2.9.0", "transformers>=4.50,<5", "numpy>=1.24,<3", "safetensors>=0.4,<1",
        "pyarrow", "huggingface_hub", "gguf", "cmake",
    )
    .add_local_file(str(QUALITY / "quality_eval.py"), "/root/quality_eval.py")
    .add_local_file(str(TE_TOOLS / "validate_multi.py"), "/root/transformer_engine_nvfp4/validate_multi.py")
    .add_local_file(str(TE_TOOLS / "validate.py"), "/root/transformer_engine_nvfp4/validate.py")
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


def _run(command, cwd=None, timeout=None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False, timeout=timeout)


def _hf_name(gguf_name: str):
    """The Hugging Face name for a GGUF tensor name (the inverse of the exporter's mapping)."""
    fixed = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    if gguf_name in fixed:
        return fixed[gguf_name]
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if not match:
        return None
    layer, suffix = match.group(1), match.group(2)
    mapped = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_q.bias": "self_attn.q_proj.bias",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_k.bias": "self_attn.k_proj.bias",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_v.bias": "self_attn.v_proj.bias",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }.get(suffix)
    return f"model.layers.{layer}.{mapped}" if mapped else None


@app.function(image=image, gpu="L4", cpu=8, memory=49152, timeout=10800)
def verify() -> dict:
    import math
    import sys

    import gguf
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/transformer_engine_nvfp4")
    import quality_eval as qe

    started = time.time()
    scratch = pathlib.Path("/tmp/gguf")
    scratch.mkdir(parents=True, exist_ok=True)
    result: dict = {"model": {"id": MODEL, "revision": MODEL_REVISION},
                    "llama_cpp": {"tag": LLAMA_TAG, "commit": LLAMA_COMMIT}}

    # --- 1. export with the CLI ----------------------------------------------------------
    build = _run(["cargo", "build", "--release", "--bin", "modelq"], cwd="/repo")
    if build.returncode != 0:
        raise RuntimeError("cargo build failed:\n" + build.stderr[-4000:])
    gguf_path = scratch / "qwen2.5-0.5b-q8_0.gguf"
    export = _run([
        "/repo/target/release/modelq", "quantize", f"hf:{MODEL}", "--format", "gguf-q8_0",
        "--revision", MODEL_REVISION, "--cache-dir", str(scratch / "cache"), "--output", str(gguf_path),
    ])
    result["export"] = {"exit_code": export.returncode, "stdout": export.stdout[-3000:],
                        "stderr": export.stderr[-2000:], "bytes": gguf_path.stat().st_size if gguf_path.exists() else None}
    if export.returncode != 0:
        raise RuntimeError("export failed:\n" + export.stdout[-2000:] + export.stderr[-2000:])

    # --- 2. the GGUF reader library -------------------------------------------------------
    reader = gguf.GGUFReader(str(gguf_path))
    fields = {name: reader.fields[name] for name in reader.fields}
    architecture = bytes(reader.fields["general.architecture"].parts[-1]).decode("utf-8")
    token_field = reader.fields["tokenizer.ggml.tokens"]
    result["gguf_reader"] = {
        "architecture": architecture,
        "tensor_count": len(reader.tensors),
        "metadata_keys": sorted(fields),
        "vocabulary_size": len(token_field.data),
    }

    # Compare each tensor, decoded, with the source weights.
    weights_path = next((scratch / "cache").rglob("model.safetensors"))
    comparisons = {}
    worst_error = 0.0
    with safe_open(weights_path, framework="pt") as source:
        for tensor in reader.tensors:
            hf_name = _hf_name(tensor.name)
            if hf_name is None:
                raise RuntimeError(f"tensor {tensor.name} has no Hugging Face name")
            reference = source.get_tensor(hf_name).to(torch.float32).numpy()
            if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
                decoded = gguf.quants.dequantize(tensor.data, gguf.GGMLQuantizationType.Q8_0)
            else:
                decoded = np.asarray(tensor.data, dtype=np.float32)
            decoded = decoded.reshape(reference.shape)
            difference = float(np.abs(decoded - reference).max())
            # Q8_0 stores each block scale as binary16, so the reconstruction error is at most
            # 0.5 d + 127 * |d16 - d|, with |d16 - d| <= d * 2**-11. With d <= max|w| / 127 this is
            # the bound below; the plain half-step bound is too strict and was wrong (ADR 0032).
            scale_bound = (0.5 + 127 * 2.0**-11) * float(np.abs(reference).max()) / 127 if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0 else 0.0
            comparisons[tensor.name] = {
                "type": tensor.tensor_type.name,
                "shape_gguf": [int(d) for d in tensor.shape],
                "shape_hf": list(reference.shape),
                "max_abs_difference": difference,
                "half_scale_bound": scale_bound,
            }
            if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
                if difference != 0.0:
                    raise RuntimeError(f"{tensor.name} (F32) differs from the source")
            else:
                if difference > scale_bound * (1 + 1e-5) + 1e-7:
                    raise RuntimeError(f"{tensor.name} exceeds half a Q8_0 scale: {difference} > {scale_bound}")
            worst_error = max(worst_error, difference)
    result["gguf_tensors"] = {"checked": len(comparisons), "worst_abs_error": worst_error,
                              "by_name": {k: comparisons[k] for k in list(comparisons)[:8]}}

    # --- 3. llama.cpp at the pinned tag ---------------------------------------------------
    llama_source = scratch / "llama.cpp"
    archive = scratch / "llama.tar.gz"
    import urllib.request

    with urllib.request.urlopen(f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_COMMIT}.tar.gz") as response:
        archive.write_bytes(response.read())
    _run(["tar", "-xzf", str(archive), "-C", str(scratch)])
    extracted = next(p for p in scratch.iterdir() if p.is_dir() and p.name.startswith("llama.cpp"))
    extracted.rename(llama_source)
    configure = _run(["cmake", "-S", str(llama_source), "-B", str(llama_source / "build"),
                      "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
                      ])
    if configure.returncode != 0:
        raise RuntimeError("cmake configure failed:\n" + configure.stderr[-4000:])
    compile_ = _run(["cmake", "--build", str(llama_source / "build"), "--config", "Release",
                     "--target", "llama-completion", "llama-perplexity", "-j", "8"])
    if compile_.returncode != 0:
        raise RuntimeError("cmake build failed:\n" + compile_.stdout[-3000:] + compile_.stderr[-3000:])
    cli = llama_source / "build" / "bin" / "llama-completion"
    perplexity = llama_source / "build" / "bin" / "llama-perplexity"

    # --- 4. generation ---------------------------------------------------------------------
    generate = _run([str(cli), "-m", str(gguf_path), "-p", PROMPT, "-n", "24", "--temp", "0",
                     "-no-cnv", "--no-display-prompt", "-t", "8"], timeout=900)
    result["generation"] = {"exit_code": generate.returncode, "stdout": generate.stdout[-1500:],
                            "stderr_tail": generate.stderr[-800:]}
    if generate.returncode != 0:
        raise RuntimeError("llama-completion failed:\n" + generate.stderr[-3000:])

    # --- 5. perplexity: the runtime and the Python evaluation on the same windows ----------
    test_path = hf_hub_download(DATASET, TEST_FILE, repo_type="dataset", revision=DATASET_REVISION)
    text = "\n\n".join(pq.read_table(test_path).column("text").to_pylist())
    text_path = scratch / "wikitext2-test.txt"
    text_path.write_text(text, encoding="utf-8")
    runtime = _run([str(perplexity), "-m", str(gguf_path), "-f", str(text_path), "-c", str(WINDOW),
                    "--chunks", str(CHUNKS), "-t", "8"], timeout=3600)
    final = re.search(r"Final estimate: PPL = ([0-9.]+)", runtime.stdout + runtime.stderr)
    result["llama_perplexity"] = {"exit_code": runtime.returncode, "chunks": CHUNKS,
                                  "final_ppl": float(final.group(1)) if final else None,
                                  "tail": (runtime.stdout + runtime.stderr)[-600:]}

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    token_ids = tokenizer(text, return_tensors=None)["input_ids"]
    windows = qe.make_windows(token_ids, WINDOW)[:CHUNKS]
    device = torch.device("cuda")
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.float32).to(device)
    variant = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.float32).to(device)
    with torch.no_grad():
        parameters = dict(variant.named_parameters())
        loaded = 0
        for tensor in reader.tensors:
            hf_name = _hf_name(tensor.name)
            if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
                values = gguf.quants.dequantize(tensor.data, gguf.GGMLQuantizationType.Q8_0)
            else:
                values = np.asarray(tensor.data, dtype=np.float32)
            parameter = parameters.get(hf_name)
            if parameter is None:
                continue
            parameter.copy_(torch.from_numpy(values.reshape(tuple(parameter.shape))).to(device))
            loaded += 1
    python_result = qe.evaluate_pair(base, variant, windows, torch, device)
    original = qe.evaluate_pair(base, base, windows, torch, device)
    result["python_evaluation"] = {
        "windows": len(windows),
        "tensors_loaded_from_gguf": loaded,
        "perplexity_original": original["perplexity_original"],
        "perplexity_decoded_gguf": python_result["perplexity_quantized"],
        "relative_increase": python_result["perplexity_quantized"] / original["perplexity_original"] - 1,
    }
    # llama.cpp's default perplexity scores only the second half of each window
    # (tokens first+1 .. n_ctx-1, with first = n_ctx // 2; see tools/perplexity in the
    # pinned release). Evaluate the same positions here, so the runtime's number and
    # the Python number are over the same tokens.
    first = WINDOW // 2
    total_original = 0.0
    total_decoded = 0.0
    scored = 0
    with torch.no_grad():
        for window in windows:
            ids = torch.tensor([window], device=device)
            logits_original = base(ids).logits[0, :-1].float()[first:]
            logits_decoded = variant(ids).logits[0, :-1].float()[first:]
            labels = ids[0, 1:][first:]
            rows = torch.arange(labels.shape[0], device=device)
            log_original = torch.log_softmax(logits_original, dim=-1)
            log_decoded = torch.log_softmax(logits_decoded, dim=-1)
            total_original += float(-log_original[rows, labels].sum())
            total_decoded += float(-log_decoded[rows, labels].sum())
            scored += labels.shape[0]
    result["llama_rule"] = {
        "scored_tokens": scored,
        "perplexity_original": math.exp(total_original / scored),
        "perplexity_decoded_gguf": math.exp(total_decoded / scored),
        "llama_cpp_perplexity": result["llama_perplexity"]["final_ppl"],
    }
    result["seconds"] = round(time.time() - started, 1)
    return result


@app.local_entrypoint()
def main() -> None:
    result = verify.remote()
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    path = results / "gguf-llama-verify.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("export", "gguf_reader", "gguf_tensors")}, indent=2)[:3000])
    print("generation:", result["generation"]["stdout"][:400])
    print("llama.cpp PPL:", result["llama_perplexity"]["final_ppl"], "chunks", result["llama_perplexity"]["chunks"])
    print("python evaluation:", json.dumps(result["python_evaluation"], indent=2))
    print(f"result written to {path}")


@app.function(image=image, cpu=8, memory=32768, timeout=5400)
def tokenizer_check() -> dict:
    """Compares llama.cpp's token ids for the WikiText-2 test text with the Hugging Face tokenizer's."""
    import sys

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    sys.path.insert(0, "/root")
    scratch = pathlib.Path("/tmp/gguf-tok")
    scratch.mkdir(parents=True, exist_ok=True)
    build = _run(["cargo", "build", "--release", "--bin", "modelq"], cwd="/repo")
    if build.returncode != 0:
        raise RuntimeError("cargo build failed:\n" + build.stderr[-4000:])
    gguf_path = scratch / "q8.gguf"
    export = _run([
        "/repo/target/release/modelq", "quantize", f"hf:{MODEL}", "--format", "gguf-q8_0",
        "--revision", MODEL_REVISION, "--cache-dir", str(scratch / "cache"), "--output", str(gguf_path),
    ])
    if export.returncode != 0:
        raise RuntimeError("export failed:\n" + export.stderr[-2000:])

    import urllib.request

    archive = scratch / "llama.tar.gz"
    with urllib.request.urlopen(f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_COMMIT}.tar.gz") as response:
        archive.write_bytes(response.read())
    _run(["tar", "-xzf", str(archive), "-C", str(scratch)])
    extracted = next(p for p in scratch.iterdir() if p.is_dir() and p.name.startswith("llama.cpp"))
    source = scratch / "llama.cpp"
    extracted.rename(source)
    configure = _run(["cmake", "-S", str(source), "-B", str(source / "build"), "-DCMAKE_BUILD_TYPE=Release",
                      "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF"])
    if configure.returncode != 0:
        raise RuntimeError("cmake configure failed:\n" + configure.stderr[-3000:])
    compile_ = _run(["cmake", "--build", str(source / "build"), "--config", "Release",
                     "--target", "llama-tokenize", "-j", "8"])
    if compile_.returncode != 0:
        raise RuntimeError("cmake build failed:\n" + compile_.stdout[-2000:] + compile_.stderr[-2000:])
    tokenize = source / "build" / "bin" / "llama-tokenize"

    test_path = hf_hub_download(DATASET, TEST_FILE, repo_type="dataset", revision=DATASET_REVISION)
    text = "\n\n".join(pq.read_table(test_path).column("text").to_pylist())
    text_path = scratch / "wikitext2-test.txt"
    text_path.write_text(text, encoding="utf-8")

    llama = _run([str(tokenize), "-m", str(gguf_path), "-f", str(text_path), "--ids", "--no-bos", "--log-disable"],
                 timeout=1800)
    ids_text = llama.stdout.strip()
    llama_ids = json.loads(ids_text) if ids_text.startswith("[") else None
    stderr_lines = [line for line in llama.stderr.splitlines() if "warn" in line.lower() or "bug" in line.lower()
                    or "control" in line.lower() or "eog" in line.lower()]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    hf_ids = tokenizer(text, return_tensors=None)["input_ids"]
    result = {"exit_code": llama.returncode, "hf_tokens": len(hf_ids),
              "llama_tokens": len(llama_ids) if llama_ids is not None else None,
              "stdout_head": llama.stdout[:300], "stderr_warnings": stderr_lines[:20],
              "stderr_head": llama.stderr[:600]}
    if llama_ids is not None:
        mismatch = next((i for i, (a, b) in enumerate(zip(hf_ids, llama_ids)) if a != b), None)
        result["identical"] = mismatch is None and len(hf_ids) == len(llama_ids)
        result["first_mismatch"] = mismatch
        if mismatch is not None:
            result["hf_around"] = hf_ids[max(0, mismatch - 4): mismatch + 6]
            result["llama_around"] = llama_ids[max(0, mismatch - 4): mismatch + 6]
            result["hf_text_around"] = tokenizer.decode(hf_ids[max(0, mismatch - 4): mismatch + 6])
    return result


@app.local_entrypoint()
def tokenizer() -> None:
    result = tokenizer_check.remote()
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    (results / "gguf-tokenizer-check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "stderr_head"}, indent=2)[:4000])
