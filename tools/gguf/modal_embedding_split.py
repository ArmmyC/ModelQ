"""Splits the Q4_0 quality cost between the token embedding and the linear layers (ADR 0033 follow-up).

Python only, on one Modal GPU. Nothing is downloaded to the local machine. Each variant sets
every two-dimensional weight to one of three values: the original weights, their Q4_0 round
trip, or their Q8_0 round trip. The Q4_0 round trip uses gguf-py's quantizer, which writes the
same bytes as ModelQ's exporter (ADR 0033, verification item 3). The Q8_0 round trip also uses
gguf-py, so it can differ from ModelQ's Q8_0 on exact rounding ties; the effect is far smaller
than the differences measured here.

The windows and the two perplexity rules are the ones in ADR 0032 and ADR 0033: llama.cpp's rule
(only the second half of each 2048-token window is scored) and all positions.

Run with::

    py -3 -m modal run tools/gguf/modal_embedding_split.py::main
"""

import json
import math
import pathlib
import time

import modal

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
QUALITY = _HERE.parents[1] / "quality_eval" if len(_HERE.parents) > 1 else _HERE.parent
TE_TOOLS = _HERE.parents[1] / "transformer_engine_nvfp4" if len(_HERE.parents) > 1 else _HERE.parent

MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TEST_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
WINDOW = 2048
CHUNKS = 20
EMBEDDING = "model.embed_tokens.weight"
# The Q4_0 file from ADR 0033 (284,084,288 bytes) and its Q4_0 embedding (76,575,744 bytes).
Q4_FILE_BYTES = 284_084_288
# The numbers ADR 0033 reports for the all-Q4_0 file, which this run must reproduce.
ADR_0033_REFERENCE = {
    "original": {"all_positions": 12.701029116355178, "llama_rule": 11.304229573464996},
    "q4_0_all": {"all_positions": 14.780485687961873, "llama_rule": 13.106253859095489},
}

ORIGINAL, Q4, Q8 = "original", "q4_0", "q8_0"
VARIANTS = {
    "original": {"embedding": ORIGINAL, "linear": ORIGINAL},
    "q4_0_all": {"embedding": Q4, "linear": Q4},
    "embedding_q4_0_linear_original": {"embedding": Q4, "linear": ORIGINAL},
    "embedding_original_linear_q4_0": {"embedding": ORIGINAL, "linear": Q4},
    "embedding_q8_0_linear_q4_0": {"embedding": Q8, "linear": Q4},
}

app = modal.App("modelq-embedding-split")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.0", "transformers>=4.50,<5", "numpy>=1.24,<3", "safetensors>=0.4,<1",
        "pyarrow", "huggingface_hub", "gguf",
    )
    .add_local_file(str(QUALITY / "quality_eval.py"), "/root/quality_eval.py")
    .add_local_file(str(TE_TOOLS / "validate_multi.py"), "/root/transformer_engine_nvfp4/validate_multi.py")
    .add_local_file(str(TE_TOOLS / "validate.py"), "/root/transformer_engine_nvfp4/validate.py")
)


@app.function(image=image, gpu="L4", cpu=8, memory=49152, timeout=7200)
def split() -> dict:
    import sys

    import gguf
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/transformer_engine_nvfp4")
    import quality_eval as qe

    started = time.time()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    test_path = hf_hub_download(DATASET, TEST_FILE, repo_type="dataset", revision=DATASET_REVISION)
    text = "\n\n".join(pq.read_table(test_path).column("text").to_pylist())
    token_ids = tokenizer(text, return_tensors=None)["input_ids"]
    windows = qe.make_windows(token_ids, WINDOW)[:CHUNKS]

    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.float32).to(device)
    model.eval()
    parameters = dict(model.named_parameters())
    # ModelQ quantizes the two-dimensional weights only; norms and biases stay F32.
    matrices = [name for name, parameter in parameters.items() if parameter.dim() == 2]
    if len(matrices) != 169 or EMBEDDING not in matrices:
        raise RuntimeError(f"expected 169 two-dimensional weights including {EMBEDDING}, found {len(matrices)}")
    elements = parameters[EMBEDDING].numel()

    def round_trip(values: np.ndarray, kind) -> torch.Tensor:
        packed = gguf.quants.quantize(values, kind)
        return torch.from_numpy(np.ascontiguousarray(gguf.quants.dequantize(packed, kind).reshape(values.shape)))

    with torch.no_grad():
        original = {name: parameters[name].detach().cpu().clone() for name in matrices}
        q4 = {}
        for name in matrices:
            values = np.ascontiguousarray(original[name].numpy(), dtype=np.float32)
            q4[name] = round_trip(values, gguf.GGMLQuantizationType.Q4_0)
        values = np.ascontiguousarray(original[EMBEDDING].numpy(), dtype=np.float32)
        q8_embedding = round_trip(values, gguf.GGMLQuantizationType.Q8_0)

    def install(choice: dict) -> None:
        with torch.no_grad():
            for name in matrices:
                source = choice["embedding"] if name == EMBEDDING else choice["linear"]
                if source == ORIGINAL:
                    tensor = original[name]
                elif source == Q4:
                    tensor = q4[name]
                else:
                    tensor = q8_embedding
                parameters[name].copy_(tensor.to(device))

    def score() -> dict:
        first = WINDOW // 2
        all_nll = half_nll = 0.0
        all_count = half_count = 0
        with torch.no_grad():
            for window in windows:
                ids = torch.tensor([window], device=device)
                logits = model(ids).logits[0, :-1].float()
                labels = ids[0, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                nll = -log_probs.gather(1, labels[:, None])[:, 0]
                all_nll += float(nll.sum())
                all_count += nll.numel()
                half_nll += float(nll[first:].sum())
                half_count += nll[first:].numel()
        return {
            "all_positions": math.exp(all_nll / all_count),
            "llama_rule": math.exp(half_nll / half_count),
            "scored_tokens_llama_rule": half_count,
        }

    # Embedding bytes: F32 is 4 bytes per value (a full-precision embedding), Q8_0 is 34 bytes per 32
    # values, Q4_0 is 18 bytes per 32 values. The file total is the Q4_0 file with its embedding swapped.
    embedding_bytes = {ORIGINAL: elements * 4, Q4: elements // 32 * 18, Q8: elements // 32 * 34}
    rows = {}
    for label, choice in VARIANTS.items():
        install(choice)
        row = score()
        row["choice"] = choice
        row["embedding_bytes"] = embedding_bytes[choice["embedding"]]
        row["file_bytes_estimate"] = Q4_FILE_BYTES - embedding_bytes[Q4] + embedding_bytes[choice["embedding"]]
        rows[label] = row

    base = rows["original"]
    for label, row in rows.items():
        row["increase_llama_rule"] = row["llama_rule"] / base["llama_rule"] - 1
        row["increase_all_positions"] = row["all_positions"] / base["all_positions"] - 1

    reproduced = {
        label: {
            key: abs(rows[label][key] - value) / value
            for key, value in ADR_0033_REFERENCE[label].items()
        }
        for label in ADR_0033_REFERENCE
    }
    return {
        "model": {"id": MODEL, "revision": MODEL_REVISION},
        "windows": len(windows),
        "variants": rows,
        "reproduces_adr_0033_relative_difference": reproduced,
        "seconds": round(time.time() - started, 1),
    }


@app.local_entrypoint()
def main() -> None:
    result = split.remote()
    results = REPO / "modal_results"
    results.mkdir(exist_ok=True)
    (results / "gguf-embedding-split.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for label, row in result["variants"].items():
        print(
            f"{label:34s} llama-rule {row['llama_rule']:9.4f} ({row['increase_llama_rule']:+7.2%})"
            f"  all {row['all_positions']:9.4f} ({row['increase_all_positions']:+7.2%})"
            f"  file ~{row['file_bytes_estimate'] / 1e6:6.0f} MB"
        )
    print("ADR 0033 reproduction, relative difference:", json.dumps(result["reproduces_adr_0033_relative_difference"]))
    print(f"result written to {results / 'gguf-embedding-split.json'}")
