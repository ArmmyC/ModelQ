# ADR 0035: Evaluating Every ModelQ Output

- Status: Accepted (Task 56, a follow-up to ADR 0028 M1 and ADR 0029). `modelq eval` reads every container that `modelq quantize` writes, and it fetches `hf:` models. Accepted on the end-to-end run below.
- Date: 2026-10-10
- Scope: `modelq eval`, its Python measurement in `tools/quality_eval/`, and the readers in `tools/quality_eval/modelq_containers.py`. The quantizers and the container formats are unchanged.
- Builds on: [ADR 0028](0028-open-source-quantizer-scope.md) (M1, the local measurement, and its open item that eval reads only Transformer Engine containers), [ADR 0029](0029-hugging-face-hub-input.md) (Hub input, and its open item that eval still downloaded through the Python tool), [ADR 0030](0030-group-wise-low-bit-formats.md) (low-bit formats, with no quality measured yet), [ADR 0032](0032-gguf-q8-0-export.md) and [ADR 0034](0034-gguf-q4-0-q8-vocabulary.md) (the GGUF exports).

## Context

`modelq eval` read only Transformer Engine NVFP4 containers from a local model directory. The other formats had to be measured with ad hoc Python scripts, and `hf:` models were not accepted, so the one-command path of ADR 0029 was not met.

## Decisions

### 1. Every format that `quantize` writes can be evaluated

`modelq eval --container` accepts:

- `int8` (ADR 0002);
- `int4`, `int3`, `int2` and `int1` (ADR 0030);
- `nvfp4`, the native layout (ADR 0011);
- `nvfp4-te`, the Transformer Engine layout (ADR 0012);
- `gguf-q8_0` and `gguf-q4_0` (ADR 0032, ADR 0034), for qwen2 models.

The format is read from the file, not from its name: the GGUF magic, or the `modelq.format` and `modelq.quantization` metadata of a SafeTensors container. Any other file is refused.

### 2. Each decoder follows its writer

- INT8: `qdata * scale`, with one F32 scale per tensor.
- Native NVFP4: `e2m1 * e4m3_block_scale * global_scale`, with the low nibble of each byte first and one E4M3 scale per 16 values.
- GGUF: Q8_0 and Q4_0 blocks decode as llama.cpp's reference and gguf-py do; F32 is raw.
- Low-bit and Transformer Engine NVFP4 keep their existing decoders in `quality_eval.py`.

### 3. Coverage is checked

Every model parameter must be written by the container, and every container tensor must correspond to a parameter of the same shape. A mismatch is an error, so no weight is left unquantized by mistake.

### 4. `hf:` models

`modelq eval --model hf:<owner>/<name> --download` fetches the model with the Hub client that `quantize` uses: verified downloads, and the same cache. `--revision` selects the revision. `Qwen/Qwen2.5-0.5B` uses the pinned revision of ADR 0024 and ADR 0026 when `--revision` is omitted, and any other repository needs `--revision`. The report records the repository and the commit that was fetched. The interpreter is checked before anything is fetched.

### 5. Downloads

Nothing is downloaded without `--download`. An `hf:` model needs it, and so does the pinned WikiText-2 split when `--dataset` is missing. The forms that worked before, a local directory with `--dataset` and a pinned id with `--download`, are unchanged.

## Verification

1. **Golden tests.** The Python decoders are checked against the writers' reference blocks: the Q4_0 block for `x_i = i - 16` (ADR 0033), codes 0 to 15 for native NVFP4, and `qdata * scale` for INT8. Substitution and dispatch are tested on small models. The 12 new Python tests, and the existing ones, pass locally: 34 in `tools/quality_eval`.

2. **Rust.** The Linux suite (Rust 1.85) passes with 359 tests, 0 failed and 2 ignored, and rustfmt is clean. Clippy on Rust 1.99 exits 0. The eval unit and integration tests cover the refusals: no `--download` for an `hf:` model, no pinned revision, and `--revision` on a local directory.

3. **End to end (Modal, L4).** The pinned Qwen2.5-0.5B revision was quantized with `modelq quantize` to all six formats. One `modelq eval --model hf:Qwen/Qwen2.5-0.5B --revision <pinned> --download --device cuda` then scored all six containers, on the first 20 windows and on the full split of 146 windows.

4. **Anchors.** Results measured earlier by other paths must come out the same:

   | Anchor | Earlier | Measured now |
   | --- | --- | --- |
   | Original model, 20 windows | 12.701029116355178 | 12.701029116355178 |
   | GGUF Q8_0, 20 windows (ADR 0032) | 12.70804975818944 | 12.70804975818944 |
   | GGUF Q4_0, 20 windows (ADR 0034) | 14.216804915771643 | 14.216804915771643 |
   | Data-free INT4, 146 windows (ADR 0031) | 19.885816067878967 | 19.885816067878967 |

   All four match exactly. The Transformer Engine NVFP4 result on 146 windows, +8.073%, top-1 85.343% and mean weight error 8.151%, matches the ADR 0028 M1 record. The perplexities differ by 0.005%, which is the difference between the two platforms. The native and Transformer Engine NVFP4 results are identical on both splits, to every printed digit. They come from two separate decoders, so this also checks the native decoder against the Transformer Engine one.

## Results

Relative to the original model: 12.701029 on 20 windows and 13.069886 on 146 windows.

| Container | Format | Size | 20 windows: perplexity | Increase | Top-1 | 146 windows: perplexity | Increase | Top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INT8 | `int8` | 494 MB | 13.287539 | +4.62% | 88.18% | 13.690255 | +4.75% | 87.96% |
| INT4 | `lowbit` | 263 MB | 19.233718 | +51.43% | 67.74% | 19.885816 | +52.15% | 67.15% |
| NVFP4, native | `nvfp4` | 474 MB | 13.709766 | +7.94% | 85.41% | 14.125071 | +8.07% | 85.34% |
| NVFP4, Transformer Engine | `nvfp4-te` | 474 MB | 13.709766 | +7.94% | 85.41% | 14.125071 | +8.07% | 85.34% |
| GGUF Q8_0 | `gguf-q8_0` | 531 MB | 12.708050 | +0.06% | 98.38% | 13.077368 | +0.06% | 98.46% |
| GGUF Q4_0 | `gguf-q4_0` | 352 MB | 14.216805 | +11.93% | 82.00% | 14.725778 | +12.67% | 81.80% |

The full numbers, with KL divergence and weight errors, are in the validation record.

## What this does not show

- The scores are simulated: the decoded weights replace the originals in float32. They measure the weight format alone. Runtime effects, such as llama.cpp's quantized activations (ADR 0032 measured a 0.2% gap), are not included, and no runtime is measured for the native formats.
- INT8 has no independent anchor. Its decoder is tested against the reference rule, but its numbers are not compared with an earlier measurement.
- One model and one dataset. GGUF is read for qwen2 models only. SafeTensors checkpoints must be a single file or an index with shards.
- The measurement needs Python with PyTorch. The Rust CLI is a wrapper around it. The Linux suite covers the wrapper. The Windows executable cannot run on the development PC, because Application Control blocks newly built executables there.

## Consequences

- Every output that `modelq quantize` writes can be scored with one command, and `hf:` models work with `eval`. The quality cost of each format is measured with the same code.
- The limits that ADR 0029 and ADR 0030 recorded for `eval` are resolved by this ADR.
