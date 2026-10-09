# ModelQ

ModelQ is a planned cross-platform, inference-independent model quantization
compiler and toolkit written in Rust. Its goal is to transform model
checkpoints into smaller, explicitly described representations without running
model inference.

The repository is in early development. It currently provides validated source
tensor metadata, borrowed views for F32, F16, and BF16 data, SafeTensors
metadata inspection, and read-only memory-mapped views over those source
tensors. It also includes scalar symmetric INT8 and group-wise INT4 reference
quantizers with round-trip dequantization, plus streaming reconstruction and
compression diagnostics. It also has a conservative, auditable policy for
deciding which tensors enter the INT8 path, plus a checked output layout
planner. The CLI currently exposes only the INT8 path; sharded input,
optimized formats, and most other user-facing commands are not implemented
yet. A portable `modelq-backend` crate now provides bounded parallel CPU
library paths for INT8 and INT4; the scalar implementations remain the
correctness reference, and the CLI still uses the bounded scalar writer path.
See [PROJECT.md](PROJECT.md) for the current project definition and
implementation roadmap. The planned ModelQ-native INT8 output convention is
documented in
[ADR 0002](docs/adr/0002-modelq-native-quantized-tensor-convention.md), and
the streaming writer now implements that convention without changing the
source mapping. The initial workspace boundary decision is documented in
[ADR 0004](docs/adr/0004-workspace-boundaries.md), and the CPU parallel
boundary is documented in [ADR 0007](docs/adr/0007-cpu-parallel-dispatch.md).

Task 18 adds a deliberately narrow GGUF compatibility spike: one GGUF v3
Q8_0 tensor, with a deterministic fixture generator and a Rust inspector. It
is currently compatibility Level 2 (container-valid) only; it is not a
general GGUF reader and does not claim that the fixture is a runnable model.
The exact layout, pinned llama.cpp reference, and external `llama-gguf`
validation command are documented in
[ADR 0008](docs/adr/0008-gguf-q8-0-compatibility-spike.md). Quantization is
still not exposed as a general GGUF model conversion command.

Task 19 adds reference element codecs for FP4 E2M1, FP8 E4M3, and FP8 E5M2.
They use documented nearest-even rounding and satfinite behavior with
exhaustive bit-pattern tests. They do not yet add scaling, FP4 array packing,
NVFP4, or runtime-specific export; see
[ADR 0009](docs/adr/0009-fp4-fp8-codecs.md).

Task 20 records the NVFP4 research boundary: E2M1 elements, hierarchical
E4M3/F32 scaling, 16-value groups, the 16x16 weight variant, and a future
Blackwell/runtime validation path. The ModelQ-native scalar reference is now
implemented, but runtime-specific export and validation are not; see
[ADR 0010](docs/adr/0010-nvfp4-research-spike.md).

The native NVFP4 reference increment is now available in
`modelq_quant::nvfp4`. It is a scalar, weight-only implementation with
ModelQ-native low-nibble-first packing, round-trip validation, and a
shape-aware entry point that checks the final dimension is divisible by 16;
it does not claim Transformer Engine or TensorRT compatibility.

The ModelQ-native NVFP4 SafeTensors convention is specified in
[ADR 0011](docs/adr/0011-nvfp4-native-safetensors-convention.md). An NVFP4
SafeTensors planner, writer, and reader are now available as a library-only
path. They emit the ADR 0011 native container and remain runtime-independent:
no NVFP4 CLI flag, runtime exporter, GPU path, or hardware validation is
included yet.

Task 26 adds an opt-in differential-fixture harness in
[`tools/nvfp4_reference.py`](tools/nvfp4_reference.py) and
[`tests/nvfp4_reference.rs`](tests/nvfp4_reference.rs). On a prepared
Blackwell machine, the tool captures a deterministic Transformer Engine
1x16 reference fixture; the Rust test then compares ModelQ's packed bytes,
scales, and reconstructed values. No external fixture is checked in yet
because this repository has no Blackwell capture environment, so this does
not claim runtime compatibility.

Task 27 adds a CPU-only Transformer Engine NVFP4 rowwise 1x16 profile planner
in `modelq_io::transformer_engine`. It maps the native packed E2M1 bytes,
E4M3 block scales, and tensor amax into explicitly named rowwise fields with
Transformer Engine's aligned scale shape. The result is profile-valid CPU data
only: it is not a serialized or runtime-loadable Transformer Engine checkpoint,
and it does not add CUDA, swizzling, columnwise data, or a hardware claim. See
[ADR 0012](docs/adr/0012-transformer-engine-nvfp4-export-profile.md) for the
boundary and follow-up requirements.

Task 28 adds a SafeTensors writer for one NVFP4 matrix with three rowwise
fields, a deterministic fixture pair, and a Python bridge pinned to Transformer
Engine 2.19.0 for one TN GEMM (`second_operand @ weight.T`). CPU container
validation is available, and the single-matrix hardware proof passed on an
NVIDIA B200 on 2026-10-06. It establishes Level 3 runtime compatibility and
Level 4 hardware validation only for this artifact schema, TE 2.19.0, and the
tested operation; it does not establish whole-model compatibility or inference
support. The GEMM's maximum absolute error was `0.000152587890625`. See
[ADR 0013](docs/adr/0013-transformer-engine-nvfp4-runtime-container.md) and the
[tool README](tools/transformer_engine_nvfp4/README.md) for the tested
environment, artifact contract, and validation commands.

Task 29 implements the ADR 0003 input contract as
`modelq_io::sharded::SafetensorsInput`. It opens a single file, a directory, or
a `*.safetensors.index.json` plus its shards, validates every shard and the
index before returning, and exposes one tensor catalog in name order. Payloads
are mapped per call, one shard at a time. Task 30 wires it into the CLI:
`modelq inspect` and `modelq quantize --format int8` accept a checkpoint
directory or index path as well as a single file. The INT8 output is still one
SafeTensors file unless `--max-shard-size` is given (Task 33).

Tasks 31 and 32 add `modelq quantize --format nvfp4`, which writes the
ModelQ-native NVFP4 container from a single or sharded checkpoint using a
bounded-memory two-pass quantizer. It quantizes floating tensors with rank two
or more and a final dimension divisible by 16, preserves everything else with a
printed reason, and keeps names containing `embed_tokens`, `lm_head`, or
`embeddings` at full precision unless `--no-default-excludes` is passed
(`--exclude <SUBSTRING>` adds more). The output is ModelQ-native only; no
runtime compatibility is implied. See
[ADR 0014](docs/adr/0014-nvfp4-cli-export.md).

Task 33 adds `--max-shard-size <SIZE>` (for example `500MB` or `2GiB`) to
`modelq quantize` for both formats. `--output` then names a directory that
receives `model-NNNNN-of-MMMMM.safetensors` shards and a
`model.safetensors.index.json` that the sharded reader accepts. A tensor and
its scales are never split across shards, a failed run removes everything it
created, and the tensors match the single-file output byte for byte. See
[ADR 0015](docs/adr/0015-sharded-output.md).

Task 34 runs the NVFP4 quantizer on multiple CPU threads. It is the default
for `modelq quantize --format nvfp4`; `--threads N` sets the worker count and
`--threads 1` selects the sequential reference path. Output is byte-identical
for every thread count. On a 16M-value benchmark the parallel path was about
4.6x faster with 18 workers (`cargo bench --bench nvfp4_parallel`). See
[ADR 0016](docs/adr/0016-parallel-nvfp4.md).

Task 35 makes the NVFP4 encode kernel about 7x faster on one thread with
byte-identical output (closed-form E2M1 and E4M3 encoders, table decoders, and
an allocation-free block kernel). It also corrects the FP4/FP8 codecs' handling
of astronomically large magnitudes, which now saturate as documented; no value
the NVFP4 path can produce is affected. See
[ADR 0017](docs/adr/0017-fast-nvfp4-encode.md) and the amendment to
[ADR 0009](docs/adr/0009-fp4-fp8-codecs.md).

Task 36 runs the INT8 CLI path on multiple CPU threads as well: progress
diagnostics, the writer, and post-write validation. `--threads N` now applies
to both formats (default: all CPUs; `1` = scalar writer). The INT8 file is
byte-identical for every thread count, and reported error metrics are too,
because sums are accumulated over fixed 4096-value blocks. On a 16M-value
benchmark the three phases together ran about 3.3x faster
(`cargo bench --bench int8_streaming`). See
[ADR 0018](docs/adr/0018-parallel-int8-cli.md).

Task 38 overlaps reading each chunk from the checkpoint with the parallel
quantization work (a reader thread fills the next chunk while workers process
the current one) and moves the default chunk to 2M values. Large tensors
convert about 20% faster at moderate worker counts with identical output. See
[ADR 0020](docs/adr/0020-overlapped-chunk-fill.md) and, for the preceding
scheduler change, [ADR 0019](docs/adr/0019-shared-parallel-scheduler.md).

Task 40 adds `modelq quantize --format nvfp4-te`, which writes many rank-two
matrices into a Transformer Engine rowwise NVFP4 container (schema v2) with a
manifest per file or shard, using the same bounded parallel quantizer, sharded
input and output, and policy flags as `--format nvfp4`. Only rank-two matrices
with a leading dimension divisible by 16 and a final dimension divisible by 32
are exported; everything else is preserved
with a printed reason. Task 41 ran it on hardware: on an NVIDIA B200 with
Transformer Engine 2.19.0, every exported matrix loads into TE, dequantizes to
the reference, and passes one TN GEMM, in both single-file and sharded form.
The run also found that the GEMM is rejected unless the final dimension is
divisible by 32, so the exporter now preserves other matrices (only the tested
operation and shapes are validated; no model loading or inference). See
[ADR 0021](docs/adr/0021-transformer-engine-multi-matrix-container.md),
[ADR 0022](docs/adr/0022-transformer-engine-multi-matrix-hardware-validation.md)
and the [tool README](tools/transformer_engine_nvfp4/README.md).

Task 42 validated the Transformer Engine export on a real checkpoint:
`Qwen/Qwen2.5-0.5B` (Apache-2.0, BF16, 988 MB) was downloaded and
hash-checked inside Modal, converted with `--format nvfp4-te` (474 MB, with the
tied embedding preserved), and all 168 exported matrices loaded into
Transformer Engine 2.19.0 on an NVIDIA B200, dequantized to a reference
re-quantized independently from the source weights, and passed one TN GEMM,
in both single-file and sharded form. This checks that the container and
runtime agree on real shapes; it does not measure model-output quality (no
inference was run). See
[ADR 0023](docs/adr/0023-real-model-validation-qwen2-5-0-5b.md).

Task 43 measured what the NVFP4 weights cost in output quality
(`tools/quality_eval/`): replacing Qwen2.5-0.5B's 168 exported matrices with
their decoded NVFP4 values raised WikiText-2 perplexity from 13.07 to 14.37
(+9.9%), with a mean KL divergence of 0.101 nats per token from the original
and 83.5% top-1 agreement. This is one model, one text and weight-only
simulated quantization in float32, with no comparison to other methods; see
[ADR 0024](docs/adr/0024-nvfp4-output-quality-qwen2-5-0-5b.md).

Task 44 adds a block-scale search, `--scale-search <RADIUS>` for
`--format nvfp4` and `nvfp4-te`: each 16-value block's E4M3 scale is chosen by
minimum reconstruction error among the codes within RADIUS of the reference
rule, instead of from the block's largest value alone. Only the stored scale
bytes change, so decoders and the Transformer Engine runtime are unaffected.
On three Qwen2.5 models (0.5B, 0.5B-Instruct and 1.5B), radius 6 reduced the
WikiText-2 perplexity increase by 19% to 24% (for example +9.93% to +8.07% on
0.5B), passed the B200 proof on all three, and cost about 1.2x wall time in
parallel (8x on one thread); larger radii added nothing. One model family, one
text; see [ADR 0025](docs/adr/0025-nvfp4-block-scale-search.md) and
[ADR 0026](docs/adr/0026-block-scale-search-across-models.md).

Task 46 makes radius 6 the CLI default, so `modelq quantize --format nvfp4`
and `nvfp4-te` now use the search without a flag. `--scale-search 0` selects
the reference rule and reproduces the output of earlier versions byte for byte;
`--threads 1` is noticeably slower with the search. The library API keeps the
reference rule as its default. See
[ADR 0027](docs/adr/0027-scale-search-default.md).

## Quantizing a Hugging Face model

`hf:<owner>/<name>` fetches a model's SafeTensors weights from the Hugging Face
Hub into a cache, checks every file against the checksum the Hub publishes,
and quantizes them. The model's license is printed before anything is
downloaded:

```bash
modelq quantize hf:Qwen/Qwen2.5-0.5B --format nvfp4-te --output ./te.safetensors
```

Add `--revision <branch|tag|commit>` to pin a version, and `--cache-dir <PATH>`
to choose where files are cached (default: `$MODELQ_CACHE`, then your user
cache). Gated or private repositories need `HF_TOKEN` set to a token that has
access, and you must accept the model's terms on its page first. Downloads are
resumable. A bare `owner/name` is treated as a local path, so a typo never
starts a download. See [ADR 0029](docs/adr/0029-hugging-face-hub-input.md).

## Evaluating a quantized output

`modelq eval` measures a ModelQ NVFP4 Transformer Engine container against the
original model on WikiText-2: perplexity, KL divergence, top-1 agreement and
weight error. It runs on your own machine. The measurement itself is Python
(PyTorch, transformers, pyarrow, huggingface_hub), so install those first:

```bash
py -3 -m pip install torch transformers pyarrow huggingface_hub numpy safetensors
modelq eval --model ./Qwen2.5-0.5B --dataset ./wikitext-2-test.parquet   --container ./te.safetensors --report ./quality.json
```

Pass `--download` with a pinned model id (currently `Qwen/Qwen2.5-0.5B`) to
let the script download the model and dataset, each verified against its
SHA-256. Without `--download`, nothing is downloaded. The script is found in a
source checkout, or at `MODELQ_EVAL_SCRIPT`; the interpreter is `--python`,
then `MODELQ_PYTHON`, then `python`. A full run on CPU takes about an hour for
Qwen2.5-0.5B; `--max-windows N` gives a quick check that is not comparable to
full runs. Measurements are not certifications; see
[ADR 0028](docs/adr/0028-open-source-quantizer-scope.md).

## Requirements

- Stable Rust 1.85 or newer

## INT8 command

The first end-to-end path accepts a SafeTensors file and writes a validated
ModelQ-native INT8 SafeTensors file:

```bash
modelq quantize input.safetensors \
  --format int8 \
  --device cpu \
  --output output.safetensors
```

The command reports per-tensor policy decisions, reconstruction diagnostics,
and final byte accounting. It reopens and dequantizes the output before
reporting success. The scalar CPU INT8 path uses bounded replay chunks; later
formats and devices are not yet available.

## Local checks

```bash
cargo build --workspace
cargo test --workspace --all-targets
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo bench --bench cpu_parallel
python tools/test_nvfp4_reference.py
```

To generate the focused GGUF fixture locally:

```bash
cargo run -p modelq-io --example gguf_q8_0_fixture -- /tmp/modelq-q8-0.gguf
```

To validate an already captured NVFP4 reference fixture:

```bash
python tools/nvfp4_reference.py validate path/to/fixture.json
MODELQ_NVFP4_REFERENCE_FIXTURE=path/to/fixture.json \
  cargo test --test nvfp4_reference -- --ignored
```
