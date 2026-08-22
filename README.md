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
```

To generate the focused GGUF fixture locally:

```bash
cargo run -p modelq-io --example gguf_q8_0_fixture -- /tmp/modelq-q8-0.gguf
```
