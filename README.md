# ModelQ

ModelQ is a planned cross-platform, inference-independent model quantization
compiler and toolkit written in Rust. Its goal is to transform model
checkpoints into smaller, explicitly described representations without running
model inference.

The repository is in early development. It currently provides validated source
tensor metadata, borrowed views for F32, F16, and BF16 data, SafeTensors
metadata inspection, and read-only memory-mapped views over those source
tensors. It also includes a scalar symmetric INT8 reference quantizer with
round-trip dequantization, plus streaming reconstruction and compression
diagnostics. It also has a conservative, auditable policy for deciding which
tensors enter the INT8 path, plus a checked output layout planner. Later
quantization formats and most user-facing commands are not implemented yet.
See [PROJECT.md](PROJECT.md) for the current project definition and
implementation roadmap. The planned ModelQ-native INT8 output convention is
documented in
[ADR 0002](docs/adr/0002-modelq-native-quantized-tensor-convention.md), and
the streaming writer now implements that convention without changing the
source mapping.

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
cargo build
cargo test
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```
