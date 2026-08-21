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
quantization formats, checkpoint writing, and most user-facing commands are
not implemented yet. See [PROJECT.md](PROJECT.md) for the current project
definition and implementation roadmap. The planned ModelQ-native INT8 output
convention is documented in
[ADR 0002](docs/adr/0002-modelq-native-quantized-tensor-convention.md); the
writer is not implemented yet.

## Requirements

- Stable Rust 1.85 or newer

## Local checks

```bash
cargo build
cargo test
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```
