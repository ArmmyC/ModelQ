# ModelQ

ModelQ is a planned cross-platform, inference-independent model quantization
compiler and toolkit written in Rust. Its goal is to transform model
checkpoints into smaller, explicitly described representations without running
model inference.

The repository is currently at the project-bootstrap stage. The binary and
library module layout exist, but quantization, checkpoint I/O, and user-facing
commands are not implemented yet. See [PROJECT.md](PROJECT.md) for the current
project definition and implementation roadmap.

## Requirements

- Stable Rust 1.85 or newer

## Local checks

```bash
cargo build
cargo test
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```
