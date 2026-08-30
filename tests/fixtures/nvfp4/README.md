# NVFP4 reference fixtures

This directory intentionally contains no captured fixture yet.  A fixture must
be produced on a CUDA-enabled NVIDIA Blackwell machine so ModelQ does not
mistake a local emulation or CPU calculation for NVIDIA validation.

## Capture

Prepare a Python environment with a pinned PyTorch and Transformer Engine
release, then run the helper from the repository root.  Pass the exact
Transformer Engine source commit used by that environment (or set
`MODELQ_NVFP4_TE_COMMIT`):

```bash
python tools/nvfp4_reference.py capture \
  tests/fixtures/nvfp4/transformer-engine-2.20.0-blackwell.json \
  --te-commit <transformer-engine-commit>
```

The helper refuses unsupported devices, non-deterministic recipe settings, or
missing provenance.  It captures a deterministic 2x32 F32 tensor with 1x16
groups and records the source bit patterns, packed FP4 bytes, E4M3 block
scales, F32 decode scale, recipe flags, and producer metadata.

## Validate and compare

Validate the JSON without importing PyTorch:

```bash
python tools/nvfp4_reference.py validate \
  tests/fixtures/nvfp4/transformer-engine-2.20.0-blackwell.json
```

Then run the ignored Rust differential test explicitly:

```bash
MODELQ_NVFP4_REFERENCE_FIXTURE=tests/fixtures/nvfp4/transformer-engine-2.20.0-blackwell.json \
  cargo test --test nvfp4_reference -- --ignored
```

The fixture test compares ModelQ's native packed bytes, block-scale bytes,
global decode scale, and reconstructed F32 values.  It does not validate a
Transformer Engine runtime layout, a GPU kernel, or a SafeTensors exporter.
