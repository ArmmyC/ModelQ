# ADR 0022: Hardware Validation of the Multi-Matrix Transformer Engine Container

- Status: Accepted (Task 41)
- Date: 2026-10-07
- Scope: runtime and hardware validation of the schema-v2 container from [ADR 0021](0021-transformer-engine-multi-matrix-container.md), per matrix, on an NVIDIA B200 with Transformer Engine 2.19.0
- Result: **validated for the tested shapes and the tested operation (Levels 3 and 4); one eligibility rule was discovered and enforced.**

## What was run

For every Transformer Engine matrix in a container, `tools/transformer_engine_nvfp4/validate_multi.py runtime`:

1. validates the container and the F32 reference on the CPU;
2. builds the pinned TE 2.19.0 `NVFP4Tensor` from the exported rowwise fields exactly as in ADR 0013 (rowwise buffers only, columnwise buffers `None`, `with_gemm_swizzled_scales=False`);
3. compares TE's own dequantization of that tensor with the reference (`rtol=1e-5`, `atol=1e-5`); and
4. runs one `general_gemm(weight, second_operand, layout="TN")` against the oracle `TE-dequantized second operand @ reference.T` for a deterministic `[64, K]` second operand (`rtol=0.125`, `atol=0.0675`), requiring a finite output of shape `[64, M]`.

A failing matrix is recorded and the run continues, so one paid run reports every incompatibility. Exit status is non-zero if any matrix fails; an unavailable GPU is not a pass.

The runs used `tools/transformer_engine_nvfp4/modal_runtime_proof.py` on Modal, in the environment of ADR 0013:

| Component | Version |
| --- | --- |
| Image | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` |
| Python | 3.12.1 |
| PyTorch | 2.9.0+cu128 |
| CUDA runtime | 12.8 |
| cuDNN | 91002 (9.10.2), the PyTorch wheel's copy via `CUDNN_HOME`, `CUDNN_PATH`, `LD_LIBRARY_PATH` |
| NVIDIA driver | 580.95.05 |
| Transformer Engine | 2.19.0 (built from source for the image) |
| GPU | NVIDIA B200, compute capability 10.0 |

The raw results of all three runs are stored in [`docs/validation/`](../validation/).

## What happened

**Run 1 (initial fixture, `...-1-initial-run.json`).** The Task 40 synthetic fixture, as the exporter then defined eligibility (rank 2, both dimensions divisible by 16). Both the single-file and the sharded container gave the same result: TE loaded all six matrices, its dequantization matched the reference (maximum absolute difference 1.9e-6) for every one, and the TN GEMM passed for five. It failed for `[144, 80]` with `cublaslt_gemm.cu:769 ... Assertion failed: status != CUBLAS_STATUS_NOT_SUPPORTED. Unable to find suitable cuBLAS GEMM algorithm`. The container format was fine; the runtime would not multiply that shape.

**Run 2 (shape sweep, `...-2-shape-sweep.json`).** 25 matrices varying `K` (16 to 192 in steps of 16, `M = 64`) and `M` (16 to 208, `K = 64`):

| `K` | 16 | 32 | 48 | 64 | 80 | 96 | 112 | 128 | 144 | 160 | 176 | 192 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TN GEMM | fail | pass | fail | pass | fail | pass | fail | pass | fail | pass | fail | pass |

All 13 values of `M` from 16 to 208 passed at `K = 64`. Together with run 1 (`K = 512` and `4096` pass), the observed rule is: **the GEMM works exactly when the final dimension `K` is a multiple of 32; the leading dimension only needs to be a multiple of 16.** This is consistent with the packed row (`K / 2` bytes) needing 16-byte alignment, but the runs establish only the pattern, not that cause. The sweep's reference was produced by the independent numpy decoder rather than the Rust example (Docker and the Windows build were unavailable on the machine at the time); that decoder had been checked against the Rust-generated reference with a difference of 0.0 on the Task 40 fixture.

**Fix.** `Nvfp4Policy::transformer_engine()` now also requires the final dimension to be divisible by 32 and reports `FinalDimensionNotRuntimeAligned`; the container planner rejects other shapes; the fixture's `[144, 80]` matrix became `[144, 96]` and `[144, 80]` stays as a preserved case. Readers and validators stay format-level (multiples of 16), so older containers remain readable.

**Run 3 (final proof, `...-3-final-proof.json`).** The whole pipeline ran on Linux inside Modal: the Rust fixture example and the `modelq` CLI built from the repository source generated the source checkpoint, the containers (single file and sharded) and the Rust-generated reference; then the B200 stage proved them. Both containers passed for every matrix:

| Matrix | TE dequantization max abs difference | TN GEMM max abs error |
| --- | --- | --- |
| `layers.0.attn.weight` `[32, 32]` | 1.9e-6 | 6.1e-5 |
| `layers.0.mlp.up.weight` `[64, 64]` | 1.9e-6 | 9.2e-5 |
| `layers.1.attn.weight` `[48, 96]` | 1.9e-6 | 1.5e-4 |
| `layers.1.mlp.up.weight` `[144, 96]` | 1.9e-6 | 2.1e-4 |
| `layers.2.mlp.up.weight` `[256, 512]` | 1.9e-6 | 7.0e-4 |
| `layers.2.mlp.down.weight` `[1024, 4096]` | 1.9e-6 | 2.0e-3 |

The exporter preserved `odd_columns.weight` `[144, 80]` and `tiny.weight` `[16, 16]` with the new reason, alongside the other ineligible classes.

## What is established

Runtime (Level 3) and hardware (Level 4) compatibility of the schema-v2 container, **for rank-two matrices with the final dimension divisible by 32 and the leading dimension divisible by 16, in TE 2.19.0 on an NVIDIA B200, for loading each matrix and running one forward TN GEMM per matrix**, in both the single-file and the sharded layout. The tested shapes are the six above plus the sweep.

## What is not established

- Other shapes outside the tested ranges, other TE versions, other GPUs, and other CUDA, cuDNN or driver versions.
- Whole-model loading, `te.Linear` integration, a model forward pass, training, columnwise or backward use, or any inference capability. The proof is one GEMM per matrix against an oracle.
- TE's own 2D 16x16 weight convention; this container is 1x16 rowwise, so its numerics are not what TE would produce for weights itself.
- Whether other alignment requirements exist for shapes not tested; stacked or grouped weights remain unsupported by design.

## Consequences

- The CLI report now states the validated scope instead of "not yet hardware-validated".
- The `[144, 80]` failure shows why a hardware run precedes a compatibility claim: the container was format-valid, CPU-validated, and unusable by the runtime for that shape.
- `modal_runtime_proof.py` builds the fixtures and runs the proof in one command (`...::build_and_prove`), so the evidence is reproducible. Running it spends Modal GPU minutes; the cached image makes a rerun short.
