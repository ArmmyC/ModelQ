# ADR 0037: AWQ Calibration in `modelq quantize` (Opt-In)

- Status: Accepted (Task 59). The end-to-end check below reproduces ADR 0031's AWQ result exactly.
- Date: 2026-10-10
- Scope: `modelq quantize --format int4 --calibration awq`. It runs the AWQ search of ADR 0031 and writes an ordinary INT4 file. The default does not change: data-free INT4 stays the default, as ADR 0028 section 4 requires.
- Builds on: [ADR 0031](0031-awq-calibration.md) (the method and its measurement), [ADR 0028](0028-open-source-quantizer-scope.md) (the calibration interface, open question 2), [ADR 0030](0030-group-wise-low-bit-formats.md) (the format), [ADR 0036](0036-int4-default.md) (the vocabulary matrices keep their source precision, and calibrated output follows that).

## Context

ADR 0031 implemented AWQ in Python and measured it. On the full split, AWQ with the embedding kept reaches +22.86%, against +39.19% for data-free INT4 with the embedding kept. The CLI could not produce an AWQ file, so the measured method was not reachable from `modelq quantize`.

Calibration needs forward passes, so it runs in Python, in the same environment that `modelq eval` already needs.

## Decisions

1. **Opt-in.** `--calibration awq` is accepted with `--format int4` only. Without it, `quantize` behaves as before.
2. **Python does the search.** `tools/calibration/modelq_awq.py` runs the search of ADR 0031. It writes a rescaled, unquantized checkpoint and a JSON report. The script is found through `MODELQ_AWQ_SCRIPT` or the source checkout. Its interpreter is `--python`, then `MODELQ_PYTHON`, then `python`.
3. **The output is an ordinary INT4 file.** The rescaled checkpoint goes through the INT4 writer, so the result is decoded by the same reader as every other INT4 file.
4. **Calibration data.** The WikiText-2 train split: 32 windows of 512 tokens at the evenly spaced offsets of ADR 0031. `--calibration-data <train.parquet>` uses a local file. `--download` fetches the pinned split, verified by SHA-256. The download needs `--download`, because the calibration text is an artifact the user did not name.
5. **Vocabulary matrices.** They keep their source precision, as ADR 0036 decides for INT4.
6. **Provenance.** The output records `modelq.transform = awq-v1` and `modelq.calibration`, a compact JSON of the report: the dataset revision and checksum, the window offsets, the group size, the batch size, the search summary, the device and the PyTorch version. The decoder does not read these keys.
7. **Temporary files.** The rescaled checkpoint, which stores the rescaled tensors as F32 and is about 1.7 GB for Qwen2.5-0.5B, and the report are written next to the output and removed afterwards, including after a failure. An existing output is refused before calibration starts.
8. **Device.** `--device cpu|cuda|auto` applies to the calibration step. The default is `cpu`, which matches `quantize`'s default.
9. **Limits of the method.** Single-file SafeTensors checkpoints only. The group size is the `--group-size` of `int4`, default 128, and the search uses ADR 0031's 20 values of alpha. The three scaling blocks of ADR 0031 are applied (the attention input, the MLP input, and up to down); the value-to-output pair is not.

## Verification

1. **Tests.** The Linux suite (Rust 1.85) passes with 371 tests, 0 failed and 2 ignored, and rustfmt is clean. Clippy on Rust 1.99 exits 0. The Rust integration tests refuse, without running Python, the wrong format, a missing data source, `--download` without `--calibration`, a missing data file, and an existing output. Unit tests cover the module. The Python tests check the window offsets against ADR 0031's recorded starts, and the refusals.

2. **End to end (Modal, L4).** `modelq quantize hf:Qwen/Qwen2.5-0.5B --format int4 --calibration awq --download --device cuda`, then `modelq eval` on the output over the full split. The run also checked the provenance metadata and that no temporary files were left.

## Results

| Quantity | Value |
| --- | --- |
| Calibration | 32 windows of 512 tokens, at the offsets of ADR 0031; 79.4 s on an L4 |
| Plan | 290 source tensors: 168 quantized, 122 preserved |
| Output | 462,747,768 bytes; validation passed (168 quantized tensors decoded, 122 preserved identical) |
| Perplexity, full split | original 13.069886; quantized 16.057086; increase +22.86% |
| Anchor (ADR 0031, AWQ INT4, embedding kept) | 16.05708635670797; difference 0.0 |
| KL divergence and top-1 agreement | 0.213160 nats per token; 77.17% |
| Provenance | `modelq.transform = awq-v1`; `modelq.calibration` holds the dataset revision and checksum, the offsets and the search summary |
| Temporary files left behind | none |

Against the default INT4 output with the embedding kept (ADR 0036, +39.19%, 462,649,320 bytes), AWQ lowers the perplexity cost by 16.3 points, and its output is 98,448 bytes larger. The rescaled norms are stored as F32, as the rescaled checkpoint writes them, and the calibration metadata adds a few kilobytes.

The evaluation also prints a per-matrix relative error. For AWQ that number is not a weight error: the file stores rescaled weights, and the evaluation compares them with the original. The perplexity and the KL divergence are the measures to use.

The validation record is [`adr-0037-awq-quantize-qwen2.5-0.5b.json`](../validation/adr-0037-awq-quantize-qwen2.5-0.5b.json).

## What this does not show

- One model. The calibration and the test text come from the same corpus (ADR 0031's caveat).
- The quality figures are simulated (weight-only decoding), as elsewhere in the evaluation. No runtime is measured.
- A CPU calibration takes much longer than the GPU run. The CPU time was not measured.
- GPTQ is not implemented.
- Sharded checkpoints are refused for calibration.

## Consequences

- The measured AWQ method is reachable from the CLI as an opt-in. Users who want better INT4 quality can ask for it.
- Calibration needs Python with PyTorch and transformers, and about 1.7 GB of free disk next to the output while it runs.
