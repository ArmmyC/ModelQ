# ADR 0038: AWQ Calibration for Sharded Checkpoints

- Status: Accepted (Task 60). The checks below ran on Modal and on the development PC. The numbers are in [`adr-0038-awq-sharded-checkpoints.json`](../validation/adr-0038-awq-sharded-checkpoints.json).
- Date: 2026-10-11
- Scope: `modelq quantize --format int4 --calibration awq` accepts a sharded SafeTensors checkpoint, meaning a model directory with `model.safetensors.index.json` and the shards it names. The single-file input and the INT4 output are unchanged. The calibration search now captures one decoder layer at a time, which changes its memory use and not its results.
- Builds on: [ADR 0037](0037-awq-quantize.md) (opt-in AWQ, single file only), [ADR 0031](0031-awq-calibration.md) (the method and its anchor), and [ADR 0003](0003-sharded-safetensors-input-design.md) (sharded input, which the data-free paths already read).

## Context

ADR 0037 accepts one `model.safetensors` and refuses anything else. Larger checkpoints are commonly split into shards, so calibration could not reach them, while the data-free paths read sharded input.

A first attempt at Qwen2.5-3B on an L4 (22 GiB usable) ran out of GPU memory during calibration. The search ran the model once over all layers and held every layer's block inputs until each layer was searched. For the 3B model that is about 1 GB per layer, roughly 36 GB in total, on top of the 12 GB F32 model. The 0.5B model fits, which is why ADR 0037 did not show the problem. The sharding was not the cause: an unsharded 3B would fail the same way.

## Decisions

1. **Input.** A model directory with `model.safetensors`, or with `model.safetensors.index.json` and every shard it names. When both forms are present, the single file is used. A missing shard is refused before any Python runs.
2. **The rescaled checkpoint keeps the input's layout.** A single-file input gives one rescaled file. A sharded input gives a temporary directory with the same shard names and the same weight map. The index's total size is recomputed, because the rescaled tensors are F32.
3. **Rescaling is per tensor.** A rescaled tensor depends only on its own values and on its block's scale. The shards are rescaled one at a time, and a layer whose tensors span two shards needs no special case. `up_proj` belongs to two blocks (C and D); its rescalings apply in block order, as the sequential fold always did.
4. **The INT4 output is unchanged.** It is one file unless `--max-shard-size` is given. The INT4 writer reads the rescaled directory as it reads any sharded checkpoint.
5. **Provenance.** The report records the layout: `checkpoint: {layout, files}`.
6. **Temporary files.** A sharded checkpoint is a directory next to the output. It is removed with the report, including after a failure.
7. **One layer's inputs at a time.** The calibration captures the block inputs of one decoder layer, searches that layer, and frees the inputs before the next layer. The model runs once per layer, so the forward work grows with depth. The blocks are the same as the all-at-once search's: on an L4, 72 of 72 blocks were bit-identical. The cache setting is left at its default. Turning the key and value cache off changed 69 of 72 scales on the GPU, so the code does not turn it off.
8. **Memory.** The model is still loaded in F32, as ADR 0037 does, so the calibration device must hold the full model plus one layer's inputs.

## Verification

- **Python, local.** `test_awq.py` and `test_modelq_awq.py`: a tiny random Qwen2 saved in shards by transformers, with at least one layer across two shards, rescales to tensors identical in value and dtype to the single-file rescaling. The index keeps its weight map and gets the recomputed size. A rescaling the index does not name is refused. The layout checks cover sharded, missing shard, neither file, and the precedence of the single file. The one-layer-at-a-time search matches the all-at-once search exactly on a tiny model, including a partial final batch. 23 tests pass (16 before).
- **Rust, Linux (Modal, Rust 1.85).** The workspace suite passes: 375 passed, 0 failed, 2 ignored (371 before this change). New tests: a missing shard is refused before Python runs; a directory with no weights is refused before Python runs; and a stand-in calibration script drives a sharded checkpoint end to end, writing the INT4 output, printing the `Calibration: awq-v1` line, and leaving no temporary files. `cargo fmt --check` and clippy with `-D warnings` are clean on the development PC.
- **GPU determinism, Modal L4, Qwen2.5-0.5B.** Each search was compared block by block with the all-at-once search, on the pinned WikiText-2 windows:

  | Comparison | Blocks with different scales | Alpha mismatches | Largest scale difference |
  | --- | --- | --- | --- |
  | All-at-once search, run twice | 0 of 72 | 0 | 0 |
  | One layer at a time, cache at its default | 0 of 72 | 0 | 0 |
  | One layer at a time, cache off | 69 of 72 | 0 | 1.9e-6 |

  Time on the L4: 63 s for the all-at-once search, 111 s for the one-layer-at-a-time search.

- **End to end, Modal L4, Qwen2.5-0.5B** (revision `060db64`). The checkpoint was re-sharded into 4 shards with the same 290 tensors and 988,065,536 bytes of tensor data. Both layouts were quantized with `--calibration awq --download`; each output is 462,747,824 bytes, and all 458 INT4 tensors are identical in name, dtype and value. The reports record `single-file` with 1 file and `sharded` with 4 files.
  The sharded output, evaluated on the full WikiText-2 test split, gives an original perplexity of 13.069886129532039 and a quantized perplexity of 16.05708635670797. That is exactly ADR 0031's anchor, a difference of 0.0. The KL divergence is 0.21315979392929757 nats per token, top-1 agreement is 77.171%, and the perplexity increase is +22.8556%. These match ADR 0037's stored values.
- **Real two-shard checkpoint, Modal L4.** Qwen/Qwen2.5-3B (revision `3aab1f1`, shards of 3.70 GB and 2.05 GB) was quantized with the same command. Calibration on CUDA took 865.0 s for 32 windows. The plan has 434 source tensors: 361 quantized and 73 preserved. The output is 2,096,724,536 bytes, and validation passed, with 361 quantized tensors decoded and 73 preserved tensors identical. The validation metrics are a maximum MSE of 5.738e-1 and a lowest SQNR of 10.30 dB. The report also counts one value beyond half a group scale, although its own text says symmetric rounding allows none. The 0.5B runs count zero. This is not yet explained.
- **Evaluation of the 3B output did not run.** `modelq eval` ran out of GPU memory while moving the 3B model onto the 22 GiB L4 (an 86 MiB allocation failed).

## What this does not show

- The 3B model's perplexity. Its evaluation needs more GPU memory than the L4 has.
- The cause of the one value beyond half a group scale in the 3B run. It may be a rounding tie at exactly half a scale, but that has not been checked.
- Calibration time on CPU, and for models other than the two above. The one-layer-at-a-time search takes about 1.8 times as long as the all-at-once search on the 0.5B model.
- GPTQ, which ADR 0028 keeps for later.
- Unusual shard layouts. The index must name only files that exist in the directory, and the check is strict.

## Consequences

- AWQ reads sharded checkpoints without a change of command. The 3B model calibrates and quantizes on a 22 GiB GPU.
- Calibration memory no longer grows with depth. Its time does, faster than linearly, because the model runs once per layer.
- Evaluating a 3B model needs more GPU memory than an L4 offers. `modelq eval` needs a change or a larger card; that is a follow-up, not part of this decision.
