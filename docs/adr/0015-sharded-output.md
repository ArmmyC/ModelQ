# ADR 0015: Sharded SafeTensors Output

- Status: Accepted and implemented (Task 33)
- Date: 2026-10-06
- Scope: `modelq quantize --max-shard-size <SIZE>` for both `int8` and `nvfp4`
- Supersedes: the "Output sharding policy" deferral in [ADR 0003](0003-sharded-safetensors-input-design.md)

## Context

ADR 0003 accepted sharded input but kept output as one file and reserved the `<stem>-NNNNN-of-MMMMM.safetensors` plus `<stem>.safetensors.index.json` convention for later. Quantized outputs of large checkpoints can be too big for one file to be convenient, and a sharded input should be convertible to a sharded output.

## Decision

`--max-shard-size <SIZE>` switches `--output` from a file path to a directory. `SIZE` is a positive integer of bytes with an optional suffix: `KB`, `MB`, `GB` (decimal) or `KiB`, `MiB`, `GiB` (binary), case-insensitive. Zero, fractions, negative values, unknown suffixes, and overflow are rejected before anything is written. Without the flag, behavior is unchanged: one file.

- **Layout.** The directory receives `model-00001-of-00003.safetensors` and so on, plus `model.safetensors.index.json` with `weight_map` (output tensor name to shard file) and `metadata.total_size` (sum of output payload bytes). This is the ADR 0003 input convention, so a sharded output reopens through `SafetensorsInput`, and through `modelq inspect` and `modelq quantize`.
- **Grouping.** Source tensors are taken in ascending name order and packed greedily into shards whose payload does not exceed the limit. Grouping is by source tensor, so everything derived from one source (`.qdata` with `.scale`, or `.qdata` with `.block_scale` and `.global_scale`) always shares a shard, and no tensor is split. A source whose output alone exceeds the limit gets its own shard. The limit counts tensor payload bytes; each shard additionally carries its small header. The result is deterministic and independent of input shard layout.
- **Self-contained shards.** Each shard is written by the existing single-file INT8 or NVFP4 writer over a `SubsetSource` of its group, so every shard carries its own manifest for exactly its tensors and is a valid ModelQ file on its own.
- **One shard is still sharded.** If everything fits in one shard, the output is still a directory with `model-00001-of-00001.safetensors` and an index, so the layout depends only on the flag.
- **Safety.** The directory must be absent or empty; a non-empty directory or an existing file at the path is rejected before any write. If any shard or the index fails, every file the run created is removed, and the directory too if the run created it, so a failed run leaves nothing behind. The index is written last, through a temporary file and a rename. The existing refusal to overwrite a source shard still applies.
- **Validation.** After writing, the command reopens the output directory with `SafetensorsInput`, which validates the index against every shard (tensor sets, duplicate names, `total_size`), then runs the same per-tensor checks as the single-file path.

## Consequences

- The tensors of a sharded output are byte-identical to the single-file output; tests assert this for both formats.
- Shard boundaries move only with the limit and tensor sizes, not with where the input was split.
- Reproducibility: two runs produce byte-identical shards and index.
- Not included: splitting one tensor across shards, a configurable stem, remote storage, resumable writes, and sharded Transformer Engine export.
