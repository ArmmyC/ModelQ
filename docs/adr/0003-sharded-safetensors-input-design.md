# ADR 0003: Sharded SafeTensors Input Discovery and Reading

- Status: Accepted for design; implementation deferred to a later change
- Date: 2026-08-21
- Scope: local SafeTensors checkpoint input

## Context

ModelQ currently opens one SafeTensors file at a time. The reader validates the
file header, exposes tensor summaries, and can create borrowed memory-mapped
views for the supported source dtypes. Large model checkpoints are commonly
stored as several independent SafeTensors files plus a JSON index. Task 14
needs a precise input contract before that support is added; it does not add
the reader implementation yet.

The SafeTensors format describes one self-contained file. Its header contains
tensor names, dtypes, shapes, and byte offsets relative to that file's data
buffer. A shard is therefore not a continuation of another shard: each shard
must be validated as a complete SafeTensors file. The index is a checkpoint
catalog that says which file owns each tensor name.

## Findings from existing conventions

The conventions below are used by Hugging Face Transformers and
`huggingface_hub` and are the baseline that ModelQ will accept:

| Convention | Observed form | Design implication |
| --- | --- | --- |
| Unsharded filename | `model.safetensors` (other ecosystem-specific stems also occur) | A directory can contain one ordinary SafeTensors file without an index. |
| Shard filename | `model-00001-of-00006.safetensors` | The ordinal in a filename is informative only; the index is authoritative. |
| Index filename | `model.safetensors.index.json` | The index is discovered from its suffix, not from parsed shard numbers. |
| Index root | JSON object with `metadata` and `weight_map` | `weight_map` is required; `metadata.total_size` is optional for compatibility. |
| Tensor mapping | `weight_map[tensor_name] = shard_filename` | Every tensor name must resolve to exactly one shard. |
| Index metadata | `{"total_size": <integer>}` | When present, it can be checked against the sum of shard payload sizes. |

The index's JSON object order is not a semantic ordering. SafeTensors itself
also treats each shard as a byte-buffer with independently validated offsets;
there is no format-level operation that concatenates shard payloads.

## Decision

ModelQ will add a metadata-first, read-only sharded input abstraction in a
future implementation. It will preserve the current single-file reader and
will make one logical tensor catalog from either one file or an index plus its
shards.

### Accepted input forms and discovery

The future input-opening operation will accept a local path with these rules:

1. An explicitly supplied regular file is opened as a single SafeTensors file,
   unless its name ends in `.index.json`, in which case it is treated as an
   index file. Explicit paths are validated by file contents; the extension is
   not a substitute for SafeTensors validation.
2. An explicitly supplied directory is inspected non-recursively:
   - collect files whose names end in `.safetensors.index.json`;
   - if exactly one index exists, use it and resolve its shard names relative
     to the directory;
   - if more than one index exists, fail with an ambiguity error and ask the
     caller to provide the intended index path explicitly;
   - if no index exists, collect files whose names end in `.safetensors`;
   - accept exactly one such file as an unsharded checkpoint; fail if there
     are none or more than one.
3. Discovery never recursively searches subdirectories, unions arbitrary
   SafeTensors files, or falls back to a single file when a discovered index is
   malformed or references a missing shard. A present index is authoritative.
4. Remote URLs, archive files, PyTorch `.bin` files, and framework-specific
   conversion are outside this reader contract.

This permits common stems such as `model.safetensors` and
`diffusion_pytorch_model.safetensors` while making a directory with multiple
possible checkpoints fail loudly instead of selecting one by filesystem order.
An explicit index path is the escape hatch for repositories containing several
independent model components.

### Index schema and validation

An index must be a UTF-8 JSON object with a `weight_map` object. Each
`weight_map` key is a logical tensor name and each value is a non-empty shard
filename string. The optional `metadata` member, when present, must be a JSON
object. The optional `metadata.total_size` value must be a non-negative JSON
integer. Unknown members are ignored so that additional producer metadata does
not make an otherwise valid checkpoint unreadable.

For the first implementation, shard references are **basenames only**. They
must not be absolute paths, drive/UNC paths, contain a directory separator, or
contain `.`/`..` path components. This matches the common index convention and
prevents an index from escaping its checkpoint directory. Supporting nested
relative shard paths can be considered separately once there is a concrete
checkpoint that needs them.

The reader will validate the complete index before returning an input handle:

- every referenced shard exists under the index directory and is a regular
  file;
- every referenced shard is parsed with the existing SafeTensors validation
  rules, including header bounds, dtype/shape byte lengths, and complete data
  coverage;
- every tensor found in a shard appears in `weight_map` exactly once;
- every `weight_map` entry appears in exactly one shard;
- `__metadata__` is file metadata, not a tensor, and cannot be a
  `weight_map` key;
- duplicate tensor names across shards are rejected;
- if `metadata.total_size` is present, it equals the checked sum of all tensor
  payload byte lengths; and
- a malformed index or any failed shard check reports the index path and, when
  applicable, the shard path and tensor name.

The index does not replace per-shard metadata. A shard's dtype, shape, and
offsets remain authoritative for its payload, while the index is authoritative
only for the tensor-to-shard relationship.

### Metadata-first and bounded payload access

Opening a sharded input has two phases:

1. Read and validate the index, then inspect each referenced shard's header and
   tensor summaries without copying tensor payloads.
2. Map or open shard payloads only when a caller requests tensor bytes or a
   typed view.

Sequential tensor processing will keep at most the currently used shard
payload mapping active. It will pass borrowed bytes or `TensorView` values to
the existing bounded-chunk quantization path and release the mapping before
moving to the next shard when possible. No API may require a concatenated
checkpoint `Vec<u8>` or a whole-checkpoint `Vec<f32>`.

The implementation may cache one recently used shard for repeated tensor
access, but caching is an optimization and not part of the correctness
contract. A tensor view cannot outlive the shard handle that owns its mapping.

### Deterministic logical tensor iteration

The sharded input will expose one logical summary per `weight_map` tensor. The
public iteration order is the ascending lexicographic order of the tensor
name's UTF-8 bytes (equivalent to Rust `String` ordering); it is independent of
JSON member order, directory enumeration order, shard filename ordinals, and
the order in which shards are opened.

Each summary will retain the source tensor name, dtype, shape, byte length, and
the owning shard path or shard identifier. A reader may group physical reads
by shard internally, but any externally visible tensor iterator and any
quantization/layout decision derived from it must use the canonical name order.
This matches ModelQ's existing deterministic output layout behavior.

### Errors and failure safety

The future reader will distinguish at least these failures:

- input path is neither a regular file nor a directory;
- directory has no unambiguous SafeTensors candidate;
- index JSON is malformed or has the wrong schema/type;
- a shard reference is unsafe, missing, unreadable, or not valid SafeTensors;
- index and shard tensor sets disagree;
- a tensor is mapped more than once; and
- `total_size` does not match validated payload sizes.

All validation happens before the quantization writer creates or replaces an
output file. A failed open therefore cannot produce a partial output or cause
ModelQ to silently skip a tensor.

### Output sharding policy

Task 14 covers **input sharding only**. ModelQ will continue to write one
SafeTensors output file for the current INT8 command. It will not emit an
output index, split a result into multiple files, or infer an output shard size
from the input directory.

If a future command is given an output directory or an explicit output-sharding
request before output sharding is implemented, it must reject the request
before creating any file. The eventual output design may reuse the
`<stem>-00001-of-000NN.safetensors` plus `<stem>.safetensors.index.json`
convention, but that is intentionally not committed by this ADR.

## Planned implementation shape

This ADR does not add public Rust types. When implementation begins, the
smallest expected boundary is conceptually equivalent to:

```text
SafetensorsInput::Single(MappedSafetensors)
SafetensorsInput::Sharded(ShardedSafetensors)

ShardedSafetensors {
    index_path,
    shard_catalog,
    tensors_sorted_by_name,
}
```

The existing `inspect_file`, `MappedSafetensors`, `TensorSummary`, and
`TensorView` behavior should be reused rather than duplicated. The sharded
layer owns discovery, index validation, tensor-to-shard lookup, and deterministic
iteration; it should not add quantization policy or CLI-specific behavior.

## Consequences

### Benefits

- Common Hugging Face-style sharded checkpoints can be consumed without
  guessing from shard filenames.
- Each shard remains independently SafeTensors-validated and memory-mappable.
- Tensor iteration and output planning are reproducible across operating
  systems and filesystem implementations.
- Index/path validation prevents accidental traversal outside the checkpoint
  directory and prevents silent tensor loss.
- Metadata-first access preserves the bounded-memory direction established by
  Task 13.

### Costs and limitations

- A directory containing multiple model components requires an explicit index
  path.
- The first implementation accepts only basename shard references.
- Opening a checkpoint must inspect every referenced shard before processing can
  start, even if the caller ultimately needs one tensor.
- Output sharding, remote storage, and non-SafeTensors checkpoint formats remain
  future work.

## Required tests when implemented

The implementation should add temporary-directory fixtures covering:

1. one unsharded file discovered from a directory;
2. two shards with tensors distributed across them and an index whose JSON
   member order differs from canonical tensor order;
3. deterministic name ordering across repeated opens and operating systems;
4. missing shard, malformed index, unsafe basename, duplicate tensor, missing
   index entry, extra shard tensor, and `total_size` mismatch errors;
5. multiple index files and multiple unindexed SafeTensors files producing an
   ambiguity error; and
6. a shard that contains unsupported-but-valid dtype metadata, proving that
   metadata inspection can preserve it while typed-view requests still report
   the existing unsupported-dtype error.

## References

- [SafeTensors format specification](https://github.com/huggingface/safetensors/blob/main/README.md#format)
- [SafeTensors metadata parsing and sharded metadata types](https://github.com/huggingface/safetensors/blob/main/docs/source/metadata_parsing.mdx)
- [Transformers sharded checkpoint documentation](https://huggingface.co/docs/transformers/main/big_models#sharded-checkpoints)
- [Transformers model-loading documentation](https://huggingface.co/docs/transformers/models#sharded-checkpoints)
- [Hugging Face Hub serialization and shard-index generation](https://huggingface.co/docs/huggingface_hub/en/package_reference/serialization)
