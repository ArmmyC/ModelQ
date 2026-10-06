# Sharded SafeTensors Input Implementation Plan (Task 29)

**Goal:** Implement the input contract of [ADR 0003](../../adr/0003-sharded-safetensors-input-design.md): one logical, deterministic, read-only tensor catalog over a single file, a directory, or an `*.safetensors.index.json` plus its shards.

**Architecture:** Add `crates/modelq-io/src/sharded.rs`, reusing `MappedSafetensors` and `TensorSummary` unchanged. The new layer owns discovery, index validation, tensor-to-shard lookup, and name-ordered iteration. No quantization policy, no CLI changes, no new dependencies (`serde_json` and `memmap2` already present).

**Branch:** `task-29-sharded-safetensors-input`, from `main`. Independent of PR #1.

## Constraints

- Preserve `MappedSafetensors` behavior and public API.
- Validate everything (index, every shard header, tensor-set agreement, `total_size`) in `open`, before any output could be created.
- Shard references are basenames only; reject absolute, drive/UNC, separators, `.`/`..`, empty.
- Logical iteration is ascending UTF-8 byte order of tensor name.
- Payload access maps lazily: hold at most one shard mapping at a time for sequential access; views cannot outlive their owning handle.
- Unsupported-but-valid dtypes stay inspectable; typed views still return `UnsupportedTensorDtype`.
- Output sharding stays out of scope.

## Interfaces

```rust
pub struct ShardedTensorSummary { pub summary: TensorSummary, pub shard: PathBuf }
pub enum SafetensorsInput { Single(..), Sharded(..) }   // opened via SafetensorsInput::open(path)
impl SafetensorsInput {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ShardedError>;
    pub fn tensors(&self) -> &[ShardedTensorSummary];            // name order
    pub fn with_tensor_bytes<R>(&self, name: &str, f: impl FnOnce(&[u8]) -> R) -> Result<R, ShardedError>;
    pub fn with_tensor<R>(&self, name: &str, f: impl FnOnce(TensorView<'_>) -> R) -> Result<R, ShardedError>;
}
```

`with_*` closures keep the mapping scoped to the call (satisfies the "release before next shard" rule without a self-referential type). `ShardedError` wraps `SafetensorsError` and adds the ADR's distinct failures (not file/dir, ambiguous or missing candidate, bad index schema, unsafe/missing shard, tensor-set mismatch, duplicate mapping, `total_size` mismatch), each carrying index path / shard path / tensor name where applicable.

## Tasks

### 1. Failing tests first (`tests/sharded_safetensors.rs`)
Temp-dir fixtures built by a small helper that writes valid SafeTensors files. Cover ADR "Required tests" 1-6: single file from directory; two shards with index member order differing from name order; repeated-open determinism; missing shard, malformed index, unsafe basenames (`../x`, `a/b`, absolute, empty), duplicate tensor across shards, `weight_map` entry absent from shards, shard tensor absent from `weight_map`, `total_size` mismatch, `__metadata__` as key; multiple indexes / multiple unindexed files / no files; unsupported dtype preserved in summary but rejected by `with_tensor`; explicit index path; explicit file path with `.index.json` suffix treated as index.

### 2. Discovery and index parsing
Directory rules per ADR; index schema validation; basename check. Unit tests for the basename validator inside the module.

### 3. Catalog validation
Open each shard with `MappedSafetensors::open`, build name→shard map, cross-check against `weight_map`, check `total_size` with checked arithmetic, sort by name. Drop mappings after validation.

### 4. Lazy payload access
Re-open the owning shard inside `with_tensor_bytes` / `with_tensor`; optionally cache the last shard behind a `RefCell`/`Mutex` only if a test shows reopening is a problem (not required for correctness).

### 5. Docs
Update ADR 0003 status to "Implemented (library-only)", add a README paragraph, state CLI is unchanged.

### 6. Verification
`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`. Windows native tests may be blocked by Smart App Control; run in Linux/CI and report honestly.

## Follow-up (not in this task)
Task 30: wire `SafetensorsInput` into the CLI and add an NVFP4 export command.
