# Native NVFP4 SafeTensors Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a library-only ModelQ-native NVFP4 SafeTensors planner and writer that follows ADR 0011 and can reopen a synthetic output for scalar reconstruction.

**Architecture:** Keep the existing INT8 layout and writer behavior unchanged. Add an NVFP4-specific plan type and writer entry point in `modelq-io`; the planner expands each explicitly selected source tensor into `.qdata`, `.block_scale`, and `.global_scale` payloads and preserves every unselected source tensor. The writer uses the existing mapped-reader and temporary-output safety patterns, while the scalar `modelq_quant::nvfp4::quantize_shaped` function remains the numerical oracle.

**Tech Stack:** Stable Rust 1.85, `modelq-core`, `modelq-quant`, `modelq-io`, `serde_json`, standard filesystem I/O, and existing SafeTensors mmap support. No new dependencies, CLI flags, GPU code, or runtime exporter.

**Spec:** `docs/adr/0011-nvfp4-native-safetensors-convention.md`

## Global Constraints

- Preserve the existing INT8 planner/writer and public behavior.
- Quantized source shapes must be non-empty, have positive dimensions, and have a final dimension divisible by 16.
- Emit U8 packed qdata, U8 E4M3 block-scale bytes, and a scalar F32 decode scale with the exact ADR 0011 names/shapes.
- Preserve unselected source tensors byte-for-byte.
- Reject reserved names, generated-name collisions, unsupported quantized source dtypes, malformed plans, non-finite values, and in-place destinations.
- Keep the scalar implementation as the correctness reference; do not add a runtime compatibility claim.
- Keep Rust 1.85 compatibility and run formatting, clippy, build, tests, docs, and diff checks.

---

### Task 1: Define the NVFP4 output plan and its red tests

**Files:**
- Create: `crates/modelq-io/src/nvfp4.rs`
- Modify: `crates/modelq-io/src/lib.rs`
- Test: `crates/modelq-io/src/nvfp4.rs` unit tests

**Interfaces:**
- `pub enum Nvfp4OutputRole { Preserved, QuantizedData, BlockScales, GlobalScale }`
- `pub struct Nvfp4OutputTensorPlan { pub name: String, pub source_name: String, pub dtype: String, pub shape: Vec<usize>, pub byte_len: u64, pub data_offsets: Range<u64>, pub role: Nvfp4OutputRole }`
- `pub struct Nvfp4OutputPlan { pub tensors: Vec<Nvfp4OutputTensorPlan>, pub total_data_bytes: u64 }`
- `pub fn plan_nvfp4_output(sources: &[TensorSummary], quantized_names: &[String]) -> Result<Nvfp4OutputPlan, Nvfp4LayoutError>`

- [ ] **Step 1: Write the failing deterministic-plan test**

Use `TensorSummary` values for `weight: F32 [2, 16]` and `norm: F32 [16]`, select only `weight`, and assert lexicographic source order: preserved `norm` first, then `weight.qdata`, `weight.block_scale`, and `weight.global_scale` with physical shapes `[2, 8]`, `[2, 1]`, `[]`, byte lengths `16`, `2`, `4`, and contiguous offsets `64..80`, `80..82`, `82..86` after the preserved 64-byte tensor.

- [ ] **Step 2: Run the focused test and verify the expected missing-API failure**

Run:

```text
cargo test -p modelq-io nvfp4::tests::plans_selected_tensor_and_preserves_the_rest
```

Expected: compilation fails because the NVFP4 plan types/function do not yet exist.

- [ ] **Step 3: Add the minimal public plan types and module export**

Add `pub mod nvfp4;` and the four plan roles/fields above. Keep fields needed by the writer public and keep selection details private until the plan API is settled.

- [ ] **Step 4: Implement only enough planning to satisfy the deterministic test**

Sort sources by name, emit selected companions in `.qdata`, `.block_scale`, `.global_scale` order, preserve other tensors, calculate checked byte ranges, and return the plan.

- [ ] **Step 5: Run the focused test and the existing `modelq-io` unit tests**

Expected: the new test and all pre-existing I/O tests pass.

### Task 2: Add validation/error coverage before broadening the planner

**Files:**
- Modify: `crates/modelq-io/src/nvfp4.rs`

**Interfaces:**
- `Nvfp4LayoutError` must report duplicate/missing/unexpected selection names, reserved names, generated collisions, unsupported source dtypes, invalid shapes, shape-product overflow, and output-byte overflow.
- `Nvfp4OutputPlan::tensor(&self, name: &str) -> Option<&Nvfp4OutputTensorPlan>` must support writer validation.

- [ ] **Step 1: Write failing tests**

Cover: final dimension `8`, zero/empty shape, F64 or U8 selected for quantization, selected name not in sources, duplicate selection, source name `__metadata__`, source name colliding with another tensor's generated suffix, and a shape whose checked product overflows.

- [ ] **Step 2: Run the focused validation tests and confirm each fails for the missing error/validation behavior**

Run:

```text
cargo test -p modelq-io nvfp4::tests::rejects_
```

- [ ] **Step 3: Implement checked shape, dtype, name, selection, and offset validation**

Use `u64` checked products for byte planning, require quantized source dtypes `F32`, `F16`, or `BF16`, derive qdata/block-scale shapes from the final dimension, and reject every collision before writing.

- [ ] **Step 4: Run all NVFP4 planner tests and refactor only after green**

Run:

```text
cargo test -p modelq-io nvfp4::tests
```

### Task 3: Add the writer’s red round-trip test and metadata contract

**Files:**
- Modify: `crates/modelq-io/src/nvfp4.rs`
- Modify: `crates/modelq-io/src/lib.rs`
- Test: `tests/nvfp4_writer.rs`

**Interfaces:**
- `pub enum Nvfp4WriterError`
- `pub fn write_nvfp4_safetensors(source: &MappedSafetensors, plan: &Nvfp4OutputPlan, destination: impl AsRef<Path>) -> Result<(), Nvfp4WriterError>`

- [ ] **Step 1: Write a failing synthetic round-trip integration test**

Create a source SafeTensors fixture containing `weight: F32 [1, 16]` and a preserved `ids: U8 [2]`. Plan `weight`, write the output, reopen it with `MappedSafetensors`, assert the five metadata/payload names, inspect the ADR 0011 manifest, and compare the unpacked/dequantized weight values to `modelq_quant::nvfp4::quantize_shaped`.

- [ ] **Step 2: Run the integration test and verify it fails because the writer API is absent**

Run:

```text
cargo test --test nvfp4_writer writes_and_reopens_native_nvfp4_output
```

- [ ] **Step 3: Implement deterministic header generation**

Emit the required file metadata, `modelq.nvfp4.manifest.v1`, preserved records, quantized records, U8/F32 descriptors, physical shapes, and planned data offsets using ordered maps and compact JSON.

- [ ] **Step 4: Implement payload writing**

For each selected tensor, collect that tensor’s mapped F32/F16/BF16 values, call `quantize_shaped`, write packed bytes, block-scale bytes, and the little-endian F32 global scale to the planned ranges. Copy preserved bytes directly from the source mapping.

- [ ] **Step 5: Reuse the existing safe destination behavior**

Reject invalid/in-place/existing destinations, write to a unique temporary file beside the destination, flush/sync, refuse a destination that appeared during conversion, and rename only after all payloads succeed. Report source, quantization, serialization, offset, and I/O errors through `Nvfp4WriterError`.

- [ ] **Step 6: Run the round-trip test and existing writer tests**

Expected: the new output reopens as valid SafeTensors, its manifest is self-describing, and existing INT8 writer behavior remains unchanged.

### Task 4: Add determinism and failure-safety tests

**Files:**
- Modify: `tests/nvfp4_writer.rs`
- Modify: `crates/modelq-io/src/nvfp4.rs` if error messages need focused coverage

- [ ] **Step 1: Write failing tests for deterministic repeated writes**

Write the same source and plan twice and assert byte-for-byte identical outputs, including metadata ordering and all payload bytes.

- [ ] **Step 2: Write failing tests for non-finite input, destination collision, and in-place output**

Assert the source remains unchanged, an existing destination remains unchanged, and no partial destination is left behind after each failure.

- [ ] **Step 3: Implement only the missing safety checks**

Keep the writer’s failure behavior atomic and ensure the supplied plan is recomputed from the current source and selection before any output is created.

- [ ] **Step 4: Run the focused integration tests**

Run:

```text
cargo test --test nvfp4_writer
```

### Task 5: Documentation, quality gates, and handoff

**Files:**
- Modify: `README.md` to state that the native NVFP4 writer exists but remains runtime-independent.
- Modify: `docs/adr/0011-nvfp4-native-safetensors-convention.md` only if implementation details expose a genuine ambiguity.

- [ ] **Step 1: Add the README usage/status paragraph**

Document the library-only writer and explicitly state that no CLI flag, runtime exporter, GPU path, or hardware validation is included.

- [ ] **Step 2: Run the full local quality commands**

Run:

```text
cargo fmt --all -- --check
cargo build --workspace --all-targets
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --no-deps
git diff --check
```

If Windows Application Control blocks an existing executable, record the exact command and rely on the hosted matrix for the complete all-targets result; do not hide the limitation.

- [ ] **Step 3: Review the diff against ADR 0011**

Check names, shapes, dtypes, manifest fields, compatibility claims, collision rules, and no-overwrite behavior line by line.

- [ ] **Step 4: Commit the focused implementation**

```text
git add docs/superpowers/plans/2026-08-26-nvfp4-safetensors-writer.md crates/modelq-io/src/lib.rs crates/modelq-io/src/nvfp4.rs tests/nvfp4_writer.rs README.md
git commit -m "feat: add native NVFP4 SafeTensors writer"
```
