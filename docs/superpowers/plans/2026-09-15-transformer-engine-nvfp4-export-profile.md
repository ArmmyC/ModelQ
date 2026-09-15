# Transformer Engine NVFP4 Rowwise Export Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add a CPU-only Transformer Engine NVFP4 rowwise 1x16 profile exporter that maps ModelQ's validated native tensor into padded rowwise buffers and explicit runtime field names without claiming runtime or hardware compatibility.

**Architecture:** Keep modelq_quant::nvfp4::QuantizedTensor as the numerical source of truth. Add one focused modelq_io::transformer_engine module that validates the source shape and native payload, copies packed rowwise data, expands E4M3 block scales into Transformer Engine's aligned scale matrix, derives tensor amax, and exposes deterministic field names. Do not add a container writer, CUDA code, columnwise data, swizzles, or CLI behavior.

**Tech Stack:** Stable Rust 1.85, existing modelq-io and modelq-quant crates, standard-library collections and checked arithmetic, existing Rust integration-test conventions, and Markdown ADR/documentation. No new dependencies.

**Spec:** docs/superpowers/specs/2026-09-15-transformer-engine-nvfp4-export-profile-design.md

## Global Constraints

- The profile identifier is transformer-engine.nvfp4.rowwise.1x16.v1.
- Rowwise packed data uses ModelQ's existing low-nibble-first E2M1 bytes without repacking.
- Rowwise E4M3 block scales occupy the leading [M, K / 16] region of a zero-padded [round_up(M, 128), round_up(K / 16, 4)] U8 matrix.
- Tensor amax is one F32 value; nonzero tensors use global_decode_scale * (448.0 * 6.0) and all-zero tensors use 0.0.
- Shapes must have rank at least two, positive dimensions, and a final dimension divisible by 16; shape arithmetic is checked for overflow.
- The result is profile-valid CPU data only; no Level 3 runtime-compatible or Level 4 hardware-validated claim is permitted.
- Preserve the scalar NVFP4 implementation as the numerical oracle and do not add Transformer Engine, PyTorch, CUDA, GPU, or other external dependencies.
- Keep main untouched until the feature branch is verified; use the plain branch name task-27-transformer-engine-profile.

---

### Task 1: Add the exporter module's red tests

**Files:**
- Create: tests/transformer_engine_nvfp4.rs
- Modify: crates/modelq-io/src/lib.rs:1-8 (module declaration will be added after the red test is written)

**Interfaces:**
- Consumes: modelq_io::transformer_engine::export_transformer_engine_nvfp4 and TransformerEngineNvfp4Error (the test names the intended public API before it exists), plus modelq_quant::nvfp4::quantize_shaped.
- Produces: executable examples of the profile contract that the implementation must satisfy.

- [ ] **Step 1: Write the failing happy-path and determinism tests**

Add tests that quantize a deterministic [2, 32] F32 source with quantize_shaped,
export it, and assert the exact public contract:

~~~rust
use modelq_io::transformer_engine::{
    export_transformer_engine_nvfp4, TRANSFORMER_ENGINE_NVFP4_PROFILE,
};
use modelq_quant::nvfp4::quantize_shaped;

#[test]
fn exports_rowwise_data_and_aligned_scale_matrix() {
    let source = (0..64).map(|index| index as f32 - 32.0).collect::<Vec<_>>();
    let native = quantize_shaped(&source, &[2, 32]).expect("native shape is valid");
    let exported = export_transformer_engine_nvfp4("layer.weight", &[2, 32], &native)
        .expect("the profile accepts a complete rowwise tensor");

    assert_eq!(TRANSFORMER_ENGINE_NVFP4_PROFILE, "transformer-engine.nvfp4.rowwise.1x16.v1");
    assert_eq!(exported.source_shape, vec![2, 32]);
    assert_eq!(exported.rowwise_data_shape, vec![2, 16]);
    assert_eq!(exported.rowwise_scale_inv_shape, vec![128, 4]);
    assert_eq!(exported.rowwise_data, native.packed_values());
    assert_eq!(&exported.rowwise_scale_inv[0..2], &native.block_scales()[0..2]);
    assert_eq!(&exported.rowwise_scale_inv[2..4], &[0, 0]);
    assert_eq!(&exported.rowwise_scale_inv[4..6], &native.block_scales()[2..4]);
    assert!(exported.rowwise_scale_inv[6..].iter().all(|&byte| byte == 0));
    assert!((exported.amax_rowwise - native.global_scale() * (448.0 * 6.0)).abs() < 1e-6);
    assert_eq!(exported.rowwise_data_name(), "layer.weight.rowwise_data");
    assert_eq!(
        exported.rowwise_scale_inv_name(),
        "layer.weight.rowwise_scale_inv"
    );
    assert_eq!(exported.amax_rowwise_name(), "layer.weight.amax_rowwise");
}

#[test]
fn repeated_exports_are_equal() {
    let source = [
        0.0_f32, 1.0, -2.0, 3.0, 4.0, -5.0, 6.0, -1.0,
        0.5, -0.75, 1.25, -1.5, 2.25, -2.5, 3.5, -4.0,
        2.0, -3.0, 4.0, -5.0, 6.0, -0.5, 0.75, -1.25,
        1.5, -2.25, 2.5, -3.5, 4.0, -4.5, 5.0, -6.0,
    ];
    let native = quantize_shaped(&source, &[2, 16]).expect("native shape is valid");
    let first = export_transformer_engine_nvfp4("weight", &[2, 16], &native)
        .expect("first export succeeds");
    let second = export_transformer_engine_nvfp4("weight", &[2, 16], &native)
        .expect("second export succeeds");
    assert_eq!(first, second);
}
~~~

- [ ] **Step 2: Add red validation tests**

Extend the same file with tests for the specified error boundaries:

~~~rust
#[test]
fn rejects_invalid_names_and_shapes() {
    let native = quantize_shaped(&[0.0_f32; 32], &[2, 16]).expect("native shape is valid");
    for name in ["", "__metadata__"] {
        let error = export_transformer_engine_nvfp4(name, &[2, 16], &native)
            .expect_err("profile names must be usable tensor names");
        assert!(matches!(error, TransformerEngineNvfp4Error::InvalidName { .. }));
    }
    for shape in [vec![], vec![32], vec![2, 0], vec![2, 8]] {
        let error = export_transformer_engine_nvfp4("weight", &shape, &native)
            .expect_err("the rowwise profile requires a matrix with 16-wide blocks");
        assert!(matches!(error, TransformerEngineNvfp4Error::InvalidShape { .. }));
    }
}

#[test]
fn rejects_shape_payload_mismatch() {
    let native = quantize_shaped(&[0.0_f32; 32], &[2, 16]).expect("native shape is valid");
    let error = export_transformer_engine_nvfp4("weight", &[2, 32], &native)
        .expect_err("the native payload must describe the requested shape");
    assert!(matches!(
        error,
        TransformerEngineNvfp4Error::PayloadLengthMismatch { .. }
    ));
}

#[test]
fn exports_zero_amax_for_an_all_zero_tensor() {
    let native = quantize_shaped(&[0.0_f32; 32], &[2, 16]).expect("native shape is valid");
    let exported = export_transformer_engine_nvfp4("weight", &[2, 16], &native)
        .expect("zero tensors have an explicit profile representation");
    assert_eq!(exported.amax_rowwise, 0.0);
    assert!(exported.rowwise_data.iter().all(|&byte| byte == 0));
    assert!(exported.rowwise_scale_inv.iter().all(|&byte| byte == 0));
}
~~~

- [ ] **Step 3: Run the focused test to confirm it fails for the missing module**

Run: cargo test -p modelq-io --test transformer_engine_nvfp4

Expected: compilation fails because modelq_io::transformer_engine, the export
function, the result type, and the error type do not exist yet.

- [ ] **Step 4: Commit the red tests**

~~~powershell
git add tests/transformer_engine_nvfp4.rs
git commit -m "test: specify Transformer Engine NVFP4 export profile"
~~~

### Task 2: Implement the CPU-only profile mapper

**Files:**
- Create: crates/modelq-io/src/transformer_engine.rs
- Modify: crates/modelq-io/src/lib.rs:1-8

**Interfaces:**
- Consumes: modelq_quant::nvfp4::QuantizedTensor, its packed_values, block_scales, global_scale, and len accessors.
- Produces: TRANSFORMER_ENGINE_NVFP4_PROFILE, TransformerEngineNvfp4Tensor, TransformerEngineNvfp4Error, and export_transformer_engine_nvfp4 for tests and later runtime/container work.

- [ ] **Step 1: Declare the module and public profile constants**

Add pub mod transformer_engine; to crates/modelq-io/src/lib.rs. In the new
module define:

~~~rust
pub const TRANSFORMER_ENGINE_NVFP4_PROFILE: &str =
    "transformer-engine.nvfp4.rowwise.1x16.v1";
pub const TRANSFORMER_ENGINE_NVFP4_BLOCK_SIZE: usize = 16;
pub const TRANSFORMER_ENGINE_NVFP4_SCALE_ROW_ALIGNMENT: usize = 128;
pub const TRANSFORMER_ENGINE_NVFP4_SCALE_COLUMN_ALIGNMENT: usize = 4;
~~~

Document that the module maps ModelQ-native data into profile-valid CPU fields
and does not allocate CUDA buffers or claim direct runtime loading.

- [ ] **Step 2: Define the error enum and result type**

Implement a documented TransformerEngineNvfp4Error with Debug, Clone, PartialEq,
and Eq where possible, plus Display and std::error::Error. Use these exact
variants:

~~~rust
pub enum TransformerEngineNvfp4Error {
    InvalidName { name: String },
    InvalidShape { shape: Vec<usize> },
    ShapeElementCountOverflow { shape: Vec<usize> },
    PayloadLengthMismatch { expected: usize, actual: usize },
    PackedLengthMismatch { expected: usize, actual: usize },
    BlockScaleCountMismatch { expected: usize, actual: usize },
    AlignmentOverflow { dimension: usize, alignment: usize },
    ScaleBufferSizeOverflow,
    InvalidGlobalScale { scale: f32 },
    DerivedAmaxNonFinite { scale: f32 },
    InvalidNativePayload { source: modelq_quant::nvfp4::Nvfp4Error },
}
~~~

Use messages that name the profile and the offending dimension/length. Do not
silently truncate, pad source data, or convert errors into String only.

- [ ] **Step 3: Define the owned output and deterministic name helpers**

Add:

~~~rust
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerEngineNvfp4Tensor {
    pub name: String,
    pub source_shape: Vec<usize>,
    pub rowwise_data: Vec<u8>,
    pub rowwise_data_shape: Vec<usize>,
    pub rowwise_scale_inv: Vec<u8>,
    pub rowwise_scale_inv_shape: Vec<usize>,
    pub amax_rowwise: f32,
}

impl TransformerEngineNvfp4Tensor {
    pub fn rowwise_data_name(&self) -> String;
    pub fn rowwise_scale_inv_name(&self) -> String;
    pub fn amax_rowwise_name(&self) -> String;
}
~~~

The helpers append .rowwise_data, .rowwise_scale_inv, and .amax_rowwise to the
exact source name. Keep source and physical shapes separate so consumers
cannot mistake padding for logical model dimensions.

- [ ] **Step 4: Implement checked shape and payload validation**

In export_transformer_engine_nvfp4, validate in this order:

1. reject an empty name and __metadata__;
2. reject rank below two, empty/zero dimensions, or a final dimension not
   divisible by 16;
3. compute elements = product(shape) and rows = product(shape[..last]) using
   checked_mul;
4. compare quantized.len() with elements;
5. compare packed length with modelq_quant::nvfp4::packed_len(elements);
6. compare block-scale length with modelq_quant::nvfp4::block_count(elements);
7. call modelq_quant::nvfp4::validate_parts and wrap any failure; and
8. reject a non-finite or non-positive native global scale.

Do not inspect or rewrite source floating-point values; the native quantized
object is already the numerical oracle.

- [ ] **Step 5: Implement the rowwise data and padded scale mapping**

Compute:

~~~rust
let k = *shape.last().expect("shape validation guarantees a final dimension");
let blocks_per_row = k / TRANSFORMER_ENGINE_NVFP4_BLOCK_SIZE;
let padded_rows = round_up(rows, TRANSFORMER_ENGINE_NVFP4_SCALE_ROW_ALIGNMENT)?;
let padded_blocks = round_up(
    blocks_per_row,
    TRANSFORMER_ENGINE_NVFP4_SCALE_COLUMN_ALIGNMENT,
)?;
~~~

Use checked round_up and checked multiplication for the scale buffer length.
Copy quantized.packed_values() unchanged into rowwise_data. Allocate a zeroed
padded_rows * padded_blocks scale buffer and copy each logical row's
blocks_per_row bytes from the native flat block-scale stream into the first
columns of that row. This makes rows and columns beyond the logical shape
deterministically zero without adding data padding.

Set rowwise_data_shape to the original shape with its final dimension halved
and rowwise_scale_inv_shape to [padded_rows, padded_blocks].

- [ ] **Step 6: Derive the tensor amax and construct the result**

If every native block-scale byte is zero, set amax_rowwise to 0.0. This is the
explicit all-zero profile case. Otherwise compute the F32 product
quantized.global_scale() * (448.0 * 6.0), reject a non-finite result, and use
that as the single tensor amax. Construct the owned result and return it.

Keep the global relationship in rustdoc: a nonzero consumer recovers the
global decode factor as amax_rowwise / (448.0 * 6.0), while the copied E4M3
bytes remain local block decode scales.

- [ ] **Step 7: Run the focused tests and refactor only after green**

Run: cargo test -p modelq-io --test transformer_engine_nvfp4

Expected: all profile mapping, naming, determinism, validation, and zero-amax
tests pass.

- [ ] **Step 8: Commit the implementation**

~~~powershell
git add crates/modelq-io/src/lib.rs crates/modelq-io/src/transformer_engine.rs
git commit -m "feat: add Transformer Engine NVFP4 export profile"
~~~

### Task 3: Add profile decision documentation and user-facing notes

**Files:**
- Create: docs/adr/0012-transformer-engine-nvfp4-export-profile.md
- Modify: README.md near the NVFP4/reference-fixture notes

**Interfaces:**
- Consumes: the implemented modelq_io::transformer_engine profile and the approved design spec.
- Produces: an accepted architecture record and clear user-facing compatibility language for future exporters.

- [ ] **Step 1: Write ADR 0012**

Document context, decision, alternatives, consequences, and compatibility
impact. Record the profile identifier, pinned recipe flags (1x16,
deterministic, 2D/RHT/stochastic/4over6 disabled), data/scale shapes, 128-by-4
scale padding, names, amax direction, and the fact that this is profile-valid
CPU data only. Link to the current primary Transformer Engine NVFP4 guide and
PyTorch API, plus ADRs 0010 and 0011. State explicitly that a serialized
container, CUDA object construction, GEMM swizzle, and Blackwell load test
remain outside this increment.

- [ ] **Step 2: Update README.md**

Add a short Transformer Engine NVFP4 profile paragraph explaining that the
library now exposes a CPU-only rowwise buffer planner, what the three fields
mean, and why the output is not yet a runtime-loadable Transformer Engine
checkpoint. Keep the existing optional fixture instructions and do not add a
new dependency or CLI command.

- [ ] **Step 3: Run Markdown and diff checks**

Run: git diff --check

Expected: no whitespace errors. Search the new ADR and README for TBD, TODO,
or contradictory runtime-compatible claims and remove any ambiguous wording
before committing.

- [ ] **Step 4: Commit documentation**

~~~powershell
git add docs/adr/0012-transformer-engine-nvfp4-export-profile.md README.md
git commit -m "docs: record Transformer Engine NVFP4 profile boundary"
~~~

### Task 4: Run the complete local quality gate

**Files:**
- No source changes expected; inspect all tracked changes.

**Interfaces:**
- Consumes: the completed profile module, integration tests, ADR, and README.
- Produces: verified branch state ready to publish.

- [ ] **Step 1: Run focused and workspace tests**

~~~powershell
cargo test -p modelq-io --test transformer_engine_nvfp4
cargo test --workspace --all-targets
~~~

Expected: both commands exit successfully; the optional external Transformer
Engine fixture test remains ignored when no fixture is configured.

- [ ] **Step 2: Run formatting and linting**

~~~powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
~~~

Expected: no formatting changes, warnings, or errors.

- [ ] **Step 3: Inspect the final diff and status**

~~~powershell
git diff --check
git status --short --branch
git diff main...HEAD --stat
git diff main...HEAD --name-only
~~~

Expected changed paths are only:

~~~text
crates/modelq-io/src/lib.rs
crates/modelq-io/src/transformer_engine.rs
tests/transformer_engine_nvfp4.rs
docs/adr/0012-transformer-engine-nvfp4-export-profile.md
README.md
docs/superpowers/specs/2026-09-15-transformer-engine-nvfp4-export-profile-design.md
docs/superpowers/plans/2026-09-15-transformer-engine-nvfp4-export-profile.md
~~~

The design spec and plan are already committed on this branch; do not add
generated fixtures, build output, dependency changes, CI files, or a Cargo
workspace change.

- [ ] **Step 4: Commit any final formatting-only adjustment, then publish**

If the quality gate changes formatting, commit only those intended changes:

~~~powershell
git add crates/modelq-io/src README.md docs/adr tests/transformer_engine_nvfp4.rs
git commit -m "style: format Transformer Engine NVFP4 profile"
~~~

Push the plain branch and record the resulting commit and hosted CI status:

~~~powershell
git push -u origin task-27-transformer-engine-profile
~~~

Do not fast-forward main until hosted branch checks are green. After the
branch checks pass, fast-forward main, push it, and wait for the main-branch
checks before reporting the task complete.
