# Transformer Engine NVFP4 Runtime Container and GEMM Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Serialize one ModelQ rowwise NVFP4 matrix in a versioned SafeTensors contract, load it through a Transformer Engine 2.19.0 bridge, and prove one TN GEMM on Blackwell.

**Architecture:** Keep the existing Rust NVFP4 profile as the numeric source of truth and add a narrow SafeTensors writer beside the existing writer. A Rust fixture example emits the runtime artifact and a separate F32 reference artifact. A Python tool first validates both files on CPU, then—only in an explicit runtime mode—constructs the TE tensor and runs the one GEMM.

**Tech Stack:** Stable Rust 1.85 workspace, existing serde_json and ModelQ SafeTensors reader, Python 3.10+ with NumPy and SafeTensors, Transformer Engine 2.19.0, PyTorch, CUDA, and Linux Blackwell hardware for the optional runtime proof. No Rust dependency is added.

**Spec:** docs/superpowers/specs/2026-10-03-transformer-engine-nvfp4-runtime-container-design.md

For TE calls, use the release-pinned v2.19 sources linked by the spec, not the moving main branch: https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/pytorch/tensor/nvfp4_tensor.py and https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/pytorch/cpp_extensions/gemm.py.

## Global Constraints

- The runtime artifact contains exactly the three rowwise fields named in the spec; it contains no columnwise tensors, reference values, pickle, or Torch checkpoint.
- The manifest schema is version 1, the profile is transformer-engine.nvfp4.rowwise.1x16.v1, and the runtime pin is transformer-engine 2.19.0.
- The tested matrix and synthetic second operand are both [64, 64]; with `general_gemm(weight, operand, layout="TN")`, Transformer Engine 2.19's PyTorch row-major operation is `operand @ weight.T`, producing [64, 64]. This fixed proof does not claim a general shape contract.
- Store block scales with [128, 4] alignment padding and zero-filled padding; do not swizzle the serialized scale matrix.
- The Rust library and default Cargo tests remain CPU-only; do not add TE, PyTorch, CUDA, GPU, or Python dependencies to Cargo.toml or Cargo.lock.
- The Python CPU mode must validate both SafeTensors files without importing Transformer Engine or creating a CUDA context.
- The runtime command must fail, not skip or pass, when TE is not exactly 2.19.0, CUDA is unavailable, the GPU is below compute capability 10.0, or the numerical check fails.
- Use rtol=0.125 and atol=0.0675 for the GEMM comparison; use rtol=1e-5 and atol=1e-5 for TE dequantization versus the Rust F32 reference.
- Do not claim runtime or hardware compatibility unless the explicit Blackwell runtime command passes. Without that machine, report the hardware proof as unverified.
- Do not add a CLI command, full-model loader, CI workflow, benchmark, checked-in binary fixture, or general-purpose Transformer Engine checkpoint support.
- Continue on task-28-transformer-engine-nvfp4-container-design; do not create a codex/-prefixed branch.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| Modify crates/modelq-io/src/writer.rs | Add one profile-specific SafeTensors writer, schema validation, and deterministic serialization while preserving the existing INT8 writer behavior. |
| Create tests/transformer_engine_nvfp4_safetensors.rs | Exercise the Rust writer through ModelQ's existing SafeTensors reader, including manifest, shapes, payload bytes, padding, determinism, and failure behavior. |
| Create crates/modelq-io/examples/transformer_engine_nvfp4_fixture.rs | Generate a deterministic 64-by-64 runtime artifact and a separate F32 reference SafeTensors file. |
| Create tools/transformer_engine_nvfp4/validate.py | Provide a dependency-light CPU container check and a separate opt-in TE/Blackwell GEMM mode. |
| Create tools/transformer_engine_nvfp4/test_validate.py | Add stdlib unittest coverage for valid and malformed CPU artifacts; no GPU is required. |
| Create tools/transformer_engine_nvfp4/requirements.txt | List NumPy and SafeTensors for the CPU bridge and pin the Transformer Engine PyTorch extra to 2.19.0 for the Linux runtime bridge. |
| Create tools/transformer_engine_nvfp4/README.md | Document Python environments, artifact generation, CPU validation, Blackwell prerequisites, and runtime validation commands. |
| Create docs/adr/0013-transformer-engine-nvfp4-runtime-container.md | Record the implemented artifact boundary and the exact scope of any verified compatibility claim. |
| Modify README.md | Summarize the one-matrix container/runtime proof and state whether Blackwell validation is still unverified. |
| Existing branch file docs/superpowers/specs/2026-10-03-transformer-engine-nvfp4-runtime-container-design.md | Approved design contract. The user explicitly approved the Task 4 TN-operation correction after review identified the pinned TE semantic mismatch; keep the spec and plan aligned with that ruling. |
| Create docs/superpowers/plans/2026-10-03-transformer-engine-nvfp4-runtime-container.md | This task-by-task execution plan. |

Do not modify the existing profile mapper or general INT8 writer contract unless a failing focused test demonstrates a necessary compatibility fix. No Cargo manifest or lockfile changes are expected.

### Task 1: Specify and implement the profile-specific SafeTensors writer

**Files:**
- Modify: crates/modelq-io/src/writer.rs
- Create: tests/transformer_engine_nvfp4_safetensors.rs

**Interfaces:**
- Consumes: modelq_io::transformer_engine::TransformerEngineNvfp4Tensor, serde_json already used by modelq-io, MappedSafetensors, and the writer's existing temporary-file/atomic-destination helpers.
- Produces: modelq_io::writer::write_transformer_engine_nvfp4_safetensors(profile: &TransformerEngineNvfp4Tensor, destination: impl AsRef<Path>) -> Result<(), WriterError>.
- Add WriterError::InvalidTransformerEngineNvfp4Tensor { message: String } for invalid public profile buffers. Continue using existing destination, serialization, header-size, and I/O errors where they apply.

- [ ] **Step 1: Add the integration-test fixture and happy-path assertions**

In the new integration test, define profile_fixture(name: &str) -> TransformerEngineNvfp4Tensor by quantizing the deterministic values (0..64).map(|i| i as f32 - 32.0) with shape [2, 32] and calling export_transformer_engine_nvfp4. Define a TempArtifact RAII helper using a process-id plus AtomicU64 filename under std::env::temp_dir(); its Drop removes only that exact test file.

Add the main contract test using the API that the implementation will provide:

~~~rust
let output = TempArtifact::new("roundtrip");
let profile = profile_fixture("layer.weight");
write_transformer_engine_nvfp4_safetensors(&profile, output.path())
    .expect("valid rowwise profile writes");
let file = MappedSafetensors::open(output.path()).expect("output is valid SafeTensors");

assert_eq!(
    file.metadata().get("modelq.format").map(String::as_str),
    Some("transformer-engine-nvfp4-safetensors-v1")
);
let manifest: serde_json::Value = serde_json::from_str(
    file.metadata().get("modelq.manifest").expect("manifest exists"),
).expect("manifest is JSON");
assert_eq!(manifest["profile_id"], "transformer-engine.nvfp4.rowwise.1x16.v1");
assert_eq!(manifest["runtime"]["version"], "2.19.0");
assert_eq!(manifest["logical_shape"], serde_json::json!([2, 32]));
assert_eq!(file.tensors().len(), 3);
~~~

Also assert the exact three names and no others, exactly two metadata keys, U8/U8/F32 dtypes, and shapes [2, 16], [128, 4], and [1]. Check manifest schema_version=1; runtime exactly `{ "name": "transformer_engine", "version": "2.19.0" }`; all three field-name strings; `quantization` exactly contains E2M1, E4M3, block_size=16, and rowwise_1x16_tensor_global; `scale_storage` exactly contains padding [128,4] with gemm_swizzled=false; and denominator 2688.0. Assert the manifest does not contain the obsolete `encoding` or `scale_padding` keys. Compare rowwise_data and the full padded rowwise_scale_inv payload byte-for-byte with the profile, assert the logical leading scale regions match the profile, check every padding byte is zero, and compare the amax payload with profile.amax_rowwise.to_le_bytes().

- [ ] **Step 2: Add deterministic-write and invalid-profile tests**

Write the same profile to two different TempArtifact paths and assert that the full files are byte-for-byte equal. Add one invalid-profile test per validator boundary: empty name, non-rank-two shape, zero dimension, K not divisible by 16, overflow in derived byte/scale element counts, incorrect rowwise_data shape, incorrect rowwise_data length, incorrect rowwise_scale_inv shape or length, a nonzero scale-padding byte, nonfinite or negative amax, and an amax/logical-zero-scale mismatch. Each must assert InvalidTransformerEngineNvfp4Tensor and that the destination was not created. Clone a profile, alter rowwise_scale_inv_shape to [1, 1], call the writer, assert WriterError::InvalidTransformerEngineNvfp4Tensor, and assert that the destination was not created. Pre-create another destination with sentinel bytes and assert a second write returns DestinationExists without changing those bytes.

Use these assertions for the two failure cases:

~~~rust
let mut malformed = profile_fixture("weight");
malformed.rowwise_scale_inv_shape = vec![1, 1];
assert!(matches!(
    write_transformer_engine_nvfp4_safetensors(&malformed, invalid.path()),
    Err(WriterError::InvalidTransformerEngineNvfp4Tensor { .. })
));
assert!(!invalid.path().exists());

fs::write(existing.path(), b"preserve me").expect("sentinel writes");
assert!(matches!(
    write_transformer_engine_nvfp4_safetensors(&profile, existing.path()),
    Err(WriterError::DestinationExists { .. })
));
assert_eq!(fs::read(existing.path()).unwrap(), b"preserve me");
~~~

- [ ] **Step 3: Run the focused tests and confirm the red state**

Run: cargo test -p modelq --test transformer_engine_nvfp4_safetensors

Expected: compilation fails because write_transformer_engine_nvfp4_safetensors and InvalidTransformerEngineNvfp4Tensor do not exist yet.

- [ ] **Step 4: Define the profile validator and deterministic manifest**

Implement the named public function in writer.rs. Validate a nonempty, non-reserved tensor name; rank exactly two; positive M and K; K divisible by 16; rowwise_data_shape=[M,K/2]; data byte length M*K/2; rowwise_scale_inv_shape=[round_up(M,128),round_up(K/16,4)]; scale buffer length equal to the product of that shape; all scale padding zero; and finite amax. Use checked multiplication for derived data and scale element counts, rejecting overflow as an invalid profile instead of panicking or wrapping. Require amax=0.0 exactly when all logical block-scale bytes are zero; otherwise require amax>0.0. Return InvalidTransformerEngineNvfp4Tensor with the offending field in its message on any mismatch.

Use checked arithmetic in the local alignment helper rather than unchecked addition:

~~~rust
fn round_up_checked(value: usize, alignment: usize) -> Option<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}
~~~

After deriving padded_rows and padded_blocks, inspect each row's logical block scales at row * padded_blocks .. row * padded_blocks + K/16. Reject any nonzero byte in the row or column padding. Check the amax/zero-tensor rule against those logical scales.

Build modelq.manifest through BTreeMap-backed JSON objects so its serialized bytes are deterministic and match the approved design spec exactly: runtime is {"name":"transformer_engine","version":"2.19.0"}; encoding metadata is under `quantization` with `data_format=E2M1`, `block_scale_format=E4M3`, `block_size=16`, and `scaling=rowwise_1x16_tensor_global`; and padded scale metadata is under `scale_storage` with `padding=[128,4]` and `gemm_swizzled=false`. Also include schema_version=1, profile_id, tensor_name, logical_shape, the exact three field names, and global_scale_denominator=2688.0. Do not emit the alternate keys `encoding` or `scale_padding`, or use `transformer-engine` as the runtime name. Set modelq.format to transformer-engine-nvfp4-safetensors-v1. The file-level metadata map has exactly modelq.format and modelq.manifest.

- [ ] **Step 5: Write the three tensor records and payload atomically**

Emit only the profile's rowwise_data and rowwise_scale_inv U8 tensors and amax_rowwise F32 tensor with shape [1]. Compute checked contiguous offsets in lexicographic tensor-name order; use the same order when writing bytes. Encode amax with f32::to_le_bytes(). Reuse the existing unique temporary file and final rename pattern so an existing destination is never overwritten and an error does not leave a partial requested output.

The byte-writing order is weight.amax_rowwise, weight.rowwise_data, weight.rowwise_scale_inv (substitute profile.name for weight). Build each end offset with checked_add before serializing the descriptor:

~~~rust
let end = offset
    .checked_add(byte_len)
    .ok_or(WriterError::HeaderLengthOverflow)?;
descriptor.insert(
    "data_offsets".to_owned(),
    Value::Array(vec![Value::from(offset), Value::from(end)]),
);
offset = end;
~~~

Write the payload bytes in that same order: profile.amax_rowwise.to_le_bytes(), profile.rowwise_data, profile.rowwise_scale_inv.

- [ ] **Step 6: Run focused and existing writer tests**

Run:

~~~powershell
cargo test -p modelq --test transformer_engine_nvfp4_safetensors
cargo test -p modelq --test safetensors_writer
cargo fmt --all -- --check
~~~

Expected: all new serialization assertions pass and the existing INT8 writer tests remain unchanged and green.

- [ ] **Step 7: Commit the Rust writer and contract tests**

~~~powershell
git add crates/modelq-io/src/writer.rs tests/transformer_engine_nvfp4_safetensors.rs
git commit -m "feat: write Transformer Engine NVFP4 SafeTensors"
~~~

### Task 2: Add the deterministic Rust fixture generator

**Files:**
- Create: crates/modelq-io/examples/transformer_engine_nvfp4_fixture.rs

**Interfaces:**
- Consumes: quantize_shaped, export_transformer_engine_nvfp4, write_transformer_engine_nvfp4_safetensors, QuantizedTensor::dequantize, serde_json, and MappedSafetensors.
- Produces: command cargo run -p modelq-io --example transformer_engine_nvfp4_fixture -- <runtime-artifact.safetensors> <reference.safetensors>; the first path is the three-field runtime file, and the second is a separate one-tensor F32 reference file.

- [ ] **Step 1: Write failing tests for the local F32 reference writer**

Create the example source with a test module first. Define a TempArtifact RAII helper using a process ID and AtomicU64 filename in `std::env::temp_dir()`. Add a test that calls the not-yet-implemented local function `write_reference_safetensors(values: &[f32], destination: &Path) -> Result<(), Box<dyn std::error::Error>>`, reopens the result through MappedSafetensors, and asserts exactly one tensor named `weight.dequantized_reference`, dtype F32, shape [64,64], the exact reference metadata value, and payload bytes equal to the F32 values encoded in little-endian order. Set the first values to 1.0 and -2.5 so the test checks ordinary and negative values. Add a second test that pre-creates the destination with sentinel bytes, calls the function, and asserts it returns an error without changing the existing bytes.

Use this core assertion sequence:

~~~rust
let file = MappedSafetensors::open(output.path()).expect("reference file is valid SafeTensors");
assert_eq!(file.tensors().len(), 1);
assert_eq!(file.tensors()[0].name, "weight.dequantized_reference");
assert_eq!(file.tensors()[0].dtype, "F32");
assert_eq!(file.tensors()[0].shape, [64, 64]);
assert_eq!(
    file.metadata().get("modelq.reference_schema").map(String::as_str),
    Some("transformer-engine-nvfp4-reference-v1")
);
let expected = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
assert_eq!(file.tensor_bytes("weight.dequantized_reference").unwrap(), expected);
~~~

- [ ] **Step 2: Run the focused test and confirm the red state**

Run: `cargo test -p modelq-io --example transformer_engine_nvfp4_fixture`

Expected: compilation fails because the local `write_reference_safetensors` function does not exist yet.

- [ ] **Step 3: Implement the deterministic fixture generator and reference writer**

Require exactly two distinct output paths and print this usage on malformed arguments:

~~~text
usage: transformer_engine_nvfp4_fixture <runtime-artifact.safetensors> <reference.safetensors>
~~~

Inside fn run() -> Result<(), Box<dyn std::error::Error>>, generate exactly 4096 finite F32 values with this deterministic formula, quantize them with shape [64,64], export under tensor name weight, and send the profile to the Task 1 writer:

~~~rust
let values = (0_i32..4096)
    .map(|index| (((index * 37) % 251 - 125) as f32) / 32.0)
    .collect::<Vec<_>>();
let native = quantize_shaped(&values, &[64, 64])?;
let profile = export_transformer_engine_nvfp4("weight", &[64, 64], &native)?;
write_transformer_engine_nvfp4_safetensors(&profile, runtime_path)?;
~~~

Dequantize the same QuantizedTensor and serialize the result in a one-tensor SafeTensors file named weight.dequantized_reference, dtype F32, shape [64,64], with metadata modelq.reference_schema=transformer-engine-nvfp4-reference-v1. Use little-endian F32 bytes, the standard eight-byte header-length prefix, and an eight-byte-aligned JSON header. Write to a uniquely named sibling temporary file opened with create_new(true), sync it, and rename only after confirming the destination does not exist; remove only that temporary file on failure. Keep this helper local to the example; do not expose a general F32 writer API.

Dequantize the same QuantizedTensor and serialize the result in a one-tensor SafeTensors file named weight.dequantized_reference, dtype F32, shape [64,64], with metadata modelq.reference_schema=transformer-engine-nvfp4-reference-v1. Use little-endian F32 bytes, the standard eight-byte header-length prefix, and an eight-byte-aligned JSON header. Write to a uniquely named sibling temporary file opened with create_new(true), sync it, and rename only after confirming the destination does not exist; remove only that temporary file on failure. Keep this helper local to the example; do not expose a general F32 writer API.

The exact one-tensor header before SafeTensors padding is:

~~~json
{
  "__metadata__": {
    "modelq.reference_schema": "transformer-engine-nvfp4-reference-v1"
  },
  "weight.dequantized_reference": {
    "dtype": "F32",
    "shape": [64, 64],
    "data_offsets": [0, 16384]
  }
}
~~~

Build the reference payload with values.iter().flat_map(|value| value.to_le_bytes()). Write the padded JSON length as a little-endian u64, then the JSON bytes, ASCII-space padding to an eight-byte boundary, and the 16384 payload bytes.

Reopen both files with MappedSafetensors and assert the runtime file has exactly three tensors and the reference tensor is F32 [64,64]. Print both paths and tensor shapes only after those checks pass. On any error, write a path-specific diagnostic to stderr and return a nonzero exit status.

- [ ] **Step 4: Run focused tests and exercise the example in a temporary directory**

Run: `cargo test -p modelq-io --example transformer_engine_nvfp4_fixture`

Run: cargo run -p modelq-io --example transformer_engine_nvfp4_fixture -- <temp>/te-runtime.safetensors <temp>/te-reference.safetensors

Expected: both files are generated, the Rust reader reopens them, and no fixture is created under the repository. Run the command again with two fresh paths and compare corresponding Get-FileHash SHA256 values; both pairs must match byte-for-byte.

- [ ] **Step 5: Commit the fixture generator**

~~~powershell
git add crates/modelq-io/examples/transformer_engine_nvfp4_fixture.rs
git commit -m "test: add Transformer Engine NVFP4 fixture generator"
~~~

### Task 3: Implement and test CPU-only Python artifact validation

**Files:**
- Create: tools/transformer_engine_nvfp4/validate.py
- Create: tools/transformer_engine_nvfp4/test_validate.py
- Create: tools/transformer_engine_nvfp4/requirements.txt

**Interfaces:**
- Consumes: the Task 1 manifest/tensor schema, Task 2 fixture names, Python standard library, NumPy, and SafeTensors.
- Produces: ValidationError; frozen ValidatedArtifact(manifest: dict[str, Any], tensors: dict[str, numpy.ndarray]); validate_container(path: pathlib.Path) -> ValidatedArtifact; validate_reference(path: pathlib.Path, expected_shape: tuple[int,int]) -> numpy.ndarray; validate_cpu_fixture(artifact_path: pathlib.Path, reference_path: pathlib.Path) -> tuple[ValidatedArtifact,numpy.ndarray]; and CLI subcommand cpu.
- CPU code must not import torch or any transformer_engine module. Keep those imports inside the runtime function added in Task 4.

- [ ] **Step 1: Add CPU validator tests for a valid artifact**

Import builtins, copy, json, pathlib, tempfile, unittest, unittest.mock, numpy as np, validate, and safetensors.numpy.save_file. Define VALID_MANIFEST, valid_tensors(), and write_artifact(path, *, manifest=None, tensors=None, file_format="transformer-engine-nvfp4-safetensors-v1"). write_artifact serializes the manifest with sort_keys=True and separators=(",", ":"), supplies the two metadata keys, writes via save_file, and returns path. Use a TemporaryDirectory per test. Start with this manifest and array set; each tensor name is derived from tensor_name:

~~~python
VALID_MANIFEST = {
    "schema_version": 1,
    "profile_id": "transformer-engine.nvfp4.rowwise.1x16.v1",
    "runtime": {"name": "transformer_engine", "version": "2.19.0"},
    "tensor_name": "weight",
    "logical_shape": [2, 16],
    "fields": {
        "rowwise_data": "weight.rowwise_data",
        "rowwise_scale_inv": "weight.rowwise_scale_inv",
        "amax_rowwise": "weight.amax_rowwise",
    },
    "quantization": {
        "data_format": "E2M1",
        "block_scale_format": "E4M3",
        "block_size": 16,
        "scaling": "rowwise_1x16_tensor_global",
    },
    "scale_storage": {"padding": [128, 4], "gemm_swizzled": False},
    "global_scale_denominator": 2688.0,
}
def valid_tensors():
    return {
        "weight.rowwise_data": np.zeros((2, 8), dtype=np.uint8),
        "weight.rowwise_scale_inv": np.zeros((128, 4), dtype=np.uint8),
        "weight.amax_rowwise": np.zeros((1,), dtype=np.float32),
    }


def write_artifact(path, *, manifest=None, tensors=None,
                   file_format="transformer-engine-nvfp4-safetensors-v1"):
    actual_manifest = copy.deepcopy(VALID_MANIFEST if manifest is None else manifest)
    actual_tensors = valid_tensors() if tensors is None else tensors
    metadata = {
        "modelq.format": file_format,
        "modelq.manifest": json.dumps(actual_manifest, sort_keys=True, separators=(",", ":")),
    }
    save_file(actual_tensors, str(path), metadata=metadata)
    return pathlib.Path(path)


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.temp_directory = pathlib.Path(self.temp.name)

    def test_accepts_valid_artifact(self):
        path = write_artifact(self.temp_directory / "valid.safetensors")
        artifact = validate.validate_container(path)
        self.assertEqual(artifact.manifest, VALID_MANIFEST)
        self.assertEqual(artifact.tensors["weight.rowwise_data"].shape, (2, 8))

    def test_accepts_separate_reference(self):
        path = self.temp_directory / "reference.safetensors"
        save_file(
            {"weight.dequantized_reference": np.zeros((2, 16), dtype=np.float32)},
            str(path),
            metadata={"modelq.reference_schema": "transformer-engine-nvfp4-reference-v1"},
        )
        reference = validate.validate_reference(path, (2, 16))
        self.assertEqual(reference.shape, (2, 16))
        self.assertEqual(reference.dtype, np.float32)
~~~

- [ ] **Step 2: Add malformed-container tests**

Create independent temporary artifacts and assert ValidationError in separately named tests: test_rejects_missing_scale_field, test_rejects_wrong_format, test_rejects_wrong_te_version, test_rejects_wrong_data_dtype, test_rejects_wrong_data_shape, test_rejects_nonzero_scale_padding, test_rejects_extra_columnwise_tensor, test_rejects_nonfinite_amax, and test_rejects_inconsistent_manifest_field_names. Each test changes exactly one metadata value or tensor and calls validate_container.

Add this method to ValidatorTests:

~~~python
    def test_rejects_missing_scale_field(self):
        tensors = valid_tensors()
        del tensors["weight.rowwise_scale_inv"]
        path = write_artifact(self.temp_directory / "missing-scale.safetensors", tensors=tensors)
        with self.assertRaisesRegex(validate.ValidationError, "rowwise_scale_inv"):
            validate.validate_container(path)
~~~

- [ ] **Step 3: Run tests to confirm the red state**

Run: python -m unittest discover -s tools/transformer_engine_nvfp4 -p "test_*.py" -v

Expected: import or attribute errors identify the missing validate.py API.

- [ ] **Step 4: Implement CPU-safe schema and payload validation**

Read files with safetensors.safe_open(..., framework="np"); validate metadata before accepting tensors. Require exactly three tensors, exact manifest version/profile/runtime values, exact field-name mapping, rank-two positive logical shape, K divisible by 16, and exact U8/U8/F32 dtypes and derived shapes. Verify the scale padding rows and columns are all zero and the amax is finite and nonnegative. The reference reader must require modelq.reference_schema=transformer-engine-nvfp4-reference-v1 and exactly one finite F32 tensor named weight.dequantized_reference with the expected logical shape.

Implement the container reader in this order so schema errors precede runtime work:

~~~python
with safe_open(str(path), framework="np") as reader:
    metadata = reader.metadata() or {}
    tensors = {name: reader.get_tensor(name) for name in reader.keys()}
manifest = json.loads(metadata["modelq.manifest"])
validate_manifest(metadata, manifest, tensors)
return ValidatedArtifact(manifest=manifest, tensors=tensors)
~~~

The validation helpers must compare each array's NumPy dtype and shape, derive padded scale dimensions with integer ceiling division, inspect both padding regions, and reject any unexpected tensor name.

- [ ] **Step 5: Add the CPU CLI and direct dependency list**

Implement argparse command cpu <artifact> <reference>. It calls validate_cpu_fixture, prints the schema/profile and shape, and states that this is a CPU/container check only. Return status zero only on full success; print ValidationError to stderr and return nonzero otherwise.

~~~python
commands = parser.add_subparsers(dest="command", required=True)
cpu = commands.add_parser("cpu", help="validate SafeTensors fields without CUDA")
cpu.add_argument("artifact", type=pathlib.Path)
cpu.add_argument("reference", type=pathlib.Path)
~~~

Set requirements.txt to:

~~~text
numpy>=1.24,<3
safetensors>=0.4,<1
transformer-engine[pytorch]==2.19.0
~~~

The CPU-only install instructions will install only NumPy and SafeTensors. The runtime environment must already have a CUDA-enabled PyTorch build compatible with its CUDA toolkit; then install this full requirements file to add the pinned Transformer Engine PyTorch integration. Keep that GPU environment separate from CPU-only validation.

Add a test of validate_cpu_fixture using the valid artifact/reference pair and a test that it rejects a reference with the wrong logical shape.

- [ ] **Step 6: Run Python tests and validate Rust-generated files**

Run:

~~~powershell
python -m pip install "numpy>=1.24,<3" "safetensors>=0.4,<1"
python -m unittest discover -s tools/transformer_engine_nvfp4 -p "test_*.py" -v
python tools/transformer_engine_nvfp4/validate.py cpu <temp>/te-runtime.safetensors <temp>/te-reference.safetensors
~~~

Expected: all positive and negative CPU tests pass; the generated pair validates without importing Transformer Engine or running a GPU operation.

- [ ] **Step 7: Commit the CPU bridge and requirements**

~~~powershell
git add tools/transformer_engine_nvfp4/validate.py tools/transformer_engine_nvfp4/test_validate.py tools/transformer_engine_nvfp4/requirements.txt
git commit -m "feat: validate Transformer Engine NVFP4 artifacts on CPU"
~~~

### Task 4: Add the explicit Transformer Engine 2.19.0 Blackwell GEMM mode

**Files:**
- Modify: tools/transformer_engine_nvfp4/validate.py
- Modify: tools/transformer_engine_nvfp4/test_validate.py

**Interfaces:**
- Consumes: validate_cpu_fixture, ValidatedArtifact, the reference array, and the v2.19.0 NVFP4Tensor/NVFP4Quantizer/general_gemm APIs.
- Produces: run_blackwell_gemm(artifact_path: pathlib.Path, reference_path: pathlib.Path) -> dict[str, Any] with keys max_abs_error, python, pytorch, cuda_runtime, cudnn, driver, transformer_engine, gpu, and compute_capability; and CLI subcommand runtime.
- Uses only the TN path with the exported matrix as the first, logically transposed operand and the TE-quantized synthetic rowwise matrix as the second operand; the resulting PyTorch operation is `operand @ weight.T`.

- [x] **Step 1: Test that runtime preflight rejects unsupported environments**

Add these methods to ValidatorTests around this pure helper: validate_runtime_preflight(system: str, machine: str, te_version: str, cuda_available: bool, cuda_version: str | None, cudnn_version: int | None, capability: tuple[int,int] | None) -> None. Assert it rejects a non-Linux OS, a machine other than x86_64/AMD64, a TE package version not exactly 2.19.0, CUDA unavailable, a CUDA version below 12.8, cuDNN below 90300, and compute capability below 10.0. Assert each error is explicit and nonzero; do not treat unsupported hardware as unittest skip or success.

~~~python
def test_runtime_preflight_rejects_wrong_te_version(self):
    with self.assertRaisesRegex(validate.ValidationError, "2.19.0"):
        validate.validate_runtime_preflight("Linux", "x86_64", "2.19.1", True, "12.8", 90300, (10, 0))

def test_runtime_preflight_rejects_missing_cuda(self):
    with self.assertRaisesRegex(validate.ValidationError, "CUDA"):
        validate.validate_runtime_preflight("Linux", "x86_64", "2.19.0", False, None, None, None)

def test_runtime_preflight_rejects_old_cuda_and_cudnn(self):
    with self.assertRaisesRegex(validate.ValidationError, "CUDA 12.8"):
        validate.validate_runtime_preflight("Linux", "x86_64", "2.19.0", True, "12.1", 90300, (10, 0))
    with self.assertRaisesRegex(validate.ValidationError, "cuDNN 9.3"):
        validate.validate_runtime_preflight("Linux", "x86_64", "2.19.0", True, "12.8", 90200, (10, 0))

def test_runtime_preflight_rejects_pre_blackwell_gpu(self):
    with self.assertRaisesRegex(validate.ValidationError, "compute capability"):
        validate.validate_runtime_preflight("Linux", "x86_64", "2.19.0", True, "12.8", 90300, (9, 0))

def test_runtime_preflight_rejects_unsupported_platform(self):
    with self.assertRaisesRegex(validate.ValidationError, "Linux"):
        validate.validate_runtime_preflight("Windows", "x86_64", "2.19.0", True, "12.8", 90300, (10, 0))
    with self.assertRaisesRegex(validate.ValidationError, "x86_64"):
        validate.validate_runtime_preflight("Linux", "aarch64", "2.19.0", True, "12.8", 90300, (10, 0))
~~~

- [x] **Step 2: Implement lazy runtime imports and hardware checks**

Run validate_cpu_fixture before importing torch or Transformer Engine. Pass platform.system() and platform.machine() into validate_runtime_preflight and require Linux plus x86_64 (accept the normalized alias AMD64). Check importlib.metadata.version("transformer-engine") equals "2.19.0" and convert a missing distribution into ValidationError. Then import torch, query torch.cuda.is_available(), torch.version.cuda, torch.backends.cudnn.version(), and the active-device capability, and pass those values to validate_runtime_preflight. Parse CUDA's major/minor numbers and require at least 12.8; require cuDNN's integer version to be at least 90300; require capability major >= 10. Read the installed driver with nvidia-smi --query-gpu=driver_version --format=csv,noheader and the active GPU name from torch.cuda.get_device_name(); fail if the driver query cannot be completed. Keep errors actionable and return no report on any failed check.

- [x] **Step 3: Construct the exported TE weight without altering its serialized fields**

On the runtime path, bind artifact, reference_values = validate_cpu_fixture(artifact_path, reference_path) and manifest = artifact.manifest. Require logical shape [64,64], copy the three NumPy arrays to CUDA tensors, and build NVFP4Quantizer with fp4_dtype=DType.kFloat4E2M1, rowwise=True, columnwise=False, with_2d_quantization=False, with_rht=False, stochastic_rounding=False, with_random_sign_mask=False, and nvfp4_use_4over6=False.

Prepare device inputs without changing logical values:

~~~python
device = torch.device("cuda", torch.cuda.current_device())
rowwise_data = torch.as_tensor(
    artifact.tensors[manifest["fields"]["rowwise_data"]], device=device
)
rowwise_scale_inv = torch.as_tensor(
    artifact.tensors[manifest["fields"]["rowwise_scale_inv"]], device=device
)
amax_rowwise = torch.as_tensor(
    artifact.tensors[manifest["fields"]["amax_rowwise"]], device=device
)
~~~

Use these v2.19 imports and quantizer/tensor constructor arguments:

~~~python
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch import NVFP4Quantizer, NVFP4Tensor
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

quantizer = NVFP4Quantizer(
    fp4_dtype=DType.kFloat4E2M1,
    rowwise=True,
    columnwise=False,
    with_2d_quantization=False,
    with_rht=False,
    stochastic_rounding=False,
    with_random_sign_mask=False,
    nvfp4_use_4over6=False,
)
weight = NVFP4Tensor(
    shape=(64, 64),
    dtype=torch.float32,
    rowwise_data=rowwise_data,
    rowwise_scale_inv=rowwise_scale_inv,
    columnwise_data=None,
    columnwise_scale_inv=None,
    amax_rowwise=amax_rowwise,
    amax_columnwise=None,
    fp4_dtype=DType.kFloat4E2M1,
    quantizer=quantizer,
    with_gemm_swizzled_scales=False,
    row_scaled_nvfp4=False,
    nvfp4_use_4over6=False,
    device=device,
)
~~~

Let the pinned TE GEMM path perform any required internal runtime preparation. Do not requantize, transpose, manufacture columnwise/swizzled arrays, rewrite the artifact, or save runtime-prepared buffers.

- [x] **Step 4: Run the TE dequantization and one TN GEMM oracle**

Under torch.no_grad(), compare weight.dequantize(dtype=torch.float32) with the Rust reference at rtol=1e-5 and atol=1e-5. Build deterministic F32 values for the [64,64] second operand from the same integer formula used by the fixture, quantize only that operand with an equivalent rowwise 1x16 NVFP4Quantizer, and dequantize it for the reference. For `general_gemm(weight, rhs_quantized, layout="TN")`, Transformer Engine 2.19 computes the row-major result `rhs_dequantized @ reference_weight.T`.

Before changing the GEMM oracle, add a CPU-only regression test for `tn_reference_output(rhs, weight)`. Use the hand-checkable nonsymmetric matrices `weight=[[1,2],[3,4]]` and `rhs=[[5,6],[7,8]]`; require the literal result `[[17,39],[23,53]]`. This test must fail if the helper instead computes `weight.T @ rhs`, so the transpose/order conflict is observable without CUDA.

Use this exact oracle sequence. rhs_values is the deterministic [64,64] F32 matrix generated from the integer formula in Task 2; rhs_quantized is produced only by the TE quantizer:

~~~python
with torch.no_grad():
    reference_weight = torch.as_tensor(reference_values, dtype=torch.float32, device=device)
    indices = torch.arange(4096, dtype=torch.int32, device=device)
    rhs_values = (((indices * 37) % 251) - 125).to(torch.float32).div_(32.0).reshape(64, 64)
    torch.testing.assert_close(
        weight.dequantize(dtype=torch.float32),
        reference_weight,
        rtol=1e-5,
        atol=1e-5,
    )
    rhs_quantized = quantizer.quantize(rhs_values)
    rhs_dequantized = rhs_quantized.dequantize(dtype=torch.float32)
    out, _, _, _ = general_gemm(
        weight,
        rhs_quantized,
        out_dtype=torch.float32,
        layout="TN",
    )
    expected = tn_reference_output(rhs_dequantized, reference_weight)
    assert tuple(out.shape) == (64, 64)
    assert torch.isfinite(out).all().item()
    torch.testing.assert_close(out, expected, rtol=0.125, atol=0.0675)
    max_abs_error = (out - expected).abs().max().item()
~~~

Report maximum absolute error. Any constructor, preparation, kernel, shape, finite-value, or tolerance failure is a hard error.

- [x] **Step 5: Add runtime CLI output and CPU-mode isolation test**

Implement runtime <artifact> <reference> by calling run_blackwell_gemm. Report Python, PyTorch, CUDA runtime, NVIDIA driver, TE, GPU model, and compute capability. Keep CPU mode independent: add a test that blocks imports of torch and transformer_engine while CPU validation runs, then assert it succeeds on the valid fixture pair.

The import-isolation test can patch the import hook after importing the CPU dependencies:

~~~python
real_import = builtins.__import__

def reject_gpu_imports(name, *args, **kwargs):
    if name == "torch" or name.startswith("transformer_engine"):
        raise AssertionError(f"CPU validation imported {name}")
    return real_import(name, *args, **kwargs)

with unittest.mock.patch("builtins.__import__", side_effect=reject_gpu_imports):
    validate.validate_cpu_fixture(artifact_path, reference_path)
~~~

- [ ] **Step 6: Run available Python checks and the explicit hardware proof**

Always run:

~~~bash
python -m unittest discover -s tools/transformer_engine_nvfp4 -p 'test_*.py' -v
python tools/transformer_engine_nvfp4/validate.py cpu /tmp/te-runtime.safetensors /tmp/te-reference.safetensors
~~~

On Linux x86_64 with CUDA 12.8 or newer, a compatible NVIDIA driver, cuDNN 9.3 or newer, Transformer Engine 2.19.0, and Blackwell-or-newer hardware, run:

~~~bash
python tools/transformer_engine_nvfp4/validate.py runtime /tmp/te-runtime.safetensors /tmp/te-reference.safetensors
~~~

Expected on a supported host: one GEMM passes both numerical comparisons and prints all runtime versions. If no supported host is available, record the runtime proof as unverified; do not mark this step as passing and do not claim Level 3 or Level 4.

- [x] **Step 7: Commit the opt-in runtime validator**

~~~powershell
git add tools/transformer_engine_nvfp4/validate.py tools/transformer_engine_nvfp4/test_validate.py
git commit -m "test: add Blackwell Transformer Engine NVFP4 GEMM proof"
~~~

### Task 5: Record the contract and user-facing compatibility status

**Files:**
- Create: docs/adr/0013-transformer-engine-nvfp4-runtime-container.md
- Create: tools/transformer_engine_nvfp4/README.md
- Modify: README.md

**Interfaces:**
- Consumes: completed Rust writer, fixture command, Python CPU/runtime commands, approved design spec, and actual runtime-test result if a compatible host was available.
- Produces: an accepted ADR 0013, reproducible environment instructions, and accurate root README status.

- [x] **Step 1: Write ADR 0013 from the implemented behavior**

Record goal, context, accepted container contract, exact manifest values, three fields and physical shapes, padding rules, TE 2.19.0 constructor/GEMM boundary, numerical oracle, tolerances, prerequisites, alternatives, and consequences. Distinguish the CPU container check from the hardware GEMM proof. Record Level 3/4 only if the runtime command actually passed; otherwise state that hardware validation is unverified.

- [x] **Step 2: Document CPU and Blackwell setup separately**

In tools/transformer_engine_nvfp4/README.md, show:

~~~bash
cargo run -p modelq-io --example transformer_engine_nvfp4_fixture -- /tmp/te-runtime.safetensors /tmp/te-reference.safetensors
python -m pip install 'numpy>=1.24,<3' 'safetensors>=0.4,<1'
python tools/transformer_engine_nvfp4/validate.py cpu /tmp/te-runtime.safetensors /tmp/te-reference.safetensors
~~~

Then document Linux x86_64, CUDA 12.8+, compatible NVIDIA driver, cuDNN 9.3+, Blackwell-or-newer GPU, and a compatible CUDA-enabled PyTorch install. Show installation of requirements.txt (which selects the TE PyTorch extra pinned at 2.19.0) in the separate runtime environment, then the explicit runtime command. State that the CPU command is not a runtime-compatibility claim and that an unavailable GPU is not a pass.

- [x] **Step 3: Update the root README without overstating support**

Add a short Task 28 paragraph that describes one matrix, three rowwise SafeTensors fields, the pinned bridge, and one GEMM. Link ADR 0013 and the tool README. If no Blackwell runtime has passed, say hardware compatibility is unverified; do not describe ModelQ as loading a model or providing inference.

- [x] **Step 4: Review wording and commit documentation**

Run git diff --check. Search the ADR and both READMEs for unfinished-marker text, unsupported whole-model claims, and any Level 3/4 statement inconsistent with the actual test result.

~~~powershell
git add docs/adr/0013-transformer-engine-nvfp4-runtime-container.md tools/transformer_engine_nvfp4/README.md README.md
git commit -m "docs: record Transformer Engine NVFP4 runtime boundary"
~~~

### Task 6: Run the full quality gate and publish the verified branch

**Files:**
- No source changes expected; inspect the complete branch diff.

**Interfaces:**
- Consumes: all Rust, Python, example, and documentation changes from Tasks 1-5.
- Produces: a clean verified branch and an explicit report of whether the Blackwell proof passed.

- [x] **Step 1: Run Rust tests, format, and lint**

Run:

~~~powershell
cargo test --workspace --all-targets
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
~~~

Expected: all commands pass with no warnings. Cargo.lock remains unchanged because no Rust dependency was added.

- [x] **Step 2: Generate fresh artifacts and rerun the CPU validation**

Generate runtime and reference files outside the repository with the fixture example. Reopen them through the Rust reader via the example, run the Rust contract tests, run all Python unittest tests, and run the Python cpu command. Confirm that the runtime artifact contains exactly three tensors and the reference file is separate.

- [ ] **Step 3: Run the Blackwell test only on an eligible host** — this Windows host is ineligible; hardware proof remains unverified.

If the host meets the documented Linux/CUDA/driver/cuDNN/TE/Blackwell prerequisites, run the explicit runtime command and preserve its printed environment and error metrics in the task report. If it cannot run here, leave the status unverified and do not call it a passing test.

- [ ] **Step 4: Inspect the final file list and status**

Track the approved plan as its own documentation commit before the final diff review:

~~~powershell
git add docs/superpowers/plans/2026-10-03-transformer-engine-nvfp4-runtime-container.md
git commit -m "docs: add Transformer Engine NVFP4 implementation plan"
~~~

~~~powershell
git diff --check
git status --short --branch
git diff main...HEAD --stat
git diff main...HEAD --name-only
~~~

Expected branch paths are limited to the files listed in this plan, including the approved spec and this plan. There must be no target output, Python bytecode, generated SafeTensors files, Cargo dependency change, CLI feature, or workflow.

- [ ] **Step 5: Push the plain-named feature branch after local verification**

~~~powershell
git push -u origin task-28-transformer-engine-nvfp4-container-design
~~~

Do not merge into main unless the user explicitly asks. If the GPU proof was unavailable, say so plainly even when every CPU and Rust check passes.
