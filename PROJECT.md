# ModelQ Project Brief

> **Status:** project definition and implementation plan  
> **Language:** Rust  
> **Repository:** `ArmmyC/ModelQ`  
> **Primary goal:** build a cross-platform, inference-independent model quantization compiler/toolkit that can transform model checkpoints which may be too large to run on the user's machine into smaller, explicitly described quantized representations for supported runtimes and hardware.

---

## 1. Project summary

ModelQ is a Rust-native model quantization framework and end-user tool.

The core user story is simple:

1. A user downloads a model checkpoint.
2. The original model is too large to run comfortably on the user's hardware.
3. ModelQ reads the checkpoint without requiring model inference.
4. ModelQ quantizes the model using CPU by default, with optional GPU acceleration later.
5. ModelQ writes a smaller checkpoint in a representation that is either:
   - ModelQ-native and explicitly self-described, or
   - compatible with a specific supported inference runtime and hardware target.
6. The user runs the resulting model with the compatible runtime.

Example future workflows:

```bash
modelq quantize ./model --format int8 --device cpu
```

```bash
modelq quantize ./model --format int4 --group-size 128 --device auto
```

```bash
modelq quantize ./model \
  --target nvidia-blackwell \
  --runtime tensorrt \
  --format nvfp4
```

ModelQ should also be useful as an experimentation platform for unusual representations such as INT3, INT2, INT1, custom bit widths, custom scale types, and new low-precision formats.

The project is intentionally challenging. It should expose us to numerical formats, model checkpoint formats, bit packing, memory mapping, streaming I/O, SIMD, parallel programming, GPU compute, hardware-specific data layouts, interoperability, testing numerical software, CLI/GUI design, and Rust systems engineering.

---

## 2. Product philosophy

### 2.1 ModelQ is a quantization compiler, not an inference engine

ModelQ transforms model weights and metadata.

Conceptually:

```text
High precision checkpoint
        |
        v
+-----------------------+
|       ModelQ          |
|                       |
| parse                 |
| inspect               |
| plan                  |
| quantize              |
| pack                  |
| validate              |
| export                |
+-----------+-----------+
            |
            v
Quantized checkpoint
```

The core does **not** need to:

- tokenize prompts
- generate tokens
- implement attention
- implement KV caches
- implement sampling
- implement inference scheduling
- provide CUDA/Metal inference kernels
- train models
- provide autograd

This boundary keeps the project focused.

### 2.2 No inference does not mean no validation

ModelQ should still verify the numerical transformation.

For a tensor `W`:

```text
W -> quantize -> Q(W) -> dequantize -> W_hat
```

ModelQ can compare `W` and `W_hat` without ever running the neural network.

Useful metrics include:

- MSE
- MAE
- maximum absolute error
- SQNR
- saturation/clipping count
- compression ratio
- source bytes vs quantized bytes
- scale overhead
- number of tensors quantized, preserved, or skipped

This allows ModelQ to detect broken encoders, bad scaling, packing bugs, and extreme numerical degradation.

### 2.3 CPU must remain a universal fallback

A user should not need a compatible GPU just to create a quantized checkpoint.

Long-term execution backends:

```text
Execution backend
    |
    +-- CPU scalar       Windows / Linux / macOS
    +-- CPU SIMD         x86-64 / ARM64
    +-- CUDA             Linux / Windows, optional
    +-- Metal            macOS, optional
```

A target format must not automatically imply that quantization itself must run on that target hardware unless the algorithm truly requires it.

For example, the architecture should allow the idea of producing an NVIDIA-targeted format using a CPU on another platform if the representation and exporter can be implemented correctly.

### 2.4 Cross-platform architecture from day one

ModelQ should target:

- Linux
- Windows
- macOS

Initial support should mean the scalar CPU implementation compiles and works on all three.

Platform-specific acceleration comes later.

Permanent portability rules:

1. Core quantization concepts must not depend on an OS.
2. CUDA must never be a mandatory dependency.
3. Metal must never be a mandatory dependency.
4. Use `std::path::{Path, PathBuf}` instead of constructing paths with string separators.
5. Do not depend on shell commands such as `sed`, `grep`, `cp`, or `rm` for core functionality.
6. Put OS-specific code behind clear modules, traits, and/or Cargo feature flags.
7. CI should eventually run formatting, linting, and tests on Linux, Windows, and macOS.

---

## 3. Important terminology and separations

Several concepts that look similar must remain independent in the design.

### 3.1 Source checkpoint format

Examples:

- SafeTensors
- GGUF later
- PyTorch `.bin` / `.pt` eventually, if worth supporting

This describes how the source tensors are stored.

### 3.2 Quantization representation

Examples:

- INT8
- INT4
- FP8 E4M3
- FP8 E5M2
- FP4 E2M1
- NVFP4
- experimental INT3 / INT2 / INT1

This describes how values are represented.

### 3.3 Quantization algorithm

Examples:

- symmetric min-max
- asymmetric min-max
- per-tensor scaling
- per-channel scaling
- group-wise scaling
- block-wise scaling
- MSE-optimized scaling
- future calibration-based algorithms

The storage representation and the algorithm are not the same thing.

### 3.4 Execution backend

Examples:

- scalar CPU
- parallel CPU
- AVX2 / AVX-512
- ARM NEON
- CUDA
- Metal

This answers: **where does quantization execute?**

### 3.5 Target hardware

Examples:

- generic CPU
- Apple Silicon
- NVIDIA Blackwell

This answers: **what hardware is the output intended for?**

### 3.6 Output runtime

Examples:

- ModelQ-native reader
- llama.cpp / GGML
- TensorRT
- NVIDIA Transformer Engine ecosystem
- another explicitly supported runtime

This answers: **who will consume the result?**

### 3.7 Container/output format

Examples:

- SafeTensors
- GGUF
- a future ModelQ-native container if necessary

A quantized representation is not automatically equivalent to a file format.

Keep the mental model:

```text
execution backend
      !=
target hardware
      !=
quantization representation
      !=
quantization algorithm
      !=
container format
      !=
inference runtime
```

---

## 4. Compatibility levels

ModelQ must be explicit about what "supported" means.

A major project risk is producing bytes that are mathematically correct but claiming that another runtime can use them when its required layout or metadata differs.

Define compatibility in levels:

### Level 1: representation-valid

ModelQ can encode and decode the representation correctly according to our implementation/specification.

Example: ModelQ can encode E2M1 values and reconstruct them.

### Level 2: container-valid

The generated file is structurally valid for its container format.

Example: the output SafeTensors file can be parsed as SafeTensors.

### Level 3: runtime-compatible

A named runtime and supported version can load and use the generated representation.

Example: a particular GGUF quantization output successfully loads in a tested llama.cpp version.

### Level 4: hardware-validated

The output has been tested with the target runtime on the intended hardware.

Example: a Blackwell NVFP4 artifact is successfully used by the supported NVIDIA runtime on Blackwell hardware.

**Never claim Level 3 or Level 4 compatibility just because Level 1 works.**

The CLI/report should eventually expose this distinction.

---

## 5. ModelQ-native representation vs runtime exporters

SafeTensors is excellent for safe tensor storage and memory-mapped reading, but arbitrary low-bit representations may not have a universally understood SafeTensors dtype/layout convention.

Therefore the project should distinguish:

### ModelQ-native quantized tensors

Used for:

- developing algorithms
- round-trip tests
- diagnostics
- experimentation
- formats unsupported by mainstream runtimes

A ModelQ-native SafeTensors convention may store packed bytes and scale tensors with explicit metadata. It is **not** automatically runtime-compatible.

Example conceptual representation:

```text
layer.weight.qdata      U8 packed payload
layer.weight.scales     F32/F16 scales
layer.weight.zeros      optional zero points

metadata:
  modelq.quantization = int4
  modelq.group_size = 128
  modelq.scheme = symmetric
  modelq.packing = ...
```

The exact convention should be specified in an ADR before stabilizing it.

### Runtime exporters

Runtime-specific exporters take ModelQ's internal quantized representation and generate the exact layout, metadata, tensor naming, packing, and container conventions required by a supported runtime.

Example:

```text
QuantizedTensor IR
       |
       +-- ModelQ SafeTensors exporter
       +-- GGUF exporter
       +-- NVIDIA runtime exporter
       +-- future exporters
```

This is where true interoperability belongs.

---

## 6. NVFP4 scope

NVFP4 is an important long-term target and a useful test of the architecture.

At a high level, NVIDIA NVFP4 uses:

- FP4 E2M1 values
- fine-grained FP8 E4M3 scaling
- an additional FP32 global scale
- hardware/runtime-specific expectations around layout and usage

The exact algorithm, grouping/layout, scale semantics, and serialization must be implemented against current NVIDIA documentation and reference implementations at the time the feature is built.

Important rule:

> "ModelQ can mathematically encode NVFP4-like tensors" and "ModelQ produces a checkpoint directly usable by a particular Blackwell runtime" are separate milestones.

For NVFP4 work, always verify against current primary NVIDIA sources such as:

- Transformer Engine documentation
- NVIDIA Model Optimizer documentation
- CUTLASS documentation and examples
- TensorRT documentation where runtime compatibility is claimed

Do not infer runtime compatibility from E2M1 encoding alone.

---

## 7. Data-free vs calibration-based quantization

ModelQ v0.x should focus on **data-free, weight-only quantization**.

Supported philosophy for early versions:

```text
checkpoint weights
      |
      v
inspect weights
      |
compute scales/parameters from weights
      |
quantize
      |
write
```

No forward pass is required.

Examples in early scope:

- symmetric INT8
- group-wise INT4
- basic FP4/FP8 experiments
- weight-only NVFP4 representation when the algorithm can be defined from weights alone

Out of initial scope:

- AWQ-style calibration workflows
- GPTQ-style workflows
- activation quantization requiring representative datasets
- QAT
- training

However, the architecture should not make future calibration impossible.

Long-term conceptual split:

```text
QuantizationRecipe
    |
    +-- DataFreeRecipe
    |
    +-- CalibrationRecipe   future
```

Do not implement calibration abstractions until there is a concrete algorithm that requires them.

---

## 8. Architecture

Long-term architecture:

```text
                         +----------------+
                         |      CLI       |
                         +-------+--------+
                                 |
                         +-------v--------+
                         |      GUI       |  later, same library API
                         +-------+--------+
                                 |
                                 v
+------------+        +----------------------+        +----------------+
| Checkpoint |------->| Quantization Planner |------->| Compatibility  |
| Reader     |        +----------+-----------+        | / Target Rules |
+------------+                   |                    +----------------+
                                 v
                        +--------------------+
                        | Quantization Engine|
                        +---------+----------+
                                  |
                   +--------------+--------------+
                   |              |              |
                   v              v              v
                INT8           INT4           FP/NVFP4
                   |              |              |
                   +--------------+--------------+
                                  |
                                  v
                         QuantizedTensor IR
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
              SafeTensors       GGUF        future runtime
               exporter        exporter        exporter
```

### 8.1 Streaming pipeline

A fundamental requirement is processing models that the user cannot load for inference.

Initial target:

```text
open checkpoint
      |
read metadata/header
      |
for each tensor
      |
view mapped bytes
      |
scan / calculate parameters
      |
quantize in bounded chunks
      |
write output
      |
next tensor
```

Avoid loading the entire model into RAM.

Eventually, avoid even requiring the largest tensor to be fully allocated by using mapped input and chunked output.

Peak working memory should trend toward:

```text
small metadata
+ input mapping/page cache
+ quantization working chunk
+ scale buffers
+ output chunk
```

rather than:

```text
entire source model
+ entire quantized model
```

### 8.2 SafeTensors output planning

SafeTensors stores a header containing tensor byte offsets before the data region. A streaming writer therefore needs a deliberate layout strategy.

Potential design:

1. Parse all source tensor metadata.
2. Apply the quantization policy to determine which tensors are quantized or preserved.
3. Calculate output tensor shapes and byte sizes before doing the heavy conversion.
4. Allocate output offsets.
5. Write/finalize the output header.
6. Quantize tensors and write their payloads directly to predetermined offsets or through a temporary staging strategy.

Do not solve this with one giant output `Vec<u8>` for real model sizes.

---

## 9. Initial implementation strategy

Do **not** begin by creating a large multi-crate framework.

The framework should emerge from a working vertical slice.

### Initial repository structure

Start with a single Rust package:

```text
ModelQ/
├── Cargo.toml
├── README.md
├── PROJECT.md
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── error.rs
│   ├── tensor.rs
│   ├── diagnostics.rs
│   ├── quant/
│   │   ├── mod.rs
│   │   └── int8.rs
│   ├── io/
│   │   ├── mod.rs
│   │   └── safetensors.rs
│   └── backend/
│       ├── mod.rs
│       └── cpu.rs
├── tests/
│   ├── fixtures/
│   └── int8_e2e.rs
└── benches/                 later
```

The first end-to-end goal is:

```bash
modelq quantize input.safetensors \
  --format int8 \
  --device cpu \
  --output output.safetensors
```

It should:

- parse the source
- enumerate tensors
- quantize supported floating-point weight tensors
- preserve or explicitly skip unsupported tensors according to policy
- write a structurally valid output
- write enough metadata for ModelQ to decode its own result
- compute reconstruction diagnostics
- avoid loading the entire source model into RAM

### When to split into a workspace

Split only when code boundaries have become real and repeatedly useful.

Likely eventual workspace:

```text
ModelQ/
├── Cargo.toml
├── crates/
│   ├── modelq-core/
│   ├── modelq-quant/
│   ├── modelq-io/
│   ├── modelq-backend/
│   ├── modelq-compat/
│   ├── modelq-cli/
│   └── modelq-gui/          later
├── tests/
├── benches/
├── docs/
│   └── adr/
└── PROJECT.md
```

Likely responsibilities:

#### `modelq-core`

Pure concepts and stable data structures:

- `DType`
- `TensorInfo`
- `TensorView`
- `QuantizedTensor`
- `QuantizationRecipe`
- errors shared across crates
- compatibility enums and identifiers once stable

No CUDA, Metal, GUI, or OS-specific logic.

#### `modelq-quant`

Algorithms and representation logic:

- INT8
- INT4
- FP8
- FP4
- NVFP4
- scaling strategies
- packing/unpacking
- dequantization/reference reconstruction

#### `modelq-io`

Checkpoint/container I/O:

- SafeTensors reader/writer
- sharded SafeTensors support
- GGUF reader/writer/exporter later
- output layout planning

#### `modelq-backend`

Execution implementations:

- scalar CPU
- parallel CPU
- x86 SIMD
- ARM SIMD
- CUDA later
- Metal later

#### `modelq-compat`

Runtime and target compatibility:

- target profiles
- supported recipe/container combinations
- runtime exporter capabilities
- compatibility level reporting

Do not add this crate until compatibility logic becomes substantial enough to justify it.

#### `modelq-cli`

End-user command-line application.

#### `modelq-gui`

Future desktop GUI that calls the same public library APIs as the CLI.

### Dependency direction

Avoid cyclic dependencies.

Desired direction:

```text
modelq-cli ----+
modelq-gui ----+----> orchestration/public library APIs
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
          quant          io        compat
             \           |           /
              \          |          /
               +-------> core <-----+

backend implementations depend on core and the minimum quant primitives needed.
```

Exact crate edges should be decided only when the workspace extraction occurs.

---

## 10. Core types and API sketches

These are design sketches, not immutable APIs.

Codex should prefer a small correct API over implementing all of these abstractions immediately.

### 10.1 `TensorView`

A non-owning view over source tensor bytes and metadata.

```rust
pub struct TensorView<'a> {
    pub name: &'a str,
    pub dtype: DType,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}
```

Desired properties:

- no unnecessary copy
- validates byte length against dtype and shape
- little-endian interpretation where required by format
- methods/iterators for conversion from F32/F16/BF16 to working `f32`
- never assumes a particular model architecture

### 10.2 `QuantizationRecipe`

A description of what the user wants.

Conceptual example:

```rust
pub struct QuantizationRecipe {
    pub format: ElementFormat,
    pub scheme: QuantizationScheme,
    pub granularity: Granularity,
    pub scale_dtype: ScaleDType,
    pub rounding: RoundingMode,
}
```

Possible enums:

```rust
pub enum ElementFormat {
    Int8,
    Int4,
    Fp8E4M3,
    Fp8E5M2,
    Fp4E2M1,
    NvFp4,
    ExperimentalInt { bits: u8 },
}

pub enum QuantizationScheme {
    Symmetric,
    Asymmetric,
}

pub enum Granularity {
    PerTensor,
    PerChannel { axis: usize },
    GroupWise { group_size: usize },
    BlockWise { block_size: usize },
}
```

Do not force every format into every scheme. Compatibility validation should reject meaningless combinations.

### 10.3 `QuantizedTensor`

The intermediate representation for quantized data.

```rust
pub struct QuantizedTensor {
    pub name: String,
    pub source_dtype: DType,
    pub shape: Vec<usize>,
    pub recipe: QuantizationRecipe,
    pub payload: Vec<u8>,
    pub scales: ScaleStorage,
    pub zero_points: Option<Vec<u8>>,
    pub layout: QuantizedLayout,
    pub extra: QuantizedMetadata,
}
```

This shape will evolve.

Important idea: the IR should describe the quantized representation independently of SafeTensors, GGUF, CUDA, Metal, or a specific runtime.

For very large tensors, an eventual streaming/chunked representation may replace or complement a single owned `Vec<u8>`.

### 10.4 Quantizer / algorithm abstraction

A simple early trait could be:

```rust
pub trait Quantizer {
    fn quantize(
        &self,
        input: &TensorView<'_>,
        recipe: &QuantizationRecipe,
    ) -> Result<QuantizedTensor>;
}
```

Later we may separate algorithm/scaling strategy from representation:

```rust
pub trait QuantizationAlgorithm {
    fn compute_params(
        &self,
        input: &TensorView<'_>,
        recipe: &QuantizationRecipe,
    ) -> Result<QuantizationParams>;
}
```

Potential implementations:

- `MinMax`
- `MseOptimized`
- future calibration-based strategies

Do not extract this trait until at least two real algorithms require it.

### 10.5 Backend abstraction

Conceptual future interface:

```rust
pub trait Backend {
    fn name(&self) -> &'static str;

    fn supports(&self, recipe: &QuantizationRecipe) -> bool;

    fn quantize_tensor(
        &self,
        input: &TensorView<'_>,
        recipe: &QuantizationRecipe,
    ) -> Result<QuantizedTensor>;
}
```

Potential implementations:

```text
CpuScalarBackend
CpuParallelBackend
CpuAvx2Backend
CpuNeonBackend
CudaBackend
MetalBackend
```

Again, v0.1 can use ordinary Rust functions instead of a backend trait. Extract the trait when there is a second backend.

### 10.6 Reader abstraction

Conceptual:

```rust
pub trait ModelReader {
    fn metadata(&self) -> &ModelMetadata;
    fn tensors(&self) -> Result<impl Iterator<Item = Result<TensorEntry<'_>>>>;
}
```

The first implementation is SafeTensors.

### 10.7 Writer/exporter abstraction

Conceptual:

```rust
pub trait ModelWriter {
    fn supports(&self, recipe: &QuantizationRecipe) -> bool;

    fn plan_tensor(
        &mut self,
        source: &TensorInfo,
        recipe: &QuantizationRecipe,
    ) -> Result<OutputTensorPlan>;

    fn write_tensor(
        &mut self,
        tensor: &QuantizedTensor,
    ) -> Result<()>;

    fn finish(self) -> Result<()>;
}
```

Runtime exporters may be more specific than generic container writers.

### 10.8 Diagnostics

Conceptual:

```rust
pub struct TensorDiagnostics {
    pub elements: u64,
    pub source_bytes: u64,
    pub quantized_bytes: u64,
    pub mse: f64,
    pub mae: f64,
    pub max_abs_error: f64,
    pub sqnr_db: Option<f64>,
    pub saturated_values: u64,
}
```

Model-level summary aggregates tensor diagnostics without requiring inference.

---

## 11. Model policies

Being "model agnostic" does not mean quantizing every tensor identically.

Likely policy categories:

```text
Linear weights          quantize
Embedding weights       configurable
Biases                  preserve initially
Normalization weights   preserve initially
Small tensors           preserve initially
Non-floating tensors    preserve
Unknown/special tensors preserve + report
```

Early versions should use conservative defaults.

The policy engine can initially be simple, for example based on:

- dtype
- tensor rank
- tensor size
- tensor name patterns

Architecture-specific policies should be introduced only when runtime interoperability demands them.

Never silently drop tensors.

Every tensor must end in one of:

- quantized
- preserved
- intentionally skipped with explicit reason
- error

---

## 12. CLI design

### 12.1 Core commands

Initial:

```bash
modelq inspect <MODEL>
```

Shows:

- source format
- tensors
- dtypes
- shapes
- parameter count where calculable
- source size
- shard information later

```bash
modelq quantize <MODEL> \
  --format int8 \
  --output <PATH>
```

Later:

```bash
modelq validate <MODEL>
```

```bash
modelq formats
```

```bash
modelq targets
```

```bash
modelq info
```

`modelq info` can eventually show local capabilities:

```text
OS: macOS
Arch: aarch64
CPU backend: scalar + NEON
GPU backend: Metal
CUDA: unavailable
```

### 12.2 Suggested future quantize flags

```text
--format int8|int4|fp8-e4m3|fp8-e5m2|fp4-e2m1|nvfp4|...
--scheme symmetric|asymmetric
--group-size <N>
--scale-dtype f32|f16|...
--device auto|cpu|cuda|metal
--target <hardware-profile>
--runtime <runtime-profile>
--container safetensors|gguf|...
--report <path>
--threads <N>
--memory-limit <size>       later
--experimental             required for unstable formats
```

Do not expose options before the engine implements them correctly.

### 12.3 Auto mode

Long-term:

```bash
modelq quantize ./model --target my-hardware
```

or:

```bash
modelq quantize ./model \
  --target nvidia-blackwell \
  --runtime tensorrt
```

The planner can then choose among compatible recipes.

This should happen only after compatibility profiles are trustworthy.

---

## 13. GUI direction

Do not build the GUI before the CLI/library stabilizes.

The GUI should call the same public API as the CLI.

Conceptual layout:

```text
+--------------------------------------------------+
| ModelQ                                           |
+--------------------------------------------------+
| Model: /path/to/model                   [Browse] |
|                                                  |
| Source: BF16 | Parameters: ... | Size: ...      |
|                                                  |
| Target hardware: [Auto / Blackwell / ...]       |
| Runtime:         [Auto / llama.cpp / ...]       |
| Format:          [INT8 / INT4 / NVFP4 / ...]    |
| Execution:       [Auto / CPU / CUDA / Metal]    |
|                                                  |
| Estimated output size: ...                       |
| Compatibility level: ...                         |
|                                                  |
|                  [Quantize]                      |
|                                                  |
| Progress: [====================      ]           |
| Tensor: ...                                      |
| MSE: ...                                         |
+--------------------------------------------------+
```

A Rust-native GUI such as `egui/eframe` is a reasonable future candidate, but the choice should be revisited when GUI work begins.

---

## 14. Quantization roadmap

### Stage A: INT8 reference implementation

Purpose: prove the complete data path.

Start with symmetric per-tensor INT8:

```text
max_abs = max(abs(w))
scale = max_abs / 127
q = round(w / scale)
q = clamp(q, -127, 127)
w_hat = q * scale
```

Handle edge cases explicitly:

- all-zero tensor
- NaN/Inf policy
- empty tensor if format permits it
- invalid byte lengths
- scale zero
- very small values
- very large values

Correctness is more important than speed.

### Stage B: INT4

Add:

- symmetric INT4
- group-wise scaling
- configurable group size
- nibble packing
- unpack/dequant reference path
- odd element counts
- scale storage overhead calculation

INT4 is where representation, grouping, packing, and output metadata become significantly more interesting.

### Stage C: richer algorithms

Add when useful:

- per-channel scaling
- asymmetric quantization
- MSE-based scale selection
- policy configuration

### Stage D: FP8 / FP4

Implement reference encoders/decoders with exhaustive tests over representable bit patterns where possible.

Important topics:

- exponent/mantissa representation
- rounding mode
- subnormals where applicable
- zero handling
- infinities/NaNs according to the format
- saturation behavior
- packing

### Stage E: NVFP4

Implement in sub-milestones:

1. E2M1 codec and exhaustive tests.
2. E4M3 codec/reference dependency and exhaustive tests.
3. NVFP4 mathematical scaling/encoding in ModelQ IR.
4. Cross-validation against NVIDIA reference behavior on synthetic tensors.
5. Runtime-specific exporter.
6. Hardware/runtime compatibility test on Blackwell.

### Stage F: experimental low-bit formats

Examples:

- INT3
- INT2
- INT1
- arbitrary signed integer width where meaningful

These should be explicitly marked experimental unless a real runtime/exporter supports them.

The project should allow users to explore numerical compression even when no mainstream runtime can consume the output.

---

## 15. CPU and GPU roadmap

### 15.1 CPU scalar first

Use clear, deterministic scalar Rust as the reference implementation.

This becomes the correctness oracle for optimized backends.

### 15.2 CPU parallelism

Potential strategies:

- process independent chunks/groups in parallel
- process independent tensors in parallel only if memory limits permit it

`rayon` is a reasonable candidate, but do not introduce unbounded concurrent tensor allocations.

Bounded memory matters more than maximizing task count.

### 15.3 SIMD

Likely architecture paths:

```text
x86-64
  +-- AVX2
  +-- AVX-512 later if useful

aarch64
  +-- NEON
```

Runtime feature detection should select optimized paths where possible.

Every SIMD implementation must be checked against the scalar reference.

### 15.4 CUDA

Optional feature for Linux/Windows NVIDIA systems.

Potential work:

- CUDA device discovery
- pinned host buffers
- asynchronous copies
- chunked GPU quantization
- custom kernels
- overlap I/O and compute later

CUDA must not leak into `modelq-core`.

### 15.5 Metal

Optional macOS GPU backend.

Metal support is independent from CUDA support.

The architecture should allow Apple Silicon users to benefit from GPU acceleration without affecting CPU portability.

---

## 16. Cross-platform rules

Target matrix:

| Platform | v0.x CPU | SIMD later | GPU later |
|---|---:|---:|---:|
| Linux x86-64 | yes | AVX2/AVX-512 | CUDA optional |
| Windows x86-64 | yes | AVX2/AVX-512 | CUDA optional |
| macOS Apple Silicon | yes | NEON | Metal optional |
| Linux ARM64 | expected later | NEON | device-dependent |

Initial CI matrix:

```text
ubuntu-latest
windows-latest
macos-latest
```

Run at minimum:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
```

Feature-dependent GPU code may require separate CI and should not block the base CPU build when unavailable.

---

## 17. Dependencies

Keep dependencies small and deliberate.

Likely early candidates:

- `safetensors` for SafeTensors parsing/serialization concepts
- `half` for F16/BF16 conversion
- `memmap2` for large-file mappings
- `clap` for CLI parsing
- `thiserror` for structured errors
- `serde` / `serde_json` when needed for metadata/config
- `tracing` and `tracing-subscriber` for structured logs if useful
- `indicatif` for CLI progress if it remains lightweight
- `tempfile` for integration tests
- `proptest` for property-based numerical/packing tests
- `criterion` for benchmarks later
- `rayon` only when parallel CPU work begins

Avoid adding a full ML framework to the core simply to gain tensor operations.

Potential reference projects/libraries should be used to cross-check behavior, not automatically imported as runtime dependencies.

---

## 18. Testing strategy

Numerical and binary code needs stronger testing than ordinary application code.

### 18.1 Golden vector tests

For every format, define small known tensors with exact expected:

- scale(s)
- quantized integers/bit patterns
- packed bytes
- dequantized values

Examples:

```text
[0.0]
[-1.0, 0.0, 1.0]
constant non-zero tensor
positive-only tensor
negative-only tensor
values exactly on rounding boundaries
values outside representable range
```

### 18.2 Property tests

Useful properties:

```text
unpack(pack(x)) == x
```

for valid quantized values.

```text
encode(decode(bits)) == canonical(bits)
```

for custom FP representations.

```text
quantize(all_zero) produces finite valid scales and zero reconstruction error
```

### 18.3 Exhaustive codec tests

For 4-bit formats there are only 16 bit patterns.

Test every bit pattern.

For FP8 there are only 256 bit patterns, so exhaustive encode/decode table tests are practical.

### 18.4 Round-trip reconstruction tests

For random tensors:

```text
source -> quantize -> dequantize
```

Verify:

- no unexpected NaNs
- error metrics are finite where expected
- shapes preserved
- scale counts correct
- output byte count correct

### 18.5 SafeTensors integration tests

Create tiny synthetic SafeTensors fixtures.

Test:

- header parsing
- multiple tensors
- F32/F16/BF16
- malformed offsets
- truncated files
- empty metadata
- output re-open
- metadata round-trip

### 18.6 Streaming tests

Generate files much larger than the configured working chunk without requiring a real model.

Verify:

- peak application allocations do not scale with total file size
- each chunk/tensor is processed once or according to documented multi-pass behavior
- output matches the non-streaming reference implementation

### 18.7 Cross-validation tests

For each mature format, compare ModelQ outputs against a trusted reference implementation when possible.

Examples:

- TorchAO/reference Python scripts for selected INT schemes
- llama.cpp/GGML reference for GGUF quant types
- NVIDIA reference tools/tests for NVFP4

Do not copy implementation behavior without understanding the exact convention being matched.

### 18.8 Runtime compatibility tests

A runtime exporter is not complete until a fixture generated by ModelQ is successfully loaded by the target runtime.

Where practical, CI can pin a tested runtime version.

### 18.9 Fuzzing

Good fuzz targets:

- SafeTensors metadata parser boundaries
- packing/unpacking
- custom format codecs
- malformed ModelQ metadata

Unsafe mmap/file logic deserves special attention.

---

## 19. Benchmark strategy

Correctness comes first, then measure before optimizing.

Track:

### Quantization throughput

```text
input bytes / second
input elements / second
```

Separate:

- scale calculation
- quantization
- packing
- I/O

### Peak memory

Important because bounded-memory conversion is a core product goal.

Benchmark source sizes larger than RAM where practical using generated fixtures/sparse test files carefully.

### Compression ratio

Report:

```text
source tensor payload bytes
quantized payload bytes
scale bytes
zero-point bytes
metadata/container overhead
final file bytes
```

### Backend comparison

Eventually:

```text
scalar CPU
parallel CPU
AVX2
NEON
CUDA
Metal
```

All optimized results must match reference behavior within the format's defined tolerance/rounding semantics.

### Benchmark reproducibility

Record:

- ModelQ commit
- OS
- CPU/GPU
- thread count
- source dtype
- quantization recipe
- input size
- storage device when I/O is measured

---

## 20. Error handling and safety

ModelQ manipulates huge files and raw byte representations. Fail loudly and safely.

Requirements:

- use structured Rust errors
- validate tensor byte counts
- validate shapes for overflow before multiplying dimensions
- validate source offsets and ranges
- validate group/block sizes
- validate recipe compatibility before writing output
- do not silently truncate values or tensor data
- do not silently overwrite source files
- write outputs atomically where practical
- incomplete output files should be recognizable or removed safely
- isolate `unsafe` code, especially mmap/SIMD/FFI
- every `unsafe` block needs a safety comment explaining its invariant

Do not use `unwrap()`/`expect()` in normal library paths except for conditions proven impossible and documented.

CLI startup/testing code can be more pragmatic but should still return useful errors.

---

## 21. Observability and user reports

Quantizing a large model can take significant time, so users need meaningful progress.

Desired progress information:

```text
Model: ...
Source: BF16
Recipe: INT4 symmetric group=128
Execution: CPU
Output: ...

[ 43% ] 182 / 421 tensors
Current: model.layers.17.mlp.down_proj.weight
Read: 31.4 GB
Written: 8.1 GB

Current tensor MSE: ...
Estimated final size: ...
```

Final report:

```text
Tensors
  quantized: ...
  preserved: ...
  skipped: ...

Size
  source: ...
  output: ...
  compression: ...x

Diagnostics
  mean MSE: ...
  worst MSE tensor: ...
  max abs error: ...
  saturation: ...

Compatibility
  representation-valid: yes
  container-valid: yes
  runtime-compatible: not claimed / tested runtime
  hardware-validated: not claimed / tested hardware
```

---

## 22. Roadmap and releases

Version names are targets, not deadlines.

### v0.1 - working vertical slice

Goal: prove ModelQ can safely transform a real SafeTensors checkpoint without inference.

Required:

- Rust CLI
- Linux, Windows, macOS CPU builds
- inspect SafeTensors metadata
- F32/F16/BF16 input support
- symmetric per-tensor INT8 reference quantization
- INT8 reference dequantization
- tensor-level diagnostics
- ModelQ-native output representation in SafeTensors or a clearly documented early convention
- no full-model in-memory load
- synthetic end-to-end fixtures
- CI on three desktop OSes
- clear documentation that output is ModelQ-native unless a runtime exporter says otherwise

Not required:

- GUI
- CUDA
- Metal
- SIMD
- GGUF
- INT4
- NVFP4
- calibration
- automatic hardware recommendations

### v0.2 - useful low-bit framework

Target features:

- INT4
- group-wise scaling
- packing/unpacking
- model quantization policy
- bounded chunk processing
- sharded SafeTensors input
- richer diagnostics
- output planning improvements
- parallel CPU path after correctness is stable
- initial extraction of reusable framework modules/crates if the code now justifies it

### v0.3+ - interoperability and performance

Likely sequence:

1. GGUF format investigation and exporter for one exact supported quant type.
2. Runtime compatibility harness.
3. CPU SIMD.
4. FP8/FP4 codecs.
5. NVFP4 mathematical representation.
6. NVIDIA runtime exporter and Blackwell validation.
7. CUDA backend.
8. Metal backend.
9. target/runtime planner.
10. GUI.
11. experimental INT3/INT2/INT1 and plugin/extension design if justified.

### v1.0 - stable tool and library

A realistic v1.0 does **not** mean every model and every format.

A good v1.0 means:

- stable public Rust API for core quantization workflows
- robust SafeTensors support including common sharded checkpoints
- multiple useful data-free quantization recipes
- at least one genuinely runtime-compatible export path tested end-to-end
- bounded-memory conversion
- cross-platform scalar CPU support
- at least one optimized CPU path
- mature diagnostics
- versioned ModelQ-native metadata convention
- compatibility matrix with honest support levels
- reproducible benchmark suite
- strong automated tests
- documented extension path for formats/backends/exporters

---

## 23. Major risks

### Risk 1: "any model to any format" becomes impossible scope

Mitigation:

- model-agnostic tensor transformation where possible
- explicit policies for special tensors
- runtime/architecture-specific exporters where required
- capability matrix instead of universal claims

### Risk 2: mathematically correct output is not runtime-compatible

Mitigation:

- compatibility levels
- runtime-specific exporters
- integration tests against named runtimes
- never infer compatibility from bit width alone

### Risk 3: SafeTensors low-bit conventions are not universally understood

Mitigation:

- treat early SafeTensors low-bit output as ModelQ-native
- version metadata convention
- add GGUF/runtime exporters separately

### Risk 4: memory usage defeats the original user story

Mitigation:

- mmap/read by range
- plan output offsets
- bounded chunk size
- no entire-model buffers
- memory benchmarks and tests

### Risk 5: premature abstractions slow the project

Mitigation:

- build INT8 vertically first
- extract traits only after a second implementation demonstrates the need
- do not create plugin systems/config languages before real use cases

### Risk 6: optimized backend disagrees with reference backend

Mitigation:

- scalar implementation is the oracle
- golden vectors
- property tests
- cross-backend differential tests

### Risk 7: NVFP4 semantics change or differ by workflow/runtime

Mitigation:

- use primary NVIDIA docs at implementation time
- separate representation from runtime layout
- pin tested runtime/tool versions in compatibility tests

### Risk 8: GPU support destroys portability

Mitigation:

- optional Cargo features
- CPU always available
- platform backends isolated
- base CI never requires a GPU SDK

---

## 24. Non-goals for early development

Do not let Codex expand the project into these areas unless explicitly requested:

- training framework
- inference engine
- tokenizer implementation
- model downloader/hub client
- model serving
- chat UI
- Python bindings
- distributed quantization
- automatic calibration datasets
- every GGUF quant type
- every NVIDIA format
- every model architecture
- arbitrary `.pt` pickle execution
- a generic tensor framework competing with PyTorch/Candle

Some may become future integrations, but they are not the initial problem.

---

## 25. Coding conventions

Initial conventions:

- stable Rust unless a specific feature justifies nightly
- `cargo fmt`
- `cargo clippy` with warnings denied in CI
- explicit error types
- minimal `unsafe`
- no hidden global mutable state
- deterministic reference quantization
- public APIs documented with rustdoc once exposed
- unit tests live near pure algorithms
- integration tests cover checkpoint I/O and CLI behavior
- keep modules small enough to understand
- prefer clear numeric code before clever optimization
- optimization PRs should include benchmark evidence

Naming guidance:

- use `quantize` for float/reference -> quantized representation
- use `dequantize` for reconstruction
- use `pack` / `unpack` only for bit-level layout transformation
- use `encode` / `decode` for custom floating-point bit patterns
- use `export` when adapting ModelQ IR to a runtime-specific representation
- use `write` for container serialization

---

## 26. Documentation and ADRs

As architecture becomes concrete, add Architecture Decision Records under:

```text
docs/adr/
```

Recommended ADRs:

1. `0001-project-scope-and-non-goals.md`
2. `0002-modelq-native-quantized-tensor-convention.md`
3. `0003-streaming-safetensors-output-layout.md`
4. `0004-quantized-tensor-ir.md`
5. `0005-runtime-compatibility-levels.md`
6. `0006-workspace-crate-split.md` when it actually happens
7. `0007-cpu-simd-dispatch.md`
8. `0008-gpu-backend-boundary.md`
9. `0009-nvfp4-runtime-target.md`

An ADR should record:

- context
- decision
- alternatives considered
- consequences
- compatibility impact

---

## 27. First Codex-sized implementation tasks

Codex should work in small testable steps. Prefer one focused PR/commit series per task.

### Task 1 - Bootstrap the Rust project

Create:

- `Cargo.toml`
- `src/main.rs`
- `src/lib.rs`
- basic module layout
- `.gitignore`
- minimal `README.md`

Acceptance criteria:

```bash
cargo build
cargo test
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

all pass locally.

Do not add quantization functionality yet.

### Task 2 - Add cross-platform CI

Add GitHub Actions matrix for:

- Ubuntu
- Windows
- macOS

Acceptance criteria:

- fmt check
- clippy
- tests
- no CUDA/Metal requirement

### Task 3 - Define source dtype and tensor metadata

Implement minimal:

```text
DType
TensorInfo
TensorView
```

Support source dtypes:

- F32
- F16
- BF16

Acceptance criteria:

- validates byte length vs shape
- safe iteration/conversion to f32 reference values
- tests for each dtype
- shape multiplication checks overflow

### Task 4 - SafeTensors inspection

Implement:

```bash
modelq inspect fixture.safetensors
```

Acceptance criteria:

- prints tensor name, dtype, shape, byte size
- does not deserialize tensors into a full model
- malformed file returns useful error
- tests use synthetic fixture

### Task 5 - Memory-mapped SafeTensors access

Add a reader that can expose each tensor as a mapped/ranged `TensorView`.

Acceptance criteria:

- source file remains owned for lifetime safety
- no whole-file copy into `Vec<u8>`
- tests compare mapped tensor values to known fixture values
- `unsafe` mmap usage, if any, is isolated and documented

### Task 6 - INT8 scalar quantize/dequantize

Implement symmetric per-tensor INT8 reference algorithm.

Acceptance criteria:

- zero tensor works
- positive/negative values work
- deterministic rounding policy documented
- quantized values stay in defined range
- dequant reconstructs correctly
- golden tests included

### Task 7 - Diagnostics

Implement:

- MSE
- MAE
- max absolute error
- saturation count
- compression byte accounting

Acceptance criteria:

- tests against hand-calculated small arrays
- no full duplicate reconstructed tensor required if metrics can be streamed

### Task 8 - Quantization policy v0

Add conservative policy for deciding whether a tensor is quantized or preserved.

Initial suggestion:

- floating tensor above configurable minimum element count: quantize
- non-floating tensor: preserve
- small floating tensors: preserve

Acceptance criteria:

- every tensor gets a recorded action and reason
- policy unit tests

### Task 9 - Define ModelQ INT8 output convention

Before coding the writer, write an ADR specifying:

- qdata tensor naming
- scale tensor naming
- metadata keys
- version marker
- preservation rules
- how original shape/dtype are represented

Acceptance criteria:

- format can be decoded without model-specific knowledge
- convention clearly marked ModelQ-native, not runtime-compatible

### Task 10 - Output layout planner

Given source metadata and quantization policy, calculate all output tensors and byte sizes before writing.

Acceptance criteria:

- deterministic offsets
- checked arithmetic
- supports quantized payload + scale tensors + preserved tensors
- unit tests for multi-tensor layouts

### Task 11 - Streaming SafeTensors writer

Write the planned output without building the whole output model in memory.

Acceptance criteria:

- output can be reopened by SafeTensors parser
- data offsets are correct
- output bytes match plan
- failure does not overwrite input
- integration tests use temp files

### Task 12 - End-to-end INT8 command

Implement:

```bash
modelq quantize input.safetensors \
  --format int8 \
  --device cpu \
  --output output.safetensors
```

Acceptance criteria:

- real end-to-end fixture passes
- prints progress and final report
- ModelQ can read/dequantize its own result for validation
- output is smaller for suitable floating-weight fixtures

This completes the first meaningful vertical slice.

### Task 13 - Bounded chunk processing

Refactor the scalar path so huge tensors do not require source or reconstructed `Vec<f32>` allocations.

Acceptance criteria:

- processes a generated tensor larger than the working chunk size
- output matches reference small/non-chunked implementation
- diagnostic calculation remains correct

### Task 14 - Sharded SafeTensors investigation and design

Research common SafeTensors shard/index conventions and write an ADR/design before implementation.

Acceptance criteria:

- defines reader behavior for model directories and index files
- defines deterministic tensor iteration order
- defines output sharding policy or explicitly defers output sharding

### Task 15 - INT4 reference implementation

Add symmetric group-wise INT4 without optimization.

Acceptance criteria:

- group size validation
- correct per-group scales
- correct signed 4-bit representation
- pack/unpack golden tests
- odd lengths handled
- quantize/dequant diagnostics

### Task 16 - Extract framework boundaries

Only now review the code for a workspace split.

Acceptance criteria:

- write ADR explaining why split is now useful
- preserve public behavior
- no dependency cycles
- full workspace tests on all OSes

Possible first split:

```text
modelq-core
modelq-io
modelq-quant
modelq-cli
```

Do not create `modelq-backend` until there is genuinely more than one execution implementation unless separation is already clearly useful.

### Task 17 - CPU parallel INT4/INT8

Add bounded parallelism.

Acceptance criteria:

- results match scalar reference exactly or according to documented numeric tolerance
- memory use stays bounded
- benchmark demonstrates whether it helps

### Task 18 - GGUF compatibility spike

Do not implement "all GGUF".

Choose one exact GGUF quantization type/runtime path.

Acceptance criteria:

- document exact GGUF type and llama.cpp/runtime version
- map ModelQ IR to the exact required block layout
- generated tiny fixture loads in target runtime
- mark compatibility level accordingly

### Task 19 - FP4/FP8 codec module

Implement reference bit codecs.

Acceptance criteria:

- exhaustive bit-pattern tests
- documented rounding/saturation semantics
- independent encode/decode tests

### Task 20 - NVFP4 research spike

Before full implementation, produce an ADR based on current NVIDIA primary docs.

Acceptance criteria:

- exact element encoding
- scale hierarchy
- block/group layout
- weight-only offline conversion algorithm
- distinction between ModelQ-native NVFP4 representation and specific runtime-compatible layout
- test strategy against NVIDIA reference tools
- identified Blackwell/runtime validation path

Only then implement NVFP4 in multiple incremental tasks.

---

## 28. Codex operating instructions

When Codex works on this repository, follow these rules unless the project owner explicitly changes them.

### Work style

1. Read `PROJECT.md` before planning a feature.
2. Inspect the current repository before proposing architecture changes.
3. Work in vertical, testable increments.
4. Do not implement unrelated roadmap features in the same task.
5. Preserve cross-platform CPU builds.
6. Add or update tests with every numerical or binary-layout change.
7. Prefer primary specifications/reference implementations for format compatibility.
8. State when an output is only ModelQ-native rather than runtime-compatible.
9. Do not introduce a general abstraction until a concrete second use case justifies it.
10. Keep the scalar reference implementation available even after optimization.

### Before implementing a quantization format

Codex should answer:

- What exact numerical representation is being implemented?
- What scaling/granularity is used?
- What rounding behavior is required?
- What packing layout is required?
- Is this ModelQ-native or intended for a named runtime?
- What reference implementation/specification will tests compare against?
- Can it be implemented data-free?
- What tensors should be preserved?

### Before claiming runtime support

Codex must identify:

- runtime name
- tested runtime version or commit
- container format
- exact quantization layout/type
- model architecture constraints
- required metadata/tensor naming
- target hardware constraints
- an integration test or documented manual validation procedure

### Performance rules

- correctness first
- benchmark before and after optimization
- never remove scalar/reference tests
- avoid increasing peak memory for small throughput gains without discussion
- keep I/O, conversion, and packing costs measurable separately

### Dependency rules

Before adding a dependency, explain:

- why standard library/current dependencies are insufficient
- whether it affects Windows/Linux/macOS
- whether it introduces a native/system dependency
- whether it affects optional GPU-free builds

---

## 29. Definition of project success

The project succeeds if ModelQ eventually makes this workflow trustworthy:

```text
User has a checkpoint too large to run
                |
                v
         ModelQ inspects it
                |
                v
      User chooses a target
                |
                v
 ModelQ plans a valid quantization
                |
                v
CPU/GPU transforms weights with bounded memory
                |
                v
ModelQ validates numerical reconstruction
                |
                v
Exporter produces an explicitly supported artifact
                |
                v
User loads it in the declared runtime/hardware
```

The long-term challenge is not simply making numbers use fewer bits.

The challenge is building a clean, extensible bridge between:

```text
model checkpoints
+ numerical quantization
+ binary packing
+ bounded-memory systems programming
+ hardware capabilities
+ runtime-specific layouts
+ honest interoperability
```

That is the core identity of ModelQ.

---

## 30. Immediate next action

Start with **Task 1**, then proceed in order through the first vertical slice.

Do not begin with NVFP4, CUDA, a GUI, or a large plugin framework.

The first milestone should remain concrete:

```text
SafeTensors F32/F16/BF16
          |
          v
Rust scalar CPU INT8
          |
          v
ModelQ-native SafeTensors output
          |
          v
reopen + dequantize + diagnostics
```

Once this works end-to-end, use what the implementation taught us to extract the reusable ModelQ framework.

---

## 31. Reference ecosystems to consult

When implementing or validating features, prefer current primary sources and upstream code. Important ecosystems include:

- SafeTensors specification and Rust implementation
- GGML / GGUF specification
- llama.cpp quantization implementations and loaders
- PyTorch TorchAO quantization APIs and reference behavior
- NVIDIA Transformer Engine
- NVIDIA Model Optimizer
- NVIDIA CUTLASS
- NVIDIA TensorRT for runtime-specific NVFP4 compatibility
- Rust `half`, `memmap2`, and related low-level crates

Specifications and runtime conventions change. Re-check current upstream documentation when implementing later roadmap items rather than treating this planning document as the final specification.
