# ADR 0028: Scope as an Open-Source Model Quantizer for Everyone

- Status: Accepted (owner decisions recorded below; `PROJECT.md` sections 2.1, 7, 22 and 24 updated to match)
- Date: 2026-10-08
- Scope: the product goal, the rules for claims, and the milestones after the current NVFP4 work. It revises `PROJECT.md` sections 24 (non-goals), 7 (data-free only) and 22 (roadmap). `PROJECT.md` is not edited until this ADR is accepted.
- Builds on: [ADR 0024](0024-nvfp4-output-quality-qwen2-5-0-5b.md), [ADR 0025](0025-nvfp4-block-scale-search.md), [ADR 0026](0026-block-scale-search-across-models.md), [ADR 0027](0027-scale-search-default.md)

## Context

ModelQ started as a quantization compiler for checkpoints too large to load for inference (`PROJECT.md` section 1). Its work so far has been a strong vertical slice: streaming SafeTensors input and output, INT8 and NVFP4, a Transformer Engine export with a hardware proof, a block-scale search, and quality measurements.

The owner's goal is broader: an open-source model quantizer that **anyone can run on their own machine** to quantize **any model** into **any format**, without waiting for an official release. Users should be able to experiment with unusual formats such as INT1, not only mainstream ones.

Three existing decisions conflict with that goal and must be revised here:

1. `PROJECT.md` section 24 excludes a model downloader or Hub client, Python bindings, and every model architecture.
2. Section 7 keeps calibration (GPTQ- or AWQ-style) out of v0.x.
3. Section 22 places GGUF in v0.3 and the GUI last. For the user group described above, GGUF and a GUI are the two things that make the tool usable without a terminal.

The brief's own risk 1 ("any model to any format becomes impossible scope") and its compatibility levels (section 4) still hold. They govern how "any" is promised, not whether the goal is valid.

## Decision

### 1. The promise, stated precisely

- **Any model:** any checkpoint whose tensors can be read (SafeTensors first; GGUF or PyTorch later). The data-free path never depends on the architecture. Architecture-specific handling (tied embeddings, fused attention projections, mixture-of-experts tensors) comes as explicit policy profiles. A tensor the tool does not understand is **preserved and reported**, never silently dropped or guessed at (section 11).
- **Any machine:** CPU is always the fallback and must run on Windows, macOS and Linux with bounded memory (sections 2.3, 2.4, 8.1). GPU acceleration is optional and never required to produce output.
- **Any format:** any representation ModelQ can **encode, decode and measure**. A format is not a promise that a runtime will load it. Every format declares its status, and the tool reports the status it reached.

### 2. Format status is part of the product

Each format is registered with one of these statuses, shown by `modelq formats` and in every report:

| Status | Meaning | Example |
|---|---|---|
| `experimental` | Encodes and decodes, round-trip and diagnostics tested; no runtime claimed | INT1, INT2, INT3 |
| `representation-valid` | Matches a written specification, with reference tests | INT8, INT4, FP8, FP4 |
| `container-valid` | Parses with an independent reader | ModelQ SafeTensors convention |
| `runtime-compatible: <runtime> <version>` | Loads in a named runtime version | Transformer Engine 2.19.0 NVFP4 |
| `hardware-validated: <hardware>` | Verified in the named runtime on the named hardware | NVFP4 on B200 |

The tool **never** labels output with a higher status than it has reached (`PROJECT.md` section 4). An `experimental` format is allowed, but the CLI prints a warning and requires `--experimental`, as in the brief's flag list.

### 3. Model input from the Hub

ModelQ may download a model from the Hugging Face Hub **on the user's machine**, as a convenience layer over the existing local path:

- `modelq quantize <hub-id>` downloads into a local cache, verifies SHA-256 against the Hub's metadata, and then runs the same pipeline as a local file.
- Before conversion it prints the model's license from its card. ModelQ does not redistribute models, and it does not hide the license. Gated or non-commercial models are allowed with the license shown; the tool makes no claim about whether a particular use is permitted.
- Downloads are never triggered by output observed in a file or a page; the user names the model.

This reverses the Hub-client non-goal in section 24, and it is the only network feature in scope.

### 4. Calibration is allowed, behind a flag

- The data-free path stays the default and the reference for every calibration method.
- One calibration method (GPTQ or AWQ, chosen by measurement, see milestone M4) is added after the data-free path is measured end-to-end. Calibration data comes from a named source (a Hub dataset or a local file), and the report records it.
- The architecture for calibration (`QuantizationRecipe` split into data-free and calibration recipes, section 7) is introduced only with that second method, as section 7 already requires.

### 5. Output for normal users: GGUF before GUI

- **GGUF exporter** is promoted from v0.3 to the next runtime milestone after Hub input. It targets the quant types that the llama.cpp ecosystem loads, starting with one exact type (the brief's Task 18 rule), and is labeled with the llama.cpp version it was tested against. Before writing it, the existing `crates/.../gguf_q8_0.rs` test work must be reviewed; it has not been assessed in this ADR.
- **Native SafeTensors output** remains the ModelQ-native convention for experimental formats, which no mainstream runtime reads.

### 6. GUI after the library surface is stable

A GUI is required for the "normal user" goal, but the brief's rule holds: the GUI calls the same public library API as the CLI, and it starts only after that API is stable (section 13). Candidate: `egui`/`eframe`, to be confirmed when the work starts.

### 7. Distribution

Users run ModelQ on their own machines, so prebuilt binaries must be shipped for Windows, macOS and Linux. Unsigned Windows executables are blocked by Smart App Control and SmartScreen (as seen during development), so signed release binaries are a requirement for this goal. Code signing has a cost and a process, and is out of scope for this ADR's implementation work; it is listed as a release milestone.

### 8. Evaluation runs locally

The Modal harness used so far is the maintainers' verification path. The user-facing path must do the same measurement with no cloud service: `modelq eval <model> <output>` computes perplexity, KL divergence and top-1 agreement on a small built-in text on the user's CPU or GPU, and prints the cost in time and memory. Slow on CPU is acceptable; it must not be impossible.

## Milestones

Each milestone is a set of tasks in the existing style: one focused PR per task, tests with every numerical change, and honest status labels.

- **M1: Local evaluation.** `modelq eval` runs the WikiText-2 measurement on the user's machine, reproducing the numbers in ADR 0024 within a stated tolerance. Acceptance: the same result from the Linux verifier and from a Windows machine.

  **M1 tolerance (accepted by the owner):** perplexity, original and quantized, within 0.01% relative; perplexity increase, top-1 agreement and mean weight error within 0.01 percentage points; KL divergence within 0.1% relative. The comparison target is the radius-6 row of [ADR 0026](0026-block-scale-search-across-models.md) (the same container, measured on Modal's L4 GPU), because the default-rule row in ADR 0024 is not the container this evaluation uses.

  **M1 result, Windows CPU run** (Python 3.14, torch 2.12.1+cpu, transformers 5.12.1, float32, all 146 windows, about 65 minutes): original perplexity 13.06925 (-0.005%), quantized 14.12435 (-0.005%), increase +8.073% (+0.003 points), KL 0.080266 (-0.04%), top-1 85.343% (+0.003 points), mean weight error 8.151% (+0.001 points). Every metric is inside the tolerance. Evidence: [`m1-local-eval-qwen2.5-0.5b-r6-windows-cpu.json`](../validation/m1-local-eval-qwen2.5-0.5b-r6-windows-cpu.json). The Rust `modelq eval` wrapper is covered by tests on Linux; its Windows executable could not be run on the development PC because Windows Application Control blocked newly built executables.
- **M2: Hub input.** `modelq quantize <hub-id>` with cache, checksum, and license display. Acceptance: a fresh Windows machine quantizes and evaluates a 0.5B model end-to-end from one command.
- **M3: Format registry and low-bit formats.** A registry that holds the statuses in section 2. Add INT4 with group-wise scales (brief Stage B), then experimental INT3, INT2 and INT1 with the `--experimental` flag. Acceptance: each format passes exhaustive or property tests and `modelq formats` shows its status.
- **M4: Calibration.** One calibration method, chosen by measurement, with recorded data sources. Acceptance: it beats the data-free method on the same evaluation, or the ADR records that it does not.
- **M5: GGUF export.** One exact quant type, loaded by a pinned llama.cpp version. Acceptance: a generated fixture loads in that version and a model quantized this way loads in a runtime and runs.
- **M6: GUI and distribution.** A GUI over the public API, and signed binaries for three OSes.

NVFP4 hardware work continues as a separate track and is not blocked by this ADR.

## Alternatives considered

- **Keep the brief's scope and wait for the v1.0 criteria.** Rejected: the owner's goal is for normal users now, and the brief's non-goals would keep the tool a maintainers' tool.
- **Promise "any model, any format, runtime-compatible."** Rejected: this breaks the compatibility rules and would produce claims the project cannot back.
- **Build a Python package first.** Deferred: Python bindings are useful later, but the CLI and the Hub path serve normal users first, and bindings would freeze an API too early.
- **Ship only through a hosted web service.** Rejected: it contradicts the goal of running on the user's own machine and the brief's CPU-first rule.

## Consequences

- `PROJECT.md` section 24 loses the Hub client and Python bindings from its non-goals, and it gains a dated milestone for each. Section 7 gains a calibration milestone. Section 22 moves GGUF earlier.
- Every format and every model class needs a documented status. This creates more documentation work and makes the tool more honest.
- Normal users will be able to produce low-quality output (INT1, for example). The CLI and the report must make the quality cost visible, which `modelq eval` does.
- Releasing binaries introduces signing and distribution work that development has not needed so far.
- Open source requires a license for the project itself. The project license is Apache-2.0, as decided by the owner.

## Compatibility impact

None on existing outputs. The default output stays as in ADR 0027. New formats and exporters are additive and carry their own status labels.

## Open questions

1. ~~Project license~~ Decided: Apache-2.0.
2. Calibration methods: both GPTQ and AWQ are to be supported over time. M4 implements the first one; the calibration interface must accept the second without redesign. Which one comes first is still open.
3. GGUF quant type for M5: open. Candidates are Q8_0 (already spiked on main) for the first exporter, then a 4-bit type for normal users.
4. Is signed Windows distribution in scope now, or after M5?

## Calibration needs forward passes

Calibration methods run the model's layers on sample text to measure activations. `PROJECT.md` section 2.1 says the core does not run inference, and section 24 excludes a tokenizer. Calibration therefore requires a narrow exception: layer-by-layer forward passes and tokenization of calibration text only. No generation, sampling, or attention serving. This exception must be written into `PROJECT.md` section 2.1 when this ADR is accepted.
