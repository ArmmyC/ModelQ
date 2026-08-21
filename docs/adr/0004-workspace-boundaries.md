# ADR 0004: Initial Workspace Boundaries

- Status: Accepted
- Date: 2026-08-21
- Scope: reusable Rust crate boundaries after the v0.2 reference slice

## Context

ModelQ began as one package so the first vertical slice could be built and
verified quickly. That slice now contains four concrete responsibilities with
different dependency needs:

- tensor metadata, validated views, and shared errors;
- INT8/INT4 representations, policies, packing, and diagnostics;
- SafeTensors parsing, layout planning, and output writing; and
- the end-user command-line orchestration.

The boundaries are no longer hypothetical. The tensor code is useful without
checkpoint I/O, the quantizers are useful without a filesystem, and the I/O
layer already consumes quantization policy and reference algorithms through
small, explicit interfaces. Keeping all of them in one crate would make the
next work—sharded input, parallel CPU execution, and additional formats—share
implementation dependencies and make accidental reverse dependencies easy.

Task 16 is therefore the point at which a small workspace is useful. It is not
the point at which every long-term crate should exist: there is still only one
backend implementation, no compatibility matrix, and no GUI.

## Decision

Add a Cargo workspace with three reusable implementation crates while retaining
the root `modelq` package as a compatibility facade and CLI.

### Workspace members and responsibilities

```text
ModelQ/
├── Cargo.toml                 # workspace root and compatibility package
├── crates/
│   ├── modelq-core/
│   │   └── src/               # error.rs, tensor.rs
│   ├── modelq-quant/
│   │   └── src/               # int8.rs, int4.rs, policy.rs, diagnostics.rs
│   └── modelq-io/
│       └── src/               # safetensors.rs, layout.rs, writer.rs
└── src/                       # compatibility facade, backend stub, CLI
```

`modelq-core` owns pure concepts and validated tensor data:

- `DType`, `TensorInfo`, and `TensorView`;
- `ModelQError` and the shared `Result` alias; and
- the `half` dependency required to decode F16/BF16 values.

`modelq-quant` owns representation and algorithm logic:

- scalar symmetric INT8;
- scalar group-wise INT4;
- the conservative quantization policy; and
- reconstruction/compression diagnostics.

It depends on `modelq-core` only for tensor metadata used by the policy.

`modelq-io` owns checkpoint/container logic:

- SafeTensors inspection and mapped reads;
- deterministic output layout planning; and
- the streaming ModelQ-native INT8 writer.

It depends on `modelq-core` for tensor views and on `modelq-quant` for policy
decisions and INT8 streaming primitives. It does not depend on the root
`modelq` package.

The root `modelq` package remains both a library and the `modelq` binary. Its
library is intentionally thin:

```rust
pub use modelq_core::{error, tensor};
pub use modelq_io as io;
pub use modelq_quant as quant;
pub use modelq_quant::diagnostics;
```

This preserves the existing public paths (`modelq::tensor`,
`modelq::quant`, `modelq::diagnostics`, and `modelq::io`) and keeps the
existing `cargo run`/`modelq quantize` behavior. The root package continues to
own the small CLI orchestration and the current backend placeholder.

### Dependency direction

The allowed dependency graph is:

```text
modelq (facade + binary)
├── modelq-core
├── modelq-quant ───> modelq-core
└── modelq-io ──────> modelq-core
                 └──> modelq-quant
```

No reusable crate may depend on the root `modelq` package. `modelq-core` has no
dependency on quantization or I/O, `modelq-quant` has no dependency on I/O, and
`modelq-io` does not call CLI code. These rules make cycles structurally
impossible for the current members.

### Deliberately deferred crates

No `modelq-backend` crate is created yet. `src/backend/cpu.rs` is still a
documented placeholder and there is no second execution implementation that
would benefit from a shared backend interface.

No separate `modelq-cli` package is created yet. The existing root package
already provides the public `modelq` binary and library together, and the CLI
has not accumulated a reusable orchestration API. Extracting it now would
either duplicate the binary or change the established package workflow. A
future CLI extraction can depend on the three reusable crates once command
orchestration becomes independently valuable.

`modelq-compat` and `modelq-gui` remain future crates for the same reason:
their concepts are not substantial enough to justify new dependency edges.

## Public behavior compatibility

This extraction changes crate ownership, not the current feature contract:

- the root package remains named `modelq` and still builds the `modelq` binary;
- `cargo build`, `cargo test`, and `cargo run` from the repository root remain
  valid;
- existing `modelq::...` library paths are reexports of the same public items;
- the INT8 CLI syntax, output convention, and validation behavior are
  unchanged; and
- INT4 remains a library reference implementation and is not silently wired
  into the CLI or writer.

The new crate names are additive implementation boundaries. They are marked
`publish = false` until the APIs have a deliberate release/versioning policy.

## Testing and verification

The workspace must pass the same quality gates from the repository root:

```bash
cargo build --workspace
cargo test --workspace --all-targets
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The existing root integration tests remain attached to the compatibility
package, while unit tests move with their implementation modules. This keeps
the end-to-end command and writer tests exercising the public facade rather
than only testing private crate internals. CI continues to run the complete
workspace on Ubuntu, Windows, and macOS.

## Follow-up

The backend deferral above describes the state at the Task 16 decision point.
Task 17 introduced the second execution implementation and extracted the
portable `modelq-backend` crate. Its current CPU boundary and bounded worker
policy are documented in [ADR 0007](0007-cpu-parallel-dispatch.md).

## Consequences

### Benefits

- Future quantizers can compile and test without I/O or CLI dependencies.
- I/O changes cannot reach into CLI code and can reuse quantization primitives
  through an explicit edge.
- Core tensor types now have a small, reusable home for later backends or
  compatibility logic.
- Workspace-level Clippy and tests enforce the dependency graph on all target
  platforms.

### Costs and limitations

- The root facade adds a small reexport layer and keeps three path dependencies
  in the compatibility package.
- The current CLI is still coupled to the root package until its orchestration
  API is stable enough for a separate crate.
- Some public types are not yet versioned as stable external APIs; workspace
  membership alone does not promise semver compatibility.

## Alternatives considered

### Keep one package until all formats exist

Rejected. The core, quant, and I/O boundaries already have distinct dependency
sets and tests. Waiting would allow the upcoming parallel backend and format
work to deepen accidental coupling.

### Split into four packages immediately, including `modelq-cli`

Rejected for this step. Moving the current binary while preserving the root
`modelq` package would require duplicate binary ownership or a breaking change
to the established `cargo run` and integration-test workflow. The root facade
keeps behavior stable while leaving a clear future extraction seam.

### Add every long-term crate now

Rejected. Empty backend, compatibility, and GUI crates would create names and
dependency edges without real responsibilities. They can be added when their
interfaces are justified by concrete implementations.
