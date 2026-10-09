# ADR 0029: Hugging Face Hub Input

- Status: Accepted for Linux and the real Hub (Task 48, milestone M2 of ADR 0028); the Windows acceptance criterion is open, see Not verified
- Date: 2026-10-10
- Scope: `modelq quantize hf:<owner>/<name>` fetches a model's SafeTensors weights into a local cache, verifies them, and quantizes them. It does not fetch tokenizers or configs, does not run the model, and does not change any output format.
- Builds on: [ADR 0028](0028-open-source-quantizer-scope.md) (section 3, Hub input), [ADR 0003](0003-sharded-safetensors-input-design.md) (input discovery)

## Context

ADR 0028 requires that a normal user can quantize a Hugging Face model with one command on their own machine. The quantizer already accepts local files, single or sharded, so the Hub support is a fetch step in front of it. The fetch must satisfy four rules from the ADR: nothing is downloaded unless the user names the model, the license is shown before any weight is fetched, every downloaded file is checked against the Hub's published checksum, and nothing is sent to anyone but the Hub.

## Decisions

### 1. Hub models are named with an `hf:` prefix

`modelq quantize hf:Qwen/Qwen2.5-0.5B` fetches the model. A bare `Qwen/Qwen2.5-0.5B` is a local path and is never looked up on the Hub. ADR 0028 wrote `modelq quantize <hub-id>`; the prefix replaces that form because a mistyped local path such as `models/qwen` would otherwise start a download of a different repository. A missing local path is still reported as a missing local path (tested).

### 2. HTTP and TLS through the operating system

The Hub client is `ureq` 3.4.2 with the `native-tls` feature, which uses Schannel on Windows, Security.framework on macOS and OpenSSL on Linux. Three reasons:

- The `rustls` default uses `ring`, whose build script is a native executable. On the development PC, Windows Application Control blocked that build script, so `cargo check` failed. A source build by anyone on a machine with the same policy would fail the same way.
- Native TLS trusts the operating system's certificate store, which is what corporate networks and proxies rely on.
- Linux builds need the OpenSSL development package (`libssl-dev`). Stated as a requirement; it is a system library, not a bundled one.

Other new dependencies: `sha2` 0.11.0 and `sha1` 0.11.0 (RustCrypto, pure Rust, MIT or Apache-2.0). `serde_json` moves from development-only to a normal dependency; it was already in the lock file. Gzip responses are accepted through `ureq`'s `gzip` feature.

### 3. Cache layout

Files go to `<cache>/hub/<owner>/<name>/<commit>/`, keyed by the commit the Hub resolved to, never by a branch name. The cache root is `--cache-dir`, then `MODELQ_CACHE`, then `%LOCALAPPDATA%\modelq\cache` on Windows or `$XDG_CACHE_HOME/modelq` or `~/.cache/modelq` elsewhere.

The Hugging Face cache (`~/.cache/huggingface`) is not read or written. Sharing it would save disk space for users who already use the Python tools, but its layout is not a stable interface, so reuse is deferred.

### 4. Verification

- Files stored with Git LFS are checked with the SHA-256 the Hub publishes for them.
- Other files (the index of a sharded model) are checked with the git blob SHA-1 the Hub publishes (`sha1("blob <len>\0" + content)`).
- A file with no published checksum is refused.
- Every file is checked before use: a download when it arrives, and a cached file on every later run. A mismatch deletes the partial file and fails the command.
- Every request uses the commit the Hub resolved, so the files of one run always come from the same revision.

### 5. Resume and retries

Downloads go to `<file>.part`. An interrupted download resumes with an HTTP `Range` request from the bytes already on disk, and a server that ignores the range restarts the file. Three attempts are made for network and 5xx failures. Access and not-found errors are not retried.

This matters because a plain download stalled at 179 MB during development and had to be resumed by hand (see the M1 notes).

### 6. Access tokens

For gated or private repositories the user sets `HF_TOKEN`. The token is sent as a bearer header to the Hub endpoint only, and it is never printed. ModelQ does not accept license terms for the user: a gated model needs the user to accept them on its page first, and the error says so.

### 7. License display

The license from the model card (or its `license:` tag) is printed before any weight is fetched, with a link to the model page. A model without a stated license is reported as such. The tool does not decide whether a use is permitted.

### 8. Scope of the fetch

Only the files quantization needs are fetched: `model.safetensors`, or `model.safetensors.index.json` plus the shards it names. Shard names are checked as safe relative paths before use, so a malicious index cannot write outside the cache.

## Consequences

- Users can run `modelq quantize hf:<model> --format nvfp4-te --output <file>` on their own machine with one command (Linux and macOS as the first targets; Windows once a signed build exists, see below).
- The `eval` command still downloads through the Python tool (`--download`, pinned revisions, ADR 0028 M1). Making `eval` accept `hf:` is a follow-up.
- Quantized outputs do not record which Hub commit they came from. The fetch prints the commit, and the cache keeps it on disk, but the output file has no provenance metadata yet. Adding it changes output bytes, so it needs its own decision.
- Windows Application Control on the development PC blocks newly built executables, so the Windows acceptance run for this milestone cannot be done on that PC (see Verification).

## Verification

- Unit tests (20, on Linux in Modal): repo-id and file-name validation, revision validation, SHA-256 and git-blob SHA-1 against known values, metadata parsing (license, gated flag, sibling filtering, commit-id format), shard index parsing, percent-encoding.
- Integration tests (11, `tests/hub_input.rs`) run `modelq` against a local stand-in for the Hub, with no network access: a single-file model end to end, a cache hit on the second run (one download), a sharded model (index plus two shards), a checksum mismatch (nothing kept), a gated repository refused without a token and accepted with one (the token is not printed), a resumed download (the server sees a `Range` request), a repository without weights, an unknown repository, a malformed id (no request sent), a local checkpoint that must not touch the network, and a missing local path that is not treated as a Hub id.
- Real Hub, Linux in Modal (`tools/hub_input/modal_hub_check.py`): `modelq quantize hf:Qwen/Qwen2.5-0.5B --format nvfp4-te` at the pinned commit `060db6499f32`. The license (apache-2.0) was printed before the 942 MB download; the file was stored after its SHA-256 matched; the quantized output is byte-identical (473,840,528 bytes, the same SHA-256) to the container the local CLI produced in Task 46. The second run reused the verified cache (16 s against 95 s). Evidence: [`m2-hub-input-qwen2.5-0.5b-linux.json`](../validation/m2-hub-input-qwen2.5-0.5b-linux.json).

The first real-Hub run failed, and the tests had not caught it. ureq's default TLS provider is Rustls, which this build does not compile in, so the first HTTPS request panicked. The fix selects the native TLS provider explicitly. The integration tests use plain HTTP to a local server, so they cannot exercise TLS; the real-Hub check is the only test of that path, and it must be rerun after any change to the HTTP client.

## Not verified

- The Windows acceptance criterion of ADR 0028 M2 ("a fresh Windows machine quantizes and evaluates a 0.5B model from one command") is not met. On the development PC, Windows Application Control blocks newly built executables, so the Windows build cannot be run there. The fix is a signed release (ADR 0028 section 7), not a change to the system's security setting.
- "Evaluates from one command" is not met either: `modelq eval` takes a local model directory, so the evaluation is a second command that runs after the fetch. Making `eval` accept `hf:` is the follow-up noted under Consequences.
- Windows and macOS TLS (Schannel and Security.framework) are not exercised by any test yet.
