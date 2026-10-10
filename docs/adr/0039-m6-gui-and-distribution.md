# ADR 0039: M6 Plan: GUI and Signed Distribution

- Status: Proposed. The owner decisions in the last section are open. Nothing in this ADR is implemented.
- Date: 2026-10-11
- Scope: milestone M6 of ADR 0028 ("A GUI over the public API, and signed binaries for three OSes"). It sets the order of the work and lists the decisions it needs. It adds no code.
- Builds on: [ADR 0028](0028-open-source-quantizer-scope.md) (sections 5 to 7, and open question 4), [ADR 0037](0037-awq-quantize.md), and PROJECT.md section 13 (GUI direction).

## Context

- Milestones M1 to M5 are done. M4 (calibration) shipped as opt-in AWQ ([ADR 0037](0037-awq-quantize.md)), and [ADR 0038](0038-awq-sharded-checkpoints.md) lets it read sharded checkpoints.
- ADR 0028 section 6 starts the GUI only after the library surface is stable. PROJECT.md section 13 says the same: do not build the GUI before the CLI and library stabilize.
- ADR 0028 section 7 says normal users need signed binaries. Windows Application Control and Smart App Control have already blocked unsigned executables built on the development PC.

## Finding: the library does not hold the flows a GUI would call

`src/lib.rs` exports `error`, `tensor`, `io` (from `modelq-io`), `quant` (from `modelq-quant`), `diagnostics` and `backend`. The flows a user runs are private to the binary crate, because `src/main.rs` declares them with `mod`:

- quantize: `lowbit_command`, `nvfp4_command`, `output`
- calibration: `awq_command`
- Hub input: `hub`
- evaluation: `eval_command`

A GUI cannot call any of them today. The first M6 step is therefore to move these flows into the library behind a small API that the CLI also calls. That move is the real API-stability work, and it comes before any GUI code.

## Decisions proposed

1. **Stable API before GUI code.** The first M6 task moves the flows above into the library, lists every public item with a stability label (stable or experimental), and states how CLI flags are deprecated. The CLI keeps its behaviour and its tests. The GUI calls only this API, as PROJECT.md section 13 requires.
2. **GUI framework: a spike before a commitment.** The candidate is egui with eframe. It is pure Rust, uses immediate mode, builds as one binary, and targets Windows, macOS and Linux, with wgpu as the default renderer. docs.rs listed eframe 0.36.1 as the latest version when this was researched; confirm the current version on crates.io before pinning it. The spike builds one screen (pick a model, pick a format, run quantize, show the report) and records the binary size, the startup time, and whether an unsigned build runs on the development PC.
3. **Scope of the first screens.** Quantize with and without AWQ, with progress and the final report. Evaluation after quantize. Accept both a local folder and a Hub id. No model editing.
4. **Acceptance refinement.** ADR 0028 asks for signed binaries for three operating systems. Windows and macOS have a signing mechanism that the OS checks. Linux does not, so Linux ships tarballs with SHA-256 checksums. This narrows the wording of ADR 0028 M6; the owner must accept it.
5. **Release pipeline.** GitHub Actions builds release binaries on the three hosted runners that CI already uses, attaches checksums, and signs on Windows and macOS with secrets stored in the repository settings.

## Signing options (owner decision)

**Windows**

- **SignPath Foundation, free for eligible open-source projects.** The conditions reported from SignPath's open-source terms are: an OSI-approved license (ModelQ is Apache-2.0), active maintenance, a public repository with free releases, automated builds traceable to committed code, a published code-signing policy, and a download page that says what the software does. The certificate is issued to the Foundation, so builds carry the Foundation's name as publisher, and breaking the rules can end the subscription. Parts of the terms page were labelled as a draft when searched, so read the current terms at the time of application. ModelQ appears to meet the license and public-repository conditions; the rest need checking at application.
- **Azure Artifact Signing (formerly Trusted Signing).** Individual validation was reported as open only to applicants in the United States and Canada. Organizations need a supported region and a legal entity. Billing starts when the account is created, even if validation fails. These points come from Microsoft Q&A answers and forum posts, not from Microsoft's own eligibility page, so confirm them on Microsoft Learn before creating an account.
- **A commercial code-signing certificate** (OV or EV) from a certificate authority, or a cloud signing service. Prices and the SmartScreen reputation it earns are not evaluated here.
- **Unsigned.** Windows warnings and Smart App Control blocks remain. This does not meet ADR 0028 section 7.

**macOS**

- **Apple Developer Program.** US$99 a year, as reported by several third-party guides; confirm the price at developer.apple.com. Distribution outside the Mac App Store needs a Developer ID Application certificate, the hardened runtime, and notarization by Apple (with `notarytool`). Stapling attaches the notarization ticket. Without these, Gatekeeper blocks downloaded builds.

**Linux**

- No platform signing. Checksums, and optionally a GPG signature.

**Recommendation.** Apply to SignPath Foundation first, since it costs nothing and the license and repository conditions appear to be met. If the publisher name rules it out, use Artifact Signing when the owner is eligible, or a commercial certificate otherwise. Enroll in the Apple Developer Program before the first macOS release.

## Milestone plan after the decisions

- **M6.1 Stable API.** Move the flows into the library and list the public items. Acceptance: the CLI runs the same code and its tests pass unchanged.
- **M6.2 GUI spike.** One screen, built in CI on the three operating systems. Acceptance: the spike's measurements are recorded.
- **M6.3 GUI quantize and evaluation.** Progress, the final report, and an evaluation view.
- **M6.4 Release pipeline.** Builds, checksums, and signing for the chosen routes, triggered by a version tag.
- **M6.5 Install guide and known limits.** Written for normal users.

## Open questions for the owner

1. Which Windows signing route: SignPath Foundation, Artifact Signing, or a commercial certificate?
2. Enroll in the Apple Developer Program now, or with the first macOS release?
3. Accept the Linux refinement (checksums rather than signatures)?
4. Approve M6.1, which moves the flows into the library without changing the CLI's behaviour?
5. Approve the GUI spike after M6.1, with egui and eframe as the candidate?

## Consequences

- Moving the flows into the library is the largest change of M6. It makes the CLI and the GUI share one code path, so behaviour changes are tested once.
- Until the signing route is chosen, Windows users keep the warnings described in ADR 0028 section 7.
- The Apple fee and the Windows certificate costs are recurring; they need the owner's approval before any purchase.

## Sources researched on 2026-10-11

- SignPath open-source program and terms: <https://about.signpath.io/product/open-source> and <https://signpath.org/terms>
- Microsoft Learn, Artifact Signing quickstart: <https://learn.microsoft.com/azure/artifact-signing/quickstart>
- Apple, notarizing macOS software before distribution: <https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution>
- eframe on docs.rs: <https://docs.rs/crate/eframe/latest>
