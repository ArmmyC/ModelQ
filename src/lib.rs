//! Compatibility facade and library entry point for ModelQ.
//!
//! The reusable implementation now lives in workspace crates. These reexports
//! preserve the pre-workspace `modelq::tensor`, `modelq::quant`,
//! `modelq::diagnostics`, and `modelq::io` paths for existing callers while
//! the root package continues to provide the `modelq` binary.

pub use modelq_core::{error, tensor};
pub use modelq_io as io;
pub use modelq_quant as quant;
pub use modelq_quant::diagnostics;

/// Compatibility namespace for the portable CPU execution backend.
pub mod backend;
