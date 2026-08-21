//! Portable execution backends for ModelQ quantization.
//!
//! The scalar algorithms remain in [`modelq_quant`]. This crate owns execution
//! strategy, including bounded parallel CPU work, so optimized backends can be
//! compared with the scalar reference without changing the representations.

pub mod cpu;
