//! `modelq quantize --format int4|int3|int2|int1`: group-wise low-bit output
//! (ADR 0030). The codec is in `modelq-quant`, the file format in `modelq-io`;
//! this module plans, writes (single file or shards), decodes the result, and
//! prints the report.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use modelq::{
    io::{
        layout::plan_output_layout_for,
        lowbit::{
            encoding_for, quantization_name, validate_outputs, write_lowbit_safetensors_with,
        },
        safetensors::TensorSource,
        sharded::SafetensorsInput,
        writer::WriterError,
    },
    quant::{
        lowbit::{LowBitConfig, Scheme},
        nvfp4_policy::DEFAULT_EXCLUDED_NAME_PARTS,
        policy::{PolicyAction, QuantizationPolicy, TensorDecision, preserve_named},
    },
};

use crate::output::OutputTarget;

/// How the weights were prepared before quantization (ADR 0037). It is recorded in the
/// output's metadata; the data-free default has none.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Provenance {
    /// The rescaling transform, for example `awq-v1`.
    pub transform: Option<String>,
    /// The calibration settings, as a compact JSON document.
    pub calibration: Option<String>,
}

impl Provenance {
    fn metadata(&self) -> Vec<(&'static str, String)> {
        let mut entries = Vec::new();
        if let Some(transform) = &self.transform {
            entries.push(("modelq.transform", transform.clone()));
        }
        if let Some(calibration) = &self.calibration {
            entries.push(("modelq.calibration", calibration.clone()));
        }
        entries
    }
}

/// Quantizes `input` with `config` and writes the result to `target`, recording `provenance`.
pub fn run(
    input: &Path,
    target: &OutputTarget,
    config: LowBitConfig,
    provenance: &Provenance,
) -> Result<(), String> {
    let source = SafetensorsInput::open(input).map_err(|error| error.to_string())?;
    let summaries = source.tensor_summaries();
    let source_bytes = source
        .source_paths()
        .iter()
        .map(|path| {
            std::fs::metadata(path)
                .map(|metadata| metadata.len())
                .map_err(|error| format!("could not stat {}: {error}", path.display()))
        })
        .sum::<Result<u64, String>>()?;
    let candidates = summaries
        .iter()
        .map(crate::candidate_for)
        .collect::<Result<Vec<_>, _>>()?;
    let mut decisions = QuantizationPolicy::default().decide_all(candidates);
    // The INT4 default keeps the vocabulary matrices at their source precision,
    // with the names the NVFP4 default preserves (ADR 0036). The experimental
    // formats keep the quantize-everything default.
    if config.bits == 4 {
        preserve_named(&mut decisions, DEFAULT_EXCLUDED_NAME_PARTS);
    }
    let encoding = encoding_for(&config);
    let plan = plan_output_layout_for(&summaries, &decisions, encoding)
        .map_err(|error| format!("could not plan output: {error}"))?;

    let quantized = decisions
        .iter()
        .filter(|decision| decision.action == PolicyAction::Quantize)
        .count();
    println!(
        "Format: {} ({}, group size {}, {} bits per value)",
        quantization_name(&config),
        scheme_label(config.scheme),
        config.group_size,
        config.bits
    );
    println!(
        "Planning: {} source tensors, {quantized} quantized, {} preserved, {} output tensors, {} data bytes",
        summaries.len(),
        summaries.len() - quantized,
        plan.tensors.len(),
        plan.total_data_bytes
    );
    if let Some(transform) = &provenance.transform {
        println!("Calibration: {transform} (ADR 0037)");
    }
    println!("Writing output: {}", target.describe());

    let mut sized: BTreeMap<String, u64> = BTreeMap::new();
    for tensor in &plan.tensors {
        *sized.entry(tensor.source_name.clone()).or_default() += tensor.byte_len;
    }
    let sized: Vec<(String, u64)> = sized.into_iter().collect();
    let extra = provenance.metadata();
    let written = target.write(&source, &sized, |subset, path| {
        let subset_summaries = subset.tensor_summaries();
        let names: BTreeSet<&str> = subset_summaries
            .iter()
            .map(|summary| summary.name.as_str())
            .collect();
        let subset_decisions: Vec<TensorDecision> = decisions
            .iter()
            .filter(|decision| names.contains(decision.name.as_str()))
            .cloned()
            .collect();
        let subset_plan = plan_output_layout_for(&subset_summaries, &subset_decisions, encoding)
            .map_err(|error| WriterError::Layout { source: error })?;
        write_lowbit_safetensors_with(
            subset,
            &subset_plan,
            &subset_decisions,
            config,
            path,
            &extra,
        )?;
        Ok::<_, WriterError>(
            subset_plan
                .tensors
                .iter()
                .map(|tensor| (tensor.name.clone(), tensor.byte_len))
                .collect(),
        )
    })?;

    println!("Validating output by reopening and decoding it...");
    // A sharded write also lists its index file, which is JSON rather than a
    // SafeTensors shard, so only the shards are reopened and decoded.
    let shards: Vec<PathBuf> = written
        .paths
        .iter()
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .cloned()
        .collect();
    let validation = validate_outputs(&source, &shards)?;
    let output_bytes: u64 = written
        .paths
        .iter()
        .map(|path| {
            std::fs::metadata(path)
                .map(|metadata| metadata.len())
                .map_err(|error| format!("could not stat {}: {error}", path.display()))
        })
        .sum::<Result<u64, String>>()?;

    print_report(&Report {
        source_path: input,
        output_path: target.path(),
        config,
        quantized: validation.quantized_tensors,
        preserved: validation.preserved_tensors,
        source_bytes,
        output_bytes,
        max_mse: validation.max_mse,
        max_mae: validation.max_mae,
        max_abs_error: validation.max_abs_error,
        lowest_sqnr_db: validation.lowest_sqnr_db,
        scale_bound_violations: validation.scale_bound_violations,
    });
    Ok(())
}

fn scheme_label(scheme: Scheme) -> &'static str {
    match scheme {
        Scheme::Symmetric => "symmetric group-wise",
        Scheme::Sign => "sign with mean-abs scale",
    }
}

struct Report<'a> {
    source_path: &'a Path,
    output_path: &'a Path,
    config: LowBitConfig,
    quantized: usize,
    preserved: usize,
    source_bytes: u64,
    output_bytes: u64,
    max_mse: f64,
    max_mae: f64,
    max_abs_error: f64,
    lowest_sqnr_db: Option<f64>,
    scale_bound_violations: u64,
}

fn print_report(report: &Report<'_>) {
    println!();
    println!("Final report:");
    println!("  Source: {}", report.source_path.display());
    println!("  Output: {}", report.output_path.display());
    println!(
        "  Format: {} ({}, group size {})",
        quantization_name(&report.config),
        scheme_label(report.config.scheme),
        report.config.group_size
    );
    println!("  Compatibility: ModelQ-native representation; no inference runtime is claimed");
    println!(
        "  Tensors: {} quantized, {} preserved",
        report.quantized, report.preserved
    );
    println!("  Source bytes: {}", report.source_bytes);
    println!("  Output bytes: {}", report.output_bytes);
    if report.output_bytes < report.source_bytes {
        println!(
            "  Size: smaller by {} bytes",
            report.source_bytes - report.output_bytes
        );
    } else {
        println!(
            "  Size: larger by {} bytes",
            report.output_bytes.saturating_sub(report.source_bytes)
        );
    }
    println!(
        "  Validation: passed (decoded {} quantized tensors; {} preserved tensors identical)",
        report.quantized, report.preserved
    );
    println!("  Max MSE: {:.3e}", report.max_mse);
    println!("  Max MAE: {:.3e}", report.max_mae);
    println!("  Max absolute error: {:.3e}", report.max_abs_error);
    match report.lowest_sqnr_db {
        Some(sqnr_db) => println!("  Lowest SQNR: {sqnr_db:.2} dB"),
        None => println!("  Lowest SQNR: undefined"),
    }
    if report.config.scheme == Scheme::Symmetric {
        println!(
            "  Values beyond half a group scale: {} (symmetric rounding allows none)",
            report.scale_bound_violations
        );
    }
}
