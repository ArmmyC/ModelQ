//! `modelq quantize --format nvfp4`: ModelQ-native NVFP4 export.
//!
//! Reads a single or sharded SafeTensors checkpoint, applies [`Nvfp4Policy`],
//! streams the selected tensors through the bounded NVFP4 quantizer into one
//! ADR 0011 SafeTensors file, then reopens that file and checks every tensor
//! without materializing a whole tensor.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use crate::output::OutputTarget;
use modelq::{
    backend::{cpu::ParallelConfig, nvfp4 as parallel_nvfp4},
    diagnostics::reconstruction_metrics_streaming,
    io::{
        nvfp4::{
            Nvfp4Execution, Nvfp4OutputPlan, Nvfp4OutputRole, plan_nvfp4_output,
            write_nvfp4_safetensors_with,
        },
        safetensors::{TensorSource, TensorSummary},
        sharded::SafetensorsInput,
    },
    quant::{
        nvfp4::dequantize_iter,
        nvfp4_policy::{Nvfp4Candidate, Nvfp4Decision, Nvfp4Policy},
        policy::PolicyAction,
    },
};

/// Options that shape the NVFP4 selection policy.
pub struct Nvfp4Options {
    pub exclude: Vec<String>,
    pub default_excludes: bool,
    /// Worker threads; `None` uses every CPU and `Some(1)` the sequential path.
    pub threads: Option<usize>,
}

impl Nvfp4Options {
    fn execution(&self) -> Result<Nvfp4Execution, String> {
        match self.threads {
            Some(0) => Err("--threads must be at least 1".to_owned()),
            Some(1) => Ok(Nvfp4Execution::Sequential),
            Some(workers) => Ok(Nvfp4Execution::Parallel(ParallelConfig::new(
                workers,
                parallel_nvfp4::DEFAULT_CHUNK_ELEMENTS,
            ))),
            None => Ok(Nvfp4Execution::Parallel(ParallelConfig::automatic(
                parallel_nvfp4::DEFAULT_CHUNK_ELEMENTS,
            ))),
        }
    }

    fn policy(&self) -> Nvfp4Policy {
        let mut policy = Nvfp4Policy::new();
        if !self.default_excludes {
            policy = policy.without_default_exclusions();
        }
        for pattern in &self.exclude {
            policy = policy.with_exclusion(pattern.clone());
        }
        policy
    }
}

struct Nvfp4Report {
    source_path: PathBuf,
    output_path: PathBuf,
    source_bytes: u64,
    output_bytes: u64,
    quantized_tensors: usize,
    preserved_tensors: usize,
    max_mse: f64,
    max_mae: f64,
    max_abs_error: f64,
    lowest_sqnr_db: Option<f64>,
}

/// Runs the command and prints progress and the final report.
pub fn run(input: &Path, target: &OutputTarget, options: &Nvfp4Options) -> Result<(), String> {
    let report = quantize(input, target, options)?;
    print_report(&report);
    Ok(())
}

fn quantize(
    input: &Path,
    target: &OutputTarget,
    options: &Nvfp4Options,
) -> Result<Nvfp4Report, String> {
    let execution = options.execution()?;
    println!("Inspecting source: {}", input.display());
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

    let policy = options.policy();
    println!(
        "Policy: floating rank>=2 tensors with a final dimension divisible by 16 and at least {} elements; excluded name parts: {}",
        policy.minimum_elements(),
        if policy.excluded_name_parts().is_empty() {
            "none".to_owned()
        } else {
            policy.excluded_name_parts().join(", ")
        }
    );
    let candidates: Vec<Nvfp4Candidate> = summaries
        .iter()
        .map(|summary| Nvfp4Candidate {
            name: summary.name.clone(),
            is_floating: matches!(summary.dtype.as_str(), "F32" | "F16" | "BF16"),
            shape: summary.shape.clone(),
        })
        .collect();
    let decisions = policy.decide_all(&candidates);
    let selected: Vec<String> = decisions
        .iter()
        .filter(|decision| decision.is_quantized())
        .map(|decision| decision.name.clone())
        .collect();
    let plan = plan_nvfp4_output(&summaries, &selected)
        .map_err(|error| format!("could not plan output: {error}"))?;

    println!(
        "Planning: {} source tensors, {} to quantize, {} output tensors, {} data bytes",
        summaries.len(),
        selected.len(),
        plan.tensors.len(),
        plan.total_data_bytes
    );
    print_decisions(&decisions);

    match execution {
        Nvfp4Execution::Sequential => println!("Execution: sequential"),
        Nvfp4Execution::Parallel(config) => {
            println!(
                "Execution: parallel, up to {} workers",
                config.workers.min(64)
            );
        }
    }
    println!("Writing output: {}", target.describe());
    let mut sized: BTreeMap<String, u64> = BTreeMap::new();
    for tensor in &plan.tensors {
        *sized.entry(tensor.source_name.clone()).or_default() += tensor.byte_len;
    }
    let sized: Vec<(String, u64)> = sized.into_iter().collect();
    let written = target.write(&source, &sized, |subset, path| {
        let subset_summaries = subset.tensor_summaries();
        let subset_names: BTreeSet<&str> = subset_summaries
            .iter()
            .map(|summary| summary.name.as_str())
            .collect();
        let subset_selected: Vec<String> = selected
            .iter()
            .filter(|name| subset_names.contains(name.as_str()))
            .cloned()
            .collect();
        let subset_plan = plan_nvfp4_output(&subset_summaries, &subset_selected)
            .map_err(|error| format!("could not plan shard: {error}"))?;
        write_nvfp4_safetensors_with(subset, &subset_plan, path, execution)
            .map_err(|error| error.to_string())?;
        Ok::<_, String>(
            subset_plan
                .tensors
                .iter()
                .map(|tensor| (tensor.name.clone(), tensor.byte_len))
                .collect(),
        )
    })?;

    println!("Validating output by reopening and dequantizing it...");
    let output_reader = target.open_output()?;
    let mut report = Nvfp4Report {
        source_path: input.to_owned(),
        output_path: target.path().to_owned(),
        source_bytes,
        output_bytes: OutputTarget::committed_bytes(&written)?,
        quantized_tensors: 0,
        preserved_tensors: 0,
        max_mse: 0.0,
        max_mae: 0.0,
        max_abs_error: 0.0,
        lowest_sqnr_db: None,
    };
    validate_output(&source, &output_reader, &plan, &summaries, &mut report)?;
    Ok(report)
}

fn print_decisions(decisions: &[Nvfp4Decision]) {
    let mut sorted: Vec<&Nvfp4Decision> = decisions.iter().collect();
    sorted.sort_by(|left, right| left.name.cmp(&right.name));
    let total = sorted.len();
    for (index, decision) in sorted.into_iter().enumerate() {
        let action = match decision.action {
            PolicyAction::Quantize => "quantize",
            PolicyAction::Preserve => "preserve",
        };
        println!(
            "Progress: {}/{} | {} | {action} ({})",
            index + 1,
            total,
            decision.name,
            decision.reason
        );
    }
}

fn validate_output(
    source: &impl TensorSource,
    output: &impl TensorSource,
    plan: &Nvfp4OutputPlan,
    summaries: &[TensorSummary],
    report: &mut Nvfp4Report,
) -> Result<(), String> {
    let mut expected_names: Vec<&str> = plan
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    let output_summaries = output.tensor_summaries();
    let mut actual_names: Vec<&str> = output_summaries
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    expected_names.sort_unstable();
    actual_names.sort_unstable();
    if expected_names != actual_names {
        return Err("output tensor names do not match the planned layout".to_owned());
    }

    let summaries_by_name: BTreeMap<&str, &TensorSummary> = summaries
        .iter()
        .map(|summary| (summary.name.as_str(), summary))
        .collect();

    for tensor in &plan.tensors {
        match tensor.role {
            Nvfp4OutputRole::Preserved => {
                let unchanged = output
                    .with_tensor_bytes(&tensor.name, |output_bytes| {
                        source.with_tensor_bytes(&tensor.source_name, |source_bytes| {
                            source_bytes == output_bytes
                        })
                    })
                    .map_err(|error| error.to_string())?
                    .map_err(|error| error.to_string())?;
                if !unchanged {
                    return Err(format!(
                        "preserved tensor {:?} changed during writing",
                        tensor.source_name
                    ));
                }
                report.preserved_tensors += 1;
            }
            Nvfp4OutputRole::QuantizedData => {
                let source_summary = summaries_by_name
                    .get(tensor.source_name.as_str())
                    .ok_or_else(|| format!("missing source tensor {:?}", tensor.source_name))?;
                let elements = source_summary
                    .shape
                    .iter()
                    .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
                    .ok_or_else(|| {
                        format!(
                            "tensor {:?} overflows its element count",
                            tensor.source_name
                        )
                    })?;
                let block_name = format!("{}.block_scale", tensor.source_name);
                let global_name = format!("{}.global_scale", tensor.source_name);
                let metrics = output
                    .with_tensor_bytes(&tensor.name, |packed| {
                        output.with_tensor_bytes(&block_name, |block_scales| {
                            output.with_tensor_bytes(&global_name, |global_bytes| {
                                check_quantized_nvfp4(
                                    source,
                                    &tensor.source_name,
                                    packed,
                                    block_scales,
                                    global_bytes,
                                    elements,
                                )
                            })
                        })
                    })
                    .map_err(|error| error.to_string())?
                    .map_err(|error| error.to_string())?
                    .map_err(|error| error.to_string())??;
                report.quantized_tensors += 1;
                report.max_mse = report.max_mse.max(metrics.mse);
                report.max_mae = report.max_mae.max(metrics.mae);
                report.max_abs_error = report.max_abs_error.max(metrics.max_abs_error);
                if let Some(sqnr_db) = metrics.sqnr_db {
                    report.lowest_sqnr_db = Some(
                        report
                            .lowest_sqnr_db
                            .map_or(sqnr_db, |current| current.min(sqnr_db)),
                    );
                }
            }
            Nvfp4OutputRole::BlockScales | Nvfp4OutputRole::GlobalScale => {}
        }
    }
    Ok(())
}

fn check_quantized_nvfp4(
    source: &impl TensorSource,
    source_name: &str,
    packed: &[u8],
    block_scales: &[u8],
    global_bytes: &[u8],
    elements: usize,
) -> Result<modelq::diagnostics::ReconstructionMetrics, String> {
    let global_scale = f32::from_le_bytes(global_bytes.try_into().map_err(|_| {
        format!(
            "global scale for {source_name:?} has {} bytes instead of 4",
            global_bytes.len()
        )
    })?);
    let reconstructed = dequantize_iter(packed, block_scales, global_scale, elements)
        .map_err(|error| format!("could not dequantize {source_name:?}: {error}"))?;
    source
        .with_tensor(source_name, |view| {
            reconstruction_metrics_streaming(view.values(), reconstructed)
        })
        .map_err(|error| error.to_string())?
        .map_err(|error| format!("could not validate {source_name:?}: {error}"))
}

fn print_report(report: &Nvfp4Report) {
    println!();
    println!("Final report:");
    println!("  Source: {}", report.source_path.display());
    println!("  Output: {}", report.output_path.display());
    println!(
        "  Tensors: {} quantized, {} preserved",
        report.quantized_tensors, report.preserved_tensors
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
        "  Validation: passed (reopened and dequantized {} quantized tensors)",
        report.quantized_tensors
    );
    println!("  Max MSE: {:.3e}", report.max_mse);
    println!("  Max MAE: {:.3e}", report.max_mae);
    println!("  Max absolute error: {:.3e}", report.max_abs_error);
    match report.lowest_sqnr_db {
        Some(sqnr_db) => println!("  Lowest SQNR: {sqnr_db:.2} dB"),
        None => println!("  Lowest SQNR: undefined"),
    }
    println!("  Note: ModelQ-native NVFP4 output; no runtime compatibility is implied.");
}
