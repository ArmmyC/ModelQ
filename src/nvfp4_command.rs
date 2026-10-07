//! `modelq quantize --format nvfp4` and `--format nvfp4-te`: NVFP4 export.
//!
//! `nvfp4` writes the ModelQ-native container (ADR 0011); `nvfp4-te` writes
//! the Transformer Engine rowwise container (schema v2, ADR 0021).
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
            Nvfp4Execution, Nvfp4OutputPlan, Nvfp4OutputRole, Nvfp4Settings, plan_nvfp4_output,
            write_nvfp4_safetensors_settings,
        },
        safetensors::{MappedSafetensors, TensorSource, TensorSummary},
        sharded::SafetensorsInput,
        te_container::{
            TeOutputPlan, TeOutputRole, plan_te_output, read_te_container_manifest,
            te_matrix_values, write_te_nvfp4_safetensors_settings,
        },
    },
    quant::{
        nvfp4::{ScaleSelection, dequantize_iter},
        nvfp4_policy::{Nvfp4Candidate, Nvfp4Decision, Nvfp4Policy},
        policy::PolicyAction,
    },
};

/// Which container the command writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nvfp4Profile {
    /// The ModelQ-native NVFP4 container.
    Native,
    /// The Transformer Engine rowwise container (schema v2).
    TransformerEngine,
}

/// The planned output of either profile.
enum Plan {
    Native(Nvfp4OutputPlan),
    TransformerEngine(TeOutputPlan),
}

impl Plan {
    /// Output tensor count, data-section bytes, and each source's output bytes.
    fn summary(&self) -> (usize, u64, Vec<(String, u64)>) {
        let (count, total, pairs): (usize, u64, Vec<(&str, u64)>) = match self {
            Self::Native(plan) => (
                plan.tensors.len(),
                plan.total_data_bytes,
                plan.tensors
                    .iter()
                    .map(|tensor| (tensor.source_name.as_str(), tensor.byte_len))
                    .collect(),
            ),
            Self::TransformerEngine(plan) => (
                plan.tensors.len(),
                plan.total_data_bytes,
                plan.tensors
                    .iter()
                    .map(|tensor| (tensor.source_name.as_str(), tensor.byte_len))
                    .collect(),
            ),
        };
        let mut sized: BTreeMap<String, u64> = BTreeMap::new();
        for (name, bytes) in pairs {
            *sized.entry(name.to_owned()).or_default() += bytes;
        }
        (count, total, sized.into_iter().collect())
    }
}

/// Block-scale search radius used when `--scale-search` is not given.
///
/// Radius 6 lowered the WikiText-2 perplexity loss by about a fifth on three
/// Qwen2.5 models and larger radii added nothing (ADR 0026, ADR 0027).
/// `--scale-search 0` selects the reference rule.  The library keeps the
/// reference rule as its own default.
pub const DEFAULT_SCALE_SEARCH_RADIUS: u8 = 6;

/// Options that shape the NVFP4 selection policy.
pub struct Nvfp4Options {
    pub profile: Nvfp4Profile,
    pub exclude: Vec<String>,
    pub default_excludes: bool,
    /// Worker threads; `None` uses every CPU and `Some(1)` the sequential path.
    pub threads: Option<usize>,
    /// Block-scale search radius in E4M3 codes. `None` uses
    /// [`DEFAULT_SCALE_SEARCH_RADIUS`]; `Some(0)` selects the reference rule.
    pub scale_search: Option<u8>,
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

    fn scales(&self) -> ScaleSelection {
        match self.scale_search.unwrap_or(DEFAULT_SCALE_SEARCH_RADIUS) {
            0 => ScaleSelection::Amax,
            radius => ScaleSelection::MinMse { radius },
        }
    }

    fn policy(&self) -> Nvfp4Policy {
        let mut policy = match self.profile {
            Nvfp4Profile::Native => Nvfp4Policy::new(),
            Nvfp4Profile::TransformerEngine => Nvfp4Policy::transformer_engine(),
        };
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
    note: &'static str,
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
    let settings = Nvfp4Settings {
        execution,
        scales: options.scales(),
    };
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
    let shape_rule = if policy.is_transformer_engine() {
        "rank-2 tensors with a leading dimension divisible by 16 and a final dimension divisible by 32"
    } else {
        "rank>=2 tensors with a final dimension divisible by 16"
    };
    println!(
        "Policy: floating {shape_rule} and at least {} elements; excluded name parts: {}",
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
    let plan = match options.profile {
        Nvfp4Profile::Native => Plan::Native(
            plan_nvfp4_output(&summaries, &selected)
                .map_err(|error| format!("could not plan output: {error}"))?,
        ),
        Nvfp4Profile::TransformerEngine => Plan::TransformerEngine(
            plan_te_output(&summaries, &selected)
                .map_err(|error| format!("could not plan output: {error}"))?,
        ),
    };
    let (output_tensors, total_data_bytes, sized) = plan.summary();

    println!(
        "Planning: {} source tensors, {} to quantize, {} output tensors, {} data bytes",
        summaries.len(),
        selected.len(),
        output_tensors,
        total_data_bytes
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
    if settings.scales.is_default() {
        println!("Block scales: {} (reference rule)", settings.scales.label());
    } else {
        println!(
            "Block scales: {} (minimum-error search)",
            settings.scales.label()
        );
    }
    println!("Writing output: {}", target.describe());
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
        match options.profile {
            Nvfp4Profile::Native => {
                let subset_plan = plan_nvfp4_output(&subset_summaries, &subset_selected)
                    .map_err(|error| format!("could not plan shard: {error}"))?;
                write_nvfp4_safetensors_settings(subset, &subset_plan, path, settings)
                    .map_err(|error| error.to_string())?;
                Ok::<_, String>(
                    subset_plan
                        .tensors
                        .iter()
                        .map(|tensor| (tensor.name.clone(), tensor.byte_len))
                        .collect(),
                )
            }
            Nvfp4Profile::TransformerEngine => {
                let subset_plan = plan_te_output(&subset_summaries, &subset_selected)
                    .map_err(|error| format!("could not plan shard: {error}"))?;
                write_te_nvfp4_safetensors_settings(subset, &subset_plan, path, settings)
                    .map_err(|error| error.to_string())?;
                Ok::<_, String>(
                    subset_plan
                        .tensors
                        .iter()
                        .map(|tensor| (tensor.name.clone(), tensor.byte_len))
                        .collect(),
                )
            }
        }
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
        note: match options.profile {
            Nvfp4Profile::Native => {
                "ModelQ-native NVFP4 output; no runtime compatibility is implied."
            }
            Nvfp4Profile::TransformerEngine => {
                "Transformer Engine rowwise NVFP4 container (profile transformer-engine.nvfp4.rowwise.1x16.v1, TE 2.19.0). Validated on an NVIDIA B200 for loading each matrix and one TN GEMM per matrix (ADR 0022); other TE versions, GPUs, whole-model loading and inference are not validated."
            }
        },
    };
    match &plan {
        Plan::Native(plan) => {
            validate_output(&source, &output_reader, plan, &summaries, &mut report)?;
        }
        Plan::TransformerEngine(plan) => {
            for path in written
                .paths
                .iter()
                .filter(|path| !path.to_string_lossy().ends_with(".index.json"))
            {
                let file = MappedSafetensors::open(path)
                    .map_err(|error| format!("could not reopen {}: {error}", path.display()))?;
                read_te_container_manifest(&file)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
            }
            validate_te_output(&source, &output_reader, plan, &summaries, &mut report)?;
        }
    }
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

fn validate_te_output(
    source: &impl TensorSource,
    output: &impl TensorSource,
    plan: &TeOutputPlan,
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
            TeOutputRole::Preserved => {
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
            TeOutputRole::RowwiseData => {
                let summary = summaries_by_name
                    .get(tensor.source_name.as_str())
                    .ok_or_else(|| format!("missing source tensor {:?}", tensor.source_name))?;
                let [rows, columns] = summary.shape[..] else {
                    return Err(format!("{:?} is not rank two", tensor.source_name));
                };
                let scale_name = format!("{}.rowwise_scale_inv", tensor.source_name);
                let amax_name = format!("{}.amax_rowwise", tensor.source_name);
                let metrics = output
                    .with_tensor_bytes(&tensor.name, |data| {
                        output.with_tensor_bytes(&scale_name, |scales| {
                            output.with_tensor_bytes(&amax_name, |amax| {
                                check_quantized_te(
                                    source,
                                    &tensor.source_name,
                                    (rows, columns),
                                    data,
                                    scales,
                                    amax,
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
            TeOutputRole::RowwiseScaleInv | TeOutputRole::AmaxRowwise => {}
        }
    }
    Ok(())
}

fn check_quantized_te(
    source: &impl TensorSource,
    source_name: &str,
    (rows, columns): (usize, usize),
    data: &[u8],
    scales: &[u8],
    amax_bytes: &[u8],
) -> Result<modelq::diagnostics::ReconstructionMetrics, String> {
    let amax = f32::from_le_bytes(
        amax_bytes
            .try_into()
            .map_err(|_| format!("amax of {source_name:?} has {} bytes", amax_bytes.len()))?,
    );
    let reconstructed = te_matrix_values(rows, columns, data, scales, amax)
        .map_err(|error| format!("{source_name:?}: {error}"))?;
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
    println!("  Note: {}", report.note);
}
