mod nvfp4_command;
mod output;

use std::{collections::BTreeSet, path::PathBuf};

use clap::{Arg, ArgMatches, Command, value_parser};
use modelq::{
    backend::{cpu::ParallelConfig, int8 as parallel_int8},
    io::{
        layout::{OutputTensorRole, plan_output_layout},
        safetensors::{Inspection, TensorSource, TensorSummary, inspect_file},
        sharded::SafetensorsInput,
        writer::{Int8Execution, WriterError, write_safetensors_with},
    },
    quant::policy::{PolicyAction, QuantizationPolicy, TensorCandidate, TensorDecision},
};

fn main() {
    let matches = build_cli().get_matches();
    let result = match matches.subcommand() {
        Some(("inspect", matches)) => {
            let Some(path) = matches.get_one::<PathBuf>("model") else {
                print_error(&"inspect requires a model path");
            };
            run_inspect(path)
        }
        Some(("quantize", matches)) => run_quantize_command(matches),
        _ => Ok(()),
    };

    if let Err(error) = result {
        print_error(&error);
    }
}

fn run_inspect(path: &std::path::Path) -> Result<(), String> {
    if path.is_file() && !path.to_string_lossy().ends_with(".index.json") {
        return inspect_file(path)
            .map(|inspection| print_inspection(&inspection))
            .map_err(|error| error.to_string());
    }
    let input = SafetensorsInput::open(path).map_err(|error| error.to_string())?;
    let shards = input.source_paths();
    let total_bytes = input
        .tensors()
        .iter()
        .map(|tensor| tensor.summary.byte_len)
        .sum::<u64>();
    println!("Format: SafeTensors (sharded)");
    println!("Shards: {}", shards.len());
    println!("Payload bytes: {total_bytes}");
    println!("Tensors: {}", input.tensors().len());
    for tensor in input.tensors() {
        let shard = tensor
            .shard
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
        println!(
            "  {} | dtype={} | shape={:?} | bytes={} | shard={shard}",
            tensor.summary.name,
            tensor.summary.dtype,
            tensor.summary.shape,
            tensor.summary.byte_len
        );
    }
    Ok(())
}

fn print_error(error: &dyn std::fmt::Display) -> ! {
    eprintln!("error: {error}");
    std::process::exit(1);
}

fn build_cli() -> Command {
    Command::new("modelq")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Inspect and transform model checkpoints")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("inspect")
                .about("Inspect SafeTensors metadata")
                .arg(
                    Arg::new("model")
                        .value_name("MODEL")
                        .value_parser(value_parser!(PathBuf))
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("quantize")
                .about("Quantize a SafeTensors checkpoint (CPU, INT8 or ModelQ-native NVFP4)")
                .arg(
                    Arg::new("model")
                        .value_name("MODEL")
                        .value_parser(value_parser!(PathBuf))
                        .required(true),
                )
                .arg(
                    Arg::new("format")
                        .long("format")
                        .value_name("FORMAT")
                        .value_parser(value_parser!(String))
                        .required(true),
                )
                .arg(
                    Arg::new("device")
                        .long("device")
                        .value_name("DEVICE")
                        .value_parser(value_parser!(String))
                        .default_value("cpu"),
                )
                .arg(
                    Arg::new("output")
                        .long("output")
                        .value_name("PATH")
                        .value_parser(value_parser!(PathBuf))
                        .required(true),
                )
                .arg(
                    Arg::new("max-shard-size")
                        .long("max-shard-size")
                        .value_name("SIZE")
                        .value_parser(value_parser!(String))
                        .help(
                            "write a sharded directory at --output; SIZE is bytes or e.g. 500MB, 2GiB",
                        ),
                )
                .arg(
                    Arg::new("threads")
                        .long("threads")
                        .value_name("N")
                        .value_parser(value_parser!(usize))
                        .help("worker threads (default: all CPUs; 1 = sequential reference path)"),
                )
                .arg(
                    Arg::new("exclude")
                        .long("exclude")
                        .value_name("SUBSTRING")
                        .value_parser(value_parser!(String))
                        .action(clap::ArgAction::Append)
                        .help("nvfp4 only: also preserve tensors whose name contains SUBSTRING"),
                )
                .arg(
                    Arg::new("no-default-excludes")
                        .long("no-default-excludes")
                        .action(clap::ArgAction::SetTrue)
                        .help("nvfp4 only: quantize embedding and lm_head tensors too"),
                ),
        )
}

fn print_inspection(inspection: &Inspection) {
    println!("Format: SafeTensors");
    println!("File size: {} bytes", inspection.file_size);
    println!("Tensors: {}", inspection.tensors.len());
    for tensor in &inspection.tensors {
        println!(
            "  {} | dtype={} | shape={:?} | bytes={}",
            tensor.name, tensor.dtype, tensor.shape, tensor.byte_len
        );
    }
}

struct QuantizeReport {
    source_path: PathBuf,
    output_path: PathBuf,
    source_bytes: u64,
    output_bytes: u64,
    validation: ValidationReport,
}

struct ValidationReport {
    quantized_tensors: usize,
    preserved_tensors: usize,
    saturated_values: u64,
    max_mse: f64,
    max_mae: f64,
    max_abs_error: f64,
    lowest_sqnr_db: Option<f64>,
}

fn run_quantize_command(matches: &ArgMatches) -> Result<(), String> {
    let format = matches
        .get_one::<String>("format")
        .ok_or_else(|| "quantize requires --format <int8|nvfp4>".to_owned())?;
    let has_nvfp4_options =
        matches.contains_id("exclude") || matches.get_flag("no-default-excludes");
    match format.as_str() {
        "int8" => {
            if has_nvfp4_options {
                return Err(
                    "--exclude and --no-default-excludes apply only to --format nvfp4".to_owned(),
                );
            }
            run_quantize(matches).map(|report| print_quantize_report(&report))
        }
        "nvfp4" => {
            let (input, output) = quantize_paths(matches)?;
            require_cpu(matches)?;
            let target = output::OutputTarget::from_args(
                output,
                matches.get_one::<String>("max-shard-size"),
            )?;
            let options = nvfp4_command::Nvfp4Options {
                exclude: matches
                    .get_many::<String>("exclude")
                    .map(|values| values.cloned().collect())
                    .unwrap_or_default(),
                default_excludes: !matches.get_flag("no-default-excludes"),
                threads: matches.get_one::<usize>("threads").copied(),
            };
            nvfp4_command::run(&input, &target, &options)
        }
        other => Err(format!(
            "unsupported format {other:?}; supported formats are int8 and nvfp4"
        )),
    }
}

fn quantize_paths(matches: &ArgMatches) -> Result<(PathBuf, PathBuf), String> {
    let input = matches
        .get_one::<PathBuf>("model")
        .ok_or_else(|| "quantize requires a model path".to_owned())?
        .clone();
    let output = matches
        .get_one::<PathBuf>("output")
        .ok_or_else(|| "quantize requires --output <PATH>".to_owned())?
        .clone();
    Ok((input, output))
}

fn require_cpu(matches: &ArgMatches) -> Result<(), String> {
    let device = matches
        .get_one::<String>("device")
        .map(String::as_str)
        .unwrap_or("cpu");
    if device != "cpu" {
        return Err(format!(
            "unsupported device {device:?}; only cpu is supported"
        ));
    }
    Ok(())
}

fn run_quantize(matches: &ArgMatches) -> Result<QuantizeReport, String> {
    let input = matches
        .get_one::<PathBuf>("model")
        .ok_or_else(|| "quantize requires a model path".to_owned())?
        .clone();
    let target = output::OutputTarget::from_args(
        matches
            .get_one::<PathBuf>("output")
            .ok_or_else(|| "quantize requires --output <PATH>".to_owned())?
            .clone(),
        matches.get_one::<String>("max-shard-size"),
    )?;
    let format = matches
        .get_one::<String>("format")
        .ok_or_else(|| "quantize requires --format int8".to_owned())?;
    if format != "int8" {
        return Err(format!(
            "unsupported format {format:?}; Task 12 supports only int8"
        ));
    }
    let device = matches
        .get_one::<String>("device")
        .map(String::as_str)
        .unwrap_or("cpu");
    if device != "cpu" {
        return Err(format!(
            "unsupported device {device:?}; Task 12 supports only cpu"
        ));
    }

    let workers = worker_count(matches)?;
    let config = ParallelConfig::new(workers, parallel_int8::DEFAULT_CHUNK_ELEMENTS);
    let execution = if workers == 1 {
        Int8Execution::Sequential
    } else {
        Int8Execution::Parallel(config)
    };
    println!("Inspecting source: {}", input.display());
    let source = SafetensorsInput::open(&input).map_err(|error| error.to_string())?;
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
        .map(candidate_for)
        .collect::<Result<Vec<_>, _>>()?;
    let decisions = QuantizationPolicy::default().decide_all(candidates);
    let plan = plan_output_layout(&summaries, &decisions)
        .map_err(|error| format!("could not plan output: {error}"))?;

    println!(
        "Planning: {} source tensors, {} output tensors, {} data bytes",
        summaries.len(),
        plan.tensors.len(),
        plan.total_data_bytes
    );
    print_progress(&source, &summaries, &decisions, config)?;

    println!(
        "Execution: {}",
        if workers == 1 {
            "sequential writer".to_owned()
        } else {
            format!("parallel, up to {} workers", workers.min(64))
        }
    );
    println!("Writing output: {}", target.describe());
    let mut sized: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for tensor in &plan.tensors {
        *sized.entry(tensor.source_name.clone()).or_default() += tensor.byte_len;
    }
    let sized: Vec<(String, u64)> = sized.into_iter().collect();
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
        let subset_plan = plan_output_layout(&subset_summaries, &subset_decisions)
            .map_err(|source| WriterError::Layout { source })?;
        write_safetensors_with(subset, &subset_plan, &subset_decisions, path, execution)?;
        Ok::<_, WriterError>(
            subset_plan
                .tensors
                .iter()
                .map(|tensor| (tensor.name.clone(), tensor.byte_len))
                .collect(),
        )
    })?;

    println!("Validating output by reopening and dequantizing it...");
    let output_reader = target.open_output()?;
    let validation = validate_output(&source, &output_reader, &plan, config)?;

    Ok(QuantizeReport {
        source_path: input,
        output_path: target.path().to_owned(),
        source_bytes,
        output_bytes: output::OutputTarget::committed_bytes(&written)?,
        validation,
    })
}

fn candidate_for(summary: &TensorSummary) -> Result<TensorCandidate, String> {
    let element_count = checked_element_count(&summary.shape).ok_or_else(|| {
        format!(
            "tensor {:?} shape {:?} overflows its element count",
            summary.name, summary.shape
        )
    })?;
    let candidate = if is_floating_dtype(&summary.dtype) {
        TensorCandidate::floating(summary.name.clone(), element_count)
    } else {
        TensorCandidate::non_floating(summary.name.clone(), element_count)
    };
    Ok(candidate)
}

fn print_progress(
    source: &impl TensorSource,
    summaries: &[TensorSummary],
    decisions: &[TensorDecision],
    config: ParallelConfig,
) -> Result<(), String> {
    let decisions_by_name = decisions
        .iter()
        .map(|decision| (decision.name.as_str(), decision))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut sorted_summaries = summaries.iter().collect::<Vec<_>>();
    sorted_summaries.sort_by(|left, right| left.name.cmp(&right.name));

    for (index, summary) in sorted_summaries.iter().enumerate() {
        let decision = decisions_by_name
            .get(summary.name.as_str())
            .ok_or_else(|| format!("missing policy decision for {:?}", summary.name))?;
        match decision.action {
            PolicyAction::Preserve => println!(
                "Progress: {}/{} | {} | preserve ({})",
                index + 1,
                sorted_summaries.len(),
                summary.name,
                decision.reason
            ),
            PolicyAction::Quantize => {
                let (scale, diagnostics) = source
                    .with_tensor(&summary.name, |view| {
                        parallel_int8::tensor_diagnostics_replay(
                            || view.values(),
                            config,
                            summary.byte_len,
                            4,
                        )
                    })
                    .map_err(|error| error.to_string())?
                    .map_err(|error| format!("could not diagnose {:?}: {error}", summary.name))?;
                println!(
                    "Progress: {}/{} | {} | quantize | mse={:.3e} | mae={:.3e} | scale={:.6e}",
                    index + 1,
                    sorted_summaries.len(),
                    summary.name,
                    diagnostics.mse,
                    diagnostics.mae,
                    scale
                );
            }
        }
    }
    Ok(())
}

fn validate_output(
    source: &impl TensorSource,
    output: &impl TensorSource,
    plan: &modelq::io::layout::OutputLayoutPlan,
    config: ParallelConfig,
) -> Result<ValidationReport, String> {
    let expected_names = plan
        .tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    let actual_names = output
        .tensor_summaries()
        .into_iter()
        .map(|tensor| tensor.name)
        .collect::<BTreeSet<_>>();
    if expected_names != actual_names {
        return Err("output tensor names do not match the planned layout".to_owned());
    }

    let mut report = ValidationReport {
        quantized_tensors: 0,
        preserved_tensors: 0,
        saturated_values: 0,
        max_mse: 0.0,
        max_mae: 0.0,
        max_abs_error: 0.0,
        lowest_sqnr_db: None,
    };

    for tensor in &plan.tensors {
        match tensor.role {
            OutputTensorRole::Preserved => {
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
            OutputTensorRole::QuantizedData => {
                let scale_name = format!("{}.scale", tensor.source_name);
                let scale_tensor = plan
                    .tensor(&scale_name)
                    .ok_or_else(|| format!("missing scale tensor {scale_name:?}"))?;
                if scale_tensor.role != OutputTensorRole::QuantizationScale {
                    return Err(format!("tensor {scale_name:?} is not a scale tensor"));
                }
                let (metrics, saturated) = output
                    .with_tensor_bytes(&tensor.name, |qdata| {
                        output.with_tensor_bytes(&scale_name, |scale_bytes| {
                            check_quantized_int8(source, tensor, qdata, scale_bytes, config)
                        })
                    })
                    .map_err(|error| error.to_string())?
                    .map_err(|error| error.to_string())??;
                report.quantized_tensors += 1;
                report.saturated_values += saturated;
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
            OutputTensorRole::QuantizationScale => {}
        }
    }

    Ok(report)
}

fn check_quantized_int8(
    source: &impl TensorSource,
    tensor: &modelq::io::layout::OutputTensorPlan,
    qdata: &[u8],
    scale_bytes: &[u8],
    config: ParallelConfig,
) -> Result<(modelq::diagnostics::ReconstructionMetrics, u64), String> {
    if scale_bytes.len() != 4 {
        return Err(format!(
            "scale tensor for {:?} has {} bytes instead of 4",
            tensor.source_name,
            scale_bytes.len()
        ));
    }
    let scale = f32::from_le_bytes(
        scale_bytes
            .try_into()
            .expect("the scale length was checked above"),
    );
    source
        .with_tensor(&tensor.source_name, |view| {
            parallel_int8::validate_and_measure(|| view.values(), qdata, scale, config)
        })
        .map_err(|error| error.to_string())?
        .map_err(|error| format!("could not validate {:?}: {error}", tensor.name))
}

/// Worker count for the parallel paths: `--threads N`, or every CPU.
fn worker_count(matches: &ArgMatches) -> Result<usize, String> {
    match matches.get_one::<usize>("threads").copied() {
        Some(0) => Err("--threads must be at least 1".to_owned()),
        Some(workers) => Ok(workers),
        None => Ok(ParallelConfig::automatic(1).workers),
    }
}

fn print_quantize_report(report: &QuantizeReport) {
    println!();
    println!("Final report:");
    println!("  Source: {}", report.source_path.display());
    println!("  Output: {}", report.output_path.display());
    println!(
        "  Tensors: {} quantized, {} preserved",
        report.validation.quantized_tensors, report.validation.preserved_tensors
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
        report.validation.quantized_tensors
    );
    println!("  Max MSE: {:.3e}", report.validation.max_mse);
    println!("  Max MAE: {:.3e}", report.validation.max_mae);
    println!(
        "  Max absolute error: {:.3e}",
        report.validation.max_abs_error
    );
    println!(
        "  Saturated INT8 values: {}",
        report.validation.saturated_values
    );
    match report.validation.lowest_sqnr_db {
        Some(sqnr_db) => println!("  Lowest SQNR: {:.2} dB", sqnr_db),
        None => println!("  Lowest SQNR: undefined"),
    }
}

fn checked_element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
}

fn is_floating_dtype(dtype: &str) -> bool {
    matches!(dtype, "F32" | "F16" | "BF16")
}

#[cfg(test)]
mod tests {
    use super::build_cli;
    use std::path::PathBuf;

    #[test]
    fn parses_inspect_model_path() {
        let matches = build_cli()
            .try_get_matches_from(["modelq", "inspect", "fixture.safetensors"])
            .expect("inspect command arguments are valid");
        let (_, inspect) = matches
            .subcommand()
            .expect("the inspect subcommand is present");

        assert_eq!(
            inspect.get_one::<PathBuf>("model"),
            Some(&PathBuf::from("fixture.safetensors"))
        );
    }

    #[test]
    fn parses_quantize_options() {
        let matches = build_cli()
            .try_get_matches_from([
                "modelq",
                "quantize",
                "input.safetensors",
                "--format",
                "int8",
                "--device",
                "cpu",
                "--output",
                "output.safetensors",
            ])
            .expect("quantize command arguments are valid");
        let (_, quantize) = matches
            .subcommand()
            .expect("the quantize subcommand is present");

        assert_eq!(
            quantize.get_one::<PathBuf>("model"),
            Some(&PathBuf::from("input.safetensors"))
        );
        assert_eq!(
            quantize.get_one::<String>("format"),
            Some(&"int8".to_owned())
        );
        assert_eq!(
            quantize.get_one::<String>("device"),
            Some(&"cpu".to_owned())
        );
        assert_eq!(
            quantize.get_one::<PathBuf>("output"),
            Some(&PathBuf::from("output.safetensors"))
        );
    }

    #[test]
    fn rejects_missing_subcommands() {
        assert!(build_cli().try_get_matches_from(["modelq"]).is_err());
    }
}
