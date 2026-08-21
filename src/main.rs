use std::{collections::BTreeSet, path::PathBuf};

use clap::{Arg, ArgMatches, Command, value_parser};
use modelq::{
    diagnostics::{int8_tensor_diagnostics, reconstruction_metrics, saturation_count},
    io::{
        layout::{OutputTensorRole, plan_output_layout},
        safetensors::{Inspection, MappedSafetensors, TensorSummary, inspect_file},
        writer::write_safetensors,
    },
    quant::{
        int8::{dequantize, quantize},
        policy::{PolicyAction, QuantizationPolicy, TensorCandidate, TensorDecision},
    },
};

fn main() {
    let matches = build_cli().get_matches();
    let result = match matches.subcommand() {
        Some(("inspect", matches)) => {
            let Some(path) = matches.get_one::<PathBuf>("model") else {
                print_error(&"inspect requires a model path");
            };
            inspect_file(path)
                .map(|inspection| print_inspection(&inspection))
                .map_err(|error| error.to_string())
        }
        Some(("quantize", matches)) => {
            run_quantize(matches).map(|report| print_quantize_report(&report))
        }
        _ => Ok(()),
    };

    if let Err(error) = result {
        print_error(&error);
    }
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
                .about("Quantize a SafeTensors checkpoint with the CPU INT8 path")
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

fn run_quantize(matches: &ArgMatches) -> Result<QuantizeReport, String> {
    let input = matches
        .get_one::<PathBuf>("model")
        .ok_or_else(|| "quantize requires a model path".to_owned())?
        .clone();
    let output = matches
        .get_one::<PathBuf>("output")
        .ok_or_else(|| "quantize requires --output <PATH>".to_owned())?
        .clone();
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

    println!("Inspecting source: {}", input.display());
    let source = MappedSafetensors::open(&input).map_err(|error| error.to_string())?;
    let inspection = source.inspection();
    let candidates = inspection
        .tensors
        .iter()
        .map(candidate_for)
        .collect::<Result<Vec<_>, _>>()?;
    let decisions = QuantizationPolicy::default().decide_all(candidates);
    let plan = plan_output_layout(&inspection.tensors, &decisions)
        .map_err(|error| format!("could not plan output: {error}"))?;

    println!(
        "Planning: {} source tensors, {} output tensors, {} data bytes",
        inspection.tensors.len(),
        plan.tensors.len(),
        plan.total_data_bytes
    );
    print_progress(&source, &inspection.tensors, &decisions)?;

    println!("Writing output: {}", output.display());
    write_safetensors(&source, &plan, &decisions, &output)
        .map_err(|error| format!("could not write output: {error}"))?;

    println!("Validating output by reopening and dequantizing it...");
    let output_reader = MappedSafetensors::open(&output)
        .map_err(|error| format!("output could not be reopened: {error}"))?;
    let validation = validate_output(&source, &output_reader, &plan)?;

    Ok(QuantizeReport {
        source_path: input,
        output_path: output,
        source_bytes: inspection.file_size,
        output_bytes: output_reader.file_size(),
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
    source: &MappedSafetensors,
    summaries: &[TensorSummary],
    decisions: &[TensorDecision],
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
                let view = source
                    .tensor(&summary.name)
                    .map_err(|error| error.to_string())?;
                let values = view.values().collect::<Vec<_>>();
                let quantized = quantize(&values)
                    .map_err(|error| format!("could not quantize {:?}: {error}", summary.name))?;
                let diagnostics = int8_tensor_diagnostics(&values, &quantized, summary.byte_len, 4)
                    .map_err(|error| format!("could not diagnose {:?}: {error}", summary.name))?;
                println!(
                    "Progress: {}/{} | {} | quantize | mse={:.3e} | mae={:.3e} | scale={:.6e}",
                    index + 1,
                    sorted_summaries.len(),
                    summary.name,
                    diagnostics.mse,
                    diagnostics.mae,
                    quantized.scale()
                );
            }
        }
    }
    Ok(())
}

fn validate_output(
    source: &MappedSafetensors,
    output: &MappedSafetensors,
    plan: &modelq::io::layout::OutputLayoutPlan,
) -> Result<ValidationReport, String> {
    let expected_names = plan
        .tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    let actual_names = output
        .tensors()
        .map(|tensor| tensor.name.clone())
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
                let source_bytes = source
                    .tensor_bytes(&tensor.source_name)
                    .map_err(|error| error.to_string())?;
                let output_bytes = output
                    .tensor_bytes(&tensor.name)
                    .map_err(|error| error.to_string())?;
                if source_bytes != output_bytes {
                    return Err(format!(
                        "preserved tensor {:?} changed during writing",
                        tensor.source_name
                    ));
                }
                report.preserved_tensors += 1;
            }
            OutputTensorRole::QuantizedData => {
                let source_values = source
                    .tensor(&tensor.source_name)
                    .map_err(|error| error.to_string())?
                    .values()
                    .collect::<Vec<_>>();
                let qdata = output
                    .tensor_bytes(&tensor.name)
                    .map_err(|error| error.to_string())?
                    .iter()
                    .map(|&value| value as i8)
                    .collect::<Vec<_>>();
                let scale_name = format!("{}.scale", tensor.source_name);
                let scale_tensor = plan
                    .tensor(&scale_name)
                    .ok_or_else(|| format!("missing scale tensor {scale_name:?}"))?;
                if scale_tensor.role != OutputTensorRole::QuantizationScale {
                    return Err(format!("tensor {scale_name:?} is not a scale tensor"));
                }
                let scale_bytes = output
                    .tensor_bytes(&scale_name)
                    .map_err(|error| error.to_string())?;
                if scale_bytes.len() != 4 {
                    return Err(format!(
                        "scale tensor {scale_name:?} has {} bytes instead of 4",
                        scale_bytes.len()
                    ));
                }
                let scale = f32::from_le_bytes(
                    scale_bytes
                        .try_into()
                        .expect("the scale length was checked above"),
                );
                let reconstructed = dequantize(&qdata, scale)
                    .map_err(|error| format!("could not dequantize {:?}: {error}", tensor.name))?;
                let metrics = reconstruction_metrics(&source_values, reconstructed)
                    .map_err(|error| format!("could not validate {:?}: {error}", tensor.name))?;
                report.quantized_tensors += 1;
                report.saturated_values += saturation_count(&qdata);
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
