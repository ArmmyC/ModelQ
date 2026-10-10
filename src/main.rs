mod eval_command;
mod hub;
mod lowbit_command;
mod nvfp4_command;
mod output;

use std::{collections::BTreeSet, path::PathBuf};

use clap::{Arg, ArgMatches, Command, value_parser};
use modelq::{
    backend::{cpu::ParallelConfig, int8 as parallel_int8},
    io::{
        gguf_qwen2::WeightQuantization,
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
        Some(("eval", matches)) => run_eval_command(matches),
        Some(("formats", _)) => {
            print_formats();
            Ok(())
        }
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

/// Prints the format registry as a table (ADR 0028 section 2).
fn print_formats() {
    let rows: Vec<[String; 6]> = modelq::quant::formats::FORMATS
        .iter()
        .map(|spec| {
            [
                spec.id.to_owned(),
                spec.bits.to_string(),
                spec.default_group_size
                    .map_or_else(|| "-".to_owned(), |size| size.to_string()),
                spec.status.label(),
                if spec.requires_experimental_flag {
                    "--experimental".to_owned()
                } else {
                    "-".to_owned()
                },
                spec.container.to_owned(),
            ]
        })
        .collect();
    let headers = ["FORMAT", "BITS", "GROUP", "STATUS", "REQUIRES", "CONTAINER"];
    let mut widths = headers.map(str::len);
    for row in &rows {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.len());
        }
    }
    let line = |cells: [&str; 6]| {
        cells
            .iter()
            .zip(widths)
            .map(|(cell, width)| format!("{cell:<width$}"))
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_owned()
    };
    println!("{}", line(headers));
    for row in &rows {
        println!("{}", line(row.each_ref().map(String::as_str)));
    }
    println!();
    println!(
        "Group-wise formats write ModelQ-native files; no inference runtime is claimed unless STATUS says so."
    );
}

fn run_eval_command(matches: &ArgMatches) -> Result<(), String> {
    let options = eval_command::EvalOptions {
        model: matches
            .get_one::<String>("model")
            .cloned()
            .unwrap_or_default(),
        download: matches.get_flag("download"),
        dataset: matches.get_one::<PathBuf>("dataset").cloned(),
        containers: matches
            .get_many::<PathBuf>("container")
            .map(|values| values.cloned().collect())
            .unwrap_or_default(),
        device: matches
            .get_one::<String>("device")
            .cloned()
            .unwrap_or_else(|| "auto".to_owned()),
        max_windows: matches.get_one::<u64>("max-windows").copied(),
        report: matches
            .get_one::<PathBuf>("report")
            .cloned()
            .unwrap_or_default(),
    };
    let python = eval_command::python_interpreter(matches.get_one::<PathBuf>("python"));
    let script = eval_command::locate_script()?;
    eval_command::run(&options, &python, &script)
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
                        .required(true)
                        .help("a local checkpoint, or hf:<owner>/<name> to fetch it from the Hugging Face Hub"),
                )
                .arg(
                    Arg::new("group-size")
                        .long("group-size")
                        .value_name("N")
                        .value_parser(value_parser!(u64).range(1..))
                        .help("int4, int3, int2 and int1: values per scale group (default: 128)"),
                )
                .arg(
                    Arg::new("experimental")
                        .long("experimental")
                        .action(clap::ArgAction::SetTrue)
                        .help("required to write experimental formats such as int3, int2 and int1"),
                )
                .arg(
                    Arg::new("revision")
                        .long("revision")
                        .value_name("REVISION")
                        .value_parser(value_parser!(String))
                        .default_value("main")
                        .help("with hf:, the branch, tag or commit to fetch"),
                )
                .arg(
                    Arg::new("cache-dir")
                        .long("cache-dir")
                        .value_name("PATH")
                        .value_parser(value_parser!(PathBuf))
                        .help("with hf:, where downloads are cached (default: $MODELQ_CACHE, then the user cache)"),
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
                    Arg::new("scale-search")
                        .long("scale-search")
                        .value_name("RADIUS")
                        .value_parser(value_parser!(u8).range(0..=32))
                        .help(
                            "nvfp4 formats only: pick each block's E4M3 scale by minimum error within RADIUS codes of the reference rule (default: 6; 0 = the reference rule alone)",
                        ),
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
        .subcommand(
            Command::new("formats")
                .about("List the quantization formats, their status, and whether they need --experimental"),
        )
        .subcommand(
            Command::new("eval")
                .about("Measure a quantized output against the original model on WikiText-2 (runs locally; needs Python)")
                .arg(
                    Arg::new("model")
                        .long("model")
                        .value_name("MODEL")
                        .required(true)
                        .help("a local model directory, or a pinned model id with --download"),
                )
                .arg(
                    Arg::new("container")
                        .long("container")
                        .value_name("PATH")
                        .value_parser(value_parser!(PathBuf))
                        .action(clap::ArgAction::Append)
                        .required(true)
                        .help("a ModelQ NVFP4 Transformer Engine container; repeat to compare several"),
                )
                .arg(
                    Arg::new("dataset")
                        .long("dataset")
                        .value_name("PATH")
                        .value_parser(value_parser!(PathBuf))
                        .help("local WikiText-2 test Parquet file (required without --download)"),
                )
                .arg(
                    Arg::new("download")
                        .long("download")
                        .action(clap::ArgAction::SetTrue)
                        .help("allow downloading the pinned model and dataset"),
                )
                .arg(
                    Arg::new("device")
                        .long("device")
                        .value_name("DEVICE")
                        .value_parser(value_parser!(String))
                        .default_value("auto")
                        .help("auto, cpu or cuda"),
                )
                .arg(
                    Arg::new("max-windows")
                        .long("max-windows")
                        .value_name("N")
                        .value_parser(value_parser!(u64).range(1..))
                        .help("score only the first N windows (a quick check, not comparable to full runs)"),
                )
                .arg(
                    Arg::new("report")
                        .long("report")
                        .value_name("PATH")
                        .value_parser(value_parser!(PathBuf))
                        .required(true)
                        .help("where to write the JSON report"),
                )
                .arg(
                    Arg::new("python")
                        .long("python")
                        .value_name("PATH")
                        .value_parser(value_parser!(PathBuf))
                        .help("Python interpreter (default: $MODELQ_PYTHON, then python)"),
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
        .ok_or_else(|| "quantize requires --format <int8|nvfp4|nvfp4-te>".to_owned())?;
    let has_nvfp4_options = matches.contains_id("exclude")
        || matches.contains_id("scale-search")
        || matches.get_flag("no-default-excludes");
    let has_group_size = matches.contains_id("group-size");
    if let Some(spec) = modelq::quant::formats::find(format) {
        if spec.requires_experimental_flag && !matches.get_flag("experimental") {
            return Err(format!(
                "{format} is experimental; pass --experimental to write it (see `modelq formats`)"
            ));
        }
    }
    match format.as_str() {
        "int8" => {
            if has_nvfp4_options || has_group_size {
                return Err(
                    "--exclude, --no-default-excludes, --scale-search and --group-size do not apply to int8"
                        .to_owned(),
                );
            }
            run_quantize(matches).map(|report| print_quantize_report(&report))
        }
        "gguf-q8_0" | "gguf-q4_0" => {
            if has_nvfp4_options || has_group_size || matches.contains_id("max-shard-size") {
                return Err(format!(
                    "--exclude, --no-default-excludes, --scale-search, --group-size and --max-shard-size do not apply to {format}"
                ));
            }
            let quantization = if format.as_str() == "gguf-q4_0" {
                WeightQuantization::Q4_0
            } else {
                WeightQuantization::Q8_0
            };
            let (model, output) = quantize_paths(matches)?;
            require_cpu(matches)?;
            let input = resolve_model_input_with(matches, model, GGUF_EXTRAS)?;
            if !input.is_dir() {
                return Err(format!(
                    "{format} needs a model directory with config.json, tokenizer.json, tokenizer_config.json and the weights; {} is not one",
                    input.display()
                ));
            }
            let report = modelq::io::gguf_qwen2::export_qwen2(&input, &output, quantization)
                .map_err(|error| error.to_string())?;
            println!("Format: {format} (GGUF v3, qwen2 architecture, {quantization:?} weights)");
            println!(
                "Written: {} ({} bytes)",
                output.display(),
                report.output_bytes
            );
            println!(
                "Tensors: {} {quantization:?}, {} F32; metadata entries: {}; vocabulary: {}",
                report.quantized_tensors,
                report.f32_tensors,
                report.metadata_entries,
                report.vocab_size
            );
            println!(
                "Compatibility: see `modelq formats` for the verified runtime status (ADR 0032, ADR 0033)"
            );
            Ok(())
        }
        "int4" | "int3" | "int2" | "int1" => {
            if has_nvfp4_options {
                return Err(
                    "--exclude, --no-default-excludes and --scale-search apply only to the nvfp4 formats"
                        .to_owned(),
                );
            }
            let (model, output) = quantize_paths(matches)?;
            require_cpu(matches)?;
            let target = output::OutputTarget::from_args(
                output,
                matches.get_one::<String>("max-shard-size"),
            )?;
            let input = resolve_model_input(matches, model)?;
            let spec = modelq::quant::formats::find(format)
                .ok_or_else(|| format!("unknown format {format:?}"))?;
            let bits = spec.bits;
            let config = modelq::quant::lowbit::LowBitConfig {
                bits,
                group_size: matches
                    .get_one::<u64>("group-size")
                    .map(|&size| size as usize)
                    .or(spec.default_group_size)
                    .ok_or_else(|| "this format needs --group-size".to_owned())?,
                scheme: if bits == 1 {
                    modelq::quant::lowbit::Scheme::Sign
                } else {
                    modelq::quant::lowbit::Scheme::Symmetric
                },
            };
            lowbit_command::run(&input, &target, config)
        }
        "nvfp4" | "nvfp4-te" => {
            let (model, output) = quantize_paths(matches)?;
            require_cpu(matches)?;
            let target = output::OutputTarget::from_args(
                output,
                matches.get_one::<String>("max-shard-size"),
            )?;
            let input = resolve_model_input(matches, model)?;
            let options = nvfp4_command::Nvfp4Options {
                profile: if format == "nvfp4-te" {
                    nvfp4_command::Nvfp4Profile::TransformerEngine
                } else {
                    nvfp4_command::Nvfp4Profile::Native
                },
                exclude: matches
                    .get_many::<String>("exclude")
                    .map(|values| values.cloned().collect())
                    .unwrap_or_default(),
                default_excludes: !matches.get_flag("no-default-excludes"),
                threads: matches.get_one::<usize>("threads").copied(),
                scale_search: matches.get_one::<u8>("scale-search").copied(),
            };
            nvfp4_command::run(&input, &target, &options)
        }
        other => Err(format!(
            "unsupported format {other:?}; supported formats are {}",
            modelq::quant::formats::FORMATS
                .iter()
                .map(|spec| spec.id)
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// Turns `hf:<owner>/<name>` into a cached local directory; any other path is
/// returned unchanged. Called only after the output path has been validated,
/// so a bad output never causes a download.
fn resolve_model_input(matches: &ArgMatches, model: PathBuf) -> Result<PathBuf, String> {
    resolve_model_input_with(matches, model, &[])
}

/// Like [`resolve_model_input`], and for a Hub model also fetches `extras`.
fn resolve_model_input_with(
    matches: &ArgMatches,
    model: PathBuf,
    extras: &[&str],
) -> Result<PathBuf, String> {
    let repo_text = model
        .to_str()
        .and_then(|text| text.strip_prefix(hub::PREFIX))
        .map(str::to_owned);
    let Some(repo_text) = repo_text else {
        return Ok(model);
    };
    let repo = hub::RepoId::parse(&repo_text)?;
    let cache_dir = match matches.get_one::<PathBuf>("cache-dir") {
        Some(dir) => dir.clone(),
        None => hub::default_cache_dir()?,
    };
    let revision = matches
        .get_one::<String>("revision")
        .cloned()
        .unwrap_or_else(|| "main".to_owned());
    hub::fetch(
        &hub::Request {
            repo,
            revision,
            cache_dir,
            endpoint: hub::endpoint_from_env(),
            token: hub::token_from_env(),
        },
        extras,
    )
}

/// The files a GGUF export reads besides the weights.
const GGUF_EXTRAS: &[&str] = &["config.json", "tokenizer.json", "tokenizer_config.json"];

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
    let model = matches
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
    let input = resolve_model_input(matches, model)?;
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
    fn parses_eval_options() {
        let matches = build_cli()
            .try_get_matches_from([
                "modelq",
                "eval",
                "--model",
                "models/qwen",
                "--dataset",
                "wikitext.parquet",
                "--container",
                "a.safetensors",
                "--container",
                "b.safetensors",
                "--max-windows",
                "8",
                "--report",
                "out.json",
            ])
            .expect("eval command arguments are valid");
        let (_, eval) = matches
            .subcommand()
            .expect("the eval subcommand is present");

        let containers: Vec<&PathBuf> = eval.get_many::<PathBuf>("container").unwrap().collect();
        assert_eq!(containers.len(), 2);
        assert_eq!(eval.get_one::<u64>("max-windows"), Some(&8));
        assert_eq!(eval.get_one::<String>("device"), Some(&"auto".to_owned()));
        assert!(!eval.get_flag("download"));
    }

    #[test]
    fn eval_needs_a_container_and_a_report() {
        assert!(
            build_cli()
                .try_get_matches_from(["modelq", "eval", "--model", "m", "--report", "r.json"])
                .is_err()
        );
        assert!(
            build_cli()
                .try_get_matches_from([
                    "modelq",
                    "eval",
                    "--model",
                    "m",
                    "--container",
                    "c.safetensors"
                ])
                .is_err()
        );
    }

    #[test]
    fn rejects_zero_max_windows() {
        assert!(
            build_cli()
                .try_get_matches_from([
                    "modelq",
                    "eval",
                    "--model",
                    "m",
                    "--container",
                    "c",
                    "--report",
                    "r",
                    "--max-windows",
                    "0",
                ])
                .is_err()
        );
    }

    #[test]
    fn parses_group_size_and_experimental_flag() {
        let matches = build_cli()
            .try_get_matches_from([
                "modelq",
                "quantize",
                "in.safetensors",
                "--format",
                "int2",
                "--group-size",
                "64",
                "--experimental",
                "--output",
                "out.safetensors",
            ])
            .expect("quantize arguments are valid");
        let (_, quantize) = matches.subcommand().expect("quantize is present");
        assert_eq!(quantize.get_one::<u64>("group-size"), Some(&64));
        assert!(quantize.get_flag("experimental"));
    }

    #[test]
    fn parses_formats_subcommand() {
        let matches = build_cli()
            .try_get_matches_from(["modelq", "formats"])
            .expect("formats takes no arguments");
        assert_eq!(matches.subcommand_name(), Some("formats"));
    }

    #[test]
    fn rejects_zero_group_size() {
        assert!(
            build_cli()
                .try_get_matches_from([
                    "modelq",
                    "quantize",
                    "in",
                    "--format",
                    "int4",
                    "--group-size",
                    "0",
                    "--output",
                    "out",
                ])
                .is_err()
        );
    }

    #[test]
    fn rejects_missing_subcommands() {
        assert!(build_cli().try_get_matches_from(["modelq"]).is_err());
    }
}
