use std::path::PathBuf;

use clap::{Arg, Command, value_parser};
use modelq::io::safetensors::{Inspection, inspect_file};

fn main() {
    let matches = build_cli().get_matches();
    let result = match matches.subcommand() {
        Some(("inspect", matches)) => {
            let Some(path) = matches.get_one::<PathBuf>("model") else {
                eprintln!("error: inspect requires a model path");
                std::process::exit(2);
            };
            inspect_file(path).map(|inspection| print_inspection(&inspection))
        }
        _ => Ok(()),
    };

    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
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
    fn rejects_missing_subcommands() {
        assert!(build_cli().try_get_matches_from(["modelq"]).is_err());
    }
}
