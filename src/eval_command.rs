//! `modelq eval`: score a quantized output against the original model on the
//! user's own machine (ADR 0028, milestone M1).
//!
//! The measurement runs in Python, in `tools/quality_eval/modelq_eval.py`,
//! which needs Python 3 with PyTorch, transformers, pyarrow and
//! huggingface_hub. This module checks the arguments that can be checked
//! without Python, finds the script and the interpreter, runs them with the
//! terminal attached, and turns their exit status into a result. It never
//! downloads anything itself: `--download` is passed through, and only then
//! does the script fetch the pinned model and dataset.

use std::{
    ffi::{OsStr, OsString},
    io::ErrorKind,
    path::{Path, PathBuf},
    process::Command,
};

/// Environment variable that overrides the location of the evaluation script.
pub const SCRIPT_ENV: &str = "MODELQ_EVAL_SCRIPT";
/// Environment variable that overrides the Python interpreter.
pub const PYTHON_ENV: &str = "MODELQ_PYTHON";
/// Location of the script in a source checkout, relative to the working directory.
pub const DEFAULT_SCRIPT: &str = "tools/quality_eval/modelq_eval.py";
/// Interpreter used when neither `--python` nor `MODELQ_PYTHON` is set.
pub const DEFAULT_PYTHON: &str = "python";

const DEVICES: [&str; 3] = ["auto", "cpu", "cuda"];

/// The arguments of `modelq eval`, as the script receives them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalOptions {
    /// A local model directory, or a pinned model id when `download` is set.
    pub model: String,
    /// Allow the script to download the pinned model and dataset.
    pub download: bool,
    /// A local WikiText-2 test Parquet file; required unless `download` is set.
    pub dataset: Option<PathBuf>,
    /// ModelQ Transformer Engine NVFP4 containers to compare with the original.
    pub containers: Vec<PathBuf>,
    /// `auto`, `cpu` or `cuda`.
    pub device: String,
    /// Score only the first N windows of the dataset.
    pub max_windows: Option<u64>,
    /// Where the script writes its JSON report.
    pub report: PathBuf,
}

impl EvalOptions {
    /// Checks everything that does not need Python, so that a mistake is
    /// reported before a model is loaded.
    pub fn validate(&self) -> Result<(), String> {
        if self.containers.is_empty() {
            return Err("eval needs at least one --container".to_owned());
        }
        for container in &self.containers {
            if !container.is_file() {
                return Err(format!("container not found: {}", container.display()));
            }
        }
        if !DEVICES.contains(&self.device.as_str()) {
            return Err(format!(
                "--device must be one of {}, not {:?}",
                DEVICES.join(", "),
                self.device
            ));
        }
        if let Some(dataset) = &self.dataset {
            if !dataset.is_file() {
                return Err(format!("dataset not found: {}", dataset.display()));
            }
        }
        if self.download {
            return Ok(());
        }
        let model = Path::new(&self.model);
        if !model.join("model.safetensors").is_file() {
            return Err(format!(
                "{} is not a local model directory (no model.safetensors); \
                 to download a pinned model, pass --download",
                self.model
            ));
        }
        if self.dataset.is_none() {
            return Err(
                "without --download, eval needs --dataset with a local WikiText-2 test Parquet file"
                    .to_owned(),
            );
        }
        Ok(())
    }

    /// The arguments passed to the evaluation script, in order.
    pub fn script_arguments(&self) -> Vec<OsString> {
        let mut arguments = vec![OsString::from("--model"), OsString::from(&self.model)];
        if self.download {
            arguments.push(OsString::from("--download"));
        }
        if let Some(dataset) = &self.dataset {
            arguments.push(OsString::from("--dataset"));
            arguments.push(dataset.as_os_str().to_owned());
        }
        for container in &self.containers {
            arguments.push(OsString::from("--container"));
            arguments.push(container.as_os_str().to_owned());
        }
        arguments.push(OsString::from("--device"));
        arguments.push(OsString::from(&self.device));
        if let Some(windows) = self.max_windows {
            arguments.push(OsString::from("--max-windows"));
            arguments.push(OsString::from(windows.to_string()));
        }
        arguments.push(OsString::from("--report"));
        arguments.push(self.report.as_os_str().to_owned());
        arguments
    }
}

/// The interpreter to run: `--python`, then `MODELQ_PYTHON`, then `python`.
pub fn python_interpreter(flag: Option<&PathBuf>) -> OsString {
    if let Some(path) = flag {
        return path.as_os_str().to_owned();
    }
    std::env::var_os(PYTHON_ENV).unwrap_or_else(|| OsString::from(DEFAULT_PYTHON))
}

/// The evaluation script: `MODELQ_EVAL_SCRIPT`, then the source-checkout location.
pub fn locate_script() -> Result<PathBuf, String> {
    let path =
        std::env::var_os(SCRIPT_ENV).map_or_else(|| PathBuf::from(DEFAULT_SCRIPT), PathBuf::from);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "evaluation script not found at {}; run from a source checkout or set {SCRIPT_ENV}",
            path.display()
        ))
    }
}

/// Runs the script with the given interpreter and waits for it to finish.
pub fn run(options: &EvalOptions, python: &OsStr, script: &Path) -> Result<(), String> {
    options.validate()?;
    let status = Command::new(python)
        .arg(script)
        .args(options.script_arguments())
        .status()
        .map_err(|error| {
            if error.kind() == ErrorKind::NotFound {
                format!(
                    "could not find the Python interpreter {:?}; install Python 3 with PyTorch, \
                     transformers, pyarrow and huggingface_hub, or pass --python or set {PYTHON_ENV}",
                    python.to_string_lossy()
                )
            } else {
                format!("could not start {:?}: {error}", python.to_string_lossy())
            }
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("the evaluation exited with {status}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(containers: Vec<PathBuf>) -> EvalOptions {
        EvalOptions {
            model: "/models/qwen".to_owned(),
            download: false,
            dataset: Some(PathBuf::from("wikitext.parquet")),
            containers,
            device: "cpu".to_owned(),
            max_windows: None,
            report: PathBuf::from("out/report.json"),
        }
    }

    #[test]
    fn script_arguments_are_in_a_fixed_order() {
        let mut options = options(vec![
            PathBuf::from("a.safetensors"),
            PathBuf::from("b.safetensors"),
        ]);
        options.max_windows = Some(4);
        let arguments: Vec<String> = options
            .script_arguments()
            .iter()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            arguments,
            [
                "--model",
                "/models/qwen",
                "--dataset",
                "wikitext.parquet",
                "--container",
                "a.safetensors",
                "--container",
                "b.safetensors",
                "--device",
                "cpu",
                "--max-windows",
                "4",
                "--report",
                "out/report.json",
            ]
        );
    }

    #[test]
    fn download_is_forwarded_and_dataset_is_optional() {
        let mut options = options(vec![PathBuf::from("a.safetensors")]);
        options.model = "Qwen/Qwen2.5-0.5B".to_owned();
        options.download = true;
        options.dataset = None;
        let arguments = options.script_arguments();
        assert!(arguments.iter().any(|argument| argument == "--download"));
        assert!(!arguments.iter().any(|argument| argument == "--dataset"));
    }

    #[test]
    fn unknown_devices_are_rejected_before_any_work() {
        // Cargo runs tests from the package root, so this file exists.
        let mut options = options(vec![PathBuf::from("Cargo.toml")]);
        options.device = "gpu".to_owned();
        let error = options.validate().unwrap_err();
        assert!(error.contains("--device must be one of"), "{error}");
    }

    #[test]
    fn no_containers_is_an_error() {
        let error = options(vec![]).validate().unwrap_err();
        assert!(error.contains("at least one --container"), "{error}");
    }

    #[test]
    fn python_flag_wins_over_the_environment() {
        let flag = PathBuf::from("/opt/py/bin/python3");
        assert_eq!(
            python_interpreter(Some(&flag)),
            OsString::from("/opt/py/bin/python3")
        );
    }
}
