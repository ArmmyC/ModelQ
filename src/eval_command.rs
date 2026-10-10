//! `modelq eval`: score a quantized output against the original model on the
//! user's own machine (ADR 0028, milestone M1; ADR 0035).
//!
//! The measurement runs in Python, in `tools/quality_eval/modelq_eval.py`,
//! which needs Python 3 with PyTorch, transformers, pyarrow and
//! huggingface_hub. This module checks the arguments that can be checked
//! without Python, fetches a `hf:` model with the Hub client `quantize` uses,
//! finds the script and the interpreter, runs them with the terminal attached,
//! and turns their exit status into a result. Nothing is downloaded without
//! `--download`. A `hf:` model is fetched only with it, and the pinned
//! WikiText-2 split only when no `--dataset` is given.

use std::{
    ffi::{OsStr, OsString},
    io::ErrorKind,
    path::{Path, PathBuf},
    process::Command,
};

use crate::hub;

/// Environment variable that overrides the location of the evaluation script.
pub const SCRIPT_ENV: &str = "MODELQ_EVAL_SCRIPT";
/// Environment variable that overrides the Python interpreter.
pub const PYTHON_ENV: &str = "MODELQ_PYTHON";
/// Location of the script in a source checkout, relative to the working directory.
pub const DEFAULT_SCRIPT: &str = "tools/quality_eval/modelq_eval.py";
/// Interpreter used when neither `--python` nor `MODELQ_PYTHON` is set.
pub const DEFAULT_PYTHON: &str = "python";
/// The Hub files an evaluation reads besides the weights.
const MODEL_EXTRAS: &[&str] = &["config.json", "tokenizer.json", "tokenizer_config.json"];
/// Models with a pinned revision, which `hf:` uses when `--revision` is omitted.
/// These are the revisions measured in ADR 0024 and ADR 0026. Keep this list in
/// step with `PINNED_MODEL` in `modelq_eval.py`.
const PINNED_MODELS: &[(&str, &str)] = &[(
    "Qwen/Qwen2.5-0.5B",
    "060db6499f32faf8b98477b0a26969ef7d8b9987",
)];

const DEVICES: [&str; 3] = ["auto", "cpu", "cuda"];

/// The arguments of `modelq eval`, as the user gave them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalOptions {
    /// A local model directory, a pinned model id (with `download`), or `hf:owner/name`.
    pub model: String,
    /// With `hf:`: the branch, tag or commit. A pinned model needs none.
    pub revision: Option<String>,
    /// With `hf:`: where downloads are cached. The default is shared with `quantize`.
    pub cache_dir: Option<PathBuf>,
    /// Allow downloads: a `hf:` model, a pinned model id, and the pinned dataset.
    pub download: bool,
    /// A local WikiText-2 test Parquet file; required unless `download` is set.
    pub dataset: Option<PathBuf>,
    /// ModelQ containers in any format `quantize` writes, to compare with the original.
    pub containers: Vec<PathBuf>,
    /// `auto`, `cpu` or `cuda`.
    pub device: String,
    /// Score only the first N windows of the dataset.
    pub max_windows: Option<u64>,
    /// Where the script writes its JSON report.
    pub report: PathBuf,
}

/// The model as the script receives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// A local model directory, or a pinned model id that the script downloads.
    pub location: OsString,
    /// The Hub repository, when the directory was fetched with `hf:`.
    pub repository: Option<String>,
    /// The commit the files came from, when the directory was fetched with `hf:`.
    pub commit: Option<String>,
}

/// The revision a pinned model uses when none is given.
pub fn pinned_revision(repository: &str) -> Option<&'static str> {
    PINNED_MODELS
        .iter()
        .find(|(name, _)| *name == repository)
        .map(|(_, revision)| *revision)
}

fn has_weights(directory: &Path) -> bool {
    directory.join("model.safetensors").is_file()
        || directory.join("model.safetensors.index.json").is_file()
}

impl EvalOptions {
    /// Checks everything that needs neither Python nor the network, so that a
    /// mistake is reported before a model is loaded or fetched.
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
        if let Some(text) = self.model.strip_prefix(hub::PREFIX) {
            if !self.download {
                return Err(format!(
                    "{} is fetched from the Hub; pass --download to allow the download",
                    self.model
                ));
            }
            let repository = hub::RepoId::parse(text)?;
            if self.revision.is_none() && pinned_revision(&repository.path()).is_none() {
                return Err(format!(
                    "{} has no pinned revision; pass --revision with the branch, tag or commit to fetch",
                    repository.path()
                ));
            }
            return Ok(());
        }
        if self.revision.is_some() || self.cache_dir.is_some() {
            return Err("--revision and --cache-dir apply only to hf: models".to_owned());
        }
        if self.download {
            return Ok(());
        }
        if !has_weights(Path::new(&self.model)) {
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
    pub fn script_arguments(&self, model: &ResolvedModel) -> Vec<OsString> {
        let mut arguments = vec![OsString::from("--model"), model.location.clone()];
        if let Some(repository) = &model.repository {
            arguments.push(OsString::from("--model-id"));
            arguments.push(OsString::from(repository));
        }
        if let Some(commit) = &model.commit {
            arguments.push(OsString::from("--model-revision"));
            arguments.push(OsString::from(commit));
        }
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

/// Fetches a `hf:` model into the cache, as `quantize` does. Any other value is
/// passed to the script unchanged.
pub fn resolve_model(options: &EvalOptions) -> Result<ResolvedModel, String> {
    let Some(text) = options.model.strip_prefix(hub::PREFIX) else {
        return Ok(ResolvedModel {
            location: OsString::from(&options.model),
            repository: None,
            commit: None,
        });
    };
    let repository = hub::RepoId::parse(text)?;
    let revision = match &options.revision {
        Some(revision) => revision.clone(),
        None => pinned_revision(&repository.path())
            .ok_or_else(|| {
                format!(
                    "{} has no pinned revision; pass --revision",
                    repository.path()
                )
            })?
            .to_owned(),
    };
    let cache_dir = match &options.cache_dir {
        Some(dir) => dir.clone(),
        None => hub::default_cache_dir()?,
    };
    let directory = hub::fetch(
        &hub::Request {
            repo: repository.clone(),
            revision,
            cache_dir,
            endpoint: hub::endpoint_from_env(),
            token: hub::token_from_env(),
        },
        MODEL_EXTRAS,
    )?;
    let commit = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("the cache path {} has no commit name", directory.display()))?
        .to_owned();
    Ok(ResolvedModel {
        location: directory.into_os_string(),
        repository: Some(repository.path()),
        commit: Some(commit),
    })
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

pub fn interpreter_error(python: &OsStr, error: &std::io::Error) -> String {
    if error.kind() == ErrorKind::NotFound {
        format!(
            "could not find the Python interpreter {:?}; install Python 3 with PyTorch, \
             transformers, pyarrow and huggingface_hub, or pass --python or set {PYTHON_ENV}",
            python.to_string_lossy()
        )
    } else {
        format!("could not start {:?}: {error}", python.to_string_lossy())
    }
}

/// Runs the script with the given interpreter and waits for it to finish. The
/// interpreter is checked before any model is fetched.
pub fn run(options: &EvalOptions, python: &OsStr, script: &Path) -> Result<(), String> {
    options.validate()?;
    Command::new(python)
        .arg("--version")
        .output()
        .map_err(|error| interpreter_error(python, &error))?;
    let model = resolve_model(options)?;
    let status = Command::new(python)
        .arg(script)
        .args(options.script_arguments(&model))
        .status()
        .map_err(|error| interpreter_error(python, &error))?;
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
            revision: None,
            cache_dir: None,
            download: false,
            dataset: Some(PathBuf::from("wikitext.parquet")),
            containers,
            device: "cpu".to_owned(),
            max_windows: None,
            report: PathBuf::from("out/report.json"),
        }
    }

    fn local(location: &str) -> ResolvedModel {
        ResolvedModel {
            location: OsString::from(location),
            repository: None,
            commit: None,
        }
    }

    fn strings(arguments: &[OsString]) -> Vec<String> {
        arguments
            .iter()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn script_arguments_are_in_a_fixed_order() {
        let mut options = options(vec![
            PathBuf::from("a.safetensors"),
            PathBuf::from("b.gguf"),
        ]);
        options.max_windows = Some(4);
        assert_eq!(
            strings(&options.script_arguments(&local("/models/qwen"))),
            [
                "--model",
                "/models/qwen",
                "--dataset",
                "wikitext.parquet",
                "--container",
                "a.safetensors",
                "--container",
                "b.gguf",
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
        options.dataset = None;
        options.download = true;
        let arguments = strings(&options.script_arguments(&local("Qwen/Qwen2.5-0.5B")));
        assert!(arguments.iter().any(|argument| argument == "--download"));
        assert!(!arguments.iter().any(|argument| argument == "--dataset"));
    }

    #[test]
    fn a_fetched_model_passes_its_repository_and_commit() {
        let options = options(vec![PathBuf::from("a.safetensors")]);
        let model = ResolvedModel {
            location: OsString::from("cache/hub/Qwen/Qwen2.5-0.5B/060db"),
            repository: Some("Qwen/Qwen2.5-0.5B".to_owned()),
            commit: Some("060db".to_owned()),
        };
        let arguments = strings(&options.script_arguments(&model));
        assert_eq!(
            arguments[..6],
            [
                "--model",
                "cache/hub/Qwen/Qwen2.5-0.5B/060db",
                "--model-id",
                "Qwen/Qwen2.5-0.5B",
                "--model-revision",
                "060db",
            ]
        );
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
    fn a_hub_model_needs_download_before_anything_is_fetched() {
        let mut options = options(vec![PathBuf::from("Cargo.toml")]);
        options.model = "hf:Qwen/Qwen2.5-0.5B".to_owned();
        options.dataset = None;
        let error = options.validate().unwrap_err();
        assert!(error.contains("pass --download"), "{error}");
    }

    #[test]
    fn a_hub_model_without_a_pinned_revision_needs_one() {
        let mut options = options(vec![PathBuf::from("Cargo.toml")]);
        options.model = "hf:someone/other-model".to_owned();
        options.download = true;
        options.dataset = None;
        let error = options.validate().unwrap_err();
        assert!(error.contains("--revision"), "{error}");
        options.revision = Some("main".to_owned());
        assert!(options.validate().is_ok());
    }

    #[test]
    fn the_pinned_revision_is_the_measured_one() {
        assert_eq!(
            pinned_revision("Qwen/Qwen2.5-0.5B"),
            Some("060db6499f32faf8b98477b0a26969ef7d8b9987")
        );
        assert_eq!(pinned_revision("someone/other-model"), None);
    }

    #[test]
    fn revision_and_cache_dir_need_a_hub_model() {
        let mut options = options(vec![PathBuf::from("Cargo.toml")]);
        options.dataset = None;
        options.download = true;
        options.revision = Some("main".to_owned());
        let error = options.validate().unwrap_err();
        assert!(error.contains("apply only to hf: models"), "{error}");
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
