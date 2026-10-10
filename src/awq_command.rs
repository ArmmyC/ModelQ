//! `modelq quantize --calibration awq`: activation-aware calibration before the
//! INT4 writer (ADR 0037).
//!
//! The calibration search is Python, in `tools/calibration/modelq_awq.py`. It
//! needs the same environment as `modelq eval`, and it writes a rescaled,
//! unquantized checkpoint and a JSON report. This module checks the arguments
//! that need no Python, runs the script with the terminal attached, and returns
//! the checkpoint and the report. Both are temporary files next to the output,
//! and they are removed when the returned value is dropped.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::eval_command::{interpreter_error, python_interpreter};

/// Environment variable that overrides the location of the calibration script.
pub const SCRIPT_ENV: &str = "MODELQ_AWQ_SCRIPT";
/// Location of the script in a source checkout, relative to the working directory.
pub const DEFAULT_SCRIPT: &str = "tools/calibration/modelq_awq.py";
/// The transform recorded in the output as `modelq.transform`.
pub const TRANSFORM: &str = "awq-v1";

const DEVICES: [&str; 3] = ["auto", "cpu", "cuda"];

/// What the user asked for.
pub struct Request<'a> {
    /// A local model directory with one `model.safetensors`.
    pub model_dir: &'a Path,
    /// A local WikiText-2 train Parquet file.
    pub calibration_data: Option<&'a Path>,
    /// Fetch the pinned WikiText-2 train split instead.
    pub download: bool,
    /// The group size the rescaling is searched for.
    pub group_size: usize,
    /// `auto`, `cpu` or `cuda`.
    pub device: &'a str,
    /// The Python interpreter, when `--python` was given.
    pub python: Option<&'a PathBuf>,
    /// Where the quantized output will be written. It must not exist yet.
    pub output: &'a Path,
}

/// The temporary checkpoint and report. Both are removed when this value is dropped.
pub struct Calibrated {
    checkpoint: PathBuf,
    report: PathBuf,
}

impl Calibrated {
    /// The rescaled, unquantized checkpoint.
    pub fn checkpoint(&self) -> &Path {
        &self.checkpoint
    }

    /// The calibration report, as compact JSON.
    pub fn report_json(&self) -> Result<String, String> {
        let text = std::fs::read_to_string(&self.report)
            .map_err(|error| format!("could not read the calibration report: {error}"))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| format!("the calibration report is not valid JSON: {error}"))?;
        Ok(value.to_string())
    }
}

impl Drop for Calibrated {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.checkpoint);
        let _ = std::fs::remove_file(&self.report);
    }
}

/// Checks the request before any Python runs, so that a mistake costs nothing.
pub fn validate(request: &Request<'_>) -> Result<(), String> {
    if !DEVICES.contains(&request.device) {
        return Err(format!(
            "--device must be one of {} for --calibration, not {:?}",
            DEVICES.join(", "),
            request.device
        ));
    }
    match (request.calibration_data, request.download) {
        (Some(_), true) => {
            return Err("pass either --calibration-data or --download, not both".to_owned());
        }
        (None, false) => {
            return Err(
                "--calibration awq needs --calibration-data with a local WikiText-2 train Parquet file, or --download for the pinned split"
                    .to_owned(),
            );
        }
        (Some(path), false) if !path.is_file() => {
            return Err(format!("calibration data not found: {}", path.display()));
        }
        _ => {}
    }
    if !request.model_dir.join("model.safetensors").is_file() {
        return Err(format!(
            "{} has no model.safetensors; calibration reads one checkpoint file",
            request.model_dir.display()
        ));
    }
    if request.output.exists() {
        return Err(format!(
            "{} already exists; an existing output is never replaced",
            request.output.display()
        ));
    }
    Ok(())
}

/// The calibration script: `MODELQ_AWQ_SCRIPT`, then the source-checkout location.
pub fn locate_script() -> Result<PathBuf, String> {
    let path =
        std::env::var_os(SCRIPT_ENV).map_or_else(|| PathBuf::from(DEFAULT_SCRIPT), PathBuf::from);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "calibration script not found at {}; run from a source checkout or set {SCRIPT_ENV}",
            path.display()
        ))
    }
}

/// Temporary checkpoint and report names next to the output, unique to this run.
fn scratch_paths(output: &Path) -> (PathBuf, PathBuf) {
    let directory = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let stem = format!(".modelq-awq-{}-{stamp}", std::process::id());
    (
        directory.join(format!("{stem}.safetensors")),
        directory.join(format!("{stem}.json")),
    )
}

/// Runs the calibration and returns its checkpoint and report.
pub fn calibrate(request: &Request<'_>) -> Result<Calibrated, String> {
    validate(request)?;
    let script = locate_script()?;
    let python = python_interpreter(request.python);
    let (checkpoint, report) = scratch_paths(request.output);
    // Created before the run, so that a run which fails part way also cleans up.
    let calibrated = Calibrated {
        checkpoint: checkpoint.clone(),
        report: report.clone(),
    };
    let mut arguments: Vec<OsString> = vec![
        script.into_os_string(),
        OsString::from("--model"),
        request.model_dir.as_os_str().to_owned(),
        OsString::from("--output"),
        checkpoint.into_os_string(),
        OsString::from("--report"),
        report.into_os_string(),
        OsString::from("--group-size"),
        OsString::from(request.group_size.to_string()),
        OsString::from("--device"),
        OsString::from(request.device),
    ];
    if let Some(data) = request.calibration_data {
        arguments.push(OsString::from("--calibration-data"));
        arguments.push(data.as_os_str().to_owned());
    }
    if request.download {
        arguments.push(OsString::from("--download"));
    }
    let status = Command::new(&python)
        .args(&arguments)
        .status()
        .map_err(|error| interpreter_error(&python, &error))?;
    if !status.success() {
        return Err(format!("the calibration exited with {status}"));
    }
    Ok(calibrated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(
        calibration_data: Option<&'static Path>,
        download: bool,
        device: &'static str,
    ) -> Request<'static> {
        Request {
            model_dir: Path::new("model"),
            calibration_data,
            download,
            group_size: 128,
            device,
            python: None,
            output: Path::new("out.safetensors"),
        }
    }

    #[test]
    fn scratch_files_sit_next_to_the_output_and_differ() {
        let (checkpoint, report) = scratch_paths(Path::new("out/model.safetensors"));
        assert_eq!(checkpoint.parent(), Some(Path::new("out")));
        assert_eq!(report.parent(), Some(Path::new("out")));
        assert_ne!(checkpoint, report);
    }

    #[test]
    fn a_bare_output_name_uses_the_working_directory() {
        let (checkpoint, _) = scratch_paths(Path::new("model.safetensors"));
        assert_eq!(checkpoint.parent(), Some(Path::new(".")));
    }

    #[test]
    fn calibration_needs_data_or_download_but_not_both() {
        let neither = validate(&request(None, false, "cpu")).unwrap_err();
        assert!(neither.contains("--calibration-data"), "{neither}");
        let both = validate(&request(Some(Path::new("Cargo.toml")), true, "cpu")).unwrap_err();
        assert!(both.contains("not both"), "{both}");
    }

    #[test]
    fn an_unknown_device_is_refused_before_anything_else() {
        let error = validate(&request(None, true, "gpu")).unwrap_err();
        assert!(error.contains("--device must be one of"), "{error}");
    }

    #[test]
    fn missing_calibration_data_is_refused() {
        let error = validate(&request(
            Some(Path::new("no-such-file.parquet")),
            false,
            "cpu",
        ))
        .unwrap_err();
        assert!(error.contains("calibration data not found"), "{error}");
    }
}
