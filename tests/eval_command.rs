//! `modelq eval` argument checks, and how it runs the evaluation script.
//!
//! The real script needs PyTorch and downloads a model, so these tests use a
//! small stand-in script that records its arguments. The tests that need a
//! Python interpreter skip themselves when none is installed.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Output, id},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let serial = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("modelq-eval-{label}-{}-{serial}", id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn modelq() -> Command {
    Command::new(env!("CARGO_BIN_EXE_modelq"))
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A local model directory with a placeholder weights file, which is all the
/// argument checks look at.
fn model_dir(dir: &TestDir) -> PathBuf {
    let model = dir.join("model");
    fs::create_dir_all(&model).unwrap();
    fs::write(model.join("model.safetensors"), b"").unwrap();
    model
}

/// A Python interpreter, if one is installed.
fn python() -> Option<String> {
    ["python3", "python"]
        .into_iter()
        .find(|candidate| {
            Command::new(candidate)
                .arg("--version")
                .output()
                .is_ok_and(|output| output.status.success())
        })
        .map(str::to_owned)
}

/// A stand-in for the evaluation script: records its arguments in the
/// `--report` file and exits with the code in `FAKE_EXIT` (default 0).
fn fake_script(dir: &TestDir) -> PathBuf {
    let script = dir.join("fake_eval.py");
    fs::write(
        &script,
        r#"import os, pathlib, sys
args = sys.argv[1:]
report = pathlib.Path(args[args.index("--report") + 1])
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text("\n".join(args), encoding="utf-8")
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
"#,
    )
    .unwrap();
    script
}

fn eval_with_fake_script(
    python: &str,
    script: &Path,
    arguments: &[&str],
    exit_code: &str,
) -> Output {
    modelq()
        .arg("eval")
        .args(arguments)
        .env("MODELQ_PYTHON", python)
        .env("MODELQ_EVAL_SCRIPT", script)
        .env("FAKE_EXIT", exit_code)
        .output()
        .unwrap()
}

#[test]
fn help_lists_the_eval_options() {
    let output = modelq().args(["eval", "--help"]).output().unwrap();
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    for option in [
        "--model",
        "--container",
        "--dataset",
        "--download",
        "--report",
        "--python",
    ] {
        assert!(help.contains(option), "help is missing {option}");
    }
}

#[test]
fn a_missing_container_is_a_usage_error() {
    let dir = TestDir::new("usage");
    let output = modelq()
        .args(["eval", "--model"])
        .arg(model_dir(&dir))
        .args(["--dataset", "wikitext.parquet", "--report", "r.json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("--container"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_missing_container_file_is_reported_before_python_runs() {
    let dir = TestDir::new("missing-container");
    let dataset = dir.join("wikitext.parquet");
    fs::write(&dataset, b"").unwrap();
    let output = modelq()
        .arg("eval")
        .arg("--model")
        .arg(model_dir(&dir))
        .arg("--dataset")
        .arg(&dataset)
        .arg("--container")
        .arg(dir.join("absent.safetensors"))
        .arg("--report")
        .arg(dir.join("r.json"))
        .env("MODELQ_PYTHON", "/nonexistent/python")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("container not found"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_local_run_without_a_dataset_is_refused() {
    let dir = TestDir::new("no-dataset");
    let container = dir.join("te.safetensors");
    fs::write(&container, b"").unwrap();
    let output = modelq()
        .arg("eval")
        .arg("--model")
        .arg(model_dir(&dir))
        .arg("--container")
        .arg(&container)
        .arg("--report")
        .arg(dir.join("r.json"))
        .env("MODELQ_PYTHON", "/nonexistent/python")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(stderr(&output).contains("--dataset"), "{}", stderr(&output));
}

#[test]
fn a_missing_interpreter_is_explained() {
    let dir = TestDir::new("no-python");
    let container = dir.join("te.safetensors");
    let dataset = dir.join("wikitext.parquet");
    fs::write(&container, b"").unwrap();
    fs::write(&dataset, b"").unwrap();
    let script = fake_script(&dir);
    let output = modelq()
        .arg("eval")
        .arg("--model")
        .arg(model_dir(&dir))
        .arg("--dataset")
        .arg(&dataset)
        .arg("--container")
        .arg(&container)
        .arg("--report")
        .arg(dir.join("r.json"))
        .arg("--python")
        .arg(dir.join("no-such-python"))
        .env("MODELQ_EVAL_SCRIPT", &script)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("could not find the Python interpreter"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn the_script_receives_the_arguments_in_order() {
    let Some(python) = python() else {
        eprintln!("skipping: no Python interpreter installed");
        return;
    };
    let dir = TestDir::new("forward");
    let container = dir.join("te.safetensors");
    let dataset = dir.join("wikitext.parquet");
    fs::write(&container, b"").unwrap();
    fs::write(&dataset, b"").unwrap();
    let model = model_dir(&dir);
    let report = dir.join("out").join("r.json");
    let model_arg = model.to_string_lossy().into_owned();
    let dataset_arg = dataset.to_string_lossy().into_owned();
    let container_arg = container.to_string_lossy().into_owned();
    let report_arg = report.to_string_lossy().into_owned();

    let output = eval_with_fake_script(
        &python,
        &fake_script(&dir),
        &[
            "--model",
            &model_arg,
            "--dataset",
            &dataset_arg,
            "--container",
            &container_arg,
            "--device",
            "cpu",
            "--max-windows",
            "3",
            "--report",
            &report_arg,
        ],
        "0",
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let recorded = fs::read_to_string(&report).unwrap();
    let lines: Vec<&str> = recorded.lines().collect();
    let expected: Vec<&str> = vec![
        "--model",
        model_arg.as_str(),
        "--dataset",
        dataset_arg.as_str(),
        "--container",
        container_arg.as_str(),
        "--device",
        "cpu",
        "--max-windows",
        "3",
        "--report",
        report_arg.as_str(),
    ];
    assert_eq!(lines, expected);
}

#[test]
fn a_failing_script_fails_the_command() {
    let Some(python) = python() else {
        eprintln!("skipping: no Python interpreter installed");
        return;
    };
    let dir = TestDir::new("failure");
    let container = dir.join("te.safetensors");
    let dataset = dir.join("wikitext.parquet");
    fs::write(&container, b"").unwrap();
    fs::write(&dataset, b"").unwrap();
    let model = model_dir(&dir);
    let model_arg = model.to_string_lossy().into_owned();
    let dataset_arg = dataset.to_string_lossy().into_owned();
    let container_arg = container.to_string_lossy().into_owned();
    let report_arg = dir.join("r.json").to_string_lossy().into_owned();

    let output = eval_with_fake_script(
        &python,
        &fake_script(&dir),
        &[
            "--model",
            &model_arg,
            "--dataset",
            &dataset_arg,
            "--container",
            &container_arg,
            "--report",
            &report_arg,
        ],
        "3",
    );
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("exited with"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_hub_model_is_refused_without_download_before_anything_runs() {
    let dir = TestDir::new("hub-no-download");
    let container = dir.join("out.safetensors");
    fs::write(&container, b"").unwrap();
    let output = modelq()
        .args(["eval", "--model", "hf:Qwen/Qwen2.5-0.5B", "--container"])
        .arg(&container)
        .args(["--report", "r.json"])
        .env("MODELQ_PYTHON", "/nonexistent/python")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("pass --download"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn an_unpinned_hub_model_needs_a_revision() {
    let dir = TestDir::new("hub-revision");
    let container = dir.join("out.safetensors");
    fs::write(&container, b"").unwrap();
    let output = modelq()
        .args([
            "eval",
            "--model",
            "hf:someone/other-model",
            "--download",
            "--container",
        ])
        .arg(&container)
        .args(["--report", "r.json"])
        .env("MODELQ_PYTHON", "/nonexistent/python")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("--revision"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_revision_is_refused_for_a_local_model_directory() {
    let dir = TestDir::new("local-revision");
    let container = dir.join("out.safetensors");
    fs::write(&container, b"").unwrap();
    fs::write(dir.join("wikitext.parquet"), b"").unwrap();
    let output = modelq()
        .args(["eval", "--model"])
        .arg(model_dir(&dir))
        .args(["--revision", "main", "--dataset"])
        .arg(dir.join("wikitext.parquet"))
        .arg("--container")
        .arg(&container)
        .args(["--report", "r.json"])
        .env("MODELQ_PYTHON", "/nonexistent/python")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("apply only to hf:"),
        "{}",
        stderr(&output)
    );
}
