//! `modelq quantize --format int4|int3|int2|int1` and `modelq formats` (ADR 0030).

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Output, id},
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::{lowbit::read_manifest, safetensors::MappedSafetensors};
use serde_json::{Map, Value, json};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let serial = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("modelq-lowbit-cli-{label}-{}-{serial}", id()));
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

fn values(count: usize, seed: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as f32) * 0.173 + seed).sin())
        .collect()
}

/// Two quantizable matrices and a preserved vector.
fn write_source(path: &Path) {
    let tensors = [
        (
            "model.layers.0.weight",
            vec![64_usize, 128],
            values(64 * 128, 1.0),
        ),
        (
            "model.layers.1.weight",
            vec![32, 256],
            values(32 * 256, 2.0),
        ),
        ("model.norm", vec![128], values(128, 3.0)),
    ];
    let mut header = Map::new();
    let mut data = Vec::new();
    for (name, shape, tensor_values) in &tensors {
        let begin = data.len();
        for value in tensor_values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            (*name).to_owned(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}),
        );
    }
    let mut json = serde_json::to_vec(&Value::Object(header)).unwrap();
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&json);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes).unwrap();
}

fn quantize(source: &Path, arguments: &[&str], output: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("quantize")
        .arg(source)
        .args(arguments)
        .arg("--output")
        .arg(output)
        .output()
        .unwrap()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn formats_lists_every_format_with_its_status() {
    let output = Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("formats")
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    for expected in [
        "int8",
        "representation-valid",
        "int1",
        "experimental",
        "--experimental",
        "nvfp4-te",
        "hardware-validated: Transformer Engine 2.19.0 on NVIDIA B200",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
}

#[test]
fn experimental_formats_need_the_flag() {
    let dir = TestDir::new("gate");
    let source = dir.join("source.safetensors");
    write_source(&source);
    for format in ["int3", "int2", "int1"] {
        let output_path = dir.join(&format!("{format}.safetensors"));
        let result = quantize(&source, &["--format", format], &output_path);
        assert!(
            !result.status.success(),
            "{format} must be refused without the flag"
        );
        assert!(
            stderr(&result).contains("--experimental"),
            "{}",
            stderr(&result)
        );
        assert!(
            !output_path.exists(),
            "nothing is written when the flag is missing"
        );
    }
}

#[test]
fn experimental_formats_write_with_the_flag() {
    let dir = TestDir::new("flag");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output_path = dir.join("int2.safetensors");
    let result = quantize(
        &source,
        &["--format", "int2", "--experimental"],
        &output_path,
    );
    assert!(
        result.status.success(),
        "{}{}",
        stdout(&result),
        stderr(&result)
    );
    assert!(
        stdout(&result).contains("Format: int2"),
        "{}",
        stdout(&result)
    );
    assert!(
        stdout(&result).contains("Validation: passed"),
        "{}",
        stdout(&result)
    );
}

#[test]
fn int4_uses_group_size_128_by_default() {
    let dir = TestDir::new("default");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output_path = dir.join("int4.safetensors");
    let result = quantize(&source, &["--format", "int4"], &output_path);
    assert!(
        result.status.success(),
        "{}{}",
        stdout(&result),
        stderr(&result)
    );
    let reader = MappedSafetensors::open(&output_path).unwrap();
    assert_eq!(reader.metadata()["modelq.group_size"], "128");
    assert_eq!(reader.metadata()["modelq.quantization"], "int4");
    assert!(stdout(&result).contains("Compatibility: ModelQ-native representation"));
}

#[test]
fn group_size_is_recorded_and_validated() {
    let dir = TestDir::new("group");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output_path = dir.join("int4-64.safetensors");
    let result = quantize(
        &source,
        &["--format", "int4", "--group-size", "64"],
        &output_path,
    );
    assert!(
        result.status.success(),
        "{}{}",
        stdout(&result),
        stderr(&result)
    );
    let reader = MappedSafetensors::open(&output_path).unwrap();
    let manifest = read_manifest(reader.metadata()).unwrap().unwrap();
    assert_eq!(manifest.config.group_size, 64);
    assert!(
        stdout(&result).contains("Validation: passed"),
        "{}",
        stdout(&result)
    );
}

#[test]
fn the_group_size_option_is_refused_for_int8() {
    let dir = TestDir::new("int8-group");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output_path = dir.join("int8.safetensors");
    let result = quantize(
        &source,
        &["--format", "int8", "--group-size", "64"],
        &output_path,
    );
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("--group-size"),
        "{}",
        stderr(&result)
    );
    assert!(!output_path.exists());
}

#[test]
fn zero_group_size_is_a_usage_error() {
    let dir = TestDir::new("zero");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output_path = dir.join("out.safetensors");
    let result = quantize(
        &source,
        &["--format", "int4", "--group-size", "0"],
        &output_path,
    );
    assert!(!result.status.success());
    assert!(!output_path.exists());
}

#[test]
fn an_unknown_format_names_the_supported_ones() {
    let dir = TestDir::new("unknown");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output_path = dir.join("out.safetensors");
    let result = quantize(&source, &["--format", "int16"], &output_path);
    assert!(!result.status.success());
    assert!(stderr(&result).contains("int4"), "{}", stderr(&result));
    assert!(!output_path.exists());
}

#[test]
fn the_one_bit_sign_format_writes_its_code_map() {
    let dir = TestDir::new("sign");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output_path = dir.join("int1.safetensors");
    let result = quantize(
        &source,
        &["--format", "int1", "--experimental"],
        &output_path,
    );
    assert!(
        result.status.success(),
        "{}{}",
        stdout(&result),
        stderr(&result)
    );
    let reader = MappedSafetensors::open(&output_path).unwrap();
    assert_eq!(reader.metadata()["modelq.scheme"], "sign-group-wise");
    assert_eq!(reader.metadata()["modelq.code_map"], "1=+scale,0=-scale");
}

#[test]
fn sharded_output_is_validated_shard_by_shard() {
    let dir = TestDir::new("sharded");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output_dir = dir.join("shards");
    let result = quantize(
        &source,
        &[
            "--format",
            "int4",
            "--group-size",
            "32",
            "--max-shard-size",
            "8000",
        ],
        &output_dir,
    );
    assert!(
        result.status.success(),
        "{}{}",
        stdout(&result),
        stderr(&result)
    );
    assert!(
        stdout(&result).contains("Validation: passed"),
        "{}",
        stdout(&result)
    );
    let shards: Vec<PathBuf> = fs::read_dir(&output_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert!(
        shards.len() >= 2,
        "the output is split into shards: {shards:?}"
    );
}

#[test]
fn the_same_command_gives_the_same_bytes() {
    let dir = TestDir::new("repeat");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let first = dir.join("first.safetensors");
    let second = dir.join("second.safetensors");
    assert!(
        quantize(&source, &["--format", "int3", "--experimental"], &first)
            .status
            .success()
    );
    assert!(
        quantize(&source, &["--format", "int3", "--experimental"], &second)
            .status
            .success()
    );
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
}

#[test]
fn an_existing_output_is_not_replaced() {
    let dir = TestDir::new("exists");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output_path = dir.join("out.safetensors");
    fs::write(&output_path, b"previous").unwrap();
    let result = quantize(&source, &["--format", "int4"], &output_path);
    assert!(!result.status.success());
    assert_eq!(fs::read(&output_path).unwrap(), b"previous");
}

/// A SafeTensors source with the named F32 tensors a test needs.
fn write_named_source(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    let mut header = Map::new();
    let mut data = Vec::new();
    for (name, shape, tensor_values) in tensors {
        let begin = data.len();
        for value in tensor_values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            (*name).to_owned(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}),
        );
    }
    let mut json = serde_json::to_vec(&Value::Object(header)).unwrap();
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&json);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes).unwrap();
}

#[test]
fn int4_keeps_the_vocabulary_matrices_at_source_precision_by_default() {
    let dir = TestDir::new("vocabulary");
    let source = dir.join("source.safetensors");
    write_named_source(
        &source,
        &[
            (
                "model.embed_tokens.weight",
                vec![64, 128],
                values(64 * 128, 4.0),
            ),
            (
                "model.layers.0.weight",
                vec![64, 128],
                values(64 * 128, 5.0),
            ),
        ],
    );
    let output_path = dir.join("int4.safetensors");
    let result = quantize(&source, &["--format", "int4"], &output_path);
    assert!(
        result.status.success(),
        "{}{}",
        stdout(&result),
        stderr(&result)
    );
    assert!(
        stdout(&result).contains("1 quantized, 1 preserved"),
        "{}",
        stdout(&result)
    );
    let reader = MappedSafetensors::open(&output_path).unwrap();
    let manifest: Value = serde_json::from_str(&reader.metadata()["modelq.manifest"]).unwrap();
    assert_eq!(
        manifest["tensors"]["model.embed_tokens.weight"]["action"],
        "preserved"
    );
    assert_eq!(
        manifest["tensors"]["model.layers.0.weight"]["action"],
        "quantized"
    );
}

/// A model directory holding the test source as its one checkpoint file.
fn model_directory(dir: &TestDir) -> PathBuf {
    let model = dir.join("model");
    fs::create_dir_all(&model).unwrap();
    write_source(&model.join("model.safetensors"));
    model
}

/// Runs the quantizer with `MODELQ_PYTHON` pointing nowhere, so that a refusal
/// is shown to come before any Python is needed.
fn quantize_without_python(source: &Path, arguments: &[&str], output: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("quantize")
        .arg(source)
        .args(arguments)
        .arg("--output")
        .arg(output)
        .env("MODELQ_PYTHON", "/nonexistent/python")
        .output()
        .unwrap()
}

#[test]
fn calibration_is_refused_for_formats_other_than_int4() {
    let dir = TestDir::new("calibration-format");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let result = quantize_without_python(
        &source,
        &["--format", "int8", "--calibration", "awq"],
        &dir.join("out.safetensors"),
    );
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("applies only to --format int4"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn calibration_needs_data_or_download_before_python_runs() {
    let dir = TestDir::new("calibration-data");
    let model = model_directory(&dir);
    let result = quantize_without_python(
        &model,
        &["--format", "int4", "--calibration", "awq"],
        &dir.join("out.safetensors"),
    );
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("--calibration-data"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn missing_calibration_data_is_refused_before_python_runs() {
    let dir = TestDir::new("calibration-missing");
    let model = model_directory(&dir);
    let data = dir.join("train.parquet");
    let data_arg = data.to_string_lossy().into_owned();
    let result = quantize_without_python(
        &model,
        &[
            "--format",
            "int4",
            "--calibration",
            "awq",
            "--calibration-data",
            &data_arg,
        ],
        &dir.join("out.safetensors"),
    );
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("calibration data not found"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn an_existing_output_is_refused_before_calibration_starts() {
    let dir = TestDir::new("calibration-exists");
    let model = model_directory(&dir);
    let data = dir.join("train.parquet");
    fs::write(&data, b"").unwrap();
    let output = dir.join("out.safetensors");
    fs::write(&output, b"previous artifact").unwrap();
    let data_arg = data.to_string_lossy().into_owned();
    let result = quantize_without_python(
        &model,
        &[
            "--format",
            "int4",
            "--calibration",
            "awq",
            "--calibration-data",
            &data_arg,
        ],
        &output,
    );
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("already exists"),
        "{}",
        stderr(&result)
    );
    assert_eq!(fs::read(&output).unwrap(), b"previous artifact");
}

#[test]
fn download_applies_only_with_calibration() {
    let dir = TestDir::new("calibration-only");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let result = quantize_without_python(
        &source,
        &["--format", "int4", "--download"],
        &dir.join("out.safetensors"),
    );
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("apply only with --calibration"),
        "{}",
        stderr(&result)
    );
}
