use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, id},
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::{
    backend::cpu::ParallelConfig,
    io::{
        nvfp4::{
            Nvfp4Execution, Nvfp4WriterError, plan_nvfp4_output, write_nvfp4_safetensors,
            write_nvfp4_safetensors_with,
        },
        safetensors::{MappedSafetensors, TensorSource},
    },
};
use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "modelq-nvfp4-threads-{label}-{}-{}",
            id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("test directory is created");
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

/// Matrices large enough to give every worker several blocks, with varied
/// magnitudes so block scales differ across ranges.
fn write_source(path: &Path) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, rows, seed) in [("a.weight", 300_usize, 0.0_f32), ("b.weight", 128, 1.0)] {
        let start = data.len();
        for index in 0..rows * 64 {
            let value = ((index as f32) * 0.37 + seed).sin() * (1.0 + (index % 23) as f32);
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.to_owned(),
            json!({ "dtype": "F32", "shape": [rows, 64], "data_offsets": [start, data.len()] }),
        );
    }
    let start = data.len();
    data.extend_from_slice(&[1, 2, 3]);
    header.insert(
        "ids".to_owned(),
        json!({ "dtype": "U8", "shape": [3], "data_offsets": [start, data.len()] }),
    );
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes).unwrap();
}

fn run(extra: &[&str], input: &Path, output: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("quantize")
        .arg(input)
        .args(["--format", "nvfp4"])
        .args(extra)
        .arg("--output")
        .arg(output)
        .output()
        .unwrap()
}

#[test]
fn writer_output_is_identical_for_every_execution_mode() {
    let dir = TestDir::new("writer");
    let source_path = dir.join("source.safetensors");
    write_source(&source_path);
    let source = MappedSafetensors::open(&source_path).unwrap();
    let summaries = source.tensor_summaries();
    let plan = plan_nvfp4_output(&summaries, &["a.weight".to_owned(), "b.weight".to_owned()])
        .expect("plan");

    let reference = dir.join("reference.safetensors");
    write_nvfp4_safetensors(&source, &plan, &reference).unwrap();
    let expected = fs::read(&reference).unwrap();

    // Tiny chunks force many chunk boundaries; many workers exceed the work.
    for (workers, chunk_elements) in [
        (1, 16),
        (2, 64),
        (3, 1024),
        (7, 4096),
        (18, 1 << 22),
        (500, 64),
    ] {
        let path = dir.join(&format!("parallel-{workers}-{chunk_elements}.safetensors"));
        write_nvfp4_safetensors_with(
            &source,
            &plan,
            &path,
            Nvfp4Execution::Parallel(ParallelConfig::new(workers, chunk_elements)),
        )
        .unwrap();
        assert_eq!(
            fs::read(&path).unwrap(),
            expected,
            "workers={workers} chunk_elements={chunk_elements}"
        );
    }
}

#[test]
fn invalid_parallel_configuration_fails_without_creating_output() {
    let dir = TestDir::new("config");
    let source_path = dir.join("source.safetensors");
    write_source(&source_path);
    let source = MappedSafetensors::open(&source_path).unwrap();
    let plan = plan_nvfp4_output(&source.tensor_summaries(), &["a.weight".to_owned()]).unwrap();
    let output = dir.join("out.safetensors");

    let error = write_nvfp4_safetensors_with(
        &source,
        &plan,
        &output,
        Nvfp4Execution::Parallel(ParallelConfig::new(0, 1024)),
    )
    .expect_err("zero workers is invalid");

    assert!(
        matches!(error, Nvfp4WriterError::Execution { .. }),
        "{error:?}"
    );
    assert!(!output.exists());
}

#[test]
fn cli_output_does_not_depend_on_the_thread_count() {
    let dir = TestDir::new("cli");
    let source = dir.join("source.safetensors");
    write_source(&source);

    let sequential = dir.join("sequential.safetensors");
    let result = run(&["--threads", "1"], &source, &sequential);
    assert!(result.status.success(), "{result:?}");
    assert!(String::from_utf8_lossy(&result.stdout).contains("Execution: sequential"));

    let expected = fs::read(&sequential).unwrap();
    for extra in [&["--threads", "2"][..], &["--threads", "8"][..], &[][..]] {
        let output = dir.join(&format!("out-{}.safetensors", extra.join("")));
        let result = run(extra, &source, &output);
        assert!(result.status.success(), "{extra:?}: {result:?}");
        assert!(
            String::from_utf8_lossy(&result.stdout).contains("Execution: parallel"),
            "{extra:?}"
        );
        assert_eq!(fs::read(&output).unwrap(), expected, "{extra:?}");
    }
}

#[test]
fn cli_rejects_bad_thread_options_before_writing() {
    let dir = TestDir::new("cli-reject");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output = dir.join("out.safetensors");

    assert!(!run(&["--threads", "0"], &source, &output).status.success());
    assert!(
        !run(&["--threads", "many"], &source, &output)
            .status
            .success()
    );
    assert!(!output.exists());

    let int8 = Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("quantize")
        .arg(&source)
        .args(["--format", "int8", "--threads", "4", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(!int8.status.success());
    assert!(!output.exists());
}
