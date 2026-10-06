use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, id},
    sync::atomic::{AtomicU64, Ordering},
};

use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "modelq-int8-threads-{label}-{}-{}",
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

/// Matrices spanning many 4096-value metrics blocks, with varied magnitudes so
/// partial sums differ block to block.
fn write_source(path: &Path) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, rows, seed) in [("a.weight", 700_usize, 0.0_f32), ("b.weight", 130, 1.5)] {
        let start = data.len();
        for index in 0..rows * 64 {
            let value = ((index as f32) * 0.37 + seed).sin() * (1.0 + (index % 23) as f32)
                + (index % 5) as f32 * 1e-3;
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
        .args(["--format", "int8"])
        .args(extra)
        .arg("--output")
        .arg(output)
        .output()
        .unwrap()
}

/// The deterministic part of the report: everything from the progress lines
/// and the metrics block, without paths or the execution line.
fn report_lines(output: &Output) -> Vec<String> {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| {
            line.starts_with("Progress:")
                || line.trim_start().starts_with("Max ")
                || line.trim_start().starts_with("Lowest SQNR")
                || line.trim_start().starts_with("Saturated")
        })
        .map(str::to_owned)
        .collect()
}

#[test]
fn int8_output_and_reported_metrics_do_not_depend_on_the_thread_count() {
    let dir = TestDir::new("identical");
    let source = dir.join("source.safetensors");
    write_source(&source);

    let sequential = dir.join("sequential.safetensors");
    let result = run(&["--threads", "1"], &source, &sequential);
    assert!(result.status.success(), "{result:?}");
    assert!(String::from_utf8_lossy(&result.stdout).contains("Execution: sequential"));
    let expected_bytes = fs::read(&sequential).unwrap();
    let expected_report = report_lines(&result);
    assert!(expected_report.len() >= 8, "{expected_report:?}");

    for extra in [&["--threads", "2"][..], &["--threads", "8"][..], &[][..]] {
        let output = dir.join(&format!("out-{}.safetensors", extra.join("")));
        let result = run(extra, &source, &output);
        assert!(result.status.success(), "{extra:?}: {result:?}");
        assert!(
            String::from_utf8_lossy(&result.stdout).contains("Execution: parallel"),
            "{extra:?}"
        );
        assert_eq!(fs::read(&output).unwrap(), expected_bytes, "{extra:?}");
        assert_eq!(report_lines(&result), expected_report, "{extra:?}");
    }
}

#[test]
fn sharded_int8_output_is_thread_independent() {
    let dir = TestDir::new("sharded");
    let source = dir.join("source.safetensors");
    write_source(&source);

    let one = dir.join("one");
    let many = dir.join("many");
    assert!(
        run(
            &["--threads", "1", "--max-shard-size", "40000"],
            &source,
            &one
        )
        .status
        .success()
    );
    assert!(
        run(
            &["--threads", "6", "--max-shard-size", "40000"],
            &source,
            &many
        )
        .status
        .success()
    );
    let names = |directory: &Path| {
        let mut names: Vec<_> = fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        names
    };
    assert_eq!(names(&one), names(&many));
    for name in names(&one) {
        assert_eq!(
            fs::read(one.join(&name)).unwrap(),
            fs::read(many.join(&name)).unwrap(),
            "{name:?}"
        );
    }
}

#[test]
fn bad_thread_counts_fail_before_writing() {
    let dir = TestDir::new("reject");
    let source = dir.join("source.safetensors");
    write_source(&source);
    let output = dir.join("out.safetensors");

    assert!(!run(&["--threads", "0"], &source, &output).status.success());
    assert!(
        !run(&["--threads", "lots"], &source, &output)
            .status
            .success()
    );
    assert!(!output.exists());
}

#[test]
fn a_non_finite_input_still_fails_without_output_in_parallel() {
    let dir = TestDir::new("nan");
    let source = dir.join("source.safetensors");
    write_source(&source);
    // Overwrite one f32 inside the first matrix with NaN.
    let mut bytes = fs::read(&source).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let at = 8 + header_len + 4 * 12_345;
    bytes[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    fs::write(&source, bytes).unwrap();

    let output = dir.join("out.safetensors");
    let result = run(&["--threads", "4"], &source, &output);

    assert!(!result.status.success());
    assert!(!output.exists());
}

#[test]
fn int8_writer_output_is_identical_with_prefetching_on_small_chunks() {
    use modelq::{
        backend::cpu::ParallelConfig,
        io::{
            layout::plan_output_layout,
            safetensors::{MappedSafetensors, TensorSource},
            writer::{Int8Execution, write_safetensors, write_safetensors_with},
        },
        quant::policy::{QuantizationPolicy, TensorCandidate},
    };

    let dir = TestDir::new("writer");
    let source_path = dir.join("source.safetensors");
    write_source(&source_path);
    let source = MappedSafetensors::open(&source_path).unwrap();
    let summaries = source.tensor_summaries();
    let candidates: Vec<_> = summaries
        .iter()
        .map(|summary| {
            let elements = summary.shape.iter().product();
            if summary.dtype == "F32" {
                TensorCandidate::floating(summary.name.clone(), elements)
            } else {
                TensorCandidate::non_floating(summary.name.clone(), elements)
            }
        })
        .collect();
    let decisions = QuantizationPolicy::default().decide_all(candidates);
    let plan = plan_output_layout(&summaries, &decisions).unwrap();

    let reference = dir.join("reference.safetensors");
    write_safetensors(&source, &plan, &decisions, &reference).unwrap();
    let expected = fs::read(&reference).unwrap();

    // 4096-value chunks over 44,800 and 8,320 elements force many chunks, so
    // with more than one worker the source is read by a reader thread.
    for (workers, chunk_elements) in [(1, 4096), (2, 4096), (5, 8192), (18, 4096)] {
        let path = dir.join(&format!("out-{workers}-{chunk_elements}.safetensors"));
        write_safetensors_with(
            &source,
            &plan,
            &decisions,
            &path,
            Int8Execution::Parallel(ParallelConfig::new(workers, chunk_elements)),
        )
        .unwrap();
        assert_eq!(
            fs::read(&path).unwrap(),
            expected,
            "workers={workers} chunk_elements={chunk_elements}"
        );
    }
}
