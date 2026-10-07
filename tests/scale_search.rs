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
            Nvfp4Execution, Nvfp4Settings, plan_nvfp4_output, read_nvfp4_safetensors,
            write_nvfp4_safetensors, write_nvfp4_safetensors_settings,
        },
        safetensors::{MappedSafetensors, TensorSource},
        te_container::{
            plan_te_output, read_te_container_manifest, te_matrix_values,
            write_te_nvfp4_safetensors, write_te_nvfp4_safetensors_settings,
        },
    },
    quant::nvfp4::ScaleSelection,
};
use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "modelq-scale-search-{label}-{}-{}",
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

/// Weights with the spread of a trained layer: a heavy-tailed mix so blocks
/// differ in how well the reference scale suits them.
fn weights(rows: usize, columns: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..rows * columns)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0;
            let tail = if index % 97 == 0 { 6.0 } else { 1.0 };
            unit * tail * 0.05
        })
        .collect()
}

fn write_source(path: &Path, tensors: &[(&str, usize, usize, u64)]) -> Vec<Vec<f32>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut all = Vec::new();
    for (name, rows, columns, seed) in tensors {
        let values = weights(*rows, *columns, *seed);
        let start = data.len();
        for value in &values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            (*name).to_owned(),
            json!({ "dtype": "F32", "shape": [rows, columns], "data_offsets": [start, data.len()] }),
        );
        all.push(values);
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes).unwrap();
    all
}

const SEARCH: ScaleSelection = ScaleSelection::MinMse { radius: 6 };

fn settings(execution: Nvfp4Execution, scales: ScaleSelection) -> Nvfp4Settings {
    Nvfp4Settings { execution, scales }
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / a.len() as f64
}

#[test]
fn the_native_writer_records_and_reads_a_searched_file() {
    let dir = TestDir::new("native");
    let source_path = dir.join("source.safetensors");
    let originals = write_source(&source_path, &[("w", 64, 128, 1)]);
    let source = MappedSafetensors::open(&source_path).unwrap();
    let plan = plan_nvfp4_output(&source.tensor_summaries(), &["w".to_owned()]).unwrap();

    let default = dir.join("default.safetensors");
    write_nvfp4_safetensors(&source, &plan, &default).unwrap();
    let amax = dir.join("amax.safetensors");
    write_nvfp4_safetensors_settings(
        &source,
        &plan,
        &amax,
        settings(Nvfp4Execution::Sequential, ScaleSelection::Amax),
    )
    .unwrap();
    let radius_zero = dir.join("radius0.safetensors");
    write_nvfp4_safetensors_settings(
        &source,
        &plan,
        &radius_zero,
        settings(
            Nvfp4Execution::Sequential,
            ScaleSelection::MinMse { radius: 0 },
        ),
    )
    .unwrap();
    // Without a search the output is exactly what it always was.
    assert_eq!(fs::read(&default).unwrap(), fs::read(&amax).unwrap());
    assert_eq!(fs::read(&default).unwrap(), fs::read(&radius_zero).unwrap());

    let searched = dir.join("searched.safetensors");
    write_nvfp4_safetensors_settings(
        &source,
        &plan,
        &searched,
        settings(Nvfp4Execution::Sequential, SEARCH),
    )
    .unwrap();
    let reader = MappedSafetensors::open(&searched).unwrap();
    assert_eq!(
        reader
            .metadata()
            .get("modelq.scale_selection")
            .map(String::as_str),
        Some("min-mse:r6")
    );
    assert!(
        MappedSafetensors::open(&default)
            .unwrap()
            .metadata()
            .get("modelq.scale_selection")
            .is_none(),
        "the default output carries no extra key"
    );

    // The reader accepts it, and the searched tensor is closer to the source.
    let decode = |path: &Path| {
        let file = MappedSafetensors::open(path).unwrap();
        read_nvfp4_safetensors(&file).unwrap().remove(0).values
    };
    let (reference_error, searched_error) = (
        mse(&originals[0], &decode(&default)),
        mse(&originals[0], &decode(&searched)),
    );
    assert!(
        searched_error < reference_error,
        "{searched_error} !< {reference_error}"
    );

    // Execution mode never changes the bytes.
    let expected = fs::read(&searched).unwrap();
    for (workers, chunk) in [(2, 64), (5, 1024), (18, 1 << 21)] {
        let path = dir.join(&format!("parallel-{workers}-{chunk}.safetensors"));
        write_nvfp4_safetensors_settings(
            &source,
            &plan,
            &path,
            settings(
                Nvfp4Execution::Parallel(ParallelConfig::new(workers, chunk)),
                SEARCH,
            ),
        )
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), expected, "{workers}/{chunk}");
    }
}

#[test]
fn the_transformer_engine_writer_records_the_selection_and_stays_valid() {
    let dir = TestDir::new("te");
    let source_path = dir.join("source.safetensors");
    let originals = write_source(&source_path, &[("a", 64, 128, 2), ("b", 48, 96, 3)]);
    let source = MappedSafetensors::open(&source_path).unwrap();
    let plan = plan_te_output(
        &source.tensor_summaries(),
        &["a".to_owned(), "b".to_owned()],
    )
    .unwrap();

    let default = dir.join("default.safetensors");
    write_te_nvfp4_safetensors(&source, &plan, &default).unwrap();
    let radius_zero = dir.join("radius0.safetensors");
    write_te_nvfp4_safetensors_settings(
        &source,
        &plan,
        &radius_zero,
        settings(
            Nvfp4Execution::Sequential,
            ScaleSelection::MinMse { radius: 0 },
        ),
    )
    .unwrap();
    assert_eq!(fs::read(&default).unwrap(), fs::read(&radius_zero).unwrap());

    let searched = dir.join("searched.safetensors");
    write_te_nvfp4_safetensors_settings(
        &source,
        &plan,
        &searched,
        settings(Nvfp4Execution::Sequential, SEARCH),
    )
    .unwrap();
    let container = MappedSafetensors::open(&searched).unwrap();
    let manifest_text = container.metadata().get("modelq.manifest").unwrap();
    let manifest: Value = serde_json::from_str(manifest_text).unwrap();
    assert_eq!(manifest["encoder"]["scale_selection"], "min-mse:r6");
    let default_manifest: Value = serde_json::from_str(
        MappedSafetensors::open(&default)
            .unwrap()
            .metadata()
            .get("modelq.manifest")
            .unwrap(),
    )
    .unwrap();
    assert!(default_manifest.get("encoder").is_none());
    // The validators and decoder do not care how the scales were chosen.
    read_te_container_manifest(&container).expect("the manifest is valid");

    let decode = |path: &Path, name: &str, rows: usize, columns: usize| {
        let file = MappedSafetensors::open(path).unwrap();
        let bytes = |suffix: &str| {
            file.tensor_bytes(&format!("{name}.{suffix}"))
                .unwrap()
                .to_vec()
        };
        let amax = f32::from_le_bytes(bytes("amax_rowwise")[..].try_into().unwrap());
        te_matrix_values(
            rows,
            columns,
            &bytes("rowwise_data"),
            &bytes("rowwise_scale_inv"),
            amax,
        )
        .unwrap()
        .collect::<Vec<f32>>()
    };
    for (index, (name, rows, columns)) in [("a", 64, 128), ("b", 48, 96)].into_iter().enumerate() {
        let better = mse(&originals[index], &decode(&searched, name, rows, columns));
        let worse = mse(&originals[index], &decode(&default, name, rows, columns));
        assert!(better < worse, "{name}: {better} !< {worse}");
    }

    let expected = fs::read(&searched).unwrap();
    for (workers, chunk) in [(3, 128), (8, 4096)] {
        let path = dir.join(&format!("parallel-{workers}-{chunk}.safetensors"));
        write_te_nvfp4_safetensors_settings(
            &source,
            &plan,
            &path,
            settings(
                Nvfp4Execution::Parallel(ParallelConfig::new(workers, chunk)),
                SEARCH,
            ),
        )
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), expected, "{workers}/{chunk}");
    }
}

fn run(format: &str, extra: &[&str], input: &Path, output: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("quantize")
        .arg(input)
        .args(["--format", format])
        .args(extra)
        .arg("--output")
        .arg(output)
        .output()
        .unwrap()
}

fn report_value(output: &Output, label: &str) -> f64 {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.trim().strip_prefix(label))
        .and_then(|rest| rest.trim().parse().ok())
        .unwrap_or_else(|| panic!("no {label} in the report"))
}

#[test]
fn the_cli_flag_works_for_both_formats_and_is_deterministic() {
    let dir = TestDir::new("cli");
    let source = dir.join("source.safetensors");
    write_source(
        &source,
        &[("a.weight", 128, 256, 4), ("b.weight", 96, 128, 5)],
    );

    for format in ["nvfp4", "nvfp4-te"] {
        // Radius 0 is the reference rule and must equal the library's output.
        let reference = dir.join(&format!("{format}-reference.safetensors"));
        let reference_run = run(format, &["--scale-search", "0"], &source, &reference);
        assert!(
            reference_run.status.success(),
            "{format}: {reference_run:?}"
        );
        assert!(
            String::from_utf8_lossy(&reference_run.stdout)
                .contains("Block scales: amax (reference rule)"),
            "{format}"
        );

        let searched = dir.join(&format!("{format}-searched.safetensors"));
        let searched_run = run(
            format,
            &["--scale-search", "6", "--threads", "1"],
            &source,
            &searched,
        );
        assert!(searched_run.status.success(), "{format}: {searched_run:?}");
        assert!(
            String::from_utf8_lossy(&searched_run.stdout).contains("Block scales: min-mse:r6"),
            "{format}"
        );
        assert!(report_value(&searched_run, "Max MSE:") < report_value(&reference_run, "Max MSE:"));
        assert_ne!(fs::read(&searched).unwrap(), fs::read(&reference).unwrap());

        // Without the flag the CLI uses radius 6.
        let implicit = dir.join(&format!("{format}-implicit.safetensors"));
        let implicit_run = run(format, &[], &source, &implicit);
        assert!(implicit_run.status.success(), "{format}: {implicit_run:?}");
        assert!(
            String::from_utf8_lossy(&implicit_run.stdout).contains("Block scales: min-mse:r6"),
            "{format}"
        );
        assert_eq!(
            fs::read(&implicit).unwrap(),
            fs::read(&searched).unwrap(),
            "{format}: the default must equal --scale-search 6"
        );

        let expected = fs::read(&searched).unwrap();
        for threads in ["2", "8"] {
            let path = dir.join(&format!("{format}-searched-{threads}.safetensors"));
            let result = run(
                format,
                &["--scale-search", "6", "--threads", threads],
                &source,
                &path,
            );
            assert!(result.status.success(), "{format}/{threads}: {result:?}");
            assert_eq!(fs::read(&path).unwrap(), expected, "{format}/{threads}");
        }
    }
}

#[test]
fn the_flag_is_rejected_for_int8_and_out_of_range_values() {
    let dir = TestDir::new("reject");
    let source = dir.join("source.safetensors");
    write_source(&source, &[("a.weight", 64, 64, 6)]);
    let output = dir.join("out.safetensors");

    assert!(
        !run("int8", &["--scale-search", "4"], &source, &output)
            .status
            .success()
    );
    assert!(
        !run("nvfp4", &["--scale-search", "33"], &source, &output)
            .status
            .success()
    );
    assert!(
        !run("nvfp4", &["--scale-search", "-1"], &source, &output)
            .status
            .success()
    );
    assert!(!output.exists());
}
