use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, id},
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::{
    safetensors::{MappedSafetensors, TensorSource},
    sharded::SafetensorsInput,
    te_container::{plan_te_output, read_te_container_manifest, write_te_nvfp4_safetensors},
};
use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "modelq-te-cli-{label}-{}-{}",
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

type Fixture = (&'static str, &'static str, Vec<usize>, Vec<u8>);

fn matrix(name: &'static str, shape: &[usize], seed: f32) -> Fixture {
    let count: usize = shape.iter().product();
    let payload = (0..count)
        .flat_map(|index| {
            (((index as f32) * 0.37 + seed).sin() * (1.0 + (index % 11) as f32)).to_le_bytes()
        })
        .collect();
    (name, "F32", shape.to_vec(), payload)
}

fn write_safetensors(path: &Path, tensors: &[&Fixture]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, payload) in tensors {
        let start = data.len();
        data.extend_from_slice(payload);
        header.insert(
            (*name).to_owned(),
            json!({ "dtype": dtype, "shape": shape, "data_offsets": [start, data.len()] }),
        );
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes).unwrap();
}

struct Tensors {
    a: Fixture,
    b: Fixture,
    head: Fixture,
    stack: Fixture,
    odd: Fixture,
    small: Fixture,
    norm: Fixture,
    ids: Fixture,
}

fn tensors() -> Tensors {
    Tensors {
        a: matrix("a.weight", &[64, 64], 0.0),
        b: matrix("b.weight", &[48, 96], 1.5),
        head: matrix("lm_head.weight", &[64, 64], 2.0),
        stack: matrix("stack.weight", &[4, 64, 64], 3.0),
        odd: matrix("odd.weight", &[70, 64], 4.0),
        small: matrix("small.weight", &[16, 32], 5.0),
        norm: matrix("norm.weight", &[4096], 6.0),
        ids: ("ids", "U8", vec![3], vec![1, 2, 3]),
    }
}

fn single(dir: &TestDir, t: &Tensors) -> PathBuf {
    let path = dir.join("model.safetensors");
    write_safetensors(
        &path,
        &[
            &t.a, &t.b, &t.head, &t.stack, &t.odd, &t.small, &t.norm, &t.ids,
        ],
    );
    path
}

fn sharded(dir: &TestDir, t: &Tensors) -> PathBuf {
    let checkpoint = dir.join("checkpoint");
    fs::create_dir_all(&checkpoint).unwrap();
    write_safetensors(
        &checkpoint.join("s1.safetensors"),
        &[&t.a, &t.stack, &t.norm, &t.ids],
    );
    write_safetensors(
        &checkpoint.join("s2.safetensors"),
        &[&t.b, &t.head, &t.odd, &t.small],
    );
    let map: serde_json::Map<String, Value> = [
        ("a.weight", "s1.safetensors"),
        ("stack.weight", "s1.safetensors"),
        ("norm.weight", "s1.safetensors"),
        ("ids", "s1.safetensors"),
        ("b.weight", "s2.safetensors"),
        ("lm_head.weight", "s2.safetensors"),
        ("odd.weight", "s2.safetensors"),
        ("small.weight", "s2.safetensors"),
    ]
    .into_iter()
    .map(|(name, shard)| (name.to_owned(), Value::String(shard.to_owned())))
    .collect();
    fs::write(
        checkpoint.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({ "weight_map": map })).unwrap(),
    )
    .unwrap();
    checkpoint
}

fn run(extra: &[&str], input: &Path, output: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("quantize")
        .arg(input)
        .args(["--format", "nvfp4-te"])
        .args(extra)
        .arg("--output")
        .arg(output)
        .output()
        .unwrap()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn names(reader: &MappedSafetensors) -> Vec<String> {
    let mut names: Vec<String> = reader.tensors().map(|tensor| tensor.name.clone()).collect();
    names.sort();
    names
}

#[test]
fn exports_eligible_matrices_and_preserves_the_rest_with_reasons() {
    let dir = TestDir::new("basic");
    let t = tensors();
    let input = single(&dir, &t);
    let output = dir.join("out.safetensors");

    let result = run(&[], &input, &output);

    assert!(result.status.success(), "{result:?}");
    let text = stdout(&result);
    assert!(text.contains("2 quantized, 6 preserved"), "{text}");
    assert!(
        text.contains("a leading dimension divisible by 16 and a final dimension divisible by 32"),
        "{text}"
    );
    assert!(
        text.contains("rank 3 tensors are preserved by the Transformer Engine profile"),
        "{text}"
    );
    assert!(text.contains("leading dimensions multiply to 70"), "{text}");
    assert!(
        text.contains("name matches excluded pattern \"lm_head\""),
        "{text}"
    );
    assert!(text.contains("below minimum"), "{text}");
    assert!(text.contains("Validated on an NVIDIA B200"), "{text}");
    assert!(text.contains("not validated"), "{text}");
    assert!(
        !text.contains("not yet hardware-validated"),
        "the stale pre-hardware wording must be gone: {text}"
    );

    let reader = MappedSafetensors::open(&output).unwrap();
    assert_eq!(
        names(&reader),
        [
            "a.weight.amax_rowwise",
            "a.weight.rowwise_data",
            "a.weight.rowwise_scale_inv",
            "b.weight.amax_rowwise",
            "b.weight.rowwise_data",
            "b.weight.rowwise_scale_inv",
            "ids",
            "lm_head.weight",
            "norm.weight",
            "odd.weight",
            "small.weight",
            "stack.weight",
        ]
    );
    read_te_container_manifest(&reader).expect("the manifest is valid");
    assert_eq!(reader.tensor_bytes("stack.weight").unwrap(), t.stack.3);

    // The CLI writes exactly what the library writes for the same selection.
    let library = dir.join("library.safetensors");
    let source = MappedSafetensors::open(&input).unwrap();
    let plan = plan_te_output(
        &source.tensor_summaries(),
        &["a.weight".to_owned(), "b.weight".to_owned()],
    )
    .unwrap();
    write_te_nvfp4_safetensors(&source, &plan, &library).unwrap();
    assert_eq!(fs::read(&output).unwrap(), fs::read(&library).unwrap());
}

#[test]
fn exclusion_flags_change_the_selection() {
    let dir = TestDir::new("flags");
    let t = tensors();
    let input = single(&dir, &t);

    let all = dir.join("all.safetensors");
    assert!(
        run(&["--no-default-excludes"], &input, &all)
            .status
            .success()
    );
    let reader = MappedSafetensors::open(&all).unwrap();
    assert!(reader.tensor_bytes("lm_head.weight.rowwise_data").is_ok());

    let custom = dir.join("custom.safetensors");
    assert!(
        run(&["--exclude", "a.weight"], &input, &custom)
            .status
            .success()
    );
    let reader = MappedSafetensors::open(&custom).unwrap();
    assert_eq!(reader.tensor_bytes("a.weight").unwrap(), t.a.3);
    assert!(reader.tensor_bytes("b.weight.rowwise_data").is_ok());
}

#[test]
fn single_and_sharded_inputs_and_all_thread_counts_write_identical_files() {
    let dir = TestDir::new("equivalence");
    let t = tensors();
    let single_input = single(&dir, &t);
    let sharded_input = sharded(&dir, &t);

    let reference = dir.join("reference.safetensors");
    assert!(
        run(&["--threads", "1"], &single_input, &reference)
            .status
            .success()
    );
    let expected = fs::read(&reference).unwrap();

    for (label, input, extra) in [
        ("sharded", &sharded_input, vec![]),
        ("threads2", &single_input, vec!["--threads", "2"]),
        ("threads8", &sharded_input, vec!["--threads", "8"]),
        ("default", &single_input, vec![]),
    ] {
        let output = dir.join(&format!("{label}.safetensors"));
        let result = run(&extra, input, &output);
        assert!(result.status.success(), "{label}: {result:?}");
        assert_eq!(fs::read(&output).unwrap(), expected, "{label}");
    }
}

#[test]
fn sharded_output_keeps_each_matrix_together_and_every_shard_self_describing() {
    let dir = TestDir::new("shards");
    let t = tensors();
    let input = single(&dir, &t);
    let single_output = dir.join("single.safetensors");
    assert!(run(&[], &input, &single_output).status.success());

    // a: 2048+512+4 bytes, b: 2304+1024+4 bytes; force them apart.
    let sharded_output = dir.join("sharded");
    let result = run(&["--max-shard-size", "4000"], &input, &sharded_output);
    assert!(result.status.success(), "{result:?}");

    let shards: Vec<PathBuf> = {
        let mut files: Vec<PathBuf> = fs::read_dir(&sharded_output)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        files.sort();
        files
    };
    assert!(shards.len() >= 2, "{shards:?}");
    let index: Value = serde_json::from_slice(
        &fs::read(sharded_output.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let map = index["weight_map"].as_object().unwrap();
    for matrix in ["a.weight", "b.weight"] {
        assert_eq!(
            map[&format!("{matrix}.rowwise_data")],
            map[&format!("{matrix}.rowwise_scale_inv")]
        );
        assert_eq!(
            map[&format!("{matrix}.rowwise_data")],
            map[&format!("{matrix}.amax_rowwise")]
        );
    }
    for shard in &shards {
        let reader = MappedSafetensors::open(shard).unwrap();
        read_te_container_manifest(&reader).expect("each shard has a valid manifest");
    }

    // Same tensors, byte for byte, as the single-file output.
    let from_single = SafetensorsInput::open(&single_output).unwrap();
    let from_shards = SafetensorsInput::open(&sharded_output).unwrap();
    let collect = |input: &SafetensorsInput| -> Vec<(String, Vec<u8>)> {
        input
            .tensors()
            .iter()
            .map(|tensor| {
                let name = tensor.summary.name.clone();
                let bytes = input.with_tensor_bytes(&name, <[u8]>::to_vec).unwrap();
                (name, bytes)
            })
            .collect()
    };
    assert_eq!(collect(&from_single), collect(&from_shards));
}

#[test]
fn failures_and_option_errors_leave_no_output() {
    let dir = TestDir::new("errors");
    let mut t = tensors();
    let input_ok = single(&dir, &t);
    let output = dir.join("out.safetensors");

    // Existing destinations and bad options are rejected before writing.
    fs::write(&output, b"keep").unwrap();
    assert!(!run(&[], &input_ok, &output).status.success());
    assert_eq!(fs::read(&output).unwrap(), b"keep");
    fs::remove_file(&output).unwrap();
    assert!(
        !run(&["--threads", "0"], &input_ok, &output)
            .status
            .success()
    );
    assert!(!output.exists());

    // A NaN fails mid-write with no output and no temporary file.
    t.a.3[100..104].copy_from_slice(&f32::NAN.to_le_bytes());
    let bad = dir.join("bad.safetensors");
    write_safetensors(&bad, &[&t.a, &t.b]);
    let result = run(&[], &bad, &output);
    assert!(!result.status.success());
    assert!(!output.exists());
    assert!(String::from_utf8_lossy(&result.stderr).contains("a.weight"));
    let leftovers = fs::read_dir(&dir.0)
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp")
        })
        .count();
    assert_eq!(leftovers, 0);
}
