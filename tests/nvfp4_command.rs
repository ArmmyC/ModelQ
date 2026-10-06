use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, id},
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::{io::safetensors::MappedSafetensors, quant::nvfp4::quantize_shaped};
use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "modelq-nvfp4-cli-{label}-{}-{}",
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

struct Fixture {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    payload: Vec<u8>,
}

fn f32_fixture(name: &'static str, shape: &[usize], seed: f32) -> (Fixture, Vec<f32>) {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count)
        .map(|index| ((index as f32 * 0.37 + seed).sin()) * (1.0 + (index % 7) as f32))
        .collect();
    let payload = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    (
        Fixture {
            name,
            dtype: "F32",
            shape: shape.to_vec(),
            payload,
        },
        values,
    )
}

fn write_safetensors(path: &Path, tensors: &[&Fixture]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for tensor in tensors {
        let start = data.len();
        data.extend_from_slice(&tensor.payload);
        header.insert(
            tensor.name.to_owned(),
            json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [start, data.len()]
            }),
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
    weight: Fixture,
    weight_values: Vec<f32>,
    head: Fixture,
    norm: Fixture,
    ids: Fixture,
    tiny: Fixture,
    odd: Fixture,
}

fn tensors() -> Tensors {
    let (weight, weight_values) = f32_fixture("layers.0.weight", &[64, 64], 0.0);
    let (head, _) = f32_fixture("lm_head.weight", &[64, 64], 1.0);
    let (norm, _) = f32_fixture("norm.weight", &[4096], 2.0);
    let (tiny, _) = f32_fixture("tiny.weight", &[4, 16], 3.0);
    let (odd, _) = f32_fixture("odd.weight", &[64, 40], 4.0);
    let ids = Fixture {
        name: "ids",
        dtype: "U8",
        shape: vec![3],
        payload: vec![1, 2, 3],
    };
    Tensors {
        weight,
        weight_values,
        head,
        norm,
        ids,
        tiny,
        odd,
    }
}

fn single_file(dir: &TestDir, t: &Tensors) -> PathBuf {
    let path = dir.join("model.safetensors");
    write_safetensors(
        &path,
        &[&t.weight, &t.head, &t.norm, &t.ids, &t.tiny, &t.odd],
    );
    path
}

fn sharded(dir: &TestDir, t: &Tensors) -> PathBuf {
    let checkpoint = dir.join("checkpoint");
    fs::create_dir_all(&checkpoint).unwrap();
    write_safetensors(
        &checkpoint.join("s1.safetensors"),
        &[&t.weight, &t.norm, &t.ids],
    );
    write_safetensors(
        &checkpoint.join("s2.safetensors"),
        &[&t.head, &t.tiny, &t.odd],
    );
    let map: serde_json::Map<String, Value> = [
        (t.weight.name, "s1.safetensors"),
        (t.norm.name, "s1.safetensors"),
        (t.ids.name, "s1.safetensors"),
        (t.head.name, "s2.safetensors"),
        (t.tiny.name, "s2.safetensors"),
        (t.odd.name, "s2.safetensors"),
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

fn run(args: &[&str], input: &Path, output: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("quantize")
        .arg(input)
        .args(args)
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
fn nvfp4_quantizes_eligible_tensors_and_preserves_the_rest() {
    let dir = TestDir::new("basic");
    let t = tensors();
    let input = single_file(&dir, &t);
    let output = dir.join("out.safetensors");

    let result = run(&["--format", "nvfp4"], &input, &output);

    assert!(result.status.success(), "{result:?}");
    let text = stdout(&result);
    assert!(text.contains("1 quantized, 5 preserved"), "{text}");
    assert!(
        text.contains("name matches excluded pattern \"lm_head\""),
        "{text}"
    );
    assert!(text.contains("below minimum"), "{text}");
    assert!(text.contains("not divisible by 16"), "{text}");
    assert!(
        text.contains("no runtime compatibility is implied"),
        "{text}"
    );

    let reader = MappedSafetensors::open(&output).unwrap();
    assert_eq!(
        names(&reader),
        [
            "ids",
            "layers.0.weight.block_scale",
            "layers.0.weight.global_scale",
            "layers.0.weight.qdata",
            "lm_head.weight",
            "norm.weight",
            "odd.weight",
            "tiny.weight",
        ]
    );
    assert_eq!(reader.tensor_bytes("ids").unwrap(), [1, 2, 3]);
    assert_eq!(
        reader.tensor_bytes("lm_head.weight").unwrap(),
        t.head.payload
    );

    // The CLI output equals the reference quantizer on the same values.
    let reference = quantize_shaped(&t.weight_values, &[64, 64]).unwrap();
    assert_eq!(
        reader.tensor_bytes("layers.0.weight.qdata").unwrap(),
        reference.packed()
    );
    assert_eq!(
        reader.tensor_bytes("layers.0.weight.block_scale").unwrap(),
        reference.block_scales()
    );
    assert_eq!(
        reader.tensor_bytes("layers.0.weight.global_scale").unwrap(),
        reference.global_scale().to_le_bytes()
    );
}

#[test]
fn sharded_and_single_file_inputs_produce_identical_output() {
    let dir = TestDir::new("equivalence");
    let t = tensors();
    let single = single_file(&dir, &t);
    let shards = sharded(&dir, &t);
    let from_single = dir.join("single.safetensors");
    let from_shards = dir.join("sharded.safetensors");

    assert!(
        run(&["--format", "nvfp4"], &single, &from_single)
            .status
            .success()
    );
    assert!(
        run(&["--format", "nvfp4"], &shards, &from_shards)
            .status
            .success()
    );

    assert_eq!(
        fs::read(from_single).unwrap(),
        fs::read(from_shards).unwrap()
    );
}

#[test]
fn exclusion_flags_change_the_selection() {
    let dir = TestDir::new("flags");
    let t = tensors();
    let input = single_file(&dir, &t);

    let all = dir.join("all.safetensors");
    let result = run(
        &["--format", "nvfp4", "--no-default-excludes"],
        &input,
        &all,
    );
    assert!(result.status.success(), "{result:?}");
    let reader = MappedSafetensors::open(&all).unwrap();
    assert!(reader.tensor_bytes("lm_head.weight.qdata").is_ok());
    assert!(reader.tensor_bytes("layers.0.weight.qdata").is_ok());

    let custom = dir.join("custom.safetensors");
    let result = run(
        &[
            "--format",
            "nvfp4",
            "--no-default-excludes",
            "--exclude",
            "layers.0",
        ],
        &input,
        &custom,
    );
    assert!(result.status.success(), "{result:?}");
    let reader = MappedSafetensors::open(&custom).unwrap();
    assert!(reader.tensor_bytes("layers.0.weight.qdata").is_err());
    assert_eq!(
        reader.tensor_bytes("layers.0.weight").unwrap(),
        t.weight.payload
    );
    assert!(reader.tensor_bytes("lm_head.weight.qdata").is_ok());
}

#[test]
fn invalid_option_combinations_fail_before_writing() {
    let dir = TestDir::new("invalid");
    let t = tensors();
    let input = single_file(&dir, &t);
    let output = dir.join("out.safetensors");

    for args in [
        &["--format", "int8", "--exclude", "x"][..],
        &["--format", "int8", "--no-default-excludes"][..],
        &["--format", "nvfp4", "--device", "cuda"][..],
        &["--format", "fp8"][..],
    ] {
        let result = run(args, &input, &output);
        assert!(!result.status.success(), "{args:?} should fail");
        assert!(!output.exists(), "{args:?} must not create output");
    }
}

#[test]
fn existing_destination_and_source_shards_are_not_overwritten() {
    let dir = TestDir::new("overwrite");
    let t = tensors();
    let shards = sharded(&dir, &t);
    let existing = dir.join("existing.safetensors");
    fs::write(&existing, b"keep me").unwrap();

    assert!(
        !run(&["--format", "nvfp4"], &shards, &existing)
            .status
            .success()
    );
    assert_eq!(fs::read(&existing).unwrap(), b"keep me");

    let shard = shards.join("s2.safetensors");
    let before = fs::read(&shard).unwrap();
    assert!(
        !run(&["--format", "nvfp4"], &shards, &shard)
            .status
            .success()
    );
    assert_eq!(fs::read(&shard).unwrap(), before);
}

#[test]
fn non_finite_input_fails_without_creating_output() {
    let dir = TestDir::new("nan");
    let mut t = tensors();
    t.weight.payload[100..104].copy_from_slice(&f32::NAN.to_le_bytes());
    let input = single_file(&dir, &t);
    let output = dir.join("out.safetensors");

    let result = run(&["--format", "nvfp4"], &input, &output);

    assert!(!result.status.success());
    assert!(!output.exists());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("layers.0.weight"), "{error}");
}
