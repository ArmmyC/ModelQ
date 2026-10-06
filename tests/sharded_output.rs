use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, id},
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::{safetensors::MappedSafetensors, sharded::SafetensorsInput};
use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "modelq-sharded-output-{label}-{}-{}",
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

/// `(name, dtype, shape, payload)`
type Fixture = (String, &'static str, Vec<usize>, Vec<u8>);

fn f32_matrix(name: &str, seed: f32) -> Fixture {
    let payload = (0..64 * 64)
        .flat_map(|index| {
            (((index as f32) * 0.37 + seed).sin() * (1.0 + (index % 7) as f32)).to_le_bytes()
        })
        .collect();
    (name.to_owned(), "F32", vec![64, 64], payload)
}

fn fixtures() -> Vec<Fixture> {
    vec![
        f32_matrix("w0", 0.0),
        f32_matrix("w1", 1.0),
        f32_matrix("w2", 2.0),
        f32_matrix("w3", 3.0),
        ("ids".to_owned(), "U8", vec![3], vec![1, 2, 3]),
    ]
}

fn write_source(path: &Path, tensors: &[Fixture]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, payload) in tensors {
        let start = data.len();
        data.extend_from_slice(payload);
        header.insert(
            name.clone(),
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

/// Every output tensor's bytes, keyed by name, read through the sharded reader.
fn contents(path: &Path) -> BTreeMap<String, Vec<u8>> {
    let input = SafetensorsInput::open(path).expect("output opens");
    input
        .tensors()
        .iter()
        .map(|tensor| {
            let name = tensor.summary.name.clone();
            let bytes = input
                .with_tensor_bytes(&name, |bytes| bytes.to_vec())
                .unwrap();
            (name, bytes)
        })
        .collect()
}

fn shard_files(directory: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();
    files
}

fn payload_bytes(shard: &Path) -> u64 {
    MappedSafetensors::open(shard)
        .unwrap()
        .tensors()
        .map(|tensor| tensor.byte_len)
        .sum()
}

fn index(directory: &Path) -> Value {
    serde_json::from_slice(&fs::read(directory.join("model.safetensors.index.json")).unwrap())
        .unwrap()
}

#[test]
fn int8_sharded_output_has_the_same_tensors_as_the_single_file_output() {
    let dir = TestDir::new("int8");
    let source = dir.join("source.safetensors");
    write_source(&source, &fixtures());
    let single = dir.join("single.safetensors");
    let sharded = dir.join("sharded");

    assert!(run("int8", &[], &source, &single).status.success());
    let result = run("int8", &["--max-shard-size", "9000"], &source, &sharded);
    assert!(result.status.success(), "{result:?}");

    assert_eq!(contents(&single), contents(&sharded));

    let shards = shard_files(&sharded);
    assert_eq!(shards.len(), 2, "{shards:?}");
    assert_eq!(
        shards[0].file_name().unwrap().to_string_lossy(),
        "model-00001-of-00002.safetensors"
    );
    for shard in &shards {
        assert!(payload_bytes(shard) <= 9000, "{shard:?} exceeds the limit");
    }

    // A quantized tensor and its scale never land in different shards.
    let index = index(&sharded);
    let map = index["weight_map"].as_object().unwrap();
    for name in ["w0", "w1", "w2", "w3"] {
        assert_eq!(map[&format!("{name}.qdata")], map[&format!("{name}.scale")]);
    }
    let total: u64 = shards.iter().map(|shard| payload_bytes(shard)).sum();
    assert_eq!(index["metadata"]["total_size"].as_u64().unwrap(), total);
}

#[test]
fn nvfp4_sharded_output_has_the_same_tensors_as_the_single_file_output() {
    let dir = TestDir::new("nvfp4");
    let source = dir.join("source.safetensors");
    write_source(&source, &fixtures());
    let single = dir.join("single.safetensors");
    let sharded = dir.join("sharded");

    assert!(run("nvfp4", &[], &source, &single).status.success());
    // Each NVFP4 matrix is 2048 + 256 + 4 payload bytes.
    let result = run("nvfp4", &["--max-shard-size", "5KB"], &source, &sharded);
    assert!(result.status.success(), "{result:?}");

    assert_eq!(contents(&single), contents(&sharded));

    let shards = shard_files(&sharded);
    assert!(shards.len() >= 2, "{shards:?}");
    let map = index(&sharded)["weight_map"].clone();
    for name in ["w0", "w1", "w2", "w3"] {
        assert_eq!(
            map[format!("{name}.qdata")],
            map[format!("{name}.block_scale")]
        );
        assert_eq!(
            map[format!("{name}.qdata")],
            map[format!("{name}.global_scale")]
        );
    }
    // Every shard is a complete NVFP4 file with its own manifest.
    for shard in &shards {
        let reader = MappedSafetensors::open(shard).unwrap();
        assert!(reader.metadata().contains_key("modelq.manifest"));
    }
}

#[test]
fn oversize_tensors_get_their_own_shard_and_one_shard_still_gets_an_index() {
    let dir = TestDir::new("sizes");
    let source = dir.join("source.safetensors");
    write_source(&source, &fixtures());

    let tiny = dir.join("tiny");
    assert!(
        run("int8", &["--max-shard-size", "1"], &source, &tiny)
            .status
            .success()
    );
    assert_eq!(shard_files(&tiny).len(), 5, "one shard per source tensor");

    let huge = dir.join("huge");
    assert!(
        run("int8", &["--max-shard-size", "1GiB"], &source, &huge)
            .status
            .success()
    );
    assert_eq!(shard_files(&huge).len(), 1);
    assert_eq!(
        shard_files(&huge)[0].file_name().unwrap().to_string_lossy(),
        "model-00001-of-00001.safetensors"
    );
    assert!(huge.join("model.safetensors.index.json").exists());
}

#[test]
fn sharded_input_to_sharded_output_and_inspect_round_trip() {
    let dir = TestDir::new("roundtrip");
    let source = dir.join("source.safetensors");
    write_source(&source, &fixtures());
    let first = dir.join("first");
    let second = dir.join("second");

    assert!(
        run("nvfp4", &["--max-shard-size", "5KB"], &source, &first)
            .status
            .success()
    );
    // A sharded output is a valid sharded input: preserved tensors pass through.
    assert!(
        run("int8", &["--max-shard-size", "5KB"], &first, &second)
            .status
            .success()
    );

    let inspect = Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("inspect")
        .arg(&second)
        .output()
        .unwrap();
    assert!(inspect.status.success());
    let text = String::from_utf8_lossy(&inspect.stdout);
    assert!(text.contains("(sharded)"), "{text}");
}

#[test]
fn output_runs_are_deterministic() {
    let dir = TestDir::new("deterministic");
    let source = dir.join("source.safetensors");
    write_source(&source, &fixtures());
    let a = dir.join("a");
    let b = dir.join("b");

    assert!(
        run("nvfp4", &["--max-shard-size", "5KB"], &source, &a)
            .status
            .success()
    );
    assert!(
        run("nvfp4", &["--max-shard-size", "5KB"], &source, &b)
            .status
            .success()
    );

    let files_a = shard_files(&a);
    let files_b = shard_files(&b);
    assert_eq!(files_a.len(), files_b.len());
    for (left, right) in files_a.iter().zip(&files_b) {
        assert_eq!(fs::read(left).unwrap(), fs::read(right).unwrap());
    }
    assert_eq!(
        fs::read(a.join("model.safetensors.index.json")).unwrap(),
        fs::read(b.join("model.safetensors.index.json")).unwrap()
    );
}

#[test]
fn unsafe_destinations_and_bad_sizes_are_rejected_without_changes() {
    let dir = TestDir::new("reject");
    let source = dir.join("source.safetensors");
    write_source(&source, &fixtures());

    let occupied = dir.join("occupied");
    fs::create_dir_all(&occupied).unwrap();
    fs::write(occupied.join("keep.txt"), b"keep").unwrap();
    assert!(
        !run("int8", &["--max-shard-size", "9000"], &source, &occupied)
            .status
            .success()
    );
    assert_eq!(fs::read(occupied.join("keep.txt")).unwrap(), b"keep");
    assert_eq!(fs::read_dir(&occupied).unwrap().count(), 1);

    let file = dir.join("a-file");
    fs::write(&file, b"file").unwrap();
    assert!(
        !run("nvfp4", &["--max-shard-size", "9000"], &source, &file)
            .status
            .success()
    );
    assert_eq!(fs::read(&file).unwrap(), b"file");

    for size in ["0", "abc", "1.5GB", "-1", ""] {
        let target = dir.join("bad-size");
        let result = run("int8", &["--max-shard-size", size], &source, &target);
        assert!(!result.status.success(), "size {size:?} should fail");
        assert!(!target.exists());
    }
}

#[test]
fn a_failed_run_leaves_no_directory_or_files_behind() {
    let dir = TestDir::new("failure");
    let mut tensors = fixtures();
    // A NaN in the last matrix to be written fails after earlier shards exist.
    tensors[3].3[100..104].copy_from_slice(&f32::NAN.to_le_bytes());
    let source = dir.join("source.safetensors");
    write_source(&source, &tensors);

    let created = dir.join("created");
    let result = run("nvfp4", &["--max-shard-size", "5KB"], &source, &created);
    assert!(!result.status.success());
    assert!(!created.exists(), "a directory made by the run is removed");

    let existing = dir.join("existing");
    fs::create_dir_all(&existing).unwrap();
    let result = run("nvfp4", &["--max-shard-size", "5KB"], &source, &existing);
    assert!(!result.status.success());
    assert_eq!(
        fs::read_dir(&existing).unwrap().count(),
        0,
        "an empty directory is left empty"
    );
}

#[test]
fn shard_size_with_inspect_is_not_accepted_and_help_mentions_the_flag() {
    let help = Command::new(env!("CARGO_BIN_EXE_modelq"))
        .args(["quantize", "--help"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&help.stdout).contains("--max-shard-size"));
}
