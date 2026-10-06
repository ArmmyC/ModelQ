use std::{
    fs,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::{
    safetensors::SafetensorsError,
    sharded::{SafetensorsInput, ShardedError},
};
use serde_json::{Value, json};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(stem: &str) -> Self {
        let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("modelq-sharded-{stem}-{}-{id}", process::id()));
        fs::create_dir_all(&path).expect("test directory is created");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
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

/// Writes a SafeTensors file; each tensor is `(name, dtype, shape, payload)`.
fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
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
    let mut header = serde_json::to_vec(&Value::Object(header)).expect("header serializes");
    let padded = header.len().div_ceil(8) * 8;
    header.resize(padded, b' ');
    let mut bytes = (padded as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes).expect("fixture is written");
}

fn f32_tensor(
    name: &'static str,
    values: &[f32],
) -> (&'static str, &'static str, Vec<usize>, Vec<u8>) {
    (
        name,
        "F32",
        vec![values.len()],
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
}

fn write_index(path: &Path, weight_map: Value, metadata: Option<Value>) {
    let mut root = json!({ "weight_map": weight_map });
    if let Some(metadata) = metadata {
        root["metadata"] = metadata;
    }
    fs::write(path, serde_json::to_vec(&root).unwrap()).expect("index is written");
}

/// Two shards; `b` and `d` in shard 2, `a` and `c` in shard 1. Total 16 bytes.
fn two_shard_checkpoint(dir: &TestDir) {
    write_safetensors(
        &dir.join("model-00001-of-00002.safetensors"),
        &[f32_tensor("a", &[1.0]), f32_tensor("c", &[3.0])],
    );
    write_safetensors(
        &dir.join("model-00002-of-00002.safetensors"),
        &[f32_tensor("b", &[2.0]), f32_tensor("d", &[4.0])],
    );
    // JSON member order deliberately differs from name order.
    write_index(
        &dir.join("model.safetensors.index.json"),
        json!({
            "d": "model-00002-of-00002.safetensors",
            "a": "model-00001-of-00002.safetensors",
            "c": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors"
        }),
        Some(json!({ "total_size": 16 })),
    );
}

fn names(input: &SafetensorsInput) -> Vec<&str> {
    input
        .tensors()
        .iter()
        .map(|tensor| tensor.summary.name.as_str())
        .collect()
}

fn open_err(path: &Path) -> ShardedError {
    match SafetensorsInput::open(path) {
        Ok(_) => panic!("expected {} to fail to open", path.display()),
        Err(error) => error,
    }
}

#[test]
fn directory_with_one_unsharded_file_is_discovered() {
    let dir = TestDir::new("single");
    write_safetensors(
        &dir.join("model.safetensors"),
        &[f32_tensor("w", &[1.0, 2.0])],
    );

    let input = SafetensorsInput::open(dir.path()).unwrap();

    assert_eq!(names(&input), ["w"]);
    assert_eq!(input.tensors()[0].summary.byte_len, 8);
}

#[test]
fn explicit_single_file_path_opens() {
    let dir = TestDir::new("explicit-file");
    let file = dir.join("diffusion_pytorch_model.safetensors");
    write_safetensors(&file, &[f32_tensor("w", &[1.0])]);

    let input = SafetensorsInput::open(&file).unwrap();

    assert_eq!(names(&input), ["w"]);
}

#[test]
fn sharded_input_is_sorted_by_name_and_stable_across_opens() {
    let dir = TestDir::new("sorted");
    two_shard_checkpoint(&dir);

    let first = SafetensorsInput::open(dir.path()).unwrap();
    let second = SafetensorsInput::open(dir.join("model.safetensors.index.json")).unwrap();

    assert_eq!(names(&first), ["a", "b", "c", "d"]);
    assert_eq!(names(&second), ["a", "b", "c", "d"]);
    assert!(
        first.tensors()[1]
            .shard
            .ends_with("model-00002-of-00002.safetensors")
    );
}

#[test]
fn tensor_payloads_are_read_from_the_owning_shard() {
    let dir = TestDir::new("payload");
    two_shard_checkpoint(&dir);
    let input = SafetensorsInput::open(dir.path()).unwrap();

    let bytes = input
        .with_tensor_bytes("b", |bytes| bytes.to_vec())
        .unwrap();
    assert_eq!(bytes, 2.0_f32.to_le_bytes());

    let values = input
        .with_tensor("d", |view| view.shape().to_vec())
        .unwrap();
    assert_eq!(values, [1]);

    assert!(matches!(
        input.with_tensor_bytes("missing", |_| ()),
        Err(ShardedError::TensorNotFound { .. })
    ));
}

#[test]
fn unsupported_dtype_is_inspectable_but_has_no_typed_view() {
    let dir = TestDir::new("dtype");
    write_safetensors(
        &dir.join("model.safetensors"),
        &[("ids", "U8", vec![2], vec![7, 9])],
    );
    let input = SafetensorsInput::open(dir.path()).unwrap();

    assert_eq!(input.tensors()[0].summary.dtype, "U8");
    assert_eq!(
        input
            .with_tensor_bytes("ids", |bytes| bytes.to_vec())
            .unwrap(),
        [7, 9]
    );
    assert!(matches!(
        input.with_tensor("ids", |_| ()),
        Err(ShardedError::Safetensors(
            SafetensorsError::UnsupportedTensorDtype { .. }
        ))
    ));
}

#[test]
fn missing_shard_is_reported() {
    let dir = TestDir::new("missing-shard");
    write_index(
        &dir.join("model.safetensors.index.json"),
        json!({ "a": "gone.safetensors" }),
        None,
    );

    assert!(matches!(
        open_err(dir.path()),
        ShardedError::Safetensors(SafetensorsError::Io { .. })
    ));
}

#[test]
fn malformed_index_is_rejected_without_falling_back_to_a_single_file() {
    let dir = TestDir::new("malformed");
    write_safetensors(&dir.join("model.safetensors"), &[f32_tensor("a", &[1.0])]);
    fs::write(dir.join("model.safetensors.index.json"), b"{ not json").unwrap();

    assert!(matches!(
        open_err(dir.path()),
        ShardedError::InvalidIndex { .. }
    ));
}

#[test]
fn index_schema_violations_are_rejected() {
    for (label, body) in [
        ("no-weight-map", json!({})),
        ("weight-map-not-object", json!({ "weight_map": [] })),
        ("non-string-shard", json!({ "weight_map": { "a": 1 } })),
        ("empty-shard", json!({ "weight_map": { "a": "" } })),
        (
            "metadata-not-object",
            json!({ "weight_map": {}, "metadata": 3 }),
        ),
        (
            "total-size-negative",
            json!({ "weight_map": {}, "metadata": { "total_size": -1 } }),
        ),
        (
            "metadata-key-as-tensor",
            json!({ "weight_map": { "__metadata__": "s.safetensors" } }),
        ),
    ] {
        let dir = TestDir::new(label);
        fs::write(
            dir.join("m.safetensors.index.json"),
            serde_json::to_vec(&body).unwrap(),
        )
        .unwrap();
        assert!(
            matches!(open_err(dir.path()), ShardedError::InvalidIndex { .. }),
            "{label} should be an invalid index"
        );
    }
}

#[test]
fn unsafe_shard_references_are_rejected() {
    for (label, shard) in [
        ("parent", "../x.safetensors"),
        ("nested", "sub/x.safetensors"),
        ("backslash", "sub\\x.safetensors"),
        ("dot", "."),
        ("dotdot", ".."),
        ("absolute", "/etc/x.safetensors"),
        ("drive", "C:x.safetensors"),
    ] {
        let dir = TestDir::new(label);
        write_index(
            &dir.join("m.safetensors.index.json"),
            json!({ "a": shard }),
            None,
        );
        assert!(
            matches!(
                open_err(dir.path()),
                ShardedError::UnsafeShardReference { .. }
            ),
            "{label} should be unsafe"
        );
    }
}

#[test]
fn tensor_in_two_shards_is_rejected() {
    let dir = TestDir::new("duplicate");
    write_safetensors(&dir.join("one.safetensors"), &[f32_tensor("a", &[1.0])]);
    write_safetensors(
        &dir.join("two.safetensors"),
        &[f32_tensor("a", &[2.0]), f32_tensor("b", &[3.0])],
    );
    write_index(
        &dir.join("m.safetensors.index.json"),
        json!({ "a": "one.safetensors", "b": "two.safetensors" }),
        None,
    );

    assert!(matches!(
        open_err(dir.path()),
        ShardedError::DuplicateTensor { .. }
    ));
}

#[test]
fn index_entry_missing_from_shard_is_rejected() {
    let dir = TestDir::new("missing-entry");
    write_safetensors(&dir.join("one.safetensors"), &[f32_tensor("a", &[1.0])]);
    write_index(
        &dir.join("m.safetensors.index.json"),
        json!({ "a": "one.safetensors", "b": "one.safetensors" }),
        None,
    );

    assert!(matches!(
        open_err(dir.path()),
        ShardedError::TensorSetMismatch { .. }
    ));
}

#[test]
fn shard_tensor_missing_from_index_is_rejected() {
    let dir = TestDir::new("extra-tensor");
    write_safetensors(
        &dir.join("one.safetensors"),
        &[f32_tensor("a", &[1.0]), f32_tensor("extra", &[2.0])],
    );
    write_index(
        &dir.join("m.safetensors.index.json"),
        json!({ "a": "one.safetensors" }),
        None,
    );

    assert!(matches!(
        open_err(dir.path()),
        ShardedError::TensorSetMismatch { .. }
    ));
}

#[test]
fn total_size_must_match_payload_bytes() {
    let dir = TestDir::new("total-size");
    write_safetensors(&dir.join("one.safetensors"), &[f32_tensor("a", &[1.0])]);
    write_index(
        &dir.join("m.safetensors.index.json"),
        json!({ "a": "one.safetensors" }),
        Some(json!({ "total_size": 5 })),
    );

    assert!(matches!(
        open_err(dir.path()),
        ShardedError::TotalSizeMismatch {
            expected: 5,
            actual: 4,
            ..
        }
    ));
}

#[test]
fn ambiguous_or_empty_directories_are_rejected() {
    let two_indexes = TestDir::new("two-indexes");
    write_index(
        &two_indexes.join("a.safetensors.index.json"),
        json!({}),
        None,
    );
    write_index(
        &two_indexes.join("b.safetensors.index.json"),
        json!({}),
        None,
    );
    assert!(matches!(
        open_err(two_indexes.path()),
        ShardedError::AmbiguousInput { .. }
    ));

    let two_files = TestDir::new("two-files");
    write_safetensors(&two_files.join("a.safetensors"), &[f32_tensor("a", &[1.0])]);
    write_safetensors(&two_files.join("b.safetensors"), &[f32_tensor("b", &[1.0])]);
    assert!(matches!(
        open_err(two_files.path()),
        ShardedError::AmbiguousInput { .. }
    ));

    let empty = TestDir::new("empty");
    assert!(matches!(
        open_err(empty.path()),
        ShardedError::NoSafetensorsFound { .. }
    ));
}

#[test]
fn missing_path_is_rejected() {
    let dir = TestDir::new("nonexistent");
    assert!(matches!(
        open_err(&dir.join("nope")),
        ShardedError::UnsupportedInputPath { .. }
    ));
}
