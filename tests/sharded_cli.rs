use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, id},
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::safetensors::MappedSafetensors;
use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "modelq-sharded-cli-{label}-{}-{}",
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

fn write_shard(path: &Path, tensors: &[(&str, &str, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, payload) in tensors {
        let start = data.len();
        data.extend_from_slice(payload);
        let shape = if *dtype == "F32" {
            payload.len() / 4
        } else {
            payload.len()
        };
        header.insert(
            (*name).to_owned(),
            json!({ "dtype": dtype, "shape": [shape], "data_offsets": [start, data.len()] }),
        );
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes).unwrap();
}

fn f32_bytes(count: usize, offset: f32) -> Vec<u8> {
    (0..count)
        .flat_map(|index| ((index as f32 - 8.0) / 4.0 + offset).to_le_bytes())
        .collect()
}

fn checkpoint(dir: &TestDir) {
    write_shard(
        &dir.join("model-00001-of-00002.safetensors"),
        &[
            ("z.weight", "F32", f32_bytes(4096, 0.0)),
            ("ids", "U8", vec![1, 2, 3]),
        ],
    );
    write_shard(
        &dir.join("model-00002-of-00002.safetensors"),
        &[("a.weight", "F32", f32_bytes(4096, 1.0))],
    );
    fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "metadata": { "total_size": 4096 * 4 * 2 + 3 },
            "weight_map": {
                "z.weight": "model-00001-of-00002.safetensors",
                "ids": "model-00001-of-00002.safetensors",
                "a.weight": "model-00002-of-00002.safetensors"
            }
        }))
        .unwrap(),
    )
    .unwrap();
}

#[test]
fn inspect_lists_a_sharded_checkpoint_in_name_order() {
    let dir = TestDir::new("inspect");
    checkpoint(&dir);

    let output = Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("inspect")
        .arg(&dir.0)
        .output()
        .unwrap();

    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("Shards: 2"), "{text}");
    let a = text.find("a.weight").unwrap();
    let ids = text.find("ids |").unwrap();
    let z = text.find("z.weight").unwrap();
    assert!(a < ids && ids < z, "{text}");
}

#[test]
fn quantize_accepts_a_sharded_checkpoint() {
    let dir = TestDir::new("quantize");
    checkpoint(&dir);
    let output_path = dir.join("out.safetensors");

    let result = Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("quantize")
        .arg(&dir.0)
        .args(["--format", "int8", "--output"])
        .arg(&output_path)
        .output()
        .unwrap();

    assert!(result.status.success(), "{result:?}");
    let reader = MappedSafetensors::open(&output_path).unwrap();
    let names: Vec<_> = reader
        .tensors()
        .map(|tensor| tensor.name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "a.weight.qdata",
            "a.weight.scale",
            "ids",
            "z.weight.qdata",
            "z.weight.scale"
        ]
    );
    assert_eq!(reader.tensor_bytes("ids").unwrap(), [1, 2, 3]);
}

#[test]
fn quantize_rejects_output_inside_the_source_shard_set() {
    let dir = TestDir::new("conflict");
    checkpoint(&dir);

    let result = Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("quantize")
        .arg(&dir.0)
        .args(["--format", "int8", "--output"])
        .arg(dir.join("model-00002-of-00002.safetensors"))
        .output()
        .unwrap();

    assert!(!result.status.success());
}
