use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, id},
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::safetensors::{MappedSafetensors, inspect_file};
use serde_json::json;

static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

struct TempPath(PathBuf);

impl TempPath {
    fn new(label: &str, contents: Option<&[u8]>) -> Self {
        let serial = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!(
            "modelq-task12-{label}-{}-{serial}.safetensors",
            id()
        ));
        if let Some(contents) = contents {
            fs::write(&path, contents).expect("temporary fixture can be written");
        }
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn large_source() -> Vec<u8> {
    const ELEMENTS: usize = 2048;
    let weight_end = 3 + ELEMENTS * 4;
    let header = json!({
        "ids": {
            "dtype": "U8",
            "shape": [3],
            "data_offsets": [0, 3]
        },
        "weight": {
            "dtype": "F32",
            "shape": [ELEMENTS],
            "data_offsets": [3, weight_end]
        }
    });
    let mut header = serde_json::to_vec(&header).expect("fixture header serializes");
    header.resize(header.len().div_ceil(8) * 8, b' ');

    let mut file = Vec::with_capacity(8 + header.len() + weight_end);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&[1, 2, 3]);
    for index in 0..ELEMENTS {
        let value = (index as f32 - 1024.0) / 64.0;
        file.extend_from_slice(&value.to_le_bytes());
    }
    file
}

#[test]
fn quantize_command_writes_smaller_validated_output() {
    let source_bytes = large_source();
    let source_path = TempPath::new("source", Some(&source_bytes));
    let output_path = TempPath::new("output", None);

    let result = Command::new(env!("CARGO_BIN_EXE_modelq"))
        .arg("quantize")
        .arg(source_path.path())
        .args(["--format", "int8", "--device", "cpu", "--output"])
        .arg(output_path.path())
        .output()
        .expect("the modelq binary starts");

    assert!(
        result.status.success(),
        "quantize failed: stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("Progress:"));
    assert!(stdout.contains("Final report:"));
    assert!(stdout.contains("Validation: passed"));
    assert!(stdout.contains("1 quantized"));

    let source_inspection = inspect_file(source_path.path()).expect("source remains valid");
    let output_inspection = inspect_file(output_path.path()).expect("output is valid");
    assert!(output_inspection.file_size < source_inspection.file_size);
    assert_eq!(output_inspection.tensors.len(), 3);

    let output = MappedSafetensors::open(output_path.path()).expect("output reopens");
    assert_eq!(output.tensor_bytes("ids").unwrap(), [1, 2, 3]);
    assert_eq!(output.tensor_bytes("weight.qdata").unwrap().len(), 2048);
    assert_eq!(output.tensor_bytes("weight.scale").unwrap().len(), 4);
    assert_eq!(
        fs::read(source_path.path()).expect("source can be reread"),
        source_bytes
    );
}
