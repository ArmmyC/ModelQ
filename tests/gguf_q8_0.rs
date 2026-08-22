use std::{
    env, fs,
    path::{Path, PathBuf},
    process::id,
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::gguf::{
    DEFAULT_ALIGNMENT, GGML_FTYPE_MOSTLY_Q8_0, GGML_QUANTIZATION_VERSION, GGML_TYPE_Q8_0,
    encode_q8_0, inspect, read_q8_0, write_q8_0,
};

static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

struct TempPath(PathBuf);

impl TempPath {
    fn new(label: &str) -> Self {
        let serial = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        Self(env::temp_dir().join(format!("modelq-task18-{label}-{}-{serial}.gguf", id())))
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

fn values(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| (index as f32 - 31.5) / 10.0)
        .collect()
}

#[test]
fn encodes_and_reads_a_deterministic_q8_0_fixture() {
    let source = values(64);
    let first = encode_q8_0("weight", &[2, 32], &source).expect("the shaped source is valid");
    let second = encode_q8_0("weight", &[2, 32], &source).expect("the second encoding is valid");
    assert_eq!(first, second);

    let summary = inspect(&first).expect("the fixture has a valid GGUF structure");
    assert_eq!(summary.version, 3);
    assert_eq!(summary.tensor_count, 1);
    assert_eq!(summary.kv_count, 3);
    assert_eq!(summary.alignment, DEFAULT_ALIGNMENT as u32);
    assert_eq!(summary.quantization_version, GGML_QUANTIZATION_VERSION);
    assert_eq!(summary.file_type, GGML_FTYPE_MOSTLY_Q8_0);
    assert_eq!(summary.data_offset % DEFAULT_ALIGNMENT as u64, 0);
    assert_eq!(summary.tensors[0].name, "weight");
    assert_eq!(summary.tensors[0].shape, [2, 32]);
    assert_eq!(summary.tensors[0].ggml_type, GGML_TYPE_Q8_0);
    assert_eq!(summary.tensors[0].offset, 0);
    assert_eq!(summary.tensors[0].byte_len, 2 * 34);

    let (read_summary, quantized) = read_q8_0(&first).expect("the Q8_0 payload can be read");
    assert_eq!(read_summary, summary);
    assert_eq!(quantized.len(), source.len());
    assert_eq!(quantized.bytes().len(), 2 * 34);
    assert!(quantized.dequantize().iter().all(|value| value.is_finite()));
}

#[test]
fn writes_without_replacing_an_existing_destination() {
    let path = TempPath::new("write");
    let source = values(32);
    write_q8_0(path.path(), "weight", &[32], &source).expect("the destination is new");
    let error = write_q8_0(path.path(), "weight", &[32], &source)
        .expect_err("the focused writer refuses replacement");
    assert!(matches!(
        error,
        modelq::io::gguf::GgufError::DestinationExists { .. }
    ));
}

#[test]
fn rejects_non_block_aligned_shapes() {
    let error = encode_q8_0("weight", &[2, 16], &[0.0; 32])
        .expect_err("Q8_0 rows must end on a 32-value block");
    assert!(matches!(
        error,
        modelq::io::gguf::GgufError::Quantization { .. }
    ));
}
