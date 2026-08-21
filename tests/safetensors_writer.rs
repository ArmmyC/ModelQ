use std::{
    env, fs,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::{
    io::{
        layout::plan_output_layout,
        safetensors::{MappedSafetensors, inspect_file},
        writer::{WriterError, write_safetensors},
    },
    quant::policy::{QuantizationPolicy, TensorCandidate},
};
use serde_json::{Value, json};

static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

struct TempPath(PathBuf);

impl TempPath {
    fn new(label: &str, contents: Option<&[u8]>) -> Self {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!(
            "modelq-task11-{label}-{}-{id}.safetensors",
            process::id()
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

fn source_file(weight: [f32; 2]) -> Vec<u8> {
    let header = json!({
        "ids": {
            "dtype": "U8",
            "shape": [3],
            "data_offsets": [0, 3]
        },
        "weight": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [3, 11]
        }
    });
    let mut header = serde_json::to_vec(&header).expect("fixture metadata serializes");
    header.resize(header.len().div_ceil(8) * 8, b' ');

    let mut file = Vec::with_capacity(8 + header.len() + 11);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&[7, 8, 9]);
    for value in weight {
        file.extend_from_slice(&value.to_le_bytes());
    }
    file
}

fn plan_for(
    reader: &MappedSafetensors,
) -> (
    modelq::io::layout::OutputLayoutPlan,
    Vec<modelq::quant::policy::TensorDecision>,
) {
    let decisions = QuantizationPolicy::new(1).decide_all([
        TensorCandidate::non_floating("ids", 3),
        TensorCandidate::floating("weight", 2),
    ]);
    let plan = plan_output_layout(&reader.inspection().tensors, &decisions)
        .expect("the fixture layout is valid");
    (plan, decisions)
}

fn data_start(bytes: &[u8]) -> usize {
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("length is eight bytes"));
    8 + usize::try_from(header_len).expect("test header fits in usize")
}

#[test]
fn writes_reopenable_output_with_planned_offsets_and_bytes() {
    let source_path = TempPath::new("source", Some(&source_file([1.0, -0.5])));
    let output_path = TempPath::new("output", None);
    let reader = MappedSafetensors::open(source_path.path()).expect("source fixture maps");
    let (plan, decisions) = plan_for(&reader);

    write_safetensors(&reader, &plan, &decisions, output_path.path())
        .expect("the planned output writes successfully");

    let output = fs::read(output_path.path()).expect("output exists");
    let inspection = inspect_file(output_path.path()).expect("output is valid SafeTensors");
    let mapped_output =
        MappedSafetensors::open(output_path.path()).expect("output can be reopened");
    assert_eq!(
        inspection
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>(),
        ["ids", "weight.qdata", "weight.scale"]
    );
    assert_eq!(plan.total_data_bytes, 9);
    assert_eq!(inspection.tensors[0].byte_len, 3);
    assert_eq!(inspection.tensors[1].byte_len, 2);
    assert_eq!(inspection.tensors[2].byte_len, 4);
    for (planned, actual) in plan.tensors.iter().zip(&inspection.tensors) {
        assert_eq!(planned.name, actual.name);
        let (start, end) = match planned.name.as_str() {
            "ids" => (0, 3),
            "weight.qdata" => (3, 5),
            "weight.scale" => (5, 9),
            other => panic!("unexpected output tensor {other}"),
        };
        assert_eq!(planned.data_offsets, start..end);
        assert_eq!(actual.byte_len, end - start);
    }

    let start = data_start(&output);
    assert_eq!(&output[start..start + 3], [7, 8, 9]);
    assert_eq!(&output[start + 3..start + 5], [127, 192]);
    assert_eq!(
        f32::from_le_bytes(
            output[start + 5..start + 9]
                .try_into()
                .expect("scale is four bytes")
        ),
        1.0 / 127.0
    );
    assert_eq!(mapped_output.tensor_bytes("ids").unwrap(), [7, 8, 9]);

    let header_len = data_start(&output);
    let header: Value =
        serde_json::from_slice(&output[8..header_len]).expect("the padded header is valid JSON");
    let metadata = header
        .get("__metadata__")
        .and_then(Value::as_object)
        .expect("metadata is an object");
    assert_eq!(metadata["modelq.format"], "modelq-native");
    assert_eq!(metadata["modelq.format_version"], "1");
    let manifest: Value = serde_json::from_str(
        metadata["modelq.manifest"]
            .as_str()
            .expect("manifest is a string"),
    )
    .expect("manifest is JSON");
    assert_eq!(manifest["schema"], "modelq.int8.manifest.v1");
    assert_eq!(manifest["tensors"]["ids"]["action"], "preserved");
    assert_eq!(manifest["tensors"]["weight"]["action"], "quantized");
}

#[test]
fn repeated_writes_are_byte_for_byte_deterministic() {
    let source_path = TempPath::new("deterministic-source", Some(&source_file([1.0, -0.5])));
    let first_path = TempPath::new("deterministic-first", None);
    let second_path = TempPath::new("deterministic-second", None);
    let reader = MappedSafetensors::open(source_path.path()).expect("source fixture maps");
    let (plan, decisions) = plan_for(&reader);

    write_safetensors(&reader, &plan, &decisions, first_path.path()).expect("first write works");
    write_safetensors(&reader, &plan, &decisions, second_path.path()).expect("second write works");

    assert_eq!(
        fs::read(first_path.path()).expect("first output exists"),
        fs::read(second_path.path()).expect("second output exists")
    );
}

#[test]
fn quantization_failure_leaves_source_and_destination_untouched() {
    let source_bytes = source_file([f32::NAN, 1.0]);
    let source_path = TempPath::new("failure-source", Some(&source_bytes));
    let output_path = TempPath::new("failure-output", None);
    let reader = MappedSafetensors::open(source_path.path()).expect("source fixture maps");
    let (plan, decisions) = plan_for(&reader);

    let error = write_safetensors(&reader, &plan, &decisions, output_path.path())
        .expect_err("non-finite source values are rejected");
    assert!(matches!(error, WriterError::Quantization { .. }));
    assert_eq!(
        fs::read(source_path.path()).expect("source remains readable"),
        source_bytes
    );
    assert!(!output_path.path().exists());
}

#[test]
fn refuses_in_place_writing_before_touching_the_source() {
    let source_bytes = source_file([1.0, -0.5]);
    let source_path = TempPath::new("in-place-source", Some(&source_bytes));
    let reader = MappedSafetensors::open(source_path.path()).expect("source fixture maps");
    let (plan, decisions) = plan_for(&reader);

    let error = write_safetensors(&reader, &plan, &decisions, source_path.path())
        .expect_err("in-place writing is rejected");
    assert!(matches!(
        error,
        WriterError::SourceDestinationConflict { .. }
    ));
    assert_eq!(
        fs::read(source_path.path()).expect("source remains readable"),
        source_bytes
    );
}
