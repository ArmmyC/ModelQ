use std::{
    fs,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::{
    nvfp4::{Nvfp4OutputRole, Nvfp4WriterError, plan_nvfp4_output, write_nvfp4_safetensors},
    safetensors::{MappedSafetensors, TensorSummary, inspect_file},
};
use modelq::quant::nvfp4 as reference_nvfp4;
use serde_json::{Value, json};

static NEXT_TEST_PATH: AtomicU64 = AtomicU64::new(0);

struct CleanupPath(PathBuf);

impl Drop for CleanupPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn unique_path(stem: &str) -> CleanupPath {
    let id = NEXT_TEST_PATH.fetch_add(1, Ordering::Relaxed);
    CleanupPath(std::env::temp_dir().join(format!(
        "modelq-nvfp4-{stem}-{}-{id}.safetensors",
        process::id()
    )))
}

fn source_fixture(stem: &str, values: &[f32]) -> CleanupPath {
    assert_eq!(values.len(), 16);
    let ids = [7_u8, 9_u8];
    let weight = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let header = json!({
        "__metadata__": { "fixture": "task-24" },
        "ids": {
            "dtype": "U8",
            "shape": [2],
            "data_offsets": [0, 2]
        },
        "weight": {
            "dtype": "F32",
            "shape": [1, 16],
            "data_offsets": [2, 66]
        }
    });
    let mut header = serde_json::to_vec(&header).expect("fixture metadata serializes");
    let padded_len = header.len().div_ceil(8);
    let padded_len = padded_len * 8;
    header.resize(padded_len, b' ');

    let mut bytes = Vec::with_capacity(8 + padded_len + ids.len() + weight.len());
    bytes.extend_from_slice(&(padded_len as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&ids);
    bytes.extend_from_slice(&weight);

    let path = unique_path(stem);
    fs::write(&path.0, bytes).expect("temporary source fixture can be written");
    path
}

fn plan_for(source: &MappedSafetensors) -> modelq::io::nvfp4::Nvfp4OutputPlan {
    plan_nvfp4_output(&source.inspection().tensors, &["weight".to_owned()])
        .expect("fixture shape is valid for NVFP4")
}

fn read_header(path: &Path) -> Value {
    let bytes = fs::read(path).expect("output bytes can be read");
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes[..8].try_into().expect("header has a length prefix"),
    ))
    .expect("header length fits usize");
    serde_json::from_slice(&bytes[8..8 + header_len]).expect("output header is JSON")
}

#[test]
fn plans_selected_tensor_and_preserves_the_rest() {
    let sources = vec![
        TensorSummary {
            name: "weight".to_owned(),
            dtype: "F32".to_owned(),
            shape: vec![2, 16],
            byte_len: 128,
        },
        TensorSummary {
            name: "norm".to_owned(),
            dtype: "F32".to_owned(),
            shape: vec![16],
            byte_len: 64,
        },
    ];
    let selected = vec!["weight".to_owned()];

    let plan = plan_nvfp4_output(&sources, &selected).expect("the source shape is valid");

    assert_eq!(
        plan.tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>(),
        [
            "norm",
            "weight.qdata",
            "weight.block_scale",
            "weight.global_scale"
        ]
    );
    assert_eq!(plan.tensors[0].role, Nvfp4OutputRole::Preserved);
    assert_eq!(plan.tensors[0].dtype, "F32");
    assert_eq!(plan.tensors[0].shape, [16]);
    assert_eq!(plan.tensors[0].data_offsets, 0..64);

    assert_eq!(plan.tensors[1].role, Nvfp4OutputRole::QuantizedData);
    assert_eq!(plan.tensors[1].dtype, "U8");
    assert_eq!(plan.tensors[1].shape, [2, 8]);
    assert_eq!(plan.tensors[1].byte_len, 16);
    assert_eq!(plan.tensors[1].data_offsets, 64..80);

    assert_eq!(plan.tensors[2].role, Nvfp4OutputRole::BlockScales);
    assert_eq!(plan.tensors[2].dtype, "U8");
    assert_eq!(plan.tensors[2].shape, [2, 1]);
    assert_eq!(plan.tensors[2].byte_len, 2);
    assert_eq!(plan.tensors[2].data_offsets, 80..82);

    assert_eq!(plan.tensors[3].role, Nvfp4OutputRole::GlobalScale);
    assert_eq!(plan.tensors[3].dtype, "F32");
    assert_eq!(plan.tensors[3].shape, []);
    assert_eq!(plan.tensors[3].byte_len, 4);
    assert_eq!(plan.tensors[3].data_offsets, 82..86);
    assert_eq!(plan.total_data_bytes, 86);
}

#[test]
fn writes_and_reopens_native_nvfp4_output() {
    let values = (0..16).map(|index| index as f32 - 7.5).collect::<Vec<_>>();
    let source_path = source_fixture("round-trip-source", &values);
    let destination_path = unique_path("round-trip-output");
    let source = MappedSafetensors::open(&source_path.0).expect("source fixture opens");
    let plan = plan_for(&source);

    write_nvfp4_safetensors(&source, &plan, &destination_path.0)
        .expect("native NVFP4 output writes");

    let inspection = inspect_file(&destination_path.0).expect("output reopens as SafeTensors");
    assert_eq!(
        inspection
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>(),
        [
            "ids",
            "weight.qdata",
            "weight.block_scale",
            "weight.global_scale"
        ]
    );
    assert_eq!(inspection.tensors[1].shape, [1, 8]);
    assert_eq!(inspection.tensors[2].shape, [1, 1]);
    assert_eq!(inspection.tensors[3].shape, []);

    let header = read_header(&destination_path.0);
    let metadata = header
        .get("__metadata__")
        .and_then(Value::as_object)
        .expect("metadata is an object");
    for (key, expected) in [
        ("modelq.format", "modelq-native"),
        ("modelq.format_version", "1"),
        ("modelq.compatibility_level", "container-valid"),
        ("modelq.quantization", "nvfp4"),
        ("modelq.scheme", "weight-only-blockwise"),
        ("modelq.algorithm", "e2m1-e4m3-global-v0"),
        ("modelq.element_format", "fp4-e2m1"),
        ("modelq.block_scale_format", "fp8-e4m3"),
        ("modelq.global_scale_dtype", "F32"),
        ("modelq.global_scale_semantics", "decode"),
        ("modelq.block_size", "16"),
        ("modelq.packing", "e2m1-low-nibble-first"),
        ("modelq.rounding", "nearest-even"),
    ] {
        assert_eq!(metadata.get(key).and_then(Value::as_str), Some(expected));
    }

    let manifest_text = metadata
        .get("modelq.manifest")
        .and_then(Value::as_str)
        .expect("manifest is a JSON string");
    let manifest: Value = serde_json::from_str(manifest_text).expect("manifest is valid JSON");
    assert_eq!(
        manifest.get("schema").and_then(Value::as_str),
        Some("modelq.nvfp4.manifest.v1")
    );
    let quantized_record = &manifest["tensors"]["weight"];
    assert_eq!(quantized_record["action"], "quantized");
    assert_eq!(quantized_record["axis"].as_i64(), Some(-1));
    assert_eq!(quantized_record["block_size"].as_u64(), Some(16));
    assert_eq!(quantized_record["qdata_name"], "weight.qdata");
    assert_eq!(quantized_record["qdata_dtype"], "U8");
    assert_eq!(quantized_record["qdata_shape"], json!([1, 8]));
    assert_eq!(quantized_record["block_scale_name"], "weight.block_scale");
    assert_eq!(quantized_record["block_scale_dtype"], "U8");
    assert_eq!(quantized_record["block_scale_shape"], json!([1, 1]));
    assert_eq!(quantized_record["global_scale_name"], "weight.global_scale");
    assert_eq!(quantized_record["global_scale_dtype"], "F32");
    assert_eq!(quantized_record["global_scale_shape"], json!([]));
    assert_eq!(manifest["tensors"]["ids"]["action"], "preserved");

    let output = MappedSafetensors::open(&destination_path.0).expect("output mapping opens");
    assert_eq!(
        output.tensor_bytes("ids").expect("ids are preserved"),
        [7, 9]
    );
    let expected = reference_nvfp4::quantize_shaped(&values, &[1, 16])
        .expect("reference quantizer accepts the fixture");
    assert_eq!(
        output
            .tensor_bytes("weight.qdata")
            .expect("packed payload exists"),
        expected.packed_values()
    );
    assert_eq!(
        output
            .tensor_bytes("weight.block_scale")
            .expect("block scales exist"),
        expected.block_scales()
    );
    let actual_global_scale = f32::from_le_bytes(
        output
            .tensor_bytes("weight.global_scale")
            .expect("global scale exists")
            .try_into()
            .expect("global scale is four bytes"),
    );
    assert_eq!(actual_global_scale, expected.global_scale());
    let reopened = reference_nvfp4::QuantizedTensor::from_parts(
        output
            .tensor_bytes("weight.qdata")
            .expect("packed payload exists")
            .to_vec(),
        output
            .tensor_bytes("weight.block_scale")
            .expect("block scales exist")
            .to_vec(),
        actual_global_scale,
        values.len(),
    )
    .expect("reopened companion payloads form a valid NVFP4 tensor");
    assert_eq!(
        reopened.dequantize().expect("reopened tensor dequantizes"),
        expected.dequantize().expect("reference tensor dequantizes")
    );
}

#[test]
fn writes_identical_bytes_for_identical_source_and_plan() {
    let values = (0..16).map(|index| index as f32 - 7.5).collect::<Vec<_>>();
    let source_path = source_fixture("determinism-source", &values);
    let first_path = unique_path("determinism-first");
    let second_path = unique_path("determinism-second");
    let source = MappedSafetensors::open(&source_path.0).expect("source fixture opens");
    let plan = plan_for(&source);

    write_nvfp4_safetensors(&source, &plan, &first_path.0).expect("first output writes");
    write_nvfp4_safetensors(&source, &plan, &second_path.0).expect("second output writes");

    assert_eq!(
        fs::read(&first_path.0).expect("first output can be read"),
        fs::read(&second_path.0).expect("second output can be read")
    );
}

#[test]
fn rejects_nonfinite_input_without_leaving_a_partial_destination() {
    let mut values = (0..16).map(|index| index as f32 - 7.5).collect::<Vec<_>>();
    values[5] = f32::NAN;
    let source_path = source_fixture("nonfinite-source", &values);
    let destination_path = unique_path("nonfinite-output");
    let source_before = fs::read(&source_path.0).expect("source can be read");
    let source = MappedSafetensors::open(&source_path.0).expect("source fixture opens");
    let plan = plan_for(&source);

    let error = write_nvfp4_safetensors(&source, &plan, &destination_path.0)
        .expect_err("the scalar quantizer rejects NaN");
    assert!(matches!(error, Nvfp4WriterError::Quantization { .. }));
    assert!(!destination_path.0.exists());
    assert_eq!(
        fs::read(&source_path.0).expect("source remains readable"),
        source_before
    );
}

#[test]
fn rejects_existing_and_in_place_destinations() {
    let values = (0..16).map(|index| index as f32 - 7.5).collect::<Vec<_>>();
    let source_path = source_fixture("collision-source", &values);
    let existing_path = unique_path("collision-existing");
    fs::write(&existing_path.0, b"keep this artifact")
        .expect("existing destination can be written");
    let existing_before = fs::read(&existing_path.0).expect("existing destination can be read");
    let source = MappedSafetensors::open(&source_path.0).expect("source fixture opens");
    let plan = plan_for(&source);

    let existing_error = write_nvfp4_safetensors(&source, &plan, &existing_path.0)
        .expect_err("the writer never overwrites an existing file");
    assert!(matches!(
        existing_error,
        Nvfp4WriterError::DestinationExists { .. }
    ));
    assert_eq!(
        fs::read(&existing_path.0).expect("existing destination remains"),
        existing_before
    );

    let in_place_error = write_nvfp4_safetensors(&source, &plan, &source_path.0)
        .expect_err("the writer refuses in-place output");
    assert!(matches!(
        in_place_error,
        Nvfp4WriterError::SourceDestinationConflict { .. }
    ));
}
