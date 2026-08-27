use std::{
    fs,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::{
    nvfp4::{
        Nvfp4ReaderError, plan_nvfp4_output, read_nvfp4_safetensors,
        write_nvfp4_safetensors,
    },
    safetensors::MappedSafetensors,
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
        "modelq-nvfp4-reader-{stem}-{}-{id}.safetensors",
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
        "__metadata__": { "fixture": "task-25" },
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
    let padded_len = header.len().div_ceil(8) * 8;
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

fn rewrite_header(path: &Path, mutate: impl FnOnce(&mut Value)) {
    let bytes = fs::read(path).expect("output bytes can be read");
    let old_header_len = usize::try_from(u64::from_le_bytes(
        bytes[..8].try_into().expect("header has a length prefix"),
    ))
    .expect("header length fits usize");
    let mut header: Value = serde_json::from_slice(&bytes[8..8 + old_header_len])
        .expect("output header is JSON");
    mutate(&mut header);
    let mut new_header = serde_json::to_vec(&header).expect("mutated header serializes");
    let new_header_len = new_header.len().div_ceil(8) * 8;
    new_header.resize(new_header_len, b' ');

    let mut rewritten = Vec::with_capacity(bytes.len() + new_header_len - old_header_len);
    rewritten.extend_from_slice(&(new_header_len as u64).to_le_bytes());
    rewritten.extend_from_slice(&new_header);
    rewritten.extend_from_slice(&bytes[8 + old_header_len..]);
    fs::write(path, rewritten).expect("mutated output can be written");
}

fn rewrite_payload_byte(path: &Path, data_offset: usize, value: u8) {
    let mut bytes = fs::read(path).expect("output bytes can be read");
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes[..8].try_into().expect("header has a length prefix"),
    ))
    .expect("header length fits usize");
    bytes[8 + header_len + data_offset] = value;
    fs::write(path, bytes).expect("mutated payload can be written");
}

fn write_fixture_output(stem: &str) -> (CleanupPath, CleanupPath) {
    let values = (0..16).map(|index| index as f32 - 7.5).collect::<Vec<_>>();
    let source_path = source_fixture(&format!("{stem}-source"), &values);
    let destination_path = unique_path(&format!("{stem}-output"));
    let source = MappedSafetensors::open(&source_path.0).expect("source fixture opens");
    let plan = plan_nvfp4_output(&source.inspection().tensors, &["weight".to_owned()])
        .expect("fixture shape is valid for NVFP4");
    write_nvfp4_safetensors(&source, &plan, &destination_path.0)
        .expect("native NVFP4 output writes");
    (source_path, destination_path)
}

#[test]
fn reads_native_nvfp4_output_and_reconstructs_values() {
    let values = (0..16).map(|index| index as f32 - 7.5).collect::<Vec<_>>();
    let source_path = source_fixture("round-trip-source", &values);
    let destination_path = unique_path("round-trip-output");
    let source = MappedSafetensors::open(&source_path.0).expect("source fixture opens");
    let plan = plan_nvfp4_output(
        &source.inspection().tensors,
        &["weight".to_owned()],
    )
    .expect("fixture shape is valid for NVFP4");
    write_nvfp4_safetensors(&source, &plan, &destination_path.0)
        .expect("native NVFP4 output writes");

    let output = MappedSafetensors::open(&destination_path.0).expect("output mapping opens");
    let decoded = read_nvfp4_safetensors(&output).expect("native NVFP4 output reads");

    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].name, "weight");
    assert_eq!(decoded[0].original_dtype, "F32");
    assert_eq!(decoded[0].original_shape, [1, 16]);
    let expected = reference_nvfp4::quantize_shaped(&values, &[1, 16])
        .expect("reference quantizer accepts the fixture")
        .dequantize()
        .expect("reference tensor dequantizes");
    assert_eq!(decoded[0].values, expected);
}

#[test]
fn rejects_files_without_nvfp4_metadata() {
    let values = (0..16).map(|index| index as f32 - 7.5).collect::<Vec<_>>();
    let source_path = source_fixture("missing-metadata-source", &values);
    let source = MappedSafetensors::open(&source_path.0).expect("source fixture opens");

    let error = read_nvfp4_safetensors(&source).expect_err("a native envelope is required");
    assert!(matches!(
        error,
        Nvfp4ReaderError::MissingMetadata { ref key } if key == "modelq.format"
    ));
}

#[test]
fn rejects_wrong_quantization_metadata() {
    let (_source_path, destination_path) = write_fixture_output("wrong-quantization");
    rewrite_header(&destination_path.0, |header| {
        header["__metadata__"]["modelq.quantization"] = Value::String("int8".to_owned());
    });
    let output = MappedSafetensors::open(&destination_path.0).expect("container remains valid");

    let error = read_nvfp4_safetensors(&output).expect_err("the quantization tag must be NVFP4");
    assert!(matches!(
        error,
        Nvfp4ReaderError::MetadataMismatch { ref key, .. } if key == "modelq.quantization"
    ));
}

#[test]
fn rejects_unknown_manifest_schema() {
    let (_source_path, destination_path) = write_fixture_output("unknown-schema");
    rewrite_header(&destination_path.0, |header| {
        let manifest_text = header["__metadata__"]["modelq.manifest"]
            .as_str()
            .expect("manifest is a string");
        let mut manifest: Value = serde_json::from_str(manifest_text).expect("manifest is JSON");
        manifest["schema"] = Value::String("modelq.nvfp4.manifest.v999".to_owned());
        header["__metadata__"]["modelq.manifest"] = Value::String(
            serde_json::to_string(&manifest).expect("mutated manifest serializes"),
        );
    });
    let output = MappedSafetensors::open(&destination_path.0).expect("container remains valid");

    let error = read_nvfp4_safetensors(&output).expect_err("unknown schemas must be rejected");
    assert!(matches!(error, Nvfp4ReaderError::InvalidManifest { .. }));
}

#[test]
fn rejects_manifest_physical_shape_mismatch() {
    let (_source_path, destination_path) = write_fixture_output("shape-mismatch");
    rewrite_header(&destination_path.0, |header| {
        let manifest_text = header["__metadata__"]["modelq.manifest"]
            .as_str()
            .expect("manifest is a string");
        let mut manifest: Value = serde_json::from_str(manifest_text).expect("manifest is JSON");
        manifest["tensors"]["weight"]["qdata_shape"] = json!([1, 4]);
        header["__metadata__"]["modelq.manifest"] = Value::String(
            serde_json::to_string(&manifest).expect("mutated manifest serializes"),
        );
    });
    let output = MappedSafetensors::open(&destination_path.0).expect("container remains valid");

    let error = read_nvfp4_safetensors(&output).expect_err("derived shapes are authoritative");
    assert!(matches!(
        error,
        Nvfp4ReaderError::TensorMismatch { ref field, .. } if field == "qdata_shape"
    ));
}

#[test]
fn rejects_invalid_block_scale_payload() {
    let (_source_path, destination_path) = write_fixture_output("invalid-block-scale");
    // The preserved two-byte `ids` tensor comes first, followed by eight qdata
    // bytes.  The first block-scale byte is therefore data offset ten.
    rewrite_payload_byte(&destination_path.0, 10, 0);
    let output = MappedSafetensors::open(&destination_path.0).expect("container remains valid");

    let error = read_nvfp4_safetensors(&output).expect_err("zero scale cannot hide nonzero data");
    assert!(matches!(error, Nvfp4ReaderError::Quantization { .. }));
}

#[test]
fn rejects_manifest_references_to_missing_preserved_tensor() {
    let (_source_path, destination_path) = write_fixture_output("missing-preserved");
    rewrite_header(&destination_path.0, |header| {
        let manifest_text = header["__metadata__"]["modelq.manifest"]
            .as_str()
            .expect("manifest is a string");
        let mut manifest: Value = serde_json::from_str(manifest_text).expect("manifest is JSON");
        manifest["tensors"]["ids"]["tensor_name"] = Value::String("missing".to_owned());
        header["__metadata__"]["modelq.manifest"] = Value::String(
            serde_json::to_string(&manifest).expect("mutated manifest serializes"),
        );
    });
    let output = MappedSafetensors::open(&destination_path.0).expect("container remains valid");

    let error = read_nvfp4_safetensors(&output).expect_err("preserved references must resolve");
    assert!(matches!(error, Nvfp4ReaderError::InvalidManifest { .. }));
}
