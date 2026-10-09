//! Group-wise low-bit SafeTensors files (ADR 0030) at the library level: plan,
//! write, reopen, decode, and validate, with no CLI involved.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::id,
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::{
    io::{
        layout::plan_output_layout_for,
        lowbit::{encoding_for, read_manifest, validate_outputs, write_lowbit_safetensors},
        safetensors::{MappedSafetensors, TensorSource},
        writer::WriterError,
    },
    quant::{
        lowbit::{LowBitConfig, Scheme},
        policy::{QuantizationPolicy, TensorCandidate, TensorKind},
    },
};
use serde_json::{Map, Value, json};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let serial = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("modelq-lowbit-{label}-{}-{serial}", id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
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

/// Writes a SafeTensors file from (name, dtype, shape, values); BF16 values are
/// stored as the high halves of their F32 bit patterns.
fn write_source(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<f32>)]) {
    let mut header = Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, values) in tensors {
        let begin = data.len();
        for &value in values {
            match *dtype {
                "F32" => data.extend_from_slice(&value.to_le_bytes()),
                "BF16" => data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes()),
                other => panic!("unsupported test dtype {other}"),
            }
        }
        header.insert(
            (*name).to_owned(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [begin, data.len()]}),
        );
    }
    let mut json = serde_json::to_vec(&Value::Object(header)).unwrap();
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&json);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes).unwrap();
}

fn values(count: usize, seed: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as f32) * 0.173 + seed).sin() * (1.0 + seed / 3.0))
        .collect()
}

/// Two quantizable matrices (F32 and BF16), a preserved vector, and a small matrix
/// the policy keeps as it is.
fn fixture(path: &Path) {
    write_source(
        path,
        &[
            (
                "model.layers.0.weight",
                "F32",
                vec![64, 128],
                values(64 * 128, 1.0),
            ),
            ("model.layers.0.bias", "F32", vec![128], values(128, 2.0)),
            (
                "model.layers.1.weight",
                "BF16",
                vec![32, 128],
                values(32 * 128, 3.0),
            ),
            ("model.norm", "F32", vec![4, 4], values(16, 4.0)),
        ],
    );
}

fn decisions_for(source: &MappedSafetensors) -> Vec<modelq::quant::policy::TensorDecision> {
    let candidates = source.tensor_summaries().into_iter().map(|summary| {
        let kind = match summary.dtype.as_str() {
            "F32" | "F16" | "BF16" => TensorKind::Floating,
            _ => TensorKind::NonFloating,
        };
        let count: usize = summary.shape.iter().product();
        TensorCandidate::new(summary.name, count, kind)
    });
    QuantizationPolicy::default().decide_all(candidates)
}

fn config(bits: u8, group_size: usize, scheme: Scheme) -> LowBitConfig {
    LowBitConfig {
        bits,
        group_size,
        scheme,
    }
}

fn write_output(
    source_path: &Path,
    output: &Path,
    config: LowBitConfig,
) -> Result<(), WriterError> {
    let source = MappedSafetensors::open(source_path).unwrap();
    let decisions = decisions_for(&source);
    let plan = plan_output_layout_for(
        &source.tensor_summaries(),
        &decisions,
        encoding_for(&config),
    )
    .unwrap();
    write_lowbit_safetensors(&source, &plan, &decisions, config, output)
}

#[test]
fn every_configuration_writes_decodes_and_validates() {
    let dir = TestDir::new("validate");
    let source_path = dir.join("source.safetensors");
    fixture(&source_path);
    let source = MappedSafetensors::open(&source_path).unwrap();

    for config in [
        config(4, 128, Scheme::Symmetric),
        config(3, 32, Scheme::Symmetric),
        config(2, 7, Scheme::Symmetric),
        config(1, 64, Scheme::Sign),
    ] {
        let output = dir.join(&format!(
            "out-{}-{}.safetensors",
            config.bits, config.group_size
        ));
        write_output(&source_path, &output, config).unwrap();

        let reopened = MappedSafetensors::open(&output).unwrap();
        let manifest = read_manifest(reopened.metadata())
            .unwrap()
            .expect("a low-bit file");
        assert_eq!(manifest.config, config);
        assert_eq!(
            manifest.tensors.len(),
            4,
            "every source tensor is listed once"
        );

        let report = validate_outputs(&source, std::slice::from_ref(&output)).unwrap();
        assert_eq!(report.quantized_tensors, 2, "{config:?}");
        assert_eq!(report.preserved_tensors, 2, "{config:?}");
        assert!(report.max_abs_error.is_finite(), "{config:?}");
        assert!(report.lowest_sqnr_db.is_some(), "{config:?}");
        if config.scheme == Scheme::Symmetric {
            assert_eq!(
                report.scale_bound_violations, 0,
                "symmetric rounding keeps every error within half a scale ({config:?})"
            );
        }
    }
}

#[test]
fn metadata_describes_the_encoding_and_the_manifest_is_canonical() {
    let dir = TestDir::new("metadata");
    let source_path = dir.join("source.safetensors");
    fixture(&source_path);
    let output = dir.join("out.safetensors");
    write_output(&source_path, &output, config(4, 128, Scheme::Symmetric)).unwrap();

    let reopened = MappedSafetensors::open(&output).unwrap();
    let metadata = reopened.metadata();
    assert_eq!(metadata["modelq.format"], "modelq-native");
    assert_eq!(metadata["modelq.format_version"], "2");
    assert_eq!(metadata["modelq.quantization"], "int4");
    assert_eq!(metadata["modelq.scheme"], "symmetric-group-wise");
    assert_eq!(metadata["modelq.packing"], "lsb-first-bitstream");
    assert_eq!(metadata["modelq.qmax"], "7");
    assert_eq!(metadata["modelq.qmin"], "-7");
    assert_eq!(metadata["modelq.group_size"], "128");

    let manifest: Value = serde_json::from_str(&metadata["modelq.manifest"]).unwrap();
    assert_eq!(manifest["schema"], "modelq.lowbit.manifest.v1");
    let weight = &manifest["tensors"]["model.layers.0.weight"];
    assert_eq!(weight["action"], "quantized");
    assert_eq!(weight["qdata_name"], "model.layers.0.weight.qdata");
    assert_eq!(weight["qdata_dtype"], "U8");
    assert_eq!(weight["qdata_shape"], json!([64 * 128 / 2]));
    assert_eq!(weight["scale_name"], "model.layers.0.weight.scale");
    assert_eq!(weight["scale_dtype"], "F32");
    assert_eq!(weight["scale_shape"], json!([64 * 128 / 128]));
    assert_eq!(weight["elements"], 64 * 128);
    let bias = &manifest["tensors"]["model.layers.0.bias"];
    assert_eq!(bias["action"], "preserved");
}

#[test]
fn the_same_input_gives_the_same_bytes() {
    let dir = TestDir::new("determinism");
    let source_path = dir.join("source.safetensors");
    fixture(&source_path);
    let first = dir.join("first.safetensors");
    let second = dir.join("second.safetensors");
    let configuration = config(3, 32, Scheme::Symmetric);
    write_output(&source_path, &first, configuration).unwrap();
    write_output(&source_path, &second, configuration).unwrap();
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
}

#[test]
fn an_existing_destination_is_never_replaced() {
    let dir = TestDir::new("overwrite");
    let source_path = dir.join("source.safetensors");
    fixture(&source_path);
    let output = dir.join("out.safetensors");
    fs::write(&output, b"previous artifact").unwrap();
    let error = write_output(&source_path, &output, config(4, 128, Scheme::Symmetric)).unwrap_err();
    assert!(
        matches!(error, WriterError::DestinationExists { .. }),
        "{error:?}"
    );
    assert_eq!(fs::read(&output).unwrap(), b"previous artifact");
}

#[test]
fn a_plan_for_another_encoding_is_refused() {
    let dir = TestDir::new("plan");
    let source_path = dir.join("source.safetensors");
    fixture(&source_path);
    let source = MappedSafetensors::open(&source_path).unwrap();
    let decisions = decisions_for(&source);
    let plan = plan_output_layout_for(
        &source.tensor_summaries(),
        &decisions,
        encoding_for(&config(4, 128, Scheme::Symmetric)),
    )
    .unwrap();
    let error = write_lowbit_safetensors(
        &source,
        &plan,
        &decisions,
        config(4, 64, Scheme::Symmetric),
        dir.join("out.safetensors"),
    )
    .unwrap_err();
    assert!(matches!(error, WriterError::PlanMismatch), "{error:?}");
}

#[test]
fn a_file_that_is_not_low_bit_is_not_read_as_one() {
    let dir = TestDir::new("not-lowbit");
    let source_path = dir.join("source.safetensors");
    fixture(&source_path);
    let source = MappedSafetensors::open(&source_path).unwrap();
    assert!(read_manifest(source.metadata()).unwrap().is_none());
}

#[test]
fn validation_rejects_a_source_it_cannot_match() {
    let dir = TestDir::new("mismatch");
    let source_path = dir.join("source.safetensors");
    fixture(&source_path);
    let output = dir.join("out.safetensors");
    write_output(&source_path, &output, config(4, 128, Scheme::Symmetric)).unwrap();

    // A different source with the same names but changed weights: the preserved
    // bias no longer matches the output byte for byte.
    let other_path = dir.join("other.safetensors");
    write_source(
        &other_path,
        &[
            (
                "model.layers.0.weight",
                "F32",
                vec![64, 128],
                values(64 * 128, 1.0),
            ),
            ("model.layers.0.bias", "F32", vec![128], values(128, 9.0)),
            (
                "model.layers.1.weight",
                "BF16",
                vec![32, 128],
                values(32 * 128, 3.0),
            ),
            ("model.norm", "F32", vec![4, 4], values(16, 4.0)),
        ],
    );
    let other = MappedSafetensors::open(&other_path).unwrap();
    let error = validate_outputs(&other, &[output]).unwrap_err();
    assert!(error.contains("changed during writing"), "{error}");
}
