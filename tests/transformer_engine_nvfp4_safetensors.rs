use std::{
    fs,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::io::{
    safetensors::MappedSafetensors,
    transformer_engine::{TransformerEngineNvfp4Tensor, export_transformer_engine_nvfp4},
    writer::{WriterError, write_transformer_engine_nvfp4_safetensors},
};
use modelq::quant::nvfp4::quantize_shaped;
use serde_json::{Value, json};

static NEXT_TEST_PATH: AtomicU64 = AtomicU64::new(0);

struct TempArtifact(PathBuf);

impl TempArtifact {
    fn new(stem: &str) -> Self {
        let id = NEXT_TEST_PATH.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "modelq-te-nvfp4-{stem}-{}-{id}.safetensors",
            process::id()
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempArtifact {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn profile_fixture(name: &str) -> TransformerEngineNvfp4Tensor {
    let values = (0..64).map(|i| i as f32 - 32.0).collect::<Vec<_>>();
    let native = quantize_shaped(&values, &[2, 32]).expect("fixture quantizes");
    export_transformer_engine_nvfp4(name, &[2, 32], &native).expect("profile exports")
}

#[test]
fn writes_transformer_engine_profile_with_exact_manifest_and_payloads() {
    let output = TempArtifact::new("roundtrip");
    let profile = profile_fixture("layer.weight");
    write_transformer_engine_nvfp4_safetensors(&profile, output.path())
        .expect("valid rowwise profile writes");
    let file = MappedSafetensors::open(output.path()).expect("output is valid SafeTensors");

    assert_eq!(
        file.metadata().get("modelq.format").map(String::as_str),
        Some("transformer-engine-nvfp4-safetensors-v1")
    );
    assert_eq!(file.metadata().len(), 2);
    let manifest: Value = serde_json::from_str(
        file.metadata()
            .get("modelq.manifest")
            .expect("manifest exists"),
    )
    .expect("manifest is JSON");
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(
        manifest["profile_id"],
        "transformer-engine.nvfp4.rowwise.1x16.v1"
    );
    assert_eq!(manifest["runtime"]["name"], "transformer-engine");
    assert_eq!(manifest["runtime"]["version"], "2.19.0");
    assert_eq!(manifest["tensor_name"], "layer.weight");
    assert_eq!(manifest["logical_shape"], json!([2, 32]));
    assert_eq!(
        manifest["fields"]["rowwise_data"],
        "layer.weight.rowwise_data"
    );
    assert_eq!(
        manifest["fields"]["rowwise_scale_inv"],
        "layer.weight.rowwise_scale_inv"
    );
    assert_eq!(
        manifest["fields"]["amax_rowwise"],
        "layer.weight.amax_rowwise"
    );
    assert_eq!(manifest["encoding"]["values"], "E2M1");
    assert_eq!(manifest["encoding"]["scales"], "E4M3");
    assert_eq!(manifest["encoding"]["block_size"], 16);
    assert_eq!(
        manifest["encoding"]["scaling"],
        "rowwise_1x16_tensor_global"
    );
    assert_eq!(manifest["scale_padding"]["shape"], json!([128, 4]));
    assert_eq!(manifest["scale_padding"]["gemm_swizzled"], false);
    assert_eq!(manifest["global_scale_denominator"], 2688.0);

    let tensors = file.tensors().collect::<Vec<_>>();
    assert_eq!(tensors.len(), 3);
    let actual = tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>();
    assert_eq!(
        actual,
        [
            "layer.weight.amax_rowwise",
            "layer.weight.rowwise_data",
            "layer.weight.rowwise_scale_inv"
        ]
    );
    assert_eq!(tensors[0].dtype, "F32");
    assert_eq!(tensors[0].shape, [1]);
    assert_eq!(tensors[1].dtype, "U8");
    assert_eq!(tensors[1].shape, [2, 16]);
    assert_eq!(tensors[2].dtype, "U8");
    assert_eq!(tensors[2].shape, [128, 4]);
    assert_eq!(
        file.tensor_bytes("layer.weight.rowwise_data").unwrap(),
        profile.rowwise_data
    );
    assert_eq!(
        file.tensor_bytes("layer.weight.rowwise_scale_inv").unwrap(),
        profile.rowwise_scale_inv
    );
    let scale = file.tensor_bytes("layer.weight.rowwise_scale_inv").unwrap();
    assert_eq!(&scale[0..2], &profile.rowwise_scale_inv[0..2]);
    assert_eq!(&scale[4..6], &profile.rowwise_scale_inv[4..6]);
    for row in 0..128 {
        for col in 0..4 {
            if row >= 2 || col >= 2 {
                assert_eq!(scale[row * 4 + col], 0, "padding at ({row}, {col})");
            }
        }
    }
    assert_eq!(
        file.tensor_bytes("layer.weight.amax_rowwise").unwrap(),
        profile.amax_rowwise.to_le_bytes()
    );
}

#[test]
fn writes_identical_profile_bytes_deterministically() {
    let profile = profile_fixture("layer.weight");
    let first = TempArtifact::new("deterministic-a");
    let second = TempArtifact::new("deterministic-b");
    write_transformer_engine_nvfp4_safetensors(&profile, first.path()).unwrap();
    write_transformer_engine_nvfp4_safetensors(&profile, second.path()).unwrap();
    assert_eq!(
        fs::read(first.path()).unwrap(),
        fs::read(second.path()).unwrap()
    );
}

fn assert_invalid(mut profile: TransformerEngineNvfp4Tensor) {
    let output = TempArtifact::new("invalid");
    assert!(matches!(
        write_transformer_engine_nvfp4_safetensors(&profile, output.path()),
        Err(WriterError::InvalidTransformerEngineNvfp4Tensor { .. })
    ));
    assert!(!output.path().exists());
    // Ensure mutability is exercised at the call site rather than hidden in a helper.
    profile.name.shrink_to_fit();
}

#[test]
fn rejects_each_invalid_public_profile_boundary_without_creating_destination() {
    let mut p = profile_fixture("weight");
    p.name.clear();
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.source_shape = vec![32];
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.source_shape = vec![0, 32];
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.source_shape = vec![2, 24];
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.source_shape = vec![usize::MAX, 32];
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.source_shape = vec![1, usize::MAX - 15];
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.rowwise_data_shape = vec![1, 32];
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.rowwise_data.pop();
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.rowwise_scale_inv_shape = vec![1, 1];
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.rowwise_scale_inv.pop();
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.rowwise_scale_inv[8] = 1;
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.amax_rowwise = f32::INFINITY;
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.amax_rowwise = -1.0;
    assert_invalid(p);
    let mut p = profile_fixture("weight");
    p.amax_rowwise = 0.0;
    assert_invalid(p);
    let mut zero = profile_fixture("weight");
    zero.rowwise_scale_inv.fill(0);
    zero.amax_rowwise = 1.0;
    assert_invalid(zero);
}

#[test]
fn rejects_reserved_name_and_preserves_existing_destination() {
    let mut malformed = profile_fixture("weight");
    malformed.name = "__metadata__".to_owned();
    assert_invalid(malformed);

    let existing = TempArtifact::new("existing");
    fs::write(existing.path(), b"preserve me").expect("sentinel writes");
    assert!(matches!(
        write_transformer_engine_nvfp4_safetensors(&profile_fixture("weight"), existing.path()),
        Err(WriterError::DestinationExists { .. })
    ));
    assert_eq!(fs::read(existing.path()).unwrap(), b"preserve me");
}
