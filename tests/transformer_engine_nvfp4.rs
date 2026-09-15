//! Contract tests for the CPU-only Transformer Engine NVFP4 rowwise profile.

use modelq_io::transformer_engine::{
    TRANSFORMER_ENGINE_NVFP4_PROFILE, TransformerEngineNvfp4Error, export_transformer_engine_nvfp4,
};
use modelq_quant::nvfp4::quantize_shaped;

#[test]
fn exports_rowwise_data_and_aligned_scale_matrix() {
    let source = (0..64).map(|index| index as f32 - 32.0).collect::<Vec<_>>();
    let native = quantize_shaped(&source, &[2, 32]).expect("native shape is valid");
    let exported = export_transformer_engine_nvfp4("layer.weight", &[2, 32], &native)
        .expect("the profile accepts a complete rowwise tensor");

    assert_eq!(
        TRANSFORMER_ENGINE_NVFP4_PROFILE,
        "transformer-engine.nvfp4.rowwise.1x16.v1"
    );
    assert_eq!(exported.source_shape, vec![2, 32]);
    assert_eq!(exported.rowwise_data_shape, vec![2, 16]);
    assert_eq!(exported.rowwise_scale_inv_shape, vec![128, 4]);
    assert_eq!(exported.rowwise_data, native.packed_values());
    assert_eq!(
        &exported.rowwise_scale_inv[0..2],
        &native.block_scales()[0..2]
    );
    assert_eq!(&exported.rowwise_scale_inv[2..4], &[0, 0]);
    assert_eq!(
        &exported.rowwise_scale_inv[4..6],
        &native.block_scales()[2..4]
    );
    assert!(
        exported.rowwise_scale_inv[6..]
            .iter()
            .all(|&byte| byte == 0)
    );
    assert!((exported.amax_rowwise - native.global_scale() * (448.0 * 6.0)).abs() < 1e-6);
    assert_eq!(exported.rowwise_data_name(), "layer.weight.rowwise_data");
    assert_eq!(
        exported.rowwise_scale_inv_name(),
        "layer.weight.rowwise_scale_inv"
    );
    assert_eq!(exported.amax_rowwise_name(), "layer.weight.amax_rowwise");
}

#[test]
fn repeated_exports_are_equal() {
    let source = [
        0.0_f32, 1.0, -2.0, 3.0, 4.0, -5.0, 6.0, -1.0, 0.5, -0.75, 1.25, -1.5, 2.25, -2.5, 3.5,
        -4.0, 2.0, -3.0, 4.0, -5.0, 6.0, -0.5, 0.75, -1.25, 1.5, -2.25, 2.5, -3.5, 4.0, -4.5, 5.0,
        -6.0,
    ];
    let native = quantize_shaped(&source, &[2, 16]).expect("native shape is valid");
    let first = export_transformer_engine_nvfp4("weight", &[2, 16], &native)
        .expect("first export succeeds");
    let second = export_transformer_engine_nvfp4("weight", &[2, 16], &native)
        .expect("second export succeeds");
    assert_eq!(first, second);
}

#[test]
fn rejects_invalid_names_and_shapes() {
    let native = quantize_shaped(&[0.0_f32; 32], &[2, 16]).expect("native shape is valid");
    for name in ["", "__metadata__"] {
        let error = export_transformer_engine_nvfp4(name, &[2, 16], &native)
            .expect_err("profile names must be usable tensor names");
        assert!(matches!(
            error,
            TransformerEngineNvfp4Error::InvalidName { .. }
        ));
    }
    for shape in [vec![], vec![32], vec![2, 0], vec![2, 8]] {
        let error = export_transformer_engine_nvfp4("weight", &shape, &native)
            .expect_err("the rowwise profile requires a matrix with 16-wide blocks");
        assert!(matches!(
            error,
            TransformerEngineNvfp4Error::InvalidShape { .. }
        ));
    }
}

#[test]
fn rejects_shape_payload_mismatch() {
    let native = quantize_shaped(&[0.0_f32; 32], &[2, 16]).expect("native shape is valid");
    let error = export_transformer_engine_nvfp4("weight", &[2, 32], &native)
        .expect_err("the native payload must describe the requested shape");
    assert!(matches!(
        error,
        TransformerEngineNvfp4Error::PayloadLengthMismatch { .. }
    ));
}

#[test]
fn exports_zero_amax_for_an_all_zero_tensor() {
    let native = quantize_shaped(&[0.0_f32; 32], &[2, 16]).expect("native shape is valid");
    let exported = export_transformer_engine_nvfp4("weight", &[2, 16], &native)
        .expect("zero tensors have an explicit profile representation");
    assert_eq!(exported.amax_rowwise, 0.0);
    assert!(exported.rowwise_data.iter().all(|&byte| byte == 0));
    assert!(exported.rowwise_scale_inv.iter().all(|&byte| byte == 0));
}
