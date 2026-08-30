//! Opt-in differential comparison with a Transformer Engine NVFP4 fixture.
//!
//! The fixture is intentionally not checked into this repository: capturing it
//! requires a supported CUDA/Blackwell host.  Run this test explicitly with
//! `MODELQ_NVFP4_REFERENCE_FIXTURE=/path/to/fixture.json cargo test
//! --test nvfp4_reference -- --ignored` after capture.

use std::{fs, path::PathBuf};

use modelq::quant::nvfp4;
use serde_json::Value;

const SCHEMA: &str = "modelq.nvfp4.reference.v1";

fn object_field<'a>(object: &'a Value, field: &str) -> &'a Value {
    object
        .get(field)
        .unwrap_or_else(|| panic!("fixture is missing field {field:?}"))
}

fn string_field<'a>(object: &'a Value, field: &str) -> &'a str {
    object_field(object, field)
        .as_str()
        .unwrap_or_else(|| panic!("fixture field {field:?} must be a string"))
}

fn bool_field(object: &Value, field: &str) -> bool {
    object_field(object, field)
        .as_bool()
        .unwrap_or_else(|| panic!("fixture field {field:?} must be a boolean"))
}

fn usize_array(object: &Value, field: &str) -> Vec<usize> {
    object_field(object, field)
        .as_array()
        .unwrap_or_else(|| panic!("fixture field {field:?} must be an array"))
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or_else(|| {
                    panic!("fixture field {field:?}[{index}] must be a non-negative usize")
                })
        })
        .collect()
}

fn u8_array(object: &Value, field: &str) -> Vec<u8> {
    object_field(object, field)
        .as_array()
        .unwrap_or_else(|| panic!("fixture field {field:?} must be an array"))
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|value| u8::try_from(value).ok())
                .unwrap_or_else(|| panic!("fixture field {field:?}[{index}] must be a byte"))
        })
        .collect()
}

fn f32_bits_array(object: &Value, field: &str) -> Vec<u32> {
    object_field(object, field)
        .as_array()
        .unwrap_or_else(|| panic!("fixture field {field:?} must be an array"))
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or_else(|| {
                    panic!("fixture field {field:?}[{index}] must be an F32 bit pattern")
                })
        })
        .collect()
}

#[test]
#[ignore = "requires a captured Transformer Engine fixture from a Blackwell host"]
fn matches_transformer_engine_reference_fixture() {
    let path = std::env::var_os("MODELQ_NVFP4_REFERENCE_FIXTURE").unwrap_or_else(|| {
        panic!(
            "MODELQ_NVFP4_REFERENCE_FIXTURE is not set; capture a fixture before running this ignored test"
        )
    });
    let path = PathBuf::from(path);
    let document: Value = serde_json::from_slice(
        &fs::read(&path)
            .unwrap_or_else(|error| panic!("could not read {}: {error}", path.display())),
    )
    .unwrap_or_else(|error| panic!("fixture {} is not valid JSON: {error}", path.display()));

    assert_eq!(string_field(&document, "schema"), SCHEMA);

    let producer = object_field(&document, "producer");
    assert_eq!(string_field(producer, "name"), "transformer-engine");
    for field in ["version", "commit", "pytorch", "cuda", "device"] {
        assert!(
            !string_field(producer, field).trim().is_empty(),
            "producer.{field} must identify the external reference"
        );
    }
    let capability = string_field(producer, "compute_capability");
    let major = capability
        .split_once('.')
        .and_then(|(major, _minor)| major.parse::<u32>().ok())
        .expect("producer.compute_capability must be formatted as MAJOR.MINOR");
    assert!(
        major >= 10,
        "producer.compute_capability must identify a Blackwell-class GPU"
    );

    let recipe = object_field(&document, "recipe");
    assert_eq!(string_field(recipe, "quantization_dim"), "1x16");
    assert!(bool_field(recipe, "deterministic"));
    assert!(bool_field(recipe, "disable_2d_quantization"));
    assert!(bool_field(recipe, "disable_rht"));
    assert!(bool_field(recipe, "disable_stochastic_rounding"));
    assert_eq!(string_field(recipe, "nvfp4_4over6"), "none");

    let representation = object_field(&document, "representation");
    assert_eq!(string_field(representation, "fp4"), "E2M1");
    assert_eq!(string_field(representation, "block_scale"), "E4M3");
    assert_eq!(
        object_field(representation, "block_size").as_u64(),
        Some(16)
    );
    assert_eq!(string_field(representation, "global_scale"), "F32_decode");
    assert_eq!(string_field(representation, "packing"), "low_nibble_first");

    let tensor = object_field(&document, "tensor");
    assert_eq!(string_field(tensor, "source_dtype"), "F32");
    let shape = usize_array(tensor, "shape");
    assert!(!shape.is_empty());
    assert!(shape.iter().all(|&dimension| dimension > 0));
    assert_eq!(shape.last().copied().unwrap() % nvfp4::BLOCK_SIZE, 0);
    let elements = shape
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
        .expect("fixture shape element count must fit usize");

    let source_bits = f32_bits_array(tensor, "source_values_f32_bits");
    assert_eq!(source_bits.len(), elements);
    let source = source_bits
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect::<Vec<_>>();
    assert!(source.iter().all(|value| value.is_finite()));

    let packed = u8_array(tensor, "packed_u8");
    let block_scales = u8_array(tensor, "block_scale_u8");
    let global_scale_bits = object_field(tensor, "global_scale_f32_bits")
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .expect("global_scale_f32_bits must be a u32");
    let expected_dequant_bits = f32_bits_array(tensor, "expected_dequant_f32_bits");
    assert_eq!(expected_dequant_bits.len(), elements);

    let quantized = nvfp4::quantize_shaped(&source, &shape)
        .unwrap_or_else(|error| panic!("ModelQ rejected fixture source: {error}"));
    assert_eq!(
        quantized.packed_values(),
        packed.as_slice(),
        "packed FP4 bytes differ"
    );
    assert_eq!(
        quantized.block_scales(),
        block_scales.as_slice(),
        "E4M3 block-scale bytes differ"
    );
    assert_eq!(
        quantized.global_scale().to_bits(),
        global_scale_bits,
        "tensor-wide F32 decode scale differs"
    );

    let reconstructed = quantized
        .dequantize()
        .unwrap_or_else(|error| panic!("ModelQ could not reconstruct fixture: {error}"));
    let reconstructed_bits = reconstructed
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(
        reconstructed_bits, expected_dequant_bits,
        "reconstructed F32 values differ"
    );
}
