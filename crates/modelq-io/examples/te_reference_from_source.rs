//! Write the F32 reference for a real checkpoint's Transformer Engine export.
//!
//! Usage: `te_reference_from_source <source.safetensors> <reference.safetensors>`
//!
//! Applies the same Transformer Engine eligibility policy as
//! `modelq quantize --format nvfp4-te`, re-quantizes every selected matrix
//! with the native NVFP4 quantizer, and stores its dequantization as
//! `<name>.dequantized_reference` (F32, `[M, K]`) with
//! `modelq.reference_schema = transformer-engine-nvfp4-reference-v2`.
//!
//! The reference is derived from the *source* weights and the native
//! quantizer, not from the exported container, so it is an oracle independent
//! of the container writer and of the container decoders.  It is a test-only
//! artifact and is never part of a runtime container.  Peak memory is one
//! matrix, not the checkpoint.

use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use modelq_io::safetensors::MappedSafetensors;
use modelq_quant::{
    nvfp4::quantize_shaped,
    nvfp4_policy::{Nvfp4Candidate, Nvfp4Policy},
};
use serde_json::{Map, Value, json};

const REFERENCE_SCHEMA: &str = "transformer-engine-nvfp4-reference-v2";

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let usage = "usage: te_reference_from_source <source.safetensors> <reference.safetensors>";
    let source_path = arguments.next().ok_or(usage)?;
    let reference_path = arguments.next().ok_or(usage)?;
    if arguments.next().is_some() {
        return Err(usage.into());
    }
    if Path::new(&reference_path).exists() {
        return Err(format!("{reference_path} already exists").into());
    }

    let source = MappedSafetensors::open(&source_path)?;
    let policy = Policy::new();
    let selected: Vec<(String, Vec<usize>)> = source
        .tensors()
        .filter(|summary| policy.selects(&summary.name, &summary.dtype, &summary.shape))
        .map(|summary| (summary.name.clone(), summary.shape.clone()))
        .collect();
    let mut selected = selected;
    selected.sort();

    // The header is fully determined by the shapes, so write it first and
    // then stream one matrix at a time.
    let mut header = Map::new();
    let mut metadata = Map::new();
    metadata.insert(
        "modelq.reference_schema".to_owned(),
        Value::String(REFERENCE_SCHEMA.to_owned()),
    );
    header.insert("__metadata__".to_owned(), Value::Object(metadata));
    let mut offset = 0_u64;
    for (name, shape) in &selected {
        let bytes = shape.iter().product::<usize>() as u64 * 4;
        header.insert(
            format!("{name}.dequantized_reference"),
            json!({ "dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes] }),
        );
        offset += bytes;
    }
    let mut header = serde_json::to_vec(&Value::Object(header))?;
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut writer = BufWriter::new(File::create(&reference_path)?);
    writer.write_all(&(header.len() as u64).to_le_bytes())?;
    writer.write_all(&header)?;

    for (name, shape) in &selected {
        let values: Vec<f32> = source.tensor(name)?.values().collect();
        let dequantized = quantize_shaped(&values, shape)?.dequantize()?;
        for value in dequantized {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()?;
    println!(
        "wrote {} matrices ({} bytes of F32) to {reference_path}",
        selected.len(),
        offset
    );
    Ok(())
}

/// The CLI's Transformer Engine selection with default name exclusions.
struct Policy(Nvfp4Policy);

impl Policy {
    fn new() -> Self {
        Self(Nvfp4Policy::transformer_engine())
    }

    fn selects(&self, name: &str, dtype: &str, shape: &[usize]) -> bool {
        self.0
            .decide(&Nvfp4Candidate {
                name: name.to_owned(),
                is_floating: matches!(dtype, "F32" | "F16" | "BF16"),
                shape: shape.to_vec(),
            })
            .is_quantized()
    }
}
