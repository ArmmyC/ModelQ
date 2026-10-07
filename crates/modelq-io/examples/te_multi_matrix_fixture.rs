//! Generate the synthetic multi-matrix checkpoint and F32 reference used to
//! validate the Transformer Engine container (schema v2).
//!
//! Usage: `te_multi_matrix_fixture <output-directory> [--sweep]`
//!
//! With `--sweep` the checkpoint instead holds matrices that vary the final
//! and the leading dimension independently (`K` from 16 to 192 with `M = 64`,
//! and `M` from 16 to 208 with `K = 64`), used to find which shapes the
//! runtime's GEMM accepts.
//!
//! Writes two files into the (new or empty) directory:
//!
//! - `source.safetensors`: F32 matrices that exercise scale padding and the
//!   Transformer Engine eligibility rules (rank two, leading dimension a
//!   multiple of 16, final dimension a multiple of 32), plus tensors that must be
//!   preserved. Convert it with
//!   `modelq quantize source.safetensors --format nvfp4-te --output <path>`
//!   (add `--max-shard-size` for a sharded container).
//! - `reference.safetensors`: the CPU dequantization of every matrix the
//!   Transformer Engine policy exports, as `<name>.dequantized_reference`
//!   F32 `[M, K]` tensors with `modelq.reference_schema` metadata. It is a
//!   test-only oracle and is never part of a runtime container.

use std::{error::Error, fs, path::Path};

use modelq_quant::{
    nvfp4::quantize_shaped,
    nvfp4_policy::{Nvfp4Candidate, Nvfp4Policy},
};
use serde_json::{Map, Value, json};

const REFERENCE_SCHEMA: &str = "transformer-engine-nvfp4-reference-v2";

struct Tensor {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
    values: Option<Vec<f32>>,
}

fn matrix(name: &'static str, rows: usize, columns: usize, seed: f32) -> Tensor {
    let values: Vec<f32> = (0..rows * columns)
        .map(|index| {
            let wave = ((index as f32) * 0.37 + seed).sin();
            wave * (1.0 + (index % 23) as f32) + (index % 5) as f32 * 1e-3
        })
        .collect();
    Tensor {
        name,
        dtype: "F32",
        shape: vec![rows, columns],
        bytes: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        values: Some(values),
    }
}

fn leak(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

fn sweep_tensors() -> Vec<Tensor> {
    let mut tensors = Vec::new();
    for (index, columns) in (16..=192).step_by(16).enumerate() {
        tensors.push(matrix(
            leak(format!("k_sweep.M64.K{columns:03}")),
            64,
            columns,
            index as f32,
        ));
    }
    for (index, rows) in (16..=208).step_by(16).enumerate() {
        tensors.push(matrix(
            leak(format!("m_sweep.M{rows:03}.K64")),
            rows,
            64,
            50.0 + index as f32,
        ));
    }
    tensors
}

fn standard_tensors() -> Vec<Tensor> {
    let mut norm = matrix("norm.weight", 1, 4096, 6.0);
    norm.shape = vec![4096];
    vec![
        // Exported: padding cases and a multi-chunk matrix.
        matrix("layers.0.attn.weight", 32, 32, 0.0),
        matrix("layers.0.mlp.up.weight", 64, 64, 1.0),
        matrix("layers.1.attn.weight", 48, 96, 2.0),
        matrix("layers.1.mlp.up.weight", 144, 96, 3.0),
        matrix("layers.2.mlp.up.weight", 256, 512, 4.0),
        matrix("layers.2.mlp.down.weight", 1024, 4096, 5.0),
        // Preserved: excluded by name, rank 3, M not divisible by 16, K not
        // divisible by 32 (the runtime rejects it), below the size minimum, a
        // vector, and an integer tensor.
        matrix("lm_head.weight", 64, 64, 7.0),
        {
            let mut stack = matrix("experts.weight", 4, 64 * 64, 8.0);
            stack.shape = vec![4, 64, 64];
            stack
        },
        matrix("odd_rows.weight", 70, 64, 9.0),
        matrix("odd_columns.weight", 144, 80, 3.5),
        matrix("tiny.weight", 16, 16, 10.0),
        norm,
        Tensor {
            name: "token_ids",
            dtype: "U8",
            shape: vec![3],
            bytes: vec![1, 2, 3],
            values: None,
        },
    ]
}

fn write_safetensors(
    path: &Path,
    metadata: Option<Map<String, Value>>,
    tensors: &[(String, &str, Vec<usize>, Vec<u8>)],
) -> Result<(), Box<dyn Error>> {
    let mut header = Map::new();
    if let Some(metadata) = metadata {
        header.insert("__metadata__".to_owned(), Value::Object(metadata));
    }
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.clone(),
            json!({ "dtype": dtype, "shape": shape, "data_offsets": [start, data.len()] }),
        );
    }
    let mut header = serde_json::to_vec(&Value::Object(header))?;
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let usage = "usage: te_multi_matrix_fixture <output-directory> [--sweep]";
    let directory = arguments.next().ok_or(usage)?;
    let sweep = match arguments.next().as_deref() {
        None => false,
        Some("--sweep") => true,
        Some(_) => return Err(usage.into()),
    };
    if arguments.next().is_some() {
        return Err(usage.into());
    }
    let directory = Path::new(&directory);
    if directory.exists() && fs::read_dir(directory)?.next().is_some() {
        return Err(format!("{} is not empty", directory.display()).into());
    }
    fs::create_dir_all(directory)?;

    let tensors = if sweep {
        sweep_tensors()
    } else {
        standard_tensors()
    };

    let source: Vec<_> = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name.to_owned(),
                tensor.dtype,
                tensor.shape.clone(),
                tensor.bytes.clone(),
            )
        })
        .collect();
    write_safetensors(&directory.join("source.safetensors"), None, &source)?;

    // The reference covers exactly what the Transformer Engine policy exports.
    let policy = Nvfp4Policy::transformer_engine();
    let mut reference = Vec::new();
    for tensor in &tensors {
        let candidate = Nvfp4Candidate {
            name: tensor.name.to_owned(),
            is_floating: tensor.dtype == "F32",
            shape: tensor.shape.clone(),
        };
        let decision = policy.decide(&candidate);
        println!("{:<28} {}", tensor.name, decision.reason);
        if !decision.is_quantized() {
            continue;
        }
        let values = tensor
            .values
            .as_ref()
            .expect("quantized tensors are floating");
        let quantized = quantize_shaped(values, &tensor.shape)?;
        let dequantized = quantized.dequantize()?;
        reference.push((
            format!("{}.dequantized_reference", tensor.name),
            "F32",
            tensor.shape.clone(),
            dequantized
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
        ));
    }
    let mut metadata = Map::new();
    metadata.insert(
        "modelq.reference_schema".to_owned(),
        Value::String(REFERENCE_SCHEMA.to_owned()),
    );
    write_safetensors(
        &directory.join("reference.safetensors"),
        Some(metadata),
        &reference,
    )?;

    println!(
        "wrote {} and {}",
        directory.join("source.safetensors").display(),
        directory.join("reference.safetensors").display()
    );
    Ok(())
}
