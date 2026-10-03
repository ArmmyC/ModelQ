//! Generate deterministic Transformer Engine NVFP4 runtime and F32 reference fixtures.

use std::{
    env,
    error::Error,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use modelq_io::{
    safetensors::MappedSafetensors, transformer_engine::export_transformer_engine_nvfp4,
    writer::write_transformer_engine_nvfp4_safetensors,
};
use modelq_quant::nvfp4::quantize_shaped;
use serde_json::json;

static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

struct TemporaryOutput(PathBuf);

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn create_temporary_file(destination: &Path) -> io::Result<(TemporaryOutput, File)> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let name = destination
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "destination needs a filename"))?
        .to_string_lossy();
    loop {
        let sequence = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(".{name}.{}.{}.tmp", process::id(), sequence));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((TemporaryOutput(path), file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}

fn output_identity(path: &Path) -> io::Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let filename = path.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "output path needs a filename")
    })?;
    Ok(fs::canonicalize(parent)?.join(filename))
}

fn write_reference_safetensors(values: &[f32], destination: &Path) -> Result<(), Box<dyn Error>> {
    if values.len() != 4096 {
        return Err(format!("{}: expected 4096 F32 values", destination.display()).into());
    }
    if destination.exists() {
        return Err(format!("{}: destination already exists", destination.display()).into());
    }

    let payload = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let header = json!({
        "__metadata__": {
            "modelq.reference_schema": "transformer-engine-nvfp4-reference-v1"
        },
        "weight.dequantized_reference": {
            "dtype": "F32",
            "shape": [64, 64],
            "data_offsets": [0, 16384]
        }
    });
    let raw_header = serde_json::to_vec(&header)?;
    let padded_len = raw_header
        .len()
        .checked_add(7)
        .ok_or("header length overflow")?
        / 8
        * 8;
    let header_len = u64::try_from(padded_len)?;

    let (temporary, mut file) = create_temporary_file(destination)
        .map_err(|error| format!("{}: {error}", destination.display()))?;
    let write_result = (|| -> io::Result<()> {
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(&raw_header)?;
        file.write_all(&vec![b' '; padded_len - raw_header.len()])?;
        file.write_all(&payload)?;
        file.sync_all()
    })();
    drop(file);
    write_result.map_err(|error| format!("{}: {error}", destination.display()))?;
    if destination.exists() {
        return Err(format!("{}: destination already exists", destination.display()).into());
    }
    fs::rename(&temporary.0, destination)
        .map_err(|error| format!("{}: {error}", destination.display()))?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let (Some(runtime_path), Some(reference_path), None) = (args.next(), args.next(), args.next())
    else {
        return Err("usage: transformer_engine_nvfp4_fixture <runtime-artifact.safetensors> <reference.safetensors>".into());
    };
    let runtime_path = PathBuf::from(runtime_path);
    let reference_path = PathBuf::from(reference_path);
    if runtime_path == reference_path
        || matches!(
            (output_identity(&runtime_path), output_identity(&reference_path)),
            (Ok(runtime), Ok(reference)) if runtime == reference
        )
    {
        return Err("usage: transformer_engine_nvfp4_fixture <runtime-artifact.safetensors> <reference.safetensors>".into());
    }

    let values = (0_i32..4096)
        .map(|index| (((index * 37) % 251 - 125) as f32) / 32.0)
        .collect::<Vec<_>>();
    let native = quantize_shaped(&values, &[64, 64])?;
    let profile = export_transformer_engine_nvfp4("weight", &[64, 64], &native)?;
    write_transformer_engine_nvfp4_safetensors(&profile, &runtime_path)
        .map_err(|error| format!("{}: {error}", runtime_path.display()))?;

    let reference_values = native.dequantize()?;
    write_reference_safetensors(&reference_values, &reference_path)?;

    let runtime = MappedSafetensors::open(&runtime_path)
        .map_err(|error| format!("{}: {error}", runtime_path.display()))?;
    let runtime_tensors = runtime.tensors().collect::<Vec<_>>();
    if runtime_tensors.len() != 3 {
        return Err(format!("{}: expected three runtime tensors", runtime_path.display()).into());
    }
    let reference = MappedSafetensors::open(&reference_path)
        .map_err(|error| format!("{}: {error}", reference_path.display()))?;
    let tensors = reference.tensors().collect::<Vec<_>>();
    if tensors.len() != 1 || tensors[0].dtype != "F32" || tensors[0].shape != [64, 64] {
        return Err(format!(
            "{}: expected one F32 [64, 64] tensor",
            reference_path.display()
        )
        .into());
    }

    for tensor in runtime_tensors {
        println!(
            "{}: {} {:?}",
            runtime_path.display(),
            tensor.name,
            tensor.shape
        );
    }
    println!(
        "{}: reference tensor F32 [64, 64]",
        reference_path.display()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
    };

    use modelq_io::safetensors::MappedSafetensors;

    use super::write_reference_safetensors;

    static NEXT_ARTIFACT: AtomicU64 = AtomicU64::new(0);

    struct TempArtifact(PathBuf);

    impl TempArtifact {
        fn new() -> Self {
            let sequence = NEXT_ARTIFACT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "modelq-te-nvfp4-reference-{}-{sequence}.safetensors",
                std::process::id()
            ));
            Self(path)
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

    #[test]
    fn reference_file_contains_one_f32_tensor_with_exact_payload_and_metadata() {
        let output = TempArtifact::new();
        let mut values = vec![0.0_f32; 4096];
        values[0] = 1.0;
        values[1] = -2.5;

        write_reference_safetensors(&values, output.path()).expect("reference file is written");

        let file =
            MappedSafetensors::open(output.path()).expect("reference file is valid SafeTensors");
        let tensors = file.tensors().collect::<Vec<_>>();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "weight.dequantized_reference");
        assert_eq!(tensors[0].dtype, "F32");
        assert_eq!(tensors[0].shape, [64, 64]);
        assert_eq!(
            file.metadata()
                .get("modelq.reference_schema")
                .map(String::as_str),
            Some("transformer-engine-nvfp4-reference-v1")
        );
        let expected = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            file.tensor_bytes("weight.dequantized_reference").unwrap(),
            expected
        );
    }

    #[test]
    fn reference_writer_preserves_an_existing_destination() {
        let output = TempArtifact::new();
        let sentinel = b"existing data";
        fs::write(output.path(), sentinel).expect("sentinel file is written");

        assert!(write_reference_safetensors(&vec![0.0; 4096], output.path()).is_err());
        assert_eq!(fs::read(output.path()).unwrap(), sentinel);
    }
}
