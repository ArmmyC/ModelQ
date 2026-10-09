//! ModelQ-native group-wise low-bit SafeTensors files (ADR 0030).
//!
//! Every quantized source tensor becomes two output tensors: `<name>.qdata`, a
//! U8 bitstream of `bits`-bit codes, and `<name>.scale`, one F32 scale per
//! group of consecutive values. Preserved tensors are copied unchanged. The
//! `modelq.manifest` metadata entry maps each source tensor to its outputs, so
//! a decoder needs nothing about the model it came from.
//!
//! The files are ModelQ-native: no inference runtime reads them, and none is
//! claimed. Encoding streams one source tensor at a time, so memory is bounded
//! by one group of values, one tensor's group scales, and one output chunk.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use modelq_core::tensor::TensorView;
use modelq_quant::{
    lowbit::{self, BitWriter, LowBitConfig, LowBitTensor, Scheme},
    policy::{PolicyAction, TensorDecision},
};
use serde_json::{Map, Value, json};

use crate::{
    layout::{OutputLayoutPlan, OutputTensorRole, QuantizedEncoding, plan_output_layout_for},
    safetensors::{MappedSafetensors, TensorSource, TensorSummary},
    writer::{
        TemporaryOutput, WriterError, create_temporary_file, io_error, paths_refer_to_same_file,
    },
};

/// Value of `modelq.format_version` for group-wise low-bit files.
pub const FORMAT_VERSION: &str = "2";
/// Value of `schema` inside the manifest.
pub const MANIFEST_SCHEMA: &str = "modelq.lowbit.manifest.v1";

/// The layout of one group-wise low-bit encoding, as the planner needs it.
pub fn encoding_for(config: &LowBitConfig) -> QuantizedEncoding {
    QuantizedEncoding::GroupWise {
        bits: config.bits,
        group_size: config.group_size,
        scheme: config.scheme,
    }
}

/// The name written to `modelq.quantization`, such as `int4`.
pub fn quantization_name(config: &LowBitConfig) -> String {
    format!("int{}", config.bits)
}

fn scheme_name(scheme: Scheme) -> &'static str {
    match scheme {
        Scheme::Symmetric => "symmetric-group-wise",
        Scheme::Sign => "sign-group-wise",
    }
}

fn algorithm_name(scheme: Scheme) -> &'static str {
    match scheme {
        Scheme::Symmetric => "max-abs-group-scale-v1",
        Scheme::Sign => "mean-abs-group-scale-v1",
    }
}

/// Writes a group-wise low-bit file for a planned layout.
///
/// The plan must equal the one [`plan_output_layout_for`] derives for `config`
/// from the source's tensors and the decisions. The destination must not exist,
/// and the file is committed by renaming a temporary file, as the INT8 writer
/// does.
pub fn write_lowbit_safetensors(
    source: &impl TensorSource,
    plan: &OutputLayoutPlan,
    decisions: &[TensorDecision],
    config: LowBitConfig,
    destination: impl AsRef<Path>,
) -> Result<(), WriterError> {
    config.validate().map_err(|error| WriterError::Encoding {
        name: String::new(),
        detail: error.to_string(),
    })?;
    let destination = destination.as_ref().to_owned();
    if destination.file_name().is_none() {
        return Err(WriterError::InvalidDestination { path: destination });
    }
    if let Some(conflict) = source
        .source_paths()
        .into_iter()
        .find(|path| paths_refer_to_same_file(path, &destination))
    {
        return Err(WriterError::SourceDestinationConflict {
            source: conflict,
            destination,
        });
    }
    if destination.exists() {
        return Err(WriterError::DestinationExists { path: destination });
    }

    let summaries = source.tensor_summaries();
    let expected = plan_output_layout_for(&summaries, decisions, encoding_for(&config))
        .map_err(|source| WriterError::Layout { source })?;
    if expected != *plan {
        return Err(WriterError::PlanMismatch);
    }

    let manifest = build_manifest(&summaries, decisions, plan, &config)?;
    let header = build_header(plan, &config, &manifest)?;

    let (temporary_path, mut file) = create_temporary_file(&destination)?;
    let mut temporary = TemporaryOutput::new(temporary_path.clone());
    let write_result = (|| {
        file.write_all(&header)
            .map_err(|error| io_error(&destination, error))?;
        write_data(&mut file, &destination, source, plan, &config)?;
        file.sync_all()
            .map_err(|error| io_error(&destination, error))
    })();
    drop(file);
    write_result?;

    if destination.exists() {
        return Err(WriterError::DestinationExists { path: destination });
    }
    fs::rename(&temporary_path, &destination).map_err(|error| io_error(&destination, error))?;
    temporary.committed = true;
    Ok(())
}

fn build_manifest(
    summaries: &[TensorSummary],
    decisions: &[TensorDecision],
    plan: &OutputLayoutPlan,
    config: &LowBitConfig,
) -> Result<String, WriterError> {
    let decision_by_name: HashMap<&str, &TensorDecision> = decisions
        .iter()
        .map(|decision| (decision.name.as_str(), decision))
        .collect();
    let mut tensors = Map::new();
    for summary in summaries {
        let decision = decision_by_name
            .get(summary.name.as_str())
            .ok_or(WriterError::PlanMismatch)?;
        let entry = match decision.action {
            PolicyAction::Preserve => {
                let output = planned(plan, &summary.name, OutputTensorRole::Preserved)?;
                json!({
                    "action": "preserved",
                    "original_dtype": summary.dtype,
                    "original_shape": summary.shape,
                    "tensor_name": output.name,
                })
            }
            PolicyAction::Quantize => {
                let payload = planned(plan, &summary.name, OutputTensorRole::QuantizedData)?;
                let scale = planned(plan, &summary.name, OutputTensorRole::QuantizationScale)?;
                json!({
                    "action": "quantized",
                    "original_dtype": summary.dtype,
                    "original_shape": summary.shape,
                    "qdata_name": payload.name,
                    "qdata_dtype": payload.dtype,
                    "qdata_shape": payload.shape,
                    "scale_name": scale.name,
                    "scale_dtype": scale.dtype,
                    "scale_shape": scale.shape,
                    "elements": decision.element_count,
                    "bits": config.bits,
                    "group_size": config.group_size,
                })
            }
        };
        tensors.insert(summary.name.clone(), entry);
    }
    let manifest = json!({ "schema": MANIFEST_SCHEMA, "tensors": tensors });
    serde_json::to_string(&manifest).map_err(|source| WriterError::Serialization { source })
}

fn planned<'a>(
    plan: &'a OutputLayoutPlan,
    source_name: &str,
    role: OutputTensorRole,
) -> Result<&'a crate::layout::OutputTensorPlan, WriterError> {
    plan.tensors
        .iter()
        .find(|tensor| tensor.source_name == source_name && tensor.role == role)
        .ok_or(WriterError::PlanMismatch)
}

fn metadata_entries(config: &LowBitConfig, manifest: &str) -> BTreeMap<String, String> {
    let mut entries = BTreeMap::new();
    entries.insert("modelq.format".to_owned(), "modelq-native".to_owned());
    entries.insert(
        "modelq.format_version".to_owned(),
        FORMAT_VERSION.to_owned(),
    );
    entries.insert("modelq.quantization".to_owned(), quantization_name(config));
    entries.insert(
        "modelq.scheme".to_owned(),
        scheme_name(config.scheme).to_owned(),
    );
    entries.insert(
        "modelq.algorithm".to_owned(),
        algorithm_name(config.scheme).to_owned(),
    );
    entries.insert(
        "modelq.packing".to_owned(),
        "lsb-first-bitstream".to_owned(),
    );
    entries.insert(
        "modelq.rounding".to_owned(),
        "ties-away-from-zero".to_owned(),
    );
    entries.insert("modelq.bits".to_owned(), config.bits.to_string());
    entries.insert(
        "modelq.group_size".to_owned(),
        config.group_size.to_string(),
    );
    match config.scheme {
        Scheme::Symmetric => {
            let qmax = config.qmax();
            entries.insert("modelq.qmax".to_owned(), qmax.to_string());
            entries.insert("modelq.qmin".to_owned(), (-qmax).to_string());
        }
        Scheme::Sign => {
            entries.insert("modelq.code_map".to_owned(), "1=+scale,0=-scale".to_owned());
        }
    }
    entries.insert("modelq.manifest".to_owned(), manifest.to_owned());
    entries
}

fn build_header(
    plan: &OutputLayoutPlan,
    config: &LowBitConfig,
    manifest: &str,
) -> Result<Vec<u8>, WriterError> {
    let mut metadata = Map::new();
    for (key, value) in metadata_entries(config, manifest) {
        metadata.insert(key, Value::String(value));
    }
    let mut header = Map::new();
    header.insert("__metadata__".to_owned(), Value::Object(metadata));
    for tensor in &plan.tensors {
        header.insert(
            tensor.name.clone(),
            json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [tensor.data_offsets.start, tensor.data_offsets.end],
            }),
        );
    }
    let mut bytes = serde_json::to_vec(&Value::Object(header))
        .map_err(|source| WriterError::Serialization { source })?;
    while bytes.len() % 8 != 0 {
        bytes.push(b' ');
    }
    let mut output = (bytes.len() as u64).to_le_bytes().to_vec();
    output.extend_from_slice(&bytes);
    Ok(output)
}

fn write_data(
    file: &mut File,
    destination: &Path,
    source: &impl TensorSource,
    plan: &OutputLayoutPlan,
    config: &LowBitConfig,
) -> Result<(), WriterError> {
    let mut pending_scales: Option<Vec<u8>> = None;
    for tensor in &plan.tensors {
        match tensor.role {
            OutputTensorRole::Preserved => {
                let written = source
                    .with_tensor_bytes(&tensor.source_name, |bytes| {
                        if bytes.len() as u64 != tensor.byte_len {
                            return Err(WriterError::PlanMismatch);
                        }
                        file.write_all(bytes)
                            .map_err(|error| io_error(destination, error))
                    })
                    .map_err(|error| WriterError::Source { source: error })?;
                written?;
            }
            OutputTensorRole::QuantizedData => {
                let encoded = source
                    .with_tensor(&tensor.source_name, |view| {
                        encode_tensor(view, config, &tensor.source_name, file, destination)
                    })
                    .map_err(|error| WriterError::Source { source: error })??;
                if encoded.payload_bytes != tensor.byte_len {
                    return Err(WriterError::PlanMismatch);
                }
                pending_scales = Some(
                    encoded
                        .scales
                        .iter()
                        .flat_map(|scale| scale.to_le_bytes())
                        .collect(),
                );
            }
            OutputTensorRole::QuantizationScale => {
                let bytes = pending_scales.take().ok_or(WriterError::PlanMismatch)?;
                if bytes.len() as u64 != tensor.byte_len {
                    return Err(WriterError::PlanMismatch);
                }
                file.write_all(&bytes)
                    .map_err(|error| io_error(destination, error))?;
            }
        }
    }
    Ok(())
}

struct EncodedTensor {
    scales: Vec<f32>,
    payload_bytes: u64,
}

/// Encodes one tensor's values, writing the packed codes to `output` as they
/// are produced and returning the group scales.
fn encode_tensor(
    view: TensorView<'_>,
    config: &LowBitConfig,
    name: &str,
    output: &mut File,
    destination: &Path,
) -> Result<EncodedTensor, WriterError> {
    let encoding_error = |error: lowbit::LowBitError| WriterError::Encoding {
        name: name.to_owned(),
        detail: error.to_string(),
    };
    let mut writer = BitWriter::new(config.bits);
    let mut scales = Vec::new();
    let mut payload_bytes = 0_u64;
    let mut group: Vec<f32> = Vec::with_capacity(config.group_size);
    let mut group_start = 0_usize;
    for (position, value) in view.values().enumerate() {
        group.push(value);
        if group.len() == config.group_size {
            encode_group(&group, group_start, config, &mut writer, &mut scales)
                .map_err(encoding_error)?;
            group.clear();
            group_start = position + 1;
            payload_bytes += drain(&mut writer, output, destination)?;
        }
    }
    if !group.is_empty() {
        encode_group(&group, group_start, config, &mut writer, &mut scales)
            .map_err(encoding_error)?;
    }
    let tail = writer.finish();
    output
        .write_all(&tail)
        .map_err(|error| io_error(destination, error))?;
    payload_bytes += tail.len() as u64;
    Ok(EncodedTensor {
        scales,
        payload_bytes,
    })
}

fn encode_group(
    group: &[f32],
    start: usize,
    config: &LowBitConfig,
    writer: &mut BitWriter,
    scales: &mut Vec<f32>,
) -> Result<(), lowbit::LowBitError> {
    let scale = lowbit::group_scale(group, start, config)?;
    for (offset, &value) in group.iter().enumerate() {
        writer.push(lowbit::code_for(value, scale, start + offset, config)?);
    }
    scales.push(scale);
    Ok(())
}

fn drain(
    writer: &mut BitWriter,
    output: &mut File,
    destination: &Path,
) -> Result<u64, WriterError> {
    let bytes = writer.drain();
    output
        .write_all(&bytes)
        .map_err(|error| io_error(destination, error))?;
    Ok(bytes.len() as u64)
}

/// One source tensor's entry in a low-bit manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestEntry {
    /// A quantized tensor stored as a packed payload and group scales.
    Quantized {
        qdata_name: String,
        scale_name: String,
        shape: Vec<usize>,
        elements: usize,
    },
    /// A tensor copied unchanged under `tensor_name`.
    Preserved { tensor_name: String },
}

/// A parsed low-bit manifest: the codec configuration and every source tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct LowBitManifest {
    /// The codec configuration the file was written with.
    pub config: LowBitConfig,
    /// Entries keyed by source tensor name.
    pub tensors: BTreeMap<String, ManifestEntry>,
}

/// Reads the manifest from a file's `__metadata__`.
///
/// Returns `Ok(None)` when the file is not group-wise low-bit output (for
/// example an INT8 file), and an error when it claims to be but is malformed.
pub fn read_manifest(
    metadata: &BTreeMap<String, String>,
) -> Result<Option<LowBitManifest>, String> {
    if metadata.get("modelq.format").map(String::as_str) != Some("modelq-native") {
        return Ok(None);
    }
    let Some(quantization) = metadata.get("modelq.quantization") else {
        return Ok(None);
    };
    if !["int1", "int2", "int3", "int4"].contains(&quantization.as_str()) {
        return Ok(None);
    }
    let version = metadata
        .get("modelq.format_version")
        .map(String::as_str)
        .unwrap_or_default();
    if version != FORMAT_VERSION {
        return Err(format!("unsupported low-bit format version {version:?}"));
    }
    let scheme = match metadata.get("modelq.scheme").map(String::as_str) {
        Some("symmetric-group-wise") => Scheme::Symmetric,
        Some("sign-group-wise") => Scheme::Sign,
        other => return Err(format!("unsupported low-bit scheme {other:?}")),
    };
    let bits = metadata_number(metadata, "modelq.bits")?;
    let group_size = metadata_number(metadata, "modelq.group_size")?;
    let config = LowBitConfig {
        bits: u8::try_from(bits).map_err(|_| format!("invalid bit width {bits}"))?,
        group_size,
        scheme,
    };
    config.validate().map_err(|error| error.to_string())?;

    let manifest_text = metadata
        .get("modelq.manifest")
        .ok_or_else(|| "missing modelq.manifest".to_owned())?;
    let manifest: Value = serde_json::from_str(manifest_text)
        .map_err(|error| format!("modelq.manifest is not JSON: {error}"))?;
    if manifest.get("schema").and_then(Value::as_str) != Some(MANIFEST_SCHEMA) {
        return Err("unknown manifest schema".to_owned());
    }
    let tensors_value = manifest
        .get("tensors")
        .and_then(Value::as_object)
        .ok_or_else(|| "manifest has no tensors object".to_owned())?;
    let mut tensors = BTreeMap::new();
    for (name, entry) in tensors_value {
        tensors.insert(name.clone(), parse_entry(name, entry)?);
    }
    Ok(Some(LowBitManifest { config, tensors }))
}

fn metadata_number(metadata: &BTreeMap<String, String>, key: &str) -> Result<usize, String> {
    metadata
        .get(key)
        .ok_or_else(|| format!("missing {key}"))?
        .parse()
        .map_err(|_| format!("{key} is not a number"))
}

fn parse_entry(name: &str, entry: &Value) -> Result<ManifestEntry, String> {
    let string = |field: &str| -> Result<String, String> {
        entry
            .get(field)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| format!("{name:?} is missing {field}"))
    };
    match entry.get("action").and_then(Value::as_str) {
        Some("preserved") => Ok(ManifestEntry::Preserved {
            tensor_name: string("tensor_name")?,
        }),
        Some("quantized") => {
            let shape = entry
                .get("original_shape")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{name:?} is missing original_shape"))?
                .iter()
                .map(|dimension| {
                    dimension
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| format!("{name:?} has an invalid dimension"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elements = entry
                .get("elements")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("{name:?} is missing elements"))?;
            let product = shape
                .iter()
                .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
                .ok_or_else(|| format!("{name:?} has an overflowing shape"))?;
            if product != elements {
                return Err(format!(
                    "{name:?} declares {elements} elements for shape {shape:?}"
                ));
            }
            Ok(ManifestEntry::Quantized {
                qdata_name: string("qdata_name")?,
                scale_name: string("scale_name")?,
                shape,
                elements,
            })
        }
        _ => Err(format!("{name:?} has an unknown action")),
    }
}

/// Metrics from checking low-bit outputs against their source tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct LowBitValidation {
    /// Quantized tensors decoded and compared.
    pub quantized_tensors: usize,
    /// Preserved tensors compared byte for byte.
    pub preserved_tensors: usize,
    /// Largest per-tensor mean squared error.
    pub max_mse: f64,
    /// Largest per-tensor mean absolute error.
    pub max_mae: f64,
    /// Largest absolute error of any value.
    pub max_abs_error: f64,
    /// Lowest per-tensor signal-to-quantization-noise ratio in dB, if defined.
    pub lowest_sqnr_db: Option<f64>,
    /// Values whose error exceeds half their group's scale. Symmetric
    /// rounding guarantees none; the sign scheme has no such bound.
    pub scale_bound_violations: u64,
}

/// Reopens the output files, decodes every quantized tensor from its manifest,
/// and compares it with the source.
///
/// Every source tensor must appear exactly once across the files, with the
/// shape it had in the source.
pub fn validate_outputs(
    source: &impl TensorSource,
    outputs: &[PathBuf],
) -> Result<LowBitValidation, String> {
    let summaries: HashMap<String, TensorSummary> = source
        .tensor_summaries()
        .into_iter()
        .map(|summary| (summary.name.clone(), summary))
        .collect();
    let mut expected: BTreeSet<&str> = summaries.keys().map(String::as_str).collect();
    let mut report = LowBitValidation {
        quantized_tensors: 0,
        preserved_tensors: 0,
        max_mse: 0.0,
        max_mae: 0.0,
        max_abs_error: 0.0,
        lowest_sqnr_db: None,
        scale_bound_violations: 0,
    };

    for path in outputs {
        let reader = MappedSafetensors::open(path)
            .map_err(|error| format!("could not reopen {}: {error}", path.display()))?;
        let manifest = read_manifest(reader.metadata())?
            .ok_or_else(|| format!("{} is not a ModelQ low-bit file", path.display()))?;
        for (name, entry) in &manifest.tensors {
            let summary = summaries
                .get(name)
                .ok_or_else(|| format!("output names {name:?}, which the source does not have"))?;
            if !expected.remove(name.as_str()) {
                return Err(format!("source tensor {name:?} appears more than once"));
            }
            match entry {
                ManifestEntry::Preserved { tensor_name } => {
                    let output_bytes = reader
                        .tensor_bytes(tensor_name)
                        .map_err(|error| error.to_string())?;
                    let unchanged = source
                        .with_tensor_bytes(name, |source_bytes| source_bytes == output_bytes)
                        .map_err(|error| error.to_string())?;
                    if !unchanged {
                        return Err(format!("preserved tensor {name:?} changed during writing"));
                    }
                    report.preserved_tensors += 1;
                }
                ManifestEntry::Quantized {
                    qdata_name,
                    scale_name,
                    shape,
                    elements,
                } => {
                    if *shape != summary.shape {
                        return Err(format!("{name:?} changed shape during writing"));
                    }
                    let packed = reader
                        .tensor_bytes(qdata_name)
                        .map_err(|error| error.to_string())?;
                    let scale_bytes = reader
                        .tensor_bytes(scale_name)
                        .map_err(|error| error.to_string())?;
                    if scale_bytes.len() % 4 != 0 {
                        return Err(format!(
                            "{scale_name:?} is not a whole number of F32 values"
                        ));
                    }
                    let scales: Vec<f32> = scale_bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    let tensor = LowBitTensor::from_parts(
                        manifest.config,
                        *elements,
                        packed.to_vec(),
                        scales,
                    )
                    .map_err(|error| format!("{name:?}: {error}"))?;
                    let decoded = tensor
                        .dequantize()
                        .map_err(|error| format!("{name:?}: {error}"))?;
                    let original: Vec<f32> = source
                        .with_tensor(name, |view| view.values().collect())
                        .map_err(|error| error.to_string())?;
                    compare(&original, &decoded, &tensor, &manifest.config, &mut report);
                    report.quantized_tensors += 1;
                }
            }
        }
    }

    if !expected.is_empty() {
        let missing: Vec<&str> = expected.into_iter().collect();
        return Err(format!(
            "source tensors missing from the output: {}",
            missing.join(", ")
        ));
    }
    Ok(report)
}

fn compare(
    original: &[f32],
    decoded: &[f32],
    tensor: &LowBitTensor,
    config: &LowBitConfig,
    report: &mut LowBitValidation,
) {
    let mut squared = 0.0_f64;
    let mut absolute = 0.0_f64;
    let mut largest = 0.0_f64;
    let mut signal = 0.0_f64;
    for (index, (&reference, &reconstructed)) in original.iter().zip(decoded).enumerate() {
        let error = (f64::from(reconstructed) - f64::from(reference)).abs();
        squared += error * error;
        absolute += error;
        largest = largest.max(error);
        signal += f64::from(reference) * f64::from(reference);
        if config.scheme == Scheme::Symmetric {
            let scale = f64::from(tensor.scales()[index / config.group_size]);
            if error > scale / 2.0 * (1.0 + 1e-6) {
                report.scale_bound_violations += 1;
            }
        }
    }
    let count = original.len() as f64;
    if count > 0.0 {
        report.max_mse = report.max_mse.max(squared / count);
        report.max_mae = report.max_mae.max(absolute / count);
    }
    report.max_abs_error = report.max_abs_error.max(largest);
    if squared > 0.0 && signal > 0.0 {
        let sqnr = 10.0 * (signal / squared).log10();
        report.lowest_sqnr_db = Some(
            report
                .lowest_sqnr_db
                .map_or(sqnr, |lowest| lowest.min(sqnr)),
        );
    }
}
