//! SafeTensors metadata inspection and read-only memory-mapped access.

use std::{
    fmt,
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    str,
};

use memmap2::{Mmap, MmapOptions};
use serde_json::Value;

use crate::tensor::{DType, TensorView};

const HEADER_LENGTH_BYTES: u64 = 8;
const MAX_HEADER_SIZE: u64 = 100_000_000;

/// A file-level summary produced by SafeTensors inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inspection {
    /// Total size of the source file in bytes.
    pub file_size: u64,
    /// Tensor metadata ordered by data offset in the file.
    pub tensors: Vec<TensorSummary>,
}

/// Metadata for one tensor in a SafeTensors file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSummary {
    /// Tensor name from the SafeTensors header.
    pub name: String,
    /// SafeTensors dtype name, such as `F32` or `BF16`.
    pub dtype: String,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Tensor payload size in bytes.
    pub byte_len: u64,
}

/// Errors returned while opening or validating a SafeTensors file.
#[derive(Debug)]
pub enum SafetensorsError {
    /// The file could not be opened, read, or inspected.
    Io { path: PathBuf, source: io::Error },
    /// The file did not contain a complete eight-byte header length.
    HeaderTooSmall { path: PathBuf },
    /// The header exceeded the SafeTensors maximum header size.
    HeaderTooLarge { path: PathBuf, size: u64 },
    /// The header length could not be represented or added safely.
    InvalidHeaderLength { path: PathBuf },
    /// The header was not valid UTF-8.
    InvalidHeaderUtf8 {
        path: PathBuf,
        source: str::Utf8Error,
    },
    /// The header was not valid JSON.
    InvalidHeaderJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    /// The JSON had an invalid SafeTensors metadata shape.
    InvalidMetadata { path: PathBuf, message: String },
    /// Header and data offsets did not cover the file exactly.
    MetadataIncompleteBuffer {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    /// The requested tensor name was not present in the file metadata.
    TensorNotFound { path: PathBuf, name: String },
    /// The tensor is valid SafeTensors metadata but not yet supported by
    /// ModelQ's source tensor view.
    UnsupportedTensorDtype {
        path: PathBuf,
        name: String,
        dtype: String,
    },
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(formatter, "could not read {}: {source}", path.display())
            }
            Self::HeaderTooSmall { path } => write!(
                formatter,
                "SafeTensors file {} has a header smaller than 8 bytes",
                path.display()
            ),
            Self::HeaderTooLarge { path, size } => write!(
                formatter,
                "SafeTensors header in {} is too large ({size} bytes)",
                path.display()
            ),
            Self::InvalidHeaderLength { path } => write!(
                formatter,
                "SafeTensors header length in {} is invalid",
                path.display()
            ),
            Self::InvalidHeaderUtf8 { path, source } => write!(
                formatter,
                "SafeTensors header in {} is not valid UTF-8: {source}",
                path.display()
            ),
            Self::InvalidHeaderJson { path, source } => write!(
                formatter,
                "SafeTensors header in {} is not valid JSON: {source}",
                path.display()
            ),
            Self::InvalidMetadata { path, message } => write!(
                formatter,
                "SafeTensors metadata in {} is invalid: {message}",
                path.display()
            ),
            Self::MetadataIncompleteBuffer {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "SafeTensors file {} has {actual} bytes but metadata requires {expected}",
                path.display()
            ),
            Self::TensorNotFound { path, name } => write!(
                formatter,
                "SafeTensors file {} has no tensor named {name:?}",
                path.display()
            ),
            Self::UnsupportedTensorDtype { path, name, dtype } => write!(
                formatter,
                "tensor {name:?} in {} uses unsupported view dtype {dtype:?}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SafetensorsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::InvalidHeaderUtf8 { source, .. } => Some(source),
            Self::InvalidHeaderJson { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Inspects a SafeTensors file without reading tensor payload bytes.
pub fn inspect_file(path: &Path) -> Result<Inspection, SafetensorsError> {
    let mut file = File::open(path).map_err(|source| SafetensorsError::Io {
        path: path.to_owned(),
        source,
    })?;
    let file_size = file
        .metadata()
        .map_err(|source| SafetensorsError::Io {
            path: path.to_owned(),
            source,
        })?
        .len();

    if file_size < HEADER_LENGTH_BYTES {
        return Err(SafetensorsError::HeaderTooSmall {
            path: path.to_owned(),
        });
    }

    let mut header_length_bytes = [0_u8; HEADER_LENGTH_BYTES as usize];
    file.read_exact(&mut header_length_bytes)
        .map_err(|source| SafetensorsError::Io {
            path: path.to_owned(),
            source,
        })?;
    let header_size = parse_header_size(path, header_length_bytes)?;

    let header_size_usize =
        usize::try_from(header_size).map_err(|_| SafetensorsError::InvalidHeaderLength {
            path: path.to_owned(),
        })?;
    let mut header = vec![0_u8; header_size_usize];
    file.read_exact(&mut header)
        .map_err(|source| SafetensorsError::Io {
            path: path.to_owned(),
            source,
        })?;

    let tensors = parse_tensors(path, file_size, header_size, &header)?;

    Ok(Inspection {
        file_size,
        tensors: tensors.into_iter().map(|tensor| tensor.summary).collect(),
    })
}

fn parse_header_size(
    path: &Path,
    header_length_bytes: [u8; HEADER_LENGTH_BYTES as usize],
) -> Result<u64, SafetensorsError> {
    let header_size = u64::from_le_bytes(header_length_bytes);
    if header_size > MAX_HEADER_SIZE {
        return Err(SafetensorsError::HeaderTooLarge {
            path: path.to_owned(),
            size: header_size,
        });
    }
    Ok(header_size)
}

fn parse_tensors(
    path: &Path,
    file_size: u64,
    header_size: u64,
    header: &[u8],
) -> Result<Vec<RawTensor>, SafetensorsError> {
    let header = str::from_utf8(header).map_err(|source| SafetensorsError::InvalidHeaderUtf8 {
        path: path.to_owned(),
        source,
    })?;
    let document: Value =
        serde_json::from_str(header).map_err(|source| SafetensorsError::InvalidHeaderJson {
            path: path.to_owned(),
            source,
        })?;
    let object = document
        .as_object()
        .ok_or_else(|| invalid_metadata(path, "the header root must be a JSON object"))?;
    let data_start = HEADER_LENGTH_BYTES
        .checked_add(header_size)
        .ok_or_else(|| SafetensorsError::InvalidHeaderLength {
            path: path.to_owned(),
        })?;
    if data_start > file_size {
        return Err(SafetensorsError::MetadataIncompleteBuffer {
            path: path.to_owned(),
            expected: data_start,
            actual: file_size,
        });
    }
    let data_len = file_size - data_start;

    let mut tensors = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            validate_file_metadata(path, value)?;
            continue;
        }

        let tensor = value.as_object().ok_or_else(|| {
            invalid_metadata(path, format!("tensor {name:?} must be a JSON object"))
        })?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_metadata(path, format!("tensor {name:?} has no dtype")))?;
        if !is_known_dtype(dtype) {
            return Err(invalid_metadata(
                path,
                format!("tensor {name:?} has unknown dtype {dtype:?}"),
            ));
        }

        let shape_value = tensor
            .get("shape")
            .ok_or_else(|| invalid_metadata(path, format!("tensor {name:?} has no shape")))?;
        let shape = parse_shape(path, name, shape_value)?;

        let offsets_value = tensor.get("data_offsets").ok_or_else(|| {
            invalid_metadata(path, format!("tensor {name:?} has no data_offsets"))
        })?;
        let (start, end) = parse_offsets(path, name, offsets_value)?;
        if end < start || end > data_len {
            return Err(invalid_metadata(
                path,
                format!(
                    "tensor {name:?} has offsets [{start}, {end}] outside {data_len} data bytes"
                ),
            ));
        }
        let byte_len = end - start;
        let expected_byte_len = expected_byte_len(dtype, &shape).ok_or_else(|| {
            invalid_metadata(
                path,
                format!("tensor {name:?} has an invalid shape or dtype size"),
            )
        })?;
        if byte_len != expected_byte_len {
            return Err(invalid_metadata(
                path,
                format!(
                    "tensor {name:?} requires {expected_byte_len} bytes but offsets cover {byte_len}"
                ),
            ));
        }

        tensors.push(RawTensor {
            summary: TensorSummary {
                name: name.clone(),
                dtype: dtype.to_owned(),
                shape,
                byte_len,
            },
            start,
            end,
        });
    }

    tensors.sort_by_key(|tensor| tensor.start);
    let mut cursor = 0_u64;
    for tensor in &tensors {
        if tensor.start != cursor {
            return Err(invalid_metadata(
                path,
                format!(
                    "tensor {:?} leaves a gap or overlap at byte {cursor}",
                    tensor.summary.name
                ),
            ));
        }
        cursor = tensor.end;
    }
    if cursor != data_len {
        return Err(SafetensorsError::MetadataIncompleteBuffer {
            path: path.to_owned(),
            expected: data_start + cursor,
            actual: file_size,
        });
    }

    Ok(tensors)
}

#[derive(Debug)]
struct RawTensor {
    summary: TensorSummary,
    start: u64,
    end: u64,
}

/// A read-only memory-mapped SafeTensors file.
///
/// The reader owns both the source [`File`] and its mapping. Tensor views
/// borrow this reader, so the mapped bytes cannot outlive the file handle and
/// mapping that back them. Metadata is parsed once during [`Self::open`]; no
/// tensor payload is copied into an owned byte buffer.
pub struct MappedSafetensors {
    path: PathBuf,
    file_size: u64,
    data_start: usize,
    tensors: Vec<RawTensor>,
    /// Keeping the read-only source handle in the owner documents and enforces
    /// the mapping lifetime relationship required by the mmap API.
    _file: File,
    mmap: Mmap,
}

impl MappedSafetensors {
    /// Opens and validates a SafeTensors file while mapping its bytes read-only.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        let path = path.as_ref().to_owned();
        let file = File::open(&path).map_err(|source| SafetensorsError::Io {
            path: path.clone(),
            source,
        })?;
        let file_size = file
            .metadata()
            .map_err(|source| SafetensorsError::Io {
                path: path.clone(),
                source,
            })?
            .len();
        if file_size < HEADER_LENGTH_BYTES {
            return Err(SafetensorsError::HeaderTooSmall { path });
        }

        // SAFETY: the mapping is read-only and the returned `Mmap` is stored
        // alongside the source `File` in this owner. Callers must not mutate
        // or truncate the file while a reader is open, which is the contract
        // required by `memmap2` for a stable mapped range.
        let mmap =
            unsafe { MmapOptions::new().map(&file) }.map_err(|source| SafetensorsError::Io {
                path: path.clone(),
                source,
            })?;
        if mmap.len() < HEADER_LENGTH_BYTES as usize {
            return Err(SafetensorsError::HeaderTooSmall { path });
        }

        let header_length_bytes = mmap[..HEADER_LENGTH_BYTES as usize]
            .try_into()
            .expect("the mmap length was checked above");
        let header_size = parse_header_size(&path, header_length_bytes)?;
        let data_start_u64 = HEADER_LENGTH_BYTES
            .checked_add(header_size)
            .ok_or_else(|| SafetensorsError::InvalidHeaderLength { path: path.clone() })?;
        let mapped_size = u64::try_from(mmap.len()).unwrap_or(u64::MAX);
        if data_start_u64 > file_size || data_start_u64 > mapped_size {
            return Err(SafetensorsError::MetadataIncompleteBuffer {
                path,
                expected: data_start_u64,
                actual: file_size.min(mapped_size),
            });
        }
        let data_start = usize::try_from(data_start_u64)
            .map_err(|_| SafetensorsError::InvalidHeaderLength { path: path.clone() })?;
        let header = &mmap[HEADER_LENGTH_BYTES as usize..data_start];
        let tensors = parse_tensors(&path, file_size, header_size, header)?;

        Ok(Self {
            path,
            file_size,
            data_start,
            tensors,
            _file: file,
            mmap,
        })
    }

    /// Returns the total size of the mapped source file in bytes.
    pub const fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Returns the source path used to open this reader.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the validated tensor metadata in data-offset order.
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &TensorSummary> {
        self.tensors.iter().map(|tensor| &tensor.summary)
    }

    /// Returns an owned inspection summary without reading or copying payloads.
    pub fn inspection(&self) -> Inspection {
        Inspection {
            file_size: self.file_size,
            tensors: self
                .tensors
                .iter()
                .map(|tensor| tensor.summary.clone())
                .collect(),
        }
    }

    /// Exposes one tensor's validated raw payload without decoding it.
    ///
    /// This is useful for writers that must preserve tensors whose dtype is
    /// valid SafeTensors metadata but is not part of ModelQ's floating-point
    /// view API. The returned slice borrows the read-only mapping and does not
    /// allocate or copy the payload.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], SafetensorsError> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.summary.name == name)
            .ok_or_else(|| SafetensorsError::TensorNotFound {
                path: self.path.clone(),
                name: name.to_owned(),
            })?;

        self.tensor_payload(tensor)
    }

    /// Exposes one supported source tensor as a borrowed, mapped view.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, SafetensorsError> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.summary.name == name)
            .ok_or_else(|| SafetensorsError::TensorNotFound {
                path: self.path.clone(),
                name: name.to_owned(),
            })?;
        let dtype = view_dtype(&tensor.summary.dtype).ok_or_else(|| {
            SafetensorsError::UnsupportedTensorDtype {
                path: self.path.clone(),
                name: tensor.summary.name.clone(),
                dtype: tensor.summary.dtype.clone(),
            }
        })?;
        let data = self.tensor_payload(tensor)?;

        TensorView::new(&tensor.summary.name, dtype, &tensor.summary.shape, data)
            .map_err(|error| invalid_metadata(&self.path, error.to_string()))
    }

    fn tensor_payload(&self, tensor: &RawTensor) -> Result<&[u8], SafetensorsError> {
        let start = usize::try_from(tensor.start)
            .ok()
            .and_then(|offset| self.data_start.checked_add(offset))
            .ok_or_else(|| SafetensorsError::InvalidHeaderLength {
                path: self.path.clone(),
            })?;
        let end = usize::try_from(tensor.end)
            .ok()
            .and_then(|offset| self.data_start.checked_add(offset))
            .ok_or_else(|| SafetensorsError::InvalidHeaderLength {
                path: self.path.clone(),
            })?;
        self.mmap
            .get(start..end)
            .ok_or_else(|| SafetensorsError::MetadataIncompleteBuffer {
                path: self.path.clone(),
                expected: u64::try_from(end).unwrap_or(u64::MAX),
                actual: u64::try_from(self.mmap.len()).unwrap_or(u64::MAX),
            })
    }
}

fn view_dtype(dtype: &str) -> Option<DType> {
    match dtype {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::BF16),
        _ => None,
    }
}

fn invalid_metadata(path: &Path, message: impl Into<String>) -> SafetensorsError {
    SafetensorsError::InvalidMetadata {
        path: path.to_owned(),
        message: message.into(),
    }
}

fn validate_file_metadata(path: &Path, value: &Value) -> Result<(), SafetensorsError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid_metadata(path, "__metadata__ must be a JSON object"))?;
    if object.values().any(|value| value.as_str().is_none()) {
        return Err(invalid_metadata(
            path,
            "__metadata__ values must be strings",
        ));
    }
    Ok(())
}

fn parse_shape(path: &Path, name: &str, value: &Value) -> Result<Vec<usize>, SafetensorsError> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid_metadata(path, format!("tensor {name:?} shape must be an array")))?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let dimension = value.as_u64().ok_or_else(|| {
                invalid_metadata(
                    path,
                    format!("tensor {name:?} shape dimension {index} must be an integer"),
                )
            })?;
            usize::try_from(dimension).map_err(|_| {
                invalid_metadata(
                    path,
                    format!("tensor {name:?} shape dimension {index} is too large"),
                )
            })
        })
        .collect()
}

fn parse_offsets(path: &Path, name: &str, value: &Value) -> Result<(u64, u64), SafetensorsError> {
    let values = value.as_array().ok_or_else(|| {
        invalid_metadata(
            path,
            format!("tensor {name:?} data_offsets must be an array"),
        )
    })?;
    if values.len() != 2 {
        return Err(invalid_metadata(
            path,
            format!("tensor {name:?} data_offsets must contain two integers"),
        ));
    }
    let start = values[0].as_u64().ok_or_else(|| {
        invalid_metadata(
            path,
            format!("tensor {name:?} start offset must be an integer"),
        )
    })?;
    let end = values[1].as_u64().ok_or_else(|| {
        invalid_metadata(
            path,
            format!("tensor {name:?} end offset must be an integer"),
        )
    })?;
    Ok((start, end))
}

fn is_known_dtype(dtype: &str) -> bool {
    matches!(
        dtype,
        "BOOL"
            | "F4"
            | "F6_E2M3"
            | "F6_E3M2"
            | "U8"
            | "I8"
            | "F8_E5M2"
            | "F8_E4M3"
            | "F8_E8M0"
            | "F8_E4M3FNUZ"
            | "F8_E5M2FNUZ"
            | "I16"
            | "U16"
            | "F16"
            | "BF16"
            | "I32"
            | "U32"
            | "F32"
            | "C64"
            | "F64"
            | "I64"
            | "U64"
    )
}

fn expected_byte_len(dtype: &str, shape: &[usize]) -> Option<u64> {
    let bits_per_element = match dtype {
        "F4" => 4,
        "F6_E2M3" | "F6_E3M2" => 6,
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => 8,
        "I16" | "U16" | "F16" | "BF16" => 16,
        "I32" | "U32" | "F32" => 32,
        "C64" | "F64" | "I64" | "U64" => 64,
        _ => return None,
    };
    let element_count = shape.iter().try_fold(1_u64, |count, &dimension| {
        count.checked_mul(u64::try_from(dimension).ok()?)
    })?;
    let bits = element_count.checked_mul(bits_per_element)?;
    (bits % 8 == 0).then_some(bits / 8)
}

#[cfg(test)]
mod tests {
    use std::{env, fs, path::PathBuf, process};

    use serde_json::json;

    use super::{MappedSafetensors, inspect_file};

    struct TempFile(PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn temp_file(name: &str, contents: Vec<u8>) -> TempFile {
        let path = env::temp_dir().join(format!(
            "modelq-safetensors-{name}-{}.safetensors",
            process::id()
        ));
        fs::write(&path, contents).expect("temporary fixture can be written");
        TempFile(path)
    }

    fn synthetic_safetensors() -> Vec<u8> {
        let header = json!({
            "__metadata__": { "format": "test" },
            "bias": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8]
            },
            "weight": {
                "dtype": "F16",
                "shape": [2, 2],
                "data_offsets": [8, 16]
            },
            "bf16": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [16, 20]
            }
        });
        let mut header = serde_json::to_vec(&header).expect("fixture metadata serializes");
        let padded_len = header.len().div_ceil(8) * 8;
        header.resize(padded_len, b' ');

        let bias = [1.25_f32, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let weight = [0xc000_u16, 0x3800, 0x3c00, 0x0000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let bf16 = [0x3f80_u16, 0xc000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();

        let mut file = Vec::with_capacity(8 + padded_len + 20);
        file.extend_from_slice(&(padded_len as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&bias);
        file.extend_from_slice(&weight);
        file.extend_from_slice(&bf16);
        file
    }

    #[test]
    fn inspects_synthetic_safetensors_metadata() {
        let file = temp_file("valid", synthetic_safetensors());

        let inspection = inspect_file(&file.0).expect("fixture metadata is valid");

        assert_eq!(inspection.tensors.len(), 3);
        assert_eq!(inspection.tensors[0].name, "bias");
        assert_eq!(inspection.tensors[0].dtype, "F32");
        assert_eq!(inspection.tensors[0].shape, [2]);
        assert_eq!(inspection.tensors[0].byte_len, 8);
        assert_eq!(inspection.tensors[1].name, "weight");
        assert_eq!(inspection.tensors[1].dtype, "F16");
        assert_eq!(inspection.tensors[1].shape, [2, 2]);
        assert_eq!(inspection.tensors[1].byte_len, 8);
        assert_eq!(inspection.tensors[2].name, "bf16");
        assert_eq!(inspection.tensors[2].dtype, "BF16");
        assert_eq!(inspection.tensors[2].shape, [2]);
        assert_eq!(inspection.tensors[2].byte_len, 4);
    }

    #[test]
    fn maps_fixture_values_without_copying_payloads() {
        let file = temp_file("mapped", synthetic_safetensors());

        let reader = MappedSafetensors::open(&file.0).expect("fixture maps successfully");

        assert_eq!(reader.file_size(), fs::metadata(&file.0).unwrap().len());
        assert_eq!(reader.tensors().len(), 3);
        assert_eq!(
            reader
                .tensor("bias")
                .expect("f32 tensor is viewable")
                .values()
                .collect::<Vec<_>>(),
            [1.25, -2.0]
        );
        assert_eq!(
            reader
                .tensor("weight")
                .expect("f16 tensor is viewable")
                .values()
                .collect::<Vec<_>>(),
            [-2.0, 0.5, 1.0, 0.0]
        );
        assert_eq!(
            reader
                .tensor("bf16")
                .expect("bf16 tensor is viewable")
                .values()
                .collect::<Vec<_>>(),
            [1.0, -2.0]
        );
    }

    #[test]
    fn rejects_malformed_safetensors_metadata_usefully() {
        let file = temp_file("malformed", b"bad".to_vec());

        let error = inspect_file(&file.0).expect_err("the malformed file must be rejected");
        assert!(error.to_string().contains("header smaller than 8 bytes"));
    }
}
