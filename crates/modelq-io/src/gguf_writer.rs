//! A general GGUF v3 writer for runtime-compatible model files (ADR 0032).
//!
//! The layout follows the GGUF specification as llama.cpp reads it: the magic
//! `GGUF`, version 3, the tensor and key-value counts, the key-value records,
//! one tensor-info record per tensor, padding to the alignment, and then each
//! tensor's data at an offset that is a multiple of the alignment. Dimensions
//! are written in GGML order, which is the reverse of the row-major shape.
//!
//! The writer knows nothing about models. The Qwen2 exporter in
//! [`crate::gguf_qwen2`] decides what goes in.

use std::{
    fmt, fs,
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
};

/// The GGUF version written by this module.
pub const GGUF_VERSION: u32 = 3;
/// The tensor-data and general alignment, in bytes.
pub const ALIGNMENT: u64 = 32;

/// GGUF metadata value type codes.
const TYPE_U32: u32 = 4;
const TYPE_I32: u32 = 5;
const TYPE_F32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;

/// The GGML tensor types this writer can emit.
pub const GGML_TYPE_F32: u32 = 0;
/// The GGML Q8_0 type: 32-value blocks of one F16 scale and 32 signed bytes.
pub const GGML_TYPE_Q8_0: u32 = 8;

/// A metadata value, with the GGUF type it is written as.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// An unsigned 32-bit integer.
    U32(u32),
    /// A signed 32-bit integer.
    I32(i32),
    /// A 32-bit float.
    F32(f32),
    /// A boolean.
    Bool(bool),
    /// A UTF-8 string.
    String(String),
    /// An array of strings.
    StringArray(Vec<String>),
    /// An array of signed 32-bit integers.
    I32Array(Vec<i32>),
}

/// One tensor: its name, its GGML dimensions, its type and its encoded data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRecord {
    /// The GGUF tensor name, for example `blk.0.attn_q.weight`.
    pub name: String,
    /// Dimensions in GGML order (fastest-varying first).
    pub dimensions: Vec<u64>,
    /// The GGML type code of `data`.
    pub ggml_type: u32,
    /// The encoded tensor data.
    pub data: Vec<u8>,
}

/// Errors from building or writing a GGUF file.
#[derive(Debug)]
pub enum GgufWriteError {
    /// A metadata key was given twice.
    DuplicateKey { key: String },
    /// A tensor name was given twice.
    DuplicateTensor { name: String },
    /// A name or key is empty.
    EmptyName,
    /// A tensor has no dimensions or a zero dimension.
    InvalidDimensions { name: String },
    /// A count or offset does not fit in the format's integer type.
    TooLarge { what: &'static str },
    /// The destination already exists; the writer never replaces a file.
    DestinationExists { path: String },
    /// The file could not be written.
    Io { path: String, source: io::Error },
}

impl fmt::Display for GgufWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateKey { key } => write!(formatter, "metadata key {key:?} is given twice"),
            Self::DuplicateTensor { name } => write!(formatter, "tensor {name:?} is given twice"),
            Self::EmptyName => formatter.write_str("a metadata key or tensor name is empty"),
            Self::InvalidDimensions { name } => {
                write!(
                    formatter,
                    "tensor {name:?} has no dimensions or a zero dimension"
                )
            }
            Self::TooLarge { what } => write!(formatter, "{what} does not fit in the GGUF format"),
            Self::DestinationExists { path } => {
                write!(
                    formatter,
                    "{path} already exists; GGUF output is never replaced"
                )
            }
            Self::Io { path, source } => write!(formatter, "could not write {path}: {source}"),
        }
    }
}

impl std::error::Error for GgufWriteError {}

/// A GGUF file under construction.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct GgufFile {
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<TensorRecord>,
}

impl GgufFile {
    /// An empty file.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one metadata entry. Keys must be unique and non-empty.
    pub fn add_metadata(&mut self, key: &str, value: MetadataValue) -> Result<(), GgufWriteError> {
        if key.is_empty() {
            return Err(GgufWriteError::EmptyName);
        }
        if self.metadata.iter().any(|(existing, _)| existing == key) {
            return Err(GgufWriteError::DuplicateKey {
                key: key.to_owned(),
            });
        }
        self.metadata.push((key.to_owned(), value));
        Ok(())
    }

    /// Adds one tensor. Names must be unique and non-empty, and every dimension
    /// must be positive.
    pub fn add_tensor(&mut self, tensor: TensorRecord) -> Result<(), GgufWriteError> {
        if tensor.name.is_empty() {
            return Err(GgufWriteError::EmptyName);
        }
        if self
            .tensors
            .iter()
            .any(|existing| existing.name == tensor.name)
        {
            return Err(GgufWriteError::DuplicateTensor { name: tensor.name });
        }
        if tensor.dimensions.is_empty() || tensor.dimensions.contains(&0) {
            return Err(GgufWriteError::InvalidDimensions { name: tensor.name });
        }
        self.tensors.push(tensor);
        Ok(())
    }

    /// The metadata entries, in insertion order.
    pub fn metadata(&self) -> &[(String, MetadataValue)] {
        &self.metadata
    }

    /// The tensors, in insertion order.
    pub fn tensors(&self) -> &[TensorRecord] {
        &self.tensors
    }

    /// Serializes the file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, GgufWriteError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        put_u32(&mut bytes, GGUF_VERSION);
        put_u64(&mut bytes, count(self.tensors.len(), "the tensor count")?);
        put_u64(
            &mut bytes,
            count(self.metadata.len(), "the metadata count")?,
        );

        for (key, value) in &self.metadata {
            put_string(&mut bytes, key);
            put_value(&mut bytes, value)?;
        }

        // Each tensor's offset is relative to the data section, and every
        // offset is a multiple of the alignment.
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut cursor = 0_u64;
        for tensor in &self.tensors {
            let offset = align_up(cursor, ALIGNMENT).ok_or(GgufWriteError::TooLarge {
                what: "a tensor offset",
            })?;
            offsets.push(offset);
            cursor = offset
                .checked_add(count(tensor.data.len(), "a tensor size")?)
                .ok_or(GgufWriteError::TooLarge {
                    what: "the tensor data",
                })?;
        }

        for (tensor, offset) in self.tensors.iter().zip(&offsets) {
            put_string(&mut bytes, &tensor.name);
            put_u32(
                &mut bytes,
                u32::try_from(tensor.dimensions.len()).map_err(|_| GgufWriteError::TooLarge {
                    what: "a tensor rank",
                })?,
            );
            for &dimension in &tensor.dimensions {
                put_u64(&mut bytes, dimension);
            }
            put_u32(&mut bytes, tensor.ggml_type);
            put_u64(&mut bytes, *offset);
        }

        pad_to(&mut bytes, ALIGNMENT as usize);
        let data_start = bytes.len();
        for (tensor, offset) in self.tensors.iter().zip(&offsets) {
            let position = data_start
                + usize::try_from(*offset).map_err(|_| GgufWriteError::TooLarge {
                    what: "a tensor offset",
                })?;
            bytes.resize(position, 0);
            bytes.extend_from_slice(&tensor.data);
        }
        pad_to(&mut bytes, ALIGNMENT as usize);
        Ok(bytes)
    }

    /// Writes the file to a new path. An existing destination is refused.
    pub fn write_new(&self, path: impl AsRef<Path>) -> Result<(), GgufWriteError> {
        let path = path.as_ref();
        let bytes = self.to_bytes()?;
        let display = path.display().to_string();
        let mut file = match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => file,
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                return Err(GgufWriteError::DestinationExists { path: display });
            }
            Err(source) => {
                return Err(GgufWriteError::Io {
                    path: display,
                    source,
                });
            }
        };
        if let Err(source) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
            drop(file);
            let _ = fs::remove_file(path);
            return Err(GgufWriteError::Io {
                path: display,
                source,
            });
        }
        Ok(())
    }
}

fn put_value(bytes: &mut Vec<u8>, value: &MetadataValue) -> Result<(), GgufWriteError> {
    match value {
        MetadataValue::U32(value) => {
            put_u32(bytes, TYPE_U32);
            put_u32(bytes, *value);
        }
        MetadataValue::I32(value) => {
            put_u32(bytes, TYPE_I32);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        MetadataValue::F32(value) => {
            put_u32(bytes, TYPE_F32);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        MetadataValue::Bool(value) => {
            put_u32(bytes, TYPE_BOOL);
            bytes.push(u8::from(*value));
        }
        MetadataValue::String(value) => {
            put_u32(bytes, TYPE_STRING);
            put_string(bytes, value);
        }
        MetadataValue::StringArray(values) => {
            put_u32(bytes, TYPE_ARRAY);
            put_u32(bytes, TYPE_STRING);
            put_u64(bytes, count(values.len(), "an array length")?);
            for value in values {
                put_string(bytes, value);
            }
        }
        MetadataValue::I32Array(values) => {
            put_u32(bytes, TYPE_ARRAY);
            put_u32(bytes, TYPE_I32);
            put_u64(bytes, count(values.len(), "an array length")?);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(())
}

fn count(value: usize, what: &'static str) -> Result<u64, GgufWriteError> {
    u64::try_from(value).map_err(|_| GgufWriteError::TooLarge { what })
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    put_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn pad_to(bytes: &mut Vec<u8>, alignment: usize) {
    while bytes.len() % alignment != 0 {
        bytes.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, dimensions: Vec<u64>, data: Vec<u8>) -> TensorRecord {
        TensorRecord {
            name: name.to_owned(),
            dimensions,
            ggml_type: GGML_TYPE_F32,
            data,
        }
    }

    #[test]
    fn header_counts_and_magic_are_written_first() {
        let mut file = GgufFile::new();
        file.add_metadata(
            "general.architecture",
            MetadataValue::String("qwen2".to_owned()),
        )
        .unwrap();
        file.add_tensor(tensor("a", vec![2], vec![0; 8])).unwrap();
        let bytes = file.to_bytes().unwrap();
        assert_eq!(&bytes[0..4], b"GGUF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 3);
        assert_eq!(
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            1,
            "tensor count"
        );
        assert_eq!(
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            1,
            "key-value count"
        );
    }

    #[test]
    fn every_tensor_offset_is_aligned_and_data_follows_the_declared_order() {
        let mut file = GgufFile::new();
        file.add_tensor(tensor("first", vec![3], vec![1; 12]))
            .unwrap();
        file.add_tensor(tensor("second", vec![2], vec![2; 8]))
            .unwrap();
        let bytes = file.to_bytes().unwrap();
        let data_start = data_start_of(&bytes, 2);
        let first_offset = offset_of(&bytes, "first");
        let second_offset = offset_of(&bytes, "second");
        assert_eq!(first_offset % ALIGNMENT, 0);
        assert_eq!(second_offset % ALIGNMENT, 0);
        assert_eq!(first_offset, 0);
        assert_eq!(
            second_offset, 32,
            "the second tensor starts at the next aligned offset"
        );
        assert_eq!(&bytes[data_start..data_start + 12], &[1; 12]);
        assert_eq!(&bytes[data_start + 32..data_start + 40], &[2; 8]);
        assert_eq!(
            bytes.len() % 32,
            0,
            "the file ends on an alignment boundary"
        );
    }

    #[test]
    fn duplicates_and_empty_names_are_refused() {
        let mut file = GgufFile::new();
        file.add_metadata("k", MetadataValue::U32(1)).unwrap();
        assert!(matches!(
            file.add_metadata("k", MetadataValue::U32(2)),
            Err(GgufWriteError::DuplicateKey { .. })
        ));
        file.add_tensor(tensor("t", vec![1], vec![0; 4])).unwrap();
        assert!(matches!(
            file.add_tensor(tensor("t", vec![1], vec![0; 4])),
            Err(GgufWriteError::DuplicateTensor { .. })
        ));
        assert!(matches!(
            file.add_tensor(tensor("", vec![1], vec![0; 4])),
            Err(GgufWriteError::EmptyName)
        ));
        assert!(matches!(
            file.add_tensor(tensor("z", vec![0], vec![])),
            Err(GgufWriteError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn string_arrays_encode_their_element_type_and_length() {
        let mut file = GgufFile::new();
        file.add_metadata(
            "tokens",
            MetadataValue::StringArray(vec!["a".to_owned(), "bc".to_owned()]),
        )
        .unwrap();
        let bytes = file.to_bytes().unwrap();
        let key_end = 24 + 8 + "tokens".len();
        assert_eq!(
            u32::from_le_bytes(bytes[key_end..key_end + 4].try_into().unwrap()),
            TYPE_ARRAY
        );
        assert_eq!(
            u32::from_le_bytes(bytes[key_end + 4..key_end + 8].try_into().unwrap()),
            TYPE_STRING
        );
        assert_eq!(
            u64::from_le_bytes(bytes[key_end + 8..key_end + 16].try_into().unwrap()),
            2
        );
    }

    #[test]
    fn existing_destinations_are_never_replaced() {
        let directory =
            std::env::temp_dir().join(format!("modelq-gguf-writer-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("out.gguf");
        fs::write(&path, b"previous").unwrap();
        let error = GgufFile::new().write_new(&path).unwrap_err();
        assert!(matches!(error, GgufWriteError::DestinationExists { .. }));
        assert_eq!(fs::read(&path).unwrap(), b"previous");
        let _ = fs::remove_dir_all(&directory);
    }

    /// Reads the tensor count, key-value count, and the position where the tensor data starts.
    fn data_start_of(bytes: &[u8], tensor_count: usize) -> usize {
        let mut position = 24;
        let key_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        for _ in 0..key_count {
            position = skip_kv(bytes, position);
        }
        for _ in 0..tensor_count {
            let name_length =
                u64::from_le_bytes(bytes[position..position + 8].try_into().unwrap()) as usize;
            position += 8 + name_length;
            let rank =
                u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap()) as usize;
            position += 4 + 8 * rank + 4 + 8;
        }
        position.div_ceil(32) * 32
    }

    fn offset_of(bytes: &[u8], wanted: &str) -> u64 {
        let mut position = 24;
        let key_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        for _ in 0..key_count {
            position = skip_kv(bytes, position);
        }
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        for _ in 0..tensor_count {
            let name_length =
                u64::from_le_bytes(bytes[position..position + 8].try_into().unwrap()) as usize;
            let name =
                std::str::from_utf8(&bytes[position + 8..position + 8 + name_length]).unwrap();
            position += 8 + name_length;
            let rank =
                u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap()) as usize;
            position += 4 + 8 * rank + 4;
            let offset = u64::from_le_bytes(bytes[position..position + 8].try_into().unwrap());
            position += 8;
            if name == wanted {
                return offset;
            }
        }
        panic!("tensor {wanted} not found");
    }

    fn skip_kv(bytes: &[u8], mut position: usize) -> usize {
        let key_length =
            u64::from_le_bytes(bytes[position..position + 8].try_into().unwrap()) as usize;
        position += 8 + key_length;
        let value_type = u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap());
        position += 4;
        match value_type {
            TYPE_U32 | TYPE_I32 | TYPE_F32 => position + 4,
            TYPE_BOOL => position + 1,
            TYPE_STRING => {
                let length =
                    u64::from_le_bytes(bytes[position..position + 8].try_into().unwrap()) as usize;
                position + 8 + length
            }
            TYPE_ARRAY => {
                let element_type =
                    u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap());
                let length =
                    u64::from_le_bytes(bytes[position + 4..position + 12].try_into().unwrap())
                        as usize;
                position += 12;
                for _ in 0..length {
                    position = match element_type {
                        TYPE_STRING => {
                            let size = u64::from_le_bytes(
                                bytes[position..position + 8].try_into().unwrap(),
                            ) as usize;
                            position + 8 + size
                        }
                        TYPE_I32 => position + 4,
                        other => panic!("unexpected array element type {other}"),
                    };
                }
                position
            }
            other => panic!("unexpected value type {other}"),
        }
    }
}
