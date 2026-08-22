//! Minimal GGUF v3 encoding and inspection for one Q8_0 tensor.
//!
//! This is a compatibility spike, not a general GGUF implementation.  It
//! writes the standard header, the three metadata values needed to describe
//! this fixture, one tensor-info record, and an aligned Q8_0 data section.

use std::{
    fmt,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};

use modelq_quant::gguf_q8_0::{
    BLOCK_BYTES, BLOCK_ELEMENTS, Q8_0Error, QuantizedQ8_0, quantize_shaped,
};

pub use modelq_quant::gguf_q8_0::GGML_TYPE_Q8_0;

/// GGUF's current on-disk version.
pub const GGUF_VERSION: u32 = 3;
/// Alignment used by the reference GGUF writer for tensor data.
pub const DEFAULT_ALIGNMENT: usize = 32;
/// GGUF value type ID for a UINT32 metadata value.
pub const GGUF_TYPE_UINT32: u32 = 4;
/// GGUF value type ID for a UTF-8 string metadata value.
pub const GGUF_TYPE_STRING: u32 = 8;
/// GGML quantization-version value used by the current Q8_0 reference.
pub const GGML_QUANTIZATION_VERSION: u32 = 2;
/// GGML file-type value for a mostly-Q8_0 file.
pub const GGML_FTYPE_MOSTLY_Q8_0: u32 = 7;

/// A single tensor record discovered in a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorSummary {
    /// Tensor name.
    pub name: String,
    /// Tensor dimensions in ModelQ's conventional order.
    pub shape: Vec<u64>,
    /// GGML tensor type ID.
    pub ggml_type: u32,
    /// Offset relative to the aligned GGUF data section.
    pub offset: u64,
    /// Serialized tensor payload length, excluding per-tensor padding.
    pub byte_len: u64,
}

/// Metadata and tensor records discovered in a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufSummary {
    /// GGUF format version.
    pub version: u32,
    /// Number of tensor records declared in the header.
    pub tensor_count: u64,
    /// Number of metadata key/value pairs declared in the header.
    pub kv_count: u64,
    /// Tensor-data alignment in bytes.
    pub alignment: u32,
    /// GGML quantization algorithm version.
    pub quantization_version: u32,
    /// GGML file-type metadata value.
    pub file_type: u32,
    /// Absolute byte offset where the aligned tensor data section begins.
    pub data_offset: u64,
    /// Tensor records in the order stored in the file.
    pub tensors: Vec<GgufTensorSummary>,
}

/// Errors returned by the minimal GGUF writer and inspector.
#[derive(Debug)]
pub enum GgufError {
    /// The destination already exists; writing never replaces it.
    DestinationExists { path: PathBuf },
    /// A file operation failed.
    Io { path: PathBuf, source: io::Error },
    /// Q8_0 source validation or quantization failed.
    Quantization { source: Q8_0Error },
    /// Tensor names must be non-empty UTF-8 strings.
    InvalidTensorName,
    /// The GGUF bytes are malformed or do not describe this spike's format.
    Malformed { message: String },
    /// A GGUF version other than the supported version was found.
    UnsupportedVersion { version: u32 },
    /// A metadata type outside this focused parser was found.
    UnsupportedMetadataType { type_id: u32 },
    /// A tensor type other than Q8_0 was found.
    UnsupportedTensorType { type_id: u32 },
}

impl fmt::Display for GgufError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DestinationExists { path } => {
                write!(
                    formatter,
                    "GGUF destination {} already exists",
                    path.display()
                )
            }
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "GGUF I/O failed for {}: {source}",
                    path.display()
                )
            }
            Self::Quantization { source } => write!(formatter, "could not quantize Q8_0: {source}"),
            Self::InvalidTensorName => formatter.write_str("GGUF tensor name must not be empty"),
            Self::Malformed { message } => write!(formatter, "malformed GGUF: {message}"),
            Self::UnsupportedVersion { version } => {
                write!(
                    formatter,
                    "unsupported GGUF version {version}; expected {GGUF_VERSION}"
                )
            }
            Self::UnsupportedMetadataType { type_id } => {
                write!(formatter, "unsupported GGUF metadata type {type_id}")
            }
            Self::UnsupportedTensorType { type_id } => {
                write!(
                    formatter,
                    "unsupported GGML tensor type {type_id}; expected Q8_0"
                )
            }
        }
    }
}

impl std::error::Error for GgufError {}

impl From<Q8_0Error> for GgufError {
    fn from(source: Q8_0Error) -> Self {
        Self::Quantization { source }
    }
}

/// Encodes one shaped F32 tensor as a minimal GGUF v3 Q8_0 fixture.
pub fn encode_q8_0(name: &str, shape: &[usize], values: &[f32]) -> Result<Vec<u8>, GgufError> {
    if name.is_empty() {
        return Err(GgufError::InvalidTensorName);
    }
    let quantized = quantize_shaped(values, shape)?;
    let dimensions = shape
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GgufError::Malformed {
            message: "tensor dimension does not fit in u64".to_owned(),
        })?;

    let mut bytes = Vec::with_capacity(256 + quantized.bytes().len());
    bytes.extend_from_slice(b"GGUF");
    put_u32(&mut bytes, GGUF_VERSION);
    put_u64(&mut bytes, 1);
    put_u64(&mut bytes, 3);

    put_string(&mut bytes, "general.alignment");
    put_u32(&mut bytes, GGUF_TYPE_UINT32);
    put_u32(&mut bytes, DEFAULT_ALIGNMENT as u32);
    put_string(&mut bytes, "general.quantization_version");
    put_u32(&mut bytes, GGUF_TYPE_UINT32);
    put_u32(&mut bytes, GGML_QUANTIZATION_VERSION);
    put_string(&mut bytes, "general.file_type");
    put_u32(&mut bytes, GGUF_TYPE_UINT32);
    put_u32(&mut bytes, GGML_FTYPE_MOSTLY_Q8_0);

    put_string(&mut bytes, name);
    put_u32(
        &mut bytes,
        u32::try_from(dimensions.len()).map_err(|_| GgufError::Malformed {
            message: "tensor rank does not fit in u32".to_owned(),
        })?,
    );
    for &dimension in dimensions.iter().rev() {
        put_u64(&mut bytes, dimension);
    }
    put_u32(&mut bytes, GGML_TYPE_Q8_0);
    put_u64(&mut bytes, 0);

    pad_to_alignment(&mut bytes, DEFAULT_ALIGNMENT);
    bytes.extend_from_slice(quantized.bytes());
    pad_to_alignment(&mut bytes, DEFAULT_ALIGNMENT);
    Ok(bytes)
}

/// Writes one Q8_0 fixture without replacing an existing destination.
pub fn write_q8_0(
    path: impl AsRef<Path>,
    name: &str,
    shape: &[usize],
    values: &[f32],
) -> Result<(), GgufError> {
    let path = path.as_ref();
    let bytes = encode_q8_0(name, shape, values)?;
    let mut file = match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
            return Err(GgufError::DestinationExists {
                path: path.to_owned(),
            });
        }
        Err(source) => {
            return Err(GgufError::Io {
                path: path.to_owned(),
                source,
            });
        }
    };
    if let Err(source) = file.write_all(&bytes) {
        let _ = fs::remove_file(path);
        return Err(GgufError::Io {
            path: path.to_owned(),
            source,
        });
    }
    Ok(())
}

/// Inspects the header, tensor info, alignment, and bounds of a GGUF fixture.
pub fn inspect(bytes: &[u8]) -> Result<GgufSummary, GgufError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.read(4)? != b"GGUF" {
        return Err(malformed("magic is not GGUF"));
    }
    let version = cursor.u32()?;
    if version != GGUF_VERSION {
        return Err(GgufError::UnsupportedVersion { version });
    }
    let tensor_count = cursor.u64()?;
    let kv_count = cursor.u64()?;
    if tensor_count > bytes.len() as u64 || kv_count > bytes.len() as u64 {
        return Err(malformed("declared record count is too large"));
    }

    let mut alignment = None;
    let mut quantization_version = None;
    let mut file_type = None;
    for _ in 0..kv_count {
        let key = cursor.string()?;
        let type_id = cursor.u32()?;
        match type_id {
            GGUF_TYPE_UINT32 => {
                let value = cursor.u32()?;
                match key.as_str() {
                    "general.alignment" => alignment = Some(value),
                    "general.quantization_version" => quantization_version = Some(value),
                    "general.file_type" => file_type = Some(value),
                    _ => {}
                }
            }
            GGUF_TYPE_STRING => {
                let _ = cursor.string()?;
            }
            _ => return Err(GgufError::UnsupportedMetadataType { type_id }),
        }
    }
    let alignment = alignment.ok_or_else(|| malformed("general.alignment is missing"))?;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(malformed(
            "general.alignment must be a non-zero power of two",
        ));
    }
    let quantization_version =
        quantization_version.ok_or_else(|| malformed("general.quantization_version is missing"))?;
    let file_type = file_type.ok_or_else(|| malformed("general.file_type is missing"))?;

    let tensor_count_usize = usize::try_from(tensor_count)
        .map_err(|_| malformed("tensor count does not fit in usize"))?;
    let alignment_usize =
        usize::try_from(alignment).map_err(|_| malformed("alignment does not fit in usize"))?;
    let mut tensors = Vec::with_capacity(tensor_count_usize);
    for _ in 0..tensor_count_usize {
        let name = cursor.string()?;
        if name.is_empty() {
            return Err(GgufError::InvalidTensorName);
        }
        let rank = usize::try_from(cursor.u32()?)
            .map_err(|_| malformed("tensor rank does not fit in usize"))?;
        if rank == 0 {
            return Err(malformed("Q8_0 tensors must have at least one dimension"));
        }
        let mut file_dimensions = Vec::with_capacity(rank);
        for _ in 0..rank {
            file_dimensions.push(cursor.u64()?);
        }
        let shape = file_dimensions.iter().rev().copied().collect::<Vec<_>>();
        let ggml_type = cursor.u32()?;
        if ggml_type != GGML_TYPE_Q8_0 {
            return Err(GgufError::UnsupportedTensorType { type_id: ggml_type });
        }
        let offset = cursor.u64()?;
        if offset % u64::from(alignment) != 0 {
            return Err(malformed("tensor offset is not aligned"));
        }
        let elements = shape.iter().try_fold(1_u64, |count, &dimension| {
            if dimension == 0 {
                None
            } else {
                count.checked_mul(dimension)
            }
        });
        let elements = elements.ok_or_else(|| malformed("tensor shape is empty or overflows"))?;
        if shape
            .last()
            .is_none_or(|&dimension| dimension % BLOCK_ELEMENTS as u64 != 0)
            || elements % BLOCK_ELEMENTS as u64 != 0
        {
            return Err(malformed("Q8_0 tensor shape is not block aligned"));
        }
        let byte_len = (elements / BLOCK_ELEMENTS as u64)
            .checked_mul(BLOCK_BYTES as u64)
            .ok_or_else(|| malformed("Q8_0 byte length overflows u64"))?;
        tensors.push(GgufTensorSummary {
            name,
            shape,
            ggml_type,
            offset,
            byte_len,
        });
    }

    let data_offset = align_up(cursor.position(), alignment_usize)
        .ok_or_else(|| malformed("tensor data offset overflows usize"))?;
    if data_offset > bytes.len() {
        return Err(malformed("tensor data offset is outside the file"));
    }
    let mut ranges = Vec::with_capacity(tensors.len());
    for tensor in &tensors {
        let start_offset = usize::try_from(tensor.offset)
            .map_err(|_| malformed("tensor offset does not fit in usize"))?;
        let start = data_offset
            .checked_add(start_offset)
            .ok_or_else(|| malformed("tensor start overflows usize"))?;
        let length = usize::try_from(tensor.byte_len)
            .map_err(|_| malformed("tensor byte length does not fit in usize"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| malformed("tensor end overflows usize"))?;
        if end > bytes.len() {
            return Err(malformed("tensor payload extends past the file"));
        }
        ranges.push((start, end));
    }
    ranges.sort_unstable_by_key(|&(start, _)| start);
    if ranges.windows(2).any(|window| window[1].0 < window[0].1) {
        return Err(malformed("tensor payloads overlap"));
    }

    Ok(GgufSummary {
        version,
        tensor_count,
        kv_count,
        alignment,
        quantization_version,
        file_type,
        data_offset: data_offset as u64,
        tensors,
    })
}

/// Inspects and extracts the one Q8_0 tensor from a fixture.
pub fn read_q8_0(bytes: &[u8]) -> Result<(GgufSummary, QuantizedQ8_0), GgufError> {
    let summary = inspect(bytes)?;
    let tensor = summary
        .tensors
        .first()
        .ok_or_else(|| malformed("fixture contains no tensor"))?;
    if summary.tensors.len() != 1 {
        return Err(malformed("the focused reader expects exactly one tensor"));
    }
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| malformed("tensor element count overflows u64"))?;
    let elements = usize::try_from(elements)
        .map_err(|_| malformed("tensor element count does not fit in usize"))?;
    let start = usize::try_from(summary.data_offset)
        .ok()
        .and_then(|offset| usize::try_from(tensor.offset).ok()?.checked_add(offset))
        .ok_or_else(|| malformed("tensor start does not fit in usize"))?;
    let length = usize::try_from(tensor.byte_len)
        .map_err(|_| malformed("tensor byte length does not fit in usize"))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| malformed("tensor end does not fit in usize"))?;
    let data = bytes
        .get(start..end)
        .ok_or_else(|| malformed("tensor payload is outside the file"))?
        .to_vec();
    let quantized = QuantizedQ8_0::from_bytes(data, elements)?;
    Ok((summary, quantized))
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

fn pad_to_alignment(bytes: &mut Vec<u8>, alignment: usize) {
    if let Some(aligned) = align_up(bytes.len(), alignment) {
        bytes.resize(aligned, 0);
    }
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn malformed(message: impl Into<String>) -> GgufError {
    GgufError::Malformed {
        message: message.into(),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn read(&mut self, length: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| malformed("cursor position overflows usize"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| malformed("unexpected end of GGUF data"))?;
        self.position = end;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(
            self.read(4)?.try_into().expect("length is four bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(
            self.read(8)?.try_into().expect("length is eight bytes"),
        ))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let length = usize::try_from(self.u64()?)
            .map_err(|_| malformed("GGUF string length does not fit in usize"))?;
        let bytes = self.read(length)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| malformed("GGUF string is not valid UTF-8"))
    }
}
