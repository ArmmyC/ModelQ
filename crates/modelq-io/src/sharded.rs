//! Read-only, metadata-first SafeTensors input over one file or a sharded
//! checkpoint, as specified by ADR 0003.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs, io,
    path::{Path, PathBuf},
};

use modelq_core::tensor::TensorView;
use serde_json::Value;

use crate::safetensors::{MappedSafetensors, SafetensorsError, TensorSummary};

const INDEX_SUFFIX: &str = ".index.json";
const SAFETENSORS_SUFFIX: &str = ".safetensors";
const INDEX_DIRECTORY_SUFFIX: &str = ".safetensors.index.json";

/// A tensor in the logical catalog and the shard that owns its payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedTensorSummary {
    /// Source tensor metadata taken from the owning shard's header.
    pub summary: TensorSummary,
    /// Path of the shard file that stores the payload.
    pub shard: PathBuf,
}

/// Errors returned while discovering, validating, or reading sharded input.
#[derive(Debug)]
pub enum ShardedError {
    /// The input path is neither a regular file nor a directory.
    UnsupportedInputPath { path: PathBuf },
    /// A directory could not be listed or an index could not be read.
    Io { path: PathBuf, source: io::Error },
    /// A directory contains no index and no `.safetensors` file.
    NoSafetensorsFound { path: PathBuf },
    /// A directory contains several candidates; pass an explicit path.
    AmbiguousInput {
        path: PathBuf,
        candidates: Vec<PathBuf>,
    },
    /// The index is not valid JSON or does not match the accepted schema.
    InvalidIndex { path: PathBuf, message: String },
    /// A shard reference is not a plain basename.
    UnsafeShardReference { index: PathBuf, shard: String },
    /// The index and a shard disagree about which tensors the shard holds.
    TensorSetMismatch {
        index: PathBuf,
        shard: PathBuf,
        tensor: String,
        message: &'static str,
    },
    /// A tensor name appears in more than one shard.
    DuplicateTensor {
        index: PathBuf,
        tensor: String,
        first: PathBuf,
        second: PathBuf,
    },
    /// `metadata.total_size` differs from the validated payload byte total.
    TotalSizeMismatch {
        index: PathBuf,
        expected: u64,
        actual: u64,
    },
    /// The requested tensor is not in the logical catalog.
    TensorNotFound { name: String },
    /// A shard failed SafeTensors validation or typed-view creation.
    Safetensors(SafetensorsError),
}

impl fmt::Display for ShardedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedInputPath { path } => write!(
                formatter,
                "{} is neither a regular file nor a directory",
                path.display()
            ),
            Self::Io { path, source } => {
                write!(formatter, "could not read {}: {source}", path.display())
            }
            Self::NoSafetensorsFound { path } => write!(
                formatter,
                "directory {} contains no SafeTensors file or index",
                path.display()
            ),
            Self::AmbiguousInput { path, candidates } => {
                write!(
                    formatter,
                    "directory {} has several SafeTensors candidates; pass one explicitly:",
                    path.display()
                )?;
                for candidate in candidates {
                    write!(formatter, " {}", candidate.display())?;
                }
                Ok(())
            }
            Self::InvalidIndex { path, message } => {
                write!(formatter, "index {} is invalid: {message}", path.display())
            }
            Self::UnsafeShardReference { index, shard } => write!(
                formatter,
                "index {} references unsafe shard name {shard:?}; only plain file names are accepted",
                index.display()
            ),
            Self::TensorSetMismatch {
                index,
                shard,
                tensor,
                message,
            } => write!(
                formatter,
                "index {} and shard {} disagree on tensor {tensor:?}: {message}",
                index.display(),
                shard.display()
            ),
            Self::DuplicateTensor {
                index,
                tensor,
                first,
                second,
            } => write!(
                formatter,
                "index {}: tensor {tensor:?} appears in both {} and {}",
                index.display(),
                first.display(),
                second.display()
            ),
            Self::TotalSizeMismatch {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "index {} declares total_size {expected} but shard payloads total {actual} bytes",
                index.display()
            ),
            Self::TensorNotFound { name } => write!(formatter, "no tensor named {name:?}"),
            Self::Safetensors(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ShardedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Safetensors(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SafetensorsError> for ShardedError {
    fn from(error: SafetensorsError) -> Self {
        Self::Safetensors(error)
    }
}

/// A validated logical tensor catalog over one SafeTensors file or an index
/// plus its shards.
///
/// Opening validates every shard header but keeps no mapping alive. Payload
/// access maps the owning shard for the duration of one call, so at most one
/// shard payload is mapped at a time for sequential use and no view can
/// outlive its mapping.
#[derive(Debug)]
pub struct SafetensorsInput {
    tensors: Vec<ShardedTensorSummary>,
}

impl SafetensorsInput {
    /// Opens a file, directory, or index path following ADR 0003 discovery.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ShardedError> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).map_err(|_| ShardedError::UnsupportedInputPath {
            path: path.to_owned(),
        })?;
        if metadata.is_dir() {
            Self::open_directory(path)
        } else if metadata.is_file() {
            if has_suffix(path, INDEX_SUFFIX) {
                Self::open_index(path)
            } else {
                Self::open_single(path)
            }
        } else {
            Err(ShardedError::UnsupportedInputPath {
                path: path.to_owned(),
            })
        }
    }

    /// Returns the logical catalog in ascending tensor-name byte order.
    pub fn tensors(&self) -> &[ShardedTensorSummary] {
        &self.tensors
    }

    /// Calls `f` with one tensor's raw payload, mapping only its shard.
    pub fn with_tensor_bytes<R>(
        &self,
        name: &str,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<R, ShardedError> {
        let entry = self.find(name)?;
        let shard = MappedSafetensors::open(&entry.shard)?;
        Ok(f(shard.tensor_bytes(name)?))
    }

    /// Calls `f` with one tensor's typed view, mapping only its shard.
    pub fn with_tensor<R>(
        &self,
        name: &str,
        f: impl FnOnce(TensorView<'_>) -> R,
    ) -> Result<R, ShardedError> {
        let entry = self.find(name)?;
        let shard = MappedSafetensors::open(&entry.shard)?;
        Ok(f(shard.tensor(name)?))
    }

    fn find(&self, name: &str) -> Result<&ShardedTensorSummary, ShardedError> {
        self.tensors
            .binary_search_by(|entry| entry.summary.name.as_str().cmp(name))
            .map(|index| &self.tensors[index])
            .map_err(|_| ShardedError::TensorNotFound {
                name: name.to_owned(),
            })
    }

    fn open_directory(directory: &Path) -> Result<Self, ShardedError> {
        let io_error = |source| ShardedError::Io {
            path: directory.to_owned(),
            source,
        };
        let mut indexes = Vec::new();
        let mut files = Vec::new();
        for entry in fs::read_dir(directory).map_err(io_error)? {
            let path = entry.map_err(io_error)?.path();
            if !path.is_file() {
                continue;
            }
            if has_suffix(&path, INDEX_DIRECTORY_SUFFIX) {
                indexes.push(path);
            } else if has_suffix(&path, SAFETENSORS_SUFFIX) {
                files.push(path);
            }
        }
        indexes.sort();
        files.sort();

        let mut candidates = if indexes.is_empty() { files } else { indexes };
        match candidates.len() {
            0 => Err(ShardedError::NoSafetensorsFound {
                path: directory.to_owned(),
            }),
            1 => {
                let path = candidates.remove(0);
                if has_suffix(&path, INDEX_SUFFIX) {
                    Self::open_index(&path)
                } else {
                    Self::open_single(&path)
                }
            }
            _ => Err(ShardedError::AmbiguousInput {
                path: directory.to_owned(),
                candidates,
            }),
        }
    }

    fn open_single(path: &Path) -> Result<Self, ShardedError> {
        let shard = MappedSafetensors::open(path)?;
        let mut tensors: Vec<_> = shard
            .tensors()
            .map(|summary| ShardedTensorSummary {
                summary: summary.clone(),
                shard: path.to_owned(),
            })
            .collect();
        sort_by_name(&mut tensors);
        Ok(Self { tensors })
    }

    fn open_index(index_path: &Path) -> Result<Self, ShardedError> {
        let index = parse_index(index_path)?;
        let directory = index_path.parent().unwrap_or_else(|| Path::new("."));

        let mut expected_by_shard: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for (tensor, shard) in &index.weight_map {
            expected_by_shard
                .entry(shard.as_str())
                .or_default()
                .insert(tensor.as_str());
        }

        let mut tensors = Vec::with_capacity(index.weight_map.len());
        let mut owner: BTreeMap<String, PathBuf> = BTreeMap::new();
        let mut total: u64 = 0;
        for (shard_name, expected) in &expected_by_shard {
            let shard_path = directory.join(shard_name);
            let shard = MappedSafetensors::open(&shard_path)?;
            for summary in shard.tensors() {
                if let Some(first) = owner.get(&summary.name) {
                    return Err(ShardedError::DuplicateTensor {
                        index: index_path.to_owned(),
                        tensor: summary.name.clone(),
                        first: first.clone(),
                        second: shard_path,
                    });
                }
                if !expected.contains(summary.name.as_str()) {
                    return Err(mismatch(
                        index_path,
                        &shard_path,
                        &summary.name,
                        "tensor is in the shard but not mapped to it by the index",
                    ));
                }
                owner.insert(summary.name.clone(), shard_path.clone());
                total = total.checked_add(summary.byte_len).ok_or_else(|| {
                    ShardedError::InvalidIndex {
                        path: index_path.to_owned(),
                        message: "payload byte total overflows u64".to_owned(),
                    }
                })?;
                tensors.push(ShardedTensorSummary {
                    summary: summary.clone(),
                    shard: shard_path.clone(),
                });
            }
            if let Some(missing) = expected.iter().find(|name| !owner.contains_key(**name)) {
                return Err(mismatch(
                    index_path,
                    &shard_path,
                    missing,
                    "index maps the tensor to this shard but the shard lacks it",
                ));
            }
        }

        if let Some(expected) = index.total_size
            && expected != total
        {
            return Err(ShardedError::TotalSizeMismatch {
                index: index_path.to_owned(),
                expected,
                actual: total,
            });
        }

        sort_by_name(&mut tensors);
        Ok(Self { tensors })
    }
}

struct Index {
    weight_map: BTreeMap<String, String>,
    total_size: Option<u64>,
}

fn parse_index(path: &Path) -> Result<Index, ShardedError> {
    let invalid = |message: String| ShardedError::InvalidIndex {
        path: path.to_owned(),
        message,
    };
    let bytes = fs::read(path).map_err(|source| ShardedError::Io {
        path: path.to_owned(),
        source,
    })?;
    let root: Value = serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))?;
    let root = root
        .as_object()
        .ok_or_else(|| invalid("root must be a JSON object".to_owned()))?;

    let total_size = match root.get("metadata") {
        None => None,
        Some(Value::Object(metadata)) => match metadata.get("total_size") {
            None => None,
            Some(value) => Some(value.as_u64().ok_or_else(|| {
                invalid("metadata.total_size must be a non-negative integer".to_owned())
            })?),
        },
        Some(_) => return Err(invalid("metadata must be a JSON object".to_owned())),
    };

    let map = root
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("weight_map must be a JSON object".to_owned()))?;
    let mut weight_map = BTreeMap::new();
    for (tensor, shard) in map {
        if tensor == "__metadata__" {
            return Err(invalid("__metadata__ cannot be a tensor name".to_owned()));
        }
        let shard = shard
            .as_str()
            .filter(|shard| !shard.is_empty())
            .ok_or_else(|| invalid(format!("shard for {tensor:?} must be a non-empty string")))?;
        if !is_safe_basename(shard) {
            return Err(ShardedError::UnsafeShardReference {
                index: path.to_owned(),
                shard: shard.to_owned(),
            });
        }
        weight_map.insert(tensor.clone(), shard.to_owned());
    }
    Ok(Index {
        weight_map,
        total_size,
    })
}

/// Accepts only a plain file name: no separators, drive prefixes, or dot
/// components.
fn is_safe_basename(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', ':', '\0'])
}

fn mismatch(index: &Path, shard: &Path, tensor: &str, message: &'static str) -> ShardedError {
    ShardedError::TensorSetMismatch {
        index: index.to_owned(),
        shard: shard.to_owned(),
        tensor: tensor.to_owned(),
        message,
    }
}

fn sort_by_name(tensors: &mut [ShardedTensorSummary]) {
    tensors.sort_by(|left, right| left.summary.name.cmp(&right.summary.name));
}

fn has_suffix(path: &Path, suffix: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(suffix))
}
