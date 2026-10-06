//! Sharded SafeTensors output.
//!
//! A sharded output is a directory holding `model-NNNNN-of-MMMMM.safetensors`
//! files plus `model.safetensors.index.json`, the same convention ADR 0003
//! accepts as input.  Each shard is written by the caller's existing
//! single-file writer over a [`SubsetSource`], so every shard is a complete,
//! self-describing file.  A tensor is never split across shards, and the
//! tensors derived from one source tensor (for example `.qdata` and its
//! scales) always share a shard.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs, io,
    path::{Path, PathBuf},
};

use modelq_core::tensor::TensorView;
use serde_json::{Map, Value};

use crate::safetensors::{SafetensorsError, TensorSource, TensorSummary};

/// File stem shared by every shard and the index.
pub const SHARD_STEM: &str = "model";
/// Name of the index file inside a sharded output directory.
pub const INDEX_FILE_NAME: &str = "model.safetensors.index.json";

/// Parses a shard size such as `500000000`, `500MB`, `2GB`, `64MiB`.
///
/// `KB`/`MB`/`GB` are decimal and `KiB`/`MiB`/`GiB` are binary.  A bare
/// integer is bytes.  Zero and overflowing sizes are rejected.
pub fn parse_shard_size(text: &str) -> Result<u64, ShardingError<std::convert::Infallible>> {
    let invalid = || ShardingError::InvalidShardSize {
        text: text.to_owned(),
    };
    let trimmed = text.trim();
    let split = trimmed
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (digits, suffix) = trimmed.split_at(split);
    let count: u64 = digits.parse().map_err(|_| invalid())?;
    let multiplier: u64 = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "kb" => 1_000,
        "mb" => 1_000_000,
        "gb" => 1_000_000_000,
        "kib" => 1 << 10,
        "mib" => 1 << 20,
        "gib" => 1 << 30,
        _ => return Err(invalid()),
    };
    match count.checked_mul(multiplier) {
        Some(bytes) if bytes > 0 => Ok(bytes),
        _ => Err(invalid()),
    }
}

/// Groups source tensors, in the given order, into shards of at most
/// `max_bytes` payload bytes each.
///
/// Each item is `(source_name, output_payload_bytes)`.  A source whose output
/// alone exceeds `max_bytes` still gets its own shard, because a tensor cannot
/// be split.  The result never contains an empty shard.
pub fn plan_shard_groups(sized_sources: &[(String, u64)], max_bytes: u64) -> Vec<Vec<String>> {
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_bytes = 0_u64;
    for (name, bytes) in sized_sources {
        let fits = current_bytes
            .checked_add(*bytes)
            .is_some_and(|total| total <= max_bytes);
        if !current.is_empty() && !fits {
            groups.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current.push(name.clone());
        current_bytes = current_bytes.saturating_add(*bytes);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Returns the shard file name for a zero-based shard index.
pub fn shard_file_name(index: usize, count: usize) -> String {
    format!("{SHARD_STEM}-{:05}-of-{count:05}.safetensors", index + 1)
}

/// A read-only view of some of another source's tensors.
pub struct SubsetSource<'a, S: TensorSource> {
    inner: &'a S,
    names: BTreeSet<String>,
    summaries: Vec<TensorSummary>,
}

impl<'a, S: TensorSource> SubsetSource<'a, S> {
    /// Restricts `inner` to the named tensors.  Unknown names are ignored.
    pub fn new(inner: &'a S, names: impl IntoIterator<Item = String>) -> Self {
        let names: BTreeSet<String> = names.into_iter().collect();
        let summaries = inner
            .tensor_summaries()
            .into_iter()
            .filter(|summary| names.contains(&summary.name))
            .collect();
        Self {
            inner,
            names,
            summaries,
        }
    }

    fn require(&self, name: &str) -> Result<(), SafetensorsError> {
        if self.names.contains(name) {
            Ok(())
        } else {
            Err(SafetensorsError::TensorNotFound {
                path: PathBuf::new(),
                name: name.to_owned(),
            })
        }
    }
}

impl<S: TensorSource> TensorSource for SubsetSource<'_, S> {
    fn source_paths(&self) -> Vec<PathBuf> {
        self.inner.source_paths()
    }

    fn tensor_summaries(&self) -> Vec<TensorSummary> {
        self.summaries.clone()
    }

    fn with_tensor_bytes<R>(
        &self,
        name: &str,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<R, SafetensorsError> {
        self.require(name)?;
        self.inner.with_tensor_bytes(name, f)
    }

    fn with_tensor<R>(
        &self,
        name: &str,
        f: impl FnOnce(TensorView<'_>) -> R,
    ) -> Result<R, SafetensorsError> {
        self.require(name)?;
        self.inner.with_tensor(name, f)
    }
}

/// Errors from sharded output.  `E` is the caller's single-file writer error.
#[derive(Debug)]
pub enum ShardingError<E> {
    /// The shard size text was not a positive size.
    InvalidShardSize { text: String },
    /// The output path exists and is not a directory.
    NotADirectory { path: PathBuf },
    /// The output directory exists and already has entries.
    DirectoryNotEmpty { path: PathBuf },
    /// There were no tensors to write.
    NothingToWrite,
    /// A shard or the index could not be written.
    Io { path: PathBuf, source: io::Error },
    /// The caller's shard writer failed.
    Writer(E),
}

impl<E: fmt::Display> fmt::Display for ShardingError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShardSize { text } => write!(
                formatter,
                "invalid shard size {text:?}; use a positive integer with an optional KB, MB, GB, KiB, MiB, or GiB suffix"
            ),
            Self::NotADirectory { path } => write!(
                formatter,
                "sharded output {} exists and is not a directory",
                path.display()
            ),
            Self::DirectoryNotEmpty { path } => write!(
                formatter,
                "sharded output directory {} is not empty; refusing to write into it",
                path.display()
            ),
            Self::NothingToWrite => formatter.write_str("there are no tensors to write"),
            Self::Io { path, source } => {
                write!(formatter, "could not write {}: {source}", path.display())
            }
            Self::Writer(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ShardingError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Writer(error) => Some(error),
            _ => None,
        }
    }
}

/// The files produced by a successful sharded write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedOutput {
    /// Output directory.
    pub directory: PathBuf,
    /// Shard files in order.
    pub shard_paths: Vec<PathBuf>,
    /// The index file.
    pub index_path: PathBuf,
    /// Sum of every output tensor's payload bytes.
    pub total_size: u64,
}

/// Writes one shard per group and then the index.
///
/// `write_shard` receives a [`SubsetSource`] holding exactly the group's
/// source tensors and the shard's final path, writes that file with the
/// single-file writer, and returns each output tensor name it wrote with its
/// payload byte length.  The directory must be absent or empty.  If anything
/// fails, every file this call created (and the directory, if this call
/// created it) is removed before the error is returned, so a failed run
/// leaves nothing behind.
pub fn write_sharded<S, E>(
    source: &S,
    directory: &Path,
    groups: &[Vec<String>],
    mut write_shard: impl FnMut(&SubsetSource<'_, S>, &Path) -> Result<Vec<(String, u64)>, E>,
) -> Result<ShardedOutput, ShardingError<E>>
where
    S: TensorSource,
{
    if groups.is_empty() || groups.iter().any(Vec::is_empty) {
        return Err(ShardingError::NothingToWrite);
    }
    let io_error = |path: &Path, source| ShardingError::Io {
        path: path.to_owned(),
        source,
    };

    let created_directory = match fs::metadata(directory) {
        Ok(metadata) if metadata.is_dir() => {
            let mut entries = fs::read_dir(directory).map_err(|e| io_error(directory, e))?;
            if entries.next().is_some() {
                return Err(ShardingError::DirectoryNotEmpty {
                    path: directory.to_owned(),
                });
            }
            false
        }
        Ok(_) => {
            return Err(ShardingError::NotADirectory {
                path: directory.to_owned(),
            });
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(directory).map_err(|e| io_error(directory, e))?;
            true
        }
        Err(error) => return Err(io_error(directory, error)),
    };

    let mut created: Vec<PathBuf> = Vec::new();
    let result = (|| {
        let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
        let mut total_size = 0_u64;
        let mut shard_paths = Vec::with_capacity(groups.len());
        for (index, group) in groups.iter().enumerate() {
            let file_name = shard_file_name(index, groups.len());
            let path = directory.join(&file_name);
            let subset = SubsetSource::new(source, group.iter().cloned());
            let written = write_shard(&subset, &path).map_err(ShardingError::Writer)?;
            created.push(path.clone());
            shard_paths.push(path);
            for (name, bytes) in written {
                total_size = total_size.saturating_add(bytes);
                weight_map.insert(name, file_name.clone());
            }
        }

        let index_path = directory.join(INDEX_FILE_NAME);
        let mut metadata = Map::new();
        metadata.insert("total_size".to_owned(), Value::from(total_size));
        let mut root = Map::new();
        root.insert("metadata".to_owned(), Value::Object(metadata));
        root.insert(
            "weight_map".to_owned(),
            Value::Object(
                weight_map
                    .into_iter()
                    .map(|(name, file)| (name, Value::String(file)))
                    .collect(),
            ),
        );
        let mut text = serde_json::to_string_pretty(&Value::Object(root))
            .expect("a JSON object with string keys serializes");
        text.push('\n');
        let temporary = directory.join(format!(".{INDEX_FILE_NAME}.tmp"));
        created.push(temporary.clone());
        fs::write(&temporary, text).map_err(|e| io_error(&temporary, e))?;
        fs::rename(&temporary, &index_path).map_err(|e| io_error(&index_path, e))?;
        created.pop();
        created.push(index_path.clone());
        Ok(ShardedOutput {
            directory: directory.to_owned(),
            shard_paths,
            index_path,
            total_size,
        })
    })();

    if result.is_err() {
        for path in &created {
            let _ = fs::remove_file(path);
        }
        if created_directory {
            let _ = fs::remove_dir(directory);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{parse_shard_size, plan_shard_groups, shard_file_name};

    fn sized(items: &[(&str, u64)]) -> Vec<(String, u64)> {
        items
            .iter()
            .map(|(name, bytes)| ((*name).to_owned(), *bytes))
            .collect()
    }

    #[test]
    fn parses_decimal_binary_and_plain_sizes() {
        assert_eq!(parse_shard_size("1500").unwrap(), 1_500);
        assert_eq!(parse_shard_size("500MB").unwrap(), 500_000_000);
        assert_eq!(parse_shard_size("2gb").unwrap(), 2_000_000_000);
        assert_eq!(parse_shard_size("64MiB").unwrap(), 64 << 20);
        assert_eq!(parse_shard_size(" 3 KiB ").unwrap(), 3 << 10);
        for bad in [
            "",
            "0",
            "0MB",
            "MB",
            "-5",
            "1.5GB",
            "10XB",
            "99999999999999999999",
        ] {
            assert!(parse_shard_size(bad).is_err(), "{bad:?} should be rejected");
        }
        assert!(parse_shard_size("18446744073709551615GB").is_err());
    }

    #[test]
    fn groups_greedily_in_order_and_never_splits_a_source() {
        let groups = plan_shard_groups(&sized(&[("a", 40), ("b", 40), ("c", 40), ("d", 10)]), 100);
        assert_eq!(groups, [vec!["a", "b"], vec!["c", "d"]]);

        // An oversize source gets its own shard, neighbors are not merged in.
        let groups = plan_shard_groups(&sized(&[("a", 10), ("big", 500), ("c", 10)]), 100);
        assert_eq!(groups, [vec!["a"], vec!["big"], vec!["c"]]);

        // Exactly-full shards are allowed.
        let groups = plan_shard_groups(&sized(&[("a", 50), ("b", 50), ("c", 1)]), 100);
        assert_eq!(groups, [vec!["a", "b"], vec!["c"]]);

        assert!(plan_shard_groups(&[], 100).is_empty());
    }

    #[test]
    fn names_shards_with_padded_one_based_ordinals() {
        assert_eq!(shard_file_name(0, 3), "model-00001-of-00003.safetensors");
        assert_eq!(shard_file_name(2, 3), "model-00003-of-00003.safetensors");
    }
}
