//! Where `quantize` writes: one file, or a sharded directory.

use std::path::{Path, PathBuf};

use modelq::io::{
    safetensors::TensorSource,
    sharded::SafetensorsInput,
    sharding::{
        ShardedOutput, ShardingError, SubsetSource, parse_shard_size, plan_shard_groups,
        write_sharded,
    },
};

/// The requested output layout.
#[derive(Debug, Clone)]
pub enum OutputTarget {
    /// One SafeTensors file at this path.
    File(PathBuf),
    /// A directory of shards plus an index, each shard holding at most
    /// `max_shard_bytes` of tensor payload (oversize tensors get their own).
    Sharded {
        directory: PathBuf,
        max_shard_bytes: u64,
    },
}

/// Files committed by a successful write.
pub struct WrittenOutput {
    pub paths: Vec<PathBuf>,
}

impl OutputTarget {
    /// Builds the target from `--output` and the optional `--max-shard-size`.
    pub fn from_args(output: PathBuf, max_shard_size: Option<&String>) -> Result<Self, String> {
        match max_shard_size {
            None => Ok(Self::File(output)),
            Some(text) => {
                let max_shard_bytes = parse_shard_size(text).map_err(|error| error.to_string())?;
                Ok(Self::Sharded {
                    directory: output,
                    max_shard_bytes,
                })
            }
        }
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::File(path) => path,
            Self::Sharded { directory, .. } => directory,
        }
    }

    /// Describes the layout for the "Writing output" line.
    pub fn describe(&self) -> String {
        match self {
            Self::File(path) => path.display().to_string(),
            Self::Sharded {
                directory,
                max_shard_bytes,
            } => format!(
                "{} (sharded, at most {max_shard_bytes} payload bytes per shard)",
                directory.display()
            ),
        }
    }

    /// Opens the committed output for validation.  The returned reader is a
    /// [`SafetensorsInput`], which validates the index against every shard
    /// when the output is sharded.
    pub fn open_output(&self) -> Result<SafetensorsInput, String> {
        SafetensorsInput::open(self.path())
            .map_err(|error| format!("output could not be reopened: {error}"))
    }

    /// Total size of the committed files in bytes.
    pub fn committed_bytes(written: &WrittenOutput) -> Result<u64, String> {
        written
            .paths
            .iter()
            .map(|path| {
                std::fs::metadata(path)
                    .map(|metadata| metadata.len())
                    .map_err(|error| format!("could not stat {}: {error}", path.display()))
            })
            .sum()
    }

    /// Writes the output with the caller's single-file writer.
    ///
    /// `sized_sources` lists every source tensor with the payload bytes its
    /// output occupies, in name order.  `write_one` writes one file for the
    /// given subset of sources and returns each output tensor name it wrote
    /// with its payload bytes.  For a single file the subset is everything.
    pub fn write<S, E>(
        &self,
        source: &S,
        sized_sources: &[(String, u64)],
        mut write_one: impl FnMut(&SubsetSource<'_, S>, &Path) -> Result<Vec<(String, u64)>, E>,
    ) -> Result<WrittenOutput, String>
    where
        S: TensorSource,
        E: std::fmt::Display,
    {
        match self {
            Self::File(path) => {
                let everything =
                    SubsetSource::new(source, sized_sources.iter().map(|(name, _)| name.clone()));
                write_one(&everything, path)
                    .map_err(|error| format!("could not write output: {error}"))?;
                Ok(WrittenOutput {
                    paths: vec![path.clone()],
                })
            }
            Self::Sharded {
                directory,
                max_shard_bytes,
            } => {
                let groups = plan_shard_groups(sized_sources, *max_shard_bytes);
                println!("Sharding: {} shard(s)", groups.len());
                let ShardedOutput {
                    mut shard_paths,
                    index_path,
                    total_size,
                    ..
                } = write_sharded(source, directory, &groups, &mut write_one).map_err(
                    |error: ShardingError<E>| format!("could not write output: {error}"),
                )?;
                println!("Index total_size: {total_size} payload bytes");
                shard_paths.push(index_path);
                Ok(WrittenOutput { paths: shard_paths })
            }
        }
    }
}
