//! The format registry: every representation `modelq quantize` can write, and
//! the status ModelQ can honestly claim for it (ADR 0028 section 2, ADR 0030).
//!
//! A status is a claim, and a claim is never higher than the evidence. The
//! levels are ordered from weakest to strongest:
//!
//! - `experimental`: encodes and decodes; round-trip and error tests exist, but
//!   the representation has no specification beyond ModelQ's own documents.
//!   Such formats need `--experimental` on the command line.
//! - `representation-valid`: a written specification with reference and
//!   exhaustive or property tests.
//! - `runtime-compatible`: a named runtime and version load the output.
//! - `hardware-validated`: the runtime was run on the named hardware.

use std::fmt;

/// The status ModelQ claims for a format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Experimental: no claim beyond the tests.
    Experimental,
    /// The representation matches its specification and reference tests.
    RepresentationValid,
    /// A named runtime loads the output.
    RuntimeCompatible {
        /// The runtime and its tested version.
        runtime: &'static str,
    },
    /// The named runtime has been run on the named hardware.
    HardwareValidated {
        /// The runtime and its tested version.
        runtime: &'static str,
        /// The hardware the runtime was run on.
        hardware: &'static str,
    },
}

impl Status {
    /// The status word shown to users.
    pub fn label(&self) -> String {
        match self {
            Self::Experimental => "experimental".to_owned(),
            Self::RepresentationValid => "representation-valid".to_owned(),
            Self::RuntimeCompatible { runtime } => format!("runtime-compatible: {runtime}"),
            Self::HardwareValidated { runtime, hardware } => {
                format!("hardware-validated: {runtime} on {hardware}")
            }
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.label())
    }
}

/// One format the CLI can write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatSpec {
    /// The name given to `--format`.
    pub id: &'static str,
    /// Bits stored per quantized value.
    pub bits: u8,
    /// How codes map to values.
    pub scheme: &'static str,
    /// The group size used when `--group-size` is not given; `None` when the
    /// format has its own block structure or uses one scale per tensor.
    pub default_group_size: Option<usize>,
    /// The status ModelQ claims.
    pub status: Status,
    /// Whether `--experimental` is required to write the format.
    pub requires_experimental_flag: bool,
    /// The container the output is written in.
    pub container: &'static str,
    /// The specification the format follows, for users who want to read it.
    pub specification: &'static str,
}

/// Every format the CLI can write, in the order `modelq formats` lists them.
pub const FORMATS: &[FormatSpec] = &[
    FormatSpec {
        id: "int8",
        bits: 8,
        scheme: "symmetric per-tensor",
        default_group_size: None,
        status: Status::RepresentationValid,
        requires_experimental_flag: false,
        container: "ModelQ-native SafeTensors",
        specification: "ADR 0002",
    },
    FormatSpec {
        id: "int4",
        bits: 4,
        scheme: "symmetric group-wise",
        default_group_size: Some(128),
        status: Status::RepresentationValid,
        requires_experimental_flag: false,
        container: "ModelQ-native SafeTensors",
        specification: "ADR 0030",
    },
    FormatSpec {
        id: "int3",
        bits: 3,
        scheme: "symmetric group-wise",
        default_group_size: Some(128),
        status: Status::Experimental,
        requires_experimental_flag: true,
        container: "ModelQ-native SafeTensors",
        specification: "ADR 0030",
    },
    FormatSpec {
        id: "int2",
        bits: 2,
        scheme: "symmetric group-wise",
        default_group_size: Some(128),
        status: Status::Experimental,
        requires_experimental_flag: true,
        container: "ModelQ-native SafeTensors",
        specification: "ADR 0030",
    },
    FormatSpec {
        id: "int1",
        bits: 1,
        scheme: "sign with mean-abs scale",
        default_group_size: Some(128),
        status: Status::Experimental,
        requires_experimental_flag: true,
        container: "ModelQ-native SafeTensors",
        specification: "ADR 0030",
    },
    FormatSpec {
        id: "gguf-q8_0",
        bits: 8,
        scheme: "GGUF Q8_0: 32-value blocks, F16 scale",
        default_group_size: Some(32),
        status: Status::RuntimeCompatible {
            runtime: "llama.cpp v0.6.0 (Qwen2, CPU)",
        },
        requires_experimental_flag: false,
        container: "GGUF v3 (llama.cpp)",
        specification: "ADR 0008, ADR 0032",
    },
    FormatSpec {
        id: "nvfp4",
        bits: 4,
        scheme: "E2M1 values, E4M3 block scales, F32 tensor scale",
        default_group_size: Some(16),
        status: Status::RepresentationValid,
        requires_experimental_flag: false,
        container: "ModelQ-native SafeTensors",
        specification: "ADR 0011",
    },
    FormatSpec {
        id: "nvfp4-te",
        bits: 4,
        scheme: "E2M1 values, E4M3 block scales, F32 tensor scale",
        default_group_size: Some(16),
        status: Status::HardwareValidated {
            runtime: "Transformer Engine 2.19.0",
            hardware: "NVIDIA B200",
        },
        requires_experimental_flag: false,
        container: "Transformer Engine rowwise container",
        specification: "ADR 0012, ADR 0022",
    },
];

/// The registry entry for a `--format` name.
pub fn find(id: &str) -> Option<&'static FormatSpec> {
    FORMATS.iter().find(|spec| spec.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_are_unique() {
        let mut ids: Vec<&str> = FORMATS.iter().map(|spec| spec.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), FORMATS.len());
    }

    #[test]
    fn experimental_formats_need_the_flag_and_nothing_else_does() {
        for spec in FORMATS {
            let experimental = spec.status == Status::Experimental;
            assert_eq!(
                spec.requires_experimental_flag, experimental,
                "{}: the flag must follow the experimental status",
                spec.id
            );
        }
        assert!(find("int1").unwrap().requires_experimental_flag);
        assert!(!find("int4").unwrap().requires_experimental_flag);
    }

    #[test]
    fn statuses_read_as_claims() {
        assert_eq!(Status::RepresentationValid.label(), "representation-valid");
        assert_eq!(
            find("nvfp4-te").unwrap().status.label(),
            "hardware-validated: Transformer Engine 2.19.0 on NVIDIA B200"
        );
    }

    #[test]
    fn bit_widths_match_the_codecs() {
        for spec in FORMATS {
            if spec.id.starts_with("int") {
                let bits: u8 = spec.id.trim_start_matches("int").parse().unwrap();
                assert_eq!(spec.bits, bits, "{}", spec.id);
            }
        }
    }

    #[test]
    fn unknown_names_are_not_found() {
        assert!(find("int16").is_none());
        assert!(find("").is_none());
    }
}
