//! Tensor selection policy for the native NVFP4 path.
//!
//! The rules are shape- and name-based and every decision carries an explicit
//! reason, so a user can audit why a tensor was or was not quantized.  See the
//! NVFP4 CLI export design for the rationale.

use std::fmt;

use crate::{
    nvfp4::BLOCK_SIZE,
    policy::{DEFAULT_MINIMUM_ELEMENTS, PolicyAction},
};

/// Name substrings preserved by default.  Weight-only low-bit schemes
/// conventionally keep the vocabulary embedding and output head at higher
/// precision, so quantizing them must be an explicit choice.
pub const DEFAULT_EXCLUDED_NAME_PARTS: &[&str] = &["embed_tokens", "lm_head", "embeddings"];

/// The final dimension a Transformer Engine matrix must be a multiple of.
/// Measured on a B200 with TE 2.19.0 (ADR 0022): the GEMM succeeded for every
/// tested multiple of 32 and failed for every other multiple of 16.
pub const RUNTIME_COLUMN_ALIGNMENT: usize = 32;

/// Metadata consumed by [`Nvfp4Policy`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nvfp4Candidate {
    /// Source tensor name.
    pub name: String,
    /// Whether the dtype is one the NVFP4 path can read (F32, F16, BF16).
    pub is_floating: bool,
    /// Source tensor shape.
    pub shape: Vec<usize>,
}

/// Explicit explanation for an NVFP4 policy decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Nvfp4Reason {
    /// The dtype is not a supported floating-point source dtype.
    NonFloating,
    /// A dimension is zero or the shape is empty.
    EmptyTensor,
    /// The shape's element count overflows `usize`.
    ElementCountOverflow,
    /// Vectors and scalars (norms, biases) are preserved.
    RankBelowTwo { rank: usize },
    /// The final dimension is not a multiple of the 16-value block size.
    FinalDimensionNotBlockAligned { final_dimension: usize },
    /// Transformer Engine mode only: the tensor is not exactly rank two.
    /// Flattening the leading dimensions of a stacked weight would share one
    /// amax across its slices.
    RankNotTwo { rank: usize },
    /// Transformer Engine mode only: the final dimension is a multiple of 16
    /// but not of 32.  Hardware runs on a Blackwell GPU showed the NVFP4 GEMM
    /// is rejected for such shapes, consistent with the packed row of
    /// `final_dimension / 2` bytes needing 16-byte alignment.
    FinalDimensionNotRuntimeAligned { final_dimension: usize },
    /// Transformer Engine mode only: the product of the leading dimensions is
    /// not a multiple of 16, which Transformer Engine's NVFP4 quantizer
    /// requires in addition to a block-aligned final dimension.
    LeadingDimensionNotBlockAligned { rows: usize },
    /// The tensor is smaller than the configured minimum.
    BelowMinimum {
        element_count: usize,
        minimum_elements: usize,
    },
    /// The name contains an excluded substring.
    ExcludedByName { pattern: String },
    /// The tensor passed every rule.
    Eligible { element_count: usize },
}

impl fmt::Display for Nvfp4Reason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFloating => formatter.write_str("non-floating tensors are preserved"),
            Self::EmptyTensor => formatter.write_str("empty tensors are preserved"),
            Self::ElementCountOverflow => {
                formatter.write_str("element count overflows; tensor is preserved")
            }
            Self::RankBelowTwo { rank } => {
                write!(
                    formatter,
                    "rank {rank} tensors (vectors/scalars) are preserved"
                )
            }
            Self::FinalDimensionNotBlockAligned { final_dimension } => write!(
                formatter,
                "final dimension {final_dimension} is not divisible by {BLOCK_SIZE}"
            ),
            Self::RankNotTwo { rank } => write!(
                formatter,
                "rank {rank} tensors are preserved by the Transformer Engine profile (only rank 2 is exported)"
            ),
            Self::FinalDimensionNotRuntimeAligned { final_dimension } => write!(
                formatter,
                "final dimension {final_dimension} is not divisible by 32; the Transformer Engine NVFP4 GEMM was rejected for such shapes on a B200"
            ),
            Self::LeadingDimensionNotBlockAligned { rows } => write!(
                formatter,
                "leading dimensions multiply to {rows}, which is not divisible by {BLOCK_SIZE} (required by Transformer Engine)"
            ),
            Self::BelowMinimum {
                element_count,
                minimum_elements,
            } => write!(
                formatter,
                "tensor has {element_count} elements, below minimum {minimum_elements}"
            ),
            Self::ExcludedByName { pattern } => {
                write!(formatter, "name matches excluded pattern {pattern:?}")
            }
            Self::Eligible { element_count } => {
                write!(formatter, "eligible with {element_count} elements")
            }
        }
    }
}

/// A complete, auditable NVFP4 decision for one tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nvfp4Decision {
    /// Tensor name carried through from the candidate.
    pub name: String,
    /// Selected action.
    pub action: PolicyAction,
    /// Explicit reason for the selected action.
    pub reason: Nvfp4Reason,
}

impl Nvfp4Decision {
    /// Returns whether this tensor is selected for NVFP4 quantization.
    pub const fn is_quantized(&self) -> bool {
        matches!(self.action, PolicyAction::Quantize)
    }
}

/// Conservative selection policy for native NVFP4 weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nvfp4Policy {
    minimum_elements: usize,
    excluded_name_parts: Vec<String>,
    transformer_engine: bool,
}

impl Nvfp4Policy {
    /// Creates a policy with the default minimum and default name exclusions.
    pub fn new() -> Self {
        Self {
            minimum_elements: DEFAULT_MINIMUM_ELEMENTS,
            excluded_name_parts: DEFAULT_EXCLUDED_NAME_PARTS
                .iter()
                .map(|part| (*part).to_owned())
                .collect(),
            transformer_engine: false,
        }
    }

    /// Creates the policy for the Transformer Engine rowwise profile: the
    /// default rules plus exactly rank two and a leading-dimension product
    /// divisible by 16.
    pub fn transformer_engine() -> Self {
        Self {
            transformer_engine: true,
            ..Self::new()
        }
    }

    /// Returns whether the Transformer Engine rules are active.
    pub const fn is_transformer_engine(&self) -> bool {
        self.transformer_engine
    }

    /// Removes the built-in name exclusions, keeping any added later.
    #[must_use]
    pub fn without_default_exclusions(mut self) -> Self {
        self.excluded_name_parts.clear();
        self
    }

    /// Adds one excluded name substring.  Empty patterns are ignored because
    /// they would match every tensor.
    #[must_use]
    pub fn with_exclusion(mut self, pattern: impl Into<String>) -> Self {
        let pattern = pattern.into();
        if !pattern.is_empty() && !self.excluded_name_parts.contains(&pattern) {
            self.excluded_name_parts.push(pattern);
        }
        self
    }

    /// Returns the active excluded name substrings in evaluation order.
    pub fn excluded_name_parts(&self) -> &[String] {
        &self.excluded_name_parts
    }

    /// Returns the minimum element count.
    pub const fn minimum_elements(&self) -> usize {
        self.minimum_elements
    }

    /// Decides whether one tensor is quantized or preserved.
    pub fn decide(&self, candidate: &Nvfp4Candidate) -> Nvfp4Decision {
        let (action, reason) = match self.evaluate(candidate) {
            Ok(element_count) => (
                PolicyAction::Quantize,
                Nvfp4Reason::Eligible { element_count },
            ),
            Err(reason) => (PolicyAction::Preserve, reason),
        };
        Nvfp4Decision {
            name: candidate.name.clone(),
            action,
            reason,
        }
    }

    /// Decides every candidate in input order.
    pub fn decide_all<'a, I>(&self, candidates: I) -> Vec<Nvfp4Decision>
    where
        I: IntoIterator<Item = &'a Nvfp4Candidate>,
    {
        candidates
            .into_iter()
            .map(|candidate| self.decide(candidate))
            .collect()
    }

    fn evaluate(&self, candidate: &Nvfp4Candidate) -> Result<usize, Nvfp4Reason> {
        if !candidate.is_floating {
            return Err(Nvfp4Reason::NonFloating);
        }
        let shape = &candidate.shape;
        if shape.is_empty() {
            return Err(Nvfp4Reason::RankBelowTwo { rank: 0 });
        }
        if shape.contains(&0) {
            return Err(Nvfp4Reason::EmptyTensor);
        }
        if shape.len() < 2 {
            return Err(Nvfp4Reason::RankBelowTwo { rank: shape.len() });
        }
        let final_dimension = shape[shape.len() - 1];
        if final_dimension % BLOCK_SIZE != 0 {
            return Err(Nvfp4Reason::FinalDimensionNotBlockAligned { final_dimension });
        }
        if self.transformer_engine {
            if shape.len() != 2 {
                return Err(Nvfp4Reason::RankNotTwo { rank: shape.len() });
            }
            let rows = shape[0];
            if rows % BLOCK_SIZE != 0 {
                return Err(Nvfp4Reason::LeadingDimensionNotBlockAligned { rows });
            }
            if final_dimension % RUNTIME_COLUMN_ALIGNMENT != 0 {
                return Err(Nvfp4Reason::FinalDimensionNotRuntimeAligned { final_dimension });
            }
        }
        let element_count = shape
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or(Nvfp4Reason::ElementCountOverflow)?;
        if element_count < self.minimum_elements {
            return Err(Nvfp4Reason::BelowMinimum {
                element_count,
                minimum_elements: self.minimum_elements,
            });
        }
        if let Some(pattern) = self
            .excluded_name_parts
            .iter()
            .find(|pattern| candidate.name.contains(pattern.as_str()))
        {
            return Err(Nvfp4Reason::ExcludedByName {
                pattern: pattern.clone(),
            });
        }
        Ok(element_count)
    }
}

impl Default for Nvfp4Policy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{Nvfp4Candidate, Nvfp4Policy, Nvfp4Reason};
    use crate::policy::PolicyAction;

    fn candidate(name: &str, is_floating: bool, shape: &[usize]) -> Nvfp4Candidate {
        Nvfp4Candidate {
            name: name.to_owned(),
            is_floating,
            shape: shape.to_vec(),
        }
    }

    fn reason(policy: &Nvfp4Policy, candidate: &Nvfp4Candidate) -> Nvfp4Reason {
        policy.decide(candidate).reason
    }

    #[test]
    fn selects_large_block_aligned_matrices() {
        let policy = Nvfp4Policy::default();
        let decision = policy.decide(&candidate("layers.0.mlp.weight", true, &[64, 64]));
        assert_eq!(decision.action, PolicyAction::Quantize);
        assert_eq!(
            decision.reason,
            Nvfp4Reason::Eligible {
                element_count: 4096
            }
        );
        assert!(decision.is_quantized());

        // Higher ranks are eligible and use the final dimension.
        assert!(
            policy
                .decide(&candidate("conv.weight", true, &[4, 8, 2, 32]))
                .is_quantized()
        );
    }

    #[test]
    fn preserves_each_ineligible_class_with_its_reason() {
        let policy = Nvfp4Policy::default();
        assert_eq!(
            reason(&policy, &candidate("ids", false, &[64, 64])),
            Nvfp4Reason::NonFloating
        );
        assert_eq!(
            reason(&policy, &candidate("norm.weight", true, &[4096])),
            Nvfp4Reason::RankBelowTwo { rank: 1 }
        );
        assert_eq!(
            reason(&policy, &candidate("scalar", true, &[])),
            Nvfp4Reason::RankBelowTwo { rank: 0 }
        );
        assert_eq!(
            reason(&policy, &candidate("empty", true, &[0, 64])),
            Nvfp4Reason::EmptyTensor
        );
        assert_eq!(
            reason(&policy, &candidate("odd", true, &[64, 40])),
            Nvfp4Reason::FinalDimensionNotBlockAligned {
                final_dimension: 40
            }
        );
        assert_eq!(
            reason(&policy, &candidate("small", true, &[4, 16])),
            Nvfp4Reason::BelowMinimum {
                element_count: 64,
                minimum_elements: 1024
            }
        );
        assert_eq!(
            reason(&policy, &candidate("huge", true, &[usize::MAX, 16])),
            Nvfp4Reason::ElementCountOverflow
        );
    }

    #[test]
    fn default_exclusions_preserve_embeddings_and_output_heads() {
        let policy = Nvfp4Policy::default();
        for name in [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "transformer.embeddings.word.weight",
        ] {
            assert!(matches!(
                reason(&policy, &candidate(name, true, &[64, 64])),
                Nvfp4Reason::ExcludedByName { .. }
            ));
        }
    }

    #[test]
    fn exclusions_are_configurable() {
        let candidate_head = candidate("lm_head.weight", true, &[64, 64]);
        let without_defaults = Nvfp4Policy::new().without_default_exclusions();
        assert!(without_defaults.decide(&candidate_head).is_quantized());
        assert!(without_defaults.excluded_name_parts().is_empty());

        let custom = without_defaults
            .with_exclusion("router")
            .with_exclusion("router");
        assert_eq!(custom.excluded_name_parts(), ["router"]);
        assert_eq!(
            reason(&custom, &candidate("moe.router.weight", true, &[64, 64])),
            Nvfp4Reason::ExcludedByName {
                pattern: "router".to_owned()
            }
        );

        // An empty pattern would match everything, so it is ignored.
        assert!(
            Nvfp4Policy::new()
                .with_exclusion("")
                .excluded_name_parts()
                .len()
                == 3
        );
    }

    #[test]
    fn structural_reasons_take_priority_over_name_exclusions() {
        let policy = Nvfp4Policy::default();
        assert_eq!(
            reason(&policy, &candidate("lm_head.bias", true, &[4096])),
            Nvfp4Reason::RankBelowTwo { rank: 1 }
        );
    }

    #[test]
    fn decide_all_preserves_input_order() {
        let policy = Nvfp4Policy::default();
        let candidates = [candidate("b", true, &[64, 64]), candidate("a", false, &[2])];
        let names: Vec<_> = policy
            .decide_all(&candidates)
            .into_iter()
            .map(|decision| decision.name)
            .collect();
        assert_eq!(names, ["b", "a"]);
    }

    #[test]
    fn transformer_engine_mode_requires_rank_two_and_aligned_rows() {
        let policy = Nvfp4Policy::transformer_engine();
        assert!(policy.is_transformer_engine());
        assert!(!Nvfp4Policy::new().is_transformer_engine());

        assert!(
            policy
                .decide(&candidate("w", true, &[64, 64]))
                .is_quantized()
        );
        assert!(
            policy
                .decide(&candidate("w", true, &[144, 96]))
                .is_quantized()
        );
        assert_eq!(
            reason(&policy, &candidate("k80", true, &[144, 80])),
            Nvfp4Reason::FinalDimensionNotRuntimeAligned {
                final_dimension: 80
            }
        );
        // The native policy only needs a multiple of 16.
        assert!(
            Nvfp4Policy::default()
                .decide(&candidate("k80", true, &[144, 80]))
                .is_quantized()
        );
        assert_eq!(
            reason(&policy, &candidate("experts", true, &[4, 64, 64])),
            Nvfp4Reason::RankNotTwo { rank: 3 }
        );
        assert_eq!(
            reason(&policy, &candidate("odd_rows", true, &[70, 64])),
            Nvfp4Reason::LeadingDimensionNotBlockAligned { rows: 70 }
        );
        // The native policy still accepts both shapes.
        let native = Nvfp4Policy::default();
        assert!(
            native
                .decide(&candidate("experts", true, &[4, 64, 64]))
                .is_quantized()
        );
        assert!(
            native
                .decide(&candidate("odd_rows", true, &[70, 64]))
                .is_quantized()
        );
    }

    #[test]
    fn transformer_engine_mode_keeps_the_default_exclusions_and_priorities() {
        let policy = Nvfp4Policy::transformer_engine();
        assert!(matches!(
            reason(&policy, &candidate("lm_head.weight", true, &[64, 64])),
            Nvfp4Reason::ExcludedByName { .. }
        ));
        // Block alignment of the final dimension is reported before the new rules.
        assert_eq!(
            reason(&policy, &candidate("w", true, &[4, 64, 40])),
            Nvfp4Reason::FinalDimensionNotBlockAligned {
                final_dimension: 40
            }
        );
        assert_eq!(
            reason(&policy, &candidate("vector", true, &[4096])),
            Nvfp4Reason::RankBelowTwo { rank: 1 }
        );
        let without = Nvfp4Policy::transformer_engine().without_default_exclusions();
        assert!(without.is_transformer_engine());
        assert!(
            without
                .decide(&candidate("lm_head.weight", true, &[64, 64]))
                .is_quantized()
        );
    }
}
