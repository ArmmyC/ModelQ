//! Conservative v0 tensor quantization policy.

use std::fmt;

use crate::tensor::TensorInfo;

/// Default minimum number of elements for a floating tensor to be considered
/// for quantization.
pub const DEFAULT_MINIMUM_ELEMENTS: usize = 1024;

/// Broad tensor category used by the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    /// A tensor whose values are eligible for the current floating-point
    /// quantizer.
    Floating,
    /// A tensor that must be preserved by the current policy.
    NonFloating,
}

impl fmt::Display for TensorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Floating => formatter.write_str("floating"),
            Self::NonFloating => formatter.write_str("non-floating"),
        }
    }
}

/// Minimal metadata consumed by [`QuantizationPolicy`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCandidate {
    name: String,
    element_count: usize,
    kind: TensorKind,
}

impl TensorCandidate {
    /// Creates a candidate with an explicit tensor category.
    pub fn new(name: impl Into<String>, element_count: usize, kind: TensorKind) -> Self {
        Self {
            name: name.into(),
            element_count,
            kind,
        }
    }

    /// Creates a floating-point candidate.
    pub fn floating(name: impl Into<String>, element_count: usize) -> Self {
        Self::new(name, element_count, TensorKind::Floating)
    }

    /// Creates a non-floating candidate.
    pub fn non_floating(name: impl Into<String>, element_count: usize) -> Self {
        Self::new(name, element_count, TensorKind::NonFloating)
    }

    /// Creates a floating candidate from the currently supported source
    /// tensor metadata.
    pub fn from_tensor_info(info: &TensorInfo) -> Self {
        Self::floating(info.name(), info.element_count())
    }

    /// Returns the tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the number of tensor elements.
    pub const fn element_count(&self) -> usize {
        self.element_count
    }

    /// Returns the broad tensor category.
    pub const fn kind(&self) -> TensorKind {
        self.kind
    }
}

/// Action recorded for one tensor by the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyAction {
    /// Select the current scalar INT8 quantization path.
    Quantize,
    /// Keep the original tensor representation.
    Preserve,
}

impl fmt::Display for PolicyAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quantize => formatter.write_str("quantize"),
            Self::Preserve => formatter.write_str("preserve"),
        }
    }
}

/// Explicit explanation for a policy decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    /// Non-floating tensors are outside the current quantizer's scope.
    NonFloating,
    /// Empty tensors are preserved even when the configured threshold is zero.
    EmptyTensor,
    /// The floating tensor is smaller than the configured minimum.
    FloatingBelowMinimum {
        /// Number of elements in the tensor.
        element_count: usize,
        /// Configured minimum number of elements.
        minimum_elements: usize,
    },
    /// The floating tensor meets the configured minimum.
    FloatingMeetsMinimum {
        /// Number of elements in the tensor.
        element_count: usize,
        /// Configured minimum number of elements.
        minimum_elements: usize,
    },
}

impl fmt::Display for DecisionReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFloating => formatter.write_str("non-floating tensors are preserved"),
            Self::EmptyTensor => formatter.write_str("empty tensors are preserved"),
            Self::FloatingBelowMinimum {
                element_count,
                minimum_elements,
            } => write!(
                formatter,
                "floating tensor has {element_count} elements, below minimum {minimum_elements}"
            ),
            Self::FloatingMeetsMinimum {
                element_count,
                minimum_elements,
            } => write!(
                formatter,
                "floating tensor has {element_count} elements, meeting minimum {minimum_elements}"
            ),
        }
    }
}

/// A complete, auditable action and reason for one tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDecision {
    /// Tensor name carried through from the candidate.
    pub name: String,
    /// Number of elements considered by the policy.
    pub element_count: usize,
    /// Broad tensor category used for the decision.
    pub kind: TensorKind,
    /// Selected action.
    pub action: PolicyAction,
    /// Explicit reason for the selected action.
    pub reason: DecisionReason,
}

impl TensorDecision {
    /// Returns whether this tensor is selected for quantization.
    pub const fn is_quantized(&self) -> bool {
        matches!(self.action, PolicyAction::Quantize)
    }
}

/// Conservative policy for deciding whether tensors enter the INT8 path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationPolicy {
    minimum_elements: usize,
}

impl QuantizationPolicy {
    /// Creates a policy with a configurable minimum element count.
    pub const fn new(minimum_elements: usize) -> Self {
        Self { minimum_elements }
    }

    /// Returns the configured minimum element count.
    pub const fn minimum_elements(self) -> usize {
        self.minimum_elements
    }

    /// Decides whether one tensor should be quantized or preserved.
    pub fn decide(&self, candidate: &TensorCandidate) -> TensorDecision {
        let (action, reason) = match (candidate.kind, candidate.element_count) {
            (TensorKind::NonFloating, _) => (PolicyAction::Preserve, DecisionReason::NonFloating),
            (TensorKind::Floating, 0) => (PolicyAction::Preserve, DecisionReason::EmptyTensor),
            (TensorKind::Floating, element_count) if element_count >= self.minimum_elements => (
                PolicyAction::Quantize,
                DecisionReason::FloatingMeetsMinimum {
                    element_count,
                    minimum_elements: self.minimum_elements,
                },
            ),
            (TensorKind::Floating, element_count) => (
                PolicyAction::Preserve,
                DecisionReason::FloatingBelowMinimum {
                    element_count,
                    minimum_elements: self.minimum_elements,
                },
            ),
        };

        TensorDecision {
            name: candidate.name.clone(),
            element_count: candidate.element_count,
            kind: candidate.kind,
            action,
            reason,
        }
    }

    /// Decides every candidate in input order.
    pub fn decide_all<I>(&self, candidates: I) -> Vec<TensorDecision>
    where
        I: IntoIterator<Item = TensorCandidate>,
    {
        candidates
            .into_iter()
            .map(|candidate| self.decide(&candidate))
            .collect()
    }
}

impl Default for QuantizationPolicy {
    fn default() -> Self {
        Self::new(DEFAULT_MINIMUM_ELEMENTS)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_MINIMUM_ELEMENTS, DecisionReason, PolicyAction, QuantizationPolicy,
        TensorCandidate, TensorKind,
    };
    use crate::tensor::{DType, TensorInfo};

    #[test]
    fn preserves_non_floating_tensors_with_a_reason() {
        let policy = QuantizationPolicy::new(4);
        let decision = policy.decide(&TensorCandidate::non_floating("labels", 10_000));

        assert_eq!(decision.name, "labels");
        assert_eq!(decision.element_count, 10_000);
        assert_eq!(decision.kind, TensorKind::NonFloating);
        assert_eq!(decision.action, PolicyAction::Preserve);
        assert_eq!(decision.reason, DecisionReason::NonFloating);
        assert!(!decision.is_quantized());
    }

    #[test]
    fn preserves_small_floating_tensors() {
        let policy = QuantizationPolicy::new(4);
        let decision = policy.decide(&TensorCandidate::floating("bias", 3));

        assert_eq!(decision.action, PolicyAction::Preserve);
        assert_eq!(
            decision.reason,
            DecisionReason::FloatingBelowMinimum {
                element_count: 3,
                minimum_elements: 4,
            }
        );
    }

    #[test]
    fn quantizes_floating_tensors_at_or_above_the_threshold() {
        let policy = QuantizationPolicy::new(4);

        let at_threshold = policy.decide(&TensorCandidate::floating("weight", 4));
        let above_threshold = policy.decide(&TensorCandidate::floating("large", 8));

        assert_eq!(at_threshold.action, PolicyAction::Quantize);
        assert_eq!(above_threshold.action, PolicyAction::Quantize);
        assert!(at_threshold.is_quantized());
        assert!(matches!(
            at_threshold.reason,
            DecisionReason::FloatingMeetsMinimum {
                element_count: 4,
                minimum_elements: 4
            }
        ));
    }

    #[test]
    fn preserves_empty_floating_tensors_even_with_zero_threshold() {
        let policy = QuantizationPolicy::new(0);
        let decision = policy.decide(&TensorCandidate::floating("empty", 0));

        assert_eq!(decision.action, PolicyAction::Preserve);
        assert_eq!(decision.reason, DecisionReason::EmptyTensor);
    }

    #[test]
    fn records_decisions_in_candidate_order() {
        let policy = QuantizationPolicy::new(2);
        let decisions = policy.decide_all([
            TensorCandidate::floating("first", 2),
            TensorCandidate::non_floating("second", 100),
            TensorCandidate::floating("third", 1),
        ]);

        assert_eq!(
            decisions
                .iter()
                .map(|decision| decision.name.as_str())
                .collect::<Vec<_>>(),
            ["first", "second", "third"]
        );
        assert_eq!(decisions[0].action, PolicyAction::Quantize);
        assert_eq!(decisions[1].action, PolicyAction::Preserve);
        assert_eq!(decisions[2].action, PolicyAction::Preserve);
    }

    #[test]
    fn creates_floating_candidate_from_tensor_info() {
        let info = TensorInfo::new("weight", DType::F32, vec![2, 2], 16)
            .expect("the tensor metadata is valid");

        let candidate = TensorCandidate::from_tensor_info(&info);

        assert_eq!(candidate.name(), "weight");
        assert_eq!(candidate.element_count(), 4);
        assert_eq!(candidate.kind(), TensorKind::Floating);
    }

    #[test]
    fn exposes_a_conservative_default_threshold() {
        assert_eq!(
            QuantizationPolicy::default().minimum_elements(),
            DEFAULT_MINIMUM_ELEMENTS
        );
    }
}
