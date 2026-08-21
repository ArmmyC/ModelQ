//! Deterministic output tensor layout planning.
//!
//! The planner allocates contiguous data-region ranges for preserved tensors
//! and ModelQ-native INT8 qdata/scale pairs. It does not write bytes or build a
//! SafeTensors header; a later writer can consume this checked plan.

use std::{collections::BTreeSet, fmt, ops::Range};

use crate::{
    io::safetensors::TensorSummary,
    quant::policy::{PolicyAction, TensorDecision, TensorKind},
};

const INT8_DTYPE: &str = "I8";
const SCALE_DTYPE: &str = "F32";
const SCALE_BYTE_LEN: u64 = 4;
const RESERVED_METADATA_NAME: &str = "__metadata__";

/// The role of one tensor in the planned output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTensorRole {
    /// An input tensor copied to the output without quantization.
    Preserved,
    /// The signed INT8 payload for one quantized input tensor.
    QuantizedData,
    /// The scalar F32 scale paired with one quantized payload.
    QuantizationScale,
}

/// One output tensor's metadata and contiguous data-region range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputTensorPlan {
    /// Output tensor name.
    pub name: String,
    /// Original source tensor represented by this output tensor.
    pub source_name: String,
    /// Output SafeTensors dtype name.
    pub dtype: String,
    /// Output tensor shape.
    pub shape: Vec<usize>,
    /// Number of payload bytes reserved for this tensor.
    pub byte_len: u64,
    /// Half-open range relative to the SafeTensors data section.
    pub data_offsets: Range<u64>,
    /// Why this output tensor exists.
    pub role: OutputTensorRole,
}

/// A complete output data-region plan produced before writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputLayoutPlan {
    /// Output tensors in deterministic name/source order.
    pub tensors: Vec<OutputTensorPlan>,
    /// Total bytes required by the output data section.
    pub total_data_bytes: u64,
}

impl OutputLayoutPlan {
    /// Finds an output tensor by its exact output name.
    pub fn tensor(&self, name: &str) -> Option<&OutputTensorPlan> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

/// Errors returned when source metadata and policy decisions cannot form a
/// safe output layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// The planner requires one decision for every source tensor.
    DecisionCountMismatch { sources: usize, decisions: usize },
    /// A source tensor name appears more than once.
    DuplicateSourceName { name: String },
    /// A policy decision name appears more than once.
    DuplicateDecisionName { name: String },
    /// A source tensor has no matching policy decision.
    MissingDecision { name: String },
    /// A decision does not correspond to a source tensor.
    UnexpectedDecision { name: String },
    /// The source tensor name is reserved by SafeTensors.
    ReservedTensorName { name: String },
    /// A generated or preserved output name would be used more than once.
    OutputNameCollision { name: String },
    /// A policy decision reports an element count different from the shape.
    DecisionElementCountMismatch {
        name: String,
        metadata: u64,
        decision: usize,
    },
    /// A quantization action was requested for a non-floating candidate.
    QuantizationRequiresFloating { name: String, kind: TensorKind },
    /// The current INT8 path does not support this source dtype.
    UnsupportedQuantizedDtype { name: String, dtype: String },
    /// Multiplying shape dimensions overflowed `u64`.
    ShapeElementCountOverflow { name: String, shape: Vec<usize> },
    /// Advancing the output data cursor overflowed `u64`.
    OutputByteLengthOverflow {
        name: String,
        offset: u64,
        byte_len: u64,
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DecisionCountMismatch { sources, decisions } => write!(
                formatter,
                "layout needs one policy decision per source ({sources} sources, {decisions} decisions)"
            ),
            Self::DuplicateSourceName { name } => {
                write!(
                    formatter,
                    "source tensor name {name:?} appears more than once"
                )
            }
            Self::DuplicateDecisionName { name } => {
                write!(
                    formatter,
                    "policy decision name {name:?} appears more than once"
                )
            }
            Self::MissingDecision { name } => {
                write!(formatter, "source tensor {name:?} has no policy decision")
            }
            Self::UnexpectedDecision { name } => {
                write!(formatter, "policy decision {name:?} has no source tensor")
            }
            Self::ReservedTensorName { name } => write!(
                formatter,
                "source tensor name {name:?} is reserved by SafeTensors"
            ),
            Self::OutputNameCollision { name } => {
                write!(
                    formatter,
                    "output tensor name {name:?} would be used more than once"
                )
            }
            Self::DecisionElementCountMismatch {
                name,
                metadata,
                decision,
            } => write!(
                formatter,
                "policy decision for {name:?} reports {decision} elements but metadata has {metadata}"
            ),
            Self::QuantizationRequiresFloating { name, kind } => write!(
                formatter,
                "tensor {name:?} cannot be quantized because it is {kind}"
            ),
            Self::UnsupportedQuantizedDtype { name, dtype } => write!(
                formatter,
                "tensor {name:?} uses unsupported INT8 source dtype {dtype:?}"
            ),
            Self::ShapeElementCountOverflow { name, shape } => write!(
                formatter,
                "shape {shape:?} for tensor {name:?} overflows its element count"
            ),
            Self::OutputByteLengthOverflow {
                name,
                offset,
                byte_len,
            } => write!(
                formatter,
                "placing output tensor {name:?} at {offset} with {byte_len} bytes overflows the data region"
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Plans all output tensors and contiguous data offsets.
///
/// Sources and decisions may arrive in any order. The planner sorts both by
/// tensor name, then emits a preserved tensor or a qdata-then-scale pair. All
/// ranges are relative to the output SafeTensors data section, as required by
/// the format; the eventual header length is intentionally outside this plan.
pub fn plan_output_layout(
    sources: &[TensorSummary],
    decisions: &[TensorDecision],
) -> Result<OutputLayoutPlan, LayoutError> {
    if sources.len() != decisions.len() {
        return Err(LayoutError::DecisionCountMismatch {
            sources: sources.len(),
            decisions: decisions.len(),
        });
    }

    let mut sorted_sources = sources.iter().collect::<Vec<_>>();
    sorted_sources.sort_by(|left, right| left.name.cmp(&right.name));
    for pair in sorted_sources.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(LayoutError::DuplicateSourceName {
                name: pair[0].name.clone(),
            });
        }
    }

    let mut sorted_decisions = decisions.iter().collect::<Vec<_>>();
    sorted_decisions.sort_by(|left, right| left.name.cmp(&right.name));
    for pair in sorted_decisions.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(LayoutError::DuplicateDecisionName {
                name: pair[0].name.clone(),
            });
        }
    }

    let source_names = sorted_sources
        .iter()
        .map(|source| source.name.clone())
        .collect::<BTreeSet<_>>();
    if let Some(name) = source_names
        .iter()
        .find(|name| name.as_str() == RESERVED_METADATA_NAME)
    {
        return Err(LayoutError::ReservedTensorName { name: name.clone() });
    }

    let mut output_names = BTreeSet::new();
    let mut output_tensors = Vec::new();
    let mut cursor = 0_u64;

    for source in sorted_sources {
        let decision_index = sorted_decisions
            .binary_search_by(|decision| decision.name.as_str().cmp(source.name.as_str()))
            .map_err(|_| LayoutError::MissingDecision {
                name: source.name.clone(),
            })?;
        let decision = sorted_decisions[decision_index];
        let element_count = checked_element_count(source)?;
        if decision.element_count != usize::try_from(element_count).unwrap_or(usize::MAX) {
            return Err(LayoutError::DecisionElementCountMismatch {
                name: source.name.clone(),
                metadata: element_count,
                decision: decision.element_count,
            });
        }

        match decision.action {
            PolicyAction::Preserve => append_tensor(
                &mut output_tensors,
                &mut output_names,
                &mut cursor,
                PendingTensor {
                    source_name: source.name.clone(),
                    name: source.name.clone(),
                    dtype: source.dtype.clone(),
                    shape: source.shape.clone(),
                    byte_len: source.byte_len,
                    role: OutputTensorRole::Preserved,
                },
            )?,
            PolicyAction::Quantize => {
                if decision.kind != TensorKind::Floating {
                    return Err(LayoutError::QuantizationRequiresFloating {
                        name: source.name.clone(),
                        kind: decision.kind,
                    });
                }
                if !is_supported_int8_source_dtype(&source.dtype) {
                    return Err(LayoutError::UnsupportedQuantizedDtype {
                        name: source.name.clone(),
                        dtype: source.dtype.clone(),
                    });
                }

                let qdata_name = format!("{}.qdata", source.name);
                let scale_name = format!("{}.scale", source.name);
                ensure_generated_name(&qdata_name, &source_names, &output_names)?;
                ensure_generated_name(&scale_name, &source_names, &output_names)?;
                append_tensor(
                    &mut output_tensors,
                    &mut output_names,
                    &mut cursor,
                    PendingTensor {
                        source_name: source.name.clone(),
                        name: qdata_name,
                        dtype: INT8_DTYPE.to_owned(),
                        shape: source.shape.clone(),
                        byte_len: element_count,
                        role: OutputTensorRole::QuantizedData,
                    },
                )?;
                append_tensor(
                    &mut output_tensors,
                    &mut output_names,
                    &mut cursor,
                    PendingTensor {
                        source_name: source.name.clone(),
                        name: scale_name,
                        dtype: SCALE_DTYPE.to_owned(),
                        shape: Vec::new(),
                        byte_len: SCALE_BYTE_LEN,
                        role: OutputTensorRole::QuantizationScale,
                    },
                )?;
            }
        }
    }

    if let Some(decision) = sorted_decisions
        .iter()
        .find(|decision| !source_names.contains(&decision.name))
    {
        return Err(LayoutError::UnexpectedDecision {
            name: decision.name.clone(),
        });
    }

    Ok(OutputLayoutPlan {
        tensors: output_tensors,
        total_data_bytes: cursor,
    })
}

fn checked_element_count(source: &TensorSummary) -> Result<u64, LayoutError> {
    source
        .shape
        .iter()
        .try_fold(1_u64, |count, &dimension| {
            count.checked_mul(u64::try_from(dimension).ok()?)
        })
        .ok_or_else(|| LayoutError::ShapeElementCountOverflow {
            name: source.name.clone(),
            shape: source.shape.clone(),
        })
}

fn is_supported_int8_source_dtype(dtype: &str) -> bool {
    matches!(dtype, "F32" | "F16" | "BF16")
}

fn ensure_generated_name(
    name: &str,
    source_names: &BTreeSet<String>,
    output_names: &BTreeSet<String>,
) -> Result<(), LayoutError> {
    if source_names.contains(name) || output_names.contains(name) {
        return Err(LayoutError::OutputNameCollision {
            name: name.to_owned(),
        });
    }
    Ok(())
}

struct PendingTensor {
    source_name: String,
    name: String,
    dtype: String,
    shape: Vec<usize>,
    byte_len: u64,
    role: OutputTensorRole,
}

fn append_tensor(
    output_tensors: &mut Vec<OutputTensorPlan>,
    output_names: &mut BTreeSet<String>,
    cursor: &mut u64,
    pending: PendingTensor,
) -> Result<(), LayoutError> {
    if !output_names.insert(pending.name.clone()) {
        return Err(LayoutError::OutputNameCollision { name: pending.name });
    }
    let end = cursor.checked_add(pending.byte_len).ok_or_else(|| {
        LayoutError::OutputByteLengthOverflow {
            name: pending.name.clone(),
            offset: *cursor,
            byte_len: pending.byte_len,
        }
    })?;
    output_tensors.push(OutputTensorPlan {
        name: pending.name,
        source_name: pending.source_name,
        dtype: pending.dtype,
        shape: pending.shape,
        byte_len: pending.byte_len,
        data_offsets: *cursor..end,
        role: pending.role,
    });
    *cursor = end;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{LayoutError, OutputTensorRole, plan_output_layout};
    use crate::{
        io::safetensors::TensorSummary,
        quant::policy::{PolicyAction, QuantizationPolicy, TensorCandidate, TensorKind},
    };

    fn summary(name: &str, dtype: &str, shape: &[usize], byte_len: u64) -> TensorSummary {
        TensorSummary {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape: shape.to_vec(),
            byte_len,
        }
    }

    #[test]
    fn plans_sorted_preserved_and_quantized_outputs_contiguously() {
        let sources = [
            summary("weight", "F32", &[4], 16),
            summary("ids", "U8", &[3], 3),
            summary("bias", "F32", &[2], 8),
        ];
        let policy = QuantizationPolicy::new(4);
        let decisions = policy.decide_all([
            TensorCandidate::floating("weight", 4),
            TensorCandidate::non_floating("ids", 3),
            TensorCandidate::floating("bias", 2),
        ]);

        let plan = plan_output_layout(&sources, &decisions).expect("the layout is valid");

        assert_eq!(plan.total_data_bytes, 19);
        assert_eq!(
            plan.tensors
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            ["bias", "ids", "weight.qdata", "weight.scale"]
        );
        assert_eq!(plan.tensors[0].data_offsets, 0..8);
        assert_eq!(plan.tensors[1].data_offsets, 8..11);
        assert_eq!(plan.tensors[2].data_offsets, 11..15);
        assert_eq!(plan.tensors[3].data_offsets, 15..19);
        assert_eq!(plan.tensors[0].role, OutputTensorRole::Preserved);
        assert_eq!(plan.tensors[2].role, OutputTensorRole::QuantizedData);
        assert_eq!(plan.tensors[3].role, OutputTensorRole::QuantizationScale);
        assert_eq!(plan.tensors[2].dtype, "I8");
        assert_eq!(plan.tensors[3].dtype, "F32");
        assert!(plan.tensors[3].shape.is_empty());
    }

    #[test]
    fn input_order_does_not_change_the_plan() {
        let first_sources = [summary("z", "F32", &[4], 16), summary("a", "F32", &[4], 16)];
        let second_sources = [first_sources[1].clone(), first_sources[0].clone()];
        let first_decisions = QuantizationPolicy::new(1).decide_all([
            TensorCandidate::floating("z", 4),
            TensorCandidate::floating("a", 4),
        ]);
        let second_decisions = QuantizationPolicy::new(1).decide_all([
            TensorCandidate::floating("a", 4),
            TensorCandidate::floating("z", 4),
        ]);

        let first = plan_output_layout(&first_sources, &first_decisions).unwrap();
        let second = plan_output_layout(&second_sources, &second_decisions).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn rejects_name_collisions_before_writing() {
        let sources = [
            summary("weight", "F32", &[4], 16),
            summary("weight.qdata", "I8", &[4], 4),
        ];
        let decisions = [
            QuantizationPolicy::new(1).decide(&TensorCandidate::floating("weight", 4)),
            QuantizationPolicy::new(1).decide(&TensorCandidate::non_floating("weight.qdata", 4)),
        ];

        assert_eq!(
            plan_output_layout(&sources, &decisions).expect_err("the generated name collides"),
            LayoutError::OutputNameCollision {
                name: "weight.qdata".to_owned()
            }
        );
    }

    #[test]
    fn rejects_shape_overflow_and_bad_decision_counts() {
        let source = summary("huge", "F32", &[usize::MAX, 2], 0);
        let decision =
            QuantizationPolicy::new(1).decide(&TensorCandidate::floating("huge", usize::MAX));
        assert!(matches!(
            plan_output_layout(&[source], std::slice::from_ref(&decision)),
            Err(LayoutError::ShapeElementCountOverflow { .. })
        ));

        assert_eq!(
            plan_output_layout(&[], &[decision]).expect_err("a decision needs a source"),
            LayoutError::DecisionCountMismatch {
                sources: 0,
                decisions: 1
            }
        );
    }

    #[test]
    fn rejects_quantization_of_unsupported_source_dtype() {
        let source = summary("double", "F64", &[4], 32);
        let decision = crate::quant::policy::TensorDecision {
            name: "double".to_owned(),
            element_count: 4,
            kind: TensorKind::Floating,
            action: PolicyAction::Quantize,
            reason: crate::quant::policy::DecisionReason::FloatingMeetsMinimum {
                element_count: 4,
                minimum_elements: 1,
            },
        };

        assert_eq!(
            plan_output_layout(&[source], &[decision]).expect_err("F64 is not in the v0 path"),
            LayoutError::UnsupportedQuantizedDtype {
                name: "double".to_owned(),
                dtype: "F64".to_owned()
            }
        );
    }

    #[test]
    fn rejects_output_cursor_overflow() {
        let sources = [
            summary("first", "U8", &[], u64::MAX),
            summary("second", "U8", &[], 1),
        ];
        let decisions = [
            QuantizationPolicy::new(1).decide(&TensorCandidate::non_floating("first", 1)),
            QuantizationPolicy::new(1).decide(&TensorCandidate::non_floating("second", 1)),
        ];

        assert!(matches!(
            plan_output_layout(&sources, &decisions),
            Err(LayoutError::OutputByteLengthOverflow { .. })
        ));
    }
}
