//! Error types shared across ModelQ components.

use std::fmt;

/// Errors returned by ModelQ's library APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelQError {
    /// Multiplying the dimensions of a tensor shape overflowed [`usize`].
    ShapeElementCountOverflow {
        /// The shape whose element count could not be represented.
        shape: Vec<usize>,
    },
    /// Multiplying a tensor's element count by its element width overflowed
    /// [`usize`].
    TensorByteLengthOverflow {
        /// Number of elements in the tensor.
        element_count: usize,
        /// Storage width of one element in bytes.
        bytes_per_element: usize,
    },
    /// The available bytes do not match the tensor's dtype and shape.
    TensorByteLengthMismatch {
        /// Name of the invalid tensor.
        tensor_name: String,
        /// Byte length required by the dtype and shape.
        expected: usize,
        /// Byte length reported by metadata or supplied by the caller.
        actual: usize,
    },
}

impl fmt::Display for ModelQError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeElementCountOverflow { shape } => {
                write!(
                    formatter,
                    "tensor shape {shape:?} overflows the element count"
                )
            }
            Self::TensorByteLengthOverflow {
                element_count,
                bytes_per_element,
            } => write!(
                formatter,
                "tensor with {element_count} elements of {bytes_per_element} bytes overflows its byte length"
            ),
            Self::TensorByteLengthMismatch {
                tensor_name,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor {tensor_name:?} requires {expected} bytes but has {actual} bytes"
            ),
        }
    }
}

impl std::error::Error for ModelQError {}

/// Result type returned by ModelQ's library APIs.
pub type Result<T> = std::result::Result<T, ModelQError>;
