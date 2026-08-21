//! Source tensor metadata and non-owning tensor views.

use half::{bf16, f16};

use crate::error::{ModelQError, Result};

/// Floating-point data types accepted by the initial ModelQ reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// IEEE 754 single-precision floating point.
    F32,
    /// IEEE 754 binary16 floating point.
    F16,
    /// Brain floating-point format with an eight-bit exponent.
    BF16,
}

impl DType {
    /// Returns the storage width of one element in bytes.
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// Validated metadata describing a source tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    element_count: usize,
    byte_len: usize,
}

impl TensorInfo {
    /// Creates tensor metadata and validates the reported byte length.
    pub fn new(
        name: impl Into<String>,
        dtype: DType,
        shape: Vec<usize>,
        byte_len: usize,
    ) -> Result<Self> {
        let name = name.into();
        let element_count = validate_layout(&name, dtype, &shape, byte_len)?;

        Ok(Self {
            name,
            dtype,
            shape,
            element_count,
            byte_len,
        })
    }

    /// Returns the tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the source element data type.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the tensor dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the checked product of the tensor dimensions.
    pub const fn element_count(&self) -> usize {
        self.element_count
    }

    /// Returns the validated tensor payload size in bytes.
    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Returns whether the tensor payload is empty.
    pub const fn is_empty(&self) -> bool {
        self.byte_len == 0
    }
}

/// A validated, non-owning view over source tensor bytes and metadata.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    name: &'a str,
    dtype: DType,
    shape: &'a [usize],
    data: &'a [u8],
    element_count: usize,
}

impl<'a> TensorView<'a> {
    /// Creates a view after validating its shape and payload length.
    pub fn new(name: &'a str, dtype: DType, shape: &'a [usize], data: &'a [u8]) -> Result<Self> {
        let element_count = validate_layout(name, dtype, shape, data.len())?;

        Ok(Self {
            name,
            dtype,
            shape,
            data,
            element_count,
        })
    }

    /// Returns the tensor name.
    pub const fn name(&self) -> &str {
        self.name
    }

    /// Returns the source element data type.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the tensor dimensions.
    pub const fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Returns the number of elements in the tensor.
    pub const fn element_count(&self) -> usize {
        self.element_count
    }

    /// Returns the validated source payload.
    pub const fn data(&self) -> &[u8] {
        self.data
    }

    /// Iterates over the little-endian source values converted to reference
    /// [`f32`] values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        let dtype = self.dtype;
        self.data
            .chunks_exact(dtype.byte_width())
            .map(move |bytes| decode_value(dtype, bytes))
    }
}

fn checked_layout(dtype: DType, shape: &[usize]) -> Result<(usize, usize)> {
    let element_count = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or_else(|| ModelQError::ShapeElementCountOverflow {
                shape: shape.to_vec(),
            })?
    };

    let byte_len = element_count.checked_mul(dtype.byte_width()).ok_or(
        ModelQError::TensorByteLengthOverflow {
            element_count,
            bytes_per_element: dtype.byte_width(),
        },
    )?;

    Ok((element_count, byte_len))
}

fn validate_layout(
    name: &str,
    dtype: DType,
    shape: &[usize],
    actual_byte_len: usize,
) -> Result<usize> {
    let (element_count, expected_byte_len) = checked_layout(dtype, shape)?;
    if actual_byte_len != expected_byte_len {
        return Err(ModelQError::TensorByteLengthMismatch {
            tensor_name: name.to_owned(),
            expected: expected_byte_len,
            actual: actual_byte_len,
        });
    }

    Ok(element_count)
}

fn decode_value(dtype: DType, bytes: &[u8]) -> f32 {
    debug_assert_eq!(bytes.len(), dtype.byte_width());

    match dtype {
        DType::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        DType::F16 => f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32(),
        DType::BF16 => bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32(),
    }
}

#[cfg(test)]
mod tests {
    use super::{DType, TensorInfo, TensorView};
    use crate::error::ModelQError;

    #[test]
    fn reports_dtype_byte_widths() {
        assert_eq!(DType::F32.byte_width(), 4);
        assert_eq!(DType::F16.byte_width(), 2);
        assert_eq!(DType::BF16.byte_width(), 2);
    }

    #[test]
    fn creates_valid_tensor_info_for_each_dtype() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let info = TensorInfo::new("weight", dtype, vec![2, 3], 6 * dtype.byte_width())
                .expect("the metadata has a matching byte length");

            assert_eq!(info.name(), "weight");
            assert_eq!(info.dtype(), dtype);
            assert_eq!(info.shape(), [2, 3]);
            assert_eq!(info.element_count(), 6);
            assert_eq!(info.byte_len(), 6 * dtype.byte_width());
            assert!(!info.is_empty());
        }
    }

    #[test]
    fn converts_little_endian_f32_values() {
        let data = [-2.0_f32, 0.5, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let shape = [3];
        let view = TensorView::new("weight", DType::F32, &shape, &data)
            .expect("the payload contains three f32 values");

        assert_eq!(view.values().collect::<Vec<_>>(), [-2.0, 0.5, 1.0]);
        assert_eq!(view.values().len(), view.element_count());
        assert_eq!(view.name(), "weight");
        assert_eq!(view.dtype(), DType::F32);
        assert_eq!(view.shape(), shape);
        assert_eq!(view.data(), data);
    }

    #[test]
    fn converts_little_endian_f16_values() {
        let data = [0xc000_u16, 0x3800, 0x3c00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let shape = [3];
        let view = TensorView::new("weight", DType::F16, &shape, &data)
            .expect("the payload contains three f16 values");

        assert_eq!(view.values().collect::<Vec<_>>(), [-2.0, 0.5, 1.0]);
    }

    #[test]
    fn converts_little_endian_bf16_values() {
        let data = [0xc000_u16, 0x3f00, 0x3f80]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let shape = [3];
        let view = TensorView::new("weight", DType::BF16, &shape, &data)
            .expect("the payload contains three bf16 values");

        assert_eq!(view.values().collect::<Vec<_>>(), [-2.0, 0.5, 1.0]);
    }

    #[test]
    fn rejects_mismatched_byte_lengths() {
        for actual in [3, 5] {
            let error = TensorInfo::new("weight", DType::F32, vec![1], actual)
                .expect_err("one f32 requires exactly four bytes");

            assert_eq!(
                error,
                ModelQError::TensorByteLengthMismatch {
                    tensor_name: "weight".to_owned(),
                    expected: 4,
                    actual,
                }
            );
        }
    }

    #[test]
    fn rejects_mismatched_tensor_view_data() {
        let shape = [2];
        let error = TensorView::new("weight", DType::F16, &shape, &[0, 0])
            .expect_err("two f16 values require four bytes");

        assert_eq!(
            error,
            ModelQError::TensorByteLengthMismatch {
                tensor_name: "weight".to_owned(),
                expected: 4,
                actual: 2,
            }
        );
    }

    #[test]
    fn rejects_shape_element_count_overflow() {
        let shape = [usize::MAX, 2];
        let error = TensorView::new("weight", DType::F16, &shape, &[])
            .expect_err("the shape product exceeds usize");

        assert_eq!(
            error,
            ModelQError::ShapeElementCountOverflow {
                shape: shape.to_vec(),
            }
        );
    }

    #[test]
    fn rejects_tensor_byte_length_overflow() {
        let error = TensorInfo::new("weight", DType::F32, vec![usize::MAX], 0)
            .expect_err("the f32 payload size exceeds usize");

        assert_eq!(
            error,
            ModelQError::TensorByteLengthOverflow {
                element_count: usize::MAX,
                bytes_per_element: 4,
            }
        );
    }

    #[test]
    fn accepts_scalar_and_empty_tensors() {
        let scalar_bytes = 1.0_f32.to_le_bytes();
        let scalar = TensorView::new("scalar", DType::F32, &[], &scalar_bytes)
            .expect("an empty shape describes one scalar");
        assert_eq!(scalar.element_count(), 1);
        assert_eq!(scalar.values().collect::<Vec<_>>(), [1.0]);

        let empty = TensorInfo::new("empty", DType::BF16, vec![usize::MAX, 2, 0], 0)
            .expect("a zero dimension makes the tensor empty");
        assert_eq!(empty.element_count(), 0);
        assert!(empty.is_empty());
    }
}
