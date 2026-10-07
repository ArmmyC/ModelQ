//! ModelQ-native scalar NVFP4 reference quantization.
//!
//! NVFP4 combines signed FP4 E2M1 values with one positive FP8 E4M3 scale for
//! each [`BLOCK_SIZE`] values and one F32 decode scale for the tensor.  This
//! module implements the deterministic, data-free, weight-only baseline from
//! ADR 0010.  It deliberately does not implement Transformer Engine swizzles,
//! transposed runtime buffers, activation quantization, or a container format.

use std::fmt;

use crate::float::{fp4_e2m1, fp8_e4m3};

/// Number of values sharing one FP8 E4M3 block scale.
pub const BLOCK_SIZE: usize = 16;
/// Number of FP4 values stored in one byte.
pub const VALUES_PER_BYTE: usize = 2;
/// Maximum finite magnitude of the E2M1 element format.
pub const FP4_MAX: f32 = fp4_e2m1::MAX_FINITE;
/// Maximum finite magnitude of the E4M3 block-scale format.
pub const FP8_MAX: f32 = fp8_e4m3::MAX_FINITE;
const SCALE_PRODUCT: f32 = FP4_MAX * FP8_MAX;
/// Largest finite E4M3 magnitude code (`0x7e`); `0x7f` is NaN.
const MAX_E4M3_BITS: u8 = 0x7e;

/// How the encoder chooses each block's E4M3 scale.
///
/// The choice only changes which scale byte is stored; every selection
/// produces the same representation (E2M1 values, E4M3 block scales, one
/// tensor-wide decode scale), so a decoder or runtime needs no change.  The
/// per-tensor global scale is unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleSelection {
    /// The reference rule: the E4M3 value nearest `block_amax / global_amax * 448`,
    /// so the block's largest value maps to about 6.
    #[default]
    Amax,
    /// Try every E4M3 scale within `radius` codes of the reference choice and
    /// keep the one with the smallest squared reconstruction error for the
    /// block.  Ties keep the reference scale, then the nearer, then the
    /// smaller code.  `radius = 0` is identical to [`Self::Amax`].
    MinMse { radius: u8 },
}

impl ScaleSelection {
    /// Whether this is the reference rule (and so changes no output byte).
    pub const fn is_default(self) -> bool {
        matches!(self, Self::Amax) || matches!(self, Self::MinMse { radius: 0 })
    }

    /// A short, stable label recorded in container metadata.
    pub fn label(self) -> String {
        match self {
            Self::Amax => "amax".to_owned(),
            Self::MinMse { radius } => format!("min-mse:r{radius}"),
        }
    }
}
const MIN_POSITIVE_E4M3_BITS: u8 = 0x01;
const MIN_POSITIVE_F32: f32 = f32::from_bits(1);

/// Errors returned by the ModelQ-native NVFP4 reference implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum Nvfp4Error {
    /// A source value is not finite.
    NonFiniteInput { index: usize, value: f32 },
    /// A shaped tensor is empty or its final dimension is not block-aligned.
    InvalidShape { shape: Vec<usize> },
    /// The product of shaped tensor dimensions overflowed `usize`.
    ShapeElementCountOverflow { shape: Vec<usize> },
    /// The source length does not match the checked shape product.
    ShapeLengthMismatch { expected: usize, actual: usize },
    /// A packed E2M1 nibble is outside the four-bit range.
    PackedValueOutOfRange { index: usize, value: u8 },
    /// The packed payload length does not match the element count.
    PackedLengthMismatch {
        elements: usize,
        expected: usize,
        actual: usize,
    },
    /// The block-scale count does not match the element count.
    BlockScaleCountMismatch { expected: usize, actual: usize },
    /// A block scale is not a positive finite E4M3 value or zero.
    InvalidBlockScale { block: usize, bits: u8 },
    /// A zero block scale was paired with a nonzero E2M1 value.
    ZeroScaleWithNonzeroValue { block: usize, index: usize },
    /// The tensor-wide decode scale is not finite and positive.
    InvalidGlobalScale { scale: f32 },
    /// A reconstructed value is not finite.
    DequantizedValueOverflow { index: usize },
    /// A streaming chunk size of zero blocks was requested.
    InvalidChunkSize { chunk_blocks: usize },
    /// A chunk does not start on a block boundary.
    MisalignedChunk { first_index: usize },
}

impl fmt::Display for Nvfp4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput { index, value } => {
                write!(
                    formatter,
                    "NVFP4 input at index {index} is not finite: {value:?}"
                )
            }
            Self::InvalidShape { shape } => write!(
                formatter,
                "NVFP4 shape {shape:?} must have positive dimensions and a final dimension divisible by {BLOCK_SIZE}"
            ),
            Self::ShapeElementCountOverflow { shape } => {
                write!(
                    formatter,
                    "NVFP4 shape {shape:?} overflows its element count"
                )
            }
            Self::ShapeLengthMismatch { expected, actual } => write!(
                formatter,
                "NVFP4 shape describes {expected} values but source contains {actual}"
            ),
            Self::PackedValueOutOfRange { index, value } => write!(
                formatter,
                "NVFP4 packed value at index {index} is outside [0, 15]: {value}"
            ),
            Self::PackedLengthMismatch {
                elements,
                expected,
                actual,
            } => write!(
                formatter,
                "NVFP4 data for {elements} elements requires {expected} packed bytes but has {actual}"
            ),
            Self::BlockScaleCountMismatch { expected, actual } => write!(
                formatter,
                "NVFP4 data requires {expected} block scales but has {actual}"
            ),
            Self::InvalidBlockScale { block, bits } => write!(
                formatter,
                "NVFP4 block {block} has invalid E4M3 scale bits: {bits:#04x}"
            ),
            Self::ZeroScaleWithNonzeroValue { block, index } => write!(
                formatter,
                "NVFP4 block {block} has a zero scale but a nonzero value at index {index}"
            ),
            Self::InvalidGlobalScale { scale } => write!(
                formatter,
                "NVFP4 global decode scale must be finite and positive: {scale:?}"
            ),
            Self::DequantizedValueOverflow { index } => write!(
                formatter,
                "NVFP4 dequantized value at index {index} is not finite"
            ),
            Self::InvalidChunkSize { chunk_blocks } => write!(
                formatter,
                "NVFP4 streaming chunk size must be positive, got {chunk_blocks} blocks"
            ),
            Self::MisalignedChunk { first_index } => write!(
                formatter,
                "NVFP4 chunk starting at index {first_index} is not aligned to {BLOCK_SIZE}-value blocks"
            ),
        }
    }
}

impl std::error::Error for Nvfp4Error {}

/// A packed ModelQ-native NVFP4 tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    packed: Vec<u8>,
    block_scales: Vec<u8>,
    global_scale: f32,
    elements: usize,
}

impl QuantizedTensor {
    /// Builds an NVFP4 tensor from its native fields after validation.
    pub fn from_parts(
        packed: Vec<u8>,
        block_scales: Vec<u8>,
        global_scale: f32,
        elements: usize,
    ) -> Result<Self, Nvfp4Error> {
        validate_parts(&packed, &block_scales, global_scale, elements)?;
        Ok(Self {
            packed,
            block_scales,
            global_scale,
            elements,
        })
    }

    /// Returns packed E2M1 bytes, with element zero in each low nibble.
    pub fn packed_values(&self) -> &[u8] {
        &self.packed
    }

    /// Alias for [`Self::packed_values`].
    pub fn packed(&self) -> &[u8] {
        self.packed_values()
    }

    /// Returns one E4M3 bit pattern for each 16-value block.
    pub fn block_scales(&self) -> &[u8] {
        &self.block_scales
    }

    /// Alias for [`Self::block_scales`].
    pub fn scales(&self) -> &[u8] {
        self.block_scales()
    }

    /// Returns the F32 decode scale applied to the whole tensor.
    pub const fn global_scale(&self) -> f32 {
        self.global_scale
    }

    /// Returns the number of source values represented by this tensor.
    pub const fn len(&self) -> usize {
        self.elements
    }

    /// Returns whether this tensor represents no values.
    pub const fn is_empty(&self) -> bool {
        self.elements == 0
    }

    /// Returns the number of packed payload bytes.
    pub fn packed_len(&self) -> usize {
        self.packed.len()
    }

    /// Unpacks the E2M1 nibbles after validating the payload length.
    pub fn unpacked_values(&self) -> Result<Vec<u8>, Nvfp4Error> {
        unpack(&self.packed, self.elements)
    }

    /// Reconstructs F32 values using the stored scales.
    pub fn dequantize(&self) -> Result<Vec<f32>, Nvfp4Error> {
        dequantize(
            &self.packed,
            &self.block_scales,
            self.global_scale,
            self.elements,
        )
    }

    /// Consumes the tensor and returns packed values, block scales, global
    /// scale, and element count in native representation order.
    pub fn into_parts(self) -> (Vec<u8>, Vec<u8>, f32, usize) {
        (
            self.packed,
            self.block_scales,
            self.global_scale,
            self.elements,
        )
    }
}

/// Quantizes finite F32 values with the native 16-value NVFP4 hierarchy.
///
/// The returned global scale is a decode scale.  Each source value is
/// reconstructed as `e2m1 * e4m3_block_scale * global_scale`.  The input is
/// treated as a flattened row-major stream; shape metadata belongs to a
/// caller or a future container layer.  Use [`quantize_shaped`] when the
/// source tensor's final dimension must be checked against the native block
/// layout.
pub fn quantize(values: &[f32]) -> Result<QuantizedTensor, Nvfp4Error> {
    quantize_with(values, ScaleSelection::default())
}

/// [`quantize`] with an explicit block-scale selection.
pub fn quantize_with(
    values: &[f32],
    selection: ScaleSelection,
) -> Result<QuantizedTensor, Nvfp4Error> {
    let global_amax = max_abs(values)?;
    let global_scale = global_scale_for_amax(global_amax);

    let mut packed = vec![0_u8; packed_len(values.len())];
    let mut block_scales = vec![0_u8; block_count(values.len())];
    encode_chunk_with(
        values,
        0,
        global_amax,
        selection,
        &mut packed,
        &mut block_scales,
    )?;
    Ok(QuantizedTensor {
        packed,
        block_scales,
        global_scale,
        elements: values.len(),
    })
}

/// Quantizes a shaped row-major tensor after checking the native block rule.
///
/// NVFP4 groups consecutive values along the final dimension, so this entry
/// point requires non-zero dimensions and a final dimension divisible by
/// [`BLOCK_SIZE`].  The returned representation remains flat; callers retain
/// the original `shape` for their container or tensor metadata.
pub fn quantize_shaped(values: &[f32], shape: &[usize]) -> Result<QuantizedTensor, Nvfp4Error> {
    quantize_shaped_with(values, shape, ScaleSelection::default())
}

/// [`quantize_shaped`] with an explicit block-scale selection.
pub fn quantize_shaped_with(
    values: &[f32],
    shape: &[usize],
    selection: ScaleSelection,
) -> Result<QuantizedTensor, Nvfp4Error> {
    let expected = checked_shape_elements(shape)?;
    if expected != values.len() {
        return Err(Nvfp4Error::ShapeLengthMismatch {
            expected,
            actual: values.len(),
        });
    }

    quantize_with(values, selection)
}

/// Validates a shape for NVFP4 and returns its checked element count.
///
/// Requires a non-empty shape of positive dimensions whose final dimension is
/// divisible by [`BLOCK_SIZE`].
pub fn checked_shape_elements(shape: &[usize]) -> Result<usize, Nvfp4Error> {
    if shape.is_empty()
        || shape.contains(&0)
        || !shape
            .last()
            .is_some_and(|&dimension| dimension % BLOCK_SIZE == 0)
    {
        return Err(Nvfp4Error::InvalidShape {
            shape: shape.to_vec(),
        });
    }
    shape
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| Nvfp4Error::ShapeElementCountOverflow {
            shape: shape.to_vec(),
        })
}

/// Returns the tensor-wide F32 decode scale for a finite tensor-wide amax.
///
/// An all-zero tensor (`amax == 0`) uses `1.0`.  Exposed so alternate
/// execution backends derive exactly the reference scale.
pub fn global_scale_for_amax(global_amax: f32) -> f32 {
    if global_amax == 0.0 {
        1.0
    } else {
        (global_amax / SCALE_PRODUCT).max(MIN_POSITIVE_F32)
    }
}

/// Encodes one block of at most [`BLOCK_SIZE`] values, writing one E2M1 code
/// per value into `codes` and returning the block's E4M3 scale bits.
/// `start_index` is the flattened index of `chunk[0]` and is used only for
/// error reporting.  The reference, sequential streaming, and parallel paths
/// share this encoder so their output cannot diverge.
#[inline]
fn encode_block_into(
    chunk: &[f32],
    start_index: usize,
    global_amax: f32,
    codes: &mut [u8; BLOCK_SIZE],
) -> Result<u8, Nvfp4Error> {
    let block_amax = chunk
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    if block_amax == 0.0 {
        codes.fill(0);
        return Ok(0);
    }

    let scale_input = (block_amax / global_amax) * FP8_MAX;
    let mut scale_bits = fp8_e4m3::encode(scale_input);
    if scale_bits == 0 {
        scale_bits = MIN_POSITIVE_E4M3_BITS;
    }
    let decoded_block_scale = fp8_e4m3::decode(scale_bits);

    // A branch-free loop the compiler can vectorize.  The two divisions are
    // kept as divisions: replacing them with a reciprocal multiply would
    // change rounding and therefore the produced bits.
    let mut has_nan = false;
    for (code, &value) in codes.iter_mut().zip(chunk) {
        let scaled = ((value / global_amax) * SCALE_PRODUCT) / decoded_block_scale;
        has_nan |= scaled.is_nan();
        *code = fp4_e2m1::encode_unchecked(scaled);
    }
    if has_nan {
        // Report the first offending value, as the element codec would.
        let (offset, &value) = chunk
            .iter()
            .enumerate()
            .find(|(_, value)| {
                (((**value / global_amax) * SCALE_PRODUCT) / decoded_block_scale).is_nan()
            })
            .expect("a NaN was flagged in this block");
        return Err(Nvfp4Error::NonFiniteInput {
            index: start_index + offset,
            value,
        });
    }
    Ok(scale_bits)
}

/// Encodes one block choosing its E4M3 scale by minimum squared error.
///
/// Starts from the reference encoding (so non-finite input is reported exactly
/// as before), then tries every nonzero E4M3 code within `radius` of the
/// reference code.  A candidate is scored by re-encoding the block's values with
/// that scale and summing the squared difference to the decoded value
/// (`e2m1 * scale * global_scale`, the decoder's own arithmetic).  Candidates
/// are visited by increasing distance from the reference, lower code first, and
/// replace the incumbent only on a strictly smaller error, which makes ties
/// deterministic: reference, then nearest, then lowest code.
#[inline]
fn encode_block_min_mse(
    chunk: &[f32],
    start_index: usize,
    global_amax: f32,
    radius: u8,
    codes: &mut [u8; BLOCK_SIZE],
) -> Result<u8, Nvfp4Error> {
    let reference_bits = encode_block_into(chunk, start_index, global_amax, codes)?;
    if reference_bits == 0 || radius == 0 {
        return Ok(reference_bits);
    }

    // The same normalization the reference applies before dividing by the scale.
    let mut normalized = [0.0_f32; BLOCK_SIZE];
    for (slot, &value) in normalized.iter_mut().zip(chunk) {
        *slot = (value / global_amax) * SCALE_PRODUCT;
    }
    let global_scale = global_scale_for_amax(global_amax);
    let error_for = |bits: u8| -> f32 {
        let scale = fp8_e4m3::decode(bits);
        let mut error = 0.0_f32;
        for (&value, &scaled) in chunk.iter().zip(&normalized) {
            let code = fp4_e2m1::encode_unchecked(scaled / scale);
            let difference = value - fp4_e2m1::decode(code) * scale * global_scale;
            error += difference * difference;
        }
        error
    };

    let mut best_bits = reference_bits;
    let mut best_error = error_for(reference_bits);
    for distance in 1..=radius {
        let below = reference_bits
            .checked_sub(distance)
            .filter(|&bits| bits >= MIN_POSITIVE_E4M3_BITS);
        let above = reference_bits
            .checked_add(distance)
            .filter(|&bits| bits <= MAX_E4M3_BITS);
        for bits in below.into_iter().chain(above) {
            let error = error_for(bits);
            if error < best_error {
                best_error = error;
                best_bits = bits;
            }
        }
    }

    if best_bits != reference_bits {
        let scale = fp8_e4m3::decode(best_bits);
        for (code, &scaled) in codes.iter_mut().zip(&normalized) {
            *code = fp4_e2m1::encode_unchecked(scaled / scale);
        }
    }
    Ok(best_bits)
}

/// Returns the largest absolute value in a chunk of finite values.
///
/// `first_index` is the flattened index of `values[0]` and is used to report
/// the first non-finite value with its tensor-wide index.  Combine chunk
/// results with `f32::max` to obtain the tensor-wide amax.
pub fn scan_chunk_amax(values: &[f32], first_index: usize) -> Result<f32, Nvfp4Error> {
    let mut maximum = 0.0_f32;
    for (offset, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(Nvfp4Error::NonFiniteInput {
                index: first_index + offset,
                value,
            });
        }
        maximum = maximum.max(value.abs());
    }
    Ok(maximum)
}

/// Encodes a block-aligned run of values into caller-provided buffers.
///
/// This is the single kernel shared by the reference, sequential streaming,
/// and parallel backends, so they cannot diverge.  `first_index` must be a
/// multiple of [`BLOCK_SIZE`] (the run starts on a block boundary); only the
/// final run of a tensor may end in a partial block.  `packed` must hold
/// exactly [`packed_len`] bytes and `block_scales` exactly [`block_count`]
/// entries for `values.len()`.  `global_amax` is the tensor-wide amax.
pub fn encode_chunk(
    values: &[f32],
    first_index: usize,
    global_amax: f32,
    packed: &mut [u8],
    block_scales: &mut [u8],
) -> Result<(), Nvfp4Error> {
    encode_chunk_with(
        values,
        first_index,
        global_amax,
        ScaleSelection::default(),
        packed,
        block_scales,
    )
}

/// [`encode_chunk`] with an explicit block-scale selection.
pub fn encode_chunk_with(
    values: &[f32],
    first_index: usize,
    global_amax: f32,
    selection: ScaleSelection,
    packed: &mut [u8],
    block_scales: &mut [u8],
) -> Result<(), Nvfp4Error> {
    if first_index % BLOCK_SIZE != 0 {
        return Err(Nvfp4Error::MisalignedChunk { first_index });
    }
    let expected_packed = packed_len(values.len());
    if packed.len() != expected_packed {
        return Err(Nvfp4Error::PackedLengthMismatch {
            elements: values.len(),
            expected: expected_packed,
            actual: packed.len(),
        });
    }
    let expected_blocks = block_count(values.len());
    if block_scales.len() != expected_blocks {
        return Err(Nvfp4Error::BlockScaleCountMismatch {
            expected: expected_blocks,
            actual: block_scales.len(),
        });
    }

    let mut codes = [0_u8; BLOCK_SIZE];
    for (block, chunk) in values.chunks(BLOCK_SIZE).enumerate() {
        let start_index = first_index + block * BLOCK_SIZE;
        block_scales[block] = match selection {
            ScaleSelection::Amax => encode_block_into(chunk, start_index, global_amax, &mut codes)?,
            ScaleSelection::MinMse { radius } => {
                encode_block_min_mse(chunk, start_index, global_amax, radius, &mut codes)?
            }
        };
        // Two codes per byte, first value in the low nibble.  A partial final
        // block leaves the high nibble of its last byte zero.
        let first_byte = block * (BLOCK_SIZE / VALUES_PER_BYTE);
        for pair in 0..chunk.len().div_ceil(VALUES_PER_BYTE) {
            let low = codes[pair * VALUES_PER_BYTE];
            let high = if pair * VALUES_PER_BYTE + 1 < chunk.len() {
                codes[pair * VALUES_PER_BYTE + 1]
            } else {
                0
            };
            packed[first_byte + pair] = low | (high << 4);
        }
    }
    Ok(())
}

/// Default number of blocks quantized per emitted chunk (4096 blocks =
/// 65,536 values, 32 KiB packed).
pub const DEFAULT_CHUNK_BLOCKS: usize = 4096;

/// Tensor-level results of a bounded streaming quantization.  The packed
/// payload has already been delivered to the caller's callback.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamedQuantization {
    block_scales: Vec<u8>,
    global_scale: f32,
    elements: usize,
}

impl StreamedQuantization {
    /// Assembles tensor-level results produced by a streaming backend.
    pub fn new(block_scales: Vec<u8>, global_scale: f32, elements: usize) -> Self {
        Self {
            block_scales,
            global_scale,
            elements,
        }
    }

    /// E4M3 block scale bit patterns, one per 16 values.
    pub fn block_scales(&self) -> &[u8] {
        &self.block_scales
    }

    /// Tensor-wide F32 decode scale.
    pub const fn global_scale(&self) -> f32 {
        self.global_scale
    }

    /// Number of source elements quantized.
    pub const fn elements(&self) -> usize {
        self.elements
    }
}

/// Error returned by [`quantize_replay_chunks`].
#[derive(Debug)]
pub enum Nvfp4StreamError<E> {
    /// The source values, shape, or chunk size were invalid.
    Quantization(Nvfp4Error),
    /// The caller's chunk callback failed.
    Callback(E),
}

/// Quantizes a shaped row-major tensor from a replayable value source in
/// bounded memory.
///
/// `values` is called twice: once to find the tensor-wide amax and once to
/// produce data.  Packed bytes are passed to `emit` in order, in chunks of up
/// to `chunk_blocks` blocks; the callback must consume each borrowed chunk
/// before returning.  Only one chunk of source values and codes is held, plus
/// the returned block scales (one byte per 16 values).  The emitted bytes
/// concatenate to exactly [`quantize_shaped`]'s payload, and the scales and
/// global scale are identical to it.
pub fn quantize_replay_chunks<F, I, C, E>(
    shape: &[usize],
    values: F,
    chunk_blocks: usize,
    emit: C,
) -> Result<StreamedQuantization, Nvfp4StreamError<E>>
where
    F: FnMut() -> I,
    I: IntoIterator<Item = f32>,
    C: FnMut(&[u8]) -> Result<(), E>,
{
    quantize_replay_chunks_with(shape, values, chunk_blocks, ScaleSelection::default(), emit)
}

/// [`quantize_replay_chunks`] with an explicit block-scale selection.
pub fn quantize_replay_chunks_with<F, I, C, E>(
    shape: &[usize],
    mut values: F,
    chunk_blocks: usize,
    selection: ScaleSelection,
    mut emit: C,
) -> Result<StreamedQuantization, Nvfp4StreamError<E>>
where
    F: FnMut() -> I,
    I: IntoIterator<Item = f32>,
    C: FnMut(&[u8]) -> Result<(), E>,
{
    let fail = Nvfp4StreamError::Quantization;
    let expected = checked_shape_elements(shape).map_err(fail)?;
    if chunk_blocks == 0 {
        return Err(fail(Nvfp4Error::InvalidChunkSize { chunk_blocks }));
    }

    let mut global_amax = 0.0_f32;
    let mut seen = 0_usize;
    for value in values() {
        if !value.is_finite() {
            return Err(fail(Nvfp4Error::NonFiniteInput { index: seen, value }));
        }
        global_amax = global_amax.max(value.abs());
        seen += 1;
    }
    if seen != expected {
        return Err(fail(Nvfp4Error::ShapeLengthMismatch {
            expected,
            actual: seen,
        }));
    }
    let global_scale = global_scale_for_amax(global_amax);

    let chunk_values = chunk_blocks.saturating_mul(BLOCK_SIZE);
    let mut buffer: Vec<f32> = Vec::with_capacity(chunk_values.min(expected));
    let mut packed: Vec<u8> = Vec::with_capacity(packed_len(buffer.capacity()));
    let mut block_scales = Vec::with_capacity(block_count(expected));
    let mut start = 0_usize;
    let mut source = values().into_iter();
    loop {
        buffer.clear();
        buffer.extend(source.by_ref().take(chunk_values));
        if buffer.is_empty() {
            break;
        }
        // Chunks hold whole blocks, so each starts on a byte boundary and
        // per-chunk packing concatenates to the whole-tensor packing.
        let scale_start = block_scales.len();
        block_scales.resize(scale_start + block_count(buffer.len()), 0);
        packed.resize(packed_len(buffer.len()), 0);
        encode_chunk_with(
            &buffer,
            start,
            global_amax,
            selection,
            &mut packed,
            &mut block_scales[scale_start..],
        )
        .map_err(fail)?;
        start += buffer.len();
        emit(&packed).map_err(Nvfp4StreamError::Callback)?;
    }
    if start != expected {
        return Err(fail(Nvfp4Error::ShapeLengthMismatch {
            expected,
            actual: start,
        }));
    }

    Ok(StreamedQuantization {
        block_scales,
        global_scale,
        elements: expected,
    })
}

/// Packs E2M1 bit patterns two per byte, with the first value in the low
/// nibble.  An odd final value leaves the high nibble zero.
pub fn pack(values: &[u8]) -> Result<Vec<u8>, Nvfp4Error> {
    let mut packed = Vec::with_capacity(packed_len(values.len()));
    for (pair_index, pair) in values.chunks(VALUES_PER_BYTE).enumerate() {
        let first = pair[0];
        if first > 0x0f {
            return Err(Nvfp4Error::PackedValueOutOfRange {
                index: pair_index * VALUES_PER_BYTE,
                value: first,
            });
        }
        let second = pair.get(1).copied().unwrap_or(0);
        if second > 0x0f {
            return Err(Nvfp4Error::PackedValueOutOfRange {
                index: pair_index * VALUES_PER_BYTE + 1,
                value: second,
            });
        }
        packed.push(first | (second << 4));
    }
    Ok(packed)
}

/// Unpacks E2M1 bit patterns from low-nibble-first bytes.
pub fn unpack(packed: &[u8], elements: usize) -> Result<Vec<u8>, Nvfp4Error> {
    let expected = packed_len(elements);
    if packed.len() != expected {
        return Err(Nvfp4Error::PackedLengthMismatch {
            elements,
            expected,
            actual: packed.len(),
        });
    }

    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        let byte = packed[index / VALUES_PER_BYTE];
        values.push(if index % VALUES_PER_BYTE == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        });
    }
    Ok(values)
}

/// Reconstructs F32 values from ModelQ-native NVFP4 fields.
pub fn dequantize(
    packed: &[u8],
    block_scales: &[u8],
    global_scale: f32,
    elements: usize,
) -> Result<Vec<f32>, Nvfp4Error> {
    validate_parts(packed, block_scales, global_scale, elements)?;
    let values = unpack(packed, elements)?;
    let mut output = Vec::with_capacity(elements);
    for (index, code) in values.into_iter().enumerate() {
        let block = index / BLOCK_SIZE;
        let block_scale_bits = block_scales[block];
        let block_scale = fp8_e4m3::decode(block_scale_bits);
        let reconstructed = fp4_e2m1::decode(code) * block_scale * global_scale;
        if !reconstructed.is_finite() {
            return Err(Nvfp4Error::DequantizedValueOverflow { index });
        }
        output.push(reconstructed);
    }
    Ok(output)
}

/// Validates packed values, block scales, and the tensor-wide decode scale.
///
/// Validation reads the packed bytes in place and allocates nothing, so it is
/// safe to use on tensors too large to unpack.
pub fn validate_parts(
    packed: &[u8],
    block_scales: &[u8],
    global_scale: f32,
    elements: usize,
) -> Result<(), Nvfp4Error> {
    if !global_scale.is_finite() || global_scale <= 0.0 {
        return Err(Nvfp4Error::InvalidGlobalScale {
            scale: global_scale,
        });
    }

    let expected_packed = packed_len(elements);
    if packed.len() != expected_packed {
        return Err(Nvfp4Error::PackedLengthMismatch {
            elements,
            expected: expected_packed,
            actual: packed.len(),
        });
    }

    let expected_blocks = block_count(elements);
    if block_scales.len() != expected_blocks {
        return Err(Nvfp4Error::BlockScaleCountMismatch {
            expected: expected_blocks,
            actual: block_scales.len(),
        });
    }

    for (block, &scale_bits) in block_scales.iter().enumerate() {
        validate_block_scale(scale_bits, block)?;
        if scale_bits == 0 {
            let start = block.saturating_mul(BLOCK_SIZE);
            let end = elements.min(start.saturating_add(BLOCK_SIZE));
            if let Some(index) = (start..end).find(|&index| code_at(packed, index) & 0x07 != 0) {
                return Err(Nvfp4Error::ZeroScaleWithNonzeroValue { block, index });
            }
        }
    }
    Ok(())
}

/// Lazily reconstructs F32 values after validating the parts.
///
/// Yields exactly the values [`dequantize`] returns, one at a time, without
/// materializing the tensor.  Use it to compare large tensors in bounded
/// memory.  Values are not checked for overflow here because a validated
/// decode scale and block scales cannot exceed the finite F32 range for data
/// produced by [`quantize`]; use [`dequantize`] when untrusted parts must be
/// proven finite.
pub fn dequantize_iter<'a>(
    packed: &'a [u8],
    block_scales: &'a [u8],
    global_scale: f32,
    elements: usize,
) -> Result<impl ExactSizeIterator<Item = f32> + 'a, Nvfp4Error> {
    validate_parts(packed, block_scales, global_scale, elements)?;
    Ok((0..elements).map(move |index| {
        let block_scale = fp8_e4m3::decode(block_scales[index / BLOCK_SIZE]);
        fp4_e2m1::decode(code_at(packed, index)) * block_scale * global_scale
    }))
}

fn code_at(packed: &[u8], index: usize) -> u8 {
    let byte = packed[index / VALUES_PER_BYTE];
    if index % VALUES_PER_BYTE == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

/// Returns the packed byte count for an element count.
pub const fn packed_len(elements: usize) -> usize {
    if elements % VALUES_PER_BYTE == 0 {
        elements / VALUES_PER_BYTE
    } else {
        elements / VALUES_PER_BYTE + 1
    }
}

/// Returns the number of fixed-size NVFP4 blocks for an element count.
pub const fn block_count(elements: usize) -> usize {
    if elements % BLOCK_SIZE == 0 {
        elements / BLOCK_SIZE
    } else {
        elements / BLOCK_SIZE + 1
    }
}

fn max_abs(values: &[f32]) -> Result<f32, Nvfp4Error> {
    let mut maximum = 0.0_f32;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(Nvfp4Error::NonFiniteInput { index, value });
        }
        maximum = maximum.max(value.abs());
    }
    Ok(maximum)
}

fn validate_block_scale(bits: u8, block: usize) -> Result<(), Nvfp4Error> {
    if bits == 0 {
        return Ok(());
    }
    let decoded = fp8_e4m3::decode(bits);
    if bits & 0x80 != 0 || !decoded.is_finite() || decoded <= 0.0 {
        return Err(Nvfp4Error::InvalidBlockScale { block, bits });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK_SIZE, FP4_MAX, FP8_MAX, Nvfp4Error, Nvfp4StreamError, block_count, dequantize,
        dequantize_iter, pack, packed_len, quantize, quantize_replay_chunks, quantize_shaped,
        unpack, validate_parts,
    };

    #[test]
    fn packs_and_unpacks_low_nibble_first() {
        let values = [0x0, 0x1, 0x7, 0xf, 0x8];
        let packed = pack(&values).expect("all E2M1 codes fit in a nibble");
        assert_eq!(packed, [0x10, 0xf7, 0x08]);
        assert_eq!(
            unpack(&packed, values.len()).expect("length matches"),
            values
        );
    }

    #[test]
    fn quantizes_and_reconstructs_one_exact_e2m1_block() {
        let source = [
            -FP4_MAX, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
            FP4_MAX,
        ];
        let quantized = quantize(&source).expect("finite values are valid");

        assert_eq!(quantized.len(), BLOCK_SIZE);
        assert_eq!(quantized.block_scales(), [0x7e]);
        assert_eq!(
            quantized.unpacked_values().expect("packed length matches"),
            [
                0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                0x06, 0x07,
            ]
        );
        let reconstructed = quantized.dequantize().expect("parts validate");
        for (actual, expected) in reconstructed.iter().zip(source) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn validates_shaped_final_dimension_and_preserves_flat_encoding() {
        let source = [0.0_f32; BLOCK_SIZE * 2];
        let quantized = quantize_shaped(&source, &[2, BLOCK_SIZE])
            .expect("two complete final-dimension blocks are valid");
        assert_eq!(quantized.len(), source.len());
        assert_eq!(quantized.packed_values(), [0; BLOCK_SIZE]);

        assert!(matches!(
            quantize_shaped(&source, &[4, BLOCK_SIZE / 2]),
            Err(Nvfp4Error::InvalidShape { .. })
        ));
        assert!(matches!(
            quantize_shaped(&source, &[]),
            Err(Nvfp4Error::InvalidShape { .. })
        ));
        assert!(matches!(
            quantize_shaped(&source, &[0, BLOCK_SIZE]),
            Err(Nvfp4Error::InvalidShape { .. })
        ));
    }

    #[test]
    fn checks_shaped_element_count_without_overflow() {
        let source = [0.0_f32; BLOCK_SIZE];
        assert_eq!(
            quantize_shaped(&source, &[2, BLOCK_SIZE]),
            Err(Nvfp4Error::ShapeLengthMismatch {
                expected: BLOCK_SIZE * 2,
                actual: BLOCK_SIZE,
            })
        );
        assert!(matches!(
            quantize_shaped(&source, &[usize::MAX, BLOCK_SIZE]),
            Err(Nvfp4Error::ShapeElementCountOverflow { .. })
        ));
    }

    #[test]
    fn uses_a_global_scale_and_independent_block_scales() {
        let mut source = vec![0.0; BLOCK_SIZE * 2];
        source[0] = 1.0;
        source[BLOCK_SIZE] = 6.0;
        let quantized = quantize(&source).expect("finite values are valid");

        assert_eq!(quantized.block_scales().len(), 2);
        assert!(quantized.block_scales()[0] < quantized.block_scales()[1]);
        assert!((quantized.global_scale() - 6.0 / (FP4_MAX * FP8_MAX)).abs() < 1e-9);
        let reconstructed = quantized.dequantize().expect("parts validate");
        assert!((reconstructed[0] - 1.0).abs() < 0.05);
        assert!((reconstructed[BLOCK_SIZE] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn zero_and_empty_tensors_have_explicit_safe_scales() {
        let zero = quantize(&[0.0; BLOCK_SIZE + 1]).expect("zero values are valid");
        assert_eq!(zero.global_scale(), 1.0);
        assert_eq!(zero.block_scales(), [0, 0]);
        assert_eq!(zero.packed_values(), [0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            zero.dequantize().expect("zero parts validate"),
            [0.0; BLOCK_SIZE + 1]
        );

        let empty = quantize(&[]).expect("empty tensors are valid");
        assert!(empty.is_empty());
        assert_eq!(empty.packed_values(), []);
        assert_eq!(empty.block_scales(), []);
        assert_eq!(empty.dequantize().expect("empty parts validate"), []);
    }

    #[test]
    fn nonfinite_inputs_are_rejected_before_writing_parts() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = quantize(&[1.0, value]).expect_err("nonfinite input is invalid");
            assert!(matches!(error, Nvfp4Error::NonFiniteInput { index: 1, .. }));
        }
    }

    #[test]
    fn validates_lengths_scales_and_zero_blocks() {
        assert_eq!(
            validate_parts(&[0], &[], 1.0, 1).expect_err("one value needs one block scale"),
            Nvfp4Error::BlockScaleCountMismatch {
                expected: 1,
                actual: 0,
            }
        );
        assert_eq!(
            validate_parts(&[0], &[0], 1.0, 3).expect_err("three values need two bytes"),
            Nvfp4Error::PackedLengthMismatch {
                elements: 3,
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            validate_parts(&[0], &[0x7f], 1.0, 1).expect_err("NaN scale is invalid"),
            Nvfp4Error::InvalidBlockScale {
                block: 0,
                bits: 0x7f,
            }
        );
        assert_eq!(
            validate_parts(&[0x01], &[0], 1.0, 1)
                .expect_err("nonzero value cannot use a zero scale"),
            Nvfp4Error::ZeroScaleWithNonzeroValue { block: 0, index: 0 }
        );
        assert_eq!(
            validate_parts(&[0], &[0], 0.0, 1).expect_err("global scale must be positive"),
            Nvfp4Error::InvalidGlobalScale { scale: 0.0 }
        );
    }

    #[test]
    fn supports_parts_round_trip_and_reports_lengths() {
        let source = [0.25; BLOCK_SIZE];
        let quantized = quantize(&source).expect("finite values are valid");
        let (packed, scales, global_scale, elements) = quantized.clone().into_parts();
        assert_eq!(packed.len(), packed_len(elements));
        assert_eq!(scales.len(), block_count(elements));
        let restored = super::QuantizedTensor::from_parts(packed, scales, global_scale, elements)
            .expect("quantized parts remain valid");
        assert_eq!(restored, quantized);
        assert_eq!(
            dequantize(
                restored.packed(),
                restored.scales(),
                restored.global_scale(),
                restored.len()
            )
            .expect("parts validate"),
            restored.dequantize().expect("parts validate")
        );
    }

    #[test]
    fn clamps_e4m3_scale_underflow_without_dividing_by_zero() {
        let mut source = vec![0.0; BLOCK_SIZE * 2];
        source[0] = f32::from_bits(1);
        source[BLOCK_SIZE] = 1.0;
        let quantized = quantize(&source).expect("finite subnormal values are valid");
        assert_eq!(quantized.block_scales()[0], 0x01);
        assert!(quantized.dequantize().is_ok());
    }

    /// Deterministic values spanning many magnitudes, signs, and exact zeros.
    fn spread_values(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|index| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0;
                let exponent = ((state >> 8) % 24) as i32 - 12;
                match index % 17 {
                    0 => 0.0,
                    _ => unit * 2.0_f32.powi(exponent),
                }
            })
            .collect()
    }

    fn streamed(
        values: &[f32],
        shape: &[usize],
        chunk_blocks: usize,
    ) -> (Vec<u8>, super::StreamedQuantization) {
        let mut packed = Vec::new();
        let result = quantize_replay_chunks(
            shape,
            || values.iter().copied(),
            chunk_blocks,
            |chunk| -> Result<(), ()> {
                packed.extend_from_slice(chunk);
                Ok(())
            },
        )
        .expect("streaming quantization succeeds");
        (packed, result)
    }

    fn assert_streaming_matches_reference(values: &[f32], shape: &[usize]) {
        let reference = quantize_shaped(values, shape).expect("reference succeeds");
        // 1 and 3 split the data awkwardly; the large value is a single chunk.
        for chunk_blocks in [1, 3, 7, 4096] {
            let (packed, result) = streamed(values, shape, chunk_blocks);
            assert_eq!(packed, reference.packed(), "chunk_blocks={chunk_blocks}");
            assert_eq!(result.block_scales(), reference.block_scales());
            assert_eq!(
                result.global_scale().to_bits(),
                reference.global_scale().to_bits()
            );
            assert_eq!(result.elements(), values.len());
        }
    }

    #[test]
    fn streaming_matches_reference_bit_for_bit() {
        for (rows, columns, seed) in [(1, 16, 1), (5, 48, 2), (64, 64, 3), (3, 4096 + 16, 4)] {
            let values = spread_values(rows * columns, seed);
            assert_streaming_matches_reference(&values, &[rows, columns]);
        }
    }

    #[test]
    fn streaming_matches_reference_for_edge_inputs() {
        assert_streaming_matches_reference(&[0.0; 64], &[4, 16]);

        let mut outlier = vec![0.001_f32; 128];
        outlier[77] = 6.0e6;
        assert_streaming_matches_reference(&outlier, &[8, 16]);

        let mut tiny = vec![0.0_f32; 32];
        tiny[0] = f32::from_bits(1);
        tiny[16] = 1.0;
        assert_streaming_matches_reference(&tiny, &[2, 16]);

        let negative: Vec<f32> = (0..48).map(|index| -(index as f32) - 0.5).collect();
        assert_streaming_matches_reference(&negative, &[3, 16]);
    }

    #[test]
    fn streaming_reports_the_first_non_finite_index() {
        let mut values = vec![1.0_f32; 64];
        values[40] = f32::NAN;
        values[50] = f32::INFINITY;
        let error =
            quantize_replay_chunks(&[4, 16], || values.iter().copied(), 2, |_| Ok::<(), ()>(()))
                .expect_err("non-finite input is rejected");
        assert!(matches!(
            error,
            Nvfp4StreamError::Quantization(Nvfp4Error::NonFiniteInput { index: 40, .. })
        ));
    }

    #[test]
    fn streaming_validates_shape_chunk_size_and_length() {
        let values = vec![1.0_f32; 32];
        let run = |shape: &[usize], chunk_blocks: usize, source: &[f32]| {
            quantize_replay_chunks(
                shape,
                || source.iter().copied(),
                chunk_blocks,
                |_| Ok::<(), ()>(()),
            )
        };
        assert!(matches!(
            run(&[2, 15], 1, &values),
            Err(Nvfp4StreamError::Quantization(
                Nvfp4Error::InvalidShape { .. }
            ))
        ));
        assert!(matches!(
            run(&[2, 16], 0, &values),
            Err(Nvfp4StreamError::Quantization(
                Nvfp4Error::InvalidChunkSize { chunk_blocks: 0 }
            ))
        ));
        assert!(matches!(
            run(&[3, 16], 1, &values),
            Err(Nvfp4StreamError::Quantization(
                Nvfp4Error::ShapeLengthMismatch {
                    expected: 48,
                    actual: 32
                }
            ))
        ));
    }

    #[test]
    fn streaming_propagates_callback_errors_and_bounds_chunks() {
        let values = spread_values(16 * 10, 9);
        let mut sizes = Vec::new();
        quantize_replay_chunks(
            &[10, 16],
            || values.iter().copied(),
            4,
            |chunk| {
                sizes.push(chunk.len());
                Ok::<(), ()>(())
            },
        )
        .expect("succeeds");
        // 10 blocks in chunks of 4 blocks (32 packed bytes): 4, 4, 2 blocks.
        assert_eq!(sizes, [32, 32, 16]);

        let error = quantize_replay_chunks(
            &[10, 16],
            || values.iter().copied(),
            4,
            |_| Err("sink failed"),
        )
        .expect_err("callback failure is surfaced");
        assert!(matches!(error, Nvfp4StreamError::Callback("sink failed")));
    }

    #[test]
    fn dequantize_iter_matches_dequantize_and_rejects_bad_parts() {
        let values = spread_values(16 * 12, 21);
        let quantized = quantize_shaped(&values, &[12, 16]).expect("quantizes");
        let eager = quantized.dequantize().expect("dequantizes");
        let lazy: Vec<f32> = dequantize_iter(
            quantized.packed(),
            quantized.block_scales(),
            quantized.global_scale(),
            values.len(),
        )
        .expect("parts validate")
        .collect();
        assert_eq!(
            eager
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            lazy.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );

        assert!(matches!(
            dequantize_iter(quantized.packed(), quantized.block_scales(), 0.0, 192),
            Err(Nvfp4Error::InvalidGlobalScale { .. })
        ));
        assert!(matches!(
            dequantize_iter(&[0xff], &[0x38], 1.0, 16),
            Err(Nvfp4Error::PackedLengthMismatch { .. })
        ));
    }

    /// The original whole-tensor algorithm: per-value candidate-search
    /// codecs and a scratch `Vec` per block, kept as an oracle independent of
    /// the production kernel.
    fn reference_quantize(values: &[f32]) -> (Vec<u8>, Vec<u8>, f32) {
        use crate::float::reference;

        let global_amax = values
            .iter()
            .fold(0.0_f32, |amax, value| amax.max(value.abs()));
        let global_scale = if global_amax == 0.0 {
            1.0
        } else {
            (global_amax / super::SCALE_PRODUCT).max(f32::from_bits(1))
        };
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        for chunk in values.chunks(BLOCK_SIZE) {
            let block_amax = chunk
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max);
            if block_amax == 0.0 {
                scales.push(0);
                codes.extend(std::iter::repeat_n(0, chunk.len()));
                continue;
            }
            let scale_input = (block_amax / global_amax) * FP8_MAX;
            let mut scale_bits = reference::encode_e4m3(scale_input);
            if scale_bits == 0 {
                scale_bits = 0x01;
            }
            let decoded = reference::decode_e4m3(scale_bits);
            scales.push(scale_bits);
            for &value in chunk {
                let scaled = ((value / global_amax) * super::SCALE_PRODUCT) / decoded;
                codes.push(reference::encode_e2m1(scaled).expect("finite"));
            }
        }
        (
            pack(&codes).expect("codes fit a nibble"),
            scales,
            global_scale,
        )
    }

    fn assert_matches_original_algorithm(values: &[f32]) {
        let (packed, scales, global_scale) = reference_quantize(values);
        let quantized = quantize(values).expect("quantizes");
        assert_eq!(quantized.packed(), packed);
        assert_eq!(quantized.block_scales(), scales);
        assert_eq!(quantized.global_scale().to_bits(), global_scale.to_bits());
    }

    #[test]
    fn production_kernel_matches_the_original_algorithm() {
        // Lengths include partial final blocks and odd counts.
        for (count, seed) in [
            (1, 1),
            (15, 2),
            (16, 3),
            (17, 4),
            (31, 5),
            (4096, 6),
            (65_537, 7),
        ] {
            assert_matches_original_algorithm(&spread_values(count, seed));
        }
        assert_matches_original_algorithm(&[0.0; 64]);
        let mut outlier = vec![0.001_f32; 160];
        outlier[77] = 6.0e6;
        assert_matches_original_algorithm(&outlier);
        let mut tiny = vec![0.0_f32; 48];
        tiny[0] = f32::from_bits(1);
        tiny[16] = 1.0;
        tiny[40] = -3.0e-30;
        assert_matches_original_algorithm(&tiny);
        // Negative zero and exact tie values at the grid midpoints.
        let ties: Vec<f32> = [0.25_f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
            .iter()
            .flat_map(|&value| [value, -value])
            .chain([-0.0, 0.0])
            .collect();
        assert_matches_original_algorithm(&ties);
    }

    #[test]
    fn production_kernel_matches_the_original_on_many_random_tensors() {
        for seed in 100..160 {
            let count = 16 * (1 + (seed as usize % 37)) + (seed as usize % 3) * 5;
            assert_matches_original_algorithm(&spread_values(count, seed));
        }
    }

    #[test]
    fn encode_chunk_validates_alignment_and_buffer_lengths() {
        let values = vec![1.0_f32; 32];
        let mut packed = vec![0_u8; 16];
        let mut scales = vec![0_u8; 2];
        assert!(super::encode_chunk(&values, 8, 1.0, &mut packed, &mut scales).is_err());
        assert!(super::encode_chunk(&values, 0, 1.0, &mut packed[..15], &mut scales).is_err());
        assert!(super::encode_chunk(&values, 0, 1.0, &mut packed, &mut scales[..1]).is_err());
        assert!(super::encode_chunk(&values, 0, 1.0, &mut packed, &mut scales).is_ok());
    }

    use super::{ScaleSelection, quantize_replay_chunks_with, quantize_shaped_with, quantize_with};

    fn mse(values: &[f32], quantized: &super::QuantizedTensor) -> f64 {
        let decoded = quantized.dequantize().expect("dequantizes");
        values
            .iter()
            .zip(&decoded)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / values.len() as f64
    }

    #[test]
    fn selection_labels_and_defaults() {
        assert!(ScaleSelection::Amax.is_default());
        assert!(ScaleSelection::MinMse { radius: 0 }.is_default());
        assert!(!ScaleSelection::MinMse { radius: 4 }.is_default());
        assert_eq!(ScaleSelection::default(), ScaleSelection::Amax);
        assert_eq!(ScaleSelection::Amax.label(), "amax");
        assert_eq!(ScaleSelection::MinMse { radius: 6 }.label(), "min-mse:r6");
    }

    #[test]
    fn radius_zero_is_bit_identical_to_the_reference_rule() {
        for (count, seed) in [(16, 1), (96, 2), (4096, 3), (1000, 4)] {
            let values = spread_values(count, seed);
            let reference = quantize(&values).expect("quantizes");
            let searched =
                quantize_with(&values, ScaleSelection::MinMse { radius: 0 }).expect("quantizes");
            assert_eq!(searched, reference, "count={count}");
        }
    }

    #[test]
    fn the_search_never_increases_error_and_usually_reduces_it() {
        let mut improved = 0;
        for seed in 0..12 {
            let values = spread_values(16 * 300, 100 + seed);
            let reference = mse(&values, &quantize(&values).expect("quantizes"));
            for radius in [1_u8, 4, 8] {
                let searched =
                    quantize_with(&values, ScaleSelection::MinMse { radius }).expect("quantizes");
                let error = mse(&values, &searched);
                assert!(
                    error <= reference * (1.0 + 1e-9),
                    "seed={seed} radius={radius}: {error} > {reference}"
                );
                if error < reference * 0.999 {
                    improved += 1;
                }
            }
        }
        assert!(
            improved >= 24,
            "the search improved only {improved} of 36 runs"
        );
    }

    #[test]
    fn the_search_matches_an_independent_brute_force_oracle() {
        use crate::float::{fp4_e2m1, fp8_e4m3};

        // Re-derives the choice with the public checked codecs and the same
        // arithmetic, outside the production loop.
        fn oracle(block: &[f32], global_amax: f32, radius: u8) -> u8 {
            let block_amax = block.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            if block_amax == 0.0 {
                return 0;
            }
            let mut reference = fp8_e4m3::encode((block_amax / global_amax) * FP8_MAX);
            if reference == 0 {
                reference = 1;
            }
            let global_scale = super::global_scale_for_amax(global_amax);
            let error = |bits: u8| {
                let scale = fp8_e4m3::decode(bits);
                let mut sum = 0.0_f32;
                for &value in block {
                    let scaled = ((value / global_amax) * super::SCALE_PRODUCT) / scale;
                    let code = fp4_e2m1::encode(scaled).expect("finite");
                    let difference = value - fp4_e2m1::decode(code) * scale * global_scale;
                    sum += difference * difference;
                }
                sum
            };
            let mut candidates: Vec<u8> = (1..=0x7e_u8)
                .filter(|&bits| bits.abs_diff(reference) <= radius)
                .collect();
            // Reference first, then nearest, lower code first.
            candidates.sort_by_key(|&bits| (bits.abs_diff(reference), bits));
            let mut best = candidates[0];
            let mut best_error = error(best);
            for &bits in &candidates[1..] {
                let candidate_error = error(bits);
                if candidate_error < best_error {
                    best = bits;
                    best_error = candidate_error;
                }
            }
            best
        }

        for seed in 0..40 {
            let values = spread_values(16 * 25, 500 + seed);
            let global_amax = values.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            for radius in [1_u8, 3, 6, 12] {
                let searched =
                    quantize_with(&values, ScaleSelection::MinMse { radius }).expect("quantizes");
                for (block_index, block) in values.chunks(16).enumerate() {
                    assert_eq!(
                        searched.block_scales()[block_index],
                        oracle(block, global_amax, radius),
                        "seed={seed} radius={radius} block={block_index}"
                    );
                }
            }
        }
    }

    #[test]
    fn searched_output_is_valid_and_keeps_the_global_scale() {
        let values = spread_values(16 * 500 + 5, 77);
        let reference = quantize(&values).expect("quantizes");
        let searched =
            quantize_with(&values, ScaleSelection::MinMse { radius: 8 }).expect("quantizes");
        assert_eq!(
            searched.global_scale().to_bits(),
            reference.global_scale().to_bits()
        );
        assert_eq!(searched.packed().len(), reference.packed().len());
        validate_parts(
            searched.packed(),
            searched.block_scales(),
            searched.global_scale(),
            values.len(),
        )
        .expect("a searched tensor is a valid tensor");
        for (&bits, block) in searched.block_scales().iter().zip(values.chunks(16)) {
            if block.iter().all(|&value| value == 0.0) {
                assert_eq!(bits, 0);
            } else {
                assert!((1..=0x7e).contains(&bits), "scale {bits:#x}");
            }
        }
        assert!(
            searched
                .dequantize()
                .expect("finite")
                .iter()
                .all(|v| v.is_finite())
        );
        // A partial final block is searched too, and the result differs.
        assert_ne!(searched, reference);
    }

    #[test]
    fn streaming_equals_whole_slice_for_the_search() {
        let selection = ScaleSelection::MinMse { radius: 6 };
        let values = spread_values(16 * 700, 9);
        let shape = [700, 16];
        let reference = quantize_shaped_with(&values, &shape, selection).expect("quantizes");
        for chunk_blocks in [1, 3, 7, 4096] {
            let mut packed = Vec::new();
            let streamed = quantize_replay_chunks_with(
                &shape,
                || values.iter().copied(),
                chunk_blocks,
                selection,
                |chunk| -> Result<(), ()> {
                    packed.extend_from_slice(chunk);
                    Ok(())
                },
            )
            .expect("streams");
            assert_eq!(packed, reference.packed(), "chunk_blocks={chunk_blocks}");
            assert_eq!(streamed.block_scales(), reference.block_scales());
        }
    }

    #[test]
    fn the_search_reports_non_finite_input_like_the_reference() {
        let mut values = vec![1.0_f32; 64];
        values[37] = f32::NAN;
        let selection = ScaleSelection::MinMse { radius: 4 };
        let error = quantize_with(&values, selection).expect_err("NaN is rejected");
        assert!(matches!(
            error,
            Nvfp4Error::NonFiniteInput { index: 37, .. }
        ));
    }
}
