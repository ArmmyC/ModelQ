//! Group-wise low-bit quantization for 1 to 8 bits (ADR 0030).
//!
//! Two schemes share one packing layout:
//!
//! - [`Scheme::Symmetric`] (2 to 8 bits): each group of `group_size`
//!   consecutive values has one F32 scale, `max|w| / qmax` with
//!   `qmax = 2^(bits-1) - 1`. Values are rounded with [`f32::round`] (ties
//!   away from zero) and clamped to `[-qmax, qmax]`. The two's-complement code
//!   `-2^(bits-1)` is reserved and rejected when decoding. An all-zero group
//!   uses a scale of `1.0`, and a scale that would underflow to zero is raised
//!   to the smallest positive subnormal, exactly as in the INT4 reference
//!   ([`crate::int4`]). At 4 bits the output is therefore bit-identical to it.
//! - [`Scheme::Sign`] (1 bit): each code says whether the value is
//!   non-negative. The group scale is the mean absolute value, which is the
//!   least-squares choice for a single sign per value. An all-zero group has
//!   scale `0.0` and decodes to exact zeros.
//!
//! Codes are stored as an LSB-first bitstream: value `i` occupies bits
//! `[i * bits, (i + 1) * bits)` of the stream, and byte `k` holds stream bits
//! `[8k, 8k + 8)`. At 4 bits this is the low-nibble-first layout of INT4.
//! Groups are formed in linear storage order, and the final group may be
//! shorter than `group_size`.

use std::fmt;

/// The smallest supported bit width.
pub const MIN_BITS: u8 = 1;
/// The largest supported bit width.
pub const MAX_BITS: u8 = 8;

/// How codes map to values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// Symmetric signed codes with a max-abs scale (2 to 8 bits).
    Symmetric,
    /// One sign bit per value with a mean-abs scale (1 bit).
    Sign,
}

/// The parameters that define one low-bit encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LowBitConfig {
    /// Bits per stored value.
    pub bits: u8,
    /// Values per scale group; the final group may be shorter.
    pub group_size: usize,
    /// The code-to-value rule.
    pub scheme: Scheme,
}

impl LowBitConfig {
    /// Checks that the bit width, scheme and group size are valid together.
    pub fn validate(&self) -> Result<(), LowBitError> {
        if self.group_size == 0 {
            return Err(LowBitError::InvalidGroupSize);
        }
        match self.scheme {
            Scheme::Symmetric if (2..=MAX_BITS).contains(&self.bits) => Ok(()),
            Scheme::Sign if self.bits == 1 => Ok(()),
            _ => Err(LowBitError::InvalidBits {
                bits: self.bits,
                scheme: self.scheme,
            }),
        }
    }

    /// The largest magnitude a symmetric code can take; `0` for the sign scheme.
    pub fn qmax(&self) -> i32 {
        match self.scheme {
            Scheme::Symmetric => (1_i32 << (self.bits - 1)) - 1,
            Scheme::Sign => 0,
        }
    }
}

/// Errors from the low-bit codecs.
#[derive(Debug, Clone, PartialEq)]
pub enum LowBitError {
    /// A group size of zero cannot define a group.
    InvalidGroupSize,
    /// The bit width is outside the range of the scheme.
    InvalidBits { bits: u8, scheme: Scheme },
    /// The source contains a NaN or infinity.
    NonFiniteInput { index: usize, value: f32 },
    /// A group scale is not finite or not positive where one is required.
    InvalidScale { group: usize, scale: f32 },
    /// A scaled value is not finite, so it cannot be rounded.
    QuantizedValueOverflow { index: usize },
    /// A code has bits set above the configured width.
    CodeOutOfRange { index: usize, code: u32, bits: u8 },
    /// A code is the reserved two's-complement minimum.
    ReservedCode { index: usize },
    /// The packed byte count does not match the element count.
    PackedLengthMismatch {
        elements: usize,
        bits: u8,
        expected: usize,
        actual: usize,
    },
    /// The number of scales does not match the number of groups.
    ScaleCountMismatch { expected: usize, actual: usize },
    /// The element count times the bit width does not fit in `usize`.
    SizeOverflow { elements: usize, bits: u8 },
    /// A dequantized value is not finite.
    DequantizedValueOverflow { index: usize },
}

impl fmt::Display for LowBitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGroupSize => formatter.write_str("the group size must be at least 1"),
            Self::InvalidBits { bits, scheme } => match scheme {
                Scheme::Symmetric => write!(
                    formatter,
                    "symmetric group-wise quantization needs 2 to 8 bits, not {bits}"
                ),
                Scheme::Sign => write!(
                    formatter,
                    "sign quantization needs exactly 1 bit, not {bits}"
                ),
            },
            Self::NonFiniteInput { index, value } => {
                write!(formatter, "non-finite input {value} at index {index}")
            }
            Self::InvalidScale { group, scale } => {
                write!(formatter, "group {group} has an invalid scale {scale}")
            }
            Self::QuantizedValueOverflow { index } => {
                write!(formatter, "the scaled value at index {index} is not finite")
            }
            Self::CodeOutOfRange { index, code, bits } => {
                write!(
                    formatter,
                    "code {code} at index {index} does not fit in {bits} bits"
                )
            }
            Self::ReservedCode { index } => {
                write!(
                    formatter,
                    "the reserved minimum code appears at index {index}"
                )
            }
            Self::PackedLengthMismatch {
                elements,
                bits,
                expected,
                actual,
            } => write!(
                formatter,
                "{elements} values at {bits} bits need {expected} packed bytes, got {actual}"
            ),
            Self::ScaleCountMismatch { expected, actual } => {
                write!(formatter, "expected {expected} group scales, got {actual}")
            }
            Self::SizeOverflow { elements, bits } => {
                write!(
                    formatter,
                    "{elements} values at {bits} bits overflow the size type"
                )
            }
            Self::DequantizedValueOverflow { index } => {
                write!(
                    formatter,
                    "the dequantized value at index {index} is not finite"
                )
            }
        }
    }
}

impl std::error::Error for LowBitError {}

/// Number of groups for `elements` values.
pub fn group_count(elements: usize, group_size: usize) -> usize {
    elements / group_size + usize::from(elements % group_size != 0)
}

/// Bytes needed to store `elements` values of `bits` bits each.
pub fn packed_len(elements: usize, bits: u8) -> Result<usize, LowBitError> {
    let total_bits = elements
        .checked_mul(usize::from(bits))
        .ok_or(LowBitError::SizeOverflow { elements, bits })?;
    Ok(total_bits / 8 + usize::from(total_bits % 8 != 0))
}

/// The scale of one group, computed from its values.
///
/// `start` is the index of the group's first value, used for error reports.
pub fn group_scale(
    values: &[f32],
    start: usize,
    config: &LowBitConfig,
) -> Result<f32, LowBitError> {
    config.validate()?;
    match config.scheme {
        Scheme::Symmetric => {
            let mut max_abs = 0.0_f32;
            for (offset, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(LowBitError::NonFiniteInput {
                        index: start.saturating_add(offset),
                        value,
                    });
                }
                max_abs = max_abs.max(value.abs());
            }
            let mut scale = if max_abs == 0.0 {
                1.0
            } else {
                max_abs / config.qmax() as f32
            };
            if scale == 0.0 {
                scale = f32::from_bits(1);
            }
            Ok(scale)
        }
        Scheme::Sign => {
            let mut sum = 0.0_f64;
            for (offset, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(LowBitError::NonFiniteInput {
                        index: start.saturating_add(offset),
                        value,
                    });
                }
                sum += f64::from(value.abs());
            }
            if values.is_empty() {
                return Ok(0.0);
            }
            Ok((sum / values.len() as f64) as f32)
        }
    }
}

/// The unsigned code for one value, given its group's scale.
pub fn code_for(
    value: f32,
    scale: f32,
    index: usize,
    config: &LowBitConfig,
) -> Result<u32, LowBitError> {
    config.validate()?;
    if !value.is_finite() {
        return Err(LowBitError::NonFiniteInput { index, value });
    }
    match config.scheme {
        Scheme::Symmetric => {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(LowBitError::InvalidScale { group: 0, scale });
            }
            let scaled = value / scale;
            if !scaled.is_finite() {
                return Err(LowBitError::QuantizedValueOverflow { index });
            }
            let qmax = config.qmax() as f32;
            let quantized = scaled.round().clamp(-qmax, qmax) as i32;
            Ok((quantized as u32) & mask(config.bits))
        }
        Scheme::Sign => Ok(u32::from(value >= 0.0)),
    }
}

/// The value a code stands for, given its group's scale.
pub fn value_for(
    code: u32,
    scale: f32,
    index: usize,
    config: &LowBitConfig,
) -> Result<f32, LowBitError> {
    config.validate()?;
    if code > mask(config.bits) {
        return Err(LowBitError::CodeOutOfRange {
            index,
            code,
            bits: config.bits,
        });
    }
    match config.scheme {
        Scheme::Symmetric => {
            let sign_bit = 1_u32 << (config.bits - 1);
            let quantized = if code & sign_bit == 0 {
                code as i32
            } else {
                code as i32 - (1_i32 << config.bits)
            };
            if quantized == -(sign_bit as i32) {
                return Err(LowBitError::ReservedCode { index });
            }
            let value = quantized as f32 * scale;
            if !value.is_finite() {
                return Err(LowBitError::DequantizedValueOverflow { index });
            }
            Ok(value)
        }
        Scheme::Sign => Ok(if code == 1 { scale } else { -scale }),
    }
}

fn mask(bits: u8) -> u32 {
    (1_u32 << bits) - 1
}

/// Streams codes into an LSB-first bitstream.
///
/// Completed bytes can be taken with [`BitWriter::drain`] while encoding, so a
/// caller can write a large tensor without holding all of it in memory.
#[derive(Debug, Clone)]
pub struct BitWriter {
    bits: u8,
    buffer: u64,
    filled: u32,
    bytes: Vec<u8>,
}

impl BitWriter {
    /// Creates a writer for codes of `bits` bits each (1 to 8).
    pub fn new(bits: u8) -> Self {
        debug_assert!((MIN_BITS..=MAX_BITS).contains(&bits));
        Self {
            bits,
            buffer: 0,
            filled: 0,
            bytes: Vec::new(),
        }
    }

    /// Appends one code. The caller guarantees `code < 2^bits`.
    pub fn push(&mut self, code: u32) {
        self.buffer |= u64::from(code) << self.filled;
        self.filled += u32::from(self.bits);
        while self.filled >= 8 {
            self.bytes.push(self.buffer as u8);
            self.buffer >>= 8;
            self.filled -= 8;
        }
    }

    /// Takes the complete bytes written so far.
    pub fn drain(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.bytes)
    }

    /// Finishes the stream, padding the last byte with zero bits.
    pub fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            self.bytes.push(self.buffer as u8);
        }
        self.bytes
    }
}

/// Packs codes into an LSB-first bitstream. Every code must fit in `bits`.
pub fn pack_codes(codes: &[u32], bits: u8) -> Result<Vec<u8>, LowBitError> {
    let mut writer = BitWriter::new(bits);
    for (index, &code) in codes.iter().enumerate() {
        if code > mask(bits) {
            return Err(LowBitError::CodeOutOfRange { index, code, bits });
        }
        writer.push(code);
    }
    Ok(writer.finish())
}

/// Unpacks `elements` codes of `bits` bits from an LSB-first bitstream.
pub fn unpack_codes(packed: &[u8], elements: usize, bits: u8) -> Result<Vec<u32>, LowBitError> {
    let expected = packed_len(elements, bits)?;
    if packed.len() != expected {
        return Err(LowBitError::PackedLengthMismatch {
            elements,
            bits,
            expected,
            actual: packed.len(),
        });
    }
    let width = u32::from(bits);
    let mut codes = Vec::with_capacity(elements);
    for index in 0..elements {
        let position = index * usize::from(bits);
        let byte = position / 8;
        let shift = (position % 8) as u32;
        // A code spans at most two bytes because shift + bits <= 15.
        let low = u32::from(packed[byte]);
        let high = packed.get(byte + 1).copied().map_or(0, u32::from);
        codes.push(((low | (high << 8)) >> shift) & mask(bits));
        debug_assert!(shift + width <= 15);
    }
    Ok(codes)
}

/// A tensor encoded with one configuration: packed codes and group scales.
#[derive(Debug, Clone, PartialEq)]
pub struct LowBitTensor {
    config: LowBitConfig,
    elements: usize,
    packed: Vec<u8>,
    scales: Vec<f32>,
}

impl LowBitTensor {
    /// Quantizes a tensor's values in storage order.
    pub fn quantize(values: &[f32], config: LowBitConfig) -> Result<Self, LowBitError> {
        config.validate()?;
        let groups = group_count(values.len(), config.group_size);
        let mut scales = Vec::with_capacity(groups);
        let mut writer = BitWriter::new(config.bits);
        for group in 0..groups {
            let start = group * config.group_size;
            let end = values.len().min(start.saturating_add(config.group_size));
            let slice = &values[start..end];
            let scale = group_scale(slice, start, &config)?;
            for (offset, &value) in slice.iter().enumerate() {
                writer.push(code_for(value, scale, start + offset, &config)?);
            }
            scales.push(scale);
        }
        Ok(Self {
            config,
            elements: values.len(),
            packed: writer.finish(),
            scales,
        })
    }

    /// Builds a tensor from stored parts, checking them against each other.
    pub fn from_parts(
        config: LowBitConfig,
        elements: usize,
        packed: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self, LowBitError> {
        config.validate()?;
        let expected = packed_len(elements, config.bits)?;
        if packed.len() != expected {
            return Err(LowBitError::PackedLengthMismatch {
                elements,
                bits: config.bits,
                expected,
                actual: packed.len(),
            });
        }
        let groups = group_count(elements, config.group_size);
        if scales.len() != groups {
            return Err(LowBitError::ScaleCountMismatch {
                expected: groups,
                actual: scales.len(),
            });
        }
        Ok(Self {
            config,
            elements,
            packed,
            scales,
        })
    }

    /// The configuration this tensor was encoded with.
    pub fn config(&self) -> LowBitConfig {
        self.config
    }

    /// Number of values.
    pub fn elements(&self) -> usize {
        self.elements
    }

    /// The packed codes.
    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    /// One scale per group.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Reconstructs every value.
    pub fn dequantize(&self) -> Result<Vec<f32>, LowBitError> {
        let codes = unpack_codes(&self.packed, self.elements, self.config.bits)?;
        let mut values = Vec::with_capacity(self.elements);
        for (index, code) in codes.into_iter().enumerate() {
            let scale = self.scales[index / self.config.group_size];
            values.push(value_for(code, scale, index, &self.config)?);
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::int4;

    fn symmetric(bits: u8, group_size: usize) -> LowBitConfig {
        LowBitConfig {
            bits,
            group_size,
            scheme: Scheme::Symmetric,
        }
    }

    /// A deterministic generator, so the property tests are reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }

        fn unit(&mut self) -> f32 {
            (self.next() % 2_000_001) as f32 / 1_000_000.0 - 1.0
        }
    }

    #[test]
    fn configurations_are_checked() {
        assert!(
            symmetric(1, 8).validate().is_err(),
            "symmetric needs 2 bits"
        );
        assert!(symmetric(9, 8).validate().is_err());
        assert!(symmetric(2, 0).validate().is_err());
        assert!(
            LowBitConfig {
                bits: 2,
                group_size: 8,
                scheme: Scheme::Sign
            }
            .validate()
            .is_err()
        );
        assert!(symmetric(2, 1).validate().is_ok());
        assert!(
            LowBitConfig {
                bits: 1,
                group_size: 8,
                scheme: Scheme::Sign
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn qmax_matches_the_signed_range() {
        assert_eq!(symmetric(2, 1).qmax(), 1);
        assert_eq!(symmetric(3, 1).qmax(), 3);
        assert_eq!(symmetric(4, 1).qmax(), 7);
        assert_eq!(symmetric(8, 1).qmax(), 127);
    }

    #[test]
    fn packed_length_rounds_up_to_whole_bytes() {
        assert_eq!(packed_len(0, 3).unwrap(), 0);
        assert_eq!(packed_len(8, 3).unwrap(), 3);
        assert_eq!(packed_len(5, 1).unwrap(), 1);
        assert_eq!(packed_len(3, 8).unwrap(), 3);
        assert!(packed_len(usize::MAX, 8).is_err());
    }

    #[test]
    fn every_single_code_round_trips_at_every_position_and_width() {
        for bits in MIN_BITS..=MAX_BITS {
            for code in 0..=mask(bits) {
                for position in 0..9 {
                    let mut codes = vec![0_u32; 9];
                    codes[position] = code;
                    let packed = pack_codes(&codes, bits).unwrap();
                    assert_eq!(
                        unpack_codes(&packed, codes.len(), bits).unwrap(),
                        codes,
                        "bits {bits}"
                    );
                }
            }
        }
    }

    #[test]
    fn every_eight_value_pattern_round_trips_at_one_and_two_bits() {
        for bits in [1_u8, 2] {
            let count = 1_u32 << (8 * u32::from(bits));
            for pattern in 0..count {
                let codes: Vec<u32> = (0..8)
                    .map(|index| (pattern >> (index * u32::from(bits))) & mask(bits))
                    .collect();
                let packed = pack_codes(&codes, bits).unwrap();
                assert_eq!(packed.len(), usize::from(bits));
                assert_eq!(unpack_codes(&packed, 8, bits).unwrap(), codes);
            }
        }
    }

    #[test]
    fn random_codes_round_trip_for_every_width_and_length() {
        let mut random = Lcg(7);
        for bits in MIN_BITS..=MAX_BITS {
            for length in 0..60 {
                let codes: Vec<u32> = (0..length)
                    .map(|_| (random.next() as u32) & mask(bits))
                    .collect();
                let packed = pack_codes(&codes, bits).unwrap();
                assert_eq!(packed.len(), packed_len(length, bits).unwrap());
                assert_eq!(unpack_codes(&packed, length, bits).unwrap(), codes);
            }
        }
    }

    #[test]
    fn streaming_writer_matches_whole_slice_packing() {
        let mut random = Lcg(11);
        for bits in MIN_BITS..=MAX_BITS {
            let codes: Vec<u32> = (0..1000)
                .map(|_| (random.next() as u32) & mask(bits))
                .collect();
            let whole = pack_codes(&codes, bits).unwrap();
            let mut writer = BitWriter::new(bits);
            let mut streamed = Vec::new();
            for (index, &code) in codes.iter().enumerate() {
                writer.push(code);
                if index % 37 == 0 {
                    streamed.extend(writer.drain());
                }
            }
            streamed.extend(writer.finish());
            assert_eq!(streamed, whole, "bits {bits}");
        }
    }

    #[test]
    fn symmetric_codes_round_trip_exactly_at_unit_scale() {
        for bits in 2..=MAX_BITS {
            let config = symmetric(bits, 1);
            let reserved = -(1_i32 << (bits - 1));
            for code in 0..=mask(bits) {
                let value = value_for(code, 1.0, 0, &config);
                let as_signed = if code & (1 << (bits - 1)) == 0 {
                    code as i32
                } else {
                    code as i32 - (1 << bits)
                };
                if as_signed == reserved {
                    assert_eq!(value, Err(LowBitError::ReservedCode { index: 0 }));
                    continue;
                }
                let value = value.unwrap();
                assert_eq!(
                    code_for(value, 1.0, 0, &config).unwrap(),
                    code,
                    "bits {bits}, code {code}"
                );
            }
        }
    }

    #[test]
    fn reserved_and_out_of_range_codes_are_rejected() {
        let config = symmetric(4, 8);
        assert_eq!(
            value_for(8, 1.0, 3, &config),
            Err(LowBitError::ReservedCode { index: 3 })
        );
        assert!(matches!(
            value_for(16, 1.0, 0, &config),
            Err(LowBitError::CodeOutOfRange { .. })
        ));
    }

    #[test]
    fn rounding_is_ties_away_from_zero_and_clamped() {
        let config = symmetric(3, 4);
        // qmax is 3, so 3.5 rounds to 4 and clamps to 3; -2.5 rounds to -3.
        assert_eq!(
            value_for(code_for(3.5, 1.0, 0, &config).unwrap(), 1.0, 0, &config).unwrap(),
            3.0
        );
        assert_eq!(
            value_for(code_for(-2.5, 1.0, 0, &config).unwrap(), 1.0, 0, &config).unwrap(),
            -3.0
        );
        assert_eq!(
            value_for(code_for(0.5, 1.0, 0, &config).unwrap(), 1.0, 0, &config).unwrap(),
            1.0
        );
    }

    #[test]
    fn golden_two_bit_group() {
        // max|w| = 1, qmax = 1: codes 0, 1, -1 (= 0b11), 1 packed LSB-first.
        let config = symmetric(2, 4);
        let tensor = LowBitTensor::quantize(&[0.0, 1.0, -1.0, 0.5], config).unwrap();
        assert_eq!(tensor.scales(), &[1.0]);
        assert_eq!(tensor.packed(), &[0x74]);
        assert_eq!(tensor.dequantize().unwrap(), vec![0.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn golden_three_bit_group() {
        // max|w| = 3, qmax = 3, scale 1: codes 3, 5, 2, 6, 0, 0, 0, 0.
        let config = symmetric(3, 8);
        let values = [3.0, -3.0, 1.5, -1.5, 0.0, 0.0, 0.0, 0.0];
        let tensor = LowBitTensor::quantize(&values, config).unwrap();
        assert_eq!(tensor.scales(), &[1.0]);
        assert_eq!(tensor.packed(), &[0xAB, 0x0C, 0x00]);
        assert_eq!(
            tensor.dequantize().unwrap(),
            vec![3.0, -3.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn golden_sign_group() {
        // Mean |w| = 2; signs +, -, +, - give bits 1, 0, 1, 0 = 0b0101.
        let config = LowBitConfig {
            bits: 1,
            group_size: 4,
            scheme: Scheme::Sign,
        };
        let tensor = LowBitTensor::quantize(&[1.0, -3.0, 2.0, -2.0], config).unwrap();
        assert_eq!(tensor.scales(), &[2.0]);
        assert_eq!(tensor.packed(), &[0x05]);
        assert_eq!(tensor.dequantize().unwrap(), vec![2.0, -2.0, 2.0, -2.0]);
    }

    #[test]
    fn zero_groups_use_the_reference_scales() {
        let symmetric_tensor = LowBitTensor::quantize(&[0.0; 5], symmetric(4, 2)).unwrap();
        assert_eq!(symmetric_tensor.scales(), &[1.0, 1.0, 1.0]);
        assert_eq!(symmetric_tensor.dequantize().unwrap(), vec![0.0; 5]);

        let sign = LowBitConfig {
            bits: 1,
            group_size: 3,
            scheme: Scheme::Sign,
        };
        let sign_tensor = LowBitTensor::quantize(&[0.0; 3], sign).unwrap();
        assert_eq!(sign_tensor.scales(), &[0.0]);
        assert_eq!(sign_tensor.dequantize().unwrap(), vec![0.0; 3]);
    }

    #[test]
    fn non_finite_inputs_are_reported_with_their_index() {
        let config = symmetric(4, 2);
        assert!(matches!(
            LowBitTensor::quantize(&[1.0, 2.0, f32::NAN], config),
            Err(LowBitError::NonFiniteInput { index: 2, .. })
        ));
        assert!(matches!(
            LowBitTensor::quantize(&[f32::INFINITY], config),
            Err(LowBitError::NonFiniteInput { index: 0, .. })
        ));
    }

    #[test]
    fn partial_last_group_has_its_own_scale() {
        // Groups of 2 over 5 values: [1, 2], [3, 4], [-5].
        let config = symmetric(4, 2);
        let tensor = LowBitTensor::quantize(&[1.0, 2.0, 3.0, 4.0, -5.0], config).unwrap();
        assert_eq!(tensor.scales().len(), 3);
        assert!((tensor.scales()[2] - 5.0 / 7.0).abs() < 1e-7);
        let values = tensor.dequantize().unwrap();
        assert_eq!(values.len(), 5);
        assert!((values[4] + 5.0).abs() < 1e-6);
    }

    #[test]
    fn reconstruction_error_is_bounded_by_half_a_scale() {
        let mut random = Lcg(3);
        for bits in 2..=MAX_BITS {
            for group_size in [1_usize, 2, 7, 32, 128] {
                let values: Vec<f32> = (0..517).map(|_| random.unit() * 4.0).collect();
                let config = symmetric(bits, group_size);
                let tensor = LowBitTensor::quantize(&values, config).unwrap();
                let decoded = tensor.dequantize().unwrap();
                for (index, (&original, &reconstructed)) in values.iter().zip(&decoded).enumerate()
                {
                    let scale = tensor.scales()[index / group_size];
                    let bound = scale / 2.0 * (1.0 + 1e-6);
                    assert!(
                        (original - reconstructed).abs() <= bound,
                        "bits {bits} group {group_size} index {index}: {original} vs {reconstructed}, scale {scale}"
                    );
                }
            }
        }
    }

    #[test]
    fn four_bit_output_is_bit_identical_to_the_int4_reference() {
        let mut random = Lcg(19);
        for group_size in [1_usize, 2, 3, 16, 128, 130] {
            for length in [0_usize, 1, 2, 127, 128, 129, 1000] {
                let values: Vec<f32> = (0..length).map(|_| random.unit() * 9.0).collect();
                let lowbit = LowBitTensor::quantize(&values, symmetric(4, group_size)).unwrap();
                let reference = int4::quantize(&values, group_size).unwrap();
                assert_eq!(
                    lowbit.packed(),
                    reference.packed(),
                    "group {group_size}, length {length}"
                );
                assert_eq!(
                    lowbit.scales(),
                    reference.scales(),
                    "group {group_size}, length {length}"
                );
            }
        }
    }

    #[test]
    fn corrupt_parts_are_refused() {
        let config = symmetric(3, 4);
        assert!(matches!(
            LowBitTensor::from_parts(config, 8, vec![0; 2], vec![1.0, 1.0]),
            Err(LowBitError::PackedLengthMismatch { .. })
        ));
        assert!(matches!(
            LowBitTensor::from_parts(config, 8, vec![0; 3], vec![1.0]),
            Err(LowBitError::ScaleCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
        // Two 3-bit values fit in one byte. The stored code 0b100 is the
        // reserved minimum -4, which must fail at decode time, not decode silently.
        let tensor = LowBitTensor::from_parts(config, 2, vec![0b100], vec![1.0]).unwrap();
        assert_eq!(
            tensor.dequantize(),
            Err(LowBitError::ReservedCode { index: 0 })
        );
    }
}
