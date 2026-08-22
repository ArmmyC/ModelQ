//! Reference codecs for the small floating-point formats used by later
//! ModelQ representations.
//!
//! The codecs are intentionally element-only: they do not define scaling,
//! grouping, tensor layout, or a runtime container.  Each encoder uses
//! round-to-nearest-even and saturates finite overflow to the signed maximum
//! finite value.  The exhaustive tests below make the bit-level behavior
//! auditable before a block quantizer or runtime exporter is added.

use std::fmt;

/// Error returned when a format has no representation for a source NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    /// FP4 E2M1 has no NaN encoding, so the caller must choose a policy.
    NaNNotRepresentable,
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NaNNotRepresentable => {
                formatter.write_str("the target format has no NaN encoding")
            }
        }
    }
}

impl std::error::Error for CodecError {}

/// Four-bit E2M1 values used by FP4/NVFP4-style representations.
pub mod fp4_e2m1 {
    use super::{CodecError, nearest_finite};

    /// Maximum finite magnitude represented by E2M1.
    pub const MAX_FINITE: f32 = 6.0;
    /// Number of bits in one E2M1 element.
    pub const BITS: u8 = 4;

    /// Decodes the low four bits of an E2M1 element.
    ///
    /// The sixteen bit patterns represent signed zero, subnormal `±0.5`, and
    /// the normal magnitudes `±1`, `±1.5`, `±2`, `±3`, `±4`, and `±6`.
    pub fn decode(bits: u8) -> f32 {
        super::decode_e2m1(bits & 0x0f)
    }

    /// Encodes one F32 value with round-to-nearest-even and finite saturation.
    ///
    /// Positive and negative zero preserve their sign.  E2M1 has no NaN or
    /// infinity encoding: NaN returns [`CodecError::NaNNotRepresentable`],
    /// while infinities saturate to `±6` like finite overflow.
    pub fn encode(value: f32) -> Result<u8, CodecError> {
        if value.is_nan() {
            return Err(CodecError::NaNNotRepresentable);
        }
        let sign = if value.is_sign_negative() { 0x08 } else { 0 };
        let magnitude = value.abs();
        if magnitude.is_infinite() {
            return Ok(sign | 0x07);
        }
        let bits = nearest_finite(magnitude, 0..=0x07, super::decode_e2m1);
        Ok(sign | bits)
    }
}

/// Eight-bit E4M3 finite/NaN values.
pub mod fp8_e4m3 {
    use super::nearest_finite;

    /// Maximum finite magnitude represented by E4M3.
    pub const MAX_FINITE: f32 = 448.0;
    /// Number of bits in one E4M3 element.
    pub const BITS: u8 = 8;
    const SIGN_MASK: u8 = 0x80;
    const CANONICAL_NAN: u8 = 0x7f;

    /// Decodes all E4M3 bit patterns.
    ///
    /// `0x7f` and `0xff` decode to a canonical NaN.  E4M3 has no infinity;
    /// the remaining exponent-all-ones patterns are finite through `±448`.
    pub fn decode(bits: u8) -> f32 {
        super::decode_e4m3(bits)
    }

    /// Encodes one F32 value with round-to-nearest-even and satfinite policy.
    ///
    /// NaN becomes the canonical `0x7f` NaN.  Infinities and finite overflow
    /// become signed `0x7e`/`0xfe`, the maximum finite value.
    pub fn encode(value: f32) -> u8 {
        if value.is_nan() {
            return CANONICAL_NAN;
        }
        let sign = if value.is_sign_negative() {
            SIGN_MASK
        } else {
            0
        };
        if value.is_infinite() {
            return sign | 0x7e;
        }
        sign | nearest_finite(value.abs(), 0..=0x7e, super::decode_e4m3)
    }
}

/// Eight-bit E5M2 values with IEEE-style infinities and NaNs.
pub mod fp8_e5m2 {
    use super::nearest_finite;

    /// Maximum finite magnitude represented by E5M2.
    pub const MAX_FINITE: f32 = 57_344.0;
    /// Number of bits in one E5M2 element.
    pub const BITS: u8 = 8;
    const SIGN_MASK: u8 = 0x80;
    const CANONICAL_NAN: u8 = 0x7f;

    /// Decodes all E5M2 bit patterns.
    ///
    /// Exponent `0x1f` with zero mantissa is infinity; a non-zero mantissa is
    /// NaN.  Subnormals and signed zero are preserved.
    pub fn decode(bits: u8) -> f32 {
        super::decode_e5m2(bits)
    }

    /// Encodes one F32 value with round-to-nearest-even and satfinite policy.
    ///
    /// NaN becomes canonical `0x7f`.  Infinities and finite overflow saturate
    /// to signed `0x7b`/`0xfb`, the maximum finite value, rather than emitting
    /// the E5M2 infinity encodings.
    pub fn encode(value: f32) -> u8 {
        if value.is_nan() {
            return CANONICAL_NAN;
        }
        let sign = if value.is_sign_negative() {
            SIGN_MASK
        } else {
            0
        };
        if value.is_infinite() {
            return sign | 0x7b;
        }
        sign | nearest_finite(value.abs(), 0..=0x7b, super::decode_e5m2)
    }
}

fn nearest_finite<I, F>(magnitude: f32, bits: I, decode: F) -> u8
where
    I: IntoIterator<Item = u8>,
    F: Fn(u8) -> f32,
{
    let mut best_bits = 0;
    let mut best_distance = f32::INFINITY;
    for candidate_bits in bits {
        let candidate = decode(candidate_bits);
        let distance = (magnitude - candidate).abs();
        if distance < best_distance
            || (distance == best_distance && candidate_bits & 1 == 0 && best_bits & 1 != 0)
        {
            best_bits = candidate_bits;
            best_distance = distance;
        }
    }
    best_bits
}

fn decode_e2m1(bits: u8) -> f32 {
    let magnitude = match bits & 0x07 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        7 => 6.0,
        _ => unreachable!(),
    };
    if bits & 0x08 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn decode_e4m3(bits: u8) -> f32 {
    let negative = bits & 0x80 != 0;
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        f32::from(mantissa) * 2.0_f32.powi(-9)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if negative { -magnitude } else { magnitude }
}

fn decode_e5m2(bits: u8) -> f32 {
    let negative = bits & 0x80 != 0;
    let exponent = (bits >> 2) & 0x1f;
    let mantissa = bits & 0x03;
    if exponent == 0x1f {
        return if mantissa == 0 {
            if negative {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let magnitude = if exponent == 0 {
        f32::from(mantissa) * 2.0_f32.powi(-16)
    } else {
        (1.0 + f32::from(mantissa) / 4.0) * 2.0_f32.powi(i32::from(exponent) - 15)
    };
    if negative { -magnitude } else { magnitude }
}

#[cfg(test)]
mod tests {
    use super::{fp4_e2m1, fp8_e4m3, fp8_e5m2};

    #[test]
    fn fp4_exhaustively_round_trips_all_bit_patterns() {
        for bits in 0..=0x0f {
            let decoded = fp4_e2m1::decode(bits);
            let encoded = fp4_e2m1::encode(decoded).expect("all FP4 values are encodable");
            assert_eq!(encoded, bits, "FP4 bit pattern {bits:#x}");
        }
    }

    #[test]
    fn e4m3_exhaustively_round_trips_to_canonical_patterns() {
        for bits in u8::MIN..=u8::MAX {
            let expected = if bits & 0x7f == 0x7f { 0x7f } else { bits };
            assert_eq!(
                fp8_e4m3::encode(fp8_e4m3::decode(bits)),
                expected,
                "E4M3 bit pattern {bits:#x}"
            );
        }
    }

    #[test]
    fn e5m2_exhaustively_round_trips_to_canonical_satfinite_patterns() {
        for bits in u8::MIN..=u8::MAX {
            let exponent = (bits >> 2) & 0x1f;
            let mantissa = bits & 0x03;
            let expected = match (exponent, mantissa, bits & 0x80 != 0) {
                (0x1f, 0, false) => 0x7b,
                (0x1f, 0, true) => 0xfb,
                (0x1f, _, _) => 0x7f,
                _ => bits,
            };
            assert_eq!(
                fp8_e5m2::encode(fp8_e5m2::decode(bits)),
                expected,
                "E5M2 bit pattern {bits:#x}"
            );
        }
    }

    #[test]
    fn fp4_decode_table_is_independent_of_encoding() {
        assert_eq!(fp4_e2m1::decode(0x00), 0.0);
        assert_eq!(fp4_e2m1::decode(0x01), 0.5);
        assert_eq!(fp4_e2m1::decode(0x02), 1.0);
        assert_eq!(fp4_e2m1::decode(0x03), 1.5);
        assert_eq!(fp4_e2m1::decode(0x07), 6.0);
        assert_eq!(fp4_e2m1::decode(0x0f), -6.0);
    }

    #[test]
    fn fp8_decode_tables_cover_special_values_and_limits() {
        assert_eq!(fp8_e4m3::decode(0x38), 1.0);
        assert_eq!(fp8_e4m3::decode(0x7e), fp8_e4m3::MAX_FINITE);
        assert!(fp8_e4m3::decode(0x7f).is_nan());

        assert_eq!(fp8_e5m2::decode(0x3c), 1.0);
        assert_eq!(fp8_e5m2::decode(0x7b), fp8_e5m2::MAX_FINITE);
        assert_eq!(fp8_e5m2::decode(0x7c), f32::INFINITY);
        assert!(fp8_e5m2::decode(0x7d).is_nan());
    }

    #[test]
    fn fp4_uses_nearest_even_and_saturates() {
        assert_eq!(fp4_e2m1::encode(0.25).unwrap(), 0x00);
        assert_eq!(fp4_e2m1::encode(0.75).unwrap(), 0x02);
        assert_eq!(fp4_e2m1::encode(1.75).unwrap(), 0x04);
        assert_eq!(fp4_e2m1::encode(5.0).unwrap(), 0x06);
        assert_eq!(fp4_e2m1::encode(7.0).unwrap(), 0x07);
        assert_eq!(fp4_e2m1::encode(-f32::INFINITY).unwrap(), 0x0f);
        assert_eq!(fp4_e2m1::encode(-0.0).unwrap(), 0x08);
        assert!(fp4_e2m1::encode(f32::NAN).is_err());
    }

    #[test]
    fn fp8_encoders_use_nearest_even_and_documented_saturation() {
        assert_eq!(fp8_e4m3::encode(1.0625), 0x38);
        assert_eq!(fp8_e4m3::encode(500.0), 0x7e);
        assert_eq!(fp8_e4m3::encode(f32::NEG_INFINITY), 0xfe);
        assert_eq!(fp8_e4m3::encode(f32::NAN), 0x7f);
        assert_eq!(fp8_e4m3::encode(-0.0), 0x80);

        assert_eq!(fp8_e5m2::encode(1.125), 0x3c);
        assert_eq!(fp8_e5m2::encode(70_000.0), 0x7b);
        assert_eq!(fp8_e5m2::encode(f32::NEG_INFINITY), 0xfb);
        assert_eq!(fp8_e5m2::encode(f32::NAN), 0x7f);
        assert_eq!(fp8_e5m2::encode(-0.0), 0x80);
    }
}
