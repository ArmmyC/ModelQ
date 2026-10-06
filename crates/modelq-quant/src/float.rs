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
    use super::CodecError;

    /// Maximum finite magnitude represented by E2M1.
    pub const MAX_FINITE: f32 = 6.0;
    /// Number of bits in one E2M1 element.
    pub const BITS: u8 = 4;

    /// Decodes the low four bits of an E2M1 element.
    ///
    /// The sixteen bit patterns represent signed zero, subnormal `±0.5`, and
    /// the normal magnitudes `±1`, `±1.5`, `±2`, `±3`, `±4`, and `±6`.
    pub fn decode(bits: u8) -> f32 {
        super::E2M1_TABLE[usize::from(bits & 0x0f)]
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
        Ok(encode_unchecked(value))
    }

    /// [`encode`] without the NaN check, for hot loops that test a whole block
    /// for NaN once.  NaN encodes to an unspecified code; every other value
    /// matches [`encode`] exactly.
    #[inline]
    pub(crate) fn encode_unchecked(value: f32) -> u8 {
        let sign = if value.is_sign_negative() { 0x08 } else { 0 };
        let magnitude = value.abs();
        // The representable magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6.  Each
        // boundary below is the midpoint of two neighbors; a tie goes to the
        // neighbor with even bits, so the lower code is kept (`>`) when it is
        // even and the upper code is taken (`>=`) when the upper is even.
        // Infinity passes every comparison and saturates to code 7.
        let index = u8::from(magnitude > 0.25)
            + u8::from(magnitude >= 0.75)
            + u8::from(magnitude > 1.25)
            + u8::from(magnitude >= 1.75)
            + u8::from(magnitude > 2.5)
            + u8::from(magnitude >= 3.5)
            + u8::from(magnitude > 5.0);
        sign | index
    }
}

/// Eight-bit E4M3 finite/NaN values.
pub mod fp8_e4m3 {
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
        super::E4M3_TABLE[usize::from(bits)]
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
        let magnitude = value.abs();
        // Infinity and everything at or beyond the largest finite value
        // saturate; code 0x7f is NaN and is never produced for a number.
        if magnitude >= MAX_FINITE {
            return sign | 0x7e;
        }
        // Below 448 the grid in the binade `[2^e, 2^(e+1))` has a spacing of
        // `2^(e-3)`, and the subnormal range (`e < -6`) shares the `e = -6`
        // spacing of `2^-9`.  Scaling by that power of two is exact, so
        // rounding the quotient to an even integer is round-to-nearest-even
        // on the grid.  The integer's parity equals the code's low bit, which
        // is also how ties are broken between binades, and a carry into the
        // next binade lands on the correct next code.
        let exponent = ((magnitude.to_bits() >> 23) as i32 - 127).max(-6);
        let spacing = f32::from_bits(((exponent - 3 + 127) as u32) << 23);
        let steps = (magnitude / spacing).round_ties_even() as i32;
        let code = (exponent + 6) * 8 + steps;
        sign | code.min(0x7e) as u8
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
        // Everything at or beyond the largest finite value saturates.  The
        // candidate search below cannot decide this for huge magnitudes,
        // where `magnitude - candidate` rounds to the same F32 for every
        // candidate.
        if value.abs() >= MAX_FINITE {
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

/// Decoded E2M1 values, indexed by the four-bit pattern.
const E2M1_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decoded E4M3 values, indexed by the eight-bit pattern.
const E4M3_TABLE: [f32; 256] = build_e4m3_table();

/// Builds the E4M3 table from exact F32 bit patterns, so no floating-point
/// arithmetic or `powi` is involved.
const fn build_e4m3_table() -> [f32; 256] {
    let mut table = [0.0_f32; 256];
    let mut bits = 0_usize;
    while bits < 256 {
        let negative = bits & 0x80 != 0;
        let exponent = (bits >> 3) & 0x0f;
        let mantissa = (bits & 0x07) as u32;
        let magnitude_bits: u32 = if exponent == 0x0f && mantissa == 0x07 {
            f32::NAN.to_bits() & 0x7fff_ffff
        } else if exponent == 0 {
            if mantissa == 0 {
                0
            } else {
                // mantissa * 2^-9 with mantissa in 1..=7: normalize it.
                let top = 31 - mantissa.leading_zeros();
                let fraction = (mantissa - (1 << top)) << (23 - top);
                ((127 + top - 9) << 23) | fraction
            }
        } else {
            // (1 + mantissa / 8) * 2^(exponent - 7)
            (((exponent as u32) + 120) << 23) | (mantissa << 20)
        };
        let sign_bit = if negative { 0x8000_0000 } else { 0 };
        table[bits] = f32::from_bits(sign_bit | magnitude_bits);
        bits += 1;
    }
    table
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
/// The original candidate-search codecs, kept verbatim as the oracle for
/// the closed-form encoders and table decoders.
pub(crate) mod reference {
    use super::{CodecError, nearest_finite};

    pub(crate) fn decode_e2m1(bits: u8) -> f32 {
        let magnitude = match bits & 0x07 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        if bits & 0x08 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    pub(crate) fn decode_e4m3(bits: u8) -> f32 {
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

    pub(crate) fn encode_e2m1(value: f32) -> Result<u8, CodecError> {
        if value.is_nan() {
            return Err(CodecError::NaNNotRepresentable);
        }
        let sign = if value.is_sign_negative() { 0x08 } else { 0 };
        let magnitude = value.abs();
        if magnitude.is_infinite() {
            return Ok(sign | 0x07);
        }
        Ok(sign | nearest_finite(magnitude, 0..=0x07, decode_e2m1))
    }

    pub(crate) fn decode_e5m2(bits: u8) -> f32 {
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

    pub(crate) fn encode_e5m2(value: f32) -> u8 {
        if value.is_nan() {
            return 0x7f;
        }
        let sign = if value.is_sign_negative() { 0x80 } else { 0 };
        if value.is_infinite() {
            return sign | 0x7b;
        }
        sign | nearest_finite(value.abs(), 0..=0x7b, decode_e5m2)
    }

    pub(crate) fn encode_e4m3(value: f32) -> u8 {
        if value.is_nan() {
            return 0x7f;
        }
        let sign = if value.is_sign_negative() { 0x80 } else { 0 };
        if value.is_infinite() {
            return sign | 0x7e;
        }
        sign | nearest_finite(value.abs(), 0..=0x7e, decode_e4m3)
    }
}

#[cfg(test)]
mod tests {
    use super::{fp4_e2m1, fp8_e4m3, fp8_e5m2, reference};

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

    /// Magnitudes at or above this are outside the original search codecs'
    /// reliable range: `magnitude - candidate` rounds to the same F32 for every
    /// candidate, every candidate ties, and the first (zero) wins, which
    /// contradicts the documented saturation.  Below it the original is exact.
    const SEARCH_RELIABLE_BELOW: f32 = 1_048_576.0;

    fn assert_encoders_match(value: f32) {
        let label = format!("{value:e} ({:#010x})", value.to_bits());
        if value.is_nan() || value.abs() < SEARCH_RELIABLE_BELOW {
            assert_eq!(
                fp4_e2m1::encode(value),
                reference::encode_e2m1(value),
                "FP4 encode of {label}"
            );
            assert_eq!(
                fp8_e4m3::encode(value),
                reference::encode_e4m3(value),
                "E4M3 encode of {label}"
            );
            assert_eq!(
                fp8_e5m2::encode(value),
                reference::encode_e5m2(value),
                "E5M2 encode of {label}"
            );
        } else {
            let negative = value.is_sign_negative();
            assert_eq!(
                fp4_e2m1::encode(value),
                Ok(if negative { 0x0f } else { 0x07 }),
                "FP4 saturation of {label}"
            );
            assert_eq!(
                fp8_e4m3::encode(value),
                if negative { 0xfe } else { 0x7e },
                "E4M3 saturation of {label}"
            );
            assert_eq!(
                fp8_e5m2::encode(value),
                if negative { 0xfb } else { 0x7b },
                "E5M2 saturation of {label}"
            );
        }
    }

    #[test]
    fn decode_tables_match_the_original_decoders_bit_for_bit() {
        for bits in 0..=0x0f_u8 {
            assert_eq!(
                fp4_e2m1::decode(bits).to_bits(),
                reference::decode_e2m1(bits).to_bits(),
                "E2M1 {bits:#x}"
            );
        }
        for bits in u8::MIN..=u8::MAX {
            let (new, old) = (fp8_e4m3::decode(bits), reference::decode_e4m3(bits));
            if old.is_nan() {
                assert!(new.is_nan(), "E4M3 {bits:#x}");
            } else {
                assert_eq!(new.to_bits(), old.to_bits(), "E4M3 {bits:#x}");
            }
        }
    }

    #[test]
    fn encoders_match_the_original_at_every_grid_point_and_midpoint() {
        let mut anchors = Vec::new();
        for bits in 0..=0x0f_u8 {
            anchors.push(reference::decode_e2m1(bits));
        }
        let finite_e4m3: Vec<f32> = (0..=0x7e_u8).map(reference::decode_e4m3).collect();
        anchors.extend(finite_e4m3.iter().copied());
        for pair in finite_e4m3.windows(2) {
            anchors.push(f32::midpoint(pair[0], pair[1]));
        }
        for half in [0.25_f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] {
            anchors.push(half);
        }
        anchors.extend([
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX,
            447.9,
            448.0,
            448.1,
            460.0,
            480.0,
            1.0e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ]);
        // Each anchor and its neighbors a few ulps either side, both signs.
        for anchor in anchors {
            for delta in -4_i64..=4 {
                let bits = i64::from(anchor.to_bits()) + delta;
                let Ok(bits) = u32::try_from(bits) else {
                    continue;
                };
                for value in [f32::from_bits(bits), -f32::from_bits(bits)] {
                    assert_encoders_match(value);
                }
            }
        }
    }

    #[test]
    fn encoders_match_the_original_on_dense_pseudorandom_values() {
        // Exponents from 2^-16 to 2^10 cover the subnormal, normal, and
        // saturating ranges of both formats; the full range is covered by the
        // ignored exhaustive test.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..150_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let exponent = 127 - 16 + ((state >> 33) % 27) as u32;
            let mantissa = ((state >> 8) & 0x007f_ffff) as u32;
            let sign = ((state >> 63) as u32) << 31;
            assert_encoders_match(f32::from_bits(sign | (exponent << 23) | mantissa));
        }
    }

    /// Every one of the 2^32 F32 bit patterns, across all CPUs.  Takes about
    /// a minute in release mode:
    /// `cargo test --release -p modelq-quant -- --ignored exhaustive`.
    #[test]
    #[ignore = "checks all 2^32 F32 patterns; run explicitly in release mode"]
    fn exhaustive_f32_encoders_match_the_original() {
        let threads =
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get) as u64;
        let total = 1_u64 << 32;
        let per_thread = total.div_ceil(threads);
        std::thread::scope(|scope| {
            for thread in 0..threads {
                scope.spawn(move || {
                    let end = ((thread + 1) * per_thread).min(total);
                    for bits in thread * per_thread..end {
                        assert_encoders_match(f32::from_bits(bits as u32));
                    }
                });
            }
        });
    }
}
