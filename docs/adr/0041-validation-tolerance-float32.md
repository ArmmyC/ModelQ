# ADR 0041: The Scale-Bound Check Allows for Float32 Rounding

- Status: Accepted (Task 63).
- Date: 2026-10-11
- Scope: the validator's check that a decoded value lies within half a group scale of its source value. The check allowed a relative excess of 1e-6. It now allows the excess that float32 rounding can produce for symmetric codes, `4 * qmax * 2^-24` (`lowbit::half_scale_tolerance`). The quantizer and its codes are unchanged.
- Builds on: [ADR 0030](0030-group-wise-low-bit-formats.md) (the low-bit formats) and [ADR 0038](0038-awq-sharded-checkpoints.md) (the Qwen2.5-3B validation that reported one value beyond half a scale).

## Context

The Qwen2.5-3B AWQ INT4 validation reported one value beyond half a group scale, although the report said that symmetric rounding allows none. The 0.5B runs reported none.

A diagnostic traced the value to tensor `model.layers.19.mlp.up_proj.weight`. Its exact quotient by the group scale is 5.4999998, just below the rounding tie at 5.5. The quantizer divides in `f32`, which rounds that quotient to exactly 5.5. Rounding ties away from zero then picks code 6, which lies 0.5000002 scales from the value, while the nearest code is 5. The error therefore exceeds half a scale by 1.1e-6 relative, which is above the validator's 1e-6 tolerance but within float32 precision.

## Decisions

1. **The quantizer is unchanged.** Its codes are the reference codes of ADR 0030, and the existing anchors depend on them. Dividing in `f64` would pick code 5 for this value, but it would change codes at other near-ties too, so it is a separate decision.
2. **The tolerance follows from the arithmetic.** The quotient carries a relative error of `u = 2^-24` on a magnitude of at most `qmax`, so its error is at most `qmax * u` in units of the scale. The decoded product adds at most `qmax * u` more. An error can therefore exceed half a scale by at most `4 * qmax * u` of that half. For 4-bit symmetric codes (`qmax = 7`) this is 1.67e-6. The flagged value's excess of 1.10e-6 lies inside it.
3. **One shared bound.** `lowbit::half_scale_tolerance(config)` is used by the validator, by the quantizer's reconstruction test (which had the same 1e-6 bound), and by the report. For 8-bit codes the bound is larger, which the old fixed 1e-6 did not allow for.
4. **The report says what it counts.** It reads "values beyond half a group scale plus float32 rounding", and symmetric rounding still allows none beyond that.

## Verification

- **Regression test.** `a_rounding_tie_from_float32_division_is_within_the_tolerance` reproduces the flagged value's arithmetic from its exact `f32` bit patterns: the quotient is 5.5, the code is 6, the error exceeds half a scale, and it lies within the tolerance.
- **Local.** The quantizer's library tests pass (115), the I/O library tests pass (31), and the validator's integration tests pass (7). `cargo fmt --check` and clippy with `-D warnings` are clean.
- **Linux suite (Modal, Rust 1.85).** 376 passed, 0 failed, 2 ignored (375 before this change; the one new test is the regression test).
- **The 3B value.** Its measured excess of 1.10e-6 lies inside the new bound of 1.67e-6, so the 3B validation should now count zero. This follows from the measured excess. The 3B quantization has not been run again to confirm it.
- **The 0.5B runs.** They count zero before and after. A looser tolerance can only lower a count.

## What this does not show

- Whether dividing in `f64` would be better overall. It would fix this value, but it would change codes at other near-ties, and every anchor would need rechecking.
- The 3B quality loss of about 50% perplexity. That is a separate question from the validator.

## Consequences

- The validator counts only excesses beyond float32 rounding. A nonzero count now indicates a real error in the quantizer.
- The bound grows with `qmax`, so 8-bit codes get a larger tolerance than 4-bit codes.
