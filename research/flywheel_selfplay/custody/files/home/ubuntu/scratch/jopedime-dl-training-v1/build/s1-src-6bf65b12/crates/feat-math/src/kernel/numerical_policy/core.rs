//! Pure Rust helper body shared by the runtime and emitted Rust. The permanent
//! fault bit assignments are checked against FaultSet by the B07 tests.
const NUMERICAL_MAX_V2: f64 = 1.0e12;
const NUMERICAL_NONFINITE_INPUT_V2: u32 = 1 << 13;
const NUMERICAL_EMPTY_ROW_V2: u32 = 1 << 14;
const NUMERICAL_CLAMP_V2: u32 = 1 << 9;
const NUMERICAL_SANITIZED_V2: u32 = 1 << 8;

fn numerical_finish_v2(value: f64) -> (f64, u32) {
    if !value.is_finite() {
        (0.0, NUMERICAL_SANITIZED_V2)
    } else if value > NUMERICAL_MAX_V2 {
        (NUMERICAL_MAX_V2, NUMERICAL_CLAMP_V2)
    } else if value < -NUMERICAL_MAX_V2 {
        (-NUMERICAL_MAX_V2, NUMERICAL_CLAMP_V2)
    } else {
        (value, 0)
    }
}

// Mathematical branches from Netlib fdlibm's asinh method. This is a separate
// policy implementation; the old overflowing formula is deliberately retained.
pub fn numerical_asinh_v2(input: f64) -> (f64, u32) {
    if !input.is_finite() {
        return (0.0, NUMERICAL_NONFINITE_INPUT_V2);
    }
    let a = input.abs();
    let value = if a < f64::from_bits(0x3e30_0000_0000_0000) {
        return (input, 0); // 2^-28, including ±0 and subnormals.
    } else if a > f64::from_bits(0x41b0_0000_0000_0000) {
        a.ln() + std::f64::consts::LN_2
    } else if a > 2.0 {
        (2.0 * a + 1.0 / ((a * a + 1.0).sqrt() + a)).ln()
    } else {
        let square = a * a;
        (a + square / (1.0 + (1.0 + square).sqrt())).ln_1p()
    };
    numerical_finish_v2(value.copysign(input))
}

fn numerical_neumaier_v2(row: &[f64], absolute: bool) -> f64 {
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for &input in row {
        let value = if absolute { input.abs() } else { input };
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}

// Ordered error-free FastTwoSum expansion. Every low part is retained instead
// of accumulating all rounding errors into one lossy correction term.
fn numerical_add_partial_v2(partials: &mut Vec<f64>, mut value: f64) {
    let mut kept = 0;
    for index in 0..partials.len() {
        let mut part = partials[index];
        if value.abs() < part.abs() {
            std::mem::swap(&mut value, &mut part);
        }
        let high = value + part;
        let low = part - (high - value);
        if low != 0.0 {
            partials[kept] = low;
            kept += 1;
        }
        value = high;
    }
    partials.truncate(kept);
    if value != 0.0 {
        partials.push(value);
    }
}

fn numerical_expansion_v2(row: &[f64], absolute: bool, mean: bool) -> (f64, u32) {
    let count = row.len() as f64;
    let divisor = if mean { count } else { 1.0 };
    let mut scale = 2.0;
    while scale < count * 2.0 {
        scale *= 2.0;
    }
    let mut scaled = Vec::new();
    for &input in row {
        let value = if absolute { input.abs() } else { input };
        numerical_add_partial_v2(&mut scaled, value / scale);
    }
    // Cancellation can leave weakly overlapping parts. Re-add in magnitude
    // order before using the leading component for the conservative guard.
    let mut ordered = Vec::new();
    for part in scaled {
        numerical_add_partial_v2(&mut ordered, part);
    }
    let scaled = ordered;
    let head = scaled.last().copied().unwrap_or(0.0);
    // The factor-two margin makes this guard independent of final rounding.
    // Far-out means clamp before any potentially overflowing restoration.
    if head.abs() > 2.0 * NUMERICAL_MAX_V2 * divisor / scale {
        return (NUMERICAL_MAX_V2.copysign(head), NUMERICAL_CLAMP_V2);
    }
    let mut restored = Vec::new();
    for part in scaled {
        numerical_add_partial_v2(&mut restored, part * scale);
    }
    for &input in row {
        let value = if absolute { input.abs() } else { input };
        numerical_add_partial_v2(&mut restored, value - (value / scale) * scale);
    }
    let head = restored.last().copied().unwrap_or(0.0);
    if head == 0.0 {
        return (0.0, 0);
    }
    // Compare the exact expansion against +/-1e12*n, including boundary tails.
    // 13-bit integer chunks times 1e12 fit the 53-bit significand exactly.
    let sign = head.signum();
    let mut distance: Vec<_> = restored.iter().map(|part| part * sign).collect();
    let mut remaining = if mean { row.len() } else { 1 };
    let mut place = 1.0;
    while remaining != 0 {
        let chunk = (remaining & 0x1fff) as f64;
        numerical_add_partial_v2(&mut distance, -(chunk * NUMERICAL_MAX_V2) * place);
        remaining >>= 13;
        place *= 8192.0;
    }
    if distance.last().copied().unwrap_or(0.0) > 0.0 {
        return (NUMERICAL_MAX_V2.copysign(head), NUMERICAL_CLAMP_V2);
    }
    // The leading expansion component is a faithful numerator approximation;
    // one final division supplies the bounded mean. Rounding cannot add a fault.
    (
        (head / divisor).clamp(-NUMERICAL_MAX_V2, NUMERICAL_MAX_V2),
        0,
    )
}

fn numerical_mean_v2(row: &[f64]) -> (f64, u32) {
    numerical_expansion_v2(row, false, true)
}

// Positive dyadic squares use 26-bit digits, in units of 2^-2148. Inputs
// above the output limit refuse before accumulation. Before each comparison
// the total is at most twice 1e24, so 86 digits leave seven spare high bits.
type NumericalSquaresV2 = [u64; 86];
const NUMERICAL_DIGIT_MASK_V2: u64 = (1 << 26) - 1;

fn numerical_add_word_v2(bins: &mut NumericalSquaresV2, mut word: u64, mut offset: usize) {
    while word != 0 {
        let mut carry = (word & NUMERICAL_DIGIT_MASK_V2) << (offset % 26);
        let mut index = offset / 26;
        while carry != 0 {
            let total = bins[index] + carry;
            bins[index] = total & NUMERICAL_DIGIT_MASK_V2;
            carry = total >> 26;
            index += 1;
        }
        word >>= 26;
        offset += 26;
    }
}

fn numerical_add_square_v2(bins: &mut NumericalSquaresV2, value: f64) {
    let bits = value.to_bits() & 0x7fff_ffff_ffff_ffff;
    let exponent = ((bits >> 52) & 0x7ff) as usize;
    let mantissa = (bits & ((1 << 52) - 1)) | if exponent == 0 { 0 } else { 1 << 52 };
    let low = mantissa & NUMERICAL_DIGIT_MASK_V2;
    let high = mantissa >> 26;
    let offset = if exponent == 0 { 0 } else { 2 * exponent - 2 };
    numerical_add_word_v2(bins, low * low, offset);
    numerical_add_word_v2(bins, 2 * low * high, offset + 26);
    numerical_add_word_v2(bins, high * high, offset + 52);
}

fn numerical_square_bit_v2(bins: &NumericalSquaresV2, bit: usize) -> u64 {
    (bins[bit / 26] >> (bit % 26)) & 1
}

fn numerical_square_sticky_v2(bins: &NumericalSquaresV2, cutoff: usize) -> bool {
    bins.iter().enumerate().any(|(index, &word)| {
        let start = index * 26;
        start < cutoff && word & ((1 << (cutoff - start).min(26)) - 1) != 0
    })
}

fn numerical_square_root_v2(bins: &NumericalSquaresV2) -> f64 {
    let Some(index) = bins.iter().rposition(|&word| word != 0) else {
        return 0.0;
    };
    let top_bit = index * 26 + 63 - bins[index].leading_zeros() as usize;
    let mut leading = 0_u64;
    for position in 0..53 {
        leading <<= 1;
        if position <= top_bit {
            leading |= numerical_square_bit_v2(bins, top_bit - position);
        }
    }
    if top_bit > 52 {
        let discarded = top_bit - 52;
        if numerical_square_bit_v2(bins, discarded - 1) != 0
            && (leading & 1 != 0 || numerical_square_sticky_v2(bins, discarded - 1))
        {
            leading += 1;
        }
    }
    let normalized = leading as f64
        * if top_bit.is_multiple_of(2) {
            2.220446049250313e-16
        } else {
            4.440892098500626e-16
        };
    let exponent = top_bit as i32 / 2 - 1074;
    let scale = if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (exponent + 1074))
    };
    normalized.sqrt() * scale
}

fn numerical_l2_v2(row: &[f64]) -> (f64, u32) {
    let mut threshold = [0; 86];
    numerical_add_square_v2(&mut threshold, NUMERICAL_MAX_V2);
    let mut squares = [0; 86];
    for &input in row {
        if input.abs() > NUMERICAL_MAX_V2 {
            return (NUMERICAL_MAX_V2, NUMERICAL_CLAMP_V2);
        }
        numerical_add_square_v2(&mut squares, input);
        if squares.iter().rev().cmp(threshold.iter().rev()).is_gt() {
            return (NUMERICAL_MAX_V2, NUMERICAL_CLAMP_V2);
        }
    }
    // The exact dyadic comparison owns the fault; final sqrt rounding does not.
    (numerical_square_root_v2(&squares).min(NUMERICAL_MAX_V2), 0)
}

pub fn numerical_reduce_v2(kind: &str, row: &[f64]) -> Result<(f64, u32), &'static str> {
    if !matches!(kind, "sum" | "l1" | "mean" | "l2") {
        return Err("unsupported numerical reduction");
    }
    if row.iter().any(|value| !value.is_finite()) {
        return Ok((0.0, NUMERICAL_NONFINITE_INPUT_V2));
    }
    if row.is_empty() {
        return Ok((
            0.0,
            if kind == "mean" {
                NUMERICAL_EMPTY_ROW_V2
            } else {
                0
            },
        ));
    }
    if kind == "mean" {
        return Ok(numerical_mean_v2(row));
    }
    if kind == "l2" {
        return Ok(numerical_l2_v2(row));
    }
    // Preserve the explicitly authorized unrepresentable ordered-sum refusal.
    if !numerical_neumaier_v2(row, kind == "l1").is_finite() {
        return Ok((0.0, NUMERICAL_SANITIZED_V2));
    }
    Ok(numerical_expansion_v2(row, kind == "l1", false))
}
