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

fn numerical_mean_v2(row: &[f64]) -> (f64, u32) {
    let count = row.len() as f64;
    let mut scale = 2.0;
    while scale < count * 2.0 {
        scale *= 2.0;
    }
    let mut scaled = Vec::new();
    for &value in row {
        numerical_add_partial_v2(&mut scaled, value / scale);
    }
    let head = scaled.last().copied().unwrap_or(0.0);
    // The factor-two margin makes this guard independent of final rounding.
    // Far-out means clamp before any potentially overflowing restoration.
    if head.abs() > 2.0 * NUMERICAL_MAX_V2 * count / scale {
        return (NUMERICAL_MAX_V2.copysign(head), NUMERICAL_CLAMP_V2);
    }
    let mut restored = Vec::new();
    for part in scaled {
        numerical_add_partial_v2(&mut restored, part * scale);
    }
    for &value in row {
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
    let mut remaining = row.len();
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
    ((head / count).clamp(-NUMERICAL_MAX_V2, NUMERICAL_MAX_V2), 0)
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
        let mut scale = 0.0_f64;
        let mut squares = 1.0_f64;
        for &input in row {
            let a = input.abs();
            if a == 0.0 {
                continue;
            }
            if scale < a {
                let ratio = scale / a;
                squares = 1.0 + squares * ratio * ratio;
                scale = a;
            } else {
                let ratio = a / scale;
                squares += ratio * ratio;
            }
        }
        let root = squares.sqrt();
        // Check saturation before multiplying, including a norm above f64::MAX.
        if scale > NUMERICAL_MAX_V2 / root {
            return Ok((NUMERICAL_MAX_V2, NUMERICAL_CLAMP_V2));
        }
        return Ok(numerical_finish_v2(scale * root));
    }
    Ok(numerical_finish_v2(numerical_neumaier_v2(
        row,
        kind == "l1",
    )))
}
