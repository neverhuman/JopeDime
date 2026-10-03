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

fn numerical_neumaier_v2(row: &[f64], absolute: bool, divisor: f64) -> f64 {
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for &input in row {
        let value = if absolute { input.abs() } else { input } / divisor;
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
    let mut value = numerical_neumaier_v2(row, kind == "l1", 1.0);
    if kind == "mean" {
        let count = row.len() as f64;
        value = if value.is_finite() {
            value / count
        } else {
            // A finite mean can exist even when the unscaled sum overflows.
            numerical_neumaier_v2(row, false, count)
        };
    }
    Ok(numerical_finish_v2(value))
}
