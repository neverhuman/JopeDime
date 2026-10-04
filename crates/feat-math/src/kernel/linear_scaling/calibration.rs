// Ordered binary64 arithmetic shared by native fitting and the Rust recipe.
fn scaling_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total: f64 = 0.0;
    let mut residual = 0.0;
    for value in values {
        let next = total + value;
        residual += if total.abs() >= value.abs() {
            (total - next) + value
        } else {
            (value - next) + total
        };
        total = next;
    }
    total + residual
}

fn scaling_mean(values: &[f64]) -> f64 {
    if values.iter().all(|v| *v == values[0]) {
        return values[0];
    }
    let total = scaling_sum(values.iter().copied());
    if total.is_finite() {
        return total / values.len() as f64;
    }
    let scale = values.iter().fold(0.0_f64, |scale, v| scale.max(v.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    (scaling_sum(values.iter().map(|v| v / scale)) / values.len() as f64) * scale
}

pub(super) fn calibrate(values: &[f64], targets: &[f64]) -> Result<([f64; 2], f64), &'static str> {
    let (mut a, mut b) = (0.0, scaling_mean(targets));
    if !values.iter().all(|v| *v == values[0]) && !targets.iter().all(|v| *v == targets[0]) {
        let (fa, fe, fm, df) = scaling_centered(values);
        let (ya, ye, ym, dy) = scaling_centered(targets);
        let variance = scaling_sum(df.iter().map(|d| d * d));
        let covariance = scaling_sum(df.iter().zip(dy).map(|(f, y)| f * y));
        if variance <= 0.0 || !variance.is_finite() {
            return Err("linear-scaling variance not representable");
        }
        a = scaling_rescale(covariance / variance, ye - fe);
        if a == 0.0 && covariance != 0.0 {
            return Err("linear-scaling coefficients");
        }
        b = (ya - a * fa) + (scaling_rescale(ym, ye) - a * scaling_rescale(fm, fe));
    }
    if !a.is_finite() || !b.is_finite() {
        return Err("linear-scaling coefficients");
    }
    let mut terms = Vec::with_capacity(values.len());
    for (f, y) in values.iter().zip(targets) {
        let prediction = a * f + b;
        if !prediction.is_finite() {
            return Err("linear-scaled prediction");
        }
        let difference = prediction - y;
        if !difference.is_finite() {
            return Err("linear-scaling training loss");
        }
        terms.push(difference);
    }
    let magnitude = terms.iter().fold(0.0_f64, |s, d| s.max(d.abs()));
    if magnitude == 0.0 {
        return Ok(([a, b], 0.0));
    }
    let exponent = scaling_exponent(magnitude);
    let scale = scaling_power(exponent);
    let normalized = scaling_sum(terms.iter().map(|d| {
        let d = d / scale;
        d * d
    })) / values.len() as f64;
    let mse = scaling_rescale(normalized, 2 * exponent);
    if !mse.is_finite() || mse < 0.0 || (normalized > 0.0 && mse == 0.0) {
        return Err("linear-scaling training loss");
    }
    Ok(([a, b], mse))
}

// Translation preserves adjacent inputs (Sterbenz-exact within a binade);
// power-of-two normalization avoids rounding away variation at a large offset.
fn scaling_centered(values: &[f64]) -> (f64, i32, f64, Vec<f64>) {
    let mut anchor = values[0];
    let mut offsets: Vec<_> = values.iter().map(|v| v - anchor).collect();
    if offsets.iter().any(|v| !v.is_finite()) {
        anchor = 0.0;
        offsets.clone_from_slice(values);
    }
    let magnitude = offsets.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
    let exponent = if magnitude == 0.0 {
        0
    } else {
        scaling_exponent(magnitude)
    };
    let scale = scaling_power(exponent);
    let normalized: Vec<_> = offsets.iter().map(|v| v / scale).collect();
    let mean = scaling_sum(normalized.iter().copied()) / values.len() as f64;
    (
        anchor,
        exponent,
        mean,
        normalized.iter().map(|v| v - mean).collect(),
    )
}

fn scaling_exponent(value: f64) -> i32 {
    let word = value.to_bits();
    let biased = (word >> 52) as i32;
    if biased == 0 {
        63 - word.leading_zeros() as i32 - 1074
    } else {
        biased - 1023
    }
}

fn scaling_power(exponent: i32) -> f64 {
    f64::from_bits(if exponent >= -1022 {
        ((exponent + 1023) as u64) << 52
    } else {
        1_u64 << (exponent + 1074)
    })
}

// Equivalent to ldexp, without an overflowing intermediate scale. All normal
// power-of-two steps are exact; only the final subnormal step can round.
fn scaling_rescale(mut value: f64, mut exponent: i32) -> f64 {
    while exponent > 512 {
        value *= scaling_power(512);
        exponent -= 512;
    }
    while exponent < -512 {
        value *= scaling_power(-512);
        exponent += 512;
    }
    value * scaling_power(exponent)
}
