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
    let fm = scaling_mean(values);
    let ym = scaling_mean(targets);
    let (mut a, mut b) = (0.0, ym);
    if !values.iter().all(|v| *v == values[0]) {
        let fs = values.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        let ys = targets.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        if ys == 0.0 {
            b = 0.0;
        } else {
            let df: Vec<_> = values.iter().map(|v| v / fs - fm / fs).collect();
            let dy: Vec<_> = targets.iter().map(|v| v / ys - ym / ys).collect();
            let variance = scaling_sum(df.iter().map(|d| d * d));
            let covariance = scaling_sum(df.iter().zip(dy).map(|(f, y)| f * y));
            if variance <= 0.0 || !variance.is_finite() {
                return Err("linear-scaling variance not representable");
            }
            a = ((covariance / variance) * ys) / fs;
            b = ym - a * fm;
        }
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
        terms.push(difference * difference / values.len() as f64);
    }
    let mse = scaling_sum(terms.into_iter());
    if !mse.is_finite() || mse < 0.0 {
        return Err("linear-scaling training loss");
    }
    Ok(([a, b], mse))
}
