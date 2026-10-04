use super::{check_rows, validate, Error, LinearScaledWinner, LinearScalingFit, MathExpr, Result};

/// Ordered compensated accumulation. Inputs to covariance/variance are
/// normalized centered values in [-2,2]; finite representability is checked.
fn sum(values: impl Iterator<Item = f64>) -> f64 {
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

fn mean(values: &[f64]) -> f64 {
    if values.iter().all(|v| *v == values[0]) {
        return values[0];
    }
    let total = sum(values.iter().copied());
    if total.is_finite() {
        return total / values.len() as f64;
    }
    let scale = values.iter().fold(0.0_f64, |scale, v| scale.max(v.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    (sum(values.iter().map(|v| v / scale)) / values.len() as f64) * scale
}

pub(super) fn fit_training(
    expression: &MathExpr,
    rows: &[Vec<f64>],
    targets: &[f64],
) -> Result<LinearScalingFit> {
    if rows.is_empty() || rows.len() != targets.len() {
        return Err(Error::Shape(
            "linear-scaling needs aligned nonempty training data".into(),
        ));
    }
    if targets.iter().any(|y| !y.is_finite()) {
        return Err(Error::NonFinite("linear-scaling target".into()));
    }
    let mut winner = LinearScaledWinner {
        expression: expression.clone(),
        input_width: rows[0].len(),
        slope: 0.0,
        intercept: 0.0,
    };
    validate(&winner)?;
    check_rows(rows, winner.input_width)?;
    let compiled = expression
        .compile()
        .map_err(|e| Error::Domain(e.to_string()))?;
    let values = rows
        .iter()
        .map(|row| {
            compiled
                .eval_row(row)
                .map_err(|e| Error::Domain(e.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let fm = mean(&values);
    let ym = mean(targets);
    // Exactly constant, including a signed-zero pair: no epsilon threshold
    // silently turns a small but meaningful feature into a constant.
    if values.iter().all(|v| *v == values[0]) {
        winner.intercept = ym;
    } else {
        let fs = values.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        let ys = targets.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        if ys == 0.0 {
            winner.intercept = 0.0;
        } else {
            let df: Vec<_> = values.iter().map(|v| v / fs - fm / fs).collect();
            let dy: Vec<_> = targets.iter().map(|v| v / ys - ym / ys).collect();
            let variance = sum(df.iter().map(|d| d * d));
            let covariance = sum(df.iter().zip(dy).map(|(f, y)| f * y));
            if variance <= 0.0 || !variance.is_finite() {
                return Err(Error::NonFinite(
                    "linear-scaling variance not representable".into(),
                ));
            }
            winner.slope = ((covariance / variance) * ys) / fs;
            winner.intercept = ym - winner.slope * fm;
        }
    }
    validate(&winner)?;
    let predictions = winner.predict(rows)?;
    let training_mse = sum(predictions.iter().zip(targets).map(|(p, y)| {
        let difference = p - y;
        difference * difference / rows.len() as f64
    }));
    if !training_mse.is_finite() || training_mse < 0.0 {
        return Err(Error::NonFinite("linear-scaling training loss".into()));
    }
    Ok(LinearScalingFit {
        winner,
        training_mse,
    })
}
