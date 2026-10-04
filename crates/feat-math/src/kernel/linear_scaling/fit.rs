use super::{check_rows, validate, Error, LinearScaledWinner, LinearScalingFit, MathExpr, Result};

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
    let ([a, b], training_mse) = super::calibration::calibrate(&values, targets)
        .map_err(|message| Error::NonFinite(message.into()))?;
    winner.slope = a;
    winner.intercept = b;
    validate(&winner)?;
    Ok(LinearScalingFit {
        winner,
        training_mse,
    })
}
