//! The training-mean baseline, scored on the same evaluation rows as the model.
//!
//! The prediction is the mean of the target on the training rows. The score is
//! RMSE on the evaluation rows. No other rows are used.

use jope_api::{Warning, WarningState};

/// RMSE of the training-set mean on `evaluation`.
///
/// `None` when either side is empty or a value is non-finite.
pub fn training_mean_rmse(training: &[f64], evaluation: &[f64]) -> Option<f64> {
    if training.is_empty() || evaluation.is_empty() {
        return None;
    }
    if training.iter().any(|value| !value.is_finite())
        || evaluation.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let mean = training.iter().sum::<f64>() / training.len() as f64;
    let square_error = evaluation
        .iter()
        .map(|value| {
            let error = value - mean;
            error * error
        })
        .sum::<f64>();
    Some((square_error / evaluation.len() as f64).sqrt())
}

/// A raised `BASELINE_MEAN` warning naming the definition and the score.
pub fn warning(
    target: &str,
    evaluation_rows: usize,
    rmse: f64,
) -> Result<Warning, crate::warnings::WarningInputError> {
    crate::warnings::catalog("BASELINE_MEAN")
        .expect("BASELINE_MEAN is catalogued")
        .event(
            WarningState::Raised,
            Some(rmse),
            None,
            format!(
                "Baseline predicts the training-set mean of `{target}`. On the same {evaluation_rows} evaluation rows its RMSE is {rmse:.6}."
            ),
            "Compare the model with this number. It uses the same evaluation rows and no others.",
        )
}

/// Finite target values for `ids` in column `index`.
pub fn target_values(
    table_rows: &[Vec<Option<String>>],
    index: usize,
    ids: &[usize],
) -> Option<Vec<f64>> {
    let mut values = Vec::with_capacity(ids.len());
    for id in ids {
        let cell = table_rows.get(*id)?.get(index)?.as_deref()?;
        let value = cell
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())?;
        values.push(value);
    }
    Some(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_training_mean_is_scored_only_on_the_evaluation_rows() {
        // Training 1 and 3, mean 2. Evaluation 2 and 4. Errors 0 and 2.
        let rmse = training_mean_rmse(&[1.0, 3.0], &[2.0, 4.0]).unwrap();
        assert!((rmse - 2.0_f64.sqrt()).abs() < 1e-12);
        let note = warning("total", 2, rmse).unwrap();
        assert_eq!(note.code, "BASELINE_MEAN");
        assert_eq!(note.value, Some(rmse));
        assert!(note.message.contains("training-set mean of `total`"));
        assert!(note.message.contains("same 2 evaluation rows"));
    }

    #[test]
    fn an_empty_side_has_no_baseline() {
        assert!(training_mean_rmse(&[], &[1.0]).is_none());
        assert!(training_mean_rmse(&[1.0], &[]).is_none());
    }
}
