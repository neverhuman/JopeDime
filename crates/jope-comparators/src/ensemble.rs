//! Caruana greedy ensemble selection, the `WeightedEnsemble_L2` step of the AutoGluon-style
//! comparator, re-homed from the jain-model-zoo `autogluon_*` ports.

use crate::error::{ComparatorError, ComparatorResult};
use crate::metrics::rmse;

/// Greedy forward selection with replacement (Caruana et al., 2004).
///
/// `member_predictions[m]` holds member `m`'s predictions for the validation `targets`. Each of
/// `rounds` steps adds the member whose inclusion gives the lowest RMSE of the running average;
/// ties go to the lowest member index. The returned weights are selection counts divided by
/// `rounds`, so they are non-negative and sum to one.
pub fn caruana_weights(
    member_predictions: &[Vec<f64>],
    targets: &[f64],
    rounds: usize,
) -> ComparatorResult<Vec<f64>> {
    if member_predictions.is_empty() {
        return Err(ComparatorError::InvalidParameter { name: "members" });
    }
    if rounds == 0 {
        return Err(ComparatorError::InvalidParameter { name: "rounds" });
    }
    for predictions in member_predictions {
        // Validates length, emptiness and finiteness once, up front.
        rmse(targets, predictions)?;
    }
    let mut counts = vec![0_usize; member_predictions.len()];
    let mut running = vec![0.0; targets.len()];
    let mut candidate = vec![0.0; targets.len()];
    for step in 0..rounds {
        let divisor = (step + 1) as f64;
        let mut best: Option<(usize, f64)> = None;
        for (member, predictions) in member_predictions.iter().enumerate() {
            for ((slot, sum), prediction) in candidate.iter_mut().zip(&running).zip(predictions) {
                *slot = (sum + prediction) / divisor;
            }
            let loss = rmse(targets, &candidate)?;
            if best.is_none_or(|(_, current)| loss < current) {
                best = Some((member, loss));
            }
        }
        let Some((chosen, _)) = best else {
            return Err(ComparatorError::InvalidParameter { name: "members" });
        };
        counts[chosen] += 1;
        for (sum, prediction) in running.iter_mut().zip(&member_predictions[chosen]) {
            *sum += prediction;
        }
    }
    Ok(counts
        .into_iter()
        .map(|count| count as f64 / rounds as f64)
        .collect())
}
