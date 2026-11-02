use super::*;

/// Deterministic stratified index split: rows sorted by target, sliced into
/// deciles, each decile shuffled with its own stream, the leading `hold_frac`
/// of each shuffled decile becoming the validation fold. Fresh `seed` per
/// generation gives a distinct split every time (train_only_study discipline).
pub(crate) fn split_8020(y: &[f64], seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]).then(a.cmp(&b)));
    let deciles = 10usize.min(y.len().max(1));
    let chunk = order.len().div_ceil(deciles);
    let mut train = Vec::new();
    let mut valid = Vec::new();
    for (d, ids) in order.chunks(chunk).enumerate() {
        let mut ids: Vec<usize> = ids.to_vec();
        let mut rng = Pcg32::new(seed, 0x5350_4c49_5430 ^ d as u64);
        for i in (1..ids.len()).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            ids.swap(i, j);
        }
        let hold = (ids.len() as f64 * 0.2).round().max(1.0) as usize;
        for (k, id) in ids.into_iter().enumerate() {
            if k < hold {
                valid.push(id);
            } else {
                train.push(id);
            }
        }
    }
    if train.is_empty() || valid.is_empty() {
        let mid = y.len() * 4 / 5;
        return ((0..mid).collect(), (mid..y.len()).collect());
    }
    (train, valid)
}

/// Legacy V1 fold-count adaptation. Strict V2 never changes the declared fold
/// count: an infeasible plan is a structural failure rather than a reason to
/// substitute leave-one-out CV.
pub(crate) fn usable_folds(n: usize, configured_k: usize) -> usize {
    if n < 2 {
        return 0;
    }
    let k = configured_k.max(2).min(n);
    if n < 2 * k {
        n
    } else {
        k
    }
}

/// Deterministic stratified k-fold: per-decile shuffle, then deal round-robin.
pub(crate) fn kfold(y: &[f64], k: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]).then(a.cmp(&b)));
    let deciles = 10usize.min(y.len().max(1));
    let chunk = order.len().div_ceil(deciles);
    let mut folds = vec![Vec::new(); k];
    let mut card = 0usize;
    for (d, ids) in order.chunks(chunk).enumerate() {
        let mut ids: Vec<usize> = ids.to_vec();
        let mut rng = Pcg32::new(seed, 0x004b_464f_4c44 ^ d as u64);
        for i in (1..ids.len()).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            ids.swap(i, j);
        }
        for id in ids {
            folds[card % k].push(id);
            card += 1;
        }
    }
    folds
}

pub(crate) fn validate_strict_fold_plan(
    row_count: usize,
    folds: &[Vec<usize>],
) -> Result<(), String> {
    if folds.len() < 2 {
        return Err("strict pooled CV requires at least two folds".to_string());
    }
    let mut coverage = vec![0u8; row_count];
    for (fold_index, fold) in folds.iter().enumerate() {
        if fold.len() < STRICT_MIN_HOLD_ROWS {
            return Err(format!(
                "fold {fold_index} has {} hold rows; minimum is {STRICT_MIN_HOLD_ROWS}",
                fold.len()
            ));
        }
        let fit_rows = row_count.saturating_sub(fold.len());
        if fit_rows < STRICT_MIN_FIT_ROWS {
            return Err(format!(
                "fold {fold_index} has {fit_rows} fit rows; minimum is {STRICT_MIN_FIT_ROWS}"
            ));
        }
        for &row in fold {
            let Some(count) = coverage.get_mut(row) else {
                return Err(format!("fold {fold_index} contains out-of-range row {row}"));
            };
            *count = count.saturating_add(1);
        }
    }
    if let Some((row, count)) = coverage.iter().enumerate().find(|(_, count)| **count != 1) {
        return Err(format!(
            "row {row} appears in {count} hold folds; expected exactly one"
        ));
    }
    Ok(())
}

/// Complete pooled OOF R^2 in original row order. For a fixed truth vector,
/// ranking by this value is exactly equivalent to ranking by pooled OOF RMSE.
/// Any absent, malformed, duplicated, or nonfinite fold fails closed.
pub(crate) fn complete_pooled_oof_r2(
    y: &[f64],
    folds: &[Vec<usize>],
    fold_predictions: &[Option<&[f64]>],
) -> Option<f64> {
    if fold_predictions.len() != folds.len() || y.is_empty() {
        return None;
    }
    let mut predictions = vec![f64::NAN; y.len()];
    let mut covered = vec![false; y.len()];
    for (fold, maybe_predictions) in folds.iter().zip(fold_predictions) {
        let values = maybe_predictions.as_ref()?;
        if values.len() != fold.len() {
            return None;
        }
        for (&row, &prediction) in fold.iter().zip(values.iter()) {
            if row >= y.len() || covered[row] || !prediction.is_finite() {
                return None;
            }
            predictions[row] = prediction;
            covered[row] = true;
        }
    }
    if covered.iter().any(|covered| !covered) {
        return None;
    }
    r2(y, &predictions).map(|value| value.max(-1.0))
}

pub(crate) fn replace_generation_outcomes(
    population: &mut [Member],
    outcomes: Vec<Option<f64>>,
) -> usize {
    assert_eq!(population.len(), outcomes.len());
    let mut invalid = 0usize;
    for (member, outcome) in population.iter_mut().zip(outcomes) {
        member.scores.clear();
        member.fitness_complete = outcome.is_some();
        if !member.fitness_complete {
            invalid += 1;
        }
        member.scores.push(outcome.unwrap_or(-1.0));
    }
    invalid
}

pub(crate) fn expected_search_evaluations(schedule: &[(usize, usize)], folds: usize) -> usize {
    schedule.iter().map(|(width, _)| width * folds).sum()
}
