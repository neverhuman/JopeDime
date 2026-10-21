//! Bounded automatic selection, `--method auto` (SPECIFICATION.md §6.3).
//!
//! `jope.runtime-select/v1`, a new runtime behaviour with its own identity (it is not the
//! research panel's search or a promoted proposal policy):
//!
//! 1. Candidates: the task's native roster in registry order, or the `--candidates` subset.
//! 2. Folds: `stratified-round-robin/v1` — the k-th row of each class (each row, for regression)
//!    goes to fold `k mod folds`; deterministic, no seed. `folds = min(requested, rows / 2,
//!    smallest class count)`, at least 2.
//! 3. Each candidate is fitted on every training part and predicts its held-out part through the
//!    same [`fit_native`]/[`predict_bound`] path as a final fit. All preprocessing is refitted per
//!    part: held-out rows contribute no labels, statistics or vocabulary.
//! 4. Loss on the pooled out-of-fold predictions, matching the panel's primary metric: RMSE,
//!    `1 - AUROC`, or log loss with probabilities clipped below at 1e-15.
//! 5. The lowest loss wins; an exact tie goes to the earlier roster entry. A candidate that fails
//!    any fold, by error or by panic, is recorded with its error and not selected.
//! 6. The deadline bounds selection: it is checked before every fold fit, and candidates not
//!    reached are recorded as skipped. The chosen recipe is then refitted once on all rows; that
//!    refit always runs and its time is charged to the run.
use crate::budget::Budget;
use crate::methods::native::{fit_native, predict_bound, Predictions};
use crate::methods::RustMethod;
use crate::table::Training;
use crate::task::{self, TaskKind};
use crate::Failure;
use anyhow::{Context, Result};
use feat_math::advanced_v3::RawTable;
use serde::Serialize;
use std::time::Instant;

pub const SELECTION_POLICY: &str = "jope.runtime-select/v1";
pub const FOLD_POLICY: &str = "stratified-round-robin/v1";
pub const DEFAULT_FOLDS: usize = 5;
/// Probability floor of the log loss, the panel's epsilon.
pub const PROBABILITY_EPSILON: f64 = 1e-15;

#[derive(Clone, Debug, Serialize)]
pub struct CandidateOutcome {
    pub id: String,
    /// `scored`, `failed` or `skipped`.
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f64>,
    pub folds_completed: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Selection {
    pub policy: &'static str,
    pub fold_policy: &'static str,
    pub folds: usize,
    pub metric: &'static str,
    pub loss: &'static str,
    pub selected: String,
    pub candidates: Vec<CandidateOutcome>,
}

/// Fold of every row.
pub fn folds(kind: TaskKind, targets: &[f64], folds: usize) -> Vec<usize> {
    let mut out = vec![0; targets.len()];
    match kind {
        TaskKind::Binary | TaskKind::Multiclass => {
            let mut rank = std::collections::BTreeMap::<u64, usize>::new();
            for (row, value) in targets.iter().enumerate() {
                let r = rank.entry(value.to_bits()).or_default();
                out[row] = *r % folds;
                *r += 1;
            }
        }
        _ => {
            for (row, fold) in out.iter_mut().enumerate() {
                *fold = row % folds;
            }
        }
    }
    out
}

/// The loss name for a task.
pub fn loss_name(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Regression => "rmse",
        TaskKind::Binary => "1-auroc",
        _ => "log_loss",
    }
}

/// Loss of predictions against targets (values, or class indices).
pub fn loss(kind: TaskKind, targets: &[f64], predictions: &Predictions) -> Result<f64> {
    let value = match (kind, predictions) {
        (TaskKind::Regression, Predictions::Values(values)) => {
            feat_math::metrics::rmse(targets, values)
        }
        (TaskKind::Binary, Predictions::Probabilities { values, .. }) => {
            // IEEE -0 and +0 are the same probability and must tie in AUROC.
            let positive: Vec<f64> = values
                .iter()
                .map(|row| if row[1] == 0.0 { 0.0 } else { row[1] })
                .collect();
            1.0 - feat_math::metrics::auc(targets, &positive)
        }
        (TaskKind::Multiclass, Predictions::Probabilities { values, .. }) => {
            let sum: f64 = targets
                .iter()
                .zip(values)
                .map(|(target, row)| -row[*target as usize].max(PROBABILITY_EPSILON).ln())
                .sum();
            sum / targets.len() as f64
        }
        _ => anyhow::bail!("predictions do not match the task"),
    };
    anyhow::ensure!(value.is_finite(), "non-finite loss");
    Ok(value)
}

fn subset(table: &RawTable, rows: &[usize]) -> Result<RawTable> {
    RawTable::new(
        table.headers.clone(),
        rows.iter().map(|r| table.rows[*r].clone()).collect(),
    )
}

/// Out-of-fold predictions of one candidate, assembled in row order.
fn out_of_fold(
    method: &RustMethod,
    training: &Training,
    assignment: &[usize],
    k: usize,
    budget: &Budget,
    completed: &mut usize,
) -> Result<Predictions> {
    let n = training.targets.len();
    let mut values: Vec<Option<Vec<f64>>> = vec![None; n];
    for fold in 0..k {
        budget.check(&format!("fold {fold} of {}", method.id))?;
        let train_rows: Vec<usize> = (0..n).filter(|r| assignment[*r] != fold).collect();
        let valid_rows: Vec<usize> = (0..n).filter(|r| assignment[*r] == fold).collect();
        let train = subset(&training.table, &train_rows)?;
        let targets: Vec<f64> = train_rows.iter().map(|r| training.targets[*r]).collect();
        let model = fit_native(
            method,
            training.kind,
            &training.schema,
            &training.classes,
            &train,
            &targets,
        )
        .with_context(|| format!("fold {fold}"))?;
        let predicted = predict_bound(
            &model,
            &subset(&training.table, &valid_rows)?,
            &training.classes,
        )?;
        match predicted {
            Predictions::Values(v) => {
                for (row, value) in valid_rows.iter().zip(v) {
                    values[*row] = Some(vec![value]);
                }
            }
            Predictions::Probabilities { values: v, .. } => {
                for (row, value) in valid_rows.iter().zip(v) {
                    values[*row] = Some(value);
                }
            }
        }
        *completed += 1;
    }
    let rows: Vec<Vec<f64>> = values
        .into_iter()
        .map(|v| v.context("a row was in no fold"))
        .collect::<Result<_>>()?;
    Ok(if training.kind == TaskKind::Regression {
        Predictions::Values(rows.into_iter().map(|r| r[0]).collect())
    } else {
        Predictions::Probabilities {
            classes: training.classes.clone(),
            values: rows,
        }
    })
}

/// The effective fold count, or an input failure when selection is impossible.
pub fn effective_folds(training: &Training, requested: usize) -> Result<usize> {
    let n = training.targets.len();
    if n < 4 {
        return Err(Failure::input(format!(
            "automatic selection needs at least 4 rows; the table has {n} (name a --method)"
        )));
    }
    let mut k = requested.max(2).min(n / 2);
    if training.kind != TaskKind::Regression {
        let mut counts = vec![0usize; training.classes.len()];
        for t in &training.targets {
            counts[*t as usize] += 1;
        }
        let smallest = counts.iter().copied().min().unwrap_or(0);
        if smallest < 2 {
            let class = &training.classes[counts.iter().position(|c| *c == smallest).unwrap_or(0)];
            return Err(Failure::input(format!(
                "class {class:?} has {smallest} row; automatic selection needs at least 2 per class \
                 (name a --method)"
            )));
        }
        k = k.min(smallest);
    }
    Ok(k.max(2))
}

/// Run the selection plan; returns the record and the winning recipe.
pub fn select(
    training: &Training,
    candidates: &[RustMethod],
    requested_folds: usize,
    budget: &Budget,
) -> Result<(Selection, RustMethod)> {
    let k = effective_folds(training, requested_folds)?;
    let assignment = folds(training.kind, &training.targets, k);
    let mut outcomes = Vec::with_capacity(candidates.len());
    let mut best: Option<(f64, usize)> = None;
    for (index, method) in candidates.iter().enumerate() {
        let started = Instant::now();
        if budget.exhausted() {
            outcomes.push(CandidateOutcome {
                id: method.id.clone(),
                status: "skipped",
                loss: None,
                folds_completed: 0,
                reason: Some("time limit reached".into()),
                seconds: 0.0,
            });
            continue;
        }
        let mut completed = 0;
        let result = crate::error::contain(&method.id, || {
            out_of_fold(method, training, &assignment, k, budget, &mut completed)
        })
        .and_then(|p| loss(training.kind, &training.targets, &p));
        let seconds = started.elapsed().as_secs_f64();
        match result {
            Ok(value) => {
                if best.is_none_or(|(b, _)| value < b) {
                    best = Some((value, index));
                }
                outcomes.push(CandidateOutcome {
                    id: method.id.clone(),
                    status: "scored",
                    loss: Some(value),
                    folds_completed: completed,
                    reason: None,
                    seconds,
                });
            }
            Err(error) => {
                let skipped = crate::error::kind_of(&error) == crate::FailureKind::Budget;
                outcomes.push(CandidateOutcome {
                    id: method.id.clone(),
                    status: if skipped { "skipped" } else { "failed" },
                    loss: None,
                    folds_completed: completed,
                    reason: Some(format!("{error:#}")),
                    seconds,
                });
            }
        }
    }
    let Some((_, winner)) = best else {
        let budget_ran_out = outcomes.iter().any(|o| o.status == "skipped");
        let detail: Vec<String> = outcomes
            .iter()
            .map(|o| format!("{}: {}", o.id, o.reason.as_deref().unwrap_or(o.status)))
            .collect();
        let message = format!("no candidate completed selection ({})", detail.join("; "));
        return Err(if budget_ran_out {
            Failure::budget(message)
        } else {
            anyhow::anyhow!(message)
        });
    };
    let selection = Selection {
        policy: SELECTION_POLICY,
        fold_policy: FOLD_POLICY,
        folds: k,
        metric: task::metric(training.kind),
        loss: loss_name(training.kind),
        selected: candidates[winner].id.clone(),
        candidates: outcomes,
    };
    Ok((selection, candidates[winner].clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_are_stratified_and_deterministic() {
        let targets = [0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0];
        let a = folds(TaskKind::Multiclass, &targets, 2);
        assert_eq!(a, folds(TaskKind::Multiclass, &targets, 2));
        for class in [0.0, 1.0, 2.0] {
            let per_fold: Vec<usize> = (0..2)
                .map(|f| {
                    targets
                        .iter()
                        .zip(&a)
                        .filter(|(t, k)| **t == class && **k == f)
                        .count()
                })
                .collect();
            assert!(per_fold.iter().all(|c| *c >= 1), "{class}: {per_fold:?}");
        }
        assert_eq!(folds(TaskKind::Regression, &[1.0; 5], 2), [0, 1, 0, 1, 0]);
    }

    #[test]
    fn losses_match_the_panel_metrics() {
        let rmse = loss(
            TaskKind::Regression,
            &[1.0, 3.0],
            &Predictions::Values(vec![2.0, 2.0]),
        )
        .unwrap();
        assert!((rmse - 1.0).abs() < 1e-12);
        let p = Predictions::Probabilities {
            classes: vec!["a".into(), "b".into()],
            values: vec![vec![0.9, 0.1], vec![0.2, 0.8]],
        };
        assert_eq!(loss(TaskKind::Binary, &[0.0, 1.0], &p).unwrap(), 0.0);
        let p = Predictions::Probabilities {
            classes: vec!["a".into(), "b".into(), "c".into()],
            values: vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]],
        };
        let ll = loss(TaskKind::Multiclass, &[0.0, 1.0], &p).unwrap();
        assert!((ll - (-(1e-15f64).ln() / 2.0)).abs() < 1e-9);
    }
}
