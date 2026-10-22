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
//!
//! `jope.runtime-select/v2` (regression only) replaces step 4's loss with a portable objective
//! ([`crate::fitness`]) and may add chronological forward folds and an objective-aware output
//! calibration ([`crate::calibration`]). Each candidate's score is the better of its raw
//! out-of-fold objective and its cross-fitted calibrated objective; the winner's final output map
//! is learned on all of its out-of-fold predictions and stored with the fitted model.
use crate::budget::Budget;
use crate::calibration::{self, Calibration, OutputTransform};
use crate::fitness::{Columns, Direction, Fitness};
use crate::methods::native::{fit_native, predict_bound, Predictions};
use crate::methods::RustMethod;
use crate::table::Training;
use crate::task::{self, TaskKind};
use crate::Failure;
use anyhow::{Context, Result};
use feat_math::advanced_v3::RawTable;
use serde::{Deserialize, Serialize};
use std::time::Instant;

pub const SELECTION_POLICY: &str = "jope.runtime-select/v1";
/// Objective, forward-validation and calibration selection.
pub const SELECTION_POLICY_V2: &str = "jope.runtime-select/v2";
pub const FORWARD_FOLD_POLICY: &str = "expanding-window-gap/v1";
pub const FOLD_POLICY: &str = "stratified-round-robin/v1";
pub const DEFAULT_FOLDS: usize = 5;
/// Probability floor of the log loss, the panel's epsilon.
pub const PROBABILITY_EPSILON: f64 = 1e-15;

/// Forward folds train only on earlier rows, with an explicit embargo of `gap` rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Validation {
    Iid {},
    Forward { gap: usize },
}

impl Default for Validation {
    fn default() -> Self {
        Self::Iid {}
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CandidateOutcome {
    pub id: String,
    /// `scored`, `failed` or `skipped`.
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f64>,
    /// Natural value (before maximization is negated).
    pub fitness_value: Option<f64>,
    /// With calibration: the objective loss of the raw out-of-fold predictions (absent when the
    /// raw predictions are outside the objective's domain).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uncalibrated_loss: Option<f64>,
    /// With calibration: the cross-fitted calibrated loss.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibrated_loss: Option<f64>,
    /// Whether `loss` is the calibrated one.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub calibrated: bool,
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
    pub metric: String,
    pub loss: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fitness: Option<crate::fitness::Fitness>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fitness_sha256: Option<String>,
    pub validation: Validation,
    pub scored_rows: usize,
    pub warmup_rows: usize,
    /// Present when the selection learned output maps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationRecord>,
    pub selected: String,
    pub candidates: Vec<CandidateOutcome>,
}

/// How output maps were learned, and the map the selected candidate keeps.
#[derive(Clone, Debug, Serialize)]
pub struct CalibrationRecord {
    pub family: Calibration,
    pub method: &'static str,
    /// The selected candidate's final map, learned on all its out-of-fold predictions; absent
    /// when its raw predictions scored at least as well.
    pub output_transform: Option<OutputTransform>,
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
    validation: &Validation,
) -> Result<(Predictions, Vec<usize>)> {
    let n = training.targets.len();
    let mut values: Vec<Option<Vec<f64>>> = vec![None; n];
    for fold in 0..k {
        budget.check(&format!("fold {fold} of {}", method.id))?;
        let valid_rows: Vec<usize> = (0..n).filter(|r| assignment[*r] == fold).collect();
        let train_rows: Vec<usize> = match validation {
            Validation::Iid {} => (0..n).filter(|r| assignment[*r] != fold).collect(),
            Validation::Forward { gap } => (0..valid_rows[0].saturating_sub(*gap)).collect(),
        };
        anyhow::ensure!(
            train_rows.len() >= 2,
            "forward fold needs at least two training rows after gap"
        );
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
    let mut scored = Vec::new();
    let mut rows = Vec::new();
    for (index, value) in values.into_iter().enumerate() {
        if assignment[index] == usize::MAX {
            continue;
        }
        rows.push(value.context("a scoring row was in no fold")?);
        scored.push(index);
    }
    Ok((
        if training.kind == TaskKind::Regression {
            Predictions::Values(rows.into_iter().map(|r| r[0]).collect())
        } else {
            Predictions::Probabilities {
                classes: training.classes.clone(),
                values: rows,
            }
        },
        scored,
    ))
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
    select_with_fitness(
        training,
        candidates,
        requested_folds,
        budget,
        None,
        &Validation::Iid {},
        Calibration::None,
    )
}

/// One scored candidate's evidence, kept until the winner is known.
struct Scored {
    loss: f64,
    calibrated: bool,
    predictions: Vec<f64>,
}

/// Scored rows whose out-of-fold predictions may teach the map applied to `fold`: every other
/// fold, or with forward validation only the earlier folds (none for the first).
fn calibration_rows(folds_of_scored: &[usize], fold: usize, forward: bool) -> Vec<usize> {
    (0..folds_of_scored.len())
        .filter(|&i| {
            if forward {
                folds_of_scored[i] < fold
            } else {
                folds_of_scored[i] != fold
            }
        })
        .collect()
}

/// The objective's score of one candidate's out-of-fold predictions: raw, and cross-fitted
/// calibrated when a calibration family is in force.
#[allow(clippy::too_many_arguments)]
fn objective_score(
    fitness: &Fitness,
    family: Calibration,
    y: &[f64],
    p: &[f64],
    columns: &Columns,
    folds_of_scored: &[usize],
    forward: bool,
) -> Result<(Option<f64>, Option<f64>)> {
    let raw = fitness.evaluate_with(y, p, columns).map(|s| s.loss);
    if family == Calibration::None {
        return Ok((Some(raw?), None));
    }
    let fit_rows = |fold: usize| calibration_rows(folds_of_scored, fold, forward);
    let crossed =
        calibration::cross_fit(fitness, family, y, p, columns, folds_of_scored, &fit_rows);
    let calibrated = fitness.evaluate_with(y, &crossed, columns).map(|s| s.loss);
    match (raw, calibrated) {
        (Err(raw), Err(_)) => Err(raw),
        (raw, calibrated) => Ok((raw.ok(), calibrated.ok())),
    }
}

/// Selection with a portable regression objective, forward validation and output calibration.
/// `None` objective with IID folds and no calibration is exactly [`select`] (`v1`).
pub fn select_with_fitness(
    training: &Training,
    candidates: &[RustMethod],
    requested_folds: usize,
    budget: &Budget,
    fitness: Option<&Fitness>,
    validation: &Validation,
    calibration: Calibration,
) -> Result<(Selection, RustMethod)> {
    let family = calibration.resolve(fitness.is_some());
    if let Some(fitness) = fitness {
        fitness.validate()?;
        anyhow::ensure!(
            training.kind == TaskKind::Regression,
            "custom fitness is regression-only"
        );
        anyhow::ensure!(
            !fitness.requires_order() || matches!(validation, Validation::Forward { .. }),
            "{} depends on row order and requires forward validation",
            fitness.name()
        );
        for column in fitness.columns() {
            anyhow::ensure!(
                training.auxiliary.contains_key(&column),
                "the training table did not supply objective column {column:?}"
            );
        }
    } else {
        anyhow::ensure!(
            family == Calibration::None,
            "output calibration needs an objective (--fitness)"
        );
    }
    anyhow::ensure!(
        matches!(validation, Validation::Iid {}) || training.kind == TaskKind::Regression,
        "forward validation is regression-only"
    );
    let k = effective_folds(training, requested_folds)?;
    let n = training.targets.len();
    let assignment = match validation {
        Validation::Iid {} => folds(training.kind, &training.targets, k),
        Validation::Forward { gap } => {
            let block = n / (k + 1);
            anyhow::ensure!(
                block.saturating_sub(*gap) >= 2,
                "forward validation needs more rows or a smaller fold count/gap"
            );
            (0..n)
                .map(|r| {
                    if r < block {
                        usize::MAX
                    } else {
                        ((r - block) / block).min(k - 1)
                    }
                })
                .collect()
        }
    };
    let forward = matches!(validation, Validation::Forward { .. });
    let warmup_rows = assignment.iter().filter(|f| **f == usize::MAX).count();
    let scored_rows: Vec<usize> = (0..n).filter(|r| assignment[*r] != usize::MAX).collect();
    let scored_targets: Vec<f64> = scored_rows.iter().map(|r| training.targets[*r]).collect();
    let scored_folds: Vec<usize> = scored_rows.iter().map(|r| assignment[*r]).collect();
    let scored_columns = calibration::subset_columns(&training.auxiliary, &scored_rows);
    let mut outcomes = Vec::with_capacity(candidates.len());
    let mut best: Option<(f64, usize)> = None;
    let mut evidence: Vec<Option<Scored>> = Vec::with_capacity(candidates.len());
    for (index, method) in candidates.iter().enumerate() {
        let started = Instant::now();
        let mut outcome = CandidateOutcome {
            id: method.id.clone(),
            status: "skipped",
            loss: None,
            fitness_value: None,
            uncalibrated_loss: None,
            calibrated_loss: None,
            calibrated: false,
            folds_completed: 0,
            reason: None,
            seconds: 0.0,
        };
        evidence.push(None);
        if budget.exhausted() {
            outcome.reason = Some("time limit reached".into());
            outcomes.push(outcome);
            continue;
        }
        let mut completed = 0;
        let result = crate::error::contain(&method.id, || {
            out_of_fold(
                method,
                training,
                &assignment,
                k,
                budget,
                &mut completed,
                validation,
            )
        })
        .and_then(|(p, scored)| {
            anyhow::ensure!(
                scored == scored_rows,
                "scored rows differ between candidates"
            );
            let Some(fitness) = fitness else {
                let value = loss(training.kind, &scored_targets, &p)?;
                return Ok((value, None, None, false, Vec::new()));
            };
            let Predictions::Values(values) = p else {
                anyhow::bail!("regression fitness needs numeric predictions");
            };
            let (raw, calibrated) = objective_score(
                fitness,
                family,
                &scored_targets,
                &values,
                &scored_columns,
                &scored_folds,
                forward,
            )?;
            let use_calibrated = match (raw, calibrated) {
                (Some(r), Some(c)) => c < r,
                (None, Some(_)) => true,
                _ => false,
            };
            let value =
                if use_calibrated { calibrated } else { raw }.expect("one score is present");
            Ok((value, raw, calibrated, use_calibrated, values))
        });
        outcome.seconds = started.elapsed().as_secs_f64();
        outcome.folds_completed = completed;
        match result {
            Ok((value, raw, calibrated, use_calibrated, predictions)) => {
                if best.is_none_or(|(b, _)| value < b) {
                    best = Some((value, index));
                }
                let natural = match fitness.map(Fitness::direction) {
                    Some(Direction::Maximize) => -value,
                    _ => value,
                };
                outcome.status = "scored";
                outcome.loss = Some(value);
                outcome.fitness_value = Some(natural);
                if family != Calibration::None {
                    outcome.uncalibrated_loss = raw;
                    outcome.calibrated_loss = calibrated;
                    outcome.calibrated = use_calibrated;
                }
                evidence[index] = Some(Scored {
                    loss: value,
                    calibrated: use_calibrated,
                    predictions,
                });
            }
            Err(error) => {
                let skipped = crate::error::kind_of(&error) == crate::FailureKind::Budget;
                outcome.status = if skipped { "skipped" } else { "failed" };
                outcome.reason = Some(format!("{error:#}"));
            }
        }
        outcomes.push(outcome);
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
    let calibration_record = match (fitness, family) {
        (Some(fitness), family) if family != Calibration::None => {
            let won = evidence[winner].as_ref().expect("the winner was scored");
            debug_assert!(won.loss.is_finite());
            let output_transform = won.calibrated.then(|| {
                calibration::fit(
                    fitness,
                    family,
                    &scored_targets,
                    &won.predictions,
                    &scored_columns,
                )
            });
            Some(CalibrationRecord {
                family,
                method: calibration::CALIBRATION_METHOD,
                output_transform,
            })
        }
        _ => None,
    };
    let v2 = fitness.is_some() || forward;
    let selection = Selection {
        policy: if v2 {
            SELECTION_POLICY_V2
        } else {
            SELECTION_POLICY
        },
        fold_policy: if forward {
            FORWARD_FOLD_POLICY
        } else {
            FOLD_POLICY
        },
        folds: k,
        metric: fitness.map_or_else(|| task::metric(training.kind).to_owned(), Fitness::name),
        loss: if fitness.is_some() {
            "direction_normalized_fitness".into()
        } else {
            loss_name(training.kind).into()
        },
        fitness: fitness.cloned(),
        fitness_sha256: fitness.map(Fitness::identity).transpose()?,
        validation: validation.clone(),
        scored_rows: n - warmup_rows,
        warmup_rows,
        calibration: calibration_record,
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

#[cfg(test)]
mod fitness_tests {
    use super::*;
    use crate::fitness::{Aggregation, Direction, Expression, Fitness};
    fn training() -> Training {
        let text = String::from("x,y\n")
            + &(0..60)
                .map(|i| format!("{i},{}\n", i * i + (if i % 7 == 0 { 250 } else { 0 })))
                .collect::<String>();
        let limits = crate::limits::Limits::default();
        let csv = crate::table::parse_csv(text.as_bytes(), &limits).unwrap();
        crate::table::training(
            &csv,
            "y",
            crate::task::TaskChoice::Fixed(TaskKind::Regression),
            &limits,
        )
        .unwrap()
    }
    #[test]
    fn custom_fitness_changes_the_selected_candidate_and_records_both_directions() {
        let train = training();
        let budget = Budget::new(None, 1);
        let candidates =
            ["ridge", "histgbm"].map(|s| crate::methods::resolve(TaskKind::Regression, s).unwrap());
        let run = |direction| {
            let f = Fitness::Custom {
                name: "mean_prediction".into(),
                direction,
                expression: Expression::Pred {},
                aggregation: Aggregation::Mean {},
            };
            let (selection, method) = select_with_fitness(
                &train,
                &candidates,
                3,
                &budget,
                Some(&f),
                &Validation::Iid {},
                Calibration::None,
            )
            .unwrap();
            let best = selection
                .candidates
                .iter()
                .filter_map(|c| c.loss)
                .min_by(f64::total_cmp)
                .unwrap();
            assert_eq!(
                selection
                    .candidates
                    .iter()
                    .find(|c| c.id == method.id)
                    .unwrap()
                    .loss,
                Some(best)
            );
            assert_eq!(selection.fitness_sha256, Some(f.identity().unwrap()));
            for c in &selection.candidates {
                assert_eq!(
                    c.loss,
                    c.fitness_value
                        .map(|v| if direction == Direction::Minimize {
                            v
                        } else {
                            -v
                        })
                );
            }
            method.id
        };
        assert_ne!(run(Direction::Minimize), run(Direction::Maximize));
    }
    #[test]
    fn forward_validation_excludes_warmup_and_never_trains_on_future_labels() {
        let mut train = training();
        let budget = Budget::new(None, 1);
        let method = crate::methods::resolve(TaskKind::Regression, "ridge").unwrap();
        let validation = Validation::Forward { gap: 2 };
        let (selection, _) = select_with_fitness(
            &train,
            std::slice::from_ref(&method),
            3,
            &budget,
            Some(&Fitness::Mae {}),
            &validation,
            Calibration::Auto,
        )
        .unwrap();
        assert!(selection.calibration.is_some());
        assert_eq!(selection.warmup_rows, 15);
        assert_eq!(selection.scored_rows, 45);
        assert_eq!(selection.fold_policy, "expanding-window-gap/v1");
        let assignment = (0..60)
            .map(|r| {
                if r < 15 {
                    usize::MAX
                } else {
                    ((r - 15) / 15).min(2)
                }
            })
            .collect::<Vec<_>>();
        let (a, sa) = out_of_fold(
            &method,
            &train,
            &assignment,
            3,
            &budget,
            &mut 0,
            &validation,
        )
        .unwrap();
        // Fold 0 holds rows 15..30; gap rows 13,14 and all rows >=30 are unavailable to it.
        for i in [13, 14, 30, 45, 59] {
            train.targets[i] += 1e6;
        }
        let (b, sb) = out_of_fold(
            &method,
            &train,
            &assignment,
            3,
            &budget,
            &mut 0,
            &validation,
        )
        .unwrap();
        let (Predictions::Values(a), Predictions::Values(b)) = (a, b) else {
            panic!("regression");
        };
        assert_eq!(&a[..15], &b[..15]);
        assert_eq!(sa, (15..60).collect::<Vec<_>>());
        assert_eq!(sa, sb);
        // A map applied to a forward fold is learned only from earlier folds' predictions.
        let folds: Vec<usize> = assignment[15..].to_vec();
        assert!(calibration_rows(&folds, 0, true).is_empty());
        assert_eq!(
            calibration_rows(&folds, 2, true),
            (0..30).collect::<Vec<_>>()
        );
        assert_eq!(calibration_rows(&folds, 1, false).len(), 30);
        for order_dependent in ["sortino", "mase", "max_drawdown"] {
            assert!(select_with_fitness(
                &train,
                std::slice::from_ref(&method),
                3,
                &budget,
                Some(&Fitness::parse(order_dependent).unwrap()),
                &Validation::Iid {},
                Calibration::Auto,
            )
            .is_err());
        }
    }
    #[test]
    fn explicit_default_matches_existing_selection_and_invalid_scope_is_refused() {
        let train = training();
        let budget = Budget::new(None, 1);
        let candidates =
            ["ridge", "histgbm"].map(|s| crate::methods::resolve(TaskKind::Regression, s).unwrap());
        let (a, ma) = select(&train, &candidates, 3, &budget).unwrap();
        let (b, mb) = select_with_fitness(
            &train,
            &candidates,
            3,
            &budget,
            Some(&Fitness::Rmse {}),
            &Validation::Iid {},
            Calibration::None,
        )
        .unwrap();
        assert_eq!(ma, mb);
        assert!(b.calibration.is_none());
        for (a, b) in a.candidates.iter().zip(b.candidates) {
            assert!((a.loss.unwrap() - b.loss.unwrap()).abs() < 1e-10);
        }
        let options = crate::api::FitOptions {
            method: "ridge".into(),
            fitness: Some(Fitness::Mae {}),
            ..Default::default()
        };
        assert!(crate::api::fit(&train, &options, &budget, "", Default::default()).is_err());
        assert!(select_with_fitness(
            &train,
            &candidates,
            3,
            &budget,
            None,
            &Validation::Forward { gap: 99 },
            Calibration::None,
        )
        .is_err());
        // Calibration without an objective has nothing to optimize.
        assert!(select_with_fitness(
            &train,
            &candidates,
            3,
            &budget,
            None,
            &Validation::Iid {},
            Calibration::Affine,
        )
        .is_err());
    }
    #[test]
    fn calibration_turns_mean_models_into_the_requested_quantile_and_never_scores_worse() {
        let train = training();
        let budget = Budget::new(None, 1);
        let candidates =
            ["ridge", "histgbm"].map(|s| crate::methods::resolve(TaskKind::Regression, s).unwrap());
        let fitness = Fitness::Pinball { quantile: 0.9 };
        let (selection, _) = select_with_fitness(
            &train,
            &candidates,
            3,
            &budget,
            Some(&fitness),
            &Validation::Iid {},
            Calibration::Affine,
        )
        .unwrap();
        let record = selection.calibration.as_ref().unwrap();
        assert_eq!(record.family, Calibration::Affine);
        for c in &selection.candidates {
            let (raw, calibrated) = (c.uncalibrated_loss.unwrap(), c.calibrated_loss.unwrap());
            assert_eq!(c.loss.unwrap(), raw.min(calibrated));
            assert_eq!(c.calibrated, calibrated < raw);
        }
        let winner = selection
            .candidates
            .iter()
            .find(|c| c.id == selection.selected)
            .unwrap();
        assert_eq!(record.output_transform.is_some(), winner.calibrated);
        assert!(selection.candidates.iter().any(|c| c.calibrated));
    }
    #[test]
    fn objectives_read_auxiliary_columns_that_never_become_features() {
        let text = String::from("x,price,y\n")
            + &(0..40)
                .map(|i| format!("{i},{},{}\n", 1 + i % 3, 2 * i + i % 5))
                .collect::<String>();
        let limits = crate::limits::Limits::default();
        let csv = crate::table::parse_csv(text.as_bytes(), &limits).unwrap();
        let fitness = Fitness::parse(
            r#"{"metric":"weighted","weights":{"op":"column","name":"price"},
                "fitness":{"metric":"mae"}}"#,
        )
        .unwrap();
        let regression = crate::task::TaskChoice::Fixed(TaskKind::Regression);
        let train = crate::table::training_with_auxiliary(
            &csv,
            "y",
            regression,
            &limits,
            &fitness.columns(),
        )
        .unwrap();
        assert_eq!(train.schema.columns.len(), 1);
        let budget = Budget::new(None, 1);
        let ridge = crate::methods::resolve(TaskKind::Regression, "ridge").unwrap();
        let (selection, _) = select_with_fitness(
            &train,
            std::slice::from_ref(&ridge),
            4,
            &budget,
            Some(&fitness),
            &Validation::Iid {},
            Calibration::None,
        )
        .unwrap();
        assert_eq!(selection.metric, "weighted_mae");
        // Without the column the objective cannot be scored.
        let plain = crate::table::training(&csv, "y", regression, &limits).unwrap();
        assert!(select_with_fitness(
            &plain,
            std::slice::from_ref(&ridge),
            4,
            &budget,
            Some(&fitness),
            &Validation::Iid {},
            Calibration::None,
        )
        .is_err());
    }
}
