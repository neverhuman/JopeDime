//! Train-only identical-fold CV bake-off (SC-DEC-184 JA-03).
//!
//! Every candidate sees the same approved five-fold partition. The fitter is handed training
//! indices only, so preprocessing, neural context construction, transforms and the head are
//! fitted inside each train-side closure. A group may not span two folds. A candidate is
//! `Scored` only when every fold carries verified fit evidence; a candidate with unknown evidence
//! is `ScoredUnverified` and cannot be selected. A failed fold excludes the candidate.
//!
//! Counters are separate. `fold_attempts` counts callbacks reached. `physical_starts` is summed
//! only from verified fit evidence: a refusal before start counts zero, and any unknown
//! evidence makes the total unknown rather than zero.
use super::candidates::{Admission, CandidateSpec, RuntimeCapability};
use super::oof::{ExpectedPlan, FoldRecord, FoldStatus, OofPredictionBundle, OOF_SCHEMA};
use crate::evaluation_score::{score_predictions, Predictions};
use crate::experiment::Metric;
use anyhow::{bail, ensure, Result};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const BAKEOFF_SCHEMA: &str = "jope.cv-bakeoff-receipt/v1";
/// The approved selection protocol: five folds. A different count needs its own versioned
/// protocol and is refused here.
pub const APPROVED_FOLD_COUNT: usize = 5;
pub const SELECTION_RULE: &str = "approved five-fold partition; lowest unweighted mean of fold losses among candidates whose every fold carries verified fit evidence; ties by candidate id; pooled OOF loss reported separately; failed, refused, budget-limited and unverified candidates are never selected";

/// A fitter's refusal before any fit started. Its fold counts zero physical starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotStarted(pub String);

impl std::fmt::Display for NotStarted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "refused before start: {}", self.0)
    }
}

impl std::error::Error for NotStarted {}

/// Evidence about one fold's fit, as reported by the worker that ran it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "evidence", rename_all = "snake_case", deny_unknown_fields)]
pub enum FitEvidence {
    /// A verified worker receipt: the request it ran under and the physical fits it performed,
    /// which may be more than one when the fitter fits internally.
    Verified {
        request_sha256: String,
        physical_starts: u64,
    },
    /// Refused before any fit started.
    NotStarted,
    /// No terminal evidence exists.
    Unknown,
}

/// Fits a candidate on training rows and predicts the held-out rows. Preprocessing, neural
/// context construction, transforms and the head must all be fitted inside `fit_predict`, using
/// `train` only.
pub trait CandidateFitter {
    fn fit_predict(
        &self,
        candidate: &CandidateSpec,
        train: &[usize],
        predict: &[usize],
    ) -> Result<FoldFit>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct FoldFit {
    /// One prediction per `predict` row, in the order given.
    pub predictions: Predictions,
    pub fitted_artifact_sha256: String,
    pub evidence: FitEvidence,
}

/// The fixed evaluation population, shared by every candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct BakeoffPlan {
    pub task: Task,
    /// Frozen class order. Empty for regression.
    pub classes: Vec<String>,
    pub row_ids: Vec<String>,
    /// Target label per row: a number for regression, a class name otherwise.
    pub labels: Vec<String>,
    pub folds: Vec<u32>,
    /// Group per row. Empty when no grouping applies.
    pub group_ids: Vec<String>,
    pub request_sha256: String,
}

impl BakeoffPlan {
    /// Refuses misaligned or invalid input before any candidate is attempted. Returns the sorted
    /// distinct fold ids, which must number exactly `APPROVED_FOLD_COUNT`.
    pub fn validate(&self) -> Result<Vec<u32>> {
        let n = self.row_ids.len();
        ensure!(n > 0, "bake-off has no rows");
        let unique: BTreeSet<_> = self.row_ids.iter().collect();
        ensure!(unique.len() == n, "bake-off repeats a row id");
        ensure!(self.labels.len() == n, "labels are not row-aligned");
        ensure!(self.folds.len() == n, "folds are not row-aligned");
        ensure!(
            self.group_ids.is_empty() || self.group_ids.len() == n,
            "groups are not row-aligned"
        );
        ensure!(
            self.request_sha256.len() == 64
                && self
                    .request_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "request identity is not a sha256"
        );
        match self.task {
            Task::Regression => {
                ensure!(self.classes.is_empty(), "regression plan carries classes");
                for label in &self.labels {
                    let value: f64 = label
                        .trim()
                        .parse()
                        .map_err(|_| anyhow::anyhow!("invalid regression label {label:?}"))?;
                    ensure!(value.is_finite(), "nonfinite regression label");
                }
            }
            Task::Binary => ensure!(self.classes.len() == 2, "binary plan needs two classes"),
            Task::Multiclass => ensure!(
                self.classes.len() >= 3,
                "multiclass plan needs three classes"
            ),
            other => bail!("task {other:?} is outside the bake-off"),
        }
        if self.task != Task::Regression {
            let classes: BTreeSet<_> = self.classes.iter().collect();
            ensure!(classes.len() == self.classes.len(), "duplicate class");
            ensure!(
                self.labels.iter().all(|label| classes.contains(label)),
                "a label is outside the frozen class order"
            );
        }
        let folds: BTreeSet<u32> = self.folds.iter().copied().collect();
        ensure!(
            folds.len() == APPROVED_FOLD_COUNT,
            "bake-off uses the approved {APPROVED_FOLD_COUNT}-fold protocol; got {} folds",
            folds.len()
        );
        if !self.group_ids.is_empty() {
            let mut group_folds: BTreeMap<&str, BTreeSet<u32>> = BTreeMap::new();
            for (group, fold) in self.group_ids.iter().zip(&self.folds) {
                group_folds.entry(group.as_str()).or_default().insert(*fold);
            }
            ensure!(
                group_folds.values().all(|set| set.len() == 1),
                "a group spans more than one fold (crossed-group leakage)"
            );
        }
        Ok(folds.into_iter().collect())
    }
}

/// Whole-request budget, charged before each attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationBudget {
    cap: u64,
    charged: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetExhausted {
    pub requested: u64,
    pub remaining: u64,
}

impl EvaluationBudget {
    pub fn new(cap: u64) -> Self {
        Self { cap, charged: 0 }
    }
    pub fn charge(&mut self, units: u64) -> std::result::Result<(), BudgetExhausted> {
        let remaining = self.cap - self.charged;
        if units > remaining {
            return Err(BudgetExhausted {
                requested: units,
                remaining,
            });
        }
        self.charged += units;
        Ok(())
    }
    pub fn charged(&self) -> u64 {
        self.charged
    }
    pub fn cap(&self) -> u64 {
        self.cap
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum CandidateStatus {
    /// Every fold verified. Eligible for selection.
    Scored {
        fold_losses: Vec<f64>,
        mean_fold_loss: f64,
        pooled_oof_loss: f64,
    },
    /// Losses computed, but at least one fold lacks verified evidence. Never selected.
    ScoredUnverified {
        fold_losses: Vec<f64>,
        mean_fold_loss: f64,
        pooled_oof_loss: f64,
        unknown_evidence_folds: Vec<u32>,
    },
    FoldFailed {
        fold: u32,
        reason: String,
    },
    BudgetLimited,
    Refused {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateReport {
    pub candidate_id: String,
    pub runtime_capability: RuntimeCapability,
    pub status: CandidateStatus,
    /// Whole candidate evaluations charged against the budget: one per attempted candidate.
    pub charged_candidate_evaluations: u64,
    /// Fold callbacks reached. Not a fit count.
    pub fold_attempts: u64,
    /// Physical fits from verified evidence. `None` when any fold's evidence is unknown.
    pub physical_starts: Option<u64>,
    pub bundle_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BakeoffReceipt {
    pub schema: String,
    pub metric: Metric,
    pub direction: String,
    pub selection_rule: String,
    pub folds: usize,
    pub budget_cap: u64,
    pub charged_candidate_evaluations: u64,
    pub candidates: Vec<CandidateReport>,
    pub selected: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BakeoffOutput {
    pub receipt: BakeoffReceipt,
    /// One bundle per `Scored` candidate, in roster order.
    pub bundles: Vec<OofPredictionBundle>,
}

enum Outcome {
    Scored {
        fold_losses: Vec<f64>,
        mean_fold_loss: f64,
        pooled_oof_loss: f64,
        bundle: Box<OofPredictionBundle>,
        physical_starts: u64,
    },
    Unverified {
        fold_losses: Vec<f64>,
        mean_fold_loss: f64,
        pooled_oof_loss: f64,
        unknown_evidence_folds: Vec<u32>,
    },
    FoldFailed {
        fold: u32,
        reason: String,
        physical_starts: Option<u64>,
    },
}

fn label_subset(plan: &BakeoffPlan, rows: &[usize]) -> Vec<String> {
    rows.iter().map(|&i| plan.labels[i].clone()).collect()
}

/// Concatenates per-fold predictions into one row-aligned set in plan order.
fn assemble(plan: &BakeoffPlan, parts: &[(Vec<usize>, Predictions)]) -> Result<Predictions> {
    let n = plan.row_ids.len();
    match plan.task {
        Task::Regression => {
            let mut out: Vec<Option<f64>> = vec![None; n];
            for (rows, prediction) in parts {
                let Predictions::Values(values) = prediction else {
                    bail!("regression fold returned class probabilities");
                };
                ensure!(
                    values.len() == rows.len(),
                    "fold prediction count differs from its rows"
                );
                for (row, value) in rows.iter().zip(values) {
                    ensure!(out[*row].is_none(), "a row was predicted twice");
                    out[*row] = Some(*value);
                }
            }
            let values = out
                .into_iter()
                .collect::<Option<Vec<f64>>>()
                .ok_or_else(|| anyhow::anyhow!("a row was never predicted"))?;
            Ok(Predictions::Values(values))
        }
        _ => {
            let mut out: Vec<Option<Vec<f64>>> = vec![None; n];
            for (rows, prediction) in parts {
                let Predictions::Probabilities { classes, values } = prediction else {
                    bail!("classification fold returned values");
                };
                ensure!(
                    classes == &plan.classes,
                    "fold class order differs from the frozen order"
                );
                ensure!(
                    values.len() == rows.len(),
                    "fold prediction count differs from its rows"
                );
                for (row, probabilities) in rows.iter().zip(values) {
                    ensure!(out[*row].is_none(), "a row was predicted twice");
                    out[*row] = Some(probabilities.clone());
                }
            }
            let values = out
                .into_iter()
                .collect::<Option<Vec<Vec<f64>>>>()
                .ok_or_else(|| anyhow::anyhow!("a row was never predicted"))?;
            Ok(Predictions::Probabilities {
                classes: plan.classes.clone(),
                values,
            })
        }
    }
}

fn evaluate(
    plan: &BakeoffPlan,
    folds: &[u32],
    candidate: &CandidateSpec,
    fitter: &dyn CandidateFitter,
    fold_attempts: &mut u64,
) -> Result<Outcome> {
    let n = plan.row_ids.len();
    let mut fold_losses = Vec::with_capacity(folds.len());
    let mut records: Vec<FoldRecord> = Vec::with_capacity(folds.len());
    let mut parts: Vec<(Vec<usize>, Predictions)> = Vec::with_capacity(folds.len());
    let mut starts: Option<u64> = Some(0);
    let mut unknown_evidence_folds = Vec::new();
    for &fold in folds {
        let predict: Vec<usize> = (0..n).filter(|&i| plan.folds[i] == fold).collect();
        let train: Vec<usize> = (0..n).filter(|&i| plan.folds[i] != fold).collect();
        *fold_attempts += 1;
        let fit = match fitter.fit_predict(candidate, &train, &predict) {
            Ok(fit) => fit,
            Err(error) => {
                let not_started = error.downcast_ref::<NotStarted>().is_some();
                return Ok(Outcome::FoldFailed {
                    fold,
                    reason: format!("{error:#}"),
                    physical_starts: if not_started { starts } else { None },
                });
            }
        };
        let (status, request) = match &fit.evidence {
            FitEvidence::Verified {
                request_sha256,
                physical_starts,
            } => {
                ensure!(
                    request_sha256 == &plan.request_sha256,
                    "fold {fold} evidence is for a different request"
                );
                starts = starts.map(|total| total + physical_starts);
                (FoldStatus::Verified, request_sha256.clone())
            }
            FitEvidence::NotStarted => {
                return Ok(Outcome::FoldFailed {
                    fold,
                    reason: "fold refused before start".into(),
                    physical_starts: starts,
                })
            }
            FitEvidence::Unknown => {
                starts = None;
                unknown_evidence_folds.push(fold);
                (FoldStatus::Unknown, plan.request_sha256.clone())
            }
        };
        let loss = match score_predictions(
            plan.task,
            &plan.classes,
            &label_subset(plan, &predict),
            &fit.predictions,
        ) {
            Ok(score) => score.loss,
            Err(error) => {
                return Ok(Outcome::FoldFailed {
                    fold,
                    reason: format!("fold predictions refused: {error:#}"),
                    physical_starts: starts,
                })
            }
        };
        fold_losses.push(loss);
        records.push(FoldRecord {
            fold,
            request_sha256: request,
            fitted_artifact_sha256: fit.fitted_artifact_sha256,
            fitting_row_ids: train.iter().map(|&i| plan.row_ids[i].clone()).collect(),
            scoring_row_ids: predict.iter().map(|&i| plan.row_ids[i].clone()).collect(),
            status,
        });
        parts.push((predict, fit.predictions));
    }
    let predictions = assemble(plan, &parts)?;
    let pooled_oof_loss =
        score_predictions(plan.task, &plan.classes, &plan.labels, &predictions)?.loss;
    let mean_fold_loss = fold_losses.iter().sum::<f64>() / fold_losses.len() as f64;
    if !unknown_evidence_folds.is_empty() {
        return Ok(Outcome::Unverified {
            fold_losses,
            mean_fold_loss,
            pooled_oof_loss,
            unknown_evidence_folds,
        });
    }
    let bundle = OofPredictionBundle {
        schema: OOF_SCHEMA.into(),
        task: plan.task,
        candidate_id: candidate.id.into(),
        request_sha256: plan.request_sha256.clone(),
        fitted_artifact_sha256: records
            .iter()
            .map(|r| r.fitted_artifact_sha256.clone())
            .collect(),
        row_ids: plan.row_ids.clone(),
        folds: plan.folds.clone(),
        group_ids: plan.group_ids.clone(),
        classes: plan.classes.clone(),
        predictions,
        failed_folds: vec![],
    };
    let expected = ExpectedPlan {
        request_sha256: plan.request_sha256.clone(),
        row_ids: plan.row_ids.clone(),
        folds: plan.folds.clone(),
        group_ids: plan.group_ids.clone(),
        classes: plan.classes.clone(),
    };
    bundle.validate_against(&expected, &records)?;
    Ok(Outcome::Scored {
        fold_losses,
        mean_fold_loss,
        pooled_oof_loss,
        bundle: Box::new(bundle),
        physical_starts: starts.unwrap_or(0),
    })
}

fn report(
    candidate: &CandidateSpec,
    status: CandidateStatus,
    charged: u64,
    fold_attempts: u64,
    physical_starts: Option<u64>,
    bundle_sha256: Option<String>,
) -> CandidateReport {
    CandidateReport {
        candidate_id: candidate.id.into(),
        runtime_capability: candidate.runtime,
        status,
        charged_candidate_evaluations: charged,
        fold_attempts,
        physical_starts,
        bundle_sha256,
    }
}

/// Runs the bake-off over `candidates` in order. Nothing is fitted on rows that are being
/// predicted, and the budget is charged before each attempt.
pub fn run_bakeoff(
    plan: &BakeoffPlan,
    candidates: &[CandidateSpec],
    fitter: &dyn CandidateFitter,
    budget: &mut EvaluationBudget,
) -> Result<BakeoffOutput> {
    let folds = plan.validate()?;
    let metric = Metric::for_task(plan.task)?;
    let mut reports = Vec::with_capacity(candidates.len());
    let mut bundles = Vec::new();
    for candidate in candidates {
        if candidate.task != plan.task {
            reports.push(report(
                candidate,
                CandidateStatus::Refused {
                    reason: "candidate task differs from the plan".into(),
                },
                0,
                0,
                Some(0),
                None,
            ));
            continue;
        }
        if candidate.admission != Admission::Enabled {
            reports.push(report(
                candidate,
                CandidateStatus::Refused {
                    reason: format!(
                        "admission is {:?}; not enabled for the bake-off",
                        candidate.admission
                    ),
                },
                0,
                0,
                Some(0),
                None,
            ));
            continue;
        }
        if budget.charge(1).is_err() {
            reports.push(report(
                candidate,
                CandidateStatus::BudgetLimited,
                0,
                0,
                Some(0),
                None,
            ));
            continue;
        }
        let mut fold_attempts = 0;
        let outcome = evaluate(plan, &folds, candidate, fitter, &mut fold_attempts)?;
        let entry = match outcome {
            Outcome::Scored {
                fold_losses,
                mean_fold_loss,
                pooled_oof_loss,
                bundle,
                physical_starts,
            } => {
                let sha = bundle.identity()?;
                bundles.push(*bundle);
                report(
                    candidate,
                    CandidateStatus::Scored {
                        fold_losses,
                        mean_fold_loss,
                        pooled_oof_loss,
                    },
                    1,
                    fold_attempts,
                    Some(physical_starts),
                    Some(sha),
                )
            }
            Outcome::Unverified {
                fold_losses,
                mean_fold_loss,
                pooled_oof_loss,
                unknown_evidence_folds,
            } => report(
                candidate,
                CandidateStatus::ScoredUnverified {
                    fold_losses,
                    mean_fold_loss,
                    pooled_oof_loss,
                    unknown_evidence_folds,
                },
                1,
                fold_attempts,
                None,
                None,
            ),
            Outcome::FoldFailed {
                fold,
                reason,
                physical_starts,
            } => report(
                candidate,
                CandidateStatus::FoldFailed { fold, reason },
                1,
                fold_attempts,
                physical_starts,
                None,
            ),
        };
        reports.push(entry);
    }
    let selected = reports
        .iter()
        .filter_map(|report| match &report.status {
            CandidateStatus::Scored { mean_fold_loss, .. } => {
                Some((*mean_fold_loss, report.candidate_id.clone()))
            }
            _ => None,
        })
        .min_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
        .map(|(_, id)| id);
    Ok(BakeoffOutput {
        receipt: BakeoffReceipt {
            schema: BAKEOFF_SCHEMA.into(),
            metric,
            direction: "lower".into(),
            selection_rule: SELECTION_RULE.into(),
            folds: folds.len(),
            budget_cap: budget.cap(),
            charged_candidate_evaluations: budget.charged(),
            candidates: reports,
            selected,
        },
        bundles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Predicts the mean training label for every held-out row, and records every training set.
    struct MeanFitter {
        labels: Vec<f64>,
        seen_train: RefCell<Vec<Vec<usize>>>,
        fail_fold: Option<usize>,
        evidence: FitEvidence,
    }

    impl MeanFitter {
        fn verified(labels: Vec<f64>) -> Self {
            Self {
                labels,
                seen_train: RefCell::new(vec![]),
                fail_fold: None,
                evidence: FitEvidence::Verified {
                    request_sha256: "a".repeat(64),
                    physical_starts: 1,
                },
            }
        }
    }

    impl CandidateFitter for MeanFitter {
        fn fit_predict(
            &self,
            _: &CandidateSpec,
            train: &[usize],
            predict: &[usize],
        ) -> Result<FoldFit> {
            self.seen_train.borrow_mut().push(train.to_vec());
            if let Some(fold) = self.fail_fold {
                if self.seen_train.borrow().len() - 1 == fold {
                    return Err(NotStarted("synthetic refusal before start".into()).into());
                }
            }
            let mean = train.iter().map(|&i| self.labels[i]).sum::<f64>() / train.len() as f64;
            Ok(FoldFit {
                predictions: Predictions::Values(vec![mean; predict.len()]),
                fitted_artifact_sha256: "c".repeat(64),
                evidence: self.evidence.clone(),
            })
        }
    }

    /// Ten rows over the approved five folds: two rows per fold.
    fn plan(n: usize) -> (BakeoffPlan, Vec<f64>) {
        let labels: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let plan = BakeoffPlan {
            task: Task::Regression,
            classes: vec![],
            row_ids: (0..n).map(|i| format!("row-{i}")).collect(),
            labels: labels.iter().map(|v| v.to_string()).collect(),
            folds: (0..n).map(|i| (i % 5) as u32).collect(),
            group_ids: vec![],
            request_sha256: "a".repeat(64),
        };
        (plan, labels)
    }

    fn spec(id: &'static str) -> CandidateSpec {
        CandidateSpec {
            id,
            task: Task::Regression,
            admission: Admission::Enabled,
            runtime: RuntimeCapability::Unknown,
        }
    }

    #[test]
    fn every_fold_trains_on_the_other_folds_only_and_covers_all_rows() {
        let (plan, labels) = plan(10);
        let fitter = MeanFitter::verified(labels);
        let mut budget = EvaluationBudget::new(10);
        run_bakeoff(&plan, &[spec("rust.ridge")], &fitter, &mut budget).unwrap();
        let seen = fitter.seen_train.borrow();
        assert_eq!(seen.len(), 5, "one fit per approved fold");
        for (k, train) in seen.iter().enumerate() {
            assert!(
                train.iter().all(|&i| plan.folds[i] != k as u32),
                "fold {k} trained on its own rows"
            );
            assert_eq!(train.len(), 8);
        }
    }

    #[test]
    fn a_partition_that_is_not_five_folds_is_refused_before_any_fit() {
        let (mut plan, labels) = plan(9);
        plan.folds = (0..9).map(|i| (i % 3) as u32).collect();
        let fitter = MeanFitter::verified(labels);
        assert!(run_bakeoff(
            &plan,
            &[spec("rust.ridge")],
            &fitter,
            &mut EvaluationBudget::new(5)
        )
        .is_err());
        assert!(fitter.seen_train.borrow().is_empty());
    }

    #[test]
    fn a_group_spanning_two_folds_is_refused_as_crossed_leakage() {
        let (mut plan, labels) = plan(10);
        // Rows 0 and 1 share group g but sit in folds 0 and 1.
        plan.group_ids = (0..10)
            .map(|i| {
                if i < 2 {
                    "g".to_string()
                } else {
                    format!("u{i}")
                }
            })
            .collect();
        let fitter = MeanFitter::verified(labels);
        assert!(run_bakeoff(
            &plan,
            &[spec("rust.ridge")],
            &fitter,
            &mut EvaluationBudget::new(5)
        )
        .is_err());
        assert!(fitter.seen_train.borrow().is_empty());
    }

    #[test]
    fn a_scored_candidate_has_a_bundle_bound_to_verified_fold_records() {
        let (plan, labels) = plan(10);
        let fitter = MeanFitter::verified(labels);
        let out = run_bakeoff(
            &plan,
            &[spec("rust.ridge")],
            &fitter,
            &mut EvaluationBudget::new(10),
        )
        .unwrap();
        assert_eq!(out.bundles.len(), 1);
        assert_eq!(out.bundles[0].row_ids, plan.row_ids);
        assert_eq!(out.receipt.selected.as_deref(), Some("rust.ridge"));
        match &out.receipt.candidates[0].status {
            CandidateStatus::Scored {
                fold_losses,
                mean_fold_loss,
                pooled_oof_loss,
            } => {
                assert_eq!(fold_losses.len(), 5);
                let unweighted = fold_losses.iter().sum::<f64>() / 5.0;
                assert!((mean_fold_loss - unweighted).abs() < 1e-12);
                assert!(pooled_oof_loss.is_finite());
            }
            other => panic!("expected scored, got {other:?}"),
        }
        assert_eq!(out.receipt.candidates[0].fold_attempts, 5);
        assert_eq!(out.receipt.candidates[0].physical_starts, Some(5));
        assert_eq!(out.receipt.candidates[0].charged_candidate_evaluations, 1);
    }

    #[test]
    fn internal_multiple_fits_are_counted_from_verified_evidence_not_from_callbacks() {
        let (plan, labels) = plan(10);
        let mut fitter = MeanFitter::verified(labels);
        fitter.evidence = FitEvidence::Verified {
            request_sha256: "a".repeat(64),
            physical_starts: 3,
        };
        let out = run_bakeoff(
            &plan,
            &[spec("rust.ridge")],
            &fitter,
            &mut EvaluationBudget::new(5),
        )
        .unwrap();
        assert_eq!(out.receipt.candidates[0].fold_attempts, 5);
        assert_eq!(out.receipt.candidates[0].physical_starts, Some(15));
    }

    #[test]
    fn a_refusal_before_start_counts_zero_physical_starts() {
        let (plan, labels) = plan(10);
        let mut fitter = MeanFitter::verified(labels);
        fitter.fail_fold = Some(0);
        let out = run_bakeoff(
            &plan,
            &[spec("rust.ridge")],
            &fitter,
            &mut EvaluationBudget::new(5),
        )
        .unwrap();
        assert!(matches!(
            out.receipt.candidates[0].status,
            CandidateStatus::FoldFailed { fold: 0, .. }
        ));
        assert_eq!(out.receipt.candidates[0].fold_attempts, 1);
        assert_eq!(out.receipt.candidates[0].physical_starts, Some(0));
        assert!(out.bundles.is_empty());
    }

    #[test]
    fn unknown_fit_evidence_leaves_the_count_unknown_and_the_candidate_unselectable() {
        let (plan, labels) = plan(10);
        let mut fitter = MeanFitter::verified(labels);
        fitter.evidence = FitEvidence::Unknown;
        let out = run_bakeoff(
            &plan,
            &[spec("rust.ridge")],
            &fitter,
            &mut EvaluationBudget::new(5),
        )
        .unwrap();
        assert!(matches!(
            out.receipt.candidates[0].status,
            CandidateStatus::ScoredUnverified { .. }
        ));
        assert_eq!(out.receipt.candidates[0].physical_starts, None);
        assert_eq!(out.receipt.selected, None);
        assert!(out.bundles.is_empty());
    }

    #[test]
    fn a_failed_fold_excludes_the_candidate_and_it_is_never_selected() {
        let (plan, labels) = plan(10);
        let mut fitter = MeanFitter::verified(labels);
        fitter.fail_fold = Some(1);
        let out = run_bakeoff(
            &plan,
            &[spec("rust.ridge")],
            &fitter,
            &mut EvaluationBudget::new(10),
        )
        .unwrap();
        assert!(matches!(
            out.receipt.candidates[0].status,
            CandidateStatus::FoldFailed { fold: 1, .. }
        ));
        assert!(out.bundles.is_empty());
        assert_eq!(out.receipt.selected, None);
    }

    #[test]
    fn the_budget_is_charged_before_each_attempt_and_the_rest_are_budget_limited() {
        let (plan, labels) = plan(10);
        let fitter = MeanFitter::verified(labels);
        let mut budget = EvaluationBudget::new(1);
        let out = run_bakeoff(
            &plan,
            &[spec("rust.ridge"), spec("rust.ridge_gcv")],
            &fitter,
            &mut budget,
        )
        .unwrap();
        assert_eq!(out.receipt.charged_candidate_evaluations, 1);
        assert!(matches!(
            out.receipt.candidates[1].status,
            CandidateStatus::BudgetLimited
        ));
        assert_eq!(
            fitter.seen_train.borrow().len(),
            5,
            "the limited candidate was never fitted"
        );
    }

    #[test]
    fn a_candidate_that_is_not_enabled_is_refused_without_charge() {
        let (plan, labels) = plan(10);
        let fitter = MeanFitter::verified(labels);
        let mut gated = spec("rust.tab_boost");
        gated.admission = Admission::ReceiptRequired;
        let mut budget = EvaluationBudget::new(5);
        let out = run_bakeoff(&plan, &[gated], &fitter, &mut budget).unwrap();
        assert!(matches!(
            out.receipt.candidates[0].status,
            CandidateStatus::Refused { .. }
        ));
        assert_eq!(budget.charged(), 0);
    }

    #[test]
    fn selection_is_deterministic_and_ties_break_by_candidate_id() {
        let (plan, labels) = plan(10);
        let fitter = MeanFitter::verified(labels);
        let roster = [spec("rust.zeta"), spec("rust.alpha")];
        let first = run_bakeoff(&plan, &roster, &fitter, &mut EvaluationBudget::new(10)).unwrap();
        let second = run_bakeoff(&plan, &roster, &fitter, &mut EvaluationBudget::new(10)).unwrap();
        assert_eq!(first.receipt, second.receipt);
        assert_eq!(first.receipt.selected.as_deref(), Some("rust.alpha"));
    }

    #[test]
    fn invalid_regression_labels_are_refused_before_any_fit() {
        let (mut plan, labels) = plan(10);
        plan.labels[0] = "not-a-number".into();
        let fitter = MeanFitter::verified(labels);
        assert!(run_bakeoff(
            &plan,
            &[spec("rust.ridge")],
            &fitter,
            &mut EvaluationBudget::new(5)
        )
        .is_err());
        assert!(fitter.seen_train.borrow().is_empty());
    }
}
