//! Train-only identical-fold CV bake-off (SC-DEC-184 JA-03).
//!
//! Every candidate sees the same folds. The fitter is handed training indices only, so
//! preprocessing, neural context construction, transforms and the head are fitted inside each
//! train-side closure; the bake-off never fits on rows it is about to predict. A candidate with
//! any failed fold is not scored and cannot be selected. Selection uses the unweighted mean of
//! the fold losses. The pooled out-of-fold loss is reported separately under its own name.
use super::candidates::{Admission, CandidateSpec};
use super::oof::{OofPredictionBundle, OOF_SCHEMA};
use crate::evaluation_score::{score_predictions, Predictions};
use crate::experiment::Metric;
use anyhow::{bail, ensure, Result};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const BAKEOFF_SCHEMA: &str = "jope.cv-bakeoff-receipt/v1";
pub const SELECTION_RULE: &str = "lowest unweighted mean of fold losses; ties by candidate id; pooled OOF loss reported separately; failed, refused and budget-limited candidates are never selected";

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
    /// distinct fold ids.
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
        ensure!(folds.len() >= 2, "bake-off needs at least two folds");
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
    Scored {
        fold_losses: Vec<f64>,
        mean_fold_loss: f64,
        pooled_oof_loss: f64,
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
    pub status: CandidateStatus,
    /// Whole candidate evaluations charged against the budget: one per attempted candidate.
    pub charged_candidate_evaluations: u64,
    /// Physical fits performed: one per fold attempted.
    pub physical_fits: u64,
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
    /// One bundle per scored candidate, in roster order.
    pub bundles: Vec<OofPredictionBundle>,
}

enum Outcome {
    Scored {
        fold_losses: Vec<f64>,
        mean_fold_loss: f64,
        pooled_oof_loss: f64,
        bundle: Box<OofPredictionBundle>,
    },
    FoldFailed {
        fold: u32,
        reason: String,
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
    physical_fits: &mut u64,
) -> Result<Outcome> {
    let n = plan.row_ids.len();
    let mut fold_losses = Vec::with_capacity(folds.len());
    let mut artifacts = Vec::with_capacity(folds.len());
    let mut parts: Vec<(Vec<usize>, Predictions)> = Vec::with_capacity(folds.len());
    for &fold in folds {
        let predict: Vec<usize> = (0..n).filter(|&i| plan.folds[i] == fold).collect();
        let train: Vec<usize> = (0..n).filter(|&i| plan.folds[i] != fold).collect();
        *physical_fits += 1;
        let fit = match fitter.fit_predict(candidate, &train, &predict) {
            Ok(fit) => fit,
            Err(error) => {
                return Ok(Outcome::FoldFailed {
                    fold,
                    reason: format!("{error:#}"),
                })
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
                })
            }
        };
        fold_losses.push(loss);
        artifacts.push(fit.fitted_artifact_sha256);
        parts.push((predict, fit.predictions));
    }
    let predictions = assemble(plan, &parts)?;
    let pooled_oof_loss =
        score_predictions(plan.task, &plan.classes, &plan.labels, &predictions)?.loss;
    let mean_fold_loss = fold_losses.iter().sum::<f64>() / fold_losses.len() as f64;
    let bundle = OofPredictionBundle {
        schema: OOF_SCHEMA.into(),
        task: plan.task,
        candidate_id: candidate.id.into(),
        request_sha256: plan.request_sha256.clone(),
        fitted_artifact_sha256: artifacts,
        row_ids: plan.row_ids.clone(),
        folds: plan.folds.clone(),
        group_ids: plan.group_ids.clone(),
        classes: plan.classes.clone(),
        predictions,
        failed_folds: vec![],
    };
    bundle.validate()?;
    Ok(Outcome::Scored {
        fold_losses,
        mean_fold_loss,
        pooled_oof_loss,
        bundle: Box::new(bundle),
    })
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
            reports.push(CandidateReport {
                candidate_id: candidate.id.into(),
                status: CandidateStatus::Refused {
                    reason: "candidate task differs from the plan".into(),
                },
                charged_candidate_evaluations: 0,
                physical_fits: 0,
                bundle_sha256: None,
            });
            continue;
        }
        if candidate.admission != Admission::Enabled {
            reports.push(CandidateReport {
                candidate_id: candidate.id.into(),
                status: CandidateStatus::Refused {
                    reason: format!(
                        "admission is {:?}; not enabled for the bake-off",
                        candidate.admission
                    ),
                },
                charged_candidate_evaluations: 0,
                physical_fits: 0,
                bundle_sha256: None,
            });
            continue;
        }
        if budget.charge(1).is_err() {
            reports.push(CandidateReport {
                candidate_id: candidate.id.into(),
                status: CandidateStatus::BudgetLimited,
                charged_candidate_evaluations: 0,
                physical_fits: 0,
                bundle_sha256: None,
            });
            continue;
        }
        let mut physical_fits = 0;
        let report = match evaluate(plan, &folds, candidate, fitter, &mut physical_fits)? {
            Outcome::Scored {
                fold_losses,
                mean_fold_loss,
                pooled_oof_loss,
                bundle,
            } => {
                let sha = bundle.identity()?;
                bundles.push(*bundle);
                CandidateReport {
                    candidate_id: candidate.id.into(),
                    status: CandidateStatus::Scored {
                        fold_losses,
                        mean_fold_loss,
                        pooled_oof_loss,
                    },
                    charged_candidate_evaluations: 1,
                    physical_fits,
                    bundle_sha256: Some(sha),
                }
            }
            Outcome::FoldFailed { fold, reason } => CandidateReport {
                candidate_id: candidate.id.into(),
                status: CandidateStatus::FoldFailed { fold, reason },
                charged_candidate_evaluations: 1,
                physical_fits,
                bundle_sha256: None,
            },
        };
        reports.push(report);
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
                    bail!("synthetic fit failure");
                }
            }
            let mean = train.iter().map(|&i| self.labels[i]).sum::<f64>() / train.len() as f64;
            Ok(FoldFit {
                predictions: Predictions::Values(vec![mean; predict.len()]),
                fitted_artifact_sha256: "c".repeat(64),
            })
        }
    }

    fn plan(n: usize) -> (BakeoffPlan, Vec<f64>) {
        let labels: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let plan = BakeoffPlan {
            task: Task::Regression,
            classes: vec![],
            row_ids: (0..n).map(|i| format!("row-{i}")).collect(),
            labels: labels.iter().map(|v| v.to_string()).collect(),
            folds: (0..n).map(|i| (i % 3) as u32).collect(),
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
        }
    }

    #[test]
    fn every_fold_trains_on_the_other_folds_only() {
        let (plan, labels) = plan(9);
        let fitter = MeanFitter {
            labels,
            seen_train: RefCell::new(vec![]),
            fail_fold: None,
        };
        let mut budget = EvaluationBudget::new(10);
        run_bakeoff(&plan, &[spec("rust.ridge")], &fitter, &mut budget).unwrap();
        for (k, train) in fitter.seen_train.borrow().iter().enumerate() {
            assert!(
                train.iter().all(|&i| plan.folds[i] != k as u32),
                "fold {k} trained on its own rows"
            );
            assert_eq!(train.len(), 6);
        }
    }

    #[test]
    fn a_scored_candidate_has_a_row_aligned_bundle_and_separate_mean_and_pooled_losses() {
        let (plan, labels) = plan(9);
        let fitter = MeanFitter {
            labels,
            seen_train: RefCell::new(vec![]),
            fail_fold: None,
        };
        let mut budget = EvaluationBudget::new(10);
        let out = run_bakeoff(&plan, &[spec("rust.ridge")], &fitter, &mut budget).unwrap();
        assert_eq!(out.bundles.len(), 1);
        assert_eq!(out.bundles[0].row_ids, plan.row_ids);
        assert_eq!(out.receipt.selected.as_deref(), Some("rust.ridge"));
        match &out.receipt.candidates[0].status {
            CandidateStatus::Scored {
                fold_losses,
                mean_fold_loss,
                pooled_oof_loss,
            } => {
                assert_eq!(fold_losses.len(), 3);
                let unweighted = fold_losses.iter().sum::<f64>() / 3.0;
                assert!((mean_fold_loss - unweighted).abs() < 1e-12);
                assert!(pooled_oof_loss.is_finite());
            }
            other => panic!("expected scored, got {other:?}"),
        }
        assert_eq!(out.receipt.candidates[0].physical_fits, 3);
        assert_eq!(out.receipt.candidates[0].charged_candidate_evaluations, 1);
    }

    #[test]
    fn a_failed_fold_excludes_the_candidate_and_it_is_never_selected() {
        let (plan, labels) = plan(9);
        let fitter = MeanFitter {
            labels,
            seen_train: RefCell::new(vec![]),
            fail_fold: Some(1),
        };
        let mut budget = EvaluationBudget::new(10);
        let out = run_bakeoff(&plan, &[spec("rust.ridge")], &fitter, &mut budget).unwrap();
        assert!(matches!(
            out.receipt.candidates[0].status,
            CandidateStatus::FoldFailed { fold: 1, .. }
        ));
        assert!(out.bundles.is_empty());
        assert_eq!(out.receipt.selected, None);
    }

    #[test]
    fn the_budget_is_charged_before_each_attempt_and_the_rest_are_budget_limited() {
        let (plan, labels) = plan(9);
        let fitter = MeanFitter {
            labels,
            seen_train: RefCell::new(vec![]),
            fail_fold: None,
        };
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
            3,
            "the limited candidate was never fitted"
        );
    }

    #[test]
    fn a_candidate_that_is_not_enabled_is_refused_without_charge() {
        let (plan, labels) = plan(9);
        let fitter = MeanFitter {
            labels,
            seen_train: RefCell::new(vec![]),
            fail_fold: None,
        };
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
        let (plan, labels) = plan(9);
        let fitter = MeanFitter {
            labels,
            seen_train: RefCell::new(vec![]),
            fail_fold: None,
        };
        let roster = [spec("rust.zeta"), spec("rust.alpha")];
        let first = run_bakeoff(&plan, &roster, &fitter, &mut EvaluationBudget::new(10)).unwrap();
        let second = run_bakeoff(&plan, &roster, &fitter, &mut EvaluationBudget::new(10)).unwrap();
        assert_eq!(first.receipt, second.receipt);
        assert_eq!(first.receipt.selected.as_deref(), Some("rust.alpha"));
    }

    #[test]
    fn invalid_regression_labels_are_refused_before_any_fit() {
        let (mut plan, labels) = plan(9);
        plan.labels[0] = "not-a-number".into();
        let fitter = MeanFitter {
            labels,
            seen_train: RefCell::new(vec![]),
            fail_fold: None,
        };
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
