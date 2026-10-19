//! Post-search winner ensemble (SC-DEC-184 JA-04, revised under SC-DEC-184M JA-R1, JA-R2, JA-R4).
//!
//! Isolation (JA-R1). The blend weights are chosen by greedy forward selection with replacement,
//! scored only on the selection side: the out-of-fold predictions of models that were fitted on
//! selection rows alone. The report side is never used to choose anything. Its scores come from
//! report predictions that the caller supplies, produced by refitting each member on the whole
//! selection side. The earlier 3-fold/2-fold split of one shared table is gone: base models there
//! had been trained on the report folds, so its report loss was dependent.
//!
//! Identity (JA-R2). Every member and every label set must reproduce one `ExecutionIdentity`, which
//! is stored in the ensemble and checked on reload.
//!
//! Budget (JA-R4). An ensemble attempt is charged against the caller's whole-request budget, so it
//! shares one cap with the candidate evaluations that produced its members.
//!
//! Acceptance is decided on selection-side out-of-fold loss only. The report loss is recorded and
//! labelled with its basis; it never changes the weights or the disposition.
use super::identity::{digest_ids, ExecutionIdentity};
use super::oof::AdmittedBundle;
use super::refit::{predictions_digest, OuterRefitRecord, VerifiedRefit, REFIT_RECORD_SCHEMA};
use super::selection::{BudgetKind, EvaluationBudget};
use crate::evaluation_score::{score_predictions, validate_probabilities, Predictions};
use anyhow::{bail, ensure, Result};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ENSEMBLE_SCHEMA: &str = "jope.winner-ensemble/v2";
pub const SELECTION_RULE: &str = "greedy forward selection with replacement over non-negative counts, scored on selection-side out-of-fold predictions only; accepted when the blend beats the best single member and at least two distinct members carry weight; otherwise best-single fallback or not attempted";
/// The report loss is computed from members refitted on the whole selection side.
pub const REPORT_BASIS: &str = "outer_refit_predictions";
/// Upper bound on greedy steps. Each step adds one unit of weight.
pub const MAX_ITERATIONS: u32 = 25;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "disposition", rename_all = "snake_case", deny_unknown_fields)]
pub enum EnsembleDisposition {
    Blend,
    BestSingleFallback { reason: String },
    NotAttempted { reason: String },
}

/// Report-side predictions for each member, refitted on the whole selection side.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportPredictions {
    pub ids: Vec<String>,
    pub labels: Vec<String>,
    /// Predictions per member candidate id, aligned with `ids`.
    pub per_member: BTreeMap<String, Predictions>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WinnerEnsemble {
    pub schema: String,
    pub task: Task,
    pub selection_rule: String,
    pub identity: ExecutionIdentity,
    /// Integer weight per member candidate. Weights are counts divided by their total.
    pub counts: BTreeMap<String, u32>,
    pub disposition: EnsembleDisposition,
    pub best_single: Option<String>,
    /// Loss of the final weights on selection-side out-of-fold predictions. Used for acceptance.
    pub selection_loss: f64,
    /// Loss of the final weights on the report side. Never used to choose weights.
    pub report_loss: f64,
    pub report_loss_basis: String,
    /// Identity of each member bundle the weights were chosen from.
    pub member_bundle_sha256: BTreeMap<String, String>,
}

/// Restricts a prediction set to the given rows, keeping row order.
fn subset(predictions: &Predictions, rows: &[usize]) -> Result<Predictions> {
    Ok(match predictions {
        Predictions::Values(values) => {
            Predictions::Values(rows.iter().map(|&i| values[i]).collect())
        }
        Predictions::Probabilities { classes, values } => Predictions::Probabilities {
            classes: classes.clone(),
            values: rows.iter().map(|&i| values[i].clone()).collect(),
        },
    })
}

/// Weighted average of member predictions. Probabilities are re-validated, never renormalised.
fn blend(members: &[&Predictions], counts: &[u32]) -> Result<Predictions> {
    let total: u32 = counts.iter().sum();
    ensure!(total > 0, "blend has no weight");
    let weights: Vec<f64> = counts
        .iter()
        .map(|&c| f64::from(c) / f64::from(total))
        .collect();
    let rows = match members.first() {
        Some(Predictions::Values(v)) => v.len(),
        Some(Predictions::Probabilities { values, .. }) => values.len(),
        None => bail!("blend has no members"),
    };
    match members[0] {
        Predictions::Values(_) => {
            let mut out = vec![0.0; rows];
            for (member, weight) in members.iter().zip(&weights) {
                let Predictions::Values(values) = member else {
                    bail!("member kinds differ");
                };
                ensure!(values.len() == rows, "member row counts differ");
                for (slot, value) in out.iter_mut().zip(values) {
                    *slot += weight * value;
                }
            }
            Ok(Predictions::Values(out))
        }
        Predictions::Probabilities { classes, .. } => {
            let width = classes.len();
            let mut out = vec![vec![0.0; width]; rows];
            for (member, weight) in members.iter().zip(&weights) {
                let Predictions::Probabilities {
                    classes: member_classes,
                    values,
                } = member
                else {
                    bail!("member kinds differ");
                };
                ensure!(member_classes == classes, "member class orders differ");
                ensure!(values.len() == rows, "member row counts differ");
                for (row, probabilities) in out.iter_mut().zip(values) {
                    for (slot, p) in row.iter_mut().zip(probabilities) {
                        *slot += weight * p;
                    }
                }
            }
            validate_probabilities(&out, width)?;
            Ok(Predictions::Probabilities {
                classes: classes.clone(),
                values: out,
            })
        }
    }
}

/// Production entry point for a winner ensemble. Report losses are granted only when the outer refit
/// record agrees with the identity, the report side and the member set (SC-DEC-184R JA-F1). A
/// report loss cannot be presented without its record, because the call is refused without one.
pub fn admit_winner_ensemble(
    admitted: &[AdmittedBundle],
    selection_labels: &[String],
    identity: &ExecutionIdentity,
    report: &ReportPredictions,
    refit: &VerifiedRefit,
    budget: &mut EvaluationBudget,
) -> Result<WinnerEnsemble> {
    check_refit(identity, report, refit.record())?;
    build_winner_ensemble(admitted, selection_labels, identity, report, budget)
}

/// Reloads a persisted ensemble from disk. The record's content is checked as in admission, but
/// provenance is not required: a record read back from the artifact is a claim about the
/// artifact, and the caller verifies it by restoring its states and comparing predictions. This
/// function never grants a record admission.
pub(super) fn reload_winner_ensemble(
    admitted: &[AdmittedBundle],
    selection_labels: &[String],
    identity: &ExecutionIdentity,
    report: &ReportPredictions,
    refit: &OuterRefitRecord,
    budget: &mut EvaluationBudget,
) -> Result<WinnerEnsemble> {
    check_refit(identity, report, refit)?;
    build_winner_ensemble(admitted, selection_labels, identity, report, budget)
}

/// The record's content must agree with the identity, the report side and the member set, and each
/// member's report predictions must reproduce the bits the refit recorded.
fn check_refit(
    identity: &ExecutionIdentity,
    report: &ReportPredictions,
    refit: &OuterRefitRecord,
) -> Result<()> {
    ensure!(
        report.ids.len() == report.labels.len(),
        "report ids and labels are not aligned"
    );
    ensure!(
        refit.schema() == REFIT_RECORD_SCHEMA,
        "unknown outer refit record schema"
    );
    ensure!(
        refit.request_sha256() == identity.request_sha256,
        "the outer refit ran under a different request"
    );
    ensure!(
        refit.plan_sha256() == identity.plan_sha256,
        "the outer refit ran under a different plan"
    );
    ensure!(
        refit.train_rows_sha256() == identity.row_sha256,
        "the outer refit trained on rows other than the selection side"
    );
    ensure!(
        refit.predict_rows_sha256() == digest_ids(&report.ids),
        "the outer refit predicted rows other than the report side"
    );
    ensure!(
        refit.member_ids().eq(report.per_member.keys()),
        "the outer refit does not cover exactly the report predictions"
    );
    for (id, predictions) in &report.per_member {
        ensure!(
            refit.member_report_sha256(id) == Some(predictions_digest(predictions).as_str()),
            "the report predictions for {id} are not the ones the outer refit produced"
        );
    }
    Ok(())
}

/// The winner-ensemble core. Private: production code reaches it only through
/// `admit_winner_ensemble`, which checks the outer refit record first.
fn build_winner_ensemble(
    admitted: &[AdmittedBundle],
    selection_labels: &[String],
    identity: &ExecutionIdentity,
    report: &ReportPredictions,
    budget: &mut EvaluationBudget,
) -> Result<WinnerEnsemble> {
    ensure!(!admitted.is_empty(), "no member bundles");
    identity.verify_members(admitted, selection_labels)?;
    let first = admitted[0].bundle();
    for member in admitted {
        let bundle = member.bundle();
        ensure!(bundle.task == first.task, "member tasks differ");
        ensure!(bundle.row_ids == first.row_ids, "member rows differ");
        ensure!(bundle.folds == first.folds, "member folds differ");
        ensure!(
            bundle.classes == first.classes,
            "member class orders differ"
        );
    }
    ensure!(
        selection_labels.len() == first.row_ids.len(),
        "selection labels are not row-aligned"
    );
    let task = first.task;
    let classes = first.classes.clone();

    ensure!(
        super::identity::digest_pairs(&report.ids, &report.labels) == identity.report_sha256,
        "report rows and labels do not reproduce the report identity"
    );
    let member_ids: BTreeSet<&str> = admitted
        .iter()
        .map(|m| m.bundle().candidate_id.as_str())
        .collect();
    ensure!(
        member_ids.len() == admitted.len(),
        "duplicate member candidate"
    );
    let report_keys: BTreeSet<&str> = report.per_member.keys().map(String::as_str).collect();
    ensure!(
        report_keys == member_ids,
        "report predictions do not cover exactly the members"
    );
    for (id, prediction) in &report.per_member {
        let rows = match prediction {
            Predictions::Values(v) => v.len(),
            Predictions::Probabilities { values, .. } => values.len(),
        };
        ensure!(
            rows == report.labels.len(),
            "report predictions for {id} are not row-aligned"
        );
    }

    budget
        .check_wall_clock()
        .and_then(|()| budget.charge(BudgetKind::EnsembleAttempt, 1))
        .map_err(|e| anyhow::anyhow!("ensemble attempt over the request budget: {e:?}"))?;

    let mut sorted: Vec<&AdmittedBundle> = admitted.iter().collect();
    sorted.sort_by(|a, b| a.bundle().candidate_id.cmp(&b.bundle().candidate_id));
    let all_rows: Vec<usize> = (0..first.row_ids.len()).collect();
    let sel_members: Vec<Predictions> = sorted
        .iter()
        .map(|m| subset(&m.bundle().predictions, &all_rows))
        .collect::<Result<_>>()?;
    let sel_loss = |counts: &[u32]| -> Result<f64> {
        let refs: Vec<&Predictions> = sel_members.iter().collect();
        let predicted = blend(&refs, counts)?;
        Ok(score_predictions(task, &classes, selection_labels, &predicted)?.loss)
    };

    // Greedy forward selection with replacement over the selection side.
    let mut counts = vec![0u32; sorted.len()];
    let mut best = f64::INFINITY;
    for _ in 0..MAX_ITERATIONS {
        let mut step: Option<(usize, f64)> = None;
        for i in 0..sorted.len() {
            counts[i] += 1;
            let loss = sel_loss(&counts)?;
            counts[i] -= 1;
            if step.is_none_or(|(_, best_so_far)| loss < best_so_far) {
                step = Some((i, loss));
            }
        }
        match step {
            Some((i, loss)) if loss < best => {
                counts[i] += 1;
                best = loss;
            }
            _ => break,
        }
    }

    let mut best_single: Option<(f64, usize)> = None;
    for i in 0..sorted.len() {
        let mut solo = vec![0u32; sorted.len()];
        solo[i] = 1;
        let loss = sel_loss(&solo)?;
        if best_single.is_none_or(|(best_loss, _)| loss < best_loss) {
            best_single = Some((loss, i));
        }
    }
    let (single_loss, single_index) = best_single.ok_or_else(|| anyhow::anyhow!("no members"))?;
    let distinct_weighted = counts.iter().filter(|&&c| c > 0).count();
    let disposition;
    let final_counts;
    if sorted.len() < 2 {
        disposition = EnsembleDisposition::NotAttempted {
            reason: "fewer than two eligible members".into(),
        };
        final_counts = counts.clone();
    } else if distinct_weighted < 2 {
        disposition = EnsembleDisposition::BestSingleFallback {
            reason: "fewer than two distinct members received weight".into(),
        };
        let mut solo = vec![0u32; sorted.len()];
        solo[single_index] = 1;
        final_counts = solo;
    } else if best >= single_loss {
        disposition = EnsembleDisposition::BestSingleFallback {
            reason: "blend did not beat the best single member on the selection side".into(),
        };
        let mut solo = vec![0u32; sorted.len()];
        solo[single_index] = 1;
        final_counts = solo;
    } else {
        disposition = EnsembleDisposition::Blend;
        final_counts = counts.clone();
    }

    let selection_loss = sel_loss(&final_counts)?;
    let report_members: Vec<&Predictions> = sorted
        .iter()
        .map(|m| {
            report
                .per_member
                .get(&m.bundle().candidate_id)
                .ok_or_else(|| anyhow::anyhow!("missing report predictions"))
        })
        .collect::<Result<_>>()?;
    let report_predicted = blend(&report_members, &final_counts)?;
    let report_loss = score_predictions(task, &classes, &report.labels, &report_predicted)?.loss;

    let mut weighted = BTreeMap::new();
    let mut shas = BTreeMap::new();
    for (member, count) in sorted.iter().zip(&final_counts) {
        shas.insert(
            member.bundle().candidate_id.clone(),
            member.bundle().identity()?,
        );
        weighted.insert(member.bundle().candidate_id.clone(), *count);
    }
    Ok(WinnerEnsemble {
        schema: ENSEMBLE_SCHEMA.into(),
        task,
        selection_rule: SELECTION_RULE.into(),
        identity: identity.clone(),
        counts: weighted,
        disposition,
        best_single: Some(sorted[single_index].bundle().candidate_id.clone()),
        selection_loss,
        report_loss,
        report_loss_basis: REPORT_BASIS.into(),
        member_bundle_sha256: shas,
    })
}

#[cfg(test)]
mod tests {
    use super::super::identity::{digest_ids, digest_pairs, ExecutionIdentity};
    use super::super::oof::{admit_for_test, OofPredictionBundle, OOF_SCHEMA};
    use super::*;

    const REQUEST: &str = "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1";
    const N: usize = 10;

    fn truth() -> Vec<f64> {
        (0..N).map(|i| i as f64 * 0.5).collect()
    }

    fn labels() -> Vec<String> {
        truth().iter().map(|t| t.to_string()).collect()
    }

    fn bundle(id: &str, values: Vec<f64>) -> OofPredictionBundle {
        OofPredictionBundle {
            schema: OOF_SCHEMA.into(),
            task: Task::Regression,
            candidate_id: id.into(),
            request_sha256: REQUEST.into(),
            fitted_artifact_sha256: vec!["b".repeat(64); 5],
            row_ids: (0..N).map(|i| format!("r{i}")).collect(),
            folds: (0..N).map(|i| (i % 5) as u32).collect(),
            group_ids: vec![],
            classes: vec![],
            predictions: Predictions::Values(values),
            failed_folds: vec![],
        }
    }

    fn identity(report_ids: &[String], report_labels: &[String]) -> ExecutionIdentity {
        let rows: Vec<String> = (0..N).map(|i| format!("r{i}")).collect();
        ExecutionIdentity::new(
            REQUEST,
            &"d".repeat(64),
            &"c".repeat(64),
            &rows,
            &labels(),
            &[],
            report_ids,
            report_labels,
            &[],
        )
        .unwrap()
    }

    /// Report side: four rows that are not in the selection side.
    fn report_fixture(
        values: &dyn Fn(&str) -> Vec<f64>,
    ) -> (Vec<String>, Vec<String>, ReportPredictions) {
        let ids: Vec<String> = (0..4).map(|i| format!("q{i}")).collect();
        let report_labels: Vec<String> = (0..4).map(|i| format!("{}", i as f64 * 0.5)).collect();
        let per_member = ["rust.a", "rust.b"]
            .iter()
            .map(|id| (id.to_string(), Predictions::Values(values(id))))
            .collect();
        (
            ids.clone(),
            report_labels.clone(),
            ReportPredictions {
                ids,
                labels: report_labels,
                per_member,
            },
        )
    }

    fn admitted(members: Vec<OofPredictionBundle>) -> Vec<AdmittedBundle> {
        members
            .into_iter()
            .map(|m| admit_for_test(m).unwrap())
            .collect()
    }

    fn complementary() -> Vec<OofPredictionBundle> {
        let t = truth();
        vec![
            bundle(
                "rust.a",
                t.iter()
                    .enumerate()
                    .map(|(i, v)| v + if i % 2 == 0 { 1.0 } else { -1.0 })
                    .collect(),
            ),
            bundle(
                "rust.b",
                t.iter()
                    .enumerate()
                    .map(|(i, v)| v + if i % 2 == 0 { -1.0 } else { 1.0 })
                    .collect(),
            ),
        ]
    }

    fn report_for_complementary() -> (ExecutionIdentity, ReportPredictions) {
        let (ids, rlabels, report) = report_fixture(&|id| {
            if id == "rust.a" {
                vec![1.0, -1.0, 1.0, -1.0]
                    .into_iter()
                    .enumerate()
                    .map(|(i, d)| i as f64 * 0.5 + d)
                    .collect()
            } else {
                vec![1.0, -1.0, 1.0, -1.0]
                    .into_iter()
                    .enumerate()
                    .map(|(i, d)| i as f64 * 0.5 - d)
                    .collect()
            }
        });
        (identity(&ids, &rlabels), report)
    }

    #[test]
    fn a_useful_blend_is_accepted_and_the_report_loss_is_on_the_report_side() {
        let (id, report) = report_for_complementary();
        let out = build_winner_ensemble(
            &admitted(complementary()),
            &labels(),
            &id,
            &report,
            &mut EvaluationBudget::new(3),
        )
        .unwrap();
        assert_eq!(out.disposition, EnsembleDisposition::Blend);
        assert!(out.selection_loss < 1e-9);
        assert!(out.report_loss < 1e-9, "report loss {}", out.report_loss);
        assert_eq!(out.report_loss_basis, REPORT_BASIS);
    }

    #[test]
    fn outer_label_poison_changes_only_the_report_loss() {
        let (id, report) = report_for_complementary();
        let clean = build_winner_ensemble(
            &admitted(complementary()),
            &labels(),
            &id,
            &report,
            &mut EvaluationBudget::new(3),
        )
        .unwrap();
        let poisoned_labels: Vec<String> = vec!["90".into(), "-40".into(), "7".into(), "3".into()];
        let poisoned_id = identity(&report.ids, &poisoned_labels);
        let poisoned_report = ReportPredictions {
            labels: poisoned_labels,
            ..report.clone()
        };
        let poisoned = build_winner_ensemble(
            &admitted(complementary()),
            &labels(),
            &poisoned_id,
            &poisoned_report,
            &mut EvaluationBudget::new(3),
        )
        .unwrap();
        assert_eq!(
            clean.counts, poisoned.counts,
            "report labels must not move the weights"
        );
        assert_eq!(clean.disposition, poisoned.disposition);
        assert_eq!(clean.selection_loss, poisoned.selection_loss);
        assert_ne!(clean.report_loss, poisoned.report_loss);
    }

    #[test]
    fn a_single_useful_member_falls_back_explicitly() {
        let t = truth();
        let members = vec![
            bundle("rust.good", t.clone()),
            bundle("rust.bad", t.iter().map(|v| v + 100.0).collect()),
        ];
        let (ids, rlabels, report) = report_fixture(&|id| {
            if id == "rust.good" {
                vec![0.0, 0.5, 1.0, 1.5]
            } else {
                vec![100.0; 4]
            }
        });
        let out = build_winner_ensemble(
            &admitted(members),
            &labels(),
            &identity(&ids, &rlabels),
            &report,
            &mut EvaluationBudget::new(3),
        )
        .unwrap();
        assert!(matches!(
            out.disposition,
            EnsembleDisposition::BestSingleFallback { .. }
        ));
        assert_eq!(out.counts.get("rust.good"), Some(&1));
    }

    #[test]
    fn a_lone_member_is_not_attempted() {
        let members = vec![bundle("rust.only", truth())];
        let (ids, rlabels, report) = report_fixture(&|_| vec![0.0, 0.5, 1.0, 1.5]);
        let mut only = report.clone();
        only.per_member.retain(|k, _| k == "rust.a");
        let out = build_winner_ensemble(
            &admitted(members),
            &labels(),
            &identity(&ids, &rlabels),
            &only,
            &mut EvaluationBudget::new(3),
        );
        // The report predictions must cover exactly the members, so the lone member is keyed rust.only.
        assert!(out.is_err());
        let mut lone = report.clone();
        lone.per_member.clear();
        lone.per_member.insert(
            "rust.only".into(),
            Predictions::Values(vec![0.0, 0.5, 1.0, 1.5]),
        );
        let out = build_winner_ensemble(
            &admitted(vec![bundle("rust.only", truth())]),
            &labels(),
            &identity(&ids, &rlabels),
            &lone,
            &mut EvaluationBudget::new(3),
        )
        .unwrap();
        assert!(matches!(
            out.disposition,
            EnsembleDisposition::NotAttempted { .. }
        ));
    }

    #[test]
    fn a_different_request_is_refused_by_the_shared_identity() {
        let (id, report) = report_for_complementary();
        let mut members = complementary();
        members[1].request_sha256 = "d".repeat(64);
        assert!(build_winner_ensemble(
            &admitted(members),
            &labels(),
            &id,
            &report,
            &mut EvaluationBudget::new(3)
        )
        .is_err());
    }

    #[test]
    fn a_different_group_identity_is_refused_by_the_shared_identity() {
        let (id, report) = report_for_complementary();
        let mut members = complementary();
        members[0].group_ids = (0..N).map(|i| format!("g{i}")).collect();
        assert!(build_winner_ensemble(
            &admitted(members),
            &labels(),
            &id,
            &report,
            &mut EvaluationBudget::new(3)
        )
        .is_err());
    }

    #[test]
    fn changed_selection_labels_are_refused_by_the_cohort_identity() {
        let (id, report) = report_for_complementary();
        let mut changed = labels();
        changed[4] = "999".into();
        assert!(build_winner_ensemble(
            &admitted(complementary()),
            &changed,
            &id,
            &report,
            &mut EvaluationBudget::new(3)
        )
        .is_err());
    }

    #[test]
    fn report_rows_that_do_not_match_the_identity_are_refused() {
        let (id, mut report) = report_for_complementary();
        report.ids[0] = "other".into();
        assert!(build_winner_ensemble(
            &admitted(complementary()),
            &labels(),
            &id,
            &report,
            &mut EvaluationBudget::new(3)
        )
        .is_err());
    }

    #[test]
    fn the_ensemble_attempt_is_charged_against_the_request_budget() {
        let (id, report) = report_for_complementary();
        let mut budget = EvaluationBudget::new(0);
        assert!(build_winner_ensemble(
            &admitted(complementary()),
            &labels(),
            &id,
            &report,
            &mut budget
        )
        .is_err());
        assert_eq!(budget.charged(), 0);
    }

    #[test]
    fn the_ensemble_is_deterministic_and_member_order_does_not_matter() {
        let (id, report) = report_for_complementary();
        let mut reversed = complementary();
        reversed.reverse();
        let first = build_winner_ensemble(
            &admitted(complementary()),
            &labels(),
            &id,
            &report,
            &mut EvaluationBudget::new(3),
        )
        .unwrap();
        let second = build_winner_ensemble(
            &admitted(reversed),
            &labels(),
            &id,
            &report,
            &mut EvaluationBudget::new(3),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(digest_ids(&[]), digest_ids(&[]));
        assert_eq!(digest_pairs(&[], &[]), digest_pairs(&[], &[]));
    }
}
