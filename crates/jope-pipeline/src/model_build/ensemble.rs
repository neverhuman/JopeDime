//! Post-search winner ensemble from out-of-fold predictions (SC-DEC-184 JA-04).
//!
//! The blend is a bounded, deterministic, non-negative weighted average of member predictions.
//! Its weights are chosen by greedy forward selection with replacement, scored only on the
//! selection half of the folds. The report half is never used to choose weights, so the
//! reported loss is not the loss the weights were fitted to. An attempt is accepted only when the
//! blend beats the best single member on the selection half and at least two distinct members
//! carry weight. Otherwise the disposition is an explicit best-single fallback, or not attempted.
use super::oof::OofPredictionBundle;
use crate::evaluation_score::{score_predictions, validate_probabilities, Predictions};
use anyhow::{bail, ensure, Result};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const ENSEMBLE_SCHEMA: &str = "jope.winner-ensemble/v1";
pub const SELECTION_RULE: &str = "greedy forward selection with replacement over non-negative counts, scored on the selection half of the folds only; accepted when the blend beats the best single member on that half and at least two distinct members carry weight; otherwise best-single fallback or not attempted";
/// Upper bound on greedy steps. Each step adds one unit of weight.
pub const MAX_ITERATIONS: u32 = 25;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "disposition", rename_all = "snake_case", deny_unknown_fields)]
pub enum EnsembleDisposition {
    Blend,
    BestSingleFallback { reason: String },
    NotAttempted { reason: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WinnerEnsemble {
    pub schema: String,
    pub task: Task,
    pub selection_rule: String,
    pub selection_folds: Vec<u32>,
    pub report_folds: Vec<u32>,
    /// Integer weight per member candidate. Weights are counts divided by their total.
    pub counts: BTreeMap<String, u32>,
    pub disposition: EnsembleDisposition,
    pub best_single: Option<String>,
    /// Loss of the final weights on the selection half. Used for acceptance.
    pub selection_loss: f64,
    /// Loss of the final weights on the report half. Not used to choose weights.
    pub report_loss: f64,
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

/// Splits the sorted distinct folds into a selection half and a report half. Both are non-empty.
pub fn split_folds(folds: &[u32]) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut distinct: Vec<u32> = folds.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    ensure!(
        distinct.len() >= 2,
        "winner ensemble needs at least two folds"
    );
    let split = distinct.len().div_ceil(2);
    Ok((distinct[..split].to_vec(), distinct[split..].to_vec()))
}

fn rows_in(folds: &[u32], wanted: &[u32]) -> Vec<usize> {
    (0..folds.len())
        .filter(|&i| wanted.contains(&folds[i]))
        .collect()
}

/// Builds the winner ensemble from bundles that already share one row set, fold assignment,
/// task and class order. `labels` are aligned with those rows.
pub fn build_winner_ensemble(
    members: &[OofPredictionBundle],
    labels: &[String],
) -> Result<WinnerEnsemble> {
    ensure!(!members.is_empty(), "no member bundles");
    let first = &members[0];
    for member in members {
        member.validate()?;
        ensure!(member.task == first.task, "member tasks differ");
        ensure!(member.row_ids == first.row_ids, "member rows differ");
        ensure!(member.folds == first.folds, "member folds differ");
        ensure!(
            member.classes == first.classes,
            "member class orders differ"
        );
    }
    ensure!(
        labels.len() == first.row_ids.len(),
        "labels are not row-aligned"
    );
    let mut sorted: Vec<&OofPredictionBundle> = members.iter().collect();
    sorted.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
    for pair in sorted.windows(2) {
        ensure!(
            pair[0].candidate_id != pair[1].candidate_id,
            "duplicate member candidate"
        );
    }
    let (selection_folds, report_folds) = split_folds(&first.folds)?;
    let sel_rows = rows_in(&first.folds, &selection_folds);
    let rep_rows = rows_in(&first.folds, &report_folds);
    let sel_labels: Vec<String> = sel_rows.iter().map(|&i| labels[i].clone()).collect();
    let rep_labels: Vec<String> = rep_rows.iter().map(|&i| labels[i].clone()).collect();
    let task = first.task;
    let classes = first.classes.clone();
    let sel_members: Vec<Predictions> = sorted
        .iter()
        .map(|m| subset(&m.predictions, &sel_rows))
        .collect::<Result<_>>()?;
    let sel_loss = |counts: &[u32]| -> Result<f64> {
        let refs: Vec<&Predictions> = sel_members.iter().collect();
        let predicted = blend(&refs, counts)?;
        Ok(score_predictions(task, &classes, &sel_labels, &predicted)?.loss)
    };

    // Greedy forward selection with replacement over the selection half.
    let mut counts = vec![0u32; sorted.len()];
    let mut best = f64::INFINITY;
    for _ in 0..MAX_ITERATIONS {
        let mut step: Option<(usize, f64)> = None;
        for i in 0..sorted.len() {
            counts[i] += 1;
            let loss = sel_loss(&counts)?;
            counts[i] -= 1;
            if step.map_or(true, |(_, best_so_far)| loss < best_so_far) {
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

    // Best single member on the selection half, for acceptance and fallback.
    let mut best_single: Option<(f64, usize)> = None;
    for i in 0..sorted.len() {
        let mut solo = vec![0u32; sorted.len()];
        solo[i] = 1;
        let loss = sel_loss(&solo)?;
        if best_single.map_or(true, |(best_loss, _)| loss < best_loss) {
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
            reason: "blend did not beat the best single member on the selection half".into(),
        };
        let mut solo = vec![0u32; sorted.len()];
        solo[single_index] = 1;
        final_counts = solo;
    } else {
        disposition = EnsembleDisposition::Blend;
        final_counts = counts.clone();
    }

    let selection_loss = sel_loss(&final_counts)?;
    let rep_members: Vec<Predictions> = sorted
        .iter()
        .map(|m| subset(&m.predictions, &rep_rows))
        .collect::<Result<_>>()?;
    let refs: Vec<&Predictions> = rep_members.iter().collect();
    let report_predicted = blend(&refs, &final_counts)?;
    let report_loss = score_predictions(task, &classes, &rep_labels, &report_predicted)?.loss;

    let mut weighted = BTreeMap::new();
    let mut shas = BTreeMap::new();
    for (member, count) in sorted.iter().zip(&final_counts) {
        shas.insert(member.candidate_id.clone(), member.identity()?);
        weighted.insert(member.candidate_id.clone(), *count);
    }
    let best_single_id = sorted[single_index].candidate_id.clone();
    Ok(WinnerEnsemble {
        schema: ENSEMBLE_SCHEMA.into(),
        task,
        selection_rule: SELECTION_RULE.into(),
        selection_folds,
        report_folds,
        counts: weighted,
        disposition,
        best_single: Some(best_single_id),
        selection_loss,
        report_loss,
        member_bundle_sha256: shas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_build::oof::OOF_SCHEMA;

    fn bundle(id: &str, values: Vec<f64>) -> OofPredictionBundle {
        let n = values.len();
        OofPredictionBundle {
            schema: OOF_SCHEMA.into(),
            task: Task::Regression,
            candidate_id: id.into(),
            request_sha256: "a".repeat(64),
            fitted_artifact_sha256: vec!["b".repeat(64); 4],
            row_ids: (0..n).map(|i| format!("r{i}")).collect(),
            folds: (0..n).map(|i| (i % 4) as u32).collect(),
            group_ids: vec![],
            classes: vec![],
            predictions: Predictions::Values(values),
            failed_folds: vec![],
        }
    }

    fn labels_for(n: usize) -> Vec<String> {
        (0..n).map(|i| (i as f64 * 0.5).to_string()).collect()
    }

    #[test]
    fn folds_split_into_two_non_empty_halves() {
        let (sel, rep) = split_folds(&[0, 1, 2, 3, 0, 1]).unwrap();
        assert_eq!(sel, vec![0, 1]);
        assert_eq!(rep, vec![2, 3]);
        assert!(split_folds(&[0, 0, 0]).is_err());
    }

    #[test]
    fn a_useful_blend_is_accepted_with_non_negative_counts_summing_to_the_total() {
        let n = 16;
        let labels = labels_for(n);
        let truth: Vec<f64> = labels.iter().map(|l| l.parse().unwrap()).collect();
        // Two complementary errors: their average is exact, so the blend must win.
        let a: Vec<f64> = truth
            .iter()
            .enumerate()
            .map(|(i, t)| t + if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let b: Vec<f64> = truth
            .iter()
            .enumerate()
            .map(|(i, t)| t + if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let members = vec![bundle("rust.a", a), bundle("rust.b", b)];
        let ensemble = build_winner_ensemble(&members, &labels).unwrap();
        assert_eq!(ensemble.disposition, EnsembleDisposition::Blend);
        assert!(
            ensemble.report_loss < 1e-9,
            "report loss {}",
            ensemble.report_loss
        );
        assert_eq!(
            ensemble.counts.values().map(|c| *c as u64).sum::<u64>() > 0,
            true
        );
    }

    #[test]
    fn a_single_useful_member_falls_back_explicitly() {
        let n = 16;
        let labels = labels_for(n);
        let truth: Vec<f64> = labels.iter().map(|l| l.parse().unwrap()).collect();
        let good = bundle("rust.good", truth.clone());
        let bad = bundle("rust.bad", truth.iter().map(|t| t + 100.0).collect());
        let ensemble = build_winner_ensemble(&[good, bad], &labels).unwrap();
        assert!(matches!(
            ensemble.disposition,
            EnsembleDisposition::BestSingleFallback { .. }
        ));
        assert_eq!(ensemble.counts.get("rust.good"), Some(&1));
        assert_eq!(ensemble.counts.get("rust.bad"), Some(&0));
    }

    #[test]
    fn a_lone_member_is_not_attempted_and_nothing_is_substituted() {
        let n = 8;
        let labels = labels_for(n);
        let truth: Vec<f64> = labels.iter().map(|l| l.parse().unwrap()).collect();
        let ensemble = build_winner_ensemble(&[bundle("rust.only", truth)], &labels).unwrap();
        assert!(matches!(
            ensemble.disposition,
            EnsembleDisposition::NotAttempted { .. }
        ));
    }

    #[test]
    fn misaligned_member_rows_are_refused() {
        let labels = labels_for(4);
        let a = bundle("rust.a", vec![0.0, 0.5, 1.0, 1.5]);
        let mut b = bundle("rust.b", vec![0.0, 0.5, 1.0, 1.5]);
        b.row_ids[0] = "other".into();
        assert!(build_winner_ensemble(&[a, b], &labels).is_err());
    }

    #[test]
    fn the_ensemble_is_deterministic_and_member_order_does_not_matter() {
        let labels = labels_for(12);
        let truth: Vec<f64> = labels.iter().map(|l| l.parse().unwrap()).collect();
        let a = bundle("rust.a", truth.iter().map(|t| t + 0.3).collect());
        let b = bundle("rust.b", truth.iter().map(|t| t - 0.2).collect());
        let first = build_winner_ensemble(&[a.clone(), b.clone()], &labels).unwrap();
        let second = build_winner_ensemble(&[b, a], &labels).unwrap();
        assert_eq!(first, second);
    }
}
