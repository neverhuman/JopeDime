use std::collections::{BTreeMap, BTreeSet};

use super::{
    ArmComparisonV1, Best5ArmV1, CampaignDecisionV1, ComparisonScoreV1, ComparisonV1,
    DatasetStageV1, FailureCodeV1, BOOTSTRAP_SEED,
};
use crate::jope_generalized_gp::raw_sha256;
use crate::search_v3::validate_sha256;

fn deterministic_index(seed: u64, repetition: usize, draw: usize, width: usize) -> usize {
    let digest =
        raw_sha256(format!("jope-dlgp-best5-bootstrap|{seed}|{repetition}|{draw}").as_bytes());
    u64::from_be_bytes(hex::decode(&digest[..16]).unwrap().try_into().unwrap()) as usize % width
}

pub fn deterministic_lower_bound(
    deltas: &[f64],
    seed: u64,
    repetitions: usize,
) -> Result<f64, String> {
    if deltas.is_empty() || repetitions == 0 || deltas.iter().any(|delta| !delta.is_finite()) {
        return Err("bootstrap requires finite paired dataset deltas".into());
    }
    let mut samples = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        samples.push(
            (0..deltas.len())
                .map(|draw| deltas[deterministic_index(seed, repetition, draw, deltas.len())])
                .sum::<f64>()
                / deltas.len() as f64,
        );
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[((repetitions as f64) * 0.05).floor() as usize].min(samples[repetitions - 1]))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    }
}

pub fn compare_campaign(
    campaign_freeze_sha256: &str,
    stage: DatasetStageV1,
    expected_dataset_sha256s: &[String],
    scores: &[ComparisonScoreV1],
) -> Result<ComparisonV1, String> {
    validate_sha256("comparison campaign", campaign_freeze_sha256)?;
    let expected_count = match stage {
        DatasetStageV1::Pilot => 20,
        DatasetStageV1::Confirmation => 22,
    };
    if expected_dataset_sha256s.len() != expected_count
        || expected_dataset_sha256s
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != expected_count
    {
        return Err("expected comparison dataset freeze has the wrong cardinality".into());
    }
    for dataset in expected_dataset_sha256s {
        validate_sha256("expected comparison dataset", dataset)?;
    }
    let expected_datasets = expected_dataset_sha256s.iter().collect::<BTreeSet<_>>();
    let mut by_key = BTreeMap::new();
    for score in scores {
        score.validate()?;
        if !expected_datasets.contains(&score.dataset_sha256)
            || by_key
                .insert((score.arm, score.dataset_sha256.as_str()), score)
                .is_some()
        {
            return Err("comparison score is unexpected or duplicated".into());
        }
    }
    let controls = scores
        .iter()
        .filter(|score| score.arm == Best5ArmV1::ConfidenceGpControl)
        .collect::<Vec<_>>();
    if controls.len() != expected_count
        || controls
            .iter()
            .map(|score| &score.dataset_sha256)
            .collect::<BTreeSet<_>>()
            != expected_datasets
    {
        return Err("incomplete_sealed_results: matched control coverage is incomplete".into());
    }
    let arms = match stage {
        DatasetStageV1::Pilot => Best5ArmV1::EXPERIMENTS.to_vec(),
        DatasetStageV1::Confirmation => {
            let experiments = scores
                .iter()
                .filter(|score| score.arm.is_experiment())
                .map(|score| score.arm)
                .collect::<BTreeSet<_>>();
            if experiments.len() != 1 {
                return Err("confirmation must contain one frozen pilot winner".into());
            }
            experiments.into_iter().collect()
        }
    };
    let mut summaries = Vec::with_capacity(arms.len());
    let mut aggregate_failures = BTreeSet::new();
    for arm in arms {
        let arm_scores = scores
            .iter()
            .filter(|score| score.arm == arm)
            .collect::<Vec<_>>();
        if arm_scores.len() != expected_count
            || arm_scores
                .iter()
                .map(|score| &score.dataset_sha256)
                .collect::<BTreeSet<_>>()
                != expected_datasets
        {
            return Err(format!(
                "incomplete_sealed_results: {arm:?} lacks complete paired coverage"
            ));
        }
        let mut deltas = Vec::with_capacity(expected_count);
        let mut paired_wins = 0u32;
        let mut latency_gate = true;
        for arm_score in &arm_scores {
            let control = by_key[&(
                Best5ArmV1::ConfidenceGpControl,
                arm_score.dataset_sha256.as_str(),
            )];
            let delta = (control.rmse / arm_score.rmse).ln();
            deltas.push(delta);
            paired_wins += (delta > 0.0) as u32;
            latency_gate &= arm_score.latency_seconds <= 4.0 * control.latency_seconds;
        }
        let lower_bound = deterministic_lower_bound(&deltas, BOOTSTRAP_SEED, 10_000)?;
        let center = median(deltas.clone());
        let finalist_gate = arm_scores
            .iter()
            .all(|score| score.finalist_validity == 1.0);
        let overall_gate = mean(
            &arm_scores
                .iter()
                .map(|score| score.overall_validity)
                .collect::<Vec<_>>(),
        ) >= 0.99;
        let win_preservation = arm_scores.iter().all(|score| score.preserved_existing_win);
        let paired_win_gate = stage == DatasetStageV1::Pilot || paired_wins >= 11;
        let mut failure_codes = BTreeSet::new();
        if lower_bound <= 0.0 {
            failure_codes.insert(FailureCodeV1::NonpositiveLowerBound);
        }
        if center < 0.0 {
            failure_codes.insert(FailureCodeV1::NegativeMedian);
        }
        if !finalist_gate {
            failure_codes.insert(FailureCodeV1::FinalistValidity);
        }
        if !overall_gate {
            failure_codes.insert(FailureCodeV1::OverallValidity);
        }
        if !win_preservation {
            failure_codes.insert(if stage == DatasetStageV1::Pilot {
                FailureCodeV1::ExistingBenchmarkWinLost
            } else {
                FailureCodeV1::ConfirmationWinPreservation
            });
        }
        if !latency_gate {
            failure_codes.insert(FailureCodeV1::LatencyGate);
        }
        if !paired_win_gate {
            failure_codes.insert(FailureCodeV1::ConfirmationPairedWins);
        }
        aggregate_failures.extend(failure_codes.iter().cloned());
        summaries.push(ArmComparisonV1 {
            arm,
            mean_delta: mean(&deltas),
            median_delta: center,
            one_sided_95_lower_bound: lower_bound,
            paired_wins,
            strict_benchmark_wins: arm_scores
                .iter()
                .filter(|score| score.strict_benchmark_win)
                .count() as u32,
            predicted_cpu_seconds: arm_scores
                .iter()
                .map(|score| score.predicted_cpu_seconds)
                .sum(),
            advances: failure_codes.is_empty(),
            failure_codes: failure_codes.into_iter().collect(),
        });
    }
    let advancing = summaries
        .iter()
        .filter(|summary| summary.advances)
        .collect::<Vec<_>>();
    let decision = if advancing.is_empty() {
        if aggregate_failures.is_empty() {
            aggregate_failures.insert(FailureCodeV1::NonpositiveLowerBound);
        }
        CampaignDecisionV1::NoImprovement {
            failure_codes: aggregate_failures.into_iter().collect(),
        }
    } else if stage == DatasetStageV1::Confirmation {
        CampaignDecisionV1::Adopt {
            arm: advancing[0].arm,
        }
    } else {
        let best_lower_bound = advancing
            .iter()
            .map(|summary| summary.one_sided_95_lower_bound)
            .max_by(f64::total_cmp)
            .unwrap();
        let winner = advancing
            .into_iter()
            .filter(|summary| best_lower_bound - summary.one_sided_95_lower_bound <= 0.002)
            .min_by(|left, right| {
                right
                    .strict_benchmark_wins
                    .cmp(&left.strict_benchmark_wins)
                    .then_with(|| {
                        left.predicted_cpu_seconds
                            .total_cmp(&right.predicted_cpu_seconds)
                    })
                    .then_with(|| {
                        left.arm
                            .simplicity_order()
                            .cmp(&right.arm.simplicity_order())
                    })
            })
            .unwrap();
        CampaignDecisionV1::PilotWinner { arm: winner.arm }
    };
    let comparison = ComparisonV1 {
        schema: ComparisonV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign_freeze_sha256.into(),
        stage,
        bootstrap_seed: BOOTSTRAP_SEED,
        bootstrap_repetitions: 10_000,
        dataset_count: expected_count as u32,
        arms: summaries,
        decision,
    };
    comparison.validate()?;
    Ok(comparison)
}
