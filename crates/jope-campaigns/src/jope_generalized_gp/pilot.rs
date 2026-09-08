use std::collections::{BTreeMap, BTreeSet};

use super::contracts::*;

fn deterministic_index(
    seed_sha256: &str,
    stream: &str,
    repetition: usize,
    draw: usize,
    n: usize,
) -> usize {
    let digest =
        raw_sha256(format!("bootstrap-v1|{seed_sha256}|{stream}|{repetition}|{draw}").as_bytes());
    let value = u64::from_be_bytes(hex::decode(&digest[..16]).unwrap().try_into().unwrap());
    value as usize % n
}

pub fn deterministic_bootstrap_lower_bound(
    regression_deltas: &[f64],
    binary_deltas: &[f64],
    bootstrap_seed_sha256: &str,
    repetitions: usize,
) -> Result<f64, String> {
    crate::search_v3::validate_sha256("bootstrap seed", bootstrap_seed_sha256)?;
    if regression_deltas.is_empty()
        || binary_deltas.is_empty()
        || repetitions == 0
        || regression_deltas
            .iter()
            .chain(binary_deltas)
            .any(|value| !value.is_finite())
    {
        return Err("bootstrap requires finite deltas for both tasks".into());
    }
    let mut samples = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let regression = (0..regression_deltas.len())
            .map(|draw| {
                regression_deltas[deterministic_index(
                    bootstrap_seed_sha256,
                    "regression",
                    repetition,
                    draw,
                    regression_deltas.len(),
                )]
            })
            .sum::<f64>()
            / regression_deltas.len() as f64;
        let binary = (0..binary_deltas.len())
            .map(|draw| {
                binary_deltas[deterministic_index(
                    bootstrap_seed_sha256,
                    "binary",
                    repetition,
                    draw,
                    binary_deltas.len(),
                )]
            })
            .sum::<f64>()
            / binary_deltas.len() as f64;
        samples.push(0.5 * (regression + binary));
    }
    samples.sort_by(f64::total_cmp);
    let index = ((repetitions as f64) * 0.05).floor() as usize;
    Ok(samples[index.min(repetitions - 1)])
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile95(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64) * 0.95).ceil() as usize;
    values[index.saturating_sub(1).min(values.len() - 1)]
}

pub fn compare_pilot(
    campaign_freeze_sha256: &str,
    bootstrap_seed_sha256: &str,
    results: &[PilotDatasetResultV1],
) -> Result<PilotComparisonV1, String> {
    crate::search_v3::validate_sha256("pilot campaign", campaign_freeze_sha256)?;
    crate::search_v3::validate_sha256("pilot bootstrap seed", bootstrap_seed_sha256)?;
    let mut by_key = BTreeMap::new();
    for result in results {
        result.validate()?;
        if by_key
            .insert((result.arm, result.dataset_sha256.as_str()), result)
            .is_some()
        {
            return Err("pilot results contain a duplicate arm/dataset pair".into());
        }
    }
    let control: Vec<_> = results
        .iter()
        .filter(|result| result.arm == CampaignArmV1::LiteRaceControl)
        .collect();
    let control_datasets: BTreeSet<_> = control
        .iter()
        .map(|result| (result.dataset_sha256.as_str(), result.task))
        .collect();
    if control_datasets.len() != 20
        || control_datasets
            .iter()
            .filter(|(_, task)| *task == GpTaskV1::Regression)
            .count()
            != 10
        || control_datasets
            .iter()
            .filter(|(_, task)| *task == GpTaskV1::Binary)
            .count()
            != 10
    {
        return Err("matched control must cover ten datasets per task".into());
    }
    let control_cpu_p95 = percentile95(control.iter().map(|result| result.cpu_seconds).collect());
    let control_latency_p95 = percentile95(
        control
            .iter()
            .map(|result| result.winner_latency_seconds)
            .collect(),
    );
    let mut summaries = Vec::new();
    let mut reason_codes = BTreeSet::new();
    for arm in CampaignArmV1::EXPERIMENTS {
        let arm_results: Vec<_> = results.iter().filter(|result| result.arm == arm).collect();
        let arm_datasets: BTreeSet<_> = arm_results
            .iter()
            .map(|result| (result.dataset_sha256.as_str(), result.task))
            .collect();
        if arm_results.len() != 20 || arm_datasets != control_datasets {
            return Err(format!(
                "{arm:?} is not paired to the complete control panel"
            ));
        }
        let mut regression = Vec::with_capacity(10);
        let mut binary = Vec::with_capacity(10);
        for result in &arm_results {
            let matched = by_key
                .get(&(
                    CampaignArmV1::LiteRaceControl,
                    result.dataset_sha256.as_str(),
                ))
                .unwrap();
            if matched.task != result.task {
                return Err("pilot task changed between arm and control".into());
            }
            let delta = result.sealed_skill - matched.sealed_skill;
            match result.task {
                GpTaskV1::Regression => regression.push(delta),
                GpTaskV1::Binary => binary.push(delta),
            }
        }
        let mean_regression_delta = mean(&regression);
        let mean_binary_delta = mean(&binary);
        let lower_bound = deterministic_bootstrap_lower_bound(
            &regression,
            &binary,
            &raw_sha256(format!("{bootstrap_seed_sha256}|{arm:?}").as_bytes()),
            10_000,
        )?;
        let validity_gate = arm_results.iter().all(|result| {
            result.generation_zero_validity == 1.0 && result.finalist_validity == 1.0
        }) && mean(
            &arm_results
                .iter()
                .map(|result| result.overall_validity)
                .collect::<Vec<_>>(),
        ) >= 0.99;
        let sealed_winner_gate = arm_results.iter().all(|result| result.sealed_winner);
        let cpu_p95 = percentile95(
            arm_results
                .iter()
                .map(|result| result.cpu_seconds)
                .collect(),
        );
        let latency_p95 = percentile95(
            arm_results
                .iter()
                .map(|result| result.winner_latency_seconds)
                .collect(),
        );
        let runtime_gate =
            cpu_p95 <= 1.25 * control_cpu_p95 && latency_p95 <= 1.25 * control_latency_p95;
        let advances = lower_bound > 0.0
            && mean_regression_delta >= 0.0
            && mean_binary_delta >= 0.0
            && validity_gate
            && sealed_winner_gate
            && runtime_gate;
        if !advances {
            if lower_bound <= 0.0 {
                reason_codes.insert(format!("{:?}:nonpositive_lower_bound", arm).to_lowercase());
            }
            if mean_regression_delta < 0.0 || mean_binary_delta < 0.0 {
                reason_codes.insert(format!("{:?}:negative_task_mean", arm).to_lowercase());
            }
            if !validity_gate {
                reason_codes.insert(format!("{:?}:validity_gate", arm).to_lowercase());
            }
            if !sealed_winner_gate {
                reason_codes.insert(format!("{:?}:missing_sealed_winner", arm).to_lowercase());
            }
            if !runtime_gate {
                reason_codes.insert(format!("{:?}:runtime_gate", arm).to_lowercase());
            }
        }
        summaries.push(PilotArmSummaryV1 {
            arm,
            mean_regression_delta,
            mean_binary_delta,
            equal_task_mean_delta: 0.5 * (mean_regression_delta + mean_binary_delta),
            one_sided_95_lower_bound: lower_bound,
            predicted_cpu_seconds: arm_results.iter().map(|result| result.cpu_seconds).sum(),
            validity_gate,
            sealed_winner_gate,
            runtime_gate,
            advances,
        });
    }
    let advancing: Vec<_> = summaries
        .iter()
        .filter(|summary| summary.advances)
        .collect();
    let decision = if advancing.is_empty() {
        if reason_codes.is_empty() {
            reason_codes.insert("no_arm_cleared_all_gates".into());
        }
        let receipt = NoImprovementV1 {
            schema: NoImprovementV1::SCHEMA.into(),
            campaign_freeze_sha256: campaign_freeze_sha256.into(),
            reason_codes: reason_codes.into_iter().collect(),
        };
        receipt.validate()?;
        PilotDecisionV1::NoImprovement {
            receipt_sha256: receipt.identity()?,
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
                left.predicted_cpu_seconds
                    .total_cmp(&right.predicted_cpu_seconds)
                    .then_with(|| {
                        left.arm
                            .simplicity_order()
                            .cmp(&right.arm.simplicity_order())
                    })
            })
            .unwrap();
        PilotDecisionV1::Advance { arm: winner.arm }
    };
    let comparison = PilotComparisonV1 {
        schema: PilotComparisonV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign_freeze_sha256.into(),
        bootstrap_repetitions: 10_000,
        arms: summaries,
        decision,
    };
    comparison.validate()?;
    Ok(comparison)
}
