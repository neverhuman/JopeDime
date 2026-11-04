use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionSeedConfirmationV1 {
    pub seed: u64,
    pub raw_rmse: f64,
    pub gated_rmse: f64,
    pub raw_latency_seconds: f64,
    pub gated_latency_seconds: f64,
    pub gated_validity: f64,
    pub raw_receipt_sha256: String,
    pub gated_receipt_sha256: String,
    pub raw_fallback: bool,
}

impl HyperionSeedConfirmationV1 {
    fn validate(&self) -> Result<(), String> {
        if !CONFIRMATION_SEEDS.contains(&self.seed)
            || !self.raw_rmse.is_finite()
            || !self.gated_rmse.is_finite()
            || self.raw_rmse <= 0.0
            || self.gated_rmse <= 0.0
            || !self.raw_latency_seconds.is_finite()
            || !self.gated_latency_seconds.is_finite()
            || self.raw_latency_seconds < 0.0
            || self.gated_latency_seconds < 0.0
            || !self.gated_validity.is_finite()
            || !(0.0..=1.0).contains(&self.gated_validity)
            || self.gated_latency_seconds > self.raw_latency_seconds * 4.0
        {
            return Err("Hyperion confirmation seed metrics are invalid".into());
        }
        validate_sha256("raw confirmation receipt", &self.raw_receipt_sha256)?;
        validate_sha256("gated confirmation receipt", &self.gated_receipt_sha256)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionConfirmationDatasetV1 {
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub raw_robust_win: bool,
    pub gated_robust_win: bool,
    pub seeds: Vec<HyperionSeedConfirmationV1>,
    pub paired_delta: f64,
    pub sealed_finalizer_receipt_sha256: String,
    pub test_access_ordinal: u32,
}

impl HyperionConfirmationDatasetV1 {
    fn validate(&self) -> Result<(), String> {
        if !CONFIRMATION_DATASET_IDS.contains(&self.dataset_id.as_str())
            || !self.dataset_sha256.starts_with(&self.dataset_id)
            || self.seeds.len() != CONFIRMATION_SEEDS.len()
            || self.test_access_ordinal != 1
            || !self.paired_delta.is_finite()
            || !self
                .seeds
                .iter()
                .enumerate()
                .all(|(index, seed)| seed.seed == CONFIRMATION_SEEDS[index])
        {
            return Err("Hyperion confirmation dataset coverage is invalid".into());
        }
        for seed in &self.seeds {
            seed.validate()?;
        }
        let expected = self
            .seeds
            .iter()
            .map(|seed| (seed.raw_rmse / seed.gated_rmse).ln())
            .sum::<f64>()
            / self.seeds.len() as f64;
        if (expected - self.paired_delta).abs() > 1e-12 {
            return Err("paired confirmation delta is not the frozen two-seed log ratio".into());
        }
        validate_sha256(
            "sealed finalizer receipt",
            &self.sealed_finalizer_receipt_sha256,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionConfirmationReportV1 {
    pub schema: String,
    pub policy_freeze_sha256: String,
    pub selection_inventory_sha256: String,
    pub datasets: Vec<HyperionConfirmationDatasetV1>,
    pub raw_robust_wins: u32,
    pub gated_robust_wins: u32,
    pub positive_delta_datasets: u32,
    pub median_delta: f64,
    pub bootstrap_lower_bound: f64,
    pub aggregate_validity: f64,
    pub claim_increased_win_rate: bool,
    pub complete_immutable_receipts: bool,
    pub labels_reached_hyperion: bool,
    pub labels_reached_prediction_fit: bool,
}

impl HyperionConfirmationReportV1 {
    pub const SCHEMA: &'static str = "HyperionConfirmationReport.v1";

    pub fn validate(
        &self,
        policy: &HyperionPolicyFreezeV1,
        selections: &HyperionSelectionInventoryV1,
    ) -> Result<(), String> {
        policy.validate()?;
        selections.validate(policy)?;
        if self.schema != Self::SCHEMA
            || self.policy_freeze_sha256 != policy.identity()?
            || self.selection_inventory_sha256 != selections.identity(policy)?
            || self.datasets.len() != FROZEN_CONFIRMATION_DATASETS
            || !self.complete_immutable_receipts
            || self.labels_reached_hyperion
            || self.labels_reached_prediction_fit
        {
            return Err(
                "Hyperion confirmation report crossed or bypassed the sealed barrier".into(),
            );
        }
        let mut ids = BTreeSet::new();
        for (dataset, expected_id) in self.datasets.iter().zip(CONFIRMATION_DATASET_IDS) {
            dataset.validate()?;
            if dataset.dataset_id != expected_id || !ids.insert(dataset.dataset_id.as_str()) {
                return Err(
                    "Hyperion confirmation report has foreign or duplicate datasets".into(),
                );
            }
        }
        let deltas = self
            .datasets
            .iter()
            .map(|dataset| dataset.paired_delta)
            .collect::<Vec<_>>();
        let positive = deltas.iter().filter(|delta| **delta > 0.0).count() as u32;
        let median = median_f64(&deltas);
        let lower = bootstrap_lower_bound(&deltas)?;
        let raw_wins = self
            .datasets
            .iter()
            .filter(|dataset| dataset.raw_robust_win)
            .count() as u32;
        let gated_wins = self
            .datasets
            .iter()
            .filter(|dataset| dataset.gated_robust_win)
            .count() as u32;
        let validity = self
            .datasets
            .iter()
            .flat_map(|dataset| dataset.seeds.iter().map(|seed| seed.gated_validity))
            .sum::<f64>()
            / (self.datasets.len() * CONFIRMATION_SEEDS.len()) as f64;
        let latency_gate = self.datasets.iter().all(|dataset| {
            dataset
                .seeds
                .iter()
                .all(|seed| seed.gated_latency_seconds <= seed.raw_latency_seconds * 4.0)
        });
        let expected_claim = gated_wins > raw_wins
            && self
                .datasets
                .iter()
                .filter(|dataset| dataset.raw_robust_win)
                .all(|dataset| dataset.gated_robust_win)
            && positive >= policy.decision_rule.minimum_positive_delta_datasets
            && lower > 0.0
            && median > 0.0
            && validity
                >= policy.decision_rule.minimum_aggregate_validity_micros as f64 / 1_000_000.0
            && latency_gate;
        if self.raw_robust_wins != raw_wins
            || self.gated_robust_wins != gated_wins
            || self.positive_delta_datasets != positive
            || (self.median_delta - median).abs() > 1e-12
            || (self.bootstrap_lower_bound - lower).abs() > 1e-12
            || (self.aggregate_validity - validity).abs() > 1e-12
            || self.claim_increased_win_rate != expected_claim
        {
            return Err(
                "Hyperion confirmation report does not match its deterministic gate".into(),
            );
        }
        Ok(())
    }

    pub fn identity(
        &self,
        policy: &HyperionPolicyFreezeV1,
        selections: &HyperionSelectionInventoryV1,
    ) -> Result<String, String> {
        self.validate(policy, selections)?;
        canonical_sha256(self)
    }
}

pub(super) fn median_f64(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    }
}

fn bootstrap_lower_bound(deltas: &[f64]) -> Result<f64, String> {
    if deltas.len() != FROZEN_CONFIRMATION_DATASETS || deltas.iter().any(|value| !value.is_finite())
    {
        return Err("Hyperion bootstrap requires exactly twenty finite paired deltas".into());
    }
    let mut samples = Vec::with_capacity(HYPERION_BOOTSTRAP_REPETITIONS);
    for repetition in 0..HYPERION_BOOTSTRAP_REPETITIONS {
        let mean = (0..deltas.len())
            .map(|draw| {
                let digest = raw_sha256(
                    format!(
                        "hyperion-confirmation-bootstrap-v1|{HYPERION_BOOTSTRAP_SEED}|{repetition}|{draw}"
                    )
                    .as_bytes(),
                );
                let index = usize::from_str_radix(&digest[..16], 16).unwrap() % deltas.len();
                deltas[index]
            })
            .sum::<f64>()
            / deltas.len() as f64;
        samples.push(mean);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[(HYPERION_BOOTSTRAP_REPETITIONS * 5) / 100])
}
