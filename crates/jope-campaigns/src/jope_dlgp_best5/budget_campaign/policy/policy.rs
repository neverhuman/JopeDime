use super::*;

/// The policy is the last train-only authority written before confirmation.
/// It contains hashes and rules, never test paths, test bytes, labels, or
/// label-derived statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionPolicyFreezeV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub registry_sha256: String,
    pub coverage_sha256: String,
    pub encoder_identity_sha256: String,
    pub encoder_runtime_authority_sha256: String,
    pub raw_protocol_sha256: String,
    pub development_evidence_sha256: String,
    pub comparator_snapshot_sha256: String,
    pub finalizer_binary_sha256: String,
    pub confirmation_dataset_ids: Vec<String>,
    pub confirmation_seeds: Vec<u64>,
    pub gate_thresholds: GateThresholdsV1,
    pub decision_rule: HyperionDecisionRuleV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionDecisionRuleV1 {
    pub minimum_positive_delta_datasets: u32,
    pub bootstrap_confidence_micros: u32,
    pub bootstrap_repetitions: u32,
    pub minimum_aggregate_validity_micros: u32,
    pub maximum_latency_ratio_micros: u32,
    pub preserve_raw_robust_wins: bool,
    pub require_strictly_more_robust_wins: bool,
}

impl Default for HyperionDecisionRuleV1 {
    fn default() -> Self {
        Self {
            minimum_positive_delta_datasets: 11,
            bootstrap_confidence_micros: 950_000,
            bootstrap_repetitions: HYPERION_BOOTSTRAP_REPETITIONS as u32,
            minimum_aggregate_validity_micros: 990_000,
            maximum_latency_ratio_micros: 4_000_000,
            preserve_raw_robust_wins: true,
            require_strictly_more_robust_wins: true,
        }
    }
}

impl HyperionPolicyFreezeV1 {
    pub const SCHEMA: &'static str = "HyperionPolicyFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.confirmation_dataset_ids
                != CONFIRMATION_DATASET_IDS
                    .iter()
                    .map(|value| (*value).to_owned())
                    .collect::<Vec<_>>()
            || self.confirmation_seeds != CONFIRMATION_SEEDS
            || self.gate_thresholds != GateThresholdsV1::default()
            || self.decision_rule != HyperionDecisionRuleV1::default()
        {
            return Err(
                "Hyperion policy schema, cohort, thresholds, or decision rule drifted".into(),
            );
        }
        for (name, digest) in [
            ("campaign", &self.campaign_freeze_sha256),
            ("registry", &self.registry_sha256),
            ("coverage", &self.coverage_sha256),
            ("encoder", &self.encoder_identity_sha256),
            ("encoder runtime", &self.encoder_runtime_authority_sha256),
            ("raw protocol", &self.raw_protocol_sha256),
            ("development evidence", &self.development_evidence_sha256),
            ("comparator", &self.comparator_snapshot_sha256),
            ("finalizer", &self.finalizer_binary_sha256),
        ] {
            validate_sha256(name, digest)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}
