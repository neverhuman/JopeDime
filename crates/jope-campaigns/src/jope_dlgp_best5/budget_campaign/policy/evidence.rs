use super::*;
use super::confirmation::median_f64;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionDevelopmentSeedEvidenceV1 {
    pub dataset_id: String,
    pub seed: u64,
    pub raw_rmse: f64,
    pub gated_rmse: f64,
    pub comparator_rmse: f64,
    pub selected_view: FeatureViewKindV1,
    pub raw_receipt_sha256: String,
    pub gated_receipt_sha256: String,
}

impl HyperionDevelopmentSeedEvidenceV1 {
    fn validate(&self) -> Result<(), String> {
        if !DEVELOPMENT_DATASET_IDS.contains(&self.dataset_id.as_str())
            || !SEARCH_SEEDS.contains(&self.seed)
            || !finite_positive(&[self.raw_rmse, self.gated_rmse, self.comparator_rmse])
        {
            return Err("Hyperion development evidence is outside the frozen grid".into());
        }
        validate_sha256("development raw receipt", &self.raw_receipt_sha256)?;
        validate_sha256("development gated receipt", &self.gated_receipt_sha256)
    }

    fn log_improvement(&self) -> f64 {
        (self.raw_rmse / self.gated_rmse).ln()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionDevelopmentEvidenceV1 {
    pub schema: String,
    pub campaign_id: String,
    pub seed_evidence: Vec<HyperionDevelopmentSeedEvidenceV1>,
    pub protected_screen_trials: u32,
    pub protected_screen_wins: u32,
    pub expanded_frontier_complete: bool,
    pub existing_robust_wins: u32,
    pub retained_existing_robust_wins: u32,
    pub robust_wins: u32,
    pub median_log_improvement: f64,
    pub passed: bool,
}

impl HyperionDevelopmentEvidenceV1 {
    pub const SCHEMA: &'static str = "HyperionDevelopmentEvidence.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.campaign_id != BUDGET_CAMPAIGN_ID
            || self.seed_evidence.len() != DEVELOPMENT_DATASETS * SEARCH_SEEDS.len()
            || self.protected_screen_trials != 10
            || self.protected_screen_wins > self.protected_screen_trials
            || self.existing_robust_wins != 3
            || self.retained_existing_robust_wins > self.existing_robust_wins
        {
            return Err("Hyperion development evidence schema or cardinality drifted".into());
        }
        for (evidence, (dataset, seed)) in self.seed_evidence.iter().zip(
            DEVELOPMENT_DATASET_IDS
                .iter()
                .flat_map(|dataset| SEARCH_SEEDS.map(|seed| (*dataset, seed))),
        ) {
            evidence.validate()?;
            if evidence.dataset_id != dataset || evidence.seed != seed {
                return Err("Hyperion development evidence order or seed coverage drifted".into());
            }
        }
        let median = median_f64(
            &self
                .seed_evidence
                .iter()
                .map(HyperionDevelopmentSeedEvidenceV1::log_improvement)
                .collect::<Vec<_>>(),
        );
        let robust_wins = DEVELOPMENT_DATASET_IDS
            .iter()
            .filter(|dataset| {
                SEARCH_SEEDS.iter().all(|seed| {
                    self.seed_evidence
                        .iter()
                        .find(|evidence| evidence.dataset_id == **dataset && evidence.seed == *seed)
                        .is_some_and(|evidence| evidence.gated_rmse < evidence.comparator_rmse)
                })
            })
            .count() as u32;
        let expected_passed = self.protected_screen_wins >= 7
            && self.expanded_frontier_complete
            && median >= 1.01_f64.ln()
            && self.retained_existing_robust_wins == self.existing_robust_wins
            && robust_wins >= 7;
        if (self.median_log_improvement - median).abs() > 1e-12
            || self.robust_wins != robust_wins
            || self.passed != expected_passed
        {
            return Err("Hyperion development evidence does not match its frozen gate".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionSeedSelectionV1 {
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub seed: u64,
    pub raw_control_commit_sha256: String,
    pub gate_decision_sha256: String,
    pub cache_receipt_sha256: String,
    pub selected_view: FeatureViewKindV1,
    pub selected_predictor_sha256: String,
    pub raw_fallback_predictor_sha256: String,
    pub fallback_reason: Option<String>,
}

impl HyperionSeedSelectionV1 {
    pub fn validate(&self, policy_sha256: &str) -> Result<(), String> {
        validate_sha256("selection policy", policy_sha256)?;
        if !CONFIRMATION_DATASET_IDS.contains(&self.dataset_id.as_str())
            || !CONFIRMATION_SEEDS.contains(&self.seed)
            || !self.dataset_sha256.starts_with(&self.dataset_id)
        {
            return Err("Hyperion selection is outside the frozen confirmation grid".into());
        }
        for (name, digest) in [
            ("selection dataset", &self.dataset_sha256),
            ("raw control", &self.raw_control_commit_sha256),
            ("gate decision", &self.gate_decision_sha256),
            ("cache receipt", &self.cache_receipt_sha256),
            ("selected predictor", &self.selected_predictor_sha256),
            (
                "raw fallback predictor",
                &self.raw_fallback_predictor_sha256,
            ),
        ] {
            validate_sha256(name, digest)?;
        }
        match self.selected_view {
            FeatureViewKindV1::Raw if self.fallback_reason.as_deref().is_none_or(str::is_empty) => {
                Err("raw selection requires a frozen fallback reason".into())
            }
            FeatureViewKindV1::Raw
                if self.selected_predictor_sha256 != self.raw_fallback_predictor_sha256 =>
            {
                Err("raw selection must use its declared raw fallback predictor".into())
            }
            FeatureViewKindV1::HyperionCell
            | FeatureViewKindV1::HyperionRow128
            | FeatureViewKindV1::HyperionBoth
                if self.fallback_reason.is_some() =>
            {
                Err("expanded selection cannot carry a raw fallback reason".into())
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionSelectionInventoryV1 {
    pub schema: String,
    pub policy_freeze_sha256: String,
    pub selections: Vec<HyperionSeedSelectionV1>,
}

impl HyperionSelectionInventoryV1 {
    pub const SCHEMA: &'static str = "HyperionConfirmationSelectionInventory.v1";

    pub fn validate(&self, policy: &HyperionPolicyFreezeV1) -> Result<(), String> {
        policy.validate()?;
        let policy_sha256 = policy.identity()?;
        if self.schema != Self::SCHEMA
            || self.policy_freeze_sha256 != policy_sha256
            || self.selections.len() != FROZEN_CONFIRMATION_DATASETS * CONFIRMATION_SEEDS.len()
        {
            return Err("Hyperion selection inventory is incomplete or foreign".into());
        }
        let expected = CONFIRMATION_DATASET_IDS
            .iter()
            .flat_map(|dataset| CONFIRMATION_SEEDS.map(|seed| (*dataset, seed)))
            .collect::<Vec<_>>();
        for (selection, (dataset, seed)) in self.selections.iter().zip(expected) {
            selection.validate(&policy_sha256)?;
            if selection.dataset_id != dataset || selection.seed != seed {
                return Err("Hyperion selection order or seed coverage drifted".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self, policy: &HyperionPolicyFreezeV1) -> Result<String, String> {
        self.validate(policy)?;
        canonical_sha256(self)
    }
}
