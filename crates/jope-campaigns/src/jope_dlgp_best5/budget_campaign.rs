//! Contracts for the budget-first campaign layered on the existing Best5
//! implementation.  These types do not add or rename public Best5 commands.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use super::{FrozenPredictorComponentV1, LicenseTrackV1, FROZEN_CHECKPOINT_SHA256, SEARCH_SEEDS};
use crate::hyperion_gate::{FeatureViewKindV1, GateThresholdsV1};
use crate::jope_generalized_gp::raw_sha256;
use crate::search_v3::{canonical_sha256, validate_sha256};

pub const BUDGET_CAMPAIGN_ID: &str = "jope-dlgp-budget-first-7of22";
pub const DEVELOPMENT_DATASETS: usize = 22;
pub const FROZEN_CONFIRMATION_DATASETS: usize = 20;
pub const BUDGET_RUNGS: [u32; 4] = [1_000, 2_000, 5_000, 10_000];
pub const CONFIRMATION_SEEDS: [u64; 2] = [303, 404];
pub const STAGE1_MAX_SEARCH_EVALUATIONS: u64 = 200_000;
pub const HYPERION_BOOTSTRAP_SEED: u64 = 0x4843_4f4e_4649_524d;
pub const HYPERION_BOOTSTRAP_REPETITIONS: usize = 10_000;
pub const HYPERION_FULL_TRAIN_EXECUTIONS: usize = 42;
pub const HYPERION_CV5_EXECUTIONS: usize = 42 * 5;
pub const HYPERION_CV7_EXECUTIONS: usize = 42 * 7;
pub const HYPERION_TOTAL_ENCODER_EXECUTIONS: usize =
    HYPERION_FULL_TRAIN_EXECUTIONS + HYPERION_CV5_EXECUTIONS + HYPERION_CV7_EXECUTIONS;

/// The Hyperion cache command is campaign-specific: an arbitrary train-only
/// registry must not accidentally become the authority for this confirmation.
pub fn validate_hyperion_campaign_registry_ids(ids: &[String]) -> Result<(), String> {
    let expected = DEVELOPMENT_DATASET_IDS
        .iter()
        .chain(CONFIRMATION_DATASET_IDS.iter())
        .map(|id| (*id).to_owned())
        .collect::<Vec<_>>();
    if ids != expected {
        return Err(
            "Hyperion registry must contain the exact ordered 22-development plus 20-confirmation cohort"
                .into(),
        );
    }
    Ok(())
}

pub const DEVELOPMENT_DATASET_IDS: [&str; DEVELOPMENT_DATASETS] = [
    "1a803b27c2400dc4",
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "43b4cc3ab0ebbd97",
    "471f11cc2d38853a",
    "4f68d277d770477f",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "5f91a00bd3b66742",
    "7af85a04de0c7018",
    "860e5558f732f638",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b5beff4a53fa12c0",
    "b78d323566930bd7",
    "bd49e740a79bcb6c",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "e5a3a7b3668a1cc2",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
];

pub const CONFIRMATION_DATASET_IDS: [&str; FROZEN_CONFIRMATION_DATASETS] = [
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1cb97b1d7314d43d",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "542e75bc0cfdd1bf",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "ab0239b9ef0ba28e",
    "b253fbfae8460dc1",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "ee56a1015e344c11",
    "fe3cc716a05d036f",
];

pub const STAGE1_DATASET_IDS: [&str; 10] = [
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "471f11cc2d38853a",
    "4f68d277d770477f",
    "7af85a04de0c7018",
    "b5beff4a53fa12c0",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
];

fn sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn finite_positive(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite() && *value > 0.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignLayerV1 {
    Development,
    Confirmation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestAccessPolicyV1 {
    SearchTrainMetadataOnlySealedFinalizerOwnsTest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchDatasetBindingV1 {
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub train_sha256: String,
    pub metadata_sha256: String,
}

impl SearchDatasetBindingV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.dataset_id.len() != 16 || !self.dataset_sha256.starts_with(&self.dataset_id) {
            return Err("search dataset ID does not prefix its full digest".into());
        }
        for (name, digest) in [
            ("dataset", self.dataset_sha256.as_str()),
            ("training", self.train_sha256.as_str()),
            ("metadata", self.metadata_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetFirstCampaignFreezeV1 {
    pub schema: String,
    pub campaign_id: String,
    pub development: Vec<SearchDatasetBindingV1>,
    pub confirmation: Vec<SearchDatasetBindingV1>,
    pub comparator_snapshot_sha256: String,
    pub code_revision: String,
    pub source_tree_sha256: String,
    pub search_binary_sha256: String,
    pub evaluator_sha256: String,
    pub proposal_checkpoint_sha256: String,
    pub proposal_inference_protocol_sha256: String,
    pub metric_sha256: String,
    pub budget_rungs: Vec<u32>,
    pub development_seeds: Vec<u64>,
    pub confirmation_seeds: Vec<u64>,
    pub test_access_policy: TestAccessPolicyV1,
    pub license_tracks: BTreeSet<LicenseTrackV1>,
}

impl BudgetFirstCampaignFreezeV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBudgetFirstCampaignFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.campaign_id != BUDGET_CAMPAIGN_ID {
            return Err("budget-first campaign schema or ID drift".into());
        }
        for (name, digest) in [
            (
                "comparator snapshot",
                self.comparator_snapshot_sha256.as_str(),
            ),
            ("source tree", self.source_tree_sha256.as_str()),
            ("search binary", self.search_binary_sha256.as_str()),
            ("evaluator", self.evaluator_sha256.as_str()),
            (
                "proposal checkpoint",
                self.proposal_checkpoint_sha256.as_str(),
            ),
            (
                "proposal inference protocol",
                self.proposal_inference_protocol_sha256.as_str(),
            ),
            ("metric", self.metric_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.code_revision.trim().is_empty()
            || self.proposal_checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256
            || self.budget_rungs != BUDGET_RUNGS
            || self.development_seeds != SEARCH_SEEDS
            || self.confirmation_seeds != CONFIRMATION_SEEDS
            || self.license_tracks
                != BTreeSet::from([
                    LicenseTrackV1::ProductionCompatible,
                    LicenseTrackV1::ResearchOnly,
                ])
        {
            return Err(
                "budget-first frozen code, checkpoint, seeds, rungs, or tracks drifted".into(),
            );
        }
        validate_layer(&self.development, &DEVELOPMENT_DATASET_IDS)?;
        validate_layer(&self.confirmation, &CONFIRMATION_DATASET_IDS)?;
        let development = self
            .development
            .iter()
            .map(|dataset| dataset.dataset_sha256.as_str())
            .collect::<BTreeSet<_>>();
        if self
            .confirmation
            .iter()
            .any(|dataset| development.contains(dataset.dataset_sha256.as_str()))
        {
            return Err("development and confirmation cohorts overlap".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

fn validate_layer(bindings: &[SearchDatasetBindingV1], expected: &[&str]) -> Result<(), String> {
    for binding in bindings {
        binding.validate()?;
    }
    let actual = bindings
        .iter()
        .map(|binding| binding.dataset_id.as_str())
        .collect::<BTreeSet<_>>();
    if bindings.len() != expected.len() || actual != expected.iter().copied().collect() {
        return Err("campaign cohort differs from its frozen dataset set".into());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedTestBindingV1 {
    pub dataset_sha256: String,
    pub test_sha256: String,
    pub test_relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedTestInventoryV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub sealed_finalizer_sha256: String,
    pub dataset_claim_registry_sha256: String,
    pub tests: Vec<SealedTestBindingV1>,
}

impl SealedTestInventoryV1 {
    pub const SCHEMA: &'static str = "JopeDlgpSealedTestInventory.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.tests.len() != 42 {
            return Err("sealed test inventory schema or cardinality drift".into());
        }
        for digest in [
            &self.campaign_freeze_sha256,
            &self.sealed_finalizer_sha256,
            &self.dataset_claim_registry_sha256,
        ] {
            validate_sha256("sealed test inventory", digest)?;
        }
        let mut datasets = BTreeSet::new();
        for test in &self.tests {
            validate_sha256("sealed dataset", &test.dataset_sha256)?;
            validate_sha256("sealed test", &test.test_sha256)?;
            let path = Path::new(&test.test_relative_path);
            if path.is_absolute()
                || path.as_os_str().is_empty()
                || path.components().any(|part| part == Component::ParentDir)
                || !datasets.insert(test.dataset_sha256.as_str())
            {
                return Err("sealed test binding is unsafe or duplicated".into());
            }
        }
        Ok(())
    }
}

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

fn median_f64(values: &[f64]) -> f64 {
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

/// A single-read sealed boundary. The private labels stay in the finalizer
/// object; the only method intended for the encoder accepts feature rows and
/// appends constant dummy targets.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperionSealedFeatureStageV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub test_sha256: String,
    pub feature_count: usize,
    pub feature_rows: Vec<Vec<f64>>,
    labels: Vec<f64>,
    pub test_access_ordinal: u32,
}

impl HyperionSealedFeatureStageV1 {
    pub const SCHEMA: &'static str = "HyperionSealedFeatureStage.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.test_access_ordinal != 1
            || self.feature_count == 0
            || self.feature_rows.is_empty()
            || self.labels.len() != self.feature_rows.len()
            || self.feature_rows.iter().any(|row| {
                row.len() != self.feature_count || row.iter().any(|value| !value.is_finite())
            })
            || self.labels.iter().any(|value| !value.is_finite())
        {
            return Err("sealed feature stage has invalid shape or access boundary".into());
        }
        validate_sha256("sealed dataset", &self.dataset_sha256)?;
        validate_sha256("sealed test", &self.test_sha256)
    }

    pub fn hyperion_dummy_target_rows(&self) -> Result<Vec<Vec<f64>>, String> {
        self.validate()?;
        Ok(self
            .feature_rows
            .iter()
            .map(|row| {
                let mut output = row.clone();
                output.push(0.0);
                output
            })
            .collect())
    }

    pub fn finalizer_labels(&self) -> Result<&[f64], String> {
        self.validate()?;
        Ok(&self.labels)
    }
}

/// Parse the already-opened sealed bytes exactly once. Feature bytes are
/// materialized before labels are retained, and the resulting stage has one
/// immutable access ordinal for the finalizer receipt.
pub fn stage_hyperion_sealed_features(
    dataset_sha256: &str,
    test_sha256: &str,
    bytes: &[u8],
    feature_count: usize,
) -> Result<HyperionSealedFeatureStageV1, String> {
    validate_sha256("sealed dataset", dataset_sha256)?;
    validate_sha256("sealed test", test_sha256)?;
    if feature_count == 0 || bytes.is_empty() {
        return Err("sealed test bytes or feature count is empty".into());
    }
    let mut feature_rows = Vec::new();
    let mut labels = Vec::new();
    for (line_index, line) in std::str::from_utf8(bytes)
        .map_err(|error| format!("sealed test is not UTF-8: {error}"))?
        .lines()
        .enumerate()
    {
        if line.trim().is_empty() {
            continue;
        }
        let parsed = line
            .split(',')
            .map(|field| field.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>();
        let mut values = match parsed {
            Ok(values) => values,
            Err(_) if line_index == 0 => continue,
            Err(error) => return Err(format!("sealed test row {line_index} is invalid: {error}")),
        };
        if values.len() != feature_count + 1 {
            return Err(format!(
                "sealed test row {line_index} has {} columns, expected {}",
                values.len(),
                feature_count + 1
            ));
        }
        let label = values.pop().expect("feature count checked");
        if !label.is_finite() || values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "sealed test row {line_index} contains a nonfinite value"
            ));
        }
        feature_rows.push(values);
        labels.push(label);
    }
    let stage = HyperionSealedFeatureStageV1 {
        schema: HyperionSealedFeatureStageV1::SCHEMA.into(),
        dataset_sha256: dataset_sha256.into(),
        test_sha256: test_sha256.into(),
        feature_count,
        feature_rows,
        labels,
        test_access_ordinal: 1,
    };
    stage.validate()?;
    Ok(stage)
}

/// The fallback decision is made from encoder/shape proof only. It accepts no
/// labels and therefore cannot be retuned by sealed outcomes.
pub fn choose_predeclared_raw_fallback(encoding_verified: bool, shape_verified: bool) -> bool {
    !(encoding_verified && shape_verified)
}

/// The sealed finalizer may only accept a fresh full-train encoder result if
/// its bytes are identical to the registry-frozen full-train cache.  This
/// comparison is deliberately independent of test labels.
pub fn verify_full_train_replay(
    expected_cache_sha256: &str,
    fresh_train_bytes: &[u8],
) -> Result<String, String> {
    validate_sha256("full-train cache", expected_cache_sha256)?;
    if fresh_train_bytes.is_empty() {
        return Err("fresh full-train encoder output is empty".into());
    }
    let actual = raw_sha256(fresh_train_bytes);
    if actual != expected_cache_sha256 {
        return Err("fresh full-train encoder output differs from frozen cache".into());
    }
    Ok(actual)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetCheckpointV1 {
    pub evaluations: u32,
    pub prefix_prediction_roots: BTreeMap<u32, String>,
    pub population_sha256: String,
    pub selection_sha256: String,
    pub receipt_root_sha256: String,
    pub prediction_root_sha256: String,
    pub cv7_rmse: Vec<f64>,
    pub archive_top20_semantic_sha256s: Vec<String>,
    pub logical_operations: u64,
    pub cache_hits: u64,
    pub cpu_seconds: f64,
    pub gpu_seconds: f64,
}

impl BudgetCheckpointV1 {
    pub fn validate(&self) -> Result<(), String> {
        if !BUDGET_RUNGS.contains(&self.evaluations)
            || self.cv7_rmse.len() != 7
            || !finite_positive(&self.cv7_rmse)
            || self.archive_top20_semantic_sha256s.len() != 20
            || !sorted_unique(&self.archive_top20_semantic_sha256s)
            || self.logical_operations < self.evaluations as u64
            || self.cache_hits > self.logical_operations
            || !self.cpu_seconds.is_finite()
            || self.cpu_seconds < 0.0
            || !self.gpu_seconds.is_finite()
            || self.gpu_seconds < 0.0
        {
            return Err("budget checkpoint metrics, archive, or accounting are invalid".into());
        }
        for digest in [
            &self.population_sha256,
            &self.selection_sha256,
            &self.receipt_root_sha256,
            &self.prediction_root_sha256,
        ] {
            validate_sha256("budget checkpoint", digest)?;
        }
        for digest in &self.archive_top20_semantic_sha256s {
            validate_sha256("archive semantic", digest)?;
        }
        for digest in self.prefix_prediction_roots.values() {
            validate_sha256("checkpoint prefix", digest)?;
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
pub struct BudgetPromotionDecisionV1 {
    pub source_2k_checkpoint_sha256: String,
    pub source_5k_checkpoint_sha256: String,
    pub continue_to_10k: bool,
    pub improved_folds_at_least_point_one_percent: u32,
    pub archive_jaccard_micros: u32,
    pub cv7_mean_ratio_micros: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetLadderBundleV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub dataset_sha256: String,
    pub master_seed: u64,
    pub license_track: LicenseTrackV1,
    pub metric_sha256: String,
    pub evaluator_sha256: String,
    pub split_sha256: String,
    pub test_csv_read: bool,
    pub checkpoints: Vec<BudgetCheckpointV1>,
    pub promotion: BudgetPromotionDecisionV1,
}

impl BudgetLadderBundleV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBudgetLadderBundle.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.test_csv_read
            || !SEARCH_SEEDS.contains(&self.master_seed)
            || !(self.checkpoints.len() == 3 || self.checkpoints.len() == 4)
        {
            return Err(
                "budget ladder schema, seed, test-access, or rung cardinality drift".into(),
            );
        }
        for digest in [
            &self.campaign_freeze_sha256,
            &self.dataset_sha256,
            &self.metric_sha256,
            &self.evaluator_sha256,
            &self.split_sha256,
        ] {
            validate_sha256("budget ladder", digest)?;
        }
        for checkpoint in &self.checkpoints {
            checkpoint.validate()?;
        }
        let actual_rungs = self
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.evaluations)
            .collect::<Vec<_>>();
        if actual_rungs != BUDGET_RUNGS[..actual_rungs.len()] {
            return Err("budget ladder checkpoints are not the nested frozen prefix".into());
        }
        for (index, checkpoint) in self.checkpoints.iter().enumerate() {
            let expected = self.checkpoints[..=index]
                .iter()
                .map(|prior| (prior.evaluations, prior.prediction_root_sha256.clone()))
                .collect::<BTreeMap<_, _>>();
            if checkpoint.prefix_prediction_roots != expected {
                return Err(
                    "checkpoint prefix roots do not reproduce the earlier trajectory".into(),
                );
            }
        }
        let two_k = &self.checkpoints[1];
        let five_k = &self.checkpoints[2];
        if self.promotion.source_2k_checkpoint_sha256 != two_k.identity()?
            || self.promotion.source_5k_checkpoint_sha256 != five_k.identity()?
        {
            return Err("10k promotion decision does not bind the 2k and 5k checkpoints".into());
        }
        let improved_folds = two_k
            .cv7_rmse
            .iter()
            .zip(&five_k.cv7_rmse)
            .filter(|(two, five)| **five <= **two * 0.999)
            .count() as u32;
        let left = two_k
            .archive_top20_semantic_sha256s
            .iter()
            .collect::<BTreeSet<_>>();
        let right = five_k
            .archive_top20_semantic_sha256s
            .iter()
            .collect::<BTreeSet<_>>();
        let intersection = left.intersection(&right).count() as u64;
        let union = left.union(&right).count() as u64;
        let jaccard_micros = (intersection * 1_000_000 / union) as u32;
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        let mean_ratio_micros =
            (mean(&five_k.cv7_rmse) / mean(&two_k.cv7_rmse) * 1_000_000.0).round() as u32;
        let should_continue =
            improved_folds >= 4 || (jaccard_micros < 800_000 && mean_ratio_micros <= 1_005_000);
        if self.promotion.improved_folds_at_least_point_one_percent != improved_folds
            || self.promotion.archive_jaccard_micros != jaccard_micros
            || self.promotion.cv7_mean_ratio_micros != mean_ratio_micros
            || self.promotion.continue_to_10k != should_continue
            || should_continue != (self.checkpoints.len() == 4)
        {
            return Err("10k promotion decision differs from the frozen train-only rule".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

pub fn verify_prefix_equality(
    left: &BudgetLadderBundleV1,
    right: &BudgetLadderBundleV1,
    evaluations: u32,
) -> Result<(), String> {
    left.validate()?;
    right.validate()?;
    let left_prefix = left
        .checkpoints
        .last()
        .and_then(|checkpoint| checkpoint.prefix_prediction_roots.get(&evaluations));
    let right_prefix = right
        .checkpoints
        .last()
        .and_then(|checkpoint| checkpoint.prefix_prediction_roots.get(&evaluations));
    if left.campaign_freeze_sha256 != right.campaign_freeze_sha256
        || left.dataset_sha256 != right.dataset_sha256
        || left.master_seed != right.master_seed
        || left.license_track != right.license_track
        || left_prefix.is_none()
        || left_prefix != right_prefix
    {
        return Err("resumed/retried budget trajectories differ at their frozen prefix".into());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetStageInventoryV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub bundles: Vec<BudgetLadderBundleV1>,
}

impl BudgetStageInventoryV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBudgetStageInventory.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.bundles.len() != 20 {
            return Err("stage-1 inventory must contain ten datasets by two seeds".into());
        }
        validate_sha256("stage-1 campaign", &self.campaign_freeze_sha256)?;
        let mut actual = BTreeSet::new();
        let expected = STAGE1_DATASET_IDS
            .iter()
            .flat_map(|dataset| SEARCH_SEEDS.map(|seed| (*dataset, seed)))
            .collect::<BTreeSet<_>>();
        let mut evaluations = 0_u64;
        for bundle in &self.bundles {
            bundle.validate()?;
            let dataset_id = STAGE1_DATASET_IDS
                .iter()
                .copied()
                .find(|dataset| bundle.dataset_sha256.starts_with(dataset))
                .ok_or_else(|| "stage-1 bundle is outside the frozen frontier".to_string())?;
            if bundle.campaign_freeze_sha256 != self.campaign_freeze_sha256
                || !actual.insert((dataset_id, bundle.master_seed))
            {
                return Err("stage-1 inventory contains a foreign or duplicate bundle".into());
            }
            evaluations += bundle.checkpoints.last().unwrap().evaluations as u64;
        }
        if actual != expected || evaluations > STAGE1_MAX_SEARCH_EVALUATIONS {
            return Err("stage-1 dataset grid or 200,000-evaluation cap drifted".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuterFoldBaseReceiptV1 {
    pub outer_fold: u32,
    pub outer_training_row_ids: Vec<String>,
    pub outer_validation_row_ids: Vec<String>,
    pub row_atomic_groups: BTreeMap<String, String>,
    pub component: FrozenPredictorComponentV1,
    pub inner_oof_prediction_sha256: String,
    pub outer_validation_prediction_sha256: String,
}

impl OuterFoldBaseReceiptV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.outer_fold >= 7
            || self.outer_training_row_ids.is_empty()
            || self.outer_validation_row_ids.is_empty()
            || !sorted_unique(&self.outer_training_row_ids)
            || !sorted_unique(&self.outer_validation_row_ids)
        {
            return Err("outer-fold row inventory is invalid".into());
        }
        self.component.validate()?;
        validate_sha256("inner OOF predictions", &self.inner_oof_prediction_sha256)?;
        validate_sha256(
            "outer validation predictions",
            &self.outer_validation_prediction_sha256,
        )?;
        let training = self.outer_training_row_ids.iter().collect::<BTreeSet<_>>();
        let validation = self
            .outer_validation_row_ids
            .iter()
            .collect::<BTreeSet<_>>();
        if !training.is_disjoint(&validation)
            || self.row_atomic_groups.len() != training.len() + validation.len()
        {
            return Err("nested OOF outer partitions overlap or omit rows".into());
        }
        let training_groups = training
            .iter()
            .map(|row| self.row_atomic_groups.get(*row))
            .collect::<Option<BTreeSet<_>>>()
            .ok_or_else(|| "nested OOF training row lacks an atomic group".to_string())?;
        let validation_groups = validation
            .iter()
            .map(|row| self.row_atomic_groups.get(*row))
            .collect::<Option<BTreeSet<_>>>()
            .ok_or_else(|| "nested OOF validation row lacks an atomic group".to_string())?;
        if !training_groups.is_disjoint(&validation_groups) {
            return Err("nested OOF split crosses a duplicate-feature atomic group".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NestedOofBundleV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub dataset_sha256: String,
    pub license_track: LicenseTrackV1,
    pub cross_fit_protocol_sha256: String,
    pub folds: Vec<OuterFoldBaseReceiptV1>,
    pub selection_freeze_sha256: Option<String>,
    pub full_train_base_prediction_sha256: Option<String>,
}

impl NestedOofBundleV1 {
    pub const SCHEMA: &'static str = "JopeDlgpNestedOofBundle.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.folds.len() != 7 {
            return Err("nested OOF bundle must contain seven outer folds".into());
        }
        for digest in [
            &self.campaign_freeze_sha256,
            &self.dataset_sha256,
            &self.cross_fit_protocol_sha256,
        ] {
            validate_sha256("nested OOF bundle", digest)?;
        }
        for (index, fold) in self.folds.iter().enumerate() {
            fold.validate()?;
            if fold.outer_fold != index as u32
                || (self.license_track == LicenseTrackV1::ProductionCompatible
                    && fold.component.license_track() == LicenseTrackV1::ResearchOnly)
            {
                return Err("nested OOF fold order or license track drifted".into());
            }
        }
        match (
            &self.selection_freeze_sha256,
            &self.full_train_base_prediction_sha256,
        ) {
            (None, None) => Ok(()),
            (Some(selection), Some(prediction)) => {
                validate_sha256("selection freeze", selection)?;
                validate_sha256("full-train base prediction", prediction)
            }
            _ => Err("full-train base predictions require a prior selection freeze".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackArtifactNamespacesV1 {
    pub production_compatible_root: String,
    pub research_only_root: String,
}

impl TrackArtifactNamespacesV1 {
    pub fn validate(&self) -> Result<(), String> {
        let production = Path::new(&self.production_compatible_root);
        let research = Path::new(&self.research_only_root);
        if !production.is_absolute()
            || !research.is_absolute()
            || production == research
            || production.starts_with(research)
            || research.starts_with(production)
        {
            return Err(
                "production and research artifact trees must be disjoint absolute roots".into(),
            );
        }
        Ok(())
    }

    pub fn validate_artifact(&self, track: LicenseTrackV1, path: &str) -> Result<(), String> {
        self.validate()?;
        let artifact = Path::new(path);
        let expected = match track {
            LicenseTrackV1::ProductionCompatible => Path::new(&self.production_compatible_root),
            LicenseTrackV1::ResearchOnly => Path::new(&self.research_only_root),
        };
        if !artifact.is_absolute()
            || artifact
                .components()
                .any(|part| part == Component::ParentDir)
            || !artifact.starts_with(expected)
        {
            return Err("artifact path escapes its license-track tree".into());
        }
        Ok(())
    }
}

pub fn validate_search_artifact_json(value: &serde_json::Value) -> Result<(), String> {
    match value {
        serde_json::Value::Object(values) => {
            for (key, nested) in values {
                let normalized = key.to_ascii_lowercase();
                if normalized == "test_csv_read" {
                    if nested != &serde_json::Value::Bool(false) {
                        return Err("search artifact does not attest test_csv_read=false".into());
                    }
                } else if normalized != "test_access_policy"
                    && (normalized.starts_with("test_")
                        || normalized.ends_with("_test")
                        || matches!(
                            normalized.as_str(),
                            "target_rmse" | "comparator_target" | "label_statistics" | "test"
                        ))
                {
                    return Err("search artifact contains sealed test material".into());
                }
                validate_search_artifact_json(nested)?;
            }
        }
        serde_json::Value::Array(values) => {
            for nested in values {
                validate_search_artifact_json(nested)?;
            }
        }
        serde_json::Value::String(value) if value.to_ascii_lowercase().contains("test.csv") => {
            return Err("search artifact contains a test path".into());
        }
        _ => {}
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedDatasetTrackScoreV1 {
    pub dataset_sha256: String,
    pub license_track: LicenseTrackV1,
    pub comparator_rmse: f64,
    pub seed_rmse: BTreeMap<u64, f64>,
}

impl SealedDatasetTrackScoreV1 {
    pub fn robust_win(&self, expected_seeds: [u64; 2]) -> Result<bool, String> {
        validate_sha256("sealed score dataset", &self.dataset_sha256)?;
        if !self.comparator_rmse.is_finite()
            || self.comparator_rmse < 0.0
            || self.seed_rmse.keys().copied().collect::<Vec<_>>() != expected_seeds
            || !finite_positive(&self.seed_rmse.values().copied().collect::<Vec<_>>())
        {
            return Err("sealed paired score is invalid or uses the wrong seeds".into());
        }
        if self.comparator_rmse == 0.0 {
            return Ok(false);
        }
        Ok(self
            .seed_rmse
            .values()
            .all(|rmse| *rmse < self.comparator_rmse))
    }
}
