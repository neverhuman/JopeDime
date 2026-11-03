use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedTestResultV1 {
    pub schema: String,
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub selection_freeze_sha256: String,
    pub winner_semantic_sha256: String,
    pub task: GpTaskV1,
    pub primary_metric: f64,
    pub secondary_metric: f64,
    pub runtime: RuntimeObservationV1,
    pub valid: bool,
}

impl SealedTestResultV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpSealedTestResult.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("sealed dataset", self.dataset_sha256.as_str()),
            ("selection freeze", self.selection_freeze_sha256.as_str()),
            ("sealed winner", self.winner_semantic_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        finite("sealed primary", self.primary_metric)?;
        finite("sealed secondary", self.secondary_metric)?;
        self.runtime.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PilotDatasetResultV1 {
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub task: GpTaskV1,
    pub sealed_skill: f64,
    pub cpu_seconds: f64,
    pub winner_latency_seconds: f64,
    pub generation_zero_validity: f64,
    pub finalist_validity: f64,
    pub overall_validity: f64,
    pub sealed_winner: bool,
}

impl PilotDatasetResultV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("pilot result dataset", &self.dataset_sha256)?;
        for (name, value) in [
            ("sealed skill", self.sealed_skill),
            ("CPU seconds", self.cpu_seconds),
            ("winner latency", self.winner_latency_seconds),
            ("generation-zero validity", self.generation_zero_validity),
            ("finalist validity", self.finalist_validity),
            ("overall validity", self.overall_validity),
        ] {
            finite(name, value)?;
        }
        if self.cpu_seconds < 0.0
            || self.winner_latency_seconds < 0.0
            || [
                self.generation_zero_validity,
                self.finalist_validity,
                self.overall_validity,
            ]
            .iter()
            .any(|value| !(0.0..=1.0).contains(value))
        {
            return Err("pilot runtime or validity is outside its domain".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PilotArmSummaryV1 {
    pub arm: CampaignArmV1,
    pub mean_regression_delta: f64,
    pub mean_binary_delta: f64,
    pub equal_task_mean_delta: f64,
    pub one_sided_95_lower_bound: f64,
    pub predicted_cpu_seconds: f64,
    pub validity_gate: bool,
    pub sealed_winner_gate: bool,
    pub runtime_gate: bool,
    pub advances: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PilotDecisionV1 {
    Advance { arm: CampaignArmV1 },
    NoImprovement { receipt_sha256: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoImprovementV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub reason_codes: Vec<String>,
}

impl NoImprovementV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpNoImprovement.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("no-improvement campaign", &self.campaign_freeze_sha256)?;
        if self.reason_codes.is_empty()
            || self
                .reason_codes
                .iter()
                .any(|reason| reason.trim().is_empty())
        {
            return Err("no_improvement requires typed reason codes".into());
        }
        if !self.reason_codes.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err("no_improvement reason codes must be sorted and unique".into());
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
pub struct PilotComparisonV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub bootstrap_repetitions: u32,
    pub arms: Vec<PilotArmSummaryV1>,
    pub decision: PilotDecisionV1,
}

impl PilotComparisonV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpPilotComparison.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("pilot comparison campaign", &self.campaign_freeze_sha256)?;
        if self.bootstrap_repetitions != 10_000 || self.arms.len() != 3 {
            return Err("pilot comparison requires 10,000 bootstraps over three arms".into());
        }
        let arms: BTreeSet<_> = self.arms.iter().map(|arm| arm.arm).collect();
        if arms != CampaignArmV1::EXPERIMENTS.into_iter().collect() {
            return Err("pilot comparison contains the wrong experiment arms".into());
        }
        if let PilotDecisionV1::Advance { arm } = self.decision {
            if !arm.eligible()
                || !self
                    .arms
                    .iter()
                    .any(|summary| summary.arm == arm && summary.advances)
            {
                return Err("pilot decision advances an ineligible or gated arm".into());
            }
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
pub struct LaunchGateV1 {
    pub schema: String,
    pub real_u1_evaluator_flow: bool,
    pub fresh_supply_evidence: bool,
    pub protected_wave_count: u32,
    pub reviewed_scq_acknowledged: bool,
    pub free_cores: u32,
    pub liveness_age_seconds: u64,
    pub max_liveness_age_seconds: u64,
    pub storage_free_fraction: f64,
    pub min_storage_free_fraction: f64,
}

impl LaunchGateV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpLaunchGate.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        if !self.storage_free_fraction.is_finite()
            || !self.min_storage_free_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.storage_free_fraction)
            || !(0.0..=1.0).contains(&self.min_storage_free_fraction)
            || self.max_liveness_age_seconds == 0
        {
            return Err("launch gate thresholds are invalid".into());
        }
        Ok(())
    }

    pub fn launch_allowed(&self) -> Result<bool, String> {
        self.validate()?;
        Ok(self.real_u1_evaluator_flow
            && self.fresh_supply_evidence
            && self.protected_wave_count >= 2
            && self.reviewed_scq_acknowledged
            && self.free_cores > 0
            && self.liveness_age_seconds <= self.max_liveness_age_seconds
            && self.storage_free_fraction >= self.min_storage_free_fraction)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrancheV1 {
    pub index: u32,
    pub partition: CatalogPartitionV1,
    pub dataset_sha256s: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranchePlanV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub selected_arm: CampaignArmV1,
    pub partition: CatalogPartitionV1,
    pub pilot_comparison_sha256: String,
    pub development_completion_sha256: Option<String>,
    pub reused_pilot_dataset_sha256s: Vec<String>,
    pub tranches: Vec<TrancheV1>,
}

impl TranchePlanV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpTranchePlan.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("tranche campaign", &self.campaign_freeze_sha256)?;
        validate_sha256("pilot comparison", &self.pilot_comparison_sha256)?;
        if !self.selected_arm.eligible() || self.tranches.is_empty() {
            return Err("tranche plan requires an eligible experiment and nonempty work".into());
        }
        match (
            self.partition,
            self.development_completion_sha256.as_deref(),
        ) {
            (CatalogPartitionV1::Development, None) => {}
            (CatalogPartitionV1::Validation, Some(digest)) => {
                validate_sha256("development completion", digest)?
            }
            (CatalogPartitionV1::Development, Some(_)) => {
                return Err("development tranches cannot claim prior development completion".into())
            }
            (CatalogPartitionV1::Validation, None) => {
                return Err("validation audit cannot start before development completion".into())
            }
        }
        let mut datasets = BTreeSet::new();
        for (index, tranche) in self.tranches.iter().enumerate() {
            tranche.validate()?;
            if tranche.index != index as u32 || tranche.partition != self.partition {
                return Err("tranche ordering or partition changed".into());
            }
            for dataset in &tranche.dataset_sha256s {
                if !datasets.insert(dataset.as_str()) {
                    return Err("dataset occurs in more than one tranche".into());
                }
            }
        }
        let mut reused = BTreeSet::new();
        for dataset in &self.reused_pilot_dataset_sha256s {
            validate_sha256("reused pilot dataset", dataset)?;
            if !reused.insert(dataset.as_str()) || !datasets.contains(dataset.as_str()) {
                return Err("reused pilot evidence must name a unique planned dataset".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

impl TrancheV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.dataset_sha256s.is_empty() || self.dataset_sha256s.len() > 128 {
            return Err("tranches must contain 1 through 128 datasets".into());
        }
        let mut seen = BTreeSet::new();
        for digest in &self.dataset_sha256s {
            validate_sha256("tranche dataset", digest)?;
            if !seen.insert(digest) {
                return Err("tranche contains a duplicate dataset".into());
            }
        }
        Ok(())
    }
}

pub fn ensure_selection_before_test(
    selection: &Cv7SelectionFreezeV1,
    expected_dataset: &str,
    expected_candidate: &str,
) -> Result<String, String> {
    selection.validate()?;
    if selection.dataset_sha256 != expected_dataset
        || selection.winner_semantic_sha256 != expected_candidate
    {
        return Err("sealed-test request does not match its selection freeze".into());
    }
    selection.identity()
}

pub fn validate_prediction_binding(
    key: &PredictionCacheKeyV1,
    prediction: &PredictionObjectV1,
) -> Result<(), String> {
    key.validate()?;
    prediction.validate()?;
    if prediction.training_view_sha256 != key.training_sha256
        || prediction.fold_plan_sha256 != key.split_sha256
        || prediction.candidate_semantic_sha256 != key.candidate_semantic_sha256
        || prediction.evaluator_sha256 != key.evaluator_sha256
        || prediction.math_policy_sha256 != key.math_policy_sha256
        || prediction.backend_policy_sha256 != key.backend_policy_sha256
    {
        return Err("prediction object does not match its reusable cache key".into());
    }
    Ok(())
}

pub fn artifact_exists(path: &str) -> bool {
    Path::new(path).is_file()
}
