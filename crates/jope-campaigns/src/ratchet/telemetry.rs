use super::*;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]

pub(crate) struct RatchetLiveTelemetryV2 {
    pub(crate) schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) serving_wave_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) serving_checkpoint_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) training_checkpoint_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) wave_age_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) labels_until_wave_due: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) seconds_until_wave_due: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) pending_gate_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) confirmed_anchor_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) exact_frontier_anchor_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) anchor_freshness_labels: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) generation_zero_quota_valid: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model_seed_survival_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model_seed_win_rate: Option<f64>,
    #[serde(default)]
    pub(crate) scheduler_debt_basis_micros: BTreeMap<SchedulerLaneV4, i128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) shadow_divergence_count: Option<u64>,
}

impl RatchetLiveTelemetryV2 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != "RatchetLiveTelemetry.v2" {
            bail!("telemetry schema is not v2");
        }
        for (label, value) in [
            (
                "telemetry serving wave",
                self.serving_wave_sha256.as_deref(),
            ),
            (
                "telemetry serving checkpoint",
                self.serving_checkpoint_sha256.as_deref(),
            ),
            (
                "telemetry training checkpoint",
                self.training_checkpoint_sha256.as_deref(),
            ),
            (
                "telemetry confirmed anchor",
                self.confirmed_anchor_sha256.as_deref(),
            ),
            (
                "telemetry exact frontier anchor",
                self.exact_frontier_anchor_sha256.as_deref(),
            ),
        ] {
            if let Some(value) = value {
                validate_sha256(label, value)?;
            }
        }
        for rate in [self.model_seed_survival_rate, self.model_seed_win_rate]
            .into_iter()
            .flatten()
        {
            if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
                bail!("telemetry rate lies outside [0,1]");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct BroadShadowPilotManifestV1 {
    pub(crate) schema: String,
    pub(crate) receipt_count: usize,
    pub(crate) receipts_manifest_sha256: String,
    pub(crate) genome_token_mapping_sha256: String,
    pub(crate) grammar_sha256: String,
    pub(crate) registry_sha256: String,
    pub(crate) vocabulary_sha256: String,
    pub(crate) mmap_corpus_sha256: String,
    pub(crate) trainer_receipt_sha256: String,
    pub(crate) inference_receipt_sha256: String,
    pub(crate) gate_receipt_sha256: String,
    pub(crate) pilot_only: bool,
    pub(crate) synthetic_only: bool,
}

impl BroadShadowPilotManifestV1 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != "BroadShadowPilotManifest.v1"
            || self.receipt_count != 20_000
            || !self.pilot_only
            || !self.synthetic_only
        {
            bail!("broad-shadow pilot is not the frozen 20,000-receipt synthetic campaign");
        }
        validate_hashes([
            (
                "pilot receipts manifest",
                self.receipts_manifest_sha256.as_str(),
            ),
            (
                "pilot genome token mapping",
                self.genome_token_mapping_sha256.as_str(),
            ),
            ("pilot grammar", self.grammar_sha256.as_str()),
            ("pilot registry", self.registry_sha256.as_str()),
            ("pilot vocabulary", self.vocabulary_sha256.as_str()),
            ("pilot mmap corpus", self.mmap_corpus_sha256.as_str()),
            (
                "pilot trainer receipt",
                self.trainer_receipt_sha256.as_str(),
            ),
            (
                "pilot inference receipt",
                self.inference_receipt_sha256.as_str(),
            ),
            ("pilot gate receipt", self.gate_receipt_sha256.as_str()),
        ])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct FrozenRealSplitsV1 {
    pub(crate) schema: String,
    pub(crate) training: Vec<String>,
    pub(crate) cold_development: Vec<String>,
    pub(crate) prospective_gate: Vec<String>,
    pub(crate) double_holdout: Vec<String>,
}

impl FrozenRealSplitsV1 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != "FrozenRealDatasetSplits.v1"
            || self.training.len() != 2_090
            || self.cold_development.len() != 119
            || self.prospective_gate.len() != 64
            || self.double_holdout.len() != 49
        {
            bail!("real-data splits do not match frozen 2090/119/64/49 topology");
        }
        let mut all = BTreeSet::new();
        for (label, split) in [
            ("training split", &self.training),
            ("cold-development split", &self.cold_development),
            ("prospective split", &self.prospective_gate),
            ("double-holdout split", &self.double_holdout),
        ] {
            for dataset in split {
                validate_sha256(label, dataset)?;
                if !all.insert(dataset.as_str()) {
                    bail!("frozen real-data splits overlap");
                }
            }
        }
        Ok(())
    }

    pub(crate) fn sealed_holdout_hashes(&self) -> BTreeSet<String> {
        self.double_holdout.iter().cloned().collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TrainingExperimentManifestV1 {
    pub(crate) schema: String,
    pub(crate) real_only_experiment_sha256: String,
    pub(crate) real_plus_synthetic_experiment_sha256: String,
    pub(crate) seeds: Vec<u64>,
    pub(crate) ensemble_members: usize,
    pub(crate) epochs: usize,
    pub(crate) minimum_early_stop_epoch: usize,
    pub(crate) patience: usize,
    pub(crate) track_2_is_only_sequence_positive: bool,
    pub(crate) auxiliary_rows_are_masked: bool,
    pub(crate) sealed_holdout_opened: bool,
}

impl TrainingExperimentManifestV1 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != "ProductionModelExperimentManifest.v1"
            || self.seeds != [17, 29, 43]
            || self.ensemble_members != 3
            || self.epochs != 300
            || self.minimum_early_stop_epoch != 100
            || self.patience != 30
            || !self.track_2_is_only_sequence_positive
            || !self.auxiliary_rows_are_masked
            || self.sealed_holdout_opened
        {
            bail!("production model experiment drifted from its frozen evidence protocol");
        }
        validate_hashes([
            (
                "real-only experiment",
                self.real_only_experiment_sha256.as_str(),
            ),
            (
                "real-plus-synthetic experiment",
                self.real_plus_synthetic_experiment_sha256.as_str(),
            ),
        ])
    }
}
