use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ArtifactReleaseV1 {
    pub(crate) release: String,
    pub(crate) sha256: String,
}

impl ArtifactReleaseV1 {
    pub(in crate::ratchet) fn validate(&self, label: &str) -> Result<()> {
        if self.release.trim().is_empty() {
            bail!("{label} release is empty");
        }
        validate_sha256(label, &self.sha256)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WaveGateKindV1 {
    Offline119,
    Prospective64,
    RecurringTrainOnly,
    Corrector,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct WaveGateReceiptV1 {
    pub(crate) kind: WaveGateKindV1,
    pub(crate) receipt_sha256: String,
    pub(crate) dataset_manifest_sha256: String,
    pub(crate) dataset_count: usize,
    pub(crate) dataset_sha256s: Vec<String>,
    pub(crate) train_only: bool,
    pub(crate) passed: bool,
    pub(crate) genome_validity: f64,
    pub(crate) mean_normalized_regret_reduction: f64,
    pub(crate) median_robust_r2_improvement: f64,
    pub(crate) median_wall_time_ratio: f64,
    pub(crate) safety_delta: f64,
    pub(crate) pairwise_accuracy_delta: f64,
    pub(crate) normalized_regret_bootstrap_lower: f64,
    pub(crate) capture_1_delta: f64,
    pub(crate) capture_3_delta: f64,
    pub(crate) validity_delta: f64,
    pub(crate) median_time_to_champion_delta: f64,
    pub(crate) corrector_grouped_time_forward_passed: bool,
    pub(crate) corrector_pop256_system_passed: bool,
}

impl WaveGateReceiptV1 {
    pub(in crate::ratchet) fn validate(&self) -> Result<()> {
        validate_hashes([
            ("gate receipt", self.receipt_sha256.as_str()),
            (
                "gate dataset manifest",
                self.dataset_manifest_sha256.as_str(),
            ),
        ])?;
        if self.dataset_sha256s.len() != self.dataset_count {
            bail!("wave gate dataset membership does not match its declared count");
        }
        for dataset in &self.dataset_sha256s {
            validate_sha256("wave gate dataset", dataset)?;
        }
        require_unique(
            "wave gate datasets",
            self.dataset_sha256s.iter().map(String::as_str),
        )?;
        let finite = [
            self.genome_validity,
            self.mean_normalized_regret_reduction,
            self.median_robust_r2_improvement,
            self.median_wall_time_ratio,
            self.safety_delta,
            self.pairwise_accuracy_delta,
            self.normalized_regret_bootstrap_lower,
            self.capture_1_delta,
            self.capture_3_delta,
            self.validity_delta,
            self.median_time_to_champion_delta,
        ]
        .iter()
        .all(|value| value.is_finite());
        if !finite || !self.passed {
            bail!("wave gate is partial, non-finite, or rejected");
        }
        match self.kind {
            WaveGateKindV1::Offline119 => {
                if self.dataset_count != 119
                    || self.mean_normalized_regret_reduction < 0.10
                    || self.normalized_regret_bootstrap_lower <= 0.0
                    || self.pairwise_accuracy_delta < -0.01
                    || self.validity_delta < -0.01
                {
                    bail!(
                        "offline bootstrap gate does not satisfy the frozen 119-dataset thresholds"
                    );
                }
            }
            WaveGateKindV1::Prospective64 => {
                if self.dataset_count != 64
                    || self.normalized_regret_bootstrap_lower <= 0.0
                    || self.median_robust_r2_improvement < 0.0
                    || self.validity_delta < -0.01
                    || self.median_wall_time_ratio > 1.25
                {
                    bail!("prospective bootstrap gate does not satisfy the frozen 64-dataset thresholds");
                }
            }
            WaveGateKindV1::RecurringTrainOnly => {
                if !self.train_only
                    || self.genome_validity < 0.99
                    || self.safety_delta < 0.0
                    || self.pairwise_accuracy_delta < -0.002
                    || self.normalized_regret_bootstrap_lower <= 0.0
                    || self.capture_1_delta < -0.01
                    || self.capture_3_delta < -0.01
                    || self.validity_delta < -0.01
                    || self.median_time_to_champion_delta > 0.0
                {
                    bail!("recurring train-only gate thresholds are not satisfied");
                }
            }
            WaveGateKindV1::Corrector => {
                if !self.train_only
                    || !self.corrector_grouped_time_forward_passed
                    || !self.corrector_pop256_system_passed
                {
                    bail!("corrector gate lacks grouped/time-forward or pop-256 evidence");
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct LearningWaveManifestV1 {
    pub(crate) schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parent_wave_sha256: Option<String>,
    pub(crate) accepted_at_unix_seconds: u64,
    pub(crate) training_label_cutoff: u64,
    pub(crate) replay_manifest_sha256: String,
    pub(crate) corpus_manifest_sha256: String,
    pub(crate) policy_checkpoint_sha256: String,
    pub(crate) resume_checkpoint_sha256: String,
    pub(crate) corrector_sha256: String,
    pub(crate) registry: ArtifactReleaseV1,
    pub(crate) vocabulary: ArtifactReleaseV1,
    pub(crate) encoding_catalog: ArtifactReleaseV1,
    pub(crate) evaluator: ArtifactReleaseV1,
    pub(crate) sidecar: ArtifactReleaseV1,
    #[serde(default)]
    pub(crate) gate_receipts: Vec<WaveGateReceiptV1>,
    pub(crate) labels_since_parent: u64,
    pub(crate) seconds_since_parent: u64,
    pub(crate) synthetic_data_ratio: f64,
    pub(crate) pilot_only: bool,
    pub(crate) publication_sha256: String,
}

impl LearningWaveManifestV1 {
    pub(in crate::ratchet) fn publication_digest(&self) -> Result<String> {
        let mut unsigned = self.clone();
        unsigned.publication_sha256.clear();
        Ok(sha256_bytes(&canonical_json(&unsigned)?))
    }

    #[cfg(test)]
    pub(crate) fn seal(mut self) -> Result<Self> {
        self.publication_sha256 = self.publication_digest()?;
        Ok(self)
    }

    pub(crate) fn validate(
        &self,
        parent: Option<(&str, &LearningWaveManifestV1)>,
        sealed_holdout_hashes: &BTreeSet<String>,
    ) -> Result<()> {
        if self.schema != WAVE_SCHEMA || self.pilot_only {
            bail!("serving wave is not an authoritative LearningWaveManifest.v1");
        }
        validate_hashes([
            ("wave replay manifest", self.replay_manifest_sha256.as_str()),
            ("wave corpus manifest", self.corpus_manifest_sha256.as_str()),
            (
                "wave policy checkpoint",
                self.policy_checkpoint_sha256.as_str(),
            ),
            (
                "wave resume checkpoint",
                self.resume_checkpoint_sha256.as_str(),
            ),
            ("wave corrector", self.corrector_sha256.as_str()),
            ("wave publication", self.publication_sha256.as_str()),
        ])?;
        for (label, release) in [
            ("registry", &self.registry),
            ("vocabulary", &self.vocabulary),
            ("encoding catalog", &self.encoding_catalog),
            ("evaluator", &self.evaluator),
            ("sidecar", &self.sidecar),
        ] {
            release.validate(label)?;
        }
        if !(0.0..=1.0).contains(&self.synthetic_data_ratio)
            || !self.synthetic_data_ratio.is_finite()
        {
            bail!("wave synthetic-data ratio is invalid");
        }
        if self.publication_sha256 != self.publication_digest()? {
            bail!("wave publication hash does not bind its manifest");
        }
        require_unique(
            "wave gate kinds",
            self.gate_receipts
                .iter()
                .map(|receipt| receipt.kind.clone()),
        )?;
        for receipt in &self.gate_receipts {
            receipt.validate()?;
            if sealed_holdout_hashes.contains(&receipt.dataset_manifest_sha256)
                || receipt
                    .dataset_sha256s
                    .iter()
                    .any(|dataset| sealed_holdout_hashes.contains(dataset))
            {
                bail!("sealed double-holdout evidence appears in a wave gate");
            }
        }

        match parent {
            None => {
                if self.parent_wave_sha256.is_some() {
                    bail!("bootstrap wave unexpectedly names a parent");
                }
                for required in [
                    WaveGateKindV1::Offline119,
                    WaveGateKindV1::Prospective64,
                    WaveGateKindV1::Corrector,
                ] {
                    if !self
                        .gate_receipts
                        .iter()
                        .any(|receipt| receipt.kind == required)
                    {
                        bail!("bootstrap wave has incomplete initial production gates");
                    }
                }
            }
            Some((parent_sha256, parent_manifest)) => {
                validate_sha256("parent wave", parent_sha256)?;
                if self.parent_wave_sha256.as_deref() != Some(parent_sha256)
                    || self.accepted_at_unix_seconds <= parent_manifest.accepted_at_unix_seconds
                    || self.training_label_cutoff <= parent_manifest.training_label_cutoff
                    || self.labels_since_parent < MIN_WAVE_LABELS
                    || self.seconds_since_parent < MIN_WAVE_AGE_SECONDS
                    || self.training_label_cutoff - parent_manifest.training_label_cutoff
                        != self.labels_since_parent
                    || self.accepted_at_unix_seconds - parent_manifest.accepted_at_unix_seconds
                        != self.seconds_since_parent
                    || self.accepted_at_unix_seconds - parent_manifest.accepted_at_unix_seconds
                        < MIN_WAVE_AGE_SECONDS
                {
                    bail!("wave parentage or label/time eligibility is invalid");
                }
                for required in [
                    WaveGateKindV1::RecurringTrainOnly,
                    WaveGateKindV1::Corrector,
                ] {
                    if !self
                        .gate_receipts
                        .iter()
                        .any(|receipt| receipt.kind == required)
                    {
                        bail!("recurring wave has incomplete train-only gates");
                    }
                }
                if self.gate_receipts.iter().any(|receipt| {
                    matches!(
                        receipt.kind,
                        WaveGateKindV1::Offline119 | WaveGateKindV1::Prospective64
                    )
                }) {
                    bail!("one-time bootstrap gates were reused for a recurring wave");
                }
            }
        }
        Ok(())
    }

    pub(crate) fn manifest_sha256(&self) -> Result<String> {
        Ok(sha256_bytes(&canonical_json(self)?))
    }
}
