use super::*;
/// Selection-clean native feature row for the Rust corrector. There is
/// intentionally no Track-2/test-selected input in this contract.
#[cfg(test)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]

pub(crate) struct NativeFitnessFeaturesV1 {
    pub(crate) schema: String,
    pub(crate) candidate_semantic_sha256: String,
    pub(crate) confirmed_cv_anchor_fitness: f64,
    pub(crate) latest_exact_track_1_fitness: f64,
    pub(crate) model_prediction: f64,
    pub(crate) family_observation_count: u64,
    pub(crate) family_mean_residual: f64,
    pub(crate) family_shrunk_residual: f64,
}

#[cfg(test)]
impl NativeFitnessFeaturesV1 {
    pub(crate) fn build(
        state: &DatasetLearningStateV2,
        candidate_semantic_sha256: String,
        model_prediction: f64,
        family_observation_count: u64,
        family_mean_residual: f64,
        shrinkage_strength: f64,
    ) -> Result<Self> {
        state.validate()?;
        validate_sha256("corrector feature candidate", &candidate_semantic_sha256)?;
        if !model_prediction.is_finite()
            || !family_mean_residual.is_finite()
            || !shrinkage_strength.is_finite()
            || shrinkage_strength < 0.0
        {
            bail!("corrector feature inputs are non-finite or invalid");
        }
        let confirmed = state
            .confirmed_cv_champion
            .as_ref()
            .context("corrector features require a confirmed-CV anchor")?;
        let confirmed_cv_anchor_fitness = state
            .global_cv_frontier
            .iter()
            .find(|member| member.candidate.semantic_sha256 == confirmed.semantic_sha256)
            .context("confirmed-CV anchor is absent from frontier")?
            .robust_fitness;
        let latest_exact_track_1_fitness = state
            .track_1_history
            .last()
            .context("corrector features require Track-1 history")?
            .robust_fitness;
        let denominator = family_observation_count as f64 + shrinkage_strength;
        let weight = if denominator == 0.0 {
            0.0
        } else {
            family_observation_count as f64 / denominator
        };
        let features = Self {
            schema: "NativeFitnessFeatures.v1".into(),
            candidate_semantic_sha256,
            confirmed_cv_anchor_fitness,
            latest_exact_track_1_fitness,
            model_prediction,
            family_observation_count,
            family_mean_residual,
            family_shrunk_residual: weight * family_mean_residual,
        };
        features.validate()?;
        Ok(features)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_sha256(
            "native corrector candidate",
            &self.candidate_semantic_sha256,
        )?;
        if self.schema != "NativeFitnessFeatures.v1"
            || [
                self.confirmed_cv_anchor_fitness,
                self.latest_exact_track_1_fitness,
                self.model_prediction,
                self.family_mean_residual,
                self.family_shrunk_residual,
            ]
            .iter()
            .any(|value| !value.is_finite())
        {
            bail!("native corrector feature contract is invalid");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ArtifactReleaseV1 {
    pub(crate) release: String,
    pub(crate) sha256: String,
}

impl ArtifactReleaseV1 {
    pub(super) fn validate(&self, label: &str) -> Result<()> {
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
    pub(super) fn validate(&self) -> Result<()> {
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
    pub(super) fn publication_digest(&self) -> Result<String> {
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SeedInferenceRequestV2 {
    pub(crate) schema: String,
    pub(crate) learning_wave_sha256: String,
    pub(crate) dataset_encoding_sha256: String,
    pub(crate) checkpoint_sha256: String,
    pub(crate) corrector_sha256: String,
    pub(crate) registry_sha256: String,
    pub(crate) vocabulary_sha256: String,
    pub(crate) state_snapshot_sha256: String,
    pub(crate) draw_seed: u64,
    pub(crate) required_count: usize,
}

impl SeedInferenceRequestV2 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != SEED_REQUEST_V2_SCHEMA || self.required_count != MODEL_PROPOSAL_COUNT {
            bail!("promoted seed request is not the required v2/96 contract");
        }
        validate_hashes([
            ("seed learning wave", self.learning_wave_sha256.as_str()),
            (
                "seed dataset encoding",
                self.dataset_encoding_sha256.as_str(),
            ),
            ("seed checkpoint", self.checkpoint_sha256.as_str()),
            ("seed corrector", self.corrector_sha256.as_str()),
            ("seed registry", self.registry_sha256.as_str()),
            ("seed vocabulary", self.vocabulary_sha256.as_str()),
            ("seed state snapshot", self.state_snapshot_sha256.as_str()),
        ])
    }

    pub(crate) fn request_sha256(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256_bytes(&canonical_json(self)?))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct ModelProposalV2 {
    pub(crate) candidate: CandidateIdentityV1,
    pub(crate) reward_prediction: f64,
    pub(crate) disagreement_prediction: f64,
    pub(crate) novelty_prediction: f64,
    pub(crate) temperature_logit: f64,
}

impl ModelProposalV2 {
    pub(super) fn validate(&self) -> Result<()> {
        self.candidate.validate()?;
        if [
            self.reward_prediction,
            self.disagreement_prediction,
            self.novelty_prediction,
            self.temperature_logit,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            bail!("model proposal contains a non-finite prediction");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct SeedInferenceResponseV2 {
    pub(crate) schema: String,
    pub(crate) request_sha256: String,
    pub(crate) proposals: Vec<ModelProposalV2>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelRetentionBucketV2 {
    Reward,
    Disagreement,
    Novelty,
    SeededTemperature,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct RetainedModelProposalV2 {
    pub(crate) proposal: ModelProposalV2,
    pub(crate) bucket: ModelRetentionBucketV2,
    pub(crate) bucket_rank: usize,
}

impl SeedInferenceResponseV2 {
    pub(crate) fn validate_against(&self, request: &SeedInferenceRequestV2) -> Result<()> {
        request.validate()?;
        if self.schema != SEED_RESPONSE_V2_SCHEMA
            || self.request_sha256 != request.request_sha256()?
            || self.proposals.len() != MODEL_PROPOSAL_COUNT
        {
            bail!("Python sidecar response does not bind the v2 request or all 96 proposals");
        }
        validate_sha256("seed request receipt", &self.request_sha256)?;
        for proposal in &self.proposals {
            proposal.validate()?;
        }
        require_unique(
            "model proposals",
            self.proposals
                .iter()
                .map(|proposal| proposal.candidate.semantic_sha256.as_str()),
        )
    }

    pub(crate) fn retain_64(
        &self,
        request: &SeedInferenceRequestV2,
        excluded_semantics: &BTreeSet<String>,
    ) -> Result<Vec<RetainedModelProposalV2>> {
        self.validate_against(request)?;
        let available = self
            .proposals
            .iter()
            .enumerate()
            .filter(|(_, proposal)| {
                !excluded_semantics.contains(&proposal.candidate.semantic_sha256)
            })
            .map(|(index, proposal)| (index, proposal.clone()))
            .collect::<Vec<_>>();
        if available.len() < MODEL_RETAINED_COUNT {
            bail!("fewer than 64 unique valid non-archive model proposals remain");
        }

        let mut selected = BTreeSet::<usize>::new();
        let mut retained = Vec::with_capacity(MODEL_RETAINED_COUNT);
        select_ranked(
            &available,
            &mut selected,
            &mut retained,
            32,
            ModelRetentionBucketV2::Reward,
            |left, right| descending_f64(left.reward_prediction, right.reward_prediction),
        );
        select_ranked(
            &available,
            &mut selected,
            &mut retained,
            16,
            ModelRetentionBucketV2::Disagreement,
            |left, right| {
                descending_f64(left.disagreement_prediction, right.disagreement_prediction)
            },
        );
        select_ranked(
            &available,
            &mut selected,
            &mut retained,
            8,
            ModelRetentionBucketV2::Novelty,
            |left, right| descending_f64(left.novelty_prediction, right.novelty_prediction),
        );

        let mut temperature = available
            .iter()
            .filter(|(index, _)| !selected.contains(index))
            .map(|(index, proposal)| {
                (
                    *index,
                    proposal.clone(),
                    seeded_temperature_score(
                        request.draw_seed,
                        &proposal.candidate.semantic_sha256,
                        proposal.temperature_logit,
                    ),
                )
            })
            .collect::<Vec<_>>();
        temperature.sort_by(|left, right| {
            descending_f64(left.2, right.2).then_with(|| {
                left.1
                    .candidate
                    .semantic_sha256
                    .cmp(&right.1.candidate.semantic_sha256)
            })
        });
        for (rank, (index, proposal, _)) in temperature.into_iter().take(8).enumerate() {
            selected.insert(index);
            retained.push(RetainedModelProposalV2 {
                proposal,
                bucket: ModelRetentionBucketV2::SeededTemperature,
                bucket_rank: rank,
            });
        }
        if retained.len() != MODEL_RETAINED_COUNT {
            bail!("model retention subpool exhausted before exact 32/16/8/8 quotas");
        }
        Ok(retained)
    }
}

pub(super) fn descending_f64(left: f64, right: f64) -> Ordering {
    right.total_cmp(&left)
}

pub(super) fn select_ranked(
    available: &[(usize, ModelProposalV2)],
    selected: &mut BTreeSet<usize>,
    retained: &mut Vec<RetainedModelProposalV2>,
    count: usize,
    bucket: ModelRetentionBucketV2,
    comparison: impl Fn(&ModelProposalV2, &ModelProposalV2) -> Ordering,
) {
    let mut ranked = available
        .iter()
        .filter(|(index, _)| !selected.contains(index))
        .cloned()
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        comparison(&left.1, &right.1).then_with(|| {
            left.1
                .candidate
                .semantic_sha256
                .cmp(&right.1.candidate.semantic_sha256)
        })
    });
    for (rank, (index, proposal)) in ranked.into_iter().take(count).enumerate() {
        selected.insert(index);
        retained.push(RetainedModelProposalV2 {
            proposal,
            bucket: bucket.clone(),
            bucket_rank: rank,
        });
    }
}

pub(super) fn seeded_temperature_score(seed: u64, semantic_sha256: &str, logit: f64) -> f64 {
    let mut hash = Sha256::new();
    hash.update(b"jope-dime/seeded-temperature/v1\0");
    hash.update(seed.to_le_bytes());
    hash.update(semantic_sha256.as_bytes());
    let digest = hash.finalize();
    let raw = u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix"));
    let uniform = ((raw as f64) + 1.0) / ((u64::MAX as f64) + 2.0);
    logit - (-uniform.ln()).ln()
}

/// Read compatibility envelope. V1 remains deserializable for old receipts,
/// but callers must use `SeedInferenceResponseV2` for promoted authority.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct LegacySeedInferenceV1 {
    // Serde consumes this field when validating the v1 receipt shape.
    #[allow(
        dead_code,
        reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
    )]
    pub(crate) schema: String,
    // Preserve the complete v1 payload accepted by the compatibility reader.
    #[allow(
        dead_code,
        reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
    )]
    #[serde(flatten)]
    pub(crate) fields: BTreeMap<String, Value>,
}

pub(crate) fn read_seed_inference(bytes: &[u8]) -> Result<ReadableSeedInference> {
    let value: Value = serde_json::from_slice(bytes)?;
    match value.get("schema").and_then(Value::as_str) {
        Some(SEED_RESPONSE_V2_SCHEMA) => {
            Ok(ReadableSeedInference::V2(serde_json::from_value(value)?))
        }
        Some(schema) if schema.starts_with("SeedInference") && schema.ends_with(".v1") => {
            Ok(ReadableSeedInference::V1(serde_json::from_value(value)?))
        }
        _ => bail!("unknown seed inference receipt schema"),
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ReadableSeedInference {
    // Retain the typed payload used by the compatibility artifact validator.
    V1(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        LegacySeedInferenceV1,
    ),
    V2(SeedInferenceResponseV2),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct ModelRetentionReceiptV2 {
    pub(crate) schema: String,
    pub(crate) request_sha256: String,
    pub(crate) response_sha256: String,
    pub(crate) excluded_semantics_sha256: String,
    pub(crate) retained: Vec<RetainedModelProposalV2>,
}

impl ModelRetentionReceiptV2 {
    pub(crate) fn build(
        request: &SeedInferenceRequestV2,
        response: &SeedInferenceResponseV2,
        excluded_semantics: &BTreeSet<String>,
    ) -> Result<Self> {
        for semantic in excluded_semantics {
            validate_sha256("model-retention exclusion", semantic)?;
        }
        let retained = response.retain_64(request, excluded_semantics)?;
        let response_sha256 = sha256_bytes(&canonical_json(response)?);
        let excluded_semantics_sha256 = sha256_bytes(&canonical_json(excluded_semantics)?);
        let receipt = Self {
            schema: "ModelRetentionReceipt.v2".into(),
            request_sha256: request.request_sha256()?,
            response_sha256,
            excluded_semantics_sha256,
            retained,
        };
        receipt.validate(request, response, excluded_semantics)?;
        Ok(receipt)
    }

    pub(crate) fn validate(
        &self,
        request: &SeedInferenceRequestV2,
        response: &SeedInferenceResponseV2,
        excluded_semantics: &BTreeSet<String>,
    ) -> Result<()> {
        if self.schema != "ModelRetentionReceipt.v2"
            || self.request_sha256 != request.request_sha256()?
            || self.response_sha256 != sha256_bytes(&canonical_json(response)?)
            || self.excluded_semantics_sha256 != sha256_bytes(&canonical_json(excluded_semantics)?)
            || self.retained != response.retain_64(request, excluded_semantics)?
        {
            bail!("model-retention receipt does not reconstruct from Rust authority");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ReadableGeneration0 {
    // Retain the typed payload used by the compatibility artifact validator.
    Legacy(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        Value,
    ),
    // Retain the typed payload used by the compatibility artifact validator.
    V3(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        Generation0ReceiptV3,
    ),
}

pub(crate) fn read_generation_zero(bytes: &[u8]) -> Result<ReadableGeneration0> {
    let value: Value = serde_json::from_slice(bytes)?;
    match value.get("schema").and_then(Value::as_str) {
        Some(GENERATION_ZERO_V3_SCHEMA) => {
            Ok(ReadableGeneration0::V3(serde_json::from_value(value)?))
        }
        Some(schema) if schema.starts_with("Generation0Receipt.v") => {
            Ok(ReadableGeneration0::Legacy(value))
        }
        _ => bail!("unknown generation-zero receipt schema"),
    }
}

#[derive(Clone, Debug)]
// Preserve the existing compatibility receipt payload and allocation layout.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ReadableAdmission {
    // Retain the typed payload used by the compatibility artifact validator.
    Legacy(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        Value,
    ),
    // Retain the typed payload used by the compatibility artifact validator.
    V3(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        AttemptAdmissionManifestV3,
    ),
}

pub(crate) fn read_admission(bytes: &[u8]) -> Result<ReadableAdmission> {
    let value: Value = serde_json::from_slice(bytes)?;
    match value.get("schema").and_then(Value::as_str) {
        Some(ADMISSION_V3_SCHEMA) => Ok(ReadableAdmission::V3(serde_json::from_value(value)?)),
        Some(schema) if schema.starts_with("AttemptAdmissionManifest.v") => {
            Ok(ReadableAdmission::Legacy(value))
        }
        _ => bail!("unknown attempt-admission schema"),
    }
}

#[derive(Clone, Debug)]
// Preserve the existing compatibility receipt payload and allocation layout.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ReadableTelemetry {
    // Retain the typed payload used by the compatibility artifact validator.
    Legacy(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        Value,
    ),
    // Retain the typed payload used by the compatibility artifact validator.
    V2(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        RatchetLiveTelemetryV2,
    ),
}

pub(crate) fn read_telemetry(bytes: &[u8]) -> Result<ReadableTelemetry> {
    let value: Value = serde_json::from_slice(bytes)?;
    match value.get("schema").and_then(Value::as_str) {
        Some("RatchetLiveTelemetry.v2") => {
            Ok(ReadableTelemetry::V2(serde_json::from_value(value)?))
        }
        Some(schema) if schema.starts_with("RatchetLiveTelemetry.v") => {
            Ok(ReadableTelemetry::Legacy(value))
        }
        _ => bail!("unknown ratchet telemetry schema"),
    }
}
