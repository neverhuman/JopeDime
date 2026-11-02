use super::*;

mod validation;
pub(crate) use validation::{
    canonical_json, require_unique, sha256_bytes, validate_hashes, validate_sha256,
    validate_track_history,
};

pub(crate) const MODEL_PROPOSAL_COUNT: usize = 96;
pub(crate) const MODEL_RETAINED_COUNT: usize = 64;
pub(crate) const GENERATION_ZERO_SIZE: usize = 256;
pub(crate) const REPLAY_STEPS: usize = 100;
pub(crate) const DATASETS_PER_MICRO_BATCH: usize = 8;
pub(crate) const MIN_WAVE_LABELS: u64 = 25_000;
pub(crate) const MIN_WAVE_AGE_SECONDS: u64 = 30 * 60;

pub(crate) const DATASET_STATE_SCHEMA: &str = "DatasetLearningState.v2";
pub(crate) const WAVE_SCHEMA: &str = "LearningWaveManifest.v1";
pub(crate) const SEED_REQUEST_V2_SCHEMA: &str = "SeedInferenceRequest.v2";
pub(crate) const SEED_RESPONSE_V2_SCHEMA: &str = "SeedInferenceResponse.v2";
pub(crate) const GENERATION_ZERO_V3_SCHEMA: &str = "Generation0Receipt.v3";
pub(crate) const ADMISSION_V3_SCHEMA: &str = "AttemptAdmissionManifest.v3";
pub(crate) const REPLAY_V2_SCHEMA: &str = "ReplayManifest.v2";
pub(crate) const MICRO_BATCH_SCHEMA: &str = "LearningMicroBatchManifest.v1";
pub(crate) const TRAINING_HEAD_SCHEMA: &str = "TrainingHeadManifest.v1";
pub(crate) const POINTER_SCHEMA: &str = "RatchetArtifactPointer.v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct CandidateIdentityV1 {
    pub(crate) canonical_genome: String,
    pub(crate) semantic_sha256: String,
}

impl CandidateIdentityV1 {
    #[cfg(test)]
    pub(crate) fn from_canonical(canonical_genome: String) -> Result<Self> {
        let parsed = GenomeIr::parse(&canonical_genome).context("parse canonical genome")?;
        if parsed.canonical() != canonical_genome {
            bail!("candidate genome is not canonical");
        }
        Ok(Self {
            semantic_sha256: sha256_bytes(canonical_genome.as_bytes()),
            canonical_genome,
        })
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_sha256("candidate semantic identity", &self.semantic_sha256)?;
        let parsed = GenomeIr::parse(&self.canonical_genome).context("parse candidate genome")?;
        if self.canonical_genome.trim() != self.canonical_genome
            || parsed.canonical() != self.canonical_genome
            || sha256_bytes(self.canonical_genome.as_bytes()) != self.semantic_sha256
        {
            bail!("candidate canonical genome and semantic identity disagree");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ObservationKeyV2 {
    pub(crate) dataset_sha256: String,
    pub(crate) semantic_sha256: String,
    pub(crate) evaluator_sha256: String,
    pub(crate) folds_sha256: String,
    pub(crate) protocol_sha256: String,
}

impl ObservationKeyV2 {
    pub(crate) fn validate(&self, dataset_sha256: &str) -> Result<()> {
        validate_hashes([
            ("observation dataset", self.dataset_sha256.as_str()),
            ("observation candidate", self.semantic_sha256.as_str()),
            ("observation evaluator", self.evaluator_sha256.as_str()),
            ("observation folds", self.folds_sha256.as_str()),
            ("observation protocol", self.protocol_sha256.as_str()),
        ])?;
        if self.dataset_sha256 != dataset_sha256 {
            bail!("observation key belongs to another dataset");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LabelStatusV2 {
    Exact,
    Invalid,
    Timeout,
    RuntimeFailure,
    Calibration,
    Comparison,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct LabelValueV2 {
    pub(crate) status: LabelStatusV2,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) robust_fitness: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) wall_time_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) comparison_won: Option<bool>,
}

impl LabelValueV2 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.robust_fitness.is_some_and(|value| !value.is_finite())
            || self
                .wall_time_seconds
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            bail!("label contains non-finite or negative metrics");
        }
        if matches!(self.status, LabelStatusV2::Exact) && self.robust_fitness.is_none() {
            bail!("exact label is missing robust fitness");
        }
        if !matches!(self.status, LabelStatusV2::Comparison) && self.comparison_won.is_some() {
            bail!("comparison outcome is present on a non-comparison label");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct PhysicalObservationV2 {
    pub(crate) sequence: u64,
    pub(crate) observation_sha256: String,
    pub(crate) key: ObservationKeyV2,
    pub(crate) receipt_sha256: String,
    pub(crate) observed_at_unix_seconds: u64,
    pub(crate) provenance_complete: bool,
    pub(crate) synthetic: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct LabelRevisionV2 {
    pub(crate) sequence: u64,
    pub(crate) key: ObservationKeyV2,
    pub(crate) revision: u64,
    pub(crate) observation_sha256: String,
    pub(crate) revision_receipt_sha256: String,
    pub(crate) committed_at_unix_seconds: u64,
    pub(crate) value: LabelValueV2,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SelectionAuthorityV2 {
    TrainOnlyExact,
    ConfirmedGlobalCv,
    FeedbackTestTrack2,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct TrackHistoryEntryV2 {
    pub(crate) sequence: u64,
    pub(crate) candidate: CandidateIdentityV1,
    pub(crate) authority: SelectionAuthorityV2,
    pub(crate) robust_fitness: f64,
    pub(crate) receipt_sha256: String,
    pub(crate) committed_at_unix_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct GlobalCvFrontierMemberV2 {
    pub(crate) candidate: CandidateIdentityV1,
    pub(crate) robust_fitness: f64,
    pub(crate) comparable_fold_count: usize,
    pub(crate) confirmed: bool,
    pub(crate) receipt_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct PredictionSnapshotV2 {
    pub(crate) candidate: CandidateIdentityV1,
    pub(crate) checkpoint_sha256: String,
    pub(crate) prediction: f64,
    pub(crate) recorded_at_unix_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub(crate) struct LearningCountersV2 {
    pub(crate) provenance_complete_labels: u64,
    pub(crate) exact_labels: u64,
    pub(crate) synthetic_labels: u64,
    pub(crate) labels_since_serving_wave: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub(crate) struct StruggleStateV2 {
    pub(crate) consecutive_no_improvement_attempts: u64,
    pub(crate) confirmation_required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub(crate) struct SchedulingMetadataV2 {
    pub(crate) last_scheduled_at_unix_seconds: Option<u64>,
    pub(crate) measured_cpu_micros: u64,
    pub(crate) scheduler_draw_count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LegacyAnchorImportV2 {
    pub(crate) candidate: CandidateIdentityV1,
    pub(crate) dataset_sha256: String,
    pub(crate) folds_sha256: String,
    pub(crate) evaluator_sha256: String,
    pub(crate) receipt_sha256: String,
    pub(crate) verified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct DatasetLearningStateV2 {
    pub(crate) schema: String,
    pub(crate) dataset_sha256: String,
    pub(crate) state_sequence: u64,
    #[serde(default)]
    pub(crate) observations: Vec<PhysicalObservationV2>,
    #[serde(default)]
    pub(crate) label_revisions: Vec<LabelRevisionV2>,
    #[serde(default)]
    pub(crate) track_1_history: Vec<TrackHistoryEntryV2>,
    #[serde(default)]
    pub(crate) track_2_history: Vec<TrackHistoryEntryV2>,
    #[serde(default)]
    pub(crate) global_cv_frontier: Vec<GlobalCvFrontierMemberV2>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) confirmed_cv_champion: Option<CandidateIdentityV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) latest_exact_track_1_frontier: Option<CandidateIdentityV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) latest_model_prediction: Option<PredictionSnapshotV2>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) latest_corrector_prediction: Option<PredictionSnapshotV2>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) legacy_anchor_import: Option<LegacyAnchorImportV2>,
    pub(crate) counters: LearningCountersV2,
    pub(crate) struggle: StruggleStateV2,
    pub(crate) scheduling: SchedulingMetadataV2,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CurrentLabelV2 {
    pub(crate) revision: u64,
    pub(crate) observation_sha256: String,
    pub(crate) value: LabelValueV2,
}

impl DatasetLearningStateV2 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != DATASET_STATE_SCHEMA {
            bail!("dataset learning state schema is not v2");
        }
        validate_sha256("dataset state dataset", &self.dataset_sha256)?;
        if self.state_sequence != (self.observations.len() + self.label_revisions.len()) as u64 {
            bail!("dataset state sequence does not match append-only event count");
        }

        let mut observation_ids = BTreeMap::new();
        let mut previous_time = 0;
        for (index, observation) in self.observations.iter().enumerate() {
            if observation.sequence != index as u64 + 1 {
                bail!("physical observation sequence is not contiguous");
            }
            observation.key.validate(&self.dataset_sha256)?;
            validate_hashes([
                (
                    "physical observation identity",
                    observation.observation_sha256.as_str(),
                ),
                (
                    "physical observation receipt",
                    observation.receipt_sha256.as_str(),
                ),
            ])?;
            if !observation.provenance_complete {
                bail!("authoritative state contains provenance-incomplete observation");
            }
            if observation.observed_at_unix_seconds < previous_time {
                bail!("physical observation timestamps move backwards");
            }
            previous_time = observation.observed_at_unix_seconds;
            if observation_ids
                .insert(observation.observation_sha256.as_str(), &observation.key)
                .is_some()
            {
                bail!("physical observation identity is duplicated");
            }
        }

        let mut next_revision = BTreeMap::<&ObservationKeyV2, u64>::new();
        for (index, revision) in self.label_revisions.iter().enumerate() {
            if revision.sequence != index as u64 + 1 {
                bail!("label revision sequence is not contiguous");
            }
            revision.key.validate(&self.dataset_sha256)?;
            validate_hashes([
                (
                    "label revision observation",
                    revision.observation_sha256.as_str(),
                ),
                (
                    "label revision receipt",
                    revision.revision_receipt_sha256.as_str(),
                ),
            ])?;
            if observation_ids.get(revision.observation_sha256.as_str()) != Some(&&revision.key) {
                bail!("label revision references an unknown or differently keyed observation");
            }
            let expected = next_revision.entry(&revision.key).or_insert(1);
            if revision.revision != *expected {
                bail!("label revisions are not monotonic and contiguous for their key");
            }
            *expected += 1;
            revision.value.validate()?;
        }

        validate_track_history(
            "Track-1",
            &self.track_1_history,
            SelectionAuthorityV2::TrainOnlyExact,
        )?;
        validate_track_history(
            "Track-2",
            &self.track_2_history,
            SelectionAuthorityV2::FeedbackTestTrack2,
        )?;
        self.validate_frontier_and_anchors()?;
        self.validate_predictions()?;
        self.validate_counters()?;
        self.validate_legacy_import()?;
        Ok(())
    }

    pub(crate) fn validate_frontier_and_anchors(&self) -> Result<()> {
        if self.global_cv_frontier.len() > 5 {
            bail!("Global-CV frontier exceeds five candidates");
        }
        require_unique(
            "Global-CV frontier",
            self.global_cv_frontier
                .iter()
                .map(|member| member.candidate.semantic_sha256.as_str()),
        )?;
        for member in &self.global_cv_frontier {
            member.candidate.validate()?;
            validate_sha256("Global-CV receipt", &member.receipt_sha256)?;
            if !member.robust_fitness.is_finite() {
                bail!("Global-CV frontier contains non-finite fitness");
            }
            if member.confirmed && member.comparable_fold_count < 5 {
                bail!("Global-CV candidate was confirmed with fewer than five comparable folds");
            }
        }
        if let Some(anchor) = &self.confirmed_cv_champion {
            anchor.validate()?;
            if self.global_cv_frontier.len() < 5
                || !self.global_cv_frontier.iter().any(|member| {
                    member.confirmed && member.candidate.semantic_sha256 == anchor.semantic_sha256
                })
            {
                bail!("confirmed-CV champion lacks a five-candidate confirmed frontier");
            }
        }
        if let Some(anchor) = &self.latest_exact_track_1_frontier {
            anchor.validate()?;
            if !self.track_1_history.iter().any(|entry| {
                entry.candidate.semantic_sha256 == anchor.semantic_sha256
                    && entry.authority == SelectionAuthorityV2::TrainOnlyExact
            }) {
                bail!("latest exact Track-1 anchor is not selection-clean");
            }
        }
        if (self.confirmed_cv_champion.is_none() || self.latest_exact_track_1_frontier.is_none())
            && !self.struggle.confirmation_required
        {
            bail!("state with a missing known-best anchor is not restricted to confirmation");
        }
        Ok(())
    }

    pub(crate) fn validate_predictions(&self) -> Result<()> {
        for prediction in [
            self.latest_model_prediction.as_ref(),
            self.latest_corrector_prediction.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            prediction.candidate.validate()?;
            validate_sha256("prediction checkpoint", &prediction.checkpoint_sha256)?;
            if !prediction.prediction.is_finite() {
                bail!("latest prediction is non-finite");
            }
        }
        Ok(())
    }

    pub(crate) fn validate_counters(&self) -> Result<()> {
        let observations = self
            .observations
            .iter()
            .map(|observation| {
                (
                    observation.observation_sha256.as_str(),
                    observation.synthetic,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let provenance = self.label_revisions.len() as u64;
        let synthetic = self
            .label_revisions
            .iter()
            .filter(|revision| observations[revision.observation_sha256.as_str()])
            .count() as u64;
        let exact = self
            .current_labels()
            .values()
            .filter(|label| matches!(label.value.status, LabelStatusV2::Exact))
            .count() as u64;
        if self.counters.provenance_complete_labels != provenance
            || self.counters.synthetic_labels != synthetic
            || self.counters.exact_labels != exact
            || self.counters.labels_since_serving_wave > self.counters.provenance_complete_labels
        {
            bail!("dataset learning counters do not reconstruct from the append-only ledger");
        }
        Ok(())
    }

    pub(crate) fn validate_legacy_import(&self) -> Result<()> {
        let Some(import) = &self.legacy_anchor_import else {
            return Ok(());
        };
        import.candidate.validate()?;
        validate_hashes([
            ("legacy dataset", import.dataset_sha256.as_str()),
            ("legacy folds", import.folds_sha256.as_str()),
            ("legacy evaluator", import.evaluator_sha256.as_str()),
            ("legacy receipt", import.receipt_sha256.as_str()),
        ])?;
        if !import.verified || import.dataset_sha256 != self.dataset_sha256 {
            bail!("unverified legacy champion cannot become an authoritative anchor");
        }
        Ok(())
    }

    pub(crate) fn current_labels(&self) -> BTreeMap<ObservationKeyV2, CurrentLabelV2> {
        let mut current = BTreeMap::new();
        for revision in &self.label_revisions {
            current.insert(
                revision.key.clone(),
                CurrentLabelV2 {
                    revision: revision.revision,
                    observation_sha256: revision.observation_sha256.clone(),
                    value: revision.value.clone(),
                },
            );
        }
        current
    }

    pub(crate) fn validate_successor(&self, next: &Self) -> Result<()> {
        self.validate()?;
        next.validate()?;
        if self.dataset_sha256 != next.dataset_sha256
            || next.state_sequence < self.state_sequence
            || !next.observations.starts_with(&self.observations)
            || !next.label_revisions.starts_with(&self.label_revisions)
            || !next.track_1_history.starts_with(&self.track_1_history)
            || !next.track_2_history.starts_with(&self.track_2_history)
        {
            bail!("dataset learning state successor rewrites append-only history");
        }
        Ok(())
    }

    pub(crate) fn snapshot_sha256(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256_bytes(&canonical_json(self)?))
    }
}
