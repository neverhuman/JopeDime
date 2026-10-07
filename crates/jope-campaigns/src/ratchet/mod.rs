//! Continuous-learning authority contracts.
//!
//! The historical campaign commands in this crate are intentionally frozen.
//! This module is an additive authority boundary for continuous learning:
//! Python may train and infer, but only these Rust contracts may retain model
//! proposals, admit a generation, advance a training head, or publish a
//! serving wave.

use anyhow::{bail, Context, Result};
#[cfg(test)]
use feat_math::evolve::{
    crossover_canonical_pipelines, mutate_canonical_pipeline, random_canonical_pipeline,
};
use feat_math::GenomeIr;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) const MODEL_PROPOSAL_COUNT: usize = 96;
pub(crate) const MODEL_RETAINED_COUNT: usize = 64;
pub(crate) const GENERATION_ZERO_SIZE: usize = 256;
pub(crate) const REPLAY_STEPS: usize = 100;
pub(crate) const DATASETS_PER_MICRO_BATCH: usize = 8;
pub(crate) const MIN_WAVE_LABELS: u64 = 25_000;
pub(crate) const MIN_WAVE_AGE_SECONDS: u64 = 30 * 60;

const DATASET_STATE_SCHEMA: &str = "DatasetLearningState.v2";
const WAVE_SCHEMA: &str = "LearningWaveManifest.v1";
const SEED_REQUEST_V2_SCHEMA: &str = "SeedInferenceRequest.v2";
const SEED_RESPONSE_V2_SCHEMA: &str = "SeedInferenceResponse.v2";
const GENERATION_ZERO_V3_SCHEMA: &str = "Generation0Receipt.v3";
const ADMISSION_V3_SCHEMA: &str = "AttemptAdmissionManifest.v3";
const REPLAY_V2_SCHEMA: &str = "ReplayManifest.v2";
const MICRO_BATCH_SCHEMA: &str = "LearningMicroBatchManifest.v1";
const TRAINING_HEAD_SCHEMA: &str = "TrainingHeadManifest.v1";
const POINTER_SCHEMA: &str = "RatchetArtifactPointer.v1";

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        bail!("{label} is not a lowercase SHA-256");
    }
    Ok(())
}

fn validate_hashes<'a>(values: impl IntoIterator<Item = (&'a str, &'a str)>) -> Result<()> {
    for (label, value) in values {
        validate_sha256(label, value)?;
    }
    Ok(())
}

fn require_unique<T: Ord>(label: &str, values: impl IntoIterator<Item = T>) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            bail!("{label} contains a duplicate");
        }
    }
    Ok(())
}

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

    fn validate(&self) -> Result<()> {
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
    fn validate(&self, dataset_sha256: &str) -> Result<()> {
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
    fn validate(&self) -> Result<()> {
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

    fn validate_frontier_and_anchors(&self) -> Result<()> {
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

    fn validate_predictions(&self) -> Result<()> {
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

    fn validate_counters(&self) -> Result<()> {
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

    fn validate_legacy_import(&self) -> Result<()> {
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

fn validate_track_history(
    label: &str,
    entries: &[TrackHistoryEntryV2],
    required_authority: SelectionAuthorityV2,
) -> Result<()> {
    let mut previous_time = 0;
    for (index, entry) in entries.iter().enumerate() {
        if entry.sequence != index as u64 + 1 || entry.authority != required_authority {
            bail!("{label} history has invalid sequence or authority");
        }
        entry.candidate.validate()?;
        validate_sha256(&format!("{label} receipt"), &entry.receipt_sha256)?;
        if !entry.robust_fitness.is_finite() || entry.committed_at_unix_seconds < previous_time {
            bail!("{label} history contains invalid fitness or time");
        }
        previous_time = entry.committed_at_unix_seconds;
    }
    Ok(())
}

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
    fn validate(&self, label: &str) -> Result<()> {
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
    fn validate(&self) -> Result<()> {
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
    fn publication_digest(&self) -> Result<String> {
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
    fn validate(&self) -> Result<()> {
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

fn descending_f64(left: f64, right: f64) -> Ordering {
    right.total_cmp(&left)
}

fn select_ranked(
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

fn seeded_temperature_score(seed: u64, semantic_sha256: &str, logit: f64) -> f64 {
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
    pub(crate) schema: String,
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
    V1(LegacySeedInferenceV1),
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
    Legacy(Value),
    V3(Generation0ReceiptV3),
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
    Legacy(Value),
    V3(AttemptAdmissionManifestV3),
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
    Legacy(Value),
    V2(RatchetLiveTelemetryV2),
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GprKernelFamilyV3 {
    Rbf,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
    Periodic,
    Linear,
    WhiteComposite,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "source", rename_all = "snake_case")]
pub(crate) enum CandidateProvenanceV3 {
    Archive {
        priority: usize,
    },
    Model {
        bucket: ModelRetentionBucketV2,
        bucket_rank: usize,
    },
    Gpr {
        kernel_family: GprKernelFamilyV3,
    },
    Mutation {
        seed: u64,
        parent_semantic_sha256: String,
    },
    Crossover {
        seed: u64,
        left_parent_semantic_sha256: String,
        right_parent_semantic_sha256: String,
    },
    Random {
        seed: u64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct GenerationCandidateV3 {
    pub(crate) position: usize,
    pub(crate) candidate: CandidateIdentityV1,
    pub(crate) provenance: CandidateProvenanceV3,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GenerationRoleV3 {
    ConfirmedCvChampion,
    LatestExactTrack1Frontier,
    CheckpointRewardTop1,
    LatestModelPrediction,
    LatestCorrectorPrediction,
    LatestTrack1Winner,
    LatestTrack2Winner,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct GenerationRoleBindingV3 {
    pub(crate) role: GenerationRoleV3,
    pub(crate) semantic_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Generation0ReceiptV3 {
    pub(crate) schema: String,
    pub(crate) learning_wave_sha256: String,
    pub(crate) dataset_state_sha256: String,
    pub(crate) dataset_sha256: String,
    pub(crate) seed_request_sha256: String,
    pub(crate) seed_response_sha256: String,
    pub(crate) model_retention_receipt_sha256: String,
    pub(crate) assembly_seed: u64,
    pub(crate) candidates: Vec<GenerationCandidateV3>,
    pub(crate) role_bindings: Vec<GenerationRoleBindingV3>,
}

impl Generation0ReceiptV3 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != GENERATION_ZERO_V3_SCHEMA || self.candidates.len() != GENERATION_ZERO_SIZE
        {
            bail!("generation zero is not the promoted v3 population of 256");
        }
        validate_hashes([
            (
                "generation learning wave",
                self.learning_wave_sha256.as_str(),
            ),
            (
                "generation dataset state",
                self.dataset_state_sha256.as_str(),
            ),
            ("generation dataset", self.dataset_sha256.as_str()),
            ("generation seed request", self.seed_request_sha256.as_str()),
            (
                "generation seed response",
                self.seed_response_sha256.as_str(),
            ),
            (
                "generation model retention receipt",
                self.model_retention_receipt_sha256.as_str(),
            ),
        ])?;

        let mut semantics = BTreeSet::new();
        let mut archive = 0;
        let mut model = 0;
        let mut gpr = 0;
        let mut mutation = 0;
        let mut crossover = 0;
        let mut random = 0;
        let mut archive_priorities = BTreeSet::new();
        let mut model_buckets = BTreeMap::<ModelRetentionBucketV2, usize>::new();
        let mut gpr_families = BTreeSet::new();
        let mut mutation_seeds = BTreeSet::new();
        let mut crossover_seeds = BTreeSet::new();
        let mut random_seeds = BTreeSet::new();
        for (position, member) in self.candidates.iter().enumerate() {
            if member.position != position {
                bail!("generation zero physical positions are not contiguous");
            }
            member.candidate.validate()?;
            if !semantics.insert(member.candidate.semantic_sha256.as_str()) {
                bail!("generation zero contains a semantic duplicate");
            }
            match &member.provenance {
                CandidateProvenanceV3::Archive { priority } => {
                    archive += 1;
                    if *priority >= 32 || !archive_priorities.insert(*priority) {
                        bail!("archive priority is out of range or duplicated");
                    }
                }
                CandidateProvenanceV3::Model {
                    bucket,
                    bucket_rank,
                } => {
                    model += 1;
                    let expected = match bucket {
                        ModelRetentionBucketV2::Reward => 32,
                        ModelRetentionBucketV2::Disagreement => 16,
                        ModelRetentionBucketV2::Novelty => 8,
                        ModelRetentionBucketV2::SeededTemperature => 8,
                    };
                    if *bucket_rank >= expected {
                        bail!("model retention bucket rank exceeds its quota");
                    }
                    *model_buckets.entry(bucket.clone()).or_default() += 1;
                }
                CandidateProvenanceV3::Gpr { kernel_family } => {
                    gpr += 1;
                    if !gpr_families.insert(kernel_family.clone()) {
                        bail!("GPR kernel family is duplicated");
                    }
                }
                CandidateProvenanceV3::Mutation {
                    seed,
                    parent_semantic_sha256,
                } => {
                    mutation += 1;
                    validate_sha256("mutation parent", parent_semantic_sha256)?;
                    if !mutation_seeds.insert(*seed) {
                        bail!("mutation seed is duplicated");
                    }
                }
                CandidateProvenanceV3::Crossover {
                    seed,
                    left_parent_semantic_sha256,
                    right_parent_semantic_sha256,
                } => {
                    crossover += 1;
                    validate_hashes([
                        (
                            "crossover left parent",
                            left_parent_semantic_sha256.as_str(),
                        ),
                        (
                            "crossover right parent",
                            right_parent_semantic_sha256.as_str(),
                        ),
                    ])?;
                    if left_parent_semantic_sha256 == right_parent_semantic_sha256
                        || !crossover_seeds.insert(*seed)
                    {
                        bail!("crossover parents or seed are invalid");
                    }
                }
                CandidateProvenanceV3::Random { seed } => {
                    random += 1;
                    if !random_seeds.insert(*seed) {
                        bail!("random seed is duplicated");
                    }
                }
            }
        }
        if (archive, model, gpr, mutation, crossover, random) != (32, 64, 8, 60, 60, 32) {
            bail!("generation zero physical source topology drifted from 32/64/8/60/60/32");
        }
        for (bucket, count) in [
            (ModelRetentionBucketV2::Reward, 32),
            (ModelRetentionBucketV2::Disagreement, 16),
            (ModelRetentionBucketV2::Novelty, 8),
            (ModelRetentionBucketV2::SeededTemperature, 8),
        ] {
            if model_buckets.get(&bucket).copied() != Some(count) {
                bail!("generation zero model source quotas drifted");
            }
        }
        if gpr_families.len() != 8 {
            bail!("generation zero does not contain eight distinct GPR families");
        }

        self.validate_roles(&semantics)
    }

    fn validate_roles(&self, semantics: &BTreeSet<&str>) -> Result<()> {
        require_unique(
            "generation role bindings",
            self.role_bindings
                .iter()
                .map(|binding| binding.role.clone()),
        )?;
        for binding in &self.role_bindings {
            validate_sha256("generation role candidate", &binding.semantic_sha256)?;
            if !semantics.contains(binding.semantic_sha256.as_str()) {
                bail!("generation role refers to a candidate outside the physical population");
            }
        }
        let roles = self
            .role_bindings
            .iter()
            .map(|binding| (binding.role.clone(), binding.semantic_sha256.as_str()))
            .collect::<BTreeMap<_, _>>();
        for required in [
            GenerationRoleV3::ConfirmedCvChampion,
            GenerationRoleV3::LatestExactTrack1Frontier,
            GenerationRoleV3::CheckpointRewardTop1,
            GenerationRoleV3::LatestModelPrediction,
            GenerationRoleV3::LatestCorrectorPrediction,
            GenerationRoleV3::LatestTrack1Winner,
            GenerationRoleV3::LatestTrack2Winner,
        ] {
            if !roles.contains_key(&required) {
                bail!("generation zero is missing a mandatory role");
            }
        }
        for anchor_role in [
            GenerationRoleV3::ConfirmedCvChampion,
            GenerationRoleV3::LatestExactTrack1Frontier,
        ] {
            let semantic = roles[&anchor_role];
            if !self.candidates.iter().any(|member| {
                member.candidate.semantic_sha256 == semantic
                    && matches!(member.provenance, CandidateProvenanceV3::Archive { .. })
            }) {
                bail!("known-best anchor is not physically retained by the archive");
            }
        }
        for archive_role in [
            GenerationRoleV3::LatestCorrectorPrediction,
            GenerationRoleV3::LatestTrack1Winner,
            GenerationRoleV3::LatestTrack2Winner,
        ] {
            let semantic = roles[&archive_role];
            if !self.candidates.iter().any(|member| {
                member.candidate.semantic_sha256 == semantic
                    && matches!(member.provenance, CandidateProvenanceV3::Archive { .. })
            }) {
                bail!("mandatory archive role is not physically retained by the archive");
            }
        }
        let checkpoint_top_1 = roles[&GenerationRoleV3::CheckpointRewardTop1];
        if !self.candidates.iter().any(|member| {
            member.candidate.semantic_sha256 == checkpoint_top_1
                && matches!(
                    member.provenance,
                    CandidateProvenanceV3::Model {
                        bucket: ModelRetentionBucketV2::Reward,
                        bucket_rank: 0
                    }
                )
        }) {
            bail!("checkpoint reward top-1 is absent from the model quota");
        }
        Ok(())
    }

    pub(crate) fn receipt_sha256(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256_bytes(&canonical_json(self)?))
    }

    pub(crate) fn validate_against_state(&self, state: &DatasetLearningStateV2) -> Result<()> {
        self.validate()?;
        state.validate()?;
        if self.dataset_sha256 != state.dataset_sha256
            || self.dataset_state_sha256 != state.snapshot_sha256()?
        {
            bail!("generation zero does not bind the exact dataset-state snapshot");
        }
        let roles = self
            .role_bindings
            .iter()
            .map(|binding| (binding.role.clone(), binding.semantic_sha256.as_str()))
            .collect::<BTreeMap<_, _>>();
        let confirmed = state
            .confirmed_cv_champion
            .as_ref()
            .context("dataset has no confirmed-CV anchor; confirmation work is required")?;
        let exact = state
            .latest_exact_track_1_frontier
            .as_ref()
            .context("dataset has no exact Track-1 anchor; confirmation work is required")?;
        if roles[&GenerationRoleV3::ConfirmedCvChampion] != confirmed.semantic_sha256
            || roles[&GenerationRoleV3::LatestExactTrack1Frontier] != exact.semantic_sha256
        {
            bail!("generation zero known-best roles are stale or test-selected");
        }
        let latest_track_1 = state
            .track_1_history
            .last()
            .context("dataset has no Track-1 history")?;
        let latest_track_2 = state
            .track_2_history
            .last()
            .context("dataset has no Track-2 history")?;
        if roles[&GenerationRoleV3::LatestTrack1Winner] != latest_track_1.candidate.semantic_sha256
            || roles[&GenerationRoleV3::LatestTrack2Winner]
                != latest_track_2.candidate.semantic_sha256
        {
            bail!("generation zero latest-track archive roles are stale");
        }
        let latest_model = state
            .latest_model_prediction
            .as_ref()
            .context("dataset has no latest model prediction")?;
        let latest_corrector = state
            .latest_corrector_prediction
            .as_ref()
            .context("dataset has no latest corrector prediction")?;
        if roles[&GenerationRoleV3::LatestModelPrediction] != latest_model.candidate.semantic_sha256
            || roles[&GenerationRoleV3::LatestCorrectorPrediction]
                != latest_corrector.candidate.semantic_sha256
        {
            bail!("generation zero prediction roles are stale");
        }
        let mut required_archive = Vec::new();
        for role in [
            GenerationRoleV3::ConfirmedCvChampion,
            GenerationRoleV3::LatestExactTrack1Frontier,
            GenerationRoleV3::LatestCorrectorPrediction,
            GenerationRoleV3::LatestTrack1Winner,
            GenerationRoleV3::LatestTrack2Winner,
        ] {
            let semantic = roles[&role];
            if !required_archive.contains(&semantic) {
                required_archive.push(semantic);
            }
        }
        for (expected_priority, semantic) in required_archive.into_iter().enumerate() {
            if !self.candidates.iter().any(|member| {
                member.candidate.semantic_sha256 == semantic
                    && matches!(
                        member.provenance,
                        CandidateProvenanceV3::Archive { priority }
                            if priority == expected_priority
                    )
            }) {
                bail!("mandatory archive roles violate deterministic priority ordering");
            }
        }
        Ok(())
    }
}

/// Deterministically builds the physical generation from already validated
/// source pools. The fixed concatenation order is part of the Rust authority.
// Keep each independently validated generation pool and identity binding explicit.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn assemble_generation_zero_v3(
    learning_wave_sha256: String,
    dataset_state_sha256: String,
    dataset_sha256: String,
    seed_request_sha256: String,
    seed_response_sha256: String,
    model_retention_receipt_sha256: String,
    assembly_seed: u64,
    archive: Vec<CandidateIdentityV1>,
    model: Vec<RetainedModelProposalV2>,
    gpr: Vec<(CandidateIdentityV1, GprKernelFamilyV3)>,
    mutation: Vec<(CandidateIdentityV1, u64, String)>,
    crossover: Vec<(CandidateIdentityV1, u64, String, String)>,
    random: Vec<(CandidateIdentityV1, u64)>,
    role_bindings: Vec<GenerationRoleBindingV3>,
) -> Result<Generation0ReceiptV3> {
    if archive.len() != 32
        || model.len() != 64
        || gpr.len() != 8
        || mutation.len() != 60
        || crossover.len() != 60
        || random.len() != 32
    {
        bail!("generation source pool exhausted before its exact physical quota");
    }
    let mut candidates = Vec::with_capacity(GENERATION_ZERO_SIZE);
    for (priority, candidate) in archive.into_iter().enumerate() {
        push_generation_candidate(
            &mut candidates,
            candidate,
            CandidateProvenanceV3::Archive { priority },
        );
    }
    for retained in model {
        push_generation_candidate(
            &mut candidates,
            retained.proposal.candidate,
            CandidateProvenanceV3::Model {
                bucket: retained.bucket,
                bucket_rank: retained.bucket_rank,
            },
        );
    }
    for (candidate, kernel_family) in gpr {
        push_generation_candidate(
            &mut candidates,
            candidate,
            CandidateProvenanceV3::Gpr { kernel_family },
        );
    }
    for (candidate, seed, parent_semantic_sha256) in mutation {
        push_generation_candidate(
            &mut candidates,
            candidate,
            CandidateProvenanceV3::Mutation {
                seed,
                parent_semantic_sha256,
            },
        );
    }
    for (candidate, seed, left_parent_semantic_sha256, right_parent_semantic_sha256) in crossover {
        push_generation_candidate(
            &mut candidates,
            candidate,
            CandidateProvenanceV3::Crossover {
                seed,
                left_parent_semantic_sha256,
                right_parent_semantic_sha256,
            },
        );
    }
    for (candidate, seed) in random {
        push_generation_candidate(
            &mut candidates,
            candidate,
            CandidateProvenanceV3::Random { seed },
        );
    }
    let receipt = Generation0ReceiptV3 {
        schema: GENERATION_ZERO_V3_SCHEMA.into(),
        learning_wave_sha256,
        dataset_state_sha256,
        dataset_sha256,
        seed_request_sha256,
        seed_response_sha256,
        model_retention_receipt_sha256,
        assembly_seed,
        candidates,
        role_bindings,
    };
    receipt.validate()?;
    Ok(receipt)
}

#[cfg(test)]
fn push_generation_candidate(
    candidates: &mut Vec<GenerationCandidateV3>,
    candidate: CandidateIdentityV1,
    provenance: CandidateProvenanceV3,
) {
    candidates.push(GenerationCandidateV3 {
        position: candidates.len(),
        candidate,
        provenance,
    });
}

#[cfg(test)]
pub(crate) fn deterministic_archive_32(
    state: &DatasetLearningStateV2,
    supplemental_priority: &[CandidateIdentityV1],
) -> Result<Vec<CandidateIdentityV1>> {
    state.validate()?;
    let mut result = Vec::with_capacity(32);
    let mut seen = BTreeSet::new();
    push_unique_identity(
        &mut result,
        &mut seen,
        state
            .confirmed_cv_champion
            .as_ref()
            .context("archive requires confirmed-CV champion")?,
    )?;
    push_unique_identity(
        &mut result,
        &mut seen,
        state
            .latest_exact_track_1_frontier
            .as_ref()
            .context("archive requires exact Track-1 frontier")?,
    )?;
    push_unique_identity(
        &mut result,
        &mut seen,
        &state
            .latest_corrector_prediction
            .as_ref()
            .context("archive requires latest corrector prediction")?
            .candidate,
    )?;
    push_unique_identity(
        &mut result,
        &mut seen,
        &state
            .track_1_history
            .last()
            .context("archive requires latest Track-1 winner")?
            .candidate,
    )?;
    push_unique_identity(
        &mut result,
        &mut seen,
        &state
            .track_2_history
            .last()
            .context("archive requires latest Track-2 winner")?
            .candidate,
    )?;
    if let Some(prediction) = &state.latest_model_prediction {
        push_unique_identity(&mut result, &mut seen, &prediction.candidate)?;
    }
    let mut frontier = state.global_cv_frontier.iter().collect::<Vec<_>>();
    frontier.sort_by(|left, right| {
        right
            .robust_fitness
            .total_cmp(&left.robust_fitness)
            .then_with(|| {
                left.candidate
                    .semantic_sha256
                    .cmp(&right.candidate.semantic_sha256)
            })
    });
    for member in frontier {
        push_unique_identity(&mut result, &mut seen, &member.candidate)?;
        if result.len() == 32 {
            return Ok(result);
        }
    }
    for entry in state.track_1_history.iter().rev() {
        push_unique_identity(&mut result, &mut seen, &entry.candidate)?;
        if result.len() == 32 {
            return Ok(result);
        }
    }
    for entry in state.track_2_history.iter().rev() {
        push_unique_identity(&mut result, &mut seen, &entry.candidate)?;
        if result.len() == 32 {
            return Ok(result);
        }
    }
    for candidate in supplemental_priority {
        push_unique_identity(&mut result, &mut seen, candidate)?;
        if result.len() == 32 {
            break;
        }
    }
    if result.len() != 32 {
        bail!("deterministic archive subpool exhausted before 32 unique candidates");
    }
    Ok(result)
}

#[cfg(test)]
fn push_unique_identity(
    result: &mut Vec<CandidateIdentityV1>,
    seen: &mut BTreeSet<String>,
    candidate: &CandidateIdentityV1,
) -> Result<()> {
    candidate.validate()?;
    if result.len() < 32 && seen.insert(candidate.semantic_sha256.clone()) {
        result.push(candidate.clone());
    }
    Ok(())
}

#[cfg(test)]
pub(crate) struct OffspringPoolsV3 {
    pub(crate) mutation: Vec<(CandidateIdentityV1, u64, String)>,
    pub(crate) crossover: Vec<(CandidateIdentityV1, u64, String, String)>,
    pub(crate) random: Vec<(CandidateIdentityV1, u64)>,
}

/// Existing canonical feat-math operators are used, but Rust owns parent
/// order, seeds, semantic deduplication, quotas, and bounded failure.
#[cfg(test)]
pub(crate) fn build_bounded_offspring_pools(
    parents: &[CandidateIdentityV1],
    occupied_semantics: &BTreeSet<String>,
    assembly_seed: u64,
    max_transforms: usize,
    max_attempts_per_pool: usize,
) -> Result<OffspringPoolsV3> {
    if parents.len() < 2 || max_attempts_per_pool == 0 {
        bail!("offspring construction requires two parents and a positive attempt bound");
    }
    for parent in parents {
        parent.validate()?;
    }
    let mut seen = occupied_semantics.clone();
    for parent in parents {
        seen.insert(parent.semantic_sha256.clone());
    }

    let mut mutation = Vec::with_capacity(60);
    for attempt in 0..max_attempts_per_pool {
        if mutation.len() == 60 {
            break;
        }
        let parent = &parents[attempt % parents.len()];
        let seed = derived_seed(assembly_seed, b"mutation", attempt as u64);
        let Ok(canonical) =
            mutate_canonical_pipeline(&parent.canonical_genome, seed, max_transforms)
        else {
            continue;
        };
        let candidate = CandidateIdentityV1::from_canonical(canonical)?;
        if seen.insert(candidate.semantic_sha256.clone()) {
            mutation.push((candidate, seed, parent.semantic_sha256.clone()));
        }
    }
    if mutation.len() != 60 {
        bail!("mutation subpool exhausted its bounded refill");
    }

    let mut crossover = Vec::with_capacity(60);
    for attempt in 0..max_attempts_per_pool {
        if crossover.len() == 60 {
            break;
        }
        let left_index = attempt % parents.len();
        let mut right_index =
            (attempt + 1 + (assembly_seed as usize % (parents.len() - 1))) % parents.len();
        if right_index == left_index {
            right_index = (right_index + 1) % parents.len();
        }
        let left = &parents[left_index];
        let right = &parents[right_index];
        let seed = derived_seed(assembly_seed, b"crossover", attempt as u64);
        let Ok(canonical) = crossover_canonical_pipelines(
            &left.canonical_genome,
            &right.canonical_genome,
            seed,
            max_transforms,
        ) else {
            continue;
        };
        let candidate = CandidateIdentityV1::from_canonical(canonical)?;
        if seen.insert(candidate.semantic_sha256.clone()) {
            crossover.push((
                candidate,
                seed,
                left.semantic_sha256.clone(),
                right.semantic_sha256.clone(),
            ));
        }
    }
    if crossover.len() != 60 {
        bail!("crossover subpool exhausted its bounded refill");
    }

    let mut random = Vec::with_capacity(32);
    for attempt in 0..max_attempts_per_pool {
        if random.len() == 32 {
            break;
        }
        let seed = derived_seed(assembly_seed, b"random", attempt as u64);
        let candidate =
            CandidateIdentityV1::from_canonical(random_canonical_pipeline(seed, max_transforms))?;
        if seen.insert(candidate.semantic_sha256.clone()) {
            random.push((candidate, seed));
        }
    }
    if random.len() != 32 {
        bail!("random subpool exhausted its bounded refill");
    }
    Ok(OffspringPoolsV3 {
        mutation,
        crossover,
        random,
    })
}

#[cfg(test)]
fn derived_seed(base: u64, lane: &[u8], attempt: u64) -> u64 {
    let mut digest = Sha256::new();
    digest.update(b"jope-dime/generation-zero-offspring/v3\0");
    digest.update(base.to_le_bytes());
    digest.update(lane);
    digest.update(attempt.to_le_bytes());
    let digest = digest.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix"))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProofSlateLaneV3 {
    Selectable,
    CvOnlyCalibration,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ProofSlateCandidateV3 {
    pub(crate) candidate: CandidateIdentityV1,
    pub(crate) lane: ProofSlateLaneV3,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct AttemptAdmissionManifestV3 {
    pub(crate) schema: String,
    pub(crate) attempt_id: String,
    pub(crate) learning_wave_sha256: String,
    pub(crate) dataset_state_sha256: String,
    pub(crate) scheduler_draw_sha256: String,
    pub(crate) generation_zero_sha256: String,
    pub(crate) checkpoint_sha256: String,
    pub(crate) corrector_sha256: String,
    pub(crate) dataset_sha256: String,
    pub(crate) folds_sha256: String,
    pub(crate) proof_slate_seed: u64,
    pub(crate) proof_slate: Vec<ProofSlateCandidateV3>,
}

impl AttemptAdmissionManifestV3 {
    pub(crate) fn validate(
        &self,
        generation: &Generation0ReceiptV3,
        wave: &LearningWaveManifestV1,
        state: &DatasetLearningStateV2,
    ) -> Result<()> {
        if self.schema != ADMISSION_V3_SCHEMA || !valid_attempt_id(&self.attempt_id) {
            bail!("attempt admission is not a valid v3 manifest");
        }
        validate_hashes([
            (
                "admission learning wave",
                self.learning_wave_sha256.as_str(),
            ),
            (
                "admission dataset state",
                self.dataset_state_sha256.as_str(),
            ),
            (
                "admission scheduler draw",
                self.scheduler_draw_sha256.as_str(),
            ),
            (
                "admission generation zero",
                self.generation_zero_sha256.as_str(),
            ),
            ("admission checkpoint", self.checkpoint_sha256.as_str()),
            ("admission corrector", self.corrector_sha256.as_str()),
            ("admission dataset", self.dataset_sha256.as_str()),
            ("admission folds", self.folds_sha256.as_str()),
        ])?;
        generation.validate_against_state(state)?;
        let wave_sha256 = wave.manifest_sha256()?;
        if self.learning_wave_sha256 != wave_sha256
            || self.learning_wave_sha256 != generation.learning_wave_sha256
            || self.dataset_state_sha256 != generation.dataset_state_sha256
            || self.dataset_sha256 != generation.dataset_sha256
            || self.generation_zero_sha256 != generation.receipt_sha256()?
            || self.checkpoint_sha256 != wave.policy_checkpoint_sha256
            || self.corrector_sha256 != wave.corrector_sha256
        {
            bail!("attempt admission does not bind one frozen wave/dataset/generation");
        }
        if self.proof_slate.len() != 12 {
            bail!("attempt proof slate is not precommitted as 10+2");
        }
        require_unique(
            "attempt proof slate",
            self.proof_slate
                .iter()
                .map(|entry| entry.candidate.semantic_sha256.as_str()),
        )?;
        let selectable = self
            .proof_slate
            .iter()
            .filter(|entry| entry.lane == ProofSlateLaneV3::Selectable)
            .count();
        let calibration = self
            .proof_slate
            .iter()
            .filter(|entry| entry.lane == ProofSlateLaneV3::CvOnlyCalibration)
            .count();
        if (selectable, calibration) != (10, 2) {
            bail!("attempt proof slate lanes drifted from 10 selectable + 2 CV-only");
        }
        for entry in &self.proof_slate {
            entry.candidate.validate()?;
        }
        Ok(())
    }
}

fn valid_attempt_id(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(*byte, b'-' | b'_' | b'.'))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ReplayLabelRevisionV2 {
    pub(crate) group_sha256: String,
    pub(crate) dataset_sha256: String,
    pub(crate) revision: u64,
    pub(crate) label_revision_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ReplayStepV2 {
    pub(crate) step: usize,
    pub(crate) datasets: Vec<String>,
    pub(crate) revisions: Vec<ReplayLabelRevisionV2>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ReplayManifestV2 {
    pub(crate) schema: String,
    pub(crate) label_cutoff: u64,
    pub(crate) reservoir_manifest_sha256: String,
    pub(crate) reservoir_group_count: usize,
    pub(crate) reservoir_groups: Vec<String>,
    pub(crate) scheduled_datasets: Vec<String>,
    pub(crate) steps: Vec<ReplayStepV2>,
}

impl ReplayManifestV2 {
    pub(crate) fn validate(
        &self,
        historical_unordered_dataset_sets: &BTreeSet<Vec<String>>,
    ) -> Result<()> {
        if self.schema != REPLAY_V2_SCHEMA
            || self.reservoir_group_count != 200
            || self.reservoir_groups.len() != 200
            || self.steps.len() != REPLAY_STEPS
            || self.scheduled_datasets.len() != DATASETS_PER_MICRO_BATCH
        {
            bail!("replay v2 manifest does not bind 200 groups, eight datasets, and 100 steps");
        }
        validate_sha256("replay reservoir", &self.reservoir_manifest_sha256)?;
        for group in &self.reservoir_groups {
            validate_sha256("replay reservoir group", group)?;
        }
        require_unique(
            "replay reservoir groups",
            self.reservoir_groups.iter().map(String::as_str),
        )?;
        if self.reservoir_manifest_sha256 != sha256_bytes(&canonical_json(&self.reservoir_groups)?)
        {
            bail!("replay reservoir manifest does not bind its frozen 200 groups");
        }
        let reservoir_groups = self
            .reservoir_groups
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        validate_dataset_set("scheduled micro-batch", &self.scheduled_datasets)?;
        let mut local_sets = BTreeSet::new();
        for (index, step) in self.steps.iter().enumerate() {
            if step.step != index
                || step.datasets.len() != DATASETS_PER_MICRO_BATCH
                || step.revisions.len() != DATASETS_PER_MICRO_BATCH
            {
                bail!("replay step is not an exact unordered eight-dataset draw");
            }
            validate_dataset_set("replay step", &step.datasets)?;
            let mut normalized = step.datasets.clone();
            normalized.sort();
            if historical_unordered_dataset_sets.contains(&normalized)
                || !local_sets.insert(normalized)
            {
                bail!("replay contains a previously drawn unordered eight-dataset set");
            }
            let revision_datasets = step
                .revisions
                .iter()
                .map(|revision| revision.dataset_sha256.as_str())
                .collect::<BTreeSet<_>>();
            let step_datasets = step
                .datasets
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            if revision_datasets != step_datasets {
                bail!("replay revisions do not bind exactly the step datasets");
            }
            for revision in &step.revisions {
                validate_hashes([
                    ("replay group", revision.group_sha256.as_str()),
                    ("replay dataset", revision.dataset_sha256.as_str()),
                    (
                        "replay label revision",
                        revision.label_revision_sha256.as_str(),
                    ),
                ])?;
                if revision.revision == 0 {
                    bail!("replay references label revision zero");
                }
                if !reservoir_groups.contains(revision.group_sha256.as_str()) {
                    bail!("replay step references a group outside the frozen reservoir");
                }
            }
        }
        Ok(())
    }

    pub(crate) fn manifest_sha256(&self) -> Result<String> {
        self.validate(&BTreeSet::new())?;
        Ok(sha256_bytes(&canonical_json(self)?))
    }
}

fn validate_dataset_set(label: &str, datasets: &[String]) -> Result<()> {
    for dataset in datasets {
        validate_sha256(label, dataset)?;
    }
    require_unique(label, datasets.iter().map(String::as_str))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SchedulerLaneV4 {
    Challenge,
    Uniform,
    StaleUnderexposed,
    GlobalCvConfirmation,
}

impl SchedulerLaneV4 {
    fn all() -> [Self; 4] {
        [
            Self::Challenge,
            Self::Uniform,
            Self::StaleUnderexposed,
            Self::GlobalCvConfirmation,
        ]
    }

    fn share_basis_points(self) -> i128 {
        match self {
            Self::Challenge => 7_000,
            Self::Uniform => 1_500,
            Self::StaleUnderexposed => 1_000,
            Self::GlobalCvConfirmation => 500,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SchedulerStateV4 {
    pub(crate) schema: String,
    pub(crate) total_measured_cpu_micros: u64,
    pub(crate) measured_cpu_micros: BTreeMap<SchedulerLaneV4, u64>,
    pub(crate) lane_debt_basis_micros: BTreeMap<SchedulerLaneV4, i128>,
    pub(crate) draws: u64,
}

impl Default for SchedulerStateV4 {
    fn default() -> Self {
        Self {
            schema: "LearningSchedulerState.v4".into(),
            total_measured_cpu_micros: 0,
            measured_cpu_micros: SchedulerLaneV4::all()
                .into_iter()
                .map(|lane| (lane, 0))
                .collect(),
            lane_debt_basis_micros: SchedulerLaneV4::all()
                .into_iter()
                .map(|lane| (lane, 0))
                .collect(),
            draws: 0,
        }
    }
}

impl SchedulerStateV4 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != "LearningSchedulerState.v4"
            || self.measured_cpu_micros.len() != 4
            || self.lane_debt_basis_micros.len() != 4
            || SchedulerLaneV4::all().iter().any(|lane| {
                !self.measured_cpu_micros.contains_key(lane)
                    || !self.lane_debt_basis_micros.contains_key(lane)
            })
            || self.measured_cpu_micros.values().sum::<u64>() != self.total_measured_cpu_micros
        {
            bail!("scheduler v4 state is incomplete or does not reconcile measured CPU");
        }
        for lane in SchedulerLaneV4::all() {
            let expected = lane.share_basis_points() * self.total_measured_cpu_micros as i128
                - 10_000 * self.measured_cpu_micros[&lane] as i128;
            if self.lane_debt_basis_micros[&lane] != expected {
                bail!("scheduler lane debt does not reconstruct from measured CPU");
            }
        }
        Ok(())
    }

    /// Highest positive debt wins. Deterministic enum order breaks ties.
    #[cfg(test)]
    pub(crate) fn next_lane(&self) -> Result<SchedulerLaneV4> {
        self.validate()?;
        Ok(SchedulerLaneV4::all()
            .into_iter()
            .max_by(|left, right| {
                self.lane_debt_basis_micros[left]
                    .cmp(&self.lane_debt_basis_micros[right])
                    .then_with(|| right.cmp(left))
            })
            .expect("four scheduler lanes"))
    }

    #[cfg(test)]
    pub(crate) fn record_measured_cpu(
        &mut self,
        lane: SchedulerLaneV4,
        cpu_micros: u64,
    ) -> Result<()> {
        if cpu_micros == 0 {
            bail!("scheduler measured-CPU charge must be positive");
        }
        self.total_measured_cpu_micros = self
            .total_measured_cpu_micros
            .checked_add(cpu_micros)
            .context("scheduler total CPU overflow")?;
        *self
            .measured_cpu_micros
            .get_mut(&lane)
            .context("scheduler lane missing")? = self.measured_cpu_micros[&lane]
            .checked_add(cpu_micros)
            .context("scheduler lane CPU overflow")?;
        self.draws = self
            .draws
            .checked_add(1)
            .context("scheduler draw overflow")?;
        for current in SchedulerLaneV4::all() {
            self.lane_debt_basis_micros.insert(
                current,
                current.share_basis_points() * self.total_measured_cpu_micros as i128
                    - 10_000 * self.measured_cpu_micros[&current] as i128,
            );
        }
        self.validate()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SchedulerDrawV4 {
    pub(crate) draw_index: u64,
    pub(crate) lane: SchedulerLaneV4,
    pub(crate) dataset_sha256: String,
    pub(crate) state_snapshot_sha256: String,
    pub(crate) draw_seed: u64,
}

impl SchedulerDrawV4 {
    fn validate(&self) -> Result<()> {
        validate_hashes([
            ("scheduler dataset", self.dataset_sha256.as_str()),
            (
                "scheduler state snapshot",
                self.state_snapshot_sha256.as_str(),
            ),
        ])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LearningMicroBatchManifestV1 {
    pub(crate) schema: String,
    pub(crate) batch_index: u64,
    pub(crate) serving_wave_sha256: String,
    pub(crate) base_training_checkpoint_sha256: String,
    pub(crate) candidate_training_checkpoint_sha256: String,
    pub(crate) candidate_resume_checkpoint_sha256: String,
    pub(crate) replay_manifest_sha256: String,
    pub(crate) scheduler_draws: Vec<SchedulerDrawV4>,
    pub(crate) update_steps: usize,
    pub(crate) completed: bool,
}

impl LearningMicroBatchManifestV1 {
    pub(crate) fn validate(&self, replay: &ReplayManifestV2) -> Result<()> {
        if self.schema != MICRO_BATCH_SCHEMA
            || self.scheduler_draws.len() != DATASETS_PER_MICRO_BATCH
            || self.update_steps != REPLAY_STEPS
            || !self.completed
        {
            bail!("micro-batch is not eight datasets plus one complete 100-step update");
        }
        validate_hashes([
            (
                "micro-batch serving wave",
                self.serving_wave_sha256.as_str(),
            ),
            (
                "micro-batch base training checkpoint",
                self.base_training_checkpoint_sha256.as_str(),
            ),
            (
                "micro-batch candidate checkpoint",
                self.candidate_training_checkpoint_sha256.as_str(),
            ),
            (
                "micro-batch candidate resume checkpoint",
                self.candidate_resume_checkpoint_sha256.as_str(),
            ),
            (
                "micro-batch replay manifest",
                self.replay_manifest_sha256.as_str(),
            ),
        ])?;
        for draw in &self.scheduler_draws {
            draw.validate()?;
        }
        require_unique(
            "micro-batch scheduler datasets",
            self.scheduler_draws
                .iter()
                .map(|draw| draw.dataset_sha256.as_str()),
        )?;
        if self.replay_manifest_sha256 != replay.manifest_sha256()?
            || self
                .scheduler_draws
                .iter()
                .map(|draw| draw.dataset_sha256.as_str())
                .collect::<BTreeSet<_>>()
                != replay
                    .scheduled_datasets
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
        {
            bail!("micro-batch scheduler draws and immutable replay disagree");
        }
        Ok(())
    }

    pub(crate) fn manifest_sha256(&self, replay: &ReplayManifestV2) -> Result<String> {
        self.validate(replay)?;
        Ok(sha256_bytes(&canonical_json(self)?))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TrainingHeadManifestV1 {
    pub(crate) schema: String,
    pub(crate) checkpoint_sha256: String,
    pub(crate) resume_checkpoint_sha256: String,
    pub(crate) corrector_sha256: String,
    pub(crate) micro_batch_manifest_sha256: String,
    pub(crate) replay_manifest_sha256: String,
    pub(crate) update_steps: usize,
    pub(crate) label_cutoff: u64,
    pub(crate) published_at_unix_seconds: u64,
}

impl TrainingHeadManifestV1 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != TRAINING_HEAD_SCHEMA || self.update_steps != REPLAY_STEPS {
            bail!("training head does not represent one valid 100-step update");
        }
        validate_hashes([
            ("training head checkpoint", self.checkpoint_sha256.as_str()),
            (
                "training head resume checkpoint",
                self.resume_checkpoint_sha256.as_str(),
            ),
            ("training head corrector", self.corrector_sha256.as_str()),
            (
                "training head micro-batch",
                self.micro_batch_manifest_sha256.as_str(),
            ),
            (
                "training head replay manifest",
                self.replay_manifest_sha256.as_str(),
            ),
        ])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub(crate) struct RolloutSwitchesV1 {
    #[serde(default)]
    pub(crate) dataset_state_v2: bool,
    #[serde(default)]
    pub(crate) scheduler_v4: bool,
    #[serde(default)]
    pub(crate) replay_v2: bool,
    #[serde(default)]
    pub(crate) generation_zero_v3: bool,
    #[serde(default)]
    pub(crate) wave_authority: bool,
}

impl RolloutSwitchesV1 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.wave_authority
            && !(self.dataset_state_v2
                && self.scheduler_v4
                && self.replay_v2
                && self.generation_zero_v3)
        {
            bail!("wave authority requires every dependent rollout switch");
        }
        Ok(())
    }

    fn require_wave_authority(&self) -> Result<()> {
        self.validate()?;
        if !self.wave_authority {
            bail!("serving-wave publication is disabled");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct RatchetArtifactPointerV1 {
    pub(crate) schema: String,
    pub(crate) artifact_sha256: String,
    pub(crate) updated_at_unix_seconds: u64,
    pub(crate) serving_authority: bool,
}

impl RatchetArtifactPointerV1 {
    fn validate(&self, serving_authority: bool) -> Result<()> {
        if self.schema != POINTER_SCHEMA || self.serving_authority != serving_authority {
            bail!("ratchet pointer has the wrong authority class");
        }
        validate_sha256("ratchet pointer artifact", &self.artifact_sha256)
    }
}

pub(crate) struct RatchetStore {
    root: PathBuf,
}

impl RatchetStore {
    pub(crate) fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub(crate) fn publish_training_head(
        &self,
        head: &TrainingHeadManifestV1,
        batch: &LearningMicroBatchManifestV1,
        replay: &ReplayManifestV2,
    ) -> Result<String> {
        let _writer = WriterGuard::acquire(&self.root)?;
        head.validate()?;
        batch.validate(replay)?;
        if head.micro_batch_manifest_sha256 != batch.manifest_sha256(replay)?
            || head.replay_manifest_sha256 != replay.manifest_sha256()?
            || head.checkpoint_sha256 != batch.candidate_training_checkpoint_sha256
            || head.resume_checkpoint_sha256 != batch.candidate_resume_checkpoint_sha256
            || head.label_cutoff != replay.label_cutoff
        {
            bail!("training head does not reconstruct from its completed micro-batch and replay");
        }
        let bytes = canonical_json(head)?;
        let digest = sha256_bytes(&bytes);
        let head_object = self
            .root
            .join("training-head/objects")
            .join(format!("{digest}.json"));
        if head_object.exists() {
            if fs::read(&head_object)? != bytes {
                bail!("content-addressed training head object is corrupt");
            }
            let pointer = RatchetArtifactPointerV1 {
                schema: POINTER_SCHEMA.into(),
                artifact_sha256: digest.clone(),
                updated_at_unix_seconds: head.published_at_unix_seconds,
                serving_authority: false,
            };
            atomic_pointer_write(&self.root.join("training-head.json"), &pointer)?;
            return Ok(digest);
        }
        let historical_sets = self.historical_replay_sets()?;
        replay.validate(&historical_sets)?;
        if let Some((_, current)) = self.load_training_head()? {
            if batch.base_training_checkpoint_sha256 != current.checkpoint_sha256
                || head.label_cutoff < current.label_cutoff
                || head.published_at_unix_seconds <= current.published_at_unix_seconds
            {
                bail!("training head does not continue the current non-serving training branch");
            }
        }
        let replay_bytes = canonical_json(replay)?;
        let replay_sha256 = sha256_bytes(&replay_bytes);
        immutable_write(
            &self
                .root
                .join("replay/objects")
                .join(format!("{replay_sha256}.json")),
            &replay_bytes,
        )?;
        let batch_bytes = canonical_json(batch)?;
        let batch_sha256 = sha256_bytes(&batch_bytes);
        immutable_write(
            &self
                .root
                .join("micro-batches/objects")
                .join(format!("{batch_sha256}.json")),
            &batch_bytes,
        )?;
        immutable_write(&head_object, &bytes)?;
        let pointer = RatchetArtifactPointerV1 {
            schema: POINTER_SCHEMA.into(),
            artifact_sha256: digest.clone(),
            updated_at_unix_seconds: head.published_at_unix_seconds,
            serving_authority: false,
        };
        atomic_pointer_write(&self.root.join("training-head.json"), &pointer)?;
        Ok(digest)
    }

    pub(crate) fn load_training_head(&self) -> Result<Option<(String, TrainingHeadManifestV1)>> {
        let pointer_path = self.root.join("training-head.json");
        if !pointer_path.exists() {
            return Ok(None);
        }
        let pointer: RatchetArtifactPointerV1 = serde_json::from_slice(&fs::read(&pointer_path)?)?;
        pointer.validate(false)?;
        let bytes = fs::read(
            self.root
                .join("training-head/objects")
                .join(format!("{}.json", pointer.artifact_sha256)),
        )
        .context("training-head object is missing")?;
        if sha256_bytes(&bytes) != pointer.artifact_sha256 {
            bail!("training-head pointer does not match its immutable object");
        }
        let head: TrainingHeadManifestV1 = serde_json::from_slice(&bytes)?;
        head.validate()?;
        if canonical_json(&head)? != bytes {
            bail!("training-head object is not canonical");
        }
        Ok(Some((pointer.artifact_sha256, head)))
    }

    fn historical_replay_sets(&self) -> Result<BTreeSet<Vec<String>>> {
        let root = self.root.join("training-head/objects");
        if !root.exists() {
            return Ok(BTreeSet::new());
        }
        let mut paths = fs::read_dir(root)?
            .map(|entry| Ok(entry?.path()))
            .collect::<Result<Vec<_>>>()?;
        paths.sort();
        let mut sets = BTreeSet::new();
        for path in paths {
            let bytes = fs::read(&path)?;
            let expected = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .context("training-head object filename is invalid")?;
            validate_sha256("training-head object filename", expected)?;
            if sha256_bytes(&bytes) != expected {
                bail!("training-head object filename does not match its content");
            }
            let head: TrainingHeadManifestV1 = serde_json::from_slice(&bytes)?;
            head.validate()?;
            if canonical_json(&head)? != bytes {
                bail!("training-head object is not canonical");
            }
            let replay_bytes = fs::read(
                self.root
                    .join("replay/objects")
                    .join(format!("{}.json", head.replay_manifest_sha256)),
            )
            .context("accepted training head is missing its replay object")?;
            if sha256_bytes(&replay_bytes) != head.replay_manifest_sha256 {
                bail!("accepted training head replay object is corrupt");
            }
            let replay: ReplayManifestV2 = serde_json::from_slice(&replay_bytes)?;
            replay.validate(&sets)?;
            for step in replay.steps {
                let mut normalized = step.datasets;
                normalized.sort();
                sets.insert(normalized);
            }
        }
        Ok(sets)
    }

    fn has_eligible_training_head(&self, wave: &LearningWaveManifestV1) -> Result<bool> {
        let root = self.root.join("training-head/objects");
        if !root.exists() {
            return Ok(false);
        }
        for entry in fs::read_dir(root)? {
            let path = entry?.path();
            let bytes = fs::read(&path)?;
            let expected = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .context("training-head object filename is invalid")?;
            validate_sha256("training-head object filename", expected)?;
            if sha256_bytes(&bytes) != expected {
                bail!("training-head object filename does not match its content");
            }
            let head: TrainingHeadManifestV1 = serde_json::from_slice(&bytes)?;
            head.validate()?;
            if canonical_json(&head)? != bytes {
                bail!("training-head object is not canonical");
            }
            if head.checkpoint_sha256 == wave.policy_checkpoint_sha256
                && head.resume_checkpoint_sha256 == wave.resume_checkpoint_sha256
                && head.corrector_sha256 == wave.corrector_sha256
                && head.label_cutoff == wave.training_label_cutoff
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn commit_dataset_state(
        &self,
        state: &DatasetLearningStateV2,
        committed_at_unix_seconds: u64,
    ) -> Result<String> {
        let _writer = WriterGuard::acquire(&self.root)?;
        state.validate()?;
        if let Some((_, current)) = self.load_dataset_state(&state.dataset_sha256)? {
            current.validate_successor(state)?;
        }
        let bytes = canonical_json(state)?;
        let digest = sha256_bytes(&bytes);
        let dataset_root = self.root.join("datasets").join(&state.dataset_sha256);
        immutable_write(
            &dataset_root.join("states").join(format!("{digest}.json")),
            &bytes,
        )?;
        let pointer = RatchetArtifactPointerV1 {
            schema: POINTER_SCHEMA.into(),
            artifact_sha256: digest.clone(),
            updated_at_unix_seconds: committed_at_unix_seconds,
            serving_authority: false,
        };
        atomic_pointer_write(&dataset_root.join("current.json"), &pointer)?;
        Ok(digest)
    }

    pub(crate) fn load_dataset_state(
        &self,
        dataset_sha256: &str,
    ) -> Result<Option<(String, DatasetLearningStateV2)>> {
        validate_sha256("dataset state lookup", dataset_sha256)?;
        let dataset_root = self.root.join("datasets").join(dataset_sha256);
        let pointer_path = dataset_root.join("current.json");
        if !pointer_path.exists() {
            return Ok(None);
        }
        let pointer: RatchetArtifactPointerV1 = serde_json::from_slice(&fs::read(&pointer_path)?)?;
        pointer.validate(false)?;
        let bytes = fs::read(
            dataset_root
                .join("states")
                .join(format!("{}.json", pointer.artifact_sha256)),
        )
        .context("dataset state object is missing")?;
        if sha256_bytes(&bytes) != pointer.artifact_sha256 {
            bail!("dataset state pointer digest does not match its immutable object");
        }
        let state: DatasetLearningStateV2 = serde_json::from_slice(&bytes)?;
        state.validate()?;
        if state.dataset_sha256 != dataset_sha256
            || state.snapshot_sha256()? != pointer.artifact_sha256
        {
            bail!("dataset state object does not bind the requested dataset");
        }
        Ok(Some((pointer.artifact_sha256, state)))
    }

    pub(crate) fn publish_serving_wave(
        &self,
        wave: &LearningWaveManifestV1,
        switches: &RolloutSwitchesV1,
        sealed_holdout_hashes: &BTreeSet<String>,
    ) -> Result<String> {
        let _writer = WriterGuard::acquire(&self.root)?;
        switches.require_wave_authority()?;
        let bytes = canonical_json(wave)?;
        let digest = sha256_bytes(&bytes);
        let current = self.load_serving_wave()?;
        if current
            .as_ref()
            .is_some_and(|(current_digest, current_manifest)| {
                current_digest == &digest && current_manifest == wave
            })
        {
            return Ok(digest);
        }
        let parent = current
            .as_ref()
            .map(|(digest, manifest)| (digest.as_str(), manifest));
        wave.validate(parent, sealed_holdout_hashes)?;
        if parent.is_some() && !self.has_eligible_training_head(wave)? {
            bail!("recurring serving wave is not backed by an eligible training-head checkpoint");
        }
        immutable_write(
            &self
                .root
                .join("waves")
                .join("objects")
                .join(format!("{digest}.json")),
            &bytes,
        )?;
        let pointer = RatchetArtifactPointerV1 {
            schema: POINTER_SCHEMA.into(),
            artifact_sha256: digest.clone(),
            updated_at_unix_seconds: wave.accepted_at_unix_seconds,
            serving_authority: true,
        };
        // `waves/current.json` is deliberately written before the compatibility
        // mirror: a crash can leave a stale mirror, never a false authority.
        atomic_pointer_write(&self.root.join("waves/current.json"), &pointer)?;
        let mirror = RatchetArtifactPointerV1 {
            serving_authority: false,
            ..pointer
        };
        atomic_pointer_write(&self.root.join("dl/current.json"), &mirror)?;
        Ok(digest)
    }

    pub(crate) fn admit_attempt(
        &self,
        generation: &Generation0ReceiptV3,
        admission: &AttemptAdmissionManifestV3,
    ) -> Result<String> {
        let _writer = WriterGuard::acquire(&self.root)?;
        if !valid_attempt_id(&admission.attempt_id) {
            bail!("attempt identity is unsafe");
        }
        let admission_bytes = canonical_json(admission)?;
        let admission_sha256 = sha256_bytes(&admission_bytes);
        let attempt_root = self.root.join("attempts").join(&admission.attempt_id);
        let committed_admission = attempt_root.join("admission.v3.json");
        if committed_admission.exists() {
            if fs::read(&committed_admission)? == admission_bytes {
                return Ok(admission_sha256);
            }
            bail!("attempt identity is already bound to different immutable admission bytes");
        }
        if self.root.join("admissions-stopped.json").exists() {
            bail!("new attempt admissions are stopped by an authority fault");
        }
        let (wave_sha256, wave) = self
            .load_serving_wave()?
            .context("no accepted serving wave exists")?;
        let (state_sha256, state) = self
            .load_dataset_state(&admission.dataset_sha256)?
            .context("dataset has no authoritative v2 learning state")?;
        if generation.learning_wave_sha256 != wave_sha256
            || generation.dataset_state_sha256 != state_sha256
        {
            bail!("new attempt did not use the latest serving wave and exact dataset state");
        }
        admission.validate(generation, &wave, &state)?;

        let generation_bytes = canonical_json(generation)?;
        let generation_sha256 = sha256_bytes(&generation_bytes);
        immutable_write(
            &attempt_root
                .join("generation-zero")
                .join(format!("{generation_sha256}.json")),
            &generation_bytes,
        )?;
        // Admission is the final commit marker. A crash before this write
        // leaves no admitted attempt; retrying reconstructs identical bytes.
        immutable_write(&committed_admission, &admission_bytes)?;
        Ok(admission_sha256)
    }

    pub(crate) fn load_serving_wave(&self) -> Result<Option<(String, LearningWaveManifestV1)>> {
        let pointer_path = self.root.join("waves/current.json");
        if !pointer_path.exists() {
            return Ok(None);
        }
        let pointer: RatchetArtifactPointerV1 = serde_json::from_slice(&fs::read(&pointer_path)?)?;
        pointer.validate(true)?;
        let object_path = self
            .root
            .join("waves/objects")
            .join(format!("{}.json", pointer.artifact_sha256));
        let bytes = fs::read(&object_path).context("serving wave object is missing")?;
        if sha256_bytes(&bytes) != pointer.artifact_sha256 {
            bail!("serving wave pointer digest does not match its immutable object");
        }
        let manifest: LearningWaveManifestV1 = serde_json::from_slice(&bytes)?;
        if manifest.manifest_sha256()? != pointer.artifact_sha256 {
            bail!("serving wave object is not canonical");
        }
        Ok(Some((pointer.artifact_sha256, manifest)))
    }

    /// Stop future admissions without modifying any frozen attempt manifest.
    #[cfg(test)]
    pub(crate) fn stop_admissions(&self, fault_receipt_sha256: &str, at: u64) -> Result<()> {
        let _writer = WriterGuard::acquire(&self.root)?;
        validate_sha256("authority fault receipt", fault_receipt_sha256)?;
        let value = serde_json::json!({
            "schema": "RatchetAdmissionStop.v1",
            "fault_receipt_sha256": fault_receipt_sha256,
            "stopped_at_unix_seconds": at,
        });
        immutable_write(
            &self.root.join("admissions-stopped.json"),
            &canonical_json(&value)?,
        )
    }

    /// Fault rollback only moves the authoritative pointer to an already
    /// accepted immutable ancestor. It never republishes or edits that wave.
    #[cfg(test)]
    pub(crate) fn restore_previous_wave(&self, previous_wave_sha256: &str, at: u64) -> Result<()> {
        let _writer = WriterGuard::acquire(&self.root)?;
        validate_sha256("previous accepted wave", previous_wave_sha256)?;
        let object_path = self
            .root
            .join("waves/objects")
            .join(format!("{previous_wave_sha256}.json"));
        let bytes = fs::read(&object_path).context("previous accepted wave object is missing")?;
        if sha256_bytes(&bytes) != previous_wave_sha256 {
            bail!("previous accepted wave object is corrupt");
        }
        let manifest: LearningWaveManifestV1 = serde_json::from_slice(&bytes)?;
        if manifest.manifest_sha256()? != previous_wave_sha256 {
            bail!("previous accepted wave object is noncanonical");
        }
        let pointer = RatchetArtifactPointerV1 {
            schema: POINTER_SCHEMA.into(),
            artifact_sha256: previous_wave_sha256.into(),
            updated_at_unix_seconds: at,
            serving_authority: true,
        };
        atomic_pointer_write(&self.root.join("waves/current.json"), &pointer)?;
        let mirror = RatchetArtifactPointerV1 {
            serving_authority: false,
            ..pointer
        };
        atomic_pointer_write(&self.root.join("dl/current.json"), &mirror)
    }
}

struct WriterGuard {
    path: PathBuf,
}

impl WriterGuard {
    fn acquire(root: &Path) -> Result<Self> {
        fs::create_dir_all(root)?;
        let path = root.join(".ratchet-writer.lock");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .context("another Rust ratchet writer is active or a fault lock requires recovery")?;
        writeln!(file, "{}", std::process::id())?;
        file.sync_all()?;
        Ok(Self { path })
    }
}

impl Drop for WriterGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(bytes)?;
            file.sync_all()?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if fs::read(path)? == bytes {
                Ok(())
            } else {
                bail!("immutable artifact already exists with different bytes");
            }
        }
        Err(error) => Err(error.into()),
    }
}

fn atomic_pointer_write(path: &Path, pointer: &RatchetArtifactPointerV1) -> Result<()> {
    atomic_bytes_write(path, &canonical_json(pointer)?)
}

fn atomic_bytes_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("atomic pointer path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .context("atomic pointer filename is not UTF-8")?,
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .with_context(|| format!("create atomic temporary {}", temporary.display()))?;
    let result = (|| -> Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

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

pub(crate) fn validate_artifact_file(path: &Path) -> Result<()> {
    let bytes = fs::read(path)?;
    let value: Value = serde_json::from_slice(&bytes)?;
    let schema = value
        .get("schema")
        .and_then(Value::as_str)
        .context("ratchet artifact has no schema")?;
    match schema {
        DATASET_STATE_SCHEMA => serde_json::from_value::<DatasetLearningStateV2>(value)?.validate(),
        WAVE_SCHEMA => serde_json::from_value::<LearningWaveManifestV1>(value)?
            .validate(None, &BTreeSet::new()),
        SEED_REQUEST_V2_SCHEMA => {
            serde_json::from_value::<SeedInferenceRequestV2>(value)?.validate()
        }
        SEED_RESPONSE_V2_SCHEMA => match read_seed_inference(&bytes)? {
            ReadableSeedInference::V2(response) => {
                if response.proposals.len() != MODEL_PROPOSAL_COUNT {
                    bail!("v2 seed response does not contain 96 proposals");
                }
                for proposal in &response.proposals {
                    proposal.validate()?;
                }
                require_unique(
                    "v2 seed response",
                    response
                        .proposals
                        .iter()
                        .map(|proposal| proposal.candidate.semantic_sha256.as_str()),
                )
            }
            ReadableSeedInference::V1(_) => unreachable!(),
        },
        GENERATION_ZERO_V3_SCHEMA => {
            serde_json::from_value::<Generation0ReceiptV3>(value)?.validate()
        }
        REPLAY_V2_SCHEMA => {
            serde_json::from_value::<ReplayManifestV2>(value)?.validate(&BTreeSet::new())
        }
        TRAINING_HEAD_SCHEMA => serde_json::from_value::<TrainingHeadManifestV1>(value)?.validate(),
        "LearningSchedulerState.v4" => {
            serde_json::from_value::<SchedulerStateV4>(value)?.validate()
        }
        "RatchetLiveTelemetry.v2" => {
            serde_json::from_value::<RatchetLiveTelemetryV2>(value)?.validate()
        }
        "BroadShadowPilotManifest.v1" => {
            serde_json::from_value::<BroadShadowPilotManifestV1>(value)?.validate()
        }
        "FrozenRealDatasetSplits.v1" => {
            serde_json::from_value::<FrozenRealSplitsV1>(value)?.validate()
        }
        "ProductionModelExperimentManifest.v1" => {
            serde_json::from_value::<TrainingExperimentManifestV1>(value)?.validate()
        }
        schema if schema.starts_with("SeedInference") => {
            let _ = read_seed_inference(&bytes)?;
            Ok(())
        }
        schema if schema.starts_with("Generation0Receipt") => {
            let _ = read_generation_zero(&bytes)?;
            Ok(())
        }
        schema if schema.starts_with("AttemptAdmissionManifest") => {
            let _ = read_admission(&bytes)?;
            Ok(())
        }
        schema if schema.starts_with("RatchetLiveTelemetry") => {
            let _ = read_telemetry(&bytes)?;
            Ok(())
        }
        _ => bail!("unsupported ratchet artifact schema {schema}"),
    }
}

pub(crate) fn retain_model_seeds_from_files(
    request_path: &Path,
    response_path: &Path,
    excluded_semantics_path: &Path,
    output_path: &Path,
) -> Result<()> {
    let request: SeedInferenceRequestV2 = serde_json::from_slice(&fs::read(request_path)?)?;
    let response: SeedInferenceResponseV2 = serde_json::from_slice(&fs::read(response_path)?)?;
    let excluded: BTreeSet<String> = serde_json::from_slice(&fs::read(excluded_semantics_path)?)?;
    let receipt = ModelRetentionReceiptV2::build(&request, &response, &excluded)?;
    immutable_write(output_path, &canonical_json(&receipt)?)
}

pub(crate) fn commit_dataset_state_from_file(
    root: &Path,
    state_path: &Path,
    at: u64,
) -> Result<()> {
    let state: DatasetLearningStateV2 = serde_json::from_slice(&fs::read(state_path)?)?;
    let digest = RatchetStore::new(root).commit_dataset_state(&state, at)?;
    println!("{digest}");
    Ok(())
}

pub(crate) fn publish_training_head_from_files(
    root: &Path,
    head_path: &Path,
    micro_batch_path: &Path,
    replay_path: &Path,
) -> Result<()> {
    let head: TrainingHeadManifestV1 = serde_json::from_slice(&fs::read(head_path)?)?;
    let batch: LearningMicroBatchManifestV1 = serde_json::from_slice(&fs::read(micro_batch_path)?)?;
    let replay: ReplayManifestV2 = serde_json::from_slice(&fs::read(replay_path)?)?;
    let digest = RatchetStore::new(root).publish_training_head(&head, &batch, &replay)?;
    println!("{digest}");
    Ok(())
}

pub(crate) fn publish_serving_wave_from_files(
    root: &Path,
    wave_path: &Path,
    switches_path: &Path,
    splits_path: &Path,
) -> Result<()> {
    let wave: LearningWaveManifestV1 = serde_json::from_slice(&fs::read(wave_path)?)?;
    let switches: RolloutSwitchesV1 = serde_json::from_slice(&fs::read(switches_path)?)?;
    let splits: FrozenRealSplitsV1 = serde_json::from_slice(&fs::read(splits_path)?)?;
    splits.validate()?;
    let digest = RatchetStore::new(root).publish_serving_wave(
        &wave,
        &switches,
        &splits.sealed_holdout_hashes(),
    )?;
    println!("{digest}");
    Ok(())
}

pub(crate) fn admit_attempt_from_files(
    root: &Path,
    generation_path: &Path,
    admission_path: &Path,
) -> Result<()> {
    let generation: Generation0ReceiptV3 = serde_json::from_slice(&fs::read(generation_path)?)?;
    let admission: AttemptAdmissionManifestV3 = serde_json::from_slice(&fs::read(admission_path)?)?;
    let digest = RatchetStore::new(root).admit_attempt(&generation, &admission)?;
    println!("{digest}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use feat_math::evolve::random_canonical_pipeline;
    use tempfile::TempDir;

    fn hash(number: u64) -> String {
        format!("{number:064x}")
    }

    fn identities(count: usize, seed_start: u64) -> Vec<CandidateIdentityV1> {
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        let mut seed = seed_start;
        while result.len() < count {
            let canonical = random_canonical_pipeline(seed, 6);
            seed += 1;
            let identity = CandidateIdentityV1::from_canonical(canonical).unwrap();
            if seen.insert(identity.semantic_sha256.clone()) {
                result.push(identity);
            }
        }
        result
    }

    fn seed_request() -> SeedInferenceRequestV2 {
        SeedInferenceRequestV2 {
            schema: SEED_REQUEST_V2_SCHEMA.into(),
            learning_wave_sha256: hash(1),
            dataset_encoding_sha256: hash(2),
            checkpoint_sha256: hash(3),
            corrector_sha256: hash(4),
            registry_sha256: hash(5),
            vocabulary_sha256: hash(6),
            state_snapshot_sha256: hash(7),
            draw_seed: 17,
            required_count: MODEL_PROPOSAL_COUNT,
        }
    }

    fn seed_response(identities: &[CandidateIdentityV1]) -> SeedInferenceResponseV2 {
        let request = seed_request();
        SeedInferenceResponseV2 {
            schema: SEED_RESPONSE_V2_SCHEMA.into(),
            request_sha256: request.request_sha256().unwrap(),
            proposals: identities
                .iter()
                .enumerate()
                .map(|(index, candidate)| ModelProposalV2 {
                    candidate: candidate.clone(),
                    reward_prediction: index as f64,
                    disagreement_prediction: (index % 17) as f64,
                    novelty_prediction: (index % 11) as f64,
                    temperature_logit: (index % 7) as f64 / 10.0,
                })
                .collect(),
        }
    }

    fn gate(kind: WaveGateKindV1, manifest_number: u64) -> WaveGateReceiptV1 {
        let dataset_count = match kind {
            WaveGateKindV1::Offline119 => 119,
            WaveGateKindV1::Prospective64 => 64,
            WaveGateKindV1::RecurringTrainOnly | WaveGateKindV1::Corrector => 2_090,
        };
        WaveGateReceiptV1 {
            kind: kind.clone(),
            receipt_sha256: hash(100 + manifest_number),
            dataset_manifest_sha256: hash(manifest_number),
            dataset_count,
            dataset_sha256s: (0..dataset_count)
                .map(|index| hash(manifest_number * 100_000 + index as u64))
                .collect(),
            train_only: matches!(
                kind,
                WaveGateKindV1::RecurringTrainOnly | WaveGateKindV1::Corrector
            ),
            passed: true,
            genome_validity: 0.995,
            mean_normalized_regret_reduction: 0.11,
            median_robust_r2_improvement: 0.01,
            median_wall_time_ratio: 1.0,
            safety_delta: 0.0,
            pairwise_accuracy_delta: 0.0,
            normalized_regret_bootstrap_lower: 0.001,
            capture_1_delta: 0.0,
            capture_3_delta: 0.0,
            validity_delta: 0.0,
            median_time_to_champion_delta: 0.0,
            corrector_grouped_time_forward_passed: true,
            corrector_pop256_system_passed: true,
        }
    }

    fn release(name: &str, number: u64) -> ArtifactReleaseV1 {
        ArtifactReleaseV1 {
            release: name.into(),
            sha256: hash(number),
        }
    }

    fn bootstrap_wave() -> LearningWaveManifestV1 {
        LearningWaveManifestV1 {
            schema: WAVE_SCHEMA.into(),
            parent_wave_sha256: None,
            accepted_at_unix_seconds: 10_000,
            training_label_cutoff: 50_000,
            replay_manifest_sha256: hash(10),
            corpus_manifest_sha256: hash(11),
            policy_checkpoint_sha256: hash(12),
            resume_checkpoint_sha256: hash(13),
            corrector_sha256: hash(14),
            registry: release("registry-1", 15),
            vocabulary: release("vocabulary-1", 16),
            encoding_catalog: release("encoding-1", 17),
            evaluator: release("evaluator-1", 18),
            sidecar: release("sidecar-1", 19),
            gate_receipts: vec![
                gate(WaveGateKindV1::Offline119, 20),
                gate(WaveGateKindV1::Prospective64, 21),
                gate(WaveGateKindV1::Corrector, 22),
            ],
            labels_since_parent: 50_000,
            seconds_since_parent: 10_000,
            synthetic_data_ratio: 0.1,
            pilot_only: false,
            publication_sha256: String::new(),
        }
        .seal()
        .unwrap()
    }

    fn recurring_wave(
        parent_sha256: String,
        parent: &LearningWaveManifestV1,
    ) -> LearningWaveManifestV1 {
        LearningWaveManifestV1 {
            schema: WAVE_SCHEMA.into(),
            parent_wave_sha256: Some(parent_sha256),
            accepted_at_unix_seconds: parent.accepted_at_unix_seconds + MIN_WAVE_AGE_SECONDS,
            training_label_cutoff: parent.training_label_cutoff + MIN_WAVE_LABELS,
            replay_manifest_sha256: hash(30),
            corpus_manifest_sha256: hash(31),
            policy_checkpoint_sha256: hash(32),
            resume_checkpoint_sha256: hash(33),
            corrector_sha256: hash(34),
            registry: release("registry-1", 15),
            vocabulary: release("vocabulary-1", 16),
            encoding_catalog: release("encoding-1", 17),
            evaluator: release("evaluator-1", 18),
            sidecar: release("sidecar-1", 19),
            gate_receipts: vec![
                gate(WaveGateKindV1::RecurringTrainOnly, 23),
                gate(WaveGateKindV1::Corrector, 24),
            ],
            labels_since_parent: MIN_WAVE_LABELS,
            seconds_since_parent: MIN_WAVE_AGE_SECONDS,
            synthetic_data_ratio: 0.1,
            pilot_only: false,
            publication_sha256: String::new(),
        }
        .seal()
        .unwrap()
    }

    fn state_with_roles(ids: &[CandidateIdentityV1]) -> DatasetLearningStateV2 {
        let track_1_history = vec![
            TrackHistoryEntryV2 {
                sequence: 1,
                candidate: ids[0].clone(),
                authority: SelectionAuthorityV2::TrainOnlyExact,
                robust_fitness: 0.1,
                receipt_sha256: hash(40),
                committed_at_unix_seconds: 1,
            },
            TrackHistoryEntryV2 {
                sequence: 2,
                candidate: ids[2].clone(),
                authority: SelectionAuthorityV2::TrainOnlyExact,
                robust_fitness: 0.2,
                receipt_sha256: hash(41),
                committed_at_unix_seconds: 2,
            },
        ];
        let track_2_history = vec![TrackHistoryEntryV2 {
            sequence: 1,
            candidate: ids[3].clone(),
            authority: SelectionAuthorityV2::FeedbackTestTrack2,
            robust_fitness: 0.3,
            receipt_sha256: hash(42),
            committed_at_unix_seconds: 2,
        }];
        DatasetLearningStateV2 {
            schema: DATASET_STATE_SCHEMA.into(),
            dataset_sha256: hash(50),
            state_sequence: 0,
            observations: vec![],
            label_revisions: vec![],
            track_1_history,
            track_2_history,
            global_cv_frontier: (0..5)
                .map(|index| GlobalCvFrontierMemberV2 {
                    candidate: ids[index].clone(),
                    robust_fitness: 1.0 - index as f64 / 10.0,
                    comparable_fold_count: 7,
                    confirmed: index == 0,
                    receipt_sha256: hash(60 + index as u64),
                })
                .collect(),
            confirmed_cv_champion: Some(ids[0].clone()),
            // Shared semantic anchor is intentional.
            latest_exact_track_1_frontier: Some(ids[0].clone()),
            latest_model_prediction: Some(PredictionSnapshotV2 {
                candidate: ids[33].clone(),
                checkpoint_sha256: hash(70),
                prediction: 0.5,
                recorded_at_unix_seconds: 3,
            }),
            latest_corrector_prediction: Some(PredictionSnapshotV2 {
                candidate: ids[1].clone(),
                checkpoint_sha256: hash(71),
                prediction: 0.6,
                recorded_at_unix_seconds: 3,
            }),
            legacy_anchor_import: None,
            counters: LearningCountersV2::default(),
            struggle: StruggleStateV2::default(),
            scheduling: SchedulingMetadataV2::default(),
        }
    }

    fn generation_for_state(
        ids: &[CandidateIdentityV1],
        state: &DatasetLearningStateV2,
        wave_sha256: String,
    ) -> Generation0ReceiptV3 {
        let archive = ids[0..32].to_vec();
        let model = ids[32..96]
            .iter()
            .enumerate()
            .map(|(index, candidate)| RetainedModelProposalV2 {
                proposal: ModelProposalV2 {
                    candidate: candidate.clone(),
                    reward_prediction: index as f64,
                    disagreement_prediction: index as f64,
                    novelty_prediction: index as f64,
                    temperature_logit: index as f64,
                },
                bucket: if index < 32 {
                    ModelRetentionBucketV2::Reward
                } else if index < 48 {
                    ModelRetentionBucketV2::Disagreement
                } else if index < 56 {
                    ModelRetentionBucketV2::Novelty
                } else {
                    ModelRetentionBucketV2::SeededTemperature
                },
                bucket_rank: if index < 32 {
                    index
                } else if index < 48 {
                    index - 32
                } else if index < 56 {
                    index - 48
                } else {
                    index - 56
                },
            })
            .collect();
        let families = [
            GprKernelFamilyV3::Rbf,
            GprKernelFamilyV3::Matern12,
            GprKernelFamilyV3::Matern32,
            GprKernelFamilyV3::Matern52,
            GprKernelFamilyV3::RationalQuadratic,
            GprKernelFamilyV3::Periodic,
            GprKernelFamilyV3::Linear,
            GprKernelFamilyV3::WhiteComposite,
        ];
        let gpr = ids[96..104].iter().cloned().zip(families).collect();
        let mutation = ids[104..164]
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                (
                    candidate.clone(),
                    1_000 + index as u64,
                    ids[index % 32].semantic_sha256.clone(),
                )
            })
            .collect();
        let crossover = ids[164..224]
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                (
                    candidate.clone(),
                    2_000 + index as u64,
                    ids[index % 31].semantic_sha256.clone(),
                    ids[index % 31 + 1].semantic_sha256.clone(),
                )
            })
            .collect();
        let random = ids[224..256]
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.clone(), 3_000 + index as u64))
            .collect();
        let roles = vec![
            (
                GenerationRoleV3::ConfirmedCvChampion,
                ids[0].semantic_sha256.clone(),
            ),
            (
                GenerationRoleV3::LatestExactTrack1Frontier,
                ids[0].semantic_sha256.clone(),
            ),
            (
                GenerationRoleV3::CheckpointRewardTop1,
                ids[32].semantic_sha256.clone(),
            ),
            (
                GenerationRoleV3::LatestModelPrediction,
                ids[33].semantic_sha256.clone(),
            ),
            (
                GenerationRoleV3::LatestCorrectorPrediction,
                ids[1].semantic_sha256.clone(),
            ),
            (
                GenerationRoleV3::LatestTrack1Winner,
                ids[2].semantic_sha256.clone(),
            ),
            (
                GenerationRoleV3::LatestTrack2Winner,
                ids[3].semantic_sha256.clone(),
            ),
        ]
        .into_iter()
        .map(|(role, semantic_sha256)| GenerationRoleBindingV3 {
            role,
            semantic_sha256,
        })
        .collect();
        assemble_generation_zero_v3(
            wave_sha256,
            state.snapshot_sha256().unwrap(),
            state.dataset_sha256.clone(),
            hash(80),
            hash(81),
            hash(82),
            43,
            archive,
            model,
            gpr,
            mutation,
            crossover,
            random,
            roles,
        )
        .unwrap()
    }

    #[test]
    fn v1_receipts_remain_readable_but_are_not_promoted() {
        assert!(matches!(
            read_seed_inference(br#"{"schema":"SeedInferenceResponse.v1","seeds":[]}"#).unwrap(),
            ReadableSeedInference::V1(_)
        ));
        assert!(matches!(
            read_generation_zero(br#"{"schema":"Generation0Receipt.v2","population":[]}"#).unwrap(),
            ReadableGeneration0::Legacy(_)
        ));
        assert!(matches!(
            read_admission(br#"{"schema":"AttemptAdmissionManifest.v2"}"#).unwrap(),
            ReadableAdmission::Legacy(_)
        ));
        assert!(matches!(
            read_telemetry(br#"{"schema":"RatchetLiveTelemetry.v1"}"#).unwrap(),
            ReadableTelemetry::Legacy(_)
        ));
    }

    #[test]
    fn rust_model_retention_is_deterministic_and_exact() {
        let ids = identities(100, 1);
        let request = seed_request();
        let response = seed_response(&ids[..96]);
        let first = response.retain_64(&request, &BTreeSet::new()).unwrap();
        let second = response.retain_64(&request, &BTreeSet::new()).unwrap();
        assert_eq!(first, second);
        let counts = first.iter().fold(BTreeMap::new(), |mut counts, item| {
            *counts.entry(item.bucket.clone()).or_insert(0) += 1;
            counts
        });
        assert_eq!(counts[&ModelRetentionBucketV2::Reward], 32);
        assert_eq!(counts[&ModelRetentionBucketV2::Disagreement], 16);
        assert_eq!(counts[&ModelRetentionBucketV2::Novelty], 8);
        assert_eq!(counts[&ModelRetentionBucketV2::SeededTemperature], 8);

        let excluded = ids[..33]
            .iter()
            .map(|identity| identity.semantic_sha256.clone())
            .collect();
        assert!(response.retain_64(&request, &excluded).is_err());
    }

    #[test]
    fn generation_v3_enforces_topology_and_shared_anchor_roles() {
        let ids = identities(256, 10_000);
        let state = state_with_roles(&ids);
        let generation = generation_for_state(&ids, &state, hash(90));
        generation.validate_against_state(&state).unwrap();
        assert_eq!(
            generation.role_bindings[0].semantic_sha256,
            generation.role_bindings[1].semantic_sha256
        );
        assert_eq!(
            generation
                .candidates
                .iter()
                .filter(|candidate| candidate.candidate.semantic_sha256
                    == generation.role_bindings[0].semantic_sha256)
                .count(),
            1
        );

        let mut drifted = generation.clone();
        drifted.candidates[100].candidate = drifted.candidates[99].candidate.clone();
        assert!(drifted.validate().is_err());
    }

    #[test]
    fn archive_and_offspring_refills_are_deterministic_and_fail_closed() {
        let ids = identities(96, 15_000);
        let state = state_with_roles(&ids);
        let archive = deterministic_archive_32(&state, &ids).unwrap();
        assert_eq!(archive.len(), 32);
        assert_eq!(archive[0], ids[0]);
        assert_eq!(archive[1], ids[1]);
        assert_eq!(archive[2], ids[2]);
        assert_eq!(archive[3], ids[3]);

        let occupied = archive
            .iter()
            .map(|candidate| candidate.semantic_sha256.clone())
            .collect::<BTreeSet<_>>();
        let first = build_bounded_offspring_pools(&archive, &occupied, 123, 6, 8_192).unwrap();
        let second = build_bounded_offspring_pools(&archive, &occupied, 123, 6, 8_192).unwrap();
        assert_eq!(first.mutation, second.mutation);
        assert_eq!(first.crossover, second.crossover);
        assert_eq!(first.random, second.random);
        assert_eq!(first.mutation.len(), 60);
        assert_eq!(first.crossover.len(), 60);
        assert_eq!(first.random.len(), 32);
        assert!(build_bounded_offspring_pools(&archive, &occupied, 123, 6, 1).is_err());
    }

    #[test]
    fn track_2_identity_never_enters_native_ranking_features() {
        let ids = identities(40, 20_000);
        let state = state_with_roles(&ids);
        let features = NativeFitnessFeaturesV1::build(&state, hash(99), 0.2, 5, 0.1, 10.0).unwrap();
        let serialized = String::from_utf8(canonical_json(&features).unwrap()).unwrap();
        assert!(!serialized.contains("track_2"));
        assert!(!serialized.contains(&ids[3].semantic_sha256));
        assert!((features.family_shrunk_residual - 0.1 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn wave_parentage_thresholds_gates_and_holdout_firewall_are_closed() {
        let parent = bootstrap_wave();
        parent.validate(None, &BTreeSet::new()).unwrap();
        let parent_sha = parent.manifest_sha256().unwrap();
        let wave = recurring_wave(parent_sha.clone(), &parent);
        wave.validate(Some((&parent_sha, &parent)), &BTreeSet::from([hash(999)]))
            .unwrap();

        let mut early = wave.clone();
        early.seconds_since_parent -= 1;
        early.publication_sha256 = early.publication_digest().unwrap();
        assert!(early
            .validate(Some((&parent_sha, &parent)), &BTreeSet::new())
            .is_err());

        let mut leaked = wave.clone();
        let sealed = leaked.gate_receipts[0].dataset_sha256s[0].clone();
        leaked.publication_sha256 = leaked.publication_digest().unwrap();
        assert!(leaked
            .validate(Some((&parent_sha, &parent)), &BTreeSet::from([sealed]))
            .is_err());
    }

    #[test]
    fn append_only_dataset_state_is_content_addressed_and_rejects_rewrites() {
        let ids = identities(40, 30_000);
        let initial = state_with_roles(&ids);
        initial.validate().unwrap();
        let temporary = TempDir::new().unwrap();
        let store = RatchetStore::new(temporary.path());
        store.commit_dataset_state(&initial, 10).unwrap();

        let key = ObservationKeyV2 {
            dataset_sha256: initial.dataset_sha256.clone(),
            semantic_sha256: ids[10].semantic_sha256.clone(),
            evaluator_sha256: hash(101),
            folds_sha256: hash(102),
            protocol_sha256: hash(103),
        };
        let mut next = initial.clone();
        next.observations.push(PhysicalObservationV2 {
            sequence: 1,
            observation_sha256: hash(104),
            key: key.clone(),
            receipt_sha256: hash(105),
            observed_at_unix_seconds: 11,
            provenance_complete: true,
            synthetic: false,
        });
        next.label_revisions.push(LabelRevisionV2 {
            sequence: 1,
            key,
            revision: 1,
            observation_sha256: hash(104),
            revision_receipt_sha256: hash(106),
            committed_at_unix_seconds: 12,
            value: LabelValueV2 {
                status: LabelStatusV2::Exact,
                robust_fitness: Some(0.4),
                wall_time_seconds: Some(1.0),
                comparison_won: None,
            },
        });
        next.state_sequence = 2;
        next.counters.provenance_complete_labels = 1;
        next.counters.exact_labels = 1;
        store.commit_dataset_state(&next, 12).unwrap();
        assert_eq!(
            store
                .load_dataset_state(&initial.dataset_sha256)
                .unwrap()
                .unwrap()
                .1,
            next
        );

        let mut rewritten = next.clone();
        rewritten.observations[0].receipt_sha256 = hash(107);
        assert!(store.commit_dataset_state(&rewritten, 13).is_err());
    }

    fn replay_manifest() -> ReplayManifestV2 {
        replay_manifest_at(0, 25_000)
    }

    fn replay_manifest_at(offset: u64, label_cutoff: u64) -> ReplayManifestV2 {
        let reservoir_groups = (0..200)
            .map(|index| hash(offset + 20_000 + index))
            .collect::<Vec<_>>();
        let steps = (0..REPLAY_STEPS)
            .map(|step| {
                let datasets = (0..DATASETS_PER_MICRO_BATCH)
                    .map(|index| hash(offset + 10_000 + (step * 8 + index) as u64))
                    .collect::<Vec<_>>();
                let revisions = datasets
                    .iter()
                    .enumerate()
                    .map(|(index, dataset)| ReplayLabelRevisionV2 {
                        group_sha256: reservoir_groups[(step * 8 + index) % 200].clone(),
                        dataset_sha256: dataset.clone(),
                        revision: 1,
                        label_revision_sha256: hash(offset + 30_000 + (step * 8 + index) as u64),
                    })
                    .collect();
                ReplayStepV2 {
                    step,
                    datasets,
                    revisions,
                }
            })
            .collect::<Vec<_>>();
        ReplayManifestV2 {
            schema: REPLAY_V2_SCHEMA.into(),
            label_cutoff,
            reservoir_manifest_sha256: sha256_bytes(&canonical_json(&reservoir_groups).unwrap()),
            reservoir_group_count: 200,
            reservoir_groups,
            scheduled_datasets: steps[0].datasets.clone(),
            steps,
        }
    }

    fn publish_head_for_wave(
        store: &RatchetStore,
        wave: &LearningWaveManifestV1,
        base_checkpoint_sha256: String,
        replay_offset: u64,
        published_at_unix_seconds: u64,
    ) {
        let replay = replay_manifest_at(replay_offset, wave.training_label_cutoff);
        let batch = LearningMicroBatchManifestV1 {
            schema: MICRO_BATCH_SCHEMA.into(),
            batch_index: replay_offset,
            serving_wave_sha256: wave.parent_wave_sha256.clone().unwrap_or_else(|| hash(900)),
            base_training_checkpoint_sha256: base_checkpoint_sha256,
            candidate_training_checkpoint_sha256: wave.policy_checkpoint_sha256.clone(),
            candidate_resume_checkpoint_sha256: wave.resume_checkpoint_sha256.clone(),
            replay_manifest_sha256: replay.manifest_sha256().unwrap(),
            scheduler_draws: replay
                .scheduled_datasets
                .iter()
                .enumerate()
                .map(|(index, dataset)| SchedulerDrawV4 {
                    draw_index: replay_offset + index as u64,
                    lane: SchedulerLaneV4::Challenge,
                    dataset_sha256: dataset.clone(),
                    state_snapshot_sha256: hash(replay_offset + 60_000 + index as u64),
                    draw_seed: replay_offset + index as u64,
                })
                .collect(),
            update_steps: REPLAY_STEPS,
            completed: true,
        };
        let head = TrainingHeadManifestV1 {
            schema: TRAINING_HEAD_SCHEMA.into(),
            checkpoint_sha256: wave.policy_checkpoint_sha256.clone(),
            resume_checkpoint_sha256: wave.resume_checkpoint_sha256.clone(),
            corrector_sha256: wave.corrector_sha256.clone(),
            micro_batch_manifest_sha256: batch.manifest_sha256(&replay).unwrap(),
            replay_manifest_sha256: replay.manifest_sha256().unwrap(),
            update_steps: REPLAY_STEPS,
            label_cutoff: wave.training_label_cutoff,
            published_at_unix_seconds,
        };
        store.publish_training_head(&head, &batch, &replay).unwrap();
    }

    #[test]
    fn replay_is_exactly_one_hundred_novel_eight_dataset_steps() {
        let replay = replay_manifest();
        replay.validate(&BTreeSet::new()).unwrap();
        let mut historical = BTreeSet::new();
        let mut first = replay.steps[0].datasets.clone();
        first.sort();
        historical.insert(first);
        assert!(replay.validate(&historical).is_err());
    }

    #[test]
    fn scheduler_repays_measured_cpu_debt_at_70_15_10_5() {
        let mut scheduler = SchedulerStateV4::default();
        for _ in 0..200 {
            let lane = scheduler.next_lane().unwrap();
            scheduler.record_measured_cpu(lane, 100).unwrap();
        }
        assert_eq!(
            scheduler.measured_cpu_micros[&SchedulerLaneV4::Challenge],
            14_000
        );
        assert_eq!(
            scheduler.measured_cpu_micros[&SchedulerLaneV4::Uniform],
            3_000
        );
        assert_eq!(
            scheduler.measured_cpu_micros[&SchedulerLaneV4::StaleUnderexposed],
            2_000
        );
        assert_eq!(
            scheduler.measured_cpu_micros[&SchedulerLaneV4::GlobalCvConfirmation],
            1_000
        );
    }

    #[test]
    fn serving_and_training_pointers_are_separate_and_recovery_is_idempotent() {
        let temporary = TempDir::new().unwrap();
        let store = RatchetStore::new(temporary.path());
        let replay = replay_manifest();
        let replay_sha = replay.manifest_sha256().unwrap();
        let batch = LearningMicroBatchManifestV1 {
            schema: MICRO_BATCH_SCHEMA.into(),
            batch_index: 1,
            serving_wave_sha256: hash(298),
            base_training_checkpoint_sha256: hash(299),
            candidate_training_checkpoint_sha256: hash(300),
            candidate_resume_checkpoint_sha256: hash(301),
            replay_manifest_sha256: replay_sha.clone(),
            scheduler_draws: replay
                .scheduled_datasets
                .iter()
                .enumerate()
                .map(|(index, dataset)| SchedulerDrawV4 {
                    draw_index: index as u64,
                    lane: SchedulerLaneV4::Challenge,
                    dataset_sha256: dataset.clone(),
                    state_snapshot_sha256: hash(50_000 + index as u64),
                    draw_seed: index as u64,
                })
                .collect(),
            update_steps: REPLAY_STEPS,
            completed: true,
        };
        let head = TrainingHeadManifestV1 {
            schema: TRAINING_HEAD_SCHEMA.into(),
            checkpoint_sha256: hash(300),
            resume_checkpoint_sha256: hash(301),
            corrector_sha256: hash(304),
            micro_batch_manifest_sha256: batch.manifest_sha256(&replay).unwrap(),
            replay_manifest_sha256: replay_sha,
            update_steps: REPLAY_STEPS,
            label_cutoff: replay.label_cutoff,
            published_at_unix_seconds: 1,
        };
        let head_sha = store.publish_training_head(&head, &batch, &replay).unwrap();
        assert!(!temporary.path().join("waves/current.json").exists());
        assert_eq!(
            serde_json::from_slice::<RatchetArtifactPointerV1>(
                &fs::read(temporary.path().join("training-head.json")).unwrap()
            )
            .unwrap()
            .artifact_sha256,
            head_sha
        );

        let switches = RolloutSwitchesV1 {
            dataset_state_v2: true,
            scheduler_v4: true,
            replay_v2: true,
            generation_zero_v3: true,
            wave_authority: true,
        };
        let parent = bootstrap_wave();
        let parent_sha = store
            .publish_serving_wave(&parent, &switches, &BTreeSet::new())
            .unwrap();
        // Retrying identical publication is byte-idempotent.
        assert_eq!(
            store
                .publish_serving_wave(&parent, &switches, &BTreeSet::new())
                .unwrap(),
            parent_sha
        );
        let next = recurring_wave(parent_sha.clone(), &parent);
        assert!(store
            .publish_serving_wave(&next, &switches, &BTreeSet::new())
            .is_err());
        publish_head_for_wave(&store, &next, head.checkpoint_sha256.clone(), 100_000, 2);
        let next_sha = store
            .publish_serving_wave(&next, &switches, &BTreeSet::new())
            .unwrap();
        assert_eq!(store.load_serving_wave().unwrap().unwrap().0, next_sha);
        store.stop_admissions(&hash(400), 20_000).unwrap();
        store.restore_previous_wave(&parent_sha, 20_001).unwrap();
        assert_eq!(store.load_serving_wave().unwrap().unwrap().0, parent_sha);
        let mirror: RatchetArtifactPointerV1 =
            serde_json::from_slice(&fs::read(temporary.path().join("dl/current.json")).unwrap())
                .unwrap();
        assert!(!mirror.serving_authority);
    }

    #[test]
    fn admission_freezes_latest_wave_and_state_while_in_flight_retries_stay_bound() {
        let temporary = TempDir::new().unwrap();
        let store = RatchetStore::new(temporary.path());
        let switches = RolloutSwitchesV1 {
            dataset_state_v2: true,
            scheduler_v4: true,
            replay_v2: true,
            generation_zero_v3: true,
            wave_authority: true,
        };
        let wave = bootstrap_wave();
        let wave_sha = store
            .publish_serving_wave(&wave, &switches, &BTreeSet::new())
            .unwrap();
        let ids = identities(256, 40_000);
        let state = state_with_roles(&ids);
        store.commit_dataset_state(&state, 11_000).unwrap();
        let generation = generation_for_state(&ids, &state, wave_sha.clone());
        let admission = AttemptAdmissionManifestV3 {
            schema: ADMISSION_V3_SCHEMA.into(),
            attempt_id: "fixture-attempt-1".into(),
            learning_wave_sha256: wave_sha.clone(),
            dataset_state_sha256: state.snapshot_sha256().unwrap(),
            scheduler_draw_sha256: hash(500),
            generation_zero_sha256: generation.receipt_sha256().unwrap(),
            checkpoint_sha256: wave.policy_checkpoint_sha256.clone(),
            corrector_sha256: wave.corrector_sha256.clone(),
            dataset_sha256: state.dataset_sha256.clone(),
            folds_sha256: hash(501),
            proof_slate_seed: 29,
            proof_slate: ids[..12]
                .iter()
                .enumerate()
                .map(|(index, candidate)| ProofSlateCandidateV3 {
                    candidate: candidate.clone(),
                    lane: if index < 10 {
                        ProofSlateLaneV3::Selectable
                    } else {
                        ProofSlateLaneV3::CvOnlyCalibration
                    },
                })
                .collect(),
        };
        let receipt = store.admit_attempt(&generation, &admission).unwrap();
        assert!(temporary
            .path()
            .join("attempts/fixture-attempt-1/admission.v3.json")
            .exists());

        let next = recurring_wave(wave_sha, &wave);
        publish_head_for_wave(&store, &next, hash(29), 200_000, 11_500);
        store
            .publish_serving_wave(&next, &switches, &BTreeSet::new())
            .unwrap();
        // Recovery of the immutable admitted attempt does not switch waves.
        assert_eq!(
            store.admit_attempt(&generation, &admission).unwrap(),
            receipt
        );
        let mut new_attempt = admission.clone();
        new_attempt.attempt_id = "fixture-attempt-2".into();
        assert!(store.admit_attempt(&generation, &new_attempt).is_err());

        store.stop_admissions(&hash(502), 12_000).unwrap();
        assert_eq!(
            store.admit_attempt(&generation, &admission).unwrap(),
            receipt
        );
    }

    #[test]
    fn rollout_defaults_off_and_wave_authority_requires_all_dependencies() {
        let defaults: RolloutSwitchesV1 = serde_json::from_str("{}").unwrap();
        assert_eq!(defaults, RolloutSwitchesV1::default());
        defaults.validate().unwrap();
        let invalid = RolloutSwitchesV1 {
            wave_authority: true,
            ..RolloutSwitchesV1::default()
        };
        assert!(invalid.validate().is_err());
    }
}
