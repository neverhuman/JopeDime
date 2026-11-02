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

mod admission;
mod api;
mod generation;
mod microbatch;
mod scheduler;
mod state;
mod store;
mod telemetry;
mod wave;

#[cfg(test)]
mod tests;

use admission::valid_attempt_id;
pub(crate) use admission::{
    AttemptAdmissionManifestV3, ProofSlateCandidateV3, ProofSlateLaneV3, ReplayLabelRevisionV2,
    ReplayManifestV2, ReplayStepV2,
};
pub(crate) use api::{
    admit_attempt_from_files, commit_dataset_state_from_file, publish_serving_wave_from_files,
    publish_training_head_from_files, retain_model_seeds_from_files, validate_artifact_file,
};
pub(crate) use generation::{
    assemble_generation_zero_v3, build_bounded_offspring_pools, deterministic_archive_32,
    CandidateProvenanceV3, Generation0ReceiptV3, GenerationCandidateV3, GenerationRoleBindingV3,
    GenerationRoleV3, GprKernelFamilyV3, OffspringPoolsV3,
};
use generation::{derived_seed, push_generation_candidate, push_unique_identity};
pub(crate) use microbatch::{
    LearningMicroBatchManifestV1, RolloutSwitchesV1, TrainingHeadManifestV1,
};
use scheduler::validate_dataset_set;
pub(crate) use scheduler::{SchedulerDrawV4, SchedulerLaneV4, SchedulerStateV4};
use state::{
    canonical_json, require_unique, sha256_bytes, validate_hashes, validate_sha256,
    validate_track_history, ADMISSION_V3_SCHEMA, DATASET_STATE_SCHEMA, GENERATION_ZERO_V3_SCHEMA,
    MICRO_BATCH_SCHEMA, POINTER_SCHEMA, REPLAY_V2_SCHEMA, SEED_REQUEST_V2_SCHEMA,
    SEED_RESPONSE_V2_SCHEMA, TRAINING_HEAD_SCHEMA, WAVE_SCHEMA,
};
pub(crate) use state::{
    CandidateIdentityV1, CurrentLabelV2, DatasetLearningStateV2, GlobalCvFrontierMemberV2,
    LabelRevisionV2, LabelStatusV2, LabelValueV2, LearningCountersV2, LegacyAnchorImportV2,
    ObservationKeyV2, PhysicalObservationV2, PredictionSnapshotV2, SchedulingMetadataV2,
    SelectionAuthorityV2, StruggleStateV2, TrackHistoryEntryV2, DATASETS_PER_MICRO_BATCH,
    GENERATION_ZERO_SIZE, MIN_WAVE_AGE_SECONDS, MIN_WAVE_LABELS, MODEL_PROPOSAL_COUNT,
    MODEL_RETAINED_COUNT, REPLAY_STEPS,
};
use store::{atomic_bytes_write, atomic_pointer_write, immutable_write, WriterGuard};
pub(crate) use store::{RatchetArtifactPointerV1, RatchetStore};
pub(crate) use telemetry::{
    BroadShadowPilotManifestV1, FrozenRealSplitsV1, RatchetLiveTelemetryV2,
    TrainingExperimentManifestV1,
};
use wave::{descending_f64, seeded_temperature_score, select_ranked};
pub(crate) use wave::{
    read_admission, read_generation_zero, read_seed_inference, read_telemetry, ArtifactReleaseV1,
    LearningWaveManifestV1, LegacySeedInferenceV1, ModelProposalV2, ModelRetentionBucketV2,
    ModelRetentionReceiptV2, NativeFitnessFeaturesV1, ReadableAdmission, ReadableGeneration0,
    ReadableSeedInference, ReadableTelemetry, RetainedModelProposalV2, SeedInferenceRequestV2,
    SeedInferenceResponseV2, WaveGateKindV1, WaveGateReceiptV1,
};
