use super::{
    build_causal_evidence_ledger_v2, CausalEvaluationReceiptV2, CausalEvidenceLedgerV2,
    Sha256Digest, Task, TaskCausalEvidenceV2, CAUSAL_EVIDENCE_EPOCH_SCHEMA,
    CAUSAL_EVIDENCE_LEDGER_V3_SCHEMA, CONTEXT_PROPOSAL_ARTIFACT_V3_SCHEMA,
    DATASET_PROFILE_V1_SCHEMA, DATASET_PROVENANCE_V2_SCHEMA, FLYWHEEL_PROGRESS_V2_SCHEMA,
    MIN_PARENT_DIRECT_OUTCOMES, PACKED_CORPUS_V3_SCHEMA, SEALED_PANEL_CANDIDATE_SET_V2_SCHEMA,
    SEALED_PANEL_CONSUMPTION_V1_SCHEMA, SEALED_PANEL_CONTEXT_FACTS_V1_SCHEMA,
    SEALED_PANEL_POOL_V2_SCHEMA, TASK_PROMOTION_EVALUATION_INTENT_V3_SCHEMA,
    TASK_PROMOTION_EVALUATION_V3_SCHEMA, TASK_PROMOTION_OBSERVATION_SET_V3_SCHEMA,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const EVIDENCE_SLOTS_PER_CONTEXT: usize = 8;
pub const PREFLIGHT_COHORTS_REQUIRED: usize = 10;
pub const PREFLIGHT_CONTEXTS_PER_COHORT: usize = 64;
pub const PREFLIGHT_MIN_VALIDITY: f64 = 0.995;
pub const PROMOTION_MIN_VALIDITY: f64 = 0.99;
pub const PROMOTION_MIN_CONTEXTS: usize = 128;
pub const PROMOTION_MAX_CONTEXTS: usize = 512;
pub const PROMOTION_CONTEXT_STEP: usize = 64;
pub const PROMOTION_BOOTSTRAP_RESAMPLES: usize = 10_000;

mod preflight;
pub use preflight::*;

mod proposal_cache;
pub use proposal_cache::*;

mod dataset_profile;
pub use dataset_profile::*;

mod sealed_panels;
pub use sealed_panels::*;

mod promotion_eval;
pub use promotion_eval::*;

mod packed_corpus;
pub use packed_corpus::*;

mod plateau;
pub use plateau::*;

mod progress;
pub use progress::*;

#[cfg(test)]
mod tests;
