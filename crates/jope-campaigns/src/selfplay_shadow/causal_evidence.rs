use super::{
    DlAncestryKind, Sha256Digest, Task, TrainerMode, CANARY_INTENT_SCHEMA,
    CANDIDATE_PROVENANCE_SCHEMA, CATALOG_BINDING_SCHEMA, CAUSAL_EVALUATION_RECEIPT_SCHEMA,
    CAUSAL_EVIDENCE_LEDGER_V2_SCHEMA, CORPUS_FREEZE_STATE_SCHEMA, FLYWHEEL_LIVENESS_SCHEMA,
    MIN_PARENT_DIRECT_OUTCOMES, PACKED_CORPUS_SCHEMA, TASK_CANARY_ACCEPTANCE_V2_SCHEMA,
    TRAINER_INTENT_V3_SCHEMA,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const REQUESTED_CAUSAL_PROPOSALS_PER_CONTEXT: usize = 8;
pub const CANARY_PROPOSALS_PER_LANE: usize = 8;

mod canary;
pub use canary::{
    deterministic_canary_contexts, CanaryContextClass, CanaryContextV2, CanaryIntentV2, CanaryMode,
    TaskCanaryAcceptanceV2, TaskCanaryMetricsV2,
};

mod corpus;
mod ledger;
mod trainer;

#[cfg(test)]
mod tests;

pub use corpus::*;
pub use ledger::*;
pub use trainer::*;
