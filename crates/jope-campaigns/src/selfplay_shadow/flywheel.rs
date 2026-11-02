use super::{
    DlAncestryKind, Sha256Digest, Task, CAUSAL_EVIDENCE_LEDGER_SCHEMA, GPU_LANE_ADMISSION_SCHEMA,
    MIN_PARENT_DIRECT_OUTCOMES, SHADOW_BATTLE_SUMMARY_SCHEMA, SHADOW_CHECKPOINT_SCHEMA,
    TRAINER_INTENT_SCHEMA, TRAINER_INTENT_V3_SCHEMA,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const GPU_COMPANION_CORES: usize = 4;
pub const GPU_IDLE_ALARM_MS: u128 = 5 * 60 * 1_000;
pub const GPU_ADMISSION_RETRY_MS: u128 = 2 * 60 * 1_000;
pub const SHADOW_CANARY_CONTEXTS: usize = 16;
pub const SHADOW_PROPOSALS_PER_CONTEXT: usize = 8;
pub const SHADOW_COMPARATORS_PER_CONTEXT: usize = 8;
pub const REQUIRED_CONSECUTIVE_CANARY_ACCEPTANCES: usize = 3;

mod admission;
mod alternation;
mod contracts;
mod evidence;

pub use admission::*;
pub use alternation::*;
pub use contracts::*;
pub use evidence::*;

#[cfg(test)]
mod tests;
