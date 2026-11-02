//! Generation-zero policy consumption followed by bounded, callback-driven GP.
//!
//! Every fit goes through `ExecutionPool`; no engine authority or GPU fence is
//! changed here. Producer proof comes from `jope_policy::seeds::Proposal`, never
//! an origin string read from a seed file. A pinned bounded child transports
//! that producer result with bound request and output receipts. The Candle policy's arithmetic remains
//! explicitly unqualified against native CUDA BF16.

use anyhow::{ensure, Context, Result};
use feat_math::evolve;
use jope_policy::seeds::{ORIGIN_NEURAL, ORIGIN_NEURAL_FALLBACK, ORIGIN_RANDOM};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

use crate::{
    execution::{ArmBudget, ExecutionPool, WorkerJob, WorkerOutcome},
    experiment::{EvaluationRequest, EvaluationStatus},
    search_plan::{OofEvidence, SelectionPlan},
    sha256_hex,
    synthesized_recipe::{SynthesizedRecipe, FIT_RANDOMNESS_POLICY, RAW_INPUT_POLICY},
};

pub const PROTOCOL: &str = "asm.bounded-neural-search/v1";
pub const INITIAL_SIZE: usize = 80;
pub const NEURAL_SLOTS: usize = 64;
/// Incumbents occupy the final generation-zero slots of their own arm. In the
/// neural arm this is the exploration tail, so all 64 policy slots remain.
pub const MAX_INCUMBENTS: usize = INITIAL_SIZE - NEURAL_SLOTS;
const ELITES: usize = 4;

include!("neural_search_parts/plans.rs");
include!("neural_search_parts/candidates.rs");
include!("neural_search_parts/pair_setup.rs");
include!("neural_search_parts/verification.rs");
include!("neural_search_parts/population.rs");
include!("neural_search_parts/search.rs");
include!("neural_search_parts/pool_backend.rs");
#[cfg(test)]
mod tests {
    use super::*;
    include!("neural_search_parts/tests_fixtures.rs");
    include!("neural_search_parts/tests_receipt_validation.rs");
    include!("neural_search_parts/tests_population_replay.rs");
    include!("neural_search_parts/tests_resume.rs");
    include!("neural_search_parts/tests_incumbents.rs");
}

include!("neural_search_parts/timing.rs");
