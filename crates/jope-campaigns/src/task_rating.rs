//! Deterministic within-task adaptive/diagnostic ratings.
//!
//! Ratings are operational diagnostics only.  Every policy/context carries
//! eight slots; absent or invalid slots receive utility -1 instead of being
//! dropped.  The fitter is order-independent and anchored exactly at 1500.

use crate::four_task::bounded_task_utility;
use crate::selfplay_shadow::{Sha256Digest, Task};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const TASK_RATING_SNAPSHOT_SCHEMA: &str = "JopeTaskRatingSnapshot.v1";
pub const TASK_RATING_BUILD_INPUT_SCHEMA: &str = "JopeTaskRatingBuildInput.v1";
pub const RATING_SLOTS: usize = 8;
pub const RATING_BOOTSTRAP_RESAMPLES: usize = 10_000;
const LOGISTIC_SCALE: f64 = std::f64::consts::LN_10 / 400.0;
const INITIAL_RATING: f64 = 1500.0;
const MAX_ITERATIONS: usize = 250;
const L2_REGULARIZATION: f64 = 1e-4;
const CONVERGENCE_TOLERANCE: f64 = 1e-8;
const UPDATE_CLAMP: f64 = 40.0;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RatingKindV1 {
    AdaptiveSearch,
    DiagnosticAnchor,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RatingPlayerV1 {
    pub player: String,
    pub checkpoint_sha256: Sha256Digest,
    pub ancestry_sha256: Sha256Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RatingContextV1 {
    pub context: String,
    pub context_sha256: Sha256Digest,
    pub source_family_sha256: Sha256Digest,
    pub near_duplicate_cluster_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub protocol_sha256: Sha256Digest,
    pub runtime_sha256: Sha256Digest,
    pub bundle_sha256: Sha256Digest,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RatingSlotObservationV1 {
    pub player: String,
    pub context_sha256: Sha256Digest,
    pub slot: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub valid: bool,
    pub failed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRatingBuildInputV1 {
    pub schema: String,
    pub task: Task,
    pub rating_kind: RatingKindV1,
    pub rating_epoch: String,
    pub fixed_anchor_player: String,
    /// The player represented on the four-task operational scoreboard. This is
    /// explicit so ingestion never guesses from player names or rating order.
    pub display_player: String,
    pub proposal_selection_policy_sha256: Sha256Digest,
    pub players: Vec<RatingPlayerV1>,
    pub contexts: Vec<RatingContextV1>,
    pub observations: Vec<RatingSlotObservationV1>,
    pub bootstrap_seed: u64,
    pub bootstrap_resamples: usize,
    pub generated_at_ms: u128,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RatedSlotV1 {
    pub slot: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub utility: f64,
    pub valid: bool,
    pub failed: bool,
    pub missing: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RatedPolicyContextV1 {
    pub player: String,
    pub context_sha256: Sha256Digest,
    pub source_family_sha256: Sha256Digest,
    pub slots: Vec<RatedSlotV1>,
    pub policy_utility: f64,
    pub best_of_eight_utility: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RatedPlayerV1 {
    pub player: String,
    pub checkpoint_sha256: Sha256Digest,
    pub ancestry_sha256: Sha256Digest,
    pub rating: f64,
    pub rating_interval_95: [f64; 2],
    pub wins: u64,
    pub losses: u64,
    pub ties: u64,
    pub validity: f64,
    pub failure_rate: f64,
    pub p95_latency_ms: Option<u64>,
    pub family_count: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RatingConvergenceV1 {
    pub converged: bool,
    pub iterations: usize,
    pub tolerance: f64,
    pub maximum_update: f64,
    pub l2_regularization: f64,
    pub update_clamp: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JopeTaskRatingSnapshotV1 {
    pub schema: String,
    pub task: Task,
    pub rating_kind: RatingKindV1,
    pub rating_epoch: String,
    pub task_metric: String,
    pub utility_transform: String,
    pub fixed_anchor_player: String,
    pub display_player: String,
    pub proposal_selection_policy_sha256: Sha256Digest,
    pub proposal_slots_per_player_context: usize,
    pub invalid_slot_utility: f64,
    pub players: Vec<RatedPlayerV1>,
    pub contexts: Vec<RatingContextV1>,
    pub policy_contexts: Vec<RatedPolicyContextV1>,
    pub convergence: RatingConvergenceV1,
    pub bootstrap_resamples: usize,
    pub promotion_eligible: bool,
    pub promotion_ineligible_reason: String,
    pub generated_at_ms: u128,
}

impl JopeTaskRatingSnapshotV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TASK_RATING_SNAPSHOT_SCHEMA,
            "task rating snapshot schema is not v1"
        );
        ensure!(
            self.proposal_slots_per_player_context == RATING_SLOTS
                && self.invalid_slot_utility == -1.0
                && self.bootstrap_resamples == RATING_BOOTSTRAP_RESAMPLES,
            "rating budget/bootstrap contract drift"
        );
        ensure!(
            !self.promotion_eligible && !self.promotion_ineligible_reason.trim().is_empty(),
            "adaptive/diagnostic rating was marked as promotion evidence"
        );
        ensure!(self.convergence.converged, "rating fit did not converge");
        ensure!(
            self.players
                .iter()
                .any(|player| player.player == self.display_player),
            "declared display player is absent"
        );
        ensure!(
            self.players
                .iter()
                .find(|player| player.player == self.fixed_anchor_player)
                .is_some_and(|player| player.rating == INITIAL_RATING),
            "fixed rating anchor is not exactly 1500"
        );
        ensure!(
            self.policy_contexts.iter().all(|row| {
                row.slots.len() == RATING_SLOTS
                    && row
                        .slots
                        .iter()
                        .enumerate()
                        .all(|(index, slot)| slot.slot == index && slot.utility.is_finite())
            }),
            "rating snapshot lacks eight dense slots"
        );
        Ok(())
    }
}

mod algorithm;
pub use algorithm::build_task_rating_snapshot;
