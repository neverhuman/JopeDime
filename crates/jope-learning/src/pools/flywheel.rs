//! Multi-round `gp_distilled/v1` self-play flywheel over frozen shadow pools (CLOVER Phase 3 / J9).
//!
//! Each outer cross-fitting fold `k = 0..K-1` maintains its own strictly isolated training-split
//! acquisition trajectory over `{ j : fold(j) != k }` across rounds `r = 0..R-1`, so held-out fold
//! `k` (`{ i : fold(i) == k }`) never enters or influences outer fold `k`'s training archive at
//! any round.
//!
//! Alongside the active surrogate-guided acquisition trajectory (`active_acquired_by_fold[k]`),
//! each outer fold also maintains a matched-budget passive-acquisition control trajectory
//! (`passive_acquired_by_fold[k]`) that starts from the identical Round 0 seed set and acquires the
//! exact same number of unvisited candidates per context per round in deterministic outcome-blind
//! hash order (`sha256("round0:{fold_salt}:{context_id}:{program_sha256}")`).
//!
//! Each round `r = 0..R-1`:
//! 1. Distills the cumulative acquired training observations per outer fold into a validated
//!    `gp_distilled/v1` Parquet shard (`retention::distilled::write`), reads the shard back
//!    (`retention::distilled::read`), verifies E3 key reconciliation (`retention::distilled::reconcile`),
//!    and reconstructs each outer fold's training sub-pools from the reloaded Parquet rows.
//! 2. Evaluates closed-loop measured-tilt leaf distributions (`measured_tilt_leaf_log_probs`) and
//!    `decomposed_support_kl` contract admissibility over each training sub-pool's base vs newly
//!    acquired support. Any training context that violates the KL / new-support-mass bound is
//!    actively frozen (`kl_frozen_contexts`) so its next-round acquisition falls back to the
//!    outcome-blind base reference order instead of surrogate selection.
//! 3. Fits `ContextualSurrogateQHat` per outer fold `k` on both the active training sub-pools and
//!    the matched-budget passive control training sub-pools (`fold != k`), scores held-out fold `k`
//!    on the full frozen shadow pools across budgets `B`, advances held-out discovery trajectories
//!    (`heldout_active_acquired` vs `heldout_passive_acquired`, never used for training) by
//!    `step_budget_per_context`, and acquires Round `r + 1` training candidates strictly within
//!    outer fold `k`'s training split (`fold != k`).
//!
//! Shadow-only: never claims `qualified = true` or `l6`.

use super::bootstrap::{cluster_bootstrap_ci, IntervalVerdict, PairedInterval};
use super::evaluate::{
    evaluate_with_fold_train_pools_and_priors, fold_of, Config as EvalConfig, Inputs, Summary,
};
use super::metrics::{score_context, ContextMetrics};
use super::pool::{verify, Manifest, Member, Pool, PoolView, Status, POOL_SCHEMA};
use super::selectors::{
    pre_screen_funnel, token_bag_features, ContextualSurrogateQHat, CvEstimate,
    ExternalProgramPrior, Features, Selector,
};
use crate::canon::{canonical_sha256, sha256_hex};
use crate::clover::expansion::{measured_tilt_leaf_log_probs, MeasuredTiltPolicyConfig};
use crate::retention::distilled::{
    self, distill_selfplay_observations, distilled_rows_to_measurement_rows, score_direction,
    DistilledOutcome, ScoreDirection, SelfPlayDistillObservation, SelfPlayDistillSpec, SplitRole,
};
use crate::support::decomposed_support_kl;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

mod distill;
mod execution;
mod priors;
mod rounds;
mod types;

pub use execution::{run_distilled_flywheel, run_distilled_flywheel_with_s3_prior};
pub use priors::{build_fold_s3_priors, known_scores};
pub(super) use priors::{paired_metrics, paired_scalars, summarize_metrics};
pub use types::{
    FlywheelConfig, FlywheelFoldCv, FlywheelReceipt, FlywheelRoundSummary, RoundLiftComparison,
    FLYWHEEL_RECEIPT_SCHEMA,
};
