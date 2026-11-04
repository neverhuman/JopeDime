//! Lossless partition of one global batch across any number of ranks (W7-T06).
//!
//! Partition v1 (`task_contract::training_partition`) averages
//! microbatch-local means over microbatches and ranks. It needs
//! `global_batch % (world × microbatch) == 0` and a world in {1, 2, 4, 8}, so
//! no three-rank partition of the 2048-row batch exists. With unequal shards
//! it would also weight comparisons unequally: each rank-local mean would
//! count as much as any other, whatever its size.
//!
//! Partition v2 is a new version, never a reinterpretation of v1.
//! - **Comparisons stay whole.** The global batch is `G = global_batch / 8`
//!   complete comparisons, drawn by the same serial sampler at the same global
//!   cursors whatever the rank count, so the data of a step does not depend on
//!   the partition.
//! - **Deterministic, lossless assignment.** Rank `r` takes a contiguous block
//!   of `⌊G/W⌋ + [r < G mod W]` comparisons, so nothing is dropped. Each
//!   rank's block is split into microbatches of at most `microbatch_rows / 8`
//!   comparisons; the last microbatch may be smaller.
//! - **Global reduction.** Every term is reduced by its global numerator over
//!   its global denominator: `L = Σ_T c_T · N_T / D_T`.
//!   - The denominators depend only on labels ([`denominators`]), so every
//!     rank computes them for the whole global batch from the deterministic
//!     sampler, with no extra communication.
//!   - A microbatch's objective is `Σ_T c_T · N_T,mb / D_T` ([`composite`]),
//!     formed from the v1 microbatch-local means scaled by
//!     `D_T,mb / D_T`. A rank sums its microbatches without dividing by their
//!     count, and gradients are summed across ranks. With a mean-reducing
//!     transport each rank scales its objective by `W` ([`Plan::local_scale`]).
//! - **Empty cases are explicit.**
//!   - A rank with no comparisons has no microbatch. It backpropagates
//!     [`empty_rank_objective`], an exact zero connected to every parameter,
//!     so it contributes a defined zero gradient and still joins every
//!     collective with the same packed layout.
//!   - A term whose global denominator is zero is globally empty: value 0 and
//!     gradient 0, recorded in [`Denominators::empty_terms`]. For validated
//!     supervision only the two Bradley–Terry terms can be empty, when every
//!     pair ties.
//! - **Rank-local state.** The dropout RNG is counter based
//!   (`consumption::RNG_ID`, reset before every microbatch from seed, local
//!   step, rank and microbatch), and the sampler is a pure function of the
//!   global cursor. [`State`] persists both for every rank at a committed step,
//!   with the global committed update, and exact resume re-derives and checks
//!   them.
//! - **Rank-count changes are explicit transitions.** The objective does not
//!   depend on the rank count, but the RNG streams and shard boundaries do, so
//!   a different rank count or microbatch is a changed execution partition:
//!   exact resume refuses it, an optimizer-carry transition must declare it
//!   (`continuation::check_parent_partition`), and a weight warm start
//!   carries nothing. Exact equality with the single-process reference is
//!   claimed only with dropout off.
use super::objective::{self, row_weight, Outputs, RowLabels, COEFFICIENTS, COMPARISON_ROWS};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tch::{Kind, Tensor};

/// The record, shard type and assignment live in `task_contract`, so code
/// outside the neural build can name and check a v2 partition.
pub use crate::task_contract::{assign_partition as assign, PartitionShard as Shard};
use crate::task_contract::{PARTITION_V1_SCHEMA, PARTITION_V2_SCHEMA};

pub const SCHEMA: &str = PARTITION_V2_SCHEMA;
pub const NORMALIZATION: &str = crate::task_contract::PARTITION_V2_NORMALIZATION;
pub const ASSIGNMENT: &str = crate::task_contract::PARTITION_V2_ASSIGNMENT;
pub const STATE_SCHEMA: &str = "parallel.partition_state.v1";
/// Frozen numerical policy against the declared single-process reference:
/// the v1 [`objective::composite`] over the whole global batch as one
/// microbatch, in the same f32 arithmetic. A partitioned objective must be
/// within `LOSS_RTOL · max(1, |reference|)`; each output gradient within
/// `GRADIENT_RTOL · max|reference gradient|` (so an exactly zero reference
/// gradient must be reproduced exactly).
pub const LOSS_RTOL: f64 = crate::task_contract::PARTITION_V2_LOSS_RTOL;
pub const GRADIENT_RTOL: f64 = crate::task_contract::PARTITION_V2_GRADIENT_RTOL;
/// Rows of the fixed validation panel: under v2 the global panel is these
/// rows split by the plan, the same rows as the v1 eight-rank panel.
pub const VALIDATION_PANEL_ROWS: usize = 2048;
/// Relative slack when comparing a local f64 denominator sum with the global
/// one; only rounding of the same positive summands can exceed it.
const DENOMINATOR_SLACK: f64 = 1e-9;

fn digest(value: &Value) -> String {
    crate::manifest::hash(&serde_json::to_vec(value).expect("JSON value serializes"))
}

fn is_sha256(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

mod execution;
mod plan;
mod reduce;
mod state;

pub use execution::{execution, objective_identity, objective_sha256, Execution, Version};
pub use plan::{Plan, Slot, Transport};
pub use reduce::{
    composite, denominators, empty_rank_objective, gradient_sha256, gradient_within_policy,
    loss_within_policy, Denominators,
};
pub use state::{CommittedUpdate, RankState, State};
