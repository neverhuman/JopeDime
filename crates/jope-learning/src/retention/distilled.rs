//! `gp_distilled/v1`: the permanent (T0) table that keeps every GP result's training signal
//! after its raw bodies are packed and archived (strategy §8.3).
//!
//! One row per (program_sha, context, fold, score identity, source object). Failures and
//! partial results are rows too, with `outcome` set and no score: they are negative signal
//! (§8.5 E4). Validation refuses rows that would corrupt the table's meaning:
//! - a score must be finite and carry a known score identity; scores are never compared across
//!   identities;
//! - `qualified=true` only for a paired L6 evaluation against frozen v1 (`score_name = l6` with
//!   an `l6_delta`); L6 is a loss, lower is better, so `l6_delta < 0` is a candidate improvement;
//! - `superseded_by` needs a `champion_id`, and only qualified champions supersede (§8.5).

mod core;
mod gp_index;
mod selfplay;

pub use core::*;
pub use gp_index::*;
pub use selfplay::*;
