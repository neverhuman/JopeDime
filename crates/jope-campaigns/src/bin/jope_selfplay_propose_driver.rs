//! jope_selfplay_propose_driver — keeps the DL proposal set bound to the LIVE catalog.
//!
//! Operator mandate (WS-E / dl-seeds-gp): "the deep learning model is used to
//! seed GP runs" must stay measurably true without an agent hand-holding it.
//! Every dataset starts with eight direct-DL slots. The remaining 120 adaptive
//! slots are same-wave GP offspring; later allocation rungs are earned from
//! paired efficiency evidence. A deficient context emits fewer direct rows and
//! returns the difference to GP. Cross-context proposal backfill is forbidden.
//!
//! Before this driver existed nothing regenerated proposals when the catalog
//! grew. The expansion driver adds up to 2,048 datasets an hour; the proposal
//! set was pinned to a 2,864-dataset catalog while the live catalog reached
//! 10,533, so 72.8% of live datasets had zero DL proposals and every binary
//! DL slot in every wave was cross-dataset backfill.
//!
//! One cycle (timer-driven, cheap: two JSON reads plus `scq ls`):
//!   1. bound_catalog := catalog_sha256 the current proposal set was built for
//!   2. live_catalog  := catalog/current.json catalog_sha256
//!   3. equal            -> `up_to_date`, exit 0
//!   4. propose job live -> `waiting`, exit 0 (never two GPU propose jobs)
//!   5. job completed    -> fetch, verify, install, atomically rebind
//!   6. otherwise        -> render spec, `scq check`, `scq submit`
//!
//! Checkpoint identity is held CONSTANT across a rebind on purpose:
//! `jope_selfplay_queue.rs valid_micro_waves()` filters waves by
//! checkpoint_epoch, so moving the checkpoint here would silently invalidate
//! every existing wave and its unclaimed manifests. Promotion (stream A) owns
//! the checkpoint; this driver only ever re-binds the catalog underneath the
//! checkpoint that promotion already accepted.

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde_json::{json, Map, Value};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, default_scq_str, now_ms, read_json, sha256_file,
};
use jope_dime_regressor::spec_policy::PROPOSE;

#[path = "common/flock.rs"]
mod flock;

jope_paths::once_str!(
    fn scripts_str => paths::research_scripts_root()
);
jope_paths::once_str!(
    fn proposal_selection_policy_str => paths::campaign_manifest("selfplay/proposal-selection-policy.v1.json")
);
jope_paths::once_str!(
    fn prime_src_str => paths::runtime_bundle_pytorch()
);
jope_paths::once_str!(
    fn checkpoints_str => paths::legacy_checkpoints_root()
);
const STATE_SCHEMA: &str = "JopeProposeDriver.v1";
const EVENT_SCHEMA: &str = "JopeProposeDriverEvent.v1";
/// Initial direct-DL allocation. Future rungs are pointer-bound evidence.
const PROPOSALS_PER_DATASET: i64 = 8;
/// Deficiency is no longer fatal or backfilled; retained for CLI compatibility.
const ALLOW_DEFICIENT_FLOOR: i64 = 0;
const PROPOSAL_SEED: i64 = 20260726;
const MAX_ATTEMPTS: u64 = 3;

// ---------------------------------------------------------------------------
// WS-6: per-task checkpoint selection
// ---------------------------------------------------------------------------
//
// Operator decision: per-task accepted checkpoints. A generation that trades
// one branch against the other can never clear a single global gate, so the
// promotion gate now records a per-task verdict and advances
// `$R/training/accepted-by-task.json`
// (`jope_selfplay_promotion.rs::advance_task_accepted_pointers`). This driver
// is the consumer: it SELECTS the checkpoint per branch.
//
// The selection is a fork in the GENERATION lane only. Publication is not
// forked, because `jope_selfplay_wave.rs:1072` and `:1325` hard-fail
// `"proposal file mixes checkpoint epochs at line {n}"` for any row whose
// `model_checkpoint_sha256` differs from the cohort plan's single
// `checkpoint_sha256`, and `jope_selfplay_queue.rs valid_micro_waves()` filters
// waves by a single `checkpoint_epoch`. So:
//   * branches agree on a checkpoint -> exactly the historical behaviour: one
//     whole-catalog job, published into `proposals/current.json`.
//   * branches disagree              -> one job PER BRANCH, each carrying a
//     single (task, checkpoint) identity, published under
//     `proposals/by-task/<task>/` and named by the ADDITIVE pointer
//     `proposals/current-by-task.json`. `proposals/current.json` is left
//     untouched, so no consumer can ever be handed a mixed-epoch file.
const TASK_BRANCHES: [&str; 4] = ["binary", "regression", "multiclass", "timeseries"];
const TASK_ACCEPTED_POINTER: &str = "training/accepted-by-task.json";
const TASK_POINTER_SCHEMA: &str = "JopeSelfplayProposalPointerByTask.v1";

/// Two-letter lane tag so a per-task job id can never collide with the
/// whole-catalog family prefix `jope-selfplay-propose-<sha12>-c`.
///
/// Every task gets its OWN tag. The old `_ => "tr"` fallback would have tagged
/// timeseries jobs identically to regression jobs, so a timeseries proposal
/// job id could collide with -- and silently shadow -- a regression one.
fn task_tag(task: &str) -> &'static str {
    match task {
        "binary" => "tb",
        "regression" => "tr",
        "multiclass" => "tm",
        "timeseries" => "tt",
        other => unreachable!("task_tag called with unknown task {other}"),
    }
}

#[derive(Parser, Debug)]
#[command(about = "Rebind the DL proposal set to the live catalog, unattended")]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    /// Plan only: never submit, fetch, or rebind.
    #[arg(long)]
    dry_run: bool,
}

mod catalog;
mod cycle;
mod publish;
mod spec;

use catalog::*;
use cycle::*;
use publish::*;
use spec::*;

fn main() -> Result<()> {
    let args = Args::parse();
    let lock_path = args.root.join("ops/propose-driver.lock");
    let Some(_lock) = try_lock(&lock_path)? else {
        println!("{}", json!({"schema": EVENT_SCHEMA, "action": "locked"}));
        return Ok(());
    };
    let outcome = cycle(&args)?;
    println!("{outcome}");
    Ok(())
}

#[cfg(test)]
#[path = "jope_selfplay_propose_driver/tests.rs"]
mod tests;
