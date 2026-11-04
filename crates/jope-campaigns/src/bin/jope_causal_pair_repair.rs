//! Execute exact, receipt-level causal pair repairs without imputing outcomes.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_ingest::ingest_receipt_with_manifest;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, default_scq_str, now_ms, read_json, sha256_file,
};
use jope_dime_regressor::selfplay_shadow::{CausalLane, PairRepairRequestV1, Sha256Digest, Task};
use jope_dime_regressor::spec_policy::EVAL;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

const QUEUE_SCHEMA: &str = "CausalPairRepairQueue.v1";
const STATE_SCHEMA: &str = "CausalPairRepairExecutorState.v1";
const REPAIR_SCHEMA: &str = "CausalPairRepairManifest.v1";
const ACTIVE_STATES: &[&str] = &["queued", "starting", "dispatching", "running", "held"];

#[derive(Parser, Debug)]
#[command(
    name = "jope-causal-pair-repair",
    about = "Rerun only missing or failed causal pair members"
)]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    #[arg(long)]
    runner: Option<PathBuf>,
    #[arg(long)]
    once: bool,
    #[arg(long)]
    verify_only: bool,
    #[arg(long, default_value_t = 60)]
    interval_seconds: u64,
    #[arg(long, default_value_t = 40)]
    max_inflight: usize,
    #[arg(long, default_value_t = 20)]
    submissions_per_cycle: usize,
    #[arg(long, default_value_t = 3)]
    max_attempts: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RepairQueue {
    schema: String,
    #[allow(dead_code)]
    generated_at_ms: u128,
    repairs: Vec<PairRepairRequestV1>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RepairEntry {
    pair_id: Sha256Digest,
    task: Task,
    missing_lane: CausalLane,
    status: String,
    attempts: usize,
    identical_failures: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    retry_after_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    spec_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    repair_manifest_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    original_manifest_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    original_manifest_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    original_evaluator_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    candidate_sha256: Option<Sha256Digest>,
    updated_at_ms: u128,
}

impl RepairEntry {
    fn new(request: &PairRepairRequestV1) -> Self {
        Self {
            pair_id: request.pair_id.clone(),
            task: request.task,
            missing_lane: request.missing_lane,
            status: if request.permanent_failure {
                "permanent_failure"
            } else {
                "pending"
            }
            .to_owned(),
            attempts: 0,
            identical_failures: 0,
            last_error: None,
            retry_after_ms: 0,
            job_id: None,
            spec_path: None,
            repair_manifest_path: None,
            original_manifest_path: None,
            original_manifest_sha256: None,
            original_evaluator_sha256: None,
            candidate_sha256: None,
            updated_at_ms: now_ms(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RepairState {
    schema: String,
    updated_at_ms: u128,
    entries: BTreeMap<String, RepairEntry>,
}

impl Default for RepairState {
    fn default() -> Self {
        Self {
            schema: STATE_SCHEMA.to_owned(),
            updated_at_ms: now_ms(),
            entries: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct PairIdentity {
    task: Task,
    context_identity: String,
    accepted_parent_checkpoint_sha256: String,
    wave: u64,
    pair_index: usize,
    receipt_sha256s: Vec<String>,
}

#[derive(Clone, Debug)]
struct TargetCandidate {
    manifest_path: PathBuf,
    manifest_sha256: String,
    candidate_sha256: String,
    candidate: Value,
    manifest: Value,
}

#[derive(Clone, Debug)]
struct EvaluatorBinding {
    path: PathBuf,
    sha256: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.max_inflight > 0, "max inflight must be positive");
    ensure!(
        args.submissions_per_cycle > 0,
        "submissions per cycle must be positive"
    );
    ensure!(args.max_attempts > 0, "max attempts must be positive");
    ensure!(
        !args.verify_only || args.once,
        "--verify-only requires --once"
    );
    loop {
        let audit = cycle(&args)?;
        println!("{}", serde_json::to_string(&audit)?);
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.interval_seconds.min(120)));
    }
}

#[path = "jope_causal_pair_repair/candidate.rs"]
mod candidate;
#[path = "jope_causal_pair_repair/cycle.rs"]
mod cycle;
#[path = "jope_causal_pair_repair/evaluator.rs"]
mod evaluator;
#[path = "jope_causal_pair_repair/jobs.rs"]
mod jobs;
#[path = "jope_causal_pair_repair/repair_spec.rs"]
mod repair_spec;
#[path = "jope_causal_pair_repair/state.rs"]
mod state;
#[cfg(test)]
#[path = "jope_causal_pair_repair/tests.rs"]
mod tests;

use candidate::*;
use cycle::*;
use evaluator::*;
use jobs::*;
use repair_spec::*;
use state::*;
