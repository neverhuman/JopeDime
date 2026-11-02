//! Native Rust port of the durable single-writer self-improvement controller.
//!
//! This binary owns the promotion loop's ORCHESTRATION: the idempotent action
//! journal (`ops/controller.sqlite` + `ops/controller-events.jsonl`), the
//! tournament state machine (`ops/flywheel-state.json`), scheduler
//! interactions, checkpoint fetch (including scheduler-pruned run-dir
//! recovery), gate-proposal spec rendering/submission, the acceptance gate,
//! and the accepted-pointer promotion with ancestor CAS.
//!
//! Three ML data-pipeline phases delegate to the pinned Python modules they
//! have always run in (`test_exact_controller._freeze_corpus`,
//! `_build_packed_corpus`, `_evaluate_frozen_gate`). Promotion ends after the
//! accepted pointer/event is published. Supply is the sole cohort and wave
//! builder; promotion cannot seed the next wave.
//!
//! JSON written by this binary matches CPython's `json.dumps` formatting
//! (sorted keys, separator styles, ensure_ascii, and float `repr` notation)
//! so journal keys and artifacts remain byte-compatible with the Python
//! bridge it replaces.

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_expansion::TASKS;
use jope_dime_regressor::spec_policy::{PROPOSE, TRAINER};
use jope_dime_regressor::task::Task;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[path = "common/flock.rs"]
mod flock;

jope_paths::once_str!(
    fn root_str => paths::flywheel_root()
);
jope_paths::once_str!(
    fn scq_str => paths::scq_binary()
);
jope_paths::once_str!(
    fn runs_root_str => paths::scq_runs_root()
);
jope_paths::once_str!(
    fn scripts_str => paths::research_scripts_root()
);
jope_paths::once_str!(
    fn proposal_selection_policy_str => paths::campaign_manifest("selfplay/proposal-selection-policy.v1.json")
);
jope_paths::once_str!(
    fn prime_src_str => paths::python_src()
);
jope_paths::once_str!(
    fn evaluator_str => paths::corpus_eval_binary()
);
jope_paths::once_str!(
    fn warm_str => paths::legacy_checkpoints_root().join("fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592.model.pt")
);
const WARM_SHA: &str = "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592";
const CATALOG_STRIDE_BYTES: u64 = 2_048 * 4;
const LABEL_THRESHOLD: i64 = 50_000;
const DEFAULT_SEED: i64 = 20260726;

const PHASES: [&str; 9] = [
    "freeze_corpus",
    "build_packed_corpus",
    "launch_checkpoint_tournament",
    "fetch_candidate_checkpoints",
    "generate_gate_proposals",
    "evaluate_frozen_gate",
    "build_acceptance_evidence",
    "run_checkpoint_gate",
    "advance_accepted_pointer",
];

/// Phases delegated to the pinned Python implementations (see module docs).
const PYTHON_PHASES: [&str; 3] = [
    "freeze_corpus",
    "build_packed_corpus",
    "evaluate_frozen_gate",
];

/// The shim inserts the scripts root at run time (`@SCRIPTS@`), see [`py_shim`].
const PY_SHIM_TEMPLATE: &str = r#"
import json, sys
sys.path.insert(0, "@SCRIPTS@")
import test_exact_controller as c
name = sys.argv[1]
if name == "milestone_status":
    from test_exact_sealed_audit import milestone_status
    print(json.dumps(milestone_status(sys.argv[2]), sort_keys=True))
else:
    handler = getattr(c, "_" + name)
    with open(sys.argv[2]) as handle:
        request = json.load(handle)
    try:
        print(json.dumps(handler(request), sort_keys=True))
    except c.PendingAction as error:
        print(json.dumps({"__pending__": str(error)}, sort_keys=True))
"#;

/// [`PY_SHIM_TEMPLATE`] with the scripts root resolved once per process.
fn py_shim() -> &'static str {
    static CELL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CELL.get_or_init(|| PY_SHIM_TEMPLATE.replace("@SCRIPTS@", scripts_str()))
}

// ---------------------------------------------------------------------------
// CPython-compatible JSON serialization
// ---------------------------------------------------------------------------

mod json_compat;
use json_compat::*;
mod io_journal;
use io_journal::*;
mod catalog;
use catalog::*;
mod scheduler;
use scheduler::*;
mod python_handlers;
use python_handlers::*;
mod spec_rendering;
use spec_rendering::*;
mod checkpoint_handlers;
use checkpoint_handlers::*;
mod gate_verdicts;
use gate_verdicts::*;
mod gate_evaluation;
use gate_evaluation::*;
mod checkpoint_promotion;
use checkpoint_promotion::*;
mod promotion_pointer;
use promotion_pointer::*;
mod controller;
use controller::*;

#[derive(Parser)]
#[command(name = "jope-selfplay-promotion")]
#[command(about = "Native checkpoint tournament and promotion controller")]
struct Args {
    #[arg(long, default_value = root_str())]
    root: PathBuf,
    /// Seconds to wait before re-polling a phase that reported `pending`.
    /// A pending phase is a healthy wait on external work, not a failure, so it
    /// is polled in-process instead of being turned into a systemd restart.
    #[arg(long, default_value_t = 120)]
    pending_backoff_seconds: u64,
    /// Exit after a single cycle instead of polling a pending phase.
    #[arg(long)]
    once: bool,
    /// One-shot WS-6 lane: recompute the gate from an existing
    /// acceptance-evidence file, print the per-task verdicts, and advance
    /// `training/accepted-by-task.json`. Never runs a phase, never takes the
    /// controller lock, never touches `training/accepted.json`,
    /// `training/learning.json` or the freshness epoch.
    #[arg(long, value_name = "ACCEPTANCE_EVIDENCE_JSON")]
    task_accept: Option<PathBuf>,
    /// With --task-accept: print the verdicts and leave every pointer alone.
    #[arg(long)]
    dry_run: bool,
    /// With --task-accept: resolve the global lineage pointer from this
    /// checkpoint instead of `accepted_checkpoint()` (tests and rehearsals).
    #[arg(long, value_name = "CHECKPOINT")]
    global_checkpoint: Option<PathBuf>,
}

/// `--task-accept`: the per-task promotion decision, decoupled from the
/// tournament state machine. The gate phase for a watermark is journalled
/// exactly-once, so a controller already past `run_checkpoint_gate` will never
/// re-emit the verdict for that watermark; this path recomputes it from the
/// same evidence with the same code and publishes the branch pointers.
fn run_task_accept(args: &Args) -> Result<Value> {
    let evidence_path = args.task_accept.as_ref().expect("--task-accept");
    let evidence = read_json(evidence_path)?;
    let accepted_by_task = load_accepted_by_task(&task_accepted_pointer(&args.root))?;
    let gate = evaluate_gate(&evidence, accepted_by_task.as_ref())?;
    let (global_path, global_sha) = match &args.global_checkpoint {
        Some(path) => (path.clone(), sha256_file(path)?),
        None => accepted_checkpoint()?,
    };
    let mut out = json!({
        "schema": "JopeSelfplayTaskAcceptResult.v1",
        "evidence": evidence_path.to_string_lossy(),
        "evidence_sha256": sha256_file(evidence_path)?,
        "global_checkpoint_sha256": global_sha,
        "global_promotion_tier": gate["promotion_tier"],
        "global_pass": gate["pass"],
        "task_verdicts": gate["task_verdicts"],
        "tasks_passing": gate["tasks_passing"],
        "dry_run": args.dry_run,
        // Full recomputed gate, so a reviewer can diff it against the artifact
        // the controller already wrote and confirm the global verdict did not
        // move when the per-task block was added.
        "gate": gate.clone(),
    });
    if !args.dry_run {
        out["task_accepted_pointers"] = advance_task_accepted_pointers(
            &args.root,
            &evidence,
            &gate,
            &global_path,
            &global_sha,
        )?;
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.task_accept.is_some() {
        println!("{}", py_dumps(&run_task_accept(&args)?, PyStyle::Pretty)?);
        return Ok(());
    }
    let lock_path = args.root.join("ops/controller.lock");
    fs::create_dir_all(lock_path.parent().unwrap())?;
    let lock = fs::File::create(&lock_path)?;
    if flock_exclusive(&lock, false).is_err() {
        eprintln!("self-play controller is already running");
        std::process::exit(1);
    }
    let controller = Controller::open(&args.root)?;
    // A pending phase used to exit(0) and be restarted by systemd, so every
    // poll of a long external wait was counted as a service restart (1,920 of
    // them on the frozen gate alone) and every poll paid a fresh process +
    // snapshot cost. Polling in-process keeps the controller flock held for the
    // whole wait - which is exactly the invariant we want, since no other phase
    // can run while one is pending - and leaves NRestarts meaning "the
    // controller actually stopped", not "the controller looked again". Each
    // poll still journals its own pending detail, so nothing is hidden.
    loop {
        let snapshot = live_snapshot(&args.root)?;
        let result = controller.cycle(&snapshot)?;
        println!("{}", py_dumps(&result, PyStyle::Default)?);
        if args.once || result.get("pending") != Some(&Value::Bool(true)) {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_secs(args.pending_backoff_seconds));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
