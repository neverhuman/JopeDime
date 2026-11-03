//! Rust checkpoint-freshness authority for the internal self-play queue.
//!
//! The process is deliberately single-writer.  It owns one filesystem lock,
//! persists receipt/idempotency state with atomic renames, and exposes only the
//! bounded freshness protocol over a local Unix socket.  SQLite campaign state
//! is audited and updated through the native `sqlite3` executable because this
//! crate intentionally has no SQLite dependency.

use anyhow::{bail, Context, Result};
use clap::Parser;
use rusqlite::{
    params, types::ValueRef, Connection, OpenFlags, OptionalExtension, TransactionBehavior,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[path = "../freshness_protocol.rs"]
#[allow(dead_code)]
mod freshness_protocol;

use freshness_protocol::TOOL_NAMES;
use jope_dime_regressor::paths;

jope_paths::once_str!(
    fn default_root_str => paths::flywheel_root()
);
jope_paths::once_str!(
    fn default_scq_str => paths::scq_binary()
);
const PREVIEW_MAX_AGE_MS: i64 = 5 * 60_000;
const FIRST_JOB_DEADLINE_MS: i64 = 5 * 60_000;
const USEFUL_QUEUE_STATES: [&str; 2] = ["queued", "starting"];
const NON_INTERRUPTIBLE_STATES: [&str; 3] = ["dispatching", "running", "preempting"];
const RETIRABLE_PROJECTS: [&str; 2] = ["jope-selfplay-eval", "jope-testonly"];

#[derive(Parser)]
#[command(name = "jope-selfplay-freshness")]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long)]
    socket: Option<PathBuf>,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    #[arg(long, default_value_t = 5.0)]
    interval: f64,
    #[arg(long)]
    once: bool,
    #[arg(long)]
    audit_only: bool,
    /// Automatically apply the content-addressed stale retirement preview.
    #[arg(long)]
    auto_retire_stale: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct Preview {
    schema: String,
    kind: String,
    created_at_ms: i64,
    payload: Value,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct Mutation {
    tool: String,
    preview_sha256: String,
    result: Value,
    created_at_ms: i64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct DurableState {
    schema: String,
    active_epoch: String,
    active_checkpoint: String,
    admission_paused: bool,
    checkpoint_activated_at_ms: BTreeMap<String, i64>,
    checkpoint_superseded_at_ms: BTreeMap<String, i64>,
    previews: BTreeMap<String, Preview>,
    mutations: BTreeMap<String, Mutation>,
}

impl DurableState {
    fn new() -> Self {
        Self {
            schema: "JopeFreshnessState.v1".to_owned(),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug)]
struct InventoryJob {
    job_id: String,
    state: String,
    epoch: String,
    manifest: PathBuf,
    manifest_sha256: String,
    claim_state: String,
    reservation_owner: Option<String>,
    scheduler: SchedulerJob,
    outside_scq: bool,
}

#[derive(Clone, Debug)]
struct BacklogClaim {
    manifest: PathBuf,
    manifest_sha256: String,
    state: String,
    reservation_owner: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ManifestFingerprint {
    bytes: u64,
    device: u64,
    inode: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[derive(Clone, Debug)]
struct CachedManifestAudit {
    expected_sha256: String,
    fingerprint: ManifestFingerprint,
    freshness_scoped: bool,
    epoch: String,
}

impl InventoryJob {
    fn as_value(&self) -> Value {
        let mut object = self
            .scheduler
            .as_value()
            .as_object()
            .cloned()
            .unwrap_or_default();
        object.insert("job_id".to_owned(), json!(self.job_id));
        object.insert("state".to_owned(), json!(self.state));
        object.insert("epoch".to_owned(), json!(self.epoch));
        object.insert(
            "manifest".to_owned(),
            json!(self.manifest.to_string_lossy()),
        );
        object.insert("manifest_sha256".to_owned(), json!(self.manifest_sha256));
        object.insert("claim_state".to_owned(), json!(self.claim_state));
        object.insert(
            "reservation_owner".to_owned(),
            self.reservation_owner
                .as_ref()
                .map_or(Value::Null, |owner| json!(owner)),
        );
        if self.outside_scq {
            object.insert("outside_scq".to_owned(), json!(true));
        }
        Value::Object(object)
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct SchedulerJob {
    id: String,
    state: String,
    project: Option<String>,
    node: Option<String>,
    queue_ts: Option<String>,
    submit_ts: Option<String>,
    created_at_ms: Option<i64>,
    submitted_at_ms: Option<i64>,
    attempt: Option<u64>,
    run_dir: Option<String>,
}

impl SchedulerJob {
    fn as_value(&self) -> Value {
        json!({
            "id": self.id,
            "state": self.state,
            "project": self.project,
            "node": self.node,
            "queue_ts": self.queue_ts,
            "submit_ts": self.submit_ts,
            "created_at_ms": self.created_at_ms,
            "submitted_at_ms": self.submitted_at_ms,
            "attempt": self.attempt,
            "run_dir": self.run_dir,
        })
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct SchedulerJobsInner {
    jobs: Vec<SchedulerJob>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct SchedulerJobsEnvelope {
    #[serde(rename = "Jobs")]
    upper: SchedulerJobsInner,
    jobs: Vec<SchedulerJob>,
}

#[derive(Clone, Debug)]
struct ScqScheduler {
    binary: PathBuf,
}

trait Scheduler {
    fn jobs(&self) -> Result<Vec<SchedulerJob>>;
    fn hold(&self, job_id: &str) -> Result<()>;
    fn cancel(&self, job_id: &str) -> Result<()>;
}

impl ScqScheduler {
    fn run(&self, args: &[&str]) -> Result<Value> {
        let output = Command::new(&self.binary)
            .arg("--json")
            .args(args)
            .output()
            .with_context(|| format!("run {}", self.binary.display()))?;
        if !output.status.success() {
            bail!(
                "SCQ {:?} failed: {}",
                args,
                String::from_utf8_lossy(if output.stderr.is_empty() {
                    &output.stdout
                } else {
                    &output.stderr
                })
                .trim()
            );
        }
        let text = String::from_utf8(output.stdout)?.trim().to_owned();
        if text.is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_str(&text).with_context(|| format!("parse SCQ JSON for {args:?}"))
    }

    fn load_jobs(&self) -> Result<Vec<SchedulerJob>> {
        let output = Command::new(&self.binary)
            .args(["--json", "ls"])
            .output()
            .with_context(|| format!("run {} --json ls", self.binary.display()))?;
        if !output.status.success() {
            bail!(
                "SCQ ls failed: {}",
                String::from_utf8_lossy(if output.stderr.is_empty() {
                    &output.stdout
                } else {
                    &output.stderr
                })
                .trim()
            );
        }
        parse_scheduler_jobs(&output.stdout)
    }

    fn hold_job(&self, job_id: &str) -> Result<()> {
        self.run(&["hold", job_id])?;
        Ok(())
    }

    fn cancel_job(&self, job_id: &str) -> Result<()> {
        self.run(&["cancel", job_id])?;
        Ok(())
    }
}

impl Scheduler for ScqScheduler {
    fn jobs(&self) -> Result<Vec<SchedulerJob>> {
        self.load_jobs()
    }

    fn hold(&self, job_id: &str) -> Result<()> {
        self.hold_job(job_id)
    }

    fn cancel(&self, job_id: &str) -> Result<()> {
        self.cancel_job(job_id)
    }
}

fn parse_scheduler_jobs(bytes: &[u8]) -> Result<Vec<SchedulerJob>> {
    let mut decoded: SchedulerJobsEnvelope =
        serde_json::from_slice(bytes).context("parse minimal SCQ job inventory")?;
    if decoded.upper.jobs.is_empty() {
        Ok(decoded.jobs)
    } else {
        Ok(std::mem::take(&mut decoded.upper.jobs))
    }
}

struct Supervisor {
    root: PathBuf,
    state_path: PathBuf,
    events_path: PathBuf,
    backlog: PathBuf,
    pair_registry: PathBuf,
    scheduler: Box<dyn Scheduler>,
    state: DurableState,
    verified_checkpoints: BTreeSet<String>,
    manifest_cache: RefCell<BTreeMap<PathBuf, CachedManifestAudit>>,
}

fn main() -> Result<()> {
    freshness_impl::run()
}

#[path = "jope_selfplay_freshness/freshness_helpers.rs"]
mod freshness_helpers;
#[path = "jope_selfplay_freshness/freshness_impl/mod.rs"]
mod freshness_impl;
#[path = "jope_selfplay_freshness/freshness_sql.rs"]
mod freshness_sql;

use freshness_helpers::*;
use freshness_sql::*;
