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
        Ok(serde_json::from_str(&text).unwrap_or_else(|_| json!({"output": text})))
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

impl Supervisor {
    fn open(root: PathBuf, scq: PathBuf) -> Result<Self> {
        let ops = root.join("ops");
        fs::create_dir_all(&ops)?;
        let state_path = ops.join("freshness-state.v1.json");
        let state = if state_path.is_file() {
            let parsed: DurableState = serde_json::from_slice(&fs::read(&state_path)?)
                .with_context(|| format!("parse {}", state_path.display()))?;
            if parsed.schema != "JopeFreshnessState.v1" {
                bail!(
                    "{}: unsupported freshness state schema",
                    state_path.display()
                );
            }
            parsed
        } else {
            import_legacy_state(&ops.join("freshness.sqlite"))?
        };
        Ok(Self {
            backlog: ops.join("backlog.sqlite"),
            pair_registry: ops.join("pair-registry.sqlite"),
            events_path: ops.join("freshness-events.v1.jsonl"),
            state_path,
            scheduler: Box::new(ScqScheduler { binary: scq }),
            state,
            verified_checkpoints: BTreeSet::new(),
            manifest_cache: RefCell::new(BTreeMap::new()),
            root,
        })
    }

    fn save_state(&self) -> Result<()> {
        atomic_json(&self.state_path, &serde_json::to_value(&self.state)?)
    }

    fn event(
        &self,
        kind: &str,
        epoch: Option<&str>,
        job_id: Option<&str>,
        detail: Value,
    ) -> Result<()> {
        let event = json!({
            "schema": "JopeFreshnessAuditEvent.v1",
            "kind": kind,
            "epoch": epoch,
            "job_id": job_id,
            "detail": detail,
            "created_at_ms": now_ms()
        });
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(&self.events_path)?;
        // One write per record: an O_APPEND line lands whole instead of token by token.
        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');
        file.write_all(&line)?;
        file.sync_data()?;
        Ok(())
    }

    fn pointer_checkpoint(&mut self) -> Result<Option<String>> {
        for relative in ["training/learning.json", "training/accepted.json"] {
            let path = self.root.join(relative);
            if !path.is_file() {
                continue;
            }
            let pointer: Value =
                serde_json::from_slice(&fs::read(&path)?).with_context(|| relative.to_owned())?;
            let digest = pointer
                .get("checkpoint_sha256")
                .and_then(Value::as_str)
                .unwrap_or("");
            if !is_sha256(digest) {
                continue;
            }
            if let Some(checkpoint) = pointer.get("checkpoint").and_then(Value::as_str) {
                let checkpoint = Path::new(checkpoint);
                if checkpoint.is_file() && !self.verified_checkpoints.contains(digest) {
                    if sha256_file(checkpoint)? != digest {
                        bail!("{relative}: checkpoint pointer hash drift");
                    }
                    self.verified_checkpoints.insert(digest.to_owned());
                }
            }
            return Ok(Some(digest.to_owned()));
        }
        let proposal_pointer = self.root.join("proposals/current.json");
        if proposal_pointer.is_file() {
            let value: Value = serde_json::from_slice(&fs::read(&proposal_pointer)?)?;
            if value.get("legacy_adopted").and_then(Value::as_bool) == Some(true) {
                let digest = value
                    .get("checkpoint_sha256")
                    .and_then(Value::as_str)
                    .context("legacy proposal pointer lacks checkpoint_sha256")?;
                if !is_sha256(digest) {
                    bail!("legacy proposal checkpoint hash is invalid");
                }
                let checkpoint =
                    paths::legacy_checkpoints_root().join(format!("{digest}.model.pt"));
                if !checkpoint.is_file() {
                    bail!("legacy-adopted checkpoint artifact is missing");
                }
                if !self.verified_checkpoints.contains(digest)
                    && sha256_file(&checkpoint)? != digest
                {
                    bail!("legacy-adopted checkpoint artifact failed verification");
                }
                self.verified_checkpoints.insert(digest.to_owned());
                return Ok(Some(digest.to_owned()));
            }
        }
        Ok(None)
    }

    fn sync_checkpoint(&mut self) -> Result<Value> {
        let Some(checkpoint) = self.pointer_checkpoint()? else {
            return Ok(self.active_checkpoint());
        };
        let epoch = epoch_for_checkpoint(&checkpoint)?;
        if epoch == self.state.active_epoch {
            return Ok(self.active_checkpoint());
        }
        let timestamp = now_ms();
        let previous = self.state.active_epoch.clone();
        if !previous.is_empty() {
            self.state
                .checkpoint_superseded_at_ms
                .entry(previous.clone())
                .or_insert(timestamp);
        }
        self.state.active_epoch = epoch.clone();
        self.state.active_checkpoint = checkpoint;
        self.state
            .checkpoint_activated_at_ms
            .entry(epoch.clone())
            .or_insert(timestamp);
        self.event(
            "checkpoint_advanced",
            Some(&epoch),
            None,
            json!({"previous_epoch": null_if_empty(&previous)}),
        )?;
        self.save_state()?;
        Ok(self.active_checkpoint())
    }

    fn claims(&self) -> Result<BTreeMap<String, BacklogClaim>> {
        if !self.backlog.is_file() {
            return Ok(BTreeMap::new());
        }
        let connection =
            Connection::open_with_flags(&self.backlog, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .with_context(|| format!("open {}", self.backlog.display()))?;
        connection.busy_timeout(Duration::from_secs(60))?;
        let owner_expression = if sqlite_column_exists(&connection, "claims", "reservation_owner")?
        {
            "reservation_owner"
        } else {
            "NULL"
        };
        let mut statement = connection.prepare(&format!(
            "SELECT job_id,manifest,manifest_sha256,state,{owner_expression} FROM claims"
        ))?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                BacklogClaim {
                    manifest: PathBuf::from(row.get::<_, String>(1)?),
                    manifest_sha256: row.get(2)?,
                    state: row.get(3)?,
                    reservation_owner: row.get(4)?,
                },
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<BTreeMap<_, _>>>()?)
    }

    fn inventory(&self) -> Result<Vec<InventoryJob>> {
        self.inventory_with_manifest_policy(false)
    }

    fn inventory_with_manifest_policy(
        &self,
        force_manifest_audit: bool,
    ) -> Result<Vec<InventoryJob>> {
        let claims = self.claims()?;
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        for scheduler in self.scheduler.jobs()? {
            let job_id = scheduler.id.clone();
            let Some(claim) = claims.get(&job_id) else {
                continue;
            };
            if let Some(job) = self.inventory_job(scheduler, claim, false, force_manifest_audit)? {
                seen.insert(job_id);
                result.push(job);
            }
        }
        for (job_id, claim) in claims {
            if seen.contains(&job_id) || claim.state != "checked" {
                continue;
            }
            if let Some(job) = self.inventory_job(
                SchedulerJob {
                    id: job_id,
                    state: "checked".to_owned(),
                    ..SchedulerJob::default()
                },
                &claim,
                true,
                force_manifest_audit,
            )? {
                result.push(job);
            }
        }
        Ok(result)
    }

    fn inventory_job(
        &self,
        scheduler: SchedulerJob,
        claim: &BacklogClaim,
        outside_scq: bool,
        force_manifest_audit: bool,
    ) -> Result<Option<InventoryJob>> {
        let job_id = scheduler.id.clone();
        if claim.manifest.as_os_str().is_empty() {
            return Ok(None);
        }
        let manifest = claim.manifest.clone();
        if !manifest.is_file() {
            return Ok(None);
        }
        let (freshness_scoped, epoch) = self.audit_manifest(
            &job_id,
            &manifest,
            &claim.manifest_sha256,
            force_manifest_audit,
        )?;
        if !freshness_scoped {
            return Ok(None);
        }
        Ok(Some(InventoryJob {
            state: scheduler.state.clone(),
            epoch,
            manifest,
            manifest_sha256: claim.manifest_sha256.clone(),
            claim_state: claim.state.clone(),
            reservation_owner: claim.reservation_owner.clone(),
            scheduler,
            outside_scq,
            job_id,
        }))
    }

    fn audit_manifest(
        &self,
        job_id: &str,
        manifest: &Path,
        expected_sha256: &str,
        force: bool,
    ) -> Result<(bool, String)> {
        let metadata = manifest
            .metadata()
            .with_context(|| format!("stat {}", manifest.display()))?;
        let fingerprint = ManifestFingerprint {
            bytes: metadata.len(),
            device: metadata.dev(),
            inode: metadata.ino(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        };
        if !force {
            if let Some(cached) = self.manifest_cache.borrow().get(manifest) {
                if cached.expected_sha256 == expected_sha256 && cached.fingerprint == fingerprint {
                    return Ok((cached.freshness_scoped, cached.epoch.clone()));
                }
            }
        }
        let observed_sha256 = sha256_file(manifest)?;
        if expected_sha256 != observed_sha256 {
            bail!("{job_id}: backlog manifest hash drift");
        }
        let parsed: Value = serde_json::from_slice(&fs::read(manifest)?)
            .with_context(|| format!("parse {}", manifest.display()))?;
        let freshness_scoped = freshness_scoped_manifest(&parsed);
        let epoch = if freshness_scoped {
            manifest_epoch(&parsed)
        } else {
            String::new()
        };
        self.manifest_cache.borrow_mut().insert(
            manifest.to_path_buf(),
            CachedManifestAudit {
                expected_sha256: expected_sha256.to_owned(),
                fingerprint,
                freshness_scoped,
                epoch: epoch.clone(),
            },
        );
        Ok((freshness_scoped, epoch))
    }

    fn active_checkpoint(&self) -> Value {
        let epoch = null_if_empty(&self.state.active_epoch);
        let checkpoint = null_if_empty(&self.state.active_checkpoint);
        let activated = self
            .state
            .checkpoint_activated_at_ms
            .get(&self.state.active_epoch)
            .copied();
        json!({
            "checkpoint_epoch": epoch,
            "checkpoint_sha256": checkpoint,
            "activated_at_ms": activated,
            "admission_paused": self.state.admission_paused
        })
    }

    fn queue_epochs(&self) -> Result<Value> {
        let inventory = self.inventory()?;
        Ok(self.queue_epochs_from_inventory(&inventory))
    }

    fn queue_epochs_from_inventory(&self, inventory: &[InventoryJob]) -> Value {
        let mut counts: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
        for job in inventory {
            *counts
                .entry(job.epoch.clone())
                .or_default()
                .entry(job.state.clone())
                .or_default() += 1;
        }
        let epochs = counts
            .into_iter()
            .map(|(epoch, states)| {
                let useful_depth = USEFUL_QUEUE_STATES
                    .iter()
                    .map(|state| states.get(*state).copied().unwrap_or(0))
                    .sum::<u64>();
                json!({
                    "epoch": epoch,
                    "active": epoch == self.state.active_epoch,
                    "states": states,
                    "useful_depth": useful_depth,
                    "held": states.get("held").copied().unwrap_or(0),
                    "running": states.get("running").copied().unwrap_or(0),
                    "dispatching": states.get("dispatching").copied().unwrap_or(0)
                })
            })
            .collect::<Vec<_>>();
        json!({
            "active_epoch": null_if_empty(&self.state.active_epoch),
            "epochs": epochs
        })
    }

    fn freshness_status(&self) -> Result<Value> {
        let inventory = self.inventory()?;
        Ok(self.freshness_status_from_inventory(&inventory))
    }

    fn freshness_status_from_inventory(&self, inventory: &[InventoryJob]) -> Value {
        let queue = self.queue_epochs_from_inventory(inventory);
        let epochs = queue["epochs"].as_array().cloned().unwrap_or_default();
        let stale = epochs
            .iter()
            .filter(|row| row.get("active").and_then(Value::as_bool) != Some(true))
            .collect::<Vec<_>>();
        let activated = self
            .state
            .checkpoint_activated_at_ms
            .get(&self.state.active_epoch)
            .copied();
        let mut first_running = None;
        for job in inventory {
            if job.epoch != self.state.active_epoch || job.state != "running" {
                continue;
            }
            let observed = job
                .scheduler
                .created_at_ms
                .or(job.scheduler.submitted_at_ms)
                .unwrap_or(0);
            first_running = Some(first_running.map_or(observed, |prior: i64| prior.min(observed)));
        }
        let latency = activated
            .zip(first_running)
            .map(|(activated, running)| (running - activated).max(0));
        let stale_useful = stale
            .iter()
            .filter_map(|row| row.get("useful_depth").and_then(Value::as_u64))
            .sum::<u64>();
        let stale_held = stale
            .iter()
            .filter_map(|row| row.get("held").and_then(Value::as_u64))
            .sum::<u64>();
        let mut status = self.active_checkpoint();
        let object = status.as_object_mut().expect("active checkpoint is object");
        object.extend(Map::from_iter([
            ("schema".to_owned(), json!("JopeSelfplayFreshnessStatus.v1")),
            ("queue_epochs".to_owned(), json!(epochs)),
            ("stale_epoch_count".to_owned(), json!(stale.len())),
            ("stale_useful_depth".to_owned(), json!(stale_useful)),
            ("stale_held".to_owned(), json!(stale_held)),
            (
                "checkpoint_to_first_job_latency_ms".to_owned(),
                json!(latency),
            ),
            (
                "first_job_deadline_ms".to_owned(),
                json!(FIRST_JOB_DEADLINE_MS),
            ),
            (
                "healthy".to_owned(),
                json!(latency.is_none_or(|value| value <= FIRST_JOB_DEADLINE_MS)),
            ),
        ]));
        status
    }

    fn explain_job(&self, wanted: &str) -> Result<Value> {
        let job = self
            .inventory()?
            .into_iter()
            .find(|job| job.job_id == wanted)
            .with_context(|| format!("unknown job {wanted}"))?;
        let pairs = if self.pair_registry.is_file() {
            sqlite_rows(
                &self.pair_registry,
                &format!(
                    "SELECT * FROM pairs WHERE manifest_sha256={} \
                     ORDER BY task,dataset_hash,semantic_sha256",
                    sql_quote(&job.manifest_sha256)
                ),
            )?
        } else {
            Vec::new()
        };
        Ok(json!({
            "job": job.as_value(),
            "active_epoch": null_if_empty(&self.state.active_epoch),
            "pair_count": pairs.len(),
            "pairs": pairs
        }))
    }

    fn write_preview(&mut self, kind: &str, payload: Value) -> Result<Value> {
        self.prune_state();
        let timestamp = now_ms();
        if let Some((digest, preview)) = self
            .state
            .previews
            .iter()
            .find(|(_, preview)| {
                preview.kind == kind
                    && preview.payload == payload
                    && timestamp - preview.created_at_ms <= PREVIEW_MAX_AGE_MS
            })
            .map(|(digest, preview)| (digest.clone(), preview.clone()))
        {
            let mut result = serde_json::to_value(preview)?;
            result
                .as_object_mut()
                .expect("preview is object")
                .insert("preview_receipt_sha256".to_owned(), json!(digest));
            return Ok(result);
        }
        let preview = Preview {
            schema: "JopeFreshnessPreview.v1".to_owned(),
            kind: kind.to_owned(),
            created_at_ms: timestamp,
            payload,
        };
        let encoded = serde_json::to_value(&preview)?;
        let digest = content_hash(&encoded)?;
        self.state.previews.entry(digest.clone()).or_insert(preview);
        self.save_state()?;
        let mut result = encoded;
        result
            .as_object_mut()
            .expect("preview is object")
            .insert("preview_receipt_sha256".to_owned(), json!(digest));
        Ok(result)
    }

    fn preview_stale_retirement(&mut self) -> Result<Value> {
        let inventory = self.inventory()?;
        self.preview_stale_retirement_from_inventory(&inventory)
    }

    fn preview_stale_retirement_from_inventory(
        &mut self,
        inventory: &[InventoryJob],
    ) -> Result<Value> {
        let mut actions = Vec::new();
        let mut preserved = Vec::new();
        for job in inventory {
            if job.epoch == self.state.active_epoch {
                continue;
            }
            if !retirement_scoped_job(job) {
                preserved.push(json!({
                    "job_id": job.job_id,
                    "state": job.state,
                    "reason": "protected_project"
                }));
                continue;
            }
            if NON_INTERRUPTIBLE_STATES.contains(&job.state.as_str()) {
                preserved.push(json!({
                    "job_id": job.job_id,
                    "state": job.state,
                    "reason": "started"
                }));
                continue;
            }
            if !matches!(
                job.state.as_str(),
                "queued" | "starting" | "held" | "cancelled" | "checked"
            ) {
                continue;
            }
            let expected_claim = if job.state == "checked" {
                "checked"
            } else {
                "submitted"
            };
            if job.claim_state != expected_claim {
                preserved.push(json!({
                    "job_id": job.job_id,
                    "state": job.state,
                    "reason": "claim_not_confirmed_unstarted"
                }));
                continue;
            }
            let action = match job.state.as_str() {
                "cancelled" => "finalize_cancelled",
                "checked" => "retire_checked_unstarted",
                _ => "cancel_after_hold",
            };
            actions.push(json!({
                "job_id": job.job_id,
                "epoch": job.epoch,
                "observed_state": job.state,
                "action": action,
                "manifest": job.manifest,
                "manifest_sha256": job.manifest_sha256,
                "claim_state": job.claim_state
            }));
        }
        actions.sort_by(|left, right| left["job_id"].as_str().cmp(&right["job_id"].as_str()));
        preserved.sort_by(|left, right| left["job_id"].as_str().cmp(&right["job_id"].as_str()));
        self.write_preview(
            "stale_retirement",
            json!({
                "active_epoch": null_if_empty(&self.state.active_epoch),
                "actions": actions,
                "preserved_started": preserved
            }),
        )
    }

    fn load_preview(&self, digest: &str, expected_kind: Option<&str>) -> Result<Preview> {
        if !is_sha256(digest) {
            bail!("preview receipt must be a SHA-256 value");
        }
        let preview = self
            .state
            .previews
            .get(digest)
            .cloned()
            .context("unknown preview receipt")?;
        if expected_kind.is_some_and(|kind| kind != preview.kind) {
            bail!("preview receipt kind mismatch");
        }
        if now_ms() - preview.created_at_ms > PREVIEW_MAX_AGE_MS {
            bail!("preview receipt expired");
        }
        Ok(preview)
    }

    fn prior_mutation(
        &self,
        tool: &str,
        preview_sha256: &str,
        idempotency_key: &str,
    ) -> Result<Option<Value>> {
        if idempotency_key.trim().is_empty() {
            bail!("mutating freshness tools require an idempotency key");
        }
        if let Some(prior) = self.state.mutations.get(idempotency_key) {
            if prior.tool != tool || prior.preview_sha256 != preview_sha256 {
                bail!("idempotency key was used for a different mutation");
            }
            return Ok(Some(prior.result.clone()));
        }
        self.load_preview(preview_sha256, None)?;
        Ok(None)
    }

    fn finish_mutation(
        &mut self,
        tool: &str,
        preview_sha256: &str,
        idempotency_key: &str,
        result: Value,
    ) -> Result<Value> {
        self.state.mutations.insert(
            idempotency_key.to_owned(),
            Mutation {
                tool: tool.to_owned(),
                preview_sha256: preview_sha256.to_owned(),
                result: result.clone(),
                created_at_ms: now_ms(),
            },
        );
        self.save_state()?;
        Ok(result)
    }

    fn apply_stale_retirement(
        &mut self,
        preview_sha256: &str,
        idempotency_key: &str,
    ) -> Result<Value> {
        if let Some(prior) =
            self.prior_mutation("apply_stale_retirement", preview_sha256, idempotency_key)?
        {
            return Ok(prior);
        }
        let preview = self.load_preview(preview_sha256, Some("stale_retirement"))?;
        let actions = preview
            .payload
            .get("actions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut inventory = self
            .inventory_with_manifest_policy(true)?
            .into_iter()
            .map(|job| (job.job_id.clone(), job))
            .collect::<BTreeMap<_, _>>();
        let mut held = 0_u64;
        let mut retired = 0_u64;
        let mut skipped = Vec::new();

        for action in &actions {
            let Some(job_id) = action.get("job_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(current) = inventory.get(job_id) else {
                skipped.push(json!({"job_id": job_id, "reason": "not_active"}));
                continue;
            };
            if !retirement_scoped_job(current)
                || current.epoch == self.state.active_epoch
                || NON_INTERRUPTIBLE_STATES.contains(&current.state.as_str())
            {
                skipped.push(json!({"job_id": job_id, "reason": "started_or_current"}));
                continue;
            }
            if USEFUL_QUEUE_STATES.contains(&current.state.as_str()) {
                self.scheduler.hold(job_id)?;
                self.event(
                    "stale_job_held",
                    Some(&current.epoch),
                    Some(job_id),
                    json!({"automatic": false}),
                )?;
                held += 1;
            }
        }

        inventory = self
            .inventory_with_manifest_policy(true)?
            .into_iter()
            .map(|job| (job.job_id.clone(), job))
            .collect();
        for action in &actions {
            let Some(job_id) = action.get("job_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(current) = inventory.get(job_id) else {
                continue;
            };
            if !retirement_scoped_job(current) {
                skipped.push(json!({"job_id": job_id, "reason": "protected_project"}));
                continue;
            }
            if !matches!(current.state.as_str(), "held" | "cancelled" | "checked") {
                continue;
            }
            let expected_claim = if current.state == "checked" {
                "checked"
            } else {
                "submitted"
            };
            if current.epoch == self.state.active_epoch
                || current.manifest_sha256.is_empty()
                || current.claim_state != expected_claim
            {
                skipped.push(json!({"job_id": job_id, "reason": "provenance_audit_failed"}));
                continue;
            }
            let states = self.manifest_state_counts(&current.manifest_sha256)?;
            if states.keys().all(|state| state == "available") && !states.is_empty() {
                self.mark_claim_superseded(job_id, &current.manifest_sha256)?;
                continue;
            }
            let expected_pair_state = if current.state == "checked" {
                "reserved"
            } else {
                "submitted"
            };
            if states.len() != 1 || !states.contains_key(expected_pair_state) {
                skipped.push(json!({"job_id": job_id, "reason": "pair_state_audit_failed"}));
                continue;
            }
            let retirement = content_hash(&json!({
                "preview": preview_sha256,
                "job_id": job_id,
                "manifest_sha256": current.manifest_sha256
            }))?;

            // This second scheduler read is the hard guard against racing a
            // dispatch.  Only held/cancelled/checked jobs can cross it.
            let latest = self
                .inventory()?
                .into_iter()
                .find(|job| job.job_id == job_id);
            let Some(latest) = latest else {
                skipped.push(json!({"job_id": job_id, "reason": "state_changed_before_cancel"}));
                continue;
            };
            if !retirement_scoped_job(&latest)
                || !matches!(latest.state.as_str(), "held" | "cancelled" | "checked")
                || latest.epoch == self.state.active_epoch
            {
                skipped.push(json!({"job_id": job_id, "reason": "state_changed_before_cancel"}));
                continue;
            }
            if latest.state == "held" {
                self.scheduler.cancel(job_id)?;
            }
            let count = self.supersede_pairs(
                &current.manifest_sha256,
                job_id,
                expected_pair_state,
                &retirement,
            )?;
            if count == 0 {
                bail!("{job_id}: stale retirement selected no protocol pairs");
            }
            self.mark_claim_superseded(job_id, &current.manifest_sha256)?;
            self.event(
                "stale_job_retired",
                Some(&current.epoch),
                Some(job_id),
                json!({"retirement_receipt_sha256": retirement, "pairs": count}),
            )?;
            retired += 1;
        }
        let result = json!({
            "preview_receipt_sha256": preview_sha256,
            "held": held,
            "retired": retired,
            "skipped": skipped,
            "running_interrupted": 0
        });
        self.finish_mutation(
            "apply_stale_retirement",
            preview_sha256,
            idempotency_key,
            result,
        )
    }

    fn manifest_state_counts(&self, manifest_sha256: &str) -> Result<BTreeMap<String, u64>> {
        if !self.pair_registry.is_file() {
            bail!("pair registry is missing");
        }
        Ok(sqlite_rows(
            &self.pair_registry,
            &format!(
                "SELECT state,COUNT(*) AS count FROM pairs \
                 WHERE manifest_sha256={} GROUP BY state",
                sql_quote(manifest_sha256)
            ),
        )?
        .into_iter()
        .filter_map(|row| {
            Some((
                row.get("state")?.as_str()?.to_owned(),
                row.get("count")?.as_u64()?,
            ))
        })
        .collect())
    }

    fn supersede_pairs(
        &self,
        manifest_sha256: &str,
        job_id: &str,
        expected_state: &str,
        retirement_sha256: &str,
    ) -> Result<u64> {
        if !matches!(expected_state, "reserved" | "submitted") {
            bail!("unsupported retirement pair state");
        }
        if !is_sha256(retirement_sha256) {
            bail!("retirement receipt is not a SHA-256 value");
        }
        let mut connection = open_sqlite(&self.pair_registry)?;
        let owner_column = if sqlite_column_exists(&connection, "pairs", "reservation_owner")? {
            "reservation_owner"
        } else if sqlite_column_exists(&connection, "pairs", "owner")? {
            "owner"
        } else {
            bail!("pair registry has no ownership column");
        };
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let pair_query = if expected_state == "submitted" {
            "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,state
             FROM pairs WHERE manifest_sha256=?1 AND state='submitted' AND job_id=?2"
        } else {
            "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,state
             FROM pairs WHERE manifest_sha256=?1 AND state='reserved' AND job_id IS NULL"
        };
        let mut statement = transaction.prepare(pair_query)?;
        let retiring = statement
            .query_map(params![manifest_sha256, job_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(statement);
        if retiring.is_empty() {
            let prior: i64 = transaction.query_row(
                "SELECT COUNT(*) FROM attempts WHERE manifest_sha256=?1 AND job_id=?2
                   AND state='superseded_unstarted' AND receipt_sha256=?3",
                params![manifest_sha256, job_id, retirement_sha256],
                |row| row.get(0),
            )?;
            if prior == 0 {
                bail!("stale retirement selected no protocol pairs");
            }
            transaction.commit()?;
            return Ok(prior as u64);
        }
        let timestamp = now_ms();
        let detail = serde_json::to_string(&json!({
            "manifest_sha256": manifest_sha256,
            "job_id": job_id,
            "retirement_receipt_sha256": retirement_sha256
        }))?;
        for (task, dataset, semantic, protocol, state) in &retiring {
            let active = transaction
                .query_row(
                    "SELECT id,state,job_id FROM attempts
                     WHERE task=?1 AND dataset_hash=?2 AND semantic_sha256=?3
                       AND protocol_sha256=?4
                       AND state IN ('reserved','submitted') AND finalized_at_ms IS NULL",
                    params![task, dataset, semantic, protocol],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                        ))
                    },
                )
                .optional()?
                .context("stale retirement active-attempt audit failed")?;
            if active.1 != *state || (state == "submitted" && active.2.as_deref() != Some(job_id)) {
                bail!("stale retirement active-attempt audit failed");
            }
            if transaction.execute(
                "UPDATE attempts SET state='superseded_unstarted',
                   receipt_sha256=?1,job_id=COALESCE(job_id,?2),finalized_at_ms=?3
                 WHERE id=?4 AND state=?5 AND finalized_at_ms IS NULL",
                params![retirement_sha256, job_id, timestamp, active.0, state],
            )? != 1
            {
                bail!("stale retirement lost active attempt");
            }
            transaction.execute(
                "INSERT INTO events(
                   kind,task,dataset_hash,semantic_sha256,protocol_sha256,
                   detail_json,created_at_ms
                 ) VALUES('superseded_unstarted',?1,?2,?3,?4,?5,?6)",
                params![task, dataset, semantic, protocol, detail, timestamp],
            )?;
            let update = format!(
                "UPDATE pairs SET state='available',{owner_column}='available',
                   manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,
                   receipt_sha256=NULL,failure_class=NULL,updated_at_ms=?1
                 WHERE task=?2 AND dataset_hash=?3 AND semantic_sha256=?4
                   AND protocol_sha256=?5 AND state=?6"
            );
            if transaction.execute(
                &update,
                params![timestamp, task, dataset, semantic, protocol, state],
            )? != 1
            {
                bail!("stale retirement lost pair ownership");
            }
        }
        transaction.commit()?;
        Ok(retiring.len() as u64)
    }

    fn mark_claim_superseded(&self, job_id: &str, manifest_sha256: &str) -> Result<()> {
        if !self.backlog.is_file() {
            return Ok(());
        }
        sqlite_exec(
            &self.backlog,
            &format!(
                "BEGIN IMMEDIATE;\
                 UPDATE claims SET state='superseded_unstarted',updated_at_ms={} \
                 WHERE job_id={} AND manifest_sha256={} \
                 AND state IN ('submitted','checked');\
                 COMMIT;",
                now_ms(),
                sql_quote(job_id),
                sql_quote(manifest_sha256)
            ),
        )
    }

    fn admission_mutation(
        &mut self,
        paused: bool,
        preview_sha256: &str,
        idempotency_key: &str,
    ) -> Result<Value> {
        let tool = if paused {
            "pause_admission"
        } else {
            "resume_admission"
        };
        if let Some(prior) = self.prior_mutation(tool, preview_sha256, idempotency_key)? {
            return Ok(prior);
        }
        if !self.pair_registry.is_file() {
            bail!("pair registry is missing");
        }
        let previous = sqlite_rows(
            &self.pair_registry,
            "SELECT value FROM controls WHERE key='admission_paused'",
        )?
        .first()
        .and_then(|row| row.get("value"))
        .and_then(Value::as_str)
            == Some("1");
        let reason = format!("freshness:{idempotency_key}");
        let event_kind = if paused {
            "admission_paused"
        } else {
            "admission_resumed"
        };
        let detail = sql_quote(&serde_json::to_string(
            &json!({"reason": reason, "changed": previous != paused}),
        )?);
        sqlite_exec(
            &self.pair_registry,
            &format!(
                "BEGIN IMMEDIATE;\
                 INSERT OR IGNORE INTO controls VALUES('admission_paused','0');\
                 UPDATE controls SET value={} WHERE key='admission_paused';\
                 INSERT INTO events(kind,detail_json,created_at_ms) VALUES({},{},{});\
                 COMMIT;",
                sql_quote(if paused { "1" } else { "0" }),
                sql_quote(event_kind),
                detail,
                now_ms()
            ),
        )?;
        self.state.admission_paused = paused;
        let result = json!({"admission_paused": paused, "changed": previous != paused});
        self.finish_mutation(tool, preview_sha256, idempotency_key, result)
    }

    fn request_micro_wave(
        &mut self,
        preview_sha256: &str,
        idempotency_key: &str,
        reason: &str,
    ) -> Result<Value> {
        if let Some(prior) =
            self.prior_mutation("request_micro_wave", preview_sha256, idempotency_key)?
        {
            return Ok(prior);
        }
        let request = json!({
            "schema": "JopeMicroWaveRequest.v1",
            "checkpoint_epoch": self.state.active_epoch,
            "checkpoint_sha256": self.state.active_checkpoint,
            "requested_at_ms": now_ms(),
            "reason": reason,
            "idempotency_key": idempotency_key
        });
        let directory = self.root.join("ops/micro-wave-requests");
        fs::create_dir_all(&directory)?;
        let target = directory.join(format!("{}.json", content_hash(&request)?));
        atomic_json(&target, &request)?;
        let mut result = request;
        result.as_object_mut().unwrap().insert(
            "request".to_owned(),
            json!(target.to_string_lossy().to_string()),
        );
        self.finish_mutation(
            "request_micro_wave",
            preview_sha256,
            idempotency_key,
            result,
        )
    }

    fn hold_stale_queued_from_inventory(&self, inventory: &[InventoryJob]) -> Result<u64> {
        let mut held = 0;
        for job in inventory {
            if !retirement_scoped_job(job)
                || job.epoch == self.state.active_epoch
                || !USEFUL_QUEUE_STATES.contains(&job.state.as_str())
            {
                continue;
            }
            self.scheduler.hold(&job.job_id)?;
            self.event(
                "stale_job_held",
                Some(&job.epoch),
                Some(&job.job_id),
                json!({"automatic": true}),
            )?;
            held += 1;
        }
        Ok(held)
    }

    fn auto_retire_stale(&mut self, preview: &Value) -> Result<Value> {
        let preview_sha256 = preview
            .get("preview_receipt_sha256")
            .and_then(Value::as_str)
            .context("automatic stale retirement preview lacks receipt SHA-256")?;
        let idempotency_key =
            auto_retirement_idempotency_key(&self.state.active_epoch, preview_sha256)?;
        let result = self.apply_stale_retirement(preview_sha256, &idempotency_key)?;
        Ok(json!({
            "idempotency_key": idempotency_key,
            "preview_receipt_sha256": preview_sha256,
            "result": result
        }))
    }

    fn cycle(&mut self, audit_only: bool, auto_retire_stale: bool) -> Result<Value> {
        self.sync_checkpoint()?;
        let mut inventory = self.inventory()?;
        let automatically_held = if audit_only {
            0
        } else {
            self.hold_stale_queued_from_inventory(&inventory)?
        };
        if automatically_held > 0 {
            inventory = self.inventory()?;
        }
        let preview = self.preview_stale_retirement_from_inventory(&inventory)?;
        let automatic_retirement = if auto_retire_stale && !audit_only {
            let result = self.auto_retire_stale(&preview)?;
            inventory = self.inventory()?;
            Some(result)
        } else {
            None
        };
        let mut result = self.freshness_status_from_inventory(&inventory);
        result.as_object_mut().unwrap().extend(Map::from_iter([
            (
                "latest_preview_receipt_sha256".to_owned(),
                preview["preview_receipt_sha256"].clone(),
            ),
            ("audit_only".to_owned(), json!(audit_only)),
            (
                "auto_retire_stale".to_owned(),
                json!(auto_retire_stale && !audit_only),
            ),
            (
                "automatic_retirement".to_owned(),
                automatic_retirement.unwrap_or(Value::Null),
            ),
            ("automatically_held".to_owned(), json!(automatically_held)),
            (
                "loaded_bundle_sha256".to_owned(),
                std::env::var("JOPE_LOADED_BUNDLE_SHA256").map_or(Value::Null, Value::String),
            ),
        ]));
        atomic_json(&self.root.join("ops/freshness-status.json"), &result)?;
        Ok(result)
    }

    fn dispatch(&mut self, method: &str, arguments: &Value) -> Result<Value> {
        if !TOOL_NAMES.contains(&method) {
            bail!("unsupported freshness method {method}");
        }
        let get_string = |name: &str| -> Result<&str> {
            arguments
                .get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .with_context(|| format!("missing {name}"))
        };
        match method {
            "get_freshness_status" => self.freshness_status(),
            "get_active_checkpoint" => Ok(self.active_checkpoint()),
            "list_queue_epochs" => self.queue_epochs(),
            "explain_job_provenance" => self.explain_job(get_string("job_id")?),
            "preview_stale_retirement" => self.preview_stale_retirement(),
            "apply_stale_retirement" => self.apply_stale_retirement(
                get_string("preview_receipt_sha256")?,
                get_string("idempotency_key")?,
            ),
            "pause_admission" => self.admission_mutation(
                true,
                get_string("preview_receipt_sha256")?,
                get_string("idempotency_key")?,
            ),
            "resume_admission" => self.admission_mutation(
                false,
                get_string("preview_receipt_sha256")?,
                get_string("idempotency_key")?,
            ),
            "request_micro_wave" => self.request_micro_wave(
                get_string("preview_receipt_sha256")?,
                get_string("idempotency_key")?,
                arguments
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or("operator_request"),
            ),
            _ => unreachable!(),
        }
    }

    fn prune_state(&mut self) {
        let cutoff = now_ms() - PREVIEW_MAX_AGE_MS;
        self.state
            .previews
            .retain(|_, preview| preview.created_at_ms >= cutoff);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let audit_only =
        args.audit_only || std::env::var("JOPE_FRESHNESS_AUDIT_ONLY").as_deref() == Ok("1");
    let socket_path = args
        .socket
        .unwrap_or_else(|| args.root.join("ops/freshness.sock"));
    if args.once {
        let mut supervisor = Supervisor::open(args.root, args.scq)?;
        let status = supervisor.cycle(audit_only, args.auto_retire_stale)?;
        println!("{}", serde_json::to_string(&status)?);
        return Ok(());
    }
    serve(
        args.root,
        args.scq,
        socket_path,
        args.interval,
        audit_only,
        args.auto_retire_stale,
    )
}

fn serve(
    root: PathBuf,
    scq: PathBuf,
    socket_path: PathBuf,
    interval_seconds: f64,
    audit_only: bool,
    auto_retire_stale: bool,
) -> Result<()> {
    let ops = root.join("ops");
    fs::create_dir_all(&ops)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .mode(0o600)
        .open(ops.join("freshness.lock"))?;
    if lock.try_lock().is_err() {
        bail!("another freshness authority owns the process lock");
    }
    if socket_path.exists() {
        fs::remove_file(&socket_path)
            .with_context(|| format!("remove stale socket {}", socket_path.display()))?;
    }
    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let listener = UnixListener::bind(&socket_path)?;
    listener.set_nonblocking(true)?;
    let mut supervisor = Supervisor::open(root, scq)?;
    let interval = Duration::from_secs_f64(interval_seconds.max(1.0));
    let mut next_cycle = Instant::now();
    loop {
        if Instant::now() >= next_cycle {
            if let Err(error) = supervisor.cycle(audit_only, auto_retire_stale) {
                supervisor.event(
                    "cycle_failed",
                    None,
                    None,
                    json!({"error": format!("{error:#}")}),
                )?;
            }
            next_cycle = Instant::now() + interval;
        }
        loop {
            match listener.accept() {
                Ok((stream, _)) => handle_socket(stream, &mut supervisor)?,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn handle_socket(mut stream: UnixStream, supervisor: &mut Supervisor) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let mut line = String::new();
    BufReader::new(stream.try_clone()?)
        .take(4 << 20)
        .read_line(&mut line)?;
    let response = match serde_json::from_str::<Value>(&line) {
        Ok(request) => {
            let method = request.get("method").and_then(Value::as_str).unwrap_or("");
            let arguments = request.get("arguments").unwrap_or(&Value::Null);
            match supervisor.dispatch(method, arguments) {
                Ok(result) => json!({"ok": true, "result": result}),
                Err(error) => json!({
                    "ok": false,
                    "error": "FreshnessError",
                    "detail": format!("{error:#}")
                }),
            }
        }
        Err(error) => json!({
            "ok": false,
            "error": "JsonError",
            "detail": error.to_string()
        }),
    };
    // One write per response instead of one per JSON token on the unbuffered handle.
    let mut line = serde_json::to_vec(&response)?;
    line.push(b'\n');
    stream.write_all(&line)?;
    stream.flush()?;
    Ok(())
}

fn sqlite_rows(database: &Path, sql: &str) -> Result<Vec<Value>> {
    let connection = open_sqlite(database)?;
    let mut statement = connection
        .prepare(sql)
        .with_context(|| format!("prepare SQLite audit for {}", database.display()))?;
    let names = statement
        .column_names()
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let mut rows = statement.query([])?;
    let mut result = Vec::new();
    while let Some(row) = rows.next()? {
        let mut object = Map::new();
        for (index, name) in names.iter().enumerate() {
            let value = match row.get_ref(index)? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(value) => json!(value),
                ValueRef::Real(value) => json!(value),
                ValueRef::Text(value) => Value::String(String::from_utf8_lossy(value).into_owned()),
                ValueRef::Blob(value) => Value::String(hex::encode(value)),
            };
            object.insert(name.clone(), value);
        }
        result.push(Value::Object(object));
    }
    Ok(result)
}

fn sqlite_exec(database: &Path, sql: &str) -> Result<()> {
    open_sqlite(database)?
        .execute_batch(sql)
        .with_context(|| format!("update {}", database.display()))
}

fn open_sqlite(database: &Path) -> Result<Connection> {
    let connection =
        Connection::open(database).with_context(|| format!("open {}", database.display()))?;
    connection.busy_timeout(Duration::from_secs(60))?;
    Ok(connection)
}

fn sqlite_column_exists(connection: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(columns.iter().any(|candidate| candidate == column))
}

fn import_legacy_state(path: &Path) -> Result<DurableState> {
    if !path.is_file() {
        return Ok(DurableState::new());
    }
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("open legacy freshness state {}", path.display()))?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let has_controls: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='controls')",
        [],
        |row| row.get(0),
    )?;
    if !has_controls {
        return Ok(DurableState::new());
    }
    let control = |key: &str| -> Result<String> {
        Ok(connection
            .query_row("SELECT value FROM controls WHERE key=?1", [key], |row| {
                row.get(0)
            })
            .optional()?
            .unwrap_or_default())
    };
    let mut state = DurableState {
        schema: "JopeFreshnessState.v1".to_owned(),
        active_epoch: control("active_epoch")?,
        active_checkpoint: control("active_checkpoint")?,
        admission_paused: control("admission_paused")? == "1",
        ..DurableState::default()
    };
    let mut checkpoints = connection.prepare(
        "SELECT epoch,activated_at_ms,superseded_at_ms FROM checkpoints ORDER BY activated_at_ms",
    )?;
    for row in checkpoints.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, Option<i64>>(2)?,
        ))
    })? {
        let (epoch, activated, superseded) = row?;
        state
            .checkpoint_activated_at_ms
            .insert(epoch.clone(), activated);
        if let Some(superseded) = superseded {
            state.checkpoint_superseded_at_ms.insert(epoch, superseded);
        }
    }
    let mut previews = connection
        .prepare("SELECT receipt_sha256,payload_json FROM previews ORDER BY created_at_ms")?;
    for row in previews.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })? {
        let (digest, payload) = row?;
        let preview: Preview =
            serde_json::from_str(&payload).context("parse legacy freshness preview")?;
        state.previews.insert(digest, preview);
    }
    let mut mutations = connection.prepare(
        "SELECT idempotency_key,tool,preview_sha256,result_json,created_at_ms
         FROM mutations ORDER BY created_at_ms",
    )?;
    for row in mutations.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i64>(4)?,
        ))
    })? {
        let (key, tool, preview_sha256, result, created_at_ms) = row?;
        state.mutations.insert(
            key,
            Mutation {
                tool,
                preview_sha256,
                result: serde_json::from_str(&result).context("parse legacy freshness mutation")?,
                created_at_ms,
            },
        );
    }
    Ok(state)
}

fn sql_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn freshness_scoped_manifest(manifest: &Value) -> bool {
    if manifest.get("schema").and_then(Value::as_str) != Some("JopeFlywheelCorpusEvalShard.v1")
        || !manifest.get("candidates").is_some_and(Value::is_array)
    {
        return false;
    }
    let lane = manifest
        .get("evaluation_lane")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    !(lane.contains("gate") || lane.contains("sealed") || lane.contains("canary"))
}

fn manifest_epoch(manifest: &Value) -> String {
    if let Some(epoch) = manifest
        .get("checkpoint_epoch")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        return epoch.to_owned();
    }
    let checkpoint = manifest
        .get("label_authority")
        .and_then(|authority| authority.get("model_checkpoint_sha256"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if is_sha256(checkpoint) {
        return format!("checkpoint-{checkpoint}");
    }
    "legacy-jope-testonly".to_owned()
}

fn epoch_for_checkpoint(checkpoint: &str) -> Result<String> {
    if !is_sha256(checkpoint) {
        bail!("checkpoint epoch requires a SHA-256 checkpoint identity");
    }
    Ok(format!("checkpoint-{checkpoint}"))
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn content_hash(value: &Value) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}

fn auto_retirement_idempotency_key(epoch: &str, preview_sha256: &str) -> Result<String> {
    if epoch.is_empty() || !is_sha256(preview_sha256) {
        bail!("automatic stale retirement lacks deterministic provenance");
    }
    Ok(format!(
        "auto-stale-retirement-{}",
        content_hash(&json!({
            "schema": "JopeAutomaticStaleRetirementIdempotency.v1",
            "active_epoch": epoch,
            "preview_receipt_sha256": preview_sha256
        }))?
    ))
}

fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .mode(0o600)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn null_if_empty(value: &str) -> Value {
    if value.is_empty() {
        Value::Null
    } else {
        json!(value)
    }
}

fn retirement_scoped_job(job: &InventoryJob) -> bool {
    job.outside_scq
        || job
            .scheduler
            .project
            .as_deref()
            .is_some_and(|project| RETIRABLE_PROJECTS.contains(&project))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tempfile::tempdir;

    #[derive(Default)]
    struct MockSchedulerState {
        jobs: Vec<SchedulerJob>,
        jobs_calls: usize,
        held: Vec<String>,
        cancelled: Vec<String>,
        race_on_jobs_call: Option<usize>,
        race_state: Option<String>,
    }

    #[derive(Clone)]
    struct MockScheduler {
        state: Arc<Mutex<MockSchedulerState>>,
    }

    impl Scheduler for MockScheduler {
        fn jobs(&self) -> Result<Vec<SchedulerJob>> {
            let mut state = self.state.lock().unwrap();
            state.jobs_calls += 1;
            if state.race_on_jobs_call == Some(state.jobs_calls) {
                let raced = state
                    .race_state
                    .clone()
                    .unwrap_or_else(|| "running".to_owned());
                for job in &mut state.jobs {
                    job.state = raced.clone();
                }
            }
            Ok(state.jobs.clone())
        }

        fn hold(&self, job_id: &str) -> Result<()> {
            let mut state = self.state.lock().unwrap();
            let job = state
                .jobs
                .iter_mut()
                .find(|job| job.id == job_id)
                .context("mock hold job is absent")?;
            if matches!(job.state.as_str(), "queued" | "starting") {
                job.state = "held".to_owned();
            }
            state.held.push(job_id.to_owned());
            Ok(())
        }

        fn cancel(&self, job_id: &str) -> Result<()> {
            let mut state = self.state.lock().unwrap();
            let job = state
                .jobs
                .iter_mut()
                .find(|job| job.id == job_id)
                .context("mock cancel job is absent")?;
            if job.state != "held" {
                bail!("mock refuses to cancel a non-held job");
            }
            job.state = "cancelled".to_owned();
            state.cancelled.push(job_id.to_owned());
            Ok(())
        }
    }

    struct RetirementFixture {
        _temporary: tempfile::TempDir,
        supervisor: Supervisor,
        scheduler: Arc<Mutex<MockSchedulerState>>,
        registry: PathBuf,
        backlog: PathBuf,
    }

    fn retirement_fixture(scheduler_state: &str, project: &str) -> RetirementFixture {
        let temporary = tempdir().unwrap();
        let root = temporary.path().join("root");
        fs::create_dir_all(root.join("ops")).unwrap();
        fs::create_dir_all(root.join("manifests")).unwrap();
        let old_checkpoint = "a".repeat(64);
        let manifest = root.join("manifests/stale.json");
        atomic_json(
            &manifest,
            &json!({
                "schema": "JopeFlywheelCorpusEvalShard.v1",
                "checkpoint_epoch": format!("checkpoint-{old_checkpoint}"),
                "candidates": []
            }),
        )
        .unwrap();
        let manifest_sha256 = sha256_file(&manifest).unwrap();
        let backlog = root.join("ops/backlog.sqlite");
        let connection = Connection::open(&backlog).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE claims(
                   manifest_sha256 TEXT NOT NULL,
                   manifest TEXT NOT NULL,
                   job_id TEXT PRIMARY KEY,
                   state TEXT NOT NULL,
                   reservation_owner TEXT,
                   updated_at_ms INTEGER NOT NULL
                 );",
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO claims VALUES(?1,?2,'job-stale','submitted','owner',1)",
                params![manifest_sha256, manifest.to_string_lossy()],
            )
            .unwrap();
        drop(connection);

        let registry = root.join("ops/pair-registry.sqlite");
        let connection = Connection::open(&registry).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE pairs(
                   task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,protocol_sha256 TEXT,
                   canonical_genome TEXT,state TEXT,reservation_owner TEXT,manifest_path TEXT,
                   manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,failure_class TEXT,
                   infrastructure_retries INTEGER,created_at_ms INTEGER,updated_at_ms INTEGER,
                   PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
                 );
                 CREATE TABLE attempts(
                   id INTEGER PRIMARY KEY,task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,
                   protocol_sha256 TEXT,attempt_no INTEGER,state TEXT,reservation_owner TEXT,
                   manifest_path TEXT,manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,
                   failure_class TEXT,detail_json TEXT,created_at_ms INTEGER,finalized_at_ms INTEGER
                 );
                 CREATE TABLE events(
                   id INTEGER PRIMARY KEY,kind TEXT,task TEXT,dataset_hash TEXT,
                   semantic_sha256 TEXT,protocol_sha256 TEXT,detail_json TEXT,created_at_ms INTEGER
                 );",
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO pairs VALUES(
                   'regression','dataset','semantic','protocol','genome','submitted','owner',
                   ?1,?2,'job-stale',NULL,NULL,0,1,1
                 )",
                params![manifest.to_string_lossy(), manifest_sha256],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO attempts VALUES(
                   NULL,'regression','dataset','semantic','protocol',1,'submitted','owner',
                   ?1,?2,'job-stale',NULL,NULL,'{}',1,NULL
                 )",
                params![manifest.to_string_lossy(), manifest_sha256],
            )
            .unwrap();
        drop(connection);

        let scheduler = Arc::new(Mutex::new(MockSchedulerState {
            jobs: vec![SchedulerJob {
                id: "job-stale".to_owned(),
                state: scheduler_state.to_owned(),
                project: Some(project.to_owned()),
                ..SchedulerJob::default()
            }],
            ..MockSchedulerState::default()
        }));
        let mut supervisor =
            Supervisor::open(root, PathBuf::from("/not-used-in-this-test")).unwrap();
        supervisor.scheduler = Box::new(MockScheduler {
            state: scheduler.clone(),
        });
        let active = "b".repeat(64);
        supervisor.state.active_checkpoint = active.clone();
        supervisor.state.active_epoch = format!("checkpoint-{active}");
        RetirementFixture {
            _temporary: temporary,
            supervisor,
            scheduler,
            registry,
            backlog,
        }
    }

    #[test]
    fn checkpoint_epoch_requires_lowercase_sha256() {
        let digest = "a".repeat(64);
        assert_eq!(
            epoch_for_checkpoint(&digest).unwrap(),
            format!("checkpoint-{digest}")
        );
        assert!(epoch_for_checkpoint("not-a-hash").is_err());
        assert!(epoch_for_checkpoint(&"A".repeat(64)).is_err());
    }

    #[test]
    fn scope_excludes_protected_lanes() {
        let base = json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "candidates": []
        });
        assert!(freshness_scoped_manifest(&base));
        for lane in [
            "checkpoint_gate",
            "sealed_audit",
            "evaluator_canary",
            "milestone-sealed-zero-shot",
        ] {
            let mut protected = base.clone();
            protected["evaluation_lane"] = json!(lane);
            assert!(!freshness_scoped_manifest(&protected));
        }
    }

    #[test]
    fn manifest_epoch_falls_back_only_to_authority_or_legacy() {
        let digest = "b".repeat(64);
        assert_eq!(
            manifest_epoch(&json!({
                "schema": "JopeFlywheelCorpusEvalShard.v1",
                "candidates": [],
                "label_authority": {"model_checkpoint_sha256": digest}
            })),
            format!("checkpoint-{digest}")
        );
        assert_eq!(
            manifest_epoch(&json!({
                "schema": "JopeFlywheelCorpusEvalShard.v1",
                "candidates": []
            })),
            "legacy-jope-testonly"
        );
    }

    #[test]
    fn sql_quoting_never_changes_statement_boundaries() {
        assert_eq!(
            sql_quote("job'; DROP TABLE claims;--"),
            "'job''; DROP TABLE claims;--'"
        );
    }

    #[test]
    fn minimal_scheduler_decode_does_not_retain_large_job_payloads() {
        let command = "candidate-command-".repeat(64);
        let output = "receipt-output-".repeat(32);
        let rows = (0..12_000)
            .map(|index| {
                json!({
                    "id": format!("job-{index:05}"),
                    "state": "queued",
                    "project": "jope-selfplay-eval",
                    "node": null,
                    "queue_ts": "2026-07-27T00:00:00Z",
                    "command": command,
                    "outputs": [output, output],
                    "message": output,
                    "run_dir": format!("/runs/job-{index:05}")
                })
            })
            .collect::<Vec<_>>();
        let encoded = serde_json::to_vec(&json!({"Jobs": {"jobs": rows}})).unwrap();
        let started = Instant::now();
        let decoded = parse_scheduler_jobs(&encoded).unwrap();
        let elapsed = started.elapsed();
        let retained_string_bytes = decoded
            .iter()
            .map(|job| {
                job.id.len()
                    + job.state.len()
                    + job.project.as_deref().map_or(0, str::len)
                    + job.queue_ts.as_deref().map_or(0, str::len)
                    + job.run_dir.as_deref().map_or(0, str::len)
            })
            .sum::<usize>();
        eprintln!(
            "minimal-scq-decode jobs={} input_bytes={} retained_string_bytes={} elapsed_ms={}",
            decoded.len(),
            encoded.len(),
            retained_string_bytes,
            elapsed.as_millis()
        );
        assert_eq!(decoded.len(), 12_000);
        assert!(retained_string_bytes * 20 < encoded.len());
    }

    #[test]
    fn identical_preview_is_reused_and_expired_preview_is_pruned() {
        let temporary = tempdir().unwrap();
        let root = temporary.path().join("root");
        let mut supervisor =
            Supervisor::open(root, PathBuf::from("/not-used-in-this-test")).unwrap();
        let payload = json!({"actions": [{"job_id": "same"}]});
        let first = supervisor
            .write_preview("stale_retirement", payload.clone())
            .unwrap();
        let second = supervisor
            .write_preview("stale_retirement", payload)
            .unwrap();
        assert_eq!(
            first["preview_receipt_sha256"],
            second["preview_receipt_sha256"]
        );
        assert_eq!(supervisor.state.previews.len(), 1);

        supervisor.state.previews.insert(
            "f".repeat(64),
            Preview {
                schema: "JopeFreshnessPreview.v1".to_owned(),
                kind: "stale_retirement".to_owned(),
                created_at_ms: now_ms() - PREVIEW_MAX_AGE_MS - 1,
                payload: json!({"actions": [{"job_id": "expired"}]}),
            },
        );
        supervisor.prune_state();
        assert_eq!(supervisor.state.previews.len(), 1);
    }

    #[test]
    fn manifest_cache_reuses_immutable_audit_and_detects_metadata_change() {
        let temporary = tempdir().unwrap();
        let root = temporary.path().join("root");
        let manifest = temporary.path().join("manifest.json");
        let initial = serde_json::to_vec(&json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "checkpoint_epoch": format!("checkpoint-{}", "a".repeat(64)),
            "candidates": []
        }))
        .unwrap();
        fs::write(&manifest, initial).unwrap();
        let expected = sha256_file(&manifest).unwrap();
        let supervisor = Supervisor::open(root, PathBuf::from("/not-used-in-this-test")).unwrap();
        let first = supervisor
            .audit_manifest("job", &manifest, &expected, false)
            .unwrap();
        let second = supervisor
            .audit_manifest("job", &manifest, &expected, false)
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(supervisor.manifest_cache.borrow().len(), 1);

        fs::write(&manifest, b"{\"changed\":true,\"longer\":true}").unwrap();
        assert!(supervisor
            .audit_manifest("job", &manifest, &expected, false)
            .is_err());
    }

    #[test]
    fn retirement_preserves_attempt_and_releases_pair() {
        let temporary = tempdir().unwrap();
        let root = temporary.path().join("root");
        fs::create_dir_all(root.join("ops")).unwrap();
        let registry = root.join("ops/pair-registry.sqlite");
        sqlite_exec(
            &registry,
            r#"
CREATE TABLE pairs(
  task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,protocol_sha256 TEXT,
  canonical_genome TEXT,state TEXT,reservation_owner TEXT,manifest_path TEXT,
  manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,failure_class TEXT,
  infrastructure_retries INTEGER,created_at_ms INTEGER,updated_at_ms INTEGER,
  PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
);
CREATE TABLE attempts(
  id INTEGER PRIMARY KEY,task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,
  protocol_sha256 TEXT,attempt_no INTEGER,state TEXT,reservation_owner TEXT,
  manifest_path TEXT,manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,
  failure_class TEXT,detail_json TEXT,created_at_ms INTEGER,finalized_at_ms INTEGER
);
CREATE TABLE events(
  id INTEGER PRIMARY KEY,kind TEXT,task TEXT,dataset_hash TEXT,
  semantic_sha256 TEXT,protocol_sha256 TEXT,detail_json TEXT,created_at_ms INTEGER
);
INSERT INTO pairs VALUES(
  'regression','dataset','semantic','protocol','genome','submitted','owner',
  '/manifest','manifest','job',NULL,NULL,0,1,1
);
INSERT INTO attempts VALUES(
  NULL,'regression','dataset','semantic','protocol',1,'submitted','owner',
  '/manifest','manifest','job',NULL,NULL,'{}',1,NULL
);
"#,
        )
        .unwrap();
        let supervisor = Supervisor::open(root, PathBuf::from("/not-used-in-this-test")).unwrap();
        assert_eq!(
            supervisor
                .supersede_pairs("manifest", "job", "submitted", &"a".repeat(64))
                .unwrap(),
            1
        );
        let pair = sqlite_rows(&registry, "SELECT state,manifest_sha256 FROM pairs")
            .unwrap()
            .remove(0);
        assert_eq!(pair["state"], "available");
        assert!(pair["manifest_sha256"].is_null());
        let attempt = sqlite_rows(
            &registry,
            "SELECT state,receipt_sha256,finalized_at_ms FROM attempts",
        )
        .unwrap()
        .remove(0);
        assert_eq!(attempt["state"], "superseded_unstarted");
        assert_eq!(attempt["receipt_sha256"], "a".repeat(64));
        assert!(attempt["finalized_at_ms"].as_i64().unwrap() > 0);
    }

    #[test]
    fn automatic_promotion_turnover_holds_then_retires_only_stale_unstarted_work() {
        let mut fixture = retirement_fixture("queued", "jope-selfplay-eval");
        let status = fixture.supervisor.cycle(false, true).unwrap();
        assert_eq!(status["automatically_held"], 1);
        assert_eq!(
            status["automatic_retirement"]["result"]["running_interrupted"],
            0
        );
        assert_eq!(status["automatic_retirement"]["result"]["retired"], 1);
        let scheduler = fixture.scheduler.lock().unwrap();
        assert_eq!(scheduler.held, ["job-stale"]);
        assert_eq!(scheduler.cancelled, ["job-stale"]);
        assert_eq!(scheduler.jobs[0].state, "cancelled");
        drop(scheduler);
        let pair = sqlite_rows(&fixture.registry, "SELECT state,manifest_sha256 FROM pairs")
            .unwrap()
            .remove(0);
        assert_eq!(pair["state"], "available");
        assert!(pair["manifest_sha256"].is_null());
        let attempt = sqlite_rows(
            &fixture.registry,
            "SELECT state,finalized_at_ms FROM attempts",
        )
        .unwrap()
        .remove(0);
        assert_eq!(attempt["state"], "superseded_unstarted");
        assert!(attempt["finalized_at_ms"].as_i64().unwrap() > 0);
        let claim = sqlite_rows(&fixture.backlog, "SELECT state FROM claims")
            .unwrap()
            .remove(0);
        assert_eq!(claim["state"], "superseded_unstarted");
    }

    #[test]
    fn automatic_retirement_is_idempotent_for_epoch_and_preview() {
        let mut fixture = retirement_fixture("held", "jope-selfplay-eval");
        let inventory = fixture.supervisor.inventory().unwrap();
        let preview = fixture
            .supervisor
            .preview_stale_retirement_from_inventory(&inventory)
            .unwrap();
        let first = fixture.supervisor.auto_retire_stale(&preview).unwrap();
        let second = fixture.supervisor.auto_retire_stale(&preview).unwrap();
        assert_eq!(first, second);
        assert_eq!(fixture.supervisor.state.mutations.len(), 1);
        assert_eq!(fixture.scheduler.lock().unwrap().cancelled.len(), 1);
        let preview_sha256 = preview["preview_receipt_sha256"].as_str().unwrap();
        let key =
            auto_retirement_idempotency_key(&fixture.supervisor.state.active_epoch, preview_sha256)
                .unwrap();
        assert_eq!(first["idempotency_key"], key);
        assert_ne!(
            key,
            auto_retirement_idempotency_key(
                &format!("checkpoint-{}", "c".repeat(64)),
                preview_sha256
            )
            .unwrap()
        );
    }

    #[test]
    fn automatic_retirement_loses_dispatch_race_without_interrupting_job() {
        let mut fixture = retirement_fixture("held", "jope-selfplay-eval");
        {
            let mut scheduler = fixture.scheduler.lock().unwrap();
            scheduler.race_on_jobs_call = Some(4);
            scheduler.race_state = Some("running".to_owned());
        }
        let status = fixture.supervisor.cycle(false, true).unwrap();
        assert_eq!(status["automatic_retirement"]["result"]["retired"], 0);
        assert_eq!(
            status["automatic_retirement"]["result"]["running_interrupted"],
            0
        );
        assert_eq!(fixture.scheduler.lock().unwrap().cancelled.len(), 0);
        let pair = sqlite_rows(&fixture.registry, "SELECT state FROM pairs")
            .unwrap()
            .remove(0);
        assert_eq!(pair["state"], "submitted");
        let claim = sqlite_rows(&fixture.backlog, "SELECT state FROM claims")
            .unwrap()
            .remove(0);
        assert_eq!(claim["state"], "submitted");
    }

    #[test]
    fn automatic_retirement_never_touches_gpu_training_project() {
        let mut fixture = retirement_fixture("queued", "jope-selfplay-train");
        let status = fixture.supervisor.cycle(false, true).unwrap();
        assert_eq!(status["automatically_held"], 0);
        assert_eq!(status["automatic_retirement"]["result"]["retired"], 0);
        let scheduler = fixture.scheduler.lock().unwrap();
        assert!(scheduler.held.is_empty());
        assert!(scheduler.cancelled.is_empty());
        assert_eq!(scheduler.jobs[0].state, "queued");
    }

    #[test]
    fn audit_mode_overrides_automatic_retirement_flag() {
        let mut fixture = retirement_fixture("queued", "jope-selfplay-eval");
        let status = fixture.supervisor.cycle(true, true).unwrap();
        assert_eq!(status["audit_only"], true);
        assert_eq!(status["auto_retire_stale"], false);
        assert!(status["automatic_retirement"].is_null());
        let scheduler = fixture.scheduler.lock().unwrap();
        assert!(scheduler.held.is_empty());
        assert!(scheduler.cancelled.is_empty());
        assert_eq!(scheduler.jobs[0].state, "queued");
    }

    #[test]
    fn micro_wave_request_is_receipt_gated_and_idempotent() {
        let temporary = tempdir().unwrap();
        let root = temporary.path().join("root");
        let mut supervisor =
            Supervisor::open(root.clone(), PathBuf::from("/not-used-in-this-test")).unwrap();
        supervisor.state.active_epoch = format!("checkpoint-{}", "b".repeat(64));
        supervisor.state.active_checkpoint = "b".repeat(64);
        let preview = supervisor
            .write_preview("stale_retirement", json!({"actions": []}))
            .unwrap();
        let digest = preview["preview_receipt_sha256"].as_str().unwrap();
        let first = supervisor
            .request_micro_wave(digest, "stable-key", "test")
            .unwrap();
        let second = supervisor
            .request_micro_wave(digest, "stable-key", "changed-reason-is-ignored")
            .unwrap();
        assert_eq!(first, second);
        assert!(Path::new(first["request"].as_str().unwrap()).is_file());
        assert!(supervisor
            .request_micro_wave(digest, "other-key", "test")
            .is_ok());
        assert!(supervisor
            .request_micro_wave(&"f".repeat(64), "unknown-preview", "test")
            .is_err());
    }
}
