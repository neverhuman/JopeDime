//! jope_utilization_supervisor — the control loop whose objective is "the cluster is busy".
//!
//! WHY THIS EXISTS. On 2026-07-28 every one of the 1,052 scheduler slots went
//! idle and throughput fell from ~50-86k observations/hour to zero. Nothing in
//! the system noticed. The owner found out by looking at a screen.
//!
//! The two structural reasons nothing noticed are both design bugs this binary
//! is built to be immune to:
//!
//!  1. Every liveness guard in the flywheel checks a status file's MTIME
//!     (jope_selfplay_alarms.rs:21-27, 94-113). Both the queue keeper and the
//!     supply supervisor rewrite their status file on EVERY FAILED CYCLE, so a
//!     daemon spinning in a tight error loop looks maximally healthy: fresh
//!     mtime, systemd ActiveState=active, NRestarts=0. Therefore this binary
//!     NEVER trusts an mtime for health. It reads the scheduler itself
//!     (`scq --json snapshot`, the ground truth for what is actually running)
//!     and it reads each receipt's CONTENT — specifically the `schema` field,
//!     because both daemons switch to a distinct `*Error.v1` schema when they
//!     fail (jope_selfplay_queue.rs:275, jope_selfplay_supply.rs:258).
//!
//!  2. `below_floor` is the only queue-depth objective in the system
//!     (jope_selfplay_supply.rs:776). It is computed, written to a status
//!     file, and then acted on by nobody. `selfplay_core.rs:101` exposes
//!     `queue_refill_needed()` and nothing calls it. An objective nobody acts
//!     on is not a control loop, it is a comment.
//!
//! WHAT IT DOES. detect -> diagnose -> repair (bounded) -> escalate -> record.
//! It deliberately does NOT build waves: that is supply's job. Its job is to
//! notice the objective is unmet and make something happen.
//!
//! SAFETY. The only side effects on live state are `systemctl --user
//! restart|start` on flywheel units, plus files under `ops/` that this binary
//! alone owns. It never opens, let alone writes, any of the ledgers
//! (backlog.sqlite / pair-registry.sqlite / the corpus). A supervisor that can
//! corrupt a ledger is worse than the outage it is fixing.

use anyhow::{Context, Result};
use clap::Parser;
use jope_dime_regressor::paths;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

jope_paths::once_str!(
    fn root_str => paths::flywheel_root()
);
jope_paths::once_str!(
    fn scq_str => paths::scq_binary()
);
const SCQ_SNAPSHOT_TIMEOUT_SECS: &str = "30";
const HOUR_MS: u64 = 3_600_000;

// ---------------------------------------------------------------------------
// arguments
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(about = "Detect, diagnose, repair and escalate an under-utilized cluster")]
struct Args {
    #[arg(long, default_value = root_str())]
    root: PathBuf,
    #[arg(long, default_value = scq_str())]
    scq: PathBuf,

    /// Core-budget occupancy (%) below which the cluster counts as IDLE.
    /// Core budget, not `job_slot_target`: slot targets are multiplied up
    /// (1,052 slots over 344 real core-budget units) so slot occupancy reads
    /// ~30% on a cluster that is in fact core-saturated.
    #[arg(long, default_value_t = 25.0)]
    idle_core_pct: f64,
    /// Queued+dispatching jobs below this is "under-supplied": the cluster is
    /// coasting on in-flight work with nothing behind it.
    #[arg(long, default_value_t = 2048)]
    queue_floor_jobs: u64,
    /// Consecutive IDLE cycles before escalating to the owner.
    #[arg(long, default_value_t = 5)]
    escalate_idle_cycles: u32,
    /// Consecutive DEGRADED cycles (busy now, producer broken) before escalating.
    #[arg(long, default_value_t = 15)]
    escalate_degraded_cycles: u32,

    /// Hard ceiling on repair actions in any rolling hour. Never negotiable:
    /// a supervisor that can restart-storm is a second outage.
    #[arg(long, default_value_t = 6)]
    max_actions_per_hour: usize,
    /// First retry delay for a given action; doubles per consecutive attempt.
    #[arg(long, default_value_t = 300)]
    backoff_base_seconds: u64,
    #[arg(long, default_value_t = 3600)]
    backoff_max_seconds: u64,
    /// Minimum gap between owner notifications while an escalation is open.
    #[arg(long, default_value_t = 1800)]
    renotify_seconds: u64,

    /// Diagnose and record; never repair, never notify.
    #[arg(long)]
    dry_run: bool,
    /// Force the IDLE verdict regardless of measurement. Rehearsal only.
    #[arg(long)]
    simulate_idle: bool,
    /// Print an historical utilization report over the last N hours and exit.
    #[arg(long)]
    report_hours: Option<f64>,

    #[arg(long)]
    state_file: Option<PathBuf>,
    #[arg(long)]
    status_file: Option<PathBuf>,
    #[arg(long)]
    series_file: Option<PathBuf>,
    #[arg(long)]
    events_file: Option<PathBuf>,
    #[arg(long)]
    alarms_file: Option<PathBuf>,
    #[arg(long)]
    banner_file: Option<PathBuf>,
}

impl Args {
    fn state(&self) -> PathBuf {
        self.state_file
            .clone()
            .unwrap_or_else(|| self.root.join("ops/utilization-supervisor-state.v1.json"))
    }
    fn status(&self) -> PathBuf {
        self.status_file
            .clone()
            .unwrap_or_else(|| self.root.join("ops/utilization-supervisor-v1.json"))
    }
    fn series(&self) -> PathBuf {
        self.series_file
            .clone()
            .unwrap_or_else(|| self.root.join("ops/utilization-series.v1.jsonl"))
    }
    fn events(&self) -> PathBuf {
        self.events_file
            .clone()
            .unwrap_or_else(|| self.root.join("ops/utilization-supervisor-events.v1.jsonl"))
    }
    fn alarms(&self) -> PathBuf {
        self.alarms_file
            .clone()
            .unwrap_or_else(|| self.root.join("ops/utilization-alarms.v1.json"))
    }
    fn banner(&self) -> PathBuf {
        self.banner_file
            .clone()
            .unwrap_or_else(|| self.root.join("ops/UTILIZATION-ESCALATION.txt"))
    }
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Epoch millis -> `YYYY-MM-DDTHH:MM:SSZ` (Howard Hinnant civil-from-days).
fn iso_utc(ms: u64) -> String {
    let secs = (ms / 1000) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

/// Atomic replace: a torn status file read by the TUI mid-write is a bug report
/// about the supervisor, which defeats the point of the supervisor.
fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let temporary = path.with_extension("tmp");
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(&temporary, &bytes).with_context(|| format!("write {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("rename into {}", path.display()))?;
    Ok(())
}

fn append_jsonl(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

fn str_at<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}
fn u64_at(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}
fn f64_at(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

// ---------------------------------------------------------------------------
// DETECT — scheduler ground truth
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Utilization {
    scq_reachable: bool,
    scq_error: Option<String>,
    epoch: u64,
    nodes_up: u64,
    nodes_total: u64,
    /// Sum of per-node `running_job_slots` — the operator's "N of 1,052 slots".
    slots_in_use: u64,
    slots_total: u64,
    /// Core budget is the constraint that actually binds; slot targets are a
    /// 4x multiplier over it on the 3 big nodes.
    core_budget_total: u64,
    core_budget_free: u64,
    jobs_queued: u64,
    jobs_running: u64,
    jobs_dispatching: u64,
    throughput_per_min: u64,
    eval_running: u64,
    eval_queued: u64,
    per_node: Vec<Value>,
}

impl Utilization {
    fn core_pct(&self) -> f64 {
        if self.core_budget_total == 0 {
            return 0.0;
        }
        let used = self.core_budget_total.saturating_sub(self.core_budget_free);
        used as f64 * 100.0 / self.core_budget_total as f64
    }
    fn slot_pct(&self) -> f64 {
        if self.slots_total == 0 {
            return 0.0;
        }
        self.slots_in_use as f64 * 100.0 / self.slots_total as f64
    }
    fn queue_depth(&self) -> u64 {
        self.jobs_queued + self.jobs_dispatching
    }
}

fn measure(args: &Args) -> Utilization {
    let mut u = Utilization::default();
    let output = Command::new("/usr/bin/timeout")
        .arg("--kill-after=5")
        .arg(SCQ_SNAPSHOT_TIMEOUT_SECS)
        .arg(&args.scq)
        .arg("--json")
        .arg("snapshot")
        .output();
    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            u.scq_error = Some(format!(
                "scq snapshot exit {:?}: {}",
                o.status.code(),
                String::from_utf8_lossy(&o.stderr).trim()
            ));
            return u;
        }
        Err(error) => {
            u.scq_error = Some(format!("scq snapshot spawn failed: {error}"));
            return u;
        }
    };
    let Ok(root) = serde_json::from_slice::<Value>(&output.stdout) else {
        u.scq_error = Some("scq snapshot returned unparsable JSON".into());
        return u;
    };
    let snapshot = root.get("Snapshot").unwrap_or(&root);
    u.scq_reachable = true;
    u.epoch = u64_at(snapshot, "epoch").unwrap_or(0);

    let counters = snapshot.get("counters").cloned().unwrap_or(Value::Null);
    u.nodes_up = u64_at(&counters, "nodes_up").unwrap_or(0);
    u.nodes_total = u64_at(&counters, "nodes_total").unwrap_or(0);
    u.jobs_queued = u64_at(&counters, "jobs_queued").unwrap_or(0);
    u.jobs_running = u64_at(&counters, "jobs_running").unwrap_or(0);
    u.throughput_per_min = u64_at(&counters, "throughput_per_min").unwrap_or(0);
    u.core_budget_total = u64_at(&counters, "core_budget_total").unwrap_or(0);
    u.core_budget_free = u64_at(&counters, "core_budget_free").unwrap_or(0);

    for node in snapshot
        .get("nodes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        let name = str_at(&node, "node").unwrap_or("?").to_owned();
        let target = u64_at(&node, "job_slot_target").unwrap_or(0);
        let running = u64_at(&node, "running_job_slots").unwrap_or(0);
        u.slots_total += target;
        u.slots_in_use += running;
        u.per_node.push(json!({
            "node": name,
            "up": node.get("up").and_then(Value::as_bool).unwrap_or(false),
            "health": str_at(&node, "health").unwrap_or("?"),
            "slots_in_use": running,
            "slots_target": target,
            "free_core_budget": u64_at(&node, "free_core_budget").unwrap_or(0),
            "running_jobs": u64_at(&node, "running_jobs").unwrap_or(0),
        }));
    }

    // Families give the eval lane specifically: total running can be inflated
    // by the DL trainer and by unrelated projects sharing the cluster.
    for family in snapshot
        .get("families")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        let name = str_at(&family, "family").unwrap_or("");
        if name.starts_with("jope-selfplay-eval") || name.starts_with("jope-coverage") {
            u.eval_running += u64_at(&family, "running_jobs").unwrap_or(0);
            u.eval_queued += u64_at(&family, "queued_jobs").unwrap_or(0);
        }
    }

    // `dispatching` is a real non-terminal state (seen live in `scq ps`) that
    // the snapshot counters fold into neither queued nor running.
    if let Some(jobs) = snapshot.get("jobs").and_then(Value::as_array) {
        u.jobs_dispatching = jobs
            .iter()
            .filter(|job| str_at(job, "state") == Some("dispatching"))
            .count() as u64;
    }
    u
}

// ---------------------------------------------------------------------------
// DETECT — work available on disk (independent of every daemon's opinion)
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct WorkOnDisk {
    /// Manifests sitting in the newest built waves. `--max-overlap-waves 8` is
    /// supply's own overlap ceiling, so 8 is the honest horizon.
    wave_manifests: u64,
    waves_scanned: u64,
    newest_wave: Option<String>,
    newest_wave_age_secs: Option<u64>,
    /// Datasets encoded (sidecars complete) but not yet in the live catalog.
    activation_backlog: u64,
    /// Last first-touch backlog the coverage builder reported.
    coverage_backlog_unclaimed: u64,
}

impl WorkOnDisk {
    fn total(&self) -> u64 {
        self.wave_manifests + self.activation_backlog + self.coverage_backlog_unclaimed
    }
}

fn survey_work(args: &Args) -> WorkOnDisk {
    let mut work = WorkOnDisk::default();
    let waves_root = args.root.join("waves");
    let mut waves: Vec<(u64, PathBuf)> = Vec::new();
    if let Ok(entries) = fs::read_dir(&waves_root) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(number) = name
                .strip_prefix("strict-wave-")
                .and_then(|n| n.parse::<u64>().ok())
            else {
                continue;
            };
            waves.push((number, entry.path()));
        }
    }
    waves.sort_by(|a, b| b.0.cmp(&a.0));
    if let Some((number, path)) = waves.first() {
        work.newest_wave = Some(format!("strict-wave-{number:06}"));
        if let Ok(modified) = path
            .join("summary.json")
            .metadata()
            .and_then(|m| m.modified())
        {
            work.newest_wave_age_secs = SystemTime::now()
                .duration_since(modified)
                .ok()
                .map(|d| d.as_secs());
        }
    }
    for (_, path) in waves.iter().take(8) {
        if let Ok(entries) = fs::read_dir(path.join("manifests")) {
            work.wave_manifests += entries.flatten().count() as u64;
            work.waves_scanned += 1;
        }
    }

    // Activation backlog: expansion's own stage summary, read as data. This is
    // the only number here sourced from another daemon's receipt, and it is
    // used solely to answer "is there more to do", never to judge health.
    if let Some(driver) = read_json(&args.root.join("ops/expansion-driver-v1.json")) {
        if let Some(stages) = driver.get("stage_summary").and_then(Value::as_object) {
            for (_task, stage) in stages {
                let encoded = u64_at(stage, "sidecars_complete").unwrap_or(0);
                let active = u64_at(stage, "catalog_active").unwrap_or(0);
                work.activation_backlog += encoded.saturating_sub(active);
            }
        }
    }

    if let Some(event) = last_coverage_event(args, "wave_built") {
        work.coverage_backlog_unclaimed = event
            .get("result")
            .and_then(|r| u64_at(r, "backlog_remaining"))
            .unwrap_or(0);
    }
    work
}

/// Tail the coverage lane's append-only event log for the newest event of a
/// kind. Bounded read: the lane writes ~2 rows every 3 min, so 256 KiB is
/// several days of history.
fn last_coverage_event(args: &Args, kind: &str) -> Option<Value> {
    let path = args.root.join("coverage/ops/coverage-events.v1.jsonl");
    let bytes = tail_bytes(&path, 256 * 1024)?;
    let text = String::from_utf8_lossy(&bytes);
    text.lines()
        .rev()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|value| str_at(value, "kind") == Some(kind))
}

fn tail_bytes(path: &Path, limit: u64) -> Option<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(path).ok()?;
    let length = file.metadata().ok()?.len();
    let start = length.saturating_sub(limit);
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).ok()?;
    Some(bytes)
}

// ---------------------------------------------------------------------------
// DIAGNOSE — which stage stopped producing
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Health {
    Producing,
    Erroring,
    Stale,
    Blocked,
    Drained,
    Down,
    Unknown,
}

impl Health {
    fn as_str(self) -> &'static str {
        match self {
            Health::Producing => "producing",
            Health::Erroring => "erroring",
            Health::Stale => "stale",
            Health::Blocked => "blocked",
            Health::Drained => "drained",
            Health::Down => "down",
            Health::Unknown => "unknown",
        }
    }
    fn healthy(self) -> bool {
        matches!(self, Health::Producing)
    }
}

#[derive(Debug)]
struct Stage {
    name: &'static str,
    health: Health,
    detail: String,
    /// Receipt this verdict came from, so the verdict is auditable.
    receipt: String,
    /// Bounded repair, if one exists for this stage. `None` means "there is no
    /// safe automatic repair" — escalate instead of improvising.
    repair: Option<Repair>,
}

#[derive(Clone, Debug)]
struct Repair {
    key: String,
    verb: &'static str,
    unit: &'static str,
}

impl Repair {
    fn restart(unit: &'static str) -> Self {
        Self {
            key: format!("restart:{unit}"),
            verb: "restart",
            unit,
        }
    }
    fn start(unit: &'static str) -> Self {
        Self {
            key: format!("start:{unit}"),
            verb: "start",
            unit,
        }
    }
}

fn content_age_secs(value: &Value, now: u64) -> Option<u64> {
    let generated = u64_at(value, "generated_at_ms")?;
    Some(now.saturating_sub(generated) / 1000)
}

/// Blocked-stage verdict when the newest cohort-supply state has sat at
/// `submit_cohort_proposer` for longer than any healthy dispatch would take.
/// Returns None when the proposer pipeline is moving (no stage row at all —
/// absence of a proposer stage means "not the bottleneck").
fn proposer_stage(args: &Args, now: u64) -> Option<Stage> {
    let state_dir = args.root.join("ops/cohort-supply-state-v2");
    let newest = std::fs::read_dir(&state_dir)
        .ok()?
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .max_by_key(|entry| {
            entry
                .metadata()
                .and_then(|meta| meta.modified())
                .unwrap_or(std::time::UNIX_EPOCH)
        })?;
    let age_secs = newest
        .metadata()
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| std::time::SystemTime::now().duration_since(modified).ok())
        .map(|age| age.as_secs())?;
    let value = read_json(&newest.path())?;
    if str_at(&value, "stage") != Some("submit_cohort_proposer") || age_secs <= 900 {
        return None;
    }
    let _ = now;
    Some(Stage {
        name: "proposer",
        health: Health::Blocked,
        detail: format!(
            "{} stuck at submit_cohort_proposer {age_secs}s; job {} waiting on 1 GPU — check `scq priority ls` for jope weight flattening and GPU occupancy",
            str_at(&value, "wave").unwrap_or("?"),
            str_at(&value, "proposal_job_id").unwrap_or("?"),
        ),
        receipt: format!(
            "ops/cohort-supply-state-v2/{}",
            newest.file_name().to_string_lossy()
        ),
        repair: None,
    })
}

/// Diagnose every stage, in pipeline order. Order matters for repair priority:
/// the stage closest to the scheduler is the one whose repair puts work on the
/// cluster soonest.
fn diagnose(args: &Args, util: &Utilization, work: &WorkOnDisk) -> Vec<Stage> {
    let now = now_ms();
    let mut stages = Vec::new();

    // ---- scq itself -------------------------------------------------------
    stages.push(if !util.scq_reachable {
        Stage {
            name: "scq",
            health: Health::Down,
            detail: util
                .scq_error
                .clone()
                .unwrap_or_else(|| "unreachable".into()),
            receipt: "scq --json snapshot".into(),
            // scqd lives in the scheduler checkout (JOPE_SCQ_ROOT) and is
            // not ours to restart. Escalate; do not improvise on the scheduler.
            repair: None,
        }
    } else if util.nodes_up == 0 {
        Stage {
            name: "scq",
            health: Health::Down,
            detail: format!("0 of {} nodes up", util.nodes_total),
            receipt: "scq --json snapshot".into(),
            repair: None,
        }
    } else {
        Stage {
            name: "scq",
            health: Health::Producing,
            detail: format!(
                "{}/{} nodes up, {} running, {} queued, {}/min",
                util.nodes_up,
                util.nodes_total,
                util.jobs_running,
                util.jobs_queued,
                util.throughput_per_min
            ),
            receipt: "scq --json snapshot".into(),
            repair: None,
        }
    });

    // ---- queue keeper -----------------------------------------------------
    // NOT an mtime check. The keeper rewrites this file on every failed cycle,
    // so mtime is always fresh; the `schema` field is what actually changes.
    let keeper_path = args.root.join("ops/queue-keeper-v3.json");
    let keeper_receipt = "ops/queue-keeper-v3.json".to_owned();
    stages.push(match read_json(&keeper_path) {
        None => Stage {
            name: "keeper",
            health: Health::Unknown,
            detail: "status file missing or unparsable".into(),
            receipt: keeper_receipt.clone(),
            repair: Some(Repair::restart("jope-selfplay-queue.service")),
        },
        Some(value) => {
            let schema = str_at(&value, "schema").unwrap_or("").to_owned();
            let age = content_age_secs(&value, now);
            if schema.ends_with("Error.v1") {
                Stage {
                    name: "keeper",
                    health: Health::Erroring,
                    detail: str_at(&value, "error").unwrap_or("unspecified").to_owned(),
                    receipt: keeper_receipt.clone(),
                    repair: Some(Repair::restart("jope-selfplay-queue.service")),
                }
            } else if age.is_some_and(|age| age > 900) {
                Stage {
                    name: "keeper",
                    health: Health::Stale,
                    detail: format!("status content {}s old (>900s)", age.unwrap_or(0)),
                    receipt: keeper_receipt.clone(),
                    repair: Some(Repair::restart("jope-selfplay-queue.service")),
                }
            } else if value.get("admission_allowed").and_then(Value::as_bool) == Some(false) {
                Stage {
                    name: "keeper",
                    health: Health::Blocked,
                    detail: "admission_allowed=false".into(),
                    receipt: keeper_receipt.clone(),
                    // Admission is gated on checkpoint freshness by design.
                    // Restarting cannot un-gate it, so do not pretend it can.
                    repair: None,
                }
            } else {
                Stage {
                    name: "keeper",
                    health: Health::Producing,
                    detail: format!(
                        "{} unclaimed manifests, {} starting, {} dispatching",
                        u64_at(&value, "unclaimed_current_checkpoint_manifests").unwrap_or(0),
                        u64_at(&value, "queued_starting").unwrap_or(0),
                        u64_at(&value, "dispatching").unwrap_or(0)
                    ),
                    receipt: keeper_receipt.clone(),
                    repair: None,
                }
            }
        }
    });

    // ---- supply / wave builder -------------------------------------------
    let supply_path = args.root.join("ops/micro-wave-supply-v2.json");
    let supply_receipt = "ops/micro-wave-supply-v2.json".to_owned();
    stages.push(match read_json(&supply_path) {
        None => Stage {
            name: "supply",
            health: Health::Unknown,
            detail: "status file missing or unparsable".into(),
            receipt: supply_receipt.clone(),
            repair: Some(Repair::restart("jope-selfplay-supply.service")),
        },
        Some(value) => {
            let schema = str_at(&value, "schema").unwrap_or("").to_owned();
            let age = content_age_secs(&value, now);
            if schema.ends_with("Error.v1") {
                Stage {
                    name: "supply",
                    health: Health::Erroring,
                    detail: str_at(&value, "error").unwrap_or("unspecified").to_owned(),
                    receipt: supply_receipt.clone(),
                    repair: Some(Repair::restart("jope-selfplay-supply.service")),
                }
            } else if age.is_some_and(|age| age > 300) {
                Stage {
                    name: "supply",
                    health: Health::Stale,
                    detail: format!("status content {}s old (>300s)", age.unwrap_or(0)),
                    receipt: supply_receipt.clone(),
                    repair: Some(Repair::restart("jope-selfplay-supply.service")),
                }
            } else {
                // `awaiting_keeper_ack` is in-memory state (supply blocks on a
                // keeper ack at jope_selfplay_supply.rs:350-365). If it has
                // been pending far longer than a keeper cycle the ack is never
                // coming and only a restart clears it.
                let waiting = u64_at(&value, "awaiting_keeper_ack_since_ms")
                    .map(|since| now.saturating_sub(since) / 1000);
                match waiting {
                    Some(seconds) if seconds > 900 => Stage {
                        name: "supply",
                        health: Health::Blocked,
                        detail: format!(
                            "awaiting keeper ack for {seconds}s on wave {}",
                            str_at(&value, "awaiting_keeper_ack").unwrap_or("?")
                        ),
                        receipt: supply_receipt.clone(),
                        repair: Some(Repair::restart("jope-selfplay-supply.service")),
                    },
                    _ => Stage {
                        name: "supply",
                        health: Health::Producing,
                        detail: format!(
                            "queue_depth {} below_floor {} ready_manifests {}",
                            u64_at(&value, "queue_depth").unwrap_or(0),
                            value
                                .get("below_floor")
                                .and_then(Value::as_bool)
                                .unwrap_or(false),
                            u64_at(&value, "ready_manifests").unwrap_or(0)
                        ),
                        receipt: supply_receipt.clone(),
                        repair: None,
                    },
                }
            }
        }
    });

    // ---- cohort proposer (GPU-arbitration starvation) ---------------------
    // 2026-08-01: the supervisor had no stage for "the cohort proposer is
    // GPU-starved", so when an external weight guard pinned jope families to
    // weight 1 and the proposer sat queued 3 h 12 m, this loop blamed
    // coverage/expansion for 1,173 consecutive cycles and burned 48 repair
    // attempts restarting innocent daemons. Waves cannot compose without the
    // proposer (>=8 fresh DL proposals per task; DL_SLOT_RUNGS has no 0), so
    // a wedged submit stage IS the pipeline outage. There is no safe
    // automatic repair — the fix is scheduling policy (scq priority ls) or
    // GPU capacity — so this stage escalates with the real cause instead of
    // improvising.
    if let Some(stage) = proposer_stage(args, now) {
        stages.push(stage);
    }

    // ---- coverage lane (breadth: first-touch) -----------------------------
    let coverage_receipt = "coverage/ops/coverage-events.v1.jsonl".to_owned();
    stages.push(match last_coverage_event(args, "submit") {
        None => Stage {
            name: "coverage",
            health: Health::Unknown,
            detail: "no submit events on record".into(),
            receipt: coverage_receipt.clone(),
            repair: Some(Repair::start("jope-coverage-lane.service")),
        },
        Some(event) => {
            let age = u64_at(&event, "ts_ms")
                .map(|ts| now.saturating_sub(ts) / 1000)
                .unwrap_or(u64::MAX);
            let result = event.get("result").cloned().unwrap_or(Value::Null);
            let live = u64_at(&result, "live_before").unwrap_or(0);
            let submitted = u64_at(&result, "submitted").unwrap_or(0);
            if age > 900 {
                // The lane is a 3-minute timer; 5 missed cycles is a stop.
                Stage {
                    name: "coverage",
                    health: Health::Stale,
                    detail: format!("last submit cycle {age}s ago (>900s)"),
                    receipt: coverage_receipt.clone(),
                    repair: Some(Repair::start("jope-coverage-lane.service")),
                }
            } else if live == 0 && submitted == 0 && work.coverage_backlog_unclaimed == 0 {
                // Cycling fine, nothing to send: the lane is starved from
                // upstream (expansion has activated no new datasets). Its own
                // repair is useless here; the expansion stage carries it.
                Stage {
                    name: "coverage",
                    health: Health::Drained,
                    detail: "first-touch backlog empty; starved by expansion".into(),
                    receipt: coverage_receipt.clone(),
                    repair: None,
                }
            } else {
                Stage {
                    name: "coverage",
                    health: Health::Producing,
                    detail: format!("{live} in flight, {submitted} submitted last cycle"),
                    receipt: coverage_receipt.clone(),
                    repair: None,
                }
            }
        }
    });

    // ---- expansion (breadth: activate new datasets) -----------------------
    let expansion_path = args.root.join("ops/expansion-driver-v1.json");
    let expansion_receipt = "ops/expansion-driver-v1.json".to_owned();
    stages.push(match read_json(&expansion_path) {
        None => Stage {
            name: "expansion",
            health: Health::Unknown,
            detail: "status file missing or unparsable".into(),
            receipt: expansion_receipt.clone(),
            repair: Some(Repair::start("jope-selfplay-expansion.service")),
        },
        Some(value) => {
            let age = content_age_secs(&value, now).unwrap_or(u64::MAX);
            let blocked = str_at(&value, "guardrail_block").map(str::to_owned);
            if let Some(reason) = blocked {
                Stage {
                    name: "expansion",
                    health: Health::Blocked,
                    detail: format!("guardrail: {reason}"),
                    receipt: expansion_receipt.clone(),
                    // Re-triggering a guardrailed driver just re-evaluates the
                    // same guardrail. Only useful if the backlog is real AND
                    // the cluster is idle; the planner below decides.
                    repair: Some(Repair::start("jope-selfplay-expansion.service")),
                }
            } else if age > 7_200 {
                Stage {
                    name: "expansion",
                    health: Health::Stale,
                    detail: format!("last cycle {age}s ago (>7200s)"),
                    receipt: expansion_receipt.clone(),
                    repair: Some(Repair::start("jope-selfplay-expansion.service")),
                }
            } else {
                Stage {
                    name: "expansion",
                    health: Health::Producing,
                    detail: format!(
                        "{} datasets encoded but not activated",
                        work.activation_backlog
                    ),
                    receipt: expansion_receipt.clone(),
                    repair: None,
                }
            }
        }
    });

    // ---- wave builder output (files, not opinions) ------------------------
    stages.push(match work.newest_wave_age_secs {
        None => Stage {
            name: "waves",
            health: Health::Unknown,
            detail: "no built wave on disk".into(),
            receipt: "waves/strict-wave-*/summary.json".into(),
            repair: None,
        },
        Some(age) if age > 10_800 && work.wave_manifests == 0 => Stage {
            name: "waves",
            health: Health::Drained,
            detail: format!(
                "newest wave {} is {age}s old and the newest {} waves hold 0 manifests",
                work.newest_wave.clone().unwrap_or_default(),
                work.waves_scanned
            ),
            receipt: "waves/strict-wave-*/manifests".into(),
            repair: Some(Repair::restart("jope-selfplay-supply.service")),
        },
        Some(age) => Stage {
            name: "waves",
            health: Health::Producing,
            detail: format!(
                "{} manifests across newest {} waves; newest {} is {age}s old",
                work.wave_manifests,
                work.waves_scanned,
                work.newest_wave.clone().unwrap_or_default()
            ),
            receipt: "waves/strict-wave-*/manifests".into(),
            repair: None,
        },
    });

    stages
}

// ---------------------------------------------------------------------------
// VERDICT
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Busy and every producer is producing.
    Ok,
    /// Not busy, and there is genuinely nothing left to run. Not an alarm.
    IdleNoWork,
    /// Busy right now, but a producer is broken: the cluster is coasting on
    /// in-flight work and will fall off a cliff when it drains. This is the
    /// early-warning state that would have caught 2026-07-28 at 10:19Z, when
    /// the queue read 3 queued / 258 running twenty minutes before zero.
    Degraded,
    /// Not busy, and work is sitting on disk. The condition the owner found by
    /// looking at a screen.
    Idle,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Verdict::Ok => "ok",
            Verdict::IdleNoWork => "idle_no_work",
            Verdict::Degraded => "degraded",
            Verdict::Idle => "idle",
        }
    }
}

fn judge(args: &Args, util: &Utilization, work: &WorkOnDisk, stages: &[Stage]) -> Verdict {
    let busy = if args.simulate_idle {
        false
    } else {
        util.scq_reachable && util.core_pct() >= args.idle_core_pct
    };
    let producers_ok = stages.iter().all(|stage| stage.health.healthy());
    let under_supplied = util.queue_depth() < effective_queue_target(args, util);
    if !busy {
        return if work.total() > 0 {
            Verdict::Idle
        } else {
            Verdict::IdleNoWork
        };
    }
    if !producers_ok && under_supplied {
        return Verdict::Degraded;
    }
    if !producers_ok {
        // Busy AND deeply queued: a broken producer has not bitten yet, but it
        // is still a defect worth counting toward escalation.
        return Verdict::Degraded;
    }
    Verdict::Ok
}

fn effective_queue_target(args: &Args, util: &Utilization) -> u64 {
    let slot_target = util.slots_total.saturating_mul(5).saturating_add(3) / 4;
    args.queue_floor_jobs.max(slot_target).min(4096)
}

// ---------------------------------------------------------------------------
// REPAIR — bounded, receipted, backed off
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct State {
    consecutive_idle: u32,
    consecutive_degraded: u32,
    escalated: bool,
    escalated_at_ms: u64,
    last_notify_ms: u64,
    /// action key -> (consecutive attempts, last attempt ms)
    attempts: BTreeMap<String, (u32, u64)>,
    /// epoch-ms of every action taken, trimmed to the last hour.
    action_log: Vec<u64>,
}

fn load_state(path: &Path) -> State {
    let Some(value) = read_json(path) else {
        return State::default();
    };
    let mut state = State {
        consecutive_idle: u64_at(&value, "consecutive_idle_cycles").unwrap_or(0) as u32,
        consecutive_degraded: u64_at(&value, "consecutive_degraded_cycles").unwrap_or(0) as u32,
        escalated: value
            .get("escalated")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        escalated_at_ms: u64_at(&value, "escalated_at_ms").unwrap_or(0),
        last_notify_ms: u64_at(&value, "last_notify_ms").unwrap_or(0),
        ..State::default()
    };
    if let Some(map) = value.get("attempts").and_then(Value::as_object) {
        for (key, entry) in map {
            state.attempts.insert(
                key.clone(),
                (
                    u64_at(entry, "attempts").unwrap_or(0) as u32,
                    u64_at(entry, "last_ms").unwrap_or(0),
                ),
            );
        }
    }
    if let Some(list) = value.get("action_log").and_then(Value::as_array) {
        state.action_log = list.iter().filter_map(Value::as_u64).collect();
    }
    state
}

fn save_state(path: &Path, state: &State) -> Result<()> {
    let attempts: serde_json::Map<String, Value> = state
        .attempts
        .iter()
        .map(|(key, (attempts, last))| {
            (
                key.clone(),
                json!({ "attempts": attempts, "last_ms": last }),
            )
        })
        .collect();
    write_json_atomic(
        path,
        &json!({
            "schema": "JopeUtilizationSupervisorState.v1",
            "consecutive_idle_cycles": state.consecutive_idle,
            "consecutive_degraded_cycles": state.consecutive_degraded,
            "escalated": state.escalated,
            "escalated_at_ms": state.escalated_at_ms,
            "last_notify_ms": state.last_notify_ms,
            "attempts": Value::Object(attempts),
            "action_log": state.action_log,
        }),
    )
}

fn backoff_ms(args: &Args, attempts: u32) -> u64 {
    if attempts == 0 {
        return 0;
    }
    let shift = (attempts - 1).min(16);
    let delay = args.backoff_base_seconds.saturating_mul(1_u64 << shift);
    delay.min(args.backoff_max_seconds) * 1000
}

/// Choose at most ONE repair per cycle. One action per cycle is the whole
/// bounding argument: with a 120 s timer the ceiling is 30 actions/hour before
/// backoff, and `--max-actions-per-hour` clamps it to 6 after.
fn plan_repair<'a>(
    args: &Args,
    verdict: Verdict,
    stages: &'a [Stage],
    state: &State,
    now: u64,
) -> (Option<&'a Stage>, Option<String>) {
    if matches!(verdict, Verdict::Ok | Verdict::IdleNoWork) {
        return (None, Some("objective met; no repair considered".into()));
    }
    let recent = state
        .action_log
        .iter()
        .filter(|ms| now.saturating_sub(**ms) < HOUR_MS)
        .count();
    if recent >= args.max_actions_per_hour {
        return (
            None,
            Some(format!(
                "suppressed: {recent} actions in the last hour >= --max-actions-per-hour {}",
                args.max_actions_per_hour
            )),
        );
    }
    let mut deferred: Vec<String> = Vec::new();
    for stage in stages {
        if stage.health.healthy() {
            continue;
        }
        let Some(repair) = stage.repair.as_ref() else {
            deferred.push(format!("{}: no safe automatic repair", stage.name));
            continue;
        };
        // Re-triggering expansion is only worth an action when the cluster is
        // actually idle. While work is flowing, a guardrail block is a policy
        // decision, not an outage.
        if stage.name == "expansion" && verdict != Verdict::Idle {
            deferred.push("expansion: blocked but cluster is not idle".into());
            continue;
        }
        let (attempts, last) = state.attempts.get(&repair.key).copied().unwrap_or((0, 0));
        let wait = backoff_ms(args, attempts);
        if last > 0 && now.saturating_sub(last) < wait {
            deferred.push(format!(
                "{}: backoff {}s remaining after {attempts} attempts",
                repair.key,
                (wait - now.saturating_sub(last)) / 1000
            ));
            continue;
        }
        return (
            Some(stage),
            (!deferred.is_empty()).then(|| deferred.join("; ")),
        );
    }
    (
        None,
        Some(if deferred.is_empty() {
            "no unhealthy stage carries a repair".into()
        } else {
            deferred.join("; ")
        }),
    )
}

fn run_repair(args: &Args, repair: &Repair) -> (bool, String) {
    if args.dry_run {
        return (
            false,
            format!(
                "dry-run: would systemctl --user {} {}",
                repair.verb, repair.unit
            ),
        );
    }
    let output = Command::new("systemctl")
        .arg("--user")
        .arg(repair.verb)
        .arg(repair.unit)
        .output();
    match output {
        Ok(o) if o.status.success() => (
            true,
            format!("systemctl --user {} {} ok", repair.verb, repair.unit),
        ),
        Ok(o) => (
            false,
            format!(
                "systemctl --user {} {} exit {:?}: {}",
                repair.verb,
                repair.unit,
                o.status.code(),
                String::from_utf8_lossy(&o.stderr).trim()
            ),
        ),
        Err(error) => (false, format!("systemctl spawn failed: {error}")),
    }
}

// ---------------------------------------------------------------------------
// ESCALATE
// ---------------------------------------------------------------------------

/// What is actually available on this box, verified 2026-07-28:
///   - NO mail transport, NO ntfy/pushover/telegram/slack config, and no
///     webhook credential anywhere in the tree.
///   - `notify-send` is installed and DBUS_SESSION_BUS_ADDRESS is set, but no
///     process owns org.freedesktop.Notifications, so desktop toasts are
///     swallowed. Attempted anyway and recorded as delivered=false.
///   - `wall` reaches all logged-in ptys (47 at the time of writing). This is
///     the loudest thing that exists here and it is what the owner will see
///     from any terminal on the box, not just the one showing the TUI.
///   - `logger` -> journald, greppable after the fact.
///   - the TUI's alert panel, fed from ops/utilization-alarms.v1.json.
/// Every notification runs under `timeout` so no delivery channel can stall a
/// cycle. A supervisor blocked inside its own alarm is indistinguishable from
/// the daemon it was watching.
fn notify(argv: &[&str]) -> bool {
    Command::new("timeout")
        .arg("10")
        .args(argv)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn escalate(args: &Args, summary: &str, detail: &str) -> Vec<Value> {
    let mut channels = Vec::new();
    if args.dry_run {
        channels.push(
            json!({"channel": "dry-run", "delivered": false, "note": "no notification sent"}),
        );
        return channels;
    }

    // journald, critical priority.
    let logged = notify(&[
        "logger",
        "-p",
        "user.crit",
        "-t",
        "jope-utilization",
        summary,
    ]);
    channels.push(json!({
        "channel": "journald",
        "delivered": logged,
        "note": "journalctl --user -t jope-utilization -p crit"
    }));

    // `wall` was removed by operator request 2026-07-28: it interrupts EVERY
    // logged-in terminal on the box, including interactive work unrelated to
    // the cluster, and it fires on a 120 s timer so a sustained degradation
    // spams continuously. Escalation now goes to journald (above), the desktop
    // toast (below), the health-alarms file and the TUI banner — all of which
    // are visible without hijacking a shell. Do not reintroduce a broadcast
    // channel here; if escalation needs to be louder, make it push to a device
    // the operator carries, not to every pty.
    let _ = &detail;

    // desktop toast, best effort.
    let notified = notify(&[
        "notify-send",
        "-u",
        "critical",
        "-a",
        "jope-flywheel",
        "JOPE cluster idle",
        summary,
    ]);
    channels.push(json!({
        "channel": "notify-send",
        "delivered": notified,
        "note": "no org.freedesktop.Notifications owner on this box; expected false"
    }));

    // tmux banner into the live TUI session.
    let tmuxed = notify(&[
        "tmux",
        "-L",
        "jope",
        "display-message",
        "-d",
        "0",
        &format!("JOPE UTILIZATION ALARM: {summary}"),
    ]);
    channels.push(json!({"channel": "tmux:jope-tui", "delivered": tmuxed, "note": "sticky status-line banner"}));

    // On-disk banner, for anyone who lands in ops/ after the fact.
    let banner = args.banner();
    let wrote = fs::write(
        &banner,
        format!("{}\n\n{summary}\n\n{detail}\n", iso_utc(now_ms())),
    )
    .is_ok();
    channels.push(
        json!({"channel": "banner_file", "delivered": wrote, "note": banner.display().to_string()}),
    );

    channels
}

fn write_alarms(
    args: &Args,
    verdict: Verdict,
    escalated: bool,
    summary: &str,
    detail: &str,
) -> Result<()> {
    // DELIBERATELY NOT ops/health-alarms.v1.json. jope_selfplay_expansion_driver.rs
    // :116-141 halts dataset activation on any critical row in that file. A
    // critical `cluster_idle` row there would stop the flywheel activating
    // datasets *because the cluster is idle*, which is precisely the starvation
    // loop that emptied all 1,052 slots on 2026-07-28.
    let alarms = if escalated {
        vec![json!({
            "alarm": "cluster_idle",
            "severity": "critical",
            "observed": summary,
            "threshold": format!("core budget >= {:.0}% with producers healthy", args.idle_core_pct),
            "detail": detail,
            "at_ms": now_ms(),
        })]
    } else if verdict == Verdict::Degraded {
        vec![json!({
            "alarm": "cluster_supply_degraded",
            "severity": "warning",
            "observed": summary,
            "threshold": "all producer stages producing",
            "detail": detail,
            "at_ms": now_ms(),
        })]
    } else {
        Vec::new()
    };
    write_json_atomic(
        &args.alarms(),
        &json!({
            "schema": "JopeUtilizationAlarms.v1",
            "generated_at_ms": now_ms(),
            "alarm_count": alarms.len(),
            "alarms": alarms,
        }),
    )
}

// ---------------------------------------------------------------------------
// RECORD — historical utilization
// ---------------------------------------------------------------------------

fn report(args: &Args, hours: f64) -> Result<()> {
    let path = args.series();
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let cutoff = now_ms().saturating_sub((hours * 3_600_000.0) as u64);
    let mut rows: Vec<Value> = String::from_utf8_lossy(&bytes)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|row| u64_at(row, "ts_ms").unwrap_or(0) >= cutoff)
        .collect();
    rows.sort_by_key(|row| u64_at(row, "ts_ms").unwrap_or(0));
    if rows.is_empty() {
        println!("no samples in the last {hours} h ({})", path.display());
        return Ok(());
    }
    let samples = rows.len() as f64;
    let mean_core: f64 = rows
        .iter()
        .filter_map(|r| f64_at(r, "core_pct"))
        .sum::<f64>()
        / samples;
    let mean_slots: f64 = rows
        .iter()
        .filter_map(|r| f64_at(r, "slot_pct"))
        .sum::<f64>()
        / samples;
    let mean_running: f64 = rows
        .iter()
        .filter_map(|r| u64_at(r, "jobs_running"))
        .sum::<u64>() as f64
        / samples;

    // Contiguous runs of non-ok verdicts, which is what "idle gap" means.
    let mut gaps: Vec<(u64, u64, String)> = Vec::new();
    for row in &rows {
        let ts = u64_at(row, "ts_ms").unwrap_or(0);
        let verdict = str_at(row, "verdict").unwrap_or("?").to_owned();
        let bad = verdict == "idle" || verdict == "idle_no_work";
        match gaps.last_mut() {
            Some(last) if bad && last.2 == verdict && ts.saturating_sub(last.1) <= 600_000 => {
                last.1 = ts;
            }
            _ if bad => gaps.push((ts, ts, verdict)),
            _ => {}
        }
    }
    println!(
        "utilization report  window={hours}h  samples={}",
        rows.len()
    );
    println!(
        "  first sample  {}",
        iso_utc(u64_at(&rows[0], "ts_ms").unwrap_or(0))
    );
    println!(
        "  last sample   {}",
        iso_utc(u64_at(rows.last().unwrap(), "ts_ms").unwrap_or(0))
    );
    println!("  mean core-budget occupancy  {mean_core:.1}%");
    println!("  mean slot occupancy         {mean_slots:.1}%");
    println!("  mean running jobs           {mean_running:.0}");
    let ok = rows
        .iter()
        .filter(|r| str_at(r, "verdict") == Some("ok"))
        .count();
    println!("  cycles ok / total           {ok}/{}", rows.len());
    if gaps.is_empty() {
        println!("  idle gaps: none");
    } else {
        println!("  idle gaps ({}):", gaps.len());
        for (start, end, verdict) in &gaps {
            println!(
                "    {} -> {}  {:>5} min  [{verdict}]",
                iso_utc(*start),
                iso_utc(*end),
                (end.saturating_sub(*start)) / 60_000
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// main cycle
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(hours) = args.report_hours {
        return report(&args, hours);
    }

    let now = now_ms();
    let util = measure(&args);
    let work = survey_work(&args);
    let stages = diagnose(&args, &util, &work);
    let verdict = judge(&args, &util, &work, &stages);

    let mut state = load_state(&args.state());
    state
        .action_log
        .retain(|ms| now.saturating_sub(*ms) < HOUR_MS);

    match verdict {
        Verdict::Ok | Verdict::IdleNoWork => {
            state.consecutive_idle = 0;
            state.consecutive_degraded = 0;
        }
        Verdict::Idle => {
            state.consecutive_idle += 1;
        }
        Verdict::Degraded => {
            state.consecutive_idle = 0;
            state.consecutive_degraded += 1;
        }
    }

    // The first unhealthy stage in pipeline order is the one being blamed; the
    // rest may be downstream symptoms of it.
    let culprit = stages.iter().find(|stage| !stage.health.healthy());

    let (repair_stage, suppression) = plan_repair(&args, verdict, &stages, &state, now);
    let mut action = Value::Null;
    if let Some(stage) = repair_stage {
        let repair = stage
            .repair
            .as_ref()
            .expect("planner only returns repairable stages");
        let (ok, note) = run_repair(&args, repair);
        if !args.dry_run {
            let entry = state.attempts.entry(repair.key.clone()).or_insert((0, 0));
            entry.0 += 1;
            entry.1 = now;
            state.action_log.push(now);
        }
        action = json!({
            "stage": stage.name,
            "stage_health": stage.health.as_str(),
            "stage_detail": stage.detail,
            "action": repair.key,
            "succeeded": ok,
            "note": note,
            "attempt_no": state.attempts.get(&repair.key).map(|entry| entry.0).unwrap_or(1),
        });
        append_jsonl(
            &args.events(),
            &json!({
                "schema": "JopeUtilizationSupervisorEvent.v1",
                "ts_ms": now, "ts": iso_utc(now), "kind": "repair",
                "verdict": verdict.as_str(), "action": action.clone(),
            }),
        )?;
    }

    // ---- escalation --------------------------------------------------------
    let should_escalate = (verdict == Verdict::Idle
        && state.consecutive_idle >= args.escalate_idle_cycles)
        || (verdict == Verdict::Degraded
            && state.consecutive_degraded >= args.escalate_degraded_cycles);

    let summary =
        format!(
        "cluster {} — {:.0}% core budget ({} of {} slots busy, {} running, {} queued, {}/min); \
         culprit={} ({}); work waiting: {} wave manifests + {} datasets to activate",
        verdict.as_str(),
        util.core_pct(),
        util.slots_in_use,
        util.slots_total,
        util.jobs_running,
        util.jobs_queued,
        util.throughput_per_min,
        culprit.map(|stage| stage.name).unwrap_or("none"),
        culprit.map(|stage| stage.health.as_str()).unwrap_or("healthy"),
        work.wave_manifests,
        work.activation_backlog,
    );
    let detail = stages
        .iter()
        .map(|stage| {
            format!(
                "  {:<10} {:<10} {}",
                stage.name,
                stage.health.as_str(),
                stage.detail
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut channels = Vec::new();
    if should_escalate {
        let first_time = !state.escalated;
        let due = now.saturating_sub(state.last_notify_ms) >= args.renotify_seconds * 1000;
        if first_time || due {
            channels = escalate(&args, &summary, &detail);
            if !args.dry_run {
                state.last_notify_ms = now;
            }
            append_jsonl(
                &args.events(),
                &json!({
                    "schema": "JopeUtilizationSupervisorEvent.v1",
                    "ts_ms": now, "ts": iso_utc(now),
                    "kind": if first_time { "escalated" } else { "escalation_renotify" },
                    "verdict": verdict.as_str(), "summary": summary, "detail": detail,
                    "channels": channels.clone(),
                }),
            )?;
        }
        state.escalated = true;
        if state.escalated_at_ms == 0 {
            state.escalated_at_ms = now;
        }
    } else if state.escalated && verdict == Verdict::Ok {
        // Recovery is as newsworthy as the outage: an unanswered alarm that
        // silently clears is how the next one gets ignored.
        let recovery = format!(
            "RECOVERED after {}: {summary}",
            human_span(now.saturating_sub(state.escalated_at_ms))
        );
        if !args.dry_run {
            notify(&[
                "logger",
                "-p",
                "user.notice",
                "-t",
                "jope-utilization",
                &recovery,
            ]);
            // No `wall` here either — see the escalation path above.
            fs::remove_file(args.banner()).ok();
        }
        append_jsonl(
            &args.events(),
            &json!({
                "schema": "JopeUtilizationSupervisorEvent.v1",
                "ts_ms": now, "ts": iso_utc(now), "kind": "recovered",
                "verdict": verdict.as_str(), "summary": recovery,
            }),
        )?;
        state.escalated = false;
        state.escalated_at_ms = 0;
        state.last_notify_ms = 0;
        state.attempts.clear();
    }

    write_alarms(&args, verdict, should_escalate, &summary, &detail)?;

    // ---- receipts ----------------------------------------------------------
    let status = json!({
        "schema": "JopeUtilizationSupervisor.v1",
        "generated_at_ms": now,
        "generated_at": iso_utc(now),
        "verdict": verdict.as_str(),
        "summary": summary,
        "objective": {
            "idle_core_pct": args.idle_core_pct,
            "queue_floor_jobs": effective_queue_target(&args, &util),
            "configured_queue_floor_jobs": args.queue_floor_jobs,
            "escalate_idle_cycles": args.escalate_idle_cycles,
            "escalate_degraded_cycles": args.escalate_degraded_cycles,
            "max_actions_per_hour": args.max_actions_per_hour,
        },
        "utilization": {
            "scq_reachable": util.scq_reachable,
            "scq_error": util.scq_error,
            "scq_epoch": util.epoch,
            "nodes_up": util.nodes_up,
            "nodes_total": util.nodes_total,
            "slots_in_use": util.slots_in_use,
            "slots_total": util.slots_total,
            "slot_pct": (util.slot_pct() * 100.0).round() / 100.0,
            "core_budget_total": util.core_budget_total,
            "core_budget_free": util.core_budget_free,
            "core_pct": (util.core_pct() * 100.0).round() / 100.0,
            "jobs_queued": util.jobs_queued,
            "jobs_running": util.jobs_running,
            "jobs_dispatching": util.jobs_dispatching,
            "eval_running": util.eval_running,
            "eval_queued": util.eval_queued,
            "throughput_per_min": util.throughput_per_min,
            "per_node": util.per_node,
        },
        "work_available": {
            "wave_manifests": work.wave_manifests,
            "waves_scanned": work.waves_scanned,
            "newest_wave": work.newest_wave,
            "newest_wave_age_secs": work.newest_wave_age_secs,
            "activation_backlog_datasets": work.activation_backlog,
            "coverage_backlog_unclaimed": work.coverage_backlog_unclaimed,
            "total": work.total(),
        },
        "stages": stages.iter().map(|stage| json!({
            "stage": stage.name,
            "health": stage.health.as_str(),
            "detail": stage.detail,
            "receipt": stage.receipt,
            "repair": stage.repair.as_ref().map(|repair| repair.key.clone()),
        })).collect::<Vec<_>>(),
        "culprit": culprit.map(|stage| stage.name),
        "consecutive_idle_cycles": state.consecutive_idle,
        "consecutive_degraded_cycles": state.consecutive_degraded,
        "action": action,
        "action_suppressed": suppression,
        "actions_last_hour": state.action_log.len(),
        "escalated": state.escalated,
        "escalation_channels": channels,
        "dry_run": args.dry_run,
        "simulate_idle": args.simulate_idle,
    });
    write_json_atomic(&args.status(), &status)?;

    append_jsonl(
        &args.series(),
        &json!({
            "schema": "JopeUtilizationSample.v1",
            "ts_ms": now,
            "ts": iso_utc(now),
            "verdict": verdict.as_str(),
            "core_pct": (util.core_pct() * 100.0).round() / 100.0,
            "slot_pct": (util.slot_pct() * 100.0).round() / 100.0,
            "slots_in_use": util.slots_in_use,
            "slots_total": util.slots_total,
            "jobs_running": util.jobs_running,
            "jobs_queued": util.jobs_queued,
            "eval_running": util.eval_running,
            "throughput_per_min": util.throughput_per_min,
            "work_total": work.total(),
            "culprit": culprit.map(|stage| stage.name),
            "action": action.get("action").cloned(),
        }),
    )?;

    if !args.dry_run || args.state_file.is_some() {
        save_state(&args.state(), &state)?;
    }

    println!("{}", serde_json::to_string(&status)?);
    // Always exit 0. A supervisor whose own unit can fail is one more thing
    // that needs a supervisor.
    Ok(())
}

fn human_span(ms: u64) -> String {
    let secs = ms / 1000;
    if secs < 120 {
        format!("{secs}s")
    } else if secs < 7200 {
        format!("{}m", secs / 60)
    } else {
        format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args::parse_from(["jope_utilization_supervisor"])
    }

    fn stage(name: &'static str, health: Health) -> Stage {
        Stage {
            name,
            health,
            detail: String::new(),
            receipt: String::new(),
            repair: Some(Repair::restart("unit.service")),
        }
    }

    #[test]
    fn iso_utc_matches_known_epoch() {
        assert_eq!(iso_utc(1_785_251_604_304), "2026-07-28T15:13:24Z");
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn core_pct_uses_budget_not_slot_target() {
        // Live 2026-07-28T15:18Z reading: 1,052 slot targets, 317 slots busy,
        // but only 1 of 344 core-budget units free. Slot occupancy says 30%;
        // the cluster was in fact saturated. Judging on slots would have made
        // this supervisor cry idle on a full cluster.
        let util = Utilization {
            slots_in_use: 317,
            slots_total: 1052,
            core_budget_total: 344,
            core_budget_free: 1,
            ..Utilization::default()
        };
        assert!((util.slot_pct() - 30.13).abs() < 0.1);
        assert!(util.core_pct() > 99.0);
    }

    #[test]
    fn idle_with_work_is_idle_and_without_work_is_not_an_alarm() {
        let a = args();
        let util = Utilization {
            scq_reachable: true,
            core_budget_total: 344,
            core_budget_free: 344,
            ..Utilization::default()
        };
        let stages = vec![stage("keeper", Health::Producing)];
        let with_work = WorkOnDisk {
            wave_manifests: 7040,
            ..WorkOnDisk::default()
        };
        assert_eq!(judge(&a, &util, &with_work, &stages), Verdict::Idle);
        assert_eq!(
            judge(&a, &util, &WorkOnDisk::default(), &stages),
            Verdict::IdleNoWork
        );
    }

    #[test]
    fn busy_but_broken_producer_is_degraded() {
        // 2026-07-28T15:18Z: 552 active jobs and a saturated core budget while
        // BOTH the keeper and supply were in error loops. "Busy" alone must not
        // read as healthy or the cliff arrives unannounced.
        let a = args();
        let util = Utilization {
            scq_reachable: true,
            core_budget_total: 344,
            core_budget_free: 1,
            jobs_queued: 73,
            jobs_running: 315,
            ..Utilization::default()
        };
        let stages = vec![
            stage("scq", Health::Producing),
            stage("keeper", Health::Erroring),
        ];
        let work = WorkOnDisk {
            wave_manifests: 7040,
            ..WorkOnDisk::default()
        };
        assert_eq!(judge(&a, &util, &work, &stages), Verdict::Degraded);
    }

    #[test]
    fn all_producing_and_busy_is_ok() {
        let a = args();
        let util = Utilization {
            scq_reachable: true,
            core_budget_total: 344,
            core_budget_free: 20,
            jobs_queued: 900,
            ..Utilization::default()
        };
        let stages = vec![
            stage("scq", Health::Producing),
            stage("keeper", Health::Producing),
        ];
        assert_eq!(
            judge(
                &a,
                &util,
                &WorkOnDisk {
                    wave_manifests: 10,
                    ..WorkOnDisk::default()
                },
                &stages
            ),
            Verdict::Ok
        );
    }

    #[test]
    fn backoff_doubles_then_saturates() {
        let a = args();
        assert_eq!(backoff_ms(&a, 0), 0);
        assert_eq!(backoff_ms(&a, 1), 300_000);
        assert_eq!(backoff_ms(&a, 2), 600_000);
        assert_eq!(backoff_ms(&a, 4), 2_400_000);
        assert_eq!(backoff_ms(&a, 9), 3_600_000); // clamped at --backoff-max-seconds
        assert_eq!(backoff_ms(&a, 40), 3_600_000); // shift is clamped, no overflow
    }

    #[test]
    fn hourly_cap_suppresses_every_action() {
        let a = args();
        let now = now_ms();
        let state = State {
            action_log: vec![now; 6],
            ..State::default()
        };
        let stages = vec![stage("keeper", Health::Erroring)];
        let (chosen, note) = plan_repair(&a, Verdict::Idle, &stages, &state, now);
        assert!(chosen.is_none());
        assert!(note.unwrap().contains("max-actions-per-hour"));
    }

    #[test]
    fn backoff_defers_a_repeat_of_the_same_action() {
        let a = args();
        let now = now_ms();
        let stages = vec![stage("keeper", Health::Erroring)];
        let mut state = State::default();
        let key = stages[0].repair.as_ref().unwrap().key.clone();
        state.attempts.insert(key, (1, now - 10_000));
        let (chosen, note) = plan_repair(&a, Verdict::Idle, &stages, &state, now);
        assert!(chosen.is_none());
        assert!(note.unwrap().contains("backoff"));
    }

    #[test]
    fn one_repair_per_cycle_and_pipeline_order_wins() {
        let a = args();
        let stages = vec![
            stage("scq", Health::Producing),
            stage("keeper", Health::Erroring),
            stage("supply", Health::Erroring),
        ];
        let (chosen, _) = plan_repair(&a, Verdict::Idle, &stages, &State::default(), now_ms());
        assert_eq!(chosen.map(|stage| stage.name), Some("keeper"));
    }

    #[test]
    fn expansion_retrigger_waits_until_the_cluster_is_actually_idle() {
        let a = args();
        let stages = vec![stage("expansion", Health::Blocked)];
        let (chosen, note) =
            plan_repair(&a, Verdict::Degraded, &stages, &State::default(), now_ms());
        assert!(chosen.is_none());
        assert!(note.unwrap().contains("not idle"));
        let (chosen, _) = plan_repair(&a, Verdict::Idle, &stages, &State::default(), now_ms());
        assert_eq!(chosen.map(|stage| stage.name), Some("expansion"));
    }

    #[test]
    fn a_stage_without_a_safe_repair_is_never_improvised_on() {
        let a = args();
        let stages = vec![Stage {
            name: "scq",
            health: Health::Down,
            detail: String::new(),
            receipt: String::new(),
            repair: None,
        }];
        let (chosen, note) = plan_repair(&a, Verdict::Idle, &stages, &State::default(), now_ms());
        assert!(chosen.is_none());
        assert!(note.unwrap().contains("no safe automatic repair"));
    }

    #[test]
    fn healthy_verdicts_never_plan_a_repair() {
        let a = args();
        let stages = vec![stage("keeper", Health::Erroring)];
        for verdict in [Verdict::Ok, Verdict::IdleNoWork] {
            assert!(
                plan_repair(&a, verdict, &stages, &State::default(), now_ms())
                    .0
                    .is_none()
            );
        }
    }

    #[test]
    fn simulate_idle_forces_the_idle_verdict_on_a_saturated_cluster() {
        let mut a = args();
        a.simulate_idle = true;
        let util = Utilization {
            scq_reachable: true,
            core_budget_total: 344,
            core_budget_free: 0,
            ..Utilization::default()
        };
        let stages = vec![stage("keeper", Health::Producing)];
        let work = WorkOnDisk {
            wave_manifests: 7040,
            ..WorkOnDisk::default()
        };
        assert_eq!(judge(&a, &util, &work, &stages), Verdict::Idle);
    }
}
