use super::*;

#[derive(Debug, Default)]
pub(crate) struct Utilization {
    pub(crate) scq_reachable: bool,
    pub(crate) scq_error: Option<String>,
    pub(crate) epoch: u64,
    pub(crate) nodes_up: u64,
    pub(crate) nodes_total: u64,
    /// Sum of per-node `running_job_slots` — the operator's "N of 1,052 slots".
    pub(crate) slots_in_use: u64,
    pub(crate) slots_total: u64,
    /// Core budget is the constraint that actually binds; slot targets are a
    /// 4x multiplier over it on the 3 big nodes.
    pub(crate) core_budget_total: u64,
    pub(crate) core_budget_free: u64,
    pub(crate) jobs_queued: u64,
    pub(crate) jobs_running: u64,
    pub(crate) jobs_dispatching: u64,
    pub(crate) throughput_per_min: u64,
    pub(crate) eval_running: u64,
    pub(crate) eval_queued: u64,
    pub(crate) per_node: Vec<Value>,
}

impl Utilization {
    pub(crate) fn core_pct(&self) -> f64 {
        if self.core_budget_total == 0 {
            return 0.0;
        }
        let used = self.core_budget_total.saturating_sub(self.core_budget_free);
        used as f64 * 100.0 / self.core_budget_total as f64
    }
    pub(crate) fn slot_pct(&self) -> f64 {
        if self.slots_total == 0 {
            return 0.0;
        }
        self.slots_in_use as f64 * 100.0 / self.slots_total as f64
    }
    pub(crate) fn queue_depth(&self) -> u64 {
        self.jobs_queued + self.jobs_dispatching
    }
}

pub(crate) fn measure(args: &Args) -> Utilization {
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
pub(crate) struct WorkOnDisk {
    /// Manifests sitting in the newest built waves. `--max-overlap-waves 8` is
    /// supply's own overlap ceiling, so 8 is the honest horizon.
    pub(crate) wave_manifests: u64,
    pub(crate) waves_scanned: u64,
    pub(crate) newest_wave: Option<String>,
    pub(crate) newest_wave_age_secs: Option<u64>,
    /// Datasets encoded (sidecars complete) but not yet in the live catalog.
    pub(crate) activation_backlog: u64,
    /// Last first-touch backlog the coverage builder reported.
    pub(crate) coverage_backlog_unclaimed: u64,
}

impl WorkOnDisk {
    pub(crate) fn total(&self) -> u64 {
        self.wave_manifests + self.activation_backlog + self.coverage_backlog_unclaimed
    }
}

pub(crate) fn survey_work(args: &Args) -> WorkOnDisk {
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
    waves.sort_by_key(|row| std::cmp::Reverse(row.0));
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
pub(crate) fn last_coverage_event(args: &Args, kind: &str) -> Option<Value> {
    let path = args.root.join("coverage/ops/coverage-events.v1.jsonl");
    let bytes = tail_bytes(&path, 256 * 1024)?;
    let text = String::from_utf8_lossy(&bytes);
    text.lines()
        .rev()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|value| str_at(value, "kind") == Some(kind))
}

pub(crate) fn tail_bytes(path: &Path, limit: u64) -> Option<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(path).ok()?;
    let length = file.metadata().ok()?.len();
    let start = length.saturating_sub(limit);
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).ok()?;
    Some(bytes)
}

#[path = "diagnostics/diagnose.rs"]
mod diagnose;
pub(crate) use diagnose::*;
