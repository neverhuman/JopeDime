use super::*;

#[derive(Debug, Default)]
pub(super) struct Utilization {
    pub(super) scq_reachable: bool,
    pub(super) scq_error: Option<String>,
    pub(super) epoch: u64,
    pub(super) nodes_up: u64,
    pub(super) nodes_total: u64,
    /// Sum of per-node `running_job_slots` — the operator's "N of 1,052 slots".
    pub(super) slots_in_use: u64,
    pub(super) slots_total: u64,
    /// Core budget is the constraint that actually binds; slot targets are a
    /// 4x multiplier over it on the 3 big nodes.
    pub(super) core_budget_total: u64,
    pub(super) core_budget_free: u64,
    pub(super) jobs_queued: u64,
    pub(super) jobs_running: u64,
    pub(super) jobs_dispatching: u64,
    pub(super) throughput_per_min: u64,
    pub(super) eval_running: u64,
    pub(super) eval_queued: u64,
    pub(super) per_node: Vec<Value>,
}

impl Utilization {
    pub(super) fn core_pct(&self) -> f64 {
        if self.core_budget_total == 0 {
            return 0.0;
        }
        let used = self.core_budget_total.saturating_sub(self.core_budget_free);
        used as f64 * 100.0 / self.core_budget_total as f64
    }
    pub(super) fn slot_pct(&self) -> f64 {
        if self.slots_total == 0 {
            return 0.0;
        }
        self.slots_in_use as f64 * 100.0 / self.slots_total as f64
    }
    pub(super) fn queue_depth(&self) -> u64 {
        self.jobs_queued + self.jobs_dispatching
    }
}

pub(super) fn measure(args: &Args) -> Utilization {
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
pub(super) struct WorkOnDisk {
    /// Manifests sitting in the newest built waves. `--max-overlap-waves 8` is
    /// supply's own overlap ceiling, so 8 is the honest horizon.
    pub(super) wave_manifests: u64,
    pub(super) waves_scanned: u64,
    pub(super) newest_wave: Option<String>,
    pub(super) newest_wave_age_secs: Option<u64>,
    /// Datasets encoded (sidecars complete) but not yet in the live catalog.
    pub(super) activation_backlog: u64,
    /// Last first-touch backlog the coverage builder reported.
    pub(super) coverage_backlog_unclaimed: u64,
}

impl WorkOnDisk {
    pub(super) fn total(&self) -> u64 {
        self.wave_manifests + self.activation_backlog + self.coverage_backlog_unclaimed
    }
}

pub(super) fn survey_work(args: &Args) -> WorkOnDisk {
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
pub(super) fn last_coverage_event(args: &Args, kind: &str) -> Option<Value> {
    let path = args.root.join("coverage/ops/coverage-events.v1.jsonl");
    let bytes = tail_bytes(&path, 256 * 1024)?;
    let text = String::from_utf8_lossy(&bytes);
    text.lines()
        .rev()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|value| str_at(value, "kind") == Some(kind))
}

pub(super) fn tail_bytes(path: &Path, limit: u64) -> Option<Vec<u8>> {
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
pub(super) enum Health {
    Producing,
    Erroring,
    Stale,
    Blocked,
    Drained,
    Down,
    Unknown,
}

impl Health {
    pub(super) fn as_str(self) -> &'static str {
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
    pub(super) fn healthy(self) -> bool {
        matches!(self, Health::Producing)
    }
}

#[derive(Debug)]
pub(super) struct Stage {
    pub(super) name: &'static str,
    pub(super) health: Health,
    pub(super) detail: String,
    /// Receipt this verdict came from, so the verdict is auditable.
    pub(super) receipt: String,
    /// Bounded repair, if one exists for this stage. `None` means "there is no
    /// safe automatic repair" — escalate instead of improvising.
    pub(super) repair: Option<Repair>,
}

#[derive(Clone, Debug)]
pub(super) struct Repair {
    pub(super) key: String,
    pub(super) verb: &'static str,
    pub(super) unit: &'static str,
}

impl Repair {
    pub(super) fn restart(unit: &'static str) -> Self {
        Self {
            key: format!("restart:{unit}"),
            verb: "restart",
            unit,
        }
    }
    pub(super) fn start(unit: &'static str) -> Self {
        Self {
            key: format!("start:{unit}"),
            verb: "start",
            unit,
        }
    }
}

pub(super) fn content_age_secs(value: &Value, now: u64) -> Option<u64> {
    let generated = u64_at(value, "generated_at_ms")?;
    Some(now.saturating_sub(generated) / 1000)
}

/// Blocked-stage verdict when the newest cohort-supply state has sat at
/// `submit_cohort_proposer` for longer than any healthy dispatch would take.
/// Returns None when the proposer pipeline is moving (no stage row at all —
/// absence of a proposer stage means "not the bottleneck").
pub(super) fn proposer_stage(args: &Args, now: u64) -> Option<Stage> {
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
pub(super) fn diagnose(args: &Args, util: &Utilization, work: &WorkOnDisk) -> Vec<Stage> {
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
