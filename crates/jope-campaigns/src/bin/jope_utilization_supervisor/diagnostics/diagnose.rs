use super::*;

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
