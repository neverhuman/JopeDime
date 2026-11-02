use super::*;

pub(super) fn node_f64(node: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|k| node.get(*k).and_then(Value::as_f64))
}

pub(super) fn check_nodes(alarms: &mut Alarms) -> Option<Value> {
    let nodes = scq_json(&["nodes"])?;
    let list = nodes
        .pointer("/Nodes/nodes")
        .or_else(|| nodes.get("nodes"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for node in &list {
        let name = node
            .get("node")
            .or_else(|| node.get("id"))
            .and_then(Value::as_str)
            .unwrap_or("?");
        if let Some(disk) = node_f64(node, &["disk_use_pct", "disk_pct", "disk"]) {
            if disk >= DISK_CRIT_PCT {
                alarms.push(
                    &format!("disk_high:{name}"),
                    "critical",
                    format!("{disk:.0}%"),
                    format!("{DISK_CRIT_PCT:.0}%"),
                );
            } else if disk >= DISK_WARN_PCT {
                alarms.push(
                    &format!("disk_high:{name}"),
                    "warning",
                    format!("{disk:.0}%"),
                    format!("{DISK_WARN_PCT:.0}%"),
                );
            }
        }
        let health = node
            .get("health")
            .or_else(|| node.get("health_class"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if health == "critical" || health == "down" {
            alarms.push(
                &format!("node_health:{name}"),
                "critical",
                health.to_string(),
                "healthy|degraded".into(),
            );
        }
    }
    Some(nodes)
}

pub(super) fn check_scq_and_jobs(alarms: &mut Alarms) {
    let Some(snapshot) = scq_json(&["snapshot"]) else {
        alarms.push(
            "scq_unreachable",
            "critical",
            "snapshot failed".into(),
            "scq --json snapshot ok".into(),
        );
        return;
    };
    let snap = snapshot.get("Snapshot").unwrap_or(&snapshot);
    let jobs = snap
        .get("jobs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    // "Deep learning never stops": at least one live trainer-family job at all times.
    let live_trainer = jobs.iter().any(|job| {
        job.get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| id.contains("-train-"))
            && matches!(
                job.get("state").and_then(Value::as_str),
                Some("running" | "queued" | "starting" | "dispatching")
            )
    });
    if !live_trainer {
        // A trainer absent because policy paused it is a WARNING with the
        // pause spelled out; an unexplained absence stays CRITICAL. The
        // authority is the task-learning controller's status doc (30 s
        // cycle): "plateau_diagnosis_requires_recipe_change" and
        // "waiting_for_newer_corpus" are the two states in which zero live
        // trainer jobs is the controller's own intent. Before 2026-08-01 a
        // policy pause showed as a permanent critical no_active_trainer,
        // which trained operators to ignore the one alarm that matters when
        // the trainer dies for real.
        let controller_action =
            std::fs::read(Path::new(root_str()).join("ops/task-learning-controller.v2.json"))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .and_then(|doc| doc.get("action").and_then(Value::as_str).map(str::to_owned))
                .unwrap_or_default();
        if matches!(
            controller_action.as_str(),
            "plateau_diagnosis_requires_recipe_change" | "waiting_for_newer_corpus"
        ) {
            alarms.push(
                "trainer_paused_by_policy",
                "warning",
                format!("controller action: {controller_action}"),
                "policy pause lifts or a trainer job appears".into(),
            );
        } else {
            alarms.push(
                "no_active_trainer",
                "critical",
                "zero live jope-selfplay-train jobs".into(),
                ">=1 running/queued trainer".into(),
            );
        }
    }
    // "Blocked or throttled must be LOUD": SCQ's `held` state is sticky. On
    // 2026-08-01 a cpu-overload-guard episode stranded 996 jope-selfplay-eval
    // jobs for ~3 h on an otherwise idle cluster (3 running, 329/408 cores
    // free) with no alarm. Held work at this volume while almost none of ours
    // runs is starvation, never steady state — the queue keeper's janitor
    // releases these, so this firing means the janitor is dead or losing.
    let held_jope = jobs
        .iter()
        .filter(|job| {
            job.get("state").and_then(Value::as_str) == Some("held")
                && job
                    .get("project")
                    .and_then(Value::as_str)
                    .is_some_and(|project| project.starts_with("jope-selfplay"))
        })
        .count();
    let running_jope = jobs
        .iter()
        .filter(|job| {
            job.get("state").and_then(Value::as_str) == Some("running")
                && job
                    .get("project")
                    .and_then(Value::as_str)
                    .is_some_and(|project| project.starts_with("jope-selfplay"))
        })
        .count();
    if held_jope >= 32 && running_jope < held_jope / 4 {
        alarms.push(
            "scq_jobs_held_starving",
            "critical",
            format!("{held_jope} jope jobs held, only {running_jope} running"),
            "held jope jobs drain via the queue keeper's release janitor".into(),
        );
    }
    // A cohort proposer stuck in "queued" is the flywheel's single point of
    // failure: waves cannot compose without >=8 fresh DL proposals per task
    // (DL_SLOT_RUNGS has no 0), so one starved 13-second GPU job idles every
    // CPU core within two keeper cycles. On 2026-08-01 the proposer sat
    // queued 3 h 12 m behind an external weight-10 GPU queue while a weight
    // guard pinned every jope family to 1; utilization collapsed 98% -> 3%
    // and no alarm named the cause for five hours.
    let now_secs = (now_ms() / 1_000) as i64;
    for job in &jobs {
        let id = job.get("id").and_then(Value::as_str).unwrap_or("");
        if !id.starts_with("jope-selfplay-cohort-propose")
            || job.get("state").and_then(Value::as_str) != Some("queued")
        {
            continue;
        }
        let Some(queued_at) = job
            .get("queue_ts")
            .and_then(Value::as_str)
            .and_then(parse_rfc3339_seconds)
        else {
            continue;
        };
        let age = now_secs.saturating_sub(queued_at);
        if age > 900 {
            alarms.push(
                "proposer_gpu_starved",
                "critical",
                format!("cohort proposer {id} queued {age}s for 1 GPU"),
                "proposer dispatches <900s; check `scq priority ls` for jope weight flattening"
                    .into(),
            );
        }
    }
    // A proposer that FAILS (vs starves in queue) exhausts supply's 5-attempt
    // retry budget and hard-wedges wave production until an operator
    // intervenes (2026-08-03: bundle absent on the dispatch node -> six
    // "vanished" attempts -> eval queue drained to 1% cluster CPU). The
    // generic status_error_doc warning is not loud enough for a
    // flywheel-stopping state.
    let supply_doc = Path::new(root_str()).join("ops/micro-wave-supply-v2.json");
    if let Ok(bytes) = std::fs::read(&supply_doc) {
        if let Ok(doc) = serde_json::from_slice::<Value>(&bytes) {
            if doc.get("schema").and_then(Value::as_str) == Some("JopeMicroWaveSupplyError.v1") {
                let identical = doc
                    .get("identical_failures")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let critical = doc
                    .get("critical")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if critical || identical >= 3 {
                    let error = doc
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    alarms.push(
                        "cohort_proposer_failing",
                        "critical",
                        format!("supply error x{identical}: {error}"),
                        "wave production halted; inspect the proposer job in scq and the supply journal"
                            .into(),
                    );
                }
            }
        }
    }
    // "Self-play never stops": a fresh strict wave must exist (cadence ~9-13 min).
    let newest_wave_age = std::fs::read_dir(Path::new(root_str()).join("waves"))
        .ok()
        .and_then(|entries| {
            entries
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("strict-wave-"))
                .filter_map(|e| e.metadata().ok().and_then(|m| m.modified().ok()))
                .max()
        })
        .and_then(|newest| SystemTime::now().duration_since(newest).ok())
        .map(|d| d.as_secs());
    if let Some(age) = newest_wave_age {
        if age > 2_700 {
            alarms.push(
                "wave_stalled",
                "critical",
                format!("newest wave {age}s old"),
                "<2700s".into(),
            );
        }
    }
    for job in &jobs {
        let id = job.get("id").and_then(Value::as_str).unwrap_or("");
        let state = job.get("state").and_then(Value::as_str).unwrap_or("");
        if id.to_lowercase().contains("ratchet")
            && matches!(state, "running" | "queued" | "dispatching" | "starting")
        {
            alarms.push(
                "ratchet_revival_detected",
                "warning",
                format!("{id}:{state}"),
                "eval must vacate xbabe3 if ratchet returns".into(),
            );
        }
        let gpus = job.get("gpus").and_then(Value::as_u64).unwrap_or(0);
        let is_gpu_running = state == "running"
            && (gpus > 0
                || job
                    .get("gpu_uuids")
                    .and_then(Value::as_array)
                    .map(|a| !a.is_empty())
                    .unwrap_or(false));
        // Only the trainer family writes progress files; propose/eval jobs in the
        // same project do not. Remote-node run dirs are not visible locally, so a
        // missing attempts dir means "cannot judge", not "stalled".
        if !id.contains("-train-") {
            continue;
        }
        let run_dir = job
            .get("run_dir")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(runs_root_str()).join(id));
        if !run_dir.join("attempts").is_dir() {
            continue;
        }
        if is_gpu_running {
            match newest_progress_age(&run_dir) {
                Some(age) if age > TRAINER_STALL_SECS => alarms.push(
                    &format!("trainer_stalled:{id}"),
                    "critical",
                    format!("progress {age}s old"),
                    format!("<{TRAINER_STALL_SECS}s while running"),
                ),
                None => alarms.push(
                    &format!("trainer_no_progress_file:{id}"),
                    "warning",
                    "no train_progress.json/progress.json found".into(),
                    "progress file expected".into(),
                ),
                _ => {}
            }
        }
        check_completed_conditionality(id, state, &run_dir, alarms);
    }
}

/// A COMPLETED trainer whose final conditional advantage is <= 0 shipped a policy
/// that ignores the dataset embedding: its conditional reward head lost to its own
/// detached genome-only ablation, so the checkpoint is a genome prior with noise on
/// top. The promotion gate now refuses such a checkpoint, but silence about WHY is
/// what let one run a whole ladder unnoticed.
pub(super) fn check_completed_conditionality(
    id: &str,
    state: &str,
    run_dir: &Path,
    alarms: &mut Alarms,
) {
    if state != "completed" {
        return;
    }
    // No metrics locally means "cannot judge" (remote run dir), not "healthy".
    let Some((advantage, age)) = final_conditional_advantage(run_dir) else {
        return;
    };
    if advantage > 0.0 || age > CONDITIONAL_LOOKBACK_SECS {
        return;
    }
    alarms.push(
        &format!("policy_not_conditional:{id}"),
        "critical",
        format!("policy is not conditional: final conditional_advantage {advantage:+.4}"),
        ">0 (conditional head must beat the genome-only ablation)".into(),
    );
}

/// Newest train_progress.json / progress.json under <run_dir>/attempts/*/outputs
/// with its mtime, bounded to a shallow walk.
pub(super) fn newest_progress_file(run_dir: &Path) -> Option<(SystemTime, PathBuf)> {
    let attempts = run_dir.join("attempts");
    let mut newest: Option<(SystemTime, PathBuf)> = None;
    let entries = std::fs::read_dir(&attempts).ok()?;
    for attempt in entries.flatten() {
        let outputs = attempt.path().join("outputs");
        walk_progress(&outputs, 0, &mut newest);
    }
    newest
}

/// Newest mtime age (secs) of any train_progress.json / progress.json under
/// <run_dir>/attempts/*/outputs, bounded to a shallow walk.
pub(super) fn newest_progress_age(run_dir: &Path) -> Option<u64> {
    let (newest, _) = newest_progress_file(run_dir)?;
    SystemTime::now()
        .duration_since(newest)
        .ok()
        .map(|d| d.as_secs())
}

/// Final `val_conditional_advantage` a trainer run recorded, with the age of the
/// metrics file it came from. train_metrics.jsonl sits beside the progress file,
/// exactly where the TUI reads it; rows without the key (non-validation epochs)
/// are skipped.
pub(super) fn final_conditional_advantage(run_dir: &Path) -> Option<(f64, u64)> {
    let (_, progress) = newest_progress_file(run_dir)?;
    let metrics = progress.parent()?.join("train_metrics.jsonl");
    let age = file_age_secs(&metrics)?;
    let advantage = read_tail(&metrics, METRICS_TAIL_BYTES)?
        .lines()
        .rev()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find_map(|row| row.get("val_conditional_advantage").and_then(Value::as_f64))?;
    Some((advantage, age))
}

/// Last `cap` bytes of `path` with any leading partial line dropped.
pub(super) fn read_tail(path: &Path, cap: u64) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let offset = file.metadata().ok()?.len().saturating_sub(cap);
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut bytes = Vec::new();
    file.take(cap).read_to_end(&mut bytes).ok()?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    if offset == 0 {
        return Some(text);
    }
    text.find('\n').map(|index| text[index + 1..].to_owned())
}

pub(super) fn walk_progress(dir: &Path, depth: usize, newest: &mut Option<(SystemTime, PathBuf)>) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten().take(256) {
        let path = entry.path();
        if path.is_dir() {
            walk_progress(&path, depth + 1, newest);
        } else if matches!(
            path.file_name().and_then(|n| n.to_str()),
            Some("train_progress.json") | Some("progress.json")
        ) {
            if let Ok(modified) = entry.metadata().and_then(|m| m.modified()) {
                if newest.as_ref().map(|(n, _)| modified > *n).unwrap_or(true) {
                    *newest = Some((modified, path));
                }
            }
        }
    }
}
