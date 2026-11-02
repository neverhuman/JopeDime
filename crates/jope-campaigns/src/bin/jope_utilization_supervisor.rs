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

mod diagnostics;
mod report;
mod response;
#[cfg(test)]
mod tests;

pub(super) use diagnostics::*;
pub(super) use report::*;
pub(super) use response::*;

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
        match &self.state_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/utilization-supervisor-state.v1.json"),
        }
    }
    fn status(&self) -> PathBuf {
        match &self.status_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/utilization-supervisor-v1.json"),
        }
    }
    fn series(&self) -> PathBuf {
        match &self.series_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/utilization-series.v1.jsonl"),
        }
    }
    fn events(&self) -> PathBuf {
        match &self.events_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/utilization-supervisor-events.v1.jsonl"),
        }
    }
    fn alarms(&self) -> PathBuf {
        match &self.alarms_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/utilization-alarms.v1.json"),
        }
    }
    fn banner(&self) -> PathBuf {
        match &self.banner_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/UTILIZATION-ESCALATION.txt"),
        }
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
