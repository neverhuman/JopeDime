//! jope_selfplay_alarms — one-shot health scan for the self-play flywheel.
//!
//! Read-only. Emits `ops/health-alarms.v1.json` (schema `JopeSelfplayHealthAlarms.v1`)
//! and always exits 0: an alarm writer that can itself fail the unit would be a
//! second thing to monitor. Driven by jope-selfplay-alarms.timer (60 s).

use anyhow::Result;
use jope_dime_regressor::paths;
use jope_dime_regressor::training_history as th;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

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
    fn corpus_str => paths::corpus_db()
);

/// (relative status file, max age seconds) — ages beyond these mean the owning
/// daemon is dead or wedged even if systemd thinks it is running.
const STATUS_AGES: &[(&str, u64)] = &[
    ("ops/queue-keeper-v3.json", 900),
    ("ops/micro-wave-supply-v2.json", 120),
    // 2026-08-01: was ops/trainer-watchdog-v1.json, which had no writer left —
    // the unit now runs jope_task_learning_controller (30 s cycle) and this
    // doc is where trainer liveness actually lives. The old doc's last write
    // was 2026-07-29 and it still claimed a dead run active; watching a file
    // nobody writes trained operators to ignore a permanent red alarm.
    ("ops/task-learning-controller.v2.json", 120),
    ("ops/last-ingest.json", 900),
    ("ops/freshness-status.json", 300),
];

const DISK_WARN_PCT: f64 = 85.0;
const DISK_CRIT_PCT: f64 = 90.0;
const SWAP_FREE_MIN_MB: u64 = 256;
const CORPUS_WARN_BYTES: u64 = 20 * 1024 * 1024 * 1024;
const TRAINER_STALL_SECS: u64 = 6 * 3600;
/// Only judge conditionality for runs whose metrics are still fresh: an ancient
/// finished run must not pin a permanent red alarm.
const CONDITIONAL_LOOKBACK_SECS: u64 = 24 * 3600;
/// Tail budget when reading train_metrics.jsonl (one row per epoch).
const METRICS_TAIL_BYTES: u64 = 256 * 1024;
/// The capture timer runs every 3 min; five missed cycles means the durable
/// training-metrics archive has stopped being written and epochs are once again
/// only on the garbage collector's chopping block.
const HISTORY_INDEX_STALE_SECS: u64 = 900;
/// A live trainer emits an epoch roughly every 4 min. If a run directory's
/// metrics file has grown within this window but the archive has not advanced
/// for it, capture is silently failing for that run.
const HISTORY_ADVANCE_GRACE_SECS: u64 = 1_800;
const HISTORY_SOURCE_FRESH_SECS: u64 = 900;
/// The replica timer runs every 15 min; three missed cycles is a real stop, not
/// a slow node.
const REPLICA_RECEIPT_STALE_SECS: u64 = 2_700;
/// One target being briefly unreachable is normal on a busy cluster. Two hours
/// without a verified publish is a target that is not coming back on its own.
const REPLICA_TARGET_STALE_SECS: u64 = 7_200;
/// Source + one replica is one failure from a single copy, which is the state
/// this whole archive exists to leave behind.
const REPLICA_MIN_HEALTHY: u64 = 2;
/// Mirrors `test_exact_train.SPLIT_MEDIAN_RATIO_MAX`. The builder quarantines a
/// freeze that exceeds it; this only reports one that already got through.
const SPLIT_MEDIAN_RATIO_MAX: f64 = 2.0;
/// Freezes are triggered every 50,000 labels
/// (`test_exact_controller.py:198`, `jope_selfplay_corpus_freezer.rs`), and
/// ingestion adds ~8.4 rows/s (MEASURED 23:31-23:32Z on the live root). So
/// 200,000 unfrozen labels is ~6.6 hours with no freeze — four missed triggers,
/// not a slow one — and 1,000,000 is a day and a half.
const CORPUS_STARVED_WARN: u64 = 200_000;
const CORPUS_STARVED_CRIT: u64 = 1_000_000;
/// The freezer's own cycle is 600 s; three missed cycles is a stop.
const CORPUS_FREEZER_STALE_SECS: u64 = 1_800;

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

struct Alarms {
    rows: Vec<Value>,
}

impl Alarms {
    fn push(&mut self, alarm: &str, severity: &str, observed: String, threshold: String) {
        self.rows.push(json!({
            "alarm": alarm,
            "severity": severity,
            "observed": observed,
            "threshold": threshold,
            "at_ms": now_ms() as u64,
        }));
    }
}

fn scq_json(args: &[&str]) -> Option<Value> {
    let output = Command::new(scq_str())
        .arg("--json")
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    serde_json::from_slice(&output.stdout).ok()
}

fn file_age_secs(path: &Path) -> Option<u64> {
    let modified = path.metadata().ok()?.modified().ok()?;
    SystemTime::now()
        .duration_since(modified)
        .ok()
        .map(|d| d.as_secs())
}

/// SCQ emits UTC RFC3339 timestamps. Parsing them locally avoids pulling a
/// high-level time runtime into the scanner — same implementation as
/// jope_selfplay_queue.rs::parse_rfc3339_seconds (bins cannot share items).
fn parse_rfc3339_seconds(value: &str) -> Option<i64> {
    let year = value.get(0..4)?.parse::<i64>().ok()?;
    let month = value.get(5..7)?.parse::<i64>().ok()?;
    let day = value.get(8..10)?.parse::<i64>().ok()?;
    let hour = value.get(11..13)?.parse::<i64>().ok()?;
    let minute = value.get(14..16)?.parse::<i64>().ok()?;
    let second = value.get(17..19)?.parse::<i64>().ok()?;
    let timezone = value.find('Z').map(|_| 0).or_else(|| {
        let index = value.get(19..)?.find(['+', '-'])? + 19;
        let sign = if value.as_bytes()[index] == b'+' {
            1
        } else {
            -1
        };
        let hours = value.get(index + 1..index + 3)?.parse::<i64>().ok()?;
        let minutes = value.get(index + 4..index + 6)?.parse::<i64>().ok()?;
        Some(sign * (hours * 3_600 + minutes * 60))
    })?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let days = days_from_civil(year, month, day)?;
    Some(days * 86_400 + hour * 3_600 + minute * 60 + second - timezone)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> Option<i64> {
    let year = year - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let adjusted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Some(era * 146_097 + day_of_era - 719_468)
}

fn check_status_ages(root: &Path, alarms: &mut Alarms) {
    for (relative, max_age) in STATUS_AGES {
        let path = root.join(relative);
        match file_age_secs(&path) {
            Some(age) if age > *max_age => alarms.push(
                &format!("status_stale:{relative}"),
                "critical",
                format!("{age}s"),
                format!("{max_age}s"),
            ),
            None => alarms.push(
                &format!("status_missing:{relative}"),
                "warning",
                "missing".into(),
                "present".into(),
            ),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Content contracts: what the file SAYS, not when it was touched
// ---------------------------------------------------------------------------
//
// `check_status_ages` above is why 3.5 hours of a total cluster outage passed in
// silence on 2026-07-28. Both the queue keeper (jope_selfplay_queue.rs:273-282)
// and the supply supervisor (jope_selfplay_supply.rs:256-265) rewrite their
// canonical status file on EVERY FAILED CYCLE with a differently-shaped
// `*Error.v1` document. A daemon spinning in a tight error loop therefore
// presents the freshest possible file: mtime says "maximally healthy", content
// says "I have not done any work since 11:40". systemd agreed (NRestarts=0,
// ActiveState=active) because the process never exits.
//
// The checks below read the bytes:
//
//   status_error_doc:<file>    the file carries an `*Error.v1` (or otherwise
//                              unexpected) schema right now.
//   status_failing:<file>      it has carried one for FAILING_CYCLES straight
//                              scans — a wedged daemon, not a transient blip.
//   status_degraded:<file>     in-schema but self-reporting `degraded: true`
//                              (the pattern trainer-watchdog now uses instead
//                              of swapping its document shape).
//   status_content_frozen:<f>  fresh mtime but the document's own
//                              `generated_at_ms` has not advanced past its
//                              cadence budget — "rewritten, but not advancing".
//   contract_violation:<f>#<k> the file parses, but a field the CONSUMER
//                              requires is missing or the wrong JSON type.
//
// The last one detects the CLASS, not the instance. `queued_target` was a
// non-Option `i64` in `jope_selfplay_supply.rs`'s `QueueStatus`, so a keeper
// error document killed supply 5,893 times with "missing field queued_target".
// Rather than hand-copy that field list (which would drift the first time Codex
// touched the struct), the required set is derived at run time from the
// consumer's own source file. Add a non-Option field to `QueueStatus` and the
// live check starts enforcing it with no edit here.

jope_paths::once_str!(
    /// Rust source tree of this crate (`crates/jope-campaigns` under
    /// `JOPE_HOME`; the crate sat at the checkout root when the consumer
    /// paths below were written), for deriving consumer field contracts.
    /// Overridable with `JOPE_SRC_ROOT` so the check can run against a
    /// checkout under test.
    fn src_root_str => paths::jope_home().join("crates/jope-campaigns")
);

/// Consecutive scans a status file may stay broken before the alarm escalates
/// from "this cycle failed" to "this daemon is wedged". The timer is 60 s.
#[path = "jope_selfplay_alarms/contract_types.rs"]
mod contract_types;
pub(crate) use contract_types::*;
#[path = "jope_selfplay_alarms/content_contracts.rs"]
mod content_contracts;
pub(crate) use content_contracts::*;
#[path = "jope_selfplay_alarms/node_job_checks.rs"]
mod node_job_checks;
pub(crate) use node_job_checks::*;
#[path = "jope_selfplay_alarms/training_history_checks.rs"]
mod training_history_checks;
pub(crate) use training_history_checks::*;
#[path = "jope_selfplay_alarms/proposal_checks.rs"]
mod proposal_checks;
pub(crate) use proposal_checks::*;
#[path = "jope_selfplay_alarms/corpus_checks.rs"]
mod corpus_checks;
pub(crate) use corpus_checks::*;

/// `[<root>] [--contract-severity critical|warning]`.
///
/// The severity knob exists because of a live coupling, not because the answer
/// is unclear: jope_selfplay_expansion_driver.rs:116-142 halts dataset
/// activation on ANY critical alarm outside its skip list. A critical
/// `status_error_doc:` alarm would therefore stop new datasets being activated
/// at precisely the moment the queue keeper is broken — the exact
/// alarm-blocks-expansion-blocks-the-cluster loop that idled all 1,052 slots on
/// 2026-07-28. Default `warning` keeps the alarm loud in the TUI and in
/// `jope_ops_health_check` (which fails the unit regardless of severity)
/// without paralysing expansion. Flip to `critical` by appending
/// `--contract-severity critical` to jope-selfplay-alarms.service once the
/// driver's skip list covers `status_*` / `contract_*`.
fn parse_args() -> (PathBuf, String) {
    let mut root: Option<String> = None;
    let mut severity = "warning".to_owned();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--contract-severity" => {
                if let Some(value) = arguments.next() {
                    severity = value;
                }
            }
            other => match other.strip_prefix("--contract-severity=") {
                Some(value) => severity = value.to_owned(),
                None if !other.starts_with("--") && root.is_none() => {
                    root = Some(other.to_owned());
                }
                None => {}
            },
        }
    }
    if severity != "critical" && severity != "warning" {
        severity = "warning".to_owned();
    }
    (
        PathBuf::from(root.unwrap_or_else(|| root_str().to_owned())),
        severity,
    )
}

fn main() -> Result<()> {
    let (root, contract_severity) = parse_args();
    let mut alarms = Alarms { rows: Vec::new() };

    check_status_ages(&root, &mut alarms);
    check_status_contracts(&root, &mut alarms, &contract_severity);
    check_nodes(&mut alarms);
    check_scq_and_jobs(&mut alarms);
    check_local_memory(&mut alarms);
    check_corpus_size(&mut alarms);
    check_corpus_freeze_guard(&root, &mut alarms);
    check_trainer_intent(&root, &mut alarms);
    check_proposal_deficiency(&root, &mut alarms);
    check_warm_start_guard(&root, &mut alarms);
    check_training_history(&root, Path::new(runs_root_str()), &mut alarms);
    check_training_replica(&root, &mut alarms);

    let report = json!({
        "schema": "JopeSelfplayHealthAlarms.v1",
        "generated_at_ms": now_ms() as u64,
        "alarm_count": alarms.rows.len(),
        "alarms": alarms.rows,
    });
    let target = root.join("ops/health-alarms.v1.json");
    let temporary = root.join(format!("ops/.health-alarms.v1.json.{}", std::process::id()));
    if std::fs::write(&temporary, serde_json::to_vec_pretty(&report)?).is_ok() {
        let _ = std::fs::rename(&temporary, &target);
    }
    println!("{report}");
    // Never fail the unit: exit 0 regardless of what we saw.
    Ok(())
}

#[cfg(test)]
#[path = "jope_selfplay_alarms/tests.rs"]
mod tests;
