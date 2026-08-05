//! jope_ops_health_check — turn "my status file says I failed" into a systemd
//! failure.
//!
//! # The blind spot this closes
//!
//! On 2026-07-28 `jope-selfplay-supply.service` emitted a
//! `JopeMicroWaveSupplyError.v1` document every five seconds for hours —
//! 5,893 consecutive failed cycles — while systemd reported:
//!
//! ```text
//! ActiveState=active   SubState=running   NRestarts=0   Result=success
//! ```
//!
//! All of that is true. The process never exited; it caught its error, wrote it
//! to `ops/micro-wave-supply-v2.json`, slept, and tried again. systemd watches
//! *processes*, and the process was fine. Nothing watched the *work*.
//!
//! The same held for `jope-selfplay-queue.service` (1,097 identical failures)
//! and every mtime-based liveness guard in `jope_selfplay_alarms.rs`: a daemon
//! spinning in a tight error loop rewrites its status file more often than a
//! healthy one, so it looks *maximally* fresh.
//!
//! This unit is a `Type=oneshot` timer job that reads the CONTENT of the
//! status files and **exits non-zero** when a daemon has been publishing an
//! `*Error.v1` document (or an in-schema `degraded: true` one) for
//! `--fail-after-cycles` consecutive scans. The failure is then visible exactly
//! where an operator already looks:
//!
//! ```text
//! systemctl --user --failed
//! systemctl --user is-failed jope-ops-health-check.service   # -> failed
//! journalctl --user -u jope-ops-health-check -n 20
//! ```
//!
//! # Rules it follows itself
//!
//! * It writes `ops/ops-health-check.v1.json` on EVERY run, pass or fail, and
//!   **always in the same schema** — `status` moves between `ok`, `failing` and
//!   `check_error`, the shape never does. That is the pattern the queue keeper
//!   and supply supervisor still need (see the report in FLYWHEEL_CHAT.md);
//!   `jope_selfplay_trainer_watchdog::CycleReport::degraded` is the worked
//!   example.
//! * It only judges files whose mtime is inside `--live-window-seconds`, so a
//!   historical artifact that legitimately carries an error schema (e.g.
//!   `ops/monitor-error.json`, a test-only receipt from 2026-07-26) can never
//!   pin a permanent failure.
//! * It does not restart anything. A daemon failing deterministically — as both
//!   of today's did — is not fixed by a restart, and a restart loop would erase
//!   the evidence. Being loud is the job.

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Map, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_ROOT: &str = "/home/ubuntu/jope-flywheel/test_exact_gen0";
const RECEIPT_SCHEMA: &str = "JopeOpsHealthCheck.v1";
const STATE_SCHEMA: &str = "JopeOpsHealthCheckState.v1";
/// Status documents are all a few KiB. The multi-hundred-KiB `pre-*.json`
/// snapshots in the same directory are historical scq dumps, never live status.
const MAX_STATUS_BYTES: u64 = 1024 * 1024;

#[derive(Parser, Debug)]
#[command(about = "Fail the unit when a flywheel daemon keeps publishing an error document")]
struct Args {
    /// Flywheel root.
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
    /// Consecutive scans a file may carry an error/degraded document before
    /// this unit fails. At the 60 s timer cadence, 3 means ~3 minutes.
    #[arg(long, default_value_t = 3)]
    fail_after_cycles: u64,
    /// Only files touched this recently are treated as live status.
    #[arg(long, default_value_t = 900)]
    live_window_seconds: u64,
    /// The alarm scan is itself a daemon; if its receipt goes stale nobody is
    /// watching the watchers.
    #[arg(long, default_value_t = 600)]
    alarms_max_age_seconds: u64,
    /// Report and write receipts, but always exit 0.
    #[arg(long)]
    audit: bool,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn file_age_secs(path: &Path) -> Option<u64> {
    let modified = path.metadata().ok()?.modified().ok()?;
    SystemTime::now()
        .duration_since(modified)
        .ok()
        .map(|d| d.as_secs())
}

fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    let mut file = std::fs::File::create(&temporary)
        .with_context(|| format!("create {}", temporary.display()))?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    std::fs::rename(&temporary, path).with_context(|| format!("rename into {}", path.display()))?;
    Ok(())
}

fn truncate(text: &str, limit: usize) -> String {
    let cleaned = text.replace('\n', " ");
    match cleaned.char_indices().nth(limit) {
        Some((index, _)) => format!("{}…", &cleaned[..index]),
        None => cleaned,
    }
}

/// What is wrong with one live status document, if anything.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Verdict {
    Healthy,
    /// The daemon replaced its status document with a differently-shaped error
    /// document. Every consumer with a strict struct breaks on this.
    ErrorSchema {
        schema: String,
        error: String,
    },
    /// The daemon kept its schema and said so honestly. Still not working.
    Degraded {
        error: String,
    },
}

fn judge(value: &Value) -> Verdict {
    let schema = value.get("schema").and_then(Value::as_str).unwrap_or("");
    let error = value
        .get("error")
        .and_then(Value::as_str)
        .map(|text| truncate(text, 200))
        .unwrap_or_default();
    if schema.ends_with("Error.v1") {
        return Verdict::ErrorSchema {
            schema: schema.to_owned(),
            error,
        };
    }
    if value.get("degraded").and_then(Value::as_bool) == Some(true) {
        return Verdict::Degraded { error };
    }
    Verdict::Healthy
}

/// Every live status document in `ops/`, judged.
fn scan(root: &Path, live_window_seconds: u64) -> Result<Vec<(String, Verdict)>> {
    let ops = root.join("ops");
    let mut found = Vec::new();
    let entries = std::fs::read_dir(&ops).with_context(|| format!("read {}", ops.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() || metadata.len() > MAX_STATUS_BYTES {
            continue;
        }
        // A stale document is a different (already alarmed) problem, and a
        // historical error receipt must never pin a permanent failure here.
        match file_age_secs(&path) {
            Some(age) if age <= live_window_seconds => {}
            _ => continue,
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        let verdict = judge(&value);
        if verdict != Verdict::Healthy {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            found.push((relative, verdict));
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(found)
}

fn load_state(path: &Path) -> Map<String, Value> {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| value.get("files").cloned())
        .and_then(|files| files.as_object().cloned())
        .unwrap_or_default()
}

struct Outcome {
    receipt: Value,
    failing: bool,
}

fn evaluate(args: &Args) -> Result<Outcome> {
    let now = now_ms();
    let state_path = args.root.join("ops/ops-health-check-state.v1.json");
    let previous = load_state(&state_path);
    let mut next = Map::new();
    let mut rows = Vec::new();
    let mut failing = false;

    for (file, verdict) in scan(&args.root, args.live_window_seconds)? {
        let cycles = previous
            .get(&file)
            .and_then(|row| row.get("cycles"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
            + 1;
        let since = previous
            .get(&file)
            .and_then(|row| row.get("since_ms"))
            .and_then(Value::as_u64)
            .unwrap_or(now);
        next.insert(file.clone(), json!({ "cycles": cycles, "since_ms": since }));
        let over_budget = cycles >= args.fail_after_cycles;
        failing |= over_budget;
        let (kind, schema, error) = match &verdict {
            Verdict::ErrorSchema { schema, error } => {
                ("error_schema", schema.clone(), error.clone())
            }
            Verdict::Degraded { error } => ("degraded", String::new(), error.clone()),
            Verdict::Healthy => unreachable!("scan only returns unhealthy files"),
        };
        rows.push(json!({
            "file": file,
            "kind": kind,
            "schema": schema,
            "error": error,
            "consecutive_cycles": cycles,
            "failing_for_seconds": now.saturating_sub(since) / 1000,
            "over_budget": over_budget,
        }));
    }

    // Nobody watching the watchers is its own outage: the alarm scan writes
    // ops/health-alarms.v1.json every 60 s, and the expansion driver, the TUI
    // and an operator all read it.
    let alarms_path = args.root.join("ops/health-alarms.v1.json");
    let alarms_age = file_age_secs(&alarms_path);
    let alarms_stale = match alarms_age {
        Some(age) => age > args.alarms_max_age_seconds,
        None => true,
    };
    if alarms_stale {
        failing = true;
        rows.push(json!({
            "file": "ops/health-alarms.v1.json",
            "kind": "alarm_scan_stale",
            "schema": "",
            "error": alarms_age.map_or_else(
                || "missing".to_owned(),
                |age| format!("{age}s old, budget {}s", args.alarms_max_age_seconds),
            ),
            "consecutive_cycles": 1,
            "failing_for_seconds": 0,
            "over_budget": true,
        }));
    }

    let state = json!({
        "schema": STATE_SCHEMA,
        "generated_at_ms": now,
        "files": Value::Object(next),
    });
    let _ = atomic_json(&state_path, &state);

    let receipt = json!({
        "schema": RECEIPT_SCHEMA,
        "generated_at_ms": now,
        "status": if failing { "failing" } else { "ok" },
        "fail_after_cycles": args.fail_after_cycles,
        "live_window_seconds": args.live_window_seconds,
        "unhealthy_count": rows.len(),
        "unhealthy": rows,
        "error": Value::Null,
    });
    Ok(Outcome { receipt, failing })
}

fn main() {
    let args = Args::parse();
    let receipt_path = args.root.join("ops/ops-health-check.v1.json");
    let (receipt, failing) = match evaluate(&args) {
        Ok(outcome) => (outcome.receipt, outcome.failing),
        Err(error) => {
            // Even our own failure keeps the schema. This receipt is a contract.
            let receipt = json!({
                "schema": RECEIPT_SCHEMA,
                "generated_at_ms": now_ms(),
                "status": "check_error",
                "fail_after_cycles": args.fail_after_cycles,
                "live_window_seconds": args.live_window_seconds,
                "unhealthy_count": 0,
                "unhealthy": [],
                "error": format!("{error:#}"),
            });
            (receipt, true)
        }
    };
    let _ = atomic_json(&receipt_path, &receipt);
    println!("{}", serde_json::to_string(&receipt).unwrap_or_default());

    if failing {
        for row in receipt["unhealthy"].as_array().into_iter().flatten() {
            if row["over_budget"] == Value::Bool(true) {
                eprintln!(
                    "FAILING {} [{}] {} cycles / {}s: {}",
                    row["file"].as_str().unwrap_or("?"),
                    row["kind"].as_str().unwrap_or("?"),
                    row["consecutive_cycles"],
                    row["failing_for_seconds"],
                    row["error"].as_str().unwrap_or(""),
                );
            }
        }
        if let Some(error) = receipt["error"].as_str() {
            eprintln!("CHECK ERROR: {error}");
        }
        if !args.audit {
            // The whole point: systemd now shows this unit as failed.
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn root_with(files: &[(&str, &str)]) -> TempDir {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("ops")).unwrap();
        for (name, body) in files {
            std::fs::write(temp.path().join("ops").join(name), body).unwrap();
        }
        temp
    }

    fn args_for(root: &Path) -> Args {
        Args {
            root: root.to_path_buf(),
            fail_after_cycles: 3,
            live_window_seconds: 900,
            alarms_max_age_seconds: 600,
            audit: false,
        }
    }

    fn fresh_alarms() -> (&'static str, &'static str) {
        (
            "health-alarms.v1.json",
            r#"{"schema":"JopeSelfplayHealthAlarms.v1","alarm_count":0,"alarms":[]}"#,
        )
    }

    /// The live queue-keeper document at 15:13 on 2026-07-28, and the live
    /// supply document it caused, verbatim.
    const KEEPER_ERROR: &str = r#"{"admission_allowed":false,"error":"native attempt submission transition count mismatch (attempt_no column reservation_owner)","generated_at_ms":1785251597154,"mode":"submit","schema":"JopeSelfplayQueueKeeperError.v1"}"#;
    const SUPPLY_ERROR: &str = r#"{"builds_completed":0,"error":"parse queue status ops/queue-keeper-v3.json: missing field `queued_target` at line 1 column 218","generated_at_ms":1785251614108,"mode":"build","schema":"JopeMicroWaveSupplyError.v1"}"#;

    #[test]
    fn a_healthy_root_passes() {
        let temp = root_with(&[
            fresh_alarms(),
            (
                "queue-keeper-v3.json",
                r#"{"schema":"JopeSelfplayQueueKeeper.v3","queued_target":2048}"#,
            ),
        ]);
        let outcome = evaluate(&args_for(temp.path())).unwrap();
        assert!(!outcome.failing);
        assert_eq!(outcome.receipt["status"], "ok");
    }

    #[test]
    fn an_error_document_fails_only_after_the_streak_budget() {
        let temp = root_with(&[
            fresh_alarms(),
            ("queue-keeper-v3.json", KEEPER_ERROR),
            ("micro-wave-supply-v2.json", SUPPLY_ERROR),
        ]);
        let args = args_for(temp.path());
        for cycle in 1..=3 {
            let outcome = evaluate(&args).unwrap();
            assert_eq!(outcome.failing, cycle >= 3, "cycle {cycle}");
            assert_eq!(outcome.receipt["unhealthy_count"], 2);
        }
        let outcome = evaluate(&args).unwrap();
        assert_eq!(outcome.receipt["status"], "failing");
        let rows = outcome.receipt["unhealthy"].as_array().unwrap();
        let keeper = rows
            .iter()
            .find(|row| row["file"] == "ops/queue-keeper-v3.json")
            .unwrap();
        assert_eq!(keeper["kind"], "error_schema");
        assert_eq!(keeper["schema"], "JopeSelfplayQueueKeeperError.v1");
        assert!(keeper["error"]
            .as_str()
            .unwrap()
            .contains("transition count mismatch"));
        // And the downstream victim is named separately, not hidden behind it.
        let supply = rows
            .iter()
            .find(|row| row["file"] == "ops/micro-wave-supply-v2.json")
            .unwrap();
        assert!(supply["error"].as_str().unwrap().contains("queued_target"));
    }

    #[test]
    fn recovery_clears_the_streak() {
        let temp = root_with(&[fresh_alarms(), ("queue-keeper-v3.json", KEEPER_ERROR)]);
        let args = args_for(temp.path());
        for _ in 0..4 {
            evaluate(&args).unwrap();
        }
        std::fs::write(
            temp.path().join("ops/queue-keeper-v3.json"),
            r#"{"schema":"JopeSelfplayQueueKeeper.v3"}"#,
        )
        .unwrap();
        let outcome = evaluate(&args).unwrap();
        assert!(!outcome.failing);
        // A later relapse starts from one again, not from the old streak.
        std::fs::write(temp.path().join("ops/queue-keeper-v3.json"), KEEPER_ERROR).unwrap();
        let outcome = evaluate(&args).unwrap();
        assert!(!outcome.failing);
        assert_eq!(outcome.receipt["unhealthy"][0]["consecutive_cycles"], 1);
    }

    #[test]
    fn an_in_schema_degraded_document_counts_too() {
        let temp = root_with(&[
            fresh_alarms(),
            (
                "trainer-watchdog-v1.json",
                r#"{"schema":"JopeContinuousTrainerWatchdog.v1","degraded":true,
                    "error":"scq unreachable","action":"cycle_failed"}"#,
            ),
        ]);
        let args = args_for(temp.path());
        let mut last = evaluate(&args).unwrap();
        for _ in 0..2 {
            last = evaluate(&args).unwrap();
        }
        assert!(last.failing);
        assert_eq!(last.receipt["unhealthy"][0]["kind"], "degraded");
    }

    #[test]
    fn a_historical_error_receipt_cannot_pin_a_failure() {
        // ops/monitor-error.json really does carry JopeTestOnlyMonitorError.v1
        // on the live root, written 2026-07-26. Old artifacts are not outages.
        let temp = root_with(&[
            fresh_alarms(),
            (
                "monitor-error.json",
                r#"{"schema":"JopeTestOnlyMonitorError.v1","error":"ancient"}"#,
            ),
        ]);
        let two_days_ago = SystemTime::now() - std::time::Duration::from_secs(2 * 86_400);
        let file = std::fs::File::options()
            .write(true)
            .open(temp.path().join("ops/monitor-error.json"))
            .unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(two_days_ago))
            .unwrap();
        let args = args_for(temp.path());
        for _ in 0..5 {
            let outcome = evaluate(&args).unwrap();
            assert!(!outcome.failing, "{}", outcome.receipt);
        }
    }

    #[test]
    fn a_dead_alarm_scan_is_itself_a_failure() {
        let temp = root_with(&[]);
        let outcome = evaluate(&args_for(temp.path())).unwrap();
        assert!(outcome.failing);
        assert_eq!(outcome.receipt["unhealthy"][0]["kind"], "alarm_scan_stale");
    }

    #[test]
    fn the_receipt_keeps_one_shape_in_every_outcome() {
        let healthy = root_with(&[fresh_alarms()]);
        let ok = evaluate(&args_for(healthy.path())).unwrap().receipt;
        let broken = root_with(&[fresh_alarms(), ("queue-keeper-v3.json", KEEPER_ERROR)]);
        let bad = evaluate(&args_for(broken.path())).unwrap().receipt;
        let mut ok_keys: Vec<&String> = ok.as_object().unwrap().keys().collect();
        let mut bad_keys: Vec<&String> = bad.as_object().unwrap().keys().collect();
        ok_keys.sort();
        bad_keys.sort();
        assert_eq!(ok_keys, bad_keys);
        assert_eq!(ok["schema"], bad["schema"]);
    }
}
