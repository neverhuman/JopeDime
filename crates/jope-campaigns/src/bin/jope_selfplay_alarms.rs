//! jope_selfplay_alarms — one-shot health scan for the self-play flywheel.
//!
//! Read-only. Emits `ops/health-alarms.v1.json` (schema `JopeSelfplayHealthAlarms.v1`)
//! and always exits 0: an alarm writer that can itself fail the unit would be a
//! second thing to monitor. Driven by jope-selfplay-alarms.timer (60 s).

use anyhow::Result;
use jope_dime_regressor::training_history as th;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const ROOT: &str = "/home/ubuntu/jope-flywheel/test_exact_gen0";
const SCQ: &str = "/home/ubuntu/smart_cluster/target/release/scq";
const RUNS_ROOT: &str = "/home/ubuntu/smart_cluster/var/runs";
const CORPUS: &str = "/home/ubuntu/jope-flywheel/test_exact_gen0/corpus/strict-test-exact.sqlite";

/// (relative status file, max age seconds) — ages beyond these mean the owning
/// daemon is dead or wedged even if systemd thinks it is running.
const STATUS_AGES: &[(&str, u64)] = &[
    ("ops/queue-keeper-v3.json", 900),
    ("ops/micro-wave-supply-v1.json", 120),
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
    let output = Command::new(SCQ).arg("--json").args(args).output().ok()?;
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
        let sign = if value.as_bytes()[index] == b'+' { 1 } else { -1 };
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

/// Rust source tree, for deriving consumer field contracts. Overridable with
/// `JOPE_SRC_ROOT` so the check can run against a checkout under test.
const SRC_ROOT: &str = "/home/ubuntu/jope-dime";

/// Consecutive scans a status file may stay broken before the alarm escalates
/// from "this cycle failed" to "this daemon is wedged". The timer is 60 s.
const FAILING_CYCLES: u64 = 3;

/// Cadence budget for the content-frozen check, per file. Deliberately looser
/// than STATUS_AGES: this fires only when mtime is FRESH and the embedded
/// timestamp is old, which is the one thing mtime cannot see.
const CONTENT_FROZEN_SECS: &[(&str, u64)] = &[
    ("ops/queue-keeper-v3.json", 900),
    ("ops/micro-wave-supply-v1.json", 300),
    ("ops/task-learning-controller.v2.json", 300),
    ("ops/freshness-status.json", 600),
];

/// One live producer -> consumer contract.
struct Contract {
    /// Path relative to the flywheel root.
    file: &'static str,
    /// The one schema string a healthy document carries.
    schema: &'static str,
    /// Unit that writes it (for the alarm text — this is who to go look at).
    producer: &'static str,
    /// `(source file relative to SRC_ROOT, struct name)` of a consumer that
    /// deserializes this file into a strict struct. `None` means every known
    /// consumer reads it as a `serde_json::Value`, so the schema gate is the
    /// whole contract.
    consumer: Option<(&'static str, &'static str)>,
}

/// Verified against the live root on 2026-07-28 by reading the `schema` key of
/// every `$R/ops/*.json`. Consumers were found with
/// `grep -rn "ops/<file>" src/`.
const OPS_CONTRACTS: &[Contract] = &[
    Contract {
        file: "ops/queue-keeper-v3.json",
        schema: "JopeSelfplayQueueKeeper.v3",
        producer: "jope-selfplay-queue.service",
        // jope_selfplay_supply.rs:404-416 load_queue_status(); the struct whose
        // non-Option `queued_target` took supply down with the keeper.
        consumer: Some(("src/bin/jope_selfplay_supply.rs", "QueueStatus")),
    },
    Contract {
        file: "ops/micro-wave-supply-v1.json",
        schema: "JopeMicroWaveSupplyStatus.v1",
        producer: "jope-selfplay-supply.service",
        consumer: None,
    },
    Contract {
        file: "ops/task-learning-controller.v2.json",
        schema: "TaskLearningControllerStatus.v2",
        producer: "jope-selfplay-trainer-watchdog.service",
        consumer: None,
    },
    Contract {
        file: "ops/trainer-watchdog-state.v1.json",
        schema: "JopeContinuousTrainerWatchdogState.v1",
        producer: "jope-selfplay-trainer-watchdog.service",
        // jope_selfplay_trainer_watchdog.rs:690 load_state().
        consumer: Some(("src/bin/jope_selfplay_trainer_watchdog.rs", "DurableState")),
    },
    Contract {
        file: "ops/freshness-status.json",
        schema: "JopeSelfplayFreshnessStatus.v1",
        producer: "jope-selfplay-freshness.service",
        consumer: None,
    },
    Contract {
        file: "ops/freshness-state.v1.json",
        schema: "JopeFreshnessState.v1",
        producer: "jope-selfplay-freshness.service",
        // jope_selfplay_freshness.rs:328 — survives a wrong-shaped document
        // only because the struct is #[serde(default)]; the schema gate there
        // runs AFTER the deserialize, so it can never reject one.
        consumer: Some(("src/bin/jope_selfplay_freshness.rs", "DurableState")),
    },
    Contract {
        file: "ops/expansion-driver-v1.json",
        schema: "JopeSelfplayExpansionDriver.v1",
        producer: "jope-selfplay-expansion.service",
        consumer: None,
    },
    Contract {
        file: "ops/trainer-warm-start-guard.v1.json",
        schema: "JopeTrainerWarmStartGuard.v1",
        producer: "jope-selfplay-trainer-watchdog.service",
        consumer: None,
    },
    Contract {
        file: "ops/training-replica.v1.json",
        schema: "JopeTrainingReplicaStatus.v1",
        producer: "jope-training-replica.service",
        consumer: None,
    },
    Contract {
        file: "ops/flywheel-state.json",
        schema: "JopeSelfplayControllerState.v1",
        producer: "jope-selfplay-promotion.service",
        consumer: None,
    },
    Contract {
        file: "ops/propose-driver-v1.json",
        schema: "JopeProposeDriver.v1",
        producer: "jope-selfplay-propose.service",
        consumer: None,
    },
    Contract {
        file: "ops/seeding-curve.v1.json",
        schema: "JopeSeedingCurve.v1",
        producer: "jope-selfplay-ingestion.service",
        consumer: None,
    },
    Contract {
        file: "ops/status-v3.json",
        schema: "JopeSelfplayOperationalStatus.v3",
        producer: "jope-selfplay-ingestion.service",
        consumer: None,
    },
];

/// The JSON shape a serde field demands. `Any` means "presence only" — a nested
/// struct or enum whose internals this check deliberately does not model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Any,
    Str,
    Bool,
    Num,
    Arr,
    Obj,
}

impl Kind {
    fn matches(self, value: &Value) -> bool {
        match self {
            Kind::Any => !value.is_null(),
            Kind::Str => value.is_string(),
            Kind::Bool => value.is_boolean(),
            Kind::Num => value.is_number(),
            Kind::Arr => value.is_array(),
            Kind::Obj => value.is_object(),
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Kind::Any => "a value",
            Kind::Str => "string",
            Kind::Bool => "bool",
            Kind::Num => "number",
            Kind::Arr => "array",
            Kind::Obj => "object",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RequiredField {
    name: String,
    kind: Kind,
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn kind_of_type(ty: &str) -> Kind {
    let ty = ty.trim();
    if ty.starts_with("Option<") {
        return Kind::Any;
    }
    if ty.starts_with("Vec<") || ty.starts_with("[") {
        return Kind::Arr;
    }
    if ty.starts_with("HashMap<") || ty.starts_with("BTreeMap<") || ty.starts_with("Map<") {
        return Kind::Obj;
    }
    match ty {
        "String" | "&str" | "&'static str" | "PathBuf" | "Box<str>" => Kind::Str,
        "bool" => Kind::Bool,
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128"
        | "isize" | "f32" | "f64" => Kind::Num,
        _ => Kind::Any,
    }
}

/// Derive the fields a consumer's serde struct REQUIRES from that consumer's
/// own source. Returns `(declaration line, required fields)`.
///
/// Deliberately conservative: anything this parser is not sure about becomes
/// `Kind::Any` (presence only) or is dropped from the required set, so a false
/// alarm needs the source to actually say the field is mandatory. `None` means
/// the struct was not found or uses a serde attribute this cannot reason about
/// (`rename_all`), which the caller turns into its own visible alarm rather
/// than a silent skip.
fn derive_required_fields(source: &str, struct_name: &str) -> Option<(usize, Vec<RequiredField>)> {
    let lines: Vec<&str> = source.lines().collect();
    let head = lines.iter().position(|line| {
        let trimmed = line.trim_start();
        let trimmed = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
        trimmed.strip_prefix("struct ").and_then(|rest| {
            rest.split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
        }) == Some(struct_name)
    })?;

    // Container attributes sit immediately above the declaration.
    let mut container = String::new();
    let mut cursor = head;
    while cursor > 0 {
        let previous = lines[cursor - 1].trim();
        if previous.starts_with("#[") || previous.starts_with("///") || previous.starts_with("//") {
            container.push_str(previous);
            cursor -= 1;
        } else {
            break;
        }
    }
    if container.contains("rename_all") {
        return None;
    }
    // `#[serde(default)]` on the container makes every field optional — this is
    // exactly why jope_selfplay_freshness.rs:328 survives a wrong document.
    let container_default = container.contains("#[serde(default)]")
        || container.contains("serde(default,")
        || container.contains(", default)")
        || container.contains(", default,");

    let mut fields = Vec::new();
    let mut pending = String::new();
    for line in lines.iter().skip(head + 1) {
        let trimmed = line.trim();
        if trimmed == "}" {
            break;
        }
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("#[") {
            pending.push_str(trimmed);
            continue;
        }
        let attributes = std::mem::take(&mut pending);
        if attributes.contains("skip)")
            || attributes.contains("skip,")
            || attributes.contains("skip_deserializing")
        {
            continue;
        }
        let declaration = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
        let Some((raw_name, raw_type)) = declaration.split_once(':') else {
            continue;
        };
        let name = match attributes.find("rename = \"") {
            Some(index) => {
                let rest = &attributes[index + "rename = \"".len()..];
                match rest.find('"') {
                    Some(end) => rest[..end].to_owned(),
                    None => raw_name.trim().to_owned(),
                }
            }
            None => raw_name.trim().to_owned(),
        };
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let ty = raw_type.trim().trim_end_matches(',').trim();
        let optional =
            container_default || attributes.contains("default") || ty.starts_with("Option<");
        if optional {
            continue;
        }
        fields.push(RequiredField {
            name,
            kind: kind_of_type(ty),
        });
    }
    Some((head + 1, fields))
}

/// `(field, what is wrong)` for every required field the document breaks.
fn contract_violations(value: &Value, fields: &[RequiredField]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for field in fields {
        match value.get(&field.name) {
            None => out.push((field.name.clone(), "missing".to_owned())),
            Some(found) if !field.kind.matches(found) => out.push((
                field.name.clone(),
                format!(
                    "expected {}, found {}",
                    field.kind.describe(),
                    json_kind(found)
                ),
            )),
            Some(_) => {}
        }
    }
    out
}

/// Persisted streak counters. Small, rewritten every scan, and never load
/// bearing: losing it costs at most FAILING_CYCLES minutes of escalation.
fn load_contract_state(path: &Path) -> serde_json::Map<String, Value> {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| value.get("files").cloned())
        .and_then(|files| files.as_object().cloned())
        .unwrap_or_default()
}

fn truncate(text: &str, limit: usize) -> String {
    let cleaned = text.replace('\n', " ");
    match cleaned.char_indices().nth(limit) {
        Some((index, _)) => format!("{}…", &cleaned[..index]),
        None => cleaned,
    }
}

/// The whole point of workstream B: schema/state drift becomes a local, loud,
/// named failure instead of a silent cluster-wide outage.
fn check_status_contracts(root: &Path, alarms: &mut Alarms, severity: &str) {
    let source_root =
        PathBuf::from(std::env::var("JOPE_SRC_ROOT").unwrap_or_else(|_| SRC_ROOT.to_owned()));
    let state_path = root.join("ops/alarm-contract-state.v1.json");
    let previous = load_contract_state(&state_path);
    let mut next = serde_json::Map::new();
    let now = now_ms() as u64;

    for contract in OPS_CONTRACTS {
        let path = root.join(contract.file);
        let Ok(bytes) = std::fs::read(&path) else {
            // Absence is already covered by check_status_ages for the files it
            // watches; do not double-report here.
            continue;
        };
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => {
                alarms.push(
                    &format!("status_unparseable:{}", contract.file),
                    severity,
                    truncate(&error.to_string(), 160),
                    format!("valid JSON written by {}", contract.producer),
                );
                continue;
            }
        };

        // VERSION GATE FIRST, on a Value — the correct pattern, as at
        // jope_selfplay_supply.rs:423-430 and jope_queue_control.rs:1026. The
        // inverted form (deserialize, then check `schema`) is unreachable for
        // exactly the documents it is supposed to reject.
        let found = value.get("schema").and_then(Value::as_str).unwrap_or("");
        if found != contract.schema {
            let streak = previous
                .get(contract.file)
                .and_then(|row| row.get("cycles"))
                .and_then(Value::as_u64)
                .unwrap_or(0)
                + 1;
            let since = previous
                .get(contract.file)
                .and_then(|row| row.get("broken_since_ms"))
                .and_then(Value::as_u64)
                .unwrap_or(now);
            next.insert(
                contract.file.to_owned(),
                json!({ "cycles": streak, "broken_since_ms": since }),
            );
            let detail = value
                .get("error")
                .and_then(Value::as_str)
                .map(|error| format!("; error={}", truncate(error, 160)))
                .unwrap_or_default();
            let name = if found.ends_with("Error.v1") {
                format!("status_error_doc:{}", contract.file)
            } else if found.is_empty() {
                format!("status_schema_missing:{}", contract.file)
            } else {
                format!("status_schema_mismatch:{}", contract.file)
            };
            alarms.push(
                &name,
                severity,
                format!("schema={found:?}{detail}"),
                format!("schema={:?} from {}", contract.schema, contract.producer),
            );
            if streak >= FAILING_CYCLES {
                alarms.push(
                    &format!("status_failing:{}", contract.file),
                    severity,
                    format!(
                        "{streak} consecutive scans off-schema, {}s since first failure",
                        now.saturating_sub(since) / 1000
                    ),
                    format!("<{FAILING_CYCLES} scans; {} is wedged", contract.producer),
                );
            }
            continue;
        }

        // In-schema but self-reporting degradation: the shape stayed stable, so
        // every consumer still parses, and the alarm still fires. This is the
        // pattern jope_selfplay_trainer_watchdog.rs:372 now uses, and the one
        // proposed for the queue keeper and supply.
        if value.get("degraded").and_then(Value::as_bool) == Some(true) {
            alarms.push(
                &format!("status_degraded:{}", contract.file),
                severity,
                value
                    .get("error")
                    .and_then(Value::as_str)
                    .map_or_else(|| "degraded=true".to_owned(), |error| truncate(error, 160)),
                format!("{} completing its cycle", contract.producer),
            );
        }

        // Fresh mtime, frozen content. A daemon that rewrites an identical
        // document forever is invisible to every age-based guard in this file.
        if let Some((_, budget)) = CONTENT_FROZEN_SECS
            .iter()
            .find(|(f, _)| *f == contract.file)
        {
            let stamp = value
                .get("generated_at_ms")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let mtime_age = file_age_secs(&path).unwrap_or(0);
            if stamp > 0 && mtime_age <= *budget {
                let content_age = now.saturating_sub(stamp) / 1000;
                if content_age > *budget {
                    alarms.push(
                        &format!("status_content_frozen:{}", contract.file),
                        severity,
                        format!("generated_at_ms {content_age}s old, mtime {mtime_age}s old"),
                        format!("generated_at_ms within {budget}s"),
                    );
                }
            }
        }

        // The consumer's own struct, applied to the bytes on disk.
        if let Some((relative, struct_name)) = contract.consumer {
            let source_path = source_root.join(relative);
            let Ok(source) = std::fs::read_to_string(&source_path) else {
                alarms.push(
                    &format!("contract_source_missing:{relative}"),
                    "warning",
                    format!("cannot read {}", source_path.display()),
                    "consumer source readable for contract derivation".into(),
                );
                continue;
            };
            let Some((line, required)) = derive_required_fields(&source, struct_name) else {
                alarms.push(
                    &format!("contract_underivable:{relative}"),
                    "warning",
                    format!("struct {struct_name} not found or uses rename_all"),
                    "a serde struct this check can model".into(),
                );
                continue;
            };
            for (field, problem) in contract_violations(&value, &required) {
                alarms.push(
                    &format!("contract_violation:{}#{field}", contract.file),
                    severity,
                    problem,
                    format!("required by {relative}:{line} struct {struct_name}"),
                );
            }
        }
    }

    let state = json!({
        "schema": "JopeSelfplayAlarmContractState.v1",
        "generated_at_ms": now,
        "files": Value::Object(next),
    });
    let temporary = root.join(format!(
        "ops/.alarm-contract-state.v1.json.{}",
        std::process::id()
    ));
    if let Ok(bytes) = serde_json::to_vec(&state) {
        if std::fs::write(&temporary, bytes).is_ok() {
            let _ = std::fs::rename(&temporary, &state_path);
        }
    }
}

fn node_f64(node: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|k| node.get(*k).and_then(Value::as_f64))
}

fn check_nodes(alarms: &mut Alarms) -> Option<Value> {
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

fn check_scq_and_jobs(alarms: &mut Alarms) {
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
            std::fs::read(Path::new(ROOT).join("ops/task-learning-controller.v2.json"))
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
    let supply_doc = Path::new(ROOT).join("ops/micro-wave-supply-v1.json");
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
                    let error = doc.get("error").and_then(Value::as_str).unwrap_or("unknown");
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
    let newest_wave_age = std::fs::read_dir(Path::new(ROOT).join("waves"))
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
            .unwrap_or_else(|| Path::new(RUNS_ROOT).join(id));
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
fn check_completed_conditionality(id: &str, state: &str, run_dir: &Path, alarms: &mut Alarms) {
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
fn newest_progress_file(run_dir: &Path) -> Option<(SystemTime, PathBuf)> {
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
fn newest_progress_age(run_dir: &Path) -> Option<u64> {
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
fn final_conditional_advantage(run_dir: &Path) -> Option<(f64, u64)> {
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
fn read_tail(path: &Path, cap: u64) -> Option<String> {
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

fn walk_progress(dir: &Path, depth: usize, newest: &mut Option<(SystemTime, PathBuf)>) {
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

fn check_local_memory(alarms: &mut Alarms) {
    let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") else {
        return;
    };
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("SwapFree:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(u64::MAX);
            if kb != u64::MAX && kb / 1024 < SWAP_FREE_MIN_MB {
                alarms.push(
                    "swap_exhausted:xbabe2",
                    "warning",
                    format!("{} MB free", kb / 1024),
                    format!(">={SWAP_FREE_MIN_MB} MB"),
                );
            }
        }
    }
}

/// Silent loss of training metrics is the failure this checks for.
///
/// Every per-epoch train/validation loss the DL model ever produced is supposed
/// to end up in `training/history/`, because the run directory that holds the
/// original is a ~12 GB garbage-collection target (eight earlier runs already
/// lost their curves that way — their specs survive, their metrics do not).
/// Three ways that can quietly stop working, each made loud here:
///
/// 1. `training_history_stale` — the capture timer stopped running at all.
/// 2. `training_history_unarchived:<run>` — a run directory holds rows the
///    archive does not, i.e. it is exactly one GC pass from losing them.
/// 3. `training_history_not_advancing:<run>` — the trainer is still writing
///    epochs but the archive stopped keeping up with that particular run.
/// 4. `training_history_node_unreachable:<node>` — capture could not sweep a
///    node a trainer is allowed to run on, so coverage there is unknown.
///
/// A shortfall that is only a trailing INCOMPLETE row is deliberately NOT one of
/// those. That row never finished being written, every complete row is archived,
/// and the hole is on the truncation ledger; reporting it as critical would pin
/// a permanent red alarm that blocks the expansion driver (which halts on any
/// critical) and teaches everyone to ignore this family of alarms — while a real
/// unarchived run hides behind it. It gets its own informational alarm instead.
fn check_training_history(root: &Path, runs_root: &Path, alarms: &mut Alarms) {
    let archive = th::archive_root(root);
    let index_path = archive.join(th::INDEX_FILE);
    match file_age_secs(&index_path) {
        None => {
            alarms.push(
                "training_history_missing",
                "critical",
                "no training/history/index.v1.json".into(),
                "capture timer writes it every 3 min".into(),
            );
            return;
        }
        Some(age) if age > HISTORY_INDEX_STALE_SECS => alarms.push(
            "training_history_stale",
            "critical",
            format!("index {age}s old"),
            format!("<{HISTORY_INDEX_STALE_SECS}s"),
        ),
        _ => {}
    }

    let states = th::load_segments(&archive);
    let now = th::now_ms();

    // A trainer the scheduler says finished, with nothing in the archive under
    // its name at all. Its run directory is gone or was never scanned, so its
    // curve is already lost or about to be — either way it must be loud.
    if let Some(snapshot) = scq_json(&["snapshot"]) {
        let snap = snapshot.get("Snapshot").unwrap_or(&snapshot);
        let archived_runs: std::collections::HashSet<&str> =
            states.keys().map(|key| key.run_id.as_str()).collect();
        for job in snap
            .get("jobs")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let Some(id) = job.get("id").and_then(Value::as_str) else {
                continue;
            };
            if !id.contains(th::TRAIN_MARKER)
                || job.get("state").and_then(Value::as_str) != Some("completed")
                || archived_runs.contains(id)
            {
                continue;
            }
            alarms.push(
                &format!("training_history_run_absent:{id}"),
                "critical",
                "completed trainer has no archived metrics".into(),
                "every completed run present in training/history".into(),
            );
        }
    }
    // Read-only: the alarm never writes to the archive, it only compares what
    // the run directories hold against what the archive holds.
    for run_id in th::discover_trainer_runs(runs_root).unwrap_or_default() {
        for found in th::scan_run_dir(runs_root, &run_id) {
            let held = th::archived_bytes(&archive, &found.key);
            if held >= found.len {
                continue;
            }
            let source_age = file_age_secs(&found.path).unwrap_or(u64::MAX);
            let advance_age = states
                .get(&found.key)
                .map(|state| now.saturating_sub(state.last_advance_ms) / 1000)
                .unwrap_or(u64::MAX);

            // Is the shortfall data, or a row that was never finished? Reading
            // the bytes is the only way to know: a gap containing a newline is
            // COMPLETE rows and stays critical; a lone unterminated fragment is
            // the truncated tail of a killed trainer.
            let residual =
                th::classify_residual(&th::FileSource(found.path.clone()), held, found.len);
            if let Ok(th::Residual::PartialRow(fragment)) = &residual {
                let recorded = states
                    .get(&found.key)
                    .map(|state| {
                        state.truncation_count > 0 && held + state.truncated_tail_bytes >= found.len
                    })
                    .unwrap_or(false);
                if recorded {
                    alarms.push(
                        &format!("training_history_truncated_tail:{}", found.key.run_id),
                        "info",
                        format!(
                            "{}B trailing incomplete row written off; {} complete rows archived",
                            fragment.len(),
                            states
                                .get(&found.key)
                                .map(|state| state.archived_rows)
                                .unwrap_or(0)
                        ),
                        "complete rows archived and the hole on the truncation ledger".into(),
                    );
                    continue;
                }
                if source_age >= th::TAIL_QUIESCENT_SECS {
                    // Quiescent but not yet on the ledger: the next capture
                    // cycle records it. Visible, but not a data-loss event.
                    alarms.push(
                        &format!(
                            "training_history_truncated_tail_pending:{}",
                            found.key.run_id
                        ),
                        "warning",
                        format!(
                            "{}B trailing incomplete row, source idle {source_age}s, not yet recorded",
                            fragment.len()
                        ),
                        "capture records it within one 3 min cycle".into(),
                    );
                    continue;
                }
                // Still fresh: a row genuinely being written right now. Fall
                // through to the live-run checks below.
            }

            // A run that is actively emitting but whose archive has not moved in
            // half an hour is a capture failure, not a mid-write row.
            if source_age < HISTORY_SOURCE_FRESH_SECS && advance_age > HISTORY_ADVANCE_GRACE_SECS {
                alarms.push(
                    &format!("training_history_not_advancing:{}", found.key.run_id),
                    "critical",
                    format!(
                        "archive {held}B behind source {}B; last append {advance_age}s ago",
                        found.len
                    ),
                    format!("archive advances within {HISTORY_ADVANCE_GRACE_SECS}s"),
                );
                continue;
            }
            // Not being written any more, yet the archive is short: whatever is
            // in that gap dies with the directory.
            if source_age >= HISTORY_SOURCE_FRESH_SECS {
                alarms.push(
                    &format!("training_history_unarchived:{}", found.key.run_id),
                    "critical",
                    format!(
                        "{}B of metrics unarchived ({}, source idle {source_age}s)",
                        found.len.saturating_sub(held),
                        residual
                            .as_ref()
                            .map(|r| r.reason())
                            .unwrap_or("unreadable")
                    ),
                    "every completed run fully archived".into(),
                );
            }
        }
    }

    // Remote coverage. A trainer spec pins `xbabe1|xbabe2|xbabe3`, so capture
    // sweeps the other nodes over ssh; a node it cannot reach is a node whose
    // coverage is unknown, and that must be said out loud rather than shrinking
    // the sweep silently. Warning, not critical: nothing is lost unless a
    // trainer actually ran there, and that case is already caught cluster-wide
    // by `training_history_run_absent:<id>` from the scq snapshot. Keeping a
    // routinely-rebooting node from halting the expansion driver is the point.
    for (node, status) in th::load_nodes(&archive) {
        if status.reachable {
            continue;
        }
        let observed = if status.last_success_ms == 0 {
            format!(
                "never swept ({} consecutive failures)",
                status.consecutive_failures
            )
        } else {
            format!(
                "last swept {}s ago ({} consecutive failures): {}",
                now.saturating_sub(status.last_success_ms) / 1000,
                status.consecutive_failures,
                status.last_error
            )
        };
        alarms.push(
            &format!("training_history_node_unreachable:{node}"),
            "warning",
            observed,
            "every trainer-eligible node swept every 3 min".into(),
        );
    }
}

/// Lift the trainer watchdog's warm-start guard verdict into the health file.
///
/// The watchdog refuses to pin a tensor-deficient warm-start checkpoint (one
/// missing the binary branch); when no full checkpoint exists at all it stops
/// admitting trainers, and that stop must be loud rather than a silent gap in
/// the DL lane.
fn check_warm_start_guard(root: &Path, alarms: &mut Alarms) {
    let path = root.join("ops/trainer-warm-start-guard.v1.json");
    let Ok(bytes) = std::fs::read(&path) else {
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    if value.get("schema").and_then(Value::as_str) != Some("JopeTrainerWarmStartGuard.v1") {
        return;
    }
    if value.get("deficient").and_then(Value::as_bool) != Some(true) {
        return;
    }
    alarms.push(
        "trainer_warm_start_deficient",
        "critical",
        value
            .get("observed")
            .and_then(Value::as_str)
            .unwrap_or("no full warm-start checkpoint")
            .to_string(),
        value
            .get("threshold")
            .and_then(Value::as_str)
            .unwrap_or("full-tensor checkpoint")
            .to_string(),
    );
}

/// Proposal-set health, read off the summary the generator already writes.
///
/// Context. Until 2026-07-28 a single dataset that the policy could not give
/// `--proposals-per-dataset` distinct genomes for was a FATAL error inside
/// `scripts/test_exact_dl_generate.py`; `seed_next_strict_wave` failed 848
/// consecutive times on `binary:0208f980ee49a057 produced only 34 unique
/// proposals` and the tournament parked at phase_index 9. That check is now a
/// deterministic cross-dataset backfill, so the job completes — which is
/// exactly why the shortfall has to become loud somewhere else. This is that
/// somewhere else.
///
/// What it reads (`proposals/current.json` -> `summary`, both the controller
/// micro-cohort lane and the propose driver publish it):
///   * `deficiency_by_task[task].deficient_fraction` — share of that task's
///     contexts the model could not fill on its own.
///   * `deficiency_by_task[task].min_native_unique` — the worst context.
///   * `native_head_counts_by_task[task]` — terminal-head histogram over
///     MODEL-SAMPLED rows only. One head at ~100% is mode collapse.
///   * `proposal_validity_by_task[task]` — share of decoded sequences whose
///     head is even admissible for the task.
///
/// Measured basis for the thresholds, both sets from checkpoint fbcd5459:
///   * micro-cohort `3a89b11ac7da` (128 contexts, 2026-07-28T23:11Z):
///     regression 0/64 contexts deficient, binary 19/64 (0.297), worst context
///     20 of 64 native uniques, 583 of 8,192 rows borrowed; binary head mix
///     logit_ridge 0.867 / tab_boost 0.093 / extra_trees 0.024 / gam_boost
///     0.015 / logistic 0.0002.
///   * whole-catalog `catalog-62054d7b05cb` (14,629 contexts, same day
///     20:47Z): binary heads 132,669 of 132,669 `logistic` (1.0000) and binary
///     decode validity 0.0407 — 96% of everything the model emitted for a
///     binary context carried a regression head.
/// So a 0.10 deficient fraction is already off the healthy branch, 1.0000 head
/// share is a state this system has genuinely reached, and the two coexist
/// under one checkpoint: the alarms have to be per task and per set.
const PROPOSAL_DEFICIENT_FRACTION_WARN: f64 = 0.10;
const PROPOSAL_DEFICIENT_FRACTION_CRIT: f64 = 0.50;
/// A context whose model-sampled uniques fall to a quarter of the budget is not
/// "thin", it is a collapsed conditional distribution.
const PROPOSAL_MIN_NATIVE_RATIO_CRIT: f64 = 0.25;
/// One terminal head holding this share of a task's model-sampled proposals is
/// mode collapse: the binary allowlist has six admissible heads and regression
/// nine, and a healthy regression branch measures 0.23 for its modal head.
///
/// The WARN floor is set ABOVE the binary branch's normal skew, not at some
/// round number. Every non-collapsed binary reading taken on 2026-07-28 sits at
/// 0.878 (micro-cohort, 128 contexts) to 0.883 (4-context offline run) for
/// `logit_ridge`; the collapsed whole-catalog set sits at 1.0000 for `logistic`.
/// A floor at 0.80 would therefore be lit permanently and mean nothing. 0.90
/// clears the measured healthy band and still fires long before the 0.95
/// critical.
const PROPOSAL_HEAD_SHARE_CRIT: f64 = 0.95;
const PROPOSAL_HEAD_SHARE_WARN: f64 = 0.90;
/// Below this the decoder is spending most of its samples on heads that cannot
/// run for the task at all.
const PROPOSAL_VALIDITY_WARN: f64 = 0.50;

fn proposal_summary_path(root: &Path) -> Option<PathBuf> {
    let pointer_path = root.join("proposals/current.json");
    let bytes = std::fs::read(&pointer_path).ok()?;
    let pointer: Value = serde_json::from_slice(&bytes).ok()?;
    // Preferred: the summary the publisher recorded. It can live under
    // training/tournament-work/, which GC may reap, so fall back to the summary
    // that sits beside the archived proposal file itself.
    if let Some(path) = pointer.get("summary").and_then(Value::as_str) {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let proposals = pointer.get("proposals").and_then(Value::as_str)?;
    let sibling = PathBuf::from(proposals.strip_suffix(".jsonl")?.to_owned() + ".summary.json");
    sibling.is_file().then_some(sibling)
}

fn check_proposal_deficiency(root: &Path, alarms: &mut Alarms) {
    let Some(path) = proposal_summary_path(root) else {
        return;
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    if value.get("schema").and_then(Value::as_str) != Some("JopeTestOnlyDlProposalSet.v1") {
        return;
    }
    let budget = value
        .get("proposals_per_dataset")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let label = value
        .get("model_checkpoint_sha256")
        .and_then(Value::as_str)
        .map(|digest| digest[..digest.len().min(12)].to_owned())
        .unwrap_or_else(|| "unknown".to_owned());

    // 1. Deficiency: how much of the set the model could not produce itself.
    // Pre-deficiency summaries have no `deficiency_by_task`; those runs are
    // simply not measured here rather than silently reported as healthy.
    if let Some(by_task) = value.get("deficiency_by_task").and_then(Value::as_object) {
        for (task, row) in by_task {
            let contexts = row.get("contexts").and_then(Value::as_i64).unwrap_or(0);
            if contexts <= 0 {
                continue;
            }
            let deficient = row
                .get("deficient_contexts")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let fraction = row
                .get("deficient_fraction")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let worst = row
                .get("min_native_unique")
                .and_then(Value::as_i64)
                .unwrap_or(budget);
            let borrowed = row
                .get("backfilled_rows")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let unfilled = row
                .get("unfilled_contexts")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let worst_ratio = if budget > 0 {
                worst as f64 / budget as f64
            } else {
                1.0
            };
            let severity = if fraction >= PROPOSAL_DEFICIENT_FRACTION_CRIT
                || worst_ratio <= PROPOSAL_MIN_NATIVE_RATIO_CRIT
            {
                Some("critical")
            } else if fraction >= PROPOSAL_DEFICIENT_FRACTION_WARN {
                Some("warning")
            } else {
                None
            };
            if let Some(severity) = severity {
                alarms.push(
                    &format!("proposal_deficiency:{task}"),
                    severity,
                    format!(
                        "{deficient}/{contexts} contexts short ({:.3}); worst {worst}/{budget} \
                         native uniques; {borrowed} rows borrowed cross-dataset; \
                         checkpoint {label}",
                        fraction
                    ),
                    format!(
                        "<{:.2} deficient fraction and worst context >{:.2} of budget",
                        PROPOSAL_DEFICIENT_FRACTION_WARN, PROPOSAL_MIN_NATIVE_RATIO_CRIT
                    ),
                );
            }
            if unfilled > 0 {
                // The cross-dataset pool itself ran dry: rows below the budget
                // reached disk, which every wave consumer rejects.
                alarms.push(
                    &format!("proposal_unfilled:{task}"),
                    "critical",
                    format!("{unfilled} contexts below {budget} rows after backfill"),
                    "0 contexts below budget".into(),
                );
            }
        }
    }

    // 2. Mode collapse: the histogram, not the count. A set can be 100%
    // complete and still be one genome shape repeated.
    if let Some(by_task) = value
        .get("native_head_counts_by_task")
        .and_then(Value::as_object)
    {
        for (task, histogram) in by_task {
            let Some(histogram) = histogram.as_object() else {
                continue;
            };
            let total: i64 = histogram.values().filter_map(Value::as_i64).sum();
            if total <= 0 {
                continue;
            }
            let mut ranked: Vec<(&String, i64)> = histogram
                .iter()
                .filter_map(|(head, count)| count.as_i64().map(|count| (head, count)))
                .collect();
            ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
            let (head, count) = ranked[0];
            let share = count as f64 / total as f64;
            let severity = if share >= PROPOSAL_HEAD_SHARE_CRIT {
                Some("critical")
            } else if share >= PROPOSAL_HEAD_SHARE_WARN {
                Some("warning")
            } else {
                None
            };
            if let Some(severity) = severity {
                let mix = ranked
                    .iter()
                    .take(4)
                    .map(|(head, count)| format!("{head} {:.4}", *count as f64 / total as f64))
                    .collect::<Vec<_>>()
                    .join(", ");
                alarms.push(
                    &format!("proposal_head_collapse:{task}"),
                    severity,
                    format!(
                        "{head} holds {share:.4} of {total} model-sampled {task} proposals \
                         across {} distinct heads ({mix}); checkpoint {label}",
                        ranked.len()
                    ),
                    format!("modal head <{:.2}", PROPOSAL_HEAD_SHARE_WARN),
                );
            }
        }
    }

    // 3. Admissibility: how much of the decoder's output the task filter throws
    // away before diversity is even considered.
    if let Some(by_task) = value
        .get("proposal_validity_by_task")
        .and_then(Value::as_object)
    {
        for (task, reading) in by_task {
            let Some(validity) = reading.as_f64() else {
                continue;
            };
            if validity >= PROPOSAL_VALIDITY_WARN {
                continue;
            }
            alarms.push(
                &format!("proposal_task_validity:{task}"),
                if validity < 0.10 { "critical" } else { "warning" },
                format!("{validity:.4} of decoded {task} sequences carry an admissible head; checkpoint {label}"),
                format!(">={PROPOSAL_VALIDITY_WARN:.2}"),
            );
        }
    }
}

/// The two guardrails `ops/generalization-plan.md` asked for, plus the one that
/// would have caught the outage they were written during.
///
/// 1. `corpus_dataset_count_fell` — evidence only ever accumulates, so a corpus
///    whose dataset count DROPS between freezes is always a sampler or catalog
///    defect. `test_exact_train.publish_freeze_guard` judges this at freeze time
///    and leaves the verdict in `ops/corpus-freeze-guard.v1.json`; this lifts it
///    where operators already look. It is an alarm and not a refusal to freeze
///    on purpose: refusing would pin the trainer to the OLD corpus, which is the
///    starvation the guardrail exists to detect.
/// 2. `corpus_split_median_divergence` — train and validation per-dataset
///    epoch-draw medians within 2x. `build_corpus` already QUARANTINES a freeze
///    that violates this, so a live alarm here means an older corpus is still
///    the one being trained on.
/// 3. `corpus_freeze_starved` — the alarm neither guardrail would have raised.
///    Both of them only fire when a freeze RUNS. For two days the live failure
///    was that no freeze ran at all: `freeze_corpus` is `PHASES[0]` of the
///    promotion tournament and a new tournament only starts once the previous
///    one terminates, so a tournament parked at `phase_index 9` pinned the
///    corpus at watermark 1,805,437 while the observations table reached
///    4,690,000+. Nothing anywhere went red. This measures the pointer gap
///    directly, so it is true whether or not anybody freezes.
fn check_corpus_freeze_guard(root: &Path, alarms: &mut Alarms) {
    let guard_path = root.join("ops/corpus-freeze-guard.v1.json");
    if let Ok(bytes) = std::fs::read(&guard_path) {
        if let Ok(guard) = serde_json::from_slice::<Value>(&bytes) {
            if guard.get("regressed").and_then(Value::as_bool) == Some(true) {
                alarms.push(
                    "corpus_dataset_count_fell",
                    "critical",
                    guard
                        .get("observed")
                        .and_then(Value::as_str)
                        .unwrap_or("corpus dataset count regressed")
                        .to_owned(),
                    guard
                        .get("threshold")
                        .and_then(Value::as_str)
                        .unwrap_or("no prior freeze recorded")
                        .to_owned(),
                );
            }
            if let Some(scopes) = guard
                .get("split_median_divergence")
                .and_then(Value::as_object)
            {
                for (scope, values) in scopes {
                    let Some(ratio) = values.get("ratio").and_then(Value::as_f64) else {
                        continue;
                    };
                    if ratio <= SPLIT_MEDIAN_RATIO_MAX {
                        continue;
                    }
                    let train = values.get("train").and_then(Value::as_f64).unwrap_or(0.0);
                    let validation = values
                        .get("validation")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    alarms.push(
                        &format!("corpus_split_median_divergence:{scope}"),
                        "critical",
                        format!(
                            "{scope} epoch-draw medians train {train} vs validation \
                             {validation} ({ratio:.2}x)"
                        ),
                        format!("<={SPLIT_MEDIAN_RATIO_MAX:.1}x"),
                    );
                }
            }
        }
    }

    // The starvation measure. Both pointers are single small JSON files.
    let frozen = std::fs::read(root.join("training/current-corpus.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| value.get("watermark").and_then(Value::as_u64));
    let labels = std::fs::read(root.join("ops/last-ingest.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| {
            value
                .get("exact_rows_total")
                .or_else(|| value.get("watermark"))
                .and_then(Value::as_u64)
        });
    if let (Some(frozen), Some(labels)) = (frozen, labels) {
        let gap = labels.saturating_sub(frozen);
        let severity = if gap >= CORPUS_STARVED_CRIT {
            Some("critical")
        } else if gap >= CORPUS_STARVED_WARN {
            Some("warning")
        } else {
            None
        };
        if let Some(severity) = severity {
            alarms.push(
                "corpus_freeze_starved",
                severity,
                format!(
                    "{gap} labels outside the frozen corpus (pointer {frozen}, ingested \
                     {labels}); the trainer cannot see any of them"
                ),
                format!("<{CORPUS_STARVED_WARN}"),
            );
        }
    }

    // Only judged when the freezer is actually deployed on this root, so a root
    // without it does not carry a permanent alarm for a unit it never had.
    let status = root.join("ops/corpus-freezer-v1.json");
    if status.is_file() {
        if let Some(age) = file_age_secs(&status) {
            if age > CORPUS_FREEZER_STALE_SECS {
                alarms.push(
                    "status_stale:ops/corpus-freezer-v1.json",
                    "warning",
                    format!("{age}s"),
                    format!("{CORPUS_FREEZER_STALE_SECS}s"),
                );
            }
        }
        if let Ok(bytes) = std::fs::read(&status) {
            if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
                if value.get("lifecycle").and_then(Value::as_str) == Some("error") {
                    alarms.push(
                        "corpus_freezer_error",
                        "critical",
                        value
                            .get("error")
                            .and_then(Value::as_str)
                            .unwrap_or("corpus freezer reported an error")
                            .to_owned(),
                        "lifecycle=ok".to_owned(),
                    );
                }
                if let Some(alarm) = value.get("alarm").and_then(Value::as_str) {
                    alarms.push(
                        alarm,
                        "critical",
                        value
                            .get("detail")
                            .and_then(Value::as_str)
                            .unwrap_or(alarm)
                            .to_owned(),
                        "corpus pointer <= exact_rows_total".to_owned(),
                    );
                }
            }
        }
    }
}

/// A trainer submission intent that can NEVER succeed, however long it retries.
///
/// `jope_selfplay_trainer_watchdog.rs::render_spec` sha256s every spec input at
/// render time; `write_immutable` (:1512) then refuses to rewrite a spec whose
/// content differs, and `validate_intent` (:1355) re-checks only the watermark,
/// the checkpoint and the job id — never the input digests. So if any declared
/// input changes in the seconds between render and a successful submit, `scq
/// check` rejects the spec with `input-hash-mismatch` forever and the watchdog
/// retries the identical dead spec every ~13 s until a human clears the state.
///
/// This is not theoretical and it is not rare: it happened TWICE inside 90
/// minutes on 2026-07-28 (22:53Z and 23:33Z), both times because a workstream
/// edited `scripts/test_exact_train.py` while the watchdog was live. The first
/// occurrence cost 34 minutes of trainer downtime and was found only by reading
/// the journal by hand — every alarm in this file stayed green, because from the
/// outside a watchdog retrying looks exactly like a watchdog working.
///
/// The durable fix belongs in the watchdog (drop the intent and re-render when
/// the rejection is `input-hash-mismatch`). Until then, this makes it loud in
/// 60 seconds and prints the recovery.
fn check_trainer_intent(root: &Path, alarms: &mut Alarms) {
    let path = root.join("ops/trainer-watchdog-state.v1.json");
    let Ok(bytes) = std::fs::read(&path) else {
        return;
    };
    let Ok(state) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    let Some(intent) = state.get("intent").filter(|value| !value.is_null()) else {
        return;
    };
    let Some(error) = intent.get("last_error").and_then(Value::as_str) else {
        return;
    };
    if !error.contains("input-hash-mismatch") {
        return;
    }
    let job = intent
        .get("job_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let spec = intent.get("spec").and_then(Value::as_str).unwrap_or("");
    alarms.push(
        "trainer_intent_unsubmittable",
        "critical",
        format!(
            "{job} can never submit: a spec input changed after the spec was frozen. \
             Recover by stopping jope-selfplay-trainer-watchdog, setting \
             \"intent\": null in ops/trainer-watchdog-state.v1.json, and restarting. \
             spec={spec}"
        ),
        "no unsubmittable trainer intent".to_owned(),
    );
}

fn check_corpus_size(alarms: &mut Alarms) {
    if let Ok(meta) = std::fs::metadata(CORPUS) {
        if meta.len() > CORPUS_WARN_BYTES {
            alarms.push(
                "corpus_size",
                "warning",
                format!("{:.1} GiB", meta.len() as f64 / f64::from(1u32 << 30)),
                format!("{} GiB", CORPUS_WARN_BYTES / (1 << 30)),
            );
        }
    }
}

/// The archive surviving GC is not the same as the archive surviving the disk.
///
/// `training/history/` lives on xbabe2's `/dev/nvme2n1p2` — the same filesystem
/// as the `var/runs` tree it exists to outlive. `jope-training-replica.timer`
/// mirrors it to xbabe3 and xbabe4 every 15 minutes and writes
/// `ops/training-replica.v1.json` every cycle, pass or fail. Three ways that
/// can quietly stop, each made loud here:
///
/// 1. `training_replica_missing` / `_stale` — the replication timer stopped, so
///    the archive is silently back to a single copy.
/// 2. `training_replica_target_stale:<node>` — one target has not accepted a
///    complete, verified publish for hours (unreachable, full, or refusing).
/// 3. `training_replica_degraded` — fewer than two targets are healthy. Two is
///    the point: one replica plus the source is a single failure away from a
///    single copy.
fn check_training_replica(root: &Path, alarms: &mut Alarms) {
    let path = root.join("ops/training-replica.v1.json");
    let Ok(bytes) = std::fs::read(&path) else {
        alarms.push(
            "training_replica_missing",
            "critical",
            "no ops/training-replica.v1.json".into(),
            "replica timer writes it every 15 min".into(),
        );
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    if value.get("schema").and_then(Value::as_str) != Some("JopeTrainingReplicaStatus.v1") {
        return;
    }
    if let Some(age) = file_age_secs(&path) {
        if age > REPLICA_RECEIPT_STALE_SECS {
            alarms.push(
                "training_replica_stale",
                "critical",
                format!("receipt {age}s old"),
                format!("<{REPLICA_RECEIPT_STALE_SECS}s"),
            );
        }
    }

    let now = now_ms() as u64;
    let targets = value
        .get("targets")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for target in targets {
        let name = target.get("target").and_then(Value::as_str).unwrap_or("?");
        if target.get("ok").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        // A target that has never succeeded reports 0; treat that as infinitely
        // stale rather than as a fresh success at the epoch.
        let last = target
            .get("last_success_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let age = if last == 0 {
            u64::MAX
        } else {
            now.saturating_sub(last) / 1000
        };
        if age > REPLICA_TARGET_STALE_SECS {
            let observed = match last {
                0 => "never replicated successfully".to_string(),
                _ => format!("last good publish {age}s ago"),
            };
            alarms.push(
                &format!("training_replica_target_stale:{name}"),
                "critical",
                format!(
                    "{observed}; {}",
                    target
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("no error recorded")
                ),
                format!("a verified publish within {REPLICA_TARGET_STALE_SECS}s"),
            );
        }
    }

    let healthy = value
        .get("healthy_targets")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if healthy < REPLICA_MIN_HEALTHY {
        alarms.push(
            "training_replica_degraded",
            "critical",
            format!(
                "{healthy} of {} replica targets healthy",
                value
                    .get("target_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            ),
            format!(">={REPLICA_MIN_HEALTHY} off-node replicas"),
        );
    }
}

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
        PathBuf::from(root.unwrap_or_else(|| ROOT.to_owned())),
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
    check_training_history(&root, Path::new(RUNS_ROOT), &mut alarms);
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
mod tests {
    use super::*;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // Content contracts
    // -----------------------------------------------------------------------

    /// The real keeper error document, byte for byte off the live root at
    /// 15:13 on 2026-07-28, while all 1,052 scheduler slots were idle.
    const LIVE_KEEPER_ERROR_DOC: &str = r#"{"admission_allowed":false,"error":"native attempt submission transition count mismatch (attempt_no column reservation_owner)","generated_at_ms":1785251597154,"mode":"submit","schema":"JopeSelfplayQueueKeeperError.v1"}"#;

    fn contract_root(temp: &TempDir) -> PathBuf {
        let root = temp.path().to_path_buf();
        std::fs::create_dir_all(root.join("ops")).unwrap();
        root
    }

    fn scan(root: &Path) -> Alarms {
        let mut alarms = Alarms { rows: Vec::new() };
        check_status_contracts(root, &mut alarms, "critical");
        alarms
    }

    #[test]
    fn an_error_document_on_disk_is_an_alarm_however_fresh_the_file_is() {
        let temp = TempDir::new().unwrap();
        let root = contract_root(&temp);
        std::fs::write(root.join("ops/queue-keeper-v3.json"), LIVE_KEEPER_ERROR_DOC).unwrap();
        let alarms = scan(&root);
        let names = alarm_names(&alarms);
        assert!(
            names.contains(&"status_error_doc:ops/queue-keeper-v3.json".to_owned()),
            "{names:?}"
        );
        // check_status_ages sees a file written microseconds ago and says
        // nothing at all — that is the blind spot this replaces.
        let mut age_only = Alarms { rows: Vec::new() };
        check_status_ages(&root, &mut age_only);
        assert!(!alarm_names(&age_only)
            .iter()
            .any(|name| name.contains("queue-keeper")));
    }

    #[test]
    fn the_error_text_and_the_expected_schema_are_both_in_the_alarm() {
        let temp = TempDir::new().unwrap();
        let root = contract_root(&temp);
        std::fs::write(root.join("ops/queue-keeper-v3.json"), LIVE_KEEPER_ERROR_DOC).unwrap();
        let alarms = scan(&root);
        let row = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "status_error_doc:ops/queue-keeper-v3.json")
            .unwrap();
        assert!(row["observed"]
            .as_str()
            .unwrap()
            .contains("native attempt submission transition count mismatch"));
        assert!(row["threshold"]
            .as_str()
            .unwrap()
            .contains("JopeSelfplayQueueKeeper.v3"));
        assert_eq!(row["severity"], "critical");
    }

    #[test]
    fn a_persistent_failure_escalates_to_status_failing() {
        let temp = TempDir::new().unwrap();
        let root = contract_root(&temp);
        std::fs::write(root.join("ops/queue-keeper-v3.json"), LIVE_KEEPER_ERROR_DOC).unwrap();
        for cycle in 1..=FAILING_CYCLES {
            let names = alarm_names(&scan(&root));
            let failing = names.contains(&"status_failing:ops/queue-keeper-v3.json".to_owned());
            assert_eq!(failing, cycle >= FAILING_CYCLES, "cycle {cycle}: {names:?}");
        }
        // Recovery clears the streak.
        std::fs::write(
            root.join("ops/queue-keeper-v3.json"),
            r#"{"schema":"JopeSelfplayQueueKeeper.v3"}"#,
        )
        .unwrap();
        let names = alarm_names(&scan(&root));
        assert!(
            !names.iter().any(|name| name.starts_with("status_failing")),
            "{names:?}"
        );
    }

    #[test]
    fn a_missing_required_field_names_the_file_and_the_field() {
        let temp = TempDir::new().unwrap();
        let root = contract_root(&temp);
        // Correct schema, correct-looking document — but the field the supply
        // supervisor's own struct demands is gone. This is the class of defect,
        // detected without knowing the instance.
        std::fs::write(
            root.join("ops/queue-keeper-v3.json"),
            r#"{"schema":"JopeSelfplayQueueKeeper.v3","generated_at_ms":1785251597154,
                "admission_allowed":true,"queued_hard_cap":4096,"queued_starting":3,
                "dispatching":1,"admission_waves":[],
                "unclaimed_current_checkpoint_manifests":7}"#,
        )
        .unwrap();
        let alarms = scan(&root);
        let row = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "contract_violation:ops/queue-keeper-v3.json#queued_target")
            .unwrap_or_else(|| panic!("{:?}", alarm_names(&alarms)));
        assert_eq!(row["observed"], "missing");
        assert!(row["threshold"]
            .as_str()
            .unwrap()
            .contains("src/bin/jope_selfplay_supply.rs"));
    }

    #[test]
    fn a_wrong_typed_required_field_is_a_violation_too() {
        let temp = TempDir::new().unwrap();
        let root = contract_root(&temp);
        std::fs::write(
            root.join("ops/queue-keeper-v3.json"),
            r#"{"schema":"JopeSelfplayQueueKeeper.v3","generated_at_ms":1785251597154,
                "admission_allowed":true,"queued_target":"2048","queued_hard_cap":4096,
                "queued_starting":3,"dispatching":1,"admission_waves":[],
                "unclaimed_current_checkpoint_manifests":7}"#,
        )
        .unwrap();
        let alarms = scan(&root);
        let row = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "contract_violation:ops/queue-keeper-v3.json#queued_target")
            .unwrap_or_else(|| panic!("{:?}", alarm_names(&alarms)));
        assert_eq!(row["observed"], "expected number, found string");
    }

    #[test]
    fn a_complete_in_schema_document_is_silent() {
        let temp = TempDir::new().unwrap();
        let root = contract_root(&temp);
        std::fs::write(
            root.join("ops/queue-keeper-v3.json"),
            format!(
                r#"{{"schema":"JopeSelfplayQueueKeeper.v3","generated_at_ms":{},
                   "admission_allowed":true,"active_checkpoint_sha256":null,
                   "queued_target":2048,"queued_hard_cap":4096,"queued_starting":3,
                   "dispatching":1,"admission_waves":[],
                   "unclaimed_current_checkpoint_manifests":7}}"#,
                now_ms() as u64
            ),
        )
        .unwrap();
        assert!(alarm_names(&scan(&root)).is_empty());
    }

    #[test]
    fn a_fresh_file_whose_own_timestamp_is_frozen_is_not_healthy() {
        let temp = TempDir::new().unwrap();
        let root = contract_root(&temp);
        // Written right now (mtime is seconds old, so check_status_ages is
        // happy) but the daemon stopped advancing generated_at_ms 4 h ago.
        let stale = now_ms() as u64 - 4 * 3_600 * 1_000;
        std::fs::write(
            root.join("ops/queue-keeper-v3.json"),
            format!(
                r#"{{"schema":"JopeSelfplayQueueKeeper.v3","generated_at_ms":{stale},
                   "admission_allowed":true,"queued_target":2048,"queued_hard_cap":4096,
                   "queued_starting":3,"dispatching":1,"admission_waves":[],
                   "unclaimed_current_checkpoint_manifests":7}}"#
            ),
        )
        .unwrap();
        let names = alarm_names(&scan(&root));
        assert!(
            names.contains(&"status_content_frozen:ops/queue-keeper-v3.json".to_owned()),
            "{names:?}"
        );
    }

    #[test]
    fn an_in_schema_degraded_document_still_alarms() {
        let temp = TempDir::new().unwrap();
        let root = contract_root(&temp);
        std::fs::write(
            root.join("ops/task-learning-controller.v2.json"),
            format!(
                r#"{{"schema":"TaskLearningControllerStatus.v2","generated_at_ms":{},
                   "degraded":true,"error":"scq unreachable","action":"cycle_failed"}}"#,
                now_ms() as u64
            ),
        )
        .unwrap();
        let alarms = scan(&root);
        let row = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "status_degraded:ops/task-learning-controller.v2.json")
            .unwrap_or_else(|| panic!("{:?}", alarm_names(&alarms)));
        assert_eq!(row["observed"], "scq unreachable");
    }

    // -----------------------------------------------------------------------
    // Anti-drift: the required-field set is read out of the consumer's own
    // source, so these tests fail LOCALLY (at `cargo test`) the moment the
    // extractor stops understanding a struct it is enforcing.
    // -----------------------------------------------------------------------

    fn live_source(relative: &str) -> String {
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)).unwrap()
    }

    #[test]
    fn the_supply_queue_status_contract_is_derived_from_supplys_own_struct() {
        let source = live_source("src/bin/jope_selfplay_supply.rs");
        let (line, required) = derive_required_fields(&source, "QueueStatus").unwrap();
        assert!(line > 0);
        let by_name: Vec<&str> = required.iter().map(|f| f.name.as_str()).collect();
        // The exact field whose absence killed supply 5,893 times.
        assert!(by_name.contains(&"queued_target"), "{by_name:?}");
        assert_eq!(
            required
                .iter()
                .find(|f| f.name == "queued_target")
                .unwrap()
                .kind,
            Kind::Num
        );
        assert!(by_name.contains(&"admission_waves"));
        assert_eq!(
            required
                .iter()
                .find(|f| f.name == "admission_waves")
                .unwrap()
                .kind,
            Kind::Arr
        );
        assert!(by_name.contains(&"admission_allowed"));
        assert_eq!(
            required
                .iter()
                .find(|f| f.name == "admission_allowed")
                .unwrap()
                .kind,
            Kind::Bool
        );
        // Option fields are NOT required: serde accepts them absent.
        assert!(
            !by_name.contains(&"active_checkpoint_sha256"),
            "{by_name:?}"
        );
        assert!(!by_name.contains(&"checkpoint_epoch"), "{by_name:?}");
    }

    #[test]
    fn a_container_level_serde_default_makes_every_field_optional() {
        // Both DurableState structs carry #[serde(default)], which is the only
        // reason jope_selfplay_freshness.rs:328 and
        // jope_selfplay_trainer_watchdog.rs:691 survive a wrong-shaped
        // document. The contract must agree with serde, not with intent.
        for relative in [
            "src/bin/jope_selfplay_freshness.rs",
            "src/bin/jope_selfplay_trainer_watchdog.rs",
        ] {
            let source = live_source(relative);
            let (_, required) = derive_required_fields(&source, "DurableState").unwrap();
            assert!(required.is_empty(), "{relative}: {required:?}");
        }
    }

    #[test]
    fn every_contract_names_a_consumer_source_that_exists() {
        for contract in OPS_CONTRACTS {
            let Some((relative, struct_name)) = contract.consumer else {
                continue;
            };
            let source = live_source(relative);
            assert!(
                derive_required_fields(&source, struct_name).is_some(),
                "{relative} no longer defines struct {struct_name}"
            );
        }
    }

    #[test]
    fn the_extractor_handles_renames_skips_and_defaults() {
        let source = r#"
#[derive(Deserialize)]
struct Sample {
    schema: String,
    #[serde(rename = "queued_target")]
    target: i64,
    #[serde(default)]
    optional_count: usize,
    maybe: Option<String>,
    #[serde(skip)]
    internal: usize,
    waves: Vec<String>,
    index: BTreeMap<String, i64>,
    nested: SomeOtherStruct,
}
"#;
        let (_, required) = derive_required_fields(source, "Sample").unwrap();
        let names: Vec<&str> = required.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["schema", "queued_target", "waves", "index", "nested"]
        );
        assert_eq!(required[4].kind, Kind::Any);
        assert_eq!(required[3].kind, Kind::Obj);
    }

    /// Builds <temp>/attempts/0/outputs/training/wm/{train_progress.json,
    /// train_metrics.jsonl} the way a real trainer run lays them out.
    fn run_dir_with_metrics(temp: &TempDir, name: &str, rows: &[&str]) -> PathBuf {
        let run_dir = temp.path().join(name);
        let outputs = run_dir.join("attempts/0/outputs/training/watermark-000000001");
        std::fs::create_dir_all(&outputs).unwrap();
        std::fs::write(outputs.join("train_progress.json"), "{\"phase\":\"B\"}").unwrap();
        std::fs::write(outputs.join("train_metrics.jsonl"), rows.join("\n") + "\n").unwrap();
        run_dir
    }

    fn alarm_names(alarms: &Alarms) -> Vec<String> {
        alarms
            .rows
            .iter()
            .map(|row| row["alarm"].as_str().unwrap().to_owned())
            .collect()
    }

    /// `check_training_history` also consults the live scq snapshot, so a real
    /// completed trainer on this cluster can add `training_history_run_absent:*`
    /// rows that have nothing to do with the temp root under test. Tests that
    /// assert on an exact alarm set look only at the run they created.
    fn alarms_for(alarms: &Alarms, run_id: &str) -> Vec<Value> {
        alarms
            .rows
            .iter()
            .filter(|row| {
                row["alarm"]
                    .as_str()
                    .map(|name| name.ends_with(&format!(":{run_id}")))
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    #[test]
    fn read_tail_drops_the_partial_leading_line() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("lines.jsonl");
        std::fs::write(&path, "aaaa\nbbbb\ncccc\n").unwrap();
        assert_eq!(read_tail(&path, 1024).unwrap(), "aaaa\nbbbb\ncccc\n");
        // Cap lands mid-line: the truncated head must be discarded, not parsed.
        assert_eq!(read_tail(&path, 8).unwrap(), "cccc\n");
    }

    #[test]
    fn final_conditional_advantage_uses_the_last_row_that_has_the_key() {
        let temp = TempDir::new().unwrap();
        let run_dir = run_dir_with_metrics(
            &temp,
            "run-last-key",
            &[
                "{\"epoch\":1,\"val_conditional_advantage\":0.05}",
                "{\"epoch\":2,\"val_conditional_advantage\":-0.06}",
                // Later rows without the key (non-validation epochs) are skipped.
                "{\"epoch\":3,\"train_loss\":0.1}",
                "not json at all",
            ],
        );
        let (advantage, age) = final_conditional_advantage(&run_dir).unwrap();
        assert!((advantage - -0.06).abs() < 1e-12);
        assert!(age < 60);
    }

    #[test]
    fn missing_metrics_cannot_be_judged() {
        let temp = TempDir::new().unwrap();
        let run_dir = temp.path().join("no-such-run");
        assert!(final_conditional_advantage(&run_dir).is_none());
        let mut alarms = Alarms { rows: Vec::new() };
        check_completed_conditionality("jope-selfplay-train-x", "completed", &run_dir, &mut alarms);
        assert!(alarms.rows.is_empty());
    }

    #[test]
    fn warm_start_guard_deficiency_becomes_a_critical_alarm() {
        let temp = TempDir::new().unwrap();
        let ops = temp.path().join("ops");
        std::fs::create_dir_all(&ops).unwrap();
        let guard = ops.join("trainer-warm-start-guard.v1.json");
        let mut alarms = Alarms { rows: Vec::new() };

        // Absent guard file (pre-deploy) is silent.
        check_warm_start_guard(temp.path(), &mut alarms);
        assert!(alarms.rows.is_empty());

        // Healthy verdict is silent.
        std::fs::write(
            &guard,
            json!({"schema": "JopeTrainerWarmStartGuard.v1", "deficient": false}).to_string(),
        )
        .unwrap();
        check_warm_start_guard(temp.path(), &mut alarms);
        assert!(alarms.rows.is_empty());

        std::fs::write(
            &guard,
            json!({
                "schema": "JopeTrainerWarmStartGuard.v1",
                "deficient": true,
                "observed": "1 candidate(s), none full",
                "threshold": ">=152 tensors",
            })
            .to_string(),
        )
        .unwrap();
        check_warm_start_guard(temp.path(), &mut alarms);
        assert_eq!(alarms.rows.len(), 1);
        assert_eq!(alarms.rows[0]["alarm"], "trainer_warm_start_deficient");
        assert_eq!(alarms.rows[0]["severity"], "critical");
        assert_eq!(alarms.rows[0]["observed"], "1 candidate(s), none full");
    }

    #[test]
    fn completed_non_conditional_run_raises_a_critical_alarm() {
        let temp = TempDir::new().unwrap();
        let run_dir = run_dir_with_metrics(
            &temp,
            "run-negative",
            &["{\"epoch\":9,\"val_conditional_advantage\":-0.06}"],
        );
        let mut alarms = Alarms { rows: Vec::new() };
        check_completed_conditionality(
            "jope-selfplay-train-a-g3",
            "completed",
            &run_dir,
            &mut alarms,
        );
        assert_eq!(
            alarm_names(&alarms),
            vec!["policy_not_conditional:jope-selfplay-train-a-g3".to_owned()]
        );
        let row = &alarms.rows[0];
        assert_eq!(row["severity"], "critical");
        assert!(row["observed"]
            .as_str()
            .unwrap()
            .contains("policy is not conditional"));
        assert!(row["observed"].as_str().unwrap().contains("-0.0600"));

        // Exactly zero advantage is still not conditional.
        let zero =
            run_dir_with_metrics(&temp, "run-zero", &["{\"val_conditional_advantage\":0.0}"]);
        let mut alarms = Alarms { rows: Vec::new() };
        check_completed_conditionality("jte-zero", "completed", &zero, &mut alarms);
        assert_eq!(alarms.rows.len(), 1);
    }

    #[test]
    fn conditional_runs_and_live_runs_stay_quiet() {
        let temp = TempDir::new().unwrap();
        let healthy = run_dir_with_metrics(
            &temp,
            "run-healthy",
            &["{\"val_conditional_advantage\":0.031}"],
        );
        let mut alarms = Alarms { rows: Vec::new() };
        check_completed_conditionality("jte-healthy", "completed", &healthy, &mut alarms);
        assert!(alarms.rows.is_empty());
        // A run that is still training is judged by the stall check, not this one.
        let running =
            run_dir_with_metrics(&temp, "run-live", &["{\"val_conditional_advantage\":-0.2}"]);
        check_completed_conditionality("jte-running", "running", &running, &mut alarms);
        assert!(alarms.rows.is_empty());
    }

    /// Lay out `<runs>/<run_id>/attempts/0/outputs/training/watermark-.../train_metrics.jsonl`.
    fn trainer_metrics(runs: &Path, run_id: &str, body: &str) -> PathBuf {
        let outputs = runs
            .join(run_id)
            .join("attempts/0/outputs/training/watermark-001805437");
        std::fs::create_dir_all(&outputs).unwrap();
        let path = outputs.join(th::METRICS_FILE);
        std::fs::write(&path, body).unwrap();
        path
    }

    fn archive_now(root: &Path, runs: &Path, run_id: &str) {
        let archive = th::archive_root(root);
        for found in th::scan_run_dir(runs, run_id) {
            th::ingest_source(
                &archive,
                &found.key,
                &th::FileSource(found.path.clone()),
                &found.path.to_string_lossy(),
                "xbabe2",
            )
            .unwrap();
        }
        th::rebuild_index(root, &archive).unwrap();
    }

    #[test]
    fn a_fully_archived_run_is_silent() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let run_id = "jope-selfplay-train-001805437-a-g9";
        trainer_metrics(&runs, run_id, "{\"epoch\":1,\"val_loss\":3.4}\n");
        archive_now(&root, &runs, run_id);

        let mut alarms = Alarms { rows: Vec::new() };
        check_training_history(&root, &runs, &mut alarms);
        assert!(alarms.rows.is_empty(), "{:?}", alarm_names(&alarms));
    }

    #[test]
    fn a_missing_archive_is_critical() {
        let temp = TempDir::new().unwrap();
        let mut alarms = Alarms { rows: Vec::new() };
        check_training_history(temp.path(), &temp.path().join("runs"), &mut alarms);
        assert_eq!(alarm_names(&alarms), vec!["training_history_missing"]);
        assert_eq!(alarms.rows[0]["severity"], "critical");
    }

    #[test]
    fn an_idle_run_with_unarchived_rows_is_critical() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let run_id = "jope-selfplay-train-001805437-c";
        let path = trainer_metrics(&runs, run_id, "{\"epoch\":1,\"val_loss\":9.6}\n");
        archive_now(&root, &runs, run_id);
        // Rows appear that capture never picked up, and the trainer has since
        // gone quiet: this run is one GC pass from losing them.
        std::fs::write(
            &path,
            "{\"epoch\":1,\"val_loss\":9.6}\n{\"epoch\":2,\"val_loss\":9.7}\n",
        )
        .unwrap();
        let old =
            SystemTime::now() - std::time::Duration::from_secs(HISTORY_SOURCE_FRESH_SECS + 60);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(old))
            .unwrap();

        let mut alarms = Alarms { rows: Vec::new() };
        check_training_history(&root, &runs, &mut alarms);
        let mine = alarms_for(&alarms, run_id);
        assert_eq!(mine.len(), 1, "{mine:?}");
        assert_eq!(
            mine[0]["alarm"],
            format!("training_history_unarchived:{run_id}")
        );
        assert_eq!(mine[0]["severity"], "critical");
        // The gap spans a newline, so it is complete rows, not a truncated tail.
        assert!(mine[0]["observed"]
            .as_str()
            .unwrap()
            .contains("complete_rows_missing"));
    }

    /// The truncated-tail policy, from the alarm's side: a trailing incomplete
    /// row that has been recorded must NOT read as data loss, because a
    /// permanent critical here blocks the expansion driver and buries the real
    /// `training_history_unarchived` alarms under noise.
    #[test]
    fn a_recorded_truncated_tail_is_informational_not_critical() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let run_id = "jope-selfplay-train-001805437-b-g6";
        let path = trainer_metrics(
            &runs,
            run_id,
            "{\"epoch\":1,\"val_loss\":9.6}\n{\"epoch\":2,\"val_loss\":9.5}\n{\"epoch\":3,\"val_",
        );
        archive_now(&root, &runs, run_id);
        let old = SystemTime::now() - std::time::Duration::from_secs(th::TAIL_QUIESCENT_SECS + 60);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(old))
            .unwrap();

        // Before the hole is recorded: visible, but a warning, not critical.
        let mut alarms = Alarms { rows: Vec::new() };
        check_training_history(&root, &runs, &mut alarms);
        let mine = alarms_for(&alarms, run_id);
        assert_eq!(mine.len(), 1, "{mine:?}");
        assert_eq!(
            mine[0]["alarm"],
            format!("training_history_truncated_tail_pending:{run_id}")
        );
        assert_eq!(mine[0]["severity"], "warning");

        // Once capture records it, it drops to informational and stays there.
        let archive = th::archive_root(&root);
        let found = th::scan_run_dir(&runs, run_id).remove(0);
        let held = th::archived_bytes(&archive, &found.key);
        let th::Residual::PartialRow(fragment) =
            th::classify_residual(&th::FileSource(found.path.clone()), held, found.len).unwrap()
        else {
            panic!("expected a partial row");
        };
        th::record_truncation(
            &archive,
            &found.key,
            "xbabe2",
            &found.path.to_string_lossy(),
            &fragment,
            found.len,
            0,
        )
        .unwrap();

        let mut alarms = Alarms { rows: Vec::new() };
        check_training_history(&root, &runs, &mut alarms);
        let mine = alarms_for(&alarms, run_id);
        assert_eq!(mine.len(), 1, "{mine:?}");
        assert_eq!(
            mine[0]["alarm"],
            format!("training_history_truncated_tail:{run_id}")
        );
        assert_eq!(mine[0]["severity"], "info");
        assert!(
            !alarm_names(&alarms).contains(&format!("training_history_unarchived:{run_id}")),
            "a recorded truncated tail must not also fire the critical"
        );
    }

    #[test]
    fn an_unreachable_capture_node_is_a_warning() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let run_id = "jope-selfplay-train-001805437-a-g8";
        trainer_metrics(&runs, run_id, "{\"epoch\":1,\"val_loss\":3.4}\n");
        archive_now(&root, &runs, run_id);
        let archive = th::archive_root(&root);
        let mut nodes = std::collections::BTreeMap::new();
        nodes.insert(
            "xbabe3".to_string(),
            th::NodeStatus {
                node: "xbabe3".into(),
                last_attempt_ms: th::now_ms(),
                last_success_ms: 0,
                consecutive_failures: 4,
                last_error: "ssh exit Some(255)".into(),
                segments_seen: 0,
                reachable: false,
            },
        );
        th::save_nodes(&archive, &nodes).unwrap();

        let mut alarms = Alarms { rows: Vec::new() };
        check_training_history(&root, &runs, &mut alarms);
        let row = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "training_history_node_unreachable:xbabe3")
            .expect("node alarm");
        assert_eq!(row["severity"], "warning");
        assert!(row["observed"].as_str().unwrap().contains("never swept"));
    }

    #[test]
    fn a_row_being_written_right_now_is_not_an_alarm() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let run_id = "jope-selfplay-train-001805437-a-g9";
        let path = trainer_metrics(&runs, run_id, "{\"epoch\":1,\"val_loss\":3.4}\n");
        archive_now(&root, &runs, run_id);
        // Partial row in flight: source is ahead, but the archive just advanced,
        // so this is normal steady state and must stay quiet.
        std::fs::write(
            &path,
            "{\"epoch\":1,\"val_loss\":3.4}\n{\"epoch\":2,\"val_l",
        )
        .unwrap();

        let mut alarms = Alarms { rows: Vec::new() };
        check_training_history(&root, &runs, &mut alarms);
        assert!(alarms.rows.is_empty(), "{:?}", alarm_names(&alarms));
    }

    // -----------------------------------------------------------------------
    // Proposal deficiency / head collapse
    //
    // Every fixture below is a real reading off the live root on 2026-07-28,
    // not an invented shape.
    // -----------------------------------------------------------------------

    /// Publishes a proposal set and returns the root. `summary` is written both
    /// where the pointer names it and beside the (empty) archived jsonl, so the
    /// fallback path can be exercised by deleting one of them.
    fn proposal_root(temp: &TempDir, summary: Value) -> PathBuf {
        let root = temp.path().to_path_buf();
        std::fs::create_dir_all(root.join("proposals/by-checkpoint")).unwrap();
        let archive = root.join("proposals/by-checkpoint/set.jsonl");
        std::fs::write(&archive, "").unwrap();
        let named = root.join("proposals/named.summary.json");
        std::fs::write(&named, serde_json::to_vec(&summary).unwrap()).unwrap();
        std::fs::write(
            root.join("proposals/by-checkpoint/set.summary.json"),
            serde_json::to_vec(&summary).unwrap(),
        )
        .unwrap();
        std::fs::write(
            root.join("proposals/current.json"),
            serde_json::to_vec(&json!({
                "checkpoint_sha256": "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592",
                "proposals": archive.to_string_lossy(),
                "summary": named.to_string_lossy(),
            }))
            .unwrap(),
        )
        .unwrap();
        root
    }

    fn proposal_scan(root: &Path) -> Alarms {
        let mut alarms = Alarms { rows: Vec::new() };
        check_proposal_deficiency(root, &mut alarms);
        alarms
    }

    /// Micro-cohort set 3a89b11ac7da, 2026-07-28T23:11Z, checkpoint fbcd5459:
    /// 128 contexts, binary 19/64 short with a worst context of 20/64 and 583
    /// borrowed rows; regression clean. Head mix measured off the emitted file.
    fn live_micro_cohort_summary() -> Value {
        json!({
            "schema": "JopeTestOnlyDlProposalSet.v1",
            "model_checkpoint_sha256": "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592",
            "proposals_per_dataset": 64,
            "deficiency_by_task": {
                "binary": {
                    "contexts": 64, "deficient_contexts": 19,
                    "deficient_fraction": 19.0 / 64.0, "min_native_unique": 20,
                    "mean_native_unique": 3513.0 / 64.0, "backfilled_rows": 583,
                    "backfilled_fraction": 583.0 / 4096.0, "unfilled_contexts": 0
                },
                "regression": {
                    "contexts": 64, "deficient_contexts": 0,
                    "deficient_fraction": 0.0, "min_native_unique": 64,
                    "mean_native_unique": 64.0, "backfilled_rows": 0,
                    "backfilled_fraction": 0.0, "unfilled_contexts": 0
                }
            },
            "native_head_counts_by_task": {
                "binary": {
                    "logit_ridge": 3084, "tab_boost": 275, "extra_trees": 100,
                    "gam_boost": 53, "logistic": 1
                },
                "regression": {
                    "ridge": 913, "gpr": 900, "histgbm2": 641, "elasticnet": 626,
                    "ensemble_ridge_gbm": 325, "ensemble": 218, "histgbm": 200,
                    "gbm": 160, "ksmooth": 113
                }
            },
            "proposal_validity_by_task": {"binary": 0.9904186320754716, "regression": 0.9997577519379846}
        })
    }

    #[test]
    fn the_live_micro_cohort_set_raises_deficiency_on_binary_only() {
        let temp = TempDir::new().unwrap();
        let root = proposal_root(&temp, live_micro_cohort_summary());
        let alarms = proposal_scan(&root);
        let names = alarm_names(&alarms);
        // 19/64 = 0.297 is past the warn floor; worst context 20/64 = 0.3125 is
        // above the critical ratio, so this is a warning, not a critical.
        let row = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "proposal_deficiency:binary")
            .unwrap_or_else(|| panic!("{names:?}"));
        assert_eq!(row["severity"], "warning");
        let observed = row["observed"].as_str().unwrap();
        assert!(observed.contains("19/64 contexts short"), "{observed}");
        assert!(observed.contains("worst 20/64"), "{observed}");
        assert!(observed.contains("583 rows borrowed"), "{observed}");
        // Regression filled every context itself and its modal head is 0.223.
        assert!(
            !names.iter().any(|name| name.ends_with(":regression")),
            "{names:?}"
        );
        // logit_ridge at 0.878 is the binary branch's NORMAL skew (an independent
        // 4-context offline run of the same checkpoint measured 0.883), so it must
        // NOT light the head alarm — otherwise the alarm is on permanently and
        // says nothing when the branch actually collapses.
        assert!(
            !names.contains(&"proposal_head_collapse:binary".to_owned()),
            "{names:?}"
        );
    }

    #[test]
    fn a_binary_branch_drifting_past_its_measured_band_warns_before_it_collapses() {
        let temp = TempDir::new().unwrap();
        let mut summary = live_micro_cohort_summary();
        // Same 3,513 model-sampled binary proposals, redistributed so the modal
        // head holds 0.93: past the 0.90 measured-healthy band, short of 0.95.
        summary["native_head_counts_by_task"]["binary"] =
            json!({"logit_ridge": 3267, "tab_boost": 200, "extra_trees": 46});
        let root = proposal_root(&temp, summary);
        let alarms = proposal_scan(&root);
        let row = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "proposal_head_collapse:binary")
            .unwrap_or_else(|| panic!("{:?}", alarm_names(&alarms)));
        assert_eq!(row["severity"], "warning");
        assert!(row["observed"]
            .as_str()
            .unwrap()
            .contains("logit_ridge holds 0.93"));
    }

    /// Whole-catalog set catalog-62054d7b05cb, same checkpoint, 2026-07-28
    /// 20:47Z: 132,669 binary proposals, every one of them `logistic`, and a
    /// binary decode validity of 0.0407.
    #[test]
    fn the_live_whole_catalog_set_is_a_critical_binary_head_collapse() {
        let temp = TempDir::new().unwrap();
        let root = proposal_root(
            &temp,
            json!({
                "schema": "JopeTestOnlyDlProposalSet.v1",
                "model_checkpoint_sha256": "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592",
                "proposals_per_dataset": 64,
                "native_head_counts_by_task": {
                    "binary": {"logistic": 132_669},
                    "regression": {
                        "gpr": 184_692, "ridge": 173_227, "histgbm2": 131_428,
                        "elasticnet": 119_461, "ensemble_ridge_gbm": 62_100,
                        "ensemble": 46_134, "histgbm": 36_543, "gbm": 29_992,
                        "ksmooth": 20_007
                    }
                },
                "proposal_validity_by_task": {
                    "binary": 0.040735005833279964,
                    "regression": 0.9996332708535528
                }
            }),
        );
        let alarms = proposal_scan(&root);
        let names = alarm_names(&alarms);
        let head = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "proposal_head_collapse:binary")
            .unwrap_or_else(|| panic!("{names:?}"));
        assert_eq!(head["severity"], "critical");
        assert!(head["observed"]
            .as_str()
            .unwrap()
            .contains("logistic holds 1.0000"));
        let validity = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "proposal_task_validity:binary")
            .unwrap_or_else(|| panic!("{names:?}"));
        assert_eq!(validity["severity"], "critical");
        assert!(validity["observed"].as_str().unwrap().starts_with("0.0407"));
        assert!(
            !names.iter().any(|name| name.ends_with(":regression")),
            "{names:?}"
        );
    }

    #[test]
    fn a_context_the_pool_could_not_fill_is_always_critical() {
        let temp = TempDir::new().unwrap();
        let root = proposal_root(
            &temp,
            json!({
                "schema": "JopeTestOnlyDlProposalSet.v1",
                "proposals_per_dataset": 64,
                "deficiency_by_task": {"binary": {
                    "contexts": 64, "deficient_contexts": 1,
                    "deficient_fraction": 1.0 / 64.0, "min_native_unique": 60,
                    "backfilled_rows": 4, "unfilled_contexts": 1
                }}
            }),
        );
        let alarms = proposal_scan(&root);
        let names = alarm_names(&alarms);
        // One thin context out of 64 is under every deficiency threshold, so the
        // only thing that can be loud here is the below-budget file itself.
        assert!(
            !names.contains(&"proposal_deficiency:binary".to_owned()),
            "{names:?}"
        );
        let row = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "proposal_unfilled:binary")
            .unwrap_or_else(|| panic!("{names:?}"));
        assert_eq!(row["severity"], "critical");
    }

    #[test]
    fn a_collapsed_worst_context_is_critical_even_when_few_contexts_are_short() {
        let temp = TempDir::new().unwrap();
        let root = proposal_root(
            &temp,
            json!({
                "schema": "JopeTestOnlyDlProposalSet.v1",
                "proposals_per_dataset": 64,
                "deficiency_by_task": {"binary": {
                    "contexts": 640, "deficient_contexts": 12,
                    "deficient_fraction": 12.0 / 640.0,
                    // the original severing failure: 34 of 64. 16 is a quarter.
                    "min_native_unique": 16, "backfilled_rows": 300,
                    "unfilled_contexts": 0
                }}
            }),
        );
        let alarms = proposal_scan(&root);
        let row = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "proposal_deficiency:binary")
            .expect("deficiency alarm");
        assert_eq!(row["severity"], "critical");
    }

    #[test]
    fn a_summary_without_the_new_fields_is_silent_rather_than_falsely_healthy() {
        let temp = TempDir::new().unwrap();
        // The pre-2026-07-28 summary shape: no deficiency_by_task, no head
        // histogram. Measuring nothing must not report health.
        let root = proposal_root(
            &temp,
            json!({
                "schema": "JopeTestOnlyDlProposalSet.v1",
                "proposals_per_dataset": 64,
                "counts": {"binary": 132_669, "regression": 803_584},
                "deficient_contexts": {"binary:27390545fb3675f1": 61},
                "proposal_validity_by_task": {"binary": 0.9, "regression": 0.99}
            }),
        );
        assert!(proposal_scan(&root).rows.is_empty());
    }

    #[test]
    fn the_summary_beside_the_archive_is_used_when_the_named_one_was_reaped() {
        let temp = TempDir::new().unwrap();
        let root = proposal_root(&temp, live_micro_cohort_summary());
        // GC reaps training/tournament-work/, which is where the controller
        // lane's `summary` pointer aims.
        std::fs::remove_file(root.join("proposals/named.summary.json")).unwrap();
        let names = alarm_names(&proposal_scan(&root));
        assert!(
            names.contains(&"proposal_deficiency:binary".to_owned()),
            "{names:?}"
        );
    }

    #[test]
    fn no_published_proposal_set_is_silent() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("proposals")).unwrap();
        assert!(proposal_scan(temp.path()).rows.is_empty());
    }

    // -----------------------------------------------------------------------
    // Corpus freeze guardrails
    // -----------------------------------------------------------------------

    fn freeze_root(temp: &TempDir) -> PathBuf {
        let root = temp.path().to_path_buf();
        std::fs::create_dir_all(root.join("ops")).unwrap();
        std::fs::create_dir_all(root.join("training")).unwrap();
        root
    }

    fn freeze_scan(root: &Path) -> Alarms {
        let mut alarms = Alarms { rows: Vec::new() };
        check_corpus_freeze_guard(root, &mut alarms);
        alarms
    }

    fn write_pointers(root: &Path, frozen: u64, labels: u64) {
        std::fs::write(
            root.join("training/current-corpus.json"),
            serde_json::to_vec(&json!({"watermark": frozen})).unwrap(),
        )
        .unwrap();
        std::fs::write(
            root.join("ops/last-ingest.json"),
            serde_json::to_vec(&json!({"exact_rows_total": labels})).unwrap(),
        )
        .unwrap();
    }

    /// THE two-day outage, in numbers taken off the live root at 23:33Z: the
    /// corpus pointer pinned at 1,805,437 by a tournament parked at
    /// `phase_index 9`, against 4,691,549 ingested labels. Nothing in the alarm
    /// set said a word about it before this check existed.
    #[test]
    fn a_severed_freeze_loop_is_critical() {
        let temp = TempDir::new().unwrap();
        let root = freeze_root(&temp);
        write_pointers(&root, 1_805_437, 4_691_549);
        let alarms = freeze_scan(&root);
        let row = alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == "corpus_freeze_starved")
            .expect("expected corpus_freeze_starved");
        assert_eq!(row["severity"], "critical");
        assert!(
            row["observed"].as_str().unwrap().contains("2886112 labels"),
            "{row}"
        );
    }

    /// One freeze behind is not an outage. The step is 50,000 labels, so a gap
    /// of that order must stay silent or the alarm is noise.
    #[test]
    fn one_step_of_unfrozen_labels_is_silent() {
        let temp = TempDir::new().unwrap();
        let root = freeze_root(&temp);
        write_pointers(&root, 4_600_000, 4_650_000);
        assert!(freeze_scan(&root).rows.is_empty());
    }

    #[test]
    fn a_few_missed_freezes_is_a_warning_not_a_critical() {
        let temp = TempDir::new().unwrap();
        let root = freeze_root(&temp);
        write_pointers(&root, 4_400_000, 4_650_000);
        let alarms = freeze_scan(&root);
        assert_eq!(
            alarm_named(&alarms, "corpus_freeze_starved")["severity"],
            "warning"
        );
    }

    /// Guardrail 1 of ops/generalization-plan.md. Evidence only accumulates, so
    /// a fall in dataset count is always a defect.
    #[test]
    fn a_fallen_dataset_count_is_critical() {
        let temp = TempDir::new().unwrap();
        let root = freeze_root(&temp);
        write_pointers(&root, 4_650_000, 4_650_000);
        std::fs::write(
            root.join("ops/corpus-freeze-guard.v1.json"),
            serde_json::to_vec(&json!({
                "schema": "JopeCorpusFreezeGuard.v1",
                "regressed": true,
                "observed": "1204 datasets at watermark 4690000 (full_consolidation/eq32v32c128)",
                "threshold": ">=2857 datasets (high-water mark set at watermark 1805437)",
            }))
            .unwrap(),
        )
        .unwrap();
        let alarms = freeze_scan(&root);
        let row = alarm_named(&alarms, "corpus_dataset_count_fell");
        assert_eq!(row["severity"], "critical");
        assert!(row["threshold"].as_str().unwrap().contains("2857"));
    }

    #[test]
    fn a_healthy_freeze_guard_is_silent() {
        let temp = TempDir::new().unwrap();
        let root = freeze_root(&temp);
        write_pointers(&root, 4_650_000, 4_650_000);
        std::fs::write(
            root.join("ops/corpus-freeze-guard.v1.json"),
            serde_json::to_vec(&json!({
                "schema": "JopeCorpusFreezeGuard.v1",
                "regressed": false,
                "observed": "18000 datasets at watermark 4690000",
                "threshold": ">=2857 datasets",
                "split_median_divergence": {
                    "binary": {"train": 32.0, "validation": 32.0, "ratio": 1.0},
                    "regression": {"train": 25.0, "validation": 25.0, "ratio": 1.0},
                },
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(freeze_scan(&root).rows.is_empty());
    }

    /// Guardrail 2. The historical failure was train median 6 against validation
    /// median 485 — an 80x inversion that was read as an overfit gap for a day.
    #[test]
    fn diverged_split_medians_are_critical_per_task() {
        let temp = TempDir::new().unwrap();
        let root = freeze_root(&temp);
        write_pointers(&root, 4_650_000, 4_650_000);
        std::fs::write(
            root.join("ops/corpus-freeze-guard.v1.json"),
            serde_json::to_vec(&json!({
                "schema": "JopeCorpusFreezeGuard.v1",
                "regressed": false,
                "split_median_divergence": {
                    "binary": {"train": 32.0, "validation": 32.0, "ratio": 1.0},
                    "regression": {"train": 6.0, "validation": 485.0, "ratio": 80.83},
                },
            }))
            .unwrap(),
        )
        .unwrap();
        let names = alarm_names(&freeze_scan(&root));
        assert!(
            names.contains(&"corpus_split_median_divergence:regression".to_owned()),
            "{names:?}"
        );
        assert!(
            !names.contains(&"corpus_split_median_divergence:binary".to_owned()),
            "{names:?}"
        );
    }

    /// The freezer publishing `label_pointer_behind_corpus` is the one way it
    /// can stop the trainer watchdog, so it must arrive as a critical alarm.
    #[test]
    fn the_freezers_own_alarm_is_lifted() {
        let temp = TempDir::new().unwrap();
        let root = freeze_root(&temp);
        write_pointers(&root, 4_650_000, 4_650_000);
        std::fs::write(
            root.join("ops/corpus-freezer-v1.json"),
            serde_json::to_vec(&json!({
                "schema": "JopeSelfplayCorpusFreezer.v1",
                "lifecycle": "ok",
                "action": "froze",
                "alarm": "label_pointer_behind_corpus",
                "detail": "exact_rows_total 4690000 < frozen watermark 4691000",
            }))
            .unwrap(),
        )
        .unwrap();
        let alarms = freeze_scan(&root);
        assert_eq!(
            alarm_named(&alarms, "label_pointer_behind_corpus")["severity"],
            "critical"
        );
    }

    /// A root where the freezer was never deployed must not carry a permanent
    /// alarm for a unit it does not have.
    #[test]
    fn an_undeployed_freezer_is_silent() {
        let temp = TempDir::new().unwrap();
        let root = freeze_root(&temp);
        write_pointers(&root, 4_650_000, 4_650_000);
        let names = alarm_names(&freeze_scan(&root));
        assert!(
            !names.iter().any(|name| name.contains("corpus-freezer")),
            "{names:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Unsubmittable trainer intent
    // -----------------------------------------------------------------------

    /// The live state document at 23:24Z on 2026-07-28, byte for byte, while
    /// the trainer had been unable to submit for 31 minutes and every alarm in
    /// this file was green.
    const LIVE_BRICKED_INTENT: &str = r#"{"intent":{"checkpoint_mismatch_observations":0,"checkpoint_sha256":"43c64bde25df51e1f3771533b151c82d971cfaf026dada339e400d69950cba40","created_at_ms":1785279187223,"job_id":"jope-selfplay-train-001805437-c-r000025","last_error":"SCQ check rejected trainer: error[input-hash-mismatch]: input 'script' hash mismatch: expected e9f783d94c08274f8dbd1252efe964a5f8e369576ea55c7be43e7d04fd7a521d, got 2733e2da17dcc60a7667713ce70e552604c236bb7243ea4a530e20d09b152492","mode":"continuous_recent_plus_replay","seed":20260751,"sequence":25,"spec":"/home/ubuntu/jope-flywheel/test_exact_gen0/specs/jope-selfplay-train-001805437-c-r000025.toml","spec_sha256":"637280023e1ce5c90895896836cc6adbd2eb9ff52819cc03227e46da88234b1e","watermark":1805437},"last_evidence_watermark":1805437,"max_observed_labels":4683572,"max_observed_snapshot_watermark":1805437,"next_sequence":26,"schema":"JopeContinuousTrainerWatchdogState.v1","updated_at_ms":1785281041552}"#;

    fn intent_scan(root: &Path) -> Alarms {
        let mut alarms = Alarms { rows: Vec::new() };
        check_trainer_intent(root, &mut alarms);
        alarms
    }

    #[test]
    fn the_live_bricked_intent_is_critical_and_prints_the_recovery() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("ops")).unwrap();
        std::fs::write(
            temp.path().join("ops/trainer-watchdog-state.v1.json"),
            LIVE_BRICKED_INTENT,
        )
        .unwrap();
        let alarms = intent_scan(temp.path());
        let row = alarm_named(&alarms, "trainer_intent_unsubmittable");
        assert_eq!(row["severity"], "critical");
        let observed = row["observed"].as_str().unwrap();
        assert!(
            observed.contains("jope-selfplay-train-001805437-c-r000025"),
            "{observed}"
        );
        assert!(observed.contains("\"intent\": null"), "{observed}");
    }

    /// A capacity rejection is transient and self-heals. Alarming on it would
    /// make this check noise on a busy cluster, which is where it matters most.
    #[test]
    fn a_transient_capacity_rejection_is_not_an_alarm() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("ops")).unwrap();
        std::fs::write(
            temp.path().join("ops/trainer-watchdog-state.v1.json"),
            serde_json::to_vec(&json!({
                "intent": {
                    "job_id": "jope-selfplay-train-001805437-c-r000026",
                    "last_error": "SCQ check rejected trainer: error[dependency-unsatisfied]: dependencies cannot be verified on any eligible node",
                },
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(intent_scan(temp.path()).rows.is_empty());
    }

    #[test]
    fn a_healthy_or_absent_intent_is_silent() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("ops")).unwrap();
        assert!(intent_scan(temp.path()).rows.is_empty());
        std::fs::write(
            temp.path().join("ops/trainer-watchdog-state.v1.json"),
            serde_json::to_vec(&json!({"intent": null, "next_sequence": 27})).unwrap(),
        )
        .unwrap();
        assert!(intent_scan(temp.path()).rows.is_empty());
    }

    fn alarm_named<'a>(alarms: &'a Alarms, name: &str) -> &'a Value {
        alarms
            .rows
            .iter()
            .find(|row| row["alarm"] == name)
            .unwrap_or_else(|| panic!("missing {name}: {:?}", alarm_names(alarms)))
    }
}
