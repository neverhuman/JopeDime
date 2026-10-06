//! Native Rust port of the durable single-writer self-improvement controller.
//!
//! This binary owns the promotion loop's ORCHESTRATION: the idempotent action
//! journal (`ops/controller.sqlite` + `ops/controller-events.jsonl`), the
//! tournament state machine (`ops/flywheel-state.json`), scheduler
//! interactions, checkpoint fetch (including scheduler-pruned run-dir
//! recovery), gate-proposal spec rendering/submission, the acceptance gate,
//! and the accepted-pointer promotion with ancestor CAS.
//!
//! Three ML data-pipeline phases delegate to the pinned Python modules they
//! have always run in (`test_exact_controller._freeze_corpus`,
//! `_build_packed_corpus`, `_evaluate_frozen_gate`). Promotion ends after the
//! accepted pointer/event is published. Supply is the sole cohort and wave
//! builder; promotion cannot seed the next wave.
//!
//! JSON written by this binary matches CPython's `json.dumps` formatting
//! (sorted keys, separator styles, ensure_ascii, and float `repr` notation)
//! so journal keys and artifacts remain byte-compatible with the Python
//! bridge it replaces.

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_expansion::TASKS;
use jope_dime_regressor::spec_policy::{PROPOSE, TRAINER};
use jope_dime_regressor::task::Task;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
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
    fn scripts_str => paths::research_scripts_root()
);
jope_paths::once_str!(
    fn proposal_selection_policy_str => paths::campaign_manifest("selfplay/proposal-selection-policy.v1.json")
);
jope_paths::once_str!(
    fn prime_src_str => paths::python_src()
);
jope_paths::once_str!(
    fn evaluator_str => paths::corpus_eval_binary()
);
jope_paths::once_str!(
    fn warm_str => paths::legacy_checkpoints_root().join("fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592.model.pt")
);
const WARM_SHA: &str = "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592";
const CATALOG_STRIDE_BYTES: u64 = 2_048 * 4;
const LABEL_THRESHOLD: i64 = 50_000;
const DEFAULT_SEED: i64 = 20260726;

const PHASES: [&str; 9] = [
    "freeze_corpus",
    "build_packed_corpus",
    "launch_checkpoint_tournament",
    "fetch_candidate_checkpoints",
    "generate_gate_proposals",
    "evaluate_frozen_gate",
    "build_acceptance_evidence",
    "run_checkpoint_gate",
    "advance_accepted_pointer",
];

/// Phases delegated to the pinned Python implementations (see module docs).
const PYTHON_PHASES: [&str; 3] = [
    "freeze_corpus",
    "build_packed_corpus",
    "evaluate_frozen_gate",
];

/// The shim inserts the scripts root at run time (`@SCRIPTS@`), see [`py_shim`].
const PY_SHIM_TEMPLATE: &str = r#"
import json, sys
sys.path.insert(0, "@SCRIPTS@")
import test_exact_controller as c
name = sys.argv[1]
if name == "milestone_status":
    from test_exact_sealed_audit import milestone_status
    print(json.dumps(milestone_status(sys.argv[2]), sort_keys=True))
else:
    handler = getattr(c, "_" + name)
    with open(sys.argv[2]) as handle:
        request = json.load(handle)
    try:
        print(json.dumps(handler(request), sort_keys=True))
    except c.PendingAction as error:
        print(json.dumps({"__pending__": str(error)}, sort_keys=True))
"#;

/// [`PY_SHIM_TEMPLATE`] with the scripts root resolved once per process.
fn py_shim() -> &'static str {
    static CELL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CELL.get_or_init(|| PY_SHIM_TEMPLATE.replace("@SCRIPTS@", scripts_str()))
}

// ---------------------------------------------------------------------------
// CPython-compatible JSON serialization
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum PyStyle {
    /// separators=(",", ":")
    Compact,
    /// json.dumps defaults: separators=(", ", ": ")
    Default,
    /// indent=2, separators=(",", ": ")
    Pretty,
}

/// CPython `repr(float)` notation: shortest round-trip digits, scientific
/// form when the decimal exponent is < -4 or >= 16, exponent always signed
/// with at least two digits, integral fixed values keep a trailing `.0`.
fn py_float_repr(value: f64) -> Result<String> {
    if !value.is_finite() {
        bail!("non-finite float cannot be serialized to strict JSON");
    }
    if value == 0.0 {
        return Ok(if value.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        });
    }
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .ok_or_else(|| anyhow!("unexpected float format {scientific}"))?;
    let exponent: i32 = exponent.parse()?;
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let body = if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            let int_len = exponent as usize + 1;
            if digits.len() <= int_len {
                let padded = format!("{digits:0<width$}", width = int_len);
                format!("{padded}.0")
            } else {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            let zeros = "0".repeat((-exponent - 1) as usize);
            format!("0.{zeros}{digits}")
        }
    } else {
        let mantissa_text = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let exponent_sign = if exponent < 0 { "-" } else { "+" };
        format!("{mantissa_text}e{exponent_sign}{:02}", exponent.abs())
    };
    Ok(format!("{sign}{body}"))
}

fn py_escape_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if (c as u32) > 0x7e => {
                // ensure_ascii=True
                let mut buffer = [0u16; 2];
                for unit in c.encode_utf16(&mut buffer) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn py_write(value: &Value, style: PyStyle, depth: usize, out: &mut String) -> Result<()> {
    let (item_sep, key_sep) = match style {
        PyStyle::Compact => (",", ":"),
        PyStyle::Default => (", ", ": "),
        PyStyle::Pretty => (",", ": "),
    };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                out.push_str(&int.to_string());
            } else if let Some(int) = number.as_u64() {
                out.push_str(&int.to_string());
            } else {
                let float = number
                    .as_f64()
                    .ok_or_else(|| anyhow!("unrepresentable JSON number"))?;
                out.push_str(&py_float_repr(float)?);
            }
        }
        Value::String(text) => py_escape_string(text, out),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_sep);
                }
                if style == PyStyle::Pretty {
                    out.push('\n');
                    out.push_str(&"  ".repeat(depth + 1));
                }
                py_write(item, style, depth + 1, out)?;
            }
            if style == PyStyle::Pretty {
                out.push('\n');
                out.push_str(&"  ".repeat(depth));
            }
            out.push(']');
        }
        Value::Object(entries) => {
            if entries.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            let sorted: BTreeMap<&String, &Value> = entries.iter().collect();
            out.push('{');
            for (index, (key, item)) in sorted.into_iter().enumerate() {
                if index > 0 {
                    out.push_str(item_sep);
                }
                if style == PyStyle::Pretty {
                    out.push('\n');
                    out.push_str(&"  ".repeat(depth + 1));
                }
                py_escape_string(key, out);
                out.push_str(key_sep);
                py_write(item, style, depth + 1, out)?;
            }
            if style == PyStyle::Pretty {
                out.push('\n');
                out.push_str(&"  ".repeat(depth));
            }
            out.push('}');
        }
    }
    Ok(())
}

fn py_dumps(value: &Value, style: PyStyle) -> Result<String> {
    let mut out = String::new();
    py_write(value, style, 0, &mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn sha256_bytes(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

fn sha256_text(text: &str) -> String {
    sha256_bytes(text.as_bytes())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn is_sha256(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// `test_exact_common.atomic_json`: pretty CPython dump + newline, temp file
/// named `.{name}.tmp-{pid}` in place, then rename.
fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("atomic_json target has no parent"))?;
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("atomic_json target has no name"))?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    let text = format!("{}\n", py_dumps(value, PyStyle::Pretty)?);
    fs::write(&temporary, text)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn copy_tree(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let destination = target.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &destination)?;
        } else if kind.is_file() {
            fs::copy(entry.path(), destination)?;
        } else {
            bail!("unsupported filesystem entry in fetched outputs");
        }
    }
    Ok(())
}

fn rglob(root: &Path, name: &str, results: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    let mut children: Vec<_> = entries.flatten().collect();
    children.sort_by_key(|entry| entry.file_name());
    for entry in children {
        let path = entry.path();
        if path.is_dir() {
            rglob(&path, name, results);
        } else if entry.file_name().to_string_lossy() == name
            || (name.starts_with("*.") && entry.file_name().to_string_lossy().ends_with(&name[1..]))
        {
            results.push(path);
        }
    }
}

fn get_str<'v>(value: &'v Value, key: &str) -> Result<&'v str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing string field {key}"))
}

fn get_i64(value: &Value, key: &str) -> Result<i64> {
    let field = value
        .get(key)
        .ok_or_else(|| anyhow!("missing integer field {key}"))?;
    match field {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|f| f as i64))
            .ok_or_else(|| anyhow!("field {key} is not an integer")),
        Value::String(text) => text
            .parse()
            .map_err(|_| anyhow!("field {key} is not an integer")),
        _ => bail!("field {key} is not an integer"),
    }
}

fn get_f64(value: &Value, key: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("missing float field {key}"))
}

// ---------------------------------------------------------------------------
// Pending / failure semantics
// ---------------------------------------------------------------------------

/// The action is healthy but its external result is not complete yet.
#[derive(Debug)]
struct PendingAction(String);

impl std::fmt::Display for PendingAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl std::error::Error for PendingAction {}

fn pending(detail: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(PendingAction(detail.into()))
}

fn as_pending(error: &anyhow::Error) -> Option<&PendingAction> {
    error.downcast_ref::<PendingAction>()
}

// ---------------------------------------------------------------------------
// Action journal
// ---------------------------------------------------------------------------

fn action_key(
    action: &str,
    corpus_watermark: i64,
    accepted_ancestor: &str,
    recipe: &str,
    seed: i64,
    catalog_sha256: &str,
    evaluator_sha256: &str,
) -> Result<String> {
    let material = json!({
        "action": action,
        "corpus_watermark": corpus_watermark,
        "accepted_ancestor": accepted_ancestor,
        "recipe": recipe,
        "seed": seed,
        "catalog_sha256": catalog_sha256,
        "evaluator_sha256": evaluator_sha256,
    });
    Ok(sha256_text(&py_dumps(&material, PyStyle::Compact)?))
}

struct ActionJournal {
    db: Connection,
    events: PathBuf,
}

impl ActionJournal {
    fn open(root: &Path) -> Result<Self> {
        let ops = root.join("ops");
        fs::create_dir_all(&ops)?;
        let db = Connection::open(ops.join("controller.sqlite"))?;
        db.busy_timeout(std::time::Duration::from_secs(60))?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS actions(
              idempotency_key TEXT PRIMARY KEY,
              action TEXT NOT NULL,
              status TEXT NOT NULL,
              request_json TEXT NOT NULL,
              result_json TEXT,
              attempts INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL
            ) WITHOUT ROWID",
        )?;
        Ok(Self {
            db,
            events: ops.join("controller-events.jsonl"),
        })
    }

    fn event(&self, key: &str, action: &str, status: &str, payload: &Value) -> Result<()> {
        let line = py_dumps(
            &json!({
                "schema": "JopeSelfplayControllerEvent.v1",
                "idempotency_key": key,
                "action": action,
                "status": status,
                "at_ms": now_ms(),
                "payload": payload,
            }),
            PyStyle::Compact,
        )?;
        let mut handle = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(&self.events)?;
        handle.write_all(line.as_bytes())?;
        handle.write_all(b"\n")?;
        handle.sync_all()?;
        Ok(())
    }

    fn perform(
        &self,
        key: &str,
        action: &str,
        request: &Value,
        callback: impl FnOnce(&Value) -> Result<Value>,
    ) -> Result<Value> {
        let existing: Option<(String, Option<String>, i64)> = self
            .db
            .query_row(
                "SELECT status, result_json, attempts FROM actions WHERE idempotency_key=?",
                [key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if let Some((status, result_json, _)) = &existing {
            if status == "succeeded" {
                let stored = result_json
                    .as_ref()
                    .ok_or_else(|| anyhow!("succeeded action has no result"))?;
                return Ok(serde_json::from_str(stored)?);
            }
        }
        let attempts = existing.map(|(_, _, count)| count + 1).unwrap_or(1);
        self.db.execute(
            "INSERT INTO actions VALUES(?,?,?,?,?,?,?)
             ON CONFLICT(idempotency_key) DO UPDATE SET
               status=excluded.status,request_json=excluded.request_json,
               attempts=excluded.attempts,updated_at_ms=excluded.updated_at_ms",
            params![
                key,
                action,
                "running",
                py_dumps(request, PyStyle::Default)?,
                Option::<String>::None,
                attempts,
                now_ms(),
            ],
        )?;
        self.event(key, action, "started", &json!({"attempt": attempts}))?;
        match callback(request) {
            Ok(result) => {
                self.db.execute(
                    "UPDATE actions SET status='succeeded',result_json=?,updated_at_ms=?
                     WHERE idempotency_key=?",
                    params![py_dumps(&result, PyStyle::Default)?, now_ms(), key],
                )?;
                self.event(key, action, "succeeded", &result)?;
                Ok(result)
            }
            Err(error) => {
                if let Some(detail) = as_pending(&error) {
                    self.db.execute(
                        "UPDATE actions SET status='pending',updated_at_ms=? WHERE idempotency_key=?",
                        params![now_ms(), key],
                    )?;
                    self.event(
                        key,
                        action,
                        "pending",
                        &json!({"detail": detail.to_string()}),
                    )?;
                } else {
                    self.db.execute(
                        "UPDATE actions SET status='failed',updated_at_ms=? WHERE idempotency_key=?",
                        params![now_ms(), key],
                    )?;
                    self.event(key, action, "failed", &json!({"error": error.to_string()}))?;
                }
                Err(error)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Catalog validation (verify_contexts=False surface of the Python original)
// ---------------------------------------------------------------------------

/// Per-task authority metric (R1/D2); must agree with
/// `jope_dime_regressor::task::TaskKind::primary_metric`.
fn expected_catalog_metric(task: &str) -> Result<&'static str> {
    Task::parse(task)
        .map(Task::primary_metric)
        .ok_or_else(|| anyhow!("catalog contains unsupported task {task}"))
}

fn declared_catalog_counts(manifest: &Value) -> BTreeMap<String, i64> {
    let mut counts = BTreeMap::new();
    let declared = manifest
        .get("task_counts")
        .or_else(|| manifest.get("declared_task_counts"));
    if let Some(Value::Object(map)) = declared {
        for task in TASKS {
            // The legacy tasks must always be declared (-1 fails closed
            // below); catalogs written before the new tasks existed simply
            // omit them, which is a declared count of zero.
            let missing = if matches!(task, "regression" | "binary") {
                -1
            } else {
                0
            };
            counts.insert(
                task.to_owned(),
                map.get(task).and_then(Value::as_i64).unwrap_or(missing),
            );
        }
        return counts;
    }
    counts.insert(
        "regression".to_owned(),
        manifest
            .get("regression_count")
            .and_then(Value::as_i64)
            .unwrap_or(-1),
    );
    counts.insert(
        "binary".to_owned(),
        manifest
            .get("binary_count")
            .and_then(Value::as_i64)
            .unwrap_or(-1),
    );
    counts
}

fn validate_catalog_counts(
    entries: &[Value],
    declared: &BTreeMap<String, i64>,
) -> Result<BTreeMap<String, i64>> {
    if declared.values().any(|count| *count < 0) {
        bail!("catalog manifest does not declare every per-task count");
    }
    let known_tasks: BTreeSet<&str> = TASKS.into();
    let declared_tasks = declared.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if !declared_tasks.is_subset(&known_tasks)
        || !declared_tasks.contains("regression")
        || !declared_tasks.contains("binary")
    {
        bail!("catalog manifest declares unsupported task counts");
    }
    // Missing new-task declarations mean zero datasets for that task; the
    // legacy-task declarations stay mandatory (checked just above).
    let mut expected: BTreeMap<String, i64> =
        TASKS.iter().map(|task| ((*task).to_owned(), 0)).collect();
    for (task, count) in declared {
        expected.insert(task.clone(), *count);
    }
    let mut counts: BTreeMap<String, i64> =
        TASKS.iter().map(|task| ((*task).to_owned(), 0)).collect();
    let mut seen = BTreeSet::new();
    for entry in entries {
        let task = entry.get("task").and_then(Value::as_str).unwrap_or("");
        let dataset = entry
            .get("dataset_hash")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !counts.contains_key(task) || dataset.is_empty() {
            bail!("catalog contains an unsupported or anonymous dataset");
        }
        if !seen.insert((task.to_owned(), dataset.to_owned())) {
            bail!("catalog repeats {task}:{dataset}");
        }
        *counts.get_mut(task).expect("checked") += 1;
    }
    if counts != expected {
        bail!("catalog count mismatch: expected {declared:?}, found {counts:?}");
    }
    Ok(counts)
}

/// Port of `validate_catalog_manifest(pointer, verify_contexts=False)` for
/// both the v1 catalog schema and `JopeExpandedCatalog.v2`.
fn validate_catalog_manifest(pointer_path: &Path) -> Result<Value> {
    let value = read_json(pointer_path)?;
    for field in ["catalog", "catalog_sha256"] {
        if value
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or("")
            .is_empty()
        {
            bail!("catalog pointer is missing {field}");
        }
    }
    let catalog_path = PathBuf::from(get_str(&value, "catalog")?);
    if sha256_file(&catalog_path)? != get_str(&value, "catalog_sha256")? {
        bail!("catalog pointer hash mismatch");
    }
    let manifest = read_json(&catalog_path)?;
    let expanded = manifest.get("schema").and_then(Value::as_str) == Some("JopeExpandedCatalog.v2");
    for field in ["data", "data_sha256"] {
        if value
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or("")
            .is_empty()
        {
            if expanded {
                bail!("expanded catalog pointer is missing {field}");
            }
            bail!("catalog pointer is missing {field}");
        }
    }
    let data_path = PathBuf::from(get_str(&value, "data")?);
    let data_sha = get_str(&value, "data_sha256")?;
    if expanded {
        if !data_path.is_file()
            || sha256_file(&data_path)? != data_sha
            || manifest.get("data_sha256").and_then(Value::as_str) != Some(data_sha)
        {
            bail!("expanded catalog context pointer hash mismatch");
        }
        if manifest.get("immutable") != Some(&Value::Bool(true))
            || manifest.get("append_only") != Some(&Value::Bool(true))
        {
            bail!("expanded catalog is not immutable and append-only");
        }
    } else {
        if sha256_file(&data_path)? != data_sha {
            bail!("catalog data pointer hash mismatch");
        }
        if manifest.get("immutable") != Some(&Value::Bool(true)) {
            bail!("catalog manifest is not declared immutable");
        }
        if manifest.get("data_sha256").and_then(Value::as_str) != Some(data_sha) {
            bail!("catalog manifest and pointer disagree on context data");
        }
    }
    let entries = manifest
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("catalog manifest entries are absent"))?;
    let counts = validate_catalog_counts(entries, &declared_catalog_counts(&manifest))?;
    if manifest
        .get("row_count")
        .and_then(Value::as_i64)
        .unwrap_or(-1)
        != entries.len() as i64
    {
        bail!("catalog declared row count mismatch");
    }
    let stride = manifest
        .get("record_stride_bytes")
        .and_then(Value::as_i64)
        .unwrap_or(-1);
    if stride != CATALOG_STRIDE_BYTES as i64 {
        bail!("catalog context stride mismatch");
    }
    if fs::metadata(&data_path)?.len() != entries.len() as u64 * CATALOG_STRIDE_BYTES {
        bail!("catalog context blob row count mismatch");
    }
    let mut indexes: Vec<i64> = entries
        .iter()
        .map(|entry| {
            entry
                .get("catalog_index")
                .and_then(Value::as_i64)
                .unwrap_or(-1)
        })
        .collect();
    indexes.sort_unstable();
    if indexes != (0..entries.len() as i64).collect::<Vec<_>>() {
        bail!("catalog indexes are not unique and contiguous");
    }
    if manifest.get("source_family_disjoint") != Some(&Value::Bool(true)) {
        bail!("catalog does not declare family-disjoint splits");
    }
    let mut split_by_family: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entry in entries {
        let task = get_str(entry, "task")?;
        let metric = entry.get("metric").and_then(Value::as_str).unwrap_or("");
        let expected_metric = expected_catalog_metric(task)?;
        if metric != expected_metric {
            bail!("catalog metric mismatch for {task}");
        }
        let family = entry
            .get("source_family")
            .and_then(Value::as_str)
            .unwrap_or("");
        let split = entry.get("split").and_then(Value::as_str).unwrap_or("");
        if family.is_empty() || !matches!(split, "train" | "validation") {
            bail!("catalog contains an invalid family/split identity");
        }
        split_by_family
            .entry(family.to_owned())
            .or_default()
            .insert(split.to_owned());
        for field in ["context_sha256", "train_sha256", "test_sha256"] {
            if !is_sha256(entry.get(field).and_then(Value::as_str).unwrap_or("")) {
                bail!("catalog entry has invalid {field}");
            }
        }
    }
    if split_by_family.values().any(|splits| splits.len() != 1) {
        bail!("catalog source families cross splits");
    }
    let mut task_counts = Map::new();
    for (task, count) in counts {
        task_counts.insert(task, json!(count));
    }
    Ok(json!({
        "pointer": pointer_path.to_string_lossy(),
        "pointer_sha256": sha256_file(pointer_path)?,
        "catalog": catalog_path.to_string_lossy(),
        "catalog_sha256": get_str(&value, "catalog_sha256")?,
        "data": data_path.to_string_lossy(),
        "data_sha256": data_sha,
        "row_count": entries.len(),
        "task_counts": task_counts,
        "source_family_disjoint": true,
    }))
}

// ---------------------------------------------------------------------------
// Scheduler helpers
// ---------------------------------------------------------------------------

fn run_command(program: &str, args: &[&str]) -> Result<(i32, String, String)> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("spawn {program}"))?;
    Ok((
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

fn scq_json(args: &[&str]) -> Result<Value> {
    let (code, stdout, stderr) = run_command(scq_str(), args)?;
    if code != 0 {
        bail!(
            "{}",
            if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            }
        );
    }
    if stdout.trim().is_empty() {
        return Ok(json!({}));
    }
    Ok(serde_json::from_str(&stdout)?)
}

fn scheduler_jobs() -> Result<BTreeMap<String, Value>> {
    let value = scq_json(&["--json", "ls"])?;
    let mut jobs = BTreeMap::new();
    if let Some(rows) = value
        .get("Jobs")
        .and_then(|j| j.get("jobs"))
        .and_then(Value::as_array)
    {
        for row in rows {
            if let Some(id) = row.get("id").and_then(Value::as_str) {
                if !id.is_empty() {
                    jobs.insert(id.to_owned(), row.clone());
                }
            }
        }
    }
    Ok(jobs)
}

fn job_state<'j>(jobs: &'j BTreeMap<String, Value>, job_id: &str) -> Option<&'j str> {
    jobs.get(job_id)
        .and_then(|row| row.get("state"))
        .and_then(Value::as_str)
}

/// `scq check` then `scq submit`, tolerating an already-submitted job.
fn check_and_submit(spec: &str) -> Result<()> {
    let (code, stdout, stderr) = run_command(scq_str(), &["check", spec])?;
    if code != 0 {
        bail!(
            "{}",
            if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            }
        );
    }
    let (code, stdout, stderr) = run_command(scq_str(), &["submit", spec])?;
    let detail = if stderr.trim().is_empty() {
        stdout.trim()
    } else {
        stderr.trim()
    };
    if code != 0 && !detail.to_lowercase().contains("already exists") {
        bail!("{detail}");
    }
    Ok(())
}

/// Locate outputs for a job the scheduler has already pruned from its records.
fn pruned_job_outputs(job_id: &str) -> Option<PathBuf> {
    let attempts = Path::new(runs_root_str()).join(job_id).join("attempts");
    if !attempts.is_dir() {
        return None;
    }
    let mut outputs: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for entry in fs::read_dir(&attempts).ok()?.flatten() {
        let candidate = entry.path().join("outputs");
        if candidate.is_dir() {
            let modified = fs::metadata(&candidate)
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH);
            outputs.push((modified, candidate));
        }
    }
    outputs.sort();
    for (_, candidate) in outputs.into_iter().rev() {
        let mut results = Vec::new();
        rglob(&candidate, "train-result.json", &mut results);
        if !results.is_empty() {
            return Some(candidate);
        }
    }
    None
}

fn fetch_job_output(job_id: &str, destination: &Path) -> Result<PathBuf> {
    if destination.is_dir() {
        return Ok(destination.to_path_buf());
    }
    let name = destination
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("fetch destination has no name"))?;
    let temporary = destination.with_file_name(format!(".{name}.fetch-{}", std::process::id()));
    fs::create_dir_all(&temporary)?;
    let (code, stdout, stderr) = run_command(
        scq_str(),
        &["fetch", job_id, "--output", &temporary.to_string_lossy()],
    )?;
    if code != 0 {
        let _ = fs::remove_dir_all(&temporary);
        bail!(
            "{}",
            if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            }
        );
    }
    fs::rename(&temporary, destination)?;
    Ok(destination.to_path_buf())
}

// ---------------------------------------------------------------------------
// Python delegation shim
// ---------------------------------------------------------------------------

fn python_handler(name: &str, request: &Value, root: &Path) -> Result<Value> {
    let work = root.join("ops");
    fs::create_dir_all(&work)?;
    let request_path = work.join(format!(
        ".controller-shim-{}-{}.json",
        name,
        std::process::id()
    ));
    fs::write(&request_path, py_dumps(request, PyStyle::Default)?)?;
    let output = Command::new("/usr/bin/python3")
        .args(["-c", py_shim(), name, &request_path.to_string_lossy()])
        .current_dir(scripts_str())
        .env(
            "PYTHONPATH",
            format!(
                "{prime_src_str}:{scripts_str}",
                scripts_str = scripts_str(),
                prime_src_str = prime_src_str()
            ),
        )
        .output()
        .context("spawn python shim")?;
    let _ = fs::remove_file(&request_path);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "python handler {name} failed: {}",
            stderr
                .lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: Value = serde_json::from_str(stdout.trim())
        .with_context(|| format!("python handler {name} returned invalid JSON"))?;
    if let Some(detail) = value.get("__pending__").and_then(Value::as_str) {
        return Err(pending(detail));
    }
    Ok(value)
}

fn milestone_status(checkpoint_sha256: &str) -> Result<Value> {
    let output = Command::new("/usr/bin/python3")
        .args(["-c", py_shim(), "milestone_status", checkpoint_sha256])
        .current_dir(scripts_str())
        .env(
            "PYTHONPATH",
            format!(
                "{prime_src_str}:{scripts_str}",
                scripts_str = scripts_str(),
                prime_src_str = prime_src_str()
            ),
        )
        .output()
        .context("spawn milestone shim")?;
    if !output.status.success() {
        bail!(
            "milestone_status failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(serde_json::from_str(
        String::from_utf8_lossy(&output.stdout).trim(),
    )?)
}

// ---------------------------------------------------------------------------
// Spec rendering (ports of test_exact_ops / test_exact_train renderers)
// ---------------------------------------------------------------------------

fn toml_string(value: &str) -> Result<String> {
    py_dumps(&Value::String(value.to_owned()), PyStyle::Compact)
}

fn input_block(name: &str, source: &Path, target: &str) -> Result<Vec<String>> {
    Ok(vec![
        "[[input]]".to_owned(),
        format!("name = {}", toml_string(name)?),
        format!("source = {}", toml_string(&source.to_string_lossy())?),
        format!("target = {}", toml_string(target)?),
        format!("sha256 = {}", toml_string(&sha256_file(source)?)?),
        String::new(),
    ])
}

struct ProposalSpec {
    job_id: String,
    spec: PathBuf,
    checkpoint_sha256: String,
    split: Option<String>,
    cohort_sha256: Option<String>,
    seed: i64,
    proposals_per_dataset: i64,
}

impl ProposalSpec {
    fn to_value(&self) -> Value {
        json!({
            "job_id": self.job_id,
            "spec": self.spec.to_string_lossy(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "split": self.split,
            "cohort_sha256": self.cohort_sha256,
            "seed": self.seed,
            "proposals_per_dataset": self.proposals_per_dataset,
        })
    }
}

/// Port of `test_exact_ops.render_checkpoint_proposal_spec` (including this
/// session's widened `xbabe2|xbabe3` hostname dependency).
#[allow(clippy::too_many_arguments)]
fn render_checkpoint_proposal_spec(
    path: &Path,
    checkpoint: &Path,
    split: Option<&str>,
    cohort: Option<&Path>,
    seed: i64,
    proposals_per_dataset: i64,
    root: &Path,
    attempt: u32,
) -> Result<ProposalSpec> {
    let pointer_path = root.join("catalog").join("current.json");
    let pointer = read_json(&pointer_path)?;
    let catalog = PathBuf::from(get_str(&pointer, "catalog")?);
    let data = PathBuf::from(get_str(&pointer, "data")?);
    let script = Path::new(scripts_str()).join("test_exact_dl_generate.py");
    let common = Path::new(scripts_str()).join("test_exact_common.py");
    let checkpoint_hash = sha256_file(checkpoint)?;
    let cohort_hash = match cohort {
        Some(path) => Some(sha256_file(path)?),
        None => None,
    };
    let split_name = split.unwrap_or("all");
    let cohort_or_split = cohort_hash
        .as_deref()
        .map(|hash| hash[..12].to_owned())
        .unwrap_or_else(|| split_name.to_owned());
    let mut job_id = format!(
        "jope-selfplay-propose-{}-{}-{}-cpu4-p2",
        &checkpoint_hash[..12],
        cohort_or_split,
        seed
    );
    if attempt > 1 {
        // A terminal-failed record blocks resubmission under the same id and the
        // scheduler offers no record removal, so retries get a fresh attempt id.
        job_id.push_str(&format!("-r{attempt}"));
    }
    let output = "dl-proposals.jsonl";
    // Reconstruct the exact Python-rendered args string.
    let mut args_line = format!(
        "args = [\"{{input:script}}\", \"--catalog-pointer\", \"{{input:pointer}}\", \
         \"--catalog\", \"{{input:catalog}}\", \"--data\", \"{{input:data}}\", \
         \"--checkpoint\", \"{{input:checkpoint}}\", \
         \"--output\", \"{{output:proposals}}/{output}\", \"--device\", \"cuda\", \
         \"--selection-policy\", \"{{input:selection_policy}}\", \
         \"--seed\", \"{seed}\", \"--proposals-per-dataset\", \
         \"{proposals_per_dataset}\", \"--batch-contexts\", \"64\", \
         \"--allow-deficient-floor\", \"8\""
    );
    if let Some(split) = split {
        args_line.push_str(&format!(", \"--split\", \"{split}\""));
    }
    if cohort.is_some() {
        args_line.push_str(", \"--cohort\", \"{input:cohort}\"");
    }
    args_line.push(']');
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = {}", toml_string(&job_id)?),
        "project = \"jope-selfplay-train\"".to_owned(),
        "tier = \"normal\"".to_owned(),
        "command = \"/usr/bin/python3\"".to_owned(),
        args_line,
        String::new(),
    ];
    lines.extend(PROPOSE.render_resource_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend(PROPOSE.render_placement_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend([
        "[env]".to_owned(),
        format!(
            "PYTHONPATH = \"{prime_src_str}:{scripts_str}\"",
            scripts_str = scripts_str(),
            prime_src_str = prime_src_str()
        ),
        format!("OMP_NUM_THREADS = \"{}\"", PROPOSE.omp_threads),
        String::new(),
        "[preempt]".to_owned(),
        format!("preemptible = {}", PROPOSE.preemptible),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
    ]);
    lines.extend(PROPOSE.render_limits_block().lines().map(str::to_owned));
    lines.push(String::new());
    let catalog_name = catalog
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let data_name = data
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let checkpoint_name = checkpoint
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut inputs: Vec<(String, PathBuf, String)> = vec![
        (
            "script".into(),
            script,
            "scripts/test_exact_dl_generate.py".into(),
        ),
        (
            "common".into(),
            common,
            "scripts/test_exact_common.py".into(),
        ),
        (
            "selection_policy".into(),
            PathBuf::from(proposal_selection_policy_str()),
            "scripts/proposal-selection-policy.v1.json".into(),
        ),
        (
            "pointer".into(),
            pointer_path,
            "catalog/current.json".into(),
        ),
        ("catalog".into(), catalog, format!("catalog/{catalog_name}")),
        ("data".into(), data, format!("catalog/{data_name}")),
        (
            "checkpoint".into(),
            checkpoint.to_path_buf(),
            format!("checkpoint/{checkpoint_name}"),
        ),
    ];
    if let Some(cohort) = cohort {
        let cohort_name = cohort
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        inputs.push((
            "cohort".into(),
            cohort.to_path_buf(),
            format!("cohort/{cohort_name}"),
        ));
    }
    for (name, source, target) in &inputs {
        lines.extend(input_block(name, source, target)?);
    }
    lines.extend([
        "[[output]]".to_owned(),
        "name = \"proposals\"".to_owned(),
        "source = \"proposals\"".to_owned(),
        "directory = true".to_owned(),
        String::new(),
    ]);
    fs::create_dir_all(
        path.parent()
            .ok_or_else(|| anyhow!("spec path has no parent"))?,
    )?;
    fs::write(path, lines.join("\n"))?;
    Ok(ProposalSpec {
        job_id,
        spec: path.to_path_buf(),
        checkpoint_sha256: checkpoint_hash,
        split: split.map(str::to_owned),
        cohort_sha256: cohort_hash,
        seed,
        proposals_per_dataset,
    })
}

/// Port of `test_exact_train.render_spec` (trainer tournament spec).
fn render_trainer_spec(
    corpus_manifest: &Path,
    recipe: &str,
    seed: i64,
    packed_manifest: &Path,
    root: &Path,
) -> Result<Value> {
    let manifest = read_json(corpus_manifest)?;
    let watermark = get_i64(&manifest, "source_label_watermark")?;
    if !matches!(recipe, "a" | "b" | "c") {
        bail!("checkpoint recipe must be a, b, or c");
    }
    let job_id = format!("jope-selfplay-train-{watermark:09}-{recipe}");
    let spec = root.join("specs").join(format!("{job_id}.toml"));
    let script = Path::new(scripts_str()).join("test_exact_train.py");
    let common = Path::new(scripts_str()).join("test_exact_common.py");
    let train_source = Path::new(prime_src_str()).join("jope/policy/train_v3.py");
    let output_name = format!("watermark-{watermark:09}");
    let (warm, warm_hash) = accepted_checkpoint()?;
    let mut inputs: Vec<(String, PathBuf, String)> = vec![
        (
            "script".into(),
            script.clone(),
            format!("scripts/{}", script.file_name().unwrap().to_string_lossy()),
        ),
        (
            "common".into(),
            common.clone(),
            format!("scripts/{}", common.file_name().unwrap().to_string_lossy()),
        ),
        (
            "packed_source".into(),
            Path::new(scripts_str()).join("test_exact_packed_corpus.py"),
            "scripts/test_exact_packed_corpus.py".into(),
        ),
        (
            "manifest".into(),
            corpus_manifest.to_path_buf(),
            "corpus/corpus-manifest.json".into(),
        ),
        (
            "train".into(),
            PathBuf::from(get_str(&manifest, "train")?),
            "corpus/corpus.train.jsonl".into(),
        ),
        (
            "validation".into(),
            PathBuf::from(get_str(&manifest, "validation")?),
            "corpus/corpus.val.jsonl".into(),
        ),
        (
            "checkpoint".into(),
            warm.clone(),
            format!("checkpoint/{}", warm.file_name().unwrap().to_string_lossy()),
        ),
        (
            "trainer_source".into(),
            train_source,
            "provenance/train_v3.py".into(),
        ),
    ];
    let packed = read_json(packed_manifest)?;
    let pack_root = packed_manifest
        .parent()
        .ok_or_else(|| anyhow!("packed manifest has no parent"))?;
    inputs.push((
        "packed_manifest".into(),
        packed_manifest.to_path_buf(),
        "packed/manifest.json".into(),
    ));
    inputs.push((
        "packed_catalog".into(),
        PathBuf::from(get_str(&packed, "catalog_data")?),
        "packed/catalog-data.f32le".into(),
    ));
    let mut packed_names: Vec<String> = match packed.get("files") {
        Some(Value::Object(map)) => map.keys().cloned().collect(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => bail!("packed manifest files listing is absent"),
    };
    packed_names.sort();
    for (index, name) in packed_names.iter().enumerate() {
        inputs.push((
            format!("packed_{index}"),
            pack_root.join(name),
            format!("packed/{name}"),
        ));
    }
    let packed_arg = ", \"--packed-manifest\", \"{input:packed_manifest}\", \
         \"--packed-source-train\", \"{input:train}\", \
         \"--packed-source-validation\", \"{input:validation}\", \
         \"--packed-catalog-data\", \"{input:packed_catalog}\"";
    let args_line = format!(
        "args = [\"{{input:script}}\", \"train\", \"--corpus-manifest\", \"{{input:manifest}}\", \
         \"--output\", \"{{output:training}}/{output_name}\", \"--device\", \"cuda\", \
         \"--warm\", \"{{input:checkpoint}}\", \"--recipe\", \"{recipe}\", \"--seed\", \"{seed}\"{packed_arg}]"
    );
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = \"{job_id}\""),
        "project = \"jope-selfplay-train\"".to_owned(),
        "tier = \"normal\"".to_owned(),
        "command = \"/usr/bin/python3\"".to_owned(),
        args_line,
        String::new(),
    ];
    lines.extend(TRAINER.render_resource_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend(TRAINER.render_placement_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend([
        "[env]".to_owned(),
        format!(
            "PYTHONPATH = \"{prime_src_str}:{scripts_str}\"",
            scripts_str = scripts_str(),
            prime_src_str = prime_src_str()
        ),
        format!("OMP_NUM_THREADS = \"{}\"", TRAINER.omp_threads),
        String::new(),
        "[preempt]".to_owned(),
        format!("preemptible = {}", TRAINER.preemptible),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
    ]);
    lines.extend(TRAINER.render_limits_block().lines().map(str::to_owned));
    lines.push(String::new());
    for (name, source, target) in &inputs {
        lines.extend([
            "[[input]]".to_owned(),
            format!("name = \"{name}\""),
            format!("source = {}", toml_string(&source.to_string_lossy())?),
            format!("target = {}", toml_string(target)?),
            format!("sha256 = \"{}\"", sha256_file(source)?),
            String::new(),
        ]);
    }
    lines.extend([
        "[[output]]".to_owned(),
        "name = \"training\"".to_owned(),
        "source = \"training\"".to_owned(),
        "directory = true".to_owned(),
        String::new(),
    ]);
    fs::create_dir_all(spec.parent().unwrap())?;
    fs::write(&spec, lines.join("\n"))?;
    Ok(json!({
        "job_id": job_id,
        "spec": spec.to_string_lossy(),
        "watermark": watermark,
        "recipe": recipe,
        "seed": seed,
        "accepted_ancestor_sha256": warm_hash,
    }))
}

/// Port of `test_exact_train.accepted_checkpoint` (learning → accepted →
/// warm-start bootstrap, always against the production root).
fn accepted_checkpoint() -> Result<(PathBuf, String)> {
    for pointer in [
        Path::new(root_str()).join("training/learning.json"),
        Path::new(root_str()).join("training/accepted.json"),
    ] {
        if !pointer.is_file() {
            continue;
        }
        let value = read_json(&pointer)?;
        let path = PathBuf::from(get_str(&value, "checkpoint")?);
        let digest = get_str(&value, "checkpoint_sha256")?.to_owned();
        if path.is_file() && sha256_file(&path)? == digest {
            return Ok((path, digest));
        }
        bail!("accepted checkpoint pointer failed content verification");
    }
    Ok((PathBuf::from(warm_str()), WARM_SHA.to_owned()))
}

// ---------------------------------------------------------------------------
// Native phase handlers
// ---------------------------------------------------------------------------

fn watermark_dir(root: &Path, watermark: i64) -> PathBuf {
    root.join("training")
        .join("tournament-work")
        .join(format!("watermark-{watermark:09}"))
}

fn handle_launch_checkpoint_tournament(request: &Value, root: &Path) -> Result<Value> {
    let corpus = PathBuf::from(get_str(
        request
            .pointer("/artifacts/freeze_corpus")
            .ok_or_else(|| anyhow!("missing freeze_corpus artifact"))?,
        "manifest",
    )?);
    let packed = PathBuf::from(get_str(
        request
            .pointer("/artifacts/build_packed_corpus")
            .ok_or_else(|| anyhow!("missing build_packed_corpus artifact"))?,
        "manifest",
    )?);
    let recipes = request
        .get("recipes")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("tournament recipes are absent"))?;
    let seed = get_i64(request, "seed")?;
    let ancestor = get_str(request, "accepted_ancestor")?;
    let mut jobs = Vec::new();
    for (offset, recipe) in recipes.iter().enumerate() {
        let recipe = recipe
            .as_str()
            .ok_or_else(|| anyhow!("tournament recipe is not a string"))?;
        let rendered = render_trainer_spec(&corpus, recipe, seed + offset as i64, &packed, root)?;
        if get_str(&rendered, "accepted_ancestor_sha256")? != ancestor {
            bail!("accepted checkpoint changed before tournament dispatch");
        }
        let spec = get_str(&rendered, "spec")?.to_owned();
        check_and_submit(&spec)?;
        let mut job = rendered;
        job["spec_sha256"] = Value::String(sha256_file(Path::new(&spec))?);
        jobs.push(job);
    }
    Ok(json!({"jobs": jobs, "job_count": jobs.len()}))
}

fn handle_fetch_candidate_checkpoints(request: &Value, root: &Path) -> Result<Value> {
    let jobs = scheduler_jobs()?;
    let launched = request
        .pointer("/artifacts/launch_checkpoint_tournament/jobs")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing launch_checkpoint_tournament artifact"))?;
    let mut pruned: BTreeMap<String, Option<PathBuf>> = BTreeMap::new();
    for item in launched {
        let job_id = get_str(item, "job_id")?;
        if !jobs.contains_key(job_id) {
            pruned.insert(job_id.to_owned(), pruned_job_outputs(job_id));
        }
    }
    let mut pending_jobs = Vec::new();
    let mut failed = BTreeMap::new();
    for item in launched {
        let job_id = get_str(item, "job_id")?;
        let state = job_state(&jobs, job_id);
        if matches!(state, Some("failed") | Some("cancelled")) {
            failed.insert(job_id.to_owned(), state.unwrap_or("").to_owned());
        }
        let recovered = pruned.get(job_id).map(|p| p.is_some()).unwrap_or(false);
        if state != Some("completed") && !recovered {
            pending_jobs.push(job_id.to_owned());
        }
    }
    if !failed.is_empty() {
        bail!("checkpoint tournament jobs failed: {failed:?}");
    }
    if !pending_jobs.is_empty() {
        return Err(pending(format!(
            "waiting for checkpoint jobs: {}",
            pending_jobs.join(",")
        )));
    }
    let fetch_root = watermark_dir(root, get_i64(request, "watermark")?).join("fetched");
    fs::create_dir_all(&fetch_root)?;
    let mut candidates = Vec::new();
    for item in launched {
        let job_id = get_str(item, "job_id")?;
        let recipe = get_str(item, "recipe")?;
        let destination = fetch_root.join(recipe);
        if !destination.is_dir() {
            let temporary = fetch_root.join(format!(".{recipe}.fetch-{}", std::process::id()));
            fs::create_dir_all(&temporary)?;
            match pruned.get(job_id).and_then(|p| p.as_ref()) {
                Some(source) => copy_tree(source, &temporary.join("outputs"))?,
                None => {
                    let (code, stdout, stderr) = run_command(
                        scq_str(),
                        &["fetch", job_id, "--output", &temporary.to_string_lossy()],
                    )?;
                    if code != 0 {
                        let _ = fs::remove_dir_all(&temporary);
                        bail!(
                            "{}",
                            if stderr.trim().is_empty() {
                                stdout.trim()
                            } else {
                                stderr.trim()
                            }
                        );
                    }
                }
            }
            fs::rename(&temporary, &destination)?;
        }
        let mut results = Vec::new();
        rglob(&destination, "train-result.json", &mut results);
        if results.len() != 1 {
            bail!("{job_id} fetch has {} train-result files", results.len());
        }
        let summary = read_json(&results[0])?;
        if summary
            .get("accepted_ancestor_sha256")
            .and_then(Value::as_str)
            != Some(get_str(request, "accepted_ancestor")?)
            || summary.get("recipe").and_then(Value::as_str) != Some(recipe)
            || get_i64(&summary, "seed")? != get_i64(item, "seed")?
        {
            bail!("{job_id} training provenance drift");
        }
        let digest = summary
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let mut matches = Vec::new();
        rglob(&destination, "*.model.pt", &mut matches);
        let mut verified: Vec<PathBuf> = Vec::new();
        for path in matches {
            if sha256_file(&path)? == digest {
                verified.push(path);
            }
        }
        verified.sort_by_key(|path| {
            (
                path.file_name().map(|n| n.to_string_lossy().into_owned())
                    != Some(format!("{digest}.model.pt")),
                path.to_string_lossy().into_owned(),
            )
        });
        let Some(best) = verified.first() else {
            bail!("{job_id} has no verified checkpoint");
        };
        let immutable = fetch_root
            .join("checkpoints")
            .join(format!("{digest}.model.pt"));
        fs::create_dir_all(immutable.parent().unwrap())?;
        if !immutable.exists() {
            fs::copy(best, &immutable)?;
        }
        if sha256_file(&immutable)? != digest {
            bail!("fetched immutable checkpoint hash mismatch");
        }
        candidates.push(json!({
            "recipe": recipe,
            "seed": item.get("seed").cloned().unwrap_or(Value::Null),
            "checkpoint": immutable.to_string_lossy(),
            "checkpoint_sha256": digest,
            "train_result": results[0].to_string_lossy(),
            "train_result_sha256": sha256_file(&results[0])?,
        }));
    }
    Ok(json!({"candidates": candidates, "candidate_count": candidates.len()}))
}

fn handle_generate_gate_proposals(request: &Value, root: &Path) -> Result<Value> {
    let work = watermark_dir(root, get_i64(request, "watermark")?).join("gate-proposals");
    fs::create_dir_all(&work)?;
    let frozen_champions = work.join("frozen-champions.json");
    if !frozen_champions.is_file() {
        fs::copy(root.join("indexes/champions.json"), &frozen_champions)?;
    }
    let (baseline_path, baseline_hash) = accepted_checkpoint()?;
    if baseline_hash != get_str(request, "accepted_ancestor")? {
        bail!("gate proposal baseline changed after corpus freeze");
    }
    let mut checkpoints = vec![json!({
        "role": "baseline",
        "checkpoint": baseline_path.to_string_lossy(),
        "checkpoint_sha256": baseline_hash,
    })];
    for candidate in request
        .pointer("/artifacts/fetch_candidate_checkpoints/candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing fetch_candidate_checkpoints artifact"))?
    {
        let mut row = json!({"role": "candidate"});
        for (key, value) in candidate.as_object().unwrap() {
            row[key] = value.clone();
        }
        checkpoints.push(row);
    }
    let seed = get_i64(request, "seed")?;
    let jobs = scheduler_jobs()?;
    let mut rendered = Vec::new();
    // Raised from 3: attempts r1-r3 were burned before the deficit-tolerant
    // sampler flag existed. Long-term the attempt identity should be
    // content-aware (bind the script sha into the id) so a fixed generator
    // never has to spend the surviving budget of a broken one.
    const MAX_PROPOSAL_ATTEMPTS: u32 = 5;
    for checkpoint in &checkpoints {
        let digest = get_str(checkpoint, "checkpoint_sha256")?;
        let render_attempt = |attempt: u32| {
            let spec_name = if attempt > 1 {
                format!("{digest}-r{attempt}.toml")
            } else {
                format!("{digest}.toml")
            };
            render_checkpoint_proposal_spec(
                &work.join(spec_name),
                Path::new(get_str(checkpoint, "checkpoint")?),
                Some("validation"),
                None,
                seed,
                64,
                root,
                attempt,
            )
        };
        // The scheduler prunes terminal records on its own clock, so decide from
        // the full attempt family: a completed attempt wins, then the highest
        // live one; only then does a failed latest attempt earn a successor.
        let base_spec = render_attempt(1)?;
        let id_for = |attempt: u32| {
            if attempt > 1 {
                format!("{}-r{attempt}", base_spec.job_id)
            } else {
                base_spec.job_id.clone()
            }
        };
        let states: Vec<Option<String>> = (1..=MAX_PROPOSAL_ATTEMPTS)
            .map(|attempt| job_state(&jobs, &id_for(attempt)).map(str::to_owned))
            .collect();
        let completed = (1..=MAX_PROPOSAL_ATTEMPTS)
            .rev()
            .find(|a| states[(*a - 1) as usize].as_deref() == Some("completed"));
        let live = (1..=MAX_PROPOSAL_ATTEMPTS).rev().find(|a| {
            matches!(
                states[(*a - 1) as usize].as_deref(),
                Some("queued" | "starting" | "dispatching" | "running" | "preempting" | "held")
            )
        });
        let highest_known = (1..=MAX_PROPOSAL_ATTEMPTS)
            .rev()
            .find(|a| states[(*a - 1) as usize].is_some());
        let (attempt, needs_submit) = if let Some(attempt) = completed {
            (attempt, false)
        } else if let Some(attempt) = live {
            (attempt, false)
        } else if let Some(highest) = highest_known {
            if highest >= MAX_PROPOSAL_ATTEMPTS {
                bail!(
                    "checkpoint proposal job failed after {highest} attempts: {}",
                    id_for(highest)
                );
            }
            (highest + 1, true)
        } else {
            (1, true)
        };
        let spec = if attempt == 1 {
            base_spec
        } else {
            render_attempt(attempt)?
        };
        if needs_submit && job_state(&jobs, &spec.job_id).is_none() {
            check_and_submit(&spec.spec.to_string_lossy())?;
        }
        if spec.checkpoint_sha256 != digest {
            bail!("proposal spec checkpoint content drift");
        }
        let mut row = checkpoint.clone();
        for (key, value) in spec.to_value().as_object().unwrap() {
            row[key] = value.clone();
        }
        rendered.push(row);
    }
    let jobs = scheduler_jobs()?;
    let pending_jobs: Vec<String> = rendered
        .iter()
        .filter(|row| {
            job_state(
                &jobs,
                row.get("job_id").and_then(Value::as_str).unwrap_or(""),
            ) != Some("completed")
        })
        .map(|row| {
            row.get("job_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned()
        })
        .collect();
    if !pending_jobs.is_empty() {
        return Err(pending(format!(
            "waiting for gate proposal jobs: {}",
            pending_jobs.join(",")
        )));
    }
    let mut proposal_sets = Vec::new();
    for row in rendered {
        let job_id = get_str(&row, "job_id")?;
        let digest = get_str(&row, "checkpoint_sha256")?;
        let output = fetch_job_output(job_id, &work.join("fetched").join(digest))?;
        let mut proposals = Vec::new();
        rglob(&output, "dl-proposals.jsonl", &mut proposals);
        let mut summaries = Vec::new();
        rglob(&output, "dl-proposals.summary.json", &mut summaries);
        if proposals.len() != 1 || summaries.len() != 1 {
            bail!("{job_id} proposal fetch is incomplete");
        }
        let summary = read_json(&summaries[0])?;
        if summary
            .get("model_checkpoint_sha256")
            .and_then(Value::as_str)
            != Some(digest)
            || summary.get("split").and_then(Value::as_str) != Some("validation")
            || get_i64(&summary, "seed")? != seed
            || Some(sha256_file(&proposals[0])?.as_str())
                != summary.get("output_sha256").and_then(Value::as_str)
        {
            bail!("{job_id} proposal provenance drift");
        }
        let mut entry = row.clone();
        entry["proposals"] = Value::String(proposals[0].to_string_lossy().into_owned());
        entry["proposals_sha256"] = Value::String(sha256_file(&proposals[0])?);
        entry["summary"] = Value::String(summaries[0].to_string_lossy().into_owned());
        entry["summary_sha256"] = Value::String(sha256_file(&summaries[0])?);
        entry["proposal_validity"] = summary["proposal_validity"].clone();
        entry["p95_proposal_latency_ms"] = summary["p95_proposal_latency_ms"].clone();
        proposal_sets.push(entry);
    }
    Ok(json!({
        "development_split": "validation",
        "proposal_budget_per_dataset": 64,
        "seed": seed,
        "proposal_sets": proposal_sets,
        "frozen_champions": frozen_champions.to_string_lossy(),
        "frozen_champions_sha256": sha256_file(&frozen_champions)?,
    }))
}

fn handle_build_acceptance_evidence(request: &Value, root: &Path) -> Result<Value> {
    let gate_evaluation = request
        .pointer("/artifacts/evaluate_frozen_gate")
        .ok_or_else(|| anyhow!("missing evaluate_frozen_gate artifact"))?;
    let mut evidence = gate_evaluation
        .get("acceptance_evidence")
        .cloned()
        .unwrap_or_else(|| gate_evaluation.clone());
    if !evidence.is_object() {
        bail!("frozen gate did not produce acceptance evidence");
    }
    for key in [
        "baseline",
        "candidate",
        "proposal_validity",
        "replay_verification",
    ] {
        if evidence.get(key).is_none() {
            bail!("frozen gate evidence is missing {key}");
        }
    }
    let mut fetched = BTreeMap::new();
    for candidate in request
        .pointer("/artifacts/fetch_candidate_checkpoints/candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing fetch_candidate_checkpoints artifact"))?
    {
        fetched.insert(
            get_str(candidate, "checkpoint_sha256")?.to_owned(),
            candidate.clone(),
        );
    }
    let checkpoint_hash = evidence
        .pointer("/candidate/checkpoint_sha256")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let Some(candidate) = fetched.get(&checkpoint_hash) else {
        bail!("gate winner is not one of the fetched tournament checkpoints");
    };
    evidence["candidate"]["checkpoint_path"] = candidate["checkpoint"].clone();
    if evidence
        .pointer("/baseline/checkpoint_sha256")
        .and_then(Value::as_str)
        != Some(get_str(request, "accepted_ancestor")?)
    {
        bail!("gate baseline differs from the frozen accepted ancestor");
    }
    let destination =
        watermark_dir(root, get_i64(request, "watermark")?).join("acceptance-evidence.json");
    atomic_json(&destination, &evidence)?;
    let mut result = evidence;
    result["artifact_path"] = Value::String(destination.to_string_lossy().into_owned());
    result["artifact_sha256"] = Value::String(sha256_file(&destination)?);
    Ok(result)
}

/// Conditional advantage = conditional reward head pairwise accuracy MINUS the
/// detached genome-only ablation control. <= 0 means the head is not using the
/// dataset embedding at all: it is a genome prior with noise on top. Key order
/// mirrors `test_exact_accept.CONDITIONAL_ADVANTAGE_KEYS`.
const CONDITIONAL_ADVANTAGE_KEYS: [&str; 3] = [
    "val_conditional_advantage",
    "conditional_advantage",
    "selected_phase_b_conditional_advantage",
];
/// Margins in accuracy points, matching the macro_pairwise_lift convention.
const CONDITIONAL_INTERNAL_MARGIN: f64 = 0.005;
const CONDITIONAL_MILESTONE_MARGIN: f64 = 0.02;

/// The independent promotion branches, in the same order as
/// `selfplay_expansion::TASKS`. A generation that trades one against another
/// can never clear a single global gate — which is the observed state (gen-1
/// `ff95a336`: binary +0.0421, regression -0.0161, tier "rejected", and the
/// original anchor still accepted). `task_verdicts` records the decision each
/// branch reaches on its OWN evidence, so the branches can advance
/// independently. It never widens the global verdict: `pass`, `promotion_tier`
/// and `reasons` are computed exactly as before.
///
/// The NEW branches (timeseries, multiclass) are ADVISORY until anchored: see
/// `task_verdict_is_gating`. Keep in lockstep with
/// `test_exact_accept.TASK_BRANCHES` (same members, same order).
const TASK_BRANCHES: [&str; 4] = ["regression", "binary", "timeseries", "multiclass"];
/// Per-task ports of the global lift margins. The global tier reads the macro
/// average; a task verdict reads that task's own pairwise lift.
const TASK_INTERNAL_LIFT_MARGIN: f64 = 0.002;
const TASK_MILESTONE_LIFT_MARGIN: f64 = 0.01;
/// Shared reasons a per-task verdict supersedes when the evidence carries the
/// task-scoped measurement (`failure_rate_by_task`,
/// `equal_budget_result_by_task`). Absent those fields the aggregate reason is
/// inherited unchanged — an unmeasured branch is fail-closed, never waived.
const FAILURE_RATE_REASON: &str = "candidate_failure_rate_regressed";
const EQUAL_BUDGET_REASON: &str = "candidate_did_not_win_equal_budget_proposal_gate";

/// Port of `test_exact_accept.load_accepted_by_task`:
/// `training/accepted-by-task.json` if present, else None. A missing file
/// means no per-task anchors exist yet (every new branch stays advisory);
/// malformed JSON is surfaced, never silently waived.
fn load_accepted_by_task(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(read_json(path)?))
}

/// Port of `test_exact_accept.task_verdict_is_gating`: true when `task`'s
/// verdict participates in gating. The legacy branches always gate —
/// bit-identical to the two-branch gate. A NEW branch gates only once
/// `accepted-by-task.json` carries an accepted anchor for it; presence of the
/// task entry IS the anchor, the same rule
/// `jope_task_learning_controller.rs::accepted_pointers` applies (new tasks
/// are never bootstrapped into that file, so presence implies a promoted
/// per-task anchor).
fn task_verdict_is_gating(task: &str, accepted_by_task: Option<&Value>) -> bool {
    if matches!(task, "regression" | "binary") {
        return true;
    }
    accepted_by_task
        .and_then(|value| value.get("tasks"))
        .and_then(|tasks| tasks.get(task))
        .is_some_and(Value::is_object)
}

/// Port of `test_exact_accept.conditional_advantage`: first *present*, finite,
/// numeric reading wins; a present-but-unreadable value is terminal. `None` is
/// never treated as zero — an unmeasured checkpoint is fail-closed.
fn conditional_advantage(node: &Value) -> Option<f64> {
    for key in CONDITIONAL_ADVANTAGE_KEYS {
        if let Some(value) = node.get(key) {
            return value.as_f64().filter(|value| value.is_finite());
        }
    }
    None
}

fn optional_number(value: Option<f64>) -> Value {
    value
        .and_then(serde_json::Number::from_f64)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// `node[field][task]` as a finite f64, or None. Used for the optional
/// task-scoped measurements; a malformed or non-finite reading reads as absent
/// so the per-task verdict falls back to the aggregate reason.
fn task_number(node: &Value, field: &str, task: &str) -> Option<f64> {
    node.get(field)
        .and_then(|value| value.get(task))
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

/// Port of `test_exact_accept.task_verdicts`.
///
/// A branch verdict is the global verdict with (a) the OTHER branches' pairwise
/// regressions dropped, (b) the macro lift replaced by this branch's own lift,
/// and (c) any aggregate reason for which a task-scoped measurement exists
/// replaced by that measurement's own reason. Everything else — replay
/// verification, conditionality, proposal validity, latency — is a property of
/// the checkpoint, not of a branch, and is inherited verbatim.
///
/// A branch in `advisory_tasks` is computed identically but marked
/// `"advisory": true`: the caller guarantees such a branch contributed nothing
/// to the global verdict, and consumers must not gate on it.
#[allow(clippy::too_many_arguments)]
fn task_verdicts(
    baseline: &Value,
    candidate: &Value,
    shared_reasons: &[String],
    task_own_reasons: &BTreeMap<&str, Vec<String>>,
    task_deltas: &BTreeMap<&str, f64>,
    shared_internal_extra: &[String],
    shared_milestone_extra: &[String],
    aggregate_equal_budget_lost: bool,
    advisory_tasks: &BTreeSet<&str>,
) -> Map<String, Value> {
    let mut verdicts = Map::new();
    for task in TASK_BRANCHES {
        let delta = task_deltas.get(task).copied().unwrap_or(f64::NAN);
        let own = task_own_reasons.get(task).cloned().unwrap_or_default();

        // Task-scoped failure rate, when measured, supersedes the aggregate.
        let mut superseded: BTreeSet<&str> = BTreeSet::new();
        let mut task_specific: Vec<String> = Vec::new();
        let mut failure_delta = Value::Null;
        if let (Some(candidate_failure), Some(baseline_failure)) = (
            task_number(candidate, "failure_rate_by_task", task),
            task_number(baseline, "failure_rate_by_task", task),
        ) {
            superseded.insert(FAILURE_RATE_REASON);
            let delta = candidate_failure - baseline_failure;
            failure_delta = optional_number(Some(delta));
            if delta > 0.0025 + 1e-12 {
                task_specific.push(format!("{task}_failure_rate_regressed"));
            }
        }
        // Task-scoped equal-budget head-to-head, when measured, supersedes the
        // aggregate. Only meaningful while the aggregate reason is present:
        // when the candidate already won overall there is nothing to supersede.
        let mut equal_budget = Value::Null;
        if let (Some(candidate_equal), Some(baseline_equal)) = (
            task_number(candidate, "equal_budget_result_by_task", task),
            task_number(baseline, "equal_budget_result_by_task", task),
        ) {
            superseded.insert(EQUAL_BUDGET_REASON);
            equal_budget = json!({"candidate": candidate_equal, "baseline": baseline_equal});
            if candidate_equal <= baseline_equal + 1e-12 {
                task_specific.push(format!("{task}_did_not_win_equal_budget_proposal_gate"));
            }
        } else if aggregate_equal_budget_lost {
            // no task-scoped reading: the aggregate reason stays in `shared`.
        }

        let inherited: Vec<String> = shared_reasons
            .iter()
            .filter(|reason| !superseded.contains(reason.as_str()))
            .cloned()
            .collect();

        let mut internal_reasons: Vec<String> = own.clone();
        internal_reasons.extend(task_specific.iter().cloned());
        internal_reasons.extend(inherited.iter().cloned());
        let mut milestone_reasons = internal_reasons.clone();
        if !matches!(
            delta.partial_cmp(&(TASK_INTERNAL_LIFT_MARGIN - 1e-12)),
            Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater)
        ) {
            internal_reasons.push(format!("{task}_pairwise_lift_below_point_two"));
        }
        if !matches!(
            delta.partial_cmp(&(TASK_MILESTONE_LIFT_MARGIN - 1e-12)),
            Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater)
        ) {
            milestone_reasons.push(format!("{task}_pairwise_lift_below_one_point"));
        }
        internal_reasons.extend(shared_internal_extra.iter().cloned());
        milestone_reasons.extend(shared_milestone_extra.iter().cloned());

        let internal_pass = internal_reasons.is_empty();
        let milestone_pass = milestone_reasons.is_empty();
        let tier = if milestone_pass {
            "milestone"
        } else if internal_pass {
            "internal"
        } else {
            "rejected"
        };
        // True when nothing this branch controls is blocking it: every residual
        // reason is a property of the checkpoint as a whole. That is the signal
        // the operator asked for — "binary was held back only by task-agnostic
        // evidence" is a different fact from "binary lost on its own merits".
        let branch_owned: BTreeSet<&String> = own
            .iter()
            .chain(task_specific.iter())
            .chain(
                internal_reasons
                    .iter()
                    .filter(|reason| reason.starts_with(&format!("{task}_"))),
            )
            .collect();
        verdicts.insert(
            task.to_owned(),
            json!({
                "advisory": advisory_tasks.contains(task),
                "pass": internal_pass,
                "internal_learning_pass": internal_pass,
                "milestone_pass": milestone_pass,
                "promotion_tier": tier,
                "reasons": if internal_pass { Vec::new() } else { internal_reasons.clone() },
                "milestone_reasons": milestone_reasons,
                "pairwise_lift": optional_number(
                    Some(delta).filter(|value| value.is_finite()),
                ),
                "candidate_pairwise_accuracy": candidate
                    .get(format!("{task}_pairwise_accuracy"))
                    .cloned()
                    .unwrap_or(Value::Null),
                "baseline_pairwise_accuracy": baseline
                    .get(format!("{task}_pairwise_accuracy"))
                    .cloned()
                    .unwrap_or(Value::Null),
                "failure_rate_delta": failure_delta,
                "equal_budget_result": equal_budget,
                "blocked_only_by_shared_evidence": !internal_pass && branch_owned.is_empty(),
                "blocking_reasons_owned_by_this_task": branch_owned
                    .iter()
                    .map(|reason| Value::String((*reason).clone()))
                    .collect::<Vec<_>>(),
            }),
        );
    }
    verdicts
}

/// Port of `test_exact_accept.evaluate_gate`.
fn evaluate_gate(evidence: &Value, accepted_by_task: Option<&Value>) -> Result<Value> {
    let baseline = &evidence["baseline"];
    let candidate = &evidence["candidate"];
    // Advisory-until-anchor: a branch with no accepted anchor is recorded but
    // never gates — its evidence fields are optional and its reasons stay out
    // of the global verdict entirely.
    let advisory_tasks: BTreeSet<&str> = TASK_BRANCHES
        .into_iter()
        .filter(|task| !task_verdict_is_gating(task, accepted_by_task))
        .collect();
    let macro_lift = get_f64(candidate, "macro_pairwise_accuracy")?
        - get_f64(baseline, "macro_pairwise_accuracy")?;
    let mut common_reasons: Vec<String> = Vec::new();
    let mut task_deltas = Map::new();
    // Branch bookkeeping for `task_verdicts`. These reasons still enter
    // `common_reasons` in the original order, so the global verdict is
    // bit-identical to the pre-change gate.
    let mut task_own_reasons: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    let mut task_delta_values: BTreeMap<&str, f64> = BTreeMap::new();
    for task in TASK_BRANCHES {
        let key = format!("{task}_pairwise_accuracy");
        if advisory_tasks.contains(task) {
            // Optional evidence; a regression here reaches only this branch's
            // own advisory verdict, never the global reasons.
            let delta = match (
                candidate
                    .get(&key)
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite()),
                baseline
                    .get(&key)
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite()),
            ) {
                (Some(candidate_value), Some(baseline_value)) => {
                    Some(candidate_value - baseline_value).filter(|value| value.is_finite())
                }
                _ => None,
            };
            if let Some(value) = delta {
                task_delta_values.insert(task, value);
                if value < -0.005 - 1e-12 {
                    task_own_reasons
                        .entry(task)
                        .or_default()
                        .push(format!("{task}_regressed_more_than_half_point"));
                }
            }
            task_deltas.insert(task.to_owned(), optional_number(delta));
            continue;
        }
        let delta = get_f64(candidate, &key)? - get_f64(baseline, &key)?;
        task_delta_values.insert(task, delta);
        task_deltas.insert(
            task.to_owned(),
            serde_json::Number::from_f64(delta)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
        if delta < -0.005 - 1e-12 {
            let reason = format!("{task}_regressed_more_than_half_point");
            task_own_reasons
                .entry(task)
                .or_default()
                .push(reason.clone());
            common_reasons.push(reason);
        }
    }
    if get_f64(evidence, "proposal_validity")? < 0.99 {
        common_reasons.push("proposal_validity_below_99_percent".into());
    }
    if evidence
        .get("verified_checkpoint_origin_strict_wins")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        < 1
    {
        common_reasons.push("no_verified_checkpoint_origin_strict_test_improvement".into());
    }
    let candidate_equal_budget = evidence
        .get("candidate_equal_budget_result")
        .and_then(Value::as_f64)
        .or_else(|| {
            candidate
                .get("equal_budget_proposal_result")
                .and_then(Value::as_f64)
        })
        .unwrap_or(f64::NEG_INFINITY);
    let baseline_equal_budget = evidence
        .get("baseline_equal_budget_result")
        .and_then(Value::as_f64)
        .or_else(|| {
            baseline
                .get("equal_budget_proposal_result")
                .and_then(Value::as_f64)
        })
        .unwrap_or(f64::INFINITY);
    if candidate_equal_budget <= baseline_equal_budget + 1e-12 {
        common_reasons.push("candidate_did_not_win_equal_budget_proposal_gate".into());
    }
    let candidate_failure = candidate
        .get("failure_rate")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let baseline_failure = baseline
        .get("failure_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let failure_delta = candidate_failure - baseline_failure;
    if failure_delta > 0.0025 + 1e-12 {
        common_reasons.push("candidate_failure_rate_regressed".into());
    }
    let candidate_latency = candidate
        .get("p95_proposal_latency_ms")
        .and_then(Value::as_f64)
        .unwrap_or(f64::INFINITY);
    let baseline_latency = baseline
        .get("p95_proposal_latency_ms")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let latency_ratio = if baseline_latency <= 0.0 {
        f64::INFINITY
    } else {
        candidate_latency / baseline_latency
    };
    let exception = evidence
        .get("latency_exception")
        .cloned()
        .unwrap_or(json!({}));
    let exception_valid = exception.get("approved") == Some(&Value::Bool(true))
        && exception
            .get("accuracy_gain")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            > exception
                .get("latency_penalty")
                .and_then(Value::as_f64)
                .unwrap_or(f64::INFINITY)
        && !exception
            .get("rationale")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .is_empty();
    if latency_ratio > 1.25 + 1e-12 && !exception_valid {
        common_reasons.push("candidate_p95_latency_exceeds_limit".into());
    }
    let replay = evidence
        .get("replay_verification")
        .cloned()
        .unwrap_or(json!({}));
    for check in [
        "predictions",
        "scores",
        "evaluator_identity",
        "dataset_hashes",
        "candidate_origin",
    ] {
        if replay.get(check) != Some(&Value::Bool(true)) {
            common_reasons.push(format!("replay_verification_failed:{check}"));
        }
    }
    // Conditionality gate. A checkpoint whose conditional reward head cannot beat
    // its own genome-only ablation is not conditioned on the dataset, so it can
    // never be promoted — no macro lift buys past this, at either tier.
    let candidate_conditional = conditional_advantage(candidate);
    let baseline_conditional = conditional_advantage(baseline);
    let conditional_lift = match (candidate_conditional, baseline_conditional) {
        (Some(candidate_value), Some(baseline_value)) => Some(candidate_value - baseline_value),
        _ => None,
    };
    if candidate_conditional.is_none() {
        common_reasons.push("candidate_conditional_advantage_evidence_missing".into());
    }
    if baseline_conditional.is_none() {
        common_reasons.push("baseline_conditional_advantage_evidence_missing".into());
    }
    if candidate_conditional.is_some_and(|value| value <= 1e-12) {
        common_reasons.push("candidate_policy_is_not_conditional".into());
    }
    let mut internal_reasons = common_reasons.clone();
    let mut milestone_reasons = common_reasons.clone();
    if macro_lift < 0.002 - 1e-12 {
        internal_reasons.push("macro_pairwise_lift_below_point_two".into());
    }
    if macro_lift < 0.01 - 1e-12 {
        milestone_reasons.push("macro_pairwise_lift_below_one_point".into());
    }
    if candidate_conditional.is_some_and(|value| value < CONDITIONAL_INTERNAL_MARGIN - 1e-12) {
        internal_reasons.push("conditional_advantage_below_half_point".into());
    }
    if conditional_lift.is_some_and(|value| value < CONDITIONAL_INTERNAL_MARGIN - 1e-12) {
        internal_reasons.push("conditional_advantage_lift_below_half_point".into());
    }
    if candidate_conditional.is_some_and(|value| value < CONDITIONAL_MILESTONE_MARGIN - 1e-12) {
        milestone_reasons.push("conditional_advantage_below_two_points".into());
    }
    if conditional_lift.is_some_and(|value| value < CONDITIONAL_MILESTONE_MARGIN - 1e-12) {
        milestone_reasons.push("conditional_advantage_lift_below_two_points".into());
    }
    let internal_pass = internal_reasons.is_empty();
    let milestone_pass = milestone_reasons.is_empty();
    let reasons: Vec<String> = if internal_pass {
        Vec::new()
    } else {
        internal_reasons
    };
    let tier = if milestone_pass {
        "milestone"
    } else if internal_pass {
        "internal"
    } else {
        "rejected"
    };
    if !latency_ratio.is_finite() {
        bail!("non-finite latency ratio cannot be recorded in gate output");
    }
    // Per-task verdicts. Derived from exactly the reasons the global verdict
    // already computed, minus the branch attribution: nothing here can make the
    // global gate laxer, and everything here is additive in the artifact.
    let branch_owned_reasons: BTreeSet<&String> = task_own_reasons.values().flatten().collect();
    let shared_reasons: Vec<String> = common_reasons
        .iter()
        .filter(|reason| !branch_owned_reasons.contains(reason))
        .cloned()
        .collect();
    let mut shared_internal_extra: Vec<String> = Vec::new();
    let mut shared_milestone_extra: Vec<String> = Vec::new();
    if candidate_conditional.is_some_and(|value| value < CONDITIONAL_INTERNAL_MARGIN - 1e-12) {
        shared_internal_extra.push("conditional_advantage_below_half_point".into());
    }
    if conditional_lift.is_some_and(|value| value < CONDITIONAL_INTERNAL_MARGIN - 1e-12) {
        shared_internal_extra.push("conditional_advantage_lift_below_half_point".into());
    }
    if candidate_conditional.is_some_and(|value| value < CONDITIONAL_MILESTONE_MARGIN - 1e-12) {
        shared_milestone_extra.push("conditional_advantage_below_two_points".into());
    }
    if conditional_lift.is_some_and(|value| value < CONDITIONAL_MILESTONE_MARGIN - 1e-12) {
        shared_milestone_extra.push("conditional_advantage_lift_below_two_points".into());
    }
    let verdicts = task_verdicts(
        baseline,
        candidate,
        &shared_reasons,
        &task_own_reasons,
        &task_delta_values,
        &shared_internal_extra,
        &shared_milestone_extra,
        common_reasons
            .iter()
            .any(|reason| reason == EQUAL_BUDGET_REASON),
        &advisory_tasks,
    );
    let passing: Vec<Value> = TASK_BRANCHES
        .iter()
        .filter(|task| verdicts[**task]["pass"] == Value::Bool(true))
        .map(|task| Value::String((*task).to_owned()))
        .collect();
    Ok(json!({
        "schema": "JopeSelfplayCheckpointGate.v3",
        "pass": internal_pass,
        "internal_learning_pass": internal_pass,
        "milestone_pass": milestone_pass,
        "promotion_tier": tier,
        "reasons": reasons,
        "milestone_reasons": milestone_reasons,
        "macro_pairwise_lift": macro_lift,
        "candidate_conditional_advantage": optional_number(candidate_conditional),
        "baseline_conditional_advantage": optional_number(baseline_conditional),
        "conditional_advantage_lift": optional_number(conditional_lift),
        "task_pairwise_deltas": task_deltas,
        "failure_rate_delta": failure_delta,
        "p95_latency_ratio": latency_ratio,
        "latency_exception_applied": latency_ratio > 1.25 && exception_valid,
        "model_lift_claim_publishable": false,
        "promotion_scope": "internal-selected-test-flywheel-only",
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "accepted_ancestor_sha256": baseline["checkpoint_sha256"],
        "candidate_equal_budget_result": candidate_equal_budget,
        "baseline_equal_budget_result": baseline_equal_budget,
        "task_verdicts": Value::Object(verdicts),
        "task_verdict_schema": "JopeSelfplayCheckpointGateTaskVerdict.v1",
        "tasks_passing": passing,
    }))
}

// ---------------------------------------------------------------------------
// Per-task accepted pointers
// ---------------------------------------------------------------------------

/// Per-task accepted checkpoints. Deliberately a SEPARATE, additive file:
/// `training/accepted.json` and `training/learning.json` stay the ONE global
/// warm-start lineage pointer read by `jope_selfplay_freshness.rs:384`,
/// `jope_selfplay_supply.rs:542` (Codex-owned), `jope_selfplay_trainer_watchdog.rs:1055`,
/// `selfplay_runtime.rs:83` and `promotion.rs:accepted_checkpoint`, and they
/// alone define the freshness epoch. Nothing in this file participates in epoch
/// identity, so advancing a branch never retires a queued job.
///
/// The schema string is `JopeSelfplayAcceptedByTask.v1` — the value already on disk
/// and already written by `jope_task_learning_controller.rs` and `jope_selfplay_tui.rs`.
/// It was previously `AcceptedByTask.v2`, which is the *shadow* namespace's constant
/// (`selfplay_shadow/contracts.rs:7`) for a different artifact
/// (`accepted/by-task.v2.json`) with a different shape. Because
/// `advance_task_accepted_pointers` rewrites `document["schema"]` unconditionally — even
/// on a cycle where every branch is unchanged — the first promotion run would have
/// rewritten the live pointer to a string that four readers reject outright:
/// `jope_task_learning_controller.rs:1055`, `jope_causal_evidence.rs:302`,
/// `scripts/test_exact_train.py:843`, `train_only_study.rs:1437`. Two of those run as
/// live services against this root, so the break would have been immediate and fleet-wide.
const TASK_ACCEPTED_SCHEMA: &str = "JopeSelfplayAcceptedByTask.v1";

fn task_accepted_pointer(root: &Path) -> PathBuf {
    root.join("training/accepted-by-task.json")
}

/// Content-addressed copy into the immutable store, hash-verified both sides.
/// Same contract as the copy inside `promote`, factored so a per-task advance
/// cannot leave a pointer aimed at a scheduler run directory the GC will reap.
fn publish_immutable_checkpoint(
    immutable_root: &Path,
    checkpoint: &Path,
    checkpoint_sha: &str,
) -> Result<PathBuf> {
    let destination = immutable_root
        .join("checkpoints")
        .join(format!("{checkpoint_sha}.model.pt"));
    fs::create_dir_all(destination.parent().unwrap())?;
    if destination.exists() {
        if sha256_file(&destination)? != checkpoint_sha {
            bail!("immutable checkpoint path contains different bytes");
        }
        return Ok(destination);
    }
    if sha256_file(checkpoint)? != checkpoint_sha {
        bail!("checkpoint content hash mismatch before immutable publish");
    }
    let temporary = destination.with_file_name(format!(
        ".{}.tmp-{}",
        destination.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    fs::copy(checkpoint, &temporary)?;
    if sha256_file(&temporary)? != checkpoint_sha {
        let _ = fs::remove_file(&temporary);
        bail!("checkpoint changed while copying to immutable storage");
    }
    fs::rename(&temporary, &destination)?;
    Ok(destination)
}

/// Advance `training/accepted-by-task.json` from one gate result.
///
/// A branch advances only when its OWN verdict passed. A branch that did not
/// pass keeps whatever it already had; a branch that has never been recorded is
/// bootstrapped to the global accepted checkpoint so every reader always
/// resolves to something content-verified. The compare-and-swap is per branch:
/// the entry must currently name the gate's accepted ancestor (or be a
/// bootstrap), which is the same ancestor guard `promote` applies globally.
fn advance_task_accepted_pointers(
    root: &Path,
    evidence: &Value,
    gate: &Value,
    global_path: &Path,
    global_sha: &str,
) -> Result<Value> {
    let verdicts = gate
        .get("task_verdicts")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("gate result carries no task_verdicts"))?;
    let ancestor = get_str(gate, "accepted_ancestor_sha256")?.to_owned();
    let candidate_sha = get_str(gate, "checkpoint_sha256")?.to_owned();
    let candidate_path = evidence
        .pointer("/candidate/checkpoint_path")
        .and_then(Value::as_str)
        .map(PathBuf::from);
    let watermark = evidence
        .get("corpus_watermark")
        .cloned()
        .unwrap_or(Value::Null);
    let evidence_sha256 = evidence
        .get("artifact_sha256")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| sha256_bytes(&serde_json::to_vec(evidence).unwrap_or_default()));
    let immutable = root.join("training/immutable");
    let pointer_path = task_accepted_pointer(root);
    fs::create_dir_all(pointer_path.parent().unwrap())?;
    let lock_path = pointer_path.with_extension("json.lock");
    let lock = fs::File::create(&lock_path)?;
    flock_exclusive(&lock, true)?;

    let mut document = if pointer_path.is_file() {
        read_json(&pointer_path)?
    } else {
        json!({"schema": TASK_ACCEPTED_SCHEMA, "tasks": {}})
    };
    if document.get("tasks").and_then(Value::as_object).is_none() {
        document["tasks"] = json!({});
    }
    let mut advanced: Vec<String> = Vec::new();
    let mut unchanged: Vec<String> = Vec::new();
    let mut refused: Map<String, Value> = Map::new();
    for task in TASK_BRANCHES {
        let existing = document
            .pointer(&format!("/tasks/{task}"))
            .cloned()
            .filter(|value| value.is_object());
        let passed = verdicts
            .get(task)
            .and_then(|verdict| verdict.get("pass"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !passed {
            if existing.is_none() {
                // Advisory-until-anchor: a NEW task is never bootstrapped into
                // this file — presence of its entry IS the anchor that turns
                // its verdict gating (`task_verdict_is_gating`, and the same
                // presence rule in the task learning controller's
                // `accepted_pointers`). Only a genuinely passing verdict below
                // may create it.
                if !matches!(task, "regression" | "binary") {
                    continue;
                }
                document["tasks"][task] = json!({
                    "task": task,
                    "checkpoint": global_path.to_string_lossy(),
                    "checkpoint_sha256": global_sha,
                    "generation": 0,
                    "parent_checkpoint_sha256": Value::Null,
                    "evidence_sha256": Value::Null,
                    "proposal_sha256": Value::Null,
                    "corpus_watermark": watermark,
                    "rollback": Value::Null,
                    "source": "global_accepted_bootstrap",
                    "promotion_tier": Value::Null,
                    "updated_at_ms": now_ms(),
                });
            }
            unchanged.push(task.to_owned());
            continue;
        }
        let current = existing
            .as_ref()
            .and_then(|value| value.get("checkpoint_sha256"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if current == candidate_sha {
            unchanged.push(task.to_owned());
            continue;
        }
        if !current.is_empty() && current != ancestor {
            refused.insert(
                task.to_owned(),
                json!({"reason": "task ancestor changed before pointer advance",
                       "current": current, "expected": ancestor}),
            );
            continue;
        }
        let Some(source) = candidate_path.as_ref() else {
            refused.insert(
                task.to_owned(),
                json!({"reason": "evidence has no candidate checkpoint path"}),
            );
            continue;
        };
        let published = publish_immutable_checkpoint(&immutable, source, &candidate_sha)?;
        let generation = existing
            .as_ref()
            .and_then(|value| value.get("generation"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
            + 1;
        let proposal_sha256 = evidence
            .get("proposal_sets")
            .and_then(Value::as_array)
            .and_then(|sets| {
                sets.iter()
                    .find(|set| set.get("task").and_then(Value::as_str) == Some(task))
            })
            .and_then(|set| set.get("proposals_sha256"))
            .cloned()
            .unwrap_or(Value::Null);
        let rollback = existing.clone().unwrap_or_else(|| {
            json!({
                "checkpoint": global_path.to_string_lossy(),
                "checkpoint_sha256": global_sha,
                "generation": 0,
            })
        });
        document["tasks"][task] = json!({
            "task": task,
            "checkpoint": published.to_string_lossy(),
            "checkpoint_sha256": candidate_sha,
            "generation": generation,
            "parent_checkpoint_sha256": ancestor,
            "evidence_sha256": evidence_sha256,
            "proposal_sha256": proposal_sha256,
            "corpus_watermark": watermark,
            "rollback": rollback,
            "promotion_tier": verdicts[task]["promotion_tier"],
            "pairwise_lift": verdicts[task]["pairwise_lift"],
            "source": "per_task_gate",
            "updated_at_ms": now_ms(),
        });
        advanced.push(task.to_owned());
    }
    document["schema"] = json!(TASK_ACCEPTED_SCHEMA);
    document["global_pointer"] = json!(root.join("training/accepted.json").to_string_lossy());
    document["global_checkpoint_sha256"] = json!(global_sha);
    document["updated_at_ms"] = json!(now_ms());
    atomic_json(&pointer_path, &document)?;
    Ok(json!({
        "pointer": pointer_path.to_string_lossy(),
        "pointer_sha256": sha256_file(&pointer_path)?,
        "advanced": advanced,
        "unchanged": unchanged,
        "refused": Value::Object(refused),
    }))
}

fn handle_run_checkpoint_gate(request: &Value, root: &Path) -> Result<Value> {
    let evidence = request
        .pointer("/artifacts/build_acceptance_evidence")
        .ok_or_else(|| anyhow!("missing build_acceptance_evidence artifact"))?;
    // Advisory-until-anchor: new-task verdicts gate only once
    // accepted-by-task.json carries their anchor.
    let accepted_by_task = load_accepted_by_task(&task_accepted_pointer(root))?;
    let result = evaluate_gate(evidence, accepted_by_task.as_ref())?;
    let destination =
        watermark_dir(root, get_i64(request, "watermark")?).join("checkpoint-gate.json");
    atomic_json(&destination, &result)?;
    let mut out = result;
    // Per-task pointers are a RECORD, not a precondition: this phase must keep
    // producing its artifact even if the branch pointer cannot be written, or a
    // filesystem problem in an additive lane would stall the whole tournament.
    out["task_accepted_pointers"] = match accepted_checkpoint()
        .and_then(|(path, sha)| advance_task_accepted_pointers(root, evidence, &out, &path, &sha))
    {
        Ok(value) => value,
        Err(error) => json!({"error": error.to_string()}),
    };
    out["artifact_path"] = Value::String(destination.to_string_lossy().into_owned());
    out["artifact_sha256"] = Value::String(sha256_file(&destination)?);
    Ok(out)
}

fn flock_exclusive(file: &fs::File, blocking: bool) -> Result<()> {
    let mut operation = libc::LOCK_EX;
    if !blocking {
        operation |= libc::LOCK_NB;
    }
    let status = unsafe { libc::flock(file.as_raw_fd(), operation) };
    if status != 0 {
        bail!("flock failed: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

/// Port of `test_exact_accept.promote`.
fn promote(
    evidence_path: &Path,
    gate_path: &Path,
    accepted_pointer: &Path,
    immutable_root: &Path,
    enforce_pointer_ancestor: bool,
) -> Result<Value> {
    let evidence = read_json(evidence_path)?;
    // The per-task anchor file lives beside the global accepted pointer; it
    // decides which NEW branches gate (advisory-until-anchor).
    let accepted_by_task =
        load_accepted_by_task(&accepted_pointer.with_file_name("accepted-by-task.json"))?;
    let result = evaluate_gate(&evidence, accepted_by_task.as_ref())?;
    atomic_json(gate_path, &result)?;
    if result.get("pass") != Some(&Value::Bool(true)) {
        return Ok(result);
    }
    let checkpoint_sha = get_str(&result, "checkpoint_sha256")?.to_owned();
    let checkpoint = PathBuf::from(
        evidence
            .pointer("/candidate/checkpoint_path")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("evidence has no candidate checkpoint path"))?,
    );
    if sha256_file(&checkpoint)? != checkpoint_sha {
        bail!("accepted checkpoint content hash mismatch");
    }
    let immutable_checkpoint = immutable_root
        .join("checkpoints")
        .join(format!("{checkpoint_sha}.model.pt"));
    fs::create_dir_all(immutable_checkpoint.parent().unwrap())?;
    if !immutable_checkpoint.exists() {
        let temporary = immutable_checkpoint.with_file_name(format!(
            ".{}.tmp-{}",
            immutable_checkpoint.file_name().unwrap().to_string_lossy(),
            std::process::id()
        ));
        fs::copy(&checkpoint, &temporary)?;
        if sha256_file(&temporary)? != checkpoint_sha {
            let _ = fs::remove_file(&temporary);
            bail!("checkpoint changed while copying to immutable storage");
        }
        fs::rename(&temporary, &immutable_checkpoint)?;
    } else if sha256_file(&immutable_checkpoint)? != checkpoint_sha {
        bail!("immutable checkpoint path contains different bytes");
    }
    let evidence_hash = sha256_file(evidence_path)?;
    let immutable_evidence = immutable_root
        .join("gates")
        .join(format!("{evidence_hash}.evidence.json"));
    fs::create_dir_all(immutable_evidence.parent().unwrap())?;
    if !immutable_evidence.exists() {
        fs::copy(evidence_path, &immutable_evidence)?;
    }
    if sha256_file(&immutable_evidence)? != evidence_hash {
        bail!("immutable evidence copy hash mismatch");
    }
    let gate_hash = sha256_file(gate_path)?;
    let immutable_gate = immutable_root
        .join("gates")
        .join(format!("{gate_hash}.gate.json"));
    if !immutable_gate.exists() {
        fs::copy(gate_path, &immutable_gate)?;
    }
    if sha256_file(&immutable_gate)? != gate_hash {
        bail!("immutable gate copy hash mismatch");
    }
    let lock_path = accepted_pointer.with_extension(format!(
        "{}.lock",
        accepted_pointer
            .extension()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    fs::create_dir_all(lock_path.parent().unwrap())?;
    let lock = fs::File::create(&lock_path)?;
    flock_exclusive(&lock, true)?;
    if accepted_pointer.is_file() {
        let current = read_json(accepted_pointer)?;
        let current_sha = get_str(&current, "checkpoint_sha256")?;
        if current_sha == checkpoint_sha {
            return Ok(result);
        }
        if enforce_pointer_ancestor && current_sha != get_str(&result, "accepted_ancestor_sha256")?
        {
            bail!("accepted ancestor changed before pointer promotion");
        }
    }
    atomic_json(
        accepted_pointer,
        &json!({
            "schema": "JopeSelfplayAcceptedCheckpoint.v2",
            "checkpoint": immutable_checkpoint.to_string_lossy(),
            "checkpoint_sha256": checkpoint_sha,
            "accepted_ancestor_sha256": result["accepted_ancestor_sha256"],
            "gate": immutable_gate.to_string_lossy(),
            "gate_sha256": gate_hash,
            "evidence": immutable_evidence.to_string_lossy(),
            "evidence_sha256": evidence_hash,
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": "test-exact-reused",
            "promotion_scope": "internal-selected-test-flywheel-only",
        }),
    )?;
    Ok(result)
}

fn handle_advance_accepted_pointer(request: &Value, root: &Path) -> Result<Value> {
    let evidence = PathBuf::from(get_str(
        request
            .pointer("/artifacts/build_acceptance_evidence")
            .ok_or_else(|| anyhow!("missing build_acceptance_evidence artifact"))?,
        "artifact_path",
    )?);
    let gate = PathBuf::from(get_str(
        request
            .pointer("/artifacts/run_checkpoint_gate")
            .ok_or_else(|| anyhow!("missing run_checkpoint_gate artifact"))?,
        "artifact_path",
    )?);
    let learning = root.join("training/learning.json");
    let immutable = root.join("training/immutable");
    let result = promote(&evidence, &gate, &learning, &immutable, true)?;
    if result.get("pass") != Some(&Value::Bool(true)) {
        bail!("controller attempted to promote a rejected checkpoint");
    }
    // Compatibility pointer for existing consumers; the learning pointer is
    // authoritative and is always advanced first.
    atomic_json(&root.join("training/accepted.json"), &read_json(&learning)?)?;
    let mut milestone = Value::Null;
    if result.get("milestone_pass") == Some(&Value::Bool(true)) {
        let milestone_pointer = root.join("training/milestone.json");
        let milestone_result = promote(&evidence, &gate, &milestone_pointer, &immutable, false)?;
        milestone = json!({
            "advanced": true,
            "checkpoint_sha256": milestone_result["checkpoint_sha256"],
            "pointer": milestone_pointer.to_string_lossy(),
            "pointer_sha256": sha256_file(&milestone_pointer)?,
        });
    }
    Ok(json!({
        "advanced": true,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "promotion_tier": result["promotion_tier"],
        "learning_pointer": learning.to_string_lossy(),
        "learning_pointer_sha256": sha256_file(&learning)?,
        "milestone": milestone,
    }))
}

fn handle_run_sotarena_qualification(request: &Value, root: &Path) -> Result<Value> {
    let scan = request
        .pointer("/artifacts/freeze_winner_scan")
        .ok_or_else(|| anyhow!("missing freeze_winner_scan artifact"))?;
    let ledger_path = PathBuf::from(get_str(scan, "ledger")?);
    if sha256_file(&ledger_path)? != get_str(scan, "ledger_sha256")? {
        bail!("winner scan ledger changed before qualification enqueue");
    }
    let ledger = read_json(&ledger_path)?;
    let mut selected = Vec::new();
    for row in ledger
        .get("datasets")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("winner ledger has no datasets"))?
    {
        if row.get("qualification_state").and_then(Value::as_str) != Some("discovered") {
            continue;
        }
        let candidate = row.get("candidate").filter(|c| !c.is_null());
        selected.push(json!({
            "task": row["task"],
            "dataset_hash": row["dataset_hash"],
            "semantic_sha256": candidate
                .and_then(|c| c.get("semantic_sha256"))
                .cloned()
                .unwrap_or(Value::Null),
            "receipt_sha256": candidate
                .and_then(|c| c.get("receipt_sha256"))
                .cloned()
                .unwrap_or(Value::Null),
            "checkpoint_sha256": candidate
                .and_then(|c| c.get("model_checkpoint_sha256"))
                .cloned()
                .unwrap_or(Value::Null),
            "qualification_state": row["qualification_state"],
            "evidence_lane": "adaptive",
            "report_only": true,
        }));
    }
    let tier = request
        .pointer("/artifacts/run_checkpoint_gate/promotion_tier")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let sealed = if tier == "milestone" {
        milestone_status(
            request
                .pointer("/artifacts/advance_accepted_pointer/checkpoint_sha256")
                .and_then(Value::as_str)
                .unwrap_or(""),
        )?
    } else {
        Value::Null
    };
    let record = json!({
        "schema": "JopeSotarenaQualificationQueue.v1",
        "winner_ledger": ledger_path.to_string_lossy(),
        "winner_ledger_sha256": scan["ledger_sha256"],
        "candidate_checkpoint_sha256": request
            .pointer("/artifacts/advance_accepted_pointer/checkpoint_sha256")
            .cloned()
            .unwrap_or(Value::Null),
        "promotion_tier": tier,
        "adaptive_evidence": true,
        "zero_shot_claim_eligible": false,
        "report_only": true,
        "sealed_audit": sealed,
        "candidates": selected,
    });
    let path = root.join("qualifications").join(format!(
        "watermark-{:09}.json",
        get_i64(request, "watermark")?
    ));
    atomic_json(&path, &record)?;
    Ok(json!({
        "queue": path.to_string_lossy(),
        "queue_sha256": sha256_file(&path)?,
        "enqueued": selected.len(),
        "report_only": true,
    }))
}

/// Fail-closed production adapter for externally produced phase evidence
/// (used for every phase when running against a non-production root).
fn file_handler(name: &str, request: &Value, root: &Path) -> Result<Value> {
    let path = watermark_dir(root, get_i64(request, "watermark")?).join(format!("{name}.json"));
    if !path.is_file() {
        return Err(pending(format!("waiting for {}", path.display())));
    }
    let mut value = read_json(&path)?;
    value["artifact_path"] = Value::String(path.to_string_lossy().into_owned());
    value["artifact_sha256"] = Value::String(sha256_file(&path)?);
    Ok(value)
}

fn dispatch_phase(phase: &str, request: &Value, root: &Path, production: bool) -> Result<Value> {
    if !production {
        return file_handler(phase, request, root);
    }
    if PYTHON_PHASES.contains(&phase) {
        return python_handler(phase, request, root);
    }
    match phase {
        "launch_checkpoint_tournament" => handle_launch_checkpoint_tournament(request, root),
        "fetch_candidate_checkpoints" => handle_fetch_candidate_checkpoints(request, root),
        "generate_gate_proposals" => handle_generate_gate_proposals(request, root),
        "build_acceptance_evidence" => handle_build_acceptance_evidence(request, root),
        "run_checkpoint_gate" => handle_run_checkpoint_gate(request, root),
        "advance_accepted_pointer" => handle_advance_accepted_pointer(request, root),
        "run_sotarena_qualification" => handle_run_sotarena_qualification(request, root),
        other => file_handler(other, request, root),
    }
}

// ---------------------------------------------------------------------------
// Controller state machine
// ---------------------------------------------------------------------------

struct Controller {
    root: PathBuf,
    production: bool,
    journal: ActionJournal,
    state_path: PathBuf,
}

impl Controller {
    fn open(root: &Path) -> Result<Self> {
        Ok(Self {
            root: root.to_path_buf(),
            production: root == Path::new(root_str()),
            journal: ActionJournal::open(root)?,
            state_path: root.join("ops/flywheel-state.json"),
        })
    }

    fn state(&self) -> Result<Value> {
        if self.state_path.is_file() {
            return read_json(&self.state_path);
        }
        Ok(json!({
            "schema": "JopeSelfplayControllerState.v1",
            "last_frozen_watermark": 0,
            "consecutive_rejections": 0,
            "tournament": null,
        }))
    }

    fn cycle(&self, snapshot: &Value) -> Result<Value> {
        let entries = snapshot
            .get("catalog_entries")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("snapshot has no catalog entries"))?;
        let declared: BTreeMap<String, i64> = snapshot
            .get("catalog_counts")
            .and_then(Value::as_object)
            .map(|map| {
                map.iter()
                    .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(-1)))
                    .collect()
            })
            .ok_or_else(|| anyhow!("snapshot has no catalog counts"))?;
        let counts = validate_catalog_counts(entries, &declared)?;
        let mut state = self.state()?;
        let rejections = state
            .get("consecutive_rejections")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        // Legacy unstarted tournaments used three competing recipes.
        let needs_normalization = {
            let tournament = state.get("tournament").filter(|t| !t.is_null());
            match tournament {
                Some(tournament) => {
                    tournament
                        .get("phase_index")
                        .and_then(Value::as_i64)
                        .unwrap_or(0)
                        == 0
                        && tournament
                            .get("artifacts")
                            .and_then(Value::as_object)
                            .map(|a| a.is_empty())
                            .unwrap_or(true)
                        && tournament
                            .get("recipes")
                            .and_then(Value::as_array)
                            .map(|r| r.len() != 1)
                            .unwrap_or(true)
                }
                None => false,
            }
        };
        if needs_normalization {
            let full_consolidation = rejections >= 5;
            let tournament = state.get_mut("tournament").unwrap();
            tournament["recipes"] = json!([if full_consolidation { "a" } else { "c" }]);
            tournament["corpus_mode"] = json!(if full_consolidation {
                "full_consolidation"
            } else {
                "recent_plus_replay"
            });
            state["consecutive_rejections"] = json!(rejections);
            atomic_json(&self.state_path, &state)?;
        }
        if state.get("tournament").map(Value::is_null).unwrap_or(true) {
            let labels = get_i64(snapshot, "valid_labels")?;
            let last_frozen = state
                .get("last_frozen_watermark")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            if labels - last_frozen < LABEL_THRESHOLD {
                let mut catalog_counts = Map::new();
                for (task, count) in counts {
                    catalog_counts.insert(task, json!(count));
                }
                return Ok(json!({
                    "phase": "collecting_labels",
                    "valid_labels": labels,
                    "next_watermark": last_frozen + LABEL_THRESHOLD,
                    "catalog_counts": catalog_counts,
                }));
            }
            let full_consolidation = labels / 500_000 > last_frozen / 500_000 || rejections >= 5;
            let tournament = json!({
                "watermark": labels,
                "accepted_ancestor": snapshot["accepted_ancestor"],
                "catalog_sha256": snapshot["catalog_sha256"],
                "evaluator_sha256": snapshot["evaluator_sha256"],
                "phase_index": 0,
                "artifacts": {},
                "recipes": [if full_consolidation { "a" } else { "c" }],
                "corpus_mode": if full_consolidation {
                    "full_consolidation"
                } else {
                    "recent_plus_replay"
                },
                "seed": DEFAULT_SEED,
            });
            state["last_frozen_watermark"] = json!(labels);
            state["tournament"] = tournament;
            atomic_json(&self.state_path, &state)?;
        }
        let tournament = state["tournament"].clone();
        let phase_index = get_i64(&tournament, "phase_index")? as usize;
        let phase = PHASES[phase_index];
        let mut request = tournament.clone();
        request["action"] = json!(phase);
        let watermark = get_i64(&tournament, "watermark")?;
        let key = action_key(
            phase,
            watermark,
            get_str(&tournament, "accepted_ancestor")?,
            if phase == "launch_checkpoint_tournament" {
                "a+b+c"
            } else {
                "tournament"
            },
            get_i64(&tournament, "seed")?,
            get_str(&tournament, "catalog_sha256")?,
            get_str(&tournament, "evaluator_sha256")?,
        )?;
        let root = self.root.clone();
        let production = self.production;
        let outcome = self.journal.perform(&key, phase, &request, |req| {
            dispatch_phase(phase, req, &root, production)
        });
        let result = match outcome {
            Ok(result) => result,
            Err(error) => {
                if let Some(detail) = as_pending(&error) {
                    return Ok(json!({
                        "phase": phase,
                        "pending": true,
                        "detail": detail.to_string(),
                        "watermark": tournament["watermark"],
                    }));
                }
                return Err(error);
            }
        };
        {
            let tournament = state.get_mut("tournament").unwrap();
            tournament["artifacts"][phase] = result.clone();
            let gate_failed =
                phase == "run_checkpoint_gate" && result.get("pass") == Some(&Value::Bool(false));
            if gate_failed {
                tournament["artifacts"]["advance_accepted_pointer"] = json!({
                    "advanced": false,
                    "checkpoint_sha256": tournament["accepted_ancestor"],
                });
                tournament["phase_index"] = json!(phase_index as i64 + 2);
            } else {
                tournament["phase_index"] = json!(phase_index as i64 + 1);
            }
        }
        let final_index = state
            .pointer("/tournament/phase_index")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        if final_index >= PHASES.len() as i64 {
            let tournament = state["tournament"].clone();
            let gate = tournament
                .pointer("/artifacts/run_checkpoint_gate")
                .cloned()
                .unwrap_or(json!({}));
            let terminal = if gate.get("pass") == Some(&Value::Bool(true)) {
                "accepted"
            } else {
                "rejected"
            };
            let mut completed = tournament.clone();
            completed["terminal"] = json!(terminal);
            completed["completed_at_ms"] = json!(now_ms());
            let completed_path = self
                .root
                .join("training/tournaments")
                .join(format!("watermark-{watermark:09}.json"));
            atomic_json(&completed_path, &completed)?;
            state["tournament"] = Value::Null;
            state["last_terminal"] = json!(terminal);
            state["last_tournament"] = json!(completed_path.to_string_lossy());
            state["consecutive_rejections"] = json!(if terminal == "accepted" {
                0
            } else {
                state
                    .get("consecutive_rejections")
                    .and_then(Value::as_i64)
                    .unwrap_or(0)
                    + 1
            });
        }
        atomic_json(&self.state_path, &state)?;
        Ok(json!({
            "phase": phase,
            "completed": true,
            "result": result,
            "watermark": tournament["watermark"],
        }))
    }
}

fn live_snapshot(root: &Path) -> Result<Value> {
    let pointer_path = root.join("catalog/current.json");
    let authority = validate_catalog_manifest(&pointer_path)?;
    let pointer = read_json(&pointer_path)?;
    let catalog = read_json(Path::new(get_str(&pointer, "catalog")?))?;
    let corpus = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    corpus.busy_timeout(std::time::Duration::from_secs(60))?;
    let labels: i64 =
        corpus.query_row("SELECT COUNT(*) FROM observations", [], |row| row.get(0))?;
    drop(corpus);
    let (_, ancestor) = accepted_checkpoint()?;
    Ok(json!({
        "valid_labels": labels,
        "accepted_ancestor": ancestor,
        "catalog_sha256": pointer["catalog_sha256"],
        "catalog_entries": catalog["entries"],
        "catalog_counts": authority["task_counts"],
        "evaluator_sha256": sha256_file(Path::new(evaluator_str()))?,
    }))
}

#[derive(Parser)]
#[command(name = "jope-selfplay-promotion")]
#[command(about = "Native checkpoint tournament and promotion controller")]
struct Args {
    #[arg(long, default_value = root_str())]
    root: PathBuf,
    /// Seconds to wait before re-polling a phase that reported `pending`.
    /// A pending phase is a healthy wait on external work, not a failure, so it
    /// is polled in-process instead of being turned into a systemd restart.
    #[arg(long, default_value_t = 120)]
    pending_backoff_seconds: u64,
    /// Exit after a single cycle instead of polling a pending phase.
    #[arg(long)]
    once: bool,
    /// One-shot WS-6 lane: recompute the gate from an existing
    /// acceptance-evidence file, print the per-task verdicts, and advance
    /// `training/accepted-by-task.json`. Never runs a phase, never takes the
    /// controller lock, never touches `training/accepted.json`,
    /// `training/learning.json` or the freshness epoch.
    #[arg(long, value_name = "ACCEPTANCE_EVIDENCE_JSON")]
    task_accept: Option<PathBuf>,
    /// With --task-accept: print the verdicts and leave every pointer alone.
    #[arg(long)]
    dry_run: bool,
    /// With --task-accept: resolve the global lineage pointer from this
    /// checkpoint instead of `accepted_checkpoint()` (tests and rehearsals).
    #[arg(long, value_name = "CHECKPOINT")]
    global_checkpoint: Option<PathBuf>,
}

/// `--task-accept`: the per-task promotion decision, decoupled from the
/// tournament state machine. The gate phase for a watermark is journalled
/// exactly-once, so a controller already past `run_checkpoint_gate` will never
/// re-emit the verdict for that watermark; this path recomputes it from the
/// same evidence with the same code and publishes the branch pointers.
fn run_task_accept(args: &Args) -> Result<Value> {
    let evidence_path = args.task_accept.as_ref().expect("--task-accept");
    let evidence = read_json(evidence_path)?;
    let accepted_by_task = load_accepted_by_task(&task_accepted_pointer(&args.root))?;
    let gate = evaluate_gate(&evidence, accepted_by_task.as_ref())?;
    let (global_path, global_sha) = match &args.global_checkpoint {
        Some(path) => (path.clone(), sha256_file(path)?),
        None => accepted_checkpoint()?,
    };
    let mut out = json!({
        "schema": "JopeSelfplayTaskAcceptResult.v1",
        "evidence": evidence_path.to_string_lossy(),
        "evidence_sha256": sha256_file(evidence_path)?,
        "global_checkpoint_sha256": global_sha,
        "global_promotion_tier": gate["promotion_tier"],
        "global_pass": gate["pass"],
        "task_verdicts": gate["task_verdicts"],
        "tasks_passing": gate["tasks_passing"],
        "dry_run": args.dry_run,
        // Full recomputed gate, so a reviewer can diff it against the artifact
        // the controller already wrote and confirm the global verdict did not
        // move when the per-task block was added.
        "gate": gate.clone(),
    });
    if !args.dry_run {
        out["task_accepted_pointers"] = advance_task_accepted_pointers(
            &args.root,
            &evidence,
            &gate,
            &global_path,
            &global_sha,
        )?;
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.task_accept.is_some() {
        println!("{}", py_dumps(&run_task_accept(&args)?, PyStyle::Pretty)?);
        return Ok(());
    }
    let lock_path = args.root.join("ops/controller.lock");
    fs::create_dir_all(lock_path.parent().unwrap())?;
    let lock = fs::File::create(&lock_path)?;
    if flock_exclusive(&lock, false).is_err() {
        eprintln!("self-play controller is already running");
        std::process::exit(1);
    }
    let controller = Controller::open(&args.root)?;
    // A pending phase used to exit(0) and be restarted by systemd, so every
    // poll of a long external wait was counted as a service restart (1,920 of
    // them on the frozen gate alone) and every poll paid a fresh process +
    // snapshot cost. Polling in-process keeps the controller flock held for the
    // whole wait - which is exactly the invariant we want, since no other phase
    // can run while one is pending - and leaves NRestarts meaning "the
    // controller actually stopped", not "the controller looked again". Each
    // poll still journals its own pending detail, so nothing is hidden.
    loop {
        let snapshot = live_snapshot(&args.root)?;
        let result = controller.cycle(&snapshot)?;
        println!("{}", py_dumps(&result, PyStyle::Default)?);
        if args.once || result.get("pending") != Some(&Value::Bool(true)) {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_secs(args.pending_backoff_seconds));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_metric_map_agrees_with_task_authority() {
        use jope_dime_regressor::task::TaskKind;
        for kind in [
            TaskKind::Regression,
            TaskKind::Binary,
            TaskKind::Timeseries,
            TaskKind::Multiclass,
        ] {
            assert_eq!(
                expected_catalog_metric(kind.as_str()).unwrap(),
                kind.primary_metric()
            );
        }
        assert!(expected_catalog_metric("multi_label").is_err());
    }

    #[test]
    fn promotion_ends_at_pointer_publication_and_never_builds_supply() {
        assert_eq!(PHASES.last(), Some(&"advance_accepted_pointer"));
        assert!(!PHASES.contains(&"seed_next_strict_wave"));
        assert!(!PYTHON_PHASES.contains(&"seed_next_strict_wave"));
    }
    use tempfile::TempDir;

    #[test]
    fn python_float_repr_parity() {
        for (value, expected) in [
            (2.5, "2.5"),
            (0.1, "0.1"),
            (1e-7, "1e-07"),
            (-0.0, "-0.0"),
            (12345678901234567890123.0, "1.2345678901234568e+22"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.0, "1.0"),
            (-12.75, "-12.75"),
        ] {
            assert_eq!(py_float_repr(value).unwrap(), expected, "value {value}");
        }
    }

    #[test]
    fn python_dumps_parity() {
        let fixture = json!({
            "b": [1, 2.5, "x\"y"],
            "a": {"z": null, "k": true},
            "c": 1e-7,
            "d": 0.1,
            "e": -0.0,
            "f": 12345678901234567890123.0,
            "empty": {},
            "el": [],
        });
        assert_eq!(
            py_dumps(&fixture, PyStyle::Compact).unwrap(),
            "{\"a\":{\"k\":true,\"z\":null},\"b\":[1,2.5,\"x\\\"y\"],\"c\":1e-07,\"d\":0.1,\"e\":-0.0,\"el\":[],\"empty\":{},\"f\":1.2345678901234568e+22}"
        );
        assert_eq!(
            py_dumps(&fixture, PyStyle::Default).unwrap(),
            "{\"a\": {\"k\": true, \"z\": null}, \"b\": [1, 2.5, \"x\\\"y\"], \"c\": 1e-07, \"d\": 0.1, \"e\": -0.0, \"el\": [], \"empty\": {}, \"f\": 1.2345678901234568e+22}"
        );
        let pretty = py_dumps(&fixture, PyStyle::Pretty).unwrap();
        assert!(pretty.starts_with("{\n  \"a\": {\n    \"k\": true,\n    \"z\": null\n  },"));
        assert!(pretty.contains("\n  \"el\": [],\n  \"empty\": {},\n"));
    }

    #[test]
    fn action_key_parity_with_python() {
        let key = action_key(
            "fetch_candidate_checkpoints",
            1_805_437,
            "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592",
            "tournament",
            20_260_726,
            &"c".repeat(64),
            &"e".repeat(64),
        )
        .unwrap();
        assert_eq!(
            key,
            "93514d51284f8ba404d21e4f4fc7e34729287b7060024140256bf4657c4ca1f6"
        );
    }

    #[test]
    fn pruned_outputs_prefers_newest_attempt_with_result() {
        let temp = TempDir::new().unwrap();
        let runs = temp.path();
        let job = runs.join("jte-test/attempts");
        fs::create_dir_all(job.join("0/outputs/a")).unwrap();
        fs::create_dir_all(job.join("1/outputs/b")).unwrap();
        fs::write(job.join("0/outputs/a/train-result.json"), "{}").unwrap();
        // Only attempt 0 has a result; attempt 1 exists but is empty.
        let attempts = job.clone();
        let mut outputs: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
        for entry in fs::read_dir(&attempts).unwrap().flatten() {
            let candidate = entry.path().join("outputs");
            if candidate.is_dir() {
                let modified = fs::metadata(&candidate).and_then(|m| m.modified()).unwrap();
                outputs.push((modified, candidate));
            }
        }
        outputs.sort();
        let mut selected = None;
        for (_, candidate) in outputs.into_iter().rev() {
            let mut results = Vec::new();
            rglob(&candidate, "train-result.json", &mut results);
            if !results.is_empty() {
                selected = Some(candidate);
                break;
            }
        }
        assert_eq!(selected.unwrap(), job.join("0/outputs"));
    }

    fn synthetic_evidence(macro_lift: f64) -> Value {
        synthetic_evidence_with(macro_lift, Some(0.06), Some(0.02))
    }

    /// Same shape as the cases in `tests/fixtures/conditionality_gate_parity.json`:
    /// `None` means the key is absent from that side of the evidence entirely.
    fn synthetic_evidence_with(
        macro_lift: f64,
        candidate_conditional: Option<f64>,
        baseline_conditional: Option<f64>,
    ) -> Value {
        let mut evidence = json!({
            "baseline": {
                "checkpoint_sha256": "b".repeat(64),
                "macro_pairwise_accuracy": 0.5,
                "regression_pairwise_accuracy": 0.5,
                "binary_pairwise_accuracy": 0.5,
                "failure_rate": 0.01,
                "p95_proposal_latency_ms": 100.0,
            },
            "candidate": {
                "checkpoint_sha256": "a".repeat(64),
                "macro_pairwise_accuracy": 0.5 + macro_lift,
                "regression_pairwise_accuracy": 0.5 + macro_lift,
                "binary_pairwise_accuracy": 0.5 + macro_lift,
                "failure_rate": 0.01,
                "p95_proposal_latency_ms": 110.0,
            },
            "proposal_validity": 1.0,
            "verified_checkpoint_origin_strict_wins": 3,
            "candidate_equal_budget_result": 0.6,
            "baseline_equal_budget_result": 0.5,
            "replay_verification": {
                "predictions": true,
                "scores": true,
                "evaluator_identity": true,
                "dataset_hashes": true,
                "candidate_origin": true,
            },
        });
        if let Some(value) = candidate_conditional {
            evidence["candidate"]["val_conditional_advantage"] = json!(value);
        }
        if let Some(value) = baseline_conditional {
            evidence["baseline"]["val_conditional_advantage"] = json!(value);
        }
        evidence
    }

    fn gate_reasons(result: &Value, key: &str) -> Vec<String> {
        result[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|reason| reason.as_str().unwrap().to_owned())
            .collect()
    }

    #[test]
    fn gate_thresholds() {
        let internal = evaluate_gate(&synthetic_evidence(0.005), None).unwrap();
        assert_eq!(internal["pass"], json!(true));
        assert_eq!(internal["promotion_tier"], json!("internal"));
        let milestone = evaluate_gate(&synthetic_evidence(0.02), None).unwrap();
        assert_eq!(milestone["promotion_tier"], json!("milestone"));
        let rejected = evaluate_gate(&synthetic_evidence(0.001), None).unwrap();
        assert_eq!(rejected["pass"], json!(false));
        assert_eq!(rejected["promotion_tier"], json!("rejected"));
        assert!(rejected["reasons"]
            .as_array()
            .unwrap()
            .contains(&json!("macro_pairwise_lift_below_point_two")));
    }

    /// A checkpoint that does not condition on the dataset embedding may never
    /// be promoted, however large its macro lift is.
    #[test]
    fn conditionality_is_a_hard_promotion_gate() {
        let dead = evaluate_gate(
            &synthetic_evidence_with(0.05, Some(-0.06), Some(0.02)),
            None,
        )
        .unwrap();
        assert_eq!(dead["pass"], json!(false));
        assert_eq!(dead["promotion_tier"], json!("rejected"));
        assert!(gate_reasons(&dead, "reasons")
            .contains(&"candidate_policy_is_not_conditional".to_owned()));
        assert!(gate_reasons(&dead, "milestone_reasons")
            .contains(&"candidate_policy_is_not_conditional".to_owned()));
        assert_eq!(dead["candidate_conditional_advantage"], json!(-0.06));
        assert_eq!(dead["conditional_advantage_lift"], json!(-0.06 - 0.02));

        // Exactly zero advantage is still not a conditional policy.
        let zero =
            evaluate_gate(&synthetic_evidence_with(0.05, Some(0.0), Some(-0.01)), None).unwrap();
        assert_eq!(zero["pass"], json!(false));
        assert!(gate_reasons(&zero, "reasons")
            .contains(&"candidate_policy_is_not_conditional".to_owned()));

        // Unmeasured conditionality is fail-closed, never treated as zero.
        let missing = evaluate_gate(&synthetic_evidence_with(0.05, None, None), None).unwrap();
        assert_eq!(missing["pass"], json!(false));
        assert_eq!(missing["candidate_conditional_advantage"], Value::Null);
        assert_eq!(missing["conditional_advantage_lift"], Value::Null);
        let reasons = gate_reasons(&missing, "reasons");
        assert!(reasons.contains(&"candidate_conditional_advantage_evidence_missing".to_owned()));
        assert!(reasons.contains(&"baseline_conditional_advantage_evidence_missing".to_owned()));
        assert!(!reasons.contains(&"candidate_policy_is_not_conditional".to_owned()));

        // Positive, but under the internal margin.
        let thin = evaluate_gate(
            &synthetic_evidence_with(0.05, Some(0.004), Some(-0.02)),
            None,
        )
        .unwrap();
        assert_eq!(thin["pass"], json!(false));
        assert!(gate_reasons(&thin, "reasons")
            .contains(&"conditional_advantage_below_half_point".to_owned()));

        // Beats the baseline by half a point but not by two: internal only.
        let internal =
            evaluate_gate(&synthetic_evidence_with(0.05, Some(0.03), Some(0.02)), None).unwrap();
        assert_eq!(internal["promotion_tier"], json!("internal"));
        assert!(gate_reasons(&internal, "milestone_reasons")
            .contains(&"conditional_advantage_lift_below_two_points".to_owned()));

        // Conditional on both margins: milestone stays reachable.
        let milestone =
            evaluate_gate(&synthetic_evidence_with(0.05, Some(0.06), Some(0.02)), None).unwrap();
        assert_eq!(milestone["promotion_tier"], json!("milestone"));
    }

    /// Byte-for-byte cross-check against the Python gate: the fixture records
    /// what `scripts/test_exact_accept.evaluate_gate` produces for each case and
    /// `tests/test_durable_selfplay.py` asserts the same file from that side.
    #[test]
    fn conditionality_gate_parity_with_python() {
        let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/conditionality_gate_parity.json");
        let fixture = read_json(&fixture_path).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert!(!cases.is_empty(), "parity fixture has no cases");
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let produced = py_dumps(
                &evaluate_gate(&case["evidence"], None).unwrap(),
                PyStyle::Compact,
            )
            .unwrap();
            assert_eq!(
                produced,
                case["expected_gate_compact_json"].as_str().unwrap(),
                "gate output diverged from the Python implementation for case {name}"
            );
        }
    }

    #[test]
    fn journal_idempotency_and_pending() {
        let temp = TempDir::new().unwrap();
        let journal = ActionJournal::open(temp.path()).unwrap();
        let request = json!({"probe": 1});
        // Pending first.
        let outcome = journal.perform("k1", "probe", &request, |_| Err(pending("not yet")));
        assert!(outcome.is_err());
        let status: String = journal
            .db
            .query_row(
                "SELECT status FROM actions WHERE idempotency_key='k1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(status, "pending");
        // Then succeed.
        let result = journal
            .perform("k1", "probe", &request, |_| Ok(json!({"ok": true})))
            .unwrap();
        assert_eq!(result, json!({"ok": true}));
        // Idempotent short circuit: callback must not run again.
        let result = journal
            .perform("k1", "probe", &request, |_| {
                panic!("must not re-run succeeded action")
            })
            .unwrap();
        assert_eq!(result, json!({"ok": true}));
        let events = fs::read_to_string(temp.path().join("ops/controller-events.jsonl")).unwrap();
        assert_eq!(events.lines().count(), 4); // started,pending,started,succeeded
    }

    fn fixture_snapshot(labels: i64) -> Value {
        json!({
            "valid_labels": labels,
            "accepted_ancestor": "f".repeat(64),
            "catalog_sha256": "c".repeat(64),
            "catalog_entries": [
                {"task": "regression", "dataset_hash": "d1"},
                {"task": "binary", "dataset_hash": "d2"},
            ],
            "catalog_counts": {"regression": 1, "binary": 1},
            "evaluator_sha256": "e".repeat(64),
        })
    }

    #[test]
    fn cycle_collects_labels_below_threshold() {
        let temp = TempDir::new().unwrap();
        let controller = Controller::open(temp.path()).unwrap();
        let result = controller.cycle(&fixture_snapshot(10)).unwrap();
        assert_eq!(result["phase"], json!("collecting_labels"));
        assert_eq!(result["next_watermark"], json!(LABEL_THRESHOLD));
    }

    #[test]
    fn cycle_progresses_with_file_artifacts_and_gate_fail_skip() {
        let temp = TempDir::new().unwrap();
        let controller = Controller::open(temp.path()).unwrap();
        let snapshot = fixture_snapshot(60_000);
        // Freezes a tournament, then pends on the freeze_corpus file artifact.
        let result = controller.cycle(&snapshot).unwrap();
        assert_eq!(result["phase"], json!("freeze_corpus"));
        assert_eq!(result["pending"], json!(true));
        // Provide file artifacts for every phase up to the failing gate.
        let work = watermark_dir(temp.path(), 60_000);
        fs::create_dir_all(&work).unwrap();
        for phase in &PHASES[..7] {
            fs::write(work.join(format!("{phase}.json")), "{\"ok\": true}").unwrap();
        }
        fs::write(
            work.join("run_checkpoint_gate.json"),
            "{\"pass\": false, \"promotion_tier\": \"rejected\"}",
        )
        .unwrap();
        for _ in 0..8 {
            let outcome = controller.cycle(&snapshot).unwrap();
            assert_eq!(outcome.get("completed"), Some(&json!(true)));
        }
        // After the failed gate the pointer-advance phase must be skipped.
        let state = read_json(&temp.path().join("ops/flywheel-state.json")).unwrap();
        assert!(state["tournament"].is_null());
        assert_eq!(state["last_terminal"], json!("rejected"));
        let completed = read_json(Path::new(
            state["last_tournament"]
                .as_str()
                .expect("completed tournament path"),
        ))
        .unwrap();
        assert_eq!(
            completed.pointer("/artifacts/advance_accepted_pointer/advanced"),
            Some(&json!(false))
        );
    }

    /// WS-6: the per-task verdict must reach the WRITTEN gate artifact (not just
    /// the phase return value), and only the branch that passed on its own
    /// evidence may advance `training/accepted-by-task.json`. The global
    /// verdict, and the two global lineage pointers, must be untouched.
    #[test]
    fn run_checkpoint_gate_records_a_per_task_verdict_and_advances_only_the_passing_branch() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();
        let checkpoint = root.join("candidate.model.pt");
        fs::write(&checkpoint, b"ws6-candidate").unwrap();
        let checkpoint_sha = sha256_file(&checkpoint).unwrap();
        // Binary +0.05 (clears its own +0.002 margin), regression -0.05 (its own
        // reason). Macro stays flat, so the GLOBAL gate must still reject.
        let mut evidence = synthetic_evidence(0.0);
        evidence["candidate"]["binary_pairwise_accuracy"] = json!(0.55);
        evidence["candidate"]["regression_pairwise_accuracy"] = json!(0.45);
        evidence["candidate"]["checkpoint_sha256"] = json!(checkpoint_sha);
        evidence["candidate"]["checkpoint_path"] = json!(checkpoint.to_string_lossy());
        evidence["corpus_watermark"] = json!(60_000);
        let request =
            json!({"watermark": 60_000, "artifacts": {"build_acceptance_evidence": evidence}});

        let out = handle_run_checkpoint_gate(&request, root).unwrap();
        assert_eq!(out["pass"], json!(false), "global gate must still reject");
        assert_eq!(out["promotion_tier"], json!("rejected"));
        assert_eq!(out["tasks_passing"], json!(["binary"]));
        assert_eq!(out["task_verdicts"]["binary"]["pass"], json!(true));
        assert_eq!(out["task_verdicts"]["binary"]["advisory"], json!(false));
        assert_eq!(out["task_verdicts"]["regression"]["pass"], json!(false));
        // Unanchored new tasks: recorded as advisory, failing on their own
        // (absent) evidence, and never gating.
        for task in ["timeseries", "multiclass"] {
            assert_eq!(out["task_verdicts"][task]["advisory"], json!(true));
            assert_eq!(out["task_verdicts"][task]["pass"], json!(false));
            assert_eq!(out["task_verdicts"][task]["pairwise_lift"], Value::Null);
        }

        // The artifact on disk, not just the return value.
        let artifact =
            read_json(&watermark_dir(root, 60_000).join("checkpoint-gate.json")).unwrap();
        assert_eq!(
            artifact["task_verdict_schema"],
            json!("JopeSelfplayCheckpointGateTaskVerdict.v1")
        );
        assert_eq!(artifact["tasks_passing"], json!(["binary"]));
        assert_eq!(artifact["pass"], json!(false));
        assert_eq!(
            artifact["task_verdicts"]["regression"]["blocking_reasons_owned_by_this_task"],
            json!([
                "regression_pairwise_lift_below_point_two",
                "regression_regressed_more_than_half_point"
            ])
        );

        let pointers = read_json(&task_accepted_pointer(root)).unwrap();
        // The written schema must match what the live readers require, and what the
        // fixture below already writes. This assertion previously expected
        // "AcceptedByTask.v2" — the shadow namespace's constant for a different file —
        // which is exactly the drift that would have broken the trainer and the causal
        // ledger on the first promotion cycle.
        assert_eq!(pointers["schema"], json!("JopeSelfplayAcceptedByTask.v1"));
        assert_eq!(
            pointers["tasks"]["binary"]["checkpoint_sha256"],
            json!(checkpoint_sha)
        );
        assert_eq!(
            pointers["tasks"]["binary"]["source"],
            json!("per_task_gate")
        );
        assert_eq!(pointers["tasks"]["binary"]["generation"], json!(1));
        assert_eq!(
            pointers["tasks"]["binary"]["parent_checkpoint_sha256"],
            json!("b".repeat(64))
        );
        assert!(pointers["tasks"]["binary"]["evidence_sha256"]
            .as_str()
            .is_some_and(|value| value.len() == 64));
        assert!(pointers["tasks"]["binary"]["rollback"].is_object());
        // The failing branch is bootstrapped to the global pointer, never advanced.
        assert_eq!(
            pointers["tasks"]["regression"]["source"],
            json!("global_accepted_bootstrap")
        );
        assert_ne!(
            pointers["tasks"]["regression"]["checkpoint_sha256"],
            json!(checkpoint_sha)
        );
        // Advisory-until-anchor: the new tasks are NOT bootstrapped — presence
        // in this file is the anchor, so only a passing verdict may create it.
        assert!(pointers["tasks"].get("timeseries").is_none());
        assert!(pointers["tasks"].get("multiclass").is_none());
        // ONE global lineage pointer, and this phase does not write it: the
        // freshness epoch is defined by these two files alone.
        assert!(!root.join("training/accepted.json").exists());
        assert!(!root.join("training/learning.json").exists());
    }

    /// U3 invariant: with no anchor, a new task's verdict — even a regressing
    /// one computed from real evidence — must not alter any global decision;
    /// once anchored, the same evidence gates exactly like a legacy branch.
    #[test]
    fn new_task_verdicts_are_advisory_until_anchored() {
        let mut evidence = synthetic_evidence(0.05);
        evidence["baseline"]["timeseries_pairwise_accuracy"] = json!(0.6);
        evidence["candidate"]["timeseries_pairwise_accuracy"] = json!(0.5);

        let advisory = evaluate_gate(&evidence, None).unwrap();
        assert_eq!(advisory["pass"], json!(true));
        assert_eq!(advisory["promotion_tier"], json!("milestone"));
        assert!(!gate_reasons(&advisory, "milestone_reasons")
            .iter()
            .any(|reason| reason.starts_with("timeseries_")));
        let verdict = &advisory["task_verdicts"]["timeseries"];
        assert_eq!(verdict["advisory"], json!(true));
        assert_eq!(verdict["pass"], json!(false));
        assert!(verdict["reasons"]
            .as_array()
            .unwrap()
            .contains(&json!("timeseries_regressed_more_than_half_point")));
        assert_eq!(
            advisory["task_pairwise_deltas"]["timeseries"],
            json!(0.5 - 0.6)
        );

        // An anchor entry in accepted-by-task.json flips the branch to gating.
        let anchored = json!({
            "schema": "JopeSelfplayAcceptedByTask.v1",
            "tasks": {"timeseries": {"checkpoint_sha256": "c".repeat(64)}},
        });
        assert!(task_verdict_is_gating("timeseries", Some(&anchored)));
        assert!(!task_verdict_is_gating("multiclass", Some(&anchored)));
        let gated = evaluate_gate(&evidence, Some(&anchored)).unwrap();
        assert_eq!(gated["pass"], json!(false));
        assert_eq!(gated["promotion_tier"], json!("rejected"));
        assert!(gate_reasons(&gated, "reasons")
            .contains(&"timeseries_regressed_more_than_half_point".to_owned()));
        assert_eq!(
            gated["task_verdicts"]["timeseries"]["advisory"],
            json!(false)
        );
        // Multiclass stays advisory and, with no evidence at all, records nulls.
        assert_eq!(
            gated["task_verdicts"]["multiclass"]["advisory"],
            json!(true)
        );
        assert_eq!(gated["task_pairwise_deltas"]["multiclass"], Value::Null);
    }

    #[test]
    fn promote_refuses_ancestor_drift() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();
        let checkpoint = root.join("candidate.model.pt");
        fs::write(&checkpoint, b"weights").unwrap();
        let checkpoint_sha = sha256_file(&checkpoint).unwrap();
        let mut evidence = synthetic_evidence(0.005);
        evidence["candidate"]["checkpoint_sha256"] = json!(checkpoint_sha);
        evidence["candidate"]["checkpoint_path"] = json!(checkpoint.to_string_lossy());
        let evidence_path = root.join("evidence.json");
        atomic_json(&evidence_path, &evidence).unwrap();
        let pointer = root.join("learning.json");
        // Existing pointer whose checkpoint differs from the evidence ancestor.
        atomic_json(
            &pointer,
            &json!({"checkpoint_sha256": "9".repeat(64), "checkpoint": "x"}),
        )
        .unwrap();
        let error = promote(
            &evidence_path,
            &root.join("gate.json"),
            &pointer,
            &root.join("immutable"),
            true,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("accepted ancestor changed before pointer promotion"));
    }
}
