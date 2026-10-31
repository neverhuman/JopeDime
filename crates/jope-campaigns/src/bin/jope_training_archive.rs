//! jope_training_archive — permanent, queryable archive of every DL policy
//! training run's per-epoch metrics, back to the oldest run that still exists
//! anywhere on the cluster.
//!
//! WS-A (analysis layer). The raw byte-identical substrate is owned by WS-B
//! (`jope_training_history`, `training/history/{index.v1.json,runs/}`); this
//! binary ingests from that substrate first and falls back to the live scq run
//! directories, then adds the pre-scq lanes listed in
//! `training/history/db/imported/sources.v1.json`.
//!
//! Design rules:
//!   * every source line is stored VERBATIM (`epoch.raw_json`) — no field can be
//!     silently dropped as the trainer's schema evolves;
//!   * every scalar key is ALSO stored in a generic key/value table
//!     (`epoch_field`) so future keys are queryable without a migration;
//!   * rows are keyed by (run_key, epoch, phase) where run_key encodes
//!     (lane, run_id, attempt, watermark) — re-ingest is idempotent;
//!   * per-RUN provenance lives next to the per-EPOCH rows, because a
//!     cross-run curve without recipe / sampler / warm-start / metric-definition
//!     annotations is misleading.
//!
//! Never copies checkpoints. The whole archive is metrics only.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use jope_dime_regressor::paths;
use rusqlite::{params, Connection};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

jope_paths::once_str!(
    fn archive_root_str => paths::flywheel_root().join("training/history")
);
jope_paths::once_str!(
    fn scq_runs_str => paths::scq_runs_root()
);
jope_paths::once_str!(
    fn scq_events_str => paths::scq_events()
);
jope_paths::once_str!(
    fn specs_dir_str => paths::flywheel_root().join("specs")
);

/// mtime of `<python_src>/jope/policy/train_v3.py` after the
/// validation-metric fix (policy_v3_validation_metrics was calling
/// model.reward_stats without task_types, so on a fused-head checkpoint every
/// BINARY validation row was scored by the REGRESSION reward heads).
///
/// This boundary is wall-clock, not spec-pinned, because every trainer spec
/// sets PYTHONPATH to the live Python source tree — the executed validation code is
/// whatever was live when the process started, NOT the `provenance/train_v3.py`
/// copy staged into the run's inputs. Runs that STARTED before this instant
/// carry the pre-fix definition of val_conditional_advantage,
/// val_binary_pairwise_win_accuracy and val_conditional_score.
const METRIC_FIX_MS: i64 = 1_785_187_672_000; // 2026-07-27T21:27:52Z

const SCHEMA_SQL: &str = r#"
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- One row per distinct physical source file, deduplicated by content hash.
CREATE TABLE IF NOT EXISTS source (
  sha256        TEXT PRIMARY KEY,
  origin_node   TEXT NOT NULL,
  origin_path   TEXT NOT NULL,
  archive_path  TEXT,
  bytes         INTEGER NOT NULL,
  mtime_ms      INTEGER NOT NULL,
  rows          INTEGER NOT NULL,
  first_seen_ms INTEGER NOT NULL,
  last_ingest_ms INTEGER NOT NULL
);

-- Extra physical locations that carry byte-identical content (e.g. the 30+
-- copies of registry/compatibility/gen31/train_metrics.jsonl).
CREATE TABLE IF NOT EXISTS source_alias (
  sha256      TEXT NOT NULL,
  origin_node TEXT NOT NULL,
  origin_path TEXT NOT NULL,
  PRIMARY KEY (sha256, origin_node, origin_path)
);

-- Per-RUN provenance. run_key = lane '/' run_id '/' attempt '/' watermark.
CREATE TABLE IF NOT EXISTS run (
  run_key   TEXT PRIMARY KEY,
  lane      TEXT NOT NULL,
  lane_kind TEXT NOT NULL,          -- 'epoch' | 'step'
  in_policy_curve INTEGER NOT NULL, -- 1 = belongs on the train/val loss curve
  run_id    TEXT NOT NULL,
  attempt   INTEGER NOT NULL,
  watermark TEXT NOT NULL,
  node      TEXT,
  generation INTEGER,
  model_version TEXT,
  recipe    TEXT,
  seed      TEXT,
  sampler   TEXT,                   -- e.g. 'eq32v32c128'
  corpus_snapshot TEXT,
  anti_overfit INTEGER,             -- 0/1/NULL
  warm_start_sha256 TEXT,
  warm_start_path   TEXT,
  warm_start_tensors_loaded INTEGER,
  warm_start_tensors_total  INTEGER,
  result_checkpoint_sha256  TEXT,
  selected_checkpoint_epoch INTEGER,
  selected_checkpoint_metric_name TEXT,
  selected_checkpoint_metric REAL,
  epochs_planned  INTEGER,
  rows_recorded   INTEGER NOT NULL DEFAULT 0,
  first_epoch     INTEGER,
  last_epoch      INTEGER,
  early_stopped   INTEGER,
  completed       INTEGER,          -- 1 = train-result.json present
  started_at_ms   INTEGER,
  ended_at_ms     INTEGER,
  time_basis      TEXT,             -- how started/ended were derived
  scq_state       TEXT,
  metric_definition TEXT,           -- 'pre_task_types_fix' | 'post_task_types_fix' | 'no_split_heads'
  trainer_source_sha256 TEXT,       -- spec-render-time sha of train_v3.py (see METRIC_FIX_MS note)
  train_script_sha256   TEXT,
  source_sha256   TEXT,
  source_origin   TEXT,
  source_live     INTEGER,           -- 1 = the originating run dir / file still exists
  spec_path       TEXT,
  notes           TEXT
);

-- Per-EPOCH rows. raw_json is the source line, byte for byte.
CREATE TABLE IF NOT EXISTS epoch (
  run_key TEXT NOT NULL REFERENCES run(run_key) ON DELETE CASCADE,
  epoch   INTEGER NOT NULL,
  phase   TEXT NOT NULL,
  row_ord INTEGER NOT NULL,
  train_loss REAL, val_loss REAL, overfit_gap REAL,
  lr REAL, batch_size INTEGER, epochs INTEGER,
  elapsed_sec REAL, epoch_sec REAL,
  train_rows INTEGER, validation_rows INTEGER,
  model_version TEXT, device TEXT, world_size INTEGER,
  val_fallback INTEGER,
  macro_validation_best REAL,
  macro_validation_early_stopped INTEGER,
  val_conditional_advantage REAL,
  val_conditional_score REAL,
  val_pairwise_win_accuracy REAL,
  val_binary_pairwise_win_accuracy REAL,
  val_regression_pairwise_win_accuracy REAL,
  val_ablation_pairwise_win REAL,
  val_reward_cv_correlation REAL,
  val_top1_historical_regret REAL,
  val_seq_nll REAL,
  val_head_accuracy_featmath REAL,
  val_uncertainty_error_spearman REAL,
  anti_overfit INTEGER,
  wall_ms INTEGER,                  -- derived emission time, run start + elapsed_sec
  raw_json TEXT NOT NULL,
  raw_sha256 TEXT NOT NULL,
  PRIMARY KEY (run_key, epoch, phase)
);
CREATE INDEX IF NOT EXISTS epoch_by_run ON epoch(run_key, row_ord);
CREATE INDEX IF NOT EXISTS epoch_by_wall ON epoch(wall_ms);

-- Generic projection of every scalar key in every row. Future keys land here
-- with no migration, so nothing is ever only-in-the-blob.
CREATE TABLE IF NOT EXISTS epoch_field (
  run_key TEXT NOT NULL,
  epoch   INTEGER NOT NULL,
  phase   TEXT NOT NULL,
  key     TEXT NOT NULL,
  num     REAL,
  txt     TEXT,
  json    TEXT,
  PRIMARY KEY (run_key, epoch, phase, key)
);
CREATE INDEX IF NOT EXISTS epoch_field_by_key ON epoch_field(key);

-- Schema-evolution ledger: when each key first and last appears.
CREATE TABLE IF NOT EXISTS field_catalog (
  key TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  runs_present INTEGER NOT NULL,
  rows_present INTEGER NOT NULL,
  first_seen_ms INTEGER,
  last_seen_ms  INTEGER,
  first_run_key TEXT,
  last_run_key  TEXT
);

-- Sanity findings, recomputed on every ingest.
CREATE TABLE IF NOT EXISTS flag (
  run_key  TEXT NOT NULL,
  epoch    INTEGER NOT NULL,   -- -1 = run-level
  phase    TEXT NOT NULL,      -- '*' = run-level
  code     TEXT NOT NULL,
  severity TEXT NOT NULL,      -- 'info' | 'warn' | 'error'
  detail   TEXT,
  PRIMARY KEY (run_key, epoch, phase, code)
);
CREATE INDEX IF NOT EXISTS flag_by_code ON flag(code);

-- Curve annotations (metric-definition changes, lineage forks, ...).
CREATE TABLE IF NOT EXISTS annotation (
  id     TEXT PRIMARY KEY,
  at_ms  INTEGER NOT NULL,
  kind   TEXT NOT NULL,
  label  TEXT NOT NULL,
  detail TEXT NOT NULL,
  evidence TEXT
);

-- History that is proven gone, so the record never reads as complete.
CREATE TABLE IF NOT EXISTS gap (
  id TEXT PRIMARY KEY,
  lane TEXT NOT NULL,
  what TEXT NOT NULL,
  evidence TEXT NOT NULL,
  approx_window TEXT
);
"#;

#[derive(Parser)]
#[command(
    name = "jope_training_archive",
    about = "Permanent queryable DL training-history archive (WS-A)"
)]
struct Cli {
    /// Archive root (contains db/).
    #[arg(long, default_value = archive_root_str())]
    root: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Scan every known source and (re)load it. Idempotent.
    Ingest {
        #[arg(long)]
        dry_run: bool,
    },
    /// Re-hash every archived row against the DB and report drift.
    Verify,
    /// One-screen summary of what is loaded.
    Stats,
    /// Per-run provenance table.
    Runs {
        #[arg(long)]
        lane: Option<String>,
    },
    /// Sanity findings.
    Flags {
        #[arg(long)]
        severity: Option<String>,
    },
    /// Emit the train/val series for consumers.
    Export {
        /// json | pgfplots | both
        #[arg(long, default_value = "both")]
        format: String,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Restrict to rows on the policy curve (default true).
        #[arg(long, default_value_t = true)]
        policy_only: bool,
    },
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn mtime_ms(p: &Path) -> Result<i64> {
    let md = fs::metadata(p).with_context(|| format!("stat {}", p.display()))?;
    Ok(md.modified()?.duration_since(UNIX_EPOCH)?.as_millis() as i64)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn f(v: Option<&Value>) -> Option<f64> {
    v.and_then(|x| x.as_f64())
}
fn i(v: Option<&Value>) -> Option<i64> {
    v.and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|y| y as i64)))
}
fn s(v: Option<&Value>) -> Option<String> {
    v.and_then(|x| x.as_str().map(|y| y.to_string()))
}
fn b(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Bool(x)) => Some(*x as i64),
        Some(Value::Object(_)) => Some(1), // anti_overfit_experiment is an object when enabled
        Some(Value::Null) | None => None,
        Some(Value::Number(n)) => n.as_i64(),
        _ => None,
    }
}

// ---------------------------------------------------------------- discovery

#[derive(Debug, Clone)]
struct SourceSpec {
    lane: String,
    lane_kind: String,
    in_policy_curve: bool,
    run_id: String,
    attempt: i64,
    watermark: String,
    node: Option<String>,
    generation: Option<i64>,
    model_version_hint: Option<String>,
    /// File actually read.
    path: PathBuf,
    /// Where the bytes originally lived (may be a remote node).
    origin: String,
    sidecar_result: Option<PathBuf>,
    sidecar_kind: Option<String>,
    notes: Option<String>,
    /// Extra physical copies with identical content.
    aliases: Vec<(String, String)>,
}

impl SourceSpec {
    fn run_key(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            self.lane, self.run_id, self.attempt, self.watermark
        )
    }
}

/// scq run dirs, preferring WS-B's durable copy when it has at least as many rows.
fn discover_scq(root: &Path) -> Result<Vec<SourceSpec>> {
    let mut by_key: BTreeMap<(String, i64, String), (PathBuf, String, usize)> = BTreeMap::new();

    let mut consider = |p: PathBuf, origin: String, run_id: String, attempt: i64, wm: String| {
        let rows = match fs::read_to_string(&p) {
            Ok(t) => t.lines().filter(|l| !l.trim().is_empty()).count(),
            Err(_) => return,
        };
        let key = (run_id, attempt, wm);
        let better = match by_key.get(&key) {
            Some((_, _, prev)) => rows > *prev,
            None => true,
        };
        if better {
            by_key.insert(key, (p, origin, rows));
        }
    };

    // (a) WS-B archive: runs/<run_id>/<attempt>/<watermark>/train_metrics.jsonl
    let wsb = root.join("runs");
    if wsb.is_dir() {
        for run in fs::read_dir(&wsb)? {
            let run = run?;
            if !run.file_type()?.is_dir() {
                continue;
            }
            let run_id = run.file_name().to_string_lossy().to_string();
            for att in fs::read_dir(run.path())? {
                let att = att?;
                if !att.file_type()?.is_dir() {
                    continue;
                }
                let attempt: i64 = att.file_name().to_string_lossy().parse().unwrap_or(0);
                for wmd in fs::read_dir(att.path())? {
                    let wmd = wmd?;
                    if !wmd.file_type()?.is_dir() {
                        continue;
                    }
                    let wm = wmd
                        .file_name()
                        .to_string_lossy()
                        .trim_start_matches("watermark-")
                        .to_string();
                    let p = wmd.path().join("train_metrics.jsonl");
                    if p.is_file() {
                        let origin = format!("xbabe2:{}", p.display());
                        consider(p, origin, run_id.clone(), attempt, wm);
                    }
                }
            }
        }
    }

    // (b) live scq run dirs (authoritative while they exist; may hold newer rows)
    let scq = Path::new(scq_runs_str());
    if scq.is_dir() {
        for run in fs::read_dir(scq)? {
            let run = run?;
            let run_id = run.file_name().to_string_lossy().to_string();
            if !run_id.contains("train") || !run.file_type()?.is_dir() {
                continue;
            }
            let attempts = run.path().join("attempts");
            if !attempts.is_dir() {
                continue;
            }
            for att in fs::read_dir(&attempts)? {
                let att = att?;
                let attempt: i64 = att.file_name().to_string_lossy().parse().unwrap_or(0);
                let tdir = att.path().join("outputs/training");
                if !tdir.is_dir() {
                    continue;
                }
                for wmd in fs::read_dir(&tdir)? {
                    let wmd = wmd?;
                    let wm = wmd
                        .file_name()
                        .to_string_lossy()
                        .trim_start_matches("watermark-")
                        .to_string();
                    let p = wmd.path().join("train_metrics.jsonl");
                    if p.is_file() {
                        let origin = format!("xbabe2:{}", p.display());
                        consider(p, origin, run_id.clone(), attempt, wm);
                    }
                }
            }
        }
    }

    Ok(by_key
        .into_iter()
        .map(
            |((run_id, attempt, watermark), (path, origin, _))| SourceSpec {
                lane: "scq".into(),
                lane_kind: "epoch".into(),
                in_policy_curve: true,
                run_id,
                attempt,
                watermark,
                node: None, // resolved from scq events
                generation: None,
                model_version_hint: None,
                path,
                origin,
                sidecar_result: None, // resolved below
                sidecar_kind: Some("train_result".into()),
                notes: None,
                aliases: vec![],
            },
        )
        .collect())
}

/// Glob with a single '*' wildcard: "gen*_train_metrics.jsonl".
fn glob1(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        Some((a, b2)) => {
            name.len() >= a.len() + b2.len() && name.starts_with(a) && name.ends_with(b2)
        }
        None => pattern == name,
    }
}

/// Mirror `src` into the archive (preserving mtime) unless an identical copy is
/// already there. The archive must never depend on a directory it does not own.
fn mirror(src: &Path, dst: &Path) -> Result<()> {
    if let Some(p) = dst.parent() {
        fs::create_dir_all(p)?;
    }
    let bytes = fs::read(src)?;
    let same = fs::read(dst).map(|d| d == bytes).unwrap_or(false);
    if !same {
        let tmp = dst.with_extension("tmp-ingest");
        fs::write(&tmp, &bytes)?;
        fs::rename(&tmp, dst)?;
    }
    // keep the source mtime: it is the only wall-clock anchor these lanes have
    let mt = fs::metadata(src)?.modified()?;
    let ft = fs::File::options().write(true).open(dst)?;
    ft.set_modified(mt)?;
    Ok(())
}

/// Walk each `lane_roots` directory, mirror what it finds into the archive and
/// return one SourceSpec per file. run_id is derived from the file name so that
/// re-running is stable.
fn discover_lane_roots(root: &Path, doc: &Value) -> Result<Vec<SourceSpec>> {
    let mut out = vec![];
    for lr in doc["lane_roots"].as_array().cloned().unwrap_or_default() {
        let dir = PathBuf::from(lr["dir"].as_str().unwrap_or_default());
        if !dir.is_dir() {
            eprintln!("WARN lane_root missing: {}", dir.display());
            continue;
        }
        let pattern = lr["pattern"].as_str().unwrap_or("*");
        let lane = lr["lane"].as_str().unwrap_or("legacy").to_string();
        let subdir = lr["archive_subdir"].as_str().unwrap_or(&lane).to_string();
        let node = lr["node"].as_str().unwrap_or("xbabe2").to_string();
        let sidecar_pattern = lr["sidecar_pattern"].as_str().map(|x| x.to_string());
        let mut names: Vec<String> = fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| glob1(pattern, n))
            .collect();
        names.sort();
        for name in names {
            let src = dir.join(&name);
            let dst = root.join("db/imported").join(&subdir).join(&name);
            mirror(&src, &dst)?;
            // gen17_train_metrics.jsonl -> gen17 ; train_metrics.pre_logging_patch.jsonl -> pre_logging_patch
            let stem = name
                .trim_end_matches(".jsonl")
                .replace("_train_metrics", "")
                .replace("train_metrics.", "")
                .replace("train_metrics", "")
                .trim_matches(['_', '.', '-'])
                .to_string();
            // archive_subdir's last component is unique per lane root, and file
            // names are unique within a directory, so this run_id cannot collide.
            let base = subdir
                .rsplit('/')
                .next()
                .unwrap_or(&lane)
                .trim_start_matches("jope-runs-")
                .trim_start_matches("jope-prime-")
                .to_string();
            let run_id = if stem.is_empty() {
                base
            } else {
                format!("{base}-{stem}")
            };
            let generation = stem.strip_prefix("gen").and_then(|g| g.parse::<i64>().ok());
            // sidecar with the same generation prefix, if the lane declares one
            let sidecar = sidecar_pattern.as_ref().and_then(|sp| {
                let want = sp.replacen('*', stem.strip_prefix("gen").unwrap_or(&stem), 1);
                let s = dir.join(&want);
                if s.is_file() {
                    let d = root.join("db/imported").join(&subdir).join(&want);
                    mirror(&s, &d).ok()?;
                    Some(d)
                } else {
                    None
                }
            });
            out.push(SourceSpec {
                lane: lane.clone(),
                lane_kind: lr["lane_kind"].as_str().unwrap_or("epoch").to_string(),
                in_policy_curve: lr["in_policy_curve"].as_bool().unwrap_or(false),
                run_id,
                attempt: 0,
                watermark: "none".into(),
                node: Some(node.clone()),
                generation,
                model_version_hint: lr["model_version_hint"].as_str().map(|x| x.to_string()),
                path: dst,
                origin: format!("{node}:{}", src.display()),
                sidecar_result: sidecar,
                sidecar_kind: lr["sidecar_kind"].as_str().map(|x| x.to_string()),
                notes: lr["notes"].as_str().map(|x| x.to_string()),
                aliases: vec![],
            });
        }
    }
    Ok(out)
}

fn discover_imported(root: &Path) -> Result<(Vec<SourceSpec>, Value)> {
    let manifest_path = root.join("db/imported/sources.v1.json");
    if !manifest_path.is_file() {
        return Ok((vec![], Value::Null));
    }
    let txt = fs::read_to_string(&manifest_path)?;
    let doc: Value =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", manifest_path.display()))?;
    let mut out = discover_lane_roots(root, &doc)?;
    for src in doc["sources"].as_array().cloned().unwrap_or_default() {
        let local = src["local_path"].as_str().unwrap_or_default();
        let path = root.join("db").join(local);
        if !path.is_file() {
            eprintln!("WARN missing imported source {}", path.display());
            continue;
        }
        let node = src["node"].as_str().unwrap_or("unknown").to_string();
        out.push(SourceSpec {
            lane: src["lane"].as_str().unwrap_or("imported").to_string(),
            lane_kind: src["lane_kind"].as_str().unwrap_or("epoch").to_string(),
            in_policy_curve: src["in_policy_curve"].as_bool().unwrap_or(false),
            run_id: src["run_id"].as_str().unwrap_or("unknown").to_string(),
            attempt: src["attempt"].as_i64().unwrap_or(0),
            watermark: src["watermark"].as_str().unwrap_or("none").to_string(),
            node: Some(node.clone()),
            generation: src["generation"].as_i64(),
            model_version_hint: src["model_version_hint"].as_str().map(|x| x.to_string()),
            path,
            origin: format!("{}:{}", node, src["origin_path"].as_str().unwrap_or("?")),
            sidecar_result: src["sidecar_result"]
                .as_str()
                .map(|p| root.join("db").join(p)),
            sidecar_kind: src["sidecar_kind"].as_str().map(|x| x.to_string()),
            notes: src["notes"].as_str().map(|x| x.to_string()),
            aliases: vec![],
        });
    }
    Ok((out, doc))
}

// ------------------------------------------------------------- scq metadata

#[derive(Default, Debug, Clone)]
struct ScqTiming {
    node: Option<String>,
    submit_ms: Option<i64>,
    running_ms: Option<i64>,
    end_ms: Option<i64>,
    state: Option<String>,
}

fn parse_ts_ms(ts: &str) -> Option<i64> {
    // "2026-07-28T00:19:58.123456789+00:00" -> epoch ms, UTC offsets only.
    let bytes = ts.as_bytes();
    if bytes.len() < 19 {
        return None;
    }
    let y: i64 = ts.get(0..4)?.parse().ok()?;
    let mo: i64 = ts.get(5..7)?.parse().ok()?;
    let d: i64 = ts.get(8..10)?.parse().ok()?;
    let h: i64 = ts.get(11..13)?.parse().ok()?;
    let mi: i64 = ts.get(14..16)?.parse().ok()?;
    let sec: i64 = ts.get(17..19)?.parse().ok()?;
    let frac = if bytes.get(19) == Some(&b'.') {
        let mut ms = 0i64;
        let mut mult = 100i64;
        for c in bytes[20..].iter().take(3) {
            if !c.is_ascii_digit() {
                break;
            }
            ms += ((c - b'0') as i64) * mult;
            mult /= 10;
        }
        ms
    } else {
        0
    };
    // days since epoch (civil_from_days inverse, Howard Hinnant's algorithm)
    let yy = if mo <= 2 { y - 1 } else { y };
    let era = if yy >= 0 { yy } else { yy - 399 } / 400;
    let yoe = yy - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(((days * 86400 + h * 3600 + mi * 60 + sec) * 1000) + frac)
}

fn scq_timings() -> Result<BTreeMap<String, ScqTiming>> {
    let mut out: BTreeMap<String, ScqTiming> = BTreeMap::new();
    let path = Path::new(scq_events_str());
    if !path.is_file() {
        return Ok(out);
    }
    let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    for line in txt.lines() {
        if !line.contains("train") {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let job = match v["job"].as_str() {
            Some(j) if j.contains("train") => j.to_string(),
            _ => continue,
        };
        let kind = v["kind"].as_str().unwrap_or("");
        let ts = v["ts"].as_str().and_then(parse_ts_ms);
        let e = out.entry(job).or_default();
        if let Some(n) = v["node"].as_str() {
            e.node = Some(n.to_string());
        }
        match kind {
            "submit" => e.submit_ms = e.submit_ms.or(ts),
            "running" => {
                if e.running_ms.is_none() {
                    e.running_ms = ts;
                }
            }
            "exit" | "cancel" | "fail" => {
                e.end_ms = ts;
                e.state = Some(kind.to_string());
            }
            _ => {}
        }
        if e.state.is_none() && kind == "running" {
            e.state = Some("running".into());
        }
    }
    Ok(out)
}

#[derive(Default, Debug, Clone)]
struct SpecInfo {
    recipe: Option<String>,
    seed: Option<String>,
    corpus_snapshot: Option<String>,
    sampler: Option<String>,
    warm_start_sha256: Option<String>,
    trainer_source_sha256: Option<String>,
    train_script_sha256: Option<String>,
    path: Option<String>,
}

fn read_spec(run_id: &str) -> SpecInfo {
    let mut info = SpecInfo::default();
    let p = PathBuf::from(specs_dir_str()).join(format!("{run_id}.toml"));
    let txt = match fs::read_to_string(&p) {
        Ok(t) => t,
        Err(_) => return info,
    };
    info.path = Some(p.display().to_string());
    // args = [... "--recipe", "a", "--seed", "20260726" ...]
    let toks: Vec<&str> = txt.split('"').collect();
    for w in toks.windows(3) {
        match w[0] {
            "--recipe" => info.recipe = Some(w[2].to_string()),
            "--seed" => info.seed = Some(w[2].to_string()),
            _ => {}
        }
    }
    // [[input]] blocks: name / source / target / sha256
    let mut cur_name: Option<String> = None;
    let mut cur_source: Option<String> = None;
    for line in txt.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("name = ") {
            cur_name = Some(rest.trim_matches('"').to_string());
            cur_source = None;
        } else if let Some(rest) = l.strip_prefix("source = ") {
            cur_source = Some(rest.trim_matches('"').to_string());
        } else if let Some(rest) = l.strip_prefix("sha256 = ") {
            let sha = rest.trim_matches('"').to_string();
            match cur_name.as_deref() {
                Some("checkpoint") => info.warm_start_sha256 = Some(sha),
                Some("trainer_source") => info.trainer_source_sha256 = Some(sha),
                Some("script") => info.train_script_sha256 = Some(sha),
                Some("manifest") => {
                    if let Some(src) = &cur_source {
                        if let Some(idx) = src.find("/training/snapshot-") {
                            let tail = &src[idx + "/training/".len()..];
                            let snap = tail.split('/').next().unwrap_or(tail).to_string();
                            if let Some(pos) = snap.rfind("-eq") {
                                info.sampler = Some(snap[pos + 1..].to_string());
                            }
                            info.corpus_snapshot = Some(snap);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    info
}

/// "warm-started 130/152 tensors from `<path>`"
fn read_warm_start(run_id: &str, attempt: i64) -> Option<(i64, i64, String)> {
    let p = PathBuf::from(scq_runs_str())
        .join(run_id)
        .join("attempts")
        .join(attempt.to_string())
        .join("stdout.log");
    let txt = fs::read_to_string(p).ok()?;
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("warm-started ") {
            let mut it = rest.split_whitespace();
            let frac = it.next()?;
            let (a, bb) = frac.split_once('/')?;
            let _tensors = it.next()?; // "tensors"
            let _from = it.next()?; // "from"
            let path = it.next()?.trim_end_matches(';').to_string();
            return Some((a.parse().ok()?, bb.parse().ok()?, path));
        }
    }
    None
}

fn find_train_result(spec: &SourceSpec) -> Option<PathBuf> {
    if let Some(p) = &spec.sidecar_result {
        if p.is_file() {
            return Some(p.clone());
        }
    }
    if spec.lane != "scq" {
        return None;
    }
    let live = spec.path.parent()?.join("train-result.json");
    if live.is_file() {
        return Some(live);
    }
    None
}

// ------------------------------------------------------------------ ingest

struct RowOut {
    epoch: i64,
    phase: String,
    row_ord: i64,
    obj: Map<String, Value>,
    raw: String,
}

fn parse_rows(text: &str, lane_kind: &str) -> (Vec<RowOut>, Vec<String>) {
    let mut rows = vec![];
    let mut problems = vec![];
    let mut seen: BTreeSet<(i64, String)> = BTreeSet::new();
    for (idx, line) in text.lines().enumerate() {
        let raw = line.trim_end_matches(['\r', '\n']);
        if raw.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(e) => {
                problems.push(format!("line {}: unparseable ({e})", idx + 1));
                continue;
            }
        };
        let obj = match v {
            Value::Object(m) => m,
            _ => {
                problems.push(format!("line {}: not an object", idx + 1));
                continue;
            }
        };
        let epoch = i(obj.get("epoch"))
            .or_else(|| i(obj.get("step")))
            .or_else(|| i(obj.get("seq")))
            .unwrap_or(idx as i64 + 1);
        let mut phase = obj
            .get("phase")
            .and_then(|x| x.as_str())
            .map(|x| x.to_string())
            .unwrap_or_else(|| {
                if lane_kind == "step" {
                    "STEP".into()
                } else {
                    "?".into()
                }
            });
        // Never silently overwrite: disambiguate a repeated (epoch, phase).
        if seen.contains(&(epoch, phase.clone())) {
            let disambiguated = format!("{phase}#{}", idx + 1);
            problems.push(format!(
                "line {}: duplicate (epoch={epoch}, phase={phase}) -> stored as phase '{disambiguated}'",
                idx + 1
            ));
            phase = disambiguated;
        }
        seen.insert((epoch, phase.clone()));
        rows.push(RowOut {
            epoch,
            phase,
            row_ord: idx as i64,
            obj,
            raw: raw.to_string(),
        });
    }
    (rows, problems)
}

#[allow(clippy::too_many_lines)]
fn ingest(root: &Path, dry_run: bool) -> Result<()> {
    let db_dir = root.join("db");
    fs::create_dir_all(&db_dir)?;
    let db_path = db_dir.join("history.sqlite3");

    let mut specs = discover_scq(root)?;
    let (imported, manifest) = discover_imported(root)?;
    specs.extend(imported);

    let timings = scq_timings()?;
    let started = now_ms();

    if dry_run {
        println!("{:<58} {:>6}  path", "run_key", "rows");
        for sp in &specs {
            let n = fs::read_to_string(&sp.path)
                .map(|t| t.lines().filter(|l| !l.trim().is_empty()).count())
                .unwrap_or(0);
            println!("{:<58} {:>6}  {}", sp.run_key(), n, sp.path.display());
        }
        println!("\n{} sources, dry-run (nothing written)", specs.len());
        return Ok(());
    }

    let mut conn = Connection::open(&db_path)?;
    conn.execute_batch(SCHEMA_SQL)?;

    let mut total_rows = 0usize;
    let mut total_runs = 0usize;
    let mut all_problems: Vec<String> = vec![];
    #[allow(unused_assignments)]
    let mut truncated = 0usize;

    {
        let tx = conn.transaction()?;
        // flags/field_catalog are fully recomputed each pass
        tx.execute("DELETE FROM flag", [])?;
        tx.execute("DELETE FROM field_catalog", [])?;

        type FieldStats = (String, BTreeSet<String>, i64, i64, i64, String, String);
        let mut field_stats: BTreeMap<String, FieldStats> = BTreeMap::new();

        for sp in &specs {
            let bytes = fs::read(&sp.path)?;
            let text = String::from_utf8_lossy(&bytes).to_string();
            let src_sha = sha256_hex(&bytes);
            let src_mtime = mtime_ms(&sp.path)?;
            let (rows, problems) = parse_rows(&text, &sp.lane_kind);
            for p in &problems {
                all_problems.push(format!("{}: {p}", sp.run_key()));
            }
            if rows.is_empty() {
                continue;
            }

            let run_key = sp.run_key();

            // ---- source table
            tx.execute(
                "INSERT INTO source(sha256,origin_node,origin_path,archive_path,bytes,mtime_ms,rows,first_seen_ms,last_ingest_ms)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?8)
                 ON CONFLICT(sha256) DO UPDATE SET last_ingest_ms=?8, archive_path=?4, rows=?7",
                params![
                    src_sha,
                    sp.node.clone().unwrap_or_else(|| "xbabe2".into()),
                    sp.origin,
                    sp.path.display().to_string(),
                    bytes.len() as i64,
                    src_mtime,
                    rows.len() as i64,
                    started
                ],
            )?;
            for (node, path) in &sp.aliases {
                tx.execute(
                    "INSERT OR IGNORE INTO source_alias(sha256,origin_node,origin_path) VALUES(?1,?2,?3)",
                    params![src_sha, node, path],
                )?;
            }

            // ---- per-run provenance
            let timing = timings.get(&sp.run_id).cloned().unwrap_or_default();
            let spec = if sp.lane == "scq" {
                read_spec(&sp.run_id)
            } else {
                SpecInfo::default()
            };
            let warm = if sp.lane == "scq" {
                read_warm_start(&sp.run_id, sp.attempt)
            } else {
                None
            };

            let last = rows.last().unwrap();
            let last_elapsed = f(last.obj.get("elapsed_sec")).unwrap_or(0.0);
            let (started_ms, ended_ms, basis) = match (timing.running_ms, timing.end_ms) {
                (Some(a), Some(b2)) => (Some(a), Some(b2), "scq_events"),
                (Some(a), None) => (Some(a), Some(src_mtime), "scq_running+file_mtime"),
                _ => {
                    let st = src_mtime - (last_elapsed * 1000.0) as i64;
                    (Some(st), Some(src_mtime), "file_mtime-elapsed_sec")
                }
            };

            let result = find_train_result(sp).and_then(|p| {
                fs::read_to_string(&p)
                    .ok()
                    .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            });
            let rank0 = if sp.sidecar_kind.as_deref() == Some("rank0_log") {
                sp.sidecar_result
                    .as_ref()
                    .and_then(|p| fs::read_to_string(p).ok())
                    .and_then(|t| {
                        t.lines()
                            .rev()
                            .find(|l| l.trim_start().starts_with('{'))
                            .and_then(|l| serde_json::from_str::<Value>(l).ok())
                    })
            } else {
                None
            };
            let manifest_side = if sp.sidecar_kind.as_deref() == Some("checkpoint_manifest") {
                sp.sidecar_result
                    .as_ref()
                    .and_then(|p| fs::read_to_string(p).ok())
                    .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            } else {
                None
            };

            let model_version = last
                .obj
                .get("model_version")
                .and_then(|x| x.as_str())
                .map(|x| x.to_string())
                .or_else(|| sp.model_version_hint.clone());

            let anti_overfit = rows
                .iter()
                .any(|r| b(r.obj.get("anti_overfit_experiment")) == Some(1));
            let early_stopped = rows
                .iter()
                .any(|r| b(r.obj.get("macro_validation_early_stopped")) == Some(1));

            let (ck_sha, sel_epoch, sel_metric_name, sel_metric, epochs_planned) =
                match (&result, &rank0, &manifest_side) {
                    (Some(r), _, _) => (
                        s(r.get("checkpoint_sha256"))
                            .or_else(|| s(r.pointer("/driver/checkpoint_sha256"))),
                        i(r.get("selected_checkpoint_epoch")),
                        s(r.get("selected_checkpoint_metric_name")),
                        f(r.get("selected_checkpoint_metric")).or_else(|| f(r.get("best_metric"))),
                        i(r.get("epochs")),
                    ),
                    (None, Some(r), _) => (
                        s(r.get("checkpoint")),
                        None,
                        Some("val_pairwise_win_accuracy".to_string()),
                        f(r.get("best_metric")),
                        i(r.get("epochs")),
                    ),
                    (None, None, Some(m)) => (
                        s(m.get("model_hash")),
                        i(m.pointer("/metrics/historical_best_epoch")),
                        Some("val_pairwise_win_accuracy".to_string()),
                        f(m.pointer("/metrics/historical_best_metric_value")),
                        i(m.get("epoch")),
                    ),
                    _ => (None, None, None, None, i(last.obj.get("epochs"))),
                };

            let metric_definition = if !rows
                .iter()
                .any(|r| r.obj.contains_key("val_binary_pairwise_win_accuracy"))
            {
                "no_split_heads"
            } else if started_ms.unwrap_or(0) < METRIC_FIX_MS {
                "pre_task_types_fix"
            } else {
                "post_task_types_fix"
            };

            tx.execute(
                "INSERT INTO run(run_key,lane,lane_kind,in_policy_curve,run_id,attempt,watermark,node,generation,
                                 model_version,recipe,seed,sampler,corpus_snapshot,anti_overfit,
                                 warm_start_sha256,warm_start_path,warm_start_tensors_loaded,warm_start_tensors_total,
                                 result_checkpoint_sha256,selected_checkpoint_epoch,selected_checkpoint_metric_name,
                                 selected_checkpoint_metric,epochs_planned,rows_recorded,first_epoch,last_epoch,
                                 early_stopped,completed,started_at_ms,ended_at_ms,time_basis,scq_state,
                                 metric_definition,trainer_source_sha256,train_script_sha256,
                                 source_sha256,source_origin,spec_path,notes,source_live)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,
                        ?26,?27,?28,?29,?30,?31,?32,?33,?34,?35,?36,?37,?38,?39,?40,?41)
                 ON CONFLICT(run_key) DO UPDATE SET
                   node=excluded.node, model_version=excluded.model_version, recipe=excluded.recipe,
                   seed=excluded.seed, sampler=excluded.sampler, corpus_snapshot=excluded.corpus_snapshot,
                   anti_overfit=excluded.anti_overfit, warm_start_sha256=excluded.warm_start_sha256,
                   warm_start_path=excluded.warm_start_path,
                   warm_start_tensors_loaded=excluded.warm_start_tensors_loaded,
                   warm_start_tensors_total=excluded.warm_start_tensors_total,
                   result_checkpoint_sha256=excluded.result_checkpoint_sha256,
                   selected_checkpoint_epoch=excluded.selected_checkpoint_epoch,
                   selected_checkpoint_metric_name=excluded.selected_checkpoint_metric_name,
                   selected_checkpoint_metric=excluded.selected_checkpoint_metric,
                   epochs_planned=excluded.epochs_planned, rows_recorded=excluded.rows_recorded,
                   first_epoch=excluded.first_epoch, last_epoch=excluded.last_epoch,
                   early_stopped=excluded.early_stopped, completed=excluded.completed,
                   started_at_ms=excluded.started_at_ms, ended_at_ms=excluded.ended_at_ms,
                   time_basis=excluded.time_basis, scq_state=excluded.scq_state,
                   metric_definition=excluded.metric_definition,
                   trainer_source_sha256=excluded.trainer_source_sha256,
                   train_script_sha256=excluded.train_script_sha256,
                   source_sha256=excluded.source_sha256, source_origin=excluded.source_origin,
                   spec_path=excluded.spec_path, source_live=excluded.source_live",
                params![
                    run_key, sp.lane, sp.lane_kind, sp.in_policy_curve as i64, sp.run_id, sp.attempt,
                    sp.watermark,
                    sp.node.clone().or(timing.node.clone()),
                    sp.generation,
                    model_version, spec.recipe, spec.seed, spec.sampler, spec.corpus_snapshot,
                    anti_overfit as i64,
                    warm.as_ref().and_then(|(_, _, p)| {
                        Path::new(p).file_stem().map(|x| {
                            x.to_string_lossy().trim_end_matches(".model").to_string()
                        })
                    }).or(spec.warm_start_sha256.clone()),
                    warm.as_ref().map(|(_, _, p)| p.clone()),
                    warm.as_ref().map(|(a, _, _)| *a),
                    warm.as_ref().map(|(_, t, _)| *t),
                    ck_sha, sel_epoch, sel_metric_name, sel_metric, epochs_planned,
                    rows.len() as i64,
                    rows.first().map(|r| r.epoch),
                    rows.last().map(|r| r.epoch),
                    early_stopped as i64,
                    result.is_some() as i64,
                    started_ms, ended_ms, basis, timing.state,
                    metric_definition, spec.trainer_source_sha256, spec.train_script_sha256,
                    src_sha, sp.origin, spec.path, sp.notes,
                    {
                        // Does the ORIGINAL location still exist? For scq that is the
                        // run directory (which the GC prunes); for imported lanes it is
                        // the path the bytes were mirrored from.
                        let live = if sp.lane == "scq" {
                            Path::new(scq_runs_str()).join(&sp.run_id).is_dir()
                        } else {
                            sp.origin
                                .split_once(':')
                                .map(|(node, p)| node != "xbabe2" || Path::new(p).is_file())
                                .unwrap_or(false)
                        };
                        live as i64
                    }
                ],
            )?;
            total_runs += 1;

            // ---- per-epoch rows
            let mut stmt = tx.prepare(
                "INSERT INTO epoch(run_key,epoch,phase,row_ord,train_loss,val_loss,overfit_gap,lr,batch_size,epochs,
                                   elapsed_sec,epoch_sec,train_rows,validation_rows,model_version,device,world_size,
                                   val_fallback,macro_validation_best,macro_validation_early_stopped,
                                   val_conditional_advantage,val_conditional_score,val_pairwise_win_accuracy,
                                   val_binary_pairwise_win_accuracy,val_regression_pairwise_win_accuracy,
                                   val_ablation_pairwise_win,val_reward_cv_correlation,val_top1_historical_regret,
                                   val_seq_nll,val_head_accuracy_featmath,val_uncertainty_error_spearman,
                                   anti_overfit,wall_ms,raw_json,raw_sha256)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,
                        ?26,?27,?28,?29,?30,?31,?32,?33,?34,?35)
                 ON CONFLICT(run_key,epoch,phase) DO UPDATE SET
                   row_ord=excluded.row_ord, train_loss=excluded.train_loss, val_loss=excluded.val_loss,
                   overfit_gap=excluded.overfit_gap, lr=excluded.lr, batch_size=excluded.batch_size,
                   epochs=excluded.epochs, elapsed_sec=excluded.elapsed_sec, epoch_sec=excluded.epoch_sec,
                   train_rows=excluded.train_rows, validation_rows=excluded.validation_rows,
                   model_version=excluded.model_version, device=excluded.device, world_size=excluded.world_size,
                   val_fallback=excluded.val_fallback, macro_validation_best=excluded.macro_validation_best,
                   macro_validation_early_stopped=excluded.macro_validation_early_stopped,
                   val_conditional_advantage=excluded.val_conditional_advantage,
                   val_conditional_score=excluded.val_conditional_score,
                   val_pairwise_win_accuracy=excluded.val_pairwise_win_accuracy,
                   val_binary_pairwise_win_accuracy=excluded.val_binary_pairwise_win_accuracy,
                   val_regression_pairwise_win_accuracy=excluded.val_regression_pairwise_win_accuracy,
                   val_ablation_pairwise_win=excluded.val_ablation_pairwise_win,
                   val_reward_cv_correlation=excluded.val_reward_cv_correlation,
                   val_top1_historical_regret=excluded.val_top1_historical_regret,
                   val_seq_nll=excluded.val_seq_nll,
                   val_head_accuracy_featmath=excluded.val_head_accuracy_featmath,
                   val_uncertainty_error_spearman=excluded.val_uncertainty_error_spearman,
                   anti_overfit=excluded.anti_overfit, wall_ms=excluded.wall_ms,
                   raw_json=excluded.raw_json, raw_sha256=excluded.raw_sha256",
            )?;
            let mut fstmt = tx.prepare(
                "INSERT INTO epoch_field(run_key,epoch,phase,key,num,txt,json) VALUES(?1,?2,?3,?4,?5,?6,?7)
                 ON CONFLICT(run_key,epoch,phase,key) DO UPDATE SET num=excluded.num, txt=excluded.txt, json=excluded.json",
            )?;

            for r in &rows {
                let o = &r.obj;
                let wall = started_ms
                    .map(|st| st + (f(o.get("elapsed_sec")).unwrap_or(0.0) * 1000.0) as i64);
                stmt.execute(params![
                    run_key,
                    r.epoch,
                    r.phase,
                    r.row_ord,
                    f(o.get("train_loss")).or_else(|| f(o.get("loss"))),
                    f(o.get("val_loss")),
                    f(o.get("overfit_gap")),
                    f(o.get("lr")),
                    i(o.get("batch_size")),
                    i(o.get("epochs")),
                    f(o.get("elapsed_sec")),
                    f(o.get("epoch_sec")),
                    i(o.get("train_rows")),
                    i(o.get("validation_rows")),
                    s(o.get("model_version")),
                    s(o.get("device")),
                    i(o.get("world_size")),
                    b(o.get("val_fallback")),
                    f(o.get("macro_validation_best")),
                    b(o.get("macro_validation_early_stopped")),
                    f(o.get("val_conditional_advantage")),
                    f(o.get("val_conditional_score")),
                    f(o.get("val_pairwise_win_accuracy")),
                    f(o.get("val_binary_pairwise_win_accuracy")),
                    f(o.get("val_regression_pairwise_win_accuracy")),
                    f(o.get("val_ablation_pairwise_win")),
                    f(o.get("val_reward_cv_correlation")),
                    f(o.get("val_top1_historical_regret")),
                    f(o.get("val_seq_nll")),
                    f(o.get("val_head_accuracy_featmath")),
                    f(o.get("val_uncertainty_error_spearman")),
                    b(o.get("anti_overfit_experiment")),
                    wall,
                    r.raw,
                    sha256_hex(r.raw.as_bytes()),
                ])?;

                for (k, v) in o.iter() {
                    let (num, txt, json) = match v {
                        Value::Number(n) => (n.as_f64(), None, None),
                        Value::Bool(x) => (Some(*x as i64 as f64), None, None),
                        Value::String(x) => (None, Some(x.clone()), None),
                        Value::Null => (None, None, None),
                        other => (None, None, Some(other.to_string())),
                    };
                    fstmt.execute(params![run_key, r.epoch, r.phase, k, num, txt, json])?;
                    let kind = match v {
                        Value::Number(_) => "number",
                        Value::Bool(_) => "bool",
                        Value::String(_) => "string",
                        Value::Null => "null",
                        Value::Array(_) => "array",
                        Value::Object(_) => "object",
                    };
                    let e = field_stats.entry(k.clone()).or_insert_with(|| {
                        (
                            kind.to_string(),
                            BTreeSet::new(),
                            0,
                            i64::MAX,
                            i64::MIN,
                            String::new(),
                            String::new(),
                        )
                    });
                    e.1.insert(run_key.clone());
                    e.2 += 1;
                    let w = wall.unwrap_or(started);
                    if w < e.3 {
                        e.3 = w;
                        e.5 = run_key.clone();
                    }
                    if w > e.4 {
                        e.4 = w;
                        e.6 = run_key.clone();
                    }
                }
                total_rows += 1;
            }
        }

        for (k, (kind, runs, rows_n, first, last, frun, lrun)) in field_stats {
            tx.execute(
                "INSERT INTO field_catalog(key,kind,runs_present,rows_present,first_seen_ms,last_seen_ms,first_run_key,last_run_key)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
                params![k, kind, runs.len() as i64, rows_n, first, last, frun, lrun],
            )?;
        }

        // ---- annotations (curve-honesty markers)
        let anns: Vec<(&str, i64, &str, &str, &str, &str)> = vec![
            (
                "metric-fix-task-types",
                METRIC_FIX_MS,
                "metric_definition_change",
                "validation metrics: binary rows were scored by the REGRESSION heads before this point",
                "policy_v3_validation_metrics called model.reward_stats(z, token_ids, pad_mask) with no task_types, so _binary_task_mask defaulted every row to regression. Training was correct; MEASUREMENT was wrong. val_conditional_advantage, val_conditional_score and val_binary_pairwise_win_accuracy are NOT comparable across this line.",
                "jope-prime/src/jope/policy/train_v3.py mtime 2026-07-27T21:27:52Z. Runs a-g3 and a-g7 have byte-identical specs (diff = the job id line only: same train_v3.py provenance sha 358acf21..., same test_exact_train.py bc2652da..., same corpus snapshot-001805437-full_consolidation, same warm-start ff95a336...); a-g3 started 19:58:24Z and reports val_conditional_advantage -0.0602..-0.0869, a-g7 started 21:43:31Z and reports +0.0314..+0.0390. Every spec sets PYTHONPATH=~/jope-prime/src so the executed validation code is the LIVE file, not the staged provenance copy.",
            ),
            (
                "equal-quota-sampler",
                1_785_204_089_000,
                "data_distribution_change",
                "equal-quota sampler eq32v32c128 replaces full_consolidation / recent_plus_replay",
                "Validation loss magnitudes are not comparable across this line: the validation SET changed, not just the model.",
                "spec [[input]] name=manifest source path moves from training/snapshot-001805437-full_consolidation/ to training/snapshot-001805437-full_consolidation-eq32v32c128/ at a-g8.",
            ),
            (
                "anti-overfit-experiment",
                1_785_204_089_000,
                "regularisation_change",
                "PrimeAntiOverfitExperiment.v1 enabled (token dropout, embedding jitter, weight decay, dataset row cap)",
                "Runs with anti_overfit=1 carry different regularisation than the rest of the curve.",
                "anti_overfit_experiment key present only in a-g8, a-g9 and c-r000002 rows.",
            ),
        ];
        for (id, at, kind, label, detail, evidence) in anns {
            tx.execute(
                "INSERT INTO annotation(id,at_ms,kind,label,detail,evidence) VALUES(?1,?2,?3,?4,?5,?6)
                 ON CONFLICT(id) DO UPDATE SET at_ms=excluded.at_ms, kind=excluded.kind, label=excluded.label,
                   detail=excluded.detail, evidence=excluded.evidence",
                params![id, at, kind, label, detail, evidence],
            )?;
        }

        derive_annotations(&tx)?;

        // ---- proven-lost ledger, from the imported manifest
        if let Some(list) = manifest.get("proven_lost").and_then(|x| x.as_array()) {
            for (n, g) in list.iter().enumerate() {
                tx.execute(
                    "INSERT INTO gap(id,lane,what,evidence,approx_window) VALUES(?1,?2,?3,?4,?5)
                     ON CONFLICT(id) DO UPDATE SET lane=excluded.lane, what=excluded.what,
                       evidence=excluded.evidence, approx_window=excluded.approx_window",
                    params![
                        format!("gap-{n:02}"),
                        g["lane"].as_str().unwrap_or("?"),
                        g["what"].as_str().unwrap_or("?"),
                        g["evidence"].as_str().unwrap_or("?"),
                        g["approx_window"].as_str()
                    ],
                )?;
            }
        }

        // ---- sanity flags
        recompute_flags(&tx)?;
        truncated = ingest_truncations(&tx, root)?;
        for p in &all_problems {
            let (rk, det) = p.split_once(": ").unwrap_or(("*", p.as_str()));
            tx.execute(
                "INSERT OR REPLACE INTO flag(run_key,epoch,phase,code,severity,detail)
                 VALUES(?1,-1,'*','row_parse_anomaly','warn',?2)",
                params![rk, det],
            )?;
        }

        tx.execute(
            "INSERT INTO meta(key,value) VALUES('schema','JopeTrainingHistoryDb.v1')
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [],
        )?;
        tx.execute(
            "INSERT INTO meta(key,value) VALUES('last_ingest_ms',?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![started.to_string()],
        )?;
        tx.execute(
            "INSERT INTO meta(key,value) VALUES('metric_fix_ms',?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![METRIC_FIX_MS.to_string()],
        )?;
        tx.commit()?;
    }

    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    println!(
        "ingested {total_runs} runs, {total_rows} epoch rows -> {}",
        db_path.display()
    );
    if !all_problems.is_empty() {
        println!("{} row anomalies recorded as flags", all_problems.len());
    }
    if truncated > 0 {
        println!("{truncated} truncated tail(s) recorded as flags + gaps");
    }
    Ok(())
}

/// Annotations that must come from the DATA, not from a constant: model-version
/// transitions and the first appearance of each validation metric. Re-derived on
/// every ingest so a future schema change annotates itself.
fn derive_annotations(tx: &rusqlite::Transaction) -> Result<()> {
    // model_version transitions along the wall-clock run order
    let mut st = tx.prepare(
        "SELECT model_version, MIN(started_at_ms), MAX(COALESCE(ended_at_ms, started_at_ms))
         FROM run WHERE model_version IS NOT NULL AND lane_kind='epoch'
         GROUP BY model_version ORDER BY MIN(started_at_ms)",
    )?;
    let versions: Vec<(String, i64, i64)> = st
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(st);
    for w in versions.windows(2) {
        let (from, _, from_end) = &w[0];
        let (to, to_start, _) = &w[1];
        let at = (*from_end + *to_start) / 2;
        tx.execute(
            "INSERT INTO annotation(id,at_ms,kind,label,detail,evidence) VALUES(?1,?2,?3,?4,?5,?6)
             ON CONFLICT(id) DO UPDATE SET at_ms=excluded.at_ms, label=excluded.label,
               detail=excluded.detail, evidence=excluded.evidence",
            params![
                format!("model-version-{from}-to-{to}"),
                at,
                "model_version_change",
                format!("{from} -> {to}"),
                "Different architecture generation: loss magnitudes and head sets are not comparable across this line.",
                format!(
                    "last {from} run ends {}, first {to} run starts {} (run.model_version, derived at ingest)",
                    fmt_ms(Some(*from_end)),
                    fmt_ms(Some(*to_start))
                )
            ],
        )?;
    }

    // first appearance of each validation metric that is not universal
    let mut ks = tx.prepare(
        "SELECT key, first_seen_ms, first_run_key, runs_present FROM field_catalog
         WHERE key LIKE 'val\\_%' ESCAPE '\\' ORDER BY first_seen_ms",
    )?;
    let total_runs: i64 = tx.query_row("SELECT COUNT(*) FROM run", [], |r| r.get(0))?;
    let keys: Vec<(String, i64, String, i64)> = ks
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(ks);
    for (key, first_ms, first_run, runs_present) in keys {
        if runs_present >= total_runs {
            continue; // present everywhere: nothing changed
        }
        tx.execute(
            "INSERT INTO annotation(id,at_ms,kind,label,detail,evidence) VALUES(?1,?2,?3,?4,?5,?6)
             ON CONFLICT(id) DO UPDATE SET at_ms=excluded.at_ms, label=excluded.label,
               detail=excluded.detail, evidence=excluded.evidence",
            params![
                format!("metric-added-{key}"),
                first_ms,
                "metric_added",
                format!("{key} first emitted"),
                format!("Present in {runs_present} of {total_runs} archived runs. Earlier epochs have no value for it; a series that plots it must start here, not at the origin of the curve."),
                format!("field_catalog: first row carrying {key} is in {first_run} at {}", fmt_ms(Some(first_ms)))
            ],
        )?;
    }
    // pin the sampler / regularisation annotations to the first run that used them
    for (id, sql) in [
        (
            "equal-quota-sampler",
            "SELECT MIN(started_at_ms) FROM run WHERE sampler IS NOT NULL",
        ),
        (
            "anti-overfit-experiment",
            "SELECT MIN(started_at_ms) FROM run WHERE anti_overfit=1",
        ),
    ] {
        if let Ok(Some(at)) = tx.query_row(sql, [], |r| r.get::<_, Option<i64>>(0)) {
            tx.execute(
                "UPDATE annotation SET at_ms=?2 WHERE id=?1",
                params![id, at],
            )?;
        }
    }
    Ok(())
}

/// Project the capture layer's truncation ledger into `flag` + `gap`.
///
/// `training/history/truncations.v1.jsonl` is written by jope_training_history
/// whenever a trainer died mid-write and left a trailing INCOMPLETE JSONL row.
/// Under the operator policy that fragment is not data — the epoch never
/// finished being written — so every complete row is archived, the fragment is
/// kept verbatim next to the segment, and the run directory becomes prunable.
/// That decision has to be visible to anyone querying this database, or the
/// curve would silently read as if the run simply ended one epoch earlier.
fn ingest_truncations(tx: &rusqlite::Transaction, root: &Path) -> Result<usize> {
    let Ok(text) = fs::read_to_string(root.join("truncations.v1.jsonl")) else {
        return Ok(0);
    };
    let mut count = 0usize;
    for line in text.lines() {
        let Ok(record) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(run_key) = record["run_key"].as_str() else {
            continue;
        };
        let residual = record["residual_bytes"].as_u64().unwrap_or(0);
        let after = record["last_archived_epoch"]
            .as_i64()
            .map(|epoch| epoch.to_string())
            .unwrap_or_else(|| "?".into());
        let detail = format!(
            "writer stopped mid-row: {residual}B trailing INCOMPLETE line after epoch {after}. \
             Every complete row is archived; the fragment is kept verbatim as {} and the \
             source directory was released for pruning. Node {}, source idle {}s at record time.",
            record["evidence_file"].as_str().unwrap_or("?"),
            record["node"].as_str().unwrap_or("?"),
            record["source_idle_secs"].as_u64().unwrap_or(0),
        );
        tx.execute(
            "INSERT OR REPLACE INTO flag(run_key,epoch,phase,code,severity,detail)
             VALUES(?1,-1,'*','truncated_tail','warn',?2)",
            params![run_key, detail],
        )?;
        // Also a gap row: this is history that is proven never to have existed
        // in complete form, and the ledger is the evidence.
        tx.execute(
            "INSERT INTO gap(id,lane,what,evidence,approx_window) VALUES(?1,'scq',?2,?3,?4)
             ON CONFLICT(id) DO UPDATE SET what=excluded.what, evidence=excluded.evidence,
               approx_window=excluded.approx_window",
            params![
                format!("gap-trunc-{}", record["ts_ms"].as_u64().unwrap_or(0)),
                format!("{run_key}: final epoch after {after} was never written completely"),
                format!(
                    "truncations.v1.jsonl fragment sha256 {}",
                    record["fragment_sha256"].as_str().unwrap_or("?")
                ),
                record["ts_ms"].as_u64().map(|ms| ms.to_string()),
            ],
        )?;
        count += 1;
    }
    Ok(count)
}

fn recompute_flags(tx: &rusqlite::Transaction) -> Result<()> {
    let stmts = [
        // epoch-level
        ("INSERT OR REPLACE INTO flag SELECT e.run_key,e.epoch,e.phase,'val_loss_missing','error',
            'no val_loss on this epoch' FROM epoch e JOIN run r USING(run_key)
            WHERE e.val_loss IS NULL AND r.lane_kind='epoch'", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,epoch,phase,'val_fallback','warn',
            'trainer fell back to a substitute validation split' FROM epoch WHERE val_fallback=1", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,epoch,phase,'early_stopped','info',
            'macro_validation_early_stopped set on this epoch' FROM epoch WHERE macro_validation_early_stopped=1", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,epoch,phase,'val_loss_nonfinite','error',
            'val_loss is not finite' FROM epoch WHERE val_loss IS NOT NULL AND (val_loss!=val_loss OR val_loss>1e300)", ()),
        ("INSERT OR REPLACE INTO flag SELECT e.run_key,e.epoch,e.phase,'train_loss_missing','error',
            'no train_loss on this epoch' FROM epoch e JOIN run r USING(run_key)
            WHERE e.train_loss IS NULL AND r.lane_kind='epoch'", ()),
        // run-level
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','metric_definition_pre_fix','warn',
            'validation metrics computed with binary rows scored by the regression heads; val_conditional_advantage / val_binary_* not comparable to post-fix runs'
            FROM run WHERE metric_definition='pre_task_types_fix'", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','no_split_heads','info',
            'run predates the val_binary_*/val_regression_* split; only pooled pairwise accuracy exists'
            FROM run WHERE metric_definition='no_split_heads'", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','run_incomplete','warn',
            'no train-result.json: run was cancelled, preempted or is still live; no checkpoint was selected'
            FROM run WHERE completed=0 AND lane='scq'", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','early_stopped_run','info',
            'run early-stopped on macro validation' FROM run WHERE early_stopped=1", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','anti_overfit_experiment','info',
            'PrimeAntiOverfitExperiment.v1 active: token dropout, embedding jitter, weight decay, dataset row cap'
            FROM run WHERE anti_overfit=1", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','not_on_policy_curve','info',
            'step-granularity or different-objective lane; archived but excluded from the train/val epoch curve'
            FROM run WHERE in_policy_curve=0", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','wall_clock_derived','info',
            'start/end wall time derived from file mtime minus elapsed_sec, not from an scq event'
            FROM run WHERE time_basis='file_mtime-elapsed_sec'", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','epoch_axis_not_from_one','warn',
            'first archived epoch is not 1: earlier epochs of this stream were never captured'
            FROM run WHERE lane_kind='epoch' AND first_epoch IS NOT NULL AND first_epoch>1", ()),
        ("INSERT OR REPLACE INTO flag SELECT run_key,-1,'*','source_pruned','warn',
            'originating run directory / file no longer exists; this archive is the only copy'
            FROM run WHERE source_live=0", ()),
    ];
    for (sql, _) in stmts {
        tx.execute(sql, [])?;
    }

    // step lanes: one run-level note instead of one flag per step row
    tx.execute(
        "INSERT OR REPLACE INTO flag
         SELECT r.run_key,-1,'*','step_lane_val_sparse','info',
                'step-granularity lane: ' || SUM(CASE WHEN e.val_loss IS NULL THEN 1 ELSE 0 END)
                || ' of ' || COUNT(*) || ' rows carry no val_loss (validation runs on a slower cadence)'
         FROM run r JOIN epoch e USING(run_key)
         WHERE r.lane_kind='step' GROUP BY r.run_key",
        [],
    )?;

    // epoch-count gaps inside a run
    tx.execute(
        "INSERT OR REPLACE INTO flag
         SELECT r.run_key,-1,'*','epoch_gap','warn',
                'epochs recorded (' || r.rows_recorded || ') < span ' || r.first_epoch || '..' || r.last_epoch
         FROM run r
         WHERE r.lane_kind='epoch' AND r.first_epoch IS NOT NULL
           AND r.rows_recorded < (r.last_epoch - r.first_epoch + 1)",
        [],
    )?;
    Ok(())
}

// ------------------------------------------------------------------ reports

fn open_db(root: &Path) -> Result<Connection> {
    let p = root.join("db/history.sqlite3");
    if !p.is_file() {
        bail!("no archive at {} — run `ingest` first", p.display());
    }
    Ok(Connection::open(p)?)
}

fn cmd_stats(root: &Path) -> Result<()> {
    let conn = open_db(root)?;
    let (runs, rows): (i64, i64) = conn.query_row(
        "SELECT (SELECT COUNT(*) FROM run),(SELECT COUNT(*) FROM epoch)",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    println!("archive: {}", root.join("db/history.sqlite3").display());
    println!("runs={runs}  epoch_rows={rows}");
    println!("\nby lane:");
    let mut st = conn.prepare(
        "SELECT lane, lane_kind, in_policy_curve, COUNT(*), SUM(rows_recorded),
                MIN(started_at_ms), MAX(ended_at_ms)
         FROM run GROUP BY lane, lane_kind, in_policy_curve ORDER BY MIN(started_at_ms)",
    )?;
    println!(
        "{:<22} {:<6} {:<6} {:>5} {:>7}  window (UTC)",
        "lane", "kind", "curve", "runs", "rows"
    );
    for r in st.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, i64>(3)?,
            r.get::<_, i64>(4)?,
            r.get::<_, Option<i64>>(5)?,
            r.get::<_, Option<i64>>(6)?,
        ))
    })? {
        let (lane, kind, curve, n, rows, a, b2) = r?;
        println!(
            "{:<22} {:<6} {:<6} {:>5} {:>7}  {} .. {}",
            lane,
            kind,
            if curve == 1 { "yes" } else { "no" },
            n,
            rows,
            fmt_ms(a),
            fmt_ms(b2)
        );
    }
    println!("\npolicy curve (in_policy_curve=1, lane_kind='epoch'):");
    let (pr, pe): (i64, i64) = conn.query_row(
        "SELECT COUNT(DISTINCT r.run_key), COUNT(*) FROM epoch e JOIN run r USING(run_key)
         WHERE r.in_policy_curve=1 AND r.lane_kind='epoch'",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    println!("  {pr} runs, {pe} epochs");
    println!(
        "\ndistinct metric keys tracked: {}",
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM field_catalog", [], |r| r.get(0))?
    );
    println!(
        "epoch_field cells:            {}",
        conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM epoch_field", [], |r| r.get(0))?
    );
    println!("\nflags:");
    let mut fs2 = conn.prepare("SELECT severity,code,COUNT(*) FROM flag GROUP BY severity,code ORDER BY severity DESC, COUNT(*) DESC")?;
    for r in fs2.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
        ))
    })? {
        let (sev, code, n) = r?;
        println!("  {:<6} {:<34} {}", sev, code, n);
    }
    println!("\nknown-lost history:");
    let mut gs = conn.prepare("SELECT lane,what,approx_window FROM gap ORDER BY id")?;
    for r in gs.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Option<String>>(2)?,
        ))
    })? {
        let (lane, what, w) = r?;
        println!(
            "  [{}] {} ({})",
            lane,
            what,
            w.unwrap_or_else(|| "?".into())
        );
    }
    Ok(())
}

fn fmt_ms(ms: Option<i64>) -> String {
    let Some(ms) = ms else { return "-".into() };
    // civil_from_days
    let secs = ms / 1000;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn cmd_runs(root: &Path, lane: Option<String>) -> Result<()> {
    let conn = open_db(root)?;
    let sql = "SELECT run_key,lane,node,model_version,recipe,sampler,corpus_snapshot,anti_overfit,
                      warm_start_sha256,warm_start_tensors_loaded,warm_start_tensors_total,
                      result_checkpoint_sha256,selected_checkpoint_epoch,rows_recorded,
                      first_epoch,last_epoch,started_at_ms,ended_at_ms,metric_definition,
                      early_stopped,completed,scq_state,time_basis
               FROM run WHERE (?1 IS NULL OR lane=?1) ORDER BY started_at_ms";
    let mut st = conn.prepare(sql)?;
    let mut n = 0;
    for row in st.query_map(params![lane], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Option<String>>(2)?,
            r.get::<_, Option<String>>(3)?,
            r.get::<_, Option<String>>(4)?,
            r.get::<_, Option<String>>(5)?,
            r.get::<_, Option<String>>(6)?,
            r.get::<_, Option<i64>>(7)?,
            r.get::<_, Option<String>>(8)?,
            r.get::<_, Option<i64>>(9)?,
            r.get::<_, Option<i64>>(10)?,
            r.get::<_, Option<String>>(11)?,
            r.get::<_, Option<i64>>(12)?,
            r.get::<_, i64>(13)?,
            r.get::<_, Option<i64>>(14)?,
            r.get::<_, Option<i64>>(15)?,
            r.get::<_, Option<i64>>(16)?,
            r.get::<_, Option<i64>>(17)?,
            r.get::<_, Option<String>>(18)?,
            r.get::<_, Option<i64>>(19)?,
            r.get::<_, Option<i64>>(20)?,
            r.get::<_, Option<String>>(21)?,
            r.get::<_, Option<String>>(22)?,
        ))
    })? {
        let t = row?;
        n += 1;
        println!("── {}", t.0);
        println!(
            "   started {}  ended {}  ({})",
            fmt_ms(t.16),
            fmt_ms(t.17),
            t.22.unwrap_or_default()
        );
        println!(
            "   node={:<8} model={:<26} rows={} epochs {}..{}",
            t.2.unwrap_or_else(|| "?".into()),
            t.3.unwrap_or_else(|| "?".into()),
            t.13,
            t.14.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            t.15.map(|x| x.to_string()).unwrap_or_else(|| "?".into())
        );
        println!(
            "   recipe={:<4} sampler={:<12} corpus={}",
            t.4.unwrap_or_else(|| "-".into()),
            t.5.unwrap_or_else(|| "-".into()),
            t.6.unwrap_or_else(|| "-".into())
        );
        println!(
            "   warm_start={} ({}/{} tensors)",
            t.8.as_deref().map(|x| &x[..x.len().min(16)]).unwrap_or("-"),
            t.9.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            t.10.map(|x| x.to_string()).unwrap_or_else(|| "?".into())
        );
        println!("   result_ckpt={} selected_epoch={} anti_overfit={} early_stopped={} complete={} scq={}",
                 t.11.as_deref().map(|x| &x[..x.len().min(16)]).unwrap_or("-"),
                 t.12.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
                 t.7.unwrap_or(0), t.19.unwrap_or(0), t.20.unwrap_or(0),
                 t.21.unwrap_or_else(|| "-".into()));
        println!(
            "   metric_definition={}",
            t.18.unwrap_or_else(|| "?".into())
        );
    }
    println!("\n{n} runs");
    Ok(())
}

fn cmd_flags(root: &Path, severity: Option<String>) -> Result<()> {
    let conn = open_db(root)?;
    let mut st = conn.prepare(
        "SELECT severity,code,run_key,epoch,phase,detail FROM flag
         WHERE (?1 IS NULL OR severity=?1)
         ORDER BY CASE severity WHEN 'error' THEN 0 WHEN 'warn' THEN 1 ELSE 2 END, code, run_key, epoch",
    )?;
    let mut n = 0;
    for r in st.query_map(params![severity], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, i64>(3)?,
            r.get::<_, String>(4)?,
            r.get::<_, Option<String>>(5)?,
        ))
    })? {
        let (sev, code, rk, ep, ph, det) = r?;
        n += 1;
        let loc = if ep < 0 {
            "run".to_string()
        } else {
            format!("e{ep}/{ph}")
        };
        println!(
            "{:<6} {:<32} {:<52} {:<10} {}",
            sev,
            code,
            rk,
            loc,
            det.unwrap_or_default()
        );
    }
    println!("\n{n} flags");
    Ok(())
}

fn cmd_verify(root: &Path) -> Result<()> {
    let conn = open_db(root)?;
    let mut st = conn.prepare("SELECT run_key,epoch,phase,raw_json,raw_sha256 FROM epoch")?;
    let mut bad = 0;
    let mut n = 0;
    for r in st.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, String>(4)?,
        ))
    })? {
        let (rk, ep, ph, raw, sha) = r?;
        n += 1;
        if sha256_hex(raw.as_bytes()) != sha {
            bad += 1;
            println!("HASH MISMATCH {rk} e{ep}/{ph}");
        }
    }
    println!("verified {n} rows, {bad} hash mismatches");

    // reconcile against sources that are still readable
    let mut ss =
        conn.prepare("SELECT run_key,source_sha256,source_origin,rows_recorded FROM run")?;
    let mut drift = 0;
    for r in ss.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, Option<String>>(1)?,
            r.get::<_, Option<String>>(2)?,
            r.get::<_, i64>(3)?,
        ))
    })? {
        let (rk, sha, origin, rows) = r?;
        let Some(origin) = origin else { continue };
        let Some((node, path)) = origin.split_once(':') else {
            continue;
        };
        if node != "xbabe2" {
            continue;
        }
        let p = Path::new(path);
        if !p.is_file() {
            println!("source gone (archive is the only copy): {rk} -> {path}");
            continue;
        }
        let bytes = fs::read(p)?;
        let live_rows = String::from_utf8_lossy(&bytes)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count() as i64;
        let live_sha = sha256_hex(&bytes);
        if Some(&live_sha) != sha.as_ref() {
            if live_rows > rows {
                println!("source ADVANCED (re-ingest): {rk} {rows} -> {live_rows} rows");
            } else {
                println!("source CHANGED: {rk} archived {rows} rows, live {live_rows}");
            }
            drift += 1;
        }
    }
    println!("{drift} sources drifted from the archived copy");
    if bad > 0 {
        bail!("{bad} rows failed hash verification");
    }
    Ok(())
}

// ------------------------------------------------------------------ export

#[derive(Debug)]
struct Point {
    gidx: i64,
    run_key: String,
    lane: String,
    epoch: i64,
    phase: String,
    train_loss: Option<f64>,
    val_loss: Option<f64>,
    overfit_gap: Option<f64>,
    val_pairwise: Option<f64>,
    val_cond_adv: Option<f64>,
    wall_ms: Option<i64>,
    metric_definition: String,
    val_fallback: i64,
    early_stopped: i64,
}

fn load_points(conn: &Connection, policy_only: bool) -> Result<Vec<Point>> {
    let sql = format!(
        "SELECT e.run_key, r.lane, e.epoch, e.phase, e.train_loss, e.val_loss, e.overfit_gap,
                e.val_pairwise_win_accuracy, e.val_conditional_advantage, e.wall_ms,
                r.metric_definition, COALESCE(e.val_fallback,0), COALESCE(e.macro_validation_early_stopped,0)
         FROM epoch e JOIN run r USING(run_key)
         {}
         ORDER BY r.started_at_ms, r.run_key, e.epoch, e.row_ord",
        if policy_only { "WHERE r.in_policy_curve=1 AND r.lane_kind='epoch'" } else { "" }
    );
    let mut st = conn.prepare(&sql)?;
    let mut out = vec![];
    for (gidx, r) in (1i64..).zip(st.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, Option<f64>>(4)?,
            r.get::<_, Option<f64>>(5)?,
            r.get::<_, Option<f64>>(6)?,
            r.get::<_, Option<f64>>(7)?,
            r.get::<_, Option<f64>>(8)?,
            r.get::<_, Option<i64>>(9)?,
            r.get::<_, Option<String>>(10)?,
            r.get::<_, i64>(11)?,
            r.get::<_, i64>(12)?,
        ))
    })?) {
        let t = r?;
        out.push(Point {
            gidx,
            run_key: t.0,
            lane: t.1,
            epoch: t.2,
            phase: t.3,
            train_loss: t.4,
            val_loss: t.5,
            overfit_gap: t.6,
            val_pairwise: t.7,
            val_cond_adv: t.8,
            wall_ms: t.9,
            metric_definition: t.10.unwrap_or_else(|| "unknown".into()),
            val_fallback: t.11,
            early_stopped: t.12,
        });
    }
    Ok(out)
}

fn cmd_export(root: &Path, format: &str, out: Option<PathBuf>, policy_only: bool) -> Result<()> {
    let conn = open_db(root)?;
    let outdir = out.unwrap_or_else(|| root.join("db/export"));
    fs::create_dir_all(&outdir)?;
    let pts = load_points(&conn, policy_only)?;

    // run boundaries in global-index space
    let mut boundaries: Vec<(i64, String, String, String)> = vec![]; // gidx, run_key, recipe/sampler label, metric_definition
    let mut last_run = String::new();
    for p in &pts {
        if p.run_key != last_run {
            let label: (Option<String>, Option<String>, Option<String>, Option<i64>) = conn
                .query_row(
                "SELECT recipe, sampler, corpus_snapshot, anti_overfit FROM run WHERE run_key=?1",
                params![p.run_key],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )?;
            let short = p
                .run_key
                .rsplit('/')
                .nth(2)
                .unwrap_or(&p.run_key)
                .to_string();
            let ann = format!(
                "{}{}{}",
                label.0.map(|x| format!("recipe={x} ")).unwrap_or_default(),
                label.1.map(|x| format!("sampler={x} ")).unwrap_or_default(),
                if label.3 == Some(1) {
                    "anti_overfit "
                } else {
                    ""
                }
            );
            boundaries.push((
                p.gidx,
                short,
                ann.trim().to_string(),
                p.metric_definition.clone(),
            ));
            last_run = p.run_key.clone();
        }
    }

    if format == "json" || format == "both" {
        let mut runs_json = vec![];
        let mut st = conn.prepare(
            "SELECT run_key,lane,run_id,attempt,watermark,node,model_version,recipe,sampler,corpus_snapshot,
                    anti_overfit,warm_start_sha256,warm_start_tensors_loaded,warm_start_tensors_total,
                    result_checkpoint_sha256,selected_checkpoint_epoch,selected_checkpoint_metric_name,
                    selected_checkpoint_metric,rows_recorded,first_epoch,last_epoch,started_at_ms,ended_at_ms,
                    time_basis,scq_state,metric_definition,early_stopped,completed,in_policy_curve,lane_kind,
                    source_origin,notes
             FROM run ORDER BY started_at_ms",
        )?;
        for r in st.query_map([], |r| {
            let mut m = Map::new();
            for (idx, name) in [
                "run_key",
                "lane",
                "run_id",
                "attempt",
                "watermark",
                "node",
                "model_version",
                "recipe",
                "sampler",
                "corpus_snapshot",
                "anti_overfit",
                "warm_start_sha256",
                "warm_start_tensors_loaded",
                "warm_start_tensors_total",
                "result_checkpoint_sha256",
                "selected_checkpoint_epoch",
                "selected_checkpoint_metric_name",
                "selected_checkpoint_metric",
                "rows_recorded",
                "first_epoch",
                "last_epoch",
                "started_at_ms",
                "ended_at_ms",
                "time_basis",
                "scq_state",
                "metric_definition",
                "early_stopped",
                "completed",
                "in_policy_curve",
                "lane_kind",
                "source_origin",
                "notes",
            ]
            .iter()
            .enumerate()
            {
                let v = r.get_ref(idx)?;
                m.insert(
                    (*name).to_string(),
                    match v {
                        rusqlite::types::ValueRef::Null => Value::Null,
                        rusqlite::types::ValueRef::Integer(x) => Value::from(x),
                        rusqlite::types::ValueRef::Real(x) => Value::from(x),
                        rusqlite::types::ValueRef::Text(x) => {
                            Value::from(String::from_utf8_lossy(x).to_string())
                        }
                        rusqlite::types::ValueRef::Blob(_) => Value::Null,
                    },
                );
            }
            m.insert(
                "started_at".into(),
                Value::from(fmt_ms(m["started_at_ms"].as_i64())),
            );
            m.insert(
                "ended_at".into(),
                Value::from(fmt_ms(m["ended_at_ms"].as_i64())),
            );
            Ok(Value::Object(m))
        })? {
            runs_json.push(r?);
        }

        let series: Vec<Value> = pts
            .iter()
            .map(|p| {
                let mut m = Map::new();
                m.insert("gidx".into(), Value::from(p.gidx));
                m.insert("run_key".into(), Value::from(p.run_key.clone()));
                m.insert("lane".into(), Value::from(p.lane.clone()));
                m.insert("epoch".into(), Value::from(p.epoch));
                m.insert("phase".into(), Value::from(p.phase.clone()));
                m.insert(
                    "train_loss".into(),
                    p.train_loss.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "val_loss".into(),
                    p.val_loss.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "overfit_gap".into(),
                    p.overfit_gap.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "val_pairwise_win_accuracy".into(),
                    p.val_pairwise.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "val_conditional_advantage".into(),
                    p.val_cond_adv.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "wall_ms".into(),
                    p.wall_ms.map(Value::from).unwrap_or(Value::Null),
                );
                m.insert(
                    "metric_definition".into(),
                    Value::from(p.metric_definition.clone()),
                );
                m.insert("val_fallback".into(), Value::from(p.val_fallback));
                m.insert("early_stopped".into(), Value::from(p.early_stopped));
                Value::Object(m)
            })
            .collect();

        let mut anns = vec![];
        let mut ast = conn
            .prepare("SELECT id,at_ms,kind,label,detail,evidence FROM annotation ORDER BY at_ms")?;
        for r in ast.query_map([], |r| {
            let mut m = Map::new();
            m.insert("id".into(), Value::from(r.get::<_, String>(0)?));
            let at: i64 = r.get(1)?;
            m.insert("at_ms".into(), Value::from(at));
            m.insert("at".into(), Value::from(fmt_ms(Some(at))));
            m.insert("kind".into(), Value::from(r.get::<_, String>(2)?));
            m.insert("label".into(), Value::from(r.get::<_, String>(3)?));
            m.insert("detail".into(), Value::from(r.get::<_, String>(4)?));
            m.insert(
                "evidence".into(),
                Value::from(r.get::<_, Option<String>>(5)?.unwrap_or_default()),
            );
            Ok(Value::Object(m))
        })? {
            anns.push(r?);
        }

        let mut gaps = vec![];
        let mut gst =
            conn.prepare("SELECT id,lane,what,evidence,approx_window FROM gap ORDER BY id")?;
        for r in gst.query_map([], |r| {
            let mut m = Map::new();
            m.insert("id".into(), Value::from(r.get::<_, String>(0)?));
            m.insert("lane".into(), Value::from(r.get::<_, String>(1)?));
            m.insert("what".into(), Value::from(r.get::<_, String>(2)?));
            m.insert("evidence".into(), Value::from(r.get::<_, String>(3)?));
            m.insert(
                "approx_window".into(),
                Value::from(r.get::<_, Option<String>>(4)?.unwrap_or_default()),
            );
            Ok(Value::Object(m))
        })? {
            gaps.push(r?);
        }

        let bounds: Vec<Value> = boundaries
            .iter()
            .map(|(g, run, ann, md)| {
                let mut m = Map::new();
                m.insert("gidx".into(), Value::from(*g));
                m.insert("run".into(), Value::from(run.clone()));
                m.insert("annotation".into(), Value::from(ann.clone()));
                m.insert("metric_definition".into(), Value::from(md.clone()));
                Value::Object(m)
            })
            .collect();

        let doc = serde_json::json!({
            "schema": "JopeTrainingHistorySeries.v1",
            "generated_at_ms": now_ms(),
            "archive": root.join("db/history.sqlite3").display().to_string(),
            "policy_only": policy_only,
            "counts": {"runs": runs_json.len(), "points": series.len()},
            "runs": runs_json,
            "run_boundaries": bounds,
            "annotations": anns,
            "known_lost": gaps,
            "series": series,
        });
        let p = outdir.join("train-val-series.v1.json");
        fs::write(&p, serde_json::to_vec_pretty(&doc)?)?;
        println!("wrote {}", p.display());
    }

    if format == "pgfplots" || format == "both" {
        // main table
        let p = outdir.join("loss.dat");
        let mut w = fs::File::create(&p)?;
        writeln!(w, "# gidx epoch train_loss val_loss overfit_gap val_pairwise val_cond_adv run metric_definition")?;
        for pt in &pts {
            writeln!(
                w,
                "{} {} {} {} {} {} {} {} {}",
                pt.gidx,
                pt.epoch,
                pt.train_loss
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.val_loss
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.overfit_gap
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.val_pairwise
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.val_cond_adv
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.run_key.replace(' ', "_"),
                pt.metric_definition,
            )?;
        }
        println!("wrote {}", p.display());

        // per-run files so each generation can be drawn as its own line
        let seriesdir = outdir.join("series");
        fs::create_dir_all(&seriesdir)?;
        let mut cur: Option<(String, fs::File)> = None;
        for pt in &pts {
            let fname = pt.run_key.replace('/', "__") + ".dat";
            let need_new = cur.as_ref().map(|(n, _)| n != &fname).unwrap_or(true);
            if need_new {
                let mut fh = fs::File::create(seriesdir.join(&fname))?;
                writeln!(fh, "# gidx epoch train_loss val_loss")?;
                cur = Some((fname, fh));
            }
            let (_, fh) = cur.as_mut().unwrap();
            writeln!(
                fh,
                "{} {} {} {}",
                pt.gidx,
                pt.epoch,
                pt.train_loss
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
                pt.val_loss
                    .map(|x| format!("{x:.9}"))
                    .unwrap_or_else(|| "nan".into()),
            )?;
        }
        println!(
            "wrote {}/*.dat ({} runs)",
            seriesdir.display(),
            boundaries.len()
        );

        // boundary + annotation tables
        let bp = outdir.join("boundaries.dat");
        let mut bw = fs::File::create(&bp)?;
        writeln!(bw, "# gidx run annotation metric_definition")?;
        for (g, run, ann, md) in &boundaries {
            writeln!(
                bw,
                "{} {} {} {}",
                g,
                run,
                if ann.is_empty() { "-" } else { ann },
                md
            )?;
        }
        println!("wrote {}", bp.display());

        // the .tex
        let tex = build_tex(&pts, &boundaries);
        let tp = outdir.join("train-val-loss.tex");
        fs::write(&tp, tex)?;
        println!("wrote {}", tp.display());
    }
    Ok(())
}

fn build_tex(pts: &[Point], boundaries: &[(i64, String, String, String)]) -> String {
    let maxg = pts.last().map(|p| p.gidx).unwrap_or(1);
    let mut marks = String::new();
    // 91 run boundaries would overplot the axis. Draw a rule at every boundary,
    // but only LABEL the ones that carry information: a lane change, or a
    // recipe/sampler/regularisation change.
    let lane_of = |gidx: i64| pts.iter().find(|p| p.gidx == gidx).map(|p| p.lane.clone());
    let mut prev_lane: Option<String> = None;
    for (g, run, ann, md) in boundaries {
        let style = if md == "pre_task_types_fix" {
            "dashed, red!55"
        } else {
            "dotted, black!30"
        };
        marks.push_str(&format!(
            "    \\draw[{style}] (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymax}});\n"
        ));
        let lane = lane_of(*g);
        let lane_changed = lane != prev_lane;
        prev_lane = lane.clone();
        if lane_changed || !ann.is_empty() {
            marks.push_str(&format!(
                "    \\node[rotate=90, anchor=south west, font=\\tiny, black!65] at (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymin}}) {{{}{}{}}};\n",
                lane.map(|l| format!("{}: ", tex_escape(&l))).unwrap_or_default(),
                tex_escape(run),
                if ann.is_empty() { String::new() } else { format!(" ({})", tex_escape(ann)) }
            ));
        }
    }
    // the metric-definition fix boundary: first post-fix point
    let fix_g = pts
        .iter()
        .find(|p| p.metric_definition == "post_task_types_fix")
        .map(|p| p.gidx);
    let fix_mark = match fix_g {
        Some(g) => format!(
            "    \\draw[very thick, blue!70] (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymax}});\n    \\node[anchor=south east, font=\\scriptsize, blue!70] at (axis cs:{g},\\pgfkeysvalueof{{/pgfplots/ymax}}) {{validation-metric fix (task\\_types)}};\n"
        ),
        None => String::new(),
    };
    format!(
        r#"% Generated by jope_training_archive export --format pgfplots.
% Build:  pdflatex train-val-loss.tex   (needs loss.dat in the same directory)
\documentclass[tikz,border=6pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usepgfplotslibrary{{groupplots}}
\begin{{document}}
\begin{{tikzpicture}}
  \begin{{axis}}[
    width=22cm, height=10cm,
    xlabel={{cumulative epoch across all archived runs}},
    ylabel={{loss}},
    xmin=1, xmax={maxg},
    ymode=log,
    legend pos=north east,
    legend cell align=left,
    grid=both, grid style={{black!12}},
    title={{jope policy DL training --- train vs validation loss, complete archived history}},
  ]
    \addplot[blue, thick, mark=none, unbounded coords=jump]
      table[x index=0, y index=2, col sep=space, header=false] {{loss.dat}};
    \addlegendentry{{train\_loss}}
    \addplot[orange, thick, mark=none, unbounded coords=jump]
      table[x index=0, y index=3, col sep=space, header=false] {{loss.dat}};
    \addlegendentry{{val\_loss}}
{marks}{fix_mark}  \end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"#
    )
}

fn tex_escape(s: &str) -> String {
    s.replace('\\', "\\textbackslash{}")
        .replace('_', "\\_")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('#', "\\#")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Ingest { dry_run } => ingest(&cli.root, dry_run),
        Cmd::Verify => cmd_verify(&cli.root),
        Cmd::Stats => cmd_stats(&cli.root),
        Cmd::Runs { lane } => cmd_runs(&cli.root, lane),
        Cmd::Flags { severity } => cmd_flags(&cli.root, severity),
        Cmd::Export {
            format,
            out,
            policy_only,
        } => cmd_export(&cli.root, &format, out, policy_only),
    }
}
