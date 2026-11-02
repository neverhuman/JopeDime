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

mod commands;
mod ingest;
mod sources;

pub(super) use commands::*;
pub(super) use ingest::*;
pub(super) use sources::*;

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
