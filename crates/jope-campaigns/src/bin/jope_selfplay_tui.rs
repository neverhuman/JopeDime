//! Read-only, Rust-native operations console for JOPE self-play.

use anyhow::{bail, Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use jope_dime_regressor::four_task::{
    task_metric_contract, FourTaskTelemetryRow, JopeFourTaskTelemetryV1, FOUR_TASK_ORDER,
};
use jope_dime_regressor::paths;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, List, ListItem, Paragraph, Row,
    Table, Wrap,
};
use ratatui::{Frame, Terminal};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[path = "../tui_dashboard.rs"]
mod tui_dashboard;

jope_paths::once_str!(
    fn default_root_str => paths::flywheel_root()
);
jope_paths::once_str!(
    fn default_scq_str => paths::scq_binary()
);
jope_paths::once_str!(
    fn default_metrics_root_str => paths::scq_runs_root()
);
const DEFAULT_REFRESH_MS: u64 = 250;
const CPU_TARGET_PCT: f64 = 80.0;
const CPU_HARD_GUARD_PCT: f64 = 95.0;
const EVENT_TAIL_BYTES: u64 = 64 * 1024;
const EXPECTED_NODES: [&str; 5] = ["xbabe1", "xbabe2", "xbabe3", "xbabe4", "xbabe5"];
const GPU_MODEL_HINTS: [(&str, &str); 3] =
    [("xbabe1", "4090"), ("xbabe2", "3090"), ("xbabe3", "3090")];
const STALE_MS: u128 = 90_000;
const GAMEPLAY_RING: usize = 1_000;
const GAMEPLAY_FEED_RENDER: usize = 200;
const GAMEPLAY_STATS_WINDOW: usize = 200;
const RECEIPT_DIRS_PER_PASS: usize = 20;
const RECEIPT_PREFILL_WINDOW_MS: u128 = 15 * 60 * 1_000;
const RECEIPT_PREFILL_MAX_DIRS: usize = 200;
const SLOW_PASS_DEADLINE_MS: u128 = 200;
const MANIFEST_CACHE_MAX: usize = 64;
const DL_HISTORY_MAX: usize = 512;
const GENERATIONS_MAX: usize = 8;
/// Trainer generations carried on the cross-generation DL charts. The series is
/// rebuilt from RUN DIRECTORIES on disk, never from the live job id, so a
/// trainer completing (a-g8) and its successor appearing under a different id
/// (c-r000002 → a-g9) appends to the same curve instead of blanking it.
const DL_GENERATIONS_MAX: usize = 8;
/// Per-generation downsample cap on the continuous series (charts are ~120
/// columns wide, so more points than this buy nothing but read cost).
const DL_RUN_POINTS_MAX: usize = 128;
/// The one identity a DL trainer is recognised by, everywhere in this file: the
/// SCQ job id, which is also the run-directory name.
///
/// It is deliberately NOT the SCQ `project` field. On 2026-07-28 the DL policy
/// proposal driver was given `project = "jope-selfplay-train"` so it would
/// inherit the priority-10 lane; a `project.contains("train")` test then
/// classified it as a trainer, it took arm 0, and the DL tab header rendered
/// `live jope-selfplay-propose-… [NO DATA]` with an empty TRAINER STATE box
/// while a-g9 was training at epoch 14/24. Keying on the id prefix also makes
/// the panel self-consistent: an arm flagged `is_trainer` is exactly one whose
/// run dir feeds `collect_dl_history`.
/// Every id prefix a DL trainer run has ever used.
///
/// This must be a LIST, because the naming changed and the chart silently died
/// when it did. The task-aware shadow trainer emits
/// `jope-selfplay-shadow-<task>-<watermark>-s<seq>`, not the legacy
/// `jope-selfplay-train-<watermark>-<recipe>`. With only the legacy prefix
/// matched, `collect_dl_history` skipped every current run dir, so the loss
/// panel froze on the last legacy run -- **2026-07-29 08:52, six days stale** --
/// while a trainer was actively writing `train_metrics.jsonl` under the shadow
/// name. It kept redrawing, so it looked alive; it just never fell, because it
/// was not plotting the training that was happening.
///
/// The job's scq *project* is `jope-selfplay-train` for both, but the run dir is
/// named from the job id, so the project name cannot be used here.
const TRAINER_ID_PREFIXES: [&str; 2] = ["jope-selfplay-train-", "jope-selfplay-shadow-"];

/// Is this run id a DL trainer, under any of its historical names?
fn is_trainer_run_id(id: &str) -> bool {
    TRAINER_ID_PREFIXES
        .iter()
        .any(|prefix| id.starts_with(prefix))
}

#[cfg(test)]
mod trainer_id_tests {
    use super::is_trainer_run_id;

    /// The loss chart froze for six days because the trainer was renamed and this
    /// predicate was not. Both schemes must be recognised, and neither the
    /// proposer nor the eval lane may be mistaken for a trainer.
    #[test]
    fn both_trainer_naming_schemes_feed_the_loss_chart() {
        // Current: the task-aware shadow trainer (live 2026-08-04).
        assert!(is_trainer_run_id(
            "jope-selfplay-shadow-regression-008145630-s000014"
        ));
        assert!(is_trainer_run_id(
            "jope-selfplay-shadow-binary-005428139-s000010"
        ));
        // Legacy: still on disk, still worth plotting.
        assert!(is_trainer_run_id("jope-selfplay-train-004754860-c-r000037"));

        // Everything else must stay out of the DL history.
        assert!(!is_trainer_run_id(
            "jope-selfplay-cohort-propose-fbcd5459f2c1-0bff466581ab"
        ));
        assert!(!is_trainer_run_id("jte-051a60090d757cbbcf22cb66"));
        assert!(!is_trainer_run_id("jcv-3e506ffe465718c5c"));
        assert!(!is_trainer_run_id("jope-selfplay-coverage-001"));
    }
}
/// Newest seeding-curve generations rendered in the tab-4 generations panel.
const SEEDING_CURVE_ROWS: usize = 3;
/// (ops status file, expected write cadence seconds) for the DL tab heartbeat panel.
const DAEMON_HEARTBEATS: [(&str, u64); 5] = [
    ("ops/queue-keeper-v3.json", 450),
    ("ops/micro-wave-supply-v2.json", 60),
    // 2026-08-01: was ops/trainer-watchdog-v1.json, which lost its writer when
    // the watchdog unit was switched to jope_task_learning_controller; trainer
    // liveness now lives in the controller status doc (30 s cycle).
    ("ops/task-learning-controller.v2.json", 60),
    ("ops/last-ingest.json", 450),
    ("ops/freshness-status.json", 150),
];

mod cli;
mod formatting;
mod funnel;
mod history;
mod models;
mod receipts;
mod slow;
mod sources;
mod telemetry;
mod terminal;
mod trainer_metrics;
mod waves;

use cli::*;
use formatting::*;
use funnel::*;
use history::*;
use models::*;
use receipts::*;
use slow::*;
use sources::*;
use telemetry::*;
use terminal::*;
use trainer_metrics::*;
use waves::*;

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.scq.is_file() {
        bail!("SCQ binary is absent: {}", args.scq.display());
    }
    if args.dump_dl_json {
        return dump_dl_json(&args);
    }
    if args.dump_unified_json {
        return dump_unified_json(&args);
    }
    // Populate the GPU/trainer/watchdog collectors before entering the
    // alternate screen. Previously the first frame always said "no GPU
    // telemetry" and "watchdog unavailable" for roughly a second, which made
    // a healthy trainer look absent precisely when the operator launched the
    // TUI to check it.
    let initial_raw = collect_sources(&args);
    let mut slow = SlowState::new();
    collect_slow_sources(&args, &initial_raw, &mut slow);
    let initial = tui_dashboard::build_dashboard(&initial_raw, &slow.view());
    let shared = Arc::new(RwLock::new(initial));
    let running = Arc::new(AtomicBool::new(true));
    let collector = {
        let args = args.clone();
        let shared = Arc::clone(&shared);
        let running = Arc::clone(&running);
        thread::Builder::new()
            .name("jope-tui-collector".to_owned())
            .spawn(move || {
                let cadence = Duration::from_millis(args.refresh_ms);
                while running.load(Ordering::Relaxed) {
                    let started = Instant::now();
                    let raw = collect_sources(&args);
                    collect_slow_sources(&args, &raw, &mut slow);
                    let next = tui_dashboard::build_dashboard(&raw, &slow.view());
                    if let Ok(mut current) = shared.write() {
                        *current = next;
                    }
                    let remaining = cadence.saturating_sub(started.elapsed());
                    if !remaining.is_zero() {
                        thread::sleep(remaining);
                    }
                }
            })?
    };

    let result = run_terminal(&shared, args.refresh_ms);
    running.store(false, Ordering::Relaxed);
    let _ = collector.join();
    result
}

#[cfg(test)]
#[path = "jope_selfplay_tui/tests.rs"]
mod tests;
