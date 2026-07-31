//! Read-only, Rust-native operations console for JOPE self-play.

use anyhow::{bail, Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
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

const DEFAULT_ROOT: &str = "/home/ubuntu/jope-flywheel/test_exact_gen0";
const DEFAULT_SCQ: &str = "/home/ubuntu/smart_cluster/target/release/scq";
const DEFAULT_METRICS_ROOT: &str = "/home/ubuntu/smart_cluster/var/runs";
const DEFAULT_REFRESH_MS: u64 = 250;
const CPU_TARGET_PCT: f64 = 80.0;
const CPU_HARD_GUARD_PCT: f64 = 95.0;
const EVENT_TAIL_BYTES: u64 = 64 * 1024;
const METRICS_TAIL_BYTES: u64 = 256 * 1024;
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
const GENERATIONS_BUDGET_MS: u128 = 50;
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
const TRAINER_ID_PREFIX: &str = "jope-selfplay-train-";
/// Newest seeding-curve generations rendered in the tab-4 generations panel.
const SEEDING_CURVE_ROWS: usize = 3;
/// (ops status file, expected write cadence seconds) for the DL tab heartbeat panel.
const DAEMON_HEARTBEATS: [(&str, u64); 5] = [
    ("ops/queue-keeper-v3.json", 450),
    ("ops/micro-wave-supply-v1.json", 60),
    ("ops/trainer-watchdog-v1.json", 60),
    ("ops/last-ingest.json", 450),
    ("ops/freshness-status.json", 150),
];

#[derive(Parser, Debug, Clone)]
#[command(
    name = "jope-selfplay-tui",
    about = "Read-only Rust TUI for live SCQ and JOPE self-play state"
)]
struct Args {
    /// Flywheel state root.
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
    /// Native SCQ client.
    #[arg(long, default_value = DEFAULT_SCQ)]
    scq: PathBuf,
    /// Terminal redraw and source refresh cadence.
    #[arg(
        long,
        default_value_t = DEFAULT_REFRESH_MS,
        value_parser = clap::value_parser!(u64).range(50..=299)
    )]
    refresh_ms: u64,
    /// Root of SCQ run directories (trainer progress discovery).
    #[arg(long, default_value = DEFAULT_METRICS_ROOT)]
    metrics_root: PathBuf,
    /// OPT-IN: corpus sqlite for point lookups (default: never opened).
    #[arg(long)]
    gameplay_db: Option<PathBuf>,
    /// OPT-IN: backlog sqlite for claims-by-state (default: never opened).
    #[arg(long)]
    claims_db: Option<PathBuf>,
    /// One-shot: run the real collectors once, print the DL chart series plus
    /// the dataset funnel as JSON, and exit. No terminal required — this is how
    /// the plots are verified unattended.
    #[arg(long)]
    dump_dl_json: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Tab {
    Overview,
    #[default]
    GpuDl,
    Gameplay,
    Dl,
}

#[derive(Clone, Copy, Debug, Default)]
struct UiState {
    tab: Tab,
    /// Which training arm the DL charts follow. Parallel tournament arms run one
    /// per GPU, so this cycles rather than pinning to the first job found. Held
    /// as an index and wrapped at render time against the live trainer count, so
    /// arms appearing or exiting can never leave it dangling.
    arm: usize,
}

#[derive(Clone, Debug, Default)]
struct RawSources {
    scq: Option<Value>,
    status: Option<Value>,
    queue: Option<Value>,
    freshness: Option<Value>,
    supply: Option<Value>,
    proposal_summary: Option<Value>,
    trainer_fallback: Option<Value>,
    flywheel: Option<Value>,
    accepted_by_task: Option<Value>,
    evidence_bytes: Option<u64>,
    local_events: Vec<Value>,
    source_states: Vec<SourceState>,
    collected_at_ms: u128,
    collection_latency_ms: u128,
}

#[derive(Clone, Debug)]
struct SourceState {
    name: String,
    healthy: bool,
    note: String,
}

#[derive(Clone, Debug, Default)]
struct ClusterSummary {
    cpu_busy_pct: f64,
    psi_cpu10: f64,
    psi_mem10: f64,
    psi_io10: f64,
    physical_cores: u64,
    effective_cores: u64,
    slot_target: u64,
    running_slots: u64,
    memory_available_gb: u64,
    nodes_up: usize,
    nodes_total: usize,
    jobs_failed: u64,
    throughput_per_minute: u64,
}

#[derive(Clone, Debug, Default)]
struct NodeRow {
    node: String,
    state: String,
    health: String,
    physical_cores: u64,
    effective_cores: u64,
    multiplier: f64,
    slot_target: u64,
    running_slots: u64,
    cpu_busy_pct: Option<f64>,
    psi_cpu10: Option<f64>,
    psi_mem10: Option<f64>,
    psi_io10: Option<f64>,
    memory_available_gb: u64,
    memory_total_gb: u64,
    disk_use_pct: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct FamilyRow {
    family: String,
    weight: f64,
    queued: u64,
    starting: u64,
    dispatching: u64,
    running: u64,
    held: u64,
    oldest_queue_ts: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct QueueSummary {
    queued: u64,
    starting: u64,
    dispatching: u64,
    running: u64,
    held: u64,
    useful_depth: u64,
    target: u64,
    hard_cap: u64,
    maximum_age_seconds: f64,
    epoch: String,
    stale_epochs: u64,
    stale_held: u64,
    admission_allowed: Option<bool>,
    mode: String,
}

#[derive(Clone, Debug, Default)]
struct LearningSummary {
    checkpoint: String,
    trainer: String,
    trainer_state: String,
    labels: u64,
    frozen_watermark: u64,
    internal_promotion: String,
    milestone_promotion: String,
    proposal_regression: u64,
    proposal_binary: u64,
    proposal_validity: Option<f64>,
    consecutive_rejections: u64,
}

#[derive(Clone, Debug, Default)]
struct WaveSummary {
    active_wave: String,
    admission_waves: usize,
    unclaimed_manifests: u64,
    ready_manifests: u64,
    awaiting_keeper_ack: String,
    adaptive_regression: u64,
    adaptive_binary: u64,
    sealed_isolated: Option<bool>,
    sealed_scored: u64,
    corpus_rows: u64,
    strict_wins: u64,
    pair_count: u64,
    duplicate_pairs_prevented: u64,
    evidence_bytes: u64,
}

#[derive(Clone, Debug, Default)]
struct GpuRow {
    node: String,
    model_hint: String,
    status: String,
    owner_job: Option<String>,
    util_pct: f64,
    mem_used_mb: u64,
    mem_total_mb: u64,
    temp_c: u64,
    power_w: f64,
    power_limit_w: f64,
    throttled: bool,
}

#[derive(Clone, Debug, Default)]
struct TrainerMetricPoint {
    epoch: u64,
    train_loss: f64,
    val_loss: f64,
    bad_passes: u64,
    overfit_gap: f64,
    val_pairwise: Option<f64>,
    val_binary_pairwise: Option<f64>,
    val_regression_pairwise: Option<f64>,
    val_reward_cv: Option<f64>,
    val_top1_regret: Option<f64>,
    /// Conditional model minus the genome-only ablation control. This is the
    /// anti-collapse yardstick: at or below zero the policy is not using the
    /// dataset embedding at all, however good its raw pairwise number looks.
    val_conditional_advantage: Option<f64>,
    val_ablation_pairwise: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct TrainerDetail {
    job_id: String,
    node: String,
    is_trainer: bool,
    phase: String,
    epoch: u64,
    epochs: u64,
    batch: u64,
    total_batches: u64,
    pct: f64,
    running_loss: Option<f64>,
    val_loss: Option<f64>,
    overfit_gap: Option<f64>,
    bad_passes: u64,
    patience: u64,
    early_stopped: bool,
    label_watermark: u64,
    elapsed_sec: f64,
    validation_batch: u64,
    validation_batches: u64,
    progress_at_ms: u128,
    history: Vec<TrainerMetricPoint>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Origin {
    DlProposal,
    GpOffspring,
    LocalElite,
    CommonWinner,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EventKind {
    Eval,
    StrictWin,
}

#[derive(Clone, Debug)]
struct GameplayEvent {
    at_ms: u128,
    task: String,
    dataset_hash: String,
    score: Option<f64>,
    delta_vs_champion: Option<f64>,
    origin: Origin,
    node: String,
    wave: String,
    valid: bool,
    kind: EventKind,
}

#[derive(Clone, Debug, Default)]
struct GameplayStats {
    manifests_seen: u64,
    dl_top1: u64,
    gp_top1: u64,
    dl_slot_share_pct: f64,
    dl_beats_champion: u64,
    gp_beats_champion: u64,
}

#[derive(Clone, Debug, Default)]
struct WaveProgress {
    wave: String,
    manifest_count: u64,
    datasets: u64,
    receipts_landed: u64,
    claims_by_state: BTreeMap<String, u64>,
    generated_at_ms: u128,
}

#[derive(Clone, Debug, Default)]
struct TrainerWatchdogView {
    labels: u64,
    snapshot_watermark: u64,
    last_evidence_watermark: u64,
    next_label_threshold: u64,
    pending_unfrozen_labels: u64,
    checkpoint_sha256: String,
    action: String,
    active_trainers: Vec<String>,
    generated_at_ms: u128,
}

#[derive(Clone, Debug, Default)]
struct AcceptedTaskFront {
    task: String,
    checkpoint_sha256: String,
    generation: u64,
    updated_at_ms: u128,
}

/// One task's dataset-unlock pipeline:
/// registry → encoding (jobs queued/running) → queued (encoded, awaiting
/// activation) → active in catalog → in play (active with evidence).
#[derive(Clone, Debug, Default, PartialEq)]
struct FunnelRow {
    task: String,
    registry: u64,
    encoded: u64,
    active: u64,
    in_play: u64,
    queued: u64,
    encoding: u64,
    sealed: u64,
    delta_active_1h: i64,
    delta_in_play_1h: i64,
}

impl FunnelRow {
    fn locked(&self) -> u64 {
        self.registry
            .saturating_sub(self.active)
            .saturating_sub(self.queued)
            .saturating_sub(self.encoding)
            .saturating_sub(self.sealed)
    }
}

#[derive(Clone, Debug, Default)]
struct UnlockFunnel {
    rows: Vec<FunnelRow>,
    source: String,
    fetched_at_ms: u128,
}

/// One checkpoint generation from ops/seeding-curve.v1.json: the DL-share
/// column of the tab-4 generations panel.
#[derive(Clone, Debug, Default)]
struct SeedingPoint {
    checkpoint_short: String,
    dl_win_share_pct: Option<f64>,
    dl_obs_share_pct: Option<f64>,
    strict_wins_total: u64,
    observations_caught_up: bool,
}

/// Seeding-curve slow source (30s gate; the file itself is refreshed by the
/// ingestion reconcile loop on a minutes cadence).
#[derive(Clone, Debug, Default)]
struct SeedingCurveView {
    points: Vec<SeedingPoint>,
    fetched_at_ms: u128,
}

/// One trainer generation's per-epoch metrics, recovered from its run dir.
/// Identity is the RUN DIRECTORY name, which outlives the SCQ job, so this
/// survives cancel/resubmit/rename churn.
#[derive(Clone, Debug, Default)]
struct RunSeries {
    /// Run-dir name minus the `jope-selfplay-train-` prefix (`001805437-a-g8`).
    run_label: String,
    /// Compact generation tag for the ruler (`a-g8`).
    short_label: String,
    points: Vec<TrainerMetricPoint>,
    /// True while an SCQ job with this run-dir name is still running.
    live: bool,
    /// train_metrics.jsonl mtime — orders generations and gates re-reads.
    updated_at_ms: u128,
}

/// Cross-generation DL training history: the operator's "loss constantly going
/// down" curve. Persisted implicitly by the run dirs themselves.
#[derive(Clone, Debug, Default)]
struct DlHistory {
    runs: Vec<RunSeries>,
    fetched_at_ms: u128,
}

/// Flattened chart-ready form of `DlHistory` on one continuous epoch axis.
#[derive(Clone, Debug, Default)]
struct DlSeries {
    train: Vec<(f64, f64)>,
    val: Vec<(f64, f64)>,
    pairwise: Vec<(f64, f64)>,
    ablation: Vec<(f64, f64)>,
    /// (x where the generation starts, short label, live) — the boundaries.
    marks: Vec<(f64, String, bool)>,
    total_epochs: f64,
}

/// Immutable snapshot of the slow-source cache, handed to build_dashboard.
#[derive(Clone, Debug, Default)]
struct SlowView {
    gpus: Vec<GpuRow>,
    trainers: Vec<TrainerDetail>,
    watchdog: Option<TrainerWatchdogView>,
    gameplay: Vec<GameplayEvent>,
    gameplay_stats: GameplayStats,
    wave_progress: Option<WaveProgress>,
    health_alarms: Vec<String>,
    daemon_ages: Vec<(String, u64, u64)>,
    generations: Vec<(String, Option<f64>)>,
    seeding: SeedingCurveView,
    dl_history: DlHistory,
    funnel: UnlockFunnel,
    feed_fetched_at_ms: u128,
    newest_receipt_at_ms: u128,
    source_states: Vec<SourceState>,
}

#[derive(Clone, Debug, Default)]
struct Dashboard {
    collected_at_ms: u128,
    collection_latency_ms: u128,
    scq_epoch: Option<u64>,
    scq_ts: String,
    cluster: ClusterSummary,
    nodes: Vec<NodeRow>,
    families: Vec<FamilyRow>,
    queue: QueueSummary,
    learning: LearningSummary,
    wave: WaveSummary,
    gpus: Vec<GpuRow>,
    trainers: Vec<TrainerDetail>,
    watchdog: Option<TrainerWatchdogView>,
    accepted_tasks: Vec<AcceptedTaskFront>,
    gameplay: Vec<GameplayEvent>,
    gameplay_stats: GameplayStats,
    wave_progress: Option<WaveProgress>,
    daemon_ages: Vec<(String, u64, u64)>,
    generations: Vec<(String, Option<f64>)>,
    seeding: SeedingCurveView,
    dl_history: DlHistory,
    funnel: UnlockFunnel,
    feed_fetched_at_ms: u128,
    alerts: Vec<String>,
    failures: Vec<String>,
    events: Vec<String>,
    sources_healthy: usize,
    sources_total: usize,
}

struct TerminalSession;

impl TerminalSession {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enable raw terminal mode")?;
        if let Err(error) = execute!(io::stdout(), EnterAlternateScreen, crossterm::cursor::Hide) {
            let _ = disable_raw_mode();
            return Err(error).context("enter alternate screen");
        }
        Ok(Self)
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), crossterm::cursor::Show, LeaveAlternateScreen);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.scq.is_file() {
        bail!("SCQ binary is absent: {}", args.scq.display());
    }
    if args.dump_dl_json {
        return dump_dl_json(&args);
    }
    // Populate the GPU/trainer/watchdog collectors before entering the
    // alternate screen. Previously the first frame always said "no GPU
    // telemetry" and "watchdog unavailable" for roughly a second, which made
    // a healthy trainer look absent precisely when the operator launched the
    // TUI to check it.
    let initial_raw = collect_sources(&args);
    let mut slow = SlowState::new();
    collect_slow_sources(&args, &initial_raw, &mut slow);
    let initial = build_dashboard(&initial_raw, &slow.view());
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
                    let next = build_dashboard(&raw, &slow.view());
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

/// Headless verification path. Drives the exact collectors the TUI renders from
/// and prints what the DL charts would plot, so "the plot has data" is a
/// checkable claim rather than a screenshot.
fn dump_dl_json(args: &Args) -> Result<()> {
    let raw = collect_sources(args);
    let mut slow = SlowState::new();
    collect_slow_sources(args, &raw, &mut slow);
    let dashboard = build_dashboard(&raw, &slow.view());
    let series = continuous_series(&dashboard.dl_history);
    let generations = dashboard
        .dl_history
        .runs
        .iter()
        .map(|run| {
            json!({
                "run": run.run_label,
                "short": run.short_label,
                "live": run.live,
                "epochs": run.points.len(),
                "updated_at_ms": run.updated_at_ms as u64,
                "first_val_loss": run.points.first().map(|point| point.val_loss),
                "last_val_loss": run.points.last().map(|point| point.val_loss),
                "last_val_pairwise": run.points.last().and_then(|point| point.val_pairwise),
                "last_conditional_advantage":
                    run.points.last().and_then(|point| point.val_conditional_advantage),
            })
        })
        .collect::<Vec<_>>();
    let live_trainers = dashboard
        .trainers
        .iter()
        .filter(|trainer| trainer.is_trainer)
        .map(|trainer| {
            json!({
                "job_id": trainer.job_id,
                "node": trainer.node,
                "phase": trainer.phase,
                "epoch": trainer.epoch,
                "epochs": trainer.epochs,
                "pct": trainer.pct,
                "running_loss": trainer.running_loss,
                "history_points": trainer.history.len(),
                "progress_at_ms": trainer.progress_at_ms as u64,
            })
        })
        .collect::<Vec<_>>();
    let funnel = dashboard
        .funnel
        .rows
        .iter()
        .map(|row| {
            json!({
                "task": row.task,
                "registry": row.registry,
                "encoded": row.encoded,
                "active": row.active,
                "in_play": row.in_play,
                "queued": row.queued,
                "encoding": row.encoding,
                "sealed": row.sealed,
                "locked": row.locked(),
            })
        })
        .collect::<Vec<_>>();
    let active_total: u64 = dashboard.funnel.rows.iter().map(|row| row.active).sum();
    let payload = json!({
        "schema": "JopeTuiDlSeries.v1",
        "collected_at_ms": dashboard.collected_at_ms as u64,
        "generations": generations,
        "live_trainers": live_trainers,
        "gpus": dashboard.gpus.iter().map(|gpu| json!({
            "node": gpu.node,
            "owner_job": gpu.owner_job,
            "util_pct": gpu.util_pct,
            "mem_used_mb": gpu.mem_used_mb,
            "mem_total_mb": gpu.mem_total_mb,
            "power_w": gpu.power_w,
        })).collect::<Vec<_>>(),
        "accepted_task_fronts": dashboard.accepted_tasks.iter().map(|front| json!({
            "task": front.task,
            "checkpoint_sha256": front.checkpoint_sha256,
            "generation": front.generation,
            "updated_at_ms": front.updated_at_ms as u64,
        })).collect::<Vec<_>>(),
        "watchdog": dashboard.watchdog.as_ref().map(|watchdog| json!({
            "action": watchdog.action,
            "checkpoint_sha256": watchdog.checkpoint_sha256,
            "snapshot_watermark": watchdog.snapshot_watermark,
            "labels": watchdog.labels,
            "pending_unfrozen_labels": watchdog.pending_unfrozen_labels,
            "active_trainers": watchdog.active_trainers,
            "generated_at_ms": watchdog.generated_at_ms as u64,
        })),
        "continuous": {
            "total_epochs": series.total_epochs,
            "marks": series
                .marks
                .iter()
                .map(|(x, label, live)| json!({"x": x, "label": label, "live": live}))
                .collect::<Vec<_>>(),
            "train": series.train,
            "val": series.val,
            "pairwise": series.pairwise,
            "ablation": series.ablation,
        },
        "generations_panel": dashboard
            .generations
            .iter()
            .map(|(label, pairwise)| json!({"label": label, "val_pairwise": pairwise}))
            .collect::<Vec<_>>(),
        "funnel": funnel,
        "funnel_source": dashboard.funnel.source,
        "catalog_active_total": active_total,
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}

fn run_terminal(shared: &Arc<RwLock<Dashboard>>, refresh_ms: u64) -> Result<()> {
    let _session = TerminalSession::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    let cadence = Duration::from_millis(refresh_ms);

    let mut ui = UiState::default();
    loop {
        let snapshot = shared.read().map(|value| value.clone()).unwrap_or_default();
        terminal.draw(|frame| render(frame, &snapshot, refresh_ms, ui))?;
        if event::poll(cadence)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if matches!(key.code, KeyCode::Esc | KeyCode::Char('q'))
                        || (key.code == KeyCode::Char('c')
                            && key.modifiers.contains(KeyModifiers::CONTROL))
                    {
                        break;
                    }
                    // Arm selection is independent of the tab switch below so
                    // `n`/arrows do not fall through and reset the tab.
                    match key.code {
                        KeyCode::Char('n') | KeyCode::Right => {
                            ui.arm = ui.arm.wrapping_add(1);
                        }
                        KeyCode::Left => {
                            // Wrap without underflow; the renderer takes it modulo
                            // the live arm count, so a large value is harmless.
                            ui.arm = ui.arm.checked_sub(1).unwrap_or(usize::MAX);
                        }
                        _ => {}
                    }
                    ui.tab = match key.code {
                        KeyCode::Char('1') => Tab::Overview,
                        KeyCode::Char('2') => Tab::GpuDl,
                        KeyCode::Char('3') => Tab::Gameplay,
                        KeyCode::Char('4') | KeyCode::Char('d') => Tab::Dl,
                        KeyCode::Tab => match ui.tab {
                            Tab::Overview => Tab::GpuDl,
                            Tab::GpuDl => Tab::Gameplay,
                            Tab::Gameplay => Tab::Dl,
                            Tab::Dl => Tab::Overview,
                        },
                        KeyCode::BackTab => match ui.tab {
                            Tab::Overview => Tab::Dl,
                            Tab::GpuDl => Tab::Overview,
                            Tab::Gameplay => Tab::GpuDl,
                            Tab::Dl => Tab::Gameplay,
                        },
                        _ => ui.tab,
                    };
                }
            }
        }
    }
    Ok(())
}

fn collect_sources(args: &Args) -> RawSources {
    let started = Instant::now();
    let mut raw = RawSources {
        collected_at_ms: now_ms(),
        ..RawSources::default()
    };
    raw.scq = match load_scq_snapshot(&args.scq) {
        Ok(value) => {
            raw.source_states.push(SourceState {
                name: "scq".to_owned(),
                healthy: true,
                note: "snapshot".to_owned(),
            });
            Some(value)
        }
        Err(error) => {
            raw.source_states.push(SourceState {
                name: "scq".to_owned(),
                healthy: false,
                note: format!("{error:#}"),
            });
            None
        }
    };

    let status_path = if args.root.join("ops/status-v3.json").is_file() {
        args.root.join("ops/status-v3.json")
    } else {
        args.root.join("ops/status.json")
    };
    raw.status = read_json_source(&status_path, "status", true, &mut raw.source_states);
    raw.queue = read_json_source(
        &args.root.join("ops/queue-keeper-v3.json"),
        "queue-v3",
        true,
        &mut raw.source_states,
    );
    raw.freshness = read_json_source(
        &args.root.join("ops/freshness-status.json"),
        "freshness",
        true,
        &mut raw.source_states,
    );
    raw.supply = read_json_source(
        &args.root.join("ops/micro-wave-supply-v1.json"),
        "supply",
        true,
        &mut raw.source_states,
    );
    raw.proposal_summary = read_json_source(
        &args.root.join("proposals/dl-proposals.summary.json"),
        "proposals",
        true,
        &mut raw.source_states,
    );
    let trainer_path = args.root.join("ops/trainer-status.json");
    let trainer_fallback = if trainer_path.is_file() {
        trainer_path
    } else {
        args.root.join("ops/dl-job.json")
    };
    raw.trainer_fallback =
        read_json_source(&trainer_fallback, "trainer", false, &mut raw.source_states);
    raw.flywheel = read_json_source(
        &args.root.join("ops/flywheel-state.json"),
        "flywheel",
        false,
        &mut raw.source_states,
    );
    raw.accepted_by_task = read_json_source(
        &args.root.join("training/accepted-by-task.json"),
        "accepted-by-task",
        false,
        &mut raw.source_states,
    );
    raw.evidence_bytes = fs::metadata(args.root.join("indexes/gene-evidence.v2.jsonl"))
        .ok()
        .map(|metadata| metadata.len());
    for relative in [
        "ops/freshness-events.v1.jsonl",
        "ops/micro-wave-supply-events.v1.jsonl",
        "ops/controller-events.jsonl",
    ] {
        raw.local_events.extend(tail_json_lines(
            &args.root.join(relative),
            8,
            EVENT_TAIL_BYTES,
        ));
    }
    raw.collection_latency_ms = started.elapsed().as_millis();
    raw
}

fn read_json_source(
    path: &Path,
    name: &str,
    required: bool,
    states: &mut Vec<SourceState>,
) -> Option<Value> {
    match fs::read(path) {
        Ok(bytes) => match serde_json::from_slice(&bytes) {
            Ok(value) => {
                states.push(SourceState {
                    name: name.to_owned(),
                    healthy: true,
                    note: path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or("json")
                        .to_owned(),
                });
                Some(value)
            }
            Err(error) => {
                states.push(SourceState {
                    name: name.to_owned(),
                    healthy: false,
                    note: format!("invalid JSON: {error}"),
                });
                None
            }
        },
        Err(error) => {
            if required || error.kind() != io::ErrorKind::NotFound {
                states.push(SourceState {
                    name: name.to_owned(),
                    healthy: false,
                    note: if error.kind() == io::ErrorKind::NotFound {
                        "missing".to_owned()
                    } else {
                        error.to_string()
                    },
                });
            }
            None
        }
    }
}

fn load_scq_snapshot(scq: &Path) -> Result<Value> {
    match run_scq_json(scq, &["snapshot"]) {
        Ok(value) if value.get("Snapshot").is_some() => Ok(value),
        primary => {
            let primary_error = primary
                .err()
                .map(|error| format!("{error:#}"))
                .unwrap_or_else(|| "snapshot response lacks Snapshot".to_owned());
            let nodes = run_scq_json(scq, &["nodes"]).with_context(|| {
                format!("SCQ snapshot failed ({primary_error}); nodes fallback")
            })?;
            let jobs = run_scq_json(scq, &["ls"])
                .with_context(|| format!("SCQ snapshot failed ({primary_error}); jobs fallback"))?;
            let priorities = run_scq_json(scq, &["priority", "ls"]).unwrap_or_else(|_| json!({}));
            Ok(json!({
                "Snapshot": {
                    "epoch": null,
                    "ts": null,
                    "nodes": nodes.pointer("/Nodes/nodes").cloned().unwrap_or_else(|| json!([])),
                    "gpus": nodes.pointer("/Nodes/gpus").cloned().unwrap_or_else(|| json!([])),
                    "jobs": jobs.pointer("/Jobs/jobs").cloned().unwrap_or_else(|| json!([])),
                    "families": priorities.pointer("/FamilyPriorities/families")
                        .cloned().unwrap_or_else(|| json!([])),
                    "events": [],
                    "alerts": [format!("snapshot fallback: {primary_error}")],
                    "counters": {},
                }
            }))
        }
    }
}

fn run_scq_json(scq: &Path, arguments: &[&str]) -> Result<Value> {
    let output = Command::new(scq)
        .arg("--json")
        .args(arguments)
        .output()
        .with_context(|| format!("run {} --json {}", scq.display(), arguments.join(" ")))?;
    if !output.status.success() {
        bail!(
            "{}",
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    serde_json::from_slice(&output.stdout).context("parse SCQ JSON")
}

fn tail_json_lines(path: &Path, limit: usize, maximum_bytes: u64) -> Vec<Value> {
    let Ok(mut file) = File::open(path) else {
        return Vec::new();
    };
    let Ok(length) = file.metadata().map(|metadata| metadata.len()) else {
        return Vec::new();
    };
    let start = length.saturating_sub(maximum_bytes);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut bytes = Vec::with_capacity((length - start) as usize);
    if file.read_to_end(&mut bytes).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&bytes);
    let mut lines = text.lines().collect::<Vec<_>>();
    if start > 0 && !lines.is_empty() {
        lines.remove(0);
    }
    lines
        .into_iter()
        .rev()
        .filter_map(|line| serde_json::from_str(line).ok())
        .take(limit)
        .collect()
}

struct IntervalGate {
    last: Option<Instant>,
    every: Duration,
}

impl IntervalGate {
    fn new(every: Duration) -> Self {
        Self { last: None, every }
    }

    fn due(&mut self) -> bool {
        let now = Instant::now();
        if self
            .last
            .is_none_or(|last| now.duration_since(last) >= self.every)
        {
            self.last = Some(now);
            return true;
        }
        false
    }
}

/// Collector-thread-owned cache for sources too expensive for the 250ms loop.
struct SlowState {
    gate_progress: IntervalGate,
    gate_metrics: IntervalGate,
    gate_receipts: IntervalGate,
    gate_indexes: IntervalGate,
    gate_wave: IntervalGate,
    gate_claims: IntervalGate,
    known_receipt_dirs: HashSet<String>,
    receipts_prefilled: bool,
    wave_manifest_paths: HashMap<String, HashMap<String, String>>,
    manifest_semantics: HashMap<String, HashMap<String, (String, String)>>,
    manifest_order: VecDeque<String>,
    champions: HashMap<(String, String), f64>,
    champions_mtime: Option<SystemTime>,
    hof_seen: usize,
    hof_mtime: Option<SystemTime>,
    ring: VecDeque<GameplayEvent>,
    stats_window: VecDeque<(Origin, bool)>,
    manifests_seen: u64,
    dl_slots: u64,
    total_slots: u64,
    newest_wave_receipts: u64,
    newest_wave_name: String,
    /// job id -> (attempt the path came from, resolved train_progress.json).
    progress_paths: HashMap<String, (u64, PathBuf)>,
    metrics_mtimes: HashMap<String, SystemTime>,
    gate_generations: IntervalGate,
    generations: Vec<(String, Option<f64>)>,
    gate_dl_history: IntervalGate,
    dl_history: DlHistory,
    /// run-dir name -> (attempt the path came from, resolved
    /// train_metrics.jsonl). Skips the tree walk without surviving a requeue.
    dl_metrics_paths: HashMap<String, (u64, PathBuf)>,
    /// run-dir name -> (metrics mtime, parsed points) so a steady-state pass
    /// only re-reads the generation that is actually still writing.
    dl_points_cache: HashMap<String, (SystemTime, Vec<TrainerMetricPoint>)>,
    gate_seeding: IntervalGate,
    seeding: SeedingCurveView,
    daemon_ages: Vec<(String, u64, u64)>,
    trainers: Vec<TrainerDetail>,
    gpus: Vec<GpuRow>,
    watchdog: Option<TrainerWatchdogView>,
    wave_progress: Option<WaveProgress>,
    health_alarms: Vec<String>,
    claims_backoff_until: Option<Instant>,
    claims_backoff: Duration,
    gate_funnel: IntervalGate,
    funnel: UnlockFunnel,
    funnel_history: VecDeque<(u128, Vec<(String, u64, u64)>)>,
    feed_fetched_at_ms: u128,
    newest_receipt_at_ms: u128,
    source_states: Vec<SourceState>,
}

impl SlowState {
    fn new() -> Self {
        Self {
            gate_progress: IntervalGate::new(Duration::from_secs(1)),
            gate_metrics: IntervalGate::new(Duration::from_secs(5)),
            gate_receipts: IntervalGate::new(Duration::from_secs(2)),
            gate_indexes: IntervalGate::new(Duration::from_secs(5)),
            gate_wave: IntervalGate::new(Duration::from_secs(10)),
            gate_claims: IntervalGate::new(Duration::from_secs(15)),
            known_receipt_dirs: HashSet::new(),
            receipts_prefilled: false,
            wave_manifest_paths: HashMap::new(),
            manifest_semantics: HashMap::new(),
            manifest_order: VecDeque::new(),
            champions: HashMap::new(),
            champions_mtime: None,
            hof_seen: 0,
            hof_mtime: None,
            ring: VecDeque::new(),
            stats_window: VecDeque::new(),
            manifests_seen: 0,
            dl_slots: 0,
            total_slots: 0,
            newest_wave_receipts: 0,
            newest_wave_name: String::new(),
            progress_paths: HashMap::new(),
            metrics_mtimes: HashMap::new(),
            gate_generations: IntervalGate::new(Duration::from_secs(60)),
            generations: Vec::new(),
            gate_dl_history: IntervalGate::new(Duration::from_secs(5)),
            dl_history: DlHistory::default(),
            dl_metrics_paths: HashMap::new(),
            dl_points_cache: HashMap::new(),
            gate_seeding: IntervalGate::new(Duration::from_secs(30)),
            seeding: SeedingCurveView::default(),
            daemon_ages: Vec::new(),
            trainers: Vec::new(),
            gpus: Vec::new(),
            watchdog: None,
            wave_progress: None,
            health_alarms: Vec::new(),
            claims_backoff_until: None,
            claims_backoff: Duration::from_secs(10),
            gate_funnel: IntervalGate::new(Duration::from_secs(30)),
            funnel: UnlockFunnel::default(),
            funnel_history: VecDeque::new(),
            feed_fetched_at_ms: 0,
            newest_receipt_at_ms: 0,
            source_states: Vec::new(),
        }
    }

    fn view(&self) -> SlowView {
        let feed = self
            .ring
            .iter()
            .rev()
            .take(GAMEPLAY_FEED_RENDER)
            .cloned()
            .collect::<Vec<_>>();
        let dl_top1 = self
            .stats_window
            .iter()
            .filter(|(origin, _)| *origin == Origin::DlProposal)
            .count() as u64;
        let dl_beats = self
            .stats_window
            .iter()
            .filter(|(origin, beats)| *origin == Origin::DlProposal && *beats)
            .count() as u64;
        let gp_beats = self
            .stats_window
            .iter()
            .filter(|(origin, beats)| *origin != Origin::DlProposal && *beats)
            .count() as u64;
        SlowView {
            gpus: self.gpus.clone(),
            trainers: self.trainers.clone(),
            watchdog: self.watchdog.clone(),
            gameplay: feed,
            gameplay_stats: GameplayStats {
                manifests_seen: self.manifests_seen,
                dl_top1,
                gp_top1: self.stats_window.len() as u64 - dl_top1,
                dl_slot_share_pct: if self.total_slots == 0 {
                    0.0
                } else {
                    self.dl_slots as f64 * 100.0 / self.total_slots as f64
                },
                dl_beats_champion: dl_beats,
                gp_beats_champion: gp_beats,
            },
            wave_progress: self.wave_progress.clone(),
            health_alarms: self.health_alarms.clone(),
            daemon_ages: self.daemon_ages.clone(),
            generations: self.generations.clone(),
            seeding: self.seeding.clone(),
            dl_history: self.dl_history.clone(),
            funnel: self.funnel.clone(),
            feed_fetched_at_ms: self.feed_fetched_at_ms,
            newest_receipt_at_ms: self.newest_receipt_at_ms,
            source_states: self.source_states.clone(),
        }
    }
}

fn collect_slow_sources(args: &Args, raw: &RawSources, state: &mut SlowState) {
    let deadline = Instant::now() + Duration::from_millis(SLOW_PASS_DEADLINE_MS as u64);
    state.source_states.clear();
    let scq = raw
        .scq
        .as_ref()
        .and_then(|value| value.get("Snapshot"))
        .unwrap_or(&Value::Null);
    let gpu_jobs = discover_gpu_jobs(scq);
    state.gpus = parse_gpus(scq, &gpu_jobs);

    if state.gate_progress.due() && Instant::now() < deadline {
        refresh_trainers(args, &gpu_jobs, state);
    }
    if state.gate_metrics.due() && Instant::now() < deadline {
        for trainer in &mut state.trainers {
            if let Some((_, path)) = state.progress_paths.get(&trainer.job_id) {
                if let Some(dir) = path.parent() {
                    let metrics_path = dir.join("train_metrics.jsonl");
                    let mtime = fs::metadata(&metrics_path)
                        .and_then(|metadata| metadata.modified())
                        .unwrap_or(UNIX_EPOCH);
                    let changed = state
                        .metrics_mtimes
                        .get(&trainer.job_id)
                        .is_none_or(|prior| *prior != mtime);
                    if changed || trainer.history.is_empty() {
                        trainer.history = read_metrics_history(&metrics_path);
                        state.metrics_mtimes.insert(trainer.job_id.clone(), mtime);
                    }
                    if let Some(last) = trainer.history.last() {
                        trainer.bad_passes = last.bad_passes;
                        trainer.overfit_gap = Some(last.overfit_gap);
                    }
                }
            }
        }
        state.watchdog = read_watchdog(&args.root);
        state.health_alarms = read_health_alarms(&args.root, &mut state.source_states);
        state.daemon_ages = read_daemon_ages(&args.root);
    }
    // The cross-generation DL history is the operator's headline chart and is
    // mtime-cached down to a handful of stat() calls in the steady state, so —
    // like the funnel below — it is exempt from the pass deadline. Putting it
    // behind `< deadline` is what made the plot blank whenever a busy pass ran
    // during a trainer changeover.
    if state.gate_dl_history.due() {
        let live_runs = gpu_jobs
            .iter()
            .filter(|(_, _, _, is_trainer)| *is_trainer)
            .filter_map(|(_, _, run_dir, _)| {
                run_dir
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
            })
            .collect::<HashSet<_>>();
        state.dl_history = collect_dl_history(&args.metrics_root, &live_runs, state);
    }
    // The unlock funnel is the operator's headline number and costs one indexed
    // GROUP BY over a small table, so it runs before the expensive scans and is
    // exempt from the pass deadline. It used to sit last behind `< deadline`,
    // where a 174ms collect against a 200ms budget starved it forever — the
    // panel read "collecting…" and the cohort label read "0 active".
    if state.gate_funnel.due() {
        refresh_unlock_funnel(&args.root, state);
    }
    // Deadline is checked BEFORE `due()` in every remaining source: `due()` stamps
    // its timestamp as a side effect, so testing it first burned the interval on
    // passes that were then skipped, delaying the source by a further full cycle.
    if state.gate_generations.due() {
        state.generations = generations_from_history(&state.dl_history);
    }
    if Instant::now() < deadline && state.gate_seeding.due() {
        // Keep the prior snapshot on read failure so the STALE tag reports it.
        if let Some(seeding) = read_seeding_curve(&args.root) {
            state.seeding = seeding;
        }
    }
    if Instant::now() < deadline && state.gate_indexes.due() {
        refresh_champions(&args.root, state);
        refresh_strict_wins(&args.root, state);
    }
    if Instant::now() < deadline && state.gate_receipts.due() {
        scan_receipts(&args.root, state);
        state.feed_fetched_at_ms = now_ms();
    }
    if Instant::now() < deadline && state.gate_wave.due() {
        refresh_wave_progress(&args.root, state);
        if args.claims_db.is_some() && state.gate_claims.due() {
            refresh_claims(args, state);
        }
    }
}

fn discover_gpu_jobs(scq: &Value) -> Vec<(String, String, PathBuf, bool)> {
    scq.get("jobs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|job| string_at(job, &["state"]) == Some("running"))
        .filter(|job| {
            u64_at(job, &["gpus"]).unwrap_or(0) >= 1
                || job
                    .get("gpu_uuids")
                    .and_then(Value::as_array)
                    .is_some_and(|uuids| !uuids.is_empty())
        })
        .filter_map(|job| {
            let id = string_at(job, &["id"])?;
            let node = string_at(job, &["node"]).unwrap_or("-");
            let run_dir = string_at(job, &["run_dir"]).map(PathBuf::from)?;
            let is_trainer = id.starts_with(TRAINER_ID_PREFIX);
            Some((id.to_owned(), node.to_owned(), run_dir, is_trainer))
        })
        .collect()
}

fn parse_gpus(scq: &Value, jobs: &[(String, String, PathBuf, bool)]) -> Vec<GpuRow> {
    scq.get("gpus")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|gpu| {
            let node = string_at(gpu, &["node"])?.to_owned();
            let status = string_at(gpu, &["status"]).unwrap_or("-").to_owned();
            let owner_job = status
                .strip_prefix("allocated:")
                .map(str::to_owned)
                .or_else(|| {
                    jobs.iter()
                        .find(|(_, job_node, _, _)| *job_node == node)
                        .map(|(id, _, _, _)| id.clone())
                        .filter(|_| status != "free")
                });
            let model_hint = GPU_MODEL_HINTS
                .iter()
                .find(|(hint_node, _)| *hint_node == node)
                .map(|(_, model)| (*model).to_owned())
                .unwrap_or_else(|| "?".to_owned());
            Some(GpuRow {
                node,
                model_hint,
                status,
                owner_job,
                util_pct: f64_at(gpu, &["util_pct"]).unwrap_or(0.0),
                mem_used_mb: u64_at(gpu, &["mem_used_mb"]).unwrap_or(0),
                mem_total_mb: u64_at(gpu, &["mem_total_mb"]).unwrap_or(0),
                temp_c: u64_at(gpu, &["temp_c"]).unwrap_or(0),
                power_w: f64_at(gpu, &["power_w"]).unwrap_or(0.0),
                power_limit_w: f64_at(gpu, &["power_limit_w"]).unwrap_or(0.0),
                throttled: bool_at(gpu, &["throttled"]).unwrap_or(false),
            })
        })
        .collect()
}

fn refresh_trainers(args: &Args, jobs: &[(String, String, PathBuf, bool)], state: &mut SlowState) {
    let mut trainers = Vec::new();
    for (job_id, node, run_dir, is_trainer) in jobs {
        let run_dir = if run_dir.is_absolute() {
            run_dir.clone()
        } else {
            args.metrics_root.join(run_dir)
        };
        // Same attempt-keyed cache as `collect_dl_history`: a requeue starts a
        // fresh `attempts/<n>/` while the old attempt's train_progress.json
        // stays readable, so an `is_file()`-only guard would freeze TRAINER
        // STATE on the dead attempt for the rest of the job's life.
        let attempt = newest_attempt_id(&run_dir);
        let path = match state.progress_paths.get(job_id) {
            Some((cached_attempt, path)) if Some(*cached_attempt) == attempt && path.is_file() => {
                Some(path.clone())
            }
            _ => {
                let resolved = resolve_progress_file(&run_dir);
                if let (Some(path), Some(attempt)) = (&resolved, attempt) {
                    state
                        .progress_paths
                        .insert(job_id.clone(), (attempt, path.clone()));
                }
                resolved
            }
        };
        let mut detail = TrainerDetail {
            job_id: job_id.clone(),
            node: node.clone(),
            is_trainer: *is_trainer,
            ..TrainerDetail::default()
        };
        if let Some(path) = path {
            read_trainer_progress(&path, &mut detail);
            if let Some(existing) = state
                .trainers
                .iter()
                .find(|trainer| trainer.job_id == *job_id)
            {
                detail.history = existing.history.clone();
                detail.bad_passes = detail.bad_passes.max(existing.bad_passes);
                if detail.overfit_gap.is_none() {
                    detail.overfit_gap = existing.overfit_gap;
                }
            }
        }
        trainers.push(detail);
    }
    state
        .progress_paths
        .retain(|job_id, _| jobs.iter().any(|(id, _, _, _)| id == job_id));
    state.trainers = trainers;
}

/// Highest `attempts/<n>/` id present under a run dir, or `None` when the run
/// has not been staged yet. One cheap `read_dir` of a directory that holds a
/// handful of entries — used as the cache key for every resolved progress /
/// metrics path so an in-place SCQ requeue invalidates them.
fn newest_attempt_id(run_dir: &Path) -> Option<u64> {
    fs::read_dir(run_dir.join("attempts"))
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u64>().ok())
        })
        .max()
}

fn resolve_progress_file(run_dir: &Path) -> Option<PathBuf> {
    let attempts = run_dir.join("attempts");
    let mut attempt_ids = fs::read_dir(&attempts)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u64>().ok())
        })
        .collect::<Vec<_>>();
    attempt_ids.sort_unstable_by(|left, right| right.cmp(left));
    for attempt in attempt_ids {
        let outputs = attempts.join(attempt.to_string()).join("outputs");
        let mut best: Option<(SystemTime, PathBuf)> = None;
        walk_for_progress(&outputs, 0, &mut best);
        if let Some((_, path)) = best {
            return Some(path);
        }
    }
    None
}

fn walk_for_progress(dir: &Path, depth: usize, best: &mut Option<(SystemTime, PathBuf)>) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(|entry| entry.ok()).take(256) {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            walk_for_progress(&path, depth + 1, best);
        } else if file_type.is_file() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "train_progress.json" || name == "progress.json" {
                let modified = entry
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .unwrap_or(UNIX_EPOCH);
                if best.as_ref().is_none_or(|(prior, _)| modified > *prior) {
                    *best = Some((modified, path));
                }
            }
        }
    }
}

fn read_trainer_progress(path: &Path, detail: &mut TrainerDetail) {
    let Ok(bytes) = fs::read(path) else {
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    detail.phase = string_at(&value, &["phase"]).unwrap_or("-").to_owned();
    detail.epoch = u64_at(&value, &["epoch"]).unwrap_or(0);
    detail.epochs = u64_at(&value, &["epochs"]).unwrap_or(0);
    detail.batch = u64_at(&value, &["batch"]).unwrap_or(0);
    detail.total_batches = u64_at(&value, &["total_batches"]).unwrap_or(0);
    detail.pct = f64_at(&value, &["pct"]).unwrap_or(0.0);
    detail.running_loss = f64_at(&value, &["running_loss"]);
    detail.val_loss = f64_at(&value, &["val_loss"]);
    detail.label_watermark = u64_at(&value, &["active_label_watermark"]).unwrap_or(0);
    detail.elapsed_sec = f64_at(&value, &["elapsed_sec"]).unwrap_or(0.0);
    detail.validation_batch = u64_at(&value, &["validation_batch"]).unwrap_or(0);
    detail.validation_batches = u64_at(&value, &["validation_batches"]).unwrap_or(0);
    detail.progress_at_ms = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
}

fn parse_metric_row(row: &Value, fallback_epoch: u64) -> TrainerMetricPoint {
    TrainerMetricPoint {
        epoch: u64_at(row, &["epoch"]).unwrap_or(fallback_epoch),
        train_loss: f64_at(row, &["train_loss"]).unwrap_or(0.0),
        val_loss: f64_at(row, &["val_loss"]).unwrap_or(0.0),
        bad_passes: u64_at(row, &["macro_validation_bad_passes"]).unwrap_or(0),
        overfit_gap: f64_at(row, &["overfit_gap"]).unwrap_or(0.0),
        val_pairwise: f64_at(row, &["val_pairwise_win_accuracy"]),
        val_binary_pairwise: f64_at(row, &["val_binary_pairwise_win"])
            .or_else(|| f64_at(row, &["val_binary_pairwise_win_accuracy"])),
        val_regression_pairwise: f64_at(row, &["val_regression_pairwise_win"])
            .or_else(|| f64_at(row, &["val_regression_pairwise_win_accuracy"])),
        val_reward_cv: f64_at(row, &["val_reward_cv_correlation"]),
        val_top1_regret: f64_at(row, &["val_top1_historical_regret"]),
        val_conditional_advantage: f64_at(row, &["val_conditional_advantage"]),
        val_ablation_pairwise: f64_at(row, &["val_ablation_pairwise_win"]),
    }
}

/// Full-history read (call sites gate on mtime), downsampled for charting while
/// always preserving the newest point.
fn read_metrics_history(path: &Path) -> Vec<TrainerMetricPoint> {
    read_metrics_points(path, DL_HISTORY_MAX)
}

fn read_metrics_points(path: &Path, cap: usize) -> Vec<TrainerMetricPoint> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let points = text
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            serde_json::from_str::<Value>(line)
                .ok()
                .map(|row| parse_metric_row(&row, index as u64 + 1))
        })
        .collect::<Vec<_>>();
    downsample_points(points, cap)
}

fn downsample_points(points: Vec<TrainerMetricPoint>, cap: usize) -> Vec<TrainerMetricPoint> {
    if points.len() <= cap || cap < 2 {
        return points;
    }
    let last = points.len() - 1;
    let stride = (points.len() as f64 / (cap - 1) as f64).ceil() as usize;
    let mut sampled = points
        .iter()
        .step_by(stride.max(1))
        .cloned()
        .collect::<Vec<_>>();
    if sampled.last().map(|point| point.epoch) != Some(points[last].epoch) {
        sampled.push(points[last].clone());
    }
    sampled
}

/// (status file, observed age seconds, expected cadence seconds) for the DL tab.
fn read_daemon_ages(root: &Path) -> Vec<(String, u64, u64)> {
    DAEMON_HEARTBEATS
        .iter()
        .map(|(relative, expected)| {
            let age = fs::metadata(root.join(relative))
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|modified| SystemTime::now().duration_since(modified).ok())
                .map(|duration| duration.as_secs())
                .unwrap_or(u64::MAX);
            let name = relative
                .trim_start_matches("ops/")
                .trim_end_matches(".json")
                .to_owned();
            (name, age, *expected)
        })
        .collect()
}

/// Last recorded val_pairwise_win_accuracy per trainer generation, old→new —
/// the tab-4 improvement strip. Derived from the already-collected DL history
/// so the run dirs are walked once per pass, not twice.
fn generations_from_history(history: &DlHistory) -> Vec<(String, Option<f64>)> {
    let skip = history.runs.len().saturating_sub(GENERATIONS_MAX);
    history
        .runs
        .iter()
        .skip(skip)
        .map(|run| {
            (
                run.run_label.clone(),
                run.points.last().and_then(|point| point.val_pairwise),
            )
        })
        .collect()
}

/// Compact generation tag: `001805437-a-g8` → `a-g8`. The leading all-digit
/// segment is the label watermark, identical across a family, so it carries no
/// information on the ruler.
fn short_generation_label(run_label: &str) -> String {
    match run_label.split_once('-') {
        Some((head, tail)) if !head.is_empty() && head.chars().all(|c| c.is_ascii_digit()) => {
            tail.to_owned()
        }
        _ => run_label.to_owned(),
    }
}

/// Rebuild the cross-generation DL history from the trainer run directories.
///
/// This is deliberately keyed on the run-dir name and NOT on the live SCQ job
/// id: trainer jobs complete, get cancelled and get resubmitted under fresh ids
/// (a-g8 → c-r000002 → a-g9 inside one hour on 2026-07-28), and the previous
/// design hung the whole loss series off `state.trainers`, which is rebuilt
/// every pass from running jobs only. The chart therefore emptied the moment a
/// trainer exited. Run dirs persist, so this cannot blank on job turnover.
fn collect_dl_history(
    metrics_root: &Path,
    live_runs: &HashSet<String>,
    state: &mut SlowState,
) -> DlHistory {
    let Ok(entries) = fs::read_dir(metrics_root) else {
        // Keep the last good history rather than blanking the panel.
        return state.dl_history.clone();
    };
    let mut found: Vec<(SystemTime, String, PathBuf)> = Vec::new();
    for entry in entries.filter_map(|entry| entry.ok()) {
        let run_id = entry.file_name().to_string_lossy().to_string();
        if !run_id.starts_with(TRAINER_ID_PREFIX) {
            continue;
        }
        // Cache the resolved metrics path against the attempt it was resolved
        // from. SCQ requeues a job in place under a fresh `attempts/<n>/` and
        // LEAVES the previous attempt on disk (verified 2026-07-28: 23 jope-*
        // run dirs carry 2+ attempts, one carries 0..100, and a-g9 itself
        // reports "attempt 1 (requeues 1)"). An `is_file()`-only guard would
        // therefore pin this chart to a dead attempt for the rest of the run.
        let attempt = newest_attempt_id(&entry.path());
        let cached = state
            .dl_metrics_paths
            .get(&run_id)
            .filter(|(cached_attempt, path)| Some(*cached_attempt) == attempt && path.is_file())
            .map(|(_, path)| path.clone());
        let metrics = match cached {
            Some(path) => path,
            None => {
                let Some(path) = resolve_progress_file(&entry.path()).and_then(|progress| {
                    progress
                        .parent()
                        .map(|dir| dir.join("train_metrics.jsonl"))
                        .filter(|path| path.is_file())
                }) else {
                    continue;
                };
                if let Some(attempt) = attempt {
                    state
                        .dl_metrics_paths
                        .insert(run_id.clone(), (attempt, path.clone()));
                }
                path
            }
        };
        let Ok(modified) = fs::metadata(&metrics).and_then(|meta| meta.modified()) else {
            continue;
        };
        found.push((modified, run_id, metrics));
    }
    // Oldest → newest by last write, which for sequential trainer generations is
    // the order they ran in.
    found.sort_by(|left, right| left.0.cmp(&right.0));
    let skip = found.len().saturating_sub(DL_GENERATIONS_MAX);
    let kept = found.split_off(skip);
    let mut runs = Vec::with_capacity(kept.len());
    for (modified, run_id, metrics) in kept {
        let reuse = state
            .dl_points_cache
            .get(&run_id)
            .filter(|(cached_at, _)| *cached_at == modified)
            .map(|(_, points)| points.clone());
        let points = match reuse {
            Some(points) => points,
            None => {
                let points = read_metrics_points(&metrics, DL_RUN_POINTS_MAX);
                state
                    .dl_points_cache
                    .insert(run_id.clone(), (modified, points.clone()));
                points
            }
        };
        let run_label = run_id.trim_start_matches("jope-selfplay-train-").to_owned();
        runs.push(RunSeries {
            short_label: short_generation_label(&run_label),
            run_label,
            points,
            live: live_runs.contains(&run_id),
            updated_at_ms: modified
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or(0),
        });
    }
    // Drop caches for generations that fell out of the window so a long-lived
    // TUI cannot accumulate every trainer run ever made.
    let kept_ids = runs
        .iter()
        .map(|run| format!("jope-selfplay-train-{}", run.run_label))
        .collect::<HashSet<_>>();
    state.dl_points_cache.retain(|id, _| kept_ids.contains(id));
    state.dl_metrics_paths.retain(|id, _| kept_ids.contains(id));
    DlHistory {
        runs,
        fetched_at_ms: now_ms(),
    }
}

/// Flatten `DlHistory` onto one continuous epoch axis: generation *k* starts
/// where generation *k-1* ended, so warm-started runs read as one curve.
fn continuous_series(history: &DlHistory) -> DlSeries {
    let mut out = DlSeries::default();
    let mut offset = 0.0_f64;
    for run in &history.runs {
        if run.points.is_empty() {
            continue;
        }
        out.marks
            .push((offset + 1.0, run.short_label.clone(), run.live));
        let mut span = 0.0_f64;
        for point in &run.points {
            let x = offset + point.epoch as f64;
            // parse_metric_row defaults absent losses to 0.0; plotting those
            // would drop the curve to the floor on any partially written row.
            if point.train_loss.is_finite() && point.train_loss > 0.0 {
                out.train.push((x, point.train_loss));
            }
            if point.val_loss.is_finite() && point.val_loss > 0.0 {
                out.val.push((x, point.val_loss));
            }
            if let Some(value) = point.val_pairwise.filter(|value| value.is_finite()) {
                out.pairwise.push((x, value));
            }
            if let Some(value) = point
                .val_ablation_pairwise
                .filter(|value| value.is_finite())
            {
                out.ablation.push((x, value));
            }
            span = span.max(point.epoch as f64);
        }
        offset += span;
    }
    out.total_epochs = offset;
    out
}

/// Seeding curve (`ops/seeding-curve.v1.json`, written by the ingestion
/// reconcile loop): per model-checkpoint generation, the DL strict-win share
/// and DL observation share for the tab-4 generations panel DL-share column.
fn read_seeding_curve(root: &Path) -> Option<SeedingCurveView> {
    let bytes = fs::read(root.join("ops/seeding-curve.v1.json")).ok()?;
    let value = serde_json::from_slice::<Value>(&bytes).ok()?;
    let points = value
        .get("curve")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|point| SeedingPoint {
            checkpoint_short: string_at(point, &["model_checkpoint_sha256"])
                .unwrap_or("?")
                .chars()
                .take(8)
                .collect(),
            dl_win_share_pct: f64_at(point, &["dl_strict_win_share_pct"]),
            dl_obs_share_pct: f64_at(point, &["observations", "dl_observation_share"])
                .map(|share| share * 100.0),
            strict_wins_total: u64_at(point, &["strict_wins_total"]).unwrap_or(0),
            observations_caught_up: bool_at(point, &["observations", "caught_up"]).unwrap_or(false),
        })
        .collect();
    Some(SeedingCurveView {
        points,
        fetched_at_ms: now_ms(),
    })
}

fn read_watchdog(root: &Path) -> Option<TrainerWatchdogView> {
    let bytes = fs::read(root.join("ops/trainer-watchdog-v1.json")).ok()?;
    let value = serde_json::from_slice::<Value>(&bytes).ok()?;
    Some(TrainerWatchdogView {
        labels: u64_at(&value, &["labels"]).unwrap_or(0),
        snapshot_watermark: u64_at(&value, &["snapshot_watermark"]).unwrap_or(0),
        last_evidence_watermark: u64_at(&value, &["last_evidence_watermark"]).unwrap_or(0),
        next_label_threshold: u64_at(&value, &["next_label_threshold"]).unwrap_or(0),
        pending_unfrozen_labels: u64_at(&value, &["pending_unfrozen_labels"]).unwrap_or(0),
        checkpoint_sha256: string_at(&value, &["checkpoint_sha256"])
            .unwrap_or("-")
            .to_owned(),
        action: string_at(&value, &["action"]).unwrap_or("-").to_owned(),
        active_trainers: value
            .get("active_trainers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        generated_at_ms: u64_at(&value, &["generated_at_ms"]).unwrap_or(0) as u128,
    })
}

fn parse_accepted_task_fronts(value: Option<&Value>) -> Vec<AcceptedTaskFront> {
    let Some(tasks) = value
        .and_then(|value| value.get("tasks"))
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };
    ["binary", "regression"]
        .into_iter()
        .filter_map(|task| {
            let row = tasks.get(task)?;
            Some(AcceptedTaskFront {
                task: task.to_owned(),
                checkpoint_sha256: string_at(row, &["checkpoint_sha256"])
                    .unwrap_or("-")
                    .to_owned(),
                generation: u64_at(row, &["generation"]).unwrap_or(0),
                updated_at_ms: u64_at(row, &["updated_at_ms"]).unwrap_or(0) as u128,
            })
        })
        .collect()
}

/// Alarm files rendered in the alerts panel, in display order.
///
/// `ops/utilization-alarms.v1.json` is written by jope_utilization_supervisor
/// and is deliberately a SEPARATE file from `health-alarms.v1.json`:
/// jope_selfplay_expansion_driver.rs:116-141 halts dataset activation on any
/// critical row in health-alarms, so putting a critical `cluster_idle` row
/// there would stop the flywheel activating datasets *because the cluster is
/// idle* — the exact starvation loop that emptied all 1,052 slots on
/// 2026-07-28. The TUI is a display, so it can safely read both.
const ALARM_SOURCES: &[(&str, &str)] = &[
    ("ops/utilization-alarms.v1.json", "utilization"),
    ("ops/health-alarms.v1.json", "alarms"),
];

fn read_health_alarms(root: &Path, states: &mut Vec<SourceState>) -> Vec<String> {
    let mut rows = Vec::new();
    for (source, label) in ALARM_SOURCES {
        rows.extend(read_alarm_file(&root.join(source), label, states));
    }
    rows
}

fn read_alarm_file(path: &Path, label: &str, states: &mut Vec<SourceState>) -> Vec<String> {
    let Ok(bytes) = fs::read(path) else {
        return Vec::new();
    };
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) => value
            .get("alarms")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|alarm| {
                let name = string_at(alarm, &["alarm"])?;
                let severity = string_at(alarm, &["severity"]).unwrap_or("warn");
                let observed = alarm
                    .get("observed")
                    .map(Value::to_string)
                    .unwrap_or_else(|| "-".to_owned());
                let threshold = alarm
                    .get("threshold")
                    .map(Value::to_string)
                    .unwrap_or_else(|| "-".to_owned());
                Some(format!(
                    "ALARM[{severity}] {name}: {observed} (thr {threshold})"
                ))
            })
            .collect(),
        Err(error) => {
            states.push(SourceState {
                name: label.to_owned(),
                healthy: false,
                note: format!("invalid JSON: {error}"),
            });
            Vec::new()
        }
    }
}

fn refresh_champions(root: &Path, state: &mut SlowState) {
    let path = root.join("indexes/champions.json");
    let Ok(modified) = fs::metadata(&path).and_then(|metadata| metadata.modified()) else {
        return;
    };
    if state.champions_mtime == Some(modified) {
        return;
    }
    let Ok(bytes) = fs::read(&path) else {
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        state.source_states.push(SourceState {
            name: "champions".to_owned(),
            healthy: false,
            note: "invalid JSON".to_owned(),
        });
        return;
    };
    let entries = value
        .as_array()
        .cloned()
        .or_else(|| value.get("champions").and_then(Value::as_array).cloned())
        .unwrap_or_default();
    let mut champions = HashMap::with_capacity(entries.len());
    for entry in &entries {
        let Some(task) = string_at(entry, &["task"]) else {
            continue;
        };
        let Some(dataset) = string_at(entry, &["dataset_hash"]) else {
            continue;
        };
        let incumbent = f64_at(entry, &["recorded_incumbent", "score"]);
        let strict = if bool_at(entry, &["strictly_improves"]).unwrap_or(false) {
            f64_at(entry, &["best_new_strict", "score"])
        } else {
            None
        };
        let best = match (incumbent, strict) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (Some(left), None) => Some(left),
            (None, right) => right,
        };
        if let Some(best) = best {
            champions.insert((task.to_owned(), dataset.to_owned()), best);
        }
    }
    state.champions = champions;
    state.champions_mtime = Some(modified);
}

fn refresh_strict_wins(root: &Path, state: &mut SlowState) {
    let path = root.join("indexes/hall-of-fame.json");
    let Ok(modified) = fs::metadata(&path).and_then(|metadata| metadata.modified()) else {
        return;
    };
    if state.hof_mtime == Some(modified) {
        return;
    }
    let Ok(bytes) = fs::read(&path) else {
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    let wins = value
        .get("strict_wins")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if state.hof_mtime.is_some() && wins.len() > state.hof_seen {
        for win in wins.iter().skip(state.hof_seen) {
            let event = GameplayEvent {
                at_ms: now_ms(),
                task: string_at(win, &["task"]).unwrap_or("-").to_owned(),
                dataset_hash: string_at(win, &["dataset_hash"]).unwrap_or("-").to_owned(),
                score: f64_at(win, &["new_score"]),
                delta_vs_champion: f64_at(win, &["delta"]),
                origin: if string_at(win, &["checkpoint_origin"]).is_some() {
                    Origin::DlProposal
                } else {
                    Origin::Unknown
                },
                node: "-".to_owned(),
                wave: "-".to_owned(),
                valid: true,
                kind: EventKind::StrictWin,
            };
            push_ring(&mut state.ring, event);
        }
    }
    state.hof_seen = wins.len();
    state.hof_mtime = Some(modified);
}

fn push_ring(ring: &mut VecDeque<GameplayEvent>, event: GameplayEvent) {
    if ring.len() >= GAMEPLAY_RING {
        ring.pop_front();
    }
    ring.push_back(event);
}

fn scan_receipts(root: &Path, state: &mut SlowState) {
    let receipts_root = root.join("receipts");
    let Ok(entries) = fs::read_dir(&receipts_root) else {
        state.source_states.push(SourceState {
            name: "receipts".to_owned(),
            healthy: false,
            note: "unreadable".to_owned(),
        });
        return;
    };
    let now = now_ms();
    let first_pass = !state.receipts_prefilled;
    let mut fresh: Vec<(String, PathBuf, SystemTime)> = Vec::new();
    for entry in entries.filter_map(|entry| entry.ok()) {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with("jte-") {
            continue;
        }
        if state.known_receipt_dirs.contains(name) {
            continue;
        }
        if first_pass {
            let modified = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(UNIX_EPOCH);
            let age_ms = modified
                .duration_since(UNIX_EPOCH)
                .map(|duration| now.saturating_sub(duration.as_millis()))
                .unwrap_or(u128::MAX);
            if age_ms <= RECEIPT_PREFILL_WINDOW_MS {
                fresh.push((name.to_owned(), entry.path(), modified));
            } else {
                // Historical dir: acknowledge without processing.
                state.known_receipt_dirs.insert(name.to_owned());
            }
        } else {
            fresh.push((name.to_owned(), entry.path(), UNIX_EPOCH));
        }
    }
    let limit = if first_pass {
        fresh.sort_by(|left, right| right.2.cmp(&left.2));
        state.receipts_prefilled = true;
        RECEIPT_PREFILL_MAX_DIRS
    } else {
        RECEIPT_DIRS_PER_PASS
    };
    if first_pass {
        // Anything beyond the prefill cap is historical backlog: acknowledge unprocessed.
        for (name, _, _) in fresh.iter().skip(limit) {
            state.known_receipt_dirs.insert(name.clone());
        }
    }
    for (name, dir, _) in fresh.into_iter().take(limit) {
        state.known_receipt_dirs.insert(name);
        ingest_receipt_dir(root, &dir, state);
    }
}

fn ingest_receipt_dir(root: &Path, dir: &Path, state: &mut SlowState) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut receipt_files = Vec::new();
    let mut done_at_ms = 0u128;
    let mut done_node = String::from("-");
    for entry in entries.filter_map(|entry| entry.ok()) {
        let name = entry.file_name();
        let Some(name) = name.to_str().map(str::to_owned) else {
            continue;
        };
        if name.ends_with(".receipts.jsonl") {
            receipt_files.push(entry.path());
        } else if name.ends_with(".done") {
            if let Ok(bytes) = fs::read(entry.path()) {
                if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
                    done_at_ms = u64_at(&value, &["finished_at_ms"]).unwrap_or(0) as u128;
                    done_node = string_at(&value, &["node"]).unwrap_or("-").to_owned();
                }
            }
        }
    }
    for file in receipt_files {
        let Ok(contents) = fs::read_to_string(&file) else {
            continue;
        };
        let mut top: Option<(f64, Origin, String, String)> = None;
        let mut manifest_dl_slots = 0u64;
        let mut manifest_total = 0u64;
        for line in contents.lines() {
            let Ok(row) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let task = string_at(&row, &["task"]).unwrap_or("-").to_owned();
            let dataset = string_at(&row, &["dataset_hash"]).unwrap_or("-").to_owned();
            let wave = string_at(&row, &["generation"]).unwrap_or("-").to_owned();
            let valid = bool_at(&row, &["valid"]).unwrap_or(false);
            let score = if valid {
                f64_at(&row, &["score"])
            } else {
                None
            };
            let semantic = string_at(&row, &["semantic_sha256"]).unwrap_or("");
            let manifest_sha = string_at(&row, &["manifest_sha256"]).unwrap_or("");
            let origin = resolve_origin(root, &wave, manifest_sha, semantic, state);
            manifest_total += 1;
            if origin == Origin::DlProposal {
                manifest_dl_slots += 1;
            }
            let champion = state
                .champions
                .get(&(task.clone(), dataset.clone()))
                .copied();
            let delta = match (score, champion) {
                (Some(score), Some(champion)) => Some(score - champion),
                _ => None,
            };
            if let Some(score_value) = score {
                if top
                    .as_ref()
                    .is_none_or(|(best, _, _, _)| score_value > *best)
                {
                    top = Some((score_value, origin, task.clone(), dataset.clone()));
                }
            }
            let at_ms = u64_at(&row, &["created_at_ms"])
                .map(u128::from)
                .unwrap_or(done_at_ms);
            if at_ms > state.newest_receipt_at_ms {
                state.newest_receipt_at_ms = at_ms;
            }
            if !state.newest_wave_name.is_empty() && wave == state.newest_wave_name {
                state.newest_wave_receipts += 1;
            }
            push_ring(
                &mut state.ring,
                GameplayEvent {
                    at_ms,
                    task,
                    dataset_hash: dataset,
                    score,
                    delta_vs_champion: delta,
                    origin,
                    node: string_at(&row, &["node"]).unwrap_or(&done_node).to_owned(),
                    wave,
                    valid,
                    kind: EventKind::Eval,
                },
            );
        }
        if manifest_total > 0 {
            state.manifests_seen += 1;
            state.dl_slots += manifest_dl_slots;
            state.total_slots += manifest_total;
            if let Some((top_score, origin, task, dataset)) = top {
                let beats = state
                    .champions
                    .get(&(task, dataset))
                    .is_some_and(|champion| top_score > *champion);
                if state.stats_window.len() >= GAMEPLAY_STATS_WINDOW {
                    state.stats_window.pop_front();
                }
                state.stats_window.push_back((origin, beats));
            }
        }
    }
}

fn resolve_origin(
    root: &Path,
    wave: &str,
    manifest_sha: &str,
    semantic: &str,
    state: &mut SlowState,
) -> Origin {
    if wave == "-" || manifest_sha.is_empty() || semantic.is_empty() {
        return Origin::Unknown;
    }
    if !state.manifest_semantics.contains_key(manifest_sha) {
        let relative = wave_manifest_path(root, wave, manifest_sha, state);
        let mut semantics = HashMap::new();
        if let Some(relative) = relative {
            let manifest_path = root.join("waves").join(wave).join(relative);
            if let Ok(bytes) = fs::read(&manifest_path) {
                if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
                    for candidate in value
                        .get("candidates")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        if let Some(sha) = string_at(candidate, &["semantic_sha256"]) {
                            semantics.insert(
                                sha.to_owned(),
                                (
                                    string_at(candidate, &["source"]).unwrap_or("").to_owned(),
                                    string_at(candidate, &["slot_group"])
                                        .unwrap_or("")
                                        .to_owned(),
                                ),
                            );
                        }
                    }
                }
            }
        }
        if state.manifest_order.len() >= MANIFEST_CACHE_MAX {
            if let Some(evicted) = state.manifest_order.pop_front() {
                state.manifest_semantics.remove(&evicted);
            }
        }
        state.manifest_order.push_back(manifest_sha.to_owned());
        state
            .manifest_semantics
            .insert(manifest_sha.to_owned(), semantics);
    }
    state
        .manifest_semantics
        .get(manifest_sha)
        .and_then(|semantics| semantics.get(semantic))
        .map(|(source, slot_group)| slot_group_origin(slot_group, source))
        .unwrap_or(Origin::Unknown)
}

fn slot_group_origin(slot_group: &str, source: &str) -> Origin {
    match slot_group {
        "dl_proposal" => Origin::DlProposal,
        "offspring" => Origin::GpOffspring,
        "local_elite" => Origin::LocalElite,
        "common_winner" => Origin::CommonWinner,
        _ => {
            let lowered = source.to_ascii_lowercase();
            if lowered.contains("dl")
                || lowered.contains("proposal")
                || lowered.contains("checkpoint")
            {
                Origin::DlProposal
            } else if lowered.contains("mutation") || lowered.contains("crossover") {
                Origin::GpOffspring
            } else if lowered.contains("elite") {
                Origin::LocalElite
            } else if lowered.contains("winner") || lowered.contains("robust") {
                Origin::CommonWinner
            } else {
                Origin::Unknown
            }
        }
    }
}

fn wave_manifest_path(
    root: &Path,
    wave: &str,
    manifest_sha: &str,
    state: &mut SlowState,
) -> Option<String> {
    if !state.wave_manifest_paths.contains_key(wave) {
        let mut map = HashMap::new();
        let summary = root.join("waves").join(wave).join("summary.json");
        if let Ok(bytes) = fs::read(&summary) {
            if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
                for manifest in value
                    .get("manifests")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let sha = string_at(manifest, &["sha256"])
                        .or_else(|| string_at(manifest, &["manifest_sha256"]));
                    let path = string_at(manifest, &["relative_path"])
                        .or_else(|| string_at(manifest, &["path"]));
                    if let (Some(sha), Some(path)) = (sha, path) {
                        map.insert(sha.to_owned(), path.to_owned());
                    }
                }
            }
        }
        if state.wave_manifest_paths.len() >= 3 {
            let oldest = state.wave_manifest_paths.keys().min().cloned();
            if let Some(oldest) = oldest {
                state.wave_manifest_paths.remove(&oldest);
            }
        }
        state.wave_manifest_paths.insert(wave.to_owned(), map);
    }
    state
        .wave_manifest_paths
        .get(wave)
        .and_then(|map| map.get(manifest_sha))
        .cloned()
}

fn refresh_wave_progress(root: &Path, state: &mut SlowState) {
    let waves_root = root.join("waves");
    let Ok(entries) = fs::read_dir(&waves_root) else {
        return;
    };
    let newest = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
        .filter(|name| name.starts_with("strict-wave-"))
        .max();
    let Some(newest) = newest else {
        return;
    };
    if newest != state.newest_wave_name {
        state.newest_wave_name = newest.clone();
        state.newest_wave_receipts = 0;
    }
    let summary = waves_root.join(&newest).join("summary.json");
    let Ok(bytes) = fs::read(&summary) else {
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    let datasets = u64_at(&value, &["datasets"]).unwrap_or_else(|| {
        value
            .get("datasets")
            .and_then(Value::as_array)
            .map(|values| values.len() as u64)
            .unwrap_or(0)
    });
    let claims = state
        .wave_progress
        .take()
        .filter(|progress| progress.wave == newest)
        .map(|progress| progress.claims_by_state)
        .unwrap_or_default();
    state.wave_progress = Some(WaveProgress {
        wave: newest,
        manifest_count: u64_at(&value, &["manifest_count"]).unwrap_or_else(|| {
            value
                .get("manifests")
                .and_then(Value::as_array)
                .map(|values| values.len() as u64)
                .unwrap_or(0)
        }),
        datasets,
        receipts_landed: state.newest_wave_receipts,
        claims_by_state: claims,
        generated_at_ms: u64_at(&value, &["generated_at_ms"]).unwrap_or(0) as u128,
    });
}

fn refresh_claims(args: &Args, state: &mut SlowState) {
    let Some(path) = args.claims_db.as_ref() else {
        return;
    };
    if state
        .claims_backoff_until
        .is_some_and(|until| Instant::now() < until)
    {
        return;
    }
    match query_claims(path) {
        Ok(counts) => {
            state.claims_backoff = Duration::from_secs(10);
            state.claims_backoff_until = None;
            if let Some(progress) = state.wave_progress.as_mut() {
                progress.claims_by_state = counts;
            }
        }
        Err(error) => {
            state.source_states.push(SourceState {
                name: "claims-db".to_owned(),
                healthy: false,
                note: format!("busy/err: {error:#}"),
            });
            state.claims_backoff_until = Some(Instant::now() + state.claims_backoff);
            state.claims_backoff = (state.claims_backoff * 2).min(Duration::from_secs(60));
        }
    }
}

fn query_claims(path: &Path) -> Result<BTreeMap<String, u64>> {
    use rusqlite::OpenFlags;
    let connection = rusqlite::Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(Duration::from_millis(250))?;
    connection.pragma_update(None, "query_only", "ON")?;
    let mut statement = connection.prepare("SELECT state, COUNT(*) FROM claims GROUP BY state")?;
    let mut counts = BTreeMap::new();
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let state: String = row.get(0)?;
        let count: u64 = row.get(1)?;
        counts.insert(state, count);
    }
    Ok(counts)
}

const FUNNEL_HISTORY_CAP: usize = 200;

/// Canonical funnel display order. Every one of these tasks always gets a row:
/// tasks the data sources predate fall back to a corpus census so a pending
/// corpus is visible as "waiting" instead of silently absent.
const FUNNEL_TASKS: [&str; 4] = ["regression", "binary", "timeseries", "multiclass"];

/// On-disk corpus root for tasks the dataset-state index may not know yet.
/// "multi_label" is the multiclass corpus's legacy directory name. Regression
/// and binary are always present in the index, so they never need a census.
fn funnel_census_root(task: &str) -> Option<&'static str> {
    match task {
        "timeseries" => Some("/home/ubuntu/remote_super/quant/timeseries"),
        "multiclass" => Some("/home/ubuntu/remote_super/quant/multi_label"),
        _ => None,
    }
}

/// Installed-encodings root for census tasks: a dataset dir here means its
/// sidecars are encoded and awaiting catalog activation — the funnel's
/// "queued" stage semantics exactly.
fn funnel_encoded_root(task: &str) -> Option<&'static str> {
    match task {
        "timeseries" => Some("/home/ubuntu/jope-flywheel/encodings/timeseries-expanded"),
        "multiclass" => Some("/home/ubuntu/jope-flywheel/encodings/multiclass-expanded"),
        _ => None,
    }
}

fn count_dataset_dirs(root: &str) -> Option<u64> {
    let entries = fs::read_dir(root).ok()?;
    Some(
        entries
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .filter(|entry| !entry.file_name().to_string_lossy().starts_with('.'))
            .count() as u64,
    )
}

/// Census for tasks absent from the dataset-state index: (corpus datasets,
/// installed encodings). Both cheap directory counts (≤ a few hundred entries).
fn funnel_census_count(task: &str) -> Option<(u64, u64)> {
    let corpus = count_dataset_dirs(funnel_census_root(task)?)?;
    let encoded = funnel_encoded_root(task)
        .and_then(count_dataset_dirs)
        .unwrap_or(0);
    Some((corpus, encoded))
}

fn funnel_tag(task: &str) -> &'static str {
    match task {
        "regression" => "REG",
        "binary" => "BIN",
        "timeseries" => "TS",
        "multiclass" | "multi_label" => "MC",
        _ => "?",
    }
}

fn funnel_tag_compact(task: &str) -> &'static str {
    match task {
        "regression" => "reg",
        "binary" => "bin",
        "timeseries" => "ts",
        "multiclass" | "multi_label" => "mc",
        _ => "?",
    }
}

/// Ensure every funnel task has a row and impose canonical display order.
/// Missing tasks get census rows: registry = corpus size, queued = installed
/// encodings (encoded, awaiting activation), and `locked()` derives the
/// remainder still waiting for encoding.
fn complete_funnel_rows(rows: &mut Vec<FunnelRow>, census: impl Fn(&str) -> Option<(u64, u64)>) {
    for task in FUNNEL_TASKS {
        if !rows.iter().any(|row| row.task == task) {
            let (registry, encoded) = census(task).unwrap_or((0, 0));
            rows.push(FunnelRow {
                task: task.to_owned(),
                registry,
                encoded,
                queued: encoded.min(registry),
                ..Default::default()
            });
        }
    }
    let position = |task: &str| {
        FUNNEL_TASKS
            .iter()
            .position(|known| *known == task)
            .unwrap_or(FUNNEL_TASKS.len())
    };
    rows.sort_by(|a, b| position(&a.task).cmp(&position(&b.task)).then(a.task.cmp(&b.task)));
}

/// Freshest source: the expansion driver's per-dataset lifecycle index.
fn funnel_rows_from_index(path: &Path) -> Result<Vec<FunnelRow>> {
    use rusqlite::OpenFlags;
    let connection = rusqlite::Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(Duration::from_millis(250))?;
    connection.pragma_update(None, "query_only", "ON")?;
    // Full pipeline query; older index files without sealed/encode_job_state
    // columns fall back to the legacy stages with the new ones zeroed.
    let extended = connection.prepare(
        "SELECT task, COUNT(*), SUM(sidecars_complete), SUM(catalog_active), \
         SUM(CASE WHEN catalog_active=1 AND obs_count>0 THEN 1 ELSE 0 END), \
         SUM(CASE WHEN sidecars_complete=1 AND catalog_active=0 AND sealed=0 \
             THEN 1 ELSE 0 END), \
         SUM(CASE WHEN encode_job_state IS NOT NULL AND sidecars_complete=0 \
             THEN 1 ELSE 0 END), \
         SUM(sealed) \
         FROM datasets GROUP BY task ORDER BY task DESC",
    );
    let mut out = Vec::new();
    match extended {
        Ok(mut statement) => {
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                out.push(FunnelRow {
                    task: row.get(0)?,
                    registry: row.get::<_, i64>(1)?.max(0) as u64,
                    encoded: row.get::<_, Option<i64>>(2)?.unwrap_or(0).max(0) as u64,
                    active: row.get::<_, Option<i64>>(3)?.unwrap_or(0).max(0) as u64,
                    in_play: row.get::<_, Option<i64>>(4)?.unwrap_or(0).max(0) as u64,
                    queued: row.get::<_, Option<i64>>(5)?.unwrap_or(0).max(0) as u64,
                    encoding: row.get::<_, Option<i64>>(6)?.unwrap_or(0).max(0) as u64,
                    sealed: row.get::<_, Option<i64>>(7)?.unwrap_or(0).max(0) as u64,
                    ..Default::default()
                });
            }
        }
        Err(_) => {
            let mut statement = connection.prepare(
                "SELECT task, COUNT(*), SUM(sidecars_complete), SUM(catalog_active), \
                 SUM(CASE WHEN catalog_active=1 AND obs_count>0 THEN 1 ELSE 0 END) \
                 FROM datasets GROUP BY task ORDER BY task DESC",
            )?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                out.push(FunnelRow {
                    task: row.get(0)?,
                    registry: row.get::<_, i64>(1)?.max(0) as u64,
                    encoded: row.get::<_, Option<i64>>(2)?.unwrap_or(0).max(0) as u64,
                    active: row.get::<_, Option<i64>>(3)?.unwrap_or(0).max(0) as u64,
                    in_play: row.get::<_, Option<i64>>(4)?.unwrap_or(0).max(0) as u64,
                    ..Default::default()
                });
            }
        }
    }
    Ok(out)
}

/// Fallback: the expansion driver's embedded stage summary (refreshed each cycle).
fn funnel_rows_from_driver(value: &Value) -> Vec<FunnelRow> {
    let summary = value.get("stage_summary");
    FUNNEL_TASKS
        .iter()
        .filter_map(|task| {
            let stage = summary.and_then(|section| section.get(*task))?;
            Some(FunnelRow {
                task: (*task).to_owned(),
                registry: u64_at(stage, &["registry"]).unwrap_or(0),
                encoded: u64_at(stage, &["sidecars_complete"]).unwrap_or(0),
                active: u64_at(stage, &["catalog_active"]).unwrap_or(0),
                in_play: u64_at(stage, &["observed"]).unwrap_or(0),
                queued: u64_at(stage, &["queued"]).unwrap_or(0),
                encoding: u64_at(stage, &["encoding"]).unwrap_or(0),
                sealed: u64_at(stage, &["sealed"]).unwrap_or(0),
                ..Default::default()
            })
        })
        .collect()
}

/// Climb indicator: compare against the oldest sample inside the last hour.
fn apply_funnel_deltas(
    rows: &mut [FunnelRow],
    history: &mut VecDeque<(u128, Vec<(String, u64, u64)>)>,
    now: u128,
) {
    let sample = rows
        .iter()
        .map(|row| (row.task.clone(), row.active, row.in_play))
        .collect::<Vec<_>>();
    history.push_back((now, sample));
    while history.len() > FUNNEL_HISTORY_CAP {
        history.pop_front();
    }
    let cutoff = now.saturating_sub(3_600_000);
    let baseline = history
        .iter()
        .find(|(at, _)| *at >= cutoff)
        .or_else(|| history.front());
    if let Some((_, base)) = baseline {
        for row in rows.iter_mut() {
            if let Some((_, active, in_play)) = base.iter().find(|(task, _, _)| *task == row.task) {
                row.delta_active_1h = row.active as i64 - *active as i64;
                row.delta_in_play_1h = row.in_play as i64 - *in_play as i64;
            }
        }
    }
}

fn refresh_unlock_funnel(root: &Path, state: &mut SlowState) {
    let (mut rows, source) = match funnel_rows_from_index(&root.join("ops/dataset-state.sqlite")) {
        Ok(rows) if !rows.is_empty() => (rows, "index"),
        _ => {
            let driver = fs::read_to_string(root.join("ops/expansion-driver-v1.json"))
                .ok()
                .and_then(|text| serde_json::from_str::<Value>(&text).ok());
            match driver.as_ref().map(funnel_rows_from_driver) {
                Some(rows) if !rows.is_empty() => (rows, "driver"),
                _ => {
                    let rows = fs::read_to_string(root.join("catalog/current.json"))
                        .ok()
                        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
                        .and_then(|pointer| {
                            let path = pointer.get("catalog").and_then(Value::as_str)?.to_owned();
                            let catalog: Value =
                                serde_json::from_str(&fs::read_to_string(path).ok()?).ok()?;
                            let mut rows = vec![
                                FunnelRow {
                                    task: "regression".to_owned(),
                                    active: u64_at(&catalog, &["regression_count"]).unwrap_or(0),
                                    ..Default::default()
                                },
                                FunnelRow {
                                    task: "binary".to_owned(),
                                    active: u64_at(&catalog, &["binary_count"]).unwrap_or(0),
                                    ..Default::default()
                                },
                            ];
                            // Newer tasks only when the catalog knows them: an
                            // absent key must fall through to the corpus census
                            // instead of rendering a misleading zero registry.
                            for task in ["timeseries", "multiclass"] {
                                let key = format!("{task}_count");
                                if let Some(active) = u64_at(&catalog, &[key.as_str()]) {
                                    rows.push(FunnelRow {
                                        task: task.to_owned(),
                                        active,
                                        ..Default::default()
                                    });
                                }
                            }
                            Some(rows)
                        })
                        .unwrap_or_default();
                    (rows, "catalog")
                }
            }
        }
    };
    if rows.is_empty() {
        return; // keep the previous snapshot rather than blanking the panel
    }
    complete_funnel_rows(&mut rows, funnel_census_count);
    let now = now_ms();
    apply_funnel_deltas(&mut rows, &mut state.funnel_history, now);
    state.funnel = UnlockFunnel {
        rows,
        source: source.to_owned(),
        fetched_at_ms: now,
    };
}

const STAGE_ACTIVE: Color = Color::Green;
const STAGE_IN_PLAY: Color = Color::LightGreen;
const STAGE_QUEUED: Color = Color::Yellow;
const STAGE_ENCODING: Color = Color::Cyan;
const STAGE_LOCKED: Color = Color::DarkGray;

/// Colored per-task pipeline line (shared by tabs 1 and 4).
fn funnel_line(row: &FunnelRow) -> Line<'static> {
    let tag = funnel_tag(&row.task);
    let mut spans = vec![Span::styled(
        format!("{tag} {} ", thousands(row.registry)),
        Style::default().add_modifier(Modifier::BOLD),
    )];
    spans.push(Span::styled(
        format!("● active {} ", thousands(row.active)),
        Style::default().fg(STAGE_ACTIVE),
    ));
    spans.push(Span::styled(
        format!("(in-play {}) ", thousands(row.in_play)),
        Style::default()
            .fg(STAGE_IN_PLAY)
            .add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(
        format!("● queued +{} ", thousands(row.queued)),
        Style::default().fg(STAGE_QUEUED),
    ));
    spans.push(Span::styled(
        format!("● encoding {} ", thousands(row.encoding)),
        Style::default().fg(STAGE_ENCODING),
    ));
    spans.push(Span::styled(
        format!(
            "● locked {} (sealed {}) ",
            thousands(row.locked()),
            thousands(row.sealed)
        ),
        Style::default().fg(STAGE_LOCKED),
    ));
    spans.push(Span::styled(
        format!(
            "Δ1h active {:+} / in-play {:+}",
            row.delta_active_1h, row.delta_in_play_1h
        ),
        Style::default().fg(Color::Gray),
    ));
    Line::from(spans)
}

/// Stacked proportional bar for one task: active/queued/encoding/locked+sealed.
fn funnel_bar(row: &FunnelRow, width: usize) -> Line<'static> {
    let total = row.registry.max(1) as f64;
    let cells = |count: u64| ((count as f64 / total) * width as f64).round() as usize;
    let mut spans = Vec::new();
    for (count, color) in [
        (row.active, STAGE_ACTIVE),
        (row.queued, STAGE_QUEUED),
        (row.encoding, STAGE_ENCODING),
        (row.locked() + row.sealed, STAGE_LOCKED),
    ] {
        let n = cells(count);
        if n > 0 {
            spans.push(Span::styled("█".repeat(n), Style::default().fg(color)));
        }
    }
    Line::from(spans)
}

fn compact_funnel_line(dashboard: &Dashboard) -> Paragraph<'static> {
    if dashboard.funnel.rows.is_empty() {
        return Paragraph::new(Line::from(Span::styled(
            " DATASETS  collecting…",
            Style::default().fg(Color::Cyan),
        )));
    }
    let mut spans = vec![Span::styled(
        " DATASETS  ",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )];
    for (index, row) in dashboard.funnel.rows.iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw("  •  "));
        }
        let tag = funnel_tag_compact(&row.task);
        spans.push(Span::styled(
            format!("{tag} "),
            Style::default().add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!(
                "{}/{} active ",
                thousands(row.active),
                thousands(row.registry)
            ),
            Style::default().fg(STAGE_ACTIVE),
        ));
        spans.push(Span::styled(
            format!("({} play) ", thousands(row.in_play)),
            Style::default()
                .fg(STAGE_IN_PLAY)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!("+{} queued ", thousands(row.queued)),
            Style::default().fg(STAGE_QUEUED),
        ));
        spans.push(Span::styled(
            format!("{} encoding ", thousands(row.encoding)),
            Style::default().fg(STAGE_ENCODING),
        ));
        spans.push(Span::styled(
            format!("{:+}Δ1h", row.delta_active_1h),
            Style::default().fg(Color::Gray),
        ));
    }
    Paragraph::new(Line::from(spans))
}

fn build_dashboard(raw: &RawSources, slow: &SlowView) -> Dashboard {
    let mut dashboard = Dashboard {
        collected_at_ms: raw.collected_at_ms,
        collection_latency_ms: raw.collection_latency_ms,
        sources_healthy: raw
            .source_states
            .iter()
            .filter(|state| state.healthy)
            .count()
            + slow
                .source_states
                .iter()
                .filter(|state| state.healthy)
                .count(),
        sources_total: raw.source_states.len() + slow.source_states.len(),
        gpus: slow.gpus.clone(),
        trainers: slow.trainers.clone(),
        watchdog: slow.watchdog.clone(),
        accepted_tasks: parse_accepted_task_fronts(raw.accepted_by_task.as_ref()),
        gameplay: slow.gameplay.clone(),
        gameplay_stats: slow.gameplay_stats.clone(),
        wave_progress: slow.wave_progress.clone(),
        daemon_ages: slow.daemon_ages.clone(),
        generations: slow.generations.clone(),
        seeding: slow.seeding.clone(),
        dl_history: slow.dl_history.clone(),
        funnel: slow.funnel.clone(),
        feed_fetched_at_ms: slow.feed_fetched_at_ms,
        ..Dashboard::default()
    };
    let scq = raw
        .scq
        .as_ref()
        .and_then(|value| value.get("Snapshot"))
        .unwrap_or(&Value::Null);
    dashboard.scq_epoch = u64_at(scq, &["epoch"]);
    dashboard.scq_ts = string_at(scq, &["ts"]).unwrap_or("-").to_owned();
    dashboard.nodes = parse_nodes(scq);
    dashboard.cluster = aggregate_cluster(scq, &dashboard.nodes);
    dashboard.families = parse_families(scq);
    dashboard.queue = parse_queue(raw, scq, &dashboard.families);
    dashboard.learning = parse_learning(raw, scq);
    if let Some(watchdog) = dashboard.watchdog.as_ref() {
        dashboard.learning.labels = watchdog.labels;
        dashboard.learning.frozen_watermark = watchdog.snapshot_watermark;
        if watchdog.checkpoint_sha256 != "-" {
            dashboard.learning.checkpoint = watchdog.checkpoint_sha256.clone();
        }
        if let Some(trainer) = dashboard.trainers.iter().find(|trainer| trainer.is_trainer) {
            dashboard.learning.trainer = trainer.job_id.clone();
            dashboard.learning.trainer_state = if trainer.phase == "-" {
                "running".to_owned()
            } else {
                trainer.phase.clone()
            };
        } else if let Some(trainer) = watchdog.active_trainers.first() {
            dashboard.learning.trainer = trainer.clone();
            dashboard.learning.trainer_state = watchdog.action.clone();
        }
    }
    dashboard.wave = parse_wave(raw);
    dashboard.events = parse_events(scq, &raw.local_events);
    dashboard.failures = parse_failures(scq);
    dashboard.alerts = collect_alerts(raw, scq, &dashboard);
    // Alarm-FILE rows go first, ahead of the alerts this TUI derives itself.
    // render_alerts() only draws the first five, and it was drawing five
    // derived rows while a critical `cluster_idle` row sat below the fold —
    // which is the same failure mode as not raising the alarm at all.
    // ALARM_SOURCES already orders utilization ahead of health.
    for (index, alarm) in slow.health_alarms.iter().enumerate() {
        dashboard.alerts.insert(index, alarm.clone());
    }
    let now = raw.collected_at_ms;
    if slow.newest_receipt_at_ms > 0 && now.saturating_sub(slow.newest_receipt_at_ms) > 300_000 {
        dashboard.alerts.push(format!(
            "gameplay feed stalled: newest receipt {}s old",
            now.saturating_sub(slow.newest_receipt_at_ms) / 1_000
        ));
    }
    for trainer in &slow.trainers {
        if trainer.progress_at_ms > 0 && now.saturating_sub(trainer.progress_at_ms) > 600_000 {
            dashboard.alerts.push(format!(
                "{} progress {}s stale while running",
                short(&trainer.job_id, 32),
                now.saturating_sub(trainer.progress_at_ms) / 1_000
            ));
        }
    }
    dashboard
}

fn parse_nodes(scq: &Value) -> Vec<NodeRow> {
    let mut rows = scq
        .get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| {
            let node = string_at(value, &["node"])?;
            let up = bool_at(value, &["up"]).unwrap_or(false);
            let effective_cores = u64_at(value, &["effective_cores"])
                .unwrap_or_else(|| u64_at(value, &["cores"]).unwrap_or(0));
            let multiplier = f64_at(value, &["cpu_job_slot_multiplier"]).unwrap_or(1.0);
            let slot_target = u64_at(value, &["job_slot_target"])
                .or_else(|| u64_at(value, &["slot_target"]))
                .unwrap_or_else(|| (effective_cores as f64 * multiplier).floor() as u64);
            Some(NodeRow {
                node: node.to_owned(),
                state: if up { "up" } else { "down" }.to_owned(),
                health: string_at(value, &["health"])
                    .unwrap_or("unknown")
                    .to_owned(),
                physical_cores: u64_at(value, &["cores"]).unwrap_or(0),
                effective_cores,
                multiplier,
                slot_target,
                running_slots: u64_at(value, &["running_job_slots"])
                    .or_else(|| u64_at(value, &["running_jobs"]))
                    .unwrap_or(0),
                cpu_busy_pct: f64_at(value, &["cpu_busy_pct"]),
                psi_cpu10: f64_at(value, &["psi_cpu10"]),
                psi_mem10: f64_at(value, &["psi_mem10"]),
                psi_io10: f64_at(value, &["psi_io10"]),
                memory_available_gb: u64_at(value, &["mem_avail_gb"]).unwrap_or(0),
                memory_total_gb: u64_at(value, &["mem_total_gb"]).unwrap_or(0),
                disk_use_pct: f64_at(value, &["disk_use_pct"]),
            })
        })
        .collect::<Vec<_>>();
    let present = rows
        .iter()
        .map(|row| row.node.as_str())
        .collect::<BTreeSet<_>>();
    let missing = EXPECTED_NODES
        .iter()
        .filter(|node| !present.contains(**node))
        .map(|node| NodeRow {
            node: (*node).to_owned(),
            state: "absent".to_owned(),
            health: "no SCQ status".to_owned(),
            ..NodeRow::default()
        })
        .collect::<Vec<_>>();
    rows.extend(missing);
    rows.sort_by(|left, right| left.node.cmp(&right.node));
    rows
}

fn aggregate_cluster(scq: &Value, nodes: &[NodeRow]) -> ClusterSummary {
    let active = nodes
        .iter()
        .filter(|node| node.state == "up")
        .collect::<Vec<_>>();
    let weighted_cores = active
        .iter()
        .filter(|node| node.cpu_busy_pct.is_some())
        .map(|node| node.physical_cores)
        .sum::<u64>();
    let cpu_busy_pct = if weighted_cores == 0 {
        f64_at(scq, &["counters", "cpu_busy_pct"]).unwrap_or(0.0)
    } else {
        active
            .iter()
            .map(|node| node.cpu_busy_pct.unwrap_or(0.0) * node.physical_cores as f64)
            .sum::<f64>()
            / weighted_cores as f64
    };
    ClusterSummary {
        cpu_busy_pct,
        psi_cpu10: maximum(active.iter().filter_map(|node| node.psi_cpu10)),
        psi_mem10: maximum(active.iter().filter_map(|node| node.psi_mem10)),
        psi_io10: maximum(active.iter().filter_map(|node| node.psi_io10)),
        physical_cores: active.iter().map(|node| node.physical_cores).sum(),
        effective_cores: active.iter().map(|node| node.effective_cores).sum(),
        slot_target: active.iter().map(|node| node.slot_target).sum(),
        running_slots: active.iter().map(|node| node.running_slots).sum(),
        memory_available_gb: active.iter().map(|node| node.memory_available_gb).sum(),
        nodes_up: active.len(),
        nodes_total: nodes.iter().filter(|node| node.state != "absent").count(),
        jobs_failed: u64_at(scq, &["counters", "jobs_failed"]).unwrap_or(0),
        throughput_per_minute: u64_at(scq, &["counters", "throughput_per_min"]).unwrap_or(0),
    }
}

fn parse_families(scq: &Value) -> Vec<FamilyRow> {
    let mut jobs_by_family: BTreeMap<String, FamilyRow> = BTreeMap::new();
    if let Some(jobs) = scq.get("jobs").and_then(Value::as_array) {
        for job in jobs {
            let family = string_at(job, &["project"]).unwrap_or("unknown");
            let state = string_at(job, &["state"]).unwrap_or("unknown");
            let row = jobs_by_family
                .entry(family.to_owned())
                .or_insert_with(|| FamilyRow {
                    family: family.to_owned(),
                    ..FamilyRow::default()
                });
            match state {
                "queued" => {
                    row.queued += 1;
                    if let Some(timestamp) = string_at(job, &["queue_ts"]) {
                        if row
                            .oldest_queue_ts
                            .as_deref()
                            .is_none_or(|prior| timestamp < prior)
                        {
                            row.oldest_queue_ts = Some(timestamp.to_owned());
                        }
                    }
                }
                "starting" => row.starting += 1,
                "dispatching" => row.dispatching += 1,
                "running" => row.running += 1,
                "held" => row.held += 1,
                _ => {}
            }
        }
    }
    if let Some(families) = scq.get("families").and_then(Value::as_array) {
        for family in families {
            let Some(name) = string_at(family, &["family"]) else {
                continue;
            };
            let row = jobs_by_family
                .entry(name.to_owned())
                .or_insert_with(|| FamilyRow {
                    family: name.to_owned(),
                    ..FamilyRow::default()
                });
            row.weight = f64_at(family, &["effective_weight"]).unwrap_or(1.0);
            row.queued = u64_at(family, &["queued_jobs"]).unwrap_or(row.queued);
            row.running = u64_at(family, &["running_jobs"]).unwrap_or(row.running);
            row.held = u64_at(family, &["held_jobs"]).unwrap_or(row.held);
        }
    }
    let mut rows = jobs_by_family.into_values().collect::<Vec<_>>();
    rows.retain(|row| {
        row.queued + row.starting + row.dispatching + row.running + row.held > 0
            || row.weight >= 10.0
            || row.family.starts_with("jope")
    });
    rows.sort_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then_with(|| active_jobs(right).cmp(&active_jobs(left)))
            .then_with(|| left.family.cmp(&right.family))
    });
    rows
}

fn parse_queue(raw: &RawSources, scq: &Value, families: &[FamilyRow]) -> QueueSummary {
    let queue = raw.queue.as_ref().unwrap_or(&Value::Null);
    let freshness = raw.freshness.as_ref().unwrap_or(&Value::Null);
    let counters = scq.get("counters").unwrap_or(&Value::Null);
    QueueSummary {
        queued: u64_at(counters, &["jobs_queued"])
            .unwrap_or_else(|| families.iter().map(|row| row.queued).sum()),
        starting: families.iter().map(|row| row.starting).sum(),
        dispatching: u64_at(queue, &["dispatching"])
            .unwrap_or_else(|| families.iter().map(|row| row.dispatching).sum()),
        running: u64_at(counters, &["jobs_running"])
            .or_else(|| u64_at(queue, &["running"]))
            .unwrap_or_else(|| families.iter().map(|row| row.running).sum()),
        held: u64_at(counters, &["jobs_held"])
            .or_else(|| u64_at(queue, &["held"]))
            .unwrap_or_else(|| families.iter().map(|row| row.held).sum()),
        useful_depth: u64_at(queue, &["queued_starting"]).unwrap_or(0),
        target: u64_at(queue, &["queued_target"]).unwrap_or(0),
        hard_cap: u64_at(queue, &["queued_hard_cap"]).unwrap_or(0),
        maximum_age_seconds: f64_at(queue, &["queue_maximum_age_seconds"]).unwrap_or(0.0),
        epoch: string_at(queue, &["checkpoint_epoch"])
            .or_else(|| string_at(freshness, &["checkpoint_epoch"]))
            .unwrap_or("-")
            .to_owned(),
        stale_epochs: u64_at(freshness, &["stale_epoch_count"]).unwrap_or(0),
        stale_held: u64_at(freshness, &["stale_held"]).unwrap_or(0),
        admission_allowed: bool_at(queue, &["admission_allowed"]),
        mode: string_at(queue, &["mode"]).unwrap_or("degraded").to_owned(),
    }
}

fn parse_learning(raw: &RawSources, scq: &Value) -> LearningSummary {
    let status = raw.status.as_ref().unwrap_or(&Value::Null);
    let trainer = status.get("trainer").unwrap_or(&Value::Null);
    let ingestion = status.get("ingestion").unwrap_or(&Value::Null);
    let fallback = raw.trainer_fallback.as_ref().unwrap_or(&Value::Null);
    let proposal = raw.proposal_summary.as_ref().unwrap_or(&Value::Null);
    let flywheel = raw.flywheel.as_ref().unwrap_or(&Value::Null);
    let trainer_id = trainer
        .get("active_trainers")
        .and_then(Value::as_array)
        .and_then(|values| values.first())
        .and_then(Value::as_str)
        .or_else(|| string_at(trainer, &["trainer"]))
        .or_else(|| string_at(fallback, &["job_id"]))
        .unwrap_or("-");
    let trainer_state = scq
        .get("jobs")
        .and_then(Value::as_array)
        .and_then(|jobs| {
            jobs.iter().find(|job| {
                string_at(job, &["id"]).is_some_and(|candidate| candidate == trainer_id)
            })
        })
        .and_then(|job| string_at(job, &["state"]))
        .unwrap_or("not in SCQ");
    let promotion = status.get("promotion").unwrap_or(&Value::Null);
    LearningSummary {
        checkpoint: string_at(
            raw.queue.as_ref().unwrap_or(&Value::Null),
            &["active_checkpoint_sha256"],
        )
        .or_else(|| {
            string_at(
                raw.freshness.as_ref().unwrap_or(&Value::Null),
                &["checkpoint_sha256"],
            )
        })
        .or_else(|| string_at(proposal, &["model_checkpoint_sha256"]))
        .unwrap_or("-")
        .to_owned(),
        trainer: trainer_id.to_owned(),
        trainer_state: trainer_state.to_owned(),
        labels: u64_at(trainer, &["labels"])
            .or_else(|| u64_at(status, &["adaptive_fitness_watermark"]))
            .or_else(|| u64_at(ingestion, &["exact_rows_total"]))
            .unwrap_or(0),
        frozen_watermark: u64_at(flywheel, &["last_frozen_watermark"]).unwrap_or(0),
        internal_promotion: promotion_label(promotion.get("internal_learning")),
        milestone_promotion: promotion_label(promotion.get("milestone")),
        proposal_regression: u64_at(proposal, &["counts", "regression"]).unwrap_or(0),
        proposal_binary: u64_at(proposal, &["counts", "binary"]).unwrap_or(0),
        proposal_validity: f64_at(proposal, &["proposal_validity"]),
        consecutive_rejections: u64_at(status, &["evolution", "consecutive_rejections"])
            .or_else(|| u64_at(flywheel, &["consecutive_rejections"]))
            .unwrap_or(0),
    }
}

fn parse_wave(raw: &RawSources) -> WaveSummary {
    let status = raw.status.as_ref().unwrap_or(&Value::Null);
    let queue = raw.queue.as_ref().unwrap_or(&Value::Null);
    let supply = raw.supply.as_ref().unwrap_or(&Value::Null);
    let catalog = status.get("catalog").unwrap_or(&Value::Null);
    let sealed = status.get("sealed_audit").unwrap_or(&Value::Null);
    let ingestion = status.get("ingestion").unwrap_or(&Value::Null);
    let pairs = status.get("pair_registry").unwrap_or(&Value::Null);
    WaveSummary {
        active_wave: string_at(queue, &["newest_micro_wave"])
            .or_else(|| string_at(status, &["active_micro_cohort", "wave"]))
            .unwrap_or("-")
            .to_owned(),
        admission_waves: queue
            .get("admission_waves")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        unclaimed_manifests: u64_at(queue, &["unclaimed_current_checkpoint_manifests"])
            .or_else(|| u64_at(queue, &["manifests_unclaimed"]))
            .unwrap_or(0),
        ready_manifests: u64_at(supply, &["ready_manifests"]).unwrap_or(0),
        awaiting_keeper_ack: string_at(supply, &["awaiting_keeper_ack"])
            .unwrap_or("-")
            .to_owned(),
        adaptive_regression: u64_at(catalog, &["task_counts", "regression"]).unwrap_or(0),
        adaptive_binary: u64_at(catalog, &["task_counts", "binary"]).unwrap_or(0),
        sealed_isolated: bool_at(sealed, &["isolated_from_training"]),
        sealed_scored: u64_at(sealed, &["states", "scored"]).unwrap_or(0),
        corpus_rows: u64_at(status, &["adaptive_fitness_watermark"])
            .or_else(|| u64_at(ingestion, &["exact_rows_total"]))
            .unwrap_or(0),
        strict_wins: u64_at(ingestion, &["strict_wins"]).unwrap_or(0),
        pair_count: u64_at(pairs, &["pairs"]).unwrap_or(0),
        duplicate_pairs_prevented: u64_at(pairs, &["duplicates_prevented"]).unwrap_or(0),
        evidence_bytes: raw.evidence_bytes.unwrap_or(0),
    }
}

fn parse_events(scq: &Value, local: &[Value]) -> Vec<String> {
    let mut events = scq
        .get("events")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .rev()
        .take(8)
        .map(format_event)
        .collect::<Vec<_>>();
    events.extend(local.iter().take(6).map(format_event));
    events.truncate(12);
    events
}

fn parse_failures(scq: &Value) -> Vec<String> {
    scq.get("jobs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|job| string_at(job, &["state"]) == Some("failed"))
        .rev()
        .take(8)
        .map(|job| {
            format!(
                "{} {} {}",
                short(string_at(job, &["id"]).unwrap_or("-"), 24),
                string_at(job, &["failure_class"]).unwrap_or("failed"),
                short(string_at(job, &["message"]).unwrap_or(""), 48)
            )
        })
        .collect()
}

fn collect_alerts(raw: &RawSources, scq: &Value, dashboard: &Dashboard) -> Vec<String> {
    let mut alerts = BTreeSet::new();
    for source in raw.source_states.iter().filter(|source| !source.healthy) {
        alerts.insert(format!(
            "source {}: {}",
            source.name,
            short(&source.note, 80)
        ));
    }
    add_stale_source_alert(
        &mut alerts,
        "status",
        raw.status.as_ref(),
        "updated_at_ms",
        raw.collected_at_ms,
        120_000,
    );
    add_stale_source_alert(
        &mut alerts,
        "queue-v3",
        raw.queue.as_ref(),
        "generated_at_ms",
        raw.collected_at_ms,
        15_000,
    );
    add_stale_source_alert(
        &mut alerts,
        "supply",
        raw.supply.as_ref(),
        "generated_at_ms",
        raw.collected_at_ms,
        30_000,
    );
    if raw.collection_latency_ms >= 300 {
        alerts.insert(format!(
            "refresh latency {}ms exceeds 300ms target",
            raw.collection_latency_ms
        ));
    }
    if dashboard.cluster.cpu_busy_pct >= CPU_HARD_GUARD_PCT {
        alerts.insert(format!(
            "cluster CPU {:.1}% reached 95% hard guard",
            dashboard.cluster.cpu_busy_pct
        ));
    } else if dashboard.cluster.cpu_busy_pct >= CPU_TARGET_PCT {
        alerts.insert(format!(
            "cluster CPU {:.1}% exceeds 80% target",
            dashboard.cluster.cpu_busy_pct
        ));
    }
    for node in &dashboard.nodes {
        if node
            .cpu_busy_pct
            .is_some_and(|cpu| cpu >= CPU_HARD_GUARD_PCT)
        {
            alerts.insert(format!(
                "{} CPU hard guard at {:.1}%",
                node.node,
                node.cpu_busy_pct.unwrap()
            ));
        }
        if node.node == "xbabe5" && node.state != "up" {
            alerts.insert(format!("xbabe5 {}", node.state));
        }
    }
    if dashboard.queue.hard_cap > 0 && dashboard.queue.useful_depth > dashboard.queue.hard_cap {
        alerts.insert(format!(
            "fresh queue {} exceeds hard cap {}",
            dashboard.queue.useful_depth, dashboard.queue.hard_cap
        ));
    }
    if dashboard.queue.maximum_age_seconds > 1_800.0 {
        alerts.insert(format!(
            "queued work age {:.0}s exceeds 30m",
            dashboard.queue.maximum_age_seconds
        ));
    }
    if dashboard.queue.stale_epochs > 0 || dashboard.queue.stale_held > 0 {
        alerts.insert(format!(
            "stale epochs {} / held {}",
            dashboard.queue.stale_epochs, dashboard.queue.stale_held
        ));
    }
    if dashboard.queue.admission_allowed == Some(false) {
        alerts.insert("fresh admission is blocked".to_owned());
    }
    if let Some(values) = scq.get("alerts").and_then(Value::as_array) {
        for value in values.iter().filter_map(Value::as_str) {
            alerts.insert(format!("SCQ: {value}"));
        }
    }
    if let Some(values) = raw
        .status
        .as_ref()
        .and_then(|value| value.get("alerts"))
        .and_then(Value::as_array)
    {
        for value in values.iter().filter_map(Value::as_str) {
            alerts.insert(format!("JOPE: {value}"));
        }
    }
    alerts.into_iter().collect()
}

fn add_stale_source_alert(
    alerts: &mut BTreeSet<String>,
    name: &str,
    value: Option<&Value>,
    timestamp_key: &str,
    now: u128,
    maximum_age_ms: u128,
) {
    let Some(timestamp) = value
        .and_then(|value| u64_at(value, &[timestamp_key]))
        .map(u128::from)
    else {
        return;
    };
    let age = now.saturating_sub(timestamp);
    if age > maximum_age_ms {
        alerts.insert(format!("{name} status stale by {}s", age / 1_000));
    }
}

fn render(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, ui: UiState) {
    let area = frame.area();
    if area.width < 100 || area.height < 32 {
        match ui.tab {
            Tab::Overview | Tab::Gameplay => render_compact(frame, dashboard, refresh_ms, area),
            Tab::GpuDl | Tab::Dl => render_compact_gpu_dl(frame, dashboard, refresh_ms, area),
        }
        return;
    }
    match ui.tab {
        Tab::Overview => render_overview(frame, dashboard, refresh_ms, area),
        Tab::GpuDl => render_gpu_tab(frame, dashboard, refresh_ms, area),
        Tab::Gameplay => render_gameplay_tab(frame, dashboard, refresh_ms, area),
        Tab::Dl => render_dl_tab(frame, dashboard, refresh_ms, area, ui.arm),
    }
}

fn footer_line(dashboard: &Dashboard, refresh_ms: u64) -> Paragraph<'static> {
    Paragraph::new(format!(
        "1:overview 2:gpu/dl 3:gameplay 4:DL-charts n/←/→:arm tab:cycle • q/esc quit • read-only • refresh {}ms • collect {}ms",
        refresh_ms, dashboard.collection_latency_ms
    ))
    .style(Style::default().fg(Color::DarkGray))
}

fn render_overview(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Min(12),
            Constraint::Length(9),
            Constraint::Length(8),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, dashboard, rows[0]);
    render_global(frame, dashboard, rows[1]);
    frame.render_widget(compact_funnel_line(dashboard), rows[2]);
    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(rows[3]);
    render_nodes(frame, dashboard, main[0]);
    render_queue(frame, dashboard, main[1]);
    let learning = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[4]);
    render_learning(frame, dashboard, learning[0]);
    render_wave(frame, dashboard, learning[1]);
    let tail = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[5]);
    render_alerts(frame, dashboard, tail[0]);
    render_events(frame, dashboard, tail[1]);
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[6]);
}

fn stale_suffix(now_ms_value: u128, fetched_at_ms: u128) -> String {
    if fetched_at_ms == 0 {
        return " [NO DATA]".to_owned();
    }
    let age = now_ms_value.saturating_sub(fetched_at_ms);
    if age > STALE_MS {
        format!(" [STALE {}s]", age / 1_000)
    } else {
        String::new()
    }
}

fn sparkline(values: &[f64]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return "-".to_owned();
    }
    let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (maximum - minimum).max(f64::EPSILON);
    finite
        .iter()
        .map(|value| {
            let bucket = (((value - minimum) / range) * 7.0).round() as usize;
            BARS[bucket.min(7)]
        })
        .collect()
}

fn origin_label(origin: Origin) -> (&'static str, Color) {
    match origin {
        Origin::DlProposal => ("dl", Color::Magenta),
        Origin::GpOffspring => ("gp", Color::Cyan),
        Origin::LocalElite => ("elite", Color::Green),
        Origin::CommonWinner => ("robust", Color::Blue),
        Origin::Unknown => ("?", Color::DarkGray),
    }
}

fn render_gpu_tab(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, area: Rect) {
    let trainer_rows = (dashboard.trainers.len().max(1) * 9) as u16;
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3 + dashboard.gpus.len().max(1) as u16),
            Constraint::Length(4),
            Constraint::Min(trainer_rows.min(27)),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, dashboard, rows[0]);

    let header = Row::new([
        "node",
        "gpu",
        "status/owner",
        "util",
        "mem MiB",
        "temp",
        "power W",
        "thr",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let gpu_rows = dashboard.gpus.iter().map(|gpu| {
        let color = if gpu.temp_c >= 80 {
            Color::Red
        } else if gpu.util_pct >= 1.0 {
            Color::Green
        } else {
            Color::DarkGray
        };
        Row::new(vec![
            Cell::from(gpu.node.clone()),
            Cell::from(format!("{}?", gpu.model_hint)),
            Cell::from(short(gpu.owner_job.as_deref().unwrap_or(&gpu.status), 40)),
            Cell::from(format!("{:.0}%", gpu.util_pct)),
            Cell::from(format!("{}/{}", gpu.mem_used_mb, gpu.mem_total_mb)),
            Cell::from(format!("{}C", gpu.temp_c)),
            Cell::from(format!("{:.0}/{:.0}", gpu.power_w, gpu.power_limit_w)),
            Cell::from(if gpu.throttled { "YES" } else { "-" }),
        ])
        .style(Style::default().fg(color))
    });
    frame.render_widget(
        Table::new(
            gpu_rows,
            [
                Constraint::Length(8),
                Constraint::Length(6),
                Constraint::Min(30),
                Constraint::Length(6),
                Constraint::Length(13),
                Constraint::Length(5),
                Constraint::Length(9),
                Constraint::Length(4),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(" GPUs (all nodes) ")
                .borders(Borders::ALL),
        ),
        rows[1],
    );

    let watchdog = dashboard
        .watchdog
        .as_ref()
        .map(|watchdog| {
            format!(
                "watchdog: labels {} • watermark {} • next {} • pending {}",
                human_count(watchdog.labels),
                human_count(watchdog.last_evidence_watermark),
                human_count(watchdog.next_label_threshold),
                human_count(watchdog.pending_unfrozen_labels)
            )
        })
        .unwrap_or_else(|| "watchdog: no status".to_owned());
    let fronts = if dashboard.accepted_tasks.is_empty() {
        "fronts: unavailable".to_owned()
    } else {
        format!(
            "fronts: {}",
            dashboard
                .accepted_tasks
                .iter()
                .map(|front| format!(
                    "{}={} (g{})",
                    front.task,
                    short_hash(&front.checkpoint_sha256),
                    front.generation
                ))
                .collect::<Vec<_>>()
                .join(" • ")
        )
    };
    frame.render_widget(
        Paragraph::new(vec![Line::from(watchdog), Line::from(fronts)]).block(
            Block::default()
                .title(" Trainer watchdog / task fronts ")
                .borders(Borders::ALL),
        ),
        rows[2],
    );

    if dashboard.trainers.is_empty() {
        frame.render_widget(
            Paragraph::new("no active GPU jobs").block(
                Block::default()
                    .title(" Deep learning (live) ")
                    .borders(Borders::ALL),
            ),
            rows[3],
        );
    } else {
        // Detail blocks want 9 rows each; with parallel arms that overflows the
        // pane. Fit as many as the space allows and account for the rest in a
        // trailing note rather than drawing past the edge.
        let available = rows[3].height as usize;
        let per_block = if dashboard.trainers.len() > 2 { 6 } else { 9 };
        let shown = dashboard.trainers.len().min(available / per_block.max(1));
        let overflow = dashboard.trainers.len().saturating_sub(shown);
        let mut constraints = (0..shown)
            .map(|_| Constraint::Length(per_block as u16))
            .collect::<Vec<_>>();
        if overflow > 0 {
            constraints.push(Constraint::Length(1));
        }
        if constraints.is_empty() {
            constraints.push(Constraint::Min(1));
        }
        let blocks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(rows[3]);
        for (index, trainer) in dashboard.trainers.iter().take(shown).enumerate() {
            if index >= blocks.len() {
                break;
            }
            render_trainer_detail(frame, dashboard, trainer, blocks[index]);
        }
        if overflow > 0 {
            if let Some(area) = blocks.get(shown) {
                frame.render_widget(
                    Paragraph::new(format!(
                        "+{overflow} more GPU job(s) — press 4 for per-arm charts"
                    ))
                    .style(Style::default().fg(Color::DarkGray)),
                    *area,
                );
            }
        }
    }
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[4]);
}

fn render_trainer_detail(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    trainer: &TrainerDetail,
    area: Rect,
) {
    let title = format!(
        " {} @ {}{} ",
        short(&trainer.job_id, 44),
        trainer.node,
        stale_suffix(dashboard.collected_at_ms, trainer.progress_at_ms)
    );
    let block = Block::default().title(title).borders(Borders::ALL);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(4)])
        .split(inner);
    let epoch_ratio = if trainer.epochs > 0 {
        (trainer.epoch as f64 / trainer.epochs as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let validating = trainer.phase.starts_with("validation");
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(if validating {
                Color::Magenta
            } else {
                Color::Cyan
            }))
            .ratio(epoch_ratio)
            .label(phase_label(trainer)),
        parts[0],
    );
    let val_history = trainer
        .history
        .iter()
        .map(|point| point.val_loss)
        .collect::<Vec<_>>();
    let trend = match trainer.history.len() {
        len if len >= 2 => {
            let last = trainer.history[len - 1].val_loss;
            let prior = trainer.history[len - 2].val_loss;
            if last < prior {
                "↓"
            } else if last > prior {
                "↑"
            } else {
                "→"
            }
        }
        _ => "-",
    };
    let early_stop = if trainer.early_stopped
        || (trainer.patience > 0 && trainer.bad_passes >= trainer.patience)
    {
        Span::styled(
            " EARLY-STOPPED ",
            Style::default().fg(Color::Black).bg(Color::Red),
        )
    } else if trainer.patience > 0 && trainer.bad_passes * 4 >= trainer.patience * 3 {
        Span::styled(
            format!(" bad {}/{} ", trainer.bad_passes, trainer.patience),
            Style::default().fg(Color::Black).bg(Color::Yellow),
        )
    } else {
        Span::raw(format!(
            "bad {}/{}",
            trainer.bad_passes,
            if trainer.patience > 0 {
                trainer.patience.to_string()
            } else {
                "?".to_owned()
            }
        ))
    };
    let last_point = trainer.history.last();
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "loss {} {}  val {} {}  gap {}",
                trainer
                    .running_loss
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "-".to_owned()),
                trend,
                trainer
                    .val_loss
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "-".to_owned()),
                sparkline(&val_history),
                trainer
                    .overfit_gap
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_else(|| "-".to_owned()),
            )),
            Line::from(vec![
                early_stop,
                Span::raw(format!(
                    "  pairwise {}  binary {}",
                    last_point
                        .and_then(|point| point.val_pairwise)
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "-".to_owned()),
                    last_point
                        .and_then(|point| point.val_binary_pairwise)
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "-".to_owned()),
                )),
            ]),
            Line::from(format!(
                "watermark {}  elapsed {:.0}m{}",
                human_count(trainer.label_watermark),
                trainer.elapsed_sec / 60.0,
                if trainer.is_trainer {
                    ""
                } else {
                    "  (generic GPU job)"
                }
            )),
        ]),
        parts[1],
    );
}

fn render_gameplay_tab(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, dashboard, rows[0]);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[1]);
    if let Some(progress) = &dashboard.wave_progress {
        let ratio = if progress.manifest_count > 0 {
            (progress.receipts_landed as f64 / progress.manifest_count as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let claims = if progress.claims_by_state.is_empty() {
            String::new()
        } else {
            format!(
                " • claims {}",
                progress
                    .claims_by_state
                    .iter()
                    .map(|(state, count)| format!("{state}:{count}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        };
        frame.render_widget(
            Gauge::default()
                .block(
                    Block::default()
                        .title(format!(" {} (receipts since TUI start) ", progress.wave))
                        .borders(Borders::ALL),
                )
                .gauge_style(Style::default().fg(Color::Cyan))
                .ratio(ratio)
                .label(format!(
                    "{}/{} manifests • {} datasets{}",
                    progress.receipts_landed, progress.manifest_count, progress.datasets, claims
                )),
            top[0],
        );
    } else {
        frame.render_widget(
            Paragraph::new("no wave summary")
                .block(Block::default().title(" Wave ").borders(Borders::ALL)),
            top[0],
        );
    }
    let stats = &dashboard.gameplay_stats;
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "manifests seen {} • DL top-1 {} / GP top-1 {} (window {})",
                stats.manifests_seen, stats.dl_top1, stats.gp_top1, GAMEPLAY_STATS_WINDOW
            )),
            Line::from(format!(
                "DL slot share {:.1}% • beats-champion DL {} / GP {}",
                stats.dl_slot_share_pct, stats.dl_beats_champion, stats.gp_beats_champion
            )),
        ])
        .block(
            Block::default()
                .title(" DL vs GP (rolling) ")
                .borders(Borders::ALL),
        ),
        top[1],
    );

    let wins = dashboard
        .gameplay
        .iter()
        .filter(|event| event.kind == EventKind::StrictWin)
        .take(5)
        .map(|event| {
            let (label, _) = origin_label(event.origin);
            ListItem::new(format!(
                "{} {} {} Δ{} [{}]",
                event.task,
                short(&event.dataset_hash, 16),
                event
                    .score
                    .map(|value| format!("{value:.5}"))
                    .unwrap_or_else(|| "-".to_owned()),
                event
                    .delta_vs_champion
                    .map(|value| format!("{value:+.5}"))
                    .unwrap_or_else(|| "-".to_owned()),
                label
            ))
            .style(
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        })
        .collect::<Vec<_>>();
    let wins = if wins.is_empty() {
        vec![ListItem::new("no strict wins observed since TUI start")
            .style(Style::default().fg(Color::DarkGray))]
    } else {
        wins
    };
    frame.render_widget(
        List::new(wins).block(
            Block::default()
                .title(" Strict wins (live) ")
                .borders(Borders::ALL),
        ),
        rows[2],
    );

    let feed_title = format!(
        " Evaluation feed{} ",
        stale_suffix(dashboard.collected_at_ms, dashboard.feed_fetched_at_ms)
    );
    let header = Row::new([
        "time", "T", "dataset", "score", "Δ champ", "origin", "node", "wave",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let feed_rows = dashboard
        .gameplay
        .iter()
        .filter(|event| event.kind == EventKind::Eval)
        .take(rows[3].height.saturating_sub(3) as usize)
        .map(|event| {
            let (label, color) = origin_label(event.origin);
            let delta_cell = event
                .delta_vs_champion
                .map(|value| format!("{value:+.5}"))
                .unwrap_or_else(|| "-".to_owned());
            let style = if !event.valid {
                Style::default().fg(Color::DarkGray)
            } else if event.delta_vs_champion.is_some_and(|value| value > 0.0) {
                Style::default().fg(Color::Green)
            } else {
                Style::default()
            };
            Row::new(vec![
                Cell::from(clock_string(event.at_ms)),
                Cell::from(match event.task.as_str() {
                    "regression" => "R",
                    "binary" => "B",
                    "timeseries" => "T",
                    "multiclass" | "multi_label" => "M",
                    _ => "?",
                }),
                Cell::from(short(&event.dataset_hash, 18)),
                Cell::from(
                    event
                        .score
                        .map(|value| format!("{value:.5}"))
                        .unwrap_or_else(|| "invalid".to_owned()),
                ),
                Cell::from(delta_cell),
                Cell::from(Span::styled(label, Style::default().fg(color))),
                Cell::from(event.node.clone()),
                Cell::from(short(
                    event
                        .wave
                        .strip_prefix("strict-wave-")
                        .unwrap_or(&event.wave),
                    8,
                )),
            ])
            .style(style)
        });
    frame.render_widget(
        Table::new(
            feed_rows,
            [
                Constraint::Length(9),
                Constraint::Length(2),
                Constraint::Length(19),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(8),
            ],
        )
        .header(header)
        .block(Block::default().title(feed_title).borders(Borders::ALL)),
        rows[3],
    );
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[4]);
}

fn phase_label(trainer: &TrainerDetail) -> String {
    if trainer.phase.starts_with("validation") {
        format!(
            "epoch {}/{} • VALIDATION {}/{} • training batches done",
            trainer.epoch, trainer.epochs, trainer.validation_batch, trainer.validation_batches
        )
    } else {
        format!(
            "epoch {}/{} • batch {}/{} ({:.0}%) • {}",
            trainer.epoch,
            trainer.epochs,
            trainer.batch,
            trainer.total_batches,
            trainer.pct,
            trainer.phase
        )
    }
}

fn trend_arrow(prior: f64, last: f64) -> &'static str {
    if last < prior {
        "↓"
    } else if last > prior {
        "↑"
    } else {
        "→"
    }
}

// The per-epoch chart series (train/val loss, pairwise, ablation) are produced
// by `continuous_series` from the cross-generation `DlHistory`; the old
// single-run extractors keyed on one trainer's history were what emptied the
// chart on job turnover and have been removed.

/// Training arms in a stable display order: real trainers first (so a proposal
/// or encoder GPU job never steals arm 0), each keeping its collector ordering.
fn training_arms(dashboard: &Dashboard) -> Vec<&TrainerDetail> {
    let mut arms: Vec<&TrainerDetail> =
        dashboard.trainers.iter().filter(|t| t.is_trainer).collect();
    arms.extend(dashboard.trainers.iter().filter(|t| !t.is_trainer));
    arms
}

/// Newest metric row for an arm, or None while its metrics file is still empty.
/// Callers must render "warming up" rather than borrowing another arm's numbers.
fn arm_latest(trainer: &TrainerDetail) -> Option<&TrainerMetricPoint> {
    trainer.history.last()
}

fn gpu_for_job<'a>(dashboard: &'a Dashboard, job_id: &str) -> Option<&'a GpuRow> {
    dashboard
        .gpus
        .iter()
        .find(|gpu| gpu.owner_job.as_deref() == Some(job_id))
}

fn render_dl_tab(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
    arm: usize,
) {
    let arms = training_arms(dashboard);
    // The comparison strip is sized to the live arm count so a single-arm run
    // keeps the charts as tall as before.
    let strip_height = if arms.len() > 1 {
        arms.len() as u16 + 2
    } else {
        0
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(strip_height),
            Constraint::Min(16),
            Constraint::Length(1),
        ])
        .split(area);
    if strip_height > 0 {
        render_arm_strip(frame, dashboard, &arms, arm, rows[0]);
    }
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(rows[1]);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(11),
            Constraint::Length(1),
            Constraint::Length(8),
            Constraint::Length(6),
        ])
        .split(columns[0]);
    // The charts are driven by the on-disk generation history, not by whichever
    // job happens to be running, so they keep rendering across trainer
    // changeovers and while a successor is still staging.
    let series = continuous_series(&dashboard.dl_history);
    let live = arms.get(selected_arm(arm, arms.len())).copied();
    render_loss_chart(frame, dashboard, &series, live, left[0]);
    frame.render_widget(
        Paragraph::new(generation_ruler(&series, left[1].width)),
        left[1],
    );
    render_pairwise_chart(frame, &series, left[2]);
    match live {
        Some(trainer) => render_dl_status(frame, trainer, left[3]),
        None => frame.render_widget(
            Paragraph::new(
                "no live trainer — the watchdog submits the next one when labels accumulate",
            )
            .style(Style::default().fg(Color::Yellow))
            .block(Block::default().title(" LIVE RUN ").borders(Borders::ALL)),
            left[3],
        ),
    }
    render_dl_global(frame, dashboard, columns[1]);
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[2]);
}

/// Wrap the raw key-press counter onto the live arm count. Kept separate so the
/// tests can pin the wrap behaviour when arms appear or exit mid-run.
fn selected_arm(arm: usize, count: usize) -> usize {
    if count == 0 {
        0
    } else {
        arm % count
    }
}

/// One line per arm: the at-a-glance comparison across every GPU. Shown only
/// when more than one arm is live.
fn render_arm_strip(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    arms: &[&TrainerDetail],
    arm: usize,
    area: Rect,
) {
    let selected = selected_arm(arm, arms.len());
    let best = arms
        .iter()
        .filter_map(|t| arm_latest(t).and_then(|p| p.val_conditional_advantage))
        .fold(f64::NEG_INFINITY, f64::max);
    let lines = arms
        .iter()
        .enumerate()
        .map(|(index, trainer)| {
            let marker = if index == selected { "▶" } else { " " };
            let node = gpu_for_job(dashboard, &trainer.job_id)
                .map(|gpu| format!("{} {}", trainer.node, gpu.model_hint))
                .unwrap_or_else(|| trainer.node.clone());
            let Some(point) = arm_latest(trainer) else {
                return Line::from(vec![
                    Span::raw(format!(
                        "{marker} {:<22} {:<14} ",
                        short(&trainer.job_id, 22),
                        node
                    )),
                    Span::styled(
                        "warming up — no metrics yet",
                        Style::default().fg(Color::DarkGray),
                    ),
                ]);
            };
            let advantage = point.val_conditional_advantage;
            let advantage_style = match advantage {
                Some(value) if value >= best && best > f64::NEG_INFINITY => Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
                Some(value) if value > 0.0 => Style::default().fg(Color::Green),
                Some(_) => Style::default().fg(Color::Red),
                None => Style::default().fg(Color::DarkGray),
            };
            Line::from(vec![
                Span::raw(format!(
                    "{marker} {:<22} {:<14} ep {:>3}/{:<3} train {:>6.3} val {:>7.3} pw {:>5.3} ",
                    short(&trainer.job_id, 22),
                    node,
                    trainer.epoch,
                    trainer.epochs,
                    point.train_loss,
                    point.val_loss,
                    point.val_pairwise.unwrap_or(f64::NAN),
                )),
                Span::styled(
                    match advantage {
                        Some(value) => format!("cond {value:+.4}"),
                        None => "cond    n/a".to_owned(),
                    },
                    advantage_style,
                ),
                Span::raw(format!(
                    "  stop {}/{}",
                    point.bad_passes.max(trainer.bad_passes),
                    trainer.patience
                )),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(format!(
                    " TRAINING ARMS ({}) — n/←/→ to select ",
                    arms.len()
                ))
                .borders(Borders::ALL),
        ),
        area,
    );
}

/// Cross-generation loss chart. Takes the flattened history rather than a live
/// trainer so it renders whether or not a job is currently running.
fn render_loss_chart(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    series: &DlSeries,
    trainer: Option<&TrainerDetail>,
    area: Rect,
) {
    let (train, val) = (&series.train, &series.val);
    let all = train.iter().chain(val.iter()).map(|(_, y)| *y);
    let y_max = all.clone().fold(f64::MIN, f64::max);
    let y_min = all.fold(f64::MAX, f64::min);
    let (y_min, y_max) = if y_min > y_max {
        (0.0, 1.0)
    } else {
        ((y_min * 0.9).max(0.0), (y_max * 1.1).max(y_min + 0.1))
    };
    let x_min = train
        .first()
        .or(val.first())
        .map(|(x, _)| *x)
        .unwrap_or(0.0);
    let x_max = train
        .last()
        .or(val.last())
        .map(|(x, _)| *x)
        .unwrap_or(1.0)
        .max(x_min + 1.0);
    // Unnamed datasets stay out of the legend, so the boundaries read as plain
    // rules between generations.
    let boundaries = series
        .marks
        .iter()
        .skip(1)
        .map(|(x, _, _)| vec![(*x - 0.5, y_min), (*x - 0.5, y_max)])
        .collect::<Vec<_>>();
    let mut datasets = boundaries
        .iter()
        .map(|line| {
            Dataset::default()
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(line)
        })
        .collect::<Vec<_>>();
    datasets.push(
        Dataset::default()
            .name(format!(
                "train {}",
                decimal(
                    trainer
                        .and_then(|trainer| trainer.running_loss)
                        .or_else(|| train.last().map(|(_, y)| *y))
                )
            ))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(train),
    );
    datasets.push(
        Dataset::default()
            .name(format!("val {}", decimal(val.last().map(|(_, y)| *y))))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Red))
            .data(val),
    );
    let arms = training_arms(dashboard);
    let live_label = match trainer {
        Some(trainer) => {
            let arm_label = arms
                .iter()
                .position(|candidate| candidate.job_id == trainer.job_id)
                .filter(|_| arms.len() > 1)
                .map(|index| format!(" [arm {}/{}]", index + 1, arms.len()))
                .unwrap_or_default();
            let gpu_label = gpu_for_job(dashboard, &trainer.job_id)
                .map(|gpu| format!(" {}", gpu.model_hint))
                .unwrap_or_default();
            format!(
                "live {} @ {}{}{}{}",
                short(&trainer.job_id, 34),
                trainer.node,
                gpu_label,
                arm_label,
                stale_suffix(dashboard.collected_at_ms, trainer.progress_at_ms)
            )
        }
        None => "no live trainer".to_owned(),
    };
    let title = format!(
        " LOSS — {} gen(s) · {:.0} epochs · {}{} ",
        series.marks.len(),
        series.total_epochs,
        live_label,
        stale_suffix(
            dashboard.collected_at_ms,
            dashboard.dl_history.fetched_at_ms
        )
    );
    frame.render_widget(
        Chart::new(datasets)
            .block(Block::default().title(title).borders(Borders::ALL))
            .x_axis(
                Axis::default()
                    .title("epoch")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([x_min, x_max])
                    .labels(vec![
                        format!("{x_min:.0}"),
                        format!("{:.0}", (x_min + x_max) / 2.0),
                        format!("{x_max:.0}"),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        format!("{y_min:.2}"),
                        format!("{:.2}", (y_min + y_max) / 2.0),
                        format!("{y_max:.2}"),
                    ]),
            ),
        area,
    );
}

/// Generation ruler drawn directly under the loss chart: one tick per trainer
/// generation, positioned proportionally, so the boundaries are readable rather
/// than merely visible. The live generation is highlighted.
fn generation_ruler(series: &DlSeries, width: u16) -> Line<'static> {
    if series.marks.is_empty() || width < 8 {
        return Line::from(Span::styled(
            "no trainer generations on disk yet",
            Style::default().fg(Color::DarkGray),
        ));
    }
    // Match the chart's plotting area: 1 col of left border + 1 of right.
    let inner = width.saturating_sub(2) as usize;
    let span_epochs = series.total_epochs.max(1.0);
    let mut spans = vec![Span::raw(" ")];
    let mut cursor = 0usize;
    for (x, label, live) in &series.marks {
        let column = (((*x - 1.0) / span_epochs) * inner as f64).round() as usize;
        let text = format!("├{label}");
        if column + text.chars().count() > inner {
            continue;
        }
        if column < cursor {
            continue;
        }
        spans.push(Span::raw(" ".repeat(column - cursor)));
        let style = if *live {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        spans.push(Span::styled(text.clone(), style));
        cursor = column + text.chars().count();
    }
    Line::from(spans)
}

/// Cross-generation pairwise-win chart (the "should climb" curve).
fn render_pairwise_chart(frame: &mut Frame<'_>, series: &DlSeries, area: Rect) {
    let points = series.pairwise.clone();
    let ablation = series.ablation.clone();
    let x_min = points.first().map(|(x, _)| *x).unwrap_or(0.0);
    let x_max = points
        .last()
        .map(|(x, _)| *x)
        .unwrap_or(1.0)
        .max(x_min + 1.0);
    let y_low = points
        .iter()
        .chain(ablation.iter())
        .map(|(_, y)| *y)
        .fold(f64::MAX, f64::min)
        .min(0.48);
    let y_high = points
        .iter()
        .chain(ablation.iter())
        .map(|(_, y)| *y)
        .fold(f64::MIN, f64::max)
        .max(0.55);
    let (y_low, y_high) = if y_low > y_high {
        (0.4, 0.8)
    } else {
        ((y_low - 0.02).max(0.0), (y_high + 0.02).min(1.0))
    };
    let chance = vec![(x_min, 0.5), (x_max, 0.5)];
    let latest = points.last().map(|(_, y)| *y);
    let boundaries = series
        .marks
        .iter()
        .skip(1)
        .map(|(x, _, _)| vec![(*x - 0.5, y_low), (*x - 0.5, y_high)])
        .collect::<Vec<_>>();
    let mut datasets = boundaries
        .iter()
        .map(|line| {
            Dataset::default()
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(line)
        })
        .collect::<Vec<_>>();
    datasets.push(
        Dataset::default()
            .name("chance 0.5")
            .marker(symbols::Marker::Dot)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::DarkGray))
            .data(&chance),
    );
    datasets.push(
        Dataset::default()
            .name(format!(
                "ablation(control) {}",
                decimal(ablation.last().map(|(_, y)| *y))
            ))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Magenta))
            .data(&ablation),
    );
    datasets.push(
        Dataset::default()
            .name(format!("pairwise {}", decimal(latest)))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&points),
    );
    frame.render_widget(
        Chart::new(datasets)
            .block(
                Block::default()
                    .title(" VAL PAIRWISE-WIN — all generations (should climb) ")
                    .borders(Borders::ALL),
            )
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([x_min, x_max]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([y_low, y_high])
                    .labels(vec![
                        format!("{y_low:.2}"),
                        "0.50".to_owned(),
                        format!("{y_high:.2}"),
                    ]),
            ),
        area,
    );
}

fn render_dl_status(frame: &mut Frame<'_>, trainer: &TrainerDetail, area: Rect) {
    let history = &trainer.history;
    let gap_trend = if history.len() >= 2 {
        trend_arrow(
            history[history.len() - 2].overfit_gap,
            history[history.len() - 1].overfit_gap,
        )
    } else {
        "-"
    };
    let validating = trainer.phase.starts_with("validation");
    let epoch_minutes = if trainer.epoch > 0 {
        trainer.elapsed_sec / 60.0 / trainer.epoch as f64
    } else {
        0.0
    };
    let early = if trainer.patience > 0 && trainer.bad_passes >= trainer.patience {
        Span::styled(
            " EARLY-STOPPED ",
            Style::default().fg(Color::Black).bg(Color::Red),
        )
    } else {
        Span::styled(
            format!(
                " early-stop {}/{} ",
                trainer.bad_passes,
                trainer.patience.max(1)
            ),
            if trainer.patience > 0 && trainer.bad_passes * 4 >= trainer.patience * 3 {
                Style::default().fg(Color::Black).bg(Color::Yellow)
            } else {
                Style::default().fg(Color::Green)
            },
        )
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![Span::styled(
                phase_label(trainer),
                Style::default().fg(if validating { Color::Magenta } else { Color::Cyan }),
            )]),
            Line::from(vec![
                early,
                Span::raw(format!(
                    "  overfit-gap {} {}  •  ~{epoch_minutes:.0} min/epoch  •  elapsed {:.0}m",
                    decimal(trainer.overfit_gap),
                    gap_trend,
                    trainer.elapsed_sec / 60.0,
                )),
            ]),
            // Conditional advantage is the headline: at or below zero the model
            // is not beating the genome-only control, so a healthy-looking
            // pairwise number means nothing.
            match arm_latest(trainer).map(|point| point.val_conditional_advantage) {
                Some(Some(value)) => Line::from(vec![
                    Span::raw("conditional advantage "),
                    Span::styled(
                        format!("{value:+.4}"),
                        if value > 0.0 {
                            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
                        },
                    ),
                    Span::styled(
                        if value > 0.0 {
                            "  (beating the genome-only control)"
                        } else {
                            "  (NOT using the dataset embedding)"
                        },
                        Style::default().fg(Color::DarkGray),
                    ),
                ]),
                Some(None) => Line::from(Span::styled(
                    "conditional advantage n/a — metric absent from this run",
                    Style::default().fg(Color::DarkGray),
                )),
                None => Line::from(Span::styled(
                    "conditional advantage — warming up, no metrics yet",
                    Style::default().fg(Color::DarkGray),
                )),
            },
            Line::from(format!(
                "watermark {}  •  progress heartbeat is LIVE during validation — a full sweep takes minutes",
                human_count(trainer.label_watermark)
            )),
        ])
        .block(Block::default().title(" TRAINER STATE ").borders(Borders::ALL)),
        area,
    );
}

fn render_dl_global(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    // Two lines per task (numbers + stacked bar) plus borders. Hard-coding 4
    // truncated the BINARY row off the bottom, so the panel only ever showed
    // regression — both task totals have to be visible.
    let funnel_height = (dashboard.funnel.rows.len().max(1) * 2 + 2).min(10) as u16;
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(7),
            Constraint::Length(3),
            Constraint::Length(funnel_height),
            Constraint::Length(9),
            Constraint::Length(3),
            Constraint::Length(GENERATIONS_MAX as u16 + SEEDING_CURVE_ROWS as u16 + 3),
        ])
        .split(area);
    let alert_items = if dashboard.alerts.is_empty() {
        vec![ListItem::new(Span::styled(
            "all clear",
            Style::default().fg(Color::Green),
        ))]
    } else {
        dashboard
            .alerts
            .iter()
            .map(|alert| {
                let color = if alert.contains("critical") || alert.contains("CRITICAL") {
                    Color::Red
                } else {
                    Color::Yellow
                };
                ListItem::new(Span::styled(alert.clone(), Style::default().fg(color)))
            })
            .collect()
    };
    frame.render_widget(
        List::new(alert_items).block(
            Block::default()
                .title(" GLOBAL ERRORS / ALARMS ")
                .borders(Borders::ALL),
        ),
        parts[0],
    );
    let heartbeat_lines = dashboard
        .daemon_ages
        .iter()
        .map(|(name, age, expected)| {
            let (text, color) = if *age == u64::MAX {
                ("MISSING".to_owned(), Color::Red)
            } else if *age > expected * 2 {
                (format!("{age}s (expect ~{expected}s)"), Color::Red)
            } else if *age > *expected {
                (format!("{age}s (expect ~{expected}s)"), Color::Yellow)
            } else {
                (format!("{age}s"), Color::Green)
            };
            Line::from(vec![
                Span::raw(format!("{name:<24} ")),
                Span::styled(text, Style::default().fg(color)),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(heartbeat_lines).block(
            Block::default()
                .title(" DAEMON HEARTBEATS ")
                .borders(Borders::ALL),
        ),
        parts[1],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "failed {}  •  {} jobs/min  •  cluster CPU {:.0}% (target {:.0}%)",
            human_count(dashboard.cluster.jobs_failed),
            dashboard.cluster.throughput_per_minute,
            dashboard.cluster.cpu_busy_pct,
            CPU_TARGET_PCT
        ))
        .block(Block::default().title(" SCQ ").borders(Borders::ALL)),
        parts[2],
    );
    let funnel_lines = if dashboard.funnel.rows.is_empty() {
        vec![Line::from("collecting…")]
    } else {
        let bar_width = (parts[3].width.saturating_sub(4) as usize).clamp(10, 60);
        dashboard
            .funnel
            .rows
            .iter()
            .flat_map(|row| [funnel_line(row), funnel_bar(row, bar_width)])
            .collect()
    };
    frame.render_widget(
        Paragraph::new(funnel_lines).block(
            Block::default()
                .title(format!(
                    " DATASET UNLOCK ({}){} ",
                    dashboard.funnel.source,
                    stale_suffix(now_ms(), dashboard.funnel.fetched_at_ms)
                ))
                .borders(Borders::ALL),
        ),
        parts[3],
    );
    let trainer = dashboard
        .trainers
        .iter()
        .find(|trainer| trainer.is_trainer)
        .or_else(|| dashboard.trainers.first());
    let metric_lines = trainer
        .map(|trainer| {
            let history = &trainer.history;
            let last = history.last();
            let prior = history
                .len()
                .checked_sub(2)
                .and_then(|index| history.get(index));
            let row = |label: &str, get: &dyn Fn(&TrainerMetricPoint) -> Option<f64>| {
                let latest = last.and_then(get);
                let arrow = match (prior.and_then(get), latest) {
                    (Some(previous), Some(current)) => trend_arrow(previous, current),
                    _ => "-",
                };
                Line::from(format!("{label:<22} {}  {arrow}", decimal(latest)))
            };
            vec![
                row("val_pairwise_win", &|point| point.val_pairwise),
                row("val_regression_win", &|point| point.val_regression_pairwise),
                row("val_binary_win", &|point| point.val_binary_pairwise),
                row("reward_cv_corr", &|point| point.val_reward_cv),
                row("top1_regret (↓ good)", &|point| point.val_top1_regret),
                row("overfit_gap (↓ good)", &|point| Some(point.overfit_gap)),
                row("val_loss (↓ good)", &|point| Some(point.val_loss)),
            ]
        })
        .unwrap_or_else(|| vec![Line::from("no trainer history yet")]);
    frame.render_widget(
        Paragraph::new(metric_lines).block(
            Block::default()
                .title(" VAL METRICS (vs prior epoch) ")
                .borders(Borders::ALL),
        ),
        parts[4],
    );
    let labels_line = dashboard
        .watchdog
        .as_ref()
        .map(|watchdog| {
            format!(
                "labels {}  •  pending {}  •  next trainer at {}",
                human_count(watchdog.labels),
                human_count(watchdog.pending_unfrozen_labels),
                human_count(watchdog.next_label_threshold)
            )
        })
        .unwrap_or_else(|| "watchdog status unavailable".to_owned());
    frame.render_widget(
        Paragraph::new(labels_line).block(
            Block::default()
                .title(" LABEL SUPPLY ")
                .borders(Borders::ALL),
        ),
        parts[5],
    );
    let percent = |value: Option<f64>| {
        value
            .map(|share| format!("{share:.1}%"))
            .unwrap_or_else(|| "-".to_owned())
    };
    let mut generation_lines = vec![Line::from(Span::styled(
        format!(
            "{:<30}  {:>7}  {:>7}  {:>7}",
            "run / seed checkpoint", "val-pw", "DL-win", "DL-obs"
        ),
        Style::default().fg(Color::DarkGray),
    ))];
    if dashboard.generations.is_empty() {
        generation_lines.push(Line::from("collecting…"));
    } else {
        generation_lines.extend(dashboard.generations.iter().map(|(label, pairwise)| {
            Line::from(format!(
                "{:<30}  {:>7}  {:>7}  {:>7}",
                short(label, 30),
                decimal(*pairwise),
                "-",
                "-"
            ))
        }));
    }
    // DL-share column source: the newest seeding-curve checkpoint generations.
    let skip = dashboard
        .seeding
        .points
        .len()
        .saturating_sub(SEEDING_CURVE_ROWS);
    for point in dashboard.seeding.points.iter().skip(skip) {
        let label = format!(
            "ckpt {} ({} wins)",
            point.checkpoint_short, point.strict_wins_total
        );
        let observed = if point.observations_caught_up {
            percent(point.dl_obs_share_pct)
        } else {
            format!("~{}", percent(point.dl_obs_share_pct))
        };
        generation_lines.push(Line::from(vec![
            Span::raw(format!("{:<30}  {:>7}  ", short(&label, 30), "-")),
            Span::styled(
                format!("{:>7}", percent(point.dl_win_share_pct)),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw(format!("  {observed:>7}")),
        ]));
    }
    frame.render_widget(
        Paragraph::new(generation_lines)
            .block(
                Block::default()
                    .title(format!(
                        " GENERATIONS (val-pw • DL-share, old→new){} ",
                        stale_suffix(now_ms(), dashboard.seeding.fetched_at_ms)
                    ))
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true }),
        parts[6],
    );
}

fn render_header(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let health = if dashboard.alerts.is_empty() {
        Span::styled(
            " HEALTHY ",
            Style::default().fg(Color::Black).bg(Color::Green),
        )
    } else {
        Span::styled(
            format!(" {} ALERTS ", dashboard.alerts.len()),
            Style::default().fg(Color::Black).bg(Color::Yellow),
        )
    };
    let title = Line::from(vec![
        Span::styled(
            " JOPE REAL-TIME SELF-PLAY ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        health,
        Span::raw(format!(
            "  SCQ epoch {}  sources {}/{}  collected {}",
            dashboard
                .scq_epoch
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            dashboard.sources_healthy,
            dashboard.sources_total,
            clock_string(dashboard.collected_at_ms)
        )),
    ]);
    frame.render_widget(
        Paragraph::new(title).block(Block::default().borders(Borders::ALL)),
        area,
    );
}

fn render_global(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(38),
            Constraint::Percentage(31),
            Constraint::Percentage(31),
        ])
        .split(area);
    let cpu = dashboard.cluster.cpu_busy_pct.clamp(0.0, 100.0);
    frame.render_widget(
        Gauge::default()
            .block(
                Block::default()
                    .title(" Global CPU • target 80% / guard 95% ")
                    .borders(Borders::ALL),
            )
            .gauge_style(Style::default().fg(cpu_color(cpu)))
            .ratio(cpu / 100.0)
            .label(format!("{cpu:.1}%")),
        columns[0],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "PSI cpu {:.2}  mem {:.2}  io {:.2}\nthroughput {}/min  failed {}",
            dashboard.cluster.psi_cpu10,
            dashboard.cluster.psi_mem10,
            dashboard.cluster.psi_io10,
            dashboard.cluster.throughput_per_minute,
            dashboard.cluster.jobs_failed
        ))
        .block(
            Block::default()
                .title(" Pressure / failures ")
                .borders(Borders::ALL),
        ),
        columns[1],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "nodes {}/{}  cores {}/{}\nslots {}/{}  RAM avail {} GiB",
            dashboard.cluster.nodes_up,
            dashboard.cluster.nodes_total,
            dashboard.cluster.effective_cores,
            dashboard.cluster.physical_cores,
            dashboard.cluster.running_slots,
            dashboard.cluster.slot_target,
            dashboard.cluster.memory_available_gb
        ))
        .block(Block::default().title(" Capacity ").borders(Borders::ALL)),
        columns[2],
    );
}

fn render_nodes(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let header = Row::new([
        "node",
        "state",
        "cores",
        "mult",
        "slots",
        "CPU",
        "PSI c/m/i",
        "RAM",
        "disk",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let rows = dashboard.nodes.iter().map(|node| {
        let style = match node.state.as_str() {
            "up" => Style::default().fg(cpu_color(node.cpu_busy_pct.unwrap_or(0.0))),
            "down" => Style::default().fg(Color::Red),
            _ => Style::default().fg(Color::DarkGray),
        };
        Row::new(vec![
            Cell::from(node.node.clone()),
            Cell::from(format!("{}/{}", node.state, short(&node.health, 8))),
            Cell::from(format!("{}/{}", node.effective_cores, node.physical_cores)),
            Cell::from(format!("{:.1}x", node.multiplier)),
            Cell::from(format!("{}/{}", node.running_slots, node.slot_target)),
            Cell::from(
                node.cpu_busy_pct
                    .map(|value| format!("{value:.1}%"))
                    .unwrap_or_else(|| "-".to_owned()),
            ),
            Cell::from(format!(
                "{}/{}/{}",
                decimal(node.psi_cpu10),
                decimal(node.psi_mem10),
                decimal(node.psi_io10)
            )),
            Cell::from(format!(
                "{}/{}G",
                node.memory_available_gb, node.memory_total_gb
            )),
            Cell::from(
                node.disk_use_pct
                    .map(|value| format!("{value:.0}%"))
                    .unwrap_or_else(|| "-".to_owned()),
            ),
        ])
        .style(style)
    });
    frame.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(9),
                Constraint::Length(15),
                Constraint::Length(9),
                Constraint::Length(5),
                Constraint::Length(10),
                Constraint::Length(7),
                Constraint::Length(14),
                Constraint::Length(11),
                Constraint::Length(6),
            ],
        )
        .header(header)
        .block(Block::default().title(" SCQ nodes ").borders(Borders::ALL)),
        area,
    );
}

fn render_queue(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(4)])
        .split(area);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "fresh {}/{} cap {}  age {:.0}s  mode {}",
                dashboard.queue.useful_depth,
                dashboard.queue.target,
                dashboard.queue.hard_cap,
                dashboard.queue.maximum_age_seconds,
                dashboard.queue.mode
            )),
            Line::from(format!(
                "Q/S/D/R/H {}/{}/{}/{}/{}  epoch {}",
                dashboard.queue.queued,
                dashboard.queue.starting,
                dashboard.queue.dispatching,
                dashboard.queue.running,
                dashboard.queue.held,
                short_epoch(&dashboard.queue.epoch)
            )),
        ])
        .block(
            Block::default()
                .title(" Queue horizon / epoch ")
                .borders(Borders::ALL),
        ),
        parts[0],
    );
    let header = Row::new(["family", "prio", "Q", "S", "D", "R", "H", "oldest"])
        .style(Style::default().fg(Color::Cyan));
    let rows = dashboard.families.iter().take(8).map(|family| {
        let style = if family.weight >= 10.0 {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else if family.family.starts_with("jope-selfplay-eval") {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };
        Row::new(vec![
            Cell::from(short(&family.family, 21)),
            Cell::from(format!("{:.1}", family.weight)),
            Cell::from(family.queued.to_string()),
            Cell::from(family.starting.to_string()),
            Cell::from(family.dispatching.to_string()),
            Cell::from(family.running.to_string()),
            Cell::from(family.held.to_string()),
            Cell::from(
                family
                    .oldest_queue_ts
                    .as_deref()
                    .map(|value| short(value, 12))
                    .unwrap_or_else(|| "-".to_owned()),
            ),
        ])
        .style(style)
    });
    frame.render_widget(
        Table::new(
            rows,
            [
                Constraint::Min(16),
                Constraint::Length(5),
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(12),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(" Priority / active states ")
                .borders(Borders::ALL),
        ),
        parts[1],
    );
}

fn render_learning(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let learning = &dashboard.learning;
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!("checkpoint {}", short_hash(&learning.checkpoint))),
            Line::from(format!(
                "trainer {} [{}]",
                short(&learning.trainer, 38),
                learning.trainer_state
            )),
            Line::from(format!(
                "labels {}  frozen {}  rejects {}",
                human_count(learning.labels),
                human_count(learning.frozen_watermark),
                learning.consecutive_rejections
            )),
            Line::from(format!(
                "promotion internal={} milestone={}",
                learning.internal_promotion, learning.milestone_promotion
            )),
            Line::from(format!(
                "DL proposals R {} / B {}  validity {}",
                human_count(learning.proposal_regression),
                human_count(learning.proposal_binary),
                learning
                    .proposal_validity
                    .map(|value| format!("{:.1}%", value * 100.0))
                    .unwrap_or_else(|| "-".to_owned())
            )),
        ])
        .block(
            Block::default()
                .title(" Active checkpoint / DL trainer / promotion ")
                .borders(Borders::ALL),
        ),
        area,
    );
}

/// The per-wave cohort is a rotating sample, not the population — the label
/// must make that impossible to misread.
fn cohort_label(funnel_active_total: u64, wave: &WaveSummary) -> String {
    let population = if funnel_active_total > 0 {
        funnel_active_total
    } else {
        wave.adaptive_regression + wave.adaptive_binary
    };
    format!(
        "gameplay cohort: 128 datasets/wave (64R/64B, 256 genomes) rotating over {} active",
        thousands(population)
    )
}

fn render_wave(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let wave = &dashboard.wave;
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "wave {}  admissions {}  unclaimed {}",
                wave.active_wave, wave.admission_waves, wave.unclaimed_manifests
            )),
            Line::from(format!(
                "supply ready {}  keeper ack {}",
                wave.ready_manifests, wave.awaiting_keeper_ack
            )),
            Line::from(cohort_label(
                dashboard.funnel.rows.iter().map(|row| row.active).sum(),
                &dashboard.wave,
            )),
            Line::from(format!(
                "adaptive catalog R {} / B {}  corpus labels {}",
                wave.adaptive_regression,
                wave.adaptive_binary,
                human_count(wave.corpus_rows)
            )),
            Line::from(format!(
                "sealed isolated={} scored={}  pairs={} dedup={}",
                yes_no(wave.sealed_isolated),
                wave.sealed_scored,
                human_count(wave.pair_count),
                human_count(wave.duplicate_pairs_prevented)
            )),
            Line::from(format!(
                "strict wins {}  gene evidence {}",
                wave.strict_wins,
                human_bytes(wave.evidence_bytes)
            )),
        ])
        .block(
            Block::default()
                .title(" Micro-waves / adaptive gameplay / sealed lane ")
                .borders(Borders::ALL),
        ),
        area,
    );
}

fn render_alerts(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let mut lines = dashboard
        .alerts
        .iter()
        .take(5)
        .map(|alert| ListItem::new(short(alert, 80)).style(Style::default().fg(Color::Yellow)))
        .collect::<Vec<_>>();
    if lines.is_empty() {
        lines.push(ListItem::new("no active alerts").style(Style::default().fg(Color::Green)));
    }
    for failure in dashboard.failures.iter().take(2) {
        lines.push(ListItem::new(failure.clone()).style(Style::default().fg(Color::Red)));
    }
    frame.render_widget(
        List::new(lines).block(
            Block::default()
                .title(format!(
                    " Alerts / recent failures ({}) ",
                    dashboard.cluster.jobs_failed
                ))
                .borders(Borders::ALL),
        ),
        area,
    );
}

fn render_events(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let lines = dashboard
        .events
        .iter()
        .take(area.height.saturating_sub(2) as usize)
        .map(|event| ListItem::new(short(event, 110)))
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(lines).block(Block::default().title(" Event tail ").borders(Borders::ALL)),
        area,
    );
}

fn render_compact_gpu_dl(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
) {
    let now = dashboard.collected_at_ms;
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "DEEP LEARNING • LIVE",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            "  sources {}/{}  {}",
            dashboard.sources_healthy,
            dashboard.sources_total,
            clock_string(now)
        )),
    ])];

    if dashboard.accepted_tasks.is_empty() {
        lines.push(Line::from("fronts: accepted-by-task unavailable"));
    } else {
        lines.extend(dashboard.accepted_tasks.iter().map(|front| {
            Line::from(vec![
                Span::styled(
                    format!("{:<10}", front.task),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "accepted {}  generation {}  age {}s",
                    short_hash(&front.checkpoint_sha256),
                    front.generation,
                    now.saturating_sub(front.updated_at_ms) / 1_000
                )),
            ])
        }));
    }

    lines.push(
        dashboard
            .watchdog
            .as_ref()
            .map(|watchdog| {
                Line::from(format!(
                    "watchdog {} • corpus {} • labels {} • gap {} • ckpt {}",
                    watchdog.action,
                    human_count(watchdog.snapshot_watermark),
                    human_count(watchdog.labels),
                    human_count(watchdog.pending_unfrozen_labels),
                    short_hash(&watchdog.checkpoint_sha256)
                ))
            })
            .unwrap_or_else(|| Line::from("watchdog unavailable")),
    );

    lines.push(Line::from(Span::styled(
        "GPUs",
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    )));
    if dashboard.gpus.is_empty() {
        lines.push(Line::from("  no GPU telemetry"));
    } else {
        lines.extend(dashboard.gpus.iter().map(|gpu| {
            let active = gpu.owner_job.is_some();
            Line::from(vec![
                Span::styled(
                    format!(
                        "  {} {:>3.0}% {:>5}/{:<5} MiB {:>3.0}W ",
                        gpu.node, gpu.util_pct, gpu.mem_used_mb, gpu.mem_total_mb, gpu.power_w
                    ),
                    Style::default().fg(if active {
                        Color::Green
                    } else {
                        Color::DarkGray
                    }),
                ),
                Span::raw(short(
                    gpu.owner_job.as_deref().unwrap_or(&gpu.status),
                    area.width.saturating_sub(42) as usize,
                )),
            ])
        }));
    }

    lines.push(Line::from(Span::styled(
        "ACTIVE GPU WORK",
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    )));
    if dashboard.trainers.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default().fg(Color::Red),
        )));
    } else {
        for trainer in &dashboard.trainers {
            let heartbeat_age = now.saturating_sub(trainer.progress_at_ms) / 1_000;
            let role = if trainer.is_trainer { "TRAIN" } else { "GPU" };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {role:<5} "),
                    Style::default()
                        .fg(if trainer.is_trainer {
                            Color::Green
                        } else {
                            Color::Cyan
                        })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "{} @ {}",
                    short(&trainer.job_id, area.width.saturating_sub(22) as usize),
                    trainer.node
                )),
            ]));
            if trainer.is_trainer {
                lines.push(Line::from(format!(
                    "        {} epoch {}/{} batch {}/{} ({:.1}%) loss {} hb {}s",
                    trainer.phase,
                    trainer.epoch,
                    trainer.epochs,
                    trainer.batch,
                    trainer.total_batches,
                    trainer.pct,
                    trainer
                        .running_loss
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "-".to_owned()),
                    heartbeat_age,
                )));
            }
        }
    }

    lines.push(Line::from(format!(
        "1 overview • 2 gpu/dl • 4 charts • q quit • {}ms refresh / {}ms collect",
        refresh_ms, dashboard.collection_latency_ms
    )));
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default()
                .title(" DL CONTROL PLANE ")
                .borders(Borders::ALL),
        ),
        area,
    );
}

fn render_compact(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, area: Rect) {
    let nodes = dashboard
        .nodes
        .iter()
        .map(|node| {
            format!(
                "{}:{} CPU {} slots {}/{}",
                node.node,
                node.state,
                node.cpu_busy_pct
                    .map(|value| format!("{value:.0}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                node.running_slots,
                node.slot_target
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    frame.render_widget(
        Paragraph::new(vec![
            Line::from("JOPE SELF-PLAY • read-only Rust TUI"),
            Line::from(format!(
                "CPU {:.1}% (target 80 / guard 95) PSI {}/{}/{}",
                dashboard.cluster.cpu_busy_pct,
                dashboard.cluster.psi_cpu10,
                dashboard.cluster.psi_mem10,
                dashboard.cluster.psi_io10
            )),
            Line::from(nodes),
            Line::from(format!(
                "queue fresh {}/{} cap {} age {:.0}s epoch {}",
                dashboard.queue.useful_depth,
                dashboard.queue.target,
                dashboard.queue.hard_cap,
                dashboard.queue.maximum_age_seconds,
                short_epoch(&dashboard.queue.epoch)
            )),
            Line::from(format!(
                "checkpoint {} trainer {} [{}] labels {}",
                short_hash(&dashboard.learning.checkpoint),
                short(&dashboard.learning.trainer, 30),
                dashboard.learning.trainer_state,
                human_count(dashboard.learning.labels)
            )),
            Line::from(format!(
                "wave {} adaptive R{}/B{} sealed {}",
                dashboard.wave.active_wave,
                dashboard.wave.adaptive_regression,
                dashboard.wave.adaptive_binary,
                yes_no(dashboard.wave.sealed_isolated)
            )),
            Line::from(format!(
                "alerts: {}",
                if dashboard.alerts.is_empty() {
                    "none".to_owned()
                } else {
                    dashboard.alerts.join(" | ")
                }
            )),
            Line::from(format!(
                "1 overview • 2 gpu/dl • 4 charts • q/esc quit • {}ms refresh • {}ms collect",
                refresh_ms, dashboard.collection_latency_ms
            )),
        ])
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL)),
        area,
    );
}

fn format_event(value: &Value) -> String {
    let payload = value.get("result").unwrap_or(value);
    format!(
        "{} {:<10} {:<25} {:<8} {}",
        short(
            string_at(payload, &["ts"])
                .or_else(|| string_at(payload, &["generated_at"]))
                .unwrap_or("-"),
            19
        ),
        string_at(payload, &["kind"])
            .or_else(|| string_at(payload, &["schema"]))
            .unwrap_or("event"),
        short(
            string_at(payload, &["job"])
                .or_else(|| string_at(payload, &["job_id"]))
                .unwrap_or("-"),
            25
        ),
        string_at(payload, &["node"]).unwrap_or("-"),
        short(
            string_at(payload, &["msg"])
                .or_else(|| string_at(payload, &["error"]))
                .unwrap_or(""),
            55
        )
    )
}

fn promotion_label(value: Option<&Value>) -> String {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return "none".to_owned();
    };
    string_at(value, &["decision"])
        .or_else(|| string_at(value, &["status"]))
        .or_else(|| string_at(value, &["checkpoint_sha256"]))
        .map(|value| short(value, 12))
        .unwrap_or_else(|| "present".to_owned())
}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(Value::as_str)
}

fn u64_at(value: &Value, path: &[&str]) -> Option<u64> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_i64().and_then(|number| u64::try_from(number).ok()))
                .or_else(|| value.as_f64().map(|number| number.max(0.0) as u64))
        })
}

fn f64_at(value: &Value, path: &[&str]) -> Option<f64> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_u64().map(|number| number as f64))
        })
}

fn bool_at(value: &Value, path: &[&str]) -> Option<bool> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(Value::as_bool)
}

fn active_jobs(row: &FamilyRow) -> u64 {
    row.queued + row.starting + row.dispatching + row.running + row.held
}

fn maximum(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, f64::max)
}

fn decimal(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".to_owned())
}

fn short(value: &str, width: usize) -> String {
    if value.chars().count() <= width {
        return value.to_owned();
    }
    value
        .chars()
        .take(width.saturating_sub(1))
        .collect::<String>()
        + "…"
}

fn short_hash(value: &str) -> String {
    if value == "-" {
        "-".to_owned()
    } else {
        short(value, 12)
    }
}

fn short_epoch(value: &str) -> String {
    value
        .strip_prefix("checkpoint-")
        .map(short_hash)
        .unwrap_or_else(|| short(value, 18))
}

fn cpu_color(cpu: f64) -> Color {
    if cpu >= CPU_HARD_GUARD_PCT {
        Color::Red
    } else if cpu >= CPU_TARGET_PCT {
        Color::Yellow
    } else {
        Color::Green
    }
}

fn yes_no(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "NO",
        None => "-",
    }
}

/// Exact count with thousands separators (the unlock pipeline must show the
/// real climbing numbers, not abbreviations).
fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn human_count(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.2}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn human_bytes(value: u64) -> String {
    if value >= 1024 * 1024 * 1024 {
        format!("{:.1}GiB", value as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if value >= 1024 * 1024 {
        format!("{:.1}MiB", value as f64 / (1024.0 * 1024.0))
    } else if value >= 1024 {
        format!("{:.1}KiB", value as f64 / 1024.0)
    } else {
        format!("{value}B")
    }
}

fn clock_string(milliseconds: u128) -> String {
    let seconds = (milliseconds / 1_000) % 86_400;
    format!(
        "{:02}:{:02}:{:02}Z",
        seconds / 3_600,
        (seconds % 3_600) / 60,
        seconds % 60
    )
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> RawSources {
        RawSources {
            scq: Some(json!({
                "Snapshot": {
                    "epoch": 77,
                    "ts": "2026-07-27T00:00:00Z",
                    "nodes": [
                        {
                            "node": "xbabe1", "up": true, "health": "healthy",
                            "cores": 4, "effective_cores": 3,
                            "cpu_job_slot_multiplier": 1.0,
                            "job_slot_target": 3, "running_job_slots": 2,
                            "cpu_busy_pct": 50.0, "psi_cpu10": 0.01,
                            "psi_mem10": 0.02, "psi_io10": 0.03,
                            "mem_avail_gb": 10, "mem_total_gb": 16,
                            "disk_use_pct": 50
                        },
                        {
                            "node": "xbabe4", "up": true, "health": "healthy",
                            "cores": 12, "effective_cores": 10,
                            "cpu_job_slot_multiplier": 2.0,
                            "job_slot_target": 20, "running_job_slots": 8,
                            "cpu_busy_pct": 100.0, "psi_cpu10": 0.25,
                            "psi_mem10": 0.05, "psi_io10": 0.10,
                            "mem_avail_gb": 20, "mem_total_gb": 32,
                            "disk_use_pct": 60
                        }
                    ],
                    "jobs": [
                        {"id":"train","project":"jope-selfplay-train","state":"running"},
                        {"id":"eval-old","project":"jope-selfplay-eval","state":"queued",
                         "queue_ts":"2026-07-27T00:00:01Z"},
                        {"id":"urgent","project":"important","state":"queued",
                         "queue_ts":"2026-07-27T00:00:02Z"}
                    ],
                    "families": [
                        {"family":"jope-selfplay-eval","effective_weight":1.0,
                         "queued_jobs":1,"held_jobs":0,"running_jobs":0},
                        {"family":"jope-selfplay-train","effective_weight":10.0,
                         "queued_jobs":0,"held_jobs":0,"running_jobs":1},
                        {"family":"important","effective_weight":10.0,
                         "queued_jobs":1,"held_jobs":0,"running_jobs":0}
                    ],
                    "counters": {
                        "jobs_queued":2,"jobs_running":1,"jobs_held":0,
                        "jobs_failed":7,"throughput_per_min":42
                    },
                    "alerts": [],
                    "events": [{"ts":"now","kind":"launch","job":"train","node":"xbabe2","msg":"ok"}]
                }
            })),
            status: Some(json!({
                "trainer":{"active_trainers":["train"],"labels":150000},
                "promotion":{"internal_learning":{"decision":"accepted"},"milestone":null},
                "ingestion":{"exact_rows_total":140000,"strict_wins":12},
                "evolution":{"consecutive_rejections":2},
                "catalog":{"task_counts":{"regression":2322,"binary":542}},
                "sealed_audit":{"isolated_from_training":true,"states":{"scored":0}},
                "pair_registry":{"pairs":5000000,"duplicates_prevented":4000000},
                "alerts":[]
            })),
            queue: Some(json!({
                "active_checkpoint_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "checkpoint_epoch":"checkpoint-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "mode":"submit","admission_allowed":true,
                "queued_starting":200,"queued_target":256,"queued_hard_cap":512,
                "queue_maximum_age_seconds":20.0,"dispatching":1,"running":50,"held":0,
                "newest_micro_wave":"strict-wave-000020",
                "admission_waves":[{"wave":"strict-wave-000020"}],
                "unclaimed_current_checkpoint_manifests":300
            })),
            freshness: Some(json!({
                "checkpoint_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "stale_epoch_count":0,"stale_held":0
            })),
            supply: Some(json!({
                "ready_manifests":300,"awaiting_keeper_ack":null
            })),
            proposal_summary: Some(json!({
                "model_checkpoint_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "counts":{"regression":640,"binary":640},"proposal_validity":1.0
            })),
            flywheel: Some(json!({"last_frozen_watermark":100000})),
            accepted_by_task: Some(json!({
                "schema": "JopeSelfplayAcceptedByTask.v1",
                "tasks": {
                    "binary": {
                        "checkpoint_sha256": "b".repeat(64),
                        "updated_at_ms": 900
                    },
                    "regression": {
                        "checkpoint_sha256": "r".repeat(64),
                        "updated_at_ms": 800
                    }
                }
            })),
            evidence_bytes: Some(1024 * 1024),
            local_events: Vec::new(),
            source_states: vec![
                SourceState {
                    name: "scq".into(),
                    healthy: true,
                    note: "snapshot".into(),
                },
                SourceState {
                    name: "status".into(),
                    healthy: true,
                    note: "status-v3".into(),
                },
            ],
            collected_at_ms: 1_000,
            collection_latency_ms: 40,
            trainer_fallback: None,
        }
    }

    #[test]
    fn compact_terminal_defaults_to_live_deep_learning_and_honors_the_tab() {
        use ratatui::backend::TestBackend;

        assert_eq!(UiState::default().tab, Tab::GpuDl);
        let job_id = "jope-selfplay-train-004731939-c-r000035";
        let dashboard = Dashboard {
            collected_at_ms: 10_000,
            collection_latency_ms: 40,
            sources_healthy: 8,
            sources_total: 8,
            gpus: vec![GpuRow {
                node: "xbabe2".to_owned(),
                owner_job: Some(job_id.to_owned()),
                util_pct: 61.0,
                mem_used_mb: 9_820,
                mem_total_mb: 24_576,
                power_w: 128.0,
                ..GpuRow::default()
            }],
            trainers: vec![TrainerDetail {
                job_id: job_id.to_owned(),
                node: "xbabe2".to_owned(),
                is_trainer: true,
                phase: "cuda_train".to_owned(),
                epoch: 1,
                epochs: 24,
                batch: 35,
                total_batches: 86,
                pct: 40.7,
                running_loss: Some(3.7571),
                label_watermark: 4_731_939,
                progress_at_ms: 9_000,
                ..TrainerDetail::default()
            }],
            accepted_tasks: vec![
                AcceptedTaskFront {
                    task: "binary".to_owned(),
                    checkpoint_sha256: "b".repeat(64),
                    ..AcceptedTaskFront::default()
                },
                AcceptedTaskFront {
                    task: "regression".to_owned(),
                    checkpoint_sha256: "r".repeat(64),
                    ..AcceptedTaskFront::default()
                },
            ],
            watchdog: Some(TrainerWatchdogView {
                labels: 4_775_268,
                snapshot_watermark: 4_731_939,
                checkpoint_sha256: "7".repeat(64),
                action: "active_trainer_preserved".to_owned(),
                active_trainers: vec![job_id.to_owned()],
                ..TrainerWatchdogView::default()
            }),
            ..Dashboard::default()
        };
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render(frame, &dashboard, 250, UiState::default()))
            .unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("DEEP LEARNING • LIVE"), "{rendered}");
        assert!(rendered.contains(job_id), "{rendered}");
        assert!(rendered.contains("epoch 1/24 batch 35/86"), "{rendered}");
        assert!(rendered.contains("binary"), "{rendered}");
        assert!(rendered.contains("regression"), "{rendered}");
    }

    #[test]
    fn accepted_task_fronts_are_independent_and_visible() {
        let dashboard = build_dashboard(&fixture(), &SlowView::default());
        assert_eq!(dashboard.accepted_tasks.len(), 2);
        assert_eq!(dashboard.accepted_tasks[0].task, "binary");
        assert_eq!(
            dashboard.accepted_tasks[0].checkpoint_sha256,
            "b".repeat(64)
        );
        assert_eq!(dashboard.accepted_tasks[1].task, "regression");
        assert_eq!(
            dashboard.accepted_tasks[1].checkpoint_sha256,
            "r".repeat(64)
        );
    }

    #[test]
    fn snapshot_parses_queue_epoch_priorities_and_absent_xbabe5() {
        let dashboard = build_dashboard(&fixture(), &SlowView::default());
        assert_eq!(dashboard.scq_epoch, Some(77));
        assert_eq!(dashboard.queue.useful_depth, 200);
        assert_eq!(dashboard.queue.target, 256);
        assert_eq!(short_epoch(&dashboard.queue.epoch), "aaaaaaaaaaa…");
        assert!(dashboard
            .families
            .iter()
            .any(|row| row.family == "important" && row.weight == 10.0 && row.queued == 1));
        assert!(dashboard
            .families
            .iter()
            .any(|row| row.family == "jope-selfplay-eval" && row.weight == 1.0));
        assert!(dashboard
            .nodes
            .iter()
            .any(|row| row.node == "xbabe5" && row.state == "absent"));
        assert_eq!(dashboard.learning.trainer_state, "running");
    }

    #[test]
    fn aggregation_is_core_weighted_and_uses_maximum_pressure() {
        let dashboard = build_dashboard(&fixture(), &SlowView::default());
        assert!((dashboard.cluster.cpu_busy_pct - 87.5).abs() < f64::EPSILON);
        assert_eq!(dashboard.cluster.psi_cpu10, 0.25);
        assert_eq!(dashboard.cluster.psi_mem10, 0.05);
        assert_eq!(dashboard.cluster.psi_io10, 0.10);
        assert_eq!(dashboard.cluster.slot_target, 23);
        assert_eq!(dashboard.cluster.running_slots, 10);
        assert!(dashboard
            .alerts
            .iter()
            .any(|alert| alert.contains("xbabe4 CPU hard guard")));
    }

    #[test]
    fn degraded_sources_do_not_discard_live_scq_data() {
        let mut raw = fixture();
        raw.queue = None;
        raw.source_states.push(SourceState {
            name: "queue-v3".to_owned(),
            healthy: false,
            note: "missing".to_owned(),
        });
        let dashboard = build_dashboard(&raw, &SlowView::default());
        assert_eq!(dashboard.cluster.nodes_up, 2);
        assert_eq!(dashboard.queue.queued, 2);
        assert!(dashboard
            .alerts
            .iter()
            .any(|alert| alert == "source queue-v3: missing"));
    }

    #[test]
    fn gpu_rows_join_owner_jobs_and_hints() {
        let scq = json!({
            "gpus": [
                {"node":"xbabe2","status":"allocated:jope-selfplay-train-1","util_pct":37.0,
                 "mem_used_mb":19758,"mem_total_mb":24576,"temp_c":47,
                 "power_w":119.0,"power_limit_w":350.0,"throttled":false},
                {"node":"xbabe3","status":"free","util_pct":0.0,
                 "mem_used_mb":1,"mem_total_mb":24576,"temp_c":26,
                 "power_w":20.0,"power_limit_w":350.0,"throttled":false}
            ],
            "jobs": [
                {"id":"jope-selfplay-train-1","project":"jope-selfplay-train","state":"running",
                 "node":"xbabe2","gpus":1,"run_dir":"/nonexistent/run"}
            ]
        });
        let jobs = discover_gpu_jobs(&scq);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].0, "jope-selfplay-train-1");
        assert!(jobs[0].3, "the trainer job-id prefix marks a trainer");
        let gpus = parse_gpus(&scq, &jobs);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].owner_job.as_deref(), Some("jope-selfplay-train-1"));
        assert_eq!(gpus[0].model_hint, "3090");
        assert_eq!(gpus[1].owner_job, None);
    }

    #[test]
    fn progress_resolution_prefers_newest_attempt() {
        let base = tempfile::tempdir().expect("tempdir");
        let old = base
            .path()
            .join("attempts/0/outputs/training/wm/train_progress.json");
        let new = base
            .path()
            .join("attempts/1/outputs/training/wm/train_progress.json");
        fs::create_dir_all(old.parent().unwrap()).unwrap();
        fs::create_dir_all(new.parent().unwrap()).unwrap();
        fs::write(&old, "{\"epoch\":1}").unwrap();
        fs::write(&new, "{\"epoch\":37,\"epochs\":60,\"phase\":\"B\"}").unwrap();
        let resolved = resolve_progress_file(base.path()).expect("progress file");
        assert_eq!(resolved, new);
        let mut detail = TrainerDetail::default();
        read_trainer_progress(&resolved, &mut detail);
        assert_eq!(detail.epoch, 37);
        assert_eq!(detail.epochs, 60);
        assert_eq!(detail.phase, "B");
        assert!(detail.progress_at_ms > 0);
    }

    #[test]
    fn receipts_scan_emits_new_events_once_with_attribution() {
        let base = tempfile::tempdir().expect("tempdir");
        let root = base.path();
        let wave_dir = root.join("waves/strict-wave-000001/manifests");
        fs::create_dir_all(&wave_dir).unwrap();
        fs::write(
            root.join("waves/strict-wave-000001/summary.json"),
            json!({
                "manifest_count": 1,
                "datasets": 1,
                "generated_at_ms": 5,
                "manifests": [{"sha256":"m1","relative_path":"manifests/one.json"}]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            wave_dir.join("one.json"),
            json!({
                "candidates": [
                    {"semantic_sha256":"s1","source":"checkpoint","slot_group":"dl_proposal"},
                    {"semantic_sha256":"s2","source":"mutation","slot_group":"offspring"}
                ]
            })
            .to_string(),
        )
        .unwrap();
        let receipt_dir = root.join("receipts/jte-test1");
        fs::create_dir_all(&receipt_dir).unwrap();
        fs::write(
            receipt_dir.join("ds1.done"),
            json!({"finished_at_ms": 1000, "node":"xbabe3", "receipts":2, "valid":2}).to_string(),
        )
        .unwrap();
        let rows = [
            json!({"task":"regression","dataset_hash":"ds1","semantic_sha256":"s1",
                   "score":0.9,"valid":true,"manifest_sha256":"m1",
                   "generation":"strict-wave-000001","node":"xbabe3","created_at_ms":1000}),
            json!({"task":"regression","dataset_hash":"ds1","semantic_sha256":"s2",
                   "score":0.5,"valid":true,"manifest_sha256":"m1",
                   "generation":"strict-wave-000001","node":"xbabe3","created_at_ms":1001}),
        ];
        fs::write(
            receipt_dir.join("ds1.receipts.jsonl"),
            rows.iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        let mut state = SlowState::new();
        state
            .champions
            .insert(("regression".to_owned(), "ds1".to_owned()), 0.8);
        scan_receipts(root, &mut state);
        assert_eq!(state.ring.len(), 2);
        let dl_event = state
            .ring
            .iter()
            .find(|event| event.score == Some(0.9))
            .expect("dl event");
        assert_eq!(dl_event.origin, Origin::DlProposal);
        assert!(dl_event
            .delta_vs_champion
            .is_some_and(|delta| (delta - 0.1).abs() < 1e-9));
        let gp_event = state
            .ring
            .iter()
            .find(|event| event.score == Some(0.5))
            .expect("gp event");
        assert_eq!(gp_event.origin, Origin::GpOffspring);
        assert_eq!(state.stats_window.len(), 1);
        assert_eq!(state.stats_window[0], (Origin::DlProposal, true));
        scan_receipts(root, &mut state);
        assert_eq!(state.ring.len(), 2, "second scan must not re-emit");
    }

    #[test]
    fn origin_mapping_and_sparkline() {
        assert_eq!(slot_group_origin("dl_proposal", ""), Origin::DlProposal);
        assert_eq!(slot_group_origin("offspring", ""), Origin::GpOffspring);
        assert_eq!(slot_group_origin("local_elite", ""), Origin::LocalElite);
        assert_eq!(slot_group_origin("common_winner", ""), Origin::CommonWinner);
        assert_eq!(
            slot_group_origin("", "deterministic_mutation_crossover"),
            Origin::GpOffspring
        );
        assert_eq!(
            slot_group_origin("", "accepted_dl_checkpoint"),
            Origin::DlProposal
        );
        assert_eq!(sparkline(&[]), "-");
        assert_eq!(sparkline(&[1.0, 1.0]).chars().count(), 2);
        let line = sparkline(&[0.0, 0.5, 1.0]);
        assert!(line.starts_with('▁') && line.ends_with('█'));
    }

    #[test]
    fn slow_view_defaults_render_and_alarm_alerts_surface() {
        let mut slow = SlowView::default();
        slow.health_alarms
            .push("ALARM[crit] disk: 93 (thr 90)".to_owned());
        let dashboard = build_dashboard(&fixture(), &slow);
        assert!(dashboard
            .alerts
            .iter()
            .any(|alert| alert.contains("ALARM[crit] disk")));
        assert!(dashboard.gpus.is_empty());
        assert!(dashboard.trainers.is_empty());
    }

    #[test]
    fn metrics_history_reads_and_downsamples() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train_metrics.jsonl");
        let mut body = String::new();
        for epoch in 1..=1200_u64 {
            body.push_str(&format!(
                "{{\"epoch\":{epoch},\"train_loss\":{},\"val_loss\":{},\
                 \"macro_validation_bad_passes\":0,\"overfit_gap\":1.0,\
                 \"val_pairwise_win_accuracy\":0.6}}\n",
                2.0 - epoch as f64 / 1200.0,
                9.0 - epoch as f64 / 300.0,
            ));
        }
        std::fs::write(&path, body).unwrap();
        let history = read_metrics_history(&path);
        assert!(history.len() <= DL_HISTORY_MAX + 1);
        assert_eq!(history.first().unwrap().epoch, 1);
        assert_eq!(history.last().unwrap().epoch, 1200);
        assert!(history.last().unwrap().val_loss < history.first().unwrap().val_loss);
    }

    fn run_series(label: &str, live: bool, points: Vec<TrainerMetricPoint>) -> RunSeries {
        RunSeries {
            run_label: format!("001805437-{label}"),
            short_label: label.to_owned(),
            points,
            live,
            updated_at_ms: 0,
        }
    }

    #[test]
    fn loss_and_pairwise_series_extract_by_epoch() {
        let history = DlHistory {
            runs: vec![run_series(
                "a-g9",
                true,
                vec![
                    TrainerMetricPoint {
                        epoch: 1,
                        train_loss: 2.0,
                        val_loss: 9.0,
                        val_pairwise: Some(0.55),
                        ..TrainerMetricPoint::default()
                    },
                    TrainerMetricPoint {
                        epoch: 2,
                        train_loss: 1.5,
                        val_loss: 8.0,
                        val_pairwise: None,
                        ..TrainerMetricPoint::default()
                    },
                ],
            )],
            fetched_at_ms: 0,
        };
        let series = continuous_series(&history);
        assert_eq!(series.train, vec![(1.0, 2.0), (2.0, 1.5)]);
        assert_eq!(series.val, vec![(1.0, 9.0), (2.0, 8.0)]);
        assert_eq!(series.pairwise, vec![(1.0, 0.55)]);
        assert_eq!(series.total_epochs, 2.0);
    }

    /// The defect the operator hit: trainer a-g8 completed and its successor
    /// arrived under a new job id, so a chart keyed on the live job blanked.
    /// The series must span every generation on one axis and keep rendering
    /// with zero live trainers.
    #[test]
    fn loss_series_is_continuous_across_trainer_generations() {
        let epoch = |n: u64, train: f64, val: f64, pairwise: f64| TrainerMetricPoint {
            epoch: n,
            train_loss: train,
            val_loss: val,
            val_pairwise: Some(pairwise),
            ..TrainerMetricPoint::default()
        };
        let history = DlHistory {
            runs: vec![
                run_series(
                    "a-g8",
                    false,
                    vec![epoch(1, 9.0, 9.02, 0.61), epoch(2, 5.0, 5.10, 0.66)],
                ),
                run_series("a-g9", true, vec![epoch(1, 3.35, 3.68, 0.70)]),
            ],
            fetched_at_ms: 0,
        };
        let series = continuous_series(&history);
        // a-g9 epoch 1 lands after a-g8's two epochs, not back at x=1.
        assert_eq!(series.train, vec![(1.0, 9.0), (2.0, 5.0), (3.0, 3.35)]);
        assert_eq!(series.total_epochs, 3.0);
        assert_eq!(
            series.marks,
            vec![
                (1.0, "a-g8".to_owned(), false),
                (3.0, "a-g9".to_owned(), true),
            ]
        );
        // Prior generations survive a changeover with nothing running.
        let dead = DlHistory {
            runs: history
                .runs
                .iter()
                .map(|run| RunSeries {
                    live: false,
                    ..run.clone()
                })
                .collect(),
            fetched_at_ms: 0,
        };
        assert_eq!(continuous_series(&dead).train.len(), 3);
    }

    /// Absent losses parse as 0.0; plotting them would drop the curve to the
    /// floor on a partially written row.
    #[test]
    fn continuous_series_drops_placeholder_zero_losses() {
        let history = DlHistory {
            runs: vec![run_series(
                "a-g9",
                true,
                vec![
                    TrainerMetricPoint {
                        epoch: 1,
                        train_loss: 3.0,
                        val_loss: 3.5,
                        ..TrainerMetricPoint::default()
                    },
                    TrainerMetricPoint {
                        epoch: 2,
                        train_loss: 0.0,
                        val_loss: 0.0,
                        ..TrainerMetricPoint::default()
                    },
                ],
            )],
            fetched_at_ms: 0,
        };
        let series = continuous_series(&history);
        assert_eq!(series.train, vec![(1.0, 3.0)]);
        assert_eq!(series.val, vec![(1.0, 3.5)]);
        // The generation still spans both epochs on the axis.
        assert_eq!(series.total_epochs, 2.0);
    }

    #[test]
    fn generation_ruler_marks_every_generation() {
        let series = DlSeries {
            marks: vec![
                (1.0, "a-g7".to_owned(), false),
                (21.0, "a-g8".to_owned(), false),
                (45.0, "a-g9".to_owned(), true),
            ],
            total_epochs: 47.0,
            ..DlSeries::default()
        };
        let text = generation_ruler(&series, 80)
            .spans
            .iter()
            .map(|span| span.content.to_string())
            .collect::<String>();
        assert!(text.contains("├a-g7"), "{text}");
        assert!(text.contains("├a-g8"), "{text}");
        assert!(text.contains("├a-g9"), "{text}");
        // Ordered left to right by epoch offset.
        assert!(text.find("├a-g7") < text.find("├a-g8"), "{text}");
        assert!(text.find("├a-g8") < text.find("├a-g9"), "{text}");
        assert!(generation_ruler(&DlSeries::default(), 80)
            .spans
            .iter()
            .any(|span| span.content.contains("no trainer generations")),);
    }

    #[test]
    fn generations_panel_derives_from_history() {
        let history = DlHistory {
            runs: vec![
                run_series(
                    "a-g8",
                    false,
                    vec![TrainerMetricPoint {
                        epoch: 24,
                        val_pairwise: Some(0.72),
                        ..TrainerMetricPoint::default()
                    }],
                ),
                run_series("a-g9", true, Vec::new()),
            ],
            fetched_at_ms: 0,
        };
        assert_eq!(
            generations_from_history(&history),
            vec![
                ("001805437-a-g8".to_owned(), Some(0.72)),
                ("001805437-a-g9".to_owned(), None),
            ]
        );
    }

    /// The funnel panel was a fixed 4 rows — two lines per task plus borders
    /// means only regression fit, so the binary totals were invisible on tab 4.
    /// Now all four task lanes must be visible, including census-only rows for
    /// corpora the index has not scanned yet (everything still waiting).
    #[test]
    fn dl_tab_funnel_shows_both_task_totals() {
        use ratatui::backend::TestBackend;
        let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
        dashboard.funnel = UnlockFunnel {
            rows: vec![
                FunnelRow {
                    task: "regression".to_owned(),
                    registry: 27_525,
                    active: 8_460,
                    in_play: 6_404,
                    queued: 15_509,
                    encoding: 3_353,
                    ..FunnelRow::default()
                },
                FunnelRow {
                    task: "binary".to_owned(),
                    registry: 24_840,
                    active: 2_073,
                    in_play: 2_028,
                    encoding: 22_747,
                    ..FunnelRow::default()
                },
                FunnelRow {
                    task: "timeseries".to_owned(),
                    registry: 42,
                    ..FunnelRow::default()
                },
                FunnelRow {
                    task: "multiclass".to_owned(),
                    registry: 272,
                    encoded: 269,
                    queued: 269,
                    ..FunnelRow::default()
                },
            ],
            source: "index".to_owned(),
            fetched_at_ms: now_ms(),
        };
        let backend = TestBackend::new(160, 60);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render(
                    frame,
                    &dashboard,
                    250,
                    UiState {
                        tab: Tab::Dl,
                        arm: 0,
                    },
                )
            })
            .unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("REG 27,525"), "regression totals visible");
        assert!(rendered.contains("BIN 24,840"), "binary totals visible");
        assert!(
            rendered.contains("active 8,460"),
            "regression active visible"
        );
        assert!(rendered.contains("active 2,073"), "binary active visible");
        assert!(rendered.contains("TS 42"), "timeseries census row visible");
        assert!(rendered.contains("MC 272"), "multiclass census row visible");
        assert!(
            rendered.contains("queued +269"),
            "installed multiclass encodings visible as queued"
        );
    }

    /// Data sources that predate a task must not hide its corpus: missing
    /// tasks get census rows (registry only, everything locked/waiting) and
    /// rows always land in canonical display order.
    #[test]
    fn funnel_rows_are_task_complete_and_canonically_ordered() {
        let mut rows = vec![
            FunnelRow {
                task: "binary".to_owned(),
                registry: 10,
                ..FunnelRow::default()
            },
            FunnelRow {
                task: "regression".to_owned(),
                registry: 20,
                ..FunnelRow::default()
            },
        ];
        complete_funnel_rows(&mut rows, |task| match task {
            "timeseries" => Some((42, 0)),
            "multiclass" => Some((272, 269)),
            _ => None,
        });
        let order: Vec<&str> = rows.iter().map(|row| row.task.as_str()).collect();
        assert_eq!(order, vec!["regression", "binary", "timeseries", "multiclass"]);
        assert_eq!(rows[2].registry, 42);
        assert_eq!(rows[2].locked(), 42, "census rows report the corpus as waiting");
        assert_eq!(rows[3].registry, 272);
        assert_eq!(rows[3].queued, 269, "installed encodings surface as queued");
        assert_eq!(rows[3].encoded, 269);
        assert_eq!(rows[3].locked(), 3, "remainder still awaiting encoding");
        assert_eq!(rows[3].active, 0, "census rows are not in the catalog yet");
        // Existing rows are never duplicated or overwritten by the census.
        let mut again = rows.clone();
        complete_funnel_rows(&mut again, |_| Some((999, 999)));
        assert_eq!(again, rows);
    }

    #[test]
    fn funnel_tags_cover_all_tasks_and_flag_unknowns() {
        assert_eq!(funnel_tag("regression"), "REG");
        assert_eq!(funnel_tag("binary"), "BIN");
        assert_eq!(funnel_tag("timeseries"), "TS");
        assert_eq!(funnel_tag("multiclass"), "MC");
        assert_eq!(funnel_tag("multi_label"), "MC");
        assert_eq!(funnel_tag("mystery"), "?");
        assert_eq!(funnel_tag_compact("timeseries"), "ts");
        assert_eq!(funnel_tag_compact("multi_label"), "mc");
        assert_eq!(funnel_tag_compact("mystery"), "?");
    }

    #[test]
    fn short_generation_label_strips_watermark() {
        assert_eq!(short_generation_label("001805437-a-g9"), "a-g9");
        assert_eq!(short_generation_label("001805437-c-r000002"), "c-r000002");
        assert_eq!(short_generation_label("legacy"), "legacy");
    }

    #[test]
    fn validation_phase_never_looks_stuck() {
        let trainer = TrainerDetail {
            phase: "validation".to_owned(),
            epoch: 20,
            epochs: 60,
            batch: 96,
            total_batches: 96,
            validation_batch: 87,
            validation_batches: 175,
            ..TrainerDetail::default()
        };
        let label = phase_label(&trainer);
        assert!(label.contains("VALIDATION 87/175"), "{label}");
        let training = TrainerDetail {
            phase: "train".to_owned(),
            epoch: 20,
            epochs: 60,
            batch: 42,
            total_batches: 96,
            ..TrainerDetail::default()
        };
        assert!(phase_label(&training).contains("batch 42/96"));
    }

    #[test]
    fn trend_arrows_point_the_right_way() {
        assert_eq!(trend_arrow(2.0, 1.0), "↓");
        assert_eq!(trend_arrow(1.0, 2.0), "↑");
        assert_eq!(trend_arrow(1.0, 1.0), "→");
    }

    #[test]
    fn dl_tab_renders_headless_with_and_without_trainer() {
        use ratatui::backend::TestBackend;
        let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
        let backend = TestBackend::new(140, 45);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render(
                    frame,
                    &dashboard,
                    250,
                    UiState {
                        tab: Tab::Dl,
                        arm: 0,
                    },
                )
            })
            .unwrap();
        dashboard.trainers.push(TrainerDetail {
            job_id: "jope-selfplay-train-x".to_owned(),
            node: "xbabe2".to_owned(),
            is_trainer: true,
            phase: "validation".to_owned(),
            epoch: 20,
            epochs: 60,
            validation_batch: 87,
            validation_batches: 175,
            patience: 12,
            history: vec![
                TrainerMetricPoint {
                    epoch: 19,
                    train_loss: 1.6,
                    val_loss: 7.6,
                    overfit_gap: 6.0,
                    val_pairwise: Some(0.64),
                    ..TrainerMetricPoint::default()
                },
                TrainerMetricPoint {
                    epoch: 20,
                    train_loss: 1.5,
                    val_loss: 7.4,
                    overfit_gap: 5.9,
                    val_pairwise: Some(0.65),
                    ..TrainerMetricPoint::default()
                },
            ],
            ..TrainerDetail::default()
        });
        dashboard.daemon_ages = vec![("queue-keeper-v3".to_owned(), 30, 450)];
        dashboard.generations = vec![("001805437-c".to_owned(), Some(0.6452))];
        terminal
            .draw(|frame| {
                render(
                    frame,
                    &dashboard,
                    250,
                    UiState {
                        tab: Tab::Dl,
                        arm: 0,
                    },
                )
            })
            .unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(
            rendered.contains("VALIDATION 87/175"),
            "validation phase must be explicit"
        );
        assert!(rendered.contains("LOSS"), "loss chart title present");
    }

    /// Baseline fixture mirroring ops/seeding-curve.v1.json as the reconcile
    /// hook writes it today: one generation (fbcd5459), 5.2% DL win share.
    fn seeding_curve_fixture() -> serde_json::Value {
        json!({
            "schema": "JopeSeedingCurve.v1",
            "generated_at_ms": 1_785_300_000_000_u64,
            "curve": [{
                "generation_index": 0,
                "model_checkpoint_sha256":
                    "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592",
                "strict_wins_total": 1668,
                "strict_wins_dl": 86,
                "dl_strict_win_share": 0.051_558_752_997_601_92,
                "dl_strict_win_share_pct": 5.2,
                "champions_total": 2385,
                "observations": {
                    "rows_scanned": 3_513_076_u64,
                    "dl_rows": 291_000_u64,
                    "dl_observation_share": 0.082_833,
                    "cursor_created_at_ms": 1_785_203_238_986_i64,
                    "caught_up": true
                }
            }]
        })
    }

    #[test]
    fn seeding_curve_reader_parses_the_ops_file() {
        let temporary = tempfile::TempDir::new().unwrap();
        let root = temporary.path();
        std::fs::create_dir_all(root.join("ops")).unwrap();
        std::fs::write(
            root.join("ops/seeding-curve.v1.json"),
            serde_json::to_vec(&seeding_curve_fixture()).unwrap(),
        )
        .unwrap();
        let view = read_seeding_curve(root).expect("curve file parses");
        assert_eq!(view.points.len(), 1);
        let point = &view.points[0];
        assert_eq!(point.checkpoint_short, "fbcd5459");
        assert_eq!(point.dl_win_share_pct, Some(5.2));
        assert!((point.dl_obs_share_pct.unwrap() - 8.2833).abs() < 1e-9);
        assert_eq!(point.strict_wins_total, 1668);
        assert!(point.observations_caught_up);
        assert!(view.fetched_at_ms > 0);
        assert!(
            read_seeding_curve(&root.join("absent")).is_none(),
            "missing file keeps the prior snapshot"
        );
    }

    #[test]
    fn dl_tab_generations_panel_renders_dl_share_column() {
        use ratatui::backend::TestBackend;
        let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
        dashboard.generations = vec![("001805437-a-g8".to_owned(), Some(0.7263))];
        // The unlock funnel and the DL history each carry their own NO-DATA
        // tag; keep them quiet so the negative assertion below isolates the
        // seeding-curve tag.
        dashboard.funnel.fetched_at_ms = now_ms();
        dashboard.dl_history.fetched_at_ms = now_ms();
        dashboard.seeding = SeedingCurveView {
            points: vec![SeedingPoint {
                checkpoint_short: "fbcd5459".to_owned(),
                dl_win_share_pct: Some(5.2),
                dl_obs_share_pct: Some(8.3),
                strict_wins_total: 1668,
                observations_caught_up: true,
            }],
            fetched_at_ms: now_ms(),
        };
        let backend = TestBackend::new(140, 50);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render(
                    frame,
                    &dashboard,
                    250,
                    UiState {
                        tab: Tab::Dl,
                        arm: 0,
                    },
                )
            })
            .unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(
            rendered.contains("DL-win"),
            "DL-share column header present"
        );
        assert!(rendered.contains("5.2%"), "DL strict-win share rendered");
        assert!(rendered.contains("fbcd5459"), "seed checkpoint labeled");
        assert!(
            !rendered.contains("[NO DATA]"),
            "fresh curve must not be tagged NO DATA"
        );
        // A dashboard that never saw the curve file reports NO DATA instead.
        dashboard.seeding = SeedingCurveView::default();
        terminal
            .draw(|frame| {
                render(
                    frame,
                    &dashboard,
                    250,
                    UiState {
                        tab: Tab::Dl,
                        arm: 0,
                    },
                )
            })
            .unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(
            rendered.contains("[NO DATA]"),
            "missing curve is called out"
        );
    }

    fn arm_fixture(job: &str, node: &str, epoch: u64, advantage: Option<f64>) -> TrainerDetail {
        TrainerDetail {
            job_id: job.to_owned(),
            node: node.to_owned(),
            is_trainer: true,
            phase: "training".to_owned(),
            epoch,
            epochs: 24,
            patience: 12,
            bad_passes: 1,
            history: vec![TrainerMetricPoint {
                epoch,
                train_loss: 1.0 + epoch as f64 / 100.0,
                val_loss: 2.0 + epoch as f64 / 100.0,
                val_pairwise: Some(0.65),
                val_ablation_pairwise: Some(0.64),
                val_conditional_advantage: advantage,
                ..TrainerMetricPoint::default()
            }],
            ..TrainerDetail::default()
        }
    }

    #[test]
    fn selected_arm_wraps_against_live_count() {
        assert_eq!(selected_arm(0, 3), 0);
        assert_eq!(selected_arm(4, 3), 1);
        // An arm exiting must not leave the selection dangling.
        assert_eq!(selected_arm(usize::MAX, 3), usize::MAX % 3);
        assert_eq!(selected_arm(7, 0), 0, "no arms must not panic");
    }

    #[test]
    fn training_arms_order_puts_real_trainers_first() {
        let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
        let mut encoder = arm_fixture("jope-binexp-b2c5", "xbabe3", 2, None);
        encoder.is_trainer = false;
        dashboard.trainers = vec![
            encoder,
            arm_fixture("jope-selfplay-train-a", "xbabe1", 5, Some(0.01)),
        ];
        let arms = training_arms(&dashboard);
        assert_eq!(arms[0].job_id, "jope-selfplay-train-a");
        assert_eq!(arms[1].job_id, "jope-binexp-b2c5");
    }

    /// Regression, observed live 2026-07-28 04:0xZ: the DL proposal driver is
    /// submitted under `project = "jope-selfplay-train"` so it lands in the
    /// priority-10 lane. A `project`-based trainer test classified it as a
    /// trainer, it sorted into arm 0, and the DL tab header rendered
    /// `live jope-selfplay-propose-… [NO DATA]` with an empty TRAINER STATE
    /// box while a-g9 was at epoch 14/24. Only the job id may decide.
    #[test]
    fn propose_job_in_the_trainer_project_is_not_a_trainer_arm() {
        let scq = json!({
            "jobs": [
                {
                    "id": "jope-selfplay-propose-fbcd5459f2c1-c228d912ecb6d",
                    "project": "jope-selfplay-train",
                    "state": "running",
                    "node": "xbabe3",
                    "gpus": 1,
                    "run_dir": "/runs/jope-selfplay-propose-fbcd5459f2c1-c228d912ecb6d"
                },
                {
                    "id": "jope-selfplay-train-001805437-a-g9",
                    "project": "jope-selfplay-train",
                    "state": "running",
                    "node": "xbabe2",
                    "gpus": 1,
                    "run_dir": "/runs/jope-selfplay-train-001805437-a-g9"
                }
            ]
        });
        let jobs = discover_gpu_jobs(&scq);
        assert_eq!(jobs.len(), 2);
        let propose = jobs
            .iter()
            .find(|(id, _, _, _)| id.starts_with("jope-selfplay-propose-"))
            .expect("propose job discovered");
        assert!(!propose.3, "propose job must not be flagged as a trainer");
        let trainer = jobs
            .iter()
            .find(|(id, _, _, _)| id.starts_with(TRAINER_ID_PREFIX))
            .expect("trainer job discovered");
        assert!(trainer.3, "the real trainer must still be flagged");

        let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
        dashboard.trainers = jobs
            .iter()
            .map(|(id, node, _, is_trainer)| {
                let mut arm = arm_fixture(id, node, 3, Some(0.03));
                arm.is_trainer = *is_trainer;
                arm
            })
            .collect();
        let arms = training_arms(&dashboard);
        assert_eq!(arms[0].job_id, "jope-selfplay-train-001805437-a-g9");
    }

    /// SCQ requeues in place under a fresh `attempts/<n>/` and keeps the old
    /// attempt on disk, so a resolved-path cache guarded only by `is_file()`
    /// pins the chart to a dead attempt. The cache key must move with the
    /// newest attempt.
    #[test]
    fn newest_attempt_id_moves_when_a_run_requeues() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run = dir.path().join("jope-selfplay-train-001805437-a-g9");
        assert_eq!(newest_attempt_id(&run), None, "unstaged run has no attempt");
        fs::create_dir_all(run.join("attempts/1/outputs")).expect("attempt 1");
        assert_eq!(newest_attempt_id(&run), Some(1));
        fs::create_dir_all(run.join("attempts/2/outputs")).expect("attempt 2");
        assert_eq!(
            newest_attempt_id(&run),
            Some(2),
            "a requeue must invalidate the cached attempt"
        );
    }

    /// End-to-end version of the same defect: after a requeue the continuous
    /// series must follow the NEW attempt even though the old attempt's
    /// train_metrics.jsonl is still a readable file.
    #[test]
    fn dl_history_follows_the_new_attempt_after_a_requeue() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let run_id = "jope-selfplay-train-001805437-a-g9";
        let attempt_dir = |n: u64| {
            root.join(run_id)
                .join(format!("attempts/{n}/outputs/training/watermark-1"))
        };
        let write_attempt = |n: u64, loss: f64| {
            let out = attempt_dir(n);
            fs::create_dir_all(&out).expect("attempt dir");
            fs::write(out.join("train_progress.json"), "{\"epoch\":1}").expect("progress");
            fs::write(
                out.join("train_metrics.jsonl"),
                format!(
                    "{{\"epoch\":1,\"train_loss\":{loss},\"val_loss\":{loss},\
                     \"val_pairwise_win_accuracy\":0.7,\"val_ablation_pairwise_win\":0.6}}\n"
                ),
            )
            .expect("metrics");
        };
        write_attempt(1, 4.0);
        let live: HashSet<String> = [run_id.to_owned()].into_iter().collect();
        let mut state = SlowState::new();
        let first = collect_dl_history(root, &live, &mut state);
        assert_eq!(first.runs.len(), 1);
        assert_eq!(first.runs[0].points[0].train_loss, 4.0);

        write_attempt(2, 2.0);
        let second = collect_dl_history(root, &live, &mut state);
        assert_eq!(second.runs.len(), 1);
        assert_eq!(
            second.runs[0].points[0].train_loss, 2.0,
            "cache must not pin the series to the dead attempt"
        );
        assert!(
            attempt_dir(1).join("train_metrics.jsonl").is_file(),
            "the old attempt is still on disk - that is exactly the trap"
        );
    }

    #[test]
    fn arms_keep_separate_histories_and_report_best_advantage() {
        let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
        dashboard.trainers = vec![
            arm_fixture("jope-selfplay-train-a", "xbabe1", 5, Some(-0.004)),
            arm_fixture("jope-selfplay-train-b", "xbabe2", 9, Some(0.021)),
            arm_fixture("jope-selfplay-train-c", "xbabe3", 3, None),
        ];
        let arms = training_arms(&dashboard);
        assert_eq!(arms.len(), 3);
        // Per-arm isolation: each arm reports its own epoch and advantage.
        assert_eq!(arm_latest(arms[0]).unwrap().epoch, 5);
        assert_eq!(arm_latest(arms[1]).unwrap().epoch, 9);
        assert_eq!(
            arm_latest(arms[1]).unwrap().val_conditional_advantage,
            Some(0.021)
        );
        // An arm with no advantage recorded must not borrow a neighbour's.
        assert_eq!(arm_latest(arms[2]).unwrap().val_conditional_advantage, None);
    }

    #[test]
    fn warming_up_arm_has_no_metrics_to_show() {
        let mut warming = arm_fixture("jope-selfplay-train-new", "xbabe1", 0, None);
        warming.history.clear();
        assert!(arm_latest(&warming).is_none());
    }

    #[test]
    fn dl_tab_renders_three_arms_with_selection() {
        use ratatui::backend::TestBackend;
        let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
        // arm3 has just launched: its metrics file exists but has no rows yet.
        let mut warming = arm_fixture("jope-selfplay-train-arm3", "xbabe3", 0, None);
        warming.history.clear();
        dashboard.trainers = vec![
            arm_fixture("jope-selfplay-train-arm1", "xbabe1", 5, Some(-0.004)),
            arm_fixture("jope-selfplay-train-arm2", "xbabe2", 9, Some(0.021)),
            warming,
        ];
        let backend = TestBackend::new(160, 50);
        let mut terminal = Terminal::new(backend).unwrap();
        for arm in 0..4usize {
            terminal
                .draw(|frame| render(frame, &dashboard, 250, UiState { tab: Tab::Dl, arm }))
                .unwrap();
        }
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(
            rendered.contains("TRAINING ARMS (3)"),
            "comparison strip present"
        );
        assert!(
            rendered.contains("warming up"),
            "arm without metrics is labelled"
        );
        // arm=3 wraps back to arm 0, whose job must be the charted one.
        assert!(
            rendered.contains("arm 1/3"),
            "selected arm is identified in the chart title"
        );
    }

    #[test]
    fn gpu_tab_survives_three_simultaneous_arms() {
        use ratatui::backend::TestBackend;
        let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
        dashboard.trainers = vec![
            arm_fixture("jope-selfplay-train-arm1", "xbabe1", 5, Some(0.01)),
            arm_fixture("jope-selfplay-train-arm2", "xbabe2", 9, Some(0.02)),
            arm_fixture("jope-selfplay-train-arm3", "xbabe3", 3, Some(0.00)),
        ];
        let backend = TestBackend::new(140, 45);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render(
                    frame,
                    &dashboard,
                    250,
                    UiState {
                        tab: Tab::GpuDl,
                        arm: 0,
                    },
                )
            })
            .unwrap();
    }

    #[test]
    fn funnel_rows_parse_from_driver_summary() {
        let driver: Value = serde_json::from_str(
            r#"{"stage_summary":{"regression":{"registry":27525,"sidecars_complete":24172,
                "catalog_active":6412,"observed":2491},
                "binary":{"registry":24840,"sidecars_complete":2091,
                "catalog_active":2073,"observed":657}}}"#,
        )
        .unwrap();
        let rows = funnel_rows_from_driver(&driver);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].task, "regression");
        assert_eq!(rows[0].registry, 27_525);
        assert_eq!(rows[0].active, 6_412);
        assert_eq!(rows[1].task, "binary");
        assert_eq!(rows[1].in_play, 657);
    }

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn funnel_rows_query_from_legacy_index_fixture_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dataset-state.sqlite");
        let connection = rusqlite::Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE datasets(task TEXT NOT NULL, dataset_hash TEXT NOT NULL,
                   sidecars_complete INTEGER NOT NULL DEFAULT 0,
                   catalog_active INTEGER NOT NULL DEFAULT 0,
                   obs_count INTEGER NOT NULL DEFAULT 0,
                   PRIMARY KEY(task, dataset_hash)) WITHOUT ROWID;
                 INSERT INTO datasets VALUES
                   ('regression','r1',1,1,5),('regression','r2',1,1,0),
                   ('regression','r3',1,0,0),('binary','b1',1,1,2),
                   ('binary','b2',0,0,0);",
            )
            .unwrap();
        drop(connection);
        let rows = funnel_rows_from_index(&path).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].task, "regression");
        assert_eq!(rows[0].registry, 3);
        assert_eq!(rows[0].encoded, 3);
        assert_eq!(rows[0].active, 2);
        assert_eq!(rows[0].in_play, 1);
        assert_eq!(rows[0].queued, 0, "legacy schema zeroes the new stages");
        assert_eq!(rows[1].task, "binary");
        assert_eq!(rows[1].registry, 2);
        assert_eq!(rows[1].active, 1);
        assert_eq!(rows[1].in_play, 1);
    }

    #[test]
    fn funnel_rows_query_full_pipeline_stages() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dataset-state.sqlite");
        let connection = rusqlite::Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE datasets(task TEXT NOT NULL, dataset_hash TEXT NOT NULL,
                   sidecars_complete INTEGER NOT NULL DEFAULT 0,
                   catalog_active INTEGER NOT NULL DEFAULT 0,
                   sealed INTEGER NOT NULL DEFAULT 0,
                   encode_job_state TEXT,
                   obs_count INTEGER NOT NULL DEFAULT 0,
                   PRIMARY KEY(task, dataset_hash)) WITHOUT ROWID;
                 INSERT INTO datasets VALUES
                   ('regression','r1',1,1,0,NULL,5),
                   ('regression','r2',1,0,0,NULL,0),
                   ('regression','r3',1,0,1,NULL,0),
                   ('regression','r4',0,0,0,'jope-regexp-c01',0),
                   ('regression','r5',0,0,0,NULL,0),
                   ('binary','b1',1,1,0,NULL,2),
                   ('binary','b2',0,0,0,'jope-binexp-b2c05',0);",
            )
            .unwrap();
        drop(connection);
        let rows = funnel_rows_from_index(&path).unwrap();
        let reg = &rows[0];
        assert_eq!(reg.task, "regression");
        assert_eq!(reg.registry, 5);
        assert_eq!(reg.active, 1);
        assert_eq!(reg.in_play, 1);
        assert_eq!(reg.queued, 1, "encoded, unsealed, not active");
        assert_eq!(reg.encoding, 1, "encode job assigned, not yet encoded");
        assert_eq!(reg.sealed, 1);
        assert_eq!(reg.locked(), 1);
        let bin = &rows[1];
        assert_eq!(bin.encoding, 1);
        assert_eq!(bin.queued, 0);
    }

    #[test]
    fn funnel_line_stage_colors_and_exact_numbers() {
        let row = FunnelRow {
            task: "regression".to_owned(),
            registry: 27_525,
            encoded: 24_172,
            active: 6_412,
            in_play: 2_491,
            queued: 17_504,
            encoding: 3_353,
            sealed: 256,
            ..Default::default()
        };
        let line = funnel_line(&row);
        let text = line_text(&line);
        assert!(text.contains("27,525"), "{text}");
        assert!(text.contains("active 6,412"), "{text}");
        assert!(text.contains("(in-play 2,491)"), "{text}");
        assert!(text.contains("queued +17,504"), "{text}");
        assert!(text.contains("encoding 3,353"), "{text}");
        let style_of = |needle: &str| {
            line.spans
                .iter()
                .find(|span| span.content.contains(needle))
                .map(|span| span.style)
                .unwrap()
        };
        assert_eq!(style_of("active 6,412").fg, Some(STAGE_ACTIVE));
        assert_eq!(style_of("in-play").fg, Some(STAGE_IN_PLAY));
        assert_eq!(style_of("queued").fg, Some(STAGE_QUEUED));
        assert_eq!(style_of("encoding").fg, Some(STAGE_ENCODING));
        assert_eq!(style_of("locked").fg, Some(STAGE_LOCKED));
        let bar = funnel_bar(&row, 40);
        assert!(!bar.spans.is_empty());
        let bar_cells: usize = bar
            .spans
            .iter()
            .map(|span| span.content.chars().count())
            .sum();
        assert!((38..=42).contains(&bar_cells), "bar width {bar_cells}");
    }

    #[test]
    fn cohort_label_shows_rotation_population() {
        let wave = WaveSummary {
            adaptive_regression: 6_412,
            adaptive_binary: 2_073,
            ..Default::default()
        };
        let label = cohort_label(8_485, &wave);
        assert!(label.contains("128 datasets/wave"), "{label}");
        assert!(label.contains("rotating over 8,485 active"), "{label}");
        let fallback = cohort_label(0, &wave);
        assert!(
            fallback.contains("rotating over 8,485 active"),
            "{fallback}"
        );
    }

    #[test]
    fn thousands_formats_exact_counts() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(24_840), "24,840");
        assert_eq!(thousands(2_060_048), "2,060,048");
    }

    #[test]
    fn funnel_deltas_measure_the_climb() {
        let mut history = VecDeque::new();
        let hour_ago = 1_000_000_u128;
        history.push_back((
            hour_ago,
            vec![("regression".to_owned(), 6_000_u64, 2_000_u64)],
        ));
        let mut rows = vec![FunnelRow {
            task: "regression".to_owned(),
            registry: 27_525,
            encoded: 24_172,
            active: 6_412,
            in_play: 2_491,
            ..Default::default()
        }];
        apply_funnel_deltas(&mut rows, &mut history, hour_ago + 1_800_000);
        assert_eq!(rows[0].delta_active_1h, 412);
        assert_eq!(rows[0].delta_in_play_1h, 491);
        assert_eq!(history.len(), 2);
        let text = line_text(&funnel_line(&rows[0]));
        assert!(text.contains("REG"), "{text}");
        assert!(text.contains("+412"), "{text}");
    }
}
