use super::*;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "jope-selfplay-tui",
    about = "Read-only Rust TUI for live SCQ and JOPE self-play state"
)]
pub(super) struct Args {
    /// Flywheel state root.
    #[arg(long, default_value = default_root_str())]
    pub(super) root: PathBuf,
    /// Native SCQ client.
    #[arg(long, default_value = default_scq_str())]
    pub(super) scq: PathBuf,
    /// Terminal redraw and source refresh cadence.
    #[arg(
        long,
        default_value_t = DEFAULT_REFRESH_MS,
        value_parser = clap::value_parser!(u64).range(50..=299)
    )]
    pub(super) refresh_ms: u64,
    /// Root of SCQ run directories (trainer progress discovery).
    #[arg(long, default_value = default_metrics_root_str())]
    pub(super) metrics_root: PathBuf,
    /// OPT-IN: corpus sqlite for point lookups (default: never opened).
    #[arg(long)]
    pub(super) gameplay_db: Option<PathBuf>,
    /// OPT-IN: backlog sqlite for claims-by-state (default: never opened).
    #[arg(long)]
    pub(super) claims_db: Option<PathBuf>,
    /// One-shot: run the real collectors once, print the DL chart series plus
    /// the dataset funnel as JSON, and exit. No terminal required — this is how
    /// the plots are verified unattended.
    #[arg(long)]
    pub(super) dump_dl_json: bool,
    /// One-shot compact unified flywheel + budget-campaign status for monitors.
    #[arg(long)]
    pub(super) dump_unified_json: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Tab {
    #[default]
    Overview,
    GpuDl,
    Gameplay,
    Dl,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct UiState {
    pub(super) tab: Tab,
    /// Which training arm the DL charts follow. Parallel tournament arms run one
    /// per GPU, so this cycles rather than pinning to the first job found. Held
    /// as an index and wrapped at render time against the live trainer count, so
    /// arms appearing or exiting can never leave it dangling.
    pub(super) arm: usize,
}

#[derive(Clone, Debug, Default)]
pub(super) struct RawSources {
    pub(super) scq: Option<Value>,
    pub(super) status: Option<Value>,
    pub(super) queue: Option<Value>,
    pub(super) freshness: Option<Value>,
    pub(super) supply: Option<Value>,
    pub(super) supply_breaker: Option<Value>,
    pub(super) trainer_controller: Option<Value>,
    pub(super) proposer_state: Option<Value>,
    pub(super) gpu_admission: Option<Value>,
    pub(super) utilization: Option<Value>,
    pub(super) campaign: Option<Value>,
    pub(super) proposal_summary: Option<Value>,
    pub(super) trainer_fallback: Option<Value>,
    pub(super) flywheel: Option<Value>,
    pub(super) accepted_by_task: Option<Value>,
    pub(super) four_task_telemetry: Option<JopeFourTaskTelemetryV1>,
    pub(super) evidence_bytes: Option<u64>,
    pub(super) local_events: Vec<Value>,
    pub(super) source_states: Vec<SourceState>,
    pub(super) collected_at_ms: u128,
    pub(super) collection_latency_ms: u128,
}

#[derive(Clone, Debug)]
pub(super) struct SourceState {
    pub(super) name: String,
    pub(super) healthy: bool,
    pub(super) note: String,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ClusterSummary {
    pub(super) cpu_busy_pct: f64,
    pub(super) psi_cpu10: f64,
    pub(super) psi_mem10: f64,
    pub(super) psi_io10: f64,
    pub(super) physical_cores: u64,
    pub(super) effective_cores: u64,
    pub(super) slot_target: u64,
    pub(super) running_slots: u64,
    pub(super) memory_available_gb: u64,
    pub(super) nodes_up: usize,
    pub(super) nodes_total: usize,
    pub(super) jobs_failed: u64,
    pub(super) throughput_per_minute: u64,
}

#[derive(Clone, Debug, Default)]
pub(super) struct NodeRow {
    pub(super) node: String,
    pub(super) state: String,
    pub(super) health: String,
    pub(super) physical_cores: u64,
    pub(super) effective_cores: u64,
    pub(super) multiplier: f64,
    pub(super) slot_target: u64,
    pub(super) running_slots: u64,
    pub(super) cpu_busy_pct: Option<f64>,
    pub(super) psi_cpu10: Option<f64>,
    pub(super) psi_mem10: Option<f64>,
    pub(super) psi_io10: Option<f64>,
    pub(super) memory_available_gb: u64,
    pub(super) memory_total_gb: u64,
    pub(super) disk_use_pct: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct FamilyRow {
    pub(super) family: String,
    pub(super) weight: f64,
    pub(super) queued: u64,
    pub(super) starting: u64,
    pub(super) dispatching: u64,
    pub(super) running: u64,
    pub(super) held: u64,
    pub(super) oldest_queue_ts: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct QueueSummary {
    pub(super) queued: u64,
    pub(super) starting: u64,
    pub(super) dispatching: u64,
    pub(super) running: u64,
    pub(super) held: u64,
    pub(super) useful_depth: u64,
    pub(super) target: u64,
    pub(super) hard_cap: u64,
    pub(super) maximum_age_seconds: f64,
    pub(super) epoch: String,
    pub(super) stale_epochs: u64,
    pub(super) stale_held: u64,
    pub(super) admission_allowed: Option<bool>,
    pub(super) mode: String,
}

#[derive(Clone, Debug, Default)]
pub(super) struct LearningSummary {
    pub(super) checkpoint: String,
    pub(super) trainer: String,
    pub(super) trainer_state: String,
    pub(super) labels: u64,
    pub(super) frozen_watermark: u64,
    pub(super) internal_promotion: String,
    pub(super) milestone_promotion: String,
    pub(super) proposal_regression: u64,
    pub(super) proposal_binary: u64,
    pub(super) proposal_validity: Option<f64>,
    pub(super) consecutive_rejections: u64,
}

#[derive(Clone, Debug, Default)]
pub(super) struct CampaignSummary {
    pub(super) stage: String,
    pub(super) gate_state: String,
    pub(super) robust_wins: u64,
    pub(super) robust_target: u64,
    pub(super) robust_denominator: u64,
    pub(super) trajectories_complete: u64,
    pub(super) trajectories_total: u64,
    pub(super) evaluations_used: u64,
    pub(super) evaluations_cap: u64,
    pub(super) current_rung: String,
    pub(super) seed_agreement: String,
    pub(super) active_jobs: u64,
    pub(super) license_track: String,
    pub(super) test_opening_state: String,
    pub(super) last_advanced_at_ms: u128,
    pub(super) blockers: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct WaveSummary {
    pub(super) active_wave: String,
    pub(super) admission_waves: usize,
    pub(super) unclaimed_manifests: u64,
    pub(super) ready_manifests: u64,
    pub(super) awaiting_keeper_ack: String,
    pub(super) adaptive_regression: u64,
    pub(super) adaptive_binary: u64,
    pub(super) sealed_isolated: Option<bool>,
    pub(super) sealed_scored: u64,
    pub(super) corpus_rows: u64,
    pub(super) strict_wins: u64,
    pub(super) pair_count: u64,
    pub(super) duplicate_pairs_prevented: u64,
    pub(super) evidence_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub(super) struct GpuRow {
    pub(super) node: String,
    pub(super) model_hint: String,
    pub(super) status: String,
    pub(super) owner_job: Option<String>,
    pub(super) role: String,
    pub(super) attempt: u64,
    pub(super) admission_reason: String,
    pub(super) util_pct: f64,
    pub(super) mem_used_mb: u64,
    pub(super) mem_total_mb: u64,
    pub(super) temp_c: u64,
    pub(super) power_w: f64,
    pub(super) power_limit_w: f64,
    pub(super) throttled: bool,
}

#[derive(Clone, Debug, Default)]
pub(super) struct TrainerMetricPoint {
    pub(super) epoch: u64,
    pub(super) train_loss: f64,
    pub(super) val_loss: f64,
    pub(super) bad_passes: u64,
    pub(super) overfit_gap: f64,
    pub(super) val_pairwise: Option<f64>,
    pub(super) val_binary_pairwise: Option<f64>,
    pub(super) val_regression_pairwise: Option<f64>,
    pub(super) val_reward_cv: Option<f64>,
    pub(super) val_top1_regret: Option<f64>,
    /// Conditional model minus the genome-only ablation control. This is the
    /// anti-collapse yardstick: at or below zero the policy is not using the
    /// dataset embedding at all, however good its raw pairwise number looks.
    pub(super) val_conditional_advantage: Option<f64>,
    pub(super) val_ablation_pairwise: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct TrainerDetail {
    pub(super) job_id: String,
    pub(super) node: String,
    pub(super) is_trainer: bool,
    pub(super) phase: String,
    pub(super) epoch: u64,
    pub(super) epochs: u64,
    pub(super) batch: u64,
    pub(super) total_batches: u64,
    pub(super) pct: f64,
    pub(super) running_loss: Option<f64>,
    pub(super) val_loss: Option<f64>,
    pub(super) overfit_gap: Option<f64>,
    pub(super) bad_passes: u64,
    pub(super) patience: u64,
    pub(super) early_stopped: bool,
    pub(super) label_watermark: u64,
    pub(super) elapsed_sec: f64,
    pub(super) validation_batch: u64,
    pub(super) validation_batches: u64,
    pub(super) progress_at_ms: u128,
    pub(super) history: Vec<TrainerMetricPoint>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Origin {
    DlProposal,
    GpOffspring,
    LocalElite,
    CommonWinner,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EventKind {
    Eval,
    StrictWin,
}

#[derive(Clone, Debug)]
pub(super) struct GameplayEvent {
    pub(super) at_ms: u128,
    pub(super) task: String,
    pub(super) dataset_hash: String,
    pub(super) score: Option<f64>,
    /// Present only when a fresh typed incumbent authority supplies the exact
    /// task/context/checkpoint identity. Legacy champion indexes are forbidden.
    pub(super) delta_vs_incumbent: Option<f64>,
    pub(super) origin: Origin,
    pub(super) node: String,
    pub(super) wave: String,
    pub(super) valid: bool,
    pub(super) kind: EventKind,
}

#[derive(Clone, Debug, Default)]
pub(super) struct GameplayStats {
    pub(super) manifests_seen: u64,
    pub(super) dl_top1: u64,
    pub(super) gp_top1: u64,
    pub(super) dl_slot_share_pct: f64,
}

#[derive(Clone, Debug, Default)]
pub(super) struct WaveProgress {
    pub(super) wave: String,
    pub(super) manifest_count: u64,
    pub(super) datasets: u64,
    pub(super) receipts_landed: u64,
    pub(super) claims_by_state: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct TrainerWatchdogView {
    pub(super) labels: u64,
    pub(super) snapshot_watermark: u64,
    pub(super) last_evidence_watermark: u64,
    pub(super) next_label_threshold: u64,
    pub(super) pending_unfrozen_labels: u64,
    pub(super) checkpoint_sha256: String,
    pub(super) action: String,
    pub(super) active_trainers: Vec<String>,
    pub(super) generated_at_ms: u128,
}

#[derive(Clone, Debug, Default)]
pub(super) struct AcceptedTaskFront {
    pub(super) task: String,
    pub(super) checkpoint_sha256: String,
    pub(super) generation: u64,
    pub(super) updated_at_ms: u128,
}

/// One task's dataset-unlock pipeline:
/// registry → encoding (jobs queued/running) → queued (encoded, awaiting
/// activation) → active in catalog → in play (active with evidence).
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct FunnelRow {
    pub(super) task: String,
    pub(super) registry: u64,
    pub(super) encoded: u64,
    pub(super) active: u64,
    pub(super) in_play: u64,
    pub(super) queued: u64,
    pub(super) encoding: u64,
    pub(super) sealed: u64,
    pub(super) delta_active_1h: i64,
    pub(super) delta_in_play_1h: i64,
}

impl FunnelRow {
    pub(super) fn locked(&self) -> u64 {
        self.registry
            .saturating_sub(self.active)
            .saturating_sub(self.queued)
            .saturating_sub(self.encoding)
            .saturating_sub(self.sealed)
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct UnlockFunnel {
    pub(super) rows: Vec<FunnelRow>,
    pub(super) source: String,
    pub(super) fetched_at_ms: u128,
}

/// One checkpoint generation from ops/seeding-curve.v1.json: the DL-share
/// column of the tab-4 generations panel.
#[derive(Clone, Debug, Default)]
pub(super) struct SeedingPoint {
    pub(super) checkpoint_short: String,
    pub(super) dl_win_share_pct: Option<f64>,
    pub(super) dl_obs_share_pct: Option<f64>,
    pub(super) strict_wins_total: u64,
    pub(super) observations_caught_up: bool,
}

/// Seeding-curve slow source (30s gate; the file itself is refreshed by the
/// ingestion reconcile loop on a minutes cadence).
#[derive(Clone, Debug, Default)]
pub(super) struct SeedingCurveView {
    pub(super) points: Vec<SeedingPoint>,
    pub(super) fetched_at_ms: u128,
}

/// One trainer generation's per-epoch metrics, recovered from its run dir.
/// Identity is the RUN DIRECTORY name, which outlives the SCQ job, so this
/// survives cancel/resubmit/rename churn.
#[derive(Clone, Debug, Default)]
pub(super) struct RunSeries {
    /// Run-dir name minus the `jope-selfplay-train-` prefix (`001805437-a-g8`).
    pub(super) run_label: String,
    /// Compact generation tag for the ruler (`a-g8`).
    pub(super) short_label: String,
    pub(super) points: Vec<TrainerMetricPoint>,
    /// True while an SCQ job with this run-dir name is still running.
    pub(super) live: bool,
    /// train_metrics.jsonl mtime — orders generations and gates re-reads.
    pub(super) updated_at_ms: u128,
}

/// Cross-generation DL training history: the operator's "loss constantly going
/// down" curve. Persisted implicitly by the run dirs themselves.
#[derive(Clone, Debug, Default)]
pub(super) struct DlHistory {
    pub(super) runs: Vec<RunSeries>,
    pub(super) fetched_at_ms: u128,
}

/// Flattened chart-ready form of `DlHistory` on one continuous epoch axis.
#[derive(Clone, Debug, Default)]
pub(super) struct DlSeries {
    pub(super) train: Vec<(f64, f64)>,
    pub(super) val: Vec<(f64, f64)>,
    pub(super) pairwise: Vec<(f64, f64)>,
    pub(super) ablation: Vec<(f64, f64)>,
    /// (x where the generation starts, short label, live) — the boundaries.
    pub(super) marks: Vec<(f64, String, bool)>,
    pub(super) total_epochs: f64,
}

/// Immutable snapshot of the slow-source cache, handed to build_dashboard.
#[derive(Clone, Debug, Default)]
pub(super) struct SlowView {
    pub(super) gpus: Vec<GpuRow>,
    pub(super) trainers: Vec<TrainerDetail>,
    pub(super) watchdog: Option<TrainerWatchdogView>,
    pub(super) gameplay: Vec<GameplayEvent>,
    pub(super) gameplay_stats: GameplayStats,
    pub(super) wave_progress: Option<WaveProgress>,
    pub(super) health_alarms: Vec<String>,
    pub(super) daemon_ages: Vec<(String, u64, u64)>,
    pub(super) generations: Vec<(String, Option<f64>)>,
    pub(super) seeding: SeedingCurveView,
    pub(super) dl_history: DlHistory,
    pub(super) funnel: UnlockFunnel,
    pub(super) feed_fetched_at_ms: u128,
    pub(super) newest_receipt_at_ms: u128,
    pub(super) source_states: Vec<SourceState>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Dashboard {
    pub(super) collected_at_ms: u128,
    pub(super) collection_latency_ms: u128,
    pub(super) scq_epoch: Option<u64>,
    pub(super) scq_ts: String,
    pub(super) cluster: ClusterSummary,
    pub(super) nodes: Vec<NodeRow>,
    pub(super) families: Vec<FamilyRow>,
    pub(super) queue: QueueSummary,
    pub(super) top_state: String,
    pub(super) why_idle: String,
    pub(super) campaign: CampaignSummary,
    pub(super) supply_circuit_open: bool,
    pub(super) supply_circuit_error: String,
    pub(super) foreign_cancellations: u64,
    pub(super) stranded_trainer: bool,
    pub(super) stranded_proposer: bool,
    pub(super) priority_drift: bool,
    pub(super) stale_critical_source: bool,
    pub(super) learning: LearningSummary,
    pub(super) wave: WaveSummary,
    pub(super) gpus: Vec<GpuRow>,
    pub(super) trainers: Vec<TrainerDetail>,
    pub(super) watchdog: Option<TrainerWatchdogView>,
    pub(super) accepted_tasks: Vec<AcceptedTaskFront>,
    pub(super) four_task_telemetry: Option<JopeFourTaskTelemetryV1>,
    pub(super) gameplay: Vec<GameplayEvent>,
    pub(super) gameplay_stats: GameplayStats,
    pub(super) wave_progress: Option<WaveProgress>,
    pub(super) daemon_ages: Vec<(String, u64, u64)>,
    pub(super) generations: Vec<(String, Option<f64>)>,
    pub(super) seeding: SeedingCurveView,
    pub(super) dl_history: DlHistory,
    pub(super) funnel: UnlockFunnel,
    pub(super) feed_fetched_at_ms: u128,
    pub(super) alerts: Vec<String>,
    pub(super) failures: Vec<String>,
    pub(super) events: Vec<String>,
    pub(super) sources_healthy: usize,
    pub(super) sources_total: usize,
}
