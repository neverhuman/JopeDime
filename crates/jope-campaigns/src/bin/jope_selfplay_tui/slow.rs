use super::*;

pub(super) struct IntervalGate {
    pub(super) last: Option<Instant>,
    pub(super) every: Duration,
}

impl IntervalGate {
    pub(super) fn new(every: Duration) -> Self {
        Self { last: None, every }
    }

    pub(super) fn due(&mut self) -> bool {
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
pub(super) type FunnelHistory = VecDeque<(u128, Vec<(String, u64, u64)>)>;

pub(super) struct SlowState {
    pub(super) gate_progress: IntervalGate,
    pub(super) gate_metrics: IntervalGate,
    pub(super) gate_receipts: IntervalGate,
    pub(super) gate_wave: IntervalGate,
    pub(super) gate_claims: IntervalGate,
    pub(super) known_receipt_dirs: HashSet<String>,
    pub(super) receipts_prefilled: bool,
    pub(super) wave_manifest_paths: HashMap<String, HashMap<String, String>>,
    pub(super) manifest_semantics: HashMap<String, HashMap<String, (String, String)>>,
    pub(super) manifest_order: VecDeque<String>,
    pub(super) ring: VecDeque<GameplayEvent>,
    pub(super) stats_window: VecDeque<(Origin, bool)>,
    pub(super) manifests_seen: u64,
    pub(super) dl_slots: u64,
    pub(super) total_slots: u64,
    pub(super) newest_wave_receipts: u64,
    pub(super) newest_wave_name: String,
    /// job id -> (attempt the path came from, resolved train_progress.json).
    pub(super) progress_paths: HashMap<String, (u64, PathBuf)>,
    /// job id -> the train_metrics.jsonl the TRAINER STATE history was last
    /// read from, so a requeue re-reads even when the mtimes collide.
    pub(super) metrics_stamps: HashMap<String, MetricsStamp>,
    pub(super) gate_generations: IntervalGate,
    pub(super) generations: Vec<(String, Option<f64>)>,
    pub(super) gate_dl_history: IntervalGate,
    pub(super) dl_history: DlHistory,
    /// run-dir name -> (attempt the path came from, resolved
    /// train_metrics.jsonl). Skips the tree walk without surviving a requeue.
    pub(super) dl_metrics_paths: HashMap<String, (u64, PathBuf)>,
    /// run-dir name -> (metrics file read, parsed points) so a steady-state
    /// pass only re-reads the generation that is actually still writing.
    pub(super) dl_points_cache: HashMap<String, (MetricsStamp, Vec<TrainerMetricPoint>)>,
    pub(super) gate_seeding: IntervalGate,
    pub(super) seeding: SeedingCurveView,
    pub(super) daemon_ages: Vec<(String, u64, u64)>,
    pub(super) trainers: Vec<TrainerDetail>,
    pub(super) gpus: Vec<GpuRow>,
    pub(super) watchdog: Option<TrainerWatchdogView>,
    pub(super) wave_progress: Option<WaveProgress>,
    pub(super) health_alarms: Vec<String>,
    pub(super) claims_backoff_until: Option<Instant>,
    pub(super) claims_backoff: Duration,
    pub(super) gate_funnel: IntervalGate,
    pub(super) funnel: UnlockFunnel,
    pub(super) funnel_history: FunnelHistory,
    pub(super) feed_fetched_at_ms: u128,
    pub(super) newest_receipt_at_ms: u128,
    pub(super) source_states: Vec<SourceState>,
}

impl SlowState {
    pub(super) fn new() -> Self {
        Self {
            gate_progress: IntervalGate::new(Duration::from_secs(1)),
            gate_metrics: IntervalGate::new(Duration::from_secs(5)),
            gate_receipts: IntervalGate::new(Duration::from_secs(2)),
            gate_wave: IntervalGate::new(Duration::from_secs(10)),
            gate_claims: IntervalGate::new(Duration::from_secs(15)),
            known_receipt_dirs: HashSet::new(),
            receipts_prefilled: false,
            wave_manifest_paths: HashMap::new(),
            manifest_semantics: HashMap::new(),
            manifest_order: VecDeque::new(),
            ring: VecDeque::new(),
            stats_window: VecDeque::new(),
            manifests_seen: 0,
            dl_slots: 0,
            total_slots: 0,
            newest_wave_receipts: 0,
            newest_wave_name: String::new(),
            progress_paths: HashMap::new(),
            metrics_stamps: HashMap::new(),
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

    pub(super) fn view(&self) -> SlowView {
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

pub(super) fn collect_slow_sources(args: &Args, raw: &RawSources, state: &mut SlowState) {
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
                    let stamp = MetricsStamp::of(&metrics_path).unwrap_or_else(|_| MetricsStamp {
                        path: metrics_path.clone(),
                        modified: UNIX_EPOCH,
                        len: 0,
                    });
                    let changed = state
                        .metrics_stamps
                        .get(&trainer.job_id)
                        .is_none_or(|prior| *prior != stamp);
                    if changed || trainer.history.is_empty() {
                        trainer.history = read_metrics_history(&metrics_path);
                        state.metrics_stamps.insert(trainer.job_id.clone(), stamp);
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

pub(super) fn discover_gpu_jobs(scq: &Value) -> Vec<(String, String, PathBuf, bool)> {
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
            let is_trainer = is_trainer_run_id(id);
            Some((id.to_owned(), node.to_owned(), run_dir, is_trainer))
        })
        .collect()
}

pub(super) fn parse_gpus(scq: &Value, jobs: &[(String, String, PathBuf, bool)]) -> Vec<GpuRow> {
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
                role: String::new(),
                attempt: 0,
                admission_reason: String::new(),
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

pub(super) fn refresh_trainers(
    args: &Args,
    jobs: &[(String, String, PathBuf, bool)],
    state: &mut SlowState,
) {
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
                // Only a path resolved from the newest attempt itself may be
                // cached under it; a fallback to an older attempt is re-resolved
                // next pass, once the new trainer has written its progress.
                if let (Some(path), Some(attempt)) = (&resolved, attempt) {
                    if attempt_of(&run_dir, path) == Some(attempt) {
                        state
                            .progress_paths
                            .insert(job_id.clone(), (attempt, path.clone()));
                    }
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
