use super::*;

#[derive(Clone, Debug)]
pub(super) struct LiveBest {
    semantic_sha256: String,
    pub(super) complexity: usize,
    pub(super) robust_fitness: f64,
    pooled_oof_r2: f64,
    pub(super) pooled_oof_rmse: f64,
}

struct LiveState {
    file: File,
    human_file: File,
    started: Instant,
    completed_slots: usize,
    valid_slots: usize,
    invalid_slots: usize,
    cache_hits: usize,
    cv7_completed: usize,
    generation_completed: Vec<usize>,
    best: Option<LiveBest>,
    cv7_best: Option<LiveBest>,
    generation_best: Vec<Option<LiveBest>>,
}

pub(super) struct LiveLogger {
    total_slots: usize,
    population: usize,
    json_path: PathBuf,
    human_path: PathBuf,
    state: Mutex<LiveState>,
}

impl LiveLogger {
    pub(super) fn create(
        root: &Path,
        total_slots: usize,
        population: usize,
        rayon_threads: usize,
    ) -> Result<Self> {
        let path = root.join("live-progress.v1.jsonl");
        let file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("create live progress log {}", path.display()))?;
        let human_path = root.join("live-progress.log");
        let mut human_file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&human_path)
            .with_context(|| format!("create human live progress log {}", human_path.display()))?;
        writeln!(
            human_file,
            "# JAIN EXACT-CV LIVE | runtime capacity {rayon_threads} threads | {population} dedicated single-threaded member workers | {total_slots} quick slots | fixed CV5 fold {QUICK_CV5_HOLDOUT_FOLD} | top {FINAL_CV7_CANDIDATES} advance to full CV7"
        )?;
        writeln!(
            human_file,
            "# TabFM RMSE reference {TABFM_RMSE:.14}; all AHEAD/BEHIND labels are CV proxies until phase-two test scoring"
        )?;
        human_file.flush()?;
        human_file.sync_data()?;
        let generations = total_slots.div_ceil(population.max(1));
        Ok(Self {
            total_slots,
            population,
            json_path: path,
            human_path,
            state: Mutex::new(LiveState {
                file,
                human_file,
                started: Instant::now(),
                completed_slots: 0,
                valid_slots: 0,
                invalid_slots: 0,
                cache_hits: 0,
                cv7_completed: 0,
                generation_completed: vec![0; generations],
                best: None,
                cv7_best: None,
                generation_best: vec![None; generations],
            }),
        })
    }

    pub(super) fn record_slot(
        &self,
        generation: usize,
        slot: usize,
        member: &PopulationMember,
        evaluation: &CachedEvaluation,
        cache_miss: bool,
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live progress logger lock poisoned"))?;
        state.completed_slots += 1;
        state.generation_completed[generation] += 1;
        if !cache_miss {
            state.cache_hits += 1;
        }
        if let Some(result) = evaluation.result.as_ref() {
            state.valid_slots += 1;
            let proposed = LiveBest {
                semantic_sha256: member.semantic_sha256.clone(),
                complexity: evaluation.complexity,
                robust_fitness: result.robust_fitness,
                pooled_oof_r2: result.pooled_oof_r2,
                pooled_oof_rmse: result.pooled_oof_rmse,
            };
            if state
                .best
                .as_ref()
                .is_none_or(|current| live_best_cmp(&proposed, current).is_lt())
            {
                state.best = Some(proposed.clone());
            }
            if state.generation_best[generation]
                .as_ref()
                .is_none_or(|current| live_best_cmp(&proposed, current).is_lt())
            {
                state.generation_best[generation] = Some(proposed);
            }
        } else {
            state.invalid_slots += 1;
        }
        let best = state.best.clone();
        let generation_best = state.generation_best[generation].clone();
        let elapsed_seconds = state.started.elapsed().as_secs_f64();
        let throughput = state.completed_slots as f64 / elapsed_seconds.max(f64::MIN_POSITIVE);
        let eta_seconds = self.total_slots.saturating_sub(state.completed_slots) as f64
            / throughput.max(f64::MIN_POSITIVE);
        let progress_percent =
            100.0 * state.completed_slots as f64 / self.total_slots.max(1) as f64;
        let event = serde_json::json!({
            "schema": "JainExactCvLiveProgress.v1",
            "event": "slot_terminal",
            "scoring_phase": "QUICK_CV5_FOLD0",
            "cv5_holdout_fold": QUICK_CV5_HOLDOUT_FOLD,
            "generation": generation,
            "slot": slot,
            "completed_slots": state.completed_slots,
            "total_slots": self.total_slots,
            "generation_completed_slots": state.generation_completed[generation],
            "generation_total_slots": self.population,
            "progress_percent": progress_percent,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            "throughput_slots_per_second": throughput,
            "cumulative_valid_slots": state.valid_slots,
            "cumulative_invalid_slots": state.invalid_slots,
            "cumulative_cache_hits": state.cache_hits,
            "semantic_sha256": member.semantic_sha256,
            "cache_disposition": if cache_miss { "miss" } else { "hit" },
            "valid": evaluation.result.is_some(),
            "rejection_reason": evaluation.rejection_reason,
            "robust_fitness": evaluation.result.as_ref().map(|result| result.robust_fitness),
            "pooled_oof_r2": evaluation.result.as_ref().map(|result| result.pooled_oof_r2),
            "pooled_oof_rmse": evaluation.result.as_ref().map(|result| result.pooled_oof_rmse),
            "incumbent_semantic_sha256": best.as_ref().map(|value| value.semantic_sha256.as_str()),
            "incumbent_robust_fitness": best.as_ref().map(|value| value.robust_fitness),
            "incumbent_pooled_oof_r2": best.as_ref().map(|value| value.pooled_oof_r2),
            "incumbent_pooled_oof_rmse": best.as_ref().map(|value| value.pooled_oof_rmse),
            "generation_best_robust_fitness": generation_best.as_ref().map(|value| value.robust_fitness),
            "generation_best_pooled_oof_rmse": generation_best.as_ref().map(|value| value.pooled_oof_rmse),
            "tabfm_test_rmse_reference": TABFM_RMSE,
            "cv_rmse_signed_margin_vs_tabfm": best.as_ref().map(|value| TABFM_RMSE - value.pooled_oof_rmse),
            "cv_proxy_track_status": best.as_ref().map(|value| if value.pooled_oof_rmse < TABFM_RMSE { "CV_PROXY_AHEAD" } else { "CV_PROXY_BEHIND" }),
            "track_status_is_final_test_authority": false,
        });
        state.file.write_all(&canonical_json(&event)?)?;
        state.file.flush()?;
        state.file.sync_data()?;
        let status = best.as_ref().map_or("NO_VALID_INCUMBENT", |value| {
            if value.pooled_oof_rmse < TABFM_RMSE {
                "CV_PROXY_AHEAD"
            } else {
                "CV_PROXY_BEHIND"
            }
        });
        let global_fitness = best.as_ref().map_or_else(
            || "NA".to_string(),
            |value| format!("{:.12}", value.robust_fitness),
        );
        let global_rmse = best.as_ref().map_or_else(
            || "NA".to_string(),
            |value| format!("{:.12}", value.pooled_oof_rmse),
        );
        let tabfm_margin = best.as_ref().map_or_else(
            || "NA".to_string(),
            |value| format!("{:+.12}", TABFM_RMSE - value.pooled_oof_rmse),
        );
        let human_line = format!(
            "[{elapsed_seconds:>8.1}s] QUICK g={generation:02} {:02}/{:02} | overall={:03}/{:03} ({progress_percent:5.1}%) | member_fit={} | GLOBAL_BEST fitness={} rmse={} | TabFM_proxy_margin={} {} | valid={} invalid={} cache_hits={} | {:.3} slots/s ETA={:.1}s",
            state.generation_completed[generation],
            self.population,
            state.completed_slots,
            self.total_slots,
            evaluation
                .result
                .as_ref()
                .map_or_else(|| "NA".to_string(), |result| format!("{:.12}", result.robust_fitness)),
            global_fitness,
            global_rmse,
            tabfm_margin,
            status,
            state.valid_slots,
            state.invalid_slots,
            state.cache_hits,
            throughput,
            eta_seconds,
        );
        writeln!(state.human_file, "{human_line}")?;
        state.human_file.flush()?;
        state.human_file.sync_data()?;
        eprintln!("{human_line}");
        Ok(())
    }

    fn record_cv7_finalist(&self, finalist: &FinalistCv7) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live progress logger lock poisoned"))?;
        state.cv7_completed += 1;
        if let Some(result) = finalist.cv7.as_ref() {
            let proposed = LiveBest {
                semantic_sha256: finalist.semantic_sha256.clone(),
                complexity: finalist.complexity,
                robust_fitness: result.robust_fitness,
                pooled_oof_r2: result.pooled_oof_r2,
                pooled_oof_rmse: result.pooled_oof_rmse,
            };
            if state
                .cv7_best
                .as_ref()
                .is_none_or(|current| live_best_cmp(&proposed, current).is_lt())
            {
                state.cv7_best = Some(proposed);
            }
        }
        let cv7_best = state.cv7_best.clone();
        let elapsed_seconds = state.started.elapsed().as_secs_f64();
        let event = serde_json::json!({
            "schema": "JainExactCvLiveProgress.v1",
            "event": "finalist_full_cv7_terminal",
            "scoring_phase": "FULL_CV7_FINALIST",
            "quick_rank": finalist.quick_rank,
            "completed_finalists": state.cv7_completed,
            "total_finalists": FINAL_CV7_CANDIDATES,
            "semantic_sha256": finalist.semantic_sha256,
            "valid": finalist.cv7.is_some(),
            "rejection_reason": finalist.rejection_reason,
            "cv7_robust_fitness": finalist.cv7.as_ref().map(|value| value.robust_fitness),
            "cv7_pooled_oof_r2": finalist.cv7.as_ref().map(|value| value.pooled_oof_r2),
            "cv7_pooled_oof_rmse": finalist.cv7.as_ref().map(|value| value.pooled_oof_rmse),
            "tabfm_test_rmse_reference": TABFM_RMSE,
            "cv7_rmse_signed_margin_vs_tabfm": finalist.cv7.as_ref().map(|value| TABFM_RMSE - value.pooled_oof_rmse),
            "cv7_incumbent_semantic_sha256": cv7_best.as_ref().map(|value| value.semantic_sha256.as_str()),
            "cv7_incumbent_robust_fitness": cv7_best.as_ref().map(|value| value.robust_fitness),
            "cv7_incumbent_pooled_oof_rmse": cv7_best.as_ref().map(|value| value.pooled_oof_rmse),
            "cv7_incumbent_rmse_signed_margin_vs_tabfm": cv7_best.as_ref().map(|value| TABFM_RMSE - value.pooled_oof_rmse),
            "track_status_is_final_test_authority": false,
        });
        state.file.write_all(&canonical_json(&event)?)?;
        state.file.flush()?;
        state.file.sync_data()?;
        let status = cv7_best.as_ref().map_or("NO_VALID_CV7", |value| {
            if value.pooled_oof_rmse < TABFM_RMSE {
                "CV7_PROXY_AHEAD"
            } else {
                "CV7_PROXY_BEHIND"
            }
        });
        let human_line = format!(
            "[{elapsed_seconds:>8.1}s] FULL_CV7 {}/{} quick_rank={:02} | member_fit={} member_rmse={} | CV7_GLOBAL_BEST fitness={} rmse={} | TabFM_proxy_margin={} {}",
            state.cv7_completed,
            FINAL_CV7_CANDIDATES,
            finalist.quick_rank,
            finalist.cv7.as_ref().map_or_else(
                || "NA".to_string(),
                |value| format!("{:.12}", value.robust_fitness),
            ),
            finalist.cv7.as_ref().map_or_else(
                || "NA".to_string(),
                |value| format!("{:.12}", value.pooled_oof_rmse),
            ),
            cv7_best.as_ref().map_or_else(
                || "NA".to_string(),
                |value| format!("{:.12}", value.robust_fitness),
            ),
            cv7_best.as_ref().map_or_else(
                || "NA".to_string(),
                |value| format!("{:.12}", value.pooled_oof_rmse),
            ),
            cv7_best.as_ref().map_or_else(
                || "NA".to_string(),
                |value| format!("{:+.12}", TABFM_RMSE - value.pooled_oof_rmse),
            ),
            status,
        );
        writeln!(state.human_file, "{human_line}")?;
        state.human_file.flush()?;
        state.human_file.sync_data()?;
        eprintln!("{human_line}");
        Ok(())
    }

    pub(super) fn record_campaign_complete(&self, selection: &SelectionFreeze) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live progress logger lock poisoned"))?;
        let elapsed_seconds = state.started.elapsed().as_secs_f64();
        let event = serde_json::json!({
            "schema": "JainExactCvLiveProgress.v1",
            "event": "campaign_complete",
            "elapsed_seconds": elapsed_seconds,
            "terminal_slots": state.completed_slots,
            "completed_finalists": state.cv7_completed,
            "selected_semantic_sha256": selection.semantic_sha256,
            "selected_cv7_robust_fitness": selection.cv7.robust_fitness,
            "selected_cv7_pooled_oof_rmse": selection.cv7.pooled_oof_rmse,
            "selected_cv7_rmse_signed_margin_vs_tabfm": TABFM_RMSE - selection.cv7.pooled_oof_rmse,
            "track_status_is_final_test_authority": false,
        });
        state.file.write_all(&canonical_json(&event)?)?;
        state.file.flush()?;
        state.file.sync_data()?;
        let human_line = format!(
            "[{elapsed_seconds:>8.1}s] CAMPAIGN_COMPLETE slots={}/{} cv7={}/{} | WINNER={} | CV7 fitness={:.12} rmse={:.12} | TabFM_proxy_margin={:+.12} {} | FINAL_TEST_PENDING",
            state.completed_slots,
            self.total_slots,
            state.cv7_completed,
            FINAL_CV7_CANDIDATES,
            selection.semantic_sha256,
            selection.cv7.robust_fitness,
            selection.cv7.pooled_oof_rmse,
            TABFM_RMSE - selection.cv7.pooled_oof_rmse,
            if selection.cv7.pooled_oof_rmse < TABFM_RMSE {
                "CV7_PROXY_AHEAD"
            } else {
                "CV7_PROXY_BEHIND"
            },
        );
        writeln!(state.human_file, "{human_line}")?;
        state.human_file.flush()?;
        state.human_file.sync_data()?;
        eprintln!("{human_line}");
        Ok(())
    }

    pub(super) fn seal(&self) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("live progress logger lock poisoned"))?;
        state.file.flush()?;
        state.file.sync_all()?;
        state.human_file.flush()?;
        state.human_file.sync_all()?;
        make_readonly(&self.json_path)?;
        make_readonly(&self.human_path)
    }
}
