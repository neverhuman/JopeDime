use super::*;

/// (status file, observed age seconds, expected cadence seconds) for the DL tab.
pub(super) fn read_daemon_ages(root: &Path) -> Vec<(String, u64, u64)> {
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
pub(super) fn generations_from_history(history: &DlHistory) -> Vec<(String, Option<f64>)> {
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
pub(super) fn short_generation_label(run_label: &str) -> String {
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
pub(super) fn collect_dl_history(
    metrics_root: &Path,
    live_runs: &HashSet<String>,
    state: &mut SlowState,
) -> DlHistory {
    let Ok(entries) = fs::read_dir(metrics_root) else {
        // Keep the last good history rather than blanking the panel.
        return state.dl_history.clone();
    };
    let mut found: Vec<(MetricsStamp, String)> = Vec::new();
    for entry in entries.filter_map(|entry| entry.ok()) {
        let run_id = entry.file_name().to_string_lossy().to_string();
        if !is_trainer_run_id(&run_id) {
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
                // Between SCQ creating `attempts/<n>/` and the requeued
                // trainer's first progress write, resolution falls back to
                // an older attempt. Caching that path under attempt n would
                // pin the chart to the dead attempt until attempt n+1, so it
                // is cached only when it came from the newest attempt.
                if let Some(attempt) = attempt {
                    if attempt_of(&entry.path(), &path) == Some(attempt) {
                        state
                            .dl_metrics_paths
                            .insert(run_id.clone(), (attempt, path.clone()));
                    }
                }
                path
            }
        };
        let Ok(stamp) = MetricsStamp::of(&metrics) else {
            continue;
        };
        found.push((stamp, run_id));
    }
    // Oldest → newest by last write, which for sequential trainer generations is
    // the order they ran in.
    found.sort_by_key(|(stamp, _)| stamp.modified);
    let skip = found.len().saturating_sub(DL_GENERATIONS_MAX);
    let kept = found.split_off(skip);
    // Cache retention must be keyed by the ORIGINAL run ids. The display label
    // trims the "jope-selfplay-train-" prefix, and re-prefixing the label only
    // round-trips for train-* runs: a shadow-* run's label re-prefixes to a key
    // that never existed, so its cache entries were evicted on every 250 ms
    // refresh -- forcing a full metrics re-read for exactly the runs that
    // dominate the chart now.
    let kept_ids = kept
        .iter()
        .map(|(_, run_id)| run_id.clone())
        .collect::<HashSet<_>>();
    let mut runs = Vec::with_capacity(kept.len());
    for (stamp, run_id) in kept {
        // Keyed on the resolved file, not just its mtime: after a requeue the
        // path moves to the new attempt even when both files share an mtime.
        let reuse = state
            .dl_points_cache
            .get(&run_id)
            .filter(|(cached, _)| *cached == stamp)
            .map(|(_, points)| points.clone());
        let modified = stamp.modified;
        let points = match reuse {
            Some(points) => points,
            None => {
                let points = read_metrics_points(&stamp.path, DL_RUN_POINTS_MAX);
                state
                    .dl_points_cache
                    .insert(run_id.clone(), (stamp, points.clone()));
                points
            }
        };
        // Trim whichever trainer prefix matches so shadow-* runs get labels as
        // short as train-* runs instead of their full 40+ character ids.
        let run_label = TRAINER_ID_PREFIXES
            .iter()
            .find_map(|prefix| run_id.strip_prefix(prefix).map(str::to_owned))
            .unwrap_or_else(|| run_id.clone());
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
    state.dl_points_cache.retain(|id, _| kept_ids.contains(id));
    state.dl_metrics_paths.retain(|id, _| kept_ids.contains(id));
    DlHistory {
        runs,
        fetched_at_ms: now_ms(),
    }
}

/// Flatten `DlHistory` onto one continuous epoch axis: generation *k* starts
/// where generation *k-1* ended, so warm-started runs read as one curve.
pub(super) fn continuous_series(history: &DlHistory) -> DlSeries {
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
