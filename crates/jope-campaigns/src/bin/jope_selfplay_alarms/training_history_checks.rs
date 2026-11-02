use super::*;

pub(super) fn check_local_memory(alarms: &mut Alarms) {
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
pub(super) fn check_training_history(root: &Path, runs_root: &Path, alarms: &mut Alarms) {
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

    let states = match th::load_segments(&archive) {
        Ok(states) => states,
        Err(error) => {
            alarms.push(
                "training_history_segment_state_unreadable",
                "critical",
                format!("{error:#}"),
                "all archived segment state files are readable and valid".into(),
            );
            return;
        }
    };
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
pub(super) fn check_warm_start_guard(root: &Path, alarms: &mut Alarms) {
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
