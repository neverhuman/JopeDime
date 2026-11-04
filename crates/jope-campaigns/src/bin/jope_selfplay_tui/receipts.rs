use super::*;

pub(super) fn scan_receipts(root: &Path, state: &mut SlowState) {
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
        fresh.sort_by_key(|row| std::cmp::Reverse(row.2));
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

pub(super) fn ingest_receipt_dir(root: &Path, dir: &Path, state: &mut SlowState) {
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
                    delta_vs_incumbent: None,
                    origin,
                    node: string_at(&row, &["node"]).unwrap_or(&done_node).to_owned(),
                    wave,
                    valid,
                    kind: TuiEventKind::Eval,
                },
            );
        }
        if manifest_total > 0 {
            state.manifests_seen += 1;
            state.dl_slots += manifest_dl_slots;
            state.total_slots += manifest_total;
            if let Some((_top_score, origin, _task, _dataset)) = top {
                if state.stats_window.len() >= GAMEPLAY_STATS_WINDOW {
                    state.stats_window.pop_front();
                }
                // A typed task/context incumbent authority does not exist yet.
                // Preserve the feed, but never manufacture a comparison from
                // the legacy champion index.
                state.stats_window.push_back((origin, false));
            }
        }
    }
}
