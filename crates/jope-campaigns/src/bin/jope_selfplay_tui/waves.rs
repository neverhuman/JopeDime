use super::*;

pub(super) fn resolve_origin(
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

pub(super) fn slot_group_origin(slot_group: &str, source: &str) -> Origin {
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

pub(super) fn wave_manifest_path(
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

pub(super) fn refresh_wave_progress(root: &Path, state: &mut SlowState) {
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
    });
}

pub(super) fn refresh_claims(args: &Args, state: &mut SlowState) {
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

pub(super) fn query_claims(path: &Path) -> Result<BTreeMap<String, u64>> {
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
