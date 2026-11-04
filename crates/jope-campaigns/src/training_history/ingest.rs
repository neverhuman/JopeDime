use super::*;

mod residual;
pub use residual::*;

/// Re-derive `(rows, first_epoch, last_epoch)` from an archive file. Only used
/// on the crash-recovery path, so the full read (at most a few hundred KB) is
/// not on the steady-state cost path.
fn recount(path: &Path) -> (u64, Option<i64>, Option<i64>) {
    let Ok(text) = fs::read_to_string(path) else {
        return (0, None, None);
    };
    let mut rows = 0u64;
    let mut first = None;
    let mut last = None;
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        rows += 1;
        if let Some(epoch) = serde_json::from_str::<Value>(line)
            .ok()
            .and_then(|row| row.get("epoch").and_then(Value::as_i64))
        {
            first.get_or_insert(epoch);
            last = Some(epoch);
        }
    }
    (rows, first, last)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Scalar fields worth lifting out of a row into the index. Everything else
/// stays in the archived row itself; this is only a summary.
const SUMMARY_KEYS: &[&str] = &[
    "epoch",
    "phase",
    "train_loss",
    "val_loss",
    "overfit_gap",
    "lr",
    "batch_size",
    "epochs",
    "elapsed_sec",
    "train_rows",
    "validation_rows",
    "model_version",
    "device",
    "world_size",
    "val_conditional_advantage",
    "val_pairwise_win_accuracy",
    "macro_validation_best",
    "macro_validation_early_stopped",
    "val_fallback",
];

fn summarize_row(row: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    for key in SUMMARY_KEYS {
        if let Some(value) = row.get(*key) {
            if !value.is_null() {
                out.insert((*key).to_string(), value.clone());
            }
        }
    }
    // Recipe-defining flags are objects; record only whether they were active,
    // because a naive single curve across runs with and without the
    // anti-overfit experiment would be misleading.
    out.insert(
        "anti_overfit_experiment".into(),
        json!(row
            .get("anti_overfit_experiment")
            .map(|v| !v.is_null())
            .unwrap_or(false)),
    );
    out
}

/// Ingest everything the source has beyond the archived offset.
///
/// The whole ordering guarantee rests on this function: it returns
/// `IngestOutcome::covered() == true` only after the bytes are appended to the
/// archive file *and* fsynced, so a caller that prunes on `covered()` can never
/// delete rows that are not on durable storage.
pub fn ingest_source<R: SourceReader>(
    archive: &Path,
    key: &SegmentKey,
    reader: &R,
    source_path: &str,
    node: &str,
) -> Result<IngestOutcome> {
    if !key.is_sane() {
        anyhow::bail!("refusing unsafe segment key {key:?}");
    }
    let seg_dir = segment_dir(archive, key);
    fs::create_dir_all(&seg_dir)?;
    let metrics = SegmentState::metrics_path(&seg_dir);

    let mut state = match SegmentState::load(&seg_dir)? {
        Some(state) => state,
        None => SegmentState::initial(),
    };
    state.schema = SEGMENT_SCHEMA.to_string();
    state.run_id = key.run_id.clone();
    state.attempt = key.attempt.clone();
    state.watermark = key.watermark.clone();
    state.source_path = source_path.to_string();
    state.node = node.to_string();
    state.source_present = true;
    if state.first_seen_ms == 0 {
        state.first_seen_ms = now_ms();
    }

    // Trust the archive file on disk over the state file: if the state file was
    // written and the archive lost (or the process died between the two), the
    // file length is the real offset. Recount from the file rather than scaling
    // the offset alone, or the row/epoch counters would double-count on the
    // re-ingest that follows.
    let on_disk = fs::metadata(&metrics).map(|m| m.len()).unwrap_or(0);
    if on_disk < state.archived_bytes {
        let (rows, first_epoch, last_epoch) = recount(&metrics);
        state.archived_bytes = on_disk;
        state.archived_rows = rows;
        state.first_epoch = first_epoch;
        state.last_epoch = last_epoch;
        if on_disk == 0 {
            state.anchor_sha256.clear();
            state.anchor_len = 0;
        }
    }

    let source_bytes = reader.len()?;
    let mut outcome = IngestOutcome {
        source_bytes,
        ..Default::default()
    };

    // Divergence check. A source shorter than what we hold, or whose prefix no
    // longer hashes to our anchor, is a different stream reusing the same path
    // (a rerun). Rotate rather than truncate: nothing archived is ever lost.
    if state.archived_bytes > 0 {
        let mut rotate = source_bytes < state.archived_bytes;
        if !rotate && state.anchor_len > 0 {
            let head = reader.read_range(0, state.anchor_len as usize)?;
            rotate =
                head.len() as u64 != state.anchor_len || sha256_hex(&head) != state.anchor_sha256;
        }
        if rotate {
            let stamp = now_ms();
            let rewrites = seg_dir.join("rewrites");
            fs::create_dir_all(&rewrites)?;
            if metrics.exists() {
                fs::rename(&metrics, rewrites.join(format!("{stamp}.jsonl")))?;
            }
            if let Some(previous) = SegmentState::load(&seg_dir)? {
                let _ = write_atomic(
                    &rewrites.join(format!("{stamp}.segment.json")),
                    &serde_json::to_vec_pretty(&previous)?,
                );
            }
            state.rewrite_count += 1;
            state.archived_bytes = 0;
            state.archived_rows = 0;
            state.first_epoch = None;
            state.last_epoch = None;
            state.anchor_sha256.clear();
            state.anchor_len = 0;
            state.head = Map::new();
            state.tail = Map::new();
            outcome.rotated = true;
        }
    }

    if source_bytes > state.archived_bytes {
        let want = (source_bytes - state.archived_bytes) as usize;
        let delta = reader.read_range(state.archived_bytes, want)?;
        // Whole rows only: never archive a row the trainer is mid-write on.
        if let Some(index) = delta.iter().rposition(|b| *b == b'\n') {
            let complete = &delta[..=index];
            {
                let mut file = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&metrics)?;
                file.write_all(complete)?;
                // Durability before the caller is allowed to prune the source.
                file.sync_all()?;
            }
            state.archived_bytes += complete.len() as u64;
            outcome.appended_bytes = complete.len() as u64;

            for line in complete.split(|b| *b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                state.archived_rows += 1;
                outcome.appended_rows += 1;
                let Ok(row) = serde_json::from_slice::<Value>(line) else {
                    continue;
                };
                if let Some(epoch) = row.get("epoch").and_then(Value::as_i64) {
                    if state.first_epoch.is_none() {
                        state.first_epoch = Some(epoch);
                    }
                    state.last_epoch = Some(epoch);
                }
                if state.head.is_empty() {
                    state.head = summarize_row(&row);
                }
                state.tail = summarize_row(&row);
            }
            state.last_advance_ms = now_ms();
        }
    }

    // Refresh the anchor from the archive (not the source) so it always
    // describes bytes we actually hold.
    if state.archived_bytes > 0 && (state.anchor_len == 0 || state.anchor_len < ANCHOR_BYTES as u64)
    {
        let want = (state.archived_bytes as usize).min(ANCHOR_BYTES);
        let mut file = fs::File::open(&metrics)?;
        let mut head = vec![0u8; want];
        file.read_exact(&mut head)?;
        state.anchor_sha256 = sha256_hex(&head);
        state.anchor_len = want as u64;
    }

    state.last_check_ms = now_ms();
    outcome.archived_bytes = state.archived_bytes;
    outcome.archived_rows = state.archived_rows;
    outcome.last_epoch = state.last_epoch;
    // What is left is, by construction, the bytes after the last newline the
    // source holds: a row whose writer had not finished it.
    outcome.residual_bytes = source_bytes.saturating_sub(state.archived_bytes);
    state.save(&seg_dir)?;
    Ok(outcome)
}

/// Same as [`ingest_source`], additionally stamping the source fingerprint used
/// to skip untouched remote segments on the next pass, and the local timestamp
/// at which that fingerprint last changed.
pub fn ingest_source_fingerprinted<R: SourceReader>(
    archive: &Path,
    key: &SegmentKey,
    reader: &R,
    source_path: &str,
    node: &str,
    fingerprint: &str,
) -> Result<IngestOutcome> {
    let outcome = ingest_source(archive, key, reader, source_path, node)?;
    let seg_dir = segment_dir(archive, key);
    if let Some(mut state) = SegmentState::load(&seg_dir)? {
        if state.source_fingerprint != fingerprint || state.fingerprint_since_ms == 0 {
            state.source_fingerprint = fingerprint.to_string();
            state.fingerprint_since_ms = now_ms();
        }
        state.save(&seg_dir)?;
    }
    Ok(outcome)
}

/// Has a REMOTE source provably stopped changing?
///
/// The local rule compares the file's mtime against this clock, which is exact
/// because it is one clock. A remote mtime is a different machine's opinion of
/// the time, and writing off a trailing row early is the one way this policy
/// could cost a complete row (the collector could then prune a directory a
/// trainer is still writing to). So remote quiescence is measured as "the
/// size+mtime we observe have not changed for TAIL_QUIESCENT_SECS of OUR time",
/// which is immune to the other node's clock entirely.
pub fn remote_tail_is_abandoned(
    state: Option<&SegmentState>,
    fingerprint: &str,
    now: u64,
    quiescent_secs: u64,
) -> bool {
    let Some(state) = state else { return false };
    if state.source_fingerprint.is_empty() || state.source_fingerprint != fingerprint {
        return false;
    }
    state.fingerprint_since_ms > 0
        && now.saturating_sub(state.fingerprint_since_ms) / 1000 >= quiescent_secs
}

/// Mark a segment as having no live source any more (its run directory was
/// pruned). Purely informational; no archived byte is touched.
pub fn seal_segment(archive: &Path, key: &SegmentKey) -> Result<()> {
    let seg_dir = segment_dir(archive, key);
    let Some(mut state) = SegmentState::load(&seg_dir)? else {
        return Ok(());
    };
    if !state.source_present {
        return Ok(());
    }
    state.source_present = false;
    state.last_check_ms = now_ms();
    state.save(&seg_dir)
}

/// Copy a small sibling artifact (train-result.json) next to the archived rows.
/// Rewritten only when the bytes differ, so this is free on steady state.
pub fn archive_sidecar(archive: &Path, key: &SegmentKey, name: &str, bytes: &[u8]) -> Result<bool> {
    if !sane_component(name) {
        return Ok(false);
    }
    let target = segment_dir(archive, key).join(name);
    if let Ok(existing) = fs::read(&target) {
        if existing == bytes {
            return Ok(false);
        }
    }
    write_atomic(&target, bytes)?;
    Ok(true)
}
