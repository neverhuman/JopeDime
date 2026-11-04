use super::*;

/// One capture pass' configuration.
pub(super) struct Cfg {
    pub(super) root: PathBuf,
    pub(super) runs_root: PathBuf,
    pub(super) node: String,
    pub(super) remote_nodes: Vec<String>,
    pub(super) remote_runs_root: String,
    pub(super) ssh_deadline_secs: u64,
    /// How long a trailing partial row must sit untouched before it counts as
    /// abandoned. Overridable so the policy can be exercised in a sandbox
    /// without a 15 minute wait; production uses the default.
    pub(super) quiescent_secs: u64,
}

pub(super) fn parse_nodes(spec: &str) -> (Vec<String>, Vec<Value>) {
    let mut nodes = Vec::new();
    let mut rejected = Vec::new();
    for raw in spec.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        match th::ssh_argv(name, th::SSH_DEADLINE_SECS) {
            Ok(_) => {
                if !nodes.iter().any(|existing| existing == name) {
                    nodes.push(name.to_string());
                }
            }
            Err(error) => rejected.push(json!({"node": name, "error": error.to_string()})),
        }
    }
    (nodes, rejected)
}

/// Running totals for one capture pass.
#[derive(Default)]
pub(super) struct Totals {
    appended_rows: u64,
    appended_bytes: u64,
    rotated: u64,
    errors: Vec<Value>,
    advanced: Vec<Value>,
    truncations: Vec<Value>,
}

/// Apply the truncated-tail policy to whatever the source has past the archive.
///
/// A trailing INCOMPLETE row is not data: the epoch never finished being
/// written. Every complete row is already archived by `ingest_source` above
/// this call, so all that is left is to make the hole explicit and permanent,
/// which is what buys the right for the collector to prune the run directory.
/// Anything that is NOT a lone unterminated fragment — complete rows still
/// missing, or a fragment too big to be one row — is deliberately not recorded,
/// so those keep blocking.
// Keep archive identity, source metadata and tail outcome explicit for auditability.
#[allow(clippy::too_many_arguments)]
fn apply_tail_policy<R: th::SourceReader>(
    archive: &Path,
    key: &th::SegmentKey,
    reader: &R,
    node: &str,
    source_path: &str,
    outcome: &th::IngestOutcome,
    source_mtime_ms: u64,
    abandoned: bool,
) -> Option<Value> {
    if outcome.residual_bytes == 0 {
        return None;
    }
    // Still being written: this is a mid-write row, not an abandoned one.
    if !abandoned {
        return None;
    }
    let residual =
        th::classify_residual(reader, outcome.archived_bytes, outcome.source_bytes).ok()?;
    let th::Residual::PartialRow(fragment) = residual else {
        return None;
    };
    match th::record_truncation(
        archive,
        key,
        node,
        source_path,
        &fragment,
        outcome.source_bytes,
        source_mtime_ms,
    ) {
        Ok(true) => Some(json!({
            "run_id": key.run_id,
            "attempt": key.attempt,
            "watermark": key.watermark,
            "node": node,
            "residual_bytes": fragment.len(),
            "archived_rows": outcome.archived_rows,
            "last_epoch": outcome.last_epoch,
            "source_path": source_path,
        })),
        _ => None,
    }
}

/// Local runs root sweep. Unchanged in behaviour except that an abandoned
/// trailing partial row is now recorded instead of blocking forever.
fn capture_local(
    archive: &Path,
    cfg: &Cfg,
    seen: &mut BTreeSet<th::SegmentKey>,
    totals: &mut Totals,
) -> usize {
    let run_ids = th::discover_trainer_runs(&cfg.runs_root).unwrap_or_default();
    for run_id in &run_ids {
        for found in th::scan_run_dir(&cfg.runs_root, run_id) {
            seen.insert(found.key.clone());
            let source = found.path.to_string_lossy().into_owned();
            let reader = th::FileSource(found.path.clone());
            match th::ingest_source(archive, &found.key, &reader, &source, &cfg.node) {
                Ok(outcome) => {
                    totals.appended_rows += outcome.appended_rows;
                    totals.appended_bytes += outcome.appended_bytes;
                    if outcome.rotated {
                        totals.rotated += 1;
                    }
                    if outcome.appended_rows > 0 || outcome.rotated {
                        totals.advanced.push(json!({
                            "node": cfg.node,
                            "run_id": found.key.run_id,
                            "attempt": found.key.attempt,
                            "rows": outcome.appended_rows,
                            "bytes": outcome.appended_bytes,
                            "last_epoch": outcome.last_epoch,
                            "rotated": outcome.rotated,
                        }));
                    }
                    let mtime_ms = std::fs::metadata(&found.path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);
                    // Local: one clock, so the file's own mtime is exact.
                    let abandoned =
                        th::tail_is_abandoned_after(mtime_ms, th::now_ms(), cfg.quiescent_secs);
                    if let Some(record) = apply_tail_policy(
                        archive, &found.key, &reader, &cfg.node, &source, &outcome, mtime_ms,
                        abandoned,
                    ) {
                        totals.truncations.push(record);
                    }
                }
                Err(error) => totals.errors.push(json!({
                    "node": cfg.node,
                    "run_id": found.key.run_id,
                    "attempt": found.key.attempt,
                    "error": error.to_string(),
                })),
            }
            // The final-selection record is a few hundred bytes and is what ties
            // a curve to the checkpoint that was actually shipped.
            if let Some(parent) = found.path.parent() {
                if let Ok(bytes) = std::fs::read(parent.join(th::RESULT_FILE)) {
                    let _ = th::archive_sidecar(archive, &found.key, th::RESULT_FILE, &bytes);
                }
            }
        }
    }
    run_ids.len()
}

/// Sweep one remote node. Returns Err only when the node could not be
/// enumerated at all — a per-segment failure is recorded and the rest of the
/// node still archives, because partial coverage beats none.
fn capture_remote_node(
    archive: &Path,
    cfg: &Cfg,
    node: &str,
    seen: &mut BTreeSet<th::SegmentKey>,
    totals: &mut Totals,
) -> Result<usize> {
    let listing = th::remote_scan(node, &cfg.remote_runs_root, cfg.ssh_deadline_secs)?;
    let results: BTreeMap<th::SegmentKey, th::RemoteFound> = listing
        .results
        .iter()
        .map(|found| (found.key.clone(), found.clone()))
        .collect();

    for found in &listing.metrics {
        // Claim the segment before doing any work: a key we have seen is never
        // sealed, so a mid-pass failure cannot make a live run look pruned.
        seen.insert(found.key.clone());
        let reader = th::RemoteSource {
            node: node.to_string(),
            runs_root: cfg.remote_runs_root.clone(),
            relative: found.relative.clone(),
            len: found.len,
            deadline_secs: cfg.ssh_deadline_secs,
        };
        let source = reader.display();
        let fingerprint = found.fingerprint();
        let prior = th::SegmentState::load(&th::segment_dir(archive, &found.key))?;
        let held = th::archived_bytes(archive, &found.key);
        // Already on the truncation ledger: the outstanding bytes are a row
        // that will never be finished, so there is nothing left to fetch.
        let settled = match prior.as_ref() {
            Some(state) => {
                state.truncation_count > 0 && held + state.truncated_tail_bytes >= found.len
            }
            None => false,
        };
        // Skipping on an unchanged size+mtime is what keeps an idle remote
        // segment free. It is ONLY safe when nothing is outstanding: a killed
        // trainer's file keeps its mtime forever, so skipping while bytes are
        // still unarchived would mean its truncated tail is never recorded and
        // the directory blocks pruning for good — the exact deadlock this
        // workstream exists to remove.
        let unchanged = match prior.as_ref() {
            Some(state) => {
                !state.source_fingerprint.is_empty() && state.source_fingerprint == fingerprint
            }
            None => false,
        } && (held >= found.len || settled);
        if !unchanged {
            match th::ingest_source_fingerprinted(
                archive,
                &found.key,
                &reader,
                &source,
                node,
                &fingerprint,
            ) {
                Ok(outcome) => {
                    totals.appended_rows += outcome.appended_rows;
                    totals.appended_bytes += outcome.appended_bytes;
                    if outcome.rotated {
                        totals.rotated += 1;
                    }
                    if outcome.appended_rows > 0 || outcome.rotated {
                        totals.advanced.push(json!({
                            "node": node,
                            "run_id": found.key.run_id,
                            "attempt": found.key.attempt,
                            "rows": outcome.appended_rows,
                            "bytes": outcome.appended_bytes,
                            "last_epoch": outcome.last_epoch,
                            "rotated": outcome.rotated,
                        }));
                    }
                    // Remote: judged on how long OUR clock has seen the
                    // size+mtime unchanged, never on the remote clock.
                    let abandoned = th::remote_tail_is_abandoned(
                        prior.as_ref(),
                        &fingerprint,
                        th::now_ms(),
                        cfg.quiescent_secs,
                    );
                    if let Some(record) = apply_tail_policy(
                        archive,
                        &found.key,
                        &reader,
                        node,
                        &source,
                        &outcome,
                        found.mtime_ms,
                        abandoned,
                    ) {
                        totals.truncations.push(record);
                    }
                }
                Err(error) => totals.errors.push(json!({
                    "node": node,
                    "run_id": found.key.run_id,
                    "attempt": found.key.attempt,
                    "source": source,
                    "error": error.to_string(),
                })),
            }
        }
        // train-result.json came back in the same enumeration, so it costs a
        // round trip only when its size differs from what we already hold.
        if let Some(result) = results.get(&found.key) {
            if th::sidecar_bytes(archive, &found.key, th::RESULT_FILE) != result.len {
                let sidecar = th::RemoteSource {
                    node: node.to_string(),
                    runs_root: cfg.remote_runs_root.clone(),
                    relative: result.relative.clone(),
                    len: result.len,
                    deadline_secs: cfg.ssh_deadline_secs,
                };
                match sidecar.read_range(0, result.len as usize) {
                    Ok(bytes) => {
                        let _ = th::archive_sidecar(archive, &found.key, th::RESULT_FILE, &bytes);
                    }
                    Err(error) => totals.errors.push(json!({
                        "node": node,
                        "run_id": found.key.run_id,
                        "sidecar": th::RESULT_FILE,
                        "error": error.to_string(),
                    })),
                }
            }
        }
    }
    Ok(listing.metrics.len())
}

/// One capture pass. Returns the receipt written for this cycle.
pub(super) fn capture(cfg: &Cfg) -> Result<Value> {
    let root = cfg.root.as_path();
    let archive = th::archive_root(root);
    // Non-blocking: if the collector is mid-archive under the same lock, this
    // tick has nothing useful to add and the next one is only 3 minutes away.
    let Some(_lock) = th::CaptureLock::acquire(&archive, 1, 0)? else {
        let record = json!({
            "ts_ms": th::now_ms(),
            "event": "skipped_locked",
            "node": cfg.node,
        });
        th::append_event(&archive, record.clone())?;
        return Ok(record);
    };

    let mut seen: BTreeSet<th::SegmentKey> = BTreeSet::new();
    let mut totals = Totals::default();

    // LOCAL FIRST, ALWAYS. Whatever the remotes do, the node this process runs
    // on is already fully archived by the time any ssh is attempted.
    let runs_scanned = capture_local(&archive, cfg, &mut seen, &mut totals);

    // Nodes whose source-of-truth we actually observed this pass. Only these
    // may have segments sealed; an unreachable node tells us nothing about
    // whether its run directories still exist.
    let mut covered: BTreeSet<String> = BTreeSet::new();
    covered.insert(cfg.node.clone());
    let mut statuses = th::load_nodes(&archive);
    let mut node_reports: Vec<Value> = Vec::new();
    let mut skipped_nodes: Vec<Value> = Vec::new();

    for node in &cfg.remote_nodes {
        let started = th::now_ms();
        let status = statuses
            .entry(node.clone())
            .or_insert_with(|| th::NodeStatus {
                node: node.clone(),
                ..Default::default()
            });
        status.node = node.clone();
        status.last_attempt_ms = started;
        match capture_remote_node(&archive, cfg, node, &mut seen, &mut totals) {
            Ok(segments) => {
                status.reachable = true;
                status.last_success_ms = th::now_ms();
                status.consecutive_failures = 0;
                status.last_error.clear();
                status.segments_seen = segments as u64;
                covered.insert(node.clone());
                node_reports.push(json!({
                    "node": node,
                    "ok": true,
                    "segments": segments,
                    "elapsed_ms": th::now_ms().saturating_sub(started),
                }));
            }
            Err(error) => {
                status.reachable = false;
                status.consecutive_failures += 1;
                status.last_error = error.to_string();
                let stale_secs = if status.last_success_ms == 0 {
                    Value::Null
                } else {
                    json!(th::now_ms().saturating_sub(status.last_success_ms) / 1000)
                };
                let report = json!({
                    "node": node,
                    "ok": false,
                    "error": error.to_string(),
                    "consecutive_failures": status.consecutive_failures,
                    "last_success_ms": status.last_success_ms,
                    "stale_secs": stale_secs,
                    "elapsed_ms": th::now_ms().saturating_sub(started),
                });
                node_reports.push(report.clone());
                skipped_nodes.push(report);
            }
        }
    }
    // Only the nodes this pass is configured to sweep make a coverage claim.
    // A node dropped from the roster must not leave a stale entry warning
    // forever about coverage nobody is asking for any more.
    statuses.retain(|node, _| cfg.remote_nodes.iter().any(|name| name == node));
    let _ = th::save_nodes(&archive, &statuses);

    // Any archived segment whose source has gone is sealed, not deleted: this is
    // the state a run reaches after the collector reclaims its 12 GB. A segment
    // belonging to a node we could not reach is left alone.
    let mut sealed = 0u64;
    let mut seal_deferred = 0u64;
    for (key, state) in th::load_segments(&archive)? {
        if seen.contains(&key) {
            continue;
        }
        let owner = if state.node.is_empty() {
            cfg.node.as_str()
        } else {
            state.node.as_str()
        };
        if !covered.contains(owner) {
            seal_deferred += 1;
            continue;
        }
        if th::seal_segment(&archive, &key).is_ok() {
            sealed += 1;
        }
    }

    let index = th::rebuild_index(root, &archive)?;
    let record = json!({
        "ts_ms": th::now_ms(),
        "event": "capture",
        "node": cfg.node,
        "runs_scanned": runs_scanned,
        "segments_live": seen.len(),
        "segments_total": index["segment_count"],
        "appended_rows": totals.appended_rows,
        "appended_bytes": totals.appended_bytes,
        "rotated": totals.rotated,
        "sealed_now": sealed,
        "seal_deferred": seal_deferred,
        "archive_rows": index["total_rows"],
        "archive_bytes": index["total_bytes"],
        "advanced": totals.advanced,
        "errors": totals.errors,
        "truncations": totals.truncations,
        "nodes": node_reports,
        // Named explicitly so a pass with reduced coverage can never read as a
        // clean success.
        "nodes_skipped": skipped_nodes,
        "coverage_complete": skipped_nodes.is_empty(),
    });
    let changed = totals.appended_rows > 0
        || totals.rotated > 0
        || sealed > 0
        || !totals.errors.is_empty()
        || !totals.truncations.is_empty()
        || !skipped_nodes.is_empty();
    if changed || idle_receipt_due(&archive) {
        th::append_event(&archive, record.clone())?;
    }
    Ok(record)
}

/// True when the receipts log has had nothing written to it for long enough
/// that a heartbeat is worth the bytes.
pub(super) fn idle_receipt_due(archive: &Path) -> bool {
    let Ok(modified) = std::fs::metadata(archive.join(th::EVENTS_FILE)).and_then(|m| m.modified())
    else {
        return true;
    };
    std::time::SystemTime::now()
        .duration_since(modified)
        .map(|age| age.as_secs() >= IDLE_RECEIPT_SECS)
        .unwrap_or(true)
}
