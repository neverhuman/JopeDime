use super::*;

/// One metrics file discovered under a candidate run directory, as
/// `(run_id, path relative to runs_root, size in bytes, mtime in ms)`.
pub(super) type RemoteMetrics = (String, String, u64, u64);

/// Enumerate `train_metrics.jsonl` under the candidate directories of a remote
/// node in a single ssh round trip. The mtime comes back with the size because
/// the truncated-tail rule needs to know whether the writer is still alive.
pub(super) fn remote_metrics(
    node: &str,
    runs_root: &str,
    names: &[String],
) -> Result<Vec<RemoteMetrics>> {
    let target = ssh_target(node)?;
    let mut args: Vec<String> = target[1..].to_vec();
    args.push(format!(
        "cd {runs_root} && xargs -d '\\n' sh -c 'find \"$@\" -maxdepth 6 -name {} -printf \"%s %T@ %p\\n\" 2>/dev/null' sh",
        th::METRICS_FILE
    ));
    let listing = names.join("\n");
    let stdout = run_capture(&target[0], &args, Some(&listing))?;
    let mut rows = Vec::new();
    for line in stdout.lines() {
        let mut parts = line.split_whitespace();
        let (Some(size), Some(mtime), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if parts.next().is_some() {
            continue;
        }
        let path = path.trim_start_matches("./");
        let Some(run_id) = path.split('/').next() else {
            continue;
        };
        rows.push((
            run_id.to_string(),
            path.to_string(),
            size.parse::<u64>().unwrap_or(0),
            mtime
                .parse::<f64>()
                .map(|seconds| (seconds * 1000.0) as u64)
                .unwrap_or(0),
        ));
    }
    Ok(rows)
}

pub(super) fn remote_read(node: &str, runs_root: &str, relative: &str) -> Result<Vec<u8>> {
    // Every component came from our own `find` over names we already validated
    // with `sane_name`, but re-check before it reaches a shell.
    if !relative.split('/').all(sane_name) {
        bail!("refusing unsafe remote path {relative:?}");
    }
    let target = ssh_target(node)?;
    let mut args: Vec<String> = target[1..].to_vec();
    args.push(format!("cat -- {runs_root}/{relative}"));
    run_capture_bytes(&target[0], &args, None)
}

/// Result of the metrics precondition for one GC pass.
pub(super) struct MetricsGuard {
    /// Candidates the archive provably covers (or that hold no metrics at all).
    pub(super) safe: Vec<String>,
    /// Candidates refused because their metrics are not durably archived.
    pub(super) blocked: Vec<Value>,
    pub(super) archived_rows: u64,
    pub(super) archived_bytes: u64,
    /// Candidates cleared only after an abandoned trailing partial row was
    /// written off under the truncated-tail policy, with the hole recorded.
    pub(super) truncated: Vec<Value>,
}

/// One metrics file under a candidate directory, local or remote.
pub(super) struct GuardFile {
    /// Absolute path locally, path relative to the runs root remotely.
    source: String,
    key: th::SegmentKey,
    len: u64,
    mtime_ms: u64,
}

/// THE PRECONDITION. Archive-then-prune: every candidate's `train_metrics.jsonl`
/// is ingested into `training/history/` and re-verified here, and only
/// candidates whose rows are provably on durable storage are returned as `safe`.
///
/// Fail-closed in every direction:
/// * capture lock unavailable  -> metric-bearing candidates are refused;
/// * ingest error              -> that candidate is refused;
/// * archive still behind the source (a row mid-write) -> refused this pass;
/// * remote enumeration failed -> every `-train-` candidate on that node is
///   refused, while dirs that cannot carry trainer metrics still prune, so a
///   flaky ssh degrades throughput instead of losing data.
///
/// Note this is deliberately independent of `--prefix`: it guards whatever the
/// plan selected, so widening the collector to `jope-` (the obvious next move
/// under disk pressure) cannot silently delete a training curve.
///
/// THE ONE EXCEPTION, and it is a policy the operator set: a trainer killed
/// mid-write leaves a trailing INCOMPLETE JSONL line. That line is not data —
/// the epoch never finished being written — and refusing to prune on it leaks
/// ~12 GB per occurrence forever. So when the shortfall is exactly one
/// unterminated fragment AND the file has stopped being written, every complete
/// row is archived, the hole is recorded (`truncations.v1.jsonl`, the segment
/// state, this receipt) with the fragment kept verbatim, and the directory is
/// cleared. A shortfall that contains a newline is COMPLETE rows and still
/// blocks; so does a fragment too large to be one row, and so does a file that
/// is still being written.
pub(super) fn guard_metrics(
    node: &str,
    runs_root: &str,
    root: &Path,
    names: &[String],
) -> MetricsGuard {
    let mut guard = MetricsGuard {
        safe: Vec::new(),
        blocked: Vec::new(),
        archived_rows: 0,
        archived_bytes: 0,
        truncated: Vec::new(),
    };
    if names.is_empty() {
        return guard;
    }
    let archive = th::archive_root(root);

    // Discover what actually carries metrics. Local runs are walked directly;
    // remote nodes answer in one round trip.
    let mut by_run: HashMap<String, Vec<GuardFile>> = HashMap::new();
    let mut enumeration_failed = false;
    if node == LOCAL_NODE {
        for name in names {
            for found in th::scan_run_dir(Path::new(runs_root), name) {
                let mtime_ms = std::fs::metadata(&found.path)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                by_run.entry(name.clone()).or_default().push(GuardFile {
                    source: found.path.to_string_lossy().into_owned(),
                    key: found.key,
                    len: found.len,
                    mtime_ms,
                });
            }
        }
    } else {
        match remote_metrics(node, runs_root, names) {
            Ok(rows) => {
                for (run_id, relative, len, mtime_ms) in rows {
                    let Some(key) = th::key_from_path(&run_id, Path::new(&relative)) else {
                        continue;
                    };
                    by_run.entry(run_id).or_default().push(GuardFile {
                        source: relative,
                        key,
                        len,
                        mtime_ms,
                    });
                }
            }
            Err(_) => enumeration_failed = true,
        }
    }

    // Hold the same lock the capture timer uses, so a tick cannot interleave
    // with the archiving we are about to do.
    let lock = th::CaptureLock::acquire(&archive, LOCK_ATTEMPTS, LOCK_BACKOFF_MS)
        .ok()
        .flatten();

    for name in names {
        let Some(files) = by_run.get(name) else {
            if enumeration_failed && name.contains(th::TRAIN_MARKER) {
                guard.blocked.push(json!({
                    "dir": name,
                    "reason": "metrics_enumeration_failed",
                    "node": node,
                }));
            } else {
                guard.safe.push(name.clone());
            }
            continue;
        };
        if lock.is_none() {
            guard.blocked.push(json!({
                "dir": name,
                "reason": "capture_lock_unavailable",
                "metrics_files": files.len(),
            }));
            continue;
        }
        let mut covered = true;
        let mut detail = Vec::new();
        for file in files {
            let GuardFile {
                source,
                key,
                len: source_len,
                mtime_ms,
            } = file;
            let reader = if node == LOCAL_NODE {
                Ok(th::AnySource::File(th::FileSource(PathBuf::from(source))))
            } else {
                remote_read(node, runs_root, source)
                    .map(|bytes| th::AnySource::Bytes(th::BytesSource(bytes)))
            };
            let display = if node == LOCAL_NODE {
                source.clone()
            } else {
                format!("{node}:{runs_root}/{source}")
            };
            let outcome = match &reader {
                Ok(reader) => th::ingest_source(&archive, key, reader, &display, node),
                Err(error) => Err(anyhow::anyhow!("{error}")),
            };
            match outcome {
                Ok(outcome) => {
                    guard.archived_rows += outcome.appended_rows;
                    guard.archived_bytes += outcome.appended_bytes;
                    // Re-read the archive file length rather than trusting the
                    // in-memory outcome: this is the durable fact.
                    let held = th::archived_bytes(&archive, key);
                    if held >= *source_len {
                        continue;
                    }
                    // Shortfall. The only kind that may be written off is a
                    // lone unterminated fragment on a file nobody is writing.
                    let reader = reader.as_ref().expect("reader present on Ok outcome");
                    let residual = th::classify_residual(reader, held, *source_len);
                    // Local sources share this clock, so their own mtime is
                    // exact. A remote mtime is another machine's opinion of the
                    // time; judging on it could write off a row that is still
                    // being written, so remote quiescence is instead "the
                    // capture timer has observed this size+mtime unchanged for
                    // long enough, measured here".
                    let quiescent = if node == LOCAL_NODE {
                        th::tail_is_abandoned(*mtime_ms, now_ms())
                    } else {
                        let state = match th::SegmentState::load(&th::segment_dir(&archive, key)) {
                            Ok(state) => state,
                            Err(error) => {
                                covered = false;
                                detail.push(json!({
                                    "source": source,
                                    "reason": "segment_state_unreadable",
                                    "error": format!("{error:#}"),
                                }));
                                continue;
                            }
                        };
                        th::remote_tail_is_abandoned(
                            state.as_ref(),
                            &format!("{source_len}:{mtime_ms}"),
                            now_ms(),
                            th::TAIL_QUIESCENT_SECS,
                        )
                    };
                    match (&residual, quiescent) {
                        (Ok(th::Residual::PartialRow(fragment)), true) => {
                            match th::record_truncation(
                                &archive,
                                key,
                                node,
                                &display,
                                fragment,
                                *source_len,
                                *mtime_ms,
                            ) {
                                Ok(_) => guard.truncated.push(json!({
                                    "dir": name,
                                    "source": display,
                                    "source_bytes": source_len,
                                    "archived_bytes": held,
                                    "residual_bytes": fragment.len(),
                                    "source_idle_secs": now_ms().saturating_sub(*mtime_ms) / 1000,
                                    "reason": "truncated_tail_archived",
                                    "policy": "trailing incomplete row is not data; \
                                               all complete rows archived and the hole recorded",
                                })),
                                Err(error) => {
                                    // Could not make the record durable, so the
                                    // right to prune was never earned.
                                    covered = false;
                                    detail.push(json!({
                                        "source": display,
                                        "reason": "truncation_record_failed",
                                        "error": error.to_string(),
                                    }));
                                }
                            }
                        }
                        _ => {
                            covered = false;
                            detail.push(json!({
                                "source": display,
                                "source_bytes": source_len,
                                "archived_bytes": held,
                                "residual": residual
                                    .as_ref()
                                    .map(|r| r.reason())
                                    .unwrap_or("unreadable"),
                                "source_idle_secs": now_ms().saturating_sub(*mtime_ms) / 1000,
                                "quiescent": quiescent,
                            }));
                        }
                    }
                }
                Err(error) => {
                    covered = false;
                    detail.push(json!({"source": display, "error": error.to_string()}));
                }
            }
        }
        if covered {
            guard.safe.push(name.clone());
        } else {
            guard.blocked.push(json!({
                "dir": name,
                "reason": "metrics_unarchived",
                "detail": detail,
            }));
        }
    }

    if lock.is_some() {
        let _ = th::rebuild_index(root, &archive);
    }
    drop(lock);
    guard
}
