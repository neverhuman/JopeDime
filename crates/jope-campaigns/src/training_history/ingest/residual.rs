use super::*;

#[derive(Clone, Debug, Default)]
pub struct IngestOutcome {
    pub appended_bytes: u64,
    pub appended_rows: u64,
    pub rotated: bool,
    pub archived_bytes: u64,
    pub archived_rows: u64,
    pub source_bytes: u64,
    pub last_epoch: Option<i64>,
    /// Source bytes past the last archived byte. After a successful ingest this
    /// is exactly the trailing fragment with no newline in it — i.e. a row the
    /// writer had not finished. Zero when the source ends on a row boundary.
    pub residual_bytes: u64,
}

impl IngestOutcome {
    /// True once every byte the source currently holds is in the archive. This
    /// is the strict predicate: it is false while a row is mid-write.
    pub fn covered(&self) -> bool {
        self.archived_bytes >= self.source_bytes
    }
}

/// What the bytes between the archive head and the end of the source actually
/// are. The distinction is the whole truncated-tail policy: a trailing fragment
/// with no newline is a row that was never finished being written and is not
/// data, whereas a shortfall containing a newline is COMPLETE rows that are not
/// archived — which must never be written off.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Residual {
    /// Source ends exactly on a row boundary and everything is archived.
    None,
    /// A single unterminated trailing fragment (the partial row).
    PartialRow(Vec<u8>),
    /// At least one complete row is missing from the archive. Fail closed.
    CompleteRowsMissing { bytes: u64 },
    /// Unterminated but implausibly large for one row. Fail closed.
    OversizedFragment { bytes: u64 },
}

impl Residual {
    pub fn reason(&self) -> &'static str {
        match self {
            Residual::None => "none",
            Residual::PartialRow(_) => "partial_row",
            Residual::CompleteRowsMissing { .. } => "complete_rows_missing",
            Residual::OversizedFragment { .. } => "oversized_fragment",
        }
    }
}

/// Classify the gap between `archived` and the end of the source.
///
/// Reads at most `MAX_TRUNCATED_TAIL_BYTES + 1` bytes and never touches the
/// archive, so it is safe to call from the read-only alarm as well as from the
/// collector.
pub fn classify_residual<R: SourceReader>(
    reader: &R,
    archived: u64,
    source_bytes: u64,
) -> Result<Residual> {
    if source_bytes <= archived {
        return Ok(Residual::None);
    }
    let gap = source_bytes - archived;
    if gap > MAX_TRUNCATED_TAIL_BYTES {
        // Still worth knowing whether it is rows or one absurd fragment, but a
        // gap this size is refused either way; read a bounded window only.
        let window = reader.read_range(archived, MAX_TRUNCATED_TAIL_BYTES as usize + 1)?;
        return Ok(if window.contains(&b'\n') {
            Residual::CompleteRowsMissing { bytes: gap }
        } else {
            Residual::OversizedFragment { bytes: gap }
        });
    }
    let fragment = reader.read_range(archived, gap as usize)?;
    if fragment.contains(&b'\n') {
        return Ok(Residual::CompleteRowsMissing { bytes: gap });
    }
    if fragment.is_empty() {
        return Ok(Residual::None);
    }
    Ok(Residual::PartialRow(fragment))
}

/// Whether a LOCAL partial row has been abandoned (writer gone) rather than
/// being written right now. `source_mtime_ms == 0` means "unknown", treated as
/// live. Local only: source and observer share one clock.
pub fn tail_is_abandoned(source_mtime_ms: u64, now: u64) -> bool {
    tail_is_abandoned_after(source_mtime_ms, now, TAIL_QUIESCENT_SECS)
}

pub fn tail_is_abandoned_after(source_mtime_ms: u64, now: u64, quiescent_secs: u64) -> bool {
    if source_mtime_ms == 0 || source_mtime_ms > now {
        // Unknown, or stamped in the future by a clock we do not trust.
        return false;
    }
    now.saturating_sub(source_mtime_ms) / 1000 >= quiescent_secs
}

/// Record that a trailing INCOMPLETE row was written off for `key`.
///
/// THE POLICY, stated once: a trailing incomplete line is not data — the epoch
/// never finished being written. Every complete row is already archived by
/// `ingest_source` before this is ever called. Recording is what buys the right
/// to let the source directory be pruned, so it is deliberately loud and
/// permanent:
/// * the raw fragment is kept verbatim as evidence (`truncated-tail.<ms>.bin`),
///   so nothing is actually thrown away — it is only not counted as a row;
/// * one line lands in the segment's `truncation.v1.jsonl` and one in the
///   archive-wide `truncations.v1.jsonl`;
/// * the segment state carries `truncated_tail_bytes`, which the index, the
///   collector's receipt and the health alarm all read.
///
/// Idempotent: re-recording the same fragment for the same offset is a no-op,
/// so the 3-minute timer does not grow the ledger.
#[allow(clippy::too_many_arguments)]
pub fn record_truncation(
    archive: &Path,
    key: &SegmentKey,
    node: &str,
    source_path: &str,
    fragment: &[u8],
    source_bytes: u64,
    source_mtime_ms: u64,
) -> Result<bool> {
    if !key.is_sane() {
        anyhow::bail!("refusing unsafe segment key {key:?}");
    }
    let seg_dir = segment_dir(archive, key);
    let mut state = match SegmentState::load(&seg_dir)? {
        Some(state) => state,
        None => SegmentState::initial(),
    };
    let residual = fragment.len() as u64;
    let sha = sha256_hex(fragment);
    // Same hole, already on the ledger.
    if state.truncation_count > 0
        && state.truncated_tail_bytes == residual
        && state.archived_bytes + residual == source_bytes
    {
        return Ok(false);
    }
    let stamp = now_ms();
    fs::create_dir_all(&seg_dir)?;
    let evidence = format!("truncated-tail.{stamp}.bin");
    write_atomic(&seg_dir.join(&evidence), fragment)?;

    let record = json!({
        "schema": TRUNCATION_SCHEMA,
        "ts_ms": stamp,
        "run_key": format!("scq/{}/{}/{}", key.run_id, key.attempt, key.watermark),
        "run_id": key.run_id,
        "attempt": key.attempt,
        "watermark": key.watermark,
        "node": node,
        "source_path": source_path,
        "source_bytes": source_bytes,
        "archived_bytes": state.archived_bytes,
        "archived_rows": state.archived_rows,
        "last_archived_epoch": state.last_epoch,
        "residual_bytes": residual,
        "source_mtime_ms": source_mtime_ms,
        "source_idle_secs": stamp.saturating_sub(source_mtime_ms) / 1000,
        "fragment_sha256": sha,
        "fragment_preview": String::from_utf8_lossy(&fragment[..fragment.len().min(240)]),
        "evidence_file": evidence,
        "policy": "trailing incomplete JSONL row is not data; every complete row is archived; \
                   the fragment is kept verbatim and the source directory becomes prunable",
    });
    for path in [
        seg_dir.join(TRUNCATION_FILE),
        archive.join(TRUNCATIONS_FILE),
    ] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        writeln!(file, "{record}")?;
        file.sync_all()?;
    }

    state.truncated_tail_bytes = residual;
    state.truncation_count += 1;
    state.last_truncation_ms = stamp;
    state.last_check_ms = stamp;
    if state.run_id.is_empty() {
        state.schema = SEGMENT_SCHEMA.to_string();
        state.run_id = key.run_id.clone();
        state.attempt = key.attempt.clone();
        state.watermark = key.watermark.clone();
        state.node = node.to_string();
        state.source_path = source_path.to_string();
    }
    state.save(&seg_dir)?;
    Ok(true)
}
