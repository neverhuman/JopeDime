//! training_history — durable, append-only archive of DL trainer per-epoch metrics.
//!
//! WHY THIS EXISTS. Every policy training run writes one JSONL row per epoch to
//! `<run_dir>/attempts/<n>/outputs/training/watermark-<W>/train_metrics.jsonl`.
//! That row carries train_loss, val_loss, overfit_gap and the whole validation
//! metric block — the only record of how the model learned. The run directory
//! that holds it also holds ~12 GB of checkpoints, so it is a garbage-collection
//! target, and eight earlier trainer runs (specs a, a-g2, b, b-g2..b-g6 still
//! exist under `specs/`) have already lost their metrics that way. This module
//! copies the ~3 KB/epoch of metrics somewhere the collector never looks, so the
//! 12 GB of tensors can keep falling off disk while the curve survives forever.
//!
//! GUARANTEES.
//! * **Append-only.** The archive copy is only ever extended. A source file that
//!   shrinks or whose prefix changes is treated as a *replacement*: the existing
//!   archive segment is rotated into `rewrites/` and kept, never truncated.
//! * **Byte-identical.** Rows land exactly as the trainer emitted them, so any
//!   downstream schema can be re-derived from the archive alone.
//! * **Whole rows only.** Appends stop at the last newline, so a row being
//!   written while we read never enters the archive half-formed.
//! * **Incremental.** Progress is a byte offset per segment; an unchanged source
//!   costs one `stat`.
//! * **Idempotent.** Re-running over an unchanged source appends nothing.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const INDEX_SCHEMA: &str = "JopeTrainingHistoryIndex.v1";
pub const SEGMENT_SCHEMA: &str = "JopeTrainingHistorySegment.v1";
pub const METRICS_FILE: &str = "train_metrics.jsonl";
pub const RESULT_FILE: &str = "train-result.json";
pub const INDEX_FILE: &str = "index.v1.json";
pub const EVENTS_FILE: &str = "capture-events.v1.jsonl";
pub const LOCK_FILE: &str = ".capture.lock";
/// Per-segment truncation ledger (one line per recorded truncated tail).
pub const TRUNCATION_FILE: &str = "truncation.v1.jsonl";
/// Archive-wide truncation ledger. This is the honest record of every hole:
/// one line per (segment, truncation), never rewritten, never pruned.
pub const TRUNCATIONS_FILE: &str = "truncations.v1.jsonl";
pub const TRUNCATION_SCHEMA: &str = "JopeTrainingHistoryTruncation.v1";
/// Per-node capture reachability, so an unreachable node is visible instead of
/// silently reducing coverage.
pub const NODES_FILE: &str = "nodes.v1.json";
pub const NODES_SCHEMA: &str = "JopeTrainingHistoryNodes.v1";

/// Bytes of the archived prefix that are hashed to detect a source file which
/// was *replaced* rather than appended to. Cheap (one 4 KiB read per segment).
const ANCHOR_BYTES: usize = 4096;
/// Depth bound when walking `<run_dir>/attempts` for metrics files.
const WALK_DEPTH: usize = 6;
/// Directory-name marker identifying a trainer run directory.
pub const TRAIN_MARKER: &str = "-train-";

/// Last line of a successful remote enumeration. Its absence means the sweep
/// did not run to completion, which is never the same thing as "no runs here".
pub const SCAN_SENTINEL: &str = "__jope_scan_end__";

/// Upper bound on enumeration lines. Exceeding it is an error, not a silent
/// truncation of the listing.
const MAX_SCAN_LINES: usize = 200_000;

/// The host this process runs on. `ssh xbabe2` resolves to a different machine
/// entirely (a Mac Studio), so the local node is only ever addressed by path.
pub const LOCAL_NODE: &str = "xbabe2";
/// Nodes a trainer can legitimately be scheduled onto besides this one. The
/// trainer specs pin `hostname -qxE "xbabe1|xbabe2|xbabe3"`, and xbabe4 is in
/// the cluster roster, so all three are swept.
pub const REMOTE_NODES: &[&str] = &["xbabe1", "xbabe3", "xbabe4"];
/// ssh dial timeout. Short: a down node must not hold up the local path.
pub const SSH_CONNECT_TIMEOUT_SECS: u64 = 8;
/// Hard wall-clock bound on any single remote command, enforced by `timeout(1)`
/// so a half-open TCP connection cannot wedge the capture timer.
pub const SSH_DEADLINE_SECS: u64 = 45;

/// A trailing fragment bigger than this is not credibly one partial JSONL row
/// (a real row is ~3 KB), so it is never written off as a truncated tail — the
/// directory keeps blocking and a human gets to look at it.
pub const MAX_TRUNCATED_TAIL_BYTES: u64 = 65_536;
/// How long a metrics file must sit untouched before its trailing partial row is
/// judged abandoned rather than mid-write. The trainer emits an epoch every
/// ~4 min, so 15 min is several missed writes.
pub const TAIL_QUIESCENT_SECS: u64 = 900;

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// `<root>/training/history` — the durable archive root. Nothing under here is
/// ever a GC candidate: the collector only walks `var/runs` and `receipts/`.
pub fn archive_root(root: &Path) -> PathBuf {
    root.join("training").join("history")
}

fn sane_component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 200
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && !value.starts_with('.')
}

/// Identity of one archived metrics stream. A run may retry (`attempt`) and may
/// train against several label watermarks, so all three parts are needed.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SegmentKey {
    pub run_id: String,
    pub attempt: String,
    pub watermark: String,
}

impl SegmentKey {
    pub fn is_sane(&self) -> bool {
        sane_component(&self.run_id)
            && sane_component(&self.attempt)
            && sane_component(&self.watermark)
    }

    pub fn relative(&self) -> PathBuf {
        Path::new("runs")
            .join(&self.run_id)
            .join(&self.attempt)
            .join(&self.watermark)
    }
}

pub fn segment_dir(archive: &Path, key: &SegmentKey) -> PathBuf {
    archive.join(key.relative())
}

/// Persisted per-segment progress. `archived_bytes` is the resume offset: it is
/// always a whole number of complete rows.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SegmentState {
    #[serde(default)]
    pub schema: String,
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub attempt: String,
    #[serde(default)]
    pub watermark: String,
    /// Where the rows were read from, for provenance only.
    #[serde(default)]
    pub source_path: String,
    #[serde(default)]
    pub node: String,
    #[serde(default)]
    pub archived_bytes: u64,
    #[serde(default)]
    pub archived_rows: u64,
    #[serde(default)]
    pub first_epoch: Option<i64>,
    #[serde(default)]
    pub last_epoch: Option<i64>,
    /// sha256 of the first `min(ANCHOR_BYTES, archived_bytes)` archived bytes.
    #[serde(default)]
    pub anchor_sha256: String,
    #[serde(default)]
    pub anchor_len: u64,
    #[serde(default)]
    pub first_seen_ms: u64,
    /// Last time a row was actually appended. This is what "the archive is
    /// advancing" means, and what the alarm watches.
    #[serde(default)]
    pub last_advance_ms: u64,
    #[serde(default)]
    pub last_check_ms: u64,
    /// False once the source run directory has been pruned. The archive keeps
    /// serving the rows; this only records that no more will arrive.
    #[serde(default)]
    pub source_present: bool,
    #[serde(default)]
    pub rewrite_count: u64,
    /// `<len>:<mtime_ms>` of the source as of the last completed ingest. Only
    /// used for remote sources, where a `stat` costs an ssh round trip: an
    /// unchanged fingerprint means there is nothing to fetch.
    #[serde(default)]
    pub source_fingerprint: String,
    /// LOCAL wall clock when `source_fingerprint` last changed. This is how long
    /// the remote file has provably not been written to, measured on one clock,
    /// so judging a remote trailing row abandoned never depends on the remote
    /// node's clock being right.
    #[serde(default)]
    pub fingerprint_since_ms: u64,
    /// Bytes of a trailing INCOMPLETE row that were written off under the
    /// truncated-tail policy. Non-zero means this stream has a recorded hole.
    #[serde(default)]
    pub truncated_tail_bytes: u64,
    #[serde(default)]
    pub truncation_count: u64,
    #[serde(default)]
    pub last_truncation_ms: u64,
    /// Small scalar summary lifted from the first/last archived row, so the
    /// index can be rebuilt without re-reading every segment body.
    #[serde(default)]
    pub head: Map<String, Value>,
    #[serde(default)]
    pub tail: Map<String, Value>,
}

impl SegmentState {
    fn path(seg_dir: &Path) -> PathBuf {
        seg_dir.join("segment.v1.json")
    }

    pub fn load(seg_dir: &Path) -> Option<Self> {
        let text = fs::read_to_string(Self::path(seg_dir)).ok()?;
        serde_json::from_str(&text).ok()
    }

    fn save(&self, seg_dir: &Path) -> Result<()> {
        write_atomic(&Self::path(seg_dir), &serde_json::to_vec_pretty(self)?)
    }

    pub fn metrics_path(seg_dir: &Path) -> PathBuf {
        seg_dir.join(METRICS_FILE)
    }
}

/// Write `bytes` to `path` via a same-directory temp file plus rename, so a
/// reader never observes a partial state file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("path has no parent")?;
    fs::create_dir_all(parent)?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("tmp");
    let temporary = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)?;
    Ok(())
}

pub fn append_event(archive: &Path, record: Value) -> Result<()> {
    fs::create_dir_all(archive)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(archive.join(EVENTS_FILE))?;
    writeln!(file, "{record}")?;
    Ok(())
}

/// A source of metrics bytes. Local runs read the file directly; a run sitting
/// on another node is streamed into memory first (these files are kilobytes).
pub trait SourceReader {
    fn len(&self) -> Result<u64>;
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
    fn read_range(&self, offset: u64, len: usize) -> Result<Vec<u8>>;
}

pub struct FileSource(pub PathBuf);

impl SourceReader for FileSource {
    fn len(&self) -> Result<u64> {
        Ok(fs::metadata(&self.0)?.len())
    }

    fn read_range(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut file = fs::File::open(&self.0)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = Vec::with_capacity(len.min(1 << 22));
        file.take(len as u64).read_to_end(&mut bytes)?;
        Ok(bytes)
    }
}

pub struct BytesSource(pub Vec<u8>);

impl SourceReader for BytesSource {
    fn len(&self) -> Result<u64> {
        Ok(self.0.len() as u64)
    }

    fn read_range(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let start = (offset as usize).min(self.0.len());
        let end = start.saturating_add(len).min(self.0.len());
        Ok(self.0[start..end].to_vec())
    }
}

// ---------------------------------------------------------------------------
// Remote capture. A trainer is pinned to `xbabe1|xbabe2|xbabe3`, so a run can
// land on a node this process is not running on. Everything below exists to
// pull those rows over ssh incrementally, without ever letting a slow or dead
// node interfere with the local path.
// ---------------------------------------------------------------------------

/// argv for a bounded, non-interactive ssh to `node`.
///
/// `timeout(1)` wraps ssh because ConnectTimeout only covers the dial: a node
/// that accepts the connection and then wedges would otherwise hold the capture
/// timer open indefinitely.
pub fn ssh_argv(node: &str, deadline_secs: u64) -> Result<Vec<String>> {
    let mut argv: Vec<String> = vec![
        "timeout".into(),
        "-k".into(),
        "5".into(),
        deadline_secs.max(5).to_string(),
        "ssh".into(),
        "-o".into(),
        "BatchMode=yes".into(),
        "-o".into(),
        format!("ConnectTimeout={SSH_CONNECT_TIMEOUT_SECS}"),
        "-o".into(),
        "StrictHostKeyChecking=accept-new".into(),
    ];
    match node {
        "xbabe1" | "xbabe3" => argv.push(node.into()),
        // xbabe4 listens on a non-default port.
        "xbabe4" => {
            argv.push("-p".into());
            argv.push("2224".into());
            argv.push("xbabe4".into());
        }
        // `ssh xbabe2` reaches a Mac Studio, not this host. Never.
        LOCAL_NODE => anyhow::bail!("xbabe2 is the local host; never ssh to it"),
        other => anyhow::bail!("unknown node {other}"),
    }
    Ok(argv)
}

fn run_ssh(node: &str, deadline_secs: u64, remote: &str) -> Result<Vec<u8>> {
    let argv = ssh_argv(node, deadline_secs)?;
    let output = std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .arg(remote)
        .stdin(std::process::Stdio::null())
        .output()
        .with_context(|| format!("spawn ssh to {node}"))?;
    if !output.status.success() {
        anyhow::bail!(
            "{node}: ssh exit {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

/// A runs root is a fixed operator-configured path; refuse anything that could
/// change the meaning of the remote shell command built from it.
fn sane_root(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 512
        && path.starts_with('/')
        && path
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'-' | b'_' | b'.'))
        && !path.contains("..")
}

/// One metrics file found on a remote node.
#[derive(Clone, Debug)]
pub struct RemoteFound {
    pub key: SegmentKey,
    /// Path relative to the remote runs root.
    pub relative: String,
    pub len: u64,
    pub mtime_ms: u64,
}

impl RemoteFound {
    pub fn fingerprint(&self) -> String {
        format!("{}:{}", self.len, self.mtime_ms)
    }
}

/// Everything one node's enumeration round trip found.
#[derive(Clone, Debug, Default)]
pub struct RemoteListing {
    pub metrics: Vec<RemoteFound>,
    /// `train-result.json` siblings, so the final-selection record is pulled in
    /// the same pass without a second probe per segment.
    pub results: Vec<RemoteFound>,
}

/// Enumerate every trainer metrics file (and its train-result.json sibling) on
/// `node` in ONE ssh round trip.
///
/// The outer `find` filters on name before `-type d`, so the ~70k-entry runs
/// root costs one getdents sweep and a stat only for `-train-` matches.
pub fn remote_scan(node: &str, runs_root: &str, deadline_secs: u64) -> Result<RemoteListing> {
    if !sane_root(runs_root) {
        anyhow::bail!("refusing unsafe runs root {runs_root:?}");
    }
    // The sweep ends by echoing SCAN_SENTINEL. Without that marker an empty
    // stdout is ambiguous in exactly the way that matters: "this node holds no
    // trainer runs" looks identical to "the runs root is missing/unreadable"
    // and to "the ssh stream was cut mid-listing". The old form swallowed a
    // failed `cd` with `|| exit 0`, so a node whose runs root had moved was
    // recorded as ok:true / segments:0 / coverage_complete:true - a green
    // coverage claim over a directory nobody could read. Fail loudly instead:
    // run_ssh bails, the node lands in nodes_skipped, coverage_complete goes
    // false and training_history_node_unreachable fires.
    let remote = format!(
        "cd {runs_root} 2>/dev/null || \
         {{ echo 'runs root missing or unreadable' >&2; exit 17; }}; \
         find . -mindepth 1 -maxdepth 1 -name '*{TRAIN_MARKER}*' -type d \
         -exec find {{}} -maxdepth {WALK_DEPTH} \\( -name {METRICS_FILE} -o -name {RESULT_FILE} \\) \
         -printf '%s %T@ %p\\n' ';' 2>/dev/null; echo {SCAN_SENTINEL}"
    );
    let stdout = run_ssh(node, deadline_secs, &remote)?;
    let text = String::from_utf8_lossy(&stdout);
    let mut listing = RemoteListing::default();
    let mut saw_sentinel = false;
    let mut seen = 0usize;
    for line in text.lines() {
        seen += 1;
        if seen > MAX_SCAN_LINES {
            // Refuse rather than silently parsing a prefix of the listing: a
            // truncated listing is uncaptured segments reported as covered.
            anyhow::bail!(
                "{node}: enumeration exceeded {MAX_SCAN_LINES} lines; refusing to claim coverage"
            );
        }
        if line.trim_end() == SCAN_SENTINEL {
            saw_sentinel = true;
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(size), Some(mtime), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if parts.next().is_some() {
            // A path with whitespace cannot be one of ours.
            continue;
        }
        let relative = path.trim_start_matches("./");
        if !relative.split('/').all(sane_component) {
            continue;
        }
        let Some(run_id) = relative.split('/').next() else {
            continue;
        };
        let Some(key) = key_from_path(run_id, Path::new(relative)) else {
            continue;
        };
        // find prints %T@ as seconds with a fractional part.
        let mtime_ms = mtime
            .parse::<f64>()
            .map(|seconds| (seconds * 1000.0) as u64)
            .unwrap_or(0);
        let found = RemoteFound {
            key,
            relative: relative.to_string(),
            len: size.parse::<u64>().unwrap_or(0),
            mtime_ms,
        };
        match Path::new(relative).file_name().and_then(|n| n.to_str()) {
            Some(METRICS_FILE) => listing.metrics.push(found),
            Some(RESULT_FILE) => listing.results.push(found),
            _ => {}
        }
    }
    if !saw_sentinel {
        anyhow::bail!(
            "{node}: enumeration did not complete (no end marker); refusing to claim coverage"
        );
    }
    listing.metrics.sort_by(|a, b| a.key.cmp(&b.key));
    listing.results.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(listing)
}

/// Byte-range reader over ssh. `len()` is the length observed during
/// enumeration, so the steady state costs one round trip per changed segment
/// and zero for the rest.
pub struct RemoteSource {
    pub node: String,
    pub runs_root: String,
    pub relative: String,
    pub len: u64,
    pub deadline_secs: u64,
}

impl RemoteSource {
    pub fn display(&self) -> String {
        format!("{}:{}/{}", self.node, self.runs_root, self.relative)
    }
}

impl SourceReader for RemoteSource {
    fn len(&self) -> Result<u64> {
        Ok(self.len)
    }

    fn read_range(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        if !sane_root(&self.runs_root) || !self.relative.split('/').all(sane_component) {
            anyhow::bail!("refusing unsafe remote path {:?}", self.relative);
        }
        // `tail -c +N` is 1-based from the start of the file; `head -c` bounds
        // the transfer so only the new rows cross the wire.
        let remote = format!(
            "tail -c +{} -- {}/{} 2>/dev/null | head -c {len}",
            offset + 1,
            self.runs_root,
            self.relative
        );
        let bytes = run_ssh(&self.node, self.deadline_secs, &remote)?;
        // A short read means the transfer was cut, not that the file shrank:
        // these files are append-only, so the source cannot be shorter than the
        // length enumeration just reported. Failing here keeps a truncated ssh
        // stream from being mistaken for a rewritten stream (which would rotate)
        // or for a truncated tail (which would unblock pruning).
        let expected = (len as u64).min(self.len.saturating_sub(offset)) as usize;
        if bytes.len() < expected {
            anyhow::bail!(
                "{}: short remote read at {offset} ({} of {expected} bytes)",
                self.display(),
                bytes.len()
            );
        }
        Ok(bytes)
    }
}

/// Local file or remote stream, so one capture loop covers both.
pub enum AnySource {
    File(FileSource),
    Bytes(BytesSource),
    Remote(RemoteSource),
}

impl SourceReader for AnySource {
    fn len(&self) -> Result<u64> {
        match self {
            AnySource::File(inner) => inner.len(),
            AnySource::Bytes(inner) => inner.len(),
            AnySource::Remote(inner) => inner.len(),
        }
    }

    fn read_range(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        match self {
            AnySource::File(inner) => inner.read_range(offset, len),
            AnySource::Bytes(inner) => inner.read_range(offset, len),
            AnySource::Remote(inner) => inner.read_range(offset, len),
        }
    }
}

/// Bytes currently held for a segment sidecar, or 0 when it is not archived yet.
pub fn sidecar_bytes(archive: &Path, key: &SegmentKey, name: &str) -> u64 {
    fs::metadata(segment_dir(archive, key).join(name))
        .map(|m| m.len())
        .unwrap_or(0)
}

/// Per-node capture reachability, persisted so staleness survives a restart and
/// the health alarm can see it without doing its own ssh.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeStatus {
    #[serde(default)]
    pub node: String,
    #[serde(default)]
    pub last_attempt_ms: u64,
    #[serde(default)]
    pub last_success_ms: u64,
    #[serde(default)]
    pub consecutive_failures: u64,
    #[serde(default)]
    pub last_error: String,
    #[serde(default)]
    pub segments_seen: u64,
    #[serde(default)]
    pub reachable: bool,
}

pub fn load_nodes(archive: &Path) -> BTreeMap<String, NodeStatus> {
    let Ok(text) = fs::read_to_string(archive.join(NODES_FILE)) else {
        return BTreeMap::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return BTreeMap::new();
    };
    let mut out = BTreeMap::new();
    for entry in value
        .get("nodes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        if let Ok(status) = serde_json::from_value::<NodeStatus>(entry) {
            if !status.node.is_empty() {
                out.insert(status.node.clone(), status);
            }
        }
    }
    out
}

pub fn save_nodes(archive: &Path, nodes: &BTreeMap<String, NodeStatus>) -> Result<()> {
    let value = json!({
        "schema": NODES_SCHEMA,
        "generated_at_ms": now_ms(),
        "nodes": nodes.values().collect::<Vec<_>>(),
    });
    write_atomic(
        &archive.join(NODES_FILE),
        &serde_json::to_vec_pretty(&value)?,
    )
}

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
    let mut state = SegmentState::load(&seg_dir).unwrap_or_default();
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

    let mut state = SegmentState::load(&seg_dir).unwrap_or_default();
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
            if let Some(previous) = SegmentState::load(&seg_dir) {
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
    if let Some(mut state) = SegmentState::load(&seg_dir) {
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
    let Some(mut state) = SegmentState::load(&seg_dir) else {
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

/// One metrics file found under a run directory.
#[derive(Clone, Debug)]
pub struct FoundMetrics {
    pub key: SegmentKey,
    pub path: PathBuf,
    pub len: u64,
}

/// Split `<runs_root>/<run_id>/attempts/<attempt>/outputs/.../<watermark>/train_metrics.jsonl`
/// into a segment key. Returns None when the path is not shaped like a trainer
/// output (so junk can never create archive directories).
pub fn key_from_path(run_id: &str, path: &Path) -> Option<SegmentKey> {
    let watermark = path.parent()?.file_name()?.to_str()?.to_string();
    let mut attempt = None;
    let components: Vec<String> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
        .collect();
    for (index, part) in components.iter().enumerate() {
        if part == "attempts" {
            attempt = components.get(index + 1).cloned();
        }
    }
    let key = SegmentKey {
        run_id: run_id.to_string(),
        attempt: attempt.unwrap_or_else(|| "0".to_string()),
        watermark,
    };
    key.is_sane().then_some(key)
}

fn walk_metrics(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > WALK_DEPTH || out.len() > 512 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten().take(2048) {
        let path = entry.path();
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => walk_metrics(&path, depth + 1, out),
            Ok(kind)
                if kind.is_file()
                    && path.file_name().and_then(|n| n.to_str()) == Some(METRICS_FILE) =>
            {
                out.push(path);
            }
            _ => {}
        }
    }
}

/// Every metrics file under one run directory. Bounded walk, `attempts/` only.
pub fn scan_run_dir(runs_root: &Path, run_id: &str) -> Vec<FoundMetrics> {
    let mut paths = Vec::new();
    walk_metrics(&runs_root.join(run_id).join("attempts"), 0, &mut paths);
    let mut found = Vec::new();
    for path in paths {
        let Some(key) = key_from_path(run_id, &path) else {
            continue;
        };
        let len = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        found.push(FoundMetrics { key, path, len });
    }
    found.sort_by(|a, b| a.key.cmp(&b.key));
    found
}

/// Trainer run directory names under `runs_root`. One `getdents` sweep plus a
/// substring test — no `stat` per entry, because this root holds ~70k dirs.
pub fn discover_trainer_runs(runs_root: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in fs::read_dir(runs_root)?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.contains(TRAIN_MARKER) && sane_component(&name) {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

/// Load every segment currently in the archive, keyed for lookup.
pub fn load_segments(archive: &Path) -> BTreeMap<SegmentKey, SegmentState> {
    let mut out = BTreeMap::new();
    let runs = archive.join("runs");
    let Ok(run_entries) = fs::read_dir(&runs) else {
        return out;
    };
    for run in run_entries.flatten() {
        let run_id = run.file_name().to_string_lossy().into_owned();
        let Ok(attempts) = fs::read_dir(run.path()) else {
            continue;
        };
        for attempt in attempts.flatten() {
            let attempt_name = attempt.file_name().to_string_lossy().into_owned();
            let Ok(watermarks) = fs::read_dir(attempt.path()) else {
                continue;
            };
            for watermark in watermarks.flatten() {
                let key = SegmentKey {
                    run_id: run_id.clone(),
                    attempt: attempt_name.clone(),
                    watermark: watermark.file_name().to_string_lossy().into_owned(),
                };
                if let Some(state) = SegmentState::load(&watermark.path()) {
                    out.insert(key, state);
                }
            }
        }
    }
    out
}

/// How many bytes of a given segment the archive already holds. Read straight
/// off the archive file so a lost or stale state file cannot over-report.
pub fn archived_bytes(archive: &Path, key: &SegmentKey) -> u64 {
    fs::metadata(SegmentState::metrics_path(&segment_dir(archive, key)))
        .map(|m| m.len())
        .unwrap_or(0)
}

/// `name` -> (`sha256`, `source`) for every `[[input]]` block in a job spec.
fn spec_inputs(text: &str) -> BTreeMap<String, (String, String)> {
    let mut out = BTreeMap::new();
    for block in text.split("[[input]]").skip(1) {
        // Stop at the next table header so keys cannot leak across blocks.
        let body = block.split("\n[").next().unwrap_or(block);
        let field = |key: &str| -> Option<String> {
            body.lines()
                .find_map(|line| {
                    let (left, right) = line.split_once('=')?;
                    (left.trim() == key).then(|| right.trim().trim_matches('"').to_string())
                })
                .filter(|value| !value.is_empty())
        };
        if let Some(name) = field("name") {
            out.insert(
                name,
                (
                    field("sha256").unwrap_or_default(),
                    field("source").unwrap_or_default(),
                ),
            );
        }
    }
    out
}

/// Turn a `--warm` argument into a checkpoint sha256. The argument is normally
/// the placeholder `{input:checkpoint}`; a literal path is also accepted.
fn resolve_checkpoint(value: &str, inputs: &BTreeMap<String, (String, String)>) -> Option<String> {
    if let Some(name) = value
        .strip_prefix("{input:")
        .and_then(|v| v.strip_suffix('}'))
    {
        let (sha, source) = inputs.get(name)?;
        if !sha.is_empty() {
            return Some(sha.clone());
        }
        return Path::new(source)
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.trim_end_matches(".model.pt").to_string());
    }
    Path::new(value)
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.trim_end_matches(".model.pt").to_string())
        .filter(|n| !n.is_empty())
}

/// Lineage annotations pulled out of `specs/<run_id>.toml`: which recipe the run
/// used, which checkpoint it warm-started from, and which corpus snapshot (hence
/// which sampler) fed it. Without these a single curve across runs is a lie.
pub fn spec_annotations(root: &Path, run_id: &str) -> Map<String, Value> {
    let mut out = Map::new();
    let Ok(text) = fs::read_to_string(root.join("specs").join(format!("{run_id}.toml"))) else {
        return out;
    };
    // Command args reference inputs by placeholder (`--warm {input:checkpoint}`),
    // so the real checkpoint identity lives in the matching [[input]] block.
    let inputs = spec_inputs(&text);
    let tokens: Vec<&str> = text
        .split(|c: char| c.is_whitespace() || matches!(c, '"' | ',' | '[' | ']'))
        .filter(|t| !t.is_empty())
        .collect();
    for (index, token) in tokens.iter().enumerate() {
        let next = tokens.get(index + 1).copied();
        match *token {
            "--recipe" => {
                if let Some(value) = next {
                    out.insert("recipe".into(), json!(value));
                }
            }
            "--seed" => {
                if let Some(value) = next {
                    out.insert("seed".into(), json!(value));
                }
            }
            "--warm" => {
                if let Some(value) = next {
                    if let Some(sha) = resolve_checkpoint(value, &inputs) {
                        out.insert("warm_start".into(), json!(sha));
                    }
                }
            }
            _ => {}
        }
    }
    // The snapshot directory name encodes mode and sampler, e.g.
    // snapshot-001805437-full_consolidation-eq32v32c128.
    if let Some(start) = text.find("/training/snapshot-") {
        let rest = &text[start + "/training/".len()..];
        if let Some(end) = rest.find('/') {
            out.insert("corpus_snapshot".into(), json!(&rest[..end]));
        }
    }
    out
}

/// Final-selection facts a run publishes when it finishes.
fn result_annotations(seg_dir: &Path) -> Map<String, Value> {
    let mut out = Map::new();
    let Ok(text) = fs::read_to_string(seg_dir.join(RESULT_FILE)) else {
        return out;
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return out;
    };
    for key in [
        "checkpoint_sha256",
        "selected_checkpoint_epoch",
        "recipe",
        "epochs",
    ] {
        if let Some(found) = value.get(key) {
            if !found.is_null() {
                out.insert(format!("result_{key}"), found.clone());
            }
        }
    }
    out
}

/// Rebuild `index.v1.json` from the segments on disk. This is the file every
/// consumer (TUI, plotters, the alarm) reads; nothing downstream should ever
/// touch a run directory again.
pub fn rebuild_index(root: &Path, archive: &Path) -> Result<Value> {
    let segments = load_segments(archive);
    let mut rows = Vec::new();
    let mut total_rows = 0u64;
    let mut total_bytes = 0u64;
    let mut newest_advance = 0u64;
    let mut truncated_segments = 0u64;
    let mut truncated_bytes = 0u64;
    for (key, state) in &segments {
        let seg_dir = segment_dir(archive, key);
        let bytes = archived_bytes(archive, key);
        total_rows += state.archived_rows;
        total_bytes += bytes;
        newest_advance = newest_advance.max(state.last_advance_ms);
        if state.truncation_count > 0 {
            truncated_segments += 1;
            truncated_bytes += state.truncated_tail_bytes;
        }
        let mut row = json!({
            "run_id": key.run_id,
            "attempt": key.attempt,
            "watermark": key.watermark,
            "path": key.relative().join(METRICS_FILE).to_string_lossy(),
            "rows": state.archived_rows,
            "bytes": bytes,
            "first_epoch": state.first_epoch,
            "last_epoch": state.last_epoch,
            "first_seen_ms": state.first_seen_ms,
            "last_advance_ms": state.last_advance_ms,
            "last_check_ms": state.last_check_ms,
            "source_present": state.source_present,
            "source_path": state.source_path,
            "node": state.node,
            "rewrite_count": state.rewrite_count,
            "truncated_tail_bytes": state.truncated_tail_bytes,
            "truncation_count": state.truncation_count,
            "last_truncation_ms": state.last_truncation_ms,
            "head": state.head,
            "tail": state.tail,
        });
        let annotations = row.as_object_mut().expect("object");
        for (name, value) in spec_annotations(root, &key.run_id) {
            annotations.insert(name, value);
        }
        for (name, value) in result_annotations(&seg_dir) {
            annotations.insert(name, value);
        }
        rows.push(row);
    }
    // Oldest first: the index reads as the training history in order.
    rows.sort_by_key(|row| {
        (
            row["first_seen_ms"].as_u64().unwrap_or(0),
            row["run_id"].as_str().unwrap_or("").to_string(),
        )
    });
    let run_count = segments
        .keys()
        .map(|k| k.run_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let index = json!({
        "schema": INDEX_SCHEMA,
        "generated_at_ms": now_ms(),
        "archive_root": archive.to_string_lossy(),
        "run_count": run_count,
        "segment_count": rows.len(),
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "newest_advance_ms": newest_advance,
        // Holes, stated at the top of the record rather than buried: a
        // consumer that never opens a segment still sees them.
        "truncated_segments": truncated_segments,
        "truncated_tail_bytes": truncated_bytes,
        "nodes": load_nodes(archive).values().collect::<Vec<_>>(),
        "segments": rows,
    });
    write_atomic(
        &archive.join(INDEX_FILE),
        &serde_json::to_vec_pretty(&index)?,
    )?;
    Ok(index)
}

/// Advisory whole-archive lock. The capture timer takes it non-blocking and
/// skips; the collector takes it with retries and refuses to prune if it cannot
/// get it, because pruning without the ability to archive is the failure mode
/// this module exists to prevent.
pub struct CaptureLock {
    file: fs::File,
}

impl CaptureLock {
    pub fn acquire(archive: &Path, attempts: u32, backoff_ms: u64) -> Result<Option<Self>> {
        fs::create_dir_all(archive)?;
        let file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(archive.join(LOCK_FILE))?;
        for attempt in 0..attempts.max(1) {
            if file.try_lock().is_ok() {
                return Ok(Some(Self { file }));
            }
            if attempt + 1 < attempts.max(1) {
                std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
            }
        }
        Ok(None)
    }
}

impl Drop for CaptureLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn row(epoch: i64, train: f64, val: f64) -> String {
        format!(
            "{{\"epoch\":{epoch},\"phase\":\"A\",\"train_loss\":{train},\"val_loss\":{val},\"model_version\":\"policy_v3.3-fused-heads\"}}"
        )
    }

    fn key() -> SegmentKey {
        SegmentKey {
            run_id: "jope-selfplay-train-001805437-a-g9".into(),
            attempt: "1".into(),
            watermark: "watermark-001805437".into(),
        }
    }

    #[test]
    fn incremental_append_is_idempotent_and_whole_row_only() -> Result<()> {
        let temp = TempDir::new()?;
        let archive = temp.path().join("history");
        let source = temp.path().join(METRICS_FILE);

        fs::write(
            &source,
            format!("{}\n{}\n", row(1, 3.0, 3.5), row(2, 2.9, 3.4)),
        )?;
        let first = ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        assert_eq!(first.appended_rows, 2);
        assert_eq!(first.last_epoch, Some(2));
        assert!(first.covered());

        // Re-running with no new bytes must append nothing.
        let again = ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        assert_eq!(again.appended_rows, 0);
        assert_eq!(again.archived_rows, 2);

        // A row still being written (no trailing newline) is not archived, and
        // coverage correctly reports the source as ahead.
        let partial = format!(
            "{}\n{}\n{}",
            row(1, 3.0, 3.5),
            row(2, 2.9, 3.4),
            "{\"epoch\":3,\"train_lo"
        );
        fs::write(&source, &partial)?;
        let mid = ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        assert_eq!(mid.appended_rows, 0);
        assert!(!mid.covered());

        // Once the row completes it lands, and the archive is byte-identical.
        fs::write(
            &source,
            format!(
                "{}\n{}\n{}\n",
                row(1, 3.0, 3.5),
                row(2, 2.9, 3.4),
                row(3, 2.8, 3.3)
            ),
        )?;
        let done = ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        assert_eq!(done.appended_rows, 1);
        assert_eq!(done.last_epoch, Some(3));
        assert!(done.covered());
        let archived = fs::read(SegmentState::metrics_path(&segment_dir(&archive, &key())))?;
        assert_eq!(archived, fs::read(&source)?);
        Ok(())
    }

    #[test]
    fn a_replaced_source_rotates_instead_of_truncating() -> Result<()> {
        let temp = TempDir::new()?;
        let archive = temp.path().join("history");
        let source = temp.path().join(METRICS_FILE);
        fs::write(
            &source,
            format!("{}\n{}\n", row(1, 3.0, 3.5), row(2, 2.9, 3.4)),
        )?;
        ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;

        // A rerun restarts the file with different content.
        fs::write(&source, format!("{}\n", row(1, 9.9, 9.9)))?;
        let after = ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        assert!(after.rotated);
        assert_eq!(after.archived_rows, 1);

        let seg = segment_dir(&archive, &key());
        let rotated: Vec<_> = fs::read_dir(seg.join("rewrites"))?
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".jsonl"))
            .collect();
        assert_eq!(rotated.len(), 1, "old stream must be kept, not truncated");
        let preserved = fs::read_to_string(seg.join("rewrites").join(&rotated[0]))?;
        assert!(preserved.contains("\"epoch\":2"), "old rows survive");
        Ok(())
    }

    #[test]
    fn shrinking_source_also_rotates() -> Result<()> {
        let temp = TempDir::new()?;
        let archive = temp.path().join("history");
        let source = temp.path().join(METRICS_FILE);
        fs::write(
            &source,
            format!(
                "{}\n{}\n{}\n",
                row(1, 3.0, 3.5),
                row(2, 2.9, 3.4),
                row(3, 2.8, 3.3)
            ),
        )?;
        ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        fs::write(&source, "")?;
        let after = ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        assert!(after.rotated);
        assert_eq!(after.archived_rows, 0);
        // Coverage is trivially true against an empty source; the point is that
        // the three original rows are still on disk under rewrites/.
        assert!(after.covered());
        assert_eq!(
            fs::read_dir(segment_dir(&archive, &key()).join("rewrites"))?
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().ends_with(".jsonl"))
                .count(),
            1
        );
        Ok(())
    }

    #[test]
    fn a_lost_archive_file_is_re_ingested_without_double_counting() -> Result<()> {
        let temp = TempDir::new()?;
        let archive = temp.path().join("history");
        let source = temp.path().join(METRICS_FILE);
        fs::write(
            &source,
            format!(
                "{}\n{}\n{}\n",
                row(1, 3.0, 3.5),
                row(2, 2.9, 3.4),
                row(3, 2.8, 3.3)
            ),
        )?;
        ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;

        // Simulate a crash that lost the archive body but kept the state file:
        // the offset must fall back to the file, and the counters with it.
        let seg = segment_dir(&archive, &key());
        fs::remove_file(SegmentState::metrics_path(&seg))?;
        let stale = SegmentState::load(&seg).expect("state survived");
        assert_eq!(stale.archived_rows, 3);

        let after = ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        assert_eq!(after.archived_rows, 3, "rows recounted, not doubled");
        assert_eq!(after.last_epoch, Some(3));
        assert!(after.covered());
        assert_eq!(
            fs::read(SegmentState::metrics_path(&seg))?,
            fs::read(&source)?
        );
        Ok(())
    }

    #[test]
    fn a_partially_lost_archive_file_recounts_from_what_survived() -> Result<()> {
        let temp = TempDir::new()?;
        let archive = temp.path().join("history");
        let source = temp.path().join(METRICS_FILE);
        let body = format!(
            "{}\n{}\n{}\n",
            row(1, 3.0, 3.5),
            row(2, 2.9, 3.4),
            row(3, 2.8, 3.3)
        );
        fs::write(&source, &body)?;
        ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;

        // Truncate the archive to the first two rows behind our back.
        let seg = segment_dir(&archive, &key());
        let keep = body.find(&row(3, 2.8, 3.3)).unwrap() as u64;
        let file = fs::OpenOptions::new()
            .write(true)
            .open(SegmentState::metrics_path(&seg))?;
        file.set_len(keep)?;
        drop(file);

        let after = ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        assert_eq!(after.archived_rows, 3);
        assert_eq!(after.appended_rows, 1);
        assert_eq!(fs::read(SegmentState::metrics_path(&seg))?, body.as_bytes());
        Ok(())
    }

    #[test]
    fn coverage_tracks_the_source_length() -> Result<()> {
        let temp = TempDir::new()?;
        let archive = temp.path().join("history");
        let source = temp.path().join(METRICS_FILE);
        fs::write(&source, format!("{}\n", row(1, 3.0, 3.5)))?;
        let out = ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        assert!(out.covered());
        assert_eq!(
            archived_bytes(&archive, &key()),
            fs::metadata(&source)?.len()
        );
        Ok(())
    }

    #[test]
    fn unsafe_keys_are_refused() {
        let temp = TempDir::new().unwrap();
        let archive = temp.path().join("history");
        let bad = SegmentKey {
            run_id: "../../etc".into(),
            attempt: "0".into(),
            watermark: "w".into(),
        };
        assert!(ingest_source(&archive, &bad, &BytesSource(b"{}\n".to_vec()), "s", "n").is_err());
        assert!(!archive.join("runs").exists());
    }

    #[test]
    fn key_from_path_extracts_attempt_and_watermark() {
        let path = Path::new(
            "/runs/jope-selfplay-train-001805437-a-g9/attempts/1/outputs/training/watermark-001805437/train_metrics.jsonl",
        );
        let key = key_from_path("jope-selfplay-train-001805437-a-g9", path).unwrap();
        assert_eq!(key.attempt, "1");
        assert_eq!(key.watermark, "watermark-001805437");
    }

    #[test]
    fn scan_and_index_round_trip() -> Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        let run_id = "jope-selfplay-train-001805437-a-g9";
        let outputs = runs
            .join(run_id)
            .join("attempts/1/outputs/training/watermark-001805437");
        fs::create_dir_all(&outputs)?;
        fs::write(
            outputs.join(METRICS_FILE),
            format!("{}\n", row(7, 2.5, 3.6)),
        )?;
        fs::create_dir_all(root.join("specs"))?;
        // Shaped like a real spec: the warm checkpoint is a placeholder whose
        // identity only appears in the matching [[input]] block.
        fs::write(
            root.join("specs").join(format!("{run_id}.toml")),
            format!(
                concat!(
                    "args = [\"--recipe\", \"c\", \"--warm\", \"{{input:checkpoint}}\"]\n",
                    "\n[[input]]\nname = \"train\"\n",
                    "source = \"{flywheel}/training/snapshot-001805437-full_consolidation-eq32v32c128/corpus.train.jsonl\"\n",
                    "sha256 = \"d124ed7b\"\n",
                    "\n[[input]]\nname = \"checkpoint\"\n",
                    "source = \"/x/attempts/0/outputs/training/w/d36274de.model.pt\"\n",
                    "sha256 = \"d36274de99ddf3b12dffd534145b42bfcace1bbd6230bd65cc37288efe3a3bf3\"\n",
                ),
                flywheel = crate::paths::flywheel_root().display()
            ),
        )?;

        let found = scan_run_dir(&runs, run_id);
        assert_eq!(found.len(), 1);
        let archive = archive_root(&root);
        ingest_source(
            &archive,
            &found[0].key,
            &FileSource(found[0].path.clone()),
            &found[0].path.to_string_lossy(),
            "xbabe2",
        )?;
        let index = rebuild_index(&root, &archive)?;
        assert_eq!(index["schema"], INDEX_SCHEMA);
        assert_eq!(index["total_rows"], 1);
        let segment = &index["segments"][0];
        assert_eq!(segment["run_id"], run_id);
        assert_eq!(segment["last_epoch"], 7);
        assert_eq!(segment["recipe"], "c");
        assert_eq!(
            segment["warm_start"],
            "d36274de99ddf3b12dffd534145b42bfcace1bbd6230bd65cc37288efe3a3bf3"
        );
        assert_eq!(
            segment["corpus_snapshot"],
            "snapshot-001805437-full_consolidation-eq32v32c128"
        );
        assert_eq!(segment["tail"]["val_loss"], 3.6);
        Ok(())
    }

    #[test]
    fn discover_only_matches_trainer_dirs() -> Result<()> {
        let temp = TempDir::new()?;
        fs::create_dir_all(temp.path().join("jope-selfplay-train-001805437-c"))?;
        fs::create_dir_all(temp.path().join("jte-abc"))?;
        fs::create_dir_all(temp.path().join("jope-testonly-train-000090066"))?;
        let names = discover_trainer_runs(temp.path())?;
        assert_eq!(names.len(), 2);
        assert!(names.iter().all(|n| n.contains(TRAIN_MARKER)));
        Ok(())
    }

    #[test]
    fn sealing_marks_the_source_gone_without_touching_rows() -> Result<()> {
        let temp = TempDir::new()?;
        let archive = temp.path().join("history");
        let source = temp.path().join(METRICS_FILE);
        fs::write(&source, format!("{}\n", row(1, 3.0, 3.5)))?;
        ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        let before = archived_bytes(&archive, &key());
        seal_segment(&archive, &key())?;
        let state = SegmentState::load(&segment_dir(&archive, &key())).unwrap();
        assert!(!state.source_present);
        assert_eq!(archived_bytes(&archive, &key()), before);
        Ok(())
    }

    #[test]
    fn residual_tells_a_partial_row_from_missing_complete_rows() -> Result<()> {
        // Ends on a row boundary and fully archived: nothing outstanding.
        let full = format!("{}\n{}\n", row(1, 3.0, 3.5), row(2, 2.9, 3.4));
        let source = BytesSource(full.clone().into_bytes());
        assert_eq!(
            classify_residual(&source, full.len() as u64, full.len() as u64)?,
            Residual::None
        );

        // One unterminated fragment past the archive head: a row that was never
        // finished. This is the only shape the policy writes off.
        let partial = format!("{full}{{\"epoch\":3,\"train_lo");
        let source = BytesSource(partial.clone().into_bytes());
        let verdict = classify_residual(&source, full.len() as u64, partial.len() as u64)?;
        assert_eq!(
            verdict,
            Residual::PartialRow(b"{\"epoch\":3,\"train_lo".to_vec())
        );

        // A shortfall spanning a newline is COMPLETE rows that are not archived.
        // Writing this off would lose data, so it must be refused.
        let verdict = classify_residual(&source, 0, partial.len() as u64)?;
        assert!(matches!(verdict, Residual::CompleteRowsMissing { .. }));

        // Unterminated but far too big to be one row: also refused.
        let huge = vec![b'x'; MAX_TRUNCATED_TAIL_BYTES as usize + 10];
        let source = BytesSource(huge.clone());
        assert!(matches!(
            classify_residual(&source, 0, huge.len() as u64)?,
            Residual::OversizedFragment { .. }
        ));
        Ok(())
    }

    #[test]
    fn a_live_tail_is_not_abandoned_but_a_quiescent_one_is() {
        let now = 10_000_000u64;
        assert!(
            !tail_is_abandoned(now - 60_000, now),
            "1 min old: mid-write"
        );
        assert!(
            !tail_is_abandoned(0, now),
            "unknown mtime must never unblock pruning"
        );
        assert!(
            !tail_is_abandoned(now + 3_600_000, now),
            "an mtime in the future is a broken clock, not an old file"
        );
        assert!(tail_is_abandoned(
            now - (TAIL_QUIESCENT_SECS + 60) * 1000,
            now
        ));
    }

    #[test]
    fn remote_quiescence_is_measured_on_the_local_clock_only() {
        let now = 10_000_000u64;
        let stable = SegmentState {
            source_fingerprint: "1383:1".into(),
            // Seen unchanged here for longer than the threshold.
            fingerprint_since_ms: now - (TAIL_QUIESCENT_SECS + 60) * 1000,
            ..Default::default()
        };
        assert!(remote_tail_is_abandoned(
            Some(&stable),
            "1383:1",
            now,
            TAIL_QUIESCENT_SECS
        ));
        // The remote mtime says the file is ancient, but we have only just
        // started watching it: not abandoned. A wrong remote clock cannot make
        // us write off a row that is still being written.
        let fresh = SegmentState {
            source_fingerprint: "1383:1".into(),
            fingerprint_since_ms: now - 5_000,
            ..Default::default()
        };
        assert!(!remote_tail_is_abandoned(
            Some(&fresh),
            "1383:1",
            now,
            TAIL_QUIESCENT_SECS
        ));
        // The file changed since we last looked: definitively live.
        assert!(!remote_tail_is_abandoned(
            Some(&stable),
            "1500:2",
            now,
            TAIL_QUIESCENT_SECS
        ));
        // Never seen before: nothing to claim.
        assert!(!remote_tail_is_abandoned(
            None,
            "1383:1",
            now,
            TAIL_QUIESCENT_SECS
        ));
    }

    #[test]
    fn recording_a_truncation_keeps_the_fragment_and_is_idempotent() -> Result<()> {
        let temp = TempDir::new()?;
        let archive = temp.path().join("history");
        let source = temp.path().join(METRICS_FILE);
        let complete = format!("{}\n{}\n", row(1, 3.0, 3.5), row(2, 2.9, 3.4));
        let fragment = b"{\"epoch\":3,\"train_lo";
        fs::write(
            &source,
            format!("{complete}{}", String::from_utf8_lossy(fragment)),
        )?;

        let outcome = ingest_source(
            &archive,
            &key(),
            &FileSource(source.clone()),
            "src",
            "xbabe2",
        )?;
        assert_eq!(outcome.appended_rows, 2, "every COMPLETE row is archived");
        assert_eq!(outcome.residual_bytes, fragment.len() as u64);
        assert!(!outcome.covered(), "strict coverage is still false");

        let mtime = now_ms() - (TAIL_QUIESCENT_SECS + 60) * 1000;
        assert!(record_truncation(
            &archive,
            &key(),
            "xbabe2",
            "src",
            fragment,
            outcome.source_bytes,
            mtime
        )?);
        // Re-recording the same hole must not grow the ledger.
        assert!(!record_truncation(
            &archive,
            &key(),
            "xbabe2",
            "src",
            fragment,
            outcome.source_bytes,
            mtime
        )?);

        let seg = segment_dir(&archive, &key());
        let state = SegmentState::load(&seg).expect("state");
        assert_eq!(state.truncation_count, 1);
        assert_eq!(state.truncated_tail_bytes, fragment.len() as u64);
        assert_eq!(state.archived_rows, 2, "complete rows untouched");

        // The ledger exists per-segment and archive-wide, one line each.
        let seg_ledger = fs::read_to_string(seg.join(TRUNCATION_FILE))?;
        let all_ledger = fs::read_to_string(archive.join(TRUNCATIONS_FILE))?;
        assert_eq!(seg_ledger.lines().count(), 1);
        assert_eq!(all_ledger.lines().count(), 1);
        let record: Value = serde_json::from_str(all_ledger.lines().next().unwrap())?;
        assert_eq!(record["schema"], TRUNCATION_SCHEMA);
        assert_eq!(record["residual_bytes"], fragment.len());
        assert_eq!(record["archived_rows"], 2);
        assert_eq!(record["last_archived_epoch"], 2);

        // The fragment itself is kept verbatim: it is not counted as a row, but
        // it is not thrown away either.
        let evidence = record["evidence_file"].as_str().expect("evidence");
        assert_eq!(fs::read(seg.join(evidence))?, fragment.to_vec());

        // ...and the index says the record has a hole.
        let index = rebuild_index(temp.path(), &archive)?;
        assert_eq!(index["truncated_segments"], 1);
        assert_eq!(index["truncated_tail_bytes"], fragment.len());
        assert_eq!(index["segments"][0]["truncation_count"], 1);
        Ok(())
    }

    #[test]
    fn ssh_argv_never_targets_the_local_host_and_knows_xbabe4s_port() {
        assert!(
            ssh_argv(LOCAL_NODE, 30).is_err(),
            "ssh xbabe2 is a Mac Studio"
        );
        assert!(ssh_argv("nope", 30).is_err());
        let one = ssh_argv("xbabe1", 30).unwrap();
        assert_eq!(one[0], "timeout");
        assert_eq!(one.last().unwrap(), "xbabe1");
        assert!(one.iter().any(|a| a == "BatchMode=yes"));
        let four = ssh_argv("xbabe4", 30).unwrap();
        let port = four.iter().position(|a| a == "-p").expect("-p");
        assert_eq!(four[port + 1], "2224");
        assert!(REMOTE_NODES.iter().all(|node| *node != LOCAL_NODE));
    }

    #[test]
    fn remote_paths_that_could_reach_a_shell_are_refused() {
        let source = RemoteSource {
            node: "xbabe1".into(),
            runs_root: jope_paths::to_string(&crate::paths::scq_runs_root()),
            relative: "run/../../etc/shadow".into(),
            len: 10,
            deadline_secs: 5,
        };
        assert!(source.read_range(0, 10).is_err());
        let source = RemoteSource {
            node: "xbabe1".into(),
            runs_root: "/runs; rm -rf /".into(),
            relative: "a/b".into(),
            len: 10,
            deadline_secs: 5,
        };
        assert!(source.read_range(0, 10).is_err());
        assert!(remote_scan("xbabe1", "/runs; rm -rf /", 5).is_err());
    }

    #[test]
    fn node_status_round_trips() -> Result<()> {
        let temp = TempDir::new()?;
        let archive = temp.path().join("history");
        fs::create_dir_all(&archive)?;
        let mut nodes = BTreeMap::new();
        nodes.insert(
            "xbabe1".to_string(),
            NodeStatus {
                node: "xbabe1".into(),
                last_attempt_ms: 5,
                last_success_ms: 4,
                consecutive_failures: 2,
                last_error: "boom".into(),
                segments_seen: 3,
                reachable: false,
            },
        );
        save_nodes(&archive, &nodes)?;
        let back = load_nodes(&archive);
        assert_eq!(back["xbabe1"].consecutive_failures, 2);
        assert_eq!(back["xbabe1"].last_error, "boom");
        assert!(!back["xbabe1"].reachable);
        Ok(())
    }

    #[test]
    fn lock_is_exclusive_and_released_on_drop() -> Result<()> {
        let temp = TempDir::new()?;
        let archive = temp.path().join("history");
        let held = CaptureLock::acquire(&archive, 1, 0)?;
        assert!(held.is_some());
        // A second open file description cannot take it while the first holds.
        assert!(CaptureLock::acquire(&archive, 1, 0)?.is_none());
        drop(held);
        assert!(CaptureLock::acquire(&archive, 1, 0)?.is_some());
        Ok(())
    }
}
