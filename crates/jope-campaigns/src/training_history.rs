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
    fn initial() -> Self {
        Self::default()
    }

    fn path(seg_dir: &Path) -> PathBuf {
        seg_dir.join("segment.v1.json")
    }

    pub fn load(seg_dir: &Path) -> Result<Option<Self>> {
        let path = Self::path(seg_dir);
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error).with_context(|| format!("read segment state {}", path.display()))
            }
        };
        serde_json::from_str(&text)
            .map(Some)
            .with_context(|| format!("parse segment state {}", path.display()))
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

mod ingest;
pub use ingest::*;

mod index;
pub use index::*;

#[cfg(test)]
mod tests;
