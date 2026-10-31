use anyhow::{ensure, Context, Result};
use jope_narrator::{read_commit, Cursor, Envelope, JournalSummary};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, Metadata},
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

/// Maximum encoded bytes in one journal record.
pub const MAX_RECORD_BYTES: usize = 1024 * 1024;
/// Maximum encoded bytes in one returned narration page.
pub const MAX_PAGE_BYTES: usize = 4 * 1024 * 1024;
/// Maximum events in one returned narration page.
pub const MAX_PAGE_RECORDS: usize = 256;
const PAGE_OVERHEAD_BYTES: usize = 8192;
pub(super) const MIN_PAGE_BYTES: usize = MAX_RECORD_BYTES + 1 + PAGE_OVERHEAD_BYTES;

#[derive(Clone, Debug)]
/// Resource limits applied when opening and extending a journal index.
pub struct Limits {
    /// Maximum indexed event count, from 1 through 1,000,000.
    pub max_records: usize,
    /// Maximum journal file size, from 1 byte through 16 GiB.
    pub max_journal_bytes: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_records: 1_000_000,
            max_journal_bytes: 1024 * 1024 * 1024,
        }
    }
}
struct Entry {
    offset: u64,
    length: usize,
    sha256: String,
}

/// A fixed operator-selected journal. An incremental offset index is shared by clients.
/// Complete records are validated on ingestion and again on page reads. No whole-log
/// allocation or per-client prefix scan. Append-only violations require operator repair.
pub struct Index {
    path: PathBuf,
    pub(super) run_id: String,
    reader: BufReader<File>,
    opened_metadata: Metadata,
    entries: Vec<Entry>,
    offset: u64,
    partial_tail: bool,
    limits: Limits,
    poisoned: bool,
    capture: JournalSummary,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// One validated, bounded page of retained narration events.
pub struct Page {
    /// Page schema identifier.
    pub schema: String,
    /// Selected run identity.
    pub run_id: String,
    /// Retained events in sequence order.
    pub events: Vec<Envelope>,
    /// Exclusive cursor after the final returned event.
    pub next: Cursor,
    /// Whether more committed events remain after this page.
    pub more: bool,
    /// Whether the journal has an uncommitted trailing partial record.
    pub partial_tail: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Validated boundary and capture state for a selected narration journal.
pub struct Status {
    /// Status schema identifier.
    pub schema: String,
    /// Selected run identity.
    pub run_id: String,
    /// Last committed event sequence as a decimal string.
    pub last_sequence: String,
    /// Number of indexed journal bytes as a decimal string.
    pub indexed_bytes: String,
    /// Current journal file size as a decimal string.
    pub observed_bytes: String,
    /// Whether uncommitted trailing bytes are present.
    pub partial_tail: bool,
    /// Reader mode, currently `read-only`.
    pub mode: String,
    /// Durable producer capture watermark summary.
    pub capture: JournalSummary,
}

/// Parse a canonical unsigned decimal cursor sequence.
pub fn sequence(value: &str) -> Result<u64> {
    ensure!(
        !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()),
        "invalid cursor sequence"
    );
    ensure!(
        value == "0" || !value.starts_with('0'),
        "noncanonical cursor sequence"
    );
    value.parse().context("cursor sequence overflow")
}
fn complete_line(reader: &mut BufReader<File>, limit: usize) -> Result<Option<Vec<u8>>> {
    let start = reader.stream_position()?;
    let mut bytes = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            reader.seek(SeekFrom::Start(start))?;
            return Ok(None);
        }
        let count = available
            .iter()
            .position(|b| *b == b'\n')
            .map_or(available.len(), |n| n + 1);
        ensure!(
            bytes.len().saturating_add(count) <= limit,
            "journal record exceeds limit"
        );
        let complete = available[count - 1] == b'\n';
        bytes.extend_from_slice(&available[..count]);
        reader.consume(count);
        if complete {
            return Ok(Some(bytes));
        }
    }
}
#[cfg(unix)]
fn same_file(a: &Metadata, b: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.dev() == b.dev() && a.ino() == b.ino()
}
#[cfg(not(unix))]
fn same_file(a: &Metadata, b: &Metadata) -> bool {
    a.created().ok() == b.created().ok()
}

impl Index {
    /// Open and validate the committed prefix of one operator-selected journal.
    pub fn open(path: &Path, run_id: &str, limits: Limits) -> Result<Self> {
        ensure!(
            !run_id.is_empty() && run_id.len() <= 256 && !run_id.chars().any(char::is_control),
            "invalid run identity"
        );
        ensure!(
            (1..=1_000_000).contains(&limits.max_records),
            "invalid record cap"
        );
        ensure!(
            (1..=16 * 1024 * 1024 * 1024).contains(&limits.max_journal_bytes),
            "invalid byte cap"
        );
        let metadata = fs::symlink_metadata(path).context("journal metadata")?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "journal must be a regular file"
        );
        let file = File::open(path).context("open journal")?;
        ensure!(
            same_file(&metadata, &file.metadata()?),
            "journal replaced while opening"
        );
        let mut result = Self {
            path: path.to_owned(),
            run_id: run_id.into(),
            reader: BufReader::new(file),
            capture: read_commit(path)?,
            opened_metadata: metadata,
            entries: Vec::new(),
            offset: 0,
            partial_tail: false,
            limits,
            poisoned: false,
        };
        result.refresh()?;
        Ok(result)
    }
    /// Index newly committed records and return the number added.
    pub fn refresh(&mut self) -> Result<usize> {
        ensure!(
            !self.poisoned,
            "journal integrity failure; reopen after operator repair"
        );
        let result = self.refresh_inner();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn refresh_inner(&mut self) -> Result<usize> {
        // A committed prefix is data-synced before its marker is published. Read
        // that marker before stat: reversing these reads races an ordinary append.
        let capture = read_commit(&self.path)?;
        let metadata = fs::symlink_metadata(&self.path)?;
        ensure!(
            metadata.is_file()
                && !metadata.file_type().is_symlink()
                && same_file(&self.opened_metadata, &metadata),
            "journal replaced"
        );
        ensure!(metadata.len() >= self.offset, "journal truncated");
        ensure!(
            metadata.len() <= self.limits.max_journal_bytes,
            "journal byte cap reached"
        );
        ensure!(capture.run_id == self.run_id, "watermark run mismatch");
        let committed_bytes = sequence(&capture.bytes)?;
        let committed_records = sequence(&capture.records)?;
        ensure!(
            committed_bytes >= self.offset && committed_bytes <= metadata.len(),
            "watermark outside journal boundary"
        );
        ensure!(
            committed_records >= self.entries.len() as u64,
            "watermark regressed"
        );
        self.capture = capture;
        self.reader.seek(SeekFrom::Start(self.offset))?;
        self.partial_tail = metadata.len() > committed_bytes;
        let mut added = 0;
        // Drain this fixed, quota-bounded watermark before validating cursors or
        // reporting capture closure. Do not chase markers published during this read.
        while self.offset < committed_bytes {
            let start = self.offset;
            let Some(bytes) = complete_line(
                &mut self.reader,
                (committed_bytes - self.offset).min((MAX_RECORD_BYTES + 1) as u64) as usize,
            )?
            else {
                anyhow::bail!("committed record lacks newline");
            };
            ensure!(
                self.entries.len() < self.limits.max_records,
                "journal record cap reached"
            );
            let event: Envelope =
                serde_json::from_slice(&bytes).context("invalid complete journal record")?;
            let previous = self.entries.last().map_or("", |e| e.sha256.as_str());
            event.validate(previous, self.entries.len() as u64 + 1, &self.run_id)?;
            self.offset = self.reader.stream_position()?;
            self.entries.push(Entry {
                offset: start,
                length: bytes.len(),
                sha256: event.sha256.clone(),
            });
            added += 1;
        }
        if self.offset == committed_bytes {
            ensure!(
                self.entries.len() as u64 == committed_records,
                "watermark record count mismatch"
            );
            ensure!(
                self.entries.last().map_or("", |e| e.sha256.as_str()) == self.capture.sha256,
                "watermark hash mismatch"
            );
        }
        Ok(added)
    }
    /// Return the exclusive cursor at the current indexed boundary.
    pub fn cursor(&self) -> Cursor {
        Cursor {
            run_id: self.run_id.clone(),
            sequence: self.entries.len().to_string(),
        }
    }
    pub(super) fn checked_cursor(&self, after: &Cursor) -> Result<usize> {
        ensure!(after.run_id == self.run_id, "cursor belongs to another run");
        let n = sequence(&after.sequence)?;
        ensure!(n <= self.entries.len() as u64, "future cursor");
        usize::try_from(n).context("cursor range")
    }
    /// Refresh the journal and return its validated status summary.
    pub fn status(&mut self) -> Result<Status> {
        self.refresh()?;
        Ok(Status {
            schema: "jope.narration-status/v1".into(),
            run_id: self.run_id.clone(),
            last_sequence: self.entries.len().to_string(),
            indexed_bytes: self.offset.to_string(),
            observed_bytes: self.reader.get_ref().metadata()?.len().to_string(),
            partial_tail: self.partial_tail,
            mode: "read-only".into(),
            capture: self.capture.clone(),
        })
    }
    /// Return a bounded page after the exclusive cursor, validating records on read.
    pub fn page(&mut self, after: &Cursor, limit: usize, max_bytes: usize) -> Result<Page> {
        ensure!(
            (1..=MAX_PAGE_RECORDS).contains(&limit),
            "page record limit out of range"
        );
        ensure!(
            (MIN_PAGE_BYTES..=MAX_PAGE_BYTES).contains(&max_bytes),
            "page byte limit out of range"
        );
        self.refresh()?;
        let start = self.checked_cursor(after)?;
        let mut events = Vec::new();
        let mut total = PAGE_OVERHEAD_BYTES;
        for index in start..self.entries.len().min(start + limit) {
            let entry = &self.entries[index];
            if total + entry.length > max_bytes {
                break;
            }
            self.reader.seek(SeekFrom::Start(entry.offset))?;
            let mut bytes = vec![0; entry.length];
            self.reader.read_exact(&mut bytes)?;
            let event: Envelope =
                serde_json::from_slice(&bytes).context("retained journal record changed")?;
            let previous = if index == 0 {
                ""
            } else {
                &self.entries[index - 1].sha256
            };
            event.validate(previous, index as u64 + 1, &self.run_id)?;
            ensure!(
                event.sha256 == entry.sha256,
                "retained journal record changed"
            );
            total += entry.length;
            events.push(event);
        }
        let end = start + events.len();
        Ok(Page {
            schema: "jope.narration-page/v1".into(),
            run_id: self.run_id.clone(),
            events,
            next: Cursor {
                run_id: self.run_id.clone(),
                sequence: end.to_string(),
            },
            more: end < self.entries.len(),
            partial_tail: self.partial_tail,
        })
    }
}
