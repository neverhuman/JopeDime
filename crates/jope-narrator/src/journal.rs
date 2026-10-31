//! Create-only, hash-chained durable narration. A queued record is not a durable
//! acknowledgement. `finish` closes admission, drains accepted records and reports sync errors.
use crate::{
    decimal_u64, digest,
    render::Renderer,
    tap::{channel_with, validate_fact, validate_links},
    Limits, LineageFact, Links, Narration, Tap, MAX_FRAME_BYTES,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    thread::{self, JoinHandle},
    time::{SystemTime, UNIX_EPOCH},
};

pub const SCHEMA: &str = "jope.reasoning-event/v1";
pub const DIR_ENV: &str = "JOPE_NARRATION_DIR";
/// Bounds automatic per-call captures in a long-lived embedding process.
/// Explicit `FactJournal::create` remains controlled by its caller.
pub const MAX_ENV_SESSIONS: u64 = 64;
static ENV_SESSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ENV_LIMIT_REPORTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cursor {
    /// Identifier of the journal run containing the record.
    pub run_id: String,
    /// Canonical decimal sequence number within the run, starting at one.
    pub sequence: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    /// Versioned record schema (`jope.reasoning-event/v1`).
    pub schema: String,
    /// Versioned renderer that produced `narration`.
    pub renderer: String,
    /// Identifier shared by all records in this journal.
    pub run_id: String,
    /// Canonical decimal record sequence, starting at one.
    pub sequence: String,
    /// Run and sequence pair suitable for paging.
    pub cursor: Cursor,
    /// Search-owned fact captured by this record.
    pub fact: LineageFact,
    /// Optional source identifiers linking the fact to other records.
    pub links: Links,
    /// Bounded commentary derived from `fact` and `links`.
    pub narration: Narration,
    /// Hash-chain digest of the immediately preceding record; empty for the first record.
    pub previous_sha256: String,
    /// Lowercase SHA-256 digest of this record with this field empty.
    pub sha256: String,
}
impl Envelope {
    fn computed_hash(&self) -> Result<String> {
        let mut value = self.clone();
        value.sha256.clear();
        Ok(digest(&crate::encode_bounded(&value, MAX_FRAME_BYTES * 8)?))
    }
    pub fn validate(
        &self,
        previous_sha256: &str,
        expected_sequence: u64,
        run_id: &str,
    ) -> Result<()> {
        ensure!(self.schema == SCHEMA, "unsupported narration schema");
        ensure!(
            !run_id.is_empty()
                && run_id.len() <= 256
                && self.run_id == run_id
                && self.cursor.run_id == run_id,
            "cursor run mismatch"
        );
        ensure!(
            decimal_u64(&self.sequence)? == expected_sequence
                && expected_sequence > 0
                && self.cursor.sequence == self.sequence,
            "nonmonotonic sequence"
        );
        ensure!(
            self.previous_sha256 == previous_sha256,
            "journal hash chain mismatch"
        );
        validate_fact(&self.fact)?;
        validate_links(&self.links)?;
        ensure!(
            self.narration.claims.iter().all(|c| !c.evidence.is_empty()),
            "claim lacks evidence"
        );
        ensure!(
            !self.narration.text.chars().any(char::is_control),
            "unsafe rendered control character"
        );
        let evidence = format!("{run_id}:{expected_sequence}");
        let expected = crate::render::rederive(
            &self.renderer,
            &self.fact,
            &self.links,
            &evidence,
            self.narration.variant,
        )?;
        if self.narration != expected {
            let frame = self
                .narration
                .claims
                .first()
                .and_then(|c| c.details.get("max_frame_bytes"))
                .and_then(serde_json::Value::as_u64)
                .context("unsupported narrative derivation")? as usize;
            ensure!(
                (4096..=MAX_FRAME_BYTES).contains(&frame),
                "invalid fallback bound"
            );
            ensure!(
                self.narration == bounded_narration(&evidence, frame, expected.variant),
                "narrative claims differ from source facts"
            );
            let mut complete = self.clone();
            complete.narration = expected;
            ensure!(
                crate::encode_bounded(&complete, frame).is_err(),
                "unnecessary detail refusal"
            );
        }
        crate::encode_bounded(self, MAX_FRAME_BYTES)?;
        ensure!(
            self.computed_hash()? == self.sha256,
            "journal content hash mismatch"
        );
        Ok(())
    }
}

/// A durable prefix, published only after the journal data and this marker are synced.
/// `complete` means capture drained without loss, independently of the source run's outcome.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JournalSummary {
    /// Identifier of the journal run.
    pub run_id: String,
    /// Canonical decimal count of durable records.
    pub records: String,
    /// Canonical decimal count of facts dropped before journaling.
    pub dropped: String,
    /// Hash-chain digest of the last durable record; empty when there are none.
    pub sha256: String,
    /// Canonical decimal byte length of the journal file.
    pub bytes: String,
    /// Capture closure state: `capturing`, `complete`, `incomplete`, `resource_limit`, or `failed`.
    pub status: String,
    /// Stable error code when capture did not complete normally.
    pub error: Option<String>,
}
pub fn commit_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".commit");
    PathBuf::from(name)
}
fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}
fn publish_commit(path: &Path, state: &JournalSummary, initial: bool) -> Result<()> {
    let marker = commit_path(path);
    let mut pending = marker.as_os_str().to_os_string();
    pending.push(".pending");
    let pending = PathBuf::from(pending);
    // This fixed pending name is owned by the sole writer. A crash leaves it for inspection;
    // no subsequent create-new session can overwrite the original journal.
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&pending)?;
    file.write_all(&crate::encode_bounded(state, 4096)?)?;
    file.sync_all()?;
    if initial {
        // Never overwrite an orphaned marker from an earlier capture.
        fs::hard_link(&pending, &marker)?;
        fs::remove_file(&pending)?;
    } else {
        fs::rename(&pending, &marker)?;
    }
    File::open(parent_dir(path))?.sync_all()?;
    Ok(())
}
pub fn read_commit(path: &Path) -> Result<JournalSummary> {
    use std::io::Read;
    let marker = commit_path(path);
    let meta = fs::symlink_metadata(&marker).context("missing durable narration watermark")?;
    ensure!(
        meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= 4096,
        "invalid narration watermark"
    );
    let mut bytes = Vec::new();
    File::open(marker)?.take(4097).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 4096, "oversize narration watermark");
    let state: JournalSummary = serde_json::from_slice(&bytes)?;
    let records = decimal_u64(&state.records)?;
    decimal_u64(&state.bytes)?;
    decimal_u64(&state.dropped)?;
    ensure!(
        !state.run_id.is_empty() && state.run_id.len() <= 256,
        "invalid watermark run"
    );
    ensure!(
        (records == 0 && state.sha256.is_empty())
            || (records > 0 && crate::is_sha256(&state.sha256)),
        "invalid watermark digest"
    );
    ensure!(
        matches!(
            state.status.as_str(),
            "capturing" | "complete" | "incomplete" | "failed" | "resource_limit"
        ),
        "invalid capture status"
    );
    Ok(state)
}
pub struct FactJournal {
    path: PathBuf,
    tap: Tap,
    handle: Option<JoinHandle<Result<JournalSummary>>>,
}
impl FactJournal {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn create(path: &Path, run_id: &str, limits: Limits) -> Result<(Tap, Self)> {
        limits.validate()?;
        ensure!(
            !run_id.is_empty() && run_id.len() <= 256 && !run_id.chars().any(char::is_control),
            "invalid run ID"
        );
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create narration journal {}", path.display()))?;
        file.sync_all()?;
        File::open(parent_dir(path))?.sync_all()?;
        let mut state = JournalSummary {
            run_id: run_id.into(),
            records: "0".into(),
            dropped: "0".into(),
            sha256: String::new(),
            bytes: "0".into(),
            status: "capturing".into(),
            error: None,
        };
        publish_commit(path, &state, true)?;
        let max_frame = limits.max_frame_bytes;
        let max_records = limits.max_records;
        let max_bytes = limits.max_journal_bytes;
        let (tap, mut receiver) = channel_with(limits);
        let writer_path = path.to_owned();
        let handle = thread::Builder::new()
            .name("jope-narration-journal".into())
            .spawn(move || -> Result<JournalSummary> {
                let mut file = file;
                let mut renderer = Renderer::new();
                let mut staged = state.clone();
                let mut batch = 0;
                let mut batch_start = std::time::Instant::now();
                let result = (|| -> Result<()> {
                    while let Some((fact, links)) = receiver.recv_linked()? {
                        let sequence = decimal_u64(&staged.records)?
                            .checked_add(1)
                            .context("narration sequence exhausted")?;
                        let evidence = format!("{}:{sequence}", staged.run_id);
                        let narration = renderer.render(&fact, &links, &evidence);
                        let seq = sequence.to_string();
                        let mut record = Envelope {
                            schema: SCHEMA.into(),
                            renderer: crate::render::RENDERER.into(),
                            run_id: staged.run_id.clone(),
                            sequence: seq.clone(),
                            cursor: Cursor {
                                run_id: staged.run_id.clone(),
                                sequence: seq,
                            },
                            fact,
                            links,
                            narration,
                            previous_sha256: staged.sha256.clone(),
                            sha256: String::new(),
                        };
                        if crate::encode_bounded(&record, max_frame.saturating_sub(64)).is_err() {
                            record.narration =
                                bounded_narration(&evidence, max_frame, record.narration.variant);
                        }
                        record.sha256 = record.computed_hash()?;
                        record.validate(&staged.sha256, sequence, &staged.run_id)?;
                        let mut bytes = crate::encode_bounded(&record, max_frame)?;
                        bytes.push(b'\n');
                        let total = decimal_u64(&staged.bytes)?
                            .checked_add(bytes.len() as u64)
                            .context("journal size overflow")?;
                        if sequence > max_records || total > max_bytes {
                            commit_batch(&mut file, &writer_path, &staged, &mut state)?;
                            state.status = "resource_limit".into();
                            state.error = Some("capture_limit_reached".into());
                            anyhow::bail!("narration capture limit reached");
                        }
                        file.write_all(&bytes)?;
                        staged.records = sequence.to_string();
                        staged.bytes = total.to_string();
                        staged.sha256 = record.sha256;
                        staged.dropped = receiver.dropped().to_string();
                        batch += 1;
                        if receiver.is_empty()
                            || batch >= 64
                            || batch_start.elapsed() >= std::time::Duration::from_millis(50)
                        {
                            commit_batch(&mut file, &writer_path, &staged, &mut state)?;
                            batch = 0;
                            batch_start = std::time::Instant::now();
                        }
                    }
                    commit_batch(&mut file, &writer_path, &staged, &mut state)?;
                    file.sync_all()?;
                    Ok(())
                })();
                receiver.close();
                state.dropped = receiver.dropped().to_string();
                if result.is_err() {
                    if state.status != "resource_limit" {
                        state.status = "failed".into();
                        state.error = Some("writer_failed".into());
                    }
                } else {
                    state.status = if state.dropped == "0" {
                        "complete"
                    } else {
                        "incomplete"
                    }
                    .into();
                }
                // Best effort on an I/O failure; a stale 'capturing' marker explicitly
                // leaves closure unknown. Never substitute a source RunStopped event.
                let closure = publish_commit(&writer_path, &state, false);
                result?;
                closure?;
                Ok(state)
            })
            .context("start narration writer")?;
        Ok((
            tap.clone(),
            Self {
                path: path.to_owned(),
                tap,
                handle: Some(handle),
            },
        ))
    }
    pub fn finish(mut self) -> Result<JournalSummary> {
        self.tap.close();
        self.handle
            .take()
            .expect("writer handle present")
            .join()
            .map_err(|_| anyhow::anyhow!("narration writer panicked"))?
    }
}

fn commit_batch(
    file: &mut File,
    path: &Path,
    staged: &JournalSummary,
    state: &mut JournalSummary,
) -> Result<()> {
    if staged.records != state.records {
        file.sync_data()?;
        // If directory sync fails after rename, this prefix was still data-synced:
        // retaining it prevents a later closure attempt from regressing the marker.
        *state = staged.clone();
        publish_commit(path, state, false)?;
    }
    Ok(())
}

fn bounded_narration(evidence: &str, max_frame: usize, variant: Option<u8>) -> Narration {
    Narration{text:"Exact detail exceeds the configured message bound; the captured input remains available in this record.".into(),category:"limitation".into(),claims:vec![crate::Claim{category:"limitation".into(),predicate:"detail_budget_exceeded".into(),evidence:vec![evidence.into()],details:serde_json::json!({"max_frame_bytes":max_frame})}],delta:None,variant}
}
impl Drop for FactJournal {
    fn drop(&mut self) {
        self.tap.close();
    }
}

pub fn spawn_journal(path: &Path) -> Result<(Tap, FactJournal)> {
    spawn_journal_with(path, crate::DEFAULT_MAX_FACTS, crate::DEFAULT_MAX_BYTES)
}
pub fn spawn_journal_with(
    path: &Path,
    max_facts: usize,
    max_bytes: usize,
) -> Result<(Tap, FactJournal)> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let run_id = digest(
        format!(
            "jope-narration/run/1:{now}:{}:{}",
            std::process::id(),
            path.display()
        )
        .as_bytes(),
    );
    FactJournal::create(
        path,
        &run_id,
        Limits {
            max_facts,
            max_bytes,
            ..Limits::default()
        },
    )
}
pub fn journal_from_env() -> Result<Option<(Tap, FactJournal)>> {
    let Some(root) =
        std::env::var_os(DIR_ENV).or_else(|| std::env::var_os("JOPE_LINEAGE_FACT_DIR"))
    else {
        return Ok(None);
    };
    if ENV_SESSIONS
        .fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |n| (n < MAX_ENV_SESSIONS).then_some(n + 1),
        )
        .is_err()
    {
        if !ENV_LIMIT_REPORTED.swap(true, std::sync::atomic::Ordering::AcqRel) {
            anyhow::bail!("automatic narration session limit reached; further captures disabled for this process");
        }
        return Ok(None);
    }
    let root = PathBuf::from(root);
    fs::create_dir_all(&root)?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    spawn_journal(&root.join(format!("{stamp}-{}.jsonl", std::process::id()))).map(Some)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadPage {
    /// Identifier of the journal run.
    pub run_id: String,
    /// Validated records returned by this page.
    pub events: Vec<Envelope>,
    /// Cursor to pass as `after` for the next page, if more records may exist.
    pub next: Option<Cursor>,
    /// Whether the file ended in an incomplete trailing record.
    pub partial_tail: bool,
}

/// Bounded offline reader. Live services maintain their own validated offset index using
/// `Envelope::validate`, so polling a stream never requires rescanning historical records.
/// This helper validates through the returned page; the terminal watermark count/hash
/// is checked only when the read reaches that watermark. Use the observatory index
/// for full committed-prefix validation before serving a finite page.
pub fn read_journal(
    path: &Path,
    after: Option<&Cursor>,
    limit: usize,
    max_bytes: usize,
) -> Result<ReadPage> {
    ensure!(
        (1..=1024).contains(&limit) && (1..=64 * 1024 * 1024).contains(&max_bytes),
        "invalid page limits"
    );
    let after_seq = after
        .map(|c| decimal_u64(&c.sequence))
        .transpose()?
        .unwrap_or(0);
    let committed = read_commit(path)?;
    let committed_bytes = decimal_u64(&committed.bytes)?;
    let file = File::open(path)?;
    let observed = file.metadata()?.len();
    ensure!(
        observed >= committed_bytes,
        "journal truncated below watermark"
    );
    use std::io::Read;
    let mut reader = BufReader::new(file.take(committed_bytes));
    let mut run_id = committed.run_id.clone();
    if let Some(cursor) = after {
        ensure!(cursor.run_id == run_id, "cursor run mismatch");
    }
    let mut expected = 1u64;
    let mut previous = String::new();
    let mut events = Vec::new();
    let mut bytes = 0usize;
    let partial_tail = observed > committed_bytes;
    loop {
        let Some((line, complete)) = bounded_line(&mut reader, MAX_FRAME_BYTES)? else {
            break;
        };
        ensure!(complete, "committed record lacks newline");
        let record: Envelope =
            serde_json::from_slice(&line).context("invalid complete narration record")?;
        if run_id.is_empty() {
            run_id = record.run_id.clone();
            if let Some(cursor) = after {
                ensure!(cursor.run_id == run_id, "cursor run mismatch");
            }
        }
        record.validate(&previous, expected, &run_id)?;
        previous = record.sha256.clone();
        expected = expected.checked_add(1).context("sequence exhausted")?;
        if expected - 1 > after_seq {
            if line.len() > max_bytes.saturating_sub(bytes) {
                ensure!(!events.is_empty(), "page budget smaller than next record");
                break;
            }
            bytes += line.len();
            events.push(record);
            if events.len() >= limit {
                break;
            }
        }
    }
    if reader.fill_buf()?.is_empty() {
        ensure!(
            expected - 1 == decimal_u64(&committed.records)?,
            "watermark record count mismatch"
        );
        ensure!(previous == committed.sha256, "watermark hash mismatch");
    }
    ensure!(after_seq < expected, "future cursor");
    if run_id.is_empty() {
        if let Some(cursor) = after {
            ensure!(cursor.sequence == "0", "future cursor");
            run_id = cursor.run_id.clone();
        }
    }
    let next = events
        .last()
        .map(|e| e.cursor.clone())
        .or_else(|| after.cloned());
    Ok(ReadPage {
        run_id,
        events,
        next,
        partial_tail,
    })
}

fn bounded_line(reader: &mut impl BufRead, max: usize) -> Result<Option<(Vec<u8>, bool)>> {
    let mut line = Vec::new();
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            return Ok(if line.is_empty() {
                None
            } else {
                Some((line, false))
            });
        }
        let end = buffer.iter().position(|b| *b == b'\n');
        let take = end.map(|p| p + 1).unwrap_or(buffer.len());
        ensure!(
            take <= max.saturating_add(1).saturating_sub(line.len()),
            "narration frame exceeds limit"
        );
        line.extend_from_slice(&buffer[..take]);
        reader.consume(take);
        if end.is_some() {
            line.pop();
            return Ok(Some((line, true)));
        }
    }
}
