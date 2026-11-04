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
    io::Write,
    path::{Path, PathBuf},
};

mod reader;
pub use reader::{read_journal, ReadPage};
mod writer;
pub use writer::{journal_from_env, spawn_journal, spawn_journal_with, FactJournal};

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
    pub(super) fn computed_hash(&self) -> Result<String> {
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
                self.narration == writer::bounded_narration(&evidence, frame, expected.variant),
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
