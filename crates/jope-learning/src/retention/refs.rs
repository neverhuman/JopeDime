//! References that pin object groups: rosters, seed packets, the main tree and the evidence store.
//!
//! Declared reference files (`--refs ROLE=PATH`) are JSON or JSONL and must parse; a file that
//! cannot be read is fatal. Scan roots (`--scan ROLE=DIR`) are walked best-effort: every regular
//! file up to a size cap whose bytes mention an in-scope bucket is parsed as JSON when it can be
//! and otherwise token-scanned for `s3://` URIs; the counts are reported. A file over the cap is
//! streamed for the bucket names: one that names none could not have pinned anything, and one
//! that does fails the run (fail closed), because skipping it could hide a T0 pin.
//!
//! What counts as a reference:
//! - an object with string fields `bucket` and `key` (the roster/receipt artifact form), or a
//!   `key` without `bucket` next to a `sha256`/`version_id` pin (resolved against the ledger);
//! - an `s3://bucket/key` string whose key does not end in `/` (an object, not a prefix; bucket
//!   and prefix mentions in prose are counted separately and pin nothing);
//! - `context_id`/`context_sha256` and `program_sha256`/`program_sha` fields holding 64 hex
//!   digits, for declared reference files only.
//!
//! Owner decision D1 (ENG:watcher, 2026-10-13T12:36:00Z): in the scan roles the policy names, an
//! object ref found in a prose file (the policy's extensions, md/txt/log) is a
//! [`RetentionReferenceRole::ProseMention`]: training value that blocks EXPIRE, not a T0 pin. Structured JSON/JSONL
//! refs keep their scan role; formats the ruling does not name stay pins (fail closed).

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::Path;

/// Why a reference matters. T0 roles make the group NEVER-delete; the others give it training
/// value and, when they name objects by key, keep it readable in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RetentionReferenceRole {
    /// The `Val263` case of `RetentionReferenceRole`.
    Val263,
    /// The `Sealed` case of `RetentionReferenceRole`.
    Sealed,
    /// The `Prereg` case of `RetentionReferenceRole`.
    Prereg,
    /// The `Evidence` case of `RetentionReferenceRole`.
    Evidence,
    /// The `MainManifest` case of `RetentionReferenceRole`.
    MainManifest,
    /// The `Train` case of `RetentionReferenceRole`.
    Train,
    /// The `Frozen24` case of `RetentionReferenceRole`.
    Frozen24,
    /// The `Backlog` case of `RetentionReferenceRole`.
    Backlog,
    /// The `Seed` case of `RetentionReferenceRole`.
    Seed,
    /// Derived by a scan (D1), never accepted as a declared role.
    ProseMention,
}

/// Compatibility alias for the domain-specific `RetentionReferenceRole` type.
pub type Role = RetentionReferenceRole;

impl RetentionReferenceRole {
    /// Parses the supplied representation, returning an error when the input is invalid.
    pub fn parse(s: &str) -> Result<RetentionReferenceRole> {
        Ok(match s {
            "val263" => RetentionReferenceRole::Val263,
            "sealed" => RetentionReferenceRole::Sealed,
            "prereg" => RetentionReferenceRole::Prereg,
            "evidence" => RetentionReferenceRole::Evidence,
            "main-manifest" => RetentionReferenceRole::MainManifest,
            "train" => RetentionReferenceRole::Train,
            "frozen24" => RetentionReferenceRole::Frozen24,
            "backlog" => RetentionReferenceRole::Backlog,
            "seed" => RetentionReferenceRole::Seed,
            _ => bail!("unknown reference role {s:?} (val263 sealed prereg evidence main-manifest train frozen24 backlog seed)"),
        })
    }

    /// Returns the stable string representation of this value.
    pub fn as_str(self) -> &'static str {
        match self {
            RetentionReferenceRole::Val263 => "val263",
            RetentionReferenceRole::Sealed => "sealed",
            RetentionReferenceRole::Prereg => "prereg",
            RetentionReferenceRole::Evidence => "evidence",
            RetentionReferenceRole::MainManifest => "main-manifest",
            RetentionReferenceRole::Train => "train",
            RetentionReferenceRole::Frozen24 => "frozen24",
            RetentionReferenceRole::Backlog => "backlog",
            RetentionReferenceRole::Seed => "seed",
            RetentionReferenceRole::ProseMention => "prose-mention",
        }
    }

    /// Strategy §7.3 rule 1 / §8.1 "future qualified-eval input": these refs make a group T0.
    pub fn is_t0(self) -> bool {
        matches!(
            self,
            RetentionReferenceRole::Val263
                | RetentionReferenceRole::Sealed
                | RetentionReferenceRole::Prereg
                | RetentionReferenceRole::Evidence
                | RetentionReferenceRole::MainManifest
        )
    }

    /// Active rosters and seeds read objects by key: their refs keep a group hot.
    pub fn is_active(self) -> bool {
        matches!(
            self,
            RetentionReferenceRole::Train
                | RetentionReferenceRole::Frozen24
                | RetentionReferenceRole::Backlog
                | RetentionReferenceRole::Seed
        )
    }

    /// The scan roles whose prose refs D1 may demote to [`RetentionReferenceRole::ProseMention`].
    pub fn is_scan_pin(self) -> bool {
        matches!(
            self,
            RetentionReferenceRole::Evidence | RetentionReferenceRole::MainManifest
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
/// `ObjRef` is a record for the obj ref contract in this module: References that pin object groups: rosters, seed packets, the main tree and the evidence store.
pub struct ObjRef {
    /// Data or authority role assigned to this value.
    pub role: RetentionReferenceRole,
    /// Bucket recorded for this obj ref.
    pub bucket: String,
    /// Key recorded for this obj ref.
    pub key: String,
    /// The file the reference came from (relative to its scan root for scans).
    pub source: String,
}

#[derive(Clone, Debug, Default, Serialize)]
/// `ScanStats` is a record for the scan stats contract in this module: References that pin object groups: rosters, seed packets, the main tree and the evidence store.
pub struct ScanStats {
    /// Data or authority role assigned to this value.
    pub role: String,
    /// Root recorded for this scan stats.
    pub root: String,
    /// Files seen recorded for this scan stats.
    pub files_seen: u64,
    /// Files read recorded for this scan stats.
    pub files_read: u64,
    /// Files over the size cap, streamed and found to name no in-scope bucket.
    pub files_over_cap: u64,
    /// Files skipped because their extension is not a text format in [`SCAN_EXTENSIONS`].
    pub files_not_text: u64,
    /// Files mentioning buckets recorded for this scan stats.
    pub files_mentioning_buckets: u64,
    /// Files unreadable recorded for this scan stats.
    pub files_unreadable: u64,
    /// JSON files or JSONL lines skipped because they are inventories, not pins.
    pub inventory_skipped: u64,
    /// Bytes read recorded for this scan stats.
    pub bytes_read: u64,
    /// Object refs recorded for this scan stats.
    pub object_refs: u64,
    /// Of `object_refs`, those found in prose files and recorded as [`RetentionReferenceRole::ProseMention`].
    pub prose_refs: u64,
    /// Prefix mentions recorded for this scan stats.
    pub prefix_mentions: u64,
    /// sha256 over the sorted `relative path \t size \t sha256` lines of the files that mention
    /// an in-scope bucket: identifies what the scan actually used.
    pub matched_listing_sha256: String,
}

#[derive(Debug, Default)]
/// `RefSet` is a record for the ref set contract in this module: References that pin object groups: rosters, seed packets, the main tree and the evidence store.
pub struct RefSet {
    /// Number of objects represented by this value.
    pub objects: BTreeSet<ObjRef>,
    /// Mapping of contexts values carried by this ref set.
    pub contexts: BTreeMap<String, BTreeSet<RetentionReferenceRole>>,
    /// Mapping of programs values carried by this ref set.
    pub programs: BTreeMap<String, BTreeSet<RetentionReferenceRole>>,
    /// Prefix mentions recorded for this ref set.
    pub prefix_mentions: u64,
    /// Collection of scans values carried by this ref set.
    pub scans: Vec<ScanStats>,
    /// Source label stamped on references while one file is being read.
    source: String,
}

/// Text formats a scan reads; anything else (weights, tensors, archives) cannot hold a pin we
/// could parse and would only cost I/O.
pub const SCAN_EXTENSIONS: [&str; 16] = [
    "json", "jsonl", "md", "txt", "tsv", "csv", "yaml", "yml", "toml", "log", "sh", "rs", "py",
    "html", "sql", "dsl",
];

/// Documents that list every object (indexes, ledgers, plans) rather than pin a few. Counting
/// them as references would make everything they list T0.
pub const INVENTORY_SCHEMAS: [&str; 5] = [
    "jopedime.gp-results-index/v1",
    "jopedime.s3-object-ledger/v1",
    "jopedime.retention-plan-row/v1",
    "jopedime.retention-plan-summary/v1",
    "jopedime.retention-brain-inputs/v1",
];

fn is_inventory(v: &Value) -> bool {
    v.get("schema")
        .and_then(Value::as_str)
        .is_some_and(|s| INVENTORY_SCHEMAS.contains(&s))
}

/// Whether a file contains any of `needles`, read in 8 MiB chunks that overlap by one needle
/// length so a match across a chunk boundary is still seen. Same test as for files under the
/// cap (lossy UTF-8, substring), without holding the file in memory.
fn mentions_any(path: &Path, needles: &[String]) -> std::io::Result<bool> {
    let keep = needles.iter().map(String::len).max().unwrap_or(0);
    let mut file = fs::File::open(path)?;
    let mut buf = vec![0u8; (8 << 20) + keep];
    let mut have = 0;
    loop {
        let n = file.read(&mut buf[have..])?;
        if n == 0 {
            return Ok(false);
        }
        have += n;
        let text = String::from_utf8_lossy(&buf[..have]);
        if needles.iter().any(|nd| text.contains(nd.as_str())) {
            return Ok(true);
        }
        let tail = have.min(keep);
        buf.copy_within(have - tail..have, 0);
        have = tail;
    }
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// An object key, not a prefix (`.../`) or a pattern/placeholder (`*`, `?`, `{}`, `$`, `<>`, or an
/// elided key containing `…`, as prose abbreviates long keys).
pub fn is_object_key(key: &str) -> bool {
    !key.is_empty()
        && !key.ends_with('/')
        && !key.contains(['*', '?', '{', '}', '$', '<', '>', '\u{2026}'])
}

/// Splits `s3://bucket/key`; `None` for anything that is not an object URI.
pub fn parse_s3_uri(s: &str) -> Option<(String, String)> {
    let rest = s.strip_prefix("s3://")?;
    let (bucket, key) = rest.split_once('/')?;
    let bucket_ok = !bucket.is_empty()
        && bucket
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'.');
    if !bucket_ok || !is_object_key(key) {
        return None;
    }
    Some((bucket.to_string(), key.to_string()))
}

mod set;

#[cfg(test)]
mod tests {
    use super::mentions_any;
    use std::fs;

    #[test]
    fn oversized_files_are_streamed_across_chunk_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let needles = vec!["needle-0123456".to_string(), "other".to_string()];
        let chunk = (8 << 20) + needles[0].len();
        // The needle straddles the end of the first read.
        let mut bytes = vec![b'x'; chunk + 100];
        let at = chunk - 5;
        bytes[at..at + needles[0].len()].copy_from_slice(needles[0].as_bytes());
        let hit = dir.path().join("hit.txt");
        fs::write(&hit, &bytes).unwrap();
        assert!(mentions_any(&hit, &needles).unwrap());
        // Invalid UTF-8 around it does not hide it either.
        bytes[at - 1] = 0xff;
        fs::write(&hit, &bytes).unwrap();
        assert!(mentions_any(&hit, &needles).unwrap());
        let miss = dir.path().join("miss.txt");
        fs::write(&miss, vec![b'x'; chunk * 2]).unwrap();
        assert!(!mentions_any(&miss, &needles).unwrap());
        assert!(mentions_any(&dir.path().join("absent"), &needles).is_err());
    }
}
