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
//! [`Role::ProseMention`]: training value that blocks EXPIRE, not a T0 pin. Structured JSON/JSONL
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
pub enum Role {
    /// The `Val263` case of `Role`.
    Val263,
    /// The `Sealed` case of `Role`.
    Sealed,
    /// The `Prereg` case of `Role`.
    Prereg,
    /// The `Evidence` case of `Role`.
    Evidence,
    /// The `MainManifest` case of `Role`.
    MainManifest,
    /// The `Train` case of `Role`.
    Train,
    /// The `Frozen24` case of `Role`.
    Frozen24,
    /// The `Backlog` case of `Role`.
    Backlog,
    /// The `Seed` case of `Role`.
    Seed,
    /// Derived by a scan (D1), never accepted as a declared role.
    ProseMention,
}

impl Role {
    /// Parses the supplied representation, returning an error when the input is invalid.
    pub fn parse(s: &str) -> Result<Role> {
        Ok(match s {
            "val263" => Role::Val263,
            "sealed" => Role::Sealed,
            "prereg" => Role::Prereg,
            "evidence" => Role::Evidence,
            "main-manifest" => Role::MainManifest,
            "train" => Role::Train,
            "frozen24" => Role::Frozen24,
            "backlog" => Role::Backlog,
            "seed" => Role::Seed,
            _ => bail!("unknown reference role {s:?} (val263 sealed prereg evidence main-manifest train frozen24 backlog seed)"),
        })
    }

    /// Returns the stable string representation of this value.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Val263 => "val263",
            Role::Sealed => "sealed",
            Role::Prereg => "prereg",
            Role::Evidence => "evidence",
            Role::MainManifest => "main-manifest",
            Role::Train => "train",
            Role::Frozen24 => "frozen24",
            Role::Backlog => "backlog",
            Role::Seed => "seed",
            Role::ProseMention => "prose-mention",
        }
    }

    /// Strategy §7.3 rule 1 / §8.1 "future qualified-eval input": these refs make a group T0.
    pub fn is_t0(self) -> bool {
        matches!(
            self,
            Role::Val263 | Role::Sealed | Role::Prereg | Role::Evidence | Role::MainManifest
        )
    }

    /// Active rosters and seeds read objects by key: their refs keep a group hot.
    pub fn is_active(self) -> bool {
        matches!(
            self,
            Role::Train | Role::Frozen24 | Role::Backlog | Role::Seed
        )
    }

    /// The scan roles whose prose refs D1 may demote to [`Role::ProseMention`].
    pub fn is_scan_pin(self) -> bool {
        matches!(self, Role::Evidence | Role::MainManifest)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
/// `ObjRef` is a record for the obj ref contract in this module: References that pin object groups: rosters, seed packets, the main tree and the evidence store.
pub struct ObjRef {
    /// Data or authority role assigned to this value.
    pub role: Role,
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
    /// Of `object_refs`, those found in prose files and recorded as [`Role::ProseMention`].
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
    pub contexts: BTreeMap<String, BTreeSet<Role>>,
    /// Mapping of programs values carried by this ref set.
    pub programs: BTreeMap<String, BTreeSet<Role>>,
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

impl RefSet {
    fn add_uri(&mut self, role: Role, s: &str, count: &mut (u64, u64)) {
        match parse_s3_uri(s) {
            Some((bucket, key)) => {
                self.objects.insert(ObjRef {
                    role,
                    bucket,
                    key,
                    source: self.source.clone(),
                });
                count.0 += 1;
            }
            None => {
                self.prefix_mentions += 1;
                count.1 += 1;
            }
        }
    }

    /// Collects references from one JSON value. `ids` enables context/program id pins.
    pub fn walk_json(&mut self, role: Role, v: &Value, ids: bool, count: &mut (u64, u64)) {
        match v {
            Value::Object(m) => {
                if let (Some(Value::String(bucket)), Some(Value::String(key))) =
                    (m.get("bucket"), m.get("key"))
                {
                    if is_object_key(key) {
                        self.objects.insert(ObjRef {
                            role,
                            bucket: bucket.clone(),
                            key: key.clone(),
                            source: self.source.clone(),
                        });
                        count.0 += 1;
                    }
                } else if let (None, Some(Value::String(key))) = (m.get("bucket"), m.get("key")) {
                    // `{key, sha256|version_id}` without a bucket (e.g. frozen24 gp_hwm_ref): the
                    // bucket is resolved later from the ledger, and only when exactly one matches.
                    let pinned = m.contains_key("sha256") || m.contains_key("version_id");
                    if pinned && key.contains('/') && is_object_key(key) {
                        self.objects.insert(ObjRef {
                            role,
                            bucket: String::new(),
                            key: key.clone(),
                            source: self.source.clone(),
                        });
                        count.0 += 1;
                    }
                }
                if ids {
                    for (field, target) in [
                        ("context_id", 0u8),
                        ("context_sha256", 0),
                        ("program_sha256", 1),
                        ("program_sha", 1),
                    ] {
                        if let Some(Value::String(s)) = m.get(field) {
                            if is_hex64(s) {
                                let map = if target == 0 {
                                    &mut self.contexts
                                } else {
                                    &mut self.programs
                                };
                                map.entry(s.to_ascii_lowercase()).or_default().insert(role);
                            }
                        }
                    }
                }
                for child in m.values() {
                    self.walk_json(role, child, ids, count);
                }
            }
            Value::Array(a) => {
                for child in a {
                    self.walk_json(role, child, ids, count);
                }
            }
            Value::String(s) if s.starts_with("s3://") => self.add_uri(role, s, count),
            _ => {}
        }
    }

    /// Token-scans text for `s3://` URIs.
    pub fn scan_text(&mut self, role: Role, text: &str, count: &mut (u64, u64)) {
        let mut rest = text;
        while let Some(i) = rest.find("s3://") {
            let tail = &rest[i..];
            let end = tail
                .find(|c: char| c.is_whitespace() || "\"'`<>()[]{},;|\\".contains(c))
                .unwrap_or(tail.len());
            let token = tail[..end].trim_end_matches(['.', ':']);
            self.add_uri(role, token, count);
            rest = &tail[5..];
        }
    }

    /// A declared reference file: JSON, or JSONL when the name ends in `.jsonl`. Must parse.
    pub fn add_file(&mut self, role: Role, name: &str, bytes: &[u8]) -> Result<(u64, u64)> {
        self.source = name.to_string();
        let mut count = (0, 0);
        if name.ends_with(".jsonl") {
            let text =
                std::str::from_utf8(bytes).with_context(|| format!("{name} is not UTF-8"))?;
            for (n, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let v: Value = serde_json::from_str(line)
                    .with_context(|| format!("{name} line {} does not parse", n + 1))?;
                self.walk_json(role, &v, true, &mut count);
            }
        } else {
            let v: Value = serde_json::from_slice(bytes)
                .with_context(|| format!("{name} does not parse as JSON"))?;
            self.walk_json(role, &v, true, &mut count);
        }
        Ok(count)
    }

    /// Walks a scan root. `needles` are substrings (in-scope bucket name stems) a file must
    /// contain to be parsed; `skip_dirs` are directory names never entered. Object refs in files
    /// whose extension is in `prose_exts` are recorded as [`Role::ProseMention`] (D1).
    pub fn scan_root(
        &mut self,
        role: Role,
        root: &Path,
        max_bytes: u64,
        needles: &[String],
        skip_dirs: &[&str],
        prose_exts: &[String],
    ) -> Result<()> {
        if !root.is_dir() {
            bail!(
                "scan root {} ({}) is not a directory",
                root.display(),
                role.as_str()
            );
        }
        let mut st = ScanStats {
            role: role.as_str().into(),
            root: root.display().to_string(),
            ..Default::default()
        };
        let mut listing: Vec<String> = Vec::new();
        let walker = walkdir::WalkDir::new(root)
            .follow_links(false)
            .sort_by_file_name()
            .into_iter();
        let walker = walker.filter_entry(|e| {
            !(e.file_type().is_dir()
                && e.depth() > 0
                && skip_dirs.iter().any(|d| e.file_name() == *d))
        });
        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => {
                    st.files_unreadable += 1;
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            st.files_seen += 1;
            let ext = entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !SCAN_EXTENSIONS.contains(&ext.as_str()) {
                st.files_not_text += 1;
                continue;
            }
            let len = entry.metadata().map(|m| m.len()).unwrap_or(u64::MAX);
            if len > max_bytes {
                match mentions_any(entry.path(), needles) {
                    Ok(false) => st.files_over_cap += 1,
                    Ok(true) => bail!(
                        "{} ({len} bytes, scan {}) names an in-scope bucket but is over the scan cap of {max_bytes} bytes, so it would be skipped and could hide a pin: raise --scan-max-bytes",
                        entry.path().display(),
                        role.as_str()
                    ),
                    Err(_) => st.files_unreadable += 1,
                }
                continue;
            }
            let Ok(bytes) = fs::read(entry.path()) else {
                st.files_unreadable += 1;
                continue;
            };
            st.files_read += 1;
            st.bytes_read += bytes.len() as u64;
            let text = String::from_utf8_lossy(&bytes);
            if !needles.iter().any(|n| text.contains(n.as_str())) {
                continue;
            }
            st.files_mentioning_buckets += 1;
            let rel = entry
                .path()
                .strip_prefix(root)
                .unwrap_or(entry.path())
                .display()
                .to_string();
            listing.push(format!(
                "{rel}\t{}\t{}",
                bytes.len(),
                hex::encode(Sha256::digest(&bytes))
            ));
            self.source = rel;
            let mut count = (0, 0);
            let prose = prose_exts.contains(&ext);
            let role = if prose { Role::ProseMention } else { role };
            let name = entry.file_name().to_string_lossy();
            let parsed = if name.ends_with(".jsonl") {
                let mut ok = true;
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    match serde_json::from_str::<Value>(line) {
                        Ok(v) if is_inventory(&v) => st.inventory_skipped += 1,
                        Ok(v) => self.walk_json(role, &v, false, &mut count),
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                ok
            } else if name.ends_with(".json") {
                match serde_json::from_str::<Value>(&text) {
                    Ok(v) if is_inventory(&v) => {
                        st.inventory_skipped += 1;
                        true
                    }
                    Ok(v) => {
                        self.walk_json(role, &v, false, &mut count);
                        true
                    }
                    Err(_) => false,
                }
            } else {
                false
            };
            if !parsed {
                count = (0, 0);
                self.scan_text(role, &text, &mut count);
            }
            st.object_refs += count.0;
            if prose {
                st.prose_refs += count.0;
            }
            st.prefix_mentions += count.1;
        }
        listing.sort();
        st.matched_listing_sha256 = hex::encode(Sha256::digest(listing.join("\n").as_bytes()));
        self.scans.push(st);
        Ok(())
    }
}

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
