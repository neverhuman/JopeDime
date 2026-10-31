//! Bounded immutable progress history segments and a hash-bound exact-key shard index.
//!
//! This is a local protocol primitive. It keeps source proof bytes in content-addressed immutable
//! segments and advances a conditional root only after the segment and index shard are durable.
//! It never deletes an older segment or root. At any cap it refuses new history instead of
//! pruning proof. The product outbox/rebuild callers must adopt this reader before their historical
//! Vec/BTreeMap path can be considered bounded.

use crate::canon::{canonical_json, is_hex64, sha256_hex};
use crate::ledger::atomic_write;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

/// Schema identifier for segment values.
pub const SEGMENT_SCHEMA: &str = "jope.progress.segment/v1";
/// Schema identifier for root values.
pub const ROOT_SCHEMA: &str = "jope.progress.segment-root/v1";
/// Schema identifier for index values.
pub const INDEX_SCHEMA: &str = "jope.progress.segment-index-shard/v1";
/// Schema identifier for current values.
pub const CURRENT_SCHEMA: &str = "jope.progress.segment-current/v1";
/// Constant identifying the shards.
pub const SHARDS: usize = 256;

/// Prospective finite limits. No default is implied by the legacy pending-only outbox limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentLimits {
    /// Upper bound on records per segment.
    pub max_records_per_segment: u64,
    /// Upper bound on segment bytes.
    pub max_segment_bytes: u64,
    /// Upper bound on segments.
    pub max_segments: u64,
    /// Upper bound on shard bytes.
    pub max_shard_bytes: u64,
    /// Upper bound on entries.
    pub max_entries: u64,
    /// Upper bound on startup bytes.
    pub max_startup_bytes: u64,
    /// Upper bound on total disk bytes.
    pub max_total_disk_bytes: u64,
    /// Upper bound on hot bytes.
    pub max_hot_bytes: u64,
    /// Upper bound on root bytes.
    pub max_root_bytes: u64,
}

impl SegmentLimits {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_records_per_segment > 0
                && self.max_segment_bytes > 0
                && self.max_segments > 0
                && self.max_shard_bytes > 0
                && self.max_entries > 0
                && self.max_startup_bytes > 0
                && self.max_total_disk_bytes > 0
                && self.max_hot_bytes > 0
                && self.max_root_bytes > 0,
            "segment limits must all be positive"
        );
        ensure!(
            self.max_segment_bytes <= self.max_total_disk_bytes
                && self.max_shard_bytes <= self.max_total_disk_bytes
                && self.max_root_bytes <= self.max_total_disk_bytes
                && self.max_startup_bytes <= self.max_total_disk_bytes,
            "a single segment/root/startup allowance cannot exceed total disk capacity"
        );
        ensure!(
            self.max_segments <= self.max_entries,
            "segment count cannot exceed admitted entry count"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
/// `EvidenceClass` is an enum for the evidence class contract in this module: Bounded immutable progress history segments and a hash-bound exact-key shard index.
pub enum EvidenceClass {
    /// The `Proposed` case of `EvidenceClass`.
    Proposed,
    /// The `Computed` case of `EvidenceClass`.
    Computed,
    /// The `Applied` case of `EvidenceClass`.
    Applied,
    /// The `Committed` case of `EvidenceClass`.
    Committed,
    /// The `Uncommitted` case of `EvidenceClass`.
    Uncommitted,
    /// The `Failed` case of `EvidenceClass`.
    Failed,
    /// The `Unknown` case of `EvidenceClass`.
    Unknown,
    /// The `Inconclusive` case of `EvidenceClass`.
    Inconclusive,
    /// The `Divergent` case of `EvidenceClass`.
    Divergent,
    /// The `Quarantined` case of `EvidenceClass`.
    Quarantined,
    /// The `LocalOnly` case of `EvidenceClass`.
    LocalOnly,
    /// The `RemotePending` case of `EvidenceClass`.
    RemotePending,
}

/// One retained source item. `proof_bytes` is the exact source receipt, not a summary claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentRecord {
    /// Key recorded for this segment record.
    pub key: String,
    /// Source key recorded for this segment record.
    pub source_key: String,
    /// Class recorded for this segment record.
    pub class: EvidenceClass,
    /// SHA-256 digest identifying the associated source.
    pub source_sha256: String,
    /// Byte count for the associated proof.
    pub proof_bytes: Vec<u8>,
    /// Optional committed update key value; `None` means it is not available.
    pub committed_update_key: Option<String>,
}

impl SegmentRecord {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.key.is_empty() && self.key.len() <= 512,
            "invalid segment key"
        );
        ensure!(
            !self.source_key.is_empty() && self.source_key.len() <= 1024,
            "invalid source proof key"
        );
        ensure!(is_hex64(&self.source_sha256), "invalid source proof digest");
        ensure!(
            sha256_hex(&self.proof_bytes) == self.source_sha256,
            "source proof bytes do not match their digest"
        );
        if let Some(key) = &self.committed_update_key {
            ensure!(
                !key.is_empty() && key.len() <= 512,
                "invalid committed update key"
            );
            ensure!(
                self.class == EvidenceClass::Committed,
                "only committed source receipts may name committed update credit"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Segment` is a record for the segment contract in this module: Bounded immutable progress history segments and a hash-bound exact-key shard index.
pub struct Segment {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Generation recorded for this segment.
    pub generation: u64,
    /// SHA-256 digest identifying the associated previous root.
    pub previous_root_sha256: Option<String>,
    /// Collection of records values carried by this segment.
    pub records: Vec<SegmentRecord>,
    /// Digest recorded for this segment.
    pub digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `SegmentRef` is a record for the segment ref contract in this module: Bounded immutable progress history segments and a hash-bound exact-key shard index.
pub struct SegmentRef {
    /// Digest recorded for this segment ref.
    pub digest: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
    /// Records recorded for this segment ref.
    pub records: u64,
    /// Generation recorded for this segment ref.
    pub generation: u64,
    /// SHA-256 digest identifying the associated previous root.
    pub previous_root_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ShardRef` is a record for the shard ref contract in this module: Bounded immutable progress history segments and a hash-bound exact-key shard index.
pub struct ShardRef {
    /// Digest recorded for this shard ref.
    pub digest: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
    /// Entries recorded for this shard ref.
    pub entries: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Root` is a record for the root contract in this module: Bounded immutable progress history segments and a hash-bound exact-key shard index.
pub struct Root {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Generation recorded for this root.
    pub generation: u64,
    /// SHA-256 digest identifying the associated previous root.
    pub previous_root_sha256: Option<String>,
    /// Collection of segments values carried by this root.
    pub segments: Vec<SegmentRef>,
    /// Fixed 256-way exact index. Empty buckets are null; each nonempty page is byte-bounded.
    pub shards: Vec<Option<ShardRef>>,
    /// Total records recorded for this root.
    pub total_records: u64,
    /// Byte count for the associated total proof.
    pub total_proof_bytes: u64,
    /// Digest recorded for this root.
    pub digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexEntry {
    key: String,
    segment_sha256: String,
    ordinal: u64,
    record_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexShard {
    schema: String,
    bucket: u8,
    generation: u64,
    entries: Vec<IndexEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Current {
    schema: String,
    generation: u64,
    root_sha256: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
/// `Usage` is a record for the usage contract in this module: Bounded immutable progress history segments and a hash-bound exact-key shard index.
pub struct Usage {
    /// Generation recorded for this usage.
    pub generation: u64,
    /// Retained records recorded for this usage.
    pub retained_records: u64,
    /// Retained segments recorded for this usage.
    pub retained_segments: u64,
    /// Byte count for the associated retained proof.
    pub retained_proof_bytes: u64,
    /// Byte count for the associated current root.
    pub current_root_bytes: u64,
    /// Startup bytes read recorded for this usage.
    pub startup_bytes_read: u64,
    /// Byte count for the associated total disk.
    pub total_disk_bytes: u64,
    /// Byte count for the associated hot index.
    pub hot_index_bytes: u64,
    /// Byte count for the associated lookup.
    pub lookup_bytes: u64,
    /// Lookup pages recorded for this usage.
    pub lookup_pages: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `Inserted` is an enum for the inserted contract in this module: Bounded immutable progress history segments and a hash-bound exact-key shard index.
pub enum Inserted {
    /// The `Added` case of `Inserted`.
    Added {
        /// Generation recorded for this inserted.
        generation: u64,
        /// SHA-256 digest identifying the associated segment.
        segment_sha256: String,
    },
    /// The `Existing` case of `Inserted`.
    Existing {
        /// SHA-256 digest identifying the associated segment.
        segment_sha256: String,
    },
}

/// `SegmentStore` is a record for the segment store contract in this module: Bounded immutable progress history segments and a hash-bound exact-key shard index.
pub struct SegmentStore {
    root: PathBuf,
    limits: SegmentLimits,
    current: Current,
    root_doc: Root,
    usage: Usage,
    _lock: File,
}

fn encoded<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    canonical_json(value)
}

fn digest_body<T: Serialize>(value: &T) -> Result<String> {
    Ok(sha256_hex(&encoded(value)?))
}

fn bucket(key: &str) -> u8 {
    let digest = sha256_hex(key.as_bytes());
    u8::from_str_radix(&digest[..2], 16).expect("sha256 prefix is hexadecimal")
}

fn immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    match fs::read(path) {
        Ok(old) => ensure!(
            old == bytes,
            "immutable object {} conflicts",
            path.display()
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => atomic_write(path, bytes)?,
        Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
    }
    Ok(())
}

fn canonical_read<T: Serialize + for<'de> Deserialize<'de>>(
    path: &Path,
    cap: u64,
) -> Result<(T, Vec<u8>)> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.file_type().is_file(),
        "{} is not a regular file",
        path.display()
    );
    ensure!(
        metadata.len() <= cap,
        "{} exceeds its byte cap",
        path.display()
    );
    let bytes = fs::read(path)?;
    let value: T =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    ensure!(
        encoded(&value)? == bytes,
        "{} is not canonical",
        path.display()
    );
    Ok((value, bytes))
}

fn root_digest(root: &Root) -> Result<String> {
    #[derive(Serialize)]
    struct Body<'a> {
        schema: &'a str,
        generation: u64,
        previous_root_sha256: &'a Option<String>,
        segments: &'a [SegmentRef],
        shards: &'a [Option<ShardRef>],
        total_records: u64,
        total_proof_bytes: u64,
    }
    digest_body(&Body {
        schema: &root.schema,
        generation: root.generation,
        previous_root_sha256: &root.previous_root_sha256,
        segments: &root.segments,
        shards: &root.shards,
        total_records: root.total_records,
        total_proof_bytes: root.total_proof_bytes,
    })
}

fn segment_digest(segment: &Segment) -> Result<String> {
    #[derive(Serialize)]
    struct Body<'a> {
        schema: &'a str,
        generation: u64,
        previous_root_sha256: &'a Option<String>,
        records: &'a [SegmentRecord],
    }
    digest_body(&Body {
        schema: &segment.schema,
        generation: segment.generation,
        previous_root_sha256: &segment.previous_root_sha256,
        records: &segment.records,
    })
}

impl SegmentStore {
    /// Open a single-writer store. Only the current root and its bounded metadata are read at
    /// startup; retained segments are loaded and hash-checked on demand.
    pub fn open(path: impl AsRef<Path>, limits: SegmentLimits) -> Result<Self> {
        limits.validate()?;
        let root = path.as_ref().to_path_buf();
        for dir in ["roots", "segments", "index"] {
            fs::create_dir_all(root.join(dir))?;
        }
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join("LOCK"))?;
        match lock.try_lock() {
            Ok(()) => (),
            Err(fs::TryLockError::WouldBlock) => bail!("segment store has another writer"),
            Err(fs::TryLockError::Error(e)) => return Err(e.into()),
        }
        let current_path = root.join("CURRENT.json");
        let (current, root_doc, current_bytes, root_bytes) = if current_path.exists() {
            let (current, current_bytes) =
                canonical_read::<Current>(&current_path, limits.max_root_bytes)?;
            ensure!(
                current.schema == CURRENT_SCHEMA,
                "unsupported segment current schema"
            );
            let (root_doc, root_bytes) = canonical_read::<Root>(
                &root
                    .join("roots")
                    .join(format!("{}.json", current.root_sha256)),
                limits.max_root_bytes,
            )?;
            ensure!(
                current.generation == root_doc.generation
                    && current.root_sha256 == root_doc.digest
                    && root_digest(&root_doc)? == root_doc.digest,
                "current pointer/root digest mismatch"
            );
            (current, root_doc, current_bytes, root_bytes)
        } else {
            for dir in ["segments", "index"] {
                ensure!(
                    fs::read_dir(root.join(dir))?.next().is_none(),
                    "CURRENT is missing while retained {dir} objects exist; refusing implicit reset"
                );
            }
            let root_dir = root.join("roots");
            let mut roots = fs::read_dir(&root_dir)?;
            let genesis = if let Some(entry) = roots.next() {
                let entry = entry?;
                ensure!(
                    roots.next().is_none(),
                    "CURRENT is missing with multiple root objects; refusing implicit reset"
                );
                let path = entry.path();
                let (candidate, bytes) = canonical_read::<Root>(&path, limits.max_root_bytes)?;
                ensure!(
                    candidate.schema == ROOT_SCHEMA
                        && candidate.generation == 0
                        && candidate.previous_root_sha256.is_none()
                        && candidate.segments.is_empty()
                        && candidate.total_records == 0
                        && candidate.total_proof_bytes == 0
                        && candidate.shards.len() == SHARDS
                        && candidate.shards.iter().all(Option::is_none)
                        && root_digest(&candidate)? == candidate.digest
                        && path.file_name().and_then(|v| v.to_str())
                            == Some(format!("{}.json", candidate.digest).as_str()),
                    "orphan root is not the unique valid genesis root; refusing implicit reset"
                );
                (candidate, bytes)
            } else {
                let mut empty = Root {
                    schema: ROOT_SCHEMA.into(),
                    generation: 0,
                    previous_root_sha256: None,
                    segments: Vec::new(),
                    shards: vec![None; SHARDS],
                    total_records: 0,
                    total_proof_bytes: 0,
                    digest: String::new(),
                };
                empty.digest = root_digest(&empty)?;
                let root_bytes = encoded(&empty)?;
                immutable(
                    &root_dir.join(format!("{}.json", empty.digest)),
                    &root_bytes,
                )?;
                (empty, root_bytes)
            };
            let (empty, root_bytes) = genesis;
            let current = Current {
                schema: CURRENT_SCHEMA.into(),
                generation: 0,
                root_sha256: empty.digest.clone(),
            };
            let current_bytes = encoded(&current)?;
            atomic_write(&current_path, &current_bytes)?;
            (current, empty, current_bytes, root_bytes)
        };
        ensure!(
            root_bytes.len() as u64 <= limits.max_root_bytes,
            "root exceeds root byte cap"
        );
        ensure!(
            root_doc.shards.len() == SHARDS,
            "root must contain exactly {SHARDS} shards"
        );
        ensure!(
            root_doc.segments.len() as u64 <= limits.max_segments,
            "retained segment count exceeds cap"
        );
        ensure!(
            root_doc.schema == ROOT_SCHEMA,
            "unsupported segment root schema"
        );
        ensure!(
            root_doc.generation == root_doc.segments.len() as u64,
            "root generation/segment count mismatch"
        );
        let mut segment_records = 0u64;
        for (index, reference) in root_doc.segments.iter().enumerate() {
            ensure!(
                is_hex64(&reference.digest)
                    && reference.generation == index as u64 + 1
                    && reference.records > 0,
                "malformed retained segment reference"
            );
            ensure!(
                reference.bytes <= limits.max_segment_bytes
                    && is_hex64(&reference.previous_root_sha256),
                "segment reference exceeds its cap or has invalid root ancestry"
            );
            segment_records = segment_records
                .checked_add(reference.records)
                .context("segment record count overflow")?;
        }
        ensure!(
            segment_records == root_doc.total_records,
            "root and segment record counts differ"
        );
        ensure!(
            root_doc.previous_root_sha256.as_deref()
                == root_doc
                    .segments
                    .last()
                    .map(|reference| reference.previous_root_sha256.as_str()),
            "root and last retained segment disagree about predecessor"
        );
        let mut hot_bytes = root_bytes.len() as u64 + current_bytes.len() as u64;
        let mut startup_bytes = hot_bytes;
        let mut indexed_entries = 0u64;
        for (bucket_id, reference) in root_doc
            .shards
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.as_ref().map(|r| (i, r)))
        {
            ensure!(
                reference.bytes <= limits.max_shard_bytes,
                "index shard exceeds byte cap"
            );
            startup_bytes = startup_bytes
                .checked_add(reference.bytes)
                .context("startup byte overflow")?;
            ensure!(
                startup_bytes <= limits.max_startup_bytes,
                "root index exceeds startup byte budget"
            );
            let (shard, bytes) = canonical_read::<IndexShard>(
                &root
                    .join("index")
                    .join(format!("{}.json", reference.digest)),
                limits.max_shard_bytes,
            )?;
            ensure!(
                bytes.len() as u64 == reference.bytes,
                "index shard byte count mismatch"
            );
            ensure!(
                sha256_hex(&bytes) == reference.digest,
                "index shard digest mismatch"
            );
            ensure!(
                shard.schema == INDEX_SCHEMA
                    && shard.bucket == bucket_id as u8
                    && shard.generation <= current.generation,
                "index shard identity mismatch"
            );
            ensure!(
                shard.entries.len() as u64 == reference.entries,
                "index shard count mismatch"
            );
            ensure!(
                shard.entries.windows(2).all(|w| w[0].key < w[1].key),
                "index shard keys are not strictly sorted"
            );
            ensure!(
                shard
                    .entries
                    .iter()
                    .all(|entry| bucket(&entry.key) == shard.bucket
                        && is_hex64(&entry.segment_sha256)
                        && is_hex64(&entry.record_sha256)),
                "index shard contains malformed entry"
            );
            indexed_entries = indexed_entries
                .checked_add(reference.entries)
                .context("index entry count overflow")?;
            hot_bytes = hot_bytes
                .checked_add(bytes.len() as u64)
                .context("hot bytes overflow")?;
        }
        ensure!(
            indexed_entries == root_doc.total_records,
            "root and exact index record counts differ"
        );
        ensure!(
            hot_bytes.saturating_add(limits.max_segment_bytes.saturating_mul(2))
                <= limits.max_hot_bytes,
            "hot index metadata exceeds cap"
        );
        let max_files = limits
            .max_segments
            .checked_mul((SHARDS as u64) + 2)
            .and_then(|n| n.checked_add(4))
            .context("file count cap overflow")?;
        let total_disk_bytes = bounded_disk_usage(&root, limits.max_total_disk_bytes, max_files)?;
        let usage = Usage {
            generation: current.generation,
            retained_records: root_doc.total_records,
            retained_segments: root_doc.segments.len() as u64,
            retained_proof_bytes: root_doc.total_proof_bytes,
            current_root_bytes: root_bytes.len() as u64,
            startup_bytes_read: startup_bytes,
            total_disk_bytes,
            hot_index_bytes: hot_bytes,
            lookup_bytes: 0,
            lookup_pages: 0,
        };
        Ok(Self {
            root,
            limits,
            current,
            root_doc,
            usage,
            _lock: lock,
        })
    }

    /// Performs the usage operation and returns an error if it cannot be completed.
    pub fn usage(&self) -> &Usage {
        &self.usage
    }
    /// Performs the root digest operation and returns an error if it cannot be completed.
    pub fn root_digest(&self) -> &str {
        &self.current.root_sha256
    }

    fn shard_path(&self, digest: &str) -> PathBuf {
        self.root.join("index").join(format!("{digest}.json"))
    }
    fn segment_path(&self, digest: &str) -> PathBuf {
        self.root.join("segments").join(format!("{digest}.json"))
    }

    /// Exact-key check through one bounded hash-bound shard. Hash prefix selects a page; full
    /// key equality handles hash-prefix collisions. An absent result is exact, never probabilistic.
    pub fn lookup(&mut self, key: &str) -> Result<Option<SegmentRecord>> {
        ensure!(!key.is_empty() && key.len() <= 512, "invalid lookup key");
        let Some(reference) = self.root_doc.shards[bucket(key) as usize].as_ref() else {
            return Ok(None);
        };
        let (shard, bytes) = canonical_read::<IndexShard>(
            &self.shard_path(&reference.digest),
            self.limits.max_shard_bytes,
        )?;
        ensure!(
            sha256_hex(&bytes) == reference.digest,
            "lookup index digest mismatch"
        );
        ensure!(shard.bucket == bucket(key), "lookup index bucket mismatch");
        self.usage.lookup_bytes = bytes.len() as u64;
        self.usage.lookup_pages = 1;
        let Some(index) = shard
            .entries
            .binary_search_by(|entry| entry.key.as_str().cmp(key))
            .ok()
            .map(|i| &shard.entries[i])
        else {
            return Ok(None);
        };
        let segment_ref = self
            .root_doc
            .segments
            .iter()
            .find(|s| s.digest == index.segment_sha256)
            .context("index points outside retained root")?;
        let (segment, segment_bytes) = canonical_read::<Segment>(
            &self.segment_path(&segment_ref.digest),
            self.limits.max_segment_bytes,
        )?;
        ensure!(
            segment.digest == segment_ref.digest,
            "segment digest mismatch"
        );
        ensure!(
            segment_bytes.len() as u64 == segment_ref.bytes,
            "segment byte count mismatch"
        );
        ensure!(
            segment.generation == segment_ref.generation
                && segment.previous_root_sha256.as_deref()
                    == Some(segment_ref.previous_root_sha256.as_str()),
            "segment generation/root ancestry mismatch"
        );
        ensure!(
            segment_digest(&segment)? == segment.digest,
            "segment body digest mismatch"
        );
        let record = segment
            .records
            .get(index.ordinal as usize)
            .context("index ordinal outside segment")?
            .clone();
        ensure!(
            record.key == key && sha256_hex(&encoded(&record)?) == index.record_sha256,
            "index record identity mismatch"
        );
        self.usage.lookup_bytes = bytes.len() as u64 + segment_bytes.len() as u64;
        self.usage.lookup_pages = 2;
        Ok(Some(record))
    }

    /// Atomically retain a new immutable source segment and advance the root. Existing exact keys
    /// are idempotent only for byte-identical records. Capacity refusal leaves the old root live.
    pub fn append(&mut self, records: Vec<SegmentRecord>) -> Result<Inserted> {
        self.append_inner(records, None)
    }

    #[cfg(test)]
    fn append_crash_at(&mut self, records: Vec<SegmentRecord>, boundary: u8) -> Result<Inserted> {
        self.append_inner(records, Some(boundary))
    }

    fn append_inner(
        &mut self,
        records: Vec<SegmentRecord>,
        crash_after: Option<u8>,
    ) -> Result<Inserted> {
        ensure!(!records.is_empty(), "empty progress segment refused");
        ensure!(
            records.len() as u64 <= self.limits.max_records_per_segment,
            "segment record count exceeds cap"
        );
        let mut unique = BTreeMap::new();
        for record in &records {
            record.validate()?;
            ensure!(
                unique.insert(record.key.clone(), record).is_none(),
                "duplicate key in proposed segment"
            );
        }
        let mut new_records = Vec::new();
        let mut existing_digest = None;
        for record in records {
            if let Some(old) = self.lookup(&record.key)? {
                ensure!(
                    old == record,
                    "same progress key conflicts with retained source bytes"
                );
                let shard_ref = self.root_doc.shards[bucket(&record.key) as usize]
                    .as_ref()
                    .context("exact index entry missing its shard")?;
                let (shard, _) = canonical_read::<IndexShard>(
                    &self.shard_path(&shard_ref.digest),
                    self.limits.max_shard_bytes,
                )?;
                let entry = shard
                    .entries
                    .binary_search_by(|item| item.key.as_str().cmp(&record.key))
                    .ok()
                    .map(|i| &shard.entries[i])
                    .context("exact index entry disappeared")?;
                existing_digest.get_or_insert_with(|| entry.segment_sha256.clone());
            } else {
                new_records.push(record);
            }
        }
        if new_records.is_empty() {
            return Ok(Inserted::Existing {
                segment_sha256: existing_digest.context("empty exact-key batch")?,
            });
        }
        ensure!(
            (self.root_doc.segments.len() as u64) < self.limits.max_segments,
            "retained segment cap reached; source proof retained and new admission refused"
        );
        ensure!(
            self.root_doc
                .total_records
                .checked_add(new_records.len() as u64)
                .is_some_and(|n| n <= self.limits.max_entries),
            "retained record cap reached"
        );
        let generation = self
            .current
            .generation
            .checked_add(1)
            .context("generation overflow")?;
        let mut segment = Segment {
            schema: SEGMENT_SCHEMA.into(),
            generation,
            previous_root_sha256: Some(self.current.root_sha256.clone()),
            records: new_records,
            digest: String::new(),
        };
        segment.digest = segment_digest(&segment)?;
        let segment_bytes = encoded(&segment)?;
        ensure!(
            segment_bytes.len() as u64 <= self.limits.max_segment_bytes,
            "encoded segment exceeds byte cap"
        );
        let mut root = self.root_doc.clone();
        let mut refs = root.segments.clone();
        refs.push(SegmentRef {
            digest: segment.digest.clone(),
            bytes: segment_bytes.len() as u64,
            records: segment.records.len() as u64,
            generation,
            previous_root_sha256: self.current.root_sha256.clone(),
        });
        root.segments = refs;
        root.generation = generation;
        root.previous_root_sha256 = Some(self.current.root_sha256.clone());
        root.total_records = root
            .total_records
            .checked_add(segment.records.len() as u64)
            .context("record count overflow")?;
        root.total_proof_bytes = root
            .total_proof_bytes
            .checked_add(
                segment
                    .records
                    .iter()
                    .map(|r| r.proof_bytes.len() as u64)
                    .sum::<u64>(),
            )
            .context("proof byte count overflow")?;
        let mut shards_to_write = Vec::new();
        let mut changed_buckets = BTreeMap::<u8, Vec<IndexEntry>>::new();
        for (ordinal, record) in segment.records.iter().enumerate() {
            changed_buckets
                .entry(bucket(&record.key))
                .or_default()
                .push(IndexEntry {
                    key: record.key.clone(),
                    segment_sha256: segment.digest.clone(),
                    ordinal: ordinal as u64,
                    record_sha256: sha256_hex(&encoded(record)?),
                });
        }
        for (bucket_id, mut additions) in changed_buckets {
            if let Some(old_ref) = &root.shards[bucket_id as usize] {
                let (old, bytes) = canonical_read::<IndexShard>(
                    &self.shard_path(&old_ref.digest),
                    self.limits.max_shard_bytes,
                )?;
                ensure!(
                    sha256_hex(&bytes) == old_ref.digest,
                    "existing shard digest mismatch"
                );
                additions.extend(old.entries);
            }
            additions.sort_by(|a, b| a.key.cmp(&b.key));
            let shard = IndexShard {
                schema: INDEX_SCHEMA.into(),
                bucket: bucket_id,
                generation,
                entries: additions,
            };
            let shard_bytes = encoded(&shard)?;
            ensure!(
                shard_bytes.len() as u64 <= self.limits.max_shard_bytes,
                "exact index shard cap reached; no proof is removed"
            );
            let shard_digest = sha256_hex(&shard_bytes);
            root.shards[bucket_id as usize] = Some(ShardRef {
                digest: shard_digest.clone(),
                bytes: shard_bytes.len() as u64,
                entries: shard.entries.len() as u64,
            });
            shards_to_write.push((shard_digest, shard_bytes));
        }
        let root_size = encoded(&root)?.len() as u64;
        ensure!(
            root_size <= self.limits.max_root_bytes,
            "root metadata exceeds cap"
        );
        let mut root_with_digest = root;
        root_with_digest.digest = root_digest(&root_with_digest)?;
        let root_bytes = encoded(&root_with_digest)?;
        ensure!(
            root_bytes.len() as u64 <= self.limits.max_root_bytes,
            "root exceeds cap after digest"
        );
        let hot_bytes = root_bytes.len() as u64
            + root_with_digest
                .shards
                .iter()
                .flatten()
                .map(|r| r.bytes)
                .sum::<u64>()
            + encoded(&self.current)?.len() as u64;
        ensure!(
            hot_bytes.saturating_add(segment_bytes.len() as u64 * 2) <= self.limits.max_hot_bytes,
            "hot root and index bytes exceed cap"
        );
        let shard_total = shards_to_write
            .iter()
            .try_fold(0u64, |total, (_, bytes)| {
                total.checked_add(bytes.len() as u64)
            })
            .context("shard byte overflow")?;
        let prospective_disk = self
            .usage
            .total_disk_bytes
            .checked_add(segment_bytes.len() as u64)
            .and_then(|n| n.checked_add(shard_total))
            .and_then(|n| n.checked_add(root_bytes.len() as u64))
            .and_then(|n| n.checked_add(2048)) // current-pointer temp file and directory metadata reserve
            .context("prospective disk usage overflow")?;
        ensure!(
            prospective_disk <= self.limits.max_total_disk_bytes,
            "total durable history cap reached; old root remains current"
        );
        immutable(&self.segment_path(&segment.digest), &segment_bytes)?;
        if crash_after == Some(1) {
            bail!("simulated crash after immutable segment fsync");
        }
        for (shard_digest, shard_bytes) in &shards_to_write {
            immutable(&self.shard_path(shard_digest), shard_bytes)?;
        }
        if crash_after == Some(2) {
            bail!("simulated crash after index shard fsync");
        }
        immutable(
            &self
                .root
                .join("roots")
                .join(format!("{}.json", root_with_digest.digest)),
            &root_bytes,
        )?;
        if crash_after == Some(3) {
            bail!("simulated crash after candidate root fsync");
        }
        let current = Current {
            schema: CURRENT_SCHEMA.into(),
            generation,
            root_sha256: root_with_digest.digest.clone(),
        };
        atomic_write(&self.root.join("CURRENT.json"), &encoded(&current)?)?;
        if crash_after == Some(4) {
            bail!("simulated crash after CURRENT fsync");
        }
        let max_files = self
            .limits
            .max_segments
            .checked_mul((SHARDS as u64) + 2)
            .and_then(|n| n.checked_add(4))
            .context("file count cap overflow")?;
        let total_disk =
            bounded_disk_usage(&self.root, self.limits.max_total_disk_bytes, max_files)?;
        self.current = current;
        self.root_doc = root_with_digest;
        self.usage.generation = generation;
        self.usage.retained_records = self.root_doc.total_records;
        self.usage.retained_segments = self.root_doc.segments.len() as u64;
        self.usage.retained_proof_bytes = self.root_doc.total_proof_bytes;
        self.usage.current_root_bytes = root_bytes.len() as u64;
        self.usage.total_disk_bytes = total_disk;
        self.usage.hot_index_bytes = self
            .root_doc
            .shards
            .iter()
            .flatten()
            .map(|s| s.bytes)
            .sum::<u64>()
            + root_bytes.len() as u64;
        Ok(Inserted::Added {
            generation,
            segment_sha256: segment.digest,
        })
    }
}

fn bounded_disk_usage(root: &Path, cap: u64, max_files: u64) -> Result<u64> {
    let mut total = 0u64;
    let mut files = 0u64;
    for dir in [
        root.to_path_buf(),
        root.join("roots"),
        root.join("segments"),
        root.join("index"),
    ] {
        if !dir.exists() {
            continue;
        }
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_dir() {
                continue;
            }
            ensure!(
                kind.is_file(),
                "non-regular file {} in bounded store",
                entry.path().display()
            );
            files += 1;
            ensure!(
                files <= max_files,
                "on-disk object count cannot be checked within disk cap"
            );
            total = total
                .checked_add(entry.metadata()?.len())
                .context("disk usage overflow")?;
            ensure!(
                total <= cap,
                "retained progress state exceeds total disk cap"
            );
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    fn limits() -> SegmentLimits {
        SegmentLimits {
            max_records_per_segment: 16,
            max_segment_bytes: 32 << 10,
            max_segments: 8,
            max_shard_bytes: 64 << 10,
            max_entries: 64,
            max_startup_bytes: 1 << 20,
            max_total_disk_bytes: 2 << 20,
            max_hot_bytes: 1 << 20,
            max_root_bytes: 64 << 10,
        }
    }
    fn record(key: &str) -> SegmentRecord {
        let proof_bytes = format!("proof:{key}").into_bytes();
        SegmentRecord {
            key: key.into(),
            source_key: format!("source/{key}"),
            class: EvidenceClass::Unknown,
            source_sha256: sha256_hex(&proof_bytes),
            proof_bytes,
            committed_update_key: None,
        }
    }
    fn scratch() -> Result<tempfile::TempDir> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/task-evidence/I48/unit-segment");
        fs::create_dir_all(&root)?;
        Ok(tempfile::tempdir_in(root)?)
    }

    #[test]
    fn append_lookup_reopen_are_hash_bound_and_bounded() -> Result<()> {
        let dir = scratch()?;
        let mut store = SegmentStore::open(dir.path(), limits())?;
        let record = record("attempt/1");
        store.append(vec![record.clone()])?;
        assert_eq!(store.lookup("attempt/1")?, Some(record.clone()));
        let usage = store.usage().clone();
        assert!(usage.lookup_pages <= 2);
        assert!(usage.startup_bytes_read <= limits().max_startup_bytes);
        assert!(usage.total_disk_bytes <= limits().max_total_disk_bytes);
        drop(store);
        let mut reopened = SegmentStore::open(dir.path(), limits())?;
        assert_eq!(reopened.lookup("attempt/1")?, Some(record));
        Ok(())
    }

    #[test]
    fn every_root_publish_boundary_reopens_as_old_or_new_complete_state() -> Result<()> {
        for boundary in 1..=4 {
            let dir = scratch()?;
            let mut store = SegmentStore::open(dir.path(), limits())?;
            let item = record(&format!("crash/{boundary}"));
            assert!(store.append_crash_at(vec![item.clone()], boundary).is_err());
            drop(store);
            let mut recovered = SegmentStore::open(dir.path(), limits())?;
            if boundary < 4 {
                assert_eq!(recovered.usage().generation, 0, "boundary {boundary}");
                assert_eq!(recovered.lookup(&item.key)?, None, "boundary {boundary}");
            } else {
                assert_eq!(recovered.usage().generation, 1, "boundary {boundary}");
                assert_eq!(
                    recovered.lookup(&item.key)?,
                    Some(item),
                    "boundary {boundary}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn conflicting_key_and_caps_refuse_without_moving_root() -> Result<()> {
        let dir = scratch()?;
        let mut caps = limits();
        caps.max_entries = 1;
        caps.max_segments = 1;
        let mut store = SegmentStore::open(dir.path(), caps)?;
        let one = record("attempt/one");
        store.append(vec![one.clone()])?;
        let root = store.root_digest().to_owned();
        assert!(store.append(vec![record("attempt/one")]).is_ok());
        let mut changed = record("attempt/one");
        changed.proof_bytes.push(b'!');
        changed.source_sha256 = sha256_hex(&changed.proof_bytes);
        assert!(store.append(vec![changed]).is_err());
        assert_eq!(store.root_digest(), root);
        assert!(store.append(vec![record("attempt/two")]).is_err());
        assert_eq!(store.root_digest(), root);
        Ok(())
    }
}
