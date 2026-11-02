use super::*;

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
/// Result of appending records to the immutable progress segment store.
pub enum SegmentInsertOutcome {
    /// The `Added` case of `SegmentInsertOutcome`.
    Added {
        /// Generation recorded for this inserted.
        generation: u64,
        /// SHA-256 digest identifying the associated segment.
        segment_sha256: String,
    },
    /// The `Existing` case of `SegmentInsertOutcome`.
    Existing {
        /// SHA-256 digest identifying the associated segment.
        segment_sha256: String,
    },
}

/// Backward-compatible name for [`SegmentInsertOutcome`].
pub type Inserted = SegmentInsertOutcome;
