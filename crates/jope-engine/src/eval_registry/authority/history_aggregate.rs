//! Complete historical coverage, distinct from deduplicated guard batches.
//! Original payloads are streamed from immutable preservation, never assembled.
use super::*;
use ed25519_dalek::{Signature, VerifyingKey};
use std::os::unix::fs::OpenOptionsExt;
const PIECE: u64 = 64 << 20;
const META: u64 = 4 << 20;
const MAX_LINE: u64 = 1 << 20;
const MAX_STAGES: usize = 4096;
const DOMAIN: &[u8] = b"jope.history_namespace_review.v1\0";
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HistoryArtifact {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryIndex {
    pub artifact: HistoryArtifact,
    pub rows: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OriginalHistoryRole {
    ConvertedDatabase,
    OpaqueProvenance,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalHistorySource {
    pub original_identity: String,
    pub original_sha256: String,
    pub original_bytes: u64,
    pub role: OriginalHistoryRole,
    pub pieces: HistoryIndex,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalHistoryPiece {
    pub ordinal: u64,
    pub offset: u64,
    pub payload: HistoryArtifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryConversion {
    pub original_identity: String,
    pub exporter_receipt: HistoryArtifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryGuardStage {
    pub ordinal: u64,
    pub original_identity: String,
    pub evidence_chunk: HistoryArtifact,
    pub stage_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryAggregateBody {
    pub schema: String,
    pub scope: String,
    pub authority_instance: String,
    pub original_sources: HistoryIndex,
    pub conversions: HistoryIndex,
    pub guard_stages: HistoryIndex,
    pub legacy_fence: HistoryArtifact,
    /// Complete final remote delta, original local histories and sealed opaque
    /// boundaries, approved by the independent reviewer. No filename inference.
    pub namespace_closure: HistoryArtifact,
    pub exporter_release: HistoryArtifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryAggregate {
    pub body: HistoryAggregateBody,
    pub namespace_review: HistoryArtifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryNamespaceReviewBody {
    pub schema: String,
    pub aggregate_body_sha256: String,
    pub namespace_closure_sha256: String,
    pub authority_instance: String,
    pub scope: String,
    pub final_remote_delta_included: bool,
    pub local_original_histories_included: bool,
    pub sealed_material_excluded_from_learning: bool,
    pub all_applicable_prior_work_guarded_or_held: bool,
    pub not_before_unix_ms: u64,
    pub expires_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedHistoryNamespaceReview {
    pub authorization_public_key: String,
    pub body: HistoryNamespaceReviewBody,
    pub signature: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AggregateReceipt {
    schema: String,
    aggregate: HistoryAggregate,
    committed_unix_ms: u64,
    verification: Value,
}
struct Lines {
    reader: BufReader<File>,
    reference: HistoryIndex,
    hash: Sha256,
    bytes: u64,
    rows: u64,
    ended: bool,
}
impl Lines {
    fn new(reference: &HistoryIndex) -> Result<Self> {
        ensure!(
            reference.rows <= 10_000_000 && reference.artifact.bytes <= 16 << 30,
            "history index bound"
        );
        Ok(Self {
            reader: BufReader::new(open(&reference.artifact)?),
            reference: reference.clone(),
            hash: Sha256::new(),
            bytes: 0,
            rows: 0,
            ended: false,
        })
    }
    fn next<T: serde::de::DeserializeOwned>(&mut self) -> Result<Option<T>> {
        ensure!(!self.ended, "history index already consumed");
        let mut line = Vec::new();
        let got = (&mut self.reader)
            .take(MAX_LINE + 1)
            .read_until(b'\n', &mut line)?;
        if got == 0 {
            ensure!(
                self.rows == self.reference.rows
                    && self.bytes == self.reference.artifact.bytes
                    && hex::encode(self.hash.clone().finalize()) == self.reference.artifact.sha256,
                "history index hash/count mismatch"
            );
            self.ended = true;
            return Ok(None);
        }
        ensure!(
            got as u64 <= MAX_LINE && line.last() == Some(&b'\n'),
            "torn/oversize historical index line"
        );
        self.hash.update(&line);
        self.bytes = self
            .bytes
            .checked_add(got as u64)
            .context("index byte overflow")?;
        self.rows += 1;
        ensure!(
            self.rows <= self.reference.rows && self.bytes <= self.reference.artifact.bytes,
            "history index exceeds declaration"
        );
        Ok(Some(serde_json::from_slice(&line)?))
    }
}

#[path = "history_aggregate/artifact.rs"]
mod artifact;
use artifact::*;
#[cfg(test)]
#[path = "history_aggregate/tests.rs"]
mod tests;
#[path = "history_aggregate/validation.rs"]
mod validation;
#[cfg(test)]
use validation::{batch_ranges, checked_chunk, original_source};
