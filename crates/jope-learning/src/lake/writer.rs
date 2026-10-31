//! Disposable analytical projection publication using create-only objects and CAS head moves.

use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema identifier for view values.
pub const VIEW_SCHEMA: &str = "jope.lake-view/v1";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
/// `ManifestCommit` is a record for the manifest commit contract in this module: Disposable analytical projection publication using create-only objects and CAS head moves.
pub struct ManifestCommit {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Generation recorded for this manifest commit.
    pub generation: u64,
    /// SHA-256 digest identifying the associated parent.
    pub parent_sha256: Option<String>,
    /// SHA-256 digest identifying the associated source manifest.
    pub source_manifest_sha256: String,
    /// SHA-256 digest identifying the associated view.
    pub view_sha256: String,
    /// Requested recorded for this manifest commit.
    pub requested: u64,
    /// Observed recorded for this manifest commit.
    pub observed: u64,
    /// Excluded recorded for this manifest commit.
    pub excluded: u64,
}

impl ManifestCommit {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == VIEW_SCHEMA, "unsupported lake view schema");
        ensure!(
            is_sha(&self.source_manifest_sha256) && is_sha(&self.view_sha256),
            "invalid manifest digest"
        );
        ensure!(
            self.observed.saturating_add(self.excluded) == self.requested,
            "incomplete view denominators"
        );
        if let Some(parent) = &self.parent_sha256 {
            ensure!(is_sha(parent), "invalid parent digest");
        }
        Ok(())
    }

    /// Computes the canonical SHA-256 digest of this value.
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        Ok(hex::encode(Sha256::digest(serde_json::to_vec(self)?)))
    }
}

/// Interface for the head observation contract used by Disposable analytical projection publication using create-only objects and CAS head moves.
pub trait HeadObservation {
    /// Performs the bytes operation and returns an error if it cannot be completed.
    fn bytes(&self) -> &[u8];
}

/// Minimal draft interface over immutable writes and the existing observed-version CAS. A stale
/// listing is intentionally absent: it can discover objects but cannot authorize publication.
pub trait ProjectionStore {
    /// Public type alias for head values.
    type Head: HeadObservation;
    /// Looks up head and returns the matching value when present.
    fn get_head(&self) -> Result<Option<Self::Head>>;
    /// Looks up immutable and returns the matching value when present.
    fn get_immutable(&self, key: &str) -> Result<Option<Vec<u8>>>;
    /// Must create if absent and verify byte equality when already present (Mirror semantics).
    fn put_immutable_verified(&self, key: &str, bytes: &[u8]) -> Result<()>;
    /// Performs the put head if unchanged operation and returns an error if it cannot be completed.
    fn put_head_if_unchanged(&self, observed: Option<&Self::Head>, bytes: &[u8]) -> Result<()>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
/// `PublishResult` is a record for the publish result contract in this module: Disposable analytical projection publication using create-only objects and CAS head moves.
pub struct PublishResult {
    /// SHA-256 digest identifying the associated commit.
    pub commit_sha256: String,
    /// Manifest key recorded for this publish result.
    pub manifest_key: String,
}

/// Performs the publish operation and returns an error if it cannot be completed.
pub fn publish<S: ProjectionStore>(
    store: &S,
    commit: &ManifestCommit,
    view_bytes: &[u8],
) -> Result<PublishResult> {
    commit.validate()?;
    ensure!(
        hex::encode(Sha256::digest(view_bytes)) == commit.view_sha256,
        "view bytes do not match commit digest"
    );
    let expected_parent = store.get_head()?;
    if let Some(head) = &expected_parent {
        let current = current_commit(store, head)?;
        ensure!(
            Some(current.digest()?) == commit.parent_sha256,
            "commit parent is not the observed head"
        );
        ensure!(
            current.generation.checked_add(1) == Some(commit.generation),
            "generation must advance exactly once"
        );
    } else {
        ensure!(
            commit.parent_sha256.is_none(),
            "initial commit cannot name a parent"
        );
        ensure!(commit.generation == 1, "initial generation must be one");
    }
    let view_key = format!("views/v1/{}.json", commit.view_sha256);
    let commit_bytes = serde_json::to_vec(commit)?;
    let commit_sha256 = commit.digest()?;
    let manifest_key = format!("manifests/v1/{commit_sha256}.json");
    store.put_immutable_verified(&view_key, view_bytes)?;
    store.put_immutable_verified(&manifest_key, &commit_bytes)?;

    // The conditional adapter compares this exact observed version. A concurrent replacement
    // fails the CAS; listing results are not part of this protocol.
    let head_bytes = serde_json::to_vec(&Head {
        schema: "jope.lake-head/v1".into(),
        manifest_key: manifest_key.clone(),
        commit_sha256: commit_sha256.clone(),
    })?;
    store.put_head_if_unchanged(expected_parent.as_ref(), &head_bytes)?;
    Ok(PublishResult {
        commit_sha256,
        manifest_key,
    })
}

/// Read the committed view by following the observed head's manifest pointer. Listings are not
/// consulted, so orphaned immutable objects from interrupted/raced writes remain invisible.
pub fn read_committed<S: ProjectionStore>(store: &S) -> Result<Option<(ManifestCommit, Vec<u8>)>> {
    let Some(head) = store.get_head()? else {
        return Ok(None);
    };
    let commit = current_commit(store, &head)?;
    let view_key = format!("views/v1/{}.json", commit.view_sha256);
    let view = store
        .get_immutable(&view_key)?
        .ok_or_else(|| anyhow::anyhow!("committed view object is missing"))?;
    ensure!(
        hex::encode(Sha256::digest(&view)) == commit.view_sha256,
        "committed view digest mismatch"
    );
    Ok(Some((commit, view)))
}

fn current_commit<S: ProjectionStore>(store: &S, head: &S::Head) -> Result<ManifestCommit> {
    let pointer: Head = serde_json::from_slice(head.bytes())?;
    ensure!(
        pointer.schema == "jope.lake-head/v1",
        "unsupported lake head schema"
    );
    ensure!(
        pointer.manifest_key == format!("manifests/v1/{}.json", pointer.commit_sha256),
        "head manifest locator mismatch"
    );
    let bytes = store
        .get_immutable(&pointer.manifest_key)?
        .ok_or_else(|| anyhow::anyhow!("committed manifest is missing"))?;
    ensure!(
        hex::encode(Sha256::digest(&bytes)) == pointer.commit_sha256,
        "committed manifest digest mismatch"
    );
    let commit: ManifestCommit = serde_json::from_slice(&bytes)?;
    ensure!(
        commit.digest()? == pointer.commit_sha256,
        "committed manifest is invalid"
    );
    Ok(commit)
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Head {
    schema: String,
    manifest_key: String,
    commit_sha256: String,
}

fn is_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Performs the reject listing as authority operation and returns an error if it cannot be completed.
pub fn reject_listing_as_authority() -> Result<()> {
    bail!("object listings are discovery only; an observed head is required")
}
