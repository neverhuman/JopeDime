use super::*;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "state", content = "value", rename_all = "snake_case")]
/// `VersionId` is an enum for the version ID contract in this module: Offline model of exact-version dependencies and retention promises.
pub enum VersionId {
    /// The `Explicit` case of `VersionId`.
    Explicit(String),
    /// The `Null` case of `VersionId`.
    Null,
    /// The `Unknown` case of `VersionId`.
    Unknown,
}

/// Identity of one physical stored version. Content identity never substitutes for version identity.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct StrongPhysicalRef {
    /// Account recorded for this strong physical ref.
    pub account: String,
    /// Bucket recorded for this strong physical ref.
    pub bucket: String,
    /// Key recorded for this strong physical ref.
    pub key: String,
    /// Identifier for the associated version.
    pub version_id: VersionId,
    /// Sha-256 recorded for this strong physical ref.
    pub sha256: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
}

impl StrongPhysicalRef {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.account.is_empty() && !self.bucket.is_empty() && !self.key.is_empty(),
            "physical reference needs account, bucket, and key"
        );
        ensure!(
            self.account.len() == 12 && self.account.bytes().all(|b| b.is_ascii_digit()),
            "physical reference account must be a twelve-digit identity"
        );
        if let VersionId::Explicit(id) = &self.version_id {
            ensure!(!id.is_empty(), "explicit version id is empty");
        }
        ensure!(
            valid_digest(&self.sha256),
            "invalid physical content digest"
        );
        Ok(())
    }

    fn exact_version(&self) -> bool {
        matches!(self.version_id, VersionId::Explicit(_))
    }
}

/// Historical identity metadata. By itself, it does not keep physical bytes pinned.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceRef {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identity recorded for this provenance ref.
    pub identity: String,
    /// Sha-256 recorded for this provenance ref.
    pub sha256: String,
    /// Description recorded for this provenance ref.
    pub description: String,
}

impl ProvenanceRef {
    fn validate(&self) -> Result<()> {
        ensure!(!self.schema.is_empty(), "provenance schema is missing");
        ensure!(!self.identity.is_empty(), "provenance identity is missing");
        ensure!(valid_digest(&self.sha256), "invalid provenance digest");
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
/// `PromiseKind` is an enum for the promise kind contract in this module: Offline model of exact-version dependencies and retention promises.
pub enum PromiseKind {
    /// The `ExactResume` case of `PromiseKind`.
    ExactResume,
    /// The `Rollback` case of `PromiseKind`.
    Rollback,
    /// The `ExactRetraining` case of `PromiseKind`.
    ExactRetraining,
    /// The `ClaimReproduction` case of `PromiseKind`.
    ClaimReproduction,
}

impl PromiseKind {
    fn restore_capability(self) -> &'static str {
        match self {
            Self::ExactResume => "resume",
            Self::Rollback => "rollback",
            Self::ExactRetraining => "retrain",
            Self::ClaimReproduction => "reproduce_claim",
        }
    }
}

/// Runtime evidence for replacing one physical dependency while fulfilling the same promise.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRestoreProof {
    /// Provenance source for this value.
    pub source: StrongPhysicalRef,
    /// Replacement recorded for this runtime restore proof.
    pub replacement: StrongPhysicalRef,
    /// Capability recorded for this runtime restore proof.
    pub capability: String,
    /// Runtime tested recorded for this runtime restore proof.
    pub runtime_tested: bool,
    /// SHA-256 digest of the receipt that authenticates this result.
    pub receipt_sha256: String,
    /// Pin generation recorded for this runtime restore proof.
    pub pin_generation: String,
    /// Readers migrated recorded for this runtime restore proof.
    pub readers_migrated: bool,
}

impl RuntimeRestoreProof {
    fn validates_for(
        &self,
        kind: PromiseKind,
        source: &StrongPhysicalRef,
        generation: &str,
    ) -> bool {
        self.source == *source
            && self.replacement.exact_version()
            && self.capability == kind.restore_capability()
            && self.runtime_tested
            && self.readers_migrated
            && self.pin_generation == generation
            && valid_digest(&self.receipt_sha256)
            && valid_digest(&self.replacement.sha256)
    }
}
