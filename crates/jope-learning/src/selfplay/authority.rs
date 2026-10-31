//! One adapter for local root CAS and conditional remote publication.

use anyhow::{ensure, Context, Result};

use super::pins::RetiredVersions;
use super::snapshot::{CandidateOutcome, CandidateState, Snapshot, SnapshotRoot};
use crate::progress::remote::Remote;

const ROOT_KEY: &str = "selfplay/authority/ROOT.json";
/// The local root implementation owns durable candidate retention and atomic generation CAS.
/// Integrators bind this draft port to the admitted local state store.
pub trait LocalSnapshotAuthority: Send + Sync {
    /// Return the exact pins recorded by an already committed root, never by an alias.
    fn pins_for_root(&self, root_sha256: &str) -> Result<super::pins::CompatibilityPins>;

    /// Persist the candidate immutably before attempting the root CAS. Return `Won` idempotently
    /// if this exact root already won; otherwise return the current winner on a lost race.
    /// Implementations must preserve both candidate bytes.
    fn commit_candidate(
        &self,
        snapshot: &Snapshot,
        root: &SnapshotRoot,
        expected_generation: u64,
        expected_previous_root: Option<&str>,
    ) -> Result<LocalCas>;

    /// Persist a truthful result state for the candidate.
    fn set_candidate_state(&self, candidate_sha256: &str, state: CandidateState) -> Result<()>;
}

/// I11/I14/I27 draft boundary. An integration implementation must verify every referenced
/// artifact exists and hashes to its reference, validate the signed lineage and witness digests,
/// check checkpoint/replay/support compatibility, and authenticate the I27 input-root receipt.
/// There is intentionally no permissive default implementation.
pub trait SnapshotEvidenceVerifier: Send + Sync {
    /// Performs the verify candidate operation and returns an error if it cannot be completed.
    fn verify_candidate(&self, snapshot: &Snapshot) -> Result<()>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `LocalCas` is an enum for the local cas contract in this module: One adapter for local root CAS and conditional remote publication.
pub enum LocalCas {
    /// The `Won` case of `LocalCas`.
    Won,
    /// The `Lost` case of `LocalCas`.
    Lost {
        /// Winner recorded for this outcome.
        winner: SnapshotRoot,
    },
}

/// Routes existing immutable and conditional-pointer primitives through one snapshot authority.
pub struct AuthorityAdapter<'a, L> {
    local: &'a L,
    remote: Option<&'a dyn Remote>,
    retired: &'a RetiredVersions,
    verifier: &'a dyn SnapshotEvidenceVerifier,
}

impl<'a, L: LocalSnapshotAuthority> AuthorityAdapter<'a, L> {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(
        local: &'a L,
        remote: Option<&'a dyn Remote>,
        retired: &'a RetiredVersions,
        verifier: &'a dyn SnapshotEvidenceVerifier,
    ) -> Self {
        Self {
            local,
            remote,
            retired,
            verifier,
        }
    }

    /// Performs the publish operation and returns an error if it cannot be completed.
    pub fn publish(&mut self, snapshot: &Snapshot) -> Result<CandidateOutcome> {
        snapshot.validate(self.retired)?;
        if let Some(parent_root) = &snapshot.previous_root {
            let parent_pins = self.local.pins_for_root(parent_root)?;
            parent_pins.validate(self.retired)?;
            snapshot.pins.ensure_compatible(&parent_pins)?;
        }
        self.verifier.verify_candidate(snapshot)?;
        let snapshot_digest = snapshot.digest()?;
        let pins_digest = snapshot.pins.digest()?;
        let next_generation = snapshot
            .expected_generation
            .checked_add(1)
            .context("snapshot root generation overflow")?;
        let root = SnapshotRoot {
            schema: "jope.learning-authority/v1".into(),
            generation: next_generation,
            root_sha256: snapshot_digest.clone(),
            previous_root: snapshot.previous_root.clone(),
            pins_sha256: pins_digest,
        };
        root.validate()?;
        match self.local.commit_candidate(
            snapshot,
            &root,
            snapshot.expected_generation,
            snapshot.previous_root.as_deref(),
        )? {
            LocalCas::Lost { winner } => {
                winner.validate()?;
                self.local
                    .set_candidate_state(&snapshot_digest, CandidateState::Lost)?;
                Ok(CandidateOutcome::Lost {
                    candidate_sha256: snapshot_digest,
                    winner,
                })
            }
            LocalCas::Won => {
                self.local
                    .set_candidate_state(&snapshot_digest, CandidateState::LocalCommitted)?;
                let Some(remote) = self.remote else {
                    return Ok(CandidateOutcome::LocalCommitted { root });
                };
                self.local
                    .set_candidate_state(&snapshot_digest, CandidateState::RemotePending)?;
                match publish_remote(remote, snapshot, &root, self.retired) {
                    Ok(()) => {
                        self.local.set_candidate_state(
                            &snapshot_digest,
                            CandidateState::RemotePublished,
                        )?;
                        Ok(CandidateOutcome::RemotePublished { root })
                    }
                    Err(error) => {
                        let reason = format!("{error:#}");
                        Ok(CandidateOutcome::RemotePending { root, reason })
                    }
                }
            }
        }
    }
}

fn publish_remote(
    remote: &dyn Remote,
    snapshot: &Snapshot,
    root: &SnapshotRoot,
    retired: &RetiredVersions,
) -> Result<()> {
    let snapshot_bytes = snapshot.encode()?;
    let object_key = format!("selfplay/snapshots/{}.json", root.root_sha256);
    remote
        .put_immutable(&object_key, &snapshot_bytes)
        .with_context(|| format!("upload immutable snapshot {}", root.root_sha256))?;
    let observed = remote.get_observed(ROOT_KEY)?;
    let found = observed
        .as_ref()
        .map(|object| SnapshotRoot::decode(&object.bytes))
        .transpose()
        .context("decode remote self-play root")?;
    if let Some(found) = &found {
        let parent_key = format!("selfplay/snapshots/{}.json", found.root_sha256);
        let parent_bytes = remote.get(&parent_key)?.ok_or_else(|| {
            anyhow::anyhow!("remote root points at missing snapshot {parent_key}")
        })?;
        let parent = Snapshot::decode(&parent_bytes)?;
        found.verify_snapshot(&parent, retired)?;
        snapshot.pins.ensure_compatible(&parent.pins)?;
    }
    ensure!(
        found
            .as_ref()
            .map(|found| (found.generation, found.root_sha256.as_str()))
            == snapshot
                .previous_root
                .as_ref()
                .map(|previous| (snapshot.expected_generation, previous.as_str())),
        "remote generation/root differs from expected CAS predecessor"
    );
    let root_bytes = root.encode()?;
    if let Err(write_error) = remote.put_pointer(ROOT_KEY, &root_bytes, observed.as_ref()) {
        let current = remote
            .get_observed(ROOT_KEY)?
            .map(|object| SnapshotRoot::decode(&object.bytes))
            .transpose()
            .context("read remote root after conditional write error")?;
        if current.as_ref() == Some(root) {
            return Ok(());
        }
        return Err(write_error)
            .context("conditional remote root advance did not commit this snapshot");
    }
    Ok(())
}
