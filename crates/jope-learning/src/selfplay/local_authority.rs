//! Durable local snapshot storage. This is not an evidence verifier or an issuer authority.
//! The host selects the store; no request pathname, role string or digest grants admission.
use super::authority::{LocalCas, LocalSnapshotAuthority};
use super::pins::{CompatibilityPins, RetiredVersions};
use super::snapshot::{CandidateState, Snapshot, SnapshotRoot};
use crate::canon::{canonical_json, is_hex64};
use crate::state::State;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

mod files;
use files::LocalFiles;
#[cfg(test)]
mod tests;

const ROOT: &str = "selfplay/authority/ROOT.json";
const MAX_SNAPSHOT: u64 = 16 * 1024 * 1024;
const MAX_CONTROL: u64 = 2 * 1024 * 1024;
const MAX_HISTORY: usize = 4096;

/// Storage implementation for an explicitly selected local State. It verifies canonical bytes,
/// identities, predecessor compatibility and the local CAS, but cannot authenticate learning
/// evidence. Use AuthorityAdapter with a separately authenticated SnapshotEvidenceVerifier for
/// publication. Calling the storage port directly establishes only a local storage fact.
///
/// Remote result states are deliberately refused: this local backend has no remote proof.
/// The root is the commit fact; a crash may leave its derived state index behind, and retry
/// records that same fact idempotently. No serving or issuer state is created here.
pub struct StateSnapshotAuthority {
    files: LocalFiles,
    retired: RetiredVersions,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultState {
    schema: String,
    candidate_sha256: String,
    state: CandidateState,
}

impl StateSnapshotAuthority {
    /// The State root must already be a real directory selected by the host/controller.
    /// Only this backend's fixed local storage directories and process lock are initialized.
    pub fn open(state: &State, retired: &RetiredVersions) -> Result<Self> {
        let store = Self {
            files: LocalFiles::open(state.root())?,
            retired: retired.clone(),
        };
        {
            let _lock = store.files.lock()?;
            store.current_locked()?;
        }
        Ok(store)
    }

    /// Performs the root operation and returns an error if it cannot be completed.
    pub fn root(&self) -> Result<Option<SnapshotRoot>> {
        let _lock = self.files.lock()?;
        Ok(self.current_locked()?.0)
    }
    /// Includes retained losing/uncommitted candidates; presence is not a publication fact.
    pub fn snapshot(&self, digest: &str) -> Result<Snapshot> {
        let _lock = self.files.lock()?;
        self.snapshot_locked(digest)
    }
    /// Performs the candidate state operation and returns an error if it cannot be completed.
    pub fn candidate_state(&self, digest: &str) -> Result<Option<CandidateState>> {
        let _lock = self.files.lock()?;
        let (_, history) = self.current_locked()?;
        let recorded = self.state_locked(digest, &history)?;
        // The durable root ancestry itself proves this fact if a crash preceded indexing it.
        Ok(recorded.or_else(|| {
            history
                .contains(digest)
                .then_some(CandidateState::LocalCommitted)
        }))
    }

    fn snapshot_locked(&self, digest: &str) -> Result<Snapshot> {
        ensure!(is_hex64(digest), "invalid local snapshot digest");
        let bytes = self
            .files
            .read(&format!("selfplay/snapshots/{digest}.json"), MAX_SNAPSHOT)?
            .context("missing retained local snapshot")?;
        let snapshot = Snapshot::decode(&bytes)?;
        snapshot.validate(&self.retired)?;
        ensure!(
            snapshot.digest()? == digest,
            "retained snapshot domain identity differs"
        );
        Ok(snapshot)
    }

    fn current_locked(&self) -> Result<(Option<SnapshotRoot>, BTreeSet<String>)> {
        let Some(bytes) = self.files.read(ROOT, MAX_CONTROL)? else {
            ensure!(
                !self.files.has_results()?,
                "local root missing after candidate results were persisted"
            );
            return Ok((None, BTreeSet::new()));
        };
        let root = SnapshotRoot::decode(&bytes)?;
        let mut next = root.clone();
        let mut history = BTreeSet::new();
        for _ in 0..MAX_HISTORY {
            ensure!(
                history.insert(next.root_sha256.clone()),
                "cyclic local root ancestry"
            );
            let snapshot = self.snapshot_locked(&next.root_sha256)?;
            next.verify_snapshot(&snapshot, &self.retired)?;
            let Some(previous) = &snapshot.previous_root else {
                return Ok((Some(root), history));
            };
            let parent = self.snapshot_locked(previous)?;
            ensure!(
                parent.expected_generation.checked_add(1) == Some(snapshot.expected_generation),
                "local parent generation differs"
            );
            snapshot.pins.ensure_compatible(&parent.pins)?;
            next = SnapshotRoot {
                schema: "jope.learning-authority/v1".into(),
                generation: snapshot.expected_generation,
                root_sha256: previous.clone(),
                previous_root: parent.previous_root.clone(),
                pins_sha256: parent.pins.digest()?,
            };
        }
        anyhow::bail!("local snapshot ancestry exceeds its bound")
    }

    fn state_locked(
        &self,
        digest: &str,
        history: &BTreeSet<String>,
    ) -> Result<Option<CandidateState>> {
        ensure!(is_hex64(digest), "invalid local candidate digest");
        let Some(bytes) = self.files.read(
            &format!("selfplay/candidate-states/{digest}.json"),
            MAX_CONTROL,
        )?
        else {
            return Ok(None);
        };
        let record: ResultState = serde_json::from_slice(&bytes)?;
        ensure!(
            canonical_json(&record)? == bytes
                && record.schema == "jope.learning-candidate-state/v1"
                && record.candidate_sha256 == digest,
            "invalid local candidate state"
        );
        self.snapshot_locked(digest)?;
        Self::check_state(record.state, history.contains(digest))?;
        Ok(Some(record.state))
    }
    fn check_state(state: CandidateState, committed: bool) -> Result<()> {
        ensure!(
            matches!(state, CandidateState::LocalCommitted | CandidateState::Lost),
            "local snapshot authority cannot assert remote publication state"
        );
        ensure!(
            (state == CandidateState::LocalCommitted) == committed,
            "candidate state contradicts durable root ancestry"
        );
        Ok(())
    }
    fn record_state_locked(
        &self,
        digest: &str,
        state: CandidateState,
        history: &BTreeSet<String>,
    ) -> Result<()> {
        Self::check_state(state, history.contains(digest))?;
        if let Some(old) = self.state_locked(digest, history)? {
            ensure!(
                old == state,
                "terminal local candidate state cannot be overwritten"
            );
            return Ok(());
        }
        let record = ResultState {
            schema: "jope.learning-candidate-state/v1".into(),
            candidate_sha256: digest.into(),
            state,
        };
        self.files.write(
            &format!("selfplay/candidate-states/{digest}.json"),
            &canonical_json(&record)?,
            false,
        )
    }
}

impl LocalSnapshotAuthority for StateSnapshotAuthority {
    fn pins_for_root(&self, root_sha256: &str) -> Result<CompatibilityPins> {
        let _lock = self.files.lock()?;
        let (_, history) = self.current_locked()?;
        ensure!(
            history.contains(root_sha256),
            "pins require an already committed local root"
        );
        Ok(self.snapshot_locked(root_sha256)?.pins)
    }
    fn commit_candidate(
        &self,
        snapshot: &Snapshot,
        root: &SnapshotRoot,
        expected_generation: u64,
        expected_previous_root: Option<&str>,
    ) -> Result<LocalCas> {
        let _lock = self.files.lock()?;
        snapshot.validate(&self.retired)?;
        root.verify_snapshot(snapshot, &self.retired)?;
        ensure!(
            snapshot.expected_generation == expected_generation
                && snapshot.previous_root.as_deref() == expected_previous_root,
            "local CAS expectation differs from snapshot"
        );
        let digest = snapshot.digest()?;
        let bytes = snapshot.encode()?;
        ensure!(
            bytes.len() as u64 <= MAX_SNAPSHOT,
            "local snapshot exceeds its byte bound"
        );
        // The filename is the envelope/domain identity, NOT raw sha256(snapshot.encode()).
        // Candidate bytes are durable and immutable before reading/attempting the pointer CAS.
        self.files
            .write(&format!("selfplay/snapshots/{digest}.json"), &bytes, true)?;
        let (current, mut history) = self.current_locked()?;
        if history.contains(&digest) {
            self.record_state_locked(&digest, CandidateState::LocalCommitted, &history)?;
            return Ok(LocalCas::Won);
        }
        if let Some(previous) = expected_previous_root {
            ensure!(
                history.contains(previous),
                "local CAS predecessor was never committed"
            );
            let parent = self.snapshot_locked(previous)?;
            ensure!(
                parent.expected_generation.checked_add(1) == Some(expected_generation),
                "local CAS predecessor generation differs"
            );
            snapshot.pins.ensure_compatible(&parent.pins)?;
        }
        let matches = current
            .as_ref()
            .map(|r| (r.generation, r.root_sha256.as_str()))
            == expected_previous_root.map(|r| (expected_generation, r));
        if matches {
            ensure!(
                self.state_locked(&digest, &history)? != Some(CandidateState::Lost),
                "terminal lost candidate cannot become a winner"
            );
            self.files.write(ROOT, &root.encode()?, false)?;
            history.insert(digest.clone());
            self.record_state_locked(&digest, CandidateState::LocalCommitted, &history)?;
            Ok(LocalCas::Won)
        } else {
            let winner = current.context("local CAS requires a committed predecessor")?;
            self.record_state_locked(&digest, CandidateState::Lost, &history)?;
            Ok(LocalCas::Lost { winner })
        }
    }
    fn set_candidate_state(&self, candidate_sha256: &str, state: CandidateState) -> Result<()> {
        let _lock = self.files.lock()?;
        self.snapshot_locked(candidate_sha256)?;
        let (root, history) = self.current_locked()?;
        ensure!(
            root.is_some(),
            "candidate result requires a committed local root"
        );
        self.record_state_locked(candidate_sha256, state, &history)
    }
}
