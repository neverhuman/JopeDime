use anyhow::{bail, Result};
use jope_datasets::store::{ObservedObject, PutOutcome};
use jope_learning::mirror::Mirror;
use jope_learning::progress::remote::Remote;
use jope_learning::selfplay::authority::{
    AuthorityAdapter, LocalCas, LocalSnapshotAuthority, SnapshotEvidenceVerifier,
};
use jope_learning::selfplay::pins::{CompatibilityPins, PhysicalVersion, RetiredVersions};
use jope_learning::selfplay::snapshot::{
    ArtifactRef, CandidateArtifacts, CandidateOutcome, CandidateState, Snapshot,
    SnapshotProvenance, SnapshotRoot, SNAPSHOT_SCHEMA,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

#[derive(Default)]
struct MemoryAuthority(Mutex<MemoryState>);

#[derive(Default)]
struct MemoryState {
    root: Option<SnapshotRoot>,
    candidates: BTreeMap<String, Vec<u8>>,
    pins: BTreeMap<String, CompatibilityPins>,
    states: BTreeMap<String, CandidateState>,
}

struct FixtureVerifier;

impl SnapshotEvidenceVerifier for FixtureVerifier {
    fn verify_candidate(&self, snapshot: &Snapshot) -> Result<()> {
        anyhow::ensure!(
            snapshot.provenance.input_root_receipt_sha256.len() == 64,
            "fixture provenance receipt missing"
        );
        Ok(())
    }
}

impl LocalSnapshotAuthority for MemoryAuthority {
    fn pins_for_root(&self, root_sha256: &str) -> Result<CompatibilityPins> {
        self.0
            .lock()
            .unwrap()
            .pins
            .get(root_sha256)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("unknown committed root {root_sha256}"))
    }

    fn commit_candidate(
        &self,
        snapshot: &Snapshot,
        root: &SnapshotRoot,
        expected_generation: u64,
        expected_previous_root: Option<&str>,
    ) -> Result<LocalCas> {
        let digest = snapshot.digest()?;
        let mut state = self.0.lock().unwrap();
        state.candidates.insert(digest.clone(), snapshot.encode()?);
        let matches = state.root.as_ref().map_or(
            expected_generation == 0 && expected_previous_root.is_none(),
            |current| {
                current.generation == expected_generation
                    && Some(current.root_sha256.as_str()) == expected_previous_root
            },
        );
        if !matches {
            if state.root.as_ref() == Some(root) {
                return Ok(LocalCas::Won);
            }
            return Ok(LocalCas::Lost {
                winner: state
                    .root
                    .clone()
                    .expect("mismatched expectation has a root"),
            });
        }
        state.pins.insert(digest, snapshot.pins.clone());
        state.root = Some(root.clone());
        Ok(LocalCas::Won)
    }

    fn set_candidate_state(
        &self,
        root_sha256: &str,
        candidate_state: CandidateState,
    ) -> Result<()> {
        self.0
            .lock()
            .unwrap()
            .states
            .insert(root_sha256.to_owned(), candidate_state);
        Ok(())
    }
}

impl MemoryAuthority {
    fn state(&self) -> std::sync::MutexGuard<'_, MemoryState> {
        self.0.lock().unwrap()
    }
}

struct TestRemote {
    inner: Mirror,
    deny_immutable: AtomicBool,
}

impl TestRemote {
    fn new(deny_immutable: bool) -> Result<Self> {
        Ok(Self {
            inner: Mirror::open("memory:///")?,
            deny_immutable: AtomicBool::new(deny_immutable),
        })
    }

    fn set_denied(&self, denied: bool) {
        self.deny_immutable.store(denied, Ordering::SeqCst);
    }
}

impl Remote for TestRemote {
    fn url(&self) -> &str {
        &self.inner.url
    }

    fn put_immutable(&self, key: &str, bytes: &[u8]) -> Result<PutOutcome> {
        if self.deny_immutable.load(Ordering::SeqCst) {
            bail!("injected remote upload denial");
        }
        self.inner.put_immutable(key, bytes)
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.inner.get(key)
    }

    fn get_observed(&self, key: &str) -> Result<Option<ObservedObject>> {
        self.inner.get_observed(key)
    }

    fn put_pointer(
        &self,
        key: &str,
        bytes: &[u8],
        observed: Option<&ObservedObject>,
    ) -> Result<()> {
        self.inner.put_pointer(key, bytes, observed)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }
}

fn digest(byte: u8) -> String {
    format!("{:x}", byte % 16).repeat(64)
}

fn candidate(id: &str, vocabulary: u8) -> Snapshot {
    let execution = digest(b'f');
    let raw = digest(b'g');
    let payload = digest(b'h');
    let assessment = digest(b'i');
    Snapshot {
        schema: SNAPSHOT_SCHEMA.into(),
        candidate_id: id.into(),
        expected_generation: 0,
        previous_root: None,
        provenance: SnapshotProvenance {
            source_commit: "1".repeat(40),
            command_identity_sha256: digest(b'k'),
            input_root_receipt_sha256: digest(b'l'),
        },
        pins: CompatibilityPins {
            vocabulary_sha256: digest(vocabulary),
            support_sha256: digest(b'a'),
            critic_sha256: digest(b'b'),
            replay_sha256: digest(b'c'),
            library_sha256: digest(b'd'),
            checkpoint_sha256: digest(b'e'),
            lineage_sha256: digest(b'm'),
            training_intent_sha256: digest(b'n'),
            witness_source_sha256: digest(b'o'),
            authority_source_sha256: digest(b'p'),
            optimizer_epoch: 17,
            physical_versions: vec![PhysicalVersion {
                resource: "checkpoint".into(),
                version_id: "version-17".into(),
            }],
        },
        artifacts: CandidateArtifacts {
            execution: ArtifactRef {
                sha256: execution.clone(),
                parent_sha256: None,
            },
            raw_result: ArtifactRef {
                sha256: raw.clone(),
                parent_sha256: Some(execution),
            },
            payload: ArtifactRef {
                sha256: payload.clone(),
                parent_sha256: Some(raw),
            },
            assessment: ArtifactRef {
                sha256: assessment.clone(),
                parent_sha256: Some(payload),
            },
            capsule: ArtifactRef {
                sha256: digest(b'j'),
                parent_sha256: Some(assessment),
            },
        },
    }
}

#[test]
fn two_eligible_children_cas_one_winner_and_preserve_loser() -> Result<()> {
    let retired = Arc::new(RetiredVersions::default());
    let verifier = Arc::new(FixtureVerifier);
    let local = Arc::new(MemoryAuthority::default());
    let parent = candidate("shared-parent", b'0');
    {
        let mut authority =
            AuthorityAdapter::new(local.as_ref(), None, retired.as_ref(), verifier.as_ref());
        assert!(matches!(
            authority.publish(&parent)?,
            CandidateOutcome::LocalCommitted { .. }
        ));
    }
    let parent_root = parent.digest()?;
    let mut first = candidate("child-a", b'0');
    first.expected_generation = 1;
    first.previous_root = Some(parent_root.clone());
    let mut second = candidate("child-b", b'0');
    second.expected_generation = 1;
    second.previous_root = Some(parent_root);
    let first_digest = first.digest()?;
    let second_digest = second.digest()?;
    let barrier = Arc::new(Barrier::new(3));
    let spawn = |snapshot: Snapshot| {
        let local = Arc::clone(&local);
        let retired = Arc::clone(&retired);
        let verifier = Arc::clone(&verifier);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            let mut authority =
                AuthorityAdapter::new(local.as_ref(), None, retired.as_ref(), verifier.as_ref());
            authority.publish(&snapshot)
        })
    };
    let first_join = spawn(first);
    let second_join = spawn(second);
    barrier.wait();
    let outcomes = [first_join.join().unwrap()?, second_join.join().unwrap()?];
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, CandidateOutcome::LocalCommitted { .. }))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, CandidateOutcome::Lost { .. }))
            .count(),
        1
    );
    let state = local.state();
    let winner = state.root.as_ref().unwrap().root_sha256.as_str();
    assert!(winner == first_digest || winner == second_digest);
    assert!(state.candidates.contains_key(&first_digest));
    assert!(state.candidates.contains_key(&second_digest));
    assert_eq!(
        state.states.get(winner),
        Some(&CandidateState::LocalCommitted)
    );
    let loser = if winner == first_digest {
        &second_digest
    } else {
        &first_digest
    };
    assert_eq!(state.states.get(loser), Some(&CandidateState::Lost));
    Ok(())
}

#[test]
fn mixed_vocabulary_and_retired_live_version_are_refused() -> Result<()> {
    let verifier = FixtureVerifier;
    let retired = RetiredVersions::default();
    let parent = candidate("parent", b'0');
    let parent_digest = parent.digest()?;
    let local = MemoryAuthority::default();
    {
        let mut authority = AuthorityAdapter::new(&local, None, &retired, &verifier);
        assert!(matches!(
            authority.publish(&parent)?,
            CandidateOutcome::LocalCommitted { .. }
        ));
    }
    let mut mixed = candidate("mixed-vocabulary", b'1');
    mixed.expected_generation = 1;
    mixed.previous_root = Some(parent_digest.clone());
    {
        let mut authority = AuthorityAdapter::new(&local, None, &retired, &verifier);
        assert!(authority.publish(&mixed).is_err());
    }
    assert_eq!(
        local.state().root.as_ref().unwrap().root_sha256,
        parent_digest
    );

    let retired_live = RetiredVersions::new([("checkpoint".into(), "version-17".into())]);
    let mut child = candidate("retired-live-version", b'0');
    child.expected_generation = 1;
    child.previous_root = Some(parent_digest.clone());
    let other_local = MemoryAuthority::default();
    {
        let mut authority = AuthorityAdapter::new(&other_local, None, &retired, &verifier);
        authority.publish(&parent)?;
    }
    {
        let mut authority = AuthorityAdapter::new(&other_local, None, &retired_live, &verifier);
        assert!(authority.publish(&child).is_err());
    }
    assert_eq!(
        other_local.state().root.as_ref().unwrap().root_sha256,
        parent_digest
    );
    Ok(())
}

#[test]
fn denied_remote_upload_keeps_truthful_pending_state_and_can_retry() -> Result<()> {
    let retired = RetiredVersions::default();
    let remote = TestRemote::new(true)?;
    let verifier = FixtureVerifier;
    let local = MemoryAuthority::default();
    let snapshot = candidate("local-only-until-retry", b'0');
    let digest = snapshot.digest()?;
    let outcome = {
        let mut authority = AuthorityAdapter::new(&local, Some(&remote), &retired, &verifier);
        authority.publish(&snapshot)?
    };
    assert!(matches!(outcome, CandidateOutcome::RemotePending { .. }));
    let state = local.state();
    assert_eq!(state.root.as_ref().unwrap().root_sha256, digest);
    assert_eq!(
        state.states.get(&digest),
        Some(&CandidateState::RemotePending)
    );
    assert!(state.candidates.contains_key(&digest));
    drop(state);
    remote.set_denied(false);
    let retry = {
        let mut authority = AuthorityAdapter::new(&local, Some(&remote), &retired, &verifier);
        authority.publish(&snapshot)?
    };
    assert!(matches!(retry, CandidateOutcome::RemotePublished { .. }));
    assert!(remote.get("selfplay/authority/ROOT.json")?.is_some());
    assert_eq!(
        local.state().states.get(&digest),
        Some(&CandidateState::RemotePublished)
    );
    Ok(())
}

#[test]
fn successful_remote_publication_uses_the_conditional_root_pointer() -> Result<()> {
    let retired = RetiredVersions::default();
    let remote = TestRemote::new(false)?;
    let verifier = FixtureVerifier;
    let local = MemoryAuthority::default();
    let snapshot = candidate("published", b'0');
    let outcome = {
        let mut authority = AuthorityAdapter::new(&local, Some(&remote), &retired, &verifier);
        authority.publish(&snapshot)?
    };
    assert!(matches!(outcome, CandidateOutcome::RemotePublished { .. }));
    assert!(remote.get("selfplay/authority/ROOT.json")?.is_some());
    Ok(())
}

#[test]
fn snapshot_codec_refuses_noncanonical_bytes_and_has_no_filename_authority() -> Result<()> {
    let snapshot = candidate("name-is-not-authority", b'0');
    let bytes = snapshot.encode()?;
    assert_eq!(Snapshot::decode(&bytes)?, snapshot);
    let envelope_value: serde_json::Value = serde_json::from_slice(&bytes)?;
    let spaced = serde_json::to_vec_pretty(&envelope_value)?;
    assert!(Snapshot::decode(&spaced).is_err());
    let root = SnapshotRoot {
        schema: "jope.learning-authority/v1".into(),
        generation: 1,
        root_sha256: snapshot.digest()?,
        previous_root: None,
        pins_sha256: snapshot.pins.digest()?,
    };
    assert_eq!(SnapshotRoot::decode(&root.encode()?)?, root);
    root.verify_snapshot(&snapshot, &RetiredVersions::default())?;
    let mut renamed_payload = snapshot.clone();
    renamed_payload.candidate_id = "different-candidate-name".into();
    assert!(root
        .verify_snapshot(&renamed_payload, &RetiredVersions::default())
        .is_err());
    Ok(())
}

#[test]
fn artifact_chain_refuses_cycles_and_repeated_objects() {
    let mut snapshot = candidate("cycle", b'0');
    snapshot.artifacts.raw_result.sha256 = snapshot.artifacts.execution.sha256.clone();
    snapshot.artifacts.payload.parent_sha256 = Some(snapshot.artifacts.raw_result.sha256.clone());
    assert!(snapshot.artifacts.validate().is_err());
}
