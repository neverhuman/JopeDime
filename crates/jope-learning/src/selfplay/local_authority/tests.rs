//! Storage-only fixtures. No fixture supplies evidence authentication or publication approval.
use super::super::snapshot::{
    ArtifactRef, CandidateArtifacts, SnapshotProvenance, SNAPSHOT_SCHEMA,
};
use super::*;
use crate::canon::{canonical_json, sha256_hex};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Barrier};

fn fixture() -> (tempfile::TempDir, State) {
    let base =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/task-evidence/P6/local-authority");
    fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(base).unwrap();
    let state = State::at(dir.path());
    (dir, state)
}
fn candidate(state: &State, id: &str) -> Snapshot {
    let mut artifacts = Vec::new();
    let mut parent: Option<String> = None;
    for kind in [
        "execution",
        "raw_result",
        "payload",
        "assessment",
        "capsule",
    ] {
        let bytes = canonical_json(&serde_json::json!({
            "schema": "jope.snapshot-storage-fixture/v1", "kind": kind, "parent_sha256": parent
        }))
        .unwrap();
        let digest = state.put_blob(&bytes).unwrap();
        artifacts.push(ArtifactRef {
            sha256: digest.clone(),
            parent_sha256: parent,
        });
        parent = Some(digest);
    }
    let pin = |name: &str| {
        state
            .put_blob(format!("storage fixture {name}").as_bytes())
            .unwrap()
    };
    Snapshot {
        schema: SNAPSHOT_SCHEMA.into(),
        candidate_id: id.into(),
        expected_generation: 0,
        previous_root: None,
        provenance: SnapshotProvenance {
            source_commit: "1".repeat(40),
            command_identity_sha256: pin("command"),
            input_root_receipt_sha256: pin("input root"),
        },
        pins: CompatibilityPins {
            vocabulary_sha256: pin("vocabulary"),
            support_sha256: pin("support"),
            critic_sha256: pin("critic"),
            replay_sha256: pin("replay"),
            library_sha256: pin("library"),
            checkpoint_sha256: pin("checkpoint"),
            lineage_sha256: pin("lineage"),
            training_intent_sha256: pin("intent"),
            witness_source_sha256: pin("witness"),
            authority_source_sha256: pin("authority"),
            optimizer_epoch: 1,
            physical_versions: vec![],
        },
        artifacts: CandidateArtifacts {
            execution: artifacts.remove(0),
            raw_result: artifacts.remove(0),
            payload: artifacts.remove(0),
            assessment: artifacts.remove(0),
            capsule: artifacts.remove(0),
        },
    }
}
fn proposal(snapshot: &Snapshot) -> SnapshotRoot {
    SnapshotRoot {
        schema: "jope.learning-authority/v1".into(),
        generation: snapshot.expected_generation + 1,
        root_sha256: snapshot.digest().unwrap(),
        previous_root: snapshot.previous_root.clone(),
        pins_sha256: snapshot.pins.digest().unwrap(),
    }
}
fn commit(store: &StateSnapshotAuthority, snapshot: &Snapshot) -> LocalCas {
    store
        .commit_candidate(
            snapshot,
            &proposal(snapshot),
            snapshot.expected_generation,
            snapshot.previous_root.as_deref(),
        )
        .unwrap()
}

#[test]
fn durable_reopen_and_idempotent_commit_preserve_domain_identity_and_truthful_state() {
    let (_dir, state) = fixture();
    let snapshot = candidate(&state, "first");
    let digest = snapshot.digest().unwrap();
    assert_ne!(digest, sha256_hex(&snapshot.encode().unwrap()));
    let store = StateSnapshotAuthority::open(&state, &RetiredVersions::default()).unwrap();
    assert!(store.root().unwrap().is_none());
    assert_eq!(commit(&store, &snapshot), LocalCas::Won);
    assert_eq!(
        store.candidate_state(&digest).unwrap(),
        Some(CandidateState::LocalCommitted)
    );
    let root_bytes = fs::read(state.root().join(ROOT)).unwrap();
    drop(store);
    let reopened =
        StateSnapshotAuthority::open(&State::at(state.root()), &RetiredVersions::default())
            .unwrap();
    assert_eq!(reopened.snapshot(&digest).unwrap(), snapshot);
    assert_eq!(reopened.root().unwrap(), Some(proposal(&snapshot)));
    assert_eq!(reopened.pins_for_root(&digest).unwrap(), snapshot.pins);
    assert_eq!(commit(&reopened, &snapshot), LocalCas::Won);
    assert_eq!(fs::read(state.root().join(ROOT)).unwrap(), root_bytes);
    assert!(reopened
        .set_candidate_state(&digest, CandidateState::Lost)
        .is_err());
    assert!(reopened
        .set_candidate_state(&digest, CandidateState::RemotePublished)
        .is_err());
    assert_eq!(
        reopened.candidate_state(&digest).unwrap(),
        Some(CandidateState::LocalCommitted)
    );
}

#[test]
fn stale_cas_retains_both_candidates_and_cannot_use_loser_pins_as_committed() {
    let (_dir, state) = fixture();
    let store = StateSnapshotAuthority::open(&state, &RetiredVersions::default()).unwrap();
    let first = candidate(&state, "winner");
    let stale = candidate(&state, "stale proposal");
    assert_eq!(commit(&store, &first), LocalCas::Won);
    assert_eq!(
        commit(&store, &stale),
        LocalCas::Lost {
            winner: proposal(&first)
        }
    );
    assert_eq!(store.snapshot(&stale.digest().unwrap()).unwrap(), stale);
    assert_eq!(
        store.candidate_state(&stale.digest().unwrap()).unwrap(),
        Some(CandidateState::Lost)
    );
    assert!(store.pins_for_root(&stale.digest().unwrap()).is_err());
    assert!(store
        .set_candidate_state(&stale.digest().unwrap(), CandidateState::LocalCommitted)
        .is_err());
    assert_eq!(store.root().unwrap(), Some(proposal(&first)));
}

#[test]
fn historical_winner_is_idempotent_and_can_never_be_overwritten_with_lost() {
    let (_dir, state) = fixture();
    let store = StateSnapshotAuthority::open(&state, &RetiredVersions::default()).unwrap();
    let first = candidate(&state, "parent");
    commit(&store, &first);
    let mut child = candidate(&state, "child");
    child.expected_generation = 1;
    child.previous_root = Some(first.digest().unwrap());
    assert_eq!(commit(&store, &child), LocalCas::Won);
    assert_eq!(commit(&store, &first), LocalCas::Won);
    assert_eq!(store.root().unwrap(), Some(proposal(&child)));
    assert!(store
        .set_candidate_state(&first.digest().unwrap(), CandidateState::Lost)
        .is_err());
    assert_eq!(
        store.candidate_state(&first.digest().unwrap()).unwrap(),
        Some(CandidateState::LocalCommitted)
    );
}

#[test]
fn incompatible_checkpoint_pins_refuse_pointer_advance_but_preserve_candidate() {
    let (_dir, state) = fixture();
    let store = StateSnapshotAuthority::open(&state, &RetiredVersions::default()).unwrap();
    let first = candidate(&state, "parent");
    commit(&store, &first);
    let mut child = candidate(&state, "different checkpoint");
    child.expected_generation = 1;
    child.previous_root = Some(first.digest().unwrap());
    child.pins.checkpoint_sha256 = state
        .put_blob(b"different actual fixture checkpoint")
        .unwrap();
    assert!(store
        .commit_candidate(&child, &proposal(&child), 1, child.previous_root.as_deref())
        .is_err());
    assert_eq!(store.snapshot(&child.digest().unwrap()).unwrap(), child);
    assert_eq!(store.root().unwrap(), Some(proposal(&first)));
}

#[test]
fn corrupt_root_or_retained_snapshot_is_refused_on_reopen() {
    for root_file in [false, true] {
        let (_dir, state) = fixture();
        let store = StateSnapshotAuthority::open(&state, &RetiredVersions::default()).unwrap();
        let snapshot = candidate(&state, "committed");
        commit(&store, &snapshot);
        drop(store);
        let path = if root_file {
            state.root().join(ROOT)
        } else {
            state.root().join(format!(
                "selfplay/snapshots/{}.json",
                snapshot.digest().unwrap()
            ))
        };
        fs::write(path, b"altered canonical snapshot bytes").unwrap();
        assert!(StateSnapshotAuthority::open(&state, &RetiredVersions::default()).is_err());
    }
}

#[cfg(unix)]
#[test]
fn symlinked_root_namespace_and_pointer_are_refused_without_external_mutation() {
    use std::os::unix::fs::symlink;
    for case in 0..3 {
        let (_dir, state) = fixture();
        let (_external_dir, outside) = fixture();
        let marker = outside.root().join("marker");
        fs::write(&marker, b"external fixture stays unchanged").unwrap();
        let candidate_state = match case {
            0 => {
                let alias = state.root().join("linked-root");
                symlink(outside.root(), &alias).unwrap();
                State::at(&alias)
            }
            1 => {
                symlink(outside.root(), state.root().join("selfplay")).unwrap();
                State::at(state.root())
            }
            _ => {
                let store =
                    StateSnapshotAuthority::open(&state, &RetiredVersions::default()).unwrap();
                commit(&store, &candidate(&state, "root"));
                fs::rename(
                    state.root().join(ROOT),
                    state.root().join("original-root.json"),
                )
                .unwrap();
                symlink(&marker, state.root().join(ROOT)).unwrap();
                State::at(state.root())
            }
        };
        assert!(
            StateSnapshotAuthority::open(&candidate_state, &RetiredVersions::default()).is_err()
        );
        assert_eq!(
            fs::read(&marker).unwrap(),
            b"external fixture stays unchanged"
        );
        assert!(!outside.root().join("authority").exists());
    }
}

#[test]
fn independent_local_writers_have_one_cas_winner_and_retain_both_proposals() {
    let (_dir, state) = fixture();
    let first = candidate(&state, "racer-a");
    let second = candidate(&state, "racer-b");
    let barrier = Arc::new(Barrier::new(3));
    let spawn = |snapshot: Snapshot| {
        let path = state.root().to_owned();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            let store =
                StateSnapshotAuthority::open(&State::at(&path), &RetiredVersions::default())
                    .unwrap();
            barrier.wait();
            commit(&store, &snapshot)
        })
    };
    let a = spawn(first.clone());
    let b = spawn(second.clone());
    barrier.wait();
    let outcomes = [a.join().unwrap(), b.join().unwrap()];
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| matches!(o, LocalCas::Won))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| matches!(o, LocalCas::Lost { .. }))
            .count(),
        1
    );
    let reopened = StateSnapshotAuthority::open(&state, &RetiredVersions::default()).unwrap();
    let winner = reopened.root().unwrap().unwrap();
    for snapshot in [first, second] {
        let digest = snapshot.digest().unwrap();
        assert_eq!(reopened.snapshot(&digest).unwrap(), snapshot);
        assert_eq!(
            reopened.candidate_state(&digest).unwrap(),
            Some(if winner.root_sha256 == digest {
                CandidateState::LocalCommitted
            } else {
                CandidateState::Lost
            })
        );
    }
}

#[test]
fn deleted_committed_root_is_not_treated_as_a_fresh_store() {
    let (_dir, state) = fixture();
    let store = StateSnapshotAuthority::open(&state, &RetiredVersions::default()).unwrap();
    commit(&store, &candidate(&state, "committed"));
    fs::remove_file(state.root().join(ROOT)).unwrap();
    assert!(store.root().is_err());
    assert!(StateSnapshotAuthority::open(&state, &RetiredVersions::default()).is_err());
}
