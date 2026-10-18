#![cfg(feature = "neural")]
//! Focused successor checkpoint and publication-before-ack recovery cases.

use jope_parallel::training::{
    checkpoint_commit::{self, Phase, PHASES},
    consumption::{self, Header, Journal},
    objective::Supervision,
};
use std::{fs, path::Path};

fn header() -> Header {
    Header {
        schema: consumption::SCHEMA.into(),
        rank: 0,
        world: 1,
        microbatch: 8,
        global_batch: 8,
        accumulation: 1,
        seed: 7,
        sampler_id: "fixture".into(),
        rng: consumption::RNG_ID.into(),
        sampling_offset: 0,
        initial_optimizer_step: 0,
        target_optimizer_step: 4,
        corpus_sha256: "a".repeat(64),
        replay_policy_sha256: None,
        resume_state_sha256: None,
    }
}

fn one_applied_update(path: &Path) -> Journal {
    let mut journal = Journal::create(path, header()).unwrap();
    journal
        .microbatch(
            0,
            0,
            0,
            consumption::dropout_seed(7, 0, 0, 0),
            &(0..8).collect::<Vec<_>>(),
            Supervision {
                rows: 8,
                width: 8,
                supervised_positions: 56,
                row_weight_sum: 8.0,
                ranking_pairs: 28,
                tied_pairs: 0,
                listwise_groups: 1,
            },
        )
        .unwrap();
    journal.applied(0, 0.5).unwrap();
    journal
}

fn two_applied_updates(path: &Path) -> Journal {
    let mut journal = one_applied_update(path);
    journal
        .microbatch(
            1,
            0,
            8,
            consumption::dropout_seed(7, 1, 0, 0),
            &(8..16).collect::<Vec<_>>(),
            Supervision {
                rows: 8,
                width: 8,
                supervised_positions: 56,
                row_weight_sum: 8.0,
                ranking_pairs: 28,
                tied_pairs: 0,
                listwise_groups: 1,
            },
        )
        .unwrap();
    journal.applied(1, 0.5).unwrap();
    journal
}

#[test]
fn publication_before_ack_recovers_once_and_keeps_physical_retry_visible() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rank.jsonl");
    let mut journal = one_applied_update(&path);
    journal.sync().unwrap();
    drop(journal);
    let digest = "b".repeat(64);
    let first = consumption::recover_published_ack(&path, 1, &digest).unwrap();
    assert!(first.appended && first.acknowledged);
    assert_eq!(first.reconciliation.committed_updates, 1);
    assert_eq!(first.reconciliation.commit_events, 1);
    let second = consumption::recover_published_ack(&path, 1, &digest).unwrap();
    assert!(!second.appended && second.acknowledged);
    assert_eq!(second.reconciliation.committed_updates, 1);
    assert_eq!(second.reconciliation.computed.microbatches, 1);
}

#[test]
fn impossible_journal_checkpoint_directions_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rank.jsonl");
    let mut journal = one_applied_update(&path);
    journal.commit(1, &"c".repeat(64)).unwrap();
    drop(journal);
    let bytes = fs::read(&path).unwrap();
    assert!(
        consumption::reconcile(&bytes, Some(0)).is_err(),
        "journal commit ahead of state"
    );
    assert!(
        consumption::reconcile(&bytes, Some(2)).is_err(),
        "state ahead of applied journal"
    );
}

#[test]
fn applied_work_after_published_checkpoint_stays_visible_and_uncommitted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rank.jsonl");
    let mut journal = two_applied_updates(&path);
    journal.sync().unwrap();
    drop(journal);
    let recovered = consumption::recover_published_ack(&path, 1, &"d".repeat(64)).unwrap();
    assert!(recovered.acknowledged && recovered.appended);
    assert_eq!(recovered.reconciliation.committed_through, 1);
    assert_eq!(recovered.reconciliation.applied_through, 2);
    assert_eq!(recovered.reconciliation.uncommitted_applied_updates, 1);
}

#[test]
fn existing_publication_phase_set_remains_complete_and_ordered() {
    assert_eq!(
        PHASES,
        [
            Phase::ArtifactsWritten,
            Phase::ManifestWritten,
            Phase::StagingSynced,
            Phase::Renamed,
            Phase::LatestStaged,
            Phase::LatestPublished
        ]
    );
    assert_eq!(checkpoint_commit::PHASES.len(), 6);
}
