use anyhow::Result;
use jope_learning::canon::sha256_hex;
use jope_learning::progress::checkpoint::{
    rebuild, CheckpointEvidence, CommittedUpdate, CommittedUpdateVerifier, ProofReference,
    SegmentCheckpoint,
};
use jope_learning::progress::segment::{EvidenceClass, SegmentLimits, SegmentRecord, SegmentStore};
use std::fs;
use std::path::PathBuf;

fn scratch() -> Result<tempfile::TempDir> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/task-evidence/I48/integration");
    fs::create_dir_all(&root)?;
    Ok(tempfile::tempdir_in(root)?)
}

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

fn record(key: &str, class: EvidenceClass, update: Option<&str>) -> SegmentRecord {
    let proof_bytes = format!("retained proof bytes for {key}").into_bytes();
    SegmentRecord {
        key: key.into(),
        source_key: format!("source/{key}"),
        class,
        source_sha256: sha256_hex(&proof_bytes),
        proof_bytes,
        committed_update_key: update.map(str::to_owned),
    }
}

#[test]
fn exact_dedup_keeps_proof_and_bounded_index_across_reopen() -> Result<()> {
    let dir = scratch()?;
    let mut store = SegmentStore::open(dir.path(), limits())?;
    let entries = vec![
        record("attempt/a", EvidenceClass::Unknown, None),
        record("attempt/b", EvidenceClass::Failed, None),
        record("attempt/c", EvidenceClass::Committed, Some("update/c")),
    ];
    store.append(entries.clone())?;
    for entry in &entries {
        assert_eq!(store.lookup(&entry.key)?.as_ref(), Some(entry));
        assert!(store.usage().lookup_pages <= 2);
        assert!(store.usage().lookup_bytes <= limits().max_shard_bytes);
    }
    let usage = store.usage().clone();
    assert!(usage.startup_bytes_read <= limits().max_startup_bytes);
    assert!(usage.hot_index_bytes <= limits().max_hot_bytes);
    assert!(usage.total_disk_bytes <= limits().max_total_disk_bytes);
    drop(store);

    let mut reopened = SegmentStore::open(dir.path(), limits())?;
    assert_eq!(reopened.usage().retained_records, 3);
    assert!(matches!(
        reopened.append(entries.clone())?,
        jope_learning::progress::segment::Inserted::Existing { .. }
    ));
    assert_eq!(reopened.lookup("attempt/b")?, Some(entries[1].clone()));
    Ok(())
}

#[test]
fn committed_progress_requires_source_proof_and_duplicate_ack_does_not_count_twice() -> Result<()> {
    struct Accept;
    impl CommittedUpdateVerifier for Accept {
        fn verify_committed_update(&self, _: &CommittedUpdate) -> Result<bool> {
            Ok(true)
        }
    }
    struct Reject;
    impl CommittedUpdateVerifier for Reject {
        fn verify_committed_update(&self, _: &CommittedUpdate) -> Result<bool> {
            Ok(false)
        }
    }
    let r = |kind: &str, key: &str| ProofReference {
        kind: kind.into(),
        key: key.into(),
        sha256: "a".repeat(64),
    };
    let update = CommittedUpdate {
        update_key: "update/1".into(),
        source_commit: "commit-1".into(),
        source_root_sha256: "b".repeat(64),
        checkpoint: r("checkpoint", "cp/1"),
        intent: r("intent", "intent/1"),
        occurrence: r("occurrence", "occ/1"),
        rank: 1,
        attempt: 1,
        epoch: 1,
    };
    let committed = CheckpointEvidence::Committed(Box::new(update));
    let unknown = CheckpointEvidence::Unknown {
        key: "attempt/unknown".into(),
        source: r("attempt", "attempt/unknown"),
    };
    let first = SegmentCheckpoint::new(
        1,
        "commit-1".into(),
        "c".repeat(64),
        vec![committed.clone(), unknown],
    )?;
    let replay = SegmentCheckpoint::new(2, "commit-2".into(), "d".repeat(64), vec![committed])?;
    let rebuilt = rebuild(&[first.clone(), replay.clone()], &Accept)?;
    assert_eq!(rebuilt.committed_updates, 1);
    assert_eq!(rebuilt.held_committed, 0);
    assert_eq!(rebuilt.unknown, 1);

    let unverified = rebuild(&[first, replay], &Reject)?;
    assert_eq!(unverified.committed_updates, 0);
    assert_eq!(unverified.held_committed, 1);
    assert_eq!(unverified.unknown, 1);
    Ok(())
}

#[test]
fn bad_source_digest_and_index_conflict_are_rejected_before_root_advance() -> Result<()> {
    let dir = scratch()?;
    let mut store = SegmentStore::open(dir.path(), limits())?;
    let original = record("source/key", EvidenceClass::Failed, None);
    store.append(vec![original.clone()])?;
    let root = store.root_digest().to_owned();

    let mut bad_proof = record("source/bad", EvidenceClass::Failed, None);
    bad_proof.source_sha256 = "f".repeat(64);
    assert!(store.append(vec![bad_proof]).is_err());
    let mut conflict = original;
    conflict.class = EvidenceClass::Unknown;
    assert!(store.append(vec![conflict]).is_err());
    assert_eq!(store.root_digest(), root);
    Ok(())
}

#[test]
fn saturation_missing_source_and_stale_writer_fail_closed() -> Result<()> {
    let dir = scratch()?;
    let mut caps = limits();
    caps.max_entries = 1;
    caps.max_segments = 1;
    let mut store = SegmentStore::open(dir.path(), caps)?;
    let first = record("retained/one", EvidenceClass::Unknown, None);
    let inserted = store.append(vec![first.clone()])?;
    let root = store.root_digest().to_owned();
    assert!(
        SegmentStore::open(dir.path(), caps).is_err(),
        "a second writer must not open"
    );
    assert!(store
        .append(vec![record("retained/two", EvidenceClass::Failed, None)])
        .is_err());
    assert_eq!(store.root_digest(), root);
    assert_eq!(store.usage().retained_records, 1);

    let digest = match inserted {
        jope_learning::progress::segment::Inserted::Added { segment_sha256, .. } => segment_sha256,
        jope_learning::progress::segment::Inserted::Existing { .. } => {
            anyhow::bail!("expected a new segment")
        }
    };
    fs::remove_file(dir.path().join("segments").join(format!("{digest}.json")))?;
    assert!(
        store.lookup("retained/one").is_err(),
        "missing retained proof cannot be synthesized"
    );
    assert_eq!(store.root_digest(), root);
    Ok(())
}
