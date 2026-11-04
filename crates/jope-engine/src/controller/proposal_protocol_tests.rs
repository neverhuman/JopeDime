use super::*;

#[test]
fn larger_probe_batches_require_an_explicit_global_batch() {
    assert_eq!(probe_accumulation(8, 256, 2048).unwrap(), 1);
    assert_eq!(probe_accumulation(8, 128, 2048).unwrap(), 2);
    assert!(probe_accumulation(8, 512, 2048).is_err());
    assert_eq!(probe_accumulation(8, 512, 4096).unwrap(), 1);
    assert_eq!(probe_accumulation(8, 1024, 8192).unwrap(), 1);
    for (world, micro, global) in [
        (0, 256, 2048),
        (8, 0, 2048),
        (8, usize::MAX, 2048),
        (8, 256, 0),
    ] {
        assert!(probe_accumulation(world, micro, global).is_err());
    }
}

fn current_receipts(output: &Path) -> serde_json::Value {
    let mut spec = serde_json::json!({"checkpoint_sha256":"frozen"});
    bind_proposal_protocol(&mut spec);
    for name in ["run.json", "complete.json"] {
        write_json(&output.join(name), &spec).unwrap();
    }
    spec
}

#[test]
fn proposal_protocol_accepts_current_and_rejects_each_missing_binding() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path();
    let current = current_receipts(output);
    verify_proposal_protocol(output, true).unwrap();
    for name in ["run.json", "complete.json"] {
        for field in [
            "sampling_protocol",
            "pure_prediction_protocol",
            "gp_assembly_protocol",
            "proposal_generation_schema",
        ] {
            let mut legacy = current.clone();
            legacy.as_object_mut().unwrap().remove(field);
            let path = output.join(name);
            write_json(&path, &legacy).unwrap();
            let before = fs::read(&path).unwrap();
            assert!(verify_proposal_protocol(output, false).is_err());
            assert_eq!(fs::read(&path).unwrap(), before);
            write_json(&path, &current).unwrap();
        }
    }
}

#[test]
fn proposal_grant_hold_and_legacy_protocol_both_reject_before_snapshot_or_gpu_work() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let output = root.join("legacy");
    write_json(
        &output.join("run.json"),
        &serde_json::json!({"sampling_protocol":"shared-incumbents-and-retained-v3"}),
    )
    .unwrap();
    write_json(
        &output.join("complete.json"),
        &serde_json::json!({"world":8}),
    )
    .unwrap();
    let error = propose(
        root,
        &root.join("absent-corpus"),
        &root.join("absent-checkpoint"),
        &output,
        2,
        0,
        false,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("original native GPU grant verifier unavailable"));
    assert!(verify_proposal_protocol(&output, true)
        .unwrap_err()
        .to_string()
        .contains("proposal prediction protocol changed"));
    assert!(!root.join("champion-snapshot.json").exists());
    assert!(!root.join("runs/encoding-service.lock").exists());
}

#[test]
fn proposal_protocol_rejects_unbound_complete_receipt() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path();
    current_receipts(output);
    fs::remove_file(output.join("run.json")).unwrap();
    assert!(verify_proposal_protocol(output, false)
        .unwrap_err()
        .to_string()
        .contains("proposal run identity absent"));

    // Missing rank progress is an expected startup state; malformed persisted
    // progress is not. Keep both cases distinct from each other and from I/O errors.
    let rank_path = output.join("rank-0.json");
    assert!(read_rank_receipt(&rank_path).unwrap().is_none());
    fs::write(&rank_path, br#"{"timestamp":17}"#).unwrap();
    assert_eq!(
        read_rank_receipt(&rank_path).unwrap(),
        Some(serde_json::json!({"timestamp":17}))
    );
    assert!(read_training_rank_progress(&rank_path).is_err());
    fs::write(&rank_path, br#"{"step":4,"timestamp":17}"#).unwrap();
    assert_eq!(
        read_training_rank_progress(&rank_path)
            .unwrap()
            .map(|progress| (progress.step, progress.timestamp)),
        Some((4, 17))
    );
    fs::write(&rank_path, b"{").unwrap();
    assert!(read_rank_receipt(&rank_path)
        .unwrap_err()
        .to_string()
        .contains("parse rank receipt"));
    let directory_error = read_rank_receipt(output).unwrap_err();
    assert!(format!("{directory_error:#}").contains("read rank receipt"));
}
