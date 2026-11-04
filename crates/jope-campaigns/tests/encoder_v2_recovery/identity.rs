use super::*;

#[test]
fn identity_roles_backend_and_train_test_boundary_fail_closed() {
    let encoder = encoder();
    let input = input();
    let stable = cache_key(&encoder, &input).unwrap();

    let mut changed = encoder.clone();
    changed.wrapper.sha256 = digest(30);
    assert_ne!(stable, cache_key(&changed, &input).unwrap());
    let mut changed = encoder.clone();
    changed.runtime.memory_policy_sha256 = digest(31);
    assert_ne!(stable, cache_key(&changed, &input).unwrap());
    let mut changed = encoder.clone();
    changed.output.byte_order = "big-endian".to_owned();
    assert!(changed.validate().is_err());

    let mut leaked = input.clone();
    leaked.files_read[0].purpose = "dataset_metadata".to_owned();
    leaked.files_read[0].artifact.path = "dataset/test.csv".to_owned();
    assert!(leaked.validate().is_err());
    let mut unknown = serde_json::to_value(encoder).unwrap();
    unknown["fallback_allowed"] = Value::Bool(true);
    assert!(serde_json::from_value::<FoundationEncoderIdentityV2>(unknown).is_err());
}

#[test]
fn output_and_execution_receipts_reject_nonfinite_fallback_and_package_drift() {
    let encoder = encoder();
    let input = input();
    let bytes = output(0.0);
    let (execution, _) = execution_and_attempt(&encoder, &input, &bytes, &digest(32), 1_000);
    execution.validate_for(&encoder, &input, 1_000).unwrap();

    let mut fallback = execution.clone();
    fallback.observed.actual_backend = "cpu".to_owned();
    assert!(fallback.validate_for(&encoder, &input, 1_000).is_err());
    let mut drift = execution;
    drift
        .observed
        .package_tree_sha256
        .insert("pytorch".to_owned(), digest(33));
    assert!(drift.validate_for(&encoder, &input, 1_000).is_err());

    let mut nonfinite = bytes;
    nonfinite[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(validate_output_bytes(&nonfinite).is_err());
}

#[test]
fn object_store_converges_duplicates_and_rejects_nondeterminism() {
    let directory = tempfile::tempdir().unwrap();
    let store = EncoderV2Store::new(directory.path());
    let encoder = encoder();
    let input = input();
    let bytes = output(0.0);
    let job1 = digest(40);
    let (execution1, attempt1) = execution_and_attempt(&encoder, &input, &bytes, &job1, 1_000);
    let first = store
        .publish_success(
            "tx-1",
            &encoder,
            &input,
            &execution1,
            &attempt1,
            &bytes,
            &job1,
            1_000,
            None,
            None,
            |_| Ok(()),
        )
        .unwrap();
    assert!(matches!(first, PublishOutcome::Published(_)));

    let job2 = digest(41);
    let (execution2, attempt2) = execution_and_attempt(&encoder, &input, &bytes, &job2, 1_001);
    let duplicate = store
        .publish_success(
            "tx-2",
            &encoder,
            &input,
            &execution2,
            &attempt2,
            &bytes,
            &job2,
            1_001,
            None,
            None,
            |_| Ok(()),
        )
        .unwrap();
    assert!(matches!(duplicate, PublishOutcome::Converged(_)));

    let conflicting = output(1.0);
    let job3 = digest(42);
    let (execution3, attempt3) =
        execution_and_attempt(&encoder, &input, &conflicting, &job3, 1_002);
    let error = store
        .publish_success(
            "tx-3",
            &encoder,
            &input,
            &execution3,
            &attempt3,
            &conflicting,
            &job3,
            1_002,
            None,
            None,
            |_| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("nondeterminism"));
    assert_eq!(
        fs::read_dir(directory.path().join("conflicts"))
            .unwrap()
            .count(),
        1
    );
}

#[test]
fn store_recovers_persistent_lock_inode_and_rejects_every_v1_import() {
    let directory = tempfile::tempdir().unwrap();
    let store = EncoderV2Store::new(directory.path());
    store.initialize().unwrap();
    let encoder = encoder();
    let input = input();
    let bytes = output(0.0);
    let job = digest(79);
    let key = cache_key(&encoder, &input).unwrap();
    fs::write(
        directory.path().join("locks").join(format!("{key}.lock")),
        b"stale owner metadata",
    )
    .unwrap();
    let (execution, attempt) = execution_and_attempt(&encoder, &input, &bytes, &job, 1_000);
    let published = store.publish_success(
        "stale-lock-recovery",
        &encoder,
        &input,
        &execution,
        &attempt,
        &bytes,
        &job,
        1_000,
        None,
        None,
        |_| Ok(()),
    );
    assert!(published.is_ok(), "{published:?}");

    let rejected = store.publish_success(
        "v1-import-rejected",
        &encoder,
        &input,
        &execution,
        &attempt,
        &bytes,
        &job,
        1_000,
        Some("legacy-v1-cache"),
        None,
        |_| Ok(()),
    );
    assert!(rejected
        .unwrap_err()
        .to_string()
        .contains("may never be imported"));
}

#[test]
fn publication_crashes_leave_only_valid_bindings_or_no_binding() {
    for (index, crash_step) in [
        PublishStep::Staged,
        PublishStep::ObjectSynced,
        PublishStep::ObjectPublished,
        PublishStep::MetadataPublished,
        PublishStep::BindingPublished,
        PublishStep::AttemptPublished,
        PublishStep::PointerPublished,
    ]
    .into_iter()
    .enumerate()
    {
        let directory = tempfile::tempdir().unwrap();
        let store = EncoderV2Store::new(directory.path());
        let encoder = encoder();
        let input = input();
        let bytes = output(0.0);
        let job = digest(50 + index as u8);
        let (execution, attempt) = execution_and_attempt(&encoder, &input, &bytes, &job, 1_000);
        let error = store
            .publish_success(
                &format!("crash-{index}"),
                &encoder,
                &input,
                &execution,
                &attempt,
                &bytes,
                &job,
                1_000,
                None,
                None,
                |step| {
                    if step == crash_step {
                        bail!("injected crash")
                    }
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("injected crash"));

        let recovered = store.publish_success(
            &format!("recover-{index}"),
            &encoder,
            &input,
            &execution,
            &attempt,
            &bytes,
            &job,
            1_000,
            None,
            None,
            |_| Ok(()),
        );
        assert!(recovered.is_ok(), "{crash_step:?}: {recovered:?}");
    }
}
