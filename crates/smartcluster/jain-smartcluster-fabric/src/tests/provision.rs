use super::*;

#[test]
fn one_worker_has_three_disjoint_closed_provisions() {
    let cases = [
        (
            WorkerProvision::CodeRunner,
            [
                WorkloadClass::ContinuousIntegration,
                WorkloadClass::Coding,
                WorkloadClass::Repair,
                WorkloadClass::Maintenance,
            ],
        ),
        (
            WorkerProvision::SmartclusterCompute,
            [
                WorkloadClass::InteractiveResearch,
                WorkloadClass::BatchResearch,
                WorkloadClass::Training,
                WorkloadClass::Evaluation,
            ],
        ),
    ];
    for (provision, admitted) in cases {
        for workload in admitted {
            assert!(provision.admits(workload));
        }
    }
    for workload in [
        WorkloadClass::OneShotInference,
        WorkloadClass::ReservedInference,
    ] {
        assert!(WorkerProvision::DedicatedInference.admits(workload));
    }

    let every_workload = [
        WorkloadClass::ContinuousIntegration,
        WorkloadClass::Coding,
        WorkloadClass::InteractiveResearch,
        WorkloadClass::BatchResearch,
        WorkloadClass::Training,
        WorkloadClass::Evaluation,
        WorkloadClass::OneShotInference,
        WorkloadClass::ReservedInference,
        WorkloadClass::Repair,
        WorkloadClass::Maintenance,
    ];
    for workload in every_workload {
        assert_eq!(
            [
                WorkerProvision::CodeRunner,
                WorkerProvision::SmartclusterCompute,
                WorkerProvision::DedicatedInference,
            ]
            .into_iter()
            .filter(|provision| provision.admits(workload))
            .count(),
            1
        );
    }
}

#[test]
fn governed_projection_rejects_shell_unpinned_images_and_reserved_environment() {
    let mut candidate = projection();
    candidate.argv = vec!["bash".to_owned(), "-c".to_owned(), "echo no".to_owned()];
    assert_eq!(
        VerifiedWorkUnit::verify(
            signed_projection(candidate),
            "hub-issuer",
            hub_key().verifying_key().to_bytes(),
            "hub-issuer",
            hub_key().verifying_key().to_bytes()
        ),
        Err(FabricError::InvalidWorkUnit)
    );
    candidate = projection();
    candidate.image_reference = "registry.invalid/research:latest".to_owned();
    assert_eq!(
        VerifiedWorkUnit::verify(
            signed_projection(candidate),
            "hub-issuer",
            hub_key().verifying_key().to_bytes(),
            "hub-issuer",
            hub_key().verifying_key().to_bytes()
        ),
        Err(FabricError::UnpinnedImage)
    );
    candidate = projection();
    candidate
        .caller_environment
        .insert("JAIN_RUN_ID".to_owned(), "forged".to_owned());
    assert_eq!(
        VerifiedWorkUnit::verify(
            signed_projection(candidate),
            "hub-issuer",
            hub_key().verifying_key().to_bytes(),
            "hub-issuer",
            hub_key().verifying_key().to_bytes()
        ),
        Err(FabricError::ReservedEnvironment)
    );

    let mut forged = signed_projection(projection());
    forged
        .projection
        .argv
        .push("--forged-after-sign".to_owned());
    assert_eq!(
        VerifiedWorkUnit::verify(
            forged,
            "hub-issuer",
            hub_key().verifying_key().to_bytes(),
            "hub-issuer",
            hub_key().verifying_key().to_bytes()
        ),
        Err(FabricError::InvalidSignature)
    );

    let mut forged_class = signed_projection(projection());
    forged_class.projection.workload_class = WorkloadClass::Coding;
    assert_eq!(
        VerifiedWorkUnit::verify(
            forged_class,
            "hub-issuer",
            hub_key().verifying_key().to_bytes(),
            "hub-issuer",
            hub_key().verifying_key().to_bytes()
        ),
        Err(FabricError::InvalidSignature)
    );

    let mut unsafe_identity = projection();
    unsafe_identity.project_id = "../other-project".to_owned();
    assert_eq!(
        VerifiedWorkUnit::verify(
            signed_projection(unsafe_identity),
            "hub-issuer",
            hub_key().verifying_key().to_bytes(),
            "hub-issuer",
            hub_key().verifying_key().to_bytes()
        ),
        Err(FabricError::InvalidWorkUnit)
    );
}

#[test]
fn placement_requires_isolation_storage_egress_and_exact_gpu_without_node_selection() {
    let hub_key = hub_key();
    let node_key = SigningKey::from_bytes(&[12; 32]);
    let mut scheduler = FabricScheduler::new("hub-issuer".to_owned(), hub_key.clone()).unwrap();
    for id in ["unsupported", "no-quorum", "cached", "code-runner"] {
        scheduler
            .trust_node_key(id.to_owned(), node_key.verifying_key().to_bytes())
            .unwrap();
    }
    let mut unsupported = node("unsupported", false, &node_key);
    unsupported.isolated = false;
    unsupported.isolation_evidence_digest = [0; 32];
    unsupported.signature = DomainSignature::sign(
        SignatureAuthority::NodeLease,
        "unsupported".to_owned(),
        "node-key".to_owned(),
        &node_key,
        &unsupported.signed_bytes().unwrap(),
    );
    scheduler.upsert(unsupported).unwrap();
    let mut no_quorum = node("no-quorum", false, &node_key);
    no_quorum.storage_domains.pop_first();
    no_quorum.signature = DomainSignature::sign(
        SignatureAuthority::NodeLease,
        "no-quorum".to_owned(),
        "node-key".to_owned(),
        &node_key,
        &no_quorum.signed_bytes().unwrap(),
    );
    scheduler.upsert(no_quorum).unwrap();
    scheduler.upsert(node("cached", true, &node_key)).unwrap();
    let mut code_runner = node("code-runner", true, &node_key);
    code_runner.provision = WorkerProvision::CodeRunner;
    code_runner.signature = DomainSignature::sign(
        SignatureAuthority::NodeLease,
        "code-runner".to_owned(),
        "node-key".to_owned(),
        &node_key,
        &code_runner.signed_bytes().unwrap(),
    );
    scheduler.upsert(code_runner).unwrap();

    let placement = scheduler.place_and_reserve(&work(), 10_000).unwrap();
    assert_eq!(placement.node_id, "cached");
    assert_eq!(placement.gpu_uuids, ["GPU-cached"]);
    placement
        .verify_for(
            &work(),
            "cached",
            "session-cached",
            hub_key.verifying_key().to_bytes(),
            10_000,
        )
        .unwrap();
    let mut tampered = placement.clone();
    tampered.gpu_uuids.clear();
    assert_eq!(
        tampered.verify_for(
            &work(),
            "cached",
            "session-cached",
            hub_key.verifying_key().to_bytes(),
            10_000,
        ),
        Err(FabricError::InvalidPlacement)
    );

    let mut forged_lease = node("forged", true, &node_key);
    scheduler
        .trust_node_key("forged".to_owned(), node_key.verifying_key().to_bytes())
        .unwrap();
    forged_lease.memory_bytes_available += 1;
    assert_eq!(
        scheduler.upsert(forged_lease),
        Err(FabricError::InvalidSignature)
    );

    assert_eq!(
        scheduler.place_and_reserve(&work(), 10_000),
        Err(FabricError::CapacityExhausted)
    );
}

#[test]
fn signed_stale_lease_cannot_restore_a_reserved_gpu() {
    let hub_key = hub_key();
    let node_key = SigningKey::from_bytes(&[12; 32]);
    let mut scheduler = FabricScheduler::new("hub-issuer".to_owned(), hub_key).unwrap();
    scheduler
        .trust_node_key("cached".to_owned(), node_key.verifying_key().to_bytes())
        .unwrap();
    let stale = node("cached", true, &node_key);
    scheduler.upsert(stale.clone()).unwrap();
    scheduler.place_and_reserve(&work(), 10_000).unwrap();

    assert_eq!(scheduler.upsert(stale), Err(FabricError::ReplayedPosition));
    assert_eq!(
        scheduler.place_and_reserve(&work(), 10_000),
        Err(FabricError::CapacityExhausted)
    );
}

#[test]
fn lease_epoch_revision_and_equal_revision_updates_are_monotonic() {
    let hub_key = hub_key();
    let node_key = SigningKey::from_bytes(&[12; 32]);
    let mut scheduler = FabricScheduler::new("hub-issuer".to_owned(), hub_key).unwrap();
    scheduler
        .trust_node_key("node-1".to_owned(), node_key.verifying_key().to_bytes())
        .unwrap();
    let initial = node("node-1", true, &node_key);
    scheduler.upsert(initial.clone()).unwrap();
    scheduler.upsert(initial.clone()).unwrap();

    let mut newer = initial.clone();
    newer.lease_revision = 2;
    newer.memory_bytes_available = 1_500;
    newer.signature = DomainSignature::sign(
        SignatureAuthority::NodeLease,
        newer.node_id.clone(),
        "node-key".to_owned(),
        &node_key,
        &newer.signed_bytes().unwrap(),
    );
    scheduler.upsert(newer.clone()).unwrap();
    assert_eq!(
        scheduler.upsert(initial),
        Err(FabricError::ReplayedPosition)
    );

    let mut conflicting = newer.clone();
    conflicting.memory_bytes_available += 1;
    conflicting.signature = DomainSignature::sign(
        SignatureAuthority::NodeLease,
        conflicting.node_id.clone(),
        "node-key".to_owned(),
        &node_key,
        &conflicting.signed_bytes().unwrap(),
    );
    assert_eq!(
        scheduler.upsert(conflicting),
        Err(FabricError::ReplayedPosition)
    );

    let mut old_epoch = newer;
    old_epoch.hub_epoch = 1;
    old_epoch.lease_revision = u64::MAX;
    old_epoch.signature = DomainSignature::sign(
        SignatureAuthority::NodeLease,
        old_epoch.node_id.clone(),
        "node-key".to_owned(),
        &node_key,
        &old_epoch.signed_bytes().unwrap(),
    );
    assert_eq!(
        scheduler.upsert(old_epoch),
        Err(FabricError::ReplayedPosition)
    );
}

#[test]
fn duplicate_advertised_gpu_uuid_is_rejected_before_scheduler_mutation() {
    let hub_key = hub_key();
    let node_key = SigningKey::from_bytes(&[12; 32]);
    let mut scheduler = FabricScheduler::new("hub-issuer".to_owned(), hub_key).unwrap();
    scheduler
        .trust_node_key("duplicate".to_owned(), node_key.verifying_key().to_bytes())
        .unwrap();
    let mut duplicate = node("duplicate", true, &node_key);
    duplicate.gpus.push(duplicate.gpus[0].clone());
    duplicate.signature = DomainSignature::sign(
        SignatureAuthority::NodeLease,
        "duplicate".to_owned(),
        "node-key".to_owned(),
        &node_key,
        &duplicate.signed_bytes().unwrap(),
    );

    assert_eq!(
        scheduler.upsert(duplicate),
        Err(FabricError::InvalidGpuRequest)
    );
    assert_eq!(
        scheduler.place_and_reserve(&work(), 10_000),
        Err(FabricError::CapacityExhausted)
    );
}

#[test]
fn aggregate_cached_input_size_overflow_fails_closed_at_verification() {
    let mut overflow = projection();
    overflow.inputs[0].size_bytes = u64::MAX;
    overflow.inputs.push(ArtifactMount {
        artifact_id: "artifact-2".to_owned(),
        artifact_digest: [14; 32],
        mount_path: "/inputs/second".to_owned(),
        size_bytes: 1,
        durable_receipt_digest: [15; 32],
    });
    assert_eq!(
        VerifiedWorkUnit::verify(
            signed_projection(overflow),
            "hub-issuer",
            hub_key().verifying_key().to_bytes(),
            "hub-issuer",
            hub_key().verifying_key().to_bytes(),
        ),
        Err(FabricError::InvalidArtifact)
    );

    let mut boundary = projection();
    boundary.inputs[0].size_bytes = u64::MAX;
    let verified = VerifiedWorkUnit::verify(
        signed_projection(boundary),
        "hub-issuer",
        hub_key().verifying_key().to_bytes(),
        "hub-issuer",
        hub_key().verifying_key().to_bytes(),
    )
    .unwrap();
    let node_key = SigningKey::from_bytes(&[12; 32]);
    let mut scheduler = FabricScheduler::new("hub-issuer".to_owned(), hub_key()).unwrap();
    scheduler
        .trust_node_key("cached".to_owned(), node_key.verifying_key().to_bytes())
        .unwrap();
    scheduler.upsert(node("cached", true, &node_key)).unwrap();
    assert_eq!(
        scheduler
            .place_and_reserve(&verified, 10_000)
            .unwrap()
            .cached_input_bytes,
        u64::MAX
    );
}
