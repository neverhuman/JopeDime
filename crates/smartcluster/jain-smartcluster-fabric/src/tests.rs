use std::{
    collections::BTreeSet,
    fs,
    os::unix::fs::{symlink, MetadataExt, PermissionsExt},
};

use ulid::Ulid;

use super::*;

fn hub_key() -> SigningKey {
    SigningKey::from_bytes(&[11; 32])
}

fn projection() -> WorkUnitProjection {
    let image_digest = [2; 32];
    WorkUnitProjection {
        run_id: Ulid::from(1_u128),
        project_id: "project-1".to_owned(),
        workload_class: WorkloadClass::Training,
        image_reference: format!(
            "registry.invalid/research@sha256:{}",
            hex_digest(image_digest)
        ),
        image_digest,
        argv: vec!["/app/run".to_owned(), "--exact".to_owned()],
        smoke_argv: vec!["/app/run".to_owned(), "--smoke".to_owned()],
        smoke_timeout_ms: 1_000,
        smoke_only: false,
        working_directory: "workspace".to_owned(),
        caller_environment: BTreeMap::from([("TRIAL".to_owned(), "7".to_owned())]),
        inputs: vec![ArtifactMount {
            artifact_id: "artifact-1".to_owned(),
            artifact_digest: [4; 32],
            mount_path: "/inputs/data".to_owned(),
            size_bytes: 100,
            durable_receipt_digest: [5; 32],
        }],
        outputs: vec![OutputDeclaration {
            relative_path: "result.json".to_owned(),
            media_type: "application/json".to_owned(),
            maximum_bytes: 1_000,
            required: true,
        }],
        limits: ResourceLimits {
            cpu_millicores: 1_000,
            memory_bytes: 1_000,
            scratch_bytes: 1_000,
            pids: 32,
            wall_time_ms: 10_000,
            read_bytes: 1_000,
            write_bytes: 1_000,
            io_operations: 100,
        },
        gpu: Some(GpuRequirement {
            model: Some("RTX 3090".to_owned()),
            count: 1,
            minimum_vram_bytes: 500,
        }),
        open_outbound: true,
        network_bytes: 1_000,
        network_connections: 10,
        network_deadline_ms: 5_000,
        stdout_bytes: 1_000,
        stderr_bytes: 1_000,
        minimum_writable_domains: 2,
        attempt: 1,
        attempt_fence: 9,
        hub_epoch: 2,
        control_position: 3,
        replay_nonce: [6; 32],
    }
}

fn signed_projection(projection: WorkUnitProjection) -> SignedWorkUnitProjection {
    let key = hub_key();
    let issuer = "hub-issuer".to_owned();
    SignedWorkUnitProjection {
        image_signature: DomainSignature::sign(
            SignatureAuthority::OciImage,
            issuer.clone(),
            "hub-key".to_owned(),
            &key,
            &image_signed_bytes(&projection.image_reference, projection.image_digest).unwrap(),
        ),
        work_unit_signature: DomainSignature::sign(
            SignatureAuthority::WorkUnitIssuer,
            issuer.clone(),
            "hub-key".to_owned(),
            &key,
            &projection_signed_bytes(&projection).unwrap(),
        ),
        projection,
        issuer,
    }
}

fn work() -> VerifiedWorkUnit {
    VerifiedWorkUnit::verify(
        signed_projection(projection()),
        "hub-issuer",
        hub_key().verifying_key().to_bytes(),
        "hub-issuer",
        hub_key().verifying_key().to_bytes(),
    )
    .unwrap()
}

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

fn node(id: &str, cached: bool, key: &SigningKey) -> NodeLease {
    let mut lease = NodeLease {
        node_id: id.to_owned(),
        session_id: format!("session-{id}"),
        provision: WorkerProvision::SmartclusterCompute,
        lease_id: format!("lease-{id}"),
        lease_expires_at_ms: 20_000,
        hub_epoch: 2,
        lease_revision: 1,
        last_control_position: 1,
        isolated: true,
        isolation_evidence_digest: [7; 32],
        open_outbound_enforced: true,
        cpu_millicores_available: 2_000,
        memory_bytes_available: 2_000,
        scratch_bytes_available: 2_000,
        pids_available: 64,
        storage_domains: BTreeSet::from(["domain-a".to_owned(), "domain-b".to_owned()]),
        cached_artifacts: if cached {
            BTreeSet::from(["artifact-1".to_owned()])
        } else {
            BTreeSet::new()
        },
        gpus: vec![GpuDevice {
            uuid: format!("GPU-{id}"),
            model: "RTX 3090".to_owned(),
            free_vram_bytes: 1_000,
            exclusively_available: true,
        }],
        advertisement_digest: [8; 32],
        signature: empty_signature(SignatureAuthority::NodeLease),
    };
    lease.signature = DomainSignature::sign(
        SignatureAuthority::NodeLease,
        id.to_owned(),
        "node-key".to_owned(),
        key,
        &lease.signed_bytes().unwrap(),
    );
    lease
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

#[test]
fn replay_cancel_and_duplicate_reports_are_fenced() {
    let run_id = Ulid::from(8_u128);
    let root = tempfile::tempdir().unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let mut journal = ReplayJournal::open(root.path().to_path_buf()).unwrap();
    let assignment = AttemptFence {
        attempt: 1,
        fence: 4,
        hub_epoch: 2,
        control_position: 10,
    };
    journal
        .accept_assignment(run_id, assignment.clone())
        .unwrap();
    assert_eq!(
        journal.accept_assignment(run_id, assignment),
        Err(FabricError::ReplayedPosition)
    );
    assert_eq!(
        journal.accept_cancel(
            run_id,
            AttemptFence {
                attempt: 1,
                fence: 5,
                hub_epoch: 2,
                control_position: 11,
            }
        ),
        Err(FabricError::FenceRejected)
    );
    journal
        .accept_cancel(
            run_id,
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 11,
            },
        )
        .unwrap();
    assert!(journal.accept_terminal_receipt(run_id, [9; 32]).unwrap());
    assert!(!journal.accept_terminal_receipt(run_id, [9; 32]).unwrap());
    assert_eq!(
        journal.accept_terminal_receipt(run_id, [10; 32]),
        Err(FabricError::ConflictingWorkerReport)
    );
    drop(journal);
    let mut recovered = ReplayJournal::open(root.path().to_path_buf()).unwrap();
    assert!(!recovered.accept_terminal_receipt(run_id, [9; 32]).unwrap());

    let crashed_root = tempfile::tempdir().unwrap();
    fs::set_permissions(crashed_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(
        crashed_root.path().join("replay-journal.pending"),
        b"incomplete transition",
    )
    .unwrap();
    assert!(matches!(
        ReplayJournal::open(crashed_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));
}

#[test]
fn replay_journal_rejects_unsafe_root_and_file_custody() {
    let broad_root = tempfile::tempdir().unwrap();
    fs::set_permissions(broad_root.path(), fs::Permissions::from_mode(0o755)).unwrap();
    assert!(matches!(
        ReplayJournal::open(broad_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let mode_root = tempfile::tempdir().unwrap();
    fs::set_permissions(mode_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let run_id = Ulid::from(81_u128);
    let mut journal = ReplayJournal::open(mode_root.path().to_path_buf()).unwrap();
    journal
        .accept_assignment(
            run_id,
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        )
        .unwrap();
    drop(journal);
    let state = mode_root.path().join("replay-journal.postcard");
    fs::set_permissions(&state, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(matches!(
        ReplayJournal::open(mode_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let link_root = tempfile::tempdir().unwrap();
    fs::set_permissions(link_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let mut journal = ReplayJournal::open(link_root.path().to_path_buf()).unwrap();
    journal
        .accept_assignment(
            Ulid::from(82_u128),
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        )
        .unwrap();
    drop(journal);
    let state = link_root.path().join("replay-journal.postcard");
    fs::hard_link(&state, link_root.path().join("state-hardlink")).unwrap();
    assert!(matches!(
        ReplayJournal::open(link_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let symlink_root = tempfile::tempdir().unwrap();
    fs::set_permissions(symlink_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let mut journal = ReplayJournal::open(symlink_root.path().to_path_buf()).unwrap();
    journal
        .accept_assignment(
            Ulid::from(83_u128),
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        )
        .unwrap();
    drop(journal);
    let state = symlink_root.path().join("replay-journal.postcard");
    let backing = symlink_root.path().join("state-backing");
    fs::rename(&state, &backing).unwrap();
    symlink(&backing, &state).unwrap();
    assert!(matches!(
        ReplayJournal::open(symlink_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let pending_root = tempfile::tempdir().unwrap();
    fs::set_permissions(pending_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    symlink(
        pending_root.path().join("missing"),
        pending_root.path().join("replay-journal.pending"),
    )
    .unwrap();
    assert!(matches!(
        ReplayJournal::open(pending_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));
}

#[test]
fn replay_journal_rejects_root_replacement_after_open() {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("journal");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let mut journal = ReplayJournal::open(root.clone()).unwrap();
    fs::rename(&root, parent.path().join("original-journal")).unwrap();
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();

    assert!(matches!(
        journal.accept_assignment(
            Ulid::from(84_u128),
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        ),
        Err(FabricError::JournalIo(_))
    ));
}

#[test]
fn replay_journal_rejects_a_second_live_opener_and_preserves_all_state() {
    let root = tempfile::tempdir().unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let mut owner = ReplayJournal::open(root.path().to_path_buf()).unwrap();
    let lock = root.path().join("replay-journal.lock");
    let lock_inode = fs::metadata(&lock).unwrap().ino();

    assert!(matches!(
        ReplayJournal::open(root.path().to_path_buf()),
        Err(FabricError::JournalLocked)
    ));
    assert!(!root.path().join("replay-journal.postcard").exists());
    owner
        .accept_assignment(
            Ulid::from(101_u128),
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        )
        .unwrap();
    drop(owner);
    assert_eq!(fs::metadata(&lock).unwrap().ino(), lock_inode);

    let mut successor = ReplayJournal::open(root.path().to_path_buf()).unwrap();
    assert_eq!(fs::metadata(&lock).unwrap().ino(), lock_inode);
    assert_eq!(
        successor.accept_assignment(
            Ulid::from(101_u128),
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        ),
        Err(FabricError::ReplayedPosition)
    );
    successor
        .accept_assignment(
            Ulid::from(102_u128),
            AttemptFence {
                attempt: 1,
                fence: 5,
                hub_epoch: 2,
                control_position: 11,
            },
        )
        .unwrap();
    drop(successor);

    let mut recovered = ReplayJournal::open(root.path().to_path_buf()).unwrap();
    for (run_id, fence, position) in [(101_u128, 4, 10), (102_u128, 5, 11)] {
        assert_eq!(
            recovered.accept_assignment(
                Ulid::from(run_id),
                AttemptFence {
                    attempt: 1,
                    fence,
                    hub_epoch: 2,
                    control_position: position,
                },
            ),
            Err(FabricError::ReplayedPosition)
        );
    }
}

#[test]
fn replay_journal_rejects_unsafe_or_replaced_lock_file() {
    let broad_lock_root = tempfile::tempdir().unwrap();
    fs::set_permissions(broad_lock_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let broad_lock = broad_lock_root.path().join("replay-journal.lock");
    fs::write(&broad_lock, []).unwrap();
    fs::set_permissions(&broad_lock, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(matches!(
        ReplayJournal::open(broad_lock_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let linked_lock_root = tempfile::tempdir().unwrap();
    fs::set_permissions(linked_lock_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let linked_lock = linked_lock_root.path().join("replay-journal.lock");
    fs::write(&linked_lock, []).unwrap();
    fs::set_permissions(&linked_lock, fs::Permissions::from_mode(0o600)).unwrap();
    fs::hard_link(&linked_lock, linked_lock_root.path().join("lock-hardlink")).unwrap();
    assert!(matches!(
        ReplayJournal::open(linked_lock_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let symlink_lock_root = tempfile::tempdir().unwrap();
    fs::set_permissions(symlink_lock_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let lock_target = symlink_lock_root.path().join("lock-target");
    fs::write(&lock_target, []).unwrap();
    fs::set_permissions(&lock_target, fs::Permissions::from_mode(0o600)).unwrap();
    symlink(
        &lock_target,
        symlink_lock_root.path().join("replay-journal.lock"),
    )
    .unwrap();
    assert!(matches!(
        ReplayJournal::open(symlink_lock_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let nonempty_lock_root = tempfile::tempdir().unwrap();
    fs::set_permissions(nonempty_lock_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let nonempty_lock = nonempty_lock_root.path().join("replay-journal.lock");
    fs::write(&nonempty_lock, b"not-a-lock-marker").unwrap();
    fs::set_permissions(&nonempty_lock, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(matches!(
        ReplayJournal::open(nonempty_lock_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let replaced_lock_root = tempfile::tempdir().unwrap();
    fs::set_permissions(replaced_lock_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let lock = replaced_lock_root.path().join("replay-journal.lock");
    fs::write(&lock, []).unwrap();
    fs::set_permissions(&lock, fs::Permissions::from_mode(0o600)).unwrap();
    let mut journal = ReplayJournal::open(replaced_lock_root.path().to_path_buf()).unwrap();
    fs::rename(&lock, replaced_lock_root.path().join("replaced-lock")).unwrap();
    fs::write(&lock, []).unwrap();
    fs::set_permissions(&lock, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(matches!(
        journal.accept_assignment(
            Ulid::from(103_u128),
            AttemptFence {
                attempt: 1,
                fence: 6,
                hub_epoch: 2,
                control_position: 12,
            },
        ),
        Err(FabricError::JournalIo(_))
    ));
}
