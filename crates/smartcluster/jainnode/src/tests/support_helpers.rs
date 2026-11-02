fn isolation() -> IsolationEvidence {
    IsolationEvidence {
        digest: [7; 32],
        namespaces: true,
        device_filter: true,
        seccomp: true,
        apparmor_or_landlock: true,
        descendant_fencing: true,
        network_isolation: true,
        metadata_and_control_plane_denied: true,
        io_limits: true,
        seccomp_profile: "/etc/jain/seccomp.json".to_owned(),
        apparmor_profile: "jain-governed-rd".to_owned(),
    }
}

fn work() -> VerifiedWorkUnit {
    let key = hub_key();
    let image_digest = [2; 32];
    let image_hex = image_digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let projection = WorkUnitProjection {
        run_id: Ulid::from(1_u128),
        project_id: "project-1".to_owned(),
        workload_class: crate::WorkloadClass::Training,
        image_reference: format!("registry.invalid/research@sha256:{image_hex}"),
        image_digest,
        argv: vec!["/app/run".to_owned(), "--real".to_owned()],
        smoke_argv: vec!["/app/run".to_owned(), "--smoke".to_owned()],
        smoke_timeout_ms: 1_000,
        smoke_only: false,
        working_directory: "work".to_owned(),
        caller_environment: BTreeMap::from([("TRIAL".to_owned(), "7".to_owned())]),
        inputs: vec![ArtifactMount {
            artifact_id: "input".to_owned(),
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
    };
    let issuer = "hub-issuer".to_owned();
    VerifiedWorkUnit::verify(
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
        },
        "hub-issuer",
        key.verifying_key().to_bytes(),
        "hub-issuer",
        key.verifying_key().to_bytes(),
    )
    .unwrap()
}

fn placement() -> Placement {
    placement_with_gpus(vec!["GPU-exact".to_owned()])
}

fn placement_with_gpus(gpu_uuids: Vec<String>) -> Placement {
    let work = work();
    let key = hub_key();
    let mut placement = Placement {
        node_id: "node-1".to_owned(),
        session_id: "session-1".to_owned(),
        lease_id: "lease-1".to_owned(),
        lease_expires_at_ms: u64::MAX,
        work_unit_digest: work.work_unit_digest(),
        command_digest: work.command_digest(),
        attempt: work.attempt,
        attempt_fence: work.attempt_fence,
        hub_epoch: work.hub_epoch,
        control_position: work.control_position,
        capability_advertisement_digest: [8; 32],
        gpu_uuids,
        cached_input_bytes: 100,
        isolation_evidence_digest: [7; 32],
        signature: DomainSignature {
            authority: SignatureAuthority::PlacementOffer,
            issuer: "hub-issuer".to_owned(),
            key_id: "hub-key".to_owned(),
            public_key: key.verifying_key().to_bytes(),
            signature: Vec::new(),
        },
    };
    placement.signature = DomainSignature::sign(
        SignatureAuthority::PlacementOffer,
        "hub-issuer".to_owned(),
        "hub-key".to_owned(),
        &key,
        &placement.signed_bytes().unwrap(),
    );
    placement
}

fn identity() -> NodeRuntimeIdentity {
    NodeRuntimeIdentity::new(
        "node-1".to_owned(),
        "session-1".to_owned(),
        "node-key-1".to_owned(),
        "hub-issuer".to_owned(),
        hub_key().verifying_key().to_bytes(),
        "hub-issuer".to_owned(),
        hub_key().verifying_key().to_bytes(),
        SigningKey::from_bytes(&[12; 32]),
    )
    .unwrap()
}

fn hub_key() -> SigningKey {
    SigningKey::from_bytes(&[11; 32])
}
