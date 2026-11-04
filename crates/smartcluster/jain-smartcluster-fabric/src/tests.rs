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

mod provision;
mod replay;
