use ed25519_dalek::SigningKey;
use jain_smartcluster_fabric::{sha256, SignatureAuthority};

use crate::{
    BeginFragmentWrite, ComputeStateTelemetry, FragmentHealthTelemetry, FragmentWriteChunk,
    NodeControlFrame, NodeRuntimeIdentity, SignedStorageControlRequest,
    SignedStorageFragmentReceipt, StorageControlBody, StorageFragmentKind, StorageIoCounters,
    StorageTelemetryEmitter, StorageTelemetryObservation, StorageVolumeTelemetry,
    MAX_STORAGE_FRAGMENT_CHUNK_BYTES,
};

fn identity() -> NodeRuntimeIdentity {
    NodeRuntimeIdentity::new(
        "micro-01".to_owned(),
        "session-01".to_owned(),
        "node-key-01".to_owned(),
        "hub-issuer".to_owned(),
        [3; 32],
        "image-issuer".to_owned(),
        [4; 32],
        SigningKey::from_bytes(&[91; 32]),
    )
    .unwrap()
}

fn observation(at_ms: u64) -> StorageTelemetryObservation {
    StorageTelemetryObservation {
        lease_id: "lease-01".to_owned(),
        lease_expires_at_ms: at_ms + 10_000,
        observed_at_ms: at_ms,
        topology_digest: [5; 32],
        storage_capable: true,
        compute_capable: true,
        volumes: vec![StorageVolumeTelemetry {
            failure_domain: "rack-a".to_owned(),
            capacity_bytes: 10_000,
            used_bytes: 1_000,
        }],
        io: StorageIoCounters {
            read_bytes_total: 100,
            write_bytes_total: 200,
            read_operations_total: 3,
            write_operations_total: 4,
        },
        fragment_health: FragmentHealthTelemetry {
            healthy_fragments: 5,
            missing_fragments: 0,
            corrupt_fragments: 0,
            underprotected_objects: 0,
        },
        compute: ComputeStateTelemetry {
            slots_total: 1,
            slots_busy: 0,
            completed_execution_receipt_digest: Some([6; 32]),
        },
    }
}

fn receipt(identity: &NodeRuntimeIdentity) -> SignedStorageFragmentReceipt {
    SignedStorageFragmentReceipt::sign_fsynced(
        identity,
        "system-seed".to_owned(),
        [7; 32],
        [8; 32],
        0,
        StorageFragmentKind::Data(0),
        "rack-a".to_owned(),
        [5; 32],
        [9; 32],
        262_160,
        12_000,
    )
    .unwrap()
}

#[test]
fn telemetry_is_signed_periodic_transition_driven_and_monotone() {
    let identity = identity();
    let mut emitter = StorageTelemetryEmitter::default();
    let first = emitter
        .observe(&identity, observation(10_000))
        .unwrap()
        .unwrap();
    assert_eq!(first.facts.revision, 1);
    first
        .verify(
            identity.node_id(),
            identity.session_id(),
            "node-key-01",
            identity.public_key(),
        )
        .unwrap();

    let mut before_interval = observation(11_999);
    before_interval.io.read_bytes_total += 1;
    assert!(emitter
        .observe(&identity, before_interval)
        .unwrap()
        .is_none());

    let mut periodic = observation(12_000);
    periodic.io.read_bytes_total += 2;
    let second = emitter.observe(&identity, periodic).unwrap().unwrap();
    assert_eq!(second.facts.revision, 2);

    let mut transition = observation(12_001);
    transition.io.read_bytes_total += 3;
    transition.compute.slots_busy = 1;
    let third = emitter.observe(&identity, transition).unwrap().unwrap();
    assert_eq!(third.facts.revision, 3);

    let mut regression = observation(14_001);
    regression.io.read_bytes_total = 1;
    assert!(emitter.observe(&identity, regression).is_err());

    let mut tampered = third;
    tampered.facts.observation.fragment_health.missing_fragments = 1;
    assert!(tampered
        .verify(
            identity.node_id(),
            identity.session_id(),
            "node-key-01",
            identity.public_key(),
        )
        .is_err());
}

#[test]
fn fsync_receipt_binds_fragment_node_domain_digest_and_bytes() {
    let identity = identity();
    let signed = receipt(&identity);
    assert_eq!(
        signed.signature.authority,
        SignatureAuthority::StorageFragmentReceipt
    );
    signed
        .verify(
            identity.node_id(),
            identity.session_id(),
            "node-key-01",
            identity.public_key(),
        )
        .unwrap();

    let mut tampered = signed.clone();
    tampered.facts.failure_domain = "rack-b".to_owned();
    assert!(tampered
        .verify(
            identity.node_id(),
            identity.session_id(),
            "node-key-01",
            identity.public_key(),
        )
        .is_err());

    let mut unflushed = signed;
    unflushed.facts.fsync_completed_at_ms = 0;
    assert!(unflushed
        .verify(
            identity.node_id(),
            identity.session_id(),
            "node-key-01",
            identity.public_key(),
        )
        .is_err());
}

#[test]
fn storage_control_is_closed_signed_bounded_and_session_scoped() {
    let hub_key = SigningKey::from_bytes(&[92; 32]);
    let bytes = vec![42; MAX_STORAGE_FRAGMENT_CHUNK_BYTES];
    let request = SignedStorageControlRequest::sign(
        "hub-issuer".to_owned(),
        "hub-key-01".to_owned(),
        &hub_key,
        "session-01".to_owned(),
        11,
        20_000,
        StorageControlBody::WriteChunk(FragmentWriteChunk {
            transfer_id: "transfer-01".to_owned(),
            offset: 0,
            sha256: sha256(&bytes),
            bytes,
        }),
    )
    .unwrap();
    request
        .verify(
            "session-01",
            "hub-issuer",
            hub_key.verifying_key().to_bytes(),
            19_999,
        )
        .unwrap();
    assert!(request
        .verify(
            "session-02",
            "hub-issuer",
            hub_key.verifying_key().to_bytes(),
            19_999,
        )
        .is_err());
    assert!(NodeControlFrame::storage_control(request).is_ok());

    let oversized = vec![7; MAX_STORAGE_FRAGMENT_CHUNK_BYTES + 1];
    assert!(SignedStorageControlRequest::sign(
        "hub-issuer".to_owned(),
        "hub-key-01".to_owned(),
        &hub_key,
        "session-01".to_owned(),
        12,
        20_000,
        StorageControlBody::WriteChunk(FragmentWriteChunk {
            transfer_id: "transfer-02".to_owned(),
            offset: 0,
            sha256: sha256(&oversized),
            bytes: oversized,
        }),
    )
    .is_err());
}

#[test]
fn begin_write_exposes_only_logical_identifiers_and_expected_facts() {
    let begin = BeginFragmentWrite {
        transfer_id: "transfer-01".to_owned(),
        project_id: "system-seed".to_owned(),
        object_id: [7; 32],
        manifest_digest: [8; 32],
        fragment_index: 4,
        fragment_kind: StorageFragmentKind::Recovery(0),
        failure_domain: "rack-e".to_owned(),
        topology_digest: [5; 32],
        expected_bytes: 262_160,
        expected_sha256: [9; 32],
    };
    let debug = format!("{begin:?}");
    for forbidden in [
        "filesystem_uuid",
        "host_path",
        "container_id",
        "certificate",
        "private_key",
        "plaintext",
    ] {
        assert!(!debug.contains(forbidden));
    }
}

#[test]
fn storage_frames_are_durable_outbound_records() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outbox");
    let identity = identity();
    let mut emitter = StorageTelemetryEmitter::default();
    let telemetry = emitter
        .observe(&identity, observation(10_000))
        .unwrap()
        .unwrap();
    let outbox = crate::DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
    let first = outbox.enqueue_storage_telemetry(telemetry).unwrap();
    let second = outbox
        .enqueue_storage_fragment_receipt(receipt(&identity))
        .unwrap();
    assert_eq!((first.sequence, second.sequence), (1, 2));
    drop(outbox);

    let reopened = crate::DurableControlOutbox::open(path, identity).unwrap();
    assert_eq!(reopened.pending_count().unwrap(), 2);
    assert!(matches!(
        reopened.oldest().unwrap().unwrap().frame,
        NodeControlFrame::StorageTelemetry { .. }
    ));
}

#[test]
fn storage_contract_schemas_are_closed_and_json_bytes_are_decimal_strings() {
    let telemetry_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../schemas/jain.node.storage-telemetry-v1.schema.json"
    ))
    .unwrap();
    let receipt_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../schemas/jain.storage-fragment-receipt-v1.schema.json"
    ))
    .unwrap();
    let control_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../schemas/jain.node.storage-control-v1.schema.json"
    ))
    .unwrap();
    for (schema, id) in [
        (telemetry_schema, "jain.node.storage-telemetry/v1"),
        (receipt_schema, "jain.storage-fragment-receipt/v1"),
        (control_schema, "jain.node.storage-control/v1"),
    ] {
        assert_eq!(schema["$id"], id);
        assert_eq!(schema["additionalProperties"], false);
    }

    let identity = identity();
    let mut emitter = StorageTelemetryEmitter::default();
    let telemetry = emitter
        .observe(&identity, observation(10_000))
        .unwrap()
        .unwrap();
    let json = serde_json::to_value(telemetry).unwrap();
    assert_eq!(
        json["facts"]["observation"]["volumes"][0]["capacity_bytes"],
        "10000"
    );
    assert_eq!(
        json["facts"]["observation"]["io"]["read_bytes_total"],
        "100"
    );
    assert_eq!(
        serde_json::to_value(receipt(&identity)).unwrap()["facts"]["fragment_bytes"],
        "262160"
    );
}
