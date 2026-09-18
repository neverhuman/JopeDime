use std::{
    fs,
    os::unix::fs::{symlink, PermissionsExt},
    path::Path,
};

use ed25519_dalek::{Signer, SigningKey};
use serde_json::{json, Value};

use crate::{
    ApplianceRole, NodeProfile, NodeRoleEnvelope, RoleError, RoleV2Bindings, WorkerProvision,
};

fn signing_key() -> SigningKey {
    SigningKey::from_bytes(&[37; 32])
}

fn node_signing_key() -> SigningKey {
    SigningKey::from_bytes(&[53; 32])
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn payload(role: &str, profile: &str, compute_slots: u16, storage_enabled: bool) -> Value {
    let max_inflight = if profile == "storage_only" {
        0
    } else {
        compute_slots
    };
    let provision = if profile == "storage_only" {
        Value::Null
    } else {
        json!("smartcluster_compute")
    };
    json!({
        "schema": "jain.appliance-role/v1",
        "release": "9.0.0-appliance.1",
        "status": "candidate",
        "formal_ga": false,
        "image_digest": format!("sha256:{}", "a".repeat(64)),
        "role": role,
        "node_id": "node-1",
        "session_id": "session-1",
        "profile": profile,
        "provision": provision,
        "compute_slots": compute_slots,
        "storage_enabled": storage_enabled,
        "services": {"jainnode": true, "scqd": false},
        "state_root": "/var/lib/jain/node",
        "qualification_receipt": "/etc/jain/isolation.json",
        "control": {
            "hub_address": "127.0.0.1:9443",
            "server_name": "hub.internal",
            "ca_certificate": "/etc/jain/ca.pem",
            "client_certificate": "/etc/jain/node.pem",
            "client_private_key": "/etc/jain/node-key.pem",
            "connect_timeout_ms": 10000,
            "frame_timeout_ms": 30000
        },
        "node_key_id": "node-key-1",
        "node_public_key": hex(&node_signing_key().verifying_key().to_bytes()),
        "node_private_key": "/etc/jain/node-signing-key.raw",
        "hub_issuer": "hub-issuer",
        "hub_public_key": hex(&signing_key().verifying_key().to_bytes()),
        "image_issuer": "image-issuer",
        "image_public_key": "2".repeat(64),
        "max_inflight": max_inflight
    })
}

fn signed(payload: Value, key: &SigningKey) -> Value {
    let parsed: NodeRoleEnvelope = serde_json::from_value(payload.clone()).unwrap();
    let signature = key.sign(&parsed.signing_bytes().unwrap()).to_bytes();
    json!({"payload": payload, "signature": hex(&signature)})
}

fn signed_v2(payload: Value, bindings: Value, key: &SigningKey) -> Value {
    let parsed_payload: NodeRoleEnvelope = serde_json::from_value(payload.clone()).unwrap();
    let parsed_bindings: RoleV2Bindings = serde_json::from_value(bindings.clone()).unwrap();
    let signature = key
        .sign(&parsed_payload.signing_bytes_v2(&parsed_bindings).unwrap())
        .to_bytes();
    json!({"payload": payload, "bindings": bindings, "signature": hex(&signature)})
}

fn write_storage_config(root: &Path) -> (std::path::PathBuf, String) {
    let dedup = root.join("dedup.key");
    let kek = root.join("kek.key");
    fs::write(&dedup, [1; 32]).unwrap();
    fs::write(&kek, [2; 32]).unwrap();
    fs::set_permissions(&dedup, fs::Permissions::from_mode(0o600)).unwrap();
    fs::set_permissions(&kek, fs::Permissions::from_mode(0o600)).unwrap();
    let path = root.join("storage.json");
    let document = json!({
        "schema": "jain.node.storage-config/v1",
        "deployment_id": "deployment-1",
        "node_id": "node-1",
        "staging_parent": root.join("staging"),
        "volumes": [
            {
                "root": root.join("volume-a"),
                "filesystem_uuid": "11111111-1111-1111-1111-111111111111",
                "failure_domain": "domain-a"
            },
            {
                "root": root.join("volume-b"),
                "filesystem_uuid": "22222222-2222-2222-2222-222222222222",
                "failure_domain": "domain-b"
            }
        ],
        "projects": [{
            "project_id": "project-1",
            "dedup_key_path": dedup,
            "active_kek_version": 1,
            "kek_files": [{"version": 1, "path": kek}]
        }]
    });
    let bytes = serde_json::to_vec(&document).unwrap();
    fs::write(&path, &bytes).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    (path, hex(&jain_smartcluster_fabric::sha256(&bytes)))
}

fn v2_bindings(storage_config_path: &Path, storage_config_sha256: &str) -> Value {
    json!({
        "storage_config_path": storage_config_path,
        "storage_config_sha256": storage_config_sha256,
        "docker_socket": "/run/docker.sock",
        "isolation_broker_socket": "/run/jain/isolation-broker.sock"
    })
}

fn write_envelope(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

fn open_role(path: &Path) -> Result<NodeRoleEnvelope, crate::RoleError> {
    NodeRoleEnvelope::open(path, signing_key().verifying_key().to_bytes())
}

#[test]
fn physical_candidate_envelope_is_signed_closed_and_role_exact() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("role.json");
    write_envelope(
        &path,
        &signed(payload("edge-hub", "storage_only", 0, true), &signing_key()),
    );
    let loaded = open_role(&path).unwrap();
    assert_eq!(loaded.role(), ApplianceRole::EdgeHub);
    assert_eq!(loaded.profile(), NodeProfile::StorageOnly);
    assert_eq!(loaded.provision(), None);

    let mut unknown = signed(payload("edge-hub", "storage_only", 0, true), &signing_key());
    unknown["payload"]["undeclared_service"] = json!(true);
    write_envelope(&path, &unknown);
    assert!(open_role(&path).is_err());
}

#[test]
fn legacy_v1_signature_bytes_remain_frozen_and_cannot_authorize_execution() {
    let parsed: NodeRoleEnvelope =
        serde_json::from_value(payload("compute-node", "cpu", 2, true)).unwrap();
    let signing_bytes = parsed.signing_bytes().unwrap();
    assert_eq!(
        hex(&jain_smartcluster_fabric::sha256(&signing_bytes)),
        "71fd16c7a357dea1b3e4c16cfcc27487e00f208f3a1ed65c87248adab5875452"
    );
    assert_eq!(
        hex(&signing_key().sign(&signing_bytes).to_bytes()),
        "3fd28fe01d7282d7c55b86afbbdc4c54d432710f31f5c3eca18a55af41025774388eeeb07580d77cb86c1baded750442d9b25681f0f6066f79313cffaa5ea808"
    );
    assert!(matches!(
        parsed.execution_bindings(),
        Err(RoleError::RoleV1CannotExecute)
    ));
}

#[test]
fn v2_signature_has_a_frozen_domain_separated_vector() {
    let schema: Value = serde_json::from_str(include_str!(
        "../../schemas/jain.appliance-role-v2.schema.json"
    ))
    .unwrap();
    assert_eq!(schema["$id"], "jain.appliance-role/v2");
    assert_eq!(schema["additionalProperties"], false);

    let mut role_payload = payload("compute-node", "cpu", 2, true);
    role_payload["schema"] = json!("jain.appliance-role/v2");
    let parsed_payload: NodeRoleEnvelope = serde_json::from_value(role_payload).unwrap();
    let bindings: RoleV2Bindings = serde_json::from_value(v2_bindings(
        Path::new("/etc/jain/storage.json"),
        &"4".repeat(64),
    ))
    .unwrap();
    let signing_bytes = parsed_payload.signing_bytes_v2(&bindings).unwrap();
    assert_eq!(
        hex(&jain_smartcluster_fabric::sha256(&signing_bytes)),
        "acfa31817ce19f1a50d94892212ab39346772ef10aeb1a97a4230570dac6a34f"
    );
    assert_eq!(
        hex(&signing_key().sign(&signing_bytes).to_bytes()),
        "363ab55021e6ce242fdd8a5db5496fbb01910767a4a03f63f14780acf1b4e0d47a90a50e5383fdfaa355309d08b720b86fc4ea454b67ca25ea9ca0aec47f630f"
    );
}

#[test]
fn v2_signature_binds_storage_config_digest_and_brokered_sockets() {
    let root = tempfile::tempdir().unwrap();
    let role_path = root.path().join("role.json");
    let (storage_path, storage_sha256) = write_storage_config(root.path());
    let mut role_payload = payload("compute-node", "cpu", 2, true);
    role_payload["schema"] = json!("jain.appliance-role/v2");
    let bindings = v2_bindings(&storage_path, &storage_sha256);
    write_envelope(
        &role_path,
        &signed_v2(role_payload.clone(), bindings.clone(), &signing_key()),
    );

    let role = open_role(&role_path).unwrap();
    let verified_bindings = role.execution_bindings().unwrap();
    assert_eq!(verified_bindings.storage_config_path(), storage_path);
    assert_eq!(
        verified_bindings.open_storage_config().unwrap().node_id(),
        "node-1"
    );

    for field in [
        "storage_config_sha256",
        "docker_socket",
        "isolation_broker_socket",
    ] {
        let mut tampered = signed_v2(role_payload.clone(), bindings.clone(), &signing_key());
        tampered["bindings"][field] = if field == "storage_config_sha256" {
            json!("9".repeat(64))
        } else {
            json!(format!("/run/jain/tampered-{field}.sock"))
        };
        write_envelope(&role_path, &tampered);
        assert!(open_role(&role_path).is_err(), "accepted tampered {field}");
    }

    write_envelope(
        &role_path,
        &json!({
            "payload": role_payload,
            "signature": "0".repeat(128)
        }),
    );
    assert!(open_role(&role_path).is_err());
}

#[test]
fn role_v2_rechecks_the_exact_storage_config_bytes() {
    let root = tempfile::tempdir().unwrap();
    let role_path = root.path().join("role.json");
    let (storage_path, storage_sha256) = write_storage_config(root.path());
    let mut role_payload = payload("compute-node", "cpu", 1, false);
    role_payload["schema"] = json!("jain.appliance-role/v2");
    let bindings = v2_bindings(&storage_path, &storage_sha256);
    write_envelope(
        &role_path,
        &signed_v2(role_payload, bindings, &signing_key()),
    );
    let role = open_role(&role_path).unwrap();
    assert!(role
        .execution_bindings()
        .unwrap()
        .open_storage_config()
        .is_ok());

    let mut bytes = fs::read(&storage_path).unwrap();
    bytes.push(b' ');
    fs::write(&storage_path, bytes).unwrap();
    assert!(role
        .execution_bindings()
        .unwrap()
        .open_storage_config()
        .is_err());
}

#[test]
fn one_jainnode_accepts_exactly_the_three_worker_provisions() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("role.json");
    for (wire, expected) in [
        ("code_runner", WorkerProvision::CodeRunner),
        ("smartcluster_compute", WorkerProvision::SmartclusterCompute),
        ("dedicated_inference", WorkerProvision::DedicatedInference),
    ] {
        let mut candidate = payload("compute-node", "cpu", 1, false);
        candidate["provision"] = json!(wire);
        write_envelope(&path, &signed(candidate, &signing_key()));
        assert_eq!(open_role(&path).unwrap().provision(), Some(expected));
    }

    let mut unprovisioned = payload("compute-node", "cpu", 1, false);
    unprovisioned["provision"] = Value::Null;
    write_envelope(&path, &signed(unprovisioned, &signing_key()));
    assert!(open_role(&path).is_err());

    let mut storage_worker = payload("edge-hub", "storage_only", 0, true);
    storage_worker["provision"] = json!("code_runner");
    write_envelope(&path, &signed(storage_worker, &signing_key()));
    assert!(open_role(&path).is_err());
}

#[test]
fn envelope_requires_owner_only_single_link_physical_file() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("role.json");
    write_envelope(
        &path,
        &signed(payload("edge-hub", "storage_only", 0, true), &signing_key()),
    );

    fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
    assert!(open_role(&path).is_err());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();

    let hardlink = root.path().join("role-hardlink.json");
    fs::hard_link(&path, &hardlink).unwrap();
    assert!(open_role(&path).is_err());
    fs::remove_file(&hardlink).unwrap();

    let symlink_path = root.path().join("role-symlink.json");
    symlink(&path, &symlink_path).unwrap();
    assert!(open_role(&symlink_path).is_err());
}

#[test]
fn signature_binds_domain_key_and_every_payload_field() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("role.json");
    let base_payload = payload("compute-node", "cpu", 2, true);

    let wrong_key = SigningKey::from_bytes(&[38; 32]);
    write_envelope(&path, &signed(base_payload.clone(), &wrong_key));
    assert!(open_role(&path).is_err());

    let mut changed = signed(base_payload.clone(), &signing_key());
    changed["payload"]["compute_slots"] = json!(1);
    write_envelope(&path, &changed);
    assert!(open_role(&path).is_err());

    let parsed: NodeRoleEnvelope = serde_json::from_value(base_payload.clone()).unwrap();
    let canonical = parsed.signing_bytes().unwrap();
    let payload_offset = b"jain.node.appliance-role.v1\0".len();
    let mut wrong_domain = b"jain.node.wrong-role.v1\0".to_vec();
    wrong_domain.extend_from_slice(&canonical[payload_offset..]);
    let signature = signing_key().sign(&wrong_domain).to_bytes();
    write_envelope(
        &path,
        &json!({"payload": base_payload, "signature": hex(&signature)}),
    );
    assert!(open_role(&path).is_err());

    let mut malformed = signed(payload("compute-node", "cpu", 1, true), &signing_key());
    malformed["signature"] = json!("0".repeat(127));
    write_envelope(&path, &malformed);
    assert!(open_role(&path).is_err());
}

#[test]
fn self_issued_role_envelope_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("role.json");
    let attacker = SigningKey::from_bytes(&[38; 32]);
    let mut attacker_payload = payload("compute-node", "cpu", 1, false);
    attacker_payload["hub_public_key"] = json!(hex(&attacker.verifying_key().to_bytes()));
    write_envelope(&path, &signed(attacker_payload, &attacker));

    assert!(open_role(&path).is_err());
}

#[test]
fn release_digest_service_and_capacity_mismatches_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("role.json");
    for mut invalid in [
        payload("edge-hub", "gpu", 1, true),
        payload("compute-node", "storage_only", 0, true),
        payload("all-in-one", "cpu", 1, false),
        payload("witness-storage-node", "storage_only", 1, true),
    ] {
        write_envelope(&path, &signed(invalid.clone(), &signing_key()));
        assert!(open_role(&path).is_err(), "{invalid}");
        invalid["services"]["scqd"] = json!(true);
        write_envelope(&path, &signed(invalid, &signing_key()));
        assert!(open_role(&path).is_err());
    }

    for (field, value) in [
        ("formal_ga", json!(true)),
        ("status", json!("ga")),
        ("release", json!("9.0.0-distributed.1")),
        ("image_digest", json!(format!("sha256:{}", "0".repeat(64)))),
        ("image_digest", json!(format!("sha256:{}", "A".repeat(64)))),
        ("hub_public_key", json!("A".repeat(64))),
    ] {
        let mut invalid = signed(payload("compute-node", "cpu", 1, false), &signing_key());
        invalid["payload"][field] = value;
        write_envelope(&path, &invalid);
        assert!(open_role(&path).is_err(), "field={field}");
    }

    for mut invalid in [
        payload("edge-hub", "storage_only", 0, true),
        payload("compute-node", "cpu", 1, true),
        payload("witness-storage-node", "storage_only", 0, true),
    ] {
        invalid["max_inflight"] = if invalid["profile"] == "storage_only" {
            json!(1)
        } else {
            json!(2)
        };
        write_envelope(&path, &signed(invalid, &signing_key()));
        assert!(open_role(&path).is_err());
    }
}

#[test]
fn runtime_identity_requires_the_exact_physical_node_key_and_signs_hello() {
    let root = tempfile::tempdir().unwrap();
    let role_path = root.path().join("role.json");
    let key_path = root.path().join("node-signing-key.raw");
    fs::write(&key_path, node_signing_key().to_bytes()).unwrap();
    fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600)).unwrap();

    let mut role_payload = payload("compute-node", "cpu", 1, false);
    role_payload["node_private_key"] = json!(key_path);
    write_envelope(&role_path, &signed(role_payload.clone(), &signing_key()));
    let role = open_role(&role_path).unwrap();
    let identity = role.runtime_identity().unwrap();
    assert_eq!(
        identity.public_key(),
        node_signing_key().verifying_key().to_bytes()
    );
    let hello = identity
        .session_hello(
            NodeProfile::Cpu,
            role.provision().unwrap(),
            [7; 32],
            3,
            10_000,
        )
        .unwrap();
    hello
        .verify_session_hello(
            "node-1",
            "session-1",
            NodeProfile::Cpu,
            WorkerProvision::SmartclusterCompute,
            [7; 32],
            "node-key-1",
            identity.public_key(),
            10_010,
            100,
        )
        .unwrap();

    let mut tampered = hello.clone();
    let crate::NodeControlFrame::SessionHello { sequence, .. } = &mut tampered else {
        unreachable!();
    };
    *sequence += 1;
    assert!(tampered
        .verify_session_hello(
            "node-1",
            "session-1",
            NodeProfile::Cpu,
            WorkerProvision::CodeRunner,
            [7; 32],
            "node-key-1",
            identity.public_key(),
            10_010,
            100,
        )
        .is_err());

    let mut wrong_public_key = role_payload.clone();
    wrong_public_key["node_public_key"] = json!("3".repeat(64));
    write_envelope(&role_path, &signed(wrong_public_key, &signing_key()));
    assert!(open_role(&role_path).unwrap().runtime_identity().is_err());

    write_envelope(&role_path, &signed(role_payload.clone(), &signing_key()));
    fs::set_permissions(&key_path, fs::Permissions::from_mode(0o640)).unwrap();
    assert!(open_role(&role_path).unwrap().runtime_identity().is_err());
    fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600)).unwrap();

    let hardlink = root.path().join("node-signing-key-hardlink.raw");
    fs::hard_link(&key_path, &hardlink).unwrap();
    assert!(open_role(&role_path).unwrap().runtime_identity().is_err());
    fs::remove_file(&hardlink).unwrap();

    let symlink_path = root.path().join("node-signing-key-symlink.raw");
    symlink(&key_path, &symlink_path).unwrap();
    let mut symlinked_key = role_payload.clone();
    symlinked_key["node_private_key"] = json!(symlink_path);
    write_envelope(&role_path, &signed(symlinked_key, &signing_key()));
    assert!(open_role(&role_path).unwrap().runtime_identity().is_err());
    fs::remove_file(root.path().join("node-signing-key-symlink.raw")).unwrap();

    write_envelope(&role_path, &signed(role_payload, &signing_key()));
    for invalid_length in [31, 33] {
        fs::write(&key_path, vec![53; invalid_length]).unwrap();
        assert!(open_role(&role_path).unwrap().runtime_identity().is_err());
    }
}
