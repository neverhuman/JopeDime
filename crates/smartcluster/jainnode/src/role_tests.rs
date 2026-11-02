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

include!("role_tests/signatures.rs");

include!("role_tests/provisioning_and_rejection.rs");
