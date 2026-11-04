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
