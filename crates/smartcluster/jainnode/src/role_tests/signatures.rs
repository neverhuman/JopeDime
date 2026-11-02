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
        "../../../schemas/jain.appliance-role-v2.schema.json"
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
