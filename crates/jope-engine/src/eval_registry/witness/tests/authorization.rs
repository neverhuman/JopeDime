use super::*;

#[test]
fn public_witness_configuration_binds_bytes_and_rejects_aliases_and_wrong_keys() {
    let f = Fixture::new();
    let path = f.authority.parent().unwrap().join("witness-client.json");
    let data = serde_json::to_vec(&f.config()).unwrap();
    publish(&path, &data).unwrap();
    let loaded = WitnessClientConfig::from_file(&path, &sha(&data)).unwrap();
    assert_eq!(loaded.policy.sha256().unwrap(), f.policy.sha256().unwrap());
    assert!(WitnessClientConfig::from_file(&path, &sha(b"other config")).is_err());
    assert!(WitnessClientConfig::from_file(Path::new("relative.json"), &sha(&data)).is_err());
    let alias = path.with_extension("alias");
    std::os::unix::fs::symlink(&path, &alias).unwrap();
    assert!(WitnessClientConfig::from_file(&alias, &sha(&data)).is_err());
    let mut config = f.config();
    config.authority_key = f.ack_key.clone();
    let wrong = serde_json::to_vec(&config).unwrap();
    let other = path.with_extension("wrong-key");
    publish(&other, &wrong).unwrap();
    assert!(WitnessClientConfig::from_file(&other, &sha(&wrong)).is_err());
    assert_eq!(
        f.store().sequence,
        0,
        "configuration loading mutated witness history"
    );
}

#[test]
fn read_only_ack_recovery_uses_new_deadline_without_repairing_or_recording() {
    let f = Fixture::new();
    let mut event: Value = serde_json::from_slice(&f.event).unwrap();
    let mut start: crate::eval_registry::StartToken =
        serde_json::from_str(event["effects"][0]["row"]["start_json"].as_str().unwrap()).unwrap();
    start.authorized_monotonic_ns = 1;
    start.deadline_monotonic_ns = 2;
    event["effects"][0]["row"]["start_json"] =
        serde_json::to_value(serde_json::to_string(&start).unwrap()).unwrap();
    let expired = serde_json::to_vec(&event).unwrap();
    let request = f.client().request(&expired).unwrap();
    let original_head = read(&f.root.join("head.json")).unwrap();
    let ack = f.store().record_authorization(&request).unwrap();
    // Simulate the durable segment surviving an interrupted head publication.
    fs::write(f.root.join("head.json"), &original_head).unwrap();
    {
        let mut store = WitnessStore::open_read_only(&f.root, &f.policy, &f.ack_key).unwrap();
        assert_eq!(store.get_ack(&request).unwrap(), Some(ack.clone()));
        assert!(store.record_authorization(&request).is_err());
    }
    assert_eq!(read(&f.root.join("head.json")).unwrap(), original_head);
    let socket = f.authority.parent().unwrap().join("ack-recovery.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let root = f.root.clone();
    let policy = f.policy.clone();
    let key = f.ack_key.clone();
    let server = std::thread::spawn(move || {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let request: SignedRequest = read_frame(&mut stream).unwrap();
            let store = WitnessStore::open_read_only(&root, &policy, &key).unwrap();
            write_frame(&mut stream, &store.get_ack(&request).unwrap()).unwrap();
        }
    });
    let mut config = f.config();
    config.local_socket = Some(socket);
    let client = WitnessClient::new(config).unwrap();
    let deadline = crate::fit_boundary::deadline_after(Duration::from_secs(30)).unwrap();
    assert_eq!(client.get_ack(&expired, deadline).unwrap(), Some(ack));
    event["seq"] = json!(event["seq"].as_u64().unwrap() + 1);
    assert_eq!(
        client
            .get_ack(&serde_json::to_vec(&event).unwrap(), deadline)
            .unwrap(),
        None
    );
    server.join().unwrap();
    assert_eq!(read(&f.root.join("head.json")).unwrap(), original_head);
    assert_eq!(fs::read_dir(f.root.join("segments")).unwrap().count(), 1);
}

#[test]
fn independent_signature_and_full_original_event_survive_lost_ack_and_restart() {
    let f = Fixture::new();
    let request = f.request();
    let ack = {
        let mut store = f.store();
        let ack = store.record_authorization(&request).unwrap();
        assert_eq!(store.record_authorization(&request).unwrap(), ack);
        assert_eq!(store.get_ack(&request).unwrap(), Some(ack.clone()));
        ack
    };
    let store = f.store();
    assert_eq!(store.get_ack(&request).unwrap(), Some(ack.clone()));
    ack.verify(&f.policy, &request).unwrap();
    let record: Value =
        serde_json::from_slice(&read(&f.root.join("segments/00000000000000000001.json")).unwrap())
            .unwrap();
    assert_eq!(
        record["core"]["request"]["body"]["event"],
        serde_json::to_value(&f.event).unwrap()
    );
    assert_eq!(store.sequence, 1);
    let mut forged = ack.clone();
    forged.signature = sign(
        &private_key(&f.authority_key).unwrap(),
        "parallel.fit_event_witness.ack.v1",
        &forged.body,
    )
    .unwrap();
    assert!(forged.verify(&f.policy, &request).is_err());
}
#[test]
fn request_and_ack_tampering_wrong_keys_epochs_and_policy_are_denied() {
    let f = Fixture::new();
    let request = f.request();
    let mut store = f.store();
    let ack = store.record_authorization(&request).unwrap();
    let mut changed = request.clone();
    changed.body.event.push(b' ');
    assert!(store.record_authorization(&changed).is_err());
    let mut changed = request.clone();
    changed.signature = sign(
        &private_key(&f.ack_key).unwrap(),
        "parallel.fit_event_witness.request.v1",
        &changed.body,
    )
    .unwrap();
    assert!(store.record_authorization(&changed).is_err());
    for field in [
        "authority_instance",
        "authority_epoch",
        "witness_instance",
        "authority_source_sha256",
        "ssh_known_hosts_sha256",
    ] {
        let mut policy = serde_json::to_value(&f.policy).unwrap();
        policy[field] = json!(sha(b"wrong"));
        let policy: WitnessPolicy = serde_json::from_value(policy).unwrap();
        assert!(ack.verify(&policy, &request).is_err());
    }
    let mut changed = ack.clone();
    changed.body.authority_sequence += 1;
    assert!(changed.verify(&f.policy, &request).is_err());
    let mut policy = f.policy.clone();
    policy.witness_public_key = policy.authority_public_key.clone();
    assert!(policy.validate().is_err());
}
#[test]
fn conflicting_duplicate_and_out_of_order_authorizations_do_not_replace_original() {
    let f = Fixture::new();
    let request = f.request();
    let mut store = f.store();
    let ack = store.record_authorization(&request).unwrap();
    let mut changed: Value = serde_json::from_slice(&f.event).unwrap();
    changed["previous"] = json!(sha(b"different chain"));
    let changed = f.client().request(&bytes(&changed).unwrap()).unwrap();
    assert!(store.record_authorization(&changed).is_err());
    assert_eq!(store.get_ack(&request).unwrap(), Some(ack));
    let mut older: Value = serde_json::from_slice(&f.event).unwrap();
    older["seq"] = json!(1);
    let older = f.client().request(&bytes(&older).unwrap()).unwrap();
    assert!(store.record_authorization(&older).is_err());
}
#[test]
fn missing_committed_tail_corruption_and_unresolved_partial_segments_block_open() {
    for kind in ["missing", "corrupt", "partial", "head"] {
        let f = Fixture::new();
        f.store().record_authorization(&f.request()).unwrap();
        let path = f.root.join("segments/00000000000000000001.json");
        match kind {
            "missing" => fs::remove_file(path).unwrap(),
            "corrupt" => fs::write(path, b"{}").unwrap(),
            "partial" => fs::write(
                f.root.join("segments/00000000000000000002.json"),
                b"partial",
            )
            .unwrap(),
            _ => fs::write(
                f.root.join("head.json"),
                bytes(&serde_json::json!({
                    "sequence": 1,
                    "sha256": sha(b"wrong"),
                }))
                .unwrap(),
            )
            .unwrap(),
        };
        assert!(
            WitnessStore::open(&f.root, &f.policy, &f.ack_key).is_err(),
            "{kind}"
        );
    }
}
#[test]
fn stale_derived_head_adopts_original_signed_segment_without_new_event() {
    let f = Fixture::new();
    let request = f.request();
    let ack = f.store().record_authorization(&request).unwrap();
    fs::write(
        f.root.join("head.json"),
        bytes(&serde_json::json!({
            "sequence": 0,
            "sha256": ZERO,
        }))
        .unwrap(),
    )
    .unwrap();
    let store = f.store();
    assert_eq!(store.get_ack(&request).unwrap(), Some(ack));
    assert_eq!(store.sequence, 1);
}
#[test]
fn original_segment_tamper_and_lease_replacement_fail_during_live_custody() {
    let f = Fixture::new();
    let mut store = f.store();
    let request = f.request();
    store.record_authorization(&request).unwrap();
    fs::write(f.root.join("segments/00000000000000000001.json"), b"{}").unwrap();
    assert!(store.get_ack(&request).is_err());
    assert!(store.record_authorization(&request).is_err());
    let f = Fixture::new();
    let mut store = f.store();
    fs::rename(f.root.join("lease"), f.root.join("old-lease")).unwrap();
    publish(&f.root.join("lease"), b"").unwrap();
    assert!(store.record_authorization(&f.request()).is_err());
}
#[test]
fn unsafe_private_key_policy_source_and_transport_custody_are_denied() {
    let f = Fixture::new();
    let mut config = f.config();
    config.policy.authority_source_sha256 = sha(b"wrong");
    assert!(WitnessClient::new(config).is_err());
    let mut config = f.config();
    config.remote_executable = "/tmp/exec;touch".into();
    assert!(WitnessClient::new(config).is_err());
    let mut config = f.config();
    config.policy.ssh_known_hosts_sha256 = sha(b"unknown host");
    assert!(WitnessClient::new(config).is_err());
    fs::set_permissions(&f.authority_key, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(WitnessClient::new(f.config()).is_err());
    assert!(WitnessStore::open(&f.root, &f.policy, &f.authority_key).is_err());
}
