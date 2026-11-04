use super::*;

#[test]
fn witness_global_numerical_guard_survives_reopen_and_changed_authority_sequence() {
    let f = Fixture::new();
    let original = f.request();
    let ack = f.store().record_authorization(&original).unwrap();
    let mut repeated: Value = serde_json::from_slice(&f.event).unwrap();
    repeated["seq"] = serde_json::json!(ack.body.authority_sequence + 1);
    let repeated = f
        .client()
        .request(&serde_json::to_vec(&repeated).unwrap())
        .unwrap();
    let mut store = f.store();
    assert!(store
        .record_authorization(&repeated)
        .unwrap_err()
        .to_string()
        .contains("numerical work already witnessed"));
    assert_eq!(store.record_authorization(&original).unwrap(), ack);
    assert_eq!(store.sequence, 1);
}

#[test]
fn public_history_verifies_without_private_keys_and_refuses_live_or_torn_state() {
    let f = Fixture::new();
    let request = f.request();
    let mut store = f.store();
    store.record_authorization(&request).unwrap();
    assert!(
        WitnessStore::verify_public_history(&f.root, &f.policy).is_err(),
        "a live writer lease cannot be relabelled a frozen snapshot"
    );
    drop(store);
    fs::remove_file(&f.ack_key).unwrap();
    fs::remove_file(&f.authority_key).unwrap();
    let before = fs::read(f.root.join("head.json")).unwrap();
    let verified = WitnessStore::verify_public_history(&f.root, &f.policy).unwrap();
    assert_eq!(verified["private_key_accessed"], false);
    assert_eq!(verified["new_acknowledgement_issued"], false);
    assert_eq!(fs::read(f.root.join("head.json")).unwrap(), before);
    let first = fs::read_dir(f.root.join("segments"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    fs::remove_file(first).unwrap();
    assert!(
        WitnessStore::verify_public_history(&f.root, &f.policy).is_err(),
        "a committed missing original event must not verify"
    );
}

#[test]
fn public_snapshot_requires_exact_original_authority_ack_membership() {
    let f = Fixture::new();
    let request = f.request();
    let ack = f.store().record_authorization(&request).unwrap();
    let result = WitnessStore::verify_public_history_with_acks(
        &f.root,
        &f.policy,
        std::slice::from_ref(&ack),
    )
    .unwrap();
    assert_eq!(result["required_ack_count"], 1);
    let mut absent = ack.clone();
    absent.body.authority_sequence += 1;
    assert!(WitnessStore::verify_public_history_with_acks(&f.root, &f.policy, &[absent]).is_err());
    let mut conflicting = ack;
    conflicting.body.event_sha256 = sha(b"different original event");
    assert!(
        WitnessStore::verify_public_history_with_acks(&f.root, &f.policy, &[conflicting]).is_err()
    );
}
#[test]
fn witness_lease_fifo_refuses_without_waiting_for_writer() {
    let t = tempfile::tempdir().unwrap();
    let path = t.path().join("lease");
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    assert!(checked_file(&path).is_err());
}

#[test]
fn transport_only_loader_never_opens_authority_key_or_claims_authority_runtime() {
    let fixture = Fixture::new();
    let mut config = fixture.config();
    config.authority_key = fixture.root.join("absent-authority-secret");
    config.policy.authority_executable_sha256 = sha(b"different authority producer");
    let bytes = serde_json::to_vec(&config).unwrap();
    let path = fixture.root.join("transport-only.json");
    fs::write(&path, &bytes).unwrap();
    let loaded = WitnessClientConfig::from_transport_file(&path, &sha(&bytes)).unwrap();
    assert!(!loaded.authority_key.exists());
    assert!(WitnessClientConfig::from_file(&path, &sha(&bytes)).is_err());
    assert!(WitnessClientConfig::from_transport_file(&path, &sha(b"changed config")).is_err());
}
