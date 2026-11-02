use super::*;
use ed25519_dalek::{Signer, SigningKey};
fn grant() -> NativeGrant {
    let h = "a".repeat(64);
    NativeGrant {
        schema: "jope.smartcluster_fit_grant.v1".into(),
        root: "/tmp/root".into(),
        authority_root: "/tmp/root/authority".into(),
        witness: "/tmp/witness".into(),
        authority_instance: h.clone(),
        authority_epoch: h.clone(),
        authority_manifest_sha256: h.clone(),
        capability: "synthetic_qualification".into(),
        scope: "qualification".into(),
        tenant: "jope".into(),
        project: "qualification".into(),
        daemon_socket: "/run/native/socket".into(),
        cgroup_root: "/sys/fs/cgroup/native".into(),
        daemon_uid: 1000,
        daemon_executable_sha256: h.clone(),
        authority_executable_sha256: h.clone(),
        evaluator_executable_sha256: h.clone(),
        allowed_runtime_sha256: BTreeSet::from([h.clone()]),
        witness_public_key: h.clone(),
        decision_receipt_sha256: h.clone(),
        release_receipt_sha256: h.clone(),
        coverage_receipt_sha256: h.clone(),
        approved_task_sha256: BTreeSet::from([h.clone()]),
        period_start_unix_ms: 1,
        period_end_unix_ms: 100,
        max_candidates_per_task: 1,
        max_fit_ms: 30000,
        numerical_contract_sha256: h.clone(),
        training_sha256: BTreeSet::from([h]),
        fit_seed: 0,
    }
}
#[test]
fn signature_and_exact_original_grant_are_required() {
    let key = SigningKey::from_bytes(&[9; 32]);
    let body = grant();
    let bytes = serde_json::to_vec(&body).unwrap();
    let mut message = SIGNATURE_DOMAIN.to_vec();
    message.extend(&bytes);
    let signed = SignedNativeGrant {
        body,
        signature: hex::encode(key.sign(&message).to_bytes()),
    };
    let trust = NativeWitnessTrust {
        authorization_public_key: hex::encode(key.verifying_key().as_bytes()),
        grant_sha256: hash(&bytes),
    };
    verify_signed(&trust, &signed, 2).unwrap();
    assert!(verify_signed(&trust, &signed, 100).is_err());
    let mut changed = signed.clone();
    changed.body.max_candidates_per_task = 2;
    assert!(verify_signed(&trust, &changed, 2).is_err());
}
#[test]
fn unknown_capability_and_unbounded_tasks_fail() {
    let mut value = grant();
    value.capability = "arbitrary_execution".into();
    assert!(value.validate(2).is_err());
    value = grant();
    value.max_fit_ms = 30001;
    assert!(value.validate(2).is_err());
    value = grant();
    value.approved_task_sha256.clear();
    assert!(value.validate(2).is_err());
}
#[test]
fn hash_case_and_paths_do_not_broaden_authorization() {
    let mut g = grant();
    g.daemon_executable_sha256 = "A".repeat(64);
    assert!(g.validate(2).is_err());
    g = grant();
    g.authority_root = "/tmp/root/../outside".into();
    assert!(g.validate(2).is_err());
    g = grant();
    g.period_start_unix_ms = 3;
    assert!(g.validate(2).is_err());
}
#[test]
fn bounded_reader_rejects_fifo_symlink_and_partial_json() {
    let path =
        std::env::temp_dir().join(format!("jope-native-policy-reader-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    let fifo = path.join("fifo");
    let c = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: `c` is a live NUL-terminated path in this private test directory;
    // mkfifo borrows it for the call and receives ordinary permission bits.
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    assert!(read_file(&fifo).is_err());
    let regular = path.join("regular");
    std::fs::write(&regular, b"{\"schema\":").unwrap();
    assert!(serde_json::from_slice::<SignedNativeGrant>(&read_file(&regular).unwrap()).is_err());
    let link = path.join("link");
    std::os::unix::fs::symlink(&regular, &link).unwrap();
    assert!(read_file(&link).is_err());
    std::fs::remove_dir_all(path).unwrap();
}
