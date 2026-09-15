use super::*;
use ed25519_dalek::{Signer, SigningKey};

#[test]
fn reviewed_transition_signature_binds_exact_bytes_domain_and_original_window() {
    let directory = tempfile::tempdir().unwrap();
    let process = ProcessIdentity::current().unwrap();
    let attempt = || {
        serde_json::from_value(
            json!({"job_id":jain_smartcluster::JobId::new(),"number":1,"fence":1}),
        )
        .unwrap()
    };
    let held = BTreeMap::from([(
        "fit".into(),
        HeldBinding {
            row_sha256: "1".repeat(64),
            request_sha256: "2".repeat(64),
            custody: OriginalCustody {
                authority_attempt: attempt(),
                target_attempt: attempt(),
                authority_leader: process.clone(),
                target_leader: process.clone(),
                requesting_process: process,
            },
            historical_custody: None,
        },
    )]);
    let decision = json!({"schema":"jope.native_held_epoch_review.v1","authority_instance":"a".repeat(64),
        "parent_epoch":"b".repeat(64),"new_epoch":"c".repeat(64),"report_sha256":"d".repeat(64),
        "held_bindings_sha256":hash(&encoded(&held).unwrap()),"original_identities_remain_held":true,
        "original_outcomes_are_not_inferred":true});
    let decision_bytes = encoded(&decision).unwrap();
    let decision_path = directory.path().join("decision.json");
    fs::write(&decision_path, &decision_bytes).unwrap();
    let body = NativeRecoveryBody {
        schema: "jope.native_held_epoch.v1".into(),
        authority_root: directory.path().join("authority"),
        witness: directory.path().join("watermark"),
        authority_instance: "a".repeat(64),
        authority_manifest_sha256: "e".repeat(64),
        parent_epoch: "b".repeat(64),
        new_epoch: "c".repeat(64),
        report_sha256: "d".repeat(64),
        decision: RecoveryArtifact {
            path: decision_path,
            sha256: hash(&decision_bytes),
        },
        not_before_unix_ms: 100,
        expires_unix_ms: 200,
        daemon_socket: directory.path().join("daemon.sock"),
        daemon_uid: 1000,
        daemon_executable_sha256: "f".repeat(64),
        cgroup_root: "/sys/fs/cgroup/test".into(),
        held,
        independent_witness: PublicWitnessSnapshot {
            root: directory.path().join("public-witness"),
            policy: RecoveryArtifact {
                path: directory.path().join("public-policy.json"),
                sha256: "a".repeat(64),
            },
            head_sha256: "b".repeat(64),
        },
    };
    let signing = SigningKey::from_bytes(&[19; 32]);
    let bytes = encoded(&body).unwrap();
    let mut message = DOMAIN.to_vec();
    message.extend(&bytes);
    let mut signed = SignedNativeRecovery {
        authorization_public_key: hex::encode(signing.verifying_key().to_bytes()),
        body_sha256: hash(&bytes),
        body,
        signature: hex::encode(signing.sign(&message).to_bytes()),
    };
    let mut drain_map = serde_json::Map::new();
    for held in signed.body.held.values() {
        for (a, p) in [
            (
                held.custody.authority_attempt,
                &held.custody.authority_leader,
            ),
            (held.custody.target_attempt, &held.custody.target_leader),
        ] {
            let record = jain_smartcluster::AttemptRecord {
                attempt: a,
                state: jain_smartcluster::JobState::Failed,
                lease: jain_smartcluster::AttemptLease {
                    attempt: a,
                    acquired_at: jain_smartcluster::Timestamp::from(1),
                    expires_at: jain_smartcluster::Timestamp::from(2),
                },
                pid: Some(p.pid),
                cgroup_path: Some(
                    signed
                        .body
                        .cgroup_root
                        .join(format!(
                            "scq-{}-attempt-{}-fence-{}",
                            a.job_id, a.number, a.fence
                        ))
                        .to_string_lossy()
                        .into_owned(),
                ),
                started_at: Some(jain_smartcluster::Timestamp::from(1)),
                finished_at: Some(jain_smartcluster::Timestamp::from(2)),
                failure: None,
            };
            drain_map.insert(
                format!("{}:{}:{}", a.job_id, a.number, a.fence),
                serde_json::to_value(jain_smartcluster::AttemptInspection {
                    record,
                    active_process: None,
                })
                .unwrap(),
            );
        }
    }
    let drains = Value::Object(drain_map);
    verify_recorded_drains(&signed.body, &drains).unwrap();
    let first = drains.as_object().unwrap().keys().next().unwrap().clone();
    let mut missing = drains.clone();
    missing.as_object_mut().unwrap().remove(&first);
    assert!(verify_recorded_drains(&signed.body, &missing).is_err());
    let mut unfinished = drains.clone();
    unfinished[&first]["record"]["finished_at"] = Value::Null;
    assert!(verify_recorded_drains(&signed.body, &unfinished).is_err());
    let mut wrong = drains.clone();
    wrong[&first]["record"]["pid"] = json!(99999);
    assert!(verify_recorded_drains(&signed.body, &wrong).is_err());
    verify_signed(&signed, 100).unwrap();
    verify_signed(&signed, 199).unwrap();
    assert!(verify_signed(&signed, 99).is_err());
    assert!(verify_signed(&signed, 200).is_err());
    signed.body.new_epoch = "9".repeat(64);
    assert!(verify_signed(&signed, 150).is_err());
    signed.body.new_epoch = "c".repeat(64);
    signed.signature = hex::encode(signing.sign(&bytes).to_bytes());
    assert!(verify_signed(&signed, 150).is_err());
}

#[test]
fn read_only_epoch_status_preserves_bytes_and_refuses_pending_or_torn_history() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("authority");
    let witness = directory.path().join("watermark");
    Authority::initialize_native(
        &root,
        &witness,
        vec![ScopePolicy {
            scope: "synthetic".into(),
            training_sha256: vec!["a".repeat(64)],
            numerical_contract_sha256: vec!["b".repeat(64)],
            fit_seeds: vec![0],
        }],
    )
    .unwrap();
    let journal = fs::read(root.join("dispatch.jsonl")).unwrap();
    let watermark = fs::read(&witness).unwrap();
    let status = Authority::native_epoch_status(&root, &witness).unwrap();
    assert_eq!(status["authority_instance"], status["authority_epoch"]);
    assert!(status["recovery_receipt_sha256"].is_null());
    assert_eq!(fs::read(root.join("dispatch.jsonl")).unwrap(), journal);
    assert_eq!(fs::read(&witness).unwrap(), watermark);
    fs::write(root.join("recovery.pending"), b"retained uncertainty").unwrap();
    assert!(Authority::native_epoch_status(&root, &witness).is_err());
    fs::remove_file(root.join("recovery.pending")).unwrap();
    OpenOptions::new()
        .append(true)
        .open(root.join("dispatch.jsonl"))
        .unwrap()
        .write_all(b"torn")
        .unwrap();
    assert!(Authority::native_epoch_status(&root, &witness).is_err());
}
