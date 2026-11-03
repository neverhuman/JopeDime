use super::*;

#[test]
fn framed_socket_roundtrip_commits_original_event_before_authority_releases_start() {
    let f = Fixture::new();
    let socket = f._temp.path().join("event.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let root = f.root.clone();
    let policy = f.policy.clone();
    let ack_key = f.ack_key.clone();
    let thread = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request: SignedRequest = read_frame(&mut stream).unwrap();
        let mut store = WitnessStore::open(&root, &policy, &ack_key).unwrap();
        let ack = store.record_authorization(&request).unwrap();
        write_frame(&mut stream, &ack).unwrap();
        ack
    });
    let mut authority = Authority::open(&f.authority, &f.watermark).unwrap();
    let mut config = f.config();
    config.local_socket = Some(socket);
    authority.attach_event_witness(config).unwrap();
    let fit = f.next_fit();
    let request = RequestProvenance {
        scope: "synthetic".into(),
        task_id: "signed-witness-no-fit".into(),
        details: json!({"synthetic":true}),
    };
    let claim = match authority.claim(&fit, &request).unwrap() {
        ClaimOutcome::Claimed(c) => c,
        _ => panic!("synthetic claim denied"),
    };
    let start = authority
        .authorize_with_deadline(
            &fit,
            &claim,
            &ProcessIdentity::current().unwrap(),
            Duration::from_secs(20),
        )
        .unwrap();
    let ack = thread.join().unwrap();
    assert!(ack.body.authority_sequence > 0);
    assert_eq!(start.fit_key, fit.key());
    let journal = fs::read_to_string(f.authority.join("dispatch.jsonl")).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(journal.lines().last().unwrap()).unwrap()["kind"],
        "witness_ack"
    );
    let report = Authority::recovery_inspect(&f.authority, &f.watermark).unwrap();
    assert_eq!(report["retained_artifact_errors"], json!([]));
    assert!(f.root.join("segments/00000000000000000001.json").is_file());
}
#[test]
fn lost_ack_keeps_original_authorization_unavailable_and_deadline_is_charged() {
    let f = Fixture::new();
    let socket = f._temp.path().join("event.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let root = f.root.clone();
    let policy = f.policy.clone();
    let ack_key = f.ack_key.clone();
    let (authorization_returned, wait_for_authorization) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request: SignedRequest = read_frame(&mut stream).unwrap();
        let mut store = WitnessStore::open(&root, &policy, &ack_key).unwrap();
        store.record_authorization(&request).unwrap();
        // Withhold the ACK until the bounded caller has actually returned.
        wait_for_authorization
            .recv_timeout(Duration::from_secs(40))
            .unwrap();
        request
    });
    let mut authority = Authority::open(&f.authority, &f.watermark).unwrap();
    let mut config = f.config();
    config.local_socket = Some(socket);
    authority.attach_event_witness(config).unwrap();
    let fit = f.next_fit();
    let request = RequestProvenance {
        scope: "synthetic".into(),
        task_id: "lost-ACK-no-fit".into(),
        details: json!({}),
    };
    let claim = match authority.claim(&fit, &request).unwrap() {
        ClaimOutcome::Claimed(c) => c,
        _ => panic!("synthetic claim denied"),
    };
    // Journal preparation is charged to admission too. A 250ms input could
    // expire before the request was written under load, missing the lost-ACK case.
    let budget = Duration::from_secs(30);
    let began = std::time::Instant::now();
    let result = authority.authorize_with_deadline(
        &fit,
        &claim,
        &ProcessIdentity::current().unwrap(),
        budget,
    );
    authorization_returned.send(()).unwrap();
    assert!(result.is_err());
    assert!(began.elapsed() >= budget);
    let original = thread.join().unwrap();
    assert!(f.store().get_ack(&original).unwrap().is_some());
    assert!(Authority::open(&f.authority, &f.watermark).is_err());
    let report = Authority::recovery_inspect(&f.authority, &f.watermark).unwrap();
    assert!(!report["unresolved_original_work"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(Authority::recovery_replay(&f.authority, &f.watermark).is_err());
}
#[test]
fn bounded_frames_reject_oversized_and_truncated_payloads() {
    assert!(read_frame::<SignedRequest>(&((MAX_FRAME + 1) as u32).to_be_bytes()[..]).is_err());
    assert!(read_frame::<SignedRequest>(&[0, 0, 0, 8, b'{'][..]).is_err());
    let f = Fixture::new();
    let (mut a, mut b) = UnixStream::pair().unwrap();
    let request = f.request();
    let sender = std::thread::spawn(move || write_frame(&mut a, &request));
    let got: SignedRequest = read_frame(&mut b).unwrap();
    sender.join().unwrap().unwrap();
    let mut store = f.store();
    let ack = store.record_authorization(&got).unwrap();
    ack.verify(&f.policy, &got).unwrap();
}

#[test]
fn native_witness_requires_exact_signed_grant_and_original_child_bindings() {
    use crate::eval_registry::native_scq::policy::{
        NativeGrant, NativeWitnessTrust, SignedNativeGrant, SIGNATURE_DOMAIN,
    };
    use ed25519_dalek::Signer;
    use jain_smartcluster::{
        AttemptNumber, FenceToken, FencedAttempt, JobId, TrustedFitChildPlacement,
    };
    let f = Fixture::new();
    let mut event: Value = serde_json::from_slice(&f.event).unwrap();
    let row = &event["effects"][0]["row"];
    let identity: crate::eval_registry::FitIdentity =
        serde_json::from_str(row["identity_json"].as_str().unwrap()).unwrap();
    let start: crate::eval_registry::StartToken =
        serde_json::from_str(row["start_json"].as_str().unwrap()).unwrap();
    let digest = sha(b"synthetic native manifest");
    let descriptor = sha(b"synthetic sealed descriptor");
    let prerequisite = sha(b"synthetic no numerical authorization");
    let grant:NativeGrant=serde_json::from_value(json!({
        "schema":"jope.smartcluster_fit_grant.v1","root":f._temp.path(),"authority_root":f.authority,"witness":f.watermark,
        "authority_instance":f.policy.authority_instance,"authority_epoch":f.policy.authority_epoch,"authority_manifest_sha256":prerequisite,
        "capability":"synthetic_qualification","scope":"synthetic","tenant":"qualification","project":"native-fit",
        "daemon_socket":"/run/synthetic/socket","cgroup_root":"/sys/fs/cgroup/synthetic","daemon_uid":1000,"daemon_executable_sha256":prerequisite,
        "authority_executable_sha256":f.policy.authority_executable_sha256,"evaluator_executable_sha256":identity.executable_sha256,
        "allowed_runtime_sha256":[identity.runtime_sha256],"witness_public_key":f.policy.witness_public_key,
        "decision_receipt_sha256":prerequisite,"release_receipt_sha256":prerequisite,"coverage_receipt_sha256":prerequisite,
        "approved_task_sha256":[digest],"period_start_unix_ms":1,"period_end_unix_ms":100,"max_candidates_per_task":1,"max_fit_ms":30000,
        "numerical_contract_sha256":identity.numerical_contract_sha256,"training_sha256":[identity.training_sha256],"fit_seed":identity.fit_seed
    })).unwrap();
    let key = SigningKey::from_bytes(&[73; 32]);
    let body = serde_json::to_vec(&grant).unwrap();
    let mut message = SIGNATURE_DOMAIN.to_vec();
    message.extend(&body);
    let signed = SignedNativeGrant {
        body: grant,
        signature: hex::encode(key.sign(&message).to_bytes()),
    };
    let trust = NativeWitnessTrust {
        authorization_public_key: hex::encode(key.verifying_key().as_bytes()),
        grant_sha256: sha(&body),
    };
    let attempt = |fence| {
        FencedAttempt::new(
            JobId::new(),
            AttemptNumber::new(1).unwrap(),
            FenceToken::new(fence).unwrap(),
        )
    };
    let authority = attempt(1);
    let target = attempt(2);
    let mut placement = TrustedFitChildPlacement {
        schema: "jope.smartcluster_fit_placement.v1".into(),
        authority_attempt: authority,
        target_attempt: target,
        child_pid: start.executor.pid,
        child_birth_ticks: start.executor.birth_ticks,
        child_boot_id: start.executor.boot_id.clone(),
        child_executable_sha256: identity.executable_sha256.clone(),
        task_manifest_sha256: digest.clone(),
        descriptor_sha256: descriptor.clone(),
        candidate_index: 0,
        cgroup_path: PathBuf::from(format!(
            "/sys/fs/cgroup/synthetic/scq-{}-attempt-{}-fence-{}",
            target.job_id, target.number, target.fence
        )),
        receipt_sha256: String::new(),
    };
    let mut value = serde_json::to_value(&placement).unwrap();
    value.as_object_mut().unwrap().remove("receipt_sha256");
    placement.receipt_sha256 = sha(&serde_json::to_vec(&value).unwrap());
    let native = json!({"schema":"jope.native_fit_authorization.v1","signed_grant":signed,"authorized_unix_ms":2,"provenance":{"scope":"synthetic","task_id":start.task_id},"target_attempt":target,"authority_attempt":authority,"task_manifest_sha256":digest,"descriptor_sha256":descriptor,"candidate_index":0,"fit_key":start.fit_key,"deadline_monotonic_ns":start.deadline_monotonic_ns,"placement":placement});
    event["payload"]["native_fit"] = native.clone();
    assert!(event_identity(&f.policy, &serde_json::to_vec(&event).unwrap()).is_err());
    let mut policy = f.policy.clone();
    policy.native_fit_trust = Some(trust);
    event_identity(&policy, &serde_json::to_vec(&event).unwrap()).unwrap();
    for (field, value) in [
        ("candidate_index", json!(1)),
        ("descriptor_sha256", json!(sha(b"changed"))),
        ("schema", json!("caller-approved")),
        ("authorized_unix_ms", json!(100)),
    ] {
        let mut changed = event.clone();
        changed["payload"]["native_fit"][field] = value;
        assert!(
            event_identity(&policy, &serde_json::to_vec(&changed).unwrap()).is_err(),
            "accepted conflicting {field}"
        );
    }
    let mut changed = event.clone();
    changed["payload"]["native_fit"]["signed_grant"]["body"]["max_candidates_per_task"] = json!(2);
    assert!(event_identity(&policy, &serde_json::to_vec(&changed).unwrap()).is_err());
    let mut changed = event;
    changed["payload"]["native_fit"]["placement"]["child_birth_ticks"] =
        json!(start.executor.birth_ticks + 1);
    assert!(event_identity(&policy, &serde_json::to_vec(&changed).unwrap()).is_err());
}
