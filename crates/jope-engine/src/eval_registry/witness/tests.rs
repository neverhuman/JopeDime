use super::*;
use crate::eval_registry::{
    Authority, ClaimOutcome, ProcessIdentity, RequestProvenance, VerifiedFit,
};
use serde_json::json;
use std::os::unix::net::{UnixListener, UnixStream};

struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    authority: PathBuf,
    watermark: PathBuf,
    policy: WitnessPolicy,
    authority_key: PathBuf,
    ack_key: PathBuf,
    known_hosts: PathBuf,
    event: Vec<u8>,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("independent-witness");
        let authority = temp.path().join("authority");
        let watermark = temp.path().join("legacy-local-watermark");
        crate::eval_registry::synthetic_completion_for_test(
            &authority,
            &watermark,
            "(pipe X (ridge alpha=1))",
            vec![vec![1.]],
            vec![2.],
            b"pp".to_vec(),
            b"view".to_vec(),
            0,
        )
        .unwrap();
        let text = fs::read_to_string(authority.join("dispatch.jsonl")).unwrap();
        let event = text
            .lines()
            .find(|line| serde_json::from_str::<Value>(line).unwrap()["kind"] == "authorize")
            .unwrap()
            .as_bytes()
            .to_vec();
        let original: OriginalEvent = serde_json::from_slice(&event).unwrap();
        let authority_key = temp.path().join("authority-signing.key");
        let ack_key = temp.path().join("witness-ACK.key");
        // Private deterministic test keys are generated only inside this owned
        // temporary fixture. They authorize no real numerical execution.
        publish(&authority_key, &[17; 32]).unwrap();
        publish(&ack_key, &[29; 32]).unwrap();
        let known_hosts = temp.path().join("known_hosts");
        publish(&known_hosts, b"synthetic fixture has no remote SSH host\n").unwrap();
        let executable = sha(&fs::read("/proc/self/exe").unwrap());
        let policy = WitnessPolicy {
            schema: "parallel.fit_event_witness_policy.v1".into(),
            witness_instance: sha(b"synthetic-independent-witness"),
            authority_instance: original.instance.clone(),
            authority_epoch: original.effects[0].row["epoch"].as_str().unwrap().into(),
            authority_public_key: hex::encode(
                private_key(&authority_key)
                    .unwrap()
                    .verifying_key()
                    .to_bytes(),
            ),
            witness_public_key: hex::encode(
                private_key(&ack_key).unwrap().verifying_key().to_bytes(),
            ),
            authority_executable_sha256: executable.clone(),
            authority_source_sha256: authority_source_sha256(),
            witness_executable_sha256: executable,
            witness_source_sha256: source_sha256(),
            ssh_known_hosts_sha256: sha(&read(&known_hosts).unwrap()),
            ssh_host_alias: "synthetic".into(),
            ssh_destination: "synthetic".into(),
            native_fit_trust: None,
            native_grant_delegation: None,
            funding_trust: None,
        };
        WitnessStore::initialize(&root, &policy).unwrap();
        Self {
            _temp: temp,
            root,
            authority,
            watermark,
            policy,
            authority_key,
            ack_key,
            known_hosts,
            event,
        }
    }
    fn config(&self) -> WitnessClientConfig {
        WitnessClientConfig {
            epoch_proofs: Vec::new(),
            ssh_identity: None,
            policy: self.policy.clone(),
            authority_key: self.authority_key.clone(),
            known_hosts: self.known_hosts.clone(),
            remote_executable: "/synthetic/jope-parallel".into(),
            remote_root: "/synthetic/witness".into(),
            remote_policy: "/synthetic/policy.json".into(),
            remote_ack_key: "/private/key".into(),
            local_socket: None,
        }
    }
    fn client(&self) -> WitnessClient {
        WitnessClient::new(self.config()).unwrap()
    }
    fn store(&self) -> WitnessStore {
        WitnessStore::open(&self.root, &self.policy, &self.ack_key).unwrap()
    }
    fn request(&self) -> SignedRequest {
        self.client().request(&self.event).unwrap()
    }
    fn next_fit(&self) -> VerifiedFit {
        VerifiedFit::for_current_executor(
            "(pipe X (ridge alpha=2))",
            vec![vec![1.]],
            vec![2.],
            b"pp".to_vec(),
            b"view".to_vec(),
            0,
            &crate::eval_registry::synthetic_contract_for_test(),
        )
        .unwrap()
    }
}
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
    let record: Record =
        serde_json::from_slice(&read(&f.root.join("segments/00000000000000000001.json")).unwrap())
            .unwrap();
    assert_eq!(record.core.request.body.event, f.event);
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
                bytes(&Head {
                    sequence: 1,
                    sha256: sha(b"wrong"),
                })
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
        bytes(&Head {
            sequence: 0,
            sha256: ZERO.into(),
        })
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
    assert_eq!(ack.body.authority_sequence > 0, true);
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
    let thread = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request: SignedRequest = read_frame(&mut stream).unwrap();
        let mut store = WitnessStore::open(&root, &policy, &ack_key).unwrap();
        store.record_authorization(&request).unwrap();
        std::thread::sleep(Duration::from_millis(600));
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
    assert!(authority
        .authorize_with_deadline(
            &fit,
            &claim,
            &ProcessIdentity::current().unwrap(),
            Duration::from_millis(250)
        )
        .is_err());
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
    verify_request(&f.policy, &got).unwrap();
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

#[test]
fn pinned_ssh_identity_refuses_mutation_permissions_symlinks_and_fifo() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("transport-key");
    fs::write(
        &path,
        b"synthetic transport key bytes, never an authorization",
    )
    .unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    let identity = PinnedSshIdentity {
        path: path.clone(),
        sha256: sha(&fs::read(&path).unwrap()),
    };
    assert!(identity.open().is_ok());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
    assert!(identity.open().is_err());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(&path, b"changed").unwrap();
    assert!(identity.open().is_err());
    let alias = dir.path().join("alias");
    std::os::unix::fs::symlink(&path, &alias).unwrap();
    assert!(PinnedSshIdentity {
        path: alias,
        sha256: sha(b"changed")
    }
    .open()
    .is_err());
    fs::remove_file(&path).unwrap();
    let raw = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(raw.as_ptr(), 0o600) }, 0);
    assert!(identity.open().is_err());
}

#[test]
fn pinned_ssh_transport_uses_original_descriptor_and_disables_agent() {
    use std::os::fd::AsRawFd;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("transport-key");
    fs::write(&path, b"original fixture key").unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    let identity = PinnedSshIdentity {
        path: path.clone(),
        sha256: sha(b"original fixture key"),
    };
    let mut command = Command::new("/usr/bin/ssh");
    let retained = bind_ssh_identity(&mut command, &identity).unwrap();
    let descriptor_path = format!("/proc/{}/fd/{}", std::process::id(), retained.as_raw_fd());
    let args: Vec<_> = command
        .get_args()
        .map(|s| s.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        args,
        vec![
            "-i",
            &descriptor_path,
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "IdentityAgent=none"
        ]
    );
    fs::rename(&path, dir.path().join("retained")).unwrap();
    fs::write(&path, b"replacement fixture key").unwrap();
    assert_eq!(fs::read(&descriptor_path).unwrap(), b"original fixture key");
    assert!(identity.open().is_err());
    assert_ne!(
        unsafe { libc::fcntl(retained.as_raw_fd(), libc::F_GETFD) } & libc::FD_CLOEXEC,
        0
    );
}

#[test]
fn legacy_witness_config_omits_identity_but_native_requires_pin() {
    let fixture = Fixture::new();
    let config = fixture.config();
    let encoded = serde_json::to_value(&config).unwrap();
    assert!(encoded.get("ssh_identity").is_none());
    let mut restored: WitnessClientConfig = serde_json::from_value(encoded).unwrap();
    assert!(restored.ssh_identity.is_none());
    restored.policy.native_fit_trust = Some(NativeWitnessTrust {
        authorization_public_key: restored.policy.authority_public_key.clone(),
        grant_sha256: "a".repeat(64),
    });
    assert!(WitnessClient::new(restored)
        .err()
        .unwrap()
        .to_string()
        .contains("explicitly pinned SSH identity"));
}

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
    let result =
        WitnessStore::verify_public_history_with_acks(&f.root, &f.policy, &[ack.clone()]).unwrap();
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
