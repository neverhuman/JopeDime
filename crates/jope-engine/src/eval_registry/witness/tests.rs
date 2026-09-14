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
