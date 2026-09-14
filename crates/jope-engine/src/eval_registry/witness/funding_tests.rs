//! Private signed funding fixtures exercise the real authority journal and
//! framed witness transport. No fit, CUDA context or production trust is used.
use super::*;
use crate::eval_registry::{funding as f, funding_config::{ClientConfig, InstalledFunding}, Authority, ProcessIdentity, ScopePolicy};
use ed25519_dalek::Signer;
use rusqlite::Connection;
use serde_json::json;
use std::collections::BTreeSet;
use std::os::unix::net::UnixListener;

pub(crate) struct Fixture {
    _temp: tempfile::TempDir,
    pub(crate) authority: PathBuf,
    pub(crate) watermark: PathBuf,
    witness: PathBuf,
    socket: PathBuf,
    pub(crate) installation: PathBuf,
    policy: WitnessPolicy,
    pub(crate) client: WitnessClientConfig,
    ack_key: PathBuf,
    pub(crate) authorization: f::SignedAuthorization,
    pub(crate) reservation: f::SignedReservation,
    pub(crate) descriptor: Vec<u8>,
    execution: f::ExecutionClaim,
    delegate: SigningKey,
}

impl Fixture {
    pub(crate) fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let authority = temp.path().join("authority");
        let watermark = temp.path().join("local-watermark");
        Authority::initialize(&authority, &watermark, vec![ScopePolicy {
            scope:"private-no-numerics".into(), training_sha256:vec!["1".repeat(64)],
            numerical_contract_sha256:vec!["2".repeat(64)], fit_seeds:vec![0],
        }]).unwrap();
        let manifest_bytes = fs::read(authority.join("authority.json")).unwrap();
        let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
        let instance = manifest["instance"].as_str().unwrap().to_string();
        let process = ProcessIdentity::current().unwrap();
        let executable = sha(&fs::read("/proc/self/exe").unwrap());
        let clock = f::FundingClock::current().unwrap();
        let authorizer = SigningKey::from_bytes(&[11; 32]);
        let delegate = SigningKey::from_bytes(&[37; 32]);
        let delegated = hex::encode(delegate.verifying_key().to_bytes());
        let descriptor = serde_json::to_vec(&json!({"schema":"private.funding_task.v1",
            "task":"metadata-only witness fixture"})).unwrap();
        let body = f::FundingAuthorization {
            schema:"parallel.funding_authorization.v1".into(), domain_id:"private-witness-fixture".into(),
            scope:"private signed fixture; never a production grant".into(),
            period_start_unix_ms:clock.unix_ms - 1_000, period_end_unix_ms:clock.unix_ms + 300_000,
            authority_evidence_sha256:"3".repeat(64), historical_reconciliation_sha256:"4".repeat(64),
            release_prerequisites_sha256:"5".repeat(64), approved_task_sha256:BTreeSet::from([sha(&descriptor)]),
            historical_core_ms:500, historical_node_ms:200, core_ms:1_000_000, node_ms:500_000,
            max_concurrent_cpu_searches:16,
            allocation:f::FundingAllocation { cpu_cores:16,gpu_devices:(0..8).collect(),
                host_memory_bytes:16_000,gpu_memory_bytes_per_device:8_000 },
            capabilities:BTreeSet::from(["regression_training".into()]), reservation_public_key:delegated.clone(),
        };
        let mut signing_bytes = b"parallel.funding-authorization.signature.v1\0".to_vec();
        signing_bytes.extend(serde_json::to_vec(&body).unwrap());
        let authorization = f::SignedAuthorization {
            signature:hex::encode(authorizer.sign(&signing_bytes).to_bytes()), body,
        };
        let trust = f::FundingTrust {
            schema:"parallel.funding_trust.v1".into(),domain_id:authorization.body.domain_id.clone(),
            authority_instance:instance.clone(),authority_epoch:instance.clone(),
            authority_manifest_sha256:sha(&manifest_bytes),authority_executable_sha256:executable.clone(),
            authorization_public_key:hex::encode(authorizer.verifying_key().to_bytes()),
            authorization_sha256:f::digest(&authorization.body).unwrap(),reservation_public_key:delegated,
            host_id:fs::read_to_string("/proc/sys/kernel/hostname").unwrap().trim().into(),boot_id:process.boot_id.clone(),
        };
        let reservation = f::sign_reservation(f::TaskReservation {
            schema:"parallel.task_reservation.v1".into(),reservation_id:"6".repeat(64),
            authorization_sha256:trust.authorization_sha256.clone(),authority_instance:instance.clone(),
            authority_epoch:instance.clone(),authority_manifest_sha256:trust.authority_manifest_sha256.clone(),
            authority_executable_sha256:executable.clone(),domain_id:trust.domain_id.clone(),
            not_before_unix_ms:clock.unix_ms - 1_000,
            demand:f::TaskDemand {task_sha256:sha(&descriptor),capability:"regression_training".into(),
                host_id:trust.host_id.clone(),boot_id:process.boot_id.clone(),executable_sha256:executable.clone(),
                runtime_sha256:"7".repeat(64),deadline_unix_ms:clock.unix_ms + 120_000,
                deadline_monotonic_ns:clock.monotonic_ns + 120_000_000_000,
                resources:f::ResourceCaps {cpu_cores:2,gpus:2,world:2,host_memory_bytes:4_000,
                    gpu_memory_bytes_per_rank:2_000,cpu_searches:0,gpu_devices:vec![0,1]}},
            core_ms:242_000,node_ms:121_000,participant_slots:vec![0,1],
        }, &delegate).unwrap();
        let authority_key = temp.path().join("authority.key");
        let ack_key = temp.path().join("independent-ack.key");
        let known_hosts = temp.path().join("known_hosts");
        publish(&authority_key, &[17;32]).unwrap();
        publish(&ack_key, &[29;32]).unwrap();
        publish(&known_hosts,b"private framed local fixture; no SSH\n").unwrap();
        let policy = WitnessPolicy {
            schema:"parallel.fit_event_witness_policy.v1".into(),witness_instance:sha(b"private-funding-witness"),
            authority_instance:instance.clone(),authority_epoch:instance,
            authority_public_key:hex::encode(private_key(&authority_key).unwrap().verifying_key().to_bytes()),
            witness_public_key:hex::encode(private_key(&ack_key).unwrap().verifying_key().to_bytes()),
            authority_executable_sha256:executable.clone(),authority_source_sha256:authority_source_sha256(),
            witness_executable_sha256:executable.clone(),witness_source_sha256:source_sha256(),
            ssh_known_hosts_sha256:sha(&read(&known_hosts).unwrap()),ssh_host_alias:"private".into(),
            ssh_destination:"private".into(),funding_trust:Some(trust.clone()),
        };
        let witness = temp.path().join("independent-witness");
        WitnessStore::initialize(&witness,&policy).unwrap();
        let socket = temp.path().join("witness.sock");
        let client = WitnessClientConfig {policy:policy.clone(),authority_key,known_hosts,
            remote_executable:"/private/jope-parallel".into(),remote_root:"/private/witness".into(),
            remote_policy:"/private/policy.json".into(),remote_ack_key:"/private/ack.key".into(),
            local_socket:Some(socket.clone())};
        let installation = temp.path().join("funding-client.json");
        let config = ClientConfig {schema:"parallel.funding_client.v1".into(),
            client_executable:fs::canonicalize("/proc/self/exe").unwrap(),client_executable_sha256:executable.clone(),
            socket:temp.path().join("authority.sock"),authority_root:authority.clone(),witness_path:watermark.clone(),
            trust,witness_policy_sha256:policy.sha256().unwrap()};
        publish(&installation,&serde_json::to_vec(&config).unwrap()).unwrap();
        let execution = f::ExecutionClaim {process,executable_sha256:executable,runtime_sha256:"7".repeat(64),
            request_sha256:sha(&descriptor)};
        Self {_temp:temp,authority,watermark,witness,socket,installation,policy,client,ack_key,
            authorization,reservation,descriptor,execution,delegate}
    }

    fn open(&self) -> Authority {
        let mut authority = Authority::open(&self.authority,&self.watermark).unwrap();
        authority.attach_event_witness(self.client.clone()).unwrap();
        authority.attach_funding(InstalledFunding::for_test(&self.installation).unwrap()).unwrap();
        authority
    }
    fn reserve(&self, authority:&mut Authority) {
        authority.funding_register(&self.authorization, self.reservation.body.demand.deadline_monotonic_ns).unwrap();
        authority.funding_reserve(&self.authorization,&self.reservation, self.reservation.body.demand.deadline_monotonic_ns).unwrap();
        authority.funding_preflight(&self.authorization,&self.reservation, self.reservation.body.demand.deadline_monotonic_ns).unwrap();
    }
    pub(crate) fn use_observed_runtime(&mut self, runtime:&str) {
        self.reservation.body.demand.runtime_sha256=runtime.into();
        self.execution.runtime_sha256=runtime.into();
        self.reservation=f::sign_reservation(self.reservation.body.clone(),&self.delegate).unwrap();
    }
    pub(crate) fn server(&self, count:usize, lose_ack:bool) -> std::thread::JoinHandle<Vec<SignedRequest>> {
        let listener = UnixListener::bind(&self.socket).unwrap();
        listener.set_nonblocking(true).unwrap();
        let root=self.witness.clone();let policy=self.policy.clone();let ack_key=self.ack_key.clone();
        std::thread::spawn(move || {
            let mut requests=Vec::new();
            let stop=std::time::Instant::now()+Duration::from_secs(30);
            for _ in 0..count {
                let mut stream=loop {
                    match listener.accept() {
                        Ok((stream,_))=>break stream,
                        Err(error) if error.kind()==std::io::ErrorKind::WouldBlock=> {
                            assert!(std::time::Instant::now()<stop,"funding witness request did not arrive");
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error)=>panic!("funding fixture accept: {error}"),
                    }
                };
                stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
                stream.set_write_timeout(Some(Duration::from_secs(10))).unwrap();
                let request:SignedRequest=read_frame(&mut stream).unwrap();
                let ack=WitnessStore::open(&root,&policy,&ack_key).unwrap().record_authorization(&request).unwrap();
                if !lose_ack {write_frame(&mut stream,&ack).unwrap();}
                requests.push(request);
            }
            requests
        })
    }
    fn verify_proof(&self, proof:&Value, slot:Option<u32>) {
        let receipt:f::FundingReceipt=serde_json::from_value(proof["receipt"].clone()).unwrap();
        let request:SignedRequest=serde_json::from_value(proof["request"].clone()).unwrap();
        let ack:SignedAck=serde_json::from_value(proof["ack"].clone()).unwrap();
        let event_bytes:Vec<u8>=serde_json::from_value(proof["event_bytes"].clone()).unwrap();
        assert_eq!(receipt.event.participant_slot,slot);
        assert_eq!(receipt.event.reservation,self.reservation);
        assert_eq!(sha(&event_bytes),receipt.event_sha256);
        assert_eq!(f::digest(&request).unwrap(),receipt.request_sha256);
        assert_eq!(f::digest(&ack).unwrap(),receipt.ack_sha256);
        assert_eq!(self.policy.sha256().unwrap(),receipt.policy_sha256);
        assert_eq!(request.body.event,event_bytes);
        ack.verify(&self.policy,&request).unwrap();
        f::verify_event(self.policy.funding_trust.as_ref().unwrap(),&receipt.event,f::FundingClock::current().unwrap()).unwrap();
    }
}

#[test]
fn original_start_and_rank_are_witnessed_retained_and_single_use() {
    let fixture=Fixture::new();let mut authority=fixture.open();fixture.reserve(&mut authority);
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,Some(0),&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    let server=fixture.server(2,false);
    let start=authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,None,&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).unwrap();
    fixture.verify_proof(&start,None);
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,None,&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    let rank=authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,Some(0),&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).unwrap();
    fixture.verify_proof(&rank,Some(0));
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,Some(0),&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,Some(1),&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns)
        .unwrap_err().to_string().contains("participant process already consumed"));
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,Some(2),&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    let originals=server.join().unwrap();assert_eq!(originals.len(),2);
    // Even the real authority signing key cannot ask the independent witness
    // to attest a funding postimage that differs from the signed original.
    let client=WitnessClient::new(fixture.client.clone()).unwrap();
    let mut changed:Value=serde_json::from_slice(&originals[0].body.event).unwrap();
    changed["effects"][0]["row"]["state"]=json!("reserved");
    assert!(client.request(&serde_json::to_vec(&changed).unwrap()).is_err());
    let mut changed:Value=serde_json::from_slice(&originals[1].body.event).unwrap();
    changed["effects"][0]["row"]["slot"]=json!("1");
    assert!(client.request(&serde_json::to_vec(&changed).unwrap()).is_err());
    drop(authority);
    let db=Connection::open(fixture.authority.join("registry.sqlite")).unwrap();
    assert_eq!(db.query_row("SELECT COUNT(*) FROM funding_witness",[],|r|r.get::<_,u64>(0)).unwrap(),2);
    assert_eq!(db.query_row("SELECT COUNT(*) FROM funding_participants",[],|r|r.get::<_,u64>(0)).unwrap(),1);
    drop(db);
    let mut reopened=fixture.open();
    assert!(reopened.funding_preflight(&fixture.authorization,&fixture.reservation, fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    assert!(reopened.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,Some(0),&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    let report=Authority::recovery_inspect(&fixture.authority,&fixture.watermark).unwrap();
    assert_eq!(report["retained_artifact_errors"],json!([]));
}

#[test]
fn lost_ack_preserves_original_witness_and_blocks_replay_or_rank_release() {
    let fixture=Fixture::new();let mut authority=fixture.open();fixture.reserve(&mut authority);
    let server=fixture.server(1,true);
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,None,&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    let original=server.join().unwrap().pop().unwrap();
    let store=WitnessStore::open_read_only(&fixture.witness,&fixture.policy,&fixture.ack_key).unwrap();
    let ack=store.get_ack(&original).unwrap().expect("original ACK survives lost response");
    ack.verify(&fixture.policy,&original).unwrap();
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,None,&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,Some(0),&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    drop(authority);
    assert!(Authority::open(&fixture.authority,&fixture.watermark).is_err());
    let db=Connection::open(fixture.authority.join("registry.sqlite")).unwrap();
    assert_eq!(db.query_row("SELECT COUNT(*) FROM funding_participants",[],|r|r.get::<_,u64>(0)).unwrap(),0);
    assert_eq!(db.query_row("SELECT COUNT(*) FROM funding_reservations",[],|r|r.get::<_,u64>(0)).unwrap(),1);
    assert!(Authority::recovery_replay(&fixture.authority,&fixture.watermark).is_err());
}

#[test]
fn short_admission_deadline_keeps_late_original_ack_ambiguous() {
    let fixture=Fixture::new();let mut authority=fixture.open();fixture.reserve(&mut authority);
    let signed_deadline=fixture.reservation.body.demand.deadline_monotonic_ns;
    let admission_deadline=f::FundingClock::current().unwrap().monotonic_ns+2_000_000_000;
    assert!(admission_deadline<signed_deadline);
    let listener=UnixListener::bind(&fixture.socket).unwrap();listener.set_nonblocking(true).unwrap();
    let root=fixture.witness.clone();let policy=fixture.policy.clone();let key=fixture.ack_key.clone();
    let server=std::thread::spawn(move || {
        let stop=std::time::Instant::now()+Duration::from_secs(5);
        let mut stream=loop {
            match listener.accept() {
                Ok((stream,_))=>break stream,
                Err(error) if error.kind()==std::io::ErrorKind::WouldBlock=> {
                    assert!(std::time::Instant::now()<stop,"bounded original funding request absent");
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error)=>panic!("private witness accept: {error}"),
            }
        };
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let request:SignedRequest=read_frame(&mut stream).unwrap();
        let ack=WitnessStore::open(&root,&policy,&key).unwrap().record_authorization(&request).unwrap();
        assert!(f::FundingClock::current().unwrap().monotonic_ns<admission_deadline,
            "fixture did not retain ACK inside original admission window");
        let now=f::FundingClock::current().unwrap().monotonic_ns;
        std::thread::sleep(Duration::from_nanos(admission_deadline.saturating_sub(now)+100_000_000));
        let _=write_frame(&mut stream,&ack);
        (request,ack)
    });
    let started=std::time::Instant::now();
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,None,
        &fixture.descriptor,admission_deadline).is_err());
    assert!(started.elapsed()<Duration::from_secs(10),"short admission inherited long task deadline");
    let (original,ack)=server.join().unwrap();
    let event:Value=serde_json::from_slice(&original.body.event).unwrap();
    assert_eq!(event["payload"]["admission_deadline_monotonic_ns"],admission_deadline);
    assert_eq!(event["payload"]["original"]["demand"]["deadline_monotonic_ns"],signed_deadline);
    assert!(f::FundingClock::current().unwrap().monotonic_ns<signed_deadline);
    let store=WitnessStore::open_read_only(&fixture.witness,&fixture.policy,&fixture.ack_key).unwrap();
    assert_eq!(store.get_ack(&original).unwrap(),Some(ack));
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,None,
        &fixture.descriptor,signed_deadline).is_err());
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,Some(0),
        &fixture.descriptor,signed_deadline).is_err());
    drop(authority);
    assert!(Authority::open(&fixture.authority,&fixture.watermark).is_err());
    assert!(Authority::recovery_replay(&fixture.authority,&fixture.watermark).is_err());
}

#[test]
fn funding_lock_contention_expires_before_consumption_or_journal() {
    let fixture=Fixture::new();let mut authority=fixture.open();fixture.reserve(&mut authority);
    let before=fs::read(fixture.authority.join("dispatch.jsonl")).unwrap();
    let blocker=OpenOptions::new().read(true).write(true).open(fixture.authority.join("authority.lock")).unwrap();
    assert_eq!(unsafe {libc::flock(blocker.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)},0);
    let deadline=f::FundingClock::current().unwrap().monotonic_ns+500_000_000;
    let started=std::time::Instant::now();
    let result=authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,None,
        &fixture.descriptor,deadline);
    assert_eq!(unsafe {libc::flock(blocker.as_raw_fd(),libc::LOCK_UN)},0);
    drop(blocker);
    assert!(started.elapsed()<Duration::from_secs(2),"short admission used default10-second lock wait");
    assert!(result.unwrap_err().to_string().contains("lock admission deadline elapsed"));
    assert_eq!(fs::read(fixture.authority.join("dispatch.jsonl")).unwrap(),before);
    authority.funding_preflight(&fixture.authorization,&fixture.reservation, fixture.reservation.body.demand.deadline_monotonic_ns).unwrap();
    let db=Connection::open(fixture.authority.join("registry.sqlite")).unwrap();
    assert_eq!(db.query_row("SELECT COUNT(*) FROM funding_witness",[],|r|r.get::<_,u64>(0)).unwrap(),0);
}

#[test]
fn original_deadline_descriptor_and_installed_trust_fail_before_start_journal() {
    let mut fixture=Fixture::new();
    let mut authority=fixture.open();fixture.reserve(&mut authority);
    let before=fs::read(fixture.authority.join("dispatch.jsonl")).unwrap();
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,None,b"{}",fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    let mut foreign=fixture.execution.clone();foreign.process.birth_ticks+=1;
    // Authority-internal inputs are trusted process observations; request
    // transport derives them itself. Here test a clear original runtime clash.
    foreign.runtime_sha256="8".repeat(64);
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&foreign,None,&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    fixture.reservation.body.demand.deadline_unix_ms=f::FundingClock::current().unwrap().unix_ms-1;
    fixture.reservation=f::sign_reservation(fixture.reservation.body.clone(),&fixture.delegate).unwrap();
    assert!(authority.funding_admit(&fixture.authorization,&fixture.reservation,&fixture.execution,None,&fixture.descriptor,fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    let mut changed:Value=serde_json::from_slice(&fs::read(&fixture.installation).unwrap()).unwrap();
    changed["trust"]["authorization_sha256"]=json!("9".repeat(64));
    fs::write(&fixture.installation,serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(authority.funding_register(&fixture.authorization, fixture.reservation.body.demand.deadline_monotonic_ns).is_err());
    assert_eq!(fs::read(fixture.authority.join("dispatch.jsonl")).unwrap(),before);
}

#[test]
fn retained_v4_store_opens_without_migration_and_cannot_attach_funding() {
    let fixture=Fixture::new();
    let path=fixture.authority.join("authority.json");
    let mut manifest:Value=serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["schema"]=json!("parallel.fit_authority.v4");
    // Keep the sealed original v5 bytes. Construct a separate private v4
    // fixture inode; production authority immutability remains unchanged.
    fs::rename(&path,path.with_extension("v5-fixture.json")).unwrap();
    fs::write(&path,serde_json::to_vec(&manifest).unwrap()).unwrap();
    let db=Connection::open(fixture.authority.join("registry.sqlite")).unwrap();
    db.execute_batch("DROP TABLE funding_authorizations; DROP TABLE funding_reservations; DROP TABLE funding_participants; DROP TABLE funding_witness; PRAGMA user_version=4;").unwrap();
    drop(db);
    let before=fs::read(fixture.authority.join("dispatch.jsonl")).unwrap();
    let mut authority=Authority::open(&fixture.authority,&fixture.watermark).unwrap();
    assert!(authority.attach_funding(InstalledFunding::for_test(&fixture.installation).unwrap()).unwrap_err().to_string().contains("v5 authority"));
    drop(authority);
    assert_eq!(fs::read(fixture.authority.join("dispatch.jsonl")).unwrap(),before);
    let db=Connection::open(fixture.authority.join("registry.sqlite")).unwrap();
    assert_eq!(db.query_row("PRAGMA user_version",[],|r|r.get::<_,u32>(0)).unwrap(),4);
    assert_eq!(db.query_row("SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'funding_%'",[],|r|r.get::<_,u64>(0)).unwrap(),0);
}

#[test]
fn metadata_funding_operations_obey_transport_deadline_without_journaling() {
    let fixture = Fixture::new();
    let mut authority = fixture.open();
    fixture.reserve(&mut authority);
    let before = fs::read(fixture.authority.join("dispatch.jsonl")).unwrap();
    let blocker = OpenOptions::new().read(true).write(true)
        .open(fixture.authority.join("authority.lock")).unwrap();
    assert_eq!(unsafe { libc::flock(blocker.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, 0);
    for operation in 0..3 {
        let deadline = f::FundingClock::current().unwrap().monotonic_ns + 100_000_000;
        let started = std::time::Instant::now();
        let result = match operation {
            0 => authority.funding_register(&fixture.authorization, deadline).map(|_| ()),
            1 => authority.funding_reserve(&fixture.authorization, &fixture.reservation, deadline).map(|_| ()),
            _ => authority.funding_preflight(&fixture.authorization, &fixture.reservation, deadline),
        };
        assert!(started.elapsed() < Duration::from_secs(2), "metadata operation exceeded transport lock window");
        assert!(result.unwrap_err().to_string().contains("lock admission deadline elapsed"));
        assert_eq!(fs::read(fixture.authority.join("dispatch.jsonl")).unwrap(), before);
    }
    assert_eq!(unsafe { libc::flock(blocker.as_raw_fd(), libc::LOCK_UN) }, 0);
    drop(blocker);
    // Expired uncontended requests must fail too; releasing contention must not
    // revive the old request. A distinguishable new request may still preflight.
    let expired = f::FundingClock::current().unwrap().monotonic_ns - 1;
    assert!(authority.funding_register(&fixture.authorization, expired).is_err());
    assert!(authority.funding_reserve(&fixture.authorization, &fixture.reservation, expired).is_err());
    assert!(authority.funding_preflight(&fixture.authorization, &fixture.reservation, expired).is_err());
    assert_eq!(fs::read(fixture.authority.join("dispatch.jsonl")).unwrap(), before);
    authority.funding_preflight(&fixture.authorization, &fixture.reservation,
        fixture.reservation.body.demand.deadline_monotonic_ns).unwrap();
}
