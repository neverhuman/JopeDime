//! Private signed funding fixtures exercise the real authority journal and
//! framed witness transport. No fit, CUDA context or production trust is used.
use super::*;
use crate::eval_registry::{
    funding as f,
    funding_config::{ClientConfig, InstalledFunding},
    Authority, ProcessIdentity, ScopePolicy,
};
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
        Self::build(false)
    }

    pub(crate) fn new_resident_training() -> Self {
        Self::build(true)
    }

    fn build(resident_training: bool) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let authority = temp.path().join("authority");
        let watermark = temp.path().join("local-watermark");
        Authority::initialize(
            &authority,
            &watermark,
            vec![ScopePolicy {
                scope: "private-no-numerics".into(),
                training_sha256: vec!["1".repeat(64)],
                numerical_contract_sha256: vec!["2".repeat(64)],
                fit_seeds: vec![0],
            }],
        )
        .unwrap();
        let manifest_bytes = fs::read(authority.join("authority.json")).unwrap();
        let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
        let instance = manifest["instance"].as_str().unwrap().to_string();
        let process = ProcessIdentity::current().unwrap();
        let executable = sha(&fs::read("/proc/self/exe").unwrap());
        let clock = f::FundingClock::current().unwrap();
        let authorizer = SigningKey::from_bytes(&[11; 32]);
        let delegate = SigningKey::from_bytes(&[37; 32]);
        let delegated = hex::encode(delegate.verifying_key().to_bytes());
        let descriptor = if resident_training {
            serde_json::to_vec(&json!({
                "schema": "parallel.resident_common_task.v1",
                "root": "/private-fixture",
                "task": {"kind": "Train", "world": 2},
                "environment": {},
                "input_artifacts": {},
                "executable_sha256": executable,
                "deadline_unix_ms": clock.unix_ms + 120_000,
            }))
            .unwrap()
        } else {
            serde_json::to_vec(&json!({"schema":"private.funding_task.v1",
                "task":"metadata-only witness fixture"}))
            .unwrap()
        };
        let body = f::FundingAuthorization {
            schema: "parallel.funding_authorization.v1".into(),
            domain_id: "private-witness-fixture".into(),
            scope: "private signed fixture; never a production grant".into(),
            period_start_unix_ms: clock.unix_ms - 1_000,
            period_end_unix_ms: clock.unix_ms + 300_000,
            authority_evidence_sha256: "3".repeat(64),
            historical_reconciliation_sha256: "4".repeat(64),
            release_prerequisites_sha256: "5".repeat(64),
            approved_task_sha256: BTreeSet::from([sha(&descriptor)]),
            historical_core_ms: 500,
            historical_node_ms: 200,
            core_ms: 1_000_000,
            node_ms: 500_000,
            max_concurrent_cpu_searches: 16,
            allocation: f::FundingAllocation {
                cpu_cores: 16,
                gpu_devices: (0..8).collect(),
                host_memory_bytes: 16_000,
                gpu_memory_bytes_per_device: 8_000,
            },
            capabilities: BTreeSet::from(["regression_training".into()]),
            reservation_public_key: delegated.clone(),
        };
        let mut signing_bytes = b"parallel.funding-authorization.signature.v1\0".to_vec();
        signing_bytes.extend(serde_json::to_vec(&body).unwrap());
        let authorization = f::SignedAuthorization {
            signature: hex::encode(authorizer.sign(&signing_bytes).to_bytes()),
            body,
        };
        let trust = f::FundingTrust {
            schema: "parallel.funding_trust.v1".into(),
            domain_id: authorization.body.domain_id.clone(),
            authority_instance: instance.clone(),
            authority_epoch: instance.clone(),
            authority_manifest_sha256: sha(&manifest_bytes),
            authority_executable_sha256: executable.clone(),
            authorization_public_key: hex::encode(authorizer.verifying_key().to_bytes()),
            authorization_sha256: f::digest(&authorization.body).unwrap(),
            reservation_public_key: delegated,
            host_id: fs::read_to_string("/proc/sys/kernel/hostname")
                .unwrap()
                .trim()
                .into(),
            boot_id: process.boot_id.clone(),
        };
        let reservation = f::sign_reservation(
            f::TaskReservation {
                schema: "parallel.task_reservation.v1".into(),
                reservation_id: "6".repeat(64),
                authorization_sha256: trust.authorization_sha256.clone(),
                authority_instance: instance.clone(),
                authority_epoch: instance.clone(),
                authority_manifest_sha256: trust.authority_manifest_sha256.clone(),
                authority_executable_sha256: executable.clone(),
                domain_id: trust.domain_id.clone(),
                not_before_unix_ms: clock.unix_ms - 1_000,
                demand: f::TaskDemand {
                    task_sha256: sha(&descriptor),
                    capability: "regression_training".into(),
                    host_id: trust.host_id.clone(),
                    boot_id: process.boot_id.clone(),
                    executable_sha256: executable.clone(),
                    runtime_sha256: "7".repeat(64),
                    deadline_unix_ms: clock.unix_ms + 120_000,
                    deadline_monotonic_ns: clock.monotonic_ns + 120_000_000_000,
                    resources: f::ResourceCaps {
                        cpu_cores: 2,
                        gpus: 2,
                        world: 2,
                        host_memory_bytes: 4_000,
                        gpu_memory_bytes_per_rank: 2_000,
                        cpu_searches: 0,
                        gpu_devices: vec![0, 1],
                    },
                },
                core_ms: 242_000,
                node_ms: 121_000,
                participant_slots: vec![0, 1],
            },
            &delegate,
        )
        .unwrap();
        let authority_key = temp.path().join("authority.key");
        let ack_key = temp.path().join("independent-ack.key");
        let known_hosts = temp.path().join("known_hosts");
        publish(&authority_key, &[17; 32]).unwrap();
        publish(&ack_key, &[29; 32]).unwrap();
        publish(&known_hosts, b"private framed local fixture; no SSH\n").unwrap();
        let policy = WitnessPolicy {
            schema: "parallel.fit_event_witness_policy.v1".into(),
            witness_instance: sha(b"private-funding-witness"),
            authority_instance: instance.clone(),
            authority_epoch: instance,
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
            witness_executable_sha256: executable.clone(),
            witness_source_sha256: source_sha256(),
            ssh_known_hosts_sha256: sha(&read(&known_hosts).unwrap()),
            ssh_host_alias: "private".into(),
            ssh_destination: "private".into(),
            native_fit_trust: None,
            native_grant_delegation: None,
            funding_trust: Some(trust.clone()),
        };
        let witness = temp.path().join("independent-witness");
        WitnessStore::initialize(&witness, &policy).unwrap();
        let socket = temp.path().join("witness.sock");
        let client = WitnessClientConfig {
            epoch_proofs: Vec::new(),
            ssh_identity: None,
            policy: policy.clone(),
            authority_key,
            known_hosts,
            remote_executable: "/private/jope-parallel".into(),
            remote_root: "/private/witness".into(),
            remote_policy: "/private/policy.json".into(),
            remote_ack_key: "/private/ack.key".into(),
            local_socket: Some(socket.clone()),
        };
        let installation = temp.path().join("funding-client.json");
        let config = ClientConfig {
            schema: "parallel.funding_client.v1".into(),
            client_executable: fs::canonicalize("/proc/self/exe").unwrap(),
            client_executable_sha256: executable.clone(),
            socket: temp.path().join("authority.sock"),
            authority_root: authority.clone(),
            witness_path: watermark.clone(),
            trust,
            witness_policy_sha256: policy.sha256().unwrap(),
        };
        publish(&installation, &serde_json::to_vec(&config).unwrap()).unwrap();
        let execution = f::ExecutionClaim {
            process,
            executable_sha256: executable,
            runtime_sha256: "7".repeat(64),
            request_sha256: sha(&descriptor),
        };
        Self {
            _temp: temp,
            authority,
            watermark,
            witness,
            socket,
            installation,
            policy,
            client,
            ack_key,
            authorization,
            reservation,
            descriptor,
            execution,
            delegate,
        }
    }

    fn open(&self) -> Authority {
        let mut authority = Authority::open(&self.authority, &self.watermark).unwrap();
        authority.attach_event_witness(self.client.clone()).unwrap();
        authority
            .attach_funding(InstalledFunding::for_test(&self.installation).unwrap())
            .unwrap();
        authority
    }
    fn reserve(&self, authority: &mut Authority) {
        authority
            .funding_register(
                &self.authorization,
                self.reservation.body.demand.deadline_monotonic_ns,
            )
            .unwrap();
        authority
            .funding_reserve(
                &self.authorization,
                &self.reservation,
                self.reservation.body.demand.deadline_monotonic_ns,
            )
            .unwrap();
        authority
            .funding_preflight(
                &self.authorization,
                &self.reservation,
                self.reservation.body.demand.deadline_monotonic_ns,
            )
            .unwrap();
    }
    pub(crate) fn use_observed_runtime(&mut self, runtime: &str) {
        self.reservation.body.demand.runtime_sha256 = runtime.into();
        self.execution.runtime_sha256 = runtime.into();
        self.reservation =
            f::sign_reservation(self.reservation.body.clone(), &self.delegate).unwrap();
    }
    pub(crate) fn server(
        &self,
        count: usize,
        lose_ack: bool,
    ) -> std::thread::JoinHandle<Vec<SignedRequest>> {
        let listener = UnixListener::bind(&self.socket).unwrap();
        listener.set_nonblocking(true).unwrap();
        let root = self.witness.clone();
        let policy = self.policy.clone();
        let ack_key = self.ack_key.clone();
        std::thread::spawn(move || {
            let mut requests = Vec::new();
            let stop = std::time::Instant::now() + Duration::from_secs(30);
            for _ in 0..count {
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(
                                std::time::Instant::now() < stop,
                                "funding witness request did not arrive"
                            );
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("funding fixture accept: {error}"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                let request: SignedRequest = read_frame(&mut stream).unwrap();
                let ack = WitnessStore::open(&root, &policy, &ack_key)
                    .unwrap()
                    .record_authorization(&request)
                    .unwrap();
                if !lose_ack {
                    write_frame(&mut stream, &ack).unwrap();
                }
                requests.push(request);
            }
            requests
        })
    }
    fn verify_proof(&self, proof: &Value, slot: Option<u32>) {
        let receipt: f::FundingReceipt = serde_json::from_value(proof["receipt"].clone()).unwrap();
        let request: SignedRequest = serde_json::from_value(proof["request"].clone()).unwrap();
        let ack: SignedAck = serde_json::from_value(proof["ack"].clone()).unwrap();
        let event_bytes: Vec<u8> = serde_json::from_value(proof["event_bytes"].clone()).unwrap();
        assert_eq!(receipt.event.participant_slot, slot);
        assert_eq!(receipt.event.reservation, self.reservation);
        assert_eq!(sha(&event_bytes), receipt.event_sha256);
        assert_eq!(f::digest(&request).unwrap(), receipt.request_sha256);
        assert_eq!(f::digest(&ack).unwrap(), receipt.ack_sha256);
        assert_eq!(self.policy.sha256().unwrap(), receipt.policy_sha256);
        assert_eq!(request.body.event, event_bytes);
        ack.verify(&self.policy, &request).unwrap();
        f::verify_event(
            self.policy.funding_trust.as_ref().unwrap(),
            &receipt.event,
            f::FundingClock::current().unwrap(),
        )
        .unwrap();
    }
}

mod admission;
mod deadlines;
