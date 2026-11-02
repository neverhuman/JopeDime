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
            crate::eval_registry::FitInputs {
                x: vec![vec![1.]],
                y: vec![2.],
                preprocessing: b"pp".to_vec(),
                feature_view: b"view".to_vec(),
            },
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
mod authorization;
mod public_state;
mod ssh;
mod wire;
