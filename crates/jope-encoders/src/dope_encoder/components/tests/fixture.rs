//! Synthetic controller fixture only. These fixed vectors make no learned-model or issuer claim.
use super::super::*;
use crate::dope_encoder::{prepare_table_with_ids, Head, PrepTask};
use jope_records::replay_admission::{
    AdmissionPolicy, ComponentAuthoritySnapshot, ExposureReceipt, Observation, PolicyPin, Purpose,
    ADOPTION_PATH, ADOPTION_SCHEMA, COMPONENT_AUTHORITY_PATH, COMPONENT_AUTHORITY_SCHEMA,
    EXPOSURE_SCHEMA, POLICY_SCHEMA,
};
use jope_records::wire_contract::CanonicalPayload;
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const EPISODE: &str = "fixture-episode-a";
pub(super) const UNTIL: u64 = 4_102_444_800_000; // 2100, within I02's exact integer range.
static NEXT: AtomicU64 = AtomicU64::new(0);

pub(super) struct Fixture {
    pub root: PathBuf,
    pub request: PathBuf,
    pub scope: FitScope,
    pub artifact: FusionComponentArtifact,
    pub subject: String,
    pub proof: String,
    pub policy: String,
    pub foundation: String,
    receipts: Vec<String>,
    parents: Vec<String>,
}
pub(super) fn canonical<T: Serialize>(value: &T) -> Vec<u8> {
    CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(value).unwrap())
        .unwrap()
        .as_bytes()
        .to_vec()
}
pub(super) fn float_hex(value: f32, count: usize) -> String {
    value
        .to_le_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        .repeat(count)
}
fn receipt(subject: &str, parents: Vec<String>) -> ExposureReceipt {
    ExposureReceipt {
        schema: EXPOSURE_SCHEMA.into(),
        subject_sha256: subject.into(),
        family: "fixture-family".into(),
        meta_partition: "train_search".into(),
        original_role: "train_fit".into(),
        within_task_view: "training".into(),
        fidelity: "synthetic-unit-fixture".into(),
        producer: "fixture-controller".into(),
        policy_epoch: 1,
        observed_at_ms: 100,
        episode: Some(EPISODE.into()),
        purposes: BTreeSet::from([Purpose::EpisodeFit]),
        prior_exposure: Some(vec!["train_fit".into()]),
        parents,
        observation: Observation::Proposed,
    }
}
impl Fixture {
    pub fn new() -> Self {
        Self::with_ancestor(|_| {})
    }
    pub fn with_ancestor(change: impl FnOnce(&mut ExposureReceipt)) -> Self {
        // Evidence stays in the lane target. No ambient State/environment or default temp path.
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/task-evidence/P4/components")
            .join(format!(
                "{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(root.join("blobs/sha256")).unwrap();
        fs::create_dir_all(root.join("replay")).unwrap();
        let x = (0..8)
            .map(|i| vec![i as f64 / 8., (i % 3) as f64])
            .collect::<Vec<_>>();
        let y = (0..8).map(|i| (i * i) as f64 / 16.).collect::<Vec<_>>();
        let ids = vec![701, 5, 109, 42, 8, 9001, 63, 2];
        let mut input = TrainingInput::numeric(x, y, ids, Head::Regression);
        input.fold = Some("fixture-inner-fold-0".into());
        let prepared =
            prepare_table_with_ids(&input.x, &input.y, PrepTask::Regression, input.ids.clone())
                .unwrap();
        let scope = FitScope::from_prepared(&input, &prepared).unwrap();
        let artifact = FusionComponentArtifact {
            schema: "jope.dope-fusion-components/v1".into(),
            fit_scope: scope.clone(),
            episode: EPISODE.into(),
            permitted_use: "episode_fit".into(),
            recipe: FUSION_COMPONENT_RECIPE.into(),
            producer_inputs: BTreeMap::new(),
            foundation1024_le_f32: float_hex(1.25, 1024),
            target512_le_f32: float_hex(-2.5, 512),
        };
        let mut out = Self {
            request: root.join("component-requests.json"),
            root,
            scope,
            artifact,
            subject: String::new(),
            proof: String::new(),
            policy: String::new(),
            foundation: String::new(),
            receipts: vec![],
            parents: vec![],
        };
        // FIT evidence retains ordered physical IDs and exact input/target bit patterns.
        let fit = out.put(&canonical(&json!({
            "schema": "jope.fixture-fit-input/v1", "physical_rows": input.ids, "fold": input.fold,
            "x_f64_bits": input.x.iter().map(|row| row.iter().map(|v| format!("{:016x}", v.to_bits())).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "y_f64_bits": input.y.iter().map(|v| format!("{:016x}", v.to_bits())).collect::<Vec<_>>()
        })));
        let mut fit_receipt = receipt(&fit, vec![]);
        change(&mut fit_receipt);
        let fit_proof = out.retain(&fit_receipt);
        out.foundation = out.put(b"synthetic foundation producer; no real model weights");
        let target = out.put(b"synthetic target producer; no real model weights");
        let foundation_proof = out.retain(&receipt(&out.foundation, vec![fit_proof.clone()]));
        let target_proof = out.retain(&receipt(&target, vec![fit_proof.clone()]));
        out.receipts = vec![fit_proof, foundation_proof.clone(), target_proof.clone()];
        out.parents = vec![foundation_proof, target_proof];
        out.artifact.producer_inputs = BTreeMap::from([
            ("foundation".into(), out.foundation.clone()),
            ("target".into(), target),
        ]);
        out.publish();
        out
    }
    pub fn blob(&self, digest: &str) -> PathBuf {
        self.root.join("blobs/sha256").join(digest)
    }
    pub fn put(&self, bytes: &[u8]) -> String {
        let digest = sha256(bytes);
        fs::write(self.blob(&digest), bytes).unwrap();
        digest
    }
    pub fn retain<T: Serialize>(&self, value: &T) -> String {
        self.put(&canonical(value))
    }
    pub fn publish(&mut self) {
        self.subject = self.retain(&self.artifact);
        self.proof = self.retain(&receipt(&self.subject, self.parents.clone()));
        let mut proofs: BTreeSet<_> = self.receipts.iter().cloned().collect();
        proofs.insert(self.proof.clone());
        let policy = AdmissionPolicy {
            schema: POLICY_SCHEMA.into(),
            issuer: "fixture-controller".into(),
            epoch: 1,
            effective_from_ms: 0,
            effective_until_ms: UNTIL,
            evaluation_boundary_sha256: self.put(b"separate synthetic future evaluation boundary"),
            purposes: BTreeSet::from([Purpose::EpisodeFit]),
            receipts: proofs,
            supersedes: None,
        };
        self.policy = self.retain(&policy);
        fs::write(
            self.root.join(ADOPTION_PATH),
            canonical(&PolicyPin {
                schema: ADOPTION_SCHEMA.into(),
                policy_sha256: self.policy.clone(),
                issuer: "fixture-controller".into(),
                epoch: 1,
            }),
        )
        .unwrap();
        self.install_authority(BTreeSet::new());
        self.write_request(&self.scope, EPISODE);
    }
    pub fn install_authority(&self, revoked_digests: BTreeSet<String>) {
        // Test controller installs only this isolated fixture tree. This is not a live export
        // or evidence that a production operator has authorized any issuer or component recipe.
        let snapshot = ComponentAuthoritySnapshot {
            schema: COMPONENT_AUTHORITY_SCHEMA.into(),
            adoption_sha256: sha256(&fs::read(self.root.join(ADOPTION_PATH)).unwrap()),
            revocation_files: BTreeMap::new(),
            revoked_digests,
        };
        fs::write(
            self.root.join(COMPONENT_AUTHORITY_PATH),
            canonical(&snapshot),
        )
        .unwrap();
    }
    pub fn write_request(&self, scope: &FitScope, episode: &str) {
        fs::write(
            &self.request,
            serde_json::to_vec(&json!({
                "schema": "jope.fusion-component-requests/v1", "entries": [{
                    "fit_scope_digest": scope.digest().unwrap(), "subject_sha256": self.subject,
                    "exposure_receipt_sha256": self.proof, "episode": episode
                }]
            }))
            .unwrap(),
        )
        .unwrap();
    }
    pub fn load(&self) -> Result<ComponentInputs> {
        ComponentInputs::load_with_controller(&self.request, &self.root)
    }
}
