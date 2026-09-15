//! Append-only native grant delegation. The immutable base policy and original
//! witness journal remain the authority; adding a grant never rotates a ledger.
use super::*;

pub const ADD_NATIVE_GRANT_DOMAIN: &str = "parallel.add_native_grant.request.v1";
const ACK_DOMAIN: &str = "parallel.add_native_grant.ack.v1";
const MAX_UPDATE_BYTES: usize = 1024 * 1024;
const MAX_UPDATES: u64 = 100_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeGrantDelegation {
    pub schema: String,
    pub provisioner_public_key: String,
    pub run: String,
    pub authority_instance: String,
    pub authority_epoch: String,
    pub initial_optimizer_step: u64,
    pub cycles: u64,
    pub scientific_policy_sha256: String,
    pub controller_sha256: String,
    pub manifest_root: PathBuf,
    pub deadline_unix_ms: u64,
    pub total_evaluation_budget: u64,
    pub grant_template: NativeGrant,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativeGrantRole {
    Direct,
    Continuation,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeGrantArtifact {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddNativeGrantBody {
    pub schema: String,
    pub base_policy_sha256: String,
    pub delegation_sha256: String,
    pub sequence: u64,
    /// SHA256 of the preceding complete SignedAddNativeGrant, or all zeros.
    pub previous_update_sha256: String,
    pub signed_grant: SignedNativeGrant,
    pub grant_sha256: String,
    pub loop_request: NativeGrantArtifact,
    pub graph_admission: NativeGrantArtifact,
    pub task_manifest: NativeGrantArtifact,
    pub run: String,
    pub generation: u64,
    pub role: NativeGrantRole,
    pub evaluations: u32,
    pub issued_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedAddNativeGrant {
    pub body: AddNativeGrantBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AddNativeGrantAckBody {
    pub schema: String,
    pub base_policy_sha256: String,
    pub witness_instance: String,
    pub authority_instance: String,
    pub authority_epoch: String,
    pub request_sha256: String,
    pub grant_sha256: String,
    pub update_sequence: u64,
    pub previous_update_sha256: String,
    pub witness_sequence: u64,
    pub previous_record_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignedAddNativeGrantAck {
    pub body: AddNativeGrantAckBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeGrantProof {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub epoch_proofs: Vec<WitnessEpochProof>,
    pub request: SignedAddNativeGrant,
    pub ack: SignedAddNativeGrantAck,
}

fn bounded_text(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}
fn lexical_absolute(path: &Path) -> bool {
    path.is_absolute()
        && path.as_os_str().len() <= 4096
        && !path.components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
}
impl NativeGrantDelegation {
    pub fn sha256(&self) -> Result<String> {
        Ok(sha(&bytes(self)?))
    }
    pub(super) fn validate(&self, policy: &WitnessPolicy) -> Result<()> {
        ensure!(
            self.schema == "parallel.native_grant_delegation.v1"
                && bounded_text(&self.run)
                && self.initial_optimizer_step == 60000
                && self.cycles == 2
                && (1..=100_000_000).contains(&self.total_evaluation_budget)
                && self.authority_instance == policy.authority_instance
                && self.authority_epoch == policy.authority_epoch
                && valid_hash(&self.scientific_policy_sha256)
                && valid_hash(&self.controller_sha256)
                && lexical_absolute(&self.manifest_root),
            "invalid finite native grant delegation"
        );
        key(&self.provisioner_public_key)?;
        ensure!(
            self.provisioner_public_key != policy.authority_public_key
                && self.provisioner_public_key != policy.witness_public_key,
            "provisioner must be independent of authority and witness keys"
        );
        let template = &self.grant_template;
        template.validate(template.period_start_unix_ms)?;
        ensure!(
            template.authority_instance == self.authority_instance
                && template.authority_epoch == self.authority_epoch
                && template.authority_executable_sha256 == policy.authority_executable_sha256
                && template.witness_public_key == policy.witness_public_key
                && self.manifest_root.starts_with(&template.root)
                && template.period_end_unix_ms <= self.deadline_unix_ms
                && template.period_start_unix_ms < self.deadline_unix_ms,
            "delegation template differs from immutable witness identity/window"
        );
        Ok(())
    }
}
pub(super) fn validate_update(
    policy: &WitnessPolicy,
    request: &SignedAddNativeGrant,
) -> Result<NativeWitnessTrust> {
    policy.validate()?;
    validate_update_effective(policy, request)
}
pub(super) fn validate_update_effective(
    policy: &WitnessPolicy,
    request: &SignedAddNativeGrant,
) -> Result<NativeWitnessTrust> {
    epochs::validate_effective_shape(policy)?;
    ensure!(
        bytes(request)?.len() <= MAX_UPDATE_BYTES,
        "native grant update exceeds byte bound"
    );
    let delegation = policy
        .native_grant_delegation
        .as_ref()
        .context("native grant delegation unavailable")?;

    let b = &request.body;
    ensure!(
        b.schema == "parallel.add_native_grant.v1"
            && b.base_policy_sha256 == policy.sha256()?
            && b.delegation_sha256 == delegation.sha256()?
            && (1..=MAX_UPDATES).contains(&b.sequence)
            && valid_hash(&b.previous_update_sha256)
            && b.run == delegation.run
            && b.generation < delegation.cycles
            && (1..=1000).contains(&b.evaluations)
            && b.issued_unix_ms < delegation.deadline_unix_ms,
        "native grant update exceeds delegated identity/window"
    );
    for artifact in [&b.loop_request, &b.graph_admission, &b.task_manifest] {
        ensure!(
            valid_hash(&artifact.sha256)
                && lexical_absolute(&artifact.path)
                && artifact.path != delegation.manifest_root
                && artifact.path.starts_with(&delegation.manifest_root),
            "native grant artifact outside original manifest root"
        );
    }
    ensure!(
        b.loop_request.path != b.graph_admission.path
            && b.loop_request.path != b.task_manifest.path
            && b.graph_admission.path != b.task_manifest.path,
        "native grant artifact roles alias"
    );
    verify(
        &delegation.provisioner_public_key,
        ADD_NATIVE_GRANT_DOMAIN,
        b,
        &request.signature,
    )?;
    let grant = &b.signed_grant.body;
    let trust = NativeWitnessTrust {
        authorization_public_key: delegation.provisioner_public_key.clone(),
        grant_sha256: b.grant_sha256.clone(),
    };
    super::super::native_scq::policy::verify_signed(&trust, &b.signed_grant, b.issued_unix_ms)?;
    ensure!(
        grant.approved_task_sha256
            == std::collections::BTreeSet::from([b.task_manifest.sha256.clone()])
            && grant.max_candidates_per_task == b.evaluations
            && grant.period_start_unix_ms >= delegation.grant_template.period_start_unix_ms
            && grant.period_end_unix_ms <= delegation.grant_template.period_end_unix_ms
            && grant.max_candidates_per_task <= delegation.grant_template.max_candidates_per_task,
        "native grant task/budget/window differs from frozen update"
    );
    let mut normalized = grant.clone();
    normalized.approved_task_sha256 = delegation.grant_template.approved_task_sha256.clone();
    normalized.decision_receipt_sha256 = delegation.grant_template.decision_receipt_sha256.clone();
    normalized.period_start_unix_ms = delegation.grant_template.period_start_unix_ms;
    normalized.period_end_unix_ms = delegation.grant_template.period_end_unix_ms;
    normalized.max_candidates_per_task = delegation.grant_template.max_candidates_per_task;
    ensure!(
        normalized == delegation.grant_template,
        "native grant broadens immutable template"
    );
    Ok(trust)
}
/// Validates the complete original delegated update and witness ACK offline.
/// Expiration is checked at the ORIGINAL update time; issuing new work still
/// checks its own authorization time and the original grant window.
pub fn verify_native_grant_proof(
    policy: &WitnessPolicy,
    proof: &NativeGrantProof,
) -> Result<NativeWitnessTrust> {
    let effective = verify_epoch_chain(policy, &proof.epoch_proofs)?;
    let policy = &effective;
    let trust = validate_update_effective(policy, &proof.request)?;
    let b = &proof.ack.body;
    ensure!(
        b.schema == "parallel.add_native_grant_ack.v1"
            && b.base_policy_sha256 == policy.sha256()?
            && b.witness_instance == policy.witness_instance
            && b.authority_instance == policy.authority_instance
            && b.authority_epoch == policy.authority_epoch
            && b.request_sha256 == sha(&bytes(&proof.request)?)
            && b.grant_sha256 == proof.request.body.grant_sha256
            && b.update_sequence == proof.request.body.sequence
            && b.previous_update_sha256 == proof.request.body.previous_update_sha256
            && b.witness_sequence > 0
            && valid_hash(&b.previous_record_sha256),
        "native grant ACK differs from original update/store"
    );
    verify(
        &policy.witness_public_key,
        ACK_DOMAIN,
        b,
        &proof.ack.signature,
    )?;
    Ok(trust)
}

pub(super) type NumericalWork = Option<(String, Option<(String, u64)>)>;
#[derive(Default)]
pub(super) struct NativeGrantState {
    updates: BTreeMap<u64, (NativeGrantProof, String)>,
    pub(super) reserved_evaluations: u64,
    numerical_work: BTreeMap<String, u64>,
    candidate_slots: BTreeMap<(String, u64), u64>,
}
impl NativeGrantState {
    pub(super) fn update_cursor(&self) -> Result<(u64, String)> {
        match self.updates.last_key_value() {
            Some((n, (p, _))) => Ok((
                n.checked_add(1).context("grant sequence overflow")?,
                sha(&bytes(&p.request)?),
            )),
            None => Ok((1, ZERO.into())),
        }
    }

    fn check_update(&self, policy: &WitnessPolicy, request: &SignedAddNativeGrant) -> Result<u64> {
        validate_update_effective(policy, request)?;
        let (sequence, previous) = match self.updates.last_key_value() {
            Some((seq, (proof, _))) => (*seq + 1, sha(&bytes(&proof.request)?)),
            None => (1, ZERO.into()),
        };
        ensure!(
            request.body.sequence == sequence && request.body.previous_update_sha256 == previous,
            "stale/conflicting native grant update chain"
        );
        ensure!(
            !self
                .updates
                .values()
                .any(|(p, _)| p.request.body.task_manifest.sha256
                    == request.body.task_manifest.sha256
                    || p.request.body.loop_request == request.body.loop_request
                    || p.request.body.graph_admission == request.body.graph_admission),
            "native task/request/admission already granted"
        );
        let reserved = self
            .reserved_evaluations
            .checked_add(u64::from(request.body.evaluations))
            .context("native delegated evaluation budget overflow")?;
        ensure!(
            reserved
                <= policy
                    .native_grant_delegation
                    .as_ref()
                    .context("delegation absent")?
                    .total_evaluation_budget,
            "native delegated evaluation budget exhausted"
        );
        Ok(reserved)
    }
    fn proof_in_journal(
        &self,
        policy: &WitnessPolicy,
        proof: &NativeGrantProof,
        before: u64,
    ) -> Result<()> {
        verify_native_grant_proof(policy, proof)?;
        let (original, _) = self
            .updates
            .get(&proof.request.body.sequence)
            .context("grant update not durably witnessed")?;
        ensure!(
            bytes(original)? == bytes(proof)? && proof.ack.body.witness_sequence <= before,
            "authorization proof differs from preceding original grant update"
        );
        Ok(())
    }
    pub(super) fn check_work(
        &self,
        policy: &WitnessPolicy,
        request: &SignedRequest,
        before: u64,
    ) -> Result<NumericalWork> {
        let event: OriginalEvent = serde_json::from_slice(&request.body.event)?;
        if event.kind != "authorize" {
            return Ok(None);
        }
        let identity: super::super::FitIdentity = serde_json::from_str(
            event.effects[0].row["identity_json"]
                .as_str()
                .context("fit identity missing")?,
        )?;
        // Exactly the authority global numerical guard: executable/runtime/epoch
        // are deliberately absent, so new grants cannot repeat old mathematics.
        let work = sha(&bytes(&serde_json::json!({
            "schema":"parallel.numerical_work_guard.v1", "program":identity.canonical_program,
            "training":identity.training_sha256,"preprocessing":identity.preprocessing_sha256,
            "feature_view":identity.feature_view_sha256,"fit_seed":identity.fit_seed,
            "numerical_contract":identity.numerical_contract_sha256
        }))?);
        ensure!(
            !self.numerical_work.contains_key(&work),
            "numerical work already witnessed; grant changes cannot authorize replay"
        );
        let effective = verify_epoch_chain(policy, &request.body.epoch_proofs)?;
        let native = event.payload.get("native_fit");
        if effective.native_grant_delegation.is_some() {
            ensure!(
                native.is_some_and(|v| !v.is_null()),
                "delegated witness requires native authorization proof"
            );
        }
        let slot = if let Some(native) = native {
            if let Some(value) = native.get("native_grant_proof").filter(|v| !v.is_null()) {
                let proof: NativeGrantProof = serde_json::from_value(value.clone())?;
                self.proof_in_journal(policy, &proof, before)?;
                ensure!(
                    native["signed_grant"]
                        == serde_json::to_value(&proof.request.body.signed_grant)?,
                    "authorization grant differs from witnessed update"
                );
            }
            let slot = (
                native["task_manifest_sha256"]
                    .as_str()
                    .context("native task missing")?
                    .to_owned(),
                native["candidate_index"]
                    .as_u64()
                    .context("native candidate missing")?,
            );
            ensure!(
                !self.candidate_slots.contains_key(&slot),
                "native candidate slot already witnessed"
            );
            Some(slot)
        } else {
            None
        };
        Ok(Some((work, slot)))
    }
    pub(super) fn commit_work(&mut self, work: NumericalWork, authority_sequence: u64) {
        if let Some((work, slot)) = work {
            self.numerical_work.insert(work, authority_sequence);
            if let Some(slot) = slot {
                self.candidate_slots.insert(slot, authority_sequence);
            }
        }
    }
}
impl WitnessStore {
    pub(super) fn replay_native_grant(
        &mut self,
        proof: NativeGrantProof,
        digest: String,
    ) -> Result<()> {
        verify_native_grant_proof(&self.policy, &proof)?;
        ensure!(
            bytes(&proof.epoch_proofs)? == bytes(&self.epoch_proofs)?,
            "grant epoch not active at original journal position"
        );
        let reserved = self
            .native_grants
            .check_update(self.effective_policy(), &proof.request)?;
        ensure!(
            proof.ack.body.witness_sequence == self.sequence + 1
                && proof.ack.body.previous_record_sha256 == self.tip,
            "native grant is not the next original witness journal record"
        );
        self.sequence = proof.ack.body.witness_sequence;
        self.tip = digest.clone();
        self.native_grants.reserved_evaluations = reserved;
        self.native_grants
            .updates
            .insert(proof.request.body.sequence, (proof, digest));
        Ok(())
    }
    /// Atomically append a separately authorized grant to this SAME witness
    /// journal. Identical retries recover the original ACK, including after its
    /// window expired; new updates cannot extend the reviewed finite window.
    pub fn add_native_grant(&mut self, request: &SignedAddNativeGrant) -> Result<NativeGrantProof> {
        ensure!(!self.read_only, "read-only witness cannot append grants");
        self.verify_custody()?;
        if let Some((proof, digest)) = self.native_grants.updates.get(&request.body.sequence) {
            ensure!(
                bytes(&proof.request)? == bytes(request)?,
                "conflicting native grant retransmission"
            );
            ensure!(
                sha(&read(&self.root.join("segments").join(format!(
                    "{:020}.json",
                    proof.ack.body.witness_sequence
                )))?)
                    == *digest,
                "original native grant record changed"
            );
            return Ok(proof.clone());
        }
        validate_update_effective(self.effective_policy(), request)?;
        let now = super::super::native_scq::policy::now_ms()?;
        ensure!(
            request.body.issued_unix_ms <= now
                && now < request.body.signed_grant.body.period_end_unix_ms,
            "new grant update is future-dated or expired"
        );
        request.body.signed_grant.body.validate(now)?;
        self.native_grants
            .check_update(self.effective_policy(), request)?;
        let body = AddNativeGrantAckBody {
            schema: "parallel.add_native_grant_ack.v1".into(),
            base_policy_sha256: self.effective_policy().sha256()?,
            witness_instance: self.effective_policy().witness_instance.clone(),
            authority_instance: self.effective_policy().authority_instance.clone(),
            authority_epoch: self.effective_policy().authority_epoch.clone(),
            request_sha256: sha(&bytes(request)?),
            grant_sha256: request.body.grant_sha256.clone(),
            update_sequence: request.body.sequence,
            previous_update_sha256: request.body.previous_update_sha256.clone(),
            witness_sequence: self.sequence + 1,
            previous_record_sha256: self.tip.clone(),
        };
        let proof = NativeGrantProof {
            epoch_proofs: self.epoch_proofs.clone(),
            request: request.clone(),
            ack: SignedAddNativeGrantAck {
                signature: sign(
                    self.signing
                        .as_ref()
                        .context("public witness verifier cannot sign")?,
                    ACK_DOMAIN,
                    &body,
                )?,
                body,
            },
        };
        let data = bytes(&proof)?;
        ensure!(data.len() <= MAX_FRAME, "native grant record exceeds bound");
        // Pending bytes are outside segments, so interrupted writes cannot look
        // like a committed record. Keep failed staging as preservation evidence.
        let pending = tempfile::Builder::new()
            .prefix(".native-grant-pending-")
            .tempfile_in(&self.root)?;
        let (mut file, staging) = pending.keep()?;
        file.write_all(&data)?;
        file.sync_all()?;
        sync_dir(&self.root)?;
        let destination = self
            .root
            .join("segments")
            .join(format!("{:020}.json", self.sequence + 1));
        crate::native_persistence::publish_staged(&staging, &destination)?;
        sync_dir(&self.root.join("segments"))?;
        self.replay_native_grant(proof.clone(), sha(&data))?;
        self.publish_head()?;
        crate::native_persistence::discard_staging(&staging)?;
        sync_dir(&self.root)?;
        Ok(proof)
    }
}
/// Fixed framed stdin/stdout endpoint; key/storage paths come only from the
/// root-owned forced launcher, never from a caller-supplied command string.
pub fn serve_stdio_add_native_grant(
    root: &Path,
    policy_path: &Path,
    key_path: &Path,
) -> Result<()> {
    let policy: WitnessPolicy = serde_json::from_slice(&read(policy_path)?)?;
    let request: SignedAddNativeGrant = read_frame(std::io::stdin().lock())?;
    let mut store = WitnessStore::open(root, &policy, key_path)?;
    let proof = store.add_native_grant(&request)?;
    write_frame(std::io::stdout().lock(), &proof)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    pub(in crate::eval_registry::witness) fn fixture() -> (
        tempfile::TempDir,
        PathBuf,
        PathBuf,
        WitnessPolicy,
        SigningKey,
    ) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("same-witness-ledger");
        let ack_path = temp.path().join("test-only-ack-key");
        let ack = SigningKey::from_bytes(&[42; 32]);
        let provisioner = SigningKey::from_bytes(&[43; 32]);
        let authority = SigningKey::from_bytes(&[44; 32]);
        publish(&ack_path, &[42; 32]).unwrap();
        let now = super::super::super::native_scq::policy::now_ms().unwrap();
        let digest = sha(b"test identity");
        let executable = sha(&fs::read("/proc/self/exe").unwrap());
        let template = NativeGrant {
            schema: "jope.smartcluster_fit_grant.v1".into(),
            root: "/data".into(),
            authority_root: "/data/authority".into(),
            witness: "/data/local-witness".into(),
            authority_instance: digest.clone(),
            authority_epoch: digest.clone(),
            authority_manifest_sha256: digest.clone(),
            capability: "regression_collection".into(),
            scope: "test".into(),
            tenant: "test".into(),
            project: "test".into(),
            daemon_socket: "/run/test.sock".into(),
            cgroup_root: "/sys/fs/cgroup/test".into(),
            daemon_uid: 1000,
            daemon_executable_sha256: digest.clone(),
            authority_executable_sha256: executable.clone(),
            evaluator_executable_sha256: executable.clone(),
            allowed_runtime_sha256: std::collections::BTreeSet::from([digest.clone()]),
            witness_public_key: hex::encode(ack.verifying_key().to_bytes()),
            decision_receipt_sha256: digest.clone(),
            release_receipt_sha256: digest.clone(),
            coverage_receipt_sha256: digest.clone(),
            approved_task_sha256: std::collections::BTreeSet::from([digest.clone()]),
            period_start_unix_ms: now - 1000,
            period_end_unix_ms: now + 60_000,
            max_candidates_per_task: 4,
            max_fit_ms: 1000,
            numerical_contract_sha256: digest.clone(),
            training_sha256: std::collections::BTreeSet::from([digest.clone()]),
            fit_seed: 0,
        };
        let policy = WitnessPolicy {
            native_fit_trust: None,
            funding_trust: None,
            native_grant_delegation: Some(NativeGrantDelegation {
                schema: "parallel.native_grant_delegation.v1".into(),
                provisioner_public_key: hex::encode(provisioner.verifying_key().to_bytes()),
                run: "finite-test".into(),
                authority_instance: digest.clone(),
                authority_epoch: digest.clone(),
                initial_optimizer_step: 60000,
                cycles: 2,
                scientific_policy_sha256: digest.clone(),
                controller_sha256: digest.clone(),
                manifest_root: "/data/manifests".into(),
                deadline_unix_ms: now + 60_000,
                total_evaluation_budget: 4,
                grant_template: template,
            }),
            schema: "parallel.fit_event_witness_policy.v1".into(),
            witness_instance: sha(b"same journal"),
            authority_instance: digest.clone(),
            authority_epoch: digest.clone(),
            authority_public_key: hex::encode(authority.verifying_key().to_bytes()),
            witness_public_key: hex::encode(ack.verifying_key().to_bytes()),
            authority_executable_sha256: executable.clone(),
            authority_source_sha256: authority_source_sha256(),
            witness_executable_sha256: executable,
            witness_source_sha256: source_sha256(),
            ssh_known_hosts_sha256: digest,
            ssh_host_alias: "test".into(),
            ssh_destination: "test".into(),
        };
        WitnessStore::initialize(&root, &policy).unwrap();
        (temp, root, ack_path, policy, provisioner)
    }
    fn update(
        policy: &WitnessPolicy,
        provisioner: &SigningKey,
        sequence: u64,
        previous: String,
        evaluations: u32,
    ) -> SignedAddNativeGrant {
        let d = policy.native_grant_delegation.as_ref().unwrap();
        let mut grant = d.grant_template.clone();
        let task = NativeGrantArtifact {
            path: format!("/data/manifests/task-{sequence}.json").into(),
            sha256: sha(format!("task{sequence}").as_bytes()),
        };
        grant.approved_task_sha256 = std::collections::BTreeSet::from([task.sha256.clone()]);
        grant.max_candidates_per_task = evaluations;
        let raw = bytes(&grant).unwrap();
        let mut message = super::super::super::native_scq::policy::SIGNATURE_DOMAIN.to_vec();
        message.extend(&raw);
        let signed_grant = SignedNativeGrant {
            body: grant,
            signature: hex::encode(provisioner.sign(&message).to_bytes()),
        };
        let body = AddNativeGrantBody {
            schema: "parallel.add_native_grant.v1".into(),
            base_policy_sha256: policy.sha256().unwrap(),
            delegation_sha256: d.sha256().unwrap(),
            sequence,
            previous_update_sha256: previous,
            signed_grant,
            grant_sha256: sha(&raw),
            loop_request: NativeGrantArtifact {
                path: format!("/data/manifests/request-{sequence}.json").into(),
                sha256: sha(format!("request{sequence}").as_bytes()),
            },
            graph_admission: NativeGrantArtifact {
                path: format!("/data/manifests/admission-{sequence}.json").into(),
                sha256: sha(format!("admission{sequence}").as_bytes()),
            },
            task_manifest: task,
            run: d.run.clone(),
            generation: 0,
            role: NativeGrantRole::Direct,
            evaluations,
            issued_unix_ms: super::super::super::native_scq::policy::now_ms().unwrap(),
        };
        SignedAddNativeGrant {
            signature: sign(provisioner, ADD_NATIVE_GRANT_DOMAIN, &body).unwrap(),
            body,
        }
    }
    #[test]
    fn update_is_original_journal_record_and_retry_reuses_exact_ack() {
        let (_t, root, key, policy, provisioner) = fixture();
        let request = update(&policy, &provisioner, 1, ZERO.into(), 2);
        let original_policy = read(&root.join("policy.json")).unwrap();
        let empty_head = read(&root.join("head.json")).unwrap();
        let proof = {
            let mut store = WitnessStore::open(&root, &policy, &key).unwrap();
            let proof = store.add_native_grant(&request).unwrap();
            assert_eq!(store.sequence, 1);
            assert!(store.records.is_empty());
            assert_eq!(
                bytes(&proof).unwrap(),
                bytes(&store.add_native_grant(&request).unwrap()).unwrap()
            );
            proof
        };
        // Simulate crash after original segment fsync but before head rename.
        fs::write(root.join("head.json"), &empty_head).unwrap();
        {
            let mut store = WitnessStore::open_read_only(&root, &policy, &key).unwrap();
            assert_eq!(store.sequence, 1);
            assert!(store.add_native_grant(&request).is_err());
        }
        assert_eq!(read(&root.join("head.json")).unwrap(), empty_head);
        {
            let mut store = WitnessStore::open(&root, &policy, &key).unwrap();
            assert_eq!(
                bytes(&store.add_native_grant(&request).unwrap()).unwrap(),
                bytes(&proof).unwrap()
            );
        }
        assert_eq!(read(&root.join("policy.json")).unwrap(), original_policy);
        assert!(verify_native_grant_proof(&policy, &proof).is_ok());
    }
    #[test]
    fn cumulative_budget_sequence_conflicts_and_changed_proofs_fail_closed() {
        let (_t, root, key, policy, provisioner) = fixture();
        let request = update(&policy, &provisioner, 1, ZERO.into(), 3);
        let mut store = WitnessStore::open(&root, &policy, &key).unwrap();
        let proof = store.add_native_grant(&request).unwrap();
        let mut conflict = request.clone();
        conflict.body.generation = 1;
        conflict.signature = sign(&provisioner, ADD_NATIVE_GRANT_DOMAIN, &conflict.body).unwrap();
        assert!(store.add_native_grant(&conflict).is_err());
        let excessive = update(&policy, &provisioner, 2, sha(&bytes(&request).unwrap()), 2);
        assert!(store.add_native_grant(&excessive).is_err());
        let stale = update(&policy, &provisioner, 2, ZERO.into(), 1);
        assert!(store.add_native_grant(&stale).is_err());
        let next = update(&policy, &provisioner, 2, sha(&bytes(&request).unwrap()), 1);
        store.add_native_grant(&next).unwrap();
        assert_eq!(store.sequence, 2);
        let mut corrupt = proof;
        corrupt.ack.body.witness_instance = sha(b"other store");
        assert!(verify_native_grant_proof(&policy, &corrupt).is_err());
        drop(store);
        let store = WitnessStore::open(&root, &policy, &key).unwrap();
        assert_eq!(store.native_grants.reserved_evaluations, 4);
    }
    #[test]
    fn delegation_cannot_broaden_template_or_use_authority_key() {
        let (_t, _root, _key, policy, provisioner) = fixture();
        let request = update(&policy, &provisioner, 1, ZERO.into(), 1);
        let mut wrong = request.clone();
        wrong.body.task_manifest.path = "/outside/task.json".into();
        wrong.signature = sign(&provisioner, ADD_NATIVE_GRANT_DOMAIN, &wrong.body).unwrap();
        assert!(validate_update(&policy, &wrong).is_err());
        let mut wrong = request.clone();
        wrong.body.generation = 2;
        wrong.signature = sign(&provisioner, ADD_NATIVE_GRANT_DOMAIN, &wrong.body).unwrap();
        assert!(validate_update(&policy, &wrong).is_err());
        let mut wrong = policy.clone();
        wrong
            .native_grant_delegation
            .as_mut()
            .unwrap()
            .provisioner_public_key = wrong.authority_public_key.clone();
        assert!(wrong.validate().is_err());
        let mut wrong = request;
        wrong.body.evaluations = 0;
        wrong.signature = sign(&provisioner, ADD_NATIVE_GRANT_DOMAIN, &wrong.body).unwrap();
        assert!(validate_update(&policy, &wrong).is_err());
    }
    #[test]
    fn changing_effective_epoch_does_not_reset_original_grant_sequence_or_spent_budget() {
        let (_t, root, key, policy, provisioner) = fixture();
        let original = update(&policy, &provisioner, 1, ZERO.into(), 3);
        let mut store = WitnessStore::open(&root, &policy, &key).unwrap();
        store.add_native_grant(&original).unwrap();
        let mut successor = policy.clone();
        successor.authority_epoch = sha(b"reviewed successor");
        let d = successor.native_grant_delegation.as_mut().unwrap();
        d.authority_epoch = successor.authority_epoch.clone();
        d.grant_template.authority_epoch = successor.authority_epoch.clone();
        let reset = update(&successor, &provisioner, 1, ZERO.into(), 1);
        assert!(store
            .native_grants
            .check_update(&successor, &reset)
            .unwrap_err()
            .to_string()
            .contains("chain"));
        let too_much = update(
            &successor,
            &provisioner,
            2,
            sha(&bytes(&original).unwrap()),
            2,
        );
        assert!(store
            .native_grants
            .check_update(&successor, &too_much)
            .unwrap_err()
            .to_string()
            .contains("budget exhausted"));
        let within_remaining = update(
            &successor,
            &provisioner,
            2,
            sha(&bytes(&original).unwrap()),
            1,
        );
        assert_eq!(
            store
                .native_grants
                .check_update(&successor, &within_remaining)
                .unwrap(),
            4
        );
        assert_eq!(store.native_grants.reserved_evaluations, 3);
    }
}
