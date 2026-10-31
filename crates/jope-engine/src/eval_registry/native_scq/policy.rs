//! Installed, signed native fit authority policy. No grant minting or activation.
use anyhow::{ensure, Result};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
const MAX_BYTES: usize = 1024 * 1024;
pub(crate) const SIGNATURE_DOMAIN: &[u8] = b"jope.smartcluster_fit_grant.signature.v1\0";
pub(crate) fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(crate) fn valid_hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(crate) fn now_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeGrant {
    pub schema: String,
    pub root: PathBuf,
    pub authority_root: PathBuf,
    pub witness: PathBuf,
    pub authority_instance: String,
    pub authority_epoch: String,
    pub authority_manifest_sha256: String,
    pub capability: String,
    pub scope: String,
    pub tenant: String,
    pub project: String,
    pub daemon_socket: PathBuf,
    pub cgroup_root: PathBuf,
    pub daemon_uid: u32,
    pub daemon_executable_sha256: String,
    pub authority_executable_sha256: String,
    pub evaluator_executable_sha256: String,
    pub allowed_runtime_sha256: BTreeSet<String>,
    pub witness_public_key: String,
    pub decision_receipt_sha256: String,
    pub release_receipt_sha256: String,
    pub coverage_receipt_sha256: String,
    pub approved_task_sha256: BTreeSet<String>,
    pub period_start_unix_ms: u64,
    pub period_end_unix_ms: u64,
    pub max_candidates_per_task: u32,
    pub max_fit_ms: u64,
    pub numerical_contract_sha256: String,
    pub training_sha256: BTreeSet<String>,
    pub fit_seed: u64,
}
impl NativeGrant {
    pub(crate) fn validate(&self, now: u64) -> Result<()> {
        ensure!(
            self.schema == "jope.smartcluster_fit_grant.v1",
            "unsupported native grant"
        );
        ensure!(
            matches!(
                self.capability.as_str(),
                "synthetic_qualification" | "regression_collection"
            ),
            "native capability is not fit collection"
        );
        ensure!(
            self.period_start_unix_ms <= now && now < self.period_end_unix_ms,
            "original native grant window expired/not open"
        );
        ensure!(
            (1..=1000).contains(&self.max_candidates_per_task)
                && (1..=30000).contains(&self.max_fit_ms)
                && self.fit_seed == 0,
            "native grant exceeds evaluator limits"
        );
        for text in [&self.scope, &self.tenant, &self.project] {
            ensure!(
                !text.is_empty() && text.len() <= 256 && !text.chars().any(char::is_control),
                "invalid native scope identity"
            );
        }
        for p in [
            &self.root,
            &self.authority_root,
            &self.witness,
            &self.daemon_socket,
            &self.cgroup_root,
        ] {
            ensure!(
                p.is_absolute()
                    && !p
                        .components()
                        .any(|c| matches!(c, std::path::Component::ParentDir)),
                "noncanonical native grant path"
            );
        }
        ensure!(
            self.authority_root.starts_with(&self.root),
            "authority receipt graph outside native data root"
        );
        for h in [
            &self.authority_instance,
            &self.authority_epoch,
            &self.authority_manifest_sha256,
            &self.daemon_executable_sha256,
            &self.authority_executable_sha256,
            &self.evaluator_executable_sha256,
            &self.witness_public_key,
            &self.decision_receipt_sha256,
            &self.release_receipt_sha256,
            &self.coverage_receipt_sha256,
            &self.numerical_contract_sha256,
        ] {
            ensure!(valid_hash(h), "invalid native grant digest");
        }
        for set in [
            &self.allowed_runtime_sha256,
            &self.approved_task_sha256,
            &self.training_sha256,
        ] {
            ensure!(
                !set.is_empty() && set.len() <= 4096 && set.iter().all(|s| valid_hash(s)),
                "invalid bounded native grant set"
            );
        }
        ensure!(
            self.approved_task_sha256.len() <= 1024,
            "native task grant exceeds bound"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedNativeGrant {
    pub body: NativeGrant,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeWitnessTrust {
    pub authorization_public_key: String,
    pub grant_sha256: String,
}
pub(crate) fn verify_signed(
    trust: &NativeWitnessTrust,
    signed: &SignedNativeGrant,
    now: u64,
) -> Result<()> {
    signed.body.validate(now)?;
    let bytes = serde_json::to_vec(&signed.body)?;
    ensure!(
        valid_hash(&trust.grant_sha256) && hash(&bytes) == trust.grant_sha256,
        "grant differs from independently installed authorization"
    );
    let raw: [u8; 32] = hex::decode(&trust.authorization_public_key)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("native public key width"))?;
    let key = VerifyingKey::from_bytes(&raw)?;
    ensure!(!key.is_weak(), "weak native authorization key");
    let mut message = SIGNATURE_DOMAIN.to_vec();
    message.extend(bytes);
    key.verify_strict(
        &message,
        &Signature::from_slice(&hex::decode(&signed.signature)?)?,
    )?;
    Ok(())
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    epoch_proofs: Vec<crate::eval_registry::witness::WitnessEpochProof>,
    schema: String,
    grant: Artifact,
    trust: NativeWitnessTrust,
    decision: Artifact,
    release: Artifact,
    coverage: Artifact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    native_grant_proof: Option<Artifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    witness_policy: Option<Artifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    task_policy_directory: Option<PathBuf>,
}
pub(crate) struct InstalledPolicy {
    path: PathBuf,
    digest: String,
    pub(crate) signed: SignedNativeGrant,
    pub(crate) trust: NativeWitnessTrust,
    pub(crate) native_grant_proof: Option<crate::eval_registry::witness::NativeGrantProof>,
    witness_policy_sha256: Option<String>,
    task_policy_directory: Option<PathBuf>,
}
pub(crate) fn read_file(path: &Path) -> Result<Vec<u8>> {
    let mut file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)?;
    let before = file.metadata()?;
    ensure!(
        before.is_file() && before.len() <= MAX_BYTES as u64,
        "bounded regular native policy artifact required"
    );
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    let named = fs::symlink_metadata(path)?;
    ensure!(
        !named.file_type().is_symlink()
            && before.dev() == named.dev()
            && before.ino() == named.ino()
            && before.len() == after.len()
            && bytes.len() as u64 == before.len()
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec(),
        "native policy artifact changed while reading"
    );
    Ok(bytes)
}
pub(crate) fn protected(path: &Path) -> Result<Vec<u8>> {
    ensure!(
        path.is_absolute(),
        "installed native policy must be absolute"
    );
    for ancestor in path.ancestors() {
        let m = fs::symlink_metadata(ancestor)?;
        ensure!(
            !m.file_type().is_symlink() && m.uid() == 0 && m.mode() & 0o022 == 0,
            "native policy ancestry is not root-controlled"
        );
    }
    read_file(path)
}
fn artifact(a: &Artifact) -> Result<Vec<u8>> {
    ensure!(
        a.path.is_absolute() && valid_hash(&a.sha256),
        "native receipt binding malformed"
    );
    let bytes = read_file(&a.path)?;
    ensure!(
        hash(&bytes) == a.sha256,
        "native prerequisite receipt changed"
    );
    Ok(bytes)
}
impl InstalledPolicy {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let bytes = protected(path)?;
        let config: Config = serde_json::from_slice(&bytes)?;
        ensure!(
            config.schema == "jope.smartcluster_fit_policy.v1",
            "native policy schema"
        );
        let signed: SignedNativeGrant = serde_json::from_slice(&artifact(&config.grant)?)?;
        verify_signed(&config.trust, &signed, now_ms()?)?;
        let native_grant_proof = match (&config.native_grant_proof, &config.witness_policy) {
            (None, None) if config.task_policy_directory.is_none() => None,
            (None, Some(witness_artifact)) if config.task_policy_directory.is_some() => {
                let witness: crate::eval_registry::witness::WitnessPolicy =
                    serde_json::from_slice(&artifact(witness_artifact)?)?;
                let witness = crate::eval_registry::witness::verify_epoch_chain(
                    &witness,
                    &config.epoch_proofs,
                )?;
                ensure!(
                    witness.authority_instance == signed.body.authority_instance
                        && witness.authority_epoch == signed.body.authority_epoch
                        && witness.authority_executable_sha256
                            == signed.body.authority_executable_sha256
                        && witness.witness_public_key == signed.body.witness_public_key
                        && witness.native_grant_delegation.is_some(),
                    "bootstrap differs from immutable delegated witness identity"
                );
                None
            }
            (Some(proof_artifact), Some(witness_artifact)) => {
                let proof: crate::eval_registry::witness::NativeGrantProof =
                    serde_json::from_slice(&artifact(proof_artifact)?)?;
                let witness: crate::eval_registry::witness::WitnessPolicy =
                    serde_json::from_slice(&artifact(witness_artifact)?)?;
                ensure!(
                    config.epoch_proofs.is_empty()
                        || serde_json::to_vec(&config.epoch_proofs)?
                            == serde_json::to_vec(&proof.epoch_proofs)?,
                    "installed epoch chains conflict"
                );
                let trust =
                    crate::eval_registry::witness::verify_native_grant_proof(&witness, &proof)?;
                ensure!(
                    serde_json::to_vec(&trust)? == serde_json::to_vec(&config.trust)?
                        && serde_json::to_vec(&proof.request.body.signed_grant)?
                            == serde_json::to_vec(&signed)?
                        && crate::eval_registry::witness::verify_epoch_chain(
                            &witness,
                            &proof.epoch_proofs
                        )?
                        .sha256()?
                            == proof.request.body.base_policy_sha256,
                    "installed grant differs from independent witness delegation proof"
                );
                Some(proof)
            }
            _ => anyhow::bail!(
                "native grant proof and immutable witness policy must be bound together"
            ),
        };
        if let Some(directory) = &config.task_policy_directory {
            ensure!(
                native_grant_proof.is_none()
                    && directory.is_absolute()
                    && directory.canonicalize()? == *directory,
                "physical task policy directory required"
            );
            for ancestor in directory.ancestors() {
                let metadata = fs::symlink_metadata(ancestor)?;
                ensure!(
                    metadata.is_dir()
                        && !metadata.file_type().is_symlink()
                        && metadata.uid() == 0
                        && metadata.mode() & 0o022 == 0,
                    "task policy directory ancestry is not root controlled"
                );
            }
        }
        for (a, expected) in [
            (&config.decision, &signed.body.decision_receipt_sha256),
            (&config.release, &signed.body.release_receipt_sha256),
            (&config.coverage, &signed.body.coverage_receipt_sha256),
        ] {
            ensure!(
                &a.sha256 == expected,
                "native prerequisite differs from signed grant"
            );
            let _ = artifact(a)?;
        }
        ensure!(
            signed.body.root.canonicalize()? == signed.body.root,
            "native root must be physical canonical checkout"
        );
        Ok(Self {
            path: path.into(),
            digest: hash(&bytes),
            signed,
            trust: config.trust,
            native_grant_proof,
            witness_policy_sha256: config.witness_policy.as_ref().map(|a| a.sha256.clone()),
            task_policy_directory: config.task_policy_directory,
        })
    }
    /// Eligibility preflight only. The authority derives this digest from an
    /// authenticated live JobRecord before selecting a policy for a real claim.
    pub(crate) fn task_policy(&self, task_sha256: &str) -> Result<Option<Self>> {
        ensure!(
            valid_hash(task_sha256),
            "malformed exact task policy identity"
        );
        self.revalidate()?;
        let Some(directory) = &self.task_policy_directory else {
            return if self.signed.body.approved_task_sha256.contains(task_sha256) {
                Ok(Some(Self::load(&self.path)?))
            } else {
                Ok(None)
            };
        };
        let path = directory.join(format!("{task_sha256}.json"));
        match fs::symlink_metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
            Ok(_) => (),
        }
        let selected = Self::load(&path)?;
        ensure!(
            selected.task_policy_directory.is_none()
                && selected.native_grant_proof.is_some()
                && selected.witness_policy_sha256 == self.witness_policy_sha256,
            "task grant differs from original immutable witness base"
        );
        let grant = &selected.signed.body;
        ensure!(
            grant.approved_task_sha256
                == std::collections::BTreeSet::from([task_sha256.to_owned()]),
            "task policy permits a different manifest"
        );
        let mut normalized = grant.clone();
        normalized.approved_task_sha256 = self.signed.body.approved_task_sha256.clone();
        normalized.decision_receipt_sha256 = self.signed.body.decision_receipt_sha256.clone();
        ensure!(
            grant.period_start_unix_ms >= self.signed.body.period_start_unix_ms
                && grant.period_end_unix_ms <= self.signed.body.period_end_unix_ms
                && grant.max_candidates_per_task <= self.signed.body.max_candidates_per_task,
            "task policy broadens bootstrap budget/window"
        );
        normalized.period_start_unix_ms = self.signed.body.period_start_unix_ms;
        normalized.period_end_unix_ms = self.signed.body.period_end_unix_ms;
        normalized.max_candidates_per_task = self.signed.body.max_candidates_per_task;
        ensure!(
            normalized == self.signed.body,
            "task policy broadens bootstrap allocation or numerical scope"
        );
        self.revalidate()?;
        Ok(Some(selected))
    }
    pub(crate) fn revalidate(&self) -> Result<()> {
        let current = Self::load(&self.path)?;
        ensure!(
            current.digest == self.digest && current.trust.grant_sha256 == self.trust.grant_sha256,
            "installed native grant replaced; reconcile before admission"
        );
        Ok(())
    }
    pub(crate) fn policy_sha256(&self) -> &str {
        &self.digest
    }
}
#[cfg(test)]
mod tests {
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
        assert!(
            serde_json::from_slice::<SignedNativeGrant>(&read_file(&regular).unwrap()).is_err()
        );
        let link = path.join("link");
        std::os::unix::fs::symlink(&regular, &link).unwrap();
        assert!(read_file(&link).is_err());
        std::fs::remove_dir_all(path).unwrap();
    }
}
