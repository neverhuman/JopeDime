//! Native recovery retains every original uncertain identity; it never releases a claim.
use super::*;
use crate::eval_registry::native_scq::{self, policy};
use ed25519_dalek::{Signature, VerifyingKey};
use jain_smartcluster::{FencedAttempt, GetRequest, InspectAttemptRequest, Request, Response};
const DOMAIN: &[u8] = b"jope.native_held_epoch.signature.v1\0";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryArtifact {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OriginalCustody {
    pub authority_attempt: FencedAttempt,
    pub target_attempt: FencedAttempt,
    pub authority_leader: ProcessIdentity,
    pub target_leader: ProcessIdentity,
    pub requesting_process: ProcessIdentity,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeldBinding {
    pub row_sha256: String,
    pub request_sha256: String,
    pub custody: OriginalCustody,
    /// Older claims require separately reviewed, immutable historical custody evidence.
    /// This is not written into, or substituted for, their original provenance.
    pub historical_custody: Option<RecoveryArtifact>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicWitnessSnapshot {
    pub root: PathBuf,
    pub policy: RecoveryArtifact,
    pub head_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRecoveryBody {
    pub schema: String,
    pub authority_root: PathBuf,
    pub witness: PathBuf,
    pub authority_instance: String,
    pub authority_manifest_sha256: String,
    pub parent_epoch: String,
    pub new_epoch: String,
    pub report_sha256: String,
    pub decision: RecoveryArtifact,
    pub not_before_unix_ms: u64,
    pub expires_unix_ms: u64,
    pub daemon_socket: PathBuf,
    pub daemon_uid: u32,
    pub daemon_executable_sha256: String,
    pub cgroup_root: PathBuf,
    pub held: BTreeMap<String, HeldBinding>,
    pub independent_witness: PublicWitnessSnapshot,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedNativeRecovery {
    pub authorization_public_key: String,
    pub body_sha256: String,
    pub body: NativeRecoveryBody,
    pub signature: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: String,
    policy: RecoveryArtifact,
    committed_unix_ms: u64,
    original_report: Value,
    original_rows: BTreeMap<String, Value>,
    drain_evidence: Value,
    independent_witness_verification: Value,
}
fn artifact(reference: &RecoveryArtifact) -> Result<Vec<u8>> {
    ensure!(
        reference.path.is_absolute() && valid_sha(&reference.sha256),
        "invalid recovery artifact"
    );
    let bytes = policy::read_file(&reference.path)?;
    ensure!(
        hash(&bytes) == reference.sha256,
        "recovery artifact changed"
    );
    Ok(bytes)
}
fn verify_signed(signed: &SignedNativeRecovery, when: u64) -> Result<()> {
    let body = &signed.body;
    ensure!(
        body.schema == "jope.native_held_epoch.v1",
        "unknown held-work recovery schema"
    );
    ensure!(
        body.not_before_unix_ms <= when && when < body.expires_unix_ms,
        "recovery decision is outside its reviewed validity window"
    );
    ensure!(
        !body.held.is_empty() && body.held.len() <= 100_000,
        "unbounded or empty held set"
    );
    for value in [
        &body.authority_instance,
        &body.authority_manifest_sha256,
        &body.parent_epoch,
        &body.new_epoch,
        &body.report_sha256,
        &body.daemon_executable_sha256,
        &signed.body_sha256,
    ] {
        ensure!(valid_sha(value), "malformed recovery hash/epoch");
    }
    ensure!(
        body.parent_epoch != body.new_epoch,
        "recovery cannot reuse its parent epoch"
    );
    for path in [
        &body.authority_root,
        &body.witness,
        &body.daemon_socket,
        &body.cgroup_root,
    ] {
        ensure!(
            path.is_absolute()
                && path.components().all(|c| matches!(
                    c,
                    std::path::Component::RootDir | std::path::Component::Normal(_)
                )),
            "unsafe recovery path"
        );
    }
    let bytes = encoded(body)?;
    ensure!(
        hash(&bytes) == signed.body_sha256,
        "signed recovery body changed"
    );
    let raw: [u8; 32] = hex::decode(&signed.authorization_public_key)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("recovery public key width"))?;
    let key = VerifyingKey::from_bytes(&raw)?;
    ensure!(!key.is_weak(), "weak recovery signing key");
    let mut message = DOMAIN.to_vec();
    message.extend(bytes);
    key.verify_strict(
        &message,
        &Signature::from_slice(&hex::decode(&signed.signature)?)?,
    )?;
    let decision: Value = serde_json::from_slice(&artifact(&body.decision)?)?;
    ensure!(
        decision["schema"] == "jope.native_held_epoch_review.v1"
            && decision["authority_instance"] == body.authority_instance
            && decision["parent_epoch"] == body.parent_epoch
            && decision["new_epoch"] == body.new_epoch
            && decision["report_sha256"] == body.report_sha256
            && decision["held_bindings_sha256"] == hash(&encoded(&body.held)?)
            && decision["original_identities_remain_held"] == true
            && decision["original_outcomes_are_not_inferred"] == true,
        "independent recovery decision does not bind this exact held transition"
    );
    Ok(())
}
fn load_policy(path: &Path, at: u64) -> Result<(SignedNativeRecovery, RecoveryArtifact)> {
    // The separately installed root-controlled file pins both signer and exact body.
    // An authority process or planner cannot supply its own self-signed approval.
    let bytes = policy::protected(path)?;
    let signed: SignedNativeRecovery = serde_json::from_slice(&bytes)?;
    crate::native_review::require_signer(&signed.authorization_public_key)?;
    verify_signed(&signed, at)?;
    Ok((
        signed,
        RecoveryArtifact {
            path: path.into(),
            sha256: hash(&bytes),
        },
    ))
}
fn original_requests(authority: &Authority) -> Result<BTreeMap<String, Value>> {
    let mut claims = BTreeMap::new();
    let mut bytes = 0u64;
    let mut reader = BufReader::new(File::open(authority.root.join("dispatch.jsonl"))?);
    loop {
        let line = read_event_line(&mut reader)?;
        if line.is_empty() {
            break;
        }
        let event: Event = serde_json::from_slice(&line)?;
        if event.kind == "claim" {
            let key = event.payload["claim"]["fit_key"]
                .as_str()
                .context("original journal claim key")?
                .to_owned();
            bytes = bytes
                .checked_add(encoded(&event.payload["request"])?.len() as u64)
                .context("claim provenance byte overflow")?;
            ensure!(claims.len()<100_000 && bytes <= 64 << 20, "original claim provenance exceeds bounded native recovery; retain all original evidence");
            ensure!(
                claims
                    .insert(key, event.payload["request"].clone())
                    .is_none(),
                "duplicate original claim journal identity"
            );
        }
    }
    Ok(claims)
}

fn check_binding(key: &str, row: &Value, request: &Value, binding: &HeldBinding) -> Result<()> {
    ensure!(
        hash(&encoded(row)?) == binding.row_sha256
            && hash(&encoded(request)?) == binding.request_sha256,
        "original held entry/provenance changed"
    );
    let owner: ProcessIdentity = serde_json::from_str(
        row["owner_json"]
            .as_str()
            .context("original claim owner absent")?,
    )?;
    let peer: ProcessIdentity =
        serde_json::from_value(request["details"]["requesting_process"].clone())?;
    ensure!(
        owner == binding.custody.authority_leader && peer == binding.custody.requesting_process,
        "recovery custody differs from original committed owner/requester"
    );
    let native = &request["details"]["native_allocation"];
    if native.is_null() {
        let reference = binding
            .historical_custody
            .as_ref()
            .context("original claim lacks native fences and historical evidence")?;
        let history: Value = serde_json::from_slice(&artifact(reference)?)?;
        ensure!(
            history["schema"] == "jope.native_historical_claim_custody.v1"
                && history["fit_key"] == key
                && history["entry_sha256"] == binding.row_sha256
                && history["request_sha256"] == binding.request_sha256
                && history["custody"] == serde_json::to_value(&binding.custody)?
                && history["reviewed_original_ancestry"] == true,
            "historical original custody evidence differs"
        );
        let sources: Vec<RecoveryArtifact> =
            serde_json::from_value(history["original_sources"].clone())?;
        ensure!(
            !sources.is_empty() && sources.len() <= 1024,
            "historical custody original sources absent"
        );
        for source in sources {
            artifact(&source)?;
        }
    } else {
        ensure!(
            binding.historical_custody.is_none()
                && native["schema"] == "jope.native_claim_allocation.v1",
            "native preclaim provenance cannot be replaced by historical assertions"
        );
        ensure!(
            native["authority_attempt"] == serde_json::to_value(binding.custody.authority_attempt)?
                && native["target_attempt"]
                    == serde_json::to_value(binding.custody.target_attempt)?
                && native["authority_leader"]
                    == serde_json::to_value(&binding.custody.authority_leader)?
                && native["target_leader"] == serde_json::to_value(&binding.custody.target_leader)?,
            "original authenticated claim fences changed"
        );
    }
    Ok(())
}
fn process_gone(original: &ProcessIdentity) -> Result<()> {
    let boot = fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    if boot.trim() != original.boot_id {
        return Ok(());
    }
    let directory = PathBuf::from(format!("/proc/{}", original.pid));
    if !directory.exists() {
        return Ok(());
    }
    ensure!(
        ProcessIdentity::read(original.pid)? != *original,
        "original native process is still alive"
    );
    Ok(())
}
fn drains(body: &NativeRecoveryBody) -> Result<Value> {
    ensure!(
        body.cgroup_root.starts_with("/sys/fs/cgroup")
            && fs::canonicalize(&body.cgroup_root)? == body.cgroup_root,
        "recovery requires physical delegated cgroup root"
    );
    let filesystem = nix::sys::statfs::statfs(&body.cgroup_root)
        .map_err(|_| anyhow::anyhow!("cgroup filesystem unavailable"))?;
    ensure!(
        filesystem.filesystem_type() == nix::sys::statfs::CGROUP2_SUPER_MAGIC,
        "recovery cgroup is not kernel cgroup v2"
    );
    let mut checked = BTreeMap::new();
    for binding in body.held.values() {
        process_gone(&binding.custody.requesting_process)?;
        for (attempt, leader) in [
            (
                binding.custody.authority_attempt,
                &binding.custody.authority_leader,
            ),
            (
                binding.custody.target_attempt,
                &binding.custody.target_leader,
            ),
        ] {
            process_gone(leader)?;
            let key = format!("{}:{}:{}", attempt.job_id, attempt.number, attempt.fence);
            if checked.contains_key(&key) {
                continue;
            }
            let response = native_scq::authenticated_exchange(
                &body.daemon_socket,
                body.daemon_uid,
                &body.daemon_executable_sha256,
                None,
                Request::InspectAttempt(InspectAttemptRequest { attempt }),
            )?;
            let inspection = match response {
                Response::AttemptInspected(Some(i)) => i,
                _ => anyhow::bail!("original native attempt evidence absent"),
            };
            let expected = body.cgroup_root.join(format!(
                "scq-{}-attempt-{}-fence-{}",
                attempt.job_id, attempt.number, attempt.fence
            ));
            ensure!(
                inspection.record.attempt == attempt
                    && inspection.record.pid == Some(leader.pid)
                    && inspection.record.finished_at.is_some()
                    && inspection.active_process.is_none()
                    && inspection.record.cgroup_path.as_deref() == expected.to_str(),
                "original native attempt is not a terminal fully drained allocation"
            );
            let response = native_scq::authenticated_exchange(
                &body.daemon_socket,
                body.daemon_uid,
                &body.daemon_executable_sha256,
                None,
                Request::Get(GetRequest {
                    job_id: attempt.job_id,
                }),
            )?;
            let job = match response {
                Response::Job(Some(j)) => j,
                _ => anyhow::bail!("original native job absent"),
            };
            ensure!(
                job.active_attempt.is_none() && job.state.is_terminal(),
                "old native job can still execute/retry"
            );
            if expected.exists() {
                ensure!(
                    fs::canonicalize(&expected)? == expected,
                    "old native cgroup path replaced"
                );
                let events = fs::read_to_string(expected.join("cgroup.events"))?;
                ensure!(
                    events.lines().any(|line| line == "populated 0"),
                    "original cgroup subtree remains populated"
                );
            }
            checked.insert(key, serde_json::to_value(inspection)?);
        }
    }
    Ok(serde_json::to_value(checked)?)
}
fn verify_recorded_drains(body: &NativeRecoveryBody, evidence: &Value) -> Result<()> {
    let mut expected = BTreeSet::new();
    for binding in body.held.values() {
        for (attempt, leader) in [
            (
                binding.custody.authority_attempt,
                &binding.custody.authority_leader,
            ),
            (
                binding.custody.target_attempt,
                &binding.custody.target_leader,
            ),
        ] {
            let key = format!("{}:{}:{}", attempt.job_id, attempt.number, attempt.fence);
            expected.insert(key.clone());
            let inspection: jain_smartcluster::AttemptInspection = serde_json::from_value(
                evidence
                    .get(&key)
                    .context("original committed drain evidence missing")?
                    .clone(),
            )?;
            let path = body.cgroup_root.join(format!(
                "scq-{}-attempt-{}-fence-{}",
                attempt.job_id, attempt.number, attempt.fence
            ));
            ensure!(
                inspection.record.attempt == attempt
                    && inspection.record.pid == Some(leader.pid)
                    && inspection.record.finished_at.is_some()
                    && inspection.record.state.is_terminal()
                    && inspection.record.lease.attempt == attempt
                    && inspection.active_process.is_none()
                    && inspection.record.cgroup_path.as_deref() == path.to_str(),
                "original committed drain evidence differs from reviewed held allocation"
            );
        }
    }
    ensure!(
        evidence
            .as_object()
            .context("drain evidence map")?
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            == expected,
        "unexpected committed drain evidence identity"
    );
    Ok(())
}
#[path = "native_recovery/recovery.rs"]
mod recovery;

#[cfg(test)]
#[path = "native_recovery/tests.rs"]
mod tests;
