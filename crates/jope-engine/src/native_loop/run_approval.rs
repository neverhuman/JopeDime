//! Independent whole-run review and original native-allocation drain gate.
//! The planner has no signing or epoch-changing capability.
use super::*;
use crate::eval_registry::{
    native_scq::{self, policy},
    Authority,
};
use ed25519_dalek::{Signature, VerifyingKey};
use jain_smartcluster::{
    FencedAttempt, GetRequest, InspectAttemptRequest, JobRecord, ListFilter, ListRequest,
    ProjectId, Request as ScqRequest, Response, TenantId, Workload,
};
const DOMAIN: &[u8] = b"jope.native_run_approval.signature.v1\0";
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalAllocation {
    pub attempt: FencedAttempt,
    pub leader: Option<crate::eval_registry::ProcessIdentity>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalJobCensus {
    pub schema: String,
    pub tenant: TenantId,
    pub project: ProjectId,
    pub captured_unix_ms: u64,
    /// Exact complete native-job census through captured_unix_ms; root review
    /// checks the snapshot against the original daemon, including queued jobs.
    pub jobs: Vec<JobRecord>,
    pub allocations: Vec<OriginalAllocation>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunApprovalBody {
    pub schema: String,
    pub config: Artifact,
    pub scientific_policy: Artifact,
    pub root: PathBuf,
    pub loop_state: PathBuf,
    pub graph_state: PathBuf,
    pub worker_state: PathBuf,
    pub run: String,
    pub authority_root: PathBuf,
    pub authority_witness: PathBuf,
    pub authority_instance: String,
    pub authority_epoch: String,
    /// None is the explicitly reviewed clean original epoch. A held-work
    /// transition requires its exact committed receipt, never a new empty DB.
    pub recovery_receipt_sha256: Option<String>,
    pub original_job_census: Artifact,
    pub daemon_socket: PathBuf,
    pub daemon_uid: u32,
    pub daemon_executable_sha256: String,
    pub cgroup_root: PathBuf,
    pub resident_job: JobId,
    pub authority_job: JobId,
    pub authority_job_spec_sha256: String,
    pub resident_job_spec_sha256: String,
    pub loop_executable_sha256: String,
    pub eligibility_executable_sha256: BTreeSet<String>,
    pub not_before_unix_ms: u64,
    pub expires_unix_ms: u64,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedRunApproval {
    pub authorization_public_key: String,
    pub body_sha256: String,
    pub body: RunApprovalBody,
    pub signature: String,
}
fn verify_signature(signed: &SignedRunApproval, now: u64) -> Result<()> {
    let b = &signed.body;
    ensure!(
        b.schema == "jope.native_run_approval.v1"
            && b.not_before_unix_ms <= now
            && now < b.expires_unix_ms
            && b.resident_job != b.authority_job,
        "run approval schema/window/allocation identity invalid"
    );
    for h in [
        &b.authority_instance,
        &b.authority_epoch,
        &b.daemon_executable_sha256,
        &b.loop_executable_sha256,
        &signed.body_sha256,
    ] {
        ensure!(
            h.len() == 64
                && h.bytes()
                    .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x)),
            "run approval hash invalid"
        );
    }
    ensure!(
        !b.eligibility_executable_sha256.is_empty() && b.eligibility_executable_sha256.len() <= 8,
        "unbounded eligibility executable review"
    );
    let bytes = serde_json::to_vec(b)?;
    ensure!(
        hash(&bytes) == signed.body_sha256,
        "run approval body changed"
    );
    let raw: [u8; 32] = hex::decode(&signed.authorization_public_key)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("review key width"))?;
    let key = VerifyingKey::from_bytes(&raw)?;
    ensure!(!key.is_weak(), "weak run review key");
    let mut message = DOMAIN.to_vec();
    message.extend(bytes);
    key.verify_strict(
        &message,
        &Signature::from_slice(&hex::decode(&signed.signature)?)?,
    )?;
    Ok(())
}
pub(crate) fn verify_installed_run_signature(signed: &SignedRunApproval, at: u64) -> Result<()> {
    crate::native_review::require_signer(&signed.authorization_public_key)?;
    verify_signature(signed, at)
}
fn exchange(body: &RunApprovalBody, request: ScqRequest) -> Result<Response> {
    native_scq::authenticated_exchange(
        &body.daemon_socket,
        body.daemon_uid,
        &body.daemon_executable_sha256,
        None,
        request,
    )
}
fn job(body: &RunApprovalBody, id: JobId) -> Result<JobRecord> {
    match exchange(body, ScqRequest::Get(GetRequest { job_id: id }))? {
        Response::Job(Some(j)) => Ok(j),
        _ => anyhow::bail!("reviewed native job absent"),
    }
}
fn process_gone(p: &crate::eval_registry::ProcessIdentity) -> Result<()> {
    if fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim() != p.boot_id
        || !Path::new(&format!("/proc/{}", p.pid)).exists()
    {
        return Ok(());
    }
    ensure!(
        crate::eval_registry::ProcessIdentity::read(p.pid)? != *p,
        "original native leader remains alive"
    );
    Ok(())
}
impl Loop {
    pub(super) fn verify_run_approval(&self) -> Result<RunApprovalBody> {
        let bytes = policy::protected(&self.config.run_approval)?;
        let signed: SignedRunApproval = serde_json::from_slice(&bytes)?;
        crate::native_review::require_signer(&signed.authorization_public_key)?;
        verify_signature(&signed, now()?)?;
        let b = &signed.body;
        let approved: Config = native_genome::load(&self.root, &b.config)?;
        ensure!(
            serde_json::to_vec(&approved)? == serde_json::to_vec(&self.config)?
                && b.scientific_policy == self.config.scientific_policy
                && b.root == self.root
                && b.loop_state == self.state
                && b.graph_state == self.graph_state
                && b.run == self.config.run
                && b.resident_job == self.config.resident_job
                && self.config.task_deadline.as_unix_millis() <= b.expires_unix_ms,
            "run/config/scientific window differs from independent approval"
        );
        let executable = crate::manifest::file_hash(Path::new("/proc/self/exe"))?;
        ensure!(
            if self.read_only {
                b.eligibility_executable_sha256.contains(&executable)
                    || b.loop_executable_sha256 == executable
            } else {
                b.loop_executable_sha256 == executable
            },
            "run controller image is not independently approved"
        );
        let installed = policy::InstalledPolicy::load(&self.config.fit_policy)?;
        let grant = &installed.signed.body;
        ensure!(
            grant.authority_root == b.authority_root
                && grant.authority_instance == b.authority_instance
                && grant.authority_epoch == b.authority_epoch
                && grant.root == self.root
                && grant.witness == b.authority_witness
                && grant.daemon_socket == b.daemon_socket
                && grant.daemon_uid == b.daemon_uid
                && grant.daemon_executable_sha256 == b.daemon_executable_sha256,
            "fit authority differs from reviewed run epoch"
        );
        let status = Authority::native_epoch_status(&b.authority_root, &b.authority_witness)?;
        ensure!(
            status["authority_manifest_sha256"] == grant.authority_manifest_sha256
                && status["authority_instance"] == b.authority_instance
                && status["authority_epoch"] == b.authority_epoch
                && status["recovery_receipt_sha256"]
                    == serde_json::to_value(&b.recovery_receipt_sha256)?
                && (b.recovery_receipt_sha256.is_some()
                    || b.authority_epoch == b.authority_instance),
            "whole learning graph requires original clean or reviewed recovered authority epoch"
        );
        let census: OriginalJobCensus = native_genome::load(&self.root, &b.original_job_census)?;
        ensure!(
            census.schema == "jope.native_original_job_census.v1"
                && census.captured_unix_ms <= b.not_before_unix_ms
                && census.jobs.len() <= 100_000
                && census.allocations.len() <= 100_000,
            "original census bounds/window"
        );
        let mut old = BTreeSet::new();
        for original in &census.jobs {
            ensure!(
                old.insert(original.id)
                    && original.spec.tenant == census.tenant
                    && original.spec.project == census.project
                    && original.created_at.as_unix_millis() <= census.captured_unix_ms,
                "original census duplicate/scope/time"
            );
            let current = job(b, original.id)?;
            ensure!(
                current.spec == original.spec
                    && current.created_at == original.created_at
                    && current.state.is_terminal()
                    && current.active_attempt.is_none()
                    && current.attempt_count == original.attempt_count,
                "original queued/running native work has not fully terminated"
            );
        }
        ensure!(
            !old.contains(&b.resident_job) && !old.contains(&b.authority_job),
            "old allocation cannot be relabelled as new run service"
        );
        ensure!(
            b.cgroup_root.starts_with("/sys/fs/cgroup")
                && fs::canonicalize(&b.cgroup_root)? == b.cgroup_root,
            "run drain requires physical delegated cgroup"
        );
        let name = std::ffi::CString::new(b.cgroup_root.as_os_str().as_encoded_bytes())?;
        let mut st = std::mem::MaybeUninit::<libc::statfs>::uninit();
        ensure!(
            // SAFETY: `name` is a CString that outlives the call (NUL-terminated path) and `st`
            // has room for the one statfs that statfs writes.
            unsafe { libc::statfs(name.as_ptr(), st.as_mut_ptr()) } == 0
                // SAFETY: `&&` short-circuits, so this runs only after statfs returned 0 and
                // filled every field of `st`.
                && unsafe { st.assume_init() }.f_type == libc::CGROUP2_SUPER_MAGIC,
            "run drain requires real cgroup v2"
        );
        let mut seen = BTreeSet::new();
        for allocation in &census.allocations {
            let a = allocation.attempt;
            ensure!(
                old.contains(&a.job_id) && seen.insert((a.job_id, a.number, a.fence)),
                "unexpected/duplicate old allocation"
            );
            if let Some(leader) = &allocation.leader {
                process_gone(leader)?;
            }
            let inspected = match exchange(
                b,
                ScqRequest::InspectAttempt(InspectAttemptRequest { attempt: a }),
            )? {
                Response::AttemptInspected(Some(v)) => v,
                _ => anyhow::bail!("original attempt absent"),
            };
            let path = b.cgroup_root.join(format!(
                "scq-{}-attempt-{}-fence-{}",
                a.job_id, a.number, a.fence
            ));
            ensure!(
                inspected.record.attempt == a
                    && inspected.record.pid == allocation.leader.as_ref().map(|p| p.pid)
                    && inspected.record.finished_at.is_some()
                    && inspected.active_process.is_none()
                    && inspected.record.cgroup_path.as_deref() == path.to_str(),
                "old attempt is not fully drained"
            );
            if path.exists() {
                ensure!(
                    fs::canonicalize(&path)? == path
                        && fs::read_to_string(path.join("cgroup.events"))?
                            .lines()
                            .any(|l| l == "populated 0"),
                    "old resident/worker subtree remains populated"
                );
            }
        }
        for original in &census.jobs {
            let allocations = census
                .allocations
                .iter()
                .filter(|a| a.attempt.job_id == original.id)
                .count();
            ensure!(
                allocations == original.attempt_count as usize,
                "original census omits an old native attempt or descendant allocation"
            );
        }
        for (id, service) in [
            (b.resident_job, "jope-resident-gpu"),
            (b.authority_job, "jope-fit-authority"),
        ] {
            let current = job(b, id)?;
            ensure!(
                hash(&serde_json::to_vec(&current.spec)?)
                    == *if id == b.authority_job {
                        &b.authority_job_spec_sha256
                    } else {
                        &b.resident_job_spec_sha256
                    },
                "reviewed native service JobSpec changed"
            );
            ensure!(
                current.spec.tenant == census.tenant
                    && current.spec.project == census.project
                    && current.state == jain_smartcluster::JobState::Running
                    && current.active_attempt.is_some()
                    && matches!(&current.spec.workload,Workload::TrustedService {service: s,..} if s.as_str()==service),
                "approved native service is not currently allocated"
            );
            let a = current.active_attempt.unwrap();
            ensure!(
                matches!(exchange(b,ScqRequest::InspectAttempt(InspectAttemptRequest{attempt:a}))?,
                Response::AttemptInspected(Some(i)) if i.active_process.is_some()),
                "approved service leader not live"
            );
        }
        // Enumerate every retained scoped job. An omitted original or an unknown
        // competing run cannot hide behind a caller-provided census subset.
        let mut cursor = None;
        let mut cursors = BTreeSet::new();
        let mut total = 0usize;
        loop {
            let page = match exchange(
                b,
                ScqRequest::List(ListRequest {
                    filter: ListFilter {
                        tenant: Some(census.tenant.clone()),
                        project: Some(census.project.clone()),
                        ..Default::default()
                    },
                    cursor: cursor.clone(),
                    limit: 256,
                }),
            )? {
                Response::Jobs(p) => p,
                _ => anyhow::bail!("native census query failed"),
            };
            for j in &page.jobs {
                total += 1;
                ensure!(total <= 100_000, "native census bound exceeded");
                if j.id == b.resident_job || j.id == b.authority_job {
                    continue;
                }
                if j.created_at.as_unix_millis() <= census.captured_unix_ms {
                    ensure!(
                        old.contains(&j.id),
                        "original native job omitted from reviewed census"
                    );
                }
                if !j.state.is_terminal() || j.active_attempt.is_some() {
                    ensure!(
                        !old.contains(&j.id)
                            && j.spec.labels.get("jope.run") == Some(&self.config.run),
                        "unreviewed competing native work remains eligible"
                    );
                    let generation: u64 = j
                        .spec
                        .labels
                        .get("jope.generation")
                        .context("native generation absent")?
                        .parse()?;
                    let id = j
                        .spec
                        .labels
                        .get("jope.task")
                        .context("native task absent")?;
                    let request = self.original_request(generation, id)?;
                    let binding = self
                        .binding(&request)?
                        .context("active native job lacks original Loop binding")?;
                    ensure!(
                        binding.admission.delivery == Delivery::NativeJob
                            && j.spec
                                == jain_smartcluster::jope::job_spec(
                                    &crate::manifest::resolve(
                                        &self.root,
                                        &binding.admission.manifest.path
                                    )?,
                                    &binding.admission.manifest.sha256,
                                    census.tenant.clone(),
                                    census.project.clone(),
                                    Timestamp::from(0)
                                )?,
                        "active native allocation differs from exact original bound task"
                    );
                    let submitted = self
                        .graph_state
                        .join("tasks")
                        .join(identity(&request.run, generation, id)?)
                        .join("submitted.json");
                    let original: Value = serde_json::from_slice(&read(&submitted)?)?;
                    ensure!(
                        original["job_id"] == serde_json::to_value(j.id)?
                            && original["admission_sha256"]
                                == hash(&serde_json::to_vec(&binding.admission)?),
                        "active native job is not original graph dispatch"
                    );
                }
            }
            match page.next_cursor {
                Some(next) => {
                    ensure!(
                        cursors.insert(next.clone()),
                        "native census cursor repeated"
                    );
                    cursor = Some(next)
                }
                None => break,
            }
        }
        verify_signature(&signed, now()?)?;
        Ok(b.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    #[test]
    fn run_review_binds_entire_config_epoch_census_and_original_deadline() {
        let a = Artifact {
            path: "a.json".into(),
            sha256: "a".repeat(64),
        };
        let body = RunApprovalBody {
            schema: "jope.native_run_approval.v1".into(),
            config: a.clone(),
            scientific_policy: a.clone(),
            root: "/root/data".into(),
            loop_state: "/root/data/loop".into(),
            graph_state: "/root/data/graph".into(),
            worker_state: "/root/data/worker".into(),
            run: "run".into(),
            authority_root: "/root/data/authority".into(),
            authority_witness: "/root/data/watermark".into(),
            authority_instance: "a".repeat(64),
            authority_epoch: "a".repeat(64),
            recovery_receipt_sha256: None,
            original_job_census: a,
            daemon_socket: "/run/scq.sock".into(),
            daemon_uid: 1000,
            daemon_executable_sha256: "b".repeat(64),
            cgroup_root: "/sys/fs/cgroup/native".into(),
            resident_job: JobId::new(),
            authority_job: JobId::new(),
            authority_job_spec_sha256: "a".repeat(64),
            resident_job_spec_sha256: "b".repeat(64),
            loop_executable_sha256: "c".repeat(64),
            eligibility_executable_sha256: BTreeSet::from(["d".repeat(64)]),
            not_before_unix_ms: 100,
            expires_unix_ms: 200,
        };
        let signing = SigningKey::from_bytes(&[83; 32]);
        let bytes = serde_json::to_vec(&body).unwrap();
        let mut message = DOMAIN.to_vec();
        message.extend(&bytes);
        let signed = SignedRunApproval {
            authorization_public_key: hex::encode(signing.verifying_key().to_bytes()),
            body_sha256: hash(&bytes),
            body,
            signature: hex::encode(signing.sign(&message).to_bytes()),
        };
        verify_signature(&signed, 100).unwrap();
        assert!(verify_signature(&signed, 200).is_err());
        let mut changed = signed.clone();
        changed.body.config.sha256 = "f".repeat(64);
        assert!(verify_signature(&changed, 150).is_err());
        let mut changed = signed.clone();
        changed.body.authority_epoch = "e".repeat(64);
        assert!(verify_signature(&changed, 150).is_err());
        let mut changed = signed.clone();
        changed.body.original_job_census.sha256 = "e".repeat(64);
        assert!(verify_signature(&changed, 150).is_err());
        let mut changed = signed;
        changed.signature = hex::encode(signing.sign(&bytes).to_bytes());
        assert!(verify_signature(&changed, 150).is_err());
    }
}
