//! Closed native scheduler proof. Legacy namespace/funding routes stay held.
#[path = "native_scq/policy.rs"]
pub(crate) mod policy;
use super::{service::ServiceInputs, ProcessIdentity, RequestProvenance, VerifiedFit};
use anyhow::{ensure, Context, Result};
use jain_smartcluster::{
    AttachTrustedFitChildRequest, ClientFrame, FencedAttempt, FrameCodec, GetRequest,
    IdempotencyKey, JobId, JobRecord, JobState, ProtocolVersion, Request, RequestEnvelope,
    Response, ServerFrame, TrustedFitChildPlacement, Validate, Workload,
};
use policy::{hash, InstalledPolicy};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::os::unix::fs::MetadataExt;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeRequest {
    pub job_id: JobId,
    pub candidate_index: u32,
    pub descriptor_sha256: String,
    pub deadline_monotonic_ns: u64,
    pub destination: PathBuf,
}
/// Deliberately not deserializable or constructible by any client.
pub(crate) struct NativeAdmission {
    policy: Arc<InstalledPolicy>,
    peer: ProcessIdentity,
    target_leader: ProcessIdentity,
    authority_leader: ProcessIdentity,
    pub(crate) target: FencedAttempt,
    pub(crate) authority: FencedAttempt,
    pub(crate) task_sha256: String,
    pub(crate) descriptor_sha256: String,
    pub(crate) candidate_index: u32,
    pub(crate) task_key: String,
    pub(crate) slot_key: String,
    pub(crate) fit_key: String,
    pub(crate) deadline_monotonic_ns: u64,
    pub(crate) task_deadline_ms: u64,
    pub(crate) task_budget: u32,
    pub(crate) run: String,
    pub(crate) generation: u64,
    pub(crate) task_id: String,
    pub(crate) placement: Option<TrustedFitChildPlacement>,
}
fn exchange(policy: &InstalledPolicy, request: Request) -> Result<Response> {
    policy.revalidate()?;
    let grant = &policy.signed.body;
    authenticated_exchange(
        &grant.daemon_socket,
        grant.daemon_uid,
        &grant.daemon_executable_sha256,
        None,
        request,
    )
}
pub(crate) fn authenticated_exchange(
    socket_path: &Path,
    uid: u32,
    executable_sha256: &str,
    expected: Option<&ProcessIdentity>,
    request: Request,
) -> Result<Response> {
    // Callers include both synchronous native services and async graph/proof
    // drivers. Keep authentication on the actual stream without nesting Tokio
    // runtimes on a runtime thread.
    if tokio::runtime::Handle::try_current().is_ok() {
        return std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    authenticated_exchange(socket_path, uid, executable_sha256, expected, request)
                })
                .join()
                .map_err(|_| anyhow::anyhow!("authenticated exchange thread failed"))?
        });
    }
    request.validate()?;
    let stream = std::os::unix::net::UnixStream::connect(socket_path)?;
    let peer = crate::resident_protocol::peer_credentials(&stream)?;
    let before = ProcessIdentity::read(peer.pid as u32)?;
    // Pin the actual executable inode through hashing and the complete exchange.
    let executable = fs::File::open(format!("/proc/{}/exe", peer.pid))?;
    let pinned = PathBuf::from(format!(
        "/proc/self/fd/{}",
        std::os::fd::AsRawFd::as_raw_fd(&executable)
    ));
    ensure!(
        peer.uid == uid
            && crate::manifest::file_hash(&pinned)? == executable_sha256
            && expected.is_none_or(|value| *value == before)
            && ProcessIdentity::read(peer.pid as u32)? == before,
        "native daemon peer not qualified"
    );
    stream.set_nonblocking(true)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let response = runtime.block_on(async {
        let mut socket = tokio::net::UnixStream::from_std(stream)?;
        let envelope = RequestEnvelope::new(request);
        let id = envelope.request_id;
        let codec = FrameCodec::default();
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            codec
                .write_frame(&mut socket, &ClientFrame::Request(Box::new(envelope)))
                .await?;
            let Some(ServerFrame::Response(response)) =
                codec.read_frame::<ServerFrame, _>(&mut socket).await?
            else {
                anyhow::bail!("native daemon response missing")
            };
            ensure!(
                response.request_id == id
                    && ProtocolVersion::CURRENT.is_compatible_with(response.version),
                "native daemon response identity/protocol mismatch"
            );
            response.result.map_err(anyhow::Error::from)
        })
        .await?
    })?;
    let original_exe = executable.metadata()?;
    let current_exe = fs::metadata(format!("/proc/{}/exe", peer.pid))?;
    ensure!(
        ProcessIdentity::read(peer.pid as u32)? == before
            && original_exe.dev() == current_exe.dev()
            && original_exe.ino() == current_exe.ino(),
        "native daemon changed during exchange"
    );
    Ok(response)
}
fn current_job(policy: &InstalledPolicy, id: JobId) -> Result<JobRecord> {
    match exchange(policy, Request::Get(GetRequest { job_id: id }))? {
        Response::Job(record) => record.context("native allocation missing"),
        _ => anyhow::bail!("native job response differs"),
    }
}

pub(crate) fn active_process(
    socket: &Path,
    uid: u32,
    executable_sha256: &str,
    daemon: Option<&ProcessIdentity>,
    attempt: FencedAttempt,
) -> Result<ProcessIdentity> {
    let response = authenticated_exchange(
        socket,
        uid,
        executable_sha256,
        daemon,
        Request::InspectAttempt(jain_smartcluster::InspectAttemptRequest { attempt }),
    )?;
    let Response::AttemptInspected(Some(inspection)) = response else {
        anyhow::bail!("native active attempt inspection absent")
    };
    inspected_process(inspection, attempt)
}
fn inspected_process(
    inspection: jain_smartcluster::AttemptInspection,
    attempt: FencedAttempt,
) -> Result<ProcessIdentity> {
    inspection.record.validate()?;
    ensure!(
        inspection.record.attempt == attempt
            && matches!(
                inspection.record.state,
                JobState::Starting | JobState::Running
            ),
        "native active attempt inspection is stale/not running"
    );
    let process = inspection
        .active_process
        .context("native original leader is not active")?;
    ensure!(
        inspection.record.pid == Some(process.pid) && inspection.record.finished_at.is_none(),
        "native active process differs from original attempt record"
    );
    let process = ProcessIdentity {
        pid: process.pid,
        birth_ticks: process.birth_ticks,
        boot_id: process.boot_id,
    };
    ensure!(
        ProcessIdentity::read(process.pid)? == process,
        "native original leader identity changed"
    );
    process_parent(process.pid)?;
    ensure!(
        inspection.record.cgroup_path.as_deref() == cgroup(process.pid)?.to_str(),
        "native inspected leader cgroup changed"
    );
    Ok(process)
}
fn native_leader(policy: &InstalledPolicy, attempt: FencedAttempt) -> Result<ProcessIdentity> {
    let grant = &policy.signed.body;
    active_process(
        &grant.daemon_socket,
        grant.daemon_uid,
        &grant.daemon_executable_sha256,
        None,
        attempt,
    )
}

fn process_parent(pid: u32) -> Result<u32> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let mut fields = stat
        .rsplit_once(')')
        .context("native ancestry malformed stat")?
        .1
        .split_whitespace();
    let state = fields.next().context("native ancestry state absent")?;
    ensure!(
        !matches!(state, "Z" | "X" | "x"),
        "native requester ancestry process exited"
    );
    fields
        .next()
        .context("native ancestry parent absent")?
        .parse()
        .context("native ancestry parent malformed")
}

fn verify_ancestry(requester: &ProcessIdentity, leader: &ProcessIdentity) -> Result<()> {
    ensure!(
        requester.boot_id == leader.boot_id && ProcessIdentity::read(leader.pid)? == *leader,
        "native leader identity changed"
    );
    let mut current = requester.clone();
    let mut chain: Vec<(ProcessIdentity, u32)> = Vec::new();
    for _ in 0..32 {
        ensure!(
            ProcessIdentity::read(current.pid)? == current,
            "native requester ancestry changed"
        );
        if current == *leader {
            for (process, parent) in chain {
                ensure!(
                    ProcessIdentity::read(process.pid)? == process
                        && process_parent(process.pid)? == parent,
                    "native requester ancestry changed after observation"
                );
            }
            ensure!(
                ProcessIdentity::read(leader.pid)? == *leader,
                "native leader exited during ancestry validation"
            );
            return Ok(());
        }
        let parent = process_parent(current.pid)?;
        ensure!(
            parent > 0 && parent != current.pid,
            "requester is not a native leader descendant"
        );
        chain.push((current, parent));
        current = ProcessIdentity::read(parent)?;
    }
    anyhow::bail!("native requester ancestry exceeds bound")
}

fn cgroup(pid: u32) -> Result<PathBuf> {
    let text = fs::read_to_string(format!("/proc/{pid}/cgroup"))?;
    let name = text
        .lines()
        .find_map(|l| l.strip_prefix("0::"))
        .context("unified native cgroup absent")?;
    Ok(Path::new("/sys/fs/cgroup").join(name.trim_start_matches('/')))
}
fn attempt_path(policy: &InstalledPolicy, attempt: FencedAttempt) -> PathBuf {
    policy.signed.body.cgroup_root.join(format!(
        "scq-{}-attempt-{}-fence-{}",
        attempt.job_id, attempt.number, attempt.fence
    ))
}
fn live(record: &JobRecord) -> Result<FencedAttempt> {
    ensure!(
        record.state == JobState::Running,
        "native allocation not live"
    );
    record.active_attempt.context("native attempt absent")
}
fn service(record: &JobRecord) -> Result<(&str, &[String])> {
    match &record.spec.workload {
        Workload::TrustedService {
            service, arguments, ..
        } => Ok((service.as_str(), arguments)),
        _ => anyhow::bail!("native fit requires fixed trusted service"),
    }
}
fn monotonic_ns() -> Result<u64> {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    ensure!(
        // SAFETY: `t` is a live writable timespec and CLOCK_MONOTONIC has no
        // pointer preconditions beyond that output buffer.
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) } == 0,
        "monotonic clock unavailable"
    );
    (t.tv_sec as u64)
        .checked_mul(1_000_000_000)
        .and_then(|v| v.checked_add(t.tv_nsec as u64))
        .context("clock overflow")
}
pub(crate) fn verify_request(
    policy: Arc<InstalledPolicy>,
    peer: &ProcessIdentity,
    request: &NativeRequest,
    inputs: &ServiceInputs,
    fit: &VerifiedFit,
) -> Result<NativeAdmission> {
    policy.revalidate()?;
    // Policy selection has no caller pathname: the exact digest comes from the
    // same authenticated scheduler query used for native allocation admission.
    let bootstrap_job = current_job(&policy, request.job_id)?;
    let (bootstrap_name, bootstrap_args) = service(&bootstrap_job)?;
    ensure!(
        matches!(bootstrap_name, "jope-gp-search" | "jope-continuation")
            && bootstrap_args.len() == 4
            && bootstrap_args[0] == "--task-manifest"
            && bootstrap_args[2] == "--manifest-sha256",
        "fixed native task policy selection required"
    );
    let policy = Arc::new(
        policy
            .task_policy(&bootstrap_args[3])?
            .context("exact task policy is not independently authorized yet")?,
    );
    let grant = &policy.signed.body;
    ensure!(
        ProcessIdentity::read(peer.pid)? == *peer,
        "requesting process birth changed"
    );
    ensure!(
        crate::manifest::file_hash(Path::new(&format!("/proc/{}/exe", peer.pid)))?
            == grant.evaluator_executable_sha256,
        "native fit requester executable differs"
    );
    ensure!(
        crate::manifest::file_hash(Path::new("/proc/self/exe"))?
            == grant.authority_executable_sha256,
        "native authority executable differs"
    );
    let requester = current_job(&policy, request.job_id)?;
    let target = live(&requester)?;
    let target_leader = native_leader(&policy, target)?;
    verify_ancestry(peer, &target_leader)?;
    ensure!(
        requester.spec.tenant.as_str() == grant.tenant
            && requester.spec.project.as_str() == grant.project,
        "native requester outside authorized tenant/project"
    );
    let (name, args) = service(&requester)?;
    ensure!(
        matches!(name, "jope-gp-search" | "jope-continuation")
            && args.len() == 4
            && args[0] == "--task-manifest"
            && args[2] == "--manifest-sha256",
        "native request is not fixed GP/continuation manifest"
    );
    ensure!(
        cgroup(peer.pid)? == attempt_path(&policy, target),
        "requester outside exact native attempt cgroup"
    );
    let (task, input) =
        crate::smartcluster_worker::verify_task_inputs(&grant.root, Path::new(&args[1]), &args[3])?;
    ensure!(
        grant.approved_task_sha256.contains(&args[3]) && task.kind.service() == name,
        "native task absent from independently signed grant"
    );
    let search_binding = match &input.operation {
        crate::smartcluster_worker::Operation::Search {
            prepared,
            cut,
            output,
            evaluations,
            seconds,
            qualification,
            seeds,
            ..
        } => (
            prepared.clone(),
            *cut,
            output.clone(),
            *evaluations,
            *seconds,
            *qualification,
            seeds.is_some(),
        ),
        crate::smartcluster_worker::Operation::DirectEvaluation { manifest, sha256 }
        | crate::smartcluster_worker::Operation::SeededContinuation { manifest, sha256 } => {
            let reference = crate::native_graph::Artifact {
                path: manifest.clone(),
                sha256: sha256.clone(),
            };
            let genome_task: crate::native_genome::GenomeTask =
                crate::native_genome::load(&grant.root, &reference)?;
            let candidate = genome_task.candidate(&grant.root)?;
            ensure!(
                candidate.run == task.run && candidate.generation == task.generation,
                "native genome candidate belongs to another run/generation"
            );
            if genome_task.stage == crate::native_genome::Stage::Direct {
                ensure!(
                    fit.identity().canonical_program == candidate.genome
                        && request.candidate_index == 0,
                    "direct allocation may fit only its exact original genome"
                );
            } else {
                let _ = crate::native_genome::direct_parent(&grant.root, &genome_task, &candidate)?;
            }
            (
                candidate.context.prepared,
                candidate.context.cut,
                genome_task.output,
                genome_task.evaluations,
                genome_task.seconds,
                genome_task.qualification,
                genome_task.stage == crate::native_genome::Stage::Continuation,
            )
        }
        _ => anyhow::bail!("native fit task is not an implemented numerical operation"),
    };
    let (prepared, cut, output, evaluations, seconds, qualification, continuation) =
        &search_binding;
    ensure!(
        (name == "jope-continuation") == *continuation
            && *evaluations <= grant.max_candidates_per_task as usize
            && request.candidate_index < *evaluations as u32,
        "native candidate/service budget exceeded"
    );
    ensure!(
        *seconds > 0 && *seconds <= 600,
        "native Search clock bound invalid"
    );
    let prepared = crate::manifest::resolve(&grant.root, prepared)?;
    let metadata: crate::prepare::Prepared =
        crate::manifest::read_json(&prepared.join("prepared.json"))?;
    ensure!(
        metadata.meta_partition == "meta_train",
        "sealed/nontraining data cannot enter native fits"
    );
    ensure!(
        grant.capability != "synthetic_qualification"
            || (*qualification && metadata.qualification_only),
        "synthetic grant cannot authorize production data"
    );
    let rows = crate::prepare::load_view(&prepared, *cut, true)?;
    let x: Vec<Vec<f64>> = rows.iter().map(|r| r[..r.len() - 1].to_vec()).collect();
    let y: Vec<f64> = rows.iter().map(|r| *r.last().unwrap()).collect();
    let preprocessing = fs::read(prepared.join(format!("cut-{cut}/preprocessing.json")))?;
    let feature_view =
        crate::gp::raw_feature_view(&hash(&crate::prepare::csv_bytes(&rows)?), &preprocessing)?;
    ensure!(
        inputs.x == x
            && inputs.y == y
            && inputs.preprocessing == preprocessing
            && inputs.feature_view == feature_view,
        "sealed fit payload differs from authorized prepared cut"
    );
    ensure!(
        hash(&serde_json::to_vec(inputs)?) == request.descriptor_sha256,
        "native descriptor digest changed"
    );
    ensure!(
        grant
            .training_sha256
            .contains(&fit.identity().training_sha256)
            && grant.numerical_contract_sha256 == fit.identity().numerical_contract_sha256
            && fit.identity().fit_seed == grant.fit_seed
            && grant
                .allowed_runtime_sha256
                .contains(&fit.identity().runtime_sha256),
        "native numerical identity/runtime outside grant"
    );
    let authority_id: JobId = std::env::var("SCQ_JOB_ID")
        .context("fit authority lacks native controller allocation")?
        .parse()?;
    let authority_record = current_job(&policy, authority_id)?;
    let authority = live(&authority_record)?;
    let authority_leader = native_leader(&policy, authority)?;
    ensure!(
        authority_leader == ProcessIdentity::current()?,
        "native authority is not its original scheduler leader"
    );
    ensure!(
        service(&authority_record)?.0 == "jope-fit-authority"
            && authority_record.spec.tenant == requester.spec.tenant
            && authority_record.spec.project == requester.spec.project
            && cgroup(std::process::id())? == attempt_path(&policy, authority),
        "fit authority is not its fixed native controller allocation"
    );
    let now = policy::now_ms()?;
    let remaining = task
        .deadline
        .as_unix_millis()
        .min(grant.period_end_unix_ms)
        .checked_sub(now)
        .context("original task deadline expired")?;
    let max_duration = remaining.min(grant.max_fit_ms);
    let clock = monotonic_ns()?;
    let ceiling = clock
        .checked_add(
            max_duration
                .checked_mul(1_000_000)
                .context("deadline overflow")?,
        )
        .context("deadline overflow")?;
    ensure!(
        request.deadline_monotonic_ns > clock && request.deadline_monotonic_ns <= ceiling,
        "native fit deadline exceeds original bound"
    );
    ensure!(
        ProcessIdentity::read(peer.pid)? == *peer,
        "native requester changed during proof"
    );
    let task_key = hash(&serde_json::to_vec(&(
        &task.run,
        task.generation,
        &task.task_id,
    ))?);
    let slot_key = hash(&serde_json::to_vec(&(&task_key, request.candidate_index))?);
    ensure!(
        request.destination
            == crate::manifest::resolve(&grant.root, output)?
                .join(format!("candidate-{}.json", request.candidate_index)),
        "native candidate destination outside authorized output"
    );
    let task_sha256 = args[3].clone();
    let task_budget = *evaluations as u32;
    Ok(NativeAdmission {
        policy,
        peer: peer.clone(),
        target_leader,
        authority_leader,
        target,
        authority,
        task_sha256,
        descriptor_sha256: request.descriptor_sha256.clone(),
        candidate_index: request.candidate_index,
        task_key,
        slot_key,
        fit_key: fit.key().into(),
        deadline_monotonic_ns: request.deadline_monotonic_ns,
        task_deadline_ms: task.deadline.as_unix_millis(),
        task_budget,
        run: task.run,
        generation: task.generation,
        task_id: task.task_id,
        placement: None,
    })
}
impl NativeAdmission {
    pub(crate) fn scope(&self) -> &str {
        &self.policy.signed.body.scope
    }
    pub(crate) fn authority_instance(&self) -> &str {
        &self.policy.signed.body.authority_instance
    }
    pub(crate) fn authority_epoch(&self) -> &str {
        &self.policy.signed.body.authority_epoch
    }
    pub(crate) fn authority_manifest_sha256(&self) -> &str {
        &self.policy.signed.body.authority_manifest_sha256
    }
    pub(crate) fn coverage_sha256(&self) -> &str {
        &self.policy.signed.body.coverage_receipt_sha256
    }
    pub(crate) fn revalidate(&self) -> Result<()> {
        self.policy.revalidate()?;
        ensure!(
            ProcessIdentity::read(self.peer.pid)? == self.peer,
            "native requester died/changed"
        );
        let target = current_job(&self.policy, self.target.job_id)?;
        let authority = current_job(&self.policy, self.authority.job_id)?;
        ensure!(
            native_leader(&self.policy, self.target)? == self.target_leader
                && native_leader(&self.policy, self.authority)? == self.authority_leader
                && ProcessIdentity::current()? == self.authority_leader,
            "native original scheduler leaders changed"
        );
        verify_ancestry(&self.peer, &self.target_leader)?;
        ensure!(
            live(&target)? == self.target
                && live(&authority)? == self.authority
                && cgroup(self.peer.pid)? == attempt_path(&self.policy, self.target),
            "native fit attempt is stale"
        );
        ensure!(
            monotonic_ns()? < self.deadline_monotonic_ns,
            "native original fit deadline expired"
        );
        Ok(())
    }
    pub(crate) fn provenance(&self) -> RequestProvenance {
        RequestProvenance {
            scope: self.scope().into(),
            task_id: format!(
                "{}/{}/{}/candidate-{}",
                self.run, self.generation, self.task_id, self.candidate_index
            ),
            // Preserve authenticated original allocation custody before a claim can
            // become held; witness denial may prevent any later authorization event.
            details: json!({"native_task":self.task_key,"native_slot":self.slot_key,"deadline_monotonic_ns":self.deadline_monotonic_ns,"requesting_process":self.peer,
                "native_allocation":{"schema":"jope.native_claim_allocation.v1","authority_attempt":self.authority,"target_attempt":self.target,
                    "authority_leader":self.authority_leader,"target_leader":self.target_leader,
                    "task_manifest_sha256":self.task_sha256,"descriptor_sha256":self.descriptor_sha256,
                    "grant_sha256":self.policy.trust.grant_sha256}}),
        }
    }
    pub(crate) fn task_record(&self) -> Value {
        json!({"task_manifest_sha256":self.task_sha256,"scope":self.scope(),"budget":self.task_budget,"deadline_unix_ms":self.task_deadline_ms,"grant_sha256":self.policy.trust.grant_sha256,"run":self.run,"generation":self.generation,"task_id":self.task_id})
    }
    pub(crate) fn slot_record(&self) -> Value {
        json!({"task_key":self.task_key,"candidate_index":self.candidate_index,"fit_key":self.fit_key,"descriptor_sha256":self.descriptor_sha256,"deadline_monotonic_ns":self.deadline_monotonic_ns,"boot_id":self.peer.boot_id})
    }
    pub(crate) fn reserve_deadline(&mut self, stored: &Value) -> Result<()> {
        ensure!(
            stored["fit_key"] == self.fit_key
                && stored["descriptor_sha256"] == self.descriptor_sha256
                && stored["task_key"] == self.task_key
                && stored["candidate_index"] == self.candidate_index
                && stored["boot_id"] == self.peer.boot_id,
            "conflicting original native candidate"
        );
        self.deadline_monotonic_ns = self.deadline_monotonic_ns.min(
            stored["deadline_monotonic_ns"]
                .as_u64()
                .context("original slot deadline absent")?,
        );
        Ok(())
    }
    pub(crate) fn attach_child(&mut self, child: &ProcessIdentity) -> Result<()> {
        self.revalidate()?;
        let identity = hash(&serde_json::to_vec(&(
            self.authority,
            self.target,
            &self.descriptor_sha256,
            self.candidate_index,
            child,
        ))?);
        let request = AttachTrustedFitChildRequest {
            authority_attempt: self.authority,
            target_attempt: self.target,
            child_pid: child.pid,
            child_birth_ticks: child.birth_ticks,
            child_boot_id: child.boot_id.clone(),
            child_executable_sha256: self.policy.signed.body.evaluator_executable_sha256.clone(),
            task_manifest_sha256: self.task_sha256.clone(),
            descriptor_sha256: self.descriptor_sha256.clone(),
            candidate_index: self.candidate_index,
            idempotency_key: IdempotencyKey::new(identity)?,
        };
        let placed = match exchange(
            &self.policy,
            Request::AttachTrustedFitChild(Box::new(request)),
        )? {
            Response::TrustedFitChildPlaced(p) => p,
            _ => anyhow::bail!("native placement response differs"),
        };
        verify_placement(
            &placed,
            child,
            self.target,
            &self.task_sha256,
            &self.descriptor_sha256,
            self.candidate_index,
        )?;
        ensure!(
            placed.child_executable_sha256 == self.policy.signed.body.evaluator_executable_sha256
                && placed.authority_attempt == self.authority
                && placed.cgroup_path == attempt_path(&self.policy, self.target)
                && cgroup(child.pid)? == placed.cgroup_path,
            "child placement does not match exact native allocation"
        );
        self.placement = Some(placed);
        self.revalidate()
    }
    pub(crate) fn event(&self) -> Result<Value> {
        self.revalidate()?;
        Ok(
            json!({"schema":"jope.native_fit_authorization.v1","provenance":self.provenance(),"signed_grant":self.policy.signed,"native_grant_proof":self.policy.native_grant_proof,"requesting_process":self.peer,"target_leader":self.target_leader,"authority_leader":self.authority_leader,"policy_sha256":self.policy.policy_sha256(),"authorized_unix_ms":policy::now_ms()?,"target_attempt":self.target,"authority_attempt":self.authority,"task_manifest_sha256":self.task_sha256,"descriptor_sha256":self.descriptor_sha256,"candidate_index":self.candidate_index,"task_key":self.task_key,"slot_key":self.slot_key,"fit_key":self.fit_key,"deadline_monotonic_ns":self.deadline_monotonic_ns,"placement":self.placement.as_ref().context("native numerical child not placed")?}),
        )
    }
}
pub(crate) fn verify_placement(
    p: &TrustedFitChildPlacement,
    child: &ProcessIdentity,
    target: FencedAttempt,
    task: &str,
    descriptor: &str,
    candidate: u32,
) -> Result<()> {
    let mut value = serde_json::to_value(p)?;
    value
        .as_object_mut()
        .context("placement object")?
        .remove("receipt_sha256");
    ensure!(
        hash(&serde_json::to_vec(&value)?) == p.receipt_sha256
            && p.schema == "jope.smartcluster_fit_placement.v1"
            && p.target_attempt == target
            && p.child_pid == child.pid
            && p.child_birth_ticks == child.birth_ticks
            && p.child_boot_id == child.boot_id
            && p.task_manifest_sha256 == task
            && p.descriptor_sha256 == descriptor
            && p.candidate_index == candidate,
        "native child placement receipt mismatch"
    );
    ensure!(
        crate::manifest::file_hash(Path::new(&format!("/proc/{}/exe", child.pid)))?
            == p.child_executable_sha256
            && ProcessIdentity::read(child.pid)? == *child
            && cgroup(child.pid)? == p.cgroup_path,
        "native child birth/cgroup changed"
    );
    Ok(())
}
/// Witness checks signed current policy provenance, never a caller's boolean.
pub(crate) fn verify_witness_event(
    trust: &policy::NativeWitnessTrust,
    event: &Value,
    identity: &super::FitIdentity,
    start: &super::StartToken,
    witness: &super::witness::WitnessPolicy,
) -> Result<()> {
    ensure!(
        event["schema"] == "jope.native_fit_authorization.v1",
        "native witness authorization schema differs"
    );
    let signed: policy::SignedNativeGrant = serde_json::from_value(event["signed_grant"].clone())?;
    policy::verify_signed(
        trust,
        &signed,
        event["authorized_unix_ms"]
            .as_u64()
            .context("original native authorization time absent")?,
    )?;
    let grant = &signed.body;
    ensure!(
        event["provenance"]["scope"] == grant.scope
            && event["provenance"]["task_id"] == start.task_id
            && identity.fit_seed == grant.fit_seed,
        "native witness provenance differs"
    );
    ensure!(
        grant.authority_instance == witness.authority_instance
            && grant.authority_epoch == witness.authority_epoch
            && grant.authority_executable_sha256 == witness.authority_executable_sha256
            && grant.witness_public_key == witness.witness_public_key
            && grant
                .allowed_runtime_sha256
                .contains(&identity.runtime_sha256)
            && grant.training_sha256.contains(&identity.training_sha256)
            && grant.numerical_contract_sha256 == identity.numerical_contract_sha256,
        "native witness grant/authority/fit binding mismatch"
    );
    let task = event["task_manifest_sha256"]
        .as_str()
        .context("native task digest absent")?;
    ensure!(
        grant.approved_task_sha256.contains(task)
            && event["candidate_index"]
                .as_u64()
                .context("native candidate absent")?
                < grant.max_candidates_per_task as u64
            && event["fit_key"] == start.fit_key
            && event["deadline_monotonic_ns"]
                .as_u64()
                .context("native deadline absent")?
                >= start.deadline_monotonic_ns,
        "native witness task/budget/start differs"
    );
    let placement: TrustedFitChildPlacement = serde_json::from_value(event["placement"].clone())?;
    let mut body = serde_json::to_value(&placement)?;
    body.as_object_mut().unwrap().remove("receipt_sha256");
    ensure!(
        placement.schema == "jope.smartcluster_fit_placement.v1"
            && placement.child_executable_sha256 == grant.evaluator_executable_sha256
            && placement.cgroup_path
                == grant.cgroup_root.join(format!(
                    "scq-{}-attempt-{}-fence-{}",
                    placement.target_attempt.job_id,
                    placement.target_attempt.number,
                    placement.target_attempt.fence
                ))
            && hash(&serde_json::to_vec(&body)?) == placement.receipt_sha256
            && placement.child_pid == start.executor.pid
            && placement.child_birth_ticks == start.executor.birth_ticks
            && placement.child_boot_id == start.executor.boot_id
            && event["target_attempt"] == serde_json::to_value(placement.target_attempt)?
            && event["authority_attempt"] == serde_json::to_value(placement.authority_attempt)?
            && placement.task_manifest_sha256 == task
            && event["descriptor_sha256"] == placement.descriptor_sha256
            && event["candidate_index"] == placement.candidate_index,
        "native witness original placement differs"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn daemon_exchange_uses_authenticated_connection_even_if_path_replaced() {
        let dir = std::env::temp_dir().join(format!("jope-native-peer-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let socket = dir.join("daemon.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let identity = ProcessIdentity::current().unwrap();
        let digest = crate::manifest::file_hash(Path::new("/proc/self/exe")).unwrap();
        std::thread::scope(|scope| {
            let path = socket.clone();
            let server = scope.spawn(move || {
                let (connection, _) = listener.accept().unwrap();
                fs::remove_file(&path).unwrap();
                let _replacement = std::os::unix::net::UnixListener::bind(&path).unwrap();
                connection.set_nonblocking(true).unwrap();
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                runtime.block_on(async {
                    let mut connection = tokio::net::UnixStream::from_std(connection).unwrap();
                    let codec = FrameCodec::default();
                    let Some(ClientFrame::Request(request)) = codec
                        .read_frame::<ClientFrame, _>(&mut connection)
                        .await
                        .unwrap()
                    else {
                        panic!("request absent")
                    };
                    codec
                        .write_frame(
                            &mut connection,
                            &ServerFrame::Response(Box::new(
                                jain_smartcluster::ResponseEnvelope::ok(
                                    request.request_id,
                                    Response::Job(None),
                                ),
                            )),
                        )
                        .await
                        .unwrap();
                });
            });
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let result = runtime.block_on(async {
                authenticated_exchange(
                    &socket,
                    // SAFETY: geteuid has no preconditions and returns a scalar.
                    unsafe { libc::geteuid() },
                    &digest,
                    Some(&identity),
                    Request::Get(GetRequest {
                        job_id: JobId::new(),
                    }),
                )
                .unwrap()
            });
            assert!(matches!(result, Response::Job(None)));
            server.join().unwrap();
        });
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn daemon_exchange_rejects_wrong_executable_before_sending_request() {
        let dir =
            std::env::temp_dir().join(format!("jope-native-wrong-peer-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let socket = dir.join("daemon.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let (finished_tx, finished_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let server = scope.spawn(move || {
                use std::io::Read;
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                // Hashing the real debug image can exceed five seconds under I/O load.
                // Begin the no-request assertion only after authentication returned.
                finished_rx.recv().unwrap();
                let mut byte = [0];
                assert_eq!(stream.read(&mut byte).unwrap(), 0);
            });
            assert!(authenticated_exchange(
                &socket,
                // SAFETY: geteuid has no preconditions and returns a scalar.
                unsafe { libc::geteuid() },
                &"0".repeat(64),
                None,
                Request::Get(GetRequest {
                    job_id: JobId::new()
                })
            )
            .is_err());
            finished_tx.send(()).unwrap();
            server.join().unwrap();
        });
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn requester_ancestry_binds_original_leader_birth_and_rejects_sibling() {
        let leader = ProcessIdentity::current().unwrap();
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("5")
            .spawn()
            .unwrap();
        let mut sibling = std::process::Command::new("/bin/sleep")
            .arg("5")
            .spawn()
            .unwrap();
        let requester = ProcessIdentity::read(child.id()).unwrap();
        let other = ProcessIdentity::read(sibling.id()).unwrap();
        verify_ancestry(&requester, &leader).unwrap();
        assert!(verify_ancestry(&requester, &other).is_err());
        let mut stale = leader.clone();
        stale.birth_ticks += 1;
        assert!(verify_ancestry(&requester, &stale).is_err());
        let mut stale = requester.clone();
        stale.boot_id = "foreign-boot".into();
        assert!(verify_ancestry(&stale, &leader).is_err());
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(verify_ancestry(&requester, &leader).is_err());
        sibling.kill().unwrap();
        sibling.wait().unwrap();
    }
    #[test]
    fn inspected_finite_controller_preserves_original_leader_and_rejects_changed_fences() {
        use jain_smartcluster::{
            AttemptInspection, AttemptLease, AttemptNumber, AttemptRecord, FenceToken,
            NativeProcessIdentity, Timestamp,
        };
        let own = ProcessIdentity::current().unwrap();
        let attempt = FencedAttempt::new(
            JobId::new(),
            AttemptNumber::new(1).unwrap(),
            FenceToken::new(1).unwrap(),
        );
        let record = AttemptRecord {
            attempt,
            state: JobState::Running,
            lease: AttemptLease {
                attempt,
                acquired_at: Timestamp::from(1),
                expires_at: Timestamp::from(2),
            },
            pid: Some(own.pid),
            cgroup_path: Some(cgroup(own.pid).unwrap().to_string_lossy().into_owned()),
            started_at: Some(Timestamp::from(1)),
            finished_at: None,
            failure: None,
        };
        let inspection = AttemptInspection {
            record,
            active_process: Some(NativeProcessIdentity {
                pid: own.pid,
                birth_ticks: own.birth_ticks,
                boot_id: own.boot_id.clone(),
            }),
        };
        assert_eq!(inspected_process(inspection.clone(), attempt).unwrap(), own);
        // A replacement daemon may adopt this same finite controller; only the
        // unchanged original process/fence is accepted, never a replacement.
        assert_eq!(inspected_process(inspection.clone(), attempt).unwrap(), own);
        let stale = FencedAttempt::new(attempt.job_id, attempt.number, FenceToken::new(2).unwrap());
        assert!(inspected_process(inspection.clone(), stale).is_err());
        let mut changed = inspection.clone();
        changed.active_process.as_mut().unwrap().birth_ticks += 1;
        assert!(inspected_process(changed, attempt).is_err());
        let mut changed = inspection.clone();
        changed.record.state = JobState::Paused;
        assert!(inspected_process(changed, attempt).is_err());
        let mut changed = inspection.clone();
        changed.active_process = None;
        assert!(inspected_process(changed, attempt).is_err());
        let mut changed = inspection;
        changed.record.pid = Some(own.pid + 1);
        assert!(inspected_process(changed, attempt).is_err());
    }
}
