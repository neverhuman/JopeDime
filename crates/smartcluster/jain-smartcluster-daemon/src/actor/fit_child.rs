//! Only the root-qualified fit authority may place its own stopped numerical child.
use super::*;
use jain_smartcluster_core::{AttachTrustedFitChildRequest, TrustedFitChildPlacement, Workload};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
struct FitPlacementIntent {
    request: AttachTrustedFitChildRequest,
    authority_pid: u32,
    authority_birth: Option<u64>,
    authority_cgroup: PathBuf,
    target_cgroup: PathBuf,
}

fn refused(message: impl Into<String>) -> ServerError {
    ServerError::new(ErrorCode::Forbidden, message)
}
fn unavailable_fit(error: impl std::fmt::Display) -> ServerError {
    ServerError::new(ErrorCode::Unavailable, error.to_string())
}
fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn file_digest(path: &Path) -> Result<String, ServerError> {
    let mut file = fs::File::open(path).map_err(unavailable_fit)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(unavailable_fit)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(hex::encode(digest.finalize()))
}
fn bounded_json(path: &Path, sha: &str) -> Result<serde_json::Value, ServerError> {
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK)
        .open(path)
        .map_err(unavailable_fit)?;
    if !file.metadata().map_err(unavailable_fit)?.is_file() {
        return Err(refused("manifest is not regular"));
    }
    let mut bytes = Vec::new();
    file.take(1048577)
        .read_to_end(&mut bytes)
        .map_err(unavailable_fit)?;
    if bytes.len() > 1048576 || digest(&bytes) != sha {
        return Err(refused("manifest bytes changed or exceeded bound"));
    }
    serde_json::from_slice(&bytes).map_err(unavailable_fit)
}
#[derive(Debug, Eq, PartialEq)]
struct ProcessState {
    parent: u32,
    birth: u64,
    stopped: bool,
    dead: bool,
}
fn parse_stat(text: &str) -> Result<ProcessState, ServerError> {
    let (_, fields) = text
        .rsplit_once(") ")
        .ok_or_else(|| refused("invalid process stat"))?;
    let fields: Vec<_> = fields.split_whitespace().collect();
    if fields.len() < 20 {
        return Err(refused("short process stat"));
    }
    Ok(ProcessState {
        parent: fields[1].parse().map_err(unavailable_fit)?,
        birth: fields[19].parse().map_err(unavailable_fit)?,
        dead: matches!(fields[0], "Z" | "X" | "x"),
        stopped: fields[0] == "T", // traced children ('t') are not accepted
    })
}
fn proc_state(pid: u32) -> Result<ProcessState, ServerError> {
    parse_stat(&fs::read_to_string(format!("/proc/{pid}/stat")).map_err(unavailable_fit)?)
}
fn proc_cgroup(pid: u32) -> Result<PathBuf, ServerError> {
    let text = fs::read_to_string(format!("/proc/{pid}/cgroup")).map_err(unavailable_fit)?;
    let relative = text
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or_else(|| refused("unified cgroup absent"))?;
    Ok(Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/')))
}
fn service(job: &JobRecord) -> Result<(&str, &[String]), ServerError> {
    match &job.spec.workload {
        Workload::TrustedService {
            service, arguments, ..
        } => Ok((service.as_str(), arguments)),
        _ => Err(refused("native trusted service required")),
    }
}
fn live(job: &JobRecord, attempt: FencedAttempt) -> Result<(), ServerError> {
    if job.active_attempt != Some(attempt)
        || !matches!(job.state, JobState::Starting | JobState::Running)
        || job
            .spec
            .deadline
            .is_some_and(|deadline| deadline <= now_timestamp())
    {
        return Err(ServerError::new(
            ErrorCode::SupersededAttempt,
            "placement attempt is no longer live",
        ));
    }
    Ok(())
}
fn validate_candidate(
    job: &JobRecord,
    request: &AttachTrustedFitChildRequest,
) -> Result<(), ServerError> {
    let (name, args) = service(job)?;
    if !matches!(name, "jope-gp-search" | "jope-continuation")
        || args.len() != 4
        || args[0] != "--task-manifest"
        || args[2] != "--manifest-sha256"
        || args[3] != request.task_manifest_sha256
    {
        return Err(refused("target is not the exact native GP manifest"));
    }
    let path = Path::new(&args[1]);
    if !path.is_absolute() {
        return Err(refused("target manifest path is not absolute"));
    }
    let manifest = bounded_json(path, &request.task_manifest_sha256)?;
    let task: jain_smartcluster::jope::TaskManifest =
        serde_json::from_value(manifest.clone()).map_err(unavailable_fit)?;
    task.validate_at(now_timestamp()).map_err(unavailable_fit)?;
    // Reconstruct all resource/deadline/idempotency fields, not merely the path hint.
    let spec = jain_smartcluster::jope::job_spec(
        path,
        &request.task_manifest_sha256,
        job.spec.tenant.clone(),
        job.spec.project.clone(),
        now_timestamp(),
    )
    .map_err(unavailable_fit)?;
    if spec != job.spec {
        return Err(refused("target JobSpec differs from native task"));
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| refused("manifest stem"))?;
    let input = bounded_json(
        &path.with_file_name(format!("{stem}.input.json")),
        &task.input_manifest_sha256,
    )?;
    let operation = &input["operation"];
    if input["schema"] != "jope.smartcluster_input.v1" {
        return Err(refused("native input schema"));
    }
    let evaluations = match operation["kind"].as_str() {
        Some("Search") => {
            if (name == "jope-continuation") == operation["seeds"].is_null() {
                return Err(refused("legacy Search fixed service mismatch"));
            }
            operation["evaluations"].as_u64()
        }
        Some(kind @ ("DirectEvaluation" | "SeededContinuation")) => {
            let direct = kind == "DirectEvaluation";
            if direct != (name == "jope-gp-search") {
                return Err(refused("native genome fixed service mismatch"));
            }
            let root = Path::new(
                input["root"]
                    .as_str()
                    .ok_or_else(|| refused("native root"))?,
            );
            let nested = Path::new(
                operation["manifest"]
                    .as_str()
                    .ok_or_else(|| refused("genome manifest"))?,
            );
            if !root.is_absolute()
                || nested.as_os_str().is_empty()
                || nested
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err(refused("genome manifest path escape"));
            }
            let genome = bounded_json(
                &root.join(nested),
                operation["sha256"]
                    .as_str()
                    .ok_or_else(|| refused("genome manifest digest"))?,
            )?;
            let n = genome["evaluations"].as_u64();
            if genome["schema"] != "jope.native_genome_task.v1"
                || genome["stage"] != if direct { "direct" } else { "continuation" }
                || (direct && (n != Some(1) || !genome["direct_parent"].is_null()))
                || (!direct && genome["direct_parent"].is_null())
                || genome["seconds"].as_u64().is_none_or(|n| n == 0 || n > 600)
            {
                return Err(refused("native genome stage or budget differs"));
            }
            let candidate_ref = &genome["candidate"];
            let path = Path::new(
                candidate_ref["path"]
                    .as_str()
                    .ok_or_else(|| refused("candidate path"))?,
            );
            if path.as_os_str().is_empty()
                || path
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err(refused("candidate path escape"));
            }
            let candidate = bounded_json(
                &root.join(path),
                candidate_ref["sha256"]
                    .as_str()
                    .ok_or_else(|| refused("candidate digest"))?,
            )?;
            if candidate["schema"] != "jope.native_candidate.v1"
                || candidate["run"] != task.run
                || candidate["generation"] != task.generation
                || candidate["context"]["partition"] != "meta_train"
            {
                return Err(refused(
                    "candidate run/generation/scientific partition differs",
                ));
            }
            n
        }
        _ => return Err(refused("unsupported native numerical operation")),
    };
    if evaluations.is_none_or(|n| n == 0 || n > 1000 || u64::from(request.candidate_index) >= n) {
        return Err(refused("candidate exceeds original numerical budget"));
    }
    Ok(())
}

impl DaemonActor {
    pub(super) fn inspect_attempt(
        &self,
        peer: &AuthenticatedPeer,
        attempt: FencedAttempt,
    ) -> Result<Response, ServerError> {
        let job = self.authorized_job(peer, attempt.job_id)?;
        let Some(record) = self.store.get_attempt(attempt).map_err(store_error)? else {
            return Ok(Response::AttemptInspected(None));
        };
        if record.attempt != attempt {
            return Err(refused("stored attempt fence mismatch"));
        }
        let mut active_process = None;
        if job.active_attempt == Some(attempt)
            && matches!(
                job.state,
                JobState::Starting | JobState::Running | JobState::Paused
            )
        {
            if let Some(active) = self.active.get(&job.id) {
                if active.attempt == attempt && active.leader_exit.is_none() && !active.cancelling {
                    let state = proc_state(active.pid)?;
                    let group = active
                        .cgroup
                        .as_ref()
                        .ok_or_else(|| refused("live attempt cgroup absent"))?;
                    if state.dead
                        || Some(state.birth) != active.pid_start_time
                        || record.pid != Some(active.pid)
                        || proc_cgroup(active.pid)? != group.path
                        || record.cgroup_path.as_deref() != group.path.to_str()
                    {
                        return Err(refused("live attempt process identity changed"));
                    }
                    let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")
                        .map_err(unavailable_fit)?
                        .trim()
                        .to_owned();
                    // Recheck identity after the independent cgroup/boot observations.
                    if proc_state(active.pid)? != state || proc_cgroup(active.pid)? != group.path {
                        return Err(refused("live attempt changed during inspection"));
                    }
                    active_process = Some(jain_smartcluster_core::NativeProcessIdentity {
                        pid: active.pid,
                        birth_ticks: state.birth,
                        boot_id,
                    });
                }
            }
        }
        Ok(Response::AttemptInspected(Some(
            jain_smartcluster_core::AttemptInspection {
                record,
                active_process,
            },
        )))
    }
    pub(super) fn attach_trusted_fit_child(
        &mut self,
        peer: &AuthenticatedPeer,
        request: AttachTrustedFitChildRequest,
    ) -> Result<Response, ServerError> {
        request.validate().map_err(unavailable_fit)?;
        if !self.shutdown_replies.is_empty() {
            return Err(refused("fit placement closed during shutdown"));
        }
        let policy = self
            .execution
            .trusted_fit_child_policy
            .as_ref()
            .ok_or_else(|| refused("native fit placement not configured"))?
            .clone();
        let authority = self.authorized_job(peer, request.authority_attempt.job_id)?;
        let target = self.authorized_job(peer, request.target_attempt.job_id)?;
        live(&authority, request.authority_attempt)?;
        live(&target, request.target_attempt)?;
        if service(&authority)?.0 != "jope-fit-authority"
            || authority.id == target.id
            || authority.spec.tenant != target.spec.tenant
            || authority.spec.project != target.spec.project
            || authority.spec.tenant.as_str() != policy.tenant
            || authority.spec.project.as_str() != policy.project
        {
            return Err(refused("fit authority/target scope mismatch"));
        }
        validate_candidate(&target, &request)?;
        let active = self
            .active
            .get(&authority.id)
            .ok_or_else(|| refused("authority not running locally"))?;
        if active.attempt != request.authority_attempt
            || active.pid != peer.identity.pid as u32
            || active.leader_exit.is_some()
            || active.cancelling
            || proc_state(active.pid)?.birth
                != active
                    .pid_start_time
                    .ok_or_else(|| refused("authority birth absent"))?
            || file_digest(Path::new(&format!("/proc/{}/exe", active.pid)))?
                != policy.authority_executable_sha256
        {
            return Err(refused("caller is not the qualified authority leader"));
        }
        let authority_pid = active.pid;
        let _authority_pidfd = rustix::process::pidfd_open(
            rustix::process::Pid::from_raw(authority_pid as i32)
                .ok_or_else(|| refused("invalid authority pid"))?,
            rustix::process::PidfdFlags::empty(),
        )
        .map_err(unavailable_fit)?;
        let authority_birth = active.pid_start_time;
        let authority_group = active
            .cgroup
            .clone()
            .ok_or_else(|| refused("authority cgroup absent"))?;
        let target_active = self
            .active
            .get(&target.id)
            .ok_or_else(|| refused("target not locally active"))?;
        if target_active.attempt != request.target_attempt
            || target_active.cancelling
            || target_active.leader_exit.is_some()
        {
            return Err(refused("target is draining"));
        }
        let target_group = target_active
            .cgroup
            .clone()
            .ok_or_else(|| refused("target cgroup absent"))?;
        let target_pid = target_active.pid;
        let target_birth = target_active.pid_start_time;
        if !same_process(target_pid, target_birth)
            || proc_state(target_pid)?.dead
            || proc_cgroup(target_pid)? != target_group.path
        {
            return Err(refused("target leader identity or membership changed"));
        }
        if proc_cgroup(authority_pid)? != authority_group.path {
            return Err(refused("authority left its allocation"));
        }
        if !policy
            .evaluator_executable_sha256
            .contains(&request.child_executable_sha256)
        {
            return Err(refused("child executable not allowed"));
        }
        let boot =
            fs::read_to_string("/proc/sys/kernel/random/boot_id").map_err(unavailable_fit)?;
        if boot.trim() != request.child_boot_id {
            return Err(refused("child boot identity changed"));
        }
        // Keeping a pidfd pins the observed process identity throughout placement.
        let child_pid = rustix::process::Pid::from_raw(request.child_pid as i32)
            .ok_or_else(|| refused("invalid child pid"))?;
        let _pidfd = rustix::process::pidfd_open(child_pid, rustix::process::PidfdFlags::empty())
            .map_err(unavailable_fit)?;
        let slot = format!(
            "native-fit/{}/{}/{}/{}/{}",
            request.authority_attempt.job_id,
            request.authority_attempt.fence,
            request.target_attempt.job_id,
            request.target_attempt.fence,
            request.candidate_index
        );
        let intent_key = format!("{slot}/intent");
        let receipt_key = format!("{slot}/receipt");
        let intent = FitPlacementIntent {
            request: request.clone(),
            authority_pid,
            authority_birth,
            authority_cgroup: authority_group.path.clone(),
            target_cgroup: target_group.path.clone(),
        };
        let existing = self
            .store
            .get_meta::<FitPlacementIntent>(&intent_key)
            .map_err(store_error)?;
        if existing.as_ref().is_some_and(|value| *value != intent) {
            return Err(ServerError::new(
                ErrorCode::Conflict,
                "conflicting native fit candidate",
            ));
        }
        let tasks: Vec<_> = fs::read_dir(format!("/proc/{}/task", request.child_pid))
            .map_err(unavailable_fit)?
            .take(2)
            .collect::<Result<Vec<_>, _>>()
            .map_err(unavailable_fit)?;
        if tasks.len() != 1
            || !fs::read_to_string(format!(
                "/proc/{}/task/{}/children",
                request.child_pid, request.child_pid
            ))
            .map_err(unavailable_fit)?
            .trim()
            .is_empty()
        {
            return Err(refused(
                "fit child must be single-threaded with no descendants before placement",
            ));
        }
        let state = proc_state(request.child_pid)?;
        if state.parent != authority_pid
            || state.birth != request.child_birth_ticks
            || !state.stopped
            || fs::metadata(format!("/proc/{}", request.child_pid))
                .map_err(unavailable_fit)?
                .uid()
                != peer.identity.uid
            || file_digest(Path::new(&format!("/proc/{}/exe", request.child_pid)))?
                != request.child_executable_sha256
        {
            return Err(refused(
                "child is not the original qualified stopped direct child",
            ));
        }
        let group = proc_cgroup(request.child_pid)?;
        if group != authority_group.path && !(existing.is_some() && group == target_group.path) {
            return Err(refused("child is outside original/receipted allocation"));
        }
        if let Some(receipt) = self
            .store
            .get_meta::<TrustedFitChildPlacement>(&receipt_key)
            .map_err(store_error)?
        {
            // Historical placement may be reused, but never silently move a child again.
            let mut value = serde_json::to_value(&receipt).map_err(unavailable_fit)?;
            value
                .as_object_mut()
                .expect("placement object")
                .remove("receipt_sha256");
            if group != target_group.path
                || receipt.schema != "jope.smartcluster_fit_placement.v1"
                || receipt.authority_attempt != request.authority_attempt
                || receipt.target_attempt != request.target_attempt
                || receipt.child_pid != request.child_pid
                || receipt.child_birth_ticks != request.child_birth_ticks
                || receipt.child_boot_id != request.child_boot_id
                || receipt.task_manifest_sha256 != request.task_manifest_sha256
                || receipt.descriptor_sha256 != request.descriptor_sha256
                || receipt.candidate_index != request.candidate_index
                || receipt.child_executable_sha256 != request.child_executable_sha256
                || receipt.cgroup_path != target_group.path
                || receipt.receipt_sha256
                    != digest(&serde_json::to_vec(&value).map_err(unavailable_fit)?)
                || !same_process(authority_pid, authority_birth)
                || !same_process(target_pid, target_birth)
                || proc_state(authority_pid)?.dead
                || proc_state(target_pid)?.dead
                || proc_cgroup(authority_pid)? != authority_group.path
                || proc_cgroup(target_pid)? != target_group.path
            {
                return Err(refused(
                    "cached placement identity or current custody changed",
                ));
            }
            live(&authority, request.authority_attempt)?;
            live(&target, request.target_attempt)?;
            return Ok(Response::TrustedFitChildPlaced(receipt));
        }
        let idempotency = format!(
            "native-fit-idempotency/{}/{}/{}/{}/{}",
            request.authority_attempt.job_id,
            request.authority_attempt.fence,
            request.target_attempt.job_id,
            request.target_attempt.fence,
            digest(request.idempotency_key.as_str().as_bytes())
        );
        if self
            .store
            .get_meta::<String>(&idempotency)
            .map_err(store_error)?
            .is_some_and(|old| old != slot)
        {
            return Err(ServerError::new(
                ErrorCode::Conflict,
                "fit idempotency key reused for another candidate",
            ));
        }
        self.store
            .put_meta(&idempotency, &slot)
            .map_err(store_error)?;
        self.store
            .put_meta(&intent_key, &intent)
            .map_err(store_error)?;
        // The trusted authority must retain its original Child and never resume/wait
        // it until this RPC finishes. A stopped child cannot execute numerical work.
        if group != target_group.path {
            self.cgroups
                .attach_pid(&target_group, request.child_pid)
                .map_err(unavailable_fit)?;
        }
        let after = proc_state(request.child_pid)?;
        if after != state
            || proc_cgroup(request.child_pid)? != target_group.path
            || file_digest(Path::new(&format!("/proc/{}/exe", request.child_pid)))?
                != request.child_executable_sha256
            || !same_process(authority_pid, authority_birth)
            || proc_state(authority_pid)?.dead
            || proc_state(target_pid)?.dead
            || !same_process(target_pid, target_birth)
            || proc_cgroup(target_pid)? != target_group.path
        {
            return Err(refused(
                "placed child or authority changed; keep intent for reconciliation",
            ));
        }
        let authority = self
            .store
            .get_job(authority.id)
            .map_err(store_error)?
            .ok_or_else(|| refused("authority vanished"))?;
        let target = self
            .store
            .get_job(target.id)
            .map_err(store_error)?
            .ok_or_else(|| refused("target vanished"))?;
        live(&authority, request.authority_attempt)?;
        live(&target, request.target_attempt)?;
        let mut receipt = TrustedFitChildPlacement {
            schema: "jope.smartcluster_fit_placement.v1".into(),
            authority_attempt: request.authority_attempt,
            target_attempt: request.target_attempt,
            child_pid: request.child_pid,
            child_birth_ticks: request.child_birth_ticks,
            child_boot_id: request.child_boot_id,
            task_manifest_sha256: request.task_manifest_sha256,
            descriptor_sha256: request.descriptor_sha256,
            candidate_index: request.candidate_index,
            child_executable_sha256: request.child_executable_sha256,
            cgroup_path: target_group.path,
            receipt_sha256: String::new(),
        };
        let mut value = serde_json::to_value(&receipt).map_err(unavailable_fit)?;
        value
            .as_object_mut()
            .expect("placement object")
            .remove("receipt_sha256");
        receipt.receipt_sha256 = digest(&serde_json::to_vec(&value).map_err(unavailable_fit)?);
        self.store
            .put_meta(&receipt_key, &receipt)
            .map_err(store_error)?;
        Ok(Response::TrustedFitChildPlaced(receipt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn candidate_case(
        kind: &str,
        stage: &str,
        evaluations: u64,
        index: u32,
        partition: &str,
        candidate_generation: u64,
        tamper: bool,
    ) -> Result<(), ServerError> {
        use jain_smartcluster::jope::{TaskKind, TaskManifest};
        use jain_smartcluster_core::{IdempotencyKey, ProjectId, ResourceRequest, TenantId};
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let candidate = serde_json::json!({"schema":"jope.native_candidate.v1", "run":"test-run",
            "generation":candidate_generation,"context":{"partition":partition}});
        let candidate_bytes = serde_json::to_vec(&candidate).unwrap();
        fs::write(root.join("candidate.json"), &candidate_bytes).unwrap();
        let genome = serde_json::json!({"schema":"jope.native_genome_task.v1", "stage":stage,
            "evaluations":evaluations,"seconds":30,"direct_parent":if stage=="direct" {serde_json::Value::Null} else {serde_json::json!({"path":"parent.json","sha256":"a".repeat(64)})},
            "candidate":{"path":"candidate.json","sha256":digest(&candidate_bytes)}});
        let genome_bytes = serde_json::to_vec(&genome).unwrap();
        fs::write(root.join("genome.json"), &genome_bytes).unwrap();
        let input = serde_json::json!({"schema":"jope.smartcluster_input.v1","root":root,
            "operation":{"kind":kind,"manifest":"genome.json","sha256":digest(&genome_bytes)}});
        let input_bytes = serde_json::to_vec(&input).unwrap();
        fs::write(root.join("task.input.json"), &input_bytes).unwrap();
        let now = now_timestamp();
        let task = TaskManifest {
            schema: "jope.smartcluster_task.v1".into(),
            run: "test-run".into(),
            generation: 1,
            task_id: "candidate".into(),
            kind: if kind == "DirectEvaluation" {
                TaskKind::GpSearch
            } else {
                TaskKind::Continuation
            },
            input_manifest_sha256: digest(&input_bytes),
            dependencies: std::collections::BTreeMap::from([(
                "candidate.json".into(),
                digest(&candidate_bytes),
            )]),
            parent_checkpoint_sha256: None,
            corpus_sha256: None,
            idempotency_key: IdempotencyKey::new("candidate").unwrap(),
            deadline: Timestamp::from(u64::MAX - 1),
            retry_allowance: 0,
            resources: ResourceRequest::cpu_and_ram(1000, 1073741824),
            expected_receipt_id: "candidate/result".into(),
            checkpoint_allowance_ms: None,
        };
        let task_bytes = serde_json::to_vec(&task).unwrap();
        let path = root.join("task.json");
        fs::write(&path, &task_bytes).unwrap();
        let spec = jain_smartcluster::jope::job_spec(
            &path,
            &digest(&task_bytes),
            TenantId::new("jope").unwrap(),
            ProjectId::new("qualification").unwrap(),
            now,
        )
        .unwrap();
        let id = JobId::new();
        let job = JobRecord::queued(
            id,
            spec,
            Principal::System {
                name: jain_smartcluster_core::SystemPrincipal::new("jope").unwrap(),
            },
            now,
        );
        let attempt = FencedAttempt::new(
            id,
            jain_smartcluster_core::AttemptNumber::new(1).unwrap(),
            jain_smartcluster_core::FenceToken::new(1).unwrap(),
        );
        let request = AttachTrustedFitChildRequest {
            authority_attempt: attempt,
            target_attempt: attempt,
            child_pid: 42,
            child_birth_ticks: 1,
            child_boot_id: "11111111-1111-1111-1111-111111111111".into(),
            task_manifest_sha256: digest(&task_bytes),
            descriptor_sha256: "b".repeat(64),
            candidate_index: index,
            idempotency_key: IdempotencyKey::new("candidate").unwrap(),
            child_executable_sha256: "c".repeat(64),
        };
        if tamper {
            fs::write(root.join("candidate.json"), b"{}").unwrap();
        }
        validate_candidate(&job, &request)
    }
    #[test]
    fn genome_child_admission_binds_budget_stage_partition_and_original_bytes() {
        assert!(candidate_case("DirectEvaluation", "direct", 1, 0, "meta_train", 1, false).is_ok());
        assert!(candidate_case(
            "SeededContinuation",
            "continuation",
            16,
            15,
            "meta_train",
            1,
            false
        )
        .is_ok());
        for (kind, stage, n, index, partition, generation, tamper) in [
            ("DirectEvaluation", "direct", 2, 0, "meta_train", 1, false),
            (
                "DirectEvaluation",
                "continuation",
                1,
                0,
                "meta_train",
                1,
                false,
            ),
            (
                "SeededContinuation",
                "continuation",
                16,
                16,
                "meta_train",
                1,
                false,
            ),
            (
                "SeededContinuation",
                "continuation",
                0,
                0,
                "meta_train",
                1,
                false,
            ),
            (
                "SeededContinuation",
                "continuation",
                1001,
                0,
                "meta_train",
                1,
                false,
            ),
            ("DirectEvaluation", "direct", 1, 0, "sealed_test", 1, false),
            ("DirectEvaluation", "direct", 1, 0, "meta_train", 2, false),
            ("DirectEvaluation", "direct", 1, 0, "meta_train", 1, true),
        ] {
            assert!(candidate_case(kind, stage, n, index, partition, generation, tamper).is_err());
        }
    }
    #[test]
    fn placement_intent_survives_store_reopen_with_exact_candidate_identity() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fit.redb");
        let attempt = |fence| {
            FencedAttempt::new(
                JobId::new(),
                jain_smartcluster_core::AttemptNumber::new(1).unwrap(),
                jain_smartcluster_core::FenceToken::new(fence).unwrap(),
            )
        };
        let intent = FitPlacementIntent {
            request: AttachTrustedFitChildRequest {
                authority_attempt: attempt(1),
                target_attempt: attempt(2),
                child_pid: 42,
                child_birth_ticks: 123,
                child_boot_id: "11111111-1111-1111-1111-111111111111".into(),
                task_manifest_sha256: "a".repeat(64),
                descriptor_sha256: "b".repeat(64),
                candidate_index: 0,
                idempotency_key: jain_smartcluster_core::IdempotencyKey::new("fit-0").unwrap(),
                child_executable_sha256: "c".repeat(64),
            },
            authority_pid: 41,
            authority_birth: Some(122),
            authority_cgroup: "/sys/fs/cgroup/authority".into(),
            target_cgroup: "/sys/fs/cgroup/target".into(),
        };
        {
            let store = Store::open(&path).unwrap();
            store.put_meta("test-fit-intent", &intent).unwrap();
        }
        let store = Store::open(&path).unwrap();
        assert_eq!(
            store
                .get_meta::<FitPlacementIntent>("test-fit-intent")
                .unwrap(),
            Some(intent.clone())
        );
        let mut conflicting = intent.clone();
        conflicting.request.descriptor_sha256 = "d".repeat(64);
        assert_ne!(
            Some(conflicting),
            store
                .get_meta::<FitPlacementIntent>("test-fit-intent")
                .unwrap()
        );
    }
    #[test]
    fn proc_identity_uses_final_parenthesis_and_rejects_traced_or_running_child() {
        let fields = |state: &str| {
            format!("99 (a ) difficult name) {state} 42 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 123 0")
        };
        let stopped = parse_stat(&fields("T")).unwrap();
        assert_eq!(stopped.parent, 42);
        assert_eq!(stopped.birth, 123);
        assert!(stopped.stopped);
        assert!(!parse_stat(&fields("t")).unwrap().stopped);
        assert!(!parse_stat(&fields("R")).unwrap().stopped);
        assert!(parse_stat("broken").is_err());
    }
}
