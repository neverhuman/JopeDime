use super::*;

impl DaemonActor {
    pub(crate) fn inspect_attempt(
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
    pub(crate) fn attach_trusted_fit_child(
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
