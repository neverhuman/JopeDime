use super::*;

impl DaemonActor {
    pub(in super::super) fn submit(
        &mut self,
        peer: &AuthenticatedPeer,
        request: SubmitRequest,
    ) -> Result<Response, ServerError> {
        if !self.shutdown_replies.is_empty() {
            return Err(ServerError::new(
                ErrorCode::Unavailable,
                "daemon admission closed for shutdown",
            ));
        }
        let principal = match request.effective_principal {
            Some(principal) => principal,
            None => peer.direct_principal.clone().ok_or_else(|| {
                ServerError::new(ErrorCode::Forbidden, "web delegation requires an owner")
            })?,
        };
        self.authorizer
            .authorize_submission(
                peer,
                &principal,
                request.spec.workload.requires_administrator(),
            )
            .map_err(auth_error)?;
        if request.spec.priority == jain_smartcluster_core::PriorityClass::System
            && !peer.administrator
            && peer.role != crate::auth::PeerRole::System
        {
            return Err(ServerError::new(
                ErrorCode::Forbidden,
                "system-priority jobs require a system or administrator peer",
            ));
        }
        recovery::resident_checkpoint_allowance(&request.spec, &self.execution)
            .map_err(|message| ServerError::new(ErrorCode::InvalidRequest, message))?;
        if recovery::controlled_resident(&request.spec)
            && self.cgroups.mode() == crate::cgroup::IsolationMode::SingleAllocation
        {
            return Err(ServerError::new(
                ErrorCode::InvalidRequest,
                "controlled resident services require delegated cgroups",
            ));
        }
        let cpus = self
            .execution
            .resource_policy
            .cpus_for(&request.spec, &self.capacity.cpus)
            .map_err(|reason| ServerError::new(ErrorCode::InvalidRequest, reason))?;
        if !self.execution.resource_policy.service_cpu_sets.is_empty()
            && u64::from(request.spec.resources.cpu.maximum_millicores)
                > (cpus.len() as u64).saturating_mul(1000)
        {
            return Err(ServerError::new(
                ErrorCode::InvalidRequest,
                "CPU request exceeds fixed service pool",
            ));
        }
        self.execution
            .resource_policy
            .validate_scratch_request(&request.spec)
            .map_err(|reason| ServerError::new(ErrorCode::InvalidRequest, reason))?;
        if request.spec.resources.gpu.is_some_and(|gpu| {
            gpu.mode == jain_smartcluster_core::GpuMode::Shareable && gpu.vram_bytes < 1024 * 1024
        }) {
            return Err(ServerError::new(
                ErrorCode::InvalidRequest,
                "shareable GPU VRAM budgets must be at least one MiB",
            ));
        }
        if !tenant_matches_principal(&request.spec.tenant, &principal) {
            return Err(ServerError::new(
                ErrorCode::Forbidden,
                "job tenant must exactly match the authenticated effective principal",
            ));
        }
        if workload_has_environment(&request.spec.workload)
            && !peer.administrator
            && peer.role != crate::auth::PeerRole::System
        {
            return Err(ServerError::new(
                ErrorCode::Forbidden,
                "process environment variables require a system or administrator peer",
            ));
        }
        if matches!(
            request.spec.workload,
            jain_smartcluster_core::Workload::TrustedService { .. }
        ) && !peer.administrator
            && peer.role != crate::auth::PeerRole::System
        {
            return Err(ServerError::new(
                ErrorCode::Forbidden,
                "trusted service allocations require a system or administrator peer",
            ));
        }
        request
            .spec
            .validate_for(&principal)
            .map_err(|error| ServerError::new(ErrorCode::InvalidRequest, error.to_string()))?;
        let tenant = request.spec.tenant.clone();
        if self.queued_jobs as usize >= self.max_queued_jobs
            || self
                .queued_by_tenant
                .get(&tenant)
                .copied()
                .unwrap_or_default() as usize
                >= self.max_queued_jobs_per_tenant
        {
            let mut error = ServerError::new(
                ErrorCode::ResourceExhausted,
                "queued-job limit reached; retry after capacity is released",
            );
            error.retry_after_ms = Some(250);
            return Err(error);
        }
        let job = JobRecord::queued(JobId::new(), request.spec, principal, now_timestamp());
        let outcome = self.store.submit(&job).map_err(store_error)?;
        if let crate::store::SubmitOutcome::Created(created) = &outcome {
            self.queued_jobs = self.queued_jobs.saturating_add(1);
            *self.queued_by_tenant.entry(tenant).or_default() += 1;
            self.publish_counts();
            self.schedule_deadline(created);
        }
        Ok(Response::Submitted(outcome.job().clone()))
    }

    pub(in super::super) fn list(
        &self,
        peer: &AuthenticatedPeer,
        request: ListRequest,
    ) -> Result<Response, ServerError> {
        let cursor = request
            .cursor
            .as_deref()
            .map(str::parse::<JobId>)
            .transpose()
            .map_err(|_| ServerError::new(ErrorCode::InvalidRequest, "invalid list cursor"))?;
        let mut jobs = self.store.all_jobs().map_err(store_error)?;
        jobs.retain(|job| {
            self.authorizer
                .authorize_job_access(peer, &job.submitted_by)
                .is_ok()
                && cursor.is_none_or(|cursor| job.id > cursor)
                && matches_list_filter(job, &request.filter)
        });
        let limit = request.limit as usize;
        let has_more = jobs.len() > limit;
        jobs.truncate(limit);
        let next_cursor = has_more
            .then(|| jobs.last().map(|job| job.id.to_string()))
            .flatten();
        Ok(Response::Jobs(JobPage { jobs, next_cursor }))
    }
}
