use super::*;

impl DaemonActor {
    pub(super) async fn handle_request(
        &mut self,
        peer: &AuthenticatedPeer,
        request: Request,
        sink: ConnectionSink,
    ) -> Result<Response, ServerError> {
        match request {
            Request::InspectAttempt(request) => self.inspect_attempt(peer, request.attempt),
            Request::AttachTrustedFitChild(request) => {
                self.attach_trusted_fit_child(peer, *request)
            }
            Request::Status(_) => Ok(Response::Status(self.health.status())),
            Request::Submit(request) => self.submit(peer, *request),
            Request::Get(request) => {
                let job = self.store.get_job(request.job_id).map_err(store_error)?;
                match job {
                    Some(job) => {
                        self.authorizer
                            .authorize_job_access(peer, &job.submitted_by)
                            .map_err(auth_error)?;
                        Ok(Response::Job(Some(job)))
                    }
                    None => Ok(Response::Job(None)),
                }
            }
            Request::List(request) => self.list(peer, request),
            Request::Cancel(request) => self.cancel(peer, request.job_id, request.mode).await,
            Request::Pause(request) => self.pause(peer, request.job_id),
            Request::Resume(request) => self.resume(peer, request.job_id),
            Request::SendControl(request) => {
                let job = self.authorized_job(peer, request.job_id)?;
                let active = self.active.get_mut(&job.id).ok_or_else(|| {
                    ServerError::new(ErrorCode::Conflict, "job has no live control channel")
                })?;
                if active.control.is_none() {
                    return Err(ServerError::new(
                        ErrorCode::Conflict,
                        "adopted job has no recoverable control channel",
                    ));
                }
                let control = bounded_control(request.control, &self.execution)?;
                let attempt = active.attempt;
                if let Err(error) =
                    send_control(active, control, self.execution.actor_io_timeout()).await
                {
                    self.begin_control_recovery(attempt, FailureClass::Infrastructure)
                        .map_err(actor_server_error)?;
                    return Err(executor_error(error));
                }
                Ok(Response::ControlAccepted(OperationAck {
                    accepted_at: now_timestamp(),
                }))
            }
            Request::Subscribe(request) => {
                self.open_stream(peer, request.offset, request.filter, None, true, sink)
            }
            Request::Attach(request) => {
                let job = self.authorized_job(peer, request.job_id)?;
                self.open_stream(
                    peer,
                    request.offset,
                    EventFilter {
                        job_id: Some(job.id),
                        ..EventFilter::default()
                    },
                    Some(job.id),
                    request.read_only,
                    sink,
                )
            }
            Request::Wait(_) => unreachable!("wait requests are deferred in the actor loop"),
        }
    }

    pub(super) fn submit(
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

    pub(super) fn list(
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

    pub(super) async fn cancel(
        &mut self,
        peer: &AuthenticatedPeer,
        job_id: JobId,
        mode: CancelMode,
    ) -> Result<Response, ServerError> {
        let job = self.authorized_job(peer, job_id)?;
        if job.state.is_terminal() {
            return Ok(Response::Mutated(job));
        }
        let at = now_timestamp();
        let receipt = self
            .store
            .transition_before_effect(TransitionMutation {
                transition: FencedTransition {
                    job_id,
                    from: job.state,
                    to: JobState::Cancelled,
                    attempt: job.active_attempt,
                    at,
                    reason: Some("cancellation committed before process signal".to_owned()),
                },
                terminal: Some(TerminalStatus::Cancelled {
                    message: Some("cancelled by authenticated caller".to_owned()),
                    at,
                }),
                failure: None,
            })
            .map_err(store_error)?;
        self.record_transition(&receipt).map_err(event_error)?;
        self.notify_waiters(&receipt.job);

        let Some(attempt) = self.active.get(&job_id).map(|active| active.attempt) else {
            return Ok(Response::Mutated(receipt.job));
        };
        if let Some(active) = self.active.get_mut(&job_id) {
            active.cancelling = true;
        }
        let (delay, grace, checkpoint_signal) = match mode {
            CancelMode::Graceful { grace_ms } => (
                Duration::ZERO,
                Duration::from_millis(grace_ms).min(self.execution.terminate_grace()),
                None,
            ),
            CancelMode::CheckpointThenCancel {
                checkpoint_deadline_ms,
            } => {
                let requested = Duration::from_millis(checkpoint_deadline_ms)
                    .min(self.execution.checkpoint_timeout());
                match job.spec.preemption.checkpoint {
                    jain_smartcluster_core::CheckpointPolicy::ControlFd { deadline_ms } => {
                        let delay = requested.min(Duration::from_millis(deadline_ms));
                        if let Some(active) = self.active.get_mut(&job_id) {
                            if let Err(error) =
                                request_checkpoint(active, delay, self.execution.actor_io_timeout())
                                    .await
                            {
                                tracing::warn!(%job_id, %error, "checkpoint request failed before cancellation");
                            }
                        }
                        (delay, self.execution.terminate_grace(), None)
                    }
                    jain_smartcluster_core::CheckpointPolicy::Signal { signal, grace_ms } => (
                        requested.min(Duration::from_millis(grace_ms)),
                        self.execution.terminate_grace(),
                        Signal::try_from(signal).ok(),
                    ),
                    jain_smartcluster_core::CheckpointPolicy::None => {
                        (Duration::ZERO, self.execution.terminate_grace(), None)
                    }
                }
            }
        };
        if let Some(signal) = checkpoint_signal {
            self.signal_active_safely(job_id, attempt, signal)
                .await
                .map_err(executor_error)?;
        }
        self.queue_termination(job_id, attempt, delay, grace);
        Ok(Response::Mutated(receipt.job))
    }

    pub(super) fn pause(
        &mut self,
        peer: &AuthenticatedPeer,
        job_id: JobId,
    ) -> Result<Response, ServerError> {
        let job = self.authorized_job(peer, job_id)?;
        if job.state == JobState::Paused {
            return Ok(Response::Mutated(job));
        }
        if job.state == JobState::Queued {
            let receipt = self
                .transition(&job, JobState::Paused, "queued job held before admission")
                .map_err(actor_server_error)?;
            self.record_transition(&receipt).map_err(event_error)?;
            return Ok(Response::Mutated(receipt.job));
        }
        if job.state != JobState::Running {
            return Err(ServerError::new(
                ErrorCode::Conflict,
                "only running jobs can be paused",
            ));
        }
        if job.spec.resources.gpu.is_some() {
            return Err(ServerError::new(
                ErrorCode::Conflict,
                "freezer pause is supported only for CPU-only workloads",
            ));
        }
        let active = self.active.get(&job_id).ok_or_else(|| {
            ServerError::new(ErrorCode::Conflict, "running job has no adopted cgroup")
        })?;
        if active.cancelling {
            return Err(ServerError::new(
                ErrorCode::Conflict,
                "draining allocation cannot be frozen",
            ));
        }
        let attempt = active.attempt;
        self.store
            .put_meta(&attempt_overload_pause_key(attempt), &false)
            .map_err(store_error)?;
        if active.cgroup.is_none() {
            return Err(ServerError::new(
                ErrorCode::Unavailable,
                "automatic pause is disabled in safe single-allocation mode",
            ));
        }
        let receipt = self
            .transition(&job, JobState::Paused, "pause committed before freezer")
            .map_err(actor_server_error)?;
        self.record_transition(&receipt).map_err(event_error)?;
        self.queue_freezer(job_id, attempt, true);
        Ok(Response::Mutated(receipt.job))
    }

    pub(super) fn resume(
        &mut self,
        peer: &AuthenticatedPeer,
        job_id: JobId,
    ) -> Result<Response, ServerError> {
        let job = self.authorized_job(peer, job_id)?;
        if job.state == JobState::Running {
            return Ok(Response::Mutated(job));
        }
        if job.state != JobState::Paused {
            return Err(ServerError::new(
                ErrorCode::Conflict,
                "only paused jobs can be resumed",
            ));
        }
        if job.active_attempt.is_none() {
            let receipt = self
                .transition(&job, JobState::Queued, "held job released to queue")
                .map_err(actor_server_error)?;
            self.record_transition(&receipt).map_err(event_error)?;
            self.request_schedule();
            return Ok(Response::Mutated(receipt.job));
        }
        let active = self.active.get(&job_id).ok_or_else(|| {
            ServerError::new(ErrorCode::Conflict, "paused job has no adopted cgroup")
        })?;
        let attempt = active.attempt;
        if active.cgroup.is_none() {
            return Err(ServerError::new(
                ErrorCode::Unavailable,
                "freezer is unavailable",
            ));
        }
        let receipt = self
            .transition(&job, JobState::Running, "resume committed before thaw")
            .map_err(actor_server_error)?;
        self.record_transition(&receipt).map_err(event_error)?;
        self.queue_freezer(job_id, attempt, false);
        Ok(Response::Mutated(receipt.job))
    }

    pub(super) async fn handle_stream_input(
        &mut self,
        peer: &AuthenticatedPeer,
        frame: StreamInputFrame,
    ) -> Result<(), ServerError> {
        let registration = self
            .streams
            .get(&frame.stream_id)
            .ok_or_else(|| ServerError::new(ErrorCode::NotFound, "stream does not exist"))?;
        if registration.peer != peer.identity {
            return Err(ServerError::new(
                ErrorCode::Forbidden,
                "stream belongs to another peer",
            ));
        }
        let job_id = registration.job_id.ok_or_else(|| {
            ServerError::new(
                ErrorCode::Conflict,
                "subscription stream does not accept input",
            )
        })?;
        if registration.read_only {
            return Err(ServerError::new(
                ErrorCode::Forbidden,
                "attach stream is read-only",
            ));
        }
        let active = self.active.get_mut(&job_id).ok_or_else(|| {
            ServerError::new(ErrorCode::Conflict, "job is not attached to a live process")
        })?;
        let io = active.io.as_mut().ok_or_else(|| {
            ServerError::new(
                ErrorCode::Conflict,
                "adopted job has no recoverable terminal or pipe handles",
            )
        })?;
        match frame.input {
            AttachInput::Stdin(data) => match io.input() {
                crate::executor::WorkloadInput::Pipe(writer)
                | crate::executor::WorkloadInput::Pty(writer) => {
                    tokio::time::timeout(self.execution.actor_io_timeout(), async {
                        writer.write_all(&data).await?;
                        writer.flush().await
                    })
                    .await
                    .map_err(|_| io_timeout_error())?
                    .map_err(io_error)?;
                }
            },
            AttachInput::Resize { rows, columns } => {
                io.resize(rows, columns).map_err(executor_error)?;
            }
            AttachInput::Control(control) => {
                let control = bounded_control(control, &self.execution)?;
                let attempt = active.attempt;
                if let Err(error) =
                    send_control(active, control, self.execution.actor_io_timeout()).await
                {
                    self.begin_control_recovery(attempt, FailureClass::Infrastructure)
                        .map_err(actor_server_error)?;
                    return Err(executor_error(error));
                }
            }
            AttachInput::CloseStdin => match io.input() {
                crate::executor::WorkloadInput::Pipe(writer)
                | crate::executor::WorkloadInput::Pty(writer) => {
                    tokio::time::timeout(self.execution.actor_io_timeout(), writer.shutdown())
                        .await
                        .map_err(|_| io_timeout_error())?
                        .map_err(io_error)?;
                }
            },
        }
        Ok(())
    }

    pub(super) fn open_stream(
        &mut self,
        peer: &AuthenticatedPeer,
        offset: EventOffset,
        filter: EventFilter,
        job_id: Option<JobId>,
        read_only: bool,
        sink: ConnectionSink,
    ) -> Result<Response, ServerError> {
        let stream_id = StreamId::new();
        let (snapshot_end, subscriber) = self.journal.snapshot_and_subscribe();
        let (cancel, cancelled) = watch::channel(false);
        let (activate, activated) = oneshot::channel();
        self.streams.insert(
            stream_id,
            StreamRegistration {
                peer: peer.identity,
                job_id,
                read_only,
                cancel,
                activate: Some(activate),
            },
        );
        spawn_event_stream(EventStream {
            stream_id,
            requested: offset.get(),
            snapshot_end,
            filter,
            peer: peer.clone(),
            authorizer: self.authorizer.clone(),
            store: Arc::clone(&self.store),
            journal: self.journal.clone(),
            subscriber,
            sink,
            cancelled,
            activated,
            actor: self.sender.clone(),
        });
        Ok(if job_id.is_some() {
            Response::Attached(StreamOpened {
                stream_id,
                next_offset: offset,
            })
        } else {
            Response::Subscribed(StreamOpened {
                stream_id,
                next_offset: offset,
            })
        })
    }
}
