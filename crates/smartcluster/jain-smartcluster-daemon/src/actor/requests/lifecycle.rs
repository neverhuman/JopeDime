use super::*;

impl DaemonActor {
    pub(in super::super) async fn cancel(
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

    pub(in super::super) fn pause(
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

    pub(in super::super) fn resume(
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
}
