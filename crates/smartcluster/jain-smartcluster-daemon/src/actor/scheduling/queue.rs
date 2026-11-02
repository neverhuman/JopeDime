use super::*;

impl DaemonActor {
    pub(super) fn update_gpu_history(
        &self,
        active: Option<&ActiveWorkload>,
        successful: bool,
        failure: Option<FailureClass>,
        at: Timestamp,
    ) -> Result<(), ActorError> {
        let Some(active) = active else { return Ok(()) };
        let Some(fingerprint) = active.fingerprint.as_deref() else {
            return Ok(());
        };
        if active.allocation.gpu.is_empty() {
            return Ok(());
        }
        let observed = active.max_observed_vram_bytes;
        let exclusively_isolated = active
            .allocation
            .gpu
            .iter()
            .all(|gpu| gpu.mode == GpuAllocationMode::Exclusive);
        match gpu_history_action(exclusively_isolated, observed, successful, failure) {
            GpuHistoryAction::SharedCudaOom => {
                self.store
                    .update_resource_history(fingerprint, at, |history| {
                        history.record_shared_cuda_oom(observed, at);
                    })?;
            }
            GpuHistoryAction::CleanExclusive => {
                self.store
                    .update_resource_history(fingerprint, at, |history| {
                        history.record_clean_exclusive(observed, at);
                    })?;
            }
            GpuHistoryAction::FailedExclusive => {
                self.store
                    .update_resource_history(fingerprint, at, |history| {
                        history.record_failed_exclusive(observed, at);
                    })?;
            }
            GpuHistoryAction::None => {}
        }
        Ok(())
    }

    pub(super) fn requeue(&mut self, job_id: JobId) -> Result<(), ActorError> {
        // The old fenced allocation remains authoritative until every descendant drains.
        if self.active.contains_key(&job_id) {
            return Ok(());
        }
        let Some(job) = self.store.get_job(job_id)? else {
            return Ok(());
        };
        if job.state != JobState::RetryWait {
            return Ok(());
        }
        let receipt = self.transition(&job, JobState::Queued, "retry backoff elapsed")?;
        self.record_transition(&receipt)?;
        Ok(())
    }

    pub(super) fn enforce_deadline(
        &mut self,
        job_id: JobId,
        deadline: Timestamp,
    ) -> Result<(), ActorError> {
        let Some(job) = self.store.get_job(job_id)? else {
            return Ok(());
        };
        if job.state.is_terminal()
            || job.spec.deadline != Some(deadline)
            || !deadline_expired(&job, now_timestamp())
        {
            return Ok(());
        }
        let at = now_timestamp();
        let receipt = self.store.transition_before_effect(TransitionMutation {
            transition: FencedTransition {
                job_id,
                from: job.state,
                to: JobState::Failed,
                attempt: job.active_attempt,
                at,
                reason: Some("workload deadline elapsed".to_owned()),
            },
            terminal: Some(TerminalStatus::Failed {
                class: FailureClass::DeadlineExceeded,
                message: "workload deadline elapsed".to_owned(),
                exit_code: None,
                at,
            }),
            failure: job.active_attempt.map(|_| FailureClass::DeadlineExceeded),
        })?;
        self.record_transition(&receipt)?;
        self.notify_waiters(&receipt.job);
        let attempt = if let Some(active) = self.active.get_mut(&job_id) {
            active.cancelling = true;
            Some(active.attempt)
        } else {
            None
        };
        if let Some(attempt) = attempt {
            self.queue_termination(
                job_id,
                attempt,
                Duration::ZERO,
                self.execution.terminate_grace(),
            );
        }
        Ok(())
    }

    pub(super) fn fail_output(
        &mut self,
        attempt: FencedAttempt,
        message: String,
    ) -> Result<(), ActorError> {
        let Some(job) = self.store.get_job(attempt.job_id)? else {
            return Ok(());
        };
        if job.state.is_terminal() || job.active_attempt != Some(attempt) {
            return Ok(());
        }
        let message = format!("durable output recording failed: {message}");
        let mut terminate = false;
        if let Some(active) = self.active.get_mut(&job.id) {
            if active.attempt == attempt {
                active.pending_failure = Some((FailureClass::Infrastructure, message));
                active.cancelling = true;
                terminate = true;
            }
        }
        if terminate {
            self.queue_termination(job.id, attempt, Duration::ZERO, Duration::ZERO);
        }
        Ok(())
    }

    pub(super) fn schedule_lease_renewal(&self) {
        let sender = self.sender.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(ATTEMPT_LEASE_RENEW_MS)).await;
            let _ = sender.send(ActorCommand::RenewLeases).await;
        });
    }

    pub(super) fn renew_live_leases(&mut self) {
        let now = now_timestamp();
        let mut fenced = Vec::new();
        for active in self.active.values() {
            let renewal = self.store.get_attempt(active.attempt).and_then(|record| {
                let record = record.ok_or(StoreError::AttemptNotFound {
                    job_id: active.attempt.job_id,
                    attempt: active.attempt.number.get(),
                })?;
                let expires_at = next_lease_expiry(record.lease.expires_at, now);
                self.store.renew_lease(active.attempt, expires_at)
            });
            if let Err(error) = renewal {
                tracing::error!(job_id = %active.attempt.job_id, %error, "attempt lease renewal failed; fencing workload");
                fenced.push(active.attempt);
            }
        }
        for attempt in fenced {
            let job_id = attempt.job_id;
            if let Some(active) = self.active.get_mut(&job_id) {
                active.cancelling = true;
            }
            self.queue_termination(job_id, attempt, Duration::ZERO, Duration::ZERO);
        }
    }

    pub(super) fn schedule_deadline(&self, job: &JobRecord) {
        let Some(deadline) = job.spec.deadline else {
            return;
        };
        if job.state.is_terminal() {
            return;
        }
        let delay = Duration::from_millis(
            deadline
                .as_unix_millis()
                .saturating_sub(now_timestamp().as_unix_millis()),
        );
        let sender = self.sender.clone();
        let job_id = job.id;
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            let _ = sender
                .send(ActorCommand::Deadline { job_id, deadline })
                .await;
        });
    }
}
