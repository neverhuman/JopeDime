use super::*;

impl DaemonActor {
    pub(in crate::actor) async fn requeue_gpu_for_pressure(&mut self) -> Result<(), ActorError> {
        let candidate = self
            .active
            .iter()
            .filter(|(_, active)| !active.allocation.gpu.is_empty() && !active.cancelling)
            .max_by_key(|(_, active)| active.priority.rank())
            .map(|(job_id, _)| *job_id);
        let Some(job_id) = candidate else {
            return Ok(());
        };
        let Some(job) = self.store.get_job(job_id)? else {
            return Ok(());
        };
        if !matches!(
            job.state,
            JobState::Starting | JobState::Running | JobState::Paused
        ) {
            return Ok(());
        }
        let receipt = self.store.transition_before_effect(TransitionMutation {
            transition: FencedTransition {
                job_id,
                from: job.state,
                to: JobState::RetryWait,
                attempt: job.active_attempt,
                at: now_timestamp(),
                reason: Some("GPU allocation reclaimed under host pressure".to_owned()),
            },
            terminal: None,
            failure: Some(FailureClass::Preempted),
        })?;
        self.record_transition(&receipt)?;
        let Some(attempt) = self.active.get(&job_id).map(|active| active.attempt) else {
            return Ok(());
        };
        let preemptible = self
            .active
            .get(&job_id)
            .is_some_and(|active| active.preemptible);
        if let Some(active) = self.active.get_mut(&job_id) {
            active.cancelling = true;
        }
        let (delay, checkpoint_signal) = if preemptible {
            match job.spec.preemption.checkpoint {
                jain_smartcluster_core::CheckpointPolicy::ControlFd { deadline_ms } => {
                    let delay =
                        Duration::from_millis(deadline_ms).min(self.execution.checkpoint_timeout());
                    if let Some(active) = self.active.get_mut(&job_id) {
                        let _ =
                            request_checkpoint(active, delay, self.execution.actor_io_timeout())
                                .await;
                    }
                    (delay, None)
                }
                jain_smartcluster_core::CheckpointPolicy::Signal { signal, grace_ms } => (
                    Duration::from_millis(grace_ms).min(self.execution.checkpoint_timeout()),
                    Signal::try_from(signal).ok(),
                ),
                jain_smartcluster_core::CheckpointPolicy::None => (Duration::ZERO, None),
            }
        } else {
            (Duration::ZERO, None)
        };
        if let Some(signal) = checkpoint_signal {
            self.signal_active_safely(job_id, attempt, signal).await?;
        }
        self.queue_termination(job_id, attempt, delay, self.execution.terminate_grace());
        Ok(())
    }

    pub(in crate::actor) fn hard_kill_for_pressure(&mut self) -> Result<(), ActorError> {
        let job_ids: Vec<_> = self.active.keys().copied().collect();
        for job_id in job_ids {
            let Some(job) = self.store.get_job(job_id)? else {
                continue;
            };
            if !matches!(
                job.state,
                JobState::Starting | JobState::Running | JobState::Paused
            ) {
                continue;
            }
            let failure = FailureClass::Infrastructure;
            let can_retry =
                job.spec.retry.allows(failure) && job.failure_count < job.spec.retry.max_failures;
            let target = if can_retry {
                JobState::RetryWait
            } else {
                JobState::Failed
            };
            let at = now_timestamp();
            let message = "workload terminated under critical host pressure".to_owned();
            let receipt = self.store.transition_before_effect(TransitionMutation {
                transition: FencedTransition {
                    job_id,
                    from: job.state,
                    to: target,
                    attempt: job.active_attempt,
                    at,
                    reason: Some(message.clone()),
                },
                terminal: (target == JobState::Failed).then_some(TerminalStatus::Failed {
                    class: failure,
                    message,
                    exit_code: None,
                    at,
                }),
                failure: Some(failure),
            })?;
            self.record_transition(&receipt)?;
            if target == JobState::Failed {
                self.notify_waiters(&receipt.job);
            }
            let attempt = if let Some(active) = self.active.get_mut(&job_id) {
                active.cancelling = true;
                Some(active.attempt)
            } else {
                None
            };
            if let Some(attempt) = attempt {
                if let Some(active) = self.active.get(&job_id) {
                    if let Some(cgroup) = &active.cgroup {
                        if let Err(error) = self.cgroups.kill(cgroup) {
                            tracing::error!(%job_id, %error, "critical pressure cgroup.kill failed");
                        }
                    } else if let Err(error) = signal_active(&self.cgroups, active, Signal::SIGKILL)
                    {
                        tracing::error!(%job_id, %error, "critical pressure process kill failed");
                    }
                }
                self.queue_termination(job_id, attempt, Duration::ZERO, Duration::ZERO);
            }
        }
        Ok(())
    }
}
