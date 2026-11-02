use super::*;

impl DaemonActor {
    pub(crate) fn reject_before_attempt(
        &mut self,
        job: &JobRecord,
        message: String,
    ) -> Result<(), ActorError> {
        let at = now_timestamp();
        let receipt = self
            .store
            .transition_before_effect(prelaunch_rejection(job, message, at))?;
        self.record_transition(&receipt)?;
        self.notify_waiters(&receipt.job);
        Ok(())
    }

    pub(crate) fn install_running(
        &mut self,
        parts: RunningParts,
        job: JobRecord,
        allocation: Allocation,
    ) {
        let RunningParts {
            attempt,
            pid,
            cgroup,
            control,
            io,
            output,
            process,
        } = parts;
        let (control, receiver) = control.split();
        let output_drains_pending = output_reader_count(&output);
        spawn_control_reports(receiver, attempt, self.sender.clone());
        spawn_output_drains(output, attempt, self.journal.clone(), self.sender.clone());
        spawn_process_waiter(process, attempt, self.sender.clone());
        self.schedule_deadline(&job);
        let pid_start_time = pid_start_time(pid);
        let mps_managed = allocation_uses_mps(&allocation);
        self.active.insert(
            job.id,
            ActiveWorkload {
                attempt,
                pid,
                pid_start_time,
                cgroup,
                control: Some(control),
                control_sequence: 0,
                io: Some(io),
                resources: job.spec.resources,
                priority: job.spec.priority,
                allocation,
                tenant: job.spec.tenant.clone(),
                preemptible: job.spec.preemption.preemptible,
                started_at: now_timestamp(),
                fingerprint: self.resolver.fingerprint(&job.spec.workload),
                max_observed_vram_bytes: 0,
                shared_cuda_oom: false,
                pending_failure: None,
                mps_managed,
                overload_paused: false,
                cancelling: false,
                leader_exit: None,
                output_drains_pending,
                control_drained: !recovery::controlled_resident(&job.spec),
                process_tree_drained: false,
                tree_watcher_started: false,
            },
        );
    }

    pub(crate) fn observe_process_exit(
        &mut self,
        attempt: FencedAttempt,
        status: std::io::Result<ExitStatus>,
    ) -> Result<(), ActorError> {
        // A resident leader exit starts a bounded drain even if descendants retain the control fd.
        self.begin_control_recovery(attempt, FailureClass::Infrastructure)?;
        let Some(active) = self.active.get_mut(&attempt.job_id) else {
            return Ok(());
        };
        if active.attempt != attempt || active.leader_exit.is_some() {
            return Ok(());
        }
        active.leader_exit = Some(status);
        if let Some(cgroup) = active.cgroup.clone() {
            if recovery::cgroup_confirmed_empty(&cgroup) {
                active.process_tree_drained = true;
            } else if !active.tree_watcher_started {
                active.tree_watcher_started = true;
                spawn_fresh_cgroup_watcher(cgroup, attempt, self.sender.clone());
            }
        } else if process_group_exists(active.pid) {
            if !active.tree_watcher_started {
                active.tree_watcher_started = true;
                spawn_process_group_watcher(active.pid, attempt, self.sender.clone());
            }
        } else {
            active.process_tree_drained = true;
        }
        self.finish_process_if_drained(attempt)
    }

    pub(crate) fn observe_output_drained(
        &mut self,
        attempt: FencedAttempt,
    ) -> Result<(), ActorError> {
        if let Some(active) = self.active.get_mut(&attempt.job_id) {
            if active.attempt == attempt && active.output_drains_pending > 0 {
                active.output_drains_pending -= 1;
            }
        }
        self.finish_process_if_drained(attempt)
    }

    pub(crate) fn observe_process_group_drained(
        &mut self,
        attempt: FencedAttempt,
    ) -> Result<(), ActorError> {
        if let Some(active) = self.active.get_mut(&attempt.job_id) {
            if active.attempt == attempt {
                active.process_tree_drained = true;
            }
        }
        self.finish_process_if_drained(attempt)
    }

    pub(crate) fn observe_fresh_cgroup_drained(
        &mut self,
        attempt: FencedAttempt,
        cgroup: &JobCgroup,
    ) -> Result<(), ActorError> {
        if let Some(active) = self.active.get_mut(&attempt.job_id) {
            if active.attempt == attempt
                && active.cgroup.as_ref() == Some(cgroup)
                && recovery::cgroup_confirmed_empty(cgroup)
            {
                active.process_tree_drained = true;
            }
        }
        self.finish_process_if_drained(attempt)
    }

    pub(crate) fn finish_process_if_drained(
        &mut self,
        attempt: FencedAttempt,
    ) -> Result<(), ActorError> {
        let ready = self.active.get(&attempt.job_id).is_some_and(|active| {
            active.attempt == attempt
                && completion_ready(
                    active.leader_exit.is_some(),
                    active.process_tree_drained,
                    active.output_drains_pending,
                    active.control_drained,
                )
        });
        if !ready {
            return Ok(());
        }
        let status = self
            .active
            .get_mut(&attempt.job_id)
            .and_then(|active| active.leader_exit.take())
            .expect("drained active workload has a leader exit status");
        self.finish_process(attempt, status)?;
        self.request_schedule();
        Ok(())
    }

    pub(crate) fn launch_failed(
        &mut self,
        job: &JobRecord,
        message: String,
    ) -> Result<(), ActorError> {
        let current = self
            .store
            .get_job(job.id)?
            .ok_or(StoreError::JobNotFound(job.id))?;
        let at = now_timestamp();
        let can_retry = current.spec.retry.allows(FailureClass::Infrastructure)
            && current.failure_count < current.spec.retry.max_failures;
        let target = if can_retry {
            JobState::RetryWait
        } else {
            JobState::Failed
        };
        let receipt = self.store.transition_before_effect(TransitionMutation {
            transition: FencedTransition {
                job_id: current.id,
                from: current.state,
                to: target,
                attempt: current.active_attempt,
                at,
                reason: Some(message.clone()),
            },
            terminal: (target == JobState::Failed).then_some(TerminalStatus::Failed {
                class: FailureClass::Infrastructure,
                message,
                exit_code: None,
                at,
            }),
            failure: Some(FailureClass::Infrastructure),
        })?;
        self.record_transition(&receipt)?;
        if target == JobState::RetryWait {
            let delay = retry_delay(&receipt.job);
            let sender = self.sender.clone();
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                let _ = sender.send(ActorCommand::Requeue(receipt.job.id)).await;
            });
        } else {
            self.notify_waiters(&receipt.job);
        }
        Ok(())
    }

    pub(crate) fn finish_process(
        &mut self,
        attempt: FencedAttempt,
        status: std::io::Result<ExitStatus>,
    ) -> Result<(), ActorError> {
        if self
            .active
            .get(&attempt.job_id)
            .is_some_and(|active| active.attempt != attempt)
        {
            return Ok(());
        }
        let active = self.active.remove(&attempt.job_id);
        let cgroup_metrics = active
            .as_ref()
            .and_then(|active| active.cgroup.as_ref())
            .and_then(|cgroup| self.cgroups.metrics(cgroup).ok());
        if let Some(active) = &active {
            if let Some(cgroup) = &active.cgroup {
                let _ = self.cgroups.remove_if_empty(cgroup);
            }
        }
        let Some(job) = self.store.get_job(attempt.job_id)? else {
            return Ok(());
        };
        if job.state == JobState::Suspended {
            let receipt = self.transition(&job, JobState::Queued, "preempted process drained")?;
            self.record_transition(&receipt)?;
            return Ok(());
        }
        if job.state == JobState::RetryWait {
            self.schedule_retry_after_remaining(&job);
            return Ok(());
        }
        if job.state.is_terminal() || job.active_attempt != Some(attempt) {
            return Ok(());
        }
        let at = now_timestamp();
        let shared_oom = active.as_ref().is_some_and(|active| active.shared_cuda_oom);
        let reported_failure = recovery::resident_exit_failure(
            recovery::controlled_resident(&job.spec),
            active
                .as_ref()
                .and_then(|active| active.pending_failure.as_ref())
                .map(|(class, _)| *class),
            self.store
                .get_meta(&recovery::control_recovery_origin_key(attempt))?,
        );
        let reported_message = active
            .as_ref()
            .and_then(|active| active.pending_failure.as_ref())
            .map(|(_, message)| message.as_str());
        let host_oom = cgroup_metrics.is_some_and(|metrics| metrics.oom_kills > 0);
        let failure = exit_failure_class(
            status.as_ref().is_ok_and(ExitStatus::success),
            status.is_err(),
            host_oom,
            shared_oom,
            reported_failure,
        );
        let successful = failure.is_none();
        self.update_gpu_history(active.as_ref(), successful, failure, at)?;

        let can_retry = failure.is_some_and(|class| {
            class == FailureClass::DaemonRestart
                || (job.spec.retry.allows(class) && job.failure_count < job.spec.retry.max_failures)
        });
        let target = if successful {
            JobState::Completed
        } else if can_retry {
            JobState::RetryWait
        } else {
            JobState::Failed
        };
        let exit_code = status.as_ref().ok().and_then(ExitStatus::code);
        let message = match (&status, reported_message) {
            (_, Some(message)) => message.to_owned(),
            (Ok(status), None) if shared_oom => {
                format!("worker reported shared CUDA OOM ({status})")
            }
            (Ok(status), None) if host_oom => {
                format!("cgroup reported host OOM kill ({status})")
            }
            (Ok(status), None) => format!("worker exited with {status}"),
            (Err(error), None) => format!("worker wait failed: {error}"),
        };
        let terminal = match target {
            JobState::Completed => Some(TerminalStatus::Completed {
                exit_code: exit_code.unwrap_or(0),
                at,
            }),
            JobState::Failed => Some(TerminalStatus::Failed {
                class: failure.expect("failed target has failure class"),
                message: message.clone(),
                exit_code,
                at,
            }),
            JobState::RetryWait => None,
            _ => unreachable!("exit target is terminal or retry_wait"),
        };
        let receipt = self.store.transition_before_effect(TransitionMutation {
            transition: FencedTransition {
                job_id: job.id,
                from: job.state,
                to: target,
                attempt: job.active_attempt,
                at,
                reason: Some(message),
            },
            terminal,
            failure,
        })?;
        self.record_transition(&receipt)?;
        if target == JobState::RetryWait {
            let delay = retry_delay(&receipt.job);
            let sender = self.sender.clone();
            let job_id = receipt.job.id;
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                let _ = sender.send(ActorCommand::Requeue(job_id)).await;
            });
        } else {
            self.notify_waiters(&receipt.job);
        }
        Ok(())
    }
}
