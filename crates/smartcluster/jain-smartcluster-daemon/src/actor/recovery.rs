use super::mps::{allocation_uses_mps, job_may_use_mps};
use super::*;

impl DaemonActor {
    pub(super) async fn recover(&mut self) -> Result<(), ActorError> {
        self.gpu_quarantine = self.store.get_meta("gpu-quarantine")?.unwrap_or_default();
        self.gpu_global_quarantine = self
            .store
            .get_meta("gpu-global-quarantine")?
            .unwrap_or(false);
        for device in &mut self.gpu.devices {
            device.healthy = device.healthy
                && !self.gpu_global_quarantine
                && !self.gpu_quarantine.contains(&device.uuid)
                && device
                    .parent_uuid
                    .as_ref()
                    .is_none_or(|parent| !self.gpu_quarantine.contains(parent));
        }
        self.health.update_gpu_inventory(&self.gpu);
        for pending in self.store.pending_transition_events()? {
            self.publish_transition(&pending.transition)?;
        }
        let discovered = match self.cgroups.list_attempts() {
            Ok(cgroups) => cgroups,
            Err(_) if self.cgroups.mode() == crate::cgroup::IsolationMode::SingleAllocation => {
                Vec::new()
            }
            Err(error) => return Err(error.into()),
        };
        let mut jobs = self.store.all_jobs()?;
        let admitted_without_attempt: Vec<_> = jobs
            .iter()
            .filter(|job| job.state == JobState::Admitted && job.active_attempt.is_none())
            .cloned()
            .collect();
        for job in admitted_without_attempt {
            let receipt = self.transition(
                &job,
                JobState::Queued,
                "restart recovered admission before attempt creation",
            )?;
            self.record_transition(&receipt)?;
        }
        jobs = self.store.all_jobs()?;
        for job in &jobs {
            self.schedule_deadline(job);
        }
        let mut expected = Vec::new();
        let mut retired = Vec::new();
        let mut all_attempts = Vec::new();
        for job in &jobs {
            all_attempts.extend(self.store.attempts_for_job(job.id)?);
            if matches!(
                job.state,
                JobState::Starting | JobState::Running | JobState::Paused
            ) {
                if let Some(attempt) = job.active_attempt {
                    expected.push(crate::recovery::ExpectedAttempt {
                        key: crate::cgroup::AttemptKey {
                            job_id: attempt.job_id.to_string(),
                            attempt: attempt.number.get(),
                            fence: attempt.fence.to_string(),
                        },
                        state: match job.state {
                            JobState::Starting => crate::recovery::RecoverableState::Starting,
                            JobState::Paused => crate::recovery::RecoverableState::Paused,
                            _ => crate::recovery::RecoverableState::Running,
                        },
                    });
                }
            }
        }
        for attempt in &all_attempts {
            if matches!(
                attempt.state,
                JobState::Suspended
                    | JobState::RetryWait
                    | JobState::Completed
                    | JobState::Failed
                    | JobState::Cancelled
            ) {
                retired.push(crate::recovery::RetiredAttempt {
                    key: crate::cgroup::AttemptKey {
                        job_id: attempt.attempt.job_id.to_string(),
                        attempt: attempt.attempt.number.get(),
                        fence: attempt.attempt.fence.to_string(),
                    },
                    state: attempt.state,
                });
            }
        }

        if self.cgroups.mode() == crate::cgroup::IsolationMode::SingleAllocation {
            if jobs
                .iter()
                .any(|job| !job.state.is_terminal() && controlled_resident(&job.spec))
            {
                return Err(ActorError::Recovery(
                    "controlled resident recovery requires delegated cgroups".to_owned(),
                ));
            }
            for attempt in &all_attempts {
                let expected_live = jobs.iter().any(|job| {
                    job.active_attempt == Some(attempt.attempt)
                        && matches!(
                            job.state,
                            JobState::Starting | JobState::Running | JobState::Paused
                        )
                });
                if let Some(pid) = attempt.pid {
                    if process_exists(pid) || process_group_exists(pid) {
                        let expected: Option<u64> = self
                            .store
                            .get_meta(&attempt_pid_start_key(attempt.attempt))?;
                        let Some(expected) = expected else {
                            if expected_live {
                                return Err(ActorError::Recovery(format!(
                                    "safe-mode attempt {:?} has no durable pid fingerprint",
                                    attempt.attempt
                                )));
                            }
                            continue;
                        };
                        if process_exists(pid) && pid_start_time(pid) != Some(expected) {
                            continue;
                        }
                        if process_group_exists(pid) {
                            killpg(Pid::from_raw(pid as i32), Signal::SIGKILL)?;
                        }
                        let drained = tokio::time::timeout(Duration::from_millis(500), async {
                            while process_group_exists(pid) {
                                tokio::time::sleep(Duration::from_millis(10)).await;
                            }
                        })
                        .await
                        .is_ok();
                        if !drained {
                            return Err(ActorError::Recovery(format!(
                                "safe single-allocation restart could not drain pid group {pid} for {}",
                                attempt.attempt.job_id
                            )));
                        }
                    }
                }
            }
        }

        let mut retry_wait_with_live_cgroup = std::collections::BTreeSet::new();
        for action in crate::recovery::reconcile_attempts(expected, retired, discovered) {
            match action {
                crate::recovery::RecoveryAction::Adopt { expected, cgroup } => {
                    tracing::info!(job_id = %expected.key.job_id, cgroup = %cgroup.path.display(), "adopted live fenced cgroup");
                    let job_id = expected
                        .key
                        .job_id
                        .parse::<JobId>()
                        .map_err(|error| ActorError::Recovery(error.to_string()))?;
                    let Some(job) = self.store.get_job(job_id)? else {
                        continue;
                    };
                    let Some(attempt) = job.active_attempt else {
                        continue;
                    };
                    let attempt_record =
                        self.store
                            .get_attempt(attempt)?
                            .ok_or(StoreError::AttemptNotFound {
                                job_id,
                                attempt: attempt.number.get(),
                            })?;
                    self.store.renew_lease(
                        attempt,
                        next_lease_expiry(attempt_record.lease.expires_at, now_timestamp()),
                    )?;
                    let pid = cgroup.pids.first().copied().ok_or_else(|| {
                        ActorError::Recovery("populated cgroup has no pid".to_owned())
                    })? as u32;
                    let was_frozen = cgroup.frozen;
                    let job_cgroup = JobCgroup {
                        key: cgroup.key,
                        name: cgroup.name,
                        path: cgroup.path,
                    };
                    let allocation: Allocation = self
                        .store
                        .get_meta(&attempt_allocation_key(attempt))?
                        .ok_or_else(|| {
                            ActorError::Recovery(format!(
                                "live attempt {attempt:?} has no durable resource allocation"
                            ))
                        })?;
                    let mps_managed = allocation_uses_mps(&allocation);
                    let overload_paused = job.state == JobState::Paused
                        && self
                            .store
                            .get_meta::<bool>(&attempt_overload_pause_key(attempt))?
                            .unwrap_or(false);
                    if mps_managed && self.mps_controller.is_none() {
                        return Err(ActorError::MpsSafety(format!(
                            "cannot adopt shared MPS attempt {attempt:?} without a probed controller"
                        )));
                    }
                    if job.state == JobState::Paused && !was_frozen {
                        self.cgroups.freeze(&job_cgroup)?;
                    } else if matches!(job.state, JobState::Running | JobState::Starting)
                        && was_frozen
                    {
                        self.cgroups.resume(&job_cgroup)?;
                    }
                    spawn_adopted_watcher(job_cgroup.clone(), attempt, self.sender.clone());
                    self.active.insert(
                        job_id,
                        ActiveWorkload {
                            attempt,
                            pid,
                            pid_start_time: pid_start_time(pid),
                            cgroup: Some(job_cgroup),
                            control: None,
                            control_sequence: 0,
                            io: None,
                            resources: job.spec.resources,
                            priority: job.spec.priority,
                            allocation,
                            tenant: job.spec.tenant.clone(),
                            preemptible: job.spec.preemption.preemptible,
                            started_at: job.updated_at,
                            fingerprint: self.resolver.fingerprint(&job.spec.workload),
                            max_observed_vram_bytes: 0,
                            shared_cuda_oom: false,
                            pending_failure: None,
                            mps_managed,
                            overload_paused,
                            cancelling: false,
                            leader_exit: None,
                            output_drains_pending: 0,
                            control_drained: true,
                            process_tree_drained: false,
                            tree_watcher_started: true,
                        },
                    );
                    if controlled_resident(&job.spec) {
                        self.begin_control_recovery(attempt, FailureClass::DaemonRestart)?;
                    }
                    if job.state == JobState::Starting {
                        let receipt = self.transition(
                            &job,
                            JobState::Running,
                            "live starting cgroup adopted after restart",
                        )?;
                        self.record_transition(&receipt)?;
                    }
                }
                crate::recovery::RecoveryAction::RetryWithoutFailureCharge { expected } => {
                    let job_id = expected
                        .key
                        .job_id
                        .parse::<JobId>()
                        .map_err(|error| ActorError::Recovery(error.to_string()))?;
                    if let Some(job) = self.store.get_job(job_id)? {
                        let at = now_timestamp();
                        let failure = match job.active_attempt {
                            Some(attempt) => self
                                .store
                                .get_meta::<FailureClass>(&control_recovery_origin_key(attempt))?
                                .unwrap_or(FailureClass::DaemonRestart),
                            None => FailureClass::DaemonRestart,
                        };
                        let retry = failure == FailureClass::DaemonRestart
                            || (job.spec.retry.allows(failure)
                                && job.failure_count < job.spec.retry.max_failures);
                        let target = if retry {
                            JobState::RetryWait
                        } else {
                            JobState::Failed
                        };
                        let receipt = self.store.transition_before_effect(TransitionMutation {
                            transition: FencedTransition {
                                job_id,
                                from: job.state,
                                to: target,
                                attempt: job.active_attempt,
                                at,
                                reason: Some(
                                    "daemon restart found no live fenced cgroup".to_owned(),
                                ),
                            },
                            terminal: (!retry).then_some(TerminalStatus::Failed {
                                class: failure,
                                message: "recovery found no live fenced cgroup".to_owned(),
                                exit_code: None,
                                at,
                            }),
                            failure: Some(failure),
                        })?;
                        self.record_transition(&receipt)?;
                        if retry {
                            self.schedule_retry_after_remaining(&receipt.job);
                        } else {
                            self.notify_waiters(&receipt.job);
                        }
                    }
                }
                crate::recovery::RecoveryAction::CleanupEmpty { cgroup } => {
                    let job_cgroup = JobCgroup {
                        key: cgroup.key,
                        name: cgroup.name,
                        path: cgroup.path,
                    };
                    let _ = self.cgroups.remove_if_empty(&job_cgroup);
                }
                crate::recovery::RecoveryAction::Quarantine { cgroup, reason } => {
                    if cgroup.populated || !cgroup.pids.is_empty() {
                        return Err(ActorError::Recovery(format!(
                            "populated quarantined cgroup {} ({reason:?}) prevents safe readiness",
                            cgroup.path.display()
                        )));
                    }
                    tracing::warn!(path = %cgroup.path.display(), ?reason, "quarantined unknown cgroup without signaling it");
                }
                crate::recovery::RecoveryAction::ReapplyRetiredEffect { retired, cgroup } => {
                    let job_id = retired
                        .key
                        .job_id
                        .parse::<JobId>()
                        .map_err(|error| ActorError::Recovery(error.to_string()))?;
                    let Some(job) = self.store.get_job(job_id)? else {
                        continue;
                    };
                    let attempt = all_attempts
                        .iter()
                        .find(|attempt| {
                            attempt.attempt.job_id == job_id
                                && attempt.attempt.number.get() == retired.key.attempt
                                && attempt.attempt.fence.to_string() == retired.key.fence
                        })
                        .map(|attempt| attempt.attempt)
                        .ok_or_else(|| {
                            ActorError::Recovery("retired cgroup has no durable attempt".to_owned())
                        })?;
                    let pid = cgroup.pids.first().copied().ok_or_else(|| {
                        ActorError::Recovery("populated retired cgroup has no pid".to_owned())
                    })? as u32;
                    let job_cgroup = JobCgroup {
                        key: cgroup.key,
                        name: cgroup.name,
                        path: cgroup.path,
                    };
                    let allocation: Allocation = self
                        .store
                        .get_meta(&attempt_allocation_key(attempt))?
                        .ok_or_else(|| {
                            ActorError::Recovery(format!(
                                "retired live attempt {attempt:?} has no durable resource allocation"
                            ))
                        })?;
                    // The state transition already committed before the crash. Reapply its
                    // termination side effect idempotently before accepting new work.
                    if allocation_uses_mps(&allocation) || job_may_use_mps(&job) {
                        self.terminate_mps_clients(Some(job_cgroup.clone())).await?;
                    }
                    self.cgroups.kill(&job_cgroup)?;
                    spawn_adopted_watcher(job_cgroup.clone(), attempt, self.sender.clone());
                    self.active.insert(
                        job_id,
                        ActiveWorkload {
                            attempt,
                            pid,
                            pid_start_time: pid_start_time(pid),
                            cgroup: Some(job_cgroup),
                            control: None,
                            control_sequence: 0,
                            io: None,
                            resources: job.spec.resources,
                            priority: job.spec.priority,
                            mps_managed: allocation_uses_mps(&allocation) || job_may_use_mps(&job),
                            allocation,
                            tenant: job.spec.tenant.clone(),
                            preemptible: job.spec.preemption.preemptible,
                            started_at: job.updated_at,
                            fingerprint: self.resolver.fingerprint(&job.spec.workload),
                            max_observed_vram_bytes: 0,
                            shared_cuda_oom: false,
                            pending_failure: None,
                            overload_paused: false,
                            cancelling: true,
                            leader_exit: None,
                            output_drains_pending: 0,
                            control_drained: true,
                            process_tree_drained: false,
                            tree_watcher_started: true,
                        },
                    );
                    if retired.state == JobState::RetryWait {
                        retry_wait_with_live_cgroup.insert(job_id);
                    }
                }
            }
        }
        for job in jobs {
            if job.state == JobState::RetryWait && !retry_wait_with_live_cgroup.contains(&job.id) {
                self.schedule_retry_after_remaining(&job);
            } else if job.state == JobState::Suspended && !self.active.contains_key(&job.id) {
                let receipt = self.transition(
                    &job,
                    JobState::Queued,
                    "suspended process absent after restart",
                )?;
                self.record_transition(&receipt)?;
            }
        }
        Ok(())
    }

    pub(super) fn schedule_retry_after_remaining(&self, job: &JobRecord) {
        let delay = remaining_retry_delay(job, now_timestamp());
        let sender = self.sender.clone();
        let job_id = job.id;
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            let _ = sender.send(ActorCommand::Requeue(job_id)).await;
        });
    }

    pub(super) fn adopted_drained(
        &mut self,
        attempt: FencedAttempt,
        cgroup: &JobCgroup,
    ) -> Result<(), ActorError> {
        if !self.active.get(&attempt.job_id).is_some_and(|active| {
            active.attempt == attempt && active.cgroup.as_ref() == Some(cgroup)
        }) {
            return Ok(());
        }
        // Recheck the kernel state; a notification alone cannot release resources.
        if !cgroup_confirmed_empty(cgroup) {
            return Ok(());
        }
        self.active.remove(&attempt.job_id);
        let _ = self.cgroups.remove_if_empty(cgroup);
        let Some(job) = self.store.get_job(attempt.job_id)? else {
            return Ok(());
        };
        if job.state.is_terminal() {
            return Ok(());
        }
        if job.state == JobState::RetryWait {
            self.schedule_retry_after_remaining(&job);
            return Ok(());
        }
        if job.state == JobState::Suspended {
            let receipt = self.transition(&job, JobState::Queued, "preempted cgroup drained")?;
            self.record_transition(&receipt)?;
            return Ok(());
        }
        if job.active_attempt != Some(attempt) {
            return Ok(());
        }
        let at = now_timestamp();
        let failure = self
            .store
            .get_meta::<FailureClass>(&control_recovery_origin_key(attempt))?
            .unwrap_or(FailureClass::DaemonRestart);
        let retry = failure == FailureClass::DaemonRestart
            || (job.spec.retry.allows(failure) && job.failure_count < job.spec.retry.max_failures);
        let target = if retry {
            JobState::RetryWait
        } else {
            JobState::Failed
        };
        let message = "adopted cgroup drained without a terminal report".to_owned();
        let receipt = self.store.transition_before_effect(TransitionMutation {
            transition: FencedTransition {
                job_id: job.id,
                from: job.state,
                to: target,
                attempt: Some(attempt),
                at,
                reason: Some(message.clone()),
            },
            terminal: (!retry).then_some(TerminalStatus::Failed {
                class: failure,
                message,
                exit_code: None,
                at,
            }),
            failure: Some(failure),
        })?;
        self.record_transition(&receipt)?;
        if retry {
            self.schedule_retry_after_remaining(&receipt.job);
        } else {
            self.notify_waiters(&receipt.job);
        }
        Ok(())
    }
}

pub(super) fn controlled_resident(spec: &jain_smartcluster_core::JobSpec) -> bool {
    matches!(
        &spec.workload,
        jain_smartcluster_core::Workload::TrustedService { service, .. }
            if service.as_str() == "jope-resident-gpu"
    ) && matches!(
        spec.preemption.checkpoint,
        jain_smartcluster_core::CheckpointPolicy::ControlFd { .. }
    )
}

pub(super) fn resident_checkpoint_allowance(
    spec: &jain_smartcluster_core::JobSpec,
    execution: &ExecutionConfig,
) -> Result<Option<Duration>, String> {
    if !controlled_resident(spec) {
        return Ok(None);
    }
    let jain_smartcluster_core::CheckpointPolicy::ControlFd { deadline_ms } =
        spec.preemption.checkpoint
    else {
        unreachable!()
    };
    if deadline_ms > execution.checkpoint_timeout_ms {
        return Err(format!("resident checkpoint allowance {deadline_ms}ms exceeds configured checkpoint timeout {}ms", execution.checkpoint_timeout_ms));
    }
    Ok(Some(Duration::from_millis(deadline_ms)))
}

pub(super) fn control_recovery_origin_key(attempt: FencedAttempt) -> String {
    format!(
        "control-recovery-origin:{}:{}:{}",
        attempt.job_id,
        attempt.number.get(),
        attempt.fence
    )
}

fn record_control_recovery_origin(
    store: &Store,
    attempt: FencedAttempt,
    origin: FailureClass,
) -> Result<(), StoreError> {
    let key = control_recovery_origin_key(attempt);
    if store.get_meta::<FailureClass>(&key)?.is_none() {
        store.put_meta(&key, &origin)?;
    }
    Ok(())
}

pub(super) fn resident_exit_failure(
    resident: bool,
    reported: Option<FailureClass>,
    origin: Option<FailureClass>,
) -> Option<FailureClass> {
    reported.or_else(|| resident.then_some(origin.unwrap_or(FailureClass::Infrastructure)))
}

fn control_recovery_key(attempt: FencedAttempt) -> String {
    format!(
        "control-recovery-deadline:{}:{}:{}",
        attempt.job_id,
        attempt.number.get(),
        attempt.fence
    )
}

fn control_recovery_deadline(
    store: &Store,
    attempt: FencedAttempt,
    now: Timestamp,
    allowance: Duration,
) -> Result<Timestamp, StoreError> {
    let key = control_recovery_key(attempt);
    if let Some(deadline) = store.get_meta(&key)? {
        return Ok(deadline);
    }
    let deadline = Timestamp::from_unix_millis(
        now.as_unix_millis()
            .saturating_add(allowance.as_millis().try_into().unwrap_or(u64::MAX)),
    );
    store.put_meta(&key, &deadline)?;
    Ok(deadline)
}

impl DaemonActor {
    pub(super) fn begin_control_recovery(
        &mut self,
        attempt: FencedAttempt,
        origin: FailureClass,
    ) -> Result<(), ActorError> {
        if !self
            .active
            .get(&attempt.job_id)
            .is_some_and(|active| active.attempt == attempt)
        {
            return Ok(());
        }
        let Some(job) = self.store.get_job(attempt.job_id)? else {
            return Ok(());
        };
        if job.active_attempt != Some(attempt)
            || job.state.is_terminal()
            || !controlled_resident(&job.spec)
        {
            return Ok(());
        }
        let allowance = resident_checkpoint_allowance(&job.spec, &self.execution)
            .map_err(ActorError::Recovery)?
            .expect("checked resident contract");
        let now = now_timestamp();
        // Write intent before dropping the last control sender or thawing the worker.
        // Repeated daemon restarts reuse this exact absolute deadline.
        let reported = self
            .active
            .get(&attempt.job_id)
            .and_then(|active| active.pending_failure.as_ref())
            .map(|(class, _)| *class);
        record_control_recovery_origin(&self.store, attempt, reported.unwrap_or(origin))?;
        let deadline = control_recovery_deadline(&self.store, attempt, now, allowance)?;

        let active = self
            .active
            .get_mut(&attempt.job_id)
            .expect("checked active fence");
        active.control = None;
        active.cancelling = true;
        active.overload_paused = false;

        if let Some(cgroup) = &active.cgroup {
            self.cgroups.resume(cgroup)?;
        }
        self.queue_termination(
            attempt.job_id,
            attempt,
            Duration::from_millis(
                deadline
                    .as_unix_millis()
                    .saturating_sub(now.as_unix_millis()),
            ),
            self.execution.terminate_grace(),
        );
        Ok(())
    }
}

pub(super) fn cgroup_confirmed_empty(cgroup: &JobCgroup) -> bool {
    std::fs::File::open(cgroup.path.join("cgroup.events"))
        .and_then(|file| cgroup_is_populated(&file))
        .is_ok_and(|populated| !populated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jain_smartcluster_core::{AttemptNumber, FenceToken};

    fn attempt(number: u32) -> FencedAttempt {
        FencedAttempt::new(
            JobId::new(),
            AttemptNumber::new(number).unwrap(),
            FenceToken::new(u64::from(number)).unwrap(),
        )
    }

    #[test]
    fn checkpoint_budget_is_durable_and_fenced_across_repeated_restarts() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.redb");
        let first = attempt(1);
        {
            let store = Store::open(&path).unwrap();
            assert_eq!(
                control_recovery_deadline(&store, first, 1000.into(), Duration::from_secs(120))
                    .unwrap(),
                Timestamp::from_unix_millis(121000)
            );
        }
        let store = Store::open(&path).unwrap();
        for now in [2000, 5000, 130000] {
            assert_eq!(
                control_recovery_deadline(&store, first, now.into(), Duration::from_secs(300))
                    .unwrap(),
                Timestamp::from_unix_millis(121000)
            );
        }
        let next = FencedAttempt::new(
            first.job_id,
            AttemptNumber::new(2).unwrap(),
            FenceToken::new(2).unwrap(),
        );
        assert_eq!(
            control_recovery_deadline(&store, next, 130000.into(), Duration::from_secs(120))
                .unwrap(),
            Timestamp::from_unix_millis(250000)
        );
    }

    #[test]
    fn drain_requires_kernel_subtree_empty_and_fails_closed_when_unreadable() {
        let directory = tempfile::tempdir().unwrap();
        let cgroup = JobCgroup {
            key: crate::cgroup::AttemptKey {
                job_id: "job".to_owned(),
                attempt: 1,
                fence: "1".to_owned(),
            },
            name: "test".to_owned(),
            path: directory.path().to_owned(),
        };
        assert!(!cgroup_confirmed_empty(&cgroup));
        std::fs::write(directory.path().join("cgroup.procs"), "").unwrap();
        std::fs::write(
            directory.path().join("cgroup.events"),
            "populated 1\nfrozen 0\n",
        )
        .unwrap();
        assert!(
            !cgroup_confirmed_empty(&cgroup),
            "empty leader cgroup cannot hide live descendants"
        );
        std::fs::write(
            directory.path().join("cgroup.events"),
            "populated 0\nfrozen 0\n",
        )
        .unwrap();
        assert!(cgroup_confirmed_empty(&cgroup));
    }

    #[test]
    fn fresh_resident_exit_order_is_charged_and_reported_failures_survive_eof() {
        for reported in [
            None,
            Some(FailureClass::Dependency),
            Some(FailureClass::Application),
        ] {
            for successful_exit in [true, false] {
                let before_eof = resident_exit_failure(true, reported, None);
                let after_eof =
                    resident_exit_failure(true, reported, Some(FailureClass::Infrastructure));
                assert_eq!(
                    exit_failure_class(successful_exit, false, false, false, before_eof),
                    exit_failure_class(successful_exit, false, false, false, after_eof)
                );
                assert_eq!(
                    before_eof,
                    Some(reported.unwrap_or(FailureClass::Infrastructure))
                );
            }
        }
        assert_eq!(resident_exit_failure(false, None, None), None);
        assert_eq!(
            resident_exit_failure(true, None, Some(FailureClass::DaemonRestart)),
            Some(FailureClass::DaemonRestart)
        );
    }

    #[test]
    fn restart_cannot_erase_durable_fresh_control_failure_charge() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("origin.redb");
        let attempt = attempt(1);
        {
            let store = Store::open(&path).unwrap();
            record_control_recovery_origin(&store, attempt, FailureClass::Infrastructure).unwrap();
        }
        let store = Store::open(&path).unwrap();
        record_control_recovery_origin(&store, attempt, FailureClass::DaemonRestart).unwrap();
        assert_eq!(
            store
                .get_meta::<FailureClass>(&control_recovery_origin_key(attempt))
                .unwrap(),
            Some(FailureClass::Infrastructure)
        );
    }

    #[tokio::test]
    async fn final_failure_report_is_ordered_before_control_drain() {
        use std::io::Write;
        let pair = crate::control::ControlPair::create().unwrap();
        let mut worker = pair.clone_worker_for_test().unwrap();
        let (_, receiver) = pair.close_worker_and_open_daemon(1024).unwrap().split();
        let attempt = attempt(1);
        let frame = WorkerReportFrame {
            version: ProtocolVersion::CURRENT,
            attempt,
            sequence: 1,
            report: WorkerReport::Failure {
                class: FailureClass::Dependency,
                message: "missing input".to_owned(),
            },
        };
        let bytes = postcard::to_stdvec(&frame).unwrap();
        worker
            .write_all(&(bytes.len() as u32).to_be_bytes())
            .unwrap();
        worker.write_all(&bytes).unwrap();
        drop(worker);
        let (sender, mut commands) = mpsc::channel(4);
        spawn_control_reports(receiver, attempt, sender);
        assert!(
            matches!(commands.recv().await, Some(ActorCommand::WorkerReport(got)) if got == frame)
        );
        assert!(
            matches!(commands.recv().await, Some(ActorCommand::ControlLost(got)) if got == attempt)
        );
    }

    #[tokio::test]
    async fn control_eof_emits_exact_attempt_fence() {
        let pair = crate::control::ControlPair::create().unwrap();
        let (_, receiver) = pair.close_worker_and_open_daemon(1024).unwrap().split();
        let (sender, mut commands) = mpsc::channel(4);
        let attempt = attempt(1);
        spawn_control_reports(receiver, attempt, sender);
        let command = tokio::time::timeout(Duration::from_secs(1), commands.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(command, ActorCommand::ControlLost(received) if received == attempt));
        assert!(commands.recv().await.is_none());
    }
}
