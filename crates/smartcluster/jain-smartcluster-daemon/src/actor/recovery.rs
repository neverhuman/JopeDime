use super::mps::{allocation_uses_mps, job_may_use_mps};
use super::*;

mod recovery_scan;

impl DaemonActor {
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
