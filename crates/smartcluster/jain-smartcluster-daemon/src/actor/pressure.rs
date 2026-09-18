use super::*;

impl DaemonActor {
    fn pressure_index(resource: PsiResource) -> usize {
        match resource {
            PsiResource::Cpu => 0,
            PsiResource::Memory => 1,
            PsiResource::Io => 2,
        }
    }

    fn arm_pressure_quiet(&mut self, resource: PsiResource) {
        let index = Self::pressure_index(resource);
        self.pressure_generations[index] = self.pressure_generations[index].wrapping_add(1);
        let generation = self.pressure_generations[index];
        let sender = self.sender.clone();
        let quiet = self.recovery_quiet;
        tokio::spawn(async move {
            tokio::time::sleep(quiet).await;
            let _ = sender
                .send(ActorCommand::PressureQuiet {
                    resource,
                    generation,
                })
                .await;
        });
    }

    pub(super) async fn handle_pressure(&mut self, event: PsiEvent) {
        let index = Self::pressure_index(event.resource);
        self.pressure_levels[index] = event.level;
        self.health.set_pressure(event.resource, event.level);
        if event.level != PressureLevel::Normal {
            self.arm_pressure_quiet(event.resource);
        }
        let decision = self
            .overload
            .observe(self.aggregate_pressure(), Instant::now());
        self.health.set_overload_stage(decision.current);
        if !decision.changed {
            return;
        }
        match decision.current {
            OverloadStage::BurstReduced => {
                for active in self.active.values() {
                    if let Some(cgroup) = &active.cgroup {
                        let _ = self.cgroups.set_cpu_max(
                            cgroup,
                            u64::from(active.resources.cpu.minimum_millicores),
                        );
                    }
                }
            }
            OverloadStage::CpuJobsFrozen => {
                let candidate = self
                    .active
                    .iter()
                    .filter(|(_, active)| active.resources.gpu.is_none() && !active.cancelling)
                    .max_by_key(|(_, active)| active.priority.rank())
                    .map(|(job_id, _)| *job_id);
                if let Some(job_id) = candidate {
                    if let Some(active) = self.active.get(&job_id) {
                        if active.cgroup.is_some() {
                            let attempt = active.attempt;
                            if let Ok(Some(job)) = self.store.get_job(job_id) {
                                if let Err(error) = self
                                    .store
                                    .put_meta(&attempt_overload_pause_key(attempt), &true)
                                {
                                    tracing::error!(%job_id, %error, "cannot persist overload-pause provenance");
                                    return;
                                }
                                if let Ok(receipt) =
                                    self.transition(&job, JobState::Paused, "PSI overload pause")
                                {
                                    if self.record_transition(&receipt).is_ok() {
                                        if let Some(active) = self.active.get_mut(&job_id) {
                                            active.overload_paused = true;
                                        }
                                        self.queue_freezer(job_id, attempt, true);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            OverloadStage::CheckpointRequested => {
                let candidate = self.active.iter().find_map(|(job_id, active)| {
                    active.preemptible.then_some((*job_id, active.attempt))
                });
                if let Some((job_id, attempt)) = candidate {
                    if let Ok(Some(job)) = self.store.get_job(job_id) {
                        match job.spec.preemption.checkpoint {
                            jain_smartcluster_core::CheckpointPolicy::ControlFd { deadline_ms } => {
                                if let Some(active) = self.active.get_mut(&job_id) {
                                    if let Some(mut control) = active.control.take() {
                                        active.control_sequence =
                                            active.control_sequence.saturating_add(1);
                                        let frame = WorkerControlFrame {
                                            version: ProtocolVersion::CURRENT,
                                            attempt: active.attempt,
                                            sequence: active.control_sequence,
                                            control:
                                                jain_smartcluster_core::ControlMessage::Checkpoint {
                                                    deadline_ms: deadline_ms
                                                        .min(self.execution.checkpoint_timeout_ms),
                                                },
                                        };
                                        let timeout = self.execution.actor_io_timeout();
                                        tokio::spawn(async move {
                                            let _ =
                                                tokio::time::timeout(timeout, control.send(&frame))
                                                    .await;
                                        });
                                    }
                                }
                            }
                            jain_smartcluster_core::CheckpointPolicy::Signal { signal, .. } => {
                                if let Ok(signal) = Signal::try_from(signal) {
                                    let _ =
                                        self.signal_active_safely(job_id, attempt, signal).await;
                                }
                            }
                            jain_smartcluster_core::CheckpointPolicy::None => {}
                        }
                    }
                }
            }
            OverloadStage::GpuJobsRequeued => {
                if let Err(error) = self.requeue_gpu_for_pressure().await {
                    tracing::error!(%error, "GPU pressure reclaim failed");
                }
            }
            OverloadStage::HardKillAuthorized if event.level == PressureLevel::Critical => {
                if let Err(error) = self.hard_kill_for_pressure() {
                    tracing::error!(%error, "critical pressure termination failed");
                }
            }
            _ => {}
        }
    }

    pub(super) async fn handle_pressure_quiet(&mut self, resource: PsiResource, generation: u64) {
        let index = Self::pressure_index(resource);
        if self.pressure_generations[index] != generation {
            return;
        }
        self.pressure_levels[index] = PressureLevel::Normal;
        self.health.set_pressure(resource, PressureLevel::Normal);
        if self.aggregate_pressure() != PressureLevel::Normal {
            self.health.set_overload_stage(self.overload.stage());
            return;
        }
        let decision = self.overload.observe(PressureLevel::Normal, Instant::now());
        self.health.set_overload_stage(decision.current);
        if !decision.changed
            || decision.current != OverloadStage::Normal
            || self.gpu_global_quarantine
            || !self.gpu_quarantine.is_empty()
        {
            return;
        }

        for active in self.active.values() {
            if let Some(cgroup) = &active.cgroup {
                let _ = self
                    .cgroups
                    .set_cpu_max(cgroup, u64::from(active.resources.cpu.maximum_millicores));
            }
        }
        let paused: Vec<_> = self
            .active
            .iter()
            .filter(|(_, active)| active.overload_paused)
            .map(|(job_id, active)| (*job_id, active.attempt))
            .collect();
        for (job_id, attempt) in paused {
            let Ok(Some(job)) = self.store.get_job(job_id) else {
                continue;
            };
            if job.state != JobState::Paused {
                continue;
            }
            let Ok(receipt) = self.transition(
                &job,
                JobState::Running,
                "PSI pressure recovered after quiet window",
            ) else {
                continue;
            };
            if self.record_transition(&receipt).is_err() {
                continue;
            }
            self.queue_freezer(job_id, attempt, false);
        }
        self.request_schedule();
    }

    pub(super) async fn requeue_gpu_for_pressure(&mut self) -> Result<(), ActorError> {
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

    pub(super) fn hard_kill_for_pressure(&mut self) -> Result<(), ActorError> {
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

    pub(super) async fn handle_gpu_event(&mut self, event: GpuEvent) -> Result<(), ActorError> {
        tracing::warn!(?event, "NVML fault event clamped GPU admissions");
        if matches!(
            &event.kind,
            crate::gpu::GpuEventKind::DeviceLost | crate::gpu::GpuEventKind::MonitorFailed(_)
        ) {
            let reason = format!("NVML safety event delivery stopped: {:?}", event.kind);
            self.gpu.mps.disable(reason.clone());
            self.health.disable_mps(reason);
        }
        let decision = self
            .overload
            .observe(PressureLevel::Critical, Instant::now());
        self.health.set_overload_stage(decision.current);
        self.arm_pressure_quiet(PsiResource::Memory);
        if let Some(device_uuid) = &event.device_uuid {
            self.gpu_quarantine.insert(device_uuid.clone());
            self.store
                .put_meta("gpu-quarantine", &self.gpu_quarantine)?;
            for device in &mut self.gpu.devices {
                if &device.uuid == device_uuid || device.parent_uuid.as_ref() == Some(device_uuid) {
                    device.healthy = false;
                }
            }
        } else {
            self.gpu_global_quarantine = true;
            self.store.put_meta("gpu-global-quarantine", &true)?;
            for device in &mut self.gpu.devices {
                device.healthy = false;
            }
        }
        self.health.update_gpu_inventory(&self.gpu);
        let affected: Vec<_> = self
            .active
            .iter()
            .filter(|(_, active)| {
                !active.allocation.gpu.is_empty()
                    && event.device_uuid.as_ref().is_none_or(|device_uuid| {
                        active.allocation.gpu.iter().any(|gpu| {
                            &gpu.device_id == device_uuid
                                || self.gpu.devices.iter().any(|device| {
                                    device.uuid == gpu.device_id
                                        && device.parent_uuid.as_ref() == Some(device_uuid)
                                })
                        })
                    })
            })
            .map(|(job_id, _)| *job_id)
            .collect();
        for job_id in affected {
            let Some(job) = self.store.get_job(job_id)? else {
                continue;
            };
            if !matches!(job.state, JobState::Running | JobState::Paused) {
                continue;
            }
            let failure = FailureClass::GpuTransient;
            let can_retry =
                job.spec.retry.allows(failure) && job.failure_count < job.spec.retry.max_failures;
            let target = if can_retry {
                JobState::RetryWait
            } else {
                JobState::Failed
            };
            let at = now_timestamp();
            let message = format!("NVML fault: {:?}", event.kind);
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
                self.queue_termination(
                    job_id,
                    attempt,
                    Duration::ZERO,
                    self.execution.kill_grace(),
                );
            }
        }
        Ok(())
    }

    fn aggregate_pressure(&self) -> PressureLevel {
        if self.pressure_levels.contains(&PressureLevel::Critical) {
            PressureLevel::Critical
        } else if self.pressure_levels.contains(&PressureLevel::Warning) {
            PressureLevel::Warning
        } else {
            PressureLevel::Normal
        }
    }

    pub(super) async fn handle_worker_report(
        &mut self,
        frame: WorkerReportFrame,
    ) -> Result<(), ActorError> {
        if !ProtocolVersion::CURRENT.is_compatible_with(frame.version) {
            return Err(ActorError::Recovery(
                "worker report protocol version mismatch".to_owned(),
            ));
        }
        frame
            .report
            .validate()
            .map_err(|error| ActorError::Recovery(error.to_string()))?;
        let job_id = frame.attempt.job_id;
        let Some(active) = self.active.get_mut(&job_id) else {
            return Ok(());
        };
        if active.attempt != frame.attempt {
            return Err(ActorError::Recovery(
                "superseded worker report fence".to_owned(),
            ));
        }

        let mut shared_oom = false;
        match frame.report {
            WorkerReport::Progress(progress) => {
                self.journal.append(
                    job_id,
                    Some(frame.attempt),
                    JobEvent::Progress(progress),
                    now_timestamp(),
                )?;
            }
            WorkerReport::Metrics(metrics) => {
                active.max_observed_vram_bytes = active
                    .max_observed_vram_bytes
                    .max(metrics.gpus.iter().map(|gpu| gpu.vram_bytes).sum());
                self.journal.append(
                    job_id,
                    Some(frame.attempt),
                    JobEvent::Metrics(metrics),
                    now_timestamp(),
                )?;
            }
            WorkerReport::CheckpointCommitted(checkpoint) => {
                self.journal.append(
                    job_id,
                    Some(frame.attempt),
                    JobEvent::Checkpoint(checkpoint),
                    now_timestamp(),
                )?;
            }
            WorkerReport::Failure { class, message } => {
                if self
                    .store
                    .get_job(job_id)?
                    .is_some_and(|job| recovery::controlled_resident(&job.spec))
                {
                    // An explicit fenced worker failure supersedes a generic control-loss origin.
                    self.store.put_meta(
                        &recovery::control_recovery_origin_key(frame.attempt),
                        &class,
                    )?;
                }
                active.pending_failure = Some((class, message));
                if class == FailureClass::SharedCudaOutOfMemory {
                    active.shared_cuda_oom = true;
                    shared_oom = true;
                }
            }
            WorkerReport::Ready
            | WorkerReport::CheckpointAcknowledged { .. }
            | WorkerReport::Application { .. } => {}
        }

        if shared_oom {
            let fingerprint = active.fingerprint.clone();
            let observed = active.max_observed_vram_bytes;
            if let Some(fingerprint) = fingerprint {
                self.store
                    .update_resource_history(&fingerprint, now_timestamp(), |history| {
                        history.record_shared_cuda_oom(observed, now_timestamp())
                    })?;
            }
            let Some(job) = self.store.get_job(job_id)? else {
                return Ok(());
            };
            if matches!(job.state, JobState::Running | JobState::Paused) {
                let at = now_timestamp();
                let receipt = self.store.transition_before_effect(TransitionMutation {
                    transition: FencedTransition {
                        job_id,
                        from: job.state,
                        to: JobState::RetryWait,
                        attempt: job.active_attempt,
                        at,
                        reason: Some("worker reported shared CUDA OOM".to_owned()),
                    },
                    terminal: None,
                    failure: Some(FailureClass::SharedCudaOutOfMemory),
                })?;
                self.record_transition(&receipt)?;
                let attempt = if let Some(active) = self.active.get_mut(&job_id) {
                    active.cancelling = true;
                    Some(active.attempt)
                } else {
                    None
                };
                if let Some(attempt) = attempt {
                    self.queue_termination(job_id, attempt, Duration::ZERO, Duration::ZERO);
                }
            }
        }
        Ok(())
    }
}
