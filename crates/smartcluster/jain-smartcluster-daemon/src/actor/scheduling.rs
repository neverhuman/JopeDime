use super::mps::allocation_uses_mps;
use super::*;
use jain_smartcluster_core::PriorityClass;

impl DaemonActor {
    pub(super) async fn schedule_one(&mut self) -> Result<(), ActorError> {
        if !self.shutdown_replies.is_empty() {
            return Ok(());
        }
        self.rebalance_burst_limits();
        if self.overload.stage() != OverloadStage::Normal || self.health.status().paused {
            return Ok(());
        }
        if self.cgroups.mode() == crate::cgroup::IsolationMode::SingleAllocation
            && !self.active.is_empty()
        {
            return Ok(());
        }
        let now = now_timestamp();
        for job in self.store.expired_deadline_jobs(now, usize::MAX)? {
            if let Some(deadline) = job.spec.deadline {
                self.enforce_deadline(job.id, deadline)?;
            }
        }
        if let Err(reason) = self.execution.resource_policy.check_live() {
            tracing::warn!(%reason, "native admission held by resource reserve");
            return Ok(());
        }
        let jobs = self.store.queued_jobs(100_000)?;
        let gpu_demand = jobs.iter().any(|job| job.spec.resources.gpu.is_some())
            || self
                .active
                .values()
                .any(|active| !active.allocation.gpu.is_empty());
        if gpu_demand {
            self.refresh_gpu_availability().await;
        }
        let mut history_cache = HashMap::new();
        let mut queued = Vec::new();
        let mut cpu_pools = BTreeMap::new();
        for job in &jobs {
            if let Err(reason) = self
                .execution
                .resource_policy
                .validate_scratch_request(&job.spec)
            {
                self.reject_before_attempt(job, reason)?;
                continue;
            }
            if !self.execution.resource_policy.service_cpu_sets.is_empty() {
                cpu_pools.insert(
                    job.id,
                    self.execution
                        .resource_policy
                        .cpus_for(&job.spec, &self.capacity.cpus)
                        .map_err(ActorError::Recovery)?,
                );
            }
            queued.push(self.scheduling_candidate(job, &mut history_cache)?);
        }
        if !self.execution.resource_policy.service_cpu_sets.is_empty() {
            for active in self.active.values() {
                cpu_pools.insert(
                    active.attempt.job_id,
                    self.store
                        .get_meta(&attempt_cpu_pool_key(active.attempt))?
                        .unwrap_or_else(|| self.capacity.cpus.clone()),
                );
            }
        }
        let running = self
            .active
            .iter()
            .map(|(job_id, active)| RunningAllocation {
                job_id: *job_id,
                tenant: active.tenant.clone(),
                priority: active.priority,
                allocation: active.allocation.clone(),
                preemptible: active.preemptible && !active.cancelling,
                started_at: active.started_at,
            })
            .collect();
        let snapshot = SchedulingSnapshot {
            now,
            capacity: SchedulerCapacity {
                cpu_millicores: self.capacity.cpu_millicores,
                memory_bytes: self.capacity.memory_max_bytes,
                scratch_bytes: self
                    .execution
                    .resource_policy
                    .scratch_capacity()
                    .map_err(ActorError::Recovery)?,
                gpus: self
                    .gpu
                    .devices
                    .iter()
                    .map(|device| crate::scheduler::GpuDevice {
                        id: device.uuid.clone(),
                        kind: if device.mig_instance.is_some() {
                            GpuDeviceKind::MigInstance
                        } else {
                            GpuDeviceKind::Physical
                        },
                        vram_bytes: device.allocatable_vram_bytes,
                        healthy: device.healthy,
                    })
                    .collect(),
            },
            queued,
            running,
            tenant_weights: self.tenant_weights.clone(),
        };
        let plan = self
            .scheduler
            .schedule_with_cpu_pools(&snapshot, &cpu_pools)?;
        let admission_count = plan.admissions.len();
        let jobs_by_id: HashMap<_, _> = jobs.into_iter().map(|job| (job.id, job)).collect();
        let mut launched = 0_usize;
        for admission in plan.admissions {
            if self.cgroups.mode() == crate::cgroup::IsolationMode::SingleAllocation
                && !self.active.is_empty()
            {
                break;
            }
            let job = jobs_by_id
                .get(&admission.job_id)
                .cloned()
                .ok_or(StoreError::JobNotFound(admission.job_id))?;
            if !self.cpu_pool_available(&job)? {
                continue;
            }
            if let Err(reason) = self.execution.resource_policy.check_live() {
                tracing::warn!(%reason, "native admission held before launch");
                break;
            }
            let scratch_used = self.active.values().fold(0u64, |sum, active| {
                sum.saturating_add(active.allocation.resources.scratch_bytes)
            });
            let scratch_capacity = self
                .execution
                .resource_policy
                .scratch_capacity()
                .map_err(ActorError::Recovery)?;
            if scratch_used.saturating_add(job.spec.resources.scratch_bytes) > scratch_capacity {
                continue;
            }
            self.launch_job(job, admission.allocation).await?;
            launched = launched.saturating_add(1);
        }
        let mut preempted = 0_usize;
        for preemption in plan.preemptions {
            self.preempt(preemption.job_id, &preemption.reason).await?;
            preempted = preempted.saturating_add(1);
        }
        if launched == MAX_ADMISSIONS_PER_PASS
            || (launched > 0 && launched < admission_count)
            || preempted > 0
        {
            self.request_schedule();
        }
        Ok(())
    }

    fn cpu_pool_available(&self, job: &JobRecord) -> Result<bool, ActorError> {
        if self.execution.resource_policy.service_cpu_sets.is_empty() {
            return Ok(true);
        }
        let cpus = self
            .execution
            .resource_policy
            .cpus_for(&job.spec, &self.capacity.cpus)
            .map_err(ActorError::Recovery)?;
        let mut running = Vec::new();
        for active in self.active.values() {
            let pool = self
                .store
                .get_meta(&attempt_cpu_pool_key(active.attempt))?
                .unwrap_or_else(|| self.capacity.cpus.clone());
            running.push((pool, u64::from(active.resources.cpu.maximum_millicores)));
        }
        Ok(crate::resource_policy::cpu_pool_fits(
            &cpus,
            u64::from(job.spec.resources.cpu.maximum_millicores),
            &running,
        ))
    }

    fn rebalance_burst_limits(&self) {
        let burst_reduced = matches!(
            self.overload.stage(),
            OverloadStage::BurstReduced
                | OverloadStage::CpuJobsFrozen
                | OverloadStage::CheckpointRequested
                | OverloadStage::GpuJobsRequeued
                | OverloadStage::HardKillAuthorized
        );
        for active in self.active.values() {
            let Some(cgroup) = &active.cgroup else {
                continue;
            };
            let cpu = if burst_reduced {
                u64::from(
                    active
                        .resources
                        .cpu
                        .desired_millicores
                        .max(active.resources.cpu.minimum_millicores),
                )
            } else {
                u64::from(active.resources.cpu.maximum_millicores)
            };
            let memory_high = if burst_reduced {
                active.resources.ram.soft_bytes
            } else {
                active.resources.ram.hard_bytes
            };
            let weight = match active.priority {
                PriorityClass::System => 10_000,
                PriorityClass::Interactive => 7_500,
                PriorityClass::Batch if burst_reduced => 1_500,
                PriorityClass::Batch => 2_500,
                PriorityClass::Opportunistic => 100,
            };
            let _ = self.cgroups.set_cpu_max(cgroup, cpu);
            let _ = self.cgroups.set_cpu_weight(cgroup, weight);
            let _ = self.cgroups.set_memory_high(cgroup, memory_high);
        }
    }

    pub(super) async fn refresh_gpu_availability(&mut self) {
        let mut managed = BTreeSet::new();
        let mut mps_managed = BTreeSet::new();
        let mut membership_complete = true;
        for active in self.active.values() {
            if let Some(cgroup) = &active.cgroup {
                match self.cgroups.pids(cgroup) {
                    Ok(pids) => {
                        managed.extend(pids.iter().copied());
                        if active.mps_managed {
                            mps_managed.extend(pids);
                        }
                    }
                    Err(error) => {
                        membership_complete = false;
                        tracing::warn!(job_id = %active.attempt.job_id, %error, "cannot verify GPU process cgroup membership");
                    }
                }
            } else {
                managed.insert(active.pid);
            }
        }
        let mut external_mps_client = false;
        let mps_server_pids: BTreeSet<_> = if let Some(controller) = self.mps_controller.clone() {
            match controller.clients().await {
                Ok(clients) => {
                    let mut servers = BTreeSet::new();
                    for client in clients {
                        match crate::gpu::mps_client_matches_pids(&client, &mps_managed) {
                            Ok(true) => {
                                servers.insert(client.server_pid);
                            }
                            Ok(false) => external_mps_client = true,
                            Err(error) => {
                                external_mps_client = true;
                                membership_complete = false;
                                let reason = format!("MPS client ownership is ambiguous: {error}");
                                self.gpu.mps.disable(reason.clone());
                                self.health.disable_mps(reason);
                            }
                        }
                    }
                    if external_mps_client {
                        BTreeSet::new()
                    } else {
                        servers
                    }
                }
                Err(error) => {
                    let reason = format!("MPS process inventory refresh failed: {error}");
                    self.gpu.mps.disable(reason.clone());
                    self.health.disable_mps(reason);
                    BTreeSet::new()
                }
            }
        } else {
            BTreeSet::new()
        };
        let mut inventory =
            GpuInventory::discover(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref());
        inventory.mps = self.gpu.mps.clone();
        for device in &mut inventory.devices {
            let external_compute = device.processes.iter().any(|process| {
                !managed.contains(&process.pid) && !mps_server_pids.contains(&process.pid)
            });
            let unaccounted_usage = device.used_vram_bytes > 0
                && device.processes.is_empty()
                && !crate::gpu::qualified_idle_usage(device, &self.execution.gpu_idle_baselines);
            device.healthy &= membership_complete
                && !external_mps_client
                && !external_compute
                && !unaccounted_usage
                && !self.gpu_global_quarantine
                && !self.gpu_quarantine.contains(&device.uuid)
                && device
                    .parent_uuid
                    .as_ref()
                    .is_none_or(|parent| !self.gpu_quarantine.contains(parent));
        }
        self.gpu = inventory;
        self.health.update_gpu_inventory(&self.gpu);
    }

    pub(super) async fn preempt(&mut self, job_id: JobId, reason: &str) -> Result<(), ActorError> {
        let Some(job) = self.store.get_job(job_id)? else {
            return Ok(());
        };
        if !job.spec.preemption.preemptible
            || !matches!(job.state, JobState::Running | JobState::Paused)
        {
            return Ok(());
        }
        let receipt = self.transition(&job, JobState::Suspended, reason)?;
        self.record_transition(&receipt)?;
        let Some(attempt) = self.active.get(&job_id).map(|active| active.attempt) else {
            return Ok(());
        };
        if let Some(active) = self.active.get_mut(&job_id) {
            active.cancelling = true;
        }
        let (delay, checkpoint_signal) = match job.spec.preemption.checkpoint {
            jain_smartcluster_core::CheckpointPolicy::ControlFd { deadline_ms } => {
                let delay =
                    Duration::from_millis(deadline_ms).min(self.execution.checkpoint_timeout());
                if let Some(active) = self.active.get_mut(&job_id) {
                    if let Err(error) =
                        request_checkpoint(active, delay, self.execution.actor_io_timeout()).await
                    {
                        tracing::warn!(%job_id, %error, "checkpoint request failed before preemption");
                    }
                }
                (delay, None)
            }
            jain_smartcluster_core::CheckpointPolicy::Signal { signal, grace_ms } => (
                Duration::from_millis(grace_ms).min(self.execution.terminate_grace()),
                Signal::try_from(signal).ok(),
            ),
            jain_smartcluster_core::CheckpointPolicy::None => (Duration::ZERO, None),
        };
        if let Some(signal) = checkpoint_signal {
            self.signal_active_safely(job_id, attempt, signal).await?;
        }
        self.queue_termination(job_id, attempt, delay, self.execution.terminate_grace());
        Ok(())
    }

    pub(super) fn scheduling_candidate(
        &self,
        job: &JobRecord,
        history_cache: &mut HashMap<String, Option<GpuLearningSnapshot>>,
    ) -> Result<SchedulingCandidate, ActorError> {
        let fingerprint = self.resolver.fingerprint(&job.spec.workload);
        let learning = if let Some(fingerprint) = fingerprint.as_deref() {
            if let Some(cached) = history_cache.get(fingerprint) {
                *cached
            } else {
                let loaded = self
                    .store
                    .get_resource_history(fingerprint)?
                    .map(|history| GpuLearningSnapshot {
                        clean_exclusive_runs: history.consecutive_clean_exclusive_runs,
                        max_observed_vram_bytes: history.max_observed_vram_bytes,
                        sharing_disabled: history.sharing_disabled,
                        exclusive_retry_pending: history.exclusive_retry_pending,
                    });
                history_cache.insert(fingerprint.to_owned(), loaded);
                loaded
            }
        } else {
            None
        };
        let known_jain = matches!(
            job.spec.workload,
            jain_smartcluster_core::Workload::Jain { .. }
        );
        let mps_checkpoint_safe = !matches!(
            job.spec.preemption.checkpoint,
            jain_smartcluster_core::CheckpointPolicy::Signal { .. }
        );
        Ok(SchedulingCandidate {
            job: job.clone(),
            workload_fingerprint: fingerprint,
            gpu_compatibility: known_jain_gpu_compatibility(
                known_jain,
                self.gpu.mps.shareable && mps_checkpoint_safe,
                self.gpu.mig_inventory,
            ),
            gpu_learning: learning,
        })
    }

    pub(super) async fn launch_job(
        &mut self,
        job: JobRecord,
        allocation: Allocation,
    ) -> Result<(), ActorError> {
        if let Err(reason) = self
            .execution
            .resource_policy
            .validate_scratch_request(&job.spec)
        {
            self.reject_before_attempt(&job, reason)?;
            return Ok(());
        }
        let plan = match self.resolver.resolve(&job.spec.workload) {
            Ok(plan) => plan,
            Err(error) => {
                self.reject_before_attempt(&job, error.to_string())?;
                return Ok(());
            }
        };
        let cpus = self
            .execution
            .resource_policy
            .cpus_for(&job.spec, &self.capacity.cpus)
            .map_err(ActorError::Recovery)?;
        let admitted =
            self.transition(&job, JobState::Admitted, "scheduler admission committed")?;
        self.record_transition(&admitted)?;
        let at = now_timestamp();
        let launch = self.store.begin_attempt_before_launch(
            job.id,
            at,
            Timestamp::from_unix_millis(at.as_unix_millis().saturating_add(ATTEMPT_LEASE_MS)),
        )?;
        self.record_durable_transition(&launch.transition);
        if let Err(error) = self
            .store
            .put_meta(&attempt_allocation_key(launch.attempt.attempt), &allocation)
        {
            self.launch_failed(&launch.job, error.to_string())?;
            return Ok(());
        }
        if let Err(error) = self
            .store
            .put_meta(&attempt_cpu_pool_key(launch.attempt.attempt), &cpus)
        {
            self.launch_failed(&launch.job, error.to_string())?;
            return Ok(());
        }
        let mut daemon_environment = match self.mps_launch_environment(&allocation) {
            Ok(environment) => environment,
            Err(error) => {
                self.launch_failed(&launch.job, error.to_string())?;
                return Ok(());
            }
        };
        if job.spec.resources.scratch_bytes != 0 {
            let pool = self
                .execution
                .resource_policy
                .trusted_scratch
                .as_ref()
                .expect("validated scratch pool");
            daemon_environment.insert(
                "SCQ_SCRATCH_ROOT".to_owned(),
                pool.path.to_string_lossy().into_owned(),
            );
            daemon_environment.insert(
                "SCQ_SCRATCH_RESERVATION_BYTES".to_owned(),
                job.spec.resources.scratch_bytes.to_string(),
            );
        }
        let gpu_ids: Vec<_> = allocation
            .gpu
            .iter()
            .map(|gpu| gpu.device_id.clone())
            .collect();
        let running = match self
            .executor
            .launch(
                launch.attempt.attempt,
                plan,
                ExecutionAllocation {
                    resources: job.spec.resources,
                    priority: job.spec.priority,
                    cpus,
                    io_mode: job.spec.io,
                    gpu_devices: gpu_ids,
                    daemon_environment,
                },
            )
            .await
        {
            Ok(running) => running,
            Err(error) => {
                self.launch_failed(&launch.job, error.to_string())?;
                return Ok(());
            }
        };
        let cgroup_path = running.cgroup.as_ref().map_or_else(
            || format!("single-allocation:pid={}", running.pid),
            |cgroup| cgroup.path.display().to_string(),
        );
        let install = (|| -> Result<(), ActorError> {
            self.store
                .record_spawned(running.attempt, running.pid, cgroup_path)?;
            let start_time = pid_start_time(running.pid).ok_or_else(|| {
                ActorError::Recovery(format!("cannot fingerprint spawned pid {}", running.pid))
            })?;
            self.store
                .put_meta(&attempt_pid_start_key(running.attempt), &start_time)?;
            let starting = self
                .store
                .get_job(job.id)?
                .ok_or(StoreError::JobNotFound(job.id))?;
            let receipt =
                self.transition(&starting, JobState::Running, "worker launch observed")?;
            self.record_transition(&receipt)?;
            Ok(())
        })();
        if let Err(error) = install {
            if let Err(transition_error) = self.launch_failed(&launch.job, error.to_string()) {
                running.leave_for_recovery();
                return Err(transition_error);
            }
            self.dispose_uninstalled_workload(running).await?;
            return Ok(());
        }
        let mut running = running;
        if let Err(error) = running.start() {
            if let Err(transition_error) = self.launch_failed(&launch.job, error.to_string()) {
                running.leave_for_recovery();
                return Err(transition_error);
            }
            self.dispose_uninstalled_workload(running).await?;
            return Ok(());
        }
        let parts = running.into_parts();
        self.install_running(parts, job, allocation);
        Ok(())
    }

    fn reject_before_attempt(
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

    pub(super) fn install_running(
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

    pub(super) fn observe_process_exit(
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

    pub(super) fn observe_output_drained(
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

    pub(super) fn observe_process_group_drained(
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

    pub(super) fn observe_fresh_cgroup_drained(
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

    pub(super) fn finish_process_if_drained(
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

    pub(super) fn launch_failed(
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

    pub(super) fn finish_process(
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

pub(super) fn prelaunch_rejection(
    job: &JobRecord,
    message: String,
    at: Timestamp,
) -> TransitionMutation {
    TransitionMutation {
        transition: FencedTransition {
            job_id: job.id,
            from: job.state,
            to: JobState::Failed,
            attempt: None,
            at,
            reason: Some(message.clone()),
        },
        terminal: Some(TerminalStatus::Failed {
            class: FailureClass::Dependency,
            message,
            exit_code: None,
            at,
        }),
        failure: None,
    }
}

fn completion_ready(
    leader_exited: bool,
    tree_drained: bool,
    output_drains_pending: u8,
    control_drained: bool,
) -> bool {
    leader_exited && tree_drained && output_drains_pending == 0 && control_drained
}

fn known_jain_gpu_compatibility(
    known_jain: bool,
    mps_shareable: bool,
    mig_inventory: bool,
) -> GpuCompatibility {
    GpuCompatibility {
        mps: known_jain && mps_shareable,
        mig: known_jain && mig_inventory,
        // Registration permits learning; the scheduler still requires five clean
        // exclusive runs before selecting shared MPS.
        trusted: known_jain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_registered_jain_templates_can_unlock_gpu_sharing() {
        let known = known_jain_gpu_compatibility(true, true, true);
        assert!(known.trusted);
        assert!(known.mps);
        assert!(known.mig);

        let administrator_argv = known_jain_gpu_compatibility(false, true, true);
        assert!(!administrator_argv.trusted);
        assert!(!administrator_argv.mps);
        assert!(!administrator_argv.mig);
    }

    #[test]
    fn completion_waits_for_process_tree_and_every_output_reader() {
        assert!(!completion_ready(false, true, 0, true));
        assert!(!completion_ready(true, false, 0, true));
        assert!(!completion_ready(true, true, 1, true));
        assert!(!completion_ready(true, true, 0, false));
        assert!(completion_ready(true, true, 0, true));
    }
}
