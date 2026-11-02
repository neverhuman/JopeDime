use super::*;

impl DaemonActor {
    pub(crate) async fn refresh_gpu_availability(&mut self) {
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

    pub(crate) async fn preempt(&mut self, job_id: JobId, reason: &str) -> Result<(), ActorError> {
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

    pub(crate) fn scheduling_candidate(
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

    pub(crate) async fn launch_job(
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
}
