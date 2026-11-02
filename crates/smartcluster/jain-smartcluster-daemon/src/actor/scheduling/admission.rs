use super::*;

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
}
