//! Deterministic single-node admission and GPU sharing policy.

mod fairness;
mod gpu_policy;
mod model;

use std::collections::{BTreeMap, BTreeSet};

use crate::capacity::CpuSet;
use jain_smartcluster_core::{JobId, PriorityClass, TenantId};

use fairness::compare_candidates;
use gpu_policy::DeviceStates;

pub use gpu_policy::gpu_policy;
pub use model::{
    Admission, Allocation, BlockedJob, EffectiveCapacity, GpuAllocation, GpuAllocationMode,
    GpuCompatibility, GpuDevice, GpuDeviceKind, GpuLearningSnapshot, GpuPolicyDecision,
    GpuPolicyMode, Preemption, ResourceVector, RunningAllocation, SchedulerConfig, SchedulerError,
    SchedulingCandidate, SchedulingPlan, SchedulingSnapshot,
};

pub struct Scheduler {
    config: SchedulerConfig,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        if config.interactive_headroom_per_mille > 1_000
            || config.interactive_gpu_headroom_per_mille > 1_000
            || config.shared_vram_headroom_per_mille >= 1_000
            || config.aging_interval_ms == 0
            || config.max_admissions_per_pass == 0
            || config.max_running_jobs == 0
            || config.max_running_jobs_per_tenant == 0
            || config.max_running_jobs_per_tenant > config.max_running_jobs
        {
            return Err(SchedulerError::InvalidConfig(
                "per-mille values and aging interval are out of range",
            ));
        }
        Ok(Self { config })
    }

    pub fn schedule(
        &self,
        snapshot: &SchedulingSnapshot,
    ) -> Result<SchedulingPlan, SchedulerError> {
        self.schedule_with_cpu_pools(snapshot, &BTreeMap::new())
    }

    pub fn schedule_with_cpu_pools(
        &self,
        snapshot: &SchedulingSnapshot,
        cpu_pools: &BTreeMap<JobId, CpuSet>,
    ) -> Result<SchedulingPlan, SchedulerError> {
        if !cpu_pools.is_empty()
            && (cpu_pools.values().any(CpuSet::is_empty)
                || snapshot
                    .running
                    .iter()
                    .map(|active| active.job_id)
                    .chain(snapshot.queued.iter().map(|candidate| candidate.job.id))
                    .any(|id| !cpu_pools.contains_key(&id)))
        {
            return Err(SchedulerError::InvalidConfig(
                "CPU pool snapshot must cover every queued and running job with a nonempty set",
            ));
        }
        let capacity = snapshot.capacity.resource_vector();
        let reserve = capacity.scaled_headroom(
            self.config.interactive_headroom_per_mille,
            self.config.interactive_gpu_headroom_per_mille,
        );
        let regular_capacity = capacity.saturating_sub(reserve);
        let mut usage = ResourceVector::default();
        let mut tenant_usage: BTreeMap<TenantId, ResourceVector> = BTreeMap::new();
        let mut running_jobs = snapshot.running.len();
        let mut tenant_jobs: BTreeMap<TenantId, usize> = BTreeMap::new();
        let mut devices = DeviceStates::new(&snapshot.capacity.gpus);
        for running in &snapshot.running {
            let resources = running.allocation.admitted_resources();
            usage = usage.saturating_add(resources);
            let tenant = tenant_usage.entry(running.tenant.clone()).or_default();
            *tenant = tenant.saturating_add(resources);
            *tenant_jobs.entry(running.tenant.clone()).or_default() += 1;
            devices.add_running(&running.allocation.gpu)?;
        }

        let mut plan = SchedulingPlan::default();
        let mut remaining: Vec<_> = snapshot
            .queued
            .iter()
            .filter(|candidate| candidate.job.state == jain_smartcluster_core::JobState::Queued)
            .collect();
        let mut seen = BTreeSet::new();
        remaining.retain(|candidate| seen.insert(candidate.job.id));

        let mut order_is_current = false;
        while !remaining.is_empty() {
            if !order_is_current {
                // Reverse desired order so the deterministic minimum is popped in O(1).
                remaining.sort_by(|left, right| {
                    compare_candidates(
                        right,
                        left,
                        &tenant_usage,
                        capacity,
                        &snapshot.tenant_weights,
                        snapshot.now,
                        self.config,
                    )
                });
                order_is_current = true;
            }
            let candidate = remaining.pop().expect("remaining candidates are nonempty");
            if candidate
                .job
                .spec
                .deadline
                .is_some_and(|deadline| deadline <= snapshot.now)
            {
                plan.blocked.push(BlockedJob {
                    job_id: candidate.job.id,
                    reason: "deadline expired before admission".into(),
                });
                continue;
            }
            if running_jobs >= self.config.max_running_jobs {
                plan.blocked.push(BlockedJob {
                    job_id: candidate.job.id,
                    reason: "global live-allocation limit is reached".into(),
                });
                continue;
            }
            if tenant_jobs
                .get(&candidate.job.spec.tenant)
                .copied()
                .unwrap_or_default()
                >= self.config.max_running_jobs_per_tenant
            {
                plan.blocked.push(BlockedJob {
                    job_id: candidate.job.id,
                    reason: "tenant live-allocation limit is reached".into(),
                });
                continue;
            }

            if let Some(pool) = cpu_pools.get(&candidate.job.id) {
                if u64::from(candidate.job.spec.resources.cpu.maximum_millicores)
                    > (pool.len() as u64).saturating_mul(1000)
                {
                    plan.blocked.push(BlockedJob {
                        job_id: candidate.job.id,
                        reason: "CPU request exceeds fixed pool capacity".into(),
                    });
                    continue;
                }

                let running_pools: Vec<_> = snapshot
                    .running
                    .iter()
                    .map(|active| (active.job_id, &active.allocation))
                    .chain(
                        plan.admissions
                            .iter()
                            .map(|admission| (admission.job_id, &admission.allocation)),
                    )
                    .filter_map(|(id, allocation)| {
                        cpu_pools.get(&id).map(|pool| {
                            (
                                pool.clone(),
                                u64::from(allocation.resources.cpu.maximum_millicores),
                            )
                        })
                    })
                    .collect();
                if !crate::resource_policy::cpu_pool_fits(
                    pool,
                    u64::from(candidate.job.spec.resources.cpu.maximum_millicores),
                    &running_pools,
                ) {
                    let mut victims: Vec<_> = snapshot
                        .running
                        .iter()
                        .filter(|active| {
                            active.preemptible
                                && active.priority.rank() > candidate.job.spec.priority.rank()
                                && cpu_pools
                                    .get(&active.job_id)
                                    .is_some_and(|other| !pool.intersect(other).is_empty())
                        })
                        .collect();
                    victims.sort_by(|left, right| {
                        right
                            .priority
                            .rank()
                            .cmp(&left.priority.rank())
                            .then(right.started_at.cmp(&left.started_at))
                            .then(left.job_id.cmp(&right.job_id))
                    });
                    plan.blocked.push(BlockedJob {
                        job_id: candidate.job.id,
                        reason: "fixed CPU pool capacity is currently exhausted".into(),
                    });
                    if matches!(
                        candidate.job.spec.priority,
                        PriorityClass::System | PriorityClass::Interactive
                    ) {
                        if let Some(victim) = victims.first() {
                            plan.preemptions.push(Preemption { job_id: victim.job_id, for_job: candidate.job.id,
                                reason: "reclaim fixed CPU pool from lower-priority preemptible allocation".into() });
                            break;
                        }
                    }
                    continue;
                }
            }

            let mut trial_devices = devices.clone();
            let compatibility = if candidate.workload_fingerprint.is_some() {
                candidate.gpu_compatibility
            } else {
                GpuCompatibility::default()
            };
            let gpu = match trial_devices.allocate(
                candidate.job.spec.resources.gpu,
                compatibility,
                candidate.gpu_learning,
                self.config.shared_vram_headroom_per_mille,
            ) {
                Ok(gpu) => gpu,
                Err(reason) => {
                    let should_preempt =
                        self.recommend_preemption(candidate, &snapshot.running, &mut plan);
                    plan.blocked.push(BlockedJob {
                        job_id: candidate.job.id,
                        reason,
                    });
                    if should_preempt {
                        break;
                    }
                    continue;
                }
            };
            let mut allocation = Allocation {
                resources: candidate.job.spec.resources,
                gpu,
                backfill: false,
            };
            let needed = allocation.admitted_resources();
            if !usage.saturating_add(needed).fits_within(capacity) {
                let should_preempt =
                    self.recommend_preemption(candidate, &snapshot.running, &mut plan);
                plan.blocked.push(BlockedJob {
                    job_id: candidate.job.id,
                    reason: "effective node capacity is currently exhausted".into(),
                });
                if should_preempt {
                    break;
                }
                continue;
            }

            if matches!(
                candidate.job.spec.priority,
                PriorityClass::Batch | PriorityClass::Opportunistic
            ) && !usage.saturating_add(needed).fits_within(regular_capacity)
            {
                if !candidate.job.spec.preemption.preemptible {
                    plan.blocked.push(BlockedJob {
                        job_id: candidate.job.id,
                        reason: "interactive headroom is reserved; job is not preemptible".into(),
                    });
                    continue;
                }
                allocation.backfill = true;
            }

            devices = trial_devices;
            usage = usage.saturating_add(needed);
            let tenant = tenant_usage
                .entry(candidate.job.spec.tenant.clone())
                .or_default();
            *tenant = tenant.saturating_add(needed);
            running_jobs += 1;
            *tenant_jobs
                .entry(candidate.job.spec.tenant.clone())
                .or_default() += 1;
            plan.admissions.push(Admission {
                job_id: candidate.job.id,
                allocation,
            });
            if plan.admissions.len() >= self.config.max_admissions_per_pass {
                break;
            }
            order_is_current = false;
        }
        Ok(plan)
    }

    fn recommend_preemption(
        &self,
        candidate: &SchedulingCandidate,
        running: &[RunningAllocation],
        plan: &mut SchedulingPlan,
    ) -> bool {
        if !matches!(
            candidate.job.spec.priority,
            PriorityClass::System | PriorityClass::Interactive
        ) {
            return false;
        }
        let mut reclaimable: Vec<_> = running
            .iter()
            .filter(|allocation| allocation.preemptible && allocation.allocation.backfill)
            .collect();
        reclaimable.sort_by(|left, right| {
            right
                .priority
                .rank()
                .cmp(&left.priority.rank())
                .then(right.started_at.cmp(&left.started_at))
                .then(left.job_id.cmp(&right.job_id))
        });
        if let Some(victim) = reclaimable.first() {
            if !plan
                .preemptions
                .iter()
                .any(|preemption| preemption.job_id == victim.job_id)
            {
                plan.preemptions.push(Preemption {
                    job_id: victim.job_id,
                    for_job: candidate.job.id,
                    reason: "reclaim interactive headroom from preemptible backfill".into(),
                });
            }
            true
        } else {
            false
        }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(SchedulerConfig::default()).expect("default scheduler config is valid")
    }
}

#[cfg(test)]
mod tests;
