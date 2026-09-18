use std::collections::BTreeMap;

use jain_smartcluster_core::{
    JobId, JobRecord, PriorityClass, ResourceRequest, TenantId, Timestamp,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub(super) const PER_MILLE: u64 = 1_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceVector {
    pub cpu_millicores: u64,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub gpu_compute_per_mille: u64,
    pub gpu_vram_bytes: u64,
}

impl ResourceVector {
    pub(super) fn saturating_add(self, other: Self) -> Self {
        Self {
            cpu_millicores: self.cpu_millicores.saturating_add(other.cpu_millicores),
            memory_bytes: self.memory_bytes.saturating_add(other.memory_bytes),
            scratch_bytes: self.scratch_bytes.saturating_add(other.scratch_bytes),
            gpu_compute_per_mille: self
                .gpu_compute_per_mille
                .saturating_add(other.gpu_compute_per_mille),
            gpu_vram_bytes: self.gpu_vram_bytes.saturating_add(other.gpu_vram_bytes),
        }
    }

    pub(super) fn saturating_sub(self, other: Self) -> Self {
        Self {
            cpu_millicores: self.cpu_millicores.saturating_sub(other.cpu_millicores),
            memory_bytes: self.memory_bytes.saturating_sub(other.memory_bytes),
            scratch_bytes: self.scratch_bytes.saturating_sub(other.scratch_bytes),
            gpu_compute_per_mille: self
                .gpu_compute_per_mille
                .saturating_sub(other.gpu_compute_per_mille),
            gpu_vram_bytes: self.gpu_vram_bytes.saturating_sub(other.gpu_vram_bytes),
        }
    }

    pub(super) fn fits_within(self, capacity: Self) -> bool {
        self.cpu_millicores <= capacity.cpu_millicores
            && self.memory_bytes <= capacity.memory_bytes
            && self.scratch_bytes <= capacity.scratch_bytes
            && self.gpu_compute_per_mille <= capacity.gpu_compute_per_mille
            && self.gpu_vram_bytes <= capacity.gpu_vram_bytes
    }

    pub(super) fn scaled_headroom(self, cpu_memory_per_mille: u16, gpu_per_mille: u16) -> Self {
        let scale = |value: u64, per_mille: u16| {
            u64::try_from(u128::from(value) * u128::from(per_mille) / u128::from(PER_MILLE))
                .unwrap_or(u64::MAX)
        };
        Self {
            cpu_millicores: scale(self.cpu_millicores, cpu_memory_per_mille),
            memory_bytes: scale(self.memory_bytes, cpu_memory_per_mille),
            scratch_bytes: 0,
            gpu_compute_per_mille: scale(self.gpu_compute_per_mille, gpu_per_mille),
            gpu_vram_bytes: scale(self.gpu_vram_bytes, gpu_per_mille),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GpuDeviceKind {
    Physical,
    MigInstance,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuDevice {
    pub id: String,
    pub kind: GpuDeviceKind,
    pub vram_bytes: u64,
    pub healthy: bool,
}

/// Capacity already restricted by the daemon's effective cgroup and visible
/// device set. Host totals must not be passed here.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EffectiveCapacity {
    pub cpu_millicores: u64,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub gpus: Vec<GpuDevice>,
}

impl EffectiveCapacity {
    pub fn resource_vector(&self) -> ResourceVector {
        let healthy = self.gpus.iter().filter(|gpu| gpu.healthy);
        ResourceVector {
            cpu_millicores: self.cpu_millicores,
            memory_bytes: self.memory_bytes,
            scratch_bytes: self.scratch_bytes,
            gpu_compute_per_mille: healthy.clone().count() as u64 * PER_MILLE,
            gpu_vram_bytes: healthy.map(|gpu| gpu.vram_bytes).sum(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuCompatibility {
    /// MPS limits are supported by this workload.
    pub mps: bool,
    /// The workload supports an existing MIG instance.
    pub mig: bool,
    /// Only allowlisted Jain code may ever become shareable.
    pub trusted: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuLearningSnapshot {
    pub clean_exclusive_runs: u32,
    pub max_observed_vram_bytes: u64,
    pub sharing_disabled: bool,
    pub exclusive_retry_pending: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SchedulingCandidate {
    pub job: JobRecord,
    pub workload_fingerprint: Option<String>,
    pub gpu_compatibility: GpuCompatibility,
    pub gpu_learning: Option<GpuLearningSnapshot>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GpuAllocationMode {
    Exclusive,
    SharedMps,
    MigInstance,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuAllocation {
    pub device_id: String,
    pub mode: GpuAllocationMode,
    pub reserved_vram_bytes: u64,
    pub compute_share_per_mille: u16,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Allocation {
    pub resources: ResourceRequest,
    pub gpu: Vec<GpuAllocation>,
    /// True when a lower-class preemptible job is consuming interactive reserve.
    pub backfill: bool,
}

impl Allocation {
    pub fn admitted_resources(&self) -> ResourceVector {
        ResourceVector {
            cpu_millicores: u64::from(self.resources.cpu.minimum_millicores),
            memory_bytes: self.resources.ram.soft_bytes,
            scratch_bytes: self.resources.scratch_bytes,
            gpu_compute_per_mille: self
                .gpu
                .iter()
                .map(|gpu| u64::from(gpu.compute_share_per_mille))
                .sum(),
            gpu_vram_bytes: self.gpu.iter().map(|gpu| gpu.reserved_vram_bytes).sum(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RunningAllocation {
    pub job_id: JobId,
    pub tenant: TenantId,
    pub priority: PriorityClass,
    pub allocation: Allocation,
    pub preemptible: bool,
    pub started_at: Timestamp,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Admission {
    pub job_id: JobId,
    pub allocation: Allocation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Preemption {
    pub job_id: JobId,
    pub for_job: JobId,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockedJob {
    pub job_id: JobId,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SchedulingPlan {
    pub admissions: Vec<Admission>,
    pub preemptions: Vec<Preemption>,
    pub blocked: Vec<BlockedJob>,
}

#[derive(Clone, Debug)]
pub struct SchedulingSnapshot {
    pub now: Timestamp,
    pub capacity: EffectiveCapacity,
    pub queued: Vec<SchedulingCandidate>,
    pub running: Vec<RunningAllocation>,
    /// Relative tenant entitlement. Missing/zero entries use weight 1,000.
    pub tenant_weights: BTreeMap<TenantId, u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerConfig {
    pub interactive_headroom_per_mille: u16,
    pub interactive_gpu_headroom_per_mille: u16,
    pub shared_vram_headroom_per_mille: u16,
    pub aging_interval_ms: u64,
    pub aging_credit_micros: u64,
    /// Actor passes normally admit one job, then re-evaluate fresh capacity.
    pub max_admissions_per_pass: usize,
    /// Hard process-count guard independent of resource minima.
    pub max_running_jobs: usize,
    /// Per-tenant process-count guard independent of weighted resource share.
    pub max_running_jobs_per_tenant: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            interactive_headroom_per_mille: 100,
            interactive_gpu_headroom_per_mille: 0,
            shared_vram_headroom_per_mille: 100,
            aging_interval_ms: 30_000,
            aging_credit_micros: 5_000,
            max_admissions_per_pass: 1,
            max_running_jobs: 1_024,
            max_running_jobs_per_tenant: 128,
        }
    }
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("invalid scheduler configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("running allocation references unknown GPU {0}")]
    UnknownGpu(String),
    #[error("GPU {0} has conflicting exclusive/shared allocations")]
    ConflictingGpuAllocation(String),
    #[error("GPU {0} is overcommitted in the scheduling snapshot")]
    OvercommittedGpu(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuPolicyMode {
    Exclusive,
    SharedMps,
    MigInstance,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuPolicyDecision {
    pub mode: GpuPolicyMode,
    pub reserved_vram_bytes: u64,
    pub compute_share_per_mille: u16,
    pub reason: &'static str,
}
