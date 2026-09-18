use std::collections::BTreeMap;

use jain_smartcluster_core::{
    CheckpointPolicy, CpuRequest, Environment, GpuMode, GpuRequest, IdempotencyKey, JainWorkload,
    JobId, MemoryRequest, PreemptionPolicy, Principal, PriorityClass, ProjectId, ResourceRequest,
    RetryPolicy, TenantId, Workload,
};
use ulid::Ulid;

use super::*;

mod fairness;
mod gpu;

fn job_id(value: u128) -> JobId {
    JobId::from_ulid(Ulid::from(value))
}

fn candidate(
    id: u128,
    tenant: &str,
    priority: PriorityClass,
    cpu_millicores: u32,
    preemptible: bool,
    created_at: u64,
) -> SchedulingCandidate {
    let tenant = TenantId::new(tenant).expect("tenant");
    let spec = jain_smartcluster_core::JobSpec {
        tenant,
        project: ProjectId::new("project").expect("project"),
        idempotency_key: IdempotencyKey::new(format!("request-{id}")).expect("key"),
        priority,
        workload: Workload::Jain {
            template: JainWorkload::Training,
            arguments: Vec::new(),
            environment: Environment::default(),
        },
        resources: ResourceRequest {
            cpu: CpuRequest::fixed(cpu_millicores),
            ram: MemoryRequest::fixed(1),
            scratch_bytes: 0,
            gpu: None,
        },
        retry: RetryPolicy::default(),
        preemption: PreemptionPolicy {
            preemptible,
            checkpoint: if preemptible {
                CheckpointPolicy::ControlFd { deadline_ms: 1_000 }
            } else {
                CheckpointPolicy::None
            },
        },
        io: jain_smartcluster_core::IoMode::Pipes,
        deadline: None,
        labels: BTreeMap::new(),
    };
    SchedulingCandidate {
        job: jain_smartcluster_core::JobRecord::queued(
            job_id(id),
            spec,
            Principal::unix(1_000),
            created_at.into(),
        ),
        workload_fingerprint: None,
        gpu_compatibility: GpuCompatibility::default(),
        gpu_learning: None,
    }
}

fn capacity(cpu_millicores: u64) -> EffectiveCapacity {
    EffectiveCapacity {
        cpu_millicores,
        memory_bytes: 1_000_000,
        scratch_bytes: 1_000_000,
        gpus: Vec::new(),
    }
}

fn running(
    id: u128,
    tenant: &str,
    priority: PriorityClass,
    cpu_millicores: u32,
    backfill: bool,
) -> RunningAllocation {
    RunningAllocation {
        job_id: job_id(id),
        tenant: TenantId::new(tenant).expect("tenant"),
        priority,
        allocation: Allocation {
            resources: ResourceRequest {
                cpu: CpuRequest::fixed(cpu_millicores),
                ram: MemoryRequest::fixed(1),
                scratch_bytes: 0,
                gpu: None,
            },
            gpu: Vec::new(),
            backfill,
        },
        preemptible: backfill,
        started_at: 1.into(),
    }
}

fn gpu_candidate(
    id: u128,
    request: GpuRequest,
    fingerprint: Option<&str>,
    compatibility: GpuCompatibility,
) -> SchedulingCandidate {
    let mut candidate = candidate(id, "tenant-a", PriorityClass::Batch, 100, true, 1);
    candidate.job.spec.resources.gpu = Some(request);
    candidate.workload_fingerprint = fingerprint.map(str::to_owned);
    candidate.gpu_compatibility = compatibility;
    candidate
}

fn gpu_capacity(devices: Vec<GpuDevice>) -> EffectiveCapacity {
    EffectiveCapacity {
        cpu_millicores: 10_000,
        memory_bytes: 1_000_000,
        scratch_bytes: 1_000_000,
        gpus: devices,
    }
}

fn ten_thousand_job_snapshot() -> SchedulingSnapshot {
    let queued = (0_u128..10_000)
        .map(|index| {
            candidate(
                index + 1,
                &format!("tenant-{}", index % 100),
                PriorityClass::Batch,
                1,
                true,
                1,
            )
        })
        .collect();
    SchedulingSnapshot {
        now: 100.into(),
        capacity: capacity(10_000),
        queued,
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    }
}

#[test]
fn scratch_reservations_count_recovered_allocations_and_external_free_space() {
    let scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
    let mut queued = candidate(2, "tenant", PriorityClass::Batch, 100, false, 1);
    queued.job.spec.resources.scratch_bytes = 6;
    let mut recovered = running(1, "tenant", PriorityClass::Batch, 100, false);
    recovered.allocation.resources.scratch_bytes = 10;
    let mut snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: capacity(10000),
        queued: vec![queued],
        running: vec![recovered],
        tenant_weights: BTreeMap::new(),
    };
    snapshot.capacity.scratch_bytes = 15;
    assert!(scheduler.schedule(&snapshot).unwrap().admissions.is_empty());
    snapshot.capacity.scratch_bytes = 16;
    assert_eq!(scheduler.schedule(&snapshot).unwrap().admissions.len(), 1);
    snapshot.capacity.scratch_bytes = 9;
    assert!(scheduler.schedule(&snapshot).unwrap().admissions.is_empty());
    snapshot.running.clear();
    assert_eq!(scheduler.schedule(&snapshot).unwrap().admissions.len(), 1);
}

#[test]
fn full_fixed_pool_preempts_lower_priority_even_with_global_free_cpu() {
    let scheduler = Scheduler::default();
    let queued = candidate(2, "tenant", PriorityClass::Interactive, 1000, false, 1);
    let mut occupant = running(1, "tenant", PriorityClass::Batch, 2000, false);
    occupant.preemptible = true;
    let mut snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: capacity(10000),
        queued: vec![queued],
        running: vec![occupant],
        tenant_weights: BTreeMap::new(),
    };
    let pool = crate::capacity::CpuSet::parse("8-9").unwrap();
    let pools = BTreeMap::from([(job_id(1), pool.clone()), (job_id(2), pool)]);
    let plan = scheduler
        .schedule_with_cpu_pools(&snapshot, &pools)
        .unwrap();
    assert!(plan.admissions.is_empty());
    assert_eq!(plan.preemptions.len(), 1);
    assert_eq!(plan.preemptions[0].job_id, job_id(1));
    assert_eq!(plan.preemptions[0].for_job, job_id(2));
    snapshot.running[0].preemptible = false;
    assert!(scheduler
        .schedule_with_cpu_pools(&snapshot, &pools)
        .unwrap()
        .preemptions
        .is_empty());
    snapshot.running.clear();
    assert_eq!(
        scheduler
            .schedule_with_cpu_pools(&snapshot, &pools)
            .unwrap()
            .admissions
            .len(),
        1
    );
}

#[test]
fn fixed_pool_counts_admissions_planned_in_the_same_pass() {
    let scheduler = Scheduler::new(SchedulerConfig {
        max_admissions_per_pass: 8,
        ..Default::default()
    })
    .unwrap();
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: capacity(10000),
        queued: vec![
            candidate(1, "tenant", PriorityClass::Interactive, 2000, false, 1),
            candidate(2, "tenant", PriorityClass::Interactive, 1000, false, 1),
        ],
        running: vec![],
        tenant_weights: BTreeMap::new(),
    };
    let pool = crate::capacity::CpuSet::parse("8-9").unwrap();
    let pools = BTreeMap::from([(job_id(1), pool.clone()), (job_id(2), pool)]);
    let plan = scheduler
        .schedule_with_cpu_pools(&snapshot, &pools)
        .unwrap();
    assert_eq!(plan.admissions.len(), 1);
    assert_eq!(plan.blocked.len(), 1);
}
