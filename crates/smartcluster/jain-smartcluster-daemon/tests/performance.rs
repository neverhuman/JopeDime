use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use jain_smartcluster_core::{
    CpuRequest, Environment, IdempotencyKey, JainWorkload, JobId, JobRecord, JobSpec,
    MemoryRequest, Principal, ProjectId, ResourceRequest, TenantId, Timestamp, Workload,
};
use jain_smartcluster_daemon::scheduler::{
    EffectiveCapacity, GpuCompatibility, Scheduler, SchedulingCandidate, SchedulingSnapshot,
};

#[test]
#[ignore = "release-only target-host scheduler acceptance lane"]
fn ten_thousand_job_decision_meets_release_budget() {
    if !cfg!(not(debug_assertions)) {
        eprintln!("SKIP: performance acceptance requires --release");
        return;
    }
    let queued = (0..10_000)
        .map(|index| SchedulingCandidate {
            job: queued_job(index),
            workload_fingerprint: None,
            gpu_compatibility: GpuCompatibility::default(),
            gpu_learning: None,
        })
        .collect();
    let snapshot = SchedulingSnapshot {
        now: Timestamp::from_unix_millis(10_000),
        capacity: EffectiveCapacity {
            cpu_millicores: 64_000,
            memory_bytes: 256 * 1024 * 1024 * 1024,
            scratch_bytes: 1024 * 1024 * 1024 * 1024,
            gpus: Vec::new(),
        },
        queued,
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };
    let scheduler = Scheduler::default();

    const SAMPLE_COUNT: usize = 100;
    const BUDGET: Duration = Duration::from_millis(25);

    // Warm allocator/code paths before recording the acceptance samples.
    scheduler.schedule(&snapshot).expect("warm scheduling pass");
    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for _ in 0..SAMPLE_COUNT {
        let started = Instant::now();
        let plan = scheduler.schedule(&snapshot).expect("scheduling pass");
        assert_eq!(plan.admissions.len(), 1);
        samples.push(started.elapsed());
    }
    samples.sort_unstable();
    let p99 = samples[98];
    println!(
        "JAIN_RELEASE_METRIC {}",
        serde_json::json!({
            "schema_version": "jain.smartcluster.release-metric/v1",
            "metric": "capacity_available_scheduling_p99",
            "observed": u64::try_from(p99.as_nanos()).expect("p99 fits u64 nanoseconds"),
            "limit": u64::try_from(BUDGET.as_nanos()).expect("budget fits u64 nanoseconds"),
            "comparison": "lte",
            "unit": "nanoseconds",
            "sample_count": SAMPLE_COUNT,
            "queue_depth": 10_000,
        })
    );
    assert!(
        p99 <= BUDGET,
        "10k scheduling decision p99 took {p99:?}, budget is {BUDGET:?}"
    );
}

fn queued_job(index: u32) -> JobRecord {
    JobRecord::queued(
        JobId::new(),
        JobSpec {
            tenant: TenantId::new(format!("unix:{}", index % 32 + 1000)).expect("tenant"),
            project: ProjectId::new("performance").expect("project"),
            idempotency_key: IdempotencyKey::new(format!("job-{index}")).expect("key"),
            priority: jain_smartcluster_core::PriorityClass::Batch,
            workload: Workload::Jain {
                template: JainWorkload::Training,
                arguments: Vec::new(),
                environment: Environment::default(),
            },
            resources: ResourceRequest {
                cpu: CpuRequest::fixed(100),
                ram: MemoryRequest::fixed(64 * 1024 * 1024),
                scratch_bytes: 0,
                gpu: None,
            },
            retry: Default::default(),
            preemption: Default::default(),
            io: Default::default(),
            deadline: None,
            labels: BTreeMap::new(),
        },
        Principal::unix(index % 32 + 1000),
        Timestamp::from_unix_millis(u64::from(index)),
    )
}
