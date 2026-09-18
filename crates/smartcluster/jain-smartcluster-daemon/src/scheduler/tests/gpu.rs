use super::*;

#[test]
fn unknown_workload_is_always_exclusive() {
    let request = GpuRequest {
        count: 1,
        vram_bytes: 4 * 1024 * 1024 * 1024,
        compute_share_per_mille: 250,
        mode: GpuMode::Shareable,
    };
    assert_eq!(
        gpu_policy(request, GpuCompatibility::default(), None, false).mode,
        GpuPolicyMode::Exclusive
    );
}

#[test]
fn new_allowlisted_fingerprint_requires_five_clean_exclusive_runs() {
    let request = GpuRequest {
        count: 1,
        vram_bytes: 2 * 1024 * 1024 * 1024,
        compute_share_per_mille: 500,
        mode: GpuMode::Shareable,
    };
    let compatibility = GpuCompatibility {
        mps: true,
        mig: false,
        trusted: true,
    };
    for clean in 0..5 {
        assert_eq!(
            gpu_policy(
                request,
                compatibility,
                Some(GpuLearningSnapshot {
                    clean_exclusive_runs: clean,
                    ..GpuLearningSnapshot::default()
                }),
                false,
            )
            .mode,
            GpuPolicyMode::Exclusive
        );
    }
    assert_eq!(
        gpu_policy(
            request,
            compatibility,
            Some(GpuLearningSnapshot {
                clean_exclusive_runs: 5,
                ..GpuLearningSnapshot::default()
            }),
            false,
        )
        .mode,
        GpuPolicyMode::SharedMps
    );
}

#[test]
fn shared_oom_forces_exclusive_retry() {
    let request = GpuRequest {
        count: 1,
        vram_bytes: 1024,
        compute_share_per_mille: 500,
        mode: GpuMode::Shareable,
    };
    let decision = gpu_policy(
        request,
        GpuCompatibility {
            mps: true,
            mig: false,
            trusted: true,
        },
        Some(GpuLearningSnapshot {
            exclusive_retry_pending: true,
            ..GpuLearningSnapshot::default()
        }),
        false,
    );
    assert_eq!(decision.mode, GpuPolicyMode::Exclusive);
    assert_eq!(decision.compute_share_per_mille, 1_000);
}

#[test]
fn learned_vram_budget_includes_margin() {
    let gib = 1024 * 1024 * 1024;
    let request = GpuRequest {
        count: 1,
        vram_bytes: gib,
        compute_share_per_mille: 500,
        mode: GpuMode::Shareable,
    };
    let decision = gpu_policy(
        request,
        GpuCompatibility {
            mps: true,
            mig: false,
            trusted: true,
        },
        Some(GpuLearningSnapshot {
            clean_exclusive_runs: 5,
            max_observed_vram_bytes: 2 * gib,
            ..GpuLearningSnapshot::default()
        }),
        false,
    );
    assert_eq!(decision.reserved_vram_bytes, 3 * gib);
}

#[test]
fn missing_fingerprint_forces_exclusive_even_if_caller_marks_trusted() {
    let gib = 1024 * 1024 * 1024;
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: gpu_capacity(vec![GpuDevice {
            id: "gpu-0".into(),
            kind: GpuDeviceKind::Physical,
            vram_bytes: 10 * gib,
            healthy: true,
        }]),
        queued: vec![gpu_candidate(
            1,
            GpuRequest {
                count: 1,
                vram_bytes: gib,
                compute_share_per_mille: 500,
                mode: GpuMode::Shareable,
            },
            None,
            GpuCompatibility {
                mps: true,
                mig: true,
                trusted: true,
            },
        )],
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };
    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");
    assert_eq!(
        plan.admissions[0].allocation.gpu[0].mode,
        GpuAllocationMode::Exclusive
    );
}

#[test]
fn ten_percent_vram_headroom_rejects_conflicting_second_share() {
    let gib = 1024 * 1024 * 1024;
    let compatibility = GpuCompatibility {
        mps: true,
        mig: false,
        trusted: true,
    };
    let request = GpuRequest {
        count: 1,
        vram_bytes: 5 * gib,
        compute_share_per_mille: 500,
        mode: GpuMode::Shareable,
    };
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: gpu_capacity(vec![GpuDevice {
            id: "gpu-0".into(),
            kind: GpuDeviceKind::Physical,
            vram_bytes: 10 * gib,
            healthy: true,
        }]),
        queued: vec![
            {
                let mut candidate = gpu_candidate(1, request, Some("train:a"), compatibility);
                candidate.gpu_learning = Some(GpuLearningSnapshot {
                    clean_exclusive_runs: 5,
                    ..GpuLearningSnapshot::default()
                });
                candidate
            },
            {
                let mut candidate = gpu_candidate(2, request, Some("train:b"), compatibility);
                candidate.gpu_learning = Some(GpuLearningSnapshot {
                    clean_exclusive_runs: 5,
                    ..GpuLearningSnapshot::default()
                });
                candidate
            },
        ],
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };
    let scheduler = Scheduler::new(SchedulerConfig {
        max_admissions_per_pass: 2,
        ..SchedulerConfig::default()
    })
    .expect("scheduler");
    let plan = scheduler.schedule(&snapshot).expect("schedule");
    assert_eq!(plan.admissions.len(), 1);
    assert_eq!(plan.blocked.len(), 1);
}

#[test]
fn cpu_ram_headroom_does_not_implicitly_reserve_a_discrete_gpu() {
    let scheduler = Scheduler::new(SchedulerConfig {
        interactive_headroom_per_mille: 100,
        interactive_gpu_headroom_per_mille: 0,
        ..SchedulerConfig::default()
    })
    .expect("scheduler");
    let request = GpuRequest {
        count: 1,
        vram_bytes: 1,
        compute_share_per_mille: 1_000,
        mode: GpuMode::Exclusive,
    };
    let mut candidate = gpu_candidate(1, request, None, GpuCompatibility::default());
    candidate.job.spec.preemption = PreemptionPolicy::non_preemptible();
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: gpu_capacity(vec![GpuDevice {
            id: "gpu-0".to_owned(),
            kind: GpuDeviceKind::Physical,
            vram_bytes: 1024,
            healthy: true,
        }]),
        queued: vec![candidate],
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };

    let plan = scheduler.schedule(&snapshot).expect("schedule");

    assert_eq!(plan.admissions.len(), 1);
    assert!(!plan.admissions[0].allocation.backfill);
}

#[test]
fn exclusive_request_can_use_an_already_isolated_visible_mig_instance() {
    let request = GpuRequest {
        count: 1,
        vram_bytes: 512,
        compute_share_per_mille: 1_000,
        mode: GpuMode::Exclusive,
    };
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: gpu_capacity(vec![GpuDevice {
            id: "MIG-instance".to_owned(),
            kind: GpuDeviceKind::MigInstance,
            vram_bytes: 1024,
            healthy: true,
        }]),
        queued: vec![gpu_candidate(1, request, None, GpuCompatibility::default())],
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };

    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");

    assert_eq!(plan.admissions.len(), 1);
    assert_eq!(
        plan.admissions[0].allocation.gpu[0].mode,
        GpuAllocationMode::MigInstance
    );
}

#[test]
fn one_request_never_counts_the_same_gpu_twice() {
    let gib = 1024 * 1024 * 1024;
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: gpu_capacity(vec![GpuDevice {
            id: "gpu-0".into(),
            kind: GpuDeviceKind::Physical,
            vram_bytes: 10 * gib,
            healthy: true,
        }]),
        queued: vec![{
            let mut candidate = gpu_candidate(
                1,
                GpuRequest {
                    count: 2,
                    vram_bytes: gib,
                    compute_share_per_mille: 250,
                    mode: GpuMode::Shareable,
                },
                Some("train:two-gpu"),
                GpuCompatibility {
                    mps: true,
                    mig: false,
                    trusted: true,
                },
            );
            candidate.gpu_learning = Some(GpuLearningSnapshot {
                clean_exclusive_runs: 5,
                ..GpuLearningSnapshot::default()
            });
            candidate
        }],
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };
    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");
    assert!(plan.admissions.is_empty());
    assert_eq!(plan.blocked.len(), 1);
}

#[test]
fn mig_preference_uses_an_existing_instance_without_sharing() {
    let gib = 1024 * 1024 * 1024;
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: gpu_capacity(vec![GpuDevice {
            id: "mig-0".into(),
            kind: GpuDeviceKind::MigInstance,
            vram_bytes: 10 * gib,
            healthy: true,
        }]),
        queued: vec![gpu_candidate(
            1,
            GpuRequest {
                count: 1,
                vram_bytes: 4 * gib,
                compute_share_per_mille: 500,
                mode: GpuMode::MigPreferred,
            },
            Some("train:mig"),
            GpuCompatibility {
                mps: false,
                mig: true,
                trusted: true,
            },
        )],
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };
    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");
    assert_eq!(
        plan.admissions[0].allocation.gpu[0].mode,
        GpuAllocationMode::MigInstance
    );
}

#[test]
fn exclusive_max_vram_sentinel_reserves_the_selected_device_total() {
    let gib = 1024 * 1024 * 1024;
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: gpu_capacity(vec![GpuDevice {
            id: "gpu-0".into(),
            kind: GpuDeviceKind::Physical,
            vram_bytes: 16 * gib,
            healthy: true,
        }]),
        queued: vec![gpu_candidate(
            1,
            GpuRequest {
                count: 1,
                vram_bytes: u64::MAX,
                compute_share_per_mille: 1_000,
                mode: GpuMode::Exclusive,
            },
            Some("admin:exclusive"),
            GpuCompatibility::default(),
        )],
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };
    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");
    assert_eq!(plan.admissions.len(), 1);
    assert_eq!(
        plan.admissions[0].allocation.gpu[0].reserved_vram_bytes,
        16 * gib
    );
}
