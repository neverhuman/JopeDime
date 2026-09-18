use proptest::prelude::*;

use super::*;

#[test]
fn priority_classes_are_strict_before_fairness() {
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: capacity(10_000),
        queued: vec![
            candidate(1, "tenant-a", PriorityClass::Batch, 1_000, true, 1),
            candidate(2, "tenant-b", PriorityClass::Interactive, 1_000, false, 99),
        ],
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };
    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");
    assert_eq!(plan.admissions[0].job_id, job_id(2));
}

#[test]
fn dominant_resource_fairness_selects_the_least_served_tenant() {
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: capacity(10_000),
        queued: vec![
            candidate(1, "tenant-a", PriorityClass::Batch, 1_000, true, 90),
            candidate(2, "tenant-b", PriorityClass::Batch, 1_000, true, 90),
        ],
        running: vec![running(10, "tenant-a", PriorityClass::Batch, 5_000, false)],
        tenant_weights: BTreeMap::new(),
    };
    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");
    assert_eq!(plan.admissions[0].job_id, job_id(2));
}

#[test]
fn tenant_weights_scale_dominant_share() {
    let tenant_a = TenantId::new("tenant-a").expect("tenant");
    let tenant_b = TenantId::new("tenant-b").expect("tenant");
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: capacity(10_000),
        queued: vec![
            candidate(1, "tenant-a", PriorityClass::Batch, 500, true, 90),
            candidate(2, "tenant-b", PriorityClass::Batch, 500, true, 90),
        ],
        running: vec![
            running(10, "tenant-a", PriorityClass::Batch, 4_000, false),
            running(11, "tenant-b", PriorityClass::Batch, 3_000, false),
        ],
        tenant_weights: BTreeMap::from([(tenant_a, 2_000), (tenant_b, 1_000)]),
    };
    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");
    assert_eq!(plan.admissions[0].job_id, job_id(1));
}

#[test]
fn aging_can_break_a_within_class_fairness_tie() {
    let scheduler = Scheduler::new(SchedulerConfig {
        aging_interval_ms: 1_000,
        aging_credit_micros: 100_000,
        ..SchedulerConfig::default()
    })
    .expect("scheduler");
    let snapshot = SchedulingSnapshot {
        now: 10_000.into(),
        capacity: capacity(10_000),
        queued: vec![
            candidate(1, "tenant-a", PriorityClass::Batch, 500, true, 0),
            candidate(2, "tenant-b", PriorityClass::Batch, 500, true, 9_999),
        ],
        running: vec![running(10, "tenant-a", PriorityClass::Batch, 5_000, false)],
        tenant_weights: BTreeMap::new(),
    };
    let plan = scheduler.schedule(&snapshot).expect("schedule");
    assert_eq!(plan.admissions[0].job_id, job_id(1));
}

#[test]
fn only_preemptible_work_can_backfill_interactive_headroom() {
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: capacity(10_000),
        queued: vec![
            candidate(1, "tenant-a", PriorityClass::Batch, 9_500, false, 1),
            candidate(2, "tenant-b", PriorityClass::Batch, 9_500, true, 2),
        ],
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };
    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");
    assert!(plan
        .blocked
        .iter()
        .any(|blocked| blocked.job_id == job_id(1)));
    assert_eq!(plan.admissions.len(), 1);
    assert_eq!(plan.admissions[0].job_id, job_id(2));
    assert!(plan.admissions[0].allocation.backfill);
}

#[test]
fn interactive_pressure_recommends_backfill_reclamation() {
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: capacity(10_000),
        queued: vec![candidate(
            2,
            "tenant-b",
            PriorityClass::Interactive,
            1_000,
            false,
            2,
        )],
        running: vec![running(1, "tenant-a", PriorityClass::Batch, 9_500, true)],
        tenant_weights: BTreeMap::new(),
    };
    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");
    assert_eq!(plan.preemptions.len(), 1);
    assert_eq!(plan.preemptions[0].job_id, job_id(1));
    assert_eq!(plan.preemptions[0].for_job, job_id(2));
}

#[test]
fn ten_thousand_job_queue_produces_one_stable_capacity_decision() {
    let snapshot = ten_thousand_job_snapshot();
    let plan = Scheduler::default().schedule(&snapshot).expect("schedule");
    assert_eq!(plan.admissions.len(), 1);
    assert_eq!(plan.admissions[0].job_id, job_id(1));
    assert!(plan.preemptions.is_empty());
    assert!(plan.blocked.is_empty());
}

#[test]
fn live_allocation_caps_bound_near_zero_resource_jobs() {
    let scheduler = Scheduler::new(SchedulerConfig {
        max_admissions_per_pass: 16,
        max_running_jobs: 3,
        max_running_jobs_per_tenant: 2,
        ..SchedulerConfig::default()
    })
    .expect("scheduler");
    let mut queued: Vec<_> = (1..=5)
        .map(|id| candidate(id, "tenant-a", PriorityClass::Batch, 0, true, 1))
        .collect();
    for candidate in &mut queued {
        candidate.job.spec.resources.ram = MemoryRequest {
            soft_bytes: 0,
            hard_bytes: 1,
        };
    }
    let snapshot = SchedulingSnapshot {
        now: 100.into(),
        capacity: capacity(10_000),
        queued,
        running: Vec::new(),
        tenant_weights: BTreeMap::new(),
    };

    let plan = scheduler.schedule(&snapshot).expect("schedule");

    assert_eq!(plan.admissions.len(), 2);
    assert_eq!(plan.blocked.len(), 3);
    assert!(plan
        .blocked
        .iter()
        .all(|job| job.reason.contains("tenant live-allocation")));
}

#[test]
#[cfg(not(debug_assertions))]
#[ignore = "release-only 10,000-job scheduling acceptance budget"]
fn release_ten_thousand_job_decision_p99_is_at_most_25_ms() {
    use std::time::{Duration, Instant};

    let snapshot = ten_thousand_job_snapshot();
    let scheduler = Scheduler::default();
    let mut samples = Vec::with_capacity(100);
    for _ in 0..100 {
        let started = Instant::now();
        let plan = scheduler.schedule(&snapshot).expect("schedule");
        assert_eq!(plan.admissions.len(), 1);
        samples.push(started.elapsed());
    }
    samples.sort_unstable();
    let p99 = samples[98];
    assert!(
        p99 <= Duration::from_millis(25),
        "10,000-job scheduling p99 was {p99:?}"
    );
}

proptest! {
    #[test]
    fn queue_input_order_does_not_change_the_plan(order in prop::collection::vec(any::<u64>(), 4)) {
        let candidates = vec![
            candidate(1, "tenant-a", PriorityClass::Batch, 500, true, 1),
            candidate(2, "tenant-b", PriorityClass::Batch, 500, true, 1),
            candidate(3, "tenant-c", PriorityClass::Batch, 500, true, 1),
            candidate(4, "tenant-d", PriorityClass::Batch, 500, true, 1),
        ];
        let baseline = SchedulingSnapshot {
            now: 100.into(),
            capacity: capacity(10_000),
            queued: candidates.clone(),
            running: Vec::new(),
            tenant_weights: BTreeMap::new(),
        };
        let expected: Vec<_> = Scheduler::default()
            .schedule(&baseline)
            .expect("schedule")
            .admissions
            .into_iter()
            .map(|admission| admission.job_id)
            .collect();
        let mut indexed: Vec<_> = candidates.into_iter().enumerate().collect();
        indexed.sort_by_key(|(index, _)| (order[*index], *index));
        let shuffled = SchedulingSnapshot {
            queued: indexed.into_iter().map(|(_, candidate)| candidate).collect(),
            ..baseline
        };
        let actual: Vec<_> = Scheduler::default()
            .schedule(&shuffled)
            .expect("schedule")
            .admissions
            .into_iter()
            .map(|admission| admission.job_id)
            .collect();
        prop_assert_eq!(actual, expected);
    }
}
