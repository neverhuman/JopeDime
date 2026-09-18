use std::collections::BTreeMap;

use super::*;
use jain_smartcluster_core::{
    CpuRequest, Environment, IdempotencyKey, JainWorkload, MemoryRequest, PriorityClass, ProjectId,
    ResourceRequest, RetryPolicy, TenantId, Workload,
};

fn job(priority: PriorityClass, idempotency: &str) -> JobRecord {
    JobRecord::queued(
        JobId::new(),
        jain_smartcluster_core::JobSpec {
            tenant: TenantId::new("tenant").expect("tenant"),
            project: ProjectId::new("project").expect("project"),
            idempotency_key: IdempotencyKey::new(idempotency).expect("key"),
            priority,
            workload: Workload::Jain {
                template: JainWorkload::Training,
                arguments: Vec::new(),
                environment: Environment::default(),
            },
            resources: ResourceRequest {
                cpu: CpuRequest::fixed(1_000),
                ram: MemoryRequest::fixed(1024),
                scratch_bytes: 0,
                gpu: None,
            },
            retry: RetryPolicy::default(),
            preemption: Default::default(),
            io: Default::default(),
            deadline: None,
            labels: BTreeMap::new(),
        },
        Principal::unix(1000),
        Timestamp::from_unix_millis(1),
    )
}

#[test]
fn filters_are_owner_and_label_scoped() {
    let mut job = job(PriorityClass::Batch, "key");
    job.spec
        .labels
        .insert("kind".to_owned(), "train".to_owned());
    let filter = ListFilter {
        labels: [("kind".to_owned(), "train".to_owned())]
            .into_iter()
            .collect(),
        ..ListFilter::default()
    };
    assert!(matches_list_filter(&job, &filter));
}

#[test]
fn retry_delay_is_capped() {
    let mut job = job(PriorityClass::Batch, "key");
    job.failure_count = 100;
    assert!(retry_delay(&job) <= Duration::from_millis(job.spec.retry.maximum_backoff_ms));
}

#[test]
fn restart_retry_wait_preserves_remaining_backoff() {
    let mut job = job(PriorityClass::Batch, "key-restart");
    job.updated_at = Timestamp::from_unix_millis(1_000);
    job.failure_count = 1;
    let full = retry_delay(&job);
    let remaining = remaining_retry_delay(
        &job,
        Timestamp::from_unix_millis(1_000 + full.as_millis() as u64 / 2),
    );
    assert!(remaining <= full);
    assert!(remaining > Duration::ZERO);
    assert_eq!(
        remaining_retry_delay(
            &job,
            Timestamp::from_unix_millis(1_000 + full.as_millis() as u64 + 1),
        ),
        Duration::ZERO
    );
}

#[test]
fn tenant_is_bound_to_effective_principal() {
    let unix = Principal::unix(1000);
    assert!(tenant_matches_principal(
        &TenantId::new("unix:1000").expect("tenant"),
        &unix,
    ));
    assert!(!tenant_matches_principal(
        &TenantId::new("unix:1001").expect("tenant"),
        &unix,
    ));

    let web = Principal::web("owner-17").expect("web principal");
    assert!(tenant_matches_principal(
        &TenantId::new("web:owner-17").expect("tenant"),
        &web,
    ));
    assert!(!tenant_matches_principal(
        &TenantId::new("web:owner-18").expect("tenant"),
        &web,
    ));
}

#[test]
fn delegated_web_environment_is_detectable() {
    let mut environment = BTreeMap::new();
    environment.insert("LD_PRELOAD".to_owned(), "/tmp/not-allowed.so".to_owned());
    let workload = Workload::Jain {
        template: JainWorkload::Training,
        arguments: Vec::new(),
        environment: Environment(environment),
    };
    assert!(workload_has_environment(&workload));
}

#[test]
fn cgroup_oom_precedes_generic_exit_classification() {
    assert_eq!(
        exit_failure_class(false, false, true, false, None),
        Some(FailureClass::HostOutOfMemory)
    );
    assert_eq!(
        exit_failure_class(false, false, false, true, None),
        Some(FailureClass::SharedCudaOutOfMemory)
    );
    assert_eq!(exit_failure_class(true, false, false, false, None), None);
    assert_eq!(
        exit_failure_class(false, false, false, false, Some(FailureClass::Dependency),),
        Some(FailureClass::Dependency)
    );
}

#[test]
fn gpu_history_requires_observed_exclusive_run() {
    assert_eq!(
        gpu_history_action(true, 4 * 1024, true, None),
        GpuHistoryAction::CleanExclusive
    );
    assert_eq!(
        gpu_history_action(true, 0, true, None),
        GpuHistoryAction::None
    );
    assert_eq!(
        gpu_history_action(
            false,
            4 * 1024,
            false,
            Some(FailureClass::SharedCudaOutOfMemory),
        ),
        GpuHistoryAction::SharedCudaOom
    );
    assert_eq!(
        gpu_history_action(true, 4 * 1024, false, Some(FailureClass::Application),),
        GpuHistoryAction::FailedExclusive
    );
}

#[test]
fn only_controlled_trusted_residents_use_restart_contract() {
    let mut job = job(PriorityClass::Batch, "resident-contract");
    job.spec.preemption.checkpoint = jain_smartcluster_core::CheckpointPolicy::ControlFd {
        deadline_ms: 120_000,
    };
    assert!(
        !recovery::controlled_resident(&job.spec),
        "finite Jain jobs retain clean-exit completion"
    );
    job.spec.workload = Workload::TrustedService {
        service: jain_smartcluster_core::ServiceName::new("resident").unwrap(),
        arguments: Vec::new(),
        environment: Environment::default(),
    };
    assert!(
        !recovery::controlled_resident(&job.spec),
        "finite checkpointable trusted services retain clean-exit completion"
    );
    if let Workload::TrustedService { service, .. } = &mut job.spec.workload {
        *service = jain_smartcluster_core::ServiceName::new("jope-resident-gpu").unwrap();
    }
    assert!(recovery::controlled_resident(&job.spec));
    let mut execution = ExecutionConfig::default();
    assert!(recovery::resident_checkpoint_allowance(&job.spec, &execution).is_err());
    execution.checkpoint_timeout_ms = 120_000;
    assert_eq!(
        recovery::resident_checkpoint_allowance(&job.spec, &execution).unwrap(),
        Some(Duration::from_secs(120))
    );
    job.spec.preemption.checkpoint = jain_smartcluster_core::CheckpointPolicy::None;
    assert!(
        !recovery::controlled_resident(&job.spec),
        "finite trusted services retain clean-exit completion"
    );
}

#[test]
fn scratch_requests_require_exact_trusted_service_and_bounded_pool() {
    let mut job = job(PriorityClass::Batch, "scratch-policy");
    job.spec.resources.scratch_bytes = 10;
    let policy = crate::resource_policy::ResourcePolicy {
        trusted_scratch: Some(crate::resource_policy::ScratchReservationPool {
            path: "/mnt/scratch".into(),
            capacity_bytes: 32,
            minimum_available_bytes: 100,
            trusted_services: ["jope-pack".to_owned()].into_iter().collect(),
        }),
        ..Default::default()
    };
    assert!(policy.validate_scratch_request(&job.spec).is_err());
    job.spec.workload = Workload::TrustedService {
        service: jain_smartcluster_core::ServiceName::new("jope-pack").unwrap(),
        arguments: Vec::new(),
        environment: Environment::default(),
    };
    assert!(policy.validate_scratch_request(&job.spec).is_ok());
    job.spec.resources.scratch_bytes = 33;
    assert!(policy.validate_scratch_request(&job.spec).is_err());
}

#[test]
fn removed_scratch_allowlist_rejects_reopened_queue_before_any_attempt() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("queue.redb");
    let mut queued = job(PriorityClass::Batch, "removed-scratch-service");
    queued.spec.resources.scratch_bytes = 10;
    queued.spec.workload = Workload::TrustedService {
        service: jain_smartcluster_core::ServiceName::new("jope-packing").unwrap(),
        arguments: Vec::new(),
        environment: Environment::default(),
    };
    {
        let store = Store::open(&path).unwrap();
        store.submit(&queued).unwrap();
    }
    let store = Store::open(&path).unwrap();
    let recovered = store.get_job(queued.id).unwrap().unwrap();
    assert_eq!(recovered.state, JobState::Queued);
    let removed_policy = crate::resource_policy::ResourcePolicy::default();
    let reason = removed_policy
        .validate_scratch_request(&recovered.spec)
        .unwrap_err();
    let receipt = store
        .transition_before_effect(scheduling::prelaunch_rejection(
            &recovered,
            reason,
            100.into(),
        ))
        .unwrap();
    assert_eq!(receipt.job.state, JobState::Failed);
    assert!(receipt.job.active_attempt.is_none());
    assert!(store.attempts_for_job(queued.id).unwrap().is_empty());
    // The durable queue remains usable for subsequent valid jobs.
    store
        .submit(&job(PriorityClass::Batch, "subsequent-valid-job"))
        .unwrap();
}
