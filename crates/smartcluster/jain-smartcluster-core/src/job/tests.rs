use super::*;
use crate::{CpuRequest, Environment, JainWorkload, MemoryRequest};

fn valid_spec() -> JobSpec {
    JobSpec {
        tenant: TenantId::new("tenant-a").unwrap(),
        project: ProjectId::new("project-a").unwrap(),
        idempotency_key: IdempotencyKey::new("request-1").unwrap(),
        priority: PriorityClass::Batch,
        workload: Workload::Jain {
            template: JainWorkload::Training,
            arguments: Vec::new(),
            environment: Environment::default(),
        },
        resources: ResourceRequest {
            cpu: CpuRequest::fixed(1000),
            ram: MemoryRequest::fixed(1024),
            scratch_bytes: 0,
            gpu: None,
        },
        retry: RetryPolicy::default(),
        preemption: PreemptionPolicy::default(),
        io: IoMode::Pipes,
        deadline: None,
        labels: BTreeMap::new(),
    }
}

#[test]
fn job_spec_validates_deeply() {
    assert!(valid_spec().validate().is_ok());
    let mut invalid = valid_spec();
    invalid.resources.ram.hard_bytes = 0;
    assert!(invalid.validate().is_err());
}

#[test]
fn nonterminal_record_cannot_carry_terminal_result() {
    let mut job = JobRecord::queued(JobId::new(), valid_spec(), Principal::unix(1000), 1.into());
    job.terminal = Some(TerminalStatus::Completed {
        exit_code: 0,
        at: 2.into(),
    });
    assert!(job.validate().is_err());
}

#[test]
fn preemption_and_restart_do_not_consume_failure_budget() {
    assert!(!FailureClass::Preempted.consumes_failure_budget());
    assert!(!FailureClass::DaemonRestart.consumes_failure_budget());
    assert!(FailureClass::Infrastructure.consumes_failure_budget());
}

#[test]
fn retry_policy_rejects_unbounded_or_immediate_storms() {
    let mut policy = RetryPolicy {
        max_failures: MAX_RETRY_FAILURES + 1,
        ..RetryPolicy::default()
    };
    assert!(policy.validate().is_err());

    policy = RetryPolicy {
        initial_backoff_ms: 0,
        ..RetryPolicy::default()
    };
    assert!(policy.validate().is_err());

    policy = RetryPolicy::default();
    policy.retry_on.insert(FailureClass::Input);
    assert!(policy.validate().is_err());

    policy = RetryPolicy {
        max_failures: 0,
        initial_backoff_ms: 0,
        maximum_backoff_ms: 0,
        jitter_per_mille: 0,
        retry_on: BTreeSet::new(),
    };
    assert!(policy.validate().is_ok());
}

#[test]
fn idempotency_is_scoped_by_effective_principal_and_project() {
    let spec = valid_spec();
    let alice = IdempotencyScope::for_submission(Principal::web("alice").unwrap(), &spec);
    let repeated = IdempotencyScope::for_submission(Principal::web("alice").unwrap(), &spec);
    let bob = IdempotencyScope::for_submission(Principal::web("bob").unwrap(), &spec);
    assert_eq!(alice, repeated);
    assert_ne!(alice, bob);
}
