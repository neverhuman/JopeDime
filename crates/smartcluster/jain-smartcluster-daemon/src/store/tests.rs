use std::collections::BTreeMap;

use jain_smartcluster_core::{
    CpuRequest, Environment, IdempotencyKey, JainWorkload, MemoryRequest, PreemptionPolicy,
    Principal, PriorityClass, ResourceRequest, RetryPolicy, TenantId, Workload,
};
use tempfile::TempDir;

use super::*;

fn queued_job(idempotency: &str, at: u64) -> JobRecord {
    let spec = jain_smartcluster_core::JobSpec {
        tenant: TenantId::new("tenant-a").expect("tenant"),
        project: jain_smartcluster_core::ProjectId::new("project-a").expect("project"),
        idempotency_key: IdempotencyKey::new(idempotency).expect("key"),
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
        io: jain_smartcluster_core::IoMode::Pipes,
        deadline: None,
        labels: BTreeMap::new(),
    };
    JobRecord::queued(JobId::new(), spec, Principal::unix(1000), at.into())
}

fn transition(job: &JobRecord, to: JobState, at: u64) -> TransitionMutation {
    TransitionMutation::nonterminal(FencedTransition {
        job_id: job.id,
        from: job.state,
        to,
        attempt: job.active_attempt,
        at: at.into(),
        reason: None,
    })
}

#[test]
fn submission_is_atomic_and_idempotent_across_reopen() {
    let directory = TempDir::new().expect("tempdir");
    let path = directory.path().join("state.redb");
    let job = queued_job("same-request", 1);
    {
        let store = Store::open(&path).expect("open");
        assert!(matches!(store.submit(&job), Ok(SubmitOutcome::Created(_))));
    }
    let store = Store::open(&path).expect("reopen");
    let replay = queued_job("same-request", 2);
    let outcome = store.submit(&replay).expect("idempotent replay");
    assert!(matches!(outcome, SubmitOutcome::Existing(_)));
    assert_eq!(outcome.job().id, job.id);
    assert_eq!(store.all_jobs().expect("jobs").len(), 1);
}

#[test]
fn changed_spec_under_same_idempotency_key_conflicts() {
    let directory = TempDir::new().expect("tempdir");
    let store = Store::open(directory.path().join("state.redb")).expect("open");
    let job = queued_job("same-request", 1);
    store.submit(&job).expect("submit");
    let mut changed = queued_job("same-request", 2);
    changed.spec.resources.cpu = CpuRequest::fixed(2000);
    assert!(matches!(
        store.submit(&changed),
        Err(StoreError::IdempotencyConflict { existing_job }) if existing_job == job.id
    ));
}

#[test]
fn attempt_is_fenced_and_starting_is_durable_before_launch() {
    let directory = TempDir::new().expect("tempdir");
    let path = directory.path().join("state.redb");
    let store = Store::open(&path).expect("open");
    let job = queued_job("launch", 1);
    store.submit(&job).expect("submit");
    let admitted = store
        .transition_before_effect(transition(&job, JobState::Admitted, 2))
        .expect("admit")
        .job;
    let launch = store
        .begin_attempt_before_launch(admitted.id, 3.into(), 103.into())
        .expect("begin attempt");
    assert_eq!(launch.job.state, JobState::Starting);
    assert_eq!(launch.attempt.state, JobState::Starting);
    drop(store);

    let reopened = Store::open(&path).expect("reopen");
    let recovered = reopened.get_job(job.id).expect("read").expect("job");
    assert_eq!(recovered.active_attempt, Some(launch.attempt.attempt));
    assert!(reopened
        .begin_attempt_before_launch(job.id, 4.into(), 104.into())
        .is_err());
}

#[test]
fn queue_and_deadline_indexes_follow_durable_transitions_and_reopen() {
    let directory = TempDir::new().expect("tempdir");
    let path = directory.path().join("state.redb");
    let mut job = queued_job("indexed", 1);
    job.spec.deadline = Some(5.into());
    let store = Store::open(&path).expect("open");
    store.submit(&job).expect("submit");
    assert_eq!(store.queued_jobs(10).expect("queued"), vec![job.clone()]);
    assert_eq!(
        store
            .expired_deadline_jobs(4.into(), 10)
            .expect("not expired"),
        Vec::<JobRecord>::new()
    );
    assert_eq!(
        store
            .expired_deadline_jobs(5.into(), 10)
            .expect("expired")
            .len(),
        1
    );

    let admitted = store
        .transition_before_effect(transition(&job, JobState::Admitted, 2))
        .expect("admit")
        .job;
    assert!(store.queued_jobs(10).expect("empty queue").is_empty());
    assert_eq!(
        store
            .expired_deadline_jobs(5.into(), 10)
            .expect("admitted deadline")
            .first()
            .map(|job| job.id),
        Some(job.id)
    );

    let launch = store
        .begin_attempt_before_launch(admitted.id, 3.into(), 103.into())
        .expect("launch");
    assert!(store
        .expired_deadline_jobs(5.into(), 10)
        .expect("starting removed")
        .is_empty());
    drop(store);
    let reopened = Store::open(&path).expect("reopen");
    assert!(reopened.queued_jobs(10).expect("reopened queue").is_empty());
    assert_eq!(
        reopened
            .get_job(job.id)
            .expect("job")
            .expect("record")
            .active_attempt,
        Some(launch.attempt.attempt)
    );
}

#[test]
fn transition_event_outbox_survives_reopen_until_acknowledged() {
    let directory = TempDir::new().expect("tempdir");
    let path = directory.path().join("state.redb");
    let job = queued_job("transition-outbox", 1);
    let admitted_transition;
    {
        let store = Store::open(&path).expect("open");
        store.submit(&job).expect("submit");
        let receipt = store
            .transition_before_effect(transition(&job, JobState::Admitted, 2))
            .expect("admit");
        admitted_transition = receipt.transition;
        assert_eq!(
            store.pending_transition_events().expect("pending"),
            [PendingTransitionEvent {
                transition: admitted_transition.clone(),
            }]
        );
    }

    let store = Store::open(&path).expect("reopen");
    assert_eq!(store.pending_transition_events().expect("pending").len(), 1);
    store
        .acknowledge_transition_event(&admitted_transition)
        .expect("acknowledge");
    store
        .acknowledge_transition_event(&admitted_transition)
        .expect("idempotent acknowledge");
    assert!(store
        .pending_transition_events()
        .expect("empty outbox")
        .is_empty());
}

#[test]
fn superseded_attempt_cannot_complete_a_new_attempt() {
    let directory = TempDir::new().expect("tempdir");
    let store = Store::open(directory.path().join("state.redb")).expect("open");
    let job = queued_job("fence", 1);
    store.submit(&job).expect("submit");
    let admitted = store
        .transition_before_effect(transition(&job, JobState::Admitted, 2))
        .expect("admit")
        .job;
    let first = store
        .begin_attempt_before_launch(admitted.id, 3.into(), 100.into())
        .expect("first attempt");
    let running = store
        .transition_before_effect(transition(&first.job, JobState::Running, 4))
        .expect("running")
        .job;
    let mut retry = transition(&running, JobState::RetryWait, 5);
    retry.failure = Some(FailureClass::Infrastructure);
    let retry_wait = store
        .transition_before_effect(retry)
        .expect("retry wait")
        .job;
    let queued = store
        .transition_before_effect(transition(&retry_wait, JobState::Queued, 6))
        .expect("queue")
        .job;
    let admitted = store
        .transition_before_effect(transition(&queued, JobState::Admitted, 7))
        .expect("admit")
        .job;
    let second = store
        .begin_attempt_before_launch(admitted.id, 8.into(), 108.into())
        .expect("second attempt");
    assert_ne!(first.attempt.attempt.fence, second.attempt.attempt.fence);

    let superseded = TransitionMutation {
        transition: FencedTransition {
            job_id: job.id,
            from: JobState::Starting,
            to: JobState::Completed,
            attempt: Some(first.attempt.attempt),
            at: 9.into(),
            reason: None,
        },
        terminal: Some(TerminalStatus::Completed {
            exit_code: 0,
            at: 9.into(),
        }),
        failure: None,
    };
    assert!(store.transition_before_effect(superseded).is_err());
    assert_eq!(
        store
            .get_job(job.id)
            .expect("get")
            .expect("job")
            .active_attempt,
        Some(second.attempt.attempt)
    );
}

#[test]
fn shared_oom_requires_exclusive_retry_and_five_new_clean_runs() {
    let directory = TempDir::new().expect("tempdir");
    let store = Store::open(directory.path().join("state.redb")).expect("open");
    let history = store
        .update_resource_history("train:v1", 1.into(), |history| {
            history.record_shared_cuda_oom(8_000, 1.into());
        })
        .expect("oom");
    assert!(history.sharing_disabled);
    assert!(history.exclusive_retry_pending);
    for run in 0..5 {
        let history = store
            .update_resource_history("train:v1", (run + 2).into(), |history| {
                history.record_clean_exclusive(7_000, (run + 2).into());
            })
            .expect("clean run");
        assert_eq!(history.sharing_disabled, run < 4);
    }
}

#[test]
fn typed_metadata_round_trips() {
    let directory = TempDir::new().expect("tempdir");
    let store = Store::open(directory.path().join("state.redb")).expect("open");
    store.put_meta("controller-epoch", &42_u64).expect("put");
    assert_eq!(
        store.get_meta::<u64>("controller-epoch").expect("get"),
        Some(42)
    );
}

#[test]
fn idempotency_scope_includes_project() {
    let directory = TempDir::new().expect("tempdir");
    let store = Store::open(directory.path().join("state.redb")).expect("open");
    let first = queued_job("request", 1);
    store.submit(&first).expect("first");
    let mut second = queued_job("request", 2);
    second.spec.project = jain_smartcluster_core::ProjectId::new("project-b").expect("project");
    assert!(matches!(
        store.submit(&second),
        Ok(SubmitOutcome::Created(_))
    ));
    assert_eq!(store.all_jobs().expect("jobs").len(), 2);
}

#[test]
fn failed_terminal_validation_never_partially_mutates_a_job() {
    let directory = TempDir::new().expect("tempdir");
    let store = Store::open(directory.path().join("state.redb")).expect("open");
    let job = queued_job("invalid-terminal", 1);
    store.submit(&job).expect("submit");
    let invalid = TransitionMutation::nonterminal(FencedTransition {
        job_id: job.id,
        from: JobState::Queued,
        to: JobState::Cancelled,
        attempt: None,
        at: 2.into(),
        reason: None,
    });
    assert!(matches!(
        store.transition_before_effect(invalid),
        Err(StoreError::InvalidTransition(_))
    ));
    assert_eq!(
        store.get_job(job.id).expect("get").expect("job").state,
        JobState::Queued
    );
}

#[test]
fn failure_budget_excludes_restart_and_preemption() {
    let directory = TempDir::new().expect("tempdir");
    let store = Store::open(directory.path().join("state.redb")).expect("open");
    let job = queued_job("failure-budget", 1);
    store.submit(&job).expect("submit");
    let admitted = store
        .transition_before_effect(transition(&job, JobState::Admitted, 2))
        .expect("admit")
        .job;
    let launch = store
        .begin_attempt_before_launch(admitted.id, 3.into(), 103.into())
        .expect("launch");
    let running = store
        .transition_before_effect(transition(&launch.job, JobState::Running, 4))
        .expect("run")
        .job;
    let mut restart = transition(&running, JobState::RetryWait, 5);
    restart.failure = Some(FailureClass::DaemonRestart);
    let retry_wait = store
        .transition_before_effect(restart)
        .expect("restart")
        .job;
    assert_eq!(retry_wait.failure_count, 0);

    let queued = store
        .transition_before_effect(transition(&retry_wait, JobState::Queued, 6))
        .expect("queue")
        .job;
    let admitted = store
        .transition_before_effect(transition(&queued, JobState::Admitted, 7))
        .expect("admit")
        .job;
    let launch = store
        .begin_attempt_before_launch(admitted.id, 8.into(), 108.into())
        .expect("launch")
        .job;
    let running = store
        .transition_before_effect(transition(&launch, JobState::Running, 9))
        .expect("run")
        .job;
    let mut infrastructure = transition(&running, JobState::RetryWait, 10);
    infrastructure.failure = Some(FailureClass::Infrastructure);
    let retry_wait = store
        .transition_before_effect(infrastructure)
        .expect("failure")
        .job;
    assert_eq!(retry_wait.failure_count, 1);
}

#[test]
fn crash_point_reopen_after_each_commit_preserves_the_state_machine() {
    let directory = TempDir::new().expect("tempdir");
    let path = directory.path().join("state.redb");
    let job = queued_job("chaos-reopen", 1);
    Store::open(&path)
        .expect("open")
        .submit(&job)
        .expect("submit");

    let store = Store::open(&path).expect("reopen queued");
    let admitted = store
        .transition_before_effect(transition(&job, JobState::Admitted, 2))
        .expect("admit")
        .job;
    drop(store);

    let store = Store::open(&path).expect("reopen admitted");
    let launch = store
        .begin_attempt_before_launch(admitted.id, 3.into(), 103.into())
        .expect("launch");
    drop(store);

    let store = Store::open(&path).expect("reopen starting");
    store
        .record_spawned(launch.attempt.attempt, 42, "/scq/job/attempt-1".into())
        .expect("record spawned");
    drop(store);

    let store = Store::open(&path).expect("reopen spawned");
    let starting = store.get_job(job.id).expect("get").expect("job");
    let running = store
        .transition_before_effect(transition(&starting, JobState::Running, 4))
        .expect("running")
        .job;
    drop(store);

    let store = Store::open(&path).expect("reopen running");
    let paused = store
        .transition_before_effect(transition(&running, JobState::Paused, 5))
        .expect("paused")
        .job;
    drop(store);

    let store = Store::open(&path).expect("reopen paused");
    let running = store
        .transition_before_effect(transition(&paused, JobState::Running, 6))
        .expect("resumed")
        .job;
    drop(store);

    let store = Store::open(&path).expect("reopen resumed");
    let completed = store
        .transition_before_effect(TransitionMutation {
            transition: FencedTransition {
                job_id: job.id,
                from: JobState::Running,
                to: JobState::Completed,
                attempt: running.active_attempt,
                at: 7.into(),
                reason: None,
            },
            terminal: Some(TerminalStatus::Completed {
                exit_code: 0,
                at: 7.into(),
            }),
            failure: None,
        })
        .expect("complete")
        .job;
    drop(store);

    let store = Store::open(&path).expect("reopen terminal");
    assert_eq!(store.get_job(job.id).expect("get").expect("job"), completed);
    let attempts = store.attempts_for_job(job.id).expect("attempts");
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].finished_at, Some(7.into()));
    assert_eq!(attempts[0].pid, Some(42));
}
