use super::*;

#[test]
fn fit_memory_is_equal_pinned_and_leaves_controller_headroom() {
    let mut experiment = fixture_experiment();
    experiment.resources.workers = 20;
    experiment.resources.cpus = 20;
    experiment.resources.memory_bytes = 64 << 30;
    let policy = fit_memory_allocation(&experiment).unwrap();
    assert_eq!(policy.per_fit_bytes, 3 << 30);
    assert_eq!(policy.controller_reserve_bytes, 4 << 30);
    let pin = policy.identity().unwrap();
    assert_eq!(policy.identity().unwrap(), pin);
    experiment.source_sha = "d".repeat(40);
    assert_ne!(
        fit_memory_allocation(&experiment)
            .unwrap()
            .identity()
            .unwrap(),
        pin
    );
    for workers in 1..=20 {
        experiment.resources.workers = workers;
        for total in [32 << 20, (64u64 << 30) - 1, 64 << 30] {
            experiment.resources.memory_bytes = total;
            let p = fit_memory_allocation(&experiment).unwrap();
            assert_eq!(p.per_fit_bytes % 4096, 0);
            assert_eq!(
                p.per_fit_bytes * u64::from(workers) + p.controller_reserve_bytes,
                total
            );
            assert!(p.controller_reserve_bytes >= total / 16);
            assert!(p.per_fit_bytes > 0);
        }
    }
    experiment.resources.memory_bytes = 4096;
    assert!(fit_memory_allocation(&experiment).is_err());
}
#[test]
fn fit_memory_policy_is_immutable_and_not_applied_to_open_only_training() {
    let scratch = scratch();
    let mut experiment = fixture_experiment();
    experiment.resources.total_seconds = 3600;
    experiment.resources.per_fit_seconds = 1200;
    let directory = scratch.path().join("journal");
    let mut pool = ExecutionPool::open(
        &experiment,
        &directory,
        Supervisor {
            unit: "unused".into(),
            cgroup_root: "/unused".into(),
        },
    )
    .unwrap();
    assert!(!directory.join("fit-memory-policy.json").exists());
    assert!(pool.run(vec![], &AtomicBool::new(false)).is_err());
    assert!(!directory.join("fit-memory-policy.json").exists());
    let policy = fit_memory_allocation(&experiment).unwrap();
    immutable(&directory.join("fit-memory-policy.json"), &policy).unwrap();
    immutable(&directory.join("fit-memory-policy.json"), &policy).unwrap();
    let mut changed = policy;
    changed.per_fit_bytes += 4096;
    assert!(immutable(&directory.join("fit-memory-policy.json"), &changed).is_err());
}

#[test]
fn datasets_get_round_robin_admission() {
    let mut queue = FairQueue::default();
    for (dataset, recipe) in [
        ('a', 'a'),
        ('a', 'b'),
        ('a', 'c'),
        ('b', 'a'),
        ('b', 'b'),
        ('c', 'a'),
    ] {
        queue.push(dataset.to_string(), fixture_job(dataset, recipe));
    }
    let mut order = Vec::new();
    while let Some(job) = queue.pop() {
        order.push(job.request.dataset_sha256.chars().next().unwrap());
    }
    assert_eq!(order, ['a', 'b', 'c', 'a', 'b', 'a']);
}
#[test]
fn job_identity_binds_argv_inputs_environment_and_arm() {
    let original = fixture_job('a', 'a');
    let hash = original.identity().unwrap();
    let mut changed = original.clone();
    changed.command.args.push("--different".into());
    assert_ne!(hash, changed.identity().unwrap());
    let mut changed = original.clone();
    changed
        .command
        .environment
        .insert("PATH".into(), "/private/bin".into());
    assert_ne!(hash, changed.identity().unwrap());
    let mut changed = original.clone();
    changed.command.inputs.push(Artifact {
        path: "/private/input.json".into(),
        sha256: "b".repeat(64),
    });
    assert_ne!(hash, changed.identity().unwrap());
    let mut changed = original;
    changed.arm = Some(ArmBudget {
        id_sha256: "d".repeat(64),
        seconds: 120,
    });
    assert_ne!(hash, changed.identity().unwrap());
    changed.arm.as_mut().unwrap().seconds = 121;
    assert!(changed.identity().is_err());
}
#[test]
fn refuses_output_escape_and_thread_override() {
    for output in ["../escape", "/absolute", "stdout.log", "stderr.log"] {
        let mut job = fixture_job('a', 'a');
        job.outputs = vec![output.into()];
        assert!(job.identity().is_err(), "{output}");
    }
    let mut job = fixture_job('a', 'a');
    job.command
        .environment
        .insert("OMP_NUM_THREADS".into(), "20".into());
    assert!(job.identity().is_err());
}
#[test]
fn missing_accounting_serializes_as_null_and_not_started_is_explicit_zero() {
    let absent = serde_json::to_value(ObservedConsumption::default()).unwrap();
    assert!(absent["cpu_seconds"].is_null());
    assert!(absent["wall_seconds"].is_null());
    let refused = WorkerOutcome::unstarted(
        &fixture_job('a', 'a'),
        EvaluationStatus::Unavailable,
        "missing engine".into(),
    )
    .unwrap();
    assert_eq!(refused.consumption.cpu_seconds, Some(0.));
    assert!(refused
        .consumption
        .accounting_reason
        .starts_with("not_started"));
}
#[test]
fn immutable_journal_refuses_deadline_reset_and_changed_identity() {
    let scratch = scratch();
    let experiment = fixture_experiment();
    let window = ExecutionWindow {
        experiment_sha256: experiment.identity().unwrap(),
        started_unix_ms: 1000,
        deadline_unix_ms: 121000,
    };
    let path = scratch.path().join("window.json");
    immutable(&path, &window).unwrap();
    immutable(&path, &window).unwrap();
    let mut extended = window.clone();
    extended.deadline_unix_ms += 1;
    assert!(immutable(&path, &extended).is_err());
    let recovered: ExecutionWindow = read_json(&fs::canonicalize(&path).unwrap()).unwrap();
    recovered.validate(&experiment).unwrap();
    assert!(!recovered.may_admit(&experiment, 111000).unwrap());
    assert!(recovered.may_admit(&experiment, 999).is_err());
}
#[test]
fn unavailable_inventory_does_not_start_the_clock() {
    let scratch = scratch();
    let root = fs::canonicalize(scratch.path()).unwrap();
    let experiment = fixture_experiment();
    let supervisor = Supervisor {
        unit: "jope-asm-unavailable.service".into(),
        cgroup_root: "/not-a-cgroup".into(),
    };
    let mut pool = ExecutionPool::open(&experiment, &root.join("journal"), supervisor).unwrap();
    assert!(pool.reserve().is_err(), "missing host capability must fail");
    assert!(pool.window().is_none());
    let mut job = fixture_job('a', 'a');
    job.directory = pool
        .directory
        .join("jobs")
        .join(job.request.identity().unwrap());
    let inventory = inventory_dependencies(&[job.clone()]).unwrap();
    assert!(!inventory[0].available);
    assert!(pool.window().is_none());
    let outcomes = pool.run(vec![job], &AtomicBool::new(false)).unwrap();
    assert_eq!(outcomes[0].command_status, EvaluationStatus::Unavailable);
    assert!(pool.window().is_none());
    assert!(!pool.directory.join("window.json").exists());
}
#[test]
fn resource_accounting_requires_original_admission_and_restores_durable_spend() {
    use crate::resource_ledger::{
        Amounts, CacheState, JoinIds, Policy, ReservationRequest, Stage, Unit, SCHEMA,
    };
    let scratch = scratch();
    let root = fs::canonicalize(scratch.path()).unwrap();
    let experiment = fixture_experiment();
    let mut pool = ExecutionPool::open(
        &experiment,
        &root.join("journal"),
        Supervisor {
            unit: "unqualified-fixture".into(),
            cgroup_root: "/not-a-cgroup".into(),
        },
    )
    .unwrap();
    let policy = Policy {
        schema: SCHEMA.into(),
        experiment_sha256: experiment.identity().unwrap(),
        opened_unix_us: 100,
        deadline_unix_us: 120_000_100,
        terminal_headroom_us: 10_000_000,
        cap: Amounts::new([
            (Unit::CpuCoreMicros, 2_000_000),
            (Unit::AttemptWallMicros, 2_000_000),
            (Unit::Evaluations, 1),
        ]),
        terminal_reserve: Amounts::new([(Unit::CpuCoreMicros, 1), (Unit::AttemptWallMicros, 1)]),
        gpu_allocation: None,
        conversion: None,
    };
    let mut ledger = pool.resource_ledger_v1(policy.clone()).unwrap();
    let job = fixture_job('a', 'a');
    let req = ReservationRequest {
        ids: JoinIds {
            experiment_sha256: experiment.identity().unwrap(),
            draw_id: "1".repeat(64),
            request_id: job.request.identity().unwrap(),
            attempt_id: "2".repeat(64),
            logical_id: "3".repeat(64),
        },
        stage: Stage::Fit,
        requested: Amounts::new([
            (Unit::CpuCoreMicros, 1_000_000),
            (Unit::AttemptWallMicros, 1_000_000),
            (Unit::Evaluations, 1),
        ]),
        retry_of: None,
        cache: CacheState::Unavailable {
            reason: "fixture".into(),
        },
    };
    let requests = [(job.identity().unwrap(), req.clone())].into();
    let error = pool
        .run_accounted_v1(vec![job], &AtomicBool::new(false), &mut ledger, requests)
        .unwrap_err();
    assert!(error.to_string().contains("admit original window"));
    assert!(pool.window.is_none());
    assert!(!pool.directory.join("resource-events-v1.json").exists());
    ledger
        .reserve(req, 100, |event| pool.persist_resource_event_v1(event))
        .unwrap();
    let restored = pool.resource_ledger_v1(policy.clone()).unwrap();
    assert_eq!(restored, ledger);
    assert_eq!(
        restored
            .reconcile()
            .unwrap()
            .active_reserved
            .get(Unit::CpuCoreMicros),
        1_000_000
    );
    let mut changed = policy;
    changed.deadline_unix_us += 1;
    assert!(pool.resource_ledger_v1(changed).is_err());
}
#[test]
fn cancelled_and_missing_dependency_inventory_does_not_poison_resume() {
    let scratch = scratch();
    let root = fs::canonicalize(scratch.path()).unwrap();
    let mut pool = ExecutionPool::open(
        &fixture_experiment(),
        &root.join("journal"),
        Supervisor {
            unit: "jope-asm-unavailable.service".into(),
            cgroup_root: "/not-a-cgroup".into(),
        },
    )
    .unwrap();
    let mut job = fixture_job('a', 'a');
    job.directory = pool
        .directory
        .join("jobs")
        .join(job.request.identity().unwrap());
    let id = job.identity().unwrap();
    let cancelled = pool.run(vec![job.clone()], &AtomicBool::new(true)).unwrap();
    assert_eq!(cancelled[0].command_status, EvaluationStatus::Cancelled);
    assert_eq!(cancelled[0].consumption.cpu_seconds, Some(0.));
    assert!(!pool.started_path(&id).exists() && !pool.finished_path(&id).exists());
    let unavailable = pool
        .run(vec![job.clone()], &AtomicBool::new(false))
        .unwrap();
    assert_eq!(unavailable[0].command_status, EvaluationStatus::Unavailable);
    assert!(pool.cached(&job).unwrap().is_none());
    assert!(!pool.finished_path(&id).exists() && pool.window().is_none());
}
