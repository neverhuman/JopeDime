use super::*;

#[test]
fn resumed_prior_unit_is_unknown_and_foreign_live_group_is_not_drained() {
    let scratch = scratch();
    let root = fs::canonicalize(scratch.path()).unwrap();
    let pool = ExecutionPool::open(
        &fixture_experiment(),
        &root.join("journal"),
        Supervisor {
            unit: "jope-asm-new.service".into(),
            cgroup_root: "/sys/fs/cgroup/system.slice/jope-asm-new.service".into(),
        },
    )
    .unwrap();
    let job = fixture_job('a', 'a');
    let id = job.identity().unwrap();
    let original = Started {
        job: job.clone(),
        job_sha256: id.clone(),
        started_unix_ms: 1,
        fit_deadline_unix_ms: 1000,
        cgroup: PathBuf::from("/sys/fs/cgroup/system.slice/jope-asm-absent-prior-fixture.service")
            .join(format!("fit-{id}")),
    };
    immutable(&pool.started_path(&id), &original).unwrap();
    let outcome = pool.interrupted(&job, true).unwrap();
    assert_eq!(outcome.command_status, EvaluationStatus::Cancelled);
    assert!(outcome
        .reason
        .unwrap()
        .starts_with("controller_interrupted"));
    assert!(outcome.consumption.cpu_seconds.is_none());
    let foreign = root.join("foreign");
    fs::create_dir(&foreign).unwrap();
    fs::write(foreign.join("cgroup.events"), "populated 1\n").unwrap();
    let mut other = fixture_job('b', 'b');
    other.request.recipe_sha256 = "c".repeat(64);
    let other_id = other.identity().unwrap();
    immutable(
        &pool.started_path(&other_id),
        &Started {
            job: other.clone(),
            job_sha256: other_id,
            started_unix_ms: 1,
            fit_deadline_unix_ms: 1000,
            cgroup: foreign.clone(),
        },
    )
    .unwrap();
    assert!(pool
        .interrupted(&other, true)
        .unwrap()
        .reason
        .unwrap()
        .contains("recovery_not_drained"));
    assert!(!foreign.join("cgroup.kill").exists());
}
#[test]
fn finished_receipt_cannot_hide_live_group_from_later_batch() {
    let scratch = scratch();
    let root = fs::canonicalize(scratch.path()).unwrap();
    let pool = ExecutionPool::open(
        &fixture_experiment(),
        &root.join("journal"),
        Supervisor {
            unit: "jope-asm-new.service".into(),
            cgroup_root: "/sys/fs/cgroup/system.slice/jope-asm-new.service".into(),
        },
    )
    .unwrap();
    let job = fixture_job('c', 'd');
    let id = job.identity().unwrap();
    let foreign = root.join("foreign");
    fs::create_dir(&foreign).unwrap();
    fs::write(foreign.join("cgroup.events"), "populated 1\n").unwrap();
    let original = Started {
        job: job.clone(),
        job_sha256: id.clone(),
        started_unix_ms: 1,
        fit_deadline_unix_ms: 100,
        cgroup: foreign.clone(),
    };
    immutable(&pool.started_path(&id), &original).unwrap();
    immutable(&pool.cleanup_path(&id), &original).unwrap();
    let interrupted = pool.interrupted(&job, true).unwrap();
    immutable(&pool.finished_path(&id), &interrupted).unwrap();
    assert!(pool.cached(&job).unwrap().is_some());
    assert!(pool.reconcile_cleanup().unwrap().is_some());
    assert!(pool.cleanup_path(&id).exists());
    assert!(!foreign.join("cgroup.kill").exists());
    fs::write(foreign.join("cgroup.events"), "populated 0\n").unwrap();
    assert!(pool.reconcile_cleanup().unwrap().is_none());
    assert!(!pool.cleanup_path(&id).exists());
}
#[test]
fn foreign_asm_crash_gap_blocks_without_drain() {
    let scratch = scratch();
    let root = fs::canonicalize(scratch.path()).unwrap();
    let current = root.join("jope-asm-current.service");
    let foreign = root.join("jope-asm-old.service");
    fs::create_dir(&current).unwrap();
    fs::write(current.join("cgroup.events"), "populated 1\n").unwrap();
    fs::create_dir(&foreign).unwrap();
    fs::write(foreign.join("cgroup.events"), "populated 1\n").unwrap();
    assert!(require_no_other_asm_groups(&root, &current).is_err());
    assert!(!foreign.join("cgroup.kill").exists());
    fs::write(foreign.join("cgroup.events"), "populated 0\n").unwrap();
    require_no_other_asm_groups(&root, &current).unwrap();
    fs::remove_file(foreign.join("cgroup.events")).unwrap();
    assert!(require_no_other_asm_groups(&root, &current).is_err());
}
#[test]
fn per_fit_allowance_cannot_exceed_thirty_seconds() {
    let scratch = scratch();
    let mut experiment = fixture_experiment();
    experiment.resources.total_seconds = 3600;
    experiment.resources.per_fit_seconds = 1200;
    let mut pool = ExecutionPool::open(
        &experiment,
        &scratch.path().join("journal"),
        Supervisor {
            unit: "unused".into(),
            cgroup_root: "/unused".into(),
        },
    )
    .unwrap();
    assert!(pool.run(vec![], &AtomicBool::new(false)).is_err());
    assert!(pool.window().is_none());
}

#[test]
fn arm_window_cannot_reset_or_change_allowance() {
    let scratch = scratch();
    let root = fs::canonicalize(scratch.path()).unwrap();
    let pool = ExecutionPool::open(
        &fixture_experiment(),
        &root.join("journal"),
        Supervisor {
            unit: "unused".into(),
            cgroup_root: root,
        },
    )
    .unwrap();
    let mut arm = ArmBudget {
        id_sha256: "a".repeat(64),
        seconds: 120,
    };
    let original = pool.arm_deadline(&arm).unwrap();
    assert_eq!(pool.arm_deadline(&arm).unwrap(), original);
    arm.seconds = 119;
    assert!(pool.arm_deadline(&arm).is_err());
}
#[test]
fn changed_output_cannot_be_a_cached_success() {
    let scratch = scratch();
    let path = scratch.path().join("predictions.json");
    write_new(&path, b"[1,2]").unwrap();
    let artifact = Artifact {
        path: fs::canonicalize(&path).unwrap(),
        sha256: hash_file(&path).unwrap(),
    };
    artifact.verify().unwrap();
    fs::write(&path, b"[9,9]").unwrap();
    assert!(artifact.verify().is_err());
}
#[test]
fn capture_drains_after_retention_bound() {
    let bytes = vec![b'x'; LOG_LIMIT * 3];
    let retained = capture(std::io::Cursor::new(bytes))
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(retained.len(), LOG_LIMIT);
}
#[test]
fn cgroup_and_systemd_parsers_reject_unbounded_allowances() {
    assert_eq!(parse_cpus("2-4,4,9").unwrap(), [2, 3, 4, 9]);
    assert!(parse_cpus("4-2").is_err());
    assert!(parse_cpus("").is_err());
    assert_eq!(parse_timespan("1min 2s 3ms 4us").unwrap(), 62_003_004);
    assert_eq!(parse_timespan("1.2345678s").unwrap(), 1_234_568);
    assert_eq!(parse_timespan("2.5ms").unwrap(), 2500);
    assert!(parse_timespan("1.nan s").is_err());
    assert!(parse_timespan("infinity").is_err());
    assert_eq!(parse_timespan("0").unwrap(), 0);
}
