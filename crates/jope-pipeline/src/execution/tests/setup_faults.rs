use super::setup_lifecycle::{HostSetupCallback, HostSetupStage};
use super::*;

pub(super) fn qualify_setup_faults(pool: &mut ExecutionPool, physical: &Path, case: &str) {
    use std::sync::{mpsc, Mutex};
    let script = physical.join("setup-success.py");
    write_new(
        &script,
        b"from pathlib import Path\nPath('predictions.json').write_text('[]')\n",
    )
    .unwrap();
    let input = physical.join("setup-pinned-input");
    write_new(&input, b"original pinned bytes").unwrap();
    let mut failed = setup_host_job(pool, &script, 'a');
    failed.command.inputs.push(Artifact {
        path: input.clone(),
        sha256: hash_file(&input).unwrap(),
    });
    let blocker = setup_host_job(pool, &script, 'b');
    let replacement = setup_host_job(pool, &script, 'c');
    let failed_id = failed.identity().unwrap();
    let blocker_id = blocker.identity().unwrap();
    let replacement_id = replacement.identity().unwrap();
    let failed_group = pool.supervisor.cgroup_root.join(format!("fit-{failed_id}"));
    let cleanup = pool.cleanup_path(&failed_id);
    let started_path = pool.started_path(&failed_id);
    let original = pool.admit().unwrap().clone();
    let original_bytes = fs::read(pool.directory.join("window.json")).unwrap();
    let cancel_case = case == "setup-cancel";
    let cancellation = Arc::new(AtomicBool::new(false));
    let orphan = Arc::new(OwnedSetupOrphan::default());
    let trace = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
    let first_cpu = Arc::new(Mutex::new(None::<String>));
    let (entered, notices) = mpsc::channel::<String>();
    let (release_failed, failed_wait) = mpsc::channel::<()>();
    let (release_blocker, blocker_wait) = mpsc::channel::<()>();
    let failed_wait = Mutex::new(failed_wait);
    let blocker_wait = Mutex::new(blocker_wait);
    let callback = {
        let failed_id = failed_id.clone();
        let blocker_id = blocker_id.clone();
        let replacement_id = replacement_id.clone();
        let failed_group = failed_group.clone();
        let cleanup = cleanup.clone();
        let started_path = started_path.clone();
        let orphan = orphan.clone();
        let trace = trace.clone();
        let first_cpu = first_cpu.clone();
        Arc::new(
            move |stage: HostSetupStage, record: &Started| -> Result<()> {
                let id = &record.job_sha256;
                if id != &failed_id && id != &blocker_id && id != &replacement_id {
                    return Ok(());
                }
                if stage == HostSetupStage::Attached {
                    ensure!(
                        named_counter(&record.cgroup, "cgroup.events", "populated")? == 1,
                        "fixture did not reach an attached live gated wrapper"
                    );
                    if id == &failed_id {
                        ensure!(
                            cleanup.is_file() && !started_path.exists(),
                            "setup ownership must precede gate evidence"
                        );
                        *first_cpu.lock().unwrap() =
                            Some(control_text(&record.cgroup, "cpuset.cpus.effective")?);
                    }
                    if id == &replacement_id {
                        ensure!(
                            setup_group_drained(&failed_group)? && !cleanup.exists(),
                            "replacement admitted before failed setup drainage/cleanup"
                        );
                        orphan.reaped_after_recovery()?;
                        ensure!(
                            Some(control_text(&record.cgroup, "cpuset.cpus.effective")?)
                                == *first_cpu.lock().unwrap(),
                            "replacement did not reuse the first reserved CPU"
                        );
                    }
                    trace.lock().unwrap().push(serde_json::json!({"stage":"attached","job":id,
                        "group":record.cgroup,"deadline_unix_ms":record.fit_deadline_unix_ms,
                        "at_unix_ms":now_ms()?,"prior_group_drained":if id == &replacement_id {Some(true)}else{None}}));
                    if !cancel_case || id != &replacement_id {
                        entered.send(id.clone())?;
                    }
                    if id == &failed_id {
                        failed_wait
                            .lock()
                            .unwrap()
                            .recv_timeout(Duration::from_secs(5))?;
                    } else if id == &blocker_id {
                        blocker_wait
                            .lock()
                            .unwrap()
                            .recv_timeout(Duration::from_secs(5))?;
                    }
                } else if stage == HostSetupStage::AfterDropKill && id == &failed_id {
                    let pid = orphan.insert(record)?;
                    trace.lock().unwrap().push(
                        serde_json::json!({"stage":"owned_orphan_attached_after_drop_kill",
                        "job":id,"pid":pid,"group":record.cgroup,"at_unix_ms":now_ms()?}),
                    );
                }
                Ok(())
            },
        ) as Arc<HostSetupCallback>
    };
    let guard = HostSetupGuard::install(callback);
    let (result, driver) = thread::scope(|scope| {
        let driver_failed_id = failed_id.clone();
        let driver_blocker_id = blocker_id.clone();
        let driver_replacement_id = replacement_id.clone();
        let driver_cancellation = cancellation.clone();
        let driver = scope.spawn(move || -> Result<()> {
            let result = (|| -> Result<()> {
                if cancel_case {
                    ensure!(
                        notices.recv_timeout(Duration::from_secs(3))? == driver_failed_id,
                        "cancellation fixture did not reach actual setup"
                    );
                    driver_cancellation.store(true, Ordering::Release);
                } else {
                    let a = notices.recv_timeout(Duration::from_secs(3))?;
                    let b = notices.recv_timeout(Duration::from_secs(3))?;
                    ensure!(
                        BTreeSet::from([a, b])
                            == BTreeSet::from([
                                driver_failed_id.clone(),
                                driver_blocker_id.clone()
                            ]),
                        "two actual setups did not enter before either was released"
                    );
                    fs::write(&input, b"changed bytes after real cgroup attachment")?;
                }
                release_failed.send(())?;
                if !cancel_case {
                    ensure!(
                        notices.recv_timeout(Duration::from_secs(3))? == driver_replacement_id,
                        "replacement did not enter while second setup still held its CPU"
                    );
                }
                Ok(())
            })();
            // Always release both barriers, including assertion/error paths.
            let _ = release_failed.send(());
            let _ = release_blocker.send(());
            result
        });
        let jobs = if cancel_case {
            vec![failed.clone()]
        } else {
            vec![failed.clone(), blocker.clone(), replacement.clone()]
        };
        // The same bounded pool admission/recovery path; use its shared Arc
        // directly to make cancellation during the barrier deterministic.
        let result = pool.run_inner(jobs, cancellation.clone());
        (result, driver.join().unwrap())
    });
    let outcomes = result.unwrap();
    let diagnostic = serde_json::json!({"case":case,"outcomes":outcomes,
            "trace":*trace.lock().unwrap(),"driver_error":driver.as_ref().err().map(|e|format!("{e:#}"))});
    immutable(&physical.join("setup-outcomes.json"), &diagnostic).unwrap();
    println!("execution_setup_diagnostic={diagnostic}");
    driver.unwrap();
    let outcome = outcomes.iter().find(|o| o.job_sha256 == failed_id).unwrap();
    assert_eq!(
        outcome.command_status,
        if cancel_case {
            EvaluationStatus::Cancelled
        } else {
            EvaluationStatus::Unavailable
        }
    );
    if !cancel_case {
        assert!(outcome
            .reason
            .as_ref()
            .unwrap()
            .contains("artifact changed:"));
    }
    assert!(outcome.consumption.gate_released_unix_ms.is_none());
    assert!(outcome.consumption.cpu_seconds.is_none());
    assert!(outcome.exit_code.is_none());
    assert!(!started_path.exists());
    assert!(!failed.directory.join("predictions.json").exists());
    assert!(setup_group_drained(&failed_group).unwrap());
    assert!(!cleanup.exists());
    let orphan_pid = orphan.reaped_after_recovery().unwrap();
    let resumed = if cancel_case {
        // No replacement clock: same pool/journal/immutable original window.
        pool.run(vec![replacement.clone()], &AtomicBool::new(false))
            .unwrap()
    } else {
        assert!(outcomes
            .iter()
            .filter(|o| o.job_sha256 != failed_id)
            .all(|o| o.command_status == EvaluationStatus::Completed));
        vec![outcomes
            .iter()
            .find(|o| o.job_sha256 == replacement_id)
            .unwrap()
            .clone()]
    };
    assert_eq!(resumed[0].command_status, EvaluationStatus::Completed);
    assert!(resumed[0].consumption.gate_released_unix_ms.is_some());
    assert_eq!(
        fs::read(pool.directory.join("window.json")).unwrap(),
        original_bytes
    );
    assert_eq!(
        pool.window().unwrap().deadline_unix_ms,
        original.deadline_unix_ms
    );
    drop(guard);
    let before_setup = if cancel_case {
        qualify_before_setup_cancellation(pool, physical, &script, &original)
    } else {
        false
    };
    let receipt = serde_json::json!({"case":case,"journal":pool.directory,
            "original_window":original,"resumed_window":pool.window(),
            "window_sha256":hash_file(&pool.directory.join("window.json")).unwrap(),
            "diagnostic_path":physical.join("setup-outcomes.json"),
            "diagnostic_sha256":hash_file(&physical.join("setup-outcomes.json")).unwrap(),
            "outcome":outcome,"replacement":resumed[0],"no_failed_gate":true,
            "owned_orphan_pid":orphan_pid,"owned_orphan_killed_before_reuse":true,
            "failed_group_drained":true,"cleanup_cleared_before_reuse":true,
            "same_cpu_reused":true,"actual_parallel_setup":!cancel_case,
            "before_setup_cancellation_verified":before_setup});
    immutable(&physical.join("setup-host-receipt.json"), &receipt).unwrap();
    println!("execution_host_receipt={receipt}");
}
fn qualify_before_setup_cancellation(
    pool: &mut ExecutionPool,
    physical: &Path,
    script: &Path,
    original: &ExecutionWindow,
) -> bool {
    let job = setup_host_job(pool, script, 'd');
    let id = job.identity().unwrap();
    let group = pool.supervisor.cgroup_root.join(format!("fit-{id}"));
    let started = Started {
        job,
        job_sha256: id.clone(),
        started_unix_ms: now_ms().unwrap(),
        fit_deadline_unix_ms: original.deadline_unix_ms
            - pool.experiment.resources.reporting_seconds * 1000,
        cgroup: group.clone(),
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = cancelled.clone();
    let attached = Arc::new(AtomicBool::new(false));
    let observed = attached.clone();
    let expected = id.clone();
    let guard = HostSetupGuard::install(Arc::new(move |stage, record| {
        if record.job_sha256 == expected {
            if stage == HostSetupStage::BeforeSetup {
                flag.store(true, Ordering::Release);
            }
            if stage == HostSetupStage::Attached {
                observed.store(true, Ordering::Release);
            }
        }
        Ok(())
    }));
    immutable(&pool.cleanup_path(&id), &started).unwrap();
    let cpu =
        parse_cpus(&control_text(&pool.supervisor.cgroup_root, "cpuset.cpus.effective").unwrap())
            .unwrap()[0];
    let running = Running::start(
        started.clone(),
        cpu,
        fit_memory_allocation(&pool.experiment)
            .unwrap()
            .per_fit_bytes,
        pool.experiment.resources.per_fit_seconds,
        pool.started_path(&id),
        pool.directory.join("logs"),
        cancelled,
    )
    .unwrap();
    let outcome = running.finish().unwrap();
    immutable(&physical.join("before-setup-cancellation.json"), &outcome).unwrap();
    assert_eq!(outcome.command_status, EvaluationStatus::Cancelled);
    assert!(
        !attached.load(Ordering::Acquire),
        "cancelled startup constructed a gated process"
    );
    assert!(!group.exists());
    assert!(!pool.started_path(&id).exists());
    assert!(outcome.consumption.gate_released_unix_ms.is_none());
    assert!(pool.recover_group(&started, true).unwrap().is_empty());
    pool.clear_cleanup(&id).unwrap();
    drop(guard);
    true
}

pub(super) fn qualify_concurrency(pool: &mut ExecutionPool, physical: &Path) {
    let script = physical.join("concurrency.py");
    let shared = physical.join("barrier");
    fs::create_dir(&shared).unwrap();
    write_new(
        &script,
        br#"import fcntl,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); index=sys.argv[2]
def update(delta):
 with (root/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  state=root/'state.json'
  data=json.loads(state.read_text()) if state.exists() else {'active':0,'peak':0}
  data['active']+=delta; data['peak']=max(data['peak'],data['active'])
  state.write_text(json.dumps(data))
update(1)
(root/('ready-'+index)).write_text('ready')
try:
 deadline=time.monotonic()+8
 while len(list(root.glob('ready-*')))<20:
  if time.monotonic()>deadline: raise RuntimeError('twenty-worker barrier not reached')
  time.sleep(.005)
 time.sleep(.1)
 pathlib.Path('predictions.json').write_text('[]')
finally: update(-1)
"#,
    )
    .unwrap();
    let python = fs::canonicalize("/usr/bin/python3").unwrap();
    let executable = Artifact {
        sha256: hash_file(&python).unwrap(),
        path: python,
    };
    let script_artifact = Artifact {
        sha256: hash_file(&script).unwrap(),
        path: script.clone(),
    };
    let mut jobs = Vec::new();
    for i in 0..21 {
        let mut job = fixture_job('a', 'a');
        job.request.experiment_sha256 = pool.experiment.identity().unwrap();
        job.request.recipe_sha256 = format!("{i:064x}");
        job.directory = pool
            .directory
            .join("jobs")
            .join(job.request.identity().unwrap());
        job.command = WorkerCommand {
            program: executable.clone(),
            args: vec![
                script.to_string_lossy().into_owned(),
                shared.to_string_lossy().into_owned(),
                i.to_string(),
            ],
            inputs: vec![script_artifact.clone()],
            environment: BTreeMap::new(),
        };
        jobs.push(job);
    }
    assert!(inventory_dependencies(&jobs)
        .unwrap()
        .iter()
        .all(|check| check.available));
    let outcomes = pool.run(jobs, &AtomicBool::new(false)).unwrap();
    assert_eq!(outcomes.len(), 21);
    assert!(
        outcomes
            .iter()
            .all(|outcome| outcome.command_status == EvaluationStatus::Completed),
        "all 21 synthetic commands must finish"
    );
    let state: serde_json::Value = read_json(&shared.join("state.json")).unwrap();
    assert!(outcomes.iter().all(|o| o.consumption.cpu_seconds.is_some()));
    assert_eq!(state["peak"], 20);
    assert_eq!(state["active"], 0);
    let memory = fit_memory_allocation(&pool.experiment).unwrap();
    assert!(outcomes
        .iter()
        .all(|outcome| outcome.consumption.fit_memory_limit_bytes == Some(memory.per_fit_bytes)));
    println!(
        "execution_host_receipt={}",
        serde_json::json!({"case":"concurrency20","peak":20,"completed":outcomes.len(),"active_after":0,"fit_memory_allocation":memory,"all_fit_limits_read_back":true})
    );
}
