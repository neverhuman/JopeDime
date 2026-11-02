use super::*;

#[test]
#[ignore = "positive host qualification: requires an exclusive provisioned systemd service, never a synthetic cgroup tree"]
fn real_supervisor_resources_and_descendant_cleanup() {
    let unit = std::env::var("JOPE_EXECUTION_TEST_UNIT")
        .expect("required qualified systemd service; unavailable is failure");
    let group = PathBuf::from("/sys/fs/cgroup/system.slice").join(&unit);
    let case = std::env::var("JOPE_EXECUTION_TEST_CASE").unwrap_or_else(|_| "timeout".into());
    let mut experiment = fixture_experiment();
    if case == "concurrency20" {
        experiment.resources.workers = 20;
        experiment.resources.cpus = 20;
        experiment.resources.memory_bytes = 1024 * 1024 * 1024;
        experiment.resources.per_fit_seconds = 10;
    }
    if case == "graceful-sigterm" {
        experiment.resources.per_fit_seconds = 5;
    }
    if case == "original-cutoff" {
        experiment.resources.total_seconds = 8;
        experiment.resources.reporting_seconds = 3;
        experiment.resources.per_fit_seconds = 5;
    }
    let supervisor = Supervisor {
        unit,
        cgroup_root: group,
    };
    supervisor
        .verify(&experiment, now_ms().unwrap() + 110_000)
        .expect("real supervisor qualification");
    let scratch = scratch();
    let physical = fs::canonicalize(scratch.path()).unwrap();
    let mut pool =
        ExecutionPool::open(&experiment, &physical.join("journal"), supervisor.clone()).unwrap();
    pool.reserve()
        .expect("positive host fixture requires real reservation");
    assert!(
        pool.window().is_none(),
        "reservation must not start the clock"
    );
    assert!(!physical.join("journal/window.json").exists());
    if matches!(case.as_str(), "setup-hash-failure" | "setup-cancel") {
        let retained = scratch.keep();
        assert_eq!(fs::canonicalize(&retained).unwrap(), physical);
        qualify_setup_faults(&mut pool, &physical, &case);
        return;
    }
    if case == "concurrency20" {
        qualify_concurrency(&mut pool, &physical);
        return;
    }
    if case == "graceful-sigterm" || case == "original-cutoff" {
        // Retain the actual started/finished/window journal as host evidence.
        let retained = scratch.keep();
        assert_eq!(fs::canonicalize(&retained).unwrap(), physical);
        qualify_cancellation_and_resume(pool, &physical, &case);
        return;
    }
    let mut job = fixture_job('a', 'a');
    job.directory = physical.join("worker");
    let script = physical.join("fork.py");
    write_new(&script, b"import os,time\npid=os.fork()\nif pid==0:\n os.setsid()\n time.sleep(60)\nelse:\n open('descendant.pid','w').write(str(pid))\n time.sleep(60)\n").unwrap();
    if case == "controller-stall-success" {
        fs::write(
            &script,
            b"import time\ntime.sleep(0.1)\nopen('predictions.json','w').write('[]')\n",
        )
        .unwrap();
    }
    if case == "memory" {
        fs::write(
            &script,
            b"x=bytearray(256*1024*1024)\nopen('predictions.json','w').write('[]')\n",
        )
        .unwrap();
    }
    job.command.program.path =
        fs::canonicalize("/usr/bin/python3").expect("Python fixture unavailable");
    job.command.program.sha256 = hash_file(&job.command.program.path).unwrap();
    job.command.args = vec![script.to_string_lossy().into_owned()];
    job.command.inputs = vec![Artifact {
        sha256: hash_file(&script).unwrap(),
        path: script,
    }];
    let id = job.identity().unwrap();
    let started = Started {
        job,
        job_sha256: id.clone(),
        started_unix_ms: now_ms().unwrap(),
        fit_deadline_unix_ms: now_ms().unwrap() + if case == "memory" { 5000 } else { 1000 },
        cgroup: supervisor.cgroup_root.join(format!("fit-{id}")),
    };
    let cpu = parse_cpus(&control_text(&supervisor.cgroup_root, "cpuset.cpus.effective").unwrap())
        .unwrap()[0];
    let running = Running::start(
        started,
        cpu,
        if case == "memory" {
            32 * 1024 * 1024
        } else {
            experiment.resources.memory_bytes
        },
        if case == "memory" { 5 } else { 1 },
        physical.join("started.json"),
        physical.join("logs"),
        Arc::new(AtomicBool::new(false)),
    )
    .expect("isolated fixture launch");
    if case.starts_with("controller-stall") {
        // Simulate controller hashing without polling its fits at all.
        thread::sleep(Duration::from_millis(2500));
        assert!(running.is_done(), "watcher stalled with controller");
    }
    if let Some(receipt) = std::env::var_os("JOPE_EXECUTION_SIGKILL_ARMED") {
        let child_path = running.started.job.directory.join("descendant.pid");
        let ready_deadline = Instant::now() + Duration::from_millis(500);
        while !child_path.is_file() {
            assert!(
                Instant::now() < ready_deadline,
                "descendant fixture not armed"
            );
            thread::sleep(Duration::from_millis(5));
        }
        let pid: u32 = fs::read_to_string(&child_path).unwrap().parse().unwrap();
        immutable(Path::new(&receipt), &serde_json::json!({"unit": supervisor.unit, "cgroup": running.started.cgroup, "descendant_pid": pid})).unwrap();
        nix::sys::signal::kill(Pid::this(), Signal::SIGKILL).unwrap();
        panic!("SIGKILL unexpectedly returned");
    }
    while !running.is_done() {
        thread::sleep(Duration::from_millis(10));
    }
    let child_path = running.started.job.directory.join("descendant.pid");
    if case != "memory" && case != "controller-stall-success" {
        assert!(
            child_path.is_file(),
            "fixture actually forked before timeout"
        );
    }
    let group = running.started.cgroup.clone();
    let outcome = running.finish().unwrap();
    if case.starts_with("controller-stall") {
        assert!(
            outcome.consumption.wall_seconds.unwrap() < 1.25,
            "supervision time must exclude stalled controller/hash time"
        );
    }
    if case == "controller-stall-success" {
        assert_eq!(outcome.command_status, EvaluationStatus::Completed);
    } else if case == "memory" {
        assert_eq!(outcome.command_status, EvaluationStatus::Failed);
        assert!(named_counter(&group, "memory.events", "oom_kill").unwrap() > 0);
        assert!(outcome
            .reason
            .as_ref()
            .unwrap()
            .starts_with("memory_limit:"));
    } else {
        assert_eq!(outcome.command_status, EvaluationStatus::TimedOut);
    }
    assert_eq!(
        named_counter(&group, "cgroup.events", "populated").unwrap(),
        0
    );
    assert!(outcome.consumption.cpu_seconds.is_some());
    println!(
        "execution_host_receipt={}",
        serde_json::to_string(&outcome).unwrap()
    );
}
