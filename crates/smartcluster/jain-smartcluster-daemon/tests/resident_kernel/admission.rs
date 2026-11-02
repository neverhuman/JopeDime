use super::*;

#[test]
#[ignore = "native daemon launcher used only by the isolated kernel harness"]
fn resident_daemon_entry() -> Result<()> {
    let Some(executable) = std::env::var_os("RESIDENT_DAEMON_EXECUTABLE") else {
        eprintln!("SKIP: daemon launcher is invoked only by its harness");
        return Ok(());
    };
    let cgroup =
        PathBuf::from(std::env::var_os("RESIDENT_DAEMON_CGROUP").context("daemon cgroup")?);
    ensure!(
        cgroup.starts_with("/sys/fs/cgroup")
            && cgroup.file_name().is_some_and(|n| n == "supervisor"),
        "invalid daemon subgroup"
    );
    fs::write(cgroup.join("cgroup.procs"), std::process::id().to_string())?;
    let error = Command::new(executable)
        .arg("--config")
        .arg(std::env::var_os("RESIDENT_DAEMON_CONFIG").context("daemon config")?)
        .exec();
    Err(error.into())
}

#[test]
fn resident_fixture_specs_pass_core_admission_validation() -> Result<()> {
    for (mode, allowance) in [
        ("nested", 15000),
        ("ignore-eof", 1000),
        ("dependency", 1000),
        ("result-crash", 1000),
    ] {
        let spec = Harness::spec(
            Path::new("/tmp/resident-kernel-construction"),
            mode,
            allowance,
        )?;
        spec.validate()?;
        ensure!(
            spec.preemption.preemptible,
            "checkpoint-capable fixture must be preemptible"
        );
    }
    Ok(())
}

/// Invoked directly by scqd through this native test executable, never a shell.
#[test]
#[ignore = "native fixture entry, launched only by the isolated kernel harness"]
fn resident_fixture_entry() -> Result<()> {
    let Some(root) = std::env::var_os("RESIDENT_KERNEL_DIRECTORY") else {
        eprintln!("SKIP: fixture is invoked only by its harness");
        return Ok(());
    };
    let root = PathBuf::from(root);
    let mode = std::env::var("RESIDENT_KERNEL_MODE")?;
    if mode == "descendant" {
        let group = PathBuf::from(std::env::var("RESIDENT_KERNEL_NESTED")?);
        fs::write(group.join("cgroup.procs"), std::process::id().to_string())?;
        durable(&root.join("descendant-ready"), b"nested\n")?;
        let start = Instant::now();
        while !root.join("release-descendant").exists() && start.elapsed() < Duration::from_secs(60)
        {
            std::thread::sleep(Duration::from_millis(20));
        }
        return Ok(());
    }
    let attempt = FencedAttempt::new(
        JobId::from_str(&std::env::var("SCQ_JOB_ID")?)?,
        AttemptNumber::new(std::env::var("SCQ_ATTEMPT_NUMBER")?.parse()?)?,
        FenceToken::new(std::env::var("SCQ_ATTEMPT_FENCE")?.parse()?)?,
    );
    let number = std::env::var("SCQ_ATTEMPT_NUMBER")?;
    let fd: i32 = std::env::var("SCQ_CONTROL_FD")?.parse()?;
    // This process owns the inherited descriptor; a nested descendant must not retain it.
    fcntl(fd, FcntlArg::F_GETFD)?;
    let mut socket = Control(fd);
    fcntl(fd, FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC))?;
    let group = own_cgroup()?;
    if mode == "gpu-lease" && root.join("gpu-predecessor.json").is_file() {
        let predecessor: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("gpu-predecessor.json"))?)?;
        let old = PathBuf::from(
            predecessor["cgroup"]
                .as_str()
                .context("prior GPU allocation cgroup")?,
        );
        ensure!(old.parent() == group.parent(), "foreign predecessor cgroup");
        ensure!(
            !old.exists() || !populated(&old)?,
            "exclusive GPU successor started before predecessor cgroup drained"
        );
    }
    let mut previous = None;
    if number != "1" {
        let first: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("attempt-1.json"))?)?;
        let old = PathBuf::from(first["cgroup"].as_str().context("persisted first cgroup")?);
        ensure!(
            old.parent() == group.parent(),
            "prior attempt is outside this delegated job tree"
        );
        let current = serde_json::to_value(attempt)?;
        ensure!(
            first["attempt"]["job_id"] == current["job_id"]
                && first["attempt"]["fence"] != current["fence"],
            "prior attempt identity or fence mismatch"
        );
        match fs::read_to_string(old.join("cgroup.events")) {
            Ok(events) => ensure!(
                events.lines().any(|line| line == "populated 0"),
                "replacement started while old recursive cgroup is populated"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        previous = Some(first);
    }
    report(&mut socket, attempt, 1, WorkerReport::Ready)?;
    durable(
        &root.join(format!("attempt-{number}.json")),
        &serde_json::to_vec(
            &serde_json::json!({"attempt":attempt,"pid":std::process::id(),"cgroup":group,"previous_attempt":previous,"old_recursive_populated_at_start":false,"cuda_visible_devices":std::env::var("CUDA_VISIBLE_DEVICES").ok()}),
        )?,
    )?;
    if mode == "dependency" {
        report(
            &mut socket,
            attempt,
            2,
            WorkerReport::Failure {
                class: FailureClass::Dependency,
                message: "fixture missing immutable dependency".into(),
            },
        )?;
        // Exit immediately after write: the actor must drain reports before classifying completion.
        return Ok(());
    }
    if number == "1" && mode == "nested" {
        let nested = group.join("fixture-descendant");
        fs::create_dir(&nested)?;
        let _child = Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "resident_fixture_entry",
                "--ignored",
                "--nocapture",
            ])
            .env("RESIDENT_KERNEL_MODE", "descendant")
            .env("RESIDENT_KERNEL_NESTED", nested)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()?;
        // Keep Child alive without waiting: the test intentionally covers leader exit first.
    }
    let mut sequence = 2;
    loop {
        let mut length = [0; 4];
        if let Err(error) = socket.read_exact(&mut length) {
            ensure!(
                error.kind() == std::io::ErrorKind::UnexpectedEof,
                "control read failed: {error}"
            );
            if mode == "ignore-eof" {
                std::thread::sleep(Duration::from_secs(60));
                return Ok(());
            }
            if !root.join("eof-receipt").exists() {
                durable(&root.join("eof-receipt"), b"checkpoint-fsynced-on-eof\n")?;
            }
            return Ok(());
        }
        let length = u32::from_be_bytes(length) as usize;
        ensure!(length <= 65536, "oversized fixture control");
        let mut data = vec![0; length];
        socket.read_exact(&mut data)?;
        let frame: WorkerControlFrame = postcard::from_bytes(&data)?;
        ensure!(frame.attempt == attempt, "stale control fence");
        if let ControlMessage::Application { name, payload } = frame.control {
            ensure!(
                name == "fixture.compute" && payload == b"task-1",
                "unexpected fixture input"
            );
            let result = root.join("task-1.result");
            if !result.exists() {
                durable(&result, b"task-1:deterministic-result\n")?;
            }
            ensure!(
                fs::read(&result)? == b"task-1:deterministic-result\n",
                "conflicting durable result"
            );
            if mode == "result-crash" && number == "1" {
                // Abrupt process exit after fsync, without unwinding or a completion report.
                std::process::exit(23);
            }
            report(
                &mut socket,
                attempt,
                sequence,
                WorkerReport::Progress(StructuredProgress {
                    phase: "durable-result".into(),
                    completed: 1,
                    total: Some(1),
                    message: None,
                    fields: BTreeMap::from([("receipt".into(), result.display().to_string())]),
                }),
            )?;
            sequence += 1;
            if !root.join("result-reused").exists() {
                durable(&root.join("result-reused"), b"task-1\n")?;
            }
        }
    }
}
