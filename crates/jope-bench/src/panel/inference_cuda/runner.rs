use super::*;

pub fn run(args: RunArgs) -> Result<Value> {
    let p = read(&args.plan, &args.plan_sha256)?;
    p.binary.verify()?;
    ensure!(
        artifact(&std::env::current_exe()?)?.sha256 == p.binary.sha256,
        "binary identity mismatch"
    );
    let journal = custody(&args.journal)?;
    if let Some(unit) = args.supervisor_unit {
        for (name, expected) in &p.toolchain {
            expected.verify()?;
            let resolved = artifact(Path::new(command_text("which", &[name])?.trim()))?;
            ensure!(
                resolved.path == expected.path && resolved.sha256 == expected.sha256,
                "supervised tool identity changed: {name}"
            );
        }
        let mut pool = jope_pipeline::execution::ExecutionPool::open(
            &p.experiment,
            &journal,
            jope_pipeline::supervisor::for_unit(&unit)?,
        )?;
        pool.reserve()?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open("/run/lock/jope-inference-gpu.lock")?;
        lock.try_lock().context("GPU benchmark already reserved")?;
        let hardware = command_text(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
        )?;
        ensure!(
            hardware.lines().count() == 1
                && hardware.contains("RTX 3090")
                && hardware.contains("24576"),
            "requires one RTX3090 24GiB: {hardware}"
        );
        ensure!(
            command_text(
                "nvidia-smi",
                &["--query-compute-apps=pid", "--format=csv,noheader"]
            )?
            .trim()
            .is_empty(),
            "GPU already has compute consumers"
        );
        write(
            &journal.join("hardware.json"),
            &json!({"gpu":hardware,"rustc":command_text("rustc",&["--version"])?,"kernel":command_text("uname",&["-a"])?,"cpu":fs::read_to_string("/proc/cpuinfo")?,"source":p.experiment.source_sha,"binary":p.binary}),
        )?;
        for d in &p.inputs {
            d.raw.verify()?;
        }
        let window = pool.admit()?;
        let cutoff = record_timing_window(&journal, &p.experiment.methods_sha256, &window)?;
        let deadline_journal = journal.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(cutoff.saturating_sub(now())));
            let _ = write(
                &deadline_journal.join("deadline.json"),
                &json!({"compute_cutoff_ms":cutoff,"interrupted":true}),
            );
            std::process::exit(124);
        });
        let qualification_start = now();
        let qualification_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done = qualification_done.clone();
        let qjournal = journal.clone();
        let qdeadline = (window.started_unix_ms + 600_000).min(cutoff);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(qdeadline.saturating_sub(now())));
            if !done.load(std::sync::atomic::Ordering::SeqCst) {
                let _ = write(
                    &qjournal.join("qualification-timeout.json"),
                    &json!({"deadline_unix_ms":qdeadline,"status":"timed_out"}),
                );
                std::process::exit(124);
            }
        });
        let m = match load(&p) {
            Ok(m) => m,
            Err(e) => {
                write(
                    &journal.join("qualification.json"),
                    &json!({"schema":SCHEMA,"passed":false,"failure":format!("model loading: {e:#}"),"source_sha":p.experiment.source_sha,"plan_identity":p.experiment.methods_sha256,"binary_sha256":p.binary.sha256}),
                )?;
                return Ok(json!({"qualified":false,"publication_headline":false}));
            }
        };
        let qualification = if journal.join("qualification.json").exists() {
            serde_json::from_slice(&fs::read(journal.join("qualification.json"))?)?
        } else {
            let q = match qualify(&m, &p) {
                Ok(q) => q,
                Err(e) => {
                    json!({"schema":SCHEMA,"passed":false,"failure":format!("{e:#}"),"source_sha":p.experiment.source_sha,"plan_identity":p.experiment.methods_sha256,"binary_sha256":p.binary.sha256})
                }
            };
            write(&journal.join("qualification.json"), &q)?;
            q
        };
        qualification_done.store(true, std::sync::atomic::Ordering::SeqCst);
        if qualification["passed"] != true || now() - qualification_start > 600_000 {
            return Ok(json!({"qualified":false,"publication_headline":false}));
        }
        let direct_cutoff = phase(&journal, "direct", 1200, cutoff)?;
        for d in &p.inputs {
            for seed in 0..2 {
                if now() >= direct_cutoff {
                    break;
                }
                request(&m, d, seed, "warmup", &journal)?;
            }
        }
        for seed in 0..10 {
            for offset in 0..10 {
                if now() >= direct_cutoff {
                    break;
                }
                let d = &p.inputs[(offset + seed as usize) % 10];
                request(&m, d, seed, "warm", &journal)?;
            }
        }

        drop(m);
        for index in 0..10 {
            if now() >= direct_cutoff {
                break;
            }
            if journal.join(format!("cold-process-{index}.json")).exists() {
                continue;
            }
            let prior = journal.join(format!("cold-{}-0", p.inputs[index].id));
            if prior.exists() {
                write(
                    &journal.join(format!("cold-process-{index}.json")),
                    &json!({"dataset":p.inputs[index].id,"status":"interrupted_process_observation","elapsed_seconds":null}),
                )?;
                continue;
            }
            let authority = journal.join(format!(
                "cold-authority-{index}-{}.json",
                std::process::id()
            ));
            write(
                &authority,
                &json!({"parent_pid":std::process::id(),"plan_sha256":args.plan_sha256,"cutoff":direct_cutoff}),
            )?;
            let start = Instant::now();
            let status = Command::new(&p.binary.path)
                .env("JOPE_INFERENCE_COLD_AUTHORITY", &authority)
                .args(["benchmark", "inference-cold", "--plan"])
                .arg(&args.plan)
                .args(["--plan-sha256", &args.plan_sha256, "--journal"])
                .arg(&journal)
                .args(["--index", &index.to_string()])
                .status()?;
            let elapsed = start.elapsed().as_secs_f64();
            let directory = journal.join(format!("cold-{}-0", p.inputs[index].id));
            let receipt: Value = match fs::read(directory.join("receipt.json")) {
                Ok(bytes) => serde_json::from_slice(&bytes)?,
                Err(error) => {
                    json!({"schema":SCHEMA,"success":false,"dataset":p.inputs[index].id,"mode":"cold","seed":0,"failure":format!("fresh process exit {:?}, no request receipt: {error}",status.code())})
                }
            };

            write(
                &journal.join(format!("cold-process-{index}.json")),
                &json!({"dataset":p.inputs[index].id,"elapsed_seconds":elapsed,"exit_code":status.code(),"request":receipt}),
            )?;
        }
        let m = load(&p)?;
        let gp_cutoff = phase(&journal, "gp", 1200, cutoff)?;
        for d in p.inputs.iter().filter(|d| d.gp.is_some()) {
            if now() >= gp_cutoff {
                break;
            }
            full_gp(&m, d, &journal, gp_cutoff)?;
        }
        drop(m);
        let prediction_cutoff = phase(&journal, "prediction", 300, cutoff)?;
        // An error stops the stage (exit code 1), as an uncaught driver exception did.
        let exit_code = match super::super::prediction::run(&args.plan, &journal, prediction_cutoff)
        {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("prediction stage stopped: {error:#}");
                1
            }
        };
        write(
            &journal.join("prediction-stage.json"),
            &json!({"exit_code":exit_code,"cutoff":prediction_cutoff}),
        )?;
        if now() + 30_000 < cutoff {
            let m = load(&p)?;
            for d in &p.inputs {
                if now() >= cutoff {
                    break;
                }
                request(&m, d, 0, "profile", &journal)?;
            }
        }
        return Ok(json!({"qualified":true,"compute_completed":true}));
    }
    let arguments = vec![
        "benchmark".into(),
        "inference-run".into(),
        "--plan".into(),
        fs::canonicalize(&args.plan)?.display().to_string(),
        "--plan-sha256".into(),
        args.plan_sha256,
        "--journal".into(),
        journal.display().to_string(),
    ];
    let mut paths = Vec::new();
    for tool in p.toolchain.values() {
        tool.verify()?;
        paths.push(
            tool.path
                .parent()
                .context("tool parent")?
                .display()
                .to_string(),
        );
    }
    paths.push("/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/snap/bin".into());
    let launched = jope_pipeline::supervisor::launch_with_path(
        &p.experiment,
        &journal,
        &p.binary,
        &arguments,
        Some(&paths.join(":")),
    )?;
    write(&journal.join("launch.json"), &launched)?;
    ensure!(
        launched.exit_code == Some(0),
        "timing controller did not complete; retain launch.json and run inference-report"
    );
    Ok(json!({"launch":launched,"report_required":true}))
}
