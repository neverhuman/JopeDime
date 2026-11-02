use super::*;

pub fn probe(world: usize, output: &Path, timeout: u64, fail_rank: Option<usize>) -> Result<()> {
    crate::task_contract::require_qualified_gpu_experiment("probe-controller")?;
    ensure!(
        [1, 2, 4, 8].contains(&world) && timeout > 0,
        "probe needs 1, 2 or 8 ranks and a deadline"
    );
    ensure!(!output.exists(), "launch directory already exists");
    fs::create_dir_all(output)?;
    let executable = std::env::current_exe()?;
    let mut children = Vec::new();
    let start = Instant::now();
    let mut outcome: Result<()> = Ok(());
    for rank in 0..world {
        let log = fs::File::create(output.join(format!("rank-{rank}.log")))?;
        let mut cmd = Command::new(&executable);
        cmd.args(["nccl-probe", "--directory"])
            .arg(output)
            .arg("--world")
            .arg(world.to_string())
            .arg("--rank")
            .arg(rank.to_string())
            .env("OMP_NUM_THREADS", "2")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("RAYON_NUM_THREADS", "2")
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        bind_parent_death(&mut cmd);
        // SAFETY: the hook runs between fork and exec and only calls async-signal-safe setpgid;
        // io::Error::last_os_error reads errno without allocating.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        match cmd.spawn() {
            Ok(child) => children.push(child),
            Err(e) => {
                outcome = Err(e.into());
                break;
            }
        }
    }
    let mut injected = false;
    if outcome.is_ok() {
        loop {
            let mut complete = true;
            for (rank, child) in children.iter_mut().enumerate() {
                if !injected
                    && fail_rank == Some(rank)
                    && start.elapsed() > Duration::from_millis(500)
                {
                    let _ = child.kill();
                    injected = true;
                }
                match child.try_wait() {
                    Ok(Some(status)) => {
                        if !status.success() {
                            outcome = Err(anyhow::anyhow!("rank {rank} failed: {status}"));
                            break;
                        }
                    }
                    Ok(None) => complete = false,
                    Err(e) => {
                        outcome = Err(e.into());
                        break;
                    }
                }
            }
            if outcome.is_err() || complete {
                break;
            }
            if start.elapsed() > Duration::from_secs(timeout) {
                outcome = Err(anyhow::anyhow!("distributed probe deadline exceeded"));
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    if outcome.is_err() {
        for child in &mut children {
            // SAFETY: kill takes only integers and has no memory-safety preconditions; it targets
            // the process group each rank created with setpgid(0, 0); the result is ignored.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
        }
    }
    let mut hashes = std::collections::BTreeSet::new();
    if outcome.is_ok() {
        for rank in 0..world {
            let r: serde_json::Value = read_json(&output.join(format!("rank-{rank}.json")))?;
            ensure!(
                r["rank"] == rank && r["world"] == world && r["steps"] == 1,
                "rank receipt mismatch"
            );
            hashes.insert(r["parameter_sha256"].as_str().unwrap_or("").to_string());
        }
        ensure!(
            hashes.len() == 1 && !hashes.contains(""),
            "ranks disagree after reduction"
        );
    }
    write_json(
        &output.join("distributed.json"),
        &serde_json::json!({"schema":"parallel.nccl_probe.v1","world":world,"passed":outcome.is_ok(),"rank_failure_injected":fail_rank,"seconds":start.elapsed().as_secs_f64(),"failure":outcome.as_ref().err().map(|e|format!("{e:#}")),"rank_hashes":hashes,"coverage":"one synchronized FP32 reduction; full training and recovery qualification remains separate"}),
    )?;
    outcome
}

pub fn probe_accumulation(world: usize, microbatch: usize, global_batch: usize) -> Result<usize> {
    ensure!(
        [1, 2, 4, 8].contains(&world)
            && [64, 128, 256, 512, 1024].contains(&microbatch)
            && [2048, 4096, 8192].contains(&global_batch),
        "unsupported throughput probe topology"
    );
    let local_total = world * microbatch;
    ensure!(
        global_batch.is_multiple_of(local_total),
        "probe batch must divide the declared global batch"
    );
    Ok(global_batch / local_total)
}

pub fn training_probe(
    output: &Path,
    vocabulary: &Path,
    microbatch: usize,
    global_batch: usize,
    steps: usize,
    timeout: u64,
) -> Result<()> {
    crate::task_contract::require_qualified_gpu_experiment("training-probe-controller")?;
    probe_accumulation(8, microbatch, global_batch)?;
    let vocabulary_sha256 = file_hash(vocabulary)?;
    ensure!(
        steps > 5 && timeout > 0,
        "probe needs five warmup steps and measured steps"
    );
    ensure!(!output.exists(), "qualification directory exists");
    fs::create_dir_all(output)?;
    let executable = std::env::current_exe()?;
    let mut children = Ranks(Vec::new());
    let started = Instant::now();
    for rank in 0..8 {
        let log = fs::File::create(output.join(format!("rank-{rank}.log")))?;
        let cpu = if rank < 4 {
            rank * 2
        } else {
            62 + (rank - 4) * 2
        };
        let mut cmd = Command::new("taskset");
        cmd.args(["-c", &format!("{cpu},{}", cpu + 1)])
            .arg(&executable)
            .args(["training-probe-rank", "--directory"])
            .arg(output)
            .arg("--vocabulary")
            .arg(vocabulary)
            .arg("--rank")
            .arg(rank.to_string())
            .args(["--world", "8", "--microbatch"])
            .arg(microbatch.to_string())
            .arg("--global-batch")
            .arg(global_batch.to_string())
            .arg("--steps")
            .arg(steps.to_string())
            .env("OMP_NUM_THREADS", "2")
            .env("MKL_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        // SAFETY: the hook runs between fork and exec and makes only async-signal-safe calls
        // (setpgid, prctl, getppid); its errors are errno codes, so nothing allocates in the child.
        unsafe {
            let parent = libc::getpid();
            cmd.pre_exec(move || {
                if libc::setpgid(0, 0) != 0
                    || libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0
                {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    // The controller exited: refuse with an errno, which does not allocate
                    // (an io::Error carrying a message would box it in the forked child).
                    return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
                }
                Ok(())
            });
        }
        children.0.push(cmd.spawn()?);
    }
    let result = (|| -> Result<()> {
        loop {
            let mut complete = true;
            for (rank, child) in children.0.iter_mut().enumerate() {
                match child.try_wait()? {
                    Some(status) => {
                        ensure!(status.success(), "training rank {rank} failed: {status}")
                    }
                    None => complete = false,
                }
            }
            if complete {
                break;
            }
            ensure!(
                started.elapsed() < Duration::from_secs(timeout),
                "training qualification deadline exceeded"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
        let mut hashes = std::collections::BTreeSet::new();
        for rank in 0..8 {
            let r: serde_json::Value = read_json(&output.join(format!("complete-{rank}.json")))?;
            ensure!(
                r["steps"] == steps
                    && r["global_batch"] == global_batch
                    && r["microbatch"] == microbatch
                    && r["vocabulary_sha256"] == vocabulary_sha256,
                "rank probe specification mismatch"
            );
            hashes.insert(r["parameter_sha256"].as_str().unwrap_or("").to_string());
        }
        ensure!(
            hashes.len() == 1 && !hashes.contains(""),
            "training rank parameters diverged"
        );
        Ok(())
    })();
    if result.is_err() {
        for child in &mut children.0 {
            // SAFETY: kill takes only integers and has no memory-safety preconditions; it targets
            // the process group each rank created with setpgid(0, 0); the result is ignored.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
        }
    }
    write_json(
        &output.join("training-qualification.json"),
        &serde_json::json!({"passed":result.is_ok(),"world":8,"steps":steps,"warmup_steps":5,"microbatch":microbatch,"global_batch":global_batch,"vocabulary_sha256":vocabulary_sha256,"seconds":started.elapsed().as_secs_f64(),"failure":result.as_ref().err().map(|e|format!("{e:#}")),"eligible_checkpoint":false,"scope":"numerical architecture throughput; learning quality and real-corpus throughput require separate qualification"}),
    )?;
    result
}

/// Verify an original controller-scoped GPU task grant (`rank == None`) against a verified
/// funding admission and common task descriptor (`MR04` / `AUTH-GAP-01`).
pub fn verify_controller_gpu_grant(
    admission: &crate::eval_registry::service::VerifiedFundingAdmission,
    descriptor: &serde_json::Value,
    expected_capability: &str,
) -> Result<()> {
    crate::task_contract::verify_gpu_task_grant(admission, descriptor, None, expected_capability)
}

/// Verify a participant per-rank GPU task grant (`rank == Some(rank)`) against a verified
/// funding admission and common task descriptor (`MR04` / `AUTH-GAP-01`).
pub fn verify_participant_gpu_grant(
    admission: &crate::eval_registry::service::VerifiedFundingAdmission,
    descriptor: &serde_json::Value,
    rank: u32,
    expected_capability: &str,
) -> Result<()> {
    crate::task_contract::verify_gpu_task_grant(
        admission,
        descriptor,
        Some(rank),
        expected_capability,
    )
}
