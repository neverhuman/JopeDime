use crate::manifest::*;
use anyhow::{ensure, Context, Result};
use std::os::unix::process::CommandExt;
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use std::sync::atomic::{AtomicBool, Ordering};

struct Ranks(Vec<std::process::Child>);

pub(crate) fn bind_proposal_protocol(spec: &mut serde_json::Value) {
    spec["sampling_protocol"] = "pure-v4".into();
    spec["pure_prediction_protocol"] = crate::proposal_population::PURE_PROTOCOL.into();
    spec["gp_assembly_protocol"] = crate::proposal_population::ASSEMBLY_PROTOCOL.into();
    spec["proposal_generation_schema"] = "parallel.proposal_generation.v2".into();
}

/// Reject legacy receipts before a completed-output shortcut or snapshot mutation.
pub(crate) fn verify_proposal_protocol(output: &Path, require_complete: bool) -> Result<()> {
    let mut expected = serde_json::json!({});
    bind_proposal_protocol(&mut expected);
    let run = output.join("run.json");
    let complete = output.join("complete.json");
    ensure!(
        !(require_complete || complete.exists()) || run.is_file(),
        "proposal run identity absent: {}",
        output.display()
    );
    ensure!(
        !require_complete || complete.is_file(),
        "completed proposal receipt absent: {}",
        output.display()
    );
    for path in [&run, &complete] {
        if path.exists() {
            let saved: serde_json::Value = read_json(path)?;
            for (field, value) in expected.as_object().unwrap() {
                ensure!(
                    saved.get(field) == Some(value),
                    "proposal prediction protocol changed ({field}): {}; use a new output directory",
                    path.display()
                );
            }
        }
    }
    Ok(())
}

fn check_cancel(stop: Option<&AtomicBool>) -> Result<()> {
    ensure!(
        !stop.is_some_and(|flag| flag.load(Ordering::Relaxed)),
        "auxiliary training stopped by its controller"
    );
    crate::campaign::check_deadline()
}
pub(crate) fn gpu_lease(root: &Path, phase: &str) -> Result<fs::File> {
    gpu_lease_controlled(root, phase, None)
}
fn gpu_lease_controlled(root: &Path, phase: &str, stop: Option<&AtomicBool>) -> Result<fs::File> {
    use std::os::fd::AsRawFd;
    let lease = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(root.join("runs/encoding-service.lock"))?;
    let started = Instant::now();
    loop {
        check_cancel(stop)?;
        if unsafe { libc::flock(lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(lease);
        }
        ensure!(
            started.elapsed() < Duration::from_secs(3600),
            "GPU lease wait exceeded one hour"
        );
        write_json(
            &root.join("runs/gpu-wait.json"),
            &serde_json::json!({"phase":phase,"timestamp":crate::telemetry::now(),"seconds":started.elapsed().as_secs_f64(),"reason":"All eight GPUs are assigned to the current encoder/trainer block; waiting for its durable completion"}),
        )?;
        std::thread::sleep(Duration::from_secs(1));
    }
}
impl Drop for Ranks {
    fn drop(&mut self) {
        for child in &mut self.0 {
            if child.try_wait().ok().flatten().is_none() {
                unsafe {
                    libc::kill(-(child.id() as i32), libc::SIGKILL);
                }
                let _ = child.wait();
            }
        }
    }
}

pub fn propose(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    cuts: usize,
    cut_start: usize,
    shuffled: bool,
) -> Result<()> {
    crate::campaign::check_deadline()?;
    verify_proposal_protocol(output, false)?;
    fs::create_dir_all(output)?;
    let champions =
        crate::champions::freeze_snapshot(root, corpus, checkpoint, output, cuts, cut_start)?;
    let mut spec = serde_json::json!({"schema":"parallel.proposal_run.v1","corpus_sha256":file_hash(&corpus.join("index.json"))?,"checkpoint_sha256":file_hash(&checkpoint.join("state.json"))?,"cuts":cuts,"shuffled":shuffled,"world":8,"champion_snapshot_sha256":file_hash(&champions)?});
    bind_proposal_protocol(&mut spec);
    if cut_start != 0 {
        spec["cut_start"] = cut_start.into();
    }
    #[cfg(feature = "neural")]
    {
        spec["proposal_selection_id"] = crate::proposal_ranking::SELECTION_ID.into();
    }
    if output.join("run.json").exists() {
        ensure!(
            read_json::<serde_json::Value>(&output.join("run.json"))? == spec,
            "proposal resume specification changed"
        );
    } else {
        ensure!(
            fs::read_dir(output)?.next().is_none(),
            "unidentified proposal output exists"
        );
        write_json(&output.join("run.json"), &spec)?;
    }
    let _lease = gpu_lease(root, "neural proposal generation")?;
    let h: crate::telemetry::Hardware = read_json(&root.join("runs/hardware.json"))?;
    ensure!(
        crate::telemetry::now().saturating_sub(h.timestamp) <= 5
            && h.available_gb >= 128.0
            && h.disk_free_gb >= 50.0,
        "proposal admission headroom unavailable"
    );
    let executable = std::env::current_exe()?;
    let mut children = Ranks(Vec::new());
    let begun = Instant::now();
    for rank in 0..8 {
        crate::campaign::check_deadline()?;
        let cpu = if rank < 4 {
            rank * 2
        } else {
            62 + (rank - 4) * 2
        };
        let log = fs::File::create(output.join(format!("rank-{rank}.log")))?;
        let mut cmd = Command::new("taskset");
        cmd.args(["-c", &format!("{cpu},{}", cpu + 1)])
            .env("JOPE_CHAMPION_SNAPSHOT", &champions)
            .arg(&executable)
            .arg("--root")
            .arg(root)
            .args(["propose-rank", "--corpus"])
            .arg(corpus)
            .arg("--checkpoint")
            .arg(checkpoint)
            .arg("--output")
            .arg(output)
            .arg("--rank")
            .arg(rank.to_string())
            .args(["--world", "8", "--cuts"])
            .arg(cuts.to_string())
            .arg("--cut-start")
            .arg(cut_start.to_string())
            .env("OMP_NUM_THREADS", "2")
            .env("MKL_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        if shuffled {
            cmd.arg("--shuffled");
        }
        unsafe {
            cmd.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        children.0.push(cmd.spawn()?);
    }
    loop {
        crate::campaign::check_deadline()?;
        let mut complete = true;
        let mut ranks = Vec::new();
        for (rank, child) in children.0.iter_mut().enumerate() {
            let r: serde_json::Value =
                read_json(&output.join(format!("rank-{rank}.json"))).unwrap_or_default();
            match child.try_wait()? {
                Some(s) => ensure!(s.success(), "proposal rank {rank} failed: {s}"),
                None => {
                    complete = false;
                    let age = r["timestamp"]
                        .as_u64()
                        .map(|t| crate::telemetry::now().saturating_sub(t))
                        .unwrap_or(begun.elapsed().as_secs());
                    ensure!(age < 600, "proposal rank {rank} heartbeat timeout");
                }
            }
            ranks.push(r);
        }
        write_json(
            &root.join("runs/encoding-service.json"),
            &serde_json::json!({"phase":if complete{"completed"}else{"neural program proposals"},"timestamp":crate::telemetry::now(),"persistent":false,"world":8,"ranks":ranks,"cpu_budget":16,"output":output,"seconds":begun.elapsed().as_secs_f64()}),
        )?;
        if complete {
            break;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    let mut combined = crate::service::GpJobs {
        schema: "parallel.gp_jobs.v1".into(),
        qualification: true,
        jobs: Vec::new(),
        vocabulary: Some(
            corpus
                .join("vocabulary.json")
                .strip_prefix(root)?
                .to_path_buf(),
        ),
    };
    for rank in 0..8 {
        let jobs: crate::service::GpJobs = read_json(&output.join(format!("jobs-{rank}.json")))?;
        combined.jobs.extend(jobs.jobs);
    }
    combined
        .jobs
        .sort_by(|a, b| a.prepared.cmp(&b.prepared).then(a.cut.cmp(&b.cut)));
    write_json(&output.join("jobs.json"), &combined)?;
    let mut complete = serde_json::json!({"jobs":combined.jobs.len(),"seconds":begun.elapsed().as_secs_f64(),"shuffled_context":shuffled,"checkpoint_sha256":file_hash(&checkpoint.join("state.json"))?,"world":8,"final_tests_opened":false});
    bind_proposal_protocol(&mut complete);
    write_json(&output.join("complete.json"), &complete)?;
    Ok(())
}

/// Qualification wrappers may select the same immutable rank runtime as production.
pub(crate) fn training_executable(root: &Path) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("JOPE_TRAIN_EXECUTABLE") {
        let path = PathBuf::from(path);
        ensure!(path.is_absolute(), "JOPE_TRAIN_EXECUTABLE must be absolute");
        return crate::manifest::resolve(
            root,
            path.strip_prefix(root)
                .context("training executable must be inside workspace")?,
        );
    }
    Ok(std::env::current_exe()?)
}

/// Own all GPU ranks under a lease shared with encoding services. A failed rank
/// restarts the whole communicator from the last atomically published checkpoint.
pub fn train(
    root: &Path,
    corpus: &Path,
    output: &Path,
    world: usize,
    micro: usize,
    steps: usize,
    checkpoint_steps: usize,
    resume: bool,
    fail_rank: Option<usize>,
    initialize: Option<&Path>,
) -> Result<()> {
    train_controlled(
        root,
        corpus,
        output,
        world,
        micro,
        steps,
        checkpoint_steps,
        resume,
        fail_rank,
        initialize,
        None,
    )
}

pub(crate) fn train_controlled(
    root: &Path,
    corpus: &Path,
    output: &Path,
    world: usize,
    micro: usize,
    steps: usize,
    checkpoint_steps: usize,
    resume: bool,
    fail_rank: Option<usize>,
    initialize: Option<&Path>,
    stop: Option<&AtomicBool>,
) -> Result<()> {
    ensure!(
        [1, 2, 8].contains(&world)
            && [64, 128, 256].contains(&micro)
            && 2048 % (world * micro) == 0
            && steps > 0
            && checkpoint_steps > 0,
        "unsupported training topology"
    );
    ensure!(
        resume || !output.exists() || fs::read_dir(output)?.next().is_none(),
        "run exists; use explicit resume"
    );
    let resume = resume && output.join("run.json").is_file();
    ensure!(
        resume || !output.exists() || fs::read_dir(output)?.next().is_none(),
        "nonempty run has no launch identity"
    );
    check_cancel(stop)?;
    let _lease = gpu_lease_controlled(root, "neural policy training", stop)?;
    let h: crate::telemetry::Hardware = read_json(&root.join("runs/hardware.json"))?;
    ensure!(
        crate::telemetry::now().saturating_sub(h.timestamp) <= 5
            && h.available_gb >= 128.0
            && h.disk_free_gb >= 50.0,
        "training admission requires fresh telemetry, 128 GiB RAM and 50 GiB disk headroom"
    );
    let executable = training_executable(root)?;
    fs::create_dir_all(output)?;
    let corpus_hash = file_hash(&corpus.join("index.json"))?;
    let spec = serde_json::json!({"schema":"parallel.training_run.v1","corpus_sha256":corpus_hash,"world":world,"microbatch":micro,"global_batch":2048,"qualification":true,"executable_sha256":file_hash(&executable)?,"source_families_sampled_evenly":true,"evaluation_bias":"adaptive_cut_heldout_teacher"});
    let initialization_sha = initialize
        .map(|p| file_hash(&p.join("state.json")))
        .transpose()?;
    let mut spec = spec;
    #[cfg(feature = "neural")]
    {
        spec["sampler_id"] = crate::training::SAMPLER_ID.into();
    }
    spec["initialization_sha256"] = serde_json::json!(initialization_sha);
    if resume {
        let old: serde_json::Value = read_json(&output.join("run.json"))?;
        ensure!(
            old["corpus_sha256"] == corpus_hash
                && old["world"] == world
                && old["microbatch"] == micro
                && old["sampler_id"] == spec["sampler_id"]
                && old["executable_sha256"] == spec["executable_sha256"]
                && (initialize.is_none()
                    || old["initialization_sha256"] == serde_json::json!(initialization_sha)),
            "resume run identity mismatch"
        );
    } else {
        write_json(&output.join("run.json"), &spec)?;
    }
    let mut injected = false;
    let mut last_error = String::new();
    for attempt in 0..3 {
        check_cancel(stop)?;
        crate::checkpoint_recovery::reconcile(root, output)?;
        let launch = output.join(format!(
            "launch-{}-{}-{attempt}",
            crate::telemetry::now(),
            std::process::id()
        ));
        fs::create_dir_all(&launch)?;
        let begun = Instant::now();
        let mut children = Ranks(Vec::new());
        let should_resume = output.join("latest.json").exists();
        for rank in 0..world {
            check_cancel(stop)?;
            let cpu = if rank < 4 {
                rank * 2
            } else {
                62 + (rank - 4) * 2
            };
            let log = fs::File::create(launch.join(format!("rank-{rank}.log")))?;
            let mut cmd = Command::new("taskset");
            cmd.args(["-c", &format!("{cpu},{}", cpu + 1)])
                .arg(&executable)
                .arg("--root")
                .arg(root)
                .args(["train-rank", "--corpus"])
                .arg(corpus)
                .arg("--output")
                .arg(output)
                .arg("--launch")
                .arg(&launch)
                .arg("--rank")
                .arg(rank.to_string())
                .arg("--world")
                .arg(world.to_string())
                .arg("--microbatch")
                .arg(micro.to_string())
                .arg("--steps")
                .arg(steps.to_string())
                .arg("--checkpoint-steps")
                .arg(checkpoint_steps.to_string())
                .env("OMP_NUM_THREADS", "2")
                .env("RAYON_NUM_THREADS", "2")
                .env("MKL_NUM_THREADS", "1")
                .env("OPENBLAS_NUM_THREADS", "1")
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            if should_resume {
                cmd.arg("--resume");
            }
            if !should_resume {
                if let Some(p) = initialize {
                    let relative = if p.is_absolute() {
                        p.strip_prefix(root)?
                    } else {
                        p
                    };
                    cmd.arg("--initialize").arg(relative);
                }
            }
            unsafe {
                cmd.pre_exec(|| {
                    if libc::setpgid(0, 0) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            children.0.push(cmd.spawn()?);
        }
        let result = (|| -> Result<()> {
            loop {
                check_cancel(stop)?;
                let mut complete = true;
                for (rank, child) in children.0.iter_mut().enumerate() {
                    let receipt: serde_json::Value =
                        read_json(&launch.join(format!("rank-{rank}.json"))).unwrap_or_default();
                    if !injected
                        && fail_rank == Some(rank)
                        && output.join("latest.json").exists()
                        && receipt["step"].as_u64().unwrap_or(0) >= checkpoint_steps as u64
                    {
                        child.kill()?;
                        injected = true;
                    }
                    match child.try_wait()? {
                        Some(s) => ensure!(s.success(), "rank {rank} failed: {s}"),
                        None => {
                            complete = false;
                            let age = receipt["timestamp"]
                                .as_u64()
                                .map(|t| crate::telemetry::now().saturating_sub(t))
                                .unwrap_or(begun.elapsed().as_secs());
                            ensure!(age < 600, "rank {rank} heartbeat timed out");
                        }
                    }
                }
                if complete {
                    break;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            let mut hashes = std::collections::BTreeSet::new();
            for rank in 0..world {
                let r: serde_json::Value =
                    read_json(&launch.join(format!("complete-{rank}.json")))?;
                ensure!(
                    r["steps"] == steps && r["corpus_sha256"] == corpus_hash,
                    "rank completion identity mismatch"
                );
                hashes.insert(r["parameter_sha256"].as_str().unwrap_or("").to_string());
            }
            ensure!(
                hashes.len() == 1 && !hashes.contains(""),
                "trained parameters differ across ranks"
            );
            Ok(())
        })();
        drop(children);
        write_json(
            &launch.join("controller.json"),
            &serde_json::json!({"attempt":attempt,"passed":result.is_ok(),"failure":result.as_ref().err().map(|e|format!("{e:#}")),"rank_failure_injected":injected,"resumed":should_resume,"seconds":begun.elapsed().as_secs_f64()}),
        )?;
        match result {
            Ok(()) => {
                write_json(
                    &output.join("complete.json"),
                    &serde_json::json!({"schema":"parallel.training_complete.v1","steps":steps,"world":world,"microbatch":micro,"attempts":attempt+1,"rank_failure_injected":injected,"corpus_sha256":corpus_hash,"latest":read_json::<serde_json::Value>(&output.join("latest.json"))?,"qualification":true,"final_tests_opened":false}),
                )?;
                let mut state: serde_json::Value =
                    read_json(&root.join("runs/neural-service.json"))?;
                state["phase"] = "completed real-corpus training block".into();
                state["timestamp"] = crate::telemetry::now().into();
                write_json(&root.join("runs/neural-service.json"), &state)?;
                return Ok(());
            }
            Err(e) => last_error = format!("{e:#}"),
        }
    }
    anyhow::bail!("training exhausted three coordinated attempts: {last_error}")
}

/// A failed rank or expired deadline kills/reaps every rank. Launch directories
/// are immutable, so a stale rendezvous can never join a new communicator.
pub fn probe(world: usize, output: &Path, timeout: u64, fail_rank: Option<usize>) -> Result<()> {
    ensure!(
        [1, 2, 8].contains(&world) && timeout > 0,
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
    ensure!([1,2,8].contains(&world) && [64,128,256,512,1024].contains(&microbatch)
        && [2048,4096,8192].contains(&global_batch), "unsupported throughput probe topology");
    let local_total = world * microbatch;
    ensure!(global_batch % local_total == 0, "probe batch must divide the declared global batch");
    Ok(global_batch / local_total)
}

pub fn training_probe(output: &Path, vocabulary: &Path, microbatch: usize, global_batch: usize, steps: usize, timeout: u64) -> Result<()> {
    probe_accumulation(8, microbatch, global_batch)?;
    let vocabulary_sha256 = file_hash(vocabulary)?;
    ensure!(steps > 5 && timeout > 0, "probe needs five warmup steps and measured steps");
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
        unsafe {
            let parent = libc::getpid();
            cmd.pre_exec(move || {
                if libc::setpgid(0, 0) != 0 || libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent { return Err(std::io::Error::other("probe controller exited")); }
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
            ensure!(r["steps"] == steps && r["global_batch"] == global_batch && r["microbatch"] == microbatch && r["vocabulary_sha256"] == vocabulary_sha256, "rank probe specification mismatch");
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

#[cfg(test)]
mod proposal_protocol_tests {
    use super::*;

    #[test]
    fn larger_probe_batches_require_an_explicit_global_batch() {
        assert_eq!(probe_accumulation(8,256,2048).unwrap(),1);
        assert_eq!(probe_accumulation(8,128,2048).unwrap(),2);
        assert!(probe_accumulation(8,512,2048).is_err());
        assert_eq!(probe_accumulation(8,512,4096).unwrap(),1);
        assert_eq!(probe_accumulation(8,1024,8192).unwrap(),1);
        for (world,micro,global) in [(0,256,2048),(8,0,2048),(8,usize::MAX,2048),(8,256,0)] {
            assert!(probe_accumulation(world,micro,global).is_err());
        }
    }

    fn current_receipts(output: &Path) -> serde_json::Value {
        let mut spec = serde_json::json!({"checkpoint_sha256":"frozen"});
        bind_proposal_protocol(&mut spec);
        for name in ["run.json", "complete.json"] {
            write_json(&output.join(name), &spec).unwrap();
        }
        spec
    }

    #[test]
    fn proposal_protocol_accepts_current_and_rejects_each_missing_binding() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path();
        let current = current_receipts(output);
        verify_proposal_protocol(output, true).unwrap();
        for name in ["run.json", "complete.json"] {
            for field in [
                "sampling_protocol",
                "pure_prediction_protocol",
                "gp_assembly_protocol",
                "proposal_generation_schema",
            ] {
                let mut legacy = current.clone();
                legacy.as_object_mut().unwrap().remove(field);
                let path = output.join(name);
                write_json(&path, &legacy).unwrap();
                let before = fs::read(&path).unwrap();
                assert!(verify_proposal_protocol(output, false).is_err());
                assert_eq!(fs::read(&path).unwrap(), before);
                write_json(&path, &current).unwrap();
            }
        }
    }

    #[test]
    fn proposal_protocol_rejects_legacy_completed_run_before_snapshot_or_gpu_work() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let output = root.join("legacy");
        write_json(
            &output.join("run.json"),
            &serde_json::json!({"sampling_protocol":"shared-incumbents-and-retained-v3"}),
        )
        .unwrap();
        write_json(
            &output.join("complete.json"),
            &serde_json::json!({"world":8}),
        )
        .unwrap();
        let error = propose(
            root,
            &root.join("absent-corpus"),
            &root.join("absent-checkpoint"),
            &output,
            2,
            0,
            false,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("proposal prediction protocol changed"));
        assert!(!root.join("champion-snapshot.json").exists());
        assert!(!root.join("runs/encoding-service.lock").exists());
    }

    #[test]
    fn proposal_protocol_rejects_unbound_complete_receipt() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path();
        current_receipts(output);
        fs::remove_file(output.join("run.json")).unwrap();
        assert!(verify_proposal_protocol(output, false)
            .unwrap_err()
            .to_string()
            .contains("proposal run identity absent"));
    }
}

#[cfg(test)]
mod campaign_deadline_tests {
    use super::*;

    #[test]
    fn auxiliary_stop_reaps_only_owned_rank() {
        let mut command = Command::new("/bin/sleep");
        command.arg("30");
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn().unwrap();
        let pid = child.id() as i32;
        let signal = AtomicBool::new(false);
        let result = (|| -> Result<()> {
            let _owned = Ranks(vec![child]);
            check_cancel(Some(&signal))?;
            signal.store(true, Ordering::Relaxed);
            check_cancel(Some(&signal))
        })();
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("auxiliary training stopped"));
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }

    #[test]
    fn expired_active_wait_reaps_owned_child_and_preserves_saved_work() {
        let saved = tempfile::tempdir().unwrap();
        let checkpoint = saved.path().join("checkpoint.json");
        fs::write(&checkpoint, b"durable checkpoint").unwrap();
        let mut command = Command::new("/bin/sleep");
        command.arg("30");
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn().unwrap();
        let pid = child.id() as i32;
        let result = (|| -> Result<()> {
            let _owned = Ranks(vec![child]);
            // Expire after the rank is alive, as happens during a polling loop.
            let _deadline = crate::campaign::DeadlineGuard::new(crate::telemetry::now() + 1);
            loop {
                crate::campaign::check_deadline()?;
                std::thread::sleep(Duration::from_millis(10));
            }
        })();
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("fixed campaign deadline"));
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
        assert_eq!(fs::read(checkpoint).unwrap(), b"durable checkpoint");
        crate::campaign::check_deadline().unwrap();
    }
}
