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

/// Rank clients must die with their controller so resident workers observe the
/// disconnect and release transferred leases even after SIGKILL.
fn bind_parent_death(command: &mut Command) {
    let parent = std::process::id();
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() as u32 != parent {
                return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
}

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
    #[cfg(feature = "neural")]
    if let Ok(fd) = std::env::var("JOPE_GPU_WORKER_LEASE_FD") {
        use std::os::{fd::FromRawFd, unix::fs::MetadataExt};
        ensure!(
            std::env::var_os("JOPE_GPU_WORKER_ACTIVE").is_some(),
            "GPU lease bypass outside resident worker"
        );
        let copied = unsafe { libc::dup(fd.parse()?) };
        ensure!(copied >= 0, "resident task lost its GPU lease");
        let lease = unsafe { fs::File::from_raw_fd(copied) };
        let actual = lease.metadata()?;
        let expected = root.join("runs/encoding-service.lock").metadata()?;
        ensure!(
            actual.dev() == expected.dev()
                && actual.ino() == expected.ino()
                && unsafe { libc::flock(lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "resident GPU lease identity changed"
        );
        return Ok(lease);
    }
    let lease = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(root.join("runs/encoding-service.lock"))?;
    let started = Instant::now();
    loop {
        check_cancel(stop)?;
        if std::env::var_os("JOPE_GPU_TASK_DEADLINE_MS").is_some() {
            crate::task_contract::gpu_task_deadline_ms()?;
        }
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
    propose_with_generation(
        root, corpus, checkpoint, output, cuts, cut_start, shuffled, None,
    )
}

pub fn propose_with_generation(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    cuts: usize,
    cut_start: usize,
    shuffled: bool,
    generation: Option<&Path>,
) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("proposal-controller")?;
    crate::campaign::check_deadline()?;
    let task_deadline = crate::task_contract::gpu_task_deadline_ms()?;
    verify_proposal_protocol(output, false)?;
    fs::create_dir_all(output)?;
    if let Some(path) = generation {
        let manifest: serde_json::Value = read_json(path)?;
        if let Some(protocol) = manifest["coverage_selection_protocol"].as_str() {
            ensure!(
                protocol == crate::service::coverage::PROTOCOL,
                "unsupported coverage selection protocol"
            );
            let owner = output
                .parent()
                .context("coverage generation parent absent")?;
            let selection_path = owner.join("coverage-selection.json");
            let index = crate::teaching_store::read_index(root, corpus)?;
            let selection = crate::service::coverage::Ledger::open(root)?.freeze_population(
                root,
                corpus,
                owner,
                &index.contexts,
                cuts,
                crate::telemetry::now(),
            )?;
            if selection_path.exists() {
                ensure!(
                    read_json::<serde_json::Value>(&selection_path)? == selection,
                    "frozen coverage selection changed"
                );
            } else {
                write_json(&selection_path, &selection)?;
            }
            crate::service::coverage::selected_indices(&selection_path, corpus, &index.contexts)?;
        }
    }
    let champions =
        crate::champions::freeze_snapshot(root, corpus, checkpoint, output, cuts, cut_start)?;
    let mut spec = serde_json::json!({"schema":"parallel.proposal_run.v1","corpus_sha256":file_hash(&corpus.join("index.json"))?,"checkpoint_sha256":file_hash(&checkpoint.join("state.json"))?,"cuts":cuts,"shuffled":shuffled,"world":8,"champion_snapshot_sha256":file_hash(&champions)?});
    spec["deadline_unix_ms"] = serde_json::json!(task_deadline);
    spec["executable_sha256"] = serde_json::json!(file_hash(&std::env::current_exe()?)?);
    let selection_path = output
        .parent()
        .context("proposal parent absent")?
        .join("coverage-selection.json");
    if selection_path.exists() {
        spec["coverage_selection_sha256"] = file_hash(&selection_path)?.into();
        spec["coverage_selection_protocol"] = crate::service::coverage::PROTOCOL.into();
    }
    if let Some(path) = generation {
        let manifest: serde_json::Value = read_json(path)?;
        ensure!(
            manifest["checkpoint_sha256"] == spec["checkpoint_sha256"],
            "generation parent checkpoint changed"
        );
        let digest = file_hash(path)?;
        let bytes = hex::decode(&digest)?;
        spec["proposal_seed"] = u64::from_le_bytes(bytes[..8].try_into().unwrap()).into();
        spec["generation_manifest_sha256"] = digest.into();
    }
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
        #[cfg(feature = "neural")]
        crate::gpu_worker::inherit_lease(&mut cmd, &_lease)?;
        cmd.env("JOPE_GPU_TASK_DEADLINE_MS", task_deadline.to_string());
        bind_parent_death(&mut cmd);
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
        None,
    )
}

pub fn train_with_policy(
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
    policy: Option<&Path>,
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
        policy,
    )
}

/// A run owns one requested increment. Reopening it resumes that same window;
/// an old absolute checkpoint step can never satisfy a new run's increment.
fn training_step_window(initial_step: u64, increment: usize) -> Result<serde_json::Value> {
    ensure!(increment > 0, "training increment must be positive");
    let target = initial_step
        .checked_add(increment as u64)
        .context("training step overflow")?;
    Ok(
        serde_json::json!({"schema":"parallel.training_step_window.v1",
        "initial_optimizer_step":initial_step,"requested_step_increment":increment,
        "target_optimizer_step":target}),
    )
}

fn verify_training_rank(
    receipt: &serde_json::Value,
    window: &serde_json::Value,
    rank: usize,
    world: usize,
    corpus_hash: &str,
) -> Result<()> {
    ensure!(
        receipt["rank"] == rank
            && receipt["world"] == world
            && receipt["steps"] == window["target_optimizer_step"]
            && receipt["local_steps"] == window["requested_step_increment"]
            && receipt["corpus_sha256"] == corpus_hash,
        "rank completion step window or identity mismatch"
    );
    let initial = receipt["initial_optimizer_step"]
        .as_u64()
        .context("rank initial step absent")?;
    let executed = receipt["executed_steps"]
        .as_u64()
        .context("rank executed steps absent")?;
    let target = window["target_optimizer_step"]
        .as_u64()
        .context("target step absent")?;
    ensure!(
        initial
            >= window["initial_optimizer_step"]
                .as_u64()
                .context("initial step absent")?
            && initial.checked_add(executed) == Some(target),
        "rank execution exposure mismatch"
    );
    Ok(())
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
    policy: Option<&Path>,
) -> Result<()> {
    // Input validation below is read-only: it creates no GPU context, lease,
    // lock, output directory or receipt. It precedes the dispatch hold so an
    // invalid request is refused for its own reason, and the hold still
    // refuses every valid request before any resource admission.
    ensure!(
        [1, 2, 4, 8].contains(&world)
            && [64, 128, 256].contains(&micro)
            && 2048 % (world * micro) == 0
            && steps > 0
            && checkpoint_steps > 0,
        "unsupported training topology"
    );
    let policy = policy
        .map(|p| -> Result<PathBuf> {
            resolve(
                root,
                if p.is_absolute() {
                    p.strip_prefix(root)?
                } else {
                    p
                },
            )
        })
        .transpose()?;
    let policy_sha = policy.as_deref().map(file_hash).transpose()?;
    ensure!(
        !(resume && initialize.is_some()),
        "resume and initialize are mutually exclusive"
    );
    if let Some(path) = &policy {
        let p: serde_json::Value = read_json(path)?;
        let index: serde_json::Value = read_json(&corpus.join("index.json"))?;
        if p["schema"] == crate::checkpoint_recovery::CONTINUATION_SCHEMA {
            crate::checkpoint_recovery::verify_continuation(
                root,
                &p,
                &file_hash(&corpus.join("index.json"))?,
                index["vocabulary_sha256"]
                    .as_str()
                    .context("continuation vocabulary absent")?,
                world,
                micro,
                2048,
                steps,
            )?;
            if !resume {
                let parent =
                    initialize.context("continuation requires its original parent checkpoint")?;
                let parent = resolve(
                    root,
                    if parent.is_absolute() {
                        parent.strip_prefix(root)?
                    } else {
                        parent
                    },
                )?;
                ensure!(
                    file_hash(&parent.join("state.json"))? == p["parent_checkpoint_sha256"],
                    "continuation initialization differs from recovered parent"
                );
            }
        }
        if let Some(p) = crate::checkpoint_recovery::continuation_replay(root, &p)? {
            ensure!(
                p["schema"] == "parallel.replay_policy.v2"
                    && p["eligible"] == true
                    && p["final_tests_opened"] == false
                    && p["corpus_sha256"] == file_hash(&corpus.join("index.json"))?
                    && index["label_policy"] == "strict_midrank_v4"
                    && index["admission_policy"].is_object(),
                "training policy requires a matching strictly admitted corpus"
            );
        }
    }
    ensure!(
        resume || !output.exists() || fs::read_dir(output)?.next().is_none(),
        "run exists; use explicit resume"
    );
    let resume = resume && output.join("run.json").is_file();
    ensure!(
        resume || !output.exists() || fs::read_dir(output)?.next().is_none(),
        "nonempty run has no launch identity"
    );
    let step_window = if resume {
        let previous: serde_json::Value = read_json(&output.join("run.json"))?;
        let window = previous["step_window"].clone();
        let initial = window["initial_optimizer_step"]
            .as_u64()
            .context("legacy run has no increment identity; explicit reconciliation required")?;
        ensure!(
            window == training_step_window(initial, steps)?,
            "resume requested increment changed"
        );
        ensure!(!output.join("complete.json").exists(),
            "training run already completed; retain original completion, use a new funded task for new steps");
        window
    } else {
        let carry = policy
            .as_ref()
            .map(|p| read_json::<serde_json::Value>(p))
            .transpose()?
            .is_some_and(|p| p["optimizer"] == "carry");
        let initial = if carry {
            let checkpoint =
                initialize.context("optimizer carry requires an explicit parent checkpoint")?;
            let checkpoint = if checkpoint.is_absolute() {
                checkpoint.to_path_buf()
            } else {
                root.join(checkpoint)
            };
            read_json::<serde_json::Value>(&checkpoint.join("state.json"))?["step"]
                .as_u64()
                .context("parent optimizer step absent")?
        } else {
            0
        };
        training_step_window(initial, steps)?
    };
    crate::task_contract::require_original_gpu_grant("training-controller")?;
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
    let inherited_deadline = crate::task_contract::gpu_task_deadline_ms()?;
    let task_deadline = if resume {
        read_json::<serde_json::Value>(&output.join("run.json"))?["deadline_unix_ms"]
            .as_u64()
            .context("legacy training run lacks its original absolute deadline")?
    } else {
        inherited_deadline
    };
    ensure!(
        task_deadline == inherited_deadline,
        "resume task deadline differs from its original run"
    );
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    ensure!(
        task_deadline as u128 > now_ms,
        "original training task deadline elapsed"
    );
    spec["deadline_unix_ms"] = serde_json::json!(task_deadline);
    spec["execution_partition"] = crate::task_contract::training_partition(world, micro, 2048)?;
    #[cfg(feature = "neural")]
    {
        spec["sampler_id"] = crate::training::SAMPLER_ID.into();
    }
    spec["initialization_sha256"] = serde_json::json!(initialization_sha);
    spec["replay_policy_sha256"] = serde_json::json!(policy_sha);
    spec["training_policy_schema"] = policy
        .as_ref()
        .map(|path| read_json::<serde_json::Value>(path).map(|value| value["schema"].clone()))
        .transpose()?
        .unwrap_or(serde_json::Value::Null);
    spec["step_window"] = step_window.clone();
    // W6 follow-up: the controller declares the run's lineage mode in
    // run.json and on every rank request; each rank validates it. A resumed
    // run whose run.json predates the field keeps its implicit meaning.
    let lineage_mode = if resume {
        let old: serde_json::Value = read_json(&output.join("run.json"))?;
        ensure!(
            old["corpus_sha256"] == corpus_hash
                && old["world"] == world
                && old["microbatch"] == micro
                && old["execution_partition"] == spec["execution_partition"]
                && old["sampler_id"] == spec["sampler_id"]
                && old["executable_sha256"] == spec["executable_sha256"]
                && old["replay_policy_sha256"] == spec["replay_policy_sha256"]
                && (initialize.is_none()
                    || old["initialization_sha256"] == serde_json::json!(initialization_sha)),
            "resume run identity mismatch"
        );
        old.get("continuation_mode")
            .filter(|mode| !mode.is_null())
            .map(|mode| {
                crate::task_contract::ContinuationMode::parse(
                    mode.as_str().context("recorded continuation mode")?,
                )
            })
            .transpose()?
    } else {
        let mode = crate::task_contract::classify_continuation(
            false,
            initialize.is_some(),
            policy
                .as_ref()
                .map(|path| read_json::<serde_json::Value>(path))
                .transpose()?
                .as_ref(),
        )?;
        spec["continuation_mode"] = serde_json::json!(mode.as_str());
        spec["continuation_mode_declaration"] = serde_json::json!("controller");
        write_json(&output.join("run.json"), &spec)?;
        Some(mode)
    };
    let mut injected = false;
    let mut last_error = String::new();
    for attempt in 0..3 {
        check_cancel(stop)?;
        if std::env::var_os("JOPE_GPU_POOL").is_some() {
            ensure!(
                !output
                    .join("resident-reconciliation-required.json")
                    .try_exists()?,
                "resident run requires explicit receipt/process reconciliation"
            );
            // A controller crash can occur before it writes the failure marker.
            // Changing the launch directory on resume must not bypass custody.
            for entry in fs::read_dir(output)? {
                let entry = entry?;
                if entry.file_type()?.is_dir()
                    && entry.file_name().to_string_lossy().starts_with("launch-")
                {
                    let prior:serde_json::Value=read_json(&entry.path().join("controller.json"))
                        .context("resident launch has no committed controller result; reconciliation required")?;
                    ensure!(
                        prior["passed"] == true,
                        "resident launch failed; reconciliation required"
                    );
                }
            }
        }
        ensure!(
            policy.as_deref().map(file_hash).transpose()? == policy_sha,
            "training policy changed during recovery"
        );
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
        if should_resume {
            // Every rank consumes this launch's frozen selection. Checkpoint
            // publication may advance latest.json without changing its input.
            let selected: serde_json::Value = read_json(&output.join("latest.json"))?;
            write_json(&launch.join("resume-input.json"), &selected)?;
        }
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
                .env("JOPE_GPU_TASK_DEADLINE_MS", task_deadline.to_string())
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            if should_resume {
                cmd.arg("--resume");
            }
            if let Some(policy) = &policy {
                cmd.arg("--replay-policy").arg(policy);
            }
            if let Some(mode) = lineage_mode {
                cmd.arg("--continuation-mode").arg(mode.as_str());
            }
            #[cfg(feature = "neural")]
            crate::gpu_worker::inherit_lease(&mut cmd, &_lease)?;
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
            bind_parent_death(&mut cmd);
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
                ensure!(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_millis()
                        < task_deadline as u128,
                    "training task absolute deadline elapsed"
                );
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
                verify_training_rank(&r, &step_window, rank, world, &corpus_hash)?;
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
                    &serde_json::json!({"schema":"parallel.training_complete.v2","steps":step_window["target_optimizer_step"],"step_window":step_window,"committed_new_steps":steps,"world":world,"microbatch":micro,"attempts":attempt+1,"rank_failure_injected":injected,"corpus_sha256":corpus_hash,"latest":read_json::<serde_json::Value>(&output.join("latest.json"))?,"qualification":true,"final_tests_opened":false}),
                )?;
                let mut state: serde_json::Value =
                    read_json(&root.join("runs/neural-service.json"))?;
                state["phase"] = "completed real-corpus training block".into();
                state["timestamp"] = crate::telemetry::now().into();
                write_json(&root.join("runs/neural-service.json"), &state)?;
                return Ok(());
            }
            Err(e) => {
                // A new launch path must not turn uncertain resident acceptance
                // into a fresh task. Preserve the original launch for explicit
                // receipt/process reconciliation, including injected failures.
                if std::env::var_os("JOPE_GPU_POOL").is_some() {
                    write_json(
                        &output.join("resident-reconciliation-required.json"),
                        &serde_json::json!({"schema":"parallel.resident_reconciliation.v1",
                            "launch":launch,"attempt":attempt,"failure":format!("{e:#}"),
                            "automatic_retry_permitted":false}),
                    )?;
                    return Err(e.context(
                        "resident task acceptance requires reconciliation; automatic retry refused",
                    ));
                }
                last_error = format!("{e:#}");
            }
        }
    }
    anyhow::bail!("training exhausted three coordinated attempts: {last_error}")
}

/// A failed rank or expired deadline kills/reaps every rank. Launch directories
/// are immutable, so a stale rendezvous can never join a new communicator.
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
    ensure!(
        [1, 2, 4, 8].contains(&world)
            && [64, 128, 256, 512, 1024].contains(&microbatch)
            && [2048, 4096, 8192].contains(&global_batch),
        "unsupported throughput probe topology"
    );
    let local_total = world * microbatch;
    ensure!(
        global_batch % local_total == 0,
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
        unsafe {
            let parent = libc::getpid();
            cmd.pre_exec(move || {
                if libc::setpgid(0, 0) != 0
                    || libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0
                {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(std::io::Error::other("probe controller exited"));
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
        assert_eq!(probe_accumulation(8, 256, 2048).unwrap(), 1);
        assert_eq!(probe_accumulation(8, 128, 2048).unwrap(), 2);
        assert!(probe_accumulation(8, 512, 2048).is_err());
        assert_eq!(probe_accumulation(8, 512, 4096).unwrap(), 1);
        assert_eq!(probe_accumulation(8, 1024, 8192).unwrap(), 1);
        for (world, micro, global) in [
            (0, 256, 2048),
            (8, 0, 2048),
            (8, usize::MAX, 2048),
            (8, 256, 0),
        ] {
            assert!(probe_accumulation(world, micro, global).is_err());
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
    fn proposal_grant_hold_and_legacy_protocol_both_reject_before_snapshot_or_gpu_work() {
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
            .contains("original native GPU grant verifier unavailable"));
        assert!(verify_proposal_protocol(&output, true)
            .unwrap_err()
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

#[cfg(test)]
mod lifetime_tests {
    use super::*;
    #[test]
    #[ignore = "subprocess fixture for abrupt controller death"]
    fn parent_death_child() {
        let path = PathBuf::from(std::env::var_os("JOPE_PARENT_DEATH_TEST").unwrap());
        let mut command = Command::new("sleep");
        command.arg("60");
        bind_parent_death(&mut command);
        let child = command.spawn().unwrap();
        fs::write(path, child.id().to_string()).unwrap();
        std::thread::sleep(Duration::from_secs(60));
    }
    #[test]
    fn sigkill_of_controller_reaps_its_rank_clients() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("child.pid");
        let mut parent = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "controller::lifetime_tests::parent_death_child",
            ])
            .env("JOPE_PARENT_DEATH_TEST", &path)
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let start = Instant::now();
        while !path.is_file() {
            assert!(start.elapsed() < Duration::from_secs(10));
            std::thread::sleep(Duration::from_millis(20));
        }
        let pid = fs::read_to_string(&path).unwrap();
        parent.kill().unwrap();
        parent.wait().unwrap();
        let start = Instant::now();
        loop {
            let stat = fs::read_to_string(format!("/proc/{pid}/stat"));
            if stat.is_err()
                || stat
                    .unwrap()
                    .rsplit(')')
                    .next()
                    .unwrap()
                    .trim_start()
                    .starts_with('Z')
            {
                break;
            }
            if start.elapsed() > Duration::from_secs(5) {
                unsafe {
                    libc::kill(pid.parse().unwrap(), libc::SIGKILL);
                }
                panic!("rank survived its controller");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

#[cfg(test)]
mod training_window_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn optimizer_carry_requires_new_steps_and_exact_rank_identity() {
        let window = training_step_window(9000, 1000).unwrap();
        let receipt = json!({"rank":0,"world":8,"steps":10000,"local_steps":1000,
            "initial_optimizer_step":9000,"executed_steps":1000,"corpus_sha256":"corpus"});
        verify_training_rank(&receipt, &window, 0, 8, "corpus").unwrap();
        for (key, value) in [
            ("steps", json!(9000)),
            ("local_steps", json!(0)),
            ("executed_steps", json!(0)),
            ("rank", json!(1)),
            ("world", json!(4)),
        ] {
            let mut bad = receipt.clone();
            bad[key] = value;
            assert!(
                verify_training_rank(&bad, &window, 0, 8, "corpus").is_err(),
                "{key}"
            );
        }
    }

    #[test]
    fn exact_resume_counts_only_remaining_attempt_exposure() {
        let window = training_step_window(9000, 1000).unwrap();
        let receipt = json!({"rank":0,"world":8,"steps":10000,"local_steps":1000,
            "initial_optimizer_step":9500,"executed_steps":500,"corpus_sha256":"corpus"});
        verify_training_rank(&receipt, &window, 0, 8, "corpus").unwrap();
        assert!(training_step_window(u64::MAX, 1).is_err());
        assert!(training_step_window(0, 0).is_err());
    }
}
