use crate::{
    manifest::*,
    telemetry::{now, Hardware},
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::os::unix::process::CommandExt;
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpJob {
    pub prepared: PathBuf,
    pub prepared_sha256: String,
    pub cut: usize,
    pub output: PathBuf,
    #[serde(default)]
    pub seeds: Option<PathBuf>,
    #[serde(default)]
    pub seeds_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint_sha256: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpJobs {
    pub schema: String,
    pub qualification: bool,
    pub jobs: Vec<GpJob>,
    #[serde(default)]
    pub vocabulary: Option<PathBuf>,
}

type Slot = Option<(usize, Child, Instant)>;
struct OwnedSlots(Vec<Slot>);
impl std::ops::Deref for OwnedSlots {
    type Target = Vec<Slot>;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl std::ops::DerefMut for OwnedSlots {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}
impl Drop for OwnedSlots {
    fn drop(&mut self) {
        for (_, child, _) in self.0.iter_mut().flatten() {
            unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL); }
            let _ = child.wait();
        }
    }
}
static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
extern "C" fn stop(_: libc::c_int) { STOP.store(true, std::sync::atomic::Ordering::Relaxed); }

/// Recover compact durable results before dispatch starts. Retain only pending
/// jobs, preserving their order; never walk the much larger observation ledgers.
fn restore_completed(root: &Path, jobs: &mut Vec<GpJob>) -> Result<(usize, u64)> {
    let mut pending = Vec::with_capacity(jobs.len());
    let mut completed = 0usize;
    let mut evaluations = 0u64;
    for job in jobs.drain(..) {
        let summary_path = resolve(root, &job.output)?.join("search.json");
        if summary_path.is_file() {
            let summary: serde_json::Value = read_json(&summary_path)?;
            ensure!(
                summary["schema"] == "parallel.search.v1",
                "invalid completed GP summary"
            );
            if let Some(seed) = job.rng_seed {
                ensure!(summary["rng_seed"] == seed, "completed GP RNG identity changed");
                ensure!(summary["seeds_sha256"] == serde_json::to_value(&job.seeds_sha256)?, "completed GP population changed");
            }
            if job.generation.is_some() {
                ensure!(read_json::<serde_json::Value>(&resolve(root, &job.output)?.join("job-binding.json"))? == serde_json::to_value(&job)?, "completed GP generation changed");
            }
            let count = summary["completed_evaluations"]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("completed GP evaluation count absent"))?;
            ensure!(count <= 1000, "completed GP count exceeds view budget");
            completed += 1;
            evaluations += count;
        } else {
            pending.push(job);
        }
    }
    *jobs = pending;
    Ok((completed, evaluations))
}
pub fn run(root: &Path, manifest: &Path, workers: usize, threads: usize) -> Result<()> {
    run_with_wait(root, manifest, workers, threads, false)
}

pub fn run_with_wait(root: &Path, manifest: &Path, workers: usize, threads: usize, wait_for_work: bool) -> Result<()> {
    let mut jobs: GpJobs = read_json(manifest)?;
    ensure!(
        jobs.schema == "parallel.gp_jobs.v1" && (wait_for_work || !jobs.jobs.is_empty()),
        "invalid GP jobs"
    );
    ensure!(
        workers > 0 && threads > 0 && workers * threads <= 104,
        "GP worker CPU budget exceeds 104"
    );
    let mut cpus = (8..60).chain(70..122).collect::<Vec<usize>>();
    let allowed = fs::read_to_string("/proc/self/status")?;
    ensure!(
        allowed.contains("Cpus_allowed_list:"),
        "CPU affinity unavailable"
    );
    cpus.truncate(workers * threads);
    // All destinations stay inside the workspace. Job identities are checked
    // again at dispatch; completed immutable results are not rerun on restart.
    for j in &jobs.jobs {
        validate_generation(root, j)?;
        relative(&j.prepared)?;
        relative(&j.output)?;
        ensure!(
            file_hash(&resolve(root, &j.prepared)?.join("prepared.json"))? == j.prepared_sha256,
            "prepared job changed"
        );
    }
    let executable = std::env::current_exe()?;
    let executable_sha256 = file_hash(&executable)?;
    let manifest_sha256 = file_hash(manifest)?;
    let vocabulary = jobs
        .vocabulary
        .as_ref()
        .map(|p| resolve(root, p))
        .transpose()?;
    let vocabulary_sha256 = vocabulary.as_deref().map(file_hash).transpose()?;
    let mut known = jobs
        .jobs
        .iter()
        .map(|j| Ok((j.output.clone(), serde_json::to_value(j)?)))
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    ensure!(
        known.len() == jobs.jobs.len(),
        "duplicate GP output identity"
    );
    let started = Instant::now();
    let mut slots = OwnedSlots((0..workers).map(|_| None).collect());
    if wait_for_work {
        STOP.store(false, std::sync::atomic::Ordering::Relaxed);
        unsafe { libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t);
                 libc::signal(libc::SIGINT, stop as *const () as libc::sighandler_t); }
    }
    let mut next = 0;
    let mut failed = 0;
    let mut retry = std::collections::BTreeMap::<PathBuf, (GpJob, Instant, u32)>::new();
    let mut retry_counts = std::collections::BTreeMap::<PathBuf, u32>::new();
    let mut failed_rows = Vec::<serde_json::Value>::new();
    let mut previous_total = None;
    let mut rate = 0.0;
    let mut last_sample = Instant::now();
    let lock = root.join("runs/gp-service.lock");
    let _lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock)?;
    use std::os::fd::AsRawFd;
    ensure!(
        unsafe { libc::flock(_lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "GP service already running"
    );
    let (mut completed, mut evaluations) = restore_completed(root, &mut jobs.jobs)?;
    let mut restored_completed = completed;
    let mut restored_evaluations = evaluations;
    loop {
        if wait_for_work && STOP.load(std::sync::atomic::Ordering::Relaxed) {
            write_json(&root.join("runs/gp-service.json"), &serde_json::json!({
                "schema":"parallel.gp_service.v1", "timestamp":now(), "phase":"paused",
                "termination_reason":"paused", "active":0,"queued":jobs.jobs.len()-next,
                "completed_views":completed,"wait_for_work":true}))?;
            break;
        }
        // New frozen-checkpoint searches take the next free slot without
        // interrupting active views or expanding the worker resource budget.
        let priority_path = root.join("runs/gp-priority-jobs.json");
        if priority_path.exists() {
            let priority: GpJobs = read_json(&priority_path)?;
            ensure!(
                priority.schema == "parallel.gp_jobs.v1"
                    && priority.qualification == jobs.qualification,
                "priority qualification mismatch"
            );
            if let Some(path) = &priority.vocabulary {
                ensure!(
                    Some(file_hash(&resolve(root, path)?)?) == vocabulary_sha256,
                    "priority vocabulary mismatch"
                );
            }
            let mut added = Vec::new();
            for job in priority.jobs {
                if let Some(bound) = known.get(&job.output) {
                    ensure!(*bound == serde_json::to_value(&job)?, "existing GP output identity changed");
                    continue;
                }
                validate_generation(root, &job)?;
                relative(&job.output)?;
                ensure!(
                    file_hash(&resolve(root, &job.prepared)?.join("prepared.json"))?
                        == job.prepared_sha256,
                    "priority cut changed"
                );
                if let Some(p) = &job.seeds {
                    ensure!(
                        Some(file_hash(&resolve(root, p)?)?) == job.seeds_sha256,
                        "priority seeds changed"
                    );
                }
                known.insert(job.output.clone(), serde_json::to_value(&job)?);
                added.push(job);
            }
            let (finished, historical) = restore_completed(root, &mut added)?;
            completed += finished;
            evaluations += historical;
            restored_completed += finished;
            restored_evaluations += historical;
            // Every active slot has an index below `next`. Inserting pending
            // priority work here cannot invalidate the active job identities.
            jobs.jobs.splice(next..next, added);
        }
        let ready = retry.iter().filter(|(_, (_, at, _))| *at <= Instant::now()).map(|(path, _)| path.clone()).collect::<Vec<_>>();
        for path in ready {
            let (job, _, _) = retry.remove(&path).unwrap();
            let mut pending = vec![job];
            let (done, count) = restore_completed(root, &mut pending)?;
            completed += done; evaluations += count;
            restored_completed += done; restored_evaluations += count;
            jobs.jobs.extend(pending);
        }
        let control: serde_json::Value = if wait_for_work {
            read_json(&root.join("runs/continuous-control.json")).unwrap_or_default()
        } else { serde_json::Value::Null };
        if control["paused"] == true {
            ensure!(control["schema"] == "parallel.continuous_control.v1", "unrecognized continuous pause control");
            for slot in slots.iter_mut() {
                if let Some((index, mut child, _)) = slot.take() {
                    unsafe {libc::kill(-(child.id() as i32), libc::SIGKILL);}
                    let _ = child.wait();
                    let job = jobs.jobs[index].clone();
                    retry.insert(job.output.clone(), (job, Instant::now(), 0));
                }
            }
            write_json(&root.join("runs/gp-service.json"), &serde_json::json!({
                "schema":"parallel.gp_service.v1","timestamp":now(),"phase":"paused","active":0,
                "queued":jobs.jobs.len()-next+retry.len(),"completed_views":completed,"wait_for_work":true,
                "termination_reason":"explicit pause","workers":workers,"threads_each":threads}))?;
            std::thread::sleep(Duration::from_secs(1));
            continue;
        }
        let h: Hardware = read_json(&root.join("runs/hardware.json")).unwrap_or_default();
        let encoding: serde_json::Value =
            read_json(&root.join("runs/encoding-service.json")).unwrap_or_default();
        let encoding_active = encoding["persistent"] == true
            && encoding["phase"] != "completed"
            && now().saturating_sub(encoding["timestamp"].as_u64().unwrap_or(0)) <= 5;
        let healthy = !encoding_active
            && now().saturating_sub(h.timestamp) <= 5
            && h.available_gb >= 128.0
            && h.disk_free_gb >= 50.0;
        let gate: String = if healthy {
            "dispatching bounded real-data GP jobs"
        } else {
            "paused admission: exclusive encoding block, stale telemetry, RAM below 128 GiB, or disk below 50 GiB"
        }
        .into();
        for slot in slots.iter_mut() {
            if let Some((index, child, begun)) = slot {
                let finished = child.try_wait()?;
                if let Some(status) = finished {
                    if status.success() {
                        completed += 1;
                        let summary: serde_json::Value = read_json(
                            &resolve(root, &jobs.jobs[*index].output)?.join("search.json"),
                        )?;
                        evaluations += summary["completed_evaluations"].as_u64().unwrap_or(0);
                    } else {
                        failed += 1;
                        let job = &jobs.jobs[*index];
                        let attempts = retry_counts.entry(job.output.clone()).or_default();
                        *attempts += 1;
                        let backoff = 5u64 * 2u64.pow((*attempts).min(6));
                        let row = serde_json::json!({"timestamp":now(),"output":job.output,"attempt":attempts,
                            "exit_code":status.code(),"retry_after_seconds":if wait_for_work {Some(backoff)} else {None}});
                        write_json(&root.join(format!("runs/gp-failures/{}-{}-{failed}.json", now(), std::process::id())), &row)?;
                        failed_rows.push(row);
                        if wait_for_work {retry.insert(job.output.clone(), (job.clone(), Instant::now()+Duration::from_secs(backoff), *attempts));}
                    }
                    *slot = None;
                } else if begun.elapsed() > Duration::from_secs(660) {
                    unsafe {
                        libc::kill(-(child.id() as i32), libc::SIGKILL);
                    }
                    let _ = child.wait();
                    failed += 1;
                    let job = &jobs.jobs[*index];
                    let attempts = retry_counts.entry(job.output.clone()).or_default();
                    *attempts += 1;
                    let backoff = 5u64 * 2u64.pow((*attempts).min(6));
                    let row = serde_json::json!({"timestamp":now(),"output":job.output,"attempt":attempts,
                        "failure":"bounded search timeout", "retry_after_seconds":if wait_for_work {Some(backoff)} else {None}});
                    write_json(&root.join(format!("runs/gp-failures/{}-{}-{failed}.json", now(), std::process::id())), &row)?;
                    failed_rows.push(row);
                    if wait_for_work {retry.insert(job.output.clone(), (job.clone(), Instant::now()+Duration::from_secs(backoff), *attempts));}
                    *slot = None;
                }
            }
        }
        if healthy {
            for (slot_index, slot) in slots.iter_mut().enumerate() {
                if slot.is_some() {
                    continue;
                }
                if next == jobs.jobs.len() {
                    break;
                }
                let index = next;
                next += 1;
                let j = &jobs.jobs[index];
                let output = resolve(root, &j.output)?;
                let log_path = root.join(format!("runs/gp-worker-{slot_index}.log"));
                let log = fs::OpenOptions::new().create(true).write(true).append(wait_for_work).truncate(!wait_for_work).open(log_path)?;
                let affinity = cpus[slot_index * threads..(slot_index + 1) * threads]
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                let mut cmd = Command::new("taskset");
                cmd.args(["-c", &affinity])
                    .arg(&executable)
                    .args(["search", "--prepared"])
                    .arg(resolve(root, &j.prepared)?)
                    .arg("--cut")
                    .arg(j.cut.to_string())
                    .arg("--output")
                    .arg(&output)
                    .env("RAYON_NUM_THREADS", threads.to_string())
                    .env("OMP_NUM_THREADS", threads.to_string())
                    .env("OPENBLAS_NUM_THREADS", "1")
                    .env("MKL_NUM_THREADS", "1")
                    .stdout(Stdio::from(log.try_clone()?))
                    .stderr(Stdio::from(log));
                if let Some(seed) = j.rng_seed {
                    cmd.arg("--rng-seed").arg(seed.to_string());
                }
                if j.generation.is_some() {
                    fs::create_dir_all(&output)?;
                    let binding = output.join("job-binding.json");
                    if binding.exists() {
                        ensure!(read_json::<serde_json::Value>(&binding)? == serde_json::to_value(j)?, "GP generation binding changed");
                    } else {
                        write_json(&binding, j)?;
                    }
                }
                if jobs.qualification {
                    cmd.arg("--qualification");
                }
                if let Some(v) = &vocabulary {
                    ensure!(
                        Some(file_hash(v)?) == vocabulary_sha256,
                        "service vocabulary changed"
                    );
                    cmd.arg("--vocabulary").arg(v);
                }
                if let Some(p) = &j.seeds {
                    let path = resolve(root, p)?;
                    ensure!(
                        Some(file_hash(&path)?) == j.seeds_sha256,
                        "job seeds changed"
                    );
                    cmd.arg("--seeds").arg(path);
                }
                let parent_pid = std::process::id() as libc::pid_t;
                unsafe {
                    cmd.pre_exec(move || {
                        if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 || libc::getppid() != parent_pid {
                            return Err(std::io::Error::other("GP service parent lost"));
                        }
                        if libc::setpgid(0, 0) != 0 {
                            return Err(std::io::Error::last_os_error());
                        }
                        Ok(())
                    });
                }
                *slot = Some((index, cmd.spawn()?, Instant::now()));
            }
        }
        let mut active_rows = Vec::new();
        let mut running_evaluations = 0u64;
        for (slot_index, slot) in slots.iter().enumerate() {
            if let Some((i, child, _)) = slot {
                let j = &jobs.jobs[*i];
                let progress: serde_json::Value =
                    read_json(&resolve(root, &j.output)?.join("progress.json")).unwrap_or_default();
                let n = progress["evaluations"].as_u64().unwrap_or(0);
                running_evaluations += n;
                active_rows.push(serde_json::json!({"slot":slot_index,"pid":child.id(),"dataset":j.prepared.file_name().unwrap_or_default().to_string_lossy(),"cut":j.cut,"evaluations":n,"best_rmse":progress["best_validation_rmse"],"elapsed_seconds":progress["wall_seconds"]}));
            }
        }
        // Restored results are progress, but they are not fresh throughput.
        let total = evaluations + running_evaluations - restored_evaluations;
        if let Some(previous) = previous_total {
            let instant = total.saturating_sub(previous) as f64
                / last_sample.elapsed().as_secs_f64().max(0.1);
            rate = 0.8 * rate + 0.2 * instant;
        }
        previous_total = Some(total);
        last_sample = Instant::now();
        let active = slots.iter().filter(|s| s.is_some()).count();
        let state = serde_json::json!({"schema":"parallel.gp_service.v1","timestamp":now(),"phase":if active==0&&next==jobs.jobs.len(){if wait_for_work {if retry.is_empty() {"waiting"} else {"backoff"}} else {"completed"}}else{"running"},"wait_for_work":wait_for_work,"termination_reason":if active==0&&next==jobs.jobs.len(){Some("queue exhausted")}else{None},"qualification":jobs.qualification,"executable":executable,"executable_sha256":executable_sha256,"manifest_sha256":manifest_sha256,"vocabulary_sha256":vocabulary_sha256,"workers":workers,"threads_each":threads,"active":active,"queued":jobs.jobs.len()-next+retry.len(),"completed_views":completed,"restored_completed_views":restored_completed,"restored_completed_evaluations":restored_evaluations,"failed_views":failed,"failed_rows":failed_rows,"retry_pending":retry.len(),"completed_evaluations":evaluations,"running_evaluations":running_evaluations,"evaluations_per_second":rate,"active_rows":active_rows,"elapsed_seconds":started.elapsed().as_secs(),"admission":gate,"cpu_budget":workers*threads,"worker_memory_limit_gib":8,"candidate_timeout_seconds":30,"view_budget":"1000 evaluations / 600 seconds"});
        write_json(&root.join("runs/gp-service.json"), &state)?;
        if active == 0 && next == jobs.jobs.len() && !wait_for_work {
            break;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    Ok(())
}

/// Generations are qualification teaching work over eligible frozen contexts.
pub fn validate_generation(root: &Path, job: &GpJob) -> Result<()> {
    if job.generation.is_none() && job.parent_checkpoint.is_none() && job.parent_checkpoint_sha256.is_none() {
        return Ok(());
    }
    ensure!(job.generation.is_some() && job.rng_seed.is_some(), "generation and RNG seed required");
    let parent = job.parent_checkpoint.as_ref().ok_or_else(|| anyhow::anyhow!("generation parent missing"))?;
    ensure!(Some(file_hash(&resolve(root, parent)?.join("state.json"))?) == job.parent_checkpoint_sha256,
        "generation parent checkpoint changed");
    let prepared: serde_json::Value = read_json(&resolve(root, &job.prepared)?.join("prepared.json"))?;
    ensure!(matches!(prepared["meta_partition"].as_str(), Some("meta_train" | "meta_validation")),
        "sealed-test jobs cannot enter teaching generations");
    ensure!(job.seeds.is_some() && job.seeds_sha256.is_some(), "generation seed population missing");
    Ok(())
}
