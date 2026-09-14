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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpJob {
    pub prepared: PathBuf,
    pub prepared_sha256: String,
    pub cut: usize,
    pub output: PathBuf,
    #[serde(default)]
    pub seeds: Option<PathBuf>,
    #[serde(default)]
    pub seeds_sha256: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpJobs {
    pub schema: String,
    pub qualification: bool,
    pub jobs: Vec<GpJob>,
    #[serde(default)]
    pub vocabulary: Option<PathBuf>,
}

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
    let mut jobs: GpJobs = read_json(manifest)?;
    ensure!(
        jobs.schema == "parallel.gp_jobs.v1" && !jobs.jobs.is_empty(),
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
        .map(|j| j.output.clone())
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        known.len() == jobs.jobs.len(),
        "duplicate GP output identity"
    );
    let started = Instant::now();
    let mut slots: Vec<Option<(usize, Child, Instant)>> = (0..workers).map(|_| None).collect();
    let mut next = 0;
    let mut failed = 0;
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
                if known.contains(&job.output) {
                    continue;
                }
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
                known.insert(job.output.clone());
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
        for slot in &mut slots {
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
                    }
                    *slot = None;
                } else if begun.elapsed() > Duration::from_secs(660) {
                    unsafe {
                        libc::kill(-(child.id() as i32), libc::SIGKILL);
                    }
                    let _ = child.wait();
                    failed += 1;
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
                let log = fs::File::create(log_path)?;
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
                unsafe {
                    cmd.pre_exec(|| {
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
        let state = serde_json::json!({"schema":"parallel.gp_service.v1","timestamp":now(),"phase":if active==0&&next==jobs.jobs.len(){"completed"}else{"running"},"qualification":jobs.qualification,"executable":executable,"executable_sha256":executable_sha256,"manifest_sha256":manifest_sha256,"vocabulary_sha256":vocabulary_sha256,"workers":workers,"threads_each":threads,"active":active,"queued":jobs.jobs.len()-next,"completed_views":completed,"restored_completed_views":restored_completed,"restored_completed_evaluations":restored_evaluations,"failed_views":failed,"completed_evaluations":evaluations,"running_evaluations":running_evaluations,"evaluations_per_second":rate,"active_rows":active_rows,"elapsed_seconds":started.elapsed().as_secs(),"admission":gate,"cpu_budget":workers*threads,"worker_memory_limit_gib":8,"candidate_timeout_seconds":30,"view_budget":"1000 evaluations / 600 seconds"});
        write_json(&root.join("runs/gp-service.json"), &state)?;
        if active == 0 && next == jobs.jobs.len() {
            break;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    Ok(())
}
