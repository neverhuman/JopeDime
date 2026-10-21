use crate::{
    manifest::*,
    telemetry::{now, Hardware, StageHeartbeat},
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::os::unix::process::CommandExt;
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

pub mod coverage;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_view: Option<String>,
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
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for OwnedSlots {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Drop for OwnedSlots {
    fn drop(&mut self) {
        for (_, child, _) in self.0.iter_mut().flatten() {
            // SAFETY: kill takes only integers and has no memory-safety preconditions; every
            // path that reaps a worker frees its slot first, so a child still in a slot is
            // unreaped and its setpgid(0, 0) group id cannot have been reused.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
        }
    }
}
static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Recover compact durable results before dispatch starts. Retain only pending
/// jobs, preserving their order; never walk the much larger observation ledgers.
fn restore_completed(
    root: &Path,
    jobs: &mut Vec<GpJob>,
    coverage: &mut coverage::Ledger,
    restored: bool,
) -> Result<(usize, u64)> {
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
                ensure!(
                    summary["rng_seed"] == seed,
                    "completed GP RNG identity changed"
                );
                ensure!(
                    summary["seeds_sha256"] == serde_json::to_value(&job.seeds_sha256)?,
                    "completed GP population changed"
                );
            }
            if job.generation.is_some() {
                ensure!(
                    read_json::<serde_json::Value>(
                        &resolve(root, &job.output)?.join("job-binding.json")
                    )? == serde_json::to_value(&job)?,
                    "completed GP generation changed"
                );
            }
            let count = summary["completed_evaluations"]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("completed GP evaluation count absent"))?;
            ensure!(count <= 1000, "completed GP count exceeds view budget");
            if !restored || coverage.tracks(&job)? {
                ensure!(
                    summary["terminal"] == true
                        && summary["completed_at"].as_u64().is_some_and(|n| n > 0)
                        && summary["prepared_sha256"] == job.prepared_sha256
                        && summary["cut"] == job.cut
                        && summary["observations_sha256"]
                            == file_hash(&summary_path.with_file_name("observations.json"))?,
                    "worker completion payload or cut binding absent"
                );
            }
            coverage.reused(&job, now(), restored)?;
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

pub fn run_with_wait(
    root: &Path,
    manifest: &Path,
    workers: usize,
    threads: usize,
    wait_for_work: bool,
) -> Result<()> {
    let mut jobs: GpJobs = read_json(manifest)?;
    ensure!(
        jobs.schema == "parallel.gp_jobs.v1" && (wait_for_work || !jobs.jobs.is_empty()),
        "invalid GP jobs"
    );
    ensure!(
        workers > 0 && threads > 0 && workers * threads <= 104,
        "GP worker CPU budget exceeds 104"
    );
    // The GP slot layout (CPUs 8-59 and 70-121), restricted to the CPUs this process may run
    // on. Pinning a slot to a CPU outside the affinity mask or cgroup cpuset makes `taskset`
    // fail at dispatch, which records the job's outcome as unknown; refuse the budget up front
    // instead. On a host that allows the whole layout the slot CPUs are unchanged.
    let allowed = allowed_cpus(&fs::read_to_string("/proc/self/status")?)?;
    let mut cpus = (8..60)
        .chain(70..122)
        .filter(|cpu| allowed.contains(cpu))
        .collect::<Vec<usize>>();
    ensure!(
        workers * threads <= cpus.len(),
        "GP worker CPU budget {workers}x{threads} exceeds the {} allowed CPUs of the GP slot \
         layout (8-59, 70-121)",
        cpus.len()
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
    STOP.store(false, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: `stop` is an extern "C" fn(c_int) whose only action is a lock-free atomic store,
    // which is async-signal-safe, so it is a valid SIGTERM and SIGINT handler.
    unsafe {
        libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, stop as *const () as libc::sighandler_t);
    }
    let mut next = 0;
    let mut failed = 0;
    let mut coverage = coverage::Ledger::open(root)?;
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
    ensure!(_lock.try_lock().is_ok(), "GP service already running");
    let heartbeat = StageHeartbeat::start(
        &root.join("runs/gp-service.json"),
        serde_json::json!({
        "schema":"parallel.gp_service.v1","phase":"reconciling",
        "executable":executable,"executable_sha256":executable_sha256,
        "manifest_sha256":manifest_sha256,"workers":workers,"threads_each":threads}),
    )?;
    let (mut completed, mut evaluations) =
        restore_completed(root, &mut jobs.jobs, &mut coverage, true)?;
    let mut restored_completed = completed;
    let mut restored_evaluations = evaluations;
    let withheld = loop {
        if STOP.load(std::sync::atomic::Ordering::Relaxed) {
            let interrupted = slots.iter().flatten().count();
            for (index, _, _) in slots.iter().flatten() {
                coverage.finish(
                    &jobs.jobs[*index],
                    "unknown",
                    "controller shutdown interrupted search",
                    now(),
                )?;
            }
            // Reap owned workers before announcing shutdown. Interrupted jobs
            // still require authority reconciliation, never a new fit identity.
            drop(slots);
            heartbeat.publish(serde_json::json!({
                "schema":"parallel.gp_service.v1", "timestamp":now(), "phase":"paused",
                "termination_reason":"paused", "active":0,"queued":jobs.jobs.len()-next,
                "unknown_outcomes":interrupted,"completed_views":completed,"completed_evaluations":evaluations,
                "failed_views":failed,"executable_sha256":executable_sha256,"wait_for_work":wait_for_work}))?;
            return heartbeat.finish();
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
                    ensure!(
                        *bound == serde_json::to_value(&job)?,
                        "existing GP output identity changed"
                    );
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
            let (finished, historical) = restore_completed(root, &mut added, &mut coverage, true)?;
            completed += finished;
            evaluations += historical;
            restored_completed += finished;
            restored_evaluations += historical;
            // Every active slot has an index below `next`. Inserting pending
            // priority work here cannot invalidate the active job identities.
            jobs.jobs.splice(next..next, added);
        }
        let control: serde_json::Value = if wait_for_work {
            read_json(&root.join("runs/continuous-control.json")).unwrap_or_default()
        } else {
            serde_json::Value::Null
        };
        if control["paused"] == true {
            ensure!(
                control["schema"] == "parallel.continuous_control.v1",
                "unrecognized continuous pause control"
            );
            for slot in slots.iter_mut() {
                if let Some((index, mut child, _)) = slot.take() {
                    // SAFETY: kill takes only integers and has no memory-safety preconditions;
                    // the worker is unreaped, so its setpgid(0, 0) group id is not reused.
                    unsafe {
                        libc::kill(-(child.id() as i32), libc::SIGKILL);
                    }
                    let _ = child.wait();
                    coverage.finish(
                        &jobs.jobs[index],
                        "unknown",
                        "explicit pause interrupted search",
                        now(),
                    )?;
                }
            }
            heartbeat.publish(serde_json::json!({
                "schema":"parallel.gp_service.v1","timestamp":now(),"phase":"paused","active":0,
                "queued":jobs.jobs.len()-next,"completed_views":completed,"wait_for_work":true,
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
        let reservation = crate::qualification_slots::reserved(root, workers, threads, now());
        let reservation_error = reservation.as_ref().err().map(ToString::to_string);
        let reserved_slots = reservation.unwrap_or_default();
        let healthy = reservation_error.is_none()
            && !encoding_active
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
                    // The worker is reaped: free its slot before any fallible bookkeeping, so an
                    // early return never leaves a reaped pid for OwnedSlots::drop to signal.
                    let index = *index;
                    *slot = None;
                    if status.success() {
                        completed += 1;
                        let mut checked = vec![jobs.jobs[index].clone()];
                        let (_, count) =
                            restore_completed(root, &mut checked, &mut coverage, false)?;
                        ensure!(
                            checked.is_empty(),
                            "successful worker lacks durable completion"
                        );
                        evaluations += count;
                    } else {
                        failed += 1;
                        let job = &jobs.jobs[index];
                        let row = serde_json::json!({"timestamp":now(),"output":job.output,
                            "exit_code":status.code(),"outcome":"unknown",
                            "diagnosis":"worker exited without a verified completion; authority reconciliation required",
                            "automatic_retry":false});
                        coverage.finish(job, "unknown", "abnormal worker exit", now())?;
                        write_json(
                            &root.join(format!(
                                "runs/gp-failures/{}-{}-{failed}.json",
                                now(),
                                std::process::id()
                            )),
                            &row,
                        )?;
                        failed_rows.push(row);
                    }
                } else if begun.elapsed() > Duration::from_secs(660) {
                    // SAFETY: kill takes only integers and has no memory-safety preconditions;
                    // the worker is unreaped, so its setpgid(0, 0) group id is not reused.
                    unsafe {
                        libc::kill(-(child.id() as i32), libc::SIGKILL);
                    }
                    let _ = child.wait();
                    let index = *index;
                    *slot = None;
                    failed += 1;
                    let job = &jobs.jobs[index];
                    let row = serde_json::json!({"timestamp":now(),"output":job.output,
                        "outcome":"unknown","diagnosis":"bounded search timeout; authority reconciliation required",
                        "automatic_retry":false});
                    coverage.finish(job, "unknown", "bounded search timeout", now())?;
                    write_json(
                        &root.join(format!(
                            "runs/gp-failures/{}-{}-{failed}.json",
                            now(),
                            std::process::id()
                        )),
                        &row,
                    )?;
                    failed_rows.push(row);
                }
            }
        }
        let mut dispatch_blocked = false;
        let mut coverage_error = None::<String>;
        if healthy {
            for (slot_index, slot) in slots.iter_mut().enumerate() {
                if slot.is_some() || reserved_slots.contains(&slot_index) {
                    continue;
                }
                if next == jobs.jobs.len() {
                    break;
                }
                let Some(lease) = crate::qualification_slots::try_acquire(root, slot_index)? else {
                    continue;
                };
                let lease_fd = lease.as_raw_fd();
                let selected = match coverage.reserve(root, &jobs.jobs[next..], now()) {
                    Ok(Some(index)) => index,
                    Ok(None) => {
                        dispatch_blocked = true;
                        break;
                    }
                    Err(error) => {
                        coverage_error = Some(format!("{error:#}"));
                        dispatch_blocked = true;
                        break;
                    }
                };
                jobs.jobs.swap(next, next + selected);
                let index = next;
                next += 1;
                let j = &jobs.jobs[index];
                let output = resolve(root, &j.output)?;
                let log_path = root.join(format!("runs/gp-worker-{slot_index}.log"));
                let log = fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .append(wait_for_work)
                    .truncate(!wait_for_work)
                    .open(log_path)?;
                let affinity = cpus[slot_index * threads..(slot_index + 1) * threads]
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                let mut cmd = Command::new("taskset");
                cmd.args(["-c", &affinity])
                    .arg(&executable)
                    .arg("--root")
                    .arg(root)
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
                        ensure!(
                            read_json::<serde_json::Value>(&binding)? == serde_json::to_value(j)?,
                            "GP generation binding changed"
                        );
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
                // SAFETY: the hook runs between fork and exec and makes only async-signal-safe
                // calls (setpgid, prctl, getppid, the fcntl in qualification_slots::inherit); its
                // errors are errno codes, so nothing allocates in the child.
                unsafe {
                    cmd.pre_exec(move || {
                        crate::qualification_slots::inherit(lease_fd)?;
                        if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                            return Err(std::io::Error::last_os_error());
                        }
                        // The service exited: refuse with an errno, which does not allocate
                        // (an io::Error carrying a message would box it in the forked child).
                        if libc::getppid() != parent_pid {
                            return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
                        }
                        if libc::setpgid(0, 0) != 0 {
                            return Err(std::io::Error::last_os_error());
                        }
                        Ok(())
                    });
                }
                match cmd.spawn() {
                    Ok(child) => *slot = Some((index, child, Instant::now())),
                    Err(error) => {
                        coverage.finish(
                            j,
                            "failed_before_start",
                            &format!("spawn rejected: {error}"),
                            now(),
                        )?;
                        failed += 1;
                        failed_rows.push(serde_json::json!({"output":j.output,"outcome":"failed_before_start",
                            "diagnosis":error.to_string(),"cooldown_seconds":30,"automatic_retry":false}));
                    }
                }
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
                active_rows.push(serde_json::json!({"slot":slot_index,"pid":child.id(),"dataset":j.prepared.file_name().unwrap_or_default().to_string_lossy(),"cut":j.cut,"evaluations":n,"best_rmse":progress["best_validation_rmse"],"units":"original_target","elapsed_seconds":progress["wall_seconds"]}));
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
        let state = serde_json::json!({"schema":"parallel.gp_service.v1","timestamp":now(),"phase":if active==0&&dispatch_blocked {"unavailable"} else if active==0&&next==jobs.jobs.len(){if wait_for_work {"waiting"} else {"completed"}}else{"running"},"wait_for_work":wait_for_work,"termination_reason":if active==0&&next==jobs.jobs.len(){Some("queue exhausted")}else{None},"qualification":jobs.qualification,"executable":executable,"executable_sha256":executable_sha256,"manifest_sha256":manifest_sha256,"vocabulary_sha256":vocabulary_sha256,"workers":workers,"threads_each":threads,"active":active,"queued":jobs.jobs.len()-next,"completed_views":completed,"restored_completed_views":restored_completed,"restored_completed_evaluations":restored_evaluations,"failed_views":failed,"failed_rows":failed_rows,"retry_pending":0,"coverage":coverage.status()?,"coverage_error":coverage_error,"automatic_retry":false,"completed_evaluations":evaluations,"running_evaluations":running_evaluations,"evaluations_per_second":rate,"active_rows":active_rows,"elapsed_seconds":started.elapsed().as_secs(),"admission":gate,"cpu_budget":workers*threads,"worker_memory_limit_gib":8,"candidate_timeout_seconds":30,"view_budget":"1000 evaluations / 600 seconds","qualification_reserved_slots":reserved_slots,"qualification_reservation_error":reservation_error,"slot_leases":true});
        heartbeat.publish(state)?;
        if active == 0 && (next == jobs.jobs.len() || dispatch_blocked) && !wait_for_work {
            break dispatch_blocked;
        }
        std::thread::sleep(Duration::from_secs(1));
    };
    heartbeat.finish()?;
    ensure!(
        !withheld,
        "coverage dispatch unavailable; inspect gp-service.json"
    );
    Ok(())
}

/// Generations are qualification teaching work over eligible frozen contexts.
pub fn validate_generation(root: &Path, job: &GpJob) -> Result<()> {
    if job.generation.is_none()
        && job.parent_checkpoint.is_none()
        && job.parent_checkpoint_sha256.is_none()
    {
        return Ok(());
    }
    ensure!(
        job.generation.is_some() && job.rng_seed.is_some(),
        "generation and RNG seed required"
    );
    let parent = job
        .parent_checkpoint
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("generation parent missing"))?;
    ensure!(
        Some(file_hash(&resolve(root, parent)?.join("state.json"))?)
            == job.parent_checkpoint_sha256,
        "generation parent checkpoint changed"
    );
    let prepared: serde_json::Value =
        read_json(&resolve(root, &job.prepared)?.join("prepared.json"))?;
    ensure!(
        matches!(
            prepared["meta_partition"].as_str(),
            Some("meta_train" | "meta_validation")
        ),
        "sealed-test jobs cannot enter teaching generations"
    );
    ensure!(
        job.seeds.is_some() && job.seeds_sha256.is_some(),
        "generation seed population missing"
    );
    Ok(())
}

/// The CPUs in `/proc/self/status`'s `Cpus_allowed_list` (affinity mask and cpuset), e.g.
/// `0-3,8,10-11`.
fn allowed_cpus(status: &str) -> Result<std::collections::BTreeSet<usize>> {
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .context("CPU affinity unavailable")?
        .trim();
    let mut cpus = std::collections::BTreeSet::new();
    for part in list.split(',') {
        let (start, end) = match part.split_once('-') {
            Some((start, end)) => (start.parse::<usize>()?, end.parse::<usize>()?),
            None => {
                let cpu = part.parse::<usize>()?;
                (cpu, cpu)
            }
        };
        ensure!(
            start <= end && end < 1 << 16,
            "invalid allowed CPU range {part}"
        );
        cpus.extend(start..=end);
    }
    ensure!(!cpus.is_empty(), "allowed CPU list is empty");
    Ok(cpus)
}

#[cfg(test)]
mod allowed_cpu_tests {
    use super::allowed_cpus;

    #[test]
    fn allowed_cpu_list_parses_ranges_and_refuses_malformed_lists() {
        let status = "Name:\tjope\nCpus_allowed:\tff\nCpus_allowed_list:\t0-3,8,10-11\n";
        assert_eq!(
            allowed_cpus(status)
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            [0, 1, 2, 3, 8, 10, 11]
        );
        for bad in ["", "3-1", "1,,2", "a", "0-70000"] {
            assert!(
                allowed_cpus(&format!("Cpus_allowed_list:\t{bad}\n")).is_err(),
                "{bad}"
            );
        }
        assert!(allowed_cpus("Name:\tjope\n").is_err());
    }
}
