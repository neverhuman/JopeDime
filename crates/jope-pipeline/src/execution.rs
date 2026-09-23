//! Bounded execution of trusted, pinned fit commands on Linux cgroup v2.
//!
//! The controller must be the main process of a qualified systemd service. A
//! command is released only after its gated process joins its fit cgroup. Normal
//! fork, double-fork and setsid descendants stay in that group. This is resource
//! isolation for trusted product engines, not a sandbox for hostile same-UID code.
//! Worker completion means a clean command and retained output, never a trusted
//! quality score: the caller scores predictions against targets kept in the parent.

use crate::experiment::{EvaluationRequest, EvaluationStatus, ExecutionWindow, Experiment};
use anyhow::{ensure, Context, Result};
use nix::fcntl::{Flock, FlockArg};
use nix::sys::{
    signal::{killpg, Signal},
    statfs::{statfs, CGROUP2_SUPER_MAGIC},
};
use nix::time::{clock_gettime, ClockId};
use nix::unistd::Pid;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::{
        fs::OpenOptionsExt,
        process::{CommandExt, ExitStatusExt},
    },
    path::{Component, Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const EXECUTION_SCHEMA: &str = "jope.worker-execution/v2";
pub const HOST_LOCK: &str = "/run/lock/jope-exclusive-cpu.lock";
const LOG_LIMIT: usize = 1024 * 1024;
const POLL_MS: u64 = 10;
const DRAIN_MS: u64 = 1000;
const THREAD_VARS: &[&str] = &[
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
}
impl Artifact {
    pub fn verify(&self) -> Result<()> {
        ensure!(
            crate::experiment::sha256(&self.sha256),
            "invalid artifact digest"
        );
        regular(&self.path)?;
        ensure!(
            hash_file(&self.path)? == self.sha256,
            "artifact changed: {}",
            self.path.display()
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerCommand {
    pub program: Artifact,
    pub args: Vec<String>,
    /// Includes scripts, the request JSON and every admitted data input. No
    /// scoring target or whole-panel path belongs in this closure.
    pub inputs: Vec<Artifact>,
    /// Explicit environment is identity-bound. Numerical thread variables cannot
    /// override the pool's one-thread setting; credentials are never inherited.
    pub environment: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmBudget {
    pub id_sha256: String,
    pub seconds: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerJob {
    pub request: EvaluationRequest,
    pub command: WorkerCommand,
    /// New or previously journaled directory inside the private journal tree.
    pub directory: PathBuf,
    /// Relative output names, including the complete fitted artifact and predictions.
    pub outputs: Vec<PathBuf>,
    pub arm: Option<ArmBudget>,
}
impl WorkerJob {
    pub fn identity(&self) -> Result<String> {
        self.request.validate()?;
        ensure!(!self.outputs.is_empty(), "no declared fit outputs");
        let outputs: BTreeSet<_> = self.outputs.iter().collect();
        ensure!(
            outputs.len() == self.outputs.len(),
            "duplicate declared outputs"
        );
        for path in &self.outputs {
            ensure!(
                !path.as_os_str().is_empty()
                    && path.components().all(|c| matches!(c, Component::Normal(_))),
                "output must be a relative normal path"
            );
            ensure!(
                path != Path::new("stdout.log") && path != Path::new("stderr.log"),
                "output collides with logs"
            );
        }
        if let Some(arm) = &self.arm {
            ensure!(
                crate::experiment::sha256(&arm.id_sha256) && arm.seconds > 0 && arm.seconds <= 120,
                "invalid search arm allowance"
            );
        }
        for (key, value) in &self.command.environment {
            ensure!(
                !key.is_empty()
                    && !key.contains('=')
                    && !key.contains('\0')
                    && !value.contains('\0'),
                "invalid worker environment"
            );
            ensure!(
                !THREAD_VARS.contains(&key.as_str())
                    && !matches!(
                        key.as_str(),
                        "CUDA_VISIBLE_DEVICES"
                            | "TMPDIR"
                            | "HOME"
                            | "GH_TOKEN"
                            | "GITHUB_TOKEN"
                            | "AWS_ACCESS_KEY_ID"
                            | "AWS_SECRET_ACCESS_KEY"
                            | "AWS_SESSION_TOKEN"
                    ),
                "reserved worker environment: {key}"
            );
        }
        Ok(crate::sha256_hex(&serde_json::to_vec(self)?))
    }
    fn verify_inputs(&self) -> Result<()> {
        self.command.program.verify()?;
        ensure!(
            !self.command.inputs.is_empty(),
            "worker input closure is empty"
        );
        for input in &self.command.inputs {
            input.verify()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DependencyCheck {
    pub job_sha256: String,
    pub request_sha256: String,
    pub available: bool,
    pub reason: Option<String>,
}
/// Verifies the declared executable/input closure without starting a window or a
/// subprocess. Adapter --version/import capability probes remain caller-owned.
pub fn inventory_dependencies(jobs: &[WorkerJob]) -> Result<Vec<DependencyCheck>> {
    jobs.iter()
        .map(|job| {
            let error = job.verify_inputs().err();
            Ok(DependencyCheck {
                job_sha256: job.identity()?,
                request_sha256: job.request.identity()?,
                available: error.is_none(),
                reason: error.map(|e| format!("{e:#}")),
            })
        })
        .collect()
}

/// Null means unknown, never zero. Snapshots before an interrupted controller
/// are lower bounds retained separately from final consumption.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedConsumption {
    #[serde(default)]
    pub gate_released_unix_ms: Option<u64>,
    #[serde(default)]
    pub observed_exit_unix_ms: Option<u64>,
    pub wall_seconds: Option<f64>,
    pub cpu_seconds: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub cgroup_peak_memory_bytes: Option<u64>,
    pub sampled_peak_rss_bytes: Option<u64>,
    pub accounting_reason: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerOutcome {
    pub schema: String,
    pub job_sha256: String,
    pub request_sha256: String,
    /// Clean command exit and intact declared outputs; caller still validates
    /// prediction shape, numerical policy and scores independently.
    pub command_status: EvaluationStatus,
    pub reason: Option<String>,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub outputs: Vec<Artifact>,
    pub stdout: Option<Artifact>,
    pub stderr: Option<Artifact>,
    pub consumption: ObservedConsumption,
}
impl WorkerOutcome {
    fn unstarted(job: &WorkerJob, status: EvaluationStatus, reason: String) -> Result<Self> {
        Ok(Self {
            schema: EXECUTION_SCHEMA.into(),
            job_sha256: job.identity()?,
            request_sha256: job.request.identity()?,
            command_status: status,
            reason: Some(reason),
            exit_code: None,
            signal: None,
            outputs: vec![],
            stdout: None,
            stderr: None,
            consumption: ObservedConsumption {
                wall_seconds: Some(0.),
                cpu_seconds: Some(0.),
                accounting_reason: "not_started: gate never released".into(),
                ..Default::default()
            },
        })
    }
    pub fn verify_outputs(&self) -> Result<()> {
        ensure!(self.schema == EXECUTION_SCHEMA, "unknown execution receipt");
        for artifact in self
            .outputs
            .iter()
            .chain(self.stdout.iter())
            .chain(self.stderr.iter())
        {
            artifact.verify()?;
        }
        ensure!(
            self.command_status != EvaluationStatus::Completed || !self.outputs.is_empty(),
            "completed command lacks outputs"
        );
        Ok(())
    }
}

/// Existing provisioned service only: this library never starts/stops services,
/// changes the runner fleet, or relaxes a missing isolation capability.
#[derive(Clone, Debug)]
pub struct Supervisor {
    pub unit: String,
    pub cgroup_root: PathBuf,
}
impl Supervisor {
    fn verify(&self, experiment: &Experiment, execution_deadline: u64) -> Result<()> {
        ensure!(
            self.unit.starts_with("jope-asm-")
                && self.unit.ends_with(".service")
                && self
                    .unit
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-._".contains(&b)),
            "unexpected supervisor unit"
        );
        let props = systemd_properties(&self.unit)?;
        for (key, expected) in [
            ("Type", "exec"),
            ("ExitType", "main"),
            ("KillMode", "control-group"),
            ("SendSIGKILL", "yes"),
            ("Restart", "no"),
            ("OOMPolicy", "continue"),
            ("RemainAfterExit", "no"),
            ("ActiveState", "active"),
        ] {
            ensure!(
                props.get(key).map(String::as_str) == Some(expected),
                "supervisor {key} must be {expected}"
            );
        }
        ensure!(
            matches!(
                props.get("FinalKillSignal").map(String::as_str),
                Some("9" | "SIGKILL")
            ),
            "supervisor final signal must be SIGKILL"
        );
        ensure!(
            props.get("MainPID").and_then(|v| v.parse::<u32>().ok()) == Some(std::process::id()),
            "controller must be supervisor MainPID"
        );
        let control = props
            .get("ControlGroup")
            .context("supervisor cgroup missing")?;
        ensure!(
            self.cgroup_root == Path::new("/sys/fs/cgroup/system.slice").join(&self.unit)
                && self.cgroup_root
                    == Path::new("/sys/fs/cgroup").join(control.trim_start_matches('/')),
            "supervisor must be the expected system.slice cgroup"
        );
        ensure!(
            fs::canonicalize(&self.cgroup_root)? == self.cgroup_root
                && statfs(&self.cgroup_root)?.filesystem_type() == CGROUP2_SUPER_MAGIC,
            "real physical cgroup v2 required"
        );
        let own = fs::read_to_string("/proc/self/cgroup")?;
        ensure!(
            own.lines()
                .any(|line| line == format!("0::{control}/controller")),
            "controller must occupy delegated controller subgroup"
        );
        let runtime = parse_timespan(
            props
                .get("RuntimeMaxUSec")
                .context("runtime limit absent")?,
        )?;
        let stop = parse_timespan(props.get("TimeoutStopUSec").context("stop bound absent")?)?;
        if let Some(extra) = props.get("RuntimeRandomizedExtraUSec") {
            ensure!(
                parse_timespan(extra)? == 0,
                "randomized supervisor runtime is forbidden"
            );
        }
        ensure!(
            stop <= DRAIN_MS * 1000 && stop > 0,
            "supervisor stop grace exceeds one second"
        );
        let started: u64 = props
            .get("ActiveEnterTimestampMonotonic")
            .context("supervisor start absent")?
            .parse()?;
        let monotonic = clock_gettime(ClockId::CLOCK_MONOTONIC)?;
        let mono_us = monotonic.tv_sec() as u64 * 1_000_000 + monotonic.tv_nsec() as u64 / 1000;
        ensure!(
            mono_us >= started && runtime > mono_us - started,
            "supervisor runtime expired"
        );
        let remaining_ms = (runtime - (mono_us - started) + stop).div_ceil(1000);
        ensure!(
            now_ms()?
                .checked_add(remaining_ms)
                .is_some_and(|end| end <= execution_deadline),
            "supervisor may outlive original execution deadline"
        );
        let resources = &experiment.resources;
        verify_limit(&self.cgroup_root, "memory.max", resources.memory_bytes)?;
        ensure!(
            control_text(&self.cgroup_root, "memory.swap.max")? == "0",
            "swap must be zero"
        );
        verify_cpu(&self.cgroup_root, resources.cpus as u64)?;
        let cpus = parse_cpus(&control_text(&self.cgroup_root, "cpuset.cpus.effective")?)?;
        ensure!(
            !cpus.is_empty()
                && cpus.len() <= resources.cpus as usize
                && cpus.len() >= resources.workers as usize,
            "supervisor CPU allocation differs"
        );
        ensure!(
            control_text(&self.cgroup_root, "cgroup.procs")?.is_empty(),
            "supervisor inner cgroup has processes"
        );
        // systemd delegates these controllers and moves MainPID into /controller,
        // but the delegate must enable them for its own child groups. Initialize
        // only this already-qualified, empty service root; never an ancestor.
        let available = control_text(&self.cgroup_root, "cgroup.controllers")?;
        ensure!(
            ["cpu", "memory", "cpuset"]
                .iter()
                .all(|v| available.split_whitespace().any(|e| e == *v)),
            "required controllers were not delegated"
        );
        fs::write(
            self.cgroup_root.join("cgroup.subtree_control"),
            "+cpu +cpuset +memory",
        )
        .context("enable delegated fit controllers")?;
        let enabled = control_text(&self.cgroup_root, "cgroup.subtree_control")?;
        ensure!(
            ["cpu", "memory", "cpuset"]
                .iter()
                .all(|v| enabled.split_whitespace().any(|e| e == *v)),
            "required delegated controllers unavailable"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Started {
    job: WorkerJob,
    job_sha256: String,
    started_unix_ms: u64,
    fit_deadline_unix_ms: u64,
    cgroup: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmWindow {
    id_sha256: String,
    seconds: u64,
    started_unix_ms: u64,
    deadline_unix_ms: u64,
}

pub struct ExecutionPool {
    experiment: Experiment,
    directory: PathBuf,
    supervisor: Supervisor,
    _journal_lock: Flock<File>,
    host_lock: Option<Flock<File>>,
    window: Option<ExecutionWindow>,
}
impl ExecutionPool {
    pub fn open(experiment: &Experiment, directory: &Path, supervisor: Supervisor) -> Result<Self> {
        experiment.validate()?;
        ensure!(
            directory.is_absolute() && !directory.starts_with("/tmp"),
            "journal requires absolute durable path outside /tmp"
        );
        fs::create_dir_all(directory)?;
        let directory = fs::canonicalize(directory)?;
        ensure!(
            !directory.starts_with("/tmp"),
            "journal resolves under /tmp"
        );
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(directory.join("execution.lock"))?;
        let journal_lock = Flock::lock(lock, FlockArg::LockExclusiveNonblock)
            .map_err(|(_, error)| anyhow::anyhow!("journal already owned: {error}"))?;
        immutable(&directory.join("experiment.json"), experiment)?;
        for child in [
            "started", "finished", "arms", "jobs", "requests", "logs", "cleanup",
        ] {
            fs::create_dir_all(directory.join(child))?;
        }
        let window = if directory.join("window.json").exists() {
            let window: ExecutionWindow = read_json(&directory.join("window.json"))?;
            window.validate(experiment)?;
            Some(window)
        } else {
            None
        };
        Ok(Self {
            experiment: experiment.clone(),
            directory,
            supervisor,
            _journal_lock: journal_lock,
            host_lock: None,
            window,
        })
    }
    pub fn window(&self) -> Option<&ExecutionWindow> {
        self.window.as_ref()
    }

    /// Acquire and retain exclusive host allocation and verify the delegated
    /// supervisor before package hashing, data preparation or training preflight.
    /// This does not persist/start/reset either experiment or arm clocks. Holding
    /// the reservation forbids overlapping exclusive full-CPU CI. Drop the pool
    /// to release it; later admit/begin_arm performs the checks again.
    pub fn reserve(&mut self) -> Result<()> {
        let now = now_ms()?;
        let window = self.window.clone().unwrap_or(ExecutionWindow {
            experiment_sha256: self.experiment.identity()?,
            started_unix_ms: now,
            deadline_unix_ms: now
                .checked_add(self.experiment.resources.total_seconds * 1000)
                .context("deadline overflow")?,
        });
        ensure!(
            window.may_admit(&self.experiment, now)?,
            "original execution window exhausted"
        );
        self.acquire_host()?;
        self.supervisor.verify(
            &self.experiment,
            window.deadline_unix_ms - self.experiment.resources.reporting_seconds * 1000,
        )
    }

    /// Starts the single experiment window after the caller's dependency
    /// inventory. Call before any proposal work, not after expensive encoding.
    pub fn admit(&mut self) -> Result<&ExecutionWindow> {
        self.reserve()?;
        if self.window.is_none() {
            let now = now_ms()?;
            let window = ExecutionWindow {
                experiment_sha256: self.experiment.identity()?,
                started_unix_ms: now,
                deadline_unix_ms: now
                    .checked_add(self.experiment.resources.total_seconds * 1000)
                    .context("deadline overflow")?,
            };
            immutable(&self.directory.join("window.json"), &window)?;
            self.window = Some(window);
        }
        self.window.as_ref().context("execution window absent")
    }
    /// Returns the persisted absolute cutoff. Encoding/proposals and fits all
    /// share this arm clock. A second call never grants a fresh allowance.
    pub fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64> {
        ensure!(
            crate::experiment::sha256(&arm.id_sha256) && arm.seconds > 0 && arm.seconds <= 120,
            "invalid arm allowance"
        );
        let reporting_ms = self.experiment.resources.reporting_seconds * 1000;
        let execution_deadline = self.admit()?.deadline_unix_ms - reporting_ms;
        Ok(self.arm_deadline(arm)?.min(execution_deadline))
    }

    /// Dataset round-robin admission; each slot is a process, never an in-process fit.
    /// An interrupted started job is terminal Unknown accounting, not an implicit retry.
    pub fn run(
        &mut self,
        jobs: Vec<WorkerJob>,
        cancellation: &AtomicBool,
    ) -> Result<Vec<WorkerOutcome>> {
        // Reservation-only consumers (category training) share the supervisor
        // and original window, while every actual fit job keeps this strict cap.
        ensure!(
            self.experiment.resources.per_fit_seconds <= 30,
            "fit allowance exceeds30 seconds"
        );
        let order = jobs
            .iter()
            .enumerate()
            .map(|(i, j)| Ok((j.identity()?, i)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let active_cancel = Arc::new(AtomicBool::new(cancellation.load(Ordering::Acquire)));
        let finished = AtomicBool::new(false);
        struct Done<'a>(&'a AtomicBool);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        thread::scope(|scope| {
            let _done = Done(&finished);
            let mirrored = active_cancel.clone();
            let completion = &finished;
            scope.spawn(move || {
                while !completion.load(Ordering::Acquire) {
                    if cancellation.load(Ordering::Acquire) {
                        mirrored.store(true, Ordering::Release);
                    }
                    thread::sleep(Duration::from_millis(POLL_MS));
                }
            });
            let mut results = self.run_inner(jobs, active_cancel)?;
            results.sort_by_key(|o| order.get(&o.job_sha256).copied().unwrap_or(usize::MAX));
            Ok(results)
        })
    }
    fn run_inner(
        &mut self,
        jobs: Vec<WorkerJob>,
        cancellation: Arc<AtomicBool>,
    ) -> Result<Vec<WorkerOutcome>> {
        let mut ids = BTreeSet::new();
        let mut requests = BTreeSet::new();
        let mut queue = FairQueue::default();
        let mut outcomes = Vec::new();
        for job in jobs {
            let id = job.identity()?;
            ensure!(ids.insert(id.clone()), "duplicate job in batch");
            ensure!(
                job.request.experiment_sha256 == self.experiment.identity()?,
                "request belongs to another experiment"
            );
            let request_id = job.request.identity()?;
            let ownership = self
                .directory
                .join("requests")
                .join(format!("{request_id}.json"));
            if ownership.exists() {
                ensure!(read_json::<String>(&ownership)? == id,
                    "request identity already belongs to another arm/command; derive distinct request recipe/seed binding");
            }
            ensure!(
                requests.insert(request_id.clone()),
                "request repeated with a different command"
            );
            ensure!(
                job.directory == self.directory.join("jobs").join(request_id),
                "job directory must equal journal/jobs/<request identity>"
            );
            if let Some(prior) = self.cached(&job)? {
                outcomes.push(prior);
            } else if cancellation.load(Ordering::Acquire) && !self.started_path(&id).exists() {
                outcomes.push(WorkerOutcome::unstarted(
                    &job,
                    EvaluationStatus::Cancelled,
                    "cancelled before admission".into(),
                )?);
            } else if !self.started_path(&id).exists() {
                if let Err(error) = job.verify_inputs() {
                    let outcome = WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::Unavailable,
                        format!("pre-admission input/dependency unavailable: {error:#}"),
                    )?;
                    // Dependency inventory is not a scientific attempt.
                    outcomes.push(outcome);
                } else {
                    queue.push(job.request.dataset_sha256.clone(), job);
                }
            } else {
                queue.push(job.request.dataset_sha256.clone(), job);
            }
        }
        if queue.is_empty() {
            return Ok(outcomes);
        }
        let admitted = now_ms()?;
        let candidate_window = self.window.clone().unwrap_or(ExecutionWindow {
            experiment_sha256: self.experiment.identity()?,
            started_unix_ms: admitted,
            deadline_unix_ms: admitted
                .checked_add(self.experiment.resources.total_seconds * 1000)
                .context("deadline overflow")?,
        });
        let may_admit = candidate_window.may_admit(&self.experiment, admitted)?;
        let execution_deadline =
            candidate_window.deadline_unix_ms - self.experiment.resources.reporting_seconds * 1000;
        if !may_admit {
            while let Some(job) = queue.pop() {
                let id = job.identity()?;
                let outcome = if self.started_path(&id).exists() {
                    self.interrupted(&job, false)?
                } else {
                    WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::BudgetExhausted,
                        "original execution window exhausted".into(),
                    )?
                };
                immutable(&self.finished_path(&id), &outcome)?;
                outcomes.push(outcome);
            }
            return Ok(outcomes);
        }
        let admission = self
            .acquire_host()
            .and_then(|()| self.supervisor.verify(&self.experiment, execution_deadline));
        if let Err(error) = admission {
            while let Some(job) = queue.pop() {
                let id = job.identity()?;
                let mut outcome = WorkerOutcome::unstarted(
                    &job,
                    EvaluationStatus::Unavailable,
                    format!("execution admission unavailable: {error:#}"),
                )?;
                if self.started_path(&id).exists() {
                    outcome = self.interrupted(&job, false)?;
                    outcome.command_status = EvaluationStatus::Unavailable;
                    outcome.reason = Some(format!("original attempt unknown; current execution admission unavailable: {error:#}"));
                }
                outcomes.push(outcome);
            }
            // Admission failures are an inventory, not cached fit results or a started clock.
            atomic_json(
                &self.directory.join("admission-unavailable.json"),
                &outcomes,
            )?;
            return Ok(outcomes);
        }
        if self.window.is_none() {
            immutable(&self.directory.join("window.json"), &candidate_window)?;
            self.window = Some(candidate_window);
        }
        let cpus = parse_cpus(&control_text(
            &self.supervisor.cgroup_root,
            "cpuset.cpus.effective",
        )?)?;
        let mut active = ActiveFits(Vec::new());
        let mut admission_blocked = self.reconcile_cleanup()?;
        let mut pending = FairQueue::default();
        while let Some(job) = queue.pop() {
            let id = job.identity()?;
            if self.started_path(&id).exists() {
                let outcome = self.interrupted(&job, true)?;
                if outcome
                    .reason
                    .as_ref()
                    .is_some_and(|r| r.contains("recovery_not_drained"))
                {
                    admission_blocked = Some("prior attempt could not be safely drained".into());
                }
                if !outcome
                    .reason
                    .as_ref()
                    .is_some_and(|r| r.contains("recovery_not_drained"))
                {
                    immutable(&self.finished_path(&id), &outcome)?;
                }
                outcomes.push(outcome);
            } else {
                pending.push(job.request.dataset_sha256.clone(), job);
            }
        }
        queue = pending;
        while !queue.is_empty() || !active.is_empty() {
            for index in (0..active.len()).rev() {
                if active[index].is_done() {
                    let running = active.swap_remove(index);
                    let group = running.started.cgroup.clone();
                    let outcome = running.finish()?;
                    if !group.exists()
                        || named_counter(&group, "cgroup.events", "populated").ok() == Some(0)
                    {
                        if self.started_path(&outcome.job_sha256).exists() {
                            immutable(&self.finished_path(&outcome.job_sha256), &outcome)?;
                        }
                        self.clear_cleanup(&outcome.job_sha256)?;
                        // Kernel counters are now retained in the finished receipt.
                        let _ = fs::remove_dir(&group);
                    } else {
                        admission_blocked = Some(
                            "previous fit cgroup did not drain; no more fits admitted".to_owned(),
                        );
                    }
                    outcomes.push(outcome);
                }
            }
            while active.len() < self.experiment.resources.workers as usize {
                let Some(job) = queue.pop() else {
                    break;
                };
                let id = job.identity()?;
                let terminal = if let Some(reason) = &admission_blocked {
                    Some(WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::Unavailable,
                        reason.clone(),
                    )?)
                } else if self.started_path(&id).exists() {
                    Some(self.interrupted(&job, true)?)
                } else if cancellation.load(Ordering::Acquire) {
                    Some(WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::Cancelled,
                        "cancelled before admission".into(),
                    )?)
                } else if now_ms()? >= execution_deadline {
                    Some(WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::BudgetExhausted,
                        "original execution window exhausted".into(),
                    )?)
                } else if let Err(error) = job.verify_inputs() {
                    Some(WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::Unavailable,
                        format!("input/dependency unavailable: {error:#}"),
                    )?)
                } else {
                    None
                };
                if let Some(outcome) = terminal {
                    if self.started_path(&id).exists()
                        || outcome.command_status == EvaluationStatus::BudgetExhausted
                    {
                        immutable(&self.finished_path(&id), &outcome)?;
                    }
                    outcomes.push(outcome);
                    continue;
                }
                let arm_deadline = job
                    .arm
                    .as_ref()
                    .map(|arm| self.arm_deadline(arm))
                    .transpose()?
                    .unwrap_or(execution_deadline);
                let deadline = execution_deadline.min(arm_deadline);
                if deadline <= now_ms()? {
                    let outcome = WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::BudgetExhausted,
                        "original search arm window exhausted".into(),
                    )?;
                    immutable(&self.finished_path(&id), &outcome)?;
                    outcomes.push(outcome);
                    continue;
                }
                let used: BTreeSet<_> = active.iter().map(|a| a.cpu).collect();
                let cpu = *cpus
                    .iter()
                    .find(|cpu| !used.contains(*cpu))
                    .context("no free isolated CPU")?;
                let started = Started {
                    job,
                    job_sha256: id.clone(),
                    started_unix_ms: now_ms()?,
                    fit_deadline_unix_ms: deadline,
                    cgroup: self.supervisor.cgroup_root.join(format!("fit-{id}")),
                };
                immutable(
                    &self
                        .directory
                        .join("requests")
                        .join(format!("{}.json", started.job.request.identity()?)),
                    &id,
                )?;
                // Cleanup ownership is durable before a cgroup or process can exist.
                // It remains independent of scientific attempt/finished state.
                immutable(&self.cleanup_path(&id), &started)?;
                match Running::start(
                    started.clone(),
                    cpu,
                    self.experiment.resources.memory_bytes,
                    self.experiment.resources.per_fit_seconds,
                    self.started_path(&id),
                    self.directory.join("logs"),
                    cancellation.clone(),
                ) {
                    Ok(running) => active.push(running),
                    Err(error) => {
                        let recovery = self.recover_group(&started, true)?;
                        if recovery.is_empty() {
                            self.clear_cleanup(&id)?;
                        } else {
                            admission_blocked = Some(format!("launch cleanup pending{recovery}"));
                        }
                        let mut outcome = WorkerOutcome::unstarted(
                            &started.job,
                            EvaluationStatus::Unavailable,
                            format!("fit launch unavailable: {error:#}"),
                        )?;
                        // Setup never released the gate; retain inventory only.
                        outcome.consumption = ObservedConsumption {
                            accounting_reason:
                                "launch failed before gate release; gate/setup CPU unavailable"
                                    .into(),
                            ..Default::default()
                        };
                        outcomes.push(outcome);
                    }
                }
            }
            if !active.is_empty() {
                thread::sleep(Duration::from_millis(POLL_MS));
            }
        }
        Ok(outcomes)
    }
    fn cached(&self, job: &WorkerJob) -> Result<Option<WorkerOutcome>> {
        let id = job.identity()?;
        let path = self.finished_path(&id);
        if !path.exists() {
            return Ok(None);
        }
        let prior: WorkerOutcome = read_json(&path)?;
        ensure!(
            prior.job_sha256 == id && prior.request_sha256 == job.request.identity()?,
            "stale terminal identity"
        );
        if prior.command_status == EvaluationStatus::Completed {
            job.verify_inputs()?;
        }
        prior.verify_outputs()?;
        Ok(Some(prior))
    }
    fn interrupted(&self, job: &WorkerJob, drain: bool) -> Result<WorkerOutcome> {
        let id = job.identity()?;
        let original: Started = read_json(&self.started_path(&id))?;
        ensure!(
            original.job_sha256 == id && original.job.identity()? == id,
            "stale started identity"
        );
        let recovery = self.recover_group(&original, drain)?;
        let mut outcome = WorkerOutcome::unstarted(
            job,
            EvaluationStatus::Cancelled,
            format!("controller_interrupted: original attempt not retried{recovery}"),
        )?;
        outcome.consumption = ObservedConsumption {
            accounting_reason:
                "original controller interrupted; final counters and exit time unknown".into(),
            ..Default::default()
        };
        Ok(outcome)
    }
    fn cleanup_path(&self, id: &str) -> PathBuf {
        self.directory.join("cleanup").join(format!("{id}.json"))
    }
    fn clear_cleanup(&self, id: &str) -> Result<()> {
        let path = self.cleanup_path(id);
        if path.exists() {
            fs::remove_file(path)?;
            File::open(self.directory.join("cleanup"))?.sync_all()?;
        }
        Ok(())
    }
    /// Inspect every prior owned group, even if its job is absent from this batch
    /// or already has a terminal receipt. A finished receipt is never proof that
    /// all descendants disappeared. This also recovers pre-Started setup failures.
    fn reconcile_cleanup(&self) -> Result<Option<String>> {
        let mut records: BTreeMap<String, Started> = BTreeMap::new();
        for directory in ["started", "cleanup"] {
            for entry in fs::read_dir(self.directory.join(directory))? {
                let entry = entry?;
                let original: Started = read_json(&entry.path())?;
                ensure!(
                    original.job_sha256 == original.job.identity()?
                        && entry.file_name() == format!("{}.json", original.job_sha256).as_str(),
                    "invalid cleanup identity"
                );
                records.insert(original.job_sha256.clone(), original);
            }
        }
        let mut blocked = None;
        for (id, original) in records {
            let recovery = self.recover_group(&original, true)?;
            if recovery.is_empty() {
                self.clear_cleanup(&id)?;
            } else {
                blocked = Some(format!("prior fit {id} cleanup pending{recovery}"));
            }
        }
        Ok(blocked)
    }
    fn recover_group(&self, original: &Started, drain: bool) -> Result<String> {
        let id = original.job.identity()?;
        ensure!(original.job_sha256 == id, "stale cleanup job identity");
        let mut recovery = String::new();
        if original.cgroup.exists() {
            let expected_name = format!("fit-{id}");
            let parent = original.cgroup.parent();
            let owned_path = parent
                .is_some_and(|p| p.parent() == Some(Path::new("/sys/fs/cgroup/system.slice")))
                && original
                    .cgroup
                    .file_name()
                    .is_some_and(|n| n == expected_name.as_str())
                && fs::canonicalize(&original.cgroup).ok().as_ref() == Some(&original.cgroup);
            let same_unit = original.cgroup == self.supervisor.cgroup_root.join(&expected_name);
            let old_inactive = if owned_path && !same_unit {
                parent
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .filter(|n| n.starts_with("jope-asm-") && n.ends_with(".service"))
                    .and_then(|unit| systemd_properties(unit).ok())
                    .is_some_and(|props| {
                        matches!(
                            props.get("ActiveState").map(String::as_str),
                            Some("inactive" | "failed")
                        ) && props.get("MainPID").map(String::as_str) == Some("0")
                            && props.get("ControlGroup").is_some_and(|value| {
                                parent
                                    == Some(
                                        Path::new("/sys/fs/cgroup")
                                            .join(value.trim_start_matches('/'))
                                            .as_path(),
                                    )
                            })
                    })
            } else {
                false
            };
            if drain && owned_path && (same_unit || old_inactive) {
                if let Err(error) = drain_cgroup(&original.cgroup) {
                    recovery = format!("; recovery_not_drained: {error:#}");
                }
            } else if named_counter(&original.cgroup, "cgroup.events", "populated").ok() != Some(0)
            {
                recovery =
                    "; recovery_not_drained: foreign, active or unverifiable prior group".into();
            }
        }
        Ok(recovery)
    }
    fn acquire_host(&mut self) -> Result<()> {
        if self.host_lock.is_some() {
            return Ok(());
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(HOST_LOCK)
            .context("shared host lock is not provisioned")?;
        let guard = Flock::lock(file, FlockArg::LockExclusiveNonblock)
            .map_err(|(_, error)| anyhow::anyhow!("exclusive CPU host is busy: {error}"))?;
        let full = bounded_output(Command::new("/usr/bin/docker").args([
            "ps",
            "--filter",
            "label=jope.runner=1",
            "--filter",
            "label=jope.lane=full",
            "--format",
            "{{.ID}}",
        ]))
        .context("runner census")?;
        ensure!(
            full.status.success() && full.stdout.iter().all(u8::is_ascii_whitespace),
            "full-CPU runner container active or census unavailable"
        );
        // The old controller's flock can disappear before systemd finishes
        // killing its descendants. Recover only this journal's safely owned
        // inactive groups, then refuse every other populated ASM service.
        ensure!(
            self.reconcile_cleanup()?.is_none(),
            "prior owned fit cleanup remains unresolved"
        );
        require_no_other_asm_groups(
            Path::new("/sys/fs/cgroup/system.slice"),
            &self.supervisor.cgroup_root,
        )?;
        self.host_lock = Some(guard);
        Ok(())
    }
    fn started_path(&self, id: &str) -> PathBuf {
        self.directory.join("started").join(format!("{id}.json"))
    }
    fn finished_path(&self, id: &str) -> PathBuf {
        self.directory.join("finished").join(format!("{id}.json"))
    }
    fn arm_deadline(&self, arm: &ArmBudget) -> Result<u64> {
        ensure!(
            crate::experiment::sha256(&arm.id_sha256) && arm.seconds > 0 && arm.seconds <= 120,
            "invalid arm allowance"
        );
        let path = self
            .directory
            .join("arms")
            .join(format!("{}.json", arm.id_sha256));
        let window = if path.exists() {
            let window: ArmWindow = read_json(&path)?;
            ensure!(
                window.id_sha256 == arm.id_sha256
                    && window.seconds == arm.seconds
                    && window.started_unix_ms.checked_add(arm.seconds * 1000)
                        == Some(window.deadline_unix_ms),
                "search arm allowance changed"
            );
            window
        } else {
            let started_unix_ms = now_ms()?;
            let window = ArmWindow {
                id_sha256: arm.id_sha256.clone(),
                seconds: arm.seconds,
                started_unix_ms,
                deadline_unix_ms: started_unix_ms + arm.seconds * 1000,
            };
            immutable(&path, &window)?;
            window
        };
        ensure!(
            now_ms()? >= window.started_unix_ms,
            "clock precedes original arm start"
        );
        Ok(window.deadline_unix_ms)
    }
}

#[derive(Default)]
struct FairQueue {
    datasets: VecDeque<String>,
    jobs: BTreeMap<String, VecDeque<WorkerJob>>,
}
impl FairQueue {
    fn push(&mut self, dataset: String, job: WorkerJob) {
        if !self.jobs.contains_key(&dataset) {
            self.datasets.push_back(dataset.clone());
        }
        self.jobs.entry(dataset).or_default().push_back(job);
    }
    fn pop(&mut self) -> Option<WorkerJob> {
        let dataset = self.datasets.pop_front()?;
        let jobs = self.jobs.get_mut(&dataset)?;
        let job = jobs.pop_front();
        if jobs.is_empty() {
            self.jobs.remove(&dataset);
        } else {
            self.datasets.push_back(dataset);
        }
        job
    }
    fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }
}

struct ActiveFits(Vec<Running>);
impl std::ops::Deref for ActiveFits {
    type Target = Vec<Running>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for ActiveFits {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Drop for ActiveFits {
    fn drop(&mut self) {
        // Signal every watcher before joining any one of them on error paths.
        for running in &self.0 {
            running.stop.store(true, Ordering::Release);
        }
    }
}

/// Independent owner of each Child and deadline. Completion hashing can delay
/// reclamation of this slot, but cannot delay another fit's deadline or exit
/// observation. A local stop flag makes error-path Drop cancel only this fit.
struct Running {
    started: Started,
    cpu: u32,
    watcher: Option<thread::JoinHandle<Result<WorkerOutcome>>>,
    started_path: PathBuf,
    stop: Arc<AtomicBool>,
}
impl Running {
    fn start(
        started: Started,
        cpu: u32,
        memory_bytes: u64,
        per_fit_seconds: u64,
        started_path: PathBuf,
        logs: PathBuf,
        cancellation: Arc<AtomicBool>,
    ) -> Result<Self> {
        fs::create_dir_all(logs.join(&started.job_sha256))?;
        let process = ProcessFit::start(started.clone(), cpu, memory_bytes)?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let journal_path = started_path.clone();
        let watcher = thread::Builder::new()
            .name(format!("fit-{}", &started.job_sha256[..12]))
            .spawn(move || {
                process.watch(
                    &started_path,
                    per_fit_seconds,
                    &logs,
                    &cancellation,
                    &thread_stop,
                )
            })?;
        Ok(Self {
            started,
            cpu,
            watcher: Some(watcher),
            started_path: journal_path,
            stop,
        })
    }
    fn is_done(&self) -> bool {
        self.watcher.as_ref().is_some_and(|h| h.is_finished())
    }
    fn finish(mut self) -> Result<WorkerOutcome> {
        let result = self.watcher.take().context("fit watcher missing")?.join();
        let reason = match result {
            Ok(Ok(outcome)) => return Ok(outcome),
            Ok(Err(error)) => format!("fit supervision/output failure: {error:#}"),
            Err(_) => "fit watcher panicked; original outcome unknown".into(),
        };
        let status = if self.started_path.exists() {
            EvaluationStatus::Failed
        } else {
            EvaluationStatus::Unavailable
        };
        let mut outcome = WorkerOutcome::unstarted(&self.started.job, status, reason)?;
        outcome.consumption = ObservedConsumption {
            accounting_reason: "supervision failed; final counters and exit time unknown".into(),
            ..Default::default()
        };
        Ok(outcome)
    }
}
impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(watcher) = self.watcher.take() {
            let _ = watcher.join();
        }
    }
}

struct ProcessFit {
    started: Started,
    child: Option<Child>,
    status: Option<std::process::ExitStatus>,
    stdout: Option<thread::JoinHandle<std::io::Result<Vec<u8>>>>,
    stderr: Option<thread::JoinHandle<std::io::Result<Vec<u8>>>>,
    began: Instant,
    monotonic_deadline: Instant,
    finished_at: Option<Instant>,
    gate_released_unix_ms: Option<u64>,
    observed_exit_unix_ms: Option<u64>,
    stop_status: Option<EvaluationStatus>,
    sampled_peak_rss_bytes: Option<u64>,
    drained: bool,
}
impl ProcessFit {
    fn start(started: Started, cpu: u32, memory_bytes: u64) -> Result<Self> {
        if started.cgroup.exists() {
            ensure!(
                named_counter(&started.cgroup, "cgroup.events", "populated")? == 0,
                "pre-existing fit cgroup remains populated"
            );
            // Empty kernel setup from a launch that never released its gate.
            fs::remove_dir(&started.cgroup)?;
        }
        ensure!(
            !started.job.directory.starts_with("/tmp"),
            "fit directory under /tmp"
        );
        fs::create_dir_all(&started.job.directory)?;
        ensure!(
            fs::canonicalize(&started.job.directory)? == started.job.directory,
            "physical fit directory required"
        );
        let allowed: BTreeSet<_> = started
            .job
            .command
            .inputs
            .iter()
            .map(|a| a.path.as_path())
            .collect();
        let mut pending = vec![started.job.directory.clone()];
        while let Some(directory) = pending.pop() {
            for entry in fs::read_dir(directory)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    pending.push(entry.path());
                } else {
                    ensure!(
                        entry.file_type()?.is_file() && allowed.contains(entry.path().as_path()),
                        "undeclared file in private fit directory"
                    );
                }
            }
        }
        for output in &started.job.outputs {
            ensure!(
                !started.job.directory.join(output).try_exists()?,
                "stale declared output exists before fit"
            );
        }
        let remaining = started
            .fit_deadline_unix_ms
            .checked_sub(now_ms()?)
            .context("fit expired before launch")?;
        fs::create_dir(&started.cgroup)?;
        let began = Instant::now();
        let mut running = Self {
            started,
            child: None,
            status: None,
            stdout: None,
            stderr: None,
            began,
            monotonic_deadline: began + Duration::from_millis(remaining),
            finished_at: None,
            gate_released_unix_ms: None,
            observed_exit_unix_ms: None,
            stop_status: None,
            sampled_peak_rss_bytes: None,
            drained: false,
        };
        let group = &running.started.cgroup;
        let parent = group.parent().context("fit cgroup parent absent")?;
        control_write(
            group,
            "cpuset.mems",
            &control_text(parent, "cpuset.mems.effective")?,
        )?;
        control_write(group, "cpuset.cpus", &cpu.to_string())?;
        control_write(group, "cpu.max", "100000 100000")?;
        control_write(group, "memory.max", &memory_bytes.to_string())?;
        control_write(group, "memory.swap.max", "0")?;
        control_write(group, "memory.oom.group", "1")?;
        OpenOptions::new()
            .write(true)
            .open(group.join("cgroup.kill"))?;
        verify_cpu(group, 1)?;
        verify_limit(group, "memory.max", memory_bytes)?;
        ensure!(
            control_text(group, "memory.swap.max")? == "0"
                && parse_cpus(&control_text(group, "cpuset.cpus.effective")?)? == [cpu],
            "fit resource readback differs"
        );
        let job = &running.started.job;
        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                "read -r jope_gate || exit 125; test \"$jope_gate\" = go || exit 125; exec \"$@\"",
                "jope-fit-gate",
            ])
            .arg(&job.command.program.path)
            .args(&job.command.args)
            .current_dir(&job.directory)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LANG", "C")
            .env("LC_ALL", "C")
            .env("CUDA_VISIBLE_DEVICES", "-1")
            .env("TMPDIR", &job.directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        for (name, value) in &job.command.environment {
            command.env(name, value);
        }
        for name in THREAD_VARS {
            command.env(name, "1");
        }
        running.child = Some(command.spawn().context("spawn gated fit")?);
        let child = running.child.as_mut().context("gated child absent")?;
        let pid = child.id();
        // Prefer killing a fit over its controller under the aggregate memory cap.
        fs::write(format!("/proc/{pid}/oom_score_adj"), b"1000")?;
        running.stdout = Some(capture(child.stdout.take().context("stdout missing")?));
        running.stderr = Some(capture(child.stderr.take().context("stderr missing")?));
        control_write(group, "cgroup.procs", &pid.to_string())?;
        let expected = group.strip_prefix("/sys/fs/cgroup")?.to_string_lossy();
        ensure!(
            fs::read_to_string(format!("/proc/{pid}/cgroup"))?
                .lines()
                .any(|line| line == format!("0::/{expected}")),
            "fit attachment readback differs"
        );
        job.verify_inputs()?;
        ensure!(
            now_ms()? < running.started.fit_deadline_unix_ms,
            "deadline exhausted before gate release"
        );
        // The watcher owns Child before the gate is released. No fit runs on
        // the controller's artifact-hashing/scheduling thread.
        Ok(running)
    }
    fn watch(
        mut self,
        started_path: &Path,
        per_fit_seconds: u64,
        logs: &Path,
        cancellation: &AtomicBool,
        local_stop: &AtomicBool,
    ) -> Result<WorkerOutcome> {
        let gate_ms = now_ms()?;
        self.started.started_unix_ms = gate_ms;
        self.started.fit_deadline_unix_ms = self.started.fit_deadline_unix_ms.min(
            gate_ms
                .checked_add(per_fit_seconds * 1000)
                .context("fit deadline overflow")?,
        );
        ensure!(
            gate_ms < self.started.fit_deadline_unix_ms,
            "deadline exhausted before gate release"
        );
        immutable(started_path, &self.started)?;
        self.began = Instant::now();
        self.gate_released_unix_ms = Some(now_ms()?);
        self.monotonic_deadline = self.began
            + Duration::from_millis(self.started.fit_deadline_unix_ms.saturating_sub(now_ms()?));
        self.child
            .as_mut()
            .context("gated child absent")?
            .stdin
            .take()
            .context("gate stdin absent")?
            .write_all(b"go\n")?;
        while !self
            .is_done(cancellation.load(Ordering::Acquire) || local_stop.load(Ordering::Acquire))?
        {
            thread::sleep(Duration::from_millis(POLL_MS));
        }
        self.finish(logs)
    }
    fn is_done(&mut self, cancellation: bool) -> Result<bool> {
        // Poll and timestamp the exit independently of all controller hashing.
        // Already-observed exits retain their result if cancellation arrives later.
        if self.status.is_none() {
            self.status = self
                .child
                .as_mut()
                .context("fit child absent")?
                .try_wait()?;
        }
        if self.status.is_some() {
            self.finished_at.get_or_insert_with(Instant::now);
            self.observed_exit_unix_ms.get_or_insert(now_ms()?);
            return Ok(true);
        }
        if cancellation {
            self.stop_status = Some(EvaluationStatus::Cancelled);
        } else if Instant::now() >= self.monotonic_deadline
            || now_ms()? >= self.started.fit_deadline_unix_ms
        {
            self.stop_status = Some(EvaluationStatus::TimedOut);
        }
        if self.stop_status.is_some() {
            return Ok(true);
        }
        if let Ok(Some(rss)) = sampled_rss(&self.started.cgroup) {
            self.sampled_peak_rss_bytes = Some(self.sampled_peak_rss_bytes.unwrap_or(0).max(rss));
        }
        Ok(false)
    }
    fn finish(mut self, logs: &Path) -> Result<WorkerOutcome> {
        let mut outcome = WorkerOutcome::unstarted(
            &self.started.job,
            EvaluationStatus::Failed,
            "fit did not complete".into(),
        )?;
        if let Some(status) = self.stop_status {
            outcome.command_status = status;
            outcome.reason = Some(
                if status == EvaluationStatus::Cancelled {
                    "cancelled during fit"
                } else {
                    "fit or original arm deadline reached"
                }
                .into(),
            );
        } else if self.status.is_some_and(|s| s.success()) {
            outcome.command_status = EvaluationStatus::Completed;
            outcome.reason = None;
        } else if self.status.and_then(|s| s.code()) == Some(69) {
            outcome.command_status = EvaluationStatus::Unavailable;
            outcome.reason = Some("worker reported unavailable dependency (exit69)".into());
        } else {
            outcome.reason = Some(format!("worker exited {:?}", self.status));
        }
        // Even a successful leader may leave grandchildren. Kill and observe an
        // empty cgroup before releasing this CPU slot or reading final outputs.
        if self.status.is_none() {
            if let Some(child) = &self.child {
                let _ = killpg(Pid::from_raw(child.id().try_into()?), Signal::SIGKILL);
            }
        }
        drain_cgroup(&self.started.cgroup)?;
        self.status = Some(self.child.as_mut().context("fit child absent")?.wait()?);
        self.finished_at.get_or_insert_with(Instant::now);
        self.observed_exit_unix_ms.get_or_insert(now_ms()?);
        outcome.exit_code = self.status.and_then(|s| s.code());
        outcome.signal = self.status.and_then(|s| s.signal());
        if let Ok(oom_kills) = named_counter(&self.started.cgroup, "memory.events", "oom_kill") {
            if oom_kills > 0 {
                outcome.command_status = EvaluationStatus::Failed;
                outcome.reason = Some(format!(
                    "memory_limit: cgroup recorded {oom_kills} OOM kills"
                ));
            }
        }
        let cpu_seconds = named_counter(&self.started.cgroup, "cpu.stat", "usage_usec")
            .ok()
            .map(|v| v as f64 / 1_000_000.);
        let memory = control_text(&self.started.cgroup, "memory.peak")
            .ok()
            .and_then(|v| v.parse().ok());
        outcome.consumption = ObservedConsumption {
            gate_released_unix_ms: self.gate_released_unix_ms,
            observed_exit_unix_ms: self.observed_exit_unix_ms,
            wall_seconds: Some(
                self.finished_at
                    .unwrap_or_else(Instant::now)
                    .duration_since(self.began)
                    .as_secs_f64(),
            ),
            cpu_seconds,
            peak_rss_bytes: None,
            cgroup_peak_memory_bytes: memory,
            sampled_peak_rss_bytes: self.sampled_peak_rss_bytes,
            accounting_reason: if cpu_seconds.is_some() {
                "final cgroup CPU; cgroup memory includes cache/kernel; RSS is sampled only".into()
            } else {
                "final CPU counter unavailable; wall measured; memory/RSS availability explicit"
                    .into()
            },
        };
        for (thread, filename, destination) in [
            (&mut self.stdout, "stdout.log", &mut outcome.stdout),
            (&mut self.stderr, "stderr.log", &mut outcome.stderr),
        ] {
            let bytes = thread
                .take()
                .context("capture thread absent")?
                .join()
                .map_err(|_| anyhow::anyhow!("capture thread panicked"))??;
            let path = logs.join(&self.started.job_sha256).join(filename);
            write_new(&path, &bytes)?;
            *destination = Some(Artifact {
                path,
                sha256: crate::sha256_hex(&bytes),
            });
        }
        if outcome.command_status == EvaluationStatus::Completed {
            let retained = self
                .started
                .job
                .outputs
                .iter()
                .map(|name| {
                    let path = self.started.job.directory.join(name);
                    regular(&path)?;
                    ensure!(
                        fs::canonicalize(&path)?.starts_with(&self.started.job.directory),
                        "fit output escapes directory"
                    );
                    Ok(Artifact {
                        sha256: hash_file(&path)?,
                        path,
                    })
                })
                .collect::<Result<Vec<_>>>();
            match retained.and_then(|outputs| {
                self.started.job.verify_inputs()?;
                Ok(outputs)
            }) {
                Ok(outputs) => outcome.outputs = outputs,
                Err(error) => {
                    outcome.command_status = EvaluationStatus::Failed;
                    outcome.reason = Some(format!("output/input integrity failure: {error:#}"));
                }
            }
        }
        self.drained = true;
        // Cgroup directories hold counters: leave them for receipt recovery,
        // never remove source/evidence directories as routine pool cleanup.
        Ok(outcome)
    }
}
impl Drop for ProcessFit {
    fn drop(&mut self) {
        if !self.drained {
            let _ = control_write(&self.started.cgroup, "cgroup.kill", "1");
            if let Some(mut child) = self.child.take() {
                if self.status.is_none() {
                    let _ = killpg(Pid::from_raw(child.id() as i32), Signal::SIGKILL);
                }
                let _ = child.kill();
                if !matches!(child.try_wait(), Ok(Some(_))) {
                    // Own the eventual reap without blocking cleanup on D-state.
                    thread::spawn(move || {
                        let _ = child.wait();
                    });
                }
            }
        }
    }
}

fn capture(mut stream: impl Read + Send + 'static) -> thread::JoinHandle<std::io::Result<Vec<u8>>> {
    thread::spawn(move || {
        let mut kept = Vec::new();
        let mut buffer = [0_u8; 8192];
        loop {
            let n = stream.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            let retain = n.min(LOG_LIMIT.saturating_sub(kept.len()));
            kept.extend_from_slice(&buffer[..retain]); // Continue draining discarded output.
        }
        Ok(kept)
    })
}
fn drain_cgroup(group: &Path) -> Result<()> {
    control_write(group, "cgroup.kill", "1")?;
    let deadline = Instant::now() + Duration::from_millis(DRAIN_MS);
    while named_counter(group, "cgroup.events", "populated")? != 0 {
        ensure!(
            Instant::now() < deadline,
            "fit cgroup did not drain after SIGKILL"
        );
        thread::sleep(Duration::from_millis(POLL_MS));
    }
    Ok(())
}
fn sampled_rss(group: &Path) -> Result<Option<u64>> {
    let mut total = 0_u64;
    let mut sampled = false;
    for pid in control_text(group, "cgroup.procs")?.split_whitespace() {
        let pid: u32 = pid.parse()?;
        let Ok(status) = fs::read_to_string(format!("/proc/{pid}/status")) else {
            continue;
        };
        if let Some(kib) = status
            .lines()
            .find_map(|line| line.strip_prefix("VmRSS:"))
            .and_then(|v| v.split_whitespace().next())
            .and_then(|v| v.parse::<u64>().ok())
        {
            total = total
                .checked_add(kib.checked_mul(1024).context("RSS overflow")?)
                .context("RSS overflow")?;
            sampled = true;
        }
    }
    Ok(sampled.then_some(total))
}
fn now_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
fn control_text(group: &Path, name: &str) -> Result<String> {
    Ok(fs::read_to_string(group.join(name))?.trim().to_owned())
}
fn control_write(group: &Path, name: &str, value: &str) -> Result<()> {
    OpenOptions::new()
        .write(true)
        .open(group.join(name))?
        .write_all(value.as_bytes())?;
    Ok(())
}
fn named_counter(group: &Path, name: &str, key: &str) -> Result<u64> {
    control_text(group, name)?
        .lines()
        .find_map(|line| {
            line.split_once(' ')
                .filter(|(k, _)| *k == key)
                .map(|(_, v)| v.parse::<u64>())
        })
        .context("counter absent")?
        .map_err(Into::into)
}
fn verify_limit(group: &Path, name: &str, maximum: u64) -> Result<()> {
    let actual: u64 = control_text(group, name)?
        .parse()
        .context("unbounded resource control")?;
    ensure!(
        actual > 0 && actual <= maximum,
        "resource control exceeds allowance"
    );
    Ok(())
}
fn verify_cpu(group: &Path, maximum: u64) -> Result<()> {
    let value = control_text(group, "cpu.max")?;
    let parts: Vec<_> = value.split_whitespace().collect();
    ensure!(parts.len() == 2, "invalid CPU quota");
    let quota: u64 = parts[0].parse().context("unbounded CPU quota")?;
    let period: u64 = parts[1].parse()?;
    ensure!(
        quota > 0
            && period > 0
            && quota
                <= maximum
                    .checked_mul(period)
                    .context("CPU allowance overflow")?,
        "CPU quota exceeds allowance"
    );
    Ok(())
}
fn parse_cpus(value: &str) -> Result<Vec<u32>> {
    let mut cpus = BTreeSet::new();
    for part in value.split(',') {
        let (a, b) = part.split_once('-').unwrap_or((part, part));
        let a: u32 = a.parse()?;
        let b: u32 = b.parse()?;
        ensure!(a <= b && b < 65536, "invalid CPU range");
        cpus.extend(a..=b);
    }
    Ok(cpus.into_iter().collect())
}
/// Bounded read-only host probes. Never wait indefinitely on dockerd/systemd.
fn bounded_output(command: &mut Command) -> Result<std::process::Output> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()?;
    let stdout = capture(child.stdout.take().context("probe stdout missing")?);
    let stderr = capture(child.stderr.take().context("probe stderr missing")?);
    let deadline = Instant::now() + Duration::from_secs(2);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = killpg(Pid::from_raw(child.id() as i32), Signal::SIGKILL);
            let _ = child.kill();
            let _ = child.try_wait();
            anyhow::bail!("host dependency probe exceeded2 seconds");
        }
        thread::sleep(Duration::from_millis(POLL_MS));
    };
    Ok(std::process::Output {
        status,
        stdout: stdout
            .join()
            .map_err(|_| anyhow::anyhow!("probe stdout panic"))??,
        stderr: stderr
            .join()
            .map_err(|_| anyhow::anyhow!("probe stderr panic"))??,
    })
}

/// A foreign service is never drained here. Fail closed while any other ASM
/// allocation remains populated, including the flock-release/systemd-kill gap.
fn require_no_other_asm_groups(slice: &Path, current: &Path) -> Result<()> {
    for entry in fs::read_dir(slice)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("jope-asm-") || !name.ends_with(".service") {
            continue;
        }
        let group = entry.path();
        if group == current {
            continue;
        }
        let population = named_counter(&group, "cgroup.events", "populated");
        if !group.exists() {
            continue;
        }
        ensure!(
            population.ok() == Some(0),
            "another populated or unverifiable ASM allocation exists: {}",
            group.display()
        );
    }
    Ok(())
}

fn systemd_properties(unit: &str) -> Result<BTreeMap<String, String>> {
    let output = bounded_output(Command::new("/usr/bin/systemctl").args(["--system", "show", "--no-pager", unit, "--property=Type,ExitType,KillMode,SendSIGKILL,FinalKillSignal,RemainAfterExit,Restart,OOMPolicy,RuntimeRandomizedExtraUSec,ActiveState,MainPID,ControlGroup,RuntimeMaxUSec,TimeoutStopUSec,ActiveEnterTimestampMonotonic"]))?;
    ensure!(
        output.status.success(),
        "systemd supervisor readback failed"
    );
    Ok(String::from_utf8(output.stdout)?
        .lines()
        .filter_map(|line| line.split_once('=').map(|(k, v)| (k.into(), v.into())))
        .collect())
}
fn parse_timespan(value: &str) -> Result<u64> {
    if value == "0" {
        return Ok(0);
    }
    let mut total = 0_u64;
    for part in value.split_whitespace() {
        let suffix = ["us", "µs", "μs", "ms", "min", "h", "d", "s"]
            .into_iter()
            .find(|unit| part.ends_with(unit))
            .context("unbounded/invalid systemd duration")?;
        let raw = part.strip_suffix(suffix).context("duration suffix")?;
        let (whole, fraction) = raw.split_once('.').unwrap_or((raw, ""));
        ensure!(
            !whole.is_empty()
                && whole.bytes().all(|b| b.is_ascii_digit())
                && fraction.bytes().all(|b| b.is_ascii_digit())
                && fraction.len() <= 9,
            "invalid duration number"
        );
        let unit: u64 = match suffix {
            "us" | "µs" | "μs" => 1,
            "ms" => 1000,
            "s" => 1_000_000,
            "min" => 60_000_000,
            "h" => 3_600_000_000,
            "d" => 86_400_000_000,
            _ => unreachable!(),
        };
        let whole = whole
            .parse::<u64>()?
            .checked_mul(unit)
            .context("duration overflow")?;
        let fractional = if fraction.is_empty() {
            0
        } else {
            fraction
                .parse::<u64>()?
                .checked_mul(unit)
                .context("duration overflow")?
                .div_ceil(10_u64.pow(fraction.len() as u32))
        };
        total = total
            .checked_add(whole)
            .and_then(|v| v.checked_add(fractional))
            .context("duration overflow")?;
    }
    ensure!(total > 0, "duration must be finite and positive");
    Ok(total)
}

fn regular(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute()
            && fs::canonicalize(path)? == path
            && fs::symlink_metadata(path)?.is_file(),
        "physical regular file required: {}",
        path.display()
    );
    Ok(())
}
fn hash_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        digest.update(&buffer[..size]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    regular(path)?;
    let mut bytes = Vec::new();
    File::open(path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 16 * 1024 * 1024, "journal record too large");
    Ok(serde_json::from_slice(&bytes)?)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    File::open(path.parent().context("file parent absent")?)?.sync_all()?;
    Ok(())
}
fn atomic_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let pending = path.with_extension(format!(
        "pending-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    write_new(&pending, &bytes)?;
    fs::rename(&pending, path)?;
    File::open(path.parent().context("journal parent absent")?)?.sync_all()?;
    Ok(())
}
fn immutable<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if path.exists() {
        let existing: serde_json::Value = read_json(path)?;
        ensure!(
            existing == serde_json::to_value(value)?,
            "immutable journal record changed: {}",
            path.display()
        );
        return Ok(());
    }
    atomic_json(path, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{
        EvaluationStage, Metric, Resources, EXPERIMENT_SCHEMA, REQUEST_SCHEMA,
    };
    use jope_datasets::catalog::Task;

    fn fixture_experiment() -> Experiment {
        Experiment {
            schema: EXPERIMENT_SCHEMA.into(),
            panel_sha256: "a".repeat(64),
            methods_sha256: "b".repeat(64),
            source_sha: "c".repeat(40),
            checkpoints: BTreeMap::new(),
            seed: 1,
            resources: Resources {
                workers: 2,
                cpus: 2,
                memory_bytes: 256 * 1024 * 1024,
                swap_bytes: 0,
                gpu_tasks: 0,
                threads_per_worker: 1,
                total_seconds: 120,
                reporting_seconds: 10,
                per_fit_seconds: 2,
            },
            expected_artifacts: vec!["predictions".into()],
            protocol: "synthetic-test-only".into(),
            numerical_policy: "finite-v1".into(),
        }
    }
    fn fixture_job(dataset: char, recipe: char) -> WorkerJob {
        let request = EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: fixture_experiment().identity().unwrap(),
            dataset_sha256: dataset.to_string().repeat(64),
            split_sha256: "b".repeat(64),
            fitting_rows_sha256: "c".repeat(64),
            scoring_rows_sha256: "d".repeat(64),
            fitting_rows: 3,
            scoring_rows: 2,
            recipe_sha256: recipe.to_string().repeat(64),
            evaluator_sha256: "e".repeat(64),
            checkpoint_sha256: None,
            task: Task::Regression,
            metric: Metric::Rmse,
            classes: vec![],
            fold: 0,
            stage: EvaluationStage::Screen,
            seed: 1,
            numerical_policy: "finite-v1".into(),
        };
        WorkerJob {
            directory: PathBuf::from("/private/fixture").join(request.identity().unwrap()),
            request,
            command: WorkerCommand {
                program: Artifact {
                    path: PathBuf::from("/usr/bin/false"),
                    sha256: "a".repeat(64),
                },
                args: vec![],
                inputs: vec![],
                environment: BTreeMap::new(),
            },
            outputs: vec![PathBuf::from("predictions.json")],
            arm: None,
        }
    }
    fn scratch() -> tempfile::TempDir {
        // Reuse the lane target tree; tests never create /tmp evidence.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
        fs::create_dir_all(&root).unwrap();
        tempfile::Builder::new()
            .prefix("execution-fixture-")
            .tempdir_in(fs::canonicalize(root).unwrap())
            .unwrap()
    }
    #[test]
    fn datasets_get_round_robin_admission() {
        let mut queue = FairQueue::default();
        for (dataset, recipe) in [
            ('a', 'a'),
            ('a', 'b'),
            ('a', 'c'),
            ('b', 'a'),
            ('b', 'b'),
            ('c', 'a'),
        ] {
            queue.push(dataset.to_string(), fixture_job(dataset, recipe));
        }
        let mut order = Vec::new();
        while let Some(job) = queue.pop() {
            order.push(job.request.dataset_sha256.chars().next().unwrap());
        }
        assert_eq!(order, ['a', 'b', 'c', 'a', 'b', 'a']);
    }
    #[test]
    fn job_identity_binds_argv_inputs_environment_and_arm() {
        let original = fixture_job('a', 'a');
        let hash = original.identity().unwrap();
        let mut changed = original.clone();
        changed.command.args.push("--different".into());
        assert_ne!(hash, changed.identity().unwrap());
        let mut changed = original.clone();
        changed
            .command
            .environment
            .insert("PATH".into(), "/private/bin".into());
        assert_ne!(hash, changed.identity().unwrap());
        let mut changed = original.clone();
        changed.command.inputs.push(Artifact {
            path: "/private/input.json".into(),
            sha256: "b".repeat(64),
        });
        assert_ne!(hash, changed.identity().unwrap());
        let mut changed = original;
        changed.arm = Some(ArmBudget {
            id_sha256: "d".repeat(64),
            seconds: 120,
        });
        assert_ne!(hash, changed.identity().unwrap());
        changed.arm.as_mut().unwrap().seconds = 121;
        assert!(changed.identity().is_err());
    }
    #[test]
    fn refuses_output_escape_and_thread_override() {
        for output in ["../escape", "/absolute", "stdout.log", "stderr.log"] {
            let mut job = fixture_job('a', 'a');
            job.outputs = vec![output.into()];
            assert!(job.identity().is_err(), "{output}");
        }
        let mut job = fixture_job('a', 'a');
        job.command
            .environment
            .insert("OMP_NUM_THREADS".into(), "20".into());
        assert!(job.identity().is_err());
    }
    #[test]
    fn missing_accounting_serializes_as_null_and_not_started_is_explicit_zero() {
        let absent = serde_json::to_value(ObservedConsumption::default()).unwrap();
        assert!(absent["cpu_seconds"].is_null());
        assert!(absent["wall_seconds"].is_null());
        let refused = WorkerOutcome::unstarted(
            &fixture_job('a', 'a'),
            EvaluationStatus::Unavailable,
            "missing engine".into(),
        )
        .unwrap();
        assert_eq!(refused.consumption.cpu_seconds, Some(0.));
        assert!(refused
            .consumption
            .accounting_reason
            .starts_with("not_started"));
    }
    #[test]
    fn immutable_journal_refuses_deadline_reset_and_changed_identity() {
        let scratch = scratch();
        let experiment = fixture_experiment();
        let window = ExecutionWindow {
            experiment_sha256: experiment.identity().unwrap(),
            started_unix_ms: 1000,
            deadline_unix_ms: 121000,
        };
        let path = scratch.path().join("window.json");
        immutable(&path, &window).unwrap();
        immutable(&path, &window).unwrap();
        let mut extended = window.clone();
        extended.deadline_unix_ms += 1;
        assert!(immutable(&path, &extended).is_err());
        let recovered: ExecutionWindow = read_json(&fs::canonicalize(&path).unwrap()).unwrap();
        recovered.validate(&experiment).unwrap();
        assert!(!recovered.may_admit(&experiment, 111000).unwrap());
        assert!(recovered.may_admit(&experiment, 999).is_err());
    }
    #[test]
    fn unavailable_inventory_does_not_start_the_clock() {
        let scratch = scratch();
        let root = fs::canonicalize(scratch.path()).unwrap();
        let experiment = fixture_experiment();
        let supervisor = Supervisor {
            unit: "jope-asm-unavailable.service".into(),
            cgroup_root: "/not-a-cgroup".into(),
        };
        let mut pool = ExecutionPool::open(&experiment, &root.join("journal"), supervisor).unwrap();
        assert!(pool.reserve().is_err(), "missing host capability must fail");
        assert!(pool.window().is_none());
        let mut job = fixture_job('a', 'a');
        job.directory = pool
            .directory
            .join("jobs")
            .join(job.request.identity().unwrap());
        let inventory = inventory_dependencies(&[job.clone()]).unwrap();
        assert!(!inventory[0].available);
        assert!(pool.window().is_none());
        let outcomes = pool.run(vec![job], &AtomicBool::new(false)).unwrap();
        assert_eq!(outcomes[0].command_status, EvaluationStatus::Unavailable);
        assert!(pool.window().is_none());
        assert!(!pool.directory.join("window.json").exists());
    }
    #[test]
    fn cancelled_and_missing_dependency_inventory_does_not_poison_resume() {
        let scratch = scratch();
        let root = fs::canonicalize(scratch.path()).unwrap();
        let mut pool = ExecutionPool::open(
            &fixture_experiment(),
            &root.join("journal"),
            Supervisor {
                unit: "jope-asm-unavailable.service".into(),
                cgroup_root: "/not-a-cgroup".into(),
            },
        )
        .unwrap();
        let mut job = fixture_job('a', 'a');
        job.directory = pool
            .directory
            .join("jobs")
            .join(job.request.identity().unwrap());
        let id = job.identity().unwrap();
        let cancelled = pool.run(vec![job.clone()], &AtomicBool::new(true)).unwrap();
        assert_eq!(cancelled[0].command_status, EvaluationStatus::Cancelled);
        assert_eq!(cancelled[0].consumption.cpu_seconds, Some(0.));
        assert!(!pool.started_path(&id).exists() && !pool.finished_path(&id).exists());
        let unavailable = pool
            .run(vec![job.clone()], &AtomicBool::new(false))
            .unwrap();
        assert_eq!(unavailable[0].command_status, EvaluationStatus::Unavailable);
        assert!(pool.cached(&job).unwrap().is_none());
        assert!(!pool.finished_path(&id).exists() && pool.window().is_none());
    }
    #[test]
    fn resumed_prior_unit_is_unknown_and_foreign_live_group_is_not_drained() {
        let scratch = scratch();
        let root = fs::canonicalize(scratch.path()).unwrap();
        let pool = ExecutionPool::open(
            &fixture_experiment(),
            &root.join("journal"),
            Supervisor {
                unit: "jope-asm-new.service".into(),
                cgroup_root: "/sys/fs/cgroup/system.slice/jope-asm-new.service".into(),
            },
        )
        .unwrap();
        let job = fixture_job('a', 'a');
        let id = job.identity().unwrap();
        let original = Started {
            job: job.clone(),
            job_sha256: id.clone(),
            started_unix_ms: 1,
            fit_deadline_unix_ms: 1000,
            cgroup: PathBuf::from(
                "/sys/fs/cgroup/system.slice/jope-asm-absent-prior-fixture.service",
            )
            .join(format!("fit-{id}")),
        };
        immutable(&pool.started_path(&id), &original).unwrap();
        let outcome = pool.interrupted(&job, true).unwrap();
        assert_eq!(outcome.command_status, EvaluationStatus::Cancelled);
        assert!(outcome
            .reason
            .unwrap()
            .starts_with("controller_interrupted"));
        assert!(outcome.consumption.cpu_seconds.is_none());
        let foreign = root.join("foreign");
        fs::create_dir(&foreign).unwrap();
        fs::write(foreign.join("cgroup.events"), "populated 1\n").unwrap();
        let mut other = fixture_job('b', 'b');
        other.request.recipe_sha256 = "c".repeat(64);
        let other_id = other.identity().unwrap();
        immutable(
            &pool.started_path(&other_id),
            &Started {
                job: other.clone(),
                job_sha256: other_id,
                started_unix_ms: 1,
                fit_deadline_unix_ms: 1000,
                cgroup: foreign.clone(),
            },
        )
        .unwrap();
        assert!(pool
            .interrupted(&other, true)
            .unwrap()
            .reason
            .unwrap()
            .contains("recovery_not_drained"));
        assert!(!foreign.join("cgroup.kill").exists());
    }
    #[test]
    fn finished_receipt_cannot_hide_live_group_from_later_batch() {
        let scratch = scratch();
        let root = fs::canonicalize(scratch.path()).unwrap();
        let pool = ExecutionPool::open(
            &fixture_experiment(),
            &root.join("journal"),
            Supervisor {
                unit: "jope-asm-new.service".into(),
                cgroup_root: "/sys/fs/cgroup/system.slice/jope-asm-new.service".into(),
            },
        )
        .unwrap();
        let job = fixture_job('c', 'd');
        let id = job.identity().unwrap();
        let foreign = root.join("foreign");
        fs::create_dir(&foreign).unwrap();
        fs::write(foreign.join("cgroup.events"), "populated 1\n").unwrap();
        let original = Started {
            job: job.clone(),
            job_sha256: id.clone(),
            started_unix_ms: 1,
            fit_deadline_unix_ms: 100,
            cgroup: foreign.clone(),
        };
        immutable(&pool.started_path(&id), &original).unwrap();
        immutable(&pool.cleanup_path(&id), &original).unwrap();
        let interrupted = pool.interrupted(&job, true).unwrap();
        immutable(&pool.finished_path(&id), &interrupted).unwrap();
        assert!(pool.cached(&job).unwrap().is_some());
        assert!(pool.reconcile_cleanup().unwrap().is_some());
        assert!(pool.cleanup_path(&id).exists());
        assert!(!foreign.join("cgroup.kill").exists());
        fs::write(foreign.join("cgroup.events"), "populated 0\n").unwrap();
        assert!(pool.reconcile_cleanup().unwrap().is_none());
        assert!(!pool.cleanup_path(&id).exists());
    }
    #[test]
    fn foreign_asm_crash_gap_blocks_without_drain() {
        let scratch = scratch();
        let root = fs::canonicalize(scratch.path()).unwrap();
        let current = root.join("jope-asm-current.service");
        let foreign = root.join("jope-asm-old.service");
        fs::create_dir(&current).unwrap();
        fs::write(current.join("cgroup.events"), "populated 1\n").unwrap();
        fs::create_dir(&foreign).unwrap();
        fs::write(foreign.join("cgroup.events"), "populated 1\n").unwrap();
        assert!(require_no_other_asm_groups(&root, &current).is_err());
        assert!(!foreign.join("cgroup.kill").exists());
        fs::write(foreign.join("cgroup.events"), "populated 0\n").unwrap();
        require_no_other_asm_groups(&root, &current).unwrap();
        fs::remove_file(foreign.join("cgroup.events")).unwrap();
        assert!(require_no_other_asm_groups(&root, &current).is_err());
    }
    #[test]
    fn per_fit_allowance_cannot_exceed_thirty_seconds() {
        let scratch = scratch();
        let mut experiment = fixture_experiment();
        experiment.resources.total_seconds = 3600;
        experiment.resources.per_fit_seconds = 1200;
        let mut pool = ExecutionPool::open(
            &experiment,
            &scratch.path().join("journal"),
            Supervisor {
                unit: "unused".into(),
                cgroup_root: "/unused".into(),
            },
        )
        .unwrap();
        assert!(pool.run(vec![], &AtomicBool::new(false)).is_err());
        assert!(pool.window().is_none());
    }

    #[test]
    fn arm_window_cannot_reset_or_change_allowance() {
        let scratch = scratch();
        let root = fs::canonicalize(scratch.path()).unwrap();
        let pool = ExecutionPool::open(
            &fixture_experiment(),
            &root.join("journal"),
            Supervisor {
                unit: "unused".into(),
                cgroup_root: root,
            },
        )
        .unwrap();
        let mut arm = ArmBudget {
            id_sha256: "a".repeat(64),
            seconds: 120,
        };
        let original = pool.arm_deadline(&arm).unwrap();
        assert_eq!(pool.arm_deadline(&arm).unwrap(), original);
        arm.seconds = 119;
        assert!(pool.arm_deadline(&arm).is_err());
    }
    #[test]
    fn changed_output_cannot_be_a_cached_success() {
        let scratch = scratch();
        let path = scratch.path().join("predictions.json");
        write_new(&path, b"[1,2]").unwrap();
        let artifact = Artifact {
            path: fs::canonicalize(&path).unwrap(),
            sha256: hash_file(&path).unwrap(),
        };
        artifact.verify().unwrap();
        fs::write(&path, b"[9,9]").unwrap();
        assert!(artifact.verify().is_err());
    }
    #[test]
    fn capture_drains_after_retention_bound() {
        let bytes = vec![b'x'; LOG_LIMIT * 3];
        let retained = capture(std::io::Cursor::new(bytes))
            .join()
            .unwrap()
            .unwrap();
        assert_eq!(retained.len(), LOG_LIMIT);
    }
    #[test]
    fn cgroup_and_systemd_parsers_reject_unbounded_allowances() {
        assert_eq!(parse_cpus("2-4,4,9").unwrap(), [2, 3, 4, 9]);
        assert!(parse_cpus("4-2").is_err());
        assert!(parse_cpus("").is_err());
        assert_eq!(parse_timespan("1min 2s 3ms 4us").unwrap(), 62_003_004);
        assert_eq!(parse_timespan("1.2345678s").unwrap(), 1_234_568);
        assert_eq!(parse_timespan("2.5ms").unwrap(), 2500);
        assert!(parse_timespan("1.nan s").is_err());
        assert!(parse_timespan("infinity").is_err());
        assert_eq!(parse_timespan("0").unwrap(), 0);
    }
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
            ExecutionPool::open(&experiment, &physical.join("journal"), supervisor.clone())
                .unwrap();
        pool.reserve()
            .expect("positive host fixture requires real reservation");
        assert!(
            pool.window().is_none(),
            "reservation must not start the clock"
        );
        assert!(!physical.join("journal/window.json").exists());
        if case == "concurrency20" {
            qualify_concurrency(&mut pool, &physical);
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
        let cpu =
            parse_cpus(&control_text(&supervisor.cgroup_root, "cpuset.cpus.effective").unwrap())
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
    fn qualify_concurrency(pool: &mut ExecutionPool, physical: &Path) {
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
        println!(
            "execution_host_receipt={}",
            serde_json::json!({"case":"concurrency20","peak":20,"completed":outcomes.len(),"active_after":0})
        );
    }
}
