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
    // SAFETY: the hook only makes async-signal-safe prctl and getppid calls and builds its
    // io::Errors from errno codes without allocating; `parent` is a captured u32.
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

/// A rank may not have published its first heartbeat yet. Once a receipt exists,
/// read and parse failures are errors rather than an empty progress record.
fn read_rank_receipt(path: &Path) -> Result<Option<serde_json::Value>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("read rank receipt {}", path.display()))
        }
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("parse rank receipt {}", path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrainingRankProgress {
    step: u64,
    timestamp: u64,
}

fn read_training_rank_progress(path: &Path) -> Result<Option<TrainingRankProgress>> {
    let Some(receipt) = read_rank_receipt(path)? else {
        return Ok(None);
    };
    let step = receipt["step"]
        .as_u64()
        .with_context(|| format!("rank receipt step missing or invalid: {}", path.display()))?;
    let timestamp = receipt["timestamp"].as_u64().with_context(|| {
        format!(
            "rank receipt timestamp missing or invalid: {}",
            path.display()
        )
    })?;
    Ok(Some(TrainingRankProgress { step, timestamp }))
}

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
    #[cfg(feature = "neural")]
    if let Ok(fd) = std::env::var("JOPE_GPU_WORKER_LEASE_FD") {
        use std::os::{fd::FromRawFd, unix::fs::MetadataExt};
        ensure!(
            std::env::var_os("JOPE_GPU_WORKER_ACTIVE").is_some(),
            "GPU lease bypass outside resident worker"
        );
        // SAFETY: dup takes a plain integer and never affects the original descriptor; a number
        // that is not open makes it return -1, which is checked below.
        let copied = unsafe { libc::dup(fd.parse()?) };
        ensure!(copied >= 0, "resident task lost its GPU lease");
        // SAFETY: dup just returned `copied` as a fresh nonnegative descriptor no one else owns;
        // File takes sole ownership and closes it once, leaving the worker's original open.
        let lease = unsafe { fs::File::from_raw_fd(copied) };
        let actual = lease.metadata()?;
        let expected = root.join("runs/encoding-service.lock").metadata()?;
        ensure!(
            actual.dev() == expected.dev()
                && actual.ino() == expected.ino()
                && lease.try_lock().is_ok(),
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
        if lease.try_lock().is_ok() {
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
                // SAFETY: kill takes only integers and has no memory-safety preconditions; the
                // rank is unreaped, so the group id it took with setpgid(0, 0) is not reused.
                unsafe {
                    libc::kill(-(child.id() as i32), libc::SIGKILL);
                }
                let _ = child.wait();
            }
        }
    }
}

/// Topology, update window and initialization for a training request.
#[derive(Clone, Copy)]
pub struct TrainingOptions<'a> {
    pub world: usize,
    pub micro: usize,
    pub steps: usize,
    pub checkpoint_steps: usize,
    pub resume: bool,
    pub fail_rank: Option<usize>,
    pub initialize: Option<&'a Path>,
}

/// Own all GPU ranks under a lease shared with encoding services. A failed rank
/// restarts the whole communicator from the last atomically published checkpoint.
pub fn train(
    root: &Path,
    corpus: &Path,
    output: &Path,
    options: TrainingOptions<'_>,
) -> Result<()> {
    train_controlled(root, corpus, output, options, None, None)
}

pub fn train_with_policy(
    root: &Path,
    corpus: &Path,
    output: &Path,
    options: TrainingOptions<'_>,
    policy: Option<&Path>,
) -> Result<()> {
    train_controlled(root, corpus, output, options, None, policy)
}

mod probe;
mod proposal;
mod training;
mod training_rank;

pub use probe::{
    probe, probe_accumulation, training_probe, verify_controller_gpu_grant,
    verify_participant_gpu_grant,
};
pub use proposal::{propose, propose_with_generation, ProposalOptions};
pub(crate) use training::train_controlled;
use training_rank::{training_step_window, verify_training_rank};

#[cfg(test)]
#[path = "controller/campaign_deadline_tests.rs"]
mod campaign_deadline_tests;
#[cfg(test)]
#[path = "controller/lifetime_tests.rs"]
mod lifetime_tests;
#[cfg(test)]
#[path = "controller/proposal_protocol_tests.rs"]
mod proposal_protocol_tests;
#[cfg(test)]
#[path = "controller/training_window_tests.rs"]
mod training_window_tests;
