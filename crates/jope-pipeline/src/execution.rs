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
fn pre_gate_failure(started: &Started, now: u64, reason: String) -> Result<WorkerOutcome> {
    let expired = now >= started.fit_deadline_unix_ms;
    let mut outcome = WorkerOutcome::unstarted(
        &started.job,
        if expired {
            EvaluationStatus::BudgetExhausted
        } else {
            EvaluationStatus::Unavailable
        },
        if expired {
            format!("original cutoff reached before gate release: {reason}")
        } else {
            reason
        },
    )?;
    outcome.consumption = ObservedConsumption {
        accounting_reason: "launch failed before gate release; gate/setup CPU unavailable".into(),
        ..Default::default()
    };
    Ok(outcome)
}

fn pre_gate_cancellation(started: &Started, now: u64) -> Result<WorkerOutcome> {
    let mut outcome = pre_gate_failure(started, now, "cancelled before gate release".into())?;
    if outcome.command_status != EvaluationStatus::BudgetExhausted {
        outcome.command_status = EvaluationStatus::Cancelled;
    }
    Ok(outcome)
}

/// Independent owner of each Child and deadline. Completion hashing can delay
/// reclamation of this slot, but cannot delay another fit's deadline or exit
/// observation. A local stop flag makes error-path Drop cancel only this fit.
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
            // Reap the killed child: try_wait right after SIGKILL usually saw it still running
            // and left a zombie for the controller's lifetime. The reap is bounded too, so a
            // child stuck in uninterruptible sleep cannot hold the caller (or HOST_LOCK). The
            // capture threads are not joined, so a descendant that left the process group
            // cannot hold this probe open.
            let reap_deadline = Instant::now() + Duration::from_secs(1);
            while matches!(child.try_wait(), Ok(None)) && Instant::now() < reap_deadline {
                thread::sleep(Duration::from_millis(POLL_MS));
            }
            anyhow::bail!("host dependency probe exceeded 2 seconds");
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

mod pool_admission;
mod pool_recovery;
mod pool_run;
mod processes;
mod queue;
mod supervisor;
mod types;

use processes::Running;
use queue::{ActiveFits, FairQueue};
pub use supervisor::Supervisor;
#[cfg(all(test, feature = "neural-search"))]
pub(crate) use types::verification_hook;
pub use types::*;
use types::{DRAIN_MS, LOG_LIMIT, POLL_MS, THREAD_VARS};

#[cfg(test)]
mod tests;
