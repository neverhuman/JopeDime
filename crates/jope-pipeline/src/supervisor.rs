//! Linux service launch shared by bounded benchmark and demo front doors.
//! Service changes are limited to the newly created ASM unit. Existing runners
//! are never stopped here; the execution pool verifies exclusive allocation.
use crate::{
    execution::{Artifact, Supervisor},
    experiment::{ExecutionWindow, Experiment},
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct LaunchResult {
    pub unit: String,
    pub exit_code: Option<i32>,
    pub original_window: Option<ExecutionWindow>,
    pub reason: Option<String>,
}

pub fn cancellation() -> Result<Arc<AtomicBool>> {
    let flag = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGINT, flag.clone())?;
    signal_hook::flag::register(signal_hook::consts::SIGTERM, flag.clone())?;
    Ok(flag)
}

pub fn for_unit(unit: &str) -> Result<Supervisor> {
    ensure!(
        unit.starts_with("jope-asm-")
            && unit.ends_with(".service")
            && unit
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.')),
        "invalid ASM supervisor unit"
    );
    Ok(Supervisor {
        unit: unit.into(),
        cgroup_root: PathBuf::from("/sys/fs/cgroup/system.slice").join(unit),
    })
}

fn now_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

fn original_window(experiment: &Experiment, journal: &Path) -> Result<Option<ExecutionWindow>> {
    let path = journal.join("window.json");
    if !path.try_exists()? {
        return Ok(None);
    }
    let window: ExecutionWindow = serde_json::from_slice(&fs::read(path)?)?;
    window.validate(experiment)?;
    Ok(Some(window))
}

fn remaining_ms(
    experiment: &Experiment,
    window: Option<&ExecutionWindow>,
    now: u64,
) -> Result<u64> {
    experiment.validate()?;
    if let Some(window) = window {
        window.validate(experiment)?;
        ensure!(
            now >= window.started_unix_ms,
            "clock precedes original admission"
        );
        Ok(window.deadline_unix_ms.saturating_sub(now))
    } else {
        Ok(experiment.resources.total_seconds * 1000)
    }
}

fn cpus(status: &str, count: usize) -> Result<String> {
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .context("allowed CPU list unavailable")?
        .trim();
    let mut cpus = std::collections::BTreeSet::new();
    for item in list.split(',') {
        let (start, end) = if let Some((a, b)) = item.split_once('-') {
            (a.parse::<u32>()?, b.parse::<u32>()?)
        } else {
            let cpu = item.parse::<u32>()?;
            (cpu, cpu)
        };
        ensure!(start <= end && end - start < 65536, "invalid CPU range");
        cpus.extend(start..=end);
    }
    ensure!(
        count > 0 && cpus.len() >= count,
        "fewer than requested allowed CPUs"
    );
    Ok(cpus
        .into_iter()
        .take(count)
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(","))
}

fn system_command(program: &str) -> Command {
    if nix::unistd::Uid::effective().is_root() {
        Command::new(program)
    } else {
        let mut command = Command::new("sudo");
        command.args(["-n", program]);
        command
    }
}

/// `arguments` identifies the public command with absolute artifact paths; this
/// function supplies the internal supervisor argument. It never grants a fresh
/// window to a resumed journal. The controller itself becomes systemd MainPID.
pub fn launch(
    experiment: &Experiment,
    journal: &Path,
    program: &Artifact,
    arguments: &[String],
) -> Result<LaunchResult> {
    experiment.validate()?;
    program.verify()?;
    ensure!(
        journal.is_absolute() && !journal.starts_with("/tmp"),
        "journal requires durable absolute path"
    );
    fs::create_dir_all(journal)?;
    let journal = fs::canonicalize(journal)?;
    ensure!(!journal.starts_with("/tmp"), "journal resolves under /tmp");
    ensure!(
        !arguments
            .iter()
            .any(|a| a == "--supervisor-unit" || a.starts_with("--supervisor-unit=")),
        "supervisor argument belongs to launcher"
    );
    let invocation = fs::read_to_string("/proc/sys/kernel/random/uuid")?;
    let invocation = invocation.trim();
    ensure!(
        invocation.len() == 36
            && invocation
                .bytes()
                .all(|b| b.is_ascii_hexdigit() || b == b'-'),
        "invalid kernel invocation nonce"
    );
    let unit = format!("jope-asm-{}-{invocation}.service", experiment.identity()?);
    let previous = original_window(experiment, &journal)?;
    let remaining = remaining_ms(experiment, previous.as_ref(), now_ms()?)?;
    if remaining <= experiment.resources.reporting_seconds * 1000 || remaining <= 2000 {
        return Ok(LaunchResult {
            unit,
            exit_code: None,
            original_window: previous,
            reason: Some("original execution deadline exhausted; report retained results".into()),
        });
    }
    let allowed = cpus(
        &fs::read_to_string("/proc/self/status")?,
        experiment.resources.cpus as usize,
    )?;
    let user = nix::unistd::Uid::current().to_string();
    let tmp = journal.join("controller-tmp");
    fs::create_dir_all(&tmp)?;
    // Keep the one-second SIGKILL grace and one-second launch/readback margin
    // inside the original global window. Fits retain their earlier reporting
    // cutoff; the service no longer sends SIGTERM before that fit cutoff.
    let runtime_ms = remaining - 2000;
    let properties = [
        "Type=exec".into(),
        "ExitType=main".into(),
        format!("User={user}"),
        "Restart=no".into(),
        "OOMPolicy=continue".into(),
        "RuntimeRandomizedExtraSec=0".into(),
        "RemainAfterExit=no".into(),
        "KillMode=control-group".into(),
        "SendSIGKILL=yes".into(),
        "FinalKillSignal=SIGKILL".into(),
        "TimeoutStopSec=1s".into(),
        format!("RuntimeMaxSec={runtime_ms}ms"),
        "Delegate=cpu cpuset memory".into(),
        "DelegateSubgroup=controller".into(),
        format!("CPUQuota={}%", experiment.resources.cpus as u32 * 100),
        format!("AllowedCPUs={allowed}"),
        format!("MemoryMax={}", experiment.resources.memory_bytes),
        "MemorySwapMax=0".into(),
    ];
    let mut command = system_command("systemd-run");
    command.args(["--wait", "--pipe", "--collect", "--unit", &unit]);
    for property in &properties {
        command.arg(format!("--property={property}"));
    }
    command.arg(format!("--setenv=TMPDIR={}", tmp.display()));
    command
        .arg("--setenv=OMP_NUM_THREADS=1")
        .arg("--setenv=OPENBLAS_NUM_THREADS=1")
        .arg("--setenv=MKL_NUM_THREADS=1")
        .arg("--setenv=RAYON_NUM_THREADS=1");
    command
        .arg(&program.path)
        .args(arguments)
        .args(["--supervisor-unit", &unit]);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let cancellation = cancellation()?;
    let launched_at = now_ms()?;
    let mut child = command
        .spawn()
        .context("launch bounded service (requires noninteractive systemd permission)")?;
    let mut cancelled = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if cancellation.load(Ordering::Acquire) && !cancelled {
            cancelled = true;
            // Stop invokes the unit's one-second grace then cgroup-wide SIGKILL.
            // It can affect only the invocation-unique unit created above.
            let stopped = system_command("systemctl")
                .args(["stop", "--no-block", &unit])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .status()?;
            ensure!(
                stopped.success(),
                "unable to request cancellation of {unit}"
            );
        }
        thread::sleep(Duration::from_millis(50));
    };
    let finished_at = now_ms()?;
    let original = original_window(experiment, &journal)?;
    let execution_cutoff = original
        .as_ref()
        .map(|window| window.deadline_unix_ms - experiment.resources.reporting_seconds * 1000);
    let reporting_wall_seconds = execution_cutoff
        .map(|cutoff| finished_at.saturating_sub(cutoff.max(launched_at)) as f64 / 1000.);
    let result = LaunchResult {
        unit: unit.clone(),
        exit_code: status.code(),
        original_window: original,
        reason: (!status.success()).then(|| {
            if cancelled {
                "cancelled bounded service".into()
            } else if execution_cutoff.is_some_and(|cutoff| finished_at >= cutoff) {
                format!("original execution cutoff reached; bounded service exited with {status}")
            } else {
                format!("bounded service exited with {status}")
            }
        }),
    };
    let receipt = serde_json::json!({"schema":"jope.supervisor-launch/v1",
        "experiment_sha256":experiment.identity()?,"program":program,"arguments":arguments,
        "properties":properties,"started_unix_ms":launched_at,"finished_unix_ms":finished_at,
        "execution_cutoff_unix_ms":execution_cutoff,"reporting_wall_seconds":reporting_wall_seconds,
        "reporting_accounting":"wall after original fit cutoff; null if no original window was admitted",
        "result":result});
    // Every invocation gets a distinct append-only receipt. Original numerical
    // admission lives in window.json and is never rewritten by this launcher.
    let path = journal.join(format!("launch-{launched_at}-{}.json", std::process::id()));
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
    file.sync_all()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resume_runtime_uses_original_global_deadline_preserving_fit_cutoff() {
        let experiment = Experiment {
            schema: crate::experiment::EXPERIMENT_SCHEMA.into(),
            panel_sha256: "a".repeat(64),
            methods_sha256: "b".repeat(64),
            source_sha: "c".repeat(40),
            checkpoints: Default::default(),
            seed: 42,
            resources: crate::experiment::Resources {
                workers: 1,
                cpus: 1,
                memory_bytes: 32 << 20,
                swap_bytes: 0,
                gpu_tasks: 0,
                threads_per_worker: 1,
                total_seconds: 120,
                reporting_seconds: 5,
                per_fit_seconds: 30,
            },
            expected_artifacts: vec!["finished".into()],
            protocol: "synthetic-supervisor-test/v1".into(),
            numerical_policy: "synthetic-fixture".into(),
        };
        let mut window = ExecutionWindow {
            experiment_sha256: experiment.identity().unwrap(),
            started_unix_ms: 1000,
            deadline_unix_ms: 121000,
        };
        assert_eq!(remaining_ms(&experiment, None, 61000).unwrap(), 120000);
        assert_eq!(
            remaining_ms(&experiment, Some(&window), 61000).unwrap(),
            60000
        );
        assert_eq!(
            remaining_ms(&experiment, Some(&window), 116000).unwrap(),
            5000
        );
        assert!(!window.may_admit(&experiment, 116000).unwrap());
        assert_eq!(remaining_ms(&experiment, Some(&window), 121000).unwrap(), 0);
        assert!(remaining_ms(&experiment, Some(&window), 999).is_err());
        window.deadline_unix_ms += 1;
        assert!(remaining_ms(&experiment, Some(&window), 61000).is_err());
    }
    #[test]
    fn uses_allowed_cpu_ids_and_rejects_insufficient_capacity() {
        assert_eq!(
            cpus("Cpus_allowed_list:\t4-6,9,15-16\n", 4).unwrap(),
            "4,5,6,9"
        );
        assert!(cpus("Cpus_allowed_list:\t4-6\n", 4).is_err());
        assert!(cpus("Cpus_allowed_list:\t6-4\n", 1).is_err());
        assert!(cpus("", 1).is_err());
    }
    #[test]
    fn supervisor_paths_cannot_escape_system_slice() {
        assert!(for_unit("jope-asm-abc.service").is_ok());
        for value in [
            "other.service",
            "jope-asm-../bad.service",
            "jope-asm-a/b.service",
        ] {
            assert!(for_unit(value).is_err());
        }
    }
}
