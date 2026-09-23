//! Read-only admission of successful fit timing from original pool journals.
//! Model/input bytes are authenticated by the caller; this independently checks
//! the actual started command, original deadlines and completed cleanup.
use anyhow::{ensure, Context, Result};
use jope_pipeline::{
    execution::{ArmBudget, WorkerJob, WorkerOutcome},
    experiment::{EvaluationStatus, ExecutionWindow, Experiment},
};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmWindow {
    id_sha256: String,
    seconds: u64,
    started_unix_ms: u64,
    deadline_unix_ms: u64,
}
fn clipped_arm(
    arm: &ArmBudget,
    stored: &ArmWindow,
    window: &ExecutionWindow,
    experiment: &Experiment,
) -> Result<u64> {
    window.validate(experiment)?;
    ensure!(
        stored.id_sha256 == arm.id_sha256
            && stored.seconds == arm.seconds
            && stored.started_unix_ms.checked_add(arm.seconds * 1000)
                == Some(stored.deadline_unix_ms)
            && stored.started_unix_ms >= window.started_unix_ms,
        "arm changed original allowance or start"
    );
    Ok(stored
        .deadline_unix_ms
        .min(window.deadline_unix_ms - experiment.resources.reporting_seconds * 1000))
}
fn original(journal: &Path, expected: &str) -> Result<(Experiment, ExecutionWindow)> {
    let experiment: Experiment =
        serde_json::from_slice(&fs::read(journal.join("experiment.json"))?)?;
    ensure!(
        experiment.identity()? == expected,
        "fit journal experiment identity changed"
    );
    let window: ExecutionWindow = serde_json::from_slice(&fs::read(journal.join("window.json"))?)?;
    window.validate(&experiment)?;
    Ok((experiment, window))
}
pub fn arm_deadline(journal: &Path, arm: &ArmBudget, experiment_sha256: &str) -> Result<u64> {
    let (experiment, window) = original(journal, experiment_sha256)?;
    let value: ArmWindow = serde_json::from_slice(&fs::read(
        journal.join("arms").join(format!("{}.json", arm.id_sha256)),
    )?)?;
    clipped_arm(arm, &value, &window, &experiment)
}
pub fn completed_fit(journal: &Path, job: &WorkerJob, outcome: &WorkerOutcome) -> Result<()> {
    let (experiment, window) = original(journal, &job.request.experiment_sha256)?;
    let id = job.identity()?;
    ensure!(
        job.directory == journal.join("jobs").join(job.request.identity()?)
            && outcome.job_sha256 == id
            && outcome.request_sha256 == job.request.identity()?
            && outcome.command_status == EvaluationStatus::Completed
            && outcome.exit_code == Some(0)
            && outcome.signal.is_none()
            && outcome.reason.is_none(),
        "fit execution identity or successful exit changed"
    );
    let stored: WorkerOutcome = serde_json::from_slice(&fs::read(
        journal.join("finished").join(format!("{id}.json")),
    )?)?;
    ensure!(
        serde_json::to_value(&stored)? == serde_json::to_value(outcome)?,
        "fit outcome differs from original finished record"
    );
    let started: serde_json::Value = serde_json::from_slice(&fs::read(
        journal.join("started").join(format!("{id}.json")),
    )?)?;
    ensure!(
        started["job_sha256"] == id && started["job"] == serde_json::to_value(job)?,
        "started fit command changed"
    );
    let start = started["started_unix_ms"]
        .as_u64()
        .context("fit start unknown")?;
    let deadline = started["fit_deadline_unix_ms"]
        .as_u64()
        .context("fit deadline unknown")?;
    let gate = outcome
        .consumption
        .gate_released_unix_ms
        .context("fit gate unknown")?;
    let exit = outcome
        .consumption
        .observed_exit_unix_ms
        .context("fit exit unknown")?;
    let mut cutoff = window.deadline_unix_ms - experiment.resources.reporting_seconds * 1000;
    if let Some(arm) = &job.arm {
        let value: ArmWindow = serde_json::from_slice(&fs::read(
            journal.join("arms").join(format!("{}.json", arm.id_sha256)),
        )?)?;
        cutoff = clipped_arm(arm, &value, &window, &experiment)?;
        ensure!(
            start >= value.started_unix_ms,
            "fit precedes original arm start"
        );
    }
    ensure!(
        start >= window.started_unix_ms
            && deadline >= start
            && deadline <= start.saturating_add(experiment.resources.per_fit_seconds * 1000)
            && deadline <= cutoff
            && gate >= start
            && exit >= gate
            && exit <= deadline,
        "completed fit escaped original fit, arm or global cutoff"
    );
    ensure!(
        !journal.join("cleanup").join(format!("{id}.json")).exists(),
        "fit group cleanup remains pending"
    );
    let group = Path::new(started["cgroup"].as_str().context("fit cgroup absent")?);
    ensure!(
        group
            .file_name()
            .is_some_and(|n| n == format!("fit-{id}").as_str())
            && group.parent().and_then(|p| p.parent())
                == Some(Path::new("/sys/fs/cgroup/system.slice")),
        "fit cgroup identity changed"
    );
    if group.exists() {
        ensure!(
            fs::read_to_string(group.join("cgroup.events"))?
                .lines()
                .any(|l| l.split_whitespace().eq(["populated", "0"])),
            "fit descendants survived"
        );
    }
    let allocation = jope_pipeline::execution::fit_memory_allocation(&experiment)?;
    let recorded: jope_pipeline::execution::FitMemoryAllocation =
        serde_json::from_slice(&fs::read(journal.join("fit-memory-policy.json"))?)?;
    ensure!(
        recorded == allocation
            && outcome.consumption.fit_memory_limit_bytes == Some(allocation.per_fit_bytes),
        "fit memory policy or actual cap changed"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arm_deadline_is_original_and_clipped_by_reporting_reserve() {
        let experiment = Experiment {
            schema: jope_pipeline::experiment::EXPERIMENT_SCHEMA.into(),
            panel_sha256: "a".repeat(64),
            methods_sha256: "b".repeat(64),
            source_sha: "c".repeat(40),
            checkpoints: Default::default(),
            seed: 1,
            resources: jope_pipeline::experiment::Resources {
                workers: 1,
                cpus: 1,
                memory_bytes: 1 << 30,
                swap_bytes: 0,
                gpu_tasks: 0,
                threads_per_worker: 1,
                total_seconds: 180,
                reporting_seconds: 30,
                per_fit_seconds: 30,
            },
            expected_artifacts: vec!["output".into()],
            protocol: "fixture".into(),
            numerical_policy: "fixture".into(),
        };
        let window = ExecutionWindow {
            experiment_sha256: experiment.identity().unwrap(),
            started_unix_ms: 1000,
            deadline_unix_ms: 181000,
        };
        let arm = ArmBudget {
            id_sha256: "d".repeat(64),
            seconds: 120,
        };
        let mut stored = ArmWindow {
            id_sha256: arm.id_sha256.clone(),
            seconds: 120,
            started_unix_ms: 61000,
            deadline_unix_ms: 181000,
        };
        assert_eq!(
            clipped_arm(&arm, &stored, &window, &experiment).unwrap(),
            151000
        );
        stored.deadline_unix_ms += 1;
        assert!(clipped_arm(&arm, &stored, &window, &experiment).is_err());
        stored.deadline_unix_ms = 120999;
        stored.started_unix_ms = 999;
        assert!(clipped_arm(&arm, &stored, &window, &experiment).is_err());
    }
}
