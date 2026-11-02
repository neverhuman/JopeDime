use super::*;
use crate::{
    execution::{ObservedConsumption, WorkerCommand},
    experiment::{
        EvaluationRequest, EvaluationStage, Metric, Resources, EXPERIMENT_SCHEMA, REQUEST_SCHEMA,
    },
};
use std::{collections::BTreeMap, path::PathBuf};

// Synthetic receipt-validation fixture only. No executable is launched and
// no encoder/model/cgroup qualification is claimed by these tests.
#[allow(clippy::type_complexity)]
fn fixture() -> anyhow::Result<(
    tempfile::TempDir,
    WorkerJob,
    WorkerOutcome,
    Experiment,
    ExecutionWindow,
    Artifact,
    Artifact,
)> {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"));
    let scratch = target.join("tmp");
    std::fs::create_dir_all(&scratch)?;
    let temp = tempfile::Builder::new()
        .prefix("context-receipt-fixture-")
        .tempdir_in(scratch)?;
    let root = std::fs::canonicalize(temp.path())?;
    let experiment = Experiment {
        schema: EXPERIMENT_SCHEMA.into(),
        panel_sha256: "a".repeat(64),
        methods_sha256: "b".repeat(64),
        source_sha: "c".repeat(40),
        checkpoints: BTreeMap::new(),
        seed: 1,
        resources: Resources {
            workers: 2,
            cpus: 2,
            memory_bytes: 256 << 20,
            swap_bytes: 0,
            gpu_tasks: 0,
            threads_per_worker: 1,
            total_seconds: 120,
            reporting_seconds: 10,
            per_fit_seconds: 30,
        },
        expected_artifacts: vec!["output.json".into()],
        protocol: "synthetic-receipt-validation-only".into(),
        numerical_policy: "finite-v1".into(),
    };
    let request = EvaluationRequest {
        schema: REQUEST_SCHEMA.into(),
        experiment_sha256: experiment.identity()?,
        dataset_sha256: "d".repeat(64),
        split_sha256: "e".repeat(64),
        fitting_rows_sha256: "f".repeat(64),
        scoring_rows_sha256: "1".repeat(64),
        fitting_rows: 10,
        scoring_rows: 1,
        recipe_sha256: "2".repeat(64),
        evaluator_sha256: "3".repeat(64),
        checkpoint_sha256: None,
        task: jope_datasets::catalog::Task::Regression,
        metric: Metric::Rmse,
        classes: vec![],
        fold: 0,
        stage: EvaluationStage::Diagnostic,
        seed: 1,
        numerical_policy: "finite-v1".into(),
    };
    let directory = root.join("jobs").join(request.identity()?);
    std::fs::create_dir_all(&directory)?;
    let file = |path: PathBuf, bytes: &[u8]| -> anyhow::Result<Artifact> {
        std::fs::write(&path, bytes)?;
        Ok(Artifact {
            path,
            sha256: sha256_hex(bytes),
        })
    };
    let input = file(root.join("input.json"), b"synthetic input")?;
    let output = file(directory.join("output.json"), b"synthetic output")?;
    let program = file(
        root.join("not-an-executable"),
        b"synthetic receipt fixture; never executed",
    )?;
    let job = WorkerJob {
        request,
        command: WorkerCommand {
            program,
            args: vec![],
            inputs: vec![input.clone()],
            environment: BTreeMap::new(),
        },
        directory,
        outputs: vec!["output.json".into()],
        arm: None,
    };
    let outcome = WorkerOutcome {
        schema: "jope.worker-execution/v2".into(),
        job_sha256: job.identity()?,
        request_sha256: job.request.identity()?,
        command_status: EvaluationStatus::Completed,
        reason: None,
        exit_code: Some(0),
        signal: None,
        outputs: vec![output.clone()],
        stdout: None,
        stderr: None,
        consumption: ObservedConsumption {
            fit_memory_limit_bytes: Some(
                crate::execution::fit_memory_allocation(&experiment)?.per_fit_bytes,
            ),
            gate_released_unix_ms: Some(2_000),
            observed_exit_unix_ms: Some(3_000),
            wall_seconds: Some(1.),
            cpu_seconds: Some(0.25),
            cgroup_peak_memory_bytes: Some(4096),
            accounting_reason: "synthetic fixture; not execution evidence".into(),
            ..ObservedConsumption::default()
        },
    };
    let window = ExecutionWindow {
        experiment_sha256: experiment.identity()?,
        started_unix_ms: 1_000,
        deadline_unix_ms: 121_000,
    };
    Ok((temp, job, outcome, experiment, window, input, output))
}
#[test]
fn synthetic_attestation_requires_exit_zero_and_original_deadline() -> anyhow::Result<()> {
    let (_temp, job, good, experiment, window, input, output) = fixture()?;
    execution_attestation(&job, &good, &experiment, &window, &input, &output)?;
    let mut changed = good.clone();
    changed.exit_code = Some(69);
    assert!(execution_attestation(&job, &changed, &experiment, &window, &input, &output).is_err());
    let mut changed = good.clone();
    changed.reason = Some("cleanup incomplete".into());
    assert!(execution_attestation(&job, &changed, &experiment, &window, &input, &output).is_err());
    let mut changed = good;
    changed.consumption.observed_exit_unix_ms = Some(112_000);
    assert!(execution_attestation(&job, &changed, &experiment, &window, &input, &output).is_err());
    Ok(())
}
#[test]
fn synthetic_attestation_rejects_unknown_counters_and_unread_memory_limit() -> anyhow::Result<()> {
    let (_temp, job, good, experiment, window, input, output) = fixture()?;
    let admitted = execution_attestation(&job, &good, &experiment, &window, &input, &output)?;
    assert_eq!(
        admitted.memory_max,
        crate::execution::fit_memory_allocation(&experiment)?.per_fit_bytes
    );
    let mut changed = good.clone();
    changed.consumption.cpu_seconds = None;
    assert!(execution_attestation(&job, &changed, &experiment, &window, &input, &output).is_err());
    let mut changed = good.clone();
    changed.consumption.fit_memory_limit_bytes = None;
    assert!(execution_attestation(&job, &changed, &experiment, &window, &input, &output).is_err());
    let mut changed = good;
    changed.outputs[0].sha256 = "0".repeat(64);
    assert!(execution_attestation(&job, &changed, &experiment, &window, &input, &output).is_err());
    Ok(())
}
/// PR #114 review round 1: teacher admission (`jope_learning::eligibility`) rebuilds the
/// controller's attestation of a context execution, and so the admission pins' digest of
/// it, from the retained receipts: the program is the job's, the input and output are the
/// handoff's, the limits are the experiment's and the elapsed time is the supervisor's wall
/// time. (`ExecutionProof::attest` then pins the supervisor receipt as the outcome file's
/// digest, which teacher admission retains.) This pins every field it derives, producer side.
#[test]
fn the_attestation_is_derivable_from_the_retained_receipts() -> anyhow::Result<()> {
    let (_temp, job, outcome, experiment, window, input, output) = fixture()?;
    let a = execution_attestation(&job, &outcome, &experiment, &window, &input, &output)?;
    assert_eq!(a.schema, "jope.cpu-context-execution/controller-v1");
    assert_eq!(a.program_sha256, job.command.program.sha256);
    assert_eq!(
        (a.input_sha256, a.output_sha256),
        (input.sha256, output.sha256)
    );
    assert!(a.completed);
    assert_eq!((a.cpu_limit, a.threads), (1, 1));
    assert_eq!(
        a.memory_max,
        crate::execution::fit_memory_allocation(&experiment)?.per_fit_bytes
    );
    assert_eq!(a.swap_max, experiment.resources.swap_bytes);
    assert_eq!(Some(a.elapsed_seconds), outcome.consumption.wall_seconds);
    assert_eq!(
        a.deadline_seconds,
        experiment.resources.per_fit_seconds as f64
    );
    Ok(())
}
