use super::accounting::accounting_for_experiment;
use super::*;
use jope_pipeline::{
    execution::{ObservedConsumption, WorkerCommand, WorkerJob, EXECUTION_SCHEMA},
    experiment::{EvaluationRequest, Metric, REQUEST_SCHEMA},
};
use std::path::PathBuf;

fn scratch() -> tempfile::TempDir {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
    fs::create_dir_all(&root).unwrap();
    let directory = tempfile::Builder::new()
        .prefix("accounting-union-")
        .tempdir_in(fs::canonicalize(root).unwrap())
        .unwrap();
    fs::create_dir(directory.path().join("started")).unwrap();
    fs::create_dir(directory.path().join("finished")).unwrap();
    directory
}
fn job(journal: &Path, seed: u64, stage: EvaluationStage) -> WorkerJob {
    let request = EvaluationRequest {
        schema: REQUEST_SCHEMA.into(),
        experiment_sha256: "a".repeat(64),
        dataset_sha256: "b".repeat(64),
        split_sha256: "c".repeat(64),
        fitting_rows_sha256: "d".repeat(64),
        scoring_rows_sha256: "e".repeat(64),
        fitting_rows: 10,
        scoring_rows: 2,
        recipe_sha256: "f".repeat(64),
        evaluator_sha256: "1".repeat(64),
        checkpoint_sha256: None,
        task: jope_datasets::catalog::Task::Regression,
        metric: Metric::Rmse,
        classes: vec![],
        fold: 0,
        stage,
        seed,
        numerical_policy: "synthetic-metadata-only".into(),
    };
    WorkerJob {
        directory: journal.join("jobs").join(request.identity().unwrap()),
        request,
        command: WorkerCommand {
            program: jope_pipeline::execution::Artifact {
                path: "/synthetic/not-executed".into(),
                sha256: "2".repeat(64),
            },
            args: vec![],
            inputs: vec![],
            environment: Default::default(),
        },
        outputs: vec!["output.json".into()],
        arm: None,
    }
}
fn started(journal: &Path, job: &WorkerJob) -> PathBuf {
    let id = job.identity().unwrap();
    let path = journal.join("started").join(format!("{id}.json"));
    fs::write(
        &path,
        serde_json::to_vec(&json!({
            "job_sha256":id,"job":job,"started_unix_ms":100,"fit_deadline_unix_ms":120,
            "cgroup":"/synthetic/not-executed"
        }))
        .unwrap(),
    )
    .unwrap();
    path
}
fn failed(job: &WorkerJob) -> WorkerOutcome {
    WorkerOutcome {
        schema: EXECUTION_SCHEMA.into(),
        job_sha256: job.identity().unwrap(),
        request_sha256: job.request.identity().unwrap(),
        command_status: EvaluationStatus::Failed,
        reason: Some("synthetic retained failure".into()),
        exit_code: Some(1),
        signal: None,
        outputs: vec![],
        stdout: None,
        stderr: None,
        consumption: ObservedConsumption {
            gate_released_unix_ms: Some(100),
            observed_exit_unix_ms: Some(101),
            cpu_seconds: Some(1.),
            wall_seconds: Some(2.),
            accounting_reason: "synthetic known cost fixture".into(),
            ..Default::default()
        },
    }
}
fn finished(journal: &Path, outcome: &WorkerOutcome) -> PathBuf {
    let path = journal
        .join("finished")
        .join(format!("{}.json", outcome.job_sha256));
    fs::write(&path, serde_json::to_vec(outcome).unwrap()).unwrap();
    path
}
#[test]
fn started_only_command_is_unknown_cost_without_a_fitted_predictor() -> Result<()> {
    let scratch = scratch();
    let job = job(scratch.path(), 1, EvaluationStage::TrainOof);
    started(scratch.path(), &job);
    let value = accounting_for_experiment(&"a".repeat(64), scratch.path())?;
    assert_eq!(value["worker_command_receipts"], 1);
    assert_eq!(value["started_commands_missing_finished_receipts"], 1);
    assert_eq!(value["unknown_cpu_attempts"], 1);
    assert_eq!(value["unknown_wall_attempts"], 1);
    assert_eq!(value["unique_fitted_predictors"], 0);
    assert_eq!(value["completed_fitted_output_commands"], 0);
    assert!(value["cpu_seconds_per_unique_predictor_known"].is_null());
    Ok(())
}
#[test]
fn matched_failure_missing_final_and_unstarted_refusal_are_counted_separately() -> Result<()> {
    let scratch = scratch();
    let known = job(scratch.path(), 1, EvaluationStage::Diagnostic);
    let unknown = job(scratch.path(), 2, EvaluationStage::TrainOof);
    let refused = job(scratch.path(), 3, EvaluationStage::TrainOof);
    started(scratch.path(), &known);
    started(scratch.path(), &unknown);
    finished(scratch.path(), &failed(&known));
    let mut outcome = failed(&refused);
    outcome.command_status = EvaluationStatus::BudgetExhausted;
    outcome.exit_code = None;
    outcome.consumption = ObservedConsumption {
        accounting_reason: "pre-gate setup cost unknown".into(),
        ..Default::default()
    };
    finished(scratch.path(), &outcome);
    let value = accounting_for_experiment(&"a".repeat(64), scratch.path())?;
    assert_eq!(value["worker_command_receipts"], 2);
    assert_eq!(value["context_or_proposal_commands"], 1);
    assert_eq!(value["known_worker_cpu_seconds"], 1.);
    assert_eq!(value["known_sum_worker_wall_seconds"], 2.);
    assert_eq!(value["unknown_cpu_attempts"], 1);
    assert_eq!(value["unknown_wall_attempts"], 1);
    assert_eq!(
        value["unstarted_terminal_inventory"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(value["unstarted_terminal_inventory"][0]["cpu_seconds"].is_null());
    assert_eq!(value["unique_fitted_predictors"], 0);
    Ok(())
}
#[test]
fn accounting_rejects_mismatched_started_identity_or_experiment() -> Result<()> {
    let scratch = scratch();
    let job = job(scratch.path(), 1, EvaluationStage::TrainOof);
    let path = started(scratch.path(), &job);
    assert!(accounting_for_experiment(&"b".repeat(64), scratch.path()).is_err());
    let mut record: Value = serde_json::from_slice(&fs::read(&path)?)?;
    record["job_sha256"] = json!("3".repeat(64));
    fs::write(&path, serde_json::to_vec(&record)?)?;
    assert!(accounting_for_experiment(&"a".repeat(64), scratch.path()).is_err());
    Ok(())
}
#[test]
fn accounting_rejects_wrong_request_and_missing_started_for_launched_outcome() -> Result<()> {
    let scratch = scratch();
    let job = job(scratch.path(), 1, EvaluationStage::TrainOof);
    let path = started(scratch.path(), &job);
    let mut outcome = failed(&job);
    outcome.request_sha256 = "4".repeat(64);
    finished(scratch.path(), &outcome);
    assert!(accounting_for_experiment(&"a".repeat(64), scratch.path()).is_err());
    outcome.request_sha256 = job.request.identity()?;
    finished(scratch.path(), &outcome);
    accounting_for_experiment(&"a".repeat(64), scratch.path())?;
    fs::remove_file(path)?;
    assert!(accounting_for_experiment(&"a".repeat(64), scratch.path()).is_err());
    Ok(())
}
