//! Bounded frozen-panel comparator controller. Scores remain parent-owned.
//! Neural search supplies its frozen recipes through the same worker boundary.
use super::{
    data::{DevelopmentCut, DevelopmentData},
    methods::RustMethod,
    protocol::{self, Observation},
    score::{self, Predictions},
    worker::RustWorkerOutput,
};
use anyhow::{ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_pipeline::{
    evaluation_result::{Consumption, EvaluationResult, RESULT_SCHEMA},
    execution::{Artifact, ExecutionPool, WorkerCommand, WorkerJob, WorkerOutcome},
    experiment::{
        EvaluationRequest, EvaluationStage, EvaluationStatus, Experiment, Panel, REQUEST_SCHEMA,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

mod fit;
mod report;
mod run;
mod scoring;
mod types;

pub use fit::prepare_fit;
pub(crate) use fit::{digest, immutable, prepare_fit_impl, request};
pub(crate) use report::cpu_seconds;
pub use report::{cost, report};
pub use run::{run, run_with};
#[cfg(feature = "asm")]
pub(crate) use scoring::checked_predictions;
pub use scoring::score_fit;
pub(crate) use scoring::{reuse_fold_zero, score_batch};
pub use types::{CampaignPlan, Method, PendingFit, Runtime, PLAN_SCHEMA};

#[cfg(test)]
mod tests {
    use super::*;
    use jope_pipeline::{execution::ObservedConsumption, experiment::Metric};

    // These bytes are an explicit controller-validation fixture, not an estimator
    // or evidence of a successful external fit. No pickle is loaded in the parent.
    fn external_fixture(root: &Path, predictions: Predictions) -> (PendingFit, WorkerOutcome) {
        let method = Method::External {
            id: "fixture-external".into(),
            engine: "sklearn".into(),
            runtime: "sklearn".into(),
        };
        let request = EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: "a".repeat(64),
            dataset_sha256: "b".repeat(64),
            split_sha256: "c".repeat(64),
            fitting_rows_sha256: "d".repeat(64),
            scoring_rows_sha256: "e".repeat(64),
            fitting_rows: 10,
            scoring_rows: 2,
            recipe_sha256: method.identity().unwrap(),
            evaluator_sha256: "f".repeat(64),
            checkpoint_sha256: None,
            task: Task::Regression,
            metric: Metric::Rmse,
            classes: vec![],
            fold: 0,
            stage: EvaluationStage::Screen,
            seed: 42,
            numerical_policy: score::NUMERICAL_POLICY.into(),
        };
        let directory = fs::canonicalize(root).unwrap();
        let artifact_path = directory.join("output.json.fitted.pkl");
        let artifact_bytes = b"explicit synthetic fitted-state fixture";
        fs::write(&artifact_path, artifact_bytes).unwrap();
        let artifact_hash = jope_pipeline::sha256_hex(artifact_bytes);
        let input_hash = "1".repeat(64);
        let output = json!({"contract":super::super::methods::ADAPTER_CONTRACT,
            "input_sha256":input_hash,"engine_version":"fixture-1","predictions":predictions,
            "fitted":{"path":"output.json.fitted.pkl","sha256":artifact_hash,
                "reload_verified":false,"format":"jope.panel-fitted-python/v1",
                "preprocessing_policy":super::super::methods::INPUT_POLICY,
                "bytes":artifact_bytes.len()}});
        let output_path = directory.join("output.json");
        let output_bytes = serde_json::to_vec(&output).unwrap();
        fs::write(&output_path, &output_bytes).unwrap();
        let job = WorkerJob {
            request: request.clone(),
            command: WorkerCommand {
                program: Artifact {
                    path: directory.join("unused-fixture-command"),
                    sha256: "2".repeat(64),
                },
                args: vec![],
                inputs: vec![],
                environment: BTreeMap::new(),
            },
            directory,
            outputs: vec!["output.json".into(), "output.json.fitted.pkl".into()],
            arm: None,
        };
        let outcome = WorkerOutcome {
            schema: "jope.worker-execution/v2".into(),
            job_sha256: job.identity().unwrap(),
            request_sha256: request.identity().unwrap(),
            command_status: EvaluationStatus::Completed,
            reason: None,
            exit_code: Some(0),
            signal: None,
            outputs: vec![
                Artifact {
                    path: output_path,
                    sha256: jope_pipeline::sha256_hex(&output_bytes),
                },
                Artifact {
                    path: artifact_path,
                    sha256: artifact_hash,
                },
            ],
            stdout: None,
            stderr: None,
            consumption: ObservedConsumption {
                wall_seconds: Some(2.),
                cpu_seconds: Some(1.5),
                accounting_reason: "synthetic fixture measurement".into(),
                ..Default::default()
            },
        };
        (
            PendingFit {
                method,
                request,
                job,
                input_sha256: input_hash,
                expected_engine_version: "fixture-1".into(),
            },
            outcome,
        )
    }

    #[test]
    fn parent_scores_original_labels_and_refuses_changed_fitted_bytes() {
        let root = tempfile::tempdir().unwrap();
        let (fit, outcome) = external_fixture(root.path(), Predictions::Values(vec![2., 4.]));
        let labels = vec!["1".into(), "3".into()];
        let accepted = score_fit(&fit, &outcome, &labels).unwrap();
        assert_eq!(accepted.result.status, EvaluationStatus::Completed);
        assert_eq!(accepted.result.primary, Some(1.));
        fs::write(fit.job.directory.join("output.json.fitted.pkl"), b"changed").unwrap();
        let rejected = score_fit(&fit, &outcome, &labels).unwrap();
        assert_eq!(rejected.result.status, EvaluationStatus::Failed);
        assert!(rejected.result.primary.is_none());
        assert!(rejected.result.reason.unwrap().contains("artifact changed"));
    }

    #[test]
    fn successful_command_with_bad_prediction_shape_has_no_score() {
        let root = tempfile::tempdir().unwrap();
        let (fit, outcome) = external_fixture(root.path(), Predictions::Values(vec![2.]));
        let result = score_fit(&fit, &outcome, &["1".into(), "3".into()])
            .unwrap()
            .result;
        assert_eq!(result.status, EvaluationStatus::Failed);
        assert!(result.primary.is_none());
        assert_eq!(result.scored_rows, 0);
        assert_eq!(result.consumption.cpu_seconds, Some(1.5));
    }

    #[test]
    fn stale_worker_request_and_runtime_version_cannot_publish_scores() {
        let root = tempfile::tempdir().unwrap();
        let (mut fit, mut outcome) =
            external_fixture(root.path(), Predictions::Values(vec![2., 4.]));
        let labels = vec!["1".into(), "3".into()];
        outcome.request_sha256 = "9".repeat(64);
        assert!(score_fit(&fit, &outcome, &labels).is_err());
        outcome.request_sha256 = fit.request.identity().unwrap();
        fit.expected_engine_version = "different".into();
        let result = score_fit(&fit, &outcome, &labels).unwrap().result;
        assert_eq!(result.status, EvaluationStatus::Failed);
        assert!(result.primary.is_none());
    }

    #[test]
    fn reuse_keeps_same_cut_and_counts_physical_cost_once() {
        let root = tempfile::tempdir().unwrap();
        let (fit, outcome) = external_fixture(root.path(), Predictions::Values(vec![2., 4.]));
        let screen = score_fit(&fit, &outcome, &["1".into(), "3".into()]).unwrap();
        let mut confirm = screen.request.clone();
        confirm.stage = EvaluationStage::Confirm;
        let reused = reuse_fold_zero(&screen, confirm.clone()).unwrap();
        let accounting = cost(&[&screen, &screen, &reused]);
        assert_eq!(accounting["completed_fits"], 1);
        assert_eq!(accounting["unique_fitted_predictors"], 1);
        assert_eq!(accounting["known_worker_cpu_seconds"], 1.5);
        confirm.scoring_rows_sha256 = "3".repeat(64);
        assert!(reuse_fold_zero(&screen, confirm).is_err());
    }

    #[test]
    fn interrupted_measurements_stay_unknown_and_evidence_is_immutable() {
        let root = tempfile::tempdir().unwrap();
        let (fit, mut outcome) = external_fixture(root.path(), Predictions::Values(vec![2., 4.]));
        outcome.command_status = EvaluationStatus::Cancelled;
        outcome.reason = Some("controller_interrupted".into());
        outcome.consumption = ObservedConsumption {
            accounting_reason: "interrupted: unknown".into(),
            ..Default::default()
        };
        let observation = score_fit(&fit, &outcome, &["1".into(), "3".into()]).unwrap();
        assert!(observation.result.primary.is_none());
        let accounting = cost(&[&observation]);
        assert_eq!(accounting["unknown_cpu_attempts"], 1);
        assert_eq!(accounting["unique_fitted_predictors"], 0);
        assert!(accounting["cpu_seconds_per_unique_predictor_known"].is_null());
        let path = root.path().join("evidence.json");
        immutable(&path, &json!({"original":true})).unwrap();
        immutable(&path, &json!({"original":true})).unwrap();
        assert!(immutable(&path, &json!({"original":false})).is_err());
    }
}
