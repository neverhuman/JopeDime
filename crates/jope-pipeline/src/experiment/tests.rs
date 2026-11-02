use super::*;
use super::*;
fn experiment() -> Experiment {
    Experiment {
        schema: EXPERIMENT_SCHEMA.into(),
        panel_sha256: "a".repeat(64),
        methods_sha256: "b".repeat(64),
        source_sha: "c".repeat(40),
        checkpoints: BTreeMap::new(),
        seed: 42,
        resources: Resources {
            workers: 20,
            cpus: 20,
            memory_bytes: 64 << 30,
            swap_bytes: 0,
            gpu_tasks: 1,
            threads_per_worker: 1,
            total_seconds: 7200,
            reporting_seconds: 120,
            per_fit_seconds: 10,
        },
        expected_artifacts: vec!["report.json".into()],
        protocol: "development-v1".into(),
        numerical_policy: "strict-v1".into(),
    }
}
fn request() -> EvaluationRequest {
    EvaluationRequest {
        schema: REQUEST_SCHEMA.into(),
        experiment_sha256: "a".repeat(64),
        dataset_sha256: "b".repeat(64),
        split_sha256: "c".repeat(64),
        fitting_rows_sha256: "d".repeat(64),
        scoring_rows_sha256: "e".repeat(64),
        fitting_rows: 80,
        scoring_rows: 20,
        recipe_sha256: "f".repeat(64),
        evaluator_sha256: "0".repeat(64),
        checkpoint_sha256: None,
        task: Task::Binary,
        metric: Metric::Auroc,
        classes: vec!["no".into(), "yes".into()],
        fold: 0,
        stage: EvaluationStage::Screen,
        seed: 42,
        numerical_policy: "strict-v1".into(),
    }
}
fn result(r: &EvaluationRequest) -> EvaluationResult {
    EvaluationResult {
        schema: RESULT_SCHEMA.into(),
        request_sha256: r.identity().unwrap(),
        status: EvaluationStatus::Completed,
        reason: None,
        primary: Some(0.6),
        secondary: BTreeMap::new(),
        scored_rows: 20,
        predictions_sha256: Some("1".repeat(64)),
        fitted_state_sha256: Some("2".repeat(64)),
        consumption: Consumption {
            wall_seconds: 2.,
            cpu_seconds: 1.,
            gpu_seconds: None,
            peak_rss_bytes: Some(1000),
        },
    }
}
#[test]
fn budgets_fail_closed_and_resume_keeps_original_deadline() {
    let e = experiment();
    e.validate().unwrap();
    let window = ExecutionWindow {
        experiment_sha256: e.identity().unwrap(),
        started_unix_ms: 1000,
        deadline_unix_ms: 7_201_000,
    };
    assert!(window.may_admit(&e, 7_080_999).unwrap());
    assert!(!window.may_admit(&e, 7_081_000).unwrap());
    assert!(!window.may_admit(&e, 9_000_000).unwrap());
    assert!(window.may_admit(&e, 999).is_err());
    let mut extended = window.clone();
    extended.deadline_unix_ms += 1;
    assert!(extended.validate(&e).is_err());
    for field in [
        "workers",
        "cpus",
        "memory_bytes",
        "swap_bytes",
        "gpu_tasks",
        "threads_per_worker",
        "total_seconds",
        "reporting_seconds",
        "per_fit_seconds",
    ] {
        let mut value = serde_json::to_value(&e).unwrap();
        value["resources"][field] = match field {
            "memory_bytes" => (65u64 << 30).into(),
            "swap_bytes" => 1.into(),
            "gpu_tasks" | "threads_per_worker" => 2.into(),
            "total_seconds" => 7201.into(),
            "reporting_seconds" | "per_fit_seconds" => 7200.into(),
            _ => 21.into(),
        };
        assert!(
            serde_json::from_value::<Experiment>(value)
                .unwrap()
                .validate()
                .is_err(),
            "{field}"
        );
    }
}
#[test]
fn identity_binds_fold_classes_checkpoint_and_all_evaluation_inputs() {
    let r = request();
    let measured = result(&r);
    measured.validate(&r).unwrap();
    for field in [
        "experiment_sha256",
        "dataset_sha256",
        "split_sha256",
        "fitting_rows_sha256",
        "scoring_rows_sha256",
        "recipe_sha256",
        "evaluator_sha256",
        "checkpoint_sha256",
    ] {
        let mut changed = serde_json::to_value(&r).unwrap();
        changed[field] = "9".repeat(64).into();
        let changed: EvaluationRequest = serde_json::from_value(changed).unwrap();
        assert!(measured.validate(&changed).is_err(), "{field}");
    }
    let mut changed = r.clone();
    changed.classes.reverse();
    assert!(measured.validate(&changed).is_err());
    changed = r.clone();
    changed.stage = EvaluationStage::Confirm;
    assert!(measured.validate(&changed).is_err());
    changed.fold = 1;
    assert!(measured.validate(&changed).is_err());
    changed = r.clone();
    changed.seed += 1;
    assert!(measured.validate(&changed).is_err());
    changed = r.clone();
    changed.numerical_policy = "alternate".into();
    assert!(measured.validate(&changed).is_err());
    changed = r;
    changed.scoring_rows_sha256 = changed.fitting_rows_sha256.clone();
    assert!(changed.validate().is_err());
}
#[test]
fn zero_nonfinite_incomplete_and_wrong_task_results_cannot_pass() {
    let r = request();
    let mut m = result(&r);
    m.scored_rows = 0;
    assert!(m.validate(&r).is_err());
    for value in [f64::NAN, f64::INFINITY, -1., 1.1] {
        m = result(&r);
        m.primary = Some(value);
        assert!(m.validate(&r).is_err());
    }
    m = result(&r);
    m.status = EvaluationStatus::TimedOut;
    assert!(m.validate(&r).is_err());
    m.primary = None;
    m.scored_rows = 0;
    m.reason = Some("fit deadline exceeded".into());
    m.validate(&r).unwrap();
    let mut bad = r;
    bad.metric = Metric::Rmse;
    assert!(bad.validate().is_err());
    bad.task = Task::Multiclass;
    bad.metric = Metric::LogLoss;
    assert!(bad.validate().is_err());
    assert_eq!(Metric::Auroc.loss(0.8).unwrap(), 1. - 0.8);
    assert_eq!(Metric::Rmse.loss(0.8).unwrap(), 0.8);
}
#[test]
fn frozen_panel_has_ten_of_each_original_task_and_valid_identities() {
    let panel: Panel =
        serde_json::from_str(include_str!("../../../../benchmarks/panel.json")).unwrap();
    panel.validate().unwrap();
    let mut changed = panel.clone();
    changed.datasets[0].metric = Metric::Auroc;
    assert!(changed.validate().is_err());
    changed = panel.clone();
    changed.datasets[0].split.fold_rows[0] = 0;
    assert!(changed.validate().is_err());
    changed = panel.clone();
    changed.datasets[0].split.sealed_sha256 = changed.datasets[0].split.development_sha256.clone();
    assert!(changed.validate().is_err());
    changed = panel;
    changed.datasets[0].id = changed.datasets[1].id.clone();
    assert!(changed.validate().is_err());
}
