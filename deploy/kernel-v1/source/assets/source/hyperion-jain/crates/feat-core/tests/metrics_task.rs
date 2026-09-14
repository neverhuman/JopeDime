//! Step-2: task-aware metrics + objective glue (pure functions; no FFI).
use feat_core::backend::{LossKind, ObjectiveSpec, TaskKind as BTask};
use feat_core::metrics::{mae, ovr_macro_auc, r2, rmse, task_score};
use feat_core::task::{TaskKind, TaskSpec};

#[test]
fn regression_metrics() {
    let y = [1.0f32, 2.0, 3.0, 4.0];
    let yhat = [1.0f32, 2.0, 3.0, 4.0];
    assert!(rmse(&y, &yhat) < 1e-6 && mae(&y, &yhat) < 1e-6);
    assert!((r2(&y, &yhat) - 1.0).abs() < 1e-6, "perfect fit -> r2 == 1");
    // a worse predictor has lower r2
    let bad = [2.5f32, 2.5, 2.5, 2.5];
    assert!(r2(&y, &bad) < 0.5);
}

#[test]
fn multiclass_ovr_auc_perfect() {
    // 3 classes, rows row-major r*k+c; perfectly separating proba -> macro AUC == 1
    let y_idx = [0.0f32, 1.0, 2.0, 0.0];
    let proba = [
        0.9, 0.05, 0.05, // row0 -> class0
        0.1, 0.8, 0.1, // row1 -> class1
        0.1, 0.2, 0.7, // row2 -> class2
        0.7, 0.2, 0.1, // row3 -> class0
    ]
    .map(|v: f64| v as f32);
    assert!((ovr_macro_auc(&y_idx, &proba, 3) - 1.0).abs() < 1e-6);
}

#[test]
fn task_score_dispatch() {
    // Binary -> AUC
    let y = [0.0f32, 0.0, 1.0, 1.0];
    let s = [0.1f32, 0.2, 0.8, 0.9];
    assert!((task_score(TaskKind::Binary, &y, &s, 2) - 1.0).abs() < 1e-6);
    // Regression -> r2
    let yr = [1.0f32, 2.0, 3.0];
    assert!((task_score(TaskKind::Regression, &yr, &yr, 0) - 1.0).abs() < 1e-6);
}

#[test]
fn objective_from_task_keeps_binary_logloss() {
    let bin = TaskSpec {
        kind: TaskKind::Binary,
        n_targets: 1,
        n_classes: 2,
        class_values: vec![0.0, 1.0],
        prevalence: 0.3,
    };
    let o = ObjectiveSpec::from_task(&bin);
    assert_eq!(o.task, BTask::BinaryClass);
    assert_eq!(o.loss, LossKind::Logloss); // unchanged default
    let reg = TaskSpec {
        kind: TaskKind::Regression,
        n_targets: 1,
        n_classes: 0,
        class_values: vec![],
        prevalence: f64::NAN,
    };
    assert_eq!(ObjectiveSpec::from_task(&reg).loss, LossKind::Rmse);
}

#[test]
fn task_spec_prevalence_round_trips_null_for_non_classification() {
    let reg = TaskSpec {
        kind: TaskKind::Regression,
        n_targets: 1,
        n_classes: 0,
        class_values: vec![],
        prevalence: f64::NAN,
    };

    let text = serde_json::to_string(&reg).unwrap();
    assert!(text.contains(r#""prevalence":null"#));
    let decoded: TaskSpec = serde_json::from_str(&text).unwrap();
    assert!(decoded.prevalence.is_nan());
}
