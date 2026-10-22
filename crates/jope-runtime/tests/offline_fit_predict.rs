//! Raw CSV → fit → `model.jope` → reload → predict, for every task and every native method:
//! reloaded predictions equal in-memory ones bit for bit, prediction input carries no labels,
//! and a damaged or foreign model file is refused as an integrity failure.
use jope_runtime::api::{fit, predict, predictions_csv, FitOptions};
use jope_runtime::budget::Budget;
use jope_runtime::fitted::{CreatedBy, FittedModel};
use jope_runtime::limits::Limits;
use jope_runtime::methods::{native::Predictions, rust_methods};
use jope_runtime::table::{parse_csv, prediction, training, Training};
use jope_runtime::task::{TaskChoice, TaskKind};
use jope_runtime::{error::kind_of, FailureKind};

/// A deterministic table with a numeric, a categorical (with literal "NA") and a sparse column.
fn csv(task: TaskKind, rows: usize) -> String {
    let mut out = String::from("x1,color,x3,y\n");
    for i in 0..rows {
        let x1 = (i as f64 * 0.37).sin() * 3.0;
        let color = ["red", "blue", "NA", "green"][i % 4];
        let x3 = if i % 7 == 0 {
            String::new()
        } else {
            format!("{}", (i % 11) as f64 / 2.0)
        };
        let signal = x1 + if color == "red" { 1.5 } else { -0.5 };
        let y = match task {
            TaskKind::Regression => format!("{:.6}", signal * 2.0 + 1.0),
            TaskKind::Binary => (if signal > 0.3 { "yes" } else { "no" }).to_owned(),
            _ => ["low", "mid", "high"][(i * 7 + (signal > 0.0) as usize) % 3].to_owned(),
        };
        out.push_str(&format!("{x1:.6},{color},{x3},{y}\n"));
    }
    out
}

fn prepared(task: TaskKind, rows: usize) -> Training {
    let limits = Limits::default();
    let table = parse_csv(csv(task, rows).as_bytes(), &limits).unwrap();
    training(&table, "y", TaskChoice::Fixed(task), &limits).unwrap()
}

fn holdout(training: &Training) -> feat_math::advanced_v3::RawTable {
    // Prediction rows in another column order, with an unseen category and no target column.
    let text = "x3,color,x1\n2.5,purple,0.25\n,red,-1.0\n1.0,NA,\n";
    let table = parse_csv(text.as_bytes(), &Limits::default()).unwrap();
    let (raw, ignored) = prediction(&table, &training.schema, &training.target).unwrap();
    assert!(ignored.is_empty());
    raw
}

#[test]
fn every_native_method_fits_saves_reloads_and_predicts_identically() {
    let budget = Budget::new(None, 2);
    for task in [TaskKind::Regression, TaskKind::Binary, TaskKind::Multiclass] {
        let training = prepared(task, 60);
        let rows = holdout(&training);
        for method in rust_methods(task).unwrap() {
            let options = FitOptions {
                method: method.id.clone(),
                ..FitOptions::default()
            };
            let outcome = fit(
                &training,
                &options,
                &budget,
                &"0".repeat(64),
                CreatedBy::default(),
            )
            .unwrap_or_else(|e| panic!("{}: {e:#}", method.id));
            assert_eq!(outcome.method, method);
            let in_memory = predict(&outcome.fitted, &rows, &budget).unwrap();
            // The same fitted state can be retained in the shared object store, reloaded
            // without fitting, and exported as bound JSON without dropping preprocessing.
            let directory = tempfile::tempdir().unwrap();
            let object =
                jope_runtime::model_object::ModelObject::from_fitted(&outcome.fitted).unwrap();
            let path = object
                .save(directory.path(), Limits::default().max_model_bytes)
                .unwrap();
            assert!(object.inspect().unwrap().executable);
            let loaded =
                jope_runtime::model_object::load_fitted(&path, Limits::default().max_model_bytes)
                    .unwrap();
            let object_predictions = predict(&loaded, &rows, &budget).unwrap();
            assert_eq!(
                serde_json::to_vec(&in_memory).unwrap(),
                serde_json::to_vec(&object_predictions).unwrap(),
                "{} object round trip",
                method.id
            );
            let bound_object =
                jope_runtime::model_object::ModelObject::from_bound(&outcome.fitted.model).unwrap();
            let bound = bound_object.bound().unwrap();
            assert_eq!(
                serde_json::to_vec(&bound).unwrap(),
                serde_json::to_vec(&outcome.fitted.model).unwrap()
            );
            let bytes = outcome.fitted.to_bytes().unwrap();
            let reloaded = FittedModel::from_bytes(&bytes).unwrap();
            let again = predict(&reloaded, &rows, &budget).unwrap();
            assert_eq!(
                serde_json::to_vec(&in_memory).unwrap(),
                serde_json::to_vec(&again).unwrap(),
                "{}",
                method.id
            );
            assert_eq!(in_memory.rows(), 3);
            if let Predictions::Probabilities { classes, .. } = &again {
                assert_eq!(classes, &training.classes);
            }
            assert_eq!(reloaded.header.method.id, method.id);
            assert_eq!(
                reloaded.header.training_procedure.method_choice,
                "requested"
            );
        }
    }
}

#[test]
fn auto_selects_by_out_of_fold_loss_and_records_every_candidate() {
    let budget = Budget::new(None, 2);
    for task in [TaskKind::Regression, TaskKind::Binary, TaskKind::Multiclass] {
        let training = prepared(task, 80);
        let options = FitOptions {
            candidates: match task {
                TaskKind::Regression => vec![
                    "ridge".into(),
                    "histgbm".into(),
                    "gpr.rbf_linear".into(),
                    "rust.pls".into(),
                ],
                _ => Vec::new(),
            },
            ..FitOptions::default()
        };
        let outcome = fit(
            &training,
            &options,
            &budget,
            &"0".repeat(64),
            CreatedBy::default(),
        )
        .unwrap();
        let selection = outcome.selection.unwrap();
        assert_eq!(selection.policy, "jope.runtime-select/v1");
        assert_eq!(selection.folds, 5);
        let scored: Vec<_> = selection
            .candidates
            .iter()
            .filter(|c| c.status == "scored")
            .collect();
        assert!(!scored.is_empty());
        let best = scored
            .iter()
            .min_by(|a, b| a.loss.unwrap().total_cmp(&b.loss.unwrap()))
            .unwrap();
        assert_eq!(selection.selected, best.id);
        assert_eq!(outcome.method.id, best.id);
        let header = &outcome.fitted.header;
        assert_eq!(
            header.training_procedure.method_choice,
            "jope.runtime-select/v1"
        );
        let expected = match task {
            TaskKind::Regression => 4,
            TaskKind::Binary => 5,
            _ => 4,
        };
        assert_eq!(selection.candidates.len(), expected);
    }
}

#[test]
fn selection_is_deterministic() {
    let training = prepared(TaskKind::Binary, 60);
    let budget = Budget::new(None, 3);
    let options = FitOptions::default();
    let a = fit(&training, &options, &budget, "", CreatedBy::default()).unwrap();
    let b = fit(
        &training,
        &options,
        &Budget::new(None, 1),
        "",
        CreatedBy::default(),
    )
    .unwrap();
    assert_eq!(a.method, b.method);
    let losses = |o: &jope_runtime::api::FitOutcome| -> Vec<Option<f64>> {
        o.selection
            .as_ref()
            .unwrap()
            .candidates
            .iter()
            .map(|c| c.loss)
            .collect()
    };
    assert_eq!(losses(&a), losses(&b));
}

#[test]
fn an_exhausted_budget_is_a_budget_failure() {
    let training = prepared(TaskKind::Regression, 40);
    let budget = Budget::new(Some(std::time::Duration::ZERO), 1);
    let error = fit(
        &training,
        &FitOptions::default(),
        &budget,
        "",
        CreatedBy::default(),
    )
    .err()
    .unwrap();
    assert_eq!(kind_of(&error), FailureKind::Budget, "{error:#}");
}

#[test]
fn damaged_or_foreign_model_files_are_integrity_failures() {
    let training = prepared(TaskKind::Regression, 40);
    let budget = Budget::new(None, 1);
    let options = FitOptions {
        method: "ridge".into(),
        ..FitOptions::default()
    };
    let outcome = fit(&training, &options, &budget, "", CreatedBy::default()).unwrap();
    let bytes = outcome.fitted.to_bytes().unwrap();
    let refuse = |bytes: &[u8]| {
        let error = FittedModel::from_bytes(bytes).unwrap_err();
        assert_eq!(kind_of(&error), FailureKind::Integrity, "{error:#}");
    };
    let mut flipped = bytes.clone();
    flipped[bytes.len() / 2] ^= 1;
    refuse(&flipped);
    refuse(&bytes[..bytes.len() - 1]);
    refuse(b"PK\x03\x04 not a model at all, but long enough to have a trailer..........");
    // A header edited and re-checksummed still fails its cross-checks.
    let mut fitted = outcome.fitted.clone();
    fitted.header.method.id = "rust.ridge_gcv".into();
    fitted.header.method.family = "ridge_gcv".into();
    refuse(&fitted.to_bytes().unwrap());
    let mut fitted = outcome.fitted.clone();
    fitted.header.class_order = vec!["a".into(), "b".into()];
    refuse(&fitted.to_bytes().unwrap());
}

#[test]
fn prediction_csv_has_probabilities_in_class_order() {
    let training = prepared(TaskKind::Multiclass, 45);
    let budget = Budget::new(None, 1);
    let options = FitOptions {
        method: "multiclass_ridge".into(),
        ..FitOptions::default()
    };
    let outcome = fit(&training, &options, &budget, "", CreatedBy::default()).unwrap();
    let text = String::from_utf8(
        predictions_csv(&predict(&outcome.fitted, &holdout(&training), &budget).unwrap()).unwrap(),
    )
    .unwrap();
    let header = text.lines().next().unwrap();
    assert_eq!(header, "prediction,p_high,p_low,p_mid");
    assert_eq!(text.lines().count(), 4);
}
