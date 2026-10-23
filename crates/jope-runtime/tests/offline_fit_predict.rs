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

#[test]
fn bound_programs_keep_symbolic_target_and_ordered_input_state_on_disk() {
    use feat_math::advanced_v3::{
        fit_table, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
    };
    use jope_runtime::model_object::ModelObject;
    let root = tempfile::tempdir().unwrap();
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("amount"),
            ColumnSpec::categorical("country"),
        ],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["amount".into(), "country".into()],
        (0..24)
            .map(|i| {
                vec![
                    (i as f64 * 0.1).to_string(),
                    ["US", "CA", "FR"][i % 3].into(),
                ]
            })
            .collect(),
    )
    .unwrap();
    let rows = RawTable::new(
        table.headers.clone(),
        vec![
            vec!["".into(), "unseen".into()],
            vec!["0.8".into(), "US".into()],
        ],
    )
    .unwrap();
    let y: Vec<f64> = (0..24).map(|i| (i as f64 * 0.3).sin() + 2.0).collect();
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let x: Vec<Vec<f64>> = (0..24)
        .map(|i| vec![i as f64 * 0.1, (i % 3) as f64])
        .collect();
    let symbolic = feat_math::advanced_v3::fit_legacy_numeric_bound(
        &feat_math::GenomeIr::parse("(sym raw \"sqrt(c0) + sin(c1)\")").unwrap(),
        &x,
        &y,
    )
    .unwrap();
    let numeric_rows = RawTable::new(
        vec!["c0".into(), "c1".into()],
        vec![vec!["0.5".into(), "2".into()], vec!["0".into(), "1".into()]],
    )
    .unwrap();
    let expected = symbolic.predict_table(&numeric_rows).unwrap();
    let object = ModelObject::from_bound(&symbolic).unwrap();
    let path = object
        .save(root.path(), Limits::default().max_model_bytes)
        .unwrap();
    let reloaded = ModelObject::read(&path, Limits::default().max_model_bytes)
        .unwrap()
        .bound()
        .unwrap();
    assert_eq!(expected, reloaded.predict_table(&numeric_rows).unwrap());
    for source in [

        "(pipe X (target kind=log1p) (input numeric=median categorical=frequency) (ridge))",
        "(pipe X (target kind=rank_gauss) (input numeric=median categorical=ordered prior=5 folds=5) (standardize) (ridge))",
    ] {
        let model = fit_table(&feat_math::GenomeIr::parse(source).unwrap(),&task,&schema,&table,&y).unwrap();
        let expected = model.predict_table(&rows).unwrap();
        let object = ModelObject::from_bound(&model).unwrap();
        let path = object.save(root.path(),Limits::default().max_model_bytes).unwrap();
        let loaded = ModelObject::read(&path,Limits::default().max_model_bytes).unwrap().bound().unwrap();
        assert_eq!(expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),loaded.predict_table(&rows).unwrap().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"{source}");
    }
}

#[test]
fn importing_native_bytes_preserves_noncanonical_state_json_exactly() {
    use jope_runtime::model_object::ModelObject;
    use sha2::{Digest, Sha256};
    let outcome = fit(
        &prepared(TaskKind::Regression, 24),
        &FitOptions {
            method: "ridge".into(),
            ..FitOptions::default()
        },
        &Budget::new(None, 1),
        "",
        CreatedBy::default(),
    )
    .unwrap();
    let original = outcome.fitted.to_bytes().unwrap();
    let body = serde_json::to_vec_pretty(&outcome.fitted.model).unwrap();
    let mut header = outcome.fitted.header.clone();
    header.fitted_sections[0].bytes = body.len() as u64;
    header.fitted_sections[0].sha256 = jope_runtime::sha256_hex(&body);
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = original[..16].to_vec();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&(body.len() as u64).to_le_bytes());
    file.extend_from_slice(&body);
    let digest = Sha256::digest(&file);
    file.extend_from_slice(&digest);
    FittedModel::from_bytes(&file).unwrap();
    assert_ne!(file, original);
    let object = ModelObject::from_fitted_bytes(file.clone()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let path = object
        .save(root.path(), Limits::default().max_model_bytes)
        .unwrap();
    let loaded = ModelObject::read(&path, Limits::default().max_model_bytes).unwrap();
    assert_eq!(loaded.parts()["model.jope"], file);
}

#[test]
fn calibrated_object_round_trip_preserves_predictions_and_refuses_lossy_lowering() {
    let limits = Limits::default();
    let mut csv = String::from("x,y\n");
    for i in 0..90 {
        csv.push_str(&format!("{},{}\n", i, 20 + 3 * i + (i * 37) % 11));
    }
    let table = parse_csv(csv.as_bytes(), &limits).unwrap();
    let train = training(
        &table,
        "y",
        TaskChoice::Fixed(TaskKind::Regression),
        &limits,
    )
    .unwrap();
    let outcome = fit(
        &train,
        &FitOptions {
            candidates: vec!["ridge".into()],
            folds: 3,
            fitness: Some(jope_runtime::fitness::Fitness::Pinball { quantile: 0.9 }),
            ..FitOptions::default()
        },
        &Budget::new(None, 1),
        &table.sha256,
        CreatedBy::default(),
    )
    .unwrap();
    let transform = outcome
        .fitted
        .header
        .output_transform
        .expect("quantile calibration");
    assert!(!transform.is_identity());
    let input = parse_csv(b"x\n1\n4\n", &limits).unwrap();
    let (rows, _) = prediction(&input, &train.schema, "y").unwrap();
    let expected =
        predictions_csv(&predict(&outcome.fitted, &rows, &Budget::new(None, 1)).unwrap()).unwrap();
    let object = jope_runtime::model_object::ModelObject::from_fitted(&outcome.fitted).unwrap();
    let root = tempfile::tempdir().unwrap();
    let path = object.save(root.path(), limits.max_model_bytes).unwrap();
    let loaded = jope_runtime::model_object::load_fitted(&path, limits.max_model_bytes).unwrap();
    assert_eq!(
        expected,
        predictions_csv(&predict(&loaded, &rows, &Budget::new(None, 1)).unwrap()).unwrap()
    );
    assert!(object.inspect().unwrap().executable);
    for language in ["python", "rust", "javascript"] {
        assert!(
            object.export_source(language).is_err(),
            "{language} dropped calibration"
        );
    }
    assert!(object.bound().is_err(), "bound JSON dropped calibration");
}
