use jope_runtime::limits::Limits;
use jope_runtime::math_pipeline::{
    refittable::{fit_csv, predict_csv},
    F64_PREDICTION_VERSION, INPUT_PART, PIPELINE_PART,
};
use jope_runtime::model_object::{ModelObject, Part, MATH_PIPELINE, SCHEMA};
use std::path::{Path, PathBuf};

const RECIPE: &str = "(pipe X (ridge lambda_frac=5e-13))";
const FIT_A: &str = "ignored,a,y,b\ntext,0,10,10\ntext,1,12,10\ntext,0,14,11\ntext,1,16,11\n";
const FIT_B: &str =
    "q,ignored,response,p\n20,text,100,4\n20,text,104,8\n24,text,112,4\n24,text,116,8\n";

fn write_csv(dir: &Path, name: &str, csv: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, csv).unwrap();
    path
}

fn reload(path: &Path) -> ModelObject {
    let object = ModelObject::read(path, Limits::default().max_model_bytes).unwrap();
    let inspection = object.inspect().unwrap();
    assert!(inspection.executable && inspection.integrity_verified);
    assert_eq!(object.manifest().schema, SCHEMA);
    assert_eq!(object.manifest().format, MATH_PIPELINE);
    assert_eq!(
        object
            .parts()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [INPUT_PART, PIPELINE_PART]
    );
    object
}

fn assert_predictions(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1e-8,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn exact_recipe_source_is_independent_of_fitted_values_and_columns() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("models");
    let fit_a = write_csv(dir.path(), "fit-a.csv", FIT_A);
    let fit_b = write_csv(dir.path(), "fit-b.csv", FIT_B);
    let path_a = fit_csv(RECIPE, &fit_a, "y", &["b".into(), "a".into()], &store).unwrap();
    let path_b = fit_csv(
        RECIPE,
        &fit_b,
        "response",
        &["q".into(), "p".into()],
        &store,
    )
    .unwrap();
    let object_a = reload(&path_a);
    let object_b = reload(&path_b);
    assert_ne!(object_a.id(), object_b.id());

    let pipeline_a = object_a.math_pipeline().unwrap();
    let pipeline_b = object_b.math_pipeline().unwrap();
    assert_eq!(pipeline_a.state.dsl, RECIPE);
    assert_eq!(pipeline_b.state.dsl, RECIPE);
    assert_eq!(pipeline_a.input.prediction_version, F64_PREDICTION_VERSION);
    assert_eq!(pipeline_a.input.columns[0].name, "b");
    assert_eq!(pipeline_a.input.columns[0].min, 10.0);
    assert_eq!(pipeline_b.input.columns[0].name, "q");
    assert_eq!(pipeline_b.input.columns[0].min, 20.0);

    let source = pipeline_a.export_refittable_rust().unwrap();
    assert_eq!(source, pipeline_b.export_refittable_rust().unwrap());
    assert!(source.contains(&format!("const RECIPE: &str = {RECIPE:?};")));
    for fitted_state in ["fn fitted_model()", "INPUT_SCALERS", "SELF_CHECK"] {
        assert!(
            !source.contains(fitted_state),
            "source contains {fitted_state}"
        );
    }

    let input_a = write_csv(
        dir.path(),
        "input-a.csv",
        "a,ignored,b\n0.25,text,10.5\n1,text,NA\n-1,text,999\n",
    );
    let input_b = write_csv(dir.path(), "input-b.csv", "p,q\n6,21\n");
    // The four orthogonal standardized rows have Gram diagonal 4. The
    // regularization floor shrinks the slopes and leaves the intercept intact.
    let shrinkage = 4.0 / (4.0 + 1e-3);
    assert_predictions(
        &predict_csv(RECIPE, &path_a, &input_a).unwrap(),
        &[13.0 - 0.5 * shrinkage, 13.0 - shrinkage, 13.0 + shrinkage],
    );
    assert_predictions(
        &predict_csv(RECIPE, &path_b, &input_b).unwrap(),
        &[108.0 - 3.0 * shrinkage],
    );
}

#[test]
fn invalid_fit_and_wrong_recipe_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("models");
    let valid = write_csv(dir.path(), "fit.csv", FIT_A);
    let missing = write_csv(dir.path(), "missing.csv", "a,y,b\n0,10,\n1,12,10\n");
    let tiny_target = write_csv(dir.path(), "tiny.csv", "a,y,b\n0,0,10\n1,5e-13,11\n");
    let features = ["b".into(), "a".into()];
    assert!(fit_csv(RECIPE, &missing, "y", &features, &store).is_err());
    assert!(fit_csv(RECIPE, &tiny_target, "y", &features, &store).is_err());
    assert!(fit_csv(RECIPE, &valid, "y", &["y".into()], &store).is_err());
    assert!(fit_csv(RECIPE, &valid, "y", &[], &store).is_err());
    assert!(
        fit_csv("(pipe X (ridge unknown=1))", &valid, "y", &features, &store)
            .unwrap_err()
            .to_string()
            .contains("unsupported refittable recipe parameter")
    );
    assert!(!store.exists() || store.read_dir().unwrap().next().is_none());

    let path = fit_csv(RECIPE, &valid, "y", &features, &store).unwrap();
    assert_eq!(
        path,
        fit_csv(RECIPE, &valid, "y", &features, &store).unwrap()
    );
    let input = write_csv(dir.path(), "predict.csv", "a,b\n0.25,10.5\n");
    let wrong_recipe = "(pipe X (ridge lambda_frac=2e-13))";
    let error = predict_csv(wrong_recipe, &path, &input).unwrap_err();
    assert!(error.to_string().contains("recipe"), "{error:#}");
}

#[test]
fn histogram_recipe_reloads_and_rejects_a_mislabeled_head() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("models");
    let recipe = "(pipe X (robust_scale append=false) (histgbm depth=1 min_leaf=2 rounds=3))";
    let mut rows = String::from("x,y\n");
    for index in 0..24 {
        rows.push_str(&format!("{index},{}\n", if index < 12 { 2 } else { 17 }));
    }
    let csv = write_csv(dir.path(), "fit.csv", &rows);
    let path = fit_csv(recipe, &csv, "y", &["x".into()], &store).unwrap();
    let object = reload(&path);
    let pipeline = object.math_pipeline().unwrap();
    let source = pipeline.export_refittable_rust().unwrap();
    assert!(source.contains("(robust_scale) (histgbm depth=1 min_leaf=2 rounds=3)"));
    let input = write_csv(dir.path(), "probe.csv", "x\n3\n21\n");
    let predictions = predict_csv(recipe, &path, &input).unwrap();
    assert!(predictions[0] < predictions[1], "{predictions:?}");

    // A valid content hash and width do not establish that the recipe fits this head.
    let mut state = pipeline.state;
    state.dsl = "(pipe X (robust_scale) (ridge))".into();
    let mut parts = object.parts().clone();
    parts.insert(PIPELINE_PART.into(), serde_json::to_vec(&state).unwrap());
    let mut manifest = object.manifest().clone();
    manifest.files = parts
        .iter()
        .map(|(name, bytes)| Part::new(name, "application/json", bytes))
        .collect();
    let forged = ModelObject::new(manifest, parts, Limits::default().max_model_bytes).unwrap();
    let error = forged
        .math_pipeline()
        .unwrap()
        .export_refittable_rust()
        .unwrap_err();
    assert!(error.to_string().contains("fitted stages or head"));
}
