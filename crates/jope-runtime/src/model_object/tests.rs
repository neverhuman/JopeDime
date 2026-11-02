use super::*;

#[path = "format_tests.rs"]
mod format_tests;
#[path = "publication_tests.rs"]
mod publication_tests;
#[path = "strictness_tests.rs"]
mod strictness_tests;

fn binary_fixture() -> (FittedModel, feat_math::advanced_v3::RawTable) {
    use crate::{api, budget::Budget, fitted::CreatedBy, limits::Limits, table, task};
    let mut csv = String::from("x,y\n");
    for i in 0..40 {
        csv.push_str(&format!(
            "{},{}\n",
            i % 10,
            if i % 10 > 4 { "yes" } else { "no" }
        ));
    }
    let parsed = table::parse_csv(csv.as_bytes(), &Limits::default()).unwrap();
    let training = table::training(
        &parsed,
        "y",
        task::TaskChoice::Fixed(task::TaskKind::Binary),
        &Limits::default(),
    )
    .unwrap();
    let fitted = api::fit(
        &training,
        &api::FitOptions {
            method: "logistic".into(),
            ..Default::default()
        },
        &Budget::new(None, 1),
        &parsed.sha256,
        CreatedBy::default(),
    )
    .unwrap()
    .fitted;
    let query = table::parse_csv(b"x\n0\n3\n6\n9\n", &Limits::default()).unwrap();
    (
        fitted,
        table::prediction(&query, &training.schema, &training.target)
            .unwrap()
            .0,
    )
}

fn platt_fixture() -> calibration_platt::Calibration {
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    let mut folds = Vec::new();
    for fold in 0..3 {
        for i in 0..20 {
            scores.push(if i < 10 { 0.001 } else { 0.999 });
            labels.push(if i % 10 < (if i < 10 { 2 } else { 8 }) {
                "yes"
            } else {
                "no"
            });
            folds.push(fold);
        }
    }
    calibration_platt::cross_fit(&scores, &labels, ["no", "yes"], &folds)
        .unwrap()
        .calibration
}

fn future() -> ModelObject {
    let state = b"{\"variant\":\"future\",\"operators\":[\"org.example/solver/v7\"]}".to_vec();
    let buffer = vec![0, 255, 0, 128, 1, 7];
    ModelObject::new(
        Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: "org.example/future-gp/v7".into(),
            producer: Producer::default(),
            primary: "state.json".into(),
            required_capabilities: vec!["org.example/solver/v7".into()],
            files: vec![
                Part::new("state.json", "application/json", &state),
                Part::new("weights.bin", "application/octet-stream", &buffer),
            ],
            metadata: json!({"extension": {"unicode": "λ 日本", "float_bits": "8000000000000000"}}),
        },
        BTreeMap::from([("state.json".into(), state), ("weights.bin".into(), buffer)]),
        1024,
    )
    .unwrap()
}
