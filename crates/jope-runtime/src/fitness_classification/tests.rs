use super::*;

fn spec(metric: &str, aggregation: &str) -> ClassificationFitness {
    spec_with_options(metric, aggregation, None, None)
}

fn spec_with_options(
    metric: &str,
    aggregation: &str,
    threshold: Option<f64>,
    cost_matrix: Option<[[f64; 2]; 2]>,
) -> ClassificationFitness {
    let mut value = serde_json::json!({
        "schema": "jope.classification-fitness/v1",
        "metric": metric,
        "class_order": ["no", "yes"],
        "aggregation": aggregation,
        "undefined": "fail_candidate"
    });
    if let Some(threshold) = threshold {
        value["threshold"] = serde_json::json!(threshold);
    }
    if let Some(cost_matrix) = cost_matrix {
        value["cost_matrix"] = serde_json::json!(cost_matrix);
    }
    ClassificationFitness::parse(&value.to_string()).unwrap()
}

fn inputs<'a>(labels: &'a [&'a str], probabilities: &'a [[f64; 2]]) -> Inputs<'a> {
    Inputs {
        class_order: ["no", "yes"],
        labels,
        probabilities,
    }
}

#[test]
fn hand_computed_golden_values_and_aggregations() {
    // True-class probabilities 1/2, 1/4, 3/4, 1: sum loss ln(32/3).
    // Brier row losses 1/4, 9/16, 1/16, 0: sum 7/8, mean 7/32.
    let labels = ["no", "yes", "no", "yes"];
    let probabilities = [[0.5, 0.5], [0.75, 0.25], [0.75, 0.25], [0.0, 1.0]];
    for (metric, mean, sum) in [
        ("logloss", 0.5917809035329042, 2.367123614131617),
        ("brier", 0.21875, 0.875),
    ] {
        for (aggregation, expected) in [("mean", mean), ("sum", sum)] {
            let objective = spec(metric, aggregation);
            let score = objective.evaluate(inputs(&labels, &probabilities)).unwrap();
            assert!((score.value - expected).abs() <= 1e-12);
            assert_eq!(score.loss, score.value);
            assert_eq!(score.direction, Direction::Minimize);
            assert_eq!(score.fitness_sha256, objective.identity().unwrap());
        }
    }
}

#[test]
fn obj03_metrics_match_sklearn_goldens_with_score_ties() {
    let labels = ["no", "no", "yes", "yes", "no", "yes"];
    let probabilities = [
        [0.9, 0.1],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.9, 0.1],
    ];
    let cost_matrix = [[1.0, 4.0], [2.0, 0.0]];
    let goldens = [
        (spec("roc_auc", "mean"), 0.5),
        (spec("pr_auc", "mean"), 0.5),
        (spec("f1", "mean"), 4.0 / 7.0),
        (spec("balanced_accuracy", "mean"), 0.5),
        (
            spec_with_options("cost_matrix_loss", "mean", None, Some(cost_matrix)),
            11.0 / 6.0,
        ),
    ];
    for (objective, expected) in goldens {
        let score = objective.evaluate(inputs(&labels, &probabilities)).unwrap();
        assert!((score.value - expected).abs() <= 1e-12);
        let direction = objective.metric.direction();
        assert_eq!(score.direction, direction);
        assert_eq!(
            score.loss,
            if direction == Direction::Maximize {
                -score.value
            } else {
                score.value
            }
        );
    }

    // The threshold comparison is inclusive, including ties exactly at the threshold.
    let at_point_nine = spec_with_options("f1", "mean", Some(0.9), None)
        .evaluate(inputs(&labels, &probabilities))
        .unwrap();
    assert!((at_point_nine.value - 0.4).abs() <= 1e-12);
}

#[test]
fn obj03_selection_loss_prefers_better_predictions_for_maximize_metrics() {
    let labels = ["no", "yes"];
    let perfect = [[1.0, 0.0], [0.0, 1.0]];
    let reversed = [[0.0, 1.0], [1.0, 0.0]];
    for metric in ["roc_auc", "pr_auc", "f1", "balanced_accuracy"] {
        let objective = spec(metric, "mean");
        let better = objective.evaluate(inputs(&labels, &perfect)).unwrap();
        let worse = objective.evaluate(inputs(&labels, &reversed)).unwrap();

        assert_eq!(better.direction, Direction::Maximize, "{metric}");
        assert_eq!(worse.direction, Direction::Maximize, "{metric}");
        assert_eq!(better.value, 1.0, "{metric}");
        assert!(worse.value < better.value, "{metric}");
        assert_eq!(better.loss, -better.value, "{metric}");
        assert_eq!(worse.loss, -worse.value, "{metric}");
        assert!(better.loss < worse.loss, "{metric}");
    }
}

#[test]
fn obj03_f1_default_threshold_is_canonical_in_objective_identity() {
    let defaulted = spec("f1", "mean");
    let explicit = spec_with_options("f1", "mean", Some(0.5), None);
    assert_eq!(defaulted.identity().unwrap(), explicit.identity().unwrap());
    assert_eq!(
        defaulted
            .evaluate(inputs(&["no", "yes"], &[[0.5, 0.5], [0.5, 0.5]]))
            .unwrap()
            .value,
        2.0 / 3.0
    );
}

#[test]
fn obj03_single_class_folds_are_undefined_for_new_metrics() {
    let metrics = [
        spec("roc_auc", "mean"),
        spec("pr_auc", "mean"),
        spec("f1", "mean"),
        spec("balanced_accuracy", "mean"),
        spec_with_options(
            "cost_matrix_loss",
            "mean",
            None,
            Some([[1.0, 4.0], [2.0, 0.0]]),
        ),
    ];
    for objective in metrics {
        for single_class in [
            inputs(&["no"], &[[1.0, 0.0]]),
            inputs(&["yes"], &[[0.0, 1.0]]),
        ] {
            assert!(matches!(
                objective.evaluate(single_class),
                Err(FitnessError::Undefined(_))
            ));
        }
    }
}

#[test]
fn obj03_metric_parameters_and_aggregation_are_validated() {
    for invalid in [
        serde_json::json!({"metric":"f1","threshold":-0.1}),
        serde_json::json!({"metric":"f1","threshold":1.1}),
        serde_json::json!({"metric":"roc_auc","threshold":0.5}),
        serde_json::json!({"metric":"brier","cost_matrix":[[0,1],[1,0]]}),
        serde_json::json!({"metric":"cost_matrix_loss"}),
        serde_json::json!({"metric":"cost_matrix_loss","cost_matrix":[[0,-1],[1,0]]}),
    ] {
        let mut value = serde_json::json!({
            "schema":"jope.classification-fitness/v1",
            "metric":"f1",
            "class_order":["no","yes"],
            "aggregation":"mean",
            "undefined":"fail_candidate"
        });
        for (key, item) in invalid.as_object().unwrap() {
            value[key] = item.clone();
        }
        assert!(matches!(
            ClassificationFitness::parse(&value.to_string()),
            Err(FitnessError::InvalidSpecification(_))
        ));
    }
    assert!(ClassificationFitness::parse(
        r#"{"schema":"jope.classification-fitness/v1","metric":"roc_auc","class_order":["no","yes"],"aggregation":"sum","undefined":"fail_candidate"}"#
    )
    .is_err());
    let mut non_finite = spec("f1", "mean");
    non_finite.threshold = Some(f64::NAN);
    assert!(matches!(
        non_finite.validate(),
        Err(FitnessError::InvalidSpecification(_))
    ));
}

#[test]
fn endpoints_single_class_and_small_positive_probabilities() {
    for metric in ["logloss", "brier"] {
        let objective = spec(metric, "mean");
        let perfect = objective
            .evaluate(inputs(&["yes", "yes"], &[[0.0, 1.0], [0.0, 1.0]]))
            .unwrap();
        assert_eq!(perfect.value, 0.0);
        assert_eq!(
            spec("brier", "mean")
                .evaluate(inputs(&["no", "yes"], &[[0.0, 1.0], [1.0, 0.0]]))
                .unwrap()
                .value,
            1.0
        );
    }
    let objective = spec("logloss", "mean");
    assert!(matches!(
        objective.evaluate(inputs(&["yes"], &[[1.0, 0.0]])),
        Err(FitnessError::Undefined(_))
    ));
    let tiny = objective
        .evaluate(inputs(&["yes"], &[[1.0, f64::MIN_POSITIVE]]))
        .unwrap();
    assert!((tiny.value - 708.3964185322641).abs() <= 1e-12);
}

#[test]
fn class_names_order_and_column_mapping_are_explicit() {
    let mut objective = spec("brier", "mean");
    objective.class_order = ["yes".into(), "no".into()];
    let original = inputs(&["yes", "no"], &[[0.25, 0.75], [0.75, 0.25]]);
    assert!(matches!(
        objective.evaluate(original),
        Err(FitnessError::Undefined(_))
    ));
    let swapped = Inputs {
        class_order: ["yes", "no"],
        labels: original.labels,
        probabilities: &[[0.75, 0.25], [0.25, 0.75]],
    };
    assert_eq!(objective.evaluate(swapped).unwrap().value, 0.0625);
}

#[test]
fn malformed_specs_are_rejected_before_scoring() {
    let example = serde_json::to_value(spec("logloss", "mean")).unwrap();
    for field in [
        "schema",
        "metric",
        "class_order",
        "aggregation",
        "undefined",
    ] {
        let mut missing = example.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(ClassificationFitness::parse(&missing.to_string()).is_err());
    }
    for (field, value) in [
        (
            "schema",
            serde_json::json!("jope.classification-fitness/v2"),
        ),
        ("metric", serde_json::json!("unknown_metric")),
        ("aggregation", serde_json::json!("median")),
        ("undefined", serde_json::json!("skip")),
        ("class_order", serde_json::json!(["yes", "yes"])),
        ("class_order", serde_json::json!(["", "yes"])),
        ("class_order", serde_json::json!(["no", "yes", "other"])),
        ("class_order", serde_json::json!([0, 1])),
        ("class_order", serde_json::json!(["no", "x".repeat(257)])),
        ("extra", serde_json::json!(true)),
    ] {
        let mut invalid = example.clone();
        invalid[field] = value;
        assert!(matches!(
            ClassificationFitness::parse(&invalid.to_string()),
            Err(FitnessError::InvalidSpecification(_))
        ));
    }
    let valid = example.to_string();
    for invalid in [
        format!("{valid} {{}}"),
        valid.replacen('{', r#"{"schema":"duplicate", "#, 1),
        " ".repeat(MAX_SPEC_BYTES + 1),
    ] {
        assert!(ClassificationFitness::parse(&invalid).is_err());
    }
    let mut invalid = spec("brier", "sum");
    invalid.schema = "wrong".into();
    assert!(matches!(
        invalid.evaluate(inputs(&[], &[])),
        Err(FitnessError::InvalidSpecification(_))
    ));
}

#[test]
fn invalid_predictions_fail_only_the_candidate_without_a_score() {
    for metric in ["logloss", "brier"] {
        let objective = spec(metric, "mean");
        for invalid in [
            inputs(&[], &[]),
            inputs(&["yes"], &[]),
            inputs(&[], &[[0.5, 0.5]]),
            inputs(&["unknown"], &[[0.5, 0.5]]),
            inputs(&["yes"], &[[f64::NAN, 0.5]]),
            inputs(&["yes"], &[[0.5, f64::INFINITY]]),
            inputs(&["yes"], &[[-0.1, 1.1]]),
            inputs(&["yes"], &[[0.2, 0.3]]),
        ] {
            let error = objective.evaluate(invalid).unwrap_err();
            assert!(matches!(error, FitnessError::Undefined(_)));
            assert!(error.to_string().starts_with("FITNESS_UNDEFINED:"));
        }
        assert!(objective.evaluate(inputs(&["yes"], &[[0.5, 0.5]])).is_ok());
    }
}

#[test]
fn probability_sum_tolerance_does_not_normalize() {
    let objective = spec("logloss", "mean");
    let accepted = objective
        .evaluate(inputs(&["yes"], &[[0.5 + 5e-13, 0.5]]))
        .unwrap();
    assert_eq!(accepted.value, std::f64::consts::LN_2);
    assert!(objective
        .evaluate(inputs(&["yes"], &[[0.5 + 2e-12, 0.5]]))
        .is_err());
}

#[test]
fn schema_examples_roundtrip_and_identity_bind_every_field() {
    for example in schema()["examples"].as_array().unwrap() {
        let parsed = ClassificationFitness::parse(&example.to_string()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), *example);
    }
    let original = spec("logloss", "mean");
    let identity = original.identity().unwrap();
    let reordered = serde_json::json!({
        "undefined": "fail_candidate", "aggregation": "mean", "metric": "logloss",
        "class_order": ["no", "yes"], "schema": FITNESS_VERSION
    });
    assert_eq!(
        ClassificationFitness::parse(&reordered.to_string())
            .unwrap()
            .identity()
            .unwrap(),
        identity
    );
    for changed in [spec("brier", "mean"), spec("logloss", "sum"), {
        let mut changed = original.clone();
        changed.class_order.swap(0, 1);
        changed
    }] {
        assert_ne!(changed.identity().unwrap(), identity);
    }
}

#[test]
fn documented_input_example_has_the_golden_scores() {
    let document = schema();
    let example = &document["$defs"]["inputs"]["examples"][0];
    let order: [String; 2] = serde_json::from_value(example["class_order"].clone()).unwrap();
    let labels: Vec<String> = serde_json::from_value(example["labels"].clone()).unwrap();
    let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    let probabilities: Vec<[f64; 2]> =
        serde_json::from_value(example["probabilities"].clone()).unwrap();
    for (metric, expected) in [("logloss", 0.5917809035329042), ("brier", 0.21875)] {
        let score = spec(metric, "mean")
            .evaluate(Inputs {
                class_order: [order[0].as_str(), order[1].as_str()],
                labels: &labels,
                probabilities: &probabilities,
            })
            .unwrap();
        assert!((score.value - expected).abs() <= 1e-12);
    }
}
