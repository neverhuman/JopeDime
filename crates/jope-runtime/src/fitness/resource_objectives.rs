use super::*;

fn readings(pairs: &[(ResourceVar, f64)]) -> Resources {
    pairs.iter().copied().collect()
}
fn rmse_plus_fit_seconds(weight: f64) -> Fitness {
    Fitness::Composite {
        terms: vec![
            Term {
                weight: 1.0,
                scale: 1.0,
                fitness: Fitness::Rmse {},
            },
            Term {
                weight,
                scale: 1.0,
                fitness: Fitness::Resource {
                    name: ResourceVar::FitSeconds,
                },
            },
        ],
    }
}

#[test]
fn resource_specifications_round_trip_through_json_and_parse() {
    for text in [
        r#"{"metric":"resource","name":"fit_seconds"}"#,
        r#"{"metric":"constrained","fitness":{"metric":"rmse"},"limits":[{"resource":"row_latency_p99_us","max":250.0}]}"#,
        r#"{"metric":"custom","name":"serve","direction":"minimize","expression":{"op":"resource","name":"predict_us_per_row"},"aggregation":{"kind":"mean"}}"#,
    ] {
        let parsed = Fitness::parse(text).unwrap();
        let again: Fitness = serde_json::from_str(&serde_json::to_string(&parsed).unwrap()).unwrap();
        assert_eq!(parsed, again);
        assert!(parsed.identity().is_ok());
    }
    for bad in [
        r#"{"metric":"resource","name":"fit_minutes"}"#,
        r#"{"metric":"constrained","fitness":{"metric":"rmse"},"limits":[]}"#,
        r#"{"metric":"constrained","fitness":{"metric":"rmse"},"limits":[{"resource":"fit_seconds"}]}"#,
        r#"{"metric":"constrained","fitness":{"metric":"rmse"},"limits":[{"resource":"fit_seconds","min":2.0,"max":1.0}]}"#,
    ] {
        assert!(Fitness::parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn existing_identities_are_unchanged_by_the_resource_variants() {
    // Pinned before resource variables existed: adding enum variants must not move them.
    assert_eq!(
        Fitness::Rmse {}.identity().unwrap(),
        crate::sha256_hex(br#"["jope.regression-fitness/v1",{"metric":"rmse"}]"#)
    );
    let pinball = Fitness::parse("pinball:quantile=0.9").unwrap();
    assert_eq!(
        pinball.identity().unwrap(),
        crate::sha256_hex(br#"["jope.regression-fitness/v1",{"metric":"pinball","quantile":0.9}]"#)
    );
    assert!(Fitness::Rmse {}.resources().is_empty());
}

#[test]
fn a_composite_trades_accuracy_against_training_seconds() {
    let y = [1.0, 2.0, 3.0, 4.0];
    let accurate_slow = [1.0, 2.0, 3.0, 4.1];
    let rough_fast = [1.1, 2.1, 2.9, 4.1];
    let f = rmse_plus_fit_seconds(0.05);
    let slow = f
        .loss_with_resources(
            &y,
            &accurate_slow,
            &Columns::new(),
            &readings(&[(ResourceVar::FitSeconds, 10.0)]),
        )
        .unwrap();
    let fast = f
        .loss_with_resources(
            &y,
            &rough_fast,
            &Columns::new(),
            &readings(&[(ResourceVar::FitSeconds, 0.2)]),
        )
        .unwrap();
    assert!(fast < slow, "{fast} {slow}");
    // With a negligible price on time, accuracy wins again.
    let cheap_time = rmse_plus_fit_seconds(1e-6);
    let slow = cheap_time
        .loss_with_resources(
            &y,
            &accurate_slow,
            &Columns::new(),
            &readings(&[(ResourceVar::FitSeconds, 10.0)]),
        )
        .unwrap();
    let fast = cheap_time
        .loss_with_resources(
            &y,
            &rough_fast,
            &Columns::new(),
            &readings(&[(ResourceVar::FitSeconds, 0.2)]),
        )
        .unwrap();
    assert!(slow < fast, "{slow} {fast}");
}

#[test]
fn a_missing_resource_is_an_error_naming_it() {
    let f = rmse_plus_fit_seconds(0.05);
    let error = f.evaluate(&[1.0, 2.0], &[1.0, 2.0]).unwrap_err();
    assert!(format!("{error:#}").contains("fit_seconds"), "{error:#}");
    assert!(require_no_resources(&f, "a test").is_err());
    assert!(require_no_resources(&Fitness::Rmse {}, "a test").is_ok());
    let negative = readings(&[(ResourceVar::FitSeconds, -1.0)]);
    assert!(f
        .evaluate_with_resources(&[1.0], &[1.0], &Columns::new(), &negative)
        .is_err());
}

#[test]
fn a_constrained_objective_refuses_infeasible_candidates_with_a_typed_error() {
    let f = Fitness::parse(
        r#"{"metric":"constrained","fitness":{"metric":"rmse"},
            "limits":[{"resource":"row_latency_p99_us","max":250.0},
                      {"resource":"predict_rows_per_second","min":1000.0}]}"#,
    )
    .unwrap();
    let y = [1.0, 2.0, 3.0];
    let ok = readings(&[
        (ResourceVar::RowLatencyP99Us, 100.0),
        (ResourceVar::PredictRowsPerSecond, 5000.0),
    ]);
    let rmse = Fitness::Rmse {}.evaluate(&y, &[1.0, 2.0, 4.0]).unwrap().loss;
    let score = f
        .evaluate_with_resources(&y, &[1.0, 2.0, 4.0], &Columns::new(), &ok)
        .unwrap();
    assert_eq!(score.loss, rmse);
    assert_eq!(f.direction(), Direction::Minimize);
    let slow = readings(&[
        (ResourceVar::RowLatencyP99Us, 900.0),
        (ResourceVar::PredictRowsPerSecond, 5000.0),
    ]);
    let error = f
        .evaluate_with_resources(&y, &y, &Columns::new(), &slow)
        .unwrap_err();
    let infeasible = error.downcast_ref::<Infeasible>().expect("typed infeasibility");
    assert_eq!(infeasible.resource, ResourceVar::RowLatencyP99Us);
    assert_eq!(infeasible.bound, "max");
    assert!(error.to_string().starts_with("resource_infeasible"));
    let starved = readings(&[
        (ResourceVar::RowLatencyP99Us, 100.0),
        (ResourceVar::PredictRowsPerSecond, 10.0),
    ]);
    let error = f
        .evaluate_with_resources(&y, &y, &Columns::new(), &starved)
        .unwrap_err();
    assert_eq!(error.downcast_ref::<Infeasible>().unwrap().bound, "min");
}

#[test]
fn resource_leaves_broadcast_and_directions_follow_the_variable() {
    let f = Fitness::parse(
        r#"{"metric":"custom","name":"error_plus_serving","direction":"minimize",
            "expression":{"op":"add","left":{"op":"abs","arg":{"op":"y"}},"right":{"op":"constant","value":0.0}},
            "aggregation":{"kind":"mean"}}"#,
    );
    let _ = f;
    let f = Fitness::parse(
        r#"{"metric":"custom","name":"error_plus_serving","direction":"minimize",
            "expression":{"op":"add","left":{"op":"abs","arg":{"op":"sub","left":{"op":"y"},"right":{"op":"pred"}}},
            "right":{"op":"mul","left":{"op":"constant","value":0.5},"right":{"op":"resource","name":"predict_us_per_row"}}},
            "aggregation":{"kind":"mean"}}"#,
    )
    .unwrap();
    let value = f
        .evaluate_with_resources(
            &[1.0, 3.0],
            &[2.0, 3.0],
            &Columns::new(),
            &readings(&[(ResourceVar::PredictUsPerRow, 4.0)]),
        )
        .unwrap()
        .value;
    assert!((value - 2.5).abs() < 1e-12, "{value}"); // mean(|e|) 0.5 + 0.5*4
    assert_eq!(
        Fitness::Resource {
            name: ResourceVar::PredictRowsPerSecond
        }
        .direction(),
        Direction::Maximize
    );
    assert_eq!(
        Fitness::Resource {
            name: ResourceVar::ModelBytes
        }
        .direction(),
        Direction::Minimize
    );
    assert!(!Fitness::Resource {
        name: ResourceVar::ProgramNodes
    }
    .requires_measured_resources());
    assert!(f.requires_measured_resources());
}
