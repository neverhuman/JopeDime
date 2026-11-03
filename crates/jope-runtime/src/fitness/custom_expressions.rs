use super::*;
use crate::fitness::evaluate::columns_from_pairs;

#[test]
fn configuration_parameters_and_expression_limits_are_enforced() {
    for f in [
        Fitness::Huber { delta: 0.0 },
        Fitness::Mape { epsilon: f64::NAN },
        Fitness::Pinball { quantile: 1.0 },
        Fitness::Expectile { quantile: -0.1 },
        Fitness::TweedieDeviance { power: 2.0 },
        Fitness::Composite { terms: vec![] },
        Fitness::Linex { a: 0.0 },
        Fitness::AsymmetricAbsolute {
            under_cost: 0.0,
            over_cost: 1.0,
        },
        Fitness::Mase { season: 0 },
        Fitness::ExpectedShortfall {
            max_position: 1.0,
            cost: 0.0,
            alpha: 1.0,
        },
        Fitness::CaptureAt { fraction: 0.0 },
        // Weights on a metric whose meaning they would change, and nested weighting.
        Fitness::Weighted {
            weights: Expression::Constant { value: 1.0 },
            fitness: Box::new(Fitness::MaxError {}),
        },
        Fitness::Weighted {
            weights: Expression::Constant { value: 1.0 },
            fitness: Box::new(Fitness::Weighted {
                weights: Expression::Constant { value: 1.0 },
                fitness: Box::new(Fitness::Mae {}),
            }),
        },
    ] {
        assert!(f.validate().is_err(), "{f:?}");
    }
    let mut expr = Expression::Y {};
    for _ in 0..18 {
        expr = Expression::Abs {
            arg: Box::new(expr),
        };
    }
    assert!(Fitness::Custom {
        name: "deep".into(),
        direction: Direction::Minimize,
        expression: expr,
        aggregation: RegressionAggregation::Mean {}
    }
    .validate()
    .is_err());
    assert!(Fitness::parse(&" ".repeat(MAX_SPEC_BYTES + 1)).is_err());
    // The node limit covers the whole specification, not each expression in it.
    let chain = |n: usize| {
        let mut e = Expression::Y {};
        for _ in 0..n {
            e = Expression::Add {
                left: Box::new(e),
                right: Box::new(Expression::Constant { value: 1.0 }),
            };
        }
        e
    };
    let custom = |n: usize| Fitness::Custom {
        name: "wide".into(),
        direction: Direction::Minimize,
        expression: chain(n),
        aggregation: RegressionAggregation::Mean {},
    };
    assert!(custom(7).validate().is_ok());
    let composite = |n: usize| Fitness::Composite {
        terms: (0..2)
            .map(|_| Term {
                weight: 1.0,
                scale: 1.0,
                fitness: custom(n),
            })
            .collect(),
    };
    assert!(composite(7).validate().is_ok());
    assert!(composite(100).validate().is_err());
    let f = Fitness::Custom {
        name: "zero_division".into(),
        direction: Direction::Minimize,
        expression: Expression::Div {
            left: Box::new(Expression::Y {}),
            right: Box::new(Expression::Constant { value: 0.0 }),
        },
        aggregation: RegressionAggregation::Mean {},
    };
    assert!(f.evaluate(&[1.0], &[0.0]).is_err());
}
#[test]
fn custom_aggregations_and_scaled_composition_preserve_semantics() {
    let expr = Expression::Abs {
        arg: Box::new(error_expr()),
    };
    for (aggregation, expected) in [
        (RegressionAggregation::Mean {}, 1.0),
        (RegressionAggregation::Sum {}, 3.0),
        (RegressionAggregation::Max {}, 2.0),
        (RegressionAggregation::Quantile { quantile: 0.75 }, 1.5),
        // Largest 2 of 3 rows: (2 + 1) / 2.
        (
            RegressionAggregation::TailMean {
                fraction: 2.0 / 3.0,
            },
            1.5,
        ),
        // Half a row of mass: the largest value alone.
        (
            RegressionAggregation::TailMean {
                fraction: 1.0 / 6.0,
            },
            2.0,
        ),
        (RegressionAggregation::RootMean {}, 1.0),
    ] {
        let f = Fitness::Custom {
            name: "business_loss".into(),
            direction: Direction::Minimize,
            expression: expr.clone(),
            aggregation,
        };
        near(value(&f, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]), expected);
    }
    let f = Fitness::Composite {
        terms: vec![
            Term {
                weight: 2.0,
                scale: 4.0,
                fitness: Fitness::Mae {},
            },
            Term {
                weight: 1.0,
                scale: 2.0,
                fitness: Fitness::R2 {},
            },
        ],
    };
    near(
        f.evaluate(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]).unwrap().loss,
        -0.5,
    );
}
#[test]
fn expression_operators_columns_and_branches_compute_business_costs() {
    // Newsvendor profit with per-row price and unit cost columns.
    let profit = Fitness::parse(
        r#"{"metric":"custom","name":"profit","direction":"maximize",
            "expression":{"op":"sub",
              "left":{"op":"mul","left":{"op":"column","name":"price"},
                      "right":{"op":"min","left":{"op":"y"},"right":{"op":"pred"}}},
              "right":{"op":"mul","left":{"op":"column","name":"unit cost"},"right":{"op":"pred"}}},
            "aggregation":{"kind":"sum"}}"#,
    )
    .unwrap();
    assert_eq!(
        profit.columns().into_iter().collect::<Vec<_>>(),
        ["price", "unit cost"]
    );
    assert!(profit.evaluate(&[1.0], &[1.0]).is_err(), "columns required");
    let columns = columns_from_pairs(&[("price", vec![10.0, 4.0]), ("unit cost", vec![3.0, 1.0])]);
    // Row 0: 10*min(5,6) - 3*6 = 32; row 1: 4*min(4,2) - 1*2 = 6.
    let score = profit
        .evaluate_with(&[5.0, 4.0], &[6.0, 2.0], &columns)
        .unwrap();
    near(score.value, 38.0);
    near(score.loss, -38.0);
    // where/step/neg/pow/log1p/clip: a fixed stockout penalty plus a capped overstock cost.
    let ops = Fitness::parse(
            r#"{"metric":"custom","name":"ops","direction":"minimize",
            "expression":{"op":"add",
              "left":{"op":"mul","left":{"op":"constant","value":7},
                      "right":{"op":"step","arg":{"op":"sub","left":{"op":"y"},"right":{"op":"pred"}}}},
              "right":{"op":"where","condition":{"op":"sub","left":{"op":"pred"},"right":{"op":"y"}},
                "then":{"op":"clip","arg":{"op":"pow","left":{"op":"sub","left":{"op":"pred"},"right":{"op":"y"}},
                        "right":{"op":"constant","value":2}},"min":0,"max":5},
                "else":{"op":"neg","arg":{"op":"log1p","arg":{"op":"constant","value":0}}}}},
            "aggregation":{"kind":"sum"}}"#,
        )
        .unwrap();
    // Short by 1 (7), over by 3 (min(9,5) = 5), exact (0).
    near(value(&ops, &[3.0, 1.0, 2.0], &[2.0, 4.0, 2.0]), 12.0);
}
#[test]
fn weights_reweight_rows_and_refuse_bad_values() {
    let weighted = |fitness: Fitness| Fitness::Weighted {
        weights: Expression::Column {
            name: "exposure".into(),
        },
        fitness: Box::new(fitness),
    };
    let columns = columns_from_pairs(&[("exposure", vec![1.0, 1.0, 2.0])]);
    let y = [1.0, 2.0, 3.0];
    let p = [2.0, 2.0, 2.0];
    let mae = weighted(Fitness::Mae {});
    assert_eq!(mae.name(), "weighted_mae");
    near(mae.evaluate_with(&y, &p, &columns).unwrap().value, 0.75);
    near(
        weighted(Fitness::Rmse {})
            .evaluate_with(&y, &p, &columns)
            .unwrap()
            .value,
        0.75f64.sqrt(),
    );
    // Weighted Wape: (1 + 0 + 2) / (1 + 2 + 6).
    near(
        weighted(Fitness::Wape {})
            .evaluate_with(&y, &p, &columns)
            .unwrap()
            .value,
        3.0 / 9.0,
    );
    // Unit weights reproduce every unweighted metric exactly, quantiles included.
    let ones = columns_from_pairs(&[("exposure", vec![1.0; 3])]);
    let p90 = Fitness::Custom {
        name: "p90_error".into(),
        direction: Direction::Minimize,
        expression: Expression::Abs {
            arg: Box::new(Expression::Sub {
                left: Box::new(Expression::Y {}),
                right: Box::new(Expression::Pred {}),
            }),
        },
        aggregation: RegressionAggregation::Quantile { quantile: 0.9 },
    };
    for f in [
        Fitness::Mae {},
        Fitness::R2 {},
        Fitness::Pinball { quantile: 0.8 },
        Fitness::MedianAbsoluteError {},
        p90.clone(),
    ] {
        near(
            weighted(f.clone())
                .evaluate_with(&y, &p, &ones)
                .unwrap()
                .value,
            value(&f, &y, &p),
        );
    }
    // Ten distinct errors: the 0.9 quantile is 9.1 weighted by ones, as unweighted.
    let ten: Vec<f64> = (1..=10).map(f64::from).collect();
    let ten_ones = columns_from_pairs(&[("exposure", vec![1.0; 10])]);
    near(value(&p90, &ten, &[0.0; 10]), 9.1);
    near(
        weighted(p90)
            .evaluate_with(&ten, &[0.0; 10], &ten_ones)
            .unwrap()
            .value,
        9.1,
    );
    let negative = columns_from_pairs(&[("exposure", vec![1.0, -1.0, 2.0])]);
    assert!(mae.evaluate_with(&y, &p, &negative).is_err());
    let zero = columns_from_pairs(&[("exposure", vec![0.0; 3])]);
    assert!(mae.evaluate_with(&y, &p, &zero).is_err());
}

#[test]
fn execution_telemetry_memory_complexity_and_regime_features_score_and_compose() {
    let telemetry = ExecutionTelemetry {
        train_time_seconds: 0.25,
        inference_time_seconds: 0.05,
        train_rows: 1000,
        inference_rows: 200,
        train_memory_bytes: 4_000_000,
        inference_memory_bytes: 1_000_000,
        model_bytes: 32_768,
        complexity_stages: 4,
        complexity_parameters: 128,
        feature_count: 16,
        fold_std: 0.015,
    };
    let y = [1.0, 2.0, 4.0, 5.0];
    let p = [1.2, 1.8, 3.5, 4.7];

    // Standalone execution & memory metrics
    near(
        Fitness::TrainTimeSeconds {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        0.25,
    );
    near(
        Fitness::InferenceTimeSeconds {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        0.05,
    );
    near(
        Fitness::TotalTimeSeconds {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        0.30,
    );
    near(
        Fitness::InferenceUsPerRow {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        250.0,
    );
    let train_tput = Fitness::TrainRowsPerSecond {}
        .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
        .unwrap();
    assert_eq!(train_tput.direction, Direction::Maximize);
    near(train_tput.value, 4000.0);
    near(train_tput.loss, -4000.0);

    let inf_tput = Fitness::InferenceRowsPerSecond {}
        .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
        .unwrap();
    assert_eq!(inf_tput.direction, Direction::Maximize);
    near(inf_tput.value, 4000.0);
    near(inf_tput.loss, -4000.0);

    near(
        Fitness::TotalMemoryBytes {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        5_000_000.0,
    );
    near(
        Fitness::ModelBytes {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        32_768.0,
    );
    near(
        Fitness::ComplexityStages {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        4.0,
    );
    near(
        Fitness::ComplexityParameters {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        128.0,
    );
    near(
        Fitness::FeatureCount {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        16.0,
    );
    near(
        Fitness::FoldStd {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        0.015,
    );
    // Sequential turnover on p = [1.2, 1.8, 3.5, 4.7] -> (|0.6| + |1.7| + |1.2|) / 3 = 3.5 / 3
    near(
        Fitness::PredictionTurnover {}
            .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
            .unwrap()
            .loss,
        3.5 / 3.0,
    );

    // Custom expression combining business error, recency (row_fraction), train_time, inference_time, and total_memory
    let custom = Fitness::parse(
        r#"{"metric":"custom","name":"regime_resource_loss","direction":"minimize",
            "expression":{"op":"add",
              "left":{"op":"mul",
                "left":{"op":"abs_error"},
                "right":{"op":"add","left":{"op":"constant","value":1.0},"right":{"op":"row_fraction"}}},
              "right":{"op":"add",
                "left":{"op":"mul","left":{"op":"constant","value":0.5},"right":{"op":"train_time_seconds"}},
                "right":{"op":"add",
                  "left":{"op":"mul","left":{"op":"constant","value":2.0},"right":{"op":"inference_time_seconds"}},
                  "right":{"op":"mul","left":{"op":"constant","value":1e-7},"right":{"op":"total_memory_bytes"}}}}},
            "aggregation":{"kind":"mean"}}"#,
    )
    .unwrap();
    assert!(custom.uses_execution_telemetry());
    let custom_score = custom
        .evaluate_with_telemetry(&y, &p, &Columns::new(), &telemetry)
        .unwrap();
    // abs_error = [0.2, 0.2, 0.5, 0.3], (1 + row_fraction) = [1.0, 4/3, 5/3, 2.0]
    // weighted abs_error mean = (0.2*1 + 0.2*(4/3) + 0.5*(5/3) + 0.3*2) / 4 = 1.9 / 4 = 0.475
    // resource term = 0.5*0.25 + 2.0*0.05 + 1e-7*5_000_000 = 0.125 + 0.10 + 0.50 = 0.725
    near(custom_score.loss, 0.475 + 0.725);
}
