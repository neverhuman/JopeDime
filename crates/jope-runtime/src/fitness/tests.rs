use super::*;

use super::*;
fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}");
}
fn error_expr() -> Expression {
    Expression::Sub {
        left: Box::new(Expression::Y {}),
        right: Box::new(Expression::Pred {}),
    }
}
fn value(f: &Fitness, y: &[f64], p: &[f64]) -> f64 {
    f.evaluate(y, p).unwrap().value
}
#[test]
fn aggregation_compatibility_alias_keeps_the_wire_shape() {
    let aggregation = Aggregation::Mean {};
    assert_eq!(
        serde_json::to_value(aggregation).unwrap(),
        serde_json::json!({"kind": "mean"})
    );
}

#[test]
fn standard_metrics_match_independent_hand_calculations() {
    for (spec, expected) in [
        (Fitness::Rmse {}, (2.0f64 / 3.0).sqrt()),
        (Fitness::Mse {}, 2.0 / 3.0),
        (Fitness::Mae {}, 2.0 / 3.0),
        (Fitness::MedianAbsoluteError {}, 1.0),
        (Fitness::MaxError {}, 1.0),
        (Fitness::Mape { epsilon: 1e-8 }, 4.0 / 9.0),
        (Fitness::Smape { epsilon: 1e-8 }, 16.0 / 45.0),
        (Fitness::Wape {}, 1.0 / 3.0),
        (Fitness::R2 {}, 0.0),
        (Fitness::ExplainedVariance {}, 0.0),
        (Fitness::Huber { delta: 1.0 }, 1.0 / 3.0),
        (
            Fitness::PseudoHuber { delta: 1.0 },
            2.0 * (2.0f64.sqrt() - 1.0) / 3.0,
        ),
        (Fitness::LogCosh {}, 2.0 * 1.0f64.cosh().ln() / 3.0),
        (Fitness::Pinball { quantile: 0.9 }, 1.0 / 3.0),
        (Fitness::Expectile { quantile: 0.8 }, 1.0 / 3.0),
        (
            Fitness::Linex { a: 1.0 },
            ((-1.0f64).exp() + 1.0f64.exp() - 2.0) / 3.0,
        ),
        (
            Fitness::AsymmetricAbsolute {
                under_cost: 3.0,
                over_cost: 1.0,
            },
            4.0 / 3.0,
        ),
    ] {
        near(value(&spec, &[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0]), expected);
    }
    // A bounded relative error must not collapse to zero when only the
    // unscaled denominator overflows for finite, same-sign inputs.
    near(
        value(&Fitness::Smape { epsilon: 1e-8 }, &[1.2e308], &[9e307]),
        2.0 / 7.0,
    );
    near(
        value(
            &Fitness::AsymmetricSquared {
                under_cost: 3.0,
                over_cost: 1.0,
            },
            &[0.0, 2.0],
            &[2.0, 0.0],
        ),
        8.0,
    );
    near(
        value(
            &Fitness::Rmspe { epsilon: 1e-8 },
            &[1.0, 2.0, 4.0],
            &[2.0, 2.0, 2.0],
        ),
        (1.25f64 / 3.0).sqrt(),
    );
}
#[test]
fn log_and_deviance_metrics_match_independent_calculations() {
    near(
        value(&Fitness::Msle {}, &[0.0, 3.0], &[1.0, 1.0]),
        2.0f64.ln().powi(2),
    );
    near(
        value(&Fitness::Rmsle {}, &[0.0, 3.0], &[1.0, 1.0]),
        2.0f64.ln(),
    );
    near(
        value(
            &Fitness::PoissonDeviance {},
            &[0.0, 1.0, 2.0],
            &[1.0, 1.0, 1.0],
        ),
        4.0 * 2.0f64.ln() / 3.0,
    );
    near(
        value(&Fitness::GammaDeviance {}, &[1.0, 2.0], &[2.0, 1.0]),
        0.5,
    );
    near(
        value(
            &Fitness::TweedieDeviance { power: 1.5 },
            &[0.0, 1.0, 4.0],
            &[1.0, 1.0, 1.0],
        ),
        8.0 / 3.0,
    );
}
#[test]
fn correlations_handle_ties_signed_zero_and_direction() {
    near(
        Fitness::Pearson {}
            .evaluate(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0])
            .unwrap()
            .loss,
        1.0,
    );
    near(
        Fitness::Spearman {}
            .evaluate(&[0.0, -0.0, 2.0], &[2.0, 2.0, 5.0])
            .unwrap()
            .loss,
        -1.0,
    );
    let r = Fitness::R2 {}
        .evaluate(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0])
        .unwrap();
    assert_eq!(r.direction, Direction::Maximize);
    near(r.value, 1.0);
    near(r.loss, -1.0);
}
#[test]
fn ranking_hydrology_and_forecast_scale_metrics_match_hand_calculations() {
    // A variance that overflows fails the score instead of returning a finite wrong value.
    assert!(Fitness::Pearson {}
        .evaluate(&[-1e155, 1e155], &[-1.0, 1.0])
        .is_err());
    assert!(Fitness::R2 {}
        .evaluate(&[-1e200, 1e200, 0.0], &[0.0, 0.0, 0.0])
        .is_err());
    // Kaggle-style normalized Gini: 0.25 / 0.3125.
    let claims = [1.0, 0.0, 0.0, 3.0];
    let risk = [0.2, 0.1, 0.3, 0.9];
    near(value(&Fitness::NormalizedGini {}, &claims, &risk), 0.8);
    near(value(&Fitness::NormalizedGini {}, &claims, &claims), 1.0);
    near(
        value(&Fitness::CaptureAt { fraction: 0.25 }, &claims, &risk),
        0.75,
    );
    near(
        value(&Fitness::CaptureAt { fraction: 0.375 }, &claims, &risk),
        0.75,
    );
    // Tied predictions earn nothing from the rows' order: a constant prediction on rows
    // sorted by claims ranks nothing, and a tie shares its claims across the boundary.
    let sorted = [3.0, 1.0, 0.0, 0.0];
    near(value(&Fitness::NormalizedGini {}, &sorted, &[0.5; 4]), 0.0);
    near(
        value(&Fitness::CaptureAt { fraction: 0.25 }, &sorted, &[0.5; 4]),
        0.25,
    );
    near(
        value(
            &Fitness::CaptureAt { fraction: 0.25 },
            &sorted,
            &[0.9, 0.9, 0.1, 0.1],
        ),
        0.5,
    );
    near(
        value(&Fitness::Kge {}, &[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]),
        1.0 - 2.0f64.sqrt(),
    );
    near(
        value(
            &Fitness::HitRate {},
            &[1.0, -1.0, 2.0, 0.0],
            &[0.5, 0.5, -1.0, 3.0],
        ),
        1.0 / 3.0,
    );
    let mase = Fitness::Mase { season: 1 };
    assert!(mase.requires_order());
    near(
        value(&mase, &[1.0, 2.0, 3.0, 5.0], &[1.0, 1.0, 1.0, 1.0]),
        21.0 / 16.0,
    );
    assert!(mase.evaluate(&[1.0, 1.0], &[0.0, 0.0]).is_err());
}
#[test]
fn malformed_nonfinite_and_undefined_values_never_become_winners() {
    for (y, p) in [
        (vec![], vec![]),
        (vec![1.0], vec![]),
        (vec![f64::NAN], vec![0.0]),
        (vec![1.0], vec![f64::INFINITY]),
    ] {
        assert!(Fitness::Rmse {}.evaluate(&y, &p).is_err());
    }
    assert!(Fitness::Mse {}.evaluate(&[f64::MAX], &[-f64::MAX]).is_err());
    assert!(Fitness::MedianAbsoluteError {}
        .evaluate(&[f64::MAX, 1.0, 2.0], &[-f64::MAX, 1.0, 2.0])
        .is_err());
    for f in [
        Fitness::R2 {},
        Fitness::ExplainedVariance {},
        Fitness::Pearson {},
        Fitness::Spearman {},
        Fitness::Wape {},
        Fitness::HitRate {},
        Fitness::NormalizedGini {},
        Fitness::Kge {},
        Fitness::CaptureAt { fraction: 0.5 },
    ] {
        assert!(f.evaluate(&[0.0, 0.0], &[1.0, 2.0]).is_err(), "{f:?}");
    }
    assert!(Fitness::Msle {}.evaluate(&[-1.0], &[1.0]).is_err());
    assert!(Fitness::PoissonDeviance {}
        .evaluate(&[1.0], &[0.0])
        .is_err());
    assert!(Fitness::GammaDeviance {}.evaluate(&[0.0], &[1.0]).is_err());
    assert!(Fitness::Linex { a: 1000.0 }
        .evaluate(&[10.0], &[0.0])
        .is_err());
    assert!(Fitness::parse("{\"metric\":\"mae\",\"typo\":1}").is_err());
    assert!(Fitness::parse("import os").is_err());
}
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
fn return_ratios_use_positions_turnover_full_sample_downside_and_annualization() {
    let y = [0.1, -0.2, 0.3];
    let p = [1.0, 1.0, 1.0];
    let sortino = Fitness::Sortino {
        max_position: 1.0,
        cost: 0.0,
        target: 0.0,
        periods_per_year: 1.0,
    };
    near(
        value(&sortino, &y, &p),
        (0.2 / 3.0) / (0.04f64 / 3.0).sqrt(),
    );
    let sharpe = Fitness::Sharpe {
        max_position: 1.0,
        cost: 0.0,
        target: 0.0,
        periods_per_year: 4.0,
    };
    near(
        value(&sharpe, &y, &p),
        (0.2 / 3.0) / (0.12666666666666668f64 / 3.0).sqrt() * 2.0,
    );
    near(
        value(
            &Fitness::MeanReturn {
                max_position: 1.0,
                cost: 0.01,
            },
            &[0.1, 0.1],
            &[2.0, -2.0],
        ),
        -0.015,
    );
    assert!(sortino.requires_order());
    assert!(sortino.evaluate(&[0.1, 0.2], &[1.0, 1.0]).is_err());
    assert!(sharpe.evaluate(&[0.1, 0.1], &[1.0, 1.0]).is_err());
    // Overflow is an error, not a finite score: an infinite variance, an exploding equity.
    assert!(sharpe.evaluate(&[1e200, -1e200], &[1.0, 1.0]).is_err());
    assert!(Fitness::MaxDrawdown {
        max_position: 1.0,
        cost: 0.0
    }
    .evaluate(&[1e200, 1e200], &[1.0, 1.0])
    .is_err());
    // −0.0 and 0.0 are one objective.
    assert_eq!(
        Fitness::parse("sharpe:cost=-0.0")
            .unwrap()
            .identity()
            .unwrap(),
        Fitness::parse("sharpe:cost=0").unwrap().identity().unwrap()
    );
}
#[test]
fn drawdown_calmar_omega_and_expected_shortfall_follow_their_definitions() {
    let y = [0.1, -0.2, 0.3];
    let p = [1.0, 1.0, 1.0];
    let (max_position, cost) = (1.0, 0.0);
    // Equity 1.1, 0.88, 1.144: the drawdown from 1.1 to 0.88 is 20%.
    near(
        value(&Fitness::MaxDrawdown { max_position, cost }, &y, &p),
        0.2,
    );
    near(
        value(
            &Fitness::Calmar {
                max_position,
                cost,
                periods_per_year: 1.0,
            },
            &y,
            &p,
        ),
        (1.144f64.powf(1.0 / 3.0) - 1.0) / 0.2,
    );
    near(
        value(
            &Fitness::Omega {
                max_position,
                cost,
                threshold: 0.0,
            },
            &y,
            &p,
        ),
        2.0,
    );
    let shortfall = |alpha| Fitness::ExpectedShortfall {
        max_position,
        cost,
        alpha,
    };
    near(value(&shortfall(1.0 / 3.0), &y, &p), 0.2);
    // 1.5 rows of tail mass: all of the -0.2 loss and half of the +0.1 gain.
    near(value(&shortfall(0.5), &y, &p), 0.1);
    // A total loss ends the equity curve: 100% drawdown.
    near(
        value(
            &Fitness::MaxDrawdown { max_position, cost },
            &[-1.5, 0.5],
            &[1.0, 1.0],
        ),
        1.0,
    );
    for f in [
        Fitness::Calmar {
            max_position,
            cost,
            periods_per_year: 1.0,
        },
        Fitness::Omega {
            max_position,
            cost,
            threshold: 0.0,
        },
    ] {
        assert!(f.evaluate(&[0.1, 0.2], &[1.0, 1.0]).is_err(), "{f:?}");
    }
}
#[test]
fn catalog_and_custom_identities_round_trip_and_bind_every_parameter() {
    let catalog = Fitness::catalog();
    assert_eq!(catalog.len(), 38);
    let names: BTreeSet<_> = catalog.iter().map(|e| e.name).collect();
    assert_eq!(names.len(), catalog.len(), "duplicate catalog name");
    for entry in catalog {
        assert_eq!(entry.fitness.name(), entry.name);
        assert!(!entry.description.is_empty());
        let json = serde_json::to_string(&entry.fitness).unwrap();
        assert_eq!(Fitness::parse(entry.name).unwrap(), entry.fitness);
        assert_eq!(
            Fitness::parse(&json).unwrap().identity().unwrap(),
            entry.fitness.identity().unwrap()
        );
    }
    assert_ne!(
        Fitness::Pinball { quantile: 0.5 }.identity().unwrap(),
        Fitness::Pinball { quantile: 0.9 }.identity().unwrap()
    );
    assert_ne!(
        Fitness::Mae {}.identity().unwrap(),
        Fitness::Mse {}.identity().unwrap()
    );
}
#[test]
fn shorthand_overrides_numeric_defaults_and_refuses_unknown_parameters() {
    assert_eq!(
        Fitness::parse("pinball:quantile=0.9").unwrap(),
        Fitness::Pinball { quantile: 0.9 }
    );
    assert_eq!(
        Fitness::parse("sortino:cost=0.0005, periods_per_year=52").unwrap(),
        Fitness::Sortino {
            max_position: 1.0,
            cost: 0.0005,
            target: 0.0,
            periods_per_year: 52.0
        }
    );
    assert_eq!(
        Fitness::parse("mase:season=7").unwrap(),
        Fitness::Mase { season: 7 }
    );
    for bad in [
        "pinball:quantile=1.5",
        "pinball:quantlie=0.9",
        "pinball:metric=1",
        "pinball:quantile=high",
        "pinball:quantile",
        "mase:season=0.5",
    ] {
        assert!(Fitness::parse(bad).is_err(), "{bad}");
    }
}
#[test]
fn all_checked_in_fitness_examples_parse_and_score() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/objectives");
    let mut count = 0;
    let mut paths: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    for path in paths {
        if path.extension().is_some_and(|e| e == "json")
            && !path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("request")
        {
            let f = Fitness::parse(&std::fs::read_to_string(&path).unwrap())
                .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            let n = 12;
            let positive = !(f.requires_order() || matches!(f, Fitness::HitRate {}));
            let targets: Vec<f64> = (0..n)
                .map(|i| {
                    let v = 0.01 * (1 + i % 5) as f64;
                    if positive || i % 3 != 0 {
                        v
                    } else {
                        -v
                    }
                })
                .collect();
            let predictions: Vec<f64> = (0..n).map(|i| 0.012 * (1 + i % 4) as f64).collect();
            let columns: Columns = f
                .columns()
                .into_iter()
                .map(|name| (name, (0..n).map(|i| 1.0 + (i % 3) as f64).collect()))
                .collect();
            f.evaluate_with(&targets, &predictions, &columns)
                .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            count += 1;
        }
    }
    assert!(count >= 24, "{count} recipes");
}
