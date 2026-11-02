use super::*;

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
