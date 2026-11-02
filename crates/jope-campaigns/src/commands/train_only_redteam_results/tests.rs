use super::*;

#[test]
fn regression_metrics_match_fixed_fixture() {
    let (rmse, mae, r2) = regression_metrics(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0]).unwrap();
    assert!((rmse - (1.0_f64 / 3.0).sqrt()).abs() < 1e-15);
    assert!((mae - 1.0 / 3.0).abs() < 1e-15);
    assert!((r2 - 0.5).abs() < 1e-15);
}

#[test]
fn smart_v3_acceptance_uses_wins_and_geometric_mean() {
    let comparisons = [
        (0.9, 1.0),
        (0.9, 1.0),
        (0.9, 1.0),
        (0.9, 1.0),
        (0.9, 1.0),
        (0.9, 1.0),
        (1.0, 1.0),
        (1.05, 1.0),
        (1.05, 1.0),
        (1.05, 1.0),
    ];
    let aggregate = smart_v3_aggregate(&comparisons, 1e-12, 6).unwrap();
    let expected_geometric_mean = (0.9_f64.powi(6) * 1.05_f64.powi(3)).powf(0.1);
    assert_eq!(
        (aggregate.wins, aggregate.ties, aggregate.losses),
        (6, 1, 3)
    );
    assert!((aggregate.median_relative_rmse_change + 0.1).abs() < 1e-15);
    assert!((aggregate.geometric_mean_rmse_ratio - expected_geometric_mean).abs() < 1e-15);
    assert!(aggregate.smart_v3_improved);
}

#[test]
fn geometric_mean_alone_cannot_pass_acceptance() {
    let comparisons = [
        (0.5, 1.0),
        (0.5, 1.0),
        (0.5, 1.0),
        (0.5, 1.0),
        (0.5, 1.0),
        (1.01, 1.0),
        (1.01, 1.0),
        (1.01, 1.0),
        (1.01, 1.0),
        (1.01, 1.0),
    ];
    let aggregate = smart_v3_aggregate(&comparisons, 1e-12, 6).unwrap();
    assert_eq!(aggregate.wins, 5);
    assert!(aggregate.geometric_mean_rmse_ratio < 1.0);
    assert!(!aggregate.smart_v3_improved);
}

#[test]
fn each_baseline_gets_independent_strict_aggregate_metrics() {
    let comparisons = [
        (0.9, 1.0),
        (0.9, 1.0),
        (0.9, 1.0),
        (0.9, 1.0),
        (0.9, 1.0),
        (0.9, 1.0),
        (1.0, 1.0),
        (1.1, 1.0),
        (1.1, 1.0),
        (1.1, 1.0),
    ];
    let aggregate = smart_v3_baseline_aggregate(&comparisons, 1e-12, 6).unwrap();
    assert_eq!(
        (aggregate.wins, aggregate.ties, aggregate.losses),
        (6, 1, 3)
    );
    assert!(aggregate.smart_v3_beats_baseline);
    assert!((aggregate.median_relative_rmse_change + 0.1).abs() < 1e-15);
    let expected = (0.9_f64.powi(6) * 1.1_f64.powi(3)).powf(0.1);
    assert!((aggregate.geometric_mean_rmse_ratio - expected).abs() < 1e-15);
}
