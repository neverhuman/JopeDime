use super::*;

fn folds(n: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    (0..HYPERION_ROUTER_FOLDS)
        .map(|fold| {
            let hold: Vec<usize> = (0..n)
                .filter(|row| row % HYPERION_ROUTER_FOLDS == fold)
                .collect();
            let fit: Vec<usize> = (0..n)
                .filter(|row| row % HYPERION_ROUTER_FOLDS != fold)
                .collect();
            (fit, hold)
        })
        .collect()
}

fn strong_evidence(seed: u64) -> HyperionRegressionSeedReport {
    let y: Vec<f32> = (0..70).map(|row| row as f32).collect();
    let baseline: Vec<f32> = y.iter().map(|&value| value + 10.0).collect();
    let candidate: Vec<f32> = y.iter().map(|&value| value + 5.0).collect();
    evaluate_regression_seed(
        seed,
        &y,
        &baseline,
        &candidate,
        &folds(y.len()),
        &HyperionRegressionThresholds::default(),
    )
}

#[test]
fn strong_repeatable_cv7_evidence_routes_to_cells() {
    let thresholds = HyperionRegressionThresholds::default();
    let first = strong_evidence(303);
    let again = strong_evidence(303);
    assert_eq!(first.bootstrap_lower_bound, again.bootstrap_lower_bound);
    assert!(first.passed, "{:?}", first.failures);
    assert_eq!(first.fold_wins, 7);
    let decision = choose_regression_route(vec![first, strong_evidence(404)], &thresholds);
    assert!(decision.passed);
    assert_eq!(decision.route, HyperionRegressionRoute::RawPlusCell);
}

#[test]
fn tie_and_single_seed_evidence_fail_closed_to_raw() {
    let thresholds = HyperionRegressionThresholds::default();
    let y: Vec<f32> = (0..70).map(|row| row as f32).collect();
    let baseline: Vec<f32> = y.iter().map(|&value| value + 1.0).collect();
    let tie = evaluate_regression_seed(303, &y, &baseline, &baseline, &folds(y.len()), &thresholds);
    assert!(!tie.passed);
    assert_eq!(
        choose_regression_route(vec![tie], &thresholds).route,
        HyperionRegressionRoute::Raw
    );
}

#[test]
fn minimum_row_and_worst_fold_guards_are_enforced() {
    let thresholds = HyperionRegressionThresholds::default();
    let y: Vec<f32> = (0..35).map(|row| row as f32).collect();
    let baseline: Vec<f32> = y.iter().map(|&value| value + 2.0).collect();
    let candidate: Vec<f32> = y.iter().map(|&value| value + 1.0).collect();
    let too_small =
        evaluate_regression_seed(303, &y, &baseline, &candidate, &folds(y.len()), &thresholds);
    assert!(!too_small.passed);
    assert!(too_small
        .failures
        .iter()
        .any(|failure| failure.contains("fit rows")));

    let y: Vec<f32> = (0..70).map(|row| row as f32).collect();
    let baseline: Vec<f32> = y.iter().map(|&value| value + 10.0).collect();
    let mut candidate: Vec<f32> = y.iter().map(|&value| value + 1.0).collect();
    for row in (0..70).filter(|row| row % HYPERION_ROUTER_FOLDS == 0) {
        candidate[row] = y[row] + 11.0;
    }
    let unstable =
        evaluate_regression_seed(303, &y, &baseline, &candidate, &folds(y.len()), &thresholds);
    assert!(!unstable.passed);
    assert!(unstable
        .failures
        .iter()
        .any(|failure| failure.contains("regressed by more than 2%")));
}

#[test]
fn nonfinite_and_nonpartitioning_inputs_fail_closed() {
    let thresholds = HyperionRegressionThresholds::default();
    let y: Vec<f32> = (0..70).map(|row| row as f32).collect();
    let baseline = y.clone();
    let mut candidate = y.clone();
    candidate[3] = f32::NAN;
    let mut bad_folds = folds(y.len());
    bad_folds[0].1.pop();
    let report = evaluate_regression_seed(303, &y, &baseline, &candidate, &bad_folds, &thresholds);
    assert!(!report.passed);
    assert!(report
        .failures
        .iter()
        .any(|failure| failure.contains("non-finite")));
    assert!(report
        .failures
        .iter()
        .any(|failure| failure.contains("partition")));
}

#[test]
fn relative_and_normalized_improvement_thresholds_are_independent() {
    let thresholds = HyperionRegressionThresholds::default();
    let y: Vec<f32> = (0..70).map(|row| row as f32).collect();
    let baseline: Vec<f32> = y.iter().map(|&value| value + 1.0).collect();

    let below_relative: Vec<f32> = y.iter().map(|&value| value + 0.996).collect();
    let report = evaluate_regression_seed(
        303,
        &y,
        &baseline,
        &below_relative,
        &folds(y.len()),
        &thresholds,
    );
    assert!(report
        .failures
        .iter()
        .any(|failure| failure.contains("below 0.5%")));

    let relative_passes_but_normalized_fails: Vec<f32> =
        y.iter().map(|&value| value + 0.99).collect();
    let report = evaluate_regression_seed(
        303,
        &y,
        &baseline,
        &relative_passes_but_normalized_fails,
        &folds(y.len()),
        &thresholds,
    );
    assert!(report.relative_rmse_improvement >= 0.005);
    assert!(report
        .failures
        .iter()
        .any(|failure| failure.contains("normalized")));
}

#[test]
fn fold_win_and_bootstrap_guards_are_enforced() {
    let thresholds = HyperionRegressionThresholds::default();
    let y: Vec<f32> = (0..70).map(|row| row as f32).collect();
    let baseline: Vec<f32> = y.iter().map(|&value| value + 10.0).collect();
    let mut candidate = Vec::with_capacity(y.len());
    for (row, &value) in y.iter().enumerate() {
        let fold = row % HYPERION_ROUTER_FOLDS;
        candidate.push(value + if fold < 4 { 8.0 } else { 10.1 });
    }
    let report =
        evaluate_regression_seed(303, &y, &baseline, &candidate, &folds(y.len()), &thresholds);
    assert_eq!(report.fold_wins, 4);
    assert!(report
        .failures
        .iter()
        .any(|failure| failure.contains("fewer than 5")));

    let mut unstable = Vec::with_capacity(y.len());
    for (row, &value) in y.iter().enumerate() {
        unstable.push(
            value
                + if row % HYPERION_ROUTER_FOLDS == 0 {
                    20.0
                } else {
                    5.0
                },
        );
    }
    let report =
        evaluate_regression_seed(303, &y, &baseline, &unstable, &folds(y.len()), &thresholds);
    assert!(report.fold_wins >= 5);
    assert!(report.bootstrap_lower_bound <= 0.0);
    assert!(report
        .failures
        .iter()
        .any(|failure| failure.contains("bootstrap")));
}

#[test]
fn both_seeds_and_one_percent_median_are_mandatory() {
    let thresholds = HyperionRegressionThresholds::default();
    let mut first = strong_evidence(303);
    let mut second = strong_evidence(404);
    first.relative_rmse_improvement = 0.009;
    second.relative_rmse_improvement = 0.009;
    let decision = choose_regression_route(vec![first, second], &thresholds);
    assert_eq!(decision.route, HyperionRegressionRoute::Raw);
    assert!(!decision.passed);
    assert!(decision.reason.contains("below 1%"));
}
