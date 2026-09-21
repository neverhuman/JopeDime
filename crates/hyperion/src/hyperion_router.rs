//! Conservative, train-only routing for the Hyperion cell feature block.
//!
//! The router consumes paired out-of-fold predictions made on identical folds. It never sees an
//! external holdout/test label. Positive values mean that raw + Hyperion cells lowered RMSE.

use crate::rng::seeded;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const HYPERION_ROUTER_SEEDS: [u64; 2] = [303, 404];
pub const HYPERION_ROUTER_FOLDS: usize = 7;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HyperionRegressionThresholds {
    pub min_fit_rows: usize,
    pub min_hold_rows: usize,
    pub min_relative_rmse_improvement: f64,
    pub min_normalized_rmse_improvement: f64,
    pub min_fold_wins: usize,
    pub max_fold_relative_regression: f64,
    pub min_median_relative_rmse_improvement: f64,
    pub bootstrap_samples: usize,
    pub bootstrap_alpha: f64,
}

impl Default for HyperionRegressionThresholds {
    fn default() -> Self {
        Self {
            min_fit_rows: 32,
            min_hold_rows: 2,
            min_relative_rmse_improvement: 0.005,
            min_normalized_rmse_improvement: 0.005,
            min_fold_wins: 5,
            max_fold_relative_regression: 0.02,
            min_median_relative_rmse_improvement: 0.01,
            bootstrap_samples: 4096,
            bootstrap_alpha: 0.05,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct HyperionRegressionSeedReport {
    pub seed: u64,
    pub folds: usize,
    pub atomic_groups: usize,
    pub hold_groups_per_fold: Vec<usize>,
    pub baseline_rmse: f64,
    pub candidate_rmse: f64,
    pub relative_rmse_improvement: f64,
    pub normalized_rmse_improvement: f64,
    pub target_std: f64,
    pub fold_wins: usize,
    pub worst_fold_relative_regression: f64,
    pub bootstrap_lower_bound: f64,
    pub baseline_fold_rmse: Vec<f64>,
    pub candidate_fold_rmse: Vec<f64>,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failures: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HyperionRegressionRoute {
    #[default]
    Raw,
    RawPlusCell,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct HyperionRegressionRouteDecision {
    pub route: HyperionRegressionRoute,
    pub median_relative_rmse_improvement: f64,
    pub seed_reports: Vec<HyperionRegressionSeedReport>,
    pub passed: bool,
    pub reason: String,
}

fn rmse_rows(y: &[f32], pred: &[f32], rows: &[usize]) -> f64 {
    let sum = rows
        .iter()
        .map(|&row| {
            let d = y[row] as f64 - pred[row] as f64;
            d * d
        })
        .sum::<f64>();
    (sum / rows.len().max(1) as f64).sqrt()
}

fn target_std(y: &[f32]) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    let mean = y.iter().map(|&v| v as f64).sum::<f64>() / y.len() as f64;
    (y.iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / y.len() as f64)
        .sqrt()
}

fn relative_improvement(baseline_rmse: f64, candidate_rmse: f64) -> f64 {
    if baseline_rmse > 0.0 && baseline_rmse.is_finite() && candidate_rmse.is_finite() {
        (baseline_rmse - candidate_rmse) / baseline_rmse
    } else {
        f64::NEG_INFINITY
    }
}

fn grouped_bootstrap_lower_bound(
    y: &[f32],
    baseline_oof: &[f32],
    candidate_oof: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    seed: u64,
    samples: usize,
    alpha: f64,
) -> f64 {
    if folds.is_empty() || samples == 0 {
        return f64::NEG_INFINITY;
    }
    let mut rng = seeded(seed ^ 0x4859_5045_5249_4f4e);
    let mut gains = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut baseline_sse = 0.0;
        let mut candidate_sse = 0.0;
        let mut rows = 0usize;
        for _ in 0..folds.len() {
            let fold = rng.gen_range(0..folds.len());
            for &row in &folds[fold].1 {
                let baseline_error = y[row] as f64 - baseline_oof[row] as f64;
                let candidate_error = y[row] as f64 - candidate_oof[row] as f64;
                baseline_sse += baseline_error * baseline_error;
                candidate_sse += candidate_error * candidate_error;
                rows += 1;
            }
        }
        let baseline = (baseline_sse / rows.max(1) as f64).sqrt();
        let candidate = (candidate_sse / rows.max(1) as f64).sqrt();
        gains.push(relative_improvement(baseline, candidate));
    }
    gains.sort_by(f64::total_cmp);
    let index = ((samples - 1) as f64 * alpha.clamp(0.0, 1.0)).floor() as usize;
    gains[index]
}

fn validate_inputs(
    y: &[f32],
    baseline_oof: &[f32],
    candidate_oof: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    thresholds: &HyperionRegressionThresholds,
) -> Vec<String> {
    let mut failures = Vec::new();
    if y.is_empty() || baseline_oof.len() != y.len() || candidate_oof.len() != y.len() {
        failures.push("OOF/target shape mismatch".into());
        return failures;
    }
    if folds.len() != HYPERION_ROUTER_FOLDS {
        failures.push(format!(
            "requires exactly {HYPERION_ROUTER_FOLDS} folds, observed {}",
            folds.len()
        ));
    }
    if y.iter()
        .chain(baseline_oof)
        .chain(candidate_oof)
        .any(|v| !v.is_finite())
    {
        failures.push("non-finite target or OOF prediction".into());
    }
    let mut coverage = vec![0u8; y.len()];
    for (index, (fit, hold)) in folds.iter().enumerate() {
        if fit.len() < thresholds.min_fit_rows {
            failures.push(format!(
                "fold {index} has fewer than {} fit rows",
                thresholds.min_fit_rows
            ));
        }
        if hold.len() < thresholds.min_hold_rows {
            failures.push(format!(
                "fold {index} has fewer than {} hold rows",
                thresholds.min_hold_rows
            ));
        }
        for &row in hold {
            if row >= y.len() {
                failures.push(format!("fold {index} hold row {row} is out of range"));
            } else {
                coverage[row] = coverage[row].saturating_add(1);
            }
        }
    }
    if coverage.iter().any(|&count| count != 1) {
        failures.push("hold folds do not partition every training row exactly once".into());
    }
    failures
}

pub fn evaluate_regression_seed(
    seed: u64,
    y: &[f32],
    baseline_oof: &[f32],
    candidate_oof: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    thresholds: &HyperionRegressionThresholds,
) -> HyperionRegressionSeedReport {
    let mut report = HyperionRegressionSeedReport {
        seed,
        folds: folds.len(),
        failures: validate_inputs(y, baseline_oof, candidate_oof, folds, thresholds),
        ..HyperionRegressionSeedReport::default()
    };
    if !report.failures.is_empty() {
        return report;
    }

    let rows: Vec<usize> = (0..y.len()).collect();
    report.baseline_rmse = rmse_rows(y, baseline_oof, &rows);
    report.candidate_rmse = rmse_rows(y, candidate_oof, &rows);
    report.relative_rmse_improvement =
        relative_improvement(report.baseline_rmse, report.candidate_rmse);
    report.target_std = target_std(y);
    report.normalized_rmse_improvement = if report.target_std > 0.0 {
        (report.baseline_rmse - report.candidate_rmse) / report.target_std
    } else {
        f64::NEG_INFINITY
    };
    report.baseline_fold_rmse = folds
        .iter()
        .map(|(_, hold)| rmse_rows(y, baseline_oof, hold))
        .collect();
    report.candidate_fold_rmse = folds
        .iter()
        .map(|(_, hold)| rmse_rows(y, candidate_oof, hold))
        .collect();
    report.fold_wins = report
        .baseline_fold_rmse
        .iter()
        .zip(&report.candidate_fold_rmse)
        .filter(|(baseline, candidate)| candidate < baseline)
        .count();
    report.worst_fold_relative_regression = report
        .baseline_fold_rmse
        .iter()
        .zip(&report.candidate_fold_rmse)
        .map(|(&baseline, &candidate)| {
            if baseline > 0.0 {
                (candidate - baseline) / baseline
            } else if candidate == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .fold(f64::NEG_INFINITY, f64::max);
    report.bootstrap_lower_bound = grouped_bootstrap_lower_bound(
        y,
        baseline_oof,
        candidate_oof,
        folds,
        seed,
        thresholds.bootstrap_samples,
        thresholds.bootstrap_alpha,
    );

    if report.relative_rmse_improvement < thresholds.min_relative_rmse_improvement {
        report
            .failures
            .push("relative RMSE improvement below 0.5%".into());
    }
    if report.normalized_rmse_improvement < thresholds.min_normalized_rmse_improvement {
        report
            .failures
            .push("target-normalized RMSE improvement below 0.005".into());
    }
    if report.fold_wins < thresholds.min_fold_wins {
        report.failures.push(format!(
            "won fewer than {} of {} folds",
            thresholds.min_fold_wins, HYPERION_ROUTER_FOLDS
        ));
    }
    if report.worst_fold_relative_regression > thresholds.max_fold_relative_regression {
        report
            .failures
            .push("a fold regressed by more than 2%".into());
    }
    if !(report.bootstrap_lower_bound > 0.0) {
        report
            .failures
            .push("one-sided 95% grouped-bootstrap lower bound is not positive".into());
    }
    report.passed = report.failures.is_empty();
    report
}

pub fn choose_regression_route(
    mut reports: Vec<HyperionRegressionSeedReport>,
    thresholds: &HyperionRegressionThresholds,
) -> HyperionRegressionRouteDecision {
    reports.sort_by_key(|report| report.seed);
    let expected = HYPERION_ROUTER_SEEDS;
    let observed: Vec<u64> = reports.iter().map(|report| report.seed).collect();
    if observed != expected {
        return HyperionRegressionRouteDecision {
            seed_reports: reports,
            reason: format!("requires seed reports {expected:?}, observed {observed:?}"),
            ..HyperionRegressionRouteDecision::default()
        };
    }
    if reports.iter().any(|report| !report.passed) {
        return HyperionRegressionRouteDecision {
            seed_reports: reports,
            reason: "at least one CV7 seed failed a routing guard".into(),
            ..HyperionRegressionRouteDecision::default()
        };
    }
    let median =
        (reports[0].relative_rmse_improvement + reports[1].relative_rmse_improvement) / 2.0;
    if median < thresholds.min_median_relative_rmse_improvement {
        return HyperionRegressionRouteDecision {
            median_relative_rmse_improvement: median,
            seed_reports: reports,
            reason: "median relative RMSE improvement across seeds is below 1%".into(),
            ..HyperionRegressionRouteDecision::default()
        };
    }
    HyperionRegressionRouteDecision {
        route: HyperionRegressionRoute::RawPlusCell,
        median_relative_rmse_improvement: median,
        seed_reports: reports,
        passed: true,
        reason: "raw_plus_cell passed both train-only CV7 seed banks".into(),
    }
}

#[cfg(test)]
mod tests {
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
        let tie =
            evaluate_regression_seed(303, &y, &baseline, &baseline, &folds(y.len()), &thresholds);
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
        let report =
            evaluate_regression_seed(303, &y, &baseline, &candidate, &bad_folds, &thresholds);
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
}
