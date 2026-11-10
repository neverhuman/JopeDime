//! The roster: stable identifiers, honest labels, effort tiers, and every comparator learning,
//! reproducing and honouring its deadline through the single `fit_comparator` entry point.

use crate::fixture::{mean_baseline_rmse, nonlinear};
use jope_comparators::{
    fit_comparator, rmse, ComparatorError, ComparatorId, ComparatorTier, Deadline, FeatureMatrix,
};
use std::collections::HashSet;
use std::time::Duration;

#[test]
fn identifiers_round_trip_and_labels_are_honest() {
    let keys: HashSet<&str> = ComparatorId::ALL.iter().map(|id| id.key()).collect();
    assert_eq!(keys.len(), ComparatorId::ALL.len(), "keys are unique");
    for id in ComparatorId::ALL {
        let descriptor = id.descriptor();
        assert_eq!(descriptor.id, id);
        assert_eq!(ComparatorId::from_key(id.key()), Some(id));
        assert!(
            descriptor.label.contains("Rust"),
            "{} must say it is a Rust analogue or port",
            descriptor.label
        );
        assert!(!descriptor.reference.is_empty() && !descriptor.known_gaps.is_empty());
    }
    assert_eq!(
        ComparatorId::from_key("xgboost"),
        None,
        "only the honest -style keys resolve"
    );
    assert_eq!(ComparatorId::from_key(""), None);
}

#[test]
fn tiers_exist_only_for_medium_high_and_ultra_and_grow_monotonically() {
    assert_eq!(ComparatorTier::from_effort("fast"), None);
    assert_eq!(ComparatorTier::from_effort("turbo"), None);
    for tier in ComparatorTier::ALL {
        assert_eq!(ComparatorTier::from_effort(tier.as_str()), Some(tier));
    }
    let budgets: Vec<_> = ComparatorTier::ALL
        .iter()
        .map(|tier| tier.budget())
        .collect();
    assert_eq!(
        budgets
            .iter()
            .map(|budget| budget.stage_ceiling.as_secs())
            .collect::<Vec<_>>(),
        vec![60, 300, 1_800]
    );
    for pair in budgets.windows(2) {
        assert!(pair[1].boosting_rounds > pair[0].boosting_rounds);
        assert!(pair[1].forest_trees > pair[0].forest_trees);
        assert!(pair[1].search_candidates > pair[0].search_candidates);
        assert!(pair[1].ensemble_rounds > pair[0].ensemble_rounds);
        assert!(pair[1].folds >= pair[0].folds);
    }
}

#[test]
fn every_comparator_learns_and_reproduces_at_the_medium_tier() {
    let table = nonlinear(480, 21);
    let train_rows = 360;
    let cut = train_rows * table.cols;
    let train = FeatureMatrix::new(&table.values[..cut], train_rows, table.cols).expect("train");
    let hold = FeatureMatrix::new(&table.values[cut..], table.rows - train_rows, table.cols)
        .expect("holdout");
    let (train_y, hold_y) = table.targets.split_at(train_rows);
    let baseline = mean_baseline_rmse(train_y, hold_y);
    for id in ComparatorId::ALL {
        let fit = || {
            fit_comparator(
                id,
                ComparatorTier::Medium,
                train,
                train_y,
                5,
                &Deadline::no_limit(),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", id.key()))
        };
        let model = fit();
        assert_eq!(model.id(), id);
        assert!(
            model.model_count() >= 1 && model.model_bytes() > 0,
            "{}",
            id.key()
        );
        let predictions = model.predict(hold).expect("predict");
        let score = rmse(hold_y, &predictions).expect("score");
        assert!(
            score < 0.6 * baseline,
            "{}: {score} vs baseline {baseline}",
            id.key()
        );
        assert_eq!(
            fit().predict(hold).expect("predict"),
            predictions,
            "{} is seeded",
            id.key()
        );
    }
}

#[test]
fn every_comparator_stops_at_an_expired_deadline_and_refuses_bad_input() {
    let table = nonlinear(40, 22);
    let features = FeatureMatrix::new(&table.values, table.rows, table.cols).expect("matrix");
    let expired = Deadline::after(Duration::ZERO);
    let single = FeatureMatrix::new(&table.values[..table.cols], 1, table.cols).expect("one row");
    for id in ComparatorId::ALL {
        assert_eq!(
            fit_comparator(
                id,
                ComparatorTier::High,
                features,
                &table.targets,
                1,
                &expired
            )
            .expect_err("deadline"),
            ComparatorError::DeadlineExceeded,
            "{}",
            id.key()
        );
        assert_eq!(
            fit_comparator(
                id,
                ComparatorTier::Medium,
                single,
                &table.targets[..1],
                1,
                &Deadline::no_limit()
            )
            .expect_err("one row"),
            ComparatorError::EmptyFit,
            "{}",
            id.key()
        );
        assert_eq!(
            fit_comparator(
                id,
                ComparatorTier::Medium,
                features,
                &table.targets[1..],
                1,
                &Deadline::no_limit()
            )
            .expect_err("short targets"),
            ComparatorError::ShapeMismatch {
                expected: 40,
                actual: 39
            },
            "{}",
            id.key()
        );
    }
}
