//! Every invalid input or parameter is refused with its specific typed error, before any work.

use jope_comparators::{
    caruana_weights, rmse, ComparatorError, Deadline, EarlyStopping, FeatureMatrix, ForestModel,
    ForestParams, HistGbmModel, HistGbmParams, RidgeModel, RidgeParams, COMPARATORS_DOCS_URL,
    MAX_RIDGE_FEATURES,
};
use std::time::Duration;

fn matrix(values: &[f64], rows: usize, cols: usize) -> FeatureMatrix<'_> {
    FeatureMatrix::new(values, rows, cols).expect("valid matrix")
}

fn forest(values: &[f64], targets: &[f64], rows: usize, params: &ForestParams) -> ComparatorError {
    ForestModel::fit(
        matrix(values, rows, 1),
        targets,
        params,
        0,
        &Deadline::no_limit(),
    )
    .expect_err("fit must be refused")
}

fn boosting(params: &HistGbmParams) -> ComparatorError {
    HistGbmModel::fit(
        matrix(&[1.0, 2.0], 2, 1),
        &[1.0, 2.0],
        params,
        0,
        &Deadline::no_limit(),
    )
    .expect_err("fit must be refused")
}

#[test]
fn malformed_matrices_are_refused() {
    assert_eq!(
        FeatureMatrix::new(&[1.0, 2.0, 3.0], 2, 2).expect_err("short buffer"),
        ComparatorError::ShapeMismatch {
            expected: 4,
            actual: 3
        }
    );
    assert_eq!(
        FeatureMatrix::new(&[], 3, 0).expect_err("no columns"),
        ComparatorError::NoFeatures
    );
    assert_eq!(
        FeatureMatrix::new(&[], usize::MAX, 2).expect_err("overflow"),
        ComparatorError::ShapeOverflow
    );
    assert_eq!(
        FeatureMatrix::new(&[1.0, f64::INFINITY], 1, 2).expect_err("infinite"),
        ComparatorError::InfiniteFeature { row: 0, column: 1 }
    );
    assert!(
        FeatureMatrix::new(&[f64::NAN], 1, 1).is_ok(),
        "NaN marks a missing value"
    );
}

#[test]
fn invalid_training_sets_are_refused() {
    let params = ForestParams::random_forest(2);
    assert_eq!(forest(&[], &[], 0, &params), ComparatorError::EmptyFit);
    assert_eq!(
        forest(&[1.0, 2.0], &[1.0], 2, &params),
        ComparatorError::ShapeMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(
        forest(&[1.0, 2.0], &[1.0, f64::NAN], 2, &params),
        ComparatorError::NonFiniteTarget { row: 1 }
    );
}

#[test]
fn invalid_forest_parameters_are_named() {
    let cases = [
        (
            ForestParams {
                trees: 0,
                ..ForestParams::random_forest(1)
            },
            "trees",
        ),
        (
            ForestParams {
                max_features: 0.0,
                ..ForestParams::random_forest(1)
            },
            "max_features",
        ),
        (
            ForestParams {
                min_samples_leaf: 0,
                ..ForestParams::random_forest(1)
            },
            "min_samples_leaf",
        ),
        (
            ForestParams {
                max_bins: 1,
                ..ForestParams::random_forest(1)
            },
            "max_bins",
        ),
    ];
    for (params, name) in cases {
        assert_eq!(
            forest(&[1.0, 2.0], &[1.0, 2.0], 2, &params),
            ComparatorError::InvalidParameter { name }
        );
    }
}

#[test]
fn invalid_boosting_parameters_are_named() {
    let base = HistGbmParams::scikit_learn_style(3);
    let cases = [
        (
            HistGbmParams {
                rounds: 0,
                ..base.clone()
            },
            "rounds",
        ),
        (
            HistGbmParams {
                learning_rate: 0.0,
                ..base.clone()
            },
            "learning_rate",
        ),
        (
            HistGbmParams {
                max_leaves: Some(1),
                ..base.clone()
            },
            "max_leaves",
        ),
        (
            HistGbmParams {
                l2: -1.0,
                ..base.clone()
            },
            "l2",
        ),
        (
            HistGbmParams {
                row_subsample: 1.5,
                ..base.clone()
            },
            "row_subsample",
        ),
        (
            HistGbmParams {
                early_stopping: Some(EarlyStopping {
                    validation_fraction: 0.9,
                    patience: 1,
                }),
                ..base.clone()
            },
            "early_stopping",
        ),
    ];
    for (params, name) in cases {
        assert_eq!(
            boosting(&params),
            ComparatorError::InvalidParameter { name }
        );
    }
}

#[test]
fn ridge_refuses_bad_penalties_singular_systems_and_oversized_inputs() {
    let features = matrix(&[1.0, 2.0], 2, 1);
    assert_eq!(
        RidgeModel::fit(features, &[1.0, 2.0], &RidgeParams { alpha: f64::NAN }).expect_err("nan"),
        ComparatorError::InvalidParameter { name: "alpha" }
    );
    // Two identical columns with no penalty are collinear.
    let collinear = matrix(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 3, 2);
    assert_eq!(
        RidgeModel::fit(collinear, &[1.0, 2.0, 3.0], &RidgeParams { alpha: 0.0 })
            .expect_err("singular"),
        ComparatorError::SingularSystem
    );
    let wide = vec![0.5; MAX_RIDGE_FEATURES + 1];
    assert_eq!(
        RidgeModel::fit(
            matrix(&wide, 1, MAX_RIDGE_FEATURES + 1),
            &[1.0],
            &RidgeParams { alpha: 1.0 }
        )
        .expect_err("too wide"),
        ComparatorError::TooManyFeatures {
            limit: MAX_RIDGE_FEATURES
        }
    );
}

#[test]
fn an_expired_deadline_stops_the_fit() {
    let expired = Deadline::after(Duration::ZERO);
    assert!(expired.expired());
    assert!(!Deadline::no_limit().expired());
    let features = matrix(&[1.0, 2.0, 3.0], 3, 1);
    assert_eq!(
        ForestModel::fit(
            features,
            &[1.0, 2.0, 3.0],
            &ForestParams::random_forest(3),
            0,
            &expired
        )
        .expect_err("deadline"),
        ComparatorError::DeadlineExceeded
    );
    assert_eq!(
        HistGbmModel::fit(
            features,
            &[1.0, 2.0, 3.0],
            &HistGbmParams::scikit_learn_style(3),
            0,
            &expired
        )
        .expect_err("deadline"),
        ComparatorError::DeadlineExceeded
    );
}

#[test]
fn prediction_refuses_a_different_width() {
    let model = ForestModel::fit(
        matrix(&[1.0, 2.0], 2, 1),
        &[1.0, 2.0],
        &ForestParams::random_forest(1),
        0,
        &Deadline::no_limit(),
    )
    .expect("fit");
    assert_eq!(
        model.predict(matrix(&[1.0, 2.0], 1, 2)).expect_err("width"),
        ComparatorError::ShapeMismatch {
            expected: 1,
            actual: 2
        }
    );
    assert_eq!(
        model.predict(matrix(&[], 0, 1)).expect("empty batch"),
        Vec::<f64>::new()
    );
}

#[test]
fn scores_and_ensembles_refuse_bad_inputs() {
    assert_eq!(
        rmse(&[], &[]).expect_err("empty"),
        ComparatorError::EmptyScore
    );
    assert_eq!(
        rmse(&[1.0], &[f64::NAN]).expect_err("nan prediction"),
        ComparatorError::NonFinitePrediction { row: 0 }
    );
    assert_eq!(
        caruana_weights(&[], &[1.0], 3).expect_err("no members"),
        ComparatorError::InvalidParameter { name: "members" }
    );
    assert_eq!(
        caruana_weights(&[vec![1.0]], &[1.0], 0).expect_err("no rounds"),
        ComparatorError::InvalidParameter { name: "rounds" }
    );
    assert_eq!(
        caruana_weights(&[vec![1.0, 2.0]], &[1.0], 1).expect_err("length"),
        ComparatorError::ShapeMismatch {
            expected: 1,
            actual: 2
        }
    );
}

#[test]
fn errors_render_readable_messages() {
    let message = ComparatorError::InvalidParameter { name: "trees" }.to_string();
    assert_eq!(message, "parameter `trees` is out of range");
    assert_eq!(
        ComparatorError::DeadlineExceeded.to_string(),
        "the fit deadline passed"
    );
}

#[test]
fn every_error_carries_actionable_guidance() {
    let errors = [
        ComparatorError::EmptyFit,
        ComparatorError::EmptyScore,
        ComparatorError::NoFeatures,
        ComparatorError::ShapeMismatch {
            expected: 2,
            actual: 1,
        },
        ComparatorError::ShapeOverflow,
        ComparatorError::TooManyRows { limit: 1 },
        ComparatorError::TooManyFeatures { limit: 1 },
        ComparatorError::NonFiniteTarget { row: 0 },
        ComparatorError::InfiniteFeature { row: 0, column: 0 },
        ComparatorError::InvalidParameter { name: "trees" },
        ComparatorError::SingularSystem,
        ComparatorError::DeadlineExceeded,
        ComparatorError::NonFinitePrediction { row: 0 },
    ];
    for error in errors {
        let guidance = error.guidance();
        assert_eq!(guidance.reason, error.to_string());
        assert_eq!(guidance.docs_url, COMPARATORS_DOCS_URL);
        assert!(!guidance.purpose.is_empty());
        assert!(!guidance.common_fixes.is_empty());
        assert!(!guidance.repair_hint.is_empty());
    }
    assert!(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("README.md")
        .is_file());
}
