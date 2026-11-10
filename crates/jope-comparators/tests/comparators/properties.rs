//! Determinism, row locality and missing-value properties, checked over generated inputs.

use crate::fixture::nonlinear;
use jope_comparators::{
    Deadline, FeatureMatrix, ForestModel, ForestParams, HistGbmModel, HistGbmParams, MedianImputer,
    RidgeModel, RidgeParams,
};
use proptest::prelude::*;

/// `Fitted` is an enum for the model-under-test contract in this test target: one fitted model
/// of each family, so every property runs against all of them.
enum Fitted {
    Forest(ForestModel),
    Boosting(HistGbmModel),
    Ridge(RidgeModel),
}

impl Fitted {
    fn predict(&self, features: FeatureMatrix<'_>) -> Vec<f64> {
        match self {
            Self::Forest(model) => model.predict(features),
            Self::Boosting(model) => model.predict(features),
            Self::Ridge(model) => model.predict(features),
        }
        .expect("prediction succeeds")
    }
}

fn fit_all(values: &[f64], targets: &[f64], rows: usize, cols: usize, seed: u64) -> Vec<Fitted> {
    let features = FeatureMatrix::new(values, rows, cols).expect("valid matrix");
    let no_limit = Deadline::no_limit();
    let boosting = HistGbmParams {
        min_samples_leaf: 2,
        ..HistGbmParams::scikit_learn_style(15)
    };
    vec![
        Fitted::Forest(
            ForestModel::fit(
                features,
                targets,
                &ForestParams::random_forest(8),
                seed,
                &no_limit,
            )
            .expect("rf fit"),
        ),
        Fitted::Forest(
            ForestModel::fit(
                features,
                targets,
                &ForestParams::extra_trees(8),
                seed,
                &no_limit,
            )
            .expect("et fit"),
        ),
        Fitted::Boosting(
            HistGbmModel::fit(features, targets, &boosting, seed, &no_limit).expect("gbm fit"),
        ),
        Fitted::Ridge(
            RidgeModel::fit(features, targets, &RidgeParams { alpha: 1.0 }).expect("ridge fit"),
        ),
    ]
}

/// Generated tables: 6–40 rows of 1–4 columns with roughly one value in six missing.
fn table() -> impl Strategy<Value = (Vec<f64>, Vec<f64>, usize, usize)> {
    (6_usize..40, 1_usize..5).prop_flat_map(|(rows, cols)| {
        let cell = prop_oneof![5 => -100.0..100.0_f64, 1 => Just(f64::NAN)];
        (
            proptest::collection::vec(cell, rows * cols),
            proptest::collection::vec(-50.0..50.0_f64, rows),
            Just(rows),
            Just(cols),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn predictions_are_finite_and_reproducible((values, targets, rows, cols) in table(), seed in any::<u64>()) {
        let first = fit_all(&values, &targets, rows, cols, seed);
        let second = fit_all(&values, &targets, rows, cols, seed);
        let features = FeatureMatrix::new(&values, rows, cols).expect("valid matrix");
        for (a, b) in first.iter().zip(&second) {
            let (pa, pb) = (a.predict(features), b.predict(features));
            prop_assert!(pa.iter().all(|value| value.is_finite()));
            prop_assert_eq!(pa.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            pb.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn prediction_is_row_local((values, targets, rows, cols) in table()) {
        let models = fit_all(&values, &targets, rows, cols, 17);
        let batch = FeatureMatrix::new(&values, rows, cols).expect("valid matrix");
        let reversed: Vec<f64> = (0..rows).rev().flat_map(|row| batch.row(row).to_vec()).collect();
        let reversed = FeatureMatrix::new(&reversed, rows, cols).expect("valid matrix");
        for model in &models {
            let whole = model.predict(batch);
            let mut backwards = model.predict(reversed);
            backwards.reverse();
            prop_assert_eq!(&whole, &backwards);
            for (row, expected) in whole.iter().enumerate() {
                let single = FeatureMatrix::new(batch.row(row), 1, cols).expect("valid row");
                prop_assert_eq!(model.predict(single)[0].to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn constant_targets_are_predicted_exactly((values, _targets, rows, cols) in table(), level in -10.0..10.0_f64) {
        let targets = vec![level; rows];
        let features = FeatureMatrix::new(&values, rows, cols).expect("valid matrix");
        for model in fit_all(&values, &targets, rows, cols, 3) {
            for prediction in model.predict(features) {
                prop_assert!((prediction - level).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn imputation_uses_training_medians_only() {
    let train = [1.0, f64::NAN, 3.0, 10.0, 5.0, 20.0];
    let imputer = MedianImputer::fit(FeatureMatrix::new(&train, 3, 2).expect("valid matrix"));
    assert_eq!(imputer.medians(), &[3.0, 15.0]);
    // A holdout with very different values is imputed with the training medians, unchanged.
    let holdout = [f64::NAN, 1_000.0, -1_000.0, f64::NAN];
    let filled = imputer
        .transform(FeatureMatrix::new(&holdout, 2, 2).expect("valid matrix"))
        .expect("transform");
    assert_eq!(filled, vec![3.0, 1_000.0, -1_000.0, 15.0]);
    let empty = MedianImputer::fit(FeatureMatrix::new(&[f64::NAN, f64::NAN], 2, 1).expect("valid"));
    assert_eq!(empty.medians(), &[0.0]);
}

#[test]
fn a_different_seed_gives_a_different_forest() {
    let table = nonlinear(200, 9);
    let features = FeatureMatrix::new(&table.values, table.rows, table.cols).expect("valid matrix");
    let fit = |seed| {
        ForestModel::fit(
            features,
            &table.targets,
            &ForestParams::random_forest(5),
            seed,
            &Deadline::no_limit(),
        )
        .expect("fit")
        .predict(features)
        .expect("predict")
    };
    assert_eq!(fit(1), fit(1));
    assert_ne!(fit(1), fit(2));
}

#[test]
fn model_sizes_grow_with_capacity() {
    let table = nonlinear(200, 10);
    let features = FeatureMatrix::new(&table.values, table.rows, table.cols).expect("valid matrix");
    let no_limit = Deadline::no_limit();
    let small = ForestModel::fit(
        features,
        &table.targets,
        &ForestParams::random_forest(2),
        1,
        &no_limit,
    )
    .expect("fit");
    let large = ForestModel::fit(
        features,
        &table.targets,
        &ForestParams::random_forest(20),
        1,
        &no_limit,
    )
    .expect("fit");
    assert_eq!(small.tree_count(), 2);
    assert!(large.model_bytes() > small.model_bytes());
    let ridge =
        RidgeModel::fit(features, &table.targets, &RidgeParams { alpha: 1.0 }).expect("fit");
    assert!(ridge.model_bytes() < small.model_bytes());
    let boosting = HistGbmModel::fit(
        features,
        &table.targets,
        &HistGbmParams::scikit_learn_style(5),
        1,
        &no_limit,
    )
    .expect("fit");
    assert_eq!(boosting.rounds_kept(), 5);
    assert!(boosting.model_bytes() > 0);
}
