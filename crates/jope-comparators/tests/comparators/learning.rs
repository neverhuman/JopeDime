//! Every model must clearly beat the training-mean baseline on held-out rows it never saw.

use crate::fixture::{linear, mean_baseline_rmse, nonlinear, Table};
use jope_comparators::{
    caruana_weights, mae, r_squared, rmse, ComparatorResult, Deadline, EarlyStopping,
    FeatureMatrix, ForestModel, ForestParams, HistGbmModel, HistGbmParams, RidgeModel, RidgeParams,
};

const SEED: u64 = 20_261_110;

fn split(
    table: &Table,
    train_rows: usize,
) -> (FeatureMatrix<'_>, &[f64], FeatureMatrix<'_>, &[f64]) {
    let cut = train_rows * table.cols;
    let train = FeatureMatrix::new(&table.values[..cut], train_rows, table.cols)
        .expect("valid train matrix");
    let hold = FeatureMatrix::new(&table.values[cut..], table.rows - train_rows, table.cols)
        .expect("valid holdout matrix");
    (
        train,
        &table.targets[..train_rows],
        hold,
        &table.targets[train_rows..],
    )
}

/// Fit with `fit`, score on the holdout and return (model RMSE, mean-baseline RMSE).
fn holdout_scores(
    table: &Table,
    fit_predict: impl Fn(FeatureMatrix<'_>, &[f64], FeatureMatrix<'_>) -> ComparatorResult<Vec<f64>>,
) -> (f64, f64) {
    let (train, train_y, hold, hold_y) = split(table, 400);
    let predictions = fit_predict(train, train_y, hold).expect("fit and predict");
    let score = rmse(hold_y, &predictions).expect("score");
    (score, mean_baseline_rmse(train_y, hold_y))
}

#[test]
fn random_forest_beats_the_mean_baseline() {
    let (score, baseline) = holdout_scores(&nonlinear(600, 1), |train, y, hold| {
        ForestModel::fit(
            train,
            y,
            &ForestParams::random_forest(60),
            SEED,
            &Deadline::unbounded(),
        )?
        .predict(hold)
    });
    assert!(score < 0.45 * baseline, "rf {score} vs baseline {baseline}");
}

#[test]
fn extra_trees_beat_the_mean_baseline() {
    let (score, baseline) = holdout_scores(&nonlinear(600, 2), |train, y, hold| {
        ForestModel::fit(
            train,
            y,
            &ForestParams::extra_trees(60),
            SEED,
            &Deadline::unbounded(),
        )?
        .predict(hold)
    });
    assert!(score < 0.45 * baseline, "et {score} vs baseline {baseline}");
}

#[test]
fn histogram_boosting_beats_the_mean_baseline() {
    let (score, baseline) = holdout_scores(&nonlinear(600, 3), |train, y, hold| {
        let params = HistGbmParams::scikit_learn_style(200);
        HistGbmModel::fit(train, y, &params, SEED, &Deadline::unbounded())?.predict(hold)
    });
    assert!(
        score < 0.35 * baseline,
        "hgb {score} vs baseline {baseline}"
    );
}

#[test]
fn early_stopping_keeps_fewer_rounds_and_still_learns() {
    let table = nonlinear(600, 4);
    let (train, y, hold, hold_y) = split(&table, 400);
    let params = HistGbmParams {
        rounds: 2_000,
        learning_rate: 0.3,
        row_subsample: 0.8,
        early_stopping: Some(EarlyStopping {
            validation_fraction: 0.2,
            patience: 10,
        }),
        ..HistGbmParams::scikit_learn_style(2_000)
    };
    let model = HistGbmModel::fit(train, y, &params, SEED, &Deadline::unbounded()).expect("fit");
    assert!(model.rounds_kept() < 2_000);
    assert!(model.rounds_kept() >= 1);
    let predictions = model.predict(hold).expect("predict");
    let score = rmse(hold_y, &predictions).expect("score");
    assert!(score < 0.5 * mean_baseline_rmse(y, hold_y));
}

#[test]
fn ridge_recovers_a_linear_target() {
    let (score, baseline) = holdout_scores(&linear(600, 5), |train, y, hold| {
        RidgeModel::fit(train, y, &RidgeParams { alpha: 1e-6 })?.predict(hold)
    });
    assert!(
        score < 0.05 * baseline,
        "ridge {score} vs baseline {baseline}"
    );
}

#[test]
fn scores_agree_with_hand_computed_values() {
    let targets = [1.0, 2.0, 3.0, 4.0];
    let predictions = [1.0, 2.0, 3.0, 6.0];
    assert_eq!(rmse(&targets, &predictions).expect("rmse"), 1.0);
    assert_eq!(mae(&targets, &predictions).expect("mae"), 0.5);
    assert_eq!(
        r_squared(&targets, &predictions).expect("r2"),
        1.0 - 4.0 / 5.0
    );
    assert_eq!(
        r_squared(&[2.0, 2.0], &[2.0, 2.0]).expect("perfect constant"),
        1.0
    );
    assert_eq!(
        r_squared(&[2.0, 2.0], &[2.0, 3.0]).expect("imperfect constant"),
        0.0
    );
}

#[test]
fn caruana_selects_the_better_member_and_normalizes_weights() {
    let targets = vec![1.0, 2.0, 3.0, 4.0];
    let good = vec![1.1, 2.0, 2.9, 4.0];
    let bad = vec![4.0, 1.0, 0.0, 9.0];
    let weights = caruana_weights(&[bad, good], &targets, 10).expect("weights");
    assert_eq!(weights.len(), 2);
    assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(weights[1] > weights[0]);
}

#[test]
fn caruana_blends_complementary_members() {
    let targets = vec![0.0, 0.0, 0.0, 0.0];
    let high = vec![1.0, 1.0, 1.0, 1.0];
    let low = vec![-1.0, -1.0, -1.0, -1.0];
    let weights = caruana_weights(&[high, low], &targets, 4).expect("weights");
    assert_eq!(weights, vec![0.5, 0.5]);
}
