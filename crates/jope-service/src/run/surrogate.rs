//! Frozen r782 full-FIT two-fold recipe, inside the existing isolated worker.
//! Version 2 is the private opaque fold_spec selector; WRK framing and request/
//! response fields are unchanged. Version 1 keeps the native heuristic evaluator.
use crate::{
    dataset::jmat::Matrix,
    discover::r#loop::{numeric_rows, NativeModel},
    objective::regression::RegressionObjective,
    worker::{Request, Scores, WorkerFailure},
};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FoldSpec {
    version: u32,
    matrix_sha256: String,
    target: String,
    features: Vec<String>,
    folds: Vec<Fold>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fold {
    repeat: u8,
    train: Vec<usize>,
    validation: Vec<usize>,
}

pub(crate) fn native<B: AsRef<[u8]> + Sync>(
    request: &Request,
    matrix: &Matrix<B>,
) -> std::result::Result<Scores, WorkerFailure> {
    jope_runtime::budget::Budget::new(None, 1)
        .install(|| evaluate(request, matrix))
        .map_err(|_| WorkerFailure::Unavailable)?
        .map_err(|_| WorkerFailure::InvalidRequest)
}

fn evaluate<B: AsRef<[u8]>>(request: &Request, matrix: &Matrix<B>) -> Result<Scores> {
    let spec: FoldSpec = serde_json::from_value(request.fold_spec.clone())?;
    ensure!(
        spec.version == 2
            && spec.folds.len() == 2
            && matrix.rows() <= 20_000
            && jope_runtime::sha256_hex(matrix.bytes()) == spec.matrix_sha256,
        "invalid admitted surrogate identity"
    );
    let objective = RegressionObjective::parse("rmse")?;
    let user_objective = RegressionObjective::parse(
        request
            .objective
            .as_str()
            .context("objective must be a specification string")?,
    )?;
    let mut logs = Vec::new();
    let mut losses = Vec::new();
    let mut user_losses = Vec::new();
    let mut user_values = Vec::new();
    let mut predicted = Sha256::new();
    let mut validation_seen = BTreeSet::new();
    let mut wins = 0;
    for fold in spec.folds {
        let train: BTreeSet<_> = fold.train.iter().copied().collect();
        ensure!(
            fold.repeat == 0
                && train.len() >= 2
                && train.len() == fold.train.len()
                && fold.validation.len() >= 2
                && fold
                    .validation
                    .iter()
                    .all(|r| !train.contains(r) && validation_seen.insert(*r))
                && train.len() + fold.validation.len() == matrix.rows(),
            "surrogate requires two complementary full-FIT folds"
        );
        let (x, y) = numeric_rows(matrix, &spec.target, &spec.features, &fold.train)?;
        let fitted = NativeModel::fit(&spec.features, &request.dsl, &x, &y)?;
        let mean = y.iter().map(|v| v / y.len() as f64).sum::<f64>();
        let (vx, vy) = numeric_rows(matrix, &spec.target, &spec.features, &fold.validation)?;
        let prediction = fitted.predict(&vx)?;
        let rmse = objective.evaluate(&vy, &prediction)?.loss;
        let baseline = objective.evaluate(&vy, &vec![mean; vy.len()])?.loss;
        ensure!(
            baseline > 1e-12 && baseline.is_finite() && rmse.is_finite(),
            "undefined trained surrogate baseline"
        );
        let ratio = (rmse / baseline).max(1e-12);
        logs.push(ratio.ln());
        wins += usize::from(ratio <= 1.0);
        losses.push(rmse);
        let user_score = user_objective.evaluate(&vy, &prediction)?;
        user_losses.push(user_score.loss);
        user_values.push(user_score.value);
        for (y, p) in vy.iter().zip(prediction) {
            predicted.update(y.to_le_bytes());
            predicted.update(p.to_le_bytes());
        }
    }
    ensure!(
        validation_seen.len() == matrix.rows(),
        "incomplete surrogate FIT coverage"
    );
    let (fitness, _) = fitness(&request.dsl, logs[0], logs[1], wins)?;
    Ok(Scores {
        values: vec![
            fitness,
            0.5 * (user_losses[0] + user_losses[1]),
            0.5 * (user_values[0] + user_values[1]),
        ],
        fold_scores: losses,
        predictions_digest: format!("{:x}", predicted.finalize()),
    })
}

pub(crate) fn fitness(dsl: &str, l0: f64, l1: f64, wins: usize) -> Result<(f64, bool)> {
    ensure!(
        l0.is_finite() && l1.is_finite() && wins <= 2 && dsl.len() < 200_000,
        "invalid trained surrogate summary"
    );
    let mean = 0.5 * (l0 + l1);
    let tail = l0.max(l1);
    let sd = 0.5 * (l0 - l1).abs();
    let gate = wins == 2 && mean.exp() <= 0.90;
    let value = if gate {
        0.8 * mean
            + 0.1 * tail
            + 0.1 * sd
            + 0.0005 * dsl.bytes().filter(|b| *b == b'(').count() as f64
    } else {
        100.0 + mean.max(0.0) + tail.max(0.0) + sd.max(0.0)
    };
    ensure!(
        value.is_finite() && (value < 100.0) == gate,
        "surrogate gate ranges overlap"
    );
    Ok((value, gate))
}
