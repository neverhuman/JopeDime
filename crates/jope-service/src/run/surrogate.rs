//! Full-FIT frozen S1 worker: Q1 uses three folds per ENG T0710.
//! Historical non-Q1 two-fold calls retain their original numerical results.
//! Version 2 is the private opaque fold_spec selector; WRK framing and request/
//! response fields are unchanged. Version 1 keeps the native heuristic evaluator.
use crate::{
    dataset::jmat::Matrix,
    discover::r#loop::numeric_rows,
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
            && matches!(spec.folds.len(), 2 | 3)
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
            "surrogate requires disjoint full-FIT validation partitions"
        );
        let (x, y) = numeric_rows(matrix, &spec.target, &spec.features, &fold.train)?;
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let (vx, vy) = numeric_rows(matrix, &spec.target, &spec.features, &fold.validation)?;
        let prediction = predict_s1(&request.dsl, &x, &y, &vx)?;
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
    let (fitness, _) = fitness(&request.dsl, &logs, wins)?;
    Ok(Scores {
        values: vec![
            fitness,
            user_losses.iter().sum::<f64>() / user_losses.len() as f64,
            user_values.iter().sum::<f64>() / user_values.len() as f64,
        ],
        fold_scores: losses,
        predictions_digest: format!("{:x}", predicted.finalize()),
    })
}

// Frozen s1main::one_fit uses fold-fitted min/max without clipping. Reusing
// generic NativeModel silently clipped validation extrapolation and replaced
// positive target spans smaller than 1e-12, changing the trained surrogate.
fn predict_s1(
    dsl: &str,
    x: &feat_math::Rows,
    y: &[f64],
    validation: &feat_math::Rows,
) -> Result<Vec<f64>> {
    ensure!(
        !x.is_empty()
            && x.len() == y.len()
            && x.iter().chain(validation).flatten().all(|v| v.is_finite()),
        "frozen S1 requires finite training and validation features"
    );
    let mut columns = Vec::with_capacity(x[0].len());
    for j in 0..x[0].len() {
        let lo = x.iter().map(|r| r[j]).fold(f64::INFINITY, f64::min);
        let hi = x.iter().map(|r| r[j]).fold(f64::NEG_INFINITY, f64::max);
        columns.push((lo, if hi - lo > 0.0 { hi - lo } else { 1.0 }));
    }
    let y_lo = y.iter().copied().fold(f64::INFINITY, f64::min);
    let y_hi = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_span = if y_hi - y_lo > 0.0 { y_hi - y_lo } else { 1.0 };
    let scale = |rows: &feat_math::Rows| -> feat_math::Rows {
        rows.iter()
            .map(|row| {
                row.iter()
                    .zip(&columns)
                    .map(|(v, (lo, span))| (v - lo) / span)
                    .collect()
            })
            .collect()
    };
    let scaled_y: Vec<_> = y.iter().map(|v| (v - y_lo) / y_span).collect();
    let fitted = feat_math::fit(&feat_math::GenomeIr::parse(dsl)?, &scale(x), &scaled_y)?;
    let prediction: Vec<_> = fitted
        .predict(&scale(validation))?
        .into_iter()
        .map(|v| v * y_span + y_lo)
        .collect();
    ensure!(
        prediction.iter().all(|v| v.is_finite()),
        "invalid S1 prediction"
    );
    Ok(prediction)
}

pub(crate) fn fitness(dsl: &str, logs: &[f64], wins: usize) -> Result<(f64, bool)> {
    ensure!(
        matches!(logs.len(), 2 | 3)
            && logs.iter().all(|value| value.is_finite())
            && wins <= logs.len()
            && dsl.len() < 200_000,
        "invalid trained surrogate summary"
    );
    // Keep the historical two-fold arithmetic bit-for-bit. Q1 admission
    // requires three folds; its third log ratio affects every summary term.
    let (mean, tail, sd) = if logs.len() == 2 {
        (
            0.5 * (logs[0] + logs[1]),
            logs[0].max(logs[1]),
            0.5 * (logs[0] - logs[1]).abs(),
        )
    } else {
        let mean = logs.iter().sum::<f64>() / 3.0;
        (
            mean,
            logs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            (logs.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / 3.0).sqrt(),
        )
    };
    let gate = wins == logs.len() && mean.exp() <= 0.90;
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
