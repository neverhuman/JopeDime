//! The one evaluator: parse a program, fit it on a cut's training rows, predict the held-out
//! rows, score. Panics and errors become rejections with a reason; non-finite predictions or
//! scores are rejections too. Nothing here reads a row it should not.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use feat_math::{r2, rmse, GenomeIr, Rows};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cuts::Cut;

/// One program scored on one cut.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Outcome {
    /// `ok` or `rejected`.
    pub status: String,
    /// Why it was rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Held-out RMSE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rmse: Option<f64>,
    /// Held-out R².
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r2: Option<f64>,
    /// Wall seconds in `fit`.
    pub fit_seconds: f64,
    /// Wall seconds in `predict`.
    pub predict_seconds: f64,
    /// Held-out rows scored.
    pub hold_rows: usize,
    /// Held-out predictions, in `cut.hold` order (not serialised: reports carry scores, not rows).
    #[serde(skip)]
    pub predictions: Vec<f64>,
}

impl Outcome {
    fn rejected(reason: String, fit_seconds: f64, predict_seconds: f64, hold_rows: usize) -> Self {
        Self {
            status: "rejected".to_owned(),
            reason: Some(reason),
            rmse: None,
            r2: None,
            fit_seconds,
            predict_seconds,
            hold_rows,
            predictions: Vec::new(),
        }
    }

    /// Whether the program produced a score.
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }
}

/// Rows and targets at `indices`.
pub fn take(x: &Rows, y: &[f64], indices: &[usize]) -> (Rows, Vec<f64>) {
    let xs = indices.iter().map(|&i| x[i].clone()).collect();
    let ys = indices.iter().map(|&i| y[i]).collect();
    (xs, ys)
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_else(|| "non-string panic payload".to_owned())
}

/// Fit `dsl` on `cut.train`, predict `cut.hold`, score.
pub fn evaluate(dsl: &str, x: &Rows, y: &[f64], cut: &Cut) -> Outcome {
    let hold_rows = cut.hold.len();
    let genome = match GenomeIr::parse(dsl) {
        Ok(genome) => genome,
        Err(error) => return Outcome::rejected(format!("parse: {error}"), 0.0, 0.0, hold_rows),
    };
    let (x_train, y_train) = take(x, y, &cut.train);
    let (x_hold, y_hold) = take(x, y, &cut.hold);

    let started = Instant::now();
    let model = match catch_unwind(AssertUnwindSafe(|| {
        feat_math::fit(&genome, &x_train, &y_train)
    })) {
        Ok(Ok(model)) => model,
        Ok(Err(error)) => {
            return Outcome::rejected(
                format!("fit: {error}"),
                started.elapsed().as_secs_f64(),
                0.0,
                hold_rows,
            )
        }
        Err(payic) => {
            return Outcome::rejected(
                format!("fit panicked: {}", panic_text(payic)),
                started.elapsed().as_secs_f64(),
                0.0,
                hold_rows,
            )
        }
    };
    let fit_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let predictions = match catch_unwind(AssertUnwindSafe(|| model.predict(&x_hold))) {
        Ok(predictions) => predictions,
        Err(payload) => {
            return Outcome::rejected(
                format!("predict panicked: {}", panic_text(payload)),
                fit_seconds,
                started.elapsed().as_secs_f64(),
                hold_rows,
            )
        }
    };
    let predict_seconds = started.elapsed().as_secs_f64();
    if predictions.len() != hold_rows {
        return Outcome::rejected(
            format!(
                "predict returned {} values for {hold_rows} rows",
                predictions.len()
            ),
            fit_seconds,
            predict_seconds,
            hold_rows,
        );
    }
    if predictions.iter().any(|p| !p.is_finite()) {
        return Outcome::rejected(
            "non-finite prediction".to_owned(),
            fit_seconds,
            predict_seconds,
            hold_rows,
        );
    }
    let (rmse_value, r2_value) = (rmse(&y_hold, &predictions), r2(&y_hold, &predictions));
    if !rmse_value.is_finite() || !r2_value.is_finite() {
        return Outcome::rejected(
            "non-finite score".to_owned(),
            fit_seconds,
            predict_seconds,
            hold_rows,
        );
    }
    Outcome {
        status: "ok".to_owned(),
        reason: None,
        rmse: Some(rmse_value),
        r2: Some(r2_value),
        fit_seconds,
        predict_seconds,
        hold_rows,
        predictions,
    }
}

/// The variance-penalised fitness the gate, the league and the exact-CV campaign rank by:
/// `max(oof_r2, -1) - 0.25 * min(population_sd(fold_r2), 1)`, over finite folds only. Copied
/// from `crates/jope-campaigns/src/commands/exact_cv.rs::robust_fitness` so the benchmark ranks
/// by the same number the campaign does, without depending on the campaign crate.
pub fn robust_fitness(oof_r2: f64, fold_r2: &[f64]) -> f64 {
    let finite: Vec<f64> = fold_r2.iter().copied().filter(|v| v.is_finite()).collect();
    let population_sd = if finite.len() < 2 {
        0.0
    } else {
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64).sqrt()
    };
    oof_r2.max(-1.0) - 0.25 * population_sd.min(1.0)
}

/// One fold of a deep evaluation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FoldResult {
    /// Held-out fold.
    pub fold: usize,
    /// [`Cut::cut_id`].
    pub cut_id: String,
    /// The score.
    pub outcome: Outcome,
}

/// A program scored on every cut, pooled out of fold.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Deep {
    /// `ok` when every fold scored, else `rejected`.
    pub status: String,
    /// Per-fold outcomes, fold order.
    pub folds: Vec<FoldResult>,
    /// Per-fold R² (ok folds only, fold order).
    pub fold_r2: Vec<f64>,
    /// Pooled out-of-fold R² over every row, in row order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oof_r2: Option<f64>,
    /// Pooled out-of-fold RMSE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oof_rmse: Option<f64>,
    /// Population SD of `fold_r2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fold_sd: Option<f64>,
    /// [`robust_fitness`] of the above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust_fitness: Option<f64>,
    /// Wall seconds summed over folds (fit + predict).
    pub seconds: f64,
}

/// Score `dsl` on every cut; folds run in parallel.
pub fn evaluate_deep(dsl: &str, x: &Rows, y: &[f64], cuts: &[Cut]) -> Deep {
    let outcomes: Vec<Outcome> = cuts
        .par_iter()
        .map(|cut| evaluate(dsl, x, y, cut))
        .collect();
    let seconds = outcomes
        .iter()
        .map(|o| o.fit_seconds + o.predict_seconds)
        .sum();
    let mut oof = vec![f64::NAN; y.len()];
    let mut fold_r2 = Vec::new();
    let mut all_ok = true;
    let mut folds = Vec::with_capacity(cuts.len());
    for (cut, outcome) in cuts.iter().zip(outcomes) {
        if outcome.is_ok() {
            for (i, &row) in cut.hold.iter().enumerate() {
                oof[row] = outcome.predictions[i];
            }
            fold_r2.push(outcome.r2.expect("ok outcome has r2"));
        } else {
            all_ok = false;
        }
        folds.push(FoldResult {
            fold: cut.fold,
            cut_id: cut.cut_id.clone(),
            outcome: Outcome {
                predictions: Vec::new(),
                ..outcome
            },
        });
    }
    if !all_ok || oof.iter().any(|v| !v.is_finite()) {
        return Deep {
            status: "rejected".to_owned(),
            folds,
            fold_r2,
            oof_r2: None,
            oof_rmse: None,
            fold_sd: None,
            robust_fitness: None,
            seconds,
        };
    }
    let oof_r2 = r2(y, &oof);
    let oof_rmse = rmse(y, &oof);
    let mean = fold_r2.iter().sum::<f64>() / fold_r2.len() as f64;
    let fold_sd =
        (fold_r2.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / fold_r2.len() as f64).sqrt();
    Deep {
        status: "ok".to_owned(),
        folds,
        robust_fitness: Some(robust_fitness(oof_r2, &fold_r2)),
        fold_r2,
        oof_r2: Some(oof_r2),
        oof_rmse: Some(oof_rmse),
        fold_sd: Some(fold_sd),
        seconds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuts::{build, FoldPlan};

    fn toy() -> (Rows, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = feat_math::Pcg32::new(3, 7);
        for _ in 0..120 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            x.push(vec![a, b]);
            y.push(2.0 * a - b + 0.1 * (a * b));
        }
        (x, y)
    }

    #[test]
    fn ridge_scores_and_garbage_is_rejected() {
        let (x, y) = toy();
        let cuts = build(&FoldPlan::new("toy", 1, 4), "toy", "s", x.len()).unwrap();
        let ok = evaluate("(pipe X (ridge))", &x, &y, &cuts[0]);
        assert!(ok.is_ok(), "{ok:?}");
        assert!(ok.r2.unwrap() > 0.9);
        assert_eq!(ok.predictions.len(), cuts[0].hold.len());
        let bad = evaluate("(pipe X (no_such_head_ever))", &x, &y, &cuts[0]);
        assert_eq!(bad.status, "rejected");
        assert!(bad.reason.as_deref().unwrap_or("").contains("parse") || bad.reason.is_some());

        let deep = evaluate_deep("(pipe X (ridge))", &x, &y, &cuts);
        assert_eq!(deep.status, "ok");
        assert_eq!(deep.folds.len(), 4);
        assert!(deep.oof_r2.unwrap() > 0.9);
        assert!(deep.folds.iter().all(|f| f.outcome.predictions.is_empty()));
        let expected = robust_fitness(deep.oof_r2.unwrap(), &deep.fold_r2);
        assert_eq!(deep.robust_fitness, Some(expected));
    }

    #[test]
    fn robust_fitness_matches_the_campaign_formula() {
        assert_eq!(robust_fitness(0.5, &[0.5, 0.5]), 0.5);
        assert_eq!(robust_fitness(-3.0, &[]), -1.0);
        let penalised = robust_fitness(0.5, &[0.1, 0.9]);
        assert!((penalised - (0.5 - 0.25 * 0.4)).abs() < 1e-12);
        assert_eq!(
            robust_fitness(0.5, &[f64::NAN, 0.5]),
            0.5,
            "non-finite folds ignored"
        );
        assert!(
            (robust_fitness(0.5, &[-5.0, 5.0]) - 0.25).abs() < 1e-12,
            "sd capped at 1"
        );
    }
}
