//! The one evaluator: parse a program, fit it on a cut's training rows, predict the held-out
//! rows, score for the task. Panics and errors become rejections with a reason; non-finite or
//! out-of-range predictions and non-finite scores are rejections too. Nothing here reads a row
//! it should not.
//!
//! Regression programs run on the frozen kernel (`feat_math::fit` / `predict`). Binary and
//! multiclass programs run on the schema-bound v3 table API (`fit_table` / `predict_table`) over
//! a synthetic all-numeric schema, the only public v3 fit surface; numeric cells are rendered
//! with `{v:?}` (shortest round trip) so the encoded feature block equals the numeric matrix bit
//! for bit, exactly as the binary campaign lane does.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use anyhow::{Context, Result};
use feat_math::advanced_v3::{fit_table, ColumnSpec, DataSchema, RawTable, RowSemantics};
use feat_math::metrics::{auc, brier, logloss};
use feat_math::{macro_f1, multiclass_accuracy, r2, rmse, GenomeIr, Rows};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cuts::Cut;
use crate::task::Task;

/// Metric id → value.
pub type Metrics = BTreeMap<String, f64>;

/// One program scored on one cut.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Outcome {
    /// `ok` or `rejected`.
    pub status: String,
    /// Why it was rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The task's primary score on the held-out rows ([`Task::score_metric`]; higher is
    /// better). `None` on a rejection, and for a binary cut whose held-out rows are one class
    /// (AUC is undefined there; `note` says so and the other metrics are still recorded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Every metric on the held-out rows ([`Task::metrics`]; `auc` is absent when undefined).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: Metrics,
    /// Something worth knowing about an `ok` outcome.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Wall seconds in `fit`.
    pub fit_seconds: f64,
    /// Wall seconds in `predict` (for the table API this includes rendering the table).
    pub predict_seconds: f64,
    /// Held-out rows scored.
    pub hold_rows: usize,
    /// Held-out predictions, in `cut.hold` order (not serialised: reports carry scores, not rows).
    #[serde(skip)]
    pub predictions: Vec<f64>,
}

impl Outcome {
    pub(crate) fn rejected(
        reason: String,
        fit_seconds: f64,
        predict_seconds: f64,
        hold_rows: usize,
    ) -> Self {
        Self {
            status: "rejected".to_owned(),
            reason: Some(reason),
            score: None,
            metrics: Metrics::new(),
            note: None,
            fit_seconds,
            predict_seconds,
            hold_rows,
            predictions: Vec::new(),
        }
    }

    /// Whether the program fitted and predicted (its primary score may still be undefined).
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

/// Synthetic all-numeric schema `c0..c{features-1}`, IID rows.
pub fn numeric_schema(features: usize) -> DataSchema {
    DataSchema {
        columns: (0..features)
            .map(|j| ColumnSpec::numeric(format!("c{j}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

/// Shortest-round-trip string table: parses back to the exact f64 cells.
pub fn numeric_table(x: &Rows) -> Result<RawTable> {
    let features = x.first().map(Vec::len).unwrap_or(0);
    let headers = (0..features).map(|j| format!("c{j}")).collect::<Vec<_>>();
    let cells = x
        .iter()
        .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
        .collect();
    RawTable::new(headers, cells)
}

/// A fitted program: predicts a row block.
type Predictor = Box<dyn Fn(&Rows) -> Result<Vec<f64>>>;

fn fit(task: Task, genome: &GenomeIr, x: &Rows, y: &[f64]) -> Result<Predictor> {
    match task.v3_spec() {
        None => {
            let model = feat_math::fit(genome, x, y)?;
            Ok(Box::new(move |rows: &Rows| Ok(model.predict(rows)?)))
        }
        Some(spec) => {
            let features = x.first().map(Vec::len).unwrap_or(0);
            let schema = numeric_schema(features);
            let table = numeric_table(x)?;
            let model = fit_table(genome, &spec, &schema, &table, y)?;
            Ok(Box::new(move |rows: &Rows| {
                model.predict_table(&numeric_table(rows)?)
            }))
        }
    }
}

/// Score held-out predictions for `task`: the primary score (`None` when undefined), every
/// metric, and a note. `Err` names a prediction the task does not allow.
pub fn score(
    task: Task,
    y: &[f64],
    p: &[f64],
) -> std::result::Result<(Option<f64>, Metrics, Option<String>), String> {
    let mut metrics = Metrics::new();
    match task {
        Task::Regression => {
            let (rmse_value, r2_value) = (rmse(y, p), r2(y, p));
            if !rmse_value.is_finite() || !r2_value.is_finite() {
                return Err("non-finite score".to_owned());
            }
            metrics.insert("rmse".to_owned(), rmse_value);
            metrics.insert("r2".to_owned(), r2_value);
            Ok((Some(r2_value), metrics, None))
        }
        Task::Binary => {
            if p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
                return Err("probability outside [0, 1]".to_owned());
            }
            let (ll, br) = (logloss(y, p), brier(y, p));
            if !ll.is_finite() || !br.is_finite() {
                return Err("non-finite score".to_owned());
            }
            metrics.insert("logloss".to_owned(), ll);
            metrics.insert("brier".to_owned(), br);
            let positives = y.iter().filter(|v| **v == 1.0).count();
            if positives == 0 || positives == y.len() {
                return Ok((
                    None,
                    metrics,
                    Some("held-out rows are one class; AUC undefined".to_owned()),
                ));
            }
            let auc_value = auc(y, p);
            if !auc_value.is_finite() {
                return Err("non-finite score".to_owned());
            }
            metrics.insert("auc".to_owned(), auc_value);
            Ok((Some(auc_value), metrics, None))
        }
        Task::Multiclass => {
            if p.iter().any(|v| !v.is_finite() || v.fract() != 0.0) {
                return Err("non-integer class label".to_owned());
            }
            let (f1, acc) = (macro_f1(y, p), multiclass_accuracy(y, p));
            if !f1.is_finite() || !acc.is_finite() {
                return Err("non-finite score".to_owned());
            }
            metrics.insert("macro_f1".to_owned(), f1);
            metrics.insert("accuracy".to_owned(), acc);
            Ok((Some(f1), metrics, None))
        }
    }
}

/// Fit `dsl` on `cut.train`, predict `cut.hold`, score for `task`.
pub fn evaluate(dsl: &str, x: &Rows, y: &[f64], cut: &Cut, task: Task) -> Outcome {
    let hold_rows = cut.hold.len();
    let genome = match catch_unwind(AssertUnwindSafe(|| GenomeIr::parse(dsl))) {
        Ok(Ok(genome)) => genome,
        Ok(Err(error)) => return Outcome::rejected(format!("parse: {error}"), 0.0, 0.0, hold_rows),
        Err(payload) => {
            return Outcome::rejected(
                format!("parse panicked: {}", panic_text(payload)),
                0.0,
                0.0,
                hold_rows,
            )
        }
    };
    let (x_train, y_train) = take(x, y, &cut.train);
    let (x_hold, y_hold) = take(x, y, &cut.hold);

    let started = Instant::now();
    let predictor = match catch_unwind(AssertUnwindSafe(|| fit(task, &genome, &x_train, &y_train)))
    {
        Ok(Ok(predictor)) => predictor,
        Ok(Err(error)) => {
            return Outcome::rejected(
                format!("fit: {error:#}"),
                started.elapsed().as_secs_f64(),
                0.0,
                hold_rows,
            )
        }
        Err(payload) => {
            return Outcome::rejected(
                format!("fit panicked: {}", panic_text(payload)),
                started.elapsed().as_secs_f64(),
                0.0,
                hold_rows,
            )
        }
    };
    let fit_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let predictions = match catch_unwind(AssertUnwindSafe(|| predictor(&x_hold))) {
        Ok(Ok(predictions)) => predictions,
        Ok(Err(error)) => {
            return Outcome::rejected(
                format!("predict: {error:#}"),
                fit_seconds,
                started.elapsed().as_secs_f64(),
                hold_rows,
            )
        }
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
    match score(task, &y_hold, &predictions) {
        Ok((score, metrics, note)) => Outcome {
            status: "ok".to_owned(),
            reason: None,
            score,
            metrics,
            note,
            fit_seconds,
            predict_seconds,
            hold_rows,
            predictions,
        },
        Err(reason) => Outcome::rejected(reason, fit_seconds, predict_seconds, hold_rows),
    }
}

/// The variance-penalised fitness the gate, the league and the exact-CV campaign rank
/// regression by: `max(oof_r2, -1) - 0.25 * min(population_sd(fold_r2), 1)`, over finite folds
/// only. Copied from `crates/jope-campaigns/src/commands/exact_cv.rs::robust_fitness` so the
/// benchmark ranks by the same number the campaign does, without depending on the campaign crate.
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

/// The binary campaign lane's twin of [`robust_fitness`], on the Gini scale (`2·AUC − 1`: the
/// same "0 = chance, 1 = perfect" geometry as R², so the fold-variance penalty carries over at
/// a compatible scale). Folds whose held-out rows were one class have undefined AUC (passed as
/// NaN, or simply omitted) and are excluded from the penalty. Copied from
/// `crates/jope-campaigns/src/commands/exact_cv.rs::robust_fitness_binary`.
pub fn robust_fitness_binary(pooled_oof_auc: f64, fold_auc: &[f64]) -> f64 {
    let gini = 2.0 * pooled_oof_auc.clamp(0.0, 1.0) - 1.0;
    let fold_gini: Vec<f64> = fold_auc
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|auc| 2.0 * auc - 1.0)
        .collect();
    let population_sd = if fold_gini.len() < 2 {
        0.0
    } else {
        let mean = fold_gini.iter().sum::<f64>() / fold_gini.len() as f64;
        (fold_gini
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / fold_gini.len() as f64)
            .sqrt()
    };
    gini - 0.25 * population_sd.min(1.0)
}

/// The same penalty for a score already on a `[0, 1]` scale (multiclass macro-F1):
/// `oof - 0.25 * min(population_sd(folds), 1)`. No campaign lane defines a multiclass fitness,
/// so this is the benchmark's own definition; it is the regression rule without the `−1` floor.
pub fn robust_fitness_bounded(oof: f64, folds: &[f64]) -> f64 {
    let finite: Vec<f64> = folds.iter().copied().filter(|v| v.is_finite()).collect();
    let population_sd = if finite.len() < 2 {
        0.0
    } else {
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64).sqrt()
    };
    oof - 0.25 * population_sd.min(1.0)
}

/// The task's variance-penalised deep fitness: the pooled out-of-fold primary score penalised
/// by the population SD of the per-fold scores.
pub fn robust_fitness_for(task: Task, oof_score: f64, fold_scores: &[f64]) -> f64 {
    match task {
        Task::Regression => robust_fitness(oof_score, fold_scores),
        Task::Binary => robust_fitness_binary(oof_score, fold_scores),
        Task::Multiclass => robust_fitness_bounded(oof_score, fold_scores),
    }
}

/// One fold of a deep evaluation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FoldResult {
    /// Held-out fold.
    pub fold: usize,
    /// [`crate::cuts::Cut::cut_id`].
    pub cut_id: String,
    /// The program scored on this fold (differs per fold only when the search was re-run per
    /// cut; empty for an adapter, and when producing the program failed).
    #[serde(default)]
    pub dsl: String,
    /// The score.
    pub outcome: Outcome,
}

/// A program (or a per-cut program) scored on every cut, pooled out of fold.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Deep {
    /// `ok` when every fold scored and the pooled score is defined, else `rejected`.
    pub status: String,
    /// Why it was rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Per-fold outcomes, fold order.
    pub folds: Vec<FoldResult>,
    /// Per-fold primary scores, fold order, folds where it is defined only.
    pub fold_score: Vec<f64>,
    /// Pooled out-of-fold primary score over every row, in row order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oof_score: Option<f64>,
    /// Every pooled out-of-fold metric ([`Task::metrics`]).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub oof: Metrics,
    /// Population SD of `fold_score`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fold_sd: Option<f64>,
    /// [`robust_fitness_for`] of the above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust_fitness: Option<f64>,
    /// Wall seconds summed over folds (fit + predict).
    pub seconds: f64,
    /// Wall seconds spent producing per-fold programs (0 unless the search was re-run per cut).
    #[serde(default)]
    pub produce_seconds: f64,
}

impl Deep {
    fn rejected(
        reason: String,
        folds: Vec<FoldResult>,
        fold_score: Vec<f64>,
        seconds: f64,
    ) -> Self {
        Self {
            status: "rejected".to_owned(),
            reason: Some(reason),
            folds,
            fold_score,
            oof_score: None,
            oof: Metrics::new(),
            fold_sd: None,
            robust_fitness: None,
            seconds,
            produce_seconds: 0.0,
        }
    }
}

/// Score one program on every cut; folds run in parallel.
pub fn evaluate_deep(dsl: &str, x: &Rows, y: &[f64], cuts: &[Cut], task: Task) -> Deep {
    let programs: Vec<std::result::Result<String, String>> =
        cuts.iter().map(|_| Ok(dsl.to_owned())).collect();
    evaluate_deep_programs(&programs, x, y, cuts, task)
}

/// Score `programs[i]` on `cuts[i]` for every cut and pool out of fold. An `Err` program is a
/// fold whose program could not be produced (a failed per-cut search): that fold is a rejection
/// naming the cause, never a parse attempt on the error text. `programs.len()` must equal
/// `cuts.len()`; a mismatch is reported as a rejected deep score, never a panic.
pub fn evaluate_deep_programs(
    programs: &[std::result::Result<String, String>],
    x: &Rows,
    y: &[f64],
    cuts: &[Cut],
    task: Task,
) -> Deep {
    if programs.len() != cuts.len() {
        return Deep::rejected(
            format!("{} programs for {} cuts", programs.len(), cuts.len()),
            Vec::new(),
            Vec::new(),
            0.0,
        );
    }
    let outcomes: Vec<Outcome> = cuts
        .par_iter()
        .zip(programs.par_iter())
        .map(|(cut, program)| match program {
            Ok(dsl) => evaluate(dsl, x, y, cut, task),
            Err(reason) => Outcome::rejected(format!("search: {reason}"), 0.0, 0.0, cut.hold.len()),
        })
        .collect();
    let labels: Vec<String> = programs
        .iter()
        .map(|p| p.clone().unwrap_or_default())
        .collect();
    pool(cuts, &labels, outcomes, y, task)
}

/// Pool one outcome per cut out of fold: every row's held-out prediction in row order, scored
/// for `task`, penalised by the fold variance. `labels[i]` is recorded as fold `i`'s `dsl`
/// (the program scored there; empty for an adapter or a fold whose program could not be made).
/// Any rejected fold, any uncovered row or an undefined pooled score rejects the whole thing.
pub fn pool(
    cuts: &[Cut],
    labels: &[String],
    outcomes: Vec<Outcome>,
    y: &[f64],
    task: Task,
) -> Deep {
    if labels.len() != cuts.len() || outcomes.len() != cuts.len() {
        return Deep::rejected(
            format!(
                "{} labels and {} outcomes for {} cuts",
                labels.len(),
                outcomes.len(),
                cuts.len()
            ),
            Vec::new(),
            Vec::new(),
            0.0,
        );
    }
    let seconds = outcomes
        .iter()
        .map(|o| o.fit_seconds + o.predict_seconds)
        .sum();
    let mut oof = vec![f64::NAN; y.len()];
    let mut fold_score = Vec::new();
    let mut all_ok = true;
    let mut folds = Vec::with_capacity(cuts.len());
    for ((cut, label), outcome) in cuts.iter().zip(labels).zip(outcomes) {
        if outcome.is_ok() {
            for (i, &row) in cut.hold.iter().enumerate() {
                oof[row] = outcome.predictions[i];
            }
            if let Some(score) = outcome.score {
                fold_score.push(score);
            }
        } else {
            all_ok = false;
        }
        folds.push(FoldResult {
            fold: cut.fold,
            cut_id: cut.cut_id.clone(),
            dsl: label.clone(),
            outcome: Outcome {
                predictions: Vec::new(),
                ..outcome
            },
        });
    }
    if !all_ok {
        return Deep::rejected("a fold was rejected".to_owned(), folds, fold_score, seconds);
    }
    if oof.iter().any(|v| !v.is_finite()) {
        return Deep::rejected(
            "the cuts do not cover every row".to_owned(),
            folds,
            fold_score,
            seconds,
        );
    }
    let (oof_score, oof_metrics) = match score(task, y, &oof) {
        Ok((Some(score), metrics, _)) => (score, metrics),
        Ok((None, _, note)) => {
            return Deep::rejected(
                format!("pooled: {}", note.unwrap_or_default()),
                folds,
                fold_score,
                seconds,
            )
        }
        Err(reason) => {
            return Deep::rejected(format!("pooled: {reason}"), folds, fold_score, seconds)
        }
    };
    let fold_sd = if fold_score.len() < 2 {
        0.0
    } else {
        let mean = fold_score.iter().sum::<f64>() / fold_score.len() as f64;
        (fold_score.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / fold_score.len() as f64)
            .sqrt()
    };
    Deep {
        status: "ok".to_owned(),
        reason: None,
        folds,
        robust_fitness: Some(robust_fitness_for(task, oof_score, &fold_score)),
        fold_score,
        oof_score: Some(oof_score),
        oof: oof_metrics,
        fold_sd: Some(fold_sd),
        seconds,
        produce_seconds: 0.0,
    }
}

/// `dsl` on every cut as `Ok` programs (the shape [`evaluate_deep_programs`] wants).
pub fn fixed_programs(dsl: &str, cuts: &[Cut]) -> Vec<std::result::Result<String, String>> {
    cuts.iter().map(|_| Ok(dsl.to_owned())).collect()
}

/// Whether `dsl` parses; the error otherwise (for validating caller programs up front).
pub fn parse_check(dsl: &str) -> Result<GenomeIr> {
    GenomeIr::parse(dsl).with_context(|| format!("program {dsl:?}"))
}

#[cfg(test)]
pub(crate) mod tests {
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

    /// A separable-ish binary target on three features, plus a fourth noise feature.
    pub(crate) fn toy_binary() -> (Rows, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = feat_math::Pcg32::new(5, 9);
        for _ in 0..160 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            let c = rng.next_f64();
            let noise = rng.next_f64();
            x.push(vec![a, b, c, noise]);
            let logit = 4.0 * (a - b) + 2.0 * (c - 0.5);
            y.push(if logit + 0.5 * (noise - 0.5) > 0.0 {
                1.0
            } else {
                0.0
            });
        }
        (x, y)
    }

    /// Three classes by the largest of three linear scores, plus a noise feature.
    pub(crate) fn toy_multiclass() -> (Rows, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = feat_math::Pcg32::new(13, 21);
        for _ in 0..180 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            let c = rng.next_f64();
            let noise = rng.next_f64();
            x.push(vec![a, b, c, noise]);
            let scores = [a + 0.1 * noise, b, c];
            let class = scores
                .iter()
                .enumerate()
                .max_by(|l, r| l.1.total_cmp(r.1))
                .map(|(i, _)| i)
                .unwrap();
            y.push(class as f64);
        }
        (x, y)
    }

    #[test]
    fn ridge_scores_and_garbage_is_rejected() {
        let (x, y) = toy();
        let cuts = build(&FoldPlan::new("toy", 1, 4), "toy", "s", x.len()).unwrap();
        let ok = evaluate("(pipe X (ridge))", &x, &y, &cuts[0], Task::Regression);
        assert!(ok.is_ok(), "{ok:?}");
        assert!(ok.score.unwrap() > 0.9);
        assert_eq!(ok.score, ok.metrics.get("r2").copied());
        assert!(ok.metrics.contains_key("rmse"));
        assert_eq!(ok.predictions.len(), cuts[0].hold.len());

        // The parser accepts any stage name and fit rejects it; either way the reason names it.
        let bad = evaluate(
            "(pipe X (no_such_head_ever))",
            &x,
            &y,
            &cuts[0],
            Task::Regression,
        );
        assert_eq!(bad.status, "rejected");
        let reason = bad.reason.as_deref().unwrap_or("");
        assert!(reason.contains("no_such_head_ever"), "{reason}");
        assert!(bad.score.is_none() && bad.metrics.is_empty());
        let unparsable = evaluate("(((", &x, &y, &cuts[0], Task::Regression);
        assert_eq!(unparsable.status, "rejected");
        assert!(unparsable
            .reason
            .as_deref()
            .unwrap_or("")
            .starts_with("parse"));

        let deep = evaluate_deep("(pipe X (ridge))", &x, &y, &cuts, Task::Regression);
        assert_eq!(deep.status, "ok", "{deep:?}");
        assert_eq!(deep.folds.len(), 4);
        assert!(deep.oof_score.unwrap() > 0.9);
        assert_eq!(deep.oof_score, deep.oof.get("r2").copied());
        assert!(deep.folds.iter().all(|f| f.outcome.predictions.is_empty()));
        assert!(deep.folds.iter().all(|f| f.dsl == "(pipe X (ridge))"));
        let expected = robust_fitness(deep.oof_score.unwrap(), &deep.fold_score);
        assert_eq!(deep.robust_fitness, Some(expected));

        // Per-cut programs: one bad fold rejects the pooled number, and the fold keeps its dsl.
        let mut programs = fixed_programs("(pipe X (ridge))", &cuts);
        programs[2] = Ok("(pipe X (no_such_head_ever))".to_owned());
        let mixed = evaluate_deep_programs(&programs, &x, &y, &cuts, Task::Regression);
        assert_eq!(mixed.status, "rejected");
        assert_eq!(mixed.reason.as_deref(), Some("a fold was rejected"));
        assert_eq!(mixed.folds[2].dsl, "(pipe X (no_such_head_ever))");
        assert_eq!(mixed.fold_score.len(), 3);
        assert_eq!(
            evaluate_deep_programs(&programs[..2], &x, &y, &cuts, Task::Regression).status,
            "rejected"
        );

        // A fold whose program could not be produced is a rejection naming the cause, not a
        // parse attempt on the error text.
        programs[2] = Err("no program".to_owned());
        let failed = evaluate_deep_programs(&programs, &x, &y, &cuts, Task::Regression);
        assert_eq!(failed.status, "rejected");
        assert_eq!(failed.folds[2].dsl, "");
        assert_eq!(
            failed.folds[2].outcome.reason.as_deref(),
            Some("search: no program")
        );
    }

    #[test]
    fn binary_programs_score_auc_and_regression_heads_are_refused() {
        let (x, y) = toy_binary();
        let cuts = build(&FoldPlan::new("bin", 1, 4), "bin", "s", x.len()).unwrap();
        let ok = evaluate("(pipe X (logit_ridge))", &x, &y, &cuts[0], Task::Binary);
        assert!(ok.is_ok(), "{ok:?}");
        let auc_value = ok.score.unwrap();
        assert!(auc_value > 0.85, "{auc_value}");
        assert_eq!(ok.score, ok.metrics.get("auc").copied());
        assert!(ok.metrics.contains_key("logloss") && ok.metrics.contains_key("brier"));
        assert!(ok.predictions.iter().all(|p| (0.0..=1.0).contains(p)));

        let refused = evaluate("(pipe X (ridge))", &x, &y, &cuts[0], Task::Binary);
        assert_eq!(refused.status, "rejected");
        assert!(
            refused
                .reason
                .as_deref()
                .unwrap_or("")
                .contains("incompatible with binary"),
            "{refused:?}"
        );
        let regression_on_binary =
            evaluate("(pipe X (logit_ridge))", &x, &y, &cuts[0], Task::Regression);
        assert_eq!(regression_on_binary.status, "rejected");

        let deep = evaluate_deep("(pipe X (tab_boost))", &x, &y, &cuts, Task::Binary);
        assert_eq!(deep.status, "ok", "{deep:?}");
        assert_eq!(deep.fold_score.len(), 4);
        assert!(deep.oof_score.unwrap() > 0.85);
        assert_eq!(deep.oof_score, deep.oof.get("auc").copied());
        assert_eq!(
            deep.robust_fitness,
            Some(robust_fitness_binary(
                deep.oof_score.unwrap(),
                &deep.fold_score
            ))
        );

        // A one-class held-out split has no AUC but keeps its other metrics and its rows.
        let mut skewed = y.clone();
        for &row in &cuts[1].hold {
            skewed[row] = 1.0;
        }
        let one_class = evaluate(
            "(pipe X (logit_ridge))",
            &x,
            &skewed,
            &cuts[1],
            Task::Binary,
        );
        assert!(one_class.is_ok(), "{one_class:?}");
        assert_eq!(one_class.score, None);
        assert!(!one_class.metrics.contains_key("auc"));
        assert!(one_class.metrics.contains_key("logloss"));
        assert!(one_class
            .note
            .as_deref()
            .unwrap_or("")
            .contains("one class"));
    }

    #[test]
    fn multiclass_programs_score_macro_f1() {
        let (x, y) = toy_multiclass();
        let cuts = build(&FoldPlan::new("mc", 1, 4), "mc", "s", x.len()).unwrap();
        let ok = evaluate(
            "(pipe X (multiclass_ridge))",
            &x,
            &y,
            &cuts[0],
            Task::Multiclass,
        );
        assert!(ok.is_ok(), "{ok:?}");
        assert!(ok.score.unwrap() > 0.6, "{:?}", ok.score);
        assert_eq!(ok.score, ok.metrics.get("macro_f1").copied());
        assert!(ok.metrics.contains_key("accuracy"));
        assert!(ok.predictions.iter().all(|p| p.fract() == 0.0));
        let deep = evaluate_deep(
            "(pipe X (multiclass_nearest_centroid))",
            &x,
            &y,
            &cuts,
            Task::Multiclass,
        );
        assert_eq!(deep.status, "ok", "{deep:?}");
        assert_eq!(
            deep.robust_fitness,
            Some(robust_fitness_bounded(
                deep.oof_score.unwrap(),
                &deep.fold_score
            ))
        );
        let refused = evaluate("(pipe X (logit_ridge))", &x, &y, &cuts[0], Task::Multiclass);
        assert_eq!(refused.status, "rejected");
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
        // Binary twin (exact_cv.rs::robust_fitness_binary): Gini scale, undefined folds skipped.
        assert!((robust_fitness_binary(0.75, &[0.75, 0.75]) - 0.5).abs() < 1e-12);
        assert!((robust_fitness_binary(0.75, &[0.75, f64::NAN, 0.75]) - 0.5).abs() < 1e-12);
        assert!((robust_fitness_binary(0.75, &[0.5, 1.0]) - (0.5 - 0.25 * 0.5)).abs() < 1e-12);
        assert_eq!(
            robust_fitness_binary(2.0, &[]),
            1.0,
            "AUC clamped to [0, 1]"
        );
        assert!((robust_fitness_bounded(0.8, &[0.7, 0.9]) - (0.8 - 0.25 * 0.1)).abs() < 1e-12);
        assert_eq!(
            robust_fitness_for(Task::Regression, 0.5, &[0.5, 0.5]),
            robust_fitness(0.5, &[0.5, 0.5])
        );
    }
}
