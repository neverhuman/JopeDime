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

mod deep;
mod outcome;

pub use deep::{
    evaluate_deep, evaluate_deep_programs, fixed_programs, parse_check, pool, robust_fitness,
    robust_fitness_binary, robust_fitness_bounded, robust_fitness_for, Deep, FoldResult,
};
pub use outcome::{evaluate, numeric_schema, numeric_table, score, take, Metrics, Outcome};

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
