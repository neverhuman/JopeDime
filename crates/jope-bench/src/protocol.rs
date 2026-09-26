//! Screen, then deepen: every method on one cut; the top N on every cut. The committed number is
//! the deep one. This is the shape of the exact-CV campaign's CV5-fold-0 → top-10 → CV7 protocol,
//! applied to baselines and the searched program alike.

use anyhow::{bail, ensure, Result};
use feat_math::Rows;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::adapter::{evaluate_adapter, evaluate_deep_adapter};
use crate::cuts::Cut;
use crate::eval::{evaluate, evaluate_deep_programs, fixed_programs, take, Deep, Outcome};
use crate::methods::{search_program, Method};
use crate::task::Task;

/// Protocol parameters.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Protocol {
    /// The cut every method is screened on.
    pub screen_fold: usize,
    /// How many screened methods are re-scored on every cut.
    pub top: usize,
}

/// One method through the protocol.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MethodResult {
    /// The method.
    pub method: Method,
    /// The screen cut.
    pub screen_cut_id: String,
    /// Screen score.
    pub screen: Outcome,
    /// The screen ranking key: the task's primary score on the screen cut ([`Task::score_metric`]),
    /// **raw** (R² uncapped), so a degenerate fold (every method far below zero) still orders
    /// methods instead of tying them all at −1. When the primary score is undefined for an `ok`
    /// outcome (a binary screen cut whose held-out rows are one class, where no method has an
    /// AUC) the key is the negated log loss instead ([`screen_key`]), so the finalists are still
    /// the least-bad methods rather than an alphabetical pick.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screen_fitness: Option<f64>,
    /// 1-based rank by screen fitness (methods without a key last, then by id).
    pub screen_rank: usize,
    /// Deep score, for the top N only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deep: Option<Deep>,
    /// 1-based rank by deep robust fitness among deepened methods that scored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deep_rank: Option<usize>,
}

/// The screen ranking key of an outcome: its primary score, else (for an `ok` outcome whose
/// primary score is undefined) the negated log loss when recorded, else nothing.
pub fn screen_key(outcome: &Outcome) -> Option<f64> {
    if !outcome.is_ok() {
        return None;
    }
    outcome
        .score
        .or_else(|| outcome.metrics.get("logloss").map(|loss| -loss))
}

fn rank_key(fitness: Option<f64>, id: &str) -> (u8, f64, String) {
    match fitness {
        Some(f) => (0, -f, id.to_owned()),
        None => (1, 0.0, id.to_owned()),
    }
}

/// Indices of `entries` in rank order: by fitness descending (raw, however negative), entries
/// without a fitness last, ties by id.
pub fn rank_order(entries: &[(Option<f64>, &str)]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| {
        rank_key(entries[a].0, entries[a].1)
            .partial_cmp(&rank_key(entries[b].0, entries[b].1))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// Screen-score one method on one cut: the adapter it names, else its program.
fn screen(method: &Method, x: &Rows, y: &[f64], cut: &Cut, task: Task) -> Outcome {
    match &method.adapter {
        Some(spec) => evaluate_adapter(spec, x, y, cut, task),
        None => evaluate(&method.dsl, x, y, cut, task),
    }
}

/// Deep-score one method: the adapter on every cut; the fixed program on every cut; or, for a
/// method with `resample`, the search re-run on each cut's training rows and that cut's own
/// program scored. A cut whose search fails is a rejected fold naming the cause.
fn deepen(method: &Method, x: &Rows, y: &[f64], cuts: &[Cut], task: Task) -> Deep {
    if let Some(spec) = &method.adapter {
        return evaluate_deep_adapter(spec, x, y, cuts, task);
    }
    let Some(config) = &method.resample else {
        return evaluate_deep_programs(&fixed_programs(&method.dsl, cuts), x, y, cuts, task);
    };
    if task != Task::Regression {
        let programs: Vec<Result<String, String>> = cuts
            .iter()
            .map(|_| Err("the search is regression-only".to_owned()))
            .collect();
        return evaluate_deep_programs(&programs, x, y, cuts, task);
    }
    let started = std::time::Instant::now();
    let programs: Vec<Result<String, String>> = cuts
        .par_iter()
        .map(|cut| {
            let (x_train, y_train) = take(x, y, &cut.train);
            search_program(&x_train, &y_train, config).map_err(|error| format!("{error:#}"))
        })
        .collect();
    let produce_seconds = started.elapsed().as_secs_f64();
    let mut deep = evaluate_deep_programs(&programs, x, y, cuts, task);
    deep.produce_seconds = produce_seconds;
    deep
}

/// Run every method through the protocol. Results come back in screen-rank order. Method ids
/// must be unique: two rows with one id would collapse on the ladder.
pub fn run(
    x: &Rows,
    y: &[f64],
    cuts: &[Cut],
    methods: Vec<Method>,
    protocol: &Protocol,
    task: Task,
) -> Result<Vec<MethodResult>> {
    ensure!(!methods.is_empty(), "no methods to benchmark");
    ensure!(
        protocol.screen_fold < cuts.len(),
        "screen fold {} but only {} cuts",
        protocol.screen_fold,
        cuts.len()
    );
    let mut seen = std::collections::BTreeSet::new();
    for method in &methods {
        if !seen.insert(method.id.as_str()) {
            bail!("duplicate method id {:?}", method.id);
        }
    }
    let screen_cut = &cuts[protocol.screen_fold];
    let screens: Vec<Outcome> = methods
        .par_iter()
        .map(|method| screen(method, x, y, screen_cut, task))
        .collect();
    let mut results: Vec<MethodResult> = methods
        .into_iter()
        .zip(screens)
        .map(|(method, screen)| {
            let screen_fitness = screen_key(&screen);
            MethodResult {
                method,
                screen_cut_id: screen_cut.cut_id.clone(),
                screen: Outcome {
                    predictions: Vec::new(),
                    ..screen
                },
                screen_fitness,
                screen_rank: 0,
                deep: None,
                deep_rank: None,
            }
        })
        .collect();

    let entries: Vec<(Option<f64>, &str)> = results
        .iter()
        .map(|r| (r.screen_fitness, r.method.id.as_str()))
        .collect();
    let order = rank_order(&entries);
    for (rank, &i) in order.iter().enumerate() {
        results[i].screen_rank = rank + 1;
    }
    let finalists: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&i| results[i].screen.is_ok())
        .take(protocol.top)
        .collect();
    let deeps: Vec<Deep> = finalists
        .par_iter()
        .map(|&i| deepen(&results[i].method, x, y, cuts, task))
        .collect();
    for (&i, deep) in finalists.iter().zip(deeps) {
        results[i].deep = Some(deep);
    }
    let deep_entries: Vec<(Option<f64>, &str)> = finalists
        .iter()
        .map(|&i| {
            let r = &results[i];
            (
                r.deep.as_ref().and_then(|d| d.robust_fitness),
                r.method.id.as_str(),
            )
        })
        .collect();
    let deep_order = rank_order(&deep_entries);
    let mut deep_rank = 0;
    for &j in &deep_order {
        let i = finalists[j];
        if results[i].deep.as_ref().is_some_and(|d| d.status == "ok") {
            deep_rank += 1;
            results[i].deep_rank = Some(deep_rank);
        }
    }
    results.sort_by_key(|r| r.screen_rank);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuts::{build, FoldPlan};
    use crate::methods::{default_baselines, SearchConfig};

    fn toy() -> (Rows, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = feat_math::Pcg32::new(11, 5);
        for _ in 0..160 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            x.push(vec![a, b, rng.next_f64()]);
            y.push(a * a + 0.5 * b);
        }
        (x, y)
    }

    // `screen_ranks_everyone_and_deepens_the_top` runs the constant adapter process, so it lives
    // in tests/adapter_contract.rs, where the package binary is available.

    #[test]
    fn far_below_zero_still_orders_and_unscored_ranks_last() {
        // Two methods far below R² = −1 (where the campaign fitness would tie them) keep their
        // order by raw value; a method without a score ranks last; ties break by id.
        let entries = [
            (Some(-3.0), "baseline.b"),
            (None, "program.unscored"),
            (Some(-1.5), "baseline.a"),
            (Some(0.2), "baseline.c"),
            (Some(-1.5), "baseline.0"),
        ];
        let order = rank_order(&entries);
        let ids: Vec<&str> = order.iter().map(|&i| entries[i].1).collect();
        assert_eq!(
            ids,
            [
                "baseline.c",
                "baseline.0",
                "baseline.a",
                "baseline.b",
                "program.unscored"
            ]
        );
    }

    #[test]
    fn binary_and_multiclass_run_through_the_protocol() {
        let (x, y) = crate::eval::tests::toy_binary();
        let cuts = build(&FoldPlan::new("b", 1, 4), "b", "s", x.len()).unwrap();
        let protocol = Protocol {
            screen_fold: 0,
            top: 2,
        };
        let results = run(
            &x,
            &y,
            &cuts,
            default_baselines(Task::Binary).unwrap(),
            &protocol,
            Task::Binary,
        )
        .unwrap();
        assert_eq!(results.len(), Task::Binary.default_heads().len());
        assert!(results.iter().all(|r| r.screen.is_ok()), "{results:#?}");
        assert!(results[0].screen_fitness.unwrap() > 0.8);
        assert_eq!(results.iter().filter(|r| r.deep_rank.is_some()).count(), 2);
        let deep = results[0].deep.as_ref().unwrap();
        assert!(deep.oof.contains_key("auc") && deep.oof.contains_key("logloss"));

        // A one-class screen cut: no method has an AUC there, so the screen ranks by the negated
        // log loss instead of falling back to an alphabetical pick; the deep stage still pools
        // every row and scores AUC.
        let mut skewed = y.clone();
        for &row in &cuts[0].hold {
            skewed[row] = 1.0;
        }
        let results = run(
            &x,
            &skewed,
            &cuts,
            default_baselines(Task::Binary).unwrap(),
            &protocol,
            Task::Binary,
        )
        .unwrap();
        assert!(results
            .iter()
            .all(|r| r.screen.is_ok() && r.screen.score.is_none()));
        for r in &results {
            let expected = -r.screen.metrics["logloss"];
            assert_eq!(r.screen_fitness, Some(expected), "{}", r.method.id);
        }
        let keys: Vec<f64> = results.iter().map(|r| r.screen_fitness.unwrap()).collect();
        assert!(keys.windows(2).all(|w| w[0] >= w[1]), "{keys:?}");
        assert_eq!(results.iter().filter(|r| r.deep_rank.is_some()).count(), 2);

        let (x, y) = crate::eval::tests::toy_multiclass();
        let cuts = build(&FoldPlan::new("m", 1, 4), "m", "s", x.len()).unwrap();
        let results = run(
            &x,
            &y,
            &cuts,
            default_baselines(Task::Multiclass).unwrap(),
            &protocol,
            Task::Multiclass,
        )
        .unwrap();
        assert!(results.iter().all(|r| r.screen.is_ok()), "{results:#?}");
        assert_eq!(results.iter().filter(|r| r.deep_rank.is_some()).count(), 2);
        assert!(results[0]
            .deep
            .as_ref()
            .unwrap()
            .oof
            .contains_key("macro_f1"));
    }

    #[test]
    fn resampled_search_produces_a_program_per_cut() {
        let (x, y) = toy();
        let cuts = build(&FoldPlan::new("q", 1, 3), "q", "s", x.len()).unwrap();
        let config = SearchConfig {
            pop: 8,
            gens: 1,
            seed: 5,
            seconds: 20.0,
        };
        let (x_train, y_train) = take(&x, &y, &cuts[0].train);
        let mut searched = crate::methods::search(&x_train, &y_train, &config).unwrap();
        assert!(searched.produce_seconds > 0.0);
        searched.resample = Some(config);
        searched.id = crate::methods::RESAMPLED_SEARCH_METHOD_ID.to_owned();
        let protocol = Protocol {
            screen_fold: 0,
            top: 1,
        };
        let results = run(
            &x,
            &y,
            &cuts,
            vec![searched.clone()],
            &protocol,
            Task::Regression,
        )
        .unwrap();
        let deep = results[0].deep.as_ref().expect("deepened");
        assert_eq!(deep.folds.len(), 3);
        assert!(deep.produce_seconds > 0.0, "per-cut search time recorded");
        assert!(
            deep.folds.iter().all(|f| f.dsl.starts_with("(pipe X")),
            "{:?}",
            deep.folds
        );

        // The search is regression-only: asked to resample on a binary task, every fold is a
        // rejection that says so, not a parse error.
        let (xb, yb) = crate::eval::tests::toy_binary();
        let cuts_b = build(&FoldPlan::new("qb", 1, 3), "qb", "s", xb.len()).unwrap();
        let deep = deepen(&searched, &xb, &yb, &cuts_b, Task::Binary);
        assert_eq!(deep.status, "rejected");
        assert!(deep
            .folds
            .iter()
            .all(|f| f.outcome.reason.as_deref() == Some("search: the search is regression-only")));
    }
}
