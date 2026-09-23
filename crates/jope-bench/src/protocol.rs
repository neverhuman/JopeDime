//! Screen, then deepen: every method on one cut; the top N on every cut. The committed number is
//! the deep one. This is the shape of the exact-CV campaign's CV5-fold-0 → top-10 → CV7 protocol,
//! applied to baselines and the searched program alike.

use anyhow::{bail, ensure, Result};
use feat_math::Rows;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cuts::Cut;
use crate::eval::{evaluate, evaluate_deep_programs, take, Deep, Outcome};
use crate::methods::{search_program, Method};

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
    /// The screen ranking key: held-out R² on the screen cut, **uncapped**, so a degenerate fold
    /// (every method far below zero) still orders methods instead of tying them all at −1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screen_fitness: Option<f64>,
    /// 1-based rank by screen fitness (rejected methods last, then by id).
    pub screen_rank: usize,
    /// Deep score, for the top N only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deep: Option<Deep>,
    /// 1-based rank by deep robust fitness among deepened methods that scored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deep_rank: Option<usize>,
}

fn rank_key(fitness: Option<f64>, id: &str) -> (u8, f64, String) {
    match fitness {
        Some(f) => (0, -f, id.to_owned()),
        None => (1, 0.0, id.to_owned()),
    }
}

fn order_by<F: Fn(usize) -> (Option<f64>, String)>(n: usize, key: F) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let (fa, ia) = key(a);
        let (fb, ib) = key(b);
        rank_key(fa, &ia)
            .partial_cmp(&rank_key(fb, &ib))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// Deep-score one method: the fixed program on every cut, or, for a method with `resample`, the
/// search re-run on each cut's training rows and that cut's own program scored.
fn deepen(method: &Method, x: &Rows, y: &[f64], cuts: &[Cut]) -> Deep {
    let Some(config) = &method.resample else {
        let programs: Vec<String> = cuts.iter().map(|_| method.dsl.clone()).collect();
        return evaluate_deep_programs(&programs, x, y, cuts);
    };
    let started = std::time::Instant::now();
    let found: Vec<Result<String>> = cuts
        .par_iter()
        .map(|cut| {
            let (x_train, y_train) = take(x, y, &cut.train);
            search_program(&x_train, &y_train, config)
        })
        .collect();
    let produce_seconds = started.elapsed().as_secs_f64();
    let programs: Vec<String> = found
        .into_iter()
        .map(|r| r.unwrap_or_else(|error| format!("(search failed: {error})")))
        .collect();
    let mut deep = evaluate_deep_programs(&programs, x, y, cuts);
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
        .map(|method| evaluate(&method.dsl, x, y, screen_cut))
        .collect();
    let mut results: Vec<MethodResult> = methods
        .into_iter()
        .zip(screens)
        .map(|(method, screen)| {
            let screen_fitness = screen.r2;
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

    let order = order_by(results.len(), |i| {
        (results[i].screen_fitness, results[i].method.id.clone())
    });
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
        .map(|&i| deepen(&results[i].method, x, y, cuts))
        .collect();
    for (&i, deep) in finalists.iter().zip(deeps) {
        results[i].deep = Some(deep);
    }
    let deep_order = order_by(finalists.len(), |j| {
        let r = &results[finalists[j]];
        (
            r.deep.as_ref().and_then(|d| d.robust_fitness),
            r.method.id.clone(),
        )
    });
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
    use crate::methods::{baselines, MethodKind, SearchConfig};

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

    #[test]
    fn screen_ranks_everyone_and_deepens_the_top() {
        let (x, y) = toy();
        let cuts = build(&FoldPlan::new("p", 1, 4), "p", "s", x.len()).unwrap();
        let mut methods = baselines(&[
            "ridge".to_owned(),
            "histgbm".to_owned(),
            "ksmooth".to_owned(),
        ])
        .unwrap();
        methods.push(Method {
            id: "program.broken".to_owned(),
            kind: MethodKind::Program,
            dsl: "(pipe X (no_such_head_ever))".to_owned(),
            produce_seconds: 0.0,
            resample: None,
        });
        let results = run(
            &x,
            &y,
            &cuts,
            methods.clone(),
            &Protocol {
                screen_fold: 0,
                top: 2,
            },
        )
        .unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(
            results.iter().map(|r| r.screen_rank).collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );
        assert_eq!(
            results[3].method.id, "program.broken",
            "rejected ranks last"
        );
        assert!(results[3].deep.is_none());
        assert_eq!(results.iter().filter(|r| r.deep.is_some()).count(), 2);
        assert!(results[0].deep_rank.is_some());
        let deep = results[0].deep.as_ref().unwrap();
        assert_eq!(deep.folds.len(), 4);
        assert!(results[0].screen.predictions.is_empty());
        for r in &results {
            assert_eq!(r.screen_cut_id, cuts[0].cut_id);
        }
        // Screen fitness is the raw held-out R², so ranks follow it strictly.
        let fits: Vec<f64> = results[..3]
            .iter()
            .map(|r| r.screen_fitness.unwrap())
            .collect();
        assert!(fits[0] >= fits[1] && fits[1] >= fits[2], "{fits:?}");

        // Duplicate ids are refused.
        methods.push(methods[0].clone());
        let error = run(
            &x,
            &y,
            &cuts,
            methods,
            &Protocol {
                screen_fold: 0,
                top: 2,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("duplicate method id"), "{error}");
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
        let results = run(
            &x,
            &y,
            &cuts,
            vec![searched],
            &Protocol {
                screen_fold: 0,
                top: 1,
            },
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
    }
}
