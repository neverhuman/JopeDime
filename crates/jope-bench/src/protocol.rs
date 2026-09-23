//! Screen, then deepen: every method on one cut; the top N on every cut. The committed number is
//! the deep one. This is the shape of the exact-CV campaign's CV5-fold-0 → top-10 → CV7 protocol,
//! applied to baselines and the searched program alike.

use anyhow::{ensure, Result};
use feat_math::Rows;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cuts::Cut;
use crate::eval::{evaluate, evaluate_deep, robust_fitness, Deep, Outcome};
use crate::methods::Method;

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
    /// `robust_fitness(r2, [r2])` of the screen score, i.e. `max(r2, -1)`.
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

/// Run every method through the protocol. Results come back in screen-rank order.
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
    let screen_cut = &cuts[protocol.screen_fold];
    let screens: Vec<Outcome> = methods
        .par_iter()
        .map(|method| evaluate(&method.dsl, x, y, screen_cut))
        .collect();
    let mut results: Vec<MethodResult> = methods
        .into_iter()
        .zip(screens)
        .map(|(method, screen)| {
            let screen_fitness = screen.r2.map(|r| robust_fitness(r, &[r]));
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
        .map(|&i| evaluate_deep(&results[i].method.dsl, x, y, cuts))
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
    use crate::methods::{baselines, MethodKind};

    #[test]
    fn screen_ranks_everyone_and_deepens_the_top() {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = feat_math::Pcg32::new(11, 5);
        for _ in 0..160 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            x.push(vec![a, b, rng.next_f64()]);
            y.push(a * a + 0.5 * b);
        }
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
        });
        let results = run(
            &x,
            &y,
            &cuts,
            methods,
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
    }
}
