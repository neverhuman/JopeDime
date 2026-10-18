//! `stack` — greedy ensemble selection with replacement (Caruana et al., 2004)
//! over a fixed five-model library: ridge (lambda_frac 0.02), ridge (0.002),
//! elastic net (defaults), histgbm2 (defaults), and krr (defaults).
//!
//! Out-of-fold predictions come from contiguous 5-fold splits (fold f holds
//! rows [f*n/5, (f+1)*n/5)); greedy forward selection (<= 20 additions,
//! strict-improvement hill climb) maximizes the R^2 of the running mean of
//! selected members; final weights are selection counts / total and every
//! selected member is refit on the full training set. Deterministic, and
//! bit-identical to the emitted Python/Rust artifacts.

use crate::boost2::{HistGbm2Model, HistGbm2Params};
use crate::heads::{ElasticNetModel, RidgeModel};
use crate::krr::{KrrModel, DEFAULT_M};
use crate::metrics::r2;
use crate::Rows;

pub const MAX_ITERS: usize = 20;
pub const FOLDS: usize = 5;
/// Below this row count the library degenerates; fall back to plain ridge.
pub const MIN_N: usize = 20;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StackModel {
    /// Selection frequencies for the five members, in library order.
    pub weights: Vec<f64>,
    pub ridge_a: Option<RidgeModel>,
    pub ridge_b: Option<RidgeModel>,
    pub net: Option<ElasticNetModel>,
    pub trees: Option<HistGbm2Model>,
    pub kernel: Option<KrrModel>,
}

fn member_holdout(b: usize, xtr: &Rows, ytr: &[f64], xva: &Rows) -> Vec<f64> {
    match b {
        0 => RidgeModel::fit(xtr, ytr, 0.02).predict(xva),
        1 => RidgeModel::fit(xtr, ytr, 0.002).predict(xva),
        2 => ElasticNetModel::fit(xtr, ytr, 0.001, 0.5, 60).predict(xva),
        3 => HistGbm2Model::fit(xtr, ytr, &HistGbm2Params::default()).predict(xva),
        _ => KrrModel::fit(xtr, ytr, DEFAULT_M).predict(xva),
    }
}

impl StackModel {
    pub fn fit(x: &Rows, y: &[f64]) -> StackModel {
        let n = x.len().min(y.len());
        if n < MIN_N {
            return StackModel {
                weights: vec![1.0, 0.0, 0.0, 0.0, 0.0],
                ridge_a: Some(RidgeModel::fit(x, y, 0.02)),
                ridge_b: None,
                net: None,
                trees: None,
                kernel: None,
            };
        }
        let mut oof: Vec<Vec<f64>> = vec![vec![0.0; n]; 5];
        for f in 0..FOLDS {
            let lo = f * n / FOLDS;
            let hi = (f + 1) * n / FOLDS;
            if lo >= hi {
                continue;
            }
            let mut xtr: Rows = Vec::with_capacity(n - (hi - lo));
            let mut ytr: Vec<f64> = Vec::with_capacity(n - (hi - lo));
            for r in 0..lo {
                xtr.push(x[r].clone());
                ytr.push(y[r]);
            }
            for r in hi..n {
                xtr.push(x[r].clone());
                ytr.push(y[r]);
            }
            let xva: Rows = x[lo..hi].to_vec();
            for (b, oof_b) in oof.iter_mut().enumerate() {
                let preds = member_holdout(b, &xtr, &ytr, &xva);
                for (k, slot) in oof_b[lo..hi].iter_mut().enumerate() {
                    *slot = preds[k];
                }
            }
        }
        let yn = &y[..n];
        let mut counts = [0usize; 5];
        let mut total = 0usize;
        let mut sum_vec = vec![0.0f64; n];
        let mut current = f64::NEG_INFINITY;
        for _ in 0..MAX_ITERS {
            let mut best_b = usize::MAX;
            let mut best_score = current;
            for (b, oof_b) in oof.iter().enumerate() {
                let denom = (total + 1) as f64;
                let cand: Vec<f64> = (0..n).map(|i| (sum_vec[i] + oof_b[i]) / denom).collect();
                let s = r2(yn, &cand);
                if s > best_score {
                    best_score = s;
                    best_b = b;
                }
            }
            if best_b == usize::MAX {
                break;
            }
            for (i, sv) in sum_vec.iter_mut().enumerate() {
                *sv += oof[best_b][i];
            }
            counts[best_b] += 1;
            total += 1;
            current = best_score;
        }
        if total == 0 {
            counts[0] = 1;
            total = 1;
        }
        let weights: Vec<f64> = counts.iter().map(|&c| c as f64 / total as f64).collect();
        StackModel {
            weights,
            ridge_a: if counts[0] > 0 {
                Some(RidgeModel::fit(x, y, 0.02))
            } else {
                None
            },
            ridge_b: if counts[1] > 0 {
                Some(RidgeModel::fit(x, y, 0.002))
            } else {
                None
            },
            net: if counts[2] > 0 {
                Some(ElasticNetModel::fit(x, y, 0.001, 0.5, 60))
            } else {
                None
            },
            trees: if counts[3] > 0 {
                Some(HistGbm2Model::fit(x, y, &HistGbm2Params::default()))
            } else {
                None
            },
            kernel: if counts[4] > 0 {
                Some(KrrModel::fit(x, y, DEFAULT_M))
            } else {
                None
            },
        }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        let mut acc = vec![0.0f64; x.len()];
        if let Some(m) = &self.ridge_a {
            let p = m.predict(x);
            for (i, a) in acc.iter_mut().enumerate() {
                *a += self.weights[0] * p[i];
            }
        }
        if let Some(m) = &self.ridge_b {
            let p = m.predict(x);
            for (i, a) in acc.iter_mut().enumerate() {
                *a += self.weights[1] * p[i];
            }
        }
        if let Some(m) = &self.net {
            let p = m.predict(x);
            for (i, a) in acc.iter_mut().enumerate() {
                *a += self.weights[2] * p[i];
            }
        }
        if let Some(m) = &self.trees {
            let p = m.predict(x);
            for (i, a) in acc.iter_mut().enumerate() {
                *a += self.weights[3] * p[i];
            }
        }
        if let Some(m) = &self.kernel {
            let p = m.predict(x);
            for (i, a) in acc.iter_mut().enumerate() {
                *a += self.weights[4] * p[i];
            }
        }
        acc
    }
}

/// Explicit successor split contract. Legacy StackModel continues to use its
/// historical contiguous folds; grouped/temporal callers must provide their own splits.
pub fn numeric_v2_iid_folds(n: usize, folds: usize, seed: u64) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(folds >= 2 && n >= folds, "invalid IID fold shape");
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = crate::Pcg32::new(seed, 0x6e756d6572696332);
    for i in (1..n).rev() {
        let j = rng.next_u32() as usize % (i + 1);
        order.swap(i, j);
    }
    let mut result = vec![0; n];
    for (position, index) in order.into_iter().enumerate() {
        result[index] = position % folds;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cv_r2;
    use crate::rng::Pcg32;
    use crate::GenomeIr;

    fn toy() -> (Rows, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = Pcg32::new(1, 1);
        for _ in 0..240 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            let c = rng.next_f64();
            x.push(vec![a, b, c]);
            y.push(2.0 * a - b + 0.3 * (6.0 * c).sin());
        }
        (x, y)
    }

    #[test]
    fn stack_is_deterministic() {
        let (x, y) = toy();
        let m1 = StackModel::fit(&x, &y);
        let m2 = StackModel::fit(&x, &y);
        assert_eq!(m1.predict(&x), m2.predict(&x));
        assert_eq!(m1.weights, m2.weights);
        let wsum: f64 = m1.weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-12, "weights={:?}", m1.weights);
    }

    #[test]
    fn stack_matches_or_beats_single_heads_on_toy_cv() {
        let (x, y) = toy();
        let singles = [
            "(pipe X (ridge lambda_frac=0.02))",
            "(pipe X (ridge lambda_frac=0.002))",
            "(pipe X (elasticnet))",
            "(pipe X (histgbm2))",
            "(pipe X (krr))",
        ];
        let mut best = f64::NEG_INFINITY;
        for src in singles {
            let rep = cv_r2(&GenomeIr::parse(src).unwrap(), &x, &y, 5, "toy").unwrap();
            println!("accuracy[toy cv oof_r2] {src} = {}", rep.oof_r2);
            if rep.oof_r2 > best {
                best = rep.oof_r2;
            }
        }
        let stack = cv_r2(
            &GenomeIr::parse("(pipe X (stack))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        println!(
            "accuracy[toy cv oof_r2] stack = {} (best single {})",
            stack.oof_r2, best
        );
        assert!(
            stack.oof_r2 >= best - 0.01,
            "stack oof={} best single={}",
            stack.oof_r2,
            best
        );
    }

    #[test]
    fn stack_small_n_falls_back_to_ridge() {
        let x: Rows = (0..10).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..10).map(|i| 2.0 * i as f64).collect();
        let m = StackModel::fit(&x, &y);
        assert_eq!(m.weights[0], 1.0);
        assert!(m.ridge_a.is_some() && m.trees.is_none());
    }
}
