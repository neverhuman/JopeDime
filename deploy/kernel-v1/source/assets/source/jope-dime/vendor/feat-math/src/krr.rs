//! `krr` — exact kernel ridge regression with a Gaussian (RBF) kernel on
//! inverse-standard-deviation scaled features ("ARD-lite"). The kernel width
//! gamma and the ridge penalty lambda come from a fixed 5x5 grid scored by
//! RMSE on an internal contiguous 80/20 holdout; grid selection runs on a
//! seeded basis of at most `M_SEL` rows so the 25 Cholesky solves stay cheap,
//! and the final model is refit once on up to `m` seeded rows.
//!
//! Bit-parity contract: the emitted Python/Rust artifacts replay exactly the
//! same arithmetic order (distances, exp, Cholesky, holdout RMSE).

use crate::heads::{cholesky_solve, std_fit};
use crate::metrics::rmse;
use crate::rng::Pcg32;
use crate::Rows;

/// Default cap on kernel centers for the final solve.
pub const DEFAULT_M: usize = 800;
/// Cap on kernel centers during grid selection (25 solves).
pub const M_SEL: usize = 256;
pub const GAMMA_COEFS: [f64; 5] = [0.1, 0.3, 1.0, 3.0, 10.0];
pub const LAMBDAS: [f64; 5] = [0.001, 0.01, 0.1, 1.0, 10.0];
pub const SEL_SEED: u64 = 401;
pub const FIN_SEED: u64 = 402;
pub const SUBSET_STREAM: u64 = 54;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KrrModel {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    /// Standardized kernel-center rows (empty => constant fallback).
    pub basis: Vec<Vec<f64>>,
    pub alpha: Vec<f64>,
    pub gamma: f64,
    pub y_mean: f64,
}

/// Partial Fisher-Yates: k distinct rows out of n, then sorted ascending.
pub fn seeded_rows(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = Pcg32::new(seed, SUBSET_STREAM);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u32() as usize) % (n - i);
        idx.swap(i, j);
    }
    let mut out = idx[..k].to_vec();
    out.sort_unstable();
    out
}

fn sqdist(a: &[f64], b: &[f64]) -> f64 {
    let mut d = 0.0;
    for j in 0..a.len().min(b.len()) {
        let t = a[j] - b[j];
        d += t * t;
    }
    d
}

impl KrrModel {
    pub fn fit(x: &Rows, y: &[f64], m: usize) -> KrrModel {
        let n = x.len().min(y.len());
        let p = if x.is_empty() { 0 } else { x[0].len() };
        let (means, scales) = std_fit(x);
        let mut y_mean = 0.0;
        for &v in &y[..n] {
            y_mean += v;
        }
        y_mean /= n.max(1) as f64;
        if n < 12 || p == 0 {
            return KrrModel {
                means,
                scales,
                basis: Vec::new(),
                alpha: Vec::new(),
                gamma: 0.0,
                y_mean,
            };
        }
        let z: Rows = (0..n)
            .map(|i| (0..p).map(|j| (x[i][j] - means[j]) / scales[j]).collect())
            .collect();
        let n_tr = (n * 4) / 5;
        let n_hold = n - n_tr;
        let k_sel = n_tr.min(M_SEL);
        let sel: Vec<usize> = if n_tr > M_SEL {
            seeded_rows(n_tr, M_SEL, SEL_SEED)
        } else {
            (0..n_tr).collect()
        };
        let ys: Vec<f64> = sel.iter().map(|&i| y[i] - y_mean).collect();
        // Pairwise squared distances are gamma-independent: compute once.
        let mut d2ss = vec![0.0f64; k_sel * k_sel];
        for a in 0..k_sel {
            for b in (a + 1)..k_sel {
                let d = sqdist(&z[sel[a]], &z[sel[b]]);
                d2ss[a * k_sel + b] = d;
                d2ss[b * k_sel + a] = d;
            }
        }
        let mut d2hs = vec![0.0f64; n_hold * k_sel];
        for hh in 0..n_hold {
            for b in 0..k_sel {
                d2hs[hh * k_sel + b] = sqdist(&z[n_tr + hh], &z[sel[b]]);
            }
        }
        let y_hold = &y[n_tr..n];
        let mut best_score = f64::INFINITY;
        let mut best_gamma = GAMMA_COEFS[0] / p as f64;
        let mut best_lambda = LAMBDAS[0];
        for coef in GAMMA_COEFS {
            let gamma = coef / p as f64;
            let k0: Vec<f64> = d2ss.iter().map(|d| (-gamma * d).exp()).collect();
            let kh: Vec<f64> = d2hs.iter().map(|d| (-gamma * d).exp()).collect();
            for lam in LAMBDAS {
                let mut a = k0.clone();
                for q in 0..k_sel {
                    a[q * k_sel + q] += lam;
                }
                let mut b = ys.clone();
                let alpha = cholesky_solve(&mut a, &mut b, k_sel);
                let mut preds = Vec::with_capacity(n_hold);
                for hh in 0..n_hold {
                    let mut s = y_mean;
                    for q in 0..k_sel {
                        s += kh[hh * k_sel + q] * alpha[q];
                    }
                    preds.push(s);
                }
                let score = rmse(y_hold, &preds);
                if score < best_score {
                    best_score = score;
                    best_gamma = gamma;
                    best_lambda = lam;
                }
            }
        }
        // Final refit on up to m seeded rows of the full training set.
        let fin: Vec<usize> = if n > m {
            seeded_rows(n, m, FIN_SEED)
        } else {
            (0..n).collect()
        };
        let k_fin = fin.len();
        let basis: Rows = fin.iter().map(|&i| z[i].clone()).collect();
        let yf: Vec<f64> = fin.iter().map(|&i| y[i] - y_mean).collect();
        let mut a = vec![0.0f64; k_fin * k_fin];
        for i2 in 0..k_fin {
            for j2 in (i2 + 1)..k_fin {
                let d = sqdist(&basis[i2], &basis[j2]);
                let v = (-best_gamma * d).exp();
                a[i2 * k_fin + j2] = v;
                a[j2 * k_fin + i2] = v;
            }
        }
        for i2 in 0..k_fin {
            a[i2 * k_fin + i2] = 1.0 + best_lambda;
        }
        let mut b = yf.clone();
        let alpha = cholesky_solve(&mut a, &mut b, k_fin);
        KrrModel {
            means,
            scales,
            basis,
            alpha,
            gamma: best_gamma,
            y_mean,
        }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        let p = self.means.len();
        x.iter()
            .map(|row| {
                if self.basis.is_empty() {
                    return self.y_mean.clamp(-60.0, 60.0);
                }
                let zr: Vec<f64> = (0..p.min(row.len()))
                    .map(|j| (row[j] - self.means[j]) / self.scales[j])
                    .collect();
                let mut s = self.y_mean;
                for (b, zb) in self.basis.iter().enumerate() {
                    let d = sqdist(&zr, zb);
                    s += self.alpha[b] * (-self.gamma * d).exp();
                }
                s.clamp(-60.0, 60.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cv_r2;
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
    fn krr_is_deterministic_and_learns_toy() {
        let (x, y) = toy();
        let m1 = KrrModel::fit(&x, &y, DEFAULT_M);
        let m2 = KrrModel::fit(&x, &y, DEFAULT_M);
        assert_eq!(m1.predict(&x), m2.predict(&x));
        let rep = cv_r2(
            &GenomeIr::parse("(pipe X (krr))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        assert!(rep.oof_r2 > 0.9, "krr oof_r2={}", rep.oof_r2);
    }

    #[test]
    fn krr_beats_plain_ridge_on_nonlinearity() {
        let (x, y) = toy();
        let ridge = cv_r2(
            &GenomeIr::parse("(pipe X (ridge))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        let krr = cv_r2(
            &GenomeIr::parse("(pipe X (krr))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        assert!(
            krr.oof_r2 > ridge.oof_r2,
            "krr={} ridge={}",
            krr.oof_r2,
            ridge.oof_r2
        );
    }

    #[test]
    fn krr_subset_cap_is_exercised() {
        let (x, y) = toy();
        let m = KrrModel::fit(&x, &y, 120);
        assert_eq!(m.basis.len(), 120);
        let m2 = KrrModel::fit(&x, &y, 120);
        assert_eq!(m.predict(&x), m2.predict(&x));
    }

    #[test]
    fn krr_small_n_falls_back_to_mean() {
        let x: Rows = (0..6).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let m = KrrModel::fit(&x, &y, DEFAULT_M);
        assert!(m.basis.is_empty());
        assert!(m.predict(&x).iter().all(|v| (*v - 2.5).abs() < 1e-12));
    }
}
