//! `ksmooth` — Nadaraya-Watson kernel regression (Nadaraya, 1964; Watson,
//! 1964): the prediction is a Gaussian-kernel-weighted average of the targets
//! of the `k` nearest training rows in standardized feature space.
//!
//! Weights are w_i = exp(-gamma * (d2_i - d2_min) / p) over the k nearest rows
//! (squared Euclidean distance, ties broken by the ascending row index), where
//! subtracting the nearest distance d2_min keeps the largest weight at exactly
//! 1 so the denominator can never underflow to zero; dividing the exponent by
//! the feature count p keeps the effective bandwidth O(1) as width grows.
//! Fit stores only standardized training rows and their targets, so the head
//! is out-of-fold safe by construction under honest CV. Deterministic; the
//! emitted Python/Rust artifacts replay the same arithmetic order bit-for-bit.

use crate::heads::std_fit;
use crate::Rows;

/// Default neighbor count.
pub const DEFAULT_K: usize = 32;
/// Default kernel width multiplier.
pub const DEFAULT_GAMMA: f64 = 1.0;
/// Below this row count the smoother degenerates; fall back to the mean.
pub const MIN_N: usize = 8;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KsmoothModel {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    /// Standardized training rows (empty => constant fallback).
    pub z: Vec<Vec<f64>>,
    pub ys: Vec<f64>,
    pub k: usize,
    pub gamma: f64,
    pub y_mean: f64,
}

impl KsmoothModel {
    pub fn fit(x: &Rows, y: &[f64], k: usize, gamma: f64) -> KsmoothModel {
        let n = x.len().min(y.len());
        let p = if x.is_empty() { 0 } else { x[0].len() };
        let (means, scales) = std_fit(x);
        let mut y_mean = 0.0;
        for &v in &y[..n] {
            y_mean += v;
        }
        y_mean /= n.max(1) as f64;
        if n < MIN_N || p == 0 {
            return KsmoothModel {
                means,
                scales,
                z: Vec::new(),
                ys: Vec::new(),
                k,
                gamma,
                y_mean,
            };
        }
        let z: Rows = x[..n]
            .iter()
            .map(|row| (0..p).map(|j| (row[j] - means[j]) / scales[j]).collect())
            .collect();
        let ys: Vec<f64> = y[..n].to_vec();
        KsmoothModel {
            means,
            scales,
            z,
            ys,
            k,
            gamma,
            y_mean,
        }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        let p = self.means.len();
        if self.z.is_empty() {
            return x.iter().map(|_| self.y_mean.clamp(-60.0, 60.0)).collect();
        }
        let pf = p as f64;
        let kk = self.k.min(self.z.len());
        // Only the `kk` nearest rows (and the overall nearest, `base`) are ever read, so select
        // them in O(n) and sort just that prefix instead of sorting all n training rows per
        // query. The comparator (distance, then row index) is a strict total order, so the
        // selected prefix and its order are exactly the full sort's first `kk` entries; the
        // weighted sums below run over the same rows in the same order. The three scratch
        // buffers are reused across query rows.
        let take = kk.max(1);
        let mut zr = Vec::with_capacity(p);
        let mut d2s = Vec::with_capacity(self.z.len());
        let mut order: Vec<usize> = Vec::with_capacity(self.z.len());
        let mut out = Vec::with_capacity(x.len());
        for row in x {
            zr.clear();
            zr.extend((0..p.min(row.len())).map(|j| (row[j] - self.means[j]) / self.scales[j]));
            d2s.clear();
            for zb in &self.z {
                let mut d2 = 0.0;
                for d in 0..zr.len().min(zb.len()) {
                    let t = zr[d] - zb[d];
                    d2 += t * t;
                }
                d2s.push(d2);
            }
            order.clear();
            order.extend(0..self.z.len());
            let nearest = |a: &usize, b: &usize| d2s[*a].total_cmp(&d2s[*b]).then(a.cmp(b));
            order.select_nth_unstable_by(take - 1, nearest);
            order[..take].sort_by(nearest);
            let base = d2s[order[0]];
            let mut num = 0.0;
            let mut den = 0.0;
            for &i in order.iter().take(kk) {
                let w = (-self.gamma * (d2s[i] - base) / pf).exp();
                num += w * self.ys[i];
                den += w;
            }
            out.push((num / den).clamp(-60.0, 60.0));
        }
        out
    }
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
    fn ksmooth_is_deterministic_and_learns_toy() {
        let (x, y) = toy();
        let m1 = KsmoothModel::fit(&x, &y, DEFAULT_K, DEFAULT_GAMMA);
        let m2 = KsmoothModel::fit(&x, &y, DEFAULT_K, DEFAULT_GAMMA);
        assert_eq!(m1.predict(&x), m2.predict(&x));
        let rep = cv_r2(
            &GenomeIr::parse("(pipe X (ksmooth))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        println!("smoke[toy] ksmooth oof_r2={}", rep.oof_r2);
        assert!(rep.oof_r2 > 0.7, "ksmooth oof_r2={}", rep.oof_r2);
    }

    #[test]
    fn ksmooth_params_flow_through_the_genome() {
        let (x, y) = toy();
        let sharp = cv_r2(
            &GenomeIr::parse("(pipe X (ksmooth k=8 gamma=4.0))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        let broad = cv_r2(
            &GenomeIr::parse("(pipe X (ksmooth k=200 gamma=0.001))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        // A near-uniform 200-neighbor average must be visibly worse than a
        // local sharp smoother on this smooth signal.
        assert!(
            sharp.oof_r2 > broad.oof_r2,
            "sharp={} broad={}",
            sharp.oof_r2,
            broad.oof_r2
        );
    }

    #[test]
    fn ksmooth_small_n_falls_back_to_mean() {
        let x: Rows = (0..6).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let m = KsmoothModel::fit(&x, &y, DEFAULT_K, DEFAULT_GAMMA);
        assert!(m.z.is_empty());
        assert!(m.predict(&x).iter().all(|v| (*v - 2.5).abs() < 1e-12));
    }
}
