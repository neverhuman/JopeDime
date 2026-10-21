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
        // Scratch belongs to this prediction call, not to a global pool or the model. Reuse it
        // across query rows, so concurrent callers remain independent without per-row allocation.
        let mut zr = Vec::with_capacity(p);
        let mut d2s = Vec::with_capacity(self.z.len());
        let mut order = Vec::with_capacity(self.z.len());
        x.iter()
            .map(|row| {
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
                let compare = |&a: &usize, &b: &usize| d2s[a].total_cmp(&d2s[b]).then(a.cmp(&b));
                // Select the same k rows as the full sort, then sort only that prefix. The row
                // index makes this a total order even for ties; accumulation order stays exact.
                if kk > 0 && kk < order.len() {
                    order.select_nth_unstable_by(kk, compare);
                }
                order[..kk].sort_unstable_by(compare);
                let base = d2s[order[0]];
                let mut num = 0.0;
                let mut den = 0.0;
                for &i in order.iter().take(kk) {
                    let w = (-self.gamma * (d2s[i] - base) / pf).exp();
                    num += w * self.ys[i];
                    den += w;
                }
                (num / den).clamp(-60.0, 60.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cv_r2;
    use crate::rng::Pcg32;
    use crate::GenomeIr;

    fn full_sort_reference(model: &KsmoothModel, x: &Rows) -> Vec<f64> {
        let p = model.means.len();
        if model.z.is_empty() {
            return x.iter().map(|_| model.y_mean.clamp(-60.0, 60.0)).collect();
        }
        let pf = p as f64;
        let kk = model.k.min(model.z.len());
        x.iter()
            .map(|row| {
                let zr: Vec<f64> = (0..p.min(row.len()))
                    .map(|j| (row[j] - model.means[j]) / model.scales[j])
                    .collect();
                let mut d2s = Vec::with_capacity(model.z.len());
                for zb in &model.z {
                    let mut d2 = 0.0;
                    for d in 0..zr.len().min(zb.len()) {
                        let t = zr[d] - zb[d];
                        d2 += t * t;
                    }
                    d2s.push(d2);
                }
                let mut order: Vec<usize> = (0..model.z.len()).collect();
                order.sort_by(|&a, &b| d2s[a].total_cmp(&d2s[b]).then(a.cmp(&b)));
                let base = d2s[order[0]];
                let mut num = 0.0;
                let mut den = 0.0;
                for &i in order.iter().take(kk) {
                    let w = (-model.gamma * (d2s[i] - base) / pf).exp();
                    num += w * model.ys[i];
                    den += w;
                }
                (num / den).clamp(-60.0, 60.0)
            })
            .collect()
    }

    #[test]
    fn partial_selection_predict_is_bit_identical_to_full_sort() {
        let (x, y) = toy();
        // Coarse grids create many exact distance ties, exercising the index tie-break.
        let coarse: Rows = x
            .iter()
            .map(|r| r.iter().map(|v| (v * 4.0).round() / 4.0).collect())
            .collect();
        for rows in [&x, &coarse] {
            for k in [0, 1, 2, 7, 32, 239, 240, 512] {
                let mut m = KsmoothModel::fit(rows, &y, k.max(1), DEFAULT_GAMMA);
                m.k = k; // a deserialized model may carry k = 0
                let fast = m.predict(rows);
                let reference = full_sort_reference(&m, rows);
                let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&fast), bits(&reference), "k = {k}");
            }
        }
    }

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
    fn partial_neighbors_match_full_sort_bits_including_ties_and_short_queries() {
        let (mut x, mut y) = toy();
        // Equal distances with different targets expose cutoff and accumulation-order errors.
        x.extend(vec![vec![0.5, 0.5, 0.5]; 80]);
        y.extend((0..80).map(|i| i as f64 - 40.0));
        let mut queries = x.clone();
        queries.extend([vec![], vec![0.5], vec![0.5, 0.5], vec![0.5; 4]]);
        queries.extend([vec![f64::NAN; 3], vec![f64::INFINITY; 3]]);
        for k in [0, 1, 8, 32, 79, 80, x.len() - 1, x.len(), x.len() + 1] {
            for gamma in [1e-6, 1.0, 1e6] {
                let model = KsmoothModel::fit(&x, &y, k, gamma);
                let expected = full_sort_reference(&model, &queries);
                let actual = model.predict(&queries);
                assert_eq!(actual.len(), expected.len());
                for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "k={k} gamma={gamma} row={i}");
                }
                assert!(model.predict(&vec![]).is_empty());
            }
        }
        let model = KsmoothModel::fit(&x, &y, 32, 1.0);
        let expected = model.predict(&queries);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let actual = model.predict(&queries);
                    assert!(actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()));
                });
            }
        });
    }

    #[test]
    #[ignore = "manual CPU microbenchmark; prints timings without a speed threshold"]
    fn ksmooth_prediction_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = Pcg32::new(42, 7);
        let x: Rows = (0..20_000)
            .map(|_| (0..8).map(|_| rng.next_f64()).collect())
            .collect();
        let y: Vec<f64> = x.iter().map(|r| r[0] - r[1]).collect();
        let queries: Rows = x.iter().step_by(79).cloned().collect();
        let model = KsmoothModel::fit(&x, &y, 32, 1.0);
        for repeat in 0..5 {
            // Alternate order to reduce systematic warmup bias.
            let mut timings = [0.0; 2];
            let mut outputs = [Vec::new(), Vec::new()];
            for which in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                outputs[which] = if which == 0 {
                    full_sort_reference(black_box(&model), black_box(&queries))
                } else {
                    model.predict(black_box(&queries))
                };
                timings[which] = start.elapsed().as_secs_f64();
            }
            assert_eq!(outputs[0], outputs[1]);
            println!("ksmooth n={} q={} p=8 k=32 repeat={repeat} full_sort_seconds={:.6} partial_seconds={:.6} speedup={:.3}",
                x.len(), queries.len(), timings[0], timings[1], timings[0]/timings[1]);
        }
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
