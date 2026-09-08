//! Random Fourier Features + ridge / logistic (Rahimi–Recht). `φ(x)_j = √(2/m)·cos(ω_j·x + b_j)`,
//! `ω_j ~ N(0, 2γI)`, `b_j ~ U[0,2π)`: an explicit feature map whose inner product approximates the RBF
//! kernel `exp(-γ‖x−x'‖²)`. The smooth-kernel inductive bias closest to tabular foundation models' small-n strength, and the
//! cheapest GP-posterior-mean approximation we have (kernel-ridge = GP posterior mean). Reuses `linalg::spd_solve`.
use super::GlmFamily;
use crate::backend::{Backend, BackendKind, Model, TrainSpec};
use crate::linalg::spd_solve;
use crate::preprocess::StandardScaler;
use crate::rng::seeded;
use rand::Rng;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct RffCfg {
    pub m: usize,            // number of random features
    pub gamma: f64,          // RBF bandwidth
    pub lambda: f64,         // ridge strength
    pub newton_iters: usize, // logistic IRLS reweightings
    pub seed: u64,
}
impl Default for RffCfg {
    fn default() -> Self {
        RffCfg {
            m: 256,
            gamma: 0.1,
            lambda: 1.0,
            newton_iters: 8,
            seed: 20260616,
        }
    }
}

pub struct RffBackend {
    pub family: GlmFamily,
    pub cfg: RffCfg,
}
impl RffBackend {
    pub fn logistic() -> Self {
        RffBackend {
            family: GlmFamily::Logistic,
            cfg: RffCfg::default(),
        }
    }
    pub fn gaussian() -> Self {
        RffBackend {
            family: GlmFamily::Gaussian,
            cfg: RffCfg::default(),
        }
    }
}

struct RffModel {
    scaler: StandardScaler,
    w: Vec<f32>,          // m×p row-major (ω_j)
    b: Vec<f32>,          // length m
    heads: Vec<Vec<f64>>, // β per output column, each length m
    ybar: Vec<f64>,       // per-head target mean (Gaussian intercept; 0 for Logistic)
    scale: f32,           // √(2/m)
    family: GlmFamily,
    m: usize,
}

fn std_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Φ (n×m row-major) for standardized column-major `xs` (n×p).
fn featurize(
    xs: &[f32],
    n: usize,
    p: usize,
    w: &[f32],
    b: &[f32],
    m: usize,
    scale: f32,
) -> Vec<f32> {
    let mut phi = vec![0f32; n * m];
    phi.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
        for (j, slot) in row.iter_mut().enumerate() {
            let mut z = b[j] as f64;
            for d in 0..p {
                z += w[j * p + d] as f64 * xs[d * n + i] as f64;
            }
            *slot = scale * z.cos() as f32;
        }
    });
    phi
}

/// Gaussian ridge: solve `(ΦᵀΦ + λI) β = Φᵀy`.
fn fit_ridge(phi: &[f32], n: usize, m: usize, y: &[f32], lambda: f64) -> Vec<f64> {
    let mut g = vec![0f64; m * m];
    let mut c = vec![0f64; m];
    for i in 0..n {
        let row = &phi[i * m..i * m + m];
        let yi = y[i] as f64;
        for a in 0..m {
            let ra = row[a] as f64;
            c[a] += ra * yi;
            let ga = &mut g[a * m..a * m + m];
            for (b, &rb) in row.iter().enumerate().skip(a) {
                ga[b] += ra * rb as f64;
            }
        }
    }
    for a in 0..m {
        for b in (a + 1)..m {
            g[b * m + a] = g[a * m + b];
        }
    }
    // Trace-relative ridge: Φ=√(2/m)·cos gives a Gram diagonal ~ n/m, so an ABSOLUTE λ shrinks by λ·m/n —
    // far too hard at small n (keeps ~24% of signal at n=82 vs ~71% at n=614). Scaling λ by the mean Gram
    // diagonal makes the effective shrinkage scale-invariant, relieving the small-n over-shrink.
    let tr = (0..m).map(|a| g[a * m + a]).sum::<f64>() / m as f64;
    if !spd_solve(&mut g, m, &mut c, lambda * tr.max(1e-12)) {
        return vec![0.0; m];
    }
    c
}

/// Logistic ridge via IRLS: each step solves `(ΦᵀWΦ + λI) β = ΦᵀWz`.
fn fit_logistic_ridge(
    phi: &[f32],
    n: usize,
    m: usize,
    y: &[f32],
    lambda: f64,
    iters: usize,
) -> Vec<f64> {
    let mut beta = vec![0f64; m];
    for _ in 0..iters {
        let mut a = vec![0f64; m * m];
        let mut rhs = vec![0f64; m];
        for i in 0..n {
            let row = &phi[i * m..i * m + m];
            let eta: f64 = row.iter().zip(&beta).map(|(&r, &b)| r as f64 * b).sum();
            let pi = 1.0 / (1.0 + (-eta).exp());
            let wi = (pi * (1.0 - pi)).max(1e-6);
            let zi = eta + (y[i] as f64 - pi) / wi;
            for aa in 0..m {
                let raw = wi * row[aa] as f64;
                rhs[aa] += raw * zi;
                let arow = &mut a[aa * m..aa * m + m];
                for (bb, &rb) in row.iter().enumerate().skip(aa) {
                    arow[bb] += raw * rb as f64;
                }
            }
        }
        for aa in 0..m {
            for bb in (aa + 1)..m {
                a[bb * m + aa] = a[aa * m + bb];
            }
        }
        if !spd_solve(&mut a, m, &mut rhs, lambda) {
            break;
        }
        beta = rhs;
    }
    beta
}

impl RffBackend {
    fn sample_proj(&self, p: usize) -> (Vec<f32>, Vec<f32>, f32) {
        let m = self.cfg.m;
        let mut rng = seeded(self.cfg.seed ^ (p as u64));
        // Dimension-aware bandwidth: with standardized features E‖x−x'‖² ≈ 2p, so a fixed γ makes the RBF
        // kernel exp(−γ‖x−x'‖²) saturate to ~identity (dead kernel) as p grows. Scaling ω's variance by 1/p
        // keeps the effective kernel exp(−2γ) independent of p, restoring the nonlinear capacity.
        let std = (2.0 * self.cfg.gamma / p.max(1) as f64).sqrt();
        let mut w = vec![0f32; m * p];
        let mut b = vec![0f32; m];
        for j in 0..m {
            for d in 0..p {
                w[j * p + d] = (std_normal(&mut rng) * std) as f32;
            }
            b[j] = (rng.gen::<f64>() * 2.0 * std::f64::consts::PI) as f32;
        }
        (w, b, (2.0 / m as f32).sqrt())
    }
    /// Returns (β, ȳ). Gaussian heads fit on the CENTERED target and carry ȳ as an intercept added back at
    /// predict — without it the kernel-ridge predictor decays to 0 off-support instead of to ȳ. Logistic
    /// heads carry ȳ=0 (the sigmoid handles the base rate via its own optimization).
    fn fit_head(&self, phi: &[f32], n: usize, m: usize, y: &[f32]) -> (Vec<f64>, f64) {
        match self.family {
            GlmFamily::Gaussian => {
                let yb = y.iter().map(|&v| v as f64).sum::<f64>() / n.max(1) as f64;
                let yc: Vec<f32> = y.iter().map(|&v| (v as f64 - yb) as f32).collect();
                (fit_ridge(phi, n, m, &yc, self.cfg.lambda), yb)
            }
            GlmFamily::Logistic => (
                fit_logistic_ridge(phi, n, m, y, self.cfg.lambda, self.cfg.newton_iters),
                0.0,
            ),
        }
    }
}

impl Backend for RffBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Rff
    }
    fn train_single(
        &self,
        x: &[f32],
        n: usize,
        p: usize,
        y: &[f32],
        _s: &TrainSpec,
    ) -> Box<dyn Model> {
        let scaler = StandardScaler::fit(x, n, p);
        let xs = scaler.apply_col_major(x, n, p);
        let (w, b, scale) = self.sample_proj(p);
        let phi = featurize(&xs, n, p, &w, &b, self.cfg.m, scale);
        let (head, yb) = self.fit_head(&phi, n, self.cfg.m, y);
        Box::new(RffModel {
            scaler,
            w,
            b,
            heads: vec![head],
            ybar: vec![yb],
            scale,
            family: self.family,
            m: self.cfg.m,
        })
    }
    fn train_multi(
        &self,
        x: &[f32],
        n: usize,
        p: usize,
        y_multi: &[f32],
        k: usize,
        _s: &TrainSpec,
    ) -> Box<dyn Model> {
        let scaler = StandardScaler::fit(x, n, p);
        let xs = scaler.apply_col_major(x, n, p);
        let (w, b, scale) = self.sample_proj(p);
        let phi = featurize(&xs, n, p, &w, &b, self.cfg.m, scale);
        let (heads, ybar): (Vec<_>, Vec<_>) = (0..k)
            .map(|t| self.fit_head(&phi, n, self.cfg.m, &y_multi[t * n..(t + 1) * n]))
            .unzip();
        Box::new(RffModel {
            scaler,
            w,
            b,
            heads,
            ybar,
            scale,
            family: self.family,
            m: self.cfg.m,
        })
    }
}

impl Model for RffModel {
    fn output_dim(&self) -> usize {
        self.heads.len()
    }
    fn predict(&self, x: &[f32], rows: usize, p: usize, proba: bool) -> Vec<f32> {
        let xs = self.scaler.apply_col_major(x, rows, p);
        let phi = featurize(&xs, rows, p, &self.w, &self.b, self.m, self.scale);
        let k = self.heads.len();
        let mut out = vec![0f32; rows * k];
        for (h, beta) in self.heads.iter().enumerate() {
            let yb = self.ybar[h];
            for i in 0..rows {
                let acc: f64 = phi[i * self.m..i * self.m + self.m]
                    .iter()
                    .zip(beta)
                    .map(|(&r, &b)| r as f64 * b)
                    .sum();
                out[i * k + h] = match (self.family, proba) {
                    (GlmFamily::Logistic, true) => (1.0 / (1.0 + (-acc).exp())) as f32,
                    (GlmFamily::Logistic, false) => acc as f32,
                    (GlmFamily::Gaussian, _) => (acc + yb) as f32, // add back the centered intercept
                };
            }
        }
        out
    }
    fn importance(&self, p: usize) -> Vec<f32> {
        vec![1.0 / p.max(1) as f32; p] // random features have no per-input attribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TrainSpec;
    use crate::hashing::mix64;
    use crate::metrics::auc;

    fn pr(i: usize, s: u64) -> f32 {
        (mix64(i as u64 ^ s) % 10_000) as f32 / 10_000.0 * 2.0 - 1.0
    }

    #[test]
    fn rff_learns_nonlinear_circle() {
        // y = 1 iff x0²+x1² > 0.66 (a circle) — a NONLINEAR boundary a linear GLM cannot separate but the
        // RBF/RFF kernel can. This is exactly the bias diversity we want vs the tree/linear members.
        let (n, p) = (600, 2);
        let mut x = vec![0f32; n * p];
        for j in 0..p {
            for i in 0..n {
                x[j * n + i] = pr(i * p + j, 13);
            }
        }
        let y: Vec<f32> = (0..n)
            .map(|i| {
                if x[i] * x[i] + x[n + i] * x[n + i] > 0.66 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let mut be = RffBackend::logistic();
        be.cfg.gamma = 1.0;
        be.cfg.lambda = 0.5;
        let m = be.train_single(&x, n, p, &y, &TrainSpec::from_basic_fields(0, 0, 0.0, 1, 0));
        let pred = m.predict(&x, n, p, true);
        assert!(auc(&y, &pred) > 0.9, "RFF AUC on circle {}", auc(&y, &pred));
        assert!(pred.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
}
