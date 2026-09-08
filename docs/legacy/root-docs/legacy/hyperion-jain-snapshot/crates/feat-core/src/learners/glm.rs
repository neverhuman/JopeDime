//! Elastic-net GLM (logistic + linear) via glmnet-style coordinate descent (Gaussian) / IRLS-weighted CD
//! (logistic). A linear/additive inductive bias the tree ensemble cannot reproduce — the single most
//! reliably-decorrelated member, and a strong learner on wide / small-n problems.
use super::GlmFamily;
use crate::backend::{Backend, BackendKind, Model, TrainSpec};
use crate::preprocess::StandardScaler;

#[derive(Clone, Copy, Debug)]
pub struct GlmCfg {
    pub alpha: f64,       // elastic-net mix: 1 = lasso, 0 = ridge
    pub lambda: f64,      // penalty strength
    pub max_outer: usize, // IRLS reweightings (logistic) or robust reweightings (gaussian)
    pub max_inner: usize, // CD passes per reweighting
    pub tol: f64,
    pub huber_k: f64, // Huber tuning constant (0 = disabled)
}
impl Default for GlmCfg {
    fn default() -> Self {
        GlmCfg {
            alpha: 0.5,
            lambda: 0.01,
            max_outer: 15,
            max_inner: 5,
            tol: 1e-5,
            huber_k: 0.0,
        }
    }
}

pub struct GlmBackend {
    pub family: GlmFamily,
    pub cfg: GlmCfg,
}
impl GlmBackend {
    pub fn logistic() -> Self {
        GlmBackend {
            family: GlmFamily::Logistic,
            cfg: GlmCfg::default(),
        }
    }
    pub fn gaussian() -> Self {
        GlmBackend {
            family: GlmFamily::Gaussian,
            cfg: GlmCfg::default(),
        }
    }
}

struct GlmModel {
    scaler: StandardScaler,
    heads: Vec<(f64, Vec<f64>)>, // (beta0, beta) per output column
    family: GlmFamily,
}

#[inline]
fn soft(z: f64, g: f64) -> f64 {
    if z > g {
        z - g
    } else if z < -g {
        z + g
    } else {
        0.0
    }
}

/// Fit one elastic-net GLM head on standardized features `xs` (column-major n×p) for target `y`.
fn fit_head(
    xs: &[f32],
    n: usize,
    p: usize,
    y: &[f32],
    family: GlmFamily,
    cfg: &GlmCfg,
) -> (f64, Vec<f64>) {
    let nf = n as f64;
    let mut beta = vec![0.0f64; p];
    let mut q = vec![0.0f64; n]; // running linear predictor η

    match family {
        GlmFamily::Gaussian => {
            let mut beta0 = y.iter().map(|&v| v as f64).sum::<f64>() / nf;
            q.iter_mut().for_each(|v| *v = beta0);
            // IRLS loop with Huber reweighting
            for _ in 0..cfg.max_outer {
                // Compute Huber weights based on residuals
                let residuals: Vec<f64> = (0..n).map(|i| (y[i] as f64 - q[i]).abs()).collect();
                let k: f64 = residuals.iter().copied().fold(0.0_f64, |a, b| a.max(b)) * 0.5; // MAD-like scale
                let k = if k > 1e-8 { k } else { 1.0 };
                let w: Vec<f64> = residuals
                    .iter()
                    .map(|&r| if r >= k { k / r } else { 1.0 })
                    .collect();

                let mut maxd = 0.0f64;
                for j in 0..p {
                    let col = &xs[j * n..j * n + n];
                    let (mut sxr, mut sxx) = (0.0, 0.0);
                    for i in 0..n {
                        let xij = col[i] as f64;
                        sxr += w[i] * xij * (y[i] as f64 - q[i]);
                        sxx += w[i] * xij * xij;
                    }
                    let num = sxr / nf + (sxx / nf) * beta[j];
                    let denom = sxx / nf + cfg.lambda * (1.0 - cfg.alpha);
                    let bj = soft(num, cfg.lambda * cfg.alpha) / denom;
                    let d = bj - beta[j];
                    if d != 0.0 {
                        for i in 0..n {
                            q[i] += col[i] as f64 * d;
                        }
                        beta[j] = bj;
                        maxd = maxd.max(d.abs());
                    }
                }
                let (mut sw, mut swr) = (0.0, 0.0);
                for i in 0..n {
                    sw += w[i];
                    swr += w[i] * (y[i] as f64 - q[i]);
                }
                let d0 = swr / sw;
                beta0 += d0;
                q.iter_mut().for_each(|v| *v += d0);
                if maxd < cfg.tol {
                    break;
                }
            }
            (beta0, beta)
        }
        GlmFamily::Logistic => {
            let mp = (y.iter().filter(|&&v| v > 0.5).count() as f64 / nf).clamp(1e-4, 1.0 - 1e-4);
            let mut beta0 = (mp / (1.0 - mp)).ln();
            q.iter_mut().for_each(|v| *v = beta0);
            for _ in 0..cfg.max_outer {
                // quadratic approximation around current η: weights w, working response z
                let mut w = vec![0.0f64; n];
                let mut z = vec![0.0f64; n];
                for i in 0..n {
                    let pi = 1.0 / (1.0 + (-q[i]).exp());
                    let wi = (pi * (1.0 - pi)).max(1e-5);
                    w[i] = wi;
                    z[i] = q[i] + (y[i] as f64 - pi) / wi;
                }
                let mut maxd = 0.0f64;
                for _ in 0..cfg.max_inner {
                    for j in 0..p {
                        let col = &xs[j * n..j * n + n];
                        let (mut swxr, mut swxx) = (0.0, 0.0);
                        for i in 0..n {
                            let xij = col[i] as f64;
                            swxr += w[i] * xij * (z[i] - q[i]);
                            swxx += w[i] * xij * xij;
                        }
                        let num = swxr / nf + (swxx / nf) * beta[j];
                        let denom = swxx / nf + cfg.lambda * (1.0 - cfg.alpha);
                        let bj = soft(num, cfg.lambda * cfg.alpha) / denom;
                        let d = bj - beta[j];
                        if d != 0.0 {
                            for i in 0..n {
                                q[i] += col[i] as f64 * d;
                            }
                            beta[j] = bj;
                            maxd = maxd.max(d.abs());
                        }
                    }
                    let (mut sw, mut swr) = (0.0, 0.0);
                    for i in 0..n {
                        sw += w[i];
                        swr += w[i] * (z[i] - q[i]);
                    }
                    let d0 = swr / sw;
                    beta0 += d0;
                    q.iter_mut().for_each(|v| *v += d0);
                }
                if maxd < cfg.tol {
                    break;
                }
            }
            (beta0, beta)
        }
    }
}

impl Backend for GlmBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Glm
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
        let head = fit_head(&xs, n, p, y, self.family, &self.cfg);
        Box::new(GlmModel {
            scaler,
            heads: vec![head],
            family: self.family,
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
        let heads = (0..k)
            .map(|t| {
                fit_head(
                    &xs,
                    n,
                    p,
                    &y_multi[t * n..(t + 1) * n],
                    self.family,
                    &self.cfg,
                )
            })
            .collect();
        Box::new(GlmModel {
            scaler,
            heads,
            family: self.family,
        })
    }
}

impl Model for GlmModel {
    fn output_dim(&self) -> usize {
        self.heads.len()
    }
    fn predict(&self, x: &[f32], rows: usize, p: usize, proba: bool) -> Vec<f32> {
        let xs = self.scaler.apply_col_major(x, rows, p);
        let k = self.heads.len();
        let mut out = vec![0f32; rows * k];
        for (h, (b0, beta)) in self.heads.iter().enumerate() {
            for i in 0..rows {
                let mut eta = *b0;
                for j in 0..p.min(beta.len()) {
                    eta += beta[j] * xs[j * rows + i] as f64;
                }
                let v = match (self.family, proba) {
                    (GlmFamily::Logistic, true) => 1.0 / (1.0 + (-eta).exp()),
                    _ => eta,
                };
                out[i * k + h] = v as f32;
            }
        }
        out
    }
    fn importance(&self, p: usize) -> Vec<f32> {
        let mut imp = vec![0f32; p];
        for (_, beta) in &self.heads {
            for j in 0..p.min(beta.len()) {
                imp[j] += beta[j].abs() as f32;
            }
        }
        let s: f32 = imp.iter().sum();
        if s > 0.0 {
            imp.iter_mut().for_each(|v| *v /= s);
        }
        imp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TrainSpec;
    use crate::hashing::mix64;
    use crate::metrics::{auc, r2};

    fn pr(i: usize, s: u64) -> f32 {
        (mix64(i as u64 ^ s) % 10_000) as f32 / 10_000.0 * 2.0 - 1.0
    }

    #[test]
    fn logistic_separates_linear_data() {
        let (n, p) = (400, 4);
        let mut x = vec![0f32; n * p];
        for j in 0..p {
            for i in 0..n {
                x[j * n + i] = pr(i * p + j, 7);
            }
        }
        // y = 1 if 2*x0 - x1 > 0
        let y: Vec<f32> = (0..n)
            .map(|i| {
                if 2.0 * x[i] - x[n + i] > 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let m = GlmBackend::logistic().train_single(
            &x,
            n,
            p,
            &y,
            &TrainSpec::from_basic_fields(0, 0, 0.0, 1, 0),
        );
        let pred = m.predict(&x, n, p, true);
        assert!(auc(&y, &pred) > 0.95, "AUC {}", auc(&y, &pred));
        assert!(pred.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn gaussian_fits_linear_target() {
        let (n, p) = (500, 3);
        let mut x = vec![0f32; n * p];
        for j in 0..p {
            for i in 0..n {
                x[j * n + i] = pr(i * p + j, 11);
            }
        }
        let y: Vec<f32> = (0..n)
            .map(|i| 1.5 * x[i] - 0.8 * x[n + i] + 0.3 * x[2 * n + i])
            .collect();
        let mut be = GlmBackend::gaussian();
        be.cfg.lambda = 1e-4; // light penalty -> near-OLS
        let m = be.train_single(&x, n, p, &y, &TrainSpec::from_basic_fields(0, 0, 0.0, 1, 0));
        let pred = m.predict(&x, n, p, false);
        assert!(r2(&y, &pred) > 0.98, "R2 {}", r2(&y, &pred));
    }
}
