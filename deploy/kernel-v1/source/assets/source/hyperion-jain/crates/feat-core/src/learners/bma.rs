//! Stage-0 small-n learner (no teacher): a Bayesian-model-averaging blend of GP-posterior-mean
//! approximations. The crux: kernel-ridge IS the GP posterior mean, and AVERAGING length-scales ≈
//! marginalizing the GP bandwidth hyperparameter — operationally the same "average many regularized
//! smooth hypotheses" that makes tabular foundation models strong on smooth small-n problems, but 100% Rust and per-dataset.
//! Components = RFF-ridge at several γ (≡ several RBF length-scales) + a linear ridge anchor; they are
//! blended by NON-NEGATIVE least squares (`linalg::nnls_cd`) on internal-CV out-of-fold predictions — a
//! regularized stacker whose non-negativity keeps it off a noisy component. Leak-safe for free: every
//! component fits its own `StandardScaler` inside `train_single`, and the blend weights come only from
//! internal OOF, never the holdout. This is the member that closes tabular foundation models' small-n regression edge.
use super::glm::GlmBackend;
use super::rff::{RffBackend, RffCfg};
use super::GlmFamily;
use crate::backend::{Backend, BackendKind, Model, TrainSpec};
use crate::linalg::{nnls_cd, spd_solve};
use crate::preprocess::StandardScaler;
use crate::rng::seeded;
use rand::seq::SliceRandom;

#[derive(Clone, Debug)]
pub struct BmaCfg {
    pub gammas: Vec<f64>,        // RBF bandwidths ≡ GP length-scales to average over
    pub ridge_lambdas: Vec<f64>, // linear-ridge anchor penalties
    pub rff_m: usize,            // random features per RFF component
    pub rff_lambda: f64,         // ridge penalty inside each RFF component
    pub gp_lengthscales: Vec<f64>, // exact-GP RBF length-scales (un-approximated; tiny-n specialist)
    pub gp_noise: f64,             // GP observation noise σ² (ridge on the kernel diagonal)
    pub gp_max_n: usize,           // exact-GP only below this n (O(n³) solve)
    pub folds: usize,              // internal CV folds for the OOF stack
    pub nnls_iters: usize,
    pub seed: u64,
}
impl Default for BmaCfg {
    fn default() -> Self {
        BmaCfg {
            gammas: vec![0.03, 0.1, 0.3, 1.0],
            ridge_lambdas: vec![1.0, 10.0],
            rff_m: 256,
            rff_lambda: 1.0,
            gp_lengthscales: vec![0.5, 1.0, 2.0],
            gp_noise: 0.05,
            gp_max_n: 2000,
            folds: 5,
            nnls_iters: 300,
            seed: 20260618,
        }
    }
}

pub struct BmaBackend {
    pub family: GlmFamily,
    pub cfg: BmaCfg,
}
impl BmaBackend {
    pub fn gaussian() -> Self {
        BmaBackend {
            family: GlmFamily::Gaussian,
            cfg: BmaCfg::default(),
        }
    }
    pub fn logistic() -> Self {
        BmaBackend {
            family: GlmFamily::Logistic,
            cfg: BmaCfg::default(),
        }
    }
}

/// One library member: an RFF-ridge at a given γ (RBF approximation), an EXACT-GP at a given length-scale
/// (un-approximated posterior mean for tiny-n), or a linear ridge (elastic-net at α=0).
#[derive(Clone, Copy, Debug)]
enum Component {
    Rff {
        gamma: f64,
        m: usize,
        lambda: f64,
        seed: u64,
    },
    ExactGp {
        lengthscale: f64,
        noise: f64,
    },
    Ridge {
        lambda: f64,
    },
}
impl Component {
    fn fit(&self, family: GlmFamily, x: &[f32], n: usize, p: usize, y: &[f32]) -> Box<dyn Model> {
        let s = TrainSpec::from_basic_fields(0, 0, 0.0, 1, 0);
        match *self {
            Component::Rff {
                gamma,
                m,
                lambda,
                seed,
            } => {
                let be = RffBackend {
                    family,
                    cfg: RffCfg {
                        m,
                        gamma,
                        lambda,
                        newton_iters: 8,
                        seed,
                    },
                };
                be.train_single(x, n, p, y, &s)
            }
            Component::ExactGp { lengthscale, noise } => {
                fit_exact_gp(x, n, p, y, lengthscale, noise)
            }
            Component::Ridge { lambda } => {
                let mut be = match family {
                    GlmFamily::Gaussian => GlmBackend::gaussian(),
                    GlmFamily::Logistic => GlmBackend::logistic(),
                };
                be.cfg.alpha = 0.0; // pure L2 ridge (the smooth linear anchor)
                be.cfg.lambda = lambda;
                be.train_single(x, n, p, y, &s)
            }
        }
    }
}

/// Exact Gaussian-process posterior mean with an RBF kernel `k(a,b)=exp(-‖a−b‖²/(2ℓ²))` on standardized
/// features. Solve `(K + σ²I) α = y` once (Cholesky via `spd_solve`); predict `k(x*,X)·α`. This IS the
/// kernel-ridge / GP-posterior-mean the BMA story rests on — un-approximated (no random features), so it is
/// the strongest smooth member exactly where tabular foundation models win: very small n. O(n³), so callers gate by `gp_max_n`.
fn fit_exact_gp(
    x: &[f32],
    n: usize,
    p: usize,
    y: &[f32],
    lengthscale: f64,
    noise: f64,
) -> Box<dyn Model> {
    let scaler = StandardScaler::fit(x, n, p);
    let xs = scaler.apply_col_major(x, n, p); // column-major n×p, standardized
                                              // Dimension-aware length-scale: ‖x−x'‖² grows ∝ p on standardized features, so divide by p to keep the
                                              // off-diagonal kernel exp(−1/ℓ²) independent of dimensionality (else the GP collapses to identity).
    let inv_2l2 = 1.0 / (2.0 * lengthscale * lengthscale * p.max(1) as f64);
    let mut k = vec![0f64; n * n];
    for i in 0..n {
        k[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let mut d2 = 0.0;
            for c in 0..p {
                let diff = (xs[c * n + i] - xs[c * n + j]) as f64;
                d2 += diff * diff;
            }
            let kij = (-d2 * inv_2l2).exp();
            k[i * n + j] = kij;
            k[j * n + i] = kij;
        }
    }
    // Center the target: the GP posterior mean is ȳ + k(x*,X)·α with α = (K+σ²I)⁻¹(y−ȳ). Without this the
    // kernel-only model decays to 0 off-support instead of to ȳ (catastrophic when ȳ is large, e.g. concrete
    // strength ~35), which NNLS then prunes to zero weight — the mechanism that locked BMA onto its anchor.
    let ybar = y.iter().map(|&v| v as f64).sum::<f64>() / n.max(1) as f64;
    let mut alpha: Vec<f64> = y.iter().map(|&v| v as f64 - ybar).collect();
    if !spd_solve(&mut k, n, &mut alpha, noise.max(1e-6)) {
        alpha = vec![0.0; n];
    }
    Box::new(GpModel {
        scaler,
        xtr: xs,
        n,
        alpha,
        inv_2l2,
        ybar,
    })
}

struct GpModel {
    scaler: StandardScaler,
    xtr: Vec<f32>, // standardized training features, column-major n×p
    n: usize,
    alpha: Vec<f64>,
    inv_2l2: f64,
    ybar: f64,
}
impl Model for GpModel {
    fn output_dim(&self) -> usize {
        1
    }
    fn predict(&self, x: &[f32], rows: usize, p: usize, _proba: bool) -> Vec<f32> {
        let xs = self.scaler.apply_col_major(x, rows, p);
        let mut out = vec![0f32; rows];
        for t in 0..rows {
            let mut s = 0.0f64;
            for i in 0..self.n {
                let mut d2 = 0.0;
                for c in 0..p {
                    let diff = (xs[c * rows + t] - self.xtr[c * self.n + i]) as f64;
                    d2 += diff * diff;
                }
                s += (-d2 * self.inv_2l2).exp() * self.alpha[i];
            }
            out[t] = (s + self.ybar) as f32;
        }
        out
    }
    fn importance(&self, p: usize) -> Vec<f32> {
        vec![1.0 / p.max(1) as f32; p]
    }
}

impl BmaBackend {
    fn components(&self, n: usize) -> Vec<Component> {
        let mut v = Vec::new();
        for &g in &self.cfg.gammas {
            v.push(Component::Rff {
                gamma: g,
                m: self.cfg.rff_m,
                lambda: self.cfg.rff_lambda,
                seed: self.cfg.seed ^ g.to_bits(),
            });
        }
        // Exact GP only for small n (O(n³) factor); above gp_max_n the RFF members approximate it cheaply.
        if n <= self.cfg.gp_max_n {
            for &ls in &self.cfg.gp_lengthscales {
                v.push(Component::ExactGp {
                    lengthscale: ls,
                    noise: self.cfg.gp_noise,
                });
            }
        }
        for &l in &self.cfg.ridge_lambdas {
            v.push(Component::Ridge { lambda: l });
        }
        v
    }
    fn proba(&self) -> bool {
        matches!(self.family, GlmFamily::Logistic)
    }
    /// Fit components on full data + NNLS blend weights from internal-CV OOF, for one target column.
    fn fit_blend(&self, x: &[f32], n: usize, p: usize, y: &[f32]) -> TargetBlend {
        let comps = self.components(n);
        let c = comps.len();
        let proba = self.proba();

        // Internal k-fold OOF prediction matrix (column-per-component, length n).
        let mut rng = seeded(self.cfg.seed ^ 0xB1A5);
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);
        let folds = self.cfg.folds.clamp(2, n.max(2));
        let mut oof: Vec<Vec<f32>> = vec![vec![0f32; n]; c];
        for f in 0..folds {
            let val: Vec<usize> = perm.iter().copied().skip(f).step_by(folds).collect();
            if val.is_empty() {
                continue;
            }
            let is_val: Vec<bool> = {
                let mut b = vec![false; n];
                for &i in &val {
                    b[i] = true;
                }
                b
            };
            let tr: Vec<usize> = (0..n).filter(|&i| !is_val[i]).collect();
            if tr.is_empty() {
                continue;
            }
            let xtr = select_rows(x, n, p, &tr);
            let ytr: Vec<f32> = tr.iter().map(|&i| y[i]).collect();
            let xva = select_rows(x, n, p, &val);
            for (ci, comp) in comps.iter().enumerate() {
                let model = comp.fit(self.family, &xtr, tr.len(), p, &ytr);
                let pv = model.predict(&xva, val.len(), p, proba);
                for (k, &i) in val.iter().enumerate() {
                    oof[ci][i] = pv[k];
                }
            }
        }

        // NNLS stack on the OOF columns: min ||OOF·w − y||², w ≥ 0.
        let mut gram = vec![0f64; c * c];
        let mut aty = vec![0f64; c];
        for a in 0..c {
            for i in 0..n {
                aty[a] += oof[a][i] as f64 * y[i] as f64;
            }
            for b in a..c {
                let g: f64 = (0..n).map(|i| oof[a][i] as f64 * oof[b][i] as f64).sum();
                gram[a * c + b] = g;
                gram[b * c + a] = g;
            }
        }
        let mut w = nnls_cd(&gram, &aty, c, self.cfg.nnls_iters);
        // Degenerate guard: if NNLS collapsed to all-zero (e.g. y≈0), fall back to a uniform blend.
        if w.iter().sum::<f64>() <= 1e-9 {
            w = vec![1.0 / c as f64; c];
        }

        // Refit every component on the FULL train for serving.
        let models = comps
            .iter()
            .map(|comp| comp.fit(self.family, x, n, p, y))
            .collect();
        TargetBlend { models, weights: w }
    }
}

/// Per-target trained blend: components + their NNLS weights.
struct TargetBlend {
    models: Vec<Box<dyn Model>>,
    weights: Vec<f64>,
}

struct BmaModel {
    targets: Vec<TargetBlend>,
    proba: bool,
}

impl Backend for BmaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::SmallNBma
    }
    fn train_single(
        &self,
        x: &[f32],
        n: usize,
        p: usize,
        y: &[f32],
        _s: &TrainSpec,
    ) -> Box<dyn Model> {
        Box::new(BmaModel {
            targets: vec![self.fit_blend(x, n, p, y)],
            proba: self.proba(),
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
        let targets = (0..k)
            .map(|t| self.fit_blend(x, n, p, &y_multi[t * n..(t + 1) * n]))
            .collect();
        Box::new(BmaModel {
            targets,
            proba: self.proba(),
        })
    }
}

impl Model for BmaModel {
    fn output_dim(&self) -> usize {
        self.targets.len()
    }
    fn predict(&self, x: &[f32], rows: usize, p: usize, proba: bool) -> Vec<f32> {
        let k = self.targets.len();
        let mut out = vec![0f32; rows * k];
        for (t, blend) in self.targets.iter().enumerate() {
            for (c, model) in blend.models.iter().enumerate() {
                // Each component serves on its own scale: probabilities for logistic, raw for gaussian.
                let pc = model.predict(x, rows, p, self.proba);
                let w = blend.weights[c] as f32;
                if w == 0.0 {
                    continue;
                }
                for i in 0..rows {
                    out[i * k + t] += w * pc[i];
                }
            }
        }
        // For logistic with proba=false the caller wants a logit-like score; the NNLS blend of
        // probabilities is already a monotone score, so AUC is unaffected. Clamp to [0,1] when probs asked.
        if proba && self.proba {
            for v in &mut out {
                *v = v.clamp(0.0, 1.0);
            }
        }
        out
    }
    fn importance(&self, p: usize) -> Vec<f32> {
        vec![1.0 / p.max(1) as f32; p]
    }
}

/// Extract `idx` rows from a column-major (n×p) matrix into a column-major (|idx|×p) matrix.
fn select_rows(x: &[f32], n: usize, p: usize, idx: &[usize]) -> Vec<f32> {
    let m = idx.len();
    let mut out = vec![0f32; m * p];
    for col in 0..p {
        let src = &x[col * n..col * n + n];
        let dst = &mut out[col * m..col * m + m];
        for (k, &i) in idx.iter().enumerate() {
            dst[k] = src[i];
        }
    }
    out
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
    fn bma_fits_smooth_nonlinear_regression() {
        // y = sin(3·x0)·cos(2·x1) + 0.3·x2 — a smooth multi-scale surface where averaging RBF
        // length-scales should beat any single one. A linear model alone cannot fit this.
        let (n, p) = (500, 3);
        let mut x = vec![0f32; n * p];
        for j in 0..p {
            for i in 0..n {
                x[j * n + i] = pr(i * p + j, 31);
            }
        }
        let y: Vec<f32> = (0..n)
            .map(|i| (3.0 * x[i]).sin() * (2.0 * x[n + i]).cos() + 0.3 * x[2 * n + i])
            .collect();
        let m = BmaBackend::gaussian().train_single(
            &x,
            n,
            p,
            &y,
            &TrainSpec::from_basic_fields(0, 0, 0.0, 1, 0),
        );
        let pred = m.predict(&x, n, p, false);
        assert!(r2(&y, &pred) > 0.9, "BMA train R2 {}", r2(&y, &pred));
        assert_eq!(m.output_dim(), 1);
    }

    #[test]
    fn bma_separates_nonlinear_classification() {
        // Concentric-circle boundary: needs the RBF members, not the linear anchor.
        let (n, p) = (600, 2);
        let mut x = vec![0f32; n * p];
        for j in 0..p {
            for i in 0..n {
                x[j * n + i] = pr(i * p + j, 17);
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
        let m = BmaBackend::logistic().train_single(
            &x,
            n,
            p,
            &y,
            &TrainSpec::from_basic_fields(0, 0, 0.0, 1, 0),
        );
        let pred = m.predict(&x, n, p, true);
        assert!(
            auc(&y, &pred) > 0.85,
            "BMA AUC on circle {}",
            auc(&y, &pred)
        );
        assert!(
            pred.iter().all(|&v| (0.0..=1.0).contains(&v)),
            "probs in range"
        );
    }
}
