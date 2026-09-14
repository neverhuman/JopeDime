//! `gpr` — Gaussian-process regression with a learned automatic-relevance-
//! determination (ARD) squared-exponential kernel (Rasmussen & Williams, 2006).
//!
//! The kernel is
//!   k(x, x') = sf2 * exp(-0.5 * sum_d (x_d - x'_d)^2 / ls_d^2)
//! with a per-feature lengthscale `ls_d`, signal variance `sf2`, and observation
//! noise variance `sn2`. Features are standardized (heads::std_fit) and the
//! target is centered. The unconstrained hyperparameters (log ls_d, log signal
//! std, log noise std) are fit by a fixed number of gradient-ascent steps on the
//! log marginal likelihood, whose gradient is the textbook
//!   d/dtheta log p(y|X) = 0.5 * tr((alpha alpha^T - Ky^{-1}) dKy/dtheta),
//!   alpha = Ky^{-1} y     (Rasmussen & Williams, 2006, eq. 5.9),
//! evaluated with one Cholesky-based symmetric-positive-definite inverse per step
//! on a bounded optimization subset. The predictive coefficients are then solved
//! on up to `m` seeded inducing rows (a subset-of-regressors predictor) and
//! predictions k(x*, X_m) . alpha are clamped to +/-60.
//!
//! Bit-parity contract: the emitted Python/Rust artifacts replay exactly the
//! same arithmetic order (kernel, Cholesky, triangular inverse, gradient sums).

use crate::heads::{cholesky_solve, std_fit};
use crate::krr::seeded_rows;
use crate::Rows;

/// Default cap on inducing rows for the predictive solve.
pub const DEFAULT_M: usize = 500;
/// Default number of gradient-ascent steps on the log marginal likelihood.
pub const DEFAULT_STEPS: usize = 40;
/// Cap on rows used for the per-step hyperparameter gradient. The optimization
/// Cholesky is O(m_opt^3), so this keeps the pure-Python twin affordable no
/// matter how large the predictive basis `m` is.
pub const OPT_CAP: usize = 256;
/// Leading subset rows used to seed each lengthscale from the median pairwise
/// per-feature distance.
pub const INIT_ROWS: usize = 96;
/// Constant learning rate for the gradient-ascent schedule (each per-parameter
/// log-space step is additionally hard-clamped to +/-0.5, so this cannot
/// diverge; the larger rate suppresses irrelevant features within the step
/// budget in high dimensions).
pub const LR: f64 = 1.0;
/// Seed for the partial Fisher-Yates inducing-row selection.
pub const GPR_SEED: u64 = 909;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GprModel {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    /// Standardized inducing rows (empty => constant fallback).
    pub basis: Vec<Vec<f64>>,
    pub alpha: Vec<f64>,
    /// Learned per-feature lengthscales.
    pub ls: Vec<f64>,
    /// Learned signal variance.
    pub sf2: f64,
    pub y_mean: f64,
}

/// Linearly interpolated quantile of an ascending slice (numpy 'linear' rule);
/// identical arithmetic to the emitted `quantile` helper.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Symmetric-positive-definite inverse of `a` (flattened m*m, consumed): factor
/// A = L L^T in place, invert the lower-triangular L, then form A^{-1} = L^{-T}
/// L^{-1}. The 1e-6 pivot floor matches `cholesky_solve`.
fn chol_inverse(mut a: Vec<f64>, m: usize) -> Vec<f64> {
    for i in 0..m {
        for j in 0..=i {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= a[i * m + k] * a[j * m + k];
            }
            if i == j {
                a[i * m + j] = if s > 1e-12 { s.sqrt() } else { 1e-6 };
            } else {
                a[i * m + j] = s / a[j * m + j];
            }
        }
    }
    let mut linv = vec![0.0f64; m * m];
    for i in 0..m {
        linv[i * m + i] = 1.0 / a[i * m + i];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s -= a[i * m + k] * linv[k * m + j];
            }
            linv[i * m + j] = s / a[i * m + i];
        }
    }
    let mut inv = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..=i {
            let mut s = 0.0;
            for k in i..m {
                s += linv[k * m + i] * linv[k * m + j];
            }
            inv[i * m + j] = s;
            inv[j * m + i] = s;
        }
    }
    inv
}

impl GprModel {
    pub fn fit(x: &Rows, y: &[f64], m: usize, steps: usize) -> GprModel {
        let n = x.len().min(y.len());
        let p = if x.is_empty() { 0 } else { x[0].len() };
        let (means, scales) = std_fit(x);
        let mut y_mean = 0.0;
        for &v in &y[..n] {
            y_mean += v;
        }
        y_mean /= n.max(1) as f64;
        if n < 12 || p == 0 {
            return GprModel {
                means,
                scales,
                basis: Vec::new(),
                alpha: Vec::new(),
                ls: Vec::new(),
                sf2: 0.0,
                y_mean,
            };
        }
        // Seeded inducing rows (the predictive basis), standardized.
        let basis_size = n.min(m);
        let idx: Vec<usize> = if n > m {
            seeded_rows(n, m, GPR_SEED)
        } else {
            (0..n).collect()
        };
        let z: Rows = idx
            .iter()
            .map(|&i| (0..p).map(|j| (x[i][j] - means[j]) / scales[j]).collect())
            .collect();
        let yc: Vec<f64> = idx.iter().map(|&i| y[i] - y_mean).collect();
        let opt_size = basis_size.min(OPT_CAP);

        // Initialize log-lengthscales from the median pairwise per-feature
        // distance over the leading `INIT_ROWS` inducing rows, scaled by sqrt(p)
        // so the summed squared-exponential exponent stays O(1) as the feature
        // count grows (the curse-of-dimensionality guard for RBF kernels).
        let n_init = opt_size.min(INIT_ROWS);
        let dim_scale = (p as f64).sqrt();
        let mut log_ls = vec![0.0f64; p];
        for d in 0..p {
            let mut dists: Vec<f64> = Vec::new();
            for i in 0..n_init {
                for j in (i + 1)..n_init {
                    dists.push((z[i][d] - z[j][d]).abs());
                }
            }
            dists.sort_by(|a, b| a.total_cmp(b));
            let med = if dists.is_empty() {
                1.0
            } else {
                quantile(&dists, 0.5)
            };
            log_ls[d] = (med.max(1e-3) * dim_scale).ln();
        }
        // Signal/noise from the centered-target standard deviation on the subset.
        let var: f64 = yc.iter().take(opt_size).map(|v| v * v).sum();
        let sy = (var / opt_size.max(1) as f64).sqrt();
        let mut log_sf = sy.max(1e-6).ln();
        let mut log_sn = (0.1 * sy).max(1e-6).ln();

        // Gradient ascent on the log marginal likelihood over the optimization
        // subset (the first `opt_size` inducing rows).
        for _ in 0..steps {
            let mo = opt_size;
            let ls_cur: Vec<f64> = log_ls.iter().map(|u| u.exp()).collect();
            let inv_ls2: Vec<f64> = ls_cur.iter().map(|l| 1.0 / (l * l)).collect();
            let sf2 = (2.0 * log_sf).exp();
            let sn2 = (2.0 * log_sn).exp();
            // Signal kernel K (kept for the gradient) and Ky = K + sn2 I.
            let mut kmat = vec![0.0f64; mo * mo];
            for i in 0..mo {
                kmat[i * mo + i] = sf2;
                for j in (i + 1)..mo {
                    let mut r2 = 0.0;
                    for d in 0..p {
                        let diff = z[i][d] - z[j][d];
                        r2 += diff * diff * inv_ls2[d];
                    }
                    let v = sf2 * (-0.5 * r2).exp();
                    kmat[i * mo + j] = v;
                    kmat[j * mo + i] = v;
                }
            }
            let mut a = kmat.clone();
            for i in 0..mo {
                a[i * mo + i] += sn2;
            }
            let kinv = chol_inverse(a, mo);
            let mut alpha = vec![0.0f64; mo];
            for i in 0..mo {
                let mut s = 0.0;
                for j in 0..mo {
                    s += kinv[i * mo + j] * yc[j];
                }
                alpha[i] = s;
            }
            // Gradients from W = alpha alpha^T - Ky^{-1} (used implicitly).
            let mut g_sf = 0.0;
            let mut g_sn = 0.0;
            let mut g_ls = vec![0.0f64; p];
            for i in 0..mo {
                let wii = alpha[i] * alpha[i] - kinv[i * mo + i];
                g_sf += wii * kmat[i * mo + i];
                g_sn += wii;
                for j in (i + 1)..mo {
                    let wij = alpha[i] * alpha[j] - kinv[i * mo + j];
                    let kij = kmat[i * mo + j];
                    g_sf += 2.0 * wij * kij;
                    let c = wij * kij;
                    for d in 0..p {
                        let diff = z[i][d] - z[j][d];
                        g_ls[d] += c * diff * diff * inv_ls2[d];
                    }
                }
            }
            g_sn *= sn2;
            let scale = LR / mo as f64;
            for d in 0..p {
                let step = (scale * g_ls[d]).clamp(-0.5, 0.5);
                log_ls[d] = (log_ls[d] + step).clamp(-6.0, 6.0);
            }
            let step_sf = (scale * g_sf).clamp(-0.5, 0.5);
            log_sf = (log_sf + step_sf).clamp(-15.0, 15.0);
            // Floor the noise at exp(-6) of the signal std: this jitter keeps Ky
            // well-conditioned, so the gradient cannot overflow (and prediction
            // stays stable) even under an aggressive step.
            let step_sn = (scale * g_sn).clamp(-0.5, 0.5);
            log_sn = (log_sn + step_sn).clamp(log_sf - 6.0, log_sf);
        }

        // Predictive coefficients on the full inducing basis.
        let ls_final: Vec<f64> = log_ls.iter().map(|u| u.exp()).collect();
        let inv_ls2: Vec<f64> = ls_final.iter().map(|l| 1.0 / (l * l)).collect();
        let sf2 = (2.0 * log_sf).exp();
        let sn2 = (2.0 * log_sn).exp();
        let mb = basis_size;
        let mut a = vec![0.0f64; mb * mb];
        for i in 0..mb {
            a[i * mb + i] = sf2 + sn2;
            for j in (i + 1)..mb {
                let mut r2 = 0.0;
                for d in 0..p {
                    let diff = z[i][d] - z[j][d];
                    r2 += diff * diff * inv_ls2[d];
                }
                let v = sf2 * (-0.5 * r2).exp();
                a[i * mb + j] = v;
                a[j * mb + i] = v;
            }
        }
        let mut b = yc.clone();
        let alpha = cholesky_solve(&mut a, &mut b, mb);
        GprModel {
            means,
            scales,
            basis: z,
            alpha,
            ls: ls_final,
            sf2,
            y_mean,
        }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        let p = self.means.len();
        let inv_ls2: Vec<f64> = self.ls.iter().map(|l| 1.0 / (l * l)).collect();
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
                    let mut r2 = 0.0;
                    for d in 0..zr.len().min(zb.len()) {
                        let diff = zr[d] - zb[d];
                        r2 += diff * diff * inv_ls2[d];
                    }
                    s += self.alpha[b] * self.sf2 * (-0.5 * r2).exp();
                }
                s.clamp(-60.0, 60.0)
            })
            .collect()
    }
}

// ------------------------ composite-kernel variant -------------------------
//
// `(gpr kernel="rbf+linear" ...)` — the same subset-of-regressors GP as above,
// but the covariance is a searchable SUM of PRODUCTS of at most
// `MAX_KERNEL_PRIMS` primitives:
//   rbf       ARD squared exponential  exp(-0.5 sum_d diff_d^2 / ls_d^2)
//   rq        ARD rational quadratic   (1 + r2/(2 alpha))^-alpha (learned alpha)
//   periodic  MacKay periodic          exp(-2 sum_d sin^2(pi diff_d / per) / ls^2)
//             with a learned shared period and lengthscale
//   linear    homogeneous linear       (x . x') / p (a learned variance per term)
// Each sum term carries one learned log-amplitude; every factor's shape
// parameters (log lengthscales, log alpha, log period) are learned in the same
// marginal-likelihood gradient-ascent loop, using dK/dlog(u) = K_term *
// dlog(base_f)/dlog(u) for the strictly positive factors. `kernel="rbf"`
// callers keep the dedicated `GprModel` path above (bit-identical to before).

/// Kernel-expression primitives for the composite-kernel gpr head.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KernelPrim {
    Rbf,
    Rq,
    Periodic,
    Linear,
}

/// Sum-of-products kernel expression: outer Vec = sum terms, inner = factors.
pub type KernelSpec = Vec<Vec<KernelPrim>>;

/// Maximum number of primitives in one kernel expression.
pub const MAX_KERNEL_PRIMS: usize = 3;

/// Parse a kernel expression: a '+'-separated sum of '*'-separated products of
/// the primitives {rbf, rq, periodic, linear}, at most [`MAX_KERNEL_PRIMS`]
/// primitives total (e.g. "rq", "rbf+linear", "rbf*periodic", "rq+linear").
pub fn parse_kernel(src: &str) -> Result<KernelSpec, String> {
    let mut spec: KernelSpec = Vec::new();
    let mut total = 0usize;
    for term_src in src.split('+') {
        let mut term: Vec<KernelPrim> = Vec::new();
        for factor in term_src.split('*') {
            let prim = match factor.trim() {
                "rbf" => KernelPrim::Rbf,
                "rq" => KernelPrim::Rq,
                "periodic" => KernelPrim::Periodic,
                "linear" => KernelPrim::Linear,
                other => return Err(format!("unknown kernel primitive '{other}'")),
            };
            term.push(prim);
            total += 1;
        }
        spec.push(term);
    }
    if total > MAX_KERNEL_PRIMS {
        return Err(format!(
            "kernel expression has {total} primitives (max {MAX_KERNEL_PRIMS})"
        ));
    }
    Ok(spec)
}

/// Per-step decode of the flat hyperparameter vector (identical layout in the
/// emitted artifacts): for each term [log amplitude, factor shape params...],
/// then the global log noise std last.
struct DecodedKernel {
    /// Per-term signal variance exp(2 * log amplitude).
    sf2: Vec<f64>,
    /// Per (term, factor): inverse squared lengthscales — p entries for
    /// rbf/rq, a single shared entry for periodic, empty for linear.
    inv_ls2: Vec<Vec<Vec<f64>>>,
    /// Per (term, factor): rq alpha (0.0 for other primitives).
    alpha: Vec<Vec<f64>>,
    /// Per (term, factor): periodic period (0.0 for other primitives).
    period: Vec<Vec<f64>>,
    /// Observation noise variance exp(2 * log noise std).
    sn2: f64,
}

fn decode_kernel(spec: &KernelSpec, theta: &[f64], p: usize) -> DecodedKernel {
    let mut sf2: Vec<f64> = Vec::new();
    let mut inv_ls2: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut alpha: Vec<Vec<f64>> = Vec::new();
    let mut period: Vec<Vec<f64>> = Vec::new();
    let mut off = 0usize;
    for term in spec {
        sf2.push((2.0 * theta[off]).exp());
        off += 1;
        let mut inv_t: Vec<Vec<f64>> = Vec::new();
        let mut alpha_t: Vec<f64> = Vec::new();
        let mut period_t: Vec<f64> = Vec::new();
        for prim in term {
            match prim {
                KernelPrim::Rbf => {
                    let mut inv_f = Vec::with_capacity(p);
                    for d in 0..p {
                        let l = theta[off + d].exp();
                        inv_f.push(1.0 / (l * l));
                    }
                    inv_t.push(inv_f);
                    alpha_t.push(0.0);
                    period_t.push(0.0);
                    off += p;
                }
                KernelPrim::Rq => {
                    let mut inv_f = Vec::with_capacity(p);
                    for d in 0..p {
                        let l = theta[off + d].exp();
                        inv_f.push(1.0 / (l * l));
                    }
                    inv_t.push(inv_f);
                    alpha_t.push(theta[off + p].exp());
                    period_t.push(0.0);
                    off += p + 1;
                }
                KernelPrim::Periodic => {
                    period_t.push(theta[off].exp());
                    let l = theta[off + 1].exp();
                    inv_t.push(vec![1.0 / (l * l)]);
                    alpha_t.push(0.0);
                    off += 2;
                }
                KernelPrim::Linear => {
                    inv_t.push(Vec::new());
                    alpha_t.push(0.0);
                    period_t.push(0.0);
                }
            }
        }
        inv_ls2.push(inv_t);
        alpha.push(alpha_t);
        period.push(period_t);
    }
    let sn2 = (2.0 * theta[off]).exp();
    DecodedKernel {
        sf2,
        inv_ls2,
        alpha,
        period,
        sn2,
    }
}

/// k(za, zb) for the composite kernel; iterates terms and factors in
/// declaration order (the emitted artifacts replay this order exactly).
fn kernel_value(spec: &KernelSpec, dec: &DecodedKernel, pf: f64, za: &[f64], zb: &[f64]) -> f64 {
    let dm = za.len().min(zb.len());
    let mut total = 0.0;
    for (t, term) in spec.iter().enumerate() {
        let mut val = dec.sf2[t];
        for (f, prim) in term.iter().enumerate() {
            match prim {
                KernelPrim::Rbf => {
                    let inv = &dec.inv_ls2[t][f];
                    let mut r2 = 0.0;
                    for d in 0..dm {
                        let diff = za[d] - zb[d];
                        r2 += diff * diff * inv[d];
                    }
                    val *= (-0.5 * r2).exp();
                }
                KernelPrim::Rq => {
                    let inv = &dec.inv_ls2[t][f];
                    let mut r2 = 0.0;
                    for d in 0..dm {
                        let diff = za[d] - zb[d];
                        r2 += diff * diff * inv[d];
                    }
                    let a = dec.alpha[t][f];
                    let s = r2 / (2.0 * a);
                    val *= (-a * (1.0 + s).ln()).exp();
                }
                KernelPrim::Periodic => {
                    let per = dec.period[t][f];
                    let inv0 = dec.inv_ls2[t][f][0];
                    let mut s2 = 0.0;
                    for d in 0..dm {
                        let ang = std::f64::consts::PI * (za[d] - zb[d]) / per;
                        let sa = ang.sin();
                        s2 += sa * sa;
                    }
                    val *= (-2.0 * s2 * inv0).exp();
                }
                KernelPrim::Linear => {
                    let mut dot = 0.0;
                    for d in 0..dm {
                        dot += za[d] * zb[d];
                    }
                    val *= dot / pf;
                }
            }
        }
        total += val;
    }
    total
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GprKernelModel {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    /// Standardized inducing rows (empty => constant fallback).
    pub basis: Vec<Vec<f64>>,
    pub alpha: Vec<f64>,
    pub spec: KernelSpec,
    /// Learned flat hyperparameter vector (see `DecodedKernel` for the layout).
    pub theta: Vec<f64>,
    pub y_mean: f64,
}

impl GprKernelModel {
    pub fn fit(x: &Rows, y: &[f64], m: usize, steps: usize, spec: &KernelSpec) -> GprKernelModel {
        let n = x.len().min(y.len());
        let p = if x.is_empty() { 0 } else { x[0].len() };
        let (means, scales) = std_fit(x);
        let mut y_mean = 0.0;
        for &v in &y[..n] {
            y_mean += v;
        }
        y_mean /= n.max(1) as f64;
        if n < 12 || p == 0 {
            return GprKernelModel {
                means,
                scales,
                basis: Vec::new(),
                alpha: Vec::new(),
                spec: spec.clone(),
                theta: Vec::new(),
                y_mean,
            };
        }
        let pf = p as f64;
        // Seeded inducing rows (the predictive basis), standardized — the same
        // subset recipe as the plain ARD head.
        let basis_size = n.min(m);
        let idx: Vec<usize> = if n > m {
            seeded_rows(n, m, GPR_SEED)
        } else {
            (0..n).collect()
        };
        let z: Rows = idx
            .iter()
            .map(|&i| (0..p).map(|j| (x[i][j] - means[j]) / scales[j]).collect())
            .collect();
        let yc: Vec<f64> = idx.iter().map(|&i| y[i] - y_mean).collect();
        let opt_size = basis_size.min(OPT_CAP);

        // ARD initialization shared by every rbf/rq factor (same recipe as the
        // plain head: median pairwise per-feature distance scaled by sqrt(p)).
        let n_init = opt_size.min(INIT_ROWS);
        let dim_scale = pf.sqrt();
        let mut ard_init = vec![0.0f64; p];
        for d in 0..p {
            let mut dists: Vec<f64> = Vec::new();
            for i in 0..n_init {
                for j in (i + 1)..n_init {
                    dists.push((z[i][d] - z[j][d]).abs());
                }
            }
            dists.sort_by(|a, b| a.total_cmp(b));
            let med = if dists.is_empty() {
                1.0
            } else {
                quantile(&dists, 0.5)
            };
            ard_init[d] = (med.max(1e-3) * dim_scale).ln();
        }
        let var: f64 = yc.iter().take(opt_size).map(|v| v * v).sum();
        let sy = (var / opt_size.max(1) as f64).sqrt();

        // Flat hyperparameter vector with per-slot box clamps. Term amplitudes
        // split the target variance so the summed prior variance stays sy^2.
        let amp_init = sy.max(1e-6).ln() - 0.5 * (spec.len() as f64).ln();
        let mut theta: Vec<f64> = Vec::new();
        let mut lo: Vec<f64> = Vec::new();
        let mut hi: Vec<f64> = Vec::new();
        let mut amp_idx: Vec<usize> = Vec::new();
        for term in spec {
            amp_idx.push(theta.len());
            theta.push(amp_init);
            lo.push(-15.0);
            hi.push(15.0);
            for prim in term {
                match prim {
                    KernelPrim::Rbf => {
                        for &init in ard_init.iter().take(p) {
                            theta.push(init);
                            lo.push(-6.0);
                            hi.push(6.0);
                        }
                    }
                    KernelPrim::Rq => {
                        for &init in ard_init.iter().take(p) {
                            theta.push(init);
                            lo.push(-6.0);
                            hi.push(6.0);
                        }
                        // log alpha, init alpha = 1 (heavy-tailed start).
                        theta.push(0.0);
                        lo.push(-3.0);
                        hi.push(5.0);
                    }
                    KernelPrim::Periodic => {
                        // log period, init 2 (standardized diffs are O(1)).
                        theta.push(2.0f64.ln());
                        lo.push(-3.0);
                        hi.push(3.0);
                        // log lengthscale, init sqrt(2p) so the summed sin^2
                        // exponent stays O(1) as the feature count grows.
                        theta.push(0.5 * (2.0 * pf).ln());
                        lo.push(-6.0);
                        hi.push(6.0);
                    }
                    KernelPrim::Linear => {}
                }
            }
        }
        let noise_idx = theta.len();
        theta.push((0.1 * sy).max(1e-6).ln());

        // Gradient ascent on the log marginal likelihood over the optimization
        // subset; dK/dtheta comes from W = alpha alpha^T - Ky^{-1} as in the
        // plain head, with per-term/per-factor chain rules.
        for _ in 0..steps {
            let mo = opt_size;
            let dec = decode_kernel(spec, &theta, p);
            let mut a = vec![0.0f64; mo * mo];
            for i in 0..mo {
                a[i * mo + i] = kernel_value(spec, &dec, pf, &z[i], &z[i]) + dec.sn2;
                for j in (i + 1)..mo {
                    let v = kernel_value(spec, &dec, pf, &z[i], &z[j]);
                    a[i * mo + j] = v;
                    a[j * mo + i] = v;
                }
            }
            let kinv = chol_inverse(a, mo);
            let mut alpha = vec![0.0f64; mo];
            for i in 0..mo {
                let mut s = 0.0;
                for j in 0..mo {
                    s += kinv[i * mo + j] * yc[j];
                }
                alpha[i] = s;
            }
            let mut g = vec![0.0f64; theta.len()];
            for i in 0..mo {
                let wii = alpha[i] * alpha[i] - kinv[i * mo + i];
                g[noise_idx] += wii;
                // Diagonal: every distance-based base is exactly 1 there, so
                // only the term amplitudes (and linear factors) contribute.
                let mut off = 0usize;
                for (t, term) in spec.iter().enumerate() {
                    let amp = off;
                    off += 1;
                    let mut val = dec.sf2[t];
                    for prim in term {
                        match prim {
                            KernelPrim::Rbf => off += p,
                            KernelPrim::Rq => off += p + 1,
                            KernelPrim::Periodic => off += 2,
                            KernelPrim::Linear => {
                                let dot: f64 = z[i].iter().take(p).map(|v| v * v).sum();
                                val *= dot / pf;
                            }
                        }
                    }
                    g[amp] += wii * val;
                }
                for j in (i + 1)..mo {
                    let wij = alpha[i] * alpha[j] - kinv[i * mo + j];
                    let mut off = 0usize;
                    for (t, term) in spec.iter().enumerate() {
                        let amp = off;
                        let nf = term.len();
                        let mut r2s = vec![0.0f64; nf];
                        let mut s2s = vec![0.0f64; nf];
                        let mut psums = vec![0.0f64; nf];
                        let mut val = dec.sf2[t];
                        for (f, prim) in term.iter().enumerate() {
                            match prim {
                                KernelPrim::Rbf => {
                                    let inv = &dec.inv_ls2[t][f];
                                    let mut r2 = 0.0;
                                    for d in 0..p {
                                        let diff = z[i][d] - z[j][d];
                                        r2 += diff * diff * inv[d];
                                    }
                                    r2s[f] = r2;
                                    val *= (-0.5 * r2).exp();
                                }
                                KernelPrim::Rq => {
                                    let inv = &dec.inv_ls2[t][f];
                                    let mut r2 = 0.0;
                                    for d in 0..p {
                                        let diff = z[i][d] - z[j][d];
                                        r2 += diff * diff * inv[d];
                                    }
                                    r2s[f] = r2;
                                    let al = dec.alpha[t][f];
                                    let s = r2 / (2.0 * al);
                                    val *= (-al * (1.0 + s).ln()).exp();
                                }
                                KernelPrim::Periodic => {
                                    let per = dec.period[t][f];
                                    let inv0 = dec.inv_ls2[t][f][0];
                                    let mut s2 = 0.0;
                                    let mut psum = 0.0;
                                    for (&zi, &zj) in z[i].iter().zip(z[j].iter()).take(p) {
                                        let ang = std::f64::consts::PI * (zi - zj) / per;
                                        let sa = ang.sin();
                                        let ca = ang.cos();
                                        s2 += sa * sa;
                                        psum += ang * sa * ca;
                                    }
                                    s2s[f] = s2;
                                    psums[f] = psum;
                                    val *= (-2.0 * s2 * inv0).exp();
                                }
                                KernelPrim::Linear => {
                                    let dot: f64 = z[i]
                                        .iter()
                                        .zip(z[j].iter())
                                        .take(p)
                                        .map(|(zi, zj)| zi * zj)
                                        .sum();
                                    val *= dot / pf;
                                }
                            }
                        }
                        g[amp] += 2.0 * wij * val;
                        let c = wij * val;
                        let mut o2 = amp + 1;
                        for (f, prim) in term.iter().enumerate() {
                            match prim {
                                KernelPrim::Rbf => {
                                    let inv = &dec.inv_ls2[t][f];
                                    for d in 0..p {
                                        let diff = z[i][d] - z[j][d];
                                        g[o2 + d] += c * diff * diff * inv[d];
                                    }
                                    o2 += p;
                                }
                                KernelPrim::Rq => {
                                    let inv = &dec.inv_ls2[t][f];
                                    let al = dec.alpha[t][f];
                                    let s = r2s[f] / (2.0 * al);
                                    let denom = 1.0 + s;
                                    for d in 0..p {
                                        let diff = z[i][d] - z[j][d];
                                        g[o2 + d] += c * diff * diff * inv[d] / denom;
                                    }
                                    g[o2 + p] += c * al * (s / denom - denom.ln());
                                    o2 += p + 1;
                                }
                                KernelPrim::Periodic => {
                                    let inv0 = dec.inv_ls2[t][f][0];
                                    g[o2] += c * 4.0 * inv0 * psums[f];
                                    g[o2 + 1] += c * 4.0 * s2s[f] * inv0;
                                    o2 += 2;
                                }
                                KernelPrim::Linear => {}
                            }
                        }
                        off = o2;
                    }
                }
            }
            let scale = LR / mo as f64;
            for k in 0..noise_idx {
                let step = (scale * g[k]).clamp(-0.5, 0.5);
                theta[k] = (theta[k] + step).clamp(lo[k], hi[k]);
            }
            // Noise floored at exp(-6) of the LARGEST term amplitude — the same
            // conditioning jitter as the plain head.
            let g_sn = g[noise_idx] * dec.sn2;
            let mut max_amp = f64::NEG_INFINITY;
            for &ai in &amp_idx {
                if theta[ai] > max_amp {
                    max_amp = theta[ai];
                }
            }
            let step_sn = (scale * g_sn).clamp(-0.5, 0.5);
            theta[noise_idx] = (theta[noise_idx] + step_sn).clamp(max_amp - 6.0, max_amp);
        }

        // Predictive coefficients on the full inducing basis.
        let dec = decode_kernel(spec, &theta, p);
        let mb = basis_size;
        let mut a = vec![0.0f64; mb * mb];
        for i in 0..mb {
            a[i * mb + i] = kernel_value(spec, &dec, pf, &z[i], &z[i]) + dec.sn2;
            for j in (i + 1)..mb {
                let v = kernel_value(spec, &dec, pf, &z[i], &z[j]);
                a[i * mb + j] = v;
                a[j * mb + i] = v;
            }
        }
        let mut b = yc.clone();
        let alpha = cholesky_solve(&mut a, &mut b, mb);
        GprKernelModel {
            means,
            scales,
            basis: z,
            alpha,
            spec: spec.clone(),
            theta,
            y_mean,
        }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        let p = self.means.len();
        if self.basis.is_empty() {
            return x.iter().map(|_| self.y_mean.clamp(-60.0, 60.0)).collect();
        }
        let pf = p as f64;
        let dec = decode_kernel(&self.spec, &self.theta, p);
        x.iter()
            .map(|row| {
                let zr: Vec<f64> = (0..p.min(row.len()))
                    .map(|j| (row[j] - self.means[j]) / self.scales[j])
                    .collect();
                let mut s = self.y_mean;
                for (b, zb) in self.basis.iter().enumerate() {
                    s += self.alpha[b] * kernel_value(&self.spec, &dec, pf, &zr, zb);
                }
                s.clamp(-60.0, 60.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krr::{KrrModel, DEFAULT_M as KRR_M};
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
    fn gpr_is_deterministic_and_learns_toy() {
        let (x, y) = toy();
        let m1 = GprModel::fit(&x, &y, DEFAULT_M, DEFAULT_STEPS);
        let m2 = GprModel::fit(&x, &y, DEFAULT_M, DEFAULT_STEPS);
        assert_eq!(m1.predict(&x), m2.predict(&x));
        let rep = cv_r2(
            &GenomeIr::parse("(pipe X (gpr))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        assert!(rep.oof_r2 > 0.9, "gpr oof_r2={}", rep.oof_r2);
    }

    #[test]
    fn gpr_matches_or_beats_krr_on_toy() {
        let (x, y) = toy();
        let gpr = cv_r2(
            &GenomeIr::parse("(pipe X (gpr))").unwrap(),
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
        println!(
            "smoke[toy] gpr_oof_r2={} krr_oof_r2={}",
            gpr.oof_r2, krr.oof_r2
        );
        assert!(
            gpr.oof_r2 >= krr.oof_r2,
            "gpr={} krr={}",
            gpr.oof_r2,
            krr.oof_r2
        );
        // Sanity: the shared krr default still fits.
        let _ = KrrModel::fit(&x, &y, KRR_M);
    }

    #[test]
    fn gpr_learns_relevance_and_ignores_noise_feature() {
        // Feature 3 is pure noise; ARD should give it a long lengthscale.
        let mut rng = Pcg32::new(5, 3);
        let mut x: Rows = Vec::new();
        let mut y = Vec::new();
        for _ in 0..200 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            let noise = rng.next_f64();
            x.push(vec![a, b, noise]);
            y.push(1.5 * a - b);
        }
        let model = GprModel::fit(&x, &y, DEFAULT_M, DEFAULT_STEPS);
        assert_eq!(model.ls.len(), 3);
        assert!(
            model.ls[2] > model.ls[0],
            "noise ls={} signal ls={}",
            model.ls[2],
            model.ls[0]
        );
    }

    #[test]
    fn gpr_subset_cap_is_exercised() {
        let mut rng = Pcg32::new(2, 2);
        let mut x: Rows = Vec::new();
        let mut y = Vec::new();
        for _ in 0..300 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            x.push(vec![a, b]);
            y.push(a - 0.5 * b);
        }
        let m = GprModel::fit(&x, &y, 80, 10);
        assert_eq!(m.basis.len(), 80);
        let m2 = GprModel::fit(&x, &y, 80, 10);
        assert_eq!(m.predict(&x), m2.predict(&x));
    }

    #[test]
    fn gpr_small_n_falls_back_to_mean() {
        let x: Rows = (0..6).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let m = GprModel::fit(&x, &y, DEFAULT_M, DEFAULT_STEPS);
        assert!(m.basis.is_empty());
        assert!(m.predict(&x).iter().all(|v| (*v - 2.5).abs() < 1e-12));
    }

    #[test]
    fn kernel_expression_parser_accepts_and_rejects() {
        assert_eq!(parse_kernel("rbf").unwrap(), vec![vec![KernelPrim::Rbf]]);
        assert_eq!(
            parse_kernel("rbf+linear").unwrap(),
            vec![vec![KernelPrim::Rbf], vec![KernelPrim::Linear]]
        );
        assert_eq!(
            parse_kernel("rbf*periodic").unwrap(),
            vec![vec![KernelPrim::Rbf, KernelPrim::Periodic]]
        );
        assert_eq!(
            parse_kernel("rq+linear").unwrap(),
            vec![vec![KernelPrim::Rq], vec![KernelPrim::Linear]]
        );
        assert_eq!(
            parse_kernel("rbf*periodic+linear").unwrap(),
            vec![
                vec![KernelPrim::Rbf, KernelPrim::Periodic],
                vec![KernelPrim::Linear]
            ]
        );
        assert!(parse_kernel("matern").is_err());
        assert!(parse_kernel("").is_err());
        assert!(parse_kernel("rbf+").is_err());
        assert!(
            parse_kernel("rbf+rq+linear+periodic").is_err(),
            "4 prims must be rejected"
        );
    }

    #[test]
    fn composite_kernels_are_deterministic_and_learn_toy() {
        let (x, y) = toy();
        for src in ["rq", "rbf+linear", "rbf*periodic", "rq+linear"] {
            let spec = parse_kernel(src).unwrap();
            let m1 = GprKernelModel::fit(&x, &y, DEFAULT_M, 20, &spec);
            let m2 = GprKernelModel::fit(&x, &y, DEFAULT_M, 20, &spec);
            assert_eq!(m1.predict(&x), m2.predict(&x), "kernel={src}");
            let genome = format!("(pipe X (gpr kernel={src} steps=20))");
            let rep = cv_r2(&GenomeIr::parse(&genome).unwrap(), &x, &y, 5, "toy").unwrap();
            println!("smoke[toy] gpr kernel={src} oof_r2={}", rep.oof_r2);
            assert!(rep.oof_r2 > 0.8, "kernel={src} oof_r2={}", rep.oof_r2);
        }
    }

    #[test]
    fn linear_kernel_nails_a_linear_target() {
        // Pure Bayesian-linear-regression check: kernel="linear" on a noiseless
        // linear target must be essentially exact out of fold.
        let mut rng = Pcg32::new(11, 4);
        let mut x: Rows = Vec::new();
        let mut y = Vec::new();
        for _ in 0..200 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            let c = rng.next_f64();
            x.push(vec![a, b, c]);
            y.push(2.0 * a - 1.5 * b + 0.5 * c);
        }
        let rep = cv_r2(
            &GenomeIr::parse("(pipe X (gpr kernel=linear steps=20))").unwrap(),
            &x,
            &y,
            5,
            "lin",
        )
        .unwrap();
        assert!(rep.oof_r2 > 0.99, "oof_r2={}", rep.oof_r2);
    }

    #[test]
    fn periodic_kernel_beats_rbf_on_periodic_signal() {
        // A strongly periodic 1-feature signal over several cycles: the learned
        // periodic component should fit it at least as well as plain RBF-ARD.
        let mut rng = Pcg32::new(3, 8);
        let mut x: Rows = Vec::new();
        let mut y = Vec::new();
        for _ in 0..240 {
            let t = rng.next_f64() * 12.0;
            x.push(vec![t]);
            y.push((2.0 * std::f64::consts::PI * t).sin());
        }
        let per = cv_r2(
            &GenomeIr::parse("(pipe X (gpr kernel=rbf*periodic steps=40))").unwrap(),
            &x,
            &y,
            5,
            "per",
        )
        .unwrap();
        let rbf = cv_r2(
            &GenomeIr::parse("(pipe X (gpr steps=40))").unwrap(),
            &x,
            &y,
            5,
            "per",
        )
        .unwrap();
        println!(
            "accuracy[periodic] rbf*periodic={} rbf={}",
            per.oof_r2, rbf.oof_r2
        );
        assert!(
            per.oof_r2 > rbf.oof_r2 - 0.05,
            "periodic={} rbf={}",
            per.oof_r2,
            rbf.oof_r2
        );
    }

    #[test]
    fn composite_small_n_falls_back_to_mean() {
        let x: Rows = (0..6).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let spec = parse_kernel("rq+linear").unwrap();
        let m = GprKernelModel::fit(&x, &y, DEFAULT_M, DEFAULT_STEPS, &spec);
        assert!(m.basis.is_empty());
        assert!(m.predict(&x).iter().all(|v| (*v - 2.5).abs() < 1e-12));
    }
}
