//! Regression model: logistic squash, then correlation-screened products, then Gaussian-process regression with a composed kernel.
//!
//! This program trains a regression model on train.csv, in which every row holds the feature
//! columns followed by the target as the last column, and then writes one prediction per line for
//! the rows of test_features.csv. All arithmetic is IEEE-754 double precision in a fixed
//! evaluation order, so repeated runs give identical bits.
//!
//! The feature pipeline has 2 stages, applied in order; each stage is marked "replaced" when it
//! overwrites the columns it receives or "appended" when it keeps them and adds its outputs
//! alongside, and after every stage non-finite cells become 0.0 and each row is capped at 512
//! columns. Stage 1 (logistic_squash, appended) squashes every value through the logistic
//! function 1/(1 + exp(-v)) with the argument clipped to [-30, 30]. Stage 2
//! (correlation_screened_products, appended) multiplies pairs among the 5 columns most correlated
//! (absolute Pearson) with the target, a screened interaction expansion.
//!
//! The head (gaussian_process_composite_kernel_regression) is Gaussian-process regression
//! (Rasmussen & Williams, 2006) whose covariance is the composed kernel "rbf+linear" — a sum of
//! products of primitives in the spirit of kernel structure search (Duvenaud et al., 2013): a
//! squared-exponential factor with per-feature automatic-relevance lengthscales; a homogeneous
//! linear factor (x . x')/p, giving a Bayesian-linear-regression component. Each sum term carries
//! a learned amplitude; every factor's shape parameters and the noise variance are learned on
//! features scaled by the inverse training standard deviation via 40 steps of gradient ascent on
//! the log marginal likelihood, whose gradient 0.5 * trace((a a^T - Ky^-1) dKy/dtheta) with a =
//! Ky^-1 y is evaluated with one Cholesky-based inverse per step on at most 256 seeded rows; the
//! predictive coefficients are then solved on up to 300 seeded inducing rows and predictions
//! k(x)^T a are clipped to [-60, 60]. Seeded stages draw weights from a permuted congruential
//! generator (PCG XSH-RR 64/32; O'Neill, 2014) keyed by the stage seed and the input/output
//! widths, so training is fully deterministic. Missing or malformed CSV cells parse as 0.0 and
//! short rows are padded with zeros to the training width.
//!
//! Build: rustc -O model.rs
//! Usage: ./model train.csv test_features.csv predictions.csv

#![forbid(unsafe_code)]
#![allow(dead_code)]

/// Logistic function 1/(1+exp(-v)) with the argument clipped to [-30, 30].
fn m_sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())
}

/// Permuted congruential generator, XSH-RR 64/32 variant (O'Neill, 2014).
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64, stream: u64) -> Pcg32 {
        let mut rng = Pcg32 { state: 0, inc: (stream << 1) | 1 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}

/// Replace non-finite values with 0.0 in every row.
fn sanitize(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| row.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect())
        .collect()
}

/// Stage epilogue: optionally keep the inputs, cap width at 512, zero non-finite cells.
fn finish_stage(x: &[Vec<f64>], mut out: Vec<Vec<f64>>, append: bool) -> Vec<Vec<f64>> {
    if append {
        for (i, row) in out.iter_mut().enumerate() {
            let mut merged = x[i].clone();
            merged.append(row);
            *row = merged;
        }
    }
    for row in out.iter_mut() {
        if row.len() > 512 {
            row.truncate(512);
        }
        for v in row.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        if row.is_empty() {
            row.push(0.0);
        }
    }
    out
}

/// Linearly interpolated quantile of an ascending slice (the numpy 'linear' rule).
fn quantile(sorted_values: &[f64], q: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted_values.len() - 1) as f64;
    let lo = pos as usize;
    let hi = (lo + 1).min(sorted_values.len() - 1);
    let frac = pos - lo as f64;
    sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
}

/// Pearson correlation, 0.0 for degenerate inputs.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let mut ma = 0.0;
    for i in 0..n {
        ma += a[i];
    }
    ma /= n as f64;
    let mut mb = 0.0;
    for i in 0..n {
        mb += b[i];
    }
    mb /= n as f64;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    sab / (saa * sbb).sqrt()
}

/// Per-column mean and standard deviation (1/n convention), floored at 1e-9.
fn std_fit(x: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len().max(1) as f64;
    let p = x[0].len();
    let mut means = vec![0.0; p];
    let mut scales = vec![1.0; p];
    for j in 0..p {
        let mut s = 0.0;
        for row in x {
            s += row[j];
        }
        means[j] = s / n;
        let mut v = 0.0;
        for row in x {
            let d = row[j] - means[j];
            v += d * d;
        }
        scales[j] = (v / n).sqrt().max(1e-9);
    }
    (means, scales)
}

/// Solve the symmetric positive-definite system A w = b.
///
/// A (flattened row-major, overwritten) is factored in place as L L^T, then a
/// forward and a backward triangular solve produce w. A tiny 1e-6 pivot stands
/// in whenever a diagonal entry falls under 1e-12.
fn cholesky_solve(a: &mut [f64], b: &[f64], p: usize) -> Vec<f64> {
    for i in 0..p {
        for j in 0..(i + 1) {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= a[i * p + k] * a[j * p + k];
            }
            if i == j {
                a[i * p + j] = if s > 1e-12 { s.sqrt() } else { 1e-6 };
            } else {
                a[i * p + j] = s / a[j * p + j];
            }
        }
    }
    let mut z = vec![0.0; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * p + k] * z[k];
        }
        z[i] = s / a[i * p + i];
    }
    let mut w = vec![0.0; p];
    for i in (0..p).rev() {
        let mut s = z[i];
        for k in (i + 1)..p {
            s -= a[k * p + i] * w[k];
        }
        w[i] = s / a[i * p + i];
    }
    w
}

/// This transform squashes every value through the logistic function 1/(1 + exp(-v)) with the
/// argument clipped to [-30, 30].
fn logistic_squash_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| row.iter().map(|v| m_sigmoid(*v)).collect())
        .collect();
    finish_stage(x, out, append)
}

struct CorrelationScreenedProducts {
    pairs: Vec<(usize, usize)>,
}

/// This transform multiplies pairs among the 5 columns most correlated (absolute Pearson) with
/// the target, a screened interaction expansion.
fn correlation_screened_products_fit(x: &[Vec<f64>], y: &[f64], top_k: usize) -> CorrelationScreenedProducts {
    let cols = x[0].len();
    let top_k = top_k.min(cols);
    let mut scored: Vec<(usize, f64)> = Vec::new();
    for j in 0..cols {
        let col: Vec<f64> = x.iter().map(|row| row[j]).collect();
        scored.push((j, pearson(&col, y).abs()));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut selected: Vec<usize> = scored[..top_k].iter().map(|&(j, _)| j).collect();
    selected.sort();
    let mut pairs = Vec::new();
    for a in 0..selected.len() {
        for b in (a + 1)..selected.len() {
            pairs.push((selected[a], selected[b]));
        }
    }
    CorrelationScreenedProducts { pairs }
}

fn correlation_screened_products_apply(state: &CorrelationScreenedProducts, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| state.pairs.iter().map(|&(a, b)| row[a] * row[b]).collect())
        .collect();
    finish_stage(x, out, append)
}

struct GaussianProcessCompositeKernelRegression {
    means: Vec<f64>,
    scales: Vec<f64>,
    basis: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    theta: Vec<f64>,
    y_mean: f64,
}

/// Decoded per-evaluation kernel parameters (see kernel_decode).
struct KernelParams {
    /// Per-term signal variance exp(2 * log amplitude).
    sf2: Vec<f64>,
    /// Per (term, factor) inverse squared lengthscales: p entries for the
    /// squared-exponential / rational-quadratic factors, one shared entry for
    /// periodic factors, empty for linear factors.
    inv_len2: Vec<Vec<Vec<f64>>>,
    /// Per (term, factor) rational-quadratic mixing exponent (0.0 elsewhere).
    alpha: Vec<Vec<f64>>,
    /// Per (term, factor) period (0.0 elsewhere).
    period: Vec<Vec<f64>>,
    /// Observation noise variance exp(2 * log noise std).
    noise2: f64,
}

/// k distinct row indices out of n via a seeded partial Fisher-Yates shuffle,
/// returned in ascending order.
fn seeded_rows(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = Pcg32::new(seed, 54);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u32() as usize) % (n - i);
        let tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
    let mut out = idx[..k].to_vec();
    out.sort();
    out
}

/// Symmetric-positive-definite inverse of a (flattened m*m, consumed): factor
/// A = L L^T in place, invert the lower-triangular L, then form A^-1 = L^-T L^-1.
/// A tiny 1e-6 pivot stands in whenever a diagonal entry falls under 1e-12.
fn chol_inverse(mut a: Vec<f64>, m: usize) -> Vec<f64> {
    for i in 0..m {
        for j in 0..(i + 1) {
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
    let mut linv = vec![0.0; m * m];
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
    let mut inv = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..(i + 1) {
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

/// Decode the flat hyperparameter vector: per sum term [log amplitude, factor
/// shape parameters...], then the global log noise std last.
fn kernel_decode(kernel_terms: &[Vec<&str>], theta: &[f64], p: usize) -> KernelParams {
    let mut sf2: Vec<f64> = Vec::new();
    let mut inv_len2: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut alpha: Vec<Vec<f64>> = Vec::new();
    let mut period: Vec<Vec<f64>> = Vec::new();
    let mut off = 0usize;
    for term in kernel_terms {
        sf2.push((2.0 * theta[off]).exp());
        off += 1;
        let mut inv_t: Vec<Vec<f64>> = Vec::new();
        let mut alpha_t: Vec<f64> = Vec::new();
        let mut period_t: Vec<f64> = Vec::new();
        for kind in term {
            match *kind {
                "squared_exponential" => {
                    let mut inv_f = Vec::new();
                    for d in 0..p {
                        let length = theta[off + d].exp();
                        inv_f.push(1.0 / (length * length));
                    }
                    inv_t.push(inv_f);
                    alpha_t.push(0.0);
                    period_t.push(0.0);
                    off += p;
                }
                "rational_quadratic" => {
                    let mut inv_f = Vec::new();
                    for d in 0..p {
                        let length = theta[off + d].exp();
                        inv_f.push(1.0 / (length * length));
                    }
                    inv_t.push(inv_f);
                    alpha_t.push(theta[off + p].exp());
                    period_t.push(0.0);
                    off += p + 1;
                }
                "periodic" => {
                    period_t.push(theta[off].exp());
                    let length = theta[off + 1].exp();
                    inv_t.push(vec![1.0 / (length * length)]);
                    alpha_t.push(0.0);
                    off += 2;
                }
                _ => {
                    inv_t.push(Vec::new());
                    alpha_t.push(0.0);
                    period_t.push(0.0);
                }
            }
        }
        inv_len2.push(inv_t);
        alpha.push(alpha_t);
        period.push(period_t);
    }
    let noise2 = (2.0 * theta[off]).exp();
    KernelParams { sf2, inv_len2, alpha, period, noise2 }
}

/// Covariance k(za, zb): the sum over terms of the term amplitude times the
/// product of its factor values, in declaration order.
fn kernel_value(kernel_terms: &[Vec<&str>], dec: &KernelParams, pf: f64, za: &[f64], zb: &[f64]) -> f64 {
    let dm = za.len().min(zb.len());
    let mut total = 0.0;
    for t in 0..kernel_terms.len() {
        let term = &kernel_terms[t];
        let mut val = dec.sf2[t];
        for f in 0..term.len() {
            match term[f] {
                "squared_exponential" => {
                    let inv = &dec.inv_len2[t][f];
                    let mut r2 = 0.0;
                    for d in 0..dm {
                        let diff = za[d] - zb[d];
                        r2 += diff * diff * inv[d];
                    }
                    val *= (-0.5 * r2).exp();
                }
                "rational_quadratic" => {
                    let inv = &dec.inv_len2[t][f];
                    let mut r2 = 0.0;
                    for d in 0..dm {
                        let diff = za[d] - zb[d];
                        r2 += diff * diff * inv[d];
                    }
                    let al = dec.alpha[t][f];
                    let s = r2 / (2.0 * al);
                    val *= (-al * (1.0 + s).ln()).exp();
                }
                "periodic" => {
                    let per = dec.period[t][f];
                    let inv0 = dec.inv_len2[t][f][0];
                    let mut s2 = 0.0;
                    for d in 0..dm {
                        let ang = std::f64::consts::PI * (za[d] - zb[d]) / per;
                        let sa = ang.sin();
                        s2 += sa * sa;
                    }
                    val *= (-2.0 * s2 * inv0).exp();
                }
                _ => {
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

/// This head is Gaussian-process regression (Rasmussen & Williams, 2006) whose covariance is
/// the composed kernel "rbf+linear" — a sum of products of primitives in the spirit of kernel
/// structure search (Duvenaud et al., 2013): a squared-exponential factor with per-feature
/// automatic-relevance lengthscales; a homogeneous linear factor (x . x')/p, giving a
/// Bayesian-linear-regression component. Each sum term carries a learned amplitude; every
/// factor's shape parameters and the noise variance are learned on features scaled by the
/// inverse training standard deviation via 40 steps of gradient ascent on the log marginal
/// likelihood, whose gradient 0.5 * trace((a a^T - Ky^-1) dKy/dtheta) with a = Ky^-1 y is
/// evaluated with one Cholesky-based inverse per step on at most 256 seeded rows; the
/// predictive coefficients are then solved on up to 300 seeded inducing rows and predictions
/// k(x)^T a are clipped to [-60, 60].
fn gaussian_process_composite_kernel_regression_fit(x: &[Vec<f64>], y: &[f64], m: usize,
                                                    steps: usize, kernel_terms: &[Vec<&str>])
                                                    -> GaussianProcessCompositeKernelRegression {
    let n = x.len().min(y.len());
    let p = x[0].len();
    let (means, scales) = std_fit(x);
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    if n < 12 || p == 0 {
        return GaussianProcessCompositeKernelRegression {
            means,
            scales,
            basis: Vec::new(),
            alpha: Vec::new(),
            theta: Vec::new(),
            y_mean,
        };
    }
    let pf = p as f64;
    let basis_size = n.min(m);
    let idx: Vec<usize> = if n > m { seeded_rows(n, m, 909) } else { (0..n).collect() };
    let z: Vec<Vec<f64>> = idx
        .iter()
        .map(|&i| (0..p).map(|j| (x[i][j] - means[j]) / scales[j]).collect())
        .collect();
    let yc: Vec<f64> = idx.iter().map(|&i| y[i] - y_mean).collect();
    let opt_size = basis_size.min(256);
    // Lengthscale seed for every distance factor: the median pairwise
    // per-feature distance scaled by sqrt(p), so the summed kernel exponent
    // stays O(1) as the feature count grows.
    let n_init = opt_size.min(96);
    let dim_scale = pf.sqrt();
    let mut ard_init = vec![0.0; p];
    for d in 0..p {
        let mut dists: Vec<f64> = Vec::new();
        for i in 0..n_init {
            for j in (i + 1)..n_init {
                dists.push((z[i][d] - z[j][d]).abs());
            }
        }
        dists.sort_by(|a, b| a.total_cmp(b));
        let med = if dists.is_empty() { 1.0 } else { quantile(&dists, 0.5) };
        ard_init[d] = (med.max(1e-3) * dim_scale).ln();
    }
    let mut var = 0.0;
    for i in 0..opt_size {
        var += yc[i] * yc[i];
    }
    let sy = (var / opt_size.max(1) as f64).sqrt();
    // Flat hyperparameter vector with per-slot box clamps; term amplitudes
    // split the target variance so the summed prior variance stays sy^2.
    let amp_init = sy.max(1e-6).ln() - 0.5 * (kernel_terms.len() as f64).ln();
    let mut theta: Vec<f64> = Vec::new();
    let mut lo: Vec<f64> = Vec::new();
    let mut hi: Vec<f64> = Vec::new();
    let mut amp_idx: Vec<usize> = Vec::new();
    for term in kernel_terms {
        amp_idx.push(theta.len());
        theta.push(amp_init);
        lo.push(-15.0);
        hi.push(15.0);
        for kind in term {
            match *kind {
                "squared_exponential" => {
                    for d in 0..p {
                        theta.push(ard_init[d]);
                        lo.push(-6.0);
                        hi.push(6.0);
                    }
                }
                "rational_quadratic" => {
                    for d in 0..p {
                        theta.push(ard_init[d]);
                        lo.push(-6.0);
                        hi.push(6.0);
                    }
                    theta.push(0.0);
                    lo.push(-3.0);
                    hi.push(5.0);
                }
                "periodic" => {
                    theta.push(2.0f64.ln());
                    lo.push(-3.0);
                    hi.push(3.0);
                    theta.push(0.5 * (2.0 * pf).ln());
                    lo.push(-6.0);
                    hi.push(6.0);
                }
                _ => {}
            }
        }
    }
    let noise_idx = theta.len();
    theta.push((0.1 * sy).max(1e-6).ln());
    // Gradient ascent on the log marginal likelihood over the first opt_size
    // rows; dK/dtheta uses per-term / per-factor chain rules.
    for _ in 0..steps {
        let mo = opt_size;
        let dec = kernel_decode(kernel_terms, &theta, p);
        let mut a = vec![0.0; mo * mo];
        for i in 0..mo {
            a[i * mo + i] = kernel_value(kernel_terms, &dec, pf, &z[i], &z[i]) + dec.noise2;
            for j in (i + 1)..mo {
                let v = kernel_value(kernel_terms, &dec, pf, &z[i], &z[j]);
                a[i * mo + j] = v;
                a[j * mo + i] = v;
            }
        }
        let kinv = chol_inverse(a, mo);
        let mut alpha = vec![0.0; mo];
        for i in 0..mo {
            let mut s = 0.0;
            let base = i * mo;
            for j in 0..mo {
                s += kinv[base + j] * yc[j];
            }
            alpha[i] = s;
        }
        let mut g = vec![0.0; theta.len()];
        for i in 0..mo {
            let wii = alpha[i] * alpha[i] - kinv[i * mo + i];
            g[noise_idx] += wii;
            // Diagonal: distance-based factor values are exactly 1 there, so
            // only the term amplitudes (and linear factors) contribute.
            let mut off = 0usize;
            for t in 0..kernel_terms.len() {
                let term = &kernel_terms[t];
                let amp = off;
                off += 1;
                let mut val = dec.sf2[t];
                for kind in term {
                    match *kind {
                        "squared_exponential" => off += p,
                        "rational_quadratic" => off += p + 1,
                        "periodic" => off += 2,
                        _ => {
                            let mut dot = 0.0;
                            for d in 0..p {
                                dot += z[i][d] * z[i][d];
                            }
                            val *= dot / pf;
                        }
                    }
                }
                g[amp] += wii * val;
            }
            for j in (i + 1)..mo {
                let wij = alpha[i] * alpha[j] - kinv[i * mo + j];
                let mut off = 0usize;
                for t in 0..kernel_terms.len() {
                    let term = &kernel_terms[t];
                    let amp = off;
                    let nf = term.len();
                    let mut r2s = vec![0.0; nf];
                    let mut s2s = vec![0.0; nf];
                    let mut psums = vec![0.0; nf];
                    let mut val = dec.sf2[t];
                    for f in 0..nf {
                        match term[f] {
                            "squared_exponential" => {
                                let inv = &dec.inv_len2[t][f];
                                let mut r2 = 0.0;
                                for d in 0..p {
                                    let diff = z[i][d] - z[j][d];
                                    r2 += diff * diff * inv[d];
                                }
                                r2s[f] = r2;
                                val *= (-0.5 * r2).exp();
                            }
                            "rational_quadratic" => {
                                let inv = &dec.inv_len2[t][f];
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
                            "periodic" => {
                                let per = dec.period[t][f];
                                let inv0 = dec.inv_len2[t][f][0];
                                let mut s2 = 0.0;
                                let mut psum = 0.0;
                                for d in 0..p {
                                    let ang = std::f64::consts::PI * (z[i][d] - z[j][d]) / per;
                                    let sa = ang.sin();
                                    let ca = ang.cos();
                                    s2 += sa * sa;
                                    psum += ang * sa * ca;
                                }
                                s2s[f] = s2;
                                psums[f] = psum;
                                val *= (-2.0 * s2 * inv0).exp();
                            }
                            _ => {
                                let mut dot = 0.0;
                                for d in 0..p {
                                    dot += z[i][d] * z[j][d];
                                }
                                val *= dot / pf;
                            }
                        }
                    }
                    g[amp] += 2.0 * wij * val;
                    let c = wij * val;
                    let mut o2 = amp + 1;
                    for f in 0..nf {
                        match term[f] {
                            "squared_exponential" => {
                                let inv = &dec.inv_len2[t][f];
                                for d in 0..p {
                                    let diff = z[i][d] - z[j][d];
                                    g[o2 + d] += c * diff * diff * inv[d];
                                }
                                o2 += p;
                            }
                            "rational_quadratic" => {
                                let inv = &dec.inv_len2[t][f];
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
                            "periodic" => {
                                let inv0 = dec.inv_len2[t][f][0];
                                g[o2] += c * 4.0 * inv0 * psums[f];
                                g[o2 + 1] += c * 4.0 * s2s[f] * inv0;
                                o2 += 2;
                            }
                            _ => {}
                        }
                    }
                    off = o2;
                }
            }
        }
        let scale = 1.0 / mo as f64;
        for q in 0..noise_idx {
            let step = (scale * g[q]).clamp(-0.5, 0.5);
            theta[q] = (theta[q] + step).clamp(lo[q], hi[q]);
        }
        // Noise floored at exp(-6) of the LARGEST term amplitude (the same
        // conditioning jitter as a single-kernel Gaussian process).
        let g_sn = g[noise_idx] * dec.noise2;
        let mut max_amp = f64::NEG_INFINITY;
        for &ai in &amp_idx {
            if theta[ai] > max_amp {
                max_amp = theta[ai];
            }
        }
        let step_sn = (scale * g_sn).clamp(-0.5, 0.5);
        theta[noise_idx] = (theta[noise_idx] + step_sn).clamp(max_amp - 6.0, max_amp);
    }
    let dec = kernel_decode(kernel_terms, &theta, p);
    let mb = basis_size;
    let mut a = vec![0.0; mb * mb];
    for i in 0..mb {
        a[i * mb + i] = kernel_value(kernel_terms, &dec, pf, &z[i], &z[i]) + dec.noise2;
        for j in (i + 1)..mb {
            let v = kernel_value(kernel_terms, &dec, pf, &z[i], &z[j]);
            a[i * mb + j] = v;
            a[j * mb + i] = v;
        }
    }
    let alpha = cholesky_solve(&mut a, &yc, mb);
    GaussianProcessCompositeKernelRegression { means, scales, basis: z, alpha, theta, y_mean }
}

fn gaussian_process_composite_kernel_regression_predict(state: &GaussianProcessCompositeKernelRegression,
                                                        x: &[Vec<f64>], kernel_terms: &[Vec<&str>])
                                                        -> Vec<f64> {
    let p = state.means.len();
    if state.basis.is_empty() {
        return x.iter().map(|_| state.y_mean.clamp(-60.0, 60.0)).collect();
    }
    let pf = p as f64;
    let dec = kernel_decode(kernel_terms, &state.theta, p);
    let mut predictions = Vec::new();
    for row in x {
        let zr: Vec<f64> = (0..p.min(row.len()))
            .map(|j| (row[j] - state.means[j]) / state.scales[j])
            .collect();
        let mut s = state.y_mean;
        for b in 0..state.basis.len() {
            s += state.alpha[b] * kernel_value(kernel_terms, &dec, pf, &zr, &state.basis[b]);
        }
        predictions.push(s.clamp(-60.0, 60.0));
    }
    predictions
}

/// Feature stages applied in order, then the gaussian_process_composite_kernel_regression head.
struct Model {
    stage2: CorrelationScreenedProducts,
    head: GaussianProcessCompositeKernelRegression,
}

impl Model {
    /// Fit every stage in order on the training rows.
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Model {
        if x.is_empty() || y.len() < x.len() {
            eprintln!("training data is empty or target is short");
            std::process::exit(2);
        }
        let x = sanitize(x);
        let x = logistic_squash_apply(&x, true);
        let stage2 = correlation_screened_products_fit(&x, y, 5);
        let x = correlation_screened_products_apply(&stage2, &x, true);
        let head = gaussian_process_composite_kernel_regression_fit(&x, y, 300, 40, &[vec!["squared_exponential"], vec!["linear"]]);
        Model { stage2, head }
    }

    /// Apply the fitted stages, then the regression head.
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let x = sanitize(x);
        let x = logistic_squash_apply(&x, true);
        let x = correlation_screened_products_apply(&self.stage2, &x, true);
        gaussian_process_composite_kernel_regression_predict(&self.head, &x, &[vec!["squared_exponential"], vec!["linear"]])
    }

    /// Coefficient of determination R^2 of predict against y.
    fn score(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let n = y.len().min(predictions.len());
        if n == 0 {
            return 0.0;
        }
        let mut mean = 0.0;
        for i in 0..n {
            mean += y[i];
        }
        mean /= n as f64;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for i in 0..n {
            let d = y[i] - predictions[i];
            ss_res += d * d;
            let t = y[i] - mean;
            ss_tot += t * t;
        }
        if ss_tot <= 0.0 {
            return 0.0;
        }
        1.0 - ss_res / ss_tot
    }
}

/// Parse one CSV cell; blank, malformed, or non-finite cells become 0.0.
fn parse_cell(cell: &str) -> f64 {
    let text = cell.trim();
    if text.is_empty() || text.contains('_') {
        return 0.0;
    }
    match text.parse::<f64>() {
        Ok(value) => {
            if value.is_finite() {
                value
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

fn looks_numeric(cell: &str) -> bool {
    let text = cell.trim();
    text.is_empty() || (!text.contains('_') && text.parse::<f64>().is_ok())
}

/// Read CSV rows of floats, skipping blank lines and one header line if present.
fn read_csv(path: &str) -> Vec<Vec<f64>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("cannot read {}: {}", path, err);
            std::process::exit(2);
        }
    };
    let mut records: Vec<Vec<&str>> = Vec::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.iter().all(|cell| cell.trim().is_empty()) {
            continue;
        }
        records.push(cells);
    }
    if !records.is_empty() && !records[0].iter().all(|cell| looks_numeric(cell)) {
        records.remove(0);
    }
    records
        .iter()
        .map(|cells| cells.iter().map(|cell| parse_cell(cell)).collect())
        .collect()
}

/// Pad short rows with 0.0 (and trim long ones) to the training feature width.
fn pad(mut row: Vec<f64>, width: usize) -> Vec<f64> {
    while row.len() < width {
        row.push(0.0);
    }
    row.truncate(width);
    row
}

/// One shortest round-trip decimal per line (matches Python's repr()).
fn write_predictions(path: &str, predictions: &[f64]) {
    let mut out = String::new();
    for value in predictions {
        out.push_str(&format!("{:?}\n", value));
    }
    if let Err(err) = std::fs::write(path, out) {
        eprintln!("cannot write {}: {}", path, err);
        std::process::exit(2);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: model train.csv test_features.csv predictions.csv");
        std::process::exit(2);
    }
    let train_rows = read_csv(&args[1]);
    if train_rows.is_empty() {
        eprintln!("training data is empty");
        std::process::exit(2);
    }
    let y_train: Vec<f64> = train_rows.iter().map(|row| row[row.len() - 1]).collect();
    let features: Vec<Vec<f64>> = train_rows.iter().map(|row| row[..row.len() - 1].to_vec()).collect();
    let mut width = 0usize;
    for row in &features {
        if row.len() > width {
            width = row.len();
        }
    }
    let mut x_train: Vec<Vec<f64>> = Vec::new();
    for row in features {
        x_train.push(pad(row, width));
    }
    let mut x_test: Vec<Vec<f64>> = Vec::new();
    for row in read_csv(&args[2]) {
        x_test.push(pad(row, width));
    }
    let model = Model::fit(&x_train, &y_train);
    let predictions = model.predict(&x_test);
    write_predictions(&args[3], &predictions);
}
