//! Regression model: mid-rank transform, then normal scores, then GCV-tuned ridge regression.
//!
//! This program trains a regression model on train.csv, in which every row holds the feature
//! columns followed by the target as the last column, and then writes one prediction per line for
//! the rows of test_features.csv. All arithmetic is IEEE-754 double precision in a fixed
//! evaluation order, so repeated runs give identical bits.
//!
//! The feature pipeline has 2 stages, applied in order; each stage is marked "replaced" when it
//! overwrites the columns it receives or "appended" when it keeps them and adds its outputs
//! alongside, and after every stage non-finite cells become 0.0 and each row is capped at 512
//! columns. Stage 1 (mid_rank_transform, replaced) replaces each value by its mid-rank within the
//! training column, i.e. the average of the strict and non-strict ranks scaled to [0, 1], an
//! empirical distribution-function transform. Stage 2 (blom_normal_scores, appended) maps
//! mid-ranks to standard normal quantiles through the plotting position (r + 1/8)/(n + 1/4), a
//! normal-scores transform in the tradition of Blom (1958), computed with Acklam's rational
//! probit approximation.
//!
//! The head (gcv_ridge_regression) is ridge regression on standardized columns whose penalty is
//! chosen by generalized cross-validation, GCV(lambda) = mean squared residual / max(1 -
//! tr(S_lambda)/n, 1e-3)^2 (Golub, Heath & Wahba, 1979), over a fixed 25-point logarithmic grid
//! from 1e-5 to 1e3; when rows and columns both exceed 384 the 384 columns most correlated with
//! the target are kept (a train-only screen), one cyclic-Jacobi eigendecomposition of the smaller
//! of the p x p standardized Gram matrix and the n x n inner-product matrix (they share their
//! nonzero spectrum) scores every grid penalty in closed form, the winning penalty is refit on
//! all training rows, and predictions are clipped to [-60, 60]. Seeded stages draw weights from a
//! permuted congruential generator (PCG XSH-RR 64/32; O'Neill, 2014) keyed by the stage seed and
//! the input/output widths, so training is fully deterministic. Missing or malformed CSV cells
//! parse as 0.0 and short rows are padded with zeros to the training width.
//!
//! Build: rustc -O model.rs
//! Usage: ./model train.csv test_features.csv predictions.csv

#![forbid(unsafe_code)]
#![allow(dead_code)]

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

/// Column j of x in ascending order.
fn sorted_column(x: &[Vec<f64>], j: usize) -> Vec<f64> {
    let mut v: Vec<f64> = x.iter().map(|row| row[j]).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

/// Count of elements strictly less than v (hand-rolled binary search).
fn lower_bound(sorted_values: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = sorted_values.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if sorted_values[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Count of elements less than or equal to v (hand-rolled binary search).
fn upper_bound(sorted_values: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = sorted_values.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if sorted_values[mid] <= v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Average of the strict and non-strict ranks of v, scaled to [0, 1].
fn mid_rank(sorted_values: &[f64], v: f64) -> f64 {
    let lb = lower_bound(sorted_values, v);
    let ub = upper_bound(sorted_values, v);
    (lb + ub) as f64 / (2.0 * sorted_values.len() as f64)
}

/// Standard normal quantile function via Acklam's rational approximation.
fn probit(p: f64) -> f64 {
    let a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    let b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01];
    let c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    let d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= 1.0 - 0.02425 {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
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

/// Eigendecomposition of the symmetric p x p matrix `a` (flattened row-major,
/// overwritten) by cyclic Jacobi rotations (Jacobi, 1846): returns
/// (eigenvalues, eigenvectors) with the eigenvectors flattened row-major so
/// column i holds the unit vector of eigenvalue i. Deterministic: fixed sweep
/// order, at most 20 sweeps, stopping as soon as a whole-matrix check finds
/// the off-diagonal mass under a 1e-12-of-trace threshold.
fn jacobi_eigen(a: &mut [f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    let mut scale = 0.0;
    for i in 0..p {
        scale += a[i * p + i];
    }
    let limit = (scale * 1e-12) * (scale * 1e-12);
    for _ in 0..20 {
        let mut off = 0.0;
        for i in 0..p {
            for j in (i + 1)..p {
                off += a[i * p + j] * a[i * p + j];
            }
        }
        if off <= limit {
            break;
        }
        for i in 0..p {
            for j in (i + 1)..p {
                let apq = a[i * p + j];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[j * p + j] - a[i * p + i]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / ((theta * theta + 1.0).sqrt() - theta)
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..p {
                    if k != i && k != j {
                        let aki = a[k * p + i];
                        let akj = a[k * p + j];
                        a[k * p + i] = c * aki - s * akj;
                        a[i * p + k] = a[k * p + i];
                        a[k * p + j] = s * aki + c * akj;
                        a[j * p + k] = a[k * p + j];
                    }
                }
                let aii = a[i * p + i];
                let ajj = a[j * p + j];
                a[i * p + i] = c * c * aii - 2.0 * s * c * apq + s * s * ajj;
                a[j * p + j] = s * s * aii + 2.0 * s * c * apq + c * c * ajj;
                a[i * p + j] = 0.0;
                a[j * p + i] = 0.0;
                for k in 0..p {
                    let vki = v[k * p + i];
                    let vkj = v[k * p + j];
                    v[k * p + i] = c * vki - s * vkj;
                    v[k * p + j] = s * vki + c * vkj;
                }
            }
        }
    }
    let mut e = vec![0.0; p];
    for i in 0..p {
        e[i] = a[i * p + i];
    }
    (e, v)
}

struct MidRankTransform {
    sorted_cols: Vec<Vec<f64>>,
}

/// This transform replaces each value by its mid-rank within the training column, i.e. the
/// average of the strict and non-strict ranks scaled to [0, 1], an empirical
/// distribution-function transform.
fn mid_rank_transform_fit(x: &[Vec<f64>]) -> MidRankTransform {
    MidRankTransform { sorted_cols: (0..x[0].len()).map(|j| sorted_column(x, j)).collect() }
}

fn mid_rank_transform_apply(state: &MidRankTransform, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| mid_rank(&state.sorted_cols[j], *v))
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}

struct BlomNormalScores {
    sorted_cols: Vec<Vec<f64>>,
}

/// This transform maps mid-ranks to standard normal quantiles through the plotting position (r
/// + 1/8)/(n + 1/4), a normal-scores transform in the tradition of Blom (1958), computed with
/// Acklam's rational probit approximation.
fn blom_normal_scores_fit(x: &[Vec<f64>]) -> BlomNormalScores {
    BlomNormalScores { sorted_cols: (0..x[0].len()).map(|j| sorted_column(x, j)).collect() }
}

fn blom_normal_scores_apply(state: &BlomNormalScores, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let mut values = Vec::new();
        for (j, v) in row.iter().enumerate() {
            let n = state.sorted_cols[j].len() as f64;
            let mid = mid_rank(&state.sorted_cols[j], *v) * n;
            values.push(probit((mid + 0.125) / (n + 0.25)));
        }
        out.push(values);
    }
    finish_stage(x, out, append)
}

struct GcvRidgeRegression {
    cols: Vec<usize>,
    means: Vec<f64>,
    scales: Vec<f64>,
    w: Vec<f64>,
}

/// This head is ridge regression on standardized columns whose penalty is chosen by generalized
/// cross-validation, GCV(lambda) = mean squared residual / max(1 - tr(S_lambda)/n, 1e-3)^2
/// (Golub, Heath & Wahba, 1979), over a fixed 25-point logarithmic grid from 1e-5 to 1e3; when
/// rows and columns both exceed 384 the 384 columns most correlated with the target are kept (a
/// train-only screen), one cyclic-Jacobi eigendecomposition of the smaller of the p x p
/// standardized Gram matrix and the n x n inner-product matrix (they share their nonzero
/// spectrum) scores every grid penalty in closed form, the winning penalty is refit on all
/// training rows, and predictions are clipped to [-60, 60].
fn gcv_ridge_regression_fit(x: &[Vec<f64>], y: &[f64]) -> GcvRidgeRegression {
    let n = x.len().min(y.len());
    let mut p = x[0].len();
    let (mut means, mut scales) = std_fit(x);
    for s in scales.iter_mut() {
        if *s <= 1e-9 {
            *s = 1.0;
        }
    }
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    let mut y_var = 0.0;
    for i in 0..n {
        let d = y[i] - y_mean;
        y_var += d * d;
    }
    let mut cols: Vec<usize> = (0..p).collect();
    if n < 2 || p == 0 || (y_var / n.max(1) as f64).sqrt() < 1e-12 {
        let mut w = vec![0.0; p + 1];
        w[0] = y_mean;
        return GcvRidgeRegression { cols, means, scales, w };
    }
    if n.min(p) > 384 {
        let mut score = vec![0.0; p];
        for r in 0..n {
            let row = &x[r];
            let yr = y[r] - y_mean;
            for j in 0..p {
                score[j] += ((row[j] - means[j]) / scales[j]) * yr;
            }
        }
        for s in score.iter_mut() {
            if *s < 0.0 {
                *s = -*s;
            }
        }
        let mut used = vec![false; p];
        let mut sel = Vec::with_capacity(384);
        for _ in 0..384 {
            let mut best_j = 0;
            let mut best_s = -1.0;
            for j in 0..p {
                if !used[j] && score[j] > best_s {
                    best_s = score[j];
                    best_j = j;
                }
            }
            used[best_j] = true;
            sel.push(best_j);
        }
        sel.sort_unstable();
        cols = sel;
        p = cols.len();
        let kept_means: Vec<f64> = cols.iter().map(|&j| means[j]).collect();
        let kept_scales: Vec<f64> = cols.iter().map(|&j| scales[j]).collect();
        means = kept_means;
        scales = kept_scales;
    }
    let lambdas = [
        1e-05,
        2.1544346900318823e-05,
        4.641588833612782e-05,
        0.0001,
        0.00021544346900318823,
        0.00046415888336127773,
        0.001,
        0.002154434690031882,
        0.004641588833612777,
        0.01,
        0.021544346900318822,
        0.046415888336127774,
        0.1,
        0.21544346900318823,
        0.46415888336127725,
        1.0,
        2.154434690031882,
        4.641588833612772,
        10.0,
        21.54434690031882,
        46.41588833612773,
        100.0,
        215.44346900318823,
        464.1588833612773,
        1000.0,
    ];
    let nf = n as f64;
    if n < p {
        let mut xs = vec![0.0; n * p];
        for r in 0..n {
            let row = &x[r];
            for j in 0..p {
                xs[r * p + j] = (row[cols[j]] - means[j]) / scales[j];
            }
        }
        let mut yc = vec![0.0; n];
        let mut yty = 0.0;
        for r in 0..n {
            yc[r] = y[r] - y_mean;
            yty += yc[r] * yc[r];
        }
        let mut km = vec![0.0; n * n];
        for r in 0..n {
            for s in r..n {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += xs[r * p + j] * xs[s * p + j];
                }
                km[r * n + s] = acc;
            }
        }
        for i in 0..n {
            for j in 0..i {
                km[i * n + j] = km[j * n + i];
            }
        }
        let (mut e, q) = jacobi_eigen(&mut km, n);
        let mut e_max = 0.0;
        for i in 0..n {
            if e[i] > e_max {
                e_max = e[i];
            }
        }
        for i in 0..n {
            if e[i] < e_max * 1e-12 {
                e[i] = 0.0;
            }
        }
        let mut u = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for r in 0..n {
                s += q[r * n + i] * yc[r];
            }
            u[i] = s;
        }
        let mut best_lambda = lambdas[0];
        let mut best_gcv = f64::INFINITY;
        for &lam in lambdas.iter() {
            let mut rss = yty;
            let mut tr = 0.0;
            for i in 0..n {
                if e[i] > 0.0 {
                    let c = u[i] / (e[i] + lam);
                    let f = e[i] * c;
                    rss += f * f - 2.0 * f * u[i];
                    tr += e[i] / (e[i] + lam);
                }
            }
            if rss < 0.0 {
                rss = 0.0;
            }
            let mut denom = 1.0 - tr / nf;
            if denom < 1e-3 {
                denom = 1e-3;
            }
            let gcv = (rss / nf) / (denom * denom);
            if gcv < best_gcv {
                best_gcv = gcv;
                best_lambda = lam;
            }
        }
        let mut t = vec![0.0; n];
        for r in 0..n {
            let mut s = 0.0;
            for i in 0..n {
                if e[i] > 0.0 {
                    s += q[r * n + i] * (u[i] / (e[i] + best_lambda));
                }
            }
            t[r] = s;
        }
        let mut w = Vec::with_capacity(p + 1);
        w.push(y_mean);
        for j in 0..p {
            let mut s = 0.0;
            for r in 0..n {
                s += xs[r * p + j] * t[r];
            }
            w.push(s);
        }
        return GcvRidgeRegression { cols, means, scales, w };
    }
    let mut g = vec![0.0; p * p];
    let mut b = vec![0.0; p];
    let mut yty = 0.0;
    let mut xs_row = vec![0.0; p];
    for r in 0..n {
        let row = &x[r];
        for j in 0..p {
            xs_row[j] = (row[cols[j]] - means[j]) / scales[j];
        }
        let yr = y[r] - y_mean;
        yty += yr * yr;
        for i in 0..p {
            let xi = xs_row[i];
            b[i] += xi * yr;
            for j in i..p {
                g[i * p + j] += xi * xs_row[j];
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            g[i * p + j] = g[j * p + i];
        }
    }
    let (mut e, v) = jacobi_eigen(&mut g, p);
    let mut e_max = 0.0;
    for i in 0..p {
        if e[i] > e_max {
            e_max = e[i];
        }
    }
    for i in 0..p {
        if e[i] < e_max * 1e-12 {
            e[i] = 0.0;
        }
    }
    let mut bt = vec![0.0; p];
    for i in 0..p {
        let mut s = 0.0;
        for j in 0..p {
            s += v[j * p + i] * b[j];
        }
        bt[i] = s;
    }
    let mut best_lambda = lambdas[0];
    let mut best_gcv = f64::INFINITY;
    for &lam in lambdas.iter() {
        let mut rss = yty;
        let mut tr = 0.0;
        for i in 0..p {
            if e[i] > 0.0 {
                let c = bt[i] / (e[i] + lam);
                rss += c * c * e[i] - 2.0 * c * bt[i];
                tr += e[i] / (e[i] + lam);
            }
        }
        if rss < 0.0 {
            rss = 0.0;
        }
        let mut denom = 1.0 - tr / nf;
        if denom < 1e-3 {
            denom = 1e-3;
        }
        let gcv = (rss / nf) / (denom * denom);
        if gcv < best_gcv {
            best_gcv = gcv;
            best_lambda = lam;
        }
    }
    let mut w = Vec::with_capacity(p + 1);
    w.push(y_mean);
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..p {
            if e[i] > 0.0 {
                s += v[j * p + i] * (bt[i] / (e[i] + best_lambda));
            }
        }
        w.push(s);
    }
    GcvRidgeRegression { cols, means, scales, w }
}

fn gcv_ridge_regression_predict(state: &GcvRidgeRegression, x: &[Vec<f64>]) -> Vec<f64> {
    let mut predictions = Vec::new();
    for row in x {
        let mut z = state.w[0];
        for (k, &j) in state.cols.iter().enumerate() {
            if j < row.len() {
                z += state.w[k + 1] * ((row[j] - state.means[k]) / state.scales[k]);
            }
        }
        predictions.push(z.clamp(-60.0, 60.0));
    }
    predictions
}

/// Feature stages applied in order, then the gcv_ridge_regression head.
struct Model {
    stage1: MidRankTransform,
    stage2: BlomNormalScores,
    head: GcvRidgeRegression,
}

impl Model {
    /// Fit every stage in order on the training rows.
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Model {
        if x.is_empty() || y.len() < x.len() {
            eprintln!("training data is empty or target is short");
            std::process::exit(2);
        }
        let x = sanitize(x);
        let stage1 = mid_rank_transform_fit(&x);
        let x = mid_rank_transform_apply(&stage1, &x, false);
        let stage2 = blom_normal_scores_fit(&x);
        let x = blom_normal_scores_apply(&stage2, &x, true);
        let head = gcv_ridge_regression_fit(&x, y);
        Model { stage1, stage2, head }
    }

    /// Apply the fitted stages, then the regression head.
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let x = sanitize(x);
        let x = mid_rank_transform_apply(&self.stage1, &x, false);
        let x = blom_normal_scores_apply(&self.stage2, &x, true);
        gcv_ridge_regression_predict(&self.head, &x)
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
