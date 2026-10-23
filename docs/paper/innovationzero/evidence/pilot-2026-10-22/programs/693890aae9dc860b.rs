//! Regression model: smoothed target bin encoding, then Gaussian-process regression.
//!
//! This program trains a regression model on train.csv, in which every row holds the feature
//! columns followed by the target as the last column, and then writes one prediction per line for
//! the rows of test_features.csv. All arithmetic is IEEE-754 double precision in a fixed
//! evaluation order, so repeated runs give identical bits.
//!
//! The feature pipeline has 1 stage, applied in order; each stage is marked "replaced" when it
//! overwrites the columns it receives or "appended" when it keeps them and adds its outputs
//! alongside, and after every stage non-finite cells become 0.0 and each row is capped at 512
//! columns. Stage 1 (smoothed_target_bin_encoder, appended) replaces each value with the smoothed
//! mean of the target inside its quantile bin (12 bins, additive prior 10.0 toward the global
//! mean), following Micci-Barreca (2001).
//!
//! The head (gaussian_process_ard_regression) is Gaussian-process regression (Rasmussen &
//! Williams, 2006) with a squared-exponential automatic-relevance-determination kernel sf2 *
//! exp(-0.5 * sum_d (x_d - x'_d)^2 / ls_d^2) on features scaled by the inverse training standard
//! deviation; the per-feature lengthscales ls_d, signal variance sf2, and noise variance sn2 are
//! learned by 40 steps of gradient ascent on the log marginal likelihood, whose gradient 0.5 *
//! trace((a a^T - Ky^-1) dKy/dtheta) with a = Ky^-1 y is evaluated with one Cholesky-based
//! inverse per step on at most 256 seeded rows; the predictive coefficients are then solved on up
//! to 300 seeded inducing rows and predictions k(x)^T a are clipped to [-60, 60]. Seeded stages
//! draw weights from a permuted congruential generator (PCG XSH-RR 64/32; O'Neill, 2014) keyed by
//! the stage seed and the input/output widths, so training is fully deterministic. Missing or
//! malformed CSV cells parse as 0.0 and short rows are padded with zeros to the training width.
//!
//! Build: rustc -O model.rs
//! Usage: ./model train.csv test_features.csv predictions.csv

#![forbid(unsafe_code)]
#![allow(dead_code)]

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

/// Column j of x in ascending order.
fn sorted_column(x: &[Vec<f64>], j: usize) -> Vec<f64> {
    let mut v: Vec<f64> = x.iter().map(|row| row[j]).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v
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

/// Interior quantile cut points (1/bins .. (bins-1)/bins) for every column.
fn quantile_edges(x: &[Vec<f64>], bins: usize) -> Vec<Vec<f64>> {
    let mut edges = Vec::new();
    for j in 0..x[0].len() {
        let s = sorted_column(x, j);
        edges.push((1..bins).map(|b| quantile(&s, b as f64 / bins as f64)).collect());
    }
    edges
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

struct SmoothedTargetBinEncoder {
    edges: Vec<Vec<f64>>,
    enc: Vec<Vec<f64>>,
}

/// This transform replaces each value with the smoothed mean of the target inside its quantile
/// bin (12 bins, additive prior 10.0 toward the global mean), following Micci-Barreca (2001).
fn smoothed_target_bin_encoder_fit(x: &[Vec<f64>], y: &[f64], bins: usize, prior: f64) -> SmoothedTargetBinEncoder {
    let edges = quantile_edges(x, bins);
    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= y.len().max(1) as f64;
    let mut enc = Vec::new();
    for j in 0..x[0].len() {
        let mut sums = vec![0.0; bins];
        let mut counts = vec![0.0; bins];
        for r in 0..x.len() {
            let b = lower_bound(&edges[j], x[r][j]).min(bins - 1);
            sums[b] += y[r];
            counts[b] += 1.0;
        }
        enc.push((0..bins).map(|b| (sums[b] + prior * y_mean) / (counts[b] + prior).max(1e-9)).collect());
    }
    SmoothedTargetBinEncoder { edges, enc }
}

fn smoothed_target_bin_encoder_apply(state: &SmoothedTargetBinEncoder, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let mut values = Vec::new();
        for (j, v) in row.iter().enumerate() {
            let bins = state.enc[j].len();
            values.push(state.enc[j][lower_bound(&state.edges[j], *v).min(bins - 1)]);
        }
        out.push(values);
    }
    finish_stage(x, out, append)
}

struct GaussianProcessArdRegression {
    means: Vec<f64>,
    scales: Vec<f64>,
    basis: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    ls: Vec<f64>,
    sf2: f64,
    y_mean: f64,
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

/// This head is Gaussian-process regression (Rasmussen & Williams, 2006) with a
/// squared-exponential automatic-relevance-determination kernel sf2 * exp(-0.5 * sum_d (x_d -
/// x'_d)^2 / ls_d^2) on features scaled by the inverse training standard deviation; the
/// per-feature lengthscales ls_d, signal variance sf2, and noise variance sn2 are learned by 40
/// steps of gradient ascent on the log marginal likelihood, whose gradient 0.5 * trace((a a^T -
/// Ky^-1) dKy/dtheta) with a = Ky^-1 y is evaluated with one Cholesky-based inverse per step on
/// at most 256 seeded rows; the predictive coefficients are then solved on up to 300 seeded
/// inducing rows and predictions k(x)^T a are clipped to [-60, 60].
fn gaussian_process_ard_regression_fit(x: &[Vec<f64>], y: &[f64], m: usize, steps: usize) -> GaussianProcessArdRegression {
    let n = x.len().min(y.len());
    let p = x[0].len();
    let (means, scales) = std_fit(x);
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    if n < 12 || p == 0 {
        return GaussianProcessArdRegression {
            means,
            scales,
            basis: Vec::new(),
            alpha: Vec::new(),
            ls: Vec::new(),
            sf2: 0.0,
            y_mean,
        };
    }
    let basis_size = n.min(m);
    let idx: Vec<usize> = if n > m { seeded_rows(n, m, 909) } else { (0..n).collect() };
    let z: Vec<Vec<f64>> = idx
        .iter()
        .map(|&i| (0..p).map(|j| (x[i][j] - means[j]) / scales[j]).collect())
        .collect();
    let yc: Vec<f64> = idx.iter().map(|&i| y[i] - y_mean).collect();
    let opt_size = basis_size.min(256);
    // Seed each lengthscale from the median pairwise per-feature distance, scaled
    // by sqrt(p) so the summed kernel exponent stays O(1) as p grows.
    let n_init = opt_size.min(96);
    let dim_scale = (p as f64).sqrt();
    let mut log_ls = vec![0.0; p];
    for d in 0..p {
        let mut dists: Vec<f64> = Vec::new();
        for i in 0..n_init {
            for j in (i + 1)..n_init {
                dists.push((z[i][d] - z[j][d]).abs());
            }
        }
        dists.sort_by(|a, b| a.total_cmp(b));
        let med = if dists.is_empty() { 1.0 } else { quantile(&dists, 0.5) };
        log_ls[d] = (med.max(1e-3) * dim_scale).ln();
    }
    let mut var = 0.0;
    for i in 0..opt_size {
        var += yc[i] * yc[i];
    }
    let sy = (var / opt_size.max(1) as f64).sqrt();
    let mut log_sf = sy.max(1e-6).ln();
    let mut log_sn = (0.1 * sy).max(1e-6).ln();
    // Gradient ascent on the log marginal likelihood over the first opt_size rows.
    for _ in 0..steps {
        let mo = opt_size;
        let ls_cur: Vec<f64> = log_ls.iter().map(|u| u.exp()).collect();
        let inv_ls2: Vec<f64> = ls_cur.iter().map(|v| 1.0 / (v * v)).collect();
        let sf2 = (2.0 * log_sf).exp();
        let sn2 = (2.0 * log_sn).exp();
        let mut kmat = vec![0.0; mo * mo];
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
        let mut alpha = vec![0.0; mo];
        for i in 0..mo {
            let mut s = 0.0;
            let base = i * mo;
            for j in 0..mo {
                s += kinv[base + j] * yc[j];
            }
            alpha[i] = s;
        }
        let mut g_sf = 0.0;
        let mut g_sn = 0.0;
        let mut g_ls = vec![0.0; p];
        for i in 0..mo {
            let wii = alpha[i] * alpha[i] - kinv[i * mo + i];
            g_sf += wii * kmat[i * mo + i];
            g_sn += wii;
            let ai = alpha[i];
            for j in (i + 1)..mo {
                let wij = ai * alpha[j] - kinv[i * mo + j];
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
        let scale = 1.0 / mo as f64;
        for d in 0..p {
            let step = (scale * g_ls[d]).clamp(-0.5, 0.5);
            log_ls[d] = (log_ls[d] + step).clamp(-6.0, 6.0);
        }
        let step_sf = (scale * g_sf).clamp(-0.5, 0.5);
        log_sf = (log_sf + step_sf).clamp(-15.0, 15.0);
        let step_sn = (scale * g_sn).clamp(-0.5, 0.5);
        // Floor the noise at exp(-6) of the signal std: this jitter keeps Ky
        // well-conditioned, so the gradient cannot overflow even under an
        // aggressive step.
        log_sn = (log_sn + step_sn).clamp(log_sf - 6.0, log_sf);
    }
    let ls_final: Vec<f64> = log_ls.iter().map(|u| u.exp()).collect();
    let inv_ls2: Vec<f64> = ls_final.iter().map(|v| 1.0 / (v * v)).collect();
    let sf2 = (2.0 * log_sf).exp();
    let sn2 = (2.0 * log_sn).exp();
    let mb = basis_size;
    let mut a = vec![0.0; mb * mb];
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
    let alpha = cholesky_solve(&mut a, &yc, mb);
    GaussianProcessArdRegression { means, scales, basis: z, alpha, ls: ls_final, sf2, y_mean }
}

fn gaussian_process_ard_regression_predict(state: &GaussianProcessArdRegression, x: &[Vec<f64>]) -> Vec<f64> {
    let p = state.means.len();
    if state.basis.is_empty() {
        return x.iter().map(|_| state.y_mean.clamp(-60.0, 60.0)).collect();
    }
    let inv_ls2: Vec<f64> = state.ls.iter().map(|v| 1.0 / (v * v)).collect();
    let mut predictions = Vec::new();
    for row in x {
        let zr: Vec<f64> = (0..p.min(row.len()))
            .map(|j| (row[j] - state.means[j]) / state.scales[j])
            .collect();
        let mut s = state.y_mean;
        for b in 0..state.basis.len() {
            let zb = &state.basis[b];
            let mut r2 = 0.0;
            for d in 0..zr.len().min(zb.len()) {
                let diff = zr[d] - zb[d];
                r2 += diff * diff * inv_ls2[d];
            }
            s += state.alpha[b] * state.sf2 * (-0.5 * r2).exp();
        }
        predictions.push(s.clamp(-60.0, 60.0));
    }
    predictions
}

/// Feature stages applied in order, then the gaussian_process_ard_regression head.
struct Model {
    stage1: SmoothedTargetBinEncoder,
    head: GaussianProcessArdRegression,
}

impl Model {
    /// Fit every stage in order on the training rows.
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Model {
        if x.is_empty() || y.len() < x.len() {
            eprintln!("training data is empty or target is short");
            std::process::exit(2);
        }
        let x = sanitize(x);
        let stage1 = smoothed_target_bin_encoder_fit(&x, y, 12, 10.0);
        let x = smoothed_target_bin_encoder_apply(&stage1, &x, true);
        let head = gaussian_process_ard_regression_fit(&x, y, 300, 40);
        Model { stage1, head }
    }

    /// Apply the fitted stages, then the regression head.
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let x = sanitize(x);
        let x = smoothed_target_bin_encoder_apply(&self.stage1, &x, true);
        gaussian_process_ard_regression_predict(&self.head, &x)
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
