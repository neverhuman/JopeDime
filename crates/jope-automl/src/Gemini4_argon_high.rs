#![forbid(unsafe_code)]
#![allow(dead_code)]

use std::cmp::Ordering;

pub trait RegressorApi: Sized {
    fn new(seed: u64) -> Self;

    fn fit(
        &mut self,
        train: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), String>;

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String>;
}

pub struct Regressor {
    seed: u64,
    fitted: Option<FittedEnsemble>,
}

impl RegressorApi for Regressor {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            fitted: None,
        }
    }

    fn fit(
        &mut self,
        train: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), String> {
        // Invalidate any prior fitted state immediately so failed refits never leave stale state.
        self.fitted = None;

        if n_rows == 0 {
            return Err("n_rows must be at least 1".to_string());
        }
        if n_cols == 0 {
            return Err("n_cols must be at least 1".to_string());
        }
        let expected_len = n_rows
            .checked_mul(n_cols)
            .ok_or_else(|| "Dimension multiplication overflow in fit".to_string())?;
        if train.len() != expected_len {
            return Err(format!(
                "train.len() ({}) does not match n_rows * n_cols ({})",
                train.len(),
                expected_len
            ));
        }

        let n_features = n_cols - 1;
        let mut x_raw = Vec::new();
        if n_features > 0 {
            let feat_len = n_rows
                .checked_mul(n_features)
                .ok_or_else(|| "Feature matrix size overflow".to_string())?;
            x_raw = vec![0.0_f64; feat_len];
        }
        let mut y_raw = vec![0.0_f64; n_rows];

        for i in 0..n_rows {
            let row_offset = i * n_cols;
            for j in 0..n_features {
                let val = train[row_offset + j];
                if val.is_infinite() {
                    return Err(format!("Infinite feature value at row {}, col {}", i, j));
                }
                x_raw[i * n_features + j] = val;
            }
            let target = train[row_offset + n_features];
            if !target.is_finite() {
                return Err(format!("Non-finite target value at row {}", i));
            }
            y_raw[i] = target;
        }

        let mut rng = Rng::new(self.seed);
        let ensemble = FittedEnsemble::train(&x_raw, &y_raw, n_rows, n_features, &mut rng)?;
        self.fitted = Some(ensemble);
        Ok(())
    }

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String> {
        let model = self
            .fitted
            .as_ref()
            .ok_or_else(|| "Regressor has not been fitted".to_string())?;

        if n_features != model.n_features {
            return Err(format!(
                "Prediction n_features ({}) does not match fitted feature count ({})",
                n_features, model.n_features
            ));
        }

        let expected_len = n_rows
            .checked_mul(n_features)
            .ok_or_else(|| "Dimension multiplication overflow in predict".to_string())?;
        if features.len() != expected_len {
            return Err(format!(
                "features.len() ({}) does not match n_rows * n_features ({})",
                features.len(),
                expected_len
            ));
        }

        for (idx, &v) in features.iter().enumerate() {
            if v.is_infinite() {
                return Err(format!("Infinite feature value in predict input at index {}", idx));
            }
        }

        if n_rows == 0 {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(n_rows);
        if n_features == 0 {
            let val = model.predict_single_row(&[]);
            out.resize(n_rows, val);
            return Ok(out);
        }

        for i in 0..n_rows {
            let start = i * n_features;
            let row_slice = &features[start..start + n_features];
            let pred = model.predict_single_row(row_slice);
            out.push(pred);
        }

        Ok(out)
    }
}

// ============================================================================
// Section 1: Permitted Scalar Mathematical Helpers & Deterministic PRNG
// ============================================================================

#[inline]
fn scalar_clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[inline]
fn scalar_sign(x: f64) -> f64 {
    // Convention: 0.0 (and -0.0 or NaN) maps to 0.0; >0 maps to 1.0; <0 maps to -1.0.
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[inline]
fn scalar_step(x: f64, threshold: f64) -> f64 {
    // Convention: 1.0 if x >= threshold, else 0.0.
    if x >= threshold {
        1.0
    } else {
        0.0
    }
}

#[inline]
fn scalar_sigmoid(x: f64) -> f64 {
    let z = scalar_clip(x, -40.0, 40.0);
    1.0 / (1.0 + (-z).exp())
}

#[inline]
fn scalar_erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 rational approximation using permitted operations.
    if !x.is_finite() {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = scalar_clip(x.abs(), 0.0, 12.0);
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let t = 1.0 / (1.0 + p * ax);
    let poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    let y = 1.0 - poly * (-ax * ax).exp();
    sign * y
}

#[inline]
fn scalar_sinh(x: f64) -> f64 {
    let z = scalar_clip(x, -25.0, 25.0);
    if z.abs() < 1e-5 {
        let z2 = z * z;
        z * (1.0 + z2 / 6.0)
    } else {
        0.5 * (z.exp() - (-z).exp())
    }
}

#[inline]
fn scalar_cosh(x: f64) -> f64 {
    let z = scalar_clip(x, -25.0, 25.0);
    0.5 * (z.exp() + (-z).exp())
}

#[inline]
fn safe_div(num: f64, den: f64, eps: f64) -> f64 {
    if den.abs() < eps {
        if den >= 0.0 {
            num / eps
        } else {
            -num / eps
        }
    } else {
        num / den
    }
}

#[derive(Clone, Debug)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut rng = Self {
            state: if s == 0 { 0xBF58476D1CE4E5B9 } else { s },
        };
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        rng
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            0
        } else {
            (self.next_u64() % (bound as u64)) as usize
        }
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        if len <= 1 {
            return;
        }
        for i in (1..len).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

// ============================================================================
// Section 2: Statistical Summaries, Sorting, and Linear Algebra Solvers
// ============================================================================

#[inline]
fn sort_f64(slice: &mut [f64]) {
    slice.sort_by(|a, b| a.total_cmp(b));
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let qc = scalar_clip(q, 0.0, 1.0);
    let pos = qc * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn mean_and_std(values: &[f64]) -> (f64, f64) {
    let mut count = 0usize;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    for &v in values {
        if v.is_finite() {
            count += 1;
            let delta = v - mean;
            mean += delta / (count as f64);
            let delta2 = v - mean;
            m2 += delta * delta2;
        }
    }
    if count == 0 {
        return (0.0, 1.0);
    }
    if count == 1 {
        return (mean, 1.0);
    }
    let var = (m2 / (count as f64)).max(0.0);
    let std = var.sqrt();
    (mean, if std > 1e-12 { std } else { 1.0 })
}

fn compute_rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() || y_true.len() != y_pred.len() {
        return f64::INFINITY;
    }
    let mut sum_sq = 0.0_f64;
    for i in 0..y_true.len() {
        let diff = y_pred[i] - y_true[i];
        sum_sq += diff * diff;
    }
    (sum_sq / (y_true.len() as f64)).sqrt()
}

/// Solves (A + diag_reg * I) x = b for symmetric positive semi-definite A (d x d)
/// using Cholesky decomposition with automatic diagonal loading fallback.
fn solve_spd_cholesky(a_mat: &[f64], b_vec: &[f64], d: usize, base_reg: f64) -> Vec<f64> {
    if d == 0 {
        return Vec::new();
    }
    let mut l_mat = vec![0.0_f64; d * d];
    let mut reg = base_reg.max(1e-10);

    for attempt in 0..6 {
        let mut ok = true;
        for v in l_mat.iter_mut() {
            *v = 0.0;
        }
        for i in 0..d {
            for j in 0..=i {
                let mut sum = a_mat[i * d + j];
                if i == j {
                    sum += reg;
                }
                for k in 0..j {
                    sum -= l_mat[i * d + k] * l_mat[j * d + k];
                }
                if i == j {
                    if !sum.is_finite() || sum <= 1e-13 {
                        ok = false;
                        break;
                    }
                    l_mat[i * d + j] = sum.sqrt();
                } else {
                    let diag = l_mat[j * d + j];
                    l_mat[i * d + j] = sum / diag;
                }
            }
            if !ok {
                break;
            }
        }
        if ok {
            // Forward substitution: L y = b
            let mut y = vec![0.0_f64; d];
            for i in 0..d {
                let mut sum = b_vec[i];
                for k in 0..i {
                    sum -= l_mat[i * d + k] * y[k];
                }
                y[i] = sum / l_mat[i * d + i];
            }
            // Backward substitution: L^T x = y
            let mut x = vec![0.0_f64; d];
            for i in (0..d).rev() {
                let mut sum = y[i];
                for k in (i + 1)..d {
                    sum -= l_mat[k * d + i] * x[k];
                }
                let val = sum / l_mat[i * d + i];
                x[i] = if val.is_finite() { val } else { 0.0 };
            }
            return x;
        }
        reg *= if attempt == 0 { 10.0 } else { 25.0 };
    }

    // Diagonal fallback if matrix is severely ill-conditioned
    let mut x = vec![0.0_f64; d];
    for i in 0..d {
        let denom = a_mat[i * d + i].abs() + reg.max(1.0);
        let val = b_vec[i] / denom;
        x[i] = if val.is_finite() { val } else { 0.0 };
    }
    x
}

// ============================================================================
// Section 3: Feature Pipeline, Row Summaries, and Nonlinear Interactions
// ============================================================================

#[derive(Clone, Debug)]
struct FeatureStats {
    orig_idx: usize,
    median: f64,
    mean: f64,
    std: f64,
    iqr_scale: f64,
    min_val: f64,
    max_val: f64,
    missing_rate: f64,
    has_missing: bool,
    is_skewed: bool,
    knots: [f64; 4], // Quantile knots on standardized scale
    relevance: f64,
}

#[derive(Clone, Debug)]
struct PairInteractionSpec {
    slot_a: usize, // index into active_features
    slot_b: usize, // index into active_features
    kind: u8,      // 0: diff, 1: sum, 2: asinh(prod), 3: asinh(ratio)
}

#[derive(Clone, Debug)]
struct FeaturePipeline {
    n_orig_features: usize,
    active_features: Vec<FeatureStats>,
    use_row_summaries: bool,
    tree_interactions: Vec<PairInteractionSpec>,
    ridge_interactions: Vec<(usize, usize)>, // (slot_a, slot_b) product terms
    ridge_spline_slots: Vec<usize>,          // active feature slots receiving 4-knot splines
    ridge_periodic_slots: Vec<usize>,        // active feature slots receiving sin/cos probes
}

impl FeaturePipeline {
    fn fit(
        x: &[f64],
        y: &[f64],
        row_indices: &[usize],
        n_features: usize,
        max_active_features: usize,
    ) -> Self {
        let n = row_indices.len();
        if n_features == 0 || n == 0 {
            return Self {
                n_orig_features: n_features,
                active_features: Vec::new(),
                use_row_summaries: false,
                tree_interactions: Vec::new(),
                ridge_interactions: Vec::new(),
                ridge_spline_slots: Vec::new(),
                ridge_periodic_slots: Vec::new(),
            };
        }

        let (y_mean, y_std) = {
            let mut ys = Vec::with_capacity(n);
            for &r in row_indices {
                ys.push(y[r]);
            }
            mean_and_std(&ys)
        };

        // For large n, subsample up to 2048 rows deterministically to compute quantiles and relevance rapidly.
        let sample_step = if n > 2048 { n / 1024 } else { 1 };
        let mut candidates = Vec::new();
        let mut col_vals = Vec::with_capacity((n / sample_step) + 2);

        for j in 0..n_features {
            col_vals.clear();
            let mut missing_cnt = 0usize;
            let mut total_cnt = 0usize;
            let mut sum = 0.0_f64;
            let mut sum_sq = 0.0_f64;

            for (idx_pos, &r) in row_indices.iter().enumerate() {
                if sample_step > 1 && (idx_pos % sample_step != 0) {
                    continue;
                }
                total_cnt += 1;
                let v = x[r * n_features + j];
                if v.is_nan() {
                    missing_cnt += 1;
                } else {
                    col_vals.push(v);
                    sum += v;
                    sum_sq += v * v;
                }
            }

            let obs_cnt = col_vals.len();
            if obs_cnt < 2 {
                continue;
            }
            sort_f64(&mut col_vals);
            let min_val = col_vals[0];
            let max_val = col_vals[obs_cnt - 1];
            let mean = sum / (obs_cnt as f64);
            if (max_val - min_val).abs() <= 1e-13 * mean.abs().max(1.0) {
                continue;
            }

            let var = ((sum_sq / (obs_cnt as f64)) - mean * mean).max(0.0);
            let std = var.sqrt().max(1e-12);
            let median = quantile_sorted(&col_vals, 0.5);
            let q25 = quantile_sorted(&col_vals, 0.25);
            let q75 = quantile_sorted(&col_vals, 0.75);
            let iqr_scale = ((q75 - q25) / 1.349).max(std * 0.25).max(1e-12);
            let q10 = quantile_sorted(&col_vals, 0.10);
            let q90 = quantile_sorted(&col_vals, 0.90);
            let is_skewed = (max_val - median).abs() > 3.5 * (median - min_val).abs().max(iqr_scale)
                || (median - min_val).abs() > 3.5 * (max_val - median).abs().max(iqr_scale)
                || (q90 - q10) > 4.0 * (q75 - q25).max(1e-9);

            let knots = [
                scalar_clip((quantile_sorted(&col_vals, 0.20) - mean) / std, -5.0, 5.0),
                scalar_clip((quantile_sorted(&col_vals, 0.40) - mean) / std, -5.0, 5.0),
                scalar_clip((quantile_sorted(&col_vals, 0.60) - mean) / std, -5.0, 5.0),
                scalar_clip((quantile_sorted(&col_vals, 0.80) - mean) / std, -5.0, 5.0),
            ];

            // Compute linear correlation + 8-bin nonlinear ANOVA eta-squared with target y
            let mut cov = 0.0_f64;
            let mut bin_sum = [0.0_f64; 9];
            let mut bin_cnt = [0usize; 9];
            let bin_edges = [
                quantile_sorted(&col_vals, 0.125),
                quantile_sorted(&col_vals, 0.250),
                quantile_sorted(&col_vals, 0.375),
                quantile_sorted(&col_vals, 0.500),
                quantile_sorted(&col_vals, 0.625),
                quantile_sorted(&col_vals, 0.750),
                quantile_sorted(&col_vals, 0.875),
            ];

            let mut used_rows = 0usize;
            for (idx_pos, &r) in row_indices.iter().enumerate() {
                if sample_step > 1 && (idx_pos % sample_step != 0) {
                    continue;
                }
                used_rows += 1;
                let v = x[r * n_features + j];
                let yn = (y[r] - y_mean) / y_std;
                if v.is_nan() {
                    bin_sum[8] += yn;
                    bin_cnt[8] += 1;
                } else {
                    let xn = scalar_clip((v - mean) / std, -8.0, 8.0);
                    cov += xn * yn;
                    let mut b = 0usize;
                    while b < 7 && v > bin_edges[b] {
                        b += 1;
                    }
                    bin_sum[b] += yn;
                    bin_cnt[b] += 1;
                }
            }

            let corr = (cov / (used_rows.max(1) as f64)).abs().min(1.0);
            let mut between_ss = 0.0_f64;
            for b in 0..9 {
                if bin_cnt[b] > 0 {
                    let bm = bin_sum[b] / (bin_cnt[b] as f64);
                    between_ss += (bin_cnt[b] as f64) * bm * bm;
                }
            }
            let eta = (between_ss / (used_rows.max(1) as f64)).max(0.0).sqrt().min(1.0);
            let relevance = 0.45 * corr + 0.55 * eta + 1e-6;

            candidates.push(FeatureStats {
                orig_idx: j,
                median,
                mean,
                std,
                iqr_scale,
                min_val,
                max_val,
                missing_rate: (missing_cnt as f64) / (total_cnt.max(1) as f64),
                has_missing: missing_cnt > 0,
                is_skewed,
                knots,
                relevance,
            });
        }

        // Sort candidates by descending relevance, tie-breaking by original feature index
        candidates.sort_by(|a, b| {
            b.relevance
                .total_cmp(&a.relevance)
                .then_with(|| a.orig_idx.cmp(&b.orig_idx))
        });
        if candidates.len() > max_active_features {
            candidates.truncate(max_active_features);
        }

        let n_active = candidates.len();
        let use_row_summaries = n_active >= 2;

        // Build targeted pairwise interactions among the top features
        let top_pair_pool = n_active.min(10);
        let mut scored_pairs: Vec<(f64, usize, usize)> = Vec::new();
        for a in 0..top_pair_pool {
            for b in (a + 1)..top_pair_pool {
                let fa = &candidates[a];
                let fb = &candidates[b];
                // Score interaction alignment with target on sampled rows
                let mut sum_prod_y = 0.0_f64;
                let mut sum_diff_y = 0.0_f64;
                let mut cnt = 0usize;
                for (idx_pos, &r) in row_indices.iter().enumerate() {
                    if sample_step > 1 && (idx_pos % sample_step != 0) {
                        continue;
                    }
                    let va = x[r * n_features + fa.orig_idx];
                    let vb = x[r * n_features + fb.orig_idx];
                    let xa = if va.is_nan() {
                        0.0
                    } else {
                        scalar_clip((va - fa.mean) / fa.std, -6.0, 6.0)
                    };
                    let xb = if vb.is_nan() {
                        0.0
                    } else {
                        scalar_clip((vb - fb.mean) / fb.std, -6.0, 6.0)
                    };
                    let yn = (y[r] - y_mean) / y_std;
                    sum_prod_y += (xa * xb).asinh() * yn;
                    sum_diff_y += (xa - xb) * yn;
                    cnt += 1;
                }
                let denom = (cnt.max(1) as f64).sqrt();
                let pair_score = (sum_prod_y.abs() + 0.7 * sum_diff_y.abs()) / denom
                    + 0.25 * (fa.relevance + fb.relevance);
                scored_pairs.push((pair_score, a, b));
            }
        }
        scored_pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut tree_interactions = Vec::new();
        let max_tree_pairs = if n_active <= 16 { 12 } else { 8 };
        for &(_, a, b) in scored_pairs.iter().take(max_tree_pairs) {
            tree_interactions.push(PairInteractionSpec {
                slot_a: a,
                slot_b: b,
                kind: 0, // diff
            });
            tree_interactions.push(PairInteractionSpec {
                slot_a: a,
                slot_b: b,
                kind: 1, // sum
            });
            tree_interactions.push(PairInteractionSpec {
                slot_a: a,
                slot_b: b,
                kind: 2, // asinh(prod)
            });
            if tree_interactions.len() < 28 {
                tree_interactions.push(PairInteractionSpec {
                    slot_a: a,
                    slot_b: b,
                    kind: 3, // protected ratio
                });
            }
        }

        let mut ridge_interactions = Vec::new();
        for &(_, a, b) in scored_pairs.iter().take(18) {
            ridge_interactions.push((a, b));
        }

        let spline_count = n_active.min(20);
        let ridge_spline_slots: Vec<usize> = (0..spline_count).collect();
        let periodic_count = if n_active <= 24 { n_active.min(8) } else { 4 };
        let ridge_periodic_slots: Vec<usize> = (0..periodic_count).collect();

        Self {
            n_orig_features: n_features,
            active_features: candidates,
            use_row_summaries,
            tree_interactions,
            ridge_interactions,
            ridge_spline_slots,
            ridge_periodic_slots,
        }
    }

    #[inline]
    fn tree_feature_count(&self) -> usize {
        let base = self.active_features.len();
        let summaries = if self.use_row_summaries { 8 } else { 0 };
        base + summaries + self.tree_interactions.len()
    }

    /// Computes all 7 row summaries from Section 5.4 + missingness fraction over standardized active features.
    #[inline]
    fn compute_row_summaries(&self, orig_row: &[f64]) -> [f64; 8] {
        let mut r_sum = 0.0_f64;
        let mut r_l1 = 0.0_f64;
        let mut r_l2_sq = 0.0_f64;
        let mut r_min = f64::INFINITY;
        let mut r_max = f64::NEG_INFINITY;
        let mut obs_count = 0usize;
        let mut nan_count = 0usize;

        for fs in &self.active_features {
            let v = orig_row[fs.orig_idx];
            if v.is_nan() {
                nan_count += 1;
            } else {
                let z = scalar_clip((v - fs.mean) / fs.std, -12.0, 12.0);
                r_sum += z;
                r_l1 += z.abs();
                r_l2_sq += z * z;
                if z < r_min {
                    r_min = z;
                }
                if z > r_max {
                    r_max = z;
                }
                obs_count += 1;
            }
        }

        let total = self.active_features.len().max(1) as f64;
        let missing_frac = (nan_count as f64) / total;

        if obs_count == 0 {
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, missing_frac];
        }

        let n_obs_f = obs_count as f64;
        let r_mean = r_sum / n_obs_f;
        let r_l2 = r_l2_sq.sqrt();
        let var = ((r_l2_sq / n_obs_f) - r_mean * r_mean).max(0.0);
        let r_std = var.sqrt();

        [
            r_sum,
            r_mean,
            r_l1,
            r_l2,
            r_min,
            r_max,
            r_std,
            missing_frac,
        ]
    }

    fn extract_tree_row(&self, orig_row: &[f64], out: &mut [f64]) {
        let n_act = self.active_features.len();
        for (slot, fs) in self.active_features.iter().enumerate() {
            out[slot] = orig_row[fs.orig_idx];
        }
        let mut offset = n_act;
        if self.use_row_summaries {
            let summaries = self.compute_row_summaries(orig_row);
            for s in 0..8 {
                out[offset + s] = summaries[s];
            }
            offset += 8;
        }
        for (k, spec) in self.tree_interactions.iter().enumerate() {
            let fa = &self.active_features[spec.slot_a];
            let fb = &self.active_features[spec.slot_b];
            let va = orig_row[fa.orig_idx];
            let vb = orig_row[fb.orig_idx];
            if va.is_nan() || vb.is_nan() {
                out[offset + k] = f64::NAN;
            } else {
                let za = scalar_clip((va - fa.mean) / fa.std, -10.0, 10.0);
                let zb = scalar_clip((vb - fb.mean) / fb.std, -10.0, 10.0);
                let val = match spec.kind {
                    0 => za - zb,
                    1 => za + zb,
                    2 => (za * zb).asinh(),
                    _ => (za / (zb * zb + 0.25).sqrt()).asinh(),
                };
                out[offset + k] = val;
            }
        }
    }

    fn ridge_basis_dim(&self) -> usize {
        let n_act = self.active_features.len();
        if n_act == 0 {
            return 0;
        }
        // Per active feature: standardized linear + asinh compressed + missing indicator (if has_missing)
        let mut dim = 0usize;
        for fs in &self.active_features {
            dim += 2;
            if fs.has_missing {
                dim += 1;
            }
        }
        // 4 quantile hinge knots + tanh saturation per spline slot
        dim += self.ridge_spline_slots.len() * 5;
        // sin & cos probes per periodic slot
        dim += self.ridge_periodic_slots.len() * 2;
        // Row summaries
        if self.use_row_summaries {
            dim += 8;
        }
        // Pairwise product & difference terms
        dim += self.ridge_interactions.len() * 2;
        dim
    }

    fn extract_ridge_basis(&self, orig_row: &[f64], out: &mut [f64]) {
        let n_act = self.active_features.len();
        if n_act == 0 {
            return;
        }
        let mut idx = 0usize;
        for fs in &self.active_features {
            let v = orig_row[fs.orig_idx];
            let is_miss = v.is_nan();
            let v_imp = if is_miss { fs.median } else { v };
            let z_lin = scalar_clip((v_imp - fs.mean) / fs.std, -10.0, 10.0);
            let z_asinh = ((v_imp - fs.median) / fs.iqr_scale).asinh();
            out[idx] = z_lin;
            idx += 1;
            out[idx] = z_asinh;
            idx += 1;
            if fs.has_missing {
                out[idx] = scalar_step(if is_miss { 1.0 } else { 0.0 }, 0.5);
                idx += 1;
            }
        }

        for &slot in &self.ridge_spline_slots {
            let fs = &self.active_features[slot];
            let v = orig_row[fs.orig_idx];
            let v_imp = if v.is_nan() { fs.median } else { v };
            let z = scalar_clip((v_imp - fs.mean) / fs.std, -8.0, 8.0);
            for k in 0..4 {
                let knot = fs.knots[k];
                out[idx] = (z - knot).max(0.0);
                idx += 1;
            }
            out[idx] = (0.5 * z).tanh();
            idx += 1;
        }

        for &slot in &self.ridge_periodic_slots {
            let fs = &self.active_features[slot];
            let v = orig_row[fs.orig_idx];
            let v_imp = if v.is_nan() { fs.median } else { v };
            let z = scalar_clip((v_imp - fs.mean) / fs.std, -12.0, 12.0);
            out[idx] = z.sin();
            idx += 1;
            out[idx] = z.cos();
            idx += 1;
        }

        if self.use_row_summaries {
            let summaries = self.compute_row_summaries(orig_row);
            for s in 0..8 {
                out[idx] = scalar_clip(summaries[s], -15.0, 15.0);
                idx += 1;
            }
        }

        for &(a, b) in &self.ridge_interactions {
            let fa = &self.active_features[a];
            let fb = &self.active_features[b];
            let va = orig_row[fa.orig_idx];
            let vb = orig_row[fb.orig_idx];
            let za = scalar_clip(
                ((if va.is_nan() { fa.median } else { va }) - fa.mean) / fa.std,
                -6.0,
                6.0,
            );
            let zb = scalar_clip(
                ((if vb.is_nan() { fb.median } else { vb }) - fb.mean) / fb.std,
                -6.0,
                6.0,
            );
            out[idx] = (za * zb).asinh();
            idx += 1;
            out[idx] = (za - zb).abs();
            idx += 1;
        }
    }
}

// ============================================================================
// Section 4: Expert 1 — Penalized Spline & Interaction Ridge Regressor
// ============================================================================

#[derive(Clone, Debug)]
struct SplineRidgeModel {
    basis_dim: usize,
    col_means: Vec<f64>,
    col_stds: Vec<f64>,
    weights: Vec<f64>,
    intercept: f64,
    y_min: f64,
    y_max: f64,
}

impl SplineRidgeModel {
    fn fit(
        x_orig: &[f64],
        y: &[f64],
        row_indices: &[usize],
        n_orig_features: usize,
        pipeline: &FeaturePipeline,
    ) -> Self {
        let n = row_indices.len();
        let d = pipeline.ridge_basis_dim();
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut y_sum = 0.0_f64;
        for &r in row_indices {
            let yr = y[r];
            if yr < y_min {
                y_min = yr;
            }
            if yr > y_max {
                y_max = yr;
            }
            y_sum += yr;
        }
        let y_mean = if n > 0 { y_sum / (n as f64) } else { 0.0 };

        if d == 0 || n < 3 {
            return Self {
                basis_dim: 0,
                col_means: Vec::new(),
                col_stds: Vec::new(),
                weights: Vec::new(),
                intercept: y_mean,
                y_min,
                y_max,
            };
        }

        // Subsample if n is very large to keep O(n * d^2) normal matrix construction fast
        let max_ridge_rows = 6000usize;
        let stride = if n > max_ridge_rows {
            n / max_ridge_rows
        } else {
            1
        };
        let mut sub_rows = Vec::with_capacity((n / stride) + 1);
        for (pos, &r) in row_indices.iter().enumerate() {
            if stride == 1 || pos % stride == 0 {
                sub_rows.push(r);
            }
        }
        let ns = sub_rows.len();

        let mut phi = vec![0.0_f64; ns * d];
        let mut row_buf = vec![0.0_f64; d];
        let mut y_sub = Vec::with_capacity(ns);
        let mut y_sub_sum = 0.0_f64;

        for (i, &r) in sub_rows.iter().enumerate() {
            let orig_row = &x_orig[r * n_orig_features..(r + 1) * n_orig_features];
            pipeline.extract_ridge_basis(orig_row, &mut row_buf);
            phi[i * d..(i + 1) * d].copy_from_slice(&row_buf);
            let yr = y[r];
            y_sub.push(yr);
            y_sub_sum += yr;
        }
        let intercept = y_sub_sum / (ns as f64);

        let mut col_means = vec![0.0_f64; d];
        let mut col_stds = vec![1.0_f64; d];
        for j in 0..d {
            let mut s = 0.0_f64;
            for i in 0..ns {
                s += phi[i * d + j];
            }
            let m = s / (ns as f64);
            col_means[j] = m;
            let mut sq = 0.0_f64;
            for i in 0..ns {
                let diff = phi[i * d + j] - m;
                sq += diff * diff;
            }
            let st = (sq / (ns as f64)).sqrt();
            col_stds[j] = if st > 1e-10 { st } else { 1.0 };
            for i in 0..ns {
                phi[i * d + j] = if st > 1e-10 {
                    (phi[i * d + j] - m) / st
                } else {
                    0.0
                };
            }
        }

        // Split sub_rows into 80% internal train and 20% internal validation for lambda selection
        let lambdas = [
            1e-4_f64, 1e-3, 1e-2, 0.05, 0.2, 0.8, 3.0, 12.0, 50.0, 200.0, 1000.0,
        ];
        let best_lambda = if ns >= 12 {
            let mut g_tr = vec![0.0_f64; d * d];
            let mut b_tr = vec![0.0_f64; d];
            let mut val_indices = Vec::new();
            let mut tr_cnt = 0usize;

            for i in 0..ns {
                if i % 5 == 2 {
                    val_indices.push(i);
                } else {
                    tr_cnt += 1;
                    let yc = y_sub[i] - intercept;
                    let row_p = &phi[i * d..(i + 1) * d];
                    for c1 in 0..d {
                        let v1 = row_p[c1];
                        b_tr[c1] += v1 * yc;
                        for c2 in 0..=c1 {
                            g_tr[c1 * d + c2] += v1 * row_p[c2];
                        }
                    }
                }
            }
            let scale_tr = 1.0 / (tr_cnt.max(1) as f64);
            for c1 in 0..d {
                b_tr[c1] *= scale_tr;
                for c2 in 0..=c1 {
                    let v = g_tr[c1 * d + c2] * scale_tr;
                    g_tr[c1 * d + c2] = v;
                    g_tr[c2 * d + c1] = v;
                }
            }

            let mut best_lam = 1.0_f64;
            let mut best_mse = f64::INFINITY;
            for &lam in &lambdas {
                let w_cand = solve_spd_cholesky(&g_tr, &b_tr, d, lam);
                let mut mse = 0.0_f64;
                for &vi in &val_indices {
                    let row_p = &phi[vi * d..(vi + 1) * d];
                    let mut dot = intercept;
                    for c in 0..d {
                        dot += row_p[c] * w_cand[c];
                    }
                    let err = dot - y_sub[vi];
                    mse += err * err;
                }
                if mse < best_mse {
                    best_mse = mse;
                    best_lam = lam;
                }
            }
            best_lam
        } else {
            5.0
        };

        // Fit final weights on all ns rows using best_lambda
        let mut g_full = vec![0.0_f64; d * d];
        let mut b_full = vec![0.0_f64; d];
        let scale_full = 1.0 / (ns as f64);
        for i in 0..ns {
            let yc = y_sub[i] - intercept;
            let row_p = &phi[i * d..(i + 1) * d];
            for c1 in 0..d {
                let v1 = row_p[c1];
                b_full[c1] += v1 * yc;
                for c2 in 0..=c1 {
                    g_full[c1 * d + c2] += v1 * row_p[c2];
                }
            }
        }
        for c1 in 0..d {
            b_full[c1] *= scale_full;
            for c2 in 0..=c1 {
                let v = g_full[c1 * d + c2] * scale_full;
                g_full[c1 * d + c2] = v;
                g_full[c2 * d + c1] = v;
            }
        }

        let weights = solve_spd_cholesky(&g_full, &b_full, d, best_lambda);

        Self {
            basis_dim: d,
            col_means,
            col_stds,
            weights,
            intercept,
            y_min,
            y_max,
        }
    }

    fn predict_row(&self, orig_row: &[f64], pipeline: &FeaturePipeline, basis_buf: &mut [f64]) -> f64 {
        if self.basis_dim == 0 || self.weights.is_empty() {
            return self.intercept;
        }
        pipeline.extract_ridge_basis(orig_row, basis_buf);
        let mut sum = self.intercept;
        for j in 0..self.basis_dim {
            let z = scalar_clip((basis_buf[j] - self.col_means[j]) / self.col_stds[j], -12.0, 12.0);
            sum += z * self.weights[j];
        }
        // Soft trust-region tail damping outside training range to prevent leverage blow-up
        let span = (self.y_max - self.y_min).abs().max(1e-6);
        let lo = self.y_min - 0.75 * span;
        let hi = self.y_max + 0.75 * span;
        if sum < lo {
            lo - 0.5 * span * ((lo - sum) / (0.5 * span)).asinh()
        } else if sum > hi {
            hi + 0.5 * span * ((sum - hi) / (0.5 * span)).asinh()
        } else {
            sum
        }
    }
}

// ============================================================================
// Section 5: Histogram Quantization & Second-Order Gradient Boosting Engines
// ============================================================================

const MAX_BINS: usize = 64;
const NAN_BIN: u8 = 64;
const TOTAL_BINS: usize = 65;

#[derive(Clone, Debug)]
struct FeatureBinner {
    edges: Vec<f64>,
}

impl FeatureBinner {
    fn from_column(vals: &mut Vec<f64>, max_bins: usize) -> Self {
        vals.retain(|v| !v.is_nan() && v.is_finite());
        if vals.is_empty() {
            return Self { edges: Vec::new() };
        }
        sort_f64(vals);
        let mut edges = Vec::with_capacity(max_bins);
        for b in 1..max_bins {
            let q = (b as f64) / (max_bins as f64);
            let e = quantile_sorted(vals, q);
            if edges.last().map_or(true, |&last| e > last + 1e-13 * e.abs().max(1.0)) {
                edges.push(e);
            }
        }
        Self { edges }
    }

    #[inline]
    fn bin_value(&self, v: f64) -> u8 {
        if v.is_nan() {
            return NAN_BIN;
        }
        // Binary search in edges
        let edges = &self.edges;
        let mut lo = 0usize;
        let mut hi = edges.len();
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if v <= edges[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo as u8
    }
}

#[derive(Clone, Debug)]
struct HistogramQuantizer {
    binners: Vec<FeatureBinner>,
    n_tree_features: usize,
}

impl HistogramQuantizer {
    fn fit(
        x_orig: &[f64],
        row_indices: &[usize],
        n_orig_features: usize,
        pipeline: &FeaturePipeline,
    ) -> Self {
        let n_tree_features = pipeline.tree_feature_count();
        let n = row_indices.len();
        if n_tree_features == 0 || n == 0 {
            return Self {
                binners: Vec::new(),
                n_tree_features: 0,
            };
        }

        let stride = if n > 2048 { n / 1024 } else { 1 };
        let n_samp = (n / stride) + 1;
        let mut samp_mat = vec![0.0_f64; n_samp * n_tree_features];
        let mut actual_samp = 0usize;
        let mut row_buf = vec![0.0_f64; n_tree_features];

        for (pos, &r) in row_indices.iter().enumerate() {
            if stride == 1 || pos % stride == 0 {
                if actual_samp < n_samp {
                    let orig_row = &x_orig[r * n_orig_features..(r + 1) * n_orig_features];
                    pipeline.extract_tree_row(orig_row, &mut row_buf);
                    samp_mat[actual_samp * n_tree_features..(actual_samp + 1) * n_tree_features]
                        .copy_from_slice(&row_buf);
                    actual_samp += 1;
                }
            }
        }

        let mut binners = Vec::with_capacity(n_tree_features);
        let mut col_buf = Vec::with_capacity(actual_samp);
        for j in 0..n_tree_features {
            col_buf.clear();
            for i in 0..actual_samp {
                col_buf.push(samp_mat[i * n_tree_features + j]);
            }
            binners.push(FeatureBinner::from_column(&mut col_buf, MAX_BINS));
        }

        Self {
            binners,
            n_tree_features,
        }
    }

    fn quantize_rows(
        &self,
        x_orig: &[f64],
        row_indices: &[usize],
        n_orig_features: usize,
        pipeline: &FeaturePipeline,
    ) -> Vec<u8> {
        let n = row_indices.len();
        let p = self.n_tree_features;
        if n == 0 || p == 0 {
            return Vec::new();
        }
        let mut binned = vec![0u8; n * p];
        let mut row_buf = vec![0.0_f64; p];
        for (i, &r) in row_indices.iter().enumerate() {
            let orig_row = &x_orig[r * n_orig_features..(r + 1) * n_orig_features];
            pipeline.extract_tree_row(orig_row, &mut row_buf);
            let out_row = &mut binned[i * p..(i + 1) * p];
            for j in 0..p {
                out_row[j] = self.binners[j].bin_value(row_buf[j]);
            }
        }
        binned
    }

    #[inline]
    fn quantize_single_row(&self, tree_row: &[f64], out_bins: &mut [u8]) {
        for j in 0..self.n_tree_features {
            out_bins[j] = self.binners[j].bin_value(tree_row[j]);
        }
    }
}

/// Computes outlier-tempered Pseudo-Huber + MSE gradients and hessians.
fn compute_tempered_gradients(
    y_true: &[f64],
    y_pred: &[f64],
    grad: &mut [f64],
    hess: &mut [f64],
    abs_res_buf: &mut Vec<f64>,
) {
    let n = y_true.len();
    abs_res_buf.clear();
    let stride = if n > 1024 { n / 512 } else { 1 };
    for i in (0..n).step_by(stride) {
        abs_res_buf.push((y_pred[i] - y_true[i]).abs());
    }
    sort_f64(abs_res_buf);
    let q90 = quantile_sorted(abs_res_buf, 0.90);
    let q50 = quantile_sorted(abs_res_buf, 0.50);
    let delta = (3.5 * q50).max(1.8 * q90).max(1e-6);

    for i in 0..n {
        let r = y_pred[i] - y_true[i];
        let ar = r.abs();
        if ar <= delta {
            grad[i] = r;
            hess[i] = 1.0;
        } else {
            let ratio = ar / delta;
            let tempered_mag = delta * (1.0 + 0.40 * ratio.ln());
            grad[i] = scalar_sign(r) * tempered_mag;
            hess[i] = 1.0 / (1.0 + 0.35 * (ratio - 1.0));
        }
    }
}

// ============================================================================
// Section 6: Expert 2 — Oblivious (Symmetric) Histogram GBDT
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct ObliviousSplit {
    feature_idx: usize,
    threshold_bin: u8,
    nan_go_right: bool,
}

#[derive(Clone, Debug)]
struct ObliviousTree {
    splits: Vec<ObliviousSplit>, // length = depth (up to 6)
    leaf_values: Vec<f64>,       // length = 1 << depth
}

impl ObliviousTree {
    #[inline]
    fn predict_bins(&self, bins: &[u8]) -> f64 {
        let mut leaf_idx = 0usize;
        for (lvl, sp) in self.splits.iter().enumerate() {
            let b = bins[sp.feature_idx];
            let go_right = if b == NAN_BIN {
                sp.nan_go_right
            } else {
                b > sp.threshold_bin
            };
            if go_right {
                leaf_idx |= 1usize << lvl;
            }
        }
        self.leaf_values[leaf_idx]
    }
}

#[derive(Clone, Debug)]
struct ObliviousGbdtModel {
    trees: Vec<ObliviousTree>,
    base_value: f64,
    active_tree_count: usize,
}

impl ObliviousGbdtModel {
    fn fit_with_validation(
        binned_tr: &[u8],
        y_tr: &[f64],
        binned_val: Option<(&[u8], &[f64])>,
        n_features: usize,
        quantizer: &HistogramQuantizer,
        max_trees: usize,
        learning_rate: f64,
        l2_reg: f64,
        rng: &mut Rng,
    ) -> (Self, usize) {
        let n = y_tr.len();
        let (base_value, _) = mean_and_std(y_tr);
        if n < 2 || n_features == 0 || max_trees == 0 {
            return (
                Self {
                    trees: Vec::new(),
                    base_value,
                    active_tree_count: 0,
                },
                0,
            );
        }

        let mut pred_tr = vec![base_value; n];
        let mut pred_val = binned_val
            .map(|(_, yv)| vec![base_value; yv.len()])
            .unwrap_or_default();

        let mut grad = vec![0.0_f64; n];
        let mut hess = vec![1.0_f64; n];
        let mut abs_buf = Vec::with_capacity(1024);
        let mut leaf_assign = vec![0usize; n];

        let mut trees = Vec::with_capacity(max_trees);
        let mut best_val_rmse = if let Some((_, yv)) = binned_val {
            compute_rmse(yv, &pred_val)
        } else {
            f64::INFINITY
        };
        let mut best_tree_count = 0usize;

        let mut candidate_features: Vec<usize> = (0..n_features).collect();
        let colsample_cnt = ((n_features as f64) * 0.80).round() as usize;
        let colsample_cnt = colsample_cnt.max(1).min(n_features);

        let min_child_weight = if n < 50 {
            1.5
        } else if n < 500 {
            3.0
        } else {
            6.0
        };

        // Pre-allocate histogram buffers: max_leaves (64) * TOTAL_BINS (65)
        let mut hist_g = vec![0.0_f64; 64 * TOTAL_BINS];
        let mut hist_h = vec![0.0_f64; 64 * TOTAL_BINS];

        for t in 0..max_trees {
            compute_tempered_gradients(y_tr, &pred_tr, &mut grad, &mut hess, &mut abs_buf);

            // Cycle depths 4, 5, 6 (capped for small n)
            let depth = if n < 32 {
                3
            } else if n < 120 {
                4 + (t % 2)
            } else {
                4 + (t % 3)
            };

            rng.shuffle(&mut candidate_features);
            let active_feats = &candidate_features[..colsample_cnt];

            for v in leaf_assign.iter_mut() {
                *v = 0;
            }

            let mut splits = Vec::with_capacity(depth);

            for lvl in 0..depth {
                let n_leaves_lvl = 1usize << lvl;
                let mut best_gain = -1.0_f64;
                let mut best_split = ObliviousSplit {
                    feature_idx: active_feats[0],
                    threshold_bin: 0,
                    nan_go_right: false,
                };

                for &f_idx in active_feats {
                    let n_edges = quantizer.binners[f_idx].edges.len();
                    if n_edges == 0 {
                        continue;
                    }

                    // Clear histogram for current level leaves
                    let slice_len = n_leaves_lvl * TOTAL_BINS;
                    for z in hist_g[..slice_len].iter_mut() {
                        *z = 0.0;
                    }
                    for z in hist_h[..slice_len].iter_mut() {
                        *z = 0.0;
                    }

                    for i in 0..n {
                        let lf = leaf_assign[i];
                        let b = binned_tr[i * n_features + f_idx] as usize;
                        let idx = lf * TOTAL_BINS + b;
                        hist_g[idx] += grad[i];
                        hist_h[idx] += hess[i];
                    }

                    // Convert non-NaN bins into prefix sums per leaf
                    let mut tot_g = [0.0_f64; 32];
                    let mut tot_h = [0.0_f64; 32];
                    let mut nan_g = [0.0_f64; 32];
                    let mut nan_h = [0.0_f64; 32];

                    for lf in 0..n_leaves_lvl {
                        let base = lf * TOTAL_BINS;
                        nan_g[lf] = hist_g[base + (NAN_BIN as usize)];
                        nan_h[lf] = hist_h[base + (NAN_BIN as usize)];
                        let mut cg = 0.0_f64;
                        let mut ch = 0.0_f64;
                        for b in 0..=n_edges {
                            cg += hist_g[base + b];
                            ch += hist_h[base + b];
                            hist_g[base + b] = cg;
                            hist_h[base + b] = ch;
                        }
                        tot_g[lf] = cg + nan_g[lf];
                        tot_h[lf] = ch + nan_h[lf];
                    }

                    for b in 0..n_edges {
                        for &nan_right in &[false, true] {
                            let mut gain = 0.0_f64;
                            let mut valid_any = false;
                            for lf in 0..n_leaves_lvl {
                                let base = lf * TOTAL_BINS;
                                let mut gl = hist_g[base + b];
                                let mut hl = hist_h[base + b];
                                if !nan_right {
                                    gl += nan_g[lf];
                                    hl += nan_h[lf];
                                }
                                let gr = tot_g[lf] - gl;
                                let hr = tot_h[lf] - hl;
                                if hl >= min_child_weight && hr >= min_child_weight {
                                    valid_any = true;
                                    let parent_score = (tot_g[lf] * tot_g[lf]) / (tot_h[lf] + l2_reg);
                                    let left_score = (gl * gl) / (hl + l2_reg);
                                    let right_score = (gr * gr) / (hr + l2_reg);
                                    gain += left_score + right_score - parent_score;
                                }
                            }
                            if valid_any && gain > best_gain {
                                best_gain = gain;
                                best_split = ObliviousSplit {
                                    feature_idx: f_idx,
                                    threshold_bin: b as u8,
                                    nan_go_right: nan_right,
                                };
                            }
                        }
                    }
                }

                // Update leaf assignments for level lvl
                let f_idx = best_split.feature_idx;
                let thresh = best_split.threshold_bin;
                let nan_r = best_split.nan_go_right;
                for i in 0..n {
                    let b = binned_tr[i * n_features + f_idx];
                    let go_r = if b == NAN_BIN { nan_r } else { b > thresh };
                    if go_r {
                        leaf_assign[i] |= 1usize << lvl;
                    }
                }
                splits.push(best_split);
            }

            // Compute regularized leaf outputs blending Newton step with true MSE residual mean
            let total_leaves = 1usize << depth;
            let mut sum_g = vec![0.0_f64; total_leaves];
            let mut sum_h = vec![0.0_f64; total_leaves];
            let mut sum_raw_res = vec![0.0_f64; total_leaves];
            let mut cnt_leaf = vec![0usize; total_leaves];

            for i in 0..n {
                let lf = leaf_assign[i];
                sum_g[lf] += grad[i];
                sum_h[lf] += hess[i];
                sum_raw_res[lf] += pred_tr[i] - y_tr[i];
                cnt_leaf[lf] += 1;
            }

            let mut leaf_values = vec![0.0_f64; total_leaves];
            for lf in 0..total_leaves {
                if cnt_leaf[lf] > 0 {
                    let newton_step = -sum_g[lf] / (sum_h[lf] + l2_reg);
                    let raw_step = -sum_raw_res[lf] / ((cnt_leaf[lf] as f64) + l2_reg);
                    leaf_values[lf] = learning_rate * (0.65 * newton_step + 0.35 * raw_step);
                }
            }

            for i in 0..n {
                pred_tr[i] += leaf_values[leaf_assign[i]];
            }

            let tree = ObliviousTree {
                splits,
                leaf_values,
            };

            if let Some((bv, yv)) = binned_val {
                let nv = yv.len();
                for i in 0..nv {
                    let row_bins = &bv[i * n_features..(i + 1) * n_features];
                    pred_val[i] += tree.predict_bins(row_bins);
                }
                let rmse = compute_rmse(yv, &pred_val);
                if rmse < best_val_rmse {
                    best_val_rmse = rmse;
                    best_tree_count = t + 1;
                } else if (t + 1) > best_tree_count + 35 && (t + 1) >= 25 {
                    trees.push(tree);
                    break;
                }
            } else {
                best_tree_count = t + 1;
            }

            trees.push(tree);
        }

        let active_tree_count = best_tree_count.min(trees.len());
        (
            Self {
                trees,
                base_value,
                active_tree_count,
            },
            active_tree_count,
        )
    }

    #[inline]
    fn predict_bins(&self, bins: &[u8]) -> f64 {
        let mut sum = self.base_value;
        for tree in self.trees.iter().take(self.active_tree_count) {
            sum += tree.predict_bins(bins);
        }
        sum
    }
}

// ============================================================================
// Section 7: Expert 3 — Leaf-Wise Histogram GBDT with Damped Ridge Prior
// ============================================================================

#[derive(Clone, Debug)]
struct WiseNode {
    is_leaf: bool,
    feature_idx: usize,
    threshold_bin: u8,
    nan_go_right: bool,
    left_child: usize,
    right_child: usize,
    leaf_value: f64,
}

#[derive(Clone, Debug)]
struct WiseTree {
    nodes: Vec<WiseNode>,
}

impl WiseTree {
    #[inline]
    fn predict_bins(&self, bins: &[u8]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mut curr = 0usize;
        loop {
            let node = &self.nodes[curr];
            if node.is_leaf {
                return node.leaf_value;
            }
            let b = bins[node.feature_idx];
            let go_right = if b == NAN_BIN {
                node.nan_go_right
            } else {
                b > node.threshold_bin
            };
            curr = if go_right {
                node.right_child
            } else {
                node.left_child
            };
        }
    }
}

#[derive(Clone, Debug)]
struct WiseGbdtModel {
    trees: Vec<WiseTree>,
    base_value: f64,
    ridge_damp: f64,
    active_tree_count: usize,
}

impl WiseGbdtModel {
    fn fit_with_validation(
        binned_tr: &[u8],
        y_tr: &[f64],
        ridge_pred_tr: &[f64],
        binned_val: Option<(&[u8], &[f64], &[f64])>, // (bins_val, y_val, ridge_pred_val)
        n_features: usize,
        quantizer: &HistogramQuantizer,
        max_trees: usize,
        learning_rate: f64,
        l2_reg: f64,
        rng: &mut Rng,
    ) -> (Self, usize) {
        let n = y_tr.len();
        let (base_value, y_std) = mean_and_std(y_tr);
        if n < 2 || n_features == 0 || max_trees == 0 {
            return (
                Self {
                    trees: Vec::new(),
                    base_value,
                    ridge_damp: 0.0,
                    active_tree_count: 0,
                },
                0,
            );
        }

        // Determine optimal damping factor for the Ridge prior based on correlation/error reduction
        let ridge_rmse = compute_rmse(y_tr, ridge_pred_tr);
        let ridge_damp = if ridge_rmse < 0.85 * y_std {
            0.55
        } else if ridge_rmse < 0.98 * y_std {
            0.30
        } else {
            0.0
        };

        let mut pred_tr = vec![0.0_f64; n];
        for i in 0..n {
            pred_tr[i] = base_value + ridge_damp * (ridge_pred_tr[i] - base_value);
        }

        let mut pred_val = if let Some((_, yv, rv)) = binned_val {
            let mut pv = vec![0.0_f64; yv.len()];
            for i in 0..yv.len() {
                pv[i] = base_value + ridge_damp * (rv[i] - base_value);
            }
            pv
        } else {
            Vec::new()
        };

        let mut grad = vec![0.0_f64; n];
        let mut hess = vec![1.0_f64; n];
        let mut abs_buf = Vec::with_capacity(1024);
        let mut node_assign = vec![0usize; n];

        let max_leaves = if n < 60 {
            6
        } else if n < 400 {
            12
        } else {
            20
        };
        let min_child_weight = if n < 60 { 2.0 } else { 5.0 };

        let mut candidate_features: Vec<usize> = (0..n_features).collect();
        let colsample_cnt = ((n_features as f64) * 0.75).round() as usize;
        let colsample_cnt = colsample_cnt.max(1).min(n_features);

        let mut trees = Vec::with_capacity(max_trees);
        let mut best_val_rmse = if let Some((_, yv, _)) = binned_val {
            compute_rmse(yv, &pred_val)
        } else {
            f64::INFINITY
        };
        let mut best_tree_count = 0usize;

        let mut hist_g = vec![0.0_f64; TOTAL_BINS];
        let mut hist_h = vec![0.0_f64; TOTAL_BINS];

        for t in 0..max_trees {
            compute_tempered_gradients(y_tr, &pred_tr, &mut grad, &mut hess, &mut abs_buf);
            rng.shuffle(&mut candidate_features);
            let active_feats = &candidate_features[..colsample_cnt];

            for v in node_assign.iter_mut() {
                *v = 0;
            }

            let mut nodes = Vec::with_capacity(2 * max_leaves + 1);
            let mut tot_g0 = 0.0_f64;
            let mut tot_h0 = 0.0_f64;
            let mut tot_raw0 = 0.0_f64;
            for i in 0..n {
                tot_g0 += grad[i];
                tot_h0 += hess[i];
                tot_raw0 += pred_tr[i] - y_tr[i];
            }
            let root_val = learning_rate
                * (0.65 * (-tot_g0 / (tot_h0 + l2_reg))
                    + 0.35 * (-tot_raw0 / ((n as f64) + l2_reg)));

            nodes.push(WiseNode {
                is_leaf: true,
                feature_idx: 0,
                threshold_bin: 0,
                nan_go_right: false,
                left_child: 0,
                right_child: 0,
                leaf_value: root_val,
            });

            // Maintain list of splittable leaf indices and their depths
            let mut leaf_depths = vec![0usize; 1];
            let mut num_leaves = 1usize;

            while num_leaves < max_leaves {
                // Find the current leaf with maximum samples/hessian that can still split
                let mut best_node_idx = None;
                let mut best_split_gain = 1e-7_f64;
                let mut best_feat = 0usize;
                let mut best_bin = 0u8;
                let mut best_nan_r = false;

                for node_id in 0..nodes.len() {
                    if !nodes[node_id].is_leaf || leaf_depths[node_id] >= 5 {
                        continue;
                    }
                    let mut node_g = 0.0_f64;
                    let mut node_h = 0.0_f64;
                    for i in 0..n {
                        if node_assign[i] == node_id {
                            node_g += grad[i];
                            node_h += hess[i];
                        }
                    }
                    if node_h < 2.0 * min_child_weight {
                        continue;
                    }
                    let parent_score = (node_g * node_g) / (node_h + l2_reg);

                    for &f_idx in active_feats {
                        let n_edges = quantizer.binners[f_idx].edges.len();
                        if n_edges == 0 {
                            continue;
                        }
                        for b in 0..TOTAL_BINS {
                            hist_g[b] = 0.0;
                            hist_h[b] = 0.0;
                        }
                        for i in 0..n {
                            if node_assign[i] == node_id {
                                let b = binned_tr[i * n_features + f_idx] as usize;
                                hist_g[b] += grad[i];
                                hist_h[b] += hess[i];
                            }
                        }

                        let nan_g = hist_g[NAN_BIN as usize];
                        let nan_h = hist_h[NAN_BIN as usize];
                        let mut cg = 0.0_f64;
                        let mut ch = 0.0_f64;
                        for b in 0..n_edges {
                            cg += hist_g[b];
                            ch += hist_h[b];
                            for &nan_r in &[false, true] {
                                let gl = if nan_r { cg } else { cg + nan_g };
                                let hl = if nan_r { ch } else { ch + nan_h };
                                let gr = node_g - gl;
                                let hr = node_h - hl;
                                if hl >= min_child_weight && hr >= min_child_weight {
                                    let gain = (gl * gl) / (hl + l2_reg)
                                        + (gr * gr) / (hr + l2_reg)
                                        - parent_score;
                                    if gain > best_split_gain {
                                        best_split_gain = gain;
                                        best_node_idx = Some(node_id);
                                        best_feat = f_idx;
                                        best_bin = b as u8;
                                        best_nan_r = nan_r;
                                    }
                                }
                            }
                        }
                    }
                }

                let split_node = match best_node_idx {
                    Some(id) => id,
                    None => break,
                };

                let left_id = nodes.len();
                let right_id = left_id + 1;
                let next_depth = leaf_depths[split_node] + 1;

                let mut gl = 0.0_f64;
                let mut hl = 0.0_f64;
                let mut raw_l = 0.0_f64;
                let mut cnt_l = 0usize;

                let mut gr = 0.0_f64;
                let mut hr = 0.0_f64;
                let mut raw_r = 0.0_f64;
                let mut cnt_r = 0usize;

                for i in 0..n {
                    if node_assign[i] == split_node {
                        let b = binned_tr[i * n_features + best_feat];
                        let go_r = if b == NAN_BIN {
                            best_nan_r
                        } else {
                            b > best_bin
                        };
                        if go_r {
                            node_assign[i] = right_id;
                            gr += grad[i];
                            hr += hess[i];
                            raw_r += pred_tr[i] - y_tr[i];
                            cnt_r += 1;
                        } else {
                            node_assign[i] = left_id;
                            gl += grad[i];
                            hl += hess[i];
                            raw_l += pred_tr[i] - y_tr[i];
                            cnt_l += 1;
                        }
                    }
                }

                let val_l = learning_rate
                    * (0.65 * (-gl / (hl + l2_reg))
                        + 0.35 * (-raw_l / ((cnt_l.max(1) as f64) + l2_reg)));
                let val_r = learning_rate
                    * (0.65 * (-gr / (hr + l2_reg))
                        + 0.35 * (-raw_r / ((cnt_r.max(1) as f64) + l2_reg)));

                nodes[split_node].is_leaf = false;
                nodes[split_node].feature_idx = best_feat;
                nodes[split_node].threshold_bin = best_bin;
                nodes[split_node].nan_go_right = best_nan_r;
                nodes[split_node].left_child = left_id;
                nodes[split_node].right_child = right_id;

                nodes.push(WiseNode {
                    is_leaf: true,
                    feature_idx: 0,
                    threshold_bin: 0,
                    nan_go_right: false,
                    left_child: 0,
                    right_child: 0,
                    leaf_value: val_l,
                });
                nodes.push(WiseNode {
                    is_leaf: true,
                    feature_idx: 0,
                    threshold_bin: 0,
                    nan_go_right: false,
                    left_child: 0,
                    right_child: 0,
                    leaf_value: val_r,
                });
                leaf_depths.push(next_depth);
                leaf_depths.push(next_depth);
                num_leaves += 1;
            }

            for i in 0..n {
                pred_tr[i] += nodes[node_assign[i]].leaf_value;
            }

            let tree = WiseTree { nodes };

            if let Some((bv, yv, _)) = binned_val {
                let nv = yv.len();
                for i in 0..nv {
                    let row_bins = &bv[i * n_features..(i + 1) * n_features];
                    pred_val[i] += tree.predict_bins(row_bins);
                }
                let rmse = compute_rmse(yv, &pred_val);
                if rmse < best_val_rmse {
                    best_val_rmse = rmse;
                    best_tree_count = t + 1;
                } else if (t + 1) > best_tree_count + 30 && (t + 1) >= 20 {
                    trees.push(tree);
                    break;
                }
            } else {
                best_tree_count = t + 1;
            }

            trees.push(tree);
        }

        let active_tree_count = best_tree_count.min(trees.len());
        (
            Self {
                trees,
                base_value,
                ridge_damp,
                active_tree_count,
            },
            active_tree_count,
        )
    }

    #[inline]
    fn predict_bins(&self, bins: &[u8], ridge_pred: f64) -> f64 {
        let mut sum = self.base_value + self.ridge_damp * (ridge_pred - self.base_value);
        for tree in self.trees.iter().take(self.active_tree_count) {
            sum += tree.predict_bins(bins);
        }
        sum
    }
}

// ============================================================================
// Section 8: Expert 4 — Heavy-Tail Asinh Target GBDT + Duan's Smearing Estimator
// ============================================================================

#[derive(Clone, Debug)]
struct AsinhSmearingGbdtModel {
    inner_gbdt: ObliviousGbdtModel,
    y_center: f64,
    y_scale: f64,
    smear_cosh_mean: f64,
    smear_sinh_mean: f64,
    y_min: f64,
    y_max: f64,
}

impl AsinhSmearingGbdtModel {
    fn fit_with_validation(
        binned_tr: &[u8],
        y_tr: &[f64],
        binned_val: Option<(&[u8], &[f64])>,
        n_features: usize,
        quantizer: &HistogramQuantizer,
        max_trees: usize,
        learning_rate: f64,
        l2_reg: f64,
        rng: &mut Rng,
    ) -> (Self, usize) {
        let n = y_tr.len();
        let mut sorted_y = y_tr.to_vec();
        sort_f64(&mut sorted_y);
        let y_min = sorted_y.first().copied().unwrap_or(0.0);
        let y_max = sorted_y.last().copied().unwrap_or(0.0);
        let y_center = quantile_sorted(&sorted_y, 0.50);
        let q25 = quantile_sorted(&sorted_y, 0.25);
        let q75 = quantile_sorted(&sorted_y, 0.75);
        let (_, y_std) = mean_and_std(y_tr);
        let y_scale = ((q75 - q25) / 1.349).max(0.20 * y_std).max(1e-9);

        let mut z_tr = Vec::with_capacity(n);
        for &yr in y_tr {
            z_tr.push(((yr - y_center) / y_scale).asinh());
        }

        let z_val_vec = binned_val.map(|(_, yv)| {
            yv.iter()
                .map(|&yr| ((yr - y_center) / y_scale).asinh())
                .collect::<Vec<f64>>()
        });
        let val_ref = match (binned_val, z_val_vec.as_ref()) {
            (Some((bv, _)), Some(zv)) => Some((bv, zv.as_slice())),
            _ => None,
        };

        let (inner_gbdt, best_trees) = ObliviousGbdtModel::fit_with_validation(
            binned_tr,
            &z_tr,
            val_ref,
            n_features,
            quantizer,
            max_trees,
            learning_rate,
            l2_reg,
            rng,
        );

        // Compute Duan's nonparametric smearing residual moments:
        // E[sinh(z_hat + eps)] = sinh(z_hat) * E[cosh(eps)] + cosh(z_hat) * E[sinh(eps)]
        let mut residuals = Vec::with_capacity(n);
        for i in 0..n {
            let row_bins = &binned_tr[i * n_features..(i + 1) * n_features];
            let z_hat = inner_gbdt.predict_bins(row_bins);
            residuals.push(scalar_clip(z_tr[i] - z_hat, -4.0, 4.0));
        }
        sort_f64(&mut residuals);
        // Trim top/bottom 2.5% residuals for smearing stability
        let lo_idx = (n * 25) / 1000;
        let hi_idx = (n - lo_idx).max(lo_idx + 1);
        let trimmed = &residuals[lo_idx..hi_idx];
        let mut sum_cosh = 0.0_f64;
        let mut sum_sinh = 0.0_f64;
        for &eps in trimmed {
            sum_cosh += scalar_cosh(eps);
            sum_sinh += scalar_sinh(eps);
        }
        let denom = (trimmed.len().max(1)) as f64;
        let smear_cosh_mean = (sum_cosh / denom).clamp(1.0, 4.0);
        let smear_sinh_mean = (sum_sinh / denom).clamp(-1.5, 1.5);

        (
            Self {
                inner_gbdt,
                y_center,
                y_scale,
                smear_cosh_mean,
                smear_sinh_mean,
                y_min,
                y_max,
            },
            best_trees,
        )
    }

    #[inline]
    fn predict_bins(&self, bins: &[u8]) -> f64 {
        let z_hat = scalar_clip(self.inner_gbdt.predict_bins(bins), -12.0, 12.0);
        let sh = scalar_sinh(z_hat);
        let ch = scalar_cosh(z_hat);
        let raw = self.y_center
            + self.y_scale * (self.smear_cosh_mean * sh + self.smear_sinh_mean * ch);
        let span = (self.y_max - self.y_min).abs().max(1e-6);
        scalar_clip(raw, self.y_min - 1.5 * span, self.y_max + 1.5 * span)
    }
}

// ============================================================================
// Section 9: Expert 5 — Relevance-Weighted Metric Kernel / k-NN Regressor
// ============================================================================

#[derive(Clone, Debug)]
struct MetricKnnModel {
    n_proj: usize,
    feature_weights: Vec<f64>,
    ref_x: Vec<f64>, // flattened [n_ref * n_proj]
    ref_y: Vec<f64>,
    k_neighbors: usize,
    inv_bandwidth: f64,
    fallback_mean: f64,
}

impl MetricKnnModel {
    fn fit(
        x_orig: &[f64],
        y: &[f64],
        row_indices: &[usize],
        n_orig_features: usize,
        pipeline: &FeaturePipeline,
    ) -> Self {
        let n = row_indices.len();
        let (fallback_mean, _) = {
            let mut ys = Vec::with_capacity(n);
            for &r in row_indices {
                ys.push(y[r]);
            }
            mean_and_std(&ys)
        };

        let n_proj = pipeline.active_features.len().min(28);
        if n_proj == 0 || n < 5 {
            return Self {
                n_proj: 0,
                feature_weights: Vec::new(),
                ref_x: Vec::new(),
                ref_y: Vec::new(),
                k_neighbors: 1,
                inv_bandwidth: 1.0,
                fallback_mean,
            };
        }

        let max_ref = 1200usize;
        let stride = if n > max_ref { n / max_ref } else { 1 };
        let mut ref_rows = Vec::with_capacity((n / stride) + 1);
        for (pos, &r) in row_indices.iter().enumerate() {
            if stride == 1 || pos % stride == 0 {
                ref_rows.push(r);
            }
        }
        let nr = ref_rows.len();

        let mut feature_weights = Vec::with_capacity(n_proj);
        let mut w_sum = 0.0_f64;
        for j in 0..n_proj {
            let w = pipeline.active_features[j].relevance.max(0.02);
            feature_weights.push(w);
            w_sum += w;
        }
        for w in feature_weights.iter_mut() {
            *w = (*w / w_sum.max(1e-9)) * (n_proj as f64);
        }

        let mut ref_x = vec![0.0_f64; nr * n_proj];
        let mut ref_y = Vec::with_capacity(nr);
        for (i, &r) in ref_rows.iter().enumerate() {
            let orig_row = &x_orig[r * n_orig_features..(r + 1) * n_orig_features];
            for j in 0..n_proj {
                let fs = &pipeline.active_features[j];
                let v = orig_row[fs.orig_idx];
                let z = if v.is_nan() {
                    0.0
                } else {
                    scalar_clip((v - fs.mean) / fs.std, -8.0, 8.0)
                };
                ref_x[i * n_proj + j] = z;
            }
            ref_y.push(y[r]);
        }

        let k_neighbors = if nr < 25 {
            5.min(nr - 1)
        } else if nr < 200 {
            11
        } else {
            19
        };
        let inv_bandwidth = 1.5 / (n_proj.max(1) as f64).sqrt();

        Self {
            n_proj,
            feature_weights,
            ref_x,
            ref_y,
            k_neighbors: k_neighbors.max(1),
            inv_bandwidth,
            fallback_mean,
        }
    }

    fn predict_row(&self, orig_row: &[f64], pipeline: &FeaturePipeline) -> f64 {
        let nr = self.ref_y.len();
        if self.n_proj == 0 || nr == 0 {
            return self.fallback_mean;
        }

        let mut query = vec![0.0_f64; self.n_proj];
        for j in 0..self.n_proj {
            let fs = &pipeline.active_features[j];
            let v = orig_row[fs.orig_idx];
            query[j] = if v.is_nan() {
                0.0
            } else {
                scalar_clip((v - fs.mean) / fs.std, -8.0, 8.0)
            };
        }

        let k = self.k_neighbors.min(nr);
        let mut best_dists = vec![f64::INFINITY; k];
        let mut best_ys = vec![self.fallback_mean; k];

        for i in 0..nr {
            let r_slice = &self.ref_x[i * self.n_proj..(i + 1) * self.n_proj];
            let mut dist = 0.0_f64;
            for j in 0..self.n_proj {
                let diff = query[j] - r_slice[j];
                dist += self.feature_weights[j] * diff * diff;
            }
            if dist < best_dists[k - 1] {
                let mut pos = k - 1;
                while pos > 0 && dist < best_dists[pos - 1] {
                    best_dists[pos] = best_dists[pos - 1];
                    best_ys[pos] = best_ys[pos - 1];
                    pos -= 1;
                }
                best_dists[pos] = dist;
                best_ys[pos] = self.ref_y[i];
            }
        }

        let scale_d = best_dists[0].max(1e-6);
        let mut w_sum = 0.0_f64;
        let mut y_sum = 0.0_f64;
        for m in 0..k {
            let d_norm = best_dists[m] / scale_d;
            let w_gauss = (-self.inv_bandwidth * best_dists[m]).exp();
            let w_inv = 1.0 / (1.0 + d_norm);
            let w = 0.6 * w_gauss + 0.4 * w_inv;
            w_sum += w;
            y_sum += w * best_ys[m];
        }
        if w_sum > 1e-12 {
            y_sum / w_sum
        } else {
            self.fallback_mean
        }
    }
}

// ============================================================================
// Section 10: Single-Split Bundled Expert Suite & Out-Of-Fold Simplex Stacker
// ============================================================================

const NUM_EXPERTS: usize = 5;
// Expert indices:
// 0: SplineRidgeModel
// 1: ObliviousGbdtModel
// 2: WiseGbdtModel (boosted on damped Ridge prior)
// 3: AsinhSmearingGbdtModel
// 4: MetricKnnModel (or constant mean when disabled on large n)

#[derive(Clone, Debug)]
struct ExpertUnit {
    pipeline: FeaturePipeline,
    quantizer: HistogramQuantizer,
    ridge: SplineRidgeModel,
    oblivious_gbdt: ObliviousGbdtModel,
    wise_gbdt: WiseGbdtModel,
    asinh_gbdt: AsinhSmearingGbdtModel,
    knn: Option<MetricKnnModel>,
    target_mean: f64,
}

impl ExpertUnit {
    fn predict_experts(&self, orig_row: &[f64]) -> [f64; NUM_EXPERTS] {
        let mut ridge_buf = vec![0.0_f64; self.pipeline.ridge_basis_dim()];
        let p_ridge = self
            .ridge
            .predict_row(orig_row, &self.pipeline, &mut ridge_buf);

        let n_tree_f = self.quantizer.n_tree_features;
        if n_tree_f == 0 {
            return [
                p_ridge,
                self.target_mean,
                self.target_mean,
                self.target_mean,
                self.target_mean,
            ];
        }

        let mut tree_row = vec![0.0_f64; n_tree_f];
        let mut bins = vec![0u8; n_tree_f];
        self.pipeline.extract_tree_row(orig_row, &mut tree_row);
        self.quantizer.quantize_single_row(&tree_row, &mut bins);

        let p_obliv = self.oblivious_gbdt.predict_bins(&bins);
        let p_wise = self.wise_gbdt.predict_bins(&bins, p_ridge);
        let p_asinh = self.asinh_gbdt.predict_bins(&bins);
        let p_knn = match &self.knn {
            Some(knn_model) => knn_model.predict_row(orig_row, &self.pipeline),
            None => self.target_mean,
        };

        [p_ridge, p_obliv, p_wise, p_asinh, p_knn]
    }
}

/// Solves for non-negative simplex weights w_0..w_{M-1} (sum = 1) minimizing OOF MSE + diversity L2 penalty
/// using Frank-Wolfe / pairwise simplex coordinate descent.
fn fit_simplex_stacking_weights(oof_preds: &[[f64; NUM_EXPERTS]], y_true: &[f64]) -> [f64; NUM_EXPERTS] {
    let n = y_true.len();
    if n == 0 {
        return [0.20; NUM_EXPERTS];
    }

    // Start from the single best expert on OOF RMSE
    let mut best_m = 0usize;
    let mut best_mse = f64::INFINITY;
    let mut expert_mses = [0.0_f64; NUM_EXPERTS];
    for m in 0..NUM_EXPERTS {
        let mut s = 0.0_f64;
        for i in 0..n {
            let d = oof_preds[i][m] - y_true[i];
            s += d * d;
        }
        let mse = s / (n as f64);
        expert_mses[m] = mse;
        if mse < best_mse {
            best_mse = mse;
            best_m = m;
        }
    }

    let mut w = [0.0_f64; NUM_EXPERTS];
    w[best_m] = 1.0;

    let mut current_ens = vec![0.0_f64; n];
    for i in 0..n {
        current_ens[i] = oof_preds[i][best_m];
    }

    // Regularization toward uniform diversity scaled by target variance
    let (_, y_std) = mean_and_std(y_true);
    let div_reg = 0.002 * y_std * y_std;

    // 250 iterations of exact line-search Frank-Wolfe on the probability simplex
    for _iter in 0..250 {
        // Compute gradient of (1/n) sum (ens_i - y_i)^2 + div_reg * sum w_m^2
        let mut grad = [0.0_f64; NUM_EXPERTS];
        for m in 0..NUM_EXPERTS {
            let mut g = 0.0_f64;
            for i in 0..n {
                g += (current_ens[i] - y_true[i]) * oof_preds[i][m];
            }
            grad[m] = (2.0 / (n as f64)) * g + 2.0 * div_reg * w[m];
        }

        // Find vertex s_m with smallest gradient
        let mut min_m = 0usize;
        for m in 1..NUM_EXPERTS {
            if grad[m] < grad[min_m] {
                min_m = m;
            }
        }

        // Exact line search for step gamma in [0, 1]: w_new = (1 - gamma) w + gamma e_{min_m}
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in 0..n {
            let diff_dir = oof_preds[i][min_m] - current_ens[i];
            let res = current_ens[i] - y_true[i];
            num -= res * diff_dir;
            den += diff_dir * diff_dir;
        }
        num *= 1.0 / (n as f64);
        den *= 1.0 / (n as f64);

        for m in 0..NUM_EXPERTS {
            let dw = (if m == min_m { 1.0 } else { 0.0 }) - w[m];
            num -= div_reg * w[m] * dw;
            den += div_reg * dw * dw;
        }

        if den <= 1e-15 {
            break;
        }
        let gamma = scalar_clip(num / den, 0.0, 1.0);
        if gamma <= 1e-7 {
            break;
        }

        for m in 0..NUM_EXPERTS {
            w[m] *= 1.0 - gamma;
        }
        w[min_m] += gamma;

        for i in 0..n {
            current_ens[i] =
                (1.0 - gamma) * current_ens[i] + gamma * oof_preds[i][min_m];
        }
    }

    // Clean tiny weights and renormalize
    let mut sum_w = 0.0_f64;
    for m in 0..NUM_EXPERTS {
        if w[m] < 0.01 {
            w[m] = 0.0;
        }
        sum_w += w[m];
    }
    if sum_w <= 1e-9 {
        let mut out = [0.0_f64; NUM_EXPERTS];
        out[best_m] = 1.0;
        out
    } else {
        for m in 0..NUM_EXPERTS {
            w[m] /= sum_w;
        }
        w
    }
}

#[derive(Clone, Debug)]
struct FittedEnsemble {
    n_features: usize,
    is_constant: bool,
    target_mean: f64,
    target_min: f64,
    target_max: f64,
    expert_weights: [f64; NUM_EXPERTS],
    calibration_scale: f64,
    fold_units: Vec<ExpertUnit>,
    full_unit: Option<ExpertUnit>,
}

impl FittedEnsemble {
    fn train(
        x_raw: &[f64],
        y_raw: &[f64],
        n_rows: usize,
        n_features: usize,
        rng: &mut Rng,
    ) -> Result<Self, String> {
        let (target_mean, target_std) = mean_and_std(y_raw);
        let mut target_min = f64::INFINITY;
        let mut target_max = f64::NEG_INFINITY;
        for &yr in y_raw {
            if yr < target_min {
                target_min = yr;
            }
            if yr > target_max {
                target_max = yr;
            }
        }

        // Degenerate or zero-feature or constant-target fast path
        if n_rows == 1
            || n_features == 0
            || (target_max - target_min).abs() <= 1e-13 * target_mean.abs().max(1.0)
        {
            return Ok(Self {
                n_features,
                is_constant: true,
                target_mean,
                target_min,
                target_max,
                expert_weights: [1.0, 0.0, 0.0, 0.0, 0.0],
                calibration_scale: 1.0,
                fold_units: Vec::new(),
                full_unit: None,
            });
        }

        // Adaptive computational governor based on (n_rows, n_features)
        let max_active_features = if n_rows > 25000 {
            96
        } else if n_rows > 8000 {
            144
        } else {
            224
        };

        let n_folds = if n_rows < 15 {
            3.min(n_rows)
        } else if n_rows <= 3500 {
            5
        } else if n_rows <= 15000 {
            4
        } else {
            3
        };

        let max_trees_obliv = if n_rows > 25000 {
            95
        } else if n_rows > 6000 {
            140
        } else if n_rows >= 40 {
            180
        } else {
            65
        };

        let max_trees_wise = if n_rows > 25000 {
            75
        } else if n_rows > 6000 {
            110
        } else if n_rows >= 40 {
            140
        } else {
            50
        };

        let use_knn = n_rows >= 8 && n_rows <= 3000;

        // If n_rows is huge (> 30,000), select a stratified 20,000-row subset for CV & training
        let working_indices: Vec<usize> = if n_rows > 30000 {
            let mut order: Vec<usize> = (0..n_rows).collect();
            order.sort_by(|&a, &b| y_raw[a].total_cmp(&y_raw[b]).then_with(|| a.cmp(&b)));
            let step = n_rows as f64 / 20000.0;
            let mut sub = Vec::with_capacity(20000);
            for k in 0..20000 {
                let idx = ((k as f64 * step).floor() as usize).min(n_rows - 1);
                sub.push(order[idx]);
            }
            sub
        } else {
            (0..n_rows).collect()
        };

        let nw = working_indices.len();

        // Check if any feature is non-constant on working_indices
        let probe_pipe =
            FeaturePipeline::fit(x_raw, y_raw, &working_indices, n_features, max_active_features);
        if probe_pipe.active_features.is_empty() {
            return Ok(Self {
                n_features,
                is_constant: true,
                target_mean,
                target_min,
                target_max,
                expert_weights: [1.0, 0.0, 0.0, 0.0, 0.0],
                calibration_scale: 1.0,
                fold_units: Vec::new(),
                full_unit: None,
            });
        }

        // Stratified target quantile fold assignment
        let mut sorted_pos: Vec<usize> = (0..nw).collect();
        sorted_pos.sort_by(|&a, &b| {
            let ra = working_indices[a];
            let rb = working_indices[b];
            y_raw[ra].total_cmp(&y_raw[rb]).then_with(|| ra.cmp(&rb))
        });

        let mut fold_id = vec![0usize; nw];
        let mut block = Vec::with_capacity(n_folds);
        for chunk_start in (0..nw).step_by(n_folds) {
            let chunk_end = (chunk_start + n_folds).min(nw);
            block.clear();
            for f in 0..(chunk_end - chunk_start) {
                block.push(f);
            }
            rng.shuffle(&mut block);
            for (offset, &f) in block.iter().enumerate() {
                fold_id[sorted_pos[chunk_start + offset]] = f;
            }
        }

        let mut oof_preds = vec![[target_mean; NUM_EXPERTS]; nw];
        let mut y_working = vec![0.0_f64; nw];
        for i in 0..nw {
            y_working[i] = y_raw[working_indices[i]];
        }

        let mut fold_units = Vec::with_capacity(n_folds);
        let mut sum_opt_obliv = 0usize;
        let mut sum_opt_wise = 0usize;
        let mut sum_opt_asinh = 0usize;

        let lr_obliv = if nw < 100 { 0.05 } else { 0.065 };
        let lr_wise = if nw < 100 { 0.045 } else { 0.055 };
        let l2_reg = if nw < 100 { 4.0 } else { 6.0 };

        for f in 0..n_folds {
            let mut tr_rows = Vec::with_capacity(nw);
            let mut val_rows = Vec::with_capacity(nw / n_folds + 2);
            let mut val_local_pos = Vec::with_capacity(nw / n_folds + 2);

            for pos in 0..nw {
                let orig_r = working_indices[pos];
                if fold_id[pos] == f {
                    val_rows.push(orig_r);
                    val_local_pos.push(pos);
                } else {
                    tr_rows.push(orig_r);
                }
            }
            if tr_rows.is_empty() {
                tr_rows = val_rows.clone();
            }

            let pipe =
                FeaturePipeline::fit(x_raw, y_raw, &tr_rows, n_features, max_active_features);
            let quant = HistogramQuantizer::fit(x_raw, &tr_rows, n_features, &pipe);

            let mut y_tr = Vec::with_capacity(tr_rows.len());
            for &r in &tr_rows {
                y_tr.push(y_raw[r]);
            }
            let mut y_val = Vec::with_capacity(val_rows.len());
            for &r in &val_rows {
                y_val.push(y_raw[r]);
            }
            let (fold_mean, _) = mean_and_std(&y_tr);

            // 1. Fit SplineRidgeModel on fold training rows
            let ridge = SplineRidgeModel::fit(x_raw, y_raw, &tr_rows, n_features, &pipe);
            let mut ridge_buf = vec![0.0_f64; pipe.ridge_basis_dim()];
            let mut ridge_pred_tr = Vec::with_capacity(tr_rows.len());
            for &r in &tr_rows {
                let row_s = &x_raw[r * n_features..(r + 1) * n_features];
                ridge_pred_tr.push(ridge.predict_row(row_s, &pipe, &mut ridge_buf));
            }
            let mut ridge_pred_val = Vec::with_capacity(val_rows.len());
            for &r in &val_rows {
                let row_s = &x_raw[r * n_features..(r + 1) * n_features];
                ridge_pred_val.push(ridge.predict_row(row_s, &pipe, &mut ridge_buf));
            }

            // 2. Quantize fold train and validation rows
            let binned_tr = quant.quantize_rows(x_raw, &tr_rows, n_features, &pipe);
            let binned_val = quant.quantize_rows(x_raw, &val_rows, n_features, &pipe);
            let n_tf = quant.n_tree_features;

            // 3. Fit ObliviousGbdtModel
            let (obliv_model, opt_obliv) = ObliviousGbdtModel::fit_with_validation(
                &binned_tr,
                &y_tr,
                Some((&binned_val, &y_val)),
                n_tf,
                &quant,
                max_trees_obliv,
                lr_obliv,
                l2_reg,
                rng,
            );
            sum_opt_obliv += opt_obliv.max(10);

            // 4. Fit WiseGbdtModel on damped Ridge prior
            let (wise_model, opt_wise) = WiseGbdtModel::fit_with_validation(
                &binned_tr,
                &y_tr,
                &ridge_pred_tr,
                Some((&binned_val, &y_val, &ridge_pred_val)),
                n_tf,
                &quant,
                max_trees_wise,
                lr_wise,
                l2_reg * 1.25,
                rng,
            );
            sum_opt_wise += opt_wise.max(10);

            // 5. Fit AsinhSmearingGbdtModel
            let (asinh_model, opt_asinh) = AsinhSmearingGbdtModel::fit_with_validation(
                &binned_tr,
                &y_tr,
                Some((&binned_val, &y_val)),
                n_tf,
                &quant,
                (max_trees_obliv * 3) / 4,
                lr_obliv,
                l2_reg,
                rng,
            );
            sum_opt_asinh += opt_asinh.max(10);

            // 6. Fit MetricKnnModel
            let knn_model = if use_knn {
                Some(MetricKnnModel::fit(x_raw, y_raw, &tr_rows, n_features, &pipe))
            } else {
                None
            };

            let unit = ExpertUnit {
                pipeline: pipe,
                quantizer: quant,
                ridge,
                oblivious_gbdt: obliv_model,
                wise_gbdt: wise_model,
                asinh_gbdt: asinh_model,
                knn: knn_model,
                target_mean: fold_mean,
            };

            for (v_idx, &orig_r) in val_rows.iter().enumerate() {
                let row_s = &x_raw[orig_r * n_features..(orig_r + 1) * n_features];
                let preds = unit.predict_experts(row_s);
                oof_preds[val_local_pos[v_idx]] = preds;
            }

            fold_units.push(unit);
        }

        let expert_weights = fit_simplex_stacking_weights(&oof_preds, &y_working);

        // Compute Out-Of-Fold gentle scale calibration around target_mean
        let mut num_cal = 0.0_f64;
        let mut den_cal = 0.0_f64;
        let mut oof_ens_mse = 0.0_f64;
        for i in 0..nw {
            let mut p = 0.0_f64;
            for m in 0..NUM_EXPERTS {
                p += expert_weights[m] * oof_preds[i][m];
            }
            let err = p - y_working[i];
            oof_ens_mse += err * err;
            let pc = p - target_mean;
            let yc = y_working[i] - target_mean;
            num_cal += pc * yc;
            den_cal += pc * pc;
        }
        oof_ens_mse /= nw as f64;

        // If OOF ensemble is somehow worse than constant target_mean, fall back cleanly to target_mean
        if oof_ens_mse >= target_std * target_std * 0.999 {
            return Ok(Self {
                n_features,
                is_constant: true,
                target_mean,
                target_min,
                target_max,
                expert_weights: [1.0, 0.0, 0.0, 0.0, 0.0],
                calibration_scale: 1.0,
                fold_units: Vec::new(),
                full_unit: None,
            });
        }

        let raw_cal = if den_cal > 1e-12 {
            num_cal / den_cal
        } else {
            1.0
        };
        // Shrink calibration scale toward 1.0
        let calibration_scale = scalar_clip(0.5 * raw_cal + 0.5 * 1.0, 0.85, 1.15);

        // Fit full-data ExpertUnit using the CV-selected iteration counts scaled by (n_folds / (n_folds - 1))
        let scale_trees = (n_folds as f64) / ((n_folds - 1).max(1) as f64);
        let full_obliv_trees = (((sum_opt_obliv as f64) / (n_folds as f64)) * scale_trees).round()
            as usize;
        let full_wise_trees =
            (((sum_opt_wise as f64) / (n_folds as f64)) * scale_trees).round() as usize;
        let full_asinh_trees = (((sum_opt_asinh as f64) / (n_folds as f64)) * scale_trees).round()
            as usize;

        let full_pipe =
            FeaturePipeline::fit(x_raw, y_raw, &working_indices, n_features, max_active_features);
        let full_quant = HistogramQuantizer::fit(x_raw, &working_indices, n_features, &full_pipe);
        let full_ridge =
            SplineRidgeModel::fit(x_raw, y_raw, &working_indices, n_features, &full_pipe);

        let mut full_ridge_preds = Vec::with_capacity(nw);
        let mut full_ridge_buf = vec![0.0_f64; full_pipe.ridge_basis_dim()];
        for &r in &working_indices {
            let row_s = &x_raw[r * n_features..(r + 1) * n_features];
            full_ridge_preds.push(full_ridge.predict_row(row_s, &full_pipe, &mut full_ridge_buf));
        }

        let full_binned = full_quant.quantize_rows(x_raw, &working_indices, n_features, &full_pipe);
        let full_ntf = full_quant.n_tree_features;

        let (full_obliv, _) = ObliviousGbdtModel::fit_with_validation(
            &full_binned,
            &y_working,
            None,
            full_ntf,
            &full_quant,
            full_obliv_trees.clamp(10, max_trees_obliv),
            lr_obliv,
            l2_reg,
            rng,
        );

        let (full_wise, _) = WiseGbdtModel::fit_with_validation(
            &full_binned,
            &y_working,
            &full_ridge_preds,
            None,
            full_ntf,
            &full_quant,
            full_wise_trees.clamp(10, max_trees_wise),
            lr_wise,
            l2_reg * 1.25,
            rng,
        );

        let (full_asinh, _) = AsinhSmearingGbdtModel::fit_with_validation(
            &full_binned,
            &y_working,
            None,
            full_ntf,
            &full_quant,
            full_asinh_trees.clamp(10, max_trees_obliv),
            lr_obliv,
            l2_reg,
            rng,
        );

        let full_knn = if use_knn {
            Some(MetricKnnModel::fit(
                x_raw,
                y_raw,
                &working_indices,
                n_features,
                &full_pipe,
            ))
        } else {
            None
        };

        let full_unit = Some(ExpertUnit {
            pipeline: full_pipe,
            quantizer: full_quant,
            ridge: full_ridge,
            oblivious_gbdt: full_obliv,
            wise_gbdt: full_wise,
            asinh_gbdt: full_asinh,
            knn: full_knn,
            target_mean,
        });

        Ok(Self {
            n_features,
            is_constant: false,
            target_mean,
            target_min,
            target_max,
            expert_weights,
            calibration_scale,
            fold_units,
            full_unit,
        })
    }

    fn predict_single_row(&self, row_slice: &[f64]) -> f64 {
        if self.is_constant || self.n_features == 0 {
            return self.target_mean;
        }

        let mut fold_expert_avg = [0.0_f64; NUM_EXPERTS];
        let nf = self.fold_units.len();
        if nf > 0 {
            for unit in &self.fold_units {
                let p = unit.predict_experts(row_slice);
                for m in 0..NUM_EXPERTS {
                    fold_expert_avg[m] += p[m];
                }
            }
            let inv_nf = 1.0 / (nf as f64);
            for m in 0..NUM_EXPERTS {
                fold_expert_avg[m] *= inv_nf;
            }
        }

        let combined_experts = if let Some(full) = &self.full_unit {
            let full_p = full.predict_experts(row_slice);
            if nf > 0 {
                let mut blended = [0.0_f64; NUM_EXPERTS];
                for m in 0..NUM_EXPERTS {
                    blended[m] = 0.55 * fold_expert_avg[m] + 0.45 * full_p[m];
                }
                blended
            } else {
                full_p
            }
        } else {
            fold_expert_avg
        };

        let mut raw_ens = 0.0_f64;
        for m in 0..NUM_EXPERTS {
            raw_ens += self.expert_weights[m] * combined_experts[m];
        }

        let calibrated =
            self.target_mean + self.calibration_scale * (raw_ens - self.target_mean);

        // Smooth wide safety damper outside training target range
        let span = (self.target_max - self.target_min).abs().max(1e-6);
        let lo = self.target_min - 1.25 * span;
        let hi = self.target_max + 1.25 * span;
        let guarded = if calibrated < lo {
            lo - 0.5 * span * ((lo - calibrated) / (0.5 * span)).asinh()
        } else if calibrated > hi {
            hi + 0.5 * span * ((calibrated - hi) / (0.5 * span)).asinh()
        } else {
            calibrated
        };

        if guarded.is_finite() {
            guarded
        } else {
            self.target_mean
        }
    }
}
