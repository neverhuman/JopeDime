#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BTreeMap;

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

#[derive(Clone, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let s = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed ^ 0xBF58_476D_1CE4_E5B9
        };
        Self { state: s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        if len <= 1 {
            return;
        }
        for i in (1..len).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

#[inline]
fn total_cmp_f64(a: f64, b: f64) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else if a == b {
        if a == 0.0 {
            let a_neg = a.to_bits() >> 63;
            let b_neg = b.to_bits() >> 63;
            b_neg.cmp(&a_neg)
        } else {
            Ordering::Equal
        }
    } else {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            (false, false) => Ordering::Equal,
        }
    }
}

#[inline]
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(8.0);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

#[inline]
fn normal_quantile(p: f64) -> f64 {
    let u = p.clamp(1e-6, 1.0 - 1e-6);
    // Rational approximation to inverse standard normal CDF (Wichura / Beasley-Springer-Moro style)
    let q = u - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        let num = ((((((2.5090809287301226727e3 * r + 3.3430575583588128105e4) * r
            + 6.7265770927008700853e4)
            * r
            + 4.5921953931549871457e4)
            * r
            + 1.3731693765509461125e4)
            * r
            + 1.9715909503065514427e3)
            * r
            + 1.3314166789178437745e2)
            * r
            + 3.3871328727963666080e0;
        let den = ((((((5.2264952788528545610e3 * r + 2.8729085735721942674e4) * r
            + 3.9307895800092710610e4)
            * r
            + 2.1213794301586595867e4)
            * r
            + 5.3941960214247511077e3)
            * r
            + 6.8718700749205790830e2)
            * r
            + 4.2313330701600911252e1)
            * r
            + 1.0;
        q * num / den
    } else {
        let r = if q < 0.0 { u } else { 1.0 - u };
        let s = (-r.ln()).sqrt();
        let num = ((((((7.74545014278341407640e-4 * s + 2.27238449892691845833e-2) * s
            + 2.41780725177450611770e-1)
            * s
            + 1.27045825245236838258e0)
            * s
            + 3.64784832476320460504e0)
            * s
            + 5.76949722146069140550e0)
            * s
            + 4.63033784615654529590e0)
            * s
            + 1.42343711074968357734e0;
        let den = ((((((1.05075007164441684324e-9 * s + 5.47593808499534494600e-4) * s
            + 1.51986665636164571966e-2)
            * s
            + 1.48103976427480074590e-1)
            * s
            + 6.89767334985100004550e-1)
            * s
            + 1.67638483018380384940e0)
            * s
            + 2.05319162663775882187e0)
            * s
            + 1.0;
        let val = num / den;
        if q < 0.0 {
            -val
        } else {
            val
        }
    }
}

#[derive(Clone, Debug)]
enum TargetTransform {
    Identity {
        mean: f64,
        std: f64,
    },
    ShiftedLog {
        shift: f64,
        t_mean: f64,
        t_std: f64,
    },
    Asinh {
        median: f64,
        scale: f64,
        t_mean: f64,
        t_std: f64,
    },
}

impl TargetTransform {
    fn fit(y_raw: &[f64]) -> (Self, Self) {
        let n = y_raw.len();
        let mut sum = 0.0;
        for &y in y_raw {
            sum += y;
        }
        let mean = sum / (n.max(1) as f64);
        let mut sq = 0.0;
        for &y in y_raw {
            let d = y - mean;
            sq += d * d;
        }
        let std = (sq / (n.max(1) as f64)).sqrt().max(1e-12);
        let identity = TargetTransform::Identity { mean, std };

        if n < 20 {
            return (identity.clone(), identity);
        }

        let mut sorted = y_raw.to_vec();
        sorted.sort_by(|&a, &b| total_cmp_f64(a, b));
        let med = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
        };
        let q25 = sorted[n / 4];
        let q75 = sorted[(3 * n) / 4];
        let iqr = (q75 - q25).max(1e-9);
        let y_min = sorted[0];
        let y_max = sorted[n - 1];

        // Compute standardized third and fourth moments (skewness and kurtosis)
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        for &y in y_raw {
            let z = (y - mean) / std;
            let z2 = z * z;
            m3 += z2 * z;
            m4 += z2 * z2;
        }
        let skew = m3 / (n as f64);
        let kurt = m4 / (n as f64);

        let alt = if y_min >= -1e-9 && skew > 1.75 && (y_max - med) > 3.0 * (med - y_min).max(iqr) {
            let shift = (-y_min).max(0.0) + 0.1 * iqr;
            let mut t_sum = 0.0;
            let mut t_vals = Vec::with_capacity(n);
            for &y in y_raw {
                let tv = (y + shift).max(1e-12).ln();
                t_sum += tv;
                t_vals.push(tv);
            }
            let t_mean = t_sum / (n as f64);
            let mut t_sq = 0.0;
            for &tv in &t_vals {
                let d = tv - t_mean;
                t_sq += d * d;
            }
            let t_std = (t_sq / (n as f64)).sqrt().max(1e-12);
            TargetTransform::ShiftedLog {
                shift,
                t_mean,
                t_std,
            }
        } else if kurt > 9.0 || skew.abs() > 2.2 {
            let scale = (iqr / 1.349).max(1e-6);
            let mut t_sum = 0.0;
            let mut t_vals = Vec::with_capacity(n);
            for &y in y_raw {
                let tv = ((y - med) / scale).asinh();
                t_sum += tv;
                t_vals.push(tv);
            }
            let t_mean = t_sum / (n as f64);
            let mut t_sq = 0.0;
            for &tv in &t_vals {
                let d = tv - t_mean;
                t_sq += d * d;
            }
            let t_std = (t_sq / (n as f64)).sqrt().max(1e-12);
            TargetTransform::Asinh {
                median: med,
                scale,
                t_mean,
                t_std,
            }
        } else {
            identity.clone()
        };

        (identity, alt)
    }

    #[inline]
    fn forward(&self, y: f64) -> f64 {
        match *self {
            TargetTransform::Identity { mean, std } => (y - mean) / std,
            TargetTransform::ShiftedLog {
                shift,
                t_mean,
                t_std,
            } => ((y + shift).max(1e-12).ln() - t_mean) / t_std,
            TargetTransform::Asinh {
                median,
                scale,
                t_mean,
                t_std,
            } => (((y - median) / scale).asinh() - t_mean) / t_std,
        }
    }

    #[inline]
    fn inverse(&self, z: f64, smear_factor: f64) -> f64 {
        match *self {
            TargetTransform::Identity { mean, std } => mean + z * std,
            TargetTransform::ShiftedLog {
                shift,
                t_mean,
                t_std,
            } => {
                let u = (t_mean + z * t_std).clamp(-30.0, 30.0);
                u.exp() * smear_factor - shift
            }
            TargetTransform::Asinh {
                median,
                scale,
                t_mean,
                t_std,
            } => {
                let u = (t_mean + z * t_std).clamp(-25.0, 25.0);
                let sinh_u = 0.5 * (u.exp() - (-u).exp());
                median + scale * sinh_u * smear_factor
            }
        }
    }
}

#[derive(Clone, Debug)]
struct FeatureStats {
    median: f64,
    iqr_scale: f64,
    mean: f64,
    std: f64,
    missing_rate: f64,
    use_missing_indicator: bool,
    quantiles: Vec<f64>,
    is_categorical_candidate: bool,
    freq_map: BTreeMap<u64, f64>,
    spline_knots: [f64; 3],
}

#[inline]
fn canonical_bits(v: f64) -> u64 {
    if v == 0.0 {
        0.0f64.to_bits()
    } else if v.is_nan() {
        u64::MAX
    } else {
        v.to_bits()
    }
}

#[derive(Clone, Debug)]
struct Preprocessor {
    n_raw_features: usize,
    active_raw_cols: Vec<usize>,
    stats: Vec<FeatureStats>,
    include_row_summaries: bool,
    row_summary_means: [f64; 8],
    row_summary_stds: [f64; 8],
    top_interaction_pairs: Vec<(usize, usize, u8)>, // (ia, ib, op_type: 0=prod, 1=diff, 2=ratio)
    pair_means: Vec<f64>,
    pair_stds: Vec<f64>,
    cat_col_indices: Vec<usize>, // indices into active_raw_cols that are categorical candidates
}

impl Preprocessor {
    fn fit(raw_x: &[f64], n_rows: usize, n_features: usize, y_norm: &[f64]) -> Self {
        let stride = if n_rows > 16000 {
            (n_rows / 5000).max(1)
        } else {
            1
        };

        let mut active_raw_cols = Vec::new();
        if n_features <= 1024 {
            for j in 0..n_features {
                active_raw_cols.push(j);
            }
        } else {
            // Dual linear + rank-based / non-monotonic screening
            let mut scored: Vec<(f64, usize)> = Vec::with_capacity(n_features);
            let mut y_mean = 0.0;
            let mut y_cnt = 0usize;
            for i in (0..n_rows).step_by(stride) {
                y_mean += y_norm[i];
                y_cnt += 1;
            }
            y_mean /= (y_cnt.max(1)) as f64;

            for j in 0..n_features {
                let mut sum_x = 0.0;
                let mut sum_x2 = 0.0;
                let mut sum_xy = 0.0;
                let mut sum_abs_xy = 0.0;
                let mut cnt = 0usize;
                let mut miss_y_sum = 0.0;
                let mut miss_cnt = 0usize;

                for i in (0..n_rows).step_by(stride) {
                    let v = raw_x[i * n_features + j];
                    let yv = y_norm[i] - y_mean;
                    if v.is_nan() {
                        miss_y_sum += yv;
                        miss_cnt += 1;
                    } else {
                        let sv = v.asinh();
                        sum_x += sv;
                        sum_x2 += sv * sv;
                        sum_xy += sv * yv;
                        sum_abs_xy += sv.abs() * yv;
                        cnt += 1;
                    }
                }

                let mut score = 0.0;
                if cnt > 2 {
                    let mx = sum_x / (cnt as f64);
                    let vx = (sum_x2 / (cnt as f64) - mx * mx).max(0.0);
                    if vx > 1e-18 {
                        let sd = vx.sqrt() + 1e-12;
                        let c1 = (sum_xy / (cnt as f64)).abs() / sd;
                        let c2 = (sum_abs_xy / (cnt as f64)).abs() / sd;
                        score = c1.max(0.5 * c2);
                    }
                }
                if miss_cnt > 0 && miss_cnt < y_cnt {
                    let miss_score = (miss_y_sum / (y_cnt as f64)).abs();
                    if miss_score > score {
                        score = miss_score;
                    }
                }
                scored.push((score, j));
            }
            scored.sort_by(|a, b| total_cmp_f64(b.0, a.0));
            for k in 0..1024.min(scored.len()) {
                active_raw_cols.push(scored[k].1);
            }
            active_raw_cols.sort_unstable();
        }

        let n_active = active_raw_cols.len();
        let mut stats = Vec::with_capacity(n_active);
        let mut col_buf = Vec::with_capacity((n_rows / stride) + 1);
        let mut cat_col_indices = Vec::new();

        for (act_idx, &raw_j) in active_raw_cols.iter().enumerate() {
            col_buf.clear();
            let mut miss_cnt = 0usize;
            let mut tot_cnt = 0usize;
            let mut counts_map: BTreeMap<u64, usize> = BTreeMap::new();

            for i in (0..n_rows).step_by(stride) {
                tot_cnt += 1;
                let v = raw_x[i * n_features + raw_j];
                if v.is_nan() {
                    miss_cnt += 1;
                } else {
                    col_buf.push(v);
                    if counts_map.len() <= 512 {
                        *counts_map.entry(canonical_bits(v)).or_insert(0) += 1;
                    }
                }
            }
            col_buf.sort_by(|&a, &b| total_cmp_f64(a, b));

            let missing_rate = (miss_cnt as f64) / (tot_cnt.max(1) as f64);
            let use_missing_indicator = miss_cnt > 0 && miss_cnt < tot_cnt;

            let n_unique = counts_map.len();
            let is_categorical_candidate =
                n_unique >= 3 && n_unique <= 256 && (n_unique * 3 <= col_buf.len().max(1));

            let mut freq_map = BTreeMap::new();
            if is_categorical_candidate && cat_col_indices.len() < 24 {
                cat_col_indices.push(act_idx);
                let inv_tot = 1.0 / (tot_cnt.max(1) as f64);
                for (k, c) in counts_map {
                    freq_map.insert(k, (c as f64) * inv_tot);
                }
            }

            let (median, iqr_scale, mean, std, quantiles, spline_knots) = if col_buf.is_empty() {
                (0.0, 1.0, 0.0, 1.0, vec![0.0], [-0.67, 0.0, 0.67])
            } else {
                let m = col_buf.len();
                let med = if m % 2 == 1 {
                    col_buf[m / 2]
                } else {
                    0.5 * (col_buf[m / 2 - 1] + col_buf[m / 2])
                };
                let q25 = col_buf[m / 4];
                let q75 = col_buf[(3 * m) / 4];
                let mut iqr = (q75 - q25) / 1.349;
                if iqr.abs() < 1e-9 || !iqr.is_finite() {
                    let p10 = col_buf[m / 10];
                    let p90 = col_buf[(9 * m) / 10];
                    iqr = (p90 - p10) / 2.563;
                }
                if iqr.abs() < 1e-9 || !iqr.is_finite() {
                    let range = col_buf[m - 1] - col_buf[0];
                    iqr = if range.abs() > 1e-9 && range.is_finite() {
                        range
                    } else {
                        1.0
                    };
                }

                let mut s1 = 0.0;
                let mut s2 = 0.0;
                for &v in &col_buf {
                    let z = ((v - med) / iqr).clamp(-40.0, 40.0);
                    s1 += z;
                    s2 += z * z;
                }
                let mn = s1 / (m as f64);
                let var = (s2 / (m as f64) - mn * mn).max(0.0);
                let sd = if var > 1e-12 { var.sqrt() } else { 1.0 };

                let n_q = 127.min(m);
                let mut qs = Vec::with_capacity(n_q);
                for q_idx in 0..n_q {
                    let pos = ((q_idx * (m - 1)) / (n_q.max(1) - 1).max(1)).min(m - 1);
                    let qv = col_buf[pos];
                    if qs.is_empty() || (qv - *qs.last().unwrap()).abs() > 1e-12 {
                        qs.push(qv);
                    }
                }
                if qs.is_empty() {
                    qs.push(med);
                }

                let k1 = (((q25 - med) / iqr) - mn) / sd;
                let k2 = (0.0 - mn) / sd;
                let k3 = (((q75 - med) / iqr) - mn) / sd;
                (med, iqr, mn, sd, qs, [k1, k2, k3])
            };

            stats.push(FeatureStats {
                median,
                iqr_scale,
                mean,
                std,
                missing_rate,
                use_missing_indicator,
                quantiles,
                is_categorical_candidate,
                freq_map,
                spline_knots,
            });
        }

        let include_row_summaries = n_active >= 2;
        let mut row_summary_means = [0.0; 8];
        let mut row_summary_stds = [1.0; 8];

        if include_row_summaries {
            let mut sum_s = [0.0; 8];
            let mut sum_s2 = [0.0; 8];
            let mut r_cnt = 0usize;
            for i in (0..n_rows).step_by(stride) {
                let row = &raw_x[i * n_features..(i + 1) * n_features];
                let rs = Self::compute_raw_row_summaries(row, &active_raw_cols, &stats);
                for k in 0..8 {
                    sum_s[k] += rs[k];
                    sum_s2[k] += rs[k] * rs[k];
                }
                r_cnt += 1;
            }
            let denom = (r_cnt.max(1)) as f64;
            for k in 0..8 {
                let mn = sum_s[k] / denom;
                let vr = (sum_s2[k] / denom - mn * mn).max(0.0);
                row_summary_means[k] = mn;
                row_summary_stds[k] = if vr > 1e-12 { vr.sqrt() } else { 1.0 };
            }
        }

        // Rank active features by max of linear and quantile-normal correlation with y_norm
        let mut col_corrs: Vec<(f64, usize)> = Vec::with_capacity(n_active);
        let mut ym = 0.0;
        let mut y_c = 0usize;
        for i in (0..n_rows).step_by(stride) {
            ym += y_norm[i];
            y_c += 1;
        }
        ym /= (y_c.max(1)) as f64;

        for (idx, &raw_j) in active_raw_cols.iter().enumerate() {
            let st = &stats[idx];
            let mut sxy = 0.0;
            let mut sx2 = 0.0;
            let mut sqy = 0.0;
            let mut sq2 = 0.0;
            for i in (0..n_rows).step_by(stride) {
                let v = raw_x[i * n_features + raw_j];
                let (z, qz) = if v.is_nan() {
                    (0.0, 0.0)
                } else {
                    let z_val =
                        (((v - st.median) / st.iqr_scale).clamp(-40.0, 40.0) - st.mean) / st.std;
                    let q_val = Self::empirical_normal_score(v, &st.quantiles);
                    (z_val, q_val)
                };
                let dy = y_norm[i] - ym;
                sxy += z * dy;
                sx2 += z * z;
                sqy += qz * dy;
                sq2 += qz * qz;
            }
            let c1 = if sx2 > 1e-12 {
                sxy.abs() / sx2.sqrt()
            } else {
                0.0
            };
            let c2 = if sq2 > 1e-12 {
                sqy.abs() / sq2.sqrt()
            } else {
                0.0
            };
            col_corrs.push((c1.max(c2), idx));
        }
        col_corrs.sort_by(|a, b| total_cmp_f64(b.0, a.0));

        // Build interaction pairs: products for top 8, differences and smooth ratios for top 5
        let top_prod = 8.min(n_active);
        let top_extra = 5.min(n_active);
        let mut top_interaction_pairs = Vec::new();
        for a in 0..top_prod {
            let idx_a = col_corrs[a].1;
            for b in a..top_prod {
                let idx_b = col_corrs[b].1;
                top_interaction_pairs.push((idx_a, idx_b, 0u8));
            }
        }
        for a in 0..top_extra {
            let idx_a = col_corrs[a].1;
            for b in (a + 1)..top_extra {
                let idx_b = col_corrs[b].1;
                top_interaction_pairs.push((idx_a, idx_b, 1u8)); // difference
                top_interaction_pairs.push((idx_a, idx_b, 2u8)); // damped ratio
            }
        }

        let n_pairs = top_interaction_pairs.len();
        let mut pair_means = vec![0.0; n_pairs];
        let mut pair_stds = vec![1.0; n_pairs];
        if n_pairs > 0 {
            let mut ps1 = vec![0.0; n_pairs];
            let mut ps2 = vec![0.0; n_pairs];
            let mut pcnt = 0usize;
            for i in (0..n_rows).step_by(stride) {
                pcnt += 1;
                for (p_idx, &(ia, ib, op)) in top_interaction_pairs.iter().enumerate() {
                    let va = raw_x[i * n_features + active_raw_cols[ia]];
                    let vb = raw_x[i * n_features + active_raw_cols[ib]];
                    let val = Self::eval_pair_op(va, vb, &stats[ia], &stats[ib], op);
                    let c_val = if val.is_nan() { 0.0 } else { val };
                    ps1[p_idx] += c_val;
                    ps2[p_idx] += c_val * c_val;
                }
            }
            let denom = (pcnt.max(1)) as f64;
            for p_idx in 0..n_pairs {
                let mn = ps1[p_idx] / denom;
                let vr = (ps2[p_idx] / denom - mn * mn).max(0.0);
                pair_means[p_idx] = mn;
                pair_stds[p_idx] = if vr > 1e-12 { vr.sqrt() } else { 1.0 };
            }
        }

        Self {
            n_raw_features: n_features,
            active_raw_cols,
            stats,
            include_row_summaries,
            row_summary_means,
            row_summary_stds,
            top_interaction_pairs,
            pair_means,
            pair_stds,
            cat_col_indices,
        }
    }

    #[inline]
    fn empirical_normal_score(v: f64, quantiles: &[f64]) -> f64 {
        let m = quantiles.len();
        if m <= 1 {
            return 0.0;
        }
        let rank_u = match quantiles.binary_search_by(|&q| total_cmp_f64(q, v)) {
            Ok(pos) => (pos as f64 + 0.5) / (m as f64),
            Err(pos) => {
                if pos == 0 {
                    0.5 / (m as f64)
                } else if pos >= m {
                    ((m as f64) - 0.5) / (m as f64)
                } else {
                    let lo = quantiles[pos - 1];
                    let hi = quantiles[pos];
                    let frac = if (hi - lo).abs() > 1e-12 {
                        ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    ((pos as f64) - 0.5 + frac) / (m as f64)
                }
            }
        };
        normal_quantile(rank_u)
    }

    #[inline]
    fn eval_pair_op(va: f64, vb: f64, sta: &FeatureStats, stb: &FeatureStats, op: u8) -> f64 {
        if va.is_nan() || vb.is_nan() {
            return f64::NAN;
        }
        let za = (((va - sta.median) / sta.iqr_scale).clamp(-15.0, 15.0) - sta.mean) / sta.std;
        let zb = (((vb - stb.median) / stb.iqr_scale).clamp(-15.0, 15.0) - stb.mean) / stb.std;
        match op {
            0 => (za * zb).clamp(-50.0, 50.0),
            1 => (za - zb).clamp(-30.0, 30.0),
            _ => (za / (zb.abs() + 1.0)).clamp(-25.0, 25.0),
        }
    }

    fn compute_raw_row_summaries(
        row: &[f64],
        active_cols: &[usize],
        stats: &[FeatureStats],
    ) -> [f64; 8] {
        let mut r_sum = 0.0;
        let mut r_l1 = 0.0;
        let mut r_l2 = 0.0;
        let mut r_min = f64::INFINITY;
        let mut r_max = f64::NEG_INFINITY;
        let mut finite_cnt = 0usize;
        let mut miss_cnt = 0usize;

        for (idx, &raw_j) in active_cols.iter().enumerate() {
            let v = row[raw_j];
            if v.is_nan() {
                miss_cnt += 1;
            } else {
                let st = &stats[idx];
                let z = (((v - st.median) / st.iqr_scale).clamp(-30.0, 30.0) - st.mean) / st.std;
                r_sum += z;
                r_l1 += z.abs();
                r_l2 += z * z;
                if z < r_min {
                    r_min = z;
                }
                if z > r_max {
                    r_max = z;
                }
                finite_cnt += 1;
            }
        }

        if finite_cnt == 0 {
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, miss_cnt as f64];
        }
        let fc = finite_cnt as f64;
        let r_mean = r_sum / fc;
        let r_l2_norm = r_l2.sqrt();
        let var = (r_l2 / fc - r_mean * r_mean).max(0.0);
        let r_std = var.sqrt();
        [
            r_sum,
            r_mean,
            r_l1,
            r_l2_norm,
            r_min,
            r_max,
            r_std,
            (miss_cnt as f64) / (active_cols.len().max(1) as f64),
        ]
    }

    fn base_tree_feature_dim(&self) -> usize {
        self.active_raw_cols.len()
            + self.cat_col_indices.len() // frequency encodings
            + if self.include_row_summaries { 8 } else { 0 }
            + self.top_interaction_pairs.len()
    }

    fn total_tree_feature_dim(&self) -> usize {
        self.base_tree_feature_dim() + self.cat_col_indices.len() // + OOF Bayesian target encodings
    }

    fn transform_row_for_tree_base(&self, row: &[f64], out: &mut [f64]) {
        let n_act = self.active_raw_cols.len();
        for (idx, &raw_j) in self.active_raw_cols.iter().enumerate() {
            out[idx] = row[raw_j];
        }
        let mut offset = n_act;
        for &cat_act_idx in &self.cat_col_indices {
            let v = row[self.active_raw_cols[cat_act_idx]];
            if v.is_nan() {
                out[offset] = 0.0;
            } else {
                let freq = self.stats[cat_act_idx]
                    .freq_map
                    .get(&canonical_bits(v))
                    .copied()
                    .unwrap_or(0.0);
                out[offset] = freq;
            }
            offset += 1;
        }
        if self.include_row_summaries {
            let rs = Self::compute_raw_row_summaries(row, &self.active_raw_cols, &self.stats);
            for k in 0..8 {
                out[offset + k] = (rs[k] - self.row_summary_means[k]) / self.row_summary_stds[k];
            }
            offset += 8;
        }
        for (p_idx, &(ia, ib, op)) in self.top_interaction_pairs.iter().enumerate() {
            let va = row[self.active_raw_cols[ia]];
            let vb = row[self.active_raw_cols[ib]];
            let val = Self::eval_pair_op(va, vb, &self.stats[ia], &self.stats[ib], op);
            if val.is_nan() {
                out[offset + p_idx] = f64::NAN;
            } else {
                out[offset + p_idx] = (val - self.pair_means[p_idx]) / self.pair_stds[p_idx];
            }
        }
    }

    fn linear_feature_dim(&self) -> usize {
        let mut d = 0usize;
        let use_splines = self.active_raw_cols.len() <= 96;
        for st in &self.stats {
            // z, asinh(z), quantile_normal(v), erf(z/sqrt(2))
            d += 4;
            if use_splines {
                d += 3; // 3 piecewise-linear hinge knots
            }
            if st.use_missing_indicator {
                d += 1;
            }
        }
        d += self.cat_col_indices.len(); // frequency encodings
        if self.include_row_summaries {
            d += 8;
        }
        d += self.top_interaction_pairs.len();
        d + self.cat_col_indices.len() // + Bayesian target encodings at the tail
    }

    fn transform_row_for_linear_base(&self, row: &[f64], out: &mut [f64]) -> usize {
        let mut pos = 0usize;
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let use_splines = self.active_raw_cols.len() <= 96;

        for (idx, &raw_j) in self.active_raw_cols.iter().enumerate() {
            let v = row[raw_j];
            let st = &self.stats[idx];
            let (z, qz, is_miss) = if v.is_nan() {
                (0.0, 0.0, 1.0)
            } else {
                let scaled = ((v - st.median) / st.iqr_scale).clamp(-25.0, 25.0);
                let z_val = ((scaled - st.mean) / st.std).clamp(-15.0, 15.0);
                let q_val = Self::empirical_normal_score(v, &st.quantiles);
                (z_val, q_val, 0.0)
            };
            out[pos] = z;
            out[pos + 1] = z.asinh();
            out[pos + 2] = qz;
            out[pos + 3] = erf_approx(z * inv_sqrt2);
            pos += 4;

            if use_splines {
                out[pos] = (z - st.spline_knots[0]).max(0.0).clamp(0.0, 15.0);
                out[pos + 1] = (z - st.spline_knots[1]).max(0.0).clamp(0.0, 15.0);
                out[pos + 2] = (z - st.spline_knots[2]).max(0.0).clamp(0.0, 15.0);
                pos += 3;
            }
            if st.use_missing_indicator {
                let mr = st.missing_rate;
                let denom = (mr * (1.0 - mr)).sqrt().max(1e-6);
                out[pos] = (is_miss - mr) / denom;
                pos += 1;
            }
        }
        for &cat_act_idx in &self.cat_col_indices {
            let v = row[self.active_raw_cols[cat_act_idx]];
            let freq = if v.is_nan() {
                0.0
            } else {
                self.stats[cat_act_idx]
                    .freq_map
                    .get(&canonical_bits(v))
                    .copied()
                    .unwrap_or(0.0)
            };
            out[pos] = freq;
            pos += 1;
        }
        if self.include_row_summaries {
            let rs = Self::compute_raw_row_summaries(row, &self.active_raw_cols, &self.stats);
            for k in 0..8 {
                out[pos + k] = ((rs[k] - self.row_summary_means[k]) / self.row_summary_stds[k])
                    .clamp(-15.0, 15.0);
            }
            pos += 8;
        }
        for (p_idx, &(ia, ib, op)) in self.top_interaction_pairs.iter().enumerate() {
            let va = row[self.active_raw_cols[ia]];
            let vb = row[self.active_raw_cols[ib]];
            let val = Self::eval_pair_op(va, vb, &self.stats[ia], &self.stats[ib], op);
            let norm_val = if val.is_nan() {
                0.0
            } else {
                ((val - self.pair_means[p_idx]) / self.pair_stds[p_idx]).clamp(-15.0, 15.0)
            };
            out[pos + p_idx] = norm_val;
        }
        pos + self.top_interaction_pairs.len()
    }
}

#[derive(Clone, Debug)]
struct FoldTargetEncoder {
    prior_mean: f64,
    cat_maps: Vec<BTreeMap<u64, f64>>,
}

impl FoldTargetEncoder {
    fn fit(
        raw_x: &[f64],
        n_features: usize,
        y_norm: &[f64],
        train_idx: &[usize],
        prep: &Preprocessor,
    ) -> Self {
        let mut prior_mean = 0.0;
        if !train_idx.is_empty() {
            for &i in train_idx {
                prior_mean += y_norm[i];
            }
            prior_mean /= train_idx.len() as f64;
        }

        let smooth_m = 12.0;
        let mut cat_maps = Vec::with_capacity(prep.cat_col_indices.len());

        for &cat_act_idx in &prep.cat_col_indices {
            let raw_j = prep.active_raw_cols[cat_act_idx];
            let mut sum_cnt: BTreeMap<u64, (f64, usize)> = BTreeMap::new();
            for &i in train_idx {
                let v = raw_x[i * n_features + raw_j];
                let key = canonical_bits(v);
                let entry = sum_cnt.entry(key).or_insert((0.0, 0));
                entry.0 += y_norm[i];
                entry.1 += 1;
            }
            let mut encoded_map = BTreeMap::new();
            for (k, (s, c)) in sum_cnt {
                let cf = c as f64;
                let te = (s + smooth_m * prior_mean) / (cf + smooth_m);
                encoded_map.insert(k, (te - prior_mean).clamp(-5.0, 5.0));
            }
            cat_maps.push(encoded_map);
        }

        Self {
            prior_mean,
            cat_maps,
        }
    }

    #[inline]
    fn encode_row(&self, row: &[f64], prep: &Preprocessor, out_slice: &mut [f64]) {
        for (c_pos, &cat_act_idx) in prep.cat_col_indices.iter().enumerate() {
            let raw_j = prep.active_raw_cols[cat_act_idx];
            let v = row[raw_j];
            let val = self.cat_maps[c_pos]
                .get(&canonical_bits(v))
                .copied()
                .unwrap_or(0.0);
            out_slice[c_pos] = val;
        }
    }
}

#[derive(Clone, Debug)]
struct ElasticNetModel {
    weights: Vec<f64>,
    intercept: f64,
}

impl ElasticNetModel {
    fn solve_cholesky(a: &[f64], b: &[f64], d: usize) -> Option<Vec<f64>> {
        let mut l = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut sum = a[i * d + j];
                for k in 0..j {
                    sum -= l[i * d + k] * l[j * d + k];
                }
                if i == j {
                    if sum <= 1e-13 || !sum.is_finite() {
                        return None;
                    }
                    l[i * d + j] = sum.sqrt();
                } else {
                    l[i * d + j] = sum / l[j * d + j];
                }
            }
        }
        let mut y = vec![0.0; d];
        for i in 0..d {
            let mut sum = b[i];
            for k in 0..i {
                sum -= l[i * d + k] * y[k];
            }
            y[i] = sum / l[i * d + i];
        }
        let mut x = vec![0.0; d];
        for i in (0..d).rev() {
            let mut sum = y[i];
            for k in (i + 1)..d {
                sum -= l[k * d + i] * x[k];
            }
            x[i] = sum / l[i * d + i];
        }
        Some(x)
    }

    fn fit_with_grid(
        x_mat: &[f64],
        y: &[f64],
        d: usize,
        train_idx: &[usize],
        val_idx: &[usize],
    ) -> Self {
        let y_mean = if train_idx.is_empty() {
            0.0
        } else {
            let mut s = 0.0;
            for &i in train_idx {
                s += y[i];
            }
            s / (train_idx.len() as f64)
        };

        if d == 0 || train_idx.len() <= 1 {
            return Self {
                weights: vec![0.0; d],
                intercept: y_mean,
            };
        }

        let n_tr = train_idx.len() as f64;
        let mut best_w = vec![0.0; d];
        let mut best_mse = f64::INFINITY;

        if d <= 240 {
            let mut xtx = vec![0.0; d * d];
            let mut xty = vec![0.0; d];
            for &i in train_idx {
                let row = &x_mat[i * d..(i + 1) * d];
                let yc = y[i] - y_mean;
                for j in 0..d {
                    let rj = row[j];
                    xty[j] += rj * yc;
                    let row_out = &mut xtx[j * d..j * d + j + 1];
                    for k in 0..=j {
                        row_out[k] += rj * row[k];
                    }
                }
            }
            for j in 0..d {
                for k in (j + 1)..d {
                    xtx[j * d + k] = xtx[k * d + j];
                }
            }

            for &lam in &[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0] {
                let reg = lam * (n_tr / 100.0).max(1.0);
                let mut a = xtx.clone();
                for j in 0..d {
                    a[j * d + j] += reg;
                }
                if let Some(w) = Self::solve_cholesky(&a, &xty, d) {
                    if val_idx.is_empty() {
                        best_w = w;
                        break;
                    }
                    let mut mse = 0.0;
                    for &vi in val_idx {
                        let row = &x_mat[vi * d..(vi + 1) * d];
                        let mut pred = y_mean;
                        for j in 0..d {
                            pred += row[j] * w[j];
                        }
                        let diff = pred - y[vi];
                        mse += diff * diff;
                    }
                    if mse < best_mse {
                        best_mse = mse;
                        best_w = w;
                    }
                }
            }
        }

        // Also evaluate ElasticNet proximal coordinate descent (L1 + L2) for sparse signal recovery
        let mut col_sq = vec![0.0; d];
        for &i in train_idx {
            let row = &x_mat[i * d..(i + 1) * d];
            for j in 0..d {
                col_sq[j] += row[j] * row[j];
            }
        }

        let enet_configs = if d <= 240 {
            &[(0.5, 0.02), (2.0, 0.08)][..]
        } else {
            &[(1.0, 0.0), (10.0, 0.0), (100.0, 0.0), (2.0, 0.05)][..]
        };

        for &(l2_base, l1_base) in enet_configs {
            let l2_reg = l2_base * (n_tr / 100.0).max(1.0);
            let l1_reg = l1_base * (n_tr.sqrt()).max(1.0);
            let mut w = vec![0.0; d];
            let mut resid: Vec<f64> = train_idx.iter().map(|&i| y[i] - y_mean).collect();

            for _iter in 0..12 {
                for j in 0..d {
                    let denom = col_sq[j] + l2_reg;
                    if denom <= 1e-12 {
                        continue;
                    }
                    let mut dot = 0.0;
                    for (idx_pos, &i) in train_idx.iter().enumerate() {
                        dot += x_mat[i * d + j] * resid[idx_pos];
                    }
                    let rho = dot + col_sq[j] * w[j];
                    let new_wj = if rho > l1_reg {
                        (rho - l1_reg) / denom
                    } else if rho < -l1_reg {
                        (rho + l1_reg) / denom
                    } else {
                        0.0
                    };
                    let delta = new_wj - w[j];
                    if delta.abs() > 1e-9 {
                        for (idx_pos, &i) in train_idx.iter().enumerate() {
                            resid[idx_pos] -= delta * x_mat[i * d + j];
                        }
                        w[j] = new_wj;
                    }
                }
            }
            if val_idx.is_empty() {
                if best_mse.is_infinite() {
                    best_w = w;
                }
                break;
            }
            let mut mse = 0.0;
            for &vi in val_idx {
                let row = &x_mat[vi * d..(vi + 1) * d];
                let mut pred = y_mean;
                for j in 0..d {
                    pred += row[j] * w[j];
                }
                let diff = pred - y[vi];
                mse += diff * diff;
            }
            if mse < best_mse {
                best_mse = mse;
                best_w = w;
            }
        }

        Self {
            weights: best_w,
            intercept: y_mean,
        }
    }

    #[inline]
    fn predict_row(&self, x_row: &[f64]) -> f64 {
        let mut sum = self.intercept;
        for (w, &x) in self.weights.iter().zip(x_row.iter()) {
            sum += (*w) * x;
        }
        sum
    }
}

#[derive(Clone, Debug)]
struct ResMLPModel {
    in_dim: usize,
    hidden_dim: usize,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    w_skip: Vec<f64>,
    b2: f64,
}

impl ResMLPModel {
    fn fit(
        x_mat: &[f64],
        y: &[f64],
        in_dim: usize,
        train_idx: &[usize],
        val_idx: &[usize],
        rng: &mut XorShift64,
    ) -> Self {
        let hidden_dim = if in_dim <= 20 { 28 } else { 36 };
        let scale1 = (2.0 / (in_dim.max(1) as f64)).sqrt();
        let scale2 = (1.0 / (hidden_dim as f64)).sqrt();

        let mut w1 = Vec::with_capacity(hidden_dim * in_dim);
        for _ in 0..(hidden_dim * in_dim) {
            w1.push((rng.next_f64() * 2.0 - 1.0) * scale1);
        }
        let mut b1 = vec![0.0; hidden_dim];
        let mut w2 = Vec::with_capacity(hidden_dim);
        for _ in 0..hidden_dim {
            w2.push((rng.next_f64() * 2.0 - 1.0) * scale2);
        }
        let mut w_skip = vec![0.0; in_dim];
        let mut b2 = 0.0;

        if train_idx.is_empty() || in_dim == 0 {
            return Self {
                in_dim,
                hidden_dim,
                w1,
                b1,
                w2,
                w_skip,
                b2,
            };
        }

        let mut m_w1 = vec![0.0; hidden_dim * in_dim];
        let mut v_w1 = vec![0.0; hidden_dim * in_dim];
        let mut m_b1 = vec![0.0; hidden_dim];
        let mut v_b1 = vec![0.0; hidden_dim];
        let mut m_w2 = vec![0.0; hidden_dim];
        let mut v_w2 = vec![0.0; hidden_dim];
        let mut m_skip = vec![0.0; in_dim];
        let mut v_skip = vec![0.0; in_dim];
        let mut m_b2 = 0.0;
        let mut v_b2 = 0.0;

        let mut best_w1 = w1.clone();
        let mut best_b1 = b1.clone();
        let mut best_w2 = w2.clone();
        let mut best_skip = w_skip.clone();
        let mut best_b2 = b2;
        let mut best_val_loss = f64::INFINITY;

        let n_tr = train_idx.len();
        let batch_size = 64.min(n_tr).max(1);
        let max_epochs = if n_tr * in_dim > 1_200_000 {
            8
        } else if n_tr > 10_000 {
            16
        } else {
            36
        };

        let mut order = train_idx.to_vec();
        let mut step_t = 0i32;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let base_lr = 0.005;
        let weight_decay = 2e-4;

        let mut g_w1 = vec![0.0; hidden_dim * in_dim];
        let mut g_b1 = vec![0.0; hidden_dim];
        let mut g_w2 = vec![0.0; hidden_dim];
        let mut g_skip = vec![0.0; in_dim];
        let mut hidden_pre = vec![0.0; hidden_dim];
        let mut hidden_act = vec![0.0; hidden_dim];

        for epoch in 0..max_epochs {
            rng.shuffle(&mut order);
            let progress = (epoch as f64) / (max_epochs.max(1) as f64);
            let lr = base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()).max(0.1);

            for chunk in order.chunks(batch_size) {
                step_t += 1;
                let inv_bs = 1.0 / (chunk.len() as f64);
                g_w1.fill(0.0);
                g_b1.fill(0.0);
                g_w2.fill(0.0);
                g_skip.fill(0.0);
                let mut g_b2 = 0.0;

                for &row_i in chunk {
                    let x_row = &x_mat[row_i * in_dim..(row_i + 1) * in_dim];
                    let mut pred = b2;
                    for j in 0..in_dim {
                        pred += w_skip[j] * x_row[j];
                    }
                    for h in 0..hidden_dim {
                        let w_row = &w1[h * in_dim..(h + 1) * in_dim];
                        let mut z = b1[h];
                        for j in 0..in_dim {
                            z += w_row[j] * x_row[j];
                        }
                        z = z.clamp(-20.0, 20.0);
                        hidden_pre[h] = z;
                        let sig = 1.0 / (1.0 + (-z).exp());
                        let act = z * sig;
                        hidden_act[h] = act;
                        pred += w2[h] * act;
                    }

                    let err = pred - y[row_i];
                    let d_out = err.clamp(-4.5, 4.5) * inv_bs;
                    g_b2 += d_out;
                    for j in 0..in_dim {
                        g_skip[j] += d_out * x_row[j];
                    }

                    for h in 0..hidden_dim {
                        g_w2[h] += d_out * hidden_act[h];
                        let z = hidden_pre[h];
                        let sig = 1.0 / (1.0 + (-z).exp());
                        let d_silu = sig * (1.0 + z * (1.0 - sig));
                        let dh = d_out * w2[h] * d_silu;
                        g_b1[h] += dh;
                        let gw_row = &mut g_w1[h * in_dim..(h + 1) * in_dim];
                        for j in 0..in_dim {
                            gw_row[j] += dh * x_row[j];
                        }
                    }
                }

                let bias_corr1 = 1.0 - beta1.powi(step_t);
                let bias_corr2 = 1.0 - beta2.powi(step_t);
                let step_size = lr / bias_corr1;

                for idx in 0..w1.len() {
                    let grad = g_w1[idx] + weight_decay * w1[idx];
                    m_w1[idx] = beta1 * m_w1[idx] + (1.0 - beta1) * grad;
                    v_w1[idx] = beta2 * v_w1[idx] + (1.0 - beta2) * grad * grad;
                    let denom = (v_w1[idx] / bias_corr2).sqrt() + 1e-8;
                    w1[idx] -= step_size * m_w1[idx] / denom;
                }
                for j in 0..in_dim {
                    let grad_s = g_skip[j] + weight_decay * w_skip[j];
                    m_skip[j] = beta1 * m_skip[j] + (1.0 - beta1) * grad_s;
                    v_skip[j] = beta2 * v_skip[j] + (1.0 - beta2) * grad_s * grad_s;
                    w_skip[j] -= step_size * m_skip[j] / ((v_skip[j] / bias_corr2).sqrt() + 1e-8);
                }
                for h in 0..hidden_dim {
                    m_b1[h] = beta1 * m_b1[h] + (1.0 - beta1) * g_b1[h];
                    v_b1[h] = beta2 * v_b1[h] + (1.0 - beta2) * g_b1[h] * g_b1[h];
                    b1[h] -= step_size * m_b1[h] / ((v_b1[h] / bias_corr2).sqrt() + 1e-8);

                    let grad2 = g_w2[h] + weight_decay * w2[h];
                    m_w2[h] = beta1 * m_w2[h] + (1.0 - beta1) * grad2;
                    v_w2[h] = beta2 * v_w2[h] + (1.0 - beta2) * grad2 * grad2;
                    w2[h] -= step_size * m_w2[h] / ((v_w2[h] / bias_corr2).sqrt() + 1e-8);
                }
                m_b2 = beta1 * m_b2 + (1.0 - beta1) * g_b2;
                v_b2 = beta2 * v_b2 + (1.0 - beta2) * g_b2 * g_b2;
                b2 -= step_size * m_b2 / ((v_b2 / bias_corr2).sqrt() + 1e-8);
            }

            let eval_indices = if val_idx.is_empty() {
                train_idx
            } else {
                val_idx
            };
            let mut val_loss = 0.0;
            for &vi in eval_indices {
                let x_row = &x_mat[vi * in_dim..(vi + 1) * in_dim];
                let mut pred = b2;
                for j in 0..in_dim {
                    pred += w_skip[j] * x_row[j];
                }
                for h in 0..hidden_dim {
                    let w_row = &w1[h * in_dim..(h + 1) * in_dim];
                    let mut z = b1[h];
                    for j in 0..in_dim {
                        z += w_row[j] * x_row[j];
                    }
                    z = z.clamp(-20.0, 20.0);
                    let sig = 1.0 / (1.0 + (-z).exp());
                    pred += w2[h] * z * sig;
                }
                let diff = pred - y[vi];
                val_loss += diff * diff;
            }
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_w1.copy_from_slice(&w1);
                best_b1.copy_from_slice(&b1);
                best_w2.copy_from_slice(&w2);
                best_skip.copy_from_slice(&w_skip);
                best_b2 = b2;
            }
        }

        Self {
            in_dim,
            hidden_dim,
            w1: best_w1,
            b1: best_b1,
            w2: best_w2,
            w_skip: best_skip,
            b2: best_b2,
        }
    }

    #[inline]
    fn predict_row(&self, x_row: &[f64]) -> f64 {
        if self.in_dim == 0 {
            return self.b2;
        }
        let mut pred = self.b2;
        for j in 0..self.in_dim {
            pred += self.w_skip[j] * x_row[j];
        }
        for h in 0..self.hidden_dim {
            let w_row = &self.w1[h * self.in_dim..(h + 1) * self.in_dim];
            let mut z = self.b1[h];
            for j in 0..self.in_dim {
                z += w_row[j] * x_row[j];
            }
            z = z.clamp(-20.0, 20.0);
            let sig = 1.0 / (1.0 + (-z).exp());
            pred += self.w2[h] * z * sig;
        }
        pred
    }
}

#[derive(Clone, Debug)]
struct KNNModel {
    x_ref: Vec<f64>,
    y_ref: Vec<f64>,
    feat_weights: Vec<f64>,
    dim: usize,
    n_ref: usize,
    bandwidth: f64,
    use_l1: bool,
    fallback_mean: f64,
}

impl KNNModel {
    fn fit(
        x_mat: &[f64],
        y: &[f64],
        dim: usize,
        train_idx: &[usize],
        val_idx: &[usize],
        rng: &mut XorShift64,
    ) -> Self {
        let mut fallback_mean = 0.0;
        if !train_idx.is_empty() {
            for &i in train_idx {
                fallback_mean += y[i];
            }
            fallback_mean /= train_idx.len() as f64;
        }

        if dim == 0 || train_idx.is_empty() {
            return Self {
                x_ref: Vec::new(),
                y_ref: Vec::new(),
                feat_weights: Vec::new(),
                dim: 0,
                n_ref: 0,
                bandwidth: 1.0,
                use_l1: false,
                fallback_mean,
            };
        }

        let max_ref = 1600.min(train_idx.len());
        let mut chosen = train_idx.to_vec();
        rng.shuffle(&mut chosen);
        chosen.truncate(max_ref);

        let mut feat_weights = vec![0.0; dim];
        for j in 0..dim {
            let mut sxy = 0.0;
            let mut sx2 = 0.0;
            for &i in &chosen {
                let xv = x_mat[i * dim + j];
                let yv = y[i] - fallback_mean;
                sxy += xv * yv;
                sx2 += xv * xv;
            }
            let corr = if sx2 > 1e-12 {
                (sxy / sx2.sqrt()).abs()
            } else {
                0.0
            };
            feat_weights[j] = corr * corr + 1e-3 / (dim as f64);
        }
        let w_sum: f64 = feat_weights.iter().sum();
        if w_sum > 1e-12 {
            for w in &mut feat_weights {
                *w /= w_sum;
            }
        }

        let mut x_ref = Vec::with_capacity(max_ref * dim);
        let mut y_ref = Vec::with_capacity(max_ref);
        for &i in &chosen {
            x_ref.extend_from_slice(&x_mat[i * dim..(i + 1) * dim]);
            y_ref.push(y[i]);
        }

        let mut candidate_model = Self {
            x_ref,
            y_ref,
            feat_weights,
            dim,
            n_ref: max_ref,
            bandwidth: 0.5,
            use_l1: false,
            fallback_mean,
        };

        if !val_idx.is_empty() {
            let eval_sub = val_idx.len().min(320);
            let mut best_bw = 0.5;
            let mut best_l1 = false;
            let mut best_mse = f64::INFINITY;
            for &l1_flag in &[false, true] {
                candidate_model.use_l1 = l1_flag;
                for &bw in &[0.08, 0.25, 0.6, 1.5] {
                    candidate_model.bandwidth = bw;
                    let mut mse = 0.0;
                    for &vi in &val_idx[..eval_sub] {
                        let p = candidate_model.predict_row(&x_mat[vi * dim..(vi + 1) * dim]);
                        let d = p - y[vi];
                        mse += d * d;
                    }
                    if mse < best_mse {
                        best_mse = mse;
                        best_bw = bw;
                        best_l1 = l1_flag;
                    }
                }
            }
            candidate_model.bandwidth = best_bw;
            candidate_model.use_l1 = best_l1;
        }

        candidate_model
    }

    fn predict_row(&self, x_row: &[f64]) -> f64 {
        if self.n_ref == 0 || self.dim == 0 {
            return self.fallback_mean;
        }
        let k = 16.min(self.n_ref);
        let mut top_k = [(f64::INFINITY, 0.0); 16];

        for r in 0..self.n_ref {
            let ref_row = &self.x_ref[r * self.dim..(r + 1) * self.dim];
            let mut dist = 0.0;
            if self.use_l1 {
                for j in 0..self.dim {
                    dist += self.feat_weights[j] * (x_row[j] - ref_row[j]).abs();
                }
            } else {
                for j in 0..self.dim {
                    let d = x_row[j] - ref_row[j];
                    dist += self.feat_weights[j] * d * d;
                }
            }
            if dist < top_k[k - 1].0 {
                top_k[k - 1] = (dist, self.y_ref[r]);
                let mut pos = k - 1;
                while pos > 0 && top_k[pos].0 < top_k[pos - 1].0 {
                    top_k.swap(pos, pos - 1);
                    pos -= 1;
                }
            }
        }

        let min_d = top_k[0].0.max(0.0);
        let scale = (self.bandwidth * (min_d + 1e-4)).max(1e-7);
        let mut w_sum = 0.0;
        let mut y_sum = 0.0;
        for idx in 0..k {
            let (d, yv) = top_k[idx];
            if !d.is_finite() {
                continue;
            }
            let w = (-(d - min_d) / scale).exp();
            w_sum += w;
            y_sum += w * yv;
        }
        if w_sum > 1e-12 {
            y_sum / w_sum
        } else {
            self.fallback_mean
        }
    }
}

#[derive(Clone, Debug)]
struct HistogramBinner {
    bin_edges: Vec<Vec<f64>>,
    binned_data: Vec<u8>,
    n_rows: usize,
    n_features: usize,
}

impl HistogramBinner {
    fn build(tree_x: &[f64], n_rows: usize, n_features: usize) -> Self {
        let stride = if n_rows > 12000 {
            (n_rows / 4500).max(1)
        } else {
            1
        };
        let mut bin_edges = Vec::with_capacity(n_features);
        let mut buf = Vec::with_capacity((n_rows / stride) + 1);

        for j in 0..n_features {
            buf.clear();
            for i in (0..n_rows).step_by(stride) {
                let v = tree_x[i * n_features + j];
                if !v.is_nan() {
                    buf.push(v);
                }
            }
            buf.sort_by(|&a, &b| total_cmp_f64(a, b));
            let mut edges = Vec::new();
            if !buf.is_empty() {
                let m = buf.len();
                let max_bins = 127.min(m);
                for b in 1..max_bins {
                    let idx = (b * m) / max_bins;
                    let val = 0.5 * (buf[idx.saturating_sub(1)] + buf[idx]);
                    if edges.is_empty() || (val - *edges.last().unwrap()).abs() > 1e-12 {
                        edges.push(val);
                    }
                }
                if edges.is_empty() {
                    edges.push(buf[m / 2]);
                }
            }
            bin_edges.push(edges);
        }

        let mut binned_data = vec![255u8; n_features * n_rows];
        for j in 0..n_features {
            let edges = &bin_edges[j];
            let col_slice = &mut binned_data[j * n_rows..(j + 1) * n_rows];
            if edges.is_empty() {
                continue;
            }
            for i in 0..n_rows {
                let v = tree_x[i * n_features + j];
                if !v.is_nan() {
                    let bin = match edges.binary_search_by(|&e| total_cmp_f64(e, v)) {
                        Ok(pos) => pos,
                        Err(pos) => pos,
                    };
                    col_slice[i] = bin as u8;
                }
            }
        }

        Self {
            bin_edges,
            binned_data,
            n_rows,
            n_features,
        }
    }
}

#[derive(Clone, Debug)]
struct ObliviousTree {
    depth: usize,
    split_feats: Vec<usize>,
    split_thresh: Vec<f64>,
    split_nan_left: Vec<bool>,
    leaf_values: Vec<f64>, // length 1 << depth
}

#[derive(Clone, Debug)]
struct ObliviousGBDTModel {
    base_score: f64,
    learning_rate: f64,
    trees: Vec<ObliviousTree>,
}

impl ObliviousGBDTModel {
    #[allow(clippy::too_many_arguments)]
    fn fit(
        binner: &HistogramBinner,
        y: &[f64],
        train_idx: &[usize],
        val_idx: &[usize],
        depth: usize,
        min_child_weight: usize,
        l2_reg: f64,
        colsample_rate: f64,
        learning_rate: f64,
        max_trees: usize,
        rng: &mut XorShift64,
    ) -> Self {
        let mut base_score = 0.0;
        if !train_idx.is_empty() {
            for &i in train_idx {
                base_score += y[i];
            }
            base_score /= train_idx.len() as f64;
        }

        if train_idx.is_empty() || binner.n_features == 0 {
            return Self {
                base_score,
                learning_rate,
                trees: Vec::new(),
            };
        }

        let mut preds = vec![base_score; binner.n_rows];
        let mut grads = vec![0.0; binner.n_rows];
        let mut trees = Vec::with_capacity(max_trees);

        let n_feat = binner.n_features;
        let n_col_sample = ((n_feat as f64 * colsample_rate).ceil() as usize)
            .max(1)
            .min(n_feat);
        let mut feat_candidates: Vec<usize> = (0..n_feat).collect();

        let mut best_val_loss = f64::INFINITY;
        let mut best_tree_count = 0usize;
        let mut no_improve = 0usize;

        let n_leaves = 1usize << depth;
        let mut row_leaf = vec![0usize; train_idx.len()];
        let mut leaf_g = vec![0.0; n_leaves];
        let mut leaf_h = vec![0.0; n_leaves];

        // Flattened buffers for oblivious level-wise histograms: [leaf_idx * 129 + bin]
        let mut hist_g = vec![0.0; n_leaves * 129];
        let mut hist_cnt = vec![0usize; n_leaves * 129];
        let mut nan_g = vec![0.0; n_leaves];
        let mut nan_cnt = vec![0usize; n_leaves];

        for t_idx in 0..max_trees {
            for &i in train_idx {
                grads[i] = preds[i] - y[i];
            }

            rng.shuffle(&mut feat_candidates);
            let active_feats = &feat_candidates[..n_col_sample];

            row_leaf.fill(0);
            let mut split_feats = Vec::with_capacity(depth);
            let mut split_bins = Vec::with_capacity(depth);
            let mut split_thresh = Vec::with_capacity(depth);
            let mut split_nan_left = Vec::with_capacity(depth);

            for d in 0..depth {
                let curr_leaves = 1usize << d;
                leaf_g[..curr_leaves].fill(0.0);
                leaf_h[..curr_leaves].fill(0.0);
                for (pos, &r) in train_idx.iter().enumerate() {
                    let lf = row_leaf[pos];
                    leaf_g[lf] += grads[r];
                    leaf_h[lf] += 1.0;
                }

                let mut parent_score = 0.0;
                for lf in 0..curr_leaves {
                    if leaf_h[lf] > 0.0 {
                        parent_score += (leaf_g[lf] * leaf_g[lf]) / (leaf_h[lf] + l2_reg);
                    }
                }

                let mut best_gain = 1e-7;
                let mut best_f = 0usize;
                let mut best_b = 0usize;
                let mut best_nan_l = true;
                let mut found = false;

                for &f_idx in active_feats {
                    let edges = &binner.bin_edges[f_idx];
                    if edges.is_empty() {
                        continue;
                    }
                    let n_bins = edges.len() + 1;
                    for lf in 0..curr_leaves {
                        let base = lf * 129;
                        hist_g[base..base + n_bins].fill(0.0);
                        hist_cnt[base..base + n_bins].fill(0);
                        nan_g[lf] = 0.0;
                        nan_cnt[lf] = 0;
                    }

                    let col_bins =
                        &binner.binned_data[f_idx * binner.n_rows..(f_idx + 1) * binner.n_rows];
                    for (pos, &r) in train_idx.iter().enumerate() {
                        let lf = row_leaf[pos];
                        let b = col_bins[r];
                        if b == 255 {
                            nan_g[lf] += grads[r];
                            nan_cnt[lf] += 1;
                        } else {
                            let idx = lf * 129 + (b as usize);
                            hist_g[idx] += grads[r];
                            hist_cnt[idx] += 1;
                        }
                    }

                    for nan_dir in [true, false] {
                        let mut any_nan = false;
                        for lf in 0..curr_leaves {
                            if nan_cnt[lf] > 0 {
                                any_nan = true;
                                break;
                            }
                        }
                        if !any_nan && !nan_dir {
                            break;
                        }

                        let mut left_g = [0.0f64; 64];
                        let mut left_cnt = [0usize; 64];
                        for lf in 0..curr_leaves {
                            if nan_dir {
                                left_g[lf] = nan_g[lf];
                                left_cnt[lf] = nan_cnt[lf];
                            }
                        }

                        for b in 0..(n_bins - 1) {
                            let mut total_score = 0.0;
                            let mut valid_children = 0usize;
                            for lf in 0..curr_leaves {
                                let idx = lf * 129 + b;
                                left_g[lf] += hist_g[idx];
                                left_cnt[lf] += hist_cnt[idx];

                                let l_cnt = left_cnt[lf];
                                let r_cnt = (leaf_h[lf] as usize).saturating_sub(l_cnt);
                                if l_cnt >= min_child_weight && r_cnt >= min_child_weight {
                                    valid_children += 1;
                                }
                                let lg = left_g[lf];
                                let rg = leaf_g[lf] - lg;
                                let lh = l_cnt as f64;
                                let rh = r_cnt as f64;
                                total_score +=
                                    (lg * lg) / (lh + l2_reg) + (rg * rg) / (rh + l2_reg);
                            }
                            if valid_children > 0 {
                                let gain = total_score - parent_score;
                                if gain > best_gain {
                                    best_gain = gain;
                                    best_f = f_idx;
                                    best_b = b;
                                    best_nan_l = nan_dir;
                                    found = true;
                                }
                            }
                        }
                    }
                }

                if !found {
                    break;
                }

                let thresh = binner.bin_edges[best_f][best_b];
                let col_bins =
                    &binner.binned_data[best_f * binner.n_rows..(best_f + 1) * binner.n_rows];
                for (pos, &r) in train_idx.iter().enumerate() {
                    let b = col_bins[r];
                    let go_left = if b == 255 {
                        best_nan_l
                    } else {
                        (b as usize) <= best_b
                    };
                    row_leaf[pos] = (row_leaf[pos] << 1) | if go_left { 0 } else { 1 };
                }

                split_feats.push(best_f);
                split_bins.push(best_b);
                split_thresh.push(thresh);
                split_nan_left.push(best_nan_l);
            }

            let actual_depth = split_feats.len();
            if actual_depth == 0 {
                break;
            }
            let final_leaves = 1usize << actual_depth;
            let mut final_g = vec![0.0; final_leaves];
            let mut final_h = vec![0.0; final_leaves];
            for (pos, &r) in train_idx.iter().enumerate() {
                let lf = row_leaf[pos];
                final_g[lf] += grads[r];
                final_h[lf] += 1.0;
            }
            let mut leaf_values = vec![0.0; final_leaves];
            for lf in 0..final_leaves {
                leaf_values[lf] = -final_g[lf] / (final_h[lf] + l2_reg);
            }

            for (pos, &r) in train_idx.iter().enumerate() {
                preds[r] += learning_rate * leaf_values[row_leaf[pos]];
            }

            for &vi in val_idx {
                let mut lf = 0usize;
                for d in 0..actual_depth {
                    let f_idx = split_feats[d];
                    let b = binner.binned_data[f_idx * binner.n_rows + vi];
                    let go_left = if b == 255 {
                        split_nan_left[d]
                    } else {
                        (b as usize) <= split_bins[d]
                    };
                    lf = (lf << 1) | if go_left { 0 } else { 1 };
                }
                preds[vi] += learning_rate * leaf_values[lf];
            }

            trees.push(ObliviousTree {
                depth: actual_depth,
                split_feats,
                split_thresh,
                split_nan_left,
                leaf_values,
            });

            if !val_idx.is_empty() {
                let mut v_mse = 0.0;
                for &vi in val_idx {
                    let diff = preds[vi] - y[vi];
                    v_mse += diff * diff;
                }
                if v_mse < best_val_loss {
                    best_val_loss = v_mse;
                    best_tree_count = trees.len();
                    no_improve = 0;
                } else {
                    no_improve += 1;
                    if no_improve >= 20 && t_idx >= 30 {
                        break;
                    }
                }
            } else {
                best_tree_count = trees.len();
            }
        }

        if best_tree_count > 0 && best_tree_count < trees.len() {
            trees.truncate(best_tree_count);
        }

        Self {
            base_score,
            learning_rate,
            trees,
        }
    }

    #[inline]
    fn predict_row(&self, tree_x_row: &[f64]) -> f64 {
        let mut pred = self.base_score;
        for tree in &self.trees {
            let mut lf = 0usize;
            for d in 0..tree.depth {
                let v = tree_x_row[tree.split_feats[d]];
                let go_left = if v.is_nan() {
                    tree.split_nan_left[d]
                } else {
                    v <= tree.split_thresh[d]
                };
                lf = (lf << 1) | if go_left { 0 } else { 1 };
            }
            pred += self.learning_rate * tree.leaf_values[lf];
        }
        pred
    }
}

#[derive(Clone, Debug)]
struct TreeNode {
    is_leaf: bool,
    leaf_value: f64,
    feature_idx: usize,
    threshold: f64,
    nan_go_left: bool,
    left: usize,
    right: usize,
}

#[derive(Clone, Debug)]
struct GBDTModel {
    base_score: f64,
    learning_rate: f64,
    trees: Vec<Vec<TreeNode>>,
}

impl GBDTModel {
    #[allow(clippy::too_many_arguments)]
    fn fit(
        binner: &HistogramBinner,
        y: &[f64],
        train_idx: &[usize],
        val_idx: &[usize],
        max_depth: usize,
        min_child_weight: usize,
        l2_reg: f64,
        colsample_rate: f64,
        learning_rate: f64,
        max_trees: usize,
        use_pseudo_huber: bool,
        huber_delta: f64,
        rng: &mut XorShift64,
    ) -> Self {
        let mut base_score = 0.0;
        if !train_idx.is_empty() {
            for &i in train_idx {
                base_score += y[i];
            }
            base_score /= train_idx.len() as f64;
        }

        if train_idx.is_empty() || binner.n_features == 0 {
            return Self {
                base_score,
                learning_rate,
                trees: Vec::new(),
            };
        }

        let mut train_preds = vec![base_score; binner.n_rows];
        let mut grads = vec![0.0; binner.n_rows];
        let mut hess = vec![1.0; binner.n_rows];

        let mut trees = Vec::with_capacity(max_trees);
        let mut best_val_loss = f64::INFINITY;
        let mut best_tree_count = 0usize;
        let mut no_improve = 0usize;

        let n_feat = binner.n_features;
        let n_col_sample = ((n_feat as f64 * colsample_rate).ceil() as usize)
            .max(1)
            .min(n_feat);
        let mut feat_candidates: Vec<usize> = (0..n_feat).collect();
        let d2 = huber_delta * huber_delta;

        for t_idx in 0..max_trees {
            for &i in train_idx {
                let err = train_preds[i] - y[i];
                if use_pseudo_huber {
                    // Smooth C^2 Pseudo-Huber loss: L(r) = delta^2 * (sqrt(1 + (r/delta)^2) - 1)
                    let s = (1.0 + (err * err) / d2).sqrt();
                    grads[i] = err / s;
                    hess[i] = (1.0 / (s * s * s)).max(1e-3);
                } else {
                    grads[i] = err;
                    hess[i] = 1.0;
                }
            }

            rng.shuffle(&mut feat_candidates);
            let active_feats = &feat_candidates[..n_col_sample];

            let mut root_rows = Vec::with_capacity(train_idx.len());
            for &i in train_idx {
                if train_idx.len() < 60 || rng.next_f64() < 0.85 {
                    root_rows.push(i);
                }
            }
            if root_rows.is_empty() {
                root_rows.extend_from_slice(train_idx);
            }

            let mut tree_nodes = Vec::with_capacity((1 << (max_depth + 1)) - 1);
            Self::build_tree_recursive(
                binner,
                &grads,
                &hess,
                &root_rows,
                active_feats,
                0,
                max_depth,
                min_child_weight,
                l2_reg,
                &mut tree_nodes,
            );

            for &i in train_idx {
                train_preds[i] += learning_rate * Self::eval_tree_binned(&tree_nodes, binner, i);
            }
            for &vi in val_idx {
                train_preds[vi] += learning_rate * Self::eval_tree_binned(&tree_nodes, binner, vi);
            }

            trees.push(tree_nodes);

            if !val_idx.is_empty() {
                let mut v_mse = 0.0;
                for &vi in val_idx {
                    let diff = train_preds[vi] - y[vi];
                    v_mse += diff * diff;
                }
                if v_mse < best_val_loss {
                    best_val_loss = v_mse;
                    best_tree_count = t_idx + 1;
                    no_improve = 0;
                } else {
                    no_improve += 1;
                    if no_improve >= 18 && t_idx >= 25 {
                        break;
                    }
                }
            } else {
                best_tree_count = t_idx + 1;
            }
        }

        if best_tree_count > 0 && best_tree_count < trees.len() {
            trees.truncate(best_tree_count);
        }

        Self {
            base_score,
            learning_rate,
            trees,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tree_recursive(
        binner: &HistogramBinner,
        grads: &[f64],
        hess: &[f64],
        node_rows: &[usize],
        active_feats: &[usize],
        depth: usize,
        max_depth: usize,
        min_child_weight: usize,
        l2_reg: f64,
        nodes: &mut Vec<TreeNode>,
    ) -> usize {
        let node_idx = nodes.len();
        let mut sum_g = 0.0;
        let mut sum_h = 0.0;
        for &r in node_rows {
            sum_g += grads[r];
            sum_h += hess[r];
        }
        let leaf_val = -sum_g / (sum_h + l2_reg);

        nodes.push(TreeNode {
            is_leaf: true,
            leaf_value: leaf_val,
            feature_idx: 0,
            threshold: 0.0,
            nan_go_left: true,
            left: 0,
            right: 0,
        });

        if depth >= max_depth || node_rows.len() < 2 * min_child_weight {
            return node_idx;
        }

        let parent_score = (sum_g * sum_g) / (sum_h + l2_reg);
        let mut best_gain = 1e-7;
        let mut best_feat = 0usize;
        let mut best_bin = 0usize;
        let mut best_nan_left = true;
        let mut found_split = false;

        let mut bin_g = [0.0f64; 129];
        let mut bin_h = [0.0f64; 129];
        let mut bin_cnt = [0usize; 129];

        for &f_idx in active_feats {
            let edges = &binner.bin_edges[f_idx];
            if edges.is_empty() {
                continue;
            }
            let n_bins = edges.len() + 1;
            for b in 0..n_bins {
                bin_g[b] = 0.0;
                bin_h[b] = 0.0;
                bin_cnt[b] = 0;
            }
            let mut nan_g = 0.0;
            let mut nan_h = 0.0;
            let mut nan_cnt = 0usize;

            let col_bins = &binner.binned_data[f_idx * binner.n_rows..(f_idx + 1) * binner.n_rows];
            for &r in node_rows {
                let b = col_bins[r];
                if b == 255 {
                    nan_g += grads[r];
                    nan_h += hess[r];
                    nan_cnt += 1;
                } else {
                    let bi = b as usize;
                    bin_g[bi] += grads[r];
                    bin_h[bi] += hess[r];
                    bin_cnt[bi] += 1;
                }
            }

            for nan_dir in [true, false] {
                if nan_cnt == 0 && !nan_dir {
                    break;
                }
                let mut left_g = if nan_dir { nan_g } else { 0.0 };
                let mut left_h = if nan_dir { nan_h } else { 0.0 };
                let mut left_cnt = if nan_dir { nan_cnt } else { 0 };

                for b in 0..(n_bins - 1) {
                    left_g += bin_g[b];
                    left_h += bin_h[b];
                    left_cnt += bin_cnt[b];

                    let right_cnt = node_rows.len() - left_cnt;
                    if left_cnt < min_child_weight || right_cnt < min_child_weight {
                        continue;
                    }
                    let right_g = sum_g - left_g;
                    let right_h = sum_h - left_h;
                    let gain = (left_g * left_g) / (left_h + l2_reg)
                        + (right_g * right_g) / (right_h + l2_reg)
                        - parent_score;
                    if gain > best_gain {
                        best_gain = gain;
                        best_feat = f_idx;
                        best_bin = b;
                        best_nan_left = nan_dir;
                        found_split = true;
                    }
                }
            }
        }

        if !found_split {
            return node_idx;
        }

        let threshold = binner.bin_edges[best_feat][best_bin];
        let mut left_rows = Vec::with_capacity(node_rows.len() / 2);
        let mut right_rows = Vec::with_capacity(node_rows.len() / 2);
        let col_bins =
            &binner.binned_data[best_feat * binner.n_rows..(best_feat + 1) * binner.n_rows];

        for &r in node_rows {
            let b = col_bins[r];
            let go_left = if b == 255 {
                best_nan_left
            } else {
                (b as usize) <= best_bin
            };
            if go_left {
                left_rows.push(r);
            } else {
                right_rows.push(r);
            }
        }

        if left_rows.is_empty() || right_rows.is_empty() {
            return node_idx;
        }

        let left_child = Self::build_tree_recursive(
            binner,
            grads,
            hess,
            &left_rows,
            active_feats,
            depth + 1,
            max_depth,
            min_child_weight,
            l2_reg,
            nodes,
        );
        let right_child = Self::build_tree_recursive(
            binner,
            grads,
            hess,
            &right_rows,
            active_feats,
            depth + 1,
            max_depth,
            min_child_weight,
            l2_reg,
            nodes,
        );

        nodes[node_idx].is_leaf = false;
        nodes[node_idx].feature_idx = best_feat;
        nodes[node_idx].threshold = threshold;
        nodes[node_idx].nan_go_left = best_nan_left;
        nodes[node_idx].left = left_child;
        nodes[node_idx].right = right_child;

        node_idx
    }

    #[inline]
    fn eval_tree_binned(nodes: &[TreeNode], binner: &HistogramBinner, row_i: usize) -> f64 {
        let mut curr = 0usize;
        while !nodes[curr].is_leaf {
            let n = &nodes[curr];
            let b = binner.binned_data[n.feature_idx * binner.n_rows + row_i];
            let go_left = if b == 255 {
                n.nan_go_left
            } else {
                let edges = &binner.bin_edges[n.feature_idx];
                let bi = b as usize;
                if bi == 0 {
                    edges[0] <= n.threshold
                } else {
                    edges[bi - 1] < n.threshold
                }
            };
            curr = if go_left { n.left } else { n.right };
        }
        nodes[curr].leaf_value
    }

    #[inline]
    fn predict_row(&self, tree_x_row: &[f64]) -> f64 {
        let mut pred = self.base_score;
        for tree in &self.trees {
            let mut curr = 0usize;
            while !tree[curr].is_leaf {
                let n = &tree[curr];
                let v = tree_x_row[n.feature_idx];
                let go_left = if v.is_nan() {
                    n.nan_go_left
                } else {
                    v <= n.threshold
                };
                curr = if go_left { n.left } else { n.right };
            }
            pred += self.learning_rate * tree[curr].leaf_value;
        }
        pred
    }
}

#[derive(Clone, Debug)]
struct FittedFold {
    target_encoder: FoldTargetEncoder,
    enet: ElasticNetModel,
    oblivious_gbdt: ObliviousGBDTModel,
    gbdt_shallow: GBDTModel,
    gbdt_deep_huber: GBDTModel,
    res_mlp: ResMLPModel,
    knn: KNNModel,
}

#[derive(Clone, Debug)]
struct FittedState {
    n_features: usize,
    y_mean: f64,
    y_min: f64,
    y_max: f64,
    y_span: f64,
    is_constant_target: bool,
    target_transform: TargetTransform,
    smear_factor: f64,
    preprocessor: Preprocessor,
    folds: Vec<FittedFold>,
    ensemble_weights: [f64; 6],
    meta_scale: f64,
    meta_shift: f64,
}

#[derive(Clone, Debug)]
pub struct Regressor {
    seed: u64,
    state: Option<FittedState>,
}

impl Regressor {
    fn optimize_simplex_weights(oof_preds: &[[f64; 6]], y_target: &[f64]) -> [f64; 6] {
        let n = y_target.len();
        if n == 0 {
            return [1.0 / 6.0; 6];
        }
        let mut weights = [0.16, 0.24, 0.22, 0.18, 0.12, 0.08];

        let inv_n = 1.0 / (n as f64);
        for iter in 0..300 {
            let mut grad = [0.0; 6];
            for i in 0..n {
                let p = &oof_preds[i];
                let mut ens = 0.0;
                for m in 0..6 {
                    ens += weights[m] * p[m];
                }
                let err = ens - y_target[i];
                for m in 0..6 {
                    grad[m] += 2.0 * inv_n * err * p[m];
                }
            }
            let lr = 0.15 / (1.0 + 0.015 * (iter as f64));
            for m in 0..6 {
                weights[m] -= lr * grad[m];
            }
            // Euclidean projection onto 6-simplex
            let mut u = weights;
            u.sort_by(|a, b| total_cmp_f64(*b, *a));
            let mut cssv = 0.0;
            let mut rho = 0usize;
            for j in 0..6 {
                cssv += u[j];
                let t = (cssv - 1.0) / ((j + 1) as f64);
                if u[j] - t > 0.0 {
                    rho = j;
                }
            }
            let mut sum_rho = 0.0;
            for u_val in u.iter().take(rho + 1) {
                sum_rho += *u_val;
            }
            let theta = (sum_rho - 1.0) / ((rho + 1) as f64);
            for w in &mut weights {
                *w = (*w - theta).max(0.0);
            }
        }
        weights
    }

    fn select_best_target_transform(
        raw_x: &[f64],
        n_rows: usize,
        n_features: usize,
        y_raw: &[f64],
        id_trans: TargetTransform,
        alt_trans: TargetTransform,
    ) -> (TargetTransform, f64) {
        if matches!(alt_trans, TargetTransform::Identity { .. }) || n_rows < 60 {
            return (id_trans, 1.0);
        }

        // Quick 80/20 stratified holdout check in original target units using fast Ridge + Shallow GBDT
        let mut sorted_idx: Vec<usize> = (0..n_rows).collect();
        sorted_idx.sort_by(|&a, &b| total_cmp_f64(y_raw[a], y_raw[b]));
        let mut tr_idx = Vec::with_capacity(n_rows);
        let mut va_idx = Vec::with_capacity(n_rows / 5 + 1);
        for (rank, &i) in sorted_idx.iter().enumerate() {
            if rank % 5 == 2 {
                va_idx.push(i);
            } else {
                tr_idx.push(i);
            }
        }

        let y_id: Vec<f64> = y_raw.iter().map(|&y| id_trans.forward(y)).collect();
        let y_alt: Vec<f64> = y_raw.iter().map(|&y| alt_trans.forward(y)).collect();

        let prep = Preprocessor::fit(raw_x, n_rows, n_features, &y_id);
        let base_tree_dim = prep.base_tree_feature_dim();
        let mut x_tree = vec![0.0; n_rows * base_tree_dim];
        for i in 0..n_rows {
            prep.transform_row_for_tree_base(
                &raw_x[i * n_features..(i + 1) * n_features],
                &mut x_tree[i * base_tree_dim..(i + 1) * base_tree_dim],
            );
        }
        let binner = HistogramBinner::build(&x_tree, n_rows, base_tree_dim);

        let mut rng1 = XorShift64::new(42);
        let m_id = ObliviousGBDTModel::fit(
            &binner, &y_id, &tr_idx, &va_idx, 5, 5, 3.0, 0.8, 0.08, 45, &mut rng1,
        );
        let mut mse_id = 0.0;
        for &vi in &va_idx {
            let p_z = m_id.predict_row(&x_tree[vi * base_tree_dim..(vi + 1) * base_tree_dim]);
            let p_raw = id_trans.inverse(p_z, 1.0);
            let d = p_raw - y_raw[vi];
            mse_id += d * d;
        }

        let mut rng2 = XorShift64::new(42);
        let m_alt = ObliviousGBDTModel::fit(
            &binner, &y_alt, &tr_idx, &va_idx, 5, 5, 3.0, 0.8, 0.08, 45, &mut rng2,
        );

        // Compute Duan smearing factor on training split for unbiased nonlinear inverse
        let mut smear_sum = 0.0;
        for &ti in &tr_idx {
            let p_z = m_alt.predict_row(&x_tree[ti * base_tree_dim..(ti + 1) * base_tree_dim]);
            let r_z = y_alt[ti] - p_z;
            match alt_trans {
                TargetTransform::ShiftedLog { t_std, .. }
                | TargetTransform::Asinh { t_std, .. } => {
                    smear_sum += (r_z * t_std).clamp(-5.0, 5.0).exp();
                }
                TargetTransform::Identity { .. } => smear_sum += 1.0,
            }
        }
        let smear = (smear_sum / (tr_idx.len().max(1) as f64)).clamp(0.8, 2.5);

        let mut mse_alt = 0.0;
        for &vi in &va_idx {
            let p_z = m_alt.predict_row(&x_tree[vi * base_tree_dim..(vi + 1) * base_tree_dim]);
            let p_raw = alt_trans.inverse(p_z, smear);
            let d = p_raw - y_raw[vi];
            mse_alt += d * d;
        }

        if mse_alt < 0.97 * mse_id {
            (alt_trans, smear)
        } else {
            (id_trans, 1.0)
        }
    }
}

impl RegressorApi for Regressor {
    fn new(seed: u64) -> Self {
        Self { seed, state: None }
    }

    fn fit(
        &mut self,
        train: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), String> {
        self.state = None;

        if n_rows == 0 || n_cols == 0 {
            return Err("Invalid training dimensions: n_rows and n_cols must be >= 1".to_string());
        }
        let expected_len = n_rows
            .checked_mul(n_cols)
            .ok_or_else(|| "Dimension multiplication overflow in fit".to_string())?;
        if train.len() != expected_len {
            return Err(format!(
                "Flattened train length {} does not match n_rows ({}) * n_cols ({})",
                train.len(),
                n_rows,
                n_cols
            ));
        }

        let n_features = n_cols - 1;
        let mut raw_x = Vec::with_capacity(n_rows.saturating_mul(n_features));
        let mut y_raw = Vec::with_capacity(n_rows);

        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut y_sum = 0.0;

        for i in 0..n_rows {
            let row_start = i * n_cols;
            for j in 0..n_features {
                let v = train[row_start + j];
                if v.is_infinite() {
                    return Err(format!("Infinite feature value at row {}, col {}", i, j));
                }
                raw_x.push(v);
            }
            let target = train[row_start + n_features];
            if !target.is_finite() {
                return Err(format!("Nonfinite target value at row {}", i));
            }
            if target < y_min {
                y_min = target;
            }
            if target > y_max {
                y_max = target;
            }
            y_sum += target;
            y_raw.push(target);
        }

        let y_mean = y_sum / (n_rows as f64);
        let mut y_sq_diff = 0.0;
        for &yv in &y_raw {
            let d = yv - y_mean;
            y_sq_diff += d * d;
        }
        let y_var = y_sq_diff / (n_rows as f64);
        let is_constant_target = (y_max - y_min).abs() <= 1e-12 || y_var <= 1e-24;
        let y_std = if is_constant_target {
            1.0
        } else {
            y_var.sqrt()
        };
        let y_span = (y_max - y_min).abs().max(y_std);

        if n_features == 0 || is_constant_target || n_rows == 1 {
            let prep = Preprocessor::fit(&raw_x, n_rows, n_features, &vec![0.0; n_rows]);
            self.state = Some(FittedState {
                n_features,
                y_mean,
                y_min,
                y_max,
                y_span,
                is_constant_target: true,
                target_transform: TargetTransform::Identity {
                    mean: y_mean,
                    std: y_std,
                },
                smear_factor: 1.0,
                preprocessor: prep,
                folds: Vec::new(),
                ensemble_weights: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                meta_scale: 1.0,
                meta_shift: 0.0,
            });
            return Ok(());
        }

        let (id_trans, alt_trans) = TargetTransform::fit(&y_raw);
        let (target_transform, smear_factor) = Self::select_best_target_transform(
            &raw_x, n_rows, n_features, &y_raw, id_trans, alt_trans,
        );

        let y_norm: Vec<f64> = y_raw
            .iter()
            .map(|&yv| target_transform.forward(yv))
            .collect();

        let mut y_sorted = y_norm.clone();
        y_sorted.sort_by(|&a, &b| total_cmp_f64(a, b));
        let y_iqr = (y_sorted[(3 * n_rows) / 4] - y_sorted[n_rows / 4]).max(0.5);
        let huber_delta = (1.35 * y_iqr).clamp(0.8, 3.5);

        let mut rng = XorShift64::new(self.seed);
        let preprocessor = Preprocessor::fit(&raw_x, n_rows, n_features, &y_norm);

        let n_folds = if n_rows < 12 {
            2
        } else if n_rows <= 6500 {
            5
        } else {
            3
        };

        // Target-stratified fold assignment
        let mut sorted_row_idx: Vec<usize> = (0..n_rows).collect();
        sorted_row_idx.sort_by(|&a, &b| total_cmp_f64(y_norm[a], y_norm[b]));
        let mut fold_of_row = vec![0usize; n_rows];
        for (rank, &r_idx) in sorted_row_idx.iter().enumerate() {
            fold_of_row[r_idx] = rank % n_folds;
        }

        // Precompute fold target encoders and populate OOF target-encoded columns
        let mut fold_tes = Vec::with_capacity(n_folds);
        for f in 0..n_folds {
            let mut tr_idx = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                if fold_of_row[i] != f {
                    tr_idx.push(i);
                }
            }
            fold_tes.push(FoldTargetEncoder::fit(
                &raw_x,
                n_features,
                &y_norm,
                &tr_idx,
                &preprocessor,
            ));
        }

        let lin_dim = preprocessor.linear_feature_dim();
        let base_tree_dim = preprocessor.base_tree_feature_dim();
        let tree_dim = preprocessor.total_tree_feature_dim();
        let n_cat = preprocessor.cat_col_indices.len();

        let mut x_lin = vec![0.0; n_rows * lin_dim];
        let mut x_tree = vec![0.0; n_rows * tree_dim];

        for i in 0..n_rows {
            let r_slice = &raw_x[i * n_features..(i + 1) * n_features];
            let f_idx = fold_of_row[i];
            if lin_dim > 0 {
                let lin_row = &mut x_lin[i * lin_dim..(i + 1) * lin_dim];
                let te_offset = preprocessor.transform_row_for_linear_base(r_slice, lin_row);
                if n_cat > 0 {
                    fold_tes[f_idx].encode_row(r_slice, &preprocessor, &mut lin_row[te_offset..]);
                }
            }
            if tree_dim > 0 {
                let tree_row = &mut x_tree[i * tree_dim..(i + 1) * tree_dim];
                preprocessor.transform_row_for_tree_base(r_slice, &mut tree_row[..base_tree_dim]);
                if n_cat > 0 {
                    fold_tes[f_idx].encode_row(
                        r_slice,
                        &preprocessor,
                        &mut tree_row[base_tree_dim..],
                    );
                }
            }
        }

        let binner = HistogramBinner::build(&x_tree, n_rows, tree_dim);

        let max_trees = if n_rows * tree_dim > 4_000_000 {
            85
        } else if n_rows > 15_000 {
            135
        } else {
            210
        };
        let min_child = if n_rows < 40 {
            2
        } else if n_rows < 500 {
            5
        } else {
            12
        };

        let mut folds = Vec::with_capacity(n_folds);
        let mut oof_preds = vec![[0.0; 6]; n_rows];

        for f in 0..n_folds {
            let mut train_idx = Vec::with_capacity(n_rows);
            let mut val_idx = Vec::with_capacity(n_rows / n_folds + 1);
            for i in 0..n_rows {
                if fold_of_row[i] == f {
                    val_idx.push(i);
                } else {
                    train_idx.push(i);
                }
            }

            let enet = ElasticNetModel::fit_with_grid(&x_lin, &y_norm, lin_dim, &train_idx, &val_idx);

            let oblivious_gbdt = ObliviousGBDTModel::fit(
                &binner,
                &y_norm,
                &train_idx,
                &val_idx,
                6,
                min_child,
                3.0,
                0.85,
                0.055,
                max_trees + 30,
                &mut rng,
            );

            let gbdt_shallow = GBDTModel::fit(
                &binner,
                &y_norm,
                &train_idx,
                &val_idx,
                4,
                min_child,
                2.0,
                0.80,
                0.055,
                max_trees,
                false,
                huber_delta,
                &mut rng,
            );

            let gbdt_deep_huber = GBDTModel::fit(
                &binner,
                &y_norm,
                &train_idx,
                &val_idx,
                7,
                (min_child / 2).max(2),
                6.5,
                0.65,
                0.042,
                max_trees,
                true,
                huber_delta,
                &mut rng,
            );

            let res_mlp = ResMLPModel::fit(&x_lin, &y_norm, lin_dim, &train_idx, &val_idx, &mut rng);
            let knn = KNNModel::fit(&x_lin, &y_norm, lin_dim, &train_idx, &val_idx, &mut rng);

            for &vi in &val_idx {
                let lin_row = &x_lin[vi * lin_dim..(vi + 1) * lin_dim];
                let tree_row = &x_tree[vi * tree_dim..(vi + 1) * tree_dim];
                oof_preds[vi][0] = enet.predict_row(lin_row);
                oof_preds[vi][1] = oblivious_gbdt.predict_row(tree_row);
                oof_preds[vi][2] = gbdt_shallow.predict_row(tree_row);
                oof_preds[vi][3] = gbdt_deep_huber.predict_row(tree_row);
                oof_preds[vi][4] = res_mlp.predict_row(lin_row);
                oof_preds[vi][5] = knn.predict_row(lin_row);
            }

            folds.push(FittedFold {
                target_encoder: fold_tes[f].clone(),
                enet,
                oblivious_gbdt,
                gbdt_shallow,
                gbdt_deep_huber,
                res_mlp,
                knn,
            });
        }

        let ensemble_weights = Self::optimize_simplex_weights(&oof_preds, &y_norm);

        // Shrinkage calibration on OOF combined predictions
        let mut s_xy = 0.0;
        let mut s_xx = 0.0;
        let mut s_x = 0.0;
        let mut s_y = 0.0;
        for i in 0..n_rows {
            let mut p = 0.0;
            for m in 0..6 {
                p += ensemble_weights[m] * oof_preds[i][m];
            }
            s_x += p;
            s_y += y_norm[i];
            s_xx += p * p;
            s_xy += p * y_norm[i];
        }
        let nf = n_rows as f64;
        let mx = s_x / nf;
        let my = s_y / nf;
        let var_x = s_xx - nf * mx * mx;
        let cov_xy = s_xy - nf * mx * my;
        let (meta_scale, meta_shift) = if var_x > 1e-6 && n_rows >= 30 {
            let slope = ((cov_xy + 5.0) / (var_x + 5.0)).clamp(0.85, 1.15);
            let shift = (my - slope * mx).clamp(-0.2, 0.2);
            (slope, shift)
        } else {
            (1.0, 0.0)
        };

        self.state = Some(FittedState {
            n_features,
            y_mean,
            y_min,
            y_max,
            y_span,
            is_constant_target: false,
            target_transform,
            smear_factor,
            preprocessor,
            folds,
            ensemble_weights,
            meta_scale,
            meta_shift,
        });

        Ok(())
    }

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String> {
        let st = self
            .state
            .as_ref()
            .ok_or_else(|| "Regressor has not been successfully fitted".to_string())?;

        if n_features != st.n_features {
            return Err(format!(
                "Prediction feature count {} does not match fitted feature count {}",
                n_features, st.n_features
            ));
        }

        let expected_len = n_rows
            .checked_mul(n_features)
            .ok_or_else(|| "Dimension multiplication overflow in predict".to_string())?;
        if features.len() != expected_len {
            return Err(format!(
                "Flattened features length {} does not match n_rows ({}) * n_features ({})",
                features.len(),
                n_rows,
                n_features
            ));
        }

        for (idx, &v) in features.iter().enumerate() {
            if v.is_infinite() {
                return Err(format!("Infinite feature value at index {}", idx));
            }
        }

        if n_rows == 0 {
            return Ok(Vec::new());
        }

        if st.is_constant_target || st.n_features == 0 || st.folds.is_empty() {
            return Ok(vec![st.y_mean; n_rows]);
        }

        let lin_dim = st.preprocessor.linear_feature_dim();
        let base_tree_dim = st.preprocessor.base_tree_feature_dim();
        let tree_dim = st.preprocessor.total_tree_feature_dim();
        let n_cat = st.preprocessor.cat_col_indices.len();

        let mut lin_buf = vec![0.0; lin_dim];
        let mut tree_buf = vec![0.0; tree_dim];
        let mut preds = Vec::with_capacity(n_rows);

        let inv_folds = 1.0 / (st.folds.len() as f64);
        let clamp_lo = st.y_min - 3.0 * st.y_span;
        let clamp_hi = st.y_max + 3.0 * st.y_span;

        for i in 0..n_rows {
            let row = &features[i * n_features..(i + 1) * n_features];
            let te_lin_offset = st.preprocessor.transform_row_for_linear_base(row, &mut lin_buf);
            st.preprocessor
                .transform_row_for_tree_base(row, &mut tree_buf[..base_tree_dim]);

            let mut model_sums = [0.0; 6];
            for fold in &st.folds {
                if n_cat > 0 {
                    fold.target_encoder
                        .encode_row(row, &st.preprocessor, &mut lin_buf[te_lin_offset..]);
                    fold.target_encoder.encode_row(
                        row,
                        &st.preprocessor,
                        &mut tree_buf[base_tree_dim..],
                    );
                }
                model_sums[0] += fold.enet.predict_row(&lin_buf);
                model_sums[1] += fold.oblivious_gbdt.predict_row(&tree_buf);
                model_sums[2] += fold.gbdt_shallow.predict_row(&tree_buf);
                model_sums[3] += fold.gbdt_deep_huber.predict_row(&tree_buf);
                model_sums[4] += fold.res_mlp.predict_row(&lin_buf);
                model_sums[5] += fold.knn.predict_row(&lin_buf);
            }

            let mut norm_pred = 0.0;
            for m in 0..6 {
                norm_pred += st.ensemble_weights[m] * (model_sums[m] * inv_folds);
            }
            let calibrated_z = st.meta_scale * norm_pred + st.meta_shift;
            let raw_pred = st
                .target_transform
                .inverse(calibrated_z, st.smear_factor);

            let final_pred = if raw_pred.is_finite() {
                raw_pred.clamp(clamp_lo, clamp_hi)
            } else {
                st.y_mean
            };
            preds.push(final_pred);
        }

        Ok(preds)
    }
}
