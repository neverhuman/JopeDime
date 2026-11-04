use super::*;

pub fn sorted_col(x: &Rows, j: usize) -> Vec<f64> {
    let mut v: Vec<f64> = x.iter().map(|r| r[j]).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

/// Prefix-preserving cache of sorted training columns across pipeline stages (`G-8`).
///
/// Whenever a transform stage runs with `append = true`, `finish` preserves the input prefix
/// `0..in_cols.min(MAX_WIDTH)` verbatim (once sanitized to finite `f64`). Thus any column
/// sorted in an earlier stage (or reused by a downstream `histgbm` head) can reuse its
/// `total_cmp`-sorted vector without re-extracting or re-sorting `x[..][j]`.
#[derive(Clone, Debug, Default)]
pub(crate) struct SortedColumnsCache {
    cols: Vec<Option<std::sync::Arc<Vec<f64>>>>,
    row_count: Option<usize>,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl SortedColumnsCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ensure_cols(&mut self, x: &Rows, cols: usize) {
        if x.is_empty() || cols == 0 {
            return;
        }
        if self.row_count != Some(x.len()) {
            self.cols.clear();
            self.row_count = Some(x.len());
        }
        if self.cols.len() < cols {
            self.cols.resize(cols, None);
        }
        let mut missing = Vec::new();
        for j in 0..cols {
            if self.cols[j].is_some() {
                self.cache_hits += 1;
            } else {
                self.cache_misses += 1;
                missing.push(j);
            }
        }
        if missing.is_empty() {
            return;
        }
        if missing.len() >= 16 && x.len() * missing.len() >= 16_384 && crate::is_parallel_enabled()
        {
            let computed: Vec<(usize, std::sync::Arc<Vec<f64>>)> = missing
                .into_par_iter()
                .map(|j| (j, std::sync::Arc::new(sorted_col(x, j))))
                .collect();
            for (j, s) in computed {
                self.cols[j] = Some(s);
            }
        } else {
            for j in missing {
                self.cols[j] = Some(std::sync::Arc::new(sorted_col(x, j)));
            }
        }
    }

    pub fn col_slice(&self, j: usize) -> &[f64] {
        self.cols[j]
            .as_deref()
            .expect("column must be ensured in SortedColumnsCache before reading")
    }

    pub fn get_col_slice(&self, j: usize) -> Option<&[f64]> {
        self.cols
            .get(j)
            .and_then(|s| s.as_deref().map(Vec::as_slice))
    }

    pub fn record_probe_counts(&mut self, cols: usize) {
        for j in 0..cols {
            if self.get_col_slice(j).is_some() {
                self.cache_hits += 1;
            } else {
                self.cache_misses += 1;
            }
        }
    }

    pub fn col_vec(&self, j: usize) -> Vec<f64> {
        self.col_slice(j).to_vec()
    }

    pub fn on_stage_complete(&mut self, append: bool, in_cols: usize, new_width: usize) {
        if !append {
            self.cols.clear();
            return;
        }
        let keep_prefix = in_cols.min(new_width).min(crate::MAX_WIDTH);
        self.cols.truncate(keep_prefix);
        for slot in &mut self.cols {
            if let Some(s) = slot {
                let all_finite = s.first().is_none_or(|v| v.is_finite())
                    && s.last().is_none_or(|v| v.is_finite());
                if !all_finite {
                    *slot = None;
                }
            }
        }
        if self.cols.len() < new_width {
            self.cols.resize(new_width, None);
        }
    }
}

/// Interpolated quantile on a sorted slice (numpy 'linear' convention).
pub fn quantile_sorted(s: &[f64], q: f64) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let pos = q * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    let frac = pos - lo as f64;
    s[lo] * (1.0 - frac) + s[hi] * frac
}

/// Number of elements strictly less than v (Rust partition_point / Python bisect_left).
pub fn lower_bound(s: &[f64], v: f64) -> usize {
    s.partition_point(|e| *e < v)
}

/// Number of elements less than or equal to v (bisect_right).
pub fn upper_bound(s: &[f64], v: f64) -> usize {
    s.partition_point(|e| *e <= v)
}

/// Mid-rank of v against sorted train values, scaled to [0,1].
pub fn mid_rank(s: &[f64], v: f64) -> f64 {
    let n = s.len();
    let lb = lower_bound(s, v);
    let ub = if lb < n {
        let slb = s[lb];
        if slb > v {
            lb
        } else if slb == v && (lb + 1 == n || s[lb + 1] > v) {
            lb + 1
        } else {
            upper_bound(s, v)
        }
    } else if n == 0 || s[n - 1] < v {
        n
    } else {
        upper_bound(s, v)
    };
    (lb + ub) as f64 / (2.0 * n as f64)
}

/// Acklam's rational approximation to the standard normal quantile function.
pub fn probit(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let p_low = 0.02425;
    if p < p_low {
        let q = (-2.0 * crate::portable_math::frozen_s1::ln(p)).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - p_low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * crate::portable_math::frozen_s1::ln(1.0 - p)).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

pub(crate) fn quantile_edges_with_cache(
    x: &Rows,
    bins: usize,
    cache: &mut SortedColumnsCache,
) -> Vec<Vec<f64>> {
    let cols = x[0].len();
    cache.ensure_cols(x, cols);
    (0..cols)
        .map(|j| {
            let s = cache.col_slice(j);
            (1..bins)
                .map(|b| quantile_sorted(s, b as f64 / bins as f64))
                .collect()
        })
        .collect()
}

/// Numerically stable inverse hyperbolic sine via the identity
/// sign(v) * ln(|v| + sqrt(v*v + 1)). Never `f64::asinh`: the libm
/// implementation is not guaranteed bit-identical to the emitted Python twin,
/// this shared formula is. Used by the `symlog` stage and the scalar `asinh`.
pub(crate) fn asinh_stable(v: f64) -> f64 {
    let a = v.abs();
    let s = if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    };
    s * (a + (v * v + 1.0).sqrt()).ln()
}

pub(crate) fn get_append(stage: &Stage) -> bool {
    matches!(
        stage.params.get("append"),
        Some(crate::dsl::ParamValue::Bool(true))
    ) || stage.get_usize("append", 0) == 1
}

// ---------------------------------- fit ----------------------------------
