//! CPU implementation of the high-resolution landmark recipe.
//!
//! The FIT transform and descriptor extraction port the pinned
//! `FusedLandmarkWorkspace.fit_transform_and_extract_bundle_high` path into
//! Rust, including its 1,024-row FIT cap. The public builder returns the
//! 256-value vector and raw 24-column descriptors; ranker-facing columns are
//! projected into the existing 64-by-17 shape. Numeric feature scaling is
//! regression-only, matching the source recipe's numeric input contract.

pub const VECTOR_WIDTH: usize = 256;
pub const REPORT_WIDTH: usize = 116;
pub const TABLE_WIDTH: usize = 24;
pub const INTERACTION_WIDTH: usize = 48;
pub const BUILDING_BLOCK_WIDTH: usize = 68;
pub const RAW_COLUMN_WIDTH: usize = 24;
pub const RANKER_COLUMN_WIDTH: usize = 17;
pub const RANKER_COLUMN_COUNT: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkOutput {
    vector: [f32; VECTOR_WIDTH],
    columns: [[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT],
    raw_columns: Vec<[f32; RAW_COLUMN_WIDTH]>,
}

impl LandmarkOutput {
    pub fn from_descriptor_parts(
        report: &[f32; REPORT_WIDTH],
        table: &[f32; TABLE_WIDTH],
        interactions: &[f32; INTERACTION_WIDTH],
        building_blocks: &[f32; BUILDING_BLOCK_WIDTH],
        raw_columns: &[[f32; RAW_COLUMN_WIDTH]],
    ) -> Self {
        Self {
            vector: assemble_vector(report, table, interactions, building_blocks),
            columns: pad_ranker_columns(raw_columns),
            raw_columns: raw_columns.to_vec(),
        }
    }

    pub fn vector(&self) -> &[f32; VECTOR_WIDTH] {
        &self.vector
    }

    pub fn columns(&self) -> &[[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT] {
        &self.columns
    }

    pub fn raw_columns(&self) -> &[[f32; RAW_COLUMN_WIDTH]] {
        &self.raw_columns
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgreementSummary {
    pub sampled_rows: usize,
    pub full_rows: usize,
    pub values_compared: usize,
    pub exact_matches: usize,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgreementError {
    EmptyTable,
    SampleOverLimit,
    SampleExceedsFullTable,
}

impl std::fmt::Display for AgreementError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTable => formatter.write_str("landmark tables must contain FIT rows"),
            Self::SampleOverLimit => {
                formatter.write_str("landmark FIT sample exceeds the 1,024-row cap")
            }
            Self::SampleExceedsFullTable => {
                formatter.write_str("landmark FIT sample exceeds the full-table row count")
            }
        }
    }
}

impl std::error::Error for AgreementError {}

/// Summarize sampled-versus-full outputs without imposing a pass threshold.
/// T099 requires reporting agreement; the qualification owner sets the
/// acceptance interpretation when RES-09 fixtures and approved inputs land.
pub fn compare_sampled_to_full(
    sampled: &LandmarkOutput,
    full: &LandmarkOutput,
    sampled_rows: usize,
    full_rows: usize,
) -> Result<AgreementSummary, AgreementError> {
    if sampled_rows == 0 || full_rows == 0 {
        return Err(AgreementError::EmptyTable);
    }
    if sampled_rows > 1_024 {
        return Err(AgreementError::SampleOverLimit);
    }
    if sampled_rows > full_rows {
        return Err(AgreementError::SampleExceedsFullTable);
    }

    let mut values_compared = 0;
    let mut exact_matches = 0;
    let mut max_abs_error = 0.0_f64;
    let mut total_abs_error = 0.0_f64;

    for (sampled_value, full_value) in sampled
        .vector
        .iter()
        .chain(sampled.columns.iter().flatten())
        .zip(full.vector.iter().chain(full.columns.iter().flatten()))
    {
        let abs_error = f64::from((*sampled_value - *full_value).abs());
        values_compared += 1;
        exact_matches += usize::from(abs_error == 0.0);
        max_abs_error = max_abs_error.max(abs_error);
        total_abs_error += abs_error;
    }

    Ok(AgreementSummary {
        sampled_rows,
        full_rows,
        values_compared,
        exact_matches,
        max_abs_error,
        mean_abs_error: total_abs_error / values_compared as f64,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatencySummary {
    pub samples: usize,
    pub min_ms: f64,
    pub median_ms: f64,
    pub max_ms: f64,
}

/// Summarize repeated `landmarks.ms` measurements for a pre-stage report.
pub fn summarize_latency(samples: &[std::time::Duration]) -> Option<LatencySummary> {
    if samples.is_empty() {
        return None;
    }

    let mut millis = samples
        .iter()
        .map(std::time::Duration::as_secs_f64)
        .map(|seconds| seconds * 1_000.0)
        .collect::<Vec<_>>();
    millis.sort_by(f64::total_cmp);

    let middle = millis.len() / 2;
    let median_ms = if millis.len() % 2 == 0 {
        (millis[middle - 1] + millis[middle]) / 2.0
    } else {
        millis[middle]
    };

    Some(LatencySummary {
        samples: millis.len(),
        min_ms: millis[0],
        median_ms,
        max_ms: millis[millis.len() - 1],
    })
}

/// Join the four high-level descriptor blocks in the training order.
///
/// The source concatenates `vector116`, `table_vec24`, `inter_vec48`, and
/// `building_block_vec68`, then replaces NaN and infinities with zero.
pub fn assemble_vector(
    report: &[f32; REPORT_WIDTH],
    table: &[f32; TABLE_WIDTH],
    interactions: &[f32; INTERACTION_WIDTH],
    building_blocks: &[f32; BUILDING_BLOCK_WIDTH],
) -> [f32; VECTOR_WIDTH] {
    let mut result = [0.0; VECTOR_WIDTH];

    for (offset, value) in report
        .iter()
        .chain(table)
        .chain(interactions)
        .chain(building_blocks)
        .enumerate()
    {
        result[offset] = finite_or_zero(*value);
    }

    result
}

/// Adapt raw 24-wide per-column landmarks to the ranker's `[64, 17]` input.
///
/// The training consumer copies the first 17 values for the first 64 columns,
/// truncates excess columns, zero-pads missing columns, and sanitizes
/// non-finite values to zero.
pub fn pad_ranker_columns(
    raw_columns: &[[f32; RAW_COLUMN_WIDTH]],
) -> [[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT] {
    let mut result = [[0.0; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT];

    for (target, source) in result.iter_mut().zip(raw_columns.iter()) {
        for (target_value, source_value) in target.iter_mut().zip(source.iter()) {
            *target_value = finite_or_zero(*source_value);
        }
    }

    result
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// Full CPU implementation of the numeric-regression `high` landmark recipe.
/// `x` is row-major FIT data; no validation/test rows are accepted by this API.
pub fn fit_transform_and_extract_bundle_high(
    x: &[f32],
    rows: usize,
    features: usize,
    y: &[f32],
) -> Result<LandmarkOutput, RecipeError> {
    if rows > 1_024 {
        return Err(RecipeError::FitSampleOverLimit);
    }
    build_high_full_table(x, rows, features, y)
}

fn build_high_full_table(
    x: &[f32],
    rows: usize,
    features: usize,
    y: &[f32],
) -> Result<LandmarkOutput, RecipeError> {
    if rows == 0 || features == 0 {
        return Err(RecipeError::EmptyTable);
    }
    if x.len() != rows.saturating_mul(features) || y.len() != rows {
        return Err(RecipeError::ShapeMismatch);
    }
    let (scaled_x, scaled_y) = scale_numeric_regression(x, rows, features, y);
    let stage = extract_probes(&scaled_x, &scaled_y, rows, features);
    let blocks = building_blocks(&stage, rows, features);
    let report = report_descriptors(&stage, &blocks, rows, features);
    let table = table_descriptors(&stage, rows, features);
    let inter = inter_descriptors(&report, &table, &blocks, features);

    let mut raw = vec![[0.0_f32; RAW_COLUMN_WIDTH]; features];
    for (j, column) in raw.iter_mut().enumerate() {
        column[..9].copy_from_slice(&report.column_spectrum[j]);
        column[9..17].copy_from_slice(&table.column[j]);
        column[17..24].copy_from_slice(&blocks.column[j]);
    }
    Ok(LandmarkOutput {
        vector: assemble_vector(&report.vector, &table.vector, &inter, &blocks.vector),
        columns: pad_ranker_columns(&raw),
        raw_columns: raw,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecipeError {
    EmptyTable,
    FitSampleOverLimit,
    ShapeMismatch,
}

impl std::fmt::Display for RecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTable => f.write_str("landmark FIT table must be nonempty"),
            Self::FitSampleOverLimit => f.write_str("landmark FIT sample exceeds 1,024 rows"),
            Self::ShapeMismatch => f.write_str("landmark FIT matrix and target shapes differ"),
        }
    }
}

impl std::error::Error for RecipeError {}

#[derive(Clone)]
struct ProbeStage {
    y_scaled: Vec<f64>,
    z: Vec<f64>,
    z_unc: Vec<f64>,
    y_z: Vec<f64>,
    probes: Vec<[f64; 14]>,
    table_cols: Vec<[f64; 6]>,
    row_corr: [f64; 8],
}

#[derive(Clone)]
struct BlockStage {
    raw_gain: [f64; 16],
    vector: [f32; BUILDING_BLOCK_WIDTH],
    column: Vec<[f32; 7]>,
}

#[derive(Clone)]
struct ReportStage {
    vector: [f32; REPORT_WIDTH],
    gains: [f64; 23],
    ses: [f64; 23],
    lambdas: [f64; 23],
    present: [bool; 23],
    column_spectrum: Vec<[f32; 9]>,
}

#[derive(Clone)]
struct TableStage {
    vector: [f32; TABLE_WIDTH],
    column: Vec<[f32; 8]>,
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() <= 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    let at = q * (sorted.len() - 1) as f64;
    let lo = at.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] * (1.0 - (at - lo as f64)) + sorted[hi] * (at - lo as f64)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn stddev(values: &[f64]) -> f64 {
    let m = mean(values);
    (mean(&values.iter().map(|v| (v - m) * (v - m)).collect::<Vec<_>>())).sqrt()
}

fn squash(value: f64) -> f64 {
    if value.is_finite() {
        value / (1.0 + value.abs())
    } else {
        0.0
    }
}

fn gini(values: &[f64]) -> f64 {
    let mut a: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(f64::abs)
        .collect();
    if a.len() <= 1 || a.iter().sum::<f64>() <= 1e-12 {
        return 0.0;
    }
    a.sort_by(f64::total_cmp);
    let n = a.len() as f64;
    let weighted = a
        .iter()
        .enumerate()
        .map(|(i, v)| (i + 1) as f64 * v)
        .sum::<f64>();
    ((2.0 * weighted / (n * a.iter().sum::<f64>())) - (n + 1.0) / n).clamp(0.0, 1.0)
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || a.len() != b.len() {
        return 0.0;
    }
    let ma = mean(a);
    let mb = mean(b);
    let sa = stddev(a);
    let sb = stddev(b);
    if sa <= 1e-9 || sb <= 1e-9 {
        return 0.0;
    }
    let c = a
        .iter()
        .zip(b)
        .map(|(x, y)| ((x - ma) / sa) * ((y - mb) / sb))
        .sum::<f64>()
        / a.len() as f64;
    if c.is_finite() {
        c.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn abs_corr_with_y(values: &[f64], y_z: &[f64]) -> f64 {
    if values.is_empty() || values.len() != y_z.len() {
        return 0.0;
    }
    let m = mean(values);
    let sd = (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64).sqrt();
    if sd <= 1e-9 || !sd.is_finite() {
        return 0.0;
    }
    let value = values
        .iter()
        .zip(y_z)
        .map(|(v, y)| ((v - m) / sd) * y)
        .sum::<f64>()
        .abs()
        / values.len() as f64;
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

// NumPy 1.26's default indirect quicksort is intentionally unstable. The
// source recipe uses np.argsort twice; matching its tie permutation is needed
// for Chatterjee xi on quantized/clipped columns.
fn recipe_argsort(values: &[f64]) -> Vec<usize> {
    const SMALL_QUICKSORT: usize = 15;
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    if n < 2 {
        return order;
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    let mut depth = (usize::BITS - 1 - n.leading_zeros()) as i32 * 2;
    let mut stack: Vec<(usize, usize, i32)> = Vec::new();
    loop {
        if depth < 0 {
            // The pinned fixtures do not reach this introsort fallback; this
            // keeps worst-case runtime bounded for arbitrary caller inputs.
            order[lo..=hi].sort_unstable_by(|a, b| values[*a].total_cmp(&values[*b]));
        } else {
            while hi.saturating_sub(lo) > SMALL_QUICKSORT {
                let mid = lo + ((hi - lo) >> 1);
                if values[order[mid]] < values[order[lo]] {
                    order.swap(mid, lo);
                }
                if values[order[hi]] < values[order[mid]] {
                    order.swap(hi, mid);
                }
                if values[order[mid]] < values[order[lo]] {
                    order.swap(mid, lo);
                }
                let pivot = values[order[mid]];
                let mut pi = lo;
                let mut pj = hi - 1;
                order.swap(mid, pj);
                loop {
                    pi += 1;
                    while values[order[pi]] < pivot {
                        pi += 1;
                    }
                    pj -= 1;
                    while pivot < values[order[pj]] {
                        pj -= 1;
                    }
                    if pi >= pj {
                        break;
                    }
                    order.swap(pi, pj);
                }
                let pk = hi - 1;
                order.swap(pi, pk);
                if pi - lo < hi - pi {
                    stack.push((pi + 1, hi, depth - 1));
                    hi = pi - 1;
                } else {
                    stack.push((lo, pi - 1, depth - 1));
                    lo = pi + 1;
                }
                depth -= 1;
            }
            for pi in lo + 1..=hi {
                let vi = order[pi];
                let vp = values[vi];
                let mut pj = pi;
                while pj > lo && vp < values[order[pj - 1]] {
                    order[pj] = order[pj - 1];
                    pj -= 1;
                }
                order[pj] = vi;
            }
        }
        match stack.pop() {
            Some((next_lo, next_hi, next_depth)) => {
                lo = next_lo;
                hi = next_hi;
                depth = next_depth;
            }
            None => break,
        }
    }
    order
}

fn scale_numeric_regression(x: &[f32], n: usize, p: usize, y: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let yy: Vec<f64> = y
        .iter()
        .map(|v| if v.is_finite() { f64::from(*v) } else { 0.0 })
        .collect();
    let ym = mean(&yy);
    let ys = stddev(&yy).max(1e-6);
    let sy: Vec<f64> = y
        .iter()
        .map(|v| {
            if v.is_finite() {
                8.0 * (((f64::from(*v) - ym) / ys) / 8.0).tanh()
            } else {
                0.0
            }
        })
        .collect();
    let qgrid = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
    let norm = [
        -2.32634787,
        -1.64485363,
        -1.28155157,
        -0.67448975,
        0.0,
        0.67448975,
        1.28155157,
        1.64485363,
        2.32634787,
    ];
    let mut out = vec![0.0; n * p];
    for j in 0..p {
        let mut col: Vec<f64> = (0..n)
            .map(|i| {
                let v = f64::from(x[i * p + j]);
                if v.is_finite() {
                    v
                } else {
                    0.0
                }
            })
            .collect();
        let mut sorted = col.clone();
        sorted.sort_by(f64::total_cmp);
        let unique =
            sorted.windows(2).filter(|w| w[0] != w[1]).count() + usize::from(!sorted.is_empty());
        if unique <= 1 {
            continue;
        }
        let m = mean(&col);
        let s = stddev(&col);
        if unique == 2 {
            let den = s.max(1e-6);
            for i in 0..n {
                out[i * p + j] = 6.0 * (((col[i] - m) / den).clamp(-6.0, 6.0) / 6.0).tanh();
            }
            continue;
        }
        let q: [f64; 9] = std::array::from_fn(|k| quantile(&sorted, qgrid[k]));
        let mut skew = 0.0;
        let mut kurt = 0.0;
        if n >= 4 && s > 1e-12 {
            skew = mean(
                &col.iter()
                    .map(|v| ((v - m) / s).powi(3))
                    .collect::<Vec<_>>(),
            )
            .clamp(-50.0, 50.0);
            kurt = (mean(
                &col.iter()
                    .map(|v| ((v - m) / s).powi(4))
                    .collect::<Vec<_>>(),
            ) - 3.0)
                .clamp(-50.0, 200.0);
        }
        let med = q[4];
        let iqr = (q[5] - q[3]).max(0.0);
        let span90 = (q[7] - q[1]).max(0.0);
        let tail = if iqr > 1e-9 { span90 / iqr } else { 1.0 };
        let mut abs_dev: Vec<f64> = sorted.iter().map(|v| (v - med).abs()).collect();
        abs_dev.sort_by(f64::total_cmp);
        let robust = (iqr / 1.3489795)
            .max(quantile(&abs_dev, 0.5) * 1.4826)
            .max(s * 0.25)
            .max(1e-6);
        let use_log = skew.abs() > 1.0 || kurt > 4.0 || tail > 4.2;
        let blend = if use_log { 0.35 } else { 0.30 };
        let mut zlog = if use_log {
            col.iter()
                .map(|v| {
                    let d = v - med;
                    d.signum() * (d.abs() / robust).ln_1p()
                })
                .collect::<Vec<_>>()
        } else {
            col.iter()
                .map(|v| (v - m) / s.max(1e-6))
                .collect::<Vec<_>>()
        };
        let lm = mean(&zlog);
        let ls = stddev(&zlog).max(1e-6);
        for z in &mut zlog {
            *z = (*z - lm) / ls;
        }
        let strict = q.windows(2).all(|w| w[1] - w[0] > 1e-12);
        let iqr_safe = (q[5] - q[3]).max(1e-9);
        for i in 0..n {
            let mut z = zlog[i];
            if strict {
                let v = col[i];
                let zq = if v < q[0] {
                    norm[0] + (v - q[0]) / iqr_safe * 1.3489795
                } else if v > q[8] {
                    norm[8] + (v - q[8]) / iqr_safe * 1.3489795
                } else {
                    let seg = (0..8).find(|&k| v >= q[k] && v <= q[k + 1]).unwrap_or(0);
                    let t = (v - q[seg]) / (q[seg + 1] - q[seg]);
                    norm[seg] + t * (norm[seg + 1] - norm[seg])
                };
                z = (1.0 - blend) * z + blend * zq;
            }
            out[i * p + j] = 6.0 * (z / 6.0).tanh();
        }
        col.clear();
    }
    (out, sy)
}

fn extract_probes(x: &[f64], y: &[f64], n: usize, p: usize) -> ProbeStage {
    let ym = mean(y);
    let ys = stddev(y);
    let yz: Vec<f64> = y
        .iter()
        .map(|v| if ys > 1e-9 { (v - ym) / ys } else { 0.0 })
        .collect();
    let mut z = vec![0.0; n * p];
    let mut z_unc = vec![0.0; n * p];
    let mut probes = vec![[0.0; 14]; p];
    let mut table_cols = vec![[0.0; 6]; p];
    for j in 0..p {
        let col: Vec<f64> = (0..n).map(|i| x[i * p + j]).collect();
        let m = mean(&col);
        let s = stddev(&col);
        let unc: Vec<f64> = col
            .iter()
            .map(|v| if s > 1e-9 { (v - m) / s } else { 0.0 })
            .collect();
        let clp: Vec<f64> = unc.iter().map(|v| v.clamp(-8.0, 8.0)).collect();
        for i in 0..n {
            z[i * p + j] = clp[i];
            z_unc[i * p + j] = unc[i];
        }
        let mut sorted = unc.clone();
        sorted.sort_by(f64::total_cmp);
        let q05 = quantile(&sorted, 0.05);
        let q25 = quantile(&sorted, 0.25);
        let q50 = quantile(&sorted, 0.5);
        let q75 = quantile(&sorted, 0.75);
        let q95 = quantile(&sorted, 0.95);
        let q10 = quantile(&sorted, 0.10).clamp(-8.0, 8.0);
        let q90 = quantile(&sorted, 0.90).clamp(-8.0, 8.0);
        let tail = ((q95 - q05) / (q75 - q25).max(1e-6)).clamp(0.0, 20.0) / 10.0;
        let skew = mean(&unc.iter().map(|v| v.powi(3)).collect::<Vec<_>>());
        let kurt =
            (mean(&unc.iter().map(|v| v.powi(4)).collect::<Vec<_>>()) - 3.0).clamp(-3.0, 50.0);
        let mode = unc.iter().filter(|v| (**v - q50).abs() < 0.05).count() as f64 / n as f64;
        let pear = if ys > 1e-9 {
            unc.iter().zip(&yz).map(|(a, b)| a * b).sum::<f64>().abs() / n as f64
        } else {
            0.0
        };
        let ranks = recipe_argsort(y);
        let mut yr = vec![0.0; n];
        for (r, &i) in ranks.iter().enumerate() {
            yr[i] = r as f64;
        }
        let order = recipe_argsort(&clp);
        let rank_y: Vec<f64> = order.iter().map(|&i| yr[i]).collect();
        let xi = if ys > 1e-9 && n > 1 {
            (1.0 - 3.0 * rank_y.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>()
                / (n - 1) as f64
                / (n + 1).max(2) as f64)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let yrz: Vec<f64> = yr
            .iter()
            .map(|v| (v - 0.5 * (n - 1) as f64) / (((n * n - 1) as f64 / 12.0).sqrt().max(1e-9)))
            .collect();
        let spearman = if ys > 1e-9 {
            (order
                .iter()
                .enumerate()
                .map(|(r, &i)| {
                    (r as f64 - 0.5 * (n - 1) as f64)
                        / (((n * n - 1) as f64 / 12.0).sqrt().max(1e-9))
                        * yrz[i]
                })
                .sum::<f64>()
                / n as f64)
                .abs()
        } else {
            0.0
        };
        table_cols[j] = [skew, kurt, tail, mode, pear, spearman];
        let q50c = q50.clamp(-8.0, 8.0);
        let q25c = quantile(&sorted, 0.25).clamp(-8.0, 8.0);
        let q75c = quantile(&sorted, 0.75).clamp(-8.0, 8.0);
        let transforms: [Vec<f64>; 13] = [
            clp.iter().map(|v| v.asinh()).collect(),
            clp.iter().map(|v| v.atan()).collect(),
            clp.iter()
                .map(|v| erf_approx(v / std::f64::consts::SQRT_2))
                .collect(),
            clp.iter().map(|v| v.signum() * v.abs().cbrt()).collect(),
            clp.iter().map(|v| v.signum() * v.abs().ln_1p()).collect(),
            clp.iter().map(|v| (0.75 * v).tanh()).collect(),
            clp.iter().map(|v| (-v.abs()).exp_m1()).collect(),
            clp.iter().map(|v| v.clamp(q10, q90)).collect(),
            clp.iter()
                .map(|v| if *v >= q50c { 1.0 } else { 0.0 })
                .collect(),
            clp.iter()
                .map(|v| (v - q75c).max(0.0) - (q25c - v).max(0.0))
                .collect(),
            clp.iter()
                .map(|v| (0.5 * std::f64::consts::PI * v).sin())
                .collect(),
            clp.iter().map(|v| v - v.floor()).collect(),
            clp.clone(),
        ];
        for k in 0..13 {
            probes[j][k] = abs_corr_with_y(&transforms[k], &yz);
        }
        probes[j][12] = xi;
        probes[j][13] = pear;
        let _ = q10;
    }
    let row_corr = row_summary_corr(&z, n, p, &yz);
    ProbeStage {
        y_scaled: y.to_vec(),
        z,
        z_unc,
        y_z: yz,
        probes,
        table_cols,
        row_corr,
    }
}

fn erf_approx(x: f64) -> f64 {
    let sign = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-a * a).exp())
}

fn row_summary_corr(z: &[f64], n: usize, p: usize, yz: &[f64]) -> [f64; 8] {
    let mut rows: Vec<[f64; 8]> = Vec::with_capacity(n);
    for i in 0..n {
        let r: Vec<f64> = (0..p).map(|j| z[i * p + j]).collect();
        let m = mean(&r);
        let sd = (mean(&r.iter().map(|v| (v - m).powi(2)).collect::<Vec<_>>()) + 1e-8).sqrt();
        let l1 = mean(&r.iter().map(|v| v.abs()).collect::<Vec<_>>());
        let l2 = (mean(&r.iter().map(|v| v * v).collect::<Vec<_>>()) + 1e-8).sqrt();
        let lo = r.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let skew =
            mean(&r.iter().map(|v| ((v - m) / sd).powi(3)).collect::<Vec<_>>()).clamp(-10.0, 10.0);
        rows.push([m, sd, l1, l2, lo, hi, skew, 0.0]);
    }
    std::array::from_fn(|k| abs_corr_with_y(&rows.iter().map(|r| r[k]).collect::<Vec<_>>(), yz))
}

fn building_blocks(s: &ProbeStage, n: usize, p: usize) -> BlockStage {
    let max_probe = |k: usize| s.probes.iter().map(|r| r[k]).fold(0.0_f64, f64::max);
    let row_gain = s.row_corr.iter().copied().fold(0.0_f64, f64::max);
    let mut gain = [0.0; 16];
    gain[0] = max_probe(0);
    gain[1] = max_probe(1);
    gain[2] = max_probe(2);
    gain[3] = max_probe(3);
    gain[4] = max_probe(4);
    gain[5] = row_gain;
    gain[6] = max_probe(5);
    gain[7] = max_probe(6);
    gain[8] = max_probe(7);
    gain[9] = max_probe(8);
    gain[10] = max_probe(9);
    gain[11] = max_probe(10);
    gain[12] = max_probe(11);
    gain[11] = (0..p)
        .map(|j| {
            let sin = (0..n)
                .map(|i| (0.5 * std::f64::consts::PI * s.z[i * p + j]).sin())
                .collect::<Vec<_>>();
            let cos = (0..n)
                .map(|i| (0.5 * std::f64::consts::PI * s.z[i * p + j]).cos())
                .collect::<Vec<_>>();
            abs_corr_with_y(&sin, &s.y_z).max(abs_corr_with_y(&cos, &s.y_z))
        })
        .fold(0.0, f64::max);
    if p >= 2 && stddev(&s.y_scaled) > 1e-9 {
        let mut idx: Vec<usize> = (0..p).collect();
        idx.sort_by(|&a, &b| {
            s.probes[b][0]
                .max(s.probes[b][8])
                .max(s.probes[b][12])
                .total_cmp(&s.probes[a][0].max(s.probes[a][8]).max(s.probes[a][12]))
        });
        idx.truncate(p.min(6));
        let (mut h, mut a, mut d) = (Vec::new(), Vec::new(), Vec::new());
        for ia in 0..idx.len() {
            for ib in ia + 1..idx.len() {
                let mut hv = Vec::with_capacity(n);
                let mut av = Vec::with_capacity(n);
                let mut dv = Vec::with_capacity(n);
                for i in 0..n {
                    let x = s.z[i * p + idx[ia]];
                    let y = s.z[i * p + idx[ib]];
                    hv.push(x.hypot(y));
                    av.push(x.atan2(y));
                    dv.push((x - y) / (1.0 + x.abs() + y.abs()));
                }
                h.push(abs_corr_with_y(&hv, &s.y_z));
                a.push(abs_corr_with_y(&av, &s.y_z));
                d.push(abs_corr_with_y(&dv, &s.y_z));
            }
        }
        gain[13] = h.into_iter().fold(0.0, f64::max);
        gain[14] = a.into_iter().fold(0.0, f64::max);
        gain[15] = d.into_iter().fold(0.0, f64::max);
    }
    let high = {
        let mut v = gain;
        v[14] = 0.0;
        v[15] = 0.0;
        v
    };
    let lin = max_probe(13);
    let denom = high.iter().copied().fold(lin, f64::max).max(0.02);
    let mut contrast = [0.0; 16];
    contrast[0] = (high[0] - lin) / denom;
    contrast[1] = (high[1] - lin) / denom;
    contrast[2] = (high[2] - lin) / denom;
    contrast[3] = (high[3] - lin) / denom;
    contrast[4] = (high[5] - lin) / denom;
    contrast[5] = mean(&s.row_corr) / denom;
    contrast[6] = gini(&s.row_corr);
    contrast[7] = (high[7] - high[6]) / denom;
    contrast[8] = (high[8] - lin) / denom;
    contrast[9] = (high[9] - high[2]) / denom;
    contrast[10] = (high[10] - lin) / denom;
    contrast[11] = (high[11] - lin) / denom;
    contrast[12] = (high[12] - lin) / denom;
    contrast[13] = (high[13] - lin) / denom;
    let mut het = [0.0; 8];
    if stddev(&s.y_scaled) > 1e-9 {
        let bj = (0..p)
            .max_by(|&a, &b| s.probes[a][0].total_cmp(&s.probes[b][0]))
            .unwrap_or(0);
        let resid: Vec<f64> = (0..n)
            .map(|i| (s.y_z[i] - s.probes[bj][0] * s.z[i * p + bj]).abs())
            .collect();
        let rs = stddev(&resid);
        let rz: Vec<f64> = resid
            .iter()
            .map(|v| {
                if rs > 1e-9 {
                    (v - mean(&resid)) / rs
                } else {
                    0.0
                }
            })
            .collect();
        let hc: Vec<f64> = (0..p)
            .map(|j| abs_corr_with_y(&(0..n).map(|i| s.z[i * p + j]).collect::<Vec<_>>(), &rz))
            .collect();
        let mut sorted = hc.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        het[0] = sorted.first().copied().unwrap_or(0.0);
        het[1] = mean(&sorted[..sorted.len().min(3)]);
        het[2] = mean(&hc);
        het[3] = gini(&hc);
        let y_hi: Vec<f64> = s
            .y_z
            .iter()
            .map(|v| if *v > 1.0 { 1.0 } else { 0.0 })
            .collect();
        let y_lo: Vec<f64> = s
            .y_z
            .iter()
            .map(|v| if *v < -1.0 { 1.0 } else { 0.0 })
            .collect();
        let mut u = Vec::new();
        let mut l = Vec::new();
        for j in 0..p {
            let xu: Vec<f64> = (0..n)
                .map(|i| if s.z[i * p + j] > 1.0 { 1.0 } else { 0.0 })
                .collect();
            let xl: Vec<f64> = (0..n)
                .map(|i| if s.z[i * p + j] < -1.0 { 1.0 } else { 0.0 })
                .collect();
            u.push(abs_corr_with_y(&xu, &y_hi));
            l.push(abs_corr_with_y(&xl, &y_lo));
        }
        het[4] = u.iter().copied().fold(0.0, f64::max);
        het[5] = mean(&u);
        het[6] = l.iter().copied().fold(0.0, f64::max);
        het[7] = mean(&l);
    }
    let mc = target_simplex(&s.z, &s.y_scaled, n, p);
    let mut vec = [0.0_f32; BUILDING_BLOCK_WIDTH];
    for i in 0..16 {
        vec[i] = squash(high[i]) as f32;
    }
    for i in 0..8 {
        vec[16 + i] = squash(s.row_corr[i]) as f32;
    }
    for i in 0..16 {
        vec[24 + i] = contrast[i].clamp(-1.0, 1.0) as f32;
    }
    for i in 0..8 {
        vec[40 + i] = squash(het[i]) as f32;
    }
    for i in 0..20 {
        vec[48 + i] = mc[i] as f32;
    }
    let mut cols = vec![[0.0_f32; 7]; p];
    for (j, col) in cols.iter_mut().enumerate() {
        *col = [
            s.probes[j][0] as f32,
            s.probes[j][1] as f32,
            s.probes[j][2] as f32,
            s.probes[j][8] as f32,
            s.probes[j][9] as f32,
            s.probes[j][11] as f32,
            s.probes[j][12] as f32,
        ];
    }
    BlockStage {
        raw_gain: gain,
        vector: vec,
        column: cols,
    }
}

fn target_simplex(z: &[f64], y: &[f64], n: usize, p: usize) -> [f64; 20] {
    let mut out = [0.0; 20];
    let mut labels: Vec<f64> = y
        .iter()
        .map(|v| (v * 100_000.0).round() / 100_000.0)
        .collect();
    labels.sort_by(f64::total_cmp);
    labels.dedup_by(|a, b| *a == *b);
    let counts: Vec<usize> = labels
        .iter()
        .map(|label| y.iter().filter(|v| (**v - *label).abs() <= 1e-4).count())
        .collect();
    let k = labels.len();
    let discrete = 2 <= k && k <= 32.min((n / 3).max(2));
    if discrete {
        let priors: Vec<f64> = counts.iter().map(|c| *c as f64 / n as f64).collect();
        out[0] = (k.min(64) as f64).log2() / 6.0;
        out[1] =
            -priors.iter().map(|v| v * v.max(1e-12).ln()).sum::<f64>() / (k as f64).ln().max(1e-9);
        out[2] = priors.iter().copied().fold(f64::INFINITY, f64::min)
            / priors.iter().copied().fold(0.0, f64::max).max(1e-9);
        out[3] = gini(&priors);
        let mut centroids = vec![0.0; k * p];
        for (ci, label) in labels.iter().enumerate() {
            let rows: Vec<usize> = (0..n).filter(|&i| (y[i] - *label).abs() <= 1e-4).collect();
            for j in 0..p {
                centroids[ci * p + j] =
                    mean(&rows.iter().map(|&i| z[i * p + j]).collect::<Vec<_>>());
            }
        }
        let mut fisher = vec![0.0; p];
        for j in 0..p {
            let between = (0..k)
                .map(|ci| priors[ci] * centroids[ci * p + j].powi(2))
                .sum::<f64>();
            fisher[j] = squash(between / (1.0 - between).max(1e-4));
        }
        let mut fs = fisher.clone();
        fs.sort_by(|a, b| b.total_cmp(a));
        let fisher_count = fs.len().min(4);
        out[4..4 + fisher_count].copy_from_slice(&fs[..fisher_count]);
        out[8] = mean(&fisher);
        out[9] = gini(&fisher);
        if k >= 2 {
            let mut gram = vec![0.0; k * k];
            for a in 0..k {
                for b in 0..k {
                    gram[a * k + b] = (0..p)
                        .map(|j| {
                            priors[a].sqrt()
                                * centroids[a * p + j]
                                * priors[b].sqrt()
                                * centroids[b * p + j]
                        })
                        .sum();
                }
            }
            let (e, _) = feat_math::heads::jacobi_eigen(&mut gram, k);
            let mut singular: Vec<f64> = e.into_iter().map(|v| squash(v.max(0.0).sqrt())).collect();
            singular.sort_by(|a, b| b.total_cmp(a));
            let singular_count = singular.len().min(6);
            out[10..10 + singular_count].copy_from_slice(&singular[..singular_count]);
        }
    }
    let ym = mean(y);
    let ys = stddev(y).max(1e-9);
    let yz: Vec<f64> = y.iter().map(|v| (v - ym) / ys).collect();
    let mut sorted = yz.clone();
    sorted.sort_by(f64::total_cmp);
    let q10 = quantile(&sorted, 0.10);
    let q25 = quantile(&sorted, 0.25);
    let q75 = quantile(&sorted, 0.75);
    let q90 = quantile(&sorted, 0.90);
    out[16] = squash(mean(&yz.iter().map(|v| v.powi(3)).collect::<Vec<_>>()));
    out[17] = squash((mean(&yz.iter().map(|v| v.powi(4)).collect::<Vec<_>>()) - 3.0) / 4.0);
    out[18] = squash(mean(
        &yz.iter().map(|v| v.asinh().powi(2)).collect::<Vec<_>>(),
    ));
    out[19] = squash((q90 - q10) / (q75 - q25).max(1e-4) / 3.0);
    let _ = (z, n, p);
    out
}

fn report_descriptors(s: &ProbeStage, b: &BlockStage, n: usize, p: usize) -> ReportStage {
    let u: [f64; 16] = std::array::from_fn(|i| squash(b.raw_gain[i]) as f32 as f64);
    let probe = [
        u[4],
        u[0],
        u[3],
        u[6],
        u[1],
        u[7],
        u[10],
        u[9],
        u[11],
        u[7],
        u[5],
        u[13],
        u[11],
        0.01,
        u[0].max(u[4]),
        u[9],
        u[9].max(u[13]),
        u[9].max(u[13]) * 1.05,
        u[13] * 0.9,
        u[13] * 0.95,
        u[0].max(u[6]) * 1.05,
        u[4] * 0.85,
        u[9].max(u[13]) * 0.85,
    ];
    let mut raw_gains = [0.0; 23];
    let mut present = [false; 23];
    let mut se = [0.0; 23];
    let mut lam = [0.0; 23];
    for i in 0..23 {
        raw_gains[i] = (probe[i] * 100_000.0).round() / 100_000.0;
        present[i] = i <= 16 || i == 21;
        if present[i] {
            se[i] = ((1.0 / (n.max(16) as f64).sqrt()) * 100_000.0).round() / 100_000.0;
            lam[i] = 0.3;
        }
    }
    let mut gains = raw_gains;
    for i in 0..23 {
        if !present[i] {
            gains[i] = 0.0;
        }
    }
    let sg: [f64; 23] = std::array::from_fn(|i| squash(gains[i]));
    let mut mid = [0.0; 23];
    let active: Vec<usize> = (0..23).filter(|&i| present[i]).collect();
    for &i in &active {
        let less = active
            .iter()
            .filter(|&&j| raw_gains[j] < raw_gains[i])
            .count() as f64;
        let eq = active
            .iter()
            .filter(|&&j| raw_gains[j] == raw_gains[i])
            .count() as f64;
        mid[i] = (less + 0.5 * (eq - 1.0)) / (active.len() - 1).max(1) as f64;
    }
    let contrasts = report_contrasts(&raw_gains, &present);
    let mut raw_spec = vec![[0.0_f32; 9]; p];
    for (j, row) in raw_spec.iter_mut().enumerate() {
        let c = b.column[j];
        let vals = [c[0], c[3], c[4], c[0], c[1], c[5], c[2], c[6], c[3]];
        for k in 0..9 {
            if k != 7 {
                row[k] = vals[k];
            }
        }
    }
    let spec: Vec<[f32; 9]> = raw_spec
        .iter()
        .map(|row| std::array::from_fn(|k| squash(f64::from(row[k])) as f32))
        .collect();
    let mut lin = Vec::new();
    let mut macro_nl = Vec::new();
    let mut large_nl = Vec::new();
    let mut xis = Vec::new();
    let mut best = Vec::new();
    for row in &raw_spec {
        let v: Vec<f64> = row.iter().map(|x| squash(f64::from(*x))).collect();
        lin.push(v[0].max(0.0));
        macro_nl.push(v[3].max(v[4]).max(v[6]).max(0.0));
        large_nl.push(v[1].max(v[2]).max(v[5]).max(v[8]).max(0.0));
        xis.push(0.0);
        best.push(v.into_iter().fold(0.0_f64, f64::max));
    }
    let top1 = best.iter().copied().fold(0.0, f64::max);
    let mut best_sorted = best.clone();
    best_sorted.sort_by(|a, b| b.total_cmp(a));
    let mut summary = [0.0; 16];
    if p > 0 {
        summary = [
            lin.iter().copied().fold(0.0, f64::max),
            mean(&lin),
            gini(&lin),
            macro_nl.iter().copied().fold(0.0, f64::max),
            mean(&macro_nl),
            large_nl.iter().copied().fold(0.0, f64::max),
            mean(&large_nl),
            0.0,
            0.0,
            0.0,
            top1,
            mean(&best_sorted[..best_sorted.len().min(3)]),
            gini(&best),
            macro_nl
                .iter()
                .zip(&lin)
                .map(|(a, b)| a - b)
                .fold(f64::NEG_INFINITY, f64::max),
            large_nl
                .iter()
                .zip(&lin)
                .map(|(a, b)| a - b)
                .fold(f64::NEG_INFINITY, f64::max),
            lin.iter().map(|v| -*v).fold(f64::NEG_INFINITY, f64::max),
        ];
    }
    let mean_y = mean(&s.y_scaled);
    let std_y = stddev(&s.y_scaled);
    let shape = [
        0.0,
        (n.max(1) as f64).log10() / 6.0,
        (p.max(1) as f64).log2() / 12.0,
        ((p.max(1) as f64 / n.max(1) as f64).log10()) / 4.0,
    ];
    let macro_idx = [0, 1, 2, 3, 4, 5, 13, 14, 21];
    let larger_idx = [6, 7, 8, 9, 10, 11, 12, 15, 16];
    let fine_idx = [17, 18, 19, 20, 22];
    let macro_vals: Vec<f64> = macro_idx.iter().map(|&i| sg[i]).collect();
    let large_vals: Vec<f64> = larger_idx.iter().map(|&i| sg[i]).collect();
    let fine_vals: Vec<f64> = fine_idx.iter().map(|&i| sg[i]).collect();
    let group = [
        macro_vals.iter().copied().fold(0.0, f64::max),
        large_vals.iter().copied().fold(0.0, f64::max),
        0.0,
        mean(&macro_vals),
        mean(&large_vals),
        0.0,
        mean(&active.iter().map(|&i| squash(se[i])).collect::<Vec<_>>()),
        1.0,
        mean(&active.iter().map(|&i| lam[i]).collect::<Vec<_>>()),
    ];
    let mut v = [0.0_f32; REPORT_WIDTH];
    let mut at = 0;
    for x in sg {
        v[at] = x as f32;
        at += 1;
    }
    for &is_present in present.iter().take(23) {
        v[at] = if is_present { 1.0 } else { 0.0 };
        at += 1;
    }
    for x in mid {
        v[at] = x as f32;
        at += 1;
    }
    for x in contrasts {
        v[at] = x as f32;
        at += 1;
    }
    v[at] = squash(mean_y) as f32;
    v[at + 1] = squash(std_y) as f32;
    at += 2;
    for x in shape {
        v[at] = x as f32;
        at += 1;
    }
    for x in summary {
        v[at] = x as f32;
        at += 1;
    }
    for x in group {
        v[at] = x as f32;
        at += 1;
    }
    let _ = fine_vals;
    ReportStage {
        vector: v,
        gains: sg,
        ses: std::array::from_fn(|i| squash(se[i])),
        lambdas: lam,
        present,
        column_spectrum: spec,
    }
}

fn report_contrasts(g: &[f64; 23], present: &[bool; 23]) -> [f64; 16] {
    let max = |idx: &[usize]| idx.iter().map(|&i| g[i]).fold(f64::NEG_INFINITY, f64::max);
    let s = (0..23)
        .filter(|&i| present[i])
        .map(|i| g[i])
        .fold(0.0_f64, f64::max);
    if s < 0.01 {
        return [0.0; 16];
    }
    let sub = |a: f64, b: f64| ((a - b) / s).clamp(-1.0, 1.0);
    let _ = present;
    [
        s.clamp(-1.0, 1.0),
        (g[0] / s).clamp(-1.0, 1.0),
        (max(&[0, 1, 2, 3, 4, 5, 14]) / s).clamp(-1.0, 1.0),
        sub(max(&[1, 2, 4]), g[0]),
        sub(max(&[3, 5, 9]), g[0]),
        (g[14] / s).clamp(-1.0, 1.0),
        (g[13] / s).clamp(-1.0, 1.0),
        sub(g[0], g[21]),
        sub(g[7], g[6]),
        sub(g[8], max(&[0, 4])),
        sub(g[11], g[0]),
        sub(g[12], max(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14])),
        sub(max(&[15, 16]), max(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14])),
        sub(g[12], max(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14])),
        0.0,
        0.0,
    ]
}

fn table_descriptors(s: &ProbeStage, n: usize, p: usize) -> TableStage {
    let mut corr = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            corr[a * p + b] = if a == b {
                1.0
            } else {
                (0..n)
                    .map(|i| s.z_unc[i * p + a] * s.z_unc[i * p + b])
                    .sum::<f64>()
                    / n as f64
            };
        }
    }
    let mut max_corr = vec![0.0; p];
    let mut mean_corr = vec![0.0; p];
    for j in 0..p {
        let vals: Vec<f64> = (0..p)
            .filter(|&k| k != j)
            .map(|k| corr[j * p + k].abs())
            .collect();
        max_corr[j] = vals.iter().copied().fold(0.0, f64::max);
        mean_corr[j] = mean(&vals);
    }
    let mut eig = if p == 1 {
        vec![1.0]
    } else {
        let (e, _) = feat_math::heads::jacobi_eigen(&mut corr, p);
        e.into_iter().map(|v| v.max(0.0)).collect::<Vec<_>>()
    };
    eig.sort_by(|a, b| b.total_cmp(a));
    let es = eig.iter().sum::<f64>().max(1e-9);
    let shares: Vec<f64> = eig.iter().map(|v| v / es).collect();
    let eff = (es * es / (eig.iter().map(|v| v * v).sum::<f64>().max(1e-9))) / p as f64;
    let entropy = if p == 1 {
        1.0
    } else {
        -shares
            .iter()
            .filter(|v| **v > 1e-12)
            .map(|v| v * v.ln())
            .sum::<f64>()
            / (p as f64).ln().max(1e-9)
    };
    let skew: Vec<f64> = s.table_cols.iter().map(|v| v[0]).collect();
    let kurt: Vec<f64> = s.table_cols.iter().map(|v| v[1]).collect();
    let tail: Vec<f64> = s.table_cols.iter().map(|v| v[2]).collect();
    let mode: Vec<f64> = s.table_cols.iter().map(|v| v[3]).collect();
    let pear: Vec<f64> = s.table_cols.iter().map(|v| v[4]).collect();
    let spear: Vec<f64> = s.table_cols.iter().map(|v| v[5]).collect();
    let mut sr = pear.clone();
    sr.sort_by(|a, b| b.total_cmp(a));
    let mut ss = spear.clone();
    ss.sort_by(|a, b| b.total_cmp(a));
    let mut v = [0.0_f32; TABLE_WIDTH];
    v[0] = eff as f32;
    v[1] = entropy as f32;
    let ask = mean(&skew.iter().map(|x| x.abs()).collect::<Vec<_>>());
    v[2] = (ask / (1.0 + ask)) as f32;
    v[3] = mean(&mode) as f32;
    v[4] = sr.first().copied().unwrap_or(0.0) as f32;
    v[5] = mean(&pear) as f32;
    v[6] = gini(&pear) as f32;
    v[7] = ((n.max(1) as f64).log10() / 6.0) as f32;
    v[8] = shares.first().copied().unwrap_or(1.0) as f32;
    v[9] = shares.iter().take(3).sum::<f64>() as f32;
    v[10] = shares.iter().take(5).sum::<f64>() as f32;
    v[11] = max_corr.iter().copied().fold(0.0, f64::max) as f32;
    v[12] = mean(&mean_corr) as f32;
    v[13] = mean(&tail) as f32;
    v[14] = ss.first().copied().unwrap_or(0.0) as f32;
    v[15] = spear
        .iter()
        .zip(&pear)
        .map(|(a, b)| a - b)
        .fold(0.0, f64::max) as f32;
    let mut col = vec![[0.0_f32; 8]; p];
    for j in 0..p {
        col[j] = [
            squash(skew[j]) as f32,
            squash(kurt[j]) as f32,
            tail[j] as f32,
            mode[j] as f32,
            max_corr[j] as f32,
            pear[j] as f32,
            spear[j] as f32,
            0.0,
        ];
    }
    TableStage {
        vector: v,
        column: col,
    }
}

fn inter_descriptors(
    r: &ReportStage,
    _t: &TableStage,
    b: &BlockStage,
    p: usize,
) -> [f32; INTERACTION_WIDTH] {
    let g = r.gains;
    let m = r.present;
    let bb: [f64; 16] = std::array::from_fn(|i| squash(if i >= 14 { 0.0 } else { b.raw_gain[i] }));
    let fam: [Vec<f64>; 5] = [
        [g[0], g[1], g[2], g[4], bb[0], bb[1], bb[2], bb[3], bb[4]].to_vec(),
        [g[3], g[5], g[9], bb[6], bb[7], bb[13]].to_vec(),
        [g[6], g[7], g[15], g[16], g[17], bb[9], bb[10]].to_vec(),
        [g[8], g[12], bb[11], bb[12], 0.0].to_vec(),
        [g[10], g[11], g[18], g[19], g[20], bb[5], bb[15]].to_vec(),
    ];
    let mut active: Vec<f64> = g
        .iter()
        .enumerate()
        .filter(|(i, _)| m[*i])
        .map(|(_, value)| *value)
        .chain(bb.iter().copied().filter(|v| *v > 0.0))
        .filter(|v| v.is_finite())
        .collect();
    let smax = active.iter().copied().fold(0.02, f64::max);
    let mut out = [0.0_f32; INTERACTION_WIDTH];
    for fi in 0..5 {
        let vals: Vec<f64> = fam[fi]
            .iter()
            .copied()
            .filter(|v| v.is_finite() && v.abs() > 1e-9)
            .collect();
        if !vals.is_empty() {
            let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sd = stddev(&vals);
            out[fi * 4] = mx.clamp(-1.0, 1.0) as f32;
            out[fi * 4 + 1] = mean(&vals).clamp(-1.0, 1.0) as f32;
            out[fi * 4 + 2] = (sd / smax).clamp(0.0, 1.0) as f32;
            out[fi * 4 + 3] = (mx / smax).clamp(-1.0, 1.0) as f32;
        }
    }
    let lin = g[0].max(bb[4] * 0.5).max(0.0);
    let best1 = g[14].max(lin).max(0.01);
    let tail = g[1].max(g[2]).max(bb[0]).max(bb[3]);
    let sharp = g[7].max(g[15]).max(bb[9]);
    let smooth = g[4].max(bb[1]).max(bb[2]).max(bb[6]);
    let pair = g[11].max(bb[13]).max(bb[14]).max(bb[15]);
    let row = g[10].max(bb[5]);
    let per = g[8].max(bb[11]).max(bb[12]);
    let dlin = if m[0] && m[21] { g[0] - g[21] } else { 0.0 };
    let entropy = if active.len() > 1 {
        let max = active.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = active
            .iter()
            .map(|v| (8.0 * v / smax - max).exp())
            .collect();
        let sum = ex.iter().sum::<f64>().max(1e-12);
        -ex.iter()
            .map(|v| {
                let q = v / sum;
                q * q.max(1e-12).ln()
            })
            .sum::<f64>()
            / (active.len() as f64).ln().max(1e-9)
    } else {
        0.0
    };
    let ratios = [
        (tail - lin) / smax,
        (sharp - smooth) / smax,
        (g[16] - g[15]) / smax,
        0.0,
        (pair - best1) / smax,
        (row - best1) / smax,
        squash(row / (lin + 0.02)),
        (per - out[0] as f64) / smax,
        0.0,
        0.0,
        dlin / smax,
        entropy,
    ];
    for i in 0..12 {
        out[20 + i] = if i == 6 {
            ratios[i] as f32
        } else {
            ratios[i].clamp(-1.0, 1.0) as f32
        };
    }
    let mut ga = Vec::new();
    let mut sea = Vec::new();
    let mut la = Vec::new();
    for i in 0..23 {
        if m[i] {
            ga.push(g[i]);
            sea.push(r.ses[i].max(1e-3));
            la.push(r.lambdas[i]);
        }
    }
    if !ga.is_empty() {
        let weights: Vec<f64> = sea.iter().map(|v| 1.0 / (v * v + 1e-4)).collect();
        let ts: Vec<f64> = ga.iter().zip(&sea).map(|(a, b)| a / b).collect();
        out[32] = (weights.iter().zip(&ga).map(|(w, g)| w * g).sum::<f64>()
            / weights.iter().sum::<f64>().max(1e-9))
        .clamp(-1.0, 1.0) as f32;
        out[33] = squash(ts.iter().copied().fold(0.0, f64::max) / 10.0) as f32;
        out[34] = squash(mean(&ts) / 10.0) as f32;
        out[35] = squash(stddev(&ts) / 10.0) as f32;
        out[36] = ts.iter().filter(|v| **v > 2.0).count() as f32 / ts.len() as f32;
        out[37] = mean(&la).clamp(0.0, 1.0) as f32;
        out[38] = stddev(&la).clamp(0.0, 1.0) as f32;
        out[39] = corr(&ga, &la) as f32;
    }
    if p > 0 {
        let mut lin_c = Vec::new();
        let mut step = Vec::new();
        let mut xi = Vec::new();
        let mut asinh = Vec::new();
        let mut nonlin = Vec::new();
        for j in 0..p {
            let cs = r.column_spectrum[j];
            let cb = b.column[j];
            lin_c.push(f64::from(cs[0]));
            step.push(f64::from(cb[3].max(cs[1])));
            xi.push(f64::from(cb[6]));
            asinh.push(f64::from(cb[0]));
            nonlin.push(cb[1..].iter().copied().fold(0.0_f32, f32::max) as f64);
        }
        out[40] = corr(&lin_c, &step) as f32;
        out[41] = corr(&lin_c, &xi) as f32;
        out[42] = corr(&lin_c, &asinh) as f32;
        let mut orth: Vec<f64> = nonlin
            .iter()
            .zip(&lin_c)
            .map(|(a, b)| (a - b).max(0.0))
            .collect();
        orth.sort_by(|a, b| b.total_cmp(a));
        out[43] = squash(orth[0]) as f32;
        out[44] = squash(mean(&orth[..orth.len().min(3)])) as f32;
        let mut ent = 0.0;
        for j in 0..p {
            let vals = b.column[j]
                .iter()
                .map(|v| f64::from(*v).max(1e-6))
                .collect::<Vec<_>>();
            let sm = vals.iter().sum::<f64>();
            ent += -vals
                .iter()
                .map(|v| {
                    let q = v / sm;
                    q * q.ln()
                })
                .sum::<f64>()
                / (7.0_f64).ln();
        }
        out[45] = (ent / p as f64).clamp(0.0, 1.0) as f32;
        out[46] = squash(gini(&nonlin) / (gini(&lin_c) + 0.05)) as f32;
        out[47] = squash(
            f64::from(b.vector[5])
                / (nonlin
                    .iter()
                    .copied()
                    .fold(0.02, f64::max)
                    .max(lin_c.iter().copied().fold(0.0, f64::max))),
        ) as f32;
    }
    active.clear();
    out
}

#[cfg(test)]
mod tests {
    use super::{
        assemble_vector, build_high_full_table, compare_sampled_to_full,
        fit_transform_and_extract_bundle_high, pad_ranker_columns, summarize_latency,
        AgreementError, AgreementSummary, LandmarkOutput, BUILDING_BLOCK_WIDTH, INTERACTION_WIDTH,
        RANKER_COLUMN_COUNT, RANKER_COLUMN_WIDTH, RAW_COLUMN_WIDTH, REPORT_WIDTH, TABLE_WIDTH,
        VECTOR_WIDTH,
    };

    #[test]
    #[ignore = "requires an externally staged, hash-pinned sampled/full recipe capture"]
    fn report_external_sampled_vs_full_capture() {
        use std::{env, fs};

        let capture_path = env::var("JOPE_RNK05_CAPTURE_JSON").unwrap();
        let report_path = env::var("JOPE_RNK05_REPORT_JSON").unwrap();
        let capture: serde_json::Value = serde_json::from_slice(&fs::read(capture_path).unwrap())
            .expect("decode sampled/full recipe capture");

        let decode_output = |name: &str| {
            let output = &capture["outputs"][name];
            let vector: Vec<f32> = serde_json::from_value(output["vector256"].clone()).unwrap();
            assert_eq!(vector.len(), VECTOR_WIDTH, "{name}: vector width");
            let columns: Vec<[f32; RAW_COLUMN_WIDTH]> =
                serde_json::from_value(output["col_full"].clone()).unwrap();
            assert!(
                !columns.is_empty(),
                "{name}: raw column output must be present"
            );

            let report: [f32; REPORT_WIDTH] = vector[..REPORT_WIDTH].try_into().unwrap();
            let table: [f32; TABLE_WIDTH] = vector[REPORT_WIDTH..REPORT_WIDTH + TABLE_WIDTH]
                .try_into()
                .unwrap();
            let interactions: [f32; INTERACTION_WIDTH] = vector
                [REPORT_WIDTH + TABLE_WIDTH..REPORT_WIDTH + TABLE_WIDTH + INTERACTION_WIDTH]
                .try_into()
                .unwrap();
            let building_blocks: [f32; BUILDING_BLOCK_WIDTH] = vector
                [REPORT_WIDTH + TABLE_WIDTH + INTERACTION_WIDTH..]
                .try_into()
                .unwrap();
            let rows = output["rows"].as_u64().unwrap() as usize;

            (
                LandmarkOutput::from_descriptor_parts(
                    &report,
                    &table,
                    &interactions,
                    &building_blocks,
                    &columns,
                ),
                rows,
            )
        };

        let (sampled, sampled_rows) = decode_output("sampled");
        let (full, full_rows) = decode_output("full");
        let sampled_recipe_elapsed_ms = capture["outputs"]["sampled"]["elapsed_ns"]
            .as_u64()
            .unwrap() as f64
            / 1_000_000.0;
        let full_recipe_elapsed_ms =
            capture["outputs"]["full"]["elapsed_ns"].as_u64().unwrap() as f64 / 1_000_000.0;
        assert!(sampled_rows <= 1_024);
        assert!(full_rows > sampled_rows);
        let summary = compare_sampled_to_full(&sampled, &full, sampled_rows, full_rows).unwrap();
        let report = serde_json::json!({
            "schema": "rnk05.sampled-full-agreement/v1",
            "dataset_id": capture["dataset_id"],
            "training_sha256": capture["training_sha256"],
            "sample_row_indices_sha256": capture["sample_row_indices_sha256"],
            "sample_method": capture["sample_method"],
            "sample_seed": capture["sample_seed"],
            "sampled_recipe_elapsed_ms": sampled_recipe_elapsed_ms,
            "full_recipe_elapsed_ms": full_recipe_elapsed_ms,
            "sampled_rows": summary.sampled_rows,
            "full_rows": summary.full_rows,
            "values_compared": summary.values_compared,
            "exact_matches": summary.exact_matches,
            "max_abs_error": summary.max_abs_error,
            "mean_abs_error": summary.mean_abs_error,
        });
        fs::write(report_path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }

    #[test]
    fn vector_blocks_are_concatenated_in_training_order() {
        let report = std::array::from_fn(|i| i as f32);
        let table = std::array::from_fn(|i| (REPORT_WIDTH + i) as f32);
        let interactions = std::array::from_fn(|i| (REPORT_WIDTH + TABLE_WIDTH + i) as f32);
        let building_blocks =
            std::array::from_fn(|i| (REPORT_WIDTH + TABLE_WIDTH + INTERACTION_WIDTH + i) as f32);

        let vector = assemble_vector(&report, &table, &interactions, &building_blocks);

        assert_eq!(vector.len(), VECTOR_WIDTH);
        assert_eq!(vector[0], 0.0);
        assert_eq!(vector[REPORT_WIDTH - 1], (REPORT_WIDTH - 1) as f32);
        assert_eq!(vector[REPORT_WIDTH], REPORT_WIDTH as f32);
        assert_eq!(
            vector[REPORT_WIDTH + TABLE_WIDTH],
            (REPORT_WIDTH + TABLE_WIDTH) as f32
        );
        assert_eq!(vector[VECTOR_WIDTH - 1], (VECTOR_WIDTH - 1) as f32);
    }

    #[test]
    fn columns_keep_first_17_values_and_zero_pad_or_truncate_to_64() {
        let raw: Vec<[f32; RAW_COLUMN_WIDTH]> = (0..RANKER_COLUMN_COUNT + 1)
            .map(|row| std::array::from_fn(|column| (row * 100 + column) as f32))
            .collect();

        let columns = pad_ranker_columns(&raw);

        assert_eq!(columns.len(), RANKER_COLUMN_COUNT);
        assert_eq!(columns[0].len(), RANKER_COLUMN_WIDTH);
        assert_eq!(columns[0][0], 0.0);
        assert_eq!(columns[0][16], 16.0);
        assert_eq!(columns[63][0], 6300.0);
        assert_eq!(columns[63][16], 6316.0);

        let short = pad_ranker_columns(&raw[..1]);
        assert_eq!(short[0][16], 16.0);
        assert_eq!(short[1], [0.0; RANKER_COLUMN_WIDTH]);
        assert_eq!(short[63], [0.0; RANKER_COLUMN_WIDTH]);
    }

    #[test]
    fn non_finite_descriptor_values_are_zeroed() {
        let mut report = [1.0; REPORT_WIDTH];
        report[0] = f32::NAN;
        let mut table = [2.0; TABLE_WIDTH];
        table[TABLE_WIDTH - 1] = f32::INFINITY;
        let interactions = [3.0; INTERACTION_WIDTH];
        let building_blocks = [4.0; BUILDING_BLOCK_WIDTH];

        let vector = assemble_vector(&report, &table, &interactions, &building_blocks);
        assert_eq!(vector[0], 0.0);
        assert_eq!(vector[REPORT_WIDTH + TABLE_WIDTH - 1], 0.0);

        let mut raw = [[5.0; RAW_COLUMN_WIDTH]; 1];
        raw[0][16] = f32::NEG_INFINITY;
        let columns = pad_ranker_columns(&raw);
        assert_eq!(columns[0][16], 0.0);
        assert_eq!(columns[1], [0.0; RANKER_COLUMN_WIDTH]);
    }

    fn output_with_report_value(value: f32) -> LandmarkOutput {
        let mut report = [0.0; REPORT_WIDTH];
        report[0] = value;
        LandmarkOutput::from_descriptor_parts(
            &report,
            &[0.0; TABLE_WIDTH],
            &[0.0; INTERACTION_WIDTH],
            &[0.0; BUILDING_BLOCK_WIDTH],
            &[],
        )
    }

    #[test]
    fn table_agreement_reports_full_shape_and_absolute_error_summary() {
        let sampled = output_with_report_value(2.0);
        let full = output_with_report_value(3.0);

        let summary = compare_sampled_to_full(&sampled, &full, 12, 100).unwrap();

        assert_eq!(
            summary,
            AgreementSummary {
                sampled_rows: 12,
                full_rows: 100,
                values_compared: VECTOR_WIDTH + RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH,
                exact_matches: VECTOR_WIDTH + RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH - 1,
                max_abs_error: 1.0,
                mean_abs_error: 1.0
                    / (VECTOR_WIDTH + RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH) as f64,
            }
        );
    }

    #[test]
    fn table_agreement_rejects_invalid_fit_row_counts() {
        let output = output_with_report_value(0.0);

        assert_eq!(
            compare_sampled_to_full(&output, &output, 0, 1),
            Err(AgreementError::EmptyTable)
        );
        assert_eq!(
            compare_sampled_to_full(&output, &output, 1_025, 2_000),
            Err(AgreementError::SampleOverLimit)
        );
        assert_eq!(
            compare_sampled_to_full(&output, &output, 4, 3),
            Err(AgreementError::SampleExceedsFullTable)
        );
    }

    #[test]
    fn latency_summary_reports_median_and_range_in_milliseconds() {
        use std::time::Duration;

        assert_eq!(summarize_latency(&[]), None);
        assert_eq!(
            summarize_latency(&[
                Duration::from_millis(3),
                Duration::from_millis(1),
                Duration::from_millis(2),
            ]),
            Some(super::LatencySummary {
                samples: 3,
                min_ms: 1.0,
                median_ms: 2.0,
                max_ms: 3.0,
            })
        );
    }

    #[test]
    #[ignore = "requires externally staged, hash-pinned RES-07 CPU fixtures and bundle"]
    fn matches_res07_real_landmark_output_assembly() {
        use super::super::adapter::real_cpu_parity::Fixtures;

        // PROVISIONAL, WATCHER-SET; owner may override. This reuses the frozen
        // RES-07/RNK-12 CPU parity arithmetic, not an owner-approved RNK-05 rule.
        const ATOL: f64 = 1e-5;
        const RTOL: f64 = 1e-4;

        fn assert_matches_golden(context: &str, name: &str, actual: &[f32], expected: &[f32]) {
            assert_eq!(actual.len(), expected.len(), "{context}: {name} length");
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                let actual = f64::from(actual);
                let expected = f64::from(expected);
                assert!(
                    actual.is_finite() && expected.is_finite(),
                    "{context}: {name}[{index}] must be finite"
                );
                let tolerance = ATOL + RTOL * expected.abs();
                let error = (actual - expected).abs();
                assert!(
                    error <= tolerance,
                    "{context}: {name}[{index}] actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
                );
            }
        }

        let fixtures = Fixtures::open();
        assert_eq!(fixtures.contexts.len(), 50);

        for context in &fixtures.contexts {
            let name = format!("landmarks/{context}.safetensors");
            let fixture = fixtures.load(&name);
            let schema = &fixtures.index["files"][&name]["tensors"];
            let input_shape = schema["input.X"]["shape"].as_array().unwrap();
            let fit_rows = input_shape[0].as_u64().unwrap() as usize;
            let feature_count = input_shape[1].as_u64().unwrap() as usize;

            assert!((1..=1_024).contains(&fit_rows), "{context}: FIT row cap");
            assert_eq!(
                schema["input.row_ids"]["shape"][0].as_u64().unwrap() as usize,
                fit_rows,
                "{context}: source row IDs must cover the FIT table"
            );
            assert_eq!(
                schema["input.y"]["shape"][0].as_u64().unwrap() as usize,
                fit_rows,
                "{context}: target rows must match the FIT table"
            );
            assert_eq!(
                schema["col_full"]["shape"][0].as_u64().unwrap() as usize,
                feature_count,
                "{context}: raw column output count must match feature count"
            );

            let vector = fixture.f32("v256");
            assert_eq!(vector.len(), VECTOR_WIDTH, "{context}: v256 width");
            let report: [f32; REPORT_WIDTH] = vector[..REPORT_WIDTH].try_into().unwrap();
            let table: [f32; TABLE_WIDTH] = vector[REPORT_WIDTH..REPORT_WIDTH + TABLE_WIDTH]
                .try_into()
                .unwrap();
            let interaction_start = REPORT_WIDTH + TABLE_WIDTH;
            let interactions: [f32; INTERACTION_WIDTH] = vector
                [interaction_start..interaction_start + INTERACTION_WIDTH]
                .try_into()
                .unwrap();
            let building_block_start = interaction_start + INTERACTION_WIDTH;
            let building_blocks: [f32; BUILDING_BLOCK_WIDTH] =
                vector[building_block_start..].try_into().unwrap();

            let raw_column_values = fixture.f32("col_full");
            assert_eq!(
                raw_column_values.len() % RAW_COLUMN_WIDTH,
                0,
                "{context}: raw column width"
            );
            let raw_columns = raw_column_values
                .chunks_exact(RAW_COLUMN_WIDTH)
                .map(|row| row.try_into().unwrap())
                .collect::<Vec<[f32; RAW_COLUMN_WIDTH]>>();

            let actual = LandmarkOutput::from_descriptor_parts(
                &report,
                &table,
                &interactions,
                &building_blocks,
                &raw_columns,
            );
            assert_matches_golden(context, "v256", actual.vector().as_slice(), &vector);

            let expected_columns = fixture.f32("col");
            let actual_columns = actual
                .columns()
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            assert_matches_golden(context, "col", &actual_columns, &expected_columns);
        }
    }

    #[test]
    #[ignore = "requires hash-pinned RES-07 inputs and captured full-recipe outputs"]
    fn matches_res07_full_recipe_outputs_and_measures_rust_latency() {
        use std::time::Instant;

        const ATOL: f64 = 1e-5;
        const RTOL: f64 = 1e-4;
        fn check(context: &str, name: &str, got: &[f32], expected: &[f32]) {
            assert_eq!(got.len(), expected.len(), "{context}: {name} shape");
            for (i, (&a, &g)) in got.iter().zip(expected).enumerate() {
                assert!(
                    a.is_finite() && g.is_finite(),
                    "{context}: {name}[{i}] nonfinite"
                );
                let error = (f64::from(a) - f64::from(g)).abs();
                let tolerance = ATOL + RTOL * f64::from(g).abs();
                assert!(error <= tolerance, "{context}: {name}[{i}] actual={a} golden={g} error={error} tolerance={tolerance}");
            }
        }

        let fixtures = super::super::adapter::real_cpu_parity::Fixtures::open();
        assert_eq!(fixtures.contexts.len(), 50);
        let mut elapsed = Vec::with_capacity(fixtures.contexts.len());
        let mut v_compared = 0;
        let mut raw_compared = 0;
        let mut ranker_compared = 0;
        let mut max_error = 0.0_f64;
        for context in &fixtures.contexts {
            let name = format!("landmarks/{context}.safetensors");
            let fixture = fixtures.load(&name);
            let schema = &fixtures.index["files"][&name]["tensors"];
            let shape = schema["input.X"]["shape"].as_array().unwrap();
            let rows = shape[0].as_u64().unwrap() as usize;
            let features = shape[1].as_u64().unwrap() as usize;
            assert!((1..=1_024).contains(&rows), "{context}: FIT row cap");
            let x = fixture.f32("input.X");
            let y = fixture.f32("input.y");
            let start = Instant::now();
            let got = fit_transform_and_extract_bundle_high(&x, rows, features, &y)
                .unwrap_or_else(|error| panic!("{context}: full Rust recipe failed: {error}"));
            elapsed.push(start.elapsed());

            let golden_vector = fixture.f32("v256");
            let raw_golden = fixture.f32("col_full");
            let raw_actual: Vec<f32> = got.raw_columns().iter().flatten().copied().collect();
            let ranker_golden = fixture.f32("col");
            let ranker_actual: Vec<f32> = got.columns().iter().flatten().copied().collect();
            check(context, "v256", got.vector(), &golden_vector);
            check(context, "col_full", &raw_actual, &raw_golden);
            check(context, "col", &ranker_actual, &ranker_golden);
            v_compared += golden_vector.len();
            raw_compared += raw_golden.len();
            ranker_compared += ranker_golden.len();
            for (a, g) in got
                .vector()
                .iter()
                .zip(&golden_vector)
                .chain(raw_actual.iter().zip(&raw_golden))
                .chain(ranker_actual.iter().zip(&ranker_golden))
            {
                max_error = max_error.max((f64::from(*a) - f64::from(*g)).abs());
            }
        }
        let latency = summarize_latency(&elapsed).unwrap();
        eprintln!("RNK-05 Rust full-recipe parity: contexts={} v256={} col_full={} col={} misses=0 max_abs_error={max_error:.9} landmarks.ms samples={} min={:.6} median={:.6} max={:.6}", fixtures.contexts.len(), v_compared, raw_compared, ranker_compared, latency.samples, latency.min_ms, latency.median_ms, latency.max_ms);
    }

    #[test]
    #[ignore = "requires hash-pinned RES-07 sampled/full training capture"]
    fn reports_res07_sampled_full_rust_agreement() {
        use sha2::{Digest, Sha256};
        use std::{env, fs, time::Instant};

        const ATOL: f64 = 1e-5;
        const RTOL: f64 = 1e-4;
        let train_path = env::var("JOPE_RNK05_TRAIN_CSV").unwrap();
        let indices_path = env::var("JOPE_RNK05_SAMPLE_INDICES").unwrap();
        let capture_path = env::var("JOPE_RNK05_CAPTURE_JSON").unwrap();
        let report_path = env::var("JOPE_RNK05_REPORT_JSON").unwrap();
        let train_bytes = fs::read(train_path).unwrap();
        let indices_bytes = fs::read(indices_path).unwrap();
        let capture: serde_json::Value =
            serde_json::from_slice(&fs::read(capture_path).unwrap()).unwrap();
        let train_sha = format!("{:x}", Sha256::digest(&train_bytes));
        let indices_sha = format!("{:x}", Sha256::digest(&indices_bytes));
        assert_eq!(train_sha, capture["train_file_sha256"]);
        assert_eq!(indices_sha, capture["sample_row_indices_sha256"]);
        assert_eq!(capture["schema"], "rnk05.sampled-full-recipe-capture/v1");

        let rows: Vec<Vec<f32>> = String::from_utf8(train_bytes)
            .unwrap()
            .lines()
            .map(|line| {
                line.split(',')
                    .map(|value| value.parse::<f32>().unwrap())
                    .collect()
            })
            .collect();
        let full_rows = rows.len();
        let features = rows[0].len() - 1;
        assert_eq!(
            full_rows,
            capture["training_rows"].as_u64().unwrap() as usize
        );
        assert_eq!(
            features + 1,
            capture["training_columns_including_target"]
                .as_u64()
                .unwrap() as usize
        );
        assert_eq!(indices_bytes.len() % 4, 0);
        let indices: Vec<usize> = indices_bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) as usize)
            .collect();
        assert_eq!(
            indices.len(),
            capture["sample_rows"].as_u64().unwrap() as usize
        );
        assert!(indices.iter().all(|&index| index < full_rows));
        let flatten_x = |selected: &[usize]| {
            selected
                .iter()
                .flat_map(|&i| rows[i][..features].iter().copied())
                .collect::<Vec<_>>()
        };
        let flatten_y = |selected: &[usize]| {
            selected
                .iter()
                .map(|&i| rows[i][features])
                .collect::<Vec<_>>()
        };
        let all_indices: Vec<usize> = (0..full_rows).collect();
        let sampled_x = flatten_x(&indices);
        let sampled_y = flatten_y(&indices);
        let full_x = flatten_x(&all_indices);
        let full_y = flatten_y(&all_indices);

        let started = Instant::now();
        let sampled =
            fit_transform_and_extract_bundle_high(&sampled_x, indices.len(), features, &sampled_y)
                .unwrap();
        let sampled_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = Instant::now();
        let full = build_high_full_table(&full_x, full_rows, features, &full_y).unwrap();
        let full_ms = started.elapsed().as_secs_f64() * 1_000.0;
        assert!(indices.len() <= 1_024 && full_rows > indices.len());

        let check_golden = |label: &str, got: &[f32], expected: &[f32]| {
            assert_eq!(got.len(), expected.len(), "{label} shape");
            for (index, (&actual, &golden)) in got.iter().zip(expected).enumerate() {
                assert!(
                    actual.is_finite() && golden.is_finite(),
                    "{label}[{index}] nonfinite"
                );
                let error = (f64::from(actual) - f64::from(golden)).abs();
                assert!(
                    error <= ATOL + RTOL * f64::from(golden).abs(),
                    "{label}[{index}] actual={actual} golden={golden} error={error}"
                );
            }
        };
        for (name, output) in [("sampled", &sampled), ("full", &full)] {
            let expected_vector: Vec<f32> =
                serde_json::from_value(capture["outputs"][name]["vector256"].clone()).unwrap();
            let expected_columns: Vec<Vec<f32>> =
                serde_json::from_value(capture["outputs"][name]["col_full"].clone()).unwrap();
            let expected_flat: Vec<f32> = expected_columns.into_iter().flatten().collect();
            let actual_columns: Vec<f32> = output.raw_columns().iter().flatten().copied().collect();
            check_golden(&format!("{name}.v256"), output.vector(), &expected_vector);
            check_golden(&format!("{name}.col_full"), &actual_columns, &expected_flat);
        }
        let agreement = compare_sampled_to_full(&sampled, &full, indices.len(), full_rows).unwrap();
        let report = serde_json::json!({
            "schema": "rnk05.sampled-full-rust-agreement/v1",
            "dataset_id": capture["dataset_id"],
            "dataset_canonical_sha256": capture["dataset_canonical_sha256"],
            "training_sha256": train_sha,
            "sample_row_indices_sha256": indices_sha,
            "sample_method": capture["sample_method"],
            "sample_seed": capture["sample_seed"],
            "sampled_rows": agreement.sampled_rows,
            "full_rows": agreement.full_rows,
            "values_compared": agreement.values_compared,
            "exact_matches": agreement.exact_matches,
            "max_abs_error": agreement.max_abs_error,
            "mean_abs_error": agreement.mean_abs_error,
            "sampled_landmarks_ms": sampled_ms,
            "full_landmarks_ms": full_ms,
            "golden_tolerance": "finite; abs(actual-golden) <= 1e-5 + 1e-4*abs(golden)",
            "golden_misses": 0
        });
        fs::write(report_path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}
