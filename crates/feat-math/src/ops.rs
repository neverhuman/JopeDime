//! Battle-dialect feature transforms: fit on train, apply anywhere.
//!
//! Every operator here must have an exact twin in the emitted Python and Rust
//! artifacts (battle-transpile snippets). Keep loop order and draw order
//! IDENTICAL across the three implementations — parity is bit-for-bit.
//!
//! Conventions:
//! - input rows are `Vec<Vec<f64>>`, all rows the same width;
//! - every op honors the `append` param (default false): append=true emits
//!   `[input ‖ output]` (nm_ semantics), append=false replaces;
//! - after every stage the pipeline sanitizes non-finite values to 0.0 and
//!   truncates to `MAX_WIDTH` columns;
//! - seeded ops draw from `Pcg32::for_stage(seed, in_cols, n_out)` with a
//!   documented draw order (weights row-major first, then offsets).

use crate::dsl::Stage;
use crate::math_expr::MathExpr;
use crate::rng::Pcg32;
use crate::Rows;
use anyhow::{bail, Result};

const PI2: f64 = std::f64::consts::PI * 2.0;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum FittedOp {
    Standardize {
        means: Vec<f64>,
        scales: Vec<f64>,
        append: bool,
    },
    RobustScale {
        medians: Vec<f64>,
        scales: Vec<f64>,
        append: bool,
    },
    Rank {
        sorted_cols: Vec<Vec<f64>>,
        append: bool,
    },
    Blom {
        sorted_cols: Vec<Vec<f64>>,
        append: bool,
    },
    QuantileBins {
        edges: Vec<Vec<f64>>,
        bins: usize,
        append: bool,
    },
    BinOnehot {
        edges: Vec<Vec<f64>>,
        bins: usize,
        append: bool,
    },
    Hinge {
        knots: Vec<Vec<f64>>,
        append: bool,
    },
    WinsorClip {
        lo: Vec<f64>,
        hi: Vec<f64>,
        append: bool,
    },
    Cheb {
        degree: usize,
        append: bool,
    },
    PolyInteract {
        pairs: Vec<(usize, usize)>,
        append: bool,
    },
    RffCos {
        w: Vec<Vec<f64>>,
        b: Vec<f64>,
        append: bool,
    },
    RffSin {
        w: Vec<Vec<f64>>,
        b: Vec<f64>,
        append: bool,
    },
    RandProj {
        w: Vec<Vec<f64>>,
        append: bool,
    },
    LandmarkDist {
        landmarks: Vec<Vec<f64>>,
        gamma: f64,
        append: bool,
    },
    ObliqueHinge {
        w: Vec<Vec<f64>>,
        t: Vec<f64>,
        append: bool,
    },
    RowStats {
        append: bool,
    },
    PairProd {
        pairs: Vec<(usize, usize)>,
        append: bool,
    },
    TargetBinEncode {
        edges: Vec<Vec<f64>>,
        enc: Vec<Vec<f64>>,
        append: bool,
    },
    Sigmoid {
        append: bool,
    },
    /// Per-column hard indicator `1.0 if v > t else 0.0` at parameter threshold
    /// `t` (default 0.5). Stateless — the emitter renders it as an explicit
    /// conditional (mirrored bit-for-bit), never a `clip(...*1e15,...)` trick.
    Step {
        t: f64,
        append: bool,
    },
    NormRow {
        append: bool,
    },
    Polar {
        append: bool,
    },
    Lag {
        k: usize,
        append: bool,
    },
    Roll {
        k: usize,
        append: bool,
    },
    FftMag {
        k: usize,
        append: bool,
    },
    RowSpectral {
        append: bool,
    },
    RowDiffStats {
        append: bool,
    },
    SvdProject {
        means: Vec<f64>,
        components: Vec<Vec<f64>>,
        append: bool,
    },
    Nystrom {
        landmarks: Vec<Vec<f64>>,
        gamma: f64,
        append: bool,
    },
    MapExpr {
        source: String,
        expr: MathExpr,
        append: bool,
    },
    DeriveExpr {
        source: String,
        expr: MathExpr,
        append: bool,
    },
    /// Signed shifted-log compression asinh(v / c): smooth, monotone, and
    /// linear near zero — a heavy-tail compressor that (unlike rank/blom)
    /// extrapolates beyond the train range. Stateless.
    Symlog {
        c: f64,
        append: bool,
    },
    /// Top-k column screening by |pearson(x_j, y)| (target-aware) or train
    /// variance; kept indices stored ascending so downstream column order is
    /// preserved. The explicit width pressure-valve for wide datasets.
    SelectCorr {
        indices: Vec<usize>,
        append: bool,
    },
    /// Zero-as-missing imputation: exact-0.0 cells (the loader's missing
    /// sentinel, also common integer-coded missingness) are replaced by the
    /// fitted per-column fill; `indicator_cols` lists the columns that had at
    /// least one zero in train and get a trailing 0/1 missingness indicator.
    Impute {
        fill: Vec<f64>,
        indicator_cols: Vec<usize>,
        append: bool,
    },
    /// Target-guided regularized ratios x_a * x_b / (x_b^2 + eps): the
    /// screened ordered pairs with the strongest |pearson(ratio, y)|, where
    /// eps = max(1e-12, (0.01 * train std of x_b)^2) keeps every ratio finite
    /// and smooth. `eps[i]` is the denominator guard for `pairs[i]`.
    PairRatio {
        pairs: Vec<(usize, usize)>,
        eps: Vec<f64>,
        append: bool,
    },
    /// Natural cubic spline basis at train quantile knots (ESL §5.2.1): per
    /// column emits [x, N_2(x), .., N_{K-1}(x)], linear beyond the boundary
    /// knots; a near-constant column falls back to its raw value only.
    Spline {
        knots: Vec<Vec<f64>>,
        append: bool,
    },
}

// ---------- shared fitted-statistics helpers (mirrored in artifacts) ----------

pub fn sorted_col(x: &Rows, j: usize) -> Vec<f64> {
    let mut v: Vec<f64> = x.iter().map(|r| r[j]).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v
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
    let lb = lower_bound(s, v) as f64;
    let ub = upper_bound(s, v) as f64;
    (lb + ub) / (2.0 * s.len() as f64)
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
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - p_low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

fn quantile_edges(x: &Rows, bins: usize) -> Vec<Vec<f64>> {
    let cols = x[0].len();
    (0..cols)
        .map(|j| {
            let s = sorted_col(x, j);
            (1..bins)
                .map(|b| quantile_sorted(&s, b as f64 / bins as f64))
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

fn get_append(stage: &Stage) -> bool {
    matches!(
        stage.params.get("append"),
        Some(crate::dsl::ParamValue::Bool(true))
    ) || stage.get_usize("append", 0) == 1
}

// ---------------------------------- fit ----------------------------------

pub fn fit_op(stage: &Stage, x: &Rows, y: &[f64]) -> Result<FittedOp> {
    if x.is_empty() || x[0].is_empty() {
        bail!("empty input to stage {}", stage.op);
    }
    let cols = x[0].len();
    let n = x.len();
    let append = get_append(stage);
    let op = match stage.op.as_str() {
        "standardize" | "norm_features" => {
            let mut means = vec![0.0; cols];
            let mut scales = vec![1.0; cols];
            for j in 0..cols {
                let mut s = 0.0;
                for r in x {
                    s += r[j];
                }
                let m = s / n as f64;
                let mut v = 0.0;
                for r in x {
                    let d = r[j] - m;
                    v += d * d;
                }
                means[j] = m;
                scales[j] = (v / n as f64).sqrt().max(1e-9);
            }
            FittedOp::Standardize {
                means,
                scales,
                append,
            }
        }
        "robust_scale" => {
            let mut medians = vec![0.0; cols];
            let mut scales = vec![1.0; cols];
            for j in 0..cols {
                let s = sorted_col(x, j);
                medians[j] = quantile_sorted(&s, 0.5);
                let iqr = quantile_sorted(&s, 0.75) - quantile_sorted(&s, 0.25);
                scales[j] = iqr.max(1e-9);
            }
            FittedOp::RobustScale {
                medians,
                scales,
                append,
            }
        }
        "rank" => FittedOp::Rank {
            sorted_cols: (0..cols).map(|j| sorted_col(x, j)).collect(),
            append,
        },
        "blom" => FittedOp::Blom {
            sorted_cols: (0..cols).map(|j| sorted_col(x, j)).collect(),
            append,
        },
        "quantile_bins" => {
            let bins = stage.get_usize("bins", 8).clamp(2, 64);
            FittedOp::QuantileBins {
                edges: quantile_edges(x, bins),
                bins,
                append,
            }
        }
        "bin_onehot" => {
            let bins = stage.get_usize("bins", 6).clamp(2, 32);
            FittedOp::BinOnehot {
                edges: quantile_edges(x, bins),
                bins,
                append,
            }
        }
        "hinge" => {
            let k = stage.get_usize("knots", 4).clamp(1, 16);
            let knots = (0..cols)
                .map(|j| {
                    let s = sorted_col(x, j);
                    (1..=k)
                        .map(|i| quantile_sorted(&s, i as f64 / (k + 1) as f64))
                        .collect()
                })
                .collect();
            FittedOp::Hinge { knots, append }
        }
        "winsor_clip" | "winsor_moments" => {
            let q = stage.get_f64("q", 0.05).clamp(0.0, 0.25);
            let mut lo = vec![0.0; cols];
            let mut hi = vec![0.0; cols];
            for j in 0..cols {
                let s = sorted_col(x, j);
                lo[j] = quantile_sorted(&s, q);
                hi[j] = quantile_sorted(&s, 1.0 - q).max(lo[j]);
            }
            FittedOp::WinsorClip { lo, hi, append }
        }
        "cheb" => FittedOp::Cheb {
            degree: stage.get_usize("degree", 4).clamp(1, 8),
            append,
        },
        "poly_interact" => {
            let mut pairs = Vec::new();
            for i in 0..cols {
                pairs.push((i, i));
            }
            'outer: for i in 0..cols {
                for j in (i + 1)..cols {
                    if pairs.len() >= crate::MAX_WIDTH {
                        break 'outer;
                    }
                    pairs.push((i, j));
                }
            }
            FittedOp::PolyInteract { pairs, append }
        }
        "rff_cos" | "rff_sin" => {
            let nf = stage.get_usize("n", 32).clamp(1, 256);
            let gamma = stage.get_f64("gamma", 1.0).max(1e-6);
            let seed = stage.get_u64("seed", 7);
            let mut rng = Pcg32::for_stage(seed, cols, nf);
            let scale = (2.0 * gamma).sqrt();
            let mut w = Vec::with_capacity(nf);
            for _ in 0..nf {
                let mut row = Vec::with_capacity(cols);
                for _ in 0..cols {
                    row.push(rng.gaussian() * scale);
                }
                w.push(row);
            }
            let b: Vec<f64> = (0..nf).map(|_| rng.range_f64(0.0, PI2)).collect();
            if stage.op == "rff_cos" {
                FittedOp::RffCos { w, b, append }
            } else {
                FittedOp::RffSin { w, b, append }
            }
        }
        "rand_proj" => {
            let k = stage.get_usize("k", 16).clamp(1, 256);
            let seed = stage.get_u64("seed", 7);
            let mut rng = Pcg32::for_stage(seed, cols, k);
            let mut w = Vec::with_capacity(k);
            for _ in 0..k {
                let mut row = Vec::with_capacity(cols);
                for _ in 0..cols {
                    row.push(rng.gaussian());
                }
                w.push(row);
            }
            FittedOp::RandProj { w, append }
        }
        "landmark_dist" => {
            let k = stage.get_usize("k", 16).clamp(1, 128);
            let gamma = stage.get_f64("gamma", 1.0).max(1e-6);
            let seed = stage.get_u64("seed", 7);
            let mut rng = Pcg32::for_stage(seed, cols, k);
            let mut landmarks = Vec::with_capacity(k);
            for _ in 0..k {
                let mut row = Vec::with_capacity(cols);
                for _ in 0..cols {
                    row.push(rng.range_f64(0.0, 1.0));
                }
                landmarks.push(row);
            }
            FittedOp::LandmarkDist {
                landmarks,
                gamma,
                append,
            }
        }
        "oblique_hinge" => {
            let nf = stage.get_usize("n", 32).clamp(1, 256);
            let seed = stage.get_u64("seed", 7);
            let mut rng = Pcg32::for_stage(seed, cols, nf);
            let inv = 1.0 / (cols as f64).sqrt();
            let mut w = Vec::with_capacity(nf);
            for _ in 0..nf {
                let mut row = Vec::with_capacity(cols);
                for _ in 0..cols {
                    row.push(rng.gaussian() * inv);
                }
                w.push(row);
            }
            // Fitted thresholds: median of each projection over train rows.
            let mut t = Vec::with_capacity(nf);
            for wi in &w {
                let mut proj: Vec<f64> = x.iter().map(|r| dot(wi, r)).collect();
                proj.sort_by(|a, b| a.total_cmp(b));
                t.push(quantile_sorted(&proj, 0.5));
            }
            FittedOp::ObliqueHinge { w, t, append }
        }
        "row_stats" => FittedOp::RowStats { append },
        "pair_prod" => {
            let topk = stage.get_usize("topk", 8).clamp(2, 24).min(cols);
            // Target-guided column selection: |pearson(x_j, y)| descending, index ascending.
            let mut scored: Vec<(usize, f64)> = (0..cols)
                .map(|j| {
                    let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
                    (j, crate::metrics::pearson(&col, y).abs())
                })
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut sel: Vec<usize> = scored[..topk].iter().map(|(j, _)| *j).collect();
            sel.sort_unstable();
            let mut pairs = Vec::new();
            for a in 0..sel.len() {
                for b in (a + 1)..sel.len() {
                    pairs.push((sel[a], sel[b]));
                }
            }
            FittedOp::PairProd { pairs, append }
        }
        "target_bin_encode" => {
            let bins = stage.get_usize("bins", 8).clamp(2, 32);
            let prior = stage.get_f64("prior", 10.0).max(0.0);
            let edges = quantile_edges(x, bins);
            let y_mean = y.iter().sum::<f64>() / y.len().max(1) as f64;
            let mut enc = Vec::with_capacity(cols);
            for j in 0..cols {
                let mut sums = vec![0.0; bins];
                let mut cnts = vec![0.0; bins];
                for (r, row) in x.iter().enumerate() {
                    let b = lower_bound(&edges[j], row[j]).min(bins - 1);
                    sums[b] += y[r];
                    cnts[b] += 1.0;
                }
                enc.push(
                    (0..bins)
                        .map(|b| (sums[b] + prior * y_mean) / (cnts[b] + prior).max(1e-9))
                        .collect(),
                );
            }
            FittedOp::TargetBinEncode { edges, enc, append }
        }
        // NOTE: no `concat_raw` op (and no separate `winsor_moments`):
        // `winsor_moments` already aliases `winsor_clip` above, and a pure
        // passthrough whose only effect under append=true is duplicating the
        // input adds width without information (the 512-column cap makes that
        // strictly harmful). Deliberately skipped.
        "sigmoid" => FittedOp::Sigmoid { append },
        // Stateless hard indicator at a fixed threshold; no fit statistics.
        "step" => FittedOp::Step {
            t: stage.get_f64("t", 0.5),
            append,
        },
        "norm_row" => FittedOp::NormRow { append },
        "polar" => FittedOp::Polar { append },
        "lag" => FittedOp::Lag {
            k: stage.get_usize("k", 1).max(1),
            append,
        },
        "roll" => FittedOp::Roll {
            k: stage.get_usize("k", 1).max(1),
            append,
        },
        "fft_mag" => {
            // Frequencies 1..=k of the plain per-row DFT; k is capped at d/2
            // (at least 1) using the fit-time width, which the pipeline keeps
            // constant at apply time.
            let k = stage.get_usize("k", 8).clamp(1, (cols / 2).max(1));
            FittedOp::FftMag { k, append }
        }
        "row_spectral" => FittedOp::RowSpectral { append },
        "row_diff_stats" => FittedOp::RowDiffStats { append },
        "svd_project" => {
            let k = stage.get_usize("k", 8).clamp(1, 64).min(cols);
            let iters = stage.get_usize("iters", 20).clamp(1, 200);
            let seed = stage.get_u64("seed", 7);
            let (means, components) = svd_power_fit(x, k, iters, seed);
            FittedOp::SvdProject {
                means,
                components,
                append,
            }
        }
        "nystrom" => {
            let k = stage.get_usize("k", 16).clamp(1, 128).min(n);
            let gamma = stage.get_f64("gamma", 1.0).max(1e-6);
            let seed = stage.get_u64("seed", 7);
            // k distinct TRAIN rows as RBF landmarks via partial Fisher–Yates
            // over idx = [0..n). Draw order: for i = 0..k exactly one
            // next_u32(); j = i + (u32 % (n − i)); swap idx[i], idx[j].
            // Landmarks are stored in selection order idx[0..k].
            let mut rng = Pcg32::for_stage(seed, cols, k);
            let mut idx: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = i + (rng.next_u32() as usize) % (n - i);
                idx.swap(i, j);
            }
            let landmarks: Rows = idx[..k].iter().map(|&i| x[i].clone()).collect();
            FittedOp::Nystrom {
                landmarks,
                gamma,
                append,
            }
        }
        "symlog" => FittedOp::Symlog {
            c: stage.get_f64("c", 1.0).clamp(1e-9, 1e9),
            append,
        },
        "select_corr" => {
            let k = stage.get_usize("k", 32).clamp(1, 512).min(cols);
            let by_var = matches!(
                stage.params.get("by"),
                Some(crate::dsl::ParamValue::Str(s)) if s == "var"
            );
            // Same screening idiom as pair_prod: score descending, ties broken
            // by the lower column index, kept indices re-sorted ascending.
            let mut scored: Vec<(usize, f64)> = Vec::new();
            for j in 0..cols {
                let score = if by_var {
                    let mut s = 0.0;
                    for r in x {
                        s += r[j];
                    }
                    let m = s / n as f64;
                    let mut v = 0.0;
                    for r in x {
                        let d = r[j] - m;
                        v += d * d;
                    }
                    v / n as f64
                } else {
                    let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
                    crate::metrics::pearson(&col, y).abs()
                };
                scored.push((j, score));
            }
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut indices: Vec<usize> = scored[..k].iter().map(|&(j, _)| j).collect();
            indices.sort_unstable();
            FittedOp::SelectCorr { indices, append }
        }
        "spline" => {
            // Same knot placement as hinge: interior quantiles i/(k+1).
            let k = stage.get_usize("knots", 5).clamp(3, 12);
            let knots = (0..cols)
                .map(|j| {
                    let s = sorted_col(x, j);
                    (1..=k)
                        .map(|i| quantile_sorted(&s, i as f64 / (k + 1) as f64))
                        .collect()
                })
                .collect();
            FittedOp::Spline { knots, append }
        }
        "pair_ratio" => {
            let topk = stage.get_usize("topk", 8).clamp(1, 64);
            // Same target-guided screen as pair_prod: |pearson| descending,
            // ties to the lower index, kept indices re-sorted ascending.
            let m = cols.min(32);
            let mut scored: Vec<(usize, f64)> = (0..cols)
                .map(|j| {
                    let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
                    (j, crate::metrics::pearson(&col, y).abs())
                })
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut screened: Vec<usize> = scored[..m].iter().map(|&(j, _)| j).collect();
            screened.sort_unstable();
            // Per screened column: population std -> denominator guard
            // eps = max(1e-12, (0.01 * std)^2).
            let mut guards = Vec::with_capacity(screened.len());
            for &j in &screened {
                let mut s = 0.0;
                for r in x {
                    s += r[j];
                }
                let mean = s / n as f64;
                let mut v = 0.0;
                for r in x {
                    let d = r[j] - mean;
                    v += d * d;
                }
                let e = 0.01 * (v / n as f64).sqrt();
                guards.push((e * e).max(1e-12));
            }
            // Score every ordered pair by |pearson(ratio, y)|.
            let mut candidates: Vec<(usize, usize, f64, f64)> = Vec::new();
            for (ai, &a) in screened.iter().enumerate() {
                for (bi, &b) in screened.iter().enumerate() {
                    if ai == bi {
                        continue;
                    }
                    let eps = guards[bi];
                    let ratio: Vec<f64> = x
                        .iter()
                        .map(|r| r[a] * r[b] / (r[b] * r[b] + eps))
                        .collect();
                    candidates.push((a, b, eps, crate::metrics::pearson(&ratio, y).abs()));
                }
            }
            candidates.sort_by(|l, r| r.3.total_cmp(&l.3).then(l.0.cmp(&r.0)).then(l.1.cmp(&r.1)));
            candidates.truncate(topk.min(candidates.len()));
            FittedOp::PairRatio {
                pairs: candidates.iter().map(|&(a, b, _, _)| (a, b)).collect(),
                eps: candidates.iter().map(|&(_, _, e, _)| e).collect(),
                append,
            }
        }
        "impute" => {
            // strategy encoding shared with the emitted twins: 0=mean,
            // 1=median (default), 2=const. `value` is the const-strategy fill;
            // it is passed but unused for mean/median.
            let strategy = match stage.params.get("strategy") {
                Some(crate::dsl::ParamValue::Str(s)) if s == "mean" => 0usize,
                Some(crate::dsl::ParamValue::Str(s)) if s == "const" => 2,
                _ => 1,
            };
            let value = stage.get_f64("value", 0.0).clamp(-1e12, 1e12);
            let add_indicator = matches!(
                stage.params.get("add_indicator"),
                Some(crate::dsl::ParamValue::Bool(true))
            ) || stage.get_usize("add_indicator", 0) == 1;
            let mut fill = vec![0.0; cols];
            let mut indicator_cols = Vec::new();
            for j in 0..cols {
                let mut vals: Vec<f64> = Vec::new();
                let mut missing = false;
                for r in x {
                    if r[j] == 0.0 {
                        missing = true;
                    } else {
                        vals.push(r[j]);
                    }
                }
                fill[j] = if vals.is_empty() {
                    // An all-zero column has no observed values to summarize.
                    0.0
                } else if strategy == 0 {
                    let mut s = 0.0;
                    for v in &vals {
                        s += *v;
                    }
                    s / vals.len() as f64
                } else if strategy == 2 {
                    value
                } else {
                    vals.sort_by(|a, b| a.total_cmp(b));
                    quantile_sorted(&vals, 0.5)
                };
                if missing && add_indicator && cols + indicator_cols.len() < crate::MAX_WIDTH {
                    indicator_cols.push(j);
                }
            }
            FittedOp::Impute {
                fill,
                indicator_cols,
                append,
            }
        }
        "map" | "derive" => {
            // Expression is parsed at FIT time; a parse failure fails the fit.
            let source = match stage.params.get("expr") {
                Some(crate::dsl::ParamValue::Str(s)) => s.clone(),
                _ => bail!("{} requires a string parameter expr=\"...\"", stage.op),
            };
            let expr = MathExpr::parse(&source)?;
            if stage.op == "map" {
                FittedOp::MapExpr {
                    source,
                    expr,
                    append,
                }
            } else {
                FittedOp::DeriveExpr {
                    source,
                    expr,
                    append,
                }
            }
        }
        other => bail!("unsupported battle op: {other}"),
    };
    Ok(op)
}

/// Top-k PCA basis by power iteration on the covariance action
/// w = Σ_i ((x_i − μ)·v)(x_i − μ), with Gram–Schmidt deflation against
/// already-accepted components inside EVERY iteration. Deterministic:
/// fixed `iters`, no convergence-dependent branching. Draw order: for each
/// component c = 0..k, exactly `cols` gaussians are drawn (in order) for its
/// init vector before any iteration; nothing else is drawn. Shared by the
/// `svd_project` stage (k pre-clamped there) and the portfolio-level wide-data
/// view (k up to 256).
pub fn svd_power_fit(x: &Rows, k: usize, iters: usize, seed: u64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let cols = x[0].len();
    let n = x.len();
    // Column means on train; centering is implicit in fit and apply.
    let mut means = vec![0.0; cols];
    for j in 0..cols {
        let mut s = 0.0;
        for r in x {
            s += r[j];
        }
        means[j] = s / n as f64;
    }
    let mut rng = Pcg32::for_stage(seed, cols, k);
    let mut components: Vec<Vec<f64>> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut v: Vec<f64> = (0..cols).map(|_| rng.gaussian()).collect();
        normalize(&mut v);
        for _ in 0..iters {
            let mut w = vec![0.0; cols];
            for r in x {
                let mut s = 0.0;
                for j in 0..cols {
                    s += (r[j] - means[j]) * v[j];
                }
                for j in 0..cols {
                    w[j] += s * (r[j] - means[j]);
                }
            }
            for u in &components {
                let d = dot(&w, u);
                for j in 0..cols {
                    w[j] -= d * u[j];
                }
            }
            normalize(&mut w);
            v = w;
        }
        // Sign convention: flip so the largest-|value| coordinate is
        // positive; ties keep the FIRST index among maxima (strict `>`).
        let mut best = 0;
        for j in 1..cols {
            if v[j].abs() > v[best].abs() {
                best = j;
            }
        }
        if v[best] < 0.0 {
            for e in v.iter_mut() {
                *e = -*e;
            }
        }
        components.push(v);
    }
    (means, components)
}

// --------------------------------- apply ---------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len().min(b.len()) {
        s += a[i] * b[i];
    }
    s
}

/// Scale to unit L2 norm; the 1e-12 floor keeps a degenerate vector finite
/// (branch-free, so cross-language behavior is identical).
fn normalize(v: &mut [f64]) {
    let mut s = 0.0;
    for e in v.iter() {
        s += e * e;
    }
    let inv = 1.0 / s.sqrt().max(1e-12);
    for e in v.iter_mut() {
        *e *= inv;
    }
}

/// Training-pass variant of `apply_op`. Ops whose train-time output must
/// differ from their inference output (e.g. leave-one-out target encoders)
/// get dedicated arms here; every other op forwards to `apply_op`, so the
/// training pass stays bit-identical for the existing vocabulary.
pub fn apply_op_train(op: &FittedOp, x: &Rows) -> Rows {
    apply_op(op, x)
}

pub fn apply_op(op: &FittedOp, x: &Rows) -> Rows {
    let out: Rows = match op {
        FittedOp::Standardize { means, scales, .. } => x
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| ((v - means[j]) / scales[j]).clamp(-20.0, 20.0))
                    .collect()
            })
            .collect(),
        FittedOp::RobustScale {
            medians, scales, ..
        } => x
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| (v - medians[j]) / scales[j])
                    .collect()
            })
            .collect(),
        FittedOp::Rank { sorted_cols, .. } => x
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| mid_rank(&sorted_cols[j], *v))
                    .collect()
            })
            .collect(),
        FittedOp::Blom { sorted_cols, .. } => x
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| {
                        let n = sorted_cols[j].len() as f64;
                        let mid = mid_rank(&sorted_cols[j], *v) * n;
                        probit((mid + 0.125) / (n + 0.25))
                    })
                    .collect()
            })
            .collect(),
        FittedOp::QuantileBins { edges, bins, .. } => x
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| lower_bound(&edges[j], *v) as f64 / (*bins as f64 - 1.0))
                    .collect()
            })
            .collect(),
        FittedOp::BinOnehot { edges, bins, .. } => x
            .iter()
            .map(|r| {
                let mut o = vec![0.0; r.len() * bins];
                for (j, v) in r.iter().enumerate() {
                    let b = lower_bound(&edges[j], *v).min(bins - 1);
                    o[j * bins + b] = 1.0;
                }
                o
            })
            .collect(),
        FittedOp::Hinge { knots, .. } => x
            .iter()
            .map(|r| {
                let mut o = Vec::with_capacity(r.len() * knots[0].len());
                for (j, v) in r.iter().enumerate() {
                    for t in &knots[j] {
                        o.push((v - t).max(0.0));
                    }
                }
                o
            })
            .collect(),
        FittedOp::WinsorClip { lo, hi, .. } => x
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| v.clamp(lo[j], hi[j]))
                    .collect()
            })
            .collect(),
        FittedOp::Cheb { degree, .. } => x
            .iter()
            .map(|r| {
                let mut o = Vec::with_capacity(r.len() * degree);
                for v in r {
                    let u = (2.0 * v - 1.0).clamp(-1.0, 1.0);
                    let mut t_prev = 1.0;
                    let mut t_cur = u;
                    for _ in 0..*degree {
                        o.push(t_cur);
                        let t_next = 2.0 * u * t_cur - t_prev;
                        t_prev = t_cur;
                        t_cur = t_next;
                    }
                }
                o
            })
            .collect(),
        FittedOp::PolyInteract { pairs, .. } | FittedOp::PairProd { pairs, .. } => x
            .iter()
            .map(|r| pairs.iter().map(|(a, b)| r[*a] * r[*b]).collect())
            .collect(),
        FittedOp::RffCos { w, b, .. } => x
            .iter()
            .map(|r| {
                let scale = (2.0 / w.len() as f64).sqrt();
                (0..w.len())
                    .map(|i| scale * (dot(&w[i], r) + b[i]).cos())
                    .collect()
            })
            .collect(),
        FittedOp::RffSin { w, b, .. } => x
            .iter()
            .map(|r| {
                let scale = (2.0 / w.len() as f64).sqrt();
                (0..w.len())
                    .map(|i| scale * (dot(&w[i], r) + b[i]).sin())
                    .collect()
            })
            .collect(),
        FittedOp::RandProj { w, .. } => x
            .iter()
            .map(|r| {
                let inv = 1.0 / (w.len() as f64).sqrt();
                (0..w.len()).map(|i| dot(&w[i], r) * inv).collect()
            })
            .collect(),
        FittedOp::LandmarkDist {
            landmarks, gamma, ..
        }
        | FittedOp::Nystrom {
            landmarks, gamma, ..
        } => x
            .iter()
            .map(|r| {
                landmarks
                    .iter()
                    .map(|l| {
                        let mut d2 = 0.0;
                        for j in 0..l.len().min(r.len()) {
                            let d = r[j] - l[j];
                            d2 += d * d;
                        }
                        (-gamma * d2).exp()
                    })
                    .collect()
            })
            .collect(),
        FittedOp::ObliqueHinge { w, t, .. } => x
            .iter()
            .map(|r| {
                (0..w.len())
                    .map(|i| (dot(&w[i], r) - t[i]).max(0.0))
                    .collect()
            })
            .collect(),
        FittedOp::RowStats { .. } => x.iter().map(|r| row_stats(r)).collect(),
        FittedOp::TargetBinEncode { edges, enc, .. } => x
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| {
                        let bins = enc[j].len();
                        enc[j][lower_bound(&edges[j], *v).min(bins - 1)]
                    })
                    .collect()
            })
            .collect(),
        // Same clamp as MathExpr's sigmoid: keeps Python's math.exp overflow-free.
        FittedOp::Sigmoid { .. } => x
            .iter()
            .map(|r| {
                r.iter()
                    .map(|v| 1.0 / (1.0 + (-v.clamp(-30.0, 30.0)).exp()))
                    .collect()
            })
            .collect(),
        // Hard indicator per cell: 1.0 when v > t else 0.0 (NaN => 0.0). The
        // explicit conditional mirrors the emitted Python/Rust exactly.
        FittedOp::Step { t, .. } => x
            .iter()
            .map(|r| r.iter().map(|v| if *v > *t { 1.0 } else { 0.0 }).collect())
            .collect(),
        FittedOp::NormRow { .. } => x
            .iter()
            .map(|r| {
                let mut s = 0.0;
                for v in r {
                    s += v * v;
                }
                let inv = 1.0 / s.sqrt().max(1e-9);
                r.iter().map(|v| v * inv).collect()
            })
            .collect(),
        // Per consecutive pair (0,1),(2,3),..: radius then angle atan2(b, a);
        // an odd trailing column passes through unchanged (width preserved).
        FittedOp::Polar { .. } => x
            .iter()
            .map(|r| {
                let d = r.len();
                let mut o = Vec::with_capacity(d);
                let mut j = 0;
                while j + 1 < d {
                    let (a, b) = (r[j], r[j + 1]);
                    o.push((a * a + b * b).sqrt());
                    o.push(b.atan2(a));
                    j += 2;
                }
                if d % 2 == 1 {
                    o.push(r[d - 1]);
                }
                o
            })
            .collect(),
        // Row-internal shift: out[j] = x[j-k] for j >= k else 0.0 (k >= d ⇒ zeros).
        FittedOp::Lag { k, .. } => x
            .iter()
            .map(|r| {
                (0..r.len())
                    .map(|j| if j >= *k { r[j - *k] } else { 0.0 })
                    .collect()
            })
            .collect(),
        // Circular shift: out[j] = x[(j - k) mod d].
        FittedOp::Roll { k, .. } => x
            .iter()
            .map(|r| {
                let d = r.len();
                let s = *k % d;
                (0..d).map(|j| r[(j + d - s) % d]).collect()
            })
            .collect(),
        FittedOp::FftMag { k, .. } => x.iter().map(|r| dft_mags(r, *k)).collect(),
        FittedOp::RowSpectral { .. } => x.iter().map(|r| row_spectral(r)).collect(),
        FittedOp::RowDiffStats { .. } => x.iter().map(|r| row_diff_stats(r)).collect(),
        FittedOp::SvdProject {
            means, components, ..
        } => x
            .iter()
            .map(|r| {
                components
                    .iter()
                    .map(|c| {
                        let mut s = 0.0;
                        for j in 0..means.len().min(r.len()) {
                            s += (r[j] - means[j]) * c[j];
                        }
                        s
                    })
                    .collect()
            })
            .collect(),
        // Elementwise with row context: out[i][j] = expr(x = r[j], row = r).
        FittedOp::MapExpr { expr, .. } => x
            .iter()
            .map(|r| {
                r.iter()
                    .map(|v| expr.eval_element(*v, r).unwrap_or(0.0))
                    .collect()
            })
            .collect(),
        // One derived column per row: out[i] = [expr(row = r)].
        FittedOp::DeriveExpr { expr, .. } => x
            .iter()
            .map(|r| vec![expr.eval_row(r).unwrap_or(0.0)])
            .collect(),
        FittedOp::Symlog { c, .. } => x
            .iter()
            .map(|r| r.iter().map(|v| asinh_stable(v / c)).collect())
            .collect(),
        FittedOp::SelectCorr { indices, .. } => x
            .iter()
            .map(|r| indices.iter().map(|&j| r[j]).collect())
            .collect(),
        FittedOp::Impute {
            fill,
            indicator_cols,
            ..
        } => {
            let mut out: Rows = Vec::new();
            for r in x {
                let mut vals: Vec<f64> = (0..r.len())
                    .map(|j| if r[j] == 0.0 { fill[j] } else { r[j] })
                    .collect();
                for &j in indicator_cols.iter() {
                    vals.push(if r[j] == 0.0 { 1.0 } else { 0.0 });
                }
                out.push(vals);
            }
            out
        }
        FittedOp::PairRatio { pairs, eps, .. } => x
            .iter()
            .map(|r| {
                pairs
                    .iter()
                    .zip(eps)
                    .map(|(&(a, b), e)| r[a] * r[b] / (r[b] * r[b] + e))
                    .collect()
            })
            .collect(),
        // Truncated-power natural cubic basis; like hinge, over-wide output
        // relies on the shared MAX_WIDTH truncation in `finish`.
        FittedOp::Spline { knots, .. } => x
            .iter()
            .map(|r| {
                let mut o = Vec::new();
                for (j, v) in r.iter().enumerate() {
                    let xi = &knots[j];
                    let k = xi.len();
                    if xi[k - 1] - xi[0] < 1e-9 {
                        o.push(*v);
                    } else {
                        o.push(*v);
                        let pl = (v - xi[k - 1]).max(0.0);
                        let pk = (v - xi[k - 2]).max(0.0);
                        let dlast =
                            (pk * pk * pk - pl * pl * pl) / (xi[k - 1] - xi[k - 2]).max(1e-9);
                        for t in 0..(k - 2) {
                            let pt = (v - xi[t]).max(0.0);
                            let d = (pt * pt * pt - pl * pl * pl) / (xi[k - 1] - xi[t]).max(1e-9);
                            o.push(d - dlast);
                        }
                    }
                }
                o
            })
            .collect(),
    };
    finish(op, x, out)
}

fn op_append(op: &FittedOp) -> bool {
    match op {
        FittedOp::Standardize { append, .. }
        | FittedOp::RobustScale { append, .. }
        | FittedOp::Rank { append, .. }
        | FittedOp::Blom { append, .. }
        | FittedOp::QuantileBins { append, .. }
        | FittedOp::BinOnehot { append, .. }
        | FittedOp::Hinge { append, .. }
        | FittedOp::WinsorClip { append, .. }
        | FittedOp::Cheb { append, .. }
        | FittedOp::PolyInteract { append, .. }
        | FittedOp::RffCos { append, .. }
        | FittedOp::RffSin { append, .. }
        | FittedOp::RandProj { append, .. }
        | FittedOp::LandmarkDist { append, .. }
        | FittedOp::ObliqueHinge { append, .. }
        | FittedOp::RowStats { append }
        | FittedOp::PairProd { append, .. }
        | FittedOp::TargetBinEncode { append, .. }
        | FittedOp::Sigmoid { append }
        | FittedOp::Step { append, .. }
        | FittedOp::NormRow { append }
        | FittedOp::Polar { append }
        | FittedOp::Lag { append, .. }
        | FittedOp::Roll { append, .. }
        | FittedOp::FftMag { append, .. }
        | FittedOp::RowSpectral { append }
        | FittedOp::RowDiffStats { append }
        | FittedOp::SvdProject { append, .. }
        | FittedOp::Nystrom { append, .. }
        | FittedOp::MapExpr { append, .. }
        | FittedOp::DeriveExpr { append, .. }
        | FittedOp::Symlog { append, .. }
        | FittedOp::SelectCorr { append, .. }
        | FittedOp::Impute { append, .. }
        | FittedOp::PairRatio { append, .. }
        | FittedOp::Spline { append, .. } => *append,
    }
}

/// Append semantics + width cap + sanitize (mirrors the artifact epilogue of
/// every transform function).
fn finish(op: &FittedOp, input: &Rows, mut out: Rows) -> Rows {
    if op_append(op) {
        for (i, row) in out.iter_mut().enumerate() {
            let mut merged = input[i].clone();
            merged.append(row);
            *row = merged;
        }
    }
    for row in out.iter_mut() {
        if row.len() > crate::MAX_WIDTH {
            row.truncate(crate::MAX_WIDTH);
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

/// Ten per-row summary statistics: mean, std, min, max, median, q25, q75,
/// skewness, excess kurtosis, energy.
fn row_stats(r: &[f64]) -> Vec<f64> {
    let n = r.len() as f64;
    let mean = r.iter().sum::<f64>() / n;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    let mut energy = 0.0;
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for v in r {
        let d = v - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
        energy += v * v;
        if *v < mn {
            mn = *v;
        }
        if *v > mx {
            mx = *v;
        }
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
    energy /= n;
    let std = m2.sqrt();
    let skew = if m2 > 1e-24 { m3 / (m2 * std) } else { 0.0 };
    let kurt = if m2 > 1e-24 {
        m4 / (m2 * m2) - 3.0
    } else {
        0.0
    };
    let mut s = r.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    vec![
        mean,
        std,
        mn,
        mx,
        quantile_sorted(&s, 0.5),
        quantile_sorted(&s, 0.25),
        quantile_sorted(&s, 0.75),
        skew,
        kurt,
        energy,
    ]
}

/// Magnitudes of the plain O(d²) DFT of one row at frequencies f = 1..=k:
/// mag_f = sqrt((Σ_j x_j cos(2π f j / d))² + (Σ_j x_j sin(2π f j / d))²) / d.
/// Angle is computed left-to-right as ((2π · f) · j) / d for exact Python parity.
fn dft_mags(r: &[f64], k: usize) -> Vec<f64> {
    let d = r.len() as f64;
    let mut out = Vec::with_capacity(k);
    for f in 1..=k {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, v) in r.iter().enumerate() {
            let ang = PI2 * f as f64 * j as f64 / d;
            re += v * ang.cos();
            im += v * ang.sin();
        }
        out.push((re * re + im * im).sqrt() / d);
    }
    out
}

/// Five per-row spectral summaries over the full DFT magnitudes at f = 1..=d/2:
/// spectral entropy of the normalized spectrum (-Σ p ln p, terms with
/// p <= 1e-12 skipped), spectral moments m1 = Σ f·p and m2 = Σ f²·p, peak
/// frequency (lowest f on ties, as f64), peak magnitude. Width-1 rows (no
/// frequencies) emit all zeros.
fn row_spectral(r: &[f64]) -> Vec<f64> {
    let nf = r.len() / 2;
    if nf == 0 {
        return vec![0.0; 5];
    }
    let mags = dft_mags(r, nf);
    let mut total = 0.0;
    for m in &mags {
        total += m;
    }
    let denom = total.max(1e-12);
    let mut ent = 0.0;
    let mut m1 = 0.0;
    let mut m2 = 0.0;
    let mut peak = 0usize;
    for (i, m) in mags.iter().enumerate() {
        let p = m / denom;
        if p > 1e-12 {
            ent -= p * p.ln();
        }
        let f = (i + 1) as f64;
        m1 += f * p;
        m2 += f * f * p;
        // Strict `>` keeps the lowest frequency among tied maxima.
        if *m > mags[peak] {
            peak = i;
        }
    }
    vec![ent, m1, m2, (peak + 1) as f64, mags[peak]]
}

/// Eight per-row stats over first differences d_j = x_{j+1} − x_j:
/// mean, std (population), min, max, mean |d|, sign changes of consecutive
/// diffs / (d−1), lag-1 autocorrelation of the ORIGINAL row (Pearson of
/// x[..d−1] vs x[1..], zero variance → 0), energy = mean d². Rows shorter
/// than 2 emit all zeros.
fn row_diff_stats(r: &[f64]) -> Vec<f64> {
    let d = r.len();
    if d < 2 {
        return vec![0.0; 8];
    }
    let nd = d - 1;
    let ndf = nd as f64;
    let diffs: Vec<f64> = (0..nd).map(|j| r[j + 1] - r[j]).collect();
    let mean = diffs.iter().sum::<f64>() / ndf;
    let mut m2 = 0.0;
    let mut mean_abs = 0.0;
    let mut energy = 0.0;
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    let mut signs = 0usize;
    for (j, v) in diffs.iter().enumerate() {
        let dd = v - mean;
        m2 += dd * dd;
        mean_abs += v.abs();
        energy += v * v;
        if *v < mn {
            mn = *v;
        }
        if *v > mx {
            mx = *v;
        }
        // A sign change is a strictly negative product of consecutive diffs.
        if j > 0 && diffs[j - 1] * v < 0.0 {
            signs += 1;
        }
    }
    let std = (m2 / ndf).sqrt();
    let auto = crate::metrics::pearson(&r[..nd], &r[1..]);
    vec![
        mean,
        std,
        mn,
        mx,
        mean_abs / ndf,
        signs as f64 / ndf,
        auto,
        energy / ndf,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::parse_dsl;
    use crate::model::cv_r2;
    use crate::GenomeIr;

    fn stage(src: &str) -> Stage {
        parse_dsl(&format!("(pipe X {src})")).unwrap().stages[0].clone()
    }

    fn toy(rows: usize, cols: usize) -> (Rows, Vec<f64>) {
        let mut rng = Pcg32::new(11, 3);
        let mut x = Vec::with_capacity(rows);
        let mut y = Vec::with_capacity(rows);
        for _ in 0..rows {
            let r: Vec<f64> = (0..cols).map(|_| rng.next_f64()).collect();
            y.push(2.0 * r[0] - r[1 % cols] + 0.3 * (6.0 * r[2 % cols]).sin());
            x.push(r);
        }
        (x, y)
    }

    fn tiny() -> (Rows, Vec<f64>) {
        (
            vec![vec![3.0, 4.0, 0.5], vec![1.0, 0.0, 2.0]],
            vec![0.0, 0.0],
        )
    }

    #[test]
    fn impute_fills_zeros_and_appends_indicators() {
        // Column 0 has zeros (missing); column 1 is fully observed.
        let x: Rows = vec![
            vec![0.0, 1.0],
            vec![1.0, 3.0],
            vec![2.0, 5.0],
            vec![9.0, 7.0],
        ];
        let y = vec![0.0; 4];

        // Median of the non-zero values [1, 2, 9] is 2.0.
        let op = fit_op(
            &stage("(impute strategy=median add_indicator=true)"),
            &x,
            &y,
        )
        .unwrap();
        match &op {
            FittedOp::Impute {
                fill,
                indicator_cols,
                ..
            } => {
                assert_eq!(fill[0], 2.0);
                assert_eq!(indicator_cols, &vec![0]);
            }
            other => panic!("wrong variant: {other:?}"),
        }
        let out = apply_op(&op, &x);
        assert_eq!(out[0], vec![2.0, 1.0, 1.0]); // filled + indicator hot
        assert_eq!(out[1], vec![1.0, 3.0, 0.0]);
        assert_eq!(out[3], vec![9.0, 7.0, 0.0]);

        // Mean of [1, 2, 9] is 4.0.
        let op = fit_op(&stage("(impute strategy=mean)"), &x, &y).unwrap();
        let out = apply_op(&op, &x);
        assert_eq!(out[0], vec![4.0, 1.0]); // no indicator without the flag
        assert_eq!(out[0].len(), 2);

        // Const fill uses the value literal.
        let op = fit_op(&stage("(impute strategy=const value=7.5)"), &x, &y).unwrap();
        let out = apply_op(&op, &x);
        assert_eq!(out[0][0], 7.5);

        // An all-zero column keeps 0.0 (nothing observed to summarize).
        let xz: Rows = vec![vec![0.0, 1.0], vec![0.0, 2.0], vec![0.0, 3.0]];
        let op = fit_op(&stage("(impute strategy=mean)"), &xz, &y[..3]).unwrap();
        let out = apply_op(&op, &xz);
        assert_eq!(out[0][0], 0.0);

        // Determinism: double fit is bit-identical.
        let s = stage("(impute strategy=median add_indicator=true)");
        let a = apply_op(&fit_op(&s, &x, &y).unwrap(), &x);
        let b = apply_op(&fit_op(&s, &x, &y).unwrap(), &x);
        assert_eq!(a, b);
    }

    #[test]
    fn svd_project_is_deterministic_and_orthonormal() {
        let (x, y) = toy(120, 8);
        let s = stage("(svd_project k=4 iters=20 seed=7)");
        let op1 = fit_op(&s, &x, &y).unwrap();
        let op2 = fit_op(&s, &x, &y).unwrap();
        let comps = match &op1 {
            FittedOp::SvdProject { components, .. } => components.clone(),
            other => panic!("wrong variant: {other:?}"),
        };
        match &op2 {
            FittedOp::SvdProject { components, .. } => {
                for (a, b) in comps.iter().zip(components) {
                    for (u, v) in a.iter().zip(b) {
                        assert_eq!(u.to_bits(), v.to_bits());
                    }
                }
            }
            other => panic!("wrong variant: {other:?}"),
        }
        assert_eq!(comps.len(), 4);
        for i in 0..comps.len() {
            assert!(
                (dot(&comps[i], &comps[i]) - 1.0).abs() < 1e-9,
                "component {i} not unit"
            );
            for j in 0..i {
                assert!(
                    dot(&comps[i], &comps[j]).abs() < 1e-6,
                    "components {i},{j} not orthogonal"
                );
            }
            // Sign convention: largest-|value| coordinate is positive.
            let mut best = 0;
            for j in 1..comps[i].len() {
                if comps[i][j].abs() > comps[i][best].abs() {
                    best = j;
                }
            }
            assert!(comps[i][best] > 0.0);
        }
        let out = apply_op(&op1, &x);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn nystrom_landmarks_are_deterministic_train_rows() {
        let (x, y) = toy(60, 5);
        let s = stage("(nystrom k=8 gamma=1.5 seed=9)");
        let op1 = fit_op(&s, &x, &y).unwrap();
        let op2 = fit_op(&s, &x, &y).unwrap();
        let l1 = match &op1 {
            FittedOp::Nystrom { landmarks, .. } => landmarks.clone(),
            other => panic!("wrong variant: {other:?}"),
        };
        let l2 = match &op2 {
            FittedOp::Nystrom { landmarks, .. } => landmarks.clone(),
            other => panic!("wrong variant: {other:?}"),
        };
        assert_eq!(l1, l2);
        assert_eq!(l1.len(), 8);
        for l in &l1 {
            assert!(x.iter().any(|r| r == l), "landmark is not a train row");
        }
        for i in 0..l1.len() {
            for j in 0..i {
                assert_ne!(l1[i], l1[j], "landmarks {i},{j} not distinct");
            }
        }
        let out = apply_op(&op1, &x);
        assert_eq!(out[0].len(), 8);
        for v in out.iter().flatten() {
            assert!(v.is_finite() && *v > 0.0 && *v <= 1.0);
        }
    }

    #[test]
    fn map_and_derive_round_trip_expressions() {
        let (x, y) = toy(40, 3);
        let m = stage("(map expr=\"sigmoid(col(0))\")");
        let op = fit_op(&m, &x, &y).unwrap();
        match &op {
            FittedOp::MapExpr { source, .. } => assert_eq!(source, "sigmoid(col(0))"),
            other => panic!("wrong variant: {other:?}"),
        }
        let out = apply_op(&op, &x);
        assert_eq!(out[0].len(), 3);
        for (i, r) in x.iter().enumerate() {
            let want = 1.0 / (1.0 + (-r[0].clamp(-30.0, 30.0)).exp());
            for v in &out[i] {
                assert!((v - want).abs() < 1e-15);
            }
        }
        let d = stage("(derive expr=\"row_mean + col(1)\")");
        let opd = fit_op(&d, &x, &y).unwrap();
        let outd = apply_op(&opd, &x);
        assert_eq!(outd[0].len(), 1);
        for (i, r) in x.iter().enumerate() {
            let want = r.iter().sum::<f64>() / r.len() as f64 + r[1];
            assert!((outd[i][0] - want).abs() < 1e-12);
        }
        // Parse failure and missing expr bail at fit time.
        assert!(fit_op(&stage("(map expr=\"nope(\")"), &x, &y).is_err());
        assert!(fit_op(&stage("(derive)"), &x, &y).is_err());
    }

    #[test]
    fn scalar_batch_functions_evaluate_with_guarded_semantics() {
        let (x, y) = toy(30, 3);
        // Every new function parses, fits, and matches its reference formula.
        let cases: [(&str, fn(f64) -> f64); 8] = [
            ("exp", |v| v.clamp(-30.0, 30.0).exp().min(1.0e12)),
            ("log", |v| if v <= 1e-300 { 0.0 } else { v.ln() }),
            ("asinh", asinh_stable),
            ("atan", |v| v.atan()),
            ("erf", |v| {
                let a = v.abs();
                let s = if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let t = 1.0 / (1.0 + 0.3275911 * a);
                let poly = t
                    * (0.254829592
                        + t * (-0.284496736
                            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
                s * (1.0 - poly * (-(a * a)).exp())
            }),
            ("sign", |v| {
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }),
            ("floor", |v| v.floor()),
            ("round", |v| {
                if v >= 0.0 {
                    (v + 0.5).floor()
                } else {
                    (v - 0.5).ceil()
                }
            }),
        ];
        for (name, reference) in cases {
            let m = stage(&format!("(map expr=\"{name}(col(0))\")"));
            let op = fit_op(&m, &x, &y).unwrap();
            let out = apply_op(&op, &x);
            for (i, r) in x.iter().enumerate() {
                let want = reference(r[0]);
                assert_eq!(out[i][0], want, "{name}({}) mismatch", r[0]);
                assert!(out[i][0].is_finite(), "{name} produced non-finite");
            }
        }
        // Hand values: erf accuracy, round half-away, sign at zero.
        let probe = |src: &str, v: f64| -> f64 {
            let rows = vec![vec![v]];
            let op = fit_op(&stage(src), &rows, &[0.0]).unwrap();
            apply_op(&op, &rows)[0][0]
        };
        assert!((probe("(map expr=\"erf(col(0))\")", 1.0) - 0.8427007929497149).abs() < 3e-7);
        assert!((probe("(map expr=\"erf(col(0))\")", -2.0) + 0.9953222650189527).abs() < 3e-7);
        assert_eq!(probe("(map expr=\"round(col(0))\")", 2.5), 3.0);
        assert_eq!(probe("(map expr=\"round(col(0))\")", -2.5), -3.0);
        assert_eq!(probe("(map expr=\"round(col(0))\")", 0.5), 1.0);
        assert_eq!(probe("(map expr=\"sign(col(0))\")", 0.0), 0.0);
        assert_eq!(probe("(map expr=\"floor(col(0))\")", -1.5), -2.0);
        assert_eq!(probe("(map expr=\"log(col(0))\")", -5.0), 0.0);
        // exp's clamp caps the argument at 30; the expression's final safe()
        // then caps exp(30) ~= 1.07e13 at the 1e12 output bound.
        assert_eq!(probe("(map expr=\"exp(col(0))\")", 100.0), 1.0e12);
    }

    #[test]
    fn fft_mag_and_row_spectral_are_finite() {
        let (x, y) = toy(30, 9);
        let op = fit_op(&stage("(fft_mag k=4)"), &x, &y).unwrap();
        let out = apply_op(&op, &x);
        assert_eq!(out[0].len(), 4);
        for v in out.iter().flatten() {
            assert!(v.is_finite());
        }
        // Default k=8 clamps to d/2 = 4 for 9 columns.
        let op = fit_op(&stage("(fft_mag)"), &x, &y).unwrap();
        assert_eq!(apply_op(&op, &x)[0].len(), 4);
        let op = fit_op(&stage("(row_spectral)"), &x, &y).unwrap();
        let out = apply_op(&op, &x);
        assert_eq!(out[0].len(), 5);
        for r in &out {
            for v in r {
                assert!(v.is_finite());
            }
            assert!(r[0] >= 0.0, "entropy negative: {}", r[0]);
            assert!(
                r[3] >= 1.0 && r[3] <= 4.0,
                "peak freq out of range: {}",
                r[3]
            );
            assert!(r[4] >= 0.0);
        }
    }

    #[test]
    fn polar_lag_roll_shapes_and_values() {
        let (x, y) = tiny();
        let out = apply_op(&fit_op(&stage("(polar)"), &x, &y).unwrap(), &x);
        assert_eq!(out[0].len(), 3);
        assert!((out[0][0] - 5.0).abs() < 1e-12);
        assert!((out[0][1] - 4.0f64.atan2(3.0)).abs() < 1e-15);
        assert_eq!(out[0][2], 0.5);
        let out = apply_op(&fit_op(&stage("(lag k=1)"), &x, &y).unwrap(), &x);
        assert_eq!(out[0], vec![0.0, 3.0, 4.0]);
        let out = apply_op(&fit_op(&stage("(lag k=5)"), &x, &y).unwrap(), &x);
        assert_eq!(out[1], vec![0.0, 0.0, 0.0]);
        let out = apply_op(&fit_op(&stage("(roll k=1)"), &x, &y).unwrap(), &x);
        assert_eq!(out[0], vec![0.5, 3.0, 4.0]);
        // k = 4 ≡ 1 (mod 3).
        let out = apply_op(&fit_op(&stage("(roll k=4)"), &x, &y).unwrap(), &x);
        assert_eq!(out[0], vec![0.5, 3.0, 4.0]);
        // append=true keeps the input on the left.
        let out = apply_op(
            &fit_op(&stage("(lag k=1 append=true)"), &x, &y).unwrap(),
            &x,
        );
        assert_eq!(out[0], vec![3.0, 4.0, 0.5, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn sigmoid_norm_row_and_diff_stats_values() {
        let (x, y) = tiny();
        let out = apply_op(&fit_op(&stage("(sigmoid)"), &x, &y).unwrap(), &x);
        assert!((out[1][1] - 0.5).abs() < 1e-15, "sigmoid(0) != 0.5");
        assert!(out.iter().flatten().all(|v| *v > 0.0 && *v < 1.0));
        let out = apply_op(&fit_op(&stage("(norm_row)"), &x, &y).unwrap(), &x);
        let n0 = out[0].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-12);
        // Row [3,4,0.5]: diffs [1,-3.5] → mean -1.25, sign ratio 1/2, lag-1
        // autocorr of the original row = -1 (n=2 Pearson), energy 6.625.
        let out = apply_op(&fit_op(&stage("(row_diff_stats)"), &x, &y).unwrap(), &x);
        assert_eq!(out[0].len(), 8);
        assert!((out[0][0] + 1.25).abs() < 1e-12);
        assert!((out[0][5] - 0.5).abs() < 1e-12);
        assert!((out[0][6] + 1.0).abs() < 1e-12);
        assert!((out[0][7] - 6.625).abs() < 1e-12);
    }

    /// Clamp-audit property test: near-constant columns (values `c ± 1e-16`)
    /// make interpolated quantiles cross by one ulp, which used to panic
    /// `f64::clamp` in winsor (min > max). Every registry op must fit+apply
    /// on such data without panicking and with all-finite output.
    #[test]
    fn every_op_survives_near_constant_columns() {
        // One stage per fit_op arm (aliases included), default params; map and
        // derive carry the minimal expr they require.
        const STAGES: &[&str] = &[
            "(standardize)",
            "(norm_features)",
            "(robust_scale)",
            "(rank)",
            "(blom)",
            "(quantile_bins)",
            "(bin_onehot)",
            "(hinge)",
            "(winsor_clip)",
            "(winsor_moments)",
            "(cheb)",
            "(poly_interact)",
            "(rff_cos)",
            "(rff_sin)",
            "(rand_proj)",
            "(landmark_dist)",
            "(oblique_hinge)",
            "(row_stats)",
            "(pair_prod)",
            "(target_bin_encode)",
            "(sigmoid)",
            "(norm_row)",
            "(polar)",
            "(lag)",
            "(roll)",
            "(fft_mag)",
            "(row_spectral)",
            "(row_diff_stats)",
            "(svd_project)",
            "(nystrom)",
            "(map expr=\"sigmoid(col(0))\")",
            "(derive expr=\"row_mean + col(0)\")",
            "(symlog)",
            "(symlog c=0.5)",
            "(select_corr)",
            "(select_corr k=4 by=corr)",
            "(select_corr k=4 by=var)",
            "(impute)",
            "(impute strategy=median add_indicator=true)",
            "(impute strategy=mean)",
            "(impute strategy=const value=1.5)",
            "(pair_ratio)",
            "(pair_ratio topk=4)",
            "(spline)",
            "(spline knots=3)",
            "(spline knots=12)",
        ];
        let mut rng = Pcg32::new(20260702, 9);
        for trial in 0..50u32 {
            // Row counts where the ulp crossing PROVABLY happens for the
            // centers below (q=0.05: e.g. n=24 & sqrt2 reproduces the logged
            // `min = 1.4142135623730951, max = 1.414213562373095` panic),
            // plus random sizes.
            let rows = match trial % 5 {
                0 => 24,
                1 => 37,
                2 => 39,
                _ => 20 + (rng.next_u32() as usize) % 80,
            };
            let cols = 3 + (rng.next_u32() as usize) % 5;
            // Per-column plan: col 0 exactly constant [c; n], col 1
            // near-constant (c ± 1e-16 jitter), last col varying, the rest a
            // random mix. Centers include the exact panic-log values.
            let centers = [
                std::f64::consts::SQRT_2,
                1.0 / 3.0,
                0.333333343,
                -0.05798013915354207,
            ];
            #[derive(Clone, Copy)]
            enum ColPlan {
                Exact(f64),
                Jitter(f64),
                Vary,
            }
            let mut pick_center = |rng: &mut Pcg32, forced: usize| -> f64 {
                if rng.next_f64() < 0.5 {
                    centers[forced % centers.len()]
                } else {
                    rng.range_f64(-2.0, 2.0)
                }
            };
            let plan: Vec<ColPlan> = (0..cols)
                .map(|j| {
                    if j == 0 {
                        ColPlan::Exact(centers[trial as usize % centers.len()])
                    } else if j == 1 {
                        ColPlan::Jitter(pick_center(&mut rng, trial as usize + 1))
                    } else if j + 1 == cols {
                        ColPlan::Vary
                    } else {
                        match rng.next_u32() % 3 {
                            0 => ColPlan::Exact(pick_center(&mut rng, trial as usize + j)),
                            1 => ColPlan::Jitter(pick_center(&mut rng, trial as usize + j)),
                            _ => ColPlan::Vary,
                        }
                    }
                })
                .collect();
            let mut x: Rows = Vec::with_capacity(rows);
            let mut y: Vec<f64> = Vec::with_capacity(rows);
            for _ in 0..rows {
                let r: Vec<f64> = plan
                    .iter()
                    .map(|p| match p {
                        ColPlan::Exact(c) => *c,
                        // ±1e-16 jitter: rounds to c or a 1-ulp neighbour
                        ColPlan::Jitter(c) => c + (rng.next_f64() - 0.5) * 2e-16,
                        ColPlan::Vary => rng.range_f64(-2.0, 2.0),
                    })
                    .collect();
                y.push(2.0 * r[cols - 1] - r[0] + 0.1 * rng.gaussian());
                x.push(r);
            }
            for src in STAGES {
                let op = fit_op(&stage(src), &x, &y)
                    .unwrap_or_else(|e| panic!("trial {trial}: fit {src} failed: {e}"));
                let out = apply_op(&op, &x);
                assert_eq!(out.len(), rows, "trial {trial}: {src} row count");
                for (i, row) in out.iter().enumerate() {
                    assert!(!row.is_empty(), "trial {trial}: {src} empty row {i}");
                    for v in row {
                        assert!(v.is_finite(), "trial {trial}: {src} non-finite output {v}");
                    }
                }
            }
        }
    }

    #[test]
    fn svd_pipeline_cv_smoke() {
        // Same toy pattern as model.rs tests (3 cols → svd k clamps to 3).
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
        let g = GenomeIr::parse("(pipe X (svd_project k=4) (rff_cos n=16 append=true) (ridge))")
            .unwrap();
        let rep = cv_r2(&g, &x, &y, 5, "toy").unwrap();
        assert!(rep.oof_r2.is_finite());
        assert!(rep.oof_r2 > 0.5, "oof_r2={}", rep.oof_r2);
    }
}
