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
use crate::kernel::width::{reject_zero_width_rows, WidthMismatch};
use crate::math_expr::MathExpr;
use crate::rng::Pcg32;
use crate::Rows;
use anyhow::{bail, Result};
use rayon::prelude::*;

mod apply;
mod apply_helpers;
mod fit;
mod stats;

pub use apply::{apply_op, apply_op_train};
pub(crate) use apply::{apply_op_generated, op_append};
pub(crate) use apply_helpers::{
    apply_row_vector_v2, dft_mags, finish, row_diff_stats, row_spectral, row_stats,
};
pub(crate) use fit::fit_op_with_cache;
pub(crate) use stats::{asinh_stable, get_append, quantile_edges_with_cache, SortedColumnsCache};
pub use stats::{lower_bound, mid_rank, probit, quantile_sorted, sorted_col, upper_bound};

fn map_rows<F>(x: &Rows, f: F) -> Rows
where
    F: Fn(&Vec<f64>) -> Vec<f64> + Sync + Send,
{
    let width = x.first().map_or(0, Vec::len);
    if x.len() * width.max(1) >= 65_536 && crate::is_parallel_enabled() {
        x.par_iter().map(f).collect()
    } else {
        x.iter().map(f).collect()
    }
}

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

pub fn fit_op(stage: &Stage, x: &Rows, y: &[f64]) -> Result<FittedOp> {
    let mut cache = SortedColumnsCache::new();
    fit_op_with_cache(stage, x, y, &mut cache)
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

/// Returns `true` when `op` has `append = true`, `x` is already sanitized to finite values,
/// and every row of `x` has width `== MAX_WIDTH`. Under those conditions `apply_op(op, x)` is
/// the identity transform on `x` (`G-6`).
pub(crate) fn is_append_noop_at_max_width(
    op: &FittedOp,
    x: &Rows,
    x_already_sanitized: bool,
) -> bool {
    op_append(op)
        && x_already_sanitized
        && !x.is_empty()
        && x.iter().all(|r| r.len() == crate::MAX_WIDTH)
}

/// Training-pass variant of `apply_op`. Ops whose train-time output must
/// differ from their inference output (e.g. leave-one-out target encoders)
/// get dedicated arms here; every other op forwards to `apply_op`, so the
/// training pass stays bit-identical for the existing vocabulary.
#[cfg(test)]
use crate::{dsl::parse_dsl, model::cv_r2, GenomeIr};

#[cfg(test)]

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

#[cfg(test)]
#[path = "ops_tests_fit.rs"]
mod tests_fit;
#[cfg(test)]
#[path = "ops_tests_transform.rs"]
mod tests_transform;
