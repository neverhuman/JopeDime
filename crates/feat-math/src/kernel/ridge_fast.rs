//! B11: certified ridge tools (`feat-math.ridge-fast.v1`).
//!
//! `heads::RidgeModel::fit` computes the legacy fit bit for bit with cache-tiled normal
//! equations; the tests here certify it against a verbatim copy of the former loop. This module
//! adds tools that keep the legacy operation order and are certified against references:
//!
//! - [`Factor`]: the legacy-order Cholesky factor with its forward vector. The factor of a
//!   design's leading columns is exactly the leading block of its full factor, so
//!   [`Factor::extend`] resumes from a prefix bit for bit, and [`Factor::schur_gain`] gives the
//!   objective decrease an appended column block buys.
//! - [`leave_one_out`]: PRESS at a frozen penalty by the Sherman-Morrison identity, checked
//!   against brute-force refits.
//! - [`Refusal`]: a pivot below `λ/2` (impossible in exact arithmetic), any non-finite pivot,
//!   factor entry or forward value, and any leverage past the floor refuses, so the caller refits
//!   with the legacy authority. Pairing a factor with its own design is the caller's guarantee;
//!   only another width or penalty can be detected and refused.
//!
//! No fused multiply-add and no reassociation is used; Rust does neither by default.
use crate::heads::std_fit;
use crate::Rows;

/// The algorithm identity recorded in receipts.
pub const PROFILE: &str = "feat-math.ridge-fast.v1";

/// The clamped standardized design of the first `n` rows, column-major: `z[j * n + r]`.
fn standardized(x: &Rows, n: usize, p: usize, means: &[f64], scales: &[f64]) -> Vec<f64> {
    let mut z = vec![0.0f64; n * p];
    for (r, row) in x[..n].iter().enumerate() {
        for j in 0..p {
            z[j * n + r] = ((row[j] - means[j]) / scales[j]).clamp(-20.0, 20.0);
        }
    }
    z
}

/// Why a certified route refuses an input. The caller then refits with the legacy authority,
/// [`crate::heads::RidgeModel::fit`], whose result is always defined.
#[derive(Clone, Debug, PartialEq)]
pub enum RidgeRefusal {
    Shape(&'static str),
    NonFinite(&'static str),
    /// A pivot before its square root was not finite or fell below `lambda / 2`; in exact
    /// arithmetic every pivot is at least `lambda`. Every design's penalty is at least `2e-12`, so
    /// this also covers the legacy `s <= 1e-12` replacement branch: no pivot the legacy solve
    /// would replace is ever accepted.
    UnsafePivot {
        index: usize,
        pivot: f64,
        lambda: f64,
    },
    /// A leave-one-out denominator `1 - h_ii` fell below [`LEVERAGE_FLOOR`].
    Leverage {
        row: usize,
        one_minus_h: f64,
    },
}

/// The smallest leave-one-out denominator `1 - h_ii` the certified PRESS accepts.
pub const LEVERAGE_FLOOR: f64 = 1e-4;

/// A frozen ridge design: standardized columns, labels centered at their mean, and an absolute
/// penalty. Prefix factors and leave-one-out residuals keep these bits: the column
/// standardization, the label mean and the penalty are never recomputed for a subset of rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Design {
    n: usize,
    p: usize,
    /// Column-major: `z[j * n + r]`.
    z: Vec<f64>,
    centered: Vec<f64>,
    lambda: f64,
}

impl Design {
    /// The design [`crate::heads::RidgeModel::fit`] solves for `x, y` at `lambda_frac`, or a refusal when a
    /// standardized value or a label is not finite.
    pub fn from_rows(x: &Rows, y: &[f64], lambda_frac: f64) -> Result<Design, Refusal> {
        let n = x.len().min(y.len());
        if n == 0 {
            return Err(Refusal::Shape("ridge design needs at least one row"));
        }
        let p = x[0].len();
        // std_fit reads every row's first p values, as RidgeModel::fit does: a shorter row is
        // refused here instead of panicking there.
        if x.iter().any(|row| row.len() < p) {
            return Err(Refusal::Shape(
                "every ridge design row needs the first row's columns",
            ));
        }
        let (means, scales) = std_fit(x);
        let y_mean = y[..n].iter().sum::<f64>() / n as f64;
        let design = Design {
            n,
            p,
            z: standardized(x, n, p, &means, &scales),
            centered: y[..n].iter().map(|value| value - y_mean).collect(),
            lambda: (lambda_frac * n as f64).max(1e-3),
        };
        if !design
            .z
            .iter()
            .chain(&design.centered)
            .all(|value| value.is_finite())
        {
            return Err(Refusal::NonFinite("ridge design and labels must be finite"));
        }
        Ok(design)
    }

    /// A design from column-major standardized columns `z` (`n` rows, `p` columns), labels that
    /// are centered here at their mean, and an absolute penalty `lambda ≥ 2e-12`.
    pub fn from_columns(
        z: Vec<f64>,
        n: usize,
        p: usize,
        y: &[f64],
        lambda: f64,
    ) -> Result<Design, Refusal> {
        if n == 0 || n.checked_mul(p) != Some(z.len()) || y.len() != n {
            return Err(Refusal::Shape(
                "ridge design needs n > 0, n x p columns and n labels",
            ));
        }
        // At least 2e-12, so a pivot under the legacy 1e-12 replacement threshold is always under
        // lambda / 2 and refuses.
        if !(lambda.is_finite() && lambda >= 2e-12) {
            return Err(Refusal::NonFinite(
                "ridge penalty must be finite and at least 2e-12",
            ));
        }
        if !z.iter().chain(y).all(|value| value.is_finite()) {
            return Err(Refusal::NonFinite("ridge design and labels must be finite"));
        }
        let mean = y.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = y.iter().map(|value| value - mean).collect();
        if !centered.iter().all(|value| value.is_finite()) {
            return Err(Refusal::NonFinite("ridge labels overflow when centered"));
        }
        Ok(Design {
            n,
            p,
            z,
            centered,
            lambda,
        })
    }

    pub fn rows(&self) -> usize {
        self.n
    }

    pub fn columns(&self) -> usize {
        self.p
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The same design restricted to its first `p_a` columns.
    pub fn leading(&self, p_a: usize) -> Design {
        let p_a = p_a.min(self.p);
        Design {
            n: self.n,
            p: p_a,
            z: self.z[..p_a * self.n].to_vec(),
            centered: self.centered.clone(),
            lambda: self.lambda,
        }
    }

    fn column(&self, j: usize) -> &[f64] {
        &self.z[j * self.n..(j + 1) * self.n]
    }

    /// The design without row `skip`, keeping every frozen bit of the others.
    fn without_row(&self, skip: usize) -> Design {
        let keep = |values: &[f64]| -> Vec<f64> {
            values
                .iter()
                .enumerate()
                .filter(|&(r, _)| r != skip)
                .map(|(_, &value)| value)
                .collect()
        };
        let mut z = Vec::with_capacity((self.n - 1) * self.p);
        for j in 0..self.p {
            z.extend(keep(self.column(j)));
        }
        Design {
            n: self.n - 1,
            p: self.p,
            z,
            centered: keep(&self.centered),
            lambda: self.lambda,
        }
    }
}

/// `G = ZᵀZ + lambda I = L Lᵀ` in [`crate::heads::cholesky_solve`]'s operation order, with the forward vector
/// `f = L⁻¹ Zᵀ yc`. Row `i` of `L` and entry `f_i` depend only on the first `i + 1` columns, so
/// the factor of a design's leading columns is exactly the leading block of its full factor, and
/// [`Factor::extend`] resumes from it bit for bit.
#[derive(Clone, Debug, PartialEq)]
pub struct Factor {
    p: usize,
    /// Row-major `p x p`; the strict upper triangle is zero.
    l: Vec<f64>,
    f: Vec<f64>,
    lambda: f64,
}

impl Factor {
    /// The factor of `design`.
    pub fn new(design: &Design) -> Result<Factor, Refusal> {
        Self::resume(None, design)
    }

    /// The factor of `design`, resumed from `prefix`, the factor of its first `prefix.p` columns.
    /// The caller guarantees those columns, labels and penalty are the ones `prefix` was built
    /// from; the result then equals [`Factor::new`] bit for bit.
    pub fn extend(prefix: &Factor, design: &Design) -> Result<Factor, Refusal> {
        if prefix.p > design.p || prefix.lambda.to_bits() != design.lambda.to_bits() {
            return Err(Refusal::Shape(
                "a ridge prefix needs fewer columns and the same penalty",
            ));
        }
        Self::resume(Some(prefix), design)
    }

    fn resume(prefix: Option<&Factor>, design: &Design) -> Result<Factor, Refusal> {
        let p = design.p;
        let mut l = vec![0.0f64; p * p];
        let mut f = vec![0.0f64; p];
        let start = prefix.map_or(0, |prefix| prefix.p);
        if let Some(prefix) = prefix {
            for i in 0..start {
                l[i * p..i * p + i + 1].copy_from_slice(&prefix.l[i * start..i * start + i + 1]);
            }
            f[..start].copy_from_slice(&prefix.f);
        }
        for i in start..p {
            let zi = design.column(i);
            for j in 0..=i {
                let mut s = dot(zi, design.column(j));
                if i == j {
                    s += design.lambda;
                }
                for k in 0..j {
                    s -= l[i * p + k] * l[j * p + k];
                }
                if i == j {
                    // An overflowed Gram entry gives a +inf pivot, which would zero every
                    // leverage; it refuses like any pivot exact arithmetic cannot produce.
                    if !s.is_finite() || s < design.lambda / 2.0 {
                        return Err(Refusal::UnsafePivot {
                            index: i,
                            pivot: s,
                            lambda: design.lambda,
                        });
                    }
                    l[i * p + i] = s.sqrt();
                } else {
                    l[i * p + j] = s / l[j * p + j];
                    if !l[i * p + j].is_finite() {
                        return Err(Refusal::NonFinite("ridge factor entry overflowed"));
                    }
                }
            }
            let mut s = dot(zi, &design.centered);
            for k in 0..i {
                s -= l[i * p + k] * f[k];
            }
            f[i] = s / l[i * p + i];
            if !f[i].is_finite() {
                return Err(Refusal::NonFinite("ridge forward vector overflowed"));
            }
        }
        Ok(Factor {
            p,
            l,
            f,
            lambda: design.lambda,
        })
    }

    pub fn columns(&self) -> usize {
        self.p
    }

    /// The ridge coefficients, by [`crate::heads::cholesky_solve`]'s backward substitution.
    pub fn coefficients(&self) -> Vec<f64> {
        let p = self.p;
        let mut w = vec![0.0f64; p];
        for i in (0..p).rev() {
            let mut s = self.f[i];
            for (k, &later) in w.iter().enumerate().skip(i + 1) {
                s -= self.l[k * p + i] * later;
            }
            w[i] = s / self.l[i * p + i];
        }
        w
    }

    /// The minimum of `‖yc - Zw‖² + lambda ‖w‖²`: `ycᵀyc - ‖f‖²`.
    pub fn objective(&self, design: &Design) -> f64 {
        dot(&design.centered, &design.centered) - dot(&self.f, &self.f)
    }

    /// The decrease of the objective from the design's first `p_a` columns to all of them: the
    /// tail energy `sum_{k >= p_a} f_k²` of the forward vector (Schur complement identity).
    pub fn schur_gain(&self, p_a: usize) -> f64 {
        dot(&self.f[p_a.min(self.p)..], &self.f[p_a.min(self.p)..])
    }

    /// `L⁻¹ v` by forward substitution.
    fn forward(&self, v: &[f64]) -> Vec<f64> {
        let p = self.p;
        let mut out = vec![0.0f64; p];
        for i in 0..p {
            let mut s = v[i];
            for (k, &earlier) in out.iter().enumerate().take(i) {
                s -= self.l[i * p + k] * earlier;
            }
            out[i] = s / self.l[i * p + i];
        }
        out
    }
}

/// Leave-one-out residuals of a frozen design: each row predicted by the fit on the other rows,
/// with the same standardized rows, label mean and penalty.
#[derive(Clone, Debug, PartialEq)]
pub struct LeaveOneOut {
    /// `e_(i) = e_i / (1 - h_ii)`, the residual of row `i` under the fit without it.
    pub residuals: Vec<f64>,
    /// `1 - h_ii`, where `h_ii = ‖L⁻¹ z_i‖²` is row `i`'s ridge leverage.
    pub one_minus_h: Vec<f64>,
    /// The predicted residual sum of squares, `sum_i e_(i)²` (Allen 1974).
    pub press: f64,
}

/// [`LeaveOneOut`] by the Sherman-Morrison identity, refusing any row whose `1 - h_ii` is below
/// [`LEVERAGE_FLOOR`]. `factor` must be `design`'s own, which the caller guarantees as for
/// [`Factor::extend`]; only a factor of another width or penalty can be detected and refused.
pub fn leave_one_out(design: &Design, factor: &Factor) -> Result<LeaveOneOut, Refusal> {
    if factor.p != design.p || factor.lambda.to_bits() != design.lambda.to_bits() {
        return Err(Refusal::Shape(
            "leave-one-out needs a factor of the design's width and penalty",
        ));
    }
    let w = factor.coefficients();
    let (n, p) = (design.n, design.p);
    let mut residuals = Vec::with_capacity(n);
    let mut one_minus_h = Vec::with_capacity(n);
    let mut row = vec![0.0f64; p];
    for r in 0..n {
        for (j, value) in row.iter_mut().enumerate() {
            *value = design.z[j * n + r];
        }
        let fitted = dot(&row, &w);
        let projected = factor.forward(&row);
        let denominator = 1.0 - dot(&projected, &projected);
        if denominator.is_nan() || denominator < LEVERAGE_FLOOR {
            return Err(Refusal::Leverage {
                row: r,
                one_minus_h: denominator,
            });
        }
        residuals.push((design.centered[r] - fitted) / denominator);
        one_minus_h.push(denominator);
    }
    let press = dot(&residuals, &residuals);
    Ok(LeaveOneOut {
        residuals,
        one_minus_h,
        press,
    })
}

/// The leave-one-out residuals by `n` refits without each row: the reference for
/// [`leave_one_out`].
pub fn brute_force_leave_one_out(design: &Design) -> Result<Vec<f64>, Refusal> {
    let (n, p) = (design.n, design.p);
    if n < 2 {
        return Err(Refusal::Shape("leave-one-out needs at least two rows"));
    }
    (0..n)
        .map(|r| {
            let rest = design.without_row(r);
            let w = Factor::new(&rest)?.coefficients();
            let fitted: f64 = (0..p).map(|j| design.z[j * n + r] * w[j]).sum();
            Ok(design.centered[r] - fitted)
        })
        .collect()
}

/// `0.0 + sum_r a_r b_r`, rows ascending: the legacy accumulation of one entry.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for (&left, &right) in a.iter().zip(b) {
        sum += left * right;
    }
    sum
}

/// Compatibility alias for the domain-specific `RidgeRefusal` type.
pub type Refusal = RidgeRefusal;

#[cfg(test)]
mod tests;

/// B11 timing benches, both `#[ignore]`d. They are part of this module's unit tests, not a
/// separate benchmark target, and stay out of `tests/bench.rs`, which `legacy_v2_freeze` pins byte
/// for byte. Run: `cargo test --release -p feat-math --lib kernel::ridge_fast::bench -- --ignored
/// --nocapture`.
#[cfg(test)]
mod bench;
