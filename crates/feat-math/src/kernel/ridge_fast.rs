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
pub enum Refusal {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heads::{cholesky_solve, RidgeModel};
    use crate::Pcg32;

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Bits, with every NaN mapped to one value: NaN positions are kept, while a NaN result's sign
    /// and payload, which Rust leaves unspecified, are not compared.
    fn canonical(values: &[f64]) -> Vec<u64> {
        values
            .iter()
            .map(|v| if v.is_nan() { u64::MAX } else { v.to_bits() })
            .collect()
    }

    /// The oracle: `RidgeModel::fit` and `std_fit` as they stood before B11 (main e043aebfd),
    /// verbatim apart from names: per-column strided moments and one rank-one update of the
    /// whole Gram triangle per row.
    fn legacy_fit(x: &Rows, y: &[f64], lambda_frac: f64) -> RidgeModel {
        let n = x.len().min(y.len());
        let p = x[0].len();
        let (means, scales) = {
            let n = x.len().max(1) as f64;
            let mut means = vec![0.0; p];
            let mut scales = vec![1.0; p];
            for j in 0..p {
                let mut s = 0.0;
                for r in x {
                    s += r[j];
                }
                means[j] = s / n;
                let mut v = 0.0;
                for r in x {
                    let d = r[j] - means[j];
                    v += d * d;
                }
                scales[j] = (v / n).sqrt().max(1e-9);
            }
            (means, scales)
        };
        let lambda = (lambda_frac * n as f64).max(1e-3);
        let y_mean = y[..n].iter().sum::<f64>() / n.max(1) as f64;
        let mut ata = vec![0.0f64; p * p];
        let mut atb = vec![0.0f64; p];
        let mut xs_row = vec![0.0f64; p];
        for r in 0..n {
            for j in 0..p {
                xs_row[j] = ((x[r][j] - means[j]) / scales[j]).clamp(-20.0, 20.0);
            }
            let yr = y[r] - y_mean;
            for i in 0..p {
                let xi = xs_row[i];
                atb[i] += xi * yr;
                for j in i..p {
                    ata[i * p + j] += xi * xs_row[j];
                }
            }
        }
        for i in 0..p {
            ata[i * p + i] += lambda;
            for j in 0..i {
                ata[i * p + j] = ata[j * p + i];
            }
        }
        let coeffs = cholesky_solve(&mut ata, &mut atb, p);
        let mut w = Vec::with_capacity(p + 1);
        w.push(y_mean);
        w.extend(coeffs);
        RidgeModel { means, scales, w }
    }

    fn assert_identical(x: &Rows, y: &[f64], lambda_frac: f64) {
        let legacy = legacy_fit(x, y, lambda_frac);
        let fast = RidgeModel::fit(x, y, lambda_frac);
        assert_eq!(bits(&std_fit(x).0), bits(&legacy.means), "std_fit means");
        assert_eq!(bits(&std_fit(x).1), bits(&legacy.scales), "std_fit scales");
        assert_eq!(bits(&fast.means), bits(&legacy.means), "means");
        assert_eq!(bits(&fast.scales), bits(&legacy.scales), "scales");
        assert_eq!(bits(&fast.w), bits(&legacy.w), "weights");
        let probe: Rows = x.iter().rev().take(7).cloned().collect();
        assert_eq!(bits(&fast.predict(&probe)), bits(&legacy.predict(&probe)));
    }

    /// Gaussian columns mixed with the hard kinds: constant, heavy-tailed, near-duplicate,
    /// integer-valued and signed zeros. A finite column standardizes to |z| ≤ √(n − 1), so designs
    /// of at most 300 rows stay inside the ±20 clamp. The 500-row design of
    /// `a_design_that_binds_the_clamp_matches_the_legacy_fit` passes it, and so do the ignored
    /// `kernel_benchmark`'s 3333-row designs.
    fn design(seed: u64, n: usize, p: usize) -> (Rows, Vec<f64>) {
        let mut rng = Pcg32::new(seed, 11);
        let x: Rows = (0..n)
            .map(|r| {
                let base = rng.gaussian();
                (0..p)
                    .map(|j| match j % 6 {
                        0 => 3.5,
                        1 => (3.0 * rng.gaussian()).exp(),
                        2 => base,
                        3 => base + 1e-12 * rng.gaussian(),
                        4 => f64::from((r % 7) as u32) - 3.0,
                        _ => {
                            if r % 3 == 0 {
                                -0.0
                            } else {
                                rng.gaussian()
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        let y = x
            .iter()
            .map(|row| row.iter().take(4).sum::<f64>() * 0.3 + rng.gaussian())
            .collect();
        (x, y)
    }

    #[test]
    fn fast_fit_is_the_legacy_fit_bit_for_bit() {
        for (seed, n, p) in [
            (1, 1, 1),
            (2, 2, 3),
            (3, 7, 5),
            (4, 64, 17),
            (5, 300, 48),
            (6, 64, 200),
            (7, 128, 131),
        ] {
            let (x, y) = design(seed, n, p);
            for lambda_frac in [0.02, 0.2, 1e-9] {
                assert_identical(&x, &y, lambda_frac);
            }
        }
    }

    #[test]
    fn a_design_that_binds_the_clamp_matches_the_legacy_fit() {
        // A value that dominates its column standardizes to nearly ±√(n − 1), Samuelson's bound:
        // ≈ 22.3 for 500 rows, past the ±20 clamp, which both fits then apply. One positive and
        // one negative plant bind both bounds.
        let (mut x, y) = design(8, 500, 13);
        x[0][1] = 1e12;
        x[1][7] = -1e12;
        let (means, scales) = std_fit(&x);
        let high = (x[0][1] - means[1]) / scales[1];
        let low = (x[1][7] - means[7]) / scales[7];
        assert!(
            high > 20.0,
            "the positive plant must pass the clamp, z = {high}"
        );
        assert!(
            low < -20.0,
            "the negative plant must pass the clamp, z = {low}"
        );
        for lambda_frac in [0.02, 0.2, 1e-9] {
            assert_identical(&x, &y, lambda_frac);
        }
        // The certified design standardizes through the same clamp: its factor solves to the
        // fit's weights bit for bit, which fails if either side drops either bound. PRESS still
        // matches refits.
        let d = Design::from_rows(&x, &y, 0.02).unwrap();
        let factor = Factor::new(&d).unwrap();
        let fit = RidgeModel::fit(&x, &y, 0.02);
        assert_eq!(bits(&factor.coefficients()), bits(&fit.w[1..]));
        let loo = leave_one_out(&d, &factor).unwrap();
        let reference = brute_force_leave_one_out(&d).unwrap();
        let scale = reference.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        for (fast, slow) in loo.residuals.iter().zip(&reference) {
            assert!((fast - slow).abs() <= 1e-10 * scale);
        }
    }

    #[test]
    fn more_rows_than_labels_and_short_label_sets_match() {
        // The moments use every row of `x`; the normal equations only the first min(x, y) rows.
        let (x, y) = design(8, 40, 9);
        assert_identical(&x, &y[..25], 0.02);
        let (x, y) = design(9, 12, 6);
        let mut longer = y.clone();
        longer.extend([5.0, -5.0]);
        assert_identical(&x, &longer, 0.02);
    }

    /// Kernel micro-benchmark, timing only: `cargo test -p feat-math --release --lib
    /// ridge_fast::tests::kernel_benchmark -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn kernel_benchmark() {
        use std::time::Instant;
        for p in [16, 64, 256, 512] {
            let (x, y) = design(42, 3333, p);
            assert_identical(&x, &y, 0.02);
            let repeats = (2_000_000 / (p * p)).clamp(3, 200);
            let time = |f: &dyn Fn() -> RidgeModel| {
                let started = Instant::now();
                for _ in 0..repeats {
                    std::hint::black_box(f());
                }
                started.elapsed().as_secs_f64() / repeats as f64
            };
            let legacy = time(&|| legacy_fit(&x, &y, 0.02));
            let fast = time(&|| RidgeModel::fit(&x, &y, 0.02));
            println!(
                "B11_KERNEL n=3333 p={p} repeats={repeats} legacy_ms={:.3} fast_ms={:.3} speedup={:.2}",
                legacy * 1e3,
                fast * 1e3,
                legacy / fast
            );
        }
    }

    #[test]
    fn the_factor_reproduces_the_legacy_solve_and_resumes_bit_for_bit() {
        for (seed, n, p) in [(21, 7, 5), (22, 64, 17), (23, 300, 48), (24, 64, 131)] {
            let (x, y) = design(seed, n, p);
            let legacy = RidgeModel::fit(&x, &y, 0.02);
            let d = Design::from_rows(&x, &y, 0.02).unwrap();
            let full = Factor::new(&d).unwrap();
            assert_eq!(bits(&full.coefficients()), bits(&legacy.w[1..]));
            for p_a in [0, 1, p / 2, p - 1, p] {
                let prefix = Factor::new(&d.leading(p_a)).unwrap();
                assert_eq!(Factor::extend(&prefix, &d).unwrap(), full, "p_a={p_a}");
                // Two steps through an intermediate width give the same factor.
                let middle = Factor::extend(&prefix, &d.leading((p_a + p) / 2)).unwrap();
                assert_eq!(Factor::extend(&middle, &d).unwrap(), full);
            }
        }
    }

    /// Gaussian standardized columns with labels driven by the first ones.
    fn columns(seed: u64, n: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Pcg32::new(seed, 13);
        let z: Vec<f64> = (0..n * p).map(|_| rng.gaussian()).collect();
        let y = (0..n)
            .map(|r| z[r] * 1.5 - z[n + r] * 0.5 + 0.3 * rng.gaussian())
            .collect();
        (z, y)
    }

    #[test]
    fn schur_gain_is_the_objective_decrease_of_an_appended_block() {
        for (seed, n, p_a, p_b, lambda) in [
            (31, 50, 3, 1, 1.0),
            (32, 200, 20, 5, 4.0),
            (33, 200, 40, 24, 1e-3),
            (34, 50, 0, 6, 1.0),
        ] {
            let p = p_a + p_b;
            let (z, y) = columns(seed, n, p);
            let d = Design::from_columns(z, n, p, &y, lambda).unwrap();
            let full = Factor::new(&d).unwrap();
            let lead = Factor::new(&d.leading(p_a)).unwrap();
            let gain = full.schur_gain(p_a);
            assert!(gain >= 0.0);
            // Both objectives from explicit coefficients and residuals, independently of the
            // forward vector the gain is read from.
            let explicit = |factor: &Factor| {
                let w = factor.coefficients();
                let mut value = lambda * dot(&w, &w);
                for r in 0..n {
                    let fitted: f64 = (0..w.len()).map(|j| d.z[j * n + r] * w[j]).sum();
                    value += (d.centered[r] - fitted).powi(2);
                }
                value
            };
            let (lead_value, full_value) = (explicit(&lead), explicit(&full));
            let decrease = lead_value - full_value;
            assert!(
                (decrease - gain).abs() <= 1e-10 * lead_value,
                "seed {seed}: {decrease} vs {gain}"
            );
            assert!((full_value - full.objective(&d)).abs() <= 1e-10 * full_value);
            assert!((lead_value - lead.objective(&d)).abs() <= 1e-10 * lead_value);
        }
        // An all-zero appended block gains exactly nothing.
        let (mut z, y) = columns(35, 40, 3);
        z.extend(std::iter::repeat_n(0.0, 40 * 2));
        let d = Design::from_columns(z, 40, 5, &y, 0.8).unwrap();
        assert_eq!(Factor::new(&d).unwrap().schur_gain(3), 0.0);
    }

    #[test]
    fn press_matches_brute_force_leave_one_out() {
        for (seed, n, p, lambda) in [
            (41, 12, 3, 0.24),
            (42, 40, 8, 0.8),
            (43, 128, 16, 2.56),
            (44, 96, 48, 1.92),
        ] {
            let (z, y) = columns(seed, n, p);
            let d = Design::from_columns(z, n, p, &y, lambda).unwrap();
            let loo = leave_one_out(&d, &Factor::new(&d).unwrap()).unwrap();
            let reference = brute_force_leave_one_out(&d).unwrap();
            let scale = reference.iter().fold(0.0f64, |m, e| m.max(e.abs()));
            for (fast, slow) in loo.residuals.iter().zip(&reference) {
                assert!((fast - slow).abs() <= 1e-10 * scale, "seed {seed}");
            }
            let press: f64 = reference.iter().map(|e| e * e).sum();
            assert!((loo.press - press).abs() <= 1e-10 * press);
        }
        // The legacy surrogate design, with its heavy-tailed columns, agrees as well.
        let (x, y) = design(45, 60, 12);
        let d = Design::from_rows(&x, &y, 0.02).unwrap();
        let loo = leave_one_out(&d, &Factor::new(&d).unwrap()).unwrap();
        let reference = brute_force_leave_one_out(&d).unwrap();
        let scale = reference.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        for (fast, slow) in loo.residuals.iter().zip(&reference) {
            assert!((fast - slow).abs() <= 1e-10 * scale);
        }
    }

    #[test]
    fn unsafe_pivots_high_leverage_and_bad_designs_are_refused() {
        // Two huge near-identical columns: the second pivot's roundoff dwarfs the penalty.
        let n = 30;
        let mut rng = Pcg32::new(51, 1);
        let base: Vec<f64> = (0..n).map(|_| 1e8 * rng.gaussian()).collect();
        let mut z = base.clone();
        z.extend(base.iter().map(|v| v + 1e-8));
        let d = Design::from_columns(z, n, 2, &vec![1.0; n], 1e-3).unwrap();
        assert!(matches!(
            Factor::new(&d),
            Err(Refusal::UnsafePivot { index: 1, .. })
        ));
        // One dominant row under a tiny penalty has leverage near one.
        let mut z = vec![0.01; 6];
        z[0] = 1e4;
        let d = Design::from_columns(z, 6, 1, &[3.0, 0.0, 1.0, 0.0, 1.0, 0.0], 1e-6).unwrap();
        assert!(matches!(
            leave_one_out(&d, &Factor::new(&d).unwrap()),
            Err(Refusal::Leverage { row: 0, .. })
        ));
        assert!(matches!(
            Design::from_columns(vec![1.0; 5], 2, 2, &[1.0, 2.0], 1.0),
            Err(Refusal::Shape(_))
        ));
        assert!(matches!(
            Design::from_columns(vec![f64::NAN, 1.0], 2, 1, &[1.0, 2.0], 1.0),
            Err(Refusal::NonFinite(_))
        ));
        assert!(matches!(
            Design::from_columns(vec![1.0, 2.0], 2, 1, &[1.0, 2.0], 0.0),
            Err(Refusal::NonFinite(_))
        ));
        // A prefix with a different penalty or more columns cannot be resumed.
        let (z, y) = columns(52, 20, 4);
        let d = Design::from_columns(z.clone(), 20, 4, &y, 1.0).unwrap();
        let other = Design::from_columns(z, 20, 4, &y, 2.0).unwrap();
        let prefix = Factor::new(&other.leading(2)).unwrap();
        assert!(matches!(
            Factor::extend(&prefix, &d),
            Err(Refusal::Shape(_))
        ));
        let wide = Factor::new(&d).unwrap();
        assert!(matches!(
            Factor::extend(&wide, &d.leading(2)),
            Err(Refusal::Shape(_))
        ));
        // Leave-one-out needs the design's own factor, not a narrower one or another penalty's.
        assert!(matches!(
            leave_one_out(&d, &Factor::new(&d.leading(2)).unwrap()),
            Err(Refusal::Shape(_))
        ));
        assert!(matches!(
            leave_one_out(&d, &Factor::new(&other).unwrap()),
            Err(Refusal::Shape(_))
        ));
        // A penalty under 2e-12 could accept a pivot the legacy solve replaces, so it refuses.
        assert!(matches!(
            Design::from_columns(vec![1.0, 2.0], 2, 1, &[1.0, 2.0], 1e-12),
            Err(Refusal::NonFinite(_))
        ));
        // The legacy design refuses a non-finite label and an empty table.
        let (x, mut y) = design(53, 6, 3);
        y[2] = f64::INFINITY;
        assert!(matches!(
            Design::from_rows(&x, &y, 0.02),
            Err(Refusal::NonFinite(_))
        ));
        assert!(matches!(
            Design::from_rows(&x, &[], 0.02),
            Err(Refusal::Shape(_))
        ));
        // Finite columns whose Gram entry overflows: the +inf pivot refuses instead of zeroing
        // every leverage.
        let d = Design::from_columns(vec![1e160, 1e160], 2, 1, &[1.0, 2.0], 1.0).unwrap();
        assert!(matches!(
            Factor::new(&d),
            Err(Refusal::UnsafePivot { index: 0, .. })
        ));
        // A finite pivot whose right-hand side overflows refuses too.
        let d = Design::from_columns(vec![1.0, -1.0], 2, 1, &[1e308, -1e308], 1.0).unwrap();
        assert!(matches!(Factor::new(&d), Err(Refusal::NonFinite(_))));
        // Labels whose centering overflows refuse at construction.
        assert!(matches!(
            Design::from_columns(vec![1.0, 2.0], 2, 1, &[f64::MAX, f64::MAX], 1.0),
            Err(Refusal::NonFinite(_))
        ));
        // A ragged table refuses instead of panicking, and n x p cannot overflow past the check.
        let ragged: Rows = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(matches!(
            Design::from_rows(&ragged, &[1.0, 2.0], 0.02),
            Err(Refusal::Shape(_))
        ));
        assert!(matches!(
            Design::from_columns(Vec::new(), usize::MAX, 2, &[], 1.0),
            Err(Refusal::Shape(_))
        ));
    }

    #[test]
    fn overflowing_and_non_finite_inputs_keep_the_legacy_bits_and_nan_positions() {
        let base = |r: usize| -> Vec<f64> { (0..4).map(|j| ((r * 3 + j) % 5) as f64).collect() };
        let labels: Vec<f64> = (0..9).map(|r| r as f64 * 0.7 - 2.0).collect();
        let table = || -> Rows { (0..9).map(base).collect() };
        let mut cases: Vec<(Rows, Vec<f64>)> = Vec::new();
        // A column whose sum overflows, and one whose squared deviations overflow.
        let mut x = table();
        x.iter_mut().for_each(|row| row[1] = f64::MAX);
        cases.push((x, labels.clone()));
        let mut x = table();
        for (r, row) in x.iter_mut().enumerate() {
            row[2] = if r % 2 == 0 { f64::MAX } else { -f64::MAX };
        }
        cases.push((x, labels.clone()));
        // A single +inf and a NaN cell.
        let mut x = table();
        x[8][0] = f64::INFINITY;
        cases.push((x, labels.clone()));
        let mut x = table();
        x[4][3] = f64::NAN;
        cases.push((x, labels.clone()));
        // Non-finite labels.
        let mut y = labels.clone();
        y[3] = f64::NEG_INFINITY;
        cases.push((table(), y));
        let mut y = labels.clone();
        y[5] = f64::NAN;
        cases.push((table(), y));
        for (index, (x, y)) in cases.iter().enumerate() {
            let legacy = legacy_fit(x, y, 0.02);
            let fast = RidgeModel::fit(x, y, 0.02);
            // Each case really leaves the finite range, so it cannot pass as a finite one.
            assert!(
                legacy
                    .means
                    .iter()
                    .chain(&legacy.scales)
                    .chain(&legacy.w)
                    .any(|v| !v.is_finite()),
                "case {index} stays finite"
            );
            assert_eq!(
                canonical(&fast.means),
                canonical(&legacy.means),
                "case {index}"
            );
            assert_eq!(
                canonical(&fast.scales),
                canonical(&legacy.scales),
                "case {index}"
            );
            assert_eq!(canonical(&fast.w), canonical(&legacy.w), "case {index}");
            let probe: Rows = x.iter().rev().take(4).cloned().collect();
            assert_eq!(
                canonical(&fast.predict(&probe)),
                canonical(&legacy.predict(&probe)),
                "case {index}"
            );
        }
    }

    #[test]
    fn zero_variance_and_huge_constant_columns_match_the_legacy_fit() {
        // Constant, 1e300 and signed-zero columns standardize to (±)0. The penalty floor keeps
        // every pivot at lambda or more, so `cholesky_solve`'s 1e-12 replacement branch (shared
        // code on both sides) is not reached here; `Factor` refuses any pivot that would reach it.
        let x: Rows = (0..5).map(|_| vec![2.0, 2.0, 2.0]).collect();
        assert_identical(&x, &[1.0, 2.0, 3.0, 4.0, 5.0], 1e-12);
        let x: Rows = (0..9).map(|r| vec![f64::from(r), 1e300, -0.0]).collect();
        assert_identical(&x, &[0.5; 9], 0.02);
    }
}

/// B11 timing benches, both `#[ignore]`d. They are part of this module's unit tests, not a
/// separate benchmark target, and stay out of `tests/bench.rs`, which `legacy_v2_freeze` pins byte
/// for byte. Run: `cargo test --release -p feat-math --lib kernel::ridge_fast::bench -- --ignored
/// --nocapture`.
#[cfg(test)]
mod bench {

    use crate::dsl::{ParamValue, Stage};
    use crate::evolve::{
        evolve, random_canonical_pipeline, EvolveConfig, GenStats, Observer, RidgeScoring, Scored,
        HEAD_OPS,
    };
    use crate::heads::RidgeModel;
    use crate::ops::{apply_op_train, fit_op};
    use crate::transpile::Emitted;
    use crate::{GenomeIr, Pcg32, Rows};
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    /// The pinned 5000-row B11 fixture: twelve gaussian columns, two heavy-tailed columns that
    /// activate the ridge clamp, one integer column and one near-constant column.
    fn ridge_fixture() -> (Rows, Vec<f64>) {
        let mut rng = Pcg32::new(0xB11, 11);
        let x: Rows = (0..5000)
            .map(|_| {
                let mut row: Vec<f64> = (0..12).map(|_| rng.gaussian()).collect();
                row.push((2.0 * rng.gaussian()).exp());
                row.push((2.0 * rng.gaussian()).exp());
                row.push(f64::from(rng.next_u32() % 7));
                row.push(1.0 + 1e-9 * rng.gaussian());
                row
            })
            .collect();
        let y = x
            .iter()
            .map(|r| {
                1.5 * r[0] - 0.8 * r[1] * r[1]
                    + 0.6 * (2.0 * r[2]).sin()
                    + 0.4 * r[3] * r[4]
                    + 0.1 * rng.gaussian()
            })
            .collect();
        (x, y)
    }

    /// Where a surrogate fold fit spends its time. Mirrors the search surrogate on random generated
    /// genomes: transforms fitted and applied on a two-thirds training split, then the ridge head on
    /// the final frame.
    #[test]
    #[ignore]
    fn ridge_share_of_surrogate_fits() {
        let (x, y) = ridge_fixture();
        let train: Rows = x[..3333].to_vec();
        let target = &y[..3333];
        let (mut transforms, mut ridge, mut widths) = (0.0f64, 0.0f64, Vec::new());
        for seed in 0..120u64 {
            let GenomeIr::Pipeline(mut genome) =
                GenomeIr::parse(&random_canonical_pipeline(seed, 6)).unwrap()
            else {
                continue;
            };
            // The search surrogate: knobs capped as the search caps them, the head a plain ridge.
            for stage in &mut genome.stages {
                for (key, cap) in [("degree", 4i64), ("n", 64)] {
                    if let Some(value) = stage.params.get_mut(key) {
                        if value.as_u64().is_some_and(|v| v as i64 > cap) {
                            *value = ParamValue::Int(cap);
                        }
                    }
                }
            }
            if genome
                .stages
                .last()
                .is_some_and(|last| HEAD_OPS.contains(&last.op.as_str()))
            {
                genome.stages.pop();
            }
            let stages: Vec<Stage> = genome.stages;
            let started = Instant::now();
            let mut frame = train.clone();
            crate::sanitize(&mut frame);
            let mut ok = true;
            for stage in &stages {
                match fit_op(stage, &frame, target)
                    .ok()
                    .and_then(|op| apply_op_train(&op, &frame).ok())
                {
                    Some(next) => frame = next,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            transforms += started.elapsed().as_secs_f64();
            let started = Instant::now();
            std::hint::black_box(RidgeModel::fit(&frame, target, 0.02));
            ridge += started.elapsed().as_secs_f64();
            widths.push(frame[0].len());
        }
        widths.sort_unstable();
        println!(
            "B11_SHARE genomes={} transforms_s={transforms:.3} ridge_s={ridge:.3} ridge_share={:.3} \
             width_median={} width_max={}",
            widths.len(),
            ridge / (ridge + transforms),
            widths[widths.len() / 2],
            widths[widths.len() - 1]
        );
    }

    /// Surrogate-search throughput on the pinned fixture. Prints wall time, evaluations per second
    /// and a digest of the ranked result, so that two builds can be compared for speed and for bit
    /// identity.
    #[test]
    #[ignore]
    fn ridge_fast_v1_throughput() {
        struct Evals(AtomicU64);
        impl Observer for Evals {
            fn on_generation(&self, stats: &GenStats) {
                self.0.store(stats.evals, Ordering::Relaxed);
            }
            fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _stats: &GenStats) {}
        }

        let (x, y) = ridge_fixture();
        let metric = |truth: &[f64], predicted: &[f64]| crate::metrics::r2(truth, predicted);
        // Three pairs, the order alternating per pair, so drifting host load hits both arms alike.
        for run in 0..6 {
            let scoring = if (run % 2 == 0) == (run / 2 % 2 == 0) {
                RidgeScoring::Refit
            } else {
                RidgeScoring::Certified
            };
            let cfg = EvolveConfig {
                pop: 64,
                gens: 8,
                folds: 3,
                seed: 0x1EA7_BEEF,
                max_transforms: 6,
                ridge_scoring: scoring,
                ..EvolveConfig::default()
            };
            let obs = Evals(AtomicU64::new(0));
            let started = Instant::now();
            let ranked = evolve(&x, &y, &metric, &cfg, &obs);
            let secs = started.elapsed().as_secs_f64();
            let mut digest = Sha256::new();
            for scored in &ranked {
                digest.update(scored.dsl.as_bytes());
                digest.update(scored.cv_mean.to_bits().to_le_bytes());
                for fold in &scored.fold_scores {
                    digest.update(fold.to_bits().to_le_bytes());
                }
            }
            let evals = obs.0.load(Ordering::Relaxed);
            println!(
                "B11_E2E run={run} scoring={scoring:?} secs={secs:.3} evals={evals} \
                 evals_per_sec={:.2} ranked={} digest={:x}",
                evals as f64 / secs,
                ranked.len(),
                digest.finalize()
            );
        }
    }
}
