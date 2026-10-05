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
//! - [`Refusal`]: every unsafe pivot or leverage refuses, so the caller refits with the legacy
//!   authority.
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
/// [`RidgeModel::fit`], whose result is always defined.
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    Shape(&'static str),
    NonFinite(&'static str),
    /// A pivot before its square root fell below `lambda / 2`; in exact arithmetic every pivot is
    /// at least `lambda`. This also covers the legacy `s <= 1e-12` replacement branch, because
    /// `lambda >= 1e-3`.
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
    /// The design [`fit`] solves for `x, y` at `lambda_frac`.
    pub fn legacy(x: &Rows, y: &[f64], lambda_frac: f64) -> Design {
        let n = x.len().min(y.len());
        let p = x[0].len();
        let (means, scales) = std_fit(x);
        let y_mean = y[..n].iter().sum::<f64>() / n.max(1) as f64;
        Design {
            n,
            p,
            z: standardized(x, n, p, &means, &scales),
            centered: y[..n].iter().map(|value| value - y_mean).collect(),
            lambda: (lambda_frac * n as f64).max(1e-3),
        }
    }

    /// A design from column-major standardized columns `z` (`n` rows, `p` columns), labels that
    /// are centered here at their mean, and an absolute penalty `lambda > 0`.
    pub fn from_columns(
        z: Vec<f64>,
        n: usize,
        p: usize,
        y: &[f64],
        lambda: f64,
    ) -> Result<Design, Refusal> {
        if n == 0 || z.len() != n * p || y.len() != n {
            return Err(Refusal::Shape(
                "ridge design needs n > 0, n x p columns and n labels",
            ));
        }
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(Refusal::NonFinite(
                "ridge penalty must be finite and positive",
            ));
        }
        if !z.iter().chain(y).all(|value| value.is_finite()) {
            return Err(Refusal::NonFinite("ridge design and labels must be finite"));
        }
        let mean = y.iter().sum::<f64>() / n as f64;
        Ok(Design {
            n,
            p,
            z,
            centered: y.iter().map(|value| value - mean).collect(),
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

/// `G = ZᵀZ + lambda I = L Lᵀ` in [`cholesky_solve`]'s operation order, with the forward vector
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
                    if s.is_nan() || s < design.lambda / 2.0 {
                        return Err(Refusal::UnsafePivot {
                            index: i,
                            pivot: s,
                            lambda: design.lambda,
                        });
                    }
                    l[i * p + i] = s.sqrt();
                } else {
                    l[i * p + j] = s / l[j * p + j];
                }
            }
            let mut s = dot(zi, &design.centered);
            for k in 0..i {
                s -= l[i * p + k] * f[k];
            }
            f[i] = s / l[i * p + i];
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

    /// The ridge coefficients, by [`cholesky_solve`]'s backward substitution.
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
/// [`LEVERAGE_FLOOR`].
pub fn leave_one_out(design: &Design, factor: &Factor) -> Result<LeaveOneOut, Refusal> {
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

    /// Gaussian columns mixed with the hard kinds: constant, heavy-tailed (the clamp is active),
    /// near-duplicate, integer-valued and signed zeros.
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
            let d = Design::legacy(&x, &y, 0.02);
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
            let decrease = lead.objective(&d) - full.objective(&d);
            let gain = full.schur_gain(p_a);
            assert!(gain >= 0.0);
            assert!(
                (decrease - gain).abs() <= 1e-10 * lead.objective(&d),
                "seed {seed}: {decrease} vs {gain}"
            );
            // The objective from explicit coefficients agrees with the forward-vector form.
            let w = full.coefficients();
            let mut explicit = lambda * dot(&w, &w);
            for r in 0..n {
                let fitted: f64 = (0..p).map(|j| d.z[j * n + r] * w[j]).sum();
                explicit += (d.centered[r] - fitted).powi(2);
            }
            assert!((explicit - full.objective(&d)).abs() <= 1e-10 * explicit);
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
        // The legacy surrogate design, with its clamped heavy-tailed columns, agrees as well.
        let (x, y) = design(45, 60, 12);
        let d = Design::legacy(&x, &y, 0.02);
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
    }

    #[test]
    fn degenerate_pivots_take_the_legacy_branch() {
        // A zero-variance design scales by 1e-9, and with the lambda floor every pivot is tiny.
        let x: Rows = (0..5).map(|_| vec![2.0, 2.0, 2.0]).collect();
        assert_identical(&x, &[1.0, 2.0, 3.0, 4.0, 5.0], 1e-12);
        let x: Rows = (0..9).map(|r| vec![f64::from(r), 1e300, -0.0]).collect();
        assert_identical(&x, &[0.5; 9], 0.02);
    }
}
