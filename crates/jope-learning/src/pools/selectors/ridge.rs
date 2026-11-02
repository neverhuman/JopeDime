use super::*;

/// Simple supervised predictor: ridge regression of the training percentile on program
/// features (with intercept), solved in closed form. Context-blind.
pub struct RidgeProgram<'a> {
    lambda: f64,
    program_features: &'a Features,
    weights: Vec<f64>,
}

/// Solve `A x = b` for symmetric positive definite `A` (row-major, `n x n`) by Cholesky.
pub(crate) fn cholesky_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Result<Vec<f64>> {
    ensure!(
        a.len() == n * n && b.len() == n && a.iter().chain(&b).all(|v| v.is_finite()),
        "nonfinite or malformed ridge system"
    );
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        ensure!(d > 0.0, "ridge system is not positive definite");
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    ensure!(b.iter().all(|v| v.is_finite()), "nonfinite ridge solution");
    Ok(b)
}

impl<'a> RidgeProgram<'a> {
    /// Performs the fit operation and returns an error if it cannot be completed.
    pub fn fit(train: &[&Pool], program_features: &'a Features, lambda: f64) -> Result<Self> {
        ensure!(
            lambda > 0.0 && lambda.is_finite(),
            "lambda must be positive"
        );
        let d = program_features.dim + 1;
        let (mut xtx, mut xty) = (vec![0.0; d * d], vec![0.0; d]);
        let mut rows = 0usize;
        for pool in train {
            for (m, t) in pool.members.iter().zip(rank_targets(pool)) {
                let phi = program_features
                    .rows
                    .get(&m.program_sha256)
                    .with_context(|| format!("no program features for {}", m.program_sha256))?;
                let x: Vec<f64> = std::iter::once(1.0).chain(phi.iter().copied()).collect();
                for i in 0..d {
                    xty[i] += x[i] * t;
                    for j in 0..d {
                        xtx[i * d + j] += x[i] * x[j];
                    }
                }
                rows += 1;
            }
        }
        ensure!(rows > 0, "ridge needs training rows");
        // The intercept is not penalized.
        for i in 1..d {
            xtx[i * d + i] += lambda;
        }
        xtx[0] += 1e-12;
        Ok(Self {
            lambda,
            program_features,
            weights: cholesky_solve(xtx, xty, d)?,
        })
    }
}

impl Selector for RidgeProgram<'_> {
    fn name(&self) -> &str {
        "ridge_program"
    }
    fn class(&self) -> SelectorClass {
        SelectorClass::Baseline
    }
    fn rule(&self) -> String {
        format!(
            "ridge_program/v1: closed-form ridge (lambda={}, unpenalized intercept) of training percentile on {} program features",
            self.lambda, self.program_features.dim
        )
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        view.programs
            .iter()
            .map(|p| {
                let phi = self
                    .program_features
                    .rows
                    .get(*p)
                    .with_context(|| format!("no program features for {p}"))?;
                Ok(self.weights[0]
                    + phi
                        .iter()
                        .zip(&self.weights[1..])
                        .map(|(x, w)| x * w)
                        .sum::<f64>())
            })
            .collect()
    }
}
