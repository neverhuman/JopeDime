//! Fit-on-train standardization + mean-imputation for the dense (non-tree) learners. Fitting INSIDE a
//! learner's `train_single` (which only ever sees fold-train rows) makes it leak-safe by construction
//! (Invariant P), exactly like CatBoost's internal preprocessing. Column-major, NaN→mean (→0 after centering).

#[derive(Clone, Debug)]
pub struct StandardScaler {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl StandardScaler {
    /// Fit per-column mean/std over FINITE values of a column-major `n×p` matrix.
    pub fn fit(x: &[f32], n: usize, p: usize) -> Self {
        let mut mean = vec![0f32; p];
        let mut std = vec![1f32; p];
        for j in 0..p {
            let col = &x[j * n..j * n + n];
            let (mut s, mut c) = (0f64, 0usize);
            for &v in col {
                if v.is_finite() {
                    s += v as f64;
                    c += 1;
                }
            }
            let mu = if c > 0 { (s / c as f64) as f32 } else { 0.0 };
            let mut var = 0f64;
            for &v in col {
                if v.is_finite() {
                    let d = (v - mu) as f64;
                    var += d * d;
                }
            }
            let sd = if c > 1 {
                (var / c as f64).sqrt() as f32
            } else {
                1.0
            };
            mean[j] = mu;
            std[j] = if sd > 1e-9 { sd } else { 1.0 };
        }
        StandardScaler { mean, std }
    }

    /// Standardize a column-major `rows×p` matrix; NaN → 0 (the train mean after centering).
    pub fn apply_col_major(&self, x: &[f32], rows: usize, p: usize) -> Vec<f32> {
        let mut out = vec![0f32; rows * p];
        for j in 0..p {
            let (mu, sd) = (self.mean[j], self.std[j]);
            for i in 0..rows {
                let v = x[j * rows + i];
                out[j * rows + i] = if v.is_finite() { (v - mu) / sd } else { 0.0 };
            }
        }
        out
    }

    /// Standardize one row (length p) into `out` (hot path for kNN/MLP per-row inference).
    pub fn apply_row(&self, row: &[f32], out: &mut [f32]) {
        for (j, &v) in row.iter().enumerate() {
            out[j] = if v.is_finite() {
                (v - self.mean[j]) / self.std[j]
            } else {
                0.0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standardizes_and_imputes() {
        // 2 cols, 3 rows, column-major. col0=[0,2,4] (mean2,std~1.633), col1=[1,NaN,3] (mean2 over finite).
        let x = vec![0.0, 2.0, 4.0, 1.0, f32::NAN, 3.0];
        let sc = StandardScaler::fit(&x, 3, 2);
        assert!((sc.mean[0] - 2.0).abs() < 1e-6 && (sc.mean[1] - 2.0).abs() < 1e-6);
        let z = sc.apply_col_major(&x, 3, 2);
        // centered col0 mean ~0; the NaN in col1 maps to 0 (the mean).
        assert!(
            (z[3]/*row0 col1*/).abs() > 0.0 && z[4] == 0.0,
            "NaN -> 0 after centering: {z:?}"
        );
        let zmean0: f32 = (z[0] + z[1] + z[2]) / 3.0;
        assert!(zmean0.abs() < 1e-5);
    }
}
