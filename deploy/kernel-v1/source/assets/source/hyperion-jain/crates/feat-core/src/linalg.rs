//! Hand-rolled dense f64 linear algebra (no BLAS dependency). Sizes are small: the stacker Gram is
//! ~M≤16, the RFF Gram is m≈128–1024, the BMA/GP solves are n<~1500. Everything is row-major and
//! symmetric-PD-oriented (ridge/Gram systems), so a plain Cholesky + triangular solves suffice. These are
//! the shared primitives for RFF-ridge, the ridge/NNLS stacker, the exact-GP small-n backend, and Platt
//! calibration.

/// In-place lower-triangular Cholesky factor of a symmetric positive-definite row-major `m×m` matrix.
/// On success `a` holds `L` (lower) with the strict upper triangle zeroed; returns false if not PD
/// (a non-positive pivot), so callers can add ridge and retry.
pub fn cholesky_factor(a: &mut [f64], m: usize) -> bool {
    for j in 0..m {
        let mut d = a[j * m + j];
        for k in 0..j {
            d -= a[j * m + k] * a[j * m + k];
        }
        if d <= 1e-12 {
            return false;
        }
        let ljj = d.sqrt();
        a[j * m + j] = ljj;
        for i in (j + 1)..m {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= a[i * m + k] * a[j * m + k];
            }
            a[i * m + j] = s / ljj;
        }
    }
    for i in 0..m {
        for j in (i + 1)..m {
            a[i * m + j] = 0.0;
        }
    }
    true
}

/// Solve `L Lᵀ x = b` in place on `b`, given `l` from `cholesky_factor` (forward then back substitution).
pub fn cholesky_solve(l: &[f64], m: usize, b: &mut [f64]) {
    // forward: L y = b
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * m + k] * b[k];
        }
        b[i] = s / l[i * m + i];
    }
    // back: Lᵀ x = y
    for i in (0..m).rev() {
        let mut s = b[i];
        for k in (i + 1)..m {
            s -= l[k * m + i] * b[k];
        }
        b[i] = s / l[i * m + i];
    }
}

/// Solve `(G + ridge·I) x = rhs` for a symmetric PD `G` (row-major `m×m`), in place on `g` and `rhs`.
/// Adds the ridge to the diagonal, factors, and solves. Returns false if even the ridged system is not PD.
pub fn spd_solve(g: &mut [f64], m: usize, rhs: &mut [f64], ridge: f64) -> bool {
    for i in 0..m {
        g[i * m + i] += ridge;
    }
    if !cholesky_factor(g, m) {
        return false;
    }
    cholesky_solve(g, m, rhs);
    true
}

/// Non-negative least squares `min ||A w − y||²  s.t. w ≥ 0` by coordinate descent on the precomputed
/// normal-equation pieces `gram = AᵀA` (row-major `m×m`) and `aty = Aᵀy`. Used by the non-negative ridge
/// stacker (the non-negativity is the key regularizer keeping the stacker off a noisy member).
pub fn nnls_cd(gram: &[f64], aty: &[f64], m: usize, iters: usize) -> Vec<f64> {
    let mut w = vec![0.0f64; m];
    for _ in 0..iters {
        for j in 0..m {
            let gjj = gram[j * m + j];
            if gjj <= 1e-12 {
                continue;
            }
            let mut num = aty[j];
            for k in 0..m {
                if k != j {
                    num -= gram[j * m + k] * w[k];
                }
            }
            w[j] = (num / gjj).max(0.0);
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_known_spd() {
        // A = [[4,2],[2,3]] (SPD). Solve A x = [1, 1] -> x = [1/8, 1/4].
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        let mut b = vec![1.0, 1.0];
        assert!(cholesky_factor(&mut a, 2));
        cholesky_solve(&a, 2, &mut b);
        assert!(
            (b[0] - 0.125).abs() < 1e-9 && (b[1] - 0.25).abs() < 1e-9,
            "{b:?}"
        );
    }

    #[test]
    fn spd_solve_with_ridge() {
        // (I*2) x = [2,4,6] -> x=[1,2,3]; here G=0, ridge=2.
        let mut g = vec![0.0; 9];
        let mut rhs = vec![2.0, 4.0, 6.0];
        assert!(spd_solve(&mut g, 3, &mut rhs, 2.0));
        for (i, &v) in rhs.iter().enumerate() {
            assert!((v - (i as f64 + 1.0)).abs() < 1e-9, "{rhs:?}");
        }
    }

    #[test]
    fn cholesky_rejects_non_pd() {
        let mut a = vec![1.0, 2.0, 2.0, 1.0]; // eigenvalues 3,-1 -> not PD
        assert!(!cholesky_factor(&mut a, 2));
    }

    #[test]
    fn nnls_matches_unconstrained_when_positive() {
        // A = I (2x2), y = [3, 5] -> w = [3,5], all non-negative.
        let gram = vec![1.0, 0.0, 0.0, 1.0];
        let aty = vec![3.0, 5.0];
        let w = nnls_cd(&gram, &aty, 2, 50);
        assert!(
            (w[0] - 3.0).abs() < 1e-6 && (w[1] - 5.0).abs() < 1e-6,
            "{w:?}"
        );
    }

    #[test]
    fn nnls_clamps_negative() {
        // Unconstrained solution would be negative on coord 0; NNLS must clamp it to 0.
        // A'A = [[1,0],[0,1]], A'y = [-2, 4] -> unconstrained [-2,4]; NNLS -> [0,4].
        let gram = vec![1.0, 0.0, 0.0, 1.0];
        let aty = vec![-2.0, 4.0];
        let w = nnls_cd(&gram, &aty, 2, 50);
        assert!(
            w[0] >= 0.0 && (w[0]).abs() < 1e-9 && (w[1] - 4.0).abs() < 1e-6,
            "{w:?}"
        );
    }
}
