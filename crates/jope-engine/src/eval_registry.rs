//! Durable native fit authority. Normal startup never creates storage.
//! Deployment also requires every numerical entry point and reviewed history.
mod authority;
use anyhow::{ensure, Result};
pub use authority::*;

/// Rank-softmax log-weights. Index zero is best.
pub fn logprob_from_rank(index: usize, n: usize) -> f64 {
    let z: f64 = (0..n).map(|i| (-(i as f64)).exp()).sum();
    -(index as f64) - z.ln()
}

/// Stable log-softmax of ranking scores (higher is better).
pub fn logprob_from_scores(scores: &[f64]) -> Result<Vec<f64>> {
    ensure!(!scores.is_empty(), "empty ranking scores");
    ensure!(
        scores.iter().all(|s| s.is_finite()),
        "nonfinite ranking score"
    );
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_z = scores.iter().map(|s| (s - max).exp()).sum::<f64>().ln();
    Ok(scores.iter().map(|s| s - max - log_z).collect())
}
