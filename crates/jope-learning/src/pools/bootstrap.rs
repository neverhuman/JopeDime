//! Parent-cluster paired intervals, following the merged GB-07 `judge/cluster.py` contract:
//! cuts that share a parent are one cluster, the percentile interval resamples parents with
//! replacement and a drawn parent contributes all of its contexts once per draw. The draws use
//! SplitMix64 rather than Python's Mersenne Twister, so intervals match that module in method,
//! not bit for bit. A CI that contains or touches zero is INCONCLUSIVE, as in `ablations/compare.py`.

use anyhow::{ensure, Result};
use serde::Serialize;

/// Two-sided 95 % normal quantile, as `judge/power.py` `Z_ALPHA_95`.
pub const Z_ALPHA_95: f64 = 1.959_963_984_540_054;

/// Deterministic, platform-independent generator for bootstrap indices.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Performs the next u64 operation.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Unbiased draw from `0..n` by rejection.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "empty range");
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % n;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
/// `ContextInterval` is a record for the context interval contract in this module: Parent-cluster paired intervals, following the merged GB-07 judge/cluster.py contract: cuts that share a parent are one cluster, the percentile interval resamples parents with replacement and a drawn parent.
pub struct ContextInterval {
    /// Number of n represented here.
    pub n: usize,
    /// Mean recorded for this context interval.
    pub mean: f64,
    /// Sd recorded for this context interval.
    pub sd: f64,
    /// Se recorded for this context interval.
    pub se: f64,
    /// Low recorded for this context interval.
    pub low: f64,
    /// High recorded for this context interval.
    pub high: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
/// `ClusterBootstrap` is a record for the cluster bootstrap contract in this module: Parent-cluster paired intervals, following the merged GB-07 judge/cluster.py contract: cuts that share a parent are one cluster, the percentile interval resamples parents with replacement and a drawn parent.
pub struct ClusterBootstrap {
    /// Number of draws included in the calculation.
    pub draws: usize,
    /// Seed recorded for this cluster bootstrap.
    pub seed: u64,
    /// Alpha recorded for this cluster bootstrap.
    pub alpha: f64,
    /// Method recorded for this cluster bootstrap.
    pub method: &'static str,
    /// Low recorded for this cluster bootstrap.
    pub low: f64,
    /// High recorded for this cluster bootstrap.
    pub high: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
/// `PairedInterval` is a record for the paired interval contract in this module: Parent-cluster paired intervals, following the merged GB-07 judge/cluster.py contract: cuts that share a parent are one cluster, the percentile interval resamples parents with replacement and a drawn parent.
pub struct PairedInterval {
    /// N contexts recorded for this paired interval.
    pub n_contexts: usize,
    /// N clusters recorded for this paired interval.
    pub n_clusters: usize,
    /// Mean recorded for this paired interval.
    pub mean: f64,
    /// Context recorded for this paired interval.
    pub context: ContextInterval,
    /// Cluster bootstrap recorded for this paired interval.
    pub cluster_bootstrap: ClusterBootstrap,
    /// Verdict recorded for this paired interval.
    pub verdict: Verdict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
/// `Verdict` is an enum for the verdict contract in this module: Parent-cluster paired intervals, following the merged GB-07 judge/cluster.py contract: cuts that share a parent are one cluster, the percentile interval resamples parents with replacement and a drawn parent.
pub enum Verdict {
    /// The `Greater` case of `Verdict`.
    Greater,
    /// The `Less` case of `Verdict`.
    Less,
    /// The `Inconclusive` case of `Verdict`.
    Inconclusive,
}

impl Verdict {
    /// Strict: a bound that touches zero is INCONCLUSIVE.
    pub fn from_interval(low: f64, high: f64) -> Self {
        if low > 0.0 {
            Self::Greater
        } else if high < 0.0 {
            Self::Less
        } else {
            Self::Inconclusive
        }
    }
}

fn context_interval(values: &[f64]) -> Result<ContextInterval> {
    let n = values.len();
    ensure!(n >= 2, "need at least two paired contexts");
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    Ok(ContextInterval {
        n,
        mean,
        sd: var.sqrt(),
        se,
        low: mean - Z_ALPHA_95 * se,
        high: mean + Z_ALPHA_95 * se,
    })
}

/// Percentile CI of the context-weighted mean, resampling parent clusters with replacement.
/// `groups` holds the paired differences of each parent's contexts. The verdict reads the
/// cluster interval; the independent-context interval is reported beside it, never instead.
pub fn cluster_bootstrap_ci(
    groups: &[Vec<f64>],
    draws: usize,
    seed: u64,
    alpha: f64,
) -> Result<PairedInterval> {
    ensure!(draws >= 1, "draws must be positive");
    ensure!(
        !groups.is_empty() && groups.iter().all(|g| !g.is_empty()),
        "every cluster needs at least one paired difference"
    );
    ensure!(
        groups.iter().flatten().all(|v| v.is_finite()),
        "paired differences must be finite"
    );
    ensure!(
        alpha.is_finite() && alpha > 0.0 && alpha < 1.0,
        "alpha must be between 0 and 1"
    );
    let pooled: Vec<f64> = groups.iter().flatten().copied().collect();
    let sums: Vec<f64> = groups.iter().map(|g| g.iter().sum()).collect();
    let counts: Vec<usize> = groups.iter().map(Vec::len).collect();
    let g = groups.len();
    let mut rng = SplitMix64::new(seed);
    let mut means = Vec::with_capacity(draws);
    for _ in 0..draws {
        let (mut num, mut den) = (0.0, 0usize);
        for _ in 0..g {
            let i = rng.below(g as u64) as usize;
            num += sums[i];
            den += counts[i];
        }
        means.push(num / den as f64);
    }
    means.sort_by(f64::total_cmp);
    let at = |pos: f64| {
        let i = pos.floor() as usize;
        let j = (i + 1).min(draws - 1);
        let w = pos - i as f64;
        means[i] * (1.0 - w) + means[j] * w
    };
    let low = at((alpha / 2.0) * (draws - 1) as f64);
    let high = at((1.0 - alpha / 2.0) * (draws - 1) as f64);
    Ok(PairedInterval {
        n_contexts: pooled.len(),
        n_clusters: g,
        mean: pooled.iter().sum::<f64>() / pooled.len() as f64,
        context: context_interval(&pooled)?,
        cluster_bootstrap: ClusterBootstrap {
            draws,
            seed,
            alpha,
            method: "percentile_parent_resample",
            low,
            high,
        },
        verdict: Verdict::from_interval(low, high),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference_vector() {
        // First outputs of SplitMix64 seeded with 0 (Vigna's reference implementation).
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn identical_groups_have_a_point_interval() {
        let groups = vec![vec![0.25, 0.25], vec![0.25], vec![0.25, 0.25, 0.25]];
        let ci = cluster_bootstrap_ci(&groups, 500, 7, 0.05).unwrap();
        assert_eq!(ci.n_contexts, 6);
        assert_eq!(ci.n_clusters, 3);
        assert!((ci.cluster_bootstrap.low - 0.25).abs() < 1e-15);
        assert!((ci.cluster_bootstrap.high - 0.25).abs() < 1e-15);
        assert_eq!(ci.verdict, Verdict::Greater);
    }

    #[test]
    fn a_touching_bound_is_inconclusive() {
        assert_eq!(Verdict::from_interval(0.0, 1.0), Verdict::Inconclusive);
        assert_eq!(Verdict::from_interval(-1.0, 0.0), Verdict::Inconclusive);
        assert_eq!(Verdict::from_interval(-2.0, -1.0), Verdict::Less);
    }

    #[test]
    fn cluster_interval_is_wider_than_context_interval_for_correlated_parents() {
        // Two parents with strongly shared effects: the parent resample must not shrink below
        // the spread of the parent means.
        let groups = vec![vec![1.0; 20], vec![-0.8; 20], vec![0.9; 20], vec![-0.7; 20]];
        let ci = cluster_bootstrap_ci(&groups, 2000, 20_261_013, 0.05).unwrap();
        let cluster_width = ci.cluster_bootstrap.high - ci.cluster_bootstrap.low;
        let context_width = ci.context.high - ci.context.low;
        assert!(
            cluster_width > context_width,
            "{cluster_width} <= {context_width}"
        );
        assert_eq!(ci.verdict, Verdict::Inconclusive);
    }

    #[test]
    fn refuses_bad_inputs() {
        assert!(cluster_bootstrap_ci(&[], 10, 1, 0.05).is_err());
        assert!(cluster_bootstrap_ci(&[vec![]], 10, 1, 0.05).is_err());
        assert!(cluster_bootstrap_ci(&[vec![f64::NAN, 1.0]], 10, 1, 0.05).is_err());
        assert!(cluster_bootstrap_ci(&[vec![1.0, 2.0]], 0, 1, 0.05).is_err());
        assert!(cluster_bootstrap_ci(&[vec![1.0, 2.0]], 10, 1, 1.0).is_err());
    }
}
