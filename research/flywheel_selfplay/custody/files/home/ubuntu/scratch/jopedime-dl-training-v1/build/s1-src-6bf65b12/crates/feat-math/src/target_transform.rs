//! Target-side transforms: the `(target kind=...)` pseudo-stage.
//!
//! A target transform is fitted on the TRAIN target only, the pipeline + head
//! are then fitted on the transformed target y', and predictions are mapped
//! back through the exact inverse before the standard ±60 clamp. It must be
//! the FIRST stage of a pipeline (model.rs enforces this); `cv_r2` refits it
//! inside every fold like any other fitted state, so it is leak-free in
//! honest CV.
//!
//! Every arithmetic step here has an exact twin in the emitted Python and Rust
//! artifacts (battle-transpile). Only parity-proven primitives are used:
//! total-order sorting, binary search (lower/upper bound), linear
//! interpolation, the shared Acklam probit, and plain ln/exp.
//!
//! Kinds:
//! - `rank_gauss` — normal-scores transform: y' = probit(mid_rank(y));
//!   inverse = piecewise-linear interpolation through the fitted
//!   (score, value) table, clamped to the extreme train values.
//! - `log1p`     — y' = ln(1 + max(y, -0.999999)); inverse = exp(z) - 1
//!   (argument clipped to [-745, 700] so both languages stay in range).
//!   Battle targets are [0,1]-scaled, so the floor is a no-op there.
//! - `winsor`    — y' = clip(y, q-quantile, (1-q)-quantile); the inverse is
//!   the identity. Winsorized targets bias extreme predictions toward the
//!   bulk of the target — that is the intended robustness trade.

use crate::dsl::{ParamValue, Stage};
use crate::ops::{lower_bound, mid_rank, probit, quantile_sorted};
use anyhow::{bail, Result};

/// Floor applied before the shifted logarithm (keeps ln(1+y) finite).
pub const LOG1P_FLOOR: f64 = -0.999999;
/// Exponent clip for the log1p inverse (exp overflows CPython above ~709).
pub const EXP_CLIP_LO: f64 = -745.0;
pub const EXP_CLIP_HI: f64 = 700.0;

/// True for the target pseudo-stage op.
pub fn is_target_op(op: &str) -> bool {
    op == "target"
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum FittedTarget {
    /// Mid-rank → probit forward; the deduplicated, strictly increasing
    /// (value, score) table drives the interpolated inverse.
    RankGauss {
        sorted_values: Vec<f64>,
        table_values: Vec<f64>,
        table_scores: Vec<f64>,
    },
    Log1p,
    /// Fitted train quantiles; forward clips, inverse is identity.
    Winsor {
        lo: f64,
        hi: f64,
    },
    /// Train mean/std standardization plus the train-range prediction clamp
    /// bounds (min/max widened by 5% of the range). Makes every head's
    /// lambda_frac scaling and the final prediction clamp scale-invariant —
    /// the historical ±60 clamp silently destroys any target scaled past it.
    ZScore {
        mu: f64,
        sigma: f64,
        clamp_lo: f64,
        clamp_hi: f64,
    },
}

/// Fit a target transform from its stage params on the train target.
pub fn fit_target(stage: &Stage, y: &[f64]) -> Result<FittedTarget> {
    if y.is_empty() {
        bail!("target transform requires a non-empty train target");
    }
    let kind = match stage.params.get("kind") {
        Some(ParamValue::Str(s)) => s.as_str(),
        _ => bail!("target stage requires kind=rank_gauss|log1p|winsor|zscore"),
    };
    match kind {
        "rank_gauss" => {
            let mut sorted_values = y.to_vec();
            sorted_values.sort_by(|a, b| a.total_cmp(b));
            // Deduplicate into a strictly increasing table: ties share one
            // mid-rank, so one entry per distinct value.
            let mut table_values = Vec::new();
            let mut table_scores = Vec::new();
            for i in 0..sorted_values.len() {
                if i == 0 || sorted_values[i] > sorted_values[i - 1] {
                    table_values.push(sorted_values[i]);
                    table_scores.push(probit(mid_rank(&sorted_values, sorted_values[i])));
                }
            }
            Ok(FittedTarget::RankGauss {
                sorted_values,
                table_values,
                table_scores,
            })
        }
        "log1p" => Ok(FittedTarget::Log1p),
        "zscore" => {
            // Explicit sequential loops: the emitted Python and Rust twins use
            // the same accumulation order, so all three agree bit-for-bit.
            let n = y.len() as f64;
            let mut mu = 0.0;
            for v in y {
                mu += v;
            }
            mu /= n;
            let mut var = 0.0;
            for v in y {
                let d = v - mu;
                var += d * d;
            }
            var /= n;
            let sigma = var.sqrt().max(1e-9);
            let mut y_min = f64::INFINITY;
            let mut y_max = f64::NEG_INFINITY;
            for v in y {
                y_min = y_min.min(*v);
                y_max = y_max.max(*v);
            }
            let spread = (y_max - y_min).max(1e-9);
            // The clamp interval is the UNION of the train range ±5% and the
            // legacy ±60 window: never tighter than either, so small-scale
            // targets keep their legitimate extrapolation headroom while
            // large-scale targets still survive past ±60. The min/max also
            // guarantee clamp_lo <= clamp_hi for any target.
            Ok(FittedTarget::ZScore {
                mu,
                sigma,
                clamp_lo: (y_min - 0.05 * spread).min(-60.0),
                clamp_hi: (y_max + 0.05 * spread).max(60.0),
            })
        }
        "winsor" => {
            let q = stage.get_f64("q", 0.05).clamp(0.0, 0.25);
            let mut s = y.to_vec();
            s.sort_by(|a, b| a.total_cmp(b));
            let lo = quantile_sorted(&s, q);
            // Interpolated quantiles on a near-constant target can cross by
            // one ulp; the max keeps clamp() panic-free (same guard as
            // winsor_clip).
            let hi = quantile_sorted(&s, 1.0 - q).max(lo);
            Ok(FittedTarget::Winsor { lo, hi })
        }
        other => bail!("unknown target transform kind: {other}"),
    }
}

impl FittedTarget {
    /// Map one train-target value into the transformed space.
    pub fn forward(&self, v: f64) -> f64 {
        match self {
            FittedTarget::RankGauss { sorted_values, .. } => probit(mid_rank(sorted_values, v)),
            FittedTarget::Log1p => (1.0 + v.max(LOG1P_FLOOR)).ln(),
            FittedTarget::Winsor { lo, hi } => v.clamp(*lo, *hi),
            FittedTarget::ZScore { mu, sigma, .. } => (v - mu) / sigma,
        }
    }

    /// Map the whole train target (convenience over `forward`).
    pub fn forward_all(&self, y: &[f64]) -> Vec<f64> {
        y.iter().map(|v| self.forward(*v)).collect()
    }

    /// Exact inverse of `forward` back to the original target scale
    /// (identity for winsor — clipped targets stay clipped by design).
    pub fn inverse(&self, z: f64) -> f64 {
        match self {
            FittedTarget::RankGauss {
                table_values,
                table_scores,
                ..
            } => {
                let values = table_values;
                let scores = table_scores;
                let m = values.len();
                // NaN also routes to the low end.
                if z.partial_cmp(&scores[0]) != Some(std::cmp::Ordering::Greater) {
                    return values[0];
                }
                if z >= scores[m - 1] {
                    return values[m - 1];
                }
                let hi = lower_bound(scores, z);
                let lo = hi - 1;
                let ds = scores[hi] - scores[lo];
                if ds > 0.0 {
                    values[lo] + (z - scores[lo]) / ds * (values[hi] - values[lo])
                } else {
                    values[lo]
                }
            }
            FittedTarget::Log1p => z.clamp(EXP_CLIP_LO, EXP_CLIP_HI).exp() - 1.0,
            FittedTarget::Winsor { .. } => z,
            FittedTarget::ZScore { mu, sigma, .. } => z * sigma + mu,
        }
    }

    /// Final clamp applied after `inverse` on the prediction path. Legacy
    /// kinds keep the historical ±60 clamp (bit-parity frozen); train-range-
    /// aware kinds clamp to the fitted target range ±5% instead, because ±60
    /// silently destroys any target whose scale exceeds it.
    pub fn final_clamp(&self, v: f64) -> f64 {
        match self {
            FittedTarget::RankGauss { .. } | FittedTarget::Log1p | FittedTarget::Winsor { .. } => {
                v.clamp(-60.0, 60.0)
            }
            FittedTarget::ZScore {
                clamp_lo, clamp_hi, ..
            } => v.clamp(*clamp_lo, *clamp_hi),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::parse_dsl;
    use crate::Pcg32;

    fn target_stage(src: &str) -> Stage {
        parse_dsl(&format!("(pipe X {src})")).unwrap().stages[0].clone()
    }

    #[test]
    fn rank_gauss_round_trips_train_values() {
        let mut rng = Pcg32::new(5, 2);
        let mut y: Vec<f64> = (0..200).map(|_| (3.0 * rng.next_f64()).exp()).collect();
        // Inject ties to exercise the deduplicated table.
        y[10] = y[20];
        y[30] = y[20];
        let t = fit_target(&target_stage("(target kind=rank_gauss)"), &y).unwrap();
        for v in &y {
            let z = t.forward(*v);
            assert!(z.is_finite());
            let back = t.inverse(z);
            assert!(
                (back - v).abs() <= 1e-9 * v.abs().max(1.0),
                "round trip {v} -> {z} -> {back}"
            );
        }
        // Out-of-range scores clamp to the extreme train values.
        let (mn, mx) = y
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
                (a.min(*v), b.max(*v))
            });
        assert_eq!(t.inverse(-9.0), mn);
        assert_eq!(t.inverse(9.0), mx);
        assert_eq!(t.inverse(f64::NAN), mn);
    }

    #[test]
    fn rank_gauss_constant_target_is_safe() {
        let y = vec![0.5; 30];
        let t = fit_target(&target_stage("(target kind=rank_gauss)"), &y).unwrap();
        assert_eq!(t.forward(0.5), 0.0); // probit(0.5)
        assert_eq!(t.inverse(0.0), 0.5);
        assert_eq!(t.inverse(2.0), 0.5);
        assert_eq!(t.inverse(-2.0), 0.5);
    }

    #[test]
    fn log1p_is_exact_inverse_on_unit_scale() {
        let y: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let t = fit_target(&target_stage("(target kind=log1p)"), &y).unwrap();
        for v in &y {
            let back = t.inverse(t.forward(*v));
            assert!((back - v).abs() < 1e-12, "{v} -> {back}");
        }
        // Values below the floor saturate instead of producing -inf.
        assert!(t.forward(-5.0).is_finite());
    }

    #[test]
    fn winsor_clips_and_inverse_is_identity() {
        let mut y: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        y[0] = -50.0;
        y[99] = 50.0;
        let t = fit_target(&target_stage("(target kind=winsor q=0.05)"), &y).unwrap();
        let yp = t.forward_all(&y);
        let (lo, hi) = match &t {
            FittedTarget::Winsor { lo, hi } => (*lo, *hi),
            other => panic!("wrong variant: {other:?}"),
        };
        assert!(lo > -50.0 && hi < 50.0);
        assert!(yp.iter().all(|v| *v >= lo && *v <= hi));
        assert_eq!(t.inverse(123.0), 123.0);
    }

    #[test]
    fn zscore_round_trips_and_survives_large_scale_targets() {
        // A target scaled far past the legacy ±60 clamp: zscore must round-trip
        // it exactly instead of destroying it.
        let y: Vec<f64> = (0..100).map(|i| 5000.0 + 37.0 * i as f64).collect();
        let t = fit_target(&target_stage("(target kind=zscore)"), &y).unwrap();
        for v in &y {
            let z = t.forward(*v);
            assert!(z.is_finite() && z.abs() < 3.0, "standardized {v} -> {z}");
            let back = t.final_clamp(t.inverse(z));
            assert!(
                (back - v).abs() <= 1e-9 * v.abs(),
                "round trip {v} -> {z} -> {back}"
            );
        }
        // The final clamp is the union of the train range ±5% and ±60: the
        // upper bound tracks the large-scale target, the lower bound falls
        // back to the legacy -60 (never tighter than either window).
        let (lo, hi) = match &t {
            FittedTarget::ZScore {
                clamp_lo, clamp_hi, ..
            } => (*clamp_lo, *clamp_hi),
            other => panic!("wrong variant: {other:?}"),
        };
        let spread = 37.0 * 99.0;
        assert_eq!(lo, -60.0);
        assert_eq!(hi, 5000.0 + spread + 0.05 * spread);
        assert_eq!(t.final_clamp(t.inverse(1e9)), hi);
        assert_eq!(t.final_clamp(t.inverse(-1e9)), lo);
    }

    #[test]
    fn zscore_constant_target_is_safe() {
        let y = vec![42.0; 25];
        let t = fit_target(&target_stage("(target kind=zscore)"), &y).unwrap();
        assert_eq!(t.forward(42.0), 0.0);
        assert_eq!(t.final_clamp(t.inverse(0.0)), 42.0);
        // sigma and range are floored, so wild scores still land near 42.
        let back = t.final_clamp(t.inverse(1e6));
        assert!((back - 42.0).abs() <= 1.0, "constant target inverse {back}");
    }

    #[test]
    fn fit_target_rejects_bad_kinds() {
        let y = vec![0.0, 1.0, 2.0];
        assert!(fit_target(&target_stage("(target)"), &y).is_err());
        assert!(fit_target(&target_stage("(target kind=bogus)"), &y).is_err());
        assert!(fit_target(&target_stage("(target kind=rank_gauss)"), &[]).is_err());
    }
}
