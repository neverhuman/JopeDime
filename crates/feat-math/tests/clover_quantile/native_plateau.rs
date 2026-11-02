use super::*;
use feat_math::{
    dsl::{ParamValue, Stage},
    ops::FittedOp,
};

#[test]
fn tie_plateau_edges_diverge_from_legacy_searches() {
    // Build a deterministic 24x3 dataset whose three columns each have a tie
    // plateau that triggers the 1-ULP non-monotone dip in `s_lo * (1.0 - f) + s_hi * f`
    // for `bins=27` (col 0, `v0 = 9.0 / 19.0`), `bins=16` (col 1, `v1 = 7.0 / 19.0`),
    // and `bins=13` (col 2, `v2 = 31.0 / 19.0`).
    let v0 = 9.0_f64 / 19.0;
    let v1 = 7.0_f64 / 19.0;
    let v2 = 31.0_f64 / 19.0;
    let n = 24usize;
    let mut train = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let c0 = if i < 22 {
            v0
        } else {
            v0 + ((i - 21) as f64) * 0.5
        };
        let c1 = if i < 4 {
            -10.0 + (i as f64)
        } else if i < 22 {
            v1
        } else {
            v1 + ((i - 21) as f64) * 0.5
        };
        let c2 = if i < 8 {
            -10.0 + (i as f64)
        } else if i < 13 {
            v2
        } else {
            v2 + ((i - 12) as f64) * 0.5
        };
        train.push(vec![c0, c1, c2]);
        y.push(0.6 * c0 - 0.35 * c1 + 0.2 * c2 + ((i % 5) as f64) * 0.07);
    }
    let mut holdout = train.clone();
    holdout.push(vec![v0, v1, v2]);
    holdout.push(vec![
        f64::from_bits(v0.to_bits() - 1),
        f64::from_bits(v1.to_bits() - 1),
        f64::from_bits(v2.to_bits() - 1),
    ]);

    let legacy_lo_hi = |s: &[f64], v: f64| -> usize {
        let (mut lo, mut hi) = (0usize, s.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if s[mid] < v {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    };

    // Verify the 1-ULP tie-plateau non-monotonicity and that both the legacy `(lo, hi)`
    // `mid + 1` search and the old bound-Python linear scan diverge from
    // `slice::partition_point` for all three bin counts.
    let mut non_monotone_edges = Vec::new();
    for (col, bins, probe) in [
        (0usize, 27i64, v0),
        (1usize, 16i64, v1),
        (2usize, 13i64, v2),
    ] {
        let mut stage = Stage::new("quantile_bins");
        stage.params.insert("bins".into(), ParamValue::Int(bins));
        let fitted = fit_op(&stage, &train, &y).unwrap();
        let edges = match &fitted {
            FittedOp::QuantileBins { edges, .. } => &edges[col],
            _ => unreachable!(),
        };
        assert!(
            edges.windows(2).any(|w| w[0] > w[1]),
            "col {col} (bins={bins}) tie plateau must trigger 1-ULP non-monotone quantile edges"
        );
        let native_idx = edges.partition_point(|e| *e < probe);
        let legacy_idx = legacy_lo_hi(edges, probe);
        let linear_idx = edges.iter().filter(|&&e| e < probe).count();
        assert_ne!(
            legacy_idx, native_idx,
            "col {col} (bins={bins}) fixture must reproduce old lo/hi vs partition_point divergence"
        );
        if col >= 1 {
            assert_ne!(
                linear_idx, native_idx,
                "col {col} (bins={bins}) fixture must reproduce old bound-Python linear scan vs partition_point divergence"
            );
        }
        non_monotone_edges.push((col, bins, probe, edges.clone(), native_idx));
    }

    // Also verify directly that `target_bin_encode bins=13` on `train` has the 1-ULP
    // non-monotone edge on col 2 (`v2`) where legacy `(lo, hi)` diverges from `partition_point`.
    {
        let mut tbe_stage = Stage::new("target_bin_encode");
        tbe_stage.params.insert("bins".into(), ParamValue::Int(13));
        let fitted_tbe = fit_op(&tbe_stage, &train, &y).unwrap();
        let tbe_edges = match &fitted_tbe {
            FittedOp::TargetBinEncode { edges, .. } => &edges[2],
            _ => unreachable!(),
        };
        assert!(
            tbe_edges.windows(2).any(|w| w[0] > w[1]),
            "target_bin_encode bins=13 col 2 must trigger 1-ULP non-monotone quantile edges"
        );
        assert_ne!(
            legacy_lo_hi(tbe_edges, v2),
            tbe_edges.partition_point(|e| *e < v2),
            "target_bin_encode bins=13 col 2 must diverge between legacy lo/hi and partition_point"
        );
    }
}
