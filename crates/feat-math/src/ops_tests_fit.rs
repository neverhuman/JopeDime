use super::*;

#[test]
fn svd_pipeline_cv_smoke() {
    // Same toy pattern as model.rs tests (3 cols → svd k clamps to 3).
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut rng = Pcg32::new(1, 1);
    for _ in 0..240 {
        let a = rng.next_f64();
        let b = rng.next_f64();
        let c = rng.next_f64();
        x.push(vec![a, b, c]);
        y.push(2.0 * a - b + 0.3 * (6.0 * c).sin());
    }
    let g =
        GenomeIr::parse("(pipe X (svd_project k=4) (rff_cos n=16 append=true) (ridge))").unwrap();
    let rep = cv_r2(&g, &x, &y, 5, "toy").unwrap();
    assert!(rep.oof_r2.is_finite());
    assert!(rep.oof_r2 > 0.5, "oof_r2={}", rep.oof_r2);
}

#[test]
fn g6_append_at_width_512_is_identity_and_g8_cache_reuses_sorted_prefix() {
    let (x, y) = toy(48, crate::MAX_WIDTH);
    let mut cache = SortedColumnsCache::new();
    for src in [
        "(quantile_bins bins=8 append=true)",
        "(hinge knots=3 append=true)",
        "(quantile_bins bins=12 append=true)",
        "(hinge knots=4 append=true)",
        "(cheb degree=3 append=true)",
    ] {
        let st = stage(src);
        let op_uncached = fit_op(&st, &x, &y).unwrap();
        let op_cached = fit_op_with_cache(&st, &x, &y, &mut cache).unwrap();
        assert_eq!(
            serde_json::to_string(&op_uncached).unwrap(),
            serde_json::to_string(&op_cached).unwrap(),
            "G-8 SortedColumnsCache must produce byte-identical FittedOp JSON for {src}"
        );
        let out_fast = apply_op(&op_cached, &x).unwrap();
        let out_ref = finish(&op_uncached, &x, apply_op_generated(&op_uncached, &x));
        assert_eq!(out_fast.len(), x.len());
        for (r_idx, ((rf, rr), rx)) in out_fast.iter().zip(&out_ref).zip(&x).enumerate() {
            assert_eq!(rf.len(), crate::MAX_WIDTH);
            for c_idx in 0..crate::MAX_WIDTH {
                assert_eq!(
                    rf[c_idx].to_bits(),
                    rr[c_idx].to_bits(),
                    "G-6 apply_op mismatch vs reference at ({r_idx}, {c_idx})"
                );
                assert_eq!(
                    rf[c_idx].to_bits(),
                    rx[c_idx].to_bits(),
                    "append=true at width 512 must be identity at ({r_idx}, {c_idx})"
                );
            }
        }
        cache.on_stage_complete(true, crate::MAX_WIDTH, crate::MAX_WIDTH);
    }
    assert_eq!(cache.cache_misses, crate::MAX_WIDTH);
    assert_eq!(cache.cache_hits, 3 * crate::MAX_WIDTH);
}

#[test]
fn mid_rank_fast_path_matches_reference_bounds_on_ties_and_nans() {
    fn mid_rank_ref(s: &[f64], v: f64) -> f64 {
        let n = s.len() as f64;
        let lo = lower_bound(s, v) as f64;
        let hi = upper_bound(s, v) as f64;
        (lo + hi) / (2.0 * n)
    }
    let mut slices: Vec<Vec<f64>> = vec![
        vec![0.0],
        vec![-0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![-2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 3.5, 3.5],
        vec![f64::NEG_INFINITY, -1.0, 0.0, 1.0, f64::INFINITY],
        vec![-1.0, 0.0, 0.0, 2.0, f64::NAN, f64::NAN],
    ];
    for s in &mut slices {
        s.sort_by(|a, b| a.total_cmp(b));
    }
    let queries = [
        f64::NEG_INFINITY,
        -3.0,
        -2.0,
        -1.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        1.0,
        2.0,
        3.5,
        4.0,
        f64::INFINITY,
        f64::NAN,
    ];
    for s in &slices {
        for &q in &queries {
            let got = mid_rank(s, q);
            let exp = mid_rank_ref(s, q);
            assert_eq!(
                got.to_bits(),
                exp.to_bits(),
                "mid_rank mismatch for slice={s:?}, q={q:?}"
            );
        }
    }
}

#[test]
fn parallel_map_rows_and_finish_above_65536_cells_match_reference() {
    // 256 rows x 300 cols = 76,800 > 65,536 cells exercises the parallel branches in
    // map_rows, finish (append=true and append=false), and SortedColumnsCache::ensure_cols.
    let (mut x, y) = toy(256, 300);
    x[17][42] = f64::NAN;
    x[129][211] = f64::INFINITY;
    let mut cache = SortedColumnsCache::new();
    for src in [
        "(quantile_bins bins=8 append=false)",
        "(quantile_bins bins=8 append=true)",
        "(hinge knots=2 append=true)",
    ] {
        let st = stage(src);
        let op_fit = fit_op_with_cache(&st, &x, &y, &mut cache).unwrap();
        let out_fast = apply_op(&op_fit, &x).unwrap();
        let out_ref = finish(&op_fit, &x, apply_op_generated(&op_fit, &x));
        assert_eq!(out_fast.len(), out_ref.len());
        for (rf, rr) in out_fast.iter().zip(&out_ref) {
            assert_eq!(rf.len(), rr.len());
            for (&vf, &vr) in rf.iter().zip(rr) {
                assert_eq!(vf.to_bits(), vr.to_bits());
            }
        }
    }
    // Changing row count invalidates SortedColumnsCache automatically.
    let (x_short, _) = toy(16, 300);
    cache.ensure_cols(&x_short, 4);
    assert_eq!(cache.col_slice(0).len(), 16);
}
