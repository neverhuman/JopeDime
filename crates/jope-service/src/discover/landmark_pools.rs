//! Landmark-V2 preferred stage and head pools.
//!
//! This is a pre-stage port of `select_landmark_preferred_pools` from the
//! RES-01-pinned `research/flywheel_selfplay/rust_evaluator.py`. RES-07 golden
//! fixtures are not available yet; do not treat these synthetic tests as parity
//! qualification.

const HEAVY_TAIL: &[&str] = &[
    "(winsor_clip append=true q=0.05)",
    "(winsor_clip append=true q=0.1)",
    "(winsor_clip q=0.08)",
    "(blom)",
    "(sigmoid append=true)",
];

const MONOTONE_RANK: &[&str] = &[
    "(rank)",
    "(blom)",
    "(quantile_bins append=true bins=16)",
    "(quantile_bins append=true bins=32)",
    "(target_bin_encode append=true bins=12)",
];

const NONLINEAR_POLY: &[&str] = &[
    "(cheb append=true degree=3)",
    "(cheb append=true degree=4)",
    "(cheb degree=5)",
    "(poly_interact append=true)",
    "(hinge append=true knots=3)",
    "(hinge append=true knots=5)",
];

const INTERACTION: &[&str] = &[
    "(pair_prod append=true topk=8)",
    "(pair_prod append=true topk=16)",
    "(poly_interact append=true)",
    "(oblique_hinge gamma=0.25 n=32 seed=1337)",
    "(rff_sin append=true gamma=0.2 n=32 seed=2026)",
];

const PROJECTION_SCALE: &[&str] = &[
    "(standardize append=true)",
    "(standardize append=false)",
    "(robust_scale)",
    "(row_stats append=true)",
    "(rand_proj append=true k=32 seed=701)",
];

const TREE_DEEP: &[&str] = &[
    "(histgbm depth=6 lr=0.08 rounds=160)",
    "(histgbm depth=5 lr=0.1 rounds=180)",
    "(histgbm depth=6 lr=0.06 rounds=220)",
];

const TREE_SHALLOW: &[&str] = &[
    "(histgbm depth=3 lr=0.06 rounds=150)",
    "(histgbm depth=4 lr=0.08 rounds=140)",
    "(histgbm depth=2 lr=0.05 rounds=180)",
];

const LINEAR_REGULARIZED: &[&str] = &[
    "(ridge_gcv)",
    "(ridge lambda_frac=0.01)",
    "(ridge lambda_frac=0.1)",
];

const LOCAL_KERNEL: &[&str] = &[
    "(ksmooth gamma=4.0 k=32)",
    "(ksmooth gamma=2.0 k=64)",
    "(ksmooth gamma=4.0 k=16)",
];

/// Select stage and head pools from the six Landmark-V2 contrasts used by the
/// Python reference. Indices are nonlin=36, interaction=37, rank=38,
/// symlog=39, polynomial=40, and smooth=43. Pool order and duplicates are
/// preserved exactly as in the reference function.
pub fn select_landmark_preferred_pools(
    landmark_v2_48: &[f64; 48],
) -> (Vec<&'static str>, Vec<&'static str>) {
    let nonlin_gap = landmark_v2_48[36];
    let inter_gap = landmark_v2_48[37];
    let rank_gap = landmark_v2_48[38];
    let symlog_gap = landmark_v2_48[39];
    let poly_gap = landmark_v2_48[40];
    let smooth_gap = landmark_v2_48[43];

    let mut preferred_stages = Vec::new();
    if symlog_gap < -0.01 {
        preferred_stages.extend_from_slice(HEAVY_TAIL);
    }
    if rank_gap < -0.01 {
        preferred_stages.extend_from_slice(MONOTONE_RANK);
    }
    if poly_gap < -0.01 || nonlin_gap < -0.02 {
        preferred_stages.extend_from_slice(NONLINEAR_POLY);
    }
    if inter_gap < -0.005 {
        preferred_stages.extend_from_slice(INTERACTION);
    }
    preferred_stages.extend_from_slice(PROJECTION_SCALE);

    let mut preferred_heads = Vec::new();
    if nonlin_gap < -0.03 {
        if inter_gap < -0.01 {
            preferred_heads.extend_from_slice(TREE_DEEP);
            preferred_heads.extend_from_slice(TREE_SHALLOW);
        } else {
            preferred_heads.extend_from_slice(TREE_SHALLOW);
            preferred_heads.extend_from_slice(TREE_DEEP);
        }
    } else {
        preferred_heads.extend_from_slice(LINEAR_REGULARIZED);
        preferred_heads.extend_from_slice(TREE_SHALLOW);
    }
    if smooth_gap < -0.02 {
        preferred_heads.extend_from_slice(LOCAL_KERNEL);
    }
    preferred_heads.extend_from_slice(LINEAR_REGULARIZED);

    (preferred_stages, preferred_heads)
}

#[cfg(test)]
mod tests {
    use super::select_landmark_preferred_pools;

    #[test]
    fn neutral_landmarks_keep_reference_defaults_and_duplicate_linear_pool() {
        let landmarks = [0.0; 48];
        let (stages, heads) = select_landmark_preferred_pools(&landmarks);

        assert_eq!(
            stages,
            [
                "(standardize append=true)",
                "(standardize append=false)",
                "(robust_scale)",
                "(row_stats append=true)",
                "(rand_proj append=true k=32 seed=701)",
            ]
        );
        assert_eq!(
            heads,
            [
                "(ridge_gcv)",
                "(ridge lambda_frac=0.01)",
                "(ridge lambda_frac=0.1)",
                "(histgbm depth=3 lr=0.06 rounds=150)",
                "(histgbm depth=4 lr=0.08 rounds=140)",
                "(histgbm depth=2 lr=0.05 rounds=180)",
                "(ridge_gcv)",
                "(ridge lambda_frac=0.01)",
                "(ridge lambda_frac=0.1)",
            ]
        );
    }

    #[test]
    fn negative_contrasts_append_regime_pools_in_reference_order() {
        let mut landmarks = [0.0; 48];
        landmarks[36] = -0.031;
        landmarks[37] = -0.02;
        landmarks[38] = -0.02;
        landmarks[39] = -0.02;
        landmarks[40] = -0.02;
        landmarks[43] = -0.021;
        let (stages, heads) = select_landmark_preferred_pools(&landmarks);

        assert_eq!(stages.len(), 26);
        assert_eq!(stages[0], "(winsor_clip append=true q=0.05)");
        assert_eq!(stages[5], "(rank)");
        assert_eq!(stages[10], "(cheb append=true degree=3)");
        assert_eq!(stages[16], "(pair_prod append=true topk=8)");
        assert_eq!(stages[21], "(standardize append=true)");
        assert_eq!(heads.len(), 12);
        assert_eq!(heads[0], "(histgbm depth=6 lr=0.08 rounds=160)");
        assert_eq!(heads[3], "(histgbm depth=3 lr=0.06 rounds=150)");
        assert_eq!(heads[6], "(ksmooth gamma=4.0 k=32)");
        assert_eq!(heads[9], "(ridge_gcv)");
    }

    #[test]
    fn thresholds_are_strict_and_only_documented_indices_are_read() {
        let mut landmarks = [0.0; 48];
        landmarks[0] = f64::NEG_INFINITY;
        landmarks[36] = -0.02;
        landmarks[37] = -0.005;
        landmarks[38] = -0.01;
        landmarks[39] = -0.01;
        landmarks[40] = -0.01;
        landmarks[43] = -0.02;
        let (stages, heads) = select_landmark_preferred_pools(&landmarks);

        assert_eq!(stages.len(), 5);
        assert_eq!(heads.len(), 9);
    }

    #[test]
    #[ignore = "needs RES-07 preferred-pool golden fixtures before parity qualification"]
    fn matches_res07_landmark_preferred_pool_golden_inputs() {
        // RES-07 has not published this consumer's input/output fixtures yet.
        // Keep the parity gate visible but do not invent fixture values here.
        panic!("replace with RES-07 generated fixtures before unignoring");
    }
}
