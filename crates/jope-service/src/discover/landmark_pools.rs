//! Landmark-V2 preferred stage and head pools.
//!
//! This is a port of `select_landmark_preferred_pools` from the RES-01-pinned
//! `research/flywheel_selfplay/rust_evaluator.py`. The RES-07 parity test uses
//! the hash-pinned packet selected by `JOPE_GOLDEN_DIR`.

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
    #[ignore = "requires the hash-custodied RES-07 packet; set JOPE_GOLDEN_DIR and JOPE_GOLDEN_INDEX_SHA256"]
    fn matches_res07_landmark_preferred_pool_golden_inputs() {
        use candle_core_0_11_0::{safetensors::load_buffer, Device};
        use serde_json::Value;
        use sha2::{Digest, Sha256};
        use std::{collections::HashSet, env, fs, path::PathBuf};

        let root = PathBuf::from(env::var_os("JOPE_GOLDEN_DIR").expect("set JOPE_GOLDEN_DIR"));
        let index_bytes = fs::read(root.join("index.json")).expect("read RES-07 index.json");
        let actual_index_sha = format!("{:x}", Sha256::digest(&index_bytes));
        assert_eq!(
            actual_index_sha,
            env::var("JOPE_GOLDEN_INDEX_SHA256").expect("set JOPE_GOLDEN_INDEX_SHA256"),
            "RES-07 fixture index pin differs"
        );
        let index: Value = serde_json::from_slice(&index_bytes).expect("parse RES-07 index");
        let pools_path = root.join("landmark-pools.json");
        let pools_bytes = fs::read(&pools_path).expect("read landmark-pools.json");
        let pools_pin = &index["files"]["landmark-pools.json"];
        assert_eq!(pools_pin["bytes"].as_u64(), Some(pools_bytes.len() as u64));
        assert_eq!(
            pools_pin["sha256"].as_str(),
            Some(format!("{:x}", Sha256::digest(&pools_bytes)).as_str())
        );
        let pools: Vec<Value> =
            serde_json::from_slice(&pools_bytes).expect("parse preferred pools");
        assert_eq!(
            pools.len(),
            50,
            "expected all contexts in the pinned packet"
        );

        let mut contexts = HashSet::new();
        for expected in pools {
            let context = expected["context_sha256"]
                .as_str()
                .expect("golden context_sha256 must be a string");
            assert_eq!(context.len(), 64, "context hash must be SHA256 hex");
            assert!(
                context.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "context hash must be hexadecimal"
            );
            assert!(
                contexts.insert(context.to_owned()),
                "duplicate golden context {context}"
            );

            let relative = format!("contexts/{context}.safetensors");
            let spec = &index["files"][&relative];
            assert!(!spec.is_null(), "index is missing {relative}");
            let bytes = fs::read(root.join(&relative)).expect("read context SafeTensors");
            assert_eq!(
                spec["bytes"].as_u64(),
                Some(bytes.len() as u64),
                "{relative}"
            );
            assert_eq!(
                spec["sha256"].as_str(),
                Some(format!("{:x}", Sha256::digest(&bytes)).as_str()),
                "{relative} hash differs from the pinned index"
            );

            let tensors = load_buffer(&bytes, &Device::Cpu).expect("decode context tensors");
            let lm48 = tensors
                .get("lm48")
                .expect("context fixture must contain lm48");
            assert_eq!(lm48.dims(), [1, 48], "{relative} lm48 shape");
            let values = lm48
                .flatten_all()
                .expect("flatten lm48")
                .to_vec1::<f32>()
                .expect("read lm48 values");
            let landmarks: [f64; 48] = values
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
                .try_into()
                .expect("lm48 must have 48 values");
            assert!(
                landmarks.iter().all(|value| value.is_finite()),
                "{relative}"
            );

            let (stages, heads) = select_landmark_preferred_pools(&landmarks);
            let expected_stages = expected["stages"]
                .as_array()
                .expect("golden stages must be an array")
                .iter()
                .map(|value| value.as_str().expect("stage must be a string").to_owned())
                .collect::<Vec<_>>();
            let expected_heads = expected["heads"]
                .as_array()
                .expect("golden heads must be an array")
                .iter()
                .map(|value| value.as_str().expect("head must be a string").to_owned())
                .collect::<Vec<_>>();
            assert_eq!(
                stages
                    .iter()
                    .map(|value| (*value).to_owned())
                    .collect::<Vec<_>>(),
                expected_stages,
                "stage pool mismatch for {context}"
            );
            assert_eq!(
                heads
                    .iter()
                    .map(|value| (*value).to_owned())
                    .collect::<Vec<_>>(),
                expected_heads,
                "head pool mismatch for {context}"
            );
        }
        assert_eq!(contexts.len(), 50);
    }
}
