use super::*;
use feat_math::{folds::fold_of, Model};
use serde_json::{json, Value};
use std::{io::Write, path::Path};

#[test]
fn champion_receipt_matches_recorded_emissions() {
    use feat_math::{fit, folds::fold_of, Model};
    use serde_json::{json, Value};
    use std::{io::Write, path::Path};

    const EXP008_READOUT: &str =
        include_str!("../../../../docs/internal/clover/gp-h1-readout.20261013.json");
    const EXP008_REPLAY_RECEIPT: &str = include_str!(
        "../../../../docs/internal/clover/exp-008-quantile-parity-replay.20261017.json"
    );
    const EXPECTED_EXPORT_MANIFEST_SHA256: &str =
        "eb08114756c80c942a0d08f4e6604e4d3b2f9bb9ad3d7ef223d2c41f4f862394";

    struct Exp008ChampionCase {
        index: usize,
        dataset: &'static str,
        context_sha256: &'static str,
        arm: &'static str,
        program_sha256: &'static str,
        genome: &'static str,
        default_emitted_rust_sha256: &'static str,
        partition_point_emitted_rust_sha256: &'static str,
        partition_point_emitted_python_sha256: &'static str,
        expected_train_csv_sha256: &'static str,
        expected_train_rows: usize,
        expected_columns: usize,
    }

    let champions = [
        Exp008ChampionCase {
            index: 14,
            dataset: "51d5e3d9bca34669",
            context_sha256: "15c626cec6b3f3ef75abe1fd5cd9d296af77c73f570a00b9a141b5be1e405f51",
            arm: "retrieval_seeded",
            program_sha256: "d217704734d7d9ace5ff39a73de79e01c7b21bdc8e43b0cc3b09fe3e7b5bf70a",
            genome: "(pipe X (quantile_bins append=false bins=27) (quantile_bins append=true bins=25) (quantile_bins append=true bins=11) (cheb degree=7) (elasticnet alpha=0.008098970618082419 l1_ratio=0.6803338845776383))",
            default_emitted_rust_sha256: "9d5183c29febca9bf6cfb720d1e01751087d9a1e2d661da558d4f33824a64579",
            partition_point_emitted_rust_sha256: "58781c56cc55f65cca694965d7dc921b42663823a3ce4fc342745bd8d8fb950d",
            partition_point_emitted_python_sha256: "c693fd1a2396e80037ffd2852241a679c5921eb41f25ac37c8a0eb8f6c43f708",
            expected_train_csv_sha256: "743c50f720b30bd3cd372b48b167138b95f602b73b9f1f8c61577f74a7e8a059",
            expected_train_rows: 1000,
            expected_columns: 3,
        },
        Exp008ChampionCase {
            index: 18,
            dataset: "72c1bdcbd2d62359",
            context_sha256: "634d8069d07ed47d0bc53f122e4149c1f90588071c66f652a93352b6a70626fa",
            arm: "ordinary",
            program_sha256: "14882a744768ece2e34abc1c410c772181394b140d93a0c2e3623d03dd10dedb",
            genome: "(pipe X (bin_onehot append=true bins=16) (blom) (histgbm2 bags=5 col_subsample=0.6 leaves=255 lr=0.05 min_leaf=1 rounds=1000 row_subsample=0.6 val_pct=15 val_shuffle=true))",
            default_emitted_rust_sha256: "fcf13f63f3ff8e92c0b4ed26e67ca21c6c510a2c133df496b94a2a82b03a460e",
            partition_point_emitted_rust_sha256: "07086b9f152082ae78284d6afb869eac32e420c36f406bac59fe16d7212a3b6f",
            partition_point_emitted_python_sha256: "e7200bfe310667cffba39b247284acc8de72546c8149e585e22bf3cb73d5c6ff",
            expected_train_csv_sha256: "af1b8a3c483394d282d531505356da877de40294bf41ca70a70a43509fbfe2cd",
            expected_train_rows: 1000,
            expected_columns: 21,
        },
        Exp008ChampionCase {
            index: 18,
            dataset: "72c1bdcbd2d62359",
            context_sha256: "634d8069d07ed47d0bc53f122e4149c1f90588071c66f652a93352b6a70626fa",
            arm: "policy_seeded",
            program_sha256: "6165e86c7d7641a9e73135cf74e9d331c6f14ffc76b440a9bcd5d2e48d8fa76b",
            genome: "(pipe X (cheb degree=3) (poly_interact append=true) (poly_interact append=true) (target_bin_encode bins=13) (histgbm depth=5 lr=0.17651389662515513 rounds=198))",
            default_emitted_rust_sha256: "ac780329825506a35032cd69fd0437cfea35fcd5b218d0f79442f60dc1999e92",
            partition_point_emitted_rust_sha256: "de0872bd1967137d959eee077259de130cd2c0445a5d4d23323923405c02dbf3",
            partition_point_emitted_python_sha256: "46c93cf23b3218a533287c860116308be6c97bc5ddc8d029c1b2ced9b6fd41cb",
            expected_train_csv_sha256: "af1b8a3c483394d282d531505356da877de40294bf41ca70a70a43509fbfe2cd",
            expected_train_rows: 1000,
            expected_columns: 21,
        },
    ];

    let sha256_hex = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let readout: Value = serde_json::from_str(EXP008_READOUT).unwrap();
    let replay_receipt: Value = serde_json::from_str(EXP008_REPLAY_RECEIPT).unwrap();

    let pp_opts = EmitOptions {
        partition_point_quantile_bounds: true,
        ..EmitOptions::default()
    };

    if std::env::var("CLOVER_EXP008_REPLAY_OUT").is_err() {
        assert_eq!(
            replay_receipt["schema"],
            "clover.exp008-quantile-parity-replay/v1"
        );
        assert_eq!(replay_receipt["evidence_only"], true);
        assert_eq!(replay_receipt["default_emitter_changed"], false);
        assert_eq!(replay_receipt["r5_parity_charging_changed"], false);
        assert_eq!(replay_receipt["h1_established"], false);
        assert_eq!(replay_receipt["qualified_rewards"], 0);
        assert_eq!(
            replay_receipt["source_readout_sha256"],
            sha256_hex(EXP008_READOUT.as_bytes())
        );
        assert_eq!(
            replay_receipt["export_manifest_sha256"],
            EXPECTED_EXPORT_MANIFEST_SHA256
        );

        let readout_failures = readout["parity_failures"].as_array().unwrap();
        let receipt_cases = replay_receipt["champions"].as_array().unwrap();
        assert_eq!(readout_failures.len(), 3);
        assert_eq!(receipt_cases.len(), 3);

        for (idx, (case, (ro_fail, rc_case))) in champions
            .iter()
            .zip(readout_failures.iter().zip(receipt_cases.iter()))
            .enumerate()
        {
            assert_eq!(
                sha256_hex(case.genome.as_bytes()),
                case.program_sha256,
                "champion {idx} genome SHA-256 mismatch"
            );
            let parsed = GenomeIr::parse(case.genome).unwrap();
            let default_emitted = emit(&parsed, &EmitOptions::default()).unwrap();
            let pp_emitted = emit(&parsed, &pp_opts).unwrap();
            let default_rs_sha = sha256_hex(default_emitted.rust.as_bytes());
            let pp_rs_sha = sha256_hex(pp_emitted.rust.as_bytes());
            let pp_py_sha = sha256_hex(pp_emitted.python.as_bytes());

            assert_eq!(
                default_rs_sha, case.default_emitted_rust_sha256,
                "champion {idx} default emitted Rust SHA-256 must match historical EXP-008 receipt"
            );
            assert_eq!(
                pp_rs_sha, case.partition_point_emitted_rust_sha256,
                "champion {idx} partition_point emitted Rust SHA-256 mismatch"
            );
            assert_eq!(
                pp_py_sha, case.partition_point_emitted_python_sha256,
                "champion {idx} partition_point emitted Python SHA-256 mismatch"
            );

            // Cross-check against `docs/internal/clover/gp-h1-readout.20261013.json`.
            assert_eq!(ro_fail["index"], case.index);
            assert_eq!(ro_fail["dataset"], case.dataset);
            assert_eq!(ro_fail["context_sha256"], case.context_sha256);
            assert_eq!(ro_fail["arm"], case.arm);
            assert_eq!(ro_fail["program_sha256"], case.program_sha256);

            // Cross-check against `docs/internal/clover/exp-008-quantile-parity-replay.20261017.json`.
            assert_eq!(rc_case["index"], case.index);
            assert_eq!(rc_case["dataset"], case.dataset);
            assert_eq!(rc_case["context_sha256"], case.context_sha256);
            assert_eq!(rc_case["arm"], case.arm);
            assert_eq!(rc_case["program_sha256"], case.program_sha256);
            assert_eq!(rc_case["genome"], case.genome);
            assert_eq!(
                rc_case["default_emitted_rust_sha256"],
                case.default_emitted_rust_sha256
            );
            assert_eq!(rc_case["partition_point_emitted_rust_sha256"], pp_rs_sha);
            assert_eq!(rc_case["partition_point_emitted_python_sha256"], pp_py_sha);
            assert_eq!(rc_case["default_emit_options_passed"], false);
            assert_eq!(rc_case["partition_point_rust_passed"], true);

            let ro_cuts = ro_fail["cuts"].as_array().unwrap();
            let rc_cuts = rc_case["cuts"].as_array().unwrap();
            assert_eq!(ro_cuts.len(), 2);
            assert_eq!(rc_cuts.len(), 2);
            for (ro_cut, rc_cut) in ro_cuts.iter().zip(rc_cuts.iter()) {
                assert_eq!(rc_cut["cut"], ro_cut["cut"]);
                assert_eq!(rc_cut["predictions"], ro_cut["predictions"]);
                assert_eq!(rc_cut["probe_sha256"], ro_cut["probe_sha256"]);
                assert_eq!(
                    rc_cut["default_rust_predictions_sha256"],
                    ro_cut["predictions_sha256"]
                );
                assert_eq!(
                    rc_cut["default_rust_bitwise_equal"],
                    ro_cut["bitwise_equal"]
                );
                assert_eq!(
                    rc_cut["default_rust_max_absolute_delta"],
                    ro_cut["max_absolute_delta"]
                );
                assert_eq!(rc_cut["partition_point_rust_bitwise_equal"], true);
                assert_eq!(rc_cut["partition_point_rust_max_absolute_delta"], 0.0);
                assert_eq!(
                    rc_cut["partition_point_rust_predictions_sha256"],
                    rc_cut["native_predictions_sha256"]
                );
                if ro_cut["bitwise_equal"] == false {
                    assert!(
                        rc_cut["non_monotone_tie_plateau_edge_vectors"].as_u64().unwrap() > 0,
                        "failing default cut must exhibit non-monotone tie-plateau fitted edge vectors"
                    );
                    assert!(
                        rc_cut["non_monotone_tie_plateau_inversions"].as_u64().unwrap() > 0,
                        "failing default cut must exhibit non-monotone tie-plateau adjacent inversions"
                    );
                    let max_ulps = rc_cut["max_inversion_ulps"].as_u64().unwrap();
                    assert!(
                        (1..=2).contains(&max_ulps),
                        "failing default cut max_inversion_ulps must be 1 or 2 ULPs, got {max_ulps}"
                    );
                }
            }
        }
    }
}
