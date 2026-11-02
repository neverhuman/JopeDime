use super::*;
use feat_math::{fit, folds::fold_of, Model};
use serde_json::{json, Value};
use std::{io::Write, path::Path};

#[test]
fn optional_champion_export_replay_matches_receipt() {
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

    // Optional live replay against the read-only host custody export
    // (`CLOVER_EXP008_EXPORT_DIR=<jope-evidence-dir>/claude-h1-export/gb06-train24-20261013T0453Z`).
    if let Ok(export_dir_str) = std::env::var("CLOVER_EXP008_EXPORT_DIR") {
        let export_dir = Path::new(&export_dir_str);
        let export_manifest_bytes =
            fs::read(export_dir.join("EXPORT_MANIFEST.json")).expect("read EXPORT_MANIFEST.json");
        assert_eq!(
            sha256_hex(&export_manifest_bytes),
            EXPECTED_EXPORT_MANIFEST_SHA256,
            "EXPORT_MANIFEST.json SHA-256 mismatch"
        );
        let export_manifest: Value = serde_json::from_slice(&export_manifest_bytes).unwrap();
        assert_eq!(export_manifest["train_only"], true);
        assert_eq!(export_manifest["test_csv_exported"], false);

        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        fs::create_dir_all(&root).unwrap();
        let tmp = tempfile::tempdir_in(root).unwrap();

        let write_csv = |path: &Path, x: &[Vec<f64>], y: Option<&[f64]>| {
            let mut file = fs::File::create(path).unwrap();
            for (i, row) in x.iter().enumerate() {
                for (j, val) in row.iter().enumerate() {
                    if j > 0 {
                        write!(file, ",").unwrap();
                    }
                    write!(file, "{val}").unwrap();
                }
                if let Some(targets) = y {
                    write!(file, ",{}", targets[i]).unwrap();
                }
                writeln!(file).unwrap();
            }
            file.sync_all().unwrap();
        };

        let mut replayed_champions = Vec::new();
        for (idx, case) in champions.iter().enumerate() {
            let ds_dir = export_dir.join("datasets").join(case.dataset);
            let train_bytes = fs::read(ds_dir.join("train.csv")).unwrap();
            assert_eq!(
                sha256_hex(&train_bytes),
                case.expected_train_csv_sha256,
                "train.csv SHA-256 mismatch for {}",
                case.dataset
            );
            let train_text = std::str::from_utf8(&train_bytes).unwrap();
            let rows: Vec<Vec<f64>> = train_text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|line| {
                    let r: Vec<f64> = line
                        .split(',')
                        .map(|f| f.trim().parse::<f64>().unwrap())
                        .collect();
                    assert_eq!(r.len(), case.expected_columns);
                    assert!(r.iter().all(|v| v.is_finite()));
                    r
                })
                .collect();
            assert_eq!(rows.len(), case.expected_train_rows);

            let fold_key = format!("{}:20261013", case.dataset);
            let assignment: Vec<usize> =
                (0..rows.len()).map(|r| fold_of(&fold_key, r, 5)).collect();

            let parsed = GenomeIr::parse(case.genome).unwrap();
            let default_emitted = emit(&parsed, &EmitOptions::default()).unwrap();
            let pp_emitted = emit(&parsed, &pp_opts).unwrap();

            let default_rs_src = tmp.path().join(format!("champ_{idx}_default.rs"));
            let default_rs_bin = tmp.path().join(format!("champ_{idx}_default_bin"));
            fs::write(&default_rs_src, &default_emitted.rust).unwrap();
            let compile_def = Command::new("rustc")
                .args(["--edition=2021", "-O", "-Awarnings"])
                .arg(&default_rs_src)
                .arg("-o")
                .arg(&default_rs_bin)
                .output()
                .unwrap();
            assert!(
                compile_def.status.success(),
                "{}",
                String::from_utf8_lossy(&compile_def.stderr)
            );

            let pp_rs_src = tmp.path().join(format!("champ_{idx}_pp.rs"));
            let pp_rs_bin = tmp.path().join(format!("champ_{idx}_pp_bin"));
            fs::write(&pp_rs_src, &pp_emitted.rust).unwrap();
            let compile_pp = Command::new("rustc")
                .args(["--edition=2021", "-O", "-Awarnings"])
                .arg(&pp_rs_src)
                .arg("-o")
                .arg(&pp_rs_bin)
                .output()
                .unwrap();
            assert!(
                compile_pp.status.success(),
                "{}",
                String::from_utf8_lossy(&compile_pp.stderr)
            );

            let mut cut_records = Vec::new();
            for cut_idx in 0..2usize {
                let mut train_x = Vec::new();
                let mut train_y = Vec::new();
                let mut hold_x = Vec::new();
                for (r_idx, row) in rows.iter().enumerate() {
                    let (features, target) = row.split_at(case.expected_columns - 1);
                    if assignment[r_idx] == cut_idx {
                        hold_x.push(features.to_vec());
                    } else {
                        train_x.push(features.to_vec());
                        train_y.push(target[0]);
                    }
                }

                let train_csv_path = tmp.path().join(format!("champ_{idx}_cut{cut_idx}_fit.csv"));
                let probe_csv_path = tmp
                    .path()
                    .join(format!("champ_{idx}_cut{cut_idx}_probe.csv"));
                write_csv(&train_csv_path, &train_x, Some(&train_y));
                write_csv(&probe_csv_path, &hold_x, None);
                let probe_sha256 = sha256_hex(&fs::read(&probe_csv_path).unwrap());

                let model = fit(&parsed, &train_x, &train_y).unwrap();
                let native_preds = model.predict(&hold_x).unwrap();
                let native_out_path = tmp
                    .path()
                    .join(format!("champ_{idx}_cut{cut_idx}_native.csv"));
                {
                    let mut f = fs::File::create(&native_out_path).unwrap();
                    for v in &native_preds {
                        writeln!(f, "{v}").unwrap();
                    }
                }
                let native_predictions_sha256 = sha256_hex(&fs::read(&native_out_path).unwrap());

                let mut edge_vectors_total = 0usize;
                let mut non_monotone_tie_plateau_edge_vectors = 0usize;
                let mut non_monotone_tie_plateau_inversions = 0usize;
                let mut max_inversion_ulps = 0u64;
                if let Model::Pipeline { ops, .. } = &model {
                    for op in ops {
                        let edges_opt = match op {
                            FittedOp::QuantileBins { edges, .. }
                            | FittedOp::BinOnehot { edges, .. }
                            | FittedOp::TargetBinEncode { edges, .. } => Some(edges),
                            _ => None,
                        };
                        if let Some(edges_list) = edges_opt {
                            for edges in edges_list {
                                edge_vectors_total += 1;
                                let mut has_inv = false;
                                for w in edges.windows(2) {
                                    if w[0] > w[1] {
                                        let ulp_diff = w[0].to_bits().abs_diff(w[1].to_bits());
                                        assert!(
                                            (1..=2).contains(&ulp_diff),
                                            "non-monotone quantile edge inversion must be 1 or 2 ULPs (a^+ followed by a^- on a tie plateau), got {ulp_diff}"
                                        );
                                        has_inv = true;
                                        non_monotone_tie_plateau_inversions += 1;
                                        max_inversion_ulps = max_inversion_ulps.max(ulp_diff);
                                    }
                                }
                                if has_inv {
                                    non_monotone_tie_plateau_edge_vectors += 1;
                                }
                            }
                        }
                    }
                }

                let def_pred_path = tmp
                    .path()
                    .join(format!("champ_{idx}_cut{cut_idx}_def_rs.csv"));
                let def_status = Command::new(&default_rs_bin)
                    .args([&train_csv_path, &probe_csv_path, &def_pred_path])
                    .output()
                    .unwrap();
                assert!(def_status.status.success());
                let def_raw = fs::read(&def_pred_path).unwrap();
                let def_preds: Vec<f64> = std::str::from_utf8(&def_raw)
                    .unwrap()
                    .lines()
                    .map(|l| l.trim().parse::<f64>().unwrap())
                    .collect();
                let def_equal = def_preds
                    .iter()
                    .zip(&native_preds)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits());
                let def_max_delta = def_preds
                    .iter()
                    .zip(&native_preds)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);

                let pp_rs_pred_path = tmp
                    .path()
                    .join(format!("champ_{idx}_cut{cut_idx}_pp_rs.csv"));
                let pp_rs_status = Command::new(&pp_rs_bin)
                    .args([&train_csv_path, &probe_csv_path, &pp_rs_pred_path])
                    .output()
                    .unwrap();
                assert!(pp_rs_status.status.success());
                let pp_rs_raw = fs::read(&pp_rs_pred_path).unwrap();
                let pp_rs_preds: Vec<f64> = std::str::from_utf8(&pp_rs_raw)
                    .unwrap()
                    .lines()
                    .map(|l| l.trim().parse::<f64>().unwrap())
                    .collect();
                let pp_rs_equal = pp_rs_preds
                    .iter()
                    .zip(&native_preds)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits());
                let pp_rs_max_delta = pp_rs_preds
                    .iter()
                    .zip(&native_preds)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(
                    pp_rs_equal,
                    "champ {idx} cut {cut_idx} pp Rust must match bitwise"
                );
                assert_eq!(pp_rs_max_delta, 0.0);

                cut_records.push(json!({
                    "cut": cut_idx,
                    "train_rows": train_x.len(),
                    "predictions": native_preds.len(),
                    "probe_sha256": probe_sha256,
                    "native_predictions_sha256": native_predictions_sha256,
                    "edge_vectors_total": edge_vectors_total,
                    "non_monotone_tie_plateau_edge_vectors": non_monotone_tie_plateau_edge_vectors,
                    "non_monotone_tie_plateau_inversions": non_monotone_tie_plateau_inversions,
                    "max_inversion_ulps": max_inversion_ulps,
                    "default_rust_bitwise_equal": def_equal,
                    "default_rust_max_absolute_delta": def_max_delta,
                    "default_rust_predictions_sha256": sha256_hex(&def_raw),
                    "partition_point_rust_bitwise_equal": pp_rs_equal,
                    "partition_point_rust_max_absolute_delta": pp_rs_max_delta,
                    "partition_point_rust_predictions_sha256": sha256_hex(&pp_rs_raw),
                }));
            }

            replayed_champions.push(json!({
                "index": case.index,
                "dataset": case.dataset,
                "context_sha256": case.context_sha256,
                "arm": case.arm,
                "program_sha256": case.program_sha256,
                "genome": case.genome,
                "training_csv_sha256": case.expected_train_csv_sha256,
                "default_emitted_rust_sha256": sha256_hex(default_emitted.rust.as_bytes()),
                "partition_point_emitted_rust_sha256": sha256_hex(pp_emitted.rust.as_bytes()),
                "partition_point_emitted_python_sha256": sha256_hex(pp_emitted.python.as_bytes()),
                "default_emit_options_passed": false,
                "partition_point_rust_passed": true,
                "cuts": cut_records,
            }));
        }

        if let Ok(out_path) = std::env::var("CLOVER_EXP008_REPLAY_OUT") {
            let full_doc = json!({
                "schema": "clover.exp008-quantile-parity-replay/v1",
                "qualification": "NON-QUAL; offline evidence-only diagnostic replay of the 3 pinned EXP-008 parity-failing champions on original context 14 and 18 training inner cuts; default EmitOptions and R5 parity charging unchanged",
                "evidence_only": true,
                "default_emitter_changed": false,
                "r5_parity_charging_changed": false,
                "h1_established": false,
                "qualified_rewards": 0,
                "source_readout": "docs/internal/clover/gp-h1-readout.20261013.json",
                "source_readout_sha256": sha256_hex(EXP008_READOUT.as_bytes()),
                "export_manifest_sha256": EXPECTED_EXPORT_MANIFEST_SHA256,
                "fold_seed": 20261013,
                "folds": 5,
                "cuts_evaluated": 2,
                "champions_replayed": 3,
                "default_emit_options_bitwise_pass_cuts": 3,
                "default_emit_options_bitwise_fail_cuts": 3,
                "partition_point_rust_bitwise_pass_cuts": 6,
                "partition_point_rust_bitwise_fail_cuts": 0,
                "champions": replayed_champions,
            });
            fs::write(
                &out_path,
                serde_json::to_string_pretty(&full_doc).unwrap() + "\n",
            )
            .unwrap();
        } else {
            assert_eq!(
                replayed_champions,
                *replay_receipt["champions"].as_array().unwrap(),
                "live EXP-008 replay must match committed exp-008-quantile-parity-replay.20261017.json"
            );
        }
    }
}
