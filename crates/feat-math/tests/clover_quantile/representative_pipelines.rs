use super::*;
use feat_math::{
    fit,
    transpile::{emit_cpp17, emit_javascript},
};

#[test]
fn representative_pipelines_match_partition_point_across_emitters() {
    let check_finite_and_pointwise_equal = |expected: &[f64],
                                            actual: &[f64],
                                            allow_signed_zero_equivalence: bool|
     -> Result<(), String> {
        if expected.len() != actual.len() {
            return Err(format!(
                "length mismatch: expected {}, got {}",
                expected.len(),
                actual.len()
            ));
        }
        for (idx, (&e, &a)) in expected.iter().zip(actual).enumerate() {
            if !e.is_finite() {
                return Err(format!("row {idx}: expected prediction {e} is not finite"));
            }
            if !(e == a || (e.is_nan() && a.is_nan())) {
                return Err(format!(
                    "row {idx}: value mismatch expected={e:?} actual={a:?}"
                ));
            }
            if e.to_bits() != a.to_bits()
                && !(allow_signed_zero_equivalence && e == 0.0 && a == 0.0)
            {
                return Err(format!(
                    "row {idx}: bit mismatch expected={:#018x} ({e:?}) actual={:#018x} ({a:?})",
                    e.to_bits(),
                    a.to_bits()
                ));
            }
        }
        Ok(())
    };
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

    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&root).unwrap();
    let tmp = tempfile::tempdir_in(root).unwrap();
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cpp_driver = fs::read_to_string(repo_root.join("deploy/ci/emitters/driver.cpp")).unwrap();
    let js_driver = fs::read_to_string(repo_root.join("deploy/ci/emitters/driver.js")).unwrap();

    // Verify `advanced_v3::emit::emit_bound` (bound Python & Rust) on the 24x3 dataset
    // and directly test bound Python's `lower_bound` on the three non-monotone edge arrays.
    {
        let mut train_csv = String::new();
        for (row, target) in train.iter().zip(&y) {
            train_csv.push_str(&format!(
                "{:.17e},{:.17e},{:.17e},{:.17e}\n",
                row[0], row[1], row[2], target
            ));
        }
        let mut test_csv = String::new();
        for row in &holdout {
            test_csv.push_str(&format!(
                "{:.17e},{:.17e},{:.17e}\n",
                row[0], row[1], row[2]
            ));
        }
        let train_path = tmp.path().join("train.csv");
        let test_path = tmp.path().join("test_features.csv");
        fs::write(&train_path, &train_csv).unwrap();
        fs::write(&test_path, &test_csv).unwrap();

        // Representative pipelines using the EXP-008 failing operator families
        // (`quantile_bins`, `bin_onehot`, `target_bin_encode`), NOT the pinned
        // EXP-008 champion genomes. In the `target_bin_encode` cases, `(target_bin_encode bins=13)`
        // runs on the raw 24x3 columns first so column 2 (`v2 = 31.0 / 19.0`) directly hits the
        // 1-ULP tie-plateau dip and diverges under the legacy `(lo, hi)` helper.
        let representative_cases = [
        (
            "rep_exp008_ctx14_quantile_bins_elasticnet",
            "(pipe X (quantile_bins append=false bins=27) (quantile_bins append=true bins=25) (quantile_bins append=true bins=11) (cheb degree=7) (elasticnet alpha=0.05 l1_ratio=0.9))",
            true,
        ),
        (
            "rep_exp008_ctx18_bin_onehot_histgbm2",
            "(pipe X (bin_onehot append=false bins=16) (blom) (histgbm2 bags=2 leaves=15 lr=0.08 max_iter=110 min_leaf=1))",
            false,
        ),
        (
            "rep_exp008_ctx18_target_bin_encode_histgbm",
            "(pipe X (target_bin_encode bins=13) (cheb degree=3) (poly_interact append=true) (histgbm leaves=15 lr=0.08 max_iter=60 min_leaf=1))",
            false,
        ),
        (
            "rep_exp008_ctx18_bin_onehot_ridge_4lang",
            "(pipe X (bin_onehot append=false bins=16) (blom) (ridge))",
            true,
        ),
        (
            "rep_exp008_ctx18_target_bin_encode_ridge_4lang",
            "(pipe X (target_bin_encode bins=13) (cheb degree=3) (poly_interact append=true) (ridge))",
            true,
        ),
    ];

        let parse_preds = |path: &std::path::Path, label: &str| -> Vec<f64> {
            fs::read_to_string(path)
                .expect(label)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().parse::<f64>().expect(label))
                .collect()
        };

        let fixed_opts = EmitOptions {
            partition_point_quantile_bounds: true,
            ..EmitOptions::default()
        };

        for (name, dsl, check_cpp_js) in representative_cases {
            let genome = GenomeIr::parse(dsl).expect(name);
            let native_preds = fit(&genome, &train, &y)
                .expect(name)
                .predict(&holdout)
                .expect(name);

            // First verify that legacy `EmitOptions::default()` (the base `aad22f8` `(lo, hi)`
            // helper) diverges numerically (`value mismatch` or `bit mismatch`) from `native_preds`
            // on every representative pipeline across every covered language (Python, Rust, and
            // C++17 + JS for the 4-language heads).
            let legacy_emitted = emit(&genome, &EmitOptions::default()).expect(name);
            let legacy_py_path = tmp.path().join(format!("{name}_legacy.py"));
            let legacy_py_pred_path = tmp.path().join(format!("{name}_legacy_py_pred.csv"));
            fs::write(&legacy_py_path, &legacy_emitted.python).unwrap();
            let legacy_py_out = Command::new("python3")
                .arg(&legacy_py_path)
                .arg(&train_path)
                .arg(&test_path)
                .arg(&legacy_py_pred_path)
                .output()
                .expect("python3");
            assert!(
                legacy_py_out.status.success(),
                "{name} legacy py: {}",
                String::from_utf8_lossy(&legacy_py_out.stderr)
            );
            let legacy_py_preds = parse_preds(&legacy_py_pred_path, name);
            let legacy_py_err =
                check_finite_and_pointwise_equal(&native_preds, &legacy_py_preds, false)
                    .expect_err(&format!(
                        "{name}: legacy Python helper must diverge from native partition_point"
                    ));
            assert!(
                legacy_py_err.contains("value mismatch") || legacy_py_err.contains("bit mismatch"),
                "{legacy_py_err}"
            );

            let legacy_rs_path = tmp.path().join(format!("{name}_legacy.rs"));
            let legacy_rs_bin = tmp.path().join(format!("{name}_legacy_rs"));
            let legacy_rs_pred_path = tmp.path().join(format!("{name}_legacy_rs_pred.csv"));
            fs::write(&legacy_rs_path, &legacy_emitted.rust).unwrap();
            let legacy_rs_compile = Command::new("rustc")
                .args(["--edition=2021", "-O", "-Awarnings"])
                .arg(&legacy_rs_path)
                .arg("-o")
                .arg(&legacy_rs_bin)
                .output()
                .expect("rustc");
            assert!(
                legacy_rs_compile.status.success(),
                "{name} legacy rustc: {}",
                String::from_utf8_lossy(&legacy_rs_compile.stderr)
            );
            let legacy_rs_out = Command::new(&legacy_rs_bin)
                .arg(&train_path)
                .arg(&test_path)
                .arg(&legacy_rs_pred_path)
                .output()
                .expect("legacy emitted rust bin");
            assert!(
                legacy_rs_out.status.success(),
                "{name} legacy rs: {}",
                String::from_utf8_lossy(&legacy_rs_out.stderr)
            );
            let legacy_rs_preds = parse_preds(&legacy_rs_pred_path, name);
            let legacy_rs_err =
                check_finite_and_pointwise_equal(&native_preds, &legacy_rs_preds, false)
                    .expect_err(&format!(
                        "{name}: legacy Rust helper must diverge from native partition_point"
                    ));
            assert!(
                legacy_rs_err.contains("value mismatch") || legacy_rs_err.contains("bit mismatch"),
                "{legacy_rs_err}"
            );

            if check_cpp_js {
                let mut legacy_cpp_src = emit_cpp17(&genome, &EmitOptions::default()).expect(name);
                legacy_cpp_src.push_str(&cpp_driver);
                let legacy_cpp_path = tmp.path().join(format!("{name}_legacy.cpp"));
                let legacy_cpp_bin = tmp.path().join(format!("{name}_legacy_cpp"));
                let legacy_cpp_pred_path = tmp.path().join(format!("{name}_legacy_cpp_pred.csv"));
                fs::write(&legacy_cpp_path, &legacy_cpp_src).unwrap();
                let legacy_cpp_compile = Command::new("g++")
                    .args(["-std=c++17", "-O2", "-ffp-contract=off"])
                    .arg(&legacy_cpp_path)
                    .arg("-o")
                    .arg(&legacy_cpp_bin)
                    .output()
                    .expect("g++");
                assert!(
                    legacy_cpp_compile.status.success(),
                    "{name} legacy g++: {}",
                    String::from_utf8_lossy(&legacy_cpp_compile.stderr)
                );
                let legacy_cpp_out = Command::new(&legacy_cpp_bin)
                    .arg(&train_path)
                    .arg(&test_path)
                    .arg(&legacy_cpp_pred_path)
                    .output()
                    .expect("legacy emitted cpp bin");
                assert!(
                    legacy_cpp_out.status.success(),
                    "{name} legacy cpp: {}",
                    String::from_utf8_lossy(&legacy_cpp_out.stderr)
                );
                let legacy_cpp_preds = parse_preds(&legacy_cpp_pred_path, name);
                let legacy_cpp_err =
                    check_finite_and_pointwise_equal(&native_preds, &legacy_cpp_preds, false)
                        .expect_err(&format!(
                            "{name}: legacy C++17 helper must diverge from native partition_point"
                        ));
                assert!(
                    legacy_cpp_err.contains("value mismatch")
                        || legacy_cpp_err.contains("bit mismatch"),
                    "{legacy_cpp_err}"
                );

                let mut legacy_js_src =
                    emit_javascript(&genome, &EmitOptions::default()).expect(name);
                legacy_js_src.push_str(&js_driver);
                let legacy_js_path = tmp.path().join(format!("{name}_legacy.js"));
                let legacy_js_pred_path = tmp.path().join(format!("{name}_legacy_js_pred.csv"));
                fs::write(&legacy_js_path, &legacy_js_src).unwrap();
                let legacy_js_out = Command::new("node")
                    .arg(&legacy_js_path)
                    .arg(&train_path)
                    .arg(&test_path)
                    .arg(&legacy_js_pred_path)
                    .output()
                    .expect("node");
                assert!(
                    legacy_js_out.status.success(),
                    "{name} legacy js: {}",
                    String::from_utf8_lossy(&legacy_js_out.stderr)
                );
                let legacy_js_preds = parse_preds(&legacy_js_pred_path, name);
                let legacy_js_err =
                    check_finite_and_pointwise_equal(&native_preds, &legacy_js_preds, true)
                        .expect_err(&format!(
                            "{name}: legacy JS helper must diverge from native partition_point"
                        ));
                assert!(
                    legacy_js_err.contains("value mismatch")
                        || legacy_js_err.contains("bit mismatch"),
                    "{legacy_js_err}"
                );
            }

            let emitted = emit(&genome, &fixed_opts).expect(name);

            // 1. Emitted Python (`partition_point_quantile_bounds: true`)
            let py_path = tmp.path().join(format!("{name}.py"));
            let py_pred_path = tmp.path().join(format!("{name}_py_pred.csv"));
            fs::write(&py_path, &emitted.python).unwrap();
            let py_out = Command::new("python3")
                .arg(&py_path)
                .arg(&train_path)
                .arg(&test_path)
                .arg(&py_pred_path)
                .output()
                .expect("python3");
            assert!(
                py_out.status.success(),
                "{name} py: {}",
                String::from_utf8_lossy(&py_out.stderr)
            );
            let py_preds = parse_preds(&py_pred_path, name);
            check_finite_and_pointwise_equal(&native_preds, &py_preds, false)
                .unwrap_or_else(|e| panic!("{name} Python parity failed: {e}"));

            // 2. Emitted Rust (`partition_point_quantile_bounds: true`)
            let rs_path = tmp.path().join(format!("{name}.rs"));
            let rs_bin = tmp.path().join(format!("{name}_rs"));
            let rs_pred_path = tmp.path().join(format!("{name}_rs_pred.csv"));
            fs::write(&rs_path, &emitted.rust).unwrap();
            let rs_compile = Command::new("rustc")
                .args(["--edition=2021", "-O", "-Awarnings"])
                .arg(&rs_path)
                .arg("-o")
                .arg(&rs_bin)
                .output()
                .expect("rustc");
            assert!(
                rs_compile.status.success(),
                "{name} rustc: {}",
                String::from_utf8_lossy(&rs_compile.stderr)
            );
            let rs_out = Command::new(&rs_bin)
                .arg(&train_path)
                .arg(&test_path)
                .arg(&rs_pred_path)
                .output()
                .expect("emitted rust bin");
            assert!(
                rs_out.status.success(),
                "{name} rs: {}",
                String::from_utf8_lossy(&rs_out.stderr)
            );
            let rs_preds = parse_preds(&rs_pred_path, name);
            check_finite_and_pointwise_equal(&native_preds, &rs_preds, false)
                .unwrap_or_else(|e| panic!("{name} Rust parity failed: {e}"));

            if check_cpp_js {
                // 3. Emitted C++17 (`partition_point_quantile_bounds: true`)
                let mut cpp_src = emit_cpp17(&genome, &fixed_opts).expect(name);
                cpp_src.push_str(&cpp_driver);
                let cpp_path = tmp.path().join(format!("{name}.cpp"));
                let cpp_bin = tmp.path().join(format!("{name}_cpp"));
                let cpp_pred_path = tmp.path().join(format!("{name}_cpp_pred.csv"));
                fs::write(&cpp_path, &cpp_src).unwrap();
                let cpp_compile = Command::new("g++")
                    .args(["-std=c++17", "-O2", "-ffp-contract=off"])
                    .arg(&cpp_path)
                    .arg("-o")
                    .arg(&cpp_bin)
                    .output()
                    .expect("g++");
                assert!(
                    cpp_compile.status.success(),
                    "{name} g++: {}",
                    String::from_utf8_lossy(&cpp_compile.stderr)
                );
                let cpp_out = Command::new(&cpp_bin)
                    .arg(&train_path)
                    .arg(&test_path)
                    .arg(&cpp_pred_path)
                    .output()
                    .expect("emitted cpp bin");
                assert!(
                    cpp_out.status.success(),
                    "{name} cpp: {}",
                    String::from_utf8_lossy(&cpp_out.stderr)
                );
                let cpp_preds = parse_preds(&cpp_pred_path, name);
                check_finite_and_pointwise_equal(&native_preds, &cpp_preds, false)
                    .unwrap_or_else(|e| panic!("{name} C++17 parity failed: {e}"));

                // 4. Emitted JavaScript (`partition_point_quantile_bounds: true`)
                let mut js_src = emit_javascript(&genome, &fixed_opts).expect(name);
                js_src.push_str(&js_driver);
                let js_path = tmp.path().join(format!("{name}.js"));
                let js_pred_path = tmp.path().join(format!("{name}_js_pred.csv"));
                fs::write(&js_path, &js_src).unwrap();
                let js_out = Command::new("node")
                    .arg(&js_path)
                    .arg(&train_path)
                    .arg(&test_path)
                    .arg(&js_pred_path)
                    .output()
                    .expect("node");
                assert!(
                    js_out.status.success(),
                    "{name} js: {}",
                    String::from_utf8_lossy(&js_out.stderr)
                );
                let js_preds = parse_preds(&js_pred_path, name);
                check_finite_and_pointwise_equal(&native_preds, &js_preds, true)
                    .unwrap_or_else(|e| panic!("{name} JS parity failed: {e}"));
            }
        }
    }
}
