use super::*;
use feat_math::{
    dsl::{ParamValue, Stage},
    emit_bound, fit_table,
    ops::FittedOp,
    ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
};

#[test]
fn bound_emitters_match_partition_point_on_tie_plateaus() {
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

    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&root).unwrap();
    let tmp = tempfile::tempdir_in(root).unwrap();
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cpp_driver = fs::read_to_string(repo_root.join("deploy/ci/emitters/driver.cpp")).unwrap();
    let js_driver = fs::read_to_string(repo_root.join("deploy/ci/emitters/driver.js")).unwrap();

    // Verify `advanced_v3::emit::emit_bound` (bound Python & Rust) on the 24x3 dataset
    // and directly test bound Python's `lower_bound` on the three non-monotone edge arrays.
    {
        let schema = DataSchema {
            columns: vec![
                ColumnSpec::numeric("c0"),
                ColumnSpec::numeric("c1"),
                ColumnSpec::numeric("c2"),
            ],
            row_semantics: RowSemantics::Iid,
        };
        let to_raw_table = |rows: &[Vec<f64>]| -> RawTable {
            RawTable::new(
                vec!["c0".into(), "c1".into(), "c2".into()],
                rows.iter()
                    .map(|r| {
                        vec![
                            format!("{:.17e}", r[0]),
                            format!("{:.17e}", r[1]),
                            format!("{:.17e}", r[2]),
                        ]
                    })
                    .collect(),
            )
            .unwrap()
        };
        let train_table = to_raw_table(&train);
        let holdout_table = to_raw_table(&holdout);
        let v3_genome = GenomeIr::parse(
            "(pipe X (input numeric=median categorical=frequency) (target_bin_encode bins=13 prior=10 oof=5) (ridge))",
        )
        .unwrap();
        let v3_model = fit_table(
            &v3_genome,
            &TaskSpec::regression(Objective::R2).unwrap(),
            &schema,
            &train_table,
            &y,
        )
        .unwrap();
        // Verify directly on the fitted BoundModel that its `TargetBinEncode(bins=13)`
        // edges on numeric column `c2` (`edges[4]`, since `input` with default
        // `missing=indicator` emits `[c0, c0_missing, c1, c1_missing, c2, c2_missing]`)
        // contain the 1-ULP non-monotone dip and diverge between legacy linear scan
        // (`sum(edge < v for edge in edges)`) and `partition_point`.
        let v3_model_json = serde_json::to_value(&v3_model).unwrap();
        let v3_tbe_edges: Vec<f64> = serde_json::from_value(
            v3_model_json["predictor"]["Pipeline"]["ops"][0]["Legacy"]["TargetBinEncode"]["edges"]
                [4]
            .clone(),
        )
        .unwrap();
        assert!(
            v3_tbe_edges.windows(2).any(|w| w[0] > w[1]),
            "fitted BoundModel TargetBinEncode edges[4] (c2) must contain the 1-ULP non-monotone dip"
        );
        assert_ne!(
            v3_tbe_edges.iter().filter(|&&e| e < v2).count(),
            v3_tbe_edges.partition_point(|e| *e < v2),
            "fitted BoundModel TargetBinEncode edges[4] (c2) must diverge between default sum() and partition_point"
        );

        let v3_expected = v3_model.predict_table(&holdout_table).unwrap();
        let bound_csv_path = tmp.path().join("bound_test.csv");
        let mut bound_csv = String::from("c0,c1,c2\n");
        for row in &holdout {
            bound_csv.push_str(&format!(
                "{:.17e},{:.17e},{:.17e}\n",
                row[0], row[1], row[2]
            ));
        }
        fs::write(&bound_csv_path, &bound_csv).unwrap();

        // 1. Default `EmitOptions::default()` preserves legacy bound Python bytes and
        // `emitter_version: "2.2.0"`, which diverges numerically on the 1-ULP tie plateau.
        let bound_default = emit_bound(&v3_model, &EmitOptions::default()).unwrap();
        let bound_default_manifest: serde_json::Value =
            serde_json::from_str(&bound_default.manifest).unwrap();
        assert_eq!(bound_default_manifest["emitter_version"], "2.2.0");
        assert!(!bound_default.python.contains("def lower_bound("));
        assert!(bound_default
            .python
            .contains("sum(edge < value for edge in edges)"));
        let bound_default_py_path = tmp.path().join("bound_v3_default.py");
        fs::write(&bound_default_py_path, &bound_default.python).unwrap();
        let bound_default_py_out = Command::new("python3")
            .arg(&bound_default_py_path)
            .arg(&bound_csv_path)
            .output()
            .unwrap();
        assert!(
            bound_default_py_out.status.success(),
            "bound v3 default py: {}",
            String::from_utf8_lossy(&bound_default_py_out.stderr)
        );
        let bound_default_py_preds: Vec<f64> =
            String::from_utf8_lossy(&bound_default_py_out.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().parse::<f64>().unwrap())
                .collect();
        let bound_default_err =
            check_finite_and_pointwise_equal(&v3_expected, &bound_default_py_preds, false)
                .expect_err("default legacy bound v3 Python must diverge on 1-ULP tie plateau");
        assert!(
            bound_default_err.contains("value mismatch")
                || bound_default_err.contains("bit mismatch"),
            "{bound_default_err}"
        );

        // 2. Opt-in `partition_point_quantile_bounds: true` emits `lower_bound` with
        // `emitter_version: "feat-math-quantile-partition-point-v1"` and matches bitwise.
        let bound = emit_bound(
            &v3_model,
            &EmitOptions {
                partition_point_quantile_bounds: true,
                ..EmitOptions::default()
            },
        )
        .unwrap();
        let bound_manifest: serde_json::Value = serde_json::from_str(&bound.manifest).unwrap();
        assert_eq!(
            bound_manifest["emitter_version"],
            "feat-math-quantile-partition-point-v1"
        );
        assert_ne!(bound_default.python, bound.python);
        assert_eq!(bound_default.rust, bound.rust);
        let bound_py_path = tmp.path().join("bound_v3.py");
        let bound_rs_path = tmp.path().join("bound_v3.rs");
        let bound_rs_bin = tmp.path().join("bound_v3_bin");
        fs::write(&bound_py_path, &bound.python).unwrap();
        fs::write(&bound_rs_path, &bound.rust).unwrap();

        let bound_py_out = Command::new("python3")
            .arg(&bound_py_path)
            .arg(&bound_csv_path)
            .output()
            .unwrap();
        assert!(
            bound_py_out.status.success(),
            "bound v3 py: {}",
            String::from_utf8_lossy(&bound_py_out.stderr)
        );
        let bound_py_preds: Vec<f64> = String::from_utf8_lossy(&bound_py_out.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f64>().unwrap())
            .collect();
        check_finite_and_pointwise_equal(&v3_expected, &bound_py_preds, false)
            .expect("opt-in bound v3 Python must match runtime bitwise");

        let bound_rs_compile = Command::new("rustc")
            .args(["--edition=2021", "-O", "-Awarnings"])
            .arg(&bound_rs_path)
            .arg("-o")
            .arg(&bound_rs_bin)
            .output()
            .unwrap();
        assert!(
            bound_rs_compile.status.success(),
            "bound v3 rustc: {}",
            String::from_utf8_lossy(&bound_rs_compile.stderr)
        );
        let bound_rs_out = Command::new(&bound_rs_bin)
            .arg(&bound_csv_path)
            .output()
            .unwrap();
        assert!(
            bound_rs_out.status.success(),
            "bound v3 rs: {}",
            String::from_utf8_lossy(&bound_rs_out.stderr)
        );
        let bound_rs_preds: Vec<f64> = String::from_utf8_lossy(&bound_rs_out.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f64>().unwrap())
            .collect();
        check_finite_and_pointwise_equal(&v3_expected, &bound_rs_preds, false)
            .expect("bound v3 Rust must match runtime bitwise");

        // Also invoke `lower_bound` from the emitted `bound_v3.py` module directly on the
        // non-monotone 1-ULP edge arrays for `bins=27`, `bins=16`, and `bins=13`.
        let probes_json = serde_json::to_string(
            &non_monotone_edges
                .iter()
                .map(|(_, _, probe, edges, expected_idx)| (edges, *probe, *expected_idx))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let py_probe = Command::new("python3")
            .args([
                "-c",
                "import json, runpy, sys\nmod = runpy.run_path(sys.argv[1], run_name='bound_v3_mod')\nlb = mod['lower_bound']\nfor edges, probe, expected in json.loads(sys.argv[2]):\n    got = lb(edges, probe)\n    assert got == expected, f'bound lower_bound got {got}, expected {expected}'\n",
                bound_py_path.to_str().unwrap(),
                &probes_json,
            ])
            .output()
            .unwrap();
        assert!(
            py_probe.status.success(),
            "bound v3 lower_bound non-monotone probe failed: {}",
            String::from_utf8_lossy(&py_probe.stderr)
        );
    }
}
