//! Independent semantic audit; implementation agreement alone cannot pass.
use feat_math::{
    dsl::{ParamValue, Stage},
    ops::{apply_op, fit_op, FittedOp},
    transpile::{emit, EmitOptions},
    GenomeIr,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, process::Command};

const CONTRACT: &str = include_str!("fixtures/clover_quantiles.v1.json");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Contract {
    schema: String,
    oracle: String,
    input_contract: String,
    edge_rule: String,
    tie_rule: String,
    index_rule: String,
    onehot_rule: String,
    backends: Vec<String>,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    train: Vec<f64>,
    bins: usize,
    edges: Vec<f64>,
    probe: Vec<f64>,
    indices: Vec<usize>,
}
fn contract() -> Contract {
    assert_eq!(
        format!("{:x}", Sha256::digest(CONTRACT.as_bytes())),
        "3c6367e174753dc8c5c83858659c5ae8ccbb84a3dae6d829263a30842249223e",
        "v1 semantics are immutable; changes require a new parity-contract version"
    );
    let c: Contract = serde_json::from_str(CONTRACT).unwrap();
    assert_eq!(c.schema, "clover.quantile-parity/v1");
    assert!(c.oracle.starts_with("Hand-derived"));
    assert!(!c.input_contract.is_empty());
    assert_eq!(c.edge_rule, "linear_n_minus_one");
    assert_eq!(c.tie_rule, "lower_bound_strict_less");
    assert_eq!(c.index_rule, "index_divided_by_bins_minus_one");
    assert_eq!(c.onehot_rule, "one_at_lower_bound_index");
    assert_eq!(
        c.backends,
        vec!["runtime", "emitted_rust", "emitted_python"]
    );
    c
}
fn expected(c: &Case) -> Vec<Vec<f64>> {
    c.indices
        .iter()
        .map(|&i| {
            let mut row = vec![i as f64 / (c.bins - 1) as f64];
            row.extend((0..c.bins).map(|j| if j == i { 1. } else { 0. }));
            row
        })
        .collect()
}
fn assert_bits(name: &str, actual: &[Vec<f64>], expected: &[Vec<f64>]) {
    let bits = |rows: &[Vec<f64>]| {
        rows.iter()
            .map(|r| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(actual),
        bits(expected),
        "quantile semantic drift: {name}"
    );
}

#[test]
fn runtime_matches_independent_quantile_and_binning_oracles() {
    for c in contract().cases {
        let train = c.train.iter().map(|&x| vec![x]).collect();
        let probes = c.probe.iter().map(|&x| vec![x]).collect();
        let mut actual = Vec::new();
        for op in ["quantile_bins", "bin_onehot"] {
            let mut stage = Stage::new(op);
            stage
                .params
                .insert("bins".into(), ParamValue::Int(c.bins as i64));
            let fitted = fit_op(&stage, &train, &[]).unwrap();
            let edges = match &fitted {
                FittedOp::QuantileBins { edges, .. } | FittedOp::BinOnehot { edges, .. } => edges,
                _ => unreachable!(),
            };
            assert_bits(&c.name, edges, std::slice::from_ref(&c.edges));
            let output = apply_op(&fitted, &probes).unwrap();
            if actual.is_empty() {
                actual = output;
            } else {
                for (row, onehot) in actual.iter_mut().zip(output) {
                    row.extend(onehot);
                }
            }
        }
        assert_bits(&c.name, &actual, &expected(&c));
    }
}

#[test]
fn versioned_contract_detects_shared_wrong_tie_semantics() {
    let c = contract()
        .cases
        .into_iter()
        .find(|c| c.name.starts_with("duplicate"))
        .unwrap();
    let wrong = c
        .probe
        .iter()
        .map(|v| c.edges.iter().filter(|edge| **edge <= *v).count())
        .collect::<Vec<_>>();
    assert_ne!(
        wrong, c.indices,
        "fixture must reject a shared bisect_right error"
    );
    let quantiles =
        [0., 0.25, 0.5, 0.75, 1.].map(|q| feat_math::ops::quantile_sorted(&[-4., 0., 4., 8.], q));
    assert_eq!(quantiles, [-4., -1., 2., 5., 8.]);
}

#[test]
fn emitted_rust_and_python_match_independent_oracles() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&root).unwrap();
    let tmp = tempfile::tempdir_in(root).unwrap();
    let index = emit(
        &GenomeIr::parse("(pipe X (quantile_bins bins=4) (ridge))").unwrap(),
        &EmitOptions::default(),
    )
    .unwrap();
    let onehot = emit(
        &GenomeIr::parse("(pipe X (bin_onehot bins=4) (ridge))").unwrap(),
        &EmitOptions::default(),
    )
    .unwrap();
    let py_index = tmp.path().join("index.py");
    let py_onehot = tmp.path().join("onehot.py");
    fs::write(&py_index, &index.python).unwrap();
    fs::write(&py_onehot, &onehot.python).unwrap();
    let fixture = tmp.path().join("contract.json");
    fs::write(&fixture, CONTRACT).unwrap();
    let driver = r#"import json, runpy, sys
i, o = runpy.run_path(sys.argv[1]), runpy.run_path(sys.argv[2])
for c in json.load(open(sys.argv[3]))['cases']:
    x = [[v] for v in c['train']]; p = [[v] for v in c['probe']]; b = c['bins']
    edges = i['quantile_bin_indexer_fit'](x,b)
    assert edges == [c['edges']], c['name']
    assert o['quantile_onehot_encoder_fit'](x,b) == edges
    a = i['quantile_bin_indexer_apply'](edges,p,b,False)
    h = o['quantile_onehot_encoder_apply'](edges,p,b,False)
    print(json.dumps([left+right for left,right in zip(a,h)]))
"#;
    let py = Command::new("python3")
        .args(["-c", driver])
        .arg(&py_index)
        .arg(&py_onehot)
        .arg(&fixture)
        .output()
        .expect("python3 is required for this parity contract");
    assert!(
        py.status.success(),
        "{}",
        String::from_utf8_lossy(&py.stderr)
    );
    let prefix = |s: &str| {
        s.split_once("fn main() {")
            .expect("emitted entrypoint")
            .0
            .to_owned()
    };
    // Separate modules retain the actual emitted helpers without duplicate helper symbols.
    let mut source = format!("mod idx {{ {} pub fn check(x: &[Vec<f64>], p: &[Vec<f64>], b: usize) -> (Vec<Vec<f64>>,Vec<Vec<f64>>) {{ let e = quantile_bin_indexer_fit(x,b); (e.edges.clone(),quantile_bin_indexer_apply(&e,p,b,false)) }} }}\nmod hot {{ {} pub fn check(x: &[Vec<f64>], p: &[Vec<f64>], b: usize) -> Vec<Vec<f64>> {{ let e = quantile_onehot_encoder_fit(x,b); quantile_onehot_encoder_apply(&e,p,b,false) }} }}\nfn main() {{\n",prefix(&index.rust).replace("#![","#["),prefix(&onehot.rust).replace("#![","#["));
    for c in &contract().cases {
        let train = c
            .train
            .iter()
            .map(|v| format!("vec![{v:?}]"))
            .collect::<Vec<_>>()
            .join(",");
        let probes = c
            .probe
            .iter()
            .map(|v| format!("vec![{v:?}]"))
            .collect::<Vec<_>>()
            .join(",");
        source.push_str(&format!("let x = vec![{train}]; let p = vec![{probes}]; let (edges,a) = idx::check(&x,&p,{}); assert_eq!(edges,vec![vec!{:?}]); let h = hot::check(&x,&p,{}); for (l,r) in a.into_iter().zip(h) {{ for v in l.into_iter().chain(r) {{ print!(\"{{}} \",v.to_bits()); }} println!(); }}\n",c.bins,c.edges,c.bins));
    }
    source.push_str("}\n");
    let rs = tmp.path().join("audit.rs");
    fs::write(&rs, source).unwrap();
    let bin = tmp.path().join("audit");
    let compile = Command::new("rustc")
        .args(["--edition=2021", "-O", "-Awarnings"])
        .arg(&rs)
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("rustc required");
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let output = Command::new(&bin).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rust_bits = String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse::<u64>().unwrap())
        .collect::<Vec<_>>();
    let expected_bits = contract()
        .cases
        .iter()
        .flat_map(expected)
        .flatten()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(rust_bits, expected_bits);
    let py_rows = String::from_utf8(py.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Vec<Vec<f64>>>(line).unwrap())
        .collect::<Vec<_>>();
    for (c, actual) in contract().cases.iter().zip(py_rows) {
        assert_bits(&c.name, &actual, &expected(c));
    }
}

#[test]
fn exp_next_04_tie_plateau_and_representative_pipelines_match_native_partition_point() {
    use feat_math::{
        emit_bound, fit, fit_table,
        transpile::{emit_cpp17, emit_javascript},
        ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
    };

    for tool in ["python3", "rustc", "g++", "node"] {
        let out = Command::new(tool)
            .arg("--version")
            .output()
            .unwrap_or_else(|e| panic!("{tool} must be available on PATH: {e}"));
        assert!(out.status.success(), "{tool} --version failed");
    }

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

    // Counterexample proving the old `fold(0.0_f64, f64::max)` reduction is NaN-blind
    // whereas `check_finite_and_pointwise_equal` rejects NaN-vs-finite and length mismatches.
    let nan_blind_max_abs = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    assert_eq!(nan_blind_max_abs(&[1.0, 2.0], &[1.0, f64::NAN]), 0.0);
    assert!(check_finite_and_pointwise_equal(&[1.0, 2.0], &[1.0, f64::NAN], false).is_err());
    assert!(check_finite_and_pointwise_equal(&[1.0], &[1.0, 2.0], false).is_err());
    assert!(check_finite_and_pointwise_equal(&[1.0, -0.0], &[1.0, 0.0], false).is_err());
    assert!(check_finite_and_pointwise_equal(&[1.0, -0.0], &[1.0, 0.0], true).is_ok());

    // Verify the versioned recipe transition fixture for `partition_point_quantile_bounds`.
    let transition_genome =
        GenomeIr::parse("(pipe X (quantile_bins append=false bins=27) (ridge))").unwrap();
    let legacy_transition_emit = emit(&transition_genome, &EmitOptions::default()).unwrap();
    let fixed_transition_emit = emit(
        &transition_genome,
        &EmitOptions {
            partition_point_quantile_bounds: true,
            ..EmitOptions::default()
        },
    )
    .unwrap();
    let legacy_manifest: serde_json::Value =
        serde_json::from_str(&legacy_transition_emit.manifest).unwrap();
    let fixed_manifest: serde_json::Value =
        serde_json::from_str(&fixed_transition_emit.manifest).unwrap();
    let source_transition: serde_json::Value = serde_json::from_slice(
        &fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/compat_freeze/transitions/exp_next_04_emitted_quantile_partition_point.v1.json"
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        source_transition["schema"],
        "feat-math.emitted-quantile-partition-point-transition.v1"
    );
    assert_eq!(source_transition["program"], fixed_manifest["program"]);
    assert_eq!(source_transition["before"], legacy_manifest);
    assert_eq!(source_transition["after"], fixed_manifest);
    assert_eq!(legacy_manifest["emitter_version"], "2.2.0");
    assert_eq!(
        fixed_manifest["emitter_version"],
        "feat-math-quantile-partition-point-v1"
    );
    assert_ne!(legacy_transition_emit.rust, fixed_transition_emit.rust);
    assert_ne!(legacy_transition_emit.python, fixed_transition_emit.python);

    // Verify unambiguous combined `Manifest.emitter_version` labels when
    // `partition_point_quantile_bounds` is enabled alongside `guard_zero_width_roll`
    // and/or `MathClip`, while `partition_point_quantile_bounds: false` retains
    // the exact legacy labels and bytes.
    for (dsl, guard_roll, expected_legacy, expected_opt_in) in [
        (
            "(pipe X (roll k=3) (quantile_bins bins=8) (ridge))",
            true,
            "feat-math-roll-width-guard-v1",
            "feat-math-roll-width-guard-v1+quantile-partition-point-v1",
        ),
        (
            "(pipe X (quantile_bins bins=8) (map expr=\"clip(x,-1,1)\") (ridge))",
            false,
            "feat-math-clip-safety-v2",
            "feat-math-quantile-partition-point-v1+clip-safety-v2",
        ),
        (
            "(pipe X (roll k=3) (quantile_bins bins=8) (map expr=\"clip(x,-1,1)\") (ridge))",
            true,
            "feat-math-roll-width-guard-v1",
            "feat-math-roll-width-guard-v1+quantile-partition-point-v1+clip-safety-v2",
        ),
    ] {
        let g = GenomeIr::parse(dsl).unwrap();
        let legacy_combo = emit(
            &g,
            &EmitOptions {
                guard_zero_width_roll: guard_roll,
                partition_point_quantile_bounds: false,
                ..EmitOptions::default()
            },
        )
        .unwrap();
        let opt_in_combo = emit(
            &g,
            &EmitOptions {
                guard_zero_width_roll: guard_roll,
                partition_point_quantile_bounds: true,
                ..EmitOptions::default()
            },
        )
        .unwrap();
        let legacy_combo_manifest: serde_json::Value =
            serde_json::from_str(&legacy_combo.manifest).unwrap();
        let opt_in_combo_manifest: serde_json::Value =
            serde_json::from_str(&opt_in_combo.manifest).unwrap();
        assert_eq!(legacy_combo_manifest["emitter_version"], expected_legacy);
        assert_eq!(opt_in_combo_manifest["emitter_version"], expected_opt_in);
        assert_ne!(legacy_combo.python, opt_in_combo.python);
        assert_ne!(legacy_combo.rust, opt_in_combo.rust);
    }

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
            check_finite_and_pointwise_equal(&native_preds, &legacy_py_preds, false).expect_err(
                &format!("{name}: legacy Python helper must diverge from native partition_point"),
            );
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
            check_finite_and_pointwise_equal(&native_preds, &legacy_rs_preds, false).expect_err(
                &format!("{name}: legacy Rust helper must diverge from native partition_point"),
            );
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

            let mut legacy_js_src = emit_javascript(&genome, &EmitOptions::default()).expect(name);
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
                check_finite_and_pointwise_equal(&native_preds, &legacy_js_preds, true).expect_err(
                    &format!("{name}: legacy JS helper must diverge from native partition_point"),
                );
            assert!(
                legacy_js_err.contains("value mismatch") || legacy_js_err.contains("bit mismatch"),
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

#[test]
fn exp_next_04_exp008_champion_parity_replay_matches_native_partition_point() {
    use feat_math::{fit, folds::fold_of, Model};
    use serde_json::{json, Value};
    use std::{io::Write, path::Path};

    const EXP008_READOUT: &str =
        include_str!("../../../docs/internal/clover/gp-h1-readout.20261013.json");
    const EXP008_REPLAY_RECEIPT: &str =
        include_str!("../../../docs/internal/clover/exp-008-quantile-parity-replay.20261017.json");
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

#[test]
fn numeric_v2_near_tie_edges_are_monotone_and_use_first_equal() {
    use feat_math::numeric_v2::{NumericModelV2, NumericProfileV2, TransformV2};
    let a = 0.10000000000000002_f64;
    let b = f64::from_bits(a.to_bits() + 1);
    let x = vec![vec![a], vec![a], vec![a], vec![b], vec![b]];
    let m = NumericModelV2::fit(
        NumericProfileV2::new(vec!["value".into()]),
        &GenomeIr::parse("(pipe X (quantile_bins bins=13) (ridge))").unwrap(),
        &x,
        &[0., 1., 2., 3., 4.],
    )
    .unwrap();
    let TransformV2::Quantile { edges, bins, .. } = &m.transforms[0] else {
        panic!()
    };
    assert!(edges[0].windows(2).all(|w| w[0] <= w[1]));
    let p = vec![
        vec![f64::from_bits(a.to_bits() - 1)],
        vec![a],
        vec![b],
        vec![f64::from_bits(b.to_bits() + 1)],
    ];
    let out = m.transforms[0].apply(&p).unwrap();
    for (row, probe) in out.iter().zip(&p) {
        // Independent linear count is valid because v2 fits a monotone edge sequence.
        let k = edges[0].iter().filter(|v| **v < probe[0]).count();
        assert_eq!(row[0], k as f64 / (*bins - 1) as f64);
    }
}
