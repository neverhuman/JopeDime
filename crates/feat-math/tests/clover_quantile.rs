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

    let check_finite_and_pointwise_equal =
        |expected: &[f64], actual: &[f64], allow_signed_zero_equivalence: bool| -> Result<(), String> {
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
                    return Err(format!("row {idx}: value mismatch expected={e:?} actual={a:?}"));
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
        let v3_expected = v3_model.predict_table(&holdout_table).unwrap();
        let bound = emit_bound(&v3_model, &EmitOptions::default()).unwrap();
        let bound_py_path = tmp.path().join("bound_v3.py");
        let bound_rs_path = tmp.path().join("bound_v3.rs");
        let bound_rs_bin = tmp.path().join("bound_v3_bin");
        let bound_csv_path = tmp.path().join("bound_test.csv");
        let mut bound_csv = String::from("c0,c1,c2\n");
        for row in &holdout {
            bound_csv.push_str(&format!("{:.17e},{:.17e},{:.17e}\n", row[0], row[1], row[2]));
        }
        fs::write(&bound_py_path, &bound.python).unwrap();
        fs::write(&bound_rs_path, &bound.rust).unwrap();
        fs::write(&bound_csv_path, &bound_csv).unwrap();

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
            .expect("bound v3 Python must match runtime bitwise");

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
        // helper) diverges from `native_preds` on every representative pipeline across
        // every covered language.
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
        assert!(legacy_py_err.contains("mismatch"), "{legacy_py_err}");

        if check_cpp_js {
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
            assert!(
                check_finite_and_pointwise_equal(&native_preds, &legacy_js_preds, true).is_err(),
                "{name}: legacy JS helper must diverge from native partition_point"
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
