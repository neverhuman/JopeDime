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
fn exp_next_04_tie_plateau_and_exp008_contexts_have_zero_emitter_delta() {
    use feat_math::{
        fit,
        transpile::{emit_cpp17, emit_javascript},
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
        let c1 = if i < 22 {
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

    // Verify the 1-ULP tie-plateau non-monotonicity and that the old `(lo, hi)`
    // `mid + 1` search diverges from `slice::partition_point` for all three bin counts.
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
        assert_ne!(
            legacy_lo_hi(edges, probe),
            edges.partition_point(|e| *e < probe),
            "col {col} (bins={bins}) fixture must reproduce old lo/hi vs partition_point divergence"
        );
    }

    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&root).unwrap();
    let tmp = tempfile::tempdir_in(root).unwrap();
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cpp_driver = fs::read_to_string(repo_root.join("deploy/ci/emitters/driver.cpp")).unwrap();
    let js_driver = fs::read_to_string(repo_root.join("deploy/ci/emitters/driver.js")).unwrap();

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

    // The 3 EXP-008 failing champion contexts:
    let exp008_cases = [
        (
            "ctx_14_retrieval_seeded",
            "(pipe X (quantile_bins append=false bins=27) (quantile_bins append=true bins=25) (quantile_bins append=true bins=11) (cheb degree=7) (elasticnet alpha=0.05 l1_ratio=0.9))",
            true,
        ),
        (
            "ctx_18_ordinary",
            "(pipe X (bin_onehot append=true bins=16) (blom) (histgbm2 bags=2 leaves=15 lr=0.08 max_iter=110 min_leaf=8))",
            false,
        ),
        (
            "ctx_18_policy_seeded",
            "(pipe X (cheb degree=3) (poly_interact append=true) (poly_interact append=true) (target_bin_encode bins=13) (histgbm leaves=15 lr=0.08 max_iter=60))",
            false,
        ),
        (
            "ctx_18_ordinary_4lang_head",
            "(pipe X (bin_onehot append=true bins=16) (blom) (ridge))",
            true,
        ),
        (
            "ctx_18_policy_seeded_4lang_head",
            "(pipe X (cheb degree=3) (poly_interact append=true) (poly_interact append=true) (target_bin_encode bins=13) (ridge))",
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
    let max_abs_delta = |a: &[f64], b: &[f64]| -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };

    for (name, dsl, check_cpp_js) in exp008_cases {
        let genome = GenomeIr::parse(dsl).expect(name);
        let native_preds = fit(&genome, &train, &y)
            .expect(name)
            .predict(&holdout)
            .expect(name);
        let emitted = emit(&genome, &EmitOptions::default()).expect(name);

        // 1. Emitted Python
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
        let py_delta = max_abs_delta(&native_preds, &py_preds);
        assert_eq!(
            py_delta, 0.0,
            "{name}: Python max_absolute_delta={py_delta} (expected 0.0)"
        );

        // 2. Emitted Rust
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
        let rs_delta = max_abs_delta(&native_preds, &rs_preds);
        assert_eq!(
            rs_delta, 0.0,
            "{name}: Rust max_absolute_delta={rs_delta} (expected 0.0)"
        );

        if check_cpp_js {
            // 3. Emitted C++17
            let mut cpp_src = emit_cpp17(&genome, &EmitOptions::default()).expect(name);
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
            let cpp_delta = max_abs_delta(&native_preds, &cpp_preds);
            assert_eq!(
                cpp_delta, 0.0,
                "{name}: C++17 max_absolute_delta={cpp_delta} (expected 0.0)"
            );

            // 4. Emitted JavaScript
            let mut js_src = emit_javascript(&genome, &EmitOptions::default()).expect(name);
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
            let js_delta = max_abs_delta(&native_preds, &js_preds);
            assert_eq!(
                js_delta, 0.0,
                "{name}: JS max_absolute_delta={js_delta} (expected 0.0)"
            );
        }
    }
}
