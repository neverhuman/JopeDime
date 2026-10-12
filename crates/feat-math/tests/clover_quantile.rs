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
            assert_bits(&c.name, edges, &[c.edges.clone()]);
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
