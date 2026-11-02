use super::*;

fn pipeline() -> MathPipeline {
    let (_, model) = fitted("(pipe X (robust_scale append=true) (ridge))");
    let mut names = (0..14)
        .map(|i| NumericColumn {
            name: format!("x{i}"),
            min: -2.0,
            max: 2.0,
            constant: false,
        })
        .collect::<Vec<_>>();
    names[0].name = "unicode\"\\\nλ\u{2028}".into();
    names[4].constant = true;
    names[4].max = names[4].min;
    let input = InputContract::new(
        names,
        vec![13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        TargetScale {
            name: "label".into(),
            min: 1e10 + 0.25,
            scale: 1e-5,
        },
    )
    .unwrap();
    MathPipeline::decode(
        &BTreeMap::from([
            (INPUT_PART.into(), serde_json::to_vec(&input).unwrap()),
            (
                PIPELINE_PART.into(),
                serde_json::to_vec(&PipelineState {
                    dsl: "(pipe X (robust_scale append=true) (ridge))".into(),
                    model,
                })
                .unwrap(),
            ),
        ]),
        PIPELINE_PART,
    )
    .unwrap()
}

#[test]
fn raw_cells_match_f64_engine_with_missing_tokens_extremes_and_named_columns() {
    let mut p = pipeline();
    let tokens = [
        None,
        Some(""),
        Some("NA"),
        Some("n/a"),
        Some("null"),
        Some("none"),
        Some("NaN"),
        Some("?"),
        Some("#N/A"),
        Some("true"),
        Some("yes"),
        Some("false"),
        Some("no"),
        Some("inf"),
        Some("-inf"),
        Some("1e999"),
        Some("unknown category"),
        Some("-0.0"),
        Some("1.0000000000000002"),
        Some("1.7976931348623157e308"),
        Some("-1.7976931348623157e308"),
        Some("\u{2003}0.12345678901234567\u{2003}"),
        Some("\u{feff}1"),
        Some("0x1"),
        Some("1_0"),
    ];
    let rows: Vec<_> = tokens.iter().map(|v| vec![*v; 14]).collect();
    let mut program = String::new();
    for i in 0..3 {
        if i == 1 {
            p.input.target.scale = 1.0;
            p.input.target.min = -0.0;
        }
        if i == 2 {
            p.input.target.scale = 1e-13;
            p.input.target.min = -0.0;
        }
        let source = emit_pipeline(&p).unwrap();
        let expected = p.predict_cells(&rows).unwrap();
        let rows_source = format!(
            "vec![{}]",
            tokens
                .iter()
                .map(|v| format!("vec![{v:?};14]"))
                .collect::<Vec<_>>()
                .join(",")
        );
        let expected_source = literal("", "Vec<f64>", &json!(expected)).unwrap();
        program.push_str(&format!("mod c{i} {{\n{source}\n}}\nfn check{i}() {{ let rows={rows_source};let expected:Vec<f64>={expected_source};let got=c{i}::predict_cells(&rows).unwrap();assert_eq!(got.len(),expected.len());for (a,b) in got.iter().zip(&expected) {{assert!((a-b).abs()<=1e-12+1e-10*b.abs());assert_eq!(a.to_bits(),b.to_bits());}} let names=&[{}];let mut reversed=names.to_vec();reversed.reverse();let reordered:Vec<_>=rows.iter().map(|r|r.iter().copied().rev().collect()).collect();assert_eq!(c{i}::predict_table(&reversed,&reordered).unwrap(),got);assert!(c{i}::predict_cells(&[vec![None;13]]).is_err());assert!(c{i}::predict_table(&[],&rows).is_err());assert!(c{i}::predict_cells(&[]).unwrap().is_empty()); }}\n",p.input.columns.iter().map(|c|format!("{:?}",c.name)).collect::<Vec<_>>().join(",")));
    }
    program.push_str("fn main(){check0();check1();check2();println!(\"75 raw rows: exact f64 bits, signed zero, selected order, constant target and named columns\");}");
    run_rust("raw-input", &program);
}

#[test]
fn invalid_pairings_legacy_input_and_nonfinite_constants_are_refused() {
    let (g, mut m) = fitted("(pipe X (step t=0.5) (ridge))");
    assert!(emit(&g, &m, 0).is_err());
    assert!(emit(&g, &m, 11).is_err());
    assert!(emit(
        &GenomeIr::parse("(pipe X (step t=0.25) (ridge))").unwrap(),
        &m,
        12
    )
    .is_err());
    assert!(emit(
        &GenomeIr::parse("(pipe X (sigmoid) (ridge))").unwrap(),
        &m,
        12
    )
    .is_err());
    assert!(emit(
        &GenomeIr::parse("(pipe X (step t=0.5 append=true) (ridge))").unwrap(),
        &m,
        12
    )
    .is_err());
    assert!(emit(&GenomeIr::parse("(pipe X (ridge))").unwrap(), &m, 12).is_err());
    assert!(emit(
        &GenomeIr::parse("(pipe X (step t=0.5) (elasticnet))").unwrap(),
        &m,
        12
    )
    .is_err());
    if let Model::Pipeline { head, .. } = &mut m {
        if let FittedHead::Ridge(r) = head.as_mut() {
            r.w[0] = f64::NAN;
        }
    }
    assert!(emit(&g, &m, 12).is_err());
    let (valid, mut malformed) = fitted("(pipe X (ridge))");
    if let Model::Pipeline { head, .. } = &mut malformed {
        if let FittedHead::Ridge(r) = head.as_mut() {
            r.w.pop();
        }
    }
    assert!(emit(&valid, &malformed, 12).is_err());
    let mut p = pipeline();
    p.input.prediction_version = 1;
    assert!(emit_pipeline(&p).is_err());
    p.input.prediction_version = 2;
    p.input.selected[0] = 999;
    assert!(emit_pipeline(&p).is_err());
}

#[test]
fn generated_prediction_refuses_nonfinite_output_instead_of_zero() {
    let (g, mut m) = fitted("(pipe X (ridge))");
    if let Model::Pipeline { head, .. } = &mut m {
        if let FittedHead::Ridge(r) = head.as_mut() {
            r.w.fill(f64::MAX);
            r.w[2] = -f64::MAX;
            r.scales.fill(1.0);
            r.means.fill(0.0);
        }
    }
    let source = emit(&g, &m, 12).unwrap();
    run_rust("nonfinite", &format!("{source}\nfn main() {{assert!(predict_numeric(&[vec![f64::MAX;12]]).is_err());println!(\"nonfinite output refused\");}}"));
}
#[test]
fn tree_decision_flip_rate_is_zero_for_golden_and_random_rows() {
    use feat_math::boost2::BoostNode;
    use feat_math::heads::GbmNode;
    let (mut rows, _) = training();
    rows.extend([vec![-0.0; 12], vec![1.5; 12], vec![-0.5; 12]]);
    let mut seed = 731u64;
    rows.extend((0..1000).map(|_| {
        (0..12)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 11) as f64 / ((1u64 << 53) as f64) * 2.0 - 0.5
            })
            .collect::<Vec<_>>()
    }));
    let mut program = String::new();
    let mut cases = 0;
    for dsl in [
        "(pipe X (gbm rounds=4 depth=2 min_leaf=2))",
        "(pipe X (histgbm2 rounds=4 leaves=4 min_leaf=2))",
    ] {
        let (genome, model) = fitted(dsl);
        let Model::Pipeline { head, .. } = &model else {
            unreachable!()
        };
        // Probe each real fitted tree separately, replacing only leaf values with their
        // node indices. This makes the actual engine/emitted routing decisions observable;
        // it is not a score/model-quality evaluation or a second routing implementation.
        let probes: Vec<FittedHead> = match head.as_ref() {
            FittedHead::Gbm(fitted) => fitted
                .trees
                .iter()
                .map(|tree| {
                    let mut fitted = fitted.clone();
                    let mut tree = tree.clone();
                    for (i, node) in tree.iter_mut().enumerate() {
                        if let GbmNode::Leaf(value) = node {
                            *value = i as f64;
                        }
                    }
                    fitted.init = 0.0;
                    fitted.lr = 1.0;
                    fitted.trees = vec![tree];
                    FittedHead::Gbm(fitted)
                })
                .collect(),
            FittedHead::HistGbm2(fitted) => fitted
                .bags
                .iter()
                .flatten()
                .map(|tree| {
                    let mut fitted = fitted.clone();
                    let mut tree = tree.clone();
                    for (i, node) in tree.iter_mut().enumerate() {
                        if let BoostNode::Leaf(value) = node {
                            *value = i as f64;
                        }
                    }
                    fitted.init = 0.0;
                    fitted.lr = 1.0;
                    fitted.bags = vec![vec![tree]];
                    FittedHead::HistGbm2(fitted)
                })
                .collect(),
            _ => unreachable!(),
        };
        assert!(!probes.is_empty());
        for head in probes {
            let model = Model::Pipeline {
                target: None,
                ops: Vec::new(),
                head: Box::new(head),
            };
            let source = emit(&genome, &model, 12).unwrap();
            let expected = model.predict(&rows).unwrap();
            let x = literal("", "Vec<Vec<f64>>", &json!(rows)).unwrap();
            let y = literal("", "Vec<f64>", &json!(expected)).unwrap();
            program.push_str(&format!("mod c{cases} {{\n{source}\n}}\nfn check{cases}() {{let rows={x};let expected:Vec<f64>={y};assert_eq!(c{cases}::predict_numeric(&rows).unwrap(),expected,\"leaf decision flip\");}}\n"));
            cases += 1;
        }
    }
    program.push_str("fn main(){");
    for i in 0..cases {
        program.push_str(&format!("check{i}();"));
    }
    program.push_str(&format!("println!(\"{cases} trees: 0 golden / 0 randomized row decision flips ({} row-tree decisions)\");}}",cases*rows.len()));
    run_rust("tree-decisions", &program);
}

#[test]
fn emitted_self_checks_replay_caller_rows_and_reject_perturbed_outputs() {
    let (train, targets) = training();
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let model = feat_math::model::fit(&genome, &train, &targets).unwrap();
    let inputs = train[0].clone();
    let expected = model.predict(&vec![inputs.clone()]).unwrap()[0];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fitted-self-check-proof");
    std::fs::create_dir_all(&root).unwrap();

    let js_checks = [crate::fitted_js::CheckRow {
        inputs: inputs.clone(),
        expected,
    }];
    let js = crate::fitted_js::emit_with_checks(&genome, &model, 12, &js_checks).unwrap();
    let js_path = root.join("predictor.mjs");
    std::fs::write(&js_path, js).unwrap();
    assert!(Command::new("node")
        .arg(&js_path)
        .arg("--self-check")
        .status()
        .unwrap()
        .success());
    let bad_js = crate::fitted_js::emit_with_checks(
        &genome,
        &model,
        12,
        &[crate::fitted_js::CheckRow {
            inputs: inputs.clone(),
            expected: expected + 10.0,
        }],
    )
    .unwrap();
    std::fs::write(&js_path, bad_js).unwrap();
    assert!(!Command::new("node")
        .arg(&js_path)
        .arg("--self-check")
        .status()
        .unwrap()
        .success());

    let py_checks = [crate::fitted_py::CheckRow {
        inputs: inputs.clone(),
        expected,
    }];
    let py = crate::fitted_py::emit_with_checks(&genome, &model, 12, &py_checks).unwrap();
    let py_path = root.join("predictor.py");
    std::fs::write(&py_path, py).unwrap();
    assert!(Command::new("python3")
        .arg(&py_path)
        .arg("--self-check")
        .status()
        .unwrap()
        .success());
    let bad_py = crate::fitted_py::emit_with_checks(
        &genome,
        &model,
        12,
        &[crate::fitted_py::CheckRow {
            inputs: inputs.clone(),
            expected: expected + 10.0,
        }],
    )
    .unwrap();
    std::fs::write(&py_path, bad_py).unwrap();
    assert!(!Command::new("python3")
        .arg(&py_path)
        .arg("--self-check")
        .status()
        .unwrap()
        .success());

    let rs_checks = [CheckRow {
        inputs: inputs.clone(),
        expected,
    }];
    let rs = emit_with_checks(&genome, &model, 12, &rs_checks).unwrap();
    let rs_path = root.join("predictor.rs");
    let rs_test = root.join("predictor-test");
    std::fs::write(&rs_path, rs).unwrap();
    assert!(Command::new("rustc")
        .args(["--edition=2021", "--test"])
        .arg(&rs_path)
        .arg("-o")
        .arg(&rs_test)
        .status()
        .unwrap()
        .success());
    assert!(Command::new(&rs_test).status().unwrap().success());
    let bad_rs = emit_with_checks(
        &genome,
        &model,
        12,
        &[CheckRow {
            inputs,
            expected: expected + 10.0,
        }],
    )
    .unwrap();
    std::fs::write(&rs_path, bad_rs).unwrap();
    assert!(Command::new("rustc")
        .args(["--edition=2021", "--test"])
        .arg(&rs_path)
        .arg("-o")
        .arg(&rs_test)
        .status()
        .unwrap()
        .success());
    assert!(!Command::new(&rs_test).status().unwrap().success());
}
