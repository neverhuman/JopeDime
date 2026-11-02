use super::*;

#[test]
fn calibrated_emitters_match_engine_and_binding_floor() {
    let (train, targets) = training();
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let model = feat_math::model::fit(&genome, &train, &targets).unwrap();
    let raw = model.predict(&train).unwrap();
    let affine: Vec<_> = raw.iter().map(|v| 0.7 + 1.2 * v).collect();
    let (low_index, low_value) = affine
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let (high_index, high_value) = affine
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let lower = (low_value + high_value) / 2.0;
    let transform = OutputTransform {
        intercept: 0.7,
        slope: 1.2,
        lower: Some(lower),
    };
    let input_rows = [train[low_index].clone(), train[high_index].clone()];
    let checks_js: Vec<_> = input_rows
        .iter()
        .zip([low_index, high_index])
        .map(|(inputs, index)| crate::fitted_js::CheckRow {
            inputs: inputs.clone(),
            expected: transform.apply(raw[index]),
        })
        .collect();
    let checks_py: Vec<_> = input_rows
        .iter()
        .zip([low_index, high_index])
        .map(|(inputs, index)| crate::fitted_py::CheckRow {
            inputs: inputs.clone(),
            expected: transform.apply(raw[index]),
        })
        .collect();
    let checks_rs: Vec<_> = input_rows
        .iter()
        .zip([low_index, high_index])
        .map(|(inputs, index)| CheckRow {
            inputs: inputs.clone(),
            expected: transform.apply(raw[index]),
        })
        .collect();
    assert!(transform.apply(raw[low_index]) == lower);
    assert!(transform.apply(raw[high_index]) > lower);

    assert_eq!(
        crate::fitted_js::emit(&genome, &model, 12).unwrap(),
        crate::fitted_js::emit_calibrated(&genome, &model, 12, Some(&OutputTransform::IDENTITY))
            .unwrap()
    );
    assert_eq!(
        crate::fitted_py::emit(&genome, &model, 12).unwrap(),
        crate::fitted_py::emit_calibrated(&genome, &model, 12, Some(&OutputTransform::IDENTITY))
            .unwrap()
    );
    assert_eq!(
        emit(&genome, &model, 12).unwrap(),
        emit_calibrated(&genome, &model, 12, Some(&OutputTransform::IDENTITY)).unwrap()
    );

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fitted-calibration-proof");
    std::fs::create_dir_all(&root).unwrap();
    let js_path = root.join("predictor.mjs");
    std::fs::write(
        &js_path,
        crate::fitted_js::emit_calibrated_with_checks(
            &genome,
            &model,
            12,
            Some(&transform),
            &checks_js,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(Command::new("node")
        .arg(&js_path)
        .arg("--self-check")
        .status()
        .unwrap()
        .success());
    let py_path = root.join("predictor.py");
    std::fs::write(
        &py_path,
        crate::fitted_py::emit_calibrated_with_checks(
            &genome,
            &model,
            12,
            Some(&transform),
            &checks_py,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(Command::new("python3")
        .arg(&py_path)
        .arg("--self-check")
        .status()
        .unwrap()
        .success());
    let rs_path = root.join("predictor.rs");
    let rs_test = root.join("predictor-test");
    std::fs::write(
        &rs_path,
        emit_calibrated_with_checks(&genome, &model, 12, Some(&transform), &checks_rs).unwrap(),
    )
    .unwrap();
    assert!(Command::new("rustc")
        .args(["--edition=2021", "--test"])
        .arg(&rs_path)
        .arg("-o")
        .arg(&rs_test)
        .status()
        .unwrap()
        .success());
    assert!(Command::new(&rs_test).status().unwrap().success());
}

#[test]
fn integral_calibration_constants_compile_and_match_engine() {
    let (train, targets) = training();
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let model = feat_math::model::fit(&genome, &train, &targets).unwrap();
    let raw = model.predict(&train).unwrap();
    let mut sorted = raw.clone();
    sorted.sort_by(f64::total_cmp);
    // An integral intercept that puts the zero floor inside the prediction range.
    let centre = -(4.0 * sorted[sorted.len() / 2]).round();
    let transforms = [
        // calibration::floor_only for a log-error objective: identity plus a zero floor.
        OutputTransform {
            intercept: 0.0,
            slope: 1.0,
            lower: Some(0.0),
        },
        // The shift family keeps a unit slope.
        OutputTransform {
            intercept: -1.0,
            slope: 1.0,
            lower: Some(-2.0),
        },
        OutputTransform {
            intercept: centre,
            slope: 4.0,
            lower: Some(0.0),
        },
        OutputTransform {
            intercept: 3.0,
            slope: 1.0,
            lower: None,
        },
        OutputTransform {
            intercept: -0.0,
            slope: 2.0,
            lower: Some(-0.0),
        },
    ];
    let binding = raw.iter().filter(|p| centre + 4.0 * **p < 0.0).count();
    assert!(
        binding > 0 && binding < raw.len(),
        "zero floor must bind on some rows"
    );
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fitted-calibration-integral-proof");
    std::fs::create_dir_all(&root).unwrap();
    for (index, transform) in transforms.iter().enumerate() {
        let checks: Vec<_> = train
            .iter()
            .zip(&raw)
            .map(|(inputs, prediction)| CheckRow {
                inputs: inputs.clone(),
                expected: transform.apply(*prediction),
            })
            .collect();
        let source =
            emit_calibrated_with_checks(&genome, &model, 12, Some(transform), &checks).unwrap();
        let rs_path = root.join(format!("predictor-{index}.rs"));
        let rs_test = root.join(format!("predictor-{index}-test"));
        std::fs::write(&rs_path, &source).unwrap();
        assert!(
            Command::new("rustc")
                .args(["--edition=2021", "--test"])
                .arg(&rs_path)
                .arg("-o")
                .arg(&rs_test)
                .status()
                .unwrap()
                .success(),
            "{transform:?} does not compile"
        );
        assert!(
            Command::new(&rs_test).status().unwrap().success(),
            "{transform:?} self-check failed"
        );
        if index == 0 {
            assert!(source.contains("(0.0 + 1.0 * pred).max(0.0)"));
        }
    }
}

#[test]
fn platt_emitters_match_engine_probabilities_in_node_python_and_rustc() {
    let (genome, model) = fitted("(pipe X (ridge))");
    let mut calibration = crate::model_object::calibration_platt::cross_fit(
        &[0.1, 0.2, 0.8, 0.9],
        &["cold", "cold", "hot", "hot"],
        ["cold", "hot"],
        &[0, 1, 0, 1],
    )
    .unwrap()
    .calibration;
    calibration.sigmoid.scale = 0.7;
    calibration.sigmoid.center = 0.1;
    let scores = [0.0, 0.2, 0.75, 1.0];
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fitted-platt-proof");
    std::fs::create_dir_all(&dir).unwrap();
    let no_calibration = ExportCalibration::default();
    assert_eq!(
        crate::fitted_js::emit_with_platt(&genome, &model, 12, &no_calibration).unwrap(),
        crate::fitted_js::emit(&genome, &model, 12).unwrap()
    );
    assert_eq!(
        crate::fitted_py::emit_with_platt(&genome, &model, 12, &no_calibration).unwrap(),
        crate::fitted_py::emit(&genome, &model, 12).unwrap()
    );
    assert_eq!(
        crate::fitted_rs::emit_with_platt(&genome, &model, 12, &no_calibration).unwrap(),
        crate::fitted_rs::emit(&genome, &model, 12).unwrap()
    );

    for (intercept, slope) in [(-0.35, 2.4), (1000.0, 2.4), (-1000.0, 0.0)] {
        calibration.sigmoid.intercept = intercept;
        calibration.sigmoid.slope = slope;
        let expected = scores
            .iter()
            .map(|score| calibration.apply(*score).unwrap())
            .collect::<Vec<_>>();
        let export = ExportCalibration {
            output: None,
            platt: Some(calibration.clone()),
        };

        let mut js = crate::fitted_js::emit_with_platt(&genome, &model, 12, &export).unwrap();
        assert!(js.contains("const PLATT_CLASS_ORDER=Object.freeze([\"cold\",\"hot\"])"));
        js.push_str(
            "\nfor(const s of [0,0.2,0.75,1]) console.log(JSON.stringify(fittedPlatt(s)));\n",
        );
        let js_path = dir.join("predictor.js");
        std::fs::write(&js_path, js).unwrap();
        let js_output = Command::new("node").arg(&js_path).output().unwrap();
        assert!(
            js_output.status.success(),
            "{}",
            String::from_utf8_lossy(&js_output.stderr)
        );
        assert_probability_pairs(&String::from_utf8_lossy(&js_output.stdout), &expected);

        let py = crate::fitted_py::emit_with_platt(&genome, &model, 12, &export).unwrap();
        assert!(py.contains("PLATT_CLASS_ORDER = [\"cold\",\"hot\"]"));
        let py_path = dir.join("predictor.py");
        std::fs::write(&py_path, py).unwrap();
        let py_output = Command::new("python3")
                .args([
                    "-c",
                    "import runpy,sys; m=runpy.run_path(sys.argv[1],run_name='platt_test'); [print(repr(m['fitted_platt'](s))) for s in [0,0.2,0.75,1]]",
                ])
                .arg(&py_path)
                .output()
                .unwrap();
        assert!(
            py_output.status.success(),
            "{}",
            String::from_utf8_lossy(&py_output.stderr)
        );
        assert_probability_pairs(&String::from_utf8_lossy(&py_output.stdout), &expected);

        let mut rs = crate::fitted_rs::emit_with_platt(&genome, &model, 12, &export).unwrap();
        assert!(rs.contains("pub const PLATT_CLASS_ORDER: [&str; 2] = [\"cold\", \"hot\"]"));
        rs.push_str("\nfn main() { for s in [0.0,0.2,0.75,1.0] { let p=fitted_platt(s).unwrap(); println!(\"{:?}\",p); } }\n");
        let rs_path = dir.join("predictor.rs");
        let rs_binary = dir.join("predictor");
        std::fs::write(&rs_path, rs).unwrap();
        let compile = Command::new("rustc")
            .args(["--edition=2021", "-D", "warnings"])
            .arg(&rs_path)
            .arg("-o")
            .arg(&rs_binary)
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "{}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let rs_output = Command::new(&rs_binary).output().unwrap();
        assert!(rs_output.status.success());
        assert_probability_pairs(&String::from_utf8_lossy(&rs_output.stdout), &expected);
    }
}

fn assert_probability_pairs(output: &str, expected_positive: &[f64]) {
    let rows = output.lines().collect::<Vec<_>>();
    assert_eq!(rows.len(), expected_positive.len(), "{output}");
    for (line, expected) in rows.iter().zip(expected_positive) {
        let values: Vec<f64> = line
            .trim_matches(|ch| ch == '[' || ch == ']' || ch == '(' || ch == ')')
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .filter(|part| !part.is_empty())
            .map(|part| part.parse().unwrap())
            .collect();
        assert_eq!(values.len(), 2, "{line}");
        assert!((values[0] - (1.0 - expected)).abs() < 1e-14, "{line}");
        assert!((values[1] - expected).abs() < 1e-14, "{line}");
    }
}
