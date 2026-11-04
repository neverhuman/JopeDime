#[test]
fn bounded_config_rejects_versions_axes_parameters_and_legacy_ingress() {
    for (v, id, axis, w, p, r, b) in [
        (1, 58, Axis::FeatureWithinRow, 5, 2, 1, "shrink"),
        (2, 59, Axis::FeatureWithinRow, 5, 2, 1, "shrink"),
        (2, 58, Axis::SampleTime, 5, 2, 1, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 2, 1, 0, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 4, 1, 0, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 131, 2, 0, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 3, 3, 0, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 9, 7, 0, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 5, 2, 3, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 5, 2, 0, "mirror"),
    ] {
        assert!(Config::new(v, id, axis, w, p, r, b).is_err());
    }
    assert!(config(5, 2, 0).fit(feat_math::MAX_WIDTH + 1).is_err());
    let old = feat_math::GenomeIr::parse("(pipe X (savgol w=5 p=2 r=0) (ridge lambda_frac=0.02))");
    if let Ok(old) = old {
        assert!(feat_math::fit(&old, &vec![vec![1.0; 5]; 4], &[1.0; 4]).is_err());
        assert!(feat_math::transpile::emit(&old, &Default::default()).is_err());
    }
    assert!(Program::new(1, 5, vec![], 0.02).is_err());
    assert!(Program::new(2, 0, vec![], 0.02).is_err());
    assert!(Program::new(2, 5, vec![RowStage::Savgol(config(5, 2, 0)); 9], 0.02).is_err());
}

#[test]
fn polynomial_and_derivative_reproduction_covers_clipped_and_maximum_windows() {
    for w in [3, 5, 9, 17, 129] {
        for p in 0..=6.min(w - 1) {
            for r in 0..=p {
                let n = usize::from(w) + 7;
                let row = (0..n)
                    .map(|j| (j as f64 / n as f64).powi(i32::from(p)))
                    .collect::<Vec<_>>();
                let result = config(w, p, r).fit(n).unwrap().apply(&row).unwrap();
                for (j, &actual) in result.values.iter().enumerate() {
                    let left = j.min(usize::from(w / 2));
                    let right = (n - 1 - j).min(usize::from(w / 2));
                    // Degree is preserved when the clipped window has p+1 samples.
                    if left + right < usize::from(p) {
                        continue;
                    }
                    let factorial = (0..r).fold(1.0, |a, k| a * f64::from(p - k));
                    let expected = factorial * (j as f64 / n as f64).powi(i32::from(p - r))
                        / (n as f64).powi(i32::from(r));
                    assert!(
                        (actual - expected).abs() < 2e-9,
                        "w={w} p={p} r={r} j={j} actual={actual} expected={expected}"
                    );
                    assert_eq!(result.faults[j], 0);
                }
            }
        }
    }
    // Independent analytic five-tap least-squares smoothing weights.
    let state = config(5, 2, 0).fit(5).unwrap();
    for (i, numerator) in [-3.0, 12.0, 17.0, 12.0, -3.0].iter().enumerate() {
        let mut impulse = vec![0.0; 5];
        impulse[i] = 1.0;
        let actual = state.apply(&impulse).unwrap().values[2];
        assert!((actual - numerator / 35.0).abs() < 4e-16);
    }
    // A nonzero sixth derivative prevents a loose near-zero check from
    // admitting a zero implementation at high derivative orders.
    let sixth = config(129, 6, 6)
        .fit(13)
        .unwrap()
        .apply(&(0..13).map(|i| (i as f64).powi(6)).collect::<Vec<_>>())
        .unwrap();
    assert!(
        sixth.values.iter().all(|v| (v - 720.0).abs() < 1e-5),
        "{:?}",
        sixth.values
    );
}

#[test]
fn shrink_short_empty_rows_and_faults_have_defined_bits() {
    for width in [0, 1, 2, 3, 4, 129, feat_math::MAX_WIDTH] {
        let row = vec![2.0; width];
        for r in 0..=6 {
            let out = config(129, 6, r).fit(width).unwrap().apply(&row).unwrap();
            assert_eq!(out.values.len(), width);
            assert_eq!(out.faults.len(), width);
            if usize::from(r) >= width {
                assert!(out.values.iter().all(|v| v.to_bits() == 0));
                assert!(out.faults.iter().all(|&f| f == BOUNDARY_DEGREE));
            }
        }
    }
    let row = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        -f64::MAX,
        0.0,
        -0.0,
    ];
    let output = config(3, 0, 0).fit(row.len()).unwrap().apply(&row).unwrap();
    assert!(output
        .values
        .iter()
        .all(|x| x.is_finite() && x.abs() <= 1e12));
    assert!(output.faults[0] & FaultSet::NONFINITE_INPUT.bits() != 0);
    assert!(output.faults[3] & FaultSet::MAGNITUDE_CLAMPED.bits() != 0);
    assert!(config(5, 2, 0).fit(5).unwrap().apply(&[1.0]).is_err());
    assert_eq!(FaultSet::NONFINITE_INPUT.bits(), 8192);
    assert_eq!(FaultSet::MAGNITUDE_CLAMPED.bits(), 512);
    assert_eq!(FaultSet::NONFINITE_SANITIZED.bits(), 256);
}

#[test]
fn coefficient_model_and_program_state_roundtrip_without_losing_bits() {
    let p = program(7, config(5, 2, 1));
    let bytes = p.to_bytes().unwrap();
    let loaded = Program::from_bytes(&bytes).unwrap();
    assert_eq!(loaded, p);
    let ingress = VersionedProgram::from_bytes(&bytes).unwrap();
    assert!(matches!(ingress, VersionedProgram::RowVectorV2(_)));
    assert_eq!(ingress.to_bytes().unwrap(), bytes);
    assert_eq!(
        VersionedProgram::parse_lossless_text(&p.canonical().unwrap())
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    assert!(feat_math::kernel::Program::from_bytes(&bytes).is_err());
    for index in [7, 8, 9, 10] {
        let mut bad = bytes.clone();
        bad[index] = 255;
        assert!(Program::from_bytes(&bad).is_err());
    }
    let (x, y) = training(7);
    let model = p.fit(&x, &y).unwrap();
    let json = model.to_json().unwrap();
    let restored = Model::from_json(&json).unwrap();
    assert_eq!(restored.to_json().unwrap(), json);
    assert_eq!(model.identity().unwrap(), restored.identity().unwrap());
    assert_eq!(model.predict(&x).unwrap(), restored.predict(&x).unwrap());
    assert!(matches!(
        VersionedModel::from_json(&json).unwrap(),
        VersionedModel::RowVectorV2(_)
    ));
    assert!(serde_json::from_str::<feat_math::Model>(&json).is_err());
    let mut wire: serde_json::Value = serde_json::from_str(&json).unwrap();
    wire["stages"][0]["state"]["stencils"][0]["bits"][0] = serde_json::json!(0);
    assert!(Model::from_json(&wire.to_string()).is_err());
    assert!(model.predict(&vec![vec![]]).is_err());
    assert!(model.predict(&vec![vec![1.0; 6]]).is_err());
    assert_eq!(model.predict(&vec![]).unwrap(), (vec![], vec![]));
    // Holding out any rows/labels cannot enter this API's training closure.
    let before = model.to_json().unwrap();
    let _ = model.predict(&vec![vec![f64::NAN; 7]; 3]).unwrap();
    assert_eq!(before, model.to_json().unwrap());
    assert_eq!(before, p.fit(&x, &y).unwrap().to_json().unwrap());
}

#[test]
fn model_ingress_rejects_finite_weights_that_can_overflow_prediction() {
    let p = Program::new(2, 2, vec![], 0.02).unwrap();
    let model = p
        .fit(&vec![vec![0.0, 0.0], vec![1.0, -1.0]], &[0.0, 1.0])
        .unwrap();
    let text = model.to_json().unwrap();
    assert!(Model::from_json(&text).is_ok());
    let mut wire: serde_json::Value = serde_json::from_str(&text).unwrap();
    wire["ridge_bits"] = serde_json::json!([
        [0, 0],
        [1.0f64.to_bits(), 1.0f64.to_bits()],
        [0, f64::MAX.to_bits(), f64::MAX.to_bits()]
    ]);
    let forged = wire.to_string();
    assert!(Model::from_json(&forged).is_err());
    assert!(VersionedModel::from_json(&forged).is_err());
}

#[test]
fn unknown_row_stage_wire_is_an_unsupported_version_at_both_ingresses() {
    let mut bytes = program(2, config(3, 1, 0)).to_bytes().unwrap();
    let at = bytes
        .windows(11)
        .position(|w| w == b"\"wire\":\"58\"")
        .unwrap();
    bytes[at + 8] = b'9';
    bytes[at + 9] = b'9';
    assert!(matches!(
        Program::from_bytes(&bytes),
        Err(feat_math::kernel::wire_v2::VersionError::UnsupportedVersion(_))
    ));
    assert!(matches!(
        VersionedProgram::from_bytes(&bytes),
        Err(feat_math::kernel::wire_v2::VersionError::UnsupportedVersion(_))
    ));
    let p = program(2, config(3, 1, 0));
    let (x, y) = training(2);
    let text = p.fit(&x, &y).unwrap().to_json().unwrap();
    for unknown_program in [false, true] {
        let mut wire: serde_json::Value = serde_json::from_str(&text).unwrap();
        wire["stages"][0]["wire"] = serde_json::json!("99");
        if unknown_program {
            wire["program"] = serde_json::json!(bytes);
        }
        let future_model = wire.to_string();
        assert!(matches!(
            Model::from_json(&future_model),
            Err(feat_math::kernel::wire_v2::VersionError::UnsupportedVersion(_))
        ));
        assert!(matches!(
            VersionedModel::from_json(&future_model),
            Err(feat_math::kernel::wire_v2::VersionError::UnsupportedVersion(_))
        ));
    }
}
