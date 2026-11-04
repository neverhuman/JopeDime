use super::*;

#[test]
fn bank_contract_refuses_versions_axes_banks_and_legacy_ingress() {
    for (version, wire, axis, frequencies) in [
        (1, 61, Axis::FeatureWithinRow, vec![0.1]),
        (2, 58, Axis::FeatureWithinRow, vec![0.1]),
        (2, 61, Axis::SampleTime, vec![0.1]),
        (2, 61, Axis::FeatureWithinRow, vec![]),
        (2, 61, Axis::FeatureWithinRow, vec![0.1; 33]),
        (2, 61, Axis::FeatureWithinRow, vec![0.2, f64::NAN]),
        (2, 61, Axis::FeatureWithinRow, vec![-0.25]),
        (2, 61, Axis::FeatureWithinRow, vec![0.5000000000000001]),
    ] {
        assert!(Config::new(version, wire, axis, &frequencies).is_err());
    }
    assert!(bank(&[0.1]).fit(feat_math::MAX_WIDTH + 1).is_err());
    let old = feat_math::GenomeIr::parse("(pipe X (goertzel_bank f=0.1) (ridge lambda_frac=0.02))");
    if let Ok(old) = old {
        assert!(feat_math::fit(&old, &vec![vec![1.0; 5]; 4], &[1.0; 4]).is_err());
        assert!(feat_math::transpile::emit(&old, &Default::default()).is_err());
    }
    // The bank sets the stage width; a following stage sees the bank length.
    let p = program(
        9,
        vec![
            RowStage::Goertzel(bank(&[0.0, 0.1, 0.25])),
            RowStage::Savgol(smoothing()),
        ],
    );
    let model = p.fit(&training(9).0, &training(9).1).unwrap();
    let widths: Vec<(usize, usize)> = model
        .fitted_stages()
        .iter()
        .map(|s| (s.input_width(), s.output_width()))
        .collect();
    assert_eq!(widths, [(9, 3), (3, 3)]);
}

#[test]
fn feature_declarations_bind_exactly_to_stage_content() {
    let savgol_only = program(7, vec![RowStage::Savgol(smoothing())]);
    let with_bank = program(
        7,
        vec![
            RowStage::Savgol(smoothing()),
            RowStage::Goertzel(bank(&[0.0, 0.2])),
        ],
    );
    assert_eq!(savgol_only.required_features(), REQUIRED_FEATURES);
    assert_eq!(
        with_bank.required_features(),
        REQUIRED_FEATURES | GOERTZEL_FEATURE
    );
    for (p, declared) in [(&savgol_only, 32_u64), (&with_bank, 160)] {
        let bytes = p.to_bytes().unwrap();
        let at = bytes.len()
            - bytes
                .windows(8)
                .rev()
                .position(|w| w == declared.to_le_bytes())
                .map(|i| i + 8)
                .unwrap();
        assert_eq!(Program::from_bytes(&bytes).unwrap(), *p);
        // Swapping the declaration for the other legal value is refused, as is any other value.
        for forged in [32_u64 + 160 - declared, 0, 64, 32 | 64] {
            let mut bad = bytes.clone();
            bad[at..at + 8].copy_from_slice(&forged.to_le_bytes());
            assert!(matches!(
                Program::from_bytes(&bad),
                Err(VersionError::UnsupportedVersion(_))
            ));
        }
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
    }
    // Model envelopes carry the program's feature set and the bank's own identities.
    let (x, y) = training(7);
    let model = with_bank.fit(&x, &y).unwrap();
    let json = model.to_json().unwrap();
    let wire: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(wire["required_features"], 160);
    assert_eq!(
        wire["emitter_feature_identity"],
        "feat-math.feature-row.recipes.v2+wire58+wire61"
    );
    let restored = Model::from_json(&json).unwrap();
    assert_eq!(restored.to_json().unwrap(), json);
    assert_eq!(model.predict(&x).unwrap(), restored.predict(&x).unwrap());
    assert!(matches!(
        VersionedModel::from_json(&json).unwrap(),
        VersionedModel::RowVectorV2(_)
    ));
    let savgol_json = savgol_only.fit(&x, &y).unwrap().to_json().unwrap();
    let savgol_wire: serde_json::Value = serde_json::from_str(&savgol_json).unwrap();
    for (field, bank_text, savgol_text) in [
        ("required_features", &wire, &savgol_wire),
        ("grammar_sha256", &wire, &savgol_wire),
        ("emitter_feature_identity", &wire, &savgol_wire),
    ] {
        // A bank model cannot claim the savgol-only identities, and vice versa.
        let mut forged = bank_text.clone();
        forged[field] = savgol_text[field].clone();
        assert!(Model::from_json(&forged.to_string()).is_err(), "{field}");
        let mut forged = savgol_text.clone();
        forged[field] = bank_text[field].clone();
        assert!(Model::from_json(&forged.to_string()).is_err(), "{field}");
    }
    // Tampered bank constants and forged frequencies refuse at model ingress.
    let mut forged = wire.clone();
    forged["stages"][1]["state"]["bins"][0]["sine_bits"] = serde_json::json!(1);
    assert!(Model::from_json(&forged.to_string()).is_err());
    let mut forged = wire.clone();
    forged["stages"][1]["state"]["config"]["frequency_bits"][0] =
        serde_json::json!((-0.0_f64).to_bits());
    assert!(Model::from_json(&forged.to_string()).is_err());
    // Holding out rows cannot change the fitted model.
    let before = model.to_json().unwrap();
    let _ = model.predict(&vec![vec![f64::NAN; 7]; 3]).unwrap();
    assert_eq!(before, model.to_json().unwrap());
    assert_eq!(before, with_bank.fit(&x, &y).unwrap().to_json().unwrap());
}
