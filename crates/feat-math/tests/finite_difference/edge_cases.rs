use super::*;

#[test]
fn opaque_shapes_stay_parseable_but_never_gain_recipe_support_or_fallback() {
    use feat_math::kernel::export::Route;
    for text in [
        "(pipe X (standardize) (ridge lambda_frac=0.01))",
        "(pipe X (elasticnet lambda_frac=0.01 l1_ratio=0.5))",
    ] {
        let p = DifferenceProgram::new(
            DifferenceConfig::new(2, 59, 1, 1.).unwrap(),
            1,
            OrderedInputIdentity::new("g", "t"),
            GenomeIr::parse(text).unwrap(),
        )
        .unwrap();
        let routed = VersionedProgram::from_bytes(&p.to_bytes().unwrap()).unwrap();
        assert_eq!(routed.to_bytes().unwrap(), p.to_bytes().unwrap());
        let report = export_report(&routed);
        assert!(matches!(report.rust_recipe, Route::Unsupported(_)));
        assert!(matches!(report.python_recipe, Route::Unsupported(_)));
        assert!(matches!(report.cpp17_recipe, Route::Unsupported(_)));
        assert!(matches!(report.cpp17_fitted, Route::Unsupported(_)));
        for lang in [
            DifferenceLanguage::Rust,
            DifferenceLanguage::Python,
            DifferenceLanguage::Cpp17,
        ] {
            assert!(routed.emit_ordered_recipe(lang).is_err());
        }
    }
    // Parsing remains available even when fixed metadata alone cannot fit a
    // native model. Export admission must refuse all routes before emission.
    let large = DifferenceProgram::new(
        DifferenceConfig::new(2, 59, 1, 1.).unwrap(),
        1,
        OrderedInputIdentity::new("g".repeat(3_000_000), "t"),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
    )
    .unwrap();
    let bytes = large.to_bytes().unwrap();
    let restored = VersionedProgram::from_bytes(&bytes).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), bytes);
    assert!(large.validate_recipe_closure().is_err());
    let report = export_report(&restored);
    assert!(matches!(report.rust_recipe, Route::Unsupported(_)));
    assert!(matches!(report.python_recipe, Route::Unsupported(_)));
    assert!(matches!(report.cpp17_recipe, Route::Unsupported(_)));
    assert!(matches!(report.cpp17_fitted, Route::Unsupported(_)));
    for language in [
        DifferenceLanguage::Rust,
        DifferenceLanguage::Python,
        DifferenceLanguage::Cpp17,
    ] {
        assert!(restored.emit_ordered_recipe(language).is_err());
    }
    let legacy = VersionedProgram::LegacyV1(Program::LegacyV1(
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
    ));
    assert!(legacy
        .emit_ordered_recipe(DifferenceLanguage::Rust)
        .is_err());
}

#[test]
fn fitted_codec_binds_source_profile_program_and_native_continuation() {
    let rows = rows();
    let model = program(2)
        .fit_training(
            &rows[..4],
            &[0.25, 0.5, 1., 2.],
            TaskMode::Forecasting,
            FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
        )
        .unwrap();
    let artifact = model.emit_fitted_cpp17().unwrap();
    let replay = DifferenceFittedCpp::from_codec_json(&artifact.codec).unwrap();
    assert_eq!(replay.source, artifact.source);
    assert_eq!(replay.codec, artifact.codec);
    let value: serde_json::Value = serde_json::from_str(&artifact.codec).unwrap();
    for (field, replacement) in [
        ("schema", serde_json::json!("unknown")),
        ("format_version", serde_json::json!(3)),
        ("min_reader_version", serde_json::json!(3)),
        ("required_features", serde_json::json!(64 | 128)),
        ("profile", serde_json::json!("unknown")),
    ] {
        let mut unsupported = value.clone();
        unsupported[field] = replacement;
        assert!(
            matches!(
                DifferenceFittedCpp::from_codec_json(&unsupported.to_string()),
                Err(VersionError::UnsupportedVersion(_))
            ),
            "unsupported {field} must refuse before checkpoint admission"
        );
    }
    for (field, replacement) in [
        ("schema", serde_json::json!("unknown")),
        ("min_reader_version", serde_json::json!(3)),
        ("required_features", serde_json::json!(0)),
        ("profile", serde_json::json!("unknown")),
        ("source_sha256", serde_json::json!("0".repeat(64))),
        ("program_identity", serde_json::json!("0".repeat(64))),
        ("model_json", serde_json::json!("{}")),
        ("unknown", serde_json::json!(true)),
    ] {
        let mut forged = value.clone();
        forged[field] = replacement;
        assert!(
            DifferenceFittedCpp::from_codec_json(&forged.to_string()).is_err(),
            "{field}"
        );
    }
    assert!(DifferenceFittedCpp::from_codec_json("{}").is_err());
    let oversized = format!("{}{}", artifact.codec, " ".repeat(16 * 1024 * 1024));
    assert!(
        matches!(DifferenceFittedCpp::from_codec_json(&oversized),Err(VersionError::Malformed(message)) if message.starts_with("Resource("))
    );
}
