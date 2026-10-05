use feat_math::kernel::cumulative_integration::{
    Config, Edit, ExperimentalProfile, IntegrationLanguage, IntegrationProgram, IntegrationSearch,
    Mode, OrderedRow,
};
use feat_math::kernel::export::Route;
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::wire_v2::{export_report, VersionError, VersionedModel, VersionedProgram};
use feat_math::GenomeIr;
fn child() -> GenomeIr {
    GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap()
}
fn program(mode: Mode, input: OrderedInputIdentity) -> IntegrationProgram {
    IntegrationProgram::new(
        Config::new(2, 60, mode, vec![-0.0, 3.0]).unwrap(),
        input,
        child(),
    )
    .unwrap()
}
fn rows() -> Vec<OrderedRow> {
    (0..8)
        .map(|n| OrderedRow {
            group: GroupKey::categorical(if n % 2 == 0 { "A" } else { "B" }).unwrap(),
            timestamp: n as f64 * 0.5,
            values: vec![n as f64, (n * n) as f64],
        })
        .collect()
}
#[test]
fn native_router_and_typed_artifact_admission() {
    for mode in [Mode::Rect, Mode::Trap] {
        let p = program(
            mode,
            OrderedInputIdentity::new("generation\t\u{2028}", "time"),
        );
        let node = VersionedProgram::IntegrationV2(p.clone());
        let bytes = node.to_bytes().unwrap();
        assert_eq!(
            VersionedProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let text = node.to_lossless_text().unwrap();
        assert_eq!(
            VersionedProgram::parse_lossless_text(&text)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        assert!(node.legacy_genome().is_err());
        assert!(node.emit_ordered_recipe(IntegrationLanguage::Rust).is_err());
        let report = export_report(&node);
        assert!(matches!(report.rust_recipe, Route::Ok));
        assert!(matches!(report.python_recipe, Route::Ok));
        assert!(matches!(report.cpp17_recipe, Route::Ok));
        assert!(matches!(report.cpp17_fitted, Route::Ok));
        assert!(matches!(report.javascript, Route::Unsupported(_)));
        for lang in [
            IntegrationLanguage::Rust,
            IntegrationLanguage::Python,
            IntegrationLanguage::Cpp17,
        ] {
            assert!(!node.emit_integral_recipe(lang).unwrap().source.is_empty());
        }
        let rows = rows();
        let labels = [0.25, 0.5, 1.0, 2.0];
        assert!(node
            .fit_ordered(
                &rows[..4],
                &labels,
                TaskMode::IID,
                FoldBoundary::new("cut", 4, 8).unwrap()
            )
            .is_err());
        let mut routed = node
            .fit_ordered(
                &rows[..4],
                &labels,
                TaskMode::Forecasting,
                FoldBoundary::new("cut", 4, 8).unwrap(),
            )
            .unwrap();
        let mut native = p
            .fit_training(
                &rows[..4],
                &labels,
                TaskMode::Forecasting,
                FoldBoundary::new("cut", 4, 8).unwrap(),
            )
            .unwrap();
        let VersionedModel::IntegrationV2(model) = &routed else {
            panic!()
        };
        let json = model.to_json().unwrap();
        let mut restored = VersionedModel::from_json(&json).unwrap();
        assert!(restored.legacy_model().is_err());
        assert!(restored.emit_ordered_fitted_cpp17().is_err());
        let fitted = restored.emit_integral_fitted_cpp17().unwrap();
        assert_eq!(
            feat_math::kernel::cumulative_integration::IntegrationFittedCpp::from_codec_json(
                &fitted.codec
            )
            .unwrap()
            .source,
            fitted.source
        );
        let expected = native.predict(&rows[4..]).unwrap();
        assert_eq!(routed.predict_ordered(&rows[4..]).unwrap(), expected);
        assert_eq!(restored.predict_ordered(&rows[4..]).unwrap(), expected);
        let mut future: serde_json::Value = serde_json::from_str(&json).unwrap();
        future["format_version"] = 3.into();
        assert!(matches!(
            VersionedModel::from_json(&future.to_string()),
            Err(VersionError::UnsupportedVersion(_))
        ));
        for n in 0..bytes.len() {
            assert!(VersionedProgram::from_bytes(&bytes[..n]).is_err());
        }
        let mut wrong = bytes.clone();
        wrong[8] = 6;
        assert!(VersionedProgram::from_bytes(&wrong).is_err());
        let mut wrong = bytes.clone();
        wrong[9] = 3;
        assert!(VersionedProgram::from_bytes(&wrong).is_err());
    }
    let huge = VersionedProgram::IntegrationV2(program(
        Mode::Rect,
        OrderedInputIdentity::new("g".repeat(3_000_000), "t"),
    ));
    let bytes = huge.to_bytes().unwrap();
    assert_eq!(
        VersionedProgram::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    let report = export_report(&huge);
    assert!(matches!(report.rust_recipe, Route::Unsupported(_)));
    for lang in [
        IntegrationLanguage::Rust,
        IntegrationLanguage::Python,
        IntegrationLanguage::Cpp17,
    ] {
        assert!(huge.emit_integral_recipe(lang).is_err());
    }
}
#[test]
fn bounded_profile_and_actual_gp_create_change_remove() {
    let input = OrderedInputIdentity::new("group", "time");
    let search = IntegrationSearch::new(2, input.clone(), child(), TaskMode::Forecasting).unwrap();
    assert!(IntegrationSearch::new(0, input.clone(), child(), TaskMode::Forecasting).is_err());
    assert!(IntegrationSearch::new(513, input.clone(), child(), TaskMode::Forecasting).is_err());
    assert!(IntegrationSearch::new(2, input.clone(), child(), TaskMode::IID).is_err());
    let profile = ExperimentalProfile {
        integration: true,
        prior: 0.125,
    };
    let disabled = ExperimentalProfile {
        integration: false,
        prior: 0.125,
    };
    let grow = search.edit(&search.base(), 7, Edit::Grow, profile).unwrap();
    let change = search.edit(&grow, 7, Edit::Change, profile).unwrap();
    assert_ne!(grow.to_bytes().unwrap(), change.to_bytes().unwrap());
    let VersionedProgram::IntegrationV2(g) = &grow else {
        panic!()
    };
    let VersionedProgram::IntegrationV2(c) = &change else {
        panic!()
    };
    assert_eq!(
        g.config()
            .initial()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        c.config()
            .initial()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let drop = search.edit(&change, 7, Edit::Drop, disabled).unwrap();
    assert_eq!(drop.to_bytes().unwrap(), search.base().to_bytes().unwrap());
    assert!(search.edit(&change, 7, Edit::Change, disabled).is_err());
    let mut births = 0;
    for seed in 0..1024 {
        let n = search.generate(seed, profile).unwrap();
        let bytes = n.to_bytes().unwrap();
        assert_eq!(
            search.generate(seed, profile).unwrap().to_bytes().unwrap(),
            bytes
        );
        assert_eq!(
            VersionedProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        assert_eq!(
            search.generate(seed, disabled).unwrap().to_bytes().unwrap(),
            search.base().to_bytes().unwrap()
        );
        births += usize::from(matches!(n, VersionedProgram::IntegrationV2(_)));
    }
    assert_eq!(births, 131);
    for prior in [0.0, -0.0, 0.125, 0.25] {
        let p = ExperimentalProfile {
            integration: true,
            prior,
        };
        let bytes = p.to_bytes().unwrap();
        assert_eq!(
            ExperimentalProfile::from_bytes(&bytes)
                .unwrap()
                .prior
                .to_bits(),
            prior.to_bits()
        );
        for n in 0..17 {
            assert!(ExperimentalProfile::from_bytes(&bytes[..n]).is_err());
        }
    }
    for prior in [f64::NAN, f64::INFINITY, -0.01, 0.250001] {
        assert!(ExperimentalProfile {
            integration: true,
            prior
        }
        .validate()
        .is_err());
    }
    let mut b = profile.to_bytes().unwrap();
    b[7] = 3;
    assert!(ExperimentalProfile::from_bytes(&b).is_err());
    let opaque = GenomeIr::parse("(pipe X (standardize) (ridge lambda_frac=0.01))").unwrap();
    assert!(
        IntegrationSearch::new(2, input.clone(), opaque.clone(), TaskMode::Forecasting).is_err()
    );
    let opaque = VersionedProgram::IntegrationV2(
        IntegrationProgram::new(
            Config::new(2, 60, Mode::Rect, vec![-0., 1.]).unwrap(),
            input,
            opaque.clone(),
        )
        .unwrap(),
    );
    assert!(search.edit(&opaque, 7, Edit::Change, profile).is_err());
    let drop = search.edit(&opaque, 7, Edit::Drop, disabled).unwrap();
    let VersionedProgram::LegacyV1(_) = drop else {
        panic!()
    };
    assert_eq!(
        drop.to_bytes().unwrap(),
        feat_math::kernel::wire_v2::VersionedProgram::LegacyV1(
            feat_math::kernel::Program::LegacyV1(
                GenomeIr::parse("(pipe X (standardize) (ridge lambda_frac=0.01))").unwrap()
            )
        )
        .to_bytes()
        .unwrap()
    );
}
