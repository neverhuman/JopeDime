use feat_math::kernel::export::{FittedBasis, Route};
use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarLanguage, ScalarNode,
};
use feat_math::kernel::wire_v2::{ReservedProgram, VersionError, VersionedModel, VersionedProgram};
use feat_math::math_expr::Var;

fn program(var: Var, id: Option<u16>) -> ScalarExpressionProgram {
    let node = match id {
        Some(id) => ScalarNode::Call(
            ScalarFunction::from_versioned_id(2, id).unwrap(),
            vec![ScalarNode::Var(var)],
        ),
        None => ScalarNode::Var(var),
    };
    ScalarExpressionProgram::new(node)
        .unwrap()
        .with_numerical_policy(NumericalPolicy::StableV2)
}

#[test]
fn generic_scalar_exports_preserve_native_sources_codecs_and_immutable_state() {
    let x = vec![vec![0.25, -0.75], vec![0.5, -0.5], vec![0.75, -0.25]];
    let y = [1.0, -2.0, 3.0];
    for (var, id) in [
        (Var::Col(0), Some(19)),
        (Var::Col(0), Some(20)),
        (Var::Col(0), Some(21)),
        (Var::Col(1), Some(22)),
        (Var::Col(0), Some(13)),
        (Var::RowSum, None),
        (Var::RowMean, None),
        (Var::RowL1, None),
        (Var::RowL2, None),
        (Var::X, Some(21)),
    ] {
        let carrier = if var == Var::X {
            ScalarCarrier::Elementwise
        } else {
            ScalarCarrier::Row
        };
        let p = program(var, id);
        let route = VersionedProgram::from_bytes(&p.to_bytes().unwrap()).unwrap();
        let report = route.scalar_export_report(carrier, 2, 0.02);
        assert_eq!(report.rust_recipe, Route::Ok);
        assert_eq!(report.python_recipe, Route::Ok);
        assert_eq!(report.cpp17_recipe, Route::Ok);
        assert_eq!(report.cpp17_fitted, Route::Ok);
        assert_eq!(report.cpp17_fitted_basis, FittedBasis::Declared);
        assert!(matches!(report.javascript, Route::Unsupported(_)));
        let model = route.fit_scalar(&x, &y, carrier, 0.02).unwrap();
        let direct = p.fit_training(&x, &y, carrier, 0.02).unwrap();
        let saved = direct.to_json().unwrap();
        for language in [
            ScalarLanguage::Rust,
            ScalarLanguage::Python,
            ScalarLanguage::Cpp17,
        ] {
            let source = p.emit_recipe(language, carrier, 2, 0.02).unwrap();
            assert_eq!(
                route
                    .emit_scalar_recipe(language, carrier, 2, 0.02)
                    .unwrap(),
                source
            );
            assert_eq!(model.emit_scalar_recipe(language).unwrap(), source);
            let restored = VersionedModel::from_json(&saved).unwrap();
            assert_eq!(restored.emit_scalar_recipe(language).unwrap(), source);
        }
        let expected = direct.emit_fitted_artifact().unwrap();
        let actual = model.emit_scalar_fitted_cpp17().unwrap();
        assert_eq!(actual.source, expected.source);
        assert_eq!(actual.codec, saved);
        assert_eq!(actual.source_sha256, expected.source_sha256);
        assert_eq!(actual.codec_sha256, expected.codec_sha256);
        let VersionedModel::ScalarV2(model) = model else {
            panic!("wrong scalar model")
        };
        assert_eq!(model.to_json().unwrap(), saved);
    }
}

#[test]
fn scalar_export_support_and_recipe_refusals_share_the_native_fit_scope() {
    for (var, carrier, width, lambda) in [
        (Var::X, ScalarCarrier::Row, 2, 0.02),
        (Var::Col(2), ScalarCarrier::Row, 2, 0.02),
        (Var::Col(0), ScalarCarrier::Row, 0, 0.02),
        (Var::Col(0), ScalarCarrier::Row, 513, 0.02),
        (Var::Col(0), ScalarCarrier::Row, 2, f64::NAN),
        (Var::Col(0), ScalarCarrier::Row, 2, 0.0),
        (Var::Col(0), ScalarCarrier::Row, 2, 2.01),
    ] {
        let p = program(var, Some(21));
        let route = VersionedProgram::ScalarV2(p.clone());
        let native = p
            .plan_training(carrier, width, lambda)
            .unwrap_err()
            .to_string();
        let report = route.scalar_export_report(carrier, width, lambda);
        for support in [
            report.rust_recipe,
            report.python_recipe,
            report.cpp17_recipe,
            report.cpp17_fitted,
        ] {
            assert_eq!(support, Route::Unsupported(vec![native.clone()]));
        }
        for language in [
            ScalarLanguage::Rust,
            ScalarLanguage::Python,
            ScalarLanguage::Cpp17,
        ] {
            let native =
                VersionError::from(p.emit_recipe(language, carrier, width, lambda).unwrap_err());
            assert_eq!(
                route
                    .emit_scalar_recipe(language, carrier, width, lambda)
                    .unwrap_err(),
                native
            );
        }
    }
    let legacy = feat_math::GenomeIr::parse("(pipe X (ridge))").unwrap();
    let model = VersionedModel::LegacyV1(
        feat_math::fit(&legacy, &vec![vec![0.25], vec![0.75]], &[1.0, -1.0]).unwrap(),
    );
    assert!(matches!(
        model.emit_scalar_recipe(ScalarLanguage::Rust),
        Err(VersionError::UnsupportedVersion(_))
    ));
    assert!(matches!(
        model.emit_scalar_fitted_cpp17(),
        Err(VersionError::UnsupportedVersion(_))
    ));
    for route in [
        VersionedProgram::ReservedV2(ReservedProgram::new(vec![]).unwrap()),
        VersionedProgram::LegacyV1(feat_math::kernel::Program::import_legacy(&legacy)),
    ] {
        assert!(matches!(
            route
                .scalar_export_report(ScalarCarrier::Row, 2, 0.02)
                .rust_recipe,
            Route::Unsupported(_)
        ));
        assert!(matches!(
            route.emit_scalar_recipe(ScalarLanguage::Rust, ScalarCarrier::Row, 2, 0.02),
            Err(VersionError::UnsupportedVersion(_))
        ));
    }
}
