//! B06 bounded reader proof; scalar execution and shipping admission stay held.
use feat_math::kernel::{
    export::Route,
    inverse_links::InverseLink,
    scalar_program::{ScalarExpressionProgram, ScalarFunction, ScalarNode},
    wire_v2::{
        self, Domain, ReservationRef, ReservedModel, ReservedProgram, VersionError, VersionResult,
        VersionedModel, VersionedProgram,
    },
    Program,
};
use feat_math::math_expr::{BinOp, Func, MathExpr, Var};
use feat_math::Model;

fn ingress(bytes: &[u8]) -> [VersionResult<Vec<u8>>; 3] {
    let encoded = bytes.iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
        s
    });
    std::panic::catch_unwind(|| {
        [
            ScalarExpressionProgram::from_bytes(bytes).and_then(|p| p.to_bytes()),
            VersionedProgram::from_bytes(bytes).and_then(|p| p.to_bytes()),
            VersionedProgram::parse_lossless_text(&format!("(program codec=2 hex=\"{encoded}\")"))
                .and_then(|p| p.to_bytes()),
        ]
    })
    .expect("bounded ingress must return a typed refusal, never panic")
}

fn unary(function: ScalarFunction, child: ScalarNode) -> ScalarNode {
    ScalarNode::Call(function, vec![child])
}

fn bytes(function: ScalarFunction) -> Vec<u8> {
    ScalarExpressionProgram::new(unary(function, ScalarNode::Var(Var::X)))
        .unwrap()
        .to_bytes()
        .unwrap()
}

#[test]
fn stable_function_ids_are_append_only_unary_and_version_two_only() {
    for (id, function, name) in [
        (21, ScalarFunction::Expm1, "expm1"),
        (22, ScalarFunction::Log1mexp, "log1mexp"),
    ] {
        assert_eq!(ScalarFunction::from_versioned_id(2, id).unwrap(), function);
        assert_eq!(function.id(), id);
        for version in [0, 1, 3, 255] {
            assert!(matches!(
                ScalarFunction::from_versioned_id(version, id),
                Err(VersionError::UnsupportedVersion(_))
            ));
        }
        for arity in [0, 2, 3, 64] {
            assert!(matches!(
                ScalarExpressionProgram::new(ScalarNode::Call(
                    function,
                    vec![ScalarNode::Const(1e-10); arity]
                )),
                Err(VersionError::Malformed(_))
            ));
        }
        assert!(MathExpr::parse(&format!("{name}(x)")).is_err());
    }
}

#[test]
fn nested_stable_calls_round_trip_without_normalizing_ieee_constants() {
    for bits in [
        0,
        1 << 63,
        1,
        0x000f_ffff_ffff_ffff,
        0x7fef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_1234,
        0xfff8_0000_0000_5678,
        1e-10_f64.to_bits(),
        (-1e-10_f64).to_bits(),
        (-1.0_f64).to_bits(),
        (-40.0_f64).to_bits(),
    ] {
        let root = ScalarNode::Binary(
            BinOp::Add,
            Box::new(unary(
                ScalarFunction::Expm1,
                ScalarNode::Const(f64::from_bits(bits)),
            )),
            Box::new(unary(
                ScalarFunction::Log1mexp,
                unary(
                    ScalarFunction::InverseLink(InverseLink::Logit),
                    ScalarNode::Var(Var::Col(0)),
                ),
            )),
        );
        let original = ScalarExpressionProgram::new(root)
            .unwrap()
            .to_bytes()
            .unwrap();
        assert_eq!(&original[..10], b"FMKPRG\0\x02\x01\x02");
        assert_eq!(u64::from_le_bytes(original[10..18].try_into().unwrap()), 6);
        for result in ingress(&original) {
            assert_eq!(result.unwrap(), original);
        }
    }
}

#[test]
fn existing_scalar_envelopes_keep_feature_two_and_exact_function_payloads() {
    for (id, function) in [
        (11, ScalarFunction::V1(Func::Exp)),
        (12, ScalarFunction::V1(Func::Log)),
        (19, ScalarFunction::InverseLink(InverseLink::Probit)),
        (20, ScalarFunction::InverseLink(InverseLink::Logit)),
    ] {
        let actual = bytes(function);
        let mut expected = b"FMKPRG\0\x02\x01\x02".to_vec();
        expected.extend_from_slice(&2_u64.to_le_bytes());
        expected.extend_from_slice(&[5, id]);
        expected.extend_from_slice(&1_u64.to_le_bytes());
        expected.extend_from_slice(&[2, 0]); // Var::X
        assert_eq!(actual, expected);
        for result in ingress(&expected) {
            assert_eq!(result.unwrap(), expected);
        }
    }
}

#[test]
fn stable_feature_is_derived_from_every_branch_and_must_match_the_body() {
    for root in [
        unary(ScalarFunction::Expm1, ScalarNode::Var(Var::X)),
        unary(
            ScalarFunction::V1(Func::Abs),
            unary(ScalarFunction::Log1mexp, ScalarNode::Const(-1.0)),
        ),
        ScalarNode::Binary(
            BinOp::Sub,
            Box::new(ScalarNode::Const(0.0)),
            Box::new(unary(ScalarFunction::Expm1, ScalarNode::Const(-40.0))),
        ),
    ] {
        let mut encoded = ScalarExpressionProgram::new(root)
            .unwrap()
            .to_bytes()
            .unwrap();
        assert_eq!(u64::from_le_bytes(encoded[10..18].try_into().unwrap()), 6);
        encoded[10..18].copy_from_slice(&2_u64.to_le_bytes());
        for result in ingress(&encoded) {
            assert!(matches!(result, Err(VersionError::Malformed(_))));
        }
    }
    let mut overdeclared = bytes(ScalarFunction::InverseLink(InverseLink::Probit));
    overdeclared[10..18].copy_from_slice(&6_u64.to_le_bytes());
    for result in ingress(&overdeclared) {
        assert!(matches!(result, Err(VersionError::Malformed(_))));
    }
}

#[test]
fn stable_payload_truncation_arity_overflow_and_trailing_bytes_refuse() {
    for function in [ScalarFunction::Expm1, ScalarFunction::Log1mexp] {
        let original = bytes(function);
        let mut malformed = (0..original.len())
            .map(|n| original[..n].to_vec())
            .collect::<Vec<_>>();
        for count in [0_u64, 2, 3, u64::MAX] {
            let mut bad = original.clone();
            bad[20..28].copy_from_slice(&count.to_le_bytes());
            malformed.push(bad);
        }
        let mut trailing = original;
        trailing.push(0);
        malformed.push(trailing);
        for bad in malformed {
            for result in ingress(&bad) {
                assert!(
                    matches!(result, Err(VersionError::Malformed(_))),
                    "{result:?}"
                );
            }
        }
    }
}

#[test]
fn stable_calls_do_not_expand_existing_depth_node_or_byte_budgets() {
    let mut root = ScalarNode::Var(Var::X);
    for _ in 0..15 {
        root = unary(ScalarFunction::Expm1, root);
    }
    let valid = ScalarExpressionProgram::new(root.clone())
        .unwrap()
        .to_bytes()
        .unwrap();
    assert!(ScalarExpressionProgram::new(unary(ScalarFunction::Log1mexp, root)).is_err());
    for result in ingress(&valid) {
        assert_eq!(result.unwrap(), valid);
    }
    let mut too_deep = valid[..18].to_vec();
    too_deep.extend_from_slice(&[5, 22]);
    too_deep.extend_from_slice(&1_u64.to_le_bytes());
    too_deep.extend_from_slice(&valid[18..]);
    for result in ingress(&too_deep) {
        assert!(matches!(result, Err(VersionError::Malformed(_))));
    }

    let mut level = vec![ScalarNode::Const(0.0); 32];
    for _ in 0..5 {
        level = level
            .chunks_exact(2)
            .map(|pair| {
                ScalarNode::Binary(
                    BinOp::Add,
                    Box::new(pair[0].clone()),
                    Box::new(pair[1].clone()),
                )
            })
            .collect();
    }
    let root = unary(ScalarFunction::Expm1, level.pop().unwrap()); // 64 nodes.
    let sixty_four = ScalarExpressionProgram::new(root.clone())
        .unwrap()
        .to_bytes()
        .unwrap();
    for result in ingress(&sixty_four) {
        assert_eq!(result.unwrap(), sixty_four);
    }
    assert!(matches!(
        ScalarExpressionProgram::new(unary(ScalarFunction::Log1mexp, root)),
        Err(VersionError::Malformed(_))
    ));
    for result in ingress(&vec![0; 16 * 1024 * 1024 + 1]) {
        assert!(matches!(result, Err(VersionError::Malformed(_))));
    }
}

#[test]
fn unassigned_ids_versions_features_and_node_tags_remain_unsupported() {
    for id in (23..=255).chain([u16::MAX]) {
        assert!(matches!(
            ScalarFunction::from_versioned_id(2, id),
            Err(VersionError::UnsupportedVersion(_))
        ));
    }
    let original = bytes(ScalarFunction::Expm1);
    let mut unsupported = Vec::new();
    for id in 23..=255 {
        let mut bad = original.clone();
        bad[19] = id;
        unsupported.push(bad);
    }
    for feature in [0_u64, 1, 3, 4, 5, 7, 8, u64::MAX] {
        let mut bad = original.clone();
        bad[10..18].copy_from_slice(&feature.to_le_bytes());
        unsupported.push(bad);
    }
    for offset in [7, 8, 9, 18] {
        let mut bad = original.clone();
        bad[offset] = 255;
        unsupported.push(bad);
    }
    for bad in unsupported {
        for result in ingress(&bad) {
            assert!(
                matches!(result, Err(VersionError::UnsupportedVersion(_))),
                "{result:?}"
            );
        }
    }
}

#[test]
fn stable_ingress_cannot_be_misrouted_to_execution_models_exports_or_search() {
    for function in [ScalarFunction::Expm1, ScalarFunction::Log1mexp] {
        let encoded = bytes(function);
        assert!(Program::from_bytes(&encoded).is_err());
        let parsed = VersionedProgram::from_bytes(&encoded).unwrap();
        assert!(matches!(
            parsed.legacy_genome(),
            Err(VersionError::UnsupportedVersion(_))
        ));
        let report = wire_v2::export_report(&parsed);
        assert!([
            report.rust_recipe,
            report.python_recipe,
            report.cpp17_recipe,
            report.cpp17_fitted,
            report.javascript
        ]
        .iter()
        .all(|r| matches!(r, Route::Unsupported(_))));
        let model = Model::Symbolic {
            expr: MathExpr::parse("x").unwrap(),
            a: 1.0,
            b: 0.0,
        };
        let reserved =
            ReservedProgram::new(vec![
                ReservationRef::new(Domain::Function, function.id()).unwrap()
            ])
            .unwrap();
        let envelope = ReservedModel::new(
            &model,
            &reserved,
            "a".repeat(64),
            "v2-reservations".into(),
            vec!["prediction".into()],
        )
        .unwrap();
        let mut value: serde_json::Value =
            serde_json::from_str(&envelope.to_json().unwrap()).unwrap();
        value["program"] = serde_json::to_value(&encoded).unwrap();
        for feature in [1, 2, 6] {
            value["required_features"] = feature.into();
            assert!(matches!(
                VersionedModel::from_json(&value.to_string()),
                Err(VersionError::UnsupportedVersion(_))
            ));
        }
        assert!(serde_json::from_value::<Model>(value).is_err());
    }
    for profile in feat_math::kernel::profile::shipped_kernel2_profiles() {
        let functions = profile.expr.unwrap().functions;
        assert!(!functions.contains(&"expm1") && !functions.contains(&"log1mexp"));
    }
}
