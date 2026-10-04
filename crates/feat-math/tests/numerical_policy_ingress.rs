use feat_math::kernel::{
    codec::identity,
    export::Route,
    inverse_links::InverseLink,
    numerical_policy::{NumericalPolicy, KERNEL_ID, POLICY_ID},
    scalar_program::{ScalarExpressionProgram, ScalarFunction, ScalarNode},
    wire_v2::{
        self, ReservedModel, ReservedProgram, VersionError, VersionResult, VersionedModel,
        VersionedProgram,
    },
    Program,
};
use feat_math::math_expr::{BinOp, Func, MathExpr, UnOp, Var};
use feat_math::Model;

fn text(bytes: &[u8]) -> String {
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("(program codec=2 hex=\"{hex}\")")
}

fn ingress(bytes: &[u8]) -> [VersionResult<Vec<u8>>; 3] {
    std::panic::catch_unwind(|| {
        [
            ScalarExpressionProgram::from_bytes(bytes).and_then(|p| p.to_bytes()),
            VersionedProgram::from_bytes(bytes).and_then(|p| p.to_bytes()),
            VersionedProgram::parse_lossless_text(&text(bytes)).and_then(|p| p.to_bytes()),
        ]
    })
    .expect("ingress must return a typed refusal, never panic")
}

fn encoded(root: ScalarNode) -> Vec<u8> {
    ScalarExpressionProgram::new(root)
        .unwrap()
        .with_numerical_policy(NumericalPolicy::StableV2)
        .to_bytes()
        .unwrap()
}

fn unary(function: ScalarFunction, child: ScalarNode) -> ScalarNode {
    ScalarNode::Call(function, vec![child])
}

fn refused(bytes: &[u8], unsupported: bool) {
    for result in ingress(bytes) {
        let error = result.unwrap_err();
        assert_eq!(
            matches!(error, VersionError::UnsupportedVersion(_)),
            unsupported,
            "{error:?}"
        );
    }
}

#[test]
fn explicit_numerical_policy_round_trips_before_any_execution() {
    let expression = MathExpr::parse("asinh(row_l2)").unwrap();
    let mut bytes = ScalarExpressionProgram::from_v1(&expression)
        .unwrap()
        .to_bytes()
        .unwrap();
    bytes[10..18].copy_from_slice(&10_u64.to_le_bytes());
    bytes.insert(18, 2);
    let parsed = ScalarExpressionProgram::from_bytes(&bytes)
        .expect("explicit numerical policy belongs in the versioned scalar envelope");
    assert_eq!(parsed.to_bytes().unwrap(), bytes);
    assert_eq!(parsed.numerical_policy(), NumericalPolicy::StableV2);
    for result in ingress(&bytes) {
        assert_eq!(result.unwrap(), bytes);
    }
}

#[test]
fn missing_policy_keeps_the_exact_existing_scalar_payload_and_defaults() {
    let expression = MathExpr::parse("asinh(row_sum)").unwrap();
    let program = ScalarExpressionProgram::from_v1(&expression).unwrap();
    let bytes = program.to_bytes().unwrap();
    let mut expected = b"FMKPRG\0\x02\x01\x02".to_vec();
    expected.extend_from_slice(&2_u64.to_le_bytes());
    expected.extend_from_slice(&[5, 13]);
    expected.extend_from_slice(&1_u64.to_le_bytes());
    expected.extend_from_slice(&[2, 2]);
    assert_eq!(bytes, expected);
    assert_eq!(NumericalPolicy::default(), NumericalPolicy::LegacyV1);
    assert_eq!(program.numerical_policy().name(), "v1");
    assert_eq!(program.numerical_policy().kernel_identity(), None);
    assert_eq!(
        ScalarExpressionProgram::from_bytes(&bytes)
            .unwrap()
            .numerical_policy(),
        NumericalPolicy::LegacyV1
    );
    let stable = program.with_numerical_policy(NumericalPolicy::StableV2);
    let new_bytes = stable.to_bytes().unwrap();
    assert_ne!(identity("program", &bytes), identity("program", &new_bytes));
    assert_eq!(
        stable
            .with_numerical_policy(NumericalPolicy::LegacyV1)
            .to_bytes()
            .unwrap(),
        bytes
    );
    for result in ingress(&bytes) {
        assert_eq!(result.unwrap(), bytes);
    }
}

#[test]
fn policy_changes_only_its_declaration_and_retains_every_scalar_id() {
    for id in 0..=22 {
        let function = ScalarFunction::from_versioned_id(2, id).unwrap();
        let arity = match function {
            ScalarFunction::V1(Func::Min | Func::Max | Func::Step) => 2,
            ScalarFunction::V1(Func::Clip) => 3,
            _ => 1,
        };
        let program = ScalarExpressionProgram::new(ScalarNode::Call(
            function,
            vec![ScalarNode::Const(-0.0); arity],
        ))
        .unwrap();
        let old = program.to_bytes().unwrap();
        let stable = program.with_numerical_policy(NumericalPolicy::StableV2);
        let new = stable.to_bytes().unwrap();
        assert_eq!(&new[..10], &old[..10]);
        assert_eq!(new.len(), old.len() + 1);
        assert_eq!(&new[19..], &old[18..]);
        assert_eq!(new[18], 2);
        assert_eq!(new[20], id as u8);
        let old_features = u64::from_le_bytes(old[10..18].try_into().unwrap());
        assert_eq!(
            u64::from_le_bytes(new[10..18].try_into().unwrap()),
            old_features | 8
        );
        assert_eq!(stable.numerical_policy().name(), POLICY_ID);
        assert_eq!(stable.numerical_policy().kernel_identity(), Some(KERNEL_ID));
        for result in ingress(&new) {
            assert_eq!(result.unwrap(), new);
        }
    }
    for (id, var) in [
        Var::X,
        Var::RowMean,
        Var::RowSum,
        Var::RowL1,
        Var::RowL2,
        Var::RowMin,
        Var::RowMax,
        Var::RowStd,
    ]
    .into_iter()
    .enumerate()
    {
        let bytes = encoded(ScalarNode::Var(var));
        assert_eq!(&bytes[19..], &[2, id as u8]);
        for result in ingress(&bytes) {
            assert_eq!(result.unwrap(), bytes);
        }
    }
}

#[test]
fn policy_round_trips_preserve_ieee_literals_and_canonical_fixed_points() {
    let fixed = [
        0,
        1 << 63,
        1,
        0x000f_ffff_ffff_ffff,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_1234,
        0xfff8_0000_0000_5678,
    ];
    let mut seed = 0xb07a_5e01_u64;
    let random = (0..1024).map(|_| {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        seed
    });
    for bits in fixed.into_iter().chain(random) {
        let root = ScalarNode::Binary(
            BinOp::Add,
            Box::new(unary(
                ScalarFunction::V1(Func::Asinh),
                ScalarNode::Unary(UnOp::Neg, Box::new(ScalarNode::Const(f64::from_bits(bits)))),
            )),
            Box::new(unary(
                ScalarFunction::Expm1,
                unary(
                    ScalarFunction::InverseLink(InverseLink::Logit),
                    ScalarNode::Var(Var::Col(63)),
                ),
            )),
        );
        let program = ScalarExpressionProgram::new(root).unwrap();
        let legacy = program.to_bytes().unwrap();
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            let bytes = program
                .clone()
                .with_numerical_policy(policy)
                .to_bytes()
                .unwrap();
            let parsed = ScalarExpressionProgram::from_bytes(&bytes).unwrap();
            assert_eq!(parsed.numerical_policy(), policy);
            for result in ingress(&bytes) {
                assert_eq!(result.unwrap(), bytes);
            }
            let outer = VersionedProgram::from_bytes(&bytes).unwrap();
            let canonical = outer.to_lossless_text().unwrap();
            assert_eq!(canonical, text(&bytes));
            assert_eq!(
                VersionedProgram::parse_lossless_text(&canonical)
                    .unwrap()
                    .to_lossless_text()
                    .unwrap(),
                canonical
            );
            assert_eq!(
                parsed
                    .with_numerical_policy(NumericalPolicy::LegacyV1)
                    .to_bytes()
                    .unwrap(),
                legacy
            );
        }
    }
}

#[test]
fn unsupported_policies_and_required_features_never_fall_back() {
    let original = encoded(ScalarNode::Var(Var::X));
    for policy in 0..=255 {
        if policy == 2 {
            continue;
        }
        let mut bad = original.clone();
        bad[18] = policy;
        refused(&bad, true);
    }
    for feature in [
        0_u64,
        1,
        3,
        4,
        5,
        7,
        8,
        9,
        11,
        12,
        13,
        15,
        1 << 63,
        u64::MAX,
    ] {
        let mut bad = original.clone();
        bad[10..18].copy_from_slice(&feature.to_le_bytes());
        refused(&bad, true);
    }
}

#[test]
fn policy_and_body_features_cannot_be_stripped_or_overdeclared() {
    for function in [ScalarFunction::V1(Func::Asinh), ScalarFunction::Expm1] {
        let original = encoded(unary(function, ScalarNode::Var(Var::X)));
        let features = u64::from_le_bytes(original[10..18].try_into().unwrap());
        let mut wrong_body = original.clone();
        wrong_body[10..18].copy_from_slice(&(features ^ 4).to_le_bytes());
        refused(&wrong_body, false);
        let mut stripped = original.clone();
        stripped[10..18].copy_from_slice(&(features & !8).to_le_bytes());
        refused(&stripped, false);
        let mut missing = original;
        missing.remove(18);
        refused(&missing, true);
    }
    for root in [ScalarNode::Var(Var::X), ScalarNode::Const(0.0)] {
        let mut stripped = encoded(root);
        stripped[10..18].copy_from_slice(&2_u64.to_le_bytes());
        refused(&stripped, false);
    }
}

#[test]
fn every_truncation_trailing_byte_and_invalid_arity_is_refused() {
    for root in [
        ScalarNode::Const(f64::from_bits(0x7ff8_0000_0000_1234)),
        ScalarNode::Var(Var::Col(usize::MAX)),
        unary(ScalarFunction::V1(Func::Asinh), ScalarNode::Var(Var::RowL2)),
        unary(ScalarFunction::Log1mexp, ScalarNode::Const(-1e-10)),
    ] {
        let original = encoded(root);
        for length in 0..original.len() {
            refused(&original[..length], false);
        }
        let mut trailing = original;
        trailing.push(0);
        refused(&trailing, false);
    }
    let original = encoded(unary(
        ScalarFunction::V1(Func::Asinh),
        ScalarNode::Var(Var::X),
    ));
    for arity in [0_u64, 2, 3, 64, u64::MAX] {
        let mut bad = original.clone();
        bad[21..29].copy_from_slice(&arity.to_le_bytes());
        refused(&bad, false);
    }
}

fn balanced(depth: usize) -> ScalarNode {
    if depth == 0 {
        ScalarNode::Const(1.0)
    } else {
        ScalarNode::Binary(
            BinOp::Add,
            Box::new(balanced(depth - 1)),
            Box::new(balanced(depth - 1)),
        )
    }
}

#[test]
fn numerical_policy_preserves_the_existing_node_depth_and_byte_budgets() {
    let mut root = ScalarNode::Const(-0.0);
    for _ in 0..15 {
        root = ScalarNode::Unary(UnOp::Neg, Box::new(root));
    }
    let bytes = encoded(root.clone());
    for result in ingress(&bytes) {
        assert_eq!(result.unwrap(), bytes);
    }
    assert!(matches!(
        ScalarExpressionProgram::new(ScalarNode::Unary(UnOp::Neg, Box::new(root))),
        Err(VersionError::Malformed(_))
    ));
    let mut too_deep = bytes;
    too_deep.insert(19, 3);
    refused(&too_deep, false);
    let root = ScalarNode::Unary(UnOp::Neg, Box::new(balanced(5)));
    let bytes = encoded(root.clone());
    for result in ingress(&bytes) {
        assert_eq!(result.unwrap(), bytes);
    }
    assert!(matches!(
        ScalarExpressionProgram::new(ScalarNode::Unary(UnOp::Neg, Box::new(root))),
        Err(VersionError::Malformed(_))
    ));
    let mut too_many = bytes;
    too_many.insert(19, 3);
    refused(&too_many, false);
    refused(&vec![0; 16 * 1024 * 1024 + 1], false);
}

#[test]
fn policy_does_not_assign_new_function_variable_node_or_envelope_ids() {
    let call = encoded(unary(
        ScalarFunction::V1(Func::Asinh),
        ScalarNode::Var(Var::X),
    ));
    for id in 23..=255 {
        let mut bad = call.clone();
        bad[20] = id;
        refused(&bad, true);
        assert!(matches!(
            ScalarFunction::from_versioned_id(2, u16::from(id)),
            Err(VersionError::UnsupportedVersion(_))
        ));
    }
    let var = encoded(ScalarNode::Var(Var::X));
    for id in 8..=255 {
        let mut bad = var.clone();
        bad[20] = id;
        refused(&bad, true);
    }
    for tag in 6..=255 {
        let mut bad = call.clone();
        bad[19] = tag;
        refused(&bad, true);
    }
    for offset in [7, 8, 9] {
        for version in [0, 1, 3, 255] {
            if offset == 8 && version == 1 {
                continue;
            }
            let mut bad = call.clone();
            bad[offset] = version;
            refused(&bad, true);
        }
    }
}

#[test]
fn declaring_policy_cannot_admit_legacy_execution_models_or_export() {
    let bytes = encoded(unary(
        ScalarFunction::V1(Func::Asinh),
        ScalarNode::Var(Var::RowL2),
    ));
    assert!(Program::from_bytes(&bytes).is_err());
    let parsed = VersionedProgram::from_bytes(&bytes).unwrap();
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
        expr: MathExpr::parse("asinh(x)").unwrap(),
        a: 1.0,
        b: 0.0,
    };
    let old_json = serde_json::to_string(&model).unwrap();
    assert!(matches!(
        VersionedModel::from_json(&old_json).unwrap(),
        VersionedModel::LegacyV1(_)
    ));
    let envelope = ReservedModel::new(
        &model,
        &ReservedProgram::new(vec![]).unwrap(),
        "a".repeat(64),
        "v2-reservations".into(),
        vec!["prediction".into()],
    )
    .unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&envelope.to_json().unwrap()).unwrap();
    value["program"] = serde_json::to_value(&bytes).unwrap();
    for feature in [1, 2, 6, 8, 10, 14] {
        value["required_features"] = feature.into();
        assert!(matches!(
            VersionedModel::from_json(&value.to_string()),
            Err(VersionError::UnsupportedVersion(_))
        ));
    }
    assert!(serde_json::from_value::<Model>(value).is_err());
}
