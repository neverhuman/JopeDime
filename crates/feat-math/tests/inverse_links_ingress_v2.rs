//! B05 reader qualification; execution and shipping generation remain held.
use feat_math::kernel::{
    export::Route,
    inverse_links::InverseLink,
    scalar_program::{ScalarExpressionProgram, ScalarFunction, ScalarNode},
    wire_v2::{
        self, Domain, ReservationRef, ReservedModel, ReservedProgram, VersionError, VersionedModel,
        VersionedProgram,
    },
    Program,
};
use feat_math::math_expr::{BinOp, Expr, Func, MathExpr, UnOp, Var};
use feat_math::{GenomeIr, Model};

fn call(link: InverseLink, child: ScalarNode) -> ScalarNode {
    ScalarNode::Call(ScalarFunction::InverseLink(link), vec![child])
}

fn sample() -> ScalarExpressionProgram {
    ScalarExpressionProgram::new(call(InverseLink::Probit, ScalarNode::Var(Var::Col(0)))).unwrap()
}

fn text(bytes: &[u8]) -> String {
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("(program codec=2 hex=\"{hex}\")")
}

fn rejected(bytes: &[u8], unsupported: bool) {
    let results = [
        std::panic::catch_unwind(|| ScalarExpressionProgram::from_bytes(bytes).map(|_| ()))
            .unwrap(),
        std::panic::catch_unwind(|| VersionedProgram::from_bytes(bytes).map(|_| ())).unwrap(),
        std::panic::catch_unwind(|| {
            VersionedProgram::parse_lossless_text(&text(bytes)).map(|_| ())
        })
        .unwrap(),
    ];
    for result in results {
        let error = result.unwrap_err();
        assert!(
            matches!(error, VersionError::UnsupportedVersion(_)) == unsupported,
            "{error:?}"
        );
    }
}

#[test]
fn nested_inverse_links_round_trip_through_typed_binary_and_lossless_text() {
    let root = ScalarNode::Binary(
        BinOp::Add,
        Box::new(call(
            InverseLink::Logit,
            ScalarNode::Call(
                ScalarFunction::V1(Func::Sigmoid),
                vec![ScalarNode::Var(Var::X)],
            ),
        )),
        Box::new(call(InverseLink::Probit, ScalarNode::Const(0.975))),
    );
    let program = ScalarExpressionProgram::new(root.clone()).unwrap();
    let bytes = program.to_bytes().unwrap();
    assert_eq!(&bytes[..8], b"FMKPRG\0\x02");
    assert_eq!(bytes[8], 1);
    assert_eq!(bytes[9], 2);
    assert_eq!(u64::from_le_bytes(bytes[10..18].try_into().unwrap()), 2);
    assert_eq!(
        ScalarExpressionProgram::from_bytes(&bytes).unwrap().ast(),
        &root
    );
    let decoded = VersionedProgram::from_bytes(&bytes).unwrap();
    let VersionedProgram::ScalarV2(copy) = &decoded else {
        panic!("wrong v2 variant")
    };
    assert_eq!(copy.ast(), &root);
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    assert_eq!(decoded.to_lossless_text().unwrap(), text(&bytes));
    assert_eq!(
        VersionedProgram::parse_lossless_text(&text(&bytes))
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
}

fn same_legacy_payload(root: Expr) {
    let expr = MathExpr::from_ast(root).unwrap();
    let genome = GenomeIr::Symbolic {
        source: String::new(),
        expr: expr.clone(),
        affine: false,
    };
    let old = Program::import_legacy(&genome).to_bytes().unwrap();
    // Empty source: v1 magic/variant/genome/affine/source-length occupy 19 bytes.
    let new = ScalarExpressionProgram::from_v1(&expr)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_eq!(&new[18..], &old[19..]);
    assert_eq!(
        ScalarExpressionProgram::from_bytes(&new)
            .unwrap()
            .to_bytes()
            .unwrap(),
        new
    );
    assert_eq!(Program::from_bytes(&old).unwrap().to_bytes().unwrap(), old);
    assert_eq!(
        VersionedProgram::from_bytes(&old)
            .unwrap()
            .to_bytes()
            .unwrap(),
        old
    );
}

#[test]
fn legacy_scalar_tags_arities_and_ieee_payloads_match_the_original_codec() {
    for bits in [
        0,
        1 << 63,
        1,
        0x000f_ffff_ffff_ffff,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0xfff8_0000_0000_4321,
        0.25_f64.to_bits() - 1,
        0.25_f64.to_bits() + 1,
        0.75_f64.to_bits() - 1,
        0.75_f64.to_bits() + 1,
    ] {
        same_legacy_payload(Expr::Const(f64::from_bits(bits)));
    }
    for var in [
        Var::X,
        Var::RowMean,
        Var::RowSum,
        Var::RowL1,
        Var::RowL2,
        Var::RowMin,
        Var::RowMax,
        Var::RowStd,
        Var::Col(0),
        Var::Col(63),
        Var::Col(usize::MAX),
    ] {
        same_legacy_payload(Expr::Var(var));
    }
    same_legacy_payload(Expr::Unary(UnOp::Neg, Box::new(Expr::Const(-0.0))));
    for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow] {
        same_legacy_payload(Expr::Binary(
            op,
            Box::new(Expr::Var(Var::X)),
            Box::new(Expr::Const(-0.0)),
        ));
    }
    for func in [
        Func::Abs,
        Func::Sqrt,
        Func::Log1p,
        Func::Sin,
        Func::Cos,
        Func::Tanh,
        Func::Sigmoid,
        Func::Min,
        Func::Max,
        Func::Clip,
        Func::Step,
        Func::Exp,
        Func::Log,
        Func::Asinh,
        Func::Atan,
        Func::Erf,
        Func::Sign,
        Func::Floor,
        Func::Round,
    ] {
        for n in 0..=4 {
            let root = Expr::Call(func, vec![Expr::Const(0.25); n]);
            let old = MathExpr::from_ast(root.clone());
            let new = ScalarExpressionProgram::new(ScalarNode::Call(
                ScalarFunction::V1(func),
                vec![ScalarNode::Const(0.25); n],
            ));
            assert_eq!(old.is_ok(), new.is_ok(), "{func:?} arity{n}");
            if old.is_ok() {
                same_legacy_payload(root);
            } else {
                // Direct serde ingress can bypass MathExpr::from_ast checks.
                let unchecked: MathExpr =
                    serde_json::from_value(serde_json::json!({"root": root})).unwrap();
                assert!(matches!(
                    ScalarExpressionProgram::from_v1(&unchecked),
                    Err(VersionError::Malformed(_))
                ));
            }
        }
    }
}

#[test]
fn inverse_link_ids_and_arities_require_the_v2_expression_domain() {
    for (id, link, name) in [
        (19, InverseLink::Probit, "probit"),
        (20, InverseLink::Logit, "logit"),
    ] {
        let func = ScalarFunction::from_versioned_id(2, id).unwrap();
        assert_eq!(func, ScalarFunction::InverseLink(link));
        assert_eq!(func.id(), id);
        for version in [0, 1, 3, 255] {
            assert!(matches!(
                ScalarFunction::from_versioned_id(version, id),
                Err(VersionError::UnsupportedVersion(_))
            ));
        }
        for n in [0, 2, 3, 64] {
            assert!(matches!(
                ScalarExpressionProgram::new(ScalarNode::Call(
                    func,
                    vec![ScalarNode::Const(0.5); n]
                )),
                Err(VersionError::Malformed(_))
            ));
        }
        assert!(MathExpr::parse(&format!("{name}(c0)")).is_err());
    }
    for id in 21..=255 {
        assert!(matches!(
            ScalarFunction::from_versioned_id(2, id),
            Err(VersionError::UnsupportedVersion(_))
        ));
    }
    assert!(matches!(
        ScalarFunction::from_versioned_id(2, u16::MAX),
        Err(VersionError::UnsupportedVersion(_))
    ));
}

#[test]
fn corrupt_headers_lengths_and_trailing_bytes_are_typed_without_fallback() {
    let bytes = sample().to_bytes().unwrap();
    for length in 0..bytes.len() {
        rejected(&bytes[..length], false);
    }
    for offset in [7, 8, 9] {
        for value in [3, 255] {
            let mut bad = bytes.clone();
            bad[offset] = value;
            rejected(&bad, true);
        }
    }
    for features in [0_u64, 1, 3, 1 << 63] {
        let mut bad = bytes.clone();
        bad[10..18].copy_from_slice(&features.to_le_bytes());
        rejected(&bad, true);
    }
    let mut bad = bytes.clone();
    bad[0] = b'X';
    rejected(&bad, false);
    let mut bad = bytes.clone();
    bad.push(0);
    rejected(&bad, false);
    let mut bad = bytes.clone();
    bad[20..28].copy_from_slice(&u64::MAX.to_le_bytes());
    rejected(&bad, false);
    for n in [0_u64, 2, 3] {
        let mut bad = bytes.clone();
        bad[20..28].copy_from_slice(&n.to_le_bytes());
        rejected(&bad, false);
    }
}

#[test]
fn nonadmitted_scalar_variable_operator_and_node_tags_are_refused() {
    let bytes = sample().to_bytes().unwrap();
    for id in 21..=255 {
        let mut bad = bytes.clone();
        bad[19] = id;
        rejected(&bad, true);
    }
    let vars = ScalarExpressionProgram::new(ScalarNode::Var(Var::X))
        .unwrap()
        .to_bytes()
        .unwrap();
    for id in 8..=255 {
        let mut bad = vars.clone();
        bad[19] = id;
        rejected(&bad, true);
    }
    let bin = ScalarExpressionProgram::new(ScalarNode::Binary(
        BinOp::Add,
        Box::new(ScalarNode::Const(0.0)),
        Box::new(ScalarNode::Const(1.0)),
    ))
    .unwrap()
    .to_bytes()
    .unwrap();
    for id in 5..=255 {
        let mut bad = bin.clone();
        bad[19] = id;
        rejected(&bad, true);
    }
    for id in 6..=255 {
        let mut bad = bytes.clone();
        bad[18] = id;
        rejected(&bad, true);
    }
}

fn balanced(depth: usize) -> ScalarNode {
    if depth == 0 {
        ScalarNode::Const(0.0)
    } else {
        ScalarNode::Binary(
            BinOp::Add,
            Box::new(balanced(depth - 1)),
            Box::new(balanced(depth - 1)),
        )
    }
}

#[test]
fn existing_node_depth_and_byte_budgets_bound_new_ingress() {
    let mut legacy = Expr::Var(Var::X);
    for _ in 0..16 {
        legacy = Expr::Unary(UnOp::Neg, Box::new(legacy));
    }
    let unchecked: MathExpr = serde_json::from_value(serde_json::json!({"root": legacy})).unwrap();
    assert!(matches!(
        ScalarExpressionProgram::from_v1(&unchecked),
        Err(VersionError::Malformed(_))
    ));
    let mut root = ScalarNode::Var(Var::X);
    for _ in 0..15 {
        root = ScalarNode::Unary(UnOp::Neg, Box::new(root));
    }
    let valid = ScalarExpressionProgram::new(root.clone())
        .unwrap()
        .to_bytes()
        .unwrap();
    assert!(matches!(
        ScalarExpressionProgram::new(ScalarNode::Unary(UnOp::Neg, Box::new(root))),
        Err(VersionError::Malformed(_))
    ));
    let mut bad = valid[..18].to_vec();
    bad.push(3);
    bad.extend_from_slice(&valid[18..]);
    rejected(&bad, false);
    let root = ScalarNode::Unary(UnOp::Neg, Box::new(balanced(5))); // 64 nodes.
    let valid = ScalarExpressionProgram::new(root.clone())
        .unwrap()
        .to_bytes()
        .unwrap();
    assert!(matches!(
        ScalarExpressionProgram::new(ScalarNode::Unary(UnOp::Neg, Box::new(root))),
        Err(VersionError::Malformed(_))
    ));
    let mut bad = valid[..18].to_vec();
    bad.push(3);
    bad.extend_from_slice(&valid[18..]);
    rejected(&bad, false);
    let oversized = vec![0; 16 * 1024 * 1024 + 1];
    rejected(&oversized, false);
}

#[test]
fn parsed_scalar_programs_cannot_enter_legacy_execution_models_or_export() {
    let bytes = sample().to_bytes().unwrap();
    assert!(Program::from_bytes(&bytes).is_err());
    let decoded = VersionedProgram::from_bytes(&bytes).unwrap();
    assert!(matches!(
        decoded.legacy_genome(),
        Err(VersionError::UnsupportedVersion(_))
    ));
    let report = wire_v2::export_report(&decoded);
    for route in [
        report.rust_recipe,
        report.python_recipe,
        report.cpp17_recipe,
        report.cpp17_fitted,
        report.javascript,
    ] {
        assert!(matches!(route, Route::Unsupported(_)));
    }
    let model = Model::Symbolic {
        expr: MathExpr::parse("x").unwrap(),
        a: 1.0,
        b: 0.0,
    };
    let legacy_json = serde_json::to_string(&model).unwrap();
    assert_eq!(
        serde_json::to_string(
            VersionedModel::from_json(&legacy_json)
                .unwrap()
                .legacy_model()
                .unwrap()
        )
        .unwrap(),
        legacy_json
    );
    let reservations =
        ReservedProgram::new(vec![ReservationRef::new(Domain::Function, 19).unwrap()]).unwrap();
    let envelope = ReservedModel::new(
        &model,
        &reservations,
        "a".repeat(64),
        "v2-reservations".into(),
        vec!["prediction".into()],
    )
    .unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&envelope.to_json().unwrap()).unwrap();
    value["program"] = serde_json::to_value(&bytes).unwrap();
    assert!(matches!(
        VersionedModel::from_json(&value.to_string()),
        Err(VersionError::UnsupportedVersion(_))
    ));
    value["required_features"] = 2.into();
    assert!(matches!(
        VersionedModel::from_json(&value.to_string()),
        Err(VersionError::UnsupportedVersion(_))
    ));
    assert!(serde_json::from_value::<Model>(value).is_err());
    for profile in feat_math::kernel::profile::shipped_kernel2_profiles() {
        let functions = profile.expr.unwrap().functions;
        assert!(!functions.contains(&"probit") && !functions.contains(&"logit"));
    }
}

#[test]
fn lossless_text_requires_matching_codec_and_bounded_well_formed_hex() {
    let bytes = sample().to_bytes().unwrap();
    assert!(
        VersionedProgram::parse_lossless_text(&text(&bytes).replace("codec=2", "codec=1")).is_err()
    );
    for value in [
        "(program codec=2 hex=\"0\")",
        "(program codec=2 hex=\"zz\")",
        "(program codec=2)",
    ] {
        assert!(matches!(
            VersionedProgram::parse_lossless_text(value),
            Err(VersionError::Malformed(_))
        ));
    }
    assert!(matches!(
        VersionedProgram::parse_lossless_text("(program codec=3 hex=\"00\")"),
        Err(VersionError::UnsupportedVersion(_))
    ));
}
