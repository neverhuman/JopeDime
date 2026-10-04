use feat_math::kernel::{
    codec::identity,
    export::Route,
    inverse_links::{InverseLink, MAX_P, MIN_P},
    numerical_policy::{asinh_v2, reduce_v2, NumericalPolicy, Reduction},
    scalar_execution::ScalarEvaluation,
    scalar_program::{ScalarExpressionProgram, ScalarFunction, ScalarNode},
    wire_v2::{self, VersionedProgram},
    Error, Program,
};
use feat_math::math_expr::{BinOp, Expr, FaultSet, Func, MathExpr, UnOp, Var};

fn call(function: ScalarFunction, argument: ScalarNode) -> ScalarNode {
    ScalarNode::Call(function, vec![argument])
}

fn program(node: ScalarNode, policy: NumericalPolicy) -> ScalarExpressionProgram {
    ScalarExpressionProgram::new(node)
        .unwrap()
        .with_numerical_policy(policy)
}

fn pair(program: &ScalarExpressionProgram, row: &[f64], x: Option<f64>) -> ScalarEvaluation {
    let compiled = program.compile().unwrap();
    assert_eq!(compiled.numerical_policy(), program.numerical_policy());
    let (interpreted, compiled) = match x {
        None => (program.evaluate_row(row), compiled.evaluate_row(row)),
        Some(x) => (
            program.evaluate_element(x, row),
            compiled.evaluate_element(x, row),
        ),
    };
    assert_eq!(interpreted.faults, compiled.faults);
    match (&interpreted.value, compiled.value) {
        (Ok(a), Ok(b)) => assert_eq!(a.to_bits(), b.to_bits()),
        (Err(a), Err(b)) => assert_eq!(*a, b),
        (a, b) => panic!("interpreter/compiler result mismatch: {a:?}, {b:?}"),
    }
    interpreted
}

fn value(actual: ScalarEvaluation, expected: (f64, FaultSet)) {
    assert_eq!(actual.value.unwrap().to_bits(), expected.0.to_bits());
    assert_eq!(actual.faults, expected.1);
}

fn rows() -> Vec<Vec<f64>> {
    vec![
        vec![],
        vec![0.0, -0.0],
        vec![1.0, -2.0, 0.25],
        vec![1e200, -1e200, 1.0],
        vec![f64::MAX, -f64::MAX, 3.0],
        vec![f64::from_bits(1), -f64::from_bits(1)],
        vec![f64::NAN, 1.0],
        vec![f64::INFINITY, f64::NEG_INFINITY],
    ]
}

fn legacy(expression: &MathExpr, row: &[f64], x: Option<f64>, policy: NumericalPolicy) {
    let program = ScalarExpressionProgram::from_v1(expression)
        .unwrap()
        .with_numerical_policy(policy);
    let old = expression.compile().unwrap();
    let (expected, old_compiled, succeeded) = match x {
        None => (
            expression.eval_row_traced(row),
            old.eval_row_traced(row),
            expression.eval_row(row).is_ok(),
        ),
        Some(x) => (
            expression.eval_element_traced(x, row),
            old.eval_element_traced(x, row),
            expression.eval_element(x, row).is_ok(),
        ),
    };
    assert_eq!(expected.0.to_bits(), old_compiled.0.to_bits());
    assert_eq!(expected.1, old_compiled.1);
    let actual = pair(&program, row, x);
    assert_eq!(actual.value.is_ok(), succeeded);
    assert_eq!(actual.value.unwrap_or(0.0).to_bits(), expected.0.to_bits());
    assert_eq!(actual.faults, expected.1);
}

#[test]
fn explicit_policy_executes_the_selected_asinh_in_interpreter_and_compiler() {
    let expression = MathExpr::parse("asinh(c0)").unwrap();
    let program = ScalarExpressionProgram::from_v1(&expression)
        .unwrap()
        .with_numerical_policy(NumericalPolicy::StableV2);
    let compiled = program.compile().unwrap();
    for input in [1e200, -1e200, 0.0, -0.0, f64::from_bits(1)] {
        let expected = asinh_v2(input);
        for actual in [
            program.evaluate_row(&[input]),
            compiled.evaluate_row(&[input]),
        ] {
            assert_eq!(actual.value.unwrap().to_bits(), expected.0.to_bits());
            assert_eq!(actual.faults, expected.1);
        }
    }
}

#[test]
fn absent_policy_matches_every_legacy_function_variable_and_binary_operator() {
    let mut nodes = vec![Expr::Unary(UnOp::Neg, Box::new(Expr::Var(Var::X)))];
    for id in 0..=18 {
        let ScalarFunction::V1(function) = ScalarFunction::from_versioned_id(2, id).unwrap() else {
            panic!("the frozen prefix is legacy");
        };
        let count = match function {
            Func::Min | Func::Max | Func::Step => 2,
            Func::Clip => 3,
            _ => 1,
        };
        nodes.push(Expr::Call(function, vec![Expr::Var(Var::Col(0)); count]));
    }
    for var in [
        Var::X,
        Var::Col(0),
        Var::Col(usize::MAX),
        Var::RowMean,
        Var::RowSum,
        Var::RowL1,
        Var::RowL2,
        Var::RowMin,
        Var::RowMax,
        Var::RowStd,
    ] {
        nodes.push(Expr::Var(var));
    }
    for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow] {
        nodes.push(Expr::Binary(
            op,
            Box::new(Expr::Var(Var::Col(0))),
            Box::new(Expr::Var(Var::X)),
        ));
    }
    for node in nodes {
        let expression = MathExpr::from_ast(node).unwrap();
        for row in rows() {
            for x in [None, Some(-0.0), Some(-1.5), Some(f64::NAN), Some(f64::MAX)] {
                legacy(&expression, &row, x, NumericalPolicy::LegacyV1);
            }
        }
    }
}

#[test]
fn decoded_binary_and_lossless_text_execute_without_changing_policy_identity() {
    let program = ScalarExpressionProgram::from_v1(&MathExpr::parse("asinh(row_sum)").unwrap())
        .unwrap()
        .with_numerical_policy(NumericalPolicy::StableV2);
    let bytes = program.to_bytes().unwrap();
    let versioned = VersionedProgram::from_bytes(&bytes).unwrap();
    let text = versioned.to_lossless_text().unwrap();
    for decoded in [
        versioned,
        VersionedProgram::parse_lossless_text(&text).unwrap(),
    ] {
        let VersionedProgram::ScalarV2(scalar) = decoded else {
            panic!("scalar envelope");
        };
        let before = scalar.to_bytes().unwrap();
        let row = [1e16, 1.0, -1e16];
        value(pair(&scalar, &row, None), asinh_v2(1.0));
        assert_eq!(scalar.to_bytes().unwrap(), before);
        assert_eq!(identity("program", &before), identity("program", &bytes));
        assert_eq!(scalar.numerical_policy(), NumericalPolicy::StableV2);
    }
}

#[test]
fn stable_reduction_dispatch_covers_extremes_cancellation_and_all_widths() {
    let modes = [
        (Var::RowSum, Reduction::Sum),
        (Var::RowMean, Reduction::Mean),
        (Var::RowL1, Reduction::L1),
        (Var::RowL2, Reduction::L2),
    ];
    let mut corpus = rows();
    corpus.extend([
        vec![f64::MAX; 3],
        vec![-f64::MAX; 3],
        vec![3e200, 4e200],
        vec![1e12, f64::from_bits(1)],
        vec![f64::MAX, f64::MAX, -f64::MAX, -f64::MAX, 1.0],
        vec![1e308, 1e100, 1.0, -1e308, -1e100],
    ]);
    for width in 1..=128 {
        corpus.push(
            (0..width)
                .map(|i| [f64::MAX, -f64::MAX, 1.0, f64::from_bits(1)][i % 4])
                .collect(),
        );
        corpus.push(vec![f64::from_bits(1); width]);
    }
    for (var, reduction) in modes {
        let program = program(ScalarNode::Var(var), NumericalPolicy::StableV2);
        for row in &corpus {
            value(pair(&program, row, None), reduce_v2(reduction, row));
        }
    }
}

#[test]
fn legacy_intermediates_are_not_sanitized_before_the_complete_tree() {
    let node = ScalarNode::Binary(
        BinOp::Add,
        Box::new(ScalarNode::Binary(
            BinOp::Div,
            Box::new(ScalarNode::Const(1e200)),
            Box::new(ScalarNode::Const(1e200)),
        )),
        Box::new(ScalarNode::Const(1.0)),
    );
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        value(
            pair(&program(node.clone(), policy), &[], None),
            (2.0, FaultSet::EMPTY),
        );
        value(
            pair(&program(ScalarNode::Const(-0.0), policy), &[], None),
            (-0.0, FaultSet::EMPTY),
        );
        value(
            pair(&program(ScalarNode::Const(f64::MAX), policy), &[], None),
            (1e12, FaultSet::MAGNITUDE_CLAMPED),
        );
    }
}

#[test]
fn nested_policy_calls_retain_the_certified_local_bounds_and_faults() {
    let node = ScalarNode::Binary(
        BinOp::Add,
        Box::new(call(
            ScalarFunction::V1(Func::Asinh),
            ScalarNode::Var(Var::RowL2),
        )),
        Box::new(call(
            ScalarFunction::InverseLink(InverseLink::Logit),
            ScalarNode::Const(2.0),
        )),
    );
    let program = program(node, NumericalPolicy::StableV2);
    for row in rows().into_iter().chain([vec![3e200, 4e200]]) {
        let reduced = reduce_v2(Reduction::L2, &row);
        let transformed = asinh_v2(reduced.0);
        let linked = InverseLink::Logit.evaluate(2.0);
        value(
            pair(&program, &row, None),
            (
                transformed.0 + linked.0,
                reduced.1.union(transformed.1).union(linked.1),
            ),
        );
    }
}

#[test]
fn hard_refusals_preserve_prefix_faults_and_stop_before_later_reads() {
    let guarded = ScalarNode::Binary(
        BinOp::Div,
        Box::new(ScalarNode::Const(1.0)),
        Box::new(ScalarNode::Const(0.0)),
    );
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        let node = ScalarNode::Binary(
            BinOp::Add,
            Box::new(guarded.clone()),
            Box::new(ScalarNode::Var(Var::Col(7))),
        );
        let actual = pair(&program(node, policy), &[1.0], None);
        assert!(matches!(actual.value, Err(Error::Bounds(_))));
        assert_eq!(
            actual.faults,
            FaultSet::DIV_NEAR_ZERO.union(FaultSet::COLUMN_OOB)
        );
        let node = ScalarNode::Binary(
            BinOp::Add,
            Box::new(ScalarNode::Var(Var::Col(7))),
            Box::new(guarded.clone()),
        );
        let actual = pair(&program(node, policy), &[1.0], None);
        assert!(matches!(actual.value, Err(Error::Bounds(_))));
        assert_eq!(actual.faults, FaultSet::COLUMN_OOB);
        let actual = pair(&program(ScalarNode::Var(Var::X), policy), &[], None);
        assert!(matches!(actual.value, Err(Error::Domain(_))));
        assert_eq!(actual.faults, FaultSet::ILLEGAL_SCOPE);
        let clip = ScalarNode::Call(
            ScalarFunction::V1(Func::Clip),
            vec![
                ScalarNode::Const(1.0),
                ScalarNode::Const(f64::NAN),
                ScalarNode::Const(f64::NAN),
            ],
        );
        let actual = pair(
            &program(
                ScalarNode::Binary(BinOp::Add, Box::new(guarded.clone()), Box::new(clip)),
                policy,
            ),
            &[],
            None,
        );
        assert!(matches!(actual.value, Err(Error::Domain(_))));
        assert_eq!(
            actual.faults,
            FaultSet::DIV_NEAR_ZERO.union(FaultSet::INVALID_CLIP_BOUNDS)
        );
    }
}

#[test]
fn inverse_link_execution_uses_the_existing_versioned_kernel_contract() {
    for link in [InverseLink::Probit, InverseLink::Logit] {
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            let program = program(
                call(ScalarFunction::InverseLink(link), ScalarNode::Var(Var::X)),
                policy,
            );
            for input in [
                -1.0,
                -0.0,
                MIN_P,
                0.25,
                0.5,
                0.75,
                MAX_P,
                1.0,
                f64::MAX,
                f64::NAN,
                f64::INFINITY,
            ] {
                let expected = link.evaluate(input);
                let input_fault = if input.is_finite() {
                    FaultSet::EMPTY
                } else {
                    FaultSet::NONFINITE_INPUT
                };
                value(
                    pair(&program, &[], Some(input)),
                    (expected.0, expected.1.union(input_fault)),
                );
            }
        }
    }
}

#[test]
fn unsupported_exponentials_refuse_the_whole_tree_before_reading_any_input() {
    for function in [ScalarFunction::Expm1, ScalarFunction::Log1mexp] {
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            let node = ScalarNode::Binary(
                BinOp::Add,
                Box::new(ScalarNode::Var(Var::Col(usize::MAX))),
                Box::new(call(function, ScalarNode::Var(Var::X))),
            );
            let program = program(node, policy);
            let decoded =
                ScalarExpressionProgram::from_bytes(&program.to_bytes().unwrap()).unwrap();
            assert!(matches!(decoded.compile(), Err(Error::Unsupported(_))));
            for actual in [
                decoded.evaluate_row(&[]),
                decoded.evaluate_element(f64::NAN, &[]),
            ] {
                assert!(matches!(actual.value, Err(Error::Unsupported(_))));
                assert_eq!(actual.faults, FaultSet::EMPTY);
            }
        }
    }
}

#[test]
fn admitted_node_depth_and_column_bounds_remain_safe_during_execution() {
    fn tree(depth: usize) -> ScalarNode {
        if depth == 0 {
            return ScalarNode::Const(1.0);
        }
        ScalarNode::Binary(
            BinOp::Add,
            Box::new(tree(depth - 1)),
            Box::new(tree(depth - 1)),
        )
    }
    // 63 nodes plus one negation: exactly 64, seven levels.
    let node = ScalarNode::Unary(UnOp::Neg, Box::new(tree(5)));
    let valid = program(node.clone(), NumericalPolicy::StableV2);
    value(pair(&valid, &[], None), (-32.0, FaultSet::EMPTY));
    assert!(ScalarExpressionProgram::new(ScalarNode::Unary(UnOp::Neg, Box::new(node))).is_err());
    let mut node = ScalarNode::Const(1.0);
    for _ in 0..15 {
        node = ScalarNode::Unary(UnOp::Neg, Box::new(node));
    }
    value(
        pair(&program(node.clone(), NumericalPolicy::StableV2), &[], None),
        (-1.0, FaultSet::EMPTY),
    );
    assert!(ScalarExpressionProgram::new(ScalarNode::Unary(UnOp::Neg, Box::new(node))).is_err());
    let result = std::panic::catch_unwind(|| {
        pair(
            &program(
                ScalarNode::Var(Var::Col(usize::MAX)),
                NumericalPolicy::StableV2,
            ),
            &[],
            None,
        )
    });
    let result = result.expect("invalid runtime width is a typed refusal, never a panic");
    assert!(matches!(result.value, Err(Error::Bounds(_))));
    assert_eq!(result.faults, FaultSet::COLUMN_OOB);
}

#[test]
fn executing_raw_ieee_literals_never_mutates_the_lossless_payload() {
    let mut bits = 0xa076_1d64_78bd_642f_u64;
    for _ in 0..2048 {
        bits ^= bits << 13;
        bits ^= bits >> 7;
        bits ^= bits << 17;
        let input = f64::from_bits(bits);
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            let scalar = program(ScalarNode::Const(input), policy);
            let bytes = scalar.to_bytes().unwrap();
            let decoded = ScalarExpressionProgram::from_bytes(&bytes).unwrap();
            let actual = pair(&decoded, &[], None);
            let expected = if !input.is_finite() {
                (0.0, FaultSet::NONFINITE_SANITIZED)
            } else if input.abs() > 1e12 {
                (input.clamp(-1e12, 1e12), FaultSet::MAGNITUDE_CLAMPED)
            } else {
                (input, FaultSet::EMPTY)
            };
            value(actual, expected);
            assert_eq!(decoded.to_bytes().unwrap(), bytes);
        }
    }
}

#[test]
fn stable_policy_keeps_unrelated_legacy_functions_and_row_std_unchanged() {
    for id in 0..=18 {
        let ScalarFunction::V1(function) = ScalarFunction::from_versioned_id(2, id).unwrap() else {
            panic!("the frozen prefix is legacy");
        };
        if function == Func::Asinh {
            continue;
        }
        let count = match function {
            Func::Min | Func::Max | Func::Step => 2,
            Func::Clip => 3,
            _ => 1,
        };
        let expression =
            MathExpr::from_ast(Expr::Call(function, vec![Expr::Var(Var::Col(0)); count])).unwrap();
        for row in rows() {
            legacy(&expression, &row, None, NumericalPolicy::StableV2);
        }
    }
    for var in [Var::RowMin, Var::RowMax, Var::RowStd, Var::X, Var::Col(0)] {
        let expression = MathExpr::from_ast(Expr::Var(var)).unwrap();
        for row in rows() {
            legacy(&expression, &row, Some(f64::NAN), NumericalPolicy::StableV2);
        }
    }
}

#[test]
fn standalone_execution_does_not_open_legacy_or_full_recipe_routes() {
    let scalar = program(
        call(ScalarFunction::V1(Func::Asinh), ScalarNode::Const(1e200)),
        NumericalPolicy::StableV2,
    );
    let bytes = scalar.to_bytes().unwrap();
    value(pair(&scalar, &[], None), asinh_v2(1e200));
    let versioned = VersionedProgram::from_bytes(&bytes).unwrap();
    assert!(versioned.legacy_genome().is_err());
    assert!(Program::from_bytes(&bytes).is_err());
    let report = wire_v2::export_report(&versioned);
    for route in [
        report.rust_recipe,
        report.python_recipe,
        report.cpp17_recipe,
        report.cpp17_fitted,
        report.javascript,
    ] {
        assert!(matches!(route, Route::Unsupported(_)));
    }
    assert!(wire_v2::manifest_rows().iter().all(|row| !row.generatable));
}
