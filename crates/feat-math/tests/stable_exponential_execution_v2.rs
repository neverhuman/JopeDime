//! Qualified exponential calls in bounded versioned programs; legacy routes stay closed.
use feat_math::kernel::{
    export::Route,
    inverse_links::InverseLink,
    numerical_policy::NumericalPolicy,
    scalar_execution::ScalarEvaluation,
    scalar_program::{ScalarExpressionProgram, ScalarFunction, ScalarNode},
    stable_exponential::StableExponential,
    wire_v2::{self, VersionError, VersionedProgram},
    Error, Program,
};
use feat_math::math_expr::{BinOp, FaultSet, MathExpr, UnOp, Var};
use serde::Deserialize;

fn invoke(function: ScalarFunction, child: ScalarNode) -> ScalarNode {
    ScalarNode::Call(function, vec![child])
}

fn evaluate(
    program: &ScalarExpressionProgram,
    row: &[f64],
    element: Option<f64>,
) -> ScalarEvaluation {
    let compiled = program.compile().expect("qualified functions compile");
    let (tree, code) = match element {
        Some(x) => (
            program.evaluate_element(x, row),
            compiled.evaluate_element(x, row),
        ),
        None => (program.evaluate_row(row), compiled.evaluate_row(row)),
    };
    assert_eq!(
        tree.value.as_ref().map(|x| x.to_bits()),
        code.value.as_ref().map(|x| x.to_bits())
    );
    assert_eq!(tree.faults, code.faults);
    tree
}

#[derive(Deserialize)]
struct BoundaryCorpus {
    entries: Vec<BoundaryCase>,
}

#[derive(Deserialize)]
struct BoundaryCase {
    function_id: u16,
    input_bits: String,
    oracle_bits: String,
    expected_fault_bits: String,
}

fn raw(text: &str) -> u64 {
    u64::from_str_radix(text, 16).unwrap()
}

fn expect_value(result: ScalarEvaluation, expected: (f64, FaultSet)) {
    assert_eq!(result.value.unwrap().to_bits(), expected.0.to_bits());
    assert_eq!(result.faults, expected.1);
}

fn selected(node: ScalarNode, policy: NumericalPolicy) -> ScalarExpressionProgram {
    ScalarExpressionProgram::new(node)
        .unwrap()
        .with_numerical_policy(policy)
}

#[test]
fn versioned_exponential_calls_execute_the_frozen_boundary_corpus() {
    let corpus: BoundaryCorpus = serde_json::from_str(include_str!(
        "fixtures/math_boundary/v2/stable_exponential.json"
    ))
    .unwrap();
    assert_eq!(corpus.entries.len(), 3048);
    for case in corpus.entries {
        let input = f64::from_bits(raw(&case.input_bits));
        let helper = StableExponential::from_versioned_id(2, case.function_id).unwrap();
        let expected = helper.evaluate(input);
        let reference = f64::from_bits(raw(&case.oracle_bits));
        assert_eq!(expected.0.is_sign_negative(), reference.is_sign_negative());
        let steps = expected
            .0
            .abs()
            .to_bits()
            .abs_diff(reference.abs().to_bits());
        assert!(steps <= if reference.is_normal() { 2 } else { 1 });
        assert_eq!(u64::from(expected.1.bits()), raw(&case.expected_fault_bits));
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            for (variable, row, element) in [
                (Var::X, vec![], Some(input)),
                (Var::Col(0), vec![input], None),
            ] {
                let function = ScalarFunction::from_versioned_id(2, case.function_id).unwrap();
                let source =
                    ScalarExpressionProgram::new(invoke(function, ScalarNode::Var(variable)))
                        .unwrap()
                        .with_numerical_policy(policy);
                let bytes = source.to_bytes().unwrap();
                let decoded = ScalarExpressionProgram::from_bytes(&bytes).unwrap();
                let result = evaluate(&decoded, &row, element);
                assert_eq!(result.value.unwrap().to_bits(), expected.0.to_bits());
                assert_eq!(result.faults, expected.1);
                assert_eq!(decoded.to_bytes().unwrap(), bytes);
                assert_eq!(decoded.numerical_policy(), policy);
            }
        }
    }
}

#[test]
fn nested_exponentials_preserve_qualified_guards_and_raw_intermediates() {
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        for (inner, outer) in [
            (StableExponential::Expm1, StableExponential::Log1mexp),
            (StableExponential::Log1mexp, StableExponential::Expm1),
        ] {
            let node = invoke(
                ScalarFunction::from_versioned_id(2, outer.id()).unwrap(),
                invoke(
                    ScalarFunction::from_versioned_id(2, inner.id()).unwrap(),
                    ScalarNode::Var(Var::X),
                ),
            );
            let program = selected(node, policy);
            for input in [
                f64::NAN,
                f64::INFINITY,
                -f64::MAX,
                -746.0,
                -40.0,
                -1e-10,
                -f64::from_bits(1),
                -0.0,
                0.0,
                30.0,
                f64::MAX,
            ] {
                let a = inner.evaluate(input);
                let b = outer.evaluate(a.0);
                expect_value(evaluate(&program, &[], Some(input)), (b.0, a.1.union(b.1)));
            }
        }
        let ratio = ScalarNode::Binary(
            BinOp::Div,
            Box::new(ScalarNode::Const(1e200)),
            Box::new(ScalarNode::Const(1e200)),
        );
        let program = selected(invoke(ScalarFunction::Expm1, ratio), policy);
        expect_value(
            evaluate(&program, &[], None),
            StableExponential::Expm1.evaluate(1.0),
        );
    }
}

#[test]
fn exponential_calls_keep_prefix_faults_before_typed_scope_and_width_errors() {
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        for function in [ScalarFunction::Expm1, ScalarFunction::Log1mexp] {
            let input = if function == ScalarFunction::Expm1 {
                f64::MAX
            } else {
                1.0
            };
            let guarded = invoke(function, ScalarNode::Const(input));
            let prefix = StableExponential::from_versioned_id(2, function.id())
                .unwrap()
                .evaluate(input)
                .1;
            for (bad, marker) in [
                (Var::Col(usize::MAX), FaultSet::COLUMN_OOB),
                (Var::X, FaultSet::ILLEGAL_SCOPE),
            ] {
                for reversed in [false, true] {
                    let mut children = [guarded.clone(), ScalarNode::Var(bad)];
                    if reversed {
                        children.reverse();
                    }
                    let [left, right] = children;
                    let program = selected(
                        ScalarNode::Binary(BinOp::Add, Box::new(left), Box::new(right)),
                        policy,
                    );
                    let actual = evaluate(&program, &[], None);
                    match bad {
                        Var::Col(_) => assert!(matches!(actual.value, Err(Error::Bounds(_)))),
                        Var::X => assert!(matches!(actual.value, Err(Error::Domain(_)))),
                        _ => unreachable!(),
                    }
                    assert_eq!(
                        actual.faults,
                        if reversed {
                            marker
                        } else {
                            prefix.union(marker)
                        }
                    );
                }
            }
        }
    }
}

#[test]
fn decoded_execution_retains_ieee_literals_and_policy_identity() {
    let mut state = 0x94d0_49bb_1331_11eb_u64;
    for _ in 0..2048 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let input = f64::from_bits(state);
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            for function in [ScalarFunction::Expm1, ScalarFunction::Log1mexp] {
                let source = selected(invoke(function, ScalarNode::Const(input)), policy);
                let original = source.to_bytes().unwrap();
                let versioned = VersionedProgram::from_bytes(&original).unwrap();
                let text = versioned.to_lossless_text().unwrap();
                let VersionedProgram::ScalarV2(decoded) =
                    VersionedProgram::parse_lossless_text(&text).unwrap()
                else {
                    panic!("scalar versioned envelope");
                };
                let expected = StableExponential::from_versioned_id(2, function.id())
                    .unwrap()
                    .evaluate(input);
                expect_value(evaluate(&decoded, &[], None), expected);
                assert_eq!(decoded.to_bytes().unwrap(), original);
                assert_eq!(decoded.numerical_policy(), policy);
                let features = u64::from_le_bytes(original[10..18].try_into().unwrap());
                assert_eq!(
                    features,
                    if policy == NumericalPolicy::StableV2 {
                        14
                    } else {
                        6
                    }
                );
            }
        }
    }
}

#[test]
fn exponential_nodes_keep_admitted_depth_node_and_arity_budgets() {
    fn balanced(depth: usize) -> ScalarNode {
        if depth == 0 {
            ScalarNode::Const(-0.25)
        } else {
            ScalarNode::Binary(
                BinOp::Add,
                Box::new(balanced(depth - 1)),
                Box::new(balanced(depth - 1)),
            )
        }
    }
    for function in [ScalarFunction::Expm1, ScalarFunction::Log1mexp] {
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            let mut deep = ScalarNode::Const(-0.25);
            for _ in 0..15 {
                deep = invoke(function, deep);
            }
            let valid = selected(deep.clone(), policy);
            assert!(evaluate(&valid, &[], None).value.unwrap().is_finite());
            assert!(ScalarExpressionProgram::new(invoke(function, deep)).is_err());
            let full = invoke(function, balanced(5)); // 63 children plus the call: 64 nodes.
            let valid = selected(full.clone(), policy);
            expect_value(
                evaluate(&valid, &[], None),
                StableExponential::from_versioned_id(2, function.id())
                    .unwrap()
                    .evaluate(-8.0),
            );
            assert!(
                ScalarExpressionProgram::new(ScalarNode::Unary(UnOp::Neg, Box::new(full))).is_err()
            );
        }
        for arity in [0, 2, 3, 64] {
            assert!(matches!(
                ScalarExpressionProgram::new(ScalarNode::Call(
                    function,
                    vec![ScalarNode::Const(-1.0); arity]
                )),
                Err(VersionError::Malformed(_))
            ));
        }
    }
}

#[test]
fn exponential_support_stays_closed_to_legacy_models_exports_and_profiles() {
    for (function, name) in [
        (ScalarFunction::Expm1, "expm1"),
        (ScalarFunction::Log1mexp, "log1mexp"),
    ] {
        let source = selected(
            invoke(function, ScalarNode::Const(-1e-10)),
            NumericalPolicy::StableV2,
        );
        assert!(evaluate(&source, &[], None).value.is_ok());
        let bytes = source.to_bytes().unwrap();
        let versioned = VersionedProgram::from_bytes(&bytes).unwrap();
        assert!(versioned.legacy_genome().is_err());
        assert!(Program::from_bytes(&bytes).is_err());
        assert!(MathExpr::parse(&format!("{name}(c0)")).is_err());
        let report = wire_v2::export_report(&versioned);
        assert!([
            report.rust_recipe,
            report.python_recipe,
            report.cpp17_recipe,
            report.cpp17_fitted,
            report.javascript,
        ]
        .iter()
        .all(|route| matches!(route, Route::Unsupported(_))));
        for profile in feat_math::kernel::profile::shipped_kernel2_profiles() {
            assert!(!profile.expr.unwrap().functions.contains(&name));
        }
        let mut wrong_version = bytes.clone();
        wrong_version[7] = 1;
        assert!(matches!(
            ScalarExpressionProgram::from_bytes(&wrong_version),
            Err(VersionError::UnsupportedVersion(_))
        ));
        for id in [23, 34, u8::MAX] {
            let mut unknown = bytes.clone();
            unknown[20] = id; // Explicit policy adds one byte before the call's function tag.
            assert!(matches!(
                ScalarExpressionProgram::from_bytes(&unknown),
                Err(VersionError::UnsupportedVersion(_))
            ));
        }
    }
    assert!(wire_v2::manifest_rows().iter().all(|row| !row.generatable));
}

#[test]
fn both_policies_keep_the_same_versioned_exponential_contract() {
    for function in [ScalarFunction::Expm1, ScalarFunction::Log1mexp] {
        for variable in [Var::RowSum, Var::RowMean, Var::RowL1, Var::RowL2] {
            let scalar = selected(
                invoke(function, ScalarNode::Var(variable)),
                NumericalPolicy::StableV2,
            );
            let reduction = match variable {
                Var::RowSum => feat_math::kernel::numerical_policy::Reduction::Sum,
                Var::RowMean => feat_math::kernel::numerical_policy::Reduction::Mean,
                Var::RowL1 => feat_math::kernel::numerical_policy::Reduction::L1,
                Var::RowL2 => feat_math::kernel::numerical_policy::Reduction::L2,
                _ => unreachable!(),
            };
            for row in [
                vec![],
                vec![f64::MAX; 3],
                vec![3e200, 4e200],
                vec![1e16, 1.0, -1e16],
                vec![f64::NAN],
            ] {
                let reduced = feat_math::kernel::numerical_policy::reduce_v2(reduction, &row);
                let linked = StableExponential::from_versioned_id(2, function.id())
                    .unwrap()
                    .evaluate(reduced.0);
                expect_value(
                    evaluate(&scalar, &row, None),
                    (linked.0, reduced.1.union(linked.1)),
                );
            }
        }
        let source = selected(
            invoke(function, ScalarNode::Var(Var::X)),
            NumericalPolicy::LegacyV1,
        );
        let explicit = source
            .clone()
            .with_numerical_policy(NumericalPolicy::StableV2);
        for x in [-f64::MAX, -0.0, -1e-10, 0.0, 30.0, f64::NAN] {
            let legacy = evaluate(&source, &[], Some(x));
            let stable = evaluate(&explicit, &[], Some(x));
            expect_value(stable, (legacy.value.unwrap(), legacy.faults));
        }
    }
}

#[test]
fn nested_scalar_programs_replay_interpreted_and_compiled_values_and_faults() {
    let mut state = 0xd1b5_4a32_d192_ed03_u64;
    for _ in 0..2048 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let input = f64::from_bits(state);
        let mut node = ScalarNode::Var(Var::X);
        let mut expected = (input, FaultSet::EMPTY);
        for level in 0..6 {
            let function = match (state >> (level * 3)) & 3 {
                0 => ScalarFunction::Expm1,
                1 => ScalarFunction::Log1mexp,
                2 => ScalarFunction::InverseLink(InverseLink::Probit),
                _ => ScalarFunction::InverseLink(InverseLink::Logit),
            };
            let local = match function {
                ScalarFunction::Expm1 => StableExponential::Expm1.evaluate(expected.0),
                ScalarFunction::Log1mexp => StableExponential::Log1mexp.evaluate(expected.0),
                ScalarFunction::InverseLink(link) => link.evaluate(expected.0),
                ScalarFunction::V1(_) => unreachable!(),
            };
            node = invoke(function, node);
            expected = (local.0, expected.1.union(local.1));
            if (state >> (level * 3 + 2)) & 1 != 0 {
                node = ScalarNode::Unary(UnOp::Neg, Box::new(node));
                expected.0 = -expected.0;
            }
        }
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            expect_value(
                evaluate(&selected(node.clone(), policy), &[], Some(input)),
                expected,
            );
        }
    }
}
