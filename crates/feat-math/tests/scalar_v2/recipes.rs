//! Executed native, complete-recipe and fitted-artifact scalar-v2 parity.
use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarLanguage, ScalarNode,
};
use feat_math::math_expr::{BinOp, Var};
use serde_json::Value;
use std::{fmt::Write, fs, path::Path, process::Command};

type RecipeCase = (
    ScalarExpressionProgram,
    ScalarCarrier,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<Vec<f64>>,
);

fn cases() -> Vec<RecipeCase> {
    let mut cases = Vec::new();
    let training = (0..16)
        .map(|i| vec![0.02 + i as f64 * 0.05, -0.02 - i as f64 * 0.05])
        .collect::<Vec<_>>();
    let y = (0..16).map(|i| i as f64 * 0.125 - 1.0).collect::<Vec<_>>();
    let probe = vec![
        vec![0.0, -0.0],
        vec![-0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, -0.0],
        vec![1e200, -1e200],
        vec![1e-300, -1e-300],
        vec![-1e-300, 1e-300],
        vec![f64::NAN, f64::INFINITY],
        vec![0.5, -0.5],
    ];
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
            for id in [13, 19, 20, 21, 22] {
                let argument = if carrier == ScalarCarrier::Row {
                    Var::Col(usize::from(id == 22))
                } else {
                    Var::X
                };
                let node = ScalarNode::Binary(
                    BinOp::Mul,
                    Box::new(ScalarNode::Const(0.5)),
                    Box::new(ScalarNode::Call(
                        ScalarFunction::from_versioned_id(2, id).unwrap(),
                        vec![ScalarNode::Var(argument)],
                    )),
                );
                cases.push((
                    ScalarExpressionProgram::new(node)
                        .unwrap()
                        .with_numerical_policy(policy),
                    carrier,
                    training.clone(),
                    y.clone(),
                    probe.clone(),
                ));
            }
        }
        // Every retained function is also evaluated below a new call: hidden
        // NaN, signed-zero or protected branch differences remain observable.
        for id in 0..=18 {
            let mut args = vec![ScalarNode::Var(Var::Col(0))];
            if matches!(id, 7 | 8 | 10) {
                args.push(ScalarNode::Var(Var::Col(1)));
            }
            if id == 9 {
                args.extend([ScalarNode::Const(-1.0), ScalarNode::Const(1.0)]);
            }
            let old = ScalarNode::Call(ScalarFunction::from_versioned_id(2, id).unwrap(), args);
            let node =
                ScalarNode::Call(ScalarFunction::from_versioned_id(2, 21).unwrap(), vec![old]);
            cases.push((
                ScalarExpressionProgram::new(node)
                    .unwrap()
                    .with_numerical_policy(policy),
                ScalarCarrier::Row,
                training.clone(),
                y.clone(),
                probe.clone(),
            ));
        }
        for variable in [Var::RowMin, Var::RowMax, Var::RowStd] {
            cases.push((
                ScalarExpressionProgram::new(ScalarNode::Call(
                    ScalarFunction::from_versioned_id(2, 21).unwrap(),
                    vec![ScalarNode::Var(variable)],
                ))
                .unwrap()
                .with_numerical_policy(policy),
                ScalarCarrier::Row,
                training.clone(),
                y.clone(),
                probe.clone(),
            ));
        }
        // Old protected operators nested below new calls must preserve their faults.
        let function = |id, argument| {
            ScalarNode::Call(
                ScalarFunction::from_versioned_id(2, id).unwrap(),
                vec![argument],
            )
        };
        let col = |index| ScalarNode::Var(Var::Col(index));
        let mixed = [
            function(
                21,
                ScalarNode::Call(
                    ScalarFunction::from_versioned_id(2, 9).unwrap(),
                    vec![
                        col(0),
                        col(1),
                        ScalarNode::Unary(feat_math::math_expr::UnOp::Neg, Box::new(col(1))),
                    ],
                ),
            ),
            function(
                19,
                function(
                    1,
                    ScalarNode::Unary(feat_math::math_expr::UnOp::Neg, Box::new(col(0))),
                ),
            ),
            function(21, function(12, col(0))),
            function(20, function(2, col(1))),
            function(
                22,
                ScalarNode::Binary(BinOp::Div, Box::new(col(1)), Box::new(col(0))),
            ),
            function(
                19,
                ScalarNode::Binary(
                    BinOp::Pow,
                    Box::new(col(1)),
                    Box::new(ScalarNode::Const(9.0)),
                ),
            ),
            function(20, function(6, function(21, col(0)))),
        ];
        for node in mixed {
            cases.push((
                ScalarExpressionProgram::new(node)
                    .unwrap()
                    .with_numerical_policy(policy),
                ScalarCarrier::Row,
                training.clone(),
                y.clone(),
                probe.clone(),
            ));
        }
        for var in [Var::RowSum, Var::RowMean, Var::RowL1, Var::RowL2] {
            let rows = vec![
                vec![1e20, 0.25, -1e20],
                vec![1e20, -0.5, -1e20],
                vec![-0.0, -0.0, -0.0],
                vec![1.0, 1e-16, 1.0],
            ];
            let probe = vec![
                vec![-0.0; 3],
                vec![3e200, 4e200, 0.0],
                vec![3e-200, 4e-200, 0.0],
                vec![f64::NAN, f64::INFINITY, 0.0],
            ];
            cases.push((
                ScalarExpressionProgram::new(ScalarNode::Var(var))
                    .unwrap()
                    .with_numerical_policy(policy),
                ScalarCarrier::Row,
                rows,
                vec![0.0, 1.0, 2.0, 3.0],
                probe,
            ));
        }
        // Signed zero in fitted state is observable even when final prediction adds +0.
        cases.push((
            ScalarExpressionProgram::new(ScalarNode::Var(Var::RowSum))
                .unwrap()
                .with_numerical_policy(policy),
            ScalarCarrier::Row,
            vec![vec![-0.0; 2]; 4],
            vec![-0.0; 4],
            vec![vec![-0.0; 2], vec![0.0; 2]],
        ));
    }
    cases.extend(signed_zero_label_cases());
    cases.extend(pow_literal_base_cases());
    cases
}

fn pow_literal_base_cases() -> Vec<RecipeCase> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/scalar_v2/pow_literal_bases.v1.json"
    ))
    .unwrap();
    assert_eq!(fixture["schema"], "scalar_v2.pow_literal_bases.v1");
    let rows = |value: &Value| -> Vec<Vec<f64>> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| f64::from_bits(v.as_u64().unwrap()))
                    .collect()
            })
            .collect()
    };
    let training = rows(&fixture["training_bits"]);
    let probe = rows(&fixture["probe_bits"]);
    let labels: Vec<_> = fixture["labels_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f64::from_bits(v.as_u64().unwrap()))
        .collect();
    let mut cases = vec![];
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
            let argument = if carrier == ScalarCarrier::Row {
                Var::Col(0)
            } else {
                Var::X
            };
            for base in fixture["base_bits"].as_array().unwrap() {
                let root = ScalarNode::Binary(
                    BinOp::Pow,
                    Box::new(ScalarNode::Const(f64::from_bits(base.as_u64().unwrap()))),
                    Box::new(ScalarNode::Var(argument)),
                );
                cases.push((
                    ScalarExpressionProgram::new(root)
                        .unwrap()
                        .with_numerical_policy(policy),
                    carrier,
                    training.clone(),
                    labels.clone(),
                    probe.clone(),
                ));
            }
        }
    }
    assert_eq!(cases.len(), 16);
    // Retain the direct reproducing cases first, then make the same literal
    // Pow operands observable beneath new inverse-link/exponential calls.
    for (program, carrier, training, labels, probe) in cases.clone() {
        for id in [19, 21] {
            let nested = ScalarNode::Call(
                ScalarFunction::from_versioned_id(2, id).unwrap(),
                vec![program.ast().clone()],
            );
            cases.push((
                ScalarExpressionProgram::new(nested)
                    .unwrap()
                    .with_numerical_policy(program.numerical_policy()),
                carrier,
                training.clone(),
                labels.clone(),
                probe.clone(),
            ));
        }
    }
    assert_eq!(cases.len(), 48);
    cases
}

fn signed_zero_label_cases() -> Vec<RecipeCase> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/scalar_v2/ridge_signed_zero_labels.v1.json"
    ))
    .unwrap();
    assert_eq!(fixture["schema"], "scalar_v2.ridge_signed_zero_labels.v1");
    let floats = |values: &Value| {
        values
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f64::from_bits(v.as_u64().unwrap()))
            .collect::<Vec<_>>()
    };
    let rows = |values: &Value| {
        values
            .as_array()
            .unwrap()
            .iter()
            .map(&floats)
            .collect::<Vec<_>>()
    };
    let training = rows(&fixture["training_bits"]);
    let probes = rows(&fixture["probe_bits"]);
    assert!(training
        .iter()
        .flatten()
        .all(|v| v.is_finite() && *v != 0.0));
    let mut cases = Vec::new();
    for labels in fixture["cases"].as_array().unwrap() {
        let y = floats(&labels["labels_bits"]);
        assert!(y.iter().all(|v| *v == 0.0));
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            for (carrier, variable) in [
                (ScalarCarrier::Row, Var::Col(0)),
                (ScalarCarrier::Row, Var::Col(2)),
                (ScalarCarrier::Elementwise, Var::X),
            ] {
                let p = ScalarExpressionProgram::new(ScalarNode::Var(variable))
                    .unwrap()
                    .with_numerical_policy(policy);
                let model = p.fit_training(&training, &y, carrier, 0.02).unwrap();
                let state: Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
                let weights = state["ridge_bits"][2].as_array().unwrap();
                assert_eq!(weights[0], labels["intercept_bits"], "{}", labels["name"]);
                assert!(weights[1..].iter().all(|v| v.as_u64() == Some(0)));
                cases.push((p, carrier, training.clone(), y.clone(), probes.clone()));
            }
        }
    }
    cases
}

fn expected(
    case: usize,
    p: &ScalarExpressionProgram,
    c: ScalarCarrier,
    x: &Vec<Vec<f64>>,
    y: &[f64],
    probe: &Vec<Vec<f64>>,
) -> String {
    let m = p.fit_training(x, y, c, 0.02).unwrap();
    let (f, q) = m.transform(probe).unwrap();
    let (pred, faults) = m.predict(probe).unwrap();
    assert_eq!(q, faults);
    let state = m.to_json().unwrap();
    let wire: Value = serde_json::from_str(&state).unwrap();
    let mut out = format!("C {case}\n");
    for row in f {
        for v in row {
            writeln!(out, "F {:016x}", v.to_bits()).unwrap();
        }
    }
    for v in q {
        writeln!(out, "Q {v}").unwrap();
    }
    for row in wire["ridge_bits"].as_array().unwrap() {
        for v in row.as_array().unwrap() {
            writeln!(out, "S {:016x}", v.as_u64().unwrap()).unwrap();
        }
    }
    for v in pred {
        writeln!(out, "P {:016x}", v.to_bits()).unwrap();
    }
    let _ = m.predict(&vec![vec![1e200; x[0].len()]]).unwrap();
    assert_eq!(m.to_json().unwrap(), state);
    out
}

fn literal(value: f64, language: ScalarLanguage) -> String {
    match language {
        ScalarLanguage::Rust => format!("f64::from_bits({}u64)", value.to_bits()),
        ScalarLanguage::Python => format!("scalar_bits_v2({})", value.to_bits()),
        ScalarLanguage::Cpp17 => format!("f64_from_bits({}ULL)", value.to_bits()),
    }
}
fn vector(values: &[f64], language: ScalarLanguage) -> String {
    let inner = values
        .iter()
        .map(|&v| literal(v, language))
        .collect::<Vec<_>>()
        .join(",");
    match language {
        ScalarLanguage::Rust => format!("vec![{inner}]"),
        ScalarLanguage::Python => format!("[{inner}]"),
        ScalarLanguage::Cpp17 => format!("std::vector<double>{{{inner}}}"),
    }
}
fn matrix(rows: &[Vec<f64>], language: ScalarLanguage) -> String {
    let inner = rows
        .iter()
        .map(|r| vector(r, language))
        .collect::<Vec<_>>()
        .join(",");
    match language {
        ScalarLanguage::Rust => format!("vec![{inner}]"),
        ScalarLanguage::Python => format!("[{inner}]"),
        ScalarLanguage::Cpp17 => format!("std::vector<std::vector<double>>{{{inner}}}"),
    }
}
fn run(cmd: &mut Command) -> String {
    let result = cmd.output().unwrap();
    assert!(
        result.status.success(),
        "{cmd:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap()
}

fn assert_output(observed: String, expected: &str, label: &str, path: &Path) {
    fs::write(path, &observed).unwrap();
    let actual = observed.lines().collect::<Vec<_>>();
    let expected = expected.lines().collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len(), "{label}: output count");
    let mut case = "";
    for (line, (a, b)) in actual.iter().zip(&expected).enumerate() {
        if b.starts_with("C ") {
            case = b;
        }
        assert_eq!(a, b, "{label}: {case}, line {}", line + 1);
    }
}

include!("recipes/execution.rs");
include!("recipes/admission.rs");
include!("recipes/sidecars.rs");
