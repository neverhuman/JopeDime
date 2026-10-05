//! B05–B07 emitted sources and metadata use the unchanged shipped-text lint.
use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarNode, ScalarProducerKind,
    ScalarProducerProfile,
};
use feat_math::math_expr::Var;
use feat_math::transpile::lint_shipped_text;
use serde_json::Value;

fn clean(label: &str, text: &str) {
    assert!(
        lint_shipped_text(text).is_empty(),
        "{label}: {:?}",
        lint_shipped_text(text)
    );
}

#[test]
fn bound_scalar_artifacts_and_complete_metadata_pass_existing_ip_scrub() {
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/scalar_v2/producer-ip.json")).unwrap();
    assert_eq!(fixture["schema"], "feat-math.scalar-artifact-ip-fixture.v1");
    assert_eq!(
        fixture["function_ids"],
        serde_json::json!([13, 19, 20, 21, 22])
    );
    let mut programs = vec![];
    for value in fixture["function_ids"].as_array().unwrap() {
        let id = u16::try_from(value.as_u64().unwrap()).unwrap();
        let program = ScalarExpressionProgram::new(ScalarNode::Call(
            ScalarFunction::from_versioned_id(2, id).unwrap(),
            vec![ScalarNode::Var(Var::Col(0))],
        ))
        .unwrap();
        programs.push(if id == 13 {
            program.with_numerical_policy(NumericalPolicy::StableV2)
        } else {
            program
        });
    }
    for value in fixture["stable_reductions"].as_array().unwrap() {
        let var = match value.as_str().unwrap() {
            "sum" => Var::RowSum,
            "mean" => Var::RowMean,
            "l1" => Var::RowL1,
            "l2" => Var::RowL2,
            _ => panic!("unknown scrub reduction"),
        };
        programs.push(
            ScalarExpressionProgram::new(ScalarNode::Var(var))
                .unwrap()
                .with_numerical_policy(NumericalPolicy::StableV2),
        );
    }
    for policy in [NumericalPolicy::StableV2, NumericalPolicy::LegacyV1] {
        programs.push(
            ScalarExpressionProgram::new(ScalarNode::Var(Var::Col(0)))
                .unwrap()
                .with_numerical_policy(policy),
        );
    }
    let labels: Vec<_> = fixture["label_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f64::from_bits(v.as_u64().unwrap()))
        .collect();
    let rows = vec![
        vec![0.25, -0.25],
        vec![0.5, -0.5],
        vec![0.75, -0.75],
        vec![1.0, -1.0],
    ];
    let mut artifacts = 0;
    for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
        let enabled = ScalarProducerProfile::new(ScalarProducerKind::All, carrier, 2).unwrap();
        for profile in [enabled.clone(), enabled.generation_disabled()] {
            for program in &programs {
                let model = profile.fit_training(program, &rows, &labels, 0.02).unwrap();
                let json = model.to_json().unwrap();
                clean("entire model envelope", &json);
                let (recipes, cpp) = model.emit_recipes().unwrap();
                for (label, text) in [
                    ("Rust", &recipes.rust),
                    ("Python", &recipes.python),
                    ("C++17", &cpp.source),
                    ("whole recipe manifest", &recipes.manifest),
                ] {
                    clean(label, text);
                    artifacts += 1;
                }
                let fitted = model.emit_fitted_artifact().unwrap();
                clean("fitted source", &fitted.source);
                clean("whole fitted codec", &fitted.codec);
                artifacts += 2;
                assert_eq!(model.to_json().unwrap(), json);
                let feat_math::kernel::wire_v2::VersionedModel::ScalarV2(decoded) =
                    feat_math::kernel::wire_v2::VersionedModel::from_json(&json).unwrap()
                else {
                    panic!("scalar model decoded to another model family");
                };
                assert_eq!(decoded.to_json().unwrap(), json);
            }
        }
    }
    assert_eq!(programs.len(), 11);
    assert_eq!(artifacts, 264);
}
