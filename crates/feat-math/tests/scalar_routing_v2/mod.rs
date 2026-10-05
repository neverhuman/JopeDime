use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarNode,
};
use feat_math::kernel::wire_v2::VersionError;
use feat_math::math_expr::Var;
use serde_json::{json, Value};

fn scalar(id: u16, arg: Var, policy: NumericalPolicy) -> ScalarExpressionProgram {
    ScalarExpressionProgram::new(ScalarNode::Call(
        ScalarFunction::from_versioned_id(2, id).unwrap(),
        vec![ScalarNode::Var(arg)],
    ))
    .unwrap()
    .with_numerical_policy(policy)
}

fn rows() -> (Vec<Vec<f64>>, Vec<f64>) {
    let x = (0..32)
        .map(|i| vec![0.02 + i as f64 * 0.025, -0.02 - i as f64 * 0.025])
        .collect();
    let y = (0..32).map(|i| i as f64 * 0.125 - 1.0).collect();
    (x, y)
}

fn prediction_bits(pair: (Vec<f64>, Vec<u32>)) -> (Vec<u64>, Vec<u32>) {
    (pair.0.into_iter().map(f64::to_bits).collect(), pair.1)
}

use feat_math::kernel::wire_v2::{ReservedProgram, VersionedModel, VersionedProgram};

#[test]
fn generic_scalar_route_preserves_model_bits_faults_and_explicit_versions() {
    let (x, y) = rows();
    for id in [13, 19, 20, 21, 22] {
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
                let arg = if carrier == ScalarCarrier::Row {
                    Var::Col(usize::from(id == 22))
                } else {
                    Var::X
                };
                let program = scalar(id, arg, policy);
                let body = program.to_bytes().unwrap();
                let routed = VersionedProgram::from_bytes(&body).unwrap();
                assert_eq!(routed.to_bytes().unwrap(), body);
                let text = routed.to_lossless_text().unwrap();
                assert_eq!(
                    VersionedProgram::parse_lossless_text(&text)
                        .unwrap()
                        .to_bytes()
                        .unwrap(),
                    body
                );
                assert!(matches!(
                    routed.legacy_genome(),
                    Err(VersionError::UnsupportedVersion(_))
                ));
                let direct = program.fit_training(&x, &y, carrier, 0.02).unwrap();
                let fitted = routed.fit_scalar(&x, &y, carrier, 0.02).unwrap();
                assert_eq!(
                    prediction_bits(fitted.predict_scalar(&x).unwrap()),
                    prediction_bits(direct.predict(&x).unwrap())
                );
                let VersionedModel::ScalarV2(model) = fitted else {
                    panic!("wrong route");
                };
                assert_eq!(model.to_json().unwrap(), direct.to_json().unwrap());
                let state = model.to_json().unwrap();
                let restored = VersionedModel::from_json(&state).unwrap();
                assert_eq!(
                    prediction_bits(restored.predict_scalar(&x).unwrap()),
                    prediction_bits(direct.predict(&x).unwrap())
                );
                assert!(matches!(
                    restored.legacy_model(),
                    Err(VersionError::UnsupportedVersion(_))
                ));
                assert!(restored.predict_scalar(&vec![vec![0.0]]).is_err());
                assert_eq!(restored.predict_scalar(&vec![]).unwrap(), (vec![], vec![]));
                let mut future: Value = serde_json::from_str(&state).unwrap();
                future["required_features"] = json!(128);
                future["program"] = json!([255]);
                assert!(matches!(
                    VersionedModel::from_json(&future.to_string()),
                    Err(VersionError::UnsupportedVersion(_))
                ));
            }
        }
    }
    let reserved = VersionedProgram::ReservedV2(ReservedProgram::new(vec![]).unwrap());
    assert!(matches!(
        reserved.fit_scalar(&x, &y, ScalarCarrier::Row, 0.02),
        Err(VersionError::UnsupportedVersion(_))
    ));
}

#[test]
fn generic_stable_reductions_keep_raw_rows_faults_and_training_state() {
    let x = vec![
        vec![1e20, 0.25, -1e20],
        vec![1e20, -0.5, -1e20],
        vec![f64::INFINITY, 0.0, 1.0],
        vec![f64::NAN, -0.0, 0.0],
    ];
    let y = vec![1.0, -2.0, 0.0, 3.0];
    for var in [Var::RowSum, Var::RowMean, Var::RowL1, Var::RowL2] {
        let program = ScalarExpressionProgram::new(ScalarNode::Var(var))
            .unwrap()
            .with_numerical_policy(NumericalPolicy::StableV2);
        let route = VersionedProgram::from_bytes(&program.to_bytes().unwrap()).unwrap();
        let fitted = route.fit_scalar(&x, &y, ScalarCarrier::Row, 0.02).unwrap();
        let direct = program
            .fit_training(&x, &y, ScalarCarrier::Row, 0.02)
            .unwrap();
        assert_eq!(
            prediction_bits(fitted.predict_scalar(&x).unwrap()),
            prediction_bits(direct.predict(&x).unwrap())
        );
        let VersionedModel::ScalarV2(model) = fitted else {
            panic!("wrong stable reduction route")
        };
        let state = model.to_json().unwrap();
        assert_eq!(
            prediction_bits(
                VersionedModel::from_json(&state)
                    .unwrap()
                    .predict_scalar(&x)
                    .unwrap()
            ),
            prediction_bits(direct.predict(&x).unwrap())
        );
        let _ = model.predict(&vec![vec![1e200; 3]]).unwrap();
        assert_eq!(model.to_json().unwrap(), state);
    }
}

#[test]
fn legacy_program_has_no_scalar_fit_coercion() {
    let (x, y) = rows();
    let legacy = feat_math::GenomeIr::parse("(pipe X (ridge))").unwrap();
    let route = VersionedProgram::from_bytes(
        &feat_math::kernel::Program::import_legacy(&legacy)
            .to_bytes()
            .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        route.fit_scalar(&x, &y, ScalarCarrier::Row, 0.02),
        Err(VersionError::UnsupportedVersion(_))
    ));
}

#[test]
fn generic_scalar_signed_zero_fit_never_loses_intercept_or_prediction_bits() {
    let x = vec![
        vec![-3.0, 2.0, 5.0],
        vec![-1.0, 4.0, 5.0],
        vec![2.0, -2.0, 5.0],
        vec![6.0, -5.0, 5.0],
    ];
    for y in [
        vec![-0.0; 4],
        vec![-0.0, 0.0, -0.0, 0.0],
        vec![0.0, -0.0, 0.0, -0.0],
    ] {
        for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
            let var = if carrier == ScalarCarrier::Row {
                Var::Col(0)
            } else {
                Var::X
            };
            let p = ScalarExpressionProgram::new(ScalarNode::Var(var)).unwrap();
            let direct = p.fit_training(&x, &y, carrier, 0.02).unwrap();
            let route = VersionedProgram::from_bytes(&p.to_bytes().unwrap()).unwrap();
            let model = route.fit_scalar(&x, &y, carrier, 0.02).unwrap();
            assert_eq!(
                prediction_bits(model.predict_scalar(&x).unwrap()),
                prediction_bits(direct.predict(&x).unwrap())
            );
            let VersionedModel::ScalarV2(model) = model else {
                panic!("scalar route");
            };
            let state = model.to_json().unwrap();
            assert_eq!(state, direct.to_json().unwrap());
            assert_eq!(
                prediction_bits(
                    VersionedModel::from_json(&state)
                        .unwrap()
                        .predict_scalar(&x)
                        .unwrap()
                ),
                prediction_bits(direct.predict(&x).unwrap())
            );
        }
    }
}
