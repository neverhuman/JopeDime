use feat_math::heads::RidgeModel;
use feat_math::kernel::numerical_policy::{reduce_v2, NumericalPolicy, Reduction};
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarModel, ScalarNode,
};
use feat_math::kernel::wire_v2::VersionError;
use feat_math::kernel::Error;
use feat_math::math_expr::{FaultSet, Var};
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

#[test]
fn every_new_scalar_fits_both_carriers_and_preserves_exact_state() {
    let (x, y) = rows();
    for id in 19..=22 {
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
                let argument = if carrier == ScalarCarrier::Row {
                    Var::Col(usize::from(id == 22))
                } else {
                    Var::X
                };
                let p = scalar(id, argument, policy);
                let old = p.to_bytes().unwrap();
                let m = p.fit_training(&x, &y, carrier, 0.02).unwrap();
                let (features, faults) = m.transform(&x).unwrap();
                let native = RidgeModel::fit(&features, &y, 0.02).predict(&features);
                let (actual, observed) = m.predict(&x).unwrap();
                assert_eq!(faults, observed);
                assert_eq!(
                    native.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                for (row, feature) in x.iter().zip(&features) {
                    for (column, value) in feature.iter().enumerate() {
                        let direct = if carrier == ScalarCarrier::Row {
                            p.evaluate_row(row)
                        } else {
                            p.evaluate_element(row[column], row)
                        };
                        assert_eq!(direct.value.unwrap().to_bits(), value.to_bits());
                    }
                }
                let text = m.to_json().unwrap();
                let restored = ScalarModel::from_json(&text).unwrap();
                assert_eq!(restored.to_json().unwrap(), text);
                assert_eq!(restored.identity().unwrap(), m.identity().unwrap());
                assert_eq!(restored.predict(&x).unwrap(), m.predict(&x).unwrap());
                assert_eq!(restored.program().to_bytes().unwrap(), old);
                let wire: Value = serde_json::from_str(&text).unwrap();
                assert_eq!(wire["numerical_policy"], policy.name());
                assert_eq!(wire["required_features"], p.required_features().unwrap());
                assert_eq!(
                    wire["ridge_bits"][0].as_array().unwrap().len(),
                    if carrier == ScalarCarrier::Row { 1 } else { 2 }
                );
            }
        }
    }
}

#[test]
fn numerical_policy_reaches_large_tiny_signed_zero_and_cancellation_features() {
    let x = vec![vec![0.0, -0.0], vec![1e200, -1e200], vec![1e-300, -1e-300]];
    let p = scalar(13, Var::X, NumericalPolicy::StableV2);
    let model = p
        .fit_training(&x, &[0.0, 1.0, 2.0], ScalarCarrier::Elementwise, 0.01)
        .unwrap();
    let (f, faults) = model.transform(&x).unwrap();
    assert_eq!(f[0][1].to_bits(), (-0.0f64).to_bits());
    assert!(f[1][0] > 460.0 && f[1][0] < 462.0);
    assert_eq!(f[2][0].to_bits(), 1e-300f64.to_bits());
    assert_eq!(faults, vec![0, 0, 0]);
    for (var, reduction) in [
        (Var::RowSum, Reduction::Sum),
        (Var::RowMean, Reduction::Mean),
        (Var::RowL1, Reduction::L1),
        (Var::RowL2, Reduction::L2),
    ] {
        let p = ScalarExpressionProgram::new(ScalarNode::Var(var))
            .unwrap()
            .with_numerical_policy(NumericalPolicy::StableV2);
        let x = vec![vec![1e12, 1e-10, -1e12], vec![3e200, 4e200, 0.0]];
        let m = p
            .fit_training(&x, &[0.0, 1.0], ScalarCarrier::Row, 0.01)
            .unwrap();
        let (f, faults) = m.transform(&x).unwrap();
        for i in 0..x.len() {
            let expected = reduce_v2(reduction, &x[i]);
            assert_eq!(f[i][0].to_bits(), expected.0.to_bits());
            assert_eq!(faults[i], expected.1.bits());
        }
        if var == Var::RowSum {
            assert_eq!(f[0][0].to_bits(), 1e-10f64.to_bits());
        }
    }
    // Explicit v1 still follows the frozen overflowing asinh expression.
    let legacy = p.clone().with_numerical_policy(NumericalPolicy::LegacyV1);
    assert_eq!(legacy.evaluate_element(1e200, &[1e200]).value.unwrap(), 0.0);
}

#[test]
fn nonfinite_domain_and_clamp_faults_survive_fitting_and_prediction() {
    let x = vec![vec![0.2], vec![0.4], vec![0.8]];
    for id in 19..=22 {
        let p = scalar(id, Var::Col(0), NumericalPolicy::StableV2);
        let m = p
            .fit_training(&x, &[0.0, 1.0, 2.0], ScalarCarrier::Row, 0.01)
            .unwrap();
        let probe = vec![
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![-0.0],
            vec![1000.0],
        ];
        let (f, faults) = m.transform(&probe).unwrap();
        let (predictions, observed) = m.predict(&probe).unwrap();
        assert_eq!(observed, faults);
        assert!(predictions.iter().all(|v| v.is_finite()));
        for (i, row) in probe.iter().enumerate() {
            let direct = p.evaluate_row(row);
            assert_eq!(f[i][0].to_bits(), direct.value.unwrap().to_bits());
            assert_eq!(faults[i], direct.faults.bits());
        }
        assert_ne!(faults[0], FaultSet::EMPTY.bits());
    }
}

#[test]
fn fitting_refuses_bad_shape_scope_labels_and_parameters_without_panicking() {
    let (x, y) = rows();
    let p = scalar(21, Var::Col(0), NumericalPolicy::StableV2);
    for (bad, labels) in [
        (vec![], vec![]),
        (vec![vec![]], vec![0.0]),
        (vec![vec![0.0]; 2], vec![0.0]),
        (vec![vec![0.0], vec![0.0, 1.0]], vec![0.0, 1.0]),
        (vec![vec![0.0]; 2], vec![0.0, f64::NAN]),
    ] {
        assert!(p
            .fit_training(&bad, &labels, ScalarCarrier::Row, 0.01)
            .is_err());
    }
    for lambda in [0.0, 1e-6, 2.01, f64::NAN, f64::INFINITY] {
        assert!(p.fit_training(&x, &y, ScalarCarrier::Row, lambda).is_err());
    }
    let x_node = scalar(19, Var::X, NumericalPolicy::StableV2);
    assert!(matches!(
        x_node.fit_training(&x, &y, ScalarCarrier::Row, 0.01),
        Err(Error::Domain(_))
    ));
    let col = scalar(21, Var::Col(2), NumericalPolicy::StableV2);
    assert!(matches!(
        col.fit_training(&x, &y, ScalarCarrier::Elementwise, 0.01),
        Err(Error::Bounds(_))
    ));
    let model = p.fit_training(&x, &y, ScalarCarrier::Row, 0.01).unwrap();
    assert_eq!(model.predict(&vec![]).unwrap(), (vec![], vec![]));
    assert!(matches!(
        model.predict(&vec![vec![0.0]]),
        Err(Error::Shape(_))
    ));
    assert!(p
        .fit_training(&vec![vec![0.0; 513]], &[0.0], ScalarCarrier::Row, 0.01)
        .is_err());
    assert!(p
        .fit_training(&vec![vec![0.0; 512]], &[0.0], ScalarCarrier::Row, 0.01)
        .is_ok());
}

#[test]
fn held_out_poison_cannot_change_fitted_state_or_serialization() {
    let (x, y) = rows();
    let p = scalar(20, Var::Col(0), NumericalPolicy::StableV2);
    let m = p
        .fit_training(&x[..24].to_vec(), &y[..24], ScalarCarrier::Row, 0.01)
        .unwrap();
    let before = m.to_json().unwrap();
    m.predict(&vec![vec![f64::NAN, f64::INFINITY]; 8]).unwrap();
    assert_eq!(m.to_json().unwrap(), before);
    assert_eq!(
        p.fit_training(&x[..24].to_vec(), &y[..24], ScalarCarrier::Row, 0.01)
            .unwrap()
            .to_json()
            .unwrap(),
        before
    );
}

#[test]
fn model_envelope_rejects_future_features_and_forged_state() {
    let (x, y) = rows();
    let p = scalar(21, Var::Col(0), NumericalPolicy::StableV2);
    let m = p.fit_training(&x, &y, ScalarCarrier::Row, 0.01).unwrap();
    let original: Value = serde_json::from_str(&m.to_json().unwrap()).unwrap();
    for (field, new_value) in [
        ("schema", json!("feat-math.future-model")),
        ("format_version", json!(3)),
        ("min_reader_version", json!(256)),
        ("required_features", json!(14 | (1u64 << 40))),
        ("required_features", json!(6)),
        ("numerical_policy", json!("v1")),
        ("profile", json!("unqualified")),
        ("grammar_sha256", json!("0".repeat(64))),
        ("emitter_feature_identity", json!("unknown")),
        ("output_schema", json!(["prediction:i64"])),
    ] {
        let mut wire = original.clone();
        wire[field] = new_value;
        assert!(
            matches!(
                ScalarModel::from_json(&wire.to_string()),
                Err(VersionError::UnsupportedVersion(_))
            ),
            "field {field}"
        );
    }
    for (field, new_value) in [
        ("input_width", json!(0)),
        ("input_width", json!(513)),
        ("lambda_bits", json!(f64::NAN.to_bits())),
        ("ridge_bits", json!([[], [], []])),
        ("ridge_bits", json!([[0u64], [0u64], [0u64, 0u64]])),
        (
            "ridge_bits",
            json!([
                [0u64],
                [1.0f64.to_bits()],
                [f64::MAX.to_bits(), f64::MAX.to_bits()]
            ]),
        ),
        (
            "ridge_bits",
            json!([[0u64], [1.0f64.to_bits()], [f64::NAN.to_bits(), 0u64]]),
        ),
    ] {
        let mut wire = original.clone();
        wire[field] = new_value;
        assert!(
            ScalarModel::from_json(&wire.to_string()).is_err(),
            "field {field}"
        );
    }
    let mut wire = original.clone();
    wire["unrecognized"] = json!(true);
    assert!(ScalarModel::from_json(&wire.to_string()).is_err());
    let mut future = original.clone();
    future["required_features"] = json!(14 | (1u64 << 40));
    future["program"] = json!([0]);
    future.as_object_mut().unwrap().remove("ridge_bits");
    assert!(matches!(
        ScalarModel::from_json(&future.to_string()),
        Err(VersionError::UnsupportedVersion(_))
    ));
    let mut wire = original.clone();
    wire["program"][7] = json!(3);
    assert!(matches!(
        ScalarModel::from_json(&wire.to_string()),
        Err(VersionError::UnsupportedVersion(_))
    ));
    let mut wire = original;
    wire["program"] = json!([0]);
    assert!(ScalarModel::from_json(&wire.to_string()).is_err());
    assert!(ScalarModel::from_json("{}").is_err());
}
