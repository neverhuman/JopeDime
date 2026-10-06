use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarModel, ScalarNode, ScalarProducerKind,
    ScalarProducerProfile,
};
use feat_math::kernel::wire_v2::{VersionError, VersionedModel};
use feat_math::math_expr::{BinOp, Var};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

// Preserve declaration order and change only the field under test.
fn forge_canonical_field(text: &str, key: &str, value: &Value) -> String {
    let declaration: Value = serde_json::from_str(text).unwrap();
    let original = declaration.get(key).unwrap();
    let key = serde_json::to_string(key).unwrap();
    let original = format!("{key}:{}", serde_json::to_string(original).unwrap());
    let replacement = format!("{key}:{}", serde_json::to_string(value).unwrap());
    assert_eq!(text.matches(&original).count(), 1);
    text.replacen(&original, &replacement, 1)
}

fn pins() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/scalar_v2/native_model_format2.json"
    ))
    .unwrap()
}

fn rows(pins: &Value) -> Vec<Vec<f64>> {
    pins["rows_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|cell| f64::from_bits(cell.as_u64().unwrap()))
                .collect()
        })
        .collect()
}

fn bound(record: &Value, profile: &ScalarProducerProfile) -> Value {
    let mut model: Value = serde_json::from_str(record["model_json"].as_str().unwrap()).unwrap();
    model["format_version"] = 3.into();
    model["min_reader_version"] = 3.into();
    model["producer"] = json!({"declaration":profile.canonical_json(),"sha256":profile.sha256()});
    model
}

#[test]
fn native_format_two_json_and_prediction_bits_remain_frozen() {
    let pins = pins();
    let rows = rows(&pins);
    let labels: Vec<_> = pins["labels_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f64::from_bits(v.as_u64().unwrap()))
        .collect();
    assert_eq!(pins["records"].as_array().unwrap().len(), 24);
    for record in pins["records"].as_array().unwrap() {
        let bytes: Vec<u8> = serde_json::from_value(record["program"].clone()).unwrap();
        let program = ScalarExpressionProgram::from_bytes(&bytes).unwrap();
        let carrier = serde_json::from_value(record["carrier"].clone()).unwrap();
        let lambda = f64::from_bits(record["lambda_bits"].as_u64().unwrap());
        let fresh = program
            .fit_training(&rows, &labels, carrier, lambda)
            .unwrap();
        let expected = record["model_json"].as_str().unwrap();
        assert_eq!(fresh.to_json().unwrap(), expected, "{}", record["name"]);
        let restored = ScalarModel::from_json(expected).unwrap();
        assert!(fresh.producer_profile().is_none());
        assert!(restored.producer_profile().is_none());
        assert_eq!(restored.to_json().unwrap(), expected);
        let bits: Vec<_> = restored
            .predict(&rows)
            .unwrap()
            .0
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(json!(bits), record["prediction_bits"]);
    }
}

#[test]
fn bound_models_replay_exact_state_and_do_not_fit_on_prediction() {
    let pins = pins();
    let rows = rows(&pins);
    for record in pins["records"].as_array().unwrap() {
        let carrier = serde_json::from_value(record["carrier"].clone()).unwrap();
        let profile = ScalarProducerProfile::new(ScalarProducerKind::All, carrier, 2).unwrap();
        let candidate = bound(record, &profile);
        let restored = ScalarModel::from_json(&candidate.to_string());
        if record["producer_admissible"] == false {
            assert!(restored.is_err());
            assert!(VersionedModel::from_json(&candidate.to_string()).is_err());
            continue;
        }
        let restored = restored.unwrap();
        assert_eq!(
            restored.producer_profile().unwrap().sha256(),
            profile.sha256()
        );
        let state = restored.to_json().unwrap();
        assert_eq!(
            ScalarModel::from_json(&state).unwrap().to_json().unwrap(),
            state
        );
        let native = ScalarModel::from_json(record["model_json"].as_str().unwrap()).unwrap();
        // The generic route must recognize the scalar schema before applying
        // other families' format-2 header checks to this format-3 envelope.
        let generic = VersionedModel::from_json(&state).unwrap();
        assert!(matches!(generic, VersionedModel::ScalarV2(_)));
        assert_eq!(
            generic.predict_scalar(&rows).unwrap(),
            native.predict(&rows).unwrap()
        );
        assert_eq!(
            restored.predict(&rows).unwrap(),
            native.predict(&rows).unwrap()
        );
        let poison = vec![
            vec![f64::NAN, f64::INFINITY],
            vec![1e200, -1e200],
            vec![-0.0, 0.0],
        ];
        assert_eq!(
            restored.predict(&poison).unwrap(),
            native.predict(&poison).unwrap()
        );
        assert_eq!(
            generic.predict_scalar(&poison).unwrap(),
            native.predict(&poison).unwrap()
        );
        assert_eq!(generic.predict_scalar(&vec![]).unwrap(), (vec![], vec![]));
        assert_eq!(restored.to_json().unwrap(), state);
        assert_eq!(restored.predict(&vec![]).unwrap(), (vec![], vec![]));
        // Generation/edit switches retain already typed program admission.
        let disabled = bound(record, &profile.generation_disabled());
        assert_eq!(
            ScalarModel::from_json(&disabled.to_string())
                .unwrap()
                .predict(&rows)
                .unwrap(),
            native.predict(&rows).unwrap()
        );
    }
}

#[test]
fn future_producer_identity_refuses_before_model_payload() {
    let pins = pins();
    let record = &pins["records"][2];
    let profile =
        ScalarProducerProfile::new(ScalarProducerKind::All, ScalarCarrier::Row, 2).unwrap();
    let original = bound(record, &profile);
    let declaration = profile.canonical_json();
    assert_eq!(
        forge_canonical_field(&declaration, "revision", &json!(1)),
        declaration
    );
    // Well-typed changes with a matching hash must refuse because the identity
    // is unsupported, before the missing model payload is interpreted.
    for (key, value) in [
        ("revision", json!(2)),
        ("opt_in_numerator", json!(2)),
        ("birth_stream", json!(17u64)),
        ("function_catalog", json!([999])),
        ("reduction_catalog", json!(["unknown"])),
    ] {
        let mut model = original.clone();
        let forged = forge_canonical_field(&declaration, key, &value);
        assert_ne!(forged, declaration);
        model["producer"]["sha256"] = format!("{:x}", Sha256::digest(forged.as_bytes())).into();
        model["producer"]["declaration"] = forged.into();
        model.as_object_mut().unwrap().remove("program");
        model.as_object_mut().unwrap().remove("ridge_bits");
        assert!(
            matches!(
                ScalarModel::from_json(&model.to_string()),
                Err(VersionError::UnsupportedVersion(_))
            ),
            "{key}"
        );
        assert!(
            matches!(
                VersionedModel::from_json(&model.to_string()),
                Err(VersionError::UnsupportedVersion(_))
            ),
            "generic {key}"
        );
    }
    for (key, value) in [
        ("format_version", json!(4)),
        ("min_reader_version", json!(4)),
        ("required_features", json!(1u64 << 63)),
    ] {
        let mut model = original.clone();
        model[key] = value;
        model.as_object_mut().unwrap().remove("program");
        assert!(matches!(
            ScalarModel::from_json(&model.to_string()),
            Err(VersionError::UnsupportedVersion(_))
        ));
    }
    let mut bad_hash = original.clone();
    bad_hash["producer"]["sha256"] = "unknown".into();
    bad_hash.as_object_mut().unwrap().remove("program");
    assert!(matches!(
        ScalarModel::from_json(&bad_hash.to_string()),
        Err(VersionError::UnsupportedVersion(_))
    ));
    for (format, producer) in [(2, original["producer"].clone()), (3, Value::Null)] {
        let mut model = original.clone();
        model["format_version"] = format.into();
        model["min_reader_version"] = format.into();
        model["producer"] = producer;
        assert!(ScalarModel::from_json(&model.to_string()).is_err());
    }
}

#[test]
fn producer_scope_caps_catalog_and_noncanonical_nodes_refuse() {
    let pins = pins();
    let record = &pins["records"][2];
    let all = ScalarProducerProfile::new(ScalarProducerKind::All, ScalarCarrier::Row, 2).unwrap();
    let original = bound(record, &all);
    for profile in [
        ScalarProducerProfile::new(ScalarProducerKind::All, ScalarCarrier::Row, 3).unwrap(),
        ScalarProducerProfile::new(ScalarProducerKind::All, ScalarCarrier::Elementwise, 2).unwrap(),
    ] {
        assert!(ScalarModel::from_json(&bound(record, &profile).to_string()).is_err());
    }
    for node in [
        ScalarNode::Const(1.0),
        ScalarNode::Var(Var::Col(8)),
        ScalarNode::Binary(
            BinOp::Add,
            Box::new(ScalarNode::Var(Var::Col(0))),
            Box::new(ScalarNode::Const(-1.0)),
        ),
        ScalarNode::Binary(
            BinOp::Add,
            Box::new(ScalarNode::Var(Var::Col(0))),
            Box::new(ScalarNode::Const(-0.0)),
        ),
    ] {
        let program = ScalarExpressionProgram::new(node).unwrap();
        let mut model = original.clone();
        model["program"] = json!(program.to_bytes().unwrap());
        model["required_features"] = json!(program.required_features().unwrap());
        model["numerical_policy"] = program.numerical_policy().name().into();
        assert!(ScalarModel::from_json(&model.to_string()).is_err());
    }
    let inverse_only =
        ScalarProducerProfile::new(ScalarProducerKind::InverseLinks, ScalarCarrier::Row, 2)
            .unwrap();
    let exponential = &pins["records"][4];
    assert!(ScalarModel::from_json(&bound(exponential, &inverse_only).to_string()).is_err());
    let mut no_policy = bound(record, &inverse_only);
    let program = ScalarExpressionProgram::new(ScalarNode::Var(Var::Col(0)))
        .unwrap()
        .with_numerical_policy(NumericalPolicy::StableV2);
    no_policy["program"] = json!(program.to_bytes().unwrap());
    no_policy["required_features"] = json!(program.required_features().unwrap());
    no_policy["numerical_policy"] = "v2".into();
    assert!(ScalarModel::from_json(&no_policy.to_string()).is_err());
}

#[test]
fn native_identities_refuse_before_bad_payload_in_bound_and_unbound_models() {
    let pins = pins();
    let record = &pins["records"][2];
    let profile =
        ScalarProducerProfile::new(ScalarProducerKind::All, ScalarCarrier::Row, 2).unwrap();
    let native: Value = serde_json::from_str(record["model_json"].as_str().unwrap()).unwrap();
    for original in [native, bound(record, &profile)] {
        assert!(ScalarModel::from_json(&original.to_string()).is_ok());
        assert!(VersionedModel::from_json(&original.to_string()).is_ok());
        for payload in 0..3 {
            let mut invalid = original.clone();
            match payload {
                0 => {
                    invalid.as_object_mut().unwrap().remove("program");
                    invalid.as_object_mut().unwrap().remove("ridge_bits");
                }
                1 => invalid["program"] = json!([255]),
                _ => invalid["program"] = json!("unknown-payload"),
            }
            // The supported-header control must reach the malformed payload.
            assert!(matches!(
                ScalarModel::from_json(&invalid.to_string()),
                Err(VersionError::Malformed(_))
            ));
            assert!(matches!(
                VersionedModel::from_json(&invalid.to_string()),
                Err(VersionError::Malformed(_))
            ));
            for (name, unsupported) in [
                ("profile", json!("future-scalar-profile")),
                (
                    "grammar_sha256",
                    json!("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
                ),
                ("emitter_feature_identity", json!("future-scalar-emitter")),
                ("output_schema", json!(["prediction:future"])),
            ] {
                let mut forged = invalid.clone();
                assert_ne!(forged[name], unsupported);
                forged[name] = unsupported;
                assert!(
                    matches!(
                        ScalarModel::from_json(&forged.to_string()),
                        Err(VersionError::UnsupportedVersion(_))
                    ),
                    "direct {name} payload {payload}"
                );
                assert!(
                    matches!(
                        VersionedModel::from_json(&forged.to_string()),
                        Err(VersionError::UnsupportedVersion(_))
                    ),
                    "generic {name} payload {payload}"
                );
            }
        }
    }
}
