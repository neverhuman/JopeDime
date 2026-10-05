use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarLanguage, ScalarModel,
    ScalarNode, ScalarProducerKind, ScalarProducerProfile,
};
use feat_math::math_expr::Var;
use feat_math::tokens::{count_tokens, Lang};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn rows(value: &Value) -> Vec<Vec<f64>> {
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
}

#[test]
fn admitted_fits_bind_all_artifacts_without_changing_native_state_or_source_core() {
    let pins: Value = serde_json::from_str(include_str!(
        "../fixtures/scalar_v2/native_model_format2.json"
    ))
    .unwrap();
    let training = rows(&pins["rows_bits"]);
    let labels: Vec<_> = pins["labels_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f64::from_bits(v.as_u64().unwrap()))
        .collect();
    let mut cases = Vec::new();
    for record in pins["records"].as_array().unwrap() {
        let native = ScalarModel::from_json(record["model_json"].as_str().unwrap()).unwrap();
        cases.push((native, record["producer_admissible"].as_bool().unwrap()));
    }
    // Logit is additional to the retained 24-case model fixture.
    for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
        for policy in [
            feat_math::kernel::numerical_policy::NumericalPolicy::LegacyV1,
            feat_math::kernel::numerical_policy::NumericalPolicy::StableV2,
        ] {
            let argument = if carrier == ScalarCarrier::Row {
                Var::Col(0)
            } else {
                Var::X
            };
            let program = ScalarExpressionProgram::new(ScalarNode::Call(
                ScalarFunction::from_versioned_id(2, 20).unwrap(),
                vec![ScalarNode::Var(argument)],
            ))
            .unwrap()
            .with_numerical_policy(policy);
            cases.push((
                program
                    .fit_training(&training, &labels, carrier, 0.02)
                    .unwrap(),
                true,
            ));
        }
    }
    let mut admitted = 0;
    let mut refused = 0;
    for (native, admissible) in cases {
        let profile = ScalarProducerProfile::new(
            ScalarProducerKind::All,
            native.carrier(),
            native.input_width(),
        )
        .unwrap();
        let result = profile.fit_training(native.program(), &training, &labels, native.lambda());
        if !admissible {
            assert!(result.is_err());
            refused += 1;
            continue;
        }
        let bound = result.unwrap();
        admitted += 1;
        assert_eq!(bound.producer_profile().unwrap().sha256(), profile.sha256());
        let state = bound.to_json().unwrap();
        let mut unbound_state: Value = serde_json::from_str(&state).unwrap();
        unbound_state.as_object_mut().unwrap().remove("producer");
        unbound_state["format_version"] = json!(2);
        unbound_state["min_reader_version"] = json!(2);
        assert_eq!(
            unbound_state,
            serde_json::from_str::<Value>(&native.to_json().unwrap()).unwrap()
        );
        let restored = ScalarModel::from_json(&state).unwrap();
        assert_eq!(restored.to_json().unwrap(), state);
        for probe in [
            &training,
            &vec![
                vec![f64::NAN, f64::INFINITY],
                vec![-0.0, 0.0],
                vec![1e200, -1e200],
            ],
        ] {
            let (expected, faults) = native.predict(probe).unwrap();
            let (actual, actual_faults) = restored.predict(probe).unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(actual_faults, faults);
        }
        let (recipes, cpp) = bound.emit_recipes().unwrap();
        let manifest: Value = serde_json::from_str(&recipes.manifest).unwrap();
        assert_eq!(manifest["format_version"], 3);
        assert_eq!(manifest["min_reader_version"], 3);
        assert_eq!(
            manifest["producer"],
            json!({"declaration": profile.canonical_json(), "sha256": profile.sha256()})
        );
        assert_eq!(manifest["model_sha256"], bound.identity().unwrap());
        for (language, token_language, source, source_key, token_key) in [
            (
                ScalarLanguage::Rust,
                Lang::Rust,
                &recipes.rust,
                "rs_sha256",
                "rs_tokens",
            ),
            (
                ScalarLanguage::Python,
                Lang::Python,
                &recipes.python,
                "py_sha256",
                "py_tokens",
            ),
            (
                ScalarLanguage::Cpp17,
                Lang::Cpp,
                &cpp.source,
                "cpp17_sha256",
                "cpp17_tokens",
            ),
        ] {
            let marker = if language == ScalarLanguage::Python {
                "#"
            } else {
                "//"
            };
            let header = format!(
                "{marker} scalar producer-profile format=3 min-reader=3 sha256={}\n",
                profile.sha256()
            );
            assert_eq!(*source, header + &native.emit_recipe(language).unwrap());
            assert_eq!(*source, restored.emit_recipe(language).unwrap());
            assert_eq!(manifest[source_key], hash(source.as_bytes()));
            assert_eq!(
                manifest[token_key],
                serde_json::to_value(count_tokens(source, token_language)).unwrap()
            );
        }
        assert_eq!(cpp.source_sha256, hash(cpp.source.as_bytes()));
        assert_eq!(
            recipes.timing.py_tokens,
            recipes.py_tokens.code_tokens + recipes.py_tokens.doc_tokens
        );
        assert_eq!(
            recipes.timing.rs_tokens,
            recipes.rs_tokens.code_tokens + recipes.rs_tokens.doc_tokens
        );
        assert!(recipes.timing.emit_ns > 0 && recipes.timing.tokens_per_sec.is_finite());
        let fitted = bound.emit_fitted_artifact().unwrap();
        let native_fitted = native.emit_fitted_artifact().unwrap();
        assert_eq!(
            fitted.source,
            native_fitted.source.replacen(
                &native.identity().unwrap(),
                &bound.identity().unwrap(),
                1
            )
        );
        assert_eq!(fitted.codec, state);
        assert_eq!(fitted.codec_sha256, hash(fitted.codec.as_bytes()));
        assert_eq!(fitted.source_sha256, hash(fitted.source.as_bytes()));
        assert_eq!(bound.to_json().unwrap(), state);
        assert_eq!(restored.to_json().unwrap(), state);
        let (unbound_recipes, unbound_cpp) = native.emit_recipes().unwrap();
        let (program_recipes, program_cpp) = native
            .program()
            .emit_recipes(native.carrier(), native.input_width(), native.lambda())
            .unwrap();
        assert_eq!(unbound_recipes.python, program_recipes.python);
        assert_eq!(unbound_recipes.rust, program_recipes.rust);
        assert_eq!(unbound_recipes.manifest, program_recipes.manifest);
        assert_eq!(unbound_cpp.source, program_cpp.source);
        assert_eq!(unbound_cpp.source_sha256, program_cpp.source_sha256);
    }
    assert_eq!(admitted, 26);
    assert_eq!(refused, 2);
}

#[test]
fn producer_fit_refuses_bad_scope_and_native_fit_configuration() {
    let profile =
        ScalarProducerProfile::new(ScalarProducerKind::All, ScalarCarrier::Row, 2).unwrap();
    let program = ScalarExpressionProgram::new(ScalarNode::Var(Var::Col(0))).unwrap();
    let training = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    for lambda in [f64::NAN, f64::INFINITY, -0.0, 0.0, 2.0001] {
        assert!(profile
            .fit_training(&program, &training, &[1.0, 2.0], lambda)
            .is_err());
    }
    for labels in [&[1.0][..], &[1.0, f64::NAN], &[1.0, 1e13]] {
        assert!(profile
            .fit_training(&program, &training, labels, 0.02)
            .is_err());
    }
    assert!(profile.fit_training(&program, &vec![], &[], 0.02).is_err());
    assert!(profile
        .fit_training(&program, &vec![vec![1.0]], &[1.0], 0.02)
        .is_err());
    let wrong_scope = ScalarExpressionProgram::new(ScalarNode::Var(Var::X)).unwrap();
    assert!(profile
        .fit_training(&wrong_scope, &training, &[1.0, 2.0], 0.02)
        .is_err());
    let constant = ScalarExpressionProgram::new(ScalarNode::Const(1.0)).unwrap();
    assert!(profile
        .fit_training(&constant, &training, &[1.0, 2.0], 0.02)
        .is_err());
}
