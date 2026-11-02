#[test]
fn scalar_recipe_and_fitted_sidecars_bind_the_executed_program_policy_and_state() {
    use feat_math::kernel::wire_v2::VersionedProgram;
    use sha2::{Digest, Sha256};
    let hash = |value: &[u8]| format!("{:x}", Sha256::digest(value));
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        let p = ScalarExpressionProgram::new(ScalarNode::Call(
            ScalarFunction::from_versioned_id(2, 13).unwrap(),
            vec![ScalarNode::Var(Var::Col(0))],
        ))
        .unwrap()
        .with_numerical_policy(policy);
        let (paired, cpp) = p.emit_recipes(ScalarCarrier::Row, 1, 0.02).unwrap();
        let wire: Value = serde_json::from_str(&paired.manifest).unwrap();
        let bytes: Vec<u8> = serde_json::from_value(wire["program"].clone()).unwrap();
        assert_eq!(bytes, p.to_bytes().unwrap());
        assert_eq!(wire["program_sha256"], hash(&bytes));
        assert_eq!(
            VersionedProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        assert_eq!(wire["min_reader_version"], 2);
        assert_eq!(wire["numerical_policy"], policy.name());
        assert_eq!(
            wire["emitter_feature_identity"],
            "feat-math.scalar.ridge.recipe.v2"
        );
        assert_eq!(
            wire["profile"],
            feat_math::kernel::scalar_program::MODEL_PROFILE
        );
        for (field, source) in [
            ("py_sha256", paired.python.as_str()),
            ("rs_sha256", paired.rust.as_str()),
            ("cpp17_sha256", cpp.source.as_str()),
        ] {
            assert_eq!(wire[field], hash(source.as_bytes()));
            assert!(source.contains("feat-math.scalar.ridge.recipe.v2"));
            assert!(source.contains(&format!("policy={}", policy.name())));
        }
        assert_eq!(cpp.tokens.bytes, cpp.source.len());
        assert_eq!(paired.py_tokens.bytes, paired.python.len());
        assert_eq!(paired.rs_tokens.bytes, paired.rust.len());
        let second = p.emit_recipes(ScalarCarrier::Row, 1, 0.02).unwrap();
        assert_eq!(paired.manifest, second.0.manifest);
        let model = p
            .fit_training(
                &vec![vec![-1.0], vec![1.0]],
                &[-2.0, 2.0],
                ScalarCarrier::Row,
                0.02,
            )
            .unwrap();
        let artifact = model.emit_fitted_artifact().unwrap();
        assert_eq!(artifact.codec, model.to_json().unwrap());
        assert_eq!(artifact.source_sha256, hash(artifact.source.as_bytes()));
        assert_eq!(artifact.codec_sha256, hash(artifact.codec.as_bytes()));
        assert!(artifact.source.contains(&model.identity().unwrap()));
        assert_eq!(artifact.source, model.emit_fitted_cpp17().unwrap());
        let restored =
            feat_math::kernel::scalar_program::ScalarModel::from_json(&artifact.codec).unwrap();
        assert_eq!(restored.emit_fitted_cpp17().unwrap(), artifact.source);
    }
}
