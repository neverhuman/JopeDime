use super::*;

#[test]
fn runtime_generated_prime_lime_component_differential() -> Result<()> {
    let large = generated_rows(1005);
    let reordered = large[..40].iter().rev().cloned().collect::<Vec<_>>();
    let binary = binary_edge_values();
    let all_nonfinite = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let request = OracleRequest {
        target_bundle: TARGET_BUNDLE,
        choice_populations: vec![1001, 1500, 10_000, 50_000],
        external_cases: vec![
            ExternalInput {
                name: "subsample_1005".to_owned(),
                rows_bits: large
                    .iter()
                    .map(|row| row.iter().map(|value| value.to_bits()).collect())
                    .collect(),
            },
            ExternalInput {
                name: "reordered_40".to_owned(),
                rows_bits: reordered
                    .iter()
                    .map(|row| row.iter().map(|value| value.to_bits()).collect())
                    .collect(),
            },
        ],
        target_cases: vec![
            TargetInput {
                name: "binary_threshold_repeated_nonfinite".to_owned(),
                task: "binary".to_owned(),
                values_bits: binary.iter().map(|value| value.to_bits()).collect(),
            },
            TargetInput {
                name: "all_nonfinite_rejected".to_owned(),
                task: "regression".to_owned(),
                values_bits: all_nonfinite.iter().map(|value| value.to_bits()).collect(),
            },
        ],
    };
    let oracle = frozen_oracle(&request)?;
    assert_eq!(oracle.versions.python, "3.12.3");
    assert_eq!(oracle.versions.numpy, "1.26.4");
    assert_eq!(oracle.versions.torch, "2.7.0");
    assert_eq!(oracle.versions.sklearn, "1.7.2");
    assert_eq!(oracle.versions.joblib, "1.3.2");
    assert_eq!(oracle.modules.len(), 5);
    assert_eq!(
        oracle.modules.get("external").map(String::as_str),
        Some("jope-prime/scripts/encode_external.py")
    );
    assert_eq!(
        oracle.modules.get("foundation").map(String::as_str),
        Some("jope-prime/releases/76aa503df60b405afd5aff049adce1c6f7e8819ea812981236d900bdb36ed6e5/src/jope/encoders/foundation.py")
    );
    assert_eq!(
        oracle.modules.get("fusion_experts").map(String::as_str),
        Some("remote_super/fable_god/chimera_regression/python/fusion/experts.py")
    );
    assert_eq!(
        oracle.modules.get("model_io").map(String::as_str),
        Some("remote_super/fable_god/python/elo_finetune/model_io.py")
    );
    assert_eq!(
        oracle.modules.get("target_fused").map(String::as_str),
        Some("target_maker/target_maker/fused.py")
    );
    assert_eq!(
        oracle
            .choices
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["10000", "1001", "1500", "50000"]
    );
    assert_eq!(
        oracle
            .external_cases
            .iter()
            .map(|case| case.name.as_str())
            .collect::<Vec<_>>(),
        vec!["subsample_1005", "reordered_40"]
    );
    assert_eq!(
        oracle
            .target_cases
            .iter()
            .map(|case| case.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "binary_threshold_repeated_nonfinite",
            "all_nonfinite_rejected"
        ]
    );
    for population in &request.choice_populations {
        let expected = &oracle.choices[&population.to_string()];
        let actual = jope_external_selected_indices(*population)?;
        assert_eq!(
            actual, *expected,
            "NumPy PCG64 order-preserving choice mismatch for n={population}"
        );
        let mut bytes = Vec::with_capacity(actual.len() * 8);
        for index in &actual {
            bytes.extend((*index as u64).to_le_bytes());
        }
        let expected_digest = CHOICE_REGRESSION_SHA256
            .iter()
            .find_map(|(candidate, digest)| (candidate == population).then_some(*digest))
            .context("missing fixed subset regression digest")?;
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected_digest);
    }

    let raw_cases = BTreeMap::from([("subsample_1005", large), ("reordered_40", reordered)]);
    let mut native_cases = BTreeMap::<String, JopeExternalPreprocessedV1>::new();
    for expected in &oracle.external_cases {
        let raw = &raw_cases[expected.name.as_str()];
        let native = preprocess_jope_external_train(raw)?;
        assert_eq!(native.schema, "jain.jope-external-preprocessing/v1");
        assert_eq!(native.csv_float_format, ".9g");
        assert_eq!(native.selected_indices, expected.selected_indices);
        assert!(native
            .selected_indices
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
        compare_f64(
            &format!("{}.external.low", expected.name),
            &native.scaler.low,
            &expected.scaler_low,
            0.0,
            0.0,
        )?;
        compare_f64(
            &format!("{}.external.span", expected.name),
            &native.scaler.span,
            &expected.scaler_span,
            0.0,
            0.0,
        )?;
        compare_f64(
            &format!("{}.external.encoding_rows", expected.name),
            &flat_f64(&native.encoding_rows),
            &flat_f64(&expected.encoding_rows),
            0.0,
            0.0,
        )?;
        let replayed = native.scaler.transform_for_encoding(raw)?;
        compare_f64(
            &format!("{}.external.scaler_replay", expected.name),
            &flat_f64(&replayed),
            &flat_f64(&expected.all_encoding_rows),
            0.0,
            0.0,
        )?;
        if expected.name == "subsample_1005" {
            assert_eq!(native.selected_indices.len(), 1000);
            assert_eq!(native.scaler.span[0], 1.0, "constant-column span");
            assert!(native.encoding_rows.iter().all(|row| row[0] == 0.0));
            let persisted = serde_json::to_vec(&native)?;
            let restored: JopeExternalPreprocessedV1 = serde_json::from_slice(&persisted)?;
            assert_eq!(restored.selected_indices, native.selected_indices);
            assert_eq!(restored.scaler.low, native.scaler.low);
            assert_eq!(restored.scaler.span, native.scaler.span);
        }
        native_cases.insert(expected.name.clone(), native);
    }

    let Some(root) =
        models_root::bundle_root("runtime_generated_prime_lime_component_differential")
    else {
        return Ok(());
    };
    let bundle = LoadedModelBundle::load(&root)?;
    {
        let encoder = StarlightFoundationEncoder::load_bundle(&bundle, Device::Cpu, DType::F32)?;
        for expected in &oracle.external_cases {
            let external = &native_cases[&expected.name];
            let prepared = prepare_jope_foundation_rows(&external.encoding_rows)?;
            assert_eq!(prepared.schema, "jain.jope-foundation-preprocessing/v1");
            assert_eq!(prepared.context_indices, expected.context_indices);
            assert_eq!(prepared.query_indices, expected.query_indices);
            compare_f64(
                &format!("{}.foundation.low", expected.name),
                &prepared.feature_low,
                &expected.feature_low,
                0.0,
                0.0,
            )?;
            compare_f64(
                &format!("{}.foundation.span", expected.name),
                &prepared.feature_span,
                &expected.feature_span,
                0.0,
                0.0,
            )?;
            compare(
                &format!("{}.foundation.context_x", expected.name),
                &flat_f32(&prepared.input.context_x),
                &flat_f32(&expected.context_x),
                0.0,
                0.0,
            )?;
            compare(
                &format!("{}.foundation.context_y", expected.name),
                &prepared.input.context_y,
                &expected.context_y,
                0.0,
                0.0,
            )?;
            compare(
                &format!("{}.foundation.query_x", expected.name),
                &flat_f32(&prepared.input.query_x),
                &flat_f32(&expected.query_x),
                0.0,
                0.0,
            )?;
            let actual = encoder.encode_with_intermediates(
                &prepared.input,
                FoundationEncodeOptions {
                    max_context_rows: JOPE_FOUNDATION_CONTEXT_ROWS,
                    max_query_rows: JOPE_FOUNDATION_QUERY_ROWS,
                    seed: JOPE_FOUNDATION_SEED,
                },
            )?;
            compare(
                &format!("{}.foundation.query_hidden", expected.name),
                &actual.query_hidden,
                &flat_f32(&expected.query_hidden),
                FOUNDATION_HIDDEN_MAX_ABS,
                FOUNDATION_HIDDEN_MAX_REL,
            )?;
            for (name, rust, python) in [
                ("mean", &actual.mean, &expected.mean),
                ("std", &actual.std, &expected.std),
                ("p10", &actual.p10, &expected.p10),
                ("p90", &actual.p90, &expected.p90),
                ("vector", &actual.vector, &expected.vector),
            ] {
                compare(
                    &format!("{}.foundation.{name}", expected.name),
                    rust,
                    python,
                    FOUNDATION_POOLED_MAX_ABS,
                    FOUNDATION_POOLED_MAX_REL,
                )?;
            }
        }
    }

    let target_encoder = FusedTargetEncoder::load(&bundle)?;
    for expected in &oracle.external_cases {
        let values = native_cases[&expected.name].target_values()?;
        let actual = target_encoder.encode(&values, TargetTask::Regression)?;
        compare_target(&expected.target.name, &actual, &expected.target)?;
    }
    for expected in &oracle.target_cases {
        let task = expected.task.parse::<TargetTask>()?;
        let input = request
            .target_cases
            .iter()
            .find(|case| case.name == expected.name)
            .context("Python returned an unknown target case")?;
        let values = input
            .values_bits
            .iter()
            .map(|bits| f32::from_bits(*bits))
            .collect::<Vec<_>>();
        if expected.name == "all_nonfinite_rejected" {
            assert!(expected.error.as_deref().is_some_and(|error| {
                error.contains("ValueError") && error.contains("empty target")
            }));
            assert!(target_encoder.encode(&values, task).is_err());
        } else {
            let actual = target_encoder.encode(&values, task)?;
            compare_target(&expected.name, &actual, expected)?;
        }
    }
    Ok(())
}
