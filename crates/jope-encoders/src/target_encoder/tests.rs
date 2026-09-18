use super::*;

#[test]
fn safe_unit_matches_oracle_nan_to_num_contract() {
    let output = safe_unit(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3.0, 4.0]);
    assert_eq!(output, [0.0, 0.0, 0.0, 0.6, 0.8]);
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn numpy_seed_sequence_and_pcg64_match_frozen_stream() {
    assert_eq!(
        seed_sequence(42),
        [
            11_465_652_750_463_011_511,
            15_382_171_918_060_459_190,
            9_018_504_550_953_525_431,
            3_703_499_796_004_394_495,
        ]
    );
    let mut rng = NumpyPcg64::new(42);
    assert_eq!(rng.next_u64(), 14_276_969_152_011_380_360);
    assert_eq!(rng.next_u64(), 8_095_878_257_575_067_585);
}

#[test]
fn numpy_sampling_matches_frozen_oracle() {
    let seed = stable_seed(&[
        "fusion-signature-inference".into(),
        "17".into(),
        "1".into(),
        "40".into(),
        "0".into(),
        "32".into(),
    ])
    .unwrap();
    assert_eq!(seed, 1_865_186_800_557_055_428);
    assert_eq!(
        NumpyPcg64::new(seed).choice(40, 32, false).unwrap(),
        vec![
            26, 1, 34, 16, 4, 32, 9, 13, 30, 36, 8, 14, 33, 25, 2, 39, 23, 6, 29, 31, 17, 12, 0, 5,
            10, 15, 37, 22, 35, 3, 27, 18,
        ]
    );
    let seed = stable_seed(&[
        "encode".into(),
        "17".into(),
        "40".into(),
        "32".into(),
        "32".into(),
    ])
    .unwrap();
    assert_eq!(
        NumpyPcg64::new(seed).permutation(40),
        vec![
            23, 28, 19, 25, 30, 15, 24, 11, 21, 39, 10, 17, 1, 20, 29, 32, 22, 31, 9, 6, 18, 35,
            26, 4, 0, 2, 34, 37, 33, 14, 27, 12, 36, 38, 7, 5, 13, 16, 8, 3,
        ]
    );
}

#[test]
fn short_targets_bind_effective_count_and_replacement_fraction() {
    let stats = neural_explicit_stats(&[0.25; WINDOW_SIZE], TargetTask::Regression, 7);
    assert_eq!(stats[3], 7.0f32.ln_1p() / 16.0);
    assert_eq!(stats[4], 1.0 - 7.0 / WINDOW_SIZE as f32);
}

#[test]
fn classical_signature_matches_frozen_binary_and_regression_oracle() {
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/target_encoder_fused_v1_parity.json");
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_path).unwrap()).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let task = case["input"]["task"].as_str().unwrap().parse().unwrap();
        let values = case["input"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let expected = case["raw_signature"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let actual = raw_signature(&preprocess(&values, task).unwrap(), task, 17).unwrap();
        let difference = actual
            .iter()
            .zip(&expected)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max);
        assert!(
            difference <= 1e-5,
            "{task:?} signature difference {difference}"
        );
    }
}
