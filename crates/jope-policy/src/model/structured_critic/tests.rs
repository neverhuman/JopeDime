use super::*;

use super::*;

fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn candidate(
    parameters: CriticParameters,
    calibration: Option<OutcomeCalibration>,
) -> StructuredRankV1 {
    let manifest = CriticManifest {
        schema: STRUCTURED_RANK_SCHEMA.into(),
        candidate_id: "structural-rank-test-v1".into(),
        actor_checkpoint_sha256: digest(1),
        same_pool_sha256: digest(2),
        critic_weights_sha256: parameters.sha256(),
        vocab_size: 64,
        token_order: "declared_native_token_order_v1".into(),
        feature_schema: "unigram-directed-bigram-128-v1".into(),
        normalizer: "sqrt_length_l2_v1".into(),
        max_tokens: 128,
        width: WIDTH,
        layers: LAYERS,
        backend: "candle_cpu".into(),
        precision: "f32".into(),
        calibration,
    };
    StructuredRankV1::new(manifest, parameters).unwrap()
}

fn order_sensitive_parameters() -> CriticParameters {
    let mut p = CriticParameters {
        first_weights: vec![0.0; WIDTH * FEATURE_DIM],
        first_bias: vec![0.0; WIDTH],
        second_weights: vec![0.0; WIDTH * WIDTH],
        second_bias: vec![0.0; WIDTH],
        output_weights: vec![0.0; WIDTH],
        output_bias: 0.0,
    };
    // Pass the directed bigram bucket for 22 -> 11 through both hidden layers to output.
    let bucket = pair_bucket(22, 11);
    // Hidden unit 0 reads the bucket; second-layer unit 0 reads hidden unit 0.
    p.first_weights[HALF_DIM + bucket] = 1.0;
    p.second_weights[0] = 1.0;
    p.output_weights[0] = 1.0;
    p
}

#[test]
fn same_token_multiset_in_different_order_has_distinct_semantics() {
    let ranker = candidate(order_sensitive_parameters(), None);
    let left = [11, 22, 11];
    let right = [11, 11, 22];
    assert_eq!(
        {
            let mut x = left;
            x.sort();
            x
        },
        {
            let mut x = right;
            x.sort();
            x
        }
    );
    assert_ne!(
        ranker.features(&left).unwrap(),
        ranker.features(&right).unwrap()
    );
    assert!(ranker.score(&left).unwrap() > ranker.score(&right).unwrap());
}

#[test]
fn overlength_input_is_refused_without_truncation() {
    let ranker = candidate(order_sensitive_parameters(), None);
    let over = vec![1; MAX_SUPPORTED_TOKENS + 1];
    assert!(ranker
        .features(&over)
        .unwrap_err()
        .to_string()
        .contains("overlength"));
}

#[test]
fn uncertainty_is_absent_without_heldout_calibration() {
    let ranker = candidate(order_sensitive_parameters(), None);
    assert!(ranker.manifest().calibration.is_none());
}

#[test]
fn calibration_requires_a_real_heldout_pool_and_finite_error() {
    let valid = OutcomeCalibration {
        method: "heldout_absolute_error_p90_v1".into(),
        heldout_pool_sha256: digest(3),
        examples: 40,
        absolute_error_p90: 0.25,
    };
    let mut invalid = valid.clone();
    invalid.examples = 0;
    assert!(invalid.validate().is_err());
    candidate(order_sensitive_parameters(), Some(valid));
}

#[test]
fn acquisition_ranking_uses_the_frozen_pool_and_preserves_actor_scores() {
    let ranker = candidate(order_sensitive_parameters(), None);
    let actor = ranker.manifest.actor_checkpoint_sha256.as_str();
    let pool = ranker.manifest.same_pool_sha256.as_str();
    let program_a = digest(8);
    let program_b = digest(9);
    let candidates = [
        CriticCandidate {
            program_sha256: &program_a,
            tokens: &[11, 22, 11],
            actor_score: Some(0.91),
            actor_checkpoint_sha256: actor,
            pool_sha256: pool,
        },
        CriticCandidate {
            program_sha256: &program_b,
            tokens: &[11, 11, 22],
            actor_score: Some(0.12),
            actor_checkpoint_sha256: actor,
            pool_sha256: pool,
        },
    ];
    let ranked = ranker.rank_pool(pool, actor, &candidates).unwrap();
    assert_eq!(ranked[0].actor_score, Some(0.91));
    assert_eq!(ranked[1].actor_score, Some(0.12));
    assert!(ranker.rank_pool(&digest(4), actor, &candidates).is_err());
    assert!(ranker.rank_pool(pool, &digest(5), &candidates).is_err());
}

#[test]
fn unknown_manifest_fields_and_bad_shapes_are_refused() {
    let ranker = candidate(order_sensitive_parameters(), None);
    let mut bad = order_sensitive_parameters();
    bad.first_bias.pop();
    let manifest = ranker.manifest.clone();
    assert!(StructuredRankV1::new(manifest, bad).is_err());
    let value = serde_json::to_value(ranker.manifest()).unwrap();
    let mut with_unknown = value;
    with_unknown["fake_uncertainty"] = serde_json::json!(true);
    assert!(serde_json::from_value::<CriticManifest>(with_unknown).is_err());
}
