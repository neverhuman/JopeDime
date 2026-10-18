use jope_policy::{
    decode,
    grammar::{bind_vocabulary, State},
    support_transition::{BehaviorLaw, PropensityValue},
    CHECKPOINT_VOCABULARY_SHA256,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::Path;

fn bound() -> jope_policy::grammar::BoundVocabulary {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../jope-engine/assets/grammar/featmath.json");
    bind_vocabulary(&std::fs::read(path).unwrap(), CHECKPOINT_VOCABULARY_SHA256).unwrap()
}

#[test]
fn native_sampler_and_scorer_share_masked_law_and_seeded_frequency() {
    let bound = bound();
    let law = BehaviorLaw::native(bound.sha256(), 1.0).unwrap();
    let state = State::default();
    let legal = state.legal(bound.vocab());
    let mut logits = vec![0.0f32; bound.vocab().tokens.len()];
    for (index, token) in legal.iter().enumerate() {
        logits[*token] = index as f32 * 0.25;
    }
    let expected = law.score_legal(&logits, &legal).unwrap();
    assert!((expected.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
    let mut rng = ChaCha8Rng::seed_from_u64(0x106);
    let mut counts = vec![0usize; legal.len()];
    for _ in 0..30_000 {
        let (token, propensity) = law.sample_state(&bound, &state, &logits, &mut rng).unwrap();
        let PropensityValue::Discrete { probability, .. } = propensity.value() else {
            panic!("discrete draw required")
        };
        let index = legal.binary_search(&token).unwrap();
        assert!((probability - expected[index].1).abs() < 1e-15);
        counts[index] += 1;
    }
    for (count, (_, probability)) in counts.iter().zip(&expected) {
        let frequency = *count as f64 / 30_000.0;
        let tolerance = (6.0 * (probability * (1.0 - probability) / 30_000.0).sqrt()).max(0.01);
        assert!((frequency - probability).abs() <= tolerance);
    }
    assert!(decode::score_legal(&logits, &legal, 0.0).is_err());
}

#[test]
fn unknown_selection_propensity_cannot_claim_unbiasedness() {
    let unknown = BehaviorLaw::unknown_selection("top-k handoff and dedup law not recorded");
    assert!(!unknown.permits_unbiased_claim());
    assert!(unknown.is_unknown());
    assert!(matches!(unknown.value(), PropensityValue::Unknown { .. }));
    assert!(BehaviorLaw::native("not-a-vocabulary-pin", f32::NAN).is_err());
}

#[test]
fn native_round_trip_preserves_401_vocabulary_and_refuses_overlength() {
    let bound = bound();
    assert_eq!(bound.vocab().tokens.len(), 401);
    let tokens = bound.representable_tokens("(pipe X (ridge))").unwrap();
    assert_eq!(tokens.last(), Some(&2));
    let law = BehaviorLaw::native(bound.sha256(), 1.0).unwrap();
    let rows = vec![vec![0.0f32; bound.vocab().tokens.len()]; tokens.len() - 1];
    let propensity = law.score_program(&bound, &tokens, &rows).unwrap();
    let PropensityValue::Discrete {
        probability,
        log_probability,
    } = propensity.value()
    else {
        panic!("complete discrete native program propensity required")
    };
    assert!(*probability > 0.0 && (probability.ln() - *log_probability).abs() < 1e-12);
    let too_long = format!("(pipe X {} (ridge))", "(standardize) ".repeat(140));
    assert!(bound.representable_tokens(&too_long).is_err());
}
