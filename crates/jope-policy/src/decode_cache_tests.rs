use super::*;
use crate::config::PolicyConfig;
use rand::RngCore;

#[test]
fn cached_sampling_preserves_tokens_grammar_finished_rows_and_rng_tail() {
    let vocab = Vocab::from_json(include_bytes!(
        "../../jope-engine/assets/grammar/featmath.json"
    ))
    .unwrap();
    let mut config = PolicyConfig::regression(vocab.tokens.len());
    config.width = 16;
    config.heads = 4;
    config.ffn = 32;
    config.layers = 1;
    config.dropout = 0.0;
    let model = PolicyModel::random_weights(config, 3).unwrap();
    assert!(model.bound_identity().is_none());
    let context = PolicyContext::from_frozen_values(
        (0..2048).map(|i| (i % 13) as f32 / 13.0 - 0.4).collect(),
    )
    .unwrap();
    let mut full_rng = ChaCha20Rng::seed_from_u64(7);
    let mut cached_rng = full_rng.clone();
    let full = sample_batch(
        &model,
        &vocab,
        &context,
        "regression",
        24,
        25.0,
        &mut full_rng,
    )
    .unwrap();
    let cached = sample_batch_cached(
        &model,
        &vocab,
        &context,
        "regression",
        0,
        24,
        25.0,
        &mut cached_rng,
    )
    .unwrap();
    assert_eq!(full.len(), cached.len());
    let mut finished_early = false;
    for (a, b) in full.iter().zip(&cached) {
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.program, b.program);
        assert_eq!(a.grammar_phase, b.grammar_phase);
        let mut state = vocab.prefix_state("regression").unwrap();
        for &token in &b.tokens[3..] {
            if state.done() {
                finished_early = true;
                assert_eq!(token, PAD);
            } else {
                assert!(state.legal(&vocab).contains(&token));
                state.advance(&vocab, token).unwrap();
            }
        }
    }
    assert!(
        finished_early,
        "fixture must exercise removal of completed rows"
    );
    for _ in 0..16 {
        assert_eq!(full_rng.next_u64(), cached_rng.next_u64());
    }
}
