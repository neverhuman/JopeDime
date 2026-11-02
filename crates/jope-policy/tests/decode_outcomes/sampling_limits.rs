use super::*;

#[test]
fn pure_pool_refuses_an_unbounded_sampling_budget_before_any_draw() {
    let vocab = vocab();
    let model = fixed_logit_model(128, ranked(&[]));
    let ctx = context();
    // Positive control: jope-dime's own plan (8 rounds of 128 rows) is the largest accepted
    // budget. One distinct program is ever drawn, so every round runs.
    let (pool, _) = pure_pool_unbound(
        &model,
        &vocab,
        &ctx,
        "regression",
        1,
        2,
        &SamplingOptions::default(),
    )
    .unwrap();
    assert_eq!((MAX_ROUNDS, MAX_BATCH), (8, 128));
    assert_eq!(pool.attempted, 1024);
    assert_eq!(pool.unique_neural, 1);
    assert_eq!(pool.draws().len(), 1024);
    assert_eq!(pool.count(DrawOutcome::Completed), 1);
    assert_eq!(pool.count(DrawOutcome::Duplicate), 1023);
    assert_eq!(pool.draws()[1023].draw, DrawId { round: 7, row: 127 });
    // Without the bound the cases after (9, 1) never terminate or fail allocation.
    for (rounds, batch) in [
        (0, 1),
        (1, 0),
        (9, 1),
        (1, 129),
        (usize::MAX, 1),
        (1, usize::MAX),
    ] {
        let refused = failure(pure_pool_unbound(
            &model,
            &vocab,
            &ctx,
            "regression",
            1,
            2,
            &options(rounds, batch),
        ));
        assert_eq!(refused.kind, FailureKind::InvalidBudget { rounds, batch });
        assert_eq!(refused.draw, None);
    }
}

#[test]
fn sample_batch_refuses_an_unbounded_batch() {
    let vocab = vocab();
    let model = fixed_logit_model(128, ranked(&[]));
    let ctx = context();
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    assert_eq!(
        decode::sample_batch(&model, &vocab, &ctx, "regression", 128, 1.0, &mut rng)
            .unwrap()
            .len(),
        128
    );
    for batch in [0, 129] {
        let mut untouched = rng.clone();
        let refused = failure(decode::sample_batch(
            &model,
            &vocab,
            &ctx,
            "regression",
            batch,
            1.0,
            &mut rng,
        ));
        assert_eq!(
            refused.kind,
            FailureKind::InvalidBudget { rounds: 1, batch }
        );
        assert_eq!(rng.next_u64(), untouched.next_u64());
    }
}

#[test]
fn invalid_temperature_and_missing_token_budget_are_refused_without_a_token_step() {
    let vocab = vocab();
    let ctx = context();
    // Token budget min(128, max_seq) == 3 == the proposal prefix: no token can be sampled.
    let short = fixed_logit_model(3, ranked(&[]));
    let mut rng = ChaCha20Rng::seed_from_u64(5);
    let refused = failure(decode::sample_batch(
        &short,
        &vocab,
        &ctx,
        "regression",
        2,
        f32::NAN,
        &mut rng,
    ));
    assert!(matches!(refused.kind, FailureKind::InvalidTemperature(t) if t.is_nan()));
    let refused = failure(decode::sample_batch(
        &short,
        &vocab,
        &ctx,
        "regression",
        2,
        1.0,
        &mut rng,
    ));
    assert_eq!(
        refused.kind,
        FailureKind::NoTokenBudget {
            prefix: 3,
            budget: 3
        }
    );
    // The pool does not substitute grammar-uniform fallbacks for a policy that cannot sample.
    let refused = failure(pure_pool_unbound(
        &short,
        &vocab,
        &ctx,
        "regression",
        1,
        2,
        &options(1, 2),
    ));
    assert_eq!(refused.seed, Some(1));
    assert!(matches!(refused.kind, FailureKind::NoTokenBudget { .. }));
}

/// Regression protection: an invalid temperature is refused, consumes no random draw and never
/// becomes a greedy/argmax mode; a tiny positive temperature is ordinary sampling.
#[test]
fn invalid_temperature_never_draws_and_is_not_a_greedy_mode() {
    let vocab = vocab();
    let ctx = context();
    let model = fixed_logit_model(128, ranked(&[]));
    for temperature in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        let mut untouched = rng.clone();
        let refused = failure(decode::sample_batch(
            &model,
            &vocab,
            &ctx,
            "regression",
            3,
            temperature,
            &mut rng,
        ));
        assert!(matches!(refused.kind, FailureKind::InvalidTemperature(_)));
        assert_eq!(
            rng.next_u64(),
            untouched.next_u64(),
            "temperature {temperature} drew"
        );
        let options = SamplingOptions {
            rounds: 1,
            batch: 3,
            temperature,
        };
        assert!(options.validate().is_err());
        let refused = failure(pure_pool_unbound(
            &model,
            &vocab,
            &ctx,
            "regression",
            1,
            2,
            &options,
        ));
        assert!(matches!(refused.kind, FailureKind::InvalidTemperature(_)));
    }
    let walk = deterministic_walk(&vocab, &ranked(&[]), 128);
    let mut rng = ChaCha20Rng::seed_from_u64(9);
    let tiny = f32::from_bits(1);
    for row in decode::sample_batch(&model, &vocab, &ctx, "regression", 3, tiny, &mut rng).unwrap()
    {
        assert_eq!(row.program.as_ref(), Some(&walk));
    }
}
