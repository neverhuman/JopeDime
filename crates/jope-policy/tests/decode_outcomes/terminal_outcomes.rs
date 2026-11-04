use super::*;

fn dead_end_vocab() -> Vocab {
    let mut vocab = vocab();
    // Adversarial vocabulary: `stack` gains a slot with no value and no absent token.
    vocab.ops.get_mut(&STACK).unwrap().slots = vec![Slot {
        param: "dead_end".into(),
        kind: "int".into(),
        val_ids: vec![],
        bin_texts: vec![],
        absent_id: None,
        is_expr: false,
        required: true,
        default: serde_json::Value::Null,
    }];
    vocab
}

#[test]
fn grammar_dead_end_fails_with_the_draw_identity() {
    let vocab = dead_end_vocab();
    let ctx = context();
    let model = fixed_logit_model(128, ranked(&[]));
    let mut rng = ChaCha20Rng::seed_from_u64(4);
    let result = decode::sample_batch(&model, &vocab, &ctx, "regression", 2, 1.0, &mut rng);
    let message = format!("{:#}", result.as_ref().unwrap_err());
    assert!(message.contains("round 0, row 0"), "{message}");
    let refused = failure(result);
    assert_eq!(refused.kind, FailureKind::EmptySupport);
    assert_eq!(refused.draw, Some(DrawId { round: 0, row: 0 }));
    assert_eq!(refused.tokens, [1, 3, 397, STACK]);
    assert_eq!(refused.grammar_phase, Some("value"));
    // The grammar-uniform generator stops at the same dead end with a typed failure.
    let refused = (0..2000u64)
        .find_map(|seed| {
            let mut rng = <rand_chacha::ChaCha8Rng as SeedableRng>::seed_from_u64(seed);
            decode::grammar_uniform_program(&vocab, "regression", &mut rng)
                .err()
                .map(|e| e.downcast_ref::<DecodeFailure>().cloned().expect("typed"))
        })
        .expect("a uniform walk reaches the dead end");
    assert_eq!(refused.kind, FailureKind::EmptySupport);
    assert_eq!(refused.draw, None);
    assert_eq!(refused.tokens.last(), Some(&STACK));
    assert_eq!(refused.grammar_phase, Some("value"));
}

#[test]
fn every_valid_draw_is_accounted_for() {
    let vocab = vocab();
    let ctx = context();
    // Uniform over four wide regression heads and their values; every other stage token has a
    // -Inf logit (zero probability, never embedded).
    let heads = vocab.engine_heads["featmath"].clone();
    let transforms = vocab.engine_transforms["featmath"].clone();
    let model = fixed_logit_model(128, |t| {
        if transforms.contains(&t) || (heads.contains(&t) && !WIDE_HEADS.contains(&t)) {
            f32::NEG_INFINITY
        } else {
            0.0
        }
    });
    let (pool, _) =
        pure_pool_unbound(&model, &vocab, &ctx, "regression", 21, 22, &options(1, 128)).unwrap();
    assert_eq!(pool.attempted, 128);
    assert_eq!(pool.unique_neural, 64);
    let surplus = pool.count(DrawOutcome::Surplus);
    assert!(surplus > 0, "fixture must fill the pool mid-round");
    // Every valid draw is pooled, a duplicate, or explicitly surplus.
    assert_eq!(
        pool.valid,
        pool.unique_neural + pool.within_neural_duplicates + surplus
    );
    assert_eq!(pool.count(DrawOutcome::Completed), pool.unique_neural);
    assert_eq!(
        pool.count(DrawOutcome::Duplicate),
        pool.within_neural_duplicates
    );
    // Completed draws are the pooled neural candidates, in order; surplus programs are unique
    // and never pooled; duplicates repeat an earlier valid draw.
    let completed = pool
        .draws()
        .iter()
        .filter(|d| d.outcome == DrawOutcome::Completed)
        .map(|d| d.program_sha256.clone().unwrap())
        .collect::<Vec<_>>();
    let pooled = pool.candidates()[..64]
        .iter()
        .map(|c| sha256_hex(c.genome().as_bytes()))
        .collect::<Vec<_>>();
    assert_eq!(completed, pooled);
    let mut earlier = BTreeSet::new();
    for record in pool.draws() {
        let sha = record.program_sha256.clone().unwrap();
        match record.outcome {
            DrawOutcome::Duplicate => assert!(earlier.contains(&sha)),
            DrawOutcome::Surplus => assert!(!pooled.contains(&sha) && earlier.insert(sha)),
            DrawOutcome::Completed => assert!(earlier.insert(sha)),
            outcome => panic!("unexpected {outcome:?}"),
        }
    }
    assert_eq!(pool.fallback_attempts, 0);
}

#[test]
fn every_scheduled_draw_has_exactly_one_terminal_outcome() {
    let vocab = vocab();
    let ctx = context();
    // Untrained weights at a high temperature with a 12-token budget: a mix of completed,
    // duplicate and incomplete draws.
    let model = PolicyModel::random_weights(config(12), 3).unwrap();
    let sampling = SamplingOptions {
        rounds: 2,
        batch: 32,
        temperature: 25.0,
    };
    let (pool, seen) =
        pure_pool_unbound(&model, &vocab, &ctx, "regression", 31, 32, &sampling).unwrap();
    assert_eq!(pool.token_budget, 12);
    assert_eq!((pool.options.rounds, pool.options.batch), (2, 32));
    assert_eq!(pool.attempted, 64);
    let draws = pool.draws();
    assert_eq!(draws.len(), pool.attempted);
    for (i, record) in draws.iter().enumerate() {
        assert_eq!(
            record.draw,
            DrawId {
                round: i / 32,
                row: i % 32
            }
        );
        assert_eq!(record.tokens[..3], PREFIX);
        // Replay: every sampled token was in the sampling-mode legal support.
        let mut state = vocab.prefix_state("regression").unwrap();
        for &token in &record.tokens[3..] {
            assert!(state.legal(&vocab).contains(&token));
            state.advance(&vocab, token).unwrap();
        }
        assert_eq!(state.phase_name(), record.grammar_phase);
        match record.outcome {
            DrawOutcome::Incomplete => {
                assert_eq!(record.tokens.len(), 12);
                assert_ne!(record.grammar_phase, "done");
                assert_eq!(record.program_sha256, None);
            }
            DrawOutcome::ParserRejected => panic!("the frozen grammar always parses here"),
            _ => {
                assert_eq!(record.grammar_phase, "done");
                assert_eq!(record.tokens.last(), Some(&EOS));
                let genome = decode::valid_genome(&vocab, &record.tokens).unwrap();
                assert!(seen.contains(&genome));
                assert_eq!(
                    record.program_sha256.as_deref(),
                    Some(sha256_hex(genome.as_bytes()).as_str())
                );
            }
        }
    }
    let incomplete = pool.count(DrawOutcome::Incomplete);
    assert!(incomplete > 0 && pool.count(DrawOutcome::Completed) > 0);
    assert_eq!(
        pool.attempted - pool.valid,
        incomplete + pool.count(DrawOutcome::ParserRejected)
    );
    assert_eq!(
        pool.sampled_tokens,
        draws.iter().map(|d| d.tokens.len() - 3).sum::<usize>()
    );
    // Fallback is bounded, labelled, and fills exactly the slots the policy left.
    assert!(pool.fallback_attempts > 0 && pool.fallback_attempts <= FALLBACK_ATTEMPTS);
    let origins = pool
        .candidates()
        .iter()
        .map(|c| c.origin())
        .collect::<Vec<_>>();
    assert_eq!(origins.len(), POOL_SIZE);
    assert!(origins[..pool.unique_neural]
        .iter()
        .all(|o| *o == ORIGIN_NEURAL));
    assert!(origins[pool.unique_neural..]
        .iter()
        .all(|o| *o == ORIGIN_NEURAL_FALLBACK));
}

#[test]
fn parser_rejection_is_an_explicit_outcome() {
    let mut vocab = vocab();
    // Adversarial vocabulary: the head name no longer renders as valid DSL.
    vocab.ops.get_mut(&STACK).unwrap().name = "stack)".into();
    assert!(decode::valid_genome(&vocab, &[1, 3, 397, STACK, EOS]).is_none());
    let ctx = context();
    let model = fixed_logit_model(128, ranked(&[]));
    let (pool, _) =
        pure_pool_unbound(&model, &vocab, &ctx, "regression", 5, 6, &options(1, 4)).unwrap();
    assert_eq!((pool.attempted, pool.valid, pool.unique_neural), (4, 0, 0));
    assert_eq!(pool.count(DrawOutcome::ParserRejected), 4);
    for record in pool.draws() {
        assert_eq!(record.tokens, [1, 3, 397, STACK, EOS]);
        assert_eq!(record.grammar_phase, "done");
        assert_eq!(record.program_sha256, None);
    }
    // Never relabelled: the pool is all fallback.
    assert!(pool
        .candidates()
        .iter()
        .all(|c| c.origin() == ORIGIN_NEURAL_FALLBACK));
}
