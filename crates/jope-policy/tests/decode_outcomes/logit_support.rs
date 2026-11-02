use super::*;

#[test]
fn nonfinite_logits_and_empty_support_fail_with_the_draw_identity() {
    let vocab = vocab();
    let ctx = context();
    let base = ranked(&[]);
    // Deterministic walk BOS featmath regression stack EOS.
    assert_eq!(
        deterministic_walk(&vocab, &base, 128),
        [1, 3, 397, STACK, EOS]
    );
    let nan_eos = fixed_logit_model(128, |t| if t == EOS { f32::NAN } else { base(t) });
    let inf_stack = fixed_logit_model(128, |t| if t == STACK { f32::INFINITY } else { base(t) });
    // Only the proposal prefix keeps finite logits (it is embedded as input).
    let no_support = fixed_logit_model(128, |t| {
        if PREFIX.contains(&t) {
            0.0
        } else {
            f32::NEG_INFINITY
        }
    });
    let cases: [(&PolicyModel, &[usize], &str); 3] = [
        (&nan_eos, &[1, 3, 397, STACK], "eos"),
        (&inf_stack, &PREFIX, "stage"),
        (&no_support, &PREFIX, "stage"),
    ];
    for (case, (model, tokens, phase)) in cases.into_iter().enumerate() {
        let check = |refused: &DecodeFailure| {
            match (case, &refused.kind) {
                (0, FailureKind::NonfiniteLogit { token, value }) => {
                    assert!(*token == EOS && value.is_nan())
                }
                (1, FailureKind::NonfiniteLogit { token, value }) => {
                    assert!(*token == STACK && *value == f32::INFINITY)
                }
                (2, FailureKind::EmptySupport) => {}
                (_, kind) => panic!("case {case}: unexpected failure {kind:?}"),
            }
            assert_eq!(refused.draw, Some(DrawId { round: 0, row: 0 }));
            assert_eq!(refused.tokens, tokens);
            assert_eq!(refused.grammar_phase, Some(phase));
        };
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let mut untouched = rng.clone();
        let result = decode::sample_batch(model, &vocab, &ctx, "regression", 4, 1.0, &mut rng);
        let message = format!("{:#}", result.as_ref().unwrap_err());
        assert!(message.contains("round 0, row 0"), "{message}");
        check(&failure(result));
        if case > 0 {
            // The first step failed before any random draw.
            assert_eq!(rng.next_u64(), untouched.next_u64());
        }
        // The pool entry fails too: no fallback program replaces a numerically invalid draw.
        let refused = failure(pure_pool_unbound(
            model,
            &vocab,
            &ctx,
            "regression",
            1,
            2,
            &options(2, 4),
        ));
        check(&refused);
        assert_eq!(refused.seed, Some(1));
    }
}

/// Regression protection: NaN or infinite logits of tokens that are never legal are masked
/// exactly as the reference `masked_fill(-inf)` masks them, so decoding is unchanged.
#[test]
fn nonfinite_logits_outside_the_legal_support_do_not_change_decoding() {
    let vocab = vocab();
    let ctx = context();
    let base = ranked(&[]);
    // PAD, the generate=false heads portfolio and ridge_gcv, a multiclass-only head and the
    // binary task token are never legal after the regression prefix (the prefix tokens
    // themselves are embedded as input, so they keep finite rows).
    let model = fixed_logit_model(128, |t| match t {
        0 | 369 | 371 | 324 => f32::NAN,
        398 => f32::INFINITY,
        _ => base(t),
    });
    let walk = deterministic_walk(&vocab, &base, 128);
    let mut rng = ChaCha20Rng::seed_from_u64(2);
    for row in decode::sample_batch(&model, &vocab, &ctx, "regression", 4, 1.0, &mut rng).unwrap() {
        assert_eq!(row.program.as_ref(), Some(&walk));
    }
}

/// Regression protection for the sampler arithmetic: extreme finite logits and exact ties never
/// move a draw outside the legal support, and tied tokens share the draws.
#[test]
fn extreme_finite_logits_and_ties_stay_inside_the_legal_support() {
    let logits = [f32::MAX, f32::MAX, -f32::MAX, 1e30, -1e30];
    let mut rng = ChaCha20Rng::seed_from_u64(12);
    let mut seen = BTreeSet::new();
    for _ in 0..256 {
        let token = decode::sample_legal(&logits, &[1, 2, 4], 1.0, &mut rng).unwrap();
        assert_eq!(token, 1);
        seen.insert(decode::sample_legal(&logits, &[0, 1, 2], 1.0, &mut rng).unwrap());
        let token = decode::sample_legal(&logits, &[2, 4], f32::from_bits(1), &mut rng).unwrap();
        assert_eq!(token, 4);
    }
    assert_eq!(seen, BTreeSet::from([0, 1]));
    // Through the decoder: three slot-free heads tie; every other stage token sits 1e30 below.
    let vocab = vocab();
    let ctx = context();
    let tied = [215usize, 370, STACK];
    let stage = vocab.prefix_state("regression").unwrap().legal(&vocab);
    let model = fixed_logit_model(128, |t| {
        if stage.contains(&t) && !tied.contains(&t) {
            -1e30
        } else {
            0.0
        }
    });
    let mut rng = ChaCha20Rng::seed_from_u64(13);
    let rows = decode::sample_batch(&model, &vocab, &ctx, "regression", 96, 1.0, &mut rng).unwrap();
    for head in tied {
        let n = rows.iter().filter(|r| r.tokens[3] == head).count();
        assert!(n >= 16, "tied head {head} drawn {n} of 96 times");
    }
    assert!(rows.iter().all(|r| tied.contains(&r.tokens[3])
        && r.program.as_deref() == Some(&[1, 3, 397, r.tokens[3], EOS][..])));
}
