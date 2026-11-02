use super::*;

/// Preference order that builds the longest program the sampling grammar admits: expression
/// stages first, then the transforms and head with the most slots; inside expressions, functions
/// of the highest arity.
fn longest_program_order(vocab: &Vocab) -> Vec<usize> {
    let length = |id: usize| {
        let op = &vocab.ops[&id];
        1 + op
            .slots
            .iter()
            .map(|s| {
                if s.is_expr {
                    vocab.expr.max_nodes + 1
                } else {
                    1
                }
            })
            .sum::<usize>()
            + usize::from(op.allow_append)
    };
    let mut order = (0..401).collect::<Vec<usize>>();
    order.sort_by_key(|&t| {
        let key = if vocab.ops.contains_key(&t) {
            (2, length(t))
        } else if let Some(&arity) = vocab.expr.func_arity.get(&t) {
            (1, arity)
        } else {
            (0, 0)
        };
        (std::cmp::Reverse(key), t)
    });
    order
}

#[test]
fn token_budget_boundary_is_exact() {
    let vocab = vocab();
    let ctx = context();
    let ascending = (0..401).collect::<Vec<_>>();
    let longest = longest_program_order(&vocab);
    let mut lengths = Vec::new();
    for logit in [
        Box::new(ranked(&[])) as Box<dyn Fn(usize) -> f32>,
        Box::new(ranked(&ascending)),
        Box::new(ranked(&longest)),
    ] {
        let walk = deterministic_walk(&vocab, &*logit, MAX_PROGRAM_TOKENS);
        assert_eq!(walk.last(), Some(&EOS));
        let n = walk.len();
        lengths.push(n);
        // Exactly at the budget: EOS is the last admissible token and the draw completes, on
        // the cached pool path and (for the shorter walks; it recomputes every prefix, which is
        // slow in a debug build) on the full-prefix reference path.
        let at = fixed_logit_model(n, &*logit);
        let over = fixed_logit_model(n - 1, &*logit);
        if n <= 32 {
            let mut rng = ChaCha20Rng::seed_from_u64(6);
            let rows = decode::sample_batch(&at, &vocab, &ctx, "regression", 1, 1.0, &mut rng);
            let [row] = &rows.unwrap()[..] else {
                panic!("one row requested")
            };
            assert_eq!(row.program.as_ref(), Some(&walk));
            assert_eq!(row.tokens, walk);
            assert_eq!(row.grammar_phase, "done");
            // One token short: the accepting EOS cannot be drawn and the draw is incomplete.
            let mut rng = ChaCha20Rng::seed_from_u64(6);
            let rows = decode::sample_batch(&over, &vocab, &ctx, "regression", 1, 1.0, &mut rng);
            let [row] = &rows.unwrap()[..] else {
                panic!("one row requested")
            };
            assert_eq!(row.program, None);
            assert_eq!(row.tokens, walk[..n - 1]);
            assert_eq!(row.grammar_phase, "eos");
        }
        let (pool, _) =
            pure_pool_unbound(&at, &vocab, &ctx, "regression", 1, 2, &options(1, 2)).unwrap();
        assert_eq!(pool.count(DrawOutcome::Completed), 1);
        assert_eq!(pool.count(DrawOutcome::Duplicate), 1);
        assert!(pool
            .draws()
            .iter()
            .all(|d| d.tokens == walk && d.grammar_phase == "done"));
        let (pool, _) =
            pure_pool_unbound(&over, &vocab, &ctx, "regression", 1, 2, &options(2, 2)).unwrap();
        assert_eq!((pool.attempted, pool.valid, pool.unique_neural), (4, 0, 0));
        assert_eq!(pool.count(DrawOutcome::Incomplete), 4);
        assert!(pool
            .draws()
            .iter()
            .all(|d| d.tokens == walk[..n - 1] && d.grammar_phase == "eos"));
    }
    // The longest program the sampling grammar admits (see `grammar_differential`) fits the
    // 128-token budget with room to spare.
    assert_eq!(lengths, [5, 25, 78]);
}

/// W4-T07: real incomplete draws (not relabelled records) through a whole proposal and its
/// ledger. The policy prefers the longest walk the sampling grammar admits (78 tokens) under a
/// 77-token budget, so every neural draw ends incomplete and the fallback fill tops the pool up.
/// Reproduction: a truncated draw recorded as incomplete, and an incomplete draw forged into a
/// parser refusal with consistent denominators, were both reconciled.
#[test]
fn real_incomplete_draws_reconcile_and_forgeries_are_refused() {
    use jope_policy::config::CheckpointBinding;
    use jope_policy::seeds::{ProposalSeeds, SeedBinding};
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../jope-engine/assets/grammar/featmath.json");
    let bound =
        bind_vocabulary(&std::fs::read(path).unwrap(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let vocab = bound.vocab().clone();
    let longest = longest_program_order(&vocab);
    let walk = deterministic_walk(&vocab, &ranked(&longest), MAX_PROGRAM_TOKENS);
    assert_eq!(walk.len(), 78);
    let budget = walk.len() - 1;
    let model = fixed_logit_model(budget, ranked(&longest));
    let ctx = context();
    let checkpoint = CheckpointBinding::unbound_for_testing(
        model.config().clone(),
        CHECKPOINT_VOCABULARY_SHA256,
    );
    let binding = SeedBinding::new(
        &sha256_hex(b"prepared"),
        0,
        &sha256_hex(b"training"),
        &jope_policy::context_sha256(&ctx),
    )
    .unwrap();
    let proposal = jope_policy::testing::propose_unbound(
        &model,
        &checkpoint,
        &bound,
        &ctx,
        &binding,
        &ProposalSeeds {
            sampling: 1,
            fallback: 2,
            exploration: 3,
        },
        &options(1, 4),
    )
    .unwrap();
    let ledger = proposal.ledger();
    let d = &ledger.denominators;
    assert_eq!(
        (
            d.scheduled_neural_draws,
            d.incomplete,
            d.complete_sequences,
            d.completed
        ),
        (4, 4, 0, 0)
    );
    assert_eq!(d.fallback_admitted, POOL_SIZE);
    for draw in &ledger.neural_draws {
        assert_eq!(draw.outcome, DrawOutcome::Incomplete);
        assert_eq!(draw.tokens, walk[..budget]);
        assert_ne!(draw.grammar_phase, "done");
    }
    ledger.reconcile_with_vocabulary(&vocab).unwrap();
    assert_eq!(
        ledger
            .shadow()
            .iter()
            .filter(|s| s.outcome == "Incomplete")
            .count(),
        4
    );
    // A truncated draw recorded as incomplete.
    let mut truncated = ledger.clone();
    truncated.neural_draws[0].tokens.pop();
    let truncated = truncated.reconcile().err().map(|e| format!("{e:#}"));
    // An incomplete draw forged into a parser refusal (its last token replaced by EOS), with
    // consistent denominators: the claims agree, the grammar does not.
    let mut forged = ledger.clone();
    *forged.neural_draws[0].tokens.last_mut().unwrap() = EOS;
    forged.neural_draws[0].outcome = DrawOutcome::ParserRejected;
    forged.neural_draws[0].grammar_phase = "done".into();
    forged.denominators.incomplete -= 1;
    forged.denominators.parser_rejected += 1;
    forged.denominators.complete_sequences += 1;
    forged.reconcile().unwrap();
    let forged = forged
        .reconcile_with_vocabulary(&vocab)
        .err()
        .map(|e| format!("{e:#}"));
    assert!(
        truncated
            .as_deref()
            .is_some_and(|e| e.contains("did not exhaust its token budget"))
            && forged
                .as_deref()
                .is_some_and(|e| e.contains("legal grammar walk")),
        "forged incomplete records were reconciled: truncated {truncated:?}, forged refusal {forged:?}"
    );
}
