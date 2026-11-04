use super::*;

#[test]
fn challenger_never_satisfies_the_accepted_policy_checks() {
    const NOT_ACCEPTED: &str = "not the accepted step-60000 state";
    let dir = scratch("accepted-checks");
    let (path, receipt, identity) = challenger_export(&dir, 0.0, "challenger state");
    let binding = challenger_binding(&receipt, &identity).unwrap();
    let m = PolicyModel::load_bound(&path, &binding, Device::Cpu).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let vocabulary = vocabulary();
    let ctx = context();
    let run = run_binding(&ctx);
    let proposal_seeds = ProposalSeeds::derive(3);
    let options = SamplingOptions {
        rounds: 1,
        batch: 4,
        temperature: 1.0,
    };
    assert_refused(crate::require_accepted_policy(&m), NOT_ACCEPTED);
    assert_refused(
        decode::pure_pool(&m, vocabulary.vocab(), &ctx, "regression", 1, 2, &options),
        NOT_ACCEPTED,
    );
    assert_refused(
        Bindings::for_accepted_policy(&m, &context_sha256(&ctx), &vocabulary),
        NOT_ACCEPTED,
    );
    assert!(seeds::propose(
        &m,
        &binding,
        &vocabulary,
        &ctx,
        &run,
        &proposal_seeds,
        &options
    )
    .is_err());
    let accepted = CheckpointBinding::from_receipt_bytes(&to_bytes(&accepted_receipt())).unwrap();
    assert!(seeds::propose(
        &m,
        &accepted,
        &vocabulary,
        &ctx,
        &run,
        &proposal_seeds,
        &options
    )
    .is_err());
    assert!(binding.require_accepted_identity().is_err());
    // The test helpers refuse any bound model.
    let unbound = CheckpointBinding::unbound_for_testing(tiny(), CHECKPOINT_VOCABULARY_SHA256);
    assert_refused(
        crate::testing::propose_unbound(
            &m,
            &unbound,
            &vocabulary,
            &ctx,
            &run,
            &proposal_seeds,
            &options,
        ),
        "unbound test path",
    );
    assert_refused(
        crate::testing::pure_pool_unbound(
            &m,
            vocabulary.vocab(),
            &ctx,
            "regression",
            1,
            2,
            &options,
        ),
        "unbound test path",
    );
    // Scoring: never under the accepted state or another state, only under its own.
    let programs = vec!["(pipe X (ridge))".to_string()];
    let one = Options {
        batch_size: 1,
        keep: 1,
    };
    let claimed: Bindings = serde_json::from_value(json!({
        "checkpoint_sha256": ACCEPTED_STATE_SHA256,
        "conditioning_context_sha256": context_sha256(&ctx),
        "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
    }))
    .unwrap();
    assert!(score_pool(&m, vocabulary.vocab(), &ctx, &programs, claimed, &one).is_err());
    let elsewhere = Bindings::new(
        &digest("other state"),
        &context_sha256(&ctx),
        CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap();
    assert_refused(
        score_pool(&m, vocabulary.vocab(), &ctx, &programs, elsewhere, &one),
        "scored under",
    );
    let own = Bindings::new(
        &identity.state_sha256,
        &context_sha256(&ctx),
        CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap();
    let ranking = score_pool(&m, vocabulary.vocab(), &ctx, &programs, own, &one).unwrap();
    assert_eq!(
        ranking.bindings().checkpoint_sha256(),
        identity.state_sha256
    );
}
