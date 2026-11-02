use super::*;

#[test]
fn propose_challenger_stamps_the_challenger_state() {
    let dir = scratch("propose");
    let (path, receipt, identity) = challenger_export(&dir.join("first"), 0.0, "challenger state");
    let binding = challenger_binding(&receipt, &identity).unwrap();
    let m = PolicyModel::load_bound(&path, &binding, Device::Cpu).unwrap();
    let (second_path, second_receipt, second_identity) =
        challenger_export(&dir.join("second"), 0.5, "second challenger state");
    let second_binding = challenger_binding(&second_receipt, &second_identity).unwrap();
    let second = PolicyModel::load_bound(&second_path, &second_binding, Device::Cpu).unwrap();
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
    let attempt =
        |model: &PolicyModel, checkpoint: &CheckpointBinding, pinned: &ChallengerIdentity| {
            seeds::propose_challenger(
                model,
                checkpoint,
                pinned,
                &vocabulary,
                &ctx,
                &run,
                &proposal_seeds,
                &options,
            )
        };
    let proposal = attempt(&m, &binding, &identity).unwrap();
    let population = proposal.population();
    assert_eq!(population.checkpoint_sha256(), identity.state_sha256);
    assert_eq!(
        proposal.ranking().bindings().checkpoint_sha256(),
        identity.state_sha256
    );
    assert_eq!(population.candidates().len(), 80);
    assert!(population.candidates()[..64]
        .iter()
        .all(|c| c.origin() == ORIGIN_NEURAL || c.origin() == ORIGIN_NEURAL_FALLBACK));
    assert!(population.candidates()[64..]
        .iter()
        .all(|c| c.origin() == ORIGIN_RANDOM));
    let admitted = validate_for_search(
        &population_bytes(&proposal).unwrap(),
        run.prepared_sha256(),
        run.cut(),
        run.training_sha256(),
    )
    .unwrap();
    assert_eq!(admitted.checkpoint_sha256(), identity.state_sha256);
    // (a) Untrained weights, (b) another pinned parameter hash, (c) another pinned state,
    // (d) a model bound to another challenger, and (e) the accepted receipt are all refused.
    let random = PolicyModel::random_weights(tiny(), 9).unwrap();
    assert_refused(attempt(&random, &binding, &identity), "load_bound");
    let other_parameters = ChallengerIdentity {
        parameter_sha256: digest("other parameters"),
        ..identity.clone()
    };
    assert_refused(
        attempt(&m, &binding, &other_parameters),
        "not the pinned challenger parameters",
    );
    let other_state = ChallengerIdentity {
        state_sha256: digest("other state"),
        ..identity.clone()
    };
    assert_refused(
        attempt(&m, &binding, &other_state),
        "checkpoint is not the pinned challenger",
    );
    assert_refused(
        attempt(&second, &binding, &identity),
        "bound to a different checkpoint",
    );
    let accepted = CheckpointBinding::from_receipt_bytes(&to_bytes(&accepted_receipt())).unwrap();
    assert!(attempt(&m, &accepted, &identity).is_err());
}
