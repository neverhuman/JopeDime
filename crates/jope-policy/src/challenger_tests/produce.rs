use super::*;

#[test]
fn produce_sends_a_challenger_request_through_propose_challenger() {
    let dir = scratch("produce");
    let (path, receipt, identity) = challenger_export(&dir, 0.0, "challenger state");
    let binding = challenger_binding(&receipt, &identity).unwrap();
    let m = PolicyModel::load_bound(&path, &binding, Device::Cpu).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let vocabulary = vocabulary();
    let ctx = context();
    let proposal_seeds = ProposalSeeds::derive(3);
    let request = producer::Request {
        worker_source_sha256: digest("producer source"),
        prepared_sha256: digest("prepared.json"),
        training_sha256: digest("cut-0/train.csv"),
        cut: 0,
        context_sha256: context_sha256(&ctx),
        checkpoint_sha256: identity.state_sha256.clone(),
        policy_safetensors_sha256: identity.policy_safetensors_sha256.clone(),
        vocabulary_sha256: CHECKPOINT_VOCABULARY_SHA256.into(),
        sampling_seed: proposal_seeds.sampling,
        fallback_seed: proposal_seeds.fallback,
        exploration_seed: proposal_seeds.exploration,
        sampling_rounds: 1,
        sampling_batch: 4,
        sampling_temperature: 1.0,
        challenger: Some(identity.clone()),
        context_admission_sha256: None,
    };
    // `seeds::propose` refuses this model, so success means the request reached
    // `propose_challenger`.
    let bytes = producer::produce(&request, &m, &binding, &vocabulary, &ctx).unwrap();
    let file = producer::read_claims(&bytes, &request).unwrap();
    let population = file.population();
    assert_eq!(population.checkpoint_sha256(), identity.state_sha256);
    assert_eq!(population.candidates().len(), 80);
    // The same inputs without the challenger field are refused.
    assert_refused(
        producer::produce(
            &producer::Request {
                challenger: None,
                ..request.clone()
            },
            &m,
            &binding,
            &vocabulary,
            &ctx,
        ),
        "unbound policy safetensors",
    );
}
