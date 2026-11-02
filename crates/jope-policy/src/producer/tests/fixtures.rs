pub(crate) fn request() -> Request {
    Request {
        worker_source_sha256: sha256_hex(b"synthetic producer source"),
        prepared_sha256: sha256_hex(b"synthetic prepared"),
        training_sha256: sha256_hex(b"synthetic training"),
        cut: 0,
        context_sha256: sha256_hex(b"synthetic context"),
        checkpoint_sha256: crate::ACCEPTED_STATE_SHA256.into(),
        policy_safetensors_sha256:
            "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e".into(),
        vocabulary_sha256: crate::CHECKPOINT_VOCABULARY_SHA256.into(),
        sampling_seed: 1,
        fallback_seed: 2,
        exploration_seed: 3,
        sampling_rounds: 8,
        sampling_batch: 128,
        sampling_temperature: 1.,
        challenger: None,
        context_admission_sha256: None,
    }
}
// Deliberately forgeable file claims, never a Proposal or execution proof.
fn claims() -> Envelope {
    claims_for(request())
}
fn claims_for(request: Request) -> Envelope {
    let genomes = (0..80)
        .map(|i| format!("(pipe X (ridge lambda={}))", i + 1))
        .collect::<Vec<_>>();
    let population = serde_json::json!({"schema":seeds::SEED_SCHEMA,"prepared_sha256":request.prepared_sha256,"training_sha256":request.training_sha256,"cut":request.cut,"context_sha256":request.context_sha256,"checkpoint_sha256":request.checkpoint_sha256,"candidates":genomes.iter().enumerate().map(|(i,g)|serde_json::json!({"genome":g,"origin":if i<64 { seeds::ORIGIN_NEURAL } else { seeds::ORIGIN_RANDOM },"rank_log_weight":null})).collect::<Vec<_>>()});
    let bytes = serde_json::to_vec(&population).unwrap();
    Envelope {
        schema: SCHEMA.into(),
        request_sha256: request.identity().unwrap(),
        request,
        population_json: String::from_utf8(bytes.clone()).unwrap(),
        population_sha256: sha256_hex(&bytes),
        canonical_recipe_sha256: genomes.iter().map(|g| recipe_hash(g).unwrap()).collect(),
        ranking_backend: crate::ranking::BACKEND_CPU.into(),
        proposal_ledger: None,
    }
}
#[test]
fn producer_reader_preserves_80_typed_recipes_but_is_only_claims() -> Result<()> {
    let file = read_claims(&serde_json::to_vec(&claims())?, &request())?;
    assert_eq!(file.population().candidates().len(), 80);
    assert_eq!(
        file.population().candidates()[0].genome(),
        "(pipe X (ridge lambda=1))"
    );
    Ok(())
}
#[test]
fn producer_transport_rejects_seed_request_recipe_and_context_tampering() -> Result<()> {
    let expected = request();
    let mut changed = claims();
    changed.request.sampling_seed += 1;
    assert!(read_claims(&serde_json::to_vec(&changed)?, &expected).is_err());
    let mut changed = claims();
    changed.canonical_recipe_sha256[0] = sha256_hex(b"different recipe");
    assert!(read_claims(&serde_json::to_vec(&changed)?, &expected).is_err());
    let mut changed = claims();
    changed.population_json.push(' ');
    assert!(read_claims(&serde_json::to_vec(&changed)?, &expected).is_err());
    let mut different = expected.clone();
    different.context_sha256 = sha256_hex(b"another context");
    assert!(read_claims(&serde_json::to_vec(&claims())?, &different).is_err());
    Ok(())
}

/// A real (untrained, unbound) proposal and a challenger-shaped request naming its test
/// state, so the producer transport can be exercised end to end in a unit test.
pub(crate) fn unbound_proposal() -> (Request, crate::seeds::Proposal) {
    unbound_proposal_with(1, 8, 1.0)
}
/// A second fixture whose pool fills during sampling: surplus and duplicate neural draws,
/// no fallback draws (the first fixture has only completed neural draws, many fallback
/// draws and at least one refused fallback draw).
pub(crate) fn filled_proposal() -> (Request, crate::seeds::Proposal) {
    unbound_proposal_with(2, 64, 25.0)
}
pub(crate) fn unbound_proposal_with(
    rounds: usize,
    batch: usize,
    temperature: f32,
) -> (Request, crate::seeds::Proposal) {
    let config = tiny_config(128);
    let model = crate::model::PolicyModel::random_weights(config, 5).unwrap();
    fixture_proposal(&model, &accepted_vocabulary(), rounds, batch, temperature).unwrap()
}

fn accepted_vocabulary() -> crate::grammar::BoundVocabulary {
    crate::grammar::bind_vocabulary(
        include_bytes!("../../../jope-engine/assets/grammar/featmath.json"),
        crate::CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap()
}

fn tiny_config(max_seq: usize) -> crate::config::PolicyConfig {
    let mut config = crate::config::PolicyConfig::regression(401);
    config.width = 16;
    config.heads = 4;
    config.ffn = 32;
    config.layers = 1;
    config.dropout = 0.0;
    config.max_seq = max_seq;
    config
}

/// A real proposal by untrained `model` (the test state, a challenger-shaped request naming
/// it), so the producer transport can be exercised end to end in a unit test. The accepted
/// vocabulary goes through `propose_unbound`; any other only through the unit-test entry.
fn fixture_proposal(
    model: &crate::model::PolicyModel,
    vocabulary: &crate::grammar::BoundVocabulary,
    rounds: usize,
    batch: usize,
    temperature: f32,
) -> Result<(Request, crate::seeds::Proposal)> {
    let checkpoint = crate::config::CheckpointBinding::unbound_for_testing(
        model.config().clone(),
        crate::CHECKPOINT_VOCABULARY_SHA256,
    );
    let context = PolicyContext::from_frozen_values(
        (0..2048).map(|i| (i % 13) as f32 / 13.0 - 0.4).collect(),
    )?;
    let state = crate::testing::UNBOUND_TEST_STATE_SHA256.to_string();
    let weights = sha256_hex(b"unbound test weights");
    let request = Request {
        checkpoint_sha256: state.clone(),
        policy_safetensors_sha256: weights.clone(),
        context_sha256: crate::context_sha256(&context),
        vocabulary_sha256: vocabulary.sha256().into(),
        sampling_rounds: rounds,
        sampling_batch: batch,
        sampling_temperature: temperature,
        challenger: Some(crate::config::ChallengerIdentity {
            state_sha256: state,
            parameter_sha256: sha256_hex(b"unbound test parameters"),
            config_sha256: crate::ACCEPTED_CONFIG_SHA256.into(),
            policy_safetensors_sha256: weights,
        }),
        ..request()
    };
    let binding = seeds::SeedBinding::new(
        &request.prepared_sha256,
        request.cut,
        &request.training_sha256,
        &request.context_sha256,
    )?;
    let proposal_seeds = seeds::ProposalSeeds {
        sampling: request.sampling_seed,
        fallback: request.fallback_seed,
        exploration: request.exploration_seed,
    };
    let sampling = crate::decode::SamplingOptions {
        rounds: request.sampling_rounds,
        batch: request.sampling_batch,
        temperature: request.sampling_temperature,
    };
    let proposal = if vocabulary.sha256() == crate::CHECKPOINT_VOCABULARY_SHA256 {
        seeds::propose_unbound(
            model,
            &checkpoint,
            vocabulary,
            &context,
            &binding,
            &proposal_seeds,
            &sampling,
        )?
    } else {
        seeds::propose_with_vocabulary_for_test(
            model,
            vocabulary,
            &context,
            &binding,
            &proposal_seeds,
            &sampling,
        )?
    };
    Ok((request, proposal))
}
