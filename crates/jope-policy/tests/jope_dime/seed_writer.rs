use super::*;

// ---------------------------------------------------------------- seed writer

pub(crate) fn fake_ranking(state: &str, binding: &SeedBinding, programs: &[String]) -> Ranking {
    ranking_for_testing(
        Bindings::new(
            state,
            binding.context_sha256(),
            CHECKPOINT_VOCABULARY_SHA256,
        )
        .unwrap(),
        programs
            .iter()
            .enumerate()
            .map(|(i, p)| prediction(p, -(i as f64)))
            .collect(),
    )
}

/// The test binding for untrained weights (state `UNBOUND_TEST_STATE_SHA256`).
pub(crate) fn tiny_checkpoint() -> CheckpointBinding {
    CheckpointBinding::unbound_for_testing(tiny_config(), CHECKPOINT_VOCABULARY_SHA256)
}

pub(crate) fn run_binding(ctx: &[f32]) -> SeedBinding {
    SeedBinding::new(
        &digest("prepared.json"),
        0,
        &digest("cut-0/train.csv"),
        &context_sha256(ctx),
    )
    .unwrap()
}

/// The candidate and population shape jope-dime `gp.rs:551-574` deserializes.
pub(crate) fn assert_gp_seed_shape(value: &Value) {
    let object = value.as_object().unwrap();
    assert_eq!(
        object.keys().cloned().collect::<BTreeSet<_>>(),
        [
            "schema",
            "prepared_sha256",
            "cut",
            "training_sha256",
            "context_sha256",
            "checkpoint_sha256",
            "candidates"
        ]
        .into_iter()
        .map(String::from)
        .collect()
    );
    assert_eq!(value["schema"], "parallel.seed_population.v1");
    assert!(value["cut"].is_u64());
    for key in [
        "prepared_sha256",
        "training_sha256",
        "context_sha256",
        "checkpoint_sha256",
    ] {
        let s = value[key].as_str().unwrap();
        assert!(s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()));
    }
    let candidates = value["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 80);
    for c in candidates {
        let keys = c
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        assert!(keys.contains("genome") && keys.contains("origin"));
        // W4-T07: `proposal_record_sha256` joins the slot to its proposal-ledger record; the
        // search reads it (`gp::SeedCandidate`) into the slot's lineage source.
        assert!(keys.iter().all(|k| [
            "genome",
            "origin",
            "rank_log_weight",
            "proposal_record_sha256"
        ]
        .contains(&k.as_str())));
        assert!(c["genome"].is_string() && c["origin"].is_string());
        assert!(c
            .get("proposal_record_sha256")
            .is_none_or(|r| r.as_str().is_some_and(|r| r.len() == 64)));
    }
}

#[test]
fn seed_writer_produces_the_population_search_accepts() {
    let vocab = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let model = tiny_model(6);
    let ctx = context();
    let checkpoint = tiny_checkpoint();
    let binding = run_binding(&ctx);
    let options = SamplingOptions {
        rounds: 2,
        batch: 16,
        temperature: 1.0,
    };
    let proposal = propose_unbound(
        &model,
        &checkpoint,
        &vocab,
        &ctx,
        &binding,
        &ProposalSeeds::derive(20260917),
        &options,
    )
    .unwrap();
    let population = proposal.population();
    // Untrained weights are stamped with the test state, never a checkpoint state.
    assert_eq!(population.checkpoint_sha256(), UNBOUND_TEST_STATE_SHA256);
    assert_eq!(
        proposal.ranking().bindings().checkpoint_sha256(),
        UNBOUND_TEST_STATE_SHA256
    );
    let origins = population.origins();
    assert_eq!(
        origins.get(ORIGIN_NEURAL).copied().unwrap_or(0)
            + origins.get(ORIGIN_NEURAL_FALLBACK).copied().unwrap_or(0),
        64
    );
    assert_eq!(
        origins.get(ORIGIN_NEURAL).copied().unwrap_or(0),
        proposal.pool().unique_neural
    );
    assert_eq!(origins[ORIGIN_RANDOM], 16);
    assert!(population.candidates()[..64]
        .iter()
        .all(|c| c.origin() != ORIGIN_RANDOM && c.rank_log_weight().is_some()));
    assert!(population.candidates()[64..]
        .iter()
        .all(|c| c.origin() == ORIGIN_RANDOM && c.rank_log_weight().is_none()));
    // Seed order is ranking order; weights are a normalized log-softmax of predicted quality.
    for (candidate, p) in population
        .candidates()
        .iter()
        .zip(proposal.ranking().selected())
    {
        assert_eq!(candidate.genome(), p.program);
    }
    let total = population.candidates()[..64]
        .iter()
        .map(|c| c.rank_log_weight().unwrap().exp())
        .sum::<f64>();
    assert!((total - 1.0).abs() < 1e-9);

    let dir = std::env::temp_dir().join(format!("jope-dime-seeds-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("seeds.json");
    let sha = write_population(&path, &proposal).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(sha, sha256_hex(&bytes));
    assert_eq!(bytes, serde_json::to_vec_pretty(population).unwrap());
    assert_eq!(bytes, seeds::population_bytes(&proposal).unwrap());
    assert_gp_seed_shape(&serde_json::from_slice(&bytes).unwrap());

    let admitted = validate_for_search(
        &bytes,
        binding.prepared_sha256(),
        binding.cut(),
        binding.training_sha256(),
    )
    .unwrap();
    assert_eq!(&admitted, population);
    assert!(admitted.claims_neural() == (proposal.pool().unique_neural > 0));
    // Wrong run identity is the same refusal jope-dime's search gives.
    for (prepared, cut, training) in [
        (
            digest("other prepared.json"),
            0,
            binding.training_sha256().to_string(),
        ),
        (
            binding.prepared_sha256().to_string(),
            1,
            binding.training_sha256().to_string(),
        ),
        (
            binding.prepared_sha256().to_string(),
            0,
            digest("other train"),
        ),
    ] {
        let err = validate_for_search(&bytes, &prepared, cut, &training).unwrap_err();
        assert!(format!("{err:#}").contains("identity mismatch"), "{err:#}");
    }
    // Legacy `policy_logprob` spelling is read; a file with both spellings is refused.
    let mut legacy: Value = serde_json::from_slice(&bytes).unwrap();
    let weight = legacy["candidates"][0]["rank_log_weight"].take();
    legacy["candidates"][0]
        .as_object_mut()
        .unwrap()
        .remove("rank_log_weight");
    legacy["candidates"][0]["policy_logprob"] = weight.clone();
    let legacy_bytes = serde_json::to_vec(&legacy).unwrap();
    let admit = |bytes: &[u8]| {
        validate_for_search(
            bytes,
            binding.prepared_sha256(),
            0,
            binding.training_sha256(),
        )
    };
    assert!(admit(&legacy_bytes).is_ok());
    legacy["candidates"][0]["rank_log_weight"] = weight;
    let both = serde_json::to_vec(&legacy).unwrap();
    assert!(admit(&both).is_err());
    // Count, duplicates and the rejected checkpoint are refused.
    let original: Value = serde_json::from_slice(&bytes).unwrap();
    let mut short = original.clone();
    short["candidates"].as_array_mut().unwrap().pop();
    assert!(admit(&serde_json::to_vec(&short).unwrap()).is_err());
    let mut duplicate = original.clone();
    duplicate["candidates"][79]["genome"] = original["candidates"][0]["genome"].clone();
    assert!(admit(&serde_json::to_vec(&duplicate).unwrap()).is_err());
    let mut rejected = original.clone();
    rejected["checkpoint_sha256"] = json!(REJECTED_STATE_SHA256);
    assert!(admit(&serde_json::to_vec(&rejected).unwrap()).is_err());
    assert!(admit(&serde_json::to_vec(&original).unwrap()).is_ok());
}

#[test]
fn propose_refuses_unbound_inputs() {
    let vocab = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let product = bind_vocabulary(&product_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let model = tiny_model(8);
    let ctx = context();
    let checkpoint = tiny_checkpoint();
    let options = SamplingOptions {
        rounds: 1,
        batch: 4,
        temperature: 1.0,
    };
    let seeds = ProposalSeeds::derive(1);
    // Malformed run identity, wrong context, other vocabulary binding.
    assert!(SeedBinding::new("not-a-digest", 0, &digest("t"), &context_sha256(&ctx)).is_err());
    let bad = SeedBinding::new(&digest("p"), 0, &digest("t"), &digest("other context")).unwrap();
    assert!(propose_unbound(&model, &checkpoint, &vocab, &ctx, &bad, &seeds, &options).is_err());
    let other_vocab =
        CheckpointBinding::unbound_for_testing(tiny_config(), &digest("other vocabulary"));
    for vocabulary in [&vocab, &product] {
        assert!(propose_unbound(
            &model,
            &other_vocab,
            vocabulary,
            &ctx,
            &run_binding(&ctx),
            &seeds,
            &options
        )
        .is_err());
    }
    let other_config = CheckpointBinding::unbound_for_testing(
        PolicyConfig::regression(401),
        CHECKPOINT_VOCABULARY_SHA256,
    );
    assert!(propose_unbound(
        &model,
        &other_config,
        &vocab,
        &ctx,
        &run_binding(&ctx),
        &seeds,
        &options
    )
    .is_err());
    // The production path refuses untrained weights outright, even with every other input valid.
    let err = propose(
        &model,
        &checkpoint,
        &vocab,
        &ctx,
        &run_binding(&ctx),
        &seeds,
        &options,
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("load_bound"), "{err:#}");
    // The documented product restriction is accepted.
    let proposal = propose_unbound(
        &model,
        &checkpoint,
        &product,
        &ctx,
        &run_binding(&ctx),
        &seeds,
        &options,
    )
    .unwrap();
    assert_eq!(proposal.population().candidates().len(), 80);
    assert_eq!(
        proposal.ranking().bindings().vocabulary_sha256(),
        PRODUCT_VOCABULARY_SHA256
    );
    // Assembling with a pool that differs from the ranking is refused.
    let binding = run_binding(&ctx);
    let state = UNBOUND_TEST_STATE_SHA256;
    let pool = proposal.pool().candidates();
    let programs = pool
        .iter()
        .map(|c| c.genome().to_string())
        .collect::<Vec<_>>();
    let mut seen = programs.iter().cloned().collect::<BTreeSet<_>>();
    let tail = decode::random_exploration(&base_vocab(), "regression", 77, &mut seen).unwrap();
    let ranking = fake_ranking(state, &binding, &programs);
    assert!(assemble_for_testing(&binding, state, &ranking, pool, &tail).is_ok());
    let mut other = programs.clone();
    other[3] = tail[0].genome().to_string();
    let other = fake_ranking(state, &binding, &other);
    assert!(assemble_for_testing(&binding, state, &other, pool, &tail).is_err());
    // A tail holding neural-arm candidates, or a ranking for another state, is refused.
    assert!(assemble_for_testing(&binding, state, &ranking, pool, &pool[..16]).is_err());
    let elsewhere = fake_ranking(&digest("other state"), &binding, &programs);
    assert!(assemble_for_testing(&binding, state, &elsewhere, pool, &tail).is_err());
    // Even test support never stamps the accepted state, whatever the ranking claims.
    assert!(assemble_for_testing(&binding, ACCEPTED_STATE_SHA256, &ranking, pool, &tail).is_err());
    let mut claimed = serde_json::to_value(&ranking).unwrap();
    claimed["bindings"]["checkpoint_sha256"] = json!(ACCEPTED_STATE_SHA256);
    let claimed: Ranking = serde_json::from_value(claimed).unwrap();
    for checkpoint in [ACCEPTED_STATE_SHA256, state] {
        let err = assemble_for_testing(&binding, checkpoint, &claimed, pool, &tail).unwrap_err();
        assert!(format!("{err:#}").contains("accepted state"), "{err:#}");
    }
}

/// Run identity for a production caller; the accepted state comes from the checkpoint binding.
pub(crate) fn accepted_run_binding(ctx: &[f32]) -> SeedBinding {
    run_binding(ctx)
}

/// Write deterministic tiny weights to `dir/policy.safetensors` and return a receipt that is
/// internally consistent with them (inventory, parameter hash, file hash) and names the
/// accepted state: everything a forged export could attest to.
pub(crate) fn self_attested_tiny_export(dir: &Path) -> (PathBuf, Value) {
    let config = tiny_config();
    let tensors = tiny_tensors(0.0);
    let path = dir.join("policy.safetensors");
    candle_core::safetensors::save(&tensors, &path).unwrap();
    let loaded = PolicyModel::load_safetensors(&path, config.clone(), cpu()).unwrap();
    assert!(loaded.bound_identity().is_none());
    let parameters = loaded.parameter_sha256().unwrap();
    let inventory = serde_json::to_value(loaded.tensor_inventory().unwrap()).unwrap();
    let mut receipt =
        synthetic_receipt(ACCEPTED_STATE_SHA256, ACCEPTED_MODEL_SHA256, ACCEPTED_STEP);
    receipt["checkpoint"]["config"] = serde_json::to_value(&config).unwrap();
    receipt["checkpoint"]["parameter_sha256"] = json!(parameters);
    receipt["checkpoint"]["parameter_sha256_recomputed"] = json!(parameters);
    receipt["policy"]["tensor_count"] = json!(inventory.as_array().unwrap().len());
    receipt["policy"]["tensors"] = inventory;
    receipt["policy"]["output"]["sha256"] = json!(sha256_hex(&std::fs::read(&path).unwrap()));
    (path, receipt)
}

#[test]
fn propose_requires_the_accepted_weights_loaded_through_load_bound() {
    let vocab = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let ctx = context();
    let options = SamplingOptions {
        rounds: 1,
        batch: 4,
        temperature: 1.0,
    };
    let seeds = ProposalSeeds::derive(3);
    // A random-weight model with a real (parsed, accepted-state) binding is refused.
    let accepted = parse_receipt(&accepted_receipt()).unwrap();
    let random = PolicyModel::random_weights(PolicyConfig::regression(401), 11).unwrap();
    assert!(random.bound_identity().is_none());
    let err = propose(
        &random,
        &accepted,
        &vocab,
        &ctx,
        &accepted_run_binding(&ctx),
        &seeds,
        &options,
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("load_bound"), "{err:#}");
    // The unbound test path refuses the accepted binding, so it never stamps the accepted state.
    assert!(propose_unbound(
        &random,
        &accepted,
        &vocab,
        &ctx,
        &accepted_run_binding(&ctx),
        &seeds,
        &options,
    )
    .is_err());

    // A self-consistent receipt around other weights passes load_bound, but the recomputed
    // parameter hash is not the accepted one, so propose still refuses.
    let dir = std::env::temp_dir().join(format!("jope-dime-forged-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (path, receipt) = self_attested_tiny_export(&dir);
    let forged = parse_receipt(&receipt).unwrap();
    let bound = PolicyModel::load_bound(&path, &forged, cpu()).unwrap();
    let other_file =
        CheckpointBinding::unbound_for_testing(tiny_config(), CHECKPOINT_VOCABULARY_SHA256);
    assert!(PolicyModel::load_bound(&path, &other_file, cpu()).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
    let identity = bound.bound_identity().unwrap();
    assert_eq!(identity.state_sha256(), ACCEPTED_STATE_SHA256);
    assert_eq!(identity.parameter_sha256(), forged.parameter_sha256());
    assert_ne!(identity.parameter_sha256(), ACCEPTED_PARAMETER_SHA256);
    let err = propose(
        &bound,
        &forged,
        &vocab,
        &ctx,
        &accepted_run_binding(&ctx),
        &seeds,
        &options,
    )
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("accepted step-60000 parameters"),
        "{err:#}"
    );
    // A bound model is never admitted by the unbound test path either.
    assert!(propose_unbound(
        &bound,
        &tiny_checkpoint(),
        &vocab,
        &ctx,
        &run_binding(&ctx),
        &seeds,
        &options,
    )
    .is_err());
    // The same forged-but-bound model cannot label a pool as neural or stamp a ranking with the
    // accepted state, and is never scored under another state either.
    let parameters_err = |result: anyhow::Result<()>| {
        let err = result.unwrap_err();
        assert!(
            format!("{err:#}").contains("accepted step-60000 parameters"),
            "{err:#}"
        );
    };
    let base = vocab.vocab();
    parameters_err(decode::pure_pool(&bound, base, &ctx, "regression", 1, 2, &options).map(drop));
    parameters_err(Bindings::for_accepted_policy(&bound, &context_sha256(&ctx), &vocab).map(drop));
    let programs = uniform_pool(base, 4, 9);
    let four = Options {
        batch_size: 4,
        keep: 4,
    };
    let claimed: Bindings = serde_json::from_value(json!({
        "checkpoint_sha256": ACCEPTED_STATE_SHA256,
        "conditioning_context_sha256": context_sha256(&ctx),
        "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
    }))
    .unwrap();
    parameters_err(score_pool(&bound, base, &ctx, &programs, claimed, &four).map(drop));
    let err = score_pool(&bound, base, &ctx, &programs, bindings_for(&ctx), &four).unwrap_err();
    assert!(format!("{err:#}").contains("scored under"), "{err:#}");
    // The untrained twin of that model scores fine under the test state.
    assert!(score_pool(
        &tiny_model(9),
        base,
        &ctx,
        &programs,
        bindings_for(&ctx),
        &four
    )
    .is_ok());
}
