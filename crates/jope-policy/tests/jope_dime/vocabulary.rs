use super::*;

// ---------------------------------------------------------------- vocabulary and grammar

#[test]
fn committed_vocabularies_are_the_documented_bytes() {
    assert_eq!(sha256_hex(&base_bytes()), CHECKPOINT_VOCABULARY_SHA256);
    assert_eq!(sha256_hex(&product_bytes()), PRODUCT_VOCABULARY_SHA256);
    let vocab = base_vocab();
    assert_eq!(vocab.tokens.len(), 401);
    assert_eq!(vocab.engine_ids["featmath"], 3);
    assert_eq!(vocab.task_ids["regression"], 397);
    assert_eq!(vocab.append_ids, [50, 51, 52]);
    assert_eq!(
        vocab.proposal_prefix("regression").unwrap(),
        vec![1, 3, 397]
    );
    assert!(vocab.task_id("binary").is_err());
}

#[test]
fn product_restriction_is_token_compatible_and_bound() {
    let bound = bind_vocabulary(&product_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    assert_eq!(
        *bound.identity(),
        VocabularyIdentity::ProductRestriction {
            base_sha256: CHECKPOINT_VOCABULARY_SHA256.into()
        }
    );
    verify_restriction(&base_vocab(), bound.vocab()).unwrap();
    // Removed heads are no longer encodable; kept programs encode to identical tokens.
    let base = base_vocab();
    assert!(base.encode("(pipe X (ridge_gcv))").is_ok());
    assert!(bound.vocab().encode("(pipe X (ridge_gcv))").is_err());
    for program in GENERATION_11_PROGRAMS {
        assert_eq!(
            base.encode(program).unwrap(),
            bound.vocab().encode(program).unwrap()
        );
    }
    // A restriction whose tables differ from the base fails the full proof.
    let mut tampered = bound.vocab().clone();
    tampered.tokens[60] = "OP=featmath:other".into();
    assert!(verify_restriction(&base, &tampered).is_err());
    // Extra engine table keys are refused with an error, not a panic.
    let mut extra = bound.vocab().clone();
    extra.engine_transforms.insert("phantom".into(), Vec::new());
    assert!(verify_restriction(&base, &extra).is_err());
    let mut extra = bound.vocab().clone();
    extra.engine_heads.insert("phantom".into(), Vec::new());
    assert!(verify_restriction(&base, &extra).is_err());
    let mut missing = bound.vocab().clone();
    missing.engine_heads.clear();
    assert!(verify_restriction(&base, &missing).is_err());
}

#[test]
fn wrong_vocabulary_is_refused() {
    // Bound to a different checkpoint vocabulary.
    let other = digest("another checkpoint vocabulary");
    assert!(bind_vocabulary(&base_bytes(), &other).is_err());
    assert!(bind_vocabulary(&product_bytes(), &other).is_err());
    // One changed byte (still valid JSON) is a different, undocumented vocabulary.
    let mut edited = base_bytes();
    edited.push(b'\n');
    let err = bind_vocabulary(&edited, CHECKPOINT_VOCABULARY_SHA256).unwrap_err();
    assert!(format!("{err:#}").contains("refused"), "{err:#}");
    let mut edited = product_bytes();
    edited.push(b'\n');
    assert!(bind_vocabulary(&edited, CHECKPOINT_VOCABULARY_SHA256).is_err());
    // The product file is not itself a checkpoint vocabulary.
    assert!(bind_vocabulary(&product_bytes(), PRODUCT_VOCABULARY_SHA256).is_err());
}

pub(crate) fn assert_refused<T>(result: anyhow::Result<T>, needle: &str) {
    match result {
        Ok(_) => panic!("expected a refusal containing {needle:?}"),
        Err(err) => assert!(format!("{err:#}").contains(needle), "{err:#}"),
    }
}

#[test]
fn accepted_config_is_pinned() {
    // The config.json layout of PolicyConfig::regression(401) is the file state.json binds.
    assert_eq!(
        PolicyConfig::regression(401).sha256().unwrap(),
        ACCEPTED_CONFIG_SHA256
    );
    let receipt = accepted_receipt();
    parse_receipt(&receipt)
        .unwrap()
        .require_accepted_identity()
        .unwrap();
    // State alone is not enough: step, source model, and parameters are part of the accepted
    // parent identity and must not be relabelled independently.
    let mut wrong_step = receipt.clone();
    wrong_step["checkpoint"]["step"] = json!(59999);
    assert_refused(
        parse_receipt(&wrong_step)
            .unwrap()
            .require_accepted_identity(),
        "checkpoint step",
    );
    let mut wrong_model = receipt.clone();
    wrong_model["checkpoint"]["model_sha256"] = json!(digest("other model"));
    assert_refused(
        parse_receipt(&wrong_model)
            .unwrap()
            .require_accepted_identity(),
        "model_sha256",
    );
    let mut wrong_parameters = receipt.clone();
    wrong_parameters["checkpoint"]["parameter_sha256"] = json!(digest("other parameters"));
    wrong_parameters["checkpoint"]["parameter_sha256_recomputed"] =
        wrong_parameters["checkpoint"]["parameter_sha256"].clone();
    assert_refused(
        parse_receipt(&wrong_parameters)
            .unwrap()
            .require_accepted_identity(),
        "checkpoint parameters",
    );
    // A receipt claiming another config hash is refused, even around the accepted config.
    let mut claimed = receipt.clone();
    claimed["checkpoint"]["config_sha256"] = json!(digest("config"));
    assert_refused(
        parse_receipt(&claimed).unwrap().require_accepted_identity(),
        "config_sha256",
    );
    // A forged config under the accepted claim is refused by the recomputed hash, whether the
    // change keeps every tensor shape (rms, dropout) or not (heads).
    for (field, value) in [
        ("dataset_rms", json!(2.0)),
        ("target_rms", json!(0.5)),
        ("dropout", json!(0.0)),
        ("heads", json!(8)),
    ] {
        let mut forged = receipt.clone();
        forged["checkpoint"]["config"][field] = value;
        let binding = parse_receipt(&forged).unwrap();
        assert_eq!(binding.config_sha256(), ACCEPTED_CONFIG_SHA256);
        assert_refused(binding.require_accepted_identity(), "config hashes to");
    }
    // Another carried vocabulary label is refused.
    let mut relabelled = receipt;
    relabelled["checkpoint"]["carried_bindings"]["vocabulary_sha256"] =
        json!(PRODUCT_VOCABULARY_SHA256);
    assert_refused(
        parse_receipt(&relabelled)
            .unwrap()
            .require_accepted_identity(),
        "vocabulary",
    );
}

/// The released `policy.safetensors` hash is the release decision; the receipt's own
/// `policy.output.sha256` is only the exporter's claim. Every other accepted binding (state,
/// step, model, parameters, config, vocabulary) is present in the relabelled receipts.
#[test]
fn accepted_identity_requires_the_released_weights_file() {
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/bundles/policy-step-60000.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let released = manifest["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["path"] == "policy.safetensors")
        .unwrap();
    assert_eq!(released["sha256"], RELEASED_WEIGHTS_SHA256);

    let receipt = accepted_receipt();
    parse_receipt(&receipt)
        .unwrap()
        .require_accepted_identity()
        .unwrap();
    CheckpointBinding::for_proposal(&serde_json::to_vec(&receipt).unwrap(), None).unwrap();
    for other in [digest("re-serialized accepted tensors"), digest("policy")] {
        let mut relabelled = receipt.clone();
        relabelled["policy"]["output"]["sha256"] = json!(other);
        let binding = parse_receipt(&relabelled).unwrap();
        assert_eq!(binding.parameter_sha256(), ACCEPTED_PARAMETER_SHA256);
        assert_refused(binding.require_accepted_identity(), "policy.safetensors");
        assert_refused(
            CheckpointBinding::for_proposal(&serde_json::to_vec(&relabelled).unwrap(), None),
            "policy.safetensors",
        );
    }
}

#[test]
fn vocabulary_is_judged_by_its_bytes_not_its_label() {
    let bound = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    assert_eq!(bound.sha256(), CHECKPOINT_VOCABULARY_SHA256);
    assert_eq!(*bound.identity(), VocabularyIdentity::Checkpoint);
    // Different bytes bound against their own hash: bind_vocabulary accepts the pair, but the
    // recorded hash is the real one, and proposing with it is refused.
    let mut edited = base_bytes();
    edited.push(b'\n');
    let edited_sha = sha256_hex(&edited);
    let forged = bind_vocabulary(&edited, &edited_sha).unwrap();
    assert_eq!(forged.sha256(), edited_sha);
    assert_eq!(forged.vocab(), bound.vocab());
    let model = tiny_model(8);
    let ctx = context();
    let options = SamplingOptions {
        rounds: 1,
        batch: 4,
        temperature: 1.0,
    };
    let seeds = ProposalSeeds::derive(1);
    // Under the step-60000 checkpoint label the bytes are refused.
    assert_refused(
        propose_unbound(
            &model,
            &tiny_checkpoint(),
            &forged,
            &ctx,
            &run_binding(&ctx),
            &seeds,
            &options,
        ),
        "neither the step-60000 vocabulary",
    );
    // A checkpoint that carries the forged label is refused before the vocabulary is used.
    let relabelled = CheckpointBinding::unbound_for_testing(tiny_config(), &edited_sha);
    assert_refused(
        propose_unbound(
            &model,
            &relabelled,
            &forged,
            &ctx,
            &run_binding(&ctx),
            &seeds,
            &options,
        ),
        "is not the step-60000 vocabulary",
    );
    // So is a checkpoint carrying the product hash as its own vocabulary, with either file.
    let product_labelled =
        CheckpointBinding::unbound_for_testing(tiny_config(), PRODUCT_VOCABULARY_SHA256);
    let product = bind_vocabulary(&product_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    for vocabulary in [&bound, &product] {
        assert_refused(
            propose_unbound(
                &model,
                &product_labelled,
                vocabulary,
                &ctx,
                &run_binding(&ctx),
                &seeds,
                &options,
            ),
            "is not the step-60000 vocabulary",
        );
    }
    // The genuine bytes pass the same path.
    assert!(propose_unbound(
        &model,
        &tiny_checkpoint(),
        &bound,
        &ctx,
        &run_binding(&ctx),
        &seeds,
        &options,
    )
    .is_ok());
}

#[test]
fn grammar_round_trips_programs() {
    let vocab = base_vocab();
    for program in GENERATION_11_PROGRAMS {
        let tokens = vocab.encode(program).unwrap();
        assert_eq!(tokens[..3], [BOS, 3, 397]);
        assert_eq!(tokens.last(), Some(&EOS));
        assert_eq!(&vocab.decode(&tokens).unwrap(), program);
        let legal = vocab.replay(&tokens).unwrap();
        for (set, token) in legal.iter().zip(&tokens[1..]) {
            assert!(set.contains(token));
        }
    }
    // Grammar-uniform programs: decode(encode(p)) == p and encode(decode(t)) == t.
    let mut checked = 0;
    for seed in 0..300u64 {
        let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(seed);
        if let Some(program) =
            decode::grammar_uniform_program(&vocab, "regression", &mut rng).unwrap()
        {
            let tokens = vocab.encode(&program).unwrap();
            assert_eq!(vocab.decode(&tokens).unwrap(), program);
            feat_math::GenomeIr::parse(&program).unwrap();
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} uniform programs terminated");
    // Programs outside the frozen vocabulary are refused rather than snapped.
    assert!(vocab.encode("(pipe X (ridge alpha=0.123456))").is_err());
    assert!(vocab.encode("(pipe X (not_an_op))").is_err());
    assert!(vocab
        .project_proposal("(pipe X (ridge alpha=0.123456))")
        .is_ok());
    assert!(vocab.decode(&[BOS, 3, 397, EOS]).is_err());
}
