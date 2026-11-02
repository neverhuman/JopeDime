use super::*;

// ---------------------------------------------------------------- external evidence (ignored)

pub(crate) fn env_path(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("set {name} to run this ignored evidence test")),
    )
}

#[test]
#[ignore = "needs JOPE_DIME_POLICY_SAFETENSORS (exported step-60000 policy)"]
fn exported_policy_matches_its_export_receipt() {
    let policy = env_path("JOPE_DIME_POLICY_SAFETENSORS");
    let receipt_path = std::env::var_os("JOPE_DIME_EXPORT_RECEIPT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            policy
                .parent()
                .and_then(Path::parent)
                .unwrap()
                .join("step-60000.receipt.json")
        });
    let binding =
        CheckpointBinding::from_receipt_bytes(&std::fs::read(&receipt_path).unwrap()).unwrap();
    assert_eq!(binding.state_sha256(), ACCEPTED_STATE_SHA256);
    assert_eq!(binding.parameter_sha256(), ACCEPTED_PARAMETER_SHA256);
    assert_eq!(binding.vocabulary_sha256(), CHECKPOINT_VOCABULARY_SHA256);
    assert_eq!(binding.tensors().len(), 136);
    // Name/shape/dtype set of the receipt equals the port's inventory.
    let receipt_set = binding
        .tensors()
        .iter()
        .map(|t| (t.name.clone(), t.shape.clone(), t.dtype.clone()))
        .collect::<BTreeSet<_>>();
    let port_set = expected_inventory(binding.config())
        .into_iter()
        .map(|(n, s)| (n, s, "F32".to_string()))
        .collect::<BTreeSet<_>>();
    assert_eq!(receipt_set, port_set);
    // Loading proves file hash, per-tensor bytes and the recomputed parameter hash.
    let model = PolicyModel::load_bound(&policy, &binding, cpu()).unwrap();
    assert_eq!(
        model.parameter_sha256().unwrap(),
        binding.parameter_sha256()
    );
    let identity = model.bound_identity().unwrap();
    assert_eq!(identity.parameter_sha256(), ACCEPTED_PARAMETER_SHA256);
    assert_eq!(
        identity.policy_safetensors_sha256(),
        binding.policy_safetensors_sha256()
    );
    let vocab = bind_vocabulary(&product_bytes(), binding.vocabulary_sha256()).unwrap();
    let ctx = PolicyContext::from_frozen_values(vec![0.0f32; 2048]).unwrap();
    let logits = model.next_token_logits(&ctx, &[vec![BOS, 3, 397]]).unwrap();
    assert_eq!(logits[0].len(), 401);
    assert!(logits[0].iter().all(|v| v.is_finite()));
    let program = vocab.vocab().encode(GENERATION_11_PROGRAMS[1]).unwrap();
    let (heads, runtime) = model.reward_runtime(&ctx, &[program]).unwrap();
    assert!(heads[0].iter().all(|v| v.is_finite()) && runtime[0].is_finite());
    // A receipt whose file hash differs is refused.
    let wrong = CheckpointBinding::unbound_for_testing(
        binding.config().clone(),
        CHECKPOINT_VOCABULARY_SHA256,
    );
    assert!(PolicyModel::load_bound(&policy, &wrong, cpu()).is_err());
    // The bound accepted policy is the one model the production proposal path admits.
    let options = SamplingOptions {
        rounds: 1,
        batch: 8,
        temperature: 1.0,
    };
    let proposal = propose(
        &model,
        &binding,
        &vocab,
        &ctx,
        &accepted_run_binding(&ctx),
        &ProposalSeeds::derive(60000),
        &options,
    )
    .unwrap();
    assert_eq!(
        proposal.population().checkpoint_sha256(),
        ACCEPTED_STATE_SHA256
    );
    assert_eq!(proposal.population().candidates().len(), 80);
    assert_eq!(
        proposal.ranking().bindings().vocabulary_sha256(),
        PRODUCT_VOCABULARY_SHA256
    );

    // The accepted config: claimed, recomputed, and (when the checkpoint directory the receipt
    // names is present) the config.json file itself.
    binding.require_accepted_identity().unwrap();
    assert_eq!(binding.config_sha256(), ACCEPTED_CONFIG_SHA256);
    assert_eq!(binding.config().sha256().unwrap(), ACCEPTED_CONFIG_SHA256);
    assert_eq!(
        model.bound_identity().unwrap().config_sha256(),
        ACCEPTED_CONFIG_SHA256
    );
    let receipt_bytes = std::fs::read(&receipt_path).unwrap();
    let receipt: Value = serde_json::from_slice(&receipt_bytes).unwrap();
    let config_file =
        Path::new(receipt["checkpoint"]["directory"].as_str().unwrap()).join("config.json");
    match std::fs::read(&config_file) {
        Ok(bytes) => {
            assert_eq!(sha256_hex(&bytes), ACCEPTED_CONFIG_SHA256);
            assert_eq!(
                PolicyConfig::from_json(&bytes).unwrap().sha256().unwrap(),
                ACCEPTED_CONFIG_SHA256
            );
            println!(
                "config.json {} matches ACCEPTED_CONFIG_SHA256",
                config_file.display()
            );
        }
        Err(err) => println!("config.json not checked ({}: {err})", config_file.display()),
    }

    // Forged receipts around the real weights: load_bound still succeeds (the tensors and the
    // parameter hash are real), but nothing is stamped with the accepted state.
    let run = accepted_run_binding(&ctx);
    let seeds = ProposalSeeds::derive(60000);
    let base = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let forge = |edit: &dyn Fn(&mut Value)| {
        let mut forged = receipt.clone();
        edit(&mut forged);
        let binding =
            CheckpointBinding::from_receipt_bytes(&serde_json::to_vec(&forged).unwrap()).unwrap();
        let model = PolicyModel::load_bound(&policy, &binding, cpu()).unwrap();
        (binding, model)
    };
    // Forged config (same tensor shapes) under the accepted claim.
    let (forged_binding, forged_model) =
        forge(&|r| r["checkpoint"]["config"]["dataset_rms"] = json!(2.0));
    assert_eq!(forged_binding.config_sha256(), ACCEPTED_CONFIG_SHA256);
    assert_refused(
        propose(
            &forged_model,
            &forged_binding,
            &vocab,
            &ctx,
            &run,
            &seeds,
            &options,
        ),
        "config hashes to",
    );
    assert_refused(
        Bindings::for_accepted_policy(&forged_model, &context_sha256(&ctx), &base),
        "config hashes to",
    );
    assert_refused(
        decode::pure_pool(
            &forged_model,
            base.vocab(),
            &ctx,
            "regression",
            1,
            2,
            &options,
        ),
        "config hashes to",
    );
    // Forged config_sha256 claim around the real config.
    let (claimed_binding, claimed_model) =
        forge(&|r| r["checkpoint"]["config_sha256"] = json!(digest("config")));
    assert_refused(
        propose(
            &claimed_model,
            &claimed_binding,
            &vocab,
            &ctx,
            &run,
            &seeds,
            &options,
        ),
        "config_sha256",
    );
    // Checkpoint vocabulary label mismatch.
    for label in [
        PRODUCT_VOCABULARY_SHA256.to_string(),
        digest("other vocabulary"),
    ] {
        let (relabelled, relabelled_model) =
            forge(&|r| r["checkpoint"]["carried_bindings"]["vocabulary_sha256"] = json!(label));
        assert_refused(
            propose(
                &relabelled_model,
                &relabelled,
                &vocab,
                &ctx,
                &run,
                &seeds,
                &options,
            ),
            "vocabulary",
        );
    }
    // Wrong vocabulary bytes with the real receipt, under their own hash.
    let mut edited = base_bytes();
    edited.push(b'\n');
    let wrong = bind_vocabulary(&edited, &sha256_hex(&edited)).unwrap();
    assert_refused(
        propose(&model, &binding, &wrong, &ctx, &run, &seeds, &options),
        "neither the step-60000 vocabulary",
    );
    assert_refused(
        Bindings::for_accepted_policy(&model, &context_sha256(&ctx), &wrong),
        "neither the step-60000 vocabulary",
    );
    // Bindings made for one vocabulary table cannot score with another.
    let programs = uniform_pool(base.vocab(), 4, 9);
    let four = Options {
        batch_size: 4,
        keep: 4,
    };
    let product_bindings =
        Bindings::for_accepted_policy(&model, &context_sha256(&ctx), &vocab).unwrap();
    assert_refused(
        score_pool(
            &model,
            base.vocab(),
            &ctx,
            &programs,
            product_bindings,
            &four,
        ),
        "differs from the bound vocabulary",
    );
    let base_bindings =
        Bindings::for_accepted_policy(&model, &context_sha256(&ctx), &base).unwrap();
    let ranking = score_pool(&model, base.vocab(), &ctx, &programs, base_bindings, &four).unwrap();
    assert_eq!(
        ranking.bindings().checkpoint_sha256(),
        ACCEPTED_STATE_SHA256
    );
    // Deserialized accepted bindings carry no vocabulary table and are refused.
    let claimed: Bindings = serde_json::from_value(json!({
        "checkpoint_sha256": ACCEPTED_STATE_SHA256,
        "conditioning_context_sha256": context_sha256(&ctx),
        "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
    }))
    .unwrap();
    assert_refused(
        score_pool(&model, base.vocab(), &ctx, &programs, claimed, &four),
        "bound step-60000 vocabulary",
    );
}
