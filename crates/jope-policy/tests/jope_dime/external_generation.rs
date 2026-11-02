use super::*;

/// A native challenger export (plan §4.5 DEMO-3) loads only as a challenger. The identity is
/// taken from independent sources, never from the export receipt: the `state.json` file hash,
/// the parameter and config hashes that `state.json` records, and the weights' file hash.
#[test]
#[ignore = "needs JOPE_DIME_CHALLENGER_SAFETENSORS, JOPE_DIME_CHALLENGER_RECEIPT and JOPE_DIME_CHALLENGER_STATE"]
fn native_challenger_export_loads_as_a_challenger() {
    let policy = env_path("JOPE_DIME_CHALLENGER_SAFETENSORS");
    let receipt = std::fs::read(env_path("JOPE_DIME_CHALLENGER_RECEIPT")).unwrap();
    let state_bytes = std::fs::read(env_path("JOPE_DIME_CHALLENGER_STATE")).unwrap();
    let state: Value = serde_json::from_slice(&state_bytes).unwrap();
    let identity = ChallengerIdentity {
        state_sha256: sha256_hex(&state_bytes),
        parameter_sha256: state["parameter_sha256"].as_str().unwrap().into(),
        config_sha256: state["config_sha256"].as_str().unwrap().into(),
        policy_safetensors_sha256: sha256_hex(&std::fs::read(&policy).unwrap()),
    };
    let binding = CheckpointBinding::from_challenger_receipt_bytes(&receipt, &identity).unwrap();
    // Name/shape/dtype set of the receipt equals the port's step-60000 inventory.
    let receipt_set = binding
        .tensors()
        .iter()
        .map(|t| (t.name.clone(), t.shape.clone(), t.dtype.clone()))
        .collect::<BTreeSet<_>>();
    let port_set = expected_inventory(&PolicyConfig::regression(401))
        .into_iter()
        .map(|(n, s)| (n, s, "F32".to_string()))
        .collect::<BTreeSet<_>>();
    assert_eq!(receipt_set, port_set);
    // The Candle recomputation equals the training writer's parameter hash at this step.
    let model = PolicyModel::load_bound(&policy, &binding, cpu()).unwrap();
    let bound = model.bound_identity().unwrap();
    assert_eq!(bound.state_sha256(), identity.state_sha256);
    assert_eq!(
        bound.parameter_sha256(),
        state["parameter_sha256"].as_str().unwrap()
    );
    // It never passes for the accepted policy, and the accepted parser refuses its receipt.
    let vocab = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    assert!(Bindings::for_accepted_policy(&model, &context_sha256(&context()), &vocab).is_err());
    assert!(CheckpointBinding::from_receipt_bytes(&receipt).is_err());
}

#[test]
#[ignore = "needs JOPE_DIME_GENERATION_000011 (jope-dime results directory)"]
fn generation_000011_programs_round_trip_and_rankings_follow_the_tie_break() {
    let root = env_path("JOPE_DIME_GENERATION_000011");
    let vocab = base_vocab();
    let product = bind_vocabulary(&product_bytes(), CHECKPOINT_VOCABULARY_SHA256)
        .unwrap()
        .vocab()
        .clone();
    let mut files = Vec::new();
    for dataset in std::fs::read_dir(root.join("neural")).unwrap() {
        let dataset = dataset.unwrap().path();
        if !dataset.is_dir() {
            continue;
        }
        for cut in std::fs::read_dir(&dataset).unwrap() {
            let cut = cut.unwrap().path();
            if cut.join("seeds.json").is_file() {
                files.push(cut);
            }
        }
    }
    files.sort();
    assert_eq!(
        files.len(),
        140,
        "generation-000011 has 140 neural seed files"
    );
    let (mut neural, mut product_encodable, mut others, mut others_encodable) = (0, 0, 0, 0);
    for dir in &files {
        let seeds: Value =
            serde_json::from_slice(&std::fs::read(dir.join("seeds.json")).unwrap()).unwrap();
        assert_eq!(seeds["schema"], "parallel.seed_population.v1");
        assert_eq!(seeds["checkpoint_sha256"], ACCEPTED_STATE_SHA256);
        let candidates: Vec<SeedCandidate> =
            serde_json::from_value(seeds["candidates"].clone()).unwrap();
        assert_eq!(candidates.len(), 80);
        for c in &candidates {
            if c.origin() == ORIGIN_NEURAL {
                neural += 1;
                let tokens = vocab
                    .encode(c.genome())
                    .unwrap_or_else(|e| panic!("{}: {e:#}", c.genome()));
                assert_eq!(vocab.decode(&tokens).unwrap(), c.genome());
                feat_math::GenomeIr::parse(c.genome()).unwrap();
                if product.encode(c.genome()).is_ok() {
                    product_encodable += 1;
                }
            } else {
                others += 1;
                if let Ok(tokens) = vocab.encode(c.genome()) {
                    assert_eq!(vocab.decode(&tokens).unwrap(), c.genome());
                    others_encodable += 1;
                }
            }
        }
        let ranking: Ranking =
            serde_json::from_slice(&std::fs::read(dir.join("ranking.json")).unwrap()).unwrap();
        assert_eq!(ranking.selection_id(), SELECTION_ID);
        assert_eq!(
            ranking.bindings().checkpoint_sha256(),
            ACCEPTED_STATE_SHA256
        );
        assert_eq!(
            ranking.bindings().vocabulary_sha256(),
            CHECKPOINT_VOCABULARY_SHA256
        );
        assert_eq!(
            ranking.bindings().conditioning_context_sha256(),
            seeds["context_sha256"]
        );
        let programs = ranking
            .ranked()
            .iter()
            .map(|p| p.program.clone())
            .collect::<Vec<_>>();
        assert_eq!(ranking.pool_sha256(), pool_sha256(&programs).unwrap());
        assert_eq!(
            sorted_predictions(ranking.ranked().to_vec()).unwrap(),
            ranking.ranked()
        );
        for p in ranking.ranked() {
            assert_eq!(p.program_sha256, sha256_hex(p.program.as_bytes()));
            assert_eq!(p.token_count, vocab.encode(&p.program).unwrap().len());
        }
        let seeded = candidates[..64]
            .iter()
            .map(|c| c.genome().to_string())
            .collect::<Vec<_>>();
        assert_eq!(seeded, programs[..64]);
    }
    assert_eq!(neural, 140 * 64);
    println!(
        "generation-000011: {} files, {neural} neural programs round-tripped ({product_encodable} also in featmath-jain-v1), {others_encodable}/{others} other seeds encodable",
        files.len()
    );
}

/// Cross-precision diagnostic, not parity: re-score each generation-000011 pool with the
/// exported weights on CPU float32 and compare with the reward heads jope-dime recorded under
/// CUDA BF16 autocast. BF16 rounding bounds how close these can be, so the bound is loose; the
/// atol 3e-4 parity gate still needs float32 reference fixtures (plan v2 W4 PR-B).
#[test]
#[ignore = "needs JOPE_DIME_POLICY_SAFETENSORS, JOPE_DIME_GENERATION_000011 and JOPE_DIME_CONTEXTS"]
fn generation_000011_rewards_agree_with_bf16_records() {
    let policy = env_path("JOPE_DIME_POLICY_SAFETENSORS");
    let receipt = policy
        .parent()
        .and_then(Path::parent)
        .unwrap()
        .join("step-60000.receipt.json");
    let receipt = std::env::var_os("JOPE_DIME_EXPORT_RECEIPT")
        .map(PathBuf::from)
        .unwrap_or(receipt);
    let binding = CheckpointBinding::from_receipt_bytes(&std::fs::read(receipt).unwrap()).unwrap();
    let model = PolicyModel::load_bound(&policy, &binding, cpu()).unwrap();
    let bound_vocab = bind_vocabulary(&base_bytes(), binding.vocabulary_sha256()).unwrap();
    let vocab = bound_vocab.vocab().clone();
    let root = env_path("JOPE_DIME_GENERATION_000011");
    let contexts = env_path("JOPE_DIME_CONTEXTS");
    let (mut pools, mut max_head, mut max_quality, mut same_top, mut kendall_sum) =
        (0usize, 0f64, 0f64, 0usize, 0f64);
    let uniform = uniform_pool(&vocab, 64, 4242);
    let (mut decoder_pools, mut neural_lp, mut uniform_lp) = (0usize, 0f64, 0f64);
    for dataset in std::fs::read_dir(root.join("neural")).unwrap() {
        let dataset = dataset.unwrap().path();
        if !dataset.is_dir() {
            continue;
        }
        for cut in std::fs::read_dir(&dataset).unwrap() {
            let cut = cut.unwrap().path();
            let Ok(bytes) = std::fs::read(cut.join("ranking.json")) else {
                continue;
            };
            let recorded: Ranking = serde_json::from_slice(&bytes).unwrap();
            let context_file = contexts.join(dataset.file_name().unwrap()).join(format!(
                "{}.f32",
                cut.file_name().unwrap().to_string_lossy()
            ));
            let Ok(context_bytes) = std::fs::read(&context_file) else {
                continue;
            };
            let ctx = jope_policy::context_from_bytes(&context_bytes).unwrap();
            assert_eq!(
                context_sha256(&ctx),
                recorded.bindings().conditioning_context_sha256()
            );
            let programs = recorded
                .ranked()
                .iter()
                .map(|p| p.program.clone())
                .collect::<Vec<_>>();
            let ours = score_pool(
                &model,
                &vocab,
                &ctx,
                &programs,
                {
                    // Deserialized bindings carry no vocabulary table, so re-derive them from the
                    // bound model and vocabulary and check they name what was recorded.
                    let ours = Bindings::for_accepted_policy(
                        &model,
                        recorded.bindings().conditioning_context_sha256(),
                        &bound_vocab,
                    )
                    .unwrap();
                    assert_eq!(
                        serde_json::to_value(&ours).unwrap(),
                        serde_json::to_value(recorded.bindings()).unwrap()
                    );
                    ours
                },
                &Options::default(),
            )
            .unwrap();
            assert_eq!(ours.pool_sha256(), recorded.pool_sha256());
            let by_program = ours
                .ranked()
                .iter()
                .map(|p| (p.program.as_str(), p))
                .collect::<std::collections::BTreeMap<_, _>>();
            for r in recorded.ranked() {
                let o = by_program[r.program.as_str()];
                for (a, b) in o.reward_heads.iter().zip(&r.reward_heads) {
                    max_head = max_head.max((a - b).abs());
                }
                max_quality = max_quality.max((o.predicted_quality - r.predicted_quality).abs());
            }
            let rank = |p: &str| ours.ranked().iter().position(|x| x.program == p).unwrap();
            let order = recorded
                .ranked()
                .iter()
                .map(|p| rank(&p.program))
                .collect::<Vec<_>>();
            let (mut concordant, mut total) = (0usize, 0usize);
            for i in 0..order.len() {
                for j in i + 1..order.len() {
                    total += 1;
                    if order[i] < order[j] {
                        concordant += 1;
                    }
                }
            }
            kendall_sum += 2.0 * concordant as f64 / total as f64 - 1.0;
            if decoder_pools < 6 {
                neural_lp += mean_token_logprob(&model, &vocab, &ctx, &programs);
                uniform_lp += mean_token_logprob(&model, &vocab, &ctx, &uniform);
                decoder_pools += 1;
            }
            same_top += usize::from(order[0] == 0);
            pools += 1;
        }
    }
    assert!(
        pools > 0,
        "no generation-000011 pool had its context under JOPE_DIME_CONTEXTS"
    );
    println!(
        "bf16-vs-f32 diagnostic: {pools} pools, max |head diff| {max_head:.5}, max |quality diff| {max_quality:.5}, mean kendall tau {:.4}, same top-1 {same_top}/{pools}",
        kendall_sum / pools as f64
    );
    let (neural_lp, uniform_lp) = (
        neural_lp / decoder_pools as f64,
        uniform_lp / decoder_pools as f64,
    );
    println!(
        "decoder diagnostic: {decoder_pools} contexts, mean legal-token log-prob of recorded step-60000 samples {neural_lp:.4} vs grammar-uniform programs {uniform_lp:.4}"
    );
    assert!(
        neural_lp > uniform_lp + 0.5,
        "decoder does not prefer the checkpoint's own samples"
    );
    assert!(
        max_quality < 0.25,
        "CPU float32 rewards diverge from the BF16 records"
    );
}

/// Mean grammar-masked log-probability per generated token (after the 3-token prefix).
pub(crate) fn mean_token_logprob(
    model: &PolicyModel,
    vocab: &Vocab,
    ctx: &PolicyContext,
    programs: &[String],
) -> f64 {
    let encoded = programs
        .iter()
        .map(|p| vocab.encode(p).unwrap())
        .collect::<Vec<_>>();
    let length = encoded.iter().map(Vec::len).max().unwrap();
    let rows = encoded
        .iter()
        .map(|t| {
            let mut row = t.clone();
            row.resize(length, PAD);
            row
        })
        .collect::<Vec<_>>();
    let logits = model.logits(ctx, &rows).unwrap().to_vec3::<f32>().unwrap();
    let (mut total, mut count) = (0f64, 0usize);
    for (row, tokens) in encoded.iter().enumerate() {
        let legal = vocab.replay(tokens).unwrap();
        // logits[t] predicts tokens[t]; legal[t - 1] is the set offered before tokens[t].
        for t in 3..tokens.len() {
            let set = &legal[t - 1];
            let values = set
                .iter()
                .map(|&i| logits[row][t][i] as f64)
                .collect::<Vec<_>>();
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_z = max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
            total += logits[row][t][tokens[t]] as f64 - log_z;
            count += 1;
        }
    }
    total / count as f64
}
