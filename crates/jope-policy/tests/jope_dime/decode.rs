use super::*;

// ---------------------------------------------------------------- model and decode

#[test]
fn model_refuses_a_changed_tensor_set() {
    let config = tiny_config();
    let model = tiny_model(1);
    let inventory = model.tensor_inventory().unwrap();
    // One decoder layer: 136 - 5 * 12 tensors.
    assert_eq!(inventory.len(), 76);
    assert_eq!(inventory.len(), expected_inventory(&config).len());
    let tensors = expected_inventory(&config)
        .into_iter()
        .map(|(name, shape)| {
            let t = candle_zeros(&shape);
            (name, t)
        })
        .collect::<Vec<_>>();
    assert!(PolicyModel::from_tensors(config.clone(), tensors.clone(), cpu()).is_ok());
    let mut missing = tensors.clone();
    missing.retain(|(n, _)| n != "reward_encoder.4.bias");
    assert!(PolicyModel::from_tensors(config.clone(), missing, cpu()).is_err());
    let mut extra = tensors.clone();
    extra.push(("runtime_head.4.bias".into(), candle_zeros(&[1])));
    assert!(PolicyModel::from_tensors(config.clone(), extra, cpu()).is_err());
    let mut reshaped = tensors;
    for (name, t) in reshaped.iter_mut() {
        if name == "z_proj.bias" {
            *t = candle_zeros(&[15]);
        }
    }
    assert!(PolicyModel::from_tensors(config, reshaped, cpu()).is_err());
}

pub(crate) fn cpu() -> candle_core::Device {
    candle_core::Device::Cpu
}

pub(crate) fn candle_zeros(shape: &[usize]) -> candle_core::Tensor {
    candle_core::Tensor::zeros(shape, candle_core::DType::F32, &cpu()).unwrap()
}

#[test]
fn decoder_masks_padding_and_future_tokens() {
    let model = tiny_model(2);
    let ctx = context();
    let program = base_vocab().encode(GENERATION_11_PROGRAMS[1]).unwrap();
    // Logits at position t depend only on tokens before t (PAD keys: `decoder_ignores_pad_keys`).
    let full = model
        .logits(&ctx, std::slice::from_ref(&program))
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    let prefix = program[..6].to_vec();
    let short = model
        .logits(&ctx, std::slice::from_ref(&prefix))
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    for t in 0..=prefix.len() {
        for (a, b) in full[0][t].iter().zip(&short[0][t]) {
            assert!((a - b).abs() < 1e-4, "position {t}: {a} vs {b}");
        }
    }
    let mut padded = prefix.clone();
    padded.extend([PAD; 3]);
    let with_pad = model.next_token_logits(&ctx, &[padded]).unwrap();
    assert_eq!(with_pad[0].len(), 401);
    let rewards_plain = model
        .reward_runtime(&ctx, std::slice::from_ref(&program))
        .unwrap();
    let mut longer = program.clone();
    longer.extend([PAD; 5]);
    let rewards_padded = model.reward_runtime(&ctx, &[longer]).unwrap();
    for (a, b) in rewards_plain.0[0].iter().zip(&rewards_padded.0[0]) {
        assert!((a - b).abs() < 1e-5);
    }
    assert!(rewards_plain.1[0] >= 0.0);
}

/// PAD keys are invisible: two models that differ only in the PAD embedding row give the same
/// logits at every position whose query is not PAD, including positions after an interior PAD
/// and positions of right-padded rows (the PAD logit column itself is excluded, because the
/// output projection is tied to `token_emb`).
#[test]
fn decoder_ignores_pad_keys() {
    let plain = PolicyModel::from_tensors(tiny_config(), tiny_tensors(0.0), cpu()).unwrap();
    let shifted = PolicyModel::from_tensors(tiny_config(), tiny_tensors(3.0), cpu()).unwrap();
    let ctx = context();
    let program = base_vocab().encode(GENERATION_11_PROGRAMS[1]).unwrap();
    assert!(program.len() > 8);
    let mut interior = program.clone();
    interior[4] = PAD;
    let mut padded = program[..6].to_vec();
    padded.extend([PAD; 3]);
    // Pad the interior row to the padded row's length (or vice versa) so both share a batch.
    let length = interior.len().max(padded.len());
    interior.resize(length, PAD);
    padded.resize(length, PAD);
    let rows = vec![interior, padded];
    let a = plain.logits(&ctx, &rows).unwrap().to_vec3::<f32>().unwrap();
    let b = shifted
        .logits(&ctx, &rows)
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    let mut pad_query_moved = false;
    for (r, row) in rows.iter().enumerate() {
        // Position t + 1 is the query for token t; position 0 is the context query.
        for t in 0..=row.len() {
            let pad_query = t > 0 && row[t - 1] == PAD;
            let moved = a[r][t][1..]
                .iter()
                .zip(&b[r][t][1..])
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            if pad_query {
                pad_query_moved |= moved > 1e-4;
            } else {
                assert!(moved < 1e-6, "row {r} position {t} moved by {moved}");
            }
        }
    }
    // The shift is large enough to matter wherever PAD is actually read.
    assert!(pad_query_moved);
}

#[test]
fn pure_pool_counts_only_sampled_tokens() {
    let vocab = base_vocab();
    let model = tiny_model(3);
    let ctx = context();
    let options = SamplingOptions {
        rounds: 1,
        batch: 24,
        temperature: 25.0,
    };
    let (pool, _) = pure_pool_unbound(&model, &vocab, &ctx, "regression", 7, 42, &options).unwrap();
    // The first round replays from the same ChaCha20 seed.
    let mut rng = <rand_chacha::ChaCha20Rng as rand::SeedableRng>::seed_from_u64(7);
    let rows =
        decode::sample_batch(&model, &vocab, &ctx, "regression", 24, 25.0, &mut rng).unwrap();
    let drawn = rows
        .iter()
        .map(|row| {
            let end = row.program.as_ref().map_or(row.tokens.len(), Vec::len);
            assert!(row.tokens[3..end].iter().all(|&t| t != PAD));
            assert!(row.tokens[end..].iter().all(|&t| t == PAD));
            end - 3
        })
        .sum::<usize>();
    let padded = rows.iter().map(|row| row.tokens.len() - 3).sum::<usize>();
    assert_eq!(pool.sampled_tokens, drawn);
    // At least one row finished early, so padded positions would have been over-counted.
    assert!(drawn < padded, "{drawn} vs {padded}");
}

#[test]
fn every_sampled_token_is_legal() {
    let vocab = base_vocab();
    let model = tiny_model(3);
    let ctx = context();
    let mut rng = <rand_chacha::ChaCha20Rng as rand::SeedableRng>::seed_from_u64(7);
    let rows = decode::sample_batch(&model, &vocab, &ctx, "regression", 24, 1.0, &mut rng).unwrap();
    assert_eq!(rows.len(), 24);
    let length = rows[0].tokens.len();
    let mut finished = 0;
    for row in &rows {
        assert_eq!(row.tokens.len(), length);
        assert_eq!(row.tokens[..3], [BOS, 3, 397]);
        let mut state = vocab.prefix_state("regression").unwrap();
        let mut ended = false;
        for &token in &row.tokens[3..] {
            if ended {
                assert_eq!(token, PAD, "finished rows are padded");
                continue;
            }
            // Sampling-mode legal set: generate=false operators must never appear.
            assert!(state.legal(&vocab).contains(&token), "illegal {token}");
            state.advance(&vocab, token).unwrap();
            ended = state.done();
        }
        assert_eq!(ended, row.program.is_some());
        if let Some(program) = &row.program {
            finished += 1;
            vocab.replay(program).unwrap();
            if let Some(genome) = decode::valid_genome(&vocab, program) {
                assert_eq!(&vocab.encode(&genome).unwrap(), program);
            }
        }
    }
    assert!(finished > 0);
    // Illegal tokens with overwhelming logits are still never drawn.
    let mut logits = vec![1e30f32; 401];
    let legal = [60usize, 61, 62];
    for i in legal {
        logits[i] = -5.0;
    }
    for _ in 0..200 {
        let token = decode::sample_legal(&logits, &legal, 1.0, &mut rng).unwrap();
        assert!(legal.contains(&token));
    }
    assert!(decode::sample_legal(&logits, &[], 1.0, &mut rng).is_err());
    logits[60] = f32::NAN;
    assert!(decode::sample_legal(&logits, &legal, 1.0, &mut rng).is_err());
}

#[test]
fn pure_pool_is_deterministic_and_honest_about_origins() {
    let vocab = base_vocab();
    let model = tiny_model(4);
    let ctx = context();
    let options = SamplingOptions {
        rounds: 1,
        batch: 16,
        temperature: 1.0,
    };
    let run = || pure_pool_unbound(&model, &vocab, &ctx, "regression", 11, 12, &options).unwrap();
    let (pool, seen) = run();
    let (again, _) = run();
    assert_eq!(pool.candidates(), again.candidates());
    assert_eq!(pool.candidates().len(), 64);
    assert_eq!(pool.attempted, 16);
    assert!(pool.unique_neural <= pool.valid && pool.valid <= 16);
    let neural = pool
        .candidates()
        .iter()
        .filter(|c| c.origin() == ORIGIN_NEURAL)
        .count();
    assert_eq!(neural, pool.unique_neural);
    assert!(pool.candidates()[neural..]
        .iter()
        .all(|c| c.origin() == ORIGIN_NEURAL_FALLBACK));
    assert!(pool.candidates().iter().all(|c| seen.contains(c.genome())));
    // The production pool refuses untrained weights: its samples would be labelled neural.
    let err = decode::pure_pool(&model, &vocab, &ctx, "regression", 11, 12, &options).unwrap_err();
    assert!(format!("{err:#}").contains("load_bound"), "{err:#}");
}
