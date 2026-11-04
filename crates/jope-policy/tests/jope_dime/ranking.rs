use super::*;

// ---------------------------------------------------------------- ranking

pub(crate) fn prediction(program: &str, quality: f64) -> Prediction {
    Prediction {
        program: program.into(),
        program_sha256: sha256_hex(program.as_bytes()),
        token_count: 5,
        predicted_quality: quality,
        predicted_runtime_seconds: 2.0,
        reward_heads: [quality; 4],
    }
}

#[test]
fn ranking_ties_break_by_program_hash_then_text() {
    let mut better = prediction("better", 2.0);
    better.predicted_runtime_seconds = 999.0;
    let a = prediction("a", 1.0);
    let b = prediction("b", 1.0);
    let negative_zero = prediction("nz", -0.0);
    let zero = prediction("z", 0.0);
    let first = sorted_predictions(vec![
        a.clone(),
        better.clone(),
        b.clone(),
        zero.clone(),
        negative_zero.clone(),
    ])
    .unwrap();
    let second = sorted_predictions(vec![negative_zero, b, zero, a, better]).unwrap();
    assert_eq!(first, second);
    assert_eq!(first[0].program, "better");
    assert!(first[1].program_sha256 < first[2].program_sha256);
    assert!(first[3].program_sha256 < first[4].program_sha256);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(sorted_predictions(vec![prediction("bad", value)]).is_err());
        let mut row = prediction("bad runtime", 1.0);
        row.predicted_runtime_seconds = value;
        assert!(sorted_predictions(vec![row]).is_err());
    }
    let mut row = prediction("negative runtime", 1.0);
    row.predicted_runtime_seconds = -1.0;
    assert!(sorted_predictions(vec![row]).is_err());
}

pub(crate) fn uniform_pool(vocab: &Vocab, n: usize, seed: u64) -> Vec<String> {
    decode::uniform_programs(vocab, "regression", seed, n, &mut BTreeSet::new()).unwrap()
}

/// Scoring bindings for untrained weights (the unbound test state).
pub(crate) fn bindings_for(ctx: &[f32]) -> Bindings {
    Bindings::new(
        UNBOUND_TEST_STATE_SHA256,
        &context_sha256(ctx),
        CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap()
}

#[test]
fn score_pool_order_is_deterministic_and_documented() {
    let vocab = base_vocab();
    let model = tiny_model(5);
    let ctx = context();
    let mut programs = uniform_pool(&vocab, 64, 900);
    programs.extend(GENERATION_11_PROGRAMS[1..].iter().map(|p| p.to_string()));
    let options = Options {
        batch_size: 16,
        keep: 64,
    };
    let ranking = score_pool(
        &model,
        &vocab,
        &ctx,
        &programs,
        bindings_for(&ctx),
        &options,
    )
    .unwrap();
    assert_eq!(ranking.selection_id(), SELECTION_ID);
    assert_eq!(ranking.pool_size(), programs.len());
    assert_eq!(ranking.ranked().len(), programs.len());
    assert_eq!(ranking.selected().len(), 64);
    assert_eq!(ranking.scoring_batches(), programs.len().div_ceil(16));
    for pair in ranking.ranked().windows(2) {
        assert_ne!(compare_predictions(&pair[0], &pair[1]), Ordering::Greater);
        if pair[0].predicted_quality == pair[1].predicted_quality {
            assert!(pair[0].program_sha256 < pair[1].program_sha256);
        }
    }
    for p in ranking.ranked() {
        let mean = p.reward_heads.iter().map(|v| *v as f32).sum::<f32>() / 4.0;
        assert_eq!(p.predicted_quality, mean as f64);
        assert_eq!(p.token_count, vocab.encode(&p.program).unwrap().len());
    }
    // Pool order and scoring batch size do not change the result.
    let mut reversed = programs.clone();
    reversed.reverse();
    let again = score_pool(
        &model,
        &vocab,
        &ctx,
        &reversed,
        bindings_for(&ctx),
        &options,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_vec(&ranking).unwrap(),
        serde_json::to_vec(&again).unwrap()
    );
    assert_eq!(ranking.pool_sha256(), pool_sha256(&reversed).unwrap());
    let single = score_pool(
        &model,
        &vocab,
        &ctx,
        &programs,
        bindings_for(&ctx),
        &Options {
            batch_size: 1,
            keep: 64,
        },
    )
    .unwrap();
    let order = |r: &Ranking| {
        r.ranked()
            .iter()
            .map(|p| p.program.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(order(&ranking), order(&single));
    // Wrong context identity, duplicates and unencodable programs are refused.
    let wrong = Bindings::new(
        UNBOUND_TEST_STATE_SHA256,
        &digest("other context"),
        CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap();
    assert!(score_pool(&model, &vocab, &ctx, &programs, wrong, &options).is_err());
    // Untrained weights are never stamped with the accepted state: the constructor refuses it,
    // and bindings that claim it (only deserializable under test support) are refused at scoring.
    assert!(Bindings::new(
        ACCEPTED_STATE_SHA256,
        &context_sha256(&ctx),
        CHECKPOINT_VOCABULARY_SHA256
    )
    .is_err());
    let err = Bindings::for_accepted_policy(
        &model,
        &context_sha256(&ctx),
        &bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap(),
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("load_bound"), "{err:#}");
    let claimed: Bindings = serde_json::from_value(json!({
        "checkpoint_sha256": ACCEPTED_STATE_SHA256,
        "conditioning_context_sha256": context_sha256(&ctx),
        "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
    }))
    .unwrap();
    let err = score_pool(&model, &vocab, &ctx, &programs, claimed, &options).unwrap_err();
    assert!(format!("{err:#}").contains("load_bound"), "{err:#}");
    let mut duplicated = programs.clone();
    duplicated.push(programs[0].clone());
    assert!(score_pool(
        &model,
        &vocab,
        &ctx,
        &duplicated,
        bindings_for(&ctx),
        &options
    )
    .is_err());
    let snapped = vec!["(pipe X (ridge alpha=0.123456))".to_string()];
    let one = Options {
        batch_size: 1,
        keep: 1,
    };
    assert!(score_pool(&model, &vocab, &ctx, &snapped, bindings_for(&ctx), &one).is_err());
}
