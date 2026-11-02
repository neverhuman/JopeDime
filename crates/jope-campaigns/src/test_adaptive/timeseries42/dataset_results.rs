fn seed_family(bank: &SeedBank, semantic: &str) -> Option<String> {
    bank.seeds
        .iter()
        .find(|seed| seed.semantic_sha256 == semantic)
        .map(|seed| seed.source_family.clone())
}

fn final_result_path(root: &Path, dataset_hash: &str) -> PathBuf {
    dataset_root(root, dataset_hash).join("final-result.v1.json")
}

fn archive_outdated_final_result(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let bytes = fs::read(path)?;
    let sha256 = sha256_bytes(&bytes);
    let archive = path.with_file_name(format!("final-result.outdated-{}.v1.json", &sha256[..16]));
    immutable_or_equal(&archive, &bytes)?;
    fs::remove_file(path).with_context(|| format!("remove outdated {}", path.display()))?;
    Ok(())
}

fn result_matches_campaign(
    result: &DatasetResult,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    release_sha256: &str,
) -> bool {
    result.schema == RESULT_SCHEMA
        && result.test_adaptive
        && !result.benchmark_valid
        && result.run_id == run_id
        && result.campaign_sha256 == campaign_sha256
        && result.release_binary_sha256 == release_sha256
        && result.dataset_hash == spec.dataset_hash
        && result.official_leader_model == spec.official_leader.model
        && result
            .official_leader_mase
            .total_cmp(&spec.official_leader.mase)
            == Ordering::Equal
        && matches!(result.status.as_str(), "win" | "loss" | "tie" | "exhausted")
}

fn load_reusable_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    release_sha256: &str,
) -> Result<Option<DatasetResult>> {
    let final_path = final_result_path(root, &spec.dataset_hash);
    if !final_path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&final_path)?;
    match serde_json::from_slice::<DatasetResult>(&bytes) {
        Ok(result)
            if result_matches_campaign(&result, run_id, campaign_sha256, spec, release_sha256) =>
        {
            Ok(Some(result))
        }
        Ok(_) | Err(_) => {
            archive_outdated_final_result(&final_path)?;
            Ok(None)
        }
    }
}

fn base_campaign_status(
    run_id: &str,
    strict_winner: bool,
    best_score: Option<&TimeSeriesScore>,
    official_mase: f64,
) -> &'static str {
    if is_forecast3_gen20(run_id) {
        return if strict_winner { "win" } else { "exhausted" };
    }
    match (strict_winner, best_score) {
        (true, _) => "win",
        (_, None) => "exhausted",
        (false, Some(score)) if score.mase.total_cmp(&official_mase) == Ordering::Equal => "tie",
        (false, Some(score)) if score.mase > official_mase => "loss",
        (false, Some(_)) => "exhausted",
    }
}

// Keep result identity, provenance and measured outcome explicit.
#[allow(clippy::too_many_arguments)]
fn write_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    bank: &SeedBank,
    all_slots: &[LogicalSlot],
    generation_zero_best: Option<f64>,
    release_sha256: &str,
    allow_generation_zero_win: bool,
) -> Result<DatasetResult> {
    let mut ranked = all_slots.to_vec();
    ranked.sort_by(rank_slots);
    let best = ranked.iter().find(|slot| slot.valid);
    let strict_winner = best.is_some_and(|slot| {
        slot_is_base_campaign_winner(slot, spec, generation_zero_best, allow_generation_zero_win)
    });
    let best_score = best.and_then(|slot| slot.score.as_ref());
    let best_beats_official =
        best_score.is_some_and(|score| score.mase < spec.official_leader.mase);
    let status = base_campaign_status(run_id, strict_winner, best_score, spec.official_leader.mase)
        .to_string();
    let result = DatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        base_run_id: None,
        base_status: None,
        base_strict_winner: None,
        base_best_jope_mase: None,
        continuation_status: None,
        best_evidence_run_id: None,
        best_receipt_file: best.map(|slot| slot.receipt_file.clone()),
        best_receipt_sha256: best.map(|slot| slot.receipt_sha256.clone()),
        best_prediction_sha256: best.and_then(|slot| slot.prediction_sha256.clone()),
        campaign_sha256: campaign_sha256.to_string(),
        release_binary_sha256: release_sha256.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        status,
        official_leader_model: spec.official_leader.model.clone(),
        official_leader_mase: spec.official_leader.mase,
        generation_zero_best_mase: generation_zero_best,
        best_jope_mase: best_score.map(|score| score.mase),
        best_jope_rmse: best_score.map(|score| score.rmse),
        mase_delta_vs_official: best_score.map(|score| spec.official_leader.mase - score.mase),
        strict_winner,
        best_beats_official,
        best_semantic_sha256: best.map(|slot| slot.member.semantic_sha256.clone()),
        best_canonical_genome: best.map(|slot| slot.member.canonical_genome.clone()),
        best_generation: best.map(|slot| slot.generation),
        best_slot: best.map(|slot| slot.slot),
        seed_source_family: best.and_then(|slot| seed_family(bank, &slot.member.semantic_sha256)),
        logical_evaluations: all_slots.len(),
        valid_evaluations: all_slots.iter().filter(|slot| slot.valid).count(),
        completed_generations: all_slots
            .iter()
            .map(|slot| slot.generation)
            .collect::<BTreeSet<_>>()
            .len(),
        completed_at_unix_ms: unix_millis(),
    };
    let target = final_result_path(root, &spec.dataset_hash);
    write_json(&target, &result)?;
    Ok(result)
}

fn classify_continuation_status(
    base_result: &DatasetResult,
    strict_winner: bool,
    best: Option<&LogicalSlot>,
    best_score: Option<&TimeSeriesScore>,
    spec: &DatasetSpec,
) -> String {
    if base_result.strict_winner {
        let improved = best_score
            .zip(base_result.best_jope_mase)
            .is_some_and(|(score, base_mase)| score.mase < base_mase)
            && best.is_some_and(|slot| slot.generation >= GEN30_START_GENERATION);
        if improved {
            "base_win_improved".to_string()
        } else {
            "inherited_base_win".to_string()
        }
    } else if strict_winner {
        "gen30_strict_win".to_string()
    } else {
        match best_score {
            None => "exhausted".to_string(),
            Some(score) if score.mase.total_cmp(&spec.official_leader.mase) == Ordering::Equal => {
                "tie".to_string()
            }
            Some(score) if score.mase > spec.official_leader.mase => "loss".to_string(),
            Some(_) => "exhausted".to_string(),
        }
    }
}

fn load_reusable_gen30_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    release_sha256: &str,
    base_run_id: &str,
) -> Result<Option<DatasetResult>> {
    let Some(result) =
        load_reusable_dataset_result(root, run_id, campaign_sha256, spec, release_sha256)?
    else {
        return Ok(None);
    };
    if result.base_run_id.as_deref() == Some(base_run_id)
        && result.continuation_status.is_some()
        && result.best_evidence_run_id.is_some()
    {
        Ok(Some(result))
    } else {
        archive_outdated_final_result(&final_result_path(root, &spec.dataset_hash))?;
        Ok(None)
    }
}

// Keep continuation provenance, generation state and measured outcome explicit.
#[allow(clippy::too_many_arguments)]
fn write_gen30_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    bank: &SeedBank,
    all_slots: &[LogicalSlot],
    generation_zero_best: Option<f64>,
    release_sha256: &str,
    base_run_id: &str,
    base_result: &DatasetResult,
) -> Result<DatasetResult> {
    let mut ranked = all_slots.to_vec();
    ranked.sort_by(rank_slots);
    let best = ranked.iter().find(|slot| slot.valid);
    let strict_floor = strict_generation_floor(base_result);
    let strict_winner = best
        .is_some_and(|slot| slot_is_strict_winner(slot, spec, generation_zero_best, strict_floor));
    let best_score = best.and_then(|slot| slot.score.as_ref());
    let best_beats_official =
        best_score.is_some_and(|score| score.mase < spec.official_leader.mase);
    let status = match (strict_winner, best_score) {
        (true, _) => "win",
        (_, None) => "exhausted",
        (false, Some(score))
            if score.mase.total_cmp(&spec.official_leader.mase) == Ordering::Equal =>
        {
            "tie"
        }
        (false, Some(score)) if score.mase > spec.official_leader.mase => "loss",
        (false, Some(_)) => "exhausted",
    }
    .to_string();
    let continuation_status =
        classify_continuation_status(base_result, strict_winner, best, best_score, spec);
    let result = DatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        base_run_id: Some(base_run_id.to_string()),
        base_status: Some(base_result.status.clone()),
        base_strict_winner: Some(base_result.strict_winner),
        base_best_jope_mase: base_result.best_jope_mase,
        continuation_status: Some(continuation_status),
        best_evidence_run_id: best.map(|slot| {
            if slot.generation < GEN30_START_GENERATION {
                base_run_id.to_string()
            } else {
                run_id.to_string()
            }
        }),
        best_receipt_file: best.map(|slot| slot.receipt_file.clone()),
        best_receipt_sha256: best.map(|slot| slot.receipt_sha256.clone()),
        best_prediction_sha256: best.and_then(|slot| slot.prediction_sha256.clone()),
        campaign_sha256: campaign_sha256.to_string(),
        release_binary_sha256: release_sha256.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        status,
        official_leader_model: spec.official_leader.model.clone(),
        official_leader_mase: spec.official_leader.mase,
        generation_zero_best_mase: generation_zero_best,
        best_jope_mase: best_score.map(|score| score.mase),
        best_jope_rmse: best_score.map(|score| score.rmse),
        mase_delta_vs_official: best_score.map(|score| spec.official_leader.mase - score.mase),
        strict_winner,
        best_beats_official,
        best_semantic_sha256: best.map(|slot| slot.member.semantic_sha256.clone()),
        best_canonical_genome: best.map(|slot| slot.member.canonical_genome.clone()),
        best_generation: best.map(|slot| slot.generation),
        best_slot: best.map(|slot| slot.slot),
        seed_source_family: best.and_then(|slot| seed_family(bank, &slot.member.semantic_sha256)),
        logical_evaluations: all_slots.len(),
        valid_evaluations: all_slots.iter().filter(|slot| slot.valid).count(),
        completed_generations: all_slots
            .iter()
            .map(|slot| slot.generation)
            .collect::<BTreeSet<_>>()
            .len(),
        completed_at_unix_ms: unix_millis(),
    };
    let target = final_result_path(root, &spec.dataset_hash);
    write_json(&target, &result)?;
    Ok(result)
}

// Keep dataset identity, campaign state and execution budget explicit.
#[allow(clippy::too_many_arguments)]
fn run_dataset(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    staging: &StagingReceipt,
    release_sha256: &str,
    config: &CampaignConfig,
    allow_generation_zero_win: bool,
) -> Result<DatasetResult> {
    if let Some(result) =
        load_reusable_dataset_result(root, run_id, campaign_sha256, spec, release_sha256)?
    {
        return Ok(result);
    }
    let (bank, seed_sha256) = load_or_create_seed_bank(root, spec)?;
    let mut members = bank
        .seeds
        .iter()
        .map(SeedRecord::member)
        .collect::<Vec<_>>();
    let mut all_slots = Vec::new();
    let mut generation_zero_best = None;
    for generation in 0..config.generations {
        let mut slots = evaluate_generation(
            root,
            run_id,
            campaign_sha256,
            spec,
            &seed_sha256,
            release_sha256,
            staging,
            generation,
            &members,
        )?;
        let mut ranked = slots.clone();
        ranked.sort_by(rank_slots);
        if generation == 0 {
            generation_zero_best = ranked
                .iter()
                .find(|slot| slot.valid)
                .and_then(|slot| slot.score.as_ref().map(|score| score.mase));
        }
        all_slots.append(&mut slots);
        let has_strict_winner = ranked.iter().any(|slot| {
            slot_is_base_campaign_winner(
                slot,
                spec,
                generation_zero_best,
                allow_generation_zero_win,
            )
        });
        if (has_strict_winner && generation + 1 >= config.minimum_generations.max(1))
            || generation + 1 == config.generations
        {
            break;
        }
        members = next_population(spec, &ranked, generation + 1, "")?;
    }
    write_dataset_result(
        root,
        run_id,
        campaign_sha256,
        spec,
        &bank,
        &all_slots,
        generation_zero_best,
        release_sha256,
        allow_generation_zero_win,
    )
}
