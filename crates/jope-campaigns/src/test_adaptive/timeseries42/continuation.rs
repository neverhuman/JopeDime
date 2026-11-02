fn run_gen30_dataset(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    staging: &StagingReceipt,
    release_sha256: &str,
    base: &BaseCampaign,
) -> Result<DatasetResult> {
    if let Some(result) = load_reusable_gen30_dataset_result(
        root,
        run_id,
        campaign_sha256,
        spec,
        release_sha256,
        &base.manifest.run_id,
    )? {
        return Ok(result);
    }
    let (bank, seed_sha256) = load_existing_seed_bank(&base.root, spec)?;
    let base_result: DatasetResult = load_json(&final_result_path(&base.root, &spec.dataset_hash))?;
    if !result_matches_campaign(
        &base_result,
        &base.manifest.run_id,
        &base.campaign_sha256,
        spec,
        &base.manifest.release_binary_sha256,
    ) {
        bail!("invalid base TimeSeries42 result for {}", spec.dataset_hash);
    }
    let mut all_slots = Vec::with_capacity(GEN30_GENERATIONS * POPULATION);
    let generation_zero = load_generation_slots(
        &base.root,
        &base.manifest.run_id,
        &base.campaign_sha256,
        spec,
        0,
    )?;
    let generation_zero_best = generation_best_mase(&generation_zero);
    all_slots.extend(generation_zero);
    let generation_one = load_generation_slots(
        &base.root,
        &base.manifest.run_id,
        &base.campaign_sha256,
        spec,
        1,
    )?;
    let mut ranked = generation_one.clone();
    ranked.sort_by(rank_slots);
    all_slots.extend(generation_one);

    let mut members = next_population(spec, &ranked, GEN30_START_GENERATION, "")?;
    for generation in GEN30_START_GENERATION..GEN30_GENERATIONS {
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
        let mut generation_ranked = slots.clone();
        generation_ranked.sort_by(rank_slots);
        all_slots.append(&mut slots);
        if continuation_should_stop(
            &base_result,
            &all_slots,
            spec,
            generation_zero_best,
            generation,
        ) {
            break;
        }
        members = next_population(spec, &generation_ranked, generation + 1, "")?;
    }
    write_gen30_dataset_result(
        root,
        run_id,
        campaign_sha256,
        spec,
        &bank,
        &all_slots,
        generation_zero_best,
        release_sha256,
        &base.manifest.run_id,
        &base_result,
    )
}

fn load_reusable_winall_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    release_sha256: &str,
    source_run_id: &str,
) -> Result<Option<DatasetResult>> {
    let Some(result) =
        load_reusable_dataset_result(root, run_id, campaign_sha256, spec, release_sha256)?
    else {
        return Ok(None);
    };
    if result.base_run_id.as_deref() == Some(source_run_id)
        && result.continuation_status.is_some()
        && result.best_evidence_run_id.is_some()
    {
        Ok(Some(result))
    } else {
        archive_outdated_final_result(&final_result_path(root, &spec.dataset_hash))?;
        Ok(None)
    }
}

// Keep win-all provenance, continuation state and measured outcome explicit.
#[allow(clippy::too_many_arguments)]
fn write_winall_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    bank: &SeedBank,
    all_slots: &[LogicalSlot],
    generation_zero_best: Option<f64>,
    release_sha256: &str,
    source_run_id: &str,
    source_result: &DatasetResult,
    start_generation: usize,
) -> Result<DatasetResult> {
    let mut ranked = all_slots.to_vec();
    ranked.sort_by(rank_slots);
    let best = ranked.iter().find(|slot| slot.valid);
    let strict_winner = best.is_some_and(|slot| {
        slot_is_strict_winner(slot, spec, generation_zero_best, start_generation)
    });
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
    let continuation_status = if strict_winner {
        "winall_strict_win".to_string()
    } else {
        match best_score {
            None => "exhausted".to_string(),
            Some(score) if score.mase.total_cmp(&spec.official_leader.mase) == Ordering::Equal => {
                "tie".to_string()
            }
            Some(score) if score.mase > spec.official_leader.mase => "loss".to_string(),
            Some(_) => "exhausted".to_string(),
        }
    };
    let result = DatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        base_run_id: Some(source_run_id.to_string()),
        base_status: Some(source_result.status.clone()),
        base_strict_winner: Some(source_result.strict_winner),
        base_best_jope_mase: source_result.best_jope_mase,
        continuation_status: Some(continuation_status),
        best_evidence_run_id: best.map(|slot| {
            if slot.generation < start_generation {
                source_run_id.to_string()
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

// Keep win-all campaign identity, source state and execution budget explicit.
#[allow(clippy::too_many_arguments)]
fn run_winall_dataset(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    staging: &StagingReceipt,
    release_sha256: &str,
    source: &WinallSourceCampaign,
    max_generations: usize,
    lineage_salt: &str,
) -> Result<DatasetResult> {
    if let Some(result) = load_reusable_winall_dataset_result(
        root,
        run_id,
        campaign_sha256,
        spec,
        release_sha256,
        &source.manifest.run_id,
    )? {
        return Ok(result);
    }
    let (bank, seed_sha256) = load_existing_seed_bank(&source.base.root, spec)?;
    let source_result: DatasetResult =
        load_json(&final_result_path(&source.root, &spec.dataset_hash))?;
    if source_result.strict_winner {
        bail!(
            "TimeSeries42 win-all source result is already a strict winner for {}",
            spec.dataset_hash
        );
    }
    let start_generation = source.terminal_generation + 1;
    let mut all_slots = Vec::with_capacity(max_generations * POPULATION);
    let generation_zero = load_generation_slots(
        &source.base.root,
        &source.base.manifest.run_id,
        &source.base.campaign_sha256,
        spec,
        0,
    )?;
    let generation_zero_best = generation_best_mase(&generation_zero);
    all_slots.extend(generation_zero);
    let generation_one = load_generation_slots(
        &source.base.root,
        &source.base.manifest.run_id,
        &source.base.campaign_sha256,
        spec,
        1,
    )?;
    all_slots.extend(generation_one);

    let source_start = source
        .manifest
        .continuation_start_generation
        .unwrap_or(GEN30_START_GENERATION);
    let mut ranked = Vec::new();
    for generation in source_start..=source.terminal_generation {
        let slots = load_generation_slots(
            &source.root,
            &source.manifest.run_id,
            &source.campaign_sha256,
            spec,
            generation,
        )?;
        if generation == source.terminal_generation {
            ranked = slots.clone();
        }
        all_slots.extend(slots);
    }
    ranked.sort_by(rank_slots);
    let mut members = next_population(spec, &ranked, start_generation, lineage_salt)?;
    for generation in start_generation..max_generations {
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
        let mut generation_ranked = slots.clone();
        generation_ranked.sort_by(rank_slots);
        all_slots.append(&mut slots);
        if has_strict_winner_from_generation(
            &all_slots,
            spec,
            generation_zero_best,
            start_generation,
        ) || generation + 1 == max_generations
        {
            break;
        }
        members = next_population(spec, &generation_ranked, generation + 1, lineage_salt)?;
    }
    write_winall_dataset_result(
        root,
        run_id,
        campaign_sha256,
        spec,
        &bank,
        &all_slots,
        generation_zero_best,
        release_sha256,
        &source.manifest.run_id,
        &source_result,
        start_generation,
    )
}
