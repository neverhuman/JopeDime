fn receipt_mase(receipt: &EvalReceipt) -> Option<TimeSeriesScore> {
    if receipt.task != "timeseries" || !receipt.valid {
        return None;
    }
    let mase = if receipt.secondary_metric == "mase" {
        receipt.secondary_score?
    } else if receipt.metric == "negative_mase" {
        -receipt.score?
    } else {
        return None;
    };
    if !mase.is_finite() {
        return None;
    }
    Some(TimeSeriesScore {
        mase,
        rmse: receipt.rmse.unwrap_or(f64::INFINITY),
    })
}

fn slot_from_receipt(
    generation: usize,
    slot: usize,
    member: &Member,
    receipt: &EvalReceipt,
    receipt_file: &str,
    receipt_sha256: &str,
) -> LogicalSlot {
    let score = receipt_mase(receipt);
    LogicalSlot {
        generation,
        slot,
        member: member.clone(),
        valid: score.is_some(),
        score,
        failure: receipt.failure.clone(),
        receipt_file: receipt_file.to_string(),
        receipt_sha256: receipt_sha256.to_string(),
        prediction_sha256: receipt.predictions_sha256.clone(),
        runtime_seconds: receipt.runtime_seconds,
    }
}

fn rank_slots(left: &LogicalSlot, right: &LogicalSlot) -> Ordering {
    let left_score = left.score.as_ref();
    let right_score = right.score.as_ref();
    match (left_score, right_score) {
        (Some(left_score), Some(right_score)) => left_score
            .mase
            .total_cmp(&right_score.mase)
            .then_with(|| left_score.rmse.total_cmp(&right_score.rmse))
            .then_with(|| left.member.complexity.cmp(&right.member.complexity))
            .then_with(|| {
                left.member
                    .semantic_sha256
                    .cmp(&right.member.semantic_sha256)
            }),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left
            .member
            .semantic_sha256
            .cmp(&right.member.semantic_sha256),
    }
}

// Keep campaign identity, generation state and execution limits explicit.
#[allow(clippy::too_many_arguments)]
fn evaluate_generation(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    seed_sha256: &str,
    release_sha256: &str,
    staging: &StagingReceipt,
    generation: usize,
    members: &[Member],
) -> Result<Vec<LogicalSlot>> {
    if members.len() != POPULATION {
        bail!("TimeSeries42 generation requires exactly {POPULATION} members");
    }
    let chunks = members
        .chunks(SHARD_SIZE)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    let records = execute_transactional_wave(
        root,
        run_id,
        campaign_sha256,
        spec,
        seed_sha256,
        release_sha256,
        staging,
        generation,
        &chunks,
    )?;

    let expected_protocol = if is_forecast3_gen20(run_id) {
        Some(dataset_protocol_sha256(spec)?)
    } else {
        None
    };
    let mut slots = Vec::with_capacity(POPULATION);
    for record in &records {
        let receipts = read_receipts(Path::new(&record.receipt_file))?;
        if receipts.len() != SHARD_SIZE {
            bail!("TimeSeries42 shard produced {} receipts", receipts.len());
        }
        let start = record.shard * SHARD_SIZE;
        for (offset, receipt) in receipts.iter().enumerate() {
            let slot = start + offset;
            if receipt.dataset_hash != spec.dataset_hash
                || receipt.semantic_sha256 != members[slot].semantic_sha256
                || receipt.canonical_genome != members[slot].canonical_genome
                || expected_protocol
                    .as_ref()
                    .is_some_and(|expected| receipt.protocol_sha256.as_deref() != Some(expected))
            {
                bail!("TimeSeries42 receipt/member binding mismatch");
            }
            slots.push(slot_from_receipt(
                generation,
                slot,
                &members[slot],
                receipt,
                &record.receipt_file,
                &record.receipt_sha256,
            ));
        }
    }
    let mut ranked = slots.clone();
    ranked.sort_by(rank_slots);
    let best = ranked.iter().find(|slot| slot.valid);
    let manifest = GenerationManifest {
        schema: GENERATION_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        campaign_sha256: campaign_sha256.to_string(),
        release_binary_sha256: release_sha256.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        generation,
        members: members.to_vec(),
        shards: records,
        best_semantic_sha256: best.map(|slot| slot.member.semantic_sha256.clone()),
        best_mase: best.and_then(|slot| slot.score.as_ref().map(|score| score.mase)),
        valid_candidates: slots.iter().filter(|slot| slot.valid).count(),
        completed_at_unix_ms: unix_millis(),
    };
    write_current_json(
        &generation_root(&dataset_root(root, &spec.dataset_hash), generation)
            .join("generation.v1.json"),
        &manifest,
        "generation",
    )?;
    Ok(slots)
}

fn load_generation_slots(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    generation: usize,
) -> Result<Vec<LogicalSlot>> {
    let path = generation_root(&dataset_root(root, &spec.dataset_hash), generation)
        .join("generation.v1.json");
    let manifest: GenerationManifest = load_json(&path)?;
    if manifest.schema != GENERATION_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.run_id != run_id
        || manifest.campaign_sha256 != campaign_sha256
        || manifest.dataset_hash != spec.dataset_hash
        || manifest.generation != generation
        || manifest.members.len() != POPULATION
        || manifest.shards.len() != SHARDS
    {
        bail!(
            "TimeSeries42 generation artifact is outdated or invalid for {} generation {}",
            spec.dataset_hash,
            generation
        );
    }
    let mut records = manifest.shards.clone();
    let expected_protocol = if is_forecast3_gen20(run_id) {
        Some(dataset_protocol_sha256(spec)?)
    } else {
        None
    };
    records.sort_by_key(|record| record.shard);
    let mut seen_shards = BTreeSet::new();
    let mut slots = Vec::with_capacity(POPULATION);
    for record in &records {
        if record.generation != generation
            || record.shard >= SHARDS
            || !seen_shards.insert(record.shard)
            || record.total_candidates != SHARD_SIZE
        {
            bail!(
                "TimeSeries42 shard accounting mismatch for {} generation {}",
                spec.dataset_hash,
                generation
            );
        }
        let receipt_path = Path::new(&record.receipt_file);
        if sha256_file(receipt_path)? != record.receipt_sha256 {
            bail!(
                "TimeSeries42 shard receipt hash mismatch for {} generation {} shard {}",
                spec.dataset_hash,
                generation,
                record.shard
            );
        }
        let receipts = read_receipts(receipt_path)?;
        if receipts.len() != SHARD_SIZE {
            bail!("TimeSeries42 shard produced {} receipts", receipts.len());
        }
        let start = record.shard * SHARD_SIZE;
        for (offset, receipt) in receipts.iter().enumerate() {
            let slot = start + offset;
            if receipt.dataset_hash != spec.dataset_hash
                || receipt.semantic_sha256 != manifest.members[slot].semantic_sha256
                || receipt.canonical_genome != manifest.members[slot].canonical_genome
                || expected_protocol
                    .as_ref()
                    .is_some_and(|expected| receipt.protocol_sha256.as_deref() != Some(expected))
            {
                bail!("TimeSeries42 base receipt/member binding mismatch");
            }
            slots.push(slot_from_receipt(
                generation,
                slot,
                &manifest.members[slot],
                receipt,
                &record.receipt_file,
                &record.receipt_sha256,
            ));
        }
    }
    if slots.len() != POPULATION {
        bail!(
            "TimeSeries42 loaded {} slots for {} generation {}",
            slots.len(),
            spec.dataset_hash,
            generation
        );
    }
    Ok(slots)
}

fn generation_best_mase(slots: &[LogicalSlot]) -> Option<f64> {
    let mut ranked = slots.to_vec();
    ranked.sort_by(rank_slots);
    ranked
        .iter()
        .find(|slot| slot.valid)
        .and_then(|slot| slot.score.as_ref().map(|score| score.mase))
}

fn strict_generation_floor(base_result: &DatasetResult) -> usize {
    if base_result.strict_winner {
        1
    } else {
        GEN30_START_GENERATION
    }
}

fn slot_is_strict_winner(
    slot: &LogicalSlot,
    spec: &DatasetSpec,
    generation_zero_best: Option<f64>,
    min_generation: usize,
) -> bool {
    let Some(score) = &slot.score else {
        return false;
    };
    slot.generation >= min_generation
        && slot.member.evolved_descendant
        && score.mase < spec.official_leader.mase
        && generation_zero_best.is_some_and(|g0| score.mase < g0)
}

fn slot_beats_external(slot: &LogicalSlot, spec: &DatasetSpec) -> bool {
    slot.valid
        && slot
            .score
            .as_ref()
            .is_some_and(|score| score.mase.is_finite() && score.mase < spec.official_leader.mase)
}

fn slot_is_base_campaign_winner(
    slot: &LogicalSlot,
    spec: &DatasetSpec,
    generation_zero_best: Option<f64>,
    allow_generation_zero_win: bool,
) -> bool {
    if allow_generation_zero_win {
        slot_beats_external(slot, spec)
    } else {
        slot_is_strict_winner(slot, spec, generation_zero_best, 1)
    }
}

fn has_strict_winner_from_generation(
    slots: &[LogicalSlot],
    spec: &DatasetSpec,
    generation_zero_best: Option<f64>,
    min_generation: usize,
) -> bool {
    slots
        .iter()
        .any(|slot| slot_is_strict_winner(slot, spec, generation_zero_best, min_generation))
}

fn continuation_should_stop(
    base_result: &DatasetResult,
    all_slots: &[LogicalSlot],
    spec: &DatasetSpec,
    generation_zero_best: Option<f64>,
    generation: usize,
) -> bool {
    if generation + 1 == GEN30_GENERATIONS {
        return true;
    }
    if base_result.strict_winner {
        return false;
    }
    has_strict_winner_from_generation(
        all_slots,
        spec,
        generation_zero_best,
        strict_generation_floor(base_result),
    )
}

fn offspring_seed(
    dataset_hash: &str,
    generation: usize,
    slot: usize,
    retry: usize,
    lineage_salt: &str,
) -> u64 {
    let text = if lineage_salt.is_empty() {
        format!("{RNG_SEED}:{dataset_hash}:{generation}:{slot}:{retry}")
    } else {
        format!("{RNG_SEED}:{lineage_salt}:{dataset_hash}:{generation}:{slot}:{retry}")
    };
    let digest = Sha256::digest(text.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"))
}

fn tournament_winner(rng: &mut Pcg32) -> usize {
    (0..TOURNAMENT)
        .map(|_| rng.next_u32() as usize % PARENT_POOL)
        .min()
        .expect("tournament is nonempty")
}

fn next_population(
    spec: &DatasetSpec,
    ranked: &[LogicalSlot],
    generation: usize,
    lineage_salt: &str,
) -> Result<Vec<Member>> {
    if ranked.len() < PARENT_POOL || ranked[..PARENT_POOL].iter().any(|slot| !slot.valid) {
        bail!("TimeSeries42 parent pool has fewer than {PARENT_POOL} valid candidates");
    }
    let mut next = Vec::with_capacity(POPULATION);
    let mut seen = BTreeSet::new();
    for elite in ranked.iter().take(ELITES) {
        seen.insert(elite.member.semantic_sha256.clone());
        let mut member = elite.member.clone();
        member.evolved_descendant = true;
        member.ancestry = AncestryRecord {
            operation: "elite".to_string(),
            parents: vec![elite.member.semantic_sha256.clone()],
            retry_count: 0,
        };
        next.push(member);
    }
    for slot in ELITES..POPULATION {
        let mut produced = None;
        for retry in 0..1_000 {
            let mut rng = Pcg32::new(
                offspring_seed(&spec.dataset_hash, generation, slot, retry, lineage_salt),
                0x7473_3432_6e65_7874,
            );
            let first = tournament_winner(&mut rng);
            let first_parent = &ranked[first].member;
            let maybe = if rng.next_f64() < 0.25 {
                let second = tournament_winner(&mut rng);
                let second_parent = &ranked[second].member;
                crossover_pipelines(
                    &first_parent.canonical_genome,
                    &second_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                )
                .map(|genome| {
                    (
                        genome,
                        "crossover",
                        vec![
                            first_parent.semantic_sha256.clone(),
                            second_parent.semantic_sha256.clone(),
                        ],
                    )
                })
            } else {
                mutate_pipeline(
                    &first_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                    feat_math::full_rust::MAX_TRANSFORMS,
                )
                .map(|genome| {
                    (
                        genome,
                        "mutation",
                        vec![first_parent.semantic_sha256.clone()],
                    )
                })
            };
            let Ok((candidate, operation, parents)) = maybe else {
                continue;
            };
            let canonical = match canonicalize_for_timeseries(&candidate.canonical, spec) {
                Ok(canonical) => canonical,
                Err(_) => continue,
            };
            if !seen.insert(canonical.semantic_sha256.clone()) {
                continue;
            }
            produced = Some(Member {
                canonical_genome: canonical.canonical,
                semantic_sha256: canonical.semantic_sha256,
                complexity: canonical.complexity,
                rust_source_len: canonical.rust_source_len,
                evolved_descendant: true,
                ancestry: AncestryRecord {
                    operation: operation.to_string(),
                    parents,
                    retry_count: retry,
                },
            });
            break;
        }
        next.push(produced.with_context(|| {
            format!(
                "could not generate TimeSeries42 child for {} generation {generation} slot {slot}",
                spec.dataset_hash
            )
        })?);
    }
    Ok(next)
}
