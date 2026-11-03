use super::*;
use super::{member_models::StagingReceipt, worker_models::GenerationCheckpoint};

pub(crate) fn cache_path(root: &Path, semantic_sha256: &str) -> PathBuf {
    root.join("objects")
        .join("cache")
        .join(format!("{semantic_sha256}.v1.json"))
}

pub(crate) fn load_cache(root: &Path) -> Result<HashMap<String, CacheEntry>> {
    let cache_root = root.join("objects").join("cache");
    if !cache_root.exists() {
        return Ok(HashMap::new());
    }
    let mut entries = HashMap::new();
    for entry in fs::read_dir(cache_root)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            bail!("unexpected file in Binary42 semantic cache");
        }
        let record: CacheEntry = load_json(&path)?;
        if record.schema != CACHE_SCHEMA
            || !record.test_adaptive
            || record.benchmark_valid
            || path.file_name().and_then(|value| value.to_str())
                != Some(&format!("{}.v1.json", record.semantic_sha256))
            || entries
                .insert(record.semantic_sha256.clone(), record)
                .is_some()
        {
            bail!("invalid or duplicate Binary42 semantic cache entry");
        }
    }
    Ok(entries)
}

pub(crate) fn store_genome(root: &Path, member: &MemberRecord) -> Result<()> {
    if member.semantic_sha256 != sha256_bytes(member.canonical_genome.as_bytes()) {
        bail!("Binary42 member semantic hash mismatch");
    }
    let object = GenomeObject {
        schema: GENOME_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        semantic_sha256: member.semantic_sha256.clone(),
        canonical_genome: member.canonical_genome.clone(),
        complexity: member.complexity,
    };
    write_json(
        &root
            .join("objects")
            .join("genomes")
            .join(format!("{}.v1.json", member.semantic_sha256)),
        &object,
    )?;
    Ok(())
}

pub(crate) fn validate_fresh_output(
    output: &FreshOutput,
    test: &RegressionData,
) -> Result<Option<CandidateScore>> {
    if sha256_bytes(&output.receipt_bytes) != sha256_bytes(&canonical_json(&output.receipt)?)
        || sha256_bytes(&output.prediction_bytes) != output.receipt.predictions_sha256
    {
        bail!("Binary42 fresh output content hash mismatch");
    }
    let predictions = parse_prediction_bytes(&output.prediction_bytes)?;
    match (&output.receipt.score, &output.receipt.failure) {
        (Some(score), None) => {
            if predictions.len() != test.y.len() {
                bail!("Binary42 candidate prediction length mismatch");
            }
            let replay = CandidateScore {
                auc: auc(&test.y, &predictions),
                logloss: feat_math::metrics::logloss(&test.y, &predictions),
            };
            if replay.auc.to_bits() != score.auc.to_bits()
                || !logloss_replays(score.logloss, replay.logloss)
            {
                bail!("Binary42 candidate metrics fall outside replay tolerance");
            }
            Ok(Some(replay))
        }
        (None, Some(_)) if predictions.is_empty() => Ok(None),
        _ => bail!("Binary42 candidate success/failure fields are inconsistent"),
    }
}

pub(crate) fn logloss_replays(worker: f64, controller: f64) -> bool {
    worker.is_finite()
        && controller.is_finite()
        && (worker.to_bits().abs_diff(controller.to_bits()) <= LOGLOSS_REPLAY_MAX_ULPS
            || (worker - controller).abs() <= LOGLOSS_REPLAY_MAX_ABS)
}

pub(crate) fn import_fresh_output(
    root: &Path,
    output: FreshOutput,
    test: &RegressionData,
    extended: bool,
) -> Result<CacheEntry> {
    validate_candidate_backend(&output.receipt, extended)?;
    let authoritative_score = validate_fresh_output(&output, test)?;
    let receipt_sha256 = sha256_bytes(&output.receipt_bytes);
    let prediction_sha256 = sha256_bytes(&output.prediction_bytes);
    immutable_or_equal(
        &root
            .join("objects")
            .join("receipts")
            .join(format!("{receipt_sha256}.json")),
        &output.receipt_bytes,
    )?;
    immutable_or_equal(
        &root
            .join("objects")
            .join("predictions")
            .join(format!("{prediction_sha256}.csv")),
        &output.prediction_bytes,
    )?;
    let cache = CacheEntry {
        schema: CACHE_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        semantic_sha256: output.receipt.semantic_sha256.clone(),
        receipt_sha256,
        prediction_sha256,
        score: authoritative_score,
        failure: output.receipt.failure.clone(),
        evaluation_node: output.receipt.node.clone(),
        scq_job_id: output.receipt.scq_job_id.clone(),
        first_generation: output.receipt.generation,
        first_slot: output.receipt.slot,
        runtime_seconds: extended.then_some(output.receipt.runtime_seconds),
    };
    write_json(&cache_path(root, &cache.semantic_sha256), &cache)?;
    Ok(cache)
}

pub(crate) fn record_infrastructure_failure(
    root: &Path,
    manifest: &ShardManifest,
    error: &anyhow::Error,
) -> Result<()> {
    let failure = json!({
        "schema": "JopeBinary42TestAdaptiveInfrastructureFailure.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "job_id": manifest.job_id,
        "node": manifest.node,
        "generation": manifest.generation,
        "shard": manifest.shard,
        "attempt": manifest.attempt,
        "error": format!("{error:#}"),
    });
    write_json(
        &root
            .join("jobs")
            .join("failures")
            .join(format!("{}.v1.json", manifest.job_id)),
        &failure,
    )?;
    Ok(())
}

pub(crate) fn execute_generation_shards(
    root: &Path,
    campaign: &CampaignManifest,
    staging: &StagingReceipt,
    generation: usize,
    members: &[MemberRecord],
    cache: &HashMap<String, CacheEntry>,
) -> Result<Vec<FreshOutput>> {
    let shard_sizes = campaign_shard_sizes(&campaign.config);
    let chunks = shard_members_with_sizes(members, &shard_sizes)?;
    let manifests = chunks
        .iter()
        .enumerate()
        .map(|(shard, members)| {
            shard_manifest(
                campaign,
                &staging.nodes[shard],
                generation,
                shard,
                0,
                "generation",
                members,
                cache,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let results = Arc::new(Mutex::new(Vec::<(
        usize,
        std::result::Result<Vec<FreshOutput>, anyhow::Error>,
    )>::new()));
    thread::scope(|scope| {
        for (index, manifest) in manifests.iter().enumerate() {
            let results = Arc::clone(&results);
            scope.spawn(move || {
                let result = execute_shard(root, manifest);
                results.lock().unwrap().push((index, result));
            });
        }
    });
    let mut results = Arc::try_unwrap(results)
        .expect("Binary42 shard result references remain")
        .into_inner()
        .unwrap();
    results.sort_by_key(|(index, _)| *index);
    let mut outputs = Vec::new();
    for (index, result) in results {
        match result {
            Ok(mut fresh) => outputs.append(&mut fresh),
            Err(error) => {
                record_infrastructure_failure(root, &manifests[index], &error)?;
                let mut recovered = None;
                let mut final_error = error;
                for attempt in 1..=campaign.config.infrastructure_retries {
                    let retry_node = failover_node(index, attempt)?;
                    let retry = shard_manifest(
                        campaign,
                        &staging.nodes[NODES.iter().position(|node| *node == retry_node).unwrap()],
                        generation,
                        index,
                        attempt,
                        "generation",
                        &chunks[index],
                        cache,
                    )?;
                    match execute_shard(root, &retry) {
                        Ok(fresh) => {
                            recovered = Some(fresh);
                            break;
                        }
                        Err(error) => {
                            record_infrastructure_failure(root, &retry, &error)?;
                            final_error = error;
                        }
                    }
                }
                let mut fresh = recovered.with_context(|| {
                    format!(
                        "Binary42 generation {generation} shard {index} exhausted {} failover attempts: {final_error:#}",
                        campaign.config.infrastructure_retries
                    )
                })?;
                outputs.append(&mut fresh);
            }
        }
    }
    Ok(outputs)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn process_generation(
    root: &Path,
    campaign: &CampaignManifest,
    staging: &StagingReceipt,
    generation: usize,
    members: Vec<MemberRecord>,
    previous: Option<&GenerationCheckpoint>,
    seed_semantics: &BTreeSet<String>,
    test: &RegressionData,
) -> Result<GenerationCheckpoint> {
    if members.len() != POPULATION {
        bail!("Binary42 generation population is not exactly {POPULATION}");
    }
    let shard_sizes = campaign_shard_sizes(&campaign.config);
    let generation_dir = generation_root(root, generation);
    let generation_manifest = GenerationManifest {
        schema: GENERATION_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        generation,
        logical_slots: POPULATION,
        shard_size: shard_sizes[0],
        shard_nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
        members: members.clone(),
        shard_sizes: if campaign.config.shard_sizes.is_empty() {
            Vec::new()
        } else {
            shard_sizes.clone()
        },
    };
    write_json(
        &generation_dir.join("generation-manifest.v1.json"),
        &generation_manifest,
    )?;
    for member in &members {
        store_genome(root, member)?;
    }
    let mut cache = load_cache(root)?;
    let preexisting = cache
        .iter()
        .filter(|(_, entry)| entry.first_generation < generation)
        .map(|(semantic, _)| semantic.clone())
        .collect::<BTreeSet<_>>();
    let fresh_outputs =
        execute_generation_shards(root, campaign, staging, generation, &members, &cache)?;
    let expected_fresh = members
        .iter()
        .filter(|member| !preexisting.contains(&member.semantic_sha256))
        .count();
    if fresh_outputs.len() != expected_fresh {
        bail!(
            "Binary42 generation {generation} expected {expected_fresh} fresh outputs, got {}",
            fresh_outputs.len()
        );
    }
    for output in fresh_outputs {
        let semantic = output.receipt.semantic_sha256.clone();
        if preexisting.contains(&semantic) {
            bail!("Binary42 worker returned an already-cached semantic genome");
        }
        let imported =
            import_fresh_output(root, output, test, !campaign.config.shard_sizes.is_empty())?;
        cache.insert(semantic, imported);
    }
    let mut slots = Vec::with_capacity(POPULATION);
    for (slot, member) in members.iter().enumerate() {
        let cached = cache
            .get(&member.semantic_sha256)
            .context("Binary42 semantic cache is missing a generation member")?;
        slots.push(LogicalSlot {
            generation,
            slot,
            assigned_node: assigned_node_for_slot(&shard_sizes, slot)?.to_string(),
            member: member.clone(),
            cache_disposition: if preexisting.contains(&member.semantic_sha256) {
                "reused".to_string()
            } else {
                "fresh".to_string()
            },
            cache_source_generation: preexisting
                .contains(&member.semantic_sha256)
                .then_some(cached.first_generation),
            cache_source_slot: preexisting
                .contains(&member.semantic_sha256)
                .then_some(cached.first_slot),
            evaluation_node: cached.evaluation_node.clone(),
            scq_job_id: cached.scq_job_id.clone(),
            receipt_sha256: cached.receipt_sha256.clone(),
            prediction_sha256: cached.prediction_sha256.clone(),
            score: cached.score.clone(),
            failure: cached.failure.clone(),
            runtime_seconds: cached.runtime_seconds,
        });
    }
    if slots.len() != POPULATION {
        bail!("Binary42 generation barrier has a short logical slot ledger");
    }
    let mut ranked = slots
        .iter()
        .filter(|slot| slot.score.is_some())
        .cloned()
        .collect::<Vec<_>>();
    ranked.sort_by(rank_slots);
    if ranked.len() < PARENT_POOL {
        bail!(
            "Binary42 generation {generation} has only {} valid candidates",
            ranked.len()
        );
    }
    let best = ranked
        .first()
        .context("Binary42 generation has no best slot")?;
    let generation_zero_best_auc = if generation == 0 {
        best.score.as_ref().unwrap().auc
    } else {
        previous
            .context("Binary42 nonzero generation has no previous checkpoint")?
            .generation_zero_best_auc
    };
    let qualifying = qualifying_winner_for_policy(
        generation,
        &ranked,
        seed_semantics,
        &campaign.dataset_binding.win_policy,
        campaign.dataset_binding.leader_auc,
        generation_zero_best_auc,
    );
    let reached_cap = generation_cap_reached(&campaign.config, generation);
    let next_population = if qualifying.is_some() || reached_cap {
        Vec::new()
    } else {
        let parents = ranked
            .iter()
            .map(|slot| {
                let score = slot.score.as_ref().unwrap();
                BinaryAdaptiveRanked {
                    canonical_genome: slot.member.canonical_genome.clone(),
                    semantic_sha256: slot.member.semantic_sha256.clone(),
                    complexity: slot.member.complexity,
                    auc: score.auc,
                    logloss: score.logloss,
                }
            })
            .collect::<Vec<_>>();
        test_adaptive_next_population(
            RNG_SEED,
            generation + 1,
            &parents,
            campaign.dataset_binding.features,
        )?
        .into_iter()
        .map(MemberRecord::from_binary)
        .collect::<Result<Vec<_>>>()?
    };
    let previous_checkpoint_sha256 = previous
        .map(|checkpoint| canonical_json(checkpoint).map(|bytes| sha256_bytes(&bytes)))
        .transpose()?;
    let checkpoint = GenerationCheckpoint {
        schema: CHECKPOINT_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        generation,
        previous_checkpoint_sha256,
        logical_slots: POPULATION,
        fresh_evaluations: expected_fresh,
        cache_reuses: POPULATION - expected_fresh,
        valid_scores: ranked.len(),
        generation_zero_best_auc,
        best_slot: best.slot,
        best_auc: best.score.as_ref().unwrap().auc,
        slots,
        ranked_semantics: ranked
            .iter()
            .map(|slot| slot.member.semantic_sha256.clone())
            .collect(),
        next_population,
        qualifying_winner: qualifying,
    };
    write_json(&generation_dir.join("checkpoint.v1.json"), &checkpoint)?;
    eprintln!(
        "Binary42 generation {} complete: fresh={}, cache={}, best_auc={:.17}, \
         leader_auc={:.17}, generation_zero_best_auc={:.17}, qualifying_winner={}",
        generation,
        checkpoint.fresh_evaluations,
        checkpoint.cache_reuses,
        checkpoint.best_auc,
        campaign.dataset_binding.leader_auc,
        checkpoint.generation_zero_best_auc,
        checkpoint.qualifying_winner.is_some()
    );
    if generation == 0 {
        let baseline = BaselineManifest {
            schema: BASELINE_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            dataset: campaign.dataset.clone(),
            leader_method: campaign.dataset_binding.leader_method.clone(),
            leader_auc: campaign.dataset_binding.leader_auc,
            leader_report_path: campaign.dataset_binding.leader_report_path.clone(),
            leader_report_sha256: campaign.dataset_binding.leader_report_sha256.clone(),
            generation_zero_best_auc: Some(generation_zero_best_auc),
        };
        write_json(
            &root
                .join("baselines")
                .join("generation-zero-baseline.v1.json"),
            &baseline,
        )?;
    }
    Ok(checkpoint)
}
