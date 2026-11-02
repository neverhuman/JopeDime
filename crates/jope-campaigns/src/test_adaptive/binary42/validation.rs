use super::*;

pub(super) fn validate_shard_accounting(
    root: &Path,
    campaign: &CampaignManifest,
    checkpoints: &[GenerationCheckpoint],
) -> Result<()> {
    let manifest_root = root.join("jobs").join("manifests");
    let shard_sizes = campaign_shard_sizes(&campaign.config);
    let mut primary = HashMap::<(usize, usize), ShardManifest>::new();
    if manifest_root.exists() {
        for entry in fs::read_dir(manifest_root)? {
            let path = entry?.path();
            let manifest: ShardManifest = load_json(&path)?;
            if manifest.schema != SHARD_SCHEMA
                || !manifest.test_adaptive
                || manifest.benchmark_valid
                || manifest.dataset != campaign.dataset
                || manifest.cohort_mode.as_deref()
                    != (!campaign.config.shard_sizes.is_empty()).then_some("all")
                || !matches!(manifest.kind.as_str(), "generation" | "confirmation")
                || manifest.job_id != expected_job_id(&manifest)?
            {
                bail!("invalid Binary42 immutable shard manifest");
            }
            if manifest.kind == "generation" {
                let start = shard_start(&shard_sizes, manifest.shard)?;
                if manifest.attempt > campaign.config.infrastructure_retries
                    || manifest.node != failover_node(manifest.shard, manifest.attempt)?
                    || manifest.slots.len() != shard_sizes[manifest.shard]
                    || manifest.cpus_reserved != shard_sizes[manifest.shard]
                    || manifest
                        .slots
                        .iter()
                        .enumerate()
                        .any(|(offset, slot)| slot.slot != start + offset)
                {
                    bail!("Binary42 immutable failover shard accounting mismatch");
                }
            } else if manifest.shard >= NODES.len()
                || manifest.node != NODES[manifest.shard]
                || manifest.slots.len() != 1
                || manifest.cpus_reserved != 1
            {
                bail!("Binary42 immutable confirmation shard accounting mismatch");
            }
            if manifest.kind == "generation"
                && manifest.attempt == 0
                && primary
                    .insert((manifest.generation, manifest.shard), manifest)
                    .is_some()
            {
                bail!("duplicate Binary42 primary shard manifest");
            }
        }
    }
    for checkpoint in checkpoints {
        for shard in 0..NODES.len() {
            let manifest = primary
                .get(&(checkpoint.generation, shard))
                .context("Binary42 checkpoint is missing a primary shard manifest")?;
            let start = shard_start(&shard_sizes, shard)?;
            if manifest.node != NODES[shard]
                || manifest.slots.len() != shard_sizes[shard]
                || manifest.cpus_reserved != shard_sizes[shard]
                || manifest.gpus_per_task != 0
                || manifest.candidate_threads != 1
                || manifest
                    .slots
                    .iter()
                    .enumerate()
                    .any(|(offset, slot)| slot.slot != start + offset)
            {
                bail!("Binary42 primary shard accounting mismatch");
            }
        }
    }
    Ok(())
}

pub(super) fn validate_artifact_manifest(root: &Path, campaign: &CampaignManifest) -> Result<()> {
    let path = root.join("winner").join("artifact-manifest.v1.json");
    let recorded: WinnerArtifactManifest = load_json(&path)?;
    if recorded.schema != "JopeBinary42TestAdaptiveArtifactManifest.v1"
        || !recorded.test_adaptive
        || recorded.benchmark_valid
        || recorded.run_id != campaign.run_id
        || recorded.dataset != campaign.dataset
    {
        bail!("Binary42 winner artifact manifest identity mismatch");
    }
    for artifact in &recorded.artifacts {
        let target = root.join(&artifact.path);
        if !target.is_file() || sha256_file(&target)? != artifact.sha256 {
            bail!(
                "Binary42 artifact changed or disappeared: {}",
                artifact.path
            );
        }
    }
    let rebuilt = collect_artifacts(root)?;
    if rebuilt.len() != recorded.artifacts.len()
        || rebuilt
            .iter()
            .zip(&recorded.artifacts)
            .any(|(left, right)| left.path != right.path || left.sha256 != right.sha256)
    {
        bail!("Binary42 complete artifact manifest is not exhaustive");
    }
    Ok(())
}

pub(super) fn validate_dataset_campaign(
    run_id: &str,
    dataset: &str,
    all_dataset_cohort: bool,
) -> Result<DatasetValidationSummary> {
    validate_run_id(run_id)?;
    let (spec, cohort_sha256) = if all_dataset_cohort {
        cohort_dataset(dataset)?
    } else {
        selected_dataset(dataset)?
    };
    verify_dataset_binding(&spec)?;
    let run = run_root(run_id);
    if !all_dataset_cohort {
        let entries = fs::read_dir(&run)
            .with_context(|| format!("Binary42 run {run_id} does not exist"))?
            .map(|entry| Ok(entry?.file_name()))
            .collect::<Result<Vec<_>>>()?;
        if entries.len() != 1 || entries[0] != std::ffi::OsStr::new(dataset) {
            bail!("Binary42 pilot run contains artifacts for a non-ILPD dataset");
        }
    }
    let root = campaign_root(run_id, dataset);
    let campaign: CampaignManifest = load_json(&root.join("campaign").join("campaign.v1.json"))?;
    let expected_config = if all_dataset_cohort {
        CampaignConfig::all(&spec)
    } else {
        CampaignConfig::fixed()
    };
    let expected_cohort_staging_sha256 = if all_dataset_cohort {
        Some(sha256_file(&run.join("cohort").join("staging.v1.json"))?)
    } else {
        None
    };
    let winner_seed_sources = if all_dataset_cohort {
        let cohort_campaign: AllCohortManifest =
            load_json(&run.join("cohort").join("campaign.v1.json"))?;
        validate_winner_seed_snapshot(&cohort_campaign.winner_seed_sources, &read_cohort()?.0)?;
        cohort_campaign.winner_seed_sources
    } else {
        Vec::new()
    };
    let expected_winner_seed_snapshot_sha256 = if all_dataset_cohort {
        Some(winner_seed_snapshot_sha256(&winner_seed_sources)?)
    } else {
        None
    };
    if campaign.schema != CAMPAIGN_SCHEMA
        || !campaign.test_adaptive
        || campaign.benchmark_valid
        || campaign.run_id != run_id
        || campaign.dataset != dataset
        || campaign.dataset_binding != spec
        || campaign.frozen_cohort_sha256 != cohort_sha256
        || campaign.binary_seed_sha256 != SEED_BINARY_SHA256
        || campaign.config != expected_config
        || campaign.cohort_staging_sha256 != expected_cohort_staging_sha256
        || campaign.winner_seed_snapshot_sha256 != expected_winner_seed_snapshot_sha256
        || sha256_file(&root.join(&campaign.frozen_cohort_file))? != cohort_sha256
    {
        bail!("Binary42 campaign manifest validation failed");
    }
    let staging: StagingReceipt = load_json(&root.join("staging").join("staging.v1.json"))?;
    validate_staging(
        &staging,
        &spec,
        &campaign.release_binary_sha256,
        &campaign.scq_binary_sha256,
        false,
    )?;
    if staging.cohort_staging_sha256 != campaign.cohort_staging_sha256 {
        bail!("Binary42 dataset staging does not bind the cohort staging receipt");
    }
    let seeds: SeedManifest = load_json(&root.join("seeds").join("generation-zero.v1.json"))?;
    let expected_seeds = seed_population_with_winners(spec.features, &winner_seed_sources)?;
    if seeds.schema != SEED_SCHEMA
        || !seeds.test_adaptive
        || seeds.benchmark_valid
        || seeds.seed_file_sha256 != SEED_BINARY_SHA256
        || seeds.population != POPULATION
        || seeds.members != expected_seeds
        || seeds.winner_seed_sources != winner_seed_sources
    {
        bail!("Binary42 seed manifest validation failed");
    }
    let seed_semantics = seeds
        .members
        .iter()
        .map(|member| member.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    let test = read_binary_csv(Path::new(&spec.test_path))?;
    let checkpoints = load_checkpoints(&root)?;
    if checkpoints.is_empty() {
        bail!("Binary42 campaign has no completed generation barrier");
    }
    if all_dataset_cohort
        && (checkpoints.len() < 2
            || checkpoints.len() > ALL_MAX_GENERATIONS
            || generation_root(&root, ALL_MAX_GENERATIONS).exists())
    {
        bail!("Binary42 all-dataset campaign must contain 2-10 generations numbered 0-9");
    }
    let cache = load_cache(&root)?;
    let mut validated_semantics = BTreeSet::new();
    let shard_sizes = campaign_shard_sizes(&campaign.config);
    for (generation, checkpoint) in checkpoints.iter().enumerate() {
        if checkpoint.run_id != run_id
            || checkpoint.dataset != dataset
            || checkpoint.generation != generation
            || checkpoint.fresh_evaluations + checkpoint.cache_reuses != POPULATION
        {
            bail!("Binary42 checkpoint campaign binding mismatch");
        }
        let generation_manifest_path =
            generation_root(&root, generation).join("generation-manifest.v1.json");
        let generation_manifest_bytes = fs::read(&generation_manifest_path)?;
        let generation_manifest: GenerationManifest =
            serde_json::from_slice(&generation_manifest_bytes)?;
        let expected_manifest_shard_sizes = if all_dataset_cohort {
            shard_sizes.clone()
        } else {
            Vec::new()
        };
        if generation_manifest_bytes != canonical_json(&generation_manifest)?
            || generation_manifest.schema != GENERATION_SCHEMA
            || !generation_manifest.test_adaptive
            || generation_manifest.benchmark_valid
            || generation_manifest.run_id != run_id
            || generation_manifest.dataset != dataset
            || generation_manifest.generation != generation
            || generation_manifest.logical_slots != POPULATION
            || generation_manifest.shard_nodes
                != NODES
                    .iter()
                    .map(|node| (*node).to_string())
                    .collect::<Vec<_>>()
            || generation_manifest.shard_sizes != expected_manifest_shard_sizes
            || generation_manifest.members
                != checkpoint
                    .slots
                    .iter()
                    .map(|slot| slot.member.clone())
                    .collect::<Vec<_>>()
        {
            bail!("Binary42 generation manifest validation failed");
        }
        for slot in &checkpoint.slots {
            if slot.assigned_node != assigned_node_for_slot(&shard_sizes, slot.slot)? {
                bail!("Binary42 logical slot assigned-node binding mismatch");
            }
        }
        let mut ranked = checkpoint
            .slots
            .iter()
            .filter(|slot| slot.score.is_some())
            .cloned()
            .collect::<Vec<_>>();
        ranked.sort_by(rank_slots);
        if ranked.len() != checkpoint.valid_scores
            || ranked
                .iter()
                .map(|slot| slot.member.semantic_sha256.clone())
                .collect::<Vec<_>>()
                != checkpoint.ranked_semantics
            || ranked.first().map(|slot| slot.slot) != Some(checkpoint.best_slot)
            || ranked
                .first()
                .and_then(|slot| slot.score.as_ref())
                .map(|score| score.auc.to_bits())
                != Some(checkpoint.best_auc.to_bits())
        {
            bail!("Binary42 checkpoint ranking authority mismatch");
        }
        let expected_gen0 = if generation == 0 {
            checkpoint.best_auc
        } else {
            checkpoints[0].best_auc
        };
        if checkpoint.generation_zero_best_auc.to_bits() != expected_gen0.to_bits() {
            bail!("Binary42 generation-zero baseline changed");
        }
        let expected_winner = qualifying_winner_for_policy(
            generation,
            &ranked,
            &seed_semantics,
            &spec.win_policy,
            spec.leader_auc,
            checkpoint.generation_zero_best_auc,
        );
        if checkpoint.qualifying_winner.as_ref().map(|winner| {
            (
                winner.slot,
                winner.semantic_sha256.as_str(),
                winner.auc.to_bits(),
            )
        }) != expected_winner.as_ref().map(|winner| {
            (
                winner.slot,
                winner.semantic_sha256.as_str(),
                winner.auc.to_bits(),
            )
        }) {
            bail!("Binary42 qualifying-winner comparator mismatch");
        }
        for slot in &checkpoint.slots {
            let cached = cache
                .get(&slot.member.semantic_sha256)
                .context("Binary42 logical slot lacks a semantic cache entry")?;
            if cached.receipt_sha256 != slot.receipt_sha256
                || cached.prediction_sha256 != slot.prediction_sha256
                || cached.score != slot.score
                || cached.failure != slot.failure
                || cached.runtime_seconds != slot.runtime_seconds
            {
                bail!("Binary42 logical slot differs from semantic cache");
            }
            if validated_semantics.insert(slot.member.semantic_sha256.clone()) {
                let genome_path = root
                    .join("objects")
                    .join("genomes")
                    .join(format!("{}.v1.json", slot.member.semantic_sha256));
                let genome: GenomeObject = load_json(&genome_path)?;
                if genome.semantic_sha256 != slot.member.semantic_sha256
                    || genome.canonical_genome != slot.member.canonical_genome
                    || genome.complexity != slot.member.complexity
                {
                    bail!("Binary42 content-addressed genome mismatch");
                }
                let receipt_path = root
                    .join("objects")
                    .join("receipts")
                    .join(format!("{}.json", cached.receipt_sha256));
                let prediction_path = root
                    .join("objects")
                    .join("predictions")
                    .join(format!("{}.csv", cached.prediction_sha256));
                if sha256_file(&receipt_path)? != cached.receipt_sha256
                    || sha256_file(&prediction_path)? != cached.prediction_sha256
                {
                    bail!("Binary42 content-addressed evaluation object mismatch");
                }
                let receipt_bytes = fs::read(receipt_path)?;
                let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
                validate_candidate_backend(&receipt, all_dataset_cohort)?;
                validate_fresh_output(
                    &FreshOutput {
                        receipt,
                        receipt_bytes,
                        prediction_bytes: fs::read(prediction_path)?,
                    },
                    &test,
                )?;
            }
        }
        let reached_cap = generation_cap_reached(&campaign.config, generation);
        if checkpoint.qualifying_winner.is_none() && !reached_cap {
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
            let expected_next =
                test_adaptive_next_population(RNG_SEED, generation + 1, &parents, spec.features)?
                    .into_iter()
                    .map(MemberRecord::from_binary)
                    .collect::<Result<Vec<_>>>()?;
            if checkpoint.next_population != expected_next {
                bail!("Binary42 deterministic next population changed");
            }
        } else if !checkpoint.next_population.is_empty() {
            bail!("Binary42 terminal checkpoint unexpectedly continues evolution");
        }
    }
    validate_shard_accounting(&root, &campaign, &checkpoints)?;
    rebuild_indexes(&root, &checkpoints, all_dataset_cohort)?;
    let complete = root.join("RUN_COMPLETE").exists();
    let exhausted = root.join("RUN_EXHAUSTED").exists();
    if complete && exhausted {
        bail!("Binary42 campaign has conflicting terminal receipts");
    }
    if complete {
        if checkpoints.last().unwrap().qualifying_winner.is_none()
            || checkpoints
                .iter()
                .flat_map(|checkpoint| &checkpoint.slots)
                .any(|slot| slot.score.is_none() || slot.failure.is_some())
        {
            bail!("Binary42 complete campaign has missing unique-genome scores");
        }
        let confirmation =
            existing_confirmation(&root)?.context("Binary42 completion lacks confirmations")?;
        if confirmation.confirmations.len() != NODES.len()
            || confirmation
                .confirmations
                .iter()
                .enumerate()
                .any(|(index, record)| record.node != NODES[index])
        {
            bail!("Binary42 completion lacks exact four-node replay evidence");
        }
        let final_receipt: WinnerReceipt =
            load_json(&root.join("winner").join("final-receipt.v1.json"))?;
        if final_receipt.schema != WINNER_SCHEMA
            || !final_receipt.test_adaptive
            || final_receipt.benchmark_valid
            || !policy_win(
                &spec.win_policy,
                final_receipt.auc,
                final_receipt.frozen_leader_auc,
                final_receipt.generation_zero_best_auc,
            )
            || seed_semantics.contains(&final_receipt.semantic_sha256)
            || confirmation.semantic_sha256 != final_receipt.semantic_sha256
            || confirmation.confirmations.iter().any(|record| {
                record.prediction_sha256 != final_receipt.prediction_sha256
                    || record.auc_bits != final_receipt.auc.to_bits()
            })
        {
            bail!("Binary42 final winner receipt validation failed");
        }
        validate_artifact_manifest(&root, &campaign)?;
        let complete_bytes = fs::read(root.join("RUN_COMPLETE"))?;
        let complete_marker: Value = serde_json::from_slice(&complete_bytes)?;
        let final_receipt_sha256 = sha256_bytes(&canonical_json(&final_receipt)?);
        let artifact_manifest_sha256 =
            sha256_file(&root.join("winner").join("artifact-manifest.v1.json"))?;
        if complete_bytes != canonical_json(&complete_marker)?
            || complete_marker.get("schema").and_then(Value::as_str) != Some(COMPLETE_SCHEMA)
            || complete_marker
                .get("test_adaptive")
                .and_then(Value::as_bool)
                != Some(true)
            || complete_marker
                .get("benchmark_valid")
                .and_then(Value::as_bool)
                != Some(false)
            || complete_marker.get("status").and_then(Value::as_str) != Some("complete")
            || complete_marker.get("run_id").and_then(Value::as_str) != Some(run_id)
            || complete_marker.get("dataset").and_then(Value::as_str) != Some(dataset)
            || complete_marker
                .get("winner_semantic_sha256")
                .and_then(Value::as_str)
                != Some(final_receipt.semantic_sha256.as_str())
            || complete_marker
                .get("final_receipt_sha256")
                .and_then(Value::as_str)
                != Some(final_receipt_sha256.as_str())
            || complete_marker
                .get("artifact_manifest_sha256")
                .and_then(Value::as_str)
                != Some(artifact_manifest_sha256.as_str())
        {
            bail!("Binary42 RUN_COMPLETE validation failed");
        }
    }
    if exhausted {
        if !all_dataset_cohort
            || checkpoints.len() != ALL_MAX_GENERATIONS
            || checkpoints
                .iter()
                .any(|checkpoint| checkpoint.qualifying_winner.is_some())
            || !checkpoint_next_populations_terminate_at_cap(&checkpoints)
        {
            bail!("Binary42 exhausted campaign violates the generation-9 terminal policy");
        }
        let receipt_path = root.join("best-effort").join("final-receipt.v1.json");
        let receipt_bytes = fs::read(&receipt_path)?;
        let receipt: ExhaustedReceipt = serde_json::from_slice(&receipt_bytes)?;
        let marker_bytes = fs::read(root.join("RUN_EXHAUSTED"))?;
        let marker: ExhaustedMarker = serde_json::from_slice(&marker_bytes)?;
        if receipt_bytes != canonical_json(&receipt)?
            || receipt.schema != EXHAUSTED_SCHEMA
            || !receipt.test_adaptive
            || receipt.benchmark_valid
            || receipt.status != "exhausted"
            || receipt.run_id != run_id
            || receipt.dataset != dataset
            || receipt.completed_generations != ALL_MAX_GENERATIONS
            || receipt.logical_slots != ALL_MAX_GENERATIONS * POPULATION
            || receipt.release_binary_sha256 != campaign.release_binary_sha256
            || marker_bytes != canonical_json(&marker)?
            || marker.schema != EXHAUSTED_SCHEMA
            || !marker.test_adaptive
            || marker.benchmark_valid
            || marker.status != "exhausted"
            || marker.run_id != run_id
            || marker.dataset != dataset
            || marker.completed_generations != ALL_MAX_GENERATIONS
            || marker.logical_slots != ALL_MAX_GENERATIONS * POPULATION
            || marker.best_semantic_sha256 != receipt.best_semantic_sha256
            || marker.best_auc.to_bits() != receipt.best_auc.to_bits()
            || marker.final_receipt_sha256 != sha256_bytes(&receipt_bytes)
        {
            bail!("Binary42 best-effort receipt validation failed");
        }
        let mut ranked = checkpoints
            .iter()
            .flat_map(|checkpoint| &checkpoint.slots)
            .filter(|slot| slot.score.is_some())
            .cloned()
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            rank_slots(left, right)
                .then_with(|| left.generation.cmp(&right.generation))
                .then_with(|| left.slot.cmp(&right.slot))
        });
        let best = ranked
            .first()
            .context("Binary42 exhausted campaign has no best score")?;
        let score = best.score.as_ref().unwrap();
        if receipt.best_generation != best.generation
            || receipt.best_slot != best.slot
            || receipt.best_semantic_sha256 != best.member.semantic_sha256
            || receipt.best_auc.to_bits() != score.auc.to_bits()
            || receipt.best_logloss.to_bits() != score.logloss.to_bits()
            || receipt.best_prediction_sha256 != best.prediction_sha256
            || receipt.best_receipt_sha256 != best.receipt_sha256
            || sha256_file(&root.join("best-effort").join("predictions.csv"))?
                != best.prediction_sha256
            || sha256_file(&root.join("best-effort").join("lineage.v1.json"))?
                != receipt.lineage_sha256
        {
            bail!("Binary42 best-effort artifact validation failed");
        }
        let genome: GenomeObject = load_json(&root.join("best-effort").join("genome.v1.json"))?;
        if genome.semantic_sha256 != best.member.semantic_sha256
            || genome.canonical_genome != best.member.canonical_genome
            || genome.complexity != best.member.complexity
        {
            bail!("Binary42 best-effort genome validation failed");
        }
    }
    if all_dataset_cohort && !complete && !exhausted {
        bail!("Binary42 all-dataset validation requires a terminal dataset");
    }
    let status = if complete {
        "complete"
    } else if exhausted {
        "exhausted"
    } else {
        "in_progress"
    };
    let winner = checkpoints
        .iter()
        .find_map(|checkpoint| checkpoint.qualifying_winner.as_ref());
    Ok(DatasetValidationSummary {
        dataset: spec.id,
        display_name: spec.display_name,
        frozen_order: spec.order,
        status: status.to_string(),
        completed_generations: checkpoints.len(),
        logical_slots: checkpoints.len() * POPULATION,
        unique_genomes: validated_semantics.len(),
        best_auc: checkpoints
            .iter()
            .map(|checkpoint| checkpoint.best_auc)
            .max_by(f64::total_cmp),
        winner_generation: winner.map(|record| record.generation),
        winner_semantic_sha256: winner.map(|record| record.semantic_sha256.clone()),
    })
}

pub(super) fn checkpoint_next_populations_terminate_at_cap(
    checkpoints: &[GenerationCheckpoint],
) -> bool {
    checkpoints
        .last()
        .is_some_and(|checkpoint| checkpoint.next_population.is_empty())
}

pub(super) fn validate_all_impl(run_id: &str) -> Result<()> {
    validate_run_id(run_id)?;
    let (cohort, cohort_sha256) = read_cohort()?;
    let run = run_root(run_id);
    let cohort_root = run.join("cohort");
    let campaign_path = cohort_root.join("campaign.v1.json");
    let campaign_bytes = fs::read(&campaign_path)
        .with_context(|| format!("Binary42 all-dataset run {run_id} does not exist"))?;
    let campaign: AllCohortManifest = serde_json::from_slice(&campaign_bytes)?;
    let resolved_scq = fs::canonicalize(all_scq_binary_str())?;
    let resolved_scq_path = resolved_scq.to_string_lossy().into_owned();
    let ordered = admission_order(&cohort);
    if campaign_bytes != canonical_json(&campaign)?
        || campaign.schema != ALL_CAMPAIGN_SCHEMA
        || !campaign.test_adaptive
        || campaign.benchmark_valid
        || campaign.run_id != run_id
        || campaign.frozen_cohort_sha256 != cohort_sha256
        || campaign.release_binary_sha256 != current_binary_sha256()?
        || campaign.scq_configured_path != all_scq_binary_str()
        || campaign.scq_resolved_path != resolved_scq_path
        || campaign.scq_binary_sha256 != sha256_file(&resolved_scq)?
        || campaign.config != AllCohortConfig::fixed()
        || campaign.admission_dataset_ids
            != ordered
                .iter()
                .map(|dataset| dataset.id.clone())
                .collect::<Vec<_>>()
        || sha256_file(&cohort_root.join(&campaign.frozen_cohort_file))? != cohort_sha256
    {
        bail!("Binary42 all-dataset cohort manifest validation failed");
    }
    validate_winner_seed_snapshot(&campaign.winner_seed_sources, &cohort)?;
    let staging_path = cohort_root.join("staging.v1.json");
    let staging_bytes = fs::read(&staging_path)?;
    let staging: AllStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    if staging_bytes != canonical_json(&staging)? {
        bail!("Binary42 all-dataset staging receipt is not canonical JSON");
    }
    validate_all_staging(
        &staging,
        &cohort,
        &campaign.release_binary_sha256,
        &campaign.scq_resolved_path,
        &campaign.scq_binary_sha256,
        false,
    )?;
    let expected_entries = std::iter::once("cohort".to_string())
        .chain(cohort.datasets.iter().map(|dataset| dataset.id.clone()))
        .collect::<BTreeSet<_>>();
    let actual_entries = fs::read_dir(&run)?
        .map(|entry| {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                bail!("unexpected non-directory artifact at Binary42 cohort run root");
            }
            Ok(entry.file_name().to_string_lossy().into_owned())
        })
        .collect::<Result<BTreeSet<_>>>()?;
    if actual_entries != expected_entries {
        bail!("Binary42 all-dataset run has missing or unknown dataset directories");
    }
    let mut validated = Vec::with_capacity(cohort.datasets.len());
    for spec in &cohort.datasets {
        let summary = validate_dataset_campaign(run_id, &spec.id, true)?;
        if !matches!(summary.status.as_str(), "complete" | "exhausted")
            || !(2..=ALL_MAX_GENERATIONS).contains(&summary.completed_generations)
            || summary.logical_slots != summary.completed_generations * POPULATION
        {
            bail!(
                "Binary42 dataset {} is not terminal and barrier-complete",
                spec.id
            );
        }
        validated.push(summary);
    }
    let expected_logical = validated
        .iter()
        .map(|summary| summary.logical_slots)
        .sum::<usize>();
    let (rebuilt, rebuilt_logical) = rebuild_cohort_indexes(run_id, &cohort)?;
    if rebuilt != validated || rebuilt_logical != expected_logical {
        bail!("Binary42 global index counts differ from per-dataset ledgers");
    }
    let jsonl_count = fs::read_to_string(cohort_root.join("all-evaluations.jsonl"))?
        .lines()
        .count();
    let csv_count = fs::read_to_string(cohort_root.join("all-evaluations.csv"))?
        .lines()
        .count()
        .saturating_sub(1);
    if jsonl_count != expected_logical || csv_count != expected_logical {
        bail!("Binary42 global evaluation indexes have the wrong logical row count");
    }
    let summary_path = cohort_root.join("terminal-summary.v1.json");
    let summary_bytes = fs::read(&summary_path)?;
    let summary: AllTerminalSummary = serde_json::from_slice(&summary_bytes)?;
    let winners = validated
        .iter()
        .filter(|dataset| dataset.status == "complete")
        .count();
    let exhausted = validated.len() - winners;
    if summary_bytes != canonical_json(&summary)?
        || summary.schema != ALL_SUMMARY_SCHEMA
        || !summary.test_adaptive
        || summary.benchmark_valid
        || summary.run_id != run_id
        || summary.terminal_datasets != 42
        || summary.winners != winners
        || summary.exhausted != exhausted
        || summary.logical_evaluations != expected_logical
    {
        bail!("Binary42 all-dataset terminal summary validation failed");
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "JopeBinary42AllValidation.v1",
            "test_adaptive": true,
            "benchmark_valid": false,
            "run_id": run_id,
            "terminal_datasets": validated.len(),
            "winners": winners,
            "exhausted": exhausted,
            "logical_evaluations": expected_logical,
            "four_node_confirmed_winners": winners,
            "status": "complete",
        }))?
    );
    Ok(())
}

pub(super) fn validate_impl(run_id: &str, dataset: &str) -> Result<()> {
    let summary = validate_dataset_campaign(run_id, dataset, false)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "JopeBinary42TestAdaptiveValidation.v1",
            "test_adaptive": true,
            "benchmark_valid": false,
            "run_id": run_id,
            "dataset": dataset,
            "status": summary.status,
            "completed_generations": summary.completed_generations,
            "logical_slots": summary.logical_slots,
            "unique_genomes": summary.unique_genomes,
            "four_node_confirmed": summary.status == "complete",
        }))?
    );
    Ok(())
}
