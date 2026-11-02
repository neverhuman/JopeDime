use super::*;

pub(crate) fn validate_shard_accounting(
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

pub(crate) fn validate_artifact_manifest(root: &Path, campaign: &CampaignManifest) -> Result<()> {
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

mod dataset;
pub(crate) use dataset::validate_dataset_campaign;

pub(crate) fn checkpoint_next_populations_terminate_at_cap(
    checkpoints: &[GenerationCheckpoint],
) -> bool {
    checkpoints
        .last()
        .is_some_and(|checkpoint| checkpoint.next_population.is_empty())
}

pub(crate) fn validate_all_impl(run_id: &str) -> Result<()> {
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

pub(crate) fn validate_impl(run_id: &str, dataset: &str) -> Result<()> {
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
