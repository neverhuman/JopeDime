use super::*;

pub(in crate::test_adaptive::binary42) fn existing_confirmation(
    root: &Path,
) -> Result<Option<ConfirmationManifest>> {
    let confirmation_root = root.join("winner").join("confirmations");
    if !confirmation_root.exists() {
        return Ok(None);
    }
    let mut attempts = fs::read_dir(confirmation_root)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    attempts.sort();
    for attempt in attempts.into_iter().rev() {
        let path = attempt.join("confirmation.v1.json");
        if path.exists() {
            let confirmation: ConfirmationManifest = load_json(&path)?;
            if confirmation.schema != CONFIRMATION_SCHEMA
                || !confirmation.test_adaptive
                || confirmation.benchmark_valid
                || !confirmation.prediction_hashes_identical
                || !confirmation.auc_bits_identical
                || confirmation.confirmations.len() != NODES.len()
            {
                bail!("existing Binary42 confirmation is invalid");
            }
            return Ok(Some(confirmation));
        }
    }
    Ok(None)
}

pub(in crate::test_adaptive::binary42) fn initialize_campaign(
    run_id: &str,
    dataset: &str,
    resume: bool,
) -> Result<(PathBuf, CampaignManifest, StagingReceipt)> {
    validate_run_id(run_id)?;
    let (spec, cohort_sha256) = selected_dataset(dataset)?;
    verify_dataset_binding(&spec)?;
    let root = campaign_root(run_id, dataset);
    let campaign_path = root.join("campaign").join("campaign.v1.json");
    let release_binary_sha256 = current_binary_sha256()?;
    let scq_binary_sha256 = sha256_file(Path::new(scq_binary_str()))?;
    let campaign = if campaign_path.exists() {
        if !resume && !root.join("RUN_COMPLETE").exists() {
            bail!("Binary42 campaign already exists; pass --resume");
        }
        let recorded: CampaignManifest = load_json(&campaign_path)?;
        if recorded.schema != CAMPAIGN_SCHEMA
            || !recorded.test_adaptive
            || recorded.benchmark_valid
            || recorded.run_id != run_id
            || recorded.dataset != dataset
            || recorded.frozen_cohort_sha256 != cohort_sha256
            || recorded.binary_seed_sha256 != SEED_BINARY_SHA256
            || recorded.release_binary_sha256 != release_binary_sha256
            || recorded.scq_binary_sha256 != scq_binary_sha256
            || recorded.dataset_binding != spec
            || recorded.config != CampaignConfig::fixed()
            || recorded.cohort_staging_sha256.is_some()
            || recorded.winner_seed_snapshot_sha256.is_some()
        {
            bail!("Binary42 resume identity changed");
        }
        recorded
    } else {
        if resume {
            bail!("cannot resume a Binary42 campaign that does not exist");
        }
        fs::create_dir_all(&root)?;
        immutable_write(
            &root.join("campaign").join("binary42-frozen-cohort.v1.json"),
            COHORT_BYTES,
        )?;
        let created = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            dataset: dataset.to_string(),
            frozen_cohort_file: "campaign/binary42-frozen-cohort.v1.json".to_string(),
            frozen_cohort_sha256: cohort_sha256,
            binary_seed_sha256: SEED_BINARY_SHA256.to_string(),
            release_binary_sha256,
            scq_binary_sha256,
            dataset_binding: spec,
            config: CampaignConfig::fixed(),
            started_at_unix_ms: unix_millis(),
            cohort_staging_sha256: None,
            winner_seed_snapshot_sha256: None,
        };
        write_json(&campaign_path, &created)?;
        let baseline = BaselineManifest {
            schema: BASELINE_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            dataset: dataset.to_string(),
            leader_method: created.dataset_binding.leader_method.clone(),
            leader_auc: created.dataset_binding.leader_auc,
            leader_report_path: created.dataset_binding.leader_report_path.clone(),
            leader_report_sha256: created.dataset_binding.leader_report_sha256.clone(),
            generation_zero_best_auc: None,
        };
        write_json(
            &root.join("baselines").join("frozen-leader.v1.json"),
            &baseline,
        )?;
        let members = seed_population(created.dataset_binding.features)?;
        let seeds = SeedManifest {
            schema: SEED_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            seed_file_sha256: SEED_BINARY_SHA256.to_string(),
            population: POPULATION,
            members,
            winner_seed_sources: Vec::new(),
        };
        write_json(&root.join("seeds").join("generation-zero.v1.json"), &seeds)?;
        created
    };
    let staging = stage_campaign(
        &root,
        &campaign.dataset_binding,
        &campaign.release_binary_sha256,
        &campaign.scq_binary_sha256,
    )?;
    Ok((root, campaign, staging))
}

pub(in crate::test_adaptive::binary42) fn dataset_staging_from_cohort(
    spec: &FrozenDataset,
    cohort_staging: &AllStagingReceipt,
    cohort_staging_sha256: &str,
) -> Result<StagingReceipt> {
    let mut nodes = Vec::with_capacity(NODES.len());
    for staged_node in &cohort_staging.nodes {
        let dataset = staged_node
            .datasets
            .iter()
            .find(|dataset| dataset.dataset == spec.id)
            .with_context(|| format!("Binary42 cohort staging omits dataset {}", spec.id))?;
        nodes.push(StagedNode {
            node: staged_node.node.clone(),
            hostname: staged_node.hostname.clone(),
            bundle_root: staged_node.bundle_root.clone(),
            binary_path: staged_node.binary_path.clone(),
            binary_sha256: staged_node.binary_sha256.clone(),
            train_path: dataset.train_path.clone(),
            train_sha256: dataset.train_sha256.clone(),
            test_path: dataset.test_path.clone(),
            test_sha256: dataset.test_sha256.clone(),
            canary_job_id: staged_node.canary_job_id.clone(),
            canary_show_sha256: staged_node.canary_show_sha256.clone(),
        });
    }
    Ok(StagingReceipt {
        schema: STAGING_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        release_binary_sha256: cohort_staging.release_binary_sha256.clone(),
        scq_binary_sha256: cohort_staging.scq_binary_sha256.clone(),
        doctor_output_sha256: cohort_staging.doctor_output_sha256.clone(),
        dataset: spec.id.clone(),
        nodes,
        cohort_staging_sha256: Some(cohort_staging_sha256.to_string()),
    })
}

pub(in crate::test_adaptive::binary42) fn initialize_all_dataset(
    run_id: &str,
    spec: &FrozenDataset,
    cohort_sha256: &str,
    cohort_manifest: &AllCohortManifest,
    cohort_staging: &AllStagingReceipt,
    cohort_staging_sha256: &str,
    resume: bool,
) -> Result<(PathBuf, CampaignManifest, StagingReceipt)> {
    let root = campaign_root(run_id, &spec.id);
    let campaign_path = root.join("campaign").join("campaign.v1.json");
    let expected_config = CampaignConfig::all(spec);
    let winner_seed_snapshot_sha256 =
        winner_seed_snapshot_sha256(&cohort_manifest.winner_seed_sources)?;
    let campaign = if campaign_path.exists() {
        if !resume && !root.join("RUN_COMPLETE").exists() && !root.join("RUN_EXHAUSTED").exists() {
            bail!(
                "Binary42 all-dataset campaign {} already exists; pass --resume",
                spec.id
            );
        }
        let recorded: CampaignManifest = load_json(&campaign_path)?;
        if recorded.schema != CAMPAIGN_SCHEMA
            || !recorded.test_adaptive
            || recorded.benchmark_valid
            || recorded.run_id != run_id
            || recorded.dataset != spec.id
            || recorded.frozen_cohort_sha256 != cohort_sha256
            || recorded.binary_seed_sha256 != SEED_BINARY_SHA256
            || recorded.release_binary_sha256 != cohort_manifest.release_binary_sha256
            || recorded.scq_binary_sha256 != cohort_manifest.scq_binary_sha256
            || recorded.dataset_binding != *spec
            || recorded.config != expected_config
            || recorded.cohort_staging_sha256.as_deref() != Some(cohort_staging_sha256)
            || recorded.winner_seed_snapshot_sha256.as_deref()
                != Some(winner_seed_snapshot_sha256.as_str())
        {
            bail!(
                "Binary42 all-dataset resume identity changed for {}",
                spec.id
            );
        }
        recorded
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(
            &root.join("campaign").join("binary42-frozen-cohort.v1.json"),
            COHORT_BYTES,
        )?;
        let created = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            dataset: spec.id.clone(),
            frozen_cohort_file: "campaign/binary42-frozen-cohort.v1.json".to_string(),
            frozen_cohort_sha256: cohort_sha256.to_string(),
            binary_seed_sha256: SEED_BINARY_SHA256.to_string(),
            release_binary_sha256: cohort_manifest.release_binary_sha256.clone(),
            scq_binary_sha256: cohort_manifest.scq_binary_sha256.clone(),
            dataset_binding: spec.clone(),
            config: expected_config,
            started_at_unix_ms: unix_millis(),
            cohort_staging_sha256: Some(cohort_staging_sha256.to_string()),
            winner_seed_snapshot_sha256: Some(winner_seed_snapshot_sha256),
        };
        write_json(&campaign_path, &created)?;
        let baseline = BaselineManifest {
            schema: BASELINE_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            dataset: spec.id.clone(),
            leader_method: spec.leader_method.clone(),
            leader_auc: spec.leader_auc,
            leader_report_path: spec.leader_report_path.clone(),
            leader_report_sha256: spec.leader_report_sha256.clone(),
            generation_zero_best_auc: None,
        };
        write_json(
            &root.join("baselines").join("frozen-leader.v1.json"),
            &baseline,
        )?;
        let members =
            seed_population_with_winners(spec.features, &cohort_manifest.winner_seed_sources)?;
        let seeds = SeedManifest {
            schema: SEED_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            seed_file_sha256: SEED_BINARY_SHA256.to_string(),
            population: POPULATION,
            members,
            winner_seed_sources: cohort_manifest.winner_seed_sources.clone(),
        };
        write_json(&root.join("seeds").join("generation-zero.v1.json"), &seeds)?;
        created
    };
    let staging = dataset_staging_from_cohort(spec, cohort_staging, cohort_staging_sha256)?;
    write_json(&root.join("staging").join("staging.v1.json"), &staging)?;
    validate_staging(
        &staging,
        spec,
        &campaign.release_binary_sha256,
        &campaign.scq_binary_sha256,
        false,
    )?;
    Ok((root, campaign, staging))
}
