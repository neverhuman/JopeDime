fn initialize(run_id: &str, resume: bool) -> Result<(CampaignManifest, StagingReceipt, String)> {
    validate_run_id(run_id)?;
    let expected_coverage = if is_forecast3_gen20(run_id) {
        Some(validate_coverage_binding()?)
    } else {
        None
    };
    let root = campaign_root(run_id);
    fs::create_dir_all(&root)?;
    let release_sha256 = current_binary_sha256()?;
    let corpus_eval_sha256 = current_corpus_eval_sha256()?;
    let scq_binary_sha256 = sha256_file(Path::new(scq_binary_str()))?;
    let manifest_path = root.join("campaign.v1.json");
    let expected_config = base_campaign_config(run_id);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("TimeSeries42 campaign exists; pass --resume to reuse it");
        }
        let bytes = fs::read(&manifest_path)?;
        let manifest: CampaignManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&manifest)?
            || manifest.schema != CAMPAIGN_SCHEMA
            || !manifest.test_adaptive
            || manifest.benchmark_valid
            || manifest.run_id != run_id
            || manifest.config != expected_config
            || manifest.coverage_binding != expected_coverage
            || manifest.release_binary_sha256 != release_sha256
            || manifest.corpus_eval_sha256 != corpus_eval_sha256
            || manifest.scq_binary != scq_binary_str()
            || manifest.scq_binary_sha256 != scq_binary_sha256
            || (is_forecast3_gen20(run_id)
                && manifest.seed_populations
                    != seed_population_bindings(&root, &manifest.datasets, false)?)
        {
            bail!(
                "existing TimeSeries42 campaign manifest is invalid or bound to a different binary"
            );
        }
        manifest
    } else {
        let mut specs = load_dataset_specs(false)?;
        if let Some(coverage) = expected_coverage.as_ref() {
            bind_forecast3_comparators(&mut specs, coverage)?;
        }
        let ids = ordered_ids(&specs);
        let seed_populations = if is_forecast3_gen20(run_id) {
            seed_population_bindings(&root, &specs, true)?
        } else {
            Vec::new()
        };
        let manifest = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            base_run_id: None,
            base_campaign_sha256: None,
            continuation_start_generation: None,
            continuation_terminal_generation: None,
            registry_root: registry_str().to_string(),
            output_root: output_root_str().to_string(),
            release_binary_sha256: release_sha256.clone(),
            corpus_eval_sha256: corpus_eval_sha256.clone(),
            scq_binary: scq_binary_str().to_string(),
            scq_binary_sha256,
            coverage_binding: expected_coverage,
            seed_populations,
            ordered_dataset_ids: ids.clone(),
            ordered_cohort_sha256: ordered_cohort_sha256(&ids),
            config: expected_config,
            datasets: specs,
            started_at_unix_ms: unix_millis(),
        };
        write_json(&manifest_path, &manifest)?;
        manifest
    };
    let campaign_sha256 = sha256_file(&manifest_path)?;
    let (staging, _) = stage_campaign(
        &root,
        &manifest.datasets,
        &manifest.release_binary_sha256,
        &manifest.corpus_eval_sha256,
    )?;
    preflight_representative_scq_checks(&root, &staging, run_id)?;
    Ok((manifest, staging, campaign_sha256))
}

#[derive(Clone)]
struct BaseCampaign {
    root: PathBuf,
    manifest: CampaignManifest,
    campaign_sha256: String,
}

fn load_base_campaign(base_run_id: &str) -> Result<BaseCampaign> {
    validate_run_id(base_run_id)?;
    let root = campaign_root(base_run_id);
    let manifest_path = root.join("campaign.v1.json");
    let summary_path = root.join("terminal-summary.v1.json");
    let manifest: CampaignManifest = load_json(&manifest_path)?;
    let summary: TerminalSummary = load_json(&summary_path)?;
    let campaign_sha256 = sha256_file(&manifest_path)?;
    if manifest.schema != CAMPAIGN_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.run_id != base_run_id
        || manifest.base_run_id.is_some()
        || manifest.config.generations != GENERATIONS
        || manifest.datasets.len() != 42
        || summary.schema != SUMMARY_SCHEMA
        || !summary.test_adaptive
        || summary.benchmark_valid
        || summary.run_id != base_run_id
        || summary.dataset_count != 42
        || summary.wins + summary.losses + summary.ties + summary.exhausted != 42
    {
        bail!("invalid TimeSeries42 base campaign {base_run_id}");
    }
    let mut summary_ids = summary
        .datasets
        .iter()
        .map(|row| row.dataset_hash.clone())
        .collect::<Vec<_>>();
    summary_ids.sort();
    let mut manifest_ids = ordered_ids(&manifest.datasets);
    manifest_ids.sort();
    if summary_ids != manifest_ids {
        bail!("TimeSeries42 base summary/dataset manifest mismatch");
    }
    for spec in &manifest.datasets {
        let result: DatasetResult = load_json(&final_result_path(&root, &spec.dataset_hash))?;
        if !result_matches_campaign(
            &result,
            base_run_id,
            &campaign_sha256,
            spec,
            &manifest.release_binary_sha256,
        ) || result.completed_generations != GENERATIONS
        {
            bail!(
                "invalid TimeSeries42 base dataset result for {}",
                spec.dataset_hash
            );
        }
    }
    Ok(BaseCampaign {
        root,
        manifest,
        campaign_sha256,
    })
}

#[derive(Clone)]
struct WinallSourceCampaign {
    base: BaseCampaign,
    root: PathBuf,
    manifest: CampaignManifest,
    campaign_sha256: String,
    summary: TerminalSummary,
    terminal_generation: usize,
}

fn load_winall_source_campaign(source_run_id: &str) -> Result<WinallSourceCampaign> {
    validate_run_id(source_run_id)?;
    let root = campaign_root(source_run_id);
    let manifest_path = root.join("campaign.v1.json");
    let summary_path = root.join("terminal-summary.v1.json");
    let manifest: CampaignManifest = load_json(&manifest_path)?;
    let summary: TerminalSummary = load_json(&summary_path)?;
    let campaign_sha256 = sha256_file(&manifest_path)?;
    let base_run_id = manifest
        .base_run_id
        .as_deref()
        .context("TimeSeries42 win-all source has no base_run_id")?;
    let base = load_base_campaign(base_run_id)?;
    let terminal_generation = manifest
        .continuation_terminal_generation
        .unwrap_or_else(|| manifest.config.generations.saturating_sub(1));
    if manifest.schema != CAMPAIGN_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.run_id != source_run_id
        || manifest.base_campaign_sha256.as_deref() != Some(&base.campaign_sha256)
        || manifest.continuation_start_generation != Some(GEN30_START_GENERATION)
        || terminal_generation + 1 != manifest.config.generations
        || manifest.datasets != base.manifest.datasets
        || summary.schema != SUMMARY_SCHEMA
        || !summary.test_adaptive
        || summary.benchmark_valid
        || summary.run_id != source_run_id
        || summary.dataset_count != manifest.datasets.len()
        || summary.wins + summary.losses + summary.ties + summary.exhausted
            != manifest.datasets.len()
    {
        bail!("invalid TimeSeries42 win-all source campaign {source_run_id}");
    }
    for spec in &manifest.datasets {
        let result: DatasetResult = load_json(&final_result_path(&root, &spec.dataset_hash))?;
        if !result_matches_campaign(
            &result,
            source_run_id,
            &campaign_sha256,
            spec,
            &manifest.release_binary_sha256,
        ) {
            bail!(
                "invalid TimeSeries42 win-all source result for {}",
                spec.dataset_hash
            );
        }
    }
    Ok(WinallSourceCampaign {
        base,
        root,
        manifest,
        campaign_sha256,
        summary,
        terminal_generation,
    })
}

fn winall_source_non_winners(source: &WinallSourceCampaign) -> Result<Vec<DatasetSpec>> {
    let mut results = BTreeMap::new();
    for spec in &source.manifest.datasets {
        let result: DatasetResult =
            load_json(&final_result_path(&source.root, &spec.dataset_hash))?;
        results.insert(spec.dataset_hash.as_str(), result.strict_winner);
    }
    let specs = source
        .manifest
        .datasets
        .iter()
        .filter(|spec| {
            !results
                .get(spec.dataset_hash.as_str())
                .copied()
                .unwrap_or(false)
        })
        .cloned()
        .collect::<Vec<_>>();
    if specs.is_empty() {
        bail!(
            "TimeSeries42 source campaign {} already has strict winners for every dataset",
            source.manifest.run_id
        );
    }
    Ok(specs)
}

fn initialize_gen30(
    base_run_id: &str,
    run_id: &str,
    resume: bool,
) -> Result<(BaseCampaign, CampaignManifest, StagingReceipt, String)> {
    validate_run_id(run_id)?;
    if base_run_id == run_id {
        bail!("TimeSeries42 gen30 continuation run id must differ from the base run id");
    }
    let base = load_base_campaign(base_run_id)?;
    let root = campaign_root(run_id);
    fs::create_dir_all(&root)?;
    let release_sha256 = current_binary_sha256()?;
    let corpus_eval_sha256 = current_corpus_eval_sha256()?;
    let scq_binary_sha256 = sha256_file(Path::new(scq_binary_str()))?;
    let manifest_path = root.join("campaign.v1.json");
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("TimeSeries42 gen30 campaign exists; pass --resume to reuse it");
        }
        let bytes = fs::read(&manifest_path)?;
        let manifest: CampaignManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&manifest)?
            || manifest.schema != CAMPAIGN_SCHEMA
            || !manifest.test_adaptive
            || manifest.benchmark_valid
            || manifest.run_id != run_id
            || manifest.base_run_id.as_deref() != Some(base_run_id)
            || manifest.base_campaign_sha256.as_deref() != Some(&base.campaign_sha256)
            || manifest.continuation_start_generation != Some(GEN30_START_GENERATION)
            || manifest.continuation_terminal_generation != Some(GEN30_GENERATIONS - 1)
            || manifest.config != CampaignConfig::gen30()
            || manifest.ordered_dataset_ids != base.manifest.ordered_dataset_ids
            || manifest.datasets != base.manifest.datasets
        {
            bail!("existing TimeSeries42 gen30 manifest is invalid or bound to different inputs");
        }
        manifest
    } else {
        let ids = base.manifest.ordered_dataset_ids.clone();
        let manifest = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            base_run_id: Some(base_run_id.to_string()),
            base_campaign_sha256: Some(base.campaign_sha256.clone()),
            continuation_start_generation: Some(GEN30_START_GENERATION),
            continuation_terminal_generation: Some(GEN30_GENERATIONS - 1),
            registry_root: registry_str().to_string(),
            output_root: output_root_str().to_string(),
            release_binary_sha256: release_sha256.clone(),
            corpus_eval_sha256: corpus_eval_sha256.clone(),
            scq_binary: scq_binary_str().to_string(),
            scq_binary_sha256,
            coverage_binding: None,
            seed_populations: Vec::new(),
            ordered_dataset_ids: ids.clone(),
            ordered_cohort_sha256: ordered_cohort_sha256(&ids),
            config: CampaignConfig::gen30(),
            datasets: base.manifest.datasets.clone(),
            started_at_unix_ms: unix_millis(),
        };
        write_json(&manifest_path, &manifest)?;
        manifest
    };
    let campaign_sha256 = sha256_file(&manifest_path)?;
    let (staging, _) = stage_campaign(
        &root,
        &manifest.datasets,
        &manifest.release_binary_sha256,
        &manifest.corpus_eval_sha256,
    )?;
    preflight_representative_scq_checks(&root, &staging, run_id)?;
    Ok((base, manifest, staging, campaign_sha256))
}

fn initialize_winall(
    source_run_id: &str,
    run_id: &str,
    resume: bool,
    max_generations: usize,
    lineage_salt: &str,
) -> Result<(
    WinallSourceCampaign,
    CampaignManifest,
    StagingReceipt,
    String,
)> {
    validate_run_id(run_id)?;
    if source_run_id == run_id {
        bail!("TimeSeries42 win-all run id must differ from the source run id");
    }
    let source = load_winall_source_campaign(source_run_id)?;
    let start_generation = source.terminal_generation + 1;
    if max_generations <= start_generation {
        bail!("TimeSeries42 win-all max_generations must be greater than {start_generation}");
    }
    let winall_specs = winall_source_non_winners(&source)?;
    let root = campaign_root(run_id);
    fs::create_dir_all(&root)?;
    let release_sha256 = current_binary_sha256()?;
    let corpus_eval_sha256 = current_corpus_eval_sha256()?;
    let scq_binary_sha256 = sha256_file(Path::new(scq_binary_str()))?;
    let manifest_path = root.join("campaign.v1.json");
    let expected_config = CampaignConfig::winall(max_generations, start_generation, lineage_salt);
    let ids = ordered_ids(&winall_specs);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("TimeSeries42 win-all campaign exists; pass --resume to reuse it");
        }
        let bytes = fs::read(&manifest_path)?;
        let manifest: CampaignManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&manifest)?
            || manifest.schema != CAMPAIGN_SCHEMA
            || !manifest.test_adaptive
            || manifest.benchmark_valid
            || manifest.run_id != run_id
            || manifest.base_run_id.as_deref() != Some(source_run_id)
            || manifest.base_campaign_sha256.as_deref() != Some(&source.campaign_sha256)
            || manifest.continuation_start_generation != Some(start_generation)
            || manifest.continuation_terminal_generation != Some(max_generations - 1)
            || manifest.config != expected_config
            || manifest.ordered_dataset_ids != ids
            || manifest.datasets != winall_specs
        {
            bail!("existing TimeSeries42 win-all manifest is invalid or bound to different inputs");
        }
        manifest
    } else {
        let manifest = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            base_run_id: Some(source_run_id.to_string()),
            base_campaign_sha256: Some(source.campaign_sha256.clone()),
            continuation_start_generation: Some(start_generation),
            continuation_terminal_generation: Some(max_generations - 1),
            registry_root: registry_str().to_string(),
            output_root: output_root_str().to_string(),
            release_binary_sha256: release_sha256.clone(),
            corpus_eval_sha256: corpus_eval_sha256.clone(),
            scq_binary: scq_binary_str().to_string(),
            scq_binary_sha256,
            coverage_binding: None,
            seed_populations: Vec::new(),
            ordered_dataset_ids: ids.clone(),
            ordered_cohort_sha256: ordered_cohort_sha256(&ids),
            config: expected_config,
            datasets: winall_specs,
            started_at_unix_ms: unix_millis(),
        };
        write_json(&manifest_path, &manifest)?;
        manifest
    };
    let campaign_sha256 = sha256_file(&manifest_path)?;
    let (staging, _) = stage_campaign(
        &root,
        &manifest.datasets,
        &manifest.release_binary_sha256,
        &manifest.corpus_eval_sha256,
    )?;
    preflight_representative_scq_checks(&root, &staging, run_id)?;
    Ok((source, manifest, staging, campaign_sha256))
}
