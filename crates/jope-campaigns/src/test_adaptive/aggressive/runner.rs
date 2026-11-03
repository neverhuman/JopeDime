use super::*;
use super::{checkpoint::load_checkpoints, dataset::run_dataset, model::DatasetResult};

pub(crate) fn run_golden46(
    source_registry: &Path,
    run_id: &str,
    dataset_selector: Option<&str>,
    transfer_arm: bool,
    resume: bool,
) -> Result<()> {
    if run_id != GOLDEN46_RUN_PREFIX {
        bail!("run-golden46 is restricted to run id {GOLDEN46_RUN_PREFIX}");
    }
    if transfer_arm {
        bail!("full ZTS and its transfer arm are excluded from the required study");
    }
    let effective_run_id = run_id.to_string();
    let selector = dataset_selector;
    let (registry, prepared_view_manifest_sha256) =
        prepare_golden46_evolution_views(source_registry)?;
    let (all_specs, panel_sha256, leader_sha256, view_manifest_sha256) =
        load_golden46_specs(&registry)?;
    if prepared_view_manifest_sha256 != view_manifest_sha256 {
        bail!("Golden46 evolution view changed between preparation and loading");
    }
    let binding = Golden46Bindings {
        panel_sha256,
        leader_sha256,
        evolution_view_manifest_sha256: view_manifest_sha256,
        arm: "corrected-test-adaptive-continuation".to_string(),
        leaderboard_default: true,
    };
    run_aggressive_with_specs(
        &registry,
        &effective_run_id,
        selector,
        0,
        resume,
        all_specs,
        Some(binding),
    )
}

pub(crate) fn validate_golden46(
    source_registry: &Path,
    run_id: &str,
    transfer_arm: bool,
) -> Result<()> {
    if run_id != GOLDEN46_RUN_PREFIX {
        bail!("validate-golden46 is restricted to run id {GOLDEN46_RUN_PREFIX}");
    }
    if transfer_arm {
        bail!("full ZTS and its transfer arm are excluded from the required study");
    }
    let effective_run_id = run_id.to_string();
    let (registry, prepared_view_manifest_sha256) =
        prepare_golden46_evolution_views(source_registry)?;
    let (all_specs, panel_sha256, leader_sha256, view_manifest_sha256) =
        load_golden46_specs(&registry)?;
    if prepared_view_manifest_sha256 != view_manifest_sha256 {
        bail!("Golden46 evolution view changed between preparation and validation");
    }
    let binding = Golden46Bindings {
        panel_sha256,
        leader_sha256,
        evolution_view_manifest_sha256: view_manifest_sha256,
        arm: "corrected-test-adaptive-continuation".to_string(),
        leaderboard_default: true,
    };
    validate_aggressive_with_specs(&registry, &effective_run_id, all_specs, Some(&binding))?;
    Ok(())
}

pub(crate) fn run_aggressive(
    registry: &Path,
    run_id: &str,
    dataset_selector: Option<&str>,
    start_index: usize,
    resume: bool,
) -> Result<()> {
    if is_golden46_run(run_id) {
        bail!("Golden46 runs require the dedicated run-golden46 command");
    }
    let (_, all_specs, _, _) = load_panel_specs(registry)?;
    run_aggressive_with_specs(
        registry,
        run_id,
        dataset_selector,
        start_index,
        resume,
        all_specs,
        None,
    )
}

pub(crate) fn run_aggressive_with_specs(
    registry: &Path,
    run_id: &str,
    dataset_selector: Option<&str>,
    start_index: usize,
    resume: bool,
    all_specs: Vec<DatasetSpec>,
    golden46: Option<Golden46Bindings>,
) -> Result<()> {
    validate_run_id(run_id)?;
    let campaign_root = root(run_id);
    let manifest_path = campaign_root.join(MANIFEST_FILE);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("aggressive run already exists; pass --resume");
        }
        let bytes = fs::read(&manifest_path)?;
        let manifest: AggressiveManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&manifest)?
            || manifest.schema != SCHEMA
            || manifest.benchmark_valid
            || !manifest.test_adaptive
            || manifest.run_id != run_id
            || manifest.selected_backend != "rust"
            || manifest.cpp17_admitted
            || manifest.generation_limit.is_some()
            || manifest.population != TOTAL_POPULATION
            || manifest.islands != ISLANDS
            || manifest.population_per_island != ISLAND_POPULATION
            || manifest.workers_per_node != scq_batch::AGGRESSIVE_WORKERS
            || manifest.registry_root != registry.to_string_lossy()
            || !manifest_matches_golden46_bindings(&manifest, golden46.as_ref())
            || manifest.datasets.is_empty()
            || manifest
                .datasets
                .iter()
                .any(|spec| !all_specs.contains(spec))
            || manifest.datasets != select_dataset_specs(&all_specs, dataset_selector, start_index)?
            || manifest.release_binary_sha256 != current_binary_sha256()?
            || manifest.scq_binary_sha256 != scq_batch::scq_binary_sha256()?
        {
            bail!("aggressive manifest or selected backend changed");
        }
        let preflight_bytes = fs::read(campaign_root.join(PREFLIGHT_FILE))?;
        if sha256_bytes(&preflight_bytes) != manifest.preflight_sha256 {
            bail!("aggressive preflight receipt changed");
        }
        let recorded_preflight: PreflightReceipt = serde_json::from_slice(&preflight_bytes)?;
        if preflight_bytes != canonical_json(&recorded_preflight)? {
            bail!("aggressive preflight receipt is not canonical");
        }
        validate_preflight_receipt(&recorded_preflight)?;
        let current_preflight = preflight()?;
        validate_preflight_receipt(&current_preflight)?;
        let staging_bytes = fs::read(campaign_root.join(STAGING_FILE))?;
        if sha256_bytes(&staging_bytes) != manifest.staging_sha256 {
            bail!("aggressive staging receipt changed");
        }
        let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
        if staging_bytes != canonical_json(&staging)?
            || staging.scq_binary_sha256 != manifest.scq_binary_sha256
        {
            bail!("aggressive staging receipt identity changed");
        }
        scq_batch::validate_staging(
            &staging,
            &manifest.release_binary_sha256,
            &manifest.datasets,
        )?;
        scq_batch::verify_remote_staging(&staging)?;
        let admission_bytes = fs::read(campaign_root.join(ADMISSION_FILE))?;
        if sha256_bytes(&admission_bytes) != manifest.backend_admission_sha256 {
            bail!("fitted backend admission receipt changed");
        }
        manifest
    } else {
        if campaign_root.exists() && !resume && golden46.is_none() {
            bail!("aggressive run directory already exists");
        }
        let admission = crate::aggressive::fitted_backend_admission()?;
        if admission["selected_backend"].as_str() != Some("rust")
            || admission["cpp17"]["admitted"].as_bool() != Some(false)
        {
            bail!("fitted backend admission did not fail closed to Rust");
        }
        let admission_bytes = canonical_json(&admission)?;
        let specs = select_dataset_specs(&all_specs, dataset_selector, start_index)?;
        let preflight = preflight()?;
        let preflight_bytes = canonical_json(&preflight)?;
        let release_binary = std::env::current_exe()?;
        let release_binary_sha256 = sha256_file(&release_binary)?;
        let staging = scq_batch::preflight_and_stage(
            registry,
            &release_binary,
            &release_binary_sha256,
            &specs,
        )?;
        let staging_bytes = canonical_json(&staging)?;
        fs::create_dir_all(&campaign_root)?;
        atomic_immutable_write(&campaign_root.join(ADMISSION_FILE), &admission_bytes)?;
        atomic_immutable_write(&campaign_root.join(PREFLIGHT_FILE), &preflight_bytes)?;
        atomic_immutable_write(&campaign_root.join(STAGING_FILE), &staging_bytes)?;
        let manifest = expected_manifest(
            registry,
            run_id,
            &release_binary_sha256,
            &staging.scq_binary_sha256,
            &sha256_bytes(&preflight_bytes),
            &sha256_bytes(&staging_bytes),
            &sha256_bytes(&admission_bytes),
            specs.clone(),
            golden46.as_ref(),
        );
        atomic_immutable_write(&manifest_path, &canonical_json(&manifest)?)?;
        manifest
    };
    if campaign_root.join(SUMMARY_FILE).is_file() {
        return validate_aggressive_with_specs(registry, run_id, all_specs, golden46.as_ref());
    }
    let results = if golden46.is_some() {
        run_dataset_waves(&manifest)?
    } else if manifest.datasets.len() == 1 {
        vec![run_dataset(&manifest, &manifest.datasets[0])?]
    } else {
        run_dataset_waves(&manifest)?
    };
    let confirmed_wins = results
        .iter()
        .filter(|result| result.outcome == "confirmed_strict_win")
        .count();
    let summary = Summary {
        schema: SUMMARY_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: run_id.to_string(),
        selected_backend: "rust".to_string(),
        dataset_count: results.len(),
        confirmed_wins,
        unresolved: results.len() - confirmed_wins,
        results,
    };
    atomic_immutable_write(
        &campaign_root.join(SUMMARY_FILE),
        &canonical_json(&summary)?,
    )?;
    validate_aggressive_with_specs(registry, run_id, all_specs, golden46.as_ref())
}

pub(crate) fn validate_aggressive(registry: &Path, run_id: &str) -> Result<()> {
    if is_golden46_run(run_id) {
        bail!("Golden46 runs require the dedicated validate-golden46 command");
    }
    let (_, all_specs, _, _) = load_panel_specs(registry)?;
    validate_aggressive_with_specs(registry, run_id, all_specs, None)
}

pub(crate) fn validate_aggressive_with_specs(
    registry: &Path,
    run_id: &str,
    all_specs: Vec<DatasetSpec>,
    golden46: Option<&Golden46Bindings>,
) -> Result<()> {
    validate_run_id(run_id)?;
    let campaign_root = root(run_id);
    let manifest_bytes = fs::read(campaign_root.join(MANIFEST_FILE))?;
    let manifest: AggressiveManifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest_bytes != canonical_json(&manifest)?
        || manifest.schema != SCHEMA
        || manifest.benchmark_valid
        || !manifest.test_adaptive
        || manifest.run_id != run_id
        || manifest.selected_backend != "rust"
        || manifest.cpp17_admitted
        || manifest.generation_limit.is_some()
        || manifest.registry_root != registry.to_string_lossy()
        || !manifest_matches_golden46_bindings(&manifest, golden46)
        || manifest.datasets.is_empty()
        || manifest
            .datasets
            .iter()
            .any(|spec| !all_specs.contains(spec))
        || manifest.release_binary_sha256 != current_binary_sha256()?
        || manifest.scq_binary_sha256 != scq_batch::scq_binary_sha256()?
    {
        bail!("invalid aggressive manifest");
    }
    let specs = &manifest.datasets;
    let admission_bytes = fs::read(campaign_root.join(ADMISSION_FILE))?;
    if sha256_bytes(&admission_bytes) != manifest.backend_admission_sha256 {
        bail!("aggressive backend admission receipt changed");
    }
    let preflight_bytes = fs::read(campaign_root.join(PREFLIGHT_FILE))?;
    if sha256_bytes(&preflight_bytes) != manifest.preflight_sha256 {
        bail!("aggressive preflight receipt changed");
    }
    let preflight_receipt: PreflightReceipt = serde_json::from_slice(&preflight_bytes)?;
    if preflight_bytes != canonical_json(&preflight_receipt)? {
        bail!("aggressive preflight receipt is not canonical");
    }
    validate_preflight_receipt(&preflight_receipt)?;
    let staging_bytes = fs::read(campaign_root.join(STAGING_FILE))?;
    if sha256_bytes(&staging_bytes) != manifest.staging_sha256 {
        bail!("aggressive staging receipt changed");
    }
    let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    if staging_bytes != canonical_json(&staging)?
        || staging.scq_binary_sha256 != manifest.scq_binary_sha256
    {
        bail!("aggressive staging receipt identity changed");
    }
    scq_batch::validate_staging(&staging, &manifest.release_binary_sha256, specs)?;
    scq_batch::verify_remote_staging(&staging)?;
    let mut results = Vec::with_capacity(specs.len());
    for spec in specs {
        let (_, test) = read_campaign_data(registry, spec)?;
        let checkpoints = load_checkpoints(run_id, spec)?;
        for (checkpoint, _) in &checkpoints {
            validate_checkpoint_receipts(&manifest, spec, checkpoint, &test)?;
        }
        let (terminal, digest) = checkpoints
            .last()
            .context("aggressive dataset has no complete generation barrier")?;
        if terminal.winner_slot.is_none() {
            bail!("aggressive dataset result exists without a strict winner");
        }
        let result_bytes = fs::read(result_path(run_id, &spec.dataset_hash))?;
        let result: DatasetResult = serde_json::from_slice(&result_bytes)?;
        let winner_slot = terminal
            .winner_slot
            .context("aggressive dataset result has no winner")?;
        let winner = &terminal.slots[winner_slot];
        if result_bytes != canonical_json(&result)?
            || result.schema != RESULT_SCHEMA
            || result.run_id != run_id
            || result.dataset_hash != spec.dataset_hash
            || result.terminal_checkpoint_sha256 != *digest
            || result.benchmark_valid
            || !result.test_adaptive
            || result.outcome != "confirmed_strict_win"
            || result.stopping_generation != terminal.generation
            || result.leader_rmse.to_bits() != spec.leader_rmse.to_bits()
            || result.best_rmse.to_bits() != winner.rmse().to_bits()
            || result.best_slot != winner_slot
            || result.canonical_genome != winner.member.canonical_genome
            || result.semantic_sha256 != winner.member.semantic_sha256
            || result.confirmation_sha256.is_none()
        {
            bail!("aggressive dataset result does not bind to its barrier");
        }
        if let Some(slot) = terminal.winner_slot {
            let winner = &terminal.slots[slot];
            if !winner.rmse().is_finite() || winner.rmse() >= spec.leader_rmse {
                bail!("aggressive winner is not a strict finite baseline crossing");
            }
            let confirmation_bytes = fs::read(confirmation_path(run_id, &spec.dataset_hash))?;
            let confirmation: Confirmation = serde_json::from_slice(&confirmation_bytes)?;
            if confirmation_bytes != canonical_json(&confirmation)?
                || confirmation.schema != CONFIRMATION_SCHEMA
                || confirmation.benchmark_valid
                || !confirmation.test_adaptive
                || confirmation.run_id != run_id
                || confirmation.dataset_hash != spec.dataset_hash
                || confirmation.generation != terminal.generation
                || confirmation.slot != slot
                || confirmation.checkpoint_sha256 != *digest
                || confirmation.rmse_bits != winner.rmse().to_bits()
                || result.confirmation_sha256.as_deref()
                    != Some(sha256_bytes(&confirmation_bytes).as_str())
                || confirmation.nodes
                    != scq_batch::NODES
                        .iter()
                        .map(|node| (*node).to_string())
                        .collect::<Vec<_>>()
            {
                bail!("aggressive three-node confirmation is invalid");
            }
            if confirmation.receipt_files.len() != scq_batch::NODES.len()
                || confirmation.receipt_sha256.len() != scq_batch::NODES.len()
            {
                bail!("aggressive confirmation receipt set is incomplete");
            }
            for index in 0..scq_batch::NODES.len() {
                let receipt_bytes = fs::read(
                    dataset_root(run_id, &spec.dataset_hash)
                        .join(&confirmation.receipt_files[index]),
                )?;
                if sha256_bytes(&receipt_bytes) != confirmation.receipt_sha256[index] {
                    bail!("aggressive confirmation receipt hash changed");
                }
                let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
                if receipt_bytes != canonical_json(&receipt)? {
                    bail!("aggressive confirmation receipt is not canonical");
                }
                let request_path = dataset_root(run_id, &spec.dataset_hash)
                    .join("requests")
                    .join(&receipt.purpose)
                    .join(format!("generation-{:03}", receipt.generation))
                    .join(format!(
                        "slot-{:03}-attempt-{}.v1.json",
                        receipt.slot, receipt.attempt
                    ));
                let request_bytes = fs::read(request_path)?;
                let request: CandidateRequest = serde_json::from_slice(&request_bytes)?;
                if request_bytes != canonical_json(&request)? {
                    bail!("aggressive confirmation request is not canonical");
                }
                validate_receipt_against_request(&receipt, &request, &test)?;
                let metrics = receipt
                    .metrics
                    .as_ref()
                    .context("aggressive confirmation receipt failed")?;
                let prediction_bits = receipt
                    .predictions
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect::<Vec<_>>();
                if receipt.test_adaptive != Some(true)
                    || receipt.purpose != "aggressive-confirmation"
                    || receipt.run_id != run_id
                    || receipt.dataset_hash != spec.dataset_hash
                    || receipt.train_sha256 != spec.train_sha256
                    || receipt.test_sha256 != spec.test_sha256
                    || receipt.binary_sha256 != manifest.release_binary_sha256
                    || receipt.generation != terminal.generation
                    || receipt.slot != index
                    || receipt.attempt != 0
                    || receipt.semantic_sha256 != winner.member.semantic_sha256
                    || receipt.canonical_genome != winner.member.canonical_genome
                    || receipt.node != scq_batch::NODES[index]
                    || metrics.rmse.to_bits() != confirmation.rmse_bits
                    || sha256_bytes(&prediction_bits) != confirmation.prediction_bits_sha256
                {
                    bail!("aggressive confirmation prediction or RMSE bits changed");
                }
            }
        }
        results.push(result);
    }
    let summary_bytes = fs::read(campaign_root.join(SUMMARY_FILE))?;
    let summary: Summary = serde_json::from_slice(&summary_bytes)?;
    if summary_bytes != canonical_json(&summary)?
        || summary.schema != SUMMARY_SCHEMA
        || summary.benchmark_valid
        || !summary.test_adaptive
        || summary.run_id != run_id
        || summary.selected_backend != "rust"
        || summary.results != results
        || summary.dataset_count != specs.len()
        || summary.confirmed_wins + summary.unresolved != summary.dataset_count
    {
        bail!("invalid aggressive summary");
    }
    eprintln!(
        "validated aggressive run {run_id}: {} confirmed, {} unresolved",
        summary.confirmed_wins, summary.unresolved
    );
    Ok(())
}
