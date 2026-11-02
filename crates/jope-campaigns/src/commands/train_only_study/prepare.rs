use super::*;

pub(crate) fn prepare(_registry: &Path, args: PrepareArgs) -> Result<PathBuf> {
    if args.workers == 0 {
        bail!("--workers must be at least one");
    }
    let generations = if args.smoke { 2 } else { GENERATIONS };
    let expected_datasets = if args.smoke { 1 } else { FROZEN_DATASETS.len() };
    let root = study_dir(&args.study_root, &args.run_id)?;
    fs::create_dir_all(root.join("inputs"))?;
    fs::create_dir_all(root.join("datasets"))?;
    let manifest_path = root.join("manifest.v2.json");
    if manifest_path.exists() {
        let (manifest, _): (TrainOnlyStudyManifest, _) = read_receipt(&manifest_path)?;
        if manifest.run_id != args.run_id {
            bail!("existing study manifest run id drift");
        }
        return Ok(manifest_path);
    }

    let mut checkpoint = resolve_checkpoint_provenance(&args)?;
    let pointer_snapshot_path = root.join("inputs/accepted-regression-pointer.snapshot.json");
    immutable_write(&pointer_snapshot_path, &fs::read(&args.checkpoint_pointer)?)?;
    checkpoint.accepted_pointer_path = absolute_path(&pointer_snapshot_path)?;
    let catalog_pointer_snapshot_path = root.join("inputs/context-catalog-pointer.snapshot.json");
    immutable_write(
        &catalog_pointer_snapshot_path,
        &fs::read(&args.catalog_pointer)?,
    )?;
    let (catalog_pointer, catalog) = load_catalog(&args.catalog_pointer)?;
    let exclusions =
        checkpoint_exclusions(&args.run_id, &args.sotarena_root, &checkpoint, &catalog)?;
    let exclusions_path = root.join("inputs/checkpoint-corpus-disclosure.v2.json");
    let exclusions_sha256 = write_receipt(&exclusions_path, &exclusions)?;

    let frozen_ids = if args.smoke {
        &FROZEN_DATASETS[..1]
    } else {
        FROZEN_DATASETS
    };
    let (mut eligibility, selected) = build_eligibility(
        &args.run_id,
        &args.sotarena_root,
        &catalog,
        &exclusions,
        generations,
        frozen_ids,
        Some(EXPECTED_ELIGIBLE),
    )?;
    if selected.len() != expected_datasets {
        let failure_path = root.join("preparation-failure.v2.json");
        let failure = json!({
            "schema": "TrainOnlyPreparationFailure.v2",
            "run_id": args.run_id,
            "seed": STUDY_SEED,
            "reason": "frozen_checkpoint_assisted_cohort_unavailable",
            "selected_count": selected.len(),
            "eligibility": eligibility,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "checkpoint_disclosure_sha256": exclusions_sha256,
            "no_fallback": true,
        });
        write_receipt(&failure_path, &failure)?;
        bail!(
            "only {} frozen Golden42 datasets pass the v2 study gates; preparation failed closed ({})",
            selected.len(),
            failure_path.display()
        );
    }

    let context_blob_path = root.join("inputs/contexts-1536+512.f32.bin");
    let (context_blob_sha256, encodings) =
        copy_context_slices(&catalog_pointer.data, &context_blob_path, &selected)?;
    let proposal_manifest_path = root.join("inputs/proposal-datasets.v1.json");
    let proposal_manifest = json!({
        "schema": "TrainOnlyProposalDatasetManifest.v1",
        "run_id": args.run_id,
        "catalog_sha256": catalog_pointer.catalog_sha256,
        "entries": selected.iter().map(|(entry, _, _, _)| json!({
            "task": "regression",
            "dataset_hash": entry.dataset_hash,
            "context_sha256": entry.context_sha256,
        })).collect::<Vec<_>>(),
    });
    let proposal_manifest_sha256 = write_receipt(&proposal_manifest_path, &proposal_manifest)?;
    if args.proposal_manifest_only {
        return Ok(proposal_manifest_path);
    }

    let proposals_path = if let Some(path) = args.dl_proposals {
        let output = root.join("inputs/dl-proposals.jsonl");
        immutable_write(&output, &fs::read(&path)?)?;
        output
    } else {
        let output = root.join("inputs/dl-proposals.jsonl");
        let status = Command::new(&args.python)
            .arg(&args.proposal_generator)
            .arg("--catalog-pointer")
            .arg(&args.catalog_pointer)
            .arg("--checkpoint")
            .arg(&checkpoint.checkpoint_path)
            .arg("--dataset-manifest")
            .arg(&proposal_manifest_path)
            .arg("--task")
            .arg("regression")
            .arg("--proposals-per-dataset")
            .arg("64")
            .arg("--seed")
            .arg(STUDY_SEED.to_string())
            .arg("--device")
            .arg(&args.proposal_device)
            .arg("--output")
            .arg(&output)
            .status()
            .context("launch exact-manifest DL proposal generator")?;
        if !status.success() {
            bail!("exact-manifest DL proposal generator failed");
        }
        output
    };
    let proposals_path = absolute_path(&proposals_path)?;
    let proposals_sha256 = sha256_file(&proposals_path)?;
    for (entry, data, _, _) in &selected {
        load_dl_proposals(
            &proposals_path,
            &entry.dataset_hash,
            &checkpoint.checkpoint_sha256,
            &entry.context_sha256,
            data.features,
        )?;
    }

    let observation_db_sha256 = sha256_file(&args.observation_db)?;
    let target_entries = selected
        .iter()
        .map(|(entry, _, _, _)| entry.clone())
        .collect::<Vec<_>>();
    let rankings = build_transfer_rankings(
        &args.run_id,
        &args.observation_db,
        &observation_db_sha256,
        &catalog,
        &catalog_pointer.catalog_sha256,
        &target_entries,
    )?;

    let mut datasets = Vec::with_capacity(expected_datasets);
    for (entry, data, identity, baseline_reports) in &selected {
        let dataset_root = root.join("datasets").join(&entry.dataset_hash);
        fs::create_dir_all(&dataset_root)?;
        let splits = build_splits(
            &args.run_id,
            &entry.dataset_hash,
            &entry.train_sha256,
            &data.y,
            generations,
        )?;
        let split_path = dataset_root.join("splits.v2.json");
        let split_sha256 = write_receipt(&split_path, &splits)?;
        let encoding = encodings
            .get(&entry.dataset_hash)
            .context("filtered encoding slice missing")?
            .clone();
        let ranking = rankings
            .get(&entry.dataset_hash)
            .context("target-specific transfer ranking missing")?;
        let ranking_path = dataset_root.join("transfer-ranking.v2.json");
        let ranking_sha256 = write_receipt(&ranking_path, ranking)?;
        let generation_zero = generation_zero(
            &args.run_id,
            entry,
            data,
            &encoding,
            &checkpoint,
            &proposals_path,
            &proposals_sha256,
            ranking,
            &ranking_sha256,
        )?;
        let generation_zero_path = dataset_root.join("generation-zero.v2.json");
        let generation_zero_sha256 = write_receipt(&generation_zero_path, &generation_zero)?;
        datasets.push(DatasetStudyEntry {
            dataset_hash: entry.dataset_hash.clone(),
            source_family: entry.source_family.clone(),
            target_identity: identity.clone(),
            train_path: absolute_path(&entry.train_csv)?,
            train_sha256: entry.train_sha256.clone(),
            rows: data.y.len(),
            features: data.features,
            encoding,
            split_manifest_path: absolute_path(&split_path)?,
            split_manifest_sha256: split_sha256,
            generation_zero_path: absolute_path(&generation_zero_path)?,
            generation_zero_sha256,
            transfer_ranking_path: absolute_path(&ranking_path)?,
            transfer_ranking_sha256: ranking_sha256,
            baseline_reports: baseline_reports.clone(),
        });
    }
    datasets.sort_by_key(|entry| {
        selected
            .iter()
            .position(|(selected, _, _, _)| selected.dataset_hash == entry.dataset_hash)
            .unwrap()
    });
    let selected_dataset_ids = datasets
        .iter()
        .map(|entry| entry.dataset_hash.clone())
        .collect::<Vec<_>>();
    for record in &mut eligibility {
        record.selected = selected_dataset_ids.contains(&record.dataset_hash);
    }
    let exact_golden42_dataset_ids = exclusions
        .dataset_ids
        .iter()
        .filter(|dataset| GOLDEN42.contains(&dataset.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let golden42_source_families = exclusions
        .source_families
        .iter()
        .filter(|family| {
            catalog.entries.iter().any(|entry| {
                GOLDEN42.contains(&entry.dataset_hash.as_str()) && entry.source_family == **family
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    let golden42_target_identities = eligibility
        .iter()
        .filter(|record| record.checkpoint_target_identity_overlap)
        .filter_map(|record| record.target_identity.clone())
        .collect::<Vec<_>>();
    let checkpoint_overlap_disclosure = CheckpointOverlapDisclosure {
        schema: "CheckpointOverlapDisclosure.v2".to_string(),
        checkpoint_sha256: checkpoint.checkpoint_sha256.clone(),
        selected_exact_dataset_ids: eligibility
            .iter()
            .filter(|record| record.selected && record.checkpoint_exact_overlap)
            .map(|record| record.dataset_hash.clone())
            .collect(),
        selected_source_families: eligibility
            .iter()
            .filter(|record| record.selected && record.checkpoint_source_family_overlap)
            .filter_map(|record| record.source_family.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        selected_target_identities: eligibility
            .iter()
            .filter(|record| record.selected && record.checkpoint_target_identity_overlap)
            .filter_map(|record| record.target_identity.clone())
            .collect(),
        checkpoint_was_trained_on_golden42_identities: !exact_golden42_dataset_ids.is_empty()
            || !golden42_source_families.is_empty()
            || !golden42_target_identities.is_empty(),
        exact_golden42_dataset_ids,
        golden42_source_families,
        golden42_target_identities,
        zero_shot_valid: false,
        disclosure: "The accepted checkpoint was trained with Golden42 identities. This is an exploratory checkpoint-assisted transfer study, not a zero-shot generalization benchmark.".to_string(),
    };
    let sotarena_registry_path = args.sotarena_root.join("registry/datasets.json");
    let manifest = TrainOnlyStudyManifest {
        schema: "TrainOnlyStudyManifest.v2".to_string(),
        study_design: STUDY_DESIGN.to_string(),
        zero_shot_valid: false,
        checkpoint_overlap_disclosure,
        campaign_scope: if args.smoke { "smoke" } else { "full" }.to_string(),
        run_id: args.run_id,
        seed: STUDY_SEED,
        generations,
        population: POPULATION,
        elites: ELITES,
        parent_pool: PARENT_POOL,
        tournament_size: TOURNAMENT,
        mutation_probability: 0.75,
        crossover_probability: 0.25,
        max_transforms: MAX_TRANSFORMS,
        workers: args.workers,
        kernel_version: KERNEL_VERSION.to_string(),
        sotarena_root: absolute_path(&args.sotarena_root)?,
        sotarena_registry_sha256: sha256_file(&sotarena_registry_path)?,
        sotarena_registry_path: absolute_path(&sotarena_registry_path)?,
        golden42_ids: GOLDEN42.iter().map(|value| (*value).to_string()).collect(),
        eligibility,
        selected_dataset_ids,
        checkpoint,
        corpus_exclusions_path: absolute_path(&exclusions_path)?,
        corpus_exclusions_sha256: exclusions_sha256,
        catalog_pointer_path: absolute_path(&catalog_pointer_snapshot_path)?,
        catalog_pointer_sha256: sha256_file(&catalog_pointer_snapshot_path)?,
        source_catalog_sha256: catalog_pointer.catalog_sha256,
        filtered_context_blob_sha256: context_blob_sha256,
        observation_db_path: absolute_path(&args.observation_db)?,
        observation_db_sha256,
        dl_proposals_path: proposals_path,
        dl_proposals_sha256: proposals_sha256,
        proposal_dataset_manifest_path: absolute_path(&proposal_manifest_path)?,
        proposal_dataset_manifest_sha256: proposal_manifest_sha256,
        datasets,
        firewall: PhaseOneFirewall {
            train_only: true,
            contains_test_paths: false,
            contains_test_hashes: false,
            contains_baseline_scores: false,
            test_paths_constructed: false,
        },
    };
    let serialized = String::from_utf8(canonical_json_bytes(&serde_json::to_value(&manifest)?)?)?;
    for forbidden in ["/test.csv", "\"test_sha256\"", "\"baseline_scores\""] {
        if serialized.to_ascii_lowercase().contains(forbidden) {
            bail!("sanitized phase-one manifest contains forbidden material: {forbidden}");
        }
    }
    write_receipt(&manifest_path, &manifest)?;
    Ok(manifest_path)
}

pub(super) fn validate_manifest_contract(
    manifest_path: &Path,
) -> Result<(TrainOnlyStudyManifest, String)> {
    let (manifest, digest): (TrainOnlyStudyManifest, String) = read_receipt(manifest_path)?;
    let (expected_generations, expected_ids): (usize, &[&str]) =
        if manifest.campaign_scope == "smoke" {
            (2, &FROZEN_DATASETS[..1])
        } else if manifest.campaign_scope == "full" {
            (GENERATIONS, FROZEN_DATASETS)
        } else {
            bail!("study manifest has an unsupported campaign scope");
        };
    if manifest.schema != "TrainOnlyStudyManifest.v2"
        || manifest.study_design != STUDY_DESIGN
        || manifest.zero_shot_valid
        || manifest.checkpoint_overlap_disclosure.zero_shot_valid
        || !manifest
            .checkpoint_overlap_disclosure
            .checkpoint_was_trained_on_golden42_identities
        || manifest.seed != STUDY_SEED
        || manifest.generations != expected_generations
        || manifest.population != POPULATION
        || manifest.elites != ELITES
        || manifest.parent_pool != PARENT_POOL
        || manifest.tournament_size != TOURNAMENT
        || manifest.mutation_probability != 0.75
        || manifest.crossover_probability != 0.25
        || manifest.max_transforms != MAX_TRANSFORMS
        || manifest.workers == 0
        || manifest.kernel_version != KERNEL_VERSION
        || manifest.datasets.len() != expected_ids.len()
        || manifest.selected_dataset_ids
            != expected_ids
                .iter()
                .map(|dataset| (*dataset).to_string())
                .collect::<Vec<_>>()
        || !manifest.firewall.train_only
        || manifest.firewall.contains_test_paths
        || manifest.firewall.contains_test_hashes
        || manifest.firewall.contains_baseline_scores
        || manifest.firewall.test_paths_constructed
    {
        bail!("study manifest violates the fixed train-only contract");
    }
    let unique = manifest.selected_dataset_ids.iter().collect::<HashSet<_>>();
    if unique.len() != expected_ids.len()
        || manifest
            .datasets
            .iter()
            .map(|entry| &entry.dataset_hash)
            .collect::<Vec<_>>()
            != manifest.selected_dataset_ids.iter().collect::<Vec<_>>()
    {
        bail!("study manifest dataset cohort is not ten ordered unique identities");
    }
    let bytes = fs::read(manifest_path)?;
    let lower = String::from_utf8_lossy(&bytes).to_ascii_lowercase();
    for forbidden in ["/test.csv", "\"test_sha256\"", "\"baseline_scores\""] {
        if lower.contains(forbidden) {
            bail!("phase-one manifest firewall violation: {forbidden}");
        }
    }
    Ok((manifest, digest))
}

pub(super) fn validate_prepared_inputs(manifest: &TrainOnlyStudyManifest) -> Result<()> {
    if sha256_file(&manifest.checkpoint.accepted_pointer_path)?
        != manifest.checkpoint.accepted_pointer_sha256
        || sha256_file(&manifest.checkpoint.checkpoint_path)?
            != manifest.checkpoint.checkpoint_sha256
        || sha256_file(&manifest.checkpoint.training_corpus_manifest_path)?
            != manifest.checkpoint.training_corpus_manifest_sha256
        || sha256_file(&manifest.checkpoint.training_corpus_train_path)?
            != manifest.checkpoint.training_corpus_train_sha256
        || sha256_file(&manifest.checkpoint.training_corpus_validation_path)?
            != manifest.checkpoint.training_corpus_validation_sha256
        || sha256_file(&manifest.catalog_pointer_path)? != manifest.catalog_pointer_sha256
        || sha256_file(&manifest.sotarena_registry_path)? != manifest.sotarena_registry_sha256
        || sha256_file(&manifest.dl_proposals_path)? != manifest.dl_proposals_sha256
    {
        bail!("prepared input provenance drift");
    }
    // The authoritative observation store is append-active. Preparation seals
    // its content hash into every immutable per-target ranking; validation
    // audits those receipts below instead of requiring the live SQLite path to
    // remain byte-identical after the study was frozen.
    validate_sha256(
        &manifest.observation_db_sha256,
        "frozen observation database",
    )?;
    let (_, exclusion_digest): (CorpusExclusions, _) =
        read_receipt(&manifest.corpus_exclusions_path)?;
    let (_, proposal_manifest_digest): (Value, _) =
        read_receipt(&manifest.proposal_dataset_manifest_path)?;
    if exclusion_digest != manifest.corpus_exclusions_sha256
        || proposal_manifest_digest != manifest.proposal_dataset_manifest_sha256
    {
        bail!("prepared disclosure or proposal-manifest receipt drift");
    }
    for entry in &manifest.datasets {
        if sha256_file(&entry.train_path)? != entry.train_sha256
            || sha256_file(&entry.encoding.blob_path)? != entry.encoding.blob_sha256
            || entry.encoding.blob_sha256 != manifest.filtered_context_blob_sha256
        {
            bail!("prepared dataset input drift for {}", entry.dataset_hash);
        }
        let (_, split_digest): (TrainOnlySplitManifest, _) =
            read_receipt(&entry.split_manifest_path)?;
        let (_, generation_zero_digest): (TrainOnlyGenerationZero, _) =
            read_receipt(&entry.generation_zero_path)?;
        let (ranking, ranking_digest): (GlobalRankingReceipt, _) =
            read_receipt(&entry.transfer_ranking_path)?;
        if split_digest != entry.split_manifest_sha256
            || generation_zero_digest != entry.generation_zero_sha256
            || ranking_digest != entry.transfer_ranking_sha256
            || ranking.target_dataset_id != entry.dataset_hash
            || ranking.target_source_family != entry.source_family
            || sha256_file(&entry.baseline_reports.tabpfn_v3_path)?
                != entry.baseline_reports.tabpfn_v3_sha256
            || sha256_file(&entry.baseline_reports.tabfm_path)?
                != entry.baseline_reports.tabfm_sha256
            || sha256_file(&entry.baseline_reports.tabfm_ensemble_path)?
                != entry.baseline_reports.tabfm_ensemble_sha256
        {
            bail!("prepared dataset receipt drift for {}", entry.dataset_hash);
        }
    }
    Ok(())
}

pub(super) fn validate_execution_inputs(
    manifest: &TrainOnlyStudyManifest,
    dataset: &DatasetStudyEntry,
) -> Result<()> {
    if sha256_file(&dataset.train_path)? != dataset.train_sha256
        || sha256_file(&dataset.encoding.blob_path)? != dataset.encoding.blob_sha256
        || sha256_file(&manifest.dl_proposals_path)? != manifest.dl_proposals_sha256
    {
        bail!("sharded execution input drift for {}", dataset.dataset_hash);
    }
    let (_, split_digest): (TrainOnlySplitManifest, _) =
        read_receipt(&dataset.split_manifest_path)?;
    let (_, generation_zero_digest): (TrainOnlyGenerationZero, _) =
        read_receipt(&dataset.generation_zero_path)?;
    let (_, transfer_digest): (GlobalRankingReceipt, _) =
        read_receipt(&dataset.transfer_ranking_path)?;
    if split_digest != dataset.split_manifest_sha256
        || generation_zero_digest != dataset.generation_zero_sha256
        || transfer_digest != dataset.transfer_ranking_sha256
    {
        bail!(
            "sharded execution receipt drift for {}",
            dataset.dataset_hash
        );
    }
    Ok(())
}

pub(super) fn cache_key(
    train_sha256: &str,
    semantic_sha256: &str,
    split_sha256: &str,
) -> Result<String> {
    receipt_digest(&json!({
        "train_sha256": train_sha256,
        "semantic_sha256": semantic_sha256,
        "split_sha256": split_sha256,
        "kernel_version": KERNEL_VERSION,
    }))
}
