use super::*;

pub(crate) fn validate(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be at least one");
    }
    let root = study_dir(study_root, run_id)?;
    let manifest_path = root.join("manifest.v2.json");
    let (manifest, manifest_digest) = validate_manifest_contract(&manifest_path)?;
    if absolute_path(sotarena_root)? != manifest.sotarena_root {
        bail!("--sotarena-root does not match the frozen study manifest");
    }
    if let Some(wanted) = selected_dataset {
        let dataset = manifest
            .datasets
            .iter()
            .find(|dataset| dataset.dataset_hash == wanted)
            .context("requested validation dataset is not in the frozen study manifest")?;
        validate_execution_inputs(&manifest, dataset)?;
    } else {
        validate_prepared_inputs(&manifest)?;
    }
    let (exclusions, exclusion_digest): (CorpusExclusions, _) =
        read_receipt(&manifest.corpus_exclusions_path)?;
    if exclusion_digest != manifest.corpus_exclusions_sha256
        || manifest.checkpoint_overlap_disclosure.checkpoint_sha256
            != manifest.checkpoint.checkpoint_sha256
        || manifest.checkpoint_overlap_disclosure.zero_shot_valid
        || !manifest
            .checkpoint_overlap_disclosure
            .checkpoint_was_trained_on_golden42_identities
    {
        bail!("checkpoint-overlap disclosure audit drift");
    }
    let overlapping_ids = exclusions.dataset_ids.iter().collect::<HashSet<_>>();
    let overlapping_families = exclusions.source_families.iter().collect::<HashSet<_>>();
    let overlapping_targets = exclusions.target_identities.iter().collect::<HashSet<_>>();
    for dataset in manifest
        .datasets
        .iter()
        .filter(|dataset| selected_dataset.is_none_or(|wanted| dataset.dataset_hash == wanted))
    {
        let record = manifest
            .eligibility
            .iter()
            .find(|record| record.dataset_hash == dataset.dataset_hash)
            .context("selected dataset lacks eligibility record")?;
        if !record.eligible || !record.selected || !record.reasons.is_empty() {
            bail!("selected dataset eligibility audit failed");
        }
        if record.checkpoint_exact_overlap != overlapping_ids.contains(&dataset.dataset_hash)
            || record.checkpoint_source_family_overlap
                != overlapping_families.contains(&dataset.source_family)
            || record.checkpoint_target_identity_overlap
                != overlapping_targets.contains(&dataset.target_identity)
        {
            bail!("selected dataset checkpoint-overlap disclosure drift");
        }
        let (ranking, ranking_digest): (GlobalRankingReceipt, _) =
            read_receipt(&dataset.transfer_ranking_path)?;
        if ranking_digest != dataset.transfer_ranking_sha256
            || ranking.schema != "TrainOnlyTransferGenomeRanking.v2"
            || ranking.run_id != manifest.run_id
            || ranking.target_dataset_id != dataset.dataset_hash
            || ranking.target_source_family != dataset.source_family
            || ranking.observation_db_sha256 != manifest.observation_db_sha256
            || ranking.source_catalog_sha256 != manifest.source_catalog_sha256
            || ranking.excluded_source_family != dataset.source_family
            || !ranking.excluded_dataset_ids.contains(&dataset.dataset_hash)
            || !ranking.direct_dl_sources_excluded
        {
            bail!("target transfer-ranking identity drift");
        }
        for (rank, genome) in ranking.genomes.iter().enumerate() {
            if genome.rank != rank
                || genome
                    .source_dataset_ids
                    .iter()
                    .any(|source| ranking.excluded_dataset_ids.contains(source))
                || genome.source_families.contains(&dataset.source_family)
            {
                bail!("target/source-family transfer exclusion drift");
            }
        }
        let mut blob = File::open(&dataset.encoding.blob_path)?;
        blob.seek(SeekFrom::Start(dataset.encoding.offset_bytes as u64))?;
        let mut bytes = vec![0_u8; dataset.encoding.length_bytes];
        blob.read_exact(&mut bytes)?;
        if sha256_bytes(&bytes) != dataset.encoding.context_sha256 {
            bail!("filtered context slice audit failed");
        }
    }
    let (cohort, cohort_digest): (TrainOnlyCohortSelectionComplete, _) =
        read_receipt(&root.join("selection-complete.v2.json"))?;
    if cohort.schema != "TrainOnlyCohortSelectionComplete.v2"
        || cohort.run_id != manifest.run_id
        || cohort.manifest_sha256 != manifest_digest
        || cohort.dataset_ids != manifest.selected_dataset_ids
        || cohort.selection_complete_sha256.len() != manifest.datasets.len()
        || cohort.champion_validation_sha256.len() != manifest.datasets.len()
    {
        bail!("cohort selection-complete receipt drift");
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("build study validation worker pool")?;
    if let Some(wanted) = selected_dataset {
        let dataset = manifest
            .datasets
            .iter()
            .find(|dataset| dataset.dataset_hash == wanted)
            .context("requested validation dataset is not in the frozen study manifest")?;
        validate_dataset_shard(
            &root,
            &manifest,
            &manifest_digest,
            &cohort,
            &cohort_digest,
            dataset,
            &pool,
        )?;
        return Ok(dataset_validation_path(&root, wanted));
    }
    let mut selections = BTreeMap::new();
    let mut dataset_validations = BTreeMap::new();
    for dataset in &manifest.datasets {
        let path = dataset_validation_path(&root, &dataset.dataset_hash);
        let (selection_digest, validation_digest) = if path.exists() {
            read_dataset_validation_shard(
                &root,
                &manifest,
                &manifest_digest,
                &cohort,
                &cohort_digest,
                dataset,
            )?
        } else {
            validate_dataset_shard(
                &root,
                &manifest,
                &manifest_digest,
                &cohort,
                &cohort_digest,
                dataset,
                &pool,
            )?
        };
        selections.insert(dataset.dataset_hash.clone(), selection_digest);
        dataset_validations.insert(dataset.dataset_hash.clone(), validation_digest);
    }
    let expected_cohort_cumulative = receipt_digest(&json!({
        "manifest_sha256": manifest_digest,
        "selection_complete_sha256": cohort.selection_complete_sha256,
        "champion_validation_sha256": cohort.champion_validation_sha256,
    }))?;
    if cohort.cumulative_sha256 != expected_cohort_cumulative {
        bail!("cohort cumulative digest drift");
    }
    let receipt = TrainOnlyValidationReceipt {
        schema: "TrainOnlyValidationReceipt.v2".to_string(),
        study_design: STUDY_DESIGN.to_string(),
        zero_shot_valid: false,
        checkpoint_overlap_disclosure: manifest.checkpoint_overlap_disclosure.clone(),
        run_id: manifest.run_id.clone(),
        manifest_sha256: manifest_digest,
        cohort_selection_complete_sha256: cohort_digest,
        dataset_selection_sha256: selections,
        dataset_validation_sha256: dataset_validations,
        benchmark_valid: true,
        checkpoint_disclosure_audit_passed: true,
        transfer_exclusion_audit_passed: true,
        firewall_audit_passed: true,
        population_audit_passed: true,
        paired_history_audit_passed: true,
        cache_audit_passed: true,
        validated_dataset_count: manifest.datasets.len(),
    };
    let path = root.join("validation-success.v2.json");
    write_receipt(&path, &receipt)?;
    Ok(path)
}

pub(super) fn require_validated_score_gate(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<(
    PathBuf,
    TrainOnlyStudyManifest,
    String,
    TrainOnlyValidationReceipt,
    String,
)> {
    if workers == 0 {
        bail!("--workers must be at least one");
    }
    let root = study_dir(study_root, run_id)?;
    let validation_path = root.join("validation-success.v2.json");
    if !validation_path.exists() {
        bail!(
            "post-hoc scoring requires a successful independent validation receipt: {}",
            validation_path.display()
        );
    }

    let manifest_path = root.join("manifest.v2.json");
    let (manifest, manifest_digest) = validate_manifest_contract(&manifest_path)?;
    if absolute_path(sotarena_root)? != manifest.sotarena_root {
        bail!("--sotarena-root does not match the frozen study manifest");
    }
    validate_prepared_inputs(&manifest)?;
    let (validation, validation_digest): (TrainOnlyValidationReceipt, _) =
        read_receipt(&validation_path)?;
    let (cohort, cohort_digest): (TrainOnlyCohortSelectionComplete, _) =
        read_receipt(&root.join("selection-complete.v2.json"))?;
    if validation.schema != "TrainOnlyValidationReceipt.v2"
        || validation.study_design != STUDY_DESIGN
        || validation.zero_shot_valid
        || validation.run_id != run_id
        || validation.manifest_sha256 != manifest_digest
        || validation.cohort_selection_complete_sha256 != cohort_digest
        || !validation.benchmark_valid
        || !validation.checkpoint_disclosure_audit_passed
        || !validation.transfer_exclusion_audit_passed
        || !validation.firewall_audit_passed
        || !validation.population_audit_passed
        || !validation.paired_history_audit_passed
        || !validation.cache_audit_passed
        || validation.validated_dataset_count != manifest.datasets.len()
        || validation.dataset_selection_sha256.len() != manifest.datasets.len()
        || validation.dataset_validation_sha256.len() != manifest.datasets.len()
        || cohort.manifest_sha256 != manifest_digest
        || cohort.dataset_ids != manifest.selected_dataset_ids
    {
        bail!("validation receipt does not authorize post-hoc scoring");
    }
    for dataset in &manifest.datasets {
        let (shard_selection_digest, shard_digest) = read_dataset_validation_shard(
            &root,
            &manifest,
            &manifest_digest,
            &cohort,
            &cohort_digest,
            dataset,
        )?;
        let (selection, selection_digest): (TrainOnlySelectionComplete, _) =
            read_receipt(&selection_path(&root, &dataset.dataset_hash))?;
        if validation
            .dataset_selection_sha256
            .get(&dataset.dataset_hash)
            != Some(&selection_digest)
            || cohort.selection_complete_sha256.get(&dataset.dataset_hash)
                != Some(&selection_digest)
            || shard_selection_digest != selection_digest
            || validation
                .dataset_validation_sha256
                .get(&dataset.dataset_hash)
                != Some(&shard_digest)
            || selection.generation_checkpoint_sha256.len() != manifest.generations
        {
            bail!(
                "validated selection digest drift for {}",
                dataset.dataset_hash
            );
        }
        let (_, fixed_digest): (TrainOnlyChampionValidation, _) =
            read_receipt(&champion_validation_path(&root, &dataset.dataset_hash))?;
        if cohort.champion_validation_sha256.get(&dataset.dataset_hash) != Some(&fixed_digest) {
            bail!(
                "validated champion-grid digest drift for {}",
                dataset.dataset_hash
            );
        }
        for (generation, expected_digest) in
            selection.generation_checkpoint_sha256.iter().enumerate()
        {
            let (_, digest): (TrainOnlyGenerationCheckpoint, _) =
                read_receipt(&checkpoint_path(&root, &dataset.dataset_hash, generation))?;
            if &digest != expected_digest {
                bail!(
                    "validated generation checkpoint digest drift for {} generation {}",
                    dataset.dataset_hash,
                    generation
                );
            }
        }
    }
    Ok((
        root,
        manifest,
        manifest_digest,
        validation,
        validation_digest,
    ))
}
