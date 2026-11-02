use super::*;

pub(super) fn run_impl(run_id: &str, dataset: &str, resume: bool) -> Result<()> {
    let (root, campaign, staging) = initialize_campaign(run_id, dataset, resume)?;
    if root.join("RUN_COMPLETE").exists() {
        return validate(run_id, dataset);
    }
    run_campaign_loop(&root, &campaign, &staging)?;
    validate(run_id, dataset)
}

pub(super) fn run_campaign_loop(
    root: &Path,
    campaign: &CampaignManifest,
    staging: &StagingReceipt,
) -> Result<()> {
    if root.join("RUN_COMPLETE").exists() || root.join("RUN_EXHAUSTED").exists() {
        return Ok(());
    }
    let train = read_binary_csv(Path::new(&campaign.dataset_binding.train_path))?;
    let test = read_binary_csv(Path::new(&campaign.dataset_binding.test_path))?;
    if train.x.len() != campaign.dataset_binding.train_rows
        || test.x.len() != campaign.dataset_binding.test_rows
    {
        bail!("Binary42 campaign data changed after initialization");
    }
    let seeds: SeedManifest = load_json(&root.join("seeds").join("generation-zero.v1.json"))?;
    let seed_semantics = seeds
        .members
        .iter()
        .map(|member| member.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    if seed_semantics.len() != POPULATION {
        bail!("Binary42 seed manifest does not contain 60 unique semantics");
    }
    loop {
        let mut checkpoints = load_checkpoints(root)?;
        if let Some(final_checkpoint) = checkpoints
            .last()
            .filter(|checkpoint| checkpoint.qualifying_winner.is_some())
        {
            let confirmation = if let Some(existing) = existing_confirmation(root)? {
                existing
            } else {
                confirm_winner(root, campaign, staging, final_checkpoint, &test)?
            };
            rebuild_indexes(root, &checkpoints, !campaign.config.shard_sizes.is_empty())?;
            publish_winner(root, campaign, &checkpoints, &confirmation)?;
            return Ok(());
        }
        if campaign
            .config
            .generation_cap
            .is_some_and(|cap| checkpoints.len() >= cap)
        {
            rebuild_indexes(root, &checkpoints, !campaign.config.shard_sizes.is_empty())?;
            publish_exhausted(root, campaign, &checkpoints)?;
            return Ok(());
        }
        let generation = checkpoints.len();
        let population = if let Some(previous) = checkpoints.last() {
            if previous.next_population.len() != POPULATION {
                bail!("Binary42 resume checkpoint has no complete next population");
            }
            previous.next_population.clone()
        } else {
            seeds.members.clone()
        };
        let checkpoint = process_generation(
            root,
            campaign,
            staging,
            generation,
            population,
            checkpoints.last(),
            &seed_semantics,
            &test,
        )?;
        checkpoints.push(checkpoint);
        rebuild_indexes(root, &checkpoints, !campaign.config.shard_sizes.is_empty())?;
    }
}

pub(super) fn record_cohort_event(
    cohort_root: &Path,
    event: &str,
    dataset: Option<&str>,
    detail: Option<&str>,
) -> Result<()> {
    let value = json!({
        "schema": "JopeBinary42AllProgressEvent.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "event": event,
        "dataset": dataset,
        "detail": detail,
        "at_unix_ms": unix_millis(),
    });
    let bytes = canonical_json(&value)?;
    let sha256 = sha256_bytes(&bytes);
    immutable_or_equal(
        &cohort_root
            .join("progress")
            .join(format!("{sha256}.event.v1.json")),
        &bytes,
    )
}

pub(super) fn initialize_all_cohort(
    run_id: &str,
    resume: bool,
) -> Result<(
    FrozenCohort,
    String,
    AllCohortManifest,
    AllStagingReceipt,
    String,
)> {
    validate_run_id(run_id)?;
    let (cohort, cohort_sha256) = read_cohort()?;
    for spec in &cohort.datasets {
        verify_dataset_binding(spec)?;
    }
    let release_binary_sha256 = current_binary_sha256()?;
    let resolved_scq = fs::canonicalize(all_scq_binary_str()).with_context(|| {
        format!(
            "resolve Binary42 SCQ path {all_scq_binary_str}",
            all_scq_binary_str = all_scq_binary_str()
        )
    })?;
    let scq_resolved_path = resolved_scq.to_string_lossy().into_owned();
    let scq_binary_sha256 = sha256_file(&resolved_scq)?;
    let ordered = admission_order(&cohort);
    let run = run_root(run_id);
    let cohort_root = run.join("cohort");
    let campaign_path = cohort_root.join("campaign.v1.json");
    let manifest = if campaign_path.exists() {
        if !resume {
            bail!("Binary42 all-dataset cohort already exists; pass --resume");
        }
        let bytes = fs::read(&campaign_path)?;
        let recorded: AllCohortManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&recorded)?
            || recorded.schema != ALL_CAMPAIGN_SCHEMA
            || !recorded.test_adaptive
            || recorded.benchmark_valid
            || recorded.run_id != run_id
            || recorded.frozen_cohort_sha256 != cohort_sha256
            || recorded.release_binary_sha256 != release_binary_sha256
            || recorded.scq_configured_path != all_scq_binary_str()
            || recorded.scq_resolved_path != scq_resolved_path
            || recorded.scq_binary_sha256 != scq_binary_sha256
            || recorded.config != AllCohortConfig::fixed()
            || recorded.admission_dataset_ids
                != ordered
                    .iter()
                    .map(|dataset| dataset.id.clone())
                    .collect::<Vec<_>>()
            || sha256_file(&cohort_root.join(&recorded.frozen_cohort_file))? != cohort_sha256
        {
            bail!("Binary42 all-dataset cohort resume identity changed");
        }
        recorded
    } else {
        if resume {
            bail!("cannot resume a Binary42 all-dataset cohort that does not exist");
        }
        if run.exists() && fs::read_dir(&run)?.next().is_some() {
            bail!("Binary42 all-dataset run ID collides with an existing run");
        }
        fs::create_dir_all(&cohort_root)?;
        immutable_write(
            &cohort_root.join("binary42-frozen-cohort.v1.json"),
            COHORT_BYTES,
        )?;
        let winner_seed_sources = confirmed_winner_seed_sources(run_id, &cohort)?;
        validate_winner_seed_snapshot(&winner_seed_sources, &cohort)?;
        let created = AllCohortManifest {
            schema: ALL_CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            frozen_cohort_file: "binary42-frozen-cohort.v1.json".to_string(),
            frozen_cohort_sha256: cohort_sha256.clone(),
            release_binary_sha256,
            scq_configured_path: all_scq_binary_str().to_string(),
            scq_resolved_path,
            scq_binary_sha256,
            config: AllCohortConfig::fixed(),
            admission_dataset_ids: ordered.iter().map(|dataset| dataset.id.clone()).collect(),
            winner_seed_sources,
            started_at_unix_ms: unix_millis(),
        };
        write_json(&campaign_path, &created)?;
        record_cohort_event(
            &cohort_root,
            "cohort_created",
            None,
            Some(&format!(
                "{} provenance-verified winner genomes promoted into generation zero",
                created.winner_seed_sources.len()
            )),
        )?;
        created
    };
    validate_winner_seed_snapshot(&manifest.winner_seed_sources, &cohort)?;
    let (staging, staging_sha256) = stage_all_campaign(
        &cohort_root,
        &cohort,
        &manifest.release_binary_sha256,
        &manifest.scq_resolved_path,
        &manifest.scq_binary_sha256,
    )?;
    Ok((cohort, cohort_sha256, manifest, staging, staging_sha256))
}

pub(super) fn summarize_dataset(
    run_id: &str,
    spec: &FrozenDataset,
) -> Result<(DatasetValidationSummary, Vec<AllEvaluationIndexRecord>)> {
    let root = campaign_root(run_id, &spec.id);
    let checkpoints = load_checkpoints(&root)?;
    let complete = root.join("RUN_COMPLETE").exists();
    let exhausted = root.join("RUN_EXHAUSTED").exists();
    if complete && exhausted {
        bail!(
            "Binary42 dataset {} has conflicting terminal markers",
            spec.id
        );
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
    let best_auc = checkpoints
        .iter()
        .map(|checkpoint| checkpoint.best_auc)
        .max_by(f64::total_cmp);
    let unique_genomes = checkpoints
        .iter()
        .flat_map(|checkpoint| &checkpoint.slots)
        .map(|slot| &slot.member.semantic_sha256)
        .collect::<BTreeSet<_>>()
        .len();
    let evaluations = evaluation_records(&checkpoints)
        .into_iter()
        .map(|evaluation| AllEvaluationIndexRecord {
            dataset: spec.id.clone(),
            display_name: spec.display_name.clone(),
            frozen_order: spec.order,
            evaluation,
        })
        .collect();
    Ok((
        DatasetValidationSummary {
            dataset: spec.id.clone(),
            display_name: spec.display_name.clone(),
            frozen_order: spec.order,
            status: status.to_string(),
            completed_generations: checkpoints.len(),
            logical_slots: checkpoints.len() * POPULATION,
            unique_genomes,
            best_auc,
            winner_generation: winner.map(|record| record.generation),
            winner_semantic_sha256: winner.map(|record| record.semantic_sha256.clone()),
        },
        evaluations,
    ))
}

pub(super) fn rebuild_cohort_indexes(
    run_id: &str,
    cohort: &FrozenCohort,
) -> Result<(Vec<DatasetValidationSummary>, usize)> {
    let cohort_root = run_root(run_id).join("cohort");
    let mut datasets = Vec::with_capacity(cohort.datasets.len());
    let mut evaluations = Vec::new();
    for spec in &cohort.datasets {
        let root = campaign_root(run_id, &spec.id);
        if root.exists() {
            let (summary, mut records) = summarize_dataset(run_id, spec)?;
            datasets.push(summary);
            evaluations.append(&mut records);
        } else {
            datasets.push(DatasetValidationSummary {
                dataset: spec.id.clone(),
                display_name: spec.display_name.clone(),
                frozen_order: spec.order,
                status: "pending".to_string(),
                completed_generations: 0,
                logical_slots: 0,
                unique_genomes: 0,
                best_auc: None,
                winner_generation: None,
                winner_semantic_sha256: None,
            });
        }
    }
    let dataset_index = json!({
        "schema": "JopeBinary42AllDatasetIndex.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "datasets": &datasets,
    });
    atomic_rebuild(
        &cohort_root.join("datasets.json"),
        &canonical_json(&dataset_index)?,
    )?;
    let mut dataset_csv = b"dataset,display_name,frozen_order,status,completed_generations,logical_slots,unique_genomes,best_auc,winner_generation,winner_semantic_sha256\n".to_vec();
    for dataset in &datasets {
        dataset_csv.extend_from_slice(
            format!(
                "{},{},{},{},{},{},{},{},{},{}\n",
                csv_field(&dataset.dataset),
                csv_field(&dataset.display_name),
                dataset.frozen_order,
                csv_field(&dataset.status),
                dataset.completed_generations,
                dataset.logical_slots,
                dataset.unique_genomes,
                dataset
                    .best_auc
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                dataset
                    .winner_generation
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                csv_field(dataset.winner_semantic_sha256.as_deref().unwrap_or("")),
            )
            .as_bytes(),
        );
    }
    atomic_rebuild(&cohort_root.join("datasets.csv"), &dataset_csv)?;
    let mut evaluation_jsonl = Vec::new();
    let mut evaluation_csv = b"dataset,display_name,frozen_order,test_adaptive,benchmark_valid,generation,slot,semantic_sha256,ancestry_operation,ancestry_parents,cache_disposition,assigned_node,evaluation_node,scq_job_id,auc,logloss,complexity,runtime_seconds,receipt_sha256,prediction_sha256,failure\n".to_vec();
    for record in &evaluations {
        evaluation_jsonl.extend_from_slice(&canonical_json(record)?);
        let evaluation = &record.evaluation;
        evaluation_csv.extend_from_slice(
            format!(
                "{},{},{},true,false,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                csv_field(&record.dataset),
                csv_field(&record.display_name),
                record.frozen_order,
                evaluation.generation,
                evaluation.slot,
                csv_field(&evaluation.semantic_sha256),
                csv_field(&evaluation.ancestry_operation),
                csv_field(&evaluation.ancestry_parents.join(";")),
                csv_field(&evaluation.cache_disposition),
                csv_field(&evaluation.assigned_node),
                csv_field(&evaluation.evaluation_node),
                csv_field(&evaluation.scq_job_id),
                evaluation
                    .auc
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                evaluation
                    .logloss
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                evaluation.complexity,
                evaluation
                    .runtime_seconds
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                csv_field(&evaluation.receipt_sha256),
                csv_field(&evaluation.prediction_sha256),
                csv_field(evaluation.failure.as_deref().unwrap_or("")),
            )
            .as_bytes(),
        );
    }
    atomic_rebuild(
        &cohort_root.join("all-evaluations.jsonl"),
        &evaluation_jsonl,
    )?;
    atomic_rebuild(&cohort_root.join("all-evaluations.csv"), &evaluation_csv)?;
    let mut winners = b"dataset,display_name,generation,semantic_sha256,best_auc\n".to_vec();
    let mut exhausted = b"dataset,display_name,completed_generations,best_auc\n".to_vec();
    for dataset in &datasets {
        if dataset.status == "complete" {
            winners.extend_from_slice(
                format!(
                    "{},{},{},{},{}\n",
                    csv_field(&dataset.dataset),
                    csv_field(&dataset.display_name),
                    dataset.winner_generation.unwrap(),
                    csv_field(dataset.winner_semantic_sha256.as_deref().unwrap()),
                    dataset.best_auc.unwrap(),
                )
                .as_bytes(),
            );
        } else if dataset.status == "exhausted" {
            exhausted.extend_from_slice(
                format!(
                    "{},{},{},{}\n",
                    csv_field(&dataset.dataset),
                    csv_field(&dataset.display_name),
                    dataset.completed_generations,
                    dataset.best_auc.unwrap(),
                )
                .as_bytes(),
            );
        }
    }
    atomic_rebuild(&cohort_root.join("winners.csv"), &winners)?;
    atomic_rebuild(&cohort_root.join("exhausted.csv"), &exhausted)?;
    Ok((datasets, evaluations.len()))
}

pub(super) fn run_all_impl(run_id: &str, resume: bool) -> Result<()> {
    let (cohort, cohort_sha256, manifest, staging, staging_sha256) =
        initialize_all_cohort(run_id, resume)?;
    let cohort_root = run_root(run_id).join("cohort");
    if cohort_root.join("terminal-summary.v1.json").exists() {
        return validate_all(run_id);
    }
    let ordered = admission_order(&cohort);
    let queue = Arc::new(Mutex::new(VecDeque::from(ordered)));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for _ in 0..ALL_ACTIVE_DATASETS {
            let queue = Arc::clone(&queue);
            let errors = Arc::clone(&errors);
            let manifest = &manifest;
            let staging = &staging;
            let cohort_sha256 = &cohort_sha256;
            let staging_sha256 = &staging_sha256;
            let cohort_root = &cohort_root;
            scope.spawn(move || loop {
                let Some(spec) = queue.lock().unwrap().pop_front() else {
                    break;
                };
                let result = (|| -> Result<()> {
                    record_cohort_event(cohort_root, "dataset_admitted", Some(&spec.id), None)?;
                    let (root, campaign, dataset_staging) = initialize_all_dataset(
                        run_id,
                        &spec,
                        cohort_sha256,
                        manifest,
                        staging,
                        staging_sha256,
                        resume,
                    )?;
                    run_campaign_loop(&root, &campaign, &dataset_staging)?;
                    let terminal = if root.join("RUN_COMPLETE").exists() {
                        "dataset_complete"
                    } else if root.join("RUN_EXHAUSTED").exists() {
                        "dataset_exhausted"
                    } else {
                        bail!(
                            "Binary42 dataset {} returned without a terminal marker",
                            spec.id
                        );
                    };
                    record_cohort_event(cohort_root, terminal, Some(&spec.id), None)
                })();
                if let Err(error) = result {
                    let detail = format!("{error:#}");
                    let _ = record_cohort_event(
                        cohort_root,
                        "dataset_resumable_failure",
                        Some(&spec.id),
                        Some(&detail),
                    );
                    errors
                        .lock()
                        .unwrap()
                        .push(format!("{}: {detail}", spec.id));
                }
            });
        }
    });
    let (datasets, logical_evaluations) = rebuild_cohort_indexes(run_id, &cohort)?;
    let errors = Arc::try_unwrap(errors)
        .expect("Binary42 all-dataset error references remain")
        .into_inner()
        .unwrap();
    if !errors.is_empty() {
        bail!(
            "Binary42 all-dataset cohort retained resumable failures: {}",
            errors.join("; ")
        );
    }
    if datasets
        .iter()
        .any(|dataset| !matches!(dataset.status.as_str(), "complete" | "exhausted"))
    {
        bail!("Binary42 all-dataset cohort finished scheduling without 42 terminal datasets");
    }
    let summary = AllTerminalSummary {
        schema: ALL_SUMMARY_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        terminal_datasets: datasets.len(),
        winners: datasets
            .iter()
            .filter(|dataset| dataset.status == "complete")
            .count(),
        exhausted: datasets
            .iter()
            .filter(|dataset| dataset.status == "exhausted")
            .count(),
        logical_evaluations,
        completed_at_unix_ms: unix_millis(),
    };
    write_json(&cohort_root.join("terminal-summary.v1.json"), &summary)?;
    validate_all(run_id)
}
