use super::*;

pub(crate) fn winner_seed_path(path: &Path) -> Result<String> {
    Ok(path
        .strip_prefix(campaign_state_root())?
        .to_string_lossy()
        .into_owned())
}

pub(crate) fn winner_seed_file(path: &str) -> PathBuf {
    campaign_state_root().join(path)
}

pub(crate) fn matching_confirmation_file(
    dataset_root: &Path,
    expected_sha256: &str,
) -> Result<PathBuf> {
    let confirmations_root = dataset_root.join("winner").join("confirmations");
    let mut attempts = fs::read_dir(&confirmations_root)
        .with_context(|| {
            format!(
                "read Binary42 winner confirmations {}",
                confirmations_root.display()
            )
        })?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    attempts.sort();
    attempts
        .into_iter()
        .map(|attempt| attempt.join("confirmation.v1.json"))
        .find(|path| {
            path.is_file()
                && sha256_file(path)
                    .ok()
                    .as_deref()
                    .is_some_and(|sha256| sha256 == expected_sha256)
        })
        .with_context(|| {
            format!(
                "Binary42 winner confirmation {expected_sha256} is absent under {}",
                confirmations_root.display()
            )
        })
}

pub(crate) fn validate_winner_seed_source(
    source: &WinnerSeedSource,
    cohort: &FrozenCohort,
) -> Result<()> {
    let spec = cohort
        .datasets
        .iter()
        .find(|spec| spec.id == source.source_dataset)
        .context("Binary42 winner seed references an unknown dataset")?;
    let complete_path = winner_seed_file(&source.complete_file);
    let receipt_path = winner_seed_file(&source.final_receipt_file);
    let confirmation_path = winner_seed_file(&source.confirmation_file);
    if sha256_file(&complete_path)? != source.complete_sha256
        || sha256_file(&receipt_path)? != source.final_receipt_sha256
        || sha256_file(&confirmation_path)? != source.confirmation_sha256
    {
        bail!("Binary42 winner seed source hash changed");
    }
    let complete_bytes = fs::read(&complete_path)?;
    let complete: Value = serde_json::from_slice(&complete_bytes)?;
    if complete_bytes != canonical_json(&complete)?
        || complete.get("schema").and_then(Value::as_str) != Some(COMPLETE_SCHEMA)
        || complete.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || complete.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || complete.get("status").and_then(Value::as_str) != Some("complete")
        || complete.get("run_id").and_then(Value::as_str) != Some(source.source_run_id.as_str())
        || complete.get("dataset").and_then(Value::as_str) != Some(source.source_dataset.as_str())
        || complete.get("final_receipt_sha256").and_then(Value::as_str)
            != Some(source.final_receipt_sha256.as_str())
    {
        bail!("Binary42 winner seed RUN_COMPLETE is invalid");
    }
    let receipt_bytes = fs::read(&receipt_path)?;
    let receipt: WinnerReceipt = serde_json::from_slice(&receipt_bytes)?;
    if receipt_bytes != canonical_json(&receipt)?
        || receipt.schema != WINNER_SCHEMA
        || !receipt.test_adaptive
        || receipt.benchmark_valid
        || receipt.run_id != source.source_run_id
        || receipt.dataset != source.source_dataset
        || receipt.generation == 0
        || receipt.semantic_sha256 != source.semantic_sha256
        || receipt.canonical_genome != source.canonical_genome
        || receipt.auc.to_bits() != source.auc_bits
        || receipt.frozen_leader_auc.to_bits() != source.frozen_leader_auc_bits
        || receipt.generation_zero_best_auc.to_bits() != source.generation_zero_best_auc_bits
        || receipt.four_node_confirmation_sha256 != source.confirmation_sha256
        || sha256_bytes(receipt.canonical_genome.as_bytes()) != receipt.semantic_sha256
        || !policy_win(
            &spec.win_policy,
            receipt.auc,
            receipt.frozen_leader_auc,
            receipt.generation_zero_best_auc,
        )
    {
        bail!("Binary42 winner seed final receipt is invalid");
    }
    let confirmation_bytes = fs::read(&confirmation_path)?;
    let confirmation: ConfirmationManifest = serde_json::from_slice(&confirmation_bytes)?;
    if confirmation_bytes != canonical_json(&confirmation)?
        || confirmation.schema != CONFIRMATION_SCHEMA
        || !confirmation.test_adaptive
        || confirmation.benchmark_valid
        || confirmation.run_id != source.source_run_id
        || confirmation.dataset != source.source_dataset
        || confirmation.semantic_sha256 != source.semantic_sha256
        || !confirmation.prediction_hashes_identical
        || !confirmation.auc_bits_identical
        || confirmation.confirmations.len() != NODES.len()
        || confirmation
            .confirmations
            .iter()
            .enumerate()
            .any(|(index, record)| {
                record.node != NODES[index]
                    || record.auc_bits != source.auc_bits
                    || record.prediction_sha256 != confirmation.original_prediction_sha256
            })
    {
        bail!("Binary42 winner seed confirmation is invalid");
    }
    let dataset_root = complete_path
        .parent()
        .context("Binary42 winner seed marker has no dataset directory")?;
    let source_seeds: SeedManifest =
        load_json(&dataset_root.join("seeds").join("generation-zero.v1.json"))?;
    if source_seeds
        .members
        .iter()
        .any(|member| member.semantic_sha256 == source.semantic_sha256)
    {
        bail!("Binary42 winner seed was not an evolved descendant");
    }
    Ok(())
}

pub(crate) fn confirmed_winner_seed_sources(
    current_run_id: &str,
    cohort: &FrozenCohort,
) -> Result<Vec<WinnerSeedSource>> {
    let runs_root = campaign_state_root().join("runs");
    let current_run = run_root(current_run_id);
    let mut run_roots = fs::read_dir(&runs_root)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    run_roots.sort();
    let mut best_by_dataset = BTreeMap::<String, WinnerSeedSource>::new();
    for run_path in run_roots {
        if run_path == current_run
            || !run_path.is_dir()
            || !run_path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("binary42-test-adaptive-"))
        {
            continue;
        }
        let mut dataset_roots = fs::read_dir(&run_path)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.file_type().ok()?.is_dir().then_some(entry.path()))
            .collect::<Vec<_>>();
        dataset_roots.sort();
        for dataset_root in dataset_roots {
            let complete_path = dataset_root.join("RUN_COMPLETE");
            let receipt_path = dataset_root.join("winner").join("final-receipt.v1.json");
            if !complete_path.is_file() || !receipt_path.is_file() {
                continue;
            }
            let receipt_bytes = fs::read(&receipt_path)?;
            let receipt: WinnerReceipt = serde_json::from_slice(&receipt_bytes)?;
            let confirmation_path =
                matching_confirmation_file(&dataset_root, &receipt.four_node_confirmation_sha256)?;
            let source = WinnerSeedSource {
                source_run_id: receipt.run_id.clone(),
                source_dataset: receipt.dataset.clone(),
                complete_file: winner_seed_path(&complete_path)?,
                complete_sha256: sha256_file(&complete_path)?,
                final_receipt_file: winner_seed_path(&receipt_path)?,
                final_receipt_sha256: sha256_bytes(&receipt_bytes),
                confirmation_file: winner_seed_path(&confirmation_path)?,
                confirmation_sha256: sha256_file(&confirmation_path)?,
                canonical_genome: receipt.canonical_genome.clone(),
                semantic_sha256: receipt.semantic_sha256.clone(),
                auc_bits: receipt.auc.to_bits(),
                frozen_leader_auc_bits: receipt.frozen_leader_auc.to_bits(),
                generation_zero_best_auc_bits: receipt.generation_zero_best_auc.to_bits(),
            };
            validate_winner_seed_source(&source, cohort).with_context(|| {
                format!(
                    "validate Binary42 winner seed {} from {}",
                    source.source_dataset,
                    receipt_path.display()
                )
            })?;
            match best_by_dataset.get(&source.source_dataset) {
                Some(incumbent)
                    if f64::from_bits(incumbent.auc_bits) > f64::from_bits(source.auc_bits)
                        || (incumbent.auc_bits == source.auc_bits
                            && incumbent.semantic_sha256 <= source.semantic_sha256) => {}
                _ => {
                    best_by_dataset.insert(source.source_dataset.clone(), source);
                }
            }
        }
    }
    let frozen_order = cohort
        .datasets
        .iter()
        .map(|spec| (spec.id.as_str(), spec.order))
        .collect::<BTreeMap<_, _>>();
    let mut sources = best_by_dataset.into_values().collect::<Vec<_>>();
    sources.sort_by(|left, right| {
        frozen_order
            .get(left.source_dataset.as_str())
            .cmp(&frozen_order.get(right.source_dataset.as_str()))
            .then_with(|| left.source_dataset.cmp(&right.source_dataset))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    let mut seen = BTreeSet::new();
    sources.retain(|source| seen.insert(source.semantic_sha256.clone()));
    if sources.len() > cohort.datasets.len() || sources.len() > POPULATION {
        bail!("Binary42 confirmed winner seed snapshot is unexpectedly large");
    }
    Ok(sources)
}

pub(crate) fn validate_winner_seed_snapshot(
    sources: &[WinnerSeedSource],
    cohort: &FrozenCohort,
) -> Result<()> {
    if sources.len() > cohort.datasets.len() || sources.len() > POPULATION {
        bail!("Binary42 confirmed winner seed snapshot is unexpectedly large");
    }
    let frozen_order = cohort
        .datasets
        .iter()
        .map(|spec| (spec.id.as_str(), spec.order))
        .collect::<BTreeMap<_, _>>();
    let mut datasets = BTreeSet::new();
    let mut semantics = BTreeSet::new();
    let mut previous_order = None;
    for source in sources {
        validate_winner_seed_source(source, cohort)?;
        if !datasets.insert(source.source_dataset.clone())
            || !semantics.insert(source.semantic_sha256.clone())
        {
            bail!("Binary42 winner seed snapshot is not unique");
        }
        let order = *frozen_order
            .get(source.source_dataset.as_str())
            .context("Binary42 winner seed source has no frozen order")?;
        if previous_order.is_some_and(|previous| previous >= order) {
            bail!("Binary42 winner seed snapshot is not in frozen dataset order");
        }
        previous_order = Some(order);
    }
    Ok(())
}

pub(crate) fn confirmation_attempt(root: &Path) -> Result<usize> {
    let confirmation_root = root.join("winner").join("confirmations");
    if !confirmation_root.exists() {
        return Ok(0);
    }
    Ok(fs::read_dir(confirmation_root)?.count())
}

pub(crate) fn store_confirmation_output(
    root: &Path,
    output: FreshOutput,
    test: &RegressionData,
) -> Result<ConfirmationRecord> {
    let score =
        validate_fresh_output(&output, test)?.context("Binary42 confirmation candidate failed")?;
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
    Ok(ConfirmationRecord {
        node: output.receipt.node,
        job_id: output.receipt.scq_job_id,
        receipt_sha256,
        prediction_sha256,
        auc: score.auc,
        logloss: score.logloss,
        auc_bits: score.auc.to_bits(),
    })
}

pub(crate) fn confirm_winner(
    root: &Path,
    campaign: &CampaignManifest,
    staging: &StagingReceipt,
    checkpoint: &GenerationCheckpoint,
    test: &RegressionData,
) -> Result<ConfirmationManifest> {
    let selected = checkpoint
        .qualifying_winner
        .as_ref()
        .context("Binary42 checkpoint has no qualifying winner")?;
    let slot = checkpoint
        .slots
        .get(selected.slot)
        .filter(|slot| slot.member.semantic_sha256 == selected.semantic_sha256)
        .context("Binary42 qualifying winner slot mismatch")?;
    let attempt = confirmation_attempt(root)?;
    let manifests = staging
        .nodes
        .iter()
        .enumerate()
        .map(|(node_index, staged)| {
            shard_manifest(
                campaign,
                staged,
                selected.generation,
                node_index,
                attempt * 3,
                "confirmation",
                std::slice::from_ref(&slot.member),
                &HashMap::new(),
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
                results
                    .lock()
                    .unwrap()
                    .push((index, execute_shard(root, manifest)));
            });
        }
    });
    let mut results = Arc::try_unwrap(results)
        .expect("Binary42 confirmation result references remain")
        .into_inner()
        .unwrap();
    results.sort_by_key(|(index, _)| *index);
    let mut confirmations = Vec::new();
    for (index, result) in results {
        let mut outputs = match result {
            Ok(outputs) => outputs,
            Err(error) => {
                record_infrastructure_failure(root, &manifests[index], &error)?;
                let mut recovered = None;
                let mut final_error = error;
                for retry_attempt in 1..=2 {
                    let retry = shard_manifest(
                        campaign,
                        &staging.nodes[index],
                        selected.generation,
                        index,
                        attempt * 3 + retry_attempt,
                        "confirmation",
                        std::slice::from_ref(&slot.member),
                        &HashMap::new(),
                    )?;
                    match execute_shard(root, &retry) {
                        Ok(outputs) => {
                            recovered = Some(outputs);
                            break;
                        }
                        Err(error) => {
                            record_infrastructure_failure(root, &retry, &error)?;
                            final_error = error;
                        }
                    }
                }
                if recovered.is_none() {
                    let failure = json!({
                        "schema": "JopeBinary42TestAdaptiveConfirmationInfrastructureFailure.v1",
                        "test_adaptive": true,
                        "benchmark_valid": false,
                        "run_id": campaign.run_id,
                        "dataset": campaign.dataset,
                        "generation": selected.generation,
                        "node": manifests[index].node,
                        "attempt": attempt,
                        "error": format!("{final_error:#}"),
                    });
                    write_json(
                        &root
                            .join("winner")
                            .join("confirmations")
                            .join(format!("attempt-{attempt:03}"))
                            .join("INFRASTRUCTURE_FAILED.v1.json"),
                        &failure,
                    )?;
                }
                recovered.with_context(|| {
                    format!(
                        "Binary42 confirmation failed on {} after two retries: {final_error:#}",
                        manifests[index].node
                    )
                })?
            }
        };
        if outputs.len() != 1 {
            bail!("Binary42 confirmation job returned the wrong candidate count");
        }
        confirmations.push(store_confirmation_output(root, outputs.remove(0), test)?);
    }
    if confirmations
        .iter()
        .enumerate()
        .any(|(index, confirmation)| confirmation.node != NODES[index])
    {
        bail!("Binary42 confirmation node order mismatch");
    }
    let prediction_hashes_identical = confirmations
        .iter()
        .all(|confirmation| confirmation.prediction_sha256 == slot.prediction_sha256);
    let auc_bits_identical = confirmations
        .iter()
        .all(|confirmation| confirmation.auc_bits == selected.auc.to_bits());
    let confirmation = ConfirmationManifest {
        schema: CONFIRMATION_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        generation: selected.generation,
        selected_slot: selected.slot,
        semantic_sha256: selected.semantic_sha256.clone(),
        attempt,
        original_prediction_sha256: slot.prediction_sha256.clone(),
        confirmations,
        prediction_hashes_identical,
        auc_bits_identical,
    };
    let attempt_root = root
        .join("winner")
        .join("confirmations")
        .join(format!("attempt-{attempt:03}"));
    if !prediction_hashes_identical || !auc_bits_identical {
        write_json(
            &attempt_root.join("CONFIRMATION_FAILED.v1.json"),
            &confirmation,
        )?;
        bail!("Binary42 four-node confirmation diverged; campaign retained for resume");
    }
    write_json(&attempt_root.join("confirmation.v1.json"), &confirmation)?;
    Ok(confirmation)
}
