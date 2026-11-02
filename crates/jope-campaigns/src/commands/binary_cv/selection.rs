use super::*;

/// Binary panel-facing selection: mirrors the regression lane's structure
/// (quick stratified CV5 fold-0 race, CV7 confirmation for the top 10) with
/// v2-binary receipts throughout.
pub(crate) fn select_dataset_binary(
    train: &RegressionData,
    dataset: &str,
    train_sha256: &str,
    seed: u64,
    seed_bytes: &[u8],
    artifact_root: &Path,
    generations: usize,
) -> Result<serde_json::Value> {
    let seed_sha = sha256_bytes(seed_bytes);
    if seed_sha != SEED_BINARY_SHA256 {
        bail!("fixed binary seed population hash mismatch: {seed_sha}");
    }
    require_fresh_artifact_root(artifact_root)?;
    fs::create_dir_all(artifact_root)?;
    let task = binary_task();
    let schema = binary_schema(train.features);
    let mut population = parse_binary_population(seed_bytes, POPULATION, &task, &schema)?;
    let n = train.x.len().min(train.y.len());
    let positives = train.y[..n].iter().filter(|v| **v == 1.0).count();
    let identity = BinaryCampaignIdentity {
        schema: CAMPAIGN_SCHEMA,
        evaluator: EVALUATOR,
        task: TaskKind::Binary.as_str(),
        dataset_hash: dataset,
        train_sha256,
        train_rows: n,
        positives,
        negatives: n - positives,
        generations,
        population: POPULATION,
        cv_folds: CV5_FOLDS,
        confirm_folds: CV7_FOLDS,
        elites: ELITES,
        parent_pool: PARENT_POOL,
        tournament: TOURNAMENT,
        crossover_probability: 0.25,
        mutation_probability: 0.75,
        seed,
        member_execution: "one scoped thread per population member (no core pinning)",
        initial_population_sha256: &seed_sha,
        quick_holdout_fold: QUICK_HOLDOUT_FOLD,
        final_cv7_candidates: FINAL_CV7_CANDIDATES,
        fold_authority: FOLD_AUTHORITY,
    };
    immutable_write(
        &artifact_root.join("campaign.v2-binary.json"),
        &canonical_json(&identity)?,
    )?;
    let cv5_manifest = build_binary_fold_manifest(dataset, train_sha256, &train.y[..n], CV5_FOLDS)?;
    let cv5_manifest_bytes = canonical_json(&cv5_manifest)?;
    let cv5_manifest_sha256 = sha256_bytes(&cv5_manifest_bytes);
    immutable_write(
        &artifact_root.join("folds").join("cv5.v2-binary.json"),
        &cv5_manifest_bytes,
    )?;
    let ledger_path = artifact_root.join("slots.v2-binary.jsonl");
    let mut ledger_file = OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(&ledger_path)
        .with_context(|| format!("create append-only binary ledger {}", ledger_path.display()))?;
    let fold_cache = BinaryFoldCache::new(train, dataset);
    let mut memo = HashMap::<String, BinaryEvaluation>::new();
    let mut final_ranked = Vec::<BinarySlotLedger>::new();

    for generation in 0..generations {
        let evaluations = thread::scope(|scope| -> Result<Vec<(bool, BinaryEvaluation)>> {
            let mut handles = Vec::with_capacity(population.len());
            for (slot, member) in population.iter().enumerate() {
                let memo = &memo;
                let fold_cache = &fold_cache;
                let task = &task;
                let schema = &schema;
                handles.push(
                    thread::Builder::new()
                        .name(format!("jain-bin-g{generation:02}-s{slot:02}"))
                        .spawn_scoped(scope, move || -> (bool, BinaryEvaluation) {
                            if let Some(hit) = memo.get(&member.semantic_sha256) {
                                return (false, hit.clone());
                            }
                            let evaluation =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    evaluate_binary_unique(member, fold_cache, task, schema)
                                }))
                                .unwrap_or_else(|_| {
                                    BinaryEvaluation {
                                        complexity: 0,
                                        result: None,
                                        rejection_reason: Some(
                                            "panic during evaluation".to_string(),
                                        ),
                                    }
                                });
                            (true, evaluation)
                        })
                        .with_context(|| {
                            format!("spawn binary member thread g{generation} s{slot}")
                        })?,
                );
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("binary member thread panicked"))
                })
                .collect()
        })?;
        let mut slots = Vec::with_capacity(POPULATION);
        for (slot, (miss, evaluation)) in evaluations.into_iter().enumerate() {
            memo.entry(population[slot].semantic_sha256.clone())
                .or_insert_with(|| evaluation.clone());
            let result = evaluation.result.as_ref();
            slots.push(BinarySlotLedger {
                schema: LEDGER_SCHEMA.to_string(),
                generation,
                slot,
                canonical_genome: population[slot].canonical_genome.clone(),
                semantic_sha256: population[slot].semantic_sha256.clone(),
                ancestry: population[slot].ancestry.clone(),
                complexity: evaluation.complexity,
                valid: result.is_some(),
                rejection_reason: evaluation.rejection_reason.clone(),
                cache_disposition: if miss { "miss" } else { "hit" }.to_string(),
                fold_auc: result.map(|r| optional_folds(&r.fold_auc)),
                fold_defined: result.map(|r| r.fold_defined.clone()),
                pooled_oof_auc: result.map(|r| r.pooled_oof_auc),
                pooled_oof_logloss: result.map(|r| r.pooled_oof_logloss),
                pooled_oof_brier: result.map(|r| r.pooled_oof_brier),
                robust_fitness: result.map(|r| r.robust_fitness),
            });
        }
        let mut ranked = slots
            .iter()
            .filter(|slot| slot.rankable())
            .cloned()
            .collect::<Vec<_>>();
        ranked.sort_by(rank_binary_slots);
        if ranked.len() < PARENT_POOL {
            bail!(
                "binary generation {generation} produced only {} valid genomes; selection requires {}",
                ranked.len(),
                PARENT_POOL
            );
        }
        for slot in &slots {
            ledger_file.write_all(&canonical_json(slot)?)?;
        }
        ledger_file.flush()?;
        ledger_file.sync_all()?;
        eprintln!(
            "[jain-binary-cv] generation={generation} valid={} best={:.12}",
            ranked.len(),
            ranked[0].robust_fitness.unwrap()
        );
        let next = if generation + 1 < generations {
            refill_binary_population(seed, generation + 1, &ranked, &task, &schema)?
        } else {
            Vec::new()
        };
        final_ranked = ranked;
        population = next;
    }
    drop(ledger_file);

    if final_ranked.len() < FINAL_CV7_CANDIDATES {
        bail!(
            "binary final generation has only {} candidates; CV7 requires {}",
            final_ranked.len(),
            FINAL_CV7_CANDIDATES
        );
    }
    let cv7_manifest = build_binary_fold_manifest(dataset, train_sha256, &train.y[..n], CV7_FOLDS)?;
    let cv7_manifest_bytes = canonical_json(&cv7_manifest)?;
    let cv7_manifest_sha256 = sha256_bytes(&cv7_manifest_bytes);
    immutable_write(
        &artifact_root.join("folds").join("cv7.v2-binary.json"),
        &cv7_manifest_bytes,
    )?;
    let cv7_fold_ids = (0..CV7_FOLDS).collect::<Vec<_>>();
    let mut finalists = Vec::with_capacity(FINAL_CV7_CANDIDATES);
    let mut cv7_results = HashMap::<String, BinaryCvResult>::new();
    for (quick_rank, slot) in final_ranked.iter().take(FINAL_CV7_CANDIDATES).enumerate() {
        let quick_evidence = BinarySelectionEvidence {
            fold_plan_folds: CV5_FOLDS,
            evaluated_fold_ids: vec![QUICK_HOLDOUT_FOLD],
            fold_manifest_sha256: cv5_manifest_sha256.clone(),
            fold_auc: slot.fold_auc.clone().unwrap(),
            fold_defined: slot.fold_defined.clone().unwrap(),
            pooled_oof_auc: slot.pooled_oof_auc.unwrap(),
            pooled_oof_logloss: slot.pooled_oof_logloss.unwrap(),
            pooled_oof_brier: slot.pooled_oof_brier.unwrap(),
            robust_fitness: slot.robust_fitness.unwrap(),
        };
        let genome = GenomeIr::parse(&slot.canonical_genome)?;
        let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            binary_cv_on_slices(
                &genome,
                &fold_cache,
                CV7_FOLDS,
                &cv7_fold_ids,
                &task,
                &schema,
            )
        }));
        let (cv7, rejection_reason) = match evaluated {
            Ok(Ok(result)) => {
                let evidence = BinarySelectionEvidence::from_result(
                    &result,
                    CV7_FOLDS,
                    cv7_fold_ids.clone(),
                    &cv7_manifest_sha256,
                );
                cv7_results.insert(slot.semantic_sha256.clone(), result);
                (Some(evidence), None)
            }
            Ok(Err(error)) => (None, Some(format!("cv7: {error:#}"))),
            Err(_) => (None, Some("cv7 panic".to_string())),
        };
        finalists.push(BinaryFinalist {
            quick_rank,
            generation: generations - 1,
            slot: slot.slot,
            canonical_genome: slot.canonical_genome.clone(),
            semantic_sha256: slot.semantic_sha256.clone(),
            complexity: slot.complexity,
            quick_cv5_fold0: quick_evidence,
            cv7,
            rejection_reason,
        });
    }
    for finalist in &finalists {
        immutable_write(
            &artifact_root.join("finalists").join(format!(
                "quick-rank-{:02}.cv7.v2-binary.json",
                finalist.quick_rank
            )),
            &canonical_json(finalist)?,
        )?;
    }
    let mut confirmed = finalists
        .iter()
        .filter(|finalist| finalist.cv7.is_some())
        .collect::<Vec<_>>();
    confirmed.sort_by(|left, right| {
        let l = left.cv7.as_ref().unwrap();
        let r = right.cv7.as_ref().unwrap();
        r.robust_fitness
            .total_cmp(&l.robust_fitness)
            .then_with(|| l.pooled_oof_logloss.total_cmp(&r.pooled_oof_logloss))
            .then_with(|| left.complexity.cmp(&right.complexity))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    let selected = confirmed
        .first()
        .copied()
        .context("none of the binary finalists completed full CV7")?;
    let winner_cv7 = cv7_results
        .get(&selected.semantic_sha256)
        .context("winner CV7 evidence missing")?;
    // Leak-safe, report-only thresholds from the winner's pooled CV7
    // train-OOF probabilities; frozen into the receipt, never applied to the
    // artifact's probability output.
    let (f1_threshold, f1_value) =
        best_f1_threshold(&winner_cv7.oof_truth, &winner_cv7.oof_probabilities);
    let (youden_thr, youden_value) =
        youden_threshold(&winner_cv7.oof_truth, &winner_cv7.oof_probabilities);
    let thresholds = BinaryThresholds {
        f1_threshold,
        f1_value,
        youden_threshold: youden_thr,
        youden_value,
    };
    // Winner bound artifacts: fit on the full training table, emit, publish
    // content-addressed and immutable.
    let winner_genome = GenomeIr::parse(&selected.canonical_genome)?;
    let full_table = raw_table(train.features, &train.x[..n])?;
    let winner_model = fit_table(&winner_genome, &task, &schema, &full_table, &train.y[..n])?;
    let emitted = emit_bound(&winner_model, &feat_math::transpile::EmitOptions::default())?;
    let artifact_dir = artifact_root
        .join("artifacts")
        .join(&selected.semantic_sha256);
    let python_sha256 = sha256_bytes(emitted.python.as_bytes());
    let rust_sha256 = sha256_bytes(emitted.rust.as_bytes());
    let manifest_sha256 = sha256_bytes(emitted.manifest.as_bytes());
    immutable_write(
        &artifact_dir.join(format!("{python_sha256}.py")),
        emitted.python.as_bytes(),
    )?;
    immutable_write(
        &artifact_dir.join(format!("{rust_sha256}.rs")),
        emitted.rust.as_bytes(),
    )?;
    immutable_write(
        &artifact_dir.join(format!("{manifest_sha256}.manifest.json")),
        emitted.manifest.as_bytes(),
    )?;
    let winner_artifacts = BinaryArtifactHashes {
        python_sha256,
        rust_sha256,
        manifest_sha256,
    };
    make_readonly(&ledger_path)?;
    let ledger_sha256 = sha256_file(&ledger_path)?;
    let selection = BinarySelectionFreeze {
        schema: SELECTION_SCHEMA.to_string(),
        evaluator: EVALUATOR.to_string(),
        task: TaskKind::Binary.as_str().to_string(),
        dataset_hash: dataset.to_string(),
        train_sha256: train_sha256.to_string(),
        initial_population_sha256: seed_sha,
        generation: generations - 1,
        slot: selected.slot,
        canonical_genome: selected.canonical_genome.clone(),
        semantic_sha256: selected.semantic_sha256.clone(),
        complexity: selected.complexity,
        ranking_authority: "top 10 by quick stratified-CV5 fold-0 robust fitness; final selection by full CV7 robust fitness descending, pooled OOF logloss ascending, complexity ascending, semantic_sha256 ascending".to_string(),
        ledger_sha256,
        quick_cv5_fold0: selected.quick_cv5_fold0.clone(),
        cv7_pooled_oof_auc: winner_cv7.pooled_oof_auc,
        cv7: selected.cv7.clone().unwrap(),
        thresholds,
        winner_artifacts,
        cv7_finalists: finalists,
    };
    let selection_bytes = canonical_json(&selection)?;
    let selection_sha256 = sha256_bytes(&selection_bytes);
    immutable_write(
        &artifact_root.join("selection-freeze.v2-binary.json"),
        &selection_bytes,
    )?;
    let complete = BinarySelectionComplete {
        schema: SELECTION_COMPLETE_SCHEMA.to_string(),
        status: "complete".to_string(),
        selection_file: "selection-freeze.v2-binary.json".to_string(),
        selection_sha256,
    };
    immutable_write(
        &artifact_root.join("SELECTION_COMPLETE"),
        &canonical_json(&complete)?,
    )?;
    Ok(serde_json::to_value(&selection)?)
}

/// Light immutable-freeze validation for the binary lane: receipt parses,
/// identity matches, ledger hash matches the sealed file, the winner genome
/// is canonical + admissible, and the evidence is internally consistent.
pub(crate) fn validate_freeze_binary(
    artifact_root: &Path,
    dataset: &str,
    train_sha256: &str,
    features: usize,
    generations: usize,
) -> Result<BinarySelectionFreeze> {
    let bytes = fs::read(artifact_root.join("selection-freeze.v2-binary.json"))
        .with_context(|| format!("read binary freeze under {}", artifact_root.display()))?;
    let freeze: BinarySelectionFreeze =
        serde_json::from_slice(&bytes).context("parse binary selection freeze")?;
    if freeze.schema != SELECTION_SCHEMA
        || freeze.evaluator != EVALUATOR
        || freeze.dataset_hash != dataset
        || freeze.train_sha256 != train_sha256
        || freeze.generation + 1 != generations
    {
        bail!("binary freeze identity mismatch for {dataset}");
    }
    let ledger_sha = sha256_file(&artifact_root.join("slots.v2-binary.jsonl"))?;
    if ledger_sha != freeze.ledger_sha256 {
        bail!("binary ledger hash differs from frozen receipt");
    }
    let genome = GenomeIr::parse(&freeze.canonical_genome)?;
    if genome.canonical() != freeze.canonical_genome
        || sha256_bytes(freeze.canonical_genome.as_bytes()) != freeze.semantic_sha256
    {
        bail!("binary freeze genome identity mismatch");
    }
    validate_v3_genome(&genome, &binary_task(), &binary_schema(features))?;
    if !freeze.cv7.robust_fitness.is_finite() || !freeze.cv7_pooled_oof_auc.is_finite() {
        bail!("binary freeze evidence is non-finite");
    }
    let complete_bytes = fs::read(artifact_root.join("SELECTION_COMPLETE"))
        .context("binary SELECTION_COMPLETE marker missing")?;
    let complete: BinarySelectionComplete = serde_json::from_slice(&complete_bytes)?;
    if complete.schema != SELECTION_COMPLETE_SCHEMA || complete.status != "complete" {
        bail!("binary completion marker is invalid");
    }
    Ok(freeze)
}

// ---------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BinaryScore {
    pub(crate) dataset_hash: String,
    pub(crate) semantic_sha256: String,
    pub(crate) canonical_genome: String,
    pub(crate) complexity: usize,
    pub(crate) test_rows: usize,
    pub(crate) auc: f64,
    pub(crate) logloss: f64,
    pub(crate) brier: f64,
    pub(crate) average_precision: f64,
    pub(crate) lift_at_10pct: f64,
    pub(crate) accuracy_at_half: f64,
    pub(crate) f1_at_half: f64,
    pub(crate) frozen_f1_threshold: f64,
    pub(crate) accuracy_at_frozen_threshold: f64,
    pub(crate) f1_at_frozen_threshold: f64,
    pub(crate) fit_seconds: f64,
    pub(crate) predict_seconds: f64,
    pub(crate) predictions: Vec<f64>,
}

/// Score one frozen binary selection on held-out test data. AUC is the sole
/// win/loss authority; everything else is report-only.
pub(crate) fn score_dataset_binary(
    artifact_root: &Path,
    dataset: &str,
    train: &RegressionData,
    train_sha256: &str,
    test: &RegressionData,
    generations: usize,
) -> Result<BinaryScore> {
    let freeze = validate_freeze_binary(
        artifact_root,
        dataset,
        train_sha256,
        train.features,
        generations,
    )?;
    if test.features != train.features {
        bail!("test width differs from train width for {dataset}");
    }
    let task = binary_task();
    let schema = binary_schema(train.features);
    let genome = GenomeIr::parse(&freeze.canonical_genome)?;
    let n = train.x.len().min(train.y.len());
    let train_table = raw_table(train.features, &train.x[..n])?;
    let fit_started = Instant::now();
    let model = fit_table(&genome, &task, &schema, &train_table, &train.y[..n])?;
    let fit_seconds = fit_started.elapsed().as_secs_f64();
    let test_table = raw_table(test.features, &test.x)?;
    let predict_started = Instant::now();
    let predictions = model.predict_table(&test_table)?;
    let predict_seconds = predict_started.elapsed().as_secs_f64();
    validate_probabilities(&predictions, test.y.len())?;
    let thr = freeze.thresholds.f1_threshold;
    Ok(BinaryScore {
        dataset_hash: dataset.to_string(),
        semantic_sha256: freeze.semantic_sha256.clone(),
        canonical_genome: freeze.canonical_genome.clone(),
        complexity: freeze.complexity,
        test_rows: test.y.len(),
        auc: auc(&test.y, &predictions),
        logloss: logloss(&test.y, &predictions),
        brier: brier(&test.y, &predictions),
        average_precision: average_precision(&test.y, &predictions),
        lift_at_10pct: lift_at(&test.y, &predictions, 0.1),
        accuracy_at_half: accuracy_at(&test.y, &predictions, 0.5),
        f1_at_half: f1_at(&test.y, &predictions, 0.5),
        frozen_f1_threshold: thr,
        accuracy_at_frozen_threshold: accuracy_at(&test.y, &predictions, thr),
        f1_at_frozen_threshold: f1_at(&test.y, &predictions, thr),
        fit_seconds,
        predict_seconds,
        predictions,
    })
}
