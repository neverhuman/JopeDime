use super::*;

pub(super) fn validate_static_inputs() -> Result<()> {
    if sha256_bytes(seed_bytes()) != SEED_SHA256 {
        bail!("seed population hash mismatch");
    }
    let panel_path = manifest_dir().join("panel.txt");
    if hash_file(&panel_path)? != PANEL_SHA256 {
        bail!("panel text SHA-256 mismatch");
    }
    let panel = fs::read_to_string(panel_path)?;
    let listed = panel.lines().collect::<Vec<_>>();
    if listed != DATASETS {
        bail!("compiled dataset order differs from the pinned panel");
    }
    Ok(())
}

pub(super) fn verify_inputs(registry: &Path) -> Result<()> {
    validate_static_inputs()?;
    for hash in DATASETS {
        let train_path = registry.join(hash).join("train.csv");
        let train = read_regression_csv(&train_path)?;
        if train.y.len() < 7 {
            bail!("{hash} has too few train rows for balanced CV7");
        }
        let digest = hash_file(&train_path)?;
        println!(
            "{hash} train_rows={} features={} train_sha256={digest}",
            train.y.len(),
            train.features
        );
    }
    println!("verified 42 selection inputs without opening test data or baseline metrics");
    Ok(())
}

pub(super) fn select_panel(registry: &Path, panel_id: &str, only: Option<&str>) -> Result<()> {
    validate_static_inputs()?;
    if only.is_some_and(|wanted| !DATASETS.contains(&wanted)) {
        bail!("requested dataset is absent from the pinned panel");
    }
    let root = run_root(panel_id);
    fs::create_dir_all(&root)?;
    let mut core_plan = None;
    let started = Instant::now();
    for (index, hash) in DATASETS.iter().enumerate() {
        if only.is_some_and(|wanted| wanted != *hash) {
            continue;
        }
        let train_path = registry.join(hash).join("train.csv");
        let train = read_regression_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let seed = dataset_seed(hash)?;
        let artifact = root.join(hash);
        if artifact.join("SELECTION_COMPLETE").exists()
            || artifact.join("selection-freeze.v1.json").exists()
        {
            exact_cv::validate_freeze(
                &artifact,
                hash,
                &train_sha256,
                train.y.len(),
                seed,
                ORIGINAL_GENERATIONS,
            )
            .with_context(|| format!("existing immutable freeze is invalid for {hash}"))?;
            continue;
        }
        if artifact.exists() && fs::read_dir(&artifact)?.next().transpose()?.is_some() {
            bail!(
                "incomplete nonempty campaign directory requires operator recovery: {}",
                artifact.display()
            );
        }
        let cores = match &core_plan {
            Some(cores) => cores,
            None => core_plan.insert(require_core_plan(60)?),
        };
        eprintln!(
            "[jope-dime] selecting {hash} ({}/{})",
            index + 1,
            DATASETS.len()
        );
        exact_cv::select_dataset(
            &train,
            hash,
            &train_sha256,
            seed,
            seed_bytes(),
            &artifact,
            cores.clone(),
            ORIGINAL_GENERATIONS,
        )?;
        exact_cv::validate_freeze(
            &artifact,
            hash,
            &train_sha256,
            train.y.len(),
            seed,
            ORIGINAL_GENERATIONS,
        )?;
    }
    println!(
        "selection pass complete in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

pub(super) fn load_validated_panel(
    registry: &Path,
    panel_id: &str,
) -> Result<BTreeMap<String, ValidatedFreeze>> {
    validate_static_inputs()?;
    let root = run_root(panel_id);
    let mut frozen = BTreeMap::new();
    let mut semantics = BTreeSet::new();
    let mut slots = 0usize;
    let mut finalists = 0usize;
    for hash in DATASETS {
        // Training-only firewall: no test path or baseline manifest is named here.
        let train_path = registry.join(hash).join("train.csv");
        let train = read_regression_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let validated = exact_cv::validate_freeze(
            &root.join(hash),
            hash,
            &train_sha256,
            train.y.len(),
            dataset_seed(hash)?,
            ORIGINAL_GENERATIONS,
        )
        .with_context(|| format!("selection freeze validation failed for {hash}"))?;
        if !semantics.insert(validated.semantic_sha256.clone()) {
            bail!("selected semantic genome is reused across datasets");
        }
        slots += validated.evolutionary_slots;
        finalists += validated.finalist_receipts;
        frozen.insert((*hash).to_string(), validated);
    }
    if frozen.len() != 42 || slots != 25_200 || finalists != 420 || semantics.len() != 42 {
        bail!(
            "panel accounting failed: freezes={} slots={slots} finalists={finalists} distinct_genomes={}",
            frozen.len(),
            semantics.len()
        );
    }
    Ok(frozen)
}

pub(super) fn validate_panel(registry: &Path, panel_id: &str) -> Result<()> {
    let frozen = load_validated_panel(registry, panel_id)?;
    println!(
        "validated {} independent freezes, 25200 evolutionary slots, 420 complete CV7 finalist receipts",
        frozen.len()
    );
    Ok(())
}

pub(super) fn load_baselines() -> Result<BTreeMap<String, BTreeMap<String, f64>>> {
    let path = manifest_dir().join("panel-analysis.v1.json");
    if hash_file(&path)? != PANEL_ANALYSIS_SHA256 {
        bail!("baseline panel SHA-256 mismatch");
    }
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    let rows = value["datasets"]
        .as_array()
        .context("baseline panel datasets are missing")?;
    let mut result = BTreeMap::new();
    for row in rows {
        let hash = row["dataset_hash"]
            .as_str()
            .context("baseline dataset hash is missing")?;
        if !DATASETS.contains(&hash) {
            continue;
        }
        let rmse = row["rmse"]
            .as_object()
            .context("baseline RMSE map is missing")?;
        let mut scores = BTreeMap::new();
        for model in BASELINES {
            let score = rmse
                .get(*model)
                .and_then(Value::as_f64)
                .with_context(|| format!("missing {model} RMSE for {hash}"))?;
            if !score.is_finite() || score < 0.0 {
                bail!("invalid {model} RMSE for {hash}");
            }
            scores.insert((*model).to_string(), score);
        }
        if result.insert(hash.to_string(), scores).is_some() {
            bail!("duplicate baseline row for {hash}");
        }
    }
    if result.len() != DATASETS.len() {
        bail!("baseline panel does not contain the exact pinned 42 datasets");
    }
    Ok(result)
}

pub(super) fn prediction_bytes(predictions: &[f64]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for prediction in predictions {
        if !prediction.is_finite() {
            bail!("cannot publish a non-finite prediction");
        }
        writeln!(bytes, "{prediction:.17}")?;
    }
    Ok(bytes)
}

pub(super) fn relation(left: f64, right: f64) -> &'static str {
    match left.total_cmp(&right) {
        Ordering::Less => "win",
        Ordering::Equal => "tie",
        Ordering::Greater => "loss",
    }
}

pub(super) fn baseline_comparisons(
    jope_rmse: f64,
    baselines: &BTreeMap<String, f64>,
) -> Result<(Value, f64, Vec<String>, String, String)> {
    let best = baselines.values().copied().fold(f64::INFINITY, f64::min);
    let best_models = baselines
        .iter()
        .filter_map(|(model, score)| (*score == best).then_some(model.clone()))
        .collect::<Vec<_>>();
    let mut comparisons = Map::new();
    for model in BASELINES {
        let baseline_rmse = baselines[*model];
        comparisons.insert(
            (*model).to_string(),
            json!({
                "rmse": baseline_rmse,
                "baseline_minus_jope_rmse": baseline_rmse - jope_rmse,
                "jope_outcome": relation(jope_rmse, baseline_rmse),
            }),
        );
    }
    Ok((
        Value::Object(comparisons),
        best,
        best_models,
        relation(jope_rmse, best).to_string(),
        relation(jope_rmse, baselines["TabFM ensemble"]).to_string(),
    ))
}

pub(super) fn score_dataset(
    registry: &Path,
    hash: &str,
    freeze: &ValidatedFreeze,
    baselines: &BTreeMap<String, f64>,
) -> Result<ScoredDataset> {
    // This function is called only after all 42 training-only freezes validate.
    let directory = registry.join(hash);
    let train_path = directory.join("train.csv");
    let test_path = directory.join("test.csv");
    let train = read_regression_csv(&train_path)?;
    let test = read_regression_csv(&test_path)?;
    if train.features != test.features {
        bail!("train/test feature mismatch for {hash}");
    }
    let timed = exact_cv::fit_predict_timed(&freeze.canonical_genome, &train, &test)?;
    let (rmse, r2, mae) = exact_cv::metrics(&test.y, &timed.values)?;
    let prediction_bytes = prediction_bytes(&timed.values)?;
    let prediction_sha256 = sha256_bytes(&prediction_bytes);
    let (comparisons, best_rmse, best_models, overall_outcome, tabfm_outcome) =
        baseline_comparisons(rmse, baselines)?;
    let test_sha256 = hash_file(&test_path)?;
    let report = json!({
        "schema": "JopeDimeDatasetReport.v1",
        "algorithm": {
            "name": MODEL,
            "dataset_specific": true,
            "canonical_genome": freeze.canonical_genome,
            "semantic_sha256": freeze.semantic_sha256,
            "complexity": freeze.complexity,
        },
        "dataset_hash": hash,
        "status": "completed",
        "train_rows": train.y.len(),
        "test_rows": test.y.len(),
        "n_features": train.features,
        "metrics": {"rmse": rmse, "r2": r2, "mae": mae},
        "timings": {
            "fit_seconds": timed.fit_seconds,
            "predict_seconds": timed.predict_seconds,
            "total_seconds": timed.fit_seconds + timed.predict_seconds,
        },
        "prediction": {
            "rows": timed.values.len(),
            "sha256": prediction_sha256,
        },
        "selection_proxy": {
            "authority_label": "CV7 training-only selection proxy; not final test authority",
            "generation": freeze.generation,
            "slot": freeze.slot,
            "ranking_authority": freeze.ranking_authority,
            "quick_cv5_fold0": {
                "pooled_oof_r2": freeze.quick_pooled_oof_r2,
                "pooled_oof_rmse": freeze.quick_pooled_oof_rmse,
                "robust_fitness": freeze.quick_robust_fitness,
                "fold_manifest_sha256": freeze.cv5_fold_manifest_sha256,
            },
            "full_cv7": {
                "fold_r2": freeze.cv7_fold_r2,
                "pooled_oof_r2": freeze.cv7_pooled_oof_r2,
                "pooled_oof_rmse": freeze.cv7_pooled_oof_rmse,
                "robust_fitness": freeze.cv7_robust_fitness,
                "fit_seconds": freeze.cv7_fit_seconds,
                "predict_seconds": freeze.cv7_predict_seconds,
                "fold_manifest_sha256": freeze.cv7_fold_manifest_sha256,
            },
        },
        "comparisons": {
            "baselines": comparisons,
            "best_baseline": {"models": best_models, "rmse": best_rmse},
            "strict_overall": {
                "rule": "win iff jope-dime test RMSE is strictly lower than the best baseline test RMSE",
                "outcome": overall_outcome,
                "margin_best_baseline_minus_jope_rmse": best_rmse - rmse,
            },
            "tabfm_head_to_head": {
                "outcome": tabfm_outcome,
                "tabfm_rmse": baselines["TabFM ensemble"],
                "tabfm_minus_jope_rmse": baselines["TabFM ensemble"] - rmse,
            },
        },
        "provenance": {
            "selection_sha256": freeze.selection_sha256,
            "ledger_sha256": freeze.ledger_sha256,
            "train_sha256": freeze.train_sha256,
            "test_sha256": test_sha256,
            "panel_text_sha256": PANEL_SHA256,
            "baseline_panel_sha256": PANEL_ANALYSIS_SHA256,
            "evolutionary_slots": freeze.evolutionary_slots,
            "cv7_finalist_receipts": freeze.finalist_receipts,
        },
    });
    let report_bytes = canonical_json(&report)?;
    let report_sha256 = sha256_bytes(&report_bytes);
    let mut scores = baselines.clone();
    scores.insert(MODEL.to_string(), rmse);
    let outcome = json!({
        "dataset_hash": hash,
        "genome_semantic_sha256": freeze.semantic_sha256,
        "test_metrics": {"rmse": rmse, "r2": r2, "mae": mae},
        "timings": {
            "fit_seconds": timed.fit_seconds,
            "predict_seconds": timed.predict_seconds,
        },
        "prediction_sha256": prediction_sha256,
        "report_sha256": report_sha256,
        "best_baseline_rmse": best_rmse,
        "best_baseline_models": best_models,
        "strict_overall_outcome": overall_outcome,
        "tabfm_head_to_head_outcome": tabfm_outcome,
        "rmse_by_algorithm": scores,
    });
    Ok(ScoredDataset {
        dataset_hash: hash.to_string(),
        predictions: timed.values,
        prediction_bytes,
        prediction_sha256,
        report_bytes,
        report_sha256,
        scores,
        outcome,
    })
}

pub(super) fn pairwise_results(scored: &[ScoredDataset]) -> Value {
    let players = players();
    let mut rows = Vec::new();
    for left_index in 0..players.len() {
        for right_index in left_index + 1..players.len() {
            let left = players[left_index];
            let right = players[right_index];
            let mut left_wins = 0usize;
            let mut right_wins = 0usize;
            let mut ties = 0usize;
            for dataset in scored {
                match dataset.scores[left].total_cmp(&dataset.scores[right]) {
                    Ordering::Less => left_wins += 1,
                    Ordering::Equal => ties += 1,
                    Ordering::Greater => right_wins += 1,
                }
            }
            rows.push(json!({
                "left": left,
                "right": right,
                "left_wins": left_wins,
                "right_wins": right_wins,
                "ties": ties,
            }));
        }
    }
    Value::Array(rows)
}

pub(super) fn players() -> Vec<&'static str> {
    let mut values = BASELINES.to_vec();
    values.push(MODEL);
    values
}

pub(super) fn elo_standings(scored: &[ScoredDataset]) -> Value {
    const K: f64 = 16.0;
    let players = players();
    let mut ratings = players
        .iter()
        .map(|player| ((*player).to_string(), 1500.0))
        .collect::<BTreeMap<_, _>>();
    for dataset in scored {
        let snapshot = ratings.clone();
        let mut deltas = players
            .iter()
            .map(|player| ((*player).to_string(), 0.0))
            .collect::<BTreeMap<_, _>>();
        for left_index in 0..players.len() {
            for right_index in left_index + 1..players.len() {
                let left = players[left_index];
                let right = players[right_index];
                let left_rating = snapshot[left];
                let right_rating = snapshot[right];
                let expected_left = 1.0 / (1.0 + 10_f64.powf((right_rating - left_rating) / 400.0));
                let actual_left = match dataset.scores[left].total_cmp(&dataset.scores[right]) {
                    Ordering::Less => 1.0,
                    Ordering::Equal => 0.5,
                    Ordering::Greater => 0.0,
                };
                let delta = K * (actual_left - expected_left);
                *deltas.get_mut(left).unwrap() += delta;
                *deltas.get_mut(right).unwrap() -= delta;
            }
        }
        for player in &players {
            *ratings.get_mut(*player).unwrap() += deltas[*player];
        }
    }
    let mut ordered = ratings.into_iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    Value::Array(
        ordered
            .into_iter()
            .enumerate()
            .map(|(index, (algorithm, rating))| {
                json!({"rank": index + 1, "algorithm": algorithm, "rating": rating})
            })
            .collect(),
    )
}

pub(super) fn publish_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("publication path has no parent")?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        bail!("refusing to overwrite {}", path.display());
    }
    let staged = parent.join(format!(
        ".jope-dime-{}-{}.stage",
        std::process::id(),
        &sha256_bytes(bytes)[..16]
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    let mut permissions = fs::metadata(&staged)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&staged, permissions)?;
    fs::hard_link(&staged, path).with_context(|| format!("publish {}", path.display()))?;
    fs::remove_file(staged)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

pub(super) fn ensure_publication_targets_absent(registry: &Path, panel_id: &str) -> Result<()> {
    let root = run_root(panel_id);
    let ledger = root.join("jope-dime-panel-results.v1.json");
    if ledger.exists() {
        bail!("refusing to overwrite {}", ledger.display());
    }
    for hash in DATASETS {
        for path in [
            root.join(hash).join("phase2/predictions.csv"),
            registry
                .join(hash)
                .join("predictions")
                .join(PREDICTION_FILE),
            registry.join(hash).join("reports").join(REPORT_FILE),
        ] {
            if path.exists() {
                bail!("refusing to overwrite {}", path.display());
            }
        }
    }
    Ok(())
}

pub(super) fn score_panel(registry: &Path, panel_id: &str) -> Result<()> {
    // The baseline manifest and every test.csv remain unopened until this
    // complete panel-wide training-only validation returns successfully.
    let frozen = load_validated_panel(registry, panel_id)?;
    let baselines = load_baselines()?;
    ensure_publication_targets_absent(registry, panel_id)?;

    let mut scored = Vec::with_capacity(DATASETS.len());
    for hash in DATASETS {
        eprintln!("[jope-dime] final scoring {hash}");
        scored.push(score_dataset(
            registry,
            hash,
            &frozen[*hash],
            &baselines[*hash],
        )?);
    }
    if scored.len() != 42 || scored.iter().any(|row| row.predictions.is_empty()) {
        bail!("final scoring did not produce exactly 42 complete outcomes");
    }

    let root = run_root(panel_id);
    for row in &scored {
        let local_prediction = root.join(&row.dataset_hash).join("phase2/predictions.csv");
        let public_prediction = registry
            .join(&row.dataset_hash)
            .join("predictions")
            .join(PREDICTION_FILE);
        let public_report = registry
            .join(&row.dataset_hash)
            .join("reports")
            .join(REPORT_FILE);
        publish_new(&local_prediction, &row.prediction_bytes)?;
        publish_new(&public_prediction, &row.prediction_bytes)?;
        publish_new(&public_report, &row.report_bytes)?;
        if hash_file(&local_prediction)? != row.prediction_sha256
            || hash_file(&public_prediction)? != row.prediction_sha256
            || hash_file(&public_report)? != row.report_sha256
        {
            bail!("published artifact hash mismatch for {}", row.dataset_hash);
        }
    }

    let wins = scored
        .iter()
        .filter(|row| row.outcome["strict_overall_outcome"] == "win")
        .count();
    let losses = scored
        .iter()
        .filter(|row| row.outcome["strict_overall_outcome"] == "loss")
        .count();
    let ties = 42 - wins - losses;
    let tabfm_wins = scored
        .iter()
        .filter(|row| row.outcome["tabfm_head_to_head_outcome"] == "win")
        .count();
    let tabfm_losses = scored
        .iter()
        .filter(|row| row.outcome["tabfm_head_to_head_outcome"] == "loss")
        .count();
    let tabfm_ties = 42 - tabfm_wins - tabfm_losses;
    let tabfm_status = match tabfm_wins.cmp(&tabfm_losses) {
        Ordering::Greater => "winning",
        Ordering::Equal => "tied",
        Ordering::Less => "losing",
    };
    let outcomes = scored
        .iter()
        .map(|row| row.outcome.clone())
        .collect::<Vec<_>>();
    let ledger = json!({
        "schema": "JopeDimePanelResults.v1",
        "panel_id": panel_id,
        "panel_text_sha256": PANEL_SHA256,
        "baseline_panel_sha256": PANEL_ANALYSIS_SHA256,
        "dataset_count": 42,
        "selection_accounting": {
            "independent_freezes": 42,
            "distinct_selected_semantic_genomes": 42,
            "evolutionary_slots": 25200,
            "full_cv7_finalist_receipts": 420,
        },
        "strict_overall": {
            "rule": "win iff jope-dime test RMSE is strictly lower than the best baseline test RMSE",
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "tabfm_head_to_head": {
            "wins": tabfm_wins,
            "losses": tabfm_losses,
            "ties": tabfm_ties,
            "status": tabfm_status,
        },
        "pairwise_results": pairwise_results(&scored),
        "elo": {
            "initial_rating": 1500.0,
            "k_factor": 16.0,
            "method": "Each dataset is a simultaneous 7-algorithm round robin; lower RMSE wins and exact equality ties; panel order is pinned.",
            "standings": elo_standings(&scored),
        },
        "outcomes": outcomes,
    });
    let ledger_path = root.join("jope-dime-panel-results.v1.json");
    publish_new(&ledger_path, &canonical_json(&ledger)?)?;
    println!(
        "published 42 reports and predictions: overall {wins}W/{losses}L/{ties}T; TabFM {tabfm_wins}W/{tabfm_losses}L/{tabfm_ties}T"
    );
    Ok(())
}

pub(super) fn summarize(registry: &Path, panel_id: &str) -> Result<()> {
    let path = run_root(panel_id).join("jope-dime-panel-results.v1.json");
    let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
    if value["schema"] == "JopeDimePanelResults.v2" {
        if panel_id != "grok4p5-42" {
            bail!("the v2 follow-up ledger is only valid at the original panel path");
        }
        return followup::summarize(registry);
    }
    if value["schema"] != "JopeDimePanelResults.v1"
        || value["dataset_count"].as_u64() != Some(42)
        || value["outcomes"].as_array().map(Vec::len) != Some(42)
        || value["pairwise_results"].as_array().map(Vec::len) != Some(21)
        || value["elo"]["standings"].as_array().map(Vec::len) != Some(7)
        || value["selection_accounting"]["evolutionary_slots"].as_u64() != Some(25_200)
        || value["selection_accounting"]["full_cv7_finalist_receipts"].as_u64() != Some(420)
    {
        bail!("panel ledger is incomplete or invalid");
    }
    for outcome in value["outcomes"].as_array().unwrap() {
        let hash = outcome["dataset_hash"]
            .as_str()
            .context("outcome dataset hash is missing")?;
        if !DATASETS.contains(&hash) {
            bail!("panel ledger contains an unknown dataset");
        }
        let prediction_sha256 = outcome["prediction_sha256"]
            .as_str()
            .context("outcome prediction hash is missing")?;
        let report_sha256 = outcome["report_sha256"]
            .as_str()
            .context("outcome report hash is missing")?;
        let local_prediction = run_root(panel_id).join(hash).join("phase2/predictions.csv");
        let public_prediction = registry
            .join(hash)
            .join("predictions")
            .join(PREDICTION_FILE);
        let public_report = registry.join(hash).join("reports").join(REPORT_FILE);
        if hash_file(&local_prediction)? != prediction_sha256
            || hash_file(&public_prediction)? != prediction_sha256
            || hash_file(&public_report)? != report_sha256
        {
            bail!("published artifact hash mismatch for {hash}");
        }
        let report: Value = serde_json::from_slice(&fs::read(public_report)?)?;
        if report["schema"] != "JopeDimeDatasetReport.v1"
            || report["dataset_hash"].as_str() != Some(hash)
            || report["status"] != "completed"
            || report["prediction"]["sha256"].as_str() != Some(prediction_sha256)
        {
            bail!("published report is invalid for {hash}");
        }
    }
    println!(
        "42 outcomes; strict overall {}W/{}L/{}T; TabFM {}W/{}L/{}T ({})",
        value["strict_overall"]["wins"],
        value["strict_overall"]["losses"],
        value["strict_overall"]["ties"],
        value["tabfm_head_to_head"]["wins"],
        value["tabfm_head_to_head"]["losses"],
        value["tabfm_head_to_head"]["ties"],
        value["tabfm_head_to_head"]["status"]
            .as_str()
            .unwrap_or("unknown"),
    );
    Ok(())
}
