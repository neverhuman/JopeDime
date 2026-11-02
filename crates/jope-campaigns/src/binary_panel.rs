use super::*;

pub(super) fn binary_dataset_seed(hash: &str) -> Result<u64> {
    Ok(BASE_SEED
        ^ BINARY_LANE_TAG
        ^ u64::from_str_radix(hash, 16).context("dataset hash is not hexadecimal")?)
}

pub(super) fn binary_seed_bytes() -> &'static [u8] {
    include_bytes!("../seeds/generation-zero-binary.txt")
}

/// Pinned binary panel: manifest sha + seed sha verified before any campaign.
pub(super) fn binary_panel() -> Result<Vec<String>> {
    if sha256_bytes(binary_seed_bytes()) != commands::binary_cv::SEED_BINARY_SHA256 {
        bail!("binary seed population hash mismatch");
    }
    let path = manifest_dir().join("panel-binary.txt");
    if hash_file(&path)? != PANEL_BINARY_SHA256 {
        bail!("panel-binary.txt SHA-256 mismatch");
    }
    let text = fs::read_to_string(&path)?;
    let hashes = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect::<Vec<_>>();
    if hashes.is_empty() {
        bail!("panel-binary.txt lists no datasets");
    }
    Ok(hashes)
}

/// Baseline AUC table from the pinned binary panel analysis.
pub(super) fn binary_baselines() -> Result<BTreeMap<String, BTreeMap<String, f64>>> {
    let path = manifest_dir().join("panel-analysis-binary.v1.json");
    if hash_file(&path)? != PANEL_ANALYSIS_BINARY_SHA256 {
        bail!("panel-analysis-binary.v1.json SHA-256 mismatch");
    }
    let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let mut table = BTreeMap::new();
    for entry in value["datasets"].as_array().context("datasets array")? {
        let hash = entry["dataset_hash"]
            .as_str()
            .context("dataset_hash")?
            .to_string();
        let mut aucs = BTreeMap::new();
        for (model, score) in entry["auc"].as_object().context("auc table")? {
            aucs.insert(model.clone(), score.as_f64().context("auc value")?);
        }
        table.insert(hash, aucs);
    }
    Ok(table)
}

pub(super) fn verify_inputs_binary(registry: &Path) -> Result<()> {
    for hash in binary_panel()? {
        // Training-only firewall: test.csv is never opened here.
        let train_path = registry.join(&hash).join("train.csv");
        let train = read_binary_csv(&train_path)?;
        let positives = train.y.iter().filter(|v| **v == 1.0).count();
        println!(
            "{hash} train_rows={} features={} positives={} negatives={} train_sha256={}",
            train.y.len(),
            train.features,
            positives,
            train.y.len() - positives,
            hash_file(&train_path)?
        );
    }
    println!("verified binary selection inputs without opening test data or baseline metrics");
    Ok(())
}

pub(super) fn select_panel_binary(
    registry: &Path,
    panel_id: &str,
    only: Option<&str>,
) -> Result<()> {
    let panel = binary_panel()?;
    if only.is_some_and(|wanted| !panel.iter().any(|hash| hash == wanted)) {
        bail!("requested dataset is absent from the pinned binary panel");
    }
    let root = run_root(panel_id);
    fs::create_dir_all(&root)?;
    let started = Instant::now();
    for (index, hash) in panel.iter().enumerate() {
        if only.is_some_and(|wanted| wanted != hash.as_str()) {
            continue;
        }
        let train_path = registry.join(hash).join("train.csv");
        let train = read_binary_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let artifact = root.join(hash);
        if artifact.join("SELECTION_COMPLETE").exists() {
            commands::binary_cv::validate_freeze_binary(
                &artifact,
                hash,
                &train_sha256,
                train.features,
                ORIGINAL_GENERATIONS,
            )
            .with_context(|| format!("existing binary freeze is invalid for {hash}"))?;
            continue;
        }
        if artifact.exists() && fs::read_dir(&artifact)?.next().transpose()?.is_some() {
            bail!(
                "incomplete nonempty binary campaign directory requires operator recovery: {}",
                artifact.display()
            );
        }
        eprintln!(
            "[jope-dime] selecting binary {hash} ({}/{})",
            index + 1,
            panel.len()
        );
        commands::binary_cv::select_dataset_binary(
            &train,
            hash,
            &train_sha256,
            binary_dataset_seed(hash)?,
            binary_seed_bytes(),
            &artifact,
            ORIGINAL_GENERATIONS,
        )?;
        commands::binary_cv::validate_freeze_binary(
            &artifact,
            hash,
            &train_sha256,
            train.features,
            ORIGINAL_GENERATIONS,
        )?;
    }
    println!(
        "binary selection pass complete in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

pub(super) fn validate_panel_binary(registry: &Path, panel_id: &str) -> Result<()> {
    let root = run_root(panel_id);
    for hash in binary_panel()? {
        // Training-only firewall: no test path or baseline metric is named.
        let train_path = registry.join(&hash).join("train.csv");
        let train = read_binary_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        commands::binary_cv::validate_freeze_binary(
            &root.join(&hash),
            &hash,
            &train_sha256,
            train.features,
            ORIGINAL_GENERATIONS,
        )
        .with_context(|| format!("binary freeze invalid for {hash}"))?;
        println!("{hash} binary freeze valid");
    }
    println!("validated binary panel without opening test data or baseline metrics");
    Ok(())
}

/// Round-robin Elo over per-dataset AUC tables: strictly greater AUC wins,
/// exact ties draw; K=16, initial 1500, datasets in pinned order and pairs in
/// lexicographic model order (fully deterministic).
pub(super) fn binary_elo(per_dataset: &[BTreeMap<String, f64>]) -> BTreeMap<String, f64> {
    let mut ratings = BTreeMap::<String, f64>::new();
    for table in per_dataset {
        for model in table.keys() {
            ratings.entry(model.clone()).or_insert(1500.0);
        }
    }
    for table in per_dataset {
        let models = table.keys().cloned().collect::<Vec<_>>();
        for i in 0..models.len() {
            for j in (i + 1)..models.len() {
                let (a, b) = (&models[i], &models[j]);
                let (score_a, score_b) = (table[a], table[b]);
                let outcome_a = if score_a > score_b {
                    1.0
                } else if score_b > score_a {
                    0.0
                } else {
                    0.5
                };
                let rating_a = ratings[a];
                let rating_b = ratings[b];
                let expected_a = 1.0 / (1.0 + 10f64.powf((rating_b - rating_a) / 400.0));
                *ratings.get_mut(a).unwrap() += 16.0 * (outcome_a - expected_a);
                *ratings.get_mut(b).unwrap() += 16.0 * ((1.0 - outcome_a) - (1.0 - expected_a));
            }
        }
    }
    ratings
}

pub(super) fn score_panel_binary(registry: &Path, panel_id: &str) -> Result<()> {
    let panel = binary_panel()?;
    let baselines = binary_baselines()?;
    let root = run_root(panel_id);
    let mut per_dataset = Vec::new();
    let mut outcomes = Vec::new();
    for hash in &panel {
        let train_path = registry.join(hash).join("train.csv");
        let test_path = registry.join(hash).join("test.csv");
        let train = read_binary_csv(&train_path)?;
        let test = read_binary_csv(&test_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let score = commands::binary_cv::score_dataset_binary(
            &root.join(hash),
            hash,
            &train,
            &train_sha256,
            &test,
            ORIGINAL_GENERATIONS,
        )?;
        let dataset_baselines = baselines
            .get(hash)
            .with_context(|| format!("no baseline AUCs for {hash}"))?;
        let mut table = dataset_baselines.clone();
        table.insert(MODEL_BINARY.to_string(), score.auc);
        let best_baseline = dataset_baselines
            .values()
            .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let strict_win = score.auc > best_baseline;
        let mut report = serde_json::to_value(&score)?;
        let object = report.as_object_mut().context("report object")?;
        object.remove("predictions");
        object.insert(
            "schema".to_string(),
            json!("JopeDimeBinaryDatasetReport.v1"),
        );
        object.insert("model".to_string(), json!(MODEL_BINARY));
        object.insert(
            "baseline_auc".to_string(),
            serde_json::to_value(dataset_baselines)?,
        );
        object.insert("strict_win".to_string(), json!(strict_win));
        let report_bytes = canonical_json(&report)?;
        let report_path = root.join(hash).join(BINARY_REPORT_FILE);
        if !report_path.exists() {
            fs::write(&report_path, &report_bytes)?;
        }
        let mut predictions_text = String::new();
        for value in &score.predictions {
            predictions_text.push_str(&format!("{value:?}\n"));
        }
        let prediction_path = root.join(hash).join(BINARY_PREDICTION_FILE);
        if !prediction_path.exists() {
            fs::write(&prediction_path, predictions_text)?;
        }
        println!(
            "{hash} auc={:.6} best_baseline={best_baseline:.6} strict_win={strict_win}",
            score.auc
        );
        outcomes.push(json!({
            "dataset_hash": hash,
            "auc": score.auc,
            "best_baseline_auc": best_baseline,
            "strict_win": strict_win,
        }));
        per_dataset.push(table);
    }
    let ratings = binary_elo(&per_dataset);
    let wins = outcomes
        .iter()
        .filter(|o| o["strict_win"].as_bool() == Some(true))
        .count();
    let results = json!({
        "schema": "JopeDimeBinaryPanelResults.v1",
        "panel_id": panel_id,
        "model": MODEL_BINARY,
        "authority": "test AUC, strictly greater wins; exact ties draw",
        "strict_wins": wins,
        "strict_losses": outcomes.len() - wins,
        "outcomes": outcomes,
        "elo": ratings,
    });
    let results_path = root.join(BINARY_RESULTS_FILE);
    fs::write(&results_path, canonical_json(&results)?)?;
    println!(
        "binary panel scored: {wins}/{} strict wins; results at {}",
        panel.len(),
        results_path.display()
    );
    Ok(())
}
