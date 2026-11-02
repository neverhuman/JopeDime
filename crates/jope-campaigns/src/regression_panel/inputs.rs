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
