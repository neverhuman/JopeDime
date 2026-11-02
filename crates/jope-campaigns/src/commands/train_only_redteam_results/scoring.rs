use super::*;

fn finite_rmse(value: &Value, pointer: &str) -> Result<f64> {
    let value = value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .with_context(|| format!("missing RMSE at {pointer}"))?;
    if !value.is_finite() || value < 0.0 {
        bail!("invalid RMSE at {pointer}");
    }
    Ok(value)
}

fn sum_squared_error(truth: &[f64], predictions: &[f64]) -> Result<f64> {
    if truth.len() != predictions.len()
        || truth.is_empty()
        || truth
            .iter()
            .chain(predictions)
            .any(|value| !value.is_finite())
    {
        bail!("score vectors are invalid");
    }
    Ok(truth
        .iter()
        .zip(predictions)
        .map(|(truth, prediction)| (truth - prediction).powi(2))
        .sum())
}

fn baseline_row(
    entry: &Value,
    dataset_hash: &str,
    oracle: &Value,
    oracle_source_sha256: String,
) -> Result<BaselineRow> {
    if entry.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash) {
        bail!("baseline dataset identity drift");
    }
    let stronger_method = value_string(entry, "/stronger_tabfm_algorithm")?;
    let stronger_path = if stronger_method == "tabfm_regressor" {
        "/tabfm_report_path"
    } else if stronger_method == "tabfm_ensemble_regressor" {
        "/tabfm_ensemble_report_path"
    } else {
        bail!("stronger TabFM identity drift");
    };
    let source = |path_pointer: &str, hash_pointer: &str| -> Result<String> {
        let path =
            execution::relocated_input_path(&PathBuf::from(value_string(entry, path_pointer)?));
        let expected = value_string(entry, hash_pointer)?;
        if sha256_file(&path)? != expected {
            bail!("authoritative baseline source drift");
        }
        Ok(expected)
    };
    if oracle.get("schema").and_then(Value::as_str)
        != Some("jope-dime.combined-regression-dataset-index.v1")
        || oracle.get("dataset").and_then(Value::as_str) != Some(dataset_hash)
        || oracle.get("cohort").and_then(Value::as_str) != Some("golden42")
    {
        bail!("ASPFM oracle source identity drift");
    }
    Ok(BaselineRow {
        prior_train_only_rmse: finite_rmse(entry, "/prior_aspfm_rmse")?,
        tabpfn_v3_rmse: finite_rmse(entry, "/tabpfn_v3_rmse")?,
        stronger_tabfm_method_id: stronger_method,
        stronger_tabfm_rmse: finite_rmse(entry, "/stronger_tabfm_rmse")?,
        oracle_aspfm_rmse: finite_rmse(oracle, "/best_rmse")?,
        prior_train_only_source_sha256: source(
            "/prior_aspfm_report_path",
            "/prior_aspfm_report_sha256",
        )?,
        tabpfn_v3_source_sha256: source("/tabpfn_v3_report_path", "/tabpfn_v3_report_sha256")?,
        stronger_tabfm_source_sha256: if stronger_path == "/tabfm_report_path" {
            source("/tabfm_report_path", "/tabfm_report_sha256")?
        } else {
            source(
                "/tabfm_ensemble_report_path",
                "/tabfm_ensemble_report_sha256",
            )?
        },
        oracle_source_sha256,
    })
}

fn predict_variant(
    variant: &execution::FrozenVariant,
    train: &RegressionData,
    test: &RegressionData,
) -> Result<Vec<f64>> {
    if variant.candidates.is_empty()
        || variant.candidates.len() != variant.weights.len()
        || variant
            .weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || (variant.weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
    {
        bail!("frozen full-data variant violates the convex contract");
    }
    let mut predictions = vec![0.0; test.y.len()];
    for (candidate, weight) in variant.candidates.iter().zip(&variant.weights) {
        let parsed = GenomeIr::parse(&candidate.canonical_genome)?;
        let first = fit(&parsed, &train.x, &train.y)?.predict(&test.x)?;
        let replay = fit(&parsed, &train.x, &train.y)?.predict(&test.x)?;
        if first != replay {
            bail!("full-data refit parity failure");
        }
        for (prediction, value) in predictions.iter_mut().zip(first) {
            *prediction += weight * value;
        }
    }
    Ok(predictions)
}

fn score_variant(
    name: &str,
    variant: &execution::FrozenVariant,
    train: &RegressionData,
    test: &RegressionData,
) -> Result<FrozenTestScore> {
    let predictions = predict_variant(variant, train, test)?;
    let rmse = (sum_squared_error(&test.y, &predictions)? / test.y.len() as f64).sqrt();
    Ok(FrozenTestScore {
        variant: name.to_string(),
        semantic_sha256: variant
            .candidates
            .iter()
            .map(|candidate| candidate.semantic_sha256.clone())
            .collect(),
        weights: variant.weights.clone(),
        rmse,
        predictions_sha256: receipt_digest(&predictions)?,
        full_data_refit_parity: true,
        development_only: true,
        confirmatory_valid: false,
    })
}

fn frozen_variants(
    dataset: &DatasetInput,
    g20: &execution::G20Complete,
    selector: &execution::SelectorComplete,
) -> Result<Vec<(&'static str, execution::FrozenVariant)>> {
    let (parent_checkpoint, _): (Value, _) = read_receipt(&dataset.parent_generation_paths[9])?;
    let parent_champion = parent_checkpoint
        .get("population")
        .and_then(Value::as_array)
        .and_then(|population| population.first())
        .context("parent generation-nine champion is absent")?;
    let gen9 = execution::candidate_from_parent(parent_champion)?;
    let g20_oof = g20
        .top_archive_search_oof
        .first()
        .context("G20 OOF archive is empty")?
        .candidate
        .clone();
    let single = |name: &str, candidate: execution::Candidate| execution::FrozenVariant {
        name: name.to_string(),
        candidates: vec![candidate],
        weights: vec![1.0],
    };
    Ok(vec![
        (GEN9_VARIANT, single(GEN9_VARIANT, gen9)),
        (
            G20_RAW_VARIANT,
            single(G20_RAW_VARIANT, g20.raw_generation_19.clone()),
        ),
        (G20_OOF_VARIANT, single(G20_OOF_VARIANT, g20_oof)),
        (
            SMART_SINGLE_VARIANT,
            single(SMART_SINGLE_VARIANT, selector.smart_single.clone()),
        ),
        (
            SMART_NO_JITTER_VARIANT,
            single(SMART_NO_JITTER_VARIANT, selector.no_jitter.clone()),
        ),
        (
            SMART_ORIGINAL_SEED_VARIANT,
            single(
                SMART_ORIGINAL_SEED_VARIANT,
                selector.original_decoder_seed_only.clone(),
            ),
        ),
        (FINAL_VARIANT, selector.final_variant.clone()),
    ])
}

pub(super) fn score_run(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be positive");
    }
    // This is the only gate above any construction of a test or baseline path.
    let (root, manifest, _, validation, validation_digest) =
        require_global_validation(study_root, run_id)?;

    let baseline_path = manifest
        .parent
        .manifest_path
        .parent()
        .context("parent manifest has no run root")?
        .join("baselines.v2.json");
    let (baseline_manifest, baseline_digest): (Value, _) = read_receipt(&baseline_path)?;
    if baseline_manifest.get("schema").and_then(Value::as_str)
        != Some("TrainOnlyBaselineManifest.v2")
        || baseline_manifest
            .get("entries")
            .and_then(Value::as_array)
            .is_none_or(|entries| entries.len() != manifest.datasets.len())
    {
        bail!("authoritative baseline manifest drift");
    }
    let oracle_index_path =
        sotarena_root.join("results/tabular-regression-v1/jope-dime-evolution/index.v1.json");
    let oracle_index_sha256 = sha256_file(&oracle_index_path)?;
    let oracle_index: Value = serde_json::from_slice(&fs::read(&oracle_index_path)?)?;
    if oracle_index.get("schema").and_then(Value::as_str)
        != Some("jope-dime.combined-regression-index.v1")
        || oracle_index
            .get("evaluation_records")
            .and_then(Value::as_u64)
            != Some(63_108)
        || oracle_index.get("indexed_datasets").and_then(Value::as_u64) != Some(42)
    {
        bail!("authoritative ASPFM oracle index drift");
    }

    let mut score_sha256 = BTreeMap::new();
    for dataset in &manifest.datasets {
        let selector_path = selector_path(&root, &dataset.dataset_hash);
        let (selector, selector_digest): (execution::SelectorComplete, _) =
            read_receipt(&selector_path)?;
        let shard_path = validation_shard_path(&root, &dataset.dataset_hash);
        let (shard, shard_digest): (DatasetValidationShard, _) = read_receipt(&shard_path)?;
        if validation
            .dataset_validation_sha256
            .get(&dataset.dataset_hash)
            != Some(&shard_digest)
            || shard.replay.selector_sha256 != selector_digest
            || !selector.selection_frozen_before_test_access
        {
            bail!("selector is not covered by the global validation gate");
        }
        let g20_path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("g20-extension-complete.v1.json");
        let (g20, g20_digest): (execution::G20Complete, _) = read_receipt(&g20_path)?;
        if g20_digest != shard.replay.g20_complete_sha256 {
            bail!("G20 receipt is not covered by the validation gate");
        }
        let baseline = baseline_manifest
            .get("entries")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .find(|entry| {
                entry.get("dataset_hash").and_then(Value::as_str) == Some(&dataset.dataset_hash)
            })
            .context("authoritative baseline row is absent")?;
        let oracle = oracle_index
            .get("datasets")
            .and_then(Value::as_array)
            .context("ASPFM oracle index lacks datasets")?
            .iter()
            .find(|entry| {
                entry.get("dataset").and_then(Value::as_str) == Some(&dataset.dataset_hash)
            })
            .context("ASPFM oracle row is absent")?;
        let oracle_source_sha256 = receipt_digest(oracle)?;
        let baselines = baseline_row(
            baseline,
            &dataset.dataset_hash,
            oracle,
            oracle_source_sha256,
        )?;

        let train = read_regression_csv(&dataset.train_path)?;
        let test_path = sotarena_root
            .join("data/tabular-regression-v1")
            .join(&dataset.dataset_hash)
            .join("test.csv");
        let test_sha256 = sha256_file(&test_path)?;
        let test = read_regression_csv(&test_path)?;
        if train.features != dataset.features || test.features != dataset.features {
            bail!("train/test feature shape drift");
        }
        let variants = frozen_variants(dataset, &g20, &selector)?
            .into_iter()
            .map(|(name, variant)| score_variant(name, &variant, &train, &test))
            .collect::<Result<Vec<_>>>()?;
        if variants.len() != 7
            || variants.last().map(|variant| variant.variant.as_str()) != Some(FINAL_VARIANT)
        {
            bail!("controlled ablation ledger drift");
        }
        let receipt = DatasetScoreReceipt {
            schema: "TrainOnlyRedteamDatasetScore.v1".to_string(),
            run_id: run_id.to_string(),
            dataset_hash: dataset.dataset_hash.clone(),
            validation_sha256: validation_digest.clone(),
            selector_sha256: selector_digest,
            train_sha256: dataset.train_sha256.clone(),
            test_sha256,
            baseline_manifest_sha256: baseline_digest.clone(),
            oracle_index_sha256: oracle_index_sha256.clone(),
            variants,
            baselines,
            selection_frozen_before_test_access: true,
            development_only: true,
            confirmatory_valid: false,
            test_reuse_disclosure: DISCLOSURE.to_string(),
        };
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("scores.v1.json");
        let digest = write_receipt(&path, &receipt)?;
        score_sha256.insert(dataset.dataset_hash.clone(), digest);
    }
    let receipt = ScoreIndex {
        schema: "TrainOnlyRedteamScoreIndex.v1".to_string(),
        run_id: run_id.to_string(),
        validation_sha256: validation_digest,
        baseline_manifest_sha256: baseline_digest,
        oracle_index_sha256,
        dataset_score_sha256: score_sha256,
        development_only: true,
        confirmatory_valid: false,
        test_reuse_disclosure: DISCLOSURE.to_string(),
    };
    let path = root.join("scores.v1.json");
    write_receipt(&path, &receipt)?;
    Ok(path)
}
