fn bind_forecast3_comparators(specs: &mut [DatasetSpec], coverage: &CoverageBinding) -> Result<()> {
    for spec in specs {
        let authorized = BASELINE_MODELS
            .iter()
            .copied()
            .filter(|model| !declared_baseline_skip(&spec.dataset_hash, model))
            .collect::<BTreeSet<_>>();
        let path = Path::new(coverage_comparator_root_str())
            .join("timeseries")
            .join(&spec.dataset_hash)
            .join("comparator-lock.v2.json");
        let lock: Value = load_json(&path)?;
        if lock.get("schema").and_then(Value::as_str) != Some("test-adaptive-comparator-lock-v2")
            || lock.get("task").and_then(Value::as_str) != Some("timeseries")
            || lock.get("dataset_hash").and_then(Value::as_str) != Some(spec.dataset_hash.as_str())
            || lock.get("run_id").and_then(Value::as_str) != Some(FORECAST3_RUN_ID)
            || lock.get("coverage_campaign_id").and_then(Value::as_str)
                != Some("timeseries42-forecast3-v10")
            || lock.get("coverage_manifest_sha256").and_then(Value::as_str)
                != Some(TS_COVERAGE_MANIFEST_SHA256)
            || lock
                .get("coverage_final_receipt_sha256")
                .and_then(Value::as_str)
                != Some(coverage.timeseries_final_receipt_sha256.as_str())
            || lock
                .get("timeseries_baseline_receipt_sha256")
                .and_then(Value::as_str)
                != Some(coverage.timeseries_baseline_receipt_sha256.as_str())
            || lock.get("accepted_report_count").and_then(Value::as_u64)
                != Some(authorized.len() as u64)
        {
            bail!(
                "timeseries comparator lock identity drift at {}",
                path.display()
            );
        }
        let protocol_sha256 = lock
            .get("protocol_sha256")
            .and_then(Value::as_str)
            .context("timeseries comparator lacks protocol_sha256")?;
        let mut protocol = lock
            .get("protocol")
            .cloned()
            .context("timeseries comparator lacks protocol")?;
        let evaluator_sha256 = current_corpus_eval_sha256()?;
        if protocol.get("protocol_sha256").and_then(Value::as_str) != Some(protocol_sha256)
            || protocol
                .pointer("/evaluator/sha256")
                .and_then(Value::as_str)
                != Some(evaluator_sha256.as_str())
            || protocol.get("minimum_generations").and_then(Value::as_u64) != Some(0)
            || protocol.get("maximum_generations").and_then(Value::as_u64) != Some(20)
            || protocol.get("population").and_then(Value::as_u64) != Some(60)
        {
            bail!("timeseries comparator protocol drift at {}", path.display());
        }
        protocol
            .as_object_mut()
            .context("timeseries comparator protocol must be an object")?
            .remove("protocol_sha256");
        if sha256_bytes(&serde_json::to_vec(&protocol)?) != protocol_sha256 {
            bail!(
                "timeseries comparator protocol digest drift at {}",
                path.display()
            );
        }
        let files = lock
            .get("dataset_file_hashes")
            .and_then(Value::as_object)
            .context("timeseries comparator lacks dataset file hashes")?;
        if files.get("train.csv").and_then(Value::as_str) != Some(spec.train_sha256.as_str())
            || files.get("test.csv").and_then(Value::as_str) != Some(spec.test_sha256.as_str())
            || files.get("meta.json").and_then(Value::as_str) != Some(spec.meta_sha256.as_str())
        {
            bail!("timeseries comparator input drift at {}", path.display());
        }
        let accepted = lock
            .get("accepted_reports")
            .and_then(Value::as_array)
            .context("timeseries comparator lacks accepted reports")?;
        let mut reports = Vec::with_capacity(authorized.len());
        for row in accepted {
            let model = row
                .get("model")
                .and_then(Value::as_str)
                .context("timeseries comparator report lacks model")?;
            let report_path = row
                .get("report_path")
                .and_then(Value::as_str)
                .context("timeseries comparator report lacks path")?;
            let report_sha256 = row
                .get("report_sha256")
                .and_then(Value::as_str)
                .context("timeseries comparator report lacks digest")?;
            let mase = row
                .get("primary")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite());
            let rmse = row
                .get("secondary")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite());
            if !authorized.contains(model)
                || sha256_file(Path::new(report_path))? != report_sha256
                || mase.is_none()
                || rmse.is_none()
            {
                bail!("invalid authorized comparator report at {}", path.display());
            }
            reports.push(ExternalReport {
                model: model.to_string(),
                report_path: report_path.to_string(),
                report_sha256: report_sha256.to_string(),
                mase: mase.unwrap(),
                rmse: rmse.unwrap(),
            });
        }
        if reports.len() != authorized.len()
            || reports
                .iter()
                .map(|row| row.model.as_str())
                .collect::<BTreeSet<_>>()
                != authorized
        {
            bail!(
                "timeseries comparator report membership drift at {}",
                path.display()
            );
        }
        reports.sort_by(|left, right| {
            left.mase
                .total_cmp(&right.mase)
                .then(left.rmse.total_cmp(&right.rmse))
                .then(left.model.cmp(&right.model))
                .then(left.report_sha256.cmp(&right.report_sha256))
        });
        spec.official_leader = official_leader(&reports, Path::new(&spec.train_path))?;
        spec.accepted_external_reports = reports;
        spec.comparator_lock_sha256 = embedded_payload_sha256(&lock, "lock_sha256")?;
    }
    Ok(())
}

fn declared_baseline_skip(dataset_hash: &str, model: &str) -> bool {
    matches!(
        (dataset_hash, model),
        ("080d2674945eb5d7", "autogluon_timeseries")
            | ("c400e4876a9046ae", "autogluon_timeseries")
            | ("c43cfcfc9313d051", "autogluon_timeseries")
            | ("080d2674945eb5d7", "sktime_reduction_lightgbm")
            | ("c43cfcfc9313d051", "sktime_reduction_lightgbm")
    )
}

fn verify_allmodels_baseline(
    dataset_dir: &Path,
    dataset_hash: &str,
    accepted: &[ExternalReport],
) -> Result<()> {
    let accepted_models = accepted
        .iter()
        .map(|row| row.model.as_str())
        .collect::<BTreeSet<_>>();
    for model in BASELINE_MODELS {
        let path = dataset_dir.join("reports").join(format!("{model}.json"));
        let value: Value = serde_json::from_slice(
            &fs::read(&path)
                .with_context(|| format!("missing baseline report {}", path.display()))?,
        )
        .with_context(|| format!("parse baseline report {}", path.display()))?;
        if value.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
            || value.get("task_type").and_then(Value::as_str) != Some("timeseries")
            || value.get("test_adaptive").and_then(Value::as_bool) == Some(true)
        {
            bail!("baseline report identity drift at {}", path.display());
        }
        let status = value.get("status").and_then(Value::as_str).unwrap_or("");
        if declared_baseline_skip(dataset_hash, model) {
            if status != "skipped-ineligible" {
                bail!("declared eligibility terminal drift at {}", path.display());
            }
        } else if !matches!(status, "completed" | "completed-valid")
            || finite_metric(&value, "mase").is_none()
            || finite_metric(&value, "rmse").is_none()
            || !accepted_models.contains(model)
        {
            bail!("numeric baseline is incomplete at {}", path.display());
        }
    }
    let expected_numeric = BASELINE_MODELS
        .iter()
        .filter(|model| !declared_baseline_skip(dataset_hash, model))
        .copied()
        .collect::<BTreeSet<_>>();
    if accepted_models != expected_numeric {
        bail!("baseline comparator envelope drift for {dataset_hash}");
    }
    Ok(())
}

fn load_dataset_specs(require_allmodels_baseline: bool) -> Result<Vec<DatasetSpec>> {
    let registry = Path::new(registry_str());
    let mut dirs = fs::read_dir(registry)
        .with_context(|| format!("read {registry_str}", registry_str = registry_str()))?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    dirs.retain(|path| path.is_dir());
    dirs.sort();
    if dirs.len() != 42 {
        bail!(
            "TimeSeries42 expected 42 dataset directories, found {}",
            dirs.len()
        );
    }
    let mut specs = Vec::with_capacity(dirs.len());
    for dir in dirs {
        let dataset_hash = dir
            .file_name()
            .and_then(|name| name.to_str())
            .context("time-series dataset path has no UTF-8 name")?
            .to_string();
        let train_path = dir.join("train.csv");
        let test_path = dir.join("test.csv");
        let meta_path = dir.join("meta.json");
        for path in [&train_path, &test_path, &meta_path] {
            if !path.is_file() {
                bail!(
                    "TimeSeries42 dataset {dataset_hash} is missing {}",
                    path.display()
                );
            }
        }
        let meta_bytes = fs::read(&meta_path)?;
        let meta: Value = serde_json::from_slice(&meta_bytes)
            .with_context(|| format!("parse {}", meta_path.display()))?;
        let horizon = required_usize(&meta, "horizon")?;
        let seasonality = required_usize(&meta, "seasonality")?;
        let frequency = meta
            .get("frequency")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .context("meta_json missing frequency")?
            .to_string();
        let target_is_final_column = meta
            .get("target_is_final_column")
            .and_then(Value::as_bool)
            .context("meta_json missing target_is_final_column")?;
        let known_future_covariates = string_array(&meta, "known_dynamic_columns")?;
        let past_only_covariates = string_array(&meta, "past_dynamic_columns_excluded")?;
        let manifest_spec = TimeSeriesManifestSpec {
            horizon,
            seasonality,
            frequency: frequency.clone(),
            target_is_final_column,
            known_future_covariates: known_future_covariates.clone(),
            past_only_covariates: Vec::new(),
        };
        let train = read_timeseries_csv(&train_path)?;
        let test = read_timeseries_csv(&test_path)?;
        let train_rows = train.y.len();
        let test_rows = test.y.len();
        let csv_features = train.features;
        let data = validate_timeseries_data(train, test, &meta, &manifest_spec)
            .with_context(|| format!("validate time-series data for {dataset_hash}"))?;
        let reduction_features = make_supervised_reduction(&data)?.feature_names.len();
        let accepted_external_reports = external_reports(&dir, &dataset_hash)?;
        if require_allmodels_baseline {
            verify_allmodels_baseline(&dir, &dataset_hash, &accepted_external_reports)?;
        }
        let official_leader = official_leader(&accepted_external_reports, &dir)?;
        specs.push(DatasetSpec {
            schema: DATASET_SCHEMA.to_string(),
            dataset_hash,
            display_name: meta
                .get("display_name")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            category: meta
                .get("category")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            train_path: train_path.to_string_lossy().into_owned(),
            train_sha256: sha256_file(&train_path)?,
            test_path: test_path.to_string_lossy().into_owned(),
            test_sha256: sha256_file(&test_path)?,
            meta_path: meta_path.to_string_lossy().into_owned(),
            meta_sha256: sha256_bytes(&meta_bytes),
            train_rows,
            test_rows,
            csv_features,
            reduction_features,
            horizon,
            seasonality,
            frequency,
            target_is_final_column,
            known_future_covariates,
            past_only_covariates,
            source_real_data: meta
                .get("source_real_data")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            synthetic_origin: meta
                .get("synthetic_origin")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            reduction_version: String::new(),
            accepted_external_reports,
            comparator_lock_sha256: String::new(),
            official_leader,
        });
    }
    Ok(specs)
}

fn ordered_ids(specs: &[DatasetSpec]) -> Vec<String> {
    specs.iter().map(|spec| spec.dataset_hash.clone()).collect()
}
