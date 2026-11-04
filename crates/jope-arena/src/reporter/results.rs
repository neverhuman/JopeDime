use super::*;

/// Read every `data/<task>/<hash>/dataset.json`.
pub fn load_datasets(root: &Path, config: &Config) -> Result<DatasetCatalog> {
    let mut catalog: DatasetCatalog = config
        .tasks
        .iter()
        .map(|task| (task.id.clone(), BTreeMap::new()))
        .collect();
    for task in &config.tasks {
        let directory = root.join("data").join(&task.id);
        let entries = fs::read_dir(&directory)
            .with_context(|| format!("cannot read task directory {}", directory.display()))?;
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_dir() || entry.file_name() == OsStr::new("global") {
                continue;
            }
            let path = entry.path().join("dataset.json");
            let dataset: Dataset = read_json(&path)?;
            ensure!(
                dataset.schema == "sotarena.dataset",
                "invalid dataset schema"
            );
            ensure!(
                dataset.task == task.id && dataset.metric == task.metric,
                "dataset task/metric mismatch"
            );
            ensure!(valid_hash(&dataset.dataset_hash), "invalid dataset hash");
            ensure!(
                entry.file_name() == OsStr::new(&dataset.dataset_hash),
                "dataset directory/hash mismatch"
            );
            check_data_file(&dataset.train)?;
            check_data_file(&dataset.test)?;
            ensure!(
                dataset.train.url.is_some() == dataset.test.url.is_some(),
                "dataset train/test download availability mismatch"
            );
            check_dataset_rows(&dataset.train, &dataset.test)?;
            ensure!(
                catalog
                    .get_mut(&task.id)
                    .expect("task inserted above")
                    .insert(dataset.dataset_hash.clone(), dataset)
                    .is_none(),
                "duplicate dataset hash"
            );
        }
        ensure!(
            !catalog[&task.id].is_empty(),
            "task {} has no datasets",
            task.id
        );
    }
    Ok(catalog)
}

/// Read and validate every task's analysis blacklist.
pub fn load_analysis_blacklists(
    root: &Path,
    config: &Config,
    datasets: &DatasetCatalog,
) -> Result<AnalysisBlacklists> {
    let mut blacklists = BTreeMap::new();
    for task in &config.tasks {
        let path = root.join(&task.analysis_blacklist);
        let blacklist: AnalysisBlacklist = read_json(&path)?;
        ensure!(
            blacklist.schema == "sotarena.analysis-blacklist",
            "invalid analysis blacklist schema for {}",
            task.id
        );
        ensure!(
            blacklist.task == task.id,
            "analysis blacklist task mismatch for {}",
            task.id
        );
        ensure!(
            blacklist.datasets.windows(2).all(|pair| pair[0] < pair[1]),
            "analysis blacklist datasets for {} must be sorted and unique",
            task.id
        );
        for hash in &blacklist.datasets {
            ensure!(
                datasets[&task.id].contains_key(hash),
                "unknown dataset {hash} in analysis blacklist for {}",
                task.id
            );
        }
        blacklists.insert(task.id.clone(), blacklist.datasets.into_iter().collect());
    }
    Ok(blacklists)
}

fn dataset_hash<'a>(
    datasets: &'a BTreeMap<String, Dataset>,
    reference: &'a str,
) -> Result<&'a str> {
    if datasets.contains_key(reference) {
        return Ok(reference);
    }
    datasets
        .iter()
        .find(|(_, dataset)| dataset.dataset_id == reference)
        .map(|(hash, _)| hash.as_str())
        .with_context(|| format!("unknown dataset {reference}"))
}

fn json_files(directory: &Path) -> Result<Vec<PathBuf>> {
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    Ok(fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && path.extension().and_then(OsStr::to_str) == Some("json"))
        .collect())
}

fn result_paths(
    root: &Path,
    task: &str,
    datasets: &BTreeMap<String, Dataset>,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for hash in datasets.keys() {
        paths.extend(json_files(
            &root.join("data").join(task).join(hash).join("results"),
        )?);
    }
    paths.sort();
    Ok(paths)
}

fn global_result_paths(root: &Path, config: &Config) -> Result<Vec<PathBuf>> {
    let mut paths = json_files(&root.join("data/global/results"))?;
    for task in &config.tasks {
        paths.extend(json_files(
            &root.join("data").join(&task.id).join("global/results"),
        )?);
    }
    paths.sort();
    Ok(paths)
}

/// Non-empty, no control characters.
pub fn valid_method_id(method: &str) -> bool {
    !method.trim().is_empty() && !method.chars().any(char::is_control)
}

/// Finite and inside the metric's range.
pub fn valid_metric_value(metric: &str, value: f64) -> bool {
    value.is_finite()
        && match metric {
            "roc_auc" | "macro_f1" => (0.0..=1.0).contains(&value),
            "rmse" | "mase" => value >= 0.0,
            _ => false,
        }
}

fn empty_result_catalog(config: &Config) -> ResultCatalog {
    config
        .tasks
        .iter()
        .map(|task| (task.id.clone(), BTreeMap::new()))
        .collect()
}

/// Keep the best score per (task, method, dataset).
pub fn insert_best_score(
    catalog: &mut ResultCatalog,
    task: &TaskConfig,
    method: String,
    dataset: String,
    value: f64,
) {
    let scores = catalog
        .get_mut(&task.id)
        .expect("validated task")
        .entry(method)
        .or_default();
    match scores.get_mut(&dataset) {
        Some(current) if metric_better(value, *current, task.direction) => *current = value,
        Some(_) => {}
        None => {
            scores.insert(dataset, value);
        }
    }
}

fn add_score_document(
    config: &Config,
    datasets: &DatasetCatalog,
    catalog: &mut ResultCatalog,
    valid_catalog: &mut ResultCatalog,
    document: ScoreDocument,
) -> Result<()> {
    ensure!(document.schema == "sotarena.scores", "invalid score schema");
    let task = task_config(config, &document.task)?;
    ensure!(document.metric == task.metric, "score metric mismatch");
    let task_datasets = &datasets[&task.id];
    let hash = dataset_hash(task_datasets, &document.dataset_hash)?;
    ensure!(hash == document.dataset_hash, "score dataset mismatch");
    for row in document.rows {
        let Some(value) = row.value else { continue };
        if row.status != "ok" {
            continue;
        }
        ensure!(valid_method_id(&row.method_id), "invalid method id");
        ensure!(
            valid_metric_value(&task.metric, value),
            "invalid {} score for {}",
            task.metric,
            row.method_id
        );
        insert_best_score(catalog, task, row.method_id.clone(), hash.to_owned(), value);
        if row.benchmark_valid == Some(true) {
            insert_best_score(valid_catalog, task, row.method_id, hash.to_owned(), value);
        }
    }
    Ok(())
}

fn add_result_record(
    config: &Config,
    datasets: &DatasetCatalog,
    catalog: &mut ResultCatalog,
    valid_catalog: &mut ResultCatalog,
    record: ResultRecord,
) -> Result<()> {
    ensure!(
        matches!(
            record.schema.as_str(),
            "sotarena.result" | "sotarena.result.v1"
        ),
        "invalid result schema"
    );
    let task = task_config(config, &record.task)?;
    ensure!(record.metric == task.metric, "result metric mismatch");
    let task_datasets = &datasets[&task.id];
    let hash = dataset_hash(task_datasets, &record.dataset_hash)?;
    ensure!(hash == record.dataset_hash, "result dataset mismatch");
    let dataset = &task_datasets[hash];
    ensure!(
        record.dataset_id == dataset.dataset_id || record.dataset_id == dataset.dataset_hash,
        "result dataset id mismatch"
    );
    ensure!(
        record.test_sha256 == dataset.test.sha256,
        "result test split mismatch for {}",
        record.method_id
    );
    ensure!(valid_method_id(&record.method_id), "invalid method id");
    ensure!(
        !record.display_name.trim().is_empty(),
        "empty display name for {}",
        record.method_id
    );
    ensure!(
        valid_metric_value(&task.metric, record.value),
        "invalid {} score for {}",
        task.metric,
        record.method_id
    );
    ensure!(
        !record.run_id.trim().is_empty() && !record.claim_scope.trim().is_empty(),
        "invalid result provenance"
    );
    insert_best_score(
        catalog,
        task,
        record.method_id.clone(),
        hash.to_owned(),
        record.value,
    );
    if record.benchmark_valid {
        insert_best_score(
            valid_catalog,
            task,
            record.method_id,
            hash.to_owned(),
            record.value,
        );
    }
    Ok(())
}

fn collect_dataset_references(value: &Value, references: &mut BTreeSet<String>) -> Result<()> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if key == "dataset_hash" {
                    let hash = child
                        .as_str()
                        .context("dataset_hash reference must be a string")?;
                    references.insert(hash.to_owned());
                }
                collect_dataset_references(child, references)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                collect_dataset_references(child, references)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_dataset_references(
    path: &Path,
    value: &Value,
    datasets: &DatasetCatalog,
) -> Result<()> {
    let mut references = BTreeSet::new();
    collect_dataset_references(value, &mut references)
        .with_context(|| format!("invalid dataset reference in {}", path.display()))?;
    if references.is_empty() {
        return Ok(());
    }
    let task = value
        .get("task")
        .or_else(|| value.get("task_type"))
        .and_then(Value::as_str);
    if let Some(task) = task {
        let task_datasets = datasets
            .get(task)
            .with_context(|| format!("unknown task {task} in {}", path.display()))?;
        for hash in references {
            ensure!(
                task_datasets.contains_key(&hash),
                "unknown dataset {hash} referenced by {}",
                path.display()
            );
        }
    } else {
        for hash in references {
            let matches = datasets
                .values()
                .filter(|task_datasets| task_datasets.contains_key(&hash))
                .count();
            ensure!(
                matches == 1,
                "dataset {hash} referenced by {} resolves to {matches} tasks",
                path.display()
            );
        }
    }
    Ok(())
}

/// Read every result record and score document under `data/`.
pub fn load_results(
    root: &Path,
    config: &Config,
    datasets: &DatasetCatalog,
) -> Result<LoadedResults> {
    let mut catalog = empty_result_catalog(config);
    let mut valid_catalog = empty_result_catalog(config);
    let mut result_jsons: BTreeMap<_, _> = config
        .tasks
        .iter()
        .map(|task| (task.id.clone(), 0))
        .collect();
    let mut paths = Vec::new();
    for task in &config.tasks {
        let task_paths = result_paths(root, &task.id, &datasets[&task.id])?;
        result_jsons.insert(task.id.clone(), task_paths.len());
        paths.extend(task_paths);
    }
    paths.extend(global_result_paths(root, config)?);
    paths.sort();
    paths.dedup();
    let result_jsons_total = paths.len();
    for path in paths {
        let value: Value = read_json(&path)?;
        validate_dataset_references(&path, &value, datasets)?;
        match value["schema"].as_str() {
            Some("sotarena.scores") => {
                let document: ScoreDocument = serde_json::from_value(value)?;
                add_score_document(config, datasets, &mut catalog, &mut valid_catalog, document)
                    .with_context(|| path.display().to_string())?;
            }
            Some("sotarena.result" | "sotarena.result.v1") => {
                let record: ResultRecord = serde_json::from_value(value)?;
                add_result_record(config, datasets, &mut catalog, &mut valid_catalog, record)
                    .with_context(|| path.display().to_string())?;
            }
            _ => {}
        }
    }
    Ok(LoadedResults {
        catalog,
        valid_catalog,
        result_jsons,
        result_jsons_total,
    })
}
