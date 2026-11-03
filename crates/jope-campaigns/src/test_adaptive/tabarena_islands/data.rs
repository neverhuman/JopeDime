use super::*;

pub(crate) fn campaign_root(run_id: &str) -> PathBuf {
    run_root(run_id)
}

fn encode_source_dataset(source: &Path, destination: &Path) -> Result<(usize, usize, usize)> {
    let train_text = fs::read_to_string(source.join("train.csv"))?;
    let test_text = fs::read_to_string(source.join("test.csv"))?;
    let train_table = RawTable::from_csv(&train_text)?;
    let test_table = RawTable::from_csv(&test_text)?;
    if train_table.headers != test_table.headers {
        bail!(
            "TabArena train/test headers differ for {}",
            source.display()
        );
    }
    let width = train_table.headers.len();
    if width < 2 {
        bail!("TabArena CSV has no feature columns");
    }
    let train_rows = train_table.rows;
    let test_rows = test_table.rows;
    let features = width - 1;
    let metadata: Value = serde_json::from_slice(&fs::read(source.join("metadata.json"))?)?;
    let categorical = metadata
        .get("categorical_indicator")
        .and_then(Value::as_array)
        .context("TabArena metadata lacks categorical_indicator")?
        .iter()
        .map(|value| {
            value
                .as_bool()
                .context("TabArena categorical_indicator is not boolean")
        })
        .collect::<Result<Vec<_>>>()?;
    if categorical.len() != features {
        bail!("TabArena categorical_indicator width mismatch");
    }
    let dictionaries = (0..features)
        .map(|column| {
            if !categorical[column] {
                return BTreeMap::new();
            }
            let values = train_rows
                .iter()
                .map(|row| row[column].clone())
                .collect::<BTreeSet<_>>();
            values
                .into_iter()
                .enumerate()
                .map(|(index, value)| (value, index as f64))
                .collect()
        })
        .collect::<Vec<BTreeMap<String, f64>>>();
    let numeric_fill = (0..features)
        .map(|column| {
            if categorical[column] {
                return 0.0;
            }
            let mut values = train_rows
                .iter()
                .filter_map(|row| row[column].parse::<f64>().ok())
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            match values.len() {
                0 => 0.0,
                len if len % 2 == 1 => values[len / 2],
                len => (values[len / 2 - 1] + values[len / 2]) / 2.0,
            }
        })
        .collect::<Vec<_>>();
    let encode = |rows: &[Vec<String>]| -> Result<Vec<u8>> {
        let mut output = Vec::new();
        for row in rows {
            let mut values = Vec::with_capacity(width);
            for column in 0..features {
                let value = if categorical[column] {
                    dictionaries[column]
                        .get(&row[column])
                        .copied()
                        .unwrap_or(-1.0)
                } else {
                    row[column]
                        .parse::<f64>()
                        .ok()
                        .filter(|value| value.is_finite())
                        .unwrap_or(numeric_fill[column])
                };
                values.push(format!("{value:.17}"));
            }
            let target = row[features]
                .parse::<f64>()
                .with_context(|| "TabArena target is not numeric")?;
            if !target.is_finite() {
                bail!("TabArena target is not finite");
            }
            values.push(format!("{target:.17}"));
            output.extend_from_slice(values.join(",").as_bytes());
            output.push(b'\n');
        }
        Ok(output)
    };
    fs::create_dir_all(destination)?;
    let train_bytes = encode(&train_rows)?;
    let test_bytes = encode(&test_rows)?;
    let train_path = destination.join("train.csv");
    let test_path = destination.join("test.csv");
    if train_path.exists() {
        if fs::read(&train_path)? != train_bytes || fs::read(&test_path)? != test_bytes {
            bail!("TabArena encoded dataset changed on resume");
        }
    } else {
        immutable_write(&train_path, &train_bytes)?;
        immutable_write(&test_path, &test_bytes)?;
    }
    Ok((
        train_rows.len(),
        test_rows.len(),
        categorical.iter().filter(|value| **value).count(),
    ))
}

pub(crate) fn collect_canonical_leaders(
    canonical: &str,
    leaders: &mut HashMap<String, (f64, BTreeSet<String>)>,
) -> Result<()> {
    let table = RawTable::from_csv(canonical)?;
    let column = |name: &str| {
        table
            .headers
            .iter()
            .position(|header| header == name)
            .with_context(|| format!("canonical score CSV lacks {name}"))
    };
    let method = column("method")?;
    let dataset = column("dataset")?;
    let fold = column("fold")?;
    let metric = column("metric")?;
    let metric_error = column("metric_error")?;
    for row in &table.rows {
        if row[fold] != "0" || row[metric] != "rmse" {
            continue;
        }
        let Ok(value) = row[metric_error].parse::<f64>() else {
            continue;
        };
        if value.is_finite() {
            update_leader(leaders, &row[dataset], &row[method], value);
        }
    }
    Ok(())
}

type LocalLeaders = (HashMap<String, (f64, BTreeSet<String>)>, Vec<ArtifactHash>);

fn collect_local_leaders() -> Result<LocalLeaders> {
    let mut leaders = HashMap::<String, (f64, BTreeSet<String>)>::new();
    let canonical = fs::read_to_string(canonical_scores_str())?;
    collect_canonical_leaders(&canonical, &mut leaders)?;
    let mut artifacts = vec![ArtifactHash {
        file: canonical_scores_str().to_string(),
        sha256: sha256_file(Path::new(canonical_scores_str()))?,
    }];
    let results_root = Path::new(local_results_str());
    let mut paths = Vec::new();
    for model in fs::read_dir(results_root)? {
        let model = model?.path();
        if !model.is_dir() {
            continue;
        }
        for entry in fs::read_dir(model)? {
            let path = entry?.path();
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if path.is_file()
                && (name == "consolidated.json"
                    || (name.starts_with("part-") && name.ends_with(".jsonl")))
            {
                paths.push(path);
            }
        }
    }
    paths.sort();
    for path in paths {
        let bytes = fs::read(&path)?;
        let values = if path.extension().and_then(|value| value.to_str()) == Some("jsonl") {
            let text = std::str::from_utf8(&bytes)
                .with_context(|| format!("decode results file {} as UTF-8", path.display()))?;
            let mut values = Vec::new();
            for (line_number, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                values.push(serde_json::from_str::<Value>(line).with_context(|| {
                    format!(
                        "parse results file {} line {}",
                        path.display(),
                        line_number + 1
                    )
                })?);
            }
            values
        } else {
            serde_json::from_slice::<Vec<Value>>(&bytes)
                .with_context(|| format!("parse results file {}", path.display()))?
        };
        let mut used = false;
        for value in values {
            if value.get("category").and_then(Value::as_str) != Some("regression")
                || value.get("status").and_then(Value::as_str) != Some("ok")
                || value.get("metric").and_then(Value::as_str) != Some("rmse")
                || value.get("benchmark_valid").and_then(Value::as_bool) == Some(false)
                || value
                    .get("fold")
                    .and_then(Value::as_u64)
                    .is_some_and(|fold| fold != 0)
                || value
                    .get("repeat")
                    .and_then(Value::as_u64)
                    .is_some_and(|repeat| repeat != 0)
            {
                continue;
            }
            let (Some(dataset), Some(model), Some(score)) = (
                value.get("dataset").and_then(Value::as_str),
                value.get("model").and_then(Value::as_str),
                value
                    .get("score")
                    .or_else(|| value.get("metric_error"))
                    .and_then(Value::as_f64),
            ) else {
                continue;
            };
            if score.is_finite() {
                update_leader(&mut leaders, dataset, model, score);
                used = true;
            }
        }
        if used {
            artifacts.push(ArtifactHash {
                file: path.to_string_lossy().into_owned(),
                sha256: sha256_bytes(&bytes),
            });
        }
    }
    Ok((leaders, artifacts))
}

pub(crate) fn update_leader(
    leaders: &mut HashMap<String, (f64, BTreeSet<String>)>,
    dataset: &str,
    model: &str,
    score: f64,
) {
    let entry = leaders
        .entry(dataset.to_string())
        .or_insert_with(|| (f64::INFINITY, BTreeSet::new()));
    match score.total_cmp(&entry.0) {
        Ordering::Less => {
            entry.0 = score;
            entry.1.clear();
            entry.1.insert(model.to_string());
        }
        Ordering::Equal => {
            entry.1.insert(model.to_string());
        }
        Ordering::Greater => {}
    }
}

pub(in crate::test_adaptive) fn prepare_datasets(
    run_id: &str,
) -> Result<(Vec<EncodedDataset>, Vec<ArtifactHash>)> {
    let (leaders, artifacts) = collect_local_leaders()?;
    let source_root = Path::new(source_root_str());
    let mut sources = fs::read_dir(source_root)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    sources.retain(|path| path.is_dir());
    sources.sort_by_key(|path| path.file_name().map(|name| name.to_os_string()));
    let encoded_root = campaign_root(run_id).join("encoded");
    let mut datasets = Vec::new();
    for source in sources {
        let dataset = source
            .file_name()
            .and_then(|name| name.to_str())
            .context("non-UTF8 TabArena dataset name")?
            .to_string();
        let (leader_rmse, leader_models) = leaders
            .get(&dataset)
            .with_context(|| format!("TabArena dataset {dataset} has no finite RMSE leader"))?;
        let destination = encoded_root.join(&dataset);
        let (train_rows, test_rows, categorical_features) =
            encode_source_dataset(&source, &destination)?;
        let train = read_regression_csv(&destination.join("train.csv"))?;
        let test = read_regression_csv(&destination.join("test.csv"))?;
        let binding = TabArenaSourceBinding {
            dataset: dataset.clone(),
            source_train_sha256: sha256_file(&source.join("train.csv"))?,
            source_test_sha256: sha256_file(&source.join("test.csv"))?,
            metadata_sha256: sha256_file(&source.join("metadata.json"))?,
            encoded_train_sha256: sha256_file(&destination.join("train.csv"))?,
            encoded_test_sha256: sha256_file(&destination.join("test.csv"))?,
            train_rows,
            test_rows,
            features: train.features,
            categorical_features,
            leader_models: leader_models.iter().cloned().collect(),
            leader_rmse: *leader_rmse,
        };
        if train.x.len() != train_rows
            || test.x.len() != test_rows
            || test.features != train.features
        {
            bail!("encoded TabArena dataset shape mismatch");
        }
        datasets.push(EncodedDataset {
            spec: DatasetSpec {
                dataset_hash: dataset,
                train_sha256: binding.encoded_train_sha256.clone(),
                test_sha256: binding.encoded_test_sha256.clone(),
                train_rows,
                test_rows,
                features: binding.features,
                leader_models: binding.leader_models.clone(),
                leader_rmse: binding.leader_rmse,
            },
            binding,
        });
    }
    if datasets.len() != 13 {
        bail!(
            "TabArena Lite regression cohort contains {} datasets, expected 13",
            datasets.len()
        );
    }
    Ok((datasets, artifacts))
}

pub(in crate::test_adaptive) fn select_datasets(
    datasets: Vec<EncodedDataset>,
    selector: Option<&str>,
) -> Result<Vec<EncodedDataset>> {
    let Some(selector) = selector else {
        return Ok(datasets);
    };
    let requested = selector.split(',').collect::<Vec<_>>();
    if requested.is_empty() || requested.iter().any(|name| name.is_empty()) {
        bail!("TabArena dataset selector is empty or malformed");
    }
    let requested = requested
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    if requested.len() != selector.split(',').count() {
        bail!("TabArena dataset selector contains a duplicate");
    }
    let available = datasets
        .iter()
        .map(|dataset| dataset.spec.dataset_hash.clone())
        .collect::<BTreeSet<_>>();
    let unknown = requested
        .difference(&available)
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        bail!(
            "unknown TabArena dataset selector(s): {}",
            unknown.join(",")
        );
    }
    let selected = datasets
        .into_iter()
        .filter(|dataset| requested.contains(&dataset.spec.dataset_hash))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!("TabArena dataset selector selected no datasets");
    }
    Ok(selected)
}

pub(crate) fn identity(run_id: &str, binary_sha256: &str, spec: &DatasetSpec) -> CampaignIdentity {
    CampaignIdentity {
        schema: CAMPAIGN_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: run_id.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        train_sha256: spec.train_sha256.clone(),
        test_sha256: spec.test_sha256.clone(),
        train_rows: spec.train_rows,
        test_rows: spec.test_rows,
        features: spec.features,
        leader_models: spec.leader_models.clone(),
        leader_rmse: spec.leader_rmse,
        population: TOTAL_POPULATION,
        elites: ELITES,
        parent_pool: PARENT_POOL,
        tournament: TOURNAMENT,
        crossover_probability: 0.25,
        mutation_probability: 0.75,
        max_transforms: feat_math::full_rust::MAX_TRANSFORMS,
        generations: usize::MAX,
        rng_seed: RNG_SEED,
        binary_sha256: binary_sha256.to_string(),
        execution_host: "scq".to_string(),
        execution_node: "xbabe1-xbabe2-xbabe3-ring".to_string(),
        selection_objective:
            "complete four-island generations until test RMSE beats the frozen TabArena leader"
                .to_string(),
    }
}
