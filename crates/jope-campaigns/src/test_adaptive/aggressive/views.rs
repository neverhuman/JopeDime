use super::*;

pub(crate) fn remove_csv_header(bytes: &[u8], remove: bool, label: &str) -> Result<Vec<u8>> {
    if !remove {
        return Ok(bytes.to_vec());
    }
    let newline = bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .with_context(|| format!("{label} has a header but no data line"))?;
    let data = bytes[newline + 1..].to_vec();
    if data.is_empty() {
        bail!("{label} has no rows after its header");
    }
    Ok(data)
}

pub(crate) fn golden46_panel_value() -> Result<(Value, Vec<u8>, String)> {
    let path = Path::new(golden46_panel_manifest_str());
    let bytes = fs::read(path)?;
    let value: Value = serde_json::from_slice(&bytes)?;
    let dataset_ids = value
        .get("dataset_ids")
        .and_then(Value::as_array)
        .context("frozen panel46 has no dataset_ids")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .context("frozen panel46 dataset id is not a string")
        })
        .collect::<Result<Vec<_>>>()?;
    if value.get("schema").and_then(Value::as_str) != Some("golden46.dataset-panel-manifest.v1")
        || value.get("campaign_id").and_then(Value::as_str) != Some(GOLDEN46_CAMPAIGN_ID)
        || dataset_ids.len() != 46
        || dataset_ids[42..] != GOLDEN46_PANEL_DATASETS
        || dataset_ids.iter().collect::<HashSet<_>>().len() != dataset_ids.len()
    {
        bail!("frozen panel46 identity or ordered cohort changed");
    }
    let digest = sha256_bytes(&bytes);
    Ok((value, bytes, digest))
}

pub(crate) fn prepare_golden46_evolution_views(
    source_registry: &Path,
) -> Result<(PathBuf, String)> {
    let (panel, _, panel_sha256) = golden46_panel_value()?;
    if panel.get("dataset_root").and_then(Value::as_str)
        != Some(source_registry.to_string_lossy().as_ref())
    {
        bail!("Golden46 source registry differs from the frozen panel46 data root");
    }
    let records = panel
        .get("datasets")
        .and_then(Value::as_array)
        .context("frozen panel46 has no dataset records")?;
    let by_id = records
        .iter()
        .filter_map(|record| {
            record
                .get("dataset_id")
                .and_then(Value::as_str)
                .map(|id| (id, record))
        })
        .collect::<HashMap<_, _>>();
    let mut view_bytes = Vec::<(String, Vec<u8>, Vec<u8>, EvolutionViewDataset)>::new();
    for dataset_hash in GOLDEN46_DATASETS {
        let record = by_id
            .get(dataset_hash)
            .with_context(|| format!("panel46 lacks dataset record {dataset_hash}"))?;
        let artifacts = record
            .get("artifacts")
            .and_then(Value::as_object)
            .context("panel46 dataset record has no artifact hashes")?;
        let headers_removed = record
            .get("headers_removed_from_csv")
            .and_then(Value::as_bool)
            .context("panel46 dataset record has no header policy")?;
        let source = source_registry.join(dataset_hash);
        let source_train = fs::read(source.join("train.csv"))?;
        let source_test = fs::read(source.join("test.csv"))?;
        let source_train_sha256 = sha256_bytes(&source_train);
        let source_test_sha256 = sha256_bytes(&source_test);
        if artifacts.get("train.csv").and_then(Value::as_str) != Some(source_train_sha256.as_str())
            || artifacts.get("test.csv").and_then(Value::as_str)
                != Some(source_test_sha256.as_str())
        {
            bail!("Golden46 source dataset {dataset_hash} changed after panel freeze");
        }
        let remove_header = !headers_removed;
        let train = remove_csv_header(
            &source_train,
            remove_header,
            &format!("{dataset_hash}/train.csv"),
        )?;
        let test = remove_csv_header(
            &source_test,
            remove_header,
            &format!("{dataset_hash}/test.csv"),
        )?;
        let view = EvolutionViewDataset {
            dataset_hash: dataset_hash.to_string(),
            source_train_sha256,
            source_test_sha256,
            train_sha256: sha256_bytes(&train),
            test_sha256: sha256_bytes(&test),
            removed_train_header: remove_header,
            removed_test_header: remove_header,
        };
        view_bytes.push((dataset_hash.to_string(), train, test, view));
    }
    let view_records = view_bytes
        .iter()
        .map(|(_, _, _, record)| record.clone())
        .collect::<Vec<_>>();
    let view_sha256 = sha256_bytes(&canonical_json(&view_records)?);
    let view_root = root(GOLDEN46_RUN_PREFIX)
        .join("evolution-views")
        .join(&view_sha256);
    let registry = view_root.join("datasets");
    for (dataset_hash, train, test, _) in &view_bytes {
        atomic_immutable_write(&registry.join(dataset_hash).join("train.csv"), train)?;
        atomic_immutable_write(&registry.join(dataset_hash).join("test.csv"), test)?;
    }
    let manifest = EvolutionViewManifest {
        schema: "JopeGolden46EvolutionViews.v1".to_string(),
        campaign_id: GOLDEN46_CAMPAIGN_ID.to_string(),
        panel_manifest_sha256: panel_sha256,
        source_registry_root: source_registry.to_string_lossy().into_owned(),
        view_sha256,
        datasets: view_records,
    };
    let manifest_bytes = canonical_json(&manifest)?;
    atomic_immutable_write(
        &view_root.join(EVOLUTION_VIEW_MANIFEST_FILE),
        &manifest_bytes,
    )?;
    Ok((registry, sha256_bytes(&manifest_bytes)))
}

pub(crate) fn load_golden46_specs(
    registry: &Path,
) -> Result<(Vec<DatasetSpec>, String, String, String)> {
    let (_, _, panel_sha256) = golden46_panel_value()?;
    let view_manifest_path = registry
        .parent()
        .context("Golden46 view registry has no parent")?
        .join(EVOLUTION_VIEW_MANIFEST_FILE);
    let view_manifest_bytes = fs::read(&view_manifest_path)?;
    let view_manifest: EvolutionViewManifest = serde_json::from_slice(&view_manifest_bytes)?;
    if view_manifest_bytes != canonical_json(&view_manifest)?
        || view_manifest.schema != "JopeGolden46EvolutionViews.v1"
        || view_manifest.campaign_id != GOLDEN46_CAMPAIGN_ID
        || view_manifest.panel_manifest_sha256 != panel_sha256
        || view_manifest.datasets.len() != GOLDEN46_DATASETS.len()
        || view_manifest.view_sha256 != sha256_bytes(&canonical_json(&view_manifest.datasets)?)
        || view_manifest
            .datasets
            .iter()
            .map(|dataset| dataset.dataset_hash.as_str())
            .collect::<Vec<_>>()
            != GOLDEN46_DATASETS
    {
        bail!("Golden46 evolution view manifest changed");
    }

    let leader_path = Path::new(golden46_leader_manifest_str());
    let leader_bytes = fs::read(leader_path).with_context(|| {
        format!(
            "Golden46 leaders are not frozen; run scripts/build_golden46_leaders.py after all external baselines resolve: {}",
            leader_path.display()
        )
    })?;
    let leader_value: Value = serde_json::from_slice(&leader_bytes)?;
    let required_dataset_ids = leader_value
        .get("required_dataset_ids")
        .and_then(Value::as_array)
        .context("Golden46 leader manifest has no required_dataset_ids")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .context("Golden46 required dataset id is not a string")
        })
        .collect::<Result<Vec<_>>>()?;
    if leader_value.get("schema").and_then(Value::as_str) != Some("golden46.external-leaders.v1")
        || leader_value.get("campaign_id").and_then(Value::as_str) != Some(GOLDEN46_CAMPAIGN_ID)
        || leader_value.get("benchmark_valid").and_then(Value::as_bool) != Some(true)
        || leader_value.get("panel46_sha256").and_then(Value::as_str) != Some(panel_sha256.as_str())
        || required_dataset_ids != GOLDEN46_DATASETS
        || leader_value.get("excluded_dataset_ids") != Some(&serde_json::json!(["ZTS"]))
    {
        bail!("Golden46 leader manifest identity changed");
    }
    let analysis: LeaderAnalysis = serde_json::from_slice(&leader_bytes)?;
    let by_hash = analysis
        .datasets
        .iter()
        .map(|dataset| (dataset.dataset_hash.as_str(), dataset))
        .collect::<HashMap<_, _>>();
    if by_hash.len() != GOLDEN46_DATASETS.len()
        || by_hash
            .keys()
            .any(|dataset| !GOLDEN46_DATASETS.contains(dataset))
    {
        bail!("Golden46 leader manifest dataset cohort changed");
    }

    let views = view_manifest
        .datasets
        .iter()
        .map(|view| (view.dataset_hash.as_str(), view))
        .collect::<HashMap<_, _>>();
    let mut specs = Vec::with_capacity(GOLDEN46_DATASETS.len());
    for dataset_hash in GOLDEN46_DATASETS {
        let view = views
            .get(dataset_hash)
            .with_context(|| format!("Golden46 view is missing {dataset_hash}"))?;
        let leader = by_hash
            .get(dataset_hash)
            .with_context(|| format!("Golden46 leaders are missing {dataset_hash}"))?;
        let (leader_models, leader_rmse) = select_leader_with_model_count(leader, 123)?;
        let dataset = registry.join(dataset_hash);
        let train_path = dataset.join("train.csv");
        let test_path = dataset.join("test.csv");
        if sha256_file(&train_path)? != view.train_sha256
            || sha256_file(&test_path)? != view.test_sha256
        {
            bail!("Golden46 evolution view {dataset_hash} changed");
        }
        let train = read_regression_csv(&train_path)?;
        let test = read_regression_csv(&test_path)?;
        if train.x.len() != leader.train_rows
            || test.x.len() != leader.test_rows
            || train.features != leader.features
            || test.features != leader.features
        {
            bail!("Golden46 view {dataset_hash} shape differs from the frozen leader");
        }
        specs.push(DatasetSpec {
            dataset_hash: dataset_hash.to_string(),
            train_sha256: view.train_sha256.clone(),
            test_sha256: view.test_sha256.clone(),
            train_rows: leader.train_rows,
            test_rows: leader.test_rows,
            features: leader.features,
            leader_models,
            leader_rmse,
        });
    }
    Ok((
        specs,
        panel_sha256,
        sha256_bytes(&leader_bytes),
        sha256_bytes(&view_manifest_bytes),
    ))
}

pub(crate) fn select_dataset_specs(
    specs: &[DatasetSpec],
    selector: Option<&str>,
    start_index: usize,
) -> Result<Vec<DatasetSpec>> {
    if selector.is_some() && start_index != 0 {
        bail!("aggressive dataset selector and start index are mutually exclusive");
    }
    let Some(selector) = selector else {
        if start_index >= specs.len() {
            bail!("aggressive start index {start_index} is outside the Golden42 panel");
        }
        return Ok(specs[start_index..].to_vec());
    };
    let selectors = selector.split(',').collect::<Vec<_>>();
    if selectors.iter().any(|selector| selector.is_empty()) {
        bail!("aggressive dataset selector contains an empty value");
    }
    let mut selected = HashSet::with_capacity(selectors.len());
    for selector in selectors {
        if let Some(index) = specs.iter().position(|spec| spec.dataset_hash == selector) {
            if !selected.insert(index) {
                bail!("dataset selector {selector} is duplicated");
            }
            continue;
        }
        if !selector.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!(
                "non-hexadecimal dataset identifiers such as ZTS and ZTS_small must be selected exactly"
            );
        }
        let selector = selector.to_ascii_lowercase();
        let matches = specs
            .iter()
            .enumerate()
            .filter(|(_, spec)| {
                spec.dataset_hash
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
                    && spec.dataset_hash.starts_with(&selector)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [index] => {
                if !selected.insert(*index) {
                    bail!("dataset selector {selector} is duplicated");
                }
            }
            [] => bail!("no dataset matches selector {selector}"),
            _ => bail!("dataset selector {selector} is ambiguous"),
        }
    }
    Ok(specs
        .iter()
        .enumerate()
        .filter(|(index, _)| selected.contains(index))
        .map(|(_, spec)| spec.clone())
        .collect())
}
