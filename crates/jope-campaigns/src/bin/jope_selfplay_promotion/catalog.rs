use super::*;

// ---------------------------------------------------------------------------
// Catalog validation (verify_contexts=False surface of the Python original)
// ---------------------------------------------------------------------------

/// Per-task authority metric (R1/D2); must agree with
/// `jope_dime_regressor::task::TaskKind::primary_metric`.
pub(super) fn expected_catalog_metric(task: &str) -> Result<&'static str> {
    Task::parse(task)
        .map(Task::primary_metric)
        .ok_or_else(|| anyhow!("catalog contains unsupported task {task}"))
}

pub(super) fn declared_catalog_counts(manifest: &Value) -> BTreeMap<String, i64> {
    let mut counts = BTreeMap::new();
    let declared = manifest
        .get("task_counts")
        .or_else(|| manifest.get("declared_task_counts"));
    if let Some(Value::Object(map)) = declared {
        for task in TASKS {
            // The legacy tasks must always be declared (-1 fails closed
            // below); catalogs written before the new tasks existed simply
            // omit them, which is a declared count of zero.
            let missing = if matches!(task, "regression" | "binary") {
                -1
            } else {
                0
            };
            counts.insert(
                task.to_owned(),
                map.get(task).and_then(Value::as_i64).unwrap_or(missing),
            );
        }
        return counts;
    }
    counts.insert(
        "regression".to_owned(),
        manifest
            .get("regression_count")
            .and_then(Value::as_i64)
            .unwrap_or(-1),
    );
    counts.insert(
        "binary".to_owned(),
        manifest
            .get("binary_count")
            .and_then(Value::as_i64)
            .unwrap_or(-1),
    );
    counts
}

pub(super) fn validate_catalog_counts(
    entries: &[Value],
    declared: &BTreeMap<String, i64>,
) -> Result<BTreeMap<String, i64>> {
    if declared.values().any(|count| *count < 0) {
        bail!("catalog manifest does not declare every per-task count");
    }
    let known_tasks: BTreeSet<&str> = TASKS.into();
    let declared_tasks = declared.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if !declared_tasks.is_subset(&known_tasks)
        || !declared_tasks.contains("regression")
        || !declared_tasks.contains("binary")
    {
        bail!("catalog manifest declares unsupported task counts");
    }
    // Missing new-task declarations mean zero datasets for that task; the
    // legacy-task declarations stay mandatory (checked just above).
    let mut expected: BTreeMap<String, i64> =
        TASKS.iter().map(|task| ((*task).to_owned(), 0)).collect();
    for (task, count) in declared {
        expected.insert(task.clone(), *count);
    }
    let mut counts: BTreeMap<String, i64> =
        TASKS.iter().map(|task| ((*task).to_owned(), 0)).collect();
    let mut seen = BTreeSet::new();
    for entry in entries {
        let task = entry.get("task").and_then(Value::as_str).unwrap_or("");
        let dataset = entry
            .get("dataset_hash")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !counts.contains_key(task) || dataset.is_empty() {
            bail!("catalog contains an unsupported or anonymous dataset");
        }
        if !seen.insert((task.to_owned(), dataset.to_owned())) {
            bail!("catalog repeats {task}:{dataset}");
        }
        *counts.get_mut(task).expect("checked") += 1;
    }
    if counts != expected {
        bail!("catalog count mismatch: expected {declared:?}, found {counts:?}");
    }
    Ok(counts)
}

/// Port of `validate_catalog_manifest(pointer, verify_contexts=False)` for
/// both the v1 catalog schema and `JopeExpandedCatalog.v2`.
pub(super) fn validate_catalog_manifest(pointer_path: &Path) -> Result<Value> {
    let value = read_json(pointer_path)?;
    for field in ["catalog", "catalog_sha256"] {
        if value
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or("")
            .is_empty()
        {
            bail!("catalog pointer is missing {field}");
        }
    }
    let catalog_path = PathBuf::from(get_str(&value, "catalog")?);
    if sha256_file(&catalog_path)? != get_str(&value, "catalog_sha256")? {
        bail!("catalog pointer hash mismatch");
    }
    let manifest = read_json(&catalog_path)?;
    let expanded = manifest.get("schema").and_then(Value::as_str) == Some("JopeExpandedCatalog.v2");
    for field in ["data", "data_sha256"] {
        if value
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or("")
            .is_empty()
        {
            if expanded {
                bail!("expanded catalog pointer is missing {field}");
            }
            bail!("catalog pointer is missing {field}");
        }
    }
    let data_path = PathBuf::from(get_str(&value, "data")?);
    let data_sha = get_str(&value, "data_sha256")?;
    if expanded {
        if !data_path.is_file()
            || sha256_file(&data_path)? != data_sha
            || manifest.get("data_sha256").and_then(Value::as_str) != Some(data_sha)
        {
            bail!("expanded catalog context pointer hash mismatch");
        }
        if manifest.get("immutable") != Some(&Value::Bool(true))
            || manifest.get("append_only") != Some(&Value::Bool(true))
        {
            bail!("expanded catalog is not immutable and append-only");
        }
    } else {
        if sha256_file(&data_path)? != data_sha {
            bail!("catalog data pointer hash mismatch");
        }
        if manifest.get("immutable") != Some(&Value::Bool(true)) {
            bail!("catalog manifest is not declared immutable");
        }
        if manifest.get("data_sha256").and_then(Value::as_str) != Some(data_sha) {
            bail!("catalog manifest and pointer disagree on context data");
        }
    }
    let entries = manifest
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("catalog manifest entries are absent"))?;
    let counts = validate_catalog_counts(entries, &declared_catalog_counts(&manifest))?;
    if manifest
        .get("row_count")
        .and_then(Value::as_i64)
        .unwrap_or(-1)
        != entries.len() as i64
    {
        bail!("catalog declared row count mismatch");
    }
    let stride = manifest
        .get("record_stride_bytes")
        .and_then(Value::as_i64)
        .unwrap_or(-1);
    if stride != CATALOG_STRIDE_BYTES as i64 {
        bail!("catalog context stride mismatch");
    }
    if fs::metadata(&data_path)?.len() != entries.len() as u64 * CATALOG_STRIDE_BYTES {
        bail!("catalog context blob row count mismatch");
    }
    let mut indexes: Vec<i64> = entries
        .iter()
        .map(|entry| {
            entry
                .get("catalog_index")
                .and_then(Value::as_i64)
                .unwrap_or(-1)
        })
        .collect();
    indexes.sort_unstable();
    if indexes != (0..entries.len() as i64).collect::<Vec<_>>() {
        bail!("catalog indexes are not unique and contiguous");
    }
    if manifest.get("source_family_disjoint") != Some(&Value::Bool(true)) {
        bail!("catalog does not declare family-disjoint splits");
    }
    let mut split_by_family: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entry in entries {
        let task = get_str(entry, "task")?;
        let metric = entry.get("metric").and_then(Value::as_str).unwrap_or("");
        let expected_metric = expected_catalog_metric(task)?;
        if metric != expected_metric {
            bail!("catalog metric mismatch for {task}");
        }
        let family = entry
            .get("source_family")
            .and_then(Value::as_str)
            .unwrap_or("");
        let split = entry.get("split").and_then(Value::as_str).unwrap_or("");
        if family.is_empty() || !matches!(split, "train" | "validation") {
            bail!("catalog contains an invalid family/split identity");
        }
        split_by_family
            .entry(family.to_owned())
            .or_default()
            .insert(split.to_owned());
        for field in ["context_sha256", "train_sha256", "test_sha256"] {
            if !is_sha256(entry.get(field).and_then(Value::as_str).unwrap_or("")) {
                bail!("catalog entry has invalid {field}");
            }
        }
    }
    if split_by_family.values().any(|splits| splits.len() != 1) {
        bail!("catalog source families cross splits");
    }
    let mut task_counts = Map::new();
    for (task, count) in counts {
        task_counts.insert(task, json!(count));
    }
    Ok(json!({
        "pointer": pointer_path.to_string_lossy(),
        "pointer_sha256": sha256_file(pointer_path)?,
        "catalog": catalog_path.to_string_lossy(),
        "catalog_sha256": get_str(&value, "catalog_sha256")?,
        "data": data_path.to_string_lossy(),
        "data_sha256": data_sha,
        "row_count": entries.len(),
        "task_counts": task_counts,
        "source_family_disjoint": true,
    }))
}
