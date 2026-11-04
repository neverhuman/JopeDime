use super::*;

pub(super) fn publish_corpus(
    output_root: &Path,
    archive_manifest_bytes: &[u8],
    archive_manifest: &ArchiveManifest,
    aggregates: BTreeMap<ExampleKey, Aggregate>,
    observed: ObservedSchema,
    cohort_filter: Option<&str>,
) -> Result<PathBuf> {
    let archive_sha256 = archive_manifest.all_evaluations_jsonl_sha256.as_str();
    let corpus_sha256 = match cohort_filter {
        None => archive_sha256.to_string(),
        Some(cohort) => sha256_bytes(
            format!("jope-dime.score-corpus.v1\n{archive_sha256}\n{cohort}\n").as_bytes(),
        ),
    };
    let corpus_root = output_root
        .join("dime_score")
        .join("corpora")
        .join(&corpus_sha256);
    if corpus_root.exists() {
        verify_existing_corpus(&corpus_root, archive_sha256, cohort_filter)?;
        return Ok(corpus_root);
    }
    let parent = corpus_root.parent().context("corpus path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".build-{archive_sha256}-{}", std::process::id()));
    if temporary.exists() {
        bail!(
            "stale corpus build directory exists: {}",
            temporary.display()
        );
    }
    fs::create_dir(&temporary)?;

    let percentiles = recompute_percentiles(&aggregates);
    let mut examples = Vec::new();
    let mut dataset_counts = BTreeMap::<String, usize>::new();
    let mut valid_examples = 0usize;
    for (key, aggregate) in &aggregates {
        let total = aggregate.valid_count + aggregate.invalid_count;
        let percentile = percentiles.get(key).copied().flatten();
        valid_examples += usize::from(aggregate.valid_rmse.is_some());
        *dataset_counts
            .entry(format!("{}:{}", key.cohort, key.dataset))
            .or_default() += 1;
        let example = ScoreExample {
            schema: EXAMPLE_SCHEMA,
            cohort: &key.cohort,
            dataset: &key.dataset,
            semantic_sha256: &key.semantic_sha256,
            canonical_genome: &aggregate.canonical_genome,
            genome: &aggregate.genome,
            rmse: aggregate.valid_rmse,
            percentile,
            success_rate: aggregate.valid_count as f64 / total as f64,
            valid_count: aggregate.valid_count,
            invalid_count: aggregate.invalid_count,
            total_count: total,
            failure_classes: &aggregate.failure_classes,
            runtime: runtime_statistics(&aggregate.runtimes),
            provenance: &aggregate.provenance,
        };
        examples.extend(serde_json::to_vec(&example)?);
        examples.push(b'\n');
    }
    let examples_sha256 = sha256_bytes(&examples);

    let parameters = observed
        .parameters
        .iter()
        .map(|((operation, parameter), types)| {
            json!({
                "operation": operation,
                "parameter": parameter,
                "types": types,
                "categorical_values": observed
                    .categorical_values
                    .get(&(operation.clone(), parameter.clone()))
                    .cloned()
                    .unwrap_or_default(),
            })
        })
        .collect::<Vec<_>>();
    let mut gene_schema_value = json!({
        "schema": GENE_SCHEMA,
        "parser_contract": "feat-math::GenomeIr current canonical grammar",
        "action_contract": [
            "form", "member_begin", "member_end", "stage", "parameter",
            "expression_begin", "expression_node", "expression_end"
        ],
        "numeric_contract": "finite IEEE-754 f64 values preserved without grid snapping",
        "operations": observed.operations,
        "parameters": parameters,
    });
    let contract_sha256 = sha256_bytes(&canonical_json(&gene_schema_value)?);
    gene_schema_value
        .as_object_mut()
        .context("gene schema is not an object")?
        .insert("contract_sha256".into(), json!(contract_sha256));
    let gene_schema_bytes = pretty_json(&gene_schema_value)?;
    let gene_schema_sha256 = sha256_bytes(&gene_schema_bytes);
    let archive_manifest_sha256 = sha256_bytes(archive_manifest_bytes);

    let manifest_value = json!({
        "schema": CORPUS_SCHEMA,
        "archive_sha256": archive_sha256,
        "archive_manifest_sha256": archive_manifest_sha256,
        "archive_schema": archive_manifest.schema,
        "cohort_filter": cohort_filter,
        "source_jsonl": archive_manifest.all_evaluations_jsonl,
        "source_records": archive_manifest.evaluation_records,
        "examples_file": "examples.jsonl",
        "examples_sha256": examples_sha256,
        "gene_schema_file": "gene-schema.v1.json",
        "gene_schema_sha256": gene_schema_sha256,
        "unique_examples": aggregates.len(),
        "valid_score_examples": valid_examples,
        "invalid_only_examples": aggregates.len() - valid_examples,
        "datasets": dataset_counts,
        "percentile_definition": "count(unique valid RMSE >= current) / count(unique valid RMSE) within each cohort/dataset; ties receive identical values",
        "validity_definition": "valid_count / (valid_count + invalid_count); mixed outcomes retain their valid RMSE",
    });
    let manifest_bytes = pretty_json(&manifest_value)?;

    write_synced(&temporary.join("examples.jsonl"), &examples)?;
    write_synced(&temporary.join("gene-schema.v1.json"), &gene_schema_bytes)?;
    write_synced(&temporary.join("manifest.v1.json"), &manifest_bytes)?;
    sync_directory(&temporary)?;
    match fs::rename(&temporary, &corpus_root) {
        Ok(()) => {
            sync_directory(parent)?;
            verify_existing_corpus(&corpus_root, archive_sha256, cohort_filter)?;
            Ok(corpus_root)
        }
        Err(_error) if corpus_root.exists() => {
            fs::remove_dir_all(&temporary).ok();
            verify_existing_corpus(&corpus_root, archive_sha256, cohort_filter)
                .context("concurrent corpus publication conflicts")?;
            Ok(corpus_root)
        }
        Err(error) => Err(error).context("publish immutable corpus directory"),
    }
}

fn recompute_percentiles(
    aggregates: &BTreeMap<ExampleKey, Aggregate>,
) -> BTreeMap<ExampleKey, Option<f64>> {
    let mut scores = BTreeMap::<(String, String), Vec<f64>>::new();
    for (key, aggregate) in aggregates {
        if let Some(rmse) = aggregate.valid_rmse {
            scores
                .entry((key.cohort.clone(), key.dataset.clone()))
                .or_default()
                .push(rmse);
        }
    }
    for values in scores.values_mut() {
        values.sort_by(f64::total_cmp);
    }
    aggregates
        .iter()
        .map(|(key, aggregate)| {
            let percentile = aggregate.valid_rmse.map(|rmse| {
                let values = &scores[&(key.cohort.clone(), key.dataset.clone())];
                let lower = values.partition_point(|value| *value < rmse);
                (values.len() - lower) as f64 / values.len() as f64
            });
            (key.clone(), percentile)
        })
        .collect()
}

fn runtime_statistics(values: &[f64]) -> RuntimeStatistics {
    if values.is_empty() {
        return RuntimeStatistics {
            count: 0,
            mean_seconds: None,
            std_seconds: None,
            min_seconds: None,
            p50_seconds: None,
            p90_seconds: None,
            max_seconds: None,
        };
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / sorted.len() as f64;
    RuntimeStatistics {
        count: sorted.len(),
        mean_seconds: Some(mean),
        std_seconds: Some(variance.sqrt()),
        min_seconds: sorted.first().copied(),
        p50_seconds: Some(quantile(&sorted, 0.5)),
        p90_seconds: Some(quantile(&sorted, 0.9)),
        max_seconds: sorted.last().copied(),
    }
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }
}

fn verify_existing_corpus(
    root: &Path,
    archive_sha256: &str,
    cohort_filter: Option<&str>,
) -> Result<()> {
    let manifest_path = root.join("manifest.v1.json");
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    if manifest.get("schema").and_then(Value::as_str) != Some(CORPUS_SCHEMA)
        || manifest.get("archive_sha256").and_then(Value::as_str) != Some(archive_sha256)
        || manifest.get("cohort_filter").and_then(Value::as_str) != cohort_filter
    {
        bail!("existing corpus manifest identity mismatch");
    }
    for (file_key, digest_key) in [
        ("examples_file", "examples_sha256"),
        ("gene_schema_file", "gene_schema_sha256"),
    ] {
        let relative = manifest
            .get(file_key)
            .and_then(Value::as_str)
            .context("existing corpus manifest lacks file name")?;
        let expected = manifest
            .get(digest_key)
            .and_then(Value::as_str)
            .context("existing corpus manifest lacks file digest")?;
        let path = root.join(safe_relative_path(relative)?);
        let actual = sha256_bytes(&fs::read(&path)?);
        if actual != expected {
            bail!(
                "existing corpus artifact SHA-256 mismatch: {}",
                path.display()
            );
        }
    }
    Ok(())
}

pub(super) fn safe_relative_path(value: &str) -> Result<PathBuf> {
    let path = Path::new(value);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("archive path must be a non-empty normal relative path: {value:?}");
    }
    Ok(path.to_path_buf())
}

pub(super) fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} is not a lowercase SHA-256");
    }
    Ok(())
}

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

fn pretty_json(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
