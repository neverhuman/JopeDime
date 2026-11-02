use super::*;

pub(super) fn sha256_f64(values: &[f64]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    hex::encode(hash.finalize())
}

pub(super) fn resolve_pointer_path(pointer: &Path, target: &Path) -> PathBuf {
    if target.is_absolute() {
        target.to_path_buf()
    } else {
        pointer
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(target)
    }
}

pub(super) fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

pub(super) fn resolve_checkpoint_provenance(args: &PrepareArgs) -> Result<CheckpointProvenance> {
    let pointer_bytes = fs::read(&args.checkpoint_pointer).with_context(|| {
        format!(
            "read accepted regression checkpoint pointer {}",
            args.checkpoint_pointer.display()
        )
    })?;
    let pointer: Value = serde_json::from_slice(&pointer_bytes)?;
    if pointer.get("schema").and_then(Value::as_str) != Some("JopeSelfplayAcceptedByTask.v1") {
        bail!("accepted checkpoint pointer has unsupported schema");
    }
    let regression = pointer
        .pointer("/tasks/regression")
        .context("accepted checkpoint pointer lacks regression task")?;
    let checkpoint_path = PathBuf::from(
        regression
            .get("checkpoint")
            .and_then(Value::as_str)
            .context("regression checkpoint path is absent")?,
    );
    let checkpoint_sha256 = regression
        .get("checkpoint_sha256")
        .and_then(Value::as_str)
        .context("regression checkpoint hash is absent")?
        .to_string();
    validate_sha256(&checkpoint_sha256, "accepted regression checkpoint")?;
    if sha256_file(&checkpoint_path)? != checkpoint_sha256 {
        bail!("accepted regression checkpoint pointer content drift");
    }

    let provenance_bytes = fs::read(&args.checkpoint_provenance).with_context(|| {
        format!(
            "read checkpoint provenance {}",
            args.checkpoint_provenance.display()
        )
    })?;
    let provenance: Value = serde_json::from_slice(&provenance_bytes)?;
    if provenance.get("checkpoint_sha").and_then(Value::as_str) != Some(&checkpoint_sha256) {
        bail!("checkpoint provenance does not describe the accepted regression checkpoint");
    }
    let manifest_path = PathBuf::from(
        provenance
            .get("corpus_manifest")
            .and_then(Value::as_str)
            .context("checkpoint provenance lacks corpus manifest")?,
    );
    let corpus_manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    let outputs = corpus_manifest
        .get("outputs")
        .and_then(Value::as_object)
        .context("checkpoint corpus manifest lacks outputs")?;
    let manifest_dir = manifest_path
        .parent()
        .context("corpus manifest has no parent")?;
    let train_path = manifest_dir.join("corpus.train.jsonl");
    let validation_path = manifest_dir.join("corpus.val.jsonl");
    let train_sha256 = outputs
        .get("corpus.train.jsonl")
        .and_then(Value::as_str)
        .context("checkpoint corpus manifest lacks train hash")?
        .to_string();
    let validation_sha256 = outputs
        .get("corpus.val.jsonl")
        .and_then(Value::as_str)
        .context("checkpoint corpus manifest lacks validation hash")?
        .to_string();
    if sha256_file(&train_path)? != train_sha256
        || sha256_file(&validation_path)? != validation_sha256
    {
        bail!("checkpoint training-corpus content drift");
    }
    let model_version = provenance
        .pointer("/checkpoint_config/model_version")
        .and_then(Value::as_str)
        .context("checkpoint provenance lacks model version")?
        .to_string();
    Ok(CheckpointProvenance {
        accepted_pointer_path: absolute_path(&args.checkpoint_pointer)?,
        accepted_pointer_sha256: sha256_bytes(&pointer_bytes),
        checkpoint_path,
        checkpoint_sha256,
        model_version,
        training_corpus_manifest_path: manifest_path.clone(),
        training_corpus_manifest_sha256: sha256_file(&manifest_path)?,
        training_corpus_train_path: train_path,
        training_corpus_train_sha256: train_sha256,
        training_corpus_validation_path: validation_path,
        training_corpus_validation_sha256: validation_sha256,
    })
}

pub(super) fn extract_json_string_field(line: &[u8], field: &[u8]) -> Option<String> {
    let mut start = 0usize;
    while start + field.len() <= line.len() {
        let relative = line[start..]
            .windows(field.len())
            .position(|window| window == field)?;
        let mut index = start + relative + field.len();
        while line.get(index).is_some_and(u8::is_ascii_whitespace) {
            index += 1;
        }
        if line.get(index) != Some(&b':') {
            start = index;
            continue;
        }
        index += 1;
        while line.get(index).is_some_and(u8::is_ascii_whitespace) {
            index += 1;
        }
        if line.get(index) != Some(&b'"') {
            start = index;
            continue;
        }
        index += 1;
        let end = line[index..].iter().position(|byte| *byte == b'"')? + index;
        return std::str::from_utf8(&line[index..end])
            .ok()
            .map(str::to_string);
    }
    None
}

pub(super) fn corpus_dataset_ids(paths: &[&Path]) -> Result<BTreeSet<String>> {
    let mut result = BTreeSet::new();
    for path in paths {
        let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
        let mut line = Vec::new();
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            let dataset = extract_json_string_field(&line, b"\"dataset_id\"")
                .context("checkpoint corpus row lacks dataset_id")?;
            result.insert(dataset);
        }
    }
    Ok(result)
}

pub(super) fn load_catalog(pointer_path: &Path) -> Result<(CatalogPointer, Catalog)> {
    let pointer_bytes = fs::read(pointer_path)?;
    let pointer: CatalogPointer = serde_json::from_slice(&pointer_bytes)?;
    validate_sha256(&pointer.catalog_sha256, "catalog")?;
    validate_sha256(&pointer.data_sha256, "catalog data")?;
    let catalog_path = resolve_pointer_path(pointer_path, &pointer.catalog);
    let data_path = resolve_pointer_path(pointer_path, &pointer.data);
    if sha256_file(&catalog_path)? != pointer.catalog_sha256
        || sha256_file(&data_path)? != pointer.data_sha256
    {
        bail!("context catalog pointer content drift");
    }
    let catalog: Catalog = serde_json::from_slice(&fs::read(catalog_path)?)?;
    if catalog.dataset_dimensions != 1536
        || catalog.target_dimensions != 512
        || catalog.record_stride_bytes != CONTEXT_BYTES
    {
        bail!("context catalog is not the required 1536+512 encoding");
    }
    Ok((
        CatalogPointer {
            catalog: pointer.catalog,
            catalog_sha256: pointer.catalog_sha256,
            data: data_path,
            data_sha256: pointer.data_sha256,
        },
        catalog,
    ))
}

pub(super) fn checkpoint_exclusions(
    run_id: &str,
    registry: &Path,
    checkpoint: &CheckpointProvenance,
    catalog: &Catalog,
) -> Result<CorpusExclusions> {
    let ids = corpus_dataset_ids(&[
        &checkpoint.training_corpus_train_path,
        &checkpoint.training_corpus_validation_path,
    ])?;
    let families_by_id = catalog
        .entries
        .iter()
        .map(|entry| (entry.dataset_hash.as_str(), entry.source_family.as_str()))
        .collect::<HashMap<_, _>>();
    let mut source_families = BTreeSet::new();
    let mut target_identities = BTreeSet::new();
    for dataset in &ids {
        if let Some(family) = families_by_id.get(dataset.as_str()) {
            source_families.insert((*family).to_string());
        } else if let Some(family) = registry_source_family(registry, dataset)? {
            source_families.insert(family);
        }
        let meta_path = sotarena_metadata_path(registry, dataset);
        if meta_path.is_file() {
            target_identities.insert(target_identity(registry, dataset)?);
        } else {
            target_identities.insert(format!("dataset:{dataset}"));
        }
    }
    Ok(CorpusExclusions {
        schema: "TrainOnlyCheckpointCorpusDisclosure.v2".to_string(),
        run_id: run_id.to_string(),
        checkpoint_sha256: checkpoint.checkpoint_sha256.clone(),
        training_corpus_train_sha256: checkpoint.training_corpus_train_sha256.clone(),
        training_corpus_validation_sha256: checkpoint.training_corpus_validation_sha256.clone(),
        dataset_ids: ids.into_iter().collect(),
        source_families: source_families.into_iter().collect(),
        target_identities: target_identities.into_iter().collect(),
    })
}

pub(super) fn validate_train_data(data: &RegressionData) -> Vec<String> {
    let mut reasons = Vec::new();
    if data.y.len() < 100 {
        reasons.push("fewer_than_100_training_rows".to_string());
    }
    if data.x.len() != data.y.len()
        || data
            .x
            .iter()
            .any(|row| row.len() != data.features || row.iter().any(|value| !value.is_finite()))
        || data.y.iter().any(|value| !value.is_finite())
    {
        reasons.push("nonfinite_or_ragged_numeric_training_data".to_string());
    }
    if !data.y.is_empty() {
        let mean = data.y.iter().sum::<f64>() / data.y.len() as f64;
        if data
            .y
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            <= 1e-24
        {
            reasons.push("zero_target_variance".to_string());
        }
    }
    reasons
}

pub(super) fn completed_baseline_report(
    path: &Path,
    dataset_hash: &str,
    algorithm: &str,
) -> Result<String> {
    let bytes =
        fs::read(path).with_context(|| format!("read baseline report {}", path.display()))?;
    let report: Value = serde_json::from_slice(&bytes)?;
    let status = report.get("status").and_then(Value::as_str);
    if report.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
        || report.get("algorithm").and_then(Value::as_str) != Some(algorithm)
        || !matches!(status, Some("completed" | "failed"))
    {
        bail!(
            "baseline report identity/status mismatch for {dataset_hash}: {}",
            path.display()
        );
    }
    Ok(sha256_bytes(&bytes))
}

pub(super) fn sotarena_report_hashes(
    sotarena_root: &Path,
    dataset_hash: &str,
) -> Result<SotarenaReportHashes> {
    let root = sotarena_root
        .join("results/external-references/golden42")
        .join(dataset_hash);
    let tabpfn_v3_path = root.join("tabpfn_v3_regressor.json");
    let tabfm_path = root.join("tabfm_regressor.json");
    let tabfm_ensemble_path = root.join("tabfm_ensemble_regressor.json");
    Ok(SotarenaReportHashes {
        tabpfn_v3_sha256: completed_baseline_report(
            &tabpfn_v3_path,
            dataset_hash,
            "tabpfn_v3_regressor",
        )?,
        tabpfn_v3_path: absolute_path(&tabpfn_v3_path)?,
        tabfm_sha256: completed_baseline_report(&tabfm_path, dataset_hash, "tabfm_regressor")?,
        tabfm_path: absolute_path(&tabfm_path)?,
        tabfm_ensemble_sha256: completed_baseline_report(
            &tabfm_ensemble_path,
            dataset_hash,
            "tabfm_ensemble_regressor",
        )?,
        tabfm_ensemble_path: absolute_path(&tabfm_ensemble_path)?,
    })
}
