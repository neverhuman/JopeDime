use super::*;

/// Precompute the pinned full-train, CV5, and CV7 encodings for either one
/// registry member or the complete frozen registry.  A complete run publishes
/// a coverage freeze; a targeted run only fills content-addressed entries.
pub fn build_registry_cache(
    registry_path: &Path,
    dataset_id: Option<&str>,
) -> Result<RegistryCacheCoverageV2, String> {
    if let Some(dataset_id) = dataset_id {
        validate_opaque_id(dataset_id)?;
    }
    let registry_bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&registry_bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    registry.encoder.verify_artifacts()?;
    require_cuda_device(registry.encoder.device_ordinal)?;
    let registry_sha256 = registry.identity()?;
    let cache_root = registry.resolved_cache_root(registry_root)?;
    let selected = registry
        .datasets
        .iter()
        .filter(|dataset| dataset_id.is_none_or(|wanted| dataset.dataset_id == wanted))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err("requested dataset is absent from the train-only registry".into());
    }
    let mut datasets = Vec::with_capacity(selected.len());
    for dataset in selected {
        let source = read_authorized_table(registry_root, dataset)?;
        let full_train = encode_full_train(
            &registry.encoder,
            &source,
            dataset,
            &registry_sha256,
            &cache_root,
        )?;
        let cv5_plan = build_grouped_fold_plan(GateStageV1::Cv5Screen, &dataset.atomic_groups)?;
        let cv7_plan =
            build_grouped_fold_plan(GateStageV1::Cv7Confirmation, &dataset.atomic_groups)?;
        let mut cv5 = Vec::with_capacity(5);
        let cv5_sha256 = canonical_sha256(&cv5_plan)?;
        for fold in &cv5_plan.folds {
            let encoded = encode_fold(
                &registry.encoder,
                &source,
                fold,
                &cv5_plan,
                crate::hyperion_gate::FoldEncodingScope {
                    plan_sha256: &cv5_sha256,
                    encoding_root: &cache_root,
                    registry_sha256: &registry_sha256,
                    dataset,
                },
                Instant::now(),
            )?;
            cv5.push(CacheCoverageEntryV2 {
                fold_index: Some(fold.fold_index),
                cache_key_sha256: encoded.cache_key_sha256,
                cache_receipt_sha256: encoded.cache_receipt_sha256,
            });
        }
        let mut cv7 = Vec::with_capacity(7);
        let cv7_sha256 = canonical_sha256(&cv7_plan)?;
        for fold in &cv7_plan.folds {
            let encoded = encode_fold(
                &registry.encoder,
                &source,
                fold,
                &cv7_plan,
                crate::hyperion_gate::FoldEncodingScope {
                    plan_sha256: &cv7_sha256,
                    encoding_root: &cache_root,
                    registry_sha256: &registry_sha256,
                    dataset,
                },
                Instant::now(),
            )?;
            cv7.push(CacheCoverageEntryV2 {
                fold_index: Some(fold.fold_index),
                cache_key_sha256: encoded.cache_key_sha256,
                cache_receipt_sha256: encoded.cache_receipt_sha256,
            });
        }
        datasets.push(DatasetCacheCoverageV2 {
            dataset_id: dataset.dataset_id.clone(),
            full_train,
            cv5,
            cv7,
        });
    }
    let coverage = RegistryCacheCoverageV2 {
        schema: RegistryCacheCoverageV2::SCHEMA.into(),
        registry_sha256,
        datasets,
    };
    coverage.validate()?;
    if dataset_id.is_none() {
        if coverage.datasets.len() != registry.datasets.len() {
            return Err("refusing to freeze incomplete registry cache coverage".into());
        }
        fs::create_dir_all(&cache_root)
            .map_err(|error| format!("create registry cache root: {error}"))?;
        write_immutable(
            &cache_root.join("registry-cache-coverage.v2.json"),
            &canonical_json_bytes(&coverage)?,
        )?;
        let historical = build_historical_train3_inventory(&registry, registry_root)?;
        write_immutable(
            &cache_root.join("historical-train3-inventory.v1.json"),
            &canonical_json_bytes(&historical)?,
        )?;
        verify_complete_registry_cache(registry_path)?;
    }
    Ok(coverage)
}

/// Read-only freeze check.  This never invokes Hyperion: every registry member
/// must already have one validated full-train receipt and all frozen CV5/CV7
/// fold receipts with exact shapes and content hashes.
pub fn verify_complete_registry_cache(
    registry_path: &Path,
) -> Result<RegistryCacheCoverageV2, String> {
    let bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    let registry_sha256 = registry.identity()?;
    let cache_root = registry.resolved_cache_root(registry_root)?;
    let coverage_path = cache_root.join("registry-cache-coverage.v2.json");
    let coverage: RegistryCacheCoverageV2 = serde_json::from_slice(
        &fs::read(&coverage_path)
            .map_err(|error| format!("read registry cache coverage: {error}"))?,
    )
    .map_err(|error| format!("decode registry cache coverage: {error}"))?;
    coverage.validate()?;
    if coverage.registry_sha256 != registry_sha256
        || coverage.datasets.len() != registry.datasets.len()
        || coverage
            .datasets
            .iter()
            .map(|dataset| dataset.dataset_id.as_str())
            .collect::<Vec<_>>()
            != registry
                .datasets
                .iter()
                .map(|dataset| dataset.dataset_id.as_str())
                .collect::<Vec<_>>()
    {
        return Err("registry cache coverage does not exactly match the frozen registry".into());
    }
    for (dataset, covered) in registry.datasets.iter().zip(&coverage.datasets) {
        let source = read_authorized_table(registry_root, dataset)?;
        verify_full_train_entry(
            &registry.encoder,
            &source,
            dataset,
            &registry_sha256,
            &cache_root,
            &covered.full_train,
        )?;
        for (stage, entries) in [
            (GateStageV1::Cv5Screen, covered.cv5.as_slice()),
            (GateStageV1::Cv7Confirmation, covered.cv7.as_slice()),
        ] {
            let plan = build_grouped_fold_plan(stage, &dataset.atomic_groups)?;
            let plan_sha256 = canonical_sha256(&plan)?;
            for (fold, entry) in plan.folds.iter().zip(entries) {
                verify_fold_entry(
                    &registry.encoder,
                    &source,
                    dataset,
                    &registry_sha256,
                    &cache_root,
                    &plan,
                    &plan_sha256,
                    fold,
                    entry,
                )?;
            }
        }
    }
    Ok(coverage)
}

#[allow(clippy::too_many_arguments)]
fn verify_fold_entry(
    encoder: &HyperionEncoderIdentityV1,
    source: &RegressionTable,
    dataset: &TrainOnlyDatasetV1,
    registry_sha256: &str,
    cache_root: &Path,
    plan: &GroupedFoldPlanV1,
    plan_sha256: &str,
    fold: &GroupedFoldV1,
    entry: &CacheCoverageEntryV2,
) -> Result<(), String> {
    let stage = match plan.stage {
        GateStageV1::Cv5Screen => "cv5",
        GateStageV1::Cv7Confirmation => "cv7",
    };
    let fit_input = csv_for_rows(source, &fold.fit_rows, false);
    let hold_input = csv_for_rows(source, &fold.hold_rows, true);
    let expected_key = cache_key(
        registry_sha256,
        dataset,
        encoder,
        stage,
        plan_sha256,
        Some(fold.fold_index),
        &hash_selected_strings(&dataset.stable_row_ids, &fold.fit_rows),
        &hash_selected_strings(&dataset.stable_row_ids, &fold.hold_rows),
        &raw_sha256(&fit_input),
        &raw_sha256(&hold_input),
    )?;
    if entry.fold_index != Some(fold.fold_index)
        || entry.cache_key_sha256 != expected_key.identity()?
    {
        return Err("coverage entry does not match the frozen fold order".into());
    }
    let directory = cache_root
        .join("datasets")
        .join(&dataset.dataset_id)
        .join(stage)
        .join(&entry.cache_key_sha256[..2])
        .join(&entry.cache_key_sha256);
    let receipt: FoldEncodingCacheV2 = serde_json::from_slice(
        &fs::read(directory.join("fold-encoding-cache.v2.json"))
            .map_err(|error| format!("read fold encoding cache: {error}"))?,
    )
    .map_err(|error| format!("decode fold encoding cache: {error}"))?;
    receipt.validate(encoder.device_ordinal)?;
    if receipt.key != expected_key
        || receipt.key_sha256 != entry.cache_key_sha256
        || canonical_sha256(&receipt)? != entry.cache_receipt_sha256
    {
        return Err("fold encoding receipt differs from frozen coverage".into());
    }
    let expected = BTreeMap::from([
        (
            "cell_fit",
            (
                fold.fit_rows.len(),
                dataset.feature_count as usize,
                "cell-fit.vs",
            ),
        ),
        (
            "cell_hold",
            (
                fold.hold_rows.len(),
                dataset.feature_count as usize,
                "cell-hold.vs",
            ),
        ),
        (
            "row_fit",
            (fold.fit_rows.len(), ROW_EMBEDDING_WIDTH, "row-fit.vs"),
        ),
        (
            "row_hold",
            (fold.hold_rows.len(), ROW_EMBEDDING_WIDTH, "row-hold.vs"),
        ),
    ]);
    verify_receipted_sidecars(&directory, &receipt.sidecars, &expected)
}

fn verify_full_train_entry(
    encoder: &HyperionEncoderIdentityV1,
    source: &RegressionTable,
    dataset: &TrainOnlyDatasetV1,
    registry_sha256: &str,
    cache_root: &Path,
    entry: &CacheCoverageEntryV2,
) -> Result<(), String> {
    let rows = (0..source.targets.len()).collect::<Vec<_>>();
    let train_input = csv_for_rows(source, &rows, false);
    let dummy_input = csv_for_rows(source, &rows, true);
    let row_order_sha256 = hash_selected_strings(&dataset.stable_row_ids, &rows);
    let plan_sha256 =
        raw_sha256(format!("full-train-v2|{}|{row_order_sha256}", dataset.train_sha256).as_bytes());
    let expected_key = cache_key(
        registry_sha256,
        dataset,
        encoder,
        "full_train",
        &plan_sha256,
        None,
        &row_order_sha256,
        &row_order_sha256,
        &raw_sha256(&train_input),
        &raw_sha256(&dummy_input),
    )?;
    if entry.fold_index.is_some() || entry.cache_key_sha256 != expected_key.identity()? {
        return Err("full-train coverage entry identity drifted".into());
    }
    let directory = cache_root
        .join("datasets")
        .join(&dataset.dataset_id)
        .join("full_train")
        .join(&entry.cache_key_sha256[..2])
        .join(&entry.cache_key_sha256);
    let receipt: FullTrainEncodingCacheV2 = serde_json::from_slice(
        &fs::read(directory.join("full-train-encoding-cache.v2.json"))
            .map_err(|error| format!("read full-train encoding cache: {error}"))?,
    )
    .map_err(|error| format!("decode full-train encoding cache: {error}"))?;
    receipt.validate(encoder.device_ordinal)?;
    if receipt.key != expected_key
        || receipt.key_sha256 != entry.cache_key_sha256
        || canonical_sha256(&receipt)? != entry.cache_receipt_sha256
    {
        return Err("full-train encoding receipt differs from frozen coverage".into());
    }
    let expected = BTreeMap::from([
        (
            "cell_train",
            (rows.len(), dataset.feature_count as usize, "cell-train.vs"),
        ),
        (
            "row_train",
            (rows.len(), ROW_EMBEDDING_WIDTH, "row-train.vs"),
        ),
    ]);
    verify_receipted_sidecars(&directory, &receipt.sidecars, &expected)
}

fn verify_receipted_sidecars(
    directory: &Path,
    sidecars: &[MatrixArtifactV1],
    expected: &BTreeMap<&str, (usize, usize, &str)>,
) -> Result<(), String> {
    if sidecars.len() != expected.len() {
        return Err("receipted sidecar coverage is incomplete".into());
    }
    for sidecar in sidecars {
        sidecar.validate()?;
        let (rows, columns, file) = expected
            .get(sidecar.role.as_str())
            .ok_or_else(|| "receipted sidecar role is unexpected".to_string())?;
        if sidecar.rows as usize != *rows || sidecar.columns as usize != *columns {
            return Err("receipted sidecar shape differs from the frozen fold".into());
        }
        let path = directory.join(file);
        let bytes = fs::read(&path)
            .map_err(|error| format!("read receipted sidecar {}: {error}", path.display()))?;
        if bytes.is_empty() || raw_sha256(&bytes) != sidecar.sha256 {
            return Err("receipted sidecar is empty or changed".into());
        }
        read_vs_matrix(&path, *rows, *columns, &sidecar.role)?;
    }
    Ok(())
}
