use super::*;

pub fn build_historical_train3_inventory(
    registry: &JopeHyperionTrainOnlyRegistryV2,
    registry_root: &Path,
) -> Result<HistoricalTrain3InventoryV1, String> {
    registry.validate(registry_root)?;
    let mut entries = Vec::with_capacity(registry.datasets.len());
    for dataset in &registry.datasets {
        let train = Path::new(&dataset.train_csv_path);
        let relative = train
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join("train3.vs");
        let path = safe_registry_path(registry_root, &relative.to_string_lossy())?;
        let (state, byte_length, sha256) = if path.is_file() {
            let bytes = fs::read(&path)
                .map_err(|error| format!("read historical {}: {error}", path.display()))?;
            if bytes.is_empty() {
                ("zero_byte", 0, None)
            } else {
                ("present", bytes.len() as u64, Some(raw_sha256(&bytes)))
            }
        } else {
            ("missing", 0, None)
        };
        entries.push(HistoricalTrain3EntryV1 {
            dataset_id: dataset.dataset_id.clone(),
            relative_path: relative.to_string_lossy().into_owned(),
            state: state.into(),
            byte_length,
            sha256,
        });
    }
    let inventory = HistoricalTrain3InventoryV1 {
        schema: HistoricalTrain3InventoryV1::SCHEMA.into(),
        registry_sha256: registry.identity()?,
        entries,
        cv_authority: false,
    };
    inventory.validate()?;
    Ok(inventory)
}

/// Freeze exact train-side `.vs` coverage for the registry without invoking an
/// encoder or resolving any test path.  The coverage file is immutable and is
/// suitable for the policy's coverage identity.
pub fn freeze_precomputed_train_feature_coverage(
    registry_path: &Path,
) -> Result<PrecomputedTrainFeatureCoverageV1, String> {
    let bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    let registry_sha256 = registry.identity()?;
    let datasets = registry
        .datasets
        .iter()
        .map(|dataset| {
            load_precomputed_train_features(&registry, registry_root, &registry_sha256, dataset)
                .map(|blocks| blocks.receipt)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let coverage = PrecomputedTrainFeatureCoverageV1 {
        schema: PrecomputedTrainFeatureCoverageV1::SCHEMA.into(),
        registry_sha256,
        datasets,
        test_artifacts_read: false,
    };
    coverage.validate()?;
    let root = registry.resolved_cache_root(registry_root)?;
    fs::create_dir_all(&root)
        .map_err(|error| format!("create precomputed coverage root: {error}"))?;
    write_immutable(
        &root.join("precomputed-train-feature-coverage.v1.json"),
        &canonical_json_bytes(&coverage)?,
    )?;
    verify_precomputed_train_feature_coverage(registry_path)
}

/// Re-read every authorized train-side matrix and require exact equality with
/// the immutable coverage freeze.  This operation cannot invoke Hyperion.
pub fn verify_precomputed_train_feature_coverage(
    registry_path: &Path,
) -> Result<PrecomputedTrainFeatureCoverageV1, String> {
    let bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    let registry_sha256 = registry.identity()?;
    let root = registry.resolved_cache_root(registry_root)?;
    let coverage_path = root.join("precomputed-train-feature-coverage.v1.json");
    let frozen: PrecomputedTrainFeatureCoverageV1 = serde_json::from_slice(
        &fs::read(&coverage_path).map_err(|error| format!("read precomputed coverage: {error}"))?,
    )
    .map_err(|error| format!("decode precomputed coverage: {error}"))?;
    frozen.validate()?;
    let current = PrecomputedTrainFeatureCoverageV1 {
        schema: PrecomputedTrainFeatureCoverageV1::SCHEMA.into(),
        registry_sha256: registry_sha256.clone(),
        datasets: registry
            .datasets
            .iter()
            .map(|dataset| {
                load_precomputed_train_features(&registry, registry_root, &registry_sha256, dataset)
                    .map(|blocks| blocks.receipt)
            })
            .collect::<Result<Vec<_>, _>>()?,
        test_artifacts_read: false,
    };
    current.validate()?;
    if frozen != current
        || frozen
            .datasets
            .iter()
            .map(|receipt| receipt.dataset_id.as_str())
            .collect::<Vec<_>>()
            != registry
                .datasets
                .iter()
                .map(|dataset| dataset.dataset_id.as_str())
                .collect::<Vec<_>>()
    {
        return Err("precomputed train-feature coverage changed after freezing".into());
    }
    Ok(frozen)
}

pub(crate) fn require_frozen_precomputed_dataset_binding(
    registry: &JopeHyperionTrainOnlyRegistryV2,
    registry_root: &Path,
    receipt: &PrecomputedTrainFeatureReceiptV1,
) -> Result<(), String> {
    let root = registry.resolved_cache_root(registry_root)?;
    let path = root.join("precomputed-train-feature-coverage.v1.json");
    let coverage: PrecomputedTrainFeatureCoverageV1 = serde_json::from_slice(
        &fs::read(&path).map_err(|error| format!("read precomputed coverage: {error}"))?,
    )
    .map_err(|error| format!("decode precomputed coverage: {error}"))?;
    coverage.validate()?;
    if coverage.registry_sha256 != registry.identity()?
        || coverage.datasets.len() != registry.datasets.len()
        || coverage
            .datasets
            .iter()
            .map(|entry| entry.dataset_id.as_str())
            .collect::<Vec<_>>()
            != registry
                .datasets
                .iter()
                .map(|dataset| dataset.dataset_id.as_str())
                .collect::<Vec<_>>()
        || coverage
            .datasets
            .iter()
            .find(|entry| entry.dataset_id == receipt.dataset_id)
            != Some(receipt)
    {
        return Err("dataset cache receipt differs from frozen registry-wide coverage".into());
    }
    Ok(())
}

pub(crate) fn read_authorized_table(
    registry_root: &Path,
    dataset: &TrainOnlyDatasetV1,
) -> Result<RegressionTable, String> {
    let train_path = safe_registry_path(registry_root, &dataset.train_csv_path)?;
    let bytes =
        fs::read(&train_path).map_err(|error| format!("read authorized train CSV: {error}"))?;
    if raw_sha256(&bytes) != dataset.train_sha256 {
        return Err("authorized train CSV differs from its registry hash".into());
    }
    let table = parse_regression_csv(&bytes, dataset.feature_count as usize)?;
    if table.targets.len() != dataset.stable_row_ids.len() {
        return Err("authorized train rows differ from registry row identities".into());
    }
    Ok(table)
}

fn sibling_train_sidecar(
    registry_root: &Path,
    dataset: &TrainOnlyDatasetV1,
    file_name: &str,
) -> Result<(String, PathBuf), String> {
    if !matches!(file_name, "train3.vs" | "hyperion3train.vs") {
        return Err("precomputed feature loader accepts train-side .vs names only".into());
    }
    let relative = Path::new(&dataset.train_csv_path)
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join(file_name);
    let relative = relative.to_string_lossy().into_owned();
    let path = safe_registry_path(registry_root, &relative)?;
    Ok((relative, path))
}

pub(crate) fn load_precomputed_train_features(
    registry: &JopeHyperionTrainOnlyRegistryV2,
    registry_root: &Path,
    registry_sha256: &str,
    dataset: &TrainOnlyDatasetV1,
) -> Result<PrecomputedTrainFeatureBlocks, String> {
    let rows = dataset.stable_row_ids.len();
    let columns = dataset.feature_count as usize;
    let (cell_relative_path, cell_path) =
        sibling_train_sidecar(registry_root, dataset, "train3.vs")?;
    let cell_bytes = fs::read(&cell_path)
        .map_err(|error| format!("read precomputed cell {}: {error}", cell_path.display()))?;
    if cell_bytes.is_empty() {
        return Err("precomputed train3.vs is empty".into());
    }
    let cell = read_vs_matrix(&cell_path, rows, columns, "precomputed cell train")?;
    let cell_artifact = MatrixArtifactV1 {
        role: "cell_train_precomputed".into(),
        rows: rows as u32,
        columns: columns as u32,
        sha256: raw_sha256(&cell_bytes),
    };

    // Row128 is an optional arm.  Its absence or corruption is recorded and
    // excludes row/both only; the mandatory, valid cell block remains usable.
    let (row_relative, row_path) =
        sibling_train_sidecar(registry_root, dataset, "hyperion3train.vs")?;
    let (row128_relative_path, row128_artifact, row128, row128_unavailable_reason) =
        match fs::read(&row_path) {
            Ok(bytes) if !bytes.is_empty() => {
                match read_vs_matrix(
                    &row_path,
                    rows,
                    ROW_EMBEDDING_WIDTH,
                    "precomputed row128 train",
                ) {
                    Ok(matrix) => (
                        Some(row_relative),
                        Some(MatrixArtifactV1 {
                            role: "row128_train_precomputed".into(),
                            rows: rows as u32,
                            columns: ROW_EMBEDDING_WIDTH as u32,
                            sha256: raw_sha256(&bytes),
                        }),
                        Some(matrix),
                        None,
                    ),
                    Err(reason) => (None, None, None, Some(reason)),
                }
            }
            Ok(_) => (
                None,
                None,
                None,
                Some("precomputed hyperion3train.vs is empty".into()),
            ),
            Err(error) => (
                None,
                None,
                None,
                Some(format!("precomputed row128 unavailable: {error}")),
            ),
        };
    let receipt = PrecomputedTrainFeatureReceiptV1 {
        schema: PrecomputedTrainFeatureReceiptV1::SCHEMA.into(),
        registry_sha256: registry_sha256.into(),
        dataset_id: dataset.dataset_id.clone(),
        dataset_id_sha256: raw_sha256(dataset.dataset_id.as_bytes()),
        source_training_sha256: dataset.train_sha256.clone(),
        stable_row_ids_sha256: hash_strings(&dataset.stable_row_ids),
        encoder_identity_sha256: registry.encoder.identity()?,
        row_count: rows as u32,
        raw_feature_count: dataset.feature_count,
        cell_relative_path,
        cell: cell_artifact,
        row128_relative_path,
        row128: row128_artifact,
        row128_unavailable_reason,
        test_artifacts_read: false,
    };
    receipt.validate()?;
    Ok(PrecomputedTrainFeatureBlocks {
        cell,
        row128,
        receipt,
    })
}

/// Read-only inspection used by registry freezing and tests.  This opens only
/// `train3.vs` and the optional `hyperion3train.vs` sibling for the named
/// train-only registry member.
pub fn inspect_precomputed_train_features(
    registry: &JopeHyperionTrainOnlyRegistryV2,
    registry_root: &Path,
    dataset_id: &str,
) -> Result<PrecomputedTrainFeatureReceiptV1, String> {
    registry.validate(registry_root)?;
    validate_opaque_id(dataset_id)?;
    let dataset = registry
        .datasets
        .iter()
        .find(|dataset| dataset.dataset_id == dataset_id)
        .ok_or_else(|| "opaque dataset ID is absent from the train-only registry".to_string())?;
    Ok(
        load_precomputed_train_features(registry, registry_root, &registry.identity()?, dataset)?
            .receipt,
    )
}
