use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Best5TrainingTableV1 {
    pub table: DatasetTableV1,
    pub target_rows: Vec<TargetRowV1>,
    pub training_artifact: FrozenFileV1,
    pub metadata_artifact: FrozenFileV1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Best5SealedTableV1 {
    pub table: SealedTestTableV1,
    pub target_rows: Vec<TargetRowV1>,
    pub test_artifact: FrozenFileV1,
}

fn file_artifact(path: &Path, expected: &str) -> Result<FrozenFileV1, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if raw_sha256(&bytes) != expected {
        return Err(format!("{} changed after campaign freeze", path.display()));
    }
    Ok(FrozenFileV1 {
        path: path.to_string_lossy().into_owned(),
        sha256: expected.into(),
        bytes: bytes.len() as u64,
    })
}

/// Builds the only training table accepted by Best5. It opens the frozen
/// training and metadata files and deliberately never resolves the test path.
pub fn build_training_table(
    campaign: &CampaignFreezeV1,
    dataset_id: &str,
) -> Result<Best5TrainingTableV1, String> {
    campaign.validate()?;
    let dataset = campaign
        .datasets
        .iter()
        .find(|dataset| dataset.dataset_id == dataset_id)
        .ok_or_else(|| format!("dataset {dataset_id} is not in the campaign"))?;
    let root = Path::new(&campaign.sotarena_root);
    let train_path = safe_join(root, &dataset.train_relative_path)?;
    let metadata_path = safe_join(root, &dataset.metadata_relative_path)?;
    let training_artifact = file_artifact(&train_path, &dataset.train_sha256)?;
    let metadata_artifact = file_artifact(&metadata_path, &dataset.metadata_sha256)?;
    let bytes =
        fs::read(&train_path).map_err(|error| format!("read frozen training table: {error}"))?;
    let mut rows = Vec::new();
    let mut targets = Vec::new();
    for (ordinal, raw) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if raw.is_empty() && ordinal + 1 == bytes.split(|byte| *byte == b'\n').count() {
            continue;
        }
        if raw.is_empty() {
            return Err("training CSV contains an empty source row".into());
        }
        let line = std::str::from_utf8(raw)
            .map_err(|error| format!("training CSV is not UTF-8: {error}"))?;
        let columns = line.trim_end_matches('\r').split(',').collect::<Vec<_>>();
        if columns.len() != dataset.feature_count as usize + 1 {
            return Err(format!(
                "training row {ordinal} has {} columns, expected {}",
                columns.len(),
                dataset.feature_count + 1
            ));
        }
        let target = columns
            .last()
            .unwrap()
            .trim()
            .parse::<f64>()
            .map_err(|error| format!("parse target at row {ordinal}: {error}"))?;
        if !target.is_finite() {
            return Err(format!("target at row {ordinal} is nonfinite"));
        }
        let mut feature_bits = Vec::with_capacity(dataset.feature_count as usize * 8);
        for value in &columns[..columns.len() - 1] {
            let parsed = value
                .trim()
                .parse::<f64>()
                .map_err(|error| format!("parse feature at row {ordinal}: {error}"))?;
            if !parsed.is_finite() {
                return Err(format!("feature at row {ordinal} is nonfinite"));
            }
            feature_bits.extend_from_slice(&parsed.to_bits().to_le_bytes());
        }
        let stable_row_id = raw_sha256(
            [
                dataset.dataset_sha256.as_bytes(),
                &(ordinal as u64).to_le_bytes(),
                raw,
            ]
            .concat()
            .as_slice(),
        );
        let atomic_group = raw_sha256(&feature_bits);
        let split = SplitRowV1 {
            stable_row_id: stable_row_id.clone(),
            atomic_group: atomic_group.clone(),
            regression_target: Some(target),
            binary_target: None,
        };
        rows.push(DatasetRowV1 {
            split,
            csv_line: line.into(),
        });
        targets.push(TargetRowV1 {
            stable_row_id,
            atomic_group,
            target,
        });
    }
    if rows.len() != dataset.train_rows as usize {
        return Err(format!(
            "training table has {} rows, expected {}",
            rows.len(),
            dataset.train_rows
        ));
    }
    let table = DatasetTableV1 {
        task: GpTaskV1::Regression,
        dataset_sha256: dataset.dataset_sha256.clone(),
        training_sha256: dataset.train_sha256.clone(),
        rows,
    };
    table.validate()?;
    Ok(Best5TrainingTableV1 {
        table,
        target_rows: targets,
        training_artifact,
        metadata_artifact,
    })
}

/// Opens sealed bytes only after the caller has atomically published and
/// supplied the dataset-level claim identity.
pub fn build_claimed_sealed_table(
    campaign: &CampaignFreezeV1,
    dataset_id: &str,
    claim: &SealedClaimV1,
) -> Result<Best5SealedTableV1, String> {
    campaign.validate()?;
    claim.validate()?;
    let dataset = campaign
        .datasets
        .iter()
        .find(|dataset| dataset.dataset_id == dataset_id)
        .ok_or_else(|| format!("dataset {dataset_id} is not in the campaign"))?;
    if claim.campaign_freeze_sha256 != campaign.identity()?
        || claim.dataset_sha256 != dataset.dataset_sha256
    {
        return Err("sealed claim does not bind the requested campaign dataset".into());
    }
    let test_path = safe_join(
        Path::new(&campaign.sotarena_root),
        &dataset.test_relative_path,
    )?;
    // This is intentionally the first function in the Best5 layer that opens
    // the test path.
    let test_artifact = file_artifact(&test_path, &dataset.test_sha256)?;
    let bytes = fs::read(&test_path).map_err(|error| format!("read sealed test table: {error}"))?;
    let raw_rows = bytes.split(|byte| *byte == b'\n').collect::<Vec<_>>();
    let mut row_ids = Vec::new();
    let mut targets = Vec::new();
    for (ordinal, raw) in raw_rows.iter().enumerate() {
        if raw.is_empty() && ordinal + 1 == raw_rows.len() {
            continue;
        }
        if raw.is_empty() {
            return Err("sealed test CSV contains an empty source row".into());
        }
        let line = std::str::from_utf8(raw)
            .map_err(|error| format!("sealed test CSV is not UTF-8: {error}"))?;
        let columns = line.trim_end_matches('\r').split(',').collect::<Vec<_>>();
        if columns.len() != dataset.feature_count as usize + 1 {
            return Err(format!(
                "sealed test row {ordinal} has the wrong column count"
            ));
        }
        let target = columns
            .last()
            .unwrap()
            .trim()
            .parse::<f64>()
            .map_err(|error| format!("parse sealed target at row {ordinal}: {error}"))?;
        if !target.is_finite() {
            return Err(format!("sealed target at row {ordinal} is nonfinite"));
        }
        let stable_row_id = raw_sha256(
            [
                dataset.dataset_sha256.as_bytes(),
                b"sealed-test".as_slice(),
                &(ordinal as u64).to_le_bytes(),
                *raw,
            ]
            .concat()
            .as_slice(),
        );
        row_ids.push(stable_row_id.clone());
        targets.push(TargetRowV1 {
            stable_row_id,
            atomic_group: raw_sha256(b"sealed-test-row"),
            target,
        });
    }
    let table = SealedTestTableV1 {
        dataset_sha256: dataset.dataset_sha256.clone(),
        test_csv_path: test_path.to_string_lossy().into_owned(),
        test_sha256: dataset.test_sha256.clone(),
        test_row_ids: row_ids,
    };
    table.validate_metadata()?;
    Ok(Best5SealedTableV1 {
        table,
        target_rows: targets,
        test_artifact,
    })
}
