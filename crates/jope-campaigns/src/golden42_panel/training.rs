use super::*;

pub(super) fn validate_dataset_hashes(registry: &Path) -> Result<()> {
    let root = registry.join(DATASET);
    if sha256_file(&root.join("train.csv"))? != TRAIN_SHA256
        || sha256_file(&root.join("test.csv"))? != TEST_SHA256
    {
        bail!("Golden42 V3 dataset hashes changed");
    }
    Ok(())
}

pub(super) fn numeric_raw_table(rows: &[Vec<f64>]) -> Result<feat_math::RawTable> {
    let width = rows.first().map_or(0, Vec::len);
    feat_math::RawTable::new(
        (0..width).map(|index| format!("c{index}")).collect(),
        rows.iter()
            .map(|row| row.iter().map(|value| value.to_string()).collect())
            .collect(),
    )
}

pub(super) fn campaign_training(rows: &[Vec<f64>]) -> Result<CampaignTraining> {
    let width = rows.first().map_or(0, Vec::len);
    if width == 0 || rows.iter().any(|row| row.len() != width) {
        bail!("campaign training matrix is empty or ragged");
    }
    let schema = DataSchema {
        columns: (0..width)
            .map(|index| ColumnSpec::numeric(format!("c{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    Ok(CampaignTraining {
        table: numeric_raw_table(rows)?,
        schema,
        task: TaskSpec::regression(Objective::Rmse)?,
    })
}

pub(super) fn fit_campaign_candidate(
    genome: &GenomeIr,
    training: &CampaignTraining,
    targets: &[f64],
) -> Result<BoundModel> {
    fit_table(
        genome,
        &training.task,
        &training.schema,
        &training.table,
        targets,
    )
}

pub(super) fn raw_rmse(targets: &[f64], predictions: &[f64]) -> Result<f64> {
    if targets.is_empty() || targets.len() != predictions.len() {
        bail!("RMSE tapes are empty or have different lengths");
    }
    let mut sum = 0.0;
    for (target, prediction) in targets.iter().zip(predictions) {
        if !target.is_finite() || !prediction.is_finite() {
            bail!("RMSE tape contains a non-finite value");
        }
        let error = target - prediction;
        sum += error * error;
    }
    Ok((sum / targets.len() as f64).sqrt())
}

pub(super) fn prediction_bits(predictions: &[f64]) -> Vec<u8> {
    predictions
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

pub(super) fn first_prediction_mismatch(
    expected: &[f64],
    actual: &[f64],
) -> Option<(usize, f64, f64)> {
    let shared = expected.len().min(actual.len());
    for index in 0..shared {
        if expected[index].to_bits() != actual[index].to_bits() {
            return Some((index, expected[index], actual[index]));
        }
    }
    (expected.len() != actual.len()).then(|| {
        let index = shared;
        (
            index,
            expected.get(index).copied().unwrap_or(f64::NAN),
            actual.get(index).copied().unwrap_or(f64::NAN),
        )
    })
}

pub(super) fn prediction_hash(predictions: &[f64]) -> String {
    sha256_bytes(&prediction_bits(predictions))
}

pub(super) fn context_hash(release_sha256: &str, candidate: &V3Candidate) -> String {
    sha256_bytes(
        format!(
            "golden42-v3-context-v1\0{DATASET}\0{TRAIN_SHA256}\0{TEST_SHA256}\0{release_sha256}\0{}",
            candidate.semantic_hash
        )
        .as_bytes(),
    )
}
