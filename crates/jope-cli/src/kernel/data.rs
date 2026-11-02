use super::*;

/// Read one dataset through the same bounded, no-symlink reader as admission (regular files
/// only, at most `MAX_FILE_BYTES`), then apply the full schema validation to those bytes.
pub(crate) fn read_dataset(path: &Path) -> Result<Dataset> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("dataset path needs a UTF-8 filename: {}", path.display()))?;
    let bytes = Directory::open(parent)
        .and_then(|directory| directory.read(filename, MAX_FILE_BYTES))
        .with_context(|| format!("read dataset {}", path.display()))?;
    let data = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse dataset {}", path.display()))?;
    validate_dataset(&data).with_context(|| format!("validate dataset {}", path.display()))?;
    Ok(data)
}

/// Why fit/search/route cannot use a dataset, or `None` when it is a supported regression
/// fixture with integer, non-null features.
pub(crate) fn unsupported_regression_reason(data: &Dataset) -> Option<String> {
    if data.task != Task::Regression {
        return Some(format!(
            "unsupported task {}: fit/search/route support regression only",
            data.task.name()
        ));
    }
    if data
        .columns
        .iter()
        .any(|column| column.kind != Kind::Integer || column.nullable)
    {
        return Some("fit/search/route currently admit integer, non-null features only".into());
    }
    None
}

pub(crate) fn ensure_regression_support(data: &Dataset) -> Result<()> {
    if let Some(reason) = unsupported_regression_reason(data) {
        bail!("{reason}");
    }
    Ok(())
}

/// The held-out score is the validation split. There is no fallback to test rows.
pub(super) fn holdout(data: &Dataset) -> Result<(feat_math::Rows, Vec<f64>)> {
    matrix(data, Split::Validation)
}

pub(super) fn matrix(data: &Dataset, split: Split) -> Result<(feat_math::Rows, Vec<f64>)> {
    ensure_regression_support(data)?;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for row in &data.rows {
        if row.split != split {
            continue;
        }
        let mut features = Vec::with_capacity(data.columns.len());
        for feature in &row.features {
            features.push(
                feature
                    .as_i64()
                    .with_context(|| format!("non-integer feature in {}", data.id))?
                    as f64,
            );
        }
        xs.push(features);
        ys.push(row.target as f64);
    }
    anyhow::ensure!(!xs.is_empty(), "no {split:?} rows");
    Ok((xs, ys))
}

pub(super) fn score(model: &Model, x: &feat_math::Rows, y: &[f64]) -> Result<f64> {
    let predictions = model.predict(x)?;
    anyhow::ensure!(
        predictions.len() == y.len() && predictions.iter().all(|value| value.is_finite()),
        "evaluation produced missing or non-finite predictions"
    );
    let score = r2(y, &predictions);
    anyhow::ensure!(score.is_finite(), "evaluation produced a non-finite R²");
    Ok(score)
}

pub(super) fn digest(source: &str) -> String {
    format!("{:x}", Sha256::digest(source.as_bytes()))
}
