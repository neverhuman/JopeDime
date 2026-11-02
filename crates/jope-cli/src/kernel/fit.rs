use super::*;

/// Fit `genome` on the training split of a validated regression dataset.
pub fn fit_dataset(path: &Path, genome: &str) -> Result<FitReport> {
    let data = read_dataset(path)?;
    fit_validated_dataset(&data, genome)
}

/// Fit one already-validated snapshot; the caller never reopens its path.
pub(crate) fn fit_validated_dataset(data: &Dataset, genome: &str) -> Result<FitReport> {
    ensure_regression_support(data)?;
    let (train_x, train_y) = matrix(data, Split::Train)?;
    let (hold_x, hold_y) = holdout(data)?;
    let parsed = GenomeIr::parse(genome).context("parse genome")?;
    let model = feat_math::fit(&parsed, &train_x, &train_y).context("fit genome")?;
    Ok(FitReport {
        schema: "jope.fit.v1",
        ok: true,
        dataset_id: data.id.clone(),
        genome: parsed.canonical(),
        train_rows: train_x.len(),
        hold_rows: hold_x.len(),
        train_r2: score(&model, &train_x, &train_y)?,
        hold_r2: score(&model, &hold_x, &hold_y)?,
    })
}
