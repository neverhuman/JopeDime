use super::{prepare_jope_foundation_rows, FoundationDatasetInputV1};
use anyhow::{bail, Context, Result};
use std::path::Path;

/// Parse a headered CSV.  The target column defaults to the final column; empty and
/// non-numeric feature cells are represented as NaN and are handled by Starlight's
/// train-only imputation path.
pub fn read_dataset_csv(
    path: &Path,
    target_column: Option<&str>,
) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("open {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let target = match target_column {
        Some(name) => headers
            .iter()
            .position(|header| header == name)
            .with_context(|| format!("target column {name:?} not found"))?,
        None => headers.len().saturating_sub(1),
    };
    let mut x = Vec::new();
    let mut y = Vec::new();
    for record in reader.records() {
        let record = record?;
        if record.len() != headers.len() {
            bail!("{} contains a ragged CSV row", path.display());
        }
        let mut row = Vec::with_capacity(record.len() - 1);
        for (index, field) in record.iter().enumerate() {
            if index == target {
                let value = field
                    .trim()
                    .parse::<f32>()
                    .with_context(|| format!("invalid target in {}", path.display()))?;
                if !value.is_finite() {
                    bail!("non-finite target in {}", path.display());
                }
                y.push(value);
            } else {
                row.push(field.trim().parse::<f32>().unwrap_or(f32::NAN));
            }
        }
        x.push(row);
    }
    Ok((x, y))
}

/// Reproduce the reviewed foundation input preparation exactly: a headerless
/// train matrix with the target in the final column, non-finites replaced by
/// zero, feature-wise min/max scaling fit on train, and NumPy PCG64 splitting.
pub fn read_jope_train_csv(
    path: &Path,
) -> Result<(FoundationDatasetInputV1, Vec<usize>, Vec<usize>)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut rows = Vec::<Vec<f64>>::new();
    let mut width = None;
    for record in reader.records() {
        let record = record.with_context(|| format!("read {}", path.display()))?;
        let expected = *width.get_or_insert(record.len());
        if expected < 2 || record.len() != expected {
            bail!(
                "{} must be rectangular with at least one feature",
                path.display()
            );
        }
        rows.push(
            record
                .iter()
                .map(|field| field.trim().parse::<f64>().unwrap_or(f64::NAN))
                .collect(),
        );
    }
    width.context("foundation train CSV is empty")?;
    let prepared = prepare_jope_foundation_rows(&rows)?;
    Ok((
        prepared.input,
        prepared.context_indices,
        prepared.query_indices,
    ))
}
