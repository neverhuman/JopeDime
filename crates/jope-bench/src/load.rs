//! Datasets in the catalog layout: a directory with `metadata.json` (repository layout) or
//! `manifest.json` (cache layout) describing a headerless, fully numeric `train.csv` whose last
//! column is the target. The bytes are verified against the manifest before anything is parsed.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use feat_math::Rows;
use jope_datasets::catalog::CatalogRow;
use sha2::{Digest, Sha256};

/// A loaded dataset: the catalog row plus the training rows.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// Directory the dataset was read from.
    pub dir: PathBuf,
    /// The catalog row (`metadata.json` / `manifest.json`).
    pub row: CatalogRow,
    /// Feature rows of `train.csv` (target column removed).
    pub x: Rows,
    /// Target column of `train.csv`.
    pub y: Vec<f64>,
    /// SHA-256 of the `train.csv` bytes.
    pub train_sha256: String,
}

impl Dataset {
    /// The catalog task as its JSON spelling (`regression`, `binary`, ...).
    pub fn task(&self) -> String {
        serde_json::to_value(self.row.task)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default()
    }
}

/// Lower-case hex SHA-256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Parse a headerless numeric CSV, target last. Every cell must be a finite `f64`; every row
/// must have the same width; at least two columns.
pub fn parse_numeric_csv(text: &str, what: &str) -> Result<(Rows, Vec<f64>)> {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut width = None;
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut values = Vec::new();
        for (column, cell) in line.split(',').enumerate() {
            let value: f64 = cell.trim().parse().with_context(|| {
                format!(
                    "{what} row {} column {column}: {cell:?} is not a number",
                    index + 1
                )
            })?;
            ensure!(
                value.is_finite(),
                "{what} row {} column {column}: non-finite value",
                index + 1
            );
            values.push(value);
        }
        match width {
            None => {
                ensure!(values.len() >= 2, "{what}: fewer than two columns");
                width = Some(values.len());
            }
            Some(w) => ensure!(
                values.len() == w,
                "{what} row {}: {} columns, expected {w}",
                index + 1,
                values.len()
            ),
        }
        let target = values.pop().expect("at least two columns");
        x.push(values);
        y.push(target);
    }
    ensure!(!x.is_empty(), "{what}: no rows");
    Ok((x, y))
}

/// `metadata.json` (repository layout) or `manifest.json` (cache layout) under `dir`.
pub fn find_manifest(dir: &Path) -> Option<PathBuf> {
    ["metadata.json", "manifest.json"]
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

/// Load one dataset directory, verifying `train.csv` against the manifest.
pub fn load_dataset(dir: &Path) -> Result<Dataset> {
    let manifest = find_manifest(dir)
        .with_context(|| format!("{}: no metadata.json or manifest.json", dir.display()))?;
    let row = jope_datasets::cache::read_manifest(&manifest)?;
    let train_path = dir.join("train.csv");
    let bytes =
        std::fs::read(&train_path).with_context(|| format!("read {}", train_path.display()))?;
    let train_sha256 = sha256_hex(&bytes);
    ensure!(
        train_sha256 == row.files.train.sha256,
        "{}: train.csv sha256 {train_sha256} does not match the manifest's {}",
        dir.display(),
        row.files.train.sha256
    );
    let text = std::str::from_utf8(&bytes)
        .with_context(|| format!("{} is not UTF-8", train_path.display()))?;
    let (x, y) = parse_numeric_csv(text, &train_path.display().to_string())?;
    if let Some(rows) = row.files.train.rows {
        ensure!(
            x.len() as u64 == rows,
            "{}: train.csv has {} rows, manifest says {rows}",
            dir.display(),
            x.len()
        );
    }
    let columns = x[0].len() as u64 + 1;
    ensure!(
        columns == row.shape.columns,
        "{}: train.csv has {columns} columns, manifest says {}",
        dir.display(),
        row.shape.columns
    );
    if x.len() < 16 {
        bail!(
            "{}: {} training rows is too few to cut",
            dir.display(),
            x.len()
        );
    }
    Ok(Dataset {
        dir: dir.to_path_buf(),
        row,
        x,
        y,
        train_sha256,
    })
}

/// Dataset directories under `root`: `root` itself when it holds a manifest and a `train.csv`,
/// otherwise every immediate subdirectory that does, sorted by name.
pub fn discover(root: &Path) -> Result<Vec<PathBuf>> {
    let is_dataset = |dir: &Path| find_manifest(dir).is_some() && dir.join("train.csv").is_file();
    if is_dataset(root) {
        return Ok(vec![root.to_path_buf()]);
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .with_context(|| format!("read {}", root.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir() && is_dataset(path))
        .collect();
    dirs.sort();
    ensure!(
        !dirs.is_empty(),
        "{}: no dataset directory (metadata.json + train.csv) found",
        root.display()
    );
    Ok(dirs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_csv_is_strict() {
        let (x, y) = parse_numeric_csv("1,2,3\n4,5,6\n\n", "t").unwrap();
        assert_eq!(x, vec![vec![1.0, 2.0], vec![4.0, 5.0]]);
        assert_eq!(y, vec![3.0, 6.0]);
        assert!(parse_numeric_csv("1,2\n3\n", "t").is_err(), "ragged");
        assert!(parse_numeric_csv("1,x\n", "t").is_err(), "non-numeric");
        assert!(parse_numeric_csv("1,inf\n", "t").is_err(), "non-finite");
        assert!(parse_numeric_csv("1\n", "t").is_err(), "one column");
        assert!(parse_numeric_csv("\n", "t").is_err(), "empty");
    }
}
