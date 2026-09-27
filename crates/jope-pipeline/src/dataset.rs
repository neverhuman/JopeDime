//! Stage 1: resolve one dataset offline and prove its bytes.
//!
//! A selector is either a 16-hex catalog id (`synth_`-prefixed or not), resolved
//! in the local cache under `$JOPE_HOME/datasets`, or a directory. The committed
//! `datasets/test/<slug>/` layout names its catalog row `metadata.json`; the
//! cache's `by-id/<hash>/` layout names the same document `manifest.json`. Both
//! are probed. Nothing on this path can reach the network: no `Store` is ever
//! constructed, so there is nothing to dial out with.

use anyhow::{bail, ensure, Context, Result};
use jope_datasets::{
    cache, catalog::CatalogRow, dataset_dir_hash, hash::digest_file, paths::Cache,
};
use std::path::{Path, PathBuf};

/// One file of the dataset, hashed and matched against the catalog row.
#[derive(Clone, Debug)]
pub struct VerifiedFile {
    pub name: String,
    pub sha256: String,
    pub bytes: u64,
}

/// A resolved, byte-verified dataset.
#[derive(Clone, Debug)]
pub struct Resolved {
    pub dir: PathBuf,
    pub row: CatalogRow,
    pub files: Vec<VerifiedFile>,
}

impl Resolved {
    pub fn train_csv(&self) -> PathBuf {
        self.dir.join("train.csv")
    }
}

/// Resolve `selector` without any network access, then verify every file the
/// catalog row references against its recorded sha256.
pub fn resolve(selector: &str) -> Result<Resolved> {
    let (dir, row) = locate(selector)?;
    let mut files = Vec::new();
    for (name, entry) in row.files.entries() {
        let path = dir.join(name);
        let digest = digest_file(&path).with_context(|| format!("read {}", path.display()))?;
        ensure!(
            digest.sha256 == entry.sha256,
            "{}: sha256 is {}, the catalog records {}",
            path.display(),
            digest.sha256,
            entry.sha256
        );
        ensure!(
            digest.bytes == entry.bytes,
            "{}: {} bytes, the catalog records {}",
            path.display(),
            digest.bytes,
            entry.bytes
        );
        files.push(VerifiedFile {
            name: name.to_owned(),
            sha256: digest.sha256,
            bytes: digest.bytes,
        });
    }
    Ok(Resolved { dir, row, files })
}

fn locate(selector: &str) -> Result<(PathBuf, CatalogRow)> {
    let as_path = Path::new(selector);
    if as_path.is_dir() {
        // The committed repo layout first, then the cache's registry layout.
        for name in ["metadata.json", "manifest.json"] {
            let candidate = as_path.join(name);
            if candidate.is_file() {
                return Ok((as_path.to_path_buf(), cache::read_manifest(&candidate)?));
            }
        }
        bail!(
            "{}: neither metadata.json nor manifest.json is present",
            as_path.display()
        );
    }
    let Some(hash) = dataset_dir_hash(selector) else {
        bail!(
            "{selector}: expected a dataset directory or a 16-hex catalog id \
             (for example 1cb97b1d7314d43d)"
        );
    };
    let dir = Cache::from_env().by_id_dir(hash);
    let manifest = dir.join("manifest.json");
    ensure!(
        manifest.is_file(),
        "{hash}: not in the local dataset cache ({}); link it with \
         `jope datasets link --from datasets/test` or pass the dataset directory",
        dir.display()
    );
    Ok((dir, cache::read_manifest(&manifest)?))
}

/// Load the training partition as complete numeric observations: every row is
/// `features..., target`, in file order.
///
/// The committed CSVs are headerless, comma separated, uniform width and fully
/// numeric, and `verify_repo` only proves that of each file's *first* line — so
/// every field is parsed and every row's width is checked here rather than
/// trusted.
pub fn training_rows(resolved: &Resolved) -> Result<Vec<Vec<f64>>> {
    numeric_rows(
        &resolved.train_csv(),
        resolved.row.shape.columns as usize,
        resolved.row.split.train_rows,
    )
}

/// The sealed test partition, parsed and checked exactly like [`training_rows`]. Only
/// [`crate::test_score`] reads it: nothing that searches, selects, seeds or learns ever does.
pub fn test_rows(resolved: &Resolved) -> Result<Vec<Vec<f64>>> {
    numeric_rows(
        &resolved.dir.join("test.csv"),
        resolved.row.shape.columns as usize,
        resolved.row.split.test_rows,
    )
}

fn numeric_rows(path: &Path, columns: usize, expected_rows: u64) -> Result<Vec<Vec<f64>>> {
    ensure!(
        columns >= 2,
        "{}: {columns} columns; features and a target are required",
        path.display()
    );
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut rows = Vec::with_capacity(expected_rows as usize);
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row = line
            .split(',')
            .map(|field| field.trim().parse::<f64>())
            .collect::<std::result::Result<Vec<f64>, _>>()
            .with_context(|| {
                format!(
                    "{}:{}: expected numeric CSV",
                    path.display(),
                    line_index + 1
                )
            })?;
        ensure!(
            row.len() == columns,
            "{}:{}: {} columns, the catalog records {columns}",
            path.display(),
            line_index + 1,
            row.len()
        );
        ensure!(
            row.iter().all(|value| value.is_finite()),
            "{}:{}: nonfinite value",
            path.display(),
            line_index + 1
        );
        rows.push(row);
    }
    ensure!(
        rows.len() as u64 == expected_rows,
        "{}: {} rows, the catalog records {}",
        path.display(),
        rows.len(),
        expected_rows
    );
    Ok(rows)
}
