use super::*;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn numeric_raw_table(rows: &[Vec<f64>]) -> Result<feat_math::RawTable> {
    let width = rows.first().map_or(0, Vec::len);
    feat_math::RawTable::new(
        (0..width).map(|index| format!("c{index}")).collect(),
        rows.iter()
            .map(|row| row.iter().map(|value| value.to_string()).collect())
            .collect(),
    )
}

/// Column-major predictions to row-major CSV bytes using Rust `{:?}`
/// shortest-round-trip formatting. Downstream Python must never rewrite these
/// numeric bytes.
pub(crate) fn feature_csv_bytes(columns: &[Vec<f64>], rows: usize) -> Result<Vec<u8>> {
    for column in columns {
        if column.len() != rows {
            bail!("feature column length {} != row count {rows}", column.len());
        }
    }
    let mut bytes = Vec::new();
    for row in 0..rows {
        let mut line = String::new();
        for (index, column) in columns.iter().enumerate() {
            if index > 0 {
                line.push(',');
            }
            line.push_str(&format!("{:?}", column[row]));
        }
        line.push('\n');
        bytes.extend_from_slice(line.as_bytes());
    }
    Ok(bytes)
}

pub(crate) fn validate_dataset_hash(hash: &str) -> Result<()> {
    if hash.is_empty()
        || hash.len() > 64
        || !hash.chars().all(|c| c.is_ascii_hexdigit() || c == '_')
        || hash.contains("..")
    {
        bail!("invalid dataset hash '{hash}'");
    }
    Ok(())
}

/// Immutable write: refuse to change existing bytes, accept identical reruns.
pub(crate) fn write_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        let existing = fs::read(path)?;
        if existing == bytes {
            return Ok(());
        }
        bail!(
            "refusing to overwrite existing {} with different bytes",
            path.display()
        );
    }
    let temporary = path.with_extension("tmp-genome-boost");
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Streamed (bounded memory); the same digest as hashing the whole file.
pub(crate) fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

pub(crate) fn validate_features(output_dir: &Path) -> Result<()> {
    let mut validated = 0usize;
    let mut entries: Vec<PathBuf> = fs::read_dir(output_dir)
        .with_context(|| format!("read output dir {}", output_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir() && path.join(RECEIPT_FILE).is_file())
        .collect();
    entries.sort();
    if entries.is_empty() {
        bail!(
            "no genome-boost feature receipts found under {}",
            output_dir.display()
        );
    }
    for dataset_dir in entries {
        let receipt_bytes = fs::read(dataset_dir.join(RECEIPT_FILE))?;
        let receipt: FeatureReceipt = serde_json::from_slice(&receipt_bytes)
            .with_context(|| format!("parse receipt in {}", dataset_dir.display()))?;
        if receipt.schema != RECEIPT_SCHEMA {
            bail!("{}: unexpected receipt schema", dataset_dir.display());
        }
        if !receipt.test_adaptive
            || receipt.benchmark_valid
            || !receipt.in_sample_train_features
            || receipt.feature_lineage.is_empty()
        {
            bail!(
                "{}: receipt labels are wrong; test_adaptive/benchmark_valid/\
                 in_sample_train_features must be true/false/true",
                dataset_dir.display()
            );
        }
        let directory_name = dataset_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if directory_name != receipt.dataset_hash {
            bail!(
                "{}: directory name does not match receipt dataset hash {}",
                dataset_dir.display(),
                receipt.dataset_hash
            );
        }
        let train_bytes = fs::read(dataset_dir.join(TRAIN_FEATURES_FILE))?;
        let test_bytes = fs::read(dataset_dir.join(TEST_FEATURES_FILE))?;
        if sha256_bytes(&train_bytes) != receipt.train_features_csv_sha256 {
            bail!(
                "{}: train feature block hash mismatch",
                dataset_dir.display()
            );
        }
        if sha256_bytes(&test_bytes) != receipt.test_features_csv_sha256 {
            bail!(
                "{}: test feature block hash mismatch",
                dataset_dir.display()
            );
        }
        let registry_root = PathBuf::from(&receipt.registry_root);
        let dataset_root = registry_root.join(&receipt.dataset_hash);
        if sha256_file(&dataset_root.join("train.csv"))? != receipt.train_csv_sha256 {
            bail!("{}: registry train.csv changed", dataset_dir.display());
        }
        if sha256_file(&dataset_root.join("test.csv"))? != receipt.test_csv_sha256 {
            bail!("{}: registry test.csv changed", dataset_dir.display());
        }
        check_feature_block(&train_bytes, receipt.train_rows, receipt.emitted_columns)
            .with_context(|| format!("{}: train feature block", dataset_dir.display()))?;
        check_feature_block(&test_bytes, receipt.test_rows, receipt.emitted_columns)
            .with_context(|| format!("{}: test feature block", dataset_dir.display()))?;
        if receipt.columns.len() != receipt.emitted_columns {
            bail!("{}: column receipt count mismatch", dataset_dir.display());
        }
        for (index, column) in receipt.columns.iter().enumerate() {
            if column.column_index != index {
                bail!("{}: column indices out of order", dataset_dir.display());
            }
            if !column.bit_parity_train || !column.bit_parity_test {
                bail!(
                    "{}: column {} lacks recorded bit parity",
                    dataset_dir.display(),
                    index
                );
            }
        }
        if receipt.emitted_columns
            + receipt
                .skips
                .iter()
                .filter(|s| !s.replaced_by_spare)
                .count()
            < receipt.requested_genomes
        {
            bail!(
                "{}: emitted columns plus unreplaced skips do not cover the requested genomes",
                dataset_dir.display()
            );
        }
        validated += 1;
        println!(
            "genome-boost validate: {} ok ({} columns, {} skips)",
            receipt.dataset_hash,
            receipt.emitted_columns,
            receipt.skips.len()
        );
    }
    println!("genome-boost validate: {validated} dataset feature blocks verified");
    Ok(())
}

/// Shape, finiteness, and shortest-round-trip formatting check for a feature
/// block: every field must parse as a finite f64 and reformat via `{:?}` to
/// exactly the stored bytes.
pub(crate) fn check_feature_block(bytes: &[u8], rows: usize, columns: usize) -> Result<()> {
    let text = std::str::from_utf8(bytes).context("feature block is not UTF-8")?;
    let mut row_count = 0usize;
    for (line_no, line) in text.lines().enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != columns {
            bail!(
                "line {}: expected {} columns, found {}",
                line_no + 1,
                columns,
                fields.len()
            );
        }
        for field in fields {
            let value: f64 = field
                .parse()
                .with_context(|| format!("line {}: invalid f64 '{field}'", line_no + 1))?;
            if !value.is_finite() {
                bail!("line {}: non-finite feature value", line_no + 1);
            }
            if format!("{value:?}") != field {
                bail!(
                    "line {}: field '{field}' is not shortest-round-trip formatted",
                    line_no + 1
                );
            }
        }
        row_count += 1;
    }
    if row_count != rows {
        bail!("expected {rows} rows, found {row_count}");
    }
    Ok(())
}
