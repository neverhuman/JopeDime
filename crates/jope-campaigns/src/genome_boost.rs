//! Genome-boost feature emission for the `genome-boost-20260725-v1` campaign.
//!
//! Fits manifest-listed legacy `(pipe X ...)` genomes on train rows only,
//! lowers each fitted model through the verified fitted-C++17 stack, and
//! appends the resulting train/test prediction columns as feature blocks for
//! downstream DL backends. Every emitted receipt is permanently
//! `test_adaptive: true` and `benchmark_valid: false`: the source genomes were
//! evolved directly against each dataset's own test RMSE and the train-side
//! feature block is in-sample by construction (`in_sample_train_features`).
//!
//! Bit parity between the C++ runtime and `BoundModel::predict_table` is
//! mandatory on BOTH the train and test matrices, compared via `f64::to_bits`;
//! any mismatch is a hard error (nonzero exit). Only `supports_fitted_cpp17`
//! and fit failures are fail-open-to-next-genome, drawing replacements from
//! the manifest `spares` list when present; everything else fails closed.

use crate::data::read_regression_csv;
use anyhow::{bail, Context, Result};
use feat_math::{
    encode_fitted_cpp17, fit_legacy_numeric_bound, supports_fitted_cpp17,
    validate_fitted_cpp_codec, BoundModel, EncodedTable, FittedCppRuntime, GenomeIr,
};
use jope_dime_regressor::paths;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const MANIFEST_SCHEMA: &str = "jope-dime.genome-boost-manifest.v1";
const RECEIPT_SCHEMA: &str = "jope-dime.genome-boost-features.v1";
const TRAIN_FEATURES_FILE: &str = "genome-features-train.csv";
const TEST_FEATURES_FILE: &str = "genome-features-test.csv";
const RECEIPT_FILE: &str = "genome-features-receipt.v1.json";
jope_paths::once_str!(
    fn default_registry_str => paths::jope_registry()
);
const FEATURE_LINEAGE: &str = "fitted-cpp17 genome predictions: genomes were selected by direct \
     test-RMSE adaptation on this dataset (test-adaptive, benchmark-invalid); each genome is \
     refit on the full train split only, then predicted onto the same train rows (in-sample) \
     and onto the test feature rows";

#[derive(Debug, Deserialize)]
struct BoostManifest {
    schema: String,
    campaign_id: String,
    #[serde(default)]
    mode: Option<String>,
    datasets: Vec<ManifestDataset>,
}

#[derive(Debug, Deserialize)]
struct ManifestDataset {
    dataset_hash: String,
    #[serde(default)]
    registry_root: Option<PathBuf>,
    genomes: Vec<GenomeEntry>,
    #[serde(default)]
    spares: Vec<GenomeEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct GenomeEntry {
    canonical_genome: String,
    semantic_sha256: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    test_rmse: Option<f64>,
    #[serde(default)]
    rank: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureReceipt {
    schema: String,
    campaign_id: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    in_sample_train_features: bool,
    feature_lineage: String,
    mode: Option<String>,
    dataset_hash: String,
    registry_root: String,
    train_csv_sha256: String,
    test_csv_sha256: String,
    train_rows: usize,
    test_rows: usize,
    dataset_features: usize,
    requested_genomes: usize,
    emitted_columns: usize,
    columns: Vec<ColumnReceipt>,
    skips: Vec<SkipReceipt>,
    train_features_csv_sha256: String,
    test_features_csv_sha256: String,
    /// The whole pipeline is deterministic: legacy fits derive any internal
    /// streams from the genome text itself, so no external seed exists.
    seeds: SeedReceipt,
    elapsed_ms: u128,
    created_at_unix_ms: u128,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ColumnReceipt {
    column_index: usize,
    semantic_sha256: String,
    canonical_genome: String,
    genome_sha256: String,
    codec_sha256: String,
    bit_parity_train: bool,
    bit_parity_test: bool,
    source: Option<String>,
    selection_test_rmse: Option<f64>,
    selection_rank: Option<u64>,
    fit_ms: u128,
    encode_ms: u128,
    predict_ms: u128,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SkipReceipt {
    semantic_sha256: String,
    canonical_genome: String,
    stage: String,
    error: String,
    replaced_by_spare: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedReceipt {
    deterministic: bool,
    external_seed: Option<u64>,
}

pub(crate) fn run_features(manifest_path: &Path, output_dir: &Path) -> Result<()> {
    let manifest_bytes = fs::read(manifest_path)
        .with_context(|| format!("read genome-boost manifest {}", manifest_path.display()))?;
    let manifest: BoostManifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parse genome-boost manifest {}", manifest_path.display()))?;
    if manifest.schema != MANIFEST_SCHEMA {
        bail!(
            "unsupported genome-boost manifest schema '{}'; expected {MANIFEST_SCHEMA}",
            manifest.schema
        );
    }
    if manifest.datasets.is_empty() {
        bail!("genome-boost manifest lists no datasets");
    }
    fs::create_dir_all(output_dir)?;
    let runtime_root = output_dir.join(format!(".genome-boost-runtime-{}", std::process::id()));
    fs::create_dir_all(&runtime_root)?;
    let runtime = FittedCppRuntime::new(&runtime_root)?;
    let result = (|| -> Result<()> {
        for dataset in &manifest.datasets {
            emit_dataset(&manifest, dataset, output_dir, &runtime)?;
        }
        Ok(())
    })();
    let _ = fs::remove_dir_all(&runtime_root);
    result
}

fn emit_dataset(
    manifest: &BoostManifest,
    dataset: &ManifestDataset,
    output_dir: &Path,
    runtime: &FittedCppRuntime,
) -> Result<()> {
    let started = Instant::now();
    validate_dataset_hash(&dataset.dataset_hash)?;
    if dataset.genomes.is_empty() {
        bail!("dataset {} lists no genomes", dataset.dataset_hash);
    }
    let registry_root = dataset
        .registry_root
        .clone()
        .unwrap_or_else(|| PathBuf::from(default_registry_str()));
    let dataset_root = registry_root.join(&dataset.dataset_hash);
    let train_path = dataset_root.join("train.csv");
    let test_path = dataset_root.join("test.csv");
    let train_csv_sha256 = sha256_file(&train_path)?;
    let test_csv_sha256 = sha256_file(&test_path)?;
    let train = read_regression_csv(&train_path)?;
    // Registry test.csv includes the target as its last column; the loader
    // pops it, so `test.x` is exactly the feature block the workers see.
    let test = read_regression_csv(&test_path)?;
    if train.features != test.features {
        bail!(
            "dataset {}: train has {} features but test has {}",
            dataset.dataset_hash,
            train.features,
            test.features
        );
    }
    let train_table = numeric_raw_table(&train.x)?;
    let test_table = numeric_raw_table(&test.x)?;
    let encoded_train = EncodedTable::from_numeric_rows(&train.x)?;
    let encoded_test = EncodedTable::from_numeric_rows(&test.x)?;

    let requested = dataset.genomes.len();
    let mut queue: VecDeque<GenomeEntry> = dataset.genomes.iter().cloned().collect();
    let mut spares: VecDeque<GenomeEntry> = dataset.spares.iter().cloned().collect();
    let mut columns: Vec<ColumnReceipt> = Vec::new();
    let mut skips: Vec<SkipReceipt> = Vec::new();
    let mut train_columns: Vec<Vec<f64>> = Vec::new();
    let mut test_columns: Vec<Vec<f64>> = Vec::new();

    while let Some(entry) = queue.pop_front() {
        if columns.len() >= requested {
            break;
        }
        match fit_and_lower(&entry, &train, runtime) {
            Ok(lowered) => {
                // Inner errors are hard failures (broken toolchain, not an
                // unsupported genome): propagate with a nonzero exit.
                let fitted = lowered.with_context(|| {
                    format!(
                        "dataset {} genome {}: fitted C++17 lowering failed on an admitted model",
                        dataset.dataset_hash, entry.semantic_sha256
                    )
                })?;
                let predict_started = Instant::now();
                let cpp_train = fitted.cpp.predict(&encoded_train)?;
                let cpp_test = fitted.cpp.predict(&encoded_test)?;
                let rust_train = fitted.model.predict_table(&train_table)?;
                let rust_test = fitted.model.predict_table(&test_table)?;
                let predict_ms = predict_started.elapsed().as_millis();
                if cpp_train.len() != train.x.len() || cpp_test.len() != test.x.len() {
                    bail!(
                        "dataset {} genome {}: prediction row counts diverged",
                        dataset.dataset_hash,
                        entry.semantic_sha256
                    );
                }
                enforce_bit_parity(
                    &dataset.dataset_hash,
                    &entry.semantic_sha256,
                    "train",
                    &rust_train,
                    &cpp_train,
                )?;
                enforce_bit_parity(
                    &dataset.dataset_hash,
                    &entry.semantic_sha256,
                    "test",
                    &rust_test,
                    &cpp_test,
                )?;
                for (kind, values) in [("train", &cpp_train), ("test", &cpp_test)] {
                    if values.iter().any(|value| !value.is_finite()) {
                        bail!(
                            "dataset {} genome {}: non-finite {kind} feature value",
                            dataset.dataset_hash,
                            entry.semantic_sha256
                        );
                    }
                }
                columns.push(ColumnReceipt {
                    column_index: columns.len(),
                    semantic_sha256: entry.semantic_sha256.clone(),
                    canonical_genome: entry.canonical_genome.clone(),
                    genome_sha256: fitted.model.genome_sha256.clone(),
                    codec_sha256: fitted.codec_sha256.clone(),
                    bit_parity_train: true,
                    bit_parity_test: true,
                    source: entry.source.clone(),
                    selection_test_rmse: entry.test_rmse,
                    selection_rank: entry.rank,
                    fit_ms: fitted.fit_ms,
                    encode_ms: fitted.encode_ms,
                    predict_ms,
                });
                train_columns.push(cpp_train);
                test_columns.push(cpp_test);
            }
            Err(skip) => {
                let replaced = if let Some(spare) = spares.pop_front() {
                    queue.push_back(spare);
                    true
                } else {
                    false
                };
                eprintln!(
                    "genome-boost: dataset {} skipping genome {} at {}: {} (spare drawn: {})",
                    dataset.dataset_hash, entry.semantic_sha256, skip.stage, skip.error, replaced
                );
                skips.push(SkipReceipt {
                    semantic_sha256: entry.semantic_sha256.clone(),
                    canonical_genome: entry.canonical_genome.clone(),
                    stage: skip.stage,
                    error: skip.error,
                    replaced_by_spare: replaced,
                });
            }
        }
    }
    if columns.is_empty() {
        bail!(
            "dataset {}: every manifest genome was skipped; no feature block can be emitted",
            dataset.dataset_hash
        );
    }

    let dataset_dir = output_dir.join(&dataset.dataset_hash);
    fs::create_dir_all(&dataset_dir)?;
    let train_bytes = feature_csv_bytes(&train_columns, train.x.len())?;
    let test_bytes = feature_csv_bytes(&test_columns, test.x.len())?;
    write_immutable(&dataset_dir.join(TRAIN_FEATURES_FILE), &train_bytes)?;
    write_immutable(&dataset_dir.join(TEST_FEATURES_FILE), &test_bytes)?;

    let receipt = FeatureReceipt {
        schema: RECEIPT_SCHEMA.to_string(),
        campaign_id: manifest.campaign_id.clone(),
        test_adaptive: true,
        benchmark_valid: false,
        in_sample_train_features: true,
        feature_lineage: FEATURE_LINEAGE.to_string(),
        mode: manifest.mode.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        registry_root: registry_root.display().to_string(),
        train_csv_sha256,
        test_csv_sha256,
        train_rows: train.x.len(),
        test_rows: test.x.len(),
        dataset_features: train.features,
        requested_genomes: requested,
        emitted_columns: columns.len(),
        columns,
        skips,
        train_features_csv_sha256: sha256_bytes(&train_bytes),
        test_features_csv_sha256: sha256_bytes(&test_bytes),
        seeds: SeedReceipt {
            deterministic: true,
            external_seed: None,
        },
        elapsed_ms: started.elapsed().as_millis(),
        created_at_unix_ms: unix_ms(),
    };
    let mut receipt_bytes = serde_json::to_vec_pretty(&receipt)?;
    receipt_bytes.push(b'\n');
    write_immutable(&dataset_dir.join(RECEIPT_FILE), &receipt_bytes)?;
    println!(
        "genome-boost: dataset {} emitted {} feature columns ({} skips) into {}",
        dataset.dataset_hash,
        receipt.emitted_columns,
        receipt.skips.len(),
        dataset_dir.display()
    );
    Ok(())
}

struct FittedGenome {
    model: BoundModel,
    cpp: feat_math::FittedCppModel,
    codec_sha256: String,
    fit_ms: u128,
    encode_ms: u128,
}

struct SkipError {
    stage: String,
    error: String,
}

/// Fit on train rows only and lower through the fitted C++17 stack.
///
/// Parse/`supports_fitted_cpp17`/fit failures are returned as recoverable
/// skips, as are g++ rejections of the emitted source (an emitter admission
/// gap for specific genome families — observed 2026-07-25 on gpr
/// periodic-family kernels, where the lowering emits `d(...)`; recorded so a
/// spare takes the column and the gap is reported upstream). Codec-validate
/// and dlopen failures on a compiled model stay hard errors because they
/// indicate a broken toolchain, not an unsupported genome.
fn fit_and_lower(
    entry: &GenomeEntry,
    train: &crate::data::RegressionData,
    runtime: &FittedCppRuntime,
) -> std::result::Result<Result<FittedGenome>, SkipError> {
    let genome = GenomeIr::parse(&entry.canonical_genome).map_err(|error| SkipError {
        stage: "parse".to_string(),
        error: format!("{error:#}"),
    })?;
    let fit_started = Instant::now();
    let model =
        fit_legacy_numeric_bound(&genome, &train.x, &train.y).map_err(|error| SkipError {
            stage: "fit".to_string(),
            error: format!("{error:#}"),
        })?;
    let fit_ms = fit_started.elapsed().as_millis();
    supports_fitted_cpp17(&model).map_err(|error| SkipError {
        stage: "supports_fitted_cpp17".to_string(),
        error: format!("{error:#}"),
    })?;
    let encode_started = Instant::now();
    let codec = match encode_fitted_cpp17(&model) {
        Ok(codec) => codec,
        Err(error) if format!("{error:#}").contains("compilation failed") => {
            return Err(SkipError {
                stage: "fitted_cpp_compile".to_string(),
                error: format!("{error:#}"),
            });
        }
        Err(error) => return Ok(Err(error)),
    };
    Ok((|| -> Result<FittedGenome> {
        validate_fitted_cpp_codec(&codec.bytes)?;
        let encode_ms = encode_started.elapsed().as_millis();
        let cpp = runtime.load(&codec.bytes)?;
        Ok(FittedGenome {
            model,
            cpp,
            codec_sha256: codec.sha256,
            fit_ms,
            encode_ms,
        })
    })())
}

fn enforce_bit_parity(
    dataset_hash: &str,
    semantic_sha256: &str,
    split: &str,
    rust: &[f64],
    cpp: &[f64],
) -> Result<()> {
    if rust.len() != cpp.len() {
        bail!(
            "dataset {dataset_hash} genome {semantic_sha256}: {split} prediction lengths differ \
             (Rust {}, C++ {})",
            rust.len(),
            cpp.len()
        );
    }
    for (index, (expected, actual)) in rust.iter().zip(cpp).enumerate() {
        if expected.to_bits() != actual.to_bits() {
            bail!(
                "dataset {dataset_hash} genome {semantic_sha256}: {split} bit-parity FAILED at \
                 row {index}: Rust {expected:?} ({:#018x}) vs C++ {actual:?} ({:#018x})",
                expected.to_bits(),
                actual.to_bits()
            );
        }
    }
    Ok(())
}

fn numeric_raw_table(rows: &[Vec<f64>]) -> Result<feat_math::RawTable> {
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
fn feature_csv_bytes(columns: &[Vec<f64>], rows: usize) -> Result<Vec<u8>> {
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

fn validate_dataset_hash(hash: &str) -> Result<()> {
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
fn write_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
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

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Streamed (bounded memory); the same digest as hashing the whole file.
use jope_dime_regressor::selfplay_runtime::sha256_file;

fn unix_ms() -> u128 {
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
fn check_feature_block(bytes: &[u8], rows: usize, columns: usize) -> Result<()> {
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
