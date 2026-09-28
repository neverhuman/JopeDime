//! Fit, emit, and a tiny offline search over feat-math GP genomes.

use crate::admission::validate_dataset;
use crate::schema::{Dataset, Kind, Split, Task, MAX_FILE_BYTES};
use crate::storage::Directory;
use anyhow::{bail, Context, Result};
use feat_math::transpile::{emit, emit_cpp17, emit_javascript, EmitOptions};
use feat_math::{r2, GenomeIr, Model};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub const DEFAULT_GENOME: &str = "(pipe X (ridge))";

const SEARCH_CANDIDATES: [&str; 3] = [
    "(pipe X (ridge))",
    "(pipe X (standardize) (ridge))",
    "(pipe X (robust_scale) (ridge))",
];

#[derive(Debug, Serialize)]
pub struct FitReport {
    pub schema: &'static str,
    pub ok: bool,
    pub dataset_id: String,
    pub genome: String,
    pub train_rows: usize,
    pub hold_rows: usize,
    pub train_r2: f64,
    pub hold_r2: f64,
}

#[derive(Debug, Serialize)]
pub struct Artifact {
    pub lang: String,
    pub bytes: usize,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmitReport {
    pub schema: &'static str,
    pub ok: bool,
    pub genome: String,
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Serialize)]
pub struct SearchReport {
    pub schema: &'static str,
    pub ok: bool,
    pub dataset_id: String,
    pub genome: String,
    pub hold_r2: f64,
    pub considered: usize,
}

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

pub fn emit_genome(genome: &str, lang: &str, output: Option<&Path>) -> Result<EmitReport> {
    let parsed = GenomeIr::parse(genome).context("parse genome")?;
    let opts = EmitOptions {
        title: Some("jope emit".into()),
        ..EmitOptions::default()
    };
    let mut artifacts = Vec::new();
    for name in languages(lang)? {
        let source = match name {
            "python" | "rust" => {
                let emitted = emit(&parsed, &opts).context("emit rust/python")?;
                if name == "python" {
                    emitted.python
                } else {
                    emitted.rust
                }
            }
            "cpp17" => emit_cpp17(&parsed, &opts).context("emit C++17")?,
            // C++20 compiles the one C++ source; take it from the pipeline's own call so
            // `jope emit` and `jope pipeline run` write the same `model.cpp` bytes.
            "cpp20" => match jope_pipeline::emit::cpp_source(&parsed).context("emit C++20")? {
                Ok(source) => source,
                Err(reason) => bail!("{reason}"),
            },
            "javascript" => emit_javascript(&parsed, &opts).context("emit javascript")?,
            _ => unreachable!(),
        };
        let mut path = None;
        if let Some(root) = output {
            fs::create_dir_all(root).with_context(|| format!("create {}", root.display()))?;
            let file = root.join(filename(name));
            fs::write(&file, &source).with_context(|| format!("write {}", file.display()))?;
            path = Some(file.display().to_string());
        }
        artifacts.push(Artifact {
            lang: name.into(),
            bytes: source.len(),
            sha256: digest(&source),
            path,
        });
    }
    Ok(EmitReport {
        schema: "jope.emit.v1",
        ok: true,
        genome: parsed.canonical(),
        artifacts,
    })
}

pub fn search_dataset(path: &Path) -> Result<SearchReport> {
    let data = read_dataset(path)?;
    search_validated_dataset(&data)
}

/// Search one already-validated snapshot. Selection and its score use the validation split only.
pub(crate) fn search_validated_dataset(data: &Dataset) -> Result<SearchReport> {
    ensure_regression_support(data)?;
    let (train_x, train_y) = matrix(data, Split::Train)?;
    let (hold_x, hold_y) = holdout(data)?;
    let mut best_genome = String::new();
    let mut best_r2 = f64::NEG_INFINITY;
    for genome in SEARCH_CANDIDATES {
        let parsed = GenomeIr::parse(genome).context("parse search candidate")?;
        let model =
            feat_math::fit(&parsed, &train_x, &train_y).with_context(|| format!("fit {genome}"))?;
        let hold = score(&model, &hold_x, &hold_y).with_context(|| format!("evaluate {genome}"))?;
        if hold > best_r2 {
            best_r2 = hold;
            best_genome = parsed.canonical();
        }
    }
    anyhow::ensure!(!best_genome.is_empty(), "search produced no genome");
    Ok(SearchReport {
        schema: "jope.search.v1",
        ok: true,
        dataset_id: data.id.clone(),
        genome: best_genome,
        hold_r2: best_r2,
        considered: SEARCH_CANDIDATES.len(),
    })
}

fn languages(lang: &str) -> Result<Vec<&'static str>> {
    Ok(match lang {
        "rust" => vec!["rust"],
        "python" => vec!["python"],
        "cpp17" => vec!["cpp17"],
        "cpp20" => vec!["cpp20"],
        "javascript" | "js" => vec!["javascript"],
        "all" => vec!["python", "rust", "cpp17", "javascript"],
        other => bail!("unknown language {other} (rust|python|cpp17|cpp20|javascript|all)"),
    })
}

fn filename(lang: &str) -> &'static str {
    match lang {
        "python" => "model.py",
        "rust" => "model.rs",
        "cpp17" | "cpp20" => "model.cpp",
        "javascript" => "model.js",
        _ => "model.txt",
    }
}

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
fn holdout(data: &Dataset) -> Result<(feat_math::Rows, Vec<f64>)> {
    matrix(data, Split::Validation)
}

fn matrix(data: &Dataset, split: Split) -> Result<(feat_math::Rows, Vec<f64>)> {
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

fn score(model: &Model, x: &feat_math::Rows, y: &[f64]) -> Result<f64> {
    let predictions = model.predict(x);
    anyhow::ensure!(
        predictions.len() == y.len() && predictions.iter().all(|value| value.is_finite()),
        "evaluation produced missing or non-finite predictions"
    );
    let score = r2(y, &predictions);
    anyhow::ensure!(score.is_finite(), "evaluation produced a non-finite R²");
    Ok(score)
}

fn digest(source: &str) -> String {
    format!("{:x}", Sha256::digest(source.as_bytes()))
}
