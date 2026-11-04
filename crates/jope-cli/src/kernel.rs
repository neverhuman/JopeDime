//! Fit, emit, and a tiny offline search over feat-math GP genomes.

use crate::admission::validate_dataset;
use crate::schema::{Dataset, Kind, Split, Task, MAX_FILE_BYTES};
use crate::storage::Directory;
use anyhow::{bail, Context, Result};
use feat_math::transpile::{emit, emit_cpp17, emit_javascript, EmitOptions};
use feat_math::{emit_fitted_cpp17, fit_legacy_numeric_bound, r2, GenomeIr, Model, RawTable};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

mod data;
mod emit;
mod fit;
mod numeric_emit;
mod search;

use data::*;
pub(crate) use data::{ensure_regression_support, read_dataset, unsupported_regression_reason};
pub use emit::{emit_genome, emit_genome_with_options, emit_genome_with_train};
pub use fit::fit_dataset;
pub(crate) use fit::fit_validated_dataset;
use numeric_emit::*;
pub use numeric_emit::{emit_genome_v3, emit_numeric_v2};
pub use search::search_dataset;
pub(crate) use search::search_validated_dataset;

/// Default ridge pipeline used by fit, emit, and related CLI operations.
pub const DEFAULT_GENOME: &str = "(pipe X (ridge))";

#[derive(Debug, Serialize)]
/// Training and validation scores for a fitted genome.
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
/// Metadata for one emitted source artifact.
pub struct Artifact {
    pub lang: String,
    pub bytes: usize,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Debug, Serialize)]
/// Summary of source artifacts emitted for a genome.
pub struct EmitReport {
    pub schema: &'static str,
    pub ok: bool,
    pub genome: String,
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Serialize)]
/// Best candidate and validation score from the bounded offline search.
pub struct SearchReport {
    pub schema: &'static str,
    pub ok: bool,
    pub dataset_id: String,
    pub genome: String,
    pub hold_r2: f64,
    pub considered: usize,
}

fn languages(lang: &str) -> Result<Vec<&'static str>> {
    Ok(match lang {
        "c17" | "c" => vec!["c17"],
        "rust" => vec!["rust"],
        "python" => vec!["python"],
        "cpp17" => vec!["cpp17"],
        "cpp17-recipe-v2" => vec!["cpp17-recipe-v2"],
        "cpp20" => vec!["cpp20"],
        "javascript" | "js" => vec!["javascript"],
        "typescript" | "ts" => vec!["typescript"],
        "go" | "golang" => vec!["go"],
        "cuda" | "cuda-cpp" | "cu" => vec!["cuda"],
        "all" => vec!["python", "rust", "cpp17", "javascript"],
        other => bail!(
            "unknown language {other} (rust|python|cpp17|cpp17-recipe-v2|cpp20|javascript|typescript|go|c17|cuda|all)"
        ),
    })
}

fn filename(lang: &str) -> &'static str {
    match lang {
        "python" => "model.py",
        "rust" => "model.rs",
        "cpp17" | "cpp17-recipe-v2" | "cpp20" => "model.cpp",
        "javascript" => "model.js",
        _ => "model.txt",
    }
}
