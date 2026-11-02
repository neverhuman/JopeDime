use anyhow::{bail, Context, Result};
use feat_math::dsl::{DslExpr, ParamValue, Stage};
use feat_math::math_expr::MathExpr;
use feat_math::GenomeIr;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};

const ARCHIVE_MANIFEST: &str = "archive-manifest.v1.json";
const CORPUS_SCHEMA: &str = "jope-dime.score-corpus.v1";
const GENE_SCHEMA: &str = "jope-dime.gene-schema.v1";
const EXAMPLE_SCHEMA: &str = "jope-dime.score-example.v1";
const MAX_SNAPSHOT_ATTEMPTS: usize = 8;

// Hash the owned buffers delivered to the parser, rather than revisiting mutable
// file-backed memory after a separate digest check.
struct HashingReader<R> {
    inner: R,
    digest: Sha256,
}

impl<R: Read> HashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            digest: Sha256::new(),
        }
    }

    fn into_sha256(self) -> String {
        hex::encode(self.digest.finalize())
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(buffer)?;
        self.digest.update(&buffer[..count]);
        Ok(count)
    }
}

#[derive(Debug)]
struct ArchiveChanged(&'static str);

impl fmt::Display for ArchiveChanged {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "archive changed while reading: {}", self.0)
    }
}

impl std::error::Error for ArchiveChanged {}

#[derive(Clone, Debug, Deserialize)]
struct ArchiveManifest {
    schema: String,
    all_evaluations_jsonl: String,
    all_evaluations_jsonl_sha256: String,
    evaluation_records: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ExampleKey {
    cohort: String,
    dataset: String,
    semantic_sha256: String,
}

#[derive(Clone, Debug)]
struct Aggregate {
    canonical_genome: String,
    genome: GenomeExport,
    valid_rmse: Option<f64>,
    valid_count: usize,
    invalid_count: usize,
    runtimes: Vec<f64>,
    failure_classes: BTreeMap<String, usize>,
    provenance: Vec<Value>,
}

#[derive(Clone, Debug, Serialize)]
struct GenomeExport {
    form: String,
    stages: Vec<TypedStage>,
    actions: Vec<GeneAction>,
}

#[derive(Clone, Debug, Serialize)]
struct TypedStage {
    path: String,
    operation: String,
    parameters: Vec<TypedParameter>,
}

#[derive(Clone, Debug, Serialize)]
struct TypedParameter {
    name: String,
    value_type: String,
    value: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    expression_ast: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
struct GeneAction {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    operation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    categorical_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    numeric_value: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct ObservedSchema {
    operations: BTreeSet<String>,
    parameters: BTreeMap<(String, String), BTreeSet<String>>,
    categorical_values: BTreeMap<(String, String), BTreeSet<String>>,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeStatistics {
    count: usize,
    mean_seconds: Option<f64>,
    std_seconds: Option<f64>,
    min_seconds: Option<f64>,
    p50_seconds: Option<f64>,
    p90_seconds: Option<f64>,
    max_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct ScoreExample<'a> {
    schema: &'static str,
    cohort: &'a str,
    dataset: &'a str,
    semantic_sha256: &'a str,
    canonical_genome: &'a str,
    genome: &'a GenomeExport,
    rmse: Option<f64>,
    percentile: Option<f64>,
    success_rate: f64,
    valid_count: usize,
    invalid_count: usize,
    total_count: usize,
    failure_classes: &'a BTreeMap<String, usize>,
    runtime: RuntimeStatistics,
    provenance: &'a [Value],
}

pub(crate) fn materialize(
    archive_root: &Path,
    output_root: &Path,
    cohort: Option<&str>,
) -> Result<()> {
    let mut last_change = None;
    for _ in 0..MAX_SNAPSHOT_ATTEMPTS {
        match archive::read_and_materialize_filtered(archive_root, output_root, cohort, || {}) {
            Ok(path) => {
                println!("{}", path.display());
                return Ok(());
            }
            Err(error) if error.downcast_ref::<ArchiveChanged>().is_some() => {
                last_change = Some(error);
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_change.unwrap_or_else(|| ArchiveChanged("retry limit exceeded").into()))
        .context("could not capture a stable archive snapshot")
}

mod archive;
mod export;
mod publish;

#[cfg(test)]
mod tests;
