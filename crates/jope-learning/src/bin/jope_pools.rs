//! `jope-pools`: adapt sources, freeze candidate pools, then evaluate selectors (shadow-only).
//!
//! Outputs are created, never overwritten: an existing output path is an error.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use jope_learning::canon::sha256_hex;
use jope_learning::pools::evaluate::{Config, Inputs, Unavailable};
use jope_learning::pools::pool::{
    export_views, freeze, FreezeRule, Manifest, MeasurementRow, Pool, Reference,
};
use jope_learning::pools::selectors::{External, ExternalDocument, Features, Selector};
use jope_learning::pools::source;
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "jope-pools",
    about = "J9 frozen candidate pools and shadow selector evaluation"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Adapt an R22 native coverage document (`jope.r22-lpi-scratch-coverage/v1`) into
    /// measurement rows for one partition; contexts listed in `--exclude-contexts` are dropped.
    IngestCoverage {
        #[arg(long)]
        coverage: PathBuf,
        #[arg(long, default_value = "train")]
        partition: String,
        /// Files of context ids (one per line) to leave out, e.g. an earlier held-out set.
        #[arg(long)]
        exclude_contexts: Vec<PathBuf>,
        /// Files of `context_id<TAB>program_sha256` members to leave out of their pools, e.g. each
        /// context's own incumbent program.
        #[arg(long)]
        exclude_pairs: Vec<PathBuf>,
        /// Identity of the measurement protocol (the native snapshot spec sha256).
        #[arg(long)]
        protocol_sha256: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Export raw context vectors for the pooled contexts from a joined native roster
    /// (`jope.joined-native-roster/v1` rows plus their f32 pack), checking every digest.
    IngestRosterFeatures {
        #[arg(long)]
        roster: PathBuf,
        #[arg(long)]
        pack: PathBuf,
        /// The pool manifest whose contexts need vectors.
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Export outcome-free pool views (identities and tokens only) for external scorers.
    Views {
        #[arg(long)]
        pools: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Build frozen pools from measurement rows (JSON Lines of `MeasurementRow`).
    Freeze {
        #[arg(long)]
        rows: PathBuf,
        /// New directory for `pools.jsonl` and `manifest.json`; must not exist.
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 9)]
        min_pool: usize,
        #[arg(long)]
        max_pool: Option<usize>,
        #[arg(long, default_value = "jope.j9-pool-cap/v1")]
        salt: String,
        #[arg(long, default_value_t = 2)]
        min_distinct_measured: usize,
        #[arg(long)]
        source_grade: String,
    },
    /// Evaluate selectors on frozen pools; writes one receipt.
    Evaluate {
        #[arg(long)]
        pools: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, value_enum)]
        reference: Reference,
        #[arg(long)]
        /// nested/v3 accepts context-id-bytes/v1 descriptors; unbound vectors are unavailable.
        context_features: Option<PathBuf>,
        #[arg(long)]
        program_features: Option<PathBuf>,
        /// Optional S3 `gp_distilled/v1` Parquet archive for fold-isolated cross-family program priors.
        #[arg(long)]
        s3_prior_parquet: Option<PathBuf>,
        /// External score documents (`jope.j9-external-scores/v1`), repeatable.
        #[arg(long)]
        external: Vec<PathBuf>,
        /// `name=reason` for a CLOVER selector that cannot run on this source, repeatable.
        #[arg(long)]
        unavailable: Vec<String>,
        #[arg(long, default_value_t = jope_learning::pools::metrics::DEFAULT_BUDGET)]
        budget: usize,
        #[arg(long, default_value_t = 5)]
        folds: usize,
        #[arg(long, default_value = "jope.j9-folds/v1")]
        fold_salt: String,
        #[arg(long, default_value_t = 2.0)]
        prior_weight: f64,
        #[arg(long, default_value_t = 16)]
        knn_k: usize,
        #[arg(long, default_value_t = 10.0)]
        ridge_lambda: f64,
        #[arg(long, default_value_t = 2000)]
        bootstrap_draws: usize,
        #[arg(long, default_value_t = 20_261_013)]
        bootstrap_seed: u64,
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,
        /// Full 40-hex commit of the source this binary was built from; recorded in the output.
        #[arg(long)]
        source_commit: String,
        /// New receipt path; must not exist.
        #[arg(long)]
        out: PathBuf,
    },
    /// Run the multi-round `gp_distilled/v1` self-play flywheel over frozen pools, writing
    /// per-round Parquet shards and one flywheel receipt.
    FlywheelDistill {
        #[arg(long)]
        pools: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, value_enum, default_value = "pool-floor")]
        reference: Reference,
        #[arg(long)]
        /// Verified context-id-bytes/v1 descriptors; unbound vectors refuse before archive writes.
        context_features: PathBuf,
        #[arg(long)]
        program_features: Option<PathBuf>,
        /// Optional S3 `gp_distilled/v1` Parquet archive for fold-isolated cross-family program priors.
        #[arg(long)]
        s3_prior_parquet: Option<PathBuf>,
        /// External score documents (`jope.j9-external-scores/v1`), repeatable.
        #[arg(long)]
        external: Vec<PathBuf>,
        #[arg(long, default_value_t = 1)]
        budget: usize,
        #[arg(long, value_delimiter = ',', default_values_t = vec![1usize, 4, 8])]
        eval_budgets: Vec<usize>,
        #[arg(long, default_value_t = 5)]
        rounds: usize,
        #[arg(long, default_value_t = 2)]
        initial_budget: usize,
        #[arg(long, default_value_t = 2)]
        step_budget: usize,
        #[arg(long, default_value_t = 64)]
        proposal_pool_k: usize,
        #[arg(long, default_value = "shadow_pool_q")]
        score_name: String,
        #[arg(long, default_value_t = 5)]
        folds: usize,
        #[arg(long, default_value = "jope.j9-folds/v1")]
        fold_salt: String,
        #[arg(long, default_value_t = 2.0)]
        prior_weight: f64,
        #[arg(long, default_value_t = 16)]
        knn_k: usize,
        #[arg(long, default_value_t = 10.0)]
        ridge_lambda: f64,
        #[arg(long, default_value_t = 2000)]
        bootstrap_draws: usize,
        #[arg(long, default_value_t = 20_261_013)]
        bootstrap_seed: u64,
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,
        #[arg(long)]
        parquet_dir: Option<PathBuf>,
        /// Full 40-hex commit of the source this binary was built from; recorded in the output.
        #[arg(long)]
        source_commit: String,
        /// New receipt path; must not exist.
        #[arg(long)]
        out: PathBuf,
    },
    /// Offline self-play round qualification report (`jope.selfplay-round-qualification/v1`).
    QualifyRounds {
        /// JSON file of `RoundQualPlan`.
        #[arg(long)]
        plan: PathBuf,
        /// JSON file of `ExperimentIdentity` (`clover.experiment/v1`).
        #[arg(long)]
        identity: PathBuf,
        /// JSON array or JSON Lines of `clover.selfplay-ratchet/v1` outputs.
        #[arg(long)]
        ratchets: PathBuf,
        /// JSON array or JSON Lines of `jope.pipeline-test-score/v1` receipts.
        #[arg(long)]
        test_scores: PathBuf,
        /// JSON array or JSON Lines of `ParityRecord` objects.
        #[arg(long)]
        parity: PathBuf,
        /// JSON object (`{"context_id": "parent_cluster_id"}`) or TSV (`context_id<TAB>parent_id`).
        #[arg(long)]
        parent_map: PathBuf,
        /// Optional JSON file of `SelfPlayPrereg` (`jope.selfplay-round1-prereg/v1`) bound by `plan.prereg_sha256`.
        #[arg(long)]
        prereg: Option<PathBuf>,
        /// New report path; must not exist.
        #[arg(long)]
        out: PathBuf,
    },
}

fn read(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).with_context(|| format!("read {}", path.display()))
}

fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let meta = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    anyhow::ensure!(meta.is_file(), "{} must be a regular file", path.display());
    anyhow::ensure!(
        meta.len() <= max_bytes as u64,
        "{} size ({} bytes) exceeds maximum allowed {} bytes",
        path.display(),
        meta.len(),
        max_bytes
    );
    let f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buf = Vec::with_capacity((meta.len() as usize).min(max_bytes));
    f.take((max_bytes as u64).saturating_add(1))
        .read_to_end(&mut buf)
        .with_context(|| format!("read {}", path.display()))?;
    anyhow::ensure!(
        buf.len() <= max_bytes,
        "{} exceeds maximum allowed {} bytes",
        path.display(),
        max_bytes
    );
    Ok(buf)
}

fn parse_json_or_jsonl<T: serde::de::DeserializeOwned>(bytes: &[u8], what: &str) -> Result<Vec<T>> {
    let text = std::str::from_utf8(bytes).with_context(|| format!("{what}: invalid UTF-8"))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('[') {
        return serde_json::from_str(trimmed).with_context(|| format!("{what}: JSON array"));
    }
    trimmed
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).with_context(|| format!("{what} line {}", i + 1)))
        .collect()
}

fn parse_strict_json_or_jsonl(bytes: &[u8], what: &str) -> Result<Vec<serde_json::Value>> {
    use jope_learning::clover::round_qual::parse_strict_json_value;
    let text = std::str::from_utf8(bytes).with_context(|| format!("{what}: invalid UTF-8"))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('[') {
        match parse_strict_json_value(trimmed.as_bytes(), what)? {
            serde_json::Value::Array(items) => return Ok(items),
            _ => anyhow::bail!("{what}: expected JSON array"),
        }
    }
    trimmed
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| parse_strict_json_value(l.as_bytes(), &format!("{what} line {}", i + 1)))
        .collect()
}

fn create_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {} (outputs are never overwritten)", path.display()))?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

fn pretty<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut v = serde_json::to_vec_pretty(&serde_json::to_value(value)?)?;
    v.push(b'\n');
    Ok(v)
}

fn jsonl<T: Serialize>(rows: &[T]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for row in rows {
        out.extend(jope_learning::canon::canonical_json(row)?);
        out.push(b'\n');
    }
    Ok(out)
}

fn read_pools(bytes: &[u8]) -> Result<Vec<Pool>> {
    std::str::from_utf8(bytes)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("pool line"))
        .collect()
}

fn id_lines(bytes: &[u8]) -> Result<BTreeSet<String>> {
    Ok(std::str::from_utf8(bytes)?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect())
}

#[derive(Serialize)]
struct InputDigest {
    path: String,
    sha256: String,
}

#[derive(Serialize)]
struct EvaluationOutput<'a> {
    receipt_sha256: String,
    /// Full commit of the source this binary was built from, as declared by the operator.
    source_commit: String,
    binary_sha256: String,
    inputs: Vec<InputDigest>,
    external_provenance: Vec<serde_json::Value>,
    receipt: &'a jope_learning::pools::evaluate::Receipt,
}

#[path = "jope_pools/commands.rs"]
mod jope_pools_commands;

fn main() -> Result<()> {
    jope_pools_commands::run()
}
