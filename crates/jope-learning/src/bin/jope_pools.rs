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

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::IngestCoverage {
            coverage,
            partition,
            exclude_contexts,
            exclude_pairs,
            protocol_sha256,
            out,
        } => {
            let mut excluded = BTreeSet::new();
            for p in &exclude_contexts {
                excluded.extend(id_lines(&read(p)?)?);
            }
            let mut pairs = BTreeSet::new();
            for p in &exclude_pairs {
                for line in id_lines(&read(p)?)? {
                    let (cid, program) = line
                        .split_once('\t')
                        .context("--exclude-pairs lines are context_id<TAB>program_sha256")?;
                    pairs.insert((cid.to_owned(), program.to_owned()));
                }
            }
            let doc: serde_json::Value = serde_json::from_slice(&read(&coverage)?)?;
            let (rows, report) =
                source::coverage_rows(&doc, &partition, &excluded, &pairs, &protocol_sha256)?;
            create_new(&out, &jsonl(&rows)?)?;
            println!("{}", serde_json::to_string(&report)?);
        }
        Command::IngestRosterFeatures {
            roster,
            pack,
            manifest,
            out,
        } => {
            let manifest: Manifest = serde_json::from_slice(&read(&manifest)?)?;
            let wanted: BTreeSet<String> = manifest
                .pools
                .iter()
                .map(|e| e.context_id.clone())
                .collect();
            let lines = source::roster_vectors(
                std::str::from_utf8(&read(&roster)?)?,
                &read(&pack)?,
                &wanted,
            )?;
            create_new(&out, &jsonl(&lines)?)?;
            println!("exported {} context vectors", lines.len());
        }
        Command::Views {
            pools,
            manifest,
            out,
        } => {
            let pools = read_pools(&read(&pools)?)?;
            let manifest: Manifest = serde_json::from_slice(&read(&manifest)?)?;
            let views = export_views(&pools, &manifest)?;
            create_new(&out, &jsonl(&views)?)?;
            println!("exported {} outcome-free views", views.len());
        }
        Command::Freeze {
            rows,
            out,
            min_pool,
            max_pool,
            salt,
            min_distinct_measured,
            source_grade,
        } => {
            let bytes = read(&rows)?;
            let parsed: Vec<MeasurementRow> = std::str::from_utf8(&bytes)?
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim().is_empty())
                .map(|(i, l)| {
                    serde_json::from_str(l).with_context(|| format!("row line {}", i + 1))
                })
                .collect::<Result<_>>()?;
            let rule = FreezeRule {
                min_pool,
                max_pool,
                salt,
                min_distinct_measured,
                source_grade,
            };
            let (pools, manifest) = freeze(&parsed, &rule, &sha256_hex(&bytes))?;
            fs::create_dir(&out)
                .with_context(|| format!("create {} (must not exist)", out.display()))?;
            create_new(&out.join("pools.jsonl"), &jsonl(&pools)?)?;
            create_new(&out.join("manifest.json"), &pretty(&manifest)?)?;
            println!(
                "froze {} pools ({} excluded) from {} rows; manifest {}",
                manifest.pools.len(),
                manifest.excluded.len(),
                manifest.input_rows,
                manifest.sha256()?
            );
        }
        Command::Evaluate {
            pools,
            manifest,
            reference,
            context_features,
            program_features,
            s3_prior_parquet,
            external,
            unavailable,
            budget,
            folds,
            fold_salt,
            prior_weight,
            knn_k,
            ridge_lambda,
            bootstrap_draws,
            bootstrap_seed,
            alpha,
            source_commit,
            out,
        } => {
            anyhow::ensure!(
                source_commit.len() == 40
                    && source_commit
                        .bytes()
                        .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
                "--source-commit must be a full lowercase commit"
            );
            let mut digests = Vec::new();
            let mut load = |p: &Path| -> Result<Vec<u8>> {
                let b = read(p)?;
                digests.push(InputDigest {
                    path: p.display().to_string(),
                    sha256: sha256_hex(&b),
                });
                Ok(b)
            };
            let pools = read_pools(&load(&pools)?)?;
            let manifest: Manifest = serde_json::from_slice(&load(&manifest)?)?;
            let cf = context_features
                .map(|p| Features::from_jsonl(std::str::from_utf8(&load(&p)?)?))
                .transpose()?;
            let pf = program_features
                .map(|p| Features::from_jsonl(std::str::from_utf8(&load(&p)?)?))
                .transpose()?;
            let s3_rows = s3_prior_parquet
                .map(|p| {
                    jope_learning::retention::distilled::read(
                        &load(&p)?,
                        &jope_learning::pools::flywheel::known_scores(),
                    )
                })
                .transpose()?;
            let docs: Vec<ExternalDocument> = external
                .iter()
                .map(|p| Ok(serde_json::from_slice(&load(p)?)?))
                .collect::<Result<_>>()?;
            let externals: Vec<External> = docs
                .iter()
                .map(External::from_document)
                .collect::<Result<_>>()?;
            let unavailable = unavailable
                .iter()
                .map(|s| {
                    let (name, reason) = s
                        .split_once('=')
                        .context("--unavailable takes name=reason")?;
                    Ok(Unavailable {
                        name: name.to_owned(),
                        reason: reason.to_owned(),
                    })
                })
                .collect::<Result<_>>()?;
            let inputs = Inputs {
                context_features: cf.as_ref(),
                program_features: pf.as_ref(),
                external: externals.iter().map(|e| e as &dyn Selector).collect(),
                unavailable,
            };
            let cfg = Config {
                reference,
                budget,
                folds,
                fold_salt,
                prior_weight,
                knn_k,
                ridge_lambda,
                bootstrap_draws,
                bootstrap_seed,
                alpha,
            };
            let fold_priors = match s3_rows.as_deref() {
                Some(rows) => {
                    let (p, _) =
                        jope_learning::pools::flywheel::build_fold_s3_priors(&pools, &cfg, rows)?;
                    Some(p)
                }
                None => None,
            };
            let receipt =
                jope_learning::pools::evaluate::evaluate_with_fold_train_pools_and_priors(
                    &pools,
                    None,
                    fold_priors.as_deref(),
                    &manifest,
                    &inputs,
                    &cfg,
                )?;
            let output = EvaluationOutput {
                receipt_sha256: receipt.sha256()?,
                source_commit: source_commit.clone(),
                binary_sha256: sha256_hex(&read(&std::env::current_exe()?)?),
                inputs: digests,
                external_provenance: docs.iter().map(|d| d.provenance.clone()).collect(),
                receipt: &receipt,
            };
            create_new(&out, &pretty(&output)?)?;
            println!(
                "{} contexts; strongest baseline {}; ranking_learnability {}; receipt {}",
                receipt.contexts.len(),
                receipt.strongest_baseline,
                receipt.ranking_learnability,
                output.receipt_sha256
            );
        }
        Command::FlywheelDistill {
            pools,
            manifest,
            reference,
            context_features,
            program_features,
            s3_prior_parquet,
            external,
            budget,
            eval_budgets,
            rounds,
            initial_budget,
            step_budget,
            proposal_pool_k,
            score_name,
            folds,
            fold_salt,
            prior_weight,
            knn_k,
            ridge_lambda,
            bootstrap_draws,
            bootstrap_seed,
            alpha,
            parquet_dir,
            source_commit,
            out,
        } => {
            anyhow::ensure!(
                source_commit.len() == 40
                    && source_commit
                        .bytes()
                        .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
                "--source-commit must be a full lowercase commit"
            );
            let mut digests = Vec::new();
            let mut load = |p: &Path| -> Result<Vec<u8>> {
                let b = read(p)?;
                digests.push(InputDigest {
                    path: p.display().to_string(),
                    sha256: sha256_hex(&b),
                });
                Ok(b)
            };
            let pools = read_pools(&load(&pools)?)?;
            let manifest: Manifest = serde_json::from_slice(&load(&manifest)?)?;
            let cf = Features::from_jsonl(std::str::from_utf8(&load(&context_features)?)?)?;
            let pf = program_features
                .map(|p| Features::from_jsonl(std::str::from_utf8(&load(&p)?)?))
                .transpose()?;
            let s3_rows = s3_prior_parquet
                .map(|p| {
                    jope_learning::retention::distilled::read(
                        &load(&p)?,
                        &jope_learning::pools::flywheel::known_scores(),
                    )
                })
                .transpose()?;
            let docs: Vec<ExternalDocument> = external
                .iter()
                .map(|p| Ok(serde_json::from_slice(&load(p)?)?))
                .collect::<Result<_>>()?;
            let externals: Vec<External> = docs
                .iter()
                .map(External::from_document)
                .collect::<Result<_>>()?;
            let ext_refs: Vec<&dyn Selector> =
                externals.iter().map(|e| e as &dyn Selector).collect();
            let cfg = jope_learning::pools::flywheel::FlywheelConfig {
                eval: Config {
                    reference,
                    budget,
                    folds,
                    fold_salt,
                    prior_weight,
                    knn_k,
                    ridge_lambda,
                    bootstrap_draws,
                    bootstrap_seed,
                    alpha,
                },
                eval_budgets,
                rounds,
                initial_budget_per_context: initial_budget,
                step_budget_per_context: step_budget,
                proposal_pool_k,
                score_name,
                tilt: jope_learning::clover::expansion::MeasuredTiltPolicyConfig {
                    temperature_tau: 0.25,
                    advantage_clip: 2.0,
                    ref_new_support_mass: 0.05,
                    policy_new_support_cap: 0.12,
                    step_size_eta: 0.20,
                },
                max_conditional_base_kl: 0.20,
                max_new_support_mass: 0.15,
            };
            let receipt = jope_learning::pools::flywheel::run_distilled_flywheel_with_s3_prior(
                &pools,
                &manifest,
                &cf,
                pf.as_ref(),
                &ext_refs,
                s3_rows.as_deref(),
                &cfg,
                parquet_dir.as_deref(),
            )?;
            #[derive(Serialize)]
            struct FlywheelOutput<'a> {
                receipt_sha256: String,
                source_commit: String,
                binary_sha256: String,
                inputs: Vec<InputDigest>,
                external_provenance: Vec<serde_json::Value>,
                receipt: &'a jope_learning::pools::flywheel::FlywheelReceipt,
            }
            let output = FlywheelOutput {
                receipt_sha256: receipt.sha256()?,
                source_commit,
                binary_sha256: sha256_hex(&read(&std::env::current_exe()?)?),
                inputs: digests,
                external_provenance: docs.iter().map(|d| d.provenance.clone()).collect(),
                receipt: &receipt,
            };
            create_new(&out, &pretty(&output)?)?;
            println!(
                "{} rounds; active_discovery_established {}; data_accumulation_lift_established {}; flywheel_established {}; ranking_learnability {}; receipt {}",
                receipt.rounds.len(),
                receipt.active_discovery_established,
                receipt.data_accumulation_lift_established,
                receipt.flywheel_established,
                receipt.final_round_receipt.ranking_learnability,
                output.receipt_sha256
            );
        }
        Command::QualifyRounds {
            plan,
            identity,
            ratchets,
            test_scores,
            parity,
            parent_map,
            prereg,
            out,
        } => {
            use jope_learning::clover::round_qual::{
                parse_parent_map, parse_strict_json_value, qualify_rounds_with_prereg,
                ParityRecord, QualTestScore, RoundQualInputs, RoundQualPlan,
            };
            use jope_learning::clover::ExperimentIdentity;

            let plan_val = parse_strict_json_value(&read(&plan)?, "--plan")?;
            let plan: RoundQualPlan = serde_json::from_value(plan_val).context("parse --plan")?;
            let identity_val = parse_strict_json_value(&read(&identity)?, "--identity")?;
            let identity: ExperimentIdentity =
                serde_json::from_value(identity_val).context("parse --identity")?;
            let ratchets: Vec<serde_json::Value> =
                parse_strict_json_or_jsonl(&read(&ratchets)?, "--ratchets")?;
            let test_scores: Vec<QualTestScore> =
                parse_json_or_jsonl(&read(&test_scores)?, "--test-scores")?;
            let parity: Vec<ParityRecord> = parse_json_or_jsonl(&read(&parity)?, "--parity")?;
            let parent_map = parse_parent_map(&read(&parent_map)?)?;
            let prereg_bytes = match prereg.as_deref() {
                Some(p) => Some(read_bounded(p, 1024 * 1024)?),
                None => None,
            };

            let report = qualify_rounds_with_prereg(
                RoundQualInputs {
                    plan: &plan,
                    identity: &identity,
                    ratchet_outputs: &ratchets,
                    test_scores: &test_scores,
                    parity: &parity,
                    parent_of_context: &parent_map,
                },
                prereg_bytes.as_deref(),
            )?;
            let digest = report.canonical_sha256()?;
            create_new(&out, &pretty(&report)?)?;
            println!(
                "{} contexts ({} complete paired across {} parent clusters); verdict {:?}; margin_cleared {}; provenance_binding_status {}; qualified_claim {}; report_sha256 {}",
                report.n_ratchet_contexts,
                report.n_complete_paired_contexts,
                report.n_observed_parent_clusters,
                report.verdict,
                report.margin_cleared,
                report.provenance_binding_status,
                report.qualified_claim,
                digest
            );
        }
    }
    Ok(())
}
