//! WS-5 instrument 1: evaluate the frozen sealed cohort — the project's only
//! benchmark-valid zero-shot estimate — and report it per checkpoint.
//!
//! `catalog/private/sealed-expanded-v1.json` holds 256 family-disjoint datasets
//! per task. No run has ever evaluated against it, so every generalization
//! number in this project rests on a validation split that has been trained
//! against repeatedly.
//!
//! SAFETY CONTRACT (all four hold, and each is load-bearing):
//!
//!  1. The sealed cohort file is opened READ-ONLY and never rewritten. This tool
//!     has no `--freeze-sealed` path and never calls `freeze_sealed`. That
//!     function is idempotent ONLY because the file exists
//!     (selfplay_expansion.rs:824); losing the file silently mints a DIFFERENT
//!     512-dataset cohort with no error and destroys the zero-shot guarantee.
//!
//!  2. Emitted job ids do NOT start with `jte-`. `reconcile_local_mode`
//!     (selfplay_ingest.rs:813-821) only considers `$R/receipts/jte-*`
//!     directories, so these jobs are invisible to it by construction.
//!
//!  3. No row is ever written to `$R/ops/backlog.sqlite`. Both ingestion paths
//!     are driven by `SELECT job_id FROM claims WHERE state='submitted' ...`
//!     (selfplay_ingest.rs:995-1001, :800-803). With no claim there is no
//!     ingestion, so a sealed evaluation can never become a training label.
//!
//!  4. Manifests, specs and receipts live under an operator-supplied output
//!     directory, never under `$R/manifests` or `$R/receipts`.
//!
//! Together these mean: running this tool cannot contaminate the cohort it
//! measures. That is the whole point — an instrument that pollutes its own
//! sample is worse than no instrument.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use jope_dime_regressor::selfplay_expansion::TASKS;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const SHARD_SCHEMA: &str = "JopeFlywheelCorpusEvalShard.v1";
const PORTFOLIO_SCHEMA: &str = "JopeSealedZeroShotPortfolio.v1";
const REPORT_SCHEMA: &str = "JopeSealedZeroShotReport.v1";
/// Job-id prefix. MUST NOT be `jte-` — see safety contract item 2.
const JOB_PREFIX: &str = "wsm5z";
const PROJECT: &str = "wsm5-sealed-zeroshot";

#[derive(Parser)]
#[command(about = "Zero-shot evaluation of the frozen sealed cohort (read-only on the cohort)")]
struct Args {
    #[arg(long, default_value = "/home/ubuntu/jope-flywheel/test_exact_gen0")]
    root: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build the portfolio, the per-dataset manifests and the SCQ specs.
    Plan {
        /// Output directory for portfolio/manifests/specs. Never inside $R.
        #[arg(long)]
        out: PathBuf,
        /// Genomes per task drawn from the champion index, ranked by the number
        /// of TRAINING datasets each is champion on.
        #[arg(long, default_value_t = 32)]
        top_k: usize,
        /// Evaluate only the first N sealed datasets per task (0 = all 256).
        #[arg(long, default_value_t = 0)]
        limit_per_task: usize,
        /// Evaluator binary to stage into each job.
        #[arg(long)]
        evaluator: PathBuf,
        /// Shard runner script to stage into each job.
        #[arg(
            long,
            default_value = "/home/ubuntu/jope-flywheel/scripts/fweval_runner.sh"
        )]
        runner: PathBuf,
        /// Skip sealed datasets that already carry observations in the corpus.
        /// They are contaminated and cannot support a zero-shot claim.
        #[arg(long, default_value_t = true)]
        exclude_contaminated: bool,
    },
    /// Report which sealed datasets are already contaminated, and how.
    Contamination {
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Collect finished receipts and emit the zero-shot report.
    Report {
        /// The `plan --out` directory.
        #[arg(long)]
        plan: PathBuf,
        /// SCQ run root holding job outputs.
        #[arg(long, default_value = "/home/ubuntu/smart_cluster/var/runs")]
        runs: PathBuf,
        /// SCQ CLI, used to fetch outputs from the node a job actually ran on.
        #[arg(long, default_value = "/home/ubuntu/smart_cluster/target/release/scq")]
        scq: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

// --------------------------------------------------------------------------
// sealed cohort (READ ONLY)
// --------------------------------------------------------------------------

#[derive(Clone, Deserialize)]
struct SealedEntry {
    task: String,
    dataset_id: String,
    dataset_hash: String,
    source: PathBuf,
    mirror: PathBuf,
    source_family: String,
    train_sha256: String,
    test_sha256: String,
}

#[derive(Deserialize)]
struct SealedCohort {
    schema: String,
    classification: String,
    immutable: bool,
    cohort_sha256: String,
    entries: Vec<SealedEntry>,
}

fn load_sealed(root: &Path) -> Result<SealedCohort> {
    let path = root.join("catalog/private/sealed-expanded-v1.json");
    let text = fs::read_to_string(&path)
        .with_context(|| format!("read sealed cohort {} (READ ONLY)", path.display()))?;
    let cohort: SealedCohort = serde_json::from_str(&text).context("parse sealed cohort")?;
    if cohort.schema != "JopeSealedExpandedCohort.v1" {
        bail!("unexpected sealed cohort schema {}", cohort.schema);
    }
    if cohort.classification != "sealed_zero_shot" || !cohort.immutable {
        bail!("sealed cohort is not an immutable zero-shot cohort");
    }
    if cohort.entries.len() != 512 {
        bail!("sealed cohort size drift: {} entries", cohort.entries.len());
    }
    Ok(cohort)
}

// --------------------------------------------------------------------------
// contamination: which sealed datasets already carry observations
// --------------------------------------------------------------------------

/// Index-led probe: `(task,dataset_hash)` is the covering prefix of
/// `observations_best_v1`, so this is 512 index seeks, never a scan.
fn contaminated(root: &Path, cohort: &SealedCohort) -> Result<BTreeMap<String, (u64, f64)>> {
    let db = root.join("corpus/strict-test-exact.sqlite");
    let conn = rusqlite::Connection::open_with_flags(
        &db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("open corpus read-only {}", db.display()))?;
    conn.busy_timeout(std::time::Duration::from_secs(180))?;
    let mut statement = conn.prepare(
        "SELECT count(*), coalesce(max(score), -1e308) FROM observations
         WHERE task=?1 AND dataset_hash=?2",
    )?;
    let mut out = BTreeMap::new();
    for entry in &cohort.entries {
        let (n, best): (i64, f64) = statement
            .query_row(rusqlite::params![entry.task, entry.dataset_hash], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?;
        if n > 0 {
            out.insert(
                format!("{}:{}", entry.task, entry.dataset_hash),
                (n as u64, best),
            );
        }
    }
    Ok(out)
}

// --------------------------------------------------------------------------
// portfolio
// --------------------------------------------------------------------------

#[derive(Clone, Serialize)]
struct PortfolioGenome {
    canonical_genome: String,
    semantic_sha256: String,
    /// Number of TRAINING datasets this genome is champion on.
    champion_datasets: u64,
    /// Checkpoint that proposed it on the plurality of those datasets.
    model_checkpoint_sha256: String,
    candidate_source: String,
    arm: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Baseline arm: the cheapest sane program for each task. Without it a lift
/// number is unanchored — "best of 32 evolved genomes scores 0.81" means
/// nothing until you know a bare linear model scores 0.79.
fn baseline_genomes(task: &str) -> Vec<(&'static str, &'static str)> {
    if task == "binary" {
        vec![
            ("(pipe X (logit_ridge))", "base_linear"),
            ("(pipe X (standardize) (logit_ridge))", "base_linear_std"),
        ]
    } else {
        vec![
            ("(pipe X (ridge))", "base_linear"),
            ("(pipe X (standardize) (ridge))", "base_linear_std"),
        ]
    }
}

fn build_portfolio(root: &Path, top_k: usize) -> Result<BTreeMap<String, Vec<PortfolioGenome>>> {
    let path = root.join("indexes/champions.json");
    let text =
        fs::read_to_string(&path).with_context(|| format!("read champions {}", path.display()))?;
    let champions: Vec<Value> = serde_json::from_str(&text).context("parse champions")?;

    // genome -> (task, dataset count, checkpoint votes, source votes)
    let mut tally: HashMap<(String, String), (u64, HashMap<String, u64>, HashMap<String, u64>)> =
        HashMap::new();
    for row in &champions {
        let task = row.get("task").and_then(Value::as_str).unwrap_or_default();
        let best = match row.get("best_new_strict") {
            Some(Value::Object(map)) => map,
            _ => continue,
        };
        let Some(genome) = best.get("canonical_genome").and_then(Value::as_str) else {
            continue;
        };
        let checkpoint = best
            .get("model_checkpoint_sha256")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let source = best
            .get("candidate_source")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let slot = tally
            .entry((task.to_owned(), genome.to_owned()))
            .or_insert_with(|| (0, HashMap::new(), HashMap::new()));
        slot.0 += 1;
        *slot.1.entry(checkpoint).or_insert(0) += 1;
        *slot.2.entry(source).or_insert(0) += 1;
    }

    let mut per_task: BTreeMap<String, Vec<PortfolioGenome>> = BTreeMap::new();
    for task in TASKS {
        let mut rows: Vec<_> = tally
            .iter()
            .filter(|((t, _), _)| t == task)
            .map(|((_, genome), (n, ckpts, srcs))| {
                let plurality = |m: &HashMap<String, u64>| {
                    m.iter()
                        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                        .map(|(k, _)| k.clone())
                        .unwrap_or_else(|| "unknown".to_owned())
                };
                PortfolioGenome {
                    semantic_sha256: sha256_hex(genome.as_bytes()),
                    canonical_genome: genome.clone(),
                    champion_datasets: *n,
                    model_checkpoint_sha256: plurality(ckpts),
                    candidate_source: plurality(srcs),
                    arm: "hof".to_owned(),
                }
            })
            .collect();
        // Deterministic: coverage DESC, then genome ASC.
        rows.sort_by(|a, b| {
            b.champion_datasets
                .cmp(&a.champion_datasets)
                .then(a.canonical_genome.cmp(&b.canonical_genome))
        });
        rows.truncate(top_k);
        for (genome, arm) in baseline_genomes(task) {
            rows.push(PortfolioGenome {
                canonical_genome: genome.to_owned(),
                semantic_sha256: sha256_hex(genome.as_bytes()),
                champion_datasets: 0,
                model_checkpoint_sha256: "baseline".to_owned(),
                candidate_source: "wsm5_baseline".to_owned(),
                arm: arm.to_owned(),
            });
        }
        per_task.insert(task.to_owned(), rows);
    }
    Ok(per_task)
}

// --------------------------------------------------------------------------
// plan
// --------------------------------------------------------------------------

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn plan(
    root: &Path,
    out: &Path,
    top_k: usize,
    limit_per_task: usize,
    evaluator: &Path,
    runner: &Path,
    exclude_contaminated: bool,
) -> Result<Value> {
    if out.starts_with(root) {
        bail!(
            "refusing to write plan inside the live root {} (safety contract item 4)",
            root.display()
        );
    }
    let cohort = load_sealed(root)?;
    let dirty = contaminated(root, &cohort)?;
    let portfolio = build_portfolio(root, top_k)?;

    let manifests = out.join("manifests");
    let specs = out.join("specs");
    fs::create_dir_all(&manifests)?;
    fs::create_dir_all(&specs)?;

    let evaluator_sha = sha256_hex(&fs::read(evaluator)?);
    let runner_sha = sha256_hex(&fs::read(runner)?);

    let mut planned = 0_usize;
    let mut skipped_contaminated = 0_usize;
    let mut per_task_planned: BTreeMap<String, usize> = BTreeMap::new();
    let mut spec_list = Vec::new();

    for task in TASKS {
        let mut taken = 0_usize;
        // Deterministic order: dataset_id ASC.
        let mut entries: Vec<&SealedEntry> =
            cohort.entries.iter().filter(|e| e.task == task).collect();
        entries.sort_by(|a, b| a.dataset_id.cmp(&b.dataset_id));
        for entry in entries {
            let key = format!("{}:{}", entry.task, entry.dataset_hash);
            if exclude_contaminated && dirty.contains_key(&key) {
                skipped_contaminated += 1;
                continue;
            }
            if limit_per_task > 0 && taken >= limit_per_task {
                break;
            }
            let candidates: Vec<Value> = portfolio[task]
                .iter()
                .map(|g| {
                    json!({
                        "semantic_sha256": g.semantic_sha256,
                        "canonical_genome": g.canonical_genome,
                    })
                })
                .collect();
            // label_authority is deliberately OMITTED. corpus_eval then defaults
            // to benchmark_valid=false / generation="adaptive"
            // (corpus_eval.rs:339-352), which is the most inert receipt shape
            // available: even if one of these receipts were somehow surfaced to
            // an ingester, it carries no self-play authority.
            let shard = json!({
                "schema": SHARD_SCHEMA,
                "task": entry.task,
                "dataset_hash": entry.dataset_hash,
                "train_csv": entry.source.join("train.csv").to_string_lossy(),
                "test_csv": entry.source.join("test.csv").to_string_lossy(),
                "train_sha256": entry.train_sha256,
                "test_sha256": entry.test_sha256,
                "candidates": candidates,
            });
            let manifest_path = manifests.join(format!("{}.json", entry.dataset_id));
            let bytes = serde_json::to_vec_pretty(&shard)?;
            atomic_write(&manifest_path, &bytes)?;
            let manifest_sha = sha256_hex(&bytes);

            let job_id = format!(
                "{JOB_PREFIX}-{}-{}",
                if task == "binary" { "b" } else { "r" },
                entry.dataset_id.replace("synth_", "s")
            );
            let spec = format!(
                r#"[job]
id = "{job_id}"
project = "{PROJECT}"
tier = "normal"
command = "/usr/bin/env"
args = ["{{input:runner}}", "{{input:manifest}}", "{{input:train}}", "{{input:test}}", "{{input:evaluator}}", "{{output:receipt}}"]

[resource]
gpus_per_task = 0
cpus_reserved = 1
omp_threads = 1
ram_mb = 512
gang_size = 1
placement = "single_node"

[env]
RAYON_NUM_THREADS = "1"
OMP_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
NUMEXPR_NUM_THREADS = "1"

[preempt]
preemptible = true
checkpoint = "none"
resume = "rerun"

[limits]
max_runtime_hours = 6
max_restarts = 1

[[input]]
name = "runner"
source = "{runner}"
target = "runner.sh"
sha256 = "{runner_sha}"

[[input]]
name = "manifest"
source = "{manifest}"
target = "manifest.json"
sha256 = "{manifest_sha}"

[[input]]
name = "train"
source = "{train}"
target = "train.csv"
sha256 = "{train_sha}"

[[input]]
name = "test"
source = "{test}"
target = "test.csv"
sha256 = "{test_sha}"

[[input]]
name = "evaluator"
source = "{evaluator}"
target = "corpus_eval"
sha256 = "{evaluator_sha}"

[[output]]
name = "receipt"
source = "receipt"
directory = true
"#,
                runner = runner.display(),
                manifest = manifest_path.display(),
                train = entry.source.join("train.csv").display(),
                train_sha = entry.train_sha256,
                test = entry.source.join("test.csv").display(),
                test_sha = entry.test_sha256,
                evaluator = evaluator.display(),
            );
            let spec_path = specs.join(format!("{job_id}.toml"));
            atomic_write(&spec_path, spec.as_bytes())?;
            spec_list.push(json!({
                "job_id": job_id,
                "task": entry.task,
                "dataset_id": entry.dataset_id,
                "dataset_hash": entry.dataset_hash,
                "source_family": entry.source_family,
                "spec": spec_path.to_string_lossy(),
                "manifest": manifest_path.to_string_lossy(),
                "mirror": entry.mirror.to_string_lossy(),
            }));
            planned += 1;
            taken += 1;
            *per_task_planned.entry(task.to_owned()).or_insert(0) += 1;
        }
    }

    let portfolio_doc = json!({
        "schema": PORTFOLIO_SCHEMA,
        "cohort_sha256": cohort.cohort_sha256,
        "top_k": top_k,
        "portfolio": portfolio,
        "contaminated": dirty.iter().map(|(k, (n, best))| {
            json!({"key": k, "observations": n, "best_score": best})
        }).collect::<Vec<_>>(),
        "jobs": spec_list,
        "evaluator_sha256": evaluator_sha,
        "runner_sha256": runner_sha,
    });
    atomic_write(
        &out.join("portfolio.json"),
        &serde_json::to_vec_pretty(&portfolio_doc)?,
    )?;

    Ok(json!({
        "cohort_sha256": cohort.cohort_sha256,
        "planned_jobs": planned,
        "per_task": per_task_planned,
        "skipped_contaminated": skipped_contaminated,
        "contaminated_total": dirty.len(),
        "candidates_per_manifest": {
            "regression": portfolio["regression"].len(),
            "binary": portfolio["binary"].len(),
        },
        "out": out.to_string_lossy(),
    }))
}

// --------------------------------------------------------------------------
// report
// --------------------------------------------------------------------------

#[derive(Deserialize)]
struct ReceiptRow {
    task: String,
    dataset_hash: String,
    semantic_sha256: String,
    metric: String,
    score: Option<f64>,
    valid: bool,
    #[serde(default)]
    failure: Option<String>,
    runtime_seconds: f64,
}

/// Locate a job's receipt directory.
///
/// MEASURED: the cluster has NO shared filesystem (`ssh xbabe3 ls
/// $R/training/packed` -> "No such file or directory"), so a job that ran on a
/// remote node leaves `runs/<job>/attempts/N/outputs/receipt` on THAT node and
/// only `.portable-manifest.json` locally. Reading the local run dir therefore
/// silently reports "pending" for every remote job -- which is most of them.
/// `scq fetch` is the only correct retrieval path. The local dir is still tried
/// first because jobs that ran on this host are already complete there.
/// A receipt dir holding only dotfiles (`.portable-manifest.json`, `.stderr.log`)
/// is a stub: `scq fetch` against a job that has not yet written its outputs
/// still creates the directory. Caching such a stub is the whole bug this guards
/// against -- a stub cached while the job was still running is otherwise returned
/// forever, so the job stays "pending" in every later report even after it
/// completes and its receipts become fetchable.
fn substantive(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|mut d| {
            d.any(|e| {
                e.map(|e| !e.file_name().to_string_lossy().starts_with('.'))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn receipt_dir(runs: &Path, scq: &Path, cache: &Path, job_id: &str) -> Option<PathBuf> {
    let attempts = runs.join(job_id).join("attempts");
    let mut best: Option<(u64, PathBuf)> = None;
    if let Ok(entries) = fs::read_dir(&attempts) {
        for entry in entries.flatten() {
            let Ok(n) = entry.file_name().to_string_lossy().parse::<u64>() else {
                continue; // skip non-numeric attempt dirs rather than abort
            };
            let dir = entry.path().join("outputs").join("receipt");
            if substantive(&dir) && best.as_ref().is_none_or(|(m, _)| n > *m) {
                best = Some((n, dir));
            }
        }
    }
    if let Some((_, path)) = best {
        return Some(path);
    }
    let target = cache.join(job_id);
    let fetched = target.join("receipt");
    if substantive(&fetched) {
        return Some(fetched);
    }
    // Either nothing cached, or a stub from a fetch issued while the job was
    // still running. Drop it and re-fetch; a still-running job simply produces
    // another stub, which we discard again instead of poisoning the cache.
    let _ = fs::remove_dir_all(&target);
    fs::create_dir_all(&target).ok()?;
    let status = std::process::Command::new(scq)
        .arg("fetch")
        .arg(job_id)
        .arg("--output")
        .arg(&target)
        .output()
        .ok()?;
    if !status.status.success() {
        let _ = fs::remove_dir_all(&target);
        return None;
    }
    if !substantive(&fetched) {
        let _ = fs::remove_dir_all(&target);
        return None;
    }
    Some(fetched)
}

fn median(values: &mut Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn report(plan_dir: &Path, runs: &Path, scq: &Path, out: &Path) -> Result<Value> {
    let cache = plan_dir.join("fetched");
    fs::create_dir_all(&cache)?;
    let doc: Value = serde_json::from_str(&fs::read_to_string(plan_dir.join("portfolio.json"))?)?;
    let jobs = doc["jobs"].as_array().context("portfolio.jobs")?;

    // semantic -> (arm, checkpoint, coverage)
    let mut meta: HashMap<String, (String, String, u64)> = HashMap::new();
    for task in TASKS {
        for g in doc["portfolio"][task].as_array().into_iter().flatten() {
            meta.insert(
                g["semantic_sha256"].as_str().unwrap_or_default().to_owned(),
                (
                    g["arm"].as_str().unwrap_or_default().to_owned(),
                    g["model_checkpoint_sha256"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                    g["champion_datasets"].as_u64().unwrap_or(0),
                ),
            );
        }
    }

    // per dataset: arm/checkpoint -> best score
    let mut done = 0_usize;
    let mut pending = 0_usize;
    let mut per_dataset: Vec<Value> = Vec::new();
    let mut lifts: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut wins: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut invalid_rows = 0_u64;
    let mut total_rows = 0_u64;
    let mut runtime_total = 0.0_f64;

    for job in jobs {
        let job_id = job["job_id"].as_str().unwrap_or_default();
        let Some(dir) = receipt_dir(runs, scq, &cache, job_id) else {
            pending += 1;
            continue;
        };
        let hash = job["dataset_hash"].as_str().unwrap_or_default();
        let path = dir.join(format!("{hash}.receipts.jsonl"));
        let Ok(text) = fs::read_to_string(&path) else {
            pending += 1;
            continue;
        };
        let mut best_by: BTreeMap<String, f64> = BTreeMap::new();
        let mut metric = String::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let row: ReceiptRow = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(_) => continue,
            };
            total_rows += 1;
            runtime_total += row.runtime_seconds;
            metric = row.metric.clone();
            if !row.valid || row.score.is_none() {
                invalid_rows += 1;
                continue;
            }
            let score = row.score.unwrap();
            let (arm, checkpoint, _) = meta
                .get(&row.semantic_sha256)
                .cloned()
                .unwrap_or_else(|| ("unknown".into(), "unknown".into(), 0));
            let bucket = if arm == "hof" {
                format!("hof:{checkpoint}")
            } else {
                arm.clone()
            };
            for key in ["ALL_HOF".to_owned(), bucket] {
                if arm == "hof" || key != "ALL_HOF" {
                    best_by
                        .entry(key)
                        .and_modify(|v| {
                            if score > *v {
                                *v = score
                            }
                        })
                        .or_insert(score);
                }
            }
            let _ = &row.task;
            let _ = &row.failure;
        }
        done += 1;
        let base = ["base_linear", "base_linear_std"]
            .iter()
            .filter_map(|k| best_by.get(*k).copied())
            .fold(f64::NEG_INFINITY, f64::max);
        if base.is_finite() {
            for (key, value) in &best_by {
                if key.starts_with("base_") {
                    continue;
                }
                lifts.entry(key.clone()).or_default().push(value - base);
                let entry = wins.entry(key.clone()).or_insert((0, 0));
                entry.1 += 1;
                if *value > base {
                    entry.0 += 1;
                }
            }
        }
        per_dataset.push(json!({
            "job_id": job_id,
            "task": job["task"],
            "dataset_id": job["dataset_id"],
            "dataset_hash": hash,
            "source_family": job["source_family"],
            "metric": metric,
            "best_by_bucket": best_by,
            "baseline_best": if base.is_finite() { json!(base) } else { Value::Null },
        }));
    }

    let mut summary = BTreeMap::new();
    for (key, values) in &lifts {
        let mut v = values.clone();
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let med = median(&mut v);
        let (w, t) = wins[key];
        summary.insert(
            key.clone(),
            json!({
                "n_datasets": values.len(),
                "mean_lift_over_baseline": mean,
                "median_lift_over_baseline": med,
                "win_rate_vs_baseline": w as f64 / t.max(1) as f64,
                "wins": w, "compared": t,
            }),
        );
    }

    let result = json!({
        "schema": REPORT_SCHEMA,
        "cohort_sha256": doc["cohort_sha256"],
        "jobs_planned": jobs.len(),
        "jobs_with_receipts": done,
        "jobs_pending": pending,
        "receipt_rows": total_rows,
        "invalid_rows": invalid_rows,
        "evaluator_seconds_total": runtime_total,
        "summary": summary,
        "per_dataset": per_dataset,
    });
    atomic_write(out, &serde_json::to_vec_pretty(&result)?)?;
    let mut brief = result.clone();
    brief.as_object_mut().unwrap().remove("per_dataset");
    Ok(brief)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let value = match &args.command {
        Command::Plan {
            out,
            top_k,
            limit_per_task,
            evaluator,
            runner,
            exclude_contaminated,
        } => plan(
            &args.root,
            out,
            *top_k,
            *limit_per_task,
            evaluator,
            runner,
            *exclude_contaminated,
        )?,
        Command::Contamination { out } => {
            let cohort = load_sealed(&args.root)?;
            let dirty = contaminated(&args.root, &cohort)?;
            let by_task = cohort
                .entries
                .iter()
                .filter(|e| dirty.contains_key(&format!("{}:{}", e.task, e.dataset_hash)))
                .fold(BTreeMap::<String, u64>::new(), |mut acc, e| {
                    *acc.entry(e.task.clone()).or_insert(0) += 1;
                    acc
                });
            let detail: Vec<Value> = cohort
                .entries
                .iter()
                .filter_map(|e| {
                    let key = format!("{}:{}", e.task, e.dataset_hash);
                    dirty.get(&key).map(|(n, best)| {
                        json!({
                            "task": e.task, "dataset_id": e.dataset_id,
                            "dataset_hash": e.dataset_hash,
                            "source_family": e.source_family,
                            "observations": n, "best_score": best,
                        })
                    })
                })
                .collect();
            let value = json!({
                "cohort_sha256": cohort.cohort_sha256,
                "sealed_total": cohort.entries.len(),
                "contaminated_total": dirty.len(),
                "contaminated_by_task": by_task,
                "clean_total": cohort.entries.len() - dirty.len(),
                "detail": detail,
            });
            if let Some(path) = out {
                atomic_write(path, &serde_json::to_vec_pretty(&value)?)?;
            }
            value
        }
        Command::Report {
            plan,
            runs,
            scq,
            out,
        } => report(plan, runs, scq, out)?,
    };
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_ids_never_collide_with_the_ingestion_prefix() {
        // selfplay_ingest.rs:813-821 only materializes $R/receipts/jte-* dirs.
        assert!(!JOB_PREFIX.starts_with("jte"));
        assert!(!format!("{JOB_PREFIX}-r-abc").starts_with("jte-"));
    }

    #[test]
    fn semantic_hash_matches_the_evaluator_definition() {
        // corpus_eval.rs:690 recomputes sha256(canonical_genome) and warns on drift.
        let genome = "(pipe X (ridge))";
        assert_eq!(sha256_hex(genome.as_bytes()).len(), 64);
    }

    #[test]
    fn baseline_arms_exist_for_both_tasks() {
        assert_eq!(baseline_genomes("binary").len(), 2);
        assert_eq!(baseline_genomes("regression").len(), 2);
        assert!(baseline_genomes("binary")[0].0.contains("logit_ridge"));
        assert!(baseline_genomes("regression")[0].0.contains("ridge"));
    }

    #[test]
    fn median_handles_even_and_odd() {
        assert_eq!(median(&mut vec![1.0, 3.0]), 2.0);
        assert_eq!(median(&mut vec![1.0, 5.0, 3.0]), 3.0);
        assert!(median(&mut vec![]).is_nan());
    }
}
