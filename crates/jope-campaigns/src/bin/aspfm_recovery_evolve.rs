use anyhow::{bail, ensure, Context, Result};
use feat_math::full_rust::{crossover_pipelines, mutate_pipeline, EvolvedGenome};
use feat_math::Pcg32;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

const POPULATION: usize = 60;
const ELITES: usize = 7;
const PARENT_POOL: usize = 30;
const GENERATIONS: usize = 10;
const TOURNAMENT: usize = 3;

#[derive(Clone)]
struct Member {
    genome: String,
    semantic: String,
    score: f64,
    receipt: Option<Value>,
}

#[derive(Clone)]
struct Campaign {
    task: String,
    dataset: String,
    dataset_id: String,
    target_score: f64,
    target: f64,
    train_sha256: String,
    test_sha256: String,
    train: PathBuf,
    test: PathBuf,
    features: usize,
    seeds: Vec<Member>,
}

struct Args {
    replay: PathBuf,
    output: PathBuf,
    data_root: PathBuf,
    evaluator: PathBuf,
    jobs: usize,
    evaluator_threads: usize,
    task: String,
    dataset: Option<String>,
    historical_target: Option<f64>,
}

fn parse_args() -> Result<Args> {
    let mut values = BTreeMap::new();
    let mut args = env::args_os().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .with_context(|| format!("{} requires a value", flag.to_string_lossy()))?;
        values.insert(flag.to_string_lossy().into_owned(), value);
    }
    let take_path = |key: &str| {
        values
            .get(key)
            .map(PathBuf::from)
            .with_context(|| format!("{key} is required"))
    };
    let jobs = values
        .get("--jobs")
        .context("--jobs is required")?
        .to_string_lossy()
        .parse::<usize>()?;
    ensure!(jobs > 0, "--jobs must be positive");
    let evaluator_threads = values
        .get("--evaluator-threads")
        .context("--evaluator-threads is required")?
        .to_string_lossy()
        .parse::<usize>()?;
    ensure!(
        evaluator_threads > 0,
        "--evaluator-threads must be positive"
    );
    let task = values
        .get("--task")
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "binary".to_owned());
    ensure!(
        matches!(task.as_str(), "binary" | "regression"),
        "unsupported task"
    );
    let dataset = values
        .get("--dataset")
        .map(|value| value.to_string_lossy().into_owned());
    let historical_target = values
        .get("--historical-target")
        .map(|value| value.to_string_lossy().parse::<f64>())
        .transpose()?;
    if task == "regression" {
        ensure!(dataset.is_some(), "regression recovery requires --dataset");
        ensure!(
            historical_target.is_some_and(f64::is_finite),
            "regression recovery requires a finite --historical-target"
        );
    }
    Ok(Args {
        replay: take_path("--replay")?,
        output: take_path("--output")?,
        data_root: take_path("--data-root")?,
        evaluator: take_path("--evaluator")?,
        jobs,
        evaluator_threads,
        task,
        dataset,
        historical_target,
    })
}

fn fitness(task: &str, score: f64) -> f64 {
    if task == "regression" {
        -score
    } else {
        score
    }
}

fn metric_score(task: &str, fitness: f64) -> f64 {
    if task == "regression" {
        -fitness
    } else {
        fitness
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    let parent = path.parent().context("output has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension("json.partial");
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn numeric_columns(path: &Path) -> Result<usize> {
    let text = fs::read_to_string(path)?;
    // The first non-blank line decides the width. This was a `for` that returned on its first
    // iteration, which reads as "try every line" but never could -- `never_loop` caught it.
    let Some(first) = text.lines().map(str::trim).find(|line| !line.is_empty()) else {
        bail!("empty dataset")
    };
    let columns = first.split(',').count();
    ensure!(columns >= 2, "dataset has fewer than two columns");
    Ok(columns - 1)
}

fn score_matches(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * left.abs().max(right.abs()).max(1.0)
}

fn seed_u64(dataset: &str, generation: usize, slot: usize, retry: usize) -> u64 {
    let digest =
        Sha256::digest(format!("aspfm-v1:{dataset}:{generation}:{slot}:{retry}").as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight bytes"))
}

fn tournament(rng: &mut Pcg32, pool: usize) -> usize {
    (0..TOURNAMENT)
        .map(|_| rng.next_u32() as usize % pool)
        .min()
        .expect("nonempty tournament")
}

fn evolved_member(value: EvolvedGenome) -> Member {
    Member {
        genome: value.canonical,
        semantic: value.semantic_sha256,
        score: f64::NEG_INFINITY,
        receipt: None,
    }
}

fn next_population(
    dataset: &str,
    ranked: &[Member],
    generation: usize,
    features: usize,
) -> Result<Vec<Member>> {
    ensure!(ranked.len() >= PARENT_POOL, "fewer than 30 valid parents");
    let mut next = ranked.iter().take(ELITES).cloned().collect::<Vec<_>>();
    let mut semantics = next
        .iter()
        .map(|member| member.semantic.clone())
        .collect::<BTreeSet<_>>();
    for slot in ELITES..POPULATION {
        let mut produced = None;
        for retry in 0..500 {
            let seed = seed_u64(dataset, generation, slot, retry);
            let mut rng = Pcg32::new(seed, 0x4153_5046_4d56_3100);
            let first = tournament(&mut rng, PARENT_POOL);
            let candidate = if rng.next_f64() < 0.25 {
                let second = tournament(&mut rng, PARENT_POOL);
                crossover_pipelines(
                    &ranked[first].genome,
                    &ranked[second].genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                )
            } else {
                mutate_pipeline(
                    &ranked[first].genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                    features,
                )
            };
            if let Ok(candidate) = candidate {
                if semantics.insert(candidate.semantic_sha256.clone()) {
                    produced = Some(evolved_member(candidate));
                    break;
                }
            }
        }
        let Some(produced) = produced else {
            break;
        };
        next.push(produced);
    }
    Ok(next)
}

fn evaluate_generation(
    campaign: &Campaign,
    evaluator: &Path,
    root: &Path,
    generation: usize,
    population: &mut [Member],
    evaluator_threads: usize,
) -> Result<()> {
    let pending = population
        .iter()
        .filter(|member| !member.score.is_finite())
        .map(|member| json!({"semantic_sha256":member.semantic,"canonical_genome":member.genome}))
        .collect::<Vec<_>>();
    if pending.is_empty() {
        return Ok(());
    }
    let pending_len = pending.len();
    let generation_root = root.join(format!("generation-{generation:02}"));
    let output = generation_root.join("output");
    fs::create_dir_all(&generation_root)?;
    let manifest = json!({
        "schema":"JopeFlywheelCorpusEvalShard.v1",
        "task":campaign.task,
        "dataset_hash":campaign.dataset,
        "train_csv":campaign.train,
        "test_csv":campaign.test,
        "train_sha256":campaign.train_sha256,
        "test_sha256":campaign.test_sha256,
        "candidates":pending,
    });
    let manifest_path = generation_root.join("manifest.json");
    atomic_json(&manifest_path, &manifest)?;
    let manifest_sha256 = sha256(&fs::read(&manifest_path)?);
    let done_path = output.join(format!("{}.done", campaign.dataset));
    let complete = fs::read(&done_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .is_some_and(|done| {
            done["manifest_sha256"].as_str() == Some(manifest_sha256.as_str())
                && done["receipts"].as_u64() == Some(pending_len as u64)
        })
        && output
            .join(format!("{}.receipts.jsonl", campaign.dataset))
            .is_file();
    if output.exists() && !complete {
        fs::remove_dir_all(&output)?;
    }
    if !complete {
        let status = Command::new(evaluator)
            .args(["--manifest"])
            .arg(&manifest_path)
            .args(["--output-dir"])
            .arg(&output)
            .args(["--threads", &evaluator_threads.to_string()])
            .args(["--timeout-seconds", "300"])
            .env("RAYON_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .stdout(Stdio::null())
            .status()?;
        ensure!(status.success(), "corpus evaluator failed");
    }
    let receipt_path = fs::read_dir(&output)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".receipts.jsonl"))
        })
        .context("evaluator wrote no receipt ledger")?;
    let mut receipts = BTreeMap::new();
    for line in fs::read_to_string(receipt_path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let value: Value = serde_json::from_str(line)?;
        if value.get("valid").and_then(Value::as_bool) == Some(true) {
            let score = if campaign.task == "regression" {
                value.get("secondary_score").and_then(Value::as_f64)
            } else {
                value.get("score").and_then(Value::as_f64)
            };
            if let (Some(semantic), Some(score)) =
                (value.get("semantic_sha256").and_then(Value::as_str), score)
            {
                receipts.insert(semantic.to_owned(), (fitness(&campaign.task, score), value));
            }
        }
    }
    for member in population {
        if let Some((score, receipt)) = receipts.remove(&member.semantic) {
            member.score = score;
            member.receipt = Some(receipt);
        }
    }
    Ok(())
}

#[path = "aspfm_recovery_evolve/campaign.rs"]
mod campaign;
use campaign::{build_campaigns, integrate, run_campaign};

fn main() -> Result<()> {
    let args = Arc::new(parse_args()?);
    ensure!(args.evaluator.is_file(), "evaluator missing");
    fs::create_dir_all(&args.output)?;
    let replay_bytes = fs::read(&args.replay)?;
    let mut replay: Value = serde_json::from_slice(&replay_bytes)?;
    let campaigns = build_campaigns(&replay, &args)?;
    println!("evolving {} recovery deficits", campaigns.len());
    let queue = Arc::new(Mutex::new(campaigns.into_iter()));
    let receipts = Arc::new(Mutex::new(Vec::<Value>::new()));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for _ in 0..args.jobs {
            let queue = Arc::clone(&queue);
            let receipts = Arc::clone(&receipts);
            let errors = Arc::clone(&errors);
            let args = Arc::clone(&args);
            scope.spawn(move || loop {
                let Some(campaign) = queue.lock().expect("queue poisoned").next() else {
                    break;
                };
                match run_campaign(campaign, &args) {
                    Ok(receipt) => receipts.lock().expect("receipts poisoned").push(receipt),
                    Err(error) => errors
                        .lock()
                        .expect("errors poisoned")
                        .push(format!("{error:#}")),
                }
            });
        }
    });
    let errors = Arc::try_unwrap(errors)
        .map_err(|_| anyhow::anyhow!("error workers live"))?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("errors poisoned"))?;
    ensure!(errors.is_empty(), "campaign errors:\n{}", errors.join("\n"));
    let mut receipts = Arc::try_unwrap(receipts)
        .map_err(|_| anyhow::anyhow!("receipt workers live"))?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("receipts poisoned"))?;
    receipts.sort_by(|left, right| {
        left["dataset_hash"]
            .as_str()
            .cmp(&right["dataset_hash"].as_str())
    });
    atomic_json(
        &args.output.join("summary.json"),
        &json!({"schema":"sotarena.aspfm.recovery-evolution-summary.v1","receipts":receipts}),
    )?;
    integrate(&mut replay, &receipts)?;
    atomic_json(&args.output.join("replay.json"), &replay)?;
    println!("{}", args.output.join("replay.json").display());
    Ok(())
}
