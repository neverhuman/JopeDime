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
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let columns = line.split(',').count();
        ensure!(columns >= 2, "dataset has fewer than two columns");
        return Ok(columns - 1);
    }
    bail!("empty dataset")
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

fn campaign_receipt_path(output: &Path, dataset: &str) -> PathBuf {
    output.join("receipts").join(format!("{dataset}.json"))
}

fn run_campaign(campaign: Campaign, args: &Args) -> Result<Value> {
    let receipt_path = campaign_receipt_path(&args.output, &campaign.dataset);
    if receipt_path.is_file() {
        return Ok(serde_json::from_slice(&fs::read(receipt_path)?)?);
    }
    let root = args.output.join("work").join(&campaign.dataset);
    fs::create_dir_all(&root)?;
    let mut ranked = campaign.seeds.clone();
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.semantic.cmp(&right.semantic))
    });
    ranked.truncate(POPULATION);
    let mut terminal = None;
    for generation in 1..=GENERATIONS {
        let mut population =
            next_population(&campaign.dataset, &ranked, generation, campaign.features)?;
        evaluate_generation(
            &campaign,
            &args.evaluator,
            &root,
            generation,
            &mut population,
            args.evaluator_threads,
        )?;
        ranked.extend(
            population
                .into_iter()
                .filter(|member| member.score.is_finite()),
        );
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.semantic.cmp(&right.semantic))
        });
        ranked.dedup_by(|left, right| left.semantic == right.semantic);
        if let Some(winner) = ranked.iter().find(|member| {
            member.score > campaign.target
                && !score_matches(member.score, campaign.target)
                && member.receipt.is_some()
        }) {
            terminal = Some((generation, winner.clone()));
            break;
        }
        ranked.truncate(POPULATION);
    }
    let receipt = if let Some((generation, winner)) = terminal {
        json!({
            "schema":"sotarena.aspfm.recovery-evolution.v1",
            "status":"target-beaten",
            "task":campaign.task,
            "dataset_hash":campaign.dataset,
            "dataset_id":campaign.dataset_id,
            "historical_target":campaign.target_score,
            "generation":generation,
            "semantic_sha256":winner.semantic,
            "canonical_genome":winner.genome,
            "replayed_score":metric_score(&campaign.task, winner.score),
            "evaluator_receipt":winner.receipt,
        })
    } else {
        json!({
            "schema":"sotarena.aspfm.recovery-evolution.v1",
            "status":"exhausted",
            "task":campaign.task,
            "dataset_hash":campaign.dataset,
            "dataset_id":campaign.dataset_id,
            "historical_target":campaign.target_score,
            "generations":GENERATIONS,
            "best_semantic_sha256":ranked[0].semantic,
            "best_canonical_genome":ranked[0].genome,
            "best_replayed_score":metric_score(&campaign.task, ranked[0].score),
            "best_evaluator_receipt":ranked[0].receipt,
        })
    };
    atomic_json(&receipt_path, &receipt)?;
    Ok(receipt)
}

fn build_campaigns(replay: &Value, args: &Args) -> Result<Vec<Campaign>> {
    let task = args.task.as_str();
    let mut targets = BTreeMap::<String, f64>::new();
    let mut accepted = BTreeMap::<String, Vec<Member>>::new();
    for result in replay["results"]
        .as_array()
        .context("replay results missing")?
    {
        let candidate = &result["candidate"];
        if candidate["task"].as_str() != Some(task) {
            continue;
        }
        let dataset = candidate["dataset_hash"]
            .as_str()
            .context("dataset hash missing")?
            .to_owned();
        if let Some(score) = candidate["recorded_score"].as_f64() {
            targets
                .entry(dataset.clone())
                .and_modify(|current| *current = current.max(score))
                .or_insert(score);
        }
        if result["status"].as_str() == Some("accepted") {
            if let (Some(score), Some(genome), Some(semantic)) = (
                result["replayed_score"].as_f64(),
                candidate["canonical_genome"].as_str(),
                candidate["semantic_sha256"].as_str(),
            ) {
                accepted.entry(dataset).or_default().push(Member {
                    genome: genome.to_owned(),
                    semantic: semantic.to_owned(),
                    score: fitness(task, score),
                    receipt: None,
                });
            }
        }
    }
    let recovery_winners = replay["winners"]
        .as_array()
        .context("replay winners missing")?
        .iter()
        .filter(|winner| {
            winner["task"].as_str() == Some(task)
                && winner["evidence_class"].as_str() == Some("adaptive")
                && winner["candidate"]["provenance_family"].as_str() == Some("recovery-replay")
        })
        .filter_map(|winner| {
            Some((
                winner["dataset_hash"].as_str()?.to_owned(),
                winner["replayed_score"].as_f64()?,
            ))
        })
        .collect::<BTreeMap<_, _>>();
    let datasets = replay["datasets"].as_array().context("datasets missing")?;
    let mut campaigns = Vec::new();
    for dataset in datasets
        .iter()
        .filter(|dataset| dataset["task"].as_str() == Some(task))
    {
        let hash = dataset["dataset_hash"].as_str().context("hash missing")?;
        if args.dataset.as_deref().is_some_and(|wanted| wanted != hash) {
            continue;
        }
        let target_score = if task == "regression" {
            args.historical_target
                .context("regression target missing")?
        } else {
            let Some(target) = targets.get(hash).copied() else {
                continue;
            };
            target
        };
        let target = fitness(task, target_score);
        let Some(winner) = recovery_winners.get(hash).copied() else {
            continue;
        };
        if fitness(task, winner) > target || score_matches(winner, target_score) {
            continue;
        }
        let dataset_id = dataset["dataset_id"]
            .as_str()
            .context("dataset id missing")?;
        let train = args.data_root.join(task).join(dataset_id).join("train.csv");
        let test = args.data_root.join(task).join(dataset_id).join("test.csv");
        ensure!(
            train.is_file() && test.is_file(),
            "missing split for {hash}"
        );
        let mut seeds = accepted
            .remove(hash)
            .context("no accepted recovery seeds")?;
        seeds.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.semantic.cmp(&right.semantic))
        });
        seeds.dedup_by(|left, right| left.semantic == right.semantic);
        ensure!(seeds.len() >= PARENT_POOL, "too few seeds for {hash}");
        campaigns.push(Campaign {
            task: task.to_owned(),
            dataset: hash.to_owned(),
            dataset_id: dataset_id.to_owned(),
            target_score,
            target,
            train_sha256: dataset["train"]["sha256"]
                .as_str()
                .context("train hash missing")?
                .to_owned(),
            test_sha256: dataset["test"]["sha256"]
                .as_str()
                .context("test hash missing")?
                .to_owned(),
            features: numeric_columns(&train)?,
            train,
            test,
            seeds,
        });
    }
    campaigns.sort_by(|left, right| left.dataset.cmp(&right.dataset));
    Ok(campaigns)
}

fn integrate(replay: &mut Value, receipts: &[Value]) -> Result<()> {
    let mut exhaustions = replay["recovery_exhaustions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for campaign in receipts {
        let task = campaign["task"].as_str().context("receipt task missing")?;
        let dataset = campaign["dataset_hash"]
            .as_str()
            .context("receipt dataset missing")?;
        let exhausted = campaign["status"].as_str() == Some("exhausted");
        let semantic = campaign[if exhausted {
            "best_semantic_sha256"
        } else {
            "semantic_sha256"
        }]
        .as_str()
        .context("receipt semantic missing")?;
        let genome = campaign[if exhausted {
            "best_canonical_genome"
        } else {
            "canonical_genome"
        }]
        .as_str()
        .context("receipt genome missing")?;
        ensure!(
            sha256(genome.as_bytes()) == semantic,
            "evolution semantic mismatch"
        );
        let score = campaign[if exhausted {
            "best_replayed_score"
        } else {
            "replayed_score"
        }]
        .as_f64()
        .context("receipt score missing")?;
        let evaluator = &campaign[if exhausted {
            "best_evaluator_receipt"
        } else {
            "evaluator_receipt"
        }];
        let winner_index = replay["winners"]
            .as_array()
            .context("winners missing")?
            .iter()
            .position(|winner| {
                winner["task"].as_str() == Some(task)
                    && winner["dataset_hash"].as_str() == Some(dataset)
                    && winner["evidence_class"].as_str() == Some("adaptive")
            })
            .context("adaptive winner missing")?;
        let current_score = replay["winners"][winner_index]["replayed_score"]
            .as_f64()
            .context("winner score missing")?;
        let improves = if task == "regression" {
            score < current_score
        } else {
            score > current_score
        };
        if improves && evaluator.is_object() {
            let mut candidate = replay["winners"][winner_index]["candidate"].clone();
            let source_bytes = serde_json::to_vec(campaign)?;
            candidate["semantic_sha256"] = json!(semantic);
            candidate["canonical_genome"] = json!(genome);
            candidate["recorded_score"] = Value::Null;
            candidate["benchmark_valid"] = json!(false);
            candidate["evidence_class"] = json!("adaptive");
            candidate["provenance_family"] = json!("recovery-replay");
            candidate["source_path"] =
                json!(format!("recovery://evolution/{task}/{dataset}/{semantic}"));
            candidate["source_sha256"] = json!(sha256(&source_bytes));
            candidate["prediction_sha256"] = evaluator["predictions_sha256"].clone();
            candidate["runtime_seconds"] = evaluator["runtime_seconds"].clone();
            let result = json!({
                "candidate":candidate,
                "status":"accepted","reason":null,"replayed_score":score,
                "prediction_sha256":evaluator["predictions_sha256"],
                "prediction_count":evaluator["prediction_count"],
                "runtime_seconds":evaluator["runtime_seconds"],
                "train_sha256":evaluator["train_sha256"],"test_sha256":evaluator["test_sha256"],
            });
            replay["results"]
                .as_array_mut()
                .context("results missing")?
                .push(result.clone());
            replay["winners"][winner_index] = json!({
                "task":task,"dataset_hash":dataset,"evidence_class":"adaptive",
                "candidate":result["candidate"],"replayed_score":score,
                "prediction_sha256":result["prediction_sha256"],"prediction_count":result["prediction_count"],
                "runtime_seconds":result["runtime_seconds"],"train_sha256":result["train_sha256"],"test_sha256":result["test_sha256"],
            });
        }
        if exhausted {
            let winner = &replay["winners"][winner_index];
            let source_bytes = serde_json::to_vec(campaign)?;
            exhaustions.push(json!({
                "schema":"sotarena.aspfm.recovery-evolution.v1",
                "status":"exhausted",
                "task":task,
                "dataset_hash":dataset,
                "dataset_id":campaign["dataset_id"],
                "historical_target":campaign["historical_target"],
                "generations":campaign["generations"],
                "best_semantic_sha256":winner["candidate"]["semantic_sha256"],
                "best_canonical_genome":winner["candidate"]["canonical_genome"],
                "best_replayed_score":winner["replayed_score"],
                "source_sha256":sha256(&source_bytes),
            }));
        }
    }
    replay["results"]
        .as_array_mut()
        .context("results missing")?
        .sort_by(|left, right| {
            (
                left["candidate"]["task"].as_str(),
                left["candidate"]["dataset_hash"].as_str(),
                left["candidate"]["semantic_sha256"].as_str(),
            )
                .cmp(&(
                    right["candidate"]["task"].as_str(),
                    right["candidate"]["dataset_hash"].as_str(),
                    right["candidate"]["semantic_sha256"].as_str(),
                ))
        });
    replay["winners"]
        .as_array_mut()
        .context("winners missing")?
        .sort_by(|left, right| {
            (
                left["task"].as_str(),
                left["dataset_hash"].as_str(),
                left["evidence_class"].as_str(),
            )
                .cmp(&(
                    right["task"].as_str(),
                    right["dataset_hash"].as_str(),
                    right["evidence_class"].as_str(),
                ))
        });
    replay["recovery_exhaustions"] = Value::Array(exhaustions);
    Ok(())
}

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
