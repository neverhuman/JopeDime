//! Execute pending three-lane canaries through pinned proposal and evaluator jobs.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_ingest::ingest_receipt_with_manifest;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, checkpoint_epoch, evaluation_protocol_sha256, now_ms, read_json, sha256_bytes,
    sha256_file, DEFAULT_ROOT, DEFAULT_SCQ,
};
use jope_dime_regressor::selfplay_shadow::{
    CanaryContextClass, CanaryContextV2, CanaryIntentV2, CanaryMode, Sha256Digest, Task,
    TaskCanaryAcceptanceV2, TaskCanaryMetricsV2, CANARY_INTENT_SCHEMA,
    TASK_CANARY_ACCEPTANCE_V2_SCHEMA,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

const ACTIVE_STATES: &[&str] = &["queued", "starting", "dispatching", "running", "held"];

#[derive(Parser, Debug)]
#[command(name = "jope-canary-executor")]
struct Args {
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
    #[arg(long, default_value = DEFAULT_SCQ)]
    scq: PathBuf,
    #[arg(
        long,
        default_value = "/home/ubuntu/jope-flywheel/scripts/fweval_runner.sh"
    )]
    runner: PathBuf,
    #[arg(long)]
    once: bool,
    #[arg(long, default_value_t = 30)]
    interval_seconds: u64,
}

#[derive(Clone, Debug)]
struct PendingIntent {
    path: PathBuf,
    value: CanaryIntentV2,
    created_at_ms: u64,
}

#[derive(Clone, Debug, Default)]
struct LaneMetrics {
    requested: usize,
    valid: usize,
    scores: Vec<f64>,
    runtimes: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct AggregateMetrics {
    candidate: LaneMetrics,
    accepted: LaneMetrics,
    gp: LaneMetrics,
    candidate_novel: usize,
    strict_wins: usize,
    context_gaps: Vec<(CanaryContextClass, f64)>,
    candidate_pairwise_gaps: Vec<f64>,
    gp_pairwise_gaps: Vec<f64>,
}

fn normalized_intent(path: &Path) -> Result<CanaryIntentV2> {
    let value = read_json(path)?;
    if value.get("schema").and_then(Value::as_str) == Some(CANARY_INTENT_SCHEMA) {
        let intent: CanaryIntentV2 = serde_json::from_value(value)?;
        intent.validate()?;
        return Ok(intent);
    }
    let schema = value
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or_default();
    ensure!(
        matches!(schema, "ShadowCanaryIntent.v1" | "TaskCanaryIntent.v1"),
        "unsupported canary intent schema"
    );
    let mode = if schema == "ShadowCanaryIntent.v1" {
        CanaryMode::Shadow
    } else {
        CanaryMode::Production
    };
    let task: Task = value
        .get("task")
        .and_then(Value::as_str)
        .context("legacy canary intent lacks task")?
        .parse()?;
    let candidate = value
        .get("shadow_checkpoint_sha256")
        .or_else(|| value.get("candidate_checkpoint_sha256"))
        .and_then(Value::as_str)
        .context("legacy canary intent lacks candidate checkpoint")?;
    let contexts = value
        .get("contexts")
        .and_then(Value::as_array)
        .context("legacy canary intent lacks contexts")?
        .iter()
        .map(|row| {
            let class = match row.get("kind").and_then(Value::as_str) {
                Some("frontier") => CanaryContextClass::Frontier,
                Some("hard") => CanaryContextClass::Hard,
                Some("anchor") => CanaryContextClass::Anchor,
                _ => bail!("legacy canary context class is invalid"),
            };
            Ok(CanaryContextV2 {
                context_identity: row
                    .get("dataset_hash")
                    .and_then(Value::as_str)
                    .context("legacy canary context lacks dataset")?
                    .to_owned(),
                class,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let intent = CanaryIntentV2 {
        schema: CANARY_INTENT_SCHEMA.to_owned(),
        mode,
        task,
        candidate_checkpoint_sha256: Sha256Digest::parse(candidate.to_owned())?,
        accepted_checkpoint_sha256: Sha256Digest::parse(
            value
                .get("accepted_parent_checkpoint_sha256")
                .and_then(Value::as_str)
                .context("legacy canary intent lacks accepted parent")?
                .to_owned(),
        )?,
        run_index: value
            .get("run_index")
            .or_else(|| value.get("independent_run"))
            .and_then(Value::as_u64)
            .map(|run| run.saturating_sub(u64::from(schema == "TaskCanaryIntent.v1")) as usize)
            .unwrap_or_default(),
        seed: value
            .get("seed")
            .and_then(Value::as_u64)
            .unwrap_or(20_260_726),
        contexts,
        candidate_proposals_per_context: 8,
        accepted_baseline_proposals_per_context: 8,
        zero_dl_gp_proposals_per_context: 8,
        created_at_ms: value
            .get("created_at_ms")
            .and_then(Value::as_u64)
            .map(u128::from),
    };
    intent.validate()?;
    Ok(intent)
}

fn pending_intents(root: &Path) -> Result<Vec<PendingIntent>> {
    let mut output = Vec::new();
    for relative in [
        "training/shadow-canaries/pending",
        "training/task-canaries/pending",
    ] {
        let directory = root.join(relative);
        if !directory.is_dir() {
            continue;
        }
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let path = entry.path();
            let raw = read_json(&path)?;
            output.push(PendingIntent {
                value: normalized_intent(&path)?,
                created_at_ms: raw
                    .get("created_at_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or_default(),
                path,
            });
        }
    }
    output.sort_by(|left, right| {
        left.value
            .mode
            .cmp(&right.value.mode)
            .then(left.created_at_ms.cmp(&right.created_at_ms))
            .then(
                left.value
                    .candidate_checkpoint_sha256
                    .cmp(&right.value.candidate_checkpoint_sha256),
            )
            .then(left.value.run_index.cmp(&right.value.run_index))
            .then(left.path.cmp(&right.path))
    });
    Ok(output)
}

fn production_predecessor_passed(root: &Path, intent: &CanaryIntentV2) -> Result<bool> {
    if intent.mode != CanaryMode::Production || intent.run_index == 0 {
        return Ok(true);
    }
    let directory = root.join("training/task-canaries/accepted");
    if !directory.is_dir() {
        return Ok(false);
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let value = read_json(&path)?;
        if value.get("schema").and_then(Value::as_str) == Some(TASK_CANARY_ACCEPTANCE_V2_SCHEMA)
            && value.get("task").and_then(Value::as_str) == Some(intent.task.as_str())
            && value
                .get("candidate_checkpoint_sha256")
                .and_then(Value::as_str)
                == Some(intent.candidate_checkpoint_sha256.as_str())
            && value.get("run_index").and_then(Value::as_u64) == Some((intent.run_index - 1) as u64)
        {
            return Ok(value.get("accepted").and_then(Value::as_bool) == Some(true));
        }
    }
    Ok(false)
}

fn checkpoint_paths(root: &Path, intent: &CanaryIntentV2) -> Result<(PathBuf, PathBuf)> {
    let accepted = read_json(&root.join("training/accepted-by-task.json"))?;
    let accepted_row = accepted
        .get("tasks")
        .and_then(|value| value.get(intent.task.as_str()))
        .context("accepted task pointer is absent")?;
    ensure!(
        accepted_row
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            == Some(intent.accepted_checkpoint_sha256.as_str()),
        "canary accepted parent is no longer current"
    );
    let accepted_path = PathBuf::from(
        accepted_row
            .get("checkpoint")
            .and_then(Value::as_str)
            .context("accepted task checkpoint path is absent")?,
    );
    ensure!(
        sha256_file(&accepted_path)? == intent.accepted_checkpoint_sha256.as_str(),
        "accepted canary checkpoint content drift"
    );
    let candidate_path = match intent.mode {
        CanaryMode::Shadow => root
            .join("training/shadow-checkpoints")
            .join(intent.task.as_str())
            .join(format!(
                "{}.model.pt",
                intent.candidate_checkpoint_sha256.as_str()
            )),
        CanaryMode::Production => {
            let metadata = root
                .join("training/production-candidates")
                .join(intent.task.as_str())
                .join(format!(
                    "{}.candidate.v1.json",
                    intent.candidate_checkpoint_sha256.as_str()
                ));
            PathBuf::from(
                read_json(&metadata)?
                    .get("checkpoint")
                    .and_then(Value::as_str)
                    .context("production candidate path is absent")?,
            )
        }
    };
    ensure!(
        candidate_path.is_file()
            && sha256_file(&candidate_path)? == intent.candidate_checkpoint_sha256.as_str(),
        "candidate canary checkpoint content drift"
    );
    Ok((candidate_path, accepted_path))
}

fn runtime_bundle() -> Result<PathBuf> {
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    executable
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .context("canary executor is not running from a runtime bundle")
}

fn toml_quote(value: &str) -> Result<String> {
    ensure!(
        !value.contains(['"', '\\', '\n', '\r']),
        "unsafe generated TOML value"
    );
    Ok(format!("\"{value}\""))
}

fn append_input(lines: &mut Vec<String>, name: &str, source: &Path, target: &str) -> Result<()> {
    lines.extend([
        "[[input]]".to_owned(),
        format!("name = {}", toml_quote(name)?),
        format!("source = {}", toml_quote(&source.to_string_lossy())?),
        format!("target = {}", toml_quote(target)?),
        format!("sha256 = {}", toml_quote(&sha256_file(source)?)?),
        String::new(),
    ]);
    Ok(())
}

fn write_immutable(path: &Path, body: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(fs::read(path)? == body, "immutable canary artifact drift");
        return Ok(());
    }
    fs::create_dir_all(path.parent().context("immutable path has no parent")?)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(body)?;
    file.sync_data()?;
    Ok(())
}

fn canary_lane_ancestry(mode: CanaryMode, lane: &str) -> &'static str {
    match lane {
        "gp" => "none",
        "candidate" if mode == CanaryMode::Shadow => "shadow_dl",
        _ => "accepted_dl",
    }
}

fn scq_jobs(args: &Args) -> Result<BTreeMap<String, String>> {
    let output = Command::new(&args.scq).args(["--json", "ls"]).output()?;
    ensure!(output.status.success(), "SCQ list failed");
    let value: Value = serde_json::from_slice(&output.stdout)?;
    Ok(value
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|row| {
            Some((
                row.get("id")?.as_str()?.to_owned(),
                row.get("state")?.as_str()?.to_owned(),
            ))
        })
        .collect())
}

fn submit_spec(args: &Args, spec: &Path) -> Result<()> {
    let spec_text = spec.to_string_lossy();
    let check = Command::new(&args.scq)
        .args(["check", spec_text.as_ref()])
        .output()?;
    ensure!(
        check.status.success(),
        "SCQ check failed: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    let submit = Command::new(&args.scq)
        .args(["submit", spec_text.as_ref()])
        .output()?;
    let message = String::from_utf8_lossy(if submit.stderr.is_empty() {
        &submit.stdout
    } else {
        &submit.stderr
    });
    ensure!(
        submit.status.success() || message.to_ascii_lowercase().contains("already exists"),
        "SCQ submit failed: {}",
        message.trim()
    );
    Ok(())
}

fn recursive_find(path: &Path, name: &str, output: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            recursive_find(&path, name, output);
        } else if path.file_name().and_then(|value| value.to_str()) == Some(name) {
            output.push(path);
        }
    }
}

fn fetch(args: &Args, job_id: &str, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    let result = Command::new(&args.scq)
        .args([
            "fetch",
            job_id,
            "--output",
            output.to_string_lossy().as_ref(),
        ])
        .output()?;
    ensure!(
        result.status.success(),
        "SCQ fetch failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}

fn proposal_job_id(intent_hash: &str, worker_sha256: &str) -> String {
    format!(
        "jope-canary-propose-{}-{}",
        &intent_hash[..12],
        &worker_sha256[..8]
    )
}

fn ensure_proposals(
    args: &Args,
    pending: &PendingIntent,
    work: &Path,
    jobs: &BTreeMap<String, String>,
) -> Result<Option<PathBuf>> {
    let proposal_set = work.join("canary-proposals.json");
    if proposal_set.is_file() {
        return Ok(Some(proposal_set));
    }
    let intent_path = work.join("intent.v2.json");
    atomic_json(&intent_path, &serde_json::to_value(&pending.value)?)?;
    let (candidate, accepted) = checkpoint_paths(&args.root, &pending.value)?;
    let bundle = runtime_bundle()?;
    let pointer = read_json(&args.root.join("catalog/current.json"))?;
    let catalog = PathBuf::from(
        pointer
            .get("catalog")
            .and_then(Value::as_str)
            .context("catalog pointer lacks catalog")?,
    );
    let data = PathBuf::from(
        pointer
            .get("data")
            .and_then(Value::as_str)
            .context("catalog pointer lacks data")?,
    );
    let seed_bank = args.root.join("corpus/robust-gp-seed-bank.v1.json");
    let intent_hash = sha256_file(&intent_path)?;
    let worker = bundle.join("pytorch/proposals/test_exact_canary_propose.py");
    let implementation_sha256 = sha256_bytes(
        format!(
            "{}\0{}",
            sha256_file(&worker)?,
            sha256_file(&std::env::current_exe()?)?
        )
        .as_bytes(),
    );
    let job_id = proposal_job_id(&intent_hash, &implementation_sha256);
    let spec = work.join(format!("{job_id}.toml"));
    if !spec.is_file() {
        let pythonpath = bundle.join("pytorch");
        let mut lines = vec![
            "[job]".to_owned(),
            format!("id = {}", toml_quote(&job_id)?),
            "project = \"jope-selfplay-train\"".to_owned(),
            "tier = \"normal\"".to_owned(),
            "command = \"/usr/bin/python3\"".to_owned(),
            "args = [\"{input:worker}\", \"--intent\", \"{input:intent}\", \"--generator\", \"{input:generator}\", \"--catalog-pointer\", \"{input:pointer}\", \"--catalog\", \"{input:catalog}\", \"--data\", \"{input:data}\", \"--candidate\", \"{input:candidate}\", \"--accepted\", \"{input:accepted}\", \"--seed-bank\", \"{input:seedbank}\", \"--output\", \"{output:canary}/canary-proposals.json\"]".to_owned(),
            String::new(),
            "[resource]".to_owned(),
            "gpus_per_task = 1".to_owned(),
            "cpus_reserved = 4".to_owned(),
            "omp_threads = 4".to_owned(),
            "ram_mb = 12288".to_owned(),
            "gang_size = 1".to_owned(),
            "placement = \"single_node\"".to_owned(),
            String::new(),
            "[[dependencies.smoke]]".to_owned(),
            "name = \"xbabe1 canary proposal GPU capacity\"".to_owned(),
            "command = \"/usr/bin/nvidia-smi\"".to_owned(),
            "args = [\"-i\", \"GPU-784030c4-b1f9-d5a3-f0eb-9337f133f503\", \"--query-gpu=uuid\", \"--format=csv,noheader\"]".to_owned(),
            "timeout_secs = 5".to_owned(),
            String::new(),
            "[[dependencies.smoke]]".to_owned(),
            "name = \"xbabe1 RuntimeBundle Python package\"".to_owned(),
            "command = \"/usr/bin/env\"".to_owned(),
            format!(
                "args = [{}, \"/usr/bin/python3\", \"-c\", \"import jope\"]",
                toml_quote(&format!("PYTHONPATH={}", pythonpath.to_string_lossy()))?
            ),
            "timeout_secs = 30".to_owned(),
            String::new(),
            "[env]".to_owned(),
            format!("PYTHONPATH = {}", toml_quote(&pythonpath.to_string_lossy())?),
            "OMP_NUM_THREADS = \"4\"".to_owned(),
            String::new(),
            "[preempt]".to_owned(),
            "preemptible = true".to_owned(),
            "checkpoint = \"none\"".to_owned(),
            "resume = \"rerun\"".to_owned(),
            String::new(),
            "[limits]".to_owned(),
            "max_runtime_hours = 24".to_owned(),
            "max_restarts = 1".to_owned(),
            String::new(),
        ];
        append_input(
            &mut lines,
            "worker",
            &worker,
            "test_exact_canary_propose.py",
        )?;
        append_input(
            &mut lines,
            "generator",
            &bundle.join("pytorch/proposals/test_exact_dl_generate.py"),
            "test_exact_dl_generate.py",
        )?;
        append_input(
            &mut lines,
            "common",
            &bundle.join("pytorch/shared/test_exact_common.py"),
            "test_exact_common.py",
        )?;
        append_input(&mut lines, "intent", &intent_path, "intent.json")?;
        append_input(
            &mut lines,
            "pointer",
            &args.root.join("catalog/current.json"),
            "catalog/pointer.json",
        )?;
        append_input(&mut lines, "catalog", &catalog, "catalog/catalog.json")?;
        append_input(&mut lines, "data", &data, "catalog/contexts.f32")?;
        append_input(
            &mut lines,
            "candidate",
            &candidate,
            &format!(
                "{}.model.pt",
                pending.value.candidate_checkpoint_sha256.as_str()
            ),
        )?;
        append_input(
            &mut lines,
            "accepted",
            &accepted,
            &format!(
                "{}.model.pt",
                pending.value.accepted_checkpoint_sha256.as_str()
            ),
        )?;
        append_input(&mut lines, "seedbank", &seed_bank, "seed-bank.json")?;
        lines.extend([
            "[[output]]".to_owned(),
            "name = \"canary\"".to_owned(),
            "source = \"canary\"".to_owned(),
            "directory = true".to_owned(),
            String::new(),
        ]);
        write_immutable(&spec, lines.join("\n").as_bytes())?;
    }
    match jobs.get(&job_id).map(String::as_str) {
        None => {
            submit_spec(args, &spec)?;
            Ok(None)
        }
        Some("completed") => {
            let fetched = work.join("proposal-fetch");
            fetch(args, &job_id, &fetched)?;
            let mut found = Vec::new();
            recursive_find(&fetched, "canary-proposals.json", &mut found);
            let source = found
                .into_iter()
                .next()
                .context("canary proposal output is absent")?;
            let bytes = fs::read(source)?;
            write_immutable(&proposal_set, &bytes)?;
            Ok(Some(proposal_set))
        }
        Some(state) if ACTIVE_STATES.contains(&state) => Ok(None),
        Some(state) => bail!("canary proposal job is terminal in state {state}"),
    }
}

fn catalog_entries(root: &Path) -> Result<(Value, BTreeMap<(Task, String), Value>)> {
    let pointer = read_json(&root.join("catalog/current.json"))?;
    let catalog = read_json(Path::new(
        pointer
            .get("catalog")
            .and_then(Value::as_str)
            .context("catalog pointer lacks catalog")?,
    ))?;
    let mut entries = BTreeMap::new();
    for row in catalog
        .get("entries")
        .and_then(Value::as_array)
        .context("catalog lacks entries")?
    {
        let task: Task = row
            .get("task")
            .and_then(Value::as_str)
            .context("catalog entry lacks task")?
            .parse()?;
        let dataset = row
            .get("dataset_hash")
            .and_then(Value::as_str)
            .context("catalog entry lacks dataset")?
            .to_owned();
        entries.insert((task, dataset), row.clone());
    }
    Ok((pointer, entries))
}

fn build_evaluations(
    args: &Args,
    pending: &PendingIntent,
    work: &Path,
    proposal_set: &Path,
) -> Result<Vec<(String, PathBuf)>> {
    let proposals = read_json(proposal_set)?;
    ensure!(
        proposals.get("schema").and_then(Value::as_str) == Some("CanaryProposalSet.v2"),
        "canary proposal output schema drift"
    );
    let (pointer, catalog) = catalog_entries(&args.root)?;
    let candidate = pending.value.candidate_checkpoint_sha256.as_str();
    let hall = args.root.join("corpus/robust-gp-seed-bank.v1.json");
    let hall_sha = sha256_file(&hall)?;
    let generation = format!("strict-wave-{:06}", 900_000 + pending.value.run_index);
    let mut output = Vec::new();
    for (index, context) in proposals
        .get("contexts")
        .and_then(Value::as_array)
        .context("canary proposal set lacks contexts")?
        .iter()
        .enumerate()
    {
        let dataset = context
            .get("context_identity")
            .and_then(Value::as_str)
            .context("canary proposal context lacks identity")?;
        let entry = catalog
            .get(&(pending.value.task, dataset.to_owned()))
            .context("canary context is absent from current catalog")?;
        let train = PathBuf::from(
            entry
                .get("train_csv")
                .and_then(Value::as_str)
                .context("canary catalog entry lacks train CSV")?,
        );
        let test = PathBuf::from(
            entry
                .get("test_csv")
                .and_then(Value::as_str)
                .context("canary catalog entry lacks test CSV")?,
        );
        let train_sha = entry
            .get("train_sha256")
            .and_then(Value::as_str)
            .context("canary catalog entry lacks train hash")?;
        let test_sha = entry
            .get("test_sha256")
            .and_then(Value::as_str)
            .context("canary catalog entry lacks test hash")?;
        ensure!(
            sha256_file(&train)? == train_sha && sha256_file(&test)? == test_sha,
            "canary dataset content drift"
        );
        let metric = if pending.value.task == Task::Binary {
            "auc"
        } else {
            "r2"
        };
        let protocol = evaluation_protocol_sha256(
            pending.value.task.as_str(),
            metric,
            "test_exact",
            train_sha,
            test_sha,
        )?;
        let mut candidates = Vec::new();
        let mut seen = BTreeSet::new();
        let mut requested = serde_json::Map::new();
        for (lane, source, origin) in [
            ("candidate", "canary_candidate_dl", "dl_direct"),
            ("accepted", "canary_accepted_baseline", "accepted_baseline"),
            ("gp", "canary_zero_dl_gp", "gp_comparator"),
        ] {
            let mut semantics = Vec::new();
            for (slot, row) in context
                .get(lane)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                let semantic = row
                    .get("semantic_sha256")
                    .and_then(Value::as_str)
                    .context("canary proposal lacks semantic")?;
                semantics.push(semantic.to_owned());
                if seen.insert(semantic.to_owned()) {
                    candidates.push(json!({
                        "canonical_genome": row.get("canonical_genome"),
                        "semantic_sha256": semantic,
                        "origin": origin,
                        "source": source,
                        "slot_group": lane,
                        "slot_index": slot,
                        "evidence": {
                            "canary_lane": lane,
                            "dl_ancestry_kind": canary_lane_ancestry(pending.value.mode, lane),
                        },
                    }));
                }
            }
            requested.insert(lane.to_owned(), json!(semantics));
        }
        let frozen_seed = sha256_bytes(&serde_json::to_vec(&json!({
            "intent": sha256_file(&pending.path)?,
            "dataset": dataset,
            "requested": requested,
        }))?);
        let manifest_path = work.join(format!("manifest-{index:02}.json"));
        let generated_at_ms = pending.created_at_ms as u128;
        let manifest = json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "receipt_purpose": "canary",
            "task": pending.value.task,
            "dataset_hash": dataset,
            "generation": generation,
            "strict_wave": generation,
            "evaluation_lane": "canary",
            "micro_lane": context.get("class"),
            "catalog_sha256": pointer.get("catalog_sha256"),
            "checkpoint_sha256": candidate,
            "checkpoint_epoch": checkpoint_epoch(candidate)?,
            "train_csv": train,
            "train_sha256": train_sha,
            "test_csv": test,
            "test_sha256": test_sha,
            "protocol_sha256": protocol,
            "generated_at_ms": generated_at_ms,
            "label_authority": {
                "score_tier": "test_exact",
                "metric": metric,
                "test_adaptive": true,
                "benchmark_valid": false,
                "selection_bias": "test-exact-reused",
                "generation": generation,
                "frozen_seed_sha256": frozen_seed,
                "model_checkpoint_sha256": candidate,
                "hall_of_fame_sha256": hall_sha,
            },
            "canary": {
                "mode": pending.value.mode,
                "run_index": pending.value.run_index,
                "accepted_parent_checkpoint_sha256": pending.value.accepted_checkpoint_sha256,
                "candidate_checkpoint_sha256": pending.value.candidate_checkpoint_sha256,
                "requested": requested,
            },
            "candidates": candidates,
        });
        if manifest_path.is_file() {
            let existing = read_json(&manifest_path)?;
            ensure!(
                existing.get("schema").and_then(Value::as_str)
                    == Some("JopeFlywheelCorpusEvalShard.v1")
                    && existing.get("receipt_purpose").and_then(Value::as_str) == Some("canary")
                    && existing.get("task") == manifest.get("task")
                    && existing.get("dataset_hash") == manifest.get("dataset_hash")
                    && existing.get("generation") == manifest.get("generation")
                    && existing.get("checkpoint_sha256") == manifest.get("checkpoint_sha256")
                    && existing.get("train_sha256") == manifest.get("train_sha256")
                    && existing.get("test_sha256") == manifest.get("test_sha256")
                    && existing.pointer("/canary/mode") == manifest.pointer("/canary/mode")
                    && existing.pointer("/canary/run_index")
                        == manifest.pointer("/canary/run_index")
                    && existing.pointer("/canary/accepted_parent_checkpoint_sha256")
                        == manifest.pointer("/canary/accepted_parent_checkpoint_sha256")
                    && existing.pointer("/canary/candidate_checkpoint_sha256")
                        == manifest.pointer("/canary/candidate_checkpoint_sha256")
                    && existing.pointer("/canary/requested")
                        == manifest.pointer("/canary/requested"),
                "existing immutable canary manifest is not bound to the active intent"
            );
        } else {
            write_immutable(&manifest_path, &serde_json::to_vec(&manifest)?)?;
        }
        let intent_hash = sha256_file(&pending.path)?;
        let job_id = format!("jte-canary-{}-{index:02}", &intent_hash[..16]);
        let spec_path = work.join(format!("{job_id}.toml"));
        if !spec_path.is_file() {
            let bundle = runtime_bundle()?;
            let evaluator = bundle.join("runtime/bin/corpus_eval");
            let mut lines = vec![
                "[job]".to_owned(),
                format!("id = {}", toml_quote(&job_id)?),
                "project = \"jope-selfplay-eval\"".to_owned(),
                "tier = \"normal\"".to_owned(),
                "command = \"/usr/bin/env\"".to_owned(),
                "args = [\"{input:runner}\", \"{input:manifest}\", \"{input:train}\", \"{input:test}\", \"{input:evaluator}\", \"{output:receipt}\"]".to_owned(),
                String::new(),
                "[resource]".to_owned(),
                "gpus_per_task = 0".to_owned(),
                "cpus_reserved = 1".to_owned(),
                "omp_threads = 1".to_owned(),
                "ram_mb = 512".to_owned(),
                "gang_size = 1".to_owned(),
                "placement = \"single_node\"".to_owned(),
                String::new(),
                "[env]".to_owned(),
                "RAYON_NUM_THREADS = \"1\"".to_owned(),
                "OMP_NUM_THREADS = \"1\"".to_owned(),
                String::new(),
                "[preempt]".to_owned(),
                "preemptible = true".to_owned(),
                "checkpoint = \"none\"".to_owned(),
                "resume = \"rerun\"".to_owned(),
                String::new(),
                "[limits]".to_owned(),
                "max_runtime_hours = 24".to_owned(),
                "max_restarts = 1".to_owned(),
                String::new(),
            ];
            append_input(&mut lines, "runner", &args.runner, "runner.sh")?;
            append_input(&mut lines, "manifest", &manifest_path, "manifest.json")?;
            append_input(&mut lines, "train", &train, "train.csv")?;
            append_input(&mut lines, "test", &test, "test.csv")?;
            append_input(&mut lines, "evaluator", &evaluator, "corpus_eval")?;
            lines.extend([
                "[[output]]".to_owned(),
                "name = \"receipt\"".to_owned(),
                "source = \"receipt\"".to_owned(),
                "directory = true".to_owned(),
                String::new(),
            ]);
            write_immutable(&spec_path, lines.join("\n").as_bytes())?;
        }
        output.push((job_id, manifest_path));
    }
    Ok(output)
}

fn ensure_evaluations(
    args: &Args,
    work: &Path,
    evaluations: &[(String, PathBuf)],
    jobs: &BTreeMap<String, String>,
) -> Result<Option<Vec<PathBuf>>> {
    let mut fetched_outputs = Vec::new();
    let mut waiting = false;
    for (job_id, manifest) in evaluations {
        let spec = work.join(format!("{job_id}.toml"));
        match jobs.get(job_id).map(String::as_str) {
            None => {
                submit_spec(args, &spec)?;
                waiting = true;
            }
            Some("completed") => {
                let output = work.join("evaluation-fetch").join(job_id);
                if !output.join("FETCHED").is_file() {
                    fetch(args, job_id, &output)?;
                    fs::write(output.join("FETCHED"), b"ok\n")?;
                }
                let mut receipts = Vec::new();
                recursive_find(
                    &output,
                    &format!(
                        "{}.receipts.jsonl",
                        read_json(manifest)?
                            .get("dataset_hash")
                            .and_then(Value::as_str)
                            .context("canary manifest lacks dataset")?
                    ),
                    &mut receipts,
                );
                let receipt = receipts
                    .into_iter()
                    .next()
                    .context("canary receipt is absent")?;
                let receipt_root = receipt
                    .parent()
                    .context("canary receipt has no parent")?
                    .to_path_buf();
                ingest_receipt_with_manifest(
                    &args.root.join("corpus/strict-test-exact.sqlite"),
                    job_id,
                    &receipt_root,
                    manifest,
                )?;
                fetched_outputs.push(receipt);
            }
            Some(state) if ACTIVE_STATES.contains(&state) => waiting = true,
            Some(state @ ("failed" | "cancelled")) => {
                // A failed canary member is evidence of zero validity, not a
                // reason to wedge the executor forever. Keep an explicit empty
                // receipt plus a sidecar; summarize counts all requested slots
                // as invalid and marks replay verification incomplete. No
                // synthetic score is ingested or imputed.
                let failures = work.join("evaluation-failures");
                fs::create_dir_all(&failures)?;
                let receipt = failures.join(format!("{job_id}.receipts.jsonl"));
                write_immutable(&receipt, b"")?;
                atomic_json(
                    &failures.join(format!("{job_id}.v1.json")),
                    &json!({
                        "schema": "CanaryEvaluationFailure.v1",
                        "job_id": job_id,
                        "terminal_state": state,
                        "manifest_sha256": sha256_file(manifest)?,
                        "imputed": false,
                    }),
                )?;
                fetched_outputs.push(receipt);
            }
            Some(state) => bail!("{job_id}: canary evaluator has unknown terminal state {state}"),
        }
    }
    Ok((!waiting && fetched_outputs.len() == evaluations.len()).then_some(fetched_outputs))
}

fn lane_metrics(requested: &[Value], scores: &BTreeMap<String, (f64, f64)>) -> LaneMetrics {
    let mut output = LaneMetrics {
        requested: requested.len(),
        ..LaneMetrics::default()
    };
    let mut seen = BTreeSet::new();
    for semantic in requested.iter().filter_map(Value::as_str) {
        if !seen.insert(semantic) {
            continue;
        }
        if let Some((score, runtime)) = scores.get(semantic) {
            output.valid += 1;
            output.scores.push(*score);
            output.runtimes.push(*runtime);
        }
    }
    output
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn summarize(
    args: &Args,
    pending: &PendingIntent,
    proposal_set: &Path,
    receipts: &[PathBuf],
) -> Result<Value> {
    let proposals = read_json(proposal_set)?;
    let contexts = proposals
        .get("contexts")
        .and_then(Value::as_array)
        .context("canary proposals lack contexts")?;
    ensure!(
        contexts.len() == receipts.len(),
        "canary receipt count drift"
    );
    let replay_verification_complete = receipts.iter().all(|path| {
        path.metadata()
            .map(|value| value.len() > 0)
            .unwrap_or(false)
    });
    let mut aggregate = AggregateMetrics::default();
    let mut merit_winners = Vec::new();
    for (context, receipt_path) in contexts.iter().zip(receipts) {
        let mut scores = BTreeMap::new();
        for line in fs::read_to_string(receipt_path)?.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let row: Value = serde_json::from_str(line)?;
            if row.get("valid").and_then(Value::as_bool) != Some(true) {
                continue;
            }
            scores.insert(
                row.get("semantic_sha256")
                    .and_then(Value::as_str)
                    .context("canary receipt lacks semantic")?
                    .to_owned(),
                (
                    row.get("score")
                        .and_then(Value::as_f64)
                        .context("canary receipt lacks score")?,
                    row.get("runtime_seconds")
                        .and_then(Value::as_f64)
                        .context("canary receipt lacks runtime")?,
                ),
            );
        }
        let manifest = read_json(
            &args
                .root
                .join("training/canary-executor/work")
                .join(sha256_file(&pending.path)?)
                .join(format!("manifest-{:02}.json", aggregate.context_gaps.len())),
        )?;
        let requested = manifest
            .pointer("/canary/requested")
            .and_then(Value::as_object)
            .context("canary manifest lacks requested lanes")?;
        let candidate = lane_metrics(
            requested["candidate"]
                .as_array()
                .context("candidate lane absent")?,
            &scores,
        );
        let accepted = lane_metrics(
            requested["accepted"]
                .as_array()
                .context("accepted lane absent")?,
            &scores,
        );
        let gp = lane_metrics(
            requested["gp"].as_array().context("GP lane absent")?,
            &scores,
        );
        let candidate_semantics = requested["candidate"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<BTreeSet<_>>();
        let other_semantics = requested["accepted"]
            .as_array()
            .into_iter()
            .flatten()
            .chain(requested["gp"].as_array().into_iter().flatten())
            .filter_map(Value::as_str)
            .collect::<BTreeSet<_>>();
        aggregate.candidate_novel += candidate_semantics.difference(&other_semantics).count();
        let mut candidate_sorted = candidate.scores.clone();
        let mut accepted_sorted = accepted.scores.clone();
        let mut gp_sorted = gp.scores.clone();
        candidate_sorted.sort_by(|a, b| b.total_cmp(a));
        accepted_sorted.sort_by(|a, b| b.total_cmp(a));
        gp_sorted.sort_by(|a, b| b.total_cmp(a));
        aggregate.candidate_pairwise_gaps.extend(
            candidate_sorted
                .iter()
                .zip(&accepted_sorted)
                .map(|(candidate, accepted)| candidate - accepted),
        );
        aggregate.gp_pairwise_gaps.extend(
            candidate_sorted
                .iter()
                .zip(&gp_sorted)
                .map(|(candidate, gp)| candidate - gp),
        );
        let candidate_best = candidate_sorted
            .first()
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let accepted_best = accepted_sorted
            .first()
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        if candidate_best > accepted_best {
            aggregate.strict_wins += 1;
            if let Some(semantic) = requested["candidate"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .find(|semantic| {
                    scores
                        .get(*semantic)
                        .is_some_and(|(score, _)| *score == candidate_best)
                })
            {
                merit_winners.push(semantic.to_owned());
            }
        }
        let class: CanaryContextClass = serde_json::from_value(
            context
                .get("class")
                .cloned()
                .context("canary class absent")?,
        )?;
        aggregate
            .context_gaps
            .push((class, candidate_best - accepted_best));
        aggregate.candidate.requested += candidate.requested;
        aggregate.candidate.valid += candidate.valid;
        aggregate.candidate.scores.extend(candidate.scores);
        aggregate.candidate.runtimes.extend(candidate.runtimes);
        aggregate.accepted.requested += accepted.requested;
        aggregate.accepted.valid += accepted.valid;
        aggregate.accepted.scores.extend(accepted.scores);
        aggregate.accepted.runtimes.extend(accepted.runtimes);
        aggregate.gp.requested += gp.requested;
        aggregate.gp.valid += gp.valid;
        aggregate.gp.scores.extend(gp.scores);
        aggregate.gp.runtimes.extend(gp.runtimes);
    }
    let validity = |lane: &LaneMetrics| {
        if lane.requested == 0 {
            0.0
        } else {
            lane.valid as f64 / lane.requested as f64
        }
    };
    let candidate_validity = validity(&aggregate.candidate);
    let accepted_validity = validity(&aggregate.accepted);
    let gp_validity = validity(&aggregate.gp);
    let pairwise_lift = mean(&aggregate.candidate_pairwise_gaps);
    let gp_lift = mean(&aggregate.gp_pairwise_gaps);
    let latency_ratio =
        mean(&aggregate.candidate.runtimes) / mean(&aggregate.accepted.runtimes).max(1e-9);
    let failure_rate_increase = (1.0 - candidate_validity) - (1.0 - accepted_validity);
    let maximum_regression = aggregate
        .context_gaps
        .iter()
        .map(|(_, gap)| (-gap).max(0.0))
        .fold(0.0, f64::max);
    let conditional_lift = mean(
        &aggregate
            .context_gaps
            .iter()
            .filter(|(class, _)| *class != CanaryContextClass::Anchor)
            .map(|(_, gap)| *gap)
            .collect::<Vec<_>>(),
    );
    let novelty = aggregate.candidate_novel as f64 / aggregate.candidate.requested.max(1) as f64;
    let stability = aggregate
        .context_gaps
        .iter()
        .filter(|(_, gap)| gap.is_finite())
        .count() as f64
        / aggregate.context_gaps.len().max(1) as f64;
    let efficiency = pairwise_lift / mean(&aggregate.candidate.runtimes).max(1e-9);
    let summary = json!({
        "schema": "ShadowBattleSummary.v1",
        "task": pending.value.task,
        "mode": pending.value.mode,
        "accepted_parent_checkpoint_sha256": pending.value.accepted_checkpoint_sha256,
        "shadow_checkpoint_sha256": pending.value.candidate_checkpoint_sha256,
        "candidate_checkpoint_sha256": pending.value.candidate_checkpoint_sha256,
        "run_index": pending.value.run_index,
        "contexts": pending.value.contexts,
        "shadow_proposals_per_context": 8,
        "accepted_baseline_proposals_per_context": 8,
        "matched_gp_comparators_per_context": 8,
        "candidate_vs_accepted": {
            "validity": candidate_validity,
            "novelty": novelty,
            "win_rate": aggregate.strict_wins as f64 / contexts.len().max(1) as f64,
            "pairwise_lift": pairwise_lift,
            "stability": stability,
            "efficiency": efficiency,
            "latency_ratio": latency_ratio,
            "conditionality": conditional_lift,
        },
        "candidate_vs_gp": {
            "validity": candidate_validity.min(gp_validity),
            "pairwise_lift": gp_lift,
        },
        "accepted_validity": accepted_validity,
        "gp_validity": gp_validity,
        "strict_wins": aggregate.strict_wins,
        "replay_verification_complete": replay_verification_complete,
        "gp_seed_eligible_winners": merit_winners,
        "promotion_eligible": false,
        "created_at_ms": now_ms(),
    });
    if pending.value.mode == CanaryMode::Production {
        let acceptance = TaskCanaryAcceptanceV2 {
            schema: TASK_CANARY_ACCEPTANCE_V2_SCHEMA.to_owned(),
            task: pending.value.task,
            accepted_parent_checkpoint_sha256: pending.value.accepted_checkpoint_sha256.clone(),
            candidate_checkpoint_sha256: pending.value.candidate_checkpoint_sha256.clone(),
            run_index: pending.value.run_index,
            metrics: TaskCanaryMetricsV2 {
                task_pairwise_lift: pairwise_lift,
                maximum_task_regression: maximum_regression,
                validity: candidate_validity,
                strict_wins: aggregate.strict_wins,
                equal_budget_win: mean(&aggregate.candidate.scores)
                    > mean(&aggregate.accepted.scores),
                failure_rate_increase,
                latency_ratio,
                replay_verification_complete,
                conditional_advantage_lift: conditional_lift,
            },
            accepted: false,
            rejection_reasons: Vec::new(),
        }
        .evaluate()?;
        let acceptance_path = args
            .root
            .join("training/task-canaries/accepted")
            .join(format!(
                "{}-{}-run{}.v2.json",
                pending.value.task.as_str(),
                pending.value.candidate_checkpoint_sha256,
                pending.value.run_index
            ));
        atomic_json(&acceptance_path, &serde_json::to_value(acceptance)?)?;
    }
    Ok(summary)
}

fn consume_intent(args: &Args, pending: &PendingIntent) -> Result<Value> {
    if !production_predecessor_passed(&args.root, &pending.value)? {
        return Ok(json!({
            "action": "waiting_for_prior_production_canary",
            "task": pending.value.task,
            "candidate": pending.value.candidate_checkpoint_sha256,
            "run_index": pending.value.run_index,
        }));
    }
    let intent_hash = sha256_file(&pending.path)?;
    let work = args
        .root
        .join("training/canary-executor/work")
        .join(&intent_hash);
    fs::create_dir_all(&work)?;
    let jobs = scq_jobs(args)?;
    let Some(proposal_set) = ensure_proposals(args, pending, &work, &jobs)? else {
        return Ok(json!({
            "action": "proposal_job_pending",
            "intent_sha256": intent_hash,
        }));
    };
    let evaluations = build_evaluations(args, pending, &work, &proposal_set)?;
    let jobs = scq_jobs(args)?;
    let Some(receipts) = ensure_evaluations(args, &work, &evaluations, &jobs)? else {
        return Ok(json!({
            "action": "evaluation_jobs_pending",
            "intent_sha256": intent_hash,
            "evaluation_jobs": evaluations.len(),
        }));
    };
    let summary = summarize(args, pending, &proposal_set, &receipts)?;
    let completed_root = match pending.value.mode {
        CanaryMode::Shadow => args.root.join("training/shadow-canaries"),
        CanaryMode::Production => args.root.join("training/task-canaries"),
    };
    let summary_path = completed_root
        .join("completed")
        .join(format!("{intent_hash}.summary.v1.json"));
    atomic_json(&summary_path, &summary)?;
    let intent_destination = completed_root.join("intents").join(
        pending
            .path
            .file_name()
            .context("canary intent has no filename")?,
    );
    fs::create_dir_all(
        intent_destination
            .parent()
            .context("intent destination has no parent")?,
    )?;
    fs::rename(&pending.path, &intent_destination)?;
    Ok(json!({
        "action": "completed",
        "intent_sha256": intent_hash,
        "summary": summary_path,
        "consumed_intent": intent_destination,
    }))
}

fn cycle(args: &Args) -> Result<Value> {
    let pending = pending_intents(&args.root)?;
    for intent in &pending {
        if intent.value.mode == CanaryMode::Production
            && !production_predecessor_passed(&args.root, &intent.value)?
        {
            continue;
        }
        let result = consume_intent(args, intent)?;
        return Ok(json!({
            "schema": "CanaryExecutorStatus.v2",
            "generated_at_ms": now_ms(),
            "pending_intents": pending.len(),
            "result": result,
        }));
    }
    Ok(json!({
        "schema": "CanaryExecutorStatus.v2",
        "generated_at_ms": now_ms(),
        "pending_intents": pending.len(),
        "result": {"action": "idle"},
    }))
}

fn transient_capacity_error(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && ((message.contains("GPU capacity") && message.contains("smoke test exited with exit 6"))
            || message.contains("no eligible launch-ready node currently"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut previous_error = String::new();
    let mut identical_failures = 0_u32;
    loop {
        let status = match cycle(&args) {
            Ok(status) => {
                previous_error.clear();
                identical_failures = 0;
                status
            }
            Err(error) => {
                let message = format!("{error:#}");
                let capacity_deferred = transient_capacity_error(&message);
                if capacity_deferred {
                    previous_error.clear();
                    identical_failures = 0;
                } else if message == previous_error {
                    identical_failures = identical_failures.saturating_add(1);
                } else {
                    previous_error = message.clone();
                    identical_failures = 1;
                }
                json!({
                    "schema": "CanaryExecutorStatus.v2",
                    "generated_at_ms": now_ms(),
                    "result": {
                        "action": if capacity_deferred { "deferred_capacity" } else { "error" },
                        "error": message,
                        "error_class": if capacity_deferred { "transient_capacity" } else { "permanent_or_unknown" },
                        "identical_failures": identical_failures,
                        "critical": !capacity_deferred && identical_failures >= 3,
                    },
                })
            }
        };
        atomic_json(&args.root.join("ops/canary-executor.v2.json"), &status)?;
        println!("{}", serde_json::to_string(&status)?);
        if args.once {
            return Ok(());
        }
        let backoff = if identical_failures == 0 {
            args.interval_seconds
        } else {
            args.interval_seconds
                .saturating_mul(1_u64 << identical_failures.saturating_sub(1).min(4))
                .min(120)
        };
        thread::sleep(Duration::from_secs(backoff));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn v2_intent_accepts_creation_timestamp_emitted_by_controller() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("intent.v2.json");
        let contexts = (0..16)
            .map(|index| {
                json!({
                    "context_identity": format!("context-{index}"),
                    "class": if index < 12 {
                        "frontier"
                    } else if index < 14 {
                        "hard"
                    } else {
                        "anchor"
                    },
                })
            })
            .collect::<Vec<_>>();
        atomic_json(
            &path,
            &json!({
                "schema": "CanaryIntent.v2",
                "mode": "shadow",
                "task": "regression",
                "candidate_checkpoint_sha256": "a".repeat(64),
                "accepted_parent_checkpoint_sha256": "b".repeat(64),
                "run_index": 0,
                "seed": 1,
                "contexts": contexts,
                "candidate_proposals_per_context": 8,
                "accepted_baseline_proposals_per_context": 8,
                "zero_dl_gp_proposals_per_context": 8,
                "created_at_ms": 123,
            }),
        )
        .unwrap();

        let intent = normalized_intent(&path).unwrap();
        assert_eq!(intent.created_at_ms, Some(123));
    }

    #[test]
    fn lane_validity_counts_missing_requested_slots() {
        let requested = (0..8)
            .map(|index| Value::String(format!("{index:064x}")))
            .collect::<Vec<_>>();
        let scores = requested[..7]
            .iter()
            .map(|value| (value.as_str().unwrap().to_owned(), (1.0, 1.0)))
            .collect::<BTreeMap<_, _>>();
        let metrics = lane_metrics(&requested, &scores);
        assert_eq!(metrics.requested, 8);
        assert_eq!(metrics.valid, 7);
    }

    #[test]
    fn gpu_capacity_contention_does_not_trip_permanent_failure_counter() {
        assert!(transient_capacity_error(
            "SCQ check failed: error[dependency-unsatisfied]: dependencies unsatisfied \
             on all eligible nodes; smoke xbabe1 canary proposal GPU capacity: \
             smoke test exited with exit 6"
        ));
        assert!(transient_capacity_error(
            "SCQ check failed: error[dependency-unsatisfied]: dependencies cannot be \
             verified; no eligible launch-ready node currently has the requested resources"
        ));
        assert!(!transient_capacity_error(
            "SCQ check failed: error[dependency-unsatisfied]: python is absent"
        ));
    }

    #[test]
    fn proposal_job_identity_changes_with_implementation_content() {
        let intent = "a".repeat(64);
        let first = proposal_job_id(&intent, &"b".repeat(64));
        let second = proposal_job_id(&intent, &"c".repeat(64));
        assert_ne!(first, second);
        assert_eq!(first, "jope-canary-propose-aaaaaaaaaaaa-bbbbbbbb");
    }

    #[test]
    fn shadow_candidate_ancestry_is_not_accepted_ancestry() {
        assert_eq!(
            canary_lane_ancestry(CanaryMode::Shadow, "candidate"),
            "shadow_dl"
        );
        assert_eq!(
            canary_lane_ancestry(CanaryMode::Production, "candidate"),
            "accepted_dl"
        );
        assert_eq!(canary_lane_ancestry(CanaryMode::Shadow, "gp"), "none");
    }
}
