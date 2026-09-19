//! Execute exact, receipt-level causal pair repairs without imputing outcomes.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_ingest::ingest_receipt_with_manifest;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, default_scq_str, now_ms, read_json, sha256_file,
};
use jope_dime_regressor::selfplay_shadow::{CausalLane, PairRepairRequestV1, Sha256Digest, Task};
use jope_dime_regressor::spec_policy::EVAL;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

const QUEUE_SCHEMA: &str = "CausalPairRepairQueue.v1";
const STATE_SCHEMA: &str = "CausalPairRepairExecutorState.v1";
const REPAIR_SCHEMA: &str = "CausalPairRepairManifest.v1";
const ACTIVE_STATES: &[&str] = &["queued", "starting", "dispatching", "running", "held"];

#[derive(Parser, Debug)]
#[command(
    name = "jope-causal-pair-repair",
    about = "Rerun only missing or failed causal pair members"
)]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    #[arg(long)]
    runner: Option<PathBuf>,
    #[arg(long)]
    once: bool,
    #[arg(long)]
    verify_only: bool,
    #[arg(long, default_value_t = 60)]
    interval_seconds: u64,
    #[arg(long, default_value_t = 40)]
    max_inflight: usize,
    #[arg(long, default_value_t = 20)]
    submissions_per_cycle: usize,
    #[arg(long, default_value_t = 3)]
    max_attempts: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RepairQueue {
    schema: String,
    #[allow(dead_code)]
    generated_at_ms: u128,
    repairs: Vec<PairRepairRequestV1>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RepairEntry {
    pair_id: Sha256Digest,
    task: Task,
    missing_lane: CausalLane,
    status: String,
    attempts: usize,
    identical_failures: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    retry_after_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    spec_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    repair_manifest_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    original_manifest_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    original_manifest_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    original_evaluator_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    candidate_sha256: Option<Sha256Digest>,
    updated_at_ms: u128,
}

impl RepairEntry {
    fn new(request: &PairRepairRequestV1) -> Self {
        Self {
            pair_id: request.pair_id.clone(),
            task: request.task,
            missing_lane: request.missing_lane,
            status: if request.permanent_failure {
                "permanent_failure"
            } else {
                "pending"
            }
            .to_owned(),
            attempts: 0,
            identical_failures: 0,
            last_error: None,
            retry_after_ms: 0,
            job_id: None,
            spec_path: None,
            repair_manifest_path: None,
            original_manifest_path: None,
            original_manifest_sha256: None,
            original_evaluator_sha256: None,
            candidate_sha256: None,
            updated_at_ms: now_ms(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RepairState {
    schema: String,
    updated_at_ms: u128,
    entries: BTreeMap<String, RepairEntry>,
}

impl Default for RepairState {
    fn default() -> Self {
        Self {
            schema: STATE_SCHEMA.to_owned(),
            updated_at_ms: now_ms(),
            entries: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct PairIdentity {
    task: Task,
    context_identity: String,
    accepted_parent_checkpoint_sha256: String,
    wave: u64,
    pair_index: usize,
    receipt_sha256s: Vec<String>,
}

#[derive(Clone, Debug)]
struct TargetCandidate {
    manifest_path: PathBuf,
    manifest_sha256: String,
    candidate_sha256: String,
    candidate: Value,
    manifest: Value,
}

#[derive(Clone, Debug)]
struct EvaluatorBinding {
    path: PathBuf,
    sha256: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.max_inflight > 0, "max inflight must be positive");
    ensure!(
        args.submissions_per_cycle > 0,
        "submissions per cycle must be positive"
    );
    ensure!(args.max_attempts > 0, "max attempts must be positive");
    ensure!(
        !args.verify_only || args.once,
        "--verify-only requires --once"
    );
    loop {
        let audit = cycle(&args)?;
        println!("{}", serde_json::to_string(&audit)?);
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.interval_seconds.min(120)));
    }
}

fn lane_name(lane: CausalLane) -> &'static str {
    match lane {
        CausalLane::NativeDlDirect => "native_dl_direct",
        CausalLane::MatchedGpComparator => "matched_gp_comparator",
    }
}

fn entry_key(pair_id: &Sha256Digest, lane: CausalLane) -> String {
    format!("{}:{}", pair_id, lane_name(lane))
}

fn load_queue(root: &Path) -> Result<RepairQueue> {
    let queue: RepairQueue = serde_json::from_value(read_json(
        &root.join("ops/causal-pair-repair-queue.v1.json"),
    )?)?;
    ensure!(
        queue.schema == QUEUE_SCHEMA,
        "pair repair queue schema drift"
    );
    Ok(queue)
}

fn load_state(root: &Path) -> Result<RepairState> {
    let path = root.join("ops/causal-pair-repair-state.v1.json");
    if !path.is_file() {
        return Ok(RepairState::default());
    }
    let state: RepairState = serde_json::from_value(read_json(&path)?)?;
    ensure!(
        state.schema == STATE_SCHEMA,
        "pair repair state schema drift"
    );
    Ok(state)
}

fn save_state(root: &Path, state: &mut RepairState) -> Result<()> {
    state.updated_at_ms = now_ms();
    atomic_json(
        &root.join("ops/causal-pair-repair-state.v1.json"),
        &serde_json::to_value(state)?,
    )
}

fn scq_jobs(args: &Args) -> Result<BTreeMap<String, String>> {
    let output = Command::new(&args.scq).args(["--json", "ls"]).output()?;
    ensure!(
        output.status.success(),
        "SCQ list failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
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
    let spec = spec.to_string_lossy();
    let check = Command::new(&args.scq)
        .args(["check", spec.as_ref()])
        .output()?;
    ensure!(
        check.status.success(),
        "SCQ check failed: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    let submit = Command::new(&args.scq)
        .args(["submit", spec.as_ref()])
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

fn pair_identity(root: &Path, request: &PairRepairRequestV1) -> Result<PairIdentity> {
    let connection = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let mut statement = connection.prepare(
        "SELECT task,context_identity,accepted_parent_checkpoint_sha256,
                wave,pair_index,receipt_sha256
           FROM causal_evaluation_receipts_v2
          WHERE pair_id=?1 AND canary=0
          ORDER BY created_at_ms,receipt_sha256",
    )?;
    let rows = statement
        .query_map([request.pair_id.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, u64>(3)?,
                row.get::<_, usize>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let first = rows.first().context("repair pair has no raw receipt")?;
    let task: Task = first.0.parse()?;
    ensure!(task == request.task, "repair pair task drift");
    ensure!(
        rows.iter().all(|row| {
            row.0 == first.0
                && row.1 == first.1
                && row.2 == first.2
                && row.3 == first.3
                && row.4 == first.4
        }),
        "repair pair receipt identity is ambiguous"
    );
    Ok(PairIdentity {
        task,
        context_identity: first.1.clone(),
        accepted_parent_checkpoint_sha256: first.2.clone(),
        wave: first.3,
        pair_index: first.4,
        receipt_sha256s: rows.iter().map(|row| row.5.clone()).collect(),
    })
}

fn target_candidate(
    root: &Path,
    request: &PairRepairRequestV1,
    identity: &PairIdentity,
) -> Result<TargetCandidate> {
    let directory = root
        .join("waves")
        .join(format!("strict-wave-{:06}", identity.wave))
        .join("manifests");
    ensure!(
        directory.is_dir(),
        "repair wave manifest directory is absent"
    );
    let mut matches = Vec::new();
    for entry in fs::read_dir(&directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json")
            || !path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.contains(&identity.context_identity))
        {
            continue;
        }
        let manifest = read_json(&path)?;
        if manifest.get("task").and_then(Value::as_str) != Some(identity.task.as_str())
            || manifest.get("dataset_hash").and_then(Value::as_str)
                != Some(identity.context_identity.as_str())
            || manifest
                .pointer("/label_authority/model_checkpoint_sha256")
                .and_then(Value::as_str)
                != Some(identity.accepted_parent_checkpoint_sha256.as_str())
        {
            continue;
        }
        for candidate in manifest
            .get("candidates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let evidence = candidate.get("evidence").unwrap_or(&Value::Null);
            if evidence.get("evaluation_lane").and_then(Value::as_str)
                == Some(lane_name(request.missing_lane))
                && evidence.get("matched_pair_index").and_then(Value::as_u64)
                    == Some(identity.pair_index as u64)
            {
                if let Some(pair_id) = evidence.get("pair_id").and_then(Value::as_str) {
                    ensure!(
                        pair_id == request.pair_id.as_str(),
                        "target manifest carries a conflicting pair id"
                    );
                }
                matches.push(TargetCandidate {
                    manifest_path: path.clone(),
                    manifest_sha256: sha256_file(&path)?,
                    candidate_sha256: candidate
                        .get("semantic_sha256")
                        .and_then(Value::as_str)
                        .context("repair candidate lacks semantic hash")?
                        .to_owned(),
                    candidate: candidate.clone(),
                    manifest: manifest.clone(),
                });
            }
        }
    }
    ensure!(
        matches.len() == 1,
        "repair target member count is {}, expected exactly one",
        matches.len()
    );
    Ok(matches.remove(0))
}

fn parse_evaluator_input(spec: &Path) -> Result<(PathBuf, String)> {
    let text = fs::read_to_string(spec)
        .with_context(|| format!("read original evaluator spec {}", spec.display()))?;
    let mut in_input = false;
    let mut name = None;
    let mut source = None;
    let mut sha256 = None;
    for line in text.lines().map(str::trim) {
        if line == "[[input]]" {
            if name.as_deref() == Some("evaluator") {
                break;
            }
            in_input = true;
            name = None;
            source = None;
            sha256 = None;
            continue;
        }
        if line.starts_with('[') {
            if name.as_deref() == Some("evaluator") {
                break;
            }
            in_input = false;
            continue;
        }
        if !in_input {
            continue;
        }
        let Some((key, raw)) = line.split_once('=') else {
            continue;
        };
        let value = || -> Result<String> {
            serde_json::from_str(raw.trim()).context("generated TOML string is not JSON-quoted")
        };
        match key.trim() {
            "name" => name = Some(value()?),
            "source" => source = Some(PathBuf::from(value()?)),
            "sha256" => sha256 = Some(value()?),
            _ => {}
        }
    }
    ensure!(
        name.as_deref() == Some("evaluator"),
        "spec lacks evaluator input"
    );
    Ok((
        source.context("evaluator input lacks source")?,
        sha256.context("evaluator input lacks sha256")?,
    ))
}

fn evaluator_from_spec(
    root: &Path,
    spec: &Path,
    bundle_evaluators: &BTreeMap<String, PathBuf>,
) -> Result<EvaluatorBinding> {
    let (source, expected) = parse_evaluator_input(spec)?;
    if source.is_file() && sha256_file(&source)? == expected {
        return Ok(EvaluatorBinding {
            path: source,
            sha256: expected,
        });
    }
    let path = bundle_evaluators.get(&expected).cloned().with_context(|| {
        format!(
            "original evaluator {expected} is not retained under {}",
            root.join("runtime-bundles").display()
        )
    })?;
    Ok(EvaluatorBinding {
        path,
        sha256: expected,
    })
}

fn evaluator_binding(
    root: &Path,
    identity: &PairIdentity,
    target: &TargetCandidate,
    bundle_evaluators: &BTreeMap<String, PathBuf>,
) -> Result<EvaluatorBinding> {
    let backlog = Connection::open_with_flags(
        root.join("ops/backlog.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    backlog.busy_timeout(Duration::from_secs(60))?;
    let target_spec = backlog
        .query_row(
            "SELECT spec FROM claims WHERE manifest_sha256=?1
             ORDER BY attempts DESC,updated_at_ms DESC LIMIT 1",
            [&target.manifest_sha256],
            |row| row.get::<_, String>(0),
        )
        .ok();
    if let Some(spec) = target_spec {
        return evaluator_from_spec(root, Path::new(&spec), bundle_evaluators);
    }

    let corpus = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    corpus.busy_timeout(Duration::from_secs(60))?;
    for receipt_sha256 in &identity.receipt_sha256s {
        let jobs = {
            let mut statement = corpus.prepare(
                "SELECT job_id FROM ingested_jobs WHERE receipt_sha256=?1 ORDER BY job_id",
            )?;
            let jobs = statement
                .query_map([receipt_sha256], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            jobs
        };
        for job_id in jobs {
            let spec = backlog
                .query_row(
                    "SELECT spec FROM claims WHERE job_id=?1 LIMIT 1",
                    [&job_id],
                    |row| row.get::<_, String>(0),
                )
                .ok();
            if let Some(spec) = spec {
                return evaluator_from_spec(root, Path::new(&spec), bundle_evaluators);
            }
        }
    }
    bail!("no immutable original evaluator binding exists for the repair pair")
}

/// Content-addressed evaluators available for replay, keyed by sha256.
///
/// Bundles are the primary source, but a bundle directory can be gutted by
/// cleanup while the evaluator it carried is still referenced by historical
/// repair state -- which is exactly the live failure: evaluator 6bf93239 is
/// referenced by 40 pairs, survives nowhere under `runtime-bundles`, and has
/// accumulated 120,349 identical retry failures because this lookup could not
/// see it. `retain_evaluators` already hard-links every bundle evaluator into
/// `training/evaluator-vault/<digest>/corpus_eval` for precisely this reason,
/// and `jope_causal_evidence::retained_evaluator` already resolves against the
/// vault. This function was the odd one out.
///
/// The vault is content-addressed, so admitting it invents no provenance: the
/// digest IS the identity, and it is re-verified here rather than trusted from
/// the path. A bundle entry always wins over a vault entry for the same digest.
fn bundle_evaluators(root: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let mut output = BTreeMap::new();
    let bundles = root.join("runtime-bundles");
    if bundles.is_dir() {
        for entry in fs::read_dir(bundles)? {
            let evaluator = entry?.path().join("runtime/bin/corpus_eval");
            if evaluator.is_file() {
                output.entry(sha256_file(&evaluator)?).or_insert(evaluator);
            }
        }
    }
    let vault = root.join("training/evaluator-vault");
    if vault.is_dir() {
        for entry in fs::read_dir(vault)? {
            let dir = entry?.path();
            let evaluator = dir.join("corpus_eval");
            if !evaluator.is_file() {
                continue;
            }
            let digest = sha256_file(&evaluator)?;
            // The directory name is a claim; the bytes are the proof. Skip any
            // entry whose contents do not hash to the name it is filed under.
            if dir.file_name().and_then(|n| n.to_str()) != Some(digest.as_str()) {
                continue;
            }
            output.entry(digest).or_insert(evaluator);
        }
    }
    Ok(output)
}

fn runtime_runner(args: &Args) -> Result<PathBuf> {
    if let Some(runner) = &args.runner {
        ensure!(runner.is_file(), "configured repair runner is absent");
        return Ok(runner.clone());
    }
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    let bundle = executable
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .context("repair executor is not running from a runtime bundle")?;
    let runner = bundle.join("runtime/scripts/fweval_runner.sh");
    ensure!(runner.is_file(), "runtime-bundle repair runner is absent");
    Ok(runner)
}

fn write_immutable(path: &Path, body: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(
            fs::read(path)? == body,
            "immutable pair repair artifact drift at {}",
            path.display()
        );
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

fn repair_manifest(
    root: &Path,
    request: &PairRepairRequestV1,
    target: &TargetCandidate,
    evaluator: &EvaluatorBinding,
) -> Result<PathBuf> {
    let directory = root.join("ops/causal-pair-repairs/manifests");
    let path = directory.join(format!(
        "{}-{}.json",
        request.pair_id,
        lane_name(request.missing_lane)
    ));
    let mut manifest = target.manifest.clone();
    manifest["evaluation_lane"] = json!("causal_pair_repair");
    manifest["candidates"] = json!([target.candidate]);
    manifest["causal_pair_repair"] = json!({
        "schema": REPAIR_SCHEMA,
        "pair_id": request.pair_id,
        "missing_lane": request.missing_lane,
        "original_manifest_path": target.manifest_path,
        "original_manifest_sha256": target.manifest_sha256,
        "original_evaluator_sha256": evaluator.sha256,
    });
    write_immutable(&path, &serde_json::to_vec(&manifest)?)?;
    Ok(path)
}

fn toml_quote(value: &str) -> Result<String> {
    ensure!(
        !value.contains(['"', '\\', '\n', '\r']),
        "unsafe generated TOML value"
    );
    Ok(format!("\"{value}\""))
}

fn append_input(
    lines: &mut Vec<String>,
    name: &str,
    source: &Path,
    target: &str,
    expected_sha256: Option<&str>,
) -> Result<()> {
    let actual = sha256_file(source)?;
    if let Some(expected) = expected_sha256 {
        ensure!(actual == expected, "{name} content pin drift");
    }
    lines.extend([
        "[[input]]".to_owned(),
        format!("name = {}", toml_quote(name)?),
        format!("source = {}", toml_quote(&source.to_string_lossy())?),
        format!("target = {}", toml_quote(target)?),
        format!("sha256 = {}", toml_quote(&actual)?),
        String::new(),
    ]);
    Ok(())
}

fn render_spec(
    args: &Args,
    runner: &Path,
    request: &PairRepairRequestV1,
    target: &TargetCandidate,
    evaluator: &EvaluatorBinding,
    manifest: &Path,
    attempt: usize,
) -> Result<(String, PathBuf)> {
    let job_id = format!(
        "jte-pair-repair-{}-{}-a{attempt}",
        &request.pair_id.as_str()[..20],
        match request.missing_lane {
            CausalLane::NativeDlDirect => "n",
            CausalLane::MatchedGpComparator => "g",
        }
    );
    let path = args
        .root
        .join("ops/causal-pair-repairs/specs")
        .join(format!("{job_id}.toml"));
    let train = PathBuf::from(
        target
            .manifest
            .get("train_csv")
            .and_then(Value::as_str)
            .context("repair manifest lacks train CSV")?,
    );
    let test = PathBuf::from(
        target
            .manifest
            .get("test_csv")
            .and_then(Value::as_str)
            .context("repair manifest lacks test CSV")?,
    );
    let train_sha = target
        .manifest
        .get("train_sha256")
        .and_then(Value::as_str)
        .context("repair manifest lacks train hash")?;
    let test_sha = target
        .manifest
        .get("test_sha256")
        .and_then(Value::as_str)
        .context("repair manifest lacks test hash")?;
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = {}", toml_quote(&job_id)?),
        "project = \"jope-selfplay-eval\"".to_owned(),
        "tier = \"normal\"".to_owned(),
        "command = \"/usr/bin/env\"".to_owned(),
        "args = [\"{input:runner}\", \"{input:manifest}\", \"{input:train}\", \"{input:test}\", \"{input:evaluator}\", \"{output:receipt}\"]".to_owned(),
        String::new(),
    ];
    lines.extend(EVAL.render_resource_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend(EVAL.render_placement_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend([
        "[env]".to_owned(),
        format!("RAYON_NUM_THREADS = \"{}\"", EVAL.omp_threads),
        format!("OMP_NUM_THREADS = \"{}\"", EVAL.omp_threads),
        format!("OPENBLAS_NUM_THREADS = \"{}\"", EVAL.omp_threads),
        format!("MKL_NUM_THREADS = \"{}\"", EVAL.omp_threads),
        format!("NUMEXPR_NUM_THREADS = \"{}\"", EVAL.omp_threads),
        String::new(),
        "[preempt]".to_owned(),
        format!("preemptible = {}", EVAL.preemptible),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
    ]);
    lines.extend(EVAL.render_limits_block().lines().map(str::to_owned));
    lines.push(String::new());
    append_input(&mut lines, "runner", runner, "runner.sh", None)?;
    append_input(&mut lines, "manifest", manifest, "manifest.json", None)?;
    append_input(&mut lines, "train", &train, "train.csv", Some(train_sha))?;
    append_input(&mut lines, "test", &test, "test.csv", Some(test_sha))?;
    append_input(
        &mut lines,
        "evaluator",
        &evaluator.path,
        "corpus_eval",
        Some(&evaluator.sha256),
    )?;
    lines.extend([
        "[[output]]".to_owned(),
        "name = \"receipt\"".to_owned(),
        "source = \"receipt\"".to_owned(),
        "directory = true".to_owned(),
        String::new(),
    ]);
    write_immutable(&path, lines.join("\n").as_bytes())?;
    Ok((job_id, path))
}

fn repair_succeeded(root: &Path, entry: &RepairEntry, receipt_sha256: &str) -> Result<bool> {
    let connection = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let candidate = entry
        .candidate_sha256
        .as_ref()
        .context("repair state lacks candidate identity")?;
    let success: bool = connection.query_row(
        "SELECT EXISTS(
           SELECT 1 FROM causal_evaluation_receipts_v2
            WHERE pair_id=?1 AND lane=?2 AND candidate_sha256=?3
              AND receipt_sha256=?4 AND success=1 AND proposal_valid=1
         )",
        (
            entry.pair_id.as_str(),
            lane_name(entry.missing_lane),
            candidate.as_str(),
            receipt_sha256,
        ),
        |row| row.get(0),
    )?;
    Ok(success)
}

fn ingest_completed(args: &Args, entry: &RepairEntry) -> Result<bool> {
    let job_id = entry
        .job_id
        .as_deref()
        .context("repair state lacks job id")?;
    let manifest = entry
        .repair_manifest_path
        .as_ref()
        .context("repair state lacks manifest path")?;
    let output = args
        .root
        .join("ops/causal-pair-repairs/receipts")
        .join(job_id);
    let mut markers = Vec::new();
    recursive_find(&output, "SHARD_OK", &mut markers);
    if markers.is_empty() {
        fetch(args, job_id, &output)?;
        recursive_find(&output, "SHARD_OK", &mut markers);
    }
    ensure!(markers.len() == 1, "repair fetch has no unique SHARD_OK");
    let receipt_root = markers[0]
        .parent()
        .context("repair SHARD_OK has no parent")?;
    let dataset = read_json(manifest)?
        .get("dataset_hash")
        .and_then(Value::as_str)
        .context("repair manifest lacks dataset")?
        .to_owned();
    let receipt_path = receipt_root.join(format!("{dataset}.receipts.jsonl"));
    let receipt_sha256 = sha256_file(&receipt_path)?;
    ingest_receipt_with_manifest(
        &args.root.join("corpus/strict-test-exact.sqlite"),
        job_id,
        receipt_root,
        manifest,
    )?;
    repair_succeeded(&args.root, entry, &receipt_sha256)
}

fn record_failure(entry: &mut RepairEntry, message: String, max_attempts: usize) {
    entry.identical_failures = if entry.last_error.as_deref() == Some(message.as_str()) {
        entry.identical_failures.saturating_add(1)
    } else {
        1
    };
    entry.last_error = Some(message);
    entry.job_id = None;
    entry.spec_path = None;
    entry.status = if entry.attempts >= max_attempts {
        "permanent_failure"
    } else {
        "pending"
    }
    .to_owned();
    let exponent = entry.identical_failures.saturating_sub(1).min(7) as u32;
    let delay_seconds = 2_u128.pow(exponent).min(120);
    entry.retry_after_ms = now_ms().saturating_add(delay_seconds * 1_000);
    entry.updated_at_ms = now_ms();
}

fn note_control_error(entry: &mut RepairEntry, error: anyhow::Error) {
    let message = format!("{error:#}");
    entry.identical_failures = if entry.last_error.as_deref() == Some(message.as_str()) {
        entry.identical_failures.saturating_add(1)
    } else {
        1
    };
    entry.last_error = Some(message);
    entry.status = "blocked".to_owned();
    let exponent = entry.identical_failures.saturating_sub(1).min(7) as u32;
    let delay_seconds = 2_u128.pow(exponent).min(120);
    entry.retry_after_ms = now_ms().saturating_add(delay_seconds * 1_000);
    entry.updated_at_ms = now_ms();
}

fn complete_jobs(
    args: &Args,
    state: &mut RepairState,
    jobs: &BTreeMap<String, String>,
) -> Result<()> {
    let keys = state
        .entries
        .iter()
        .filter(|(_, entry)| entry.status == "submitted")
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    for key in keys {
        let entry = state.entries.get_mut(&key).expect("entry key exists");
        let job_id = entry
            .job_id
            .clone()
            .context("submitted repair lacks job id")?;
        match jobs.get(&job_id).map(String::as_str) {
            Some("completed") => match ingest_completed(args, entry) {
                Ok(true) => {
                    entry.status = "completed".to_owned();
                    entry.last_error = None;
                    entry.identical_failures = 0;
                    entry.retry_after_ms = 0;
                    entry.updated_at_ms = now_ms();
                }
                Ok(false) => record_failure(
                    entry,
                    "evaluator returned an unsuccessful exact member".to_owned(),
                    args.max_attempts,
                ),
                Err(error) => note_control_error(entry, error),
            },
            Some(state) if ACTIVE_STATES.contains(&state) => {}
            Some(state @ ("failed" | "cancelled")) => record_failure(
                entry,
                format!("SCQ terminal state {state}"),
                args.max_attempts,
            ),
            Some(state_name) => note_control_error(
                entry,
                anyhow::anyhow!("SCQ returned unknown repair state {state_name}"),
            ),
            None => {
                let spec = entry
                    .spec_path
                    .as_ref()
                    .context("submitted repair lacks spec")?;
                if let Err(error) = submit_spec(args, spec) {
                    note_control_error(entry, error);
                }
            }
        }
    }
    Ok(())
}

fn task_priority(root: &Path) -> Task {
    let ledger =
        read_json(&root.join("training/causal-evidence-ledger.v2.json")).unwrap_or(Value::Null);
    let score = |task: Task| {
        let row = &ledger["tasks"][task.as_str()];
        let pairs = row
            .get("complete_pairs")
            .and_then(Value::as_u64)
            .unwrap_or_default() as f64;
        let validity = row
            .get("proposal_validity")
            .and_then(Value::as_f64)
            .unwrap_or_default();
        (pairs / 2_048.0).min(1.0) + validity
    };
    if score(Task::Regression) > score(Task::Binary) {
        Task::Regression
    } else {
        Task::Binary
    }
}

fn allocate_requests(
    requests: &[PairRepairRequestV1],
    limit: usize,
    priority: Task,
) -> Vec<PairRepairRequestV1> {
    let mut by_task = BTreeMap::<Task, Vec<PairRepairRequestV1>>::new();
    for request in requests {
        by_task
            .entry(request.task)
            .or_default()
            .push(request.clone());
    }
    for rows in by_task.values_mut() {
        rows.sort_by(|left, right| left.pair_id.cmp(&right.pair_id));
    }
    let priority_quota = (limit * 3).div_ceil(5);
    // The repair lane still splits its budget across the two legacy tasks;
    // `task_priority` can only return Regression or Binary until new-task
    // repair activates with its own allocation policy.
    let other = if priority == Task::Binary {
        Task::Regression
    } else {
        Task::Binary
    };
    let mut output = Vec::new();
    for (task, quota) in [(priority, priority_quota), (other, limit - priority_quota)] {
        output.extend(
            by_task
                .get_mut(&task)
                .into_iter()
                .flat_map(|rows| rows.drain(..rows.len().min(quota))),
        );
    }
    if output.len() < limit {
        for task in [priority, other] {
            let remaining = limit - output.len();
            output.extend(
                by_task
                    .get_mut(&task)
                    .into_iter()
                    .flat_map(|rows| rows.drain(..rows.len().min(remaining))),
            );
        }
    }
    output
}

fn prepare_and_submit(
    args: &Args,
    request: &PairRepairRequestV1,
    entry: &mut RepairEntry,
    runner: &Path,
    evaluators: &BTreeMap<String, PathBuf>,
    jobs: &BTreeMap<String, String>,
) -> Result<()> {
    let identity = pair_identity(&args.root, request)?;
    let target = target_candidate(&args.root, request, &identity)?;
    let evaluator = evaluator_binding(&args.root, &identity, &target, evaluators)?;
    let manifest = repair_manifest(&args.root, request, &target, &evaluator)?;
    let attempt = entry.attempts.saturating_add(1);
    let (job_id, spec) = render_spec(
        args, runner, request, &target, &evaluator, &manifest, attempt,
    )?;
    match jobs.get(&job_id).map(String::as_str) {
        None => submit_spec(args, &spec)?,
        Some(_) => {}
    }
    entry.status = "submitted".to_owned();
    entry.attempts = attempt;
    entry.identical_failures = 0;
    entry.last_error = None;
    entry.retry_after_ms = 0;
    entry.job_id = Some(job_id);
    entry.spec_path = Some(spec);
    entry.repair_manifest_path = Some(manifest);
    entry.original_manifest_path = Some(target.manifest_path);
    entry.original_manifest_sha256 = Some(Sha256Digest::parse(target.manifest_sha256.clone())?);
    entry.original_evaluator_sha256 = Some(Sha256Digest::parse(evaluator.sha256)?);
    entry.candidate_sha256 = Some(Sha256Digest::parse(target.candidate_sha256)?);
    entry.updated_at_ms = now_ms();
    Ok(())
}

fn cycle(args: &Args) -> Result<Value> {
    let queue = load_queue(&args.root)?;
    if args.verify_only {
        let evaluators = bundle_evaluators(&args.root)?;
        let selected = allocate_requests(
            &queue
                .repairs
                .iter()
                .filter(|request| !request.permanent_failure)
                .cloned()
                .collect::<Vec<_>>(),
            args.submissions_per_cycle,
            task_priority(&args.root),
        );
        let mut verified = Vec::new();
        let mut failures = Vec::new();
        for request in selected {
            let result = (|| -> Result<Value> {
                let identity = pair_identity(&args.root, &request)?;
                let target = target_candidate(&args.root, &request, &identity)?;
                let evaluator = evaluator_binding(&args.root, &identity, &target, &evaluators)?;
                Ok(json!({
                    "pair_id": request.pair_id,
                    "lane": request.missing_lane,
                    "task": request.task,
                    "target_manifest_sha256": target.manifest_sha256,
                    "candidate_sha256": target.candidate_sha256,
                    "evaluator_sha256": evaluator.sha256,
                }))
            })();
            match result {
                Ok(row) => verified.push(row),
                Err(error) => failures.push(json!({
                    "pair_id": request.pair_id,
                    "lane": request.missing_lane,
                    "task": request.task,
                    "error": format!("{error:#}"),
                })),
            }
        }
        let audit = json!({
            "schema": "CausalPairRepairExecutor.v1",
            "generated_at_ms": now_ms(),
            "verify_only": true,
            "queue_depth": queue.repairs.len(),
            "verified": verified,
            "failures": failures,
            "permanent_failures_are_imputed": false,
        });
        atomic_json(
            &args
                .root
                .join("ops/causal-pair-repair-verification.v1.json"),
            &audit,
        )?;
        return Ok(audit);
    }
    let mut state = load_state(&args.root)?;
    for request in &queue.repairs {
        let key = entry_key(&request.pair_id, request.missing_lane);
        let entry = state
            .entries
            .entry(key)
            .or_insert_with(|| RepairEntry::new(request));
        if request.permanent_failure && entry.status != "completed" {
            entry.status = "permanent_failure".to_owned();
            entry.last_error = Some("ledger marked the pair member irreparable".to_owned());
            entry.updated_at_ms = now_ms();
        }
    }

    let jobs = scq_jobs(args)?;
    complete_jobs(args, &mut state, &jobs)?;
    let inflight = state
        .entries
        .values()
        .filter(|entry| entry.status == "submitted")
        .count();
    let capacity = args.max_inflight.saturating_sub(inflight);
    let limit = capacity.min(args.submissions_per_cycle);
    let active_queue_keys = queue
        .repairs
        .iter()
        .map(|request| entry_key(&request.pair_id, request.missing_lane))
        .collect::<BTreeSet<_>>();
    let eligible = queue
        .repairs
        .iter()
        .filter(|request| {
            let key = entry_key(&request.pair_id, request.missing_lane);
            state.entries.get(&key).is_some_and(|entry| {
                matches!(entry.status.as_str(), "pending" | "blocked")
                    && entry.attempts < args.max_attempts
                    && entry.retry_after_ms <= now_ms()
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    let selected = allocate_requests(&eligible, limit, task_priority(&args.root));
    let runner = runtime_runner(args)?;
    let evaluators = bundle_evaluators(&args.root)?;
    for request in selected {
        let key = entry_key(&request.pair_id, request.missing_lane);
        let entry = state.entries.get_mut(&key).expect("queue entry exists");
        if let Err(error) = prepare_and_submit(args, &request, entry, &runner, &evaluators, &jobs) {
            note_control_error(entry, error);
        }
    }
    save_state(&args.root, &mut state)?;

    let counts =
        state
            .entries
            .values()
            .fold(BTreeMap::<String, usize>::new(), |mut counts, entry| {
                *counts.entry(entry.status.clone()).or_default() += 1;
                counts
            });
    let critical = state.entries.values().filter(|entry| {
        active_queue_keys.contains(&entry_key(&entry.pair_id, entry.missing_lane))
            && entry.identical_failures >= 3
            && entry.status != "permanent_failure"
    });
    let critical = critical
        .map(|entry| {
            json!({
                "pair_id": entry.pair_id,
                "lane": entry.missing_lane,
                "error": entry.last_error,
            })
        })
        .collect::<Vec<_>>();
    let audit = json!({
        "schema": "CausalPairRepairExecutor.v1",
        "generated_at_ms": now_ms(),
        "queue_depth": queue.repairs.len(),
        "state_counts": counts,
        "inflight": state.entries.values().filter(|entry| entry.status == "submitted").count(),
        "task_priority": task_priority(&args.root),
        "critical_identical_failures": critical,
        "permanent_failures_are_imputed": false,
    });
    atomic_json(
        &args.root.join("ops/causal-pair-repair-executor.v1.json"),
        &audit,
    )?;
    Ok(audit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> Sha256Digest {
        Sha256Digest::parse(byte.to_string().repeat(64)).unwrap()
    }

    fn request(task: Task, index: usize) -> PairRepairRequestV1 {
        PairRepairRequestV1 {
            pair_id: Sha256Digest::of_bytes(format!("pair-{task}-{index}").as_bytes()),
            task,
            missing_lane: CausalLane::NativeDlDirect,
            original_manifest_sha256: digest('f'),
            permanent_failure: false,
        }
    }

    #[test]
    fn deficit_allocation_preserves_the_sixty_forty_bound() {
        let mut requests = Vec::new();
        for index in 0..12 {
            requests.push(request(Task::Regression, index));
        }
        for index in 0..8 {
            requests.push(request(Task::Binary, index));
        }
        let selected = allocate_requests(&requests, 20, Task::Regression);
        assert_eq!(selected.len(), 20);
        assert_eq!(
            selected
                .iter()
                .filter(|row| row.task == Task::Regression)
                .count(),
            12
        );
        assert_eq!(
            selected
                .iter()
                .filter(|row| row.task == Task::Binary)
                .count(),
            8
        );
    }

    #[test]
    fn original_evaluator_input_is_parsed_from_the_immutable_spec() {
        let temporary = tempfile::tempdir().unwrap();
        let spec = temporary.path().join("job.toml");
        fs::write(
            &spec,
            r#"
[[input]]
name = "manifest"
source = "/tmp/manifest"
sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[[input]]
name = "evaluator"
source = "/tmp/corpus_eval"
target = "corpus_eval"
sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

[[output]]
name = "receipt"
"#,
        )
        .unwrap();
        let (path, sha256) = parse_evaluator_input(&spec).unwrap();
        assert_eq!(path, PathBuf::from("/tmp/corpus_eval"));
        assert_eq!(sha256, "b".repeat(64));
    }
}
