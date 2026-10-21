//! Native receipt ingestion, deterministic winner indexes, and status v3.
//!
//! This module deliberately contains no scheduler mutation.  A receipt must
//! already be durable below `receipts/` before it can enter the corpus.

use crate::four_task::{
    bounded_task_utility, operational_loss, Distribution, FourTaskTelemetryRow,
    JopeFourTaskTelemetryV1, RatingTelemetry, SourceBinding, TelemetryCounts, TelemetryEligibility,
    TelemetryMetrics, TrainerValidationTelemetry, TypedCount, TypedDistribution, TypedNumber,
    TypedRating, TypedText, FOUR_TASK_ORDER, FOUR_TASK_TELEMETRY_SCHEMA,
};
use crate::selfplay_shadow::{Sha256Digest, Task};
use crate::task_rating::{JopeTaskRatingSnapshotV1, RatingKindV1};
use anyhow::{bail, ensure, Context, Result};
use rusqlite::types::ValueRef;
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const SERVICES: [&str; 5] = ["queue", "reserve", "ingestion", "controller", "watchdog"];
const SELECTION_BIAS: &str = "test-exact-reused";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestSummary {
    pub rows: usize,
    pub valid: usize,
    pub failed: usize,
    pub already_ingested: bool,
}

#[derive(Debug, Clone)]
struct Authority {
    kind: &'static str,
    task: String,
    metric: String,
    generation: String,
    frozen_seed_sha256: String,
    model_checkpoint_sha256: String,
    hall_of_fame_sha256: String,
    selection_bias: Option<String>,
}

#[derive(Debug)]
struct ValidatedReceipt {
    value: Value,
    candidate_source: String,
    origin_class: String,
    parentage_json: Option<String>,
    predictions_path: Option<String>,
}

#[derive(Debug, Clone)]
struct ExpectedCandidate {
    canonical_genome: String,
    source: String,
    origin_class: String,
    parentage_json: Option<String>,
    evaluation_lane: Option<String>,
    matched_pair_index: Option<u64>,
    pair_id: Option<String>,
    proposal_valid: bool,
    ancestry: String,
    cross_context_backfill: bool,
    valid_lineage: bool,
}

#[derive(Debug)]
struct CandidateDraft {
    canonical_genome: String,
    source: String,
    parents: Vec<String>,
    logical_source: Option<String>,
    evaluation_lane: Option<String>,
    matched_pair_index: Option<u64>,
    pair_id: Option<String>,
    proposal_valid: bool,
    zero_dl_ancestry: bool,
    ancestry: Option<String>,
    cross_context_backfill: bool,
    valid_lineage: bool,
}

#[derive(Debug)]
struct CausalReceiptDraft {
    candidate_sha256: String,
    lane: String,
    pair_index: u64,
    pair_id: String,
    score: Option<f64>,
    success: bool,
    proposal_valid: bool,
    runtime_seconds: f64,
    ancestry: String,
    cross_context_backfill: bool,
    valid_lineage: bool,
    created_at_ms: i64,
}

fn dl_source(source: &str) -> bool {
    matches!(
        source,
        "accepted_dl_checkpoint" | "current_checkpoint_dl_proposal"
    ) || source.contains("dl_proposal")
}

fn strict_wave_number(generation: &str) -> Option<u64> {
    generation.strip_prefix("strict-wave-")?.parse().ok()
}

fn causal_pair_identity(
    parent: &str,
    wave: u64,
    task: &str,
    context_identity: &str,
    pair_index: u64,
) -> String {
    sha256_bytes(
        format!("CausalPair.v1\0{parent}\0{wave}\0{task}\0{context_identity}\0{pair_index}")
            .as_bytes(),
    )
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis() as i64
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut input =
        BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_reader(
        File::open(path).with_context(|| format!("open JSON {}", path.display()))?,
    )
    .with_context(|| format!("parse JSON {}", path.display()))
}

fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("JSON output has no filename")?;
    let temporary = path.with_file_name(format!(".{filename}.tmp-{}", std::process::id()));
    let mut output = File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn atomic_jsonl(path: &Path, rows: &[Value]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("JSONL output has no filename")?;
    let temporary = path.with_file_name(format!(".{filename}.tmp-{}", std::process::id()));
    let mut output = File::create(&temporary)?;
    for row in rows {
        serde_json::to_writer(&mut output, row)?;
        output.write_all(b"\n")?;
    }
    output.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn required_str<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|field| !field.is_empty())
        .with_context(|| format!("missing string field {field}"))
}

fn finite_f64(value: &Value, field: &str) -> Result<f64> {
    let number = value
        .get(field)
        .and_then(Value::as_f64)
        .with_context(|| format!("missing numeric field {field}"))?;
    if !number.is_finite() {
        bail!("{field} is non-finite");
    }
    Ok(number)
}

/// Per-task authority metric (R1/D2); must agree with
/// `crate::task::TaskKind::primary_metric`.
fn expected_authority_metric(task: &str) -> Result<&'static str> {
    Task::parse(task)
        .map(Task::primary_metric)
        .ok_or_else(|| anyhow::anyhow!("unsupported task {task}"))
}

fn authority(value: &Value, task: &str, receipt: bool) -> Result<Authority> {
    let expected_metric = expected_authority_metric(task)?;
    let tier_field = if receipt && value.get("tier").is_some() {
        "tier"
    } else {
        "score_tier"
    };
    if required_str(value, tier_field)? != "test_exact"
        || required_str(value, "metric")? != expected_metric
    {
        bail!("authority task, metric, or tier mismatch");
    }
    let frozen_seed_sha256 = required_str(value, "frozen_seed_sha256")?.to_owned();
    let model_checkpoint_sha256 = required_str(value, "model_checkpoint_sha256")?.to_owned();
    let hall_of_fame_sha256 = required_str(value, "hall_of_fame_sha256")?.to_owned();
    for (name, digest) in [
        ("frozen_seed_sha256", frozen_seed_sha256.as_str()),
        ("model_checkpoint_sha256", model_checkpoint_sha256.as_str()),
        ("hall_of_fame_sha256", hall_of_fame_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            bail!("{name} is not a lowercase SHA-256 value");
        }
    }
    let test_adaptive = value
        .get("test_adaptive")
        .and_then(Value::as_bool)
        .context("authority test_adaptive is absent")?;
    let benchmark_valid = value
        .get("benchmark_valid")
        .and_then(Value::as_bool)
        .context("authority benchmark_valid is absent")?;
    let selection_bias = value
        .get("selection_bias")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_owned);
    let generation = required_str(value, "generation")?.to_owned();
    let kind = if test_adaptive
        && !benchmark_valid
        && selection_bias.as_deref() == Some(SELECTION_BIAS)
        && generation
            .strip_prefix("strict-wave-")
            .is_some_and(|suffix| suffix.len() == 6 && suffix.bytes().all(|b| b.is_ascii_digit()))
    {
        "corrected"
    } else if !test_adaptive && benchmark_valid && selection_bias.is_none() && generation == "zero"
    {
        "legacy-superseded"
    } else {
        bail!("invalid label authority");
    };
    if receipt {
        if value.get("valid") != Some(&Value::Bool(true)) {
            bail!("valid receipt authority requested for a failed row");
        }
        finite_f64(value, "score")?;
    }
    Ok(Authority {
        kind,
        task: task.to_owned(),
        metric: expected_metric.to_owned(),
        generation,
        frozen_seed_sha256,
        model_checkpoint_sha256,
        hall_of_fame_sha256,
        selection_bias,
    })
}

fn protocol_for_manifest(manifest: &Value, authority: &Authority) -> Result<String> {
    let train = required_str(manifest, "train_sha256")?;
    let test = required_str(manifest, "test_sha256")?;
    if !is_sha256(train) || !is_sha256(test) {
        bail!("manifest content hashes are invalid");
    }
    let material = json!({
        "metric": authority.metric,
        "score_tier": "test_exact",
        "task": authority.task,
        "test_sha256": test,
        "train_sha256": train,
        "version": "test-exact-v1",
    });
    let computed = sha256_bytes(&serde_json::to_vec(&material)?);
    if let Some(pinned) = manifest.get("protocol_sha256").and_then(Value::as_str) {
        if pinned != computed {
            bail!("manifest protocol hash drift");
        }
    }
    Ok(computed)
}

pub fn open_corpus(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let connection =
        Connection::open(path).with_context(|| format!("open corpus {}", path.display()))?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    connection.execute_batch(
        r#"
        PRAGMA journal_mode=WAL;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-65536;
        PRAGMA mmap_size=0;
        CREATE TABLE IF NOT EXISTS observations(
          task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
          semantic_sha256 TEXT NOT NULL,canonical_genome TEXT NOT NULL,
          candidate_source TEXT NOT NULL,
          metric TEXT NOT NULL,score REAL NOT NULL,
          secondary_metric TEXT,secondary_score REAL,
          runtime_seconds REAL NOT NULL,
          train_sha256 TEXT NOT NULL,test_sha256 TEXT NOT NULL,
          receipt_sha256 TEXT NOT NULL,
          receipt_path TEXT NOT NULL,manifest_sha256 TEXT NOT NULL,
          frozen_seed_sha256 TEXT NOT NULL,
          model_checkpoint_sha256 TEXT NOT NULL,
          hall_of_fame_sha256 TEXT NOT NULL,
          created_at_ms INTEGER NOT NULL,
          authority_kind TEXT NOT NULL DEFAULT 'legacy-superseded',
          selection_bias TEXT,
          strict_wave TEXT,
          predictions_path TEXT,
          predictions_sha256 TEXT,
          prediction_count INTEGER,
          balanced_accuracy REAL,
          protocol_sha256 TEXT,
          origin_class TEXT NOT NULL DEFAULT 'unknown',
          parentage_json TEXT,
          PRIMARY KEY(task,dataset_hash,semantic_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS failures(
          task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,
          failure TEXT,failure_message TEXT,runtime_seconds REAL,
          receipt_sha256 TEXT,created_at_ms INTEGER,
          PRIMARY KEY(task,dataset_hash,semantic_sha256,receipt_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS ingested_jobs(
          job_id TEXT PRIMARY KEY,manifest_sha256 TEXT NOT NULL,
          receipt_sha256 TEXT NOT NULL,rows INTEGER NOT NULL,
          valid INTEGER NOT NULL,ingested_at_ms INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS causal_evaluation_receipts_v2(
          schema TEXT NOT NULL,
          receipt_sha256 TEXT NOT NULL,
          manifest_sha256 TEXT NOT NULL,
          candidate_sha256 TEXT NOT NULL,
          task TEXT NOT NULL,
          context_identity TEXT NOT NULL,
          accepted_parent_checkpoint_sha256 TEXT NOT NULL,
          wave INTEGER NOT NULL,
          lane TEXT NOT NULL,
          pair_index INTEGER NOT NULL,
          pair_id TEXT NOT NULL,
          score REAL,
          success INTEGER NOT NULL,
          proposal_valid INTEGER NOT NULL,
          runtime_seconds REAL NOT NULL,
          ancestry TEXT NOT NULL,
          cross_context_backfill INTEGER NOT NULL,
          valid_lineage INTEGER NOT NULL,
          canary INTEGER NOT NULL,
          strict_wave TEXT NOT NULL,
          created_at_ms INTEGER NOT NULL,
          PRIMARY KEY(receipt_sha256,manifest_sha256,candidate_sha256)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS causal_receipts_pair_v2
          ON causal_evaluation_receipts_v2(
            task,accepted_parent_checkpoint_sha256,pair_id,lane
          );
        CREATE TRIGGER IF NOT EXISTS causal_receipts_no_update_v2
          BEFORE UPDATE ON causal_evaluation_receipts_v2
          BEGIN SELECT RAISE(ABORT, 'causal receipts are append-only'); END;
        CREATE TRIGGER IF NOT EXISTS causal_receipts_no_delete_v2
          BEFORE DELETE ON causal_evaluation_receipts_v2
          BEGIN SELECT RAISE(ABORT, 'causal receipts are append-only'); END;
        "#,
    )?;
    let existing = table_columns(&connection, "observations")?;
    for (name, declaration) in [
        (
            "authority_kind",
            "TEXT NOT NULL DEFAULT 'legacy-superseded'",
        ),
        ("selection_bias", "TEXT"),
        ("strict_wave", "TEXT"),
        ("predictions_path", "TEXT"),
        ("predictions_sha256", "TEXT"),
        ("prediction_count", "INTEGER"),
        ("balanced_accuracy", "REAL"),
        ("protocol_sha256", "TEXT"),
        ("origin_class", "TEXT NOT NULL DEFAULT 'unknown'"),
        ("parentage_json", "TEXT"),
    ] {
        if !existing.contains(name) {
            connection.execute(
                &format!("ALTER TABLE observations ADD COLUMN {name} {declaration}"),
                [],
            )?;
        }
    }
    connection.execute_batch(
        r#"
        UPDATE observations SET origin_class=CASE
          WHEN candidate_source IN ('accepted_dl_checkpoint','current_checkpoint_dl_proposal')
            OR candidate_source LIKE '%dl_proposal%' THEN 'dl_origin'
          WHEN candidate_source IN (
            'deterministic_rust_mutation_crossover',
            'registry_aware_deterministic_backfill'
          ) THEN 'dl_descendant'
          ELSE 'gp_origin'
        END
        WHERE origin_class='unknown';
        CREATE INDEX IF NOT EXISTS observations_dataset_protocol_best_v2
          ON observations(task,protocol_sha256,dataset_hash,score DESC,semantic_sha256 ASC);
        CREATE INDEX IF NOT EXISTS observations_semantic_protocol_best_v2
          ON observations(task,protocol_sha256,semantic_sha256,score DESC,dataset_hash ASC);
        CREATE INDEX IF NOT EXISTS observations_origin_protocol_lift_v2
          ON observations(
            task,protocol_sha256,origin_class,dataset_hash,score DESC,semantic_sha256 ASC
          );
        CREATE INDEX IF NOT EXISTS observations_checkpoint_protocol_lift_v2
          ON observations(
            task,protocol_sha256,model_checkpoint_sha256,dataset_hash,score DESC
          );
        CREATE INDEX IF NOT EXISTS observations_checkpoint_v1
          ON observations(model_checkpoint_sha256, created_at_ms);
        "#,
    )?;
    Ok(connection)
}

fn table_columns(connection: &Connection, table: &str) -> Result<BTreeSet<String>> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(columns)
}

pub fn ingest_receipt(corpus_path: &Path, job_id: &str, output: &Path) -> Result<IngestSummary> {
    ingest_receipt_with_manifest(
        corpus_path,
        job_id,
        output,
        &output.join(".portable-manifest.json"),
    )
}

pub fn ingest_receipt_with_manifest(
    corpus_path: &Path,
    job_id: &str,
    output: &Path,
    manifest_path: &Path,
) -> Result<IngestSummary> {
    let mut connection = open_corpus(corpus_path)?;
    ingest_receipt_with_manifest_connection(&mut connection, job_id, output, manifest_path)
}

/// Replay a durable receipt through an already migrated corpus connection.
///
/// Bulk receipt recovery uses this entry point so schema migration and SQLite
/// page-cache warmup happen once per backfill rather than once per job. Each
/// job still commits independently, preserving crash-safe resumability.
pub fn ingest_receipt_with_manifest_connection(
    connection: &mut Connection,
    job_id: &str,
    output: &Path,
    manifest_path: &Path,
) -> Result<IngestSummary> {
    if job_id.trim().is_empty() {
        bail!("job id is empty");
    }
    if !manifest_path.is_file() || !output.join("SHARD_OK").is_file() {
        bail!("{job_id}: portable manifest or SHARD_OK missing");
    }
    let manifest = read_json(manifest_path)?;
    let task = required_str(&manifest, "task")?;
    let dataset_hash = required_str(&manifest, "dataset_hash")?;
    let manifest_authority = authority(
        manifest
            .get("label_authority")
            .context("manifest label authority is absent")?,
        task,
        false,
    )?;
    let protocol_sha256 = protocol_for_manifest(&manifest, &manifest_authority)?;
    let candidates = manifest
        .get("candidates")
        .and_then(Value::as_array)
        .context("manifest candidates are absent")?;
    if candidates.is_empty() {
        bail!("{job_id}: manifest candidate set is empty");
    }
    let mut drafts = BTreeMap::<String, CandidateDraft>::new();
    for candidate in candidates {
        let semantic = required_str(candidate, "semantic_sha256")?.to_owned();
        if !is_sha256(&semantic) {
            bail!("{job_id}: candidate semantic hash is invalid");
        }
        let genome = required_str(candidate, "canonical_genome")?.to_owned();
        let source = candidate
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let parents = candidate
            .get("parentage")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .filter(|parent| is_sha256(parent))
                            .map(str::to_owned)
                            .context("candidate parentage contains an invalid semantic")
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        let logical_source = candidate
            .get("evidence")
            .and_then(|value| value.get("logical_source"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let evidence = candidate.get("evidence");
        let evaluation_lane = evidence
            .and_then(|value| value.get("evaluation_lane"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        if evaluation_lane.as_deref().is_some_and(|lane| {
            !matches!(
                lane,
                "native_dl_direct" | "matched_gp_comparator" | "gp_exploration"
            )
        }) {
            bail!("{job_id}: candidate evaluation lane is invalid");
        }
        let matched_pair_index = evidence
            .and_then(|value| value.get("matched_pair_index"))
            .and_then(Value::as_u64);
        let pair_id = evidence
            .and_then(|value| value.get("pair_id"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        if pair_id.as_deref().is_some_and(|value| !is_sha256(value)) {
            bail!("{job_id}: candidate causal pair identity is invalid");
        }
        if drafts
            .insert(
                semantic,
                CandidateDraft {
                    canonical_genome: genome,
                    source,
                    parents,
                    logical_source,
                    evaluation_lane,
                    matched_pair_index,
                    pair_id,
                    proposal_valid: evidence
                        .and_then(|value| value.get("proposal_valid"))
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                    zero_dl_ancestry: evidence
                        .and_then(|value| value.get("zero_dl_ancestry"))
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    ancestry: evidence
                        .and_then(|value| value.get("dl_ancestry_kind"))
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    cross_context_backfill: evidence
                        .and_then(|value| value.get("cross_context_backfill"))
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    valid_lineage: evidence
                        .and_then(|value| value.get("valid_lineage"))
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                },
            )
            .is_some()
        {
            bail!("{job_id}: duplicate candidate semantic");
        }
    }
    let mut dl_contributed = drafts
        .iter()
        .filter(|(_, draft)| dl_source(&draft.source))
        .map(|(semantic, _)| semantic.clone())
        .collect::<BTreeSet<_>>();
    loop {
        let before = dl_contributed.len();
        for (semantic, draft) in &drafts {
            if draft.logical_source.as_deref().is_some_and(dl_source)
                || draft
                    .parents
                    .iter()
                    .any(|parent| dl_contributed.contains(parent))
            {
                dl_contributed.insert(semantic.clone());
            }
        }
        if dl_contributed.len() == before {
            break;
        }
    }
    let expected = drafts
        .into_iter()
        .map(|(semantic, draft)| {
            let origin_class = if dl_source(&draft.source) {
                "dl_origin"
            } else if dl_contributed.contains(&semantic) {
                "dl_descendant"
            } else {
                "gp_origin"
            };
            let parentage_json = (!draft.parents.is_empty())
                .then(|| serde_json::to_string(&draft.parents))
                .transpose()?;
            let ancestry = draft.ancestry.unwrap_or_else(|| {
                if draft.zero_dl_ancestry || origin_class == "gp_origin" {
                    "none".to_owned()
                } else if origin_class == "dl_origin" || origin_class == "dl_descendant" {
                    "accepted_dl".to_owned()
                } else {
                    "mixed".to_owned()
                }
            });
            Ok((
                semantic,
                ExpectedCandidate {
                    canonical_genome: draft.canonical_genome,
                    source: draft.source,
                    origin_class: origin_class.to_owned(),
                    parentage_json,
                    evaluation_lane: draft.evaluation_lane,
                    matched_pair_index: draft.matched_pair_index,
                    pair_id: draft.pair_id,
                    proposal_valid: draft.proposal_valid,
                    ancestry,
                    cross_context_backfill: draft.cross_context_backfill,
                    valid_lineage: draft.valid_lineage,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let receipts_path = output.join(format!("{dataset_hash}.receipts.jsonl"));
    let receipt_sha256 = sha256_file(&receipts_path)?;
    let mut valid = Vec::<ValidatedReceipt>::new();
    let mut failures = Vec::<Value>::new();
    let mut causal_receipts = Vec::<CausalReceiptDraft>::new();
    let causal_wave = strict_wave_number(&manifest_authority.generation);
    let canary = manifest
        .get("receipt_purpose")
        .and_then(Value::as_str)
        .is_some_and(|purpose| purpose == "canary")
        || manifest
            .get("canary")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let mut seen = BTreeSet::new();
    for (offset, line) in BufReader::new(File::open(&receipts_path)?)
        .lines()
        .enumerate()
    {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .with_context(|| format!("{job_id}:{}: invalid JSON", offset + 1))?;
        let semantic = required_str(&row, "semantic_sha256")?;
        let candidate = expected
            .get(semantic)
            .with_context(|| format!("{job_id}:{}: unknown candidate", offset + 1))?;
        if !seen.insert(semantic.to_owned()) {
            bail!("{job_id}:{}: duplicate candidate receipt", offset + 1);
        }
        if required_str(&row, "task")? != task
            || required_str(&row, "dataset_hash")? != dataset_hash
            || required_str(&row, "canonical_genome")? != candidate.canonical_genome
        {
            bail!("{job_id}:{}: receipt identity drift", offset + 1);
        }
        for field in [
            "frozen_seed_sha256",
            "model_checkpoint_sha256",
            "hall_of_fame_sha256",
        ] {
            if row.get(field) != manifest["label_authority"].get(field) {
                bail!("{job_id}:{}: provenance pin mismatch", offset + 1);
            }
        }
        if let (Some(wave), Some(lane), Some(pair_index)) = (
            causal_wave,
            candidate.evaluation_lane.as_deref(),
            candidate.matched_pair_index,
        ) {
            if wave >= 191 && matches!(lane, "native_dl_direct" | "matched_gp_comparator") {
                let derived_pair_id = causal_pair_identity(
                    &manifest_authority.model_checkpoint_sha256,
                    wave,
                    task,
                    dataset_hash,
                    pair_index,
                );
                if candidate
                    .pair_id
                    .as_deref()
                    .is_some_and(|value| value != derived_pair_id)
                {
                    bail!("{job_id}:{}: causal pair identity drift", offset + 1);
                }
                let success = row.get("valid") == Some(&Value::Bool(true));
                let score = row.get("score").and_then(Value::as_f64);
                if score.is_some_and(|value| !value.is_finite()) {
                    bail!(
                        "{job_id}:{}: causal receipt score is not finite",
                        offset + 1
                    );
                }
                causal_receipts.push(CausalReceiptDraft {
                    candidate_sha256: semantic.to_owned(),
                    lane: lane.to_owned(),
                    pair_index,
                    pair_id: derived_pair_id,
                    score,
                    success,
                    proposal_valid: candidate.proposal_valid,
                    runtime_seconds: finite_f64(&row, "runtime_seconds")?,
                    ancestry: candidate.ancestry.clone(),
                    cross_context_backfill: candidate.cross_context_backfill,
                    valid_lineage: candidate.valid_lineage,
                    created_at_ms: row
                        .get("created_at_ms")
                        .and_then(Value::as_i64)
                        .unwrap_or(0),
                });
            }
        } else if causal_wave.is_some_and(|wave| wave >= 191)
            && candidate
                .evaluation_lane
                .as_deref()
                .is_some_and(|lane| matches!(lane, "native_dl_direct" | "matched_gp_comparator"))
        {
            bail!(
                "{job_id}:{}: paired causal lane lacks pair index",
                offset + 1
            );
        }
        if row.get("valid") == Some(&Value::Bool(true)) {
            let receipt_authority = authority(&row, task, true)?;
            if receipt_authority.kind != manifest_authority.kind {
                bail!("{job_id}:{}: authority generation drift", offset + 1);
            }
            let predictions_path = if receipt_authority.kind == "corrected" {
                Some(validate_prediction_evidence(
                    output,
                    &row,
                    job_id,
                    offset + 1,
                )?)
            } else {
                None
            };
            valid.push(ValidatedReceipt {
                value: row,
                candidate_source: candidate.source.clone(),
                origin_class: candidate.origin_class.clone(),
                parentage_json: candidate.parentage_json.clone(),
                predictions_path,
            });
        } else {
            for field in [
                "test_adaptive",
                "benchmark_valid",
                "selection_bias",
                "generation",
            ] {
                if row.get(field) != manifest["label_authority"].get(field) {
                    bail!("{job_id}:{}: failure authority drift", offset + 1);
                }
            }
            finite_f64(&row, "runtime_seconds")?;
            failures.push(row);
        }
    }
    if seen != expected.keys().cloned().collect() {
        bail!(
            "{job_id}: expected {} rows, found {}",
            expected.len(),
            seen.len()
        );
    }
    let manifest_sha256 = valid
        .first()
        .map(|row| required_str(&row.value, "manifest_sha256"))
        .transpose()?
        .or_else(|| {
            failures
                .first()
                .and_then(|row| row.get("manifest_sha256"))
                .and_then(Value::as_str)
        })
        .context("receipt manifest hash is absent")?
        .to_owned();

    let already: Option<(String, String, i64, i64)> = connection
        .query_row(
            "SELECT manifest_sha256,receipt_sha256,rows,valid FROM ingested_jobs WHERE job_id=?",
            [job_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?;
    if let Some((prior_manifest, prior_receipt, rows, valid_rows)) = already {
        if prior_manifest != manifest_sha256 || prior_receipt != receipt_sha256 {
            bail!("{job_id}: immutable ingested job receipt conflicts with durable corpus");
        }
        if causal_receipts.is_empty() {
            return Ok(IngestSummary {
                rows: rows as usize,
                valid: valid_rows as usize,
                failed: (rows - valid_rows) as usize,
                already_ingested: true,
            });
        }
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        append_causal_receipts(
            &tx,
            &causal_receipts,
            &receipt_sha256,
            &manifest_sha256,
            task,
            dataset_hash,
            &manifest_authority,
            causal_wave,
            canary,
        )?;
        tx.commit()?;
        return Ok(IngestSummary {
            rows: rows as usize,
            valid: valid_rows as usize,
            failed: (rows - valid_rows) as usize,
            already_ingested: true,
        });
    }
    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    append_causal_receipts(
        &tx,
        &causal_receipts,
        &receipt_sha256,
        &manifest_sha256,
        task,
        dataset_hash,
        &manifest_authority,
        causal_wave,
        canary,
    )?;
    if !canary {
        for receipt in &valid {
            upsert_observation(
                &tx,
                receipt,
                &receipt_sha256,
                &receipts_path,
                &manifest_authority,
                &protocol_sha256,
            )?;
        }
        for row in &failures {
            tx.execute(
                "INSERT OR IGNORE INTO failures VALUES(?,?,?,?,?,?,?,?)",
                params![
                    required_str(row, "task")?,
                    required_str(row, "dataset_hash")?,
                    required_str(row, "semantic_sha256")?,
                    row.get("failure").and_then(Value::as_str),
                    row.get("failure_message").and_then(Value::as_str),
                    finite_f64(row, "runtime_seconds")?,
                    receipt_sha256,
                    row.get("created_at_ms")
                        .and_then(Value::as_i64)
                        .unwrap_or(0),
                ],
            )?;
        }
    }
    tx.execute(
        "INSERT INTO ingested_jobs VALUES(?,?,?,?,?,?)",
        params![
            job_id,
            manifest_sha256,
            receipt_sha256,
            seen.len() as i64,
            valid.len() as i64,
            now_ms()
        ],
    )?;
    tx.commit()?;
    Ok(IngestSummary {
        rows: seen.len(),
        valid: valid.len(),
        failed: failures.len(),
        already_ingested: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_causal_receipts(
    tx: &rusqlite::Transaction<'_>,
    receipts: &[CausalReceiptDraft],
    receipt_sha256: &str,
    manifest_sha256: &str,
    task: &str,
    context_identity: &str,
    authority: &Authority,
    wave: Option<u64>,
    canary: bool,
) -> Result<()> {
    let Some(wave) = wave else {
        return Ok(());
    };
    for receipt in receipts {
        tx.execute(
            r#"
            INSERT OR IGNORE INTO causal_evaluation_receipts_v2(
              schema,receipt_sha256,manifest_sha256,candidate_sha256,
              task,context_identity,accepted_parent_checkpoint_sha256,wave,
              lane,pair_index,pair_id,score,success,proposal_valid,
              runtime_seconds,ancestry,cross_context_backfill,valid_lineage,
              canary,strict_wave,created_at_ms
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            "#,
            params![
                "CausalEvaluationReceipt.v2",
                receipt_sha256,
                manifest_sha256,
                receipt.candidate_sha256,
                task,
                context_identity,
                authority.model_checkpoint_sha256,
                wave as i64,
                receipt.lane,
                receipt.pair_index as i64,
                receipt.pair_id,
                receipt.score,
                receipt.success,
                receipt.proposal_valid,
                receipt.runtime_seconds,
                receipt.ancestry,
                receipt.cross_context_backfill,
                receipt.valid_lineage,
                canary,
                authority.generation,
                receipt.created_at_ms,
            ],
        )?;
    }
    Ok(())
}

fn validate_prediction_evidence(
    output: &Path,
    row: &Value,
    job_id: &str,
    line: usize,
) -> Result<String> {
    let relative = Path::new(required_str(row, "predictions_path")?);
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        bail!("{job_id}:{line}: unsafe prediction evidence path");
    }
    let prediction_path = output.join(relative);
    let prediction_count = row
        .get("prediction_count")
        .and_then(Value::as_u64)
        .filter(|count| *count > 0)
        .context("prediction count is absent")?;
    if !prediction_path.is_file()
        || prediction_path.metadata()?.len() != prediction_count.saturating_mul(8)
    {
        bail!("{job_id}:{line}: prediction row alignment drift");
    }
    if sha256_file(&prediction_path)? != required_str(row, "predictions_sha256")? {
        bail!("{job_id}:{line}: prediction evidence hash drift");
    }
    Ok(prediction_path.to_string_lossy().into_owned())
}

fn upsert_observation(
    tx: &rusqlite::Transaction<'_>,
    receipt: &ValidatedReceipt,
    receipt_sha256: &str,
    receipts_path: &Path,
    authority: &Authority,
    protocol_sha256: &str,
) -> Result<()> {
    let row = &receipt.value;
    let task = required_str(row, "task")?;
    let dataset = required_str(row, "dataset_hash")?;
    let semantic = required_str(row, "semantic_sha256")?;
    let existing: Option<(f64, String, Option<String>)> = tx
        .query_row(
            "SELECT score,receipt_sha256,protocol_sha256 FROM observations
             WHERE task=? AND dataset_hash=? AND semantic_sha256=?",
            params![task, dataset, semantic],
            |query| Ok((query.get(0)?, query.get(1)?, query.get(2)?)),
        )
        .optional()?;
    let score = finite_f64(row, "score")?;
    let should_write =
        existing
            .as_ref()
            .is_none_or(|(prior_score, prior_receipt, prior_protocol)| {
                prior_protocol.as_deref() != Some(protocol_sha256)
                    || score > *prior_score
                    || (score == *prior_score && receipt_sha256 > prior_receipt.as_str())
            });
    if !should_write {
        return Ok(());
    }
    tx.execute(
        r#"
        INSERT OR REPLACE INTO observations(
          task,dataset_hash,semantic_sha256,canonical_genome,candidate_source,
          metric,score,secondary_metric,secondary_score,runtime_seconds,
          train_sha256,test_sha256,receipt_sha256,receipt_path,manifest_sha256,
          frozen_seed_sha256,model_checkpoint_sha256,hall_of_fame_sha256,
          created_at_ms,authority_kind,selection_bias,strict_wave,
          predictions_path,predictions_sha256,prediction_count,
          balanced_accuracy,protocol_sha256,origin_class,parentage_json
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        "#,
        params![
            task,
            dataset,
            semantic,
            required_str(row, "canonical_genome")?,
            receipt.candidate_source,
            required_str(row, "metric")?,
            score,
            row.get("secondary_metric").and_then(Value::as_str),
            row.get("secondary_score").and_then(Value::as_f64),
            finite_f64(row, "runtime_seconds")?,
            required_str(row, "train_sha256")?,
            required_str(row, "test_sha256")?,
            receipt_sha256,
            receipts_path.to_string_lossy(),
            required_str(row, "manifest_sha256")?,
            authority.frozen_seed_sha256,
            authority.model_checkpoint_sha256,
            authority.hall_of_fame_sha256,
            row.get("created_at_ms")
                .and_then(Value::as_i64)
                .unwrap_or(0),
            authority.kind,
            authority.selection_bias,
            if authority.kind == "corrected" {
                Some(authority.generation.as_str())
            } else {
                None
            },
            receipt.predictions_path,
            row.get("predictions_sha256").and_then(Value::as_str),
            row.get("prediction_count").and_then(Value::as_i64),
            row.get("balanced_accuracy").and_then(Value::as_f64),
            protocol_sha256,
            receipt.origin_class,
            receipt.parentage_json,
        ],
    )?;
    Ok(())
}

pub fn reconcile_local(root: &Path, limit: usize) -> Result<Value> {
    reconcile_local_mode(root, limit, true)
}

pub fn reconcile_local_mode(root: &Path, limit: usize, refresh_indexes: bool) -> Result<Value> {
    let receipts_root = root.join("receipts");
    let corpus_path = root.join("corpus/strict-test-exact.sqlite");
    let backlog_path = root.join("ops/backlog.sqlite");
    let connection = open_corpus(&corpus_path)?;
    let already: BTreeSet<String> = {
        let mut statement = connection.prepare("SELECT job_id FROM ingested_jobs")?;
        let jobs = statement
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        jobs
    };
    drop(connection);
    let submitted = if backlog_path.is_file() {
        let backlog = Connection::open_with_flags(
            &backlog_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        backlog.busy_timeout(std::time::Duration::from_secs(60))?;
        let mut statement = backlog.prepare(
            "SELECT job_id,updated_at_ms FROM claims
             WHERE state='submitted'",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?
            .collect::<rusqlite::Result<BTreeMap<_, _>>>()?;
        rows
    } else {
        BTreeMap::new()
    };
    let mut receipt_dirs = if receipts_root.is_dir() {
        fs::read_dir(&receipts_root)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|job_id| job_id.starts_with("jte-"))
            })
            .filter(|entry| {
                submitted.is_empty()
                    || entry
                        .file_name()
                        .to_str()
                        .is_some_and(|job_id| submitted.contains_key(job_id))
            })
            .map(|entry| entry.path())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    receipt_dirs.sort_by(|left, right| {
        let left_job = left
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let right_job = right
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        submitted
            .get(right_job)
            .unwrap_or(&0)
            .cmp(submitted.get(left_job).unwrap_or(&0))
            .then_with(|| left_job.cmp(right_job))
    });
    let mut ingested = 0_u64;
    let mut failures = 0_u64;
    let mut rows = 0_u64;
    let mut valid = 0_u64;
    let mut errors = Vec::new();
    for receipt_dir in receipt_dirs
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|job| !already.contains(job))
        })
        .take(limit)
    {
        let job_id = receipt_dir
            .file_name()
            .and_then(|name| name.to_str())
            .context("receipt directory has no job id")?;
        match ingest_receipt(&corpus_path, job_id, &receipt_dir) {
            Ok(summary) => {
                ingested += u64::from(!summary.already_ingested);
                rows += summary.rows as u64;
                valid += summary.valid as u64;
                update_claim(root, job_id, "ingested")?;
            }
            Err(error) => {
                failures += 1;
                let message = error.to_string();
                errors.push(json!({"job_id": job_id, "error": message}));
                update_claim(
                    root,
                    job_id,
                    &format!(
                        "ingest_failed:{}",
                        message.chars().take(500).collect::<String>()
                    ),
                )?;
            }
        }
    }
    let indexes = if refresh_indexes {
        rebuild_indexes(root)?
    } else {
        let corpus = open_corpus(&corpus_path)?;
        let observations: i64 =
            corpus.query_row("SELECT COUNT(*) FROM observations", [], |row| row.get(0))?;
        let evaluator_failures: i64 =
            corpus.query_row("SELECT COUNT(*) FROM failures", [], |row| row.get(0))?;
        json!({
            "exact_rows_total": observations,
            "evaluator_failures_total": evaluator_failures,
            "strict_wins": Value::Null,
            "indexes_refreshed": false,
        })
    };
    let result = json!({
        // A document with no `schema` key can never be version-gated: every
        // consumer has to guess, and a future shape change is undetectable
        // until something downstream dies. `ops/last-ingest.json` is read as
        // the label pointer by jope_selfplay_trainer_watchdog.rs:107 and as a
        // heartbeat by jope_selfplay_tui.rs:80; both read fields off a
        // serde_json::Value, so adding keys is backward compatible.
        "schema": "JopeSelfplayIngestReceipt.v1",
        "generated_at_ms": now_ms(),
        "completed_seen": ingested + failures,
        "ingested": ingested,
        "failures": failures,
        "rows": rows,
        "valid": valid,
        "errors": errors,
        "exact_rows_total": indexes["exact_rows_total"],
        "evaluator_failures_total": indexes["evaluator_failures_total"],
        "strict_wins": indexes["strict_wins"],
    });
    atomic_json(&root.join("ops/last-ingest.json"), &result)?;
    Ok(result)
}

/// Materialize completed SCQ receipts, then run the same fail-closed local
/// ingestion path. This is the only scheduler interaction in the ingestion
/// module and it is read/fetch-only: it never holds, releases, submits, or
/// cancels work.
pub fn reconcile_scheduler(
    root: &Path,
    scq: &Path,
    runs_root: &Path,
    limit: usize,
    refresh_indexes: bool,
) -> Result<Value> {
    let scheduler = Command::new(scq)
        .args(["--json", "ls"])
        .output()
        .with_context(|| format!("run {} --json ls", scq.display()))?;
    if !scheduler.status.success() {
        bail!(
            "SCQ inventory failed: {}",
            String::from_utf8_lossy(if scheduler.stderr.is_empty() {
                &scheduler.stdout
            } else {
                &scheduler.stderr
            })
            .trim()
        );
    }
    let inventory: Value =
        serde_json::from_slice(&scheduler.stdout).context("parse SCQ inventory JSON")?;
    let completed = inventory
        .get("Jobs")
        .and_then(|value| value.get("jobs"))
        .or_else(|| inventory.get("jobs"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|job| {
            (job.get("state").and_then(Value::as_str) == Some("completed"))
                .then(|| {
                    job.get("id")
                        .or_else(|| job.get("job_id"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .flatten()
        })
        .collect::<BTreeSet<_>>();

    let backlog_path = root.join("ops/backlog.sqlite");
    let corpus_path = root.join("corpus/strict-test-exact.sqlite");
    let receipts_root = root.join("receipts");
    fs::create_dir_all(&receipts_root)?;
    let backlog = Connection::open_with_flags(
        &backlog_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("open backlog {}", backlog_path.display()))?;
    backlog.busy_timeout(std::time::Duration::from_secs(60))?;
    let corpus = open_corpus(&corpus_path)?;
    let already = {
        let mut statement = corpus.prepare("SELECT job_id FROM ingested_jobs")?;
        let rows = statement
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<BTreeSet<String>>>()?;
        rows
    };
    drop(corpus);

    let mut statement = backlog.prepare(
        "SELECT job_id FROM claims
         WHERE state='submitted' OR state LIKE 'ingest_failed:%'
         ORDER BY CASE WHEN state='submitted' THEN 0 ELSE 1 END,
                  updated_at_ms DESC,job_id",
    )?;
    let claim_ids = statement
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut materialized = 0_u64;
    let mut fetch_failures = Vec::new();
    for job_id in claim_ids
        .into_iter()
        .filter(|job_id| !already.contains(job_id))
        .filter(|job_id| !receipts_root.join(job_id).is_dir())
        .filter(|job_id| {
            local_receipt_dir(runs_root, job_id).is_some() || completed.contains(job_id)
        })
        .take(limit)
    {
        let final_path = receipts_root.join(&job_id);
        let temporary = receipts_root.join(format!(".materialize-{job_id}-{}", std::process::id()));
        if temporary.exists() {
            bail!("stale ingestion temporary path {}", temporary.display());
        }
        let result = if let Some(local) = local_receipt_dir(runs_root, &job_id) {
            copy_tree(&local, &temporary)
        } else {
            fs::create_dir(&temporary)?;
            let fetched = Command::new(scq)
                .arg("fetch")
                .arg(&job_id)
                .arg("--output")
                .arg(&temporary)
                .output()
                .with_context(|| format!("fetch completed SCQ job {job_id}"))?;
            if !fetched.status.success() {
                Err(anyhow::anyhow!(
                    "{}",
                    String::from_utf8_lossy(if fetched.stderr.is_empty() {
                        &fetched.stdout
                    } else {
                        &fetched.stderr
                    })
                    .trim()
                ))
            } else {
                let source = resolve_receipt_dir(&temporary)?;
                let staged =
                    temporary.with_file_name(format!(".receipt-{job_id}-{}", std::process::id()));
                fs::rename(source, &staged)?;
                fs::remove_dir_all(&temporary)?;
                fs::rename(staged, &temporary)?;
                Ok(())
            }
        };
        match result {
            Ok(()) => {
                fs::rename(&temporary, &final_path)?;
                // A prior read-before-start failure is recoverable once a
                // complete durable receipt has actually materialized.
                update_claim(root, &job_id, "submitted")?;
                materialized += 1;
            }
            Err(error) => {
                if temporary.is_dir() {
                    fs::remove_dir_all(&temporary)?;
                }
                fetch_failures.push(json!({"job_id": job_id, "error": error.to_string()}));
            }
        }
    }
    let mut result = reconcile_local_mode(root, limit, refresh_indexes)?;
    let object = result
        .as_object_mut()
        .context("local reconciliation did not return an object")?;
    object.insert("materialized".to_owned(), json!(materialized));
    object.insert("fetch_failures".to_owned(), json!(fetch_failures));
    Ok(result)
}

fn local_receipt_dir(runs_root: &Path, job_id: &str) -> Option<PathBuf> {
    let attempts = runs_root.join(job_id).join("attempts");
    let mut candidates = fs::read_dir(attempts)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.reverse();
    candidates
        .into_iter()
        .map(|attempt| attempt.join("outputs/receipt"))
        .find(|receipt| receipt.join("SHARD_OK").is_file())
}

fn resolve_receipt_dir(fetch_root: &Path) -> Result<PathBuf> {
    let direct = fetch_root.join("receipt");
    if direct.is_dir() {
        return Ok(direct);
    }
    let mut matches = Vec::new();
    collect_named_file_parents(fetch_root, "SHARD_OK", &mut matches)?;
    if matches.len() != 1 {
        bail!(
            "could not uniquely resolve receipt under {}",
            fetch_root.display()
        );
    }
    Ok(matches.remove(0))
}

fn collect_named_file_parents(root: &Path, name: &str, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            bail!("symlink is not allowed in fetched receipt");
        }
        if file_type.is_dir() {
            collect_named_file_parents(&entry.path(), name, output)?;
        } else if entry.file_name() == name {
            output.push(
                entry
                    .path()
                    .parent()
                    .context("matched receipt marker has no parent")?
                    .to_path_buf(),
            );
        }
    }
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            bail!("symlink is not allowed in durable receipt");
        }
        let target = destination.join(entry.file_name());
        if file_type.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), target)?;
        } else {
            bail!("unsupported filesystem entry in durable receipt");
        }
    }
    Ok(())
}

fn update_claim(root: &Path, job_id: &str, state: &str) -> Result<()> {
    let path = root.join("ops/backlog.sqlite");
    if !path.is_file() {
        return Ok(());
    }
    let connection = Connection::open(path)?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    if table_exists(&connection, "claims")? {
        connection.execute(
            "UPDATE claims SET state=?,updated_at_ms=? WHERE job_id=?",
            params![state, now_ms(), job_id],
        )?;
    }
    Ok(())
}

fn table_exists(connection: &Connection, table: &str) -> Result<bool> {
    Ok(connection
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            [table],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn sqlite_row_json(row: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    let mut value = Map::new();
    for index in 0..row.as_ref().column_count() {
        let name = row.as_ref().column_name(index)?.to_owned();
        let item = match row.get_ref(index)? {
            ValueRef::Null => Value::Null,
            ValueRef::Integer(number) => json!(number),
            ValueRef::Real(number) => json!(number),
            ValueRef::Text(text) => Value::String(String::from_utf8_lossy(text).into_owned()),
            ValueRef::Blob(bytes) => Value::String(hex::encode(bytes)),
        };
        value.insert(name, item);
    }
    Ok(Value::Object(value))
}

fn best_rows(connection: &Connection) -> Result<Vec<Value>> {
    let mut statement = connection.prepare(
        r#"
        SELECT observations.* FROM observations
        JOIN (
          SELECT task,dataset_hash,MAX(score) AS score
          FROM observations GROUP BY task,dataset_hash
        ) best USING(task,dataset_hash,score)
        WHERE semantic_sha256=(
          SELECT MIN(tie.semantic_sha256) FROM observations tie
          WHERE tie.task=observations.task
            AND tie.dataset_hash=observations.dataset_hash
            AND tie.score=observations.score
        )
        ORDER BY task,dataset_hash
        "#,
    )?;
    let rows = statement
        .query_map([], sqlite_row_json)?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

fn known_rows(root: &Path) -> Result<BTreeMap<(String, String), Value>> {
    let pointer_path = root.join("hof/current.json");
    if !pointer_path.is_file() {
        return Ok(BTreeMap::new());
    }
    let pointer = read_json(&pointer_path)?;
    let path = PathBuf::from(required_str(&pointer, "known")?);
    if !path.is_file() {
        bail!("known HOF database is absent");
    }
    if let Some(expected) = pointer.get("known_sha256").and_then(Value::as_str) {
        if sha256_file(&path)? != expected {
            bail!("immutable known-exact snapshot hash drift");
        }
    }
    let known = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    known.execute_batch("PRAGMA temp_store=FILE; PRAGMA cache_size=-65536; PRAGMA mmap_size=0;")?;
    let mut statement = known.prepare(
        r#"
        SELECT known.* FROM known
        JOIN (
          SELECT task,dataset_hash,MAX(score) AS score
          FROM known GROUP BY task,dataset_hash
        ) best USING(task,dataset_hash,score)
        WHERE semantic_sha256=(
          SELECT MIN(tie.semantic_sha256) FROM known tie
          WHERE tie.task=known.task AND tie.dataset_hash=known.dataset_hash
            AND tie.score=known.score
        )
        ORDER BY task,dataset_hash
        "#,
    )?;
    let rows = statement
        .query_map([], sqlite_row_json)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows
        .into_iter()
        .map(|row| {
            (
                (
                    row["task"].as_str().unwrap_or_default().to_owned(),
                    row["dataset_hash"].as_str().unwrap_or_default().to_owned(),
                ),
                row,
            )
        })
        .collect())
}

fn validate_evidence_scope(task: &str, protocol_sha256: &str, limit: usize) -> Result<()> {
    if !matches!(task, "regression" | "binary" | "timeseries" | "multiclass") {
        bail!("evidence task must be regression, binary, timeseries, or multiclass");
    }
    if !is_sha256(protocol_sha256) {
        bail!("evidence protocol is not a SHA-256 value");
    }
    if !(1..=10_000).contains(&limit) {
        bail!("evidence query limit must be in 1..=10000");
    }
    Ok(())
}

pub fn ensure_pair_query_indexes(path: &Path) -> Result<Value> {
    if !path.is_file() {
        return Ok(json!({"present": false, "indexes": []}));
    }
    let connection =
        Connection::open(path).with_context(|| format!("open pair registry {}", path.display()))?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    connection.execute_batch(
        r#"
        CREATE INDEX IF NOT EXISTS pairs_dataset_protocol_v2
          ON pairs(task,dataset_hash,protocol_sha256,semantic_sha256,state);
        CREATE INDEX IF NOT EXISTS pairs_semantic_protocol_v2
          ON pairs(task,semantic_sha256,protocol_sha256,dataset_hash,state);
        "#,
    )?;
    Ok(json!({
        "present": true,
        "indexes": ["pairs_dataset_protocol_v2", "pairs_semantic_protocol_v2"],
    }))
}

fn explain_details(
    connection: &Connection,
    sql: &str,
    values: &[&dyn rusqlite::ToSql],
) -> Result<Vec<String>> {
    let mut statement = connection.prepare(&format!("EXPLAIN QUERY PLAN {sql}"))?;
    let rows = statement.query_map(values, |row| row.get::<_, String>(3))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

// NOTE: this in-process exporter is NOT the production seed-bank path. The
// canonical corpus/robust-gp-seed-bank.v1.json is regenerated every 6h by the
// jope-selfplay-seedbank.{service,timer} systemd user units (see
// jope-dime/systemd/), which run `jope_queue_control --export-seed-bank` with
// the DL-origin merit merge capped at floor(limit/3) per task and write
// receipts under test_exact_gen0/ops/seed-bank-receipts/. Kept for
// jope_gp_bridge's dynamic per-protocol evidence view; do not delete.
pub fn dynamic_seed_bank(
    corpus_path: &Path,
    task: &str,
    protocol_sha256: &str,
    limit: usize,
    minimum_datasets: usize,
) -> Result<Value> {
    validate_evidence_scope(task, protocol_sha256, limit)?;
    if minimum_datasets < 2 {
        bail!("dynamic seed export requires at least two repeat datasets");
    }
    let connection = open_corpus(corpus_path)?;
    let sql = r#"
WITH ranked AS (
  SELECT task,dataset_hash,semantic_sha256,canonical_genome,score,
         ROW_NUMBER() OVER (
           PARTITION BY task,dataset_hash ORDER BY score,semantic_sha256
         ) AS ascending_rank,
         COUNT(*) OVER (PARTITION BY task,dataset_hash) AS dataset_size,
         RANK() OVER (
           PARTITION BY task,dataset_hash ORDER BY score DESC
         ) AS descending_rank
  FROM observations INDEXED BY observations_dataset_protocol_best_v2
  WHERE task=?1 AND protocol_sha256=?2 AND origin_class='gp_origin'
),
aggregated AS (
  SELECT task,semantic_sha256,MAX(canonical_genome) AS canonical_genome,
         COUNT(*) AS dataset_count,
         AVG(CASE WHEN dataset_size=1 THEN 1.0
                  ELSE 1.0*(ascending_rank-1)/(dataset_size-1) END) AS mean_percentile,
         MIN(CASE WHEN dataset_size=1 THEN 1.0
                  ELSE 1.0*(ascending_rank-1)/(dataset_size-1) END) AS worst_percentile,
         SUM(CASE WHEN descending_rank=1 THEN 1 ELSE 0 END) AS strict_wins
  FROM ranked
  GROUP BY task,semantic_sha256
  HAVING COUNT(*)>=?3
)
SELECT task,semantic_sha256,canonical_genome,dataset_count,
       mean_percentile AS mean_within_dataset_percentile,
       worst_percentile AS worst_within_dataset_percentile,
       strict_wins AS strict_win_count
FROM aggregated
ORDER BY mean_percentile DESC,worst_percentile DESC,
         strict_wins DESC,dataset_count DESC,semantic_sha256
LIMIT ?4
"#;
    let mut statement = connection.prepare(sql)?;
    let rows = statement.query_map(
        params![task, protocol_sha256, minimum_datasets as i64, limit as i64],
        sqlite_row_json,
    )?;
    let seeds = rows.collect::<rusqlite::Result<Vec<_>>>()?;
    let watermark: (i64, i64) = connection.query_row(
        "SELECT COUNT(*),COALESCE(MAX(created_at_ms),0) FROM observations
         WHERE task=?1 AND protocol_sha256=?2",
        params![task, protocol_sha256],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok(json!({
        "schema": "JopeRobustGpSeedBank.v1",
        "source": corpus_path,
        "protocol_sha256": protocol_sha256,
        "selection": {
            "task_separated": true,
            "origin_class": "gp_origin",
            "minimum_repeat_datasets": minimum_datasets,
            "limit_per_task": limit,
            "dynamic_current_evidence": true,
        },
        "evidence_watermark": {
            "rows": watermark.0,
            "max_created_at_ms": watermark.1,
        },
        "tasks": {task: seeds},
    }))
}

pub fn evidence_report(
    root: &Path,
    task: &str,
    protocol_sha256: &str,
    dataset: Option<&str>,
    semantic: Option<&str>,
    limit: usize,
) -> Result<Value> {
    validate_evidence_scope(task, protocol_sha256, limit)?;
    if semantic.is_some_and(|value| !is_sha256(value)) {
        bail!("semantic filter is not a SHA-256 value");
    }
    let corpus_path = root.join("corpus/strict-test-exact.sqlite");
    let connection = open_corpus(&corpus_path)?;
    let pair_indexes = ensure_pair_query_indexes(&root.join("ops/pair-registry.sqlite"))?;

    let dataset_sql = r#"
WITH ranked AS (
  SELECT dataset_hash,semantic_sha256,canonical_genome,score,secondary_score,
         candidate_source,origin_class,model_checkpoint_sha256,
         ROW_NUMBER() OVER (
           PARTITION BY dataset_hash ORDER BY score DESC,semantic_sha256
         ) AS evidence_rank
  FROM observations INDEXED BY observations_dataset_protocol_best_v2
  WHERE task=?1 AND protocol_sha256=?2
    AND (?3 IS NULL OR dataset_hash=?3)
)
SELECT dataset_hash,semantic_sha256,canonical_genome,score,secondary_score,
       candidate_source,origin_class,model_checkpoint_sha256
FROM ranked WHERE evidence_rank=1 ORDER BY dataset_hash LIMIT ?4
"#;
    let mut statement = connection.prepare(dataset_sql)?;
    let best_per_dataset = statement
        .query_map(
            params![task, protocol_sha256, dataset, limit as i64],
            sqlite_row_json,
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let semantic_sql = r#"
WITH ranked AS (
  SELECT semantic_sha256,dataset_hash,canonical_genome,score,origin_class,
         ROW_NUMBER() OVER (
           PARTITION BY semantic_sha256 ORDER BY score DESC,dataset_hash
         ) AS dataset_rank
  FROM observations INDEXED BY observations_semantic_protocol_best_v2
  WHERE task=?1 AND protocol_sha256=?2
    AND (?3 IS NULL OR semantic_sha256=?3)
)
SELECT semantic_sha256,dataset_hash,canonical_genome,score,origin_class,dataset_rank
FROM ranked WHERE dataset_rank<=?4
ORDER BY semantic_sha256,dataset_rank
"#;
    let mut statement = connection.prepare(semantic_sql)?;
    let best_datasets_per_semantic = statement
        .query_map(
            params![task, protocol_sha256, semantic, limit as i64],
            sqlite_row_json,
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let repeat_sql = r#"
WITH ranked AS (
  SELECT dataset_hash,semantic_sha256,canonical_genome,score,
         ROW_NUMBER() OVER (
           PARTITION BY dataset_hash ORDER BY score,semantic_sha256
         ) AS ascending_rank,
         COUNT(*) OVER (PARTITION BY dataset_hash) AS dataset_size,
         RANK() OVER (PARTITION BY dataset_hash ORDER BY score DESC) AS winner_rank
  FROM observations INDEXED BY observations_dataset_protocol_best_v2
  WHERE task=?1 AND protocol_sha256=?2
),
aggregated AS (
  SELECT semantic_sha256,MAX(canonical_genome) AS canonical_genome,
         COUNT(*) AS dataset_count,
         AVG(CASE WHEN dataset_size=1 THEN 1.0
                  ELSE 1.0*(ascending_rank-1)/(dataset_size-1) END) AS mean_percentile,
         MIN(CASE WHEN dataset_size=1 THEN 1.0
                  ELSE 1.0*(ascending_rank-1)/(dataset_size-1) END) AS worst_percentile,
         SUM(CASE WHEN winner_rank=1 THEN 1 ELSE 0 END) AS strict_wins
  FROM ranked GROUP BY semantic_sha256 HAVING COUNT(*)>=2
)
SELECT * FROM aggregated
ORDER BY mean_percentile DESC,worst_percentile DESC,
         strict_wins DESC,dataset_count DESC,semantic_sha256
LIMIT ?3
"#;
    let mut statement = connection.prepare(repeat_sql)?;
    let repeat_winners = statement
        .query_map(
            params![task, protocol_sha256, limit as i64],
            sqlite_row_json,
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let mut origin_statement = connection.prepare(
        "SELECT dataset_hash,origin_class,MAX(score) AS score
         FROM observations INDEXED BY observations_origin_protocol_lift_v2
         WHERE task=?1 AND protocol_sha256=?2
           AND (?3 IS NULL OR dataset_hash=?3)
         GROUP BY dataset_hash,origin_class ORDER BY dataset_hash,origin_class",
    )?;
    let mut origins = BTreeMap::<String, BTreeMap<String, f64>>::new();
    for row in origin_statement.query_map(params![task, protocol_sha256, dataset], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)?,
        ))
    })? {
        let (dataset_hash, origin, score) = row?;
        origins
            .entry(dataset_hash)
            .or_default()
            .insert(origin, score);
    }
    let known = known_rows(root)?;
    let mut lift = Vec::new();
    for (dataset_hash, scores) in origins.into_iter().take(limit) {
        let dl_parent = scores.get("dl_origin").copied();
        let dl_descendant = scores.get("dl_descendant").copied();
        let dl_contribution = dl_parent
            .into_iter()
            .chain(dl_descendant)
            .max_by(f64::total_cmp);
        let gp = scores.get("gp_origin").copied();
        let incumbent = known
            .get(&(task.to_owned(), dataset_hash.clone()))
            .and_then(|value| value.get("score"))
            .and_then(Value::as_f64);
        lift.push(json!({
            "dataset_hash": dataset_hash,
            "dl_origin_parent_best": dl_parent,
            "dl_origin_descendant_best": dl_descendant,
            "gp_origin_best": gp,
            "incumbent": incumbent,
            "dl_lift_vs_gp": dl_contribution.zip(gp).map(|(dl, gp)| dl - gp),
            "dl_lift_vs_incumbent": dl_contribution.zip(incumbent).map(
                |(dl, incumbent)| dl - incumbent
            ),
        }));
    }

    let dataset_plan = explain_details(
        &connection,
        "SELECT semantic_sha256 FROM observations
         WHERE task=?1 AND protocol_sha256=?2 AND dataset_hash=?3
         ORDER BY score DESC,semantic_sha256",
        &[&task, &protocol_sha256, &dataset.unwrap_or("")],
    )?;
    let semantic_plan = explain_details(
        &connection,
        "SELECT dataset_hash FROM observations
         WHERE task=?1 AND protocol_sha256=?2 AND semantic_sha256=?3
         ORDER BY score DESC,dataset_hash",
        &[&task, &protocol_sha256, &semantic.unwrap_or("")],
    )?;
    Ok(json!({
        "schema": "JopeEvidenceQueryReport.v1",
        "task": task,
        "protocol_sha256": protocol_sha256,
        "filters": {"dataset_hash": dataset, "semantic_sha256": semantic},
        "best_genome_per_dataset": best_per_dataset,
        "best_datasets_per_semantic": best_datasets_per_semantic,
        "repeat_winners": repeat_winners,
        "dl_origin_parent_descendant_lift": lift,
        "query_plans": {
            "dataset_best": dataset_plan,
            "semantic_datasets": semantic_plan,
        },
        "indexes": {
            "corpus": [
                "observations_dataset_protocol_best_v2",
                "observations_semantic_protocol_best_v2",
                "observations_origin_protocol_lift_v2",
                "observations_checkpoint_protocol_lift_v2"
            ],
            "pairs": pair_indexes,
        },
    }))
}

pub fn rebuild_indexes(root: &Path) -> Result<Value> {
    let corpus_path = root.join("corpus/strict-test-exact.sqlite");
    let corpus = open_corpus(&corpus_path)?;
    corpus.execute(
        "CREATE INDEX IF NOT EXISTS observations_best_v1
         ON observations(task,dataset_hash,score DESC,semantic_sha256 ASC)",
        [],
    )?;
    let known = known_rows(root)?;
    let best = best_rows(&corpus)?;
    let mut champions = Vec::new();
    let mut wins = Vec::new();
    for new in best {
        let task = required_str(&new, "task")?;
        let dataset = required_str(&new, "dataset_hash")?;
        let old = known.get(&(task.to_owned(), dataset.to_owned()));
        let new_score = finite_f64(&new, "score")?;
        let old_score = old.and_then(|row| row.get("score")).and_then(Value::as_f64);
        let improves = old_score.is_none_or(|score| new_score > score);
        champions.push(json!({
            "task": task,
            "dataset_hash": dataset,
            "recorded_incumbent": old,
            "best_new_strict": new,
            "strictly_improves": improves,
            "promotion_status": if improves {
                "pending_fitted_cpp_parity"
            } else {
                "incumbent_retained"
            },
        }));
        if improves {
            wins.push(json!({
                "task": task,
                "dataset_hash": dataset,
                "old_score": old_score,
                "new_score": new_score,
                "delta": old_score.map(|score| new_score - score),
                "old_genome": old.and_then(|row| row.get("canonical_genome")),
                "old_semantic_sha256": old.and_then(|row| row.get("semantic_sha256")),
                "old_receipt_sha256": old.and_then(|row| row.get("receipt_ref")),
                "new_genome": new["canonical_genome"],
                "semantic_sha256": new["semantic_sha256"],
                "receipt_sha256": new["receipt_sha256"],
                "requires_fitted_cpp_confirmation": true,
                "model_checkpoint_sha256": new["model_checkpoint_sha256"],
                "checkpoint_origin": new["candidate_source"],
                "evidence_lane": "adaptive",
                "sealed": false,
            }));
        }
    }
    let observations: i64 =
        corpus.query_row("SELECT COUNT(*) FROM observations", [], |row| row.get(0))?;
    let failures: i64 = corpus.query_row("SELECT COUNT(*) FROM failures", [], |row| row.get(0))?;
    let hall = json!({
        "schema": "JopeTestOnlyStrictHallOfFame.v1",
        "observations": observations,
        "failures": failures,
        "champions": champions,
        "strict_wins": wins,
    });
    atomic_json(&root.join("indexes/champions.json"), &hall["champions"])?;
    atomic_json(&root.join("indexes/strict-wins.json"), &hall["strict_wins"])?;
    atomic_json(&root.join("indexes/hall-of-fame.json"), &hall)?;
    Ok(json!({
        "exact_rows_total": observations,
        "evaluator_failures_total": failures,
        "strict_wins": hall["strict_wins"].as_array().map_or(0, Vec::len),
    }))
}

pub fn export_gene_evidence(root: &Path, output: &Path) -> Result<Value> {
    let corpus = open_corpus(&root.join("corpus/strict-test-exact.sqlite"))?;
    let known = known_rows(root)?;
    let best = best_rows(&corpus)?;
    let mut evidence = Vec::with_capacity(best.len());
    for row in best {
        let task = required_str(&row, "task")?;
        let dataset = required_str(&row, "dataset_hash")?;
        let score = finite_f64(&row, "score")?;
        let incumbent = known
            .get(&(task.to_owned(), dataset.to_owned()))
            .and_then(|value| value.get("score"))
            .and_then(Value::as_f64);
        let strict = incumbent.is_some_and(|prior| score > prior);
        evidence.push(json!({
            "schema": "JopeGeneEvidence.v2",
            "task": task,
            "dataset_hash": dataset,
            "semantic_sha256": row["semantic_sha256"],
            "protocol_sha256": row["protocol_sha256"],
            "score": score,
            "metric": row["metric"],
            "secondary_metric": row["secondary_metric"],
            "secondary_score": row["secondary_score"],
            "within_dataset_rank": 1,
            "within_dataset_percentile": 1.0,
            "incumbent_delta": incumbent.map(|prior| score - prior),
            "canonical_genome": row["canonical_genome"],
            "checkpoint_epoch": row["model_checkpoint_sha256"].as_str().map(
                |digest| format!("checkpoint-{digest}")
            ),
            "model_checkpoint_sha256": row["model_checkpoint_sha256"],
            "proposal_slot": Value::Null,
            "candidate_source": row["candidate_source"],
            "receipt_sha256": row["receipt_sha256"],
            "classification": "adaptive",
            "test_adaptive": true,
            "benchmark_valid": false,
            "validity": "valid",
            "qualification_state": if strict { "discovered" } else { "superseded" },
            "compiled_parity_state": if strict {
                "pending"
            } else {
                "not_required"
            },
        }));
    }
    evidence.sort_by(|left, right| {
        (
            left["task"].as_str().unwrap_or_default(),
            left["dataset_hash"].as_str().unwrap_or_default(),
            left["semantic_sha256"].as_str().unwrap_or_default(),
        )
            .cmp(&(
                right["task"].as_str().unwrap_or_default(),
                right["dataset_hash"].as_str().unwrap_or_default(),
                right["semantic_sha256"].as_str().unwrap_or_default(),
            ))
    });
    atomic_jsonl(output, &evidence)?;
    Ok(json!({
        "schema": "JopeGeneEvidenceExport.v2",
        "output": output,
        "output_sha256": sha256_file(output)?,
        "rows": evidence.len(),
    }))
}

fn optional_json(path: &Path) -> Result<Value> {
    if path.is_file() {
        read_json(path)
    } else {
        Ok(json!({}))
    }
}

fn service_stale(state: &Value, now: i64) -> bool {
    let mut reference = state
        .get("last_success_at_ms")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    if state.get("lifecycle").and_then(Value::as_str) == Some("starting") {
        reference = reference.max(
            state
                .get("process_started_at_ms")
                .and_then(Value::as_i64)
                .unwrap_or(0),
        );
    }
    reference == 0 || now.saturating_sub(reference) > 5 * 60_000
}

fn pair_registry_status(path: &Path) -> Result<Value> {
    if !path.is_file() {
        return Ok(json!({}));
    }
    let connection = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    if !table_exists(&connection, "pairs")? {
        return Ok(json!({}));
    }
    let mut states = Map::new();
    let mut statement =
        connection.prepare("SELECT state,COUNT(*) FROM pairs GROUP BY state ORDER BY state")?;
    for row in statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })? {
        let (state, count) = row?;
        states.insert(state, json!(count));
    }
    let attempts = if table_exists(&connection, "attempts")? {
        connection.query_row("SELECT COUNT(*) FROM attempts", [], |row| {
            row.get::<_, i64>(0)
        })?
    } else {
        0
    };
    let total = states.values().filter_map(Value::as_i64).sum::<i64>();
    Ok(json!({"total": total, "states": states, "attempts": attempts}))
}

fn sealed_status(path: &Path) -> Result<Value> {
    if !path.is_file() {
        return Ok(json!({
            "schema": "JopeSealedAuditStatus.v1",
            "eligible_unseen_datasets": 0,
            "zero_shot_evidence": "unavailable",
            "states": {"registered": 0, "emitted": 0, "scored": 0},
            "strict_super_human": 0,
            "isolated_from_training": true,
        }));
    }
    let connection = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    let mut counts = BTreeMap::new();
    for state in ["registered", "emitted", "scored"] {
        let count: i64 = connection.query_row(
            "SELECT COUNT(*) FROM datasets WHERE state=?",
            [state],
            |row| row.get(0),
        )?;
        counts.insert(state, count);
    }
    let total: i64 = counts.values().sum();
    let super_human: i64 = connection.query_row(
        "SELECT COUNT(*) FROM datasets WHERE state='scored'
         AND score>incumbent_score AND compiled_interpreted_parity=1
         AND prediction_parity=1",
        [],
        |row| row.get(0),
    )?;
    Ok(json!({
        "schema": "JopeSealedAuditStatus.v1",
        "checkpoint_sha256": Value::Null,
        "eligible_unseen_datasets": total,
        "zero_shot_evidence": if total > 0 { "available" } else { "unavailable" },
        "states": counts,
        "strict_super_human": super_human,
        "isolated_from_training": true,
    }))
}

fn active_catalog(root: &Path) -> Result<Value> {
    let pointer_path = root.join("catalog/current.json");
    if !pointer_path.is_file() {
        return Ok(json!({}));
    }
    let pointer = read_json(&pointer_path)?;
    let catalog_path = PathBuf::from(required_str(&pointer, "catalog")?);
    if sha256_file(&catalog_path)? != required_str(&pointer, "catalog_sha256")? {
        bail!("catalog pointer hash drift");
    }
    let catalog = read_json(&catalog_path)?;
    Ok(json!({
        "schema": catalog["schema"],
        "catalog": catalog_path,
        "catalog_sha256": pointer["catalog_sha256"],
        "row_count": catalog["entries"].as_array().map_or(0, Vec::len),
        "task_counts": catalog["task_counts"],
        "source_family_disjoint": catalog["source_family_disjoint"],
    }))
}

fn loaded_bundle(root: &Path) -> Result<Value> {
    if let Ok(digest) = env::var("JOPE_LOADED_BUNDLE_SHA256") {
        return Ok(Value::String(digest));
    }
    let path = root.join("runtime-bundles/active.json");
    if !path.is_file() {
        return Ok(Value::Null);
    }
    let pointer = read_json(&path)?;
    let manifest = PathBuf::from(required_str(&pointer, "bundle")?);
    if sha256_file(&manifest)? != required_str(&pointer, "bundle_manifest_sha256")? {
        bail!("active runtime bundle pointer failed verification");
    }
    Ok(pointer.get("bundle_sha256").cloned().unwrap_or(Value::Null))
}

fn newest_cohort(root: &Path) -> Result<Value> {
    let pending = root.join("waves/pending");
    if !pending.is_dir() {
        return Ok(Value::Null);
    }
    let mut paths = fs::read_dir(pending)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".cohort.json"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    if let Some(path) = paths.last() {
        let mut value = read_json(path)?;
        if let Some(object) = value.as_object_mut() {
            object.insert("path".to_owned(), json!(path));
        }
        Ok(value)
    } else {
        Ok(Value::Null)
    }
}

/// Count the published caches under an expanded-encoding root.
///
/// A cache is published by renaming its staging directory into place, and that
/// staging directory carries the manifest *before* the rename (see
/// `selfplay_expansion::materialize_cache`). So a worker killed in that window
/// leaves behind a directory that has a manifest but was never published, and a
/// manifest-only test would count it as complete. Staging names are dot-prefixed
/// by construction and a published cache never is, so the name is the authority
/// on whether a directory was published.
fn completed_cache_count(root: &Path) -> Result<usize> {
    if !root.is_dir() {
        return Ok(0);
    }
    Ok(fs::read_dir(root)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| !name.starts_with('.'))
        })
        .filter(|entry| entry.path().join("cache-manifest.v1.json").is_file())
        .count())
}

fn expanded_cache_progress(catalog: &Value) -> Result<Value> {
    let counts = catalog.get("task_counts").cloned().unwrap_or(json!({}));
    let mut result = Map::new();
    for task in crate::selfplay_expansion::TASKS {
        let path = PathBuf::from(format!(
            "{}/{task}-expanded",
            crate::paths::flywheel_encodings_root().display()
        ));
        let completed = completed_cache_count(&path)?;
        result.insert(
            task.to_owned(),
            json!({
                "completed": completed,
                "active_catalog": counts.get(task).and_then(Value::as_u64).unwrap_or(0),
            }),
        );
    }
    Ok(Value::Object(result))
}

fn compile_cache_status(root: &Path) -> Result<Value> {
    let mut manifests = Vec::new();
    collect_named(root, "artifact-manifest.json", &mut manifests)?;
    let mut valid = 0_usize;
    for manifest in &manifests {
        let Ok(value) = read_json(manifest) else {
            continue;
        };
        let Some(artifact) = value.get("artifact").and_then(Value::as_str) else {
            continue;
        };
        if manifest.parent().unwrap_or(root).join(artifact).is_file() {
            valid += 1;
        }
    }
    let operations = optional_json(&root.join("operations.json"))?;
    let hits = operations.get("hits").and_then(Value::as_u64).unwrap_or(0);
    let misses = operations
        .get("misses")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Ok(json!({
        "entries": manifests.len(),
        "valid_entries": valid,
        "invalid_entries": manifests.len() - valid,
        "hits": hits,
        "misses": misses,
        "hit_rate": if hits + misses == 0 {
            Value::Null
        } else {
            json!(hits as f64 / (hits + misses) as f64)
        },
    }))
}

#[derive(Default)]
struct TelemetryCatalog {
    digest: Option<String>,
    entries: BTreeMap<Task, u64>,
    families: BTreeMap<Task, BTreeSet<String>>,
    family_by_context: BTreeMap<(Task, String), String>,
    sources: Vec<SourceBinding>,
}

#[derive(Default)]
struct TelemetryDatasetState {
    encoded: u64,
    active: u64,
    observed: u64,
}

#[derive(Default)]
struct TelemetryCorpusTask {
    valid: u64,
    failed: u64,
    latest_observation_at_ms: Option<u64>,
    valid_last_minute: u64,
    failures_last_minute: u64,
    primary: Option<Distribution>,
    utility: Option<Distribution>,
    loss: Option<Distribution>,
    native_errors: BTreeMap<String, Distribution>,
}

fn file_timestamp_ms(path: &Path) -> Result<u128> {
    Ok(fs::metadata(path)?
        .modified()?
        .duration_since(UNIX_EPOCH)
        .context("file timestamp predates Unix epoch")?
        .as_millis())
}

fn telemetry_source(
    path: &Path,
    schema: &str,
    freshness_budget_seconds: u64,
) -> Result<SourceBinding> {
    Ok(SourceBinding {
        path: path.to_string_lossy().into_owned(),
        schema: schema.to_owned(),
        sha256: Sha256Digest::parse(sha256_file(path)?)?,
        timestamp_ms: file_timestamp_ms(path)?,
        freshness_budget_seconds,
    })
}

fn telemetry_catalog(root: &Path) -> Result<TelemetryCatalog> {
    let pointer_path = root.join("catalog/current.json");
    if !pointer_path.is_file() {
        return Ok(TelemetryCatalog::default());
    }
    let pointer = read_json(&pointer_path)?;
    let catalog_path = PathBuf::from(required_str(&pointer, "catalog")?);
    let digest = sha256_file(&catalog_path)?;
    ensure!(
        pointer.get("catalog_sha256").and_then(Value::as_str) == Some(digest.as_str()),
        "telemetry catalog pointer drift"
    );
    let catalog = read_json(&catalog_path)?;
    let schema = catalog
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let mut output = TelemetryCatalog {
        digest: Some(digest),
        sources: vec![
            telemetry_source(&pointer_path, "JopeTestOnlyContextPointer.v1", 45)?,
            telemetry_source(&catalog_path, schema, 45)?,
        ],
        ..TelemetryCatalog::default()
    };
    for row in catalog
        .get("entries")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(task) = row
            .get("task")
            .and_then(Value::as_str)
            .and_then(|task| task.parse::<Task>().ok())
        else {
            continue;
        };
        let Some(context) = row.get("dataset_hash").and_then(Value::as_str) else {
            continue;
        };
        let Some(family) = row.get("source_family").and_then(Value::as_str) else {
            continue;
        };
        *output.entries.entry(task).or_default() += 1;
        output
            .families
            .entry(task)
            .or_default()
            .insert(family.to_owned());
        output
            .family_by_context
            .insert((task, context.to_owned()), family.to_owned());
    }
    Ok(output)
}

fn telemetry_dataset_state(root: &Path) -> Result<BTreeMap<Task, TelemetryDatasetState>> {
    let path = root.join("ops/dataset-state.sqlite");
    if !path.is_file() {
        return Ok(BTreeMap::new());
    }
    let connection = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(std::time::Duration::from_millis(900))?;
    connection.pragma_update(None, "query_only", "ON")?;
    let mut statement = connection.prepare(
        "SELECT task, COALESCE(SUM(sidecars_complete),0),
                COALESCE(SUM(catalog_active),0),
                COALESCE(SUM(CASE WHEN catalog_active=1 AND obs_count>0 THEN 1 ELSE 0 END),0)
           FROM datasets GROUP BY task",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            TelemetryDatasetState {
                encoded: row.get::<_, i64>(1)?.max(0) as u64,
                active: row.get::<_, i64>(2)?.max(0) as u64,
                observed: row.get::<_, i64>(3)?.max(0) as u64,
            },
        ))
    })?;
    let mut output = BTreeMap::new();
    for row in rows {
        let (task, state) = row?;
        if let Ok(task) = task.parse::<Task>() {
            output.insert(task, state);
        }
    }
    Ok(output)
}

fn family_distribution(
    rows: impl IntoIterator<Item = (String, f64)>,
    task: Task,
    family_by_context: &BTreeMap<(Task, String), String>,
    transform: impl Fn(f64) -> Result<f64>,
) -> Result<Option<Distribution>> {
    let mut by_family = BTreeMap::<String, Vec<f64>>::new();
    for (context, value) in rows {
        let Some(family) = family_by_context.get(&(task, context)) else {
            continue;
        };
        by_family
            .entry(family.clone())
            .or_default()
            .push(transform(value)?);
    }
    let values = by_family
        .into_values()
        .filter(|values| !values.is_empty())
        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
        .collect::<Vec<_>>();
    if values.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Distribution::from_values(values)?))
    }
}

fn query_context_averages(
    connection: &Connection,
    task: Task,
    expression: &str,
    predicate: &str,
) -> Result<Vec<(String, f64)>> {
    let sql = format!(
        "SELECT dataset_hash, AVG({expression}) FROM observations
          WHERE task=?1 AND ({predicate}) GROUP BY dataset_hash ORDER BY dataset_hash"
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement
        .query_map([task.as_str()], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn telemetry_corpus(
    root: &Path,
    catalog: &TelemetryCatalog,
    now: i64,
) -> Result<BTreeMap<Task, TelemetryCorpusTask>> {
    let path = root.join("corpus/strict-test-exact.sqlite");
    if !path.is_file() {
        return Ok(BTreeMap::new());
    }
    let connection = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(std::time::Duration::from_millis(900))?;
    connection.pragma_update(None, "query_only", "ON")?;
    let cutoff = now.saturating_sub(60_000);
    let mut output = BTreeMap::new();
    for task in FOUR_TASK_ORDER {
        let (valid, latest, recent): (u64, Option<i64>, u64) = connection.query_row(
            "SELECT COUNT(*), MAX(created_at_ms),
                    COALESCE(SUM(CASE WHEN created_at_ms>=?2 THEN 1 ELSE 0 END),0)
               FROM observations WHERE task=?1",
            params![task.as_str(), cutoff],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        let (failed, recent_failed): (u64, u64) = connection.query_row(
            "SELECT COUNT(*),
                    COALESCE(SUM(CASE WHEN created_at_ms>=?2 THEN 1 ELSE 0 END),0)
               FROM failures WHERE task=?1",
            params![task.as_str(), cutoff],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let expression = if task == Task::Regression {
            "CASE WHEN score < -1.0 THEN -1.0 WHEN score > 1.0 THEN 1.0 ELSE score END"
        } else {
            "score"
        };
        let primary_rows = query_context_averages(&connection, task, expression, "1=1")?;
        let display = family_distribution(
            primary_rows.clone(),
            task,
            &catalog.family_by_context,
            |score| {
                ensure!(score.is_finite(), "primary score is non-finite");
                Ok(if task == Task::Timeseries {
                    -score
                } else {
                    score
                })
            },
        )?;
        let utility = family_distribution(
            primary_rows.clone(),
            task,
            &catalog.family_by_context,
            |score| bounded_task_utility(task, score),
        )?;
        let loss = family_distribution(primary_rows, task, &catalog.family_by_context, |score| {
            operational_loss(task, score)
        })?;
        let mut native_errors = BTreeMap::new();
        if task == Task::Timeseries {
            if let Some(distribution) = display.clone() {
                native_errors.insert("mase".to_owned(), distribution);
            }
        }
        let secondary_names: &[&str] = match task {
            Task::Regression => &[],
            Task::Binary => &["log_loss", "balanced_accuracy"],
            Task::Multiclass => &["accuracy", "one_minus_accuracy"],
            Task::Timeseries => &["rmse", "mae", "smape", "wape"],
        };
        for name in secondary_names {
            let rows = if *name == "balanced_accuracy" {
                query_context_averages(
                    &connection,
                    task,
                    "balanced_accuracy",
                    "balanced_accuracy IS NOT NULL",
                )?
            } else {
                query_context_averages(
                    &connection,
                    task,
                    "secondary_score",
                    &format!(
                        "secondary_metric='{}' AND secondary_score IS NOT NULL",
                        name
                    ),
                )?
            };
            if let Some(distribution) =
                family_distribution(rows, task, &catalog.family_by_context, |value| {
                    ensure!(value.is_finite(), "native error is non-finite");
                    Ok(value)
                })?
            {
                native_errors.insert((*name).to_owned(), distribution);
            }
        }
        output.insert(
            task,
            TelemetryCorpusTask {
                valid,
                failed,
                latest_observation_at_ms: latest.map(|value| value.max(0) as u64),
                valid_last_minute: recent,
                failures_last_minute: recent_failed,
                primary: display,
                utility,
                loss,
                native_errors,
            },
        );
    }
    Ok(output)
}

fn typed_distribution(value: Option<Distribution>, reason: &str) -> TypedDistribution {
    value.map_or_else(|| TypedDistribution::na(reason), TypedDistribution::value)
}

fn typed_checkpoint(value: Option<&Value>, field: &str, reason: &str) -> TypedText {
    value
        .and_then(|value| value.get(field))
        .and_then(Value::as_str)
        .map(|value| TypedText::value(value.to_owned()))
        .unwrap_or_else(|| TypedText::na(reason))
}

fn latest_rating(
    root: &Path,
    task: Task,
    kind: RatingKindV1,
) -> Result<(TypedRating, Vec<SourceBinding>)> {
    let kind_name = match kind {
        RatingKindV1::AdaptiveSearch => "adaptive",
        RatingKindV1::DiagnosticAnchor => "diagnostic",
    };
    let freshness_budget = match kind {
        RatingKindV1::AdaptiveSearch => 300,
        RatingKindV1::DiagnosticAnchor => u64::MAX,
    };
    let path = root
        .join("ops/ratings")
        .join(task.as_str())
        .join(format!("{kind_name}-current.v1.json"));
    if !path.is_file() {
        return Ok((
            TypedRating::na(format!("no {kind_name} rating")),
            Vec::new(),
        ));
    }
    let current = read_json(&path)?;
    let (snapshot_path, snapshot_value, mut sources) =
        if current.get("schema").and_then(Value::as_str) == Some("JopeTaskRatingPointer.v1") {
            ensure!(
                current.get("task").and_then(Value::as_str) == Some(task.as_str())
                    && current.get("promotion_eligible").and_then(Value::as_bool) == Some(false),
                "{task} {kind_name} rating pointer scope drift"
            );
            let declared = PathBuf::from(required_str(&current, "snapshot")?);
            let snapshot_path = if declared.is_absolute() {
                declared
            } else {
                path.parent().unwrap_or(root).join(declared)
            };
            ensure!(
                sha256_file(&snapshot_path)? == required_str(&current, "snapshot_sha256")?,
                "{task} {kind_name} rating pointer hash drift"
            );
            let sources = vec![telemetry_source(
                &path,
                "JopeTaskRatingPointer.v1",
                freshness_budget,
            )?];
            let value = read_json(&snapshot_path)?;
            (snapshot_path, value, sources)
        } else {
            (path.clone(), current, Vec::new())
        };
    let value: JopeTaskRatingSnapshotV1 = serde_json::from_value(snapshot_value)?;
    value.validate()?;
    ensure!(
        value.task == task && value.rating_kind == kind && !value.promotion_eligible,
        "{task} {kind_name} rating contract drift"
    );
    let digest = Sha256Digest::parse(sha256_file(&path)?)?;
    let player = value
        .players
        .iter()
        .find(|player| player.player == value.display_player)
        .context("rating display player is absent after validation")?;
    sources.push(telemetry_source(
        &snapshot_path,
        "JopeTaskRatingSnapshot.v1",
        freshness_budget,
    )?);
    Ok((
        TypedRating::value(RatingTelemetry {
            snapshot_sha256: digest,
            player: player.player.clone(),
            rating: player.rating,
            interval_95: player.rating_interval_95,
            delta_from_anchor: player.rating - 1500.0,
            promotion_eligible: false,
        }),
        sources,
    ))
}

/// Rebuild and atomically publish the four-task telemetry authority from
/// read-only snapshots. Missing source evidence becomes typed N/A, never zero.
pub fn four_task_telemetry(root: &Path) -> Result<JopeFourTaskTelemetryV1> {
    four_task_telemetry_at(root, now_ms())
}

/// [`four_task_telemetry`] at one explicit wall-clock instant (Unix milliseconds).
/// Every age, freshness verdict and per-minute window in the snapshot is measured
/// from `now`, so the rebuild is a function of the snapshots and this instant.
fn four_task_telemetry_at(root: &Path, now: i64) -> Result<JopeFourTaskTelemetryV1> {
    let catalog = telemetry_catalog(root)?;
    let dataset_state = telemetry_dataset_state(root)?;
    let corpus = telemetry_corpus(root, &catalog, now)?;
    let accepted_path = root.join("training/accepted-by-task.json");
    let accepted = optional_json(&accepted_path)?;
    let accepted_tasks = accepted.get("tasks").and_then(Value::as_object);
    let controller_path = root.join("ops/task-learning-controller.v2.json");
    let controller = optional_json(&controller_path)?;
    let shadows = controller.get("shadow_models").and_then(Value::as_object);
    let candidates = controller
        .get("active_candidates")
        .and_then(Value::as_object);
    let corpus_pointer_path = root.join("training/current-corpus.json");
    let corpus_pointer = optional_json(&corpus_pointer_path)?;
    let corpus_watermark = corpus_pointer
        .get("watermark")
        .and_then(Value::as_u64)
        .map_or_else(
            || TypedCount::na("current corpus watermark unavailable"),
            TypedCount::value,
        );
    let mut shared_sources = catalog.sources.clone();
    if accepted_path.is_file() {
        shared_sources.push(telemetry_source(
            &accepted_path,
            accepted
                .get("schema")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            45,
        )?);
    }
    if controller_path.is_file() {
        shared_sources.push(telemetry_source(
            &controller_path,
            controller
                .get("schema")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            120,
        )?);
    }
    if corpus_pointer_path.is_file() {
        shared_sources.push(telemetry_source(
            &corpus_pointer_path,
            corpus_pointer
                .get("schema")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            60,
        )?);
    }
    let mut rows = Vec::with_capacity(4);
    for task in FOUR_TASK_ORDER {
        let catalog_entries = catalog.entries.get(&task).copied();
        let family_count = catalog.families.get(&task).map(BTreeSet::len);
        let state = dataset_state.get(&task);
        let evidence = corpus.get(&task);
        let no_observations = evidence.is_none_or(|row| row.valid == 0);
        let has_evaluation_evidence =
            evidence.is_some_and(|row| row.valid.saturating_add(row.failed) > 0);
        let missing_evidence_reason = if no_observations {
            "no observations"
        } else {
            "metric unavailable"
        };
        let eligibility_path = root
            .join("training/task-training-eligibility")
            .join(format!("{}.v1.json", task.as_str()));
        let eligibility = optional_json(&eligibility_path)?;
        let eligibility_valid = eligibility.get("schema").and_then(Value::as_str)
            == Some("JopeTaskTrainingEligibility.v1")
            && eligibility.get("task").and_then(Value::as_str) == Some(task.as_str())
            && eligibility.get("catalog_sha256").and_then(Value::as_str)
                == catalog.digest.as_deref();
        let evidence_eligible = eligibility_valid
            && eligibility
                .get("evidence_lane_enabled")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let training_eligible = eligibility_valid
            && eligibility
                .get("training_eligible")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let promotion_eligible = eligibility_valid
            && eligibility
                .get("promotion_eligible")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let mut blockers = Vec::new();
        if matches!(task, Task::Multiclass | Task::Timeseries) {
            match (family_count, state) {
                (Some(count), Some(state)) if count < 500 => blockers.push(format!(
                    "N/A — {count}/500 catalog families; {} observed datasets",
                    state.observed
                )),
                (Some(count), None) if count < 500 => blockers.push(format!(
                    "N/A — {count}/500 catalog families; observed datasets unavailable"
                )),
                (None, _) => blockers.push("N/A — catalog families unavailable".to_owned()),
                _ => {}
            }
        }
        if !eligibility_valid {
            blockers.push("missing_or_stale_catalog_bound_eligibility".to_owned());
        }
        if !training_eligible {
            blockers.push(
                eligibility
                    .get("advisory_reason")
                    .and_then(Value::as_str)
                    .unwrap_or("training_ineligible")
                    .to_owned(),
            );
        }
        if !promotion_eligible {
            blockers.push("promotion_ineligible".to_owned());
        }
        let (adaptive_rating, adaptive_source) =
            latest_rating(root, task, RatingKindV1::AdaptiveSearch)?;
        let (diagnostic_rating, diagnostic_source) =
            latest_rating(root, task, RatingKindV1::DiagnosticAnchor)?;
        let mut sources = shared_sources.clone();
        if eligibility_path.is_file() {
            sources.push(telemetry_source(
                &eligibility_path,
                "JopeTaskTrainingEligibility.v1",
                60,
            )?);
        }
        sources.extend(adaptive_source);
        sources.extend(diagnostic_source);
        let accepted_row = accepted_tasks.and_then(|tasks| tasks.get(task.as_str()));
        let shadow_row = shadows.and_then(|tasks| tasks.get(task.as_str()));
        let candidate_row = candidates.and_then(|tasks| tasks.get(task.as_str()));
        let trainer_metrics = shadow_row
            .and_then(|row| row.get("validation_metrics"))
            .and_then(Value::as_object)
            .map(|metrics| {
                metrics
                    .iter()
                    .map(|(name, value)| {
                        (
                            name.clone(),
                            value
                                .as_f64()
                                .filter(|value| value.is_finite())
                                .map_or_else(
                                    || TypedNumber::na("trainer metric unavailable"),
                                    TypedNumber::value,
                                ),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let trainer_updated = shadow_row
            .and_then(|row| row.get("updated_at_ms"))
            .and_then(Value::as_u64);
        let latest = evidence.and_then(|row| row.latest_observation_at_ms);
        let freshness_reason = latest.map_or_else(
            || "N/A — no observations".to_owned(),
            |at| {
                let age = (now.max(0) as u64).saturating_sub(at) / 1_000;
                if age <= 900 {
                    format!("fresh — last observation {age}s ago")
                } else {
                    format!("stale — last observation {age}s ago")
                }
            },
        );
        let native_errors = if task == Task::Regression {
            BTreeMap::from([(
                "rmse".to_owned(),
                TypedDistribution::na(
                    "per-context RMSE only; heterogeneous global aggregation is forbidden",
                ),
            )])
        } else if evidence.is_some_and(|row| !row.native_errors.is_empty()) {
            evidence
                .into_iter()
                .flat_map(|row| row.native_errors.iter())
                .map(|(name, distribution)| {
                    (name.clone(), TypedDistribution::value(distribution.clone()))
                })
                .collect()
        } else {
            BTreeMap::from([(
                crate::four_task::task_metric_contract(task)
                    .native_error
                    .to_owned(),
                TypedDistribution::na(missing_evidence_reason),
            )])
        };
        rows.push(FourTaskTelemetryRow {
            task,
            counts: TelemetryCounts {
                catalog_entries: catalog_entries
                    .map_or_else(|| TypedCount::na("catalog unavailable"), TypedCount::value),
                source_families: family_count.map_or_else(
                    || TypedCount::na("catalog families unavailable"),
                    |count| TypedCount::value(count as u64),
                ),
                encoded: state.map_or_else(
                    || TypedCount::na("dataset-state index unavailable"),
                    |state| TypedCount::value(state.encoded),
                ),
                active: state.map_or_else(
                    || TypedCount::na("dataset-state index unavailable"),
                    |state| TypedCount::value(state.active),
                ),
                observed: state.map_or_else(
                    || TypedCount::na("dataset-state index unavailable"),
                    |state| TypedCount::value(state.observed),
                ),
                queued: TypedCount::na("task-attributed scheduler queue unavailable"),
                running: TypedCount::na("task-attributed scheduler running set unavailable"),
                completed: evidence.filter(|_| has_evaluation_evidence).map_or_else(
                    || {
                        TypedCount::na(if evidence.is_some() {
                            "no observations"
                        } else {
                            "corpus unavailable"
                        })
                    },
                    |row| TypedCount::value(row.valid + row.failed),
                ),
                failed: evidence.filter(|_| has_evaluation_evidence).map_or_else(
                    || {
                        TypedCount::na(if evidence.is_some() {
                            "no observations"
                        } else {
                            "corpus unavailable"
                        })
                    },
                    |row| TypedCount::value(row.failed),
                ),
                quarantined: TypedCount::na("task quarantine index unavailable"),
            },
            latest_observation_at_ms: latest
                .map_or_else(|| TypedCount::na("no observations"), TypedCount::value),
            valid_observations_per_minute: evidence
                .filter(|_| has_evaluation_evidence)
                .map_or_else(
                    || TypedNumber::na("no observations"),
                    |row| TypedNumber::value(row.valid_last_minute as f64),
                ),
            evaluator_failures_per_minute: evidence
                .filter(|_| has_evaluation_evidence)
                .map_or_else(
                    || TypedNumber::na("no observations"),
                    |row| TypedNumber::value(row.failures_last_minute as f64),
                ),
            freshness_reason,
            metrics: TelemetryMetrics {
                primary_metric: crate::four_task::task_metric_contract(task)
                    .primary_metric
                    .to_owned(),
                primary_performance: typed_distribution(
                    evidence.and_then(|row| row.primary.clone()),
                    missing_evidence_reason,
                ),
                native_errors,
                bounded_utility: typed_distribution(
                    evidence.and_then(|row| row.utility.clone()),
                    missing_evidence_reason,
                ),
                operational_loss: typed_distribution(
                    evidence.and_then(|row| row.loss.clone()),
                    missing_evidence_reason,
                ),
            },
            accepted_checkpoint: typed_checkpoint(
                accepted_row,
                "checkpoint_sha256",
                "no accepted checkpoint",
            ),
            bootstrap_anchor_state: accepted_row
                .and_then(|row| row.get("accepted"))
                .and_then(Value::as_bool)
                .map_or_else(
                    || "absent_unaccepted_bootstrap_required".to_owned(),
                    |accepted| {
                        if accepted {
                            "accepted".to_owned()
                        } else {
                            "bootstrap_unaccepted".to_owned()
                        }
                    },
                ),
            latest_shadow: typed_checkpoint(
                shadow_row,
                "checkpoint_sha256",
                "no shadow checkpoint",
            ),
            active_candidate: typed_checkpoint(
                candidate_row,
                "checkpoint_sha256",
                "no active candidate",
            ),
            eligibility: TelemetryEligibility {
                evidence_eligible,
                training_eligible,
                promotion_eligible,
                blockers,
            },
            adaptive_rating,
            diagnostic_rating,
            sealed_promotion: TypedText::na("no terminal sealed promotion receipt"),
            trainer_validation: TrainerValidationTelemetry {
                metrics: trainer_metrics,
                age_seconds: trainer_updated.map_or_else(
                    || TypedCount::na("no trainer validation"),
                    |updated| {
                        TypedCount::value((now.max(0) as u64).saturating_sub(updated) / 1_000)
                    },
                ),
            },
            sources,
        });
    }
    let telemetry = JopeFourTaskTelemetryV1 {
        schema: FOUR_TASK_TELEMETRY_SCHEMA.to_owned(),
        generated_at_ms: now.max(0) as u128,
        corpus_watermark,
        rows,
    };
    telemetry.validate()?;
    atomic_json(
        &root.join("ops/four-task-telemetry.v1.json"),
        &serde_json::to_value(&telemetry)?,
    )?;
    Ok(telemetry)
}

fn task_status(
    root: &Path,
    corpus_watermark: i64,
    queue_fresh: bool,
    promotion: &Value,
) -> Result<Value> {
    let accepted = optional_json(&root.join("training/accepted-by-task.json"))?;
    let controller = optional_json(&root.join("ops/task-learning-controller.v2.json"))?;
    let accepted_tasks = accepted.get("tasks").cloned().unwrap_or_else(|| json!({}));
    let shadows = controller
        .get("shadow_models")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let learning = promotion
        .get("internal_learning")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut tasks = Map::new();
    for task in crate::selfplay_expansion::TASKS {
        let accepted_row = accepted_tasks.get(task).cloned().unwrap_or(Value::Null);
        let shadow_row = shadows.get(task).cloned().unwrap_or(Value::Null);
        let task_learning = learning
            .get("tasks")
            .and_then(|value| value.get(task))
            .cloned()
            .unwrap_or_else(|| json!({}));
        tasks.insert(
            task.to_owned(),
            json!({
                "accepted_checkpoint": accepted_row.get("checkpoint_sha256").cloned().unwrap_or(Value::Null),
                "latest_shadow": shadow_row.get("checkpoint_sha256").cloned().unwrap_or(Value::Null),
                "corpus_watermark": corpus_watermark,
                "queue_fresh": queue_fresh,
                "promotion_eligibility": task_learning.get("promotion_eligible").cloned().unwrap_or(Value::Null),
                "blocking_gate": task_learning.get("blocking_gate")
                    .or_else(|| task_learning.get("gate_reason"))
                    .cloned().unwrap_or(Value::Null),
                "last_verified_learning_delta": task_learning.get("last_verified_learning_delta").cloned().unwrap_or(Value::Null),
            }),
        );
    }
    Ok(Value::Object(tasks))
}

fn collect_named(root: &Path, filename: &str, output: &mut Vec<PathBuf>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            collect_named(&entry.path(), filename, output)?;
        } else if entry.file_name() == filename {
            output.push(entry.path());
        }
    }
    Ok(())
}

pub fn build_status(root: &Path) -> Result<Value> {
    let now = now_ms();
    let mut services = Map::new();
    let mut alerts = BTreeSet::new();
    let mut reported_bundles = BTreeSet::new();
    for service in SERVICES {
        let state = optional_json(&root.join(format!("ops/services/{service}.json")))?;
        if state.as_object().is_none_or(Map::is_empty) {
            alerts.insert(format!("{service}:no_status"));
        } else {
            if state.get("error").is_some_and(|error| !error.is_null()) {
                alerts.insert(format!("{service}:error"));
            }
            if service_stale(&state, now) {
                alerts.insert(format!("{service}:stale"));
            }
            if let Some(bundle) = state.get("loaded_bundle_sha256").and_then(Value::as_str) {
                reported_bundles.insert(bundle.to_owned());
            }
        }
        services.insert(service.to_owned(), state);
    }
    let queue_result = services
        .get("queue")
        .and_then(|value| value.get("result"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let reserve = services
        .get("reserve")
        .and_then(|value| value.get("result"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let useful_depth = queue_result
        .get("useful_depth_after")
        .or_else(|| queue_result.get("useful_depth"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let target = queue_result
        .get("queued_target")
        .and_then(Value::as_i64)
        .unwrap_or(256);
    let hard_cap = queue_result
        .get("queued_hard_cap")
        .and_then(Value::as_i64)
        .unwrap_or(512);
    if useful_depth < target {
        alerts.insert("queue:below_fresh_horizon".to_owned());
    }
    if useful_depth > hard_cap {
        alerts.insert("queue:above_hard_cap".to_owned());
    }
    if queue_result
        .get("maximum_queue_age_seconds")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        > 1_800.0
    {
        alerts.insert("queue:older_than_30_minutes".to_owned());
    }
    if reserve
        .get("ready_after")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        > target
    {
        alerts.insert("reserve:above_one_launch_ready_horizon".to_owned());
    }
    let loaded = loaded_bundle(root)?;
    if !reported_bundles.is_empty()
        && (reported_bundles.len() != 1
            || loaded
                .as_str()
                .is_none_or(|digest| !reported_bundles.contains(digest)))
    {
        alerts.insert("deployment:loaded_bundle_mismatch".to_owned());
    }
    let freshness = optional_json(&root.join("ops/freshness-status.json"))?;
    if freshness.as_object().is_none_or(Map::is_empty) {
        alerts.insert("freshness:no_status".to_owned());
    } else if freshness.get("healthy") != Some(&Value::Bool(true)) {
        alerts.insert("freshness:unhealthy".to_owned());
    }
    let catalog = active_catalog(root)?;
    let pair_status = pair_registry_status(&root.join("ops/pair-registry.sqlite"))?;
    let sealed = sealed_status(&root.join("sealed-audit/registry.sqlite"))?;
    let corpus_path = root.join("corpus/strict-test-exact.sqlite");
    let watermark = if corpus_path.is_file() {
        let corpus =
            Connection::open_with_flags(corpus_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        corpus.query_row("SELECT COUNT(*) FROM observations", [], |row| {
            row.get::<_, i64>(0)
        })?
    } else {
        0
    };
    let p95 = queue_result
        .get("p95_jobs_per_minute")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let horizon = if p95 > 0.0 {
        json!(useful_depth as f64 / p95)
    } else {
        Value::Null
    };
    let mut queue = queue_result.clone();
    let queue_object = queue
        .as_object_mut()
        .context("queue service result is not an object")?;
    queue_object.insert("queued_target".to_owned(), json!(target));
    queue_object.insert("queued_hard_cap".to_owned(), json!(hard_cap));
    queue_object.insert("useful_depth".to_owned(), json!(useful_depth));
    queue_object.insert("horizon_minutes".to_owned(), horizon);
    queue_object.insert(
        "running".to_owned(),
        queue_result
            .get("running_after")
            .or_else(|| queue_result.get("running"))
            .cloned()
            .unwrap_or(json!(0)),
    );
    queue_object.insert(
        "dispatching".to_owned(),
        queue_result
            .get("dispatching_after")
            .or_else(|| queue_result.get("dispatching"))
            .cloned()
            .unwrap_or(json!(0)),
    );
    queue_object.insert(
        "held".to_owned(),
        queue_result
            .get("held_after")
            .or_else(|| queue_result.get("held"))
            .cloned()
            .unwrap_or(json!(0)),
    );
    let flywheel = optional_json(&root.join("ops/flywheel-state.json"))?;
    let promotion = json!({
        "internal_learning": optional_json(&root.join("training/learning.json"))?,
        "milestone": optional_json(&root.join("training/milestone.json"))?,
    });
    let tasks = task_status(root, watermark, useful_depth > 0, &promotion)?;
    let four_task_telemetry = four_task_telemetry(root)?;
    let recovery = optional_json(&root.join("ops/recovery-report.v2.json"))?;
    let recovery_valid =
        recovery.get("schema").and_then(Value::as_str) == Some("JopeRecoveryReport.v2");
    let release_healthy = recovery_valid
        && recovery.get("healthy").and_then(Value::as_bool) == Some(true)
        && recovery.get("release_gate").and_then(Value::as_str) == Some("ready")
        && alerts.is_empty();
    let mut release_blockers = alerts.clone();
    if !recovery_valid {
        release_blockers.insert("missing_or_invalid_recovery_report_v2".to_owned());
    } else if !release_healthy {
        release_blockers.insert(
            recovery
                .get("release_gate")
                .and_then(Value::as_str)
                .map_or_else(
                    || "recovery_release_gate_unavailable".to_owned(),
                    |gate| format!("recovery_release_gate_{gate}"),
                ),
        );
    }
    let result = json!({
        "schema": "JopeSelfplayOperationalStatus.v3",
        "updated_at_ms": now,
        "queue": queue,
        "reserve": reserve,
        "ingestion": services.get("ingestion").and_then(
            |value| value.get("result")
        ).cloned().unwrap_or(Value::Null),
        "trainer": optional_json(&root.join("ops/trainer-status.json"))?,
        "evolution": {
            "tournament": flywheel["tournament"],
            "last_terminal": flywheel["last_terminal"],
            "last_tournament": flywheel["last_tournament"],
            "consecutive_rejections": flywheel.get("consecutive_rejections")
                .cloned().unwrap_or(json!(0)),
        },
        "promotion": promotion,
        "scq_watchdog": services.get("watchdog").and_then(
            |value| value.get("result")
        ).cloned().unwrap_or(Value::Null),
        "evaluator": {
            "active": optional_json(&root.join("evaluator/active.json"))?,
            "candidate": optional_json(&root.join("evaluator/candidate.json"))?,
        },
        "catalog": catalog,
        "pair_registry": pair_status,
        "loaded_bundle_sha256": loaded,
        "service_loaded_bundle_sha256s": reported_bundles,
        "sealed_audit": sealed,
        "freshness": freshness,
        "queue_epochs": freshness.get("queue_epochs").cloned().unwrap_or(json!([])),
        "stale_holds": freshness.get("stale_held").cloned().unwrap_or(json!(0)),
        "stale_retirements": pair_status.get("states")
            .and_then(|states| states.get("superseded_unstarted"))
            .cloned().unwrap_or(json!(0)),
        "active_micro_cohort": newest_cohort(root)?,
        "expanded_cache_progress": expanded_cache_progress(&catalog)?,
        "adaptive_fitness_watermark": watermark,
        "mcp": {
            "socket": root.join("ops/freshness.sock"),
            "healthy": root.join("ops/freshness.sock").exists()
                && !freshness.as_object().is_none_or(Map::is_empty),
            "protocol_version": "2025-06-18",
        },
        "node_slot_policy": queue_result.get("nodes").cloned().unwrap_or(json!([])),
        "compile_cache": compile_cache_status(&root.join("compiled-cache"))?,
        "checkpoint_to_first_job_latency_ms": freshness.get(
            "checkpoint_to_first_job_latency_ms"
        ).cloned().unwrap_or(Value::Null),
        "services": services,
        "tasks": tasks,
        "four_task_telemetry": four_task_telemetry,
        "release_gate": {
            "healthy": release_healthy,
            "blocking_gates": release_blockers,
            "report": "ops/recovery-report.v2.json",
        },
        "alerts": alerts,
    });
    atomic_json(&root.join("ops/status-v3.json"), &result)?;
    atomic_json(&root.join("ops/status.json"), &result)?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// Seeding curve: "the model seeds better every cycle", made measurable.
//
// One curve point per model_checkpoint_sha256 generation: strict-win counts by
// checkpoint_origin (DL share = the accepted_dl_checkpoint +
// current_checkpoint_dl_proposal families), champion counts, and the DL-origin
// observation share from the corpus. The corpus is only ever touched through
// the existing observations_checkpoint_v1 index with an incremental
// created_at_ms cursor persisted inside the output file — never a full scan.
// ---------------------------------------------------------------------------

pub const SEEDING_CURVE_SCHEMA: &str = "JopeSeedingCurve.v1";
pub const SEEDING_CURVE_RELATIVE: &str = "ops/seeding-curve.v1.json";
const SEEDING_CURVE_LOCK_RELATIVE: &str = "ops/seeding-curve.lock";
/// The ingestion unit loops reconcile every few seconds (Restart=always);
/// the curve only needs minutes-level freshness.
const SEEDING_CURVE_MIN_INTERVAL_MS: u128 = 300_000;
/// Observations newer than this are left to a later pass so receipts that are
/// fetched late cannot land behind an already-advanced cursor. Late arrivals
/// beyond this window are the documented tolerance of the curve.
const SEEDING_CURVE_SETTLE_MS: i64 = 900_000;
/// Ceiling of corpus rows walked per update; catch-up converges across passes.
pub const SEEDING_CURVE_DEFAULT_MAX_ROWS: usize = 200_000;
const SEEDING_CURVE_BATCH_ROWS: usize = 20_000;

#[derive(Debug, Default, Clone)]
struct CheckpointCurve {
    strict_wins_total: u64,
    strict_wins_by_origin: BTreeMap<String, u64>,
    strict_wins_dl: u64,
    champions_total: u64,
    champions_strictly_improving: u64,
    observed_rows: u64,
    observed_dl_rows: u64,
    cursor_created_at_ms: i64,
    caught_up: bool,
    first_seen_ms: Option<i64>,
}

impl CheckpointCurve {
    fn to_value(&self) -> Value {
        let share = (self.strict_wins_total > 0)
            .then(|| self.strict_wins_dl as f64 / self.strict_wins_total as f64);
        json!({
            "first_seen_ms": self.first_seen_ms,
            "strict_wins_total": self.strict_wins_total,
            "strict_wins_by_checkpoint_origin": &self.strict_wins_by_origin,
            "strict_wins_dl": self.strict_wins_dl,
            "dl_strict_win_share": share,
            "dl_strict_win_share_pct": share.map(|value| (value * 1000.0).round() / 10.0),
            "champions_total": self.champions_total,
            "champions_strictly_improving": self.champions_strictly_improving,
            "observations": {
                "rows_scanned": self.observed_rows,
                "dl_rows": self.observed_dl_rows,
                "dl_observation_share": (self.observed_rows > 0)
                    .then(|| self.observed_dl_rows as f64 / self.observed_rows as f64),
                "cursor_created_at_ms": self.cursor_created_at_ms,
                "caught_up": self.caught_up,
            },
        })
    }
}

fn curve_u64(entry: &Value, keys: &[&str]) -> u64 {
    let mut current = entry;
    for key in keys {
        current = current.get(key).unwrap_or(&Value::Null);
    }
    current.as_u64().unwrap_or(0)
}

/// Build and atomically publish `ops/seeding-curve.v1.json`. The corpus is
/// opened read-only with the 60s busy timeout; at most `max_rows` new
/// observation rows are walked (cursor persisted in the file itself).
pub fn seeding_curve(root: &Path, max_rows: usize) -> Result<Value> {
    let curve_path = root.join(SEEDING_CURVE_RELATIVE);
    let mut checkpoints: BTreeMap<String, CheckpointCurve> = BTreeMap::new();
    // Prior observation cursors survive across runs; win/champion counts are
    // recomputed from the index files every time.
    if let Some(map) = optional_json(&curve_path)?
        .get("checkpoints")
        .and_then(Value::as_object)
    {
        for (sha, entry) in map {
            let slot = checkpoints.entry(sha.clone()).or_default();
            slot.observed_rows = curve_u64(entry, &["observations", "rows_scanned"]);
            slot.observed_dl_rows = curve_u64(entry, &["observations", "dl_rows"]);
            slot.cursor_created_at_ms = entry
                .get("observations")
                .and_then(|observations| observations.get("cursor_created_at_ms"))
                .and_then(Value::as_i64)
                .unwrap_or(-1);
        }
    }

    let mut missing_sources = Vec::new();
    match optional_json(&root.join("indexes/strict-wins.json"))?.as_array() {
        Some(wins) => {
            for win in wins {
                let sha = win
                    .get("model_checkpoint_sha256")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let origin = win
                    .get("checkpoint_origin")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let slot = checkpoints.entry(sha.to_owned()).or_default();
                slot.strict_wins_total += 1;
                *slot
                    .strict_wins_by_origin
                    .entry(origin.to_owned())
                    .or_default() += 1;
                slot.strict_wins_dl += u64::from(dl_source(origin));
            }
        }
        None => missing_sources.push("indexes/strict-wins.json"),
    }
    match optional_json(&root.join("indexes/champions.json"))?.as_array() {
        Some(champions) => {
            for champion in champions {
                let sha = champion
                    .get("best_new_strict")
                    .and_then(|best| best.get("model_checkpoint_sha256"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let slot = checkpoints.entry(sha.to_owned()).or_default();
                slot.champions_total += 1;
                slot.champions_strictly_improving += u64::from(
                    champion.get("strictly_improves").and_then(Value::as_bool) == Some(true),
                );
            }
        }
        None => missing_sources.push("indexes/champions.json"),
    }
    let hall = optional_json(&root.join("indexes/hall-of-fame.json"))?;
    let hall_totals = if hall.get("schema").is_some() {
        json!({
            "observations": hall.get("observations").cloned().unwrap_or(Value::Null),
            "failures": hall.get("failures").cloned().unwrap_or(Value::Null),
        })
    } else {
        missing_sources.push("indexes/hall-of-fame.json");
        Value::Null
    };

    let corpus_path = root.join("corpus/strict-test-exact.sqlite");
    let mut rows_walked = 0_u64;
    let corpus_available = corpus_path.is_file();
    if corpus_available {
        let connection = Connection::open_with_flags(
            &corpus_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        connection.busy_timeout(std::time::Duration::from_secs(60))?;
        // O(number of checkpoints) index seeks, never a table scan.
        let mut last_sha = String::new();
        loop {
            let next: Option<String> = connection
                .query_row(
                    "SELECT model_checkpoint_sha256 FROM observations
                       INDEXED BY observations_checkpoint_v1
                     WHERE model_checkpoint_sha256>?1
                     ORDER BY model_checkpoint_sha256 LIMIT 1",
                    params![last_sha],
                    |row| row.get(0),
                )
                .optional()?;
            let Some(sha) = next else { break };
            checkpoints.entry(sha.clone()).or_default();
            last_sha = sha;
        }
        let watermark = now_ms().saturating_sub(SEEDING_CURVE_SETTLE_MS);
        let mut budget = max_rows as u64;
        for (sha, slot) in checkpoints.iter_mut() {
            slot.first_seen_ms = connection.query_row(
                "SELECT MIN(created_at_ms) FROM observations
                   INDEXED BY observations_checkpoint_v1
                 WHERE model_checkpoint_sha256=?1",
                params![sha],
                |row| row.get(0),
            )?;
            if slot.first_seen_ms.is_none() {
                // The checkpoint only appears in the JSON indexes, not the
                // corpus; there is nothing to scan.
                slot.caught_up = true;
                continue;
            }
            if slot.cursor_created_at_ms == 0 && slot.observed_rows == 0 {
                slot.cursor_created_at_ms = -1;
            }
            loop {
                if budget == 0 {
                    break;
                }
                let batch_limit = (budget as usize).min(SEEDING_CURVE_BATCH_ROWS);
                let mut statement = connection.prepare_cached(
                    "SELECT created_at_ms, candidate_source FROM observations
                       INDEXED BY observations_checkpoint_v1
                     WHERE model_checkpoint_sha256=?1
                       AND created_at_ms>?2 AND created_at_ms<=?3
                     ORDER BY created_at_ms LIMIT ?4",
                )?;
                let batch = statement
                    .query_map(
                        params![
                            sha,
                            slot.cursor_created_at_ms,
                            watermark,
                            batch_limit as i64
                        ],
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
                    )?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                if batch.is_empty() {
                    slot.caught_up = true;
                    break;
                }
                let full = batch.len() == batch_limit;
                let last_ts = batch.last().map(|(ts, _)| *ts).unwrap_or(-1);
                let mut consumed = 0_u64;
                for (ts, source) in &batch {
                    if *ts < last_ts {
                        slot.observed_rows += 1;
                        slot.observed_dl_rows += u64::from(dl_source(source));
                        consumed += 1;
                    }
                }
                // Close the final timestamp exactly with a bounded aggregate so
                // the strictly-greater cursor can never split a timestamp.
                let (equal_total, equal_dl): (u64, u64) = connection.query_row(
                    "SELECT COUNT(*), COALESCE(SUM(
                        CASE WHEN candidate_source='accepted_dl_checkpoint'
                               OR candidate_source='current_checkpoint_dl_proposal'
                               OR candidate_source LIKE '%dl_proposal%'
                             THEN 1 ELSE 0 END), 0)
                     FROM observations INDEXED BY observations_checkpoint_v1
                     WHERE model_checkpoint_sha256=?1 AND created_at_ms=?2",
                    params![sha, last_ts],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )?;
                slot.observed_rows += equal_total;
                slot.observed_dl_rows += equal_dl;
                slot.cursor_created_at_ms = last_ts;
                let walked = consumed + equal_total;
                rows_walked += walked;
                budget = budget.saturating_sub(walked.max(1));
                if !full {
                    slot.caught_up = true;
                    break;
                }
            }
        }
    }

    // Curve points are the checkpoints that seeded strict wins, oldest first,
    // so today's single fbcd5459 generation is the first point and every
    // promoted checkpoint appends the next one.
    let mut generations = checkpoints
        .iter()
        .filter(|(_, slot)| slot.strict_wins_total > 0)
        .map(|(sha, slot)| (slot.first_seen_ms.unwrap_or(i64::MAX), sha.clone()))
        .collect::<Vec<_>>();
    generations.sort();
    let curve = generations
        .iter()
        .enumerate()
        .map(|(index, (_, sha))| {
            let mut point = checkpoints[sha].to_value();
            if let Some(object) = point.as_object_mut() {
                object.insert("generation_index".to_owned(), json!(index));
                object.insert("model_checkpoint_sha256".to_owned(), json!(sha));
            }
            point
        })
        .collect::<Vec<_>>();
    let caught_up = corpus_available && checkpoints.values().all(|slot| slot.caught_up);
    let result = json!({
        "schema": SEEDING_CURVE_SCHEMA,
        "generated_at_ms": now_ms(),
        "sources": {
            "strict_wins": "indexes/strict-wins.json",
            "champions": "indexes/champions.json",
            "hall_of_fame": "indexes/hall-of-fame.json",
            "corpus": "corpus/strict-test-exact.sqlite",
            "corpus_index": "observations_checkpoint_v1",
        },
        "dl_origin_families": [
            "accepted_dl_checkpoint",
            "current_checkpoint_dl_proposal",
            "*dl_proposal*",
        ],
        "missing_sources": missing_sources,
        "hall_of_fame_totals": hall_totals,
        "observation_scan": {
            "corpus_available": corpus_available,
            "settle_ms": SEEDING_CURVE_SETTLE_MS,
            "max_rows_per_update": max_rows,
            "rows_walked_this_pass": rows_walked,
            "caught_up": caught_up,
        },
        "curve": curve,
        "checkpoints": checkpoints
            .iter()
            .map(|(sha, slot)| (sha.clone(), slot.to_value()))
            .collect::<Map<String, Value>>(),
    });
    atomic_json(&curve_path, &result)?;
    Ok(result)
}

struct CurveLock(#[allow(dead_code)] File);

fn try_curve_lock(root: &Path) -> Result<Option<CurveLock>> {
    let lock_path = root.join(SEEDING_CURVE_LOCK_RELATIVE);
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)?;
    if file.try_lock().is_err() {
        return Ok(None);
    }
    Ok(Some(CurveLock(file)))
}

/// CLI entrypoint: exclusive (fail-loud) seeding-curve rebuild.
pub fn seeding_curve_locked(root: &Path, max_rows: usize) -> Result<Value> {
    let Some(_lock) = try_curve_lock(root)? else {
        bail!(
            "seeding-curve lock is held by another writer: {}",
            root.join(SEEDING_CURVE_LOCK_RELATIVE).display()
        );
    };
    seeding_curve(root, max_rows)
}

/// Post-reconcile hook: throttled by output-file age, skip-on-lock, and never
/// fails the reconcile that invoked it.
pub fn seeding_curve_after_reconcile(root: &Path) -> Value {
    let curve_path = root.join(SEEDING_CURVE_RELATIVE);
    if let Ok(age) = fs::metadata(&curve_path)
        .and_then(|metadata| metadata.modified())
        .map(|modified| {
            SystemTime::now()
                .duration_since(modified)
                .unwrap_or_default()
        })
    {
        if age.as_millis() < SEEDING_CURVE_MIN_INTERVAL_MS {
            return json!({
                "written": false,
                "skipped": "fresh",
                "age_ms": age.as_millis() as u64,
            });
        }
    }
    let lock = match try_curve_lock(root) {
        Ok(Some(lock)) => lock,
        Ok(None) => return json!({"written": false, "skipped": "locked"}),
        Err(error) => return json!({"written": false, "error": error.to_string()}),
    };
    let result = match seeding_curve(root, SEEDING_CURVE_DEFAULT_MAX_ROWS) {
        Ok(curve) => json!({
            "written": true,
            "curve_points": curve["curve"].as_array().map_or(0, Vec::len),
            "rows_walked": curve["observation_scan"]["rows_walked_this_pass"],
            "caught_up": curve["observation_scan"]["caught_up"],
        }),
        Err(error) => json!({"written": false, "error": error.to_string()}),
    };
    drop(lock);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn interrupted_staging_directories_are_not_counted_as_published() {
        let root = TempDir::new().unwrap();
        let manifest = |dir: &str| {
            let path = root.path().join(dir);
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("cache-manifest.v1.json"), "{}").unwrap();
        };

        // Published caches: renamed into place, never dot-prefixed.
        manifest("synth_0049e4a0e270dbfa");
        manifest("81672f160e2b0312");

        // A worker killed between the manifest write and the rename. This is the
        // case a manifest-only test miscounts, because the manifest really is
        // there -- the publish just never happened.
        manifest(".synth_03534d1dad4573d8.tmp-3905477-3905477");
        manifest(".encoder-.synth_043a5b4fa9e6d2ee.tmp-3905478-3905478");

        // Killed before the manifest write: excluded by name and by manifest.
        fs::create_dir_all(root.path().join(".synth_044d96887c03bc0a.tmp-1-1")).unwrap();

        assert_eq!(completed_cache_count(root.path()).unwrap(), 2);
    }

    #[test]
    fn cache_count_of_a_missing_root_is_zero() {
        let root = TempDir::new().unwrap();
        let absent = root.path().join("timeseries-expanded");
        assert_eq!(completed_cache_count(&absent).unwrap(), 0);
    }

    #[test]
    fn authority_metric_map_agrees_with_task_authority() {
        use crate::task::TaskKind;
        for kind in [
            TaskKind::Regression,
            TaskKind::Binary,
            TaskKind::Timeseries,
            TaskKind::Multiclass,
        ] {
            assert_eq!(
                expected_authority_metric(kind.as_str()).unwrap(),
                kind.primary_metric()
            );
        }
        assert!(expected_authority_metric("multi_label").is_err());
    }

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn receipt_fixture(root: &Path) -> Result<(PathBuf, String)> {
        let output = root.join("receipt");
        fs::create_dir_all(output.join("predictions"))?;
        fs::write(output.join("SHARD_OK"), b"ok\n")?;
        let genome = "(+ x1 x2)";
        let semantic = sha256_bytes(genome.as_bytes());
        let prediction = [0.25_f64, 0.75_f64]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>();
        let prediction_sha = sha256_bytes(&prediction);
        fs::write(output.join("predictions/prediction.f64le"), prediction)?;
        let authority = json!({
            "score_tier": "test_exact",
            "metric": "r2",
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": SELECTION_BIAS,
            "generation": "strict-wave-000001",
            "frozen_seed_sha256": digest('1'),
            "model_checkpoint_sha256": digest('2'),
            "hall_of_fame_sha256": digest('3'),
        });
        let manifest = json!({
            "task": "regression",
            "dataset_hash": "dataset",
            "train_sha256": digest('4'),
            "test_sha256": digest('5'),
            "label_authority": authority,
            "candidates": [{
                "semantic_sha256": semantic,
                "canonical_genome": genome,
                "source": "dl_proposal",
            }],
        });
        atomic_json(&output.join(".portable-manifest.json"), &manifest)?;
        let row = json!({
            "task": "regression",
            "dataset_hash": "dataset",
            "semantic_sha256": semantic,
            "canonical_genome": genome,
            "tier": "test_exact",
            "metric": "r2",
            "score": 0.4,
            "secondary_metric": "rmse",
            "secondary_score": 0.5,
            "valid": true,
            "runtime_seconds": 1.0,
            "train_sha256": digest('4'),
            "test_sha256": digest('5'),
            "manifest_sha256": digest('6'),
            "created_at_ms": 7,
            "predictions_path": "predictions/prediction.f64le",
            "predictions_sha256": prediction_sha,
            "prediction_count": 2,
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": SELECTION_BIAS,
            "generation": "strict-wave-000001",
            "frozen_seed_sha256": digest('1'),
            "model_checkpoint_sha256": digest('2'),
            "hall_of_fame_sha256": digest('3'),
        });
        atomic_jsonl(
            &output.join("dataset.receipts.jsonl"),
            std::slice::from_ref(&row),
        )?;
        Ok((output, semantic))
    }

    #[test]
    fn job_receipt_is_exactly_once() -> Result<()> {
        let temporary = TempDir::new()?;
        let (output, _) = receipt_fixture(temporary.path())?;
        let corpus = temporary.path().join("corpus.sqlite");
        let first = ingest_receipt(&corpus, "job-1", &output)?;
        let second = ingest_receipt(&corpus, "job-1", &output)?;
        assert_eq!(first.rows, 1);
        assert!(!first.already_ingested);
        assert!(second.already_ingested);
        let database = Connection::open(corpus)?;
        assert_eq!(
            database.query_row("SELECT COUNT(*) FROM observations", [], |row| row
                .get::<_, i64>(0))?,
            1
        );
        Ok(())
    }

    #[test]
    fn causal_receipts_survive_training_table_deduplication() -> Result<()> {
        let temporary = TempDir::new()?;
        let corpus = temporary.path().join("corpus.sqlite");
        for (index, manifest_character) in [(0_u64, '6'), (1_u64, '7')] {
            let fixture_root = temporary.path().join(format!("fixture-{index}"));
            let (output, _) = receipt_fixture(&fixture_root)?;
            let mut manifest = read_json(&output.join(".portable-manifest.json"))?;
            manifest["label_authority"]["generation"] =
                Value::String("strict-wave-000191".to_owned());
            manifest["candidates"][0]["source"] = Value::String("gp_comparator".to_owned());
            manifest["candidates"][0]["evidence"] = json!({
                "evaluation_lane": "matched_gp_comparator",
                "matched_pair_index": index,
                "zero_dl_ancestry": true,
                "proposal_valid": true,
                "valid_lineage": true,
            });
            atomic_json(&output.join(".portable-manifest.json"), &manifest)?;
            let receipts_path = output.join("dataset.receipts.jsonl");
            let line = fs::read_to_string(&receipts_path)?;
            let mut receipt: Value = serde_json::from_str(line.trim())?;
            receipt["generation"] = Value::String("strict-wave-000191".to_owned());
            receipt["manifest_sha256"] = Value::String(digest(manifest_character));
            atomic_jsonl(&receipts_path, &[receipt])?;
            ingest_receipt(&corpus, &format!("causal-{index}"), &output)?;
        }
        let connection = Connection::open(&corpus)?;
        assert_eq!(
            connection.query_row("SELECT COUNT(*) FROM observations", [], |row| {
                row.get::<_, i64>(0)
            })?,
            1
        );
        assert_eq!(
            connection.query_row(
                "SELECT COUNT(*) FROM causal_evaluation_receipts_v2",
                [],
                |row| row.get::<_, i64>(0)
            )?,
            2
        );
        assert!(connection
            .execute(
                "UPDATE causal_evaluation_receipts_v2 SET runtime_seconds=0",
                []
            )
            .is_err());
        Ok(())
    }

    #[test]
    fn changed_receipt_cannot_reuse_job_identity() -> Result<()> {
        let temporary = TempDir::new()?;
        let (output, _) = receipt_fixture(temporary.path())?;
        let corpus = temporary.path().join("corpus.sqlite");
        ingest_receipt(&corpus, "job-1", &output)?;
        let path = output.join("dataset.receipts.jsonl");
        let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
        file.write_all(b"\n")?;
        let error = ingest_receipt(&corpus, "job-1", &output).unwrap_err();
        assert!(error.to_string().contains("conflicts"));
        Ok(())
    }

    #[test]
    fn gene_evidence_is_stable_and_decision_complete() -> Result<()> {
        let temporary = TempDir::new()?;
        let root = temporary.path();
        let (output, semantic) = receipt_fixture(root)?;
        ingest_receipt(
            &root.join("corpus/strict-test-exact.sqlite"),
            "job-1",
            &output,
        )?;
        let target = root.join("indexes/gene-evidence.v2.jsonl");
        let summary = export_gene_evidence(root, &target)?;
        assert_eq!(summary["rows"], 1);
        let line: Value = serde_json::from_str(fs::read_to_string(target)?.trim())?;
        assert_eq!(line["schema"], "JopeGeneEvidence.v2");
        assert_eq!(line["semantic_sha256"], semantic);
        assert_eq!(line["within_dataset_rank"], 1);
        assert_eq!(line["test_adaptive"], true);
        Ok(())
    }

    fn insert_evidence(
        connection: &Connection,
        task: &str,
        dataset: &str,
        genome: &str,
        score: f64,
        provenance: (&str, &str, &str),
        created_at_ms: i64,
    ) -> Result<String> {
        let (source, origin, protocol) = provenance;
        let semantic = sha256_bytes(genome.as_bytes());
        connection.execute(
            "INSERT INTO observations(
               task,dataset_hash,semantic_sha256,canonical_genome,candidate_source,
               metric,score,runtime_seconds,train_sha256,test_sha256,
               receipt_sha256,receipt_path,manifest_sha256,frozen_seed_sha256,
               model_checkpoint_sha256,hall_of_fame_sha256,created_at_ms,
               authority_kind,protocol_sha256,origin_class
             ) VALUES(
               ?1,?2,?3,?4,?5,'r2',?6,1.0,?7,?8,
               ?9,'receipt',?10,?11,?12,?13,?14,
               'corrected',?15,?16
             )",
            params![
                task,
                dataset,
                semantic,
                genome,
                source,
                score,
                digest('a'),
                digest('b'),
                digest('c'),
                digest('d'),
                digest('e'),
                digest('f'),
                digest('7'),
                created_at_ms,
                protocol,
                origin,
            ],
        )?;
        Ok(semantic)
    }

    #[test]
    fn evidence_queries_are_protocol_scoped_indexed_and_dynamic() -> Result<()> {
        let temporary = TempDir::new()?;
        let root = temporary.path();
        let corpus_path = root.join("corpus/strict-test-exact.sqlite");
        let corpus = open_corpus(&corpus_path)?;
        let protocol = digest('9');
        let gp_repeat = insert_evidence(
            &corpus,
            "regression",
            "dataset-1",
            "(pipe X (ridge))",
            0.5,
            (
                "cross_dataset_repeated_robust_gp_winner",
                "gp_origin",
                &protocol,
            ),
            1,
        )?;
        insert_evidence(
            &corpus,
            "regression",
            "dataset-2",
            "(pipe X (ridge))",
            0.8,
            (
                "cross_dataset_repeated_robust_gp_winner",
                "gp_origin",
                &protocol,
            ),
            2,
        )?;
        insert_evidence(
            &corpus,
            "regression",
            "dataset-1",
            "(pipe X (ridge_gcv))",
            0.6,
            ("dataset_local_elite", "gp_origin", &protocol),
            3,
        )?;
        insert_evidence(
            &corpus,
            "regression",
            "dataset-1",
            "(pipe X (elasticnet))",
            0.7,
            ("current_checkpoint_dl_proposal", "dl_origin", &protocol),
            4,
        )?;
        insert_evidence(
            &corpus,
            "regression",
            "dataset-1",
            "(pipe X (gbm))",
            0.9,
            (
                "deterministic_rust_mutation_crossover",
                "dl_descendant",
                &protocol,
            ),
            5,
        )?;
        drop(corpus);

        let report = evidence_report(
            root,
            "regression",
            &protocol,
            Some("dataset-1"),
            Some(&gp_repeat),
            10,
        )?;
        assert_eq!(
            report["best_genome_per_dataset"][0]["score"],
            Value::from(0.9)
        );
        assert_eq!(
            report["best_datasets_per_semantic"][0]["dataset_hash"],
            "dataset-2"
        );
        assert!(
            (report["dl_origin_parent_descendant_lift"][0]["dl_lift_vs_gp"]
                .as_f64()
                .unwrap()
                - 0.3)
                .abs()
                < 1.0e-12
        );
        let dataset_plan = report["query_plans"]["dataset_best"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" ");
        assert!(dataset_plan.contains("observations_dataset_protocol_best_v2"));
        let semantic_plan = report["query_plans"]["semantic_datasets"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" ");
        assert!(semantic_plan.contains("observations_semantic_protocol_best_v2"));

        let bank = dynamic_seed_bank(&corpus_path, "regression", &protocol, 10, 2)?;
        assert_eq!(bank["tasks"]["regression"][0]["semantic_sha256"], gp_repeat);
        assert_eq!(
            bank["tasks"]["regression"][0]["dataset_count"],
            Value::from(2)
        );
        Ok(())
    }

    fn insert_curve_row(
        connection: &Connection,
        dataset: &str,
        genome: &str,
        source: &str,
        checkpoint: &str,
        created_at_ms: i64,
    ) -> Result<()> {
        let semantic = sha256_bytes(genome.as_bytes());
        connection.execute(
            "INSERT INTO observations(
               task,dataset_hash,semantic_sha256,canonical_genome,candidate_source,
               metric,score,runtime_seconds,train_sha256,test_sha256,
               receipt_sha256,receipt_path,manifest_sha256,frozen_seed_sha256,
               model_checkpoint_sha256,hall_of_fame_sha256,created_at_ms,
               authority_kind,protocol_sha256,origin_class
             ) VALUES(
               'regression',?1,?2,?3,?4,'r2',0.5,1.0,?5,?5,
               ?5,'receipt',?5,?5,?6,?5,?7,
               'corrected',?5,'unknown'
             )",
            params![
                dataset,
                semantic,
                genome,
                source,
                digest('a'),
                checkpoint,
                created_at_ms,
            ],
        )?;
        Ok(())
    }

    fn seeding_curve_index_fixtures(root: &Path, checkpoint: &str) -> Result<()> {
        // 5 strict wins for one checkpoint generation, 2 from the DL families.
        let win = |origin: &str| {
            json!({
                "task": "regression",
                "dataset_hash": "dataset",
                "model_checkpoint_sha256": checkpoint,
                "checkpoint_origin": origin,
            })
        };
        atomic_json(
            &root.join("indexes/strict-wins.json"),
            &json!([
                win("accepted_dl_checkpoint"),
                win("current_checkpoint_dl_proposal"),
                win("strict_wave_policy_mutation_crossover"),
                win("task_common_winner"),
                win("deterministic_mutation_crossover"),
            ]),
        )?;
        atomic_json(
            &root.join("indexes/champions.json"),
            &json!([
                {
                    "strictly_improves": true,
                    "best_new_strict": {"model_checkpoint_sha256": checkpoint},
                },
                {
                    "strictly_improves": false,
                    "best_new_strict": {"model_checkpoint_sha256": digest('9')},
                },
            ]),
        )?;
        atomic_json(
            &root.join("indexes/hall-of-fame.json"),
            &json!({
                "schema": "JopeTestOnlyStrictHallOfFame.v1",
                "observations": 4,
                "failures": 0,
            }),
        )?;
        Ok(())
    }

    #[test]
    fn seeding_curve_counts_dl_share_and_scans_incrementally() -> Result<()> {
        let temporary = TempDir::new()?;
        let root = temporary.path();
        let checkpoint = digest('2');
        seeding_curve_index_fixtures(root, &checkpoint)?;
        let corpus = open_corpus(&root.join("corpus/strict-test-exact.sqlite"))?;
        // Two DL-family rows and one GP row for the generation checkpoint, plus
        // a legacy checkpoint whose rows all sit at created_at_ms=0.
        insert_curve_row(
            &corpus,
            "d1",
            "(pipe X (ridge))",
            "current_checkpoint_dl_proposal",
            &checkpoint,
            1_000,
        )?;
        insert_curve_row(
            &corpus,
            "d2",
            "(pipe X (lasso))",
            "accepted_dl_checkpoint",
            &checkpoint,
            1_000,
        )?;
        insert_curve_row(
            &corpus,
            "d3",
            "(pipe X (gbm))",
            "strict_wave_policy_mutation_crossover",
            &checkpoint,
            2_000,
        )?;
        insert_curve_row(
            &corpus,
            "d1",
            "(pipe X (hinge))",
            "robust_gp_seed_bank",
            &digest('3'),
            0,
        )?;
        drop(corpus);

        let first = seeding_curve(root, 10_000)?;
        assert_eq!(first["schema"], SEEDING_CURVE_SCHEMA);
        assert_eq!(first["curve"].as_array().map(Vec::len), Some(1));
        let point = &first["curve"][0];
        assert_eq!(point["generation_index"], 0);
        assert_eq!(
            point["model_checkpoint_sha256"],
            Value::from(checkpoint.clone())
        );
        assert_eq!(point["strict_wins_total"], 5);
        assert_eq!(point["strict_wins_dl"], 2);
        assert_eq!(
            point["strict_wins_by_checkpoint_origin"]["accepted_dl_checkpoint"],
            1
        );
        assert!((point["dl_strict_win_share"].as_f64().unwrap() - 0.4).abs() < 1e-12);
        assert_eq!(point["dl_strict_win_share_pct"], Value::from(40.0));
        assert_eq!(point["champions_total"], 1);
        assert_eq!(point["champions_strictly_improving"], 1);
        assert_eq!(point["observations"]["rows_scanned"], 3);
        assert_eq!(point["observations"]["dl_rows"], 2);
        assert_eq!(point["observations"]["caught_up"], true);
        // The legacy checkpoint (created_at_ms=0 rows, no strict wins) is in
        // the checkpoint map but not on the curve.
        let legacy = &first["checkpoints"][digest('3')];
        assert_eq!(legacy["observations"]["rows_scanned"], 1);
        assert_eq!(legacy["observations"]["dl_rows"], 0);
        assert_eq!(legacy["strict_wins_total"], 0);
        assert_eq!(first["observation_scan"]["caught_up"], true);
        assert_eq!(first["missing_sources"].as_array().map(Vec::len), Some(0));
        assert_eq!(first["hall_of_fame_totals"]["observations"], 4);

        // A later pass only walks the new rows past the persisted cursor and
        // accumulates without double counting.
        let corpus = open_corpus(&root.join("corpus/strict-test-exact.sqlite"))?;
        insert_curve_row(
            &corpus,
            "d4",
            "(pipe X (elasticnet))",
            "generalizing_dl_proposal_variant",
            &checkpoint,
            3_000,
        )?;
        drop(corpus);
        let second = seeding_curve(root, 10_000)?;
        let point = &second["curve"][0];
        assert_eq!(point["observations"]["rows_scanned"], 4);
        assert_eq!(point["observations"]["dl_rows"], 3);
        assert_eq!(
            second["observation_scan"]["rows_walked_this_pass"],
            Value::from(1)
        );
        let third = seeding_curve(root, 10_000)?;
        assert_eq!(third["curve"][0]["observations"]["rows_scanned"], 4);
        assert_eq!(
            third["observation_scan"]["rows_walked_this_pass"],
            Value::from(0)
        );
        Ok(())
    }

    #[test]
    fn seeding_curve_budget_splits_batches_without_double_counting() -> Result<()> {
        let temporary = TempDir::new()?;
        let root = temporary.path();
        let checkpoint = digest('2');
        seeding_curve_index_fixtures(root, &checkpoint)?;
        let corpus = open_corpus(&root.join("corpus/strict-test-exact.sqlite"))?;
        for index in 0..7_i64 {
            insert_curve_row(
                &corpus,
                &format!("d{index}"),
                &format!("(pipe X (ridge alpha={index}))"),
                if index % 2 == 0 {
                    "current_checkpoint_dl_proposal"
                } else {
                    "robust_gp_seed_bank"
                },
                &checkpoint,
                // Duplicate timestamps across the batch boundary.
                1_000 + (index / 2) * 10,
            )?;
        }
        drop(corpus);
        let mut scanned = 0_u64;
        for _ in 0..10 {
            let pass = seeding_curve(root, 3)?;
            scanned = pass["curve"][0]["observations"]["rows_scanned"]
                .as_u64()
                .unwrap();
            if pass["observation_scan"]["caught_up"] == true {
                break;
            }
        }
        assert_eq!(scanned, 7);
        let settled = seeding_curve(root, 10_000)?;
        assert_eq!(settled["curve"][0]["observations"]["rows_scanned"], 7);
        assert_eq!(settled["curve"][0]["observations"]["dl_rows"], 4);
        Ok(())
    }

    #[test]
    fn seeding_curve_after_reconcile_is_throttled_and_best_effort() -> Result<()> {
        let temporary = TempDir::new()?;
        let root = temporary.path();
        let checkpoint = digest('2');
        seeding_curve_index_fixtures(root, &checkpoint)?;
        drop(open_corpus(&root.join("corpus/strict-test-exact.sqlite"))?);
        let first = seeding_curve_after_reconcile(root);
        assert_eq!(first["written"], true);
        assert!(root.join(SEEDING_CURVE_RELATIVE).is_file());
        let second = seeding_curve_after_reconcile(root);
        assert_eq!(second["written"], false);
        assert_eq!(second["skipped"], "fresh");
        Ok(())
    }

    #[test]
    fn four_task_telemetry_is_complete_bounded_and_uses_typed_absence() -> Result<()> {
        let temporary = TempDir::new()?;
        let root = temporary.path();
        let catalog_path = root.join("catalog/catalog.json");
        let entries = [
            ("regression", "r1", "family-r"),
            ("binary", "b1", "family-b"),
            ("multiclass", "m1", "family-m"),
            ("timeseries", "t1", "family-t"),
        ]
        .into_iter()
        .map(|(task, dataset_hash, source_family)| {
            json!({
                "task": task,
                "dataset_hash": dataset_hash,
                "source_family": source_family,
            })
        })
        .collect::<Vec<_>>();
        atomic_json(
            &catalog_path,
            &json!({
                "schema": "JopeTestOnlyContextCatalog.v1",
                "entries": entries,
            }),
        )?;
        atomic_json(
            &root.join("catalog/current.json"),
            &json!({
                "schema": "JopeTestOnlyContextPointer.v1",
                "catalog": catalog_path,
                "catalog_sha256": sha256_file(&catalog_path)?,
            }),
        )?;
        atomic_json(
            &root.join("training/current-corpus.json"),
            &json!({"schema":"PackedCorpus.v3", "watermark": 17}),
        )?;
        atomic_json(
            &root.join("training/accepted-by-task.json"),
            &json!({
                "schema":"JopeSelfplayAcceptedByTask.v1",
                "tasks": {
                    "regression": {
                        "checkpoint_sha256": digest('a'),
                        "accepted": true,
                    }
                }
            }),
        )?;
        let state_path = root.join("ops/dataset-state.sqlite");
        fs::create_dir_all(state_path.parent().unwrap())?;
        let state = Connection::open(&state_path)?;
        state.execute_batch(
            "CREATE TABLE datasets(
               task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
               sidecars_complete INTEGER NOT NULL,catalog_active INTEGER NOT NULL,
               obs_count INTEGER NOT NULL);
             INSERT INTO datasets VALUES
               ('regression','r1',1,1,1),
               ('binary','b1',1,1,0),
               ('multiclass','m1',1,1,0),
               ('timeseries','t1',1,1,0);",
        )?;
        drop(state);
        // One injected instant for the observation and both rebuilds: the rows depend
        // on the snapshots and this clock only, never on how long the rebuild took.
        let now = now_ms();
        let corpus = open_corpus(&root.join("corpus/strict-test-exact.sqlite"))?;
        insert_curve_row(
            &corpus,
            "r1",
            "(pipe X (ridge))",
            "robust_gp_seed_bank",
            &digest('a'),
            now,
        )?;
        corpus.execute(
            "UPDATE observations SET score=-99.0 WHERE task='regression' AND dataset_hash='r1'",
            [],
        )?;
        drop(corpus);

        let first = four_task_telemetry_at(root, now)?;
        let second = four_task_telemetry_at(root, now)?;
        assert_eq!(first.rows, second.rows);
        assert_eq!(
            first.rows[0].freshness_reason,
            "fresh — last observation 0s ago"
        );
        // Ages follow the injected clock across the second and staleness boundaries.
        let later = four_task_telemetry_at(root, now + 1_000)?;
        assert_ne!(later.rows, first.rows);
        assert_eq!(
            later.rows[0].freshness_reason,
            "fresh — last observation 1s ago"
        );
        let stale = four_task_telemetry_at(root, now + 901_000)?;
        assert_eq!(
            stale.rows[0].freshness_reason,
            "stale — last observation 901s ago"
        );
        assert_eq!(
            first.rows.iter().map(|row| row.task).collect::<Vec<_>>(),
            FOUR_TASK_ORDER
        );
        let regression = &first.rows[0];
        assert_eq!(
            regression
                .metrics
                .primary_performance
                .value
                .as_ref()
                .unwrap()
                .minimum,
            -1.0
        );
        assert_eq!(
            regression
                .metrics
                .operational_loss
                .value
                .as_ref()
                .unwrap()
                .maximum,
            2.0
        );
        let timeseries = &first.rows[3];
        assert!(timeseries.metrics.primary_performance.value.is_none());
        assert_eq!(
            timeseries.metrics.primary_performance.na_reason.as_deref(),
            Some("no observations")
        );
        assert!(timeseries.counts.completed.value.is_none());
        assert_eq!(
            timeseries.counts.completed.na_reason.as_deref(),
            Some("no observations")
        );
        assert!(timeseries.valid_observations_per_minute.value.is_none());
        assert!(root.join("ops/four-task-telemetry.v1.json").is_file());
        Ok(())
    }
}
