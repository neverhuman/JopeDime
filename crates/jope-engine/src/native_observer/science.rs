//! Bounded, read-only scientific evidence. Slow verification never owns the
//! operational sampler thread, and recorded qualification is never production.
use super::*;
use crate::{
    manifest::hash,
    native_graph::{Artifact, Graph},
};
use serde::Deserialize;
use std::{
    os::unix::fs::MetadataExt,
    sync::{Arc, Mutex},
    thread,
};

pub const CATALOG: &str = "/etc/jope-smartcluster/scientific-catalog.json";
const MAX_METADATA: u64 = 256 * 1024 * 1024;
const STALE_MS: u64 = 30_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Catalog {
    schema: String,
    entries: Vec<Source>,
}
#[derive(Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Source {
    RecordedQualification {
        id: String,
        artifact: Artifact,
        original_at_unix_ms: Option<u64>,
    },
    GraphRun {
        id: String,
        root: PathBuf,
        graph_state: PathBuf,
        worker_state: PathBuf,
        loop_state: PathBuf,
        config: Artifact,
    },
    OrderedWinners {
        id: String,
        root: PathBuf,
        outbox: PathBuf,
        #[serde(default)]
        service_receipts: Vec<Artifact>,
    },
}
impl Source {
    fn id(&self) -> &str {
        match self {
            Self::RecordedQualification { id, .. }
            | Self::GraphRun { id, .. }
            | Self::OrderedWinners { id, .. } => id,
        }
    }
}
#[derive(Clone, Default)]
struct Rows {
    checkpoints: Vec<Value>,
    continuations: Vec<Value>,
    comparisons: Vec<Value>,
    winners: Vec<Value>,
    corpus: Option<Value>,
    notes: Vec<String>,
    evidence: Vec<Value>,
    errors: Vec<String>,
    publications: BTreeMap<String, Value>,
}
#[derive(Clone, Default)]
struct Observed {
    at: Option<u64>,
    rows: Rows,
}
/// A single verifier thread; no concurrent graph writers and no task execution.
pub struct Reader {
    latest: Arc<Mutex<Observed>>,
}
fn regular(path: &Path, limit: u64) -> Result<Vec<u8>> {
    physical(path)?;
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let before = f.metadata()?;
    ensure!(
        before.is_file() && before.len() <= limit,
        "bounded regular science evidence required"
    );
    let mut b = Vec::new();
    (&f).take(limit + 1).read_to_end(&mut b)?;
    let after = f.metadata()?;
    let named = fs::symlink_metadata(path)?;
    ensure!(
        b.len() as u64 <= limit
            && before.len() == after.len()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec()
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec()
            && before.dev() == named.dev()
            && before.ino() == named.ino()
            && !named.file_type().is_symlink(),
        "scientific source changed during read"
    );
    Ok(b)
}
fn checked(root: Option<&Path>, a: &Artifact, limit: u64) -> Result<Value> {
    ensure!(
        a.sha256.len() == 64
            && a.sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid science digest"
    );
    let path = match root {
        Some(root) => crate::manifest::resolve(root, &a.path)?,
        None => a.path.clone(),
    };
    let bytes = regular(&path, limit)?;
    ensure!(hash(&bytes) == a.sha256, "scientific artifact hash changed");
    Ok(serde_json::from_slice(&bytes)?)
}
fn artifact(v: &Value) -> Result<Artifact> {
    let a = Artifact {
        path: v["path"].as_str().context("artifact path absent")?.into(),
        sha256: v["sha256"]
            .as_str()
            .context("artifact digest absent")?
            .into(),
    };
    ensure!(
        a.path.as_os_str().len() <= 4096
            && a.sha256.len() == 64
            && a.sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "malformed scientific artifact identity"
    );
    Ok(a)
}
fn original_time(value: Option<u64>, at: u64) -> Result<Option<u64>> {
    ensure!(
        value.is_none_or(|t| t > 0 && t <= at.saturating_add(5000)),
        "original scientific timestamp is invalid/future"
    );
    Ok(value)
}
fn finite(v: &Value) -> Option<f64> {
    v.as_f64().filter(|v| v.is_finite() && *v >= 0.0)
}
fn row_evidence(
    id: &str,
    reference: &Artifact,
    kind: &str,
    original: Option<u64>,
    at: u64,
) -> Value {
    json!({"id":id,"artifact":reference,"verification_kind":kind,"original_at_unix_ms":original,
        "verified_at_unix_ms":at,"stale_after_ms":STALE_MS,"production_model_improvement_claim":false})
}
fn qualification_rows(
    id: &str,
    a: &Artifact,
    v: &Value,
    original: Option<u64>,
    at: u64,
) -> Result<Rows> {
    let mut rows = Rows::default();
    original_time(original, at)?;
    rows.evidence.push(row_evidence(
        id,
        a,
        "hash_checked_root_catalog_recorded_qualification",
        original,
        at,
    ));
    rows.notes.push(text(format!("Historical qualification {id}; original time={original:?}; receipt {}. Recorded review, not fresh numerical execution or production learning.",a.sha256)));
    match v["schema"]
        .as_str()
        .context("qualification schema absent")?
    {
        "jope.hardware_training_benchmark_verified.v1" => {
            ensure!(
                v["production_learning"] == false && v["numerical_result_verified"] == true,
                "qualification training flags disagree"
            );
            let initial = v["initial_optimizer_step"]
                .as_u64()
                .context("initial optimizer step absent")?;
            let target = v["target_optimizer_step"]
                .as_u64()
                .context("target optimizer step absent")?;
            ensure!(
                initial.checked_add(v["exact_updates"].as_u64().context("update count absent")?)
                    == Some(target),
                "qualification optimizer accounting differs"
            );
            let state = artifact(&v["checkpoint"])?;
            rows.checkpoints.push(json!({"identity":state.sha256,"status":"historical qualification checkpoint","step":target,
                "reason":text(format!("{id}; {} updates; microbatch {}; lifecycle passed={}; original receipt {}",v["exact_updates"],v["microbatch"],v["full_scheduler_lifecycle_passed"],a.sha256)),
                "qualification":true,"original_at_unix_ms":original,"source":a,"checkpoint":state}));
        }
        "jope.actual_neural_eof_recovery_qualification.v1" => {
            ensure!(
                v["production_learning"] == false
                    && v["passed"] == true
                    && v["unknown_final_tail"] == false,
                "EOF qualification incomplete"
            );
            let initial = v["original_initial_step"]
                .as_u64()
                .context("original initial absent")?;
            let target = v["original_target_step"]
                .as_u64()
                .context("original target absent")?;
            let count = v["first_attempt_committed_updates"]
                .as_u64()
                .context("first updates absent")?
                .checked_add(
                    v["retry_executed_updates"]
                        .as_u64()
                        .context("retry updates absent")?,
                )
                .context("update overflow")?;
            ensure!(
                initial.checked_add(count) == Some(target) && v["total_updates"] == count,
                "EOF original window differs"
            );
            let state = artifact(&v["final_checkpoint"])?;
            rows.checkpoints.push(json!({"identity":state.sha256,"status":"historical qualification EOF recovery","step":target,
                "reason":text(format!("{id}; original {initial} + {count} updates; job {}; not production learning",v["job"]["id"])),
                "qualification":true,"original_at_unix_ms":original,"source":a,"checkpoint":state}));
        }
        "jope.native_qualification_receipt_check.v1" => {
            ensure!(
                v["qualification_only"] == true
                    && v["output"]["schema"] == "jope.native_gp_completion.v1"
                    && v["durable_result"]["schema"] == "jope.smartcluster_completion.v1"
                    && v["durable_result"]["manifest_sha256"] == v["manifest_sha256"],
                "GP qualification binding differs"
            );
            rows.notes.push(text(format!("Historical population GP {id}: logical budget {}, native receipt {}; this is not a seeded-continuation result.",v["output"]["evaluations_budget"],v["durable_result"]["receipt_id"])));
        }
        "jope.actual_gp_archive_independent_restoration.v1" => {
            ensure!(
                v["qualification_only"] == true
                    && v["passed"] == true
                    && v["source_node_access"] == false,
                "restoration qualification flags differ"
            );
            rows.notes.push(text(format!("Historical GP archive restoration {id}: {} original files / {} bytes; {} authorization witness rows; no fresh fits inferred.",v["regular_files_verified"],v["bytes_verified"],v["table_counts"]["authorization_witness"])));
        }
        "jope.actual_native_scope_recovery_qualification.v1" => {
            ensure!(
                v["fixture_only"] == true
                    && v["fits"] == 0
                    && v["activations"] == 0
                    && v["acknowledged_scope_replayed_after_stale_db"] == true
                    && v["recovery_hold_retained"] == true,
                "scope fixture is not retained zero-fit recovery evidence"
            );
            rows.notes.push(text(format!("Historical scope-recovery fixture {id}: {} native identities restored, zero fits/activations; metadata qualification only",v["native_training_rows"])));
        }
        schema => anyhow::bail!("unsupported recorded qualification schema {schema}"),
    }
    stamp_rows(&mut rows, at);
    Ok(rows)
}
fn stamp_rows(rows: &mut Rows, at: u64) {
    for row in rows
        .checkpoints
        .iter_mut()
        .chain(&mut rows.continuations)
        .chain(&mut rows.comparisons)
        .chain(&mut rows.winners)
    {
        if row.get("scientific_verified_at_unix_ms").is_none() {
            row["scientific_verified_at_unix_ms"] = json!(at);
        }
    }
    if let Some(c) = rows.corpus.as_mut() {
        if c.get("scientific_verified_at_unix_ms").is_none() {
            c["scientific_verified_at_unix_ms"] = json!(at);
        }
    }
}
fn merge(to: &mut Rows, mut from: Rows) {
    to.checkpoints.append(&mut from.checkpoints);
    to.continuations.append(&mut from.continuations);
    to.comparisons.append(&mut from.comparisons);
    to.winners.append(&mut from.winners);
    if from.corpus.is_some() {
        to.corpus = from.corpus;
    }
    to.notes.append(&mut from.notes);
    to.evidence.append(&mut from.evidence);
    to.errors.append(&mut from.errors);
    for (event, status) in from.publications {
        if let Some(previous) = to.publications.get(&event) {
            if previous["local_commit"] != status["local_commit"] {
                to.errors.push(text(format!(
                    "Conflicting recorded publication for winner {event}"
                )));
                continue;
            }
        }
        to.publications.insert(event, status);
    }
}
fn bounded_files(path: &Path, limit: usize) -> Result<Vec<PathBuf>> {
    physical(path)?;
    let mut files = Vec::new();
    for entry in fs::read_dir(path)? {
        ensure!(
            files.len() < limit,
            "scientific directory census exceeds bound"
        );
        files.push(entry?.path());
    }
    files.sort();
    Ok(files)
}
fn graph_output_rows(
    id: &str,
    result: &crate::native_graph::VerifiedResult,
    v: &Value,
    qualification: bool,
    at: u64,
) -> Result<Rows> {
    let mut rows = Rows::default();
    let label = if qualification {
        "qualification"
    } else {
        "production"
    };
    rows.evidence.push(row_evidence(
        id,
        &result.output,
        "independently_verified_native_graph",
        None,
        at,
    ));
    match v["schema"]
        .as_str()
        .context("graph scientific output schema absent")?
    {
        "jope.smartcluster_training_completion.v1" => {
            let step = v["latest"]["step"]
                .as_u64()
                .context("checkpoint step absent")?;
            ensure!(
                v["window"]["target_optimizer_step"] == step
                    && v["numerical_receipt"]["steps"] == step,
                "verified training output window differs"
            );
            rows.checkpoints.push(json!({"identity":v["latest"]["state_sha256"],"step":step,"status":format!("{label} committed training checkpoint"),
                "reason":text(format!("Native receipt {}; quality acceptance and serving promotion are separate",result.receipt_id)),
                "qualification":qualification,"native_attempt":result.native_attempt,"source":result.output,"original_at_unix_ms":null}));
        }
        "jope.native_genome_completion.v1" => {
            rows.notes.push(text(format!(
                "{label} verified genome task {} stage {}; exact native receipt {}",
                result.task_id, v["stage"], result.receipt_id
            )));
        }
        _ => rows.notes.push(text(format!(
            "{label} verified {} output {}; native receipt {}",
            result.task_id, v["schema"], result.receipt_id
        ))),
    }
    stamp_rows(&mut rows, at);
    Ok(rows)
}
fn comparison_rows(
    v: &Value,
    generation: u64,
    reference: &Artifact,
    at: u64,
) -> Result<Vec<Value>> {
    ensure!(
        v["schema"] == "jope.native_matched_comparisons.v1"
            && v["generation"] == generation
            && v["operational_progress_is_not_model_improvement"] == true,
        "comparison schema/generation differs"
    );
    let mut rows = Vec::new();
    for name in ["generic", "shuffled"] {
        let c = &v["model_comparisons"][name];
        ensure!(
            c["independent_sampling_units_validated"] == false
                && c["independent_final_test"] == false,
            "comparison independence claims unsupported"
        );
        let wins = c["wins"].as_u64().context("comparison wins absent")?;
        let losses = c["losses"].as_u64().context("comparison losses absent")?;
        let n = wins
            .checked_add(losses)
            .context("comparison count overflow")?;
        ensure!(c["decisive_pairs"] == n, "comparison denominator differs");
        let rate = if n == 0 {
            None
        } else {
            Some(wins as f64 / n as f64)
        };
        ensure!(
            if let Some(rate) = rate {
                c["win_rate"]
                    .as_f64()
                    .is_some_and(|r| (r - rate).abs() < 1e-12)
            } else {
                c["win_rate"].is_null()
            },
            "comparison rate differs from original counts"
        );
        let (lower, upper) = if n == 0 {
            (None, None)
        } else {
            let lo = c["nominal_unclustered_wilson_95_interval"][0]
                .as_f64()
                .context("comparison lower interval absent")?;
            let hi = c["nominal_unclustered_wilson_95_interval"][1]
                .as_f64()
                .context("comparison upper interval absent")?;
            ensure!(
                (0.0..=1.0).contains(&lo) && (lo..=1.0).contains(&hi),
                "comparison interval invalid"
            );
            let nf = n as f64;
            let p = wins as f64 / nf;
            let z = 1.959963984540054;
            let den = 1.0 + z * z / nf;
            let center = (p + z * z / (2.0 * nf)) / den;
            let radius = z * ((p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt()) / den;
            ensure!(
                (lo - (center - radius)).abs() < 1e-9 && (hi - (center + radius)).abs() < 1e-9,
                "recorded Wilson interval differs from original counts"
            );
            (Some(lo), Some(hi))
        };
        rows.push(json!({"name":format!("g{generation} model vs {name} (recorded paired descriptive; unclustered interval)"),
            "wins":wins,"trials":n,"win_rate":rate,"lower":lower,"upper":upper,"confidence":if n>0{Some(0.95)}else{None},
            "ties":c["ties"],"unavailable":c["unavailable"],"source":reference,"scientific_verified_at_unix_ms":at,
            "verification_kind":"recorded Loop comparison bound to verified training decision; count arithmetic checked","independent_model_improvement_claim":false}));
    }
    Ok(rows)
}
fn decision_rows(
    root: &Path,
    loop_state: &Path,
    config: &crate::native_loop::Config,
    result: &crate::native_graph::VerifiedResult,
    result_ref: &Artifact,
    output: &Value,
    at: u64,
) -> Result<Rows> {
    let mut rows = Rows::default();
    let path = loop_state
        .join("parent-decisions")
        .join(format!("g{}.json", result.generation));
    if !path.exists() {
        rows.notes.push(text(format!(
            "{}: committed checkpoint awaits a separate proposal-parent quality decision",
            result.task_id
        )));
        return Ok(rows);
    }
    let reference: Artifact = serde_json::from_slice(&regular(&path, 1 << 20)?)?;
    let decision: crate::native_loop::ProposalParentDecision =
        serde_json::from_value(checked(Some(root), &reference, 1 << 20)?)?;
    ensure!(
        decision.run == config.run
            && decision.generation == result.generation
            && decision.training_result == *result_ref
            && output["latest"]["state_sha256"] == decision.checkpoint.sha256,
        "quality decision does not bind verified original training"
    );
    let assessment = checked(Some(root), &decision.assessment, 1 << 20)?;
    ensure!(
        assessment["schema"] == "jope.proposal_parent_quality_review.v1"
            && assessment["checkpoint"] == serde_json::to_value(&decision.checkpoint)?
            && assessment["training_result"] == serde_json::to_value(result_ref)?
            && assessment["comparison"] == serde_json::to_value(&decision.comparison)?
            && assessment["accepted"] == decision.accepted
            && assessment["sealed_test_used"] == false
            && assessment["serving_promotion"] == false,
        "quality assessment binding/partition differs"
    );
    rows.checkpoints.push(json!({"identity":decision.checkpoint.sha256,"status":if decision.accepted{"recorded accepted proposal parent"}else{"recorded rejected proposal parent"},
        "step":output["latest"]["step"],"reason":text(format!("Original assessment {}; not serving promotion; reviewer statement is recorded owner evidence",decision.assessment.sha256)),
        "source":reference,"qualification":config.qualification,"scientific_verified_at_unix_ms":at}));
    let comparison = checked(Some(root), &decision.comparison, 16 << 20)?;
    rows.comparisons = comparison_rows(&comparison, result.generation, &decision.comparison, at)?;
    Ok(rows)
}
async fn graph_run(
    id: &str,
    root: &Path,
    state: &Path,
    worker: &Path,
    loop_state: &Path,
    config: &Artifact,
    client: &Client,
    tenant: TenantId,
    project: ProjectId,
) -> Result<Rows> {
    let cv = checked(Some(root), config, 16 * 1024 * 1024)?;
    let config: crate::native_loop::Config = serde_json::from_value(cv)?;
    let graph = Graph::read_only(root, state, worker, tenant, project)?;
    let mut rows = Rows::default();
    let health_path = loop_state.join("health.json");
    ensure!(loop_state.starts_with(root), "loop state outside root");
    if health_path.exists() {
        let health: Value = serde_json::from_slice(&regular(&health_path, 1 << 20)?)?;
        ensure!(
            health["schema"] == "jope.native_loop_health.v1"
                && health["model_improvement_claimed"] == false,
            "loop health schema differs"
        );
        let stamp = health["timestamp_unix_ms"]
            .as_u64()
            .context("original loop health timestamp absent")?;
        original_time(Some(stamp), now()?)?;
        rows.notes.push(text(format!("Loop {} original health={} stale={} progress={}; declared operational progress, not model improvement",config.run,stamp,now()?.saturating_sub(stamp)>=STALE_MS,health["progress"])));
    }
    let tasks = bounded_files(&state.join("tasks"), 16384)?;
    // Bounded display sample; immutable original results remain in the graph.
    let mut candidates = Vec::new();
    for task in tasks {
        let p = task.join("verified.json");
        if p.is_file() {
            let meta = fs::metadata(&p)?;
            candidates.push((meta.modified()?, p));
        }
    }
    candidates.sort_by_key(|b| std::cmp::Reverse(b.0));
    if candidates.len() > 8 {
        rows.notes.push(text(format!(
            "{id}: showing 8 of {} recorded graph results; selection mtime is not completion time",
            candidates.len()
        )));
    }
    for (_, path) in candidates.into_iter().take(8) {
        let bytes = regular(&path, 1 << 20)?;
        let a = Artifact {
            path: path.strip_prefix(root)?.into(),
            sha256: hash(&bytes),
        };
        match graph.verified(client, &a).await {
            Ok(result) => {
                ensure!(
                    result.run == config.run,
                    "scientific graph run differs from pinned config"
                );
                let output = checked(Some(root), &result.output, MAX_METADATA)?;
                merge(
                    &mut rows,
                    graph_output_rows(id, &result, &output, config.qualification, now()?)?,
                );
                if output["schema"] == "jope.smartcluster_training_completion.v1" {
                    merge(
                        &mut rows,
                        decision_rows(root, loop_state, &config, &result, &a, &output, now()?)?,
                    );
                }
                if output["stage"] == "continuation" {
                    let search = checked(Some(root), &artifact(&output["search"])?, 16 << 20)?;
                    ensure!(
                        search["schema"] == "parallel.search.v1" && search["terminal"] == true,
                        "continuation search is not complete"
                    );
                    rows.continuations.push(json!({"generation":format!("{} g{} {}",if config.qualification{"qualification"}else{"production"},result.generation,result.task_id),
                        "completed":search["successful_measurements"],"failed":search["failed_or_unavailable_measurements"],"gain":null,"cost_seconds":finite(&search["wall_seconds"]),
                        "receipt_id":result.receipt_id,"source":result.output,"qualification":config.qualification,"scientific_verified_at_unix_ms":now()?,"original_at_unix_ms":search["completed_at"].as_u64().and_then(|n|n.checked_mul(1000))}));
                }
                if output["local_corpus_committed"] == true {
                    let index = artifact(&output["corpus_index"])?;
                    let corpus = checked(Some(root), &index, MAX_METADATA)?;
                    rows.corpus = Some(
                        json!({"sha256":index.sha256,"records":corpus["programs"].as_u64().or_else(||corpus["records"].as_u64()),"added_records":null,
                        "source":result.output,"qualification":config.qualification,"scientific_verified_at_unix_ms":now()?,"local_corpus_committed":true,"remote_verification_kind":"recorded original receipt; no fresh remote probe","remote_verified_recorded":output["remote_verified"]}),
                    );
                }
                if output["schema"] == "jain.datasets_service_receipt.v1" {
                    let captures = output["captures"]
                        .as_array()
                        .context("verified capture rows absent")?;
                    ensure!(
                        captures.len() <= 4096,
                        "verified capture rows exceed observer bound"
                    );
                    for capture in captures {
                        if let Some(event) = capture["evaluation"]
                            .as_str()
                            .and_then(|s| s.strip_prefix("champion-event:"))
                        {
                            if let Some(mut status) = publication_status(event, &output)? {
                                status["scientific_verified_at_unix_ms"] = json!(now()?);
                                status["source"] = serde_json::to_value(&result.output)?;
                                rows.publications.insert(event.into(), status);
                            }
                        }
                    }
                }
            }
            Err(e) => rows.errors.push(text(format!(
                "{id} graph result {} refused: {e:#}",
                a.sha256
            ))),
        }
    }
    Ok(rows)
}
fn publication_status(event: &str, receipt: &Value) -> Result<Option<Value>> {
    ensure!(
        receipt["schema"] == "jain.datasets_service_receipt.v1",
        "unsupported winner publication receipt"
    );
    let captures = receipt["captures"]
        .as_array()
        .context("capture rows absent")?;
    ensure!(
        captures.len() <= 4096,
        "capture metadata count exceeds bound"
    );
    let matching = captures
        .iter()
        .filter(|c| c["evaluation"] == format!("champion-event:{event}"))
        .collect::<Vec<_>>();
    ensure!(
        matching.len() <= 1,
        "winner appears twice in one capture receipt"
    );
    let Some(capture) = matching.first() else {
        return Ok(None);
    };
    ensure!(
        capture["acknowledged"] == true,
        "winner capture has no durable acknowledgement"
    );
    let publications = receipt["publications"]
        .as_array()
        .context("publication rows absent")?;
    ensure!(
        publications.len() <= 4096,
        "publication count exceeds bound"
    );
    let matching = publications
        .iter()
        .filter(|p| {
            p["producer"] == capture["producer"]
                && p["sequence"] == capture["sequence"]
                && p["envelope_sha256"] == capture["envelope_sha256"]
        })
        .collect::<Vec<_>>();
    ensure!(
        matching.len() <= 1,
        "conflicting winner publication bindings"
    );
    let Some(p) = matching.first() else {
        return Ok(Some(
            json!({"local_commit":null,"status":"original capture acknowledged; local commit pending","remote_verified_recorded":null,"capture_sequence":capture["sequence"]}),
        ));
    };
    let chain_ref = Artifact {
        path: p["chain_manifest_path"]
            .as_str()
            .context("chain path absent")?
            .into(),
        sha256: p["chain_manifest_sha256"]
            .as_str()
            .context("chain hash absent")?
            .into(),
    };
    let chain = checked(None, &chain_ref, 16 << 20)?;
    ensure!(
        chain["schema"] == "jain.publication.chain.v1"
            && chain["producer"] == capture["producer"]
            && chain["sequence"] == capture["sequence"]
            && chain["envelope_sha256"] == capture["envelope_sha256"]
            && chain["final_event_commit"] == p["final_event_commit"]
            && chain["final_event_ref"] == p["final_event_ref"],
        "original winner chain identity differs"
    );
    let commit = chain["final_event_commit"]
        .as_str()
        .context("local commit absent")?;
    ensure!(
        [40, 64].contains(&commit.len()) && commit.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid recorded commit identity"
    );
    let published = chain_ref.path.with_file_name("published-chain.json");
    let mut verified = false;
    if published.exists() {
        let v: Value = serde_json::from_slice(&regular(&published, 16 << 20)?)?;
        ensure!(
            v["schema"] == "jain.publication.chain_receipt.v1"
                && v["verified_download"] == true
                && v["chain_manifest_sha256"] == chain_ref.sha256
                && v["producer"] == capture["producer"]
                && v["sequence"] == capture["sequence"]
                && v["envelope_sha256"] == capture["envelope_sha256"]
                && v["final_event_commit"] == commit
                && v["final_event_ref"] == chain["final_event_ref"],
            "recorded remote chain verification differs"
        );
        let parts = v["parts"]
            .as_array()
            .context("remote part receipts absent")?;
        let batches = chain["batches"]
            .as_array()
            .context("original dependency batches absent")?;
        ensure!(
            !parts.is_empty() && parts.len() == batches.len() && parts.len() <= 4096,
            "remote dependency restoration count differs"
        );
        for (part, batch) in parts.iter().zip(batches) {
            ensure!(
                part["schema"] == "jain.publication.receipt.v1"
                    && part["verified_download"] == true
                    && part["commit"] == batch["commit"]
                    && part["batch_sha256"] == batch["batch_sha256"],
                "remote dependency restoration identity differs"
            );
            original_time(
                Some(
                    part["verified_at_unix_ms"]
                        .as_u64()
                        .context("original remote verification timestamp absent")?,
                ),
                now()?,
            )?;
        }
        verified = true;
    }
    Ok(Some(
        json!({"local_commit":commit,"status":if verified{"recorded original remote restoration; no fresh forge probe"}else{"recorded local commit; remote verification pending"},
        "remote_verified_recorded":verified,"capture_sequence":capture["sequence"]}),
    ))
}
fn winners(id: &str, root: &Path, outbox: &Path, receipts: &[Artifact]) -> Result<Rows> {
    ensure!(outbox.starts_with(root), "winner outbox outside root");
    let all = bounded_files(&outbox.join("by-sequence"), 100000)?;
    ensure!(
        receipts.len() <= 64,
        "winner service receipt bound exceeded"
    );
    let receipts = receipts
        .iter()
        .map(|a| checked(None, a, MAX_METADATA).map(|v| (a.clone(), v)))
        .collect::<Result<Vec<_>>>()?;
    let mut rows = Rows::default();
    rows.notes.push(text(format!("{id}: ordered winner outbox {} identities; displaying latest16; recorded publication never implies fresh forge readback",all.len())));
    for path in all.into_iter().rev().take(16) {
        let name = path
            .file_stem()
            .and_then(|v| v.to_str())
            .context("winner sequence name")?;
        ensure!(
            name.len() == 20 && name.bytes().all(|b| b.is_ascii_digit()),
            "invalid winner sequence identity"
        );
        let digest: String = serde_json::from_slice(&regular(&path, 1024)?)?;
        let a = Artifact {
            path: outbox.join(format!("{name}-{digest}.json")),
            sha256: digest.clone(),
        };
        let raw = checked(None, &a, 1 << 20)?;
        let c: crate::champions::Champion = serde_json::from_value(raw)?;
        c.validate()?;
        ensure!(
            matches!(
                c.identity.partition.as_str(),
                "meta_train" | "meta_validation"
            ),
            "sealed winner remains opaque"
        );
        let back: u64 = serde_json::from_slice(&regular(
            &outbox.join("by-event").join(format!("{digest}.json")),
            1024,
        )?)?;
        ensure!(
            back == name.parse::<u64>()?,
            "winner ordering reverse identity differs"
        );
        let original = c
            .timestamp
            .checked_mul(1000)
            .context("winner timestamp overflow")?;
        original_time(Some(original), now()?)?;
        let mut row = json!({"identity":digest,"local_commit":null,"publication_status":"original ordered winner; matching publication receipt unavailable","remote_verified":null,
            "age_seconds":now()?.saturating_sub(original)/1000,"sequence":back,"original_at_unix_ms":original,"scientific_verified_at_unix_ms":now()?,"source":a,"verification_kind":"champion identity/hash/order checked; authority provenance not independently replayed"});
        for (source, receipt) in &receipts {
            if let Some(status) = publication_status(&digest, receipt)? {
                row["local_commit"] = status["local_commit"].clone();
                row["publication_status"] = status["status"].clone();
                row["remote_verified_recorded"] = status["remote_verified_recorded"].clone();
                row["publication_source"] = serde_json::to_value(source)?;
                row["capture_sequence"] = status["capture_sequence"].clone();
            }
        }
        rows.winners.push(row);
    }
    Ok(rows)
}
fn catalog() -> Result<Catalog> {
    let bytes = crate::eval_registry::native_scq::policy::protected(Path::new(CATALOG))?;
    ensure!(bytes.len() <= 1 << 20, "science catalog exceeds1MiB");
    let catalog: Catalog = serde_json::from_slice(&bytes)?;
    ensure!(
        catalog.schema == "jope.native_scientific_catalog.v1" && catalog.entries.len() <= 64,
        "science catalog schema/count invalid"
    );
    let mut ids = BTreeSet::new();
    for source in &catalog.entries {
        ensure!(
            !source.id().is_empty() && text(source.id()) == source.id() && ids.insert(source.id()),
            "science source identity duplicate/invalid"
        );
    }
    Ok(catalog)
}
impl Reader {
    pub fn start(config: &Config) -> Result<Self> {
        let latest = Arc::new(Mutex::new(Observed::default()));
        let shared = latest.clone();
        let config = config.clone();
        thread::Builder::new()
            .name("jope-science-reader".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(_) => return,
                };
                loop {
                    let mut rows = Rows::default();
                    match catalog() {
                        Ok(c) => {
                            for source in c.entries {
                                let result = match &source {
                                    Source::RecordedQualification {
                                        id,
                                        artifact,
                                        original_at_unix_ms,
                                    } => checked(None, artifact, 16 << 20).and_then(|v| {
                                        qualification_rows(
                                            id,
                                            artifact,
                                            &v,
                                            *original_at_unix_ms,
                                            now()?,
                                        )
                                    }),
                                    Source::GraphRun {
                                        id,
                                        root,
                                        graph_state,
                                        worker_state,
                                        loop_state,
                                        config: a,
                                    } => rt.block_on(graph_run(
                                        id,
                                        root,
                                        graph_state,
                                        worker_state,
                                        loop_state,
                                        a,
                                        &Client::new(&config.socket),
                                        config.tenant.clone(),
                                        config.project.clone(),
                                    )),
                                    Source::OrderedWinners {
                                        id,
                                        root,
                                        outbox,
                                        service_receipts,
                                    } => winners(id, root, outbox, service_receipts),
                                };
                                match result {
                                    Ok(r) => merge(&mut rows, r),
                                    Err(e) => rows.errors.push(text(format!(
                                        "Science {} unavailable: {e:#}",
                                        source.id()
                                    ))),
                                }
                                // Publish partial progress so one slow original closure cannot hide prior results.
                                if let Ok(mut value) = shared.lock() {
                                    *value = Observed {
                                        at: now().ok(),
                                        rows: rows.clone(),
                                    };
                                }
                            }
                        }
                        Err(e) => rows.errors.push(text(format!(
                            "Protected scientific catalog unavailable: {e:#}"
                        ))),
                    }
                    if let Ok(mut value) = shared.lock() {
                        *value = Observed {
                            at: now().ok(),
                            rows,
                        };
                    }
                    thread::sleep(Duration::from_secs(10));
                }
            })?;
        Ok(Self { latest })
    }
    pub fn enrich(&self, snapshot: &mut Snapshot) {
        // Operational Loop health is a bounded metadata read on the sampling
        // thread, independent of potentially long scientific closure verification.
        match catalog() {
            Ok(c) => {
                for source in c.entries {
                    if let Source::GraphRun {
                        id,
                        root,
                        loop_state,
                        ..
                    } = source
                    {
                        let result = (|| -> Result<String> {
                            ensure!(
                                loop_state.starts_with(&root),
                                "loop health outside pinned root"
                            );
                            let health: Value = serde_json::from_slice(&regular(
                                &loop_state.join("health.json"),
                                1 << 20,
                            )?)?;
                            ensure!(
                                health["schema"] == "jope.native_loop_health.v1"
                                    && health["model_improvement_claimed"] == false,
                                "loop health schema differs"
                            );
                            let stamp = health["timestamp_unix_ms"]
                                .as_u64()
                                .context("loop original health timestamp absent")?;
                            original_time(Some(stamp), snapshot.observed_at_unix_ms)?;
                            Ok(text(format!("Loop {id}: original health={stamp}, stale={}, declared progress={}",snapshot.observed_at_unix_ms.saturating_sub(stamp)>=STALE_MS,health["progress"])))
                        })();
                        match result {
                            Ok(note) => snapshot.notes.push(note),
                            Err(e) => snapshot
                                .source_errors
                                .push(text(format!("Loop {id} health unavailable: {e:#}"))),
                        }
                    }
                }
            }
            Err(e) => snapshot.source_errors.push(text(format!(
                "Science catalog currently unavailable: {e:#}"
            ))),
        }
        self.enrich_science(snapshot);
    }
    fn enrich_science(&self, snapshot: &mut Snapshot) {
        let Ok(observed) = self.latest.lock().map(|v| v.clone()) else {
            return;
        };
        let stale = observed
            .at
            .is_none_or(|t| snapshot.observed_at_unix_ms.saturating_sub(t) >= STALE_MS);
        let mut rows = observed.rows;
        rows.notes.retain(|s| !s.starts_with("Loop "));
        for winner in &mut rows.winners {
            if let Some(status) = winner["identity"]
                .as_str()
                .and_then(|id| rows.publications.get(id))
            {
                winner["local_commit"] = status["local_commit"].clone();
                winner["publication_status"] = status["status"].clone();
                winner["remote_verified_recorded"] = status["remote_verified_recorded"].clone();
                winner["publication_source"] = status["source"].clone();
                winner["capture_sequence"] = status["capture_sequence"].clone();
                winner["scientific_verified_at_unix_ms"] =
                    status["scientific_verified_at_unix_ms"].clone();
            }
        }
        snapshot.notes.retain(|s| {
            !s.starts_with("No independently verified scientific receipt feed configured")
        });
        snapshot.notes.push(text(format!("Scientific verification last={:?}, stale={stale}; historical evidence is not live work or model improvement",observed.at)));
        let expired = |row: &Value| {
            stale
                || row["scientific_verified_at_unix_ms"]
                    .as_u64()
                    .is_none_or(|t| snapshot.observed_at_unix_ms.saturating_sub(t) >= STALE_MS)
        };
        for c in &mut rows.checkpoints {
            if expired(c) {
                c["status"] = json!(text(format!(
                    "STALE science: {}",
                    c["status"].as_str().unwrap_or("unknown")
                )));
            }
        }
        for c in &mut rows.continuations {
            if expired(c) {
                c["generation"] = json!(text(format!(
                    "STALE science: {}",
                    c["generation"].as_str().unwrap_or("unknown")
                )));
            }
        }
        for c in &mut rows.comparisons {
            if expired(c) {
                c["name"] = json!(text(format!(
                    "STALE science: {}",
                    c["name"].as_str().unwrap_or("unknown")
                )));
            }
        }
        for w in &mut rows.winners {
            if expired(w) {
                w["publication_status"] =
                    json!("STALE science; previous original publication evidence retained");
                w["remote_verified"] = Value::Null;
            }
        }
        if rows.corpus.as_ref().is_some_and(expired) {
            rows.notes
                .push("Current corpus withheld because its science sample is stale".into());
            rows.corpus = None;
        }
        for e in &mut rows.evidence {
            e["stale"] = json!(e["verified_at_unix_ms"]
                .as_u64()
                .is_none_or(|t| snapshot.observed_at_unix_ms.saturating_sub(t) >= STALE_MS));
        }
        snapshot.checkpoints = rows.checkpoints.into_iter().take(64).collect();
        snapshot.continuations = rows.continuations.into_iter().take(64).collect();
        snapshot.comparisons = rows.comparisons.into_iter().take(32).collect();
        snapshot.winners = rows.winners.into_iter().take(64).collect();
        snapshot.corpus = rows.corpus;
        snapshot.notes.extend(rows.notes);
        snapshot.notes.truncate(64);
        snapshot.source_errors.extend(rows.errors);
        snapshot.source_errors.truncate(32);
        snapshot.scientific_sources = rows.evidence.into_iter().take(64).collect();
        snapshot.scientific_verified_at_unix_ms = observed.at;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_fresh_source_does_not_hide_another_stale_scientific_row() {
        let rows = Rows {
            checkpoints: vec![
                json!({"identity":"old","status":"committed","scientific_verified_at_unix_ms":1000}),
                json!({"identity":"recent","status":"committed","scientific_verified_at_unix_ms":39000}),
            ],
            corpus: Some(json!({"sha256":"a".repeat(64),"scientific_verified_at_unix_ms":1000})),
            ..Default::default()
        };
        let reader = Reader {
            latest: Arc::new(Mutex::new(Observed {
                at: Some(40000),
                rows,
            })),
        };
        let mut snapshot = Snapshot {
            schema: "jain.jope_native_status.v1",
            observed_at_unix_ms: 40000,
            scope: "test".into(),
            resources: vec![],
            queues: vec![],
            continuations: vec![],
            corpus: None,
            checkpoints: vec![],
            comparisons: vec![],
            winners: vec![],
            notes: vec![],
            source_errors: vec![],
            scientific_sources: vec![],
            scientific_verified_at_unix_ms: None,
        };
        reader.enrich_science(&mut snapshot);
        assert!(snapshot.checkpoints[0]["status"]
            .as_str()
            .unwrap()
            .starts_with("STALE"));
        assert_eq!(snapshot.checkpoints[1]["status"], "committed");
        assert!(snapshot.corpus.is_none());
        assert_eq!(snapshot.observed_at_unix_ms, 40000);
    }
    #[test]
    fn descriptive_comparison_counts_cannot_become_independent_evidence() {
        let c = json!({"wins":1,"losses":1,"ties":0,"unavailable":0,"decisive_pairs":2,"win_rate":0.5,"nominal_unclustered_wilson_95_interval":[0.09453120573423074,0.9054687942657693],"independent_sampling_units_validated":false,"independent_final_test":false});
        let mut v = json!({"schema":"jope.native_matched_comparisons.v1","generation":0,"operational_progress_is_not_model_improvement":true,"model_comparisons":{"generic":c,"shuffled":c}});
        let rows = comparison_rows(&v, 0, &reference(), 1000).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["independent_model_improvement_claim"], false);
        v["model_comparisons"]["generic"]["win_rate"] = json!(0.9);
        assert!(comparison_rows(&v, 0, &reference(), 1000).is_err());
        v["model_comparisons"]["generic"]["win_rate"] = json!(0.5);
        v["model_comparisons"]["generic"]["independent_final_test"] = json!(true);
        assert!(comparison_rows(&v, 0, &reference(), 1000).is_err());
    }
    #[test]
    fn ordered_winner_joins_capture_identity_not_outbox_sequence() {
        let event = "a".repeat(64);
        let mut v = json!({"schema":"jain.datasets_service_receipt.v1","captures":[{"evaluation":format!("champion-event:{event}"),"producer":"original","sequence":71,"envelope_sha256":"b".repeat(64),"acknowledged":true}],"publications":[]});
        let status = publication_status(&event, &v).unwrap().unwrap();
        assert_eq!(status["capture_sequence"], 71);
        assert!(status["local_commit"].is_null());
        assert!(status["remote_verified_recorded"].is_null());
        assert!(publication_status(&"c".repeat(64), &v).unwrap().is_none());
        let duplicate = v["captures"][0].clone();
        v["captures"].as_array_mut().unwrap().push(duplicate);
        assert!(publication_status(&event, &v).is_err());
    }
    #[test]
    fn remote_record_must_bind_exact_original_chain_and_every_part() {
        let t = tempfile::tempdir().unwrap();
        let path = t.path().join("chain.json");
        let event = "a".repeat(64);
        let commit = "c".repeat(40);
        let chain = json!({"schema":"jain.publication.chain.v1","producer":"p","sequence":71,"envelope_sha256":"b".repeat(64),"final_event_commit":commit,"final_event_ref":"refs/events/71","batches":[{"commit":commit,"batch_sha256":"d".repeat(64)}]});
        let bytes = serde_json::to_vec(&chain).unwrap();
        fs::write(&path, &bytes).unwrap();
        let digest = hash(&bytes);
        let mut prepared = chain.clone();
        prepared["chain_manifest_path"] = json!(path);
        prepared["chain_manifest_sha256"] = json!(digest);
        let receipt = json!({"schema":"jain.datasets_service_receipt.v1","captures":[{"evaluation":format!("champion-event:{event}"),"producer":"p","sequence":71,"envelope_sha256":"b".repeat(64),"acknowledged":true}],"publications":[prepared]});
        assert_eq!(
            publication_status(&event, &receipt).unwrap().unwrap()["remote_verified_recorded"],
            false
        );
        let mut remote = json!({"schema":"jain.publication.chain_receipt.v1","chain_manifest_sha256":digest,"producer":"p","sequence":71,"envelope_sha256":"b".repeat(64),"final_event_commit":commit,"final_event_ref":"refs/events/71","verified_download":true,"parts":[{"schema":"jain.publication.receipt.v1","commit":commit,"batch_sha256":"d".repeat(64),"verified_download":true,"verified_at_unix_ms":1000}]});
        let p = t.path().join("published-chain.json");
        fs::write(&p, serde_json::to_vec(&remote).unwrap()).unwrap();
        assert_eq!(
            publication_status(&event, &receipt).unwrap().unwrap()["remote_verified_recorded"],
            true
        );
        remote["sequence"] = json!(72);
        fs::write(&p, serde_json::to_vec(&remote).unwrap()).unwrap();
        assert!(publication_status(&event, &receipt).is_err());
    }
    fn reference() -> Artifact {
        Artifact {
            path: "/immutable/qualification.json".into(),
            sha256: "a".repeat(64),
        }
    }
    #[test]
    fn qualification_checkpoint_is_never_production_or_promoted() {
        let v = json!({"schema":"jope.hardware_training_benchmark_verified.v1","production_learning":false,"numerical_result_verified":true,"initial_optimizer_step":60000,"target_optimizer_step":60010,"exact_updates":10,"checkpoint":{"path":"/original/state.json","sha256":"b".repeat(64)},"microbatch":256,"full_scheduler_lifecycle_passed":false});
        let r = qualification_rows("original", &reference(), &v, Some(1000), 2000).unwrap();
        assert_eq!(r.checkpoints[0]["qualification"], true);
        assert_eq!(r.checkpoints[0]["original_at_unix_ms"], 1000);
        assert!(r.comparisons.is_empty());
        let mut bad = v;
        bad["production_learning"] = json!(true);
        assert!(qualification_rows("bad", &reference(), &bad, None, 2000).is_err());
    }
    #[test]
    fn exact_eof_window_and_original_time_are_required() {
        let mut v = json!({"schema":"jope.actual_neural_eof_recovery_qualification.v1","production_learning":false,"passed":true,"unknown_final_tail":false,"original_initial_step":60000,"original_target_step":60100,"first_attempt_committed_updates":6,"retry_executed_updates":94,"total_updates":100,"final_checkpoint":{"path":"/old/state.json","sha256":"b".repeat(64)}});
        assert!(qualification_rows("q", &reference(), &v, None, 1000).is_ok());
        v["retry_executed_updates"] = json!(100);
        assert!(qualification_rows("q", &reference(), &v, None, 1000).is_err());
        assert!(original_time(Some(9000), 1000).is_err());
    }
    #[test]
    fn corrupt_and_nonregular_source_never_becomes_scientific_success() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("evidence");
        fs::write(&p, b"{}").unwrap();
        let a = Artifact {
            path: p.clone(),
            sha256: hash(b"{}"),
        };
        assert!(checked(None, &a, 1024).is_ok());
        fs::write(&p, b"[]").unwrap();
        assert!(checked(None, &a, 1024).is_err());
        fs::remove_file(&p).unwrap();
        std::os::unix::fs::symlink("/dev/zero", &p).unwrap();
        assert!(checked(None, &a, 1024).is_err());
    }
}
