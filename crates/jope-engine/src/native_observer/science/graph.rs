use super::recorded::{row_evidence, stamp_rows};
use super::*;

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
pub(super) fn comparison_rows(
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
/// Repository, graph-state and worker locations for a recorded run.
#[derive(Clone, Copy)]
pub(super) struct GraphLocation<'a> {
    pub root: &'a Path,
    pub state: &'a Path,
    pub worker: &'a Path,
}

pub(super) async fn graph_run(
    id: &str,
    location: GraphLocation<'_>,
    loop_state: &Path,
    config: &Artifact,
    client: &Client,
    tenant: TenantId,
    project: ProjectId,
) -> Result<Rows> {
    let GraphLocation {
        root,
        state,
        worker,
    } = location;

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
                        json!({"sha256":index.sha256,"records":corpus["programs"].as_u64() // default fallback
                            .or_else(||corpus["records"].as_u64()),"added_records":null,
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
