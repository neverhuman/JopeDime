use super::*;

/// Pair only actual same-cut/slot observations; unmatched valid outputs remain explicit.
pub(super) fn matched_comparisons(
    root: &Path,
    generation: u64,
    artifacts: &[Artifact],
) -> Result<Value> {
    let mut slots = BTreeMap::<(String, u64), BTreeMap<Arm, Artifact>>::new();
    for a in artifacts {
        let c: Candidate = native_genome::load(root, a)?;
        ensure!(c.generation == generation, "comparison generation differs");
        let key = (hash(&serde_json::to_vec(&c.context)?), c.matched_slot);
        ensure!(
            slots
                .entry(key)
                .or_default()
                .insert(c.arm, a.clone())
                .is_none(),
            "duplicate comparison arm/slot"
        );
    }
    let mut paired = Vec::new();
    let mut unmatched = Vec::new();
    for ((context, slot), arms) in slots {
        if arms.len() == 3 {
            paired.push(json!({"context_sha256":context,"slot":slot,"arms":arms}));
        } else {
            unmatched.push(json!({"context_sha256":context,"slot":slot,"arms":arms,"reason":"unequal_valid_output_counts","excluded_from_matched_statistics":true}));
        }
    }
    Ok(
        json!({"schema":"jope.native_matched_comparisons.v1","generation":generation,"matching":"same_original_cut_slot_and_numerical_budget","paired":paired,"unmatched":unmatched,"all_valid_candidates_preserved":true}),
    )
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalParentDecision {
    pub schema: String,
    pub run: String,
    pub generation: u64,
    pub checkpoint: Artifact,
    pub previous_parent: Artifact,
    pub training_result: Artifact,
    pub comparison: Artifact,
    pub assessment: Artifact,
    pub accepted: bool,
}
pub(super) fn verify_result_binding(
    request: &Request,
    admission: &Admission,
    result: &VerifiedResult,
) -> Result<()> {
    ensure!(
        admission.predecessors == request.predecessors
            && result.admission_sha256 == hash(&serde_json::to_vec(admission)?)
            && result.task_manifest_sha256 == admission.manifest.sha256
            && result.run == request.run
            && result.generation == request.generation
            && result.task_id == request.id,
        "graph result differs from exact original loop admission/predecessors"
    );
    Ok(())
}
/// Checkpoint and comparison references required by a parent decision.
#[derive(Clone, Copy)]
pub(super) struct ParentEvidence<'a> {
    pub cp: &'a Artifact,
    pub previous: &'a Artifact,
    pub train: &'a Artifact,
    pub comparison: &'a Artifact,
}

pub(super) fn verify_parent_decision(
    d: &ProposalParentDecision,
    a: &Value,
    run: &str,
    g: u64,
    evidence: ParentEvidence<'_>,
) -> Result<()> {
    let ParentEvidence {
        cp,
        previous,
        train,
        comparison,
    } = evidence;

    ensure!(
        d.schema == "jope.proposal_parent_decision.v1"
            && d.run == run
            && d.generation == g
            && &d.checkpoint == cp
            && &d.previous_parent == previous
            && &d.training_result == train
            && &d.comparison == comparison,
        "quality decision does not bind this original committed checkpoint and evidence"
    );
    ensure!(
        a["schema"] == "jope.proposal_parent_quality_review.v1"
            && a["checkpoint"] == serde_json::to_value(cp)?
            && a["previous_parent"] == serde_json::to_value(previous)?
            && a["training_result"] == serde_json::to_value(train)?
            && a["comparison"] == serde_json::to_value(comparison)?
            && a["accepted"] == d.accepted
            && a["sealed_test_used"] == false
            && a["serving_promotion"] == false
            && a["reviewer"].as_str().is_some_and(|s| !s.trim().is_empty())
            && a["rationale"]
                .as_str()
                .is_some_and(|s| !s.trim().is_empty()),
        "independent quality review absent or mixed with serving/sealed-test evidence"
    );
    Ok(())
}
fn comparison_outcomes(
    root: &Path,
    g: u64,
    targets: &[Artifact],
    candidates: &[Artifact],
    refusals: &Artifact,
) -> Result<Value> {
    let mut outcomes = BTreeMap::new();
    for target in targets {
        let value: crate::native_genome::ContinuationTarget = native_genome::load(root, target)?;
        ensure!(
            outcomes
                .insert(
                    value.candidate.sha256.clone(),
                    (target.clone(), value.clone())
                )
                .is_none(),
            "duplicate candidate outcome"
        );
    }
    let mut report = matched_comparisons(root, g, candidates)?;
    let mut comparisons = serde_json::Map::new();
    for opponent in ["generic", "shuffled"] {
        let (mut wins, mut losses, mut ties, mut unavailable) = (0u64, 0u64, 0u64, 0u64);
        for pair in report["paired"]
            .as_array()
            .context("paired comparison set")?
        {
            let model: Artifact = serde_json::from_value(pair["arms"]["model"].clone())?;
            let other: Artifact = serde_json::from_value(pair["arms"][opponent].clone())?;
            let left = outcomes
                .get(&model.sha256)
                .filter(|(_, target)| target.training_admission == "admitted")
                .and_then(|(_, target)| target.quality_rmse_cut_normalized);
            let right = outcomes
                .get(&other.sha256)
                .filter(|(_, target)| target.training_admission == "admitted")
                .and_then(|(_, target)| target.quality_rmse_cut_normalized);
            match (left, right) {
                (Some(a), Some(b)) if a < b => wins += 1,
                (Some(a), Some(b)) if a > b => losses += 1,
                (Some(_), Some(_)) => ties += 1,
                _ => unavailable += 1,
            }
        }
        let n = wins + losses;
        let interval = if n == 0 {
            Value::Null
        } else {
            let nf = n as f64;
            let p = wins as f64 / nf;
            let z = 1.959963984540054;
            let den = 1.0 + z * z / nf;
            let center = (p + z * z / (2.0 * nf)) / den;
            let radius = z * ((p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt()) / den;
            json!([center - radius, center + radius])
        };
        comparisons.insert(opponent.into(),json!({"wins":wins,"losses":losses,"ties":ties,"unavailable":unavailable,
            "decisive_pairs":n,"win_rate":if n==0 {Value::Null}else{json!(wins as f64/n as f64)},
            "nominal_unclustered_wilson_95_interval":interval,"independent_sampling_units_validated":false,"ties_excluded_from_decisive_rate":true,"independent_final_test":false}));
    }
    report["refusals"] = serde_json::to_value(refusals)?;
    report["outcomes"] = serde_json::to_value(outcomes)?;
    report["model_comparisons"] = Value::Object(comparisons);
    report["operational_progress_is_not_model_improvement"] = json!(true);
    Ok(report)
}

/// Eligibility replenishment, not resource scheduling. Native queued and running work both count.
/// Failures remain counted/visible until original recovery is resolved; no competing pool is started.
pub fn replenish(
    workers: usize,
    inflight_direct: usize,
    inflight_continuation: usize,
    mut direct: Vec<Request>,
    mut continuation: Vec<Request>,
) -> Vec<Request> {
    if inflight_direct.saturating_add(inflight_continuation) >= workers.saturating_mul(2) {
        return vec![];
    }
    direct.sort_by(|a, b| (a.generation, &a.id).cmp(&(b.generation, &b.id)));
    continuation.sort_by(|a, b| (a.generation, &a.id).cmp(&(b.generation, &b.id)));
    let slots = workers
        .saturating_mul(3)
        .saturating_sub(inflight_direct + inflight_continuation);
    let cap = if !direct.is_empty() && !continuation.is_empty() {
        (workers / 4).saturating_sub(inflight_continuation)
    } else {
        slots
    };
    let mut chosen: Vec<Request> = continuation.into_iter().take(cap.min(slots)).collect();
    chosen.extend(direct.into_iter().take(slots - chosen.len()));
    chosen
}
