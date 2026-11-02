use super::*;

pub(super) fn bounded_task_score(task: Task, score: f64) -> f64 {
    let score = if task == Task::Timeseries {
        score + 1.0
    } else {
        score
    };
    score.clamp(-1.0, 1.0)
}

pub(super) fn summarize(
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
    let receipts_replay_complete = receipts.iter().all(|path| {
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
        // Order statistics are only meaningful over equal-length vectors. When
        // an arm lost proposals to failures, the context is INCOMPLETE for that
        // comparison and is counted, not truncated: failures already lower the
        // arm's validity field, and letting them also shrink this vector let a
        // candidate's failures raise its measured lift (epoch-06 lane 05).
        if candidate_sorted.len() == accepted_sorted.len() {
            let gaps = candidate_sorted
                .iter()
                .zip(&accepted_sorted)
                .map(|(candidate, accepted)| candidate - accepted)
                .collect::<Vec<_>>();
            aggregate
                .candidate_context_comparison_gaps
                .push(mean(&gaps));
            aggregate.candidate_pairwise_gaps.extend(gaps);
        } else {
            aggregate.incomplete_accepted_contexts += 1;
        }
        if candidate_sorted.len() == gp_sorted.len() {
            let gaps = candidate_sorted
                .iter()
                .zip(&gp_sorted)
                .map(|(candidate, gp)| candidate - gp)
                .collect::<Vec<_>>();
            aggregate.gp_context_comparison_gaps.push(mean(&gaps));
            aggregate.gp_pairwise_gaps.extend(gaps);
        } else {
            aggregate.incomplete_gp_contexts += 1;
        }
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
    let accepted_lower_bound =
        registered_context_lower_bound(&aggregate.candidate_context_comparison_gaps);
    let gp_lower_bound = registered_context_lower_bound(&aggregate.gp_context_comparison_gaps);
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
    let matched_comparisons_complete =
        aggregate.incomplete_accepted_contexts == 0 && aggregate.incomplete_gp_contexts == 0;
    let replay_verification_complete = receipts_replay_complete && matched_comparisons_complete;
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
            "pairwise_lift_lower_bound": accepted_lower_bound,
            "stability": stability,
            "efficiency": efficiency,
            "latency_ratio": latency_ratio,
            "conditionality": conditional_lift,
        },
        "candidate_vs_gp": {
            "validity": candidate_validity.min(gp_validity),
            "pairwise_lift": gp_lift,
            "pairwise_lift_lower_bound": gp_lower_bound,
        },
        "comparison_lower_bound_method": "context_cluster_normal_95_v1",
        // Contexts whose arms returned unequal valid counts: their gaps are
        // EXCLUDED from the lift means above rather than silently truncated to
        // the shorter survivor vector. A reviewer reading pairwise_lift must be
        // able to see how much of the panel actually backs it.
        "incomplete_accepted_contexts": aggregate.incomplete_accepted_contexts,
        "incomplete_gp_contexts": aggregate.incomplete_gp_contexts,
        "matched_comparisons_complete": matched_comparisons_complete,
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
