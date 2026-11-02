use super::*;

/// Port of `test_exact_accept.evaluate_gate`.
pub(super) fn evaluate_gate(evidence: &Value, accepted_by_task: Option<&Value>) -> Result<Value> {
    let baseline = &evidence["baseline"];
    let candidate = &evidence["candidate"];
    // Advisory-until-anchor: a branch with no accepted anchor is recorded but
    // never gates — its evidence fields are optional and its reasons stay out
    // of the global verdict entirely.
    let advisory_tasks: BTreeSet<&str> = TASK_BRANCHES
        .into_iter()
        .filter(|task| !task_verdict_is_gating(task, accepted_by_task))
        .collect();
    let macro_lift = get_f64(candidate, "macro_pairwise_accuracy")?
        - get_f64(baseline, "macro_pairwise_accuracy")?;
    let mut common_reasons: Vec<String> = Vec::new();
    let mut task_deltas = Map::new();
    // Branch bookkeeping for `task_verdicts`. These reasons still enter
    // `common_reasons` in the original order, so the global verdict is
    // bit-identical to the pre-change gate.
    let mut task_own_reasons: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    let mut task_delta_values: BTreeMap<&str, f64> = BTreeMap::new();
    for task in TASK_BRANCHES {
        let key = format!("{task}_pairwise_accuracy");
        if advisory_tasks.contains(task) {
            // Optional evidence; a regression here reaches only this branch's
            // own advisory verdict, never the global reasons.
            let delta = match (
                candidate
                    .get(&key)
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite()),
                baseline
                    .get(&key)
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite()),
            ) {
                (Some(candidate_value), Some(baseline_value)) => {
                    Some(candidate_value - baseline_value).filter(|value| value.is_finite())
                }
                _ => None,
            };
            if let Some(value) = delta {
                task_delta_values.insert(task, value);
                if value < -0.005 - 1e-12 {
                    task_own_reasons
                        .entry(task)
                        .or_default()
                        .push(format!("{task}_regressed_more_than_half_point"));
                }
            }
            task_deltas.insert(task.to_owned(), optional_number(delta));
            continue;
        }
        let delta = get_f64(candidate, &key)? - get_f64(baseline, &key)?;
        task_delta_values.insert(task, delta);
        task_deltas.insert(
            task.to_owned(),
            serde_json::Number::from_f64(delta)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
        if delta < -0.005 - 1e-12 {
            let reason = format!("{task}_regressed_more_than_half_point");
            task_own_reasons
                .entry(task)
                .or_default()
                .push(reason.clone());
            common_reasons.push(reason);
        }
    }
    if get_f64(evidence, "proposal_validity")? < 0.99 {
        common_reasons.push("proposal_validity_below_99_percent".into());
    }
    if evidence
        .get("verified_checkpoint_origin_strict_wins")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        < 1
    {
        common_reasons.push("no_verified_checkpoint_origin_strict_test_improvement".into());
    }
    let candidate_equal_budget = evidence
        .get("candidate_equal_budget_result")
        .and_then(Value::as_f64)
        .or_else(|| {
            candidate
                .get("equal_budget_proposal_result")
                .and_then(Value::as_f64)
        })
        .unwrap_or(f64::NEG_INFINITY);
    let baseline_equal_budget = evidence
        .get("baseline_equal_budget_result")
        .and_then(Value::as_f64)
        .or_else(|| {
            baseline
                .get("equal_budget_proposal_result")
                .and_then(Value::as_f64)
        })
        .unwrap_or(f64::INFINITY);
    if candidate_equal_budget <= baseline_equal_budget + 1e-12 {
        common_reasons.push("candidate_did_not_win_equal_budget_proposal_gate".into());
    }
    let candidate_failure = candidate
        .get("failure_rate")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let baseline_failure = baseline
        .get("failure_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let failure_delta = candidate_failure - baseline_failure;
    if failure_delta > 0.0025 + 1e-12 {
        common_reasons.push("candidate_failure_rate_regressed".into());
    }
    let candidate_latency = candidate
        .get("p95_proposal_latency_ms")
        .and_then(Value::as_f64)
        .unwrap_or(f64::INFINITY);
    let baseline_latency = baseline
        .get("p95_proposal_latency_ms")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let latency_ratio = if baseline_latency <= 0.0 {
        f64::INFINITY
    } else {
        candidate_latency / baseline_latency
    };
    let exception = evidence
        .get("latency_exception")
        .cloned()
        .unwrap_or(json!({}));
    let exception_valid = exception.get("approved") == Some(&Value::Bool(true))
        && exception
            .get("accuracy_gain")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            > exception
                .get("latency_penalty")
                .and_then(Value::as_f64)
                .unwrap_or(f64::INFINITY)
        && !exception
            .get("rationale")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .is_empty();
    if latency_ratio > 1.25 + 1e-12 && !exception_valid {
        common_reasons.push("candidate_p95_latency_exceeds_limit".into());
    }
    let replay = evidence
        .get("replay_verification")
        .cloned()
        .unwrap_or(json!({}));
    for check in [
        "predictions",
        "scores",
        "evaluator_identity",
        "dataset_hashes",
        "candidate_origin",
    ] {
        if replay.get(check) != Some(&Value::Bool(true)) {
            common_reasons.push(format!("replay_verification_failed:{check}"));
        }
    }
    // Conditionality gate. A checkpoint whose conditional reward head cannot beat
    // its own genome-only ablation is not conditioned on the dataset, so it can
    // never be promoted — no macro lift buys past this, at either tier.
    let candidate_conditional = conditional_advantage(candidate);
    let baseline_conditional = conditional_advantage(baseline);
    let conditional_lift = match (candidate_conditional, baseline_conditional) {
        (Some(candidate_value), Some(baseline_value)) => Some(candidate_value - baseline_value),
        _ => None,
    };
    if candidate_conditional.is_none() {
        common_reasons.push("candidate_conditional_advantage_evidence_missing".into());
    }
    if baseline_conditional.is_none() {
        common_reasons.push("baseline_conditional_advantage_evidence_missing".into());
    }
    if candidate_conditional.is_some_and(|value| value <= 1e-12) {
        common_reasons.push("candidate_policy_is_not_conditional".into());
    }
    let mut internal_reasons = common_reasons.clone();
    let mut milestone_reasons = common_reasons.clone();
    if macro_lift < 0.002 - 1e-12 {
        internal_reasons.push("macro_pairwise_lift_below_point_two".into());
    }
    if macro_lift < 0.01 - 1e-12 {
        milestone_reasons.push("macro_pairwise_lift_below_one_point".into());
    }
    if candidate_conditional.is_some_and(|value| value < CONDITIONAL_INTERNAL_MARGIN - 1e-12) {
        internal_reasons.push("conditional_advantage_below_half_point".into());
    }
    if conditional_lift.is_some_and(|value| value < CONDITIONAL_INTERNAL_MARGIN - 1e-12) {
        internal_reasons.push("conditional_advantage_lift_below_half_point".into());
    }
    if candidate_conditional.is_some_and(|value| value < CONDITIONAL_MILESTONE_MARGIN - 1e-12) {
        milestone_reasons.push("conditional_advantage_below_two_points".into());
    }
    if conditional_lift.is_some_and(|value| value < CONDITIONAL_MILESTONE_MARGIN - 1e-12) {
        milestone_reasons.push("conditional_advantage_lift_below_two_points".into());
    }
    let internal_pass = internal_reasons.is_empty();
    let milestone_pass = milestone_reasons.is_empty();
    let reasons: Vec<String> = if internal_pass {
        Vec::new()
    } else {
        internal_reasons
    };
    let tier = if milestone_pass {
        "milestone"
    } else if internal_pass {
        "internal"
    } else {
        "rejected"
    };
    if !latency_ratio.is_finite() {
        bail!("non-finite latency ratio cannot be recorded in gate output");
    }
    // Per-task verdicts. Derived from exactly the reasons the global verdict
    // already computed, minus the branch attribution: nothing here can make the
    // global gate laxer, and everything here is additive in the artifact.
    let branch_owned_reasons: BTreeSet<&String> = task_own_reasons.values().flatten().collect();
    let shared_reasons: Vec<String> = common_reasons
        .iter()
        .filter(|reason| !branch_owned_reasons.contains(reason))
        .cloned()
        .collect();
    let mut shared_internal_extra: Vec<String> = Vec::new();
    let mut shared_milestone_extra: Vec<String> = Vec::new();
    if candidate_conditional.is_some_and(|value| value < CONDITIONAL_INTERNAL_MARGIN - 1e-12) {
        shared_internal_extra.push("conditional_advantage_below_half_point".into());
    }
    if conditional_lift.is_some_and(|value| value < CONDITIONAL_INTERNAL_MARGIN - 1e-12) {
        shared_internal_extra.push("conditional_advantage_lift_below_half_point".into());
    }
    if candidate_conditional.is_some_and(|value| value < CONDITIONAL_MILESTONE_MARGIN - 1e-12) {
        shared_milestone_extra.push("conditional_advantage_below_two_points".into());
    }
    if conditional_lift.is_some_and(|value| value < CONDITIONAL_MILESTONE_MARGIN - 1e-12) {
        shared_milestone_extra.push("conditional_advantage_lift_below_two_points".into());
    }
    let verdicts = task_verdicts(
        baseline,
        candidate,
        &shared_reasons,
        &task_own_reasons,
        &task_delta_values,
        &shared_internal_extra,
        &shared_milestone_extra,
        common_reasons
            .iter()
            .any(|reason| reason == EQUAL_BUDGET_REASON),
        &advisory_tasks,
    );
    let passing: Vec<Value> = TASK_BRANCHES
        .iter()
        .filter(|task| verdicts[**task]["pass"] == Value::Bool(true))
        .map(|task| Value::String((*task).to_owned()))
        .collect();
    Ok(json!({
        "schema": "JopeSelfplayCheckpointGate.v3",
        "pass": internal_pass,
        "internal_learning_pass": internal_pass,
        "milestone_pass": milestone_pass,
        "promotion_tier": tier,
        "reasons": reasons,
        "milestone_reasons": milestone_reasons,
        "macro_pairwise_lift": macro_lift,
        "candidate_conditional_advantage": optional_number(candidate_conditional),
        "baseline_conditional_advantage": optional_number(baseline_conditional),
        "conditional_advantage_lift": optional_number(conditional_lift),
        "task_pairwise_deltas": task_deltas,
        "failure_rate_delta": failure_delta,
        "p95_latency_ratio": latency_ratio,
        "latency_exception_applied": latency_ratio > 1.25 && exception_valid,
        "model_lift_claim_publishable": false,
        "promotion_scope": "internal-selected-test-flywheel-only",
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "accepted_ancestor_sha256": baseline["checkpoint_sha256"],
        "candidate_equal_budget_result": candidate_equal_budget,
        "baseline_equal_budget_result": baseline_equal_budget,
        "task_verdicts": Value::Object(verdicts),
        "task_verdict_schema": "JopeSelfplayCheckpointGateTaskVerdict.v1",
        "tasks_passing": passing,
    }))
}
