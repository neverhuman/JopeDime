use super::*;

/// Conditional advantage = conditional reward head pairwise accuracy MINUS the
/// detached genome-only ablation control. <= 0 means the head is not using the
/// dataset embedding at all: it is a genome prior with noise on top. Key order
/// mirrors `test_exact_accept.CONDITIONAL_ADVANTAGE_KEYS`.
pub(super) const CONDITIONAL_ADVANTAGE_KEYS: [&str; 3] = [
    "val_conditional_advantage",
    "conditional_advantage",
    "selected_phase_b_conditional_advantage",
];
/// Margins in accuracy points, matching the macro_pairwise_lift convention.
pub(super) const CONDITIONAL_INTERNAL_MARGIN: f64 = 0.005;
pub(super) const CONDITIONAL_MILESTONE_MARGIN: f64 = 0.02;

/// The independent promotion branches, in the same order as
/// `selfplay_expansion::TASKS`. A generation that trades one against another
/// can never clear a single global gate — which is the observed state (gen-1
/// `ff95a336`: binary +0.0421, regression -0.0161, tier "rejected", and the
/// original anchor still accepted). `task_verdicts` records the decision each
/// branch reaches on its OWN evidence, so the branches can advance
/// independently. It never widens the global verdict: `pass`, `promotion_tier`
/// and `reasons` are computed exactly as before.
///
/// The NEW branches (timeseries, multiclass) are ADVISORY until anchored: see
/// `task_verdict_is_gating`. Keep in lockstep with
/// `test_exact_accept.TASK_BRANCHES` (same members, same order).
pub(super) const TASK_BRANCHES: [&str; 4] = ["regression", "binary", "timeseries", "multiclass"];
/// Per-task ports of the global lift margins. The global tier reads the macro
/// average; a task verdict reads that task's own pairwise lift.
pub(super) const TASK_INTERNAL_LIFT_MARGIN: f64 = 0.002;
pub(super) const TASK_MILESTONE_LIFT_MARGIN: f64 = 0.01;
/// Shared reasons a per-task verdict supersedes when the evidence carries the
/// task-scoped measurement (`failure_rate_by_task`,
/// `equal_budget_result_by_task`). Absent those fields the aggregate reason is
/// inherited unchanged — an unmeasured branch is fail-closed, never waived.
pub(super) const FAILURE_RATE_REASON: &str = "candidate_failure_rate_regressed";
pub(super) const EQUAL_BUDGET_REASON: &str = "candidate_did_not_win_equal_budget_proposal_gate";

/// Port of `test_exact_accept.load_accepted_by_task`:
/// `training/accepted-by-task.json` if present, else None. A missing file
/// means no per-task anchors exist yet (every new branch stays advisory);
/// malformed JSON is surfaced, never silently waived.
pub(super) fn load_accepted_by_task(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(read_json(path)?))
}

/// Port of `test_exact_accept.task_verdict_is_gating`: true when `task`'s
/// verdict participates in gating. The legacy branches always gate —
/// bit-identical to the two-branch gate. A NEW branch gates only once
/// `accepted-by-task.json` carries an accepted anchor for it; presence of the
/// task entry IS the anchor, the same rule
/// `jope_task_learning_controller.rs::accepted_pointers` applies (new tasks
/// are never bootstrapped into that file, so presence implies a promoted
/// per-task anchor).
pub(super) fn task_verdict_is_gating(task: &str, accepted_by_task: Option<&Value>) -> bool {
    if matches!(task, "regression" | "binary") {
        return true;
    }
    accepted_by_task
        .and_then(|value| value.get("tasks"))
        .and_then(|tasks| tasks.get(task))
        .is_some_and(Value::is_object)
}

/// Port of `test_exact_accept.conditional_advantage`: first *present*, finite,
/// numeric reading wins; a present-but-unreadable value is terminal. `None` is
/// never treated as zero — an unmeasured checkpoint is fail-closed.
pub(super) fn conditional_advantage(node: &Value) -> Option<f64> {
    for key in CONDITIONAL_ADVANTAGE_KEYS {
        if let Some(value) = node.get(key) {
            return value.as_f64().filter(|value| value.is_finite());
        }
    }
    None
}

pub(super) fn optional_number(value: Option<f64>) -> Value {
    value
        .and_then(serde_json::Number::from_f64)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// `node[field][task]` as a finite f64, or None. Used for the optional
/// task-scoped measurements; a malformed or non-finite reading reads as absent
/// so the per-task verdict falls back to the aggregate reason.
pub(super) fn task_number(node: &Value, field: &str, task: &str) -> Option<f64> {
    node.get(field)
        .and_then(|value| value.get(task))
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

/// Port of `test_exact_accept.task_verdicts`.
///
/// A branch verdict is the global verdict with (a) the OTHER branches' pairwise
/// regressions dropped, (b) the macro lift replaced by this branch's own lift,
/// and (c) any aggregate reason for which a task-scoped measurement exists
/// replaced by that measurement's own reason. Everything else — replay
/// verification, conditionality, proposal validity, latency — is a property of
/// the checkpoint, not of a branch, and is inherited verbatim.
///
/// A branch in `advisory_tasks` is computed identically but marked
/// `"advisory": true`: the caller guarantees such a branch contributed nothing
/// to the global verdict, and consumers must not gate on it.
#[allow(clippy::too_many_arguments)]
pub(super) fn task_verdicts(
    baseline: &Value,
    candidate: &Value,
    shared_reasons: &[String],
    task_own_reasons: &BTreeMap<&str, Vec<String>>,
    task_deltas: &BTreeMap<&str, f64>,
    shared_internal_extra: &[String],
    shared_milestone_extra: &[String],
    aggregate_equal_budget_lost: bool,
    advisory_tasks: &BTreeSet<&str>,
) -> Map<String, Value> {
    let mut verdicts = Map::new();
    for task in TASK_BRANCHES {
        let delta = task_deltas.get(task).copied().unwrap_or(f64::NAN);
        let own = task_own_reasons.get(task).cloned().unwrap_or_default();

        // Task-scoped failure rate, when measured, supersedes the aggregate.
        let mut superseded: BTreeSet<&str> = BTreeSet::new();
        let mut task_specific: Vec<String> = Vec::new();
        let mut failure_delta = Value::Null;
        if let (Some(candidate_failure), Some(baseline_failure)) = (
            task_number(candidate, "failure_rate_by_task", task),
            task_number(baseline, "failure_rate_by_task", task),
        ) {
            superseded.insert(FAILURE_RATE_REASON);
            let delta = candidate_failure - baseline_failure;
            failure_delta = optional_number(Some(delta));
            if delta > 0.0025 + 1e-12 {
                task_specific.push(format!("{task}_failure_rate_regressed"));
            }
        }
        // Task-scoped equal-budget head-to-head, when measured, supersedes the
        // aggregate. Only meaningful while the aggregate reason is present:
        // when the candidate already won overall there is nothing to supersede.
        let mut equal_budget = Value::Null;
        if let (Some(candidate_equal), Some(baseline_equal)) = (
            task_number(candidate, "equal_budget_result_by_task", task),
            task_number(baseline, "equal_budget_result_by_task", task),
        ) {
            superseded.insert(EQUAL_BUDGET_REASON);
            equal_budget = json!({"candidate": candidate_equal, "baseline": baseline_equal});
            if candidate_equal <= baseline_equal + 1e-12 {
                task_specific.push(format!("{task}_did_not_win_equal_budget_proposal_gate"));
            }
        } else if aggregate_equal_budget_lost {
            // no task-scoped reading: the aggregate reason stays in `shared`.
        }

        let inherited: Vec<String> = shared_reasons
            .iter()
            .filter(|reason| !superseded.contains(reason.as_str()))
            .cloned()
            .collect();

        let mut internal_reasons: Vec<String> = own.clone();
        internal_reasons.extend(task_specific.iter().cloned());
        internal_reasons.extend(inherited.iter().cloned());
        let mut milestone_reasons = internal_reasons.clone();
        if !matches!(
            delta.partial_cmp(&(TASK_INTERNAL_LIFT_MARGIN - 1e-12)),
            Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater)
        ) {
            internal_reasons.push(format!("{task}_pairwise_lift_below_point_two"));
        }
        if !matches!(
            delta.partial_cmp(&(TASK_MILESTONE_LIFT_MARGIN - 1e-12)),
            Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater)
        ) {
            milestone_reasons.push(format!("{task}_pairwise_lift_below_one_point"));
        }
        internal_reasons.extend(shared_internal_extra.iter().cloned());
        milestone_reasons.extend(shared_milestone_extra.iter().cloned());

        let internal_pass = internal_reasons.is_empty();
        let milestone_pass = milestone_reasons.is_empty();
        let tier = if milestone_pass {
            "milestone"
        } else if internal_pass {
            "internal"
        } else {
            "rejected"
        };
        // True when nothing this branch controls is blocking it: every residual
        // reason is a property of the checkpoint as a whole. That is the signal
        // the operator asked for — "binary was held back only by task-agnostic
        // evidence" is a different fact from "binary lost on its own merits".
        let branch_owned: BTreeSet<&String> = own
            .iter()
            .chain(task_specific.iter())
            .chain(
                internal_reasons
                    .iter()
                    .filter(|reason| reason.starts_with(&format!("{task}_"))),
            )
            .collect();
        verdicts.insert(
            task.to_owned(),
            json!({
                "advisory": advisory_tasks.contains(task),
                "pass": internal_pass,
                "internal_learning_pass": internal_pass,
                "milestone_pass": milestone_pass,
                "promotion_tier": tier,
                "reasons": if internal_pass { Vec::new() } else { internal_reasons.clone() },
                "milestone_reasons": milestone_reasons,
                "pairwise_lift": optional_number(
                    Some(delta).filter(|value| value.is_finite()),
                ),
                "candidate_pairwise_accuracy": candidate
                    .get(format!("{task}_pairwise_accuracy"))
                    .cloned()
                    .unwrap_or(Value::Null),
                "baseline_pairwise_accuracy": baseline
                    .get(format!("{task}_pairwise_accuracy"))
                    .cloned()
                    .unwrap_or(Value::Null),
                "failure_rate_delta": failure_delta,
                "equal_budget_result": equal_budget,
                "blocked_only_by_shared_evidence": !internal_pass && branch_owned.is_empty(),
                "blocking_reasons_owned_by_this_task": branch_owned
                    .iter()
                    .map(|reason| Value::String((*reason).clone()))
                    .collect::<Vec<_>>(),
            }),
        );
    }
    verdicts
}
