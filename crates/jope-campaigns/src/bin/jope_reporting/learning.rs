use super::*;

pub(super) fn task_name(task: Task) -> &'static str {
    task.as_str()
}

pub(super) fn promotion_cas_valid(
    root: &Path,
    task: Task,
    checkpoint: &str,
) -> Result<Option<PathBuf>> {
    let path = root
        .join("training/task-promotions")
        .join(format!("{}-{checkpoint}.v1.json", task_name(task)));
    if !path.is_file() {
        return Ok(None);
    }
    let receipt = read_json(&path)?;
    let valid = receipt.get("schema").and_then(Value::as_str) == Some("TaskCheckpointPromotion.v2")
        && receipt.get("task").and_then(Value::as_str) == Some(task_name(task))
        && receipt
            .get("child_checkpoint_sha256")
            .and_then(Value::as_str)
            == Some(checkpoint)
        && receipt
            .get("independent_single_use_panel_acceptances")
            .and_then(Value::as_u64)
            == Some(3)
        && receipt
            .get("external_pass_receipt_ids")
            .and_then(Value::as_array)
            .is_some_and(|rows| rows.len() == 3);
    ensure!(valid, "task promotion CAS receipt contract drift");
    let pointer = read_json(&root.join("training/accepted-by-task.json"))?;
    ensure!(
        pointer
            .get("tasks")
            .and_then(|tasks| tasks.get(task_name(task)))
            .and_then(|row| row.get("checkpoint_sha256"))
            .and_then(Value::as_str)
            == Some(checkpoint),
        "promotion CAS receipt is not reflected by the accepted task pointer"
    );
    Ok(Some(path))
}

pub(super) fn observation_sets(
    root: &Path,
    panels: &BTreeSet<String>,
) -> Result<BTreeMap<String, (TaskPromotionObservationSetV3, PathBuf)>> {
    let mut paths = Vec::new();
    collect_named(
        &root.join("training/task-promotion-evaluations/work"),
        ".observations.v3.json",
        &mut paths,
    );
    paths.sort();
    let mut output = BTreeMap::new();
    for path in paths {
        let value: TaskPromotionObservationSetV3 = serde_json::from_value(read_json(&path)?)?;
        value.validate()?;
        let panel = value.panel_consumption_sha256.as_str().to_owned();
        if !panels.contains(&panel) {
            continue;
        }
        let replace = output.get(&panel).is_none_or(
            |(prior, _): &(TaskPromotionObservationSetV3, PathBuf)| {
                value.observations.len() > prior.observations.len()
            },
        );
        if replace {
            output.insert(panel, (value, path));
        }
    }
    Ok(output)
}

pub(super) fn immutable_learning_report(
    root: &Path,
    task: Task,
    checkpoint: &str,
    evaluations: Vec<(TaskPromotionEvaluationV3, PathBuf)>,
) -> Result<Option<PathBuf>> {
    let mut by_run = BTreeMap::new();
    for (evaluation, path) in evaluations {
        evaluation.validate()?;
        ensure!(
            evaluation.task == task
                && evaluation.candidate_checkpoint_sha256.as_str() == checkpoint,
            "terminal evaluation scope drift"
        );
        if let Some((prior, _)) = by_run.insert(evaluation.run_index, (evaluation.clone(), path)) {
            ensure!(
                prior == evaluation,
                "conflicting terminal evaluation for one run"
            );
        }
    }
    let rejected = by_run.values().any(|(evaluation, _)| !evaluation.accepted);
    let cas = promotion_cas_valid(root, task, checkpoint)?;
    let three_accepted = by_run.len() == 3
        && (0..3).all(|run| by_run.get(&run).is_some_and(|(row, _)| row.accepted));
    if !rejected && !(three_accepted && cas.is_some()) {
        return Ok(None);
    }
    let panels = by_run
        .values()
        .map(|(row, _)| row.panel_consumption_sha256.as_str().to_owned())
        .collect::<BTreeSet<_>>();
    let observations = observation_sets(root, &panels)?;
    let mut context_panel = BTreeMap::new();
    let mut family_panel = BTreeMap::new();
    let mut complete_runs = Vec::new();
    let mut source_paths = by_run
        .values()
        .map(|(_, path)| path.clone())
        .collect::<Vec<_>>();
    let mut total_contexts = 0_usize;
    let mut valid = 0_usize;
    let mut replay = 0_usize;
    let mut failures = 0_usize;
    let mut latencies = Vec::new();
    let mut strict_wins = 0_usize;
    let mut family_lifts = BTreeMap::<String, Vec<f64>>::new();
    let mut family_disjoint = panels.len() == by_run.len();
    for (run, (evaluation, _)) in &by_run {
        let panel = evaluation.panel_consumption_sha256.as_str();
        let Some((set, path)) = observations.get(panel) else {
            continue;
        };
        if set.observations.len() != evaluation.metrics.contexts
            || evaluation.metrics.replay_verification != 1.0
            || !set.observations.iter().all(|row| row.replay_verified)
        {
            continue;
        }
        source_paths.push(path.clone());
        complete_runs.push(*run);
        for row in &set.observations {
            family_disjoint &= context_panel
                .insert(row.context_sha256.as_str().to_owned(), panel.to_owned())
                .is_none();
            if let Some(prior) = family_panel.insert(
                row.source_family_sha256.as_str().to_owned(),
                panel.to_owned(),
            ) {
                family_disjoint &= prior == panel;
            }
            total_contexts += 1;
            valid += usize::from(row.valid);
            replay += usize::from(row.replay_verified);
            failures += usize::from(row.candidate_failed);
            latencies.push(row.candidate_latency_ms);
            strict_wins += usize::from(row.strict_win);
            family_lifts
                .entry(row.source_family_sha256.as_str().to_owned())
                .or_default()
                .push(row.candidate_score - row.accepted_score);
        }
    }
    complete_runs.sort_unstable();
    let complete_family_disjoint = family_disjoint && !complete_runs.is_empty();
    let hard_regression = by_run.values().any(|(evaluation, _)| {
        complete_runs.contains(&evaluation.run_index)
            && (evaluation.metrics.confidence.candidate_vs_accepted_lower < 0.0
                || evaluation.metrics.maximum_slice_regression > 0.005
                || evaluation.metrics.validity < 0.99
                || evaluation.metrics.failure_rate_increase > 0.0025
                || evaluation.metrics.latency_ratio > 1.25)
    });
    let cas_succeeded = cas.is_some();
    let improved =
        three_accepted && cas_succeeded && complete_runs == vec![0, 1, 2] && family_disjoint;
    let disposition = if improved {
        "improved"
    } else if complete_family_disjoint && hard_regression {
        "regressed"
    } else {
        "inconclusive"
    };
    if let Some(path) = cas {
        source_paths.push(path);
        source_paths.push(root.join("training/accepted-by-task.json"));
    }
    source_paths.sort();
    source_paths.dedup();
    latencies.sort_unstable();
    let p95_latency_ms = (!latencies.is_empty())
        .then(|| latencies[(latencies.len() * 95).div_ceil(100).saturating_sub(1)]);
    let family_means = family_lifts
        .values()
        .filter(|rows| !rows.is_empty())
        .map(|rows| rows.iter().sum::<f64>() / rows.len() as f64)
        .collect::<Vec<_>>();
    let sealed_panel_lift = (!family_means.is_empty())
        .then(|| family_means.iter().sum::<f64>() / family_means.len() as f64);
    let generated_at_ms = by_run
        .values()
        .map(|(row, _)| row.evaluated_at_ms)
        .max()
        .unwrap_or(0);
    let source_artifacts = source_paths
        .iter()
        .map(|path| {
            Ok(json!({
                "path": path,
                "sha256": sha256_file(path)?,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let report = json!({
        "schema": "JopeLearningReport.v2",
        "publisher": "jope_reporting.v1",
        "immutable": true,
        "checkpoint_sha256": checkpoint,
        "task": task,
        "generated_at_ms": generated_at_ms,
        "disposition": disposition,
        "promotion_disposition": {
            "evaluation_runs": by_run.iter().map(|(run, (row, _))| json!([run, row.accepted])).collect::<Vec<_>>(),
            "rejection_reasons": by_run.values().flat_map(|(row, _)| row.rejection_reasons.clone()).collect::<BTreeSet<_>>(),
            "pointer_compare_and_swap_succeeded": cas_succeeded,
        },
        "metrics": {
            "proposal_validity": (total_contexts > 0).then(|| valid as f64 / total_contexts as f64),
            "matched_strict_win_rate": (total_contexts > 0).then(|| strict_wins as f64 / total_contexts as f64),
            "p95_latency_ms": p95_latency_ms,
            "failure_rate": (total_contexts > 0).then(|| failures as f64 / total_contexts as f64),
            "replay_coverage": (total_contexts > 0).then(|| replay as f64 / total_contexts as f64),
            "sealed_panel_lift": sealed_panel_lift,
            "panel_coverage": panels.len(),
            "run_coverage": by_run.keys().copied().collect::<Vec<_>>(),
            "observation_run_coverage": complete_runs,
            "context_count": total_contexts,
            "source_family_count": family_panel.len(),
            "family_disjoint": family_disjoint,
            "gp_relative_lift": Value::Null,
            "causal_metrics_na_reason": "no complete authoritative causal-pair projection supplied to the reporter",
        },
        "source_artifacts": source_artifacts,
        "claims_policy": {
            "corpus_growth_is_not_improvement": true,
            "completed_training_is_not_improvement": true,
            "test_adaptive_evidence_is_benchmark_invalid": true,
            "family_disjoint_replay_is_required": true,
            "caller_supplied_metric_json_is_forbidden": true,
        },
    });
    let output = root
        .join("ops/learning-reports")
        .join(task_name(task))
        .join(format!("{checkpoint}.v2.json"));
    if output.is_file() {
        let existing = read_json(&output)?;
        if existing.get("publisher").and_then(Value::as_str) == Some("jope_reporting.v1") {
            ensure!(
                existing == report,
                "immutable learning report content collision"
            );
        } else {
            ensure!(
                existing.get("schema").and_then(Value::as_str) == Some("JopeLearningReport.v2")
                    && existing.get("task").and_then(Value::as_str) == Some(task_name(task))
                    && existing.get("checkpoint_sha256").and_then(Value::as_str)
                        == Some(checkpoint),
                "pre-existing immutable learning report scope collision"
            );
        }
    } else {
        atomic_json(&output, &report)?;
    }
    Ok(Some(output))
}

pub(super) fn publish_terminal_learning_reports(root: &Path) -> Result<usize> {
    let mut grouped = BTreeMap::<(Task, String), Vec<(TaskPromotionEvaluationV3, PathBuf)>>::new();
    for tier in ["accepted", "rejected"] {
        for path in json_files(&root.join("training/task-promotion-evaluations").join(tier)) {
            let evaluation: TaskPromotionEvaluationV3 = serde_json::from_value(read_json(&path)?)?;
            evaluation.validate()?;
            grouped
                .entry((
                    evaluation.task,
                    evaluation.candidate_checkpoint_sha256.as_str().to_owned(),
                ))
                .or_default()
                .push((evaluation, path));
        }
    }
    let mut published = 0;
    for ((task, checkpoint), evaluations) in grouped {
        let output = root
            .join("ops/learning-reports")
            .join(task_name(task))
            .join(format!("{checkpoint}.v2.json"));
        let existed = output.is_file();
        if immutable_learning_report(root, task, &checkpoint, evaluations)?.is_some() && !existed {
            published += 1;
        }
    }
    Ok(published)
}

pub(super) fn terminal_learning_inventory(root: &Path) -> TerminalLearningInventory {
    let mut inventory = TerminalLearningInventory::default();
    let mut terminal = Vec::<(String, String)>::new();
    for tier in ["accepted", "rejected"] {
        for path in json_files(&root.join("training/task-promotion-evaluations").join(tier)) {
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
                continue;
            };
            if value.get("schema").and_then(Value::as_str) != Some("TaskPromotionEvaluation.v3") {
                continue;
            }
            let Some(task) = value.get("task").and_then(Value::as_str) else {
                continue;
            };
            let Some(checkpoint) = value
                .get("candidate_checkpoint_sha256")
                .and_then(Value::as_str)
            else {
                continue;
            };
            inventory.terminal_evaluations += 1;
            terminal.push((task.to_owned(), checkpoint.to_owned()));
        }
    }
    inventory.promotion_cas_receipts = json_files(&root.join("training/task-promotions")).len();
    for task in ["regression", "binary", "multiclass", "timeseries"] {
        inventory.immutable_learning_reports +=
            json_files(&root.join("ops/learning-reports").join(task)).len();
    }
    terminal.sort();
    terminal.dedup();
    inventory.unpublished_terminal_candidates = terminal
        .into_iter()
        .filter_map(|(task, checkpoint)| {
            (!root
                .join("ops/learning-reports")
                .join(&task)
                .join(format!("{checkpoint}.v2.json"))
                .is_file())
            .then(|| format!("{task}:{checkpoint}"))
        })
        .collect();
    inventory
}
