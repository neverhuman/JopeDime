use super::*;

pub(crate) fn apply_task_canary_acceptances(
    args: &Args,
    state: &mut ControllerState,
    accepted: &BTreeMap<Task, AcceptedPointer>,
) -> Result<Vec<Task>> {
    let evaluation_root = args.root.join("training/task-promotion-evaluations");
    if !evaluation_root.is_dir() {
        return Ok(Vec::new());
    }
    let mut receipts = Vec::new();
    for tier in ["accepted", "rejected"] {
        let directory = evaluation_root.join(tier);
        if !directory.is_dir() {
            continue;
        }
        receipts.extend(
            fs::read_dir(directory)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry.path().extension().and_then(|value| value.to_str()) == Some("json")
                })
                .filter_map(|entry| {
                    let value = read_json(&entry.path()).ok()?;
                    let evaluation: TaskPromotionEvaluationV3 =
                        serde_json::from_value(value).ok()?;
                    Some((evaluation.evaluated_at_ms, entry.path(), evaluation))
                }),
        );
    }
    receipts.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let mut promoted = Vec::new();
    for (_, path, receipt) in receipts {
        receipt.validate()?;
        let receipt_id = sha256_file(&path)?;
        if state.processed_canary_receipts.contains(&receipt_id) {
            continue;
        }
        let task = receipt.task;
        let Some(candidate) = state.production_candidates.get(&task).cloned() else {
            continue;
        };
        let parent = accepted
            .get(&task)
            .cloned()
            .or(bootstrap_evaluation_anchor(&args.root, task)?)
            .context("accepted task pointer or bootstrap evaluation anchor is absent")?;
        if receipt.accepted_parent_checkpoint_sha256 != parent.digest
            || candidate.accepted_parent_checkpoint_sha256 != parent.digest
            || receipt.candidate_checkpoint_sha256 != candidate.checkpoint_sha256
        {
            continue;
        }
        let accepted_verdict = receipt.accepted;
        state.processed_canary_receipts.insert(receipt_id.clone());
        state.canary_streaks.record(task, accepted_verdict);
        let independent_passes = if accepted_verdict {
            independent_external_passes(
                &args.root,
                task,
                &parent.digest,
                &candidate.checkpoint_sha256,
            )?
        } else {
            Vec::new()
        };
        let ready = independent_passes.len() == 3;
        append_event(
            &args.root,
            "task_external_promotion_evaluation_recorded",
            &json!({
                "receipt_id": receipt_id,
                "task": task,
                "accepted": accepted_verdict,
                "independent_single_use_panel_passes": independent_passes.len(),
            }),
        )?;
        if !accepted_verdict {
            state.production_candidates.remove(&task);
            append_event(
                &args.root,
                "task_production_candidate_closed_after_rejection",
                &json!({
                    "task": task,
                    "candidate_checkpoint_sha256": candidate.checkpoint_sha256,
                    "run_index": receipt.run_index,
                    "later_panels_consumed": false,
                }),
            )?;
        } else if ready && !promoted.contains(&task) {
            promote_task_pointer(args, task, &parent, &candidate, &independent_passes)?;
            state.canary_streaks.reset_after_promotion(task);
            state.production_candidates.remove(&task);
            state.shadow_models.remove(&task);
            state.last_shadow_corpus_watermark.remove(&task);
            state.last_recipe_e_corpus_watermark.remove(&task);
            state.last_recipe_e_descendant_receipts.remove(&task);
            promoted.push(task);
        } else if !ready {
            let consumption_path = args
                .root
                .join("training/sealed-panels/consumed")
                .join(format!(
                    "{}.v1.json",
                    receipt.panel_consumption_sha256.as_str()
                ));
            let consumption: SealedPanelConsumptionV1 =
                serde_json::from_value(read_json(&consumption_path)?)?;
            ensure!(
                consumption.run_index == receipt.run_index
                    && consumption.seed > receipt.run_index as u64,
                "accepted promotion run consumption seed drift"
            );
            let base_seed = consumption.seed - receipt.run_index as u64 - 1;
            schedule_promotion_run(
                args,
                task,
                &parent.digest,
                &candidate.checkpoint_sha256,
                base_seed,
                independent_passes.len(),
            )?;
        }
    }
    Ok(promoted)
}

pub(crate) fn independent_external_passes(
    root: &Path,
    task: Task,
    parent: &Sha256Digest,
    candidate: &Sha256Digest,
) -> Result<Vec<String>> {
    let directory = root.join("training/task-promotion-evaluations/accepted");
    let pool: SealedPanelPoolV2 =
        serde_json::from_value(read_json(&root.join("training/sealed-panel-pool.v2.json"))?)?;
    pool.validate()?;
    let mut rows = fs::read_dir(&directory)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let value = read_json(&entry.path()).ok()?;
            let evaluation: TaskPromotionEvaluationV3 = serde_json::from_value(value).ok()?;
            (evaluation.accepted
                && evaluation.task == task
                && &evaluation.accepted_parent_checkpoint_sha256 == parent
                && &evaluation.candidate_checkpoint_sha256 == candidate)
                .then_some((entry.path(), evaluation))
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|(_, evaluation)| evaluation.run_index);
    let mut prior = Vec::new();
    let mut receipt_ids = Vec::new();
    for (expected_run, (path, evaluation)) in rows.into_iter().take(3).enumerate() {
        evaluation.validate()?;
        ensure!(
            evaluation.run_index == expected_run,
            "external promotion runs must be exactly 0, 1, and 2"
        );
        let consumption_path = root.join("training/sealed-panels/consumed").join(format!(
            "{}.v1.json",
            evaluation.panel_consumption_sha256.as_str()
        ));
        ensure!(
            consumption_path.is_file()
                && sha256_file(&consumption_path)? == evaluation.panel_consumption_sha256.as_str(),
            "sealed panel consumption content drift"
        );
        let consumption: SealedPanelConsumptionV1 =
            serde_json::from_value(read_json(&consumption_path)?)?;
        ensure!(
            consumption.task == task
                && consumption.run_index == expected_run
                && &consumption.candidate_checkpoint_sha256 == candidate,
            "sealed panel consumption scope drift"
        );
        validate_panel_consumption(&pool, &consumption, &prior)?;
        prior.push(consumption);
        receipt_ids.push(sha256_file(&path)?);
    }
    ensure!(
        prior
            .iter()
            .map(|row| row.seed)
            .collect::<BTreeSet<_>>()
            .len()
            == prior.len(),
        "external promotion runs must use disjoint seeds"
    );
    Ok(receipt_ids)
}
