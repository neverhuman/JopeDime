use super::*;

pub(crate) fn promotion_observations(
    pending: &PendingIntent,
    work: &Path,
    proposal_set: &Path,
    receipts: &[PathBuf],
    facts: &BTreeMap<Sha256Digest, SealedPanelContextFactsV1>,
) -> Result<TaskPromotionObservationSetV3> {
    let promotion = pending
        .promotion
        .as_ref()
        .context("promotion observations requested for a canary")?;
    let proposals = read_json(proposal_set)?;
    let contexts = proposals
        .get("contexts")
        .and_then(Value::as_array)
        .context("promotion proposal set lacks contexts")?;
    ensure!(
        receipts.len() <= contexts.len() && receipts.len() <= promotion.maximum_contexts,
        "promotion receipt prefix exceeds its proposal schedule"
    );
    let mut observations = Vec::with_capacity(receipts.len());
    for (index, (context, receipt_path)) in contexts.iter().zip(receipts).enumerate() {
        let context_sha256 = &promotion.context_schedule[index].context_sha256;
        let context_facts = facts
            .get(context_sha256)
            .context("promotion context facts are absent")?;
        let manifest = read_json(&work.join(format!("manifest-{index:02}.json")))?;
        let requested = manifest
            .pointer("/canary/requested")
            .and_then(Value::as_object)
            .context("promotion manifest lacks requested lanes")?;
        let mut scores = BTreeMap::<String, (f64, u64)>::new();
        let mut replayed = BTreeSet::new();
        for line in fs::read_to_string(receipt_path)?.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let row: Value = serde_json::from_str(line)?;
            let semantic = row
                .get("semantic_sha256")
                .and_then(Value::as_str)
                .context("promotion receipt row lacks semantic identity")?
                .to_owned();
            replayed.insert(semantic.clone());
            if row.get("valid").and_then(Value::as_bool) != Some(true) {
                continue;
            }
            let score = bounded_task_score(
                pending.value.task,
                row.get("score")
                    .and_then(Value::as_f64)
                    .context("valid promotion receipt lacks score")?,
            );
            let runtime_ms = (row
                .get("runtime_seconds")
                .and_then(Value::as_f64)
                .context("valid promotion receipt lacks runtime")?
                .max(0.0)
                * 1_000.0)
                .round()
                .min(u64::MAX as f64) as u64;
            scores.insert(semantic, (score, runtime_ms));
        }
        let lane = |name: &str| -> Result<(f64, usize, u64, BTreeSet<String>)> {
            let semantics = requested
                .get(name)
                .and_then(Value::as_array)
                .with_context(|| format!("promotion {name} lane is absent"))?;
            ensure!(
                semantics.len() <= 8,
                "promotion lane exceeds its eight-slot budget"
            );
            let identities = semantics
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .with_context(|| format!("promotion {name} lane identity is invalid"))
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                identities.iter().collect::<BTreeSet<_>>().len() == identities.len(),
                "promotion lane repeats a semantic identity"
            );
            let mut values = Vec::with_capacity(8);
            let mut runtimes = Vec::new();
            let mut valid = 0_usize;
            for semantic in &identities {
                if let Some((score, runtime)) = scores.get(semantic) {
                    values.push(*score);
                    runtimes.push(*runtime);
                    valid += 1;
                } else {
                    values.push(-1.0);
                }
            }
            values.resize(8, -1.0);
            Ok((
                mean(&values),
                valid,
                if runtimes.is_empty() {
                    0
                } else {
                    runtimes.iter().sum::<u64>() / runtimes.len() as u64
                },
                identities.into_iter().collect(),
            ))
        };
        let (candidate_score, candidate_valid, candidate_latency_ms, candidate_requested) =
            lane("candidate")?;
        let (accepted_score, accepted_valid, accepted_latency_ms, accepted_requested) =
            lane("accepted")?;
        let (gp_score, gp_valid, _, gp_requested) = lane("gp")?;
        let mut requested_union = candidate_requested;
        requested_union.extend(accepted_requested);
        requested_union.extend(gp_requested);
        let replay_verified = receipt_path.metadata()?.len() > 0
            && requested_union
                .iter()
                .all(|semantic| replayed.contains(semantic));
        let conditional = context
            .get("conditional_advantage")
            .and_then(Value::as_object)
            .context("promotion proposal set lacks common-union conditional rescoring")?;
        let candidate_conditional_advantage = conditional
            .get("candidate_checkpoint")
            .and_then(Value::as_f64)
            .context("promotion proposal set lacks candidate conditional advantage")?;
        let accepted_conditional_advantage = conditional
            .get("accepted_checkpoint")
            .and_then(Value::as_f64)
            .context("promotion proposal set lacks accepted conditional advantage")?;
        ensure!(
            candidate_conditional_advantage.is_finite()
                && accepted_conditional_advantage.is_finite(),
            "promotion conditional advantage is non-finite"
        );
        let candidate_wins = candidate_score > accepted_score;
        observations.push(PromotionObservationV3 {
            context_sha256: context_sha256.clone(),
            source_family_sha256: context_facts.source_family_sha256.clone(),
            candidate_score,
            accepted_score,
            gp_score,
            candidate_conditional_advantage,
            accepted_conditional_advantage,
            valid: candidate_valid == 8,
            replay_verified,
            candidate_failed: candidate_valid < 8,
            accepted_failed: accepted_valid < 8,
            candidate_latency_ms,
            accepted_latency_ms,
            strict_win: candidate_wins,
            equal_budget_win: candidate_wins && gp_valid == 8,
            slices: context_facts.slices(),
        });
    }
    let observation_set = TaskPromotionObservationSetV3 {
        schema: TASK_PROMOTION_OBSERVATION_SET_V3_SCHEMA.to_owned(),
        panel_consumption_sha256: promotion.panel_consumption_sha256.clone(),
        observations,
    };
    observation_set.validate()?;
    Ok(observation_set)
}

pub(crate) fn promotion_prefix(
    work: &Path,
    promotion: &TaskPromotionEvaluationIntentV3,
) -> Result<(usize, Option<TaskPromotionEvaluationV3>)> {
    let mut prefix = promotion.initial_contexts;
    loop {
        let path = work.join(format!("prefix-{prefix}.evaluation.v3.json"));
        if !path.is_file() {
            return Ok((prefix, None));
        }
        let evaluation: TaskPromotionEvaluationV3 = serde_json::from_value(read_json(&path)?)?;
        evaluation.validate()?;
        let irreversible_rejection = evaluation.metrics.replay_verification < 1.0;
        if evaluation.accepted || irreversible_rejection || prefix == promotion.maximum_contexts {
            return Ok((prefix, Some(evaluation)));
        }
        prefix = prefix.saturating_add(promotion.context_step);
        ensure!(
            prefix <= promotion.maximum_contexts,
            "promotion prefix advanced beyond its maximum"
        );
    }
}

pub(crate) fn finalize_promotion_intent(
    args: &Args,
    pending: &PendingIntent,
    evaluation: &TaskPromotionEvaluationV3,
) -> Result<Value> {
    let tier = if evaluation.accepted {
        "accepted"
    } else {
        "rejected"
    };
    let directory = args
        .root
        .join("training/task-promotion-evaluations")
        .join(tier);
    let evaluation_path = directory.join(format!(
        "{}-{}-run{}-n{}.v3.json",
        evaluation.candidate_checkpoint_sha256,
        evaluation.task.as_str(),
        evaluation.run_index,
        evaluation.metrics.contexts,
    ));
    write_immutable(&evaluation_path, &serde_json::to_vec(evaluation)?)?;
    let intent_destination = args
        .root
        .join("training/task-promotion-evaluations/intents")
        .join(
            pending
                .path
                .file_name()
                .context("promotion intent has no filename")?,
        );
    fs::create_dir_all(
        intent_destination
            .parent()
            .context("promotion intent destination has no parent")?,
    )?;
    fs::rename(&pending.path, &intent_destination)?;
    Ok(json!({
        "action": if evaluation.accepted { "promotion_run_accepted" } else { "promotion_run_rejected" },
        "evaluation": evaluation_path,
        "consumed_intent": intent_destination,
        "run_index": evaluation.run_index,
        "contexts": evaluation.metrics.contexts,
    }))
}

pub(crate) fn consume_promotion_intent(args: &Args, pending: &PendingIntent) -> Result<Value> {
    let promotion = pending
        .promotion
        .as_ref()
        .context("promotion consumer received a canary")?;
    if !predecessor_passed(&args.root, pending)? {
        return Ok(json!({
            "action": "waiting_for_prior_promotion_run",
            "task": promotion.task,
            "candidate": promotion.candidate_checkpoint_sha256,
            "run_index": promotion.run_index,
        }));
    }
    let intent_hash = sha256_file(&pending.path)?;
    let work = args
        .root
        .join("training/task-promotion-evaluations/work")
        .join(&intent_hash);
    fs::create_dir_all(&work)?;
    let (prefix, completed) = promotion_prefix(&work, promotion)?;
    if let Some(evaluation) = completed {
        return finalize_promotion_intent(args, pending, &evaluation);
    }
    let jobs = scq_jobs(args)?;
    let Some(proposal_set) = ensure_proposals(args, pending, &work, &jobs)? else {
        return Ok(json!({
            "action": "promotion_proposal_job_pending",
            "intent_sha256": intent_hash,
        }));
    };
    let facts = promotion_context_facts(args, pending, prefix)?;
    let evaluations = build_evaluations(args, pending, &work, &proposal_set, prefix)?;
    let jobs = scq_jobs(args)?;
    let Some(receipts) = ensure_evaluations(args, &work, &evaluations, &jobs)? else {
        return Ok(json!({
            "action": "promotion_evaluation_jobs_pending",
            "intent_sha256": intent_hash,
            "evaluation_jobs": evaluations.len(),
            "context_prefix": prefix,
        }));
    };
    let observation_set = promotion_observations(pending, &work, &proposal_set, &receipts, &facts)?;
    let observation_path = work.join(format!("prefix-{prefix}.observations.v3.json"));
    write_immutable(&observation_path, &serde_json::to_vec(&observation_set)?)?;
    let evaluation = evaluate_task_promotion_v3(
        promotion.task,
        promotion.accepted_parent_checkpoint_sha256.clone(),
        promotion.candidate_checkpoint_sha256.clone(),
        promotion.panel_consumption_sha256.clone(),
        promotion.run_index,
        &observation_set.observations,
        promotion.seed,
        now_ms(),
    )?;
    let evaluation_path = work.join(format!("prefix-{prefix}.evaluation.v3.json"));
    write_immutable(&evaluation_path, &serde_json::to_vec(&evaluation)?)?;
    if evaluation.accepted
        || evaluation.metrics.replay_verification < 1.0
        || prefix == promotion.maximum_contexts
    {
        return finalize_promotion_intent(args, pending, &evaluation);
    }
    Ok(json!({
        "action": "promotion_prefix_inconclusive",
        "intent_sha256": intent_hash,
        "context_prefix": prefix,
        "next_context_prefix": prefix + PROMOTION_CONTEXT_STEP,
        "observation_set": observation_path,
        "evaluation": evaluation_path,
    }))
}
