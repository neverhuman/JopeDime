use super::*;

pub(super) fn lane_metrics(
    requested: &[Value],
    scores: &BTreeMap<String, (f64, f64)>,
) -> LaneMetrics {
    let mut output = LaneMetrics {
        requested: requested.len(),
        ..LaneMetrics::default()
    };
    let mut seen = BTreeSet::new();
    for semantic in requested.iter().filter_map(Value::as_str) {
        if !seen.insert(semantic) {
            continue;
        }
        if let Some((score, runtime)) = scores.get(semantic) {
            output.valid += 1;
            output.scores.push(*score);
            output.runtimes.push(*runtime);
        }
    }
    output
}

pub(super) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Registered shadow-screen lower bound over independent context clusters.
/// Candidate rows within one context are correlated, so the uncertainty unit
/// is the context mean rather than the individual proposal comparison.
pub(super) fn registered_context_lower_bound(values: &[f64]) -> f64 {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return -2.0;
    }
    let center = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (center - 1.96 * (variance / values.len() as f64).sqrt()).clamp(-2.0, 2.0)
}

pub(super) fn dataset_slices(
    train: &Path,
    task: Task,
) -> Result<(usize, usize, String, String, String, String, String)> {
    let mut rows = 0_usize;
    let mut columns = None;
    let mut missing = 0_usize;
    let mut cells = 0_usize;
    let mut target_counts = BTreeMap::<String, usize>::new();
    for line in BufReader::new(fs::File::open(train)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split(',').collect::<Vec<_>>();
        ensure!(
            fields.len() >= 2,
            "sealed panel CSV has fewer than two columns"
        );
        match columns {
            Some(expected) => ensure!(fields.len() == expected, "sealed panel CSV width drift"),
            None => columns = Some(fields.len()),
        }
        rows += 1;
        cells += fields.len();
        missing += fields
            .iter()
            .filter(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "na" | "nan" | "null" | "none"
                )
            })
            .count();
        if matches!(task, Task::Binary | Task::Multiclass) {
            *target_counts
                .entry(fields.last().unwrap_or(&"").trim().to_owned())
                .or_default() += 1;
        }
    }
    let columns = columns.context("sealed panel CSV is empty")?;
    ensure!(rows > 0 && cells > 0, "sealed panel CSV is empty");
    let missing_fraction = missing as f64 / cells as f64;
    let size_slice = if rows < 2_000 {
        "small"
    } else if rows <= 20_000 {
        "medium"
    } else {
        "large"
    };
    let missingness_slice = if missing == 0 {
        "none"
    } else if missing_fraction <= 0.05 {
        "low"
    } else {
        "high"
    };
    let (imbalance_slice, target_shape_slice) =
        if matches!(task, Task::Binary | Task::Multiclass) && target_counts.len() >= 2 {
            let minimum = target_counts.values().copied().min().unwrap_or_default();
            let maximum = target_counts.values().copied().max().unwrap_or(1);
            let shape = if minimum as f64 / (maximum as f64) < 0.2 {
                "imbalanced"
            } else {
                "balanced"
            };
            (shape, shape)
        } else {
            ("not_applicable", "unknown")
        };
    Ok((
        rows,
        columns,
        format!("{missing_fraction:.12}"),
        size_slice.to_owned(),
        missingness_slice.to_owned(),
        imbalance_slice.to_owned(),
        target_shape_slice.to_owned(),
    ))
}

pub(super) fn promotion_context_facts(
    args: &Args,
    pending: &PendingIntent,
    context_limit: usize,
) -> Result<BTreeMap<Sha256Digest, SealedPanelContextFactsV1>> {
    let promotion = pending
        .promotion
        .as_ref()
        .context("promotion facts requested for a canary intent")?;
    ensure!(
        context_limit <= promotion.maximum_contexts,
        "promotion context prefix exceeds its schedule"
    );
    let pool: SealedPanelPoolV2 = serde_json::from_value(read_json(
        &args.root.join("training/sealed-panel-pool.v2.json"),
    )?)?;
    pool.validate()?;
    let consumption: SealedPanelConsumptionV1 =
        serde_json::from_value(read_json(&promotion.panel_consumption)?)?;
    let pool_by_context = pool
        .promotion_panel(promotion.task, &consumption.panel)
        .into_iter()
        .map(|row| (&row.context_sha256, row))
        .collect::<BTreeMap<_, _>>();
    let (pointer, catalog) = catalog_entries(&args.root)?;
    let catalog_sha256 = Sha256Digest::parse(
        pointer
            .get("catalog_sha256")
            .and_then(Value::as_str)
            .context("catalog pointer lacks catalog digest")?
            .to_owned(),
    )?;
    let facts_root = args.root.join("training/sealed-panel-context-facts");
    let mut output = BTreeMap::new();
    for scheduled in promotion.context_schedule.iter().take(context_limit) {
        let sealed = pool_by_context
            .get(&scheduled.context_sha256)
            .context("scheduled promotion context is absent from the sealed panel")?;
        let path = facts_root.join(format!("{}.v1.json", scheduled.context_sha256));
        if path.is_file() {
            let facts: SealedPanelContextFactsV1 = serde_json::from_value(read_json(&path)?)?;
            facts.validate()?;
            ensure!(
                facts.task == promotion.task
                    && facts.context_sha256 == scheduled.context_sha256
                    && facts.catalog_sha256 == catalog_sha256
                    && facts.context_identity == sealed.context_identity,
                "sealed panel context facts scope drift"
            );
            output.insert(scheduled.context_sha256.clone(), facts);
            continue;
        }
        let entry = catalog
            .get(&(promotion.task, sealed.context_identity.clone()))
            .context("sealed context is absent from the current pinned catalog")?;
        ensure!(
            entry.get("context_sha256").and_then(Value::as_str)
                == Some(scheduled.context_sha256.as_str()),
            "sealed context vector changed in the current catalog"
        );
        let train = PathBuf::from(
            entry
                .get("train_csv")
                .and_then(Value::as_str)
                .context("sealed catalog entry lacks train CSV")?,
        );
        let test = PathBuf::from(
            entry
                .get("test_csv")
                .and_then(Value::as_str)
                .context("sealed catalog entry lacks test CSV")?,
        );
        let train_sha256 = Sha256Digest::parse(sha256_file(&train)?)?;
        let test_sha256 = Sha256Digest::parse(sha256_file(&test)?)?;
        ensure!(
            entry.get("train_sha256").and_then(Value::as_str) == Some(train_sha256.as_str())
                && entry.get("test_sha256").and_then(Value::as_str) == Some(test_sha256.as_str()),
            "sealed dataset content drift"
        );
        let (
            rows,
            columns,
            missing_fraction,
            size_slice,
            missingness_slice,
            imbalance_slice,
            target_shape_slice,
        ) = dataset_slices(&train, promotion.task)?;
        let facts = SealedPanelContextFactsV1 {
            schema: SEALED_PANEL_CONTEXT_FACTS_V1_SCHEMA.to_owned(),
            task: promotion.task,
            context_identity: sealed.context_identity.clone(),
            context_sha256: scheduled.context_sha256.clone(),
            catalog_sha256: catalog_sha256.clone(),
            train_sha256,
            test_sha256,
            source_family_sha256: sealed.source_family_sha256.clone(),
            near_duplicate_cluster_sha256: sealed.near_duplicate_cluster_sha256.clone(),
            origin: sealed.origin,
            rows,
            columns,
            missing_fraction,
            size_slice,
            missingness_slice,
            imbalance_slice,
            target_shape_slice,
            created_at_ms: now_ms(),
        };
        facts.validate()?;
        write_immutable(&path, &serde_json::to_vec(&facts)?)?;
        output.insert(scheduled.context_sha256.clone(), facts);
    }
    Ok(output)
}

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

pub(super) fn promotion_observations(
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

pub(super) fn promotion_prefix(
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

pub(super) fn finalize_promotion_intent(
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

pub(super) fn consume_promotion_intent(args: &Args, pending: &PendingIntent) -> Result<Value> {
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
