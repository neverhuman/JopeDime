use super::*;

pub(super) fn normalized_intent(path: &Path) -> Result<CanaryIntentV2> {
    let value = read_json(path)?;
    if value.get("schema").and_then(Value::as_str) == Some(CANARY_INTENT_SCHEMA) {
        let intent: CanaryIntentV2 = serde_json::from_value(value)?;
        intent.validate()?;
        return Ok(intent);
    }
    let schema = value
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or_default();
    ensure!(
        matches!(schema, "ShadowCanaryIntent.v1" | "TaskCanaryIntent.v1"),
        "unsupported canary intent schema"
    );
    let mode = if schema == "ShadowCanaryIntent.v1" {
        CanaryMode::Shadow
    } else {
        CanaryMode::Production
    };
    let task: Task = value
        .get("task")
        .and_then(Value::as_str)
        .context("legacy canary intent lacks task")?
        .parse()?;
    let candidate = value
        .get("shadow_checkpoint_sha256")
        .or_else(|| value.get("candidate_checkpoint_sha256"))
        .and_then(Value::as_str)
        .context("legacy canary intent lacks candidate checkpoint")?;
    let contexts = value
        .get("contexts")
        .and_then(Value::as_array)
        .context("legacy canary intent lacks contexts")?
        .iter()
        .map(|row| {
            let class = match row.get("kind").and_then(Value::as_str) {
                Some("frontier") => CanaryContextClass::Frontier,
                Some("hard") => CanaryContextClass::Hard,
                Some("anchor") => CanaryContextClass::Anchor,
                _ => bail!("legacy canary context class is invalid"),
            };
            Ok(CanaryContextV2 {
                context_identity: row
                    .get("dataset_hash")
                    .and_then(Value::as_str)
                    .context("legacy canary context lacks dataset")?
                    .to_owned(),
                class,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let intent = CanaryIntentV2 {
        schema: CANARY_INTENT_SCHEMA.to_owned(),
        mode,
        task,
        candidate_checkpoint_sha256: Sha256Digest::parse(candidate.to_owned())?,
        accepted_checkpoint_sha256: Sha256Digest::parse(
            value
                .get("accepted_parent_checkpoint_sha256")
                .and_then(Value::as_str)
                .context("legacy canary intent lacks accepted parent")?
                .to_owned(),
        )?,
        run_index: value
            .get("run_index")
            .or_else(|| value.get("independent_run"))
            .and_then(Value::as_u64)
            .map(|run| run.saturating_sub(u64::from(schema == "TaskCanaryIntent.v1")) as usize)
            .unwrap_or_default(),
        seed: value
            .get("seed")
            .and_then(Value::as_u64)
            .unwrap_or(20_260_726),
        contexts,
        candidate_proposals_per_context: 8,
        accepted_baseline_proposals_per_context: 8,
        zero_dl_gp_proposals_per_context: 8,
        created_at_ms: value
            .get("created_at_ms")
            .and_then(Value::as_u64)
            .map(u128::from),
    };
    intent.validate()?;
    Ok(intent)
}

pub(super) fn pending_intents(root: &Path) -> Result<Vec<PendingIntent>> {
    let mut output = Vec::new();
    for relative in [
        "training/shadow-canaries/pending",
        "training/task-canaries/pending",
    ] {
        let directory = root.join(relative);
        if !directory.is_dir() {
            continue;
        }
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let path = entry.path();
            let raw = read_json(&path)?;
            output.push(PendingIntent {
                value: normalized_intent(&path)?,
                created_at_ms: raw
                    .get("created_at_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or_default(),
                path,
                promotion: None,
            });
        }
    }
    let promotion_directory = root.join("training/task-promotion-evaluations/pending");
    if promotion_directory.is_dir() {
        let pool: SealedPanelPoolV2 =
            serde_json::from_value(read_json(&root.join("training/sealed-panel-pool.v2.json"))?)?;
        pool.validate()?;
        for entry in fs::read_dir(promotion_directory)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let promotion: TaskPromotionEvaluationIntentV3 =
                serde_json::from_value(read_json(&path)?)?;
            promotion.validate()?;
            let consumption: SealedPanelConsumptionV1 =
                serde_json::from_value(read_json(&promotion.panel_consumption)?)?;
            consumption.validate()?;
            ensure!(
                consumption.task == promotion.task
                    && consumption.candidate_checkpoint_sha256
                        == promotion.candidate_checkpoint_sha256
                    && consumption.run_index == promotion.run_index
                    && consumption.seed == promotion.seed
                    && consumption.context_sha256
                        == promotion
                            .context_schedule
                            .iter()
                            .map(|row| row.context_sha256.clone())
                            .collect::<Vec<_>>(),
                "promotion intent and sealed panel consumption drift"
            );
            let panel = pool.promotion_panel(promotion.task, &consumption.panel);
            let contexts_by_digest = panel
                .into_iter()
                .map(|row| (&row.context_sha256, row.context_identity.as_str()))
                .collect::<BTreeMap<_, _>>();
            let contexts = promotion
                .context_schedule
                .iter()
                .map(|row| {
                    Ok(CanaryContextV2 {
                        context_identity: contexts_by_digest
                            .get(&row.context_sha256)
                            .with_context(|| {
                                format!(
                                    "promotion context {} is absent from its sealed panel",
                                    row.context_sha256
                                )
                            })?
                            .to_string(),
                        class: CanaryContextClass::Frontier,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let created_at_ms = u64::try_from(promotion.created_at_ms)
                .context("promotion intent creation time exceeds u64")?;
            let value = CanaryIntentV2 {
                schema: CANARY_INTENT_SCHEMA.to_owned(),
                mode: CanaryMode::Production,
                task: promotion.task,
                candidate_checkpoint_sha256: promotion.candidate_checkpoint_sha256.clone(),
                accepted_checkpoint_sha256: promotion.accepted_parent_checkpoint_sha256.clone(),
                run_index: promotion.run_index,
                seed: promotion.seed,
                contexts,
                candidate_proposals_per_context: 8,
                accepted_baseline_proposals_per_context: 8,
                zero_dl_gp_proposals_per_context: 8,
                created_at_ms: Some(promotion.created_at_ms),
            };
            output.push(PendingIntent {
                path,
                value,
                created_at_ms,
                promotion: Some(promotion),
            });
        }
    }
    output.sort_by(|left, right| {
        left.promotion
            .is_none()
            .cmp(&right.promotion.is_none())
            .then(left.value.mode.cmp(&right.value.mode))
            .then(left.created_at_ms.cmp(&right.created_at_ms))
            .then(
                left.value
                    .candidate_checkpoint_sha256
                    .cmp(&right.value.candidate_checkpoint_sha256),
            )
            .then(left.value.run_index.cmp(&right.value.run_index))
            .then(left.path.cmp(&right.path))
    });
    Ok(output)
}

pub(super) fn predecessor_passed(root: &Path, pending: &PendingIntent) -> Result<bool> {
    let Some(promotion) = &pending.promotion else {
        return production_predecessor_passed(root, &pending.value);
    };
    if promotion.run_index == 0 {
        return Ok(true);
    }
    let directory = root.join("training/task-promotion-evaluations/accepted");
    if !directory.is_dir() {
        return Ok(false);
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(evaluation) = serde_json::from_value::<TaskPromotionEvaluationV3>(read_json(&path)?)
        else {
            continue;
        };
        if evaluation.accepted
            && evaluation.task == promotion.task
            && evaluation.accepted_parent_checkpoint_sha256
                == promotion.accepted_parent_checkpoint_sha256
            && evaluation.candidate_checkpoint_sha256 == promotion.candidate_checkpoint_sha256
            && evaluation.run_index + 1 == promotion.run_index
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn production_predecessor_passed(root: &Path, intent: &CanaryIntentV2) -> Result<bool> {
    if intent.mode != CanaryMode::Production || intent.run_index == 0 {
        return Ok(true);
    }
    let directory = root.join("training/task-canaries/accepted");
    if !directory.is_dir() {
        return Ok(false);
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let value = read_json(&path)?;
        if value.get("schema").and_then(Value::as_str) == Some(TASK_CANARY_ACCEPTANCE_V2_SCHEMA)
            && value.get("task").and_then(Value::as_str) == Some(intent.task.as_str())
            && value
                .get("candidate_checkpoint_sha256")
                .and_then(Value::as_str)
                == Some(intent.candidate_checkpoint_sha256.as_str())
            && value.get("run_index").and_then(Value::as_u64) == Some((intent.run_index - 1) as u64)
        {
            return Ok(value.get("accepted").and_then(Value::as_bool) == Some(true));
        }
    }
    Ok(false)
}

pub(super) fn checkpoint_paths(root: &Path, pending: &PendingIntent) -> Result<(PathBuf, PathBuf)> {
    let intent = &pending.value;
    let accepted = read_json(&root.join("training/accepted-by-task.json"))?;
    let accepted_row = accepted
        .get("tasks")
        .and_then(|value| value.get(intent.task.as_str()))
        .cloned()
        .or_else(|| {
            pending.promotion.as_ref()?;
            let path = root
                .join("training/bootstrap-evaluation-anchors")
                .join(format!("{}.v1.json", intent.task.as_str()));
            read_json(&path).ok().filter(|value| {
                value.get("schema").and_then(Value::as_str)
                    == Some("TaskBootstrapEvaluationAnchor.v1")
                    && value.get("accepted").and_then(Value::as_bool) == Some(false)
            })
        })
        .context("accepted task pointer or immutable bootstrap evaluation anchor is absent")?;
    ensure!(
        accepted_row
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            == Some(intent.accepted_checkpoint_sha256.as_str()),
        "canary accepted parent is no longer current"
    );
    let accepted_path = PathBuf::from(
        accepted_row
            .get("checkpoint")
            .and_then(Value::as_str)
            .context("accepted task checkpoint path is absent")?,
    );
    ensure!(
        sha256_file(&accepted_path)? == intent.accepted_checkpoint_sha256.as_str(),
        "accepted canary checkpoint content drift"
    );
    let candidate_path = match intent.mode {
        CanaryMode::Shadow => root
            .join("training/shadow-checkpoints")
            .join(intent.task.as_str())
            .join(format!(
                "{}.model.pt",
                intent.candidate_checkpoint_sha256.as_str()
            )),
        CanaryMode::Production => {
            let metadata = root
                .join("training/production-candidates")
                .join(intent.task.as_str())
                .join(format!(
                    "{}.candidate.v1.json",
                    intent.candidate_checkpoint_sha256.as_str()
                ));
            PathBuf::from(
                read_json(&metadata)?
                    .get("checkpoint")
                    .and_then(Value::as_str)
                    .context("production candidate path is absent")?,
            )
        }
    };
    ensure!(
        candidate_path.is_file()
            && sha256_file(&candidate_path)? == intent.candidate_checkpoint_sha256.as_str(),
        "candidate canary checkpoint content drift"
    );
    Ok((candidate_path, accepted_path))
}
